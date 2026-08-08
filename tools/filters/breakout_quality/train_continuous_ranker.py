"""Research-only continuous rankers using the active InceptionTime model."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.breakout_quality import (
    STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE,
    STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE,
    STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE,
    STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE,
    STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_LISTWISE_PROFILE,
    CONTINUOUS_RANKER_TRAINING_OBJECTIVES,
    TRAINING_LABEL_SCOPE_ALL,
    TRAINING_LABEL_SCOPE_PASS_ONLY,
    TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION,
    TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_LISTWISE_RANKING,
    get_breakout_quality_experiment_profile,
    resolve_breakout_quality_random_seed,
)
from config.breakout_quality import (
    BREAKOUT_QUALITY_ALLOW_TF32,
    BREAKOUT_QUALITY_DEFAULT_BATCH_SIZE,
    BREAKOUT_QUALITY_DEFAULT_EPOCHS,
    BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    BREAKOUT_QUALITY_DEFAULT_GRADIENT_CLIP_NORM,
    BREAKOUT_QUALITY_DEFAULT_LEARNING_RATE,
    BREAKOUT_QUALITY_DEFAULT_WEIGHT_DECAY,
    BREAKOUT_QUALITY_DETERMINISTIC_ALGORITHMS,
    BREAKOUT_QUALITY_EARLY_STOPPING_MIN_DELTA,
    BREAKOUT_QUALITY_EARLY_STOPPING_PATIENCE,
    BREAKOUT_QUALITY_EVALUATION_BATCH_SIZE,
    BREAKOUT_QUALITY_CONTINUOUS_RANKER_REPORT_TOP_K,
    BREAKOUT_QUALITY_CONTINUOUS_RANKER_REPORT_BOUNDARY_WIDTH,
    BREAKOUT_QUALITY_INNER_VALIDATION_MONTHS,
    BREAKOUT_QUALITY_MIXED_PRECISION_DTYPE,
    BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
    BREAKOUT_QUALITY_PRELOAD_FEATURE_BANK,
    BREAKOUT_QUALITY_TORCH_DEVICE,
    BREAKOUT_QUALITY_USE_INNER_VALIDATION,
    BREAKOUT_QUALITY_USE_MIXED_PRECISION,
)
from filters.breakout_quality.artifacts import build_file_manifest
from filters.breakout_quality.continuous_ranker_quality import daily_top_k_metrics
from filters.breakout_quality.continuous_target import (
    STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
    STRATEGY_ALIGNED_TARGET_ID,
    TARGET_TRADE_MATCHES_CSV_FILENAME,
    load_validated_continuous_target_arrays,
    resolve_continuous_target_dir,
)
from filters.breakout_quality.contract import (
    ARTIFACT_CONTRACT_VERSION,
    CONTEXT_COLUMNS,
    DEFAULT_LABEL_POLICY,
    DEFAULT_MANIFEST_FILENAME,
    DEFAULT_MODEL_FILENAME,
    DEFAULT_SPLIT_FILENAME,
    FEATURE_COLUMNS,
    FILTER_FAMILY,
    LABEL_PASS,
    LABEL_REJECT,
)
from filters.breakout_quality.inference import strict_parallel_batched_logits
from filters.breakout_quality.models.factory import (
    build_model,
    count_trainable_parameters,
    require_torch,
)
from filters.breakout_quality.models.spec import (
    get_model_spec,
    validate_model_sequence_length,
)
from filters.breakout_quality.paths import (
    resolve_filter_artifact_paths,
    resolve_filter_model_output_dir,
)
from filters.breakout_quality.splits import (
    build_selection_oos_split_assignments,
    resolve_breakout_quality_outer_policy,
)
from filters.breakout_quality.torch_runtime import (
    SUPPORTED_MIXED_PRECISION_DTYPES,
    SUPPORTED_TORCH_DEVICES,
    autocast_context,
    build_grad_scaler,
    resolve_torch_execution_plan,
    seed_torch,
)
from filters.breakout_quality.workflow_io import (
    PROJECT_ROOT,
    load_validated_dataset_bundle,
    write_json,
)
from core.console_report import (
    compact_console_enabled,
    print_artifact_paths,
    project_relative_display_path,
)

RANKER_SCHEMA_VERSION = 2
RANKER_SCORE_FILENAME = "continuous_ranker_scores.csv"
RANKER_REPORT_JSON_FILENAME = "continuous_ranker_report.json"
RANKER_REPORT_MARKDOWN_FILENAME = "continuous_ranker_report.md"
RANKER_TARGET_FILENAME = "group_target_daily_percentile.npy"


PAIRWISE_TRAINING_CONTRACT = {
    "pair_scope": "same_date_non_tied_target_pairs",
    "pair_weighting": "equal_pair_weight",
    "model_margin": "pass_logit_minus_reject_logit",
    "batching": "whole_date_pack_no_date_split",
    "runtime_score": "softmax_pass_probability",
}

LISTWISE_TRAINING_CONTRACT = {
    "list_scope": "same_date_full_candidate_list",
    "target_distribution": "softmax_daily_percentile",
    "prediction_distribution": "softmax_pass_minus_reject_margin",
    "tie_handling": "equal_target_equal_distribution_weight",
    "date_weighting": "equal_rankable_date_weight",
    "model_margin": "pass_logit_minus_reject_logit",
    "batching": "whole_date_pack_no_date_split",
    "runtime_score": "softmax_pass_probability",
}


def _training_semantics(profile) -> dict[str, Any]:
    if profile.training_objective == TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING:
        return {
            "batching": PAIRWISE_TRAINING_CONTRACT["batching"],
            "pairwise_contract": dict(PAIRWISE_TRAINING_CONTRACT),
            "listwise_contract": None,
        }
    if profile.training_objective == TRAINING_OBJECTIVE_DAILY_LISTWISE_RANKING:
        return {
            "batching": LISTWISE_TRAINING_CONTRACT["batching"],
            "pairwise_contract": None,
            "listwise_contract": dict(LISTWISE_TRAINING_CONTRACT),
        }
    return {
        "batching": "shuffled_unique_group_batches",
        "pairwise_contract": None,
        "listwise_contract": None,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Research-only continuous ranking training；保留9A InceptionTime與兩logit head，"
            "由experiment profile決定continuous target與label scope"
        )
    )
    parser.add_argument("--filter-id", default=BREAKOUT_QUALITY_DEFAULT_FILTER_ID)
    parser.add_argument(
        "--model-architecture",
        default=BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
        help="continuous ranker architecture；必須是 sequence-only model",
    )
    parser.add_argument(
        "--experiment-profile",
        default=STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE,
        choices=(
            STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE,
            STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE,
            STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE,
            STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE,
            STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_LISTWISE_PROFILE,
        ),
    )
    parser.add_argument("--epochs", type=int, default=BREAKOUT_QUALITY_DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BREAKOUT_QUALITY_DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--evaluation-batch-size",
        type=int,
        default=BREAKOUT_QUALITY_EVALUATION_BATCH_SIZE,
    )
    parser.add_argument("--lr", type=float, default=BREAKOUT_QUALITY_DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=BREAKOUT_QUALITY_DEFAULT_WEIGHT_DECAY)
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=BREAKOUT_QUALITY_DEFAULT_GRADIENT_CLIP_NORM,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=resolve_breakout_quality_random_seed(),
        help="亂數種子；省略時使用config的BREAKOUT_QUALITY_RANDOM_SEED",
    )
    parser.add_argument(
        "--use-inner-validation",
        action=argparse.BooleanOptionalAction,
        default=BREAKOUT_QUALITY_USE_INNER_VALIDATION,
    )
    parser.add_argument(
        "--inner-validation-months",
        type=int,
        default=BREAKOUT_QUALITY_INNER_VALIDATION_MONTHS,
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=BREAKOUT_QUALITY_EARLY_STOPPING_PATIENCE,
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=BREAKOUT_QUALITY_EARLY_STOPPING_MIN_DELTA,
    )
    parser.add_argument("--device", choices=SUPPORTED_TORCH_DEVICES, default=BREAKOUT_QUALITY_TORCH_DEVICE)
    parser.add_argument(
        "--mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=BREAKOUT_QUALITY_USE_MIXED_PRECISION,
    )
    parser.add_argument(
        "--mixed-precision-dtype",
        choices=SUPPORTED_MIXED_PRECISION_DTYPES,
        default=BREAKOUT_QUALITY_MIXED_PRECISION_DTYPE,
    )
    parser.add_argument(
        "--deterministic-algorithms",
        action=argparse.BooleanOptionalAction,
        default=BREAKOUT_QUALITY_DETERMINISTIC_ALGORITHMS,
    )
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=BREAKOUT_QUALITY_ALLOW_TF32,
    )
    parser.add_argument(
        "--preload-feature-bank",
        action=argparse.BooleanOptionalAction,
        default=BREAKOUT_QUALITY_PRELOAD_FEATURE_BANK,
    )
    parser.add_argument(
        "--allow-stale-source",
        action="store_true",
        help="只供離線重現；預設要求來源CSV inventory與dataset一致",
    )
    args = parser.parse_args(argv)
    if int(args.seed) < 0:
        parser.error("--seed 必須 >= 0")
    return args


def _validate_args(args) -> None:
    profile = get_breakout_quality_experiment_profile(args.experiment_profile)
    model_spec = get_model_spec(str(args.model_architecture))
    if (
        bool(model_spec.requires_market_set)
        or bool(model_spec.use_dataset_context)
        or bool(model_spec.derived_context_features)
    ):
        raise ValueError("continuous ranker只允許sequence-only architecture")
    if profile.training_objective not in CONTINUOUS_RANKER_TRAINING_OBJECTIVES:
        raise ValueError("continuous ranker命令只接受continuous ranking profile")
    expected_targets = {
        STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE: (
            STRATEGY_ALIGNED_TARGET_ID,
            TRAINING_LABEL_SCOPE_ALL,
        ),
        STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE: (
            STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
            TRAINING_LABEL_SCOPE_PASS_ONLY,
        ),
        STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE: (
            STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
            TRAINING_LABEL_SCOPE_ALL,
        ),
        STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE: (
            STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
            TRAINING_LABEL_SCOPE_ALL,
        ),
        STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_LISTWISE_PROFILE: (
            STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
            TRAINING_LABEL_SCOPE_ALL,
        ),
    }
    expected = expected_targets.get(args.experiment_profile)
    if expected is None or (profile.continuous_target_id, profile.training_label_scope) != expected:
        raise ValueError("continuous ranker profile target／label scope契約不一致")
    if not bool(args.use_inner_validation):
        raise ValueError("continuous ranker必須使用inner validation選epoch")
    if int(args.epochs) < 1 or int(args.batch_size) < 2 or int(args.evaluation_batch_size) < 1:
        raise ValueError("epochs>=1、batch-size>=2、evaluation-batch-size>=1")
    if float(args.lr) <= 0.0 or float(args.weight_decay) < 0.0:
        raise ValueError("learning rate必須>0，weight decay必須>=0")
    if float(args.gradient_clip_norm) < 0.0:
        raise ValueError("gradient clip norm必須>=0")
    if int(args.inner_validation_months) < 1 or int(args.early_stopping_patience) < 0:
        raise ValueError("inner validation months必須>=1，patience必須>=0")
    if float(args.early_stopping_min_delta) < 0.0:
        raise ValueError("early stopping min delta必須>=0")


def _source_data_end(summary: dict[str, Any], events: pd.DataFrame) -> str:
    source_range = summary.get("source_data_date_range")
    if isinstance(source_range, dict):
        value = str(source_range.get("end") or "").strip()
        if value:
            return value
    return str(pd.to_datetime(events["label_eval_end_date"], errors="raise").max().date())


def _group_table(events: pd.DataFrame, event_group_index: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    frame = events[["ticker", "date", "group_index"]].copy()
    frame["event_row"] = np.arange(len(frame), dtype=np.int64)
    frame["label"] = np.asarray(labels, dtype=np.int64)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    group = frame.drop_duplicates("group_index", keep="first").sort_values("group_index", kind="mergesort")
    expected = np.arange(len(group), dtype=np.int64)
    observed = pd.to_numeric(group["group_index"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(observed, expected):
        raise ValueError("continuous ranker要求group_index連續完整")
    if not np.array_equal(
        np.asarray(event_group_index[group["event_row"].to_numpy(dtype=np.int64)], dtype=np.int64),
        expected,
    ):
        raise ValueError("continuous ranker group representative與event_group_index不一致")
    mixed = frame.groupby("group_index", sort=False)["label"].nunique()
    if bool((mixed != 1).any()):
        raise ValueError("continuous ranker發現同group混合binary label")
    return group.reset_index(drop=True)


def _profile_contract(profile) -> dict[str, str]:
    if profile.name == STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE:
        return {
            "experiment": "11B Strategy-aligned Daily Percentile Ranker",
            "phase": "11B",
            "target_description": "same_date_rank_percentile_of_strategy_aligned_opportunity_r_v1",
            "objective_description": "同日11A target percentile的MSE",
            "metric_scope": "all_labels",
        }
    if profile.name == STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE:
        return {
            "experiment": "11G PASS-conditional No-time Magnitude Ranker",
            "phase": "11G",
            "target_description": "same_date_pass_only_rank_percentile_of_strategy_aligned_opportunity_no_time_r_v1",
            "objective_description": "同日PASS-only 11F No-time target percentile的MSE",
            "metric_scope": "pass_only",
        }
    if profile.name == STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE:
        return {
            "experiment": "MR-12A All-event No-time Continuous Ranker",
            "phase": "12A",
            "target_description": "same_date_all_event_rank_percentile_of_strategy_aligned_opportunity_no_time_r_v1",
            "objective_description": "同日all-event No-time target percentile的MSE",
            "metric_scope": "all_labels",
        }
    if profile.name == STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE:
        return {
            "experiment": "MR-12B All-event No-time Pairwise Ranker",
            "phase": "12B",
            "target_description": "same_date_all_event_order_of_strategy_aligned_opportunity_no_time_r_v1",
            "objective_description": "同日all-event No-time target ordering的RankNet pairwise logistic loss",
            "metric_scope": "all_labels",
        }
    if profile.name == STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_LISTWISE_PROFILE:
        return {
            "experiment": "MR-12C All-event No-time ListNet Top-one Ranker",
            "phase": "12C",
            "target_description": "same_date_all_event_listnet_distribution_of_strategy_aligned_opportunity_no_time_r_v1",
            "objective_description": "同日all-event No-time完整候選榜單的ListNet top-one cross-entropy",
            "metric_scope": "all_labels",
        }
    raise ValueError(f"不支援的continuous ranker profile: {profile.name}")


def _scope_group_ids(
    group_ids: np.ndarray,
    group_table: pd.DataFrame,
    *,
    label_scope: str,
) -> np.ndarray:
    ids = np.asarray(group_ids, dtype=np.int64)
    if label_scope == TRAINING_LABEL_SCOPE_ALL:
        return ids
    if label_scope == TRAINING_LABEL_SCOPE_PASS_ONLY:
        labels = group_table.iloc[ids]["label"].to_numpy(dtype=np.int64)
        return ids[labels == LABEL_PASS]
    raise ValueError(f"不支援的training label scope: {label_scope}")


def build_daily_percentile_targets(
    raw_target: np.ndarray,
    valid_mask: np.ndarray,
    group_dates: pd.Series | np.ndarray,
) -> np.ndarray:
    """Return group-level [0,1] ranks using only same-date target values."""

    values = np.asarray(raw_target, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    dates = pd.to_datetime(pd.Series(group_dates), errors="raise").dt.normalize()
    if values.ndim != 1 or valid.ndim != 1 or values.shape != valid.shape or len(dates) != len(values):
        raise ValueError("daily percentile target input shape不一致")
    result = np.full(values.shape, np.nan, dtype=np.float32)
    work = pd.DataFrame({"date": dates, "target": values, "group_index": np.arange(len(values))})
    work = work[valid & np.isfinite(values)].copy()
    for _date, day in work.groupby("date", sort=True):
        count = int(len(day))
        if count == 1:
            percentile = np.array([0.5], dtype=np.float64)
        else:
            ranks = day["target"].rank(method="average").to_numpy(dtype=np.float64)
            percentile = (ranks - 1.0) / float(count - 1)
        result[day["group_index"].to_numpy(dtype=np.int64)] = percentile.astype(np.float32)
    if bool(np.any(valid & ~np.isfinite(result))):
        raise ValueError("valid continuous target無法建立daily percentile")
    if bool(np.any(np.isfinite(result) & ((result < 0.0) | (result > 1.0)))):
        raise ValueError("daily percentile超出[0,1]")
    return result


def _group_ids_from_event_rows(event_group_index: np.ndarray, rows: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    group_ids = np.unique(np.asarray(event_group_index[np.asarray(rows, dtype=np.int64)], dtype=np.int64))
    group_ids = group_ids[np.asarray(valid_mask[group_ids], dtype=bool)]
    return np.asarray(group_ids, dtype=np.int64)


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    if int(valid.sum()) < 2:
        return None
    x_rank = pd.Series(x_values[valid]).rank(method="average").to_numpy(dtype=np.float64)
    y_rank = pd.Series(y_values[valid]).rank(method="average").to_numpy(dtype=np.float64)
    if float(np.std(x_rank)) == 0.0 or float(np.std(y_rank)) == 0.0:
        return None
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _daily_rank_metrics(dates: np.ndarray, scores: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    frame = pd.DataFrame({"date": pd.to_datetime(dates), "score": scores, "target": targets})
    daily_spearman: list[float] = []
    concordant = 0.0
    pair_count = 0
    for _date, day in frame.groupby("date", sort=True):
        s = day["score"].to_numpy(dtype=np.float64)
        t = day["target"].to_numpy(dtype=np.float64)
        corr = _spearman(s, t)
        if corr is not None:
            daily_spearman.append(float(corr))
        if len(day) >= 2:
            score_diff = s[:, None] - s[None, :]
            target_diff = t[:, None] - t[None, :]
            upper = np.triu(np.ones(score_diff.shape, dtype=bool), k=1)
            comparable = upper & (target_diff != 0.0)
            if bool(comparable.any()):
                signs = score_diff[comparable] * target_diff[comparable]
                concordant += float((signs > 0.0).sum()) + 0.5 * float((signs == 0.0).sum())
                pair_count += int(comparable.sum())
    return {
        "rankable_date_count": int(len(daily_spearman)),
        "mean_daily_spearman": float(np.mean(daily_spearman)) if daily_spearman else None,
        "median_daily_spearman": float(np.median(daily_spearman)) if daily_spearman else None,
        "pairwise_concordance": float(concordant / pair_count) if pair_count else None,
        "comparable_pair_count": int(pair_count),
    }


def _daily_top_k_metrics(
    dates: np.ndarray,
    scores: np.ndarray,
    raw_targets: np.ndarray,
    percentile_targets: np.ndarray,
    *,
    top_k: int,
    boundary_width: int,
) -> dict[str, Any]:
    """Backward-compatible private alias for the shared canonical quality metric."""

    return daily_top_k_metrics(
        dates,
        scores,
        raw_targets,
        percentile_targets,
        top_k=top_k,
        boundary_width=boundary_width,
    )

def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(s) & np.isin(y, [0, 1])
    y = y[valid]
    s = s[valid]
    positive_count = int((y == LABEL_PASS).sum())
    if positive_count == 0:
        return None
    order = np.argsort(-s, kind="mergesort")
    ranked = (y[order] == LABEL_PASS).astype(np.float64)
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1, dtype=np.float64)
    return float((precision * ranked).sum() / positive_count)


def _coverage_precision(labels: np.ndarray, scores: np.ndarray, coverage: float) -> float | None:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(s) & np.isin(y, [0, 1])
    y = y[valid]
    s = s[valid]
    if len(y) == 0:
        return None
    count = max(1, int(math.ceil(len(y) * float(coverage))))
    order = np.argsort(-s, kind="mergesort")[:count]
    return float((y[order] == LABEL_PASS).mean())


def _split_metrics(
    group_ids: np.ndarray,
    group_table: pd.DataFrame,
    raw_target: np.ndarray,
    percentile_target: np.ndarray,
    scores: np.ndarray,
    *,
    include_top_k_quality: bool = False,
) -> dict[str, Any]:
    ids = np.asarray(group_ids, dtype=np.int64)
    score_values = np.asarray(scores, dtype=np.float64)
    raw_values = np.asarray(raw_target[ids], dtype=np.float64)
    pct_values = np.asarray(percentile_target[ids], dtype=np.float64)
    labels = group_table.iloc[ids]["label"].to_numpy(dtype=np.int64)
    dates = group_table.iloc[ids]["date"].to_numpy()
    if len(score_values) != len(ids):
        raise ValueError("11B score/group長度不一致")
    order = np.argsort(score_values, kind="mergesort")
    decile_count = max(1, int(math.ceil(len(ids) * 0.10)))
    bottom = order[:decile_count]
    top = order[-decile_count:]
    daily = _daily_rank_metrics(dates, score_values, raw_values)
    top_k_quality = (
        _daily_top_k_metrics(
            dates,
            score_values,
            raw_values,
            pct_values,
            top_k=BREAKOUT_QUALITY_CONTINUOUS_RANKER_REPORT_TOP_K,
            boundary_width=BREAKOUT_QUALITY_CONTINUOUS_RANKER_REPORT_BOUNDARY_WIDTH,
        )
        if include_top_k_quality
        else None
    )
    finite_percentile = np.isfinite(pct_values) & np.isfinite(score_values)
    return {
        "group_count": int(len(ids)),
        "percentile_target_count": int(finite_percentile.sum()),
        "mse_vs_daily_percentile": (
            float(np.mean((score_values[finite_percentile] - pct_values[finite_percentile]) ** 2))
            if bool(finite_percentile.any())
            else None
        ),
        "global_spearman_vs_raw_target": _spearman(score_values, raw_values),
        "global_spearman_vs_daily_percentile": _spearman(
            score_values[finite_percentile],
            pct_values[finite_percentile],
        ),
        **daily,
        "top_k_quality": top_k_quality,
        "score_mean": float(score_values.mean()),
        "score_std": float(score_values.std(ddof=1)) if len(score_values) > 1 else 0.0,
        "top_score_decile_raw_target_mean": float(raw_values[top].mean()),
        "bottom_score_decile_raw_target_mean": float(raw_values[bottom].mean()),
        "binary_pr_auc": _average_precision(labels, score_values),
        "p_at_50pct": _coverage_precision(labels, score_values, 0.50),
        "p_at_60pct": _coverage_precision(labels, score_values, 0.60),
        "p_at_70pct": _coverage_precision(labels, score_values, 0.70),
    }


def _new_model_and_optimizer(torch, *, feature_count: int, context_count: int, args, plan):
    seed_torch(torch, seed=int(args.seed), plan=plan)
    model = build_model(
        feature_count=feature_count,
        context_count=context_count,
        architecture=str(args.model_architecture),
    ).to(plan.device)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    return model, optimizer


def _date_coherent_batches(
    group_ids: np.ndarray,
    group_dates: pd.Series | np.ndarray,
    *,
    batch_size: int,
    seed: int,
) -> list[np.ndarray]:
    """Pack whole trading dates into deterministic mini-batches for cross-sectional ranking."""

    ids = np.asarray(group_ids, dtype=np.int64)
    if ids.ndim != 1:
        raise ValueError("date-coherent ranker group_ids必須是一維")
    dates = pd.to_datetime(pd.Series(group_dates), errors="raise").dt.normalize()
    if len(dates) <= int(ids.max(initial=-1)):
        raise ValueError("date-coherent ranker group_dates長度不足")
    by_date: dict[pd.Timestamp, list[int]] = {}
    for group_id in ids:
        by_date.setdefault(pd.Timestamp(dates.iloc[int(group_id)]), []).append(int(group_id))
    date_keys = np.asarray(sorted(by_date), dtype=object)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(date_keys)
    batches: list[np.ndarray] = []
    current: list[int] = []
    for date_key in date_keys:
        day_ids = by_date[pd.Timestamp(date_key)]
        if current and len(current) + len(day_ids) > int(batch_size):
            batches.append(np.asarray(current, dtype=np.int64))
            current = []
        # Never split a date: cross-sectional ranking loss must see the full same-date set.
        if len(day_ids) > int(batch_size):
            if current:
                batches.append(np.asarray(current, dtype=np.int64))
                current = []
            batches.append(np.asarray(day_ids, dtype=np.int64))
        else:
            current.extend(day_ids)
    if current:
        batches.append(np.asarray(current, dtype=np.int64))
    return batches


def _pairwise_logistic_loss(torch, margins, targets, dates) -> tuple[Any | None, int]:
    """Return equal-weight RankNet logistic loss over comparable within-day pairs only."""

    import torch.nn.functional as F

    date_values = pd.to_datetime(pd.Series(dates), errors="raise").dt.normalize().to_numpy()
    losses = []
    pair_count = 0
    for date_value in pd.unique(date_values):
        positions = np.flatnonzero(date_values == date_value)
        if len(positions) < 2:
            continue
        pos = torch.as_tensor(positions, dtype=torch.long, device=margins.device)
        day_margin = margins.index_select(0, pos)
        day_target = targets.index_select(0, pos)
        margin_diff = day_margin[:, None] - day_margin[None, :]
        target_diff = day_target[:, None] - day_target[None, :]
        upper = torch.triu(
            torch.ones_like(target_diff, dtype=torch.bool), diagonal=1
        )
        comparable = upper & (target_diff != 0)
        count = int(comparable.sum().item())
        if count == 0:
            continue
        signs = torch.sign(target_diff[comparable])
        losses.append(F.softplus(-signs * margin_diff[comparable]))
        pair_count += count
    if not losses:
        return None, 0
    return torch.cat(losses).mean(), int(pair_count)


def _listnet_top_one_loss(torch, margins, targets, dates) -> tuple[Any | None, int]:
    """ListNet top-one cross-entropy over complete same-date candidate lists.

    Daily percentile targets are converted into a target top-one distribution with
    softmax; model margins form the predicted distribution.  Equal targets therefore
    receive equal target mass without any arbitrary tie ordering.  Only dates with at
    least two distinct target levels contribute ranking supervision, and each rankable
    date has equal weight.
    """

    import torch.nn.functional as F

    date_values = pd.to_datetime(pd.Series(dates), errors="raise").dt.normalize().to_numpy()
    day_losses = []
    ranked_date_count = 0
    for date_value in pd.unique(date_values):
        positions = np.flatnonzero(date_values == date_value)
        if len(positions) < 2:
            continue
        pos = torch.as_tensor(positions, dtype=torch.long, device=margins.device)
        day_margin = margins.index_select(0, pos)
        day_target = targets.index_select(0, pos)
        if int(torch.unique(day_target.detach()).numel()) < 2:
            continue
        target_distribution = torch.softmax(day_target.detach(), dim=0)
        prediction_log_distribution = F.log_softmax(day_margin, dim=0)
        day_losses.append(-(target_distribution * prediction_log_distribution).sum())
        ranked_date_count += 1
    if not day_losses:
        return None, 0
    return torch.stack(day_losses).mean(), int(ranked_date_count)


def _train_epoch(
    torch,
    model,
    optimizer,
    feature_bank: np.ndarray,
    group_context: np.ndarray,
    group_ids: np.ndarray,
    percentile_target: np.ndarray,
    group_dates: pd.Series | np.ndarray,
    *,
    training_objective: str,
    batch_size: int,
    seed: int,
    gradient_clip_norm: float,
    plan,
    grad_scaler,
) -> float:
    model.train()
    ids_all = np.asarray(group_ids, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    if training_objective == TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION:
        order = ids_all.copy()
        rng.shuffle(order)
        batches = [
            order[start:start + int(batch_size)]
            for start in range(0, len(order), int(batch_size))
        ]
    elif training_objective in {
        TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        TRAINING_OBJECTIVE_DAILY_LISTWISE_RANKING,
    }:
        batches = _date_coherent_batches(
            ids_all,
            group_dates,
            batch_size=int(batch_size),
            seed=int(seed),
        )
    else:
        raise ValueError(f"不支援的continuous ranker training objective: {training_objective!r}")

    losses: list[float] = []
    weighted_loss_sum = 0.0
    weighted_loss_count = 0
    import torch.nn.functional as F

    for ids in batches:
        if len(ids) == 0:
            continue
        xb = torch.from_numpy(np.asarray(feature_bank[ids], dtype=np.float32)).to(plan.device)
        cb = torch.from_numpy(np.asarray(group_context[ids], dtype=np.float32)).to(plan.device)
        target = torch.from_numpy(np.asarray(percentile_target[ids], dtype=np.float32)).to(plan.device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(torch, plan):
            logits = model(xb, cb)
            if training_objective == TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION:
                score = torch.softmax(logits.float(), dim=1)[:, LABEL_PASS]
                loss = F.mse_loss(score, target, reduction="mean")
                loss_weight = int(len(ids))
            elif training_objective == TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING:
                margins = logits.float()[:, LABEL_PASS] - logits.float()[:, LABEL_REJECT]
                loss, pair_count = _pairwise_logistic_loss(
                    torch,
                    margins,
                    target,
                    pd.Series(group_dates).iloc[ids].to_numpy(),
                )
                if loss is None:
                    continue
                loss_weight = int(pair_count)
            else:
                margins = logits.float()[:, LABEL_PASS] - logits.float()[:, LABEL_REJECT]
                loss, ranked_date_count = _listnet_top_one_loss(
                    torch,
                    margins,
                    target,
                    pd.Series(group_dates).iloc[ids].to_numpy(),
                )
                if loss is None:
                    continue
                loss_weight = int(ranked_date_count)
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("continuous ranker training loss非有限值")
        if grad_scaler is None:
            loss.backward()
            if float(gradient_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip_norm))
            optimizer.step()
        else:
            grad_scaler.scale(loss).backward()
            if float(gradient_clip_norm) > 0.0:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip_norm))
            grad_scaler.step(optimizer)
            grad_scaler.update()
        value = float(loss.detach().cpu().item())
        losses.append(value)
        weighted_loss_sum += value * float(loss_weight)
        weighted_loss_count += int(loss_weight)
    if not losses or weighted_loss_count < 1:
        raise ValueError("continuous ranker training沒有任何有效batch／ranking supervision")
    if training_objective == TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION:
        # Preserve MR-12A historical reporting semantics exactly.
        return float(np.mean(losses))
    return float(weighted_loss_sum / float(weighted_loss_count))

def _predict_scores(torch, model, feature_bank: np.ndarray, group_context: np.ndarray, group_ids: np.ndarray, *, batch_size: int, plan) -> np.ndarray:
    ids = np.asarray(group_ids, dtype=np.int64)
    logits = strict_parallel_batched_logits(
        torch,
        model,
        feature_bank,
        group_context,
        indices=ids,
        batch_size=int(batch_size),
        workers=1,
        execution_plan=plan,
    )
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return (exp[:, LABEL_PASS] / exp.sum(axis=1)).astype(np.float32)


def _select_epoch(
    torch,
    feature_bank: np.ndarray,
    group_context: np.ndarray,
    group_table: pd.DataFrame,
    raw_target: np.ndarray,
    percentile_target: np.ndarray,
    train_ids: np.ndarray,
    validation_ids: np.ndarray,
    *,
    args,
    plan,
) -> dict[str, Any]:
    model, optimizer = _new_model_and_optimizer(
        torch,
        feature_count=int(feature_bank.shape[2]),
        context_count=int(group_context.shape[1]),
        args=args,
        plan=plan,
    )
    grad_scaler = build_grad_scaler(torch, plan)
    best_epoch = 0
    best_metric = -math.inf
    best_validation_mse = math.inf
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    compact_console = compact_console_enabled()
    if not compact_console:
        print("\nEpoch選擇（依Validation mean daily Spearman）")
    for epoch in range(1, int(args.epochs) + 1):
        started = time.perf_counter()
        batch_loss = _train_epoch(
            torch,
            model,
            optimizer,
            feature_bank,
            group_context,
            train_ids,
            percentile_target,
            group_table["date"],
            training_objective=get_breakout_quality_experiment_profile(args.experiment_profile).training_objective,
            batch_size=int(args.batch_size),
            seed=int(args.seed) + epoch,
            gradient_clip_norm=float(args.gradient_clip_norm),
            plan=plan,
            grad_scaler=grad_scaler,
        )
        train_scores = _predict_scores(
            torch, model, feature_bank, group_context, train_ids,
            batch_size=int(args.evaluation_batch_size), plan=plan,
        )
        validation_scores = _predict_scores(
            torch, model, feature_bank, group_context, validation_ids,
            batch_size=int(args.evaluation_batch_size), plan=plan,
        )
        train_metrics = _split_metrics(
            train_ids, group_table, raw_target, percentile_target, train_scores,
        )
        validation_metrics = _split_metrics(
            validation_ids, group_table, raw_target, percentile_target, validation_scores,
        )
        metric = validation_metrics.get("mean_daily_spearman")
        if metric is None or not math.isfinite(float(metric)):
            raise ValueError("continuous ranker validation mean daily Spearman不可用")
        validation_mse = float(validation_metrics["mse_vs_daily_percentile"])
        improved = (
            float(metric) > best_metric + float(args.early_stopping_min_delta)
            or (
                math.isclose(float(metric), best_metric, rel_tol=0.0, abs_tol=1e-12)
                and validation_mse < best_validation_mse
            )
        )
        if improved:
            best_epoch = int(epoch)
            best_metric = float(metric)
            best_validation_mse = validation_mse
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        elapsed = time.perf_counter() - started
        history.append({
            "epoch": int(epoch),
            "batch_loss": float(batch_loss),
            "inner_train_metrics": train_metrics,
            "inner_validation_metrics": validation_metrics,
            "elapsed_sec": round(float(elapsed), 3),
            "is_best_epoch": bool(improved),
        })
        if not compact_console:
            marker = " ★新最佳" if improved else ""
            print(
                f"  Epoch {epoch:>3}/{int(args.epochs)} | Train Loss {batch_loss:.6f} | Train MSE {train_metrics['mse_vs_daily_percentile']:.6f} "
                f"| Val MSE {validation_mse:.6f} | Val Daily Spearman {float(metric):.4f} "
                f"| {elapsed:.1f}s{marker}"
            )
        if int(args.early_stopping_patience) > 0 and epochs_without_improvement >= int(args.early_stopping_patience):
            break
    if best_epoch < 1:
        raise ValueError("continuous ranker無法選出best epoch")
    return {
        "best_epoch": int(best_epoch),
        "best_validation_mean_daily_spearman": float(best_metric),
        "best_validation_mse": float(best_validation_mse),
        "completed_epochs": int(len(history)),
        "history": history,
    }


def _fit_final(
    torch,
    feature_bank: np.ndarray,
    group_context: np.ndarray,
    percentile_target: np.ndarray,
    group_table: pd.DataFrame,
    final_ids: np.ndarray,
    *,
    epochs: int,
    args,
    plan,
    phase_label: str = "完整Selection重訓",
):
    model, optimizer = _new_model_and_optimizer(
        torch,
        feature_count=int(feature_bank.shape[2]),
        context_count=int(group_context.shape[1]),
        args=args,
        plan=plan,
    )
    grad_scaler = build_grad_scaler(torch, plan)
    history: list[dict[str, Any]] = []
    compact_console = compact_console_enabled()
    if not compact_console:
        print(f"\n{str(phase_label)}（{int(epochs)} Epoch）")
    for epoch in range(1, int(epochs) + 1):
        started = time.perf_counter()
        loss = _train_epoch(
            torch,
            model,
            optimizer,
            feature_bank,
            group_context,
            final_ids,
            percentile_target,
            group_table["date"],
            training_objective=get_breakout_quality_experiment_profile(args.experiment_profile).training_objective,
            batch_size=int(args.batch_size),
            seed=int(args.seed) + epoch,
            gradient_clip_norm=float(args.gradient_clip_norm),
            plan=plan,
            grad_scaler=grad_scaler,
        )
        elapsed = time.perf_counter() - started
        history.append({"epoch": int(epoch), "batch_loss": float(loss), "elapsed_sec": round(float(elapsed), 3)})
        if not compact_console:
            print(f"  Epoch {epoch:>3}/{int(epochs)} | Batch Loss {loss:.6f} | {elapsed:.1f}s")
    return model.eval(), history


def _trade_metric_block(valid: pd.DataFrame) -> dict[str, Any]:
    if valid.empty:
        return {"matched_trade_count": 0}
    ordered_score = valid.sort_values("model_score", kind="mergesort")
    score_count = max(1, int(math.ceil(len(ordered_score) * 0.10)))
    ordered_target = valid.sort_values("target_raw_r", kind="mergesort")
    target_count = max(1, int(math.ceil(len(ordered_target) * 0.10)))
    scores = valid["model_score"].to_numpy(dtype=np.float64)
    targets = valid["target_raw_r"].to_numpy(dtype=np.float64)
    r_values = valid["r_multiple"].to_numpy(dtype=np.float64)
    large = r_values >= 2.0
    median_score = float(np.median(scores))
    median_target = float(np.median(targets))
    return {
        "matched_trade_count": int(len(valid)),
        "spearman_model_score_vs_r_multiple": _spearman(scores, r_values),
        "spearman_model_score_vs_target": _spearman(scores, targets),
        "spearman_target_vs_r_multiple": _spearman(targets, r_values),
        "top_model_score_decile_average_r": float(ordered_score.tail(score_count)["r_multiple"].mean()),
        "bottom_model_score_decile_average_r": float(ordered_score.head(score_count)["r_multiple"].mean()),
        "top_target_decile_average_r": float(ordered_target.tail(target_count)["r_multiple"].mean()),
        "bottom_target_decile_average_r": float(ordered_target.head(target_count)["r_multiple"].mean()),
        "large_winner_count_r_ge_2": int(large.sum()),
        "large_winner_top_half_score_retention": (
            float((scores[large] >= median_score).mean()) if bool(large.any()) else None
        ),
        "large_winner_top_half_target_retention": (
            float((targets[large] >= median_target).mean()) if bool(large.any()) else None
        ),
    }


def _trade_alignment_metrics(score_frame: pd.DataFrame, target_dir: Path) -> dict[str, Any]:
    path = target_dir / TARGET_TRADE_MATCHES_CSV_FILENAME
    if not path.is_file():
        display_path = project_relative_display_path(path, project_root=PROJECT_ROOT)
        return {
            "available": False,
            "reason": f"not found: {display_path}",
            "path": str(path),
        }
    trades = pd.read_csv(path, encoding="utf-8-sig")
    missing = sorted({"ticker", "target_date", "r_multiple"} - set(trades.columns))
    if missing:
        return {"available": False, "reason": f"trade matches missing columns: {missing}", "path": str(path)}
    lookup = score_frame[["ticker", "date", "label", "target_raw_r", "model_score"]].copy()
    lookup["ticker"] = lookup["ticker"].astype(str)
    lookup["target_date"] = pd.to_datetime(lookup["date"], errors="raise").dt.strftime("%Y-%m-%d")
    lookup = lookup.drop(columns=["date"])
    matched = trades.copy()
    # 11A trade-match files already contain the original target_raw_r.  The
    # ranker must evaluate the target version selected by its own profile, so
    # remove overlapping research columns before joining the score lookup.
    matched = matched.drop(
        columns=["label", "target_raw_r", "model_score"],
        errors="ignore",
    )
    matched["ticker"] = matched["ticker"].astype(str)
    matched["target_date"] = matched["target_date"].astype(str)
    matched["r_multiple"] = pd.to_numeric(matched["r_multiple"], errors="coerce")
    matched = matched.merge(lookup, how="left", on=["ticker", "target_date"], validate="many_to_one")
    valid = matched[
        np.isfinite(matched["r_multiple"])
        & np.isfinite(matched["model_score"])
        & np.isfinite(matched["target_raw_r"])
        & matched["label"].isin([LABEL_REJECT, LABEL_PASS])
    ].copy()
    result = {
        "available": True,
        "path": str(path),
        "trade_count": int(len(matched)),
        "coverage_rate": float(len(valid) / len(matched)) if len(matched) else None,
        **_trade_metric_block(valid),
        "label_conditional": {},
    }
    for label_name, label_value in (("PASS", LABEL_PASS), ("REJECT", LABEL_REJECT)):
        result["label_conditional"][label_name] = _trade_metric_block(
            valid[valid["label"] == label_value].copy()
        )
    return result


def _render_markdown(payload: dict[str, Any]) -> str:
    def fmt(value, digits=4):
        return "-" if value is None else f"{float(value):.{digits}f}"

    def fmt_pct(value, digits=2):
        return "-" if value is None else f"{float(value) * 100.0:.{digits}f}%"

    phase = payload["phase"]
    scope = payload["training"]["training_label_scope"]
    lines = [
        f"# {payload['experiment']}",
        "",
        f"- Profile：`{payload['experiment_profile']}`",
        f"- Architecture：`{payload['model_architecture']}`",
        f"- Objective：{payload['training']['objective_description']}；score為softmax PASS probability。",
        f"- Training label scope：`{scope}`。",
        "- Runtime：research-only；不得匯出forward-OOS runtime scores。",
        f"- Selected epoch：`{payload['training']['selected_epoch']}`",
        f"- Epoch-selection Validation Mean Daily Spearman：`{fmt((payload['training'].get('epoch_selection') or {}).get('best_validation_mean_daily_spearman'))}`",
        "- 下列 split metrics 均為完整 Selection refit checkpoint 建立後的描述性評估；原 Validation rows 已納入完整 Selection 重訓，不再具有選模 Validation 身分。",
        "",
        f"## 1. 完整 Selection 重訓後 split metrics（{scope}）",
        "",
        "| Split | Groups | MSE | Global Spearman | Mean Daily Spearman | Pair Concordance | Top10 Target | Bottom10 Target |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("inner_train", "validation", "selection", "oos"):
        row = payload["split_metrics"][name]
        lines.append(
            f"| {name} | {int(row['group_count']):,} | {fmt(row['mse_vs_daily_percentile'], 6)} "
            f"| {fmt(row['global_spearman_vs_raw_target'])} | {fmt(row['mean_daily_spearman'])} "
            f"| {fmt(row['pairwise_concordance'])} | {fmt(row['top_score_decile_raw_target_mean'])} "
            f"| {fmt(row['bottom_score_decile_raw_target_mean'])} |"
        )

    sample_top_k = dict((payload["split_metrics"].get("oos") or {}).get("top_k_quality") or {})
    top_k = int(sample_top_k.get("top_k") or BREAKOUT_QUALITY_CONTINUOUS_RANKER_REPORT_TOP_K)
    boundary_width = int(
        sample_top_k.get("boundary_width") or BREAKOUT_QUALITY_CONTINUOUS_RANKER_REPORT_BOUNDARY_WIDTH
    )
    lines.extend([
        "",
        f"## 2. Top-K / K-boundary quality（K={top_k}, boundary width={boundary_width}）",
        "",
        "| Split | Competition days | NDCG@K | Top-K Target | Candidate Target | Lift | Oracle overlap | Boundary days | Boundary concordance | Boundary gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name in ("inner_train", "validation", "selection", "oos"):
        quality = dict((payload["split_metrics"][name] or {}).get("top_k_quality") or {})
        lines.append(
            f"| {name} | {int(quality.get('top_k_date_count', 0) or 0):,} "
            f"| {fmt(quality.get('ndcg_at_k'))} | {fmt(quality.get('top_k_raw_target_mean'))} "
            f"| {fmt(quality.get('candidate_raw_target_mean'))} | {fmt(quality.get('top_k_raw_target_lift'))} "
            f"| {fmt_pct(quality.get('oracle_top_k_overlap'))} "
            f"| {int(quality.get('boundary_date_count', 0) or 0):,} "
            f"| {fmt_pct(quality.get('boundary_concordance'))} "
            f"| {fmt(quality.get('boundary_raw_target_gap'))} |"
        )
    lines.extend([
        "",
        "- Top-K主指標只統計candidate_count > K的competition days；candidate_count <= K時排序不會改變入選集合，因此排除以避免NDCG／Oracle overlap結構性偏高。",
        "- NDCG@K 使用同日 percentile target 作 relevance；其餘 Top-K／boundary 經濟指標使用 raw strategy-aligned R。",
        "- Top-K 與 K-boundary 採 equal-date weighting；K-boundary 比較 score 排名 K 內側與 K 外側各 boundary width 名。",
    ])

    next_section = 3
    if scope == TRAINING_LABEL_SCOPE_PASS_ONLY:
        lines.extend([
            "",
            f"## {next_section}. All-label diagnostic metrics",
            "",
            "| Split | Groups | Score↔Target | Mean Daily Spearman | Binary PR-AUC | P@50% | P@60% | P@70% |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for name in ("inner_train", "validation", "selection", "oos"):
            row = payload["all_group_split_metrics"][name]
            lines.append(
                f"| {name} | {int(row['group_count']):,} | {fmt(row['global_spearman_vs_raw_target'])} "
                f"| {fmt(row['mean_daily_spearman'])} | {fmt(row['binary_pr_auc'])} "
                f"| {fmt(row['p_at_50pct'])} | {fmt(row['p_at_60pct'])} | {fmt(row['p_at_70pct'])} |"
            )
        next_section += 1

    lines.extend(["", f"## {next_section}. Actual Round-trip R", ""])
    trade = payload.get("trade_alignment") or {}
    if trade.get("available"):
        coverage_rate = trade.get("coverage_rate")
        coverage_percent = None if coverage_rate is None else float(coverage_rate) * 100.0
        lines.extend([
            f"- 配對：`{trade.get('matched_trade_count')}` / `{trade.get('trade_count')}`；coverage `{fmt(coverage_percent, 2)}%`。",
            f"- Overall Score↔R：`{fmt(trade.get('spearman_model_score_vs_r_multiple'))}`；Score↔Target：`{fmt(trade.get('spearman_model_score_vs_target'))}`；Target↔R：`{fmt(trade.get('spearman_target_vs_r_multiple'))}`。",
            f"- Overall Score top／bottom decile平均R：`{fmt(trade.get('top_model_score_decile_average_r'))}`／`{fmt(trade.get('bottom_model_score_decile_average_r'))}`。",
        ])
        conditional = trade.get("label_conditional") or {}
        lines.extend([
            "",
            "| Label | Rows | Score↔Target | Score↔R | Target↔R | Score Top10 R | Score Bottom10 R |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for label_name in ("PASS", "REJECT"):
            row = conditional.get(label_name) or {}
            lines.append(
                f"| {label_name} | {int(row.get('matched_trade_count', 0)):,} "
                f"| {fmt(row.get('spearman_model_score_vs_target'))} "
                f"| {fmt(row.get('spearman_model_score_vs_r_multiple'))} "
                f"| {fmt(row.get('spearman_target_vs_r_multiple'))} "
                f"| {fmt(row.get('top_model_score_decile_average_r'))} "
                f"| {fmt(row.get('bottom_model_score_decile_average_r'))} |"
            )
    else:
        lines.append(f"- 未取得實際R診斷：`{trade.get('reason')}`。")

    next_section += 1
    lines.extend([
        "",
        f"## {next_section}. Boundary",
        "",
        "- Epoch、loss與gradient只使用Selection內Inner Train／Validation及profile指定label scope。",
        "- OOS percentile與推論只在完整Selection重訓、模型checkpoint寫入後建立。",
        "- Top-K／K-boundary只屬checkpoint後評估指標，不參與epoch選擇、loss、gradient或selector。",
        "- 本模型不設threshold、不提供runtime gate、不覆蓋9A正式模型。",
    ])
    if phase == "11G":
        lines.append("- 11G主要判定只看OOS PASS-only排序與actual PASS trades；overall／REJECT只作診斷。")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    args = parse_args(argv)
    _validate_args(args)
    started = time.perf_counter()
    profile = get_breakout_quality_experiment_profile(args.experiment_profile)
    contract = _profile_contract(profile)
    model_spec = get_model_spec(str(args.model_architecture))
    summary, indexed_features, context, labels, events = load_validated_dataset_bundle(
        args.filter_id,
        expected_policy=DEFAULT_LABEL_POLICY.as_manifest_payload(),
        require_current_source=not bool(args.allow_stale_source),
    )
    feature_bank = (
        np.array(indexed_features.feature_bank, dtype=np.float32, copy=True, order="C")
        if bool(args.preload_feature_bank)
        else indexed_features.feature_bank
    )
    event_group_index = np.asarray(indexed_features.event_group_index, dtype=np.int64)
    group_table = _group_table(events, event_group_index, labels)
    group_count = int(len(group_table))
    representative_rows = group_table["event_row"].to_numpy(dtype=np.int64)
    group_context = np.asarray(context[representative_rows], dtype=np.float32)
    if bool(args.preload_feature_bank):
        group_context = np.array(group_context, dtype=np.float32, copy=True, order="C")
    validate_model_sequence_length(model_spec, int(feature_bank.shape[1]))

    target_manifest, raw_target, target_valid = load_validated_continuous_target_arrays(
        PROJECT_ROOT,
        args.filter_id,
        target_id=str(profile.continuous_target_id),
        expected_group_count=group_count,
        expected_dataset_policy=summary.get("policy"),
        expected_dataset_artifacts=summary.get("dataset_artifacts"),
    )
    percentile_target = np.full(raw_target.shape, np.nan, dtype=np.float32)

    outer_policy = resolve_breakout_quality_outer_policy(
        PROJECT_ROOT,
        source_data_end_date=_source_data_end(summary, events),
    )
    (
        split_assignments,
        inner_train_rows,
        validation_rows,
        selection_rows,
        oos_rows,
        split_report,
    ) = build_selection_oos_split_assignments(
        events,
        labels,
        outer_policy=outer_policy,
        use_inner_validation=True,
        inner_validation_months=int(args.inner_validation_months),
        early_stopping_enabled=bool(int(args.early_stopping_patience) > 0),
    )
    all_split_ids = {
        "inner_train": _group_ids_from_event_rows(event_group_index, inner_train_rows, target_valid),
        "validation": _group_ids_from_event_rows(event_group_index, validation_rows, target_valid),
        "selection": _group_ids_from_event_rows(event_group_index, selection_rows, target_valid),
        "oos": _group_ids_from_event_rows(event_group_index, oos_rows, target_valid),
    }
    scoped_split_ids = {
        name: _scope_group_ids(
            ids,
            group_table,
            label_scope=profile.training_label_scope,
        )
        for name, ids in all_split_ids.items()
    }
    for name, ids in all_split_ids.items():
        if len(ids) < 20:
            raise ValueError(f"continuous ranker {name}有效group不足: {len(ids)}")
    for name, ids in scoped_split_ids.items():
        if len(ids) < 20:
            raise ValueError(
                f"continuous ranker {name}在{profile.training_label_scope} scope有效group不足: {len(ids)}"
            )

    selection_target_mask = np.zeros(raw_target.shape, dtype=bool)
    selection_target_mask[scoped_split_ids["selection"]] = True
    selection_percentiles = build_daily_percentile_targets(
        raw_target,
        selection_target_mask,
        group_table["date"],
    )
    percentile_target[scoped_split_ids["selection"]] = selection_percentiles[
        scoped_split_ids["selection"]
    ]

    torch, _nn = require_torch()
    plan = resolve_torch_execution_plan(
        torch,
        requested_device=str(args.device),
        mixed_precision=bool(args.mixed_precision),
        mixed_precision_dtype=str(args.mixed_precision_dtype),
        deterministic_algorithms=bool(args.deterministic_algorithms),
        allow_tf32=bool(args.allow_tf32),
    )
    if plan.device_type == "cpu":
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as exc:
            if "cannot set number of interop threads" not in str(exc):
                raise
    print(
        f"torch=device={plan.device_type}, mixed_precision={plan.mixed_precision_enabled}, "
        f"dtype={plan.autocast_dtype_name}, deterministic={plan.deterministic_algorithms}, tf32={plan.allow_tf32}"
    )
    print(
        f"profile={profile.name} target={profile.continuous_target_id} "
        f"training_label_scope={profile.training_label_scope}"
    )

    epoch_selection = _select_epoch(
        torch,
        feature_bank,
        group_context,
        group_table,
        raw_target,
        percentile_target,
        scoped_split_ids["inner_train"],
        scoped_split_ids["validation"],
        args=args,
        plan=plan,
    )
    selected_epoch = int(epoch_selection["best_epoch"])
    model, final_history = _fit_final(
        torch,
        feature_bank,
        group_context,
        percentile_target,
        group_table,
        scoped_split_ids["selection"],
        epochs=selected_epoch,
        args=args,
        plan=plan,
    )

    artifact_paths = resolve_filter_artifact_paths(
        PROJECT_ROOT,
        args.filter_id,
        str(args.model_architecture),
        args.experiment_profile,
    )
    artifact_paths.model_dir.mkdir(parents=True, exist_ok=True)
    trainable_parameter_count = count_trainable_parameters(model)
    total_parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())
    torch.save(
        {
            "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "feature_count": int(feature_bank.shape[2]),
            "context_count": int(group_context.shape[1]),
            "sequence_length": int(feature_bank.shape[1]),
            "model_spec": model_spec.as_manifest_payload(),
            "experiment_profile": args.experiment_profile,
            "experiment_settings": profile.as_manifest_payload(),
            "training_objective": profile.training_objective,
            "training_label_scope": profile.training_label_scope,
            "continuous_target_contract": target_manifest.get("target_contract"),
            "training_semantics": _training_semantics(profile),
            "torch_execution": plan.as_manifest_payload(),
            "trainable_parameter_count": int(trainable_parameter_count),
            "total_parameter_count": int(total_parameter_count),
        },
        artifact_paths.model_path,
    )
    split_assignments.to_csv(artifact_paths.split_path, index=False, encoding="utf-8-sig")

    # OOS target transformation and model inference occur only after the frozen
    # checkpoint exists; same-date ranks never mix Selection and OOS dates.
    oos_target_mask = np.zeros(raw_target.shape, dtype=bool)
    oos_target_mask[scoped_split_ids["oos"]] = True
    oos_percentiles = build_daily_percentile_targets(
        raw_target,
        oos_target_mask,
        group_table["date"],
    )
    percentile_target[scoped_split_ids["oos"]] = oos_percentiles[
        scoped_split_ids["oos"]
    ]

    score_by_group = np.full((group_count,), np.nan, dtype=np.float32)
    all_group_split_metrics: dict[str, Any] = {}
    split_metrics: dict[str, Any] = {}
    for name, ids in all_split_ids.items():
        scores = _predict_scores(
            torch,
            model,
            feature_bank,
            group_context,
            ids,
            batch_size=int(args.evaluation_batch_size),
            plan=plan,
        )
        score_by_group[ids] = scores
        all_group_split_metrics[name] = _split_metrics(
            ids,
            group_table,
            raw_target,
            percentile_target,
            scores,
            include_top_k_quality=True,
        )
        scoped_ids = scoped_split_ids[name]
        split_metrics[name] = _split_metrics(
            scoped_ids,
            group_table,
            raw_target,
            percentile_target,
            score_by_group[scoped_ids],
            include_top_k_quality=True,
        )

    role_by_group = np.full((group_count,), "selection_other", dtype=object)
    role_by_group[all_split_ids["inner_train"]] = "inner_train"
    role_by_group[all_split_ids["validation"]] = "validation"
    role_by_group[all_split_ids["oos"]] = "oos"
    score_frames: list[pd.DataFrame] = []
    for name in ("selection", "oos"):
        ids = all_split_ids[name]
        frame = group_table.iloc[ids][["ticker", "date", "group_index", "label"]].copy()
        frame["split"] = name
        frame["selection_role"] = role_by_group[ids]
        frame["in_training_label_scope"] = np.isin(ids, scoped_split_ids[name])
        frame["target_raw_r"] = raw_target[ids]
        frame["target_daily_percentile"] = percentile_target[ids]
        frame["model_score"] = score_by_group[ids]
        score_frames.append(frame)
    score_frame = pd.concat(score_frames, ignore_index=True)
    if bool(score_frame["group_index"].duplicated().any()):
        raise ValueError("continuous ranker research scores每個group必須唯一")
    score_frame = score_frame.sort_values(
        ["date", "ticker", "group_index"],
        kind="mergesort",
    ).reset_index(drop=True)

    output_dir = resolve_filter_model_output_dir(
        PROJECT_ROOT,
        args.filter_id,
        str(args.model_architecture),
        args.experiment_profile,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / RANKER_SCORE_FILENAME
    report_json_path = output_dir / RANKER_REPORT_JSON_FILENAME
    report_markdown_path = output_dir / RANKER_REPORT_MARKDOWN_FILENAME
    percentile_path = output_dir / RANKER_TARGET_FILENAME
    score_frame.to_csv(score_path, index=False, encoding="utf-8-sig")
    np.save(percentile_path, percentile_target, allow_pickle=False)

    # Round-trip membership was established by the original 11A audit and is
    # independent of which continuous target version the ranker uses.
    trade_source_dir = resolve_continuous_target_dir(
        PROJECT_ROOT,
        args.filter_id,
        target_id=STRATEGY_ALIGNED_TARGET_ID,
    )
    trade_alignment = _trade_alignment_metrics(
        score_frame[score_frame["split"] == "oos"].copy(),
        trade_source_dir,
    )
    information_cutoff = str(
        pd.to_datetime(events.iloc[selection_rows]["label_eval_end_date"], errors="raise").max().date()
    )
    training_group_counts = {
        name: int(len(ids)) for name, ids in scoped_split_ids.items()
    }
    payload = {
        "schema_version": RANKER_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": contract["experiment"],
        "phase": contract["phase"],
        "status": "RESULT_AVAILABLE_PENDING_REVIEW",
        "filter_id": args.filter_id,
        "model_architecture": str(args.model_architecture),
        "experiment_profile": args.experiment_profile,
        "experiment_settings": profile.as_manifest_payload(),
        "training": {
            "objective": profile.training_objective,
            "objective_description": contract["objective_description"],
            "loss": profile.loss_name,
            "model_score": "softmax_pass_probability",
            "batching": _training_semantics(profile)["batching"],
            "pairwise_contract": _training_semantics(profile)["pairwise_contract"],
            "listwise_contract": _training_semantics(profile)["listwise_contract"],
            "target": contract["target_description"],
            "training_label_scope": profile.training_label_scope,
            "training_group_counts": training_group_counts,
            "selected_epoch": selected_epoch,
            "epoch_selection_metric": profile.epoch_selection_metric,
            "epoch_selection": epoch_selection,
            "final_refit_history": final_history,
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "gradient_clip_norm": float(args.gradient_clip_norm),
            "seed": int(args.seed),
        },
        "split_report": split_report,
        "split_metrics": split_metrics,
        "all_group_split_metrics": all_group_split_metrics,
        "trade_alignment": trade_alignment,
        "target_manifest": target_manifest,
        "model_information_cutoff": information_cutoff,
        "oos_used_for_training_or_epoch_selection": False,
        "oos_evaluated_after_checkpoint_write": True,
        "runtime_eligibility": {
            "eligible": False,
            "scope": "research_only",
            "reason": f"{contract['phase']} is a research continuous-ranking objective pending controlled strategy deployment validation",
        },
        "artifacts": {
            "model": build_file_manifest(artifact_paths.model_path),
            "split_assignments": build_file_manifest(artifact_paths.split_path),
            "scores": build_file_manifest(score_path),
            "daily_percentile_target": build_file_manifest(percentile_path),
        },
        "torch_execution": plan.as_manifest_payload(),
        "elapsed_sec": round(time.perf_counter() - started, 3),
    }
    write_json(report_json_path, payload)
    report_markdown_path.write_text(_render_markdown(payload), encoding="utf-8")

    manifest = {
        "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
        "research_schema_version": RANKER_SCHEMA_VERSION,
        "filter_family": FILTER_FAMILY,
        "filter_id": args.filter_id,
        "model_architecture": str(args.model_architecture),
        "experiment_profile": args.experiment_profile,
        "experiment_settings": profile.as_manifest_payload(),
        "model_spec": model_spec.as_manifest_payload(),
        "training_objective": profile.training_objective,
        "training_label_scope": profile.training_label_scope,
        "training_semantics": _training_semantics(profile),
        "continuous_target_id": profile.continuous_target_id,
        "sequence_length": int(feature_bank.shape[1]),
        "feature_columns": list(FEATURE_COLUMNS),
        "context_columns": list(CONTEXT_COLUMNS),
        "model_filename": DEFAULT_MODEL_FILENAME,
        "manifest_filename": DEFAULT_MANIFEST_FILENAME,
        "split_filename": DEFAULT_SPLIT_FILENAME,
        "model": build_file_manifest(artifact_paths.model_path),
        "split_assignments": build_file_manifest(artifact_paths.split_path),
        "outer_oos_policy": outer_policy,
        "split_report": split_report,
        "training_group_counts": training_group_counts,
        "selected_epoch": selected_epoch,
        "epoch_selection_source": (
            "inner_validation_pass_only_mean_daily_spearman"
            if profile.training_label_scope == TRAINING_LABEL_SCOPE_PASS_ONLY
            else "inner_validation_mean_daily_spearman"
        ),
        "model_information_cutoff": information_cutoff,
        "source_dataset": summary,
        "source_continuous_target": target_manifest,
        "trainable_parameter_count": int(trainable_parameter_count),
        "total_parameter_count": int(total_parameter_count),
        "torch_execution": plan.as_manifest_payload(),
        "oos_predictions_used_during_training": False,
        "runtime_eligibility": payload["runtime_eligibility"],
        "research_outputs": {
            "scores": build_file_manifest(score_path),
            "daily_percentile_target": build_file_manifest(percentile_path),
            "report_json": build_file_manifest(report_json_path),
            "report_markdown": build_file_manifest(report_markdown_path),
        },
    }
    write_json(artifact_paths.manifest_path, manifest)

    print(f"\n{contract['phase']} continuous ranker完成")
    print(
        f"selected_epoch={selected_epoch} "
        f"epoch_selection_validation_daily_spearman={epoch_selection['best_validation_mean_daily_spearman']:.4f}"
    )
    print("checkpoint後split metrics（原Validation rows已納入完整Selection重訓；以下不再用於選模）")
    for name in ("inner_train", "validation", "selection", "oos"):
        metrics = split_metrics[name]
        print(
            f"- {name:<11} scope={profile.training_label_scope} groups={metrics['group_count']:,} "
            f"daily_spearman={metrics['mean_daily_spearman']:.4f} "
            f"global_spearman={metrics['global_spearman_vs_raw_target']:.4f}"
        )
    if trade_alignment.get("available") and trade_alignment.get("matched_trade_count", 0):
        pass_trade = (trade_alignment.get("label_conditional") or {}).get("PASS") or {}
        print(
            "- trade R: "
            f"matched={trade_alignment['matched_trade_count']}/{trade_alignment['trade_count']} "
            f"overall_spearman={trade_alignment['spearman_model_score_vs_r_multiple']:.4f} "
            f"pass_spearman={pass_trade.get('spearman_model_score_vs_r_multiple')}"
        )
    else:
        print(f"- trade R: {trade_alignment.get('reason', 'unavailable')}")
    print_artifact_paths(
        (("Continuous ranker model", artifact_paths.model_path), ("Markdown", report_markdown_path)),
        project_root=PROJECT_ROOT,
    )
    return 0


__all__ = [
    "RANKER_SCHEMA_VERSION",
    "build_daily_percentile_targets",
    "main",
    "parse_args",
]


if __name__ == "__main__":
    raise SystemExit(main())
