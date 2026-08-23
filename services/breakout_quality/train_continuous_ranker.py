"""Research-only continuous rankers using the active InceptionTime model."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from types import SimpleNamespace
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.breakout_quality import (
    STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE,
    CONTINUOUS_RANKER_TRAINING_OBJECTIVES,
    CONTINUOUS_RANKER_PAIRWISE_REDUCTION_EQUAL_PAIR,
    CONTINUOUS_RANKER_PAIRWISE_REDUCTION_TARGET_GAP_WEIGHTED,
    CONTINUOUS_RANKER_PAIRWISE_REDUCTION_UPPER_TAIL_RELEVANCE,
    CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
    CONTINUOUS_RANKER_PAIRWISE_REDUCTION_PARETO_DOMINANCE,
    CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
    CONTINUOUS_RANKER_TRAINER_EVENT,
    SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES,
    TRAINING_LABEL_SCOPE_ALL,
    TRAINING_LABEL_SCOPE_PASS_ONLY,
    TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION,
    TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION,
    TRAINING_OBJECTIVE_DAILY_DUAL_COMPONENT_R_REGRESSION,
    TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_PARETO_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_LISTWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_SAFETY_PAIRWISE_RANKING,
    TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    get_breakout_quality_experiment_profile,
    get_continuous_ranker_execution_recipe,
    get_continuous_ranker_research_spec,
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
    BREAKOUT_QUALITY_CONTINUOUS_RANKER_TRAIN_PREFETCH_BATCHES,
    BREAKOUT_QUALITY_CONTINUOUS_RANKER_PREFETCH_WORKERS,
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
from filters.breakout_quality.continuous_ranker_data import build_same_date_percentile_targets, source_data_end
from filters.breakout_quality.conditional_mfe_safety import (
    ConditionalMfeSafetyTargets,
    build_conditional_mfe_safety_targets,
)
from filters.breakout_quality.continuous_ranker_quality import (
    daily_top_k_metrics as shared_daily_top_k_metrics,
)
from filters.breakout_quality.continuous_target import (
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
from filters.breakout_quality.models.active import (
    ACTIVE_MODEL_ARCHITECTURES,
    build_active_model as build_model,
    get_active_model_spec as get_model_spec,
)
from filters.breakout_quality.models.runtime import (
    count_trainable_parameters,
    require_torch,
)
from filters.breakout_quality.models.spec import validate_model_sequence_length
from filters.breakout_quality.paths import (
    build_filter_artifact_paths_from_dir,
    resolve_filter_artifact_paths,
    resolve_filter_model_output_dir,
)
from filters.breakout_quality.ranker_sample_contract import (
    build_score_eligibility_contract,
    resolve_forward_oos_score_group_ids,
)
from filters.breakout_quality.ranker_training_contract import (
    CONDITIONAL_MFE_SAFETY_TRAINING_CONTRACT,
    LISTWISE_TRAINING_CONTRACT,
    PAIRWISE_TRAINING_CONTRACT,
    training_semantics,
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
from core.training_progress import emit_trainer_pit_progress_marker

RANKER_SCHEMA_VERSION = 2
RANKER_SCORE_FILENAME = "continuous_ranker_scores.csv"
RANKER_REPORT_JSON_FILENAME = "continuous_ranker_report.json"
RANKER_REPORT_MARKDOWN_FILENAME = "continuous_ranker_report.md"
RANKER_TARGET_FILENAME = "group_target_daily_percentile.npy"
EPOCH_PROGRESS_MARKER_ENV = "BREAKOUT_QUALITY_EPOCH_PROGRESS_MARKERS"


def _emit_epoch_progress_marker(phase: str, epoch: int, total_epochs: int) -> None:
    value = os.environ.get(EPOCH_PROGRESS_MARKER_ENV, "").strip().lower()
    if value not in {"1", "true", "yes", "on"}:
        return
    # PIT context is emitted from the same canonical epoch heartbeat when the
    # caller is a point-in-time builder.  This keeps fold progress near the log
    # tail instead of forcing outer orchestrators to infer it from console text.
    emit_trainer_pit_progress_marker()
    print(
        f"__BQ_EPOCH_PROGRESS__ phase={str(phase)} epoch={int(epoch)}/{int(total_epochs)}",
        flush=True,
    )


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
        choices=ACTIVE_MODEL_ARCHITECTURES,
        help="continuous ranker architecture；實際輸入能力由profile sample provider與model spec共同驗證",
    )
    parser.add_argument(
        "--experiment-profile",
        default=STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE,
        choices=SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES,
    )
    parser.add_argument("--epochs", type=int, default=BREAKOUT_QUALITY_DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BREAKOUT_QUALITY_DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--evaluation-batch-size",
        type=int,
        default=BREAKOUT_QUALITY_EVALUATION_BATCH_SIZE,
    )
    parser.add_argument(
        "--train-prefetch-batches",
        type=int,
        default=BREAKOUT_QUALITY_CONTINUOUS_RANKER_TRAIN_PREFETCH_BATCHES,
        help="預先物化後續continuous-ranker training batches；0 表示關閉",
    )
    parser.add_argument(
        "--train-prefetch-workers",
        type=int,
        default=BREAKOUT_QUALITY_CONTINUOUS_RANKER_PREFETCH_WORKERS,
        help="continuous-ranker CPU feature materialization workers；只影響feeding效能",
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
    parser.add_argument("--model-output-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--research-output-dir", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if int(args.seed) < 0:
        parser.error("--seed 必須 >= 0")
    return args


def resolve_training_output_paths(args):
    model_override = str(getattr(args, "model_output_dir", "") or "").strip()
    research_override = str(getattr(args, "research_output_dir", "") or "").strip()
    if bool(model_override) != bool(research_override):
        raise ValueError("model-output-dir與research-output-dir必須同時指定")
    if model_override:
        artifact_paths = build_filter_artifact_paths_from_dir(
            filter_id=str(args.filter_id),
            model_architecture=str(args.model_architecture),
            experiment_profile=str(args.experiment_profile),
            model_dir=Path(model_override),
        )
        output_dir = Path(research_override).resolve()
        return artifact_paths, output_dir
    return (
        resolve_filter_artifact_paths(
            PROJECT_ROOT, args.filter_id, str(args.model_architecture), args.experiment_profile
        ),
        resolve_filter_model_output_dir(
            PROJECT_ROOT, args.filter_id, str(args.model_architecture), args.experiment_profile
        ),
    )



def validate_args(args) -> None:
    profile = get_breakout_quality_experiment_profile(args.experiment_profile)
    model_spec = get_model_spec(str(args.model_architecture))
    if profile.training_objective not in CONTINUOUS_RANKER_TRAINING_OBJECTIVES:
        raise ValueError("continuous ranker命令只接受continuous ranking profile")
    execution_recipe = get_continuous_ranker_execution_recipe(str(args.experiment_profile))
    if bool(model_spec.requires_market_set) or bool(model_spec.derived_context_features):
        raise ValueError("continuous ranker不支援market-set／derived-context architecture")
    if (
        bool(model_spec.use_dataset_context)
        and execution_recipe.trainer_family != CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL
    ):
        raise ValueError("event continuous ranker只允許sequence-only architecture")
    expected_family = (
        CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL
        if profile.training_sample_scope == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
        else CONTINUOUS_RANKER_TRAINER_EVENT
    )
    if execution_recipe.trainer_family != expected_family:
        raise ValueError(
            "continuous ranker research spec與profile sample scope不一致: "
            f"profile={args.experiment_profile}, expected={expected_family}, "
            f"actual={execution_recipe.trainer_family}"
        )
    if not bool(args.use_inner_validation):
        raise ValueError("continuous ranker必須使用inner validation選epoch")
    if int(args.epochs) < 1 or int(args.batch_size) < 2 or int(args.evaluation_batch_size) < 1:
        raise ValueError("epochs>=1、batch-size>=2、evaluation-batch-size>=1")
    if int(args.train_prefetch_batches) < 0:
        raise ValueError("train-prefetch-batches必須 >=0")
    if int(getattr(args, "train_prefetch_workers", BREAKOUT_QUALITY_CONTINUOUS_RANKER_PREFETCH_WORKERS)) < 1:
        raise ValueError("train-prefetch-workers必須 >=1")
    if float(args.lr) <= 0.0 or float(args.weight_decay) < 0.0:
        raise ValueError("learning rate必須>0，weight decay必須>=0")
    if float(args.gradient_clip_norm) < 0.0:
        raise ValueError("gradient clip norm必須>=0")
    if int(args.inner_validation_months) < 1 or int(args.early_stopping_patience) < 0:
        raise ValueError("inner validation months必須>=1，patience必須>=0")
    if float(args.early_stopping_min_delta) < 0.0:
        raise ValueError("early stopping min delta必須>=0")


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
    research_spec = get_continuous_ranker_research_spec(profile.name)
    execution_recipe = get_continuous_ranker_execution_recipe(profile.name)
    return {
        "experiment": research_spec.experiment_name,
        "phase": research_spec.phase,
        "model_research_id": research_spec.model_research_id,
        "target_description": research_spec.target_description,
        "objective_description": research_spec.objective_description,
        "metric_scope": research_spec.metric_scope,
        "score_semantic_id": execution_recipe.score_semantic_id,
    }


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
    """Compatibility façade for the canonical same-date percentile target helper."""

    return build_same_date_percentile_targets(raw_target, valid_mask, group_dates)


def _group_ids_from_event_rows(event_group_index: np.ndarray, rows: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    group_ids = np.unique(np.asarray(event_group_index[np.asarray(rows, dtype=np.int64)], dtype=np.int64))
    group_ids = group_ids[np.asarray(valid_mask[group_ids], dtype=bool)]
    return np.asarray(group_ids, dtype=np.int64)


def calculate_spearman(x: np.ndarray, y: np.ndarray) -> float | None:
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


def daily_rank_metrics(dates: np.ndarray, scores: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    frame = pd.DataFrame({"date": pd.to_datetime(dates), "score": scores, "target": targets})
    daily_spearman: list[float] = []
    concordant = 0.0
    pair_count = 0
    for _date, day in frame.groupby("date", sort=True):
        s = day["score"].to_numpy(dtype=np.float64)
        t = day["target"].to_numpy(dtype=np.float64)
        corr = calculate_spearman(s, t)
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


def daily_top_k_metrics(
    dates: np.ndarray,
    scores: np.ndarray,
    raw_targets: np.ndarray,
    percentile_targets: np.ndarray,
    *,
    top_k: int,
    boundary_width: int,
) -> dict[str, Any]:
    """Return the shared canonical same-day Top-K quality metric."""

    return shared_daily_top_k_metrics(
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


def raw_r_regression_metrics(
    scores: np.ndarray,
    raw_target: np.ndarray,
    *,
    huber_delta_r: float | None = None,
) -> dict[str, Any]:
    predicted = np.asarray(scores, dtype=np.float64)
    actual = np.asarray(raw_target, dtype=np.float64)
    finite = np.isfinite(predicted) & np.isfinite(actual)
    predicted = predicted[finite]
    actual = actual[finite]
    if len(actual) == 0:
        return {
            "count": 0,
            "mse_raw_r": None,
            "huber_loss_raw_r": None,
            "mae_raw_r": None,
            "rmse_raw_r": None,
            "bias_raw_r": None,
            "predicted_r_mean": None,
            "target_r_mean": None,
        }
    error = predicted - actual
    squared_error = error * error
    abs_error = np.abs(error)
    huber_loss = None
    if huber_delta_r is not None:
        delta = float(huber_delta_r)
        if not math.isfinite(delta) or delta <= 0.0:
            raise ValueError("raw R regression Huber delta必須為正有限值")
        quadratic = np.minimum(abs_error, delta)
        linear = abs_error - quadratic
        huber = 0.5 * quadratic * quadratic + delta * linear
        huber_loss = float(np.mean(huber))
    return {
        "count": int(len(actual)),
        "mse_raw_r": float(np.mean(squared_error)),
        "huber_loss_raw_r": huber_loss,
        "mae_raw_r": float(np.mean(abs_error)),
        "rmse_raw_r": float(np.sqrt(np.mean(squared_error))),
        "bias_raw_r": float(np.mean(error)),
        "predicted_r_mean": float(np.mean(predicted)),
        "target_r_mean": float(np.mean(actual)),
    }


def split_metrics(
    group_ids: np.ndarray,
    group_table: pd.DataFrame,
    raw_target: np.ndarray,
    percentile_target: np.ndarray,
    scores: np.ndarray,
    *,
    include_top_k_quality: bool = False,
    raw_r_regression_loss_name: str | None = None,
    raw_r_huber_delta_r: float | None = None,
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
    daily = daily_rank_metrics(dates, score_values, raw_values)
    top_k_quality = (
        daily_top_k_metrics(
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
        "global_spearman_vs_raw_target": calculate_spearman(score_values, raw_values),
        "global_spearman_vs_daily_percentile": calculate_spearman(
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
        "raw_r_regression": (
            raw_r_regression_metrics(
                score_values, raw_values, huber_delta_r=raw_r_huber_delta_r
            )
            if raw_r_regression_loss_name is not None or raw_r_huber_delta_r is not None
            else None
        ),
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


def _pairwise_logistic_loss(
    torch,
    margins,
    targets,
    dates,
    *,
    reduction: str = CONTINUOUS_RANKER_PAIRWISE_REDUCTION_EQUAL_PAIR,
) -> tuple[Any | None, int]:
    """Return the configured RankNet loss over comparable within-day pairs.

    Existing equal-pair profiles preserve the original concatenated-pair mean exactly.
    MR-13B uses the absolute same-date percentile-target gap as pair weight, normalizes
    within each date, then gives each rankable date equal loss weight.
    MR-13D keeps the original pair aggregation but weights every comparable pair by the
    arithmetic mean of its two daily target percentiles, emphasizing upper-tail local
    ordering without any K, boundary, threshold, exponent, or mixing coefficient.
    MR-13E also keeps the original pair aggregation, but weights each pair by the absolute
    full-list NDCG change caused by swapping the pair at the current predicted positions.
    Raw 0～1 daily percentiles are the DCG gains and every rank position participates;
    there is no @K cutoff, gain exponent, boundary, threshold, or mixing coefficient.
    MR-13O receives a two-column target ``[MFE percentile, low-adverse percentile]``
    and keeps only strict Pareto-dominance pairs: both component differences must have
    the same non-zero sign. Trade-off or tied pairs receive no supervision and every
    comparable pair has equal weight.
    """

    import torch.nn.functional as F

    if reduction not in {
        CONTINUOUS_RANKER_PAIRWISE_REDUCTION_EQUAL_PAIR,
        CONTINUOUS_RANKER_PAIRWISE_REDUCTION_TARGET_GAP_WEIGHTED,
        CONTINUOUS_RANKER_PAIRWISE_REDUCTION_UPPER_TAIL_RELEVANCE,
        CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        CONTINUOUS_RANKER_PAIRWISE_REDUCTION_PARETO_DOMINANCE,
    }:
        raise ValueError(f"不支援的pairwise reduction: {reduction!r}")

    date_values = pd.to_datetime(pd.Series(dates), errors="raise").dt.normalize().to_numpy()
    pair_losses = []
    day_losses = []
    pair_count = 0
    ranked_date_count = 0
    for date_value in pd.unique(date_values):
        positions = np.flatnonzero(date_values == date_value)
        if len(positions) < 2:
            continue
        pos = torch.as_tensor(positions, dtype=torch.long, device=margins.device)
        day_margin = margins.index_select(0, pos)
        day_target = targets.index_select(0, pos)
        margin_diff = day_margin[:, None] - day_margin[None, :]
        if reduction == CONTINUOUS_RANKER_PAIRWISE_REDUCTION_PARETO_DOMINANCE:
            if day_target.ndim != 2 or int(day_target.shape[1]) != 2:
                raise ValueError("Pareto pairwise target必須為[N,2] component percentiles")
            component_diff = day_target[:, None, :] - day_target[None, :, :]
            item_count = int(day_target.shape[0])
            upper = torch.triu(
                torch.ones((item_count, item_count), dtype=torch.bool, device=day_target.device),
                diagonal=1,
            )
            positive = (component_diff[:, :, 0] > 0) & (component_diff[:, :, 1] > 0)
            negative = (component_diff[:, :, 0] < 0) & (component_diff[:, :, 1] < 0)
            comparable = upper & (positive | negative)
            count = int(comparable.sum().item())
            if count == 0:
                continue
            signs = torch.where(
                positive[comparable],
                torch.ones(count, dtype=day_margin.dtype, device=day_margin.device),
                -torch.ones(count, dtype=day_margin.dtype, device=day_margin.device),
            )
            losses = F.softplus(-signs * margin_diff[comparable])
            pair_losses.append(losses)
            pair_count += count
            ranked_date_count += 1
            continue

        if day_target.ndim != 1:
            raise ValueError("scalar pairwise target必須為一維")
        target_diff = day_target[:, None] - day_target[None, :]
        upper = torch.triu(
            torch.ones_like(target_diff, dtype=torch.bool), diagonal=1
        )
        comparable = upper & (target_diff != 0)
        count = int(comparable.sum().item())
        if count == 0:
            continue
        selected_target_diff = target_diff[comparable]
        signs = torch.sign(selected_target_diff)
        losses = F.softplus(-signs * margin_diff[comparable])
        pair_count += count
        ranked_date_count += 1
        if reduction == CONTINUOUS_RANKER_PAIRWISE_REDUCTION_EQUAL_PAIR:
            pair_losses.append(losses)
            continue

        if reduction == CONTINUOUS_RANKER_PAIRWISE_REDUCTION_TARGET_GAP_WEIGHTED:
            weights = torch.abs(selected_target_diff.detach())
            weight_sum = weights.sum()
            if not bool(torch.isfinite(weight_sum).item()) or float(weight_sum.detach().cpu().item()) <= 0.0:
                raise FloatingPointError("target-gap pairwise weight sum必須為正有限值")
            day_losses.append((losses * weights).sum() / weight_sum)
            continue

        if reduction == CONTINUOUS_RANKER_PAIRWISE_REDUCTION_UPPER_TAIL_RELEVANCE:
            selected_left_target = day_target[:, None].expand_as(target_diff)[comparable].detach()
            selected_right_target = day_target[None, :].expand_as(target_diff)[comparable].detach()
            weights = (selected_left_target + selected_right_target) / 2.0
            if not bool(torch.isfinite(weights).all().item()):
                raise FloatingPointError("upper-tail relevance pairwise weights必須為有限值")
            if bool((weights < 0.0).any().item()) or bool((weights > 1.0).any().item()):
                raise ValueError("upper-tail relevance只接受0～1 daily percentile target")
            pair_losses.append(losses * weights)
            day_losses.append(weights)
            continue

        detached_target = day_target.detach().float()
        detached_margin = day_margin.detach().float()
        # Full-list Delta-NDCG is the only path below. Collapse success-path
        # validation to one host synchronization; on failure, rerun the canonical
        # checks so exception type/message remain unchanged.
        valid_inputs = (
            torch.isfinite(detached_target).all()
            & (detached_target >= 0.0).all()
            & (detached_target <= 1.0).all()
            & torch.isfinite(detached_margin).all()
        )
        if not bool(valid_inputs.item()):
            if not bool(torch.isfinite(detached_target).all().item()):
                raise FloatingPointError("full-list Delta-NDCG target必須為有限值")
            if bool((detached_target < 0.0).any().item()) or bool((detached_target > 1.0).any().item()):
                raise ValueError("full-list Delta-NDCG只接受0～1 daily percentile target")
            if not bool(torch.isfinite(detached_margin).all().item()):
                raise FloatingPointError("full-list Delta-NDCG predicted margins必須為有限值")

        item_count = int(detached_target.numel())
        rank_positions = torch.arange(
            1, item_count + 1, dtype=detached_margin.dtype, device=detached_margin.device
        )
        rank_discounts = 1.0 / torch.log2(rank_positions + 1.0)
        predicted_order = torch.argsort(detached_margin, descending=True, stable=True)
        discount_by_item = torch.empty_like(rank_discounts)
        discount_by_item[predicted_order] = rank_discounts
        ideal_order = torch.argsort(detached_target, descending=True, stable=True)
        ideal_dcg = (detached_target.index_select(0, ideal_order) * rank_discounts).sum()
        ideal_dcg_value = float(ideal_dcg.detach().cpu().item())
        if not math.isfinite(ideal_dcg_value) or ideal_dcg_value <= 0.0:
            raise FloatingPointError("full-list Delta-NDCG ideal DCG必須為正有限值")

        discount_delta = torch.abs(
            discount_by_item[:, None] - discount_by_item[None, :]
        )
        # Keep the canonical comparable mask, pair order, margin-difference
        # autograd path and final reductions unchanged. The target/relevance side
        # is detached, so selecting the same comparable entries before the
        # elementwise multiply/divide avoids two unnecessary full N×N float
        # materializations without changing any model result.
        selected_relevance_delta = torch.abs(selected_target_diff.detach())
        selected_discount_delta = discount_delta[comparable]
        weights = selected_relevance_delta * selected_discount_delta / ideal_dcg
        if not bool(torch.isfinite(weights).all().item()):
            raise FloatingPointError("full-list Delta-NDCG pairwise weights必須為有限值")
        pair_losses.append(losses * weights)
        day_losses.append(weights)

    if reduction in {
        CONTINUOUS_RANKER_PAIRWISE_REDUCTION_EQUAL_PAIR,
        CONTINUOUS_RANKER_PAIRWISE_REDUCTION_PARETO_DOMINANCE,
    }:
        if not pair_losses:
            return None, 0
        return torch.cat(pair_losses).mean(), int(pair_count)
    if reduction in {
        CONTINUOUS_RANKER_PAIRWISE_REDUCTION_UPPER_TAIL_RELEVANCE,
        CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
    }:
        if not pair_losses or not day_losses:
            return None, 0
        all_losses = torch.cat(pair_losses)
        all_weights = torch.cat(day_losses)
        weight_sum = all_weights.sum()
        if reduction == CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG:
            weight_sum_value = float(weight_sum.detach().cpu().item())
            valid_weight_sum = math.isfinite(weight_sum_value) and weight_sum_value > 0.0
        else:
            valid_weight_sum = (
                bool(torch.isfinite(weight_sum).item())
                and float(weight_sum.detach().cpu().item()) > 0.0
            )
        if not valid_weight_sum:
            label = (
                "upper-tail relevance"
                if reduction == CONTINUOUS_RANKER_PAIRWISE_REDUCTION_UPPER_TAIL_RELEVANCE
                else "full-list Delta-NDCG"
            )
            raise FloatingPointError(f"{label} pairwise weight sum必須為正有限值")
        return all_losses.sum() / weight_sum, int(pair_count)
    if not day_losses:
        return None, 0
    return torch.stack(day_losses).mean(), int(ranked_date_count)


def _materialize_feature_batch(
    feature_bank,
    ids: np.ndarray,
    *,
    torch=None,
    pin_memory: bool = False,
):
    features = np.asarray(feature_bank[np.asarray(ids, dtype=np.int64)], dtype=np.float32)
    if not bool(pin_memory):
        return features
    if torch is None:
        raise ValueError("pin_memory=True時必須提供torch")
    return torch.from_numpy(features).pin_memory()


def _iter_materialized_feature_batches(
    feature_bank,
    batches: list[np.ndarray],
    *,
    prefetch_batches: int,
    prefetch_workers: int = 1,
    torch=None,
    pin_memory: bool = False,
):
    """Materialize training batches concurrently while preserving exact batch order."""

    depth = int(prefetch_batches)
    worker_count = max(1, int(prefetch_workers))
    if depth <= 0 or len(batches) <= 1:
        for ids in batches:
            yield ids, _materialize_feature_batch(
                feature_bank, ids, torch=torch, pin_memory=bool(pin_memory)
            )
        return

    queue_depth = min(depth, len(batches))
    with ThreadPoolExecutor(
        max_workers=min(worker_count, queue_depth),
        thread_name_prefix="continuous-ranker-prefetch",
    ) as executor:
        pending = deque(
            (
                batches[index],
                executor.submit(
                    _materialize_feature_batch,
                    feature_bank,
                    batches[index],
                    torch=torch,
                    pin_memory=bool(pin_memory),
                ),
            )
            for index in range(queue_depth)
        )
        next_index = queue_depth
        while pending:
            ids, future = pending.popleft()
            features = future.result()
            if next_index < len(batches):
                next_ids = batches[next_index]
                pending.append(
                    (
                        next_ids,
                        executor.submit(
                            _materialize_feature_batch,
                            feature_bank,
                            next_ids,
                            torch=torch,
                            pin_memory=bool(pin_memory),
                        ),
                    )
                )
                next_index += 1
            yield ids, features


def _iter_device_training_batches(
    torch,
    feature_bank,
    group_context: np.ndarray,
    percentile_target: np.ndarray,
    batches: list[np.ndarray],
    *,
    plan,
    prefetch_batches: int,
    prefetch_workers: int,
):
    """Yield device-ready batches; CUDA overlaps pinned H2D for N+1 with compute of N.

    Batch identity and consumption order are unchanged.  Only feature materialization,
    pinned host staging, and host-to-device transfer are pipelined.
    """

    use_cuda = str(getattr(plan, "device_type", "")) == "cuda"
    cpu_iter = iter(
        _iter_materialized_feature_batches(
            feature_bank,
            batches,
            prefetch_batches=int(prefetch_batches),
            prefetch_workers=int(prefetch_workers),
            torch=torch if use_cuda else None,
            pin_memory=use_cuda,
        )
    )
    if not use_cuda:
        for ids, batch_features in cpu_iter:
            xb = torch.from_numpy(batch_features).to(plan.device)
            cb = torch.from_numpy(
                np.asarray(group_context[ids], dtype=np.float32)
            ).to(plan.device)
            target = torch.from_numpy(
                np.asarray(percentile_target[ids], dtype=np.float32)
            ).to(plan.device)
            yield ids, xb, cb, target
        return

    copy_stream = torch.cuda.Stream(device=plan.device)
    compute_stream = torch.cuda.current_stream(device=plan.device)

    def stage_next():
        try:
            ids, features_cpu = next(cpu_iter)
        except StopIteration:
            return None
        if not isinstance(features_cpu, torch.Tensor):
            features_cpu = torch.from_numpy(features_cpu).pin_memory()
        context_cpu = torch.from_numpy(
            np.asarray(group_context[ids], dtype=np.float32)
        ).pin_memory()
        target_cpu = torch.from_numpy(
            np.asarray(percentile_target[ids], dtype=np.float32)
        ).pin_memory()
        with torch.cuda.stream(copy_stream):
            xb = features_cpu.to(plan.device, non_blocking=True)
            cb = context_cpu.to(plan.device, non_blocking=True)
            target = target_cpu.to(plan.device, non_blocking=True)
        # Keep pinned host buffers alive until the copy stream is synchronized by the
        # consumer on the next iteration.
        return ids, xb, cb, target, (features_cpu, context_cpu, target_cpu)

    staged = stage_next()
    while staged is not None:
        compute_stream.wait_stream(copy_stream)
        ids, xb, cb, target, _host_refs = staged
        xb.record_stream(compute_stream)
        cb.record_stream(compute_stream)
        target.record_stream(compute_stream)
        # Stage N+1 before yielding N so the dedicated copy stream can overlap with
        # the forward/backward compute performed by the consumer.
        staged = stage_next()
        yield ids, xb, cb, target


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
    prefetch_batches: int = 0,
    prefetch_workers: int = 1,
    pairwise_reduction: str = CONTINUOUS_RANKER_PAIRWISE_REDUCTION_EQUAL_PAIR,
    raw_r_loss_name: str | None = None,
    raw_r_huber_delta_r: float | None = None,
) -> float:
    model.train()
    ids_all = np.asarray(group_ids, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    if training_objective in {
        TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION,
        TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION,
        TRAINING_OBJECTIVE_DAILY_DUAL_COMPONENT_R_REGRESSION,
    }:
        order = ids_all.copy()
        rng.shuffle(order)
        batches = [
            order[start:start + int(batch_size)]
            for start in range(0, len(order), int(batch_size))
        ]
    elif training_objective in {
        TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        TRAINING_OBJECTIVE_DAILY_PARETO_PAIRWISE_RANKING,
        TRAINING_OBJECTIVE_DAILY_LISTWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_SAFETY_PAIRWISE_RANKING,
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
    group_dates_series = pd.Series(group_dates)

    for ids, xb, cb, target in _iter_device_training_batches(
        torch,
        feature_bank,
        group_context,
        percentile_target,
        batches,
        plan=plan,
        prefetch_batches=int(prefetch_batches),
        prefetch_workers=int(prefetch_workers),
    ):
        if len(ids) == 0:
            continue
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(torch, plan):
            if training_objective == TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_SAFETY_PAIRWISE_RANKING:
                if target.ndim != 2 or int(target.shape[1]) != 2:
                    raise ValueError("conditional MFE-safety target必須為[N,2]")
                if not hasattr(model, "forward_conditional_heads"):
                    raise ValueError("conditional MFE-safety objective需要dual-head model architecture")
                primary_logits, conditional_logits = model.forward_conditional_heads(xb, cb)
                primary_margin = primary_logits.float()[:, LABEL_PASS] - primary_logits.float()[:, LABEL_REJECT]
                conditional_margin = conditional_logits.float()[:, LABEL_PASS] - conditional_logits.float()[:, LABEL_REJECT]
                batch_dates = group_dates_series.iloc[ids].to_numpy()
                primary_loss, primary_supervision = _pairwise_logistic_loss(
                    torch,
                    primary_margin,
                    target[:, 0],
                    batch_dates,
                    reduction=str(pairwise_reduction),
                )
                conditional_loss, conditional_supervision = _pairwise_logistic_loss(
                    torch,
                    conditional_margin,
                    target[:, 1],
                    batch_dates,
                    reduction=str(pairwise_reduction),
                )
                if primary_loss is None or conditional_loss is None:
                    continue
                # Both targets are canonical same-date percentiles on the same
                # scale. Equal head mean is fixed by contract; there is no lambda.
                loss = 0.5 * (primary_loss + conditional_loss)
                loss_weight = int(max(1, primary_supervision + conditional_supervision))
            else:
                logits = model(xb, cb)
            if training_objective == TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_SAFETY_PAIRWISE_RANKING:
                pass
            elif training_objective == TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION:
                score = torch.softmax(logits.float(), dim=1)[:, LABEL_PASS]
                loss = F.mse_loss(score, target, reduction="mean")
                loss_weight = int(len(ids))
            elif training_objective == TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION:
                predicted_r = logits.float()[:, LABEL_PASS] - logits.float()[:, LABEL_REJECT]
                loss_name = str(raw_r_loss_name or ("huber_raw_r" if raw_r_huber_delta_r is not None else "")).strip()
                if loss_name == "mse_raw_r":
                    loss = F.mse_loss(predicted_r, target, reduction="mean")
                elif loss_name == "huber_raw_r":
                    if raw_r_huber_delta_r is None:
                        raise ValueError("Huber direct R regression缺少Huber delta")
                    loss = F.huber_loss(
                        predicted_r, target, reduction="mean", delta=float(raw_r_huber_delta_r)
                    )
                else:
                    raise ValueError(f"不支援的direct R regression loss: {loss_name!r}")
                loss_weight = int(len(ids))
            elif training_objective == TRAINING_OBJECTIVE_DAILY_DUAL_COMPONENT_R_REGRESSION:
                if target.ndim != 2 or int(target.shape[1]) != 2:
                    raise ValueError("dual-component R regression target必須為[N,2]")
                # Existing two output neurons are reinterpreted only by this profile:
                # LABEL_REJECT -> adverse-to-peak R, LABEL_PASS -> favorable MFE R.
                # F.mse_loss(mean) gives both physical R components equal weight
                # without an extra lambda or auxiliary-loss coefficient.
                loss = F.mse_loss(logits.float(), target, reduction="mean")
                loss_weight = int(len(ids))
            elif training_objective in {
                TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
                TRAINING_OBJECTIVE_DAILY_PARETO_PAIRWISE_RANKING,
            }:
                margins = logits.float()[:, LABEL_PASS] - logits.float()[:, LABEL_REJECT]
                loss, supervision_count = _pairwise_logistic_loss(
                    torch,
                    margins,
                    target,
                    group_dates_series.iloc[ids].to_numpy(),
                    reduction=str(pairwise_reduction),
                )
                if loss is None:
                    continue
                loss_weight = int(supervision_count)
            else:
                margins = logits.float()[:, LABEL_PASS] - logits.float()[:, LABEL_REJECT]
                loss, ranked_date_count = _listnet_top_one_loss(
                    torch,
                    margins,
                    target,
                    group_dates_series.iloc[ids].to_numpy(),
                )
                if loss is None:
                    continue
                loss_weight = int(ranked_date_count)
        reuse_loss_scalar = bool(
            training_objective
            in {
                TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
                TRAINING_OBJECTIVE_DAILY_PARETO_PAIRWISE_RANKING,
                TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_SAFETY_PAIRWISE_RANKING,
            }
            and str(pairwise_reduction)
            == CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG
        )
        loss_scalar_value = None
        if reuse_loss_scalar:
            # Full-list Delta-NDCG already needs a host barrier here for the
            # canonical finite-loss guard. Reuse the same scalar for reporting
            # after optimizer.step instead of synchronizing a second time.
            loss_scalar_value = float(loss.detach().cpu().item())
            if not math.isfinite(loss_scalar_value):
                raise FloatingPointError("continuous ranker training loss非有限值")
        elif not bool(torch.isfinite(loss).item()):
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
        value = (
            float(loss_scalar_value)
            if loss_scalar_value is not None
            else float(loss.detach().cpu().item())
        )
        losses.append(value)
        weighted_loss_sum += value * float(loss_weight)
        weighted_loss_count += int(loss_weight)
    if not losses or weighted_loss_count < 1:
        raise ValueError("continuous ranker training沒有任何有效batch／ranking supervision")
    if training_objective in {
        TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION,
        TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION,
        TRAINING_OBJECTIVE_DAILY_DUAL_COMPONENT_R_REGRESSION,
        TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_SAFETY_PAIRWISE_RANKING,
    }:
        # Preserve historical scalar-regression reporting semantics exactly.
        # Conditional dual-head training reports the actual equal-head mean per
        # optimizer batch instead of inventing a cross-head pair-count weight.
        return float(np.mean(losses))
    return float(weighted_loss_sum / float(weighted_loss_count))

def predict_scores(
    torch,
    model,
    feature_bank: np.ndarray,
    group_context: np.ndarray,
    group_ids: np.ndarray,
    *,
    batch_size: int,
    plan,
    training_objective: str,
) -> np.ndarray:
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
    if training_objective in {
        TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION,
        TRAINING_OBJECTIVE_DAILY_DUAL_COMPONENT_R_REGRESSION,
    }:
        return (logits[:, LABEL_PASS] - logits[:, LABEL_REJECT]).astype(np.float32)
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return (exp[:, LABEL_PASS] / exp.sum(axis=1)).astype(np.float32)


def build_conditional_targets(
    group_table: pd.DataFrame,
    percentile_target: np.ndarray,
) -> ConditionalMfeSafetyTargets:
    """Return aligned MR-13P supervision for every currently-labeled row."""

    primary = np.asarray(percentile_target, dtype=np.float32)
    if primary.ndim != 1 or len(primary) != len(group_table):
        raise ValueError("conditional MFE-safety percentile target shape不一致")
    valid = np.isfinite(primary)
    return build_conditional_mfe_safety_targets(
        group_table,
        valid,
        primary_mfe_percentile=primary,
    )


def predict_conditional_mfe_safety_scores(
    torch,
    model,
    feature_bank: np.ndarray,
    group_context: np.ndarray,
    group_ids: np.ndarray,
    *,
    batch_size: int,
    plan,
) -> dict[str, np.ndarray]:
    """Return both MR-13P head probabilities from one shared-encoder inference pass."""

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
        output_head="conditional_both",
    ).astype(np.float32)
    if logits.ndim != 2 or int(logits.shape[1]) != 4:
        raise ValueError("conditional MFE-safety model output必須為[N,4]")

    def pass_probability(pair: np.ndarray) -> np.ndarray:
        shifted = pair.astype(np.float64) - pair.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return (exp[:, LABEL_PASS] / exp.sum(axis=1)).astype(np.float32)

    return {
        "primary_mfe": pass_probability(logits[:, :2]),
        "conditional_safety": pass_probability(logits[:, 2:]),
    }


def conditional_mfe_safety_metrics(
    group_ids: np.ndarray,
    group_table: pd.DataFrame,
    raw_mfe_target: np.ndarray,
    targets: ConditionalMfeSafetyTargets,
    scores: dict[str, np.ndarray],
    *,
    include_top_k_quality: bool = False,
) -> dict[str, Any]:
    """Return explicit primary-preservation and conditional-safety model gates."""

    ids = np.asarray(group_ids, dtype=np.int64)
    primary_scores = np.asarray(scores["primary_mfe"], dtype=np.float32)
    conditional_scores = np.asarray(scores["conditional_safety"], dtype=np.float32)
    if len(primary_scores) != len(ids) or len(conditional_scores) != len(ids):
        raise ValueError("conditional MFE-safety score/group長度不一致")
    return {
        "primary_mfe": split_metrics(
            ids,
            group_table,
            np.asarray(raw_mfe_target, dtype=np.float32),
            targets.primary_mfe_percentile,
            primary_scores,
            include_top_k_quality=bool(include_top_k_quality),
        ),
        "conditional_safety": split_metrics(
            ids,
            group_table,
            targets.conditional_safety_residual,
            targets.conditional_safety_percentile,
            conditional_scores,
            include_top_k_quality=bool(include_top_k_quality),
        ),
    }


def predict_dual_component_r(
    torch,
    model,
    feature_bank: np.ndarray,
    group_context: np.ndarray,
    group_ids: np.ndarray,
    *,
    batch_size: int,
    plan,
) -> dict[str, np.ndarray]:
    """Return primary MR-13L component predictions in canonical R units."""

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
    ).astype(np.float32)
    if logits.ndim != 2 or int(logits.shape[1]) != 2:
        raise ValueError("dual-component model output必須為[N,2]")
    adverse_r = logits[:, LABEL_REJECT].astype(np.float32, copy=False)
    favorable_r = logits[:, LABEL_PASS].astype(np.float32, copy=False)
    return {
        "predicted_favorable_r": favorable_r,
        "predicted_adverse_r": adverse_r,
        "model_score": (favorable_r - adverse_r).astype(np.float32, copy=False),
    }


def build_pareto_component_percentile_targets(
    group_table: pd.DataFrame,
    raw_target: np.ndarray,
) -> np.ndarray:
    """Return MR-13O [MFE percentile, low-adverse percentile] supervision targets."""

    required = ("target_favorable_r", "target_adverse_r", "date")
    missing = [column for column in required if column not in group_table.columns]
    if missing:
        raise ValueError(f"Pareto pairwise缺少canonical component columns: {missing}")
    raw = np.asarray(raw_target, dtype=np.float32)
    valid = np.isfinite(raw)
    favorable = pd.to_numeric(
        group_table["target_favorable_r"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    adverse = pd.to_numeric(
        group_table["target_adverse_r"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    component_valid = valid & np.isfinite(favorable) & np.isfinite(adverse)
    if not np.array_equal(component_valid, valid):
        raise ValueError("Pareto component-valid universe與economic target-valid universe不一致")
    mfe_percentile = build_same_date_percentile_targets(
        favorable, valid, group_table["date"]
    )
    low_adverse_percentile = build_same_date_percentile_targets(
        -adverse, valid, group_table["date"]
    )
    targets = np.column_stack([mfe_percentile, low_adverse_percentile]).astype(
        np.float32, copy=False
    )
    if bool(np.any(valid & ~np.isfinite(targets).all(axis=1))):
        raise ValueError("Pareto component percentile target產生non-finite value")
    return targets


def pareto_pair_concordance_metrics(
    group_ids: np.ndarray,
    group_table: pd.DataFrame,
    raw_target: np.ndarray,
    scores: np.ndarray,
) -> dict[str, Any]:
    """Measure score ordering only on strict same-date Pareto-comparable pairs."""

    ids = np.asarray(group_ids, dtype=np.int64)
    score = np.asarray(scores, dtype=np.float64)
    if len(ids) != len(score):
        raise ValueError("Pareto metric group_ids與scores長度不一致")
    if len(ids) == 0:
        return {
            "group_count": 0,
            "comparable_pair_count": 0,
            "all_pair_count": 0,
            "comparable_pair_rate": None,
            "rankable_date_count": 0,
            "mean_daily_pareto_pair_concordance": None,
            "global_pareto_pair_concordance": None,
        }
    component = build_pareto_component_percentile_targets(group_table, raw_target)[ids]
    dates = pd.to_datetime(group_table.iloc[ids]["date"], errors="raise").dt.normalize().to_numpy()
    valid = np.isfinite(component).all(axis=1) & np.isfinite(score)
    component = component[valid]
    score = score[valid]
    dates = dates[valid]
    daily_values: list[float] = []
    correct_sum = 0.0
    comparable_total = 0
    all_pair_total = 0
    for date_value in pd.unique(dates):
        positions = np.flatnonzero(dates == date_value)
        n = int(len(positions))
        if n < 2:
            continue
        all_pair_total += n * (n - 1) // 2
        target = component[positions]
        day_score = score[positions]
        diff = target[:, None, :] - target[None, :, :]
        upper = np.triu(np.ones((n, n), dtype=bool), k=1)
        positive = (diff[:, :, 0] > 0.0) & (diff[:, :, 1] > 0.0)
        negative = (diff[:, :, 0] < 0.0) & (diff[:, :, 1] < 0.0)
        comparable = upper & (positive | negative)
        count = int(np.count_nonzero(comparable))
        if count == 0:
            continue
        score_diff = day_score[:, None] - day_score[None, :]
        signs = np.where(positive[comparable], 1.0, -1.0)
        ordered = signs * score_diff[comparable]
        correct = np.where(ordered > 0.0, 1.0, np.where(ordered < 0.0, 0.0, 0.5))
        day_value = float(np.mean(correct))
        daily_values.append(day_value)
        correct_sum += float(np.sum(correct))
        comparable_total += count
    return {
        "group_count": int(np.count_nonzero(valid)),
        "comparable_pair_count": int(comparable_total),
        "all_pair_count": int(all_pair_total),
        "comparable_pair_rate": (
            None if all_pair_total < 1 else float(comparable_total / all_pair_total)
        ),
        "rankable_date_count": int(len(daily_values)),
        "mean_daily_pareto_pair_concordance": (
            None if not daily_values else float(np.mean(daily_values))
        ),
        "global_pareto_pair_concordance": (
            None if comparable_total < 1 else float(correct_sum / comparable_total)
        ),
    }


def _training_target_for_profile(
    profile,
    raw_target: np.ndarray,
    percentile_target: np.ndarray,
    group_table: pd.DataFrame,
) -> np.ndarray:
    if profile.training_objective == TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_SAFETY_PAIRWISE_RANKING:
        return build_conditional_targets(group_table, percentile_target).training_target
    if profile.training_objective == TRAINING_OBJECTIVE_DAILY_PARETO_PAIRWISE_RANKING:
        return build_pareto_component_percentile_targets(group_table, raw_target)
    if profile.training_objective == TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION:
        return np.asarray(raw_target, dtype=np.float32)
    if profile.training_objective == TRAINING_OBJECTIVE_DAILY_DUAL_COMPONENT_R_REGRESSION:
        required = ("target_adverse_r", "target_favorable_r")
        missing = [column for column in required if column not in group_table.columns]
        if missing:
            raise ValueError(
                "dual-component R regression缺少canonical component target columns: "
                f"{missing}"
            )
        adverse = pd.to_numeric(group_table["target_adverse_r"], errors="coerce").to_numpy(dtype=np.float32)
        favorable = pd.to_numeric(group_table["target_favorable_r"], errors="coerce").to_numpy(dtype=np.float32)
        targets = np.column_stack([adverse, favorable]).astype(np.float32, copy=False)
        valid = np.isfinite(targets).all(axis=1)
        raw_valid = np.isfinite(np.asarray(raw_target, dtype=np.float32))
        if not np.array_equal(valid, raw_valid):
            raise ValueError(
                "dual-component target-valid universe與composite raw target不一致"
            )
        return targets
    return np.asarray(percentile_target, dtype=np.float32)


def select_epoch(
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
    evaluate_train_metrics: bool = True,
) -> dict[str, Any]:
    profile = get_breakout_quality_experiment_profile(args.experiment_profile)
    research_spec = get_continuous_ranker_research_spec(str(args.experiment_profile))
    execution_recipe = get_continuous_ranker_execution_recipe(str(args.experiment_profile))
    conditional_targets = (
        build_conditional_targets(group_table, percentile_target)
        if profile.training_objective == TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_SAFETY_PAIRWISE_RANKING
        else None
    )
    training_target = (
        conditional_targets.training_target
        if conditional_targets is not None
        else _training_target_for_profile(profile, raw_target, percentile_target, group_table)
    )
    raw_r_loss_name = (
        str(profile.loss_name)
        if profile.training_objective == TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION
        else None
    )
    raw_r_delta = (
        float(profile.raw_r_huber_delta_r)
        if profile.training_objective == TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION
        and profile.raw_r_huber_delta_r is not None
        else None
    )
    model, optimizer = _new_model_and_optimizer(
        torch,
        feature_count=int(feature_bank.shape[2]),
        context_count=int(group_context.shape[1]),
        args=args,
        plan=plan,
    )
    grad_scaler = build_grad_scaler(torch, plan)
    best_epoch = 0
    best_daily_spearman = -math.inf
    best_primary_mfe_daily_spearman = -math.inf
    best_validation_mse = math.inf
    best_validation_huber = math.inf
    best_validation_mse_raw_r = math.inf
    best_validation_mae = math.inf
    best_pareto_concordance = -math.inf
    best_global_pareto_concordance = -math.inf
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    compact_console = compact_console_enabled()
    if not compact_console:
        label = (
            "Validation Huber raw R"
            if raw_r_loss_name == "huber_raw_r"
            else "Validation MSE raw R"
            if raw_r_loss_name == "mse_raw_r"
            else "Validation mean daily Pareto pair concordance"
            if profile.training_objective == TRAINING_OBJECTIVE_DAILY_PARETO_PAIRWISE_RANKING
            else "Validation conditional-safety mean daily Spearman"
            if profile.training_objective == TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_SAFETY_PAIRWISE_RANKING
            else "Validation mean daily Spearman"
        )
        print(f"\nEpoch選擇（依{label}）")
    for epoch in range(1, int(args.epochs) + 1):
        _emit_epoch_progress_marker("select", epoch, int(args.epochs))
        started = time.perf_counter()
        batch_loss = _train_epoch(
            torch,
            model,
            optimizer,
            feature_bank,
            group_context,
            train_ids,
            training_target,
            group_table["date"],
            training_objective=profile.training_objective,
            batch_size=int(args.batch_size),
            seed=int(args.seed) + epoch,
            gradient_clip_norm=float(args.gradient_clip_norm),
            plan=plan,
            grad_scaler=grad_scaler,
            prefetch_batches=int(args.train_prefetch_batches),
            prefetch_workers=int(getattr(args, "train_prefetch_workers", BREAKOUT_QUALITY_CONTINUOUS_RANKER_PREFETCH_WORKERS)),
            pairwise_reduction=(
                execution_recipe.pairwise_reduction
                or CONTINUOUS_RANKER_PAIRWISE_REDUCTION_EQUAL_PAIR
            ),
            raw_r_loss_name=raw_r_loss_name,
            raw_r_huber_delta_r=raw_r_delta,
        )
        validation_conditional_metrics = None
        if conditional_targets is not None:
            validation_head_scores = predict_conditional_mfe_safety_scores(
                torch,
                model,
                feature_bank,
                group_context,
                validation_ids,
                batch_size=int(args.evaluation_batch_size),
                plan=plan,
            )
            validation_conditional_metrics = conditional_mfe_safety_metrics(
                validation_ids,
                group_table,
                raw_target,
                conditional_targets,
                validation_head_scores,
            )
            validation_scores = validation_head_scores["primary_mfe"]
            validation_metrics = validation_conditional_metrics["primary_mfe"]
        else:
            validation_scores = predict_scores(
                torch,
                model,
                feature_bank,
                group_context,
                validation_ids,
                batch_size=int(args.evaluation_batch_size),
                plan=plan,
                training_objective=profile.training_objective,
            )
            validation_metrics = split_metrics(
                validation_ids,
                group_table,
                raw_target,
                percentile_target,
                validation_scores,
                raw_r_regression_loss_name=raw_r_loss_name,
                raw_r_huber_delta_r=raw_r_delta,
            )
        if bool(evaluate_train_metrics):
            if conditional_targets is not None:
                train_head_scores = predict_conditional_mfe_safety_scores(
                    torch, model, feature_bank, group_context, train_ids,
                    batch_size=int(args.evaluation_batch_size), plan=plan,
                )
                train_conditional_metrics = conditional_mfe_safety_metrics(
                    train_ids, group_table, raw_target, conditional_targets, train_head_scores
                )
                train_metrics = train_conditional_metrics["primary_mfe"]
            else:
                train_scores = predict_scores(
                    torch,
                    model,
                    feature_bank,
                    group_context,
                    train_ids,
                    batch_size=int(args.evaluation_batch_size),
                    plan=plan,
                    training_objective=profile.training_objective,
                )
                train_metrics = split_metrics(
                    train_ids,
                    group_table,
                    raw_target,
                    percentile_target,
                    train_scores,
                    raw_r_regression_loss_name=raw_r_loss_name,
                    raw_r_huber_delta_r=raw_r_delta,
                )
        else:
            train_metrics = {
                "group_count": int(len(train_ids)),
                "not_evaluated_reason": (
                    "daily-universal sample universe略過每epoch完整train inference；"
                    "epoch selection仍只依完整validation metric"
                ),
            }
        if validation_conditional_metrics is not None:
            primary_daily_spearman = validation_metrics.get("mean_daily_spearman")
            daily_spearman = validation_conditional_metrics["conditional_safety"].get("mean_daily_spearman")
            validation_mse = float(
                validation_conditional_metrics["conditional_safety"]["mse_vs_daily_percentile"]
            )
        else:
            primary_daily_spearman = validation_metrics.get("mean_daily_spearman")
            daily_spearman = primary_daily_spearman
            validation_mse = float(validation_metrics["mse_vs_daily_percentile"])
        if daily_spearman is None or not math.isfinite(float(daily_spearman)):
            raise ValueError("continuous model validation mean daily Spearman不可用")
        validation_pareto_metrics = None
        if profile.training_objective == TRAINING_OBJECTIVE_DAILY_PARETO_PAIRWISE_RANKING:
            validation_pareto_metrics = pareto_pair_concordance_metrics(
                validation_ids, group_table, raw_target, validation_scores
            )
            pareto_daily = validation_pareto_metrics.get(
                "mean_daily_pareto_pair_concordance"
            )
            pareto_global = validation_pareto_metrics.get(
                "global_pareto_pair_concordance"
            )
            if pareto_daily is None or not math.isfinite(float(pareto_daily)):
                raise ValueError("Pareto validation mean daily pair concordance不可用")
            if pareto_global is None or not math.isfinite(float(pareto_global)):
                raise ValueError("Pareto validation global pair concordance不可用")

        if profile.training_objective == TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION:
            regression = dict(validation_metrics.get("raw_r_regression") or {})
            validation_huber = regression.get("huber_loss_raw_r")
            validation_mse_raw_r = regression.get("mse_raw_r")
            validation_mae = regression.get("mae_raw_r")
            if validation_mse_raw_r is None or not math.isfinite(float(validation_mse_raw_r)):
                raise ValueError("direct R validation MSE不可用")
            if validation_mae is None or not math.isfinite(float(validation_mae)):
                raise ValueError("direct R validation MAE不可用")
            if raw_r_loss_name == "huber_raw_r":
                if validation_huber is None or not math.isfinite(float(validation_huber)):
                    raise ValueError("direct R validation Huber loss不可用")
                primary_metric = float(validation_huber)
                best_primary_metric = best_validation_huber
            elif raw_r_loss_name == "mse_raw_r":
                primary_metric = float(validation_mse_raw_r)
                best_primary_metric = best_validation_mse_raw_r
            else:
                raise ValueError(f"不支援的direct R epoch-selection loss: {raw_r_loss_name!r}")
            improved = (
                primary_metric < best_primary_metric - float(args.early_stopping_min_delta)
                or (
                    math.isclose(primary_metric, best_primary_metric, rel_tol=0.0, abs_tol=1e-12)
                    and float(daily_spearman) > best_daily_spearman
                )
            )
        else:
            validation_huber = None
            validation_mse_raw_r = None
            validation_mae = None
            if profile.training_objective == TRAINING_OBJECTIVE_DAILY_PARETO_PAIRWISE_RANKING:
                pareto_daily = float(
                    validation_pareto_metrics["mean_daily_pareto_pair_concordance"]
                )
                pareto_global = float(
                    validation_pareto_metrics["global_pareto_pair_concordance"]
                )
                improved = (
                    pareto_daily
                    > best_pareto_concordance + float(args.early_stopping_min_delta)
                    or (
                        math.isclose(
                            pareto_daily, best_pareto_concordance, rel_tol=0.0, abs_tol=1e-12
                        )
                        and pareto_global > best_global_pareto_concordance
                    )
                )
            else:
                improved = (
                    float(daily_spearman) > best_daily_spearman + float(args.early_stopping_min_delta)
                    or (
                        math.isclose(float(daily_spearman), best_daily_spearman, rel_tol=0.0, abs_tol=1e-12)
                        and validation_mse < best_validation_mse
                    )
                )

        if improved:
            best_epoch = int(epoch)
            best_daily_spearman = float(daily_spearman)
            if primary_daily_spearman is not None and math.isfinite(float(primary_daily_spearman)):
                best_primary_mfe_daily_spearman = float(primary_daily_spearman)
            best_validation_mse = validation_mse
            if validation_pareto_metrics is not None:
                best_pareto_concordance = float(
                    validation_pareto_metrics["mean_daily_pareto_pair_concordance"]
                )
                best_global_pareto_concordance = float(
                    validation_pareto_metrics["global_pareto_pair_concordance"]
                )
            if validation_mse_raw_r is not None:
                best_validation_mse_raw_r = float(validation_mse_raw_r)
                best_validation_mae = float(validation_mae)
            if validation_huber is not None:
                best_validation_huber = float(validation_huber)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        elapsed = time.perf_counter() - started
        history.append({
            "epoch": int(epoch),
            "batch_loss": float(batch_loss),
            "inner_train_metrics": train_metrics,
            "inner_validation_metrics": validation_metrics,
            "inner_validation_conditional_mfe_safety_metrics": validation_conditional_metrics,
            "inner_validation_pareto_metrics": validation_pareto_metrics,
            "elapsed_sec": round(float(elapsed), 3),
            "is_best_epoch": bool(improved),
        })
        if not compact_console:
            marker = " ★新最佳" if improved else ""
            if profile.training_objective == TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION:
                if raw_r_loss_name == "huber_raw_r":
                    loss_text = f"Train Huber {batch_loss:.6f} | Val Huber {float(validation_huber):.6f}"
                else:
                    loss_text = f"Train MSE {batch_loss:.6f} | Val MSE {float(validation_mse_raw_r):.6f}"
                print(
                    f"  Epoch {epoch:>3}/{int(args.epochs)} | {loss_text} | Val MAE {float(validation_mae):.4f}R "
                    f"| Val Daily Spearman {float(daily_spearman):.4f} | {elapsed:.1f}s{marker}"
                )
            elif profile.training_objective == TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_SAFETY_PAIRWISE_RANKING:
                print(
                    f"  Epoch {epoch:>3}/{int(args.epochs)} | Train Loss {batch_loss:.6f} "
                    f"| Val Conditional rho {float(daily_spearman):.4f} "
                    f"| Val MFE rho {float(primary_daily_spearman):.4f} "
                    f"| {elapsed:.1f}s{marker}"
                )
            elif profile.training_objective == TRAINING_OBJECTIVE_DAILY_PARETO_PAIRWISE_RANKING:
                print(
                    f"  Epoch {epoch:>3}/{int(args.epochs)} | Train Loss {batch_loss:.6f} "
                    f"| Val Pareto {float(validation_pareto_metrics['mean_daily_pareto_pair_concordance']):.4f} "
                    f"| Val Economic rho {float(daily_spearman):.4f} "
                    f"| {elapsed:.1f}s{marker}"
                )
            elif bool(evaluate_train_metrics):
                print(
                    f"  Epoch {epoch:>3}/{int(args.epochs)} | Train Loss {batch_loss:.6f} | Train MSE {train_metrics['mse_vs_daily_percentile']:.6f} "
                    f"| Val MSE {validation_mse:.6f} | Val Daily Spearman {float(daily_spearman):.4f} "
                    f"| {elapsed:.1f}s{marker}"
                )
            else:
                print(
                    f"  Epoch {epoch:>3}/{int(args.epochs)} | Train Loss {batch_loss:.6f} "
                    f"| Val MSE {validation_mse:.6f} | Val Daily Spearman {float(daily_spearman):.4f} "
                    f"| {elapsed:.1f}s{marker}"
                )
        if int(args.early_stopping_patience) > 0 and epochs_without_improvement >= int(args.early_stopping_patience):
            break
    if best_epoch < 1:
        raise ValueError("continuous model無法選出best epoch")
    return {
        "best_epoch": int(best_epoch),
        "best_validation_mean_daily_spearman": float(best_daily_spearman),
        "best_validation_conditional_safety_mean_daily_spearman": (
            float(best_daily_spearman)
            if profile.training_objective == TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_SAFETY_PAIRWISE_RANKING
            else None
        ),
        "best_validation_primary_mfe_mean_daily_spearman": (
            None
            if not math.isfinite(best_primary_mfe_daily_spearman)
            else float(best_primary_mfe_daily_spearman)
        ),
        "best_validation_mean_daily_pareto_pair_concordance": (
            None if not math.isfinite(best_pareto_concordance) else float(best_pareto_concordance)
        ),
        "best_validation_global_pareto_pair_concordance": (
            None if not math.isfinite(best_global_pareto_concordance) else float(best_global_pareto_concordance)
        ),
        "best_validation_mse": float(best_validation_mse),
        "best_validation_huber_raw_r": (
            None if not math.isfinite(best_validation_huber) else float(best_validation_huber)
        ),
        "best_validation_mse_raw_r": (
            None if not math.isfinite(best_validation_mse_raw_r) else float(best_validation_mse_raw_r)
        ),
        "best_validation_mae_raw_r": (
            None if not math.isfinite(best_validation_mae) else float(best_validation_mae)
        ),
        "completed_epochs": int(len(history)),
        "history": history,
    }


def fit_final(
    torch,
    feature_bank: np.ndarray,
    group_context: np.ndarray,
    raw_target: np.ndarray,
    percentile_target: np.ndarray,
    group_table: pd.DataFrame,
    final_ids: np.ndarray,
    *,
    epochs: int,
    args,
    plan,
    phase_label: str = "完整Selection重訓",
):
    profile = get_breakout_quality_experiment_profile(args.experiment_profile)
    research_spec = get_continuous_ranker_research_spec(str(args.experiment_profile))
    execution_recipe = get_continuous_ranker_execution_recipe(str(args.experiment_profile))
    training_target = _training_target_for_profile(profile, raw_target, percentile_target, group_table)
    raw_r_loss_name = (
        str(profile.loss_name)
        if profile.training_objective == TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION
        else None
    )
    raw_r_delta = (
        float(profile.raw_r_huber_delta_r)
        if profile.training_objective == TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION
        and profile.raw_r_huber_delta_r is not None
        else None
    )
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
        _emit_epoch_progress_marker("refit", epoch, int(epochs))
        started = time.perf_counter()
        loss = _train_epoch(
            torch,
            model,
            optimizer,
            feature_bank,
            group_context,
            final_ids,
            training_target,
            group_table["date"],
            training_objective=profile.training_objective,
            batch_size=int(args.batch_size),
            seed=int(args.seed) + epoch,
            gradient_clip_norm=float(args.gradient_clip_norm),
            plan=plan,
            grad_scaler=grad_scaler,
            prefetch_batches=int(args.train_prefetch_batches),
            prefetch_workers=int(getattr(args, "train_prefetch_workers", BREAKOUT_QUALITY_CONTINUOUS_RANKER_PREFETCH_WORKERS)),
            pairwise_reduction=(
                execution_recipe.pairwise_reduction
                or CONTINUOUS_RANKER_PAIRWISE_REDUCTION_EQUAL_PAIR
            ),
            raw_r_loss_name=raw_r_loss_name,
            raw_r_huber_delta_r=raw_r_delta,
        )
        elapsed = time.perf_counter() - started
        history.append({"epoch": int(epoch), "batch_loss": float(loss), "elapsed_sec": round(float(elapsed), 3)})
        if not compact_console:
            print(f"  Epoch {epoch:>3}/{int(epochs)} | Batch Loss {loss:.6f} | {elapsed:.1f}s")
    return model.eval(), history


# Legacy private-name compatibility. Production modules must use the public
# ranker training API; these aliases remain only for historical tests/audits.
_training_semantics = training_semantics
_training_output_paths = resolve_training_output_paths
_spearman = calculate_spearman
_daily_rank_metrics = daily_rank_metrics
_daily_top_k_metrics = daily_top_k_metrics
_split_metrics = split_metrics
_predict_scores = predict_scores
_select_epoch = select_epoch
_fit_final = fit_final


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
        "spearman_model_score_vs_r_multiple": calculate_spearman(scores, r_values),
        "spearman_model_score_vs_target": calculate_spearman(scores, targets),
        "spearman_target_vs_r_multiple": calculate_spearman(targets, r_values),
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


def run(args) -> int:
    validate_args(args)
    started = time.perf_counter()
    profile = get_breakout_quality_experiment_profile(args.experiment_profile)
    if get_continuous_ranker_execution_recipe(profile.name).trainer_family == CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL:
        raise ValueError(
            "daily-universal ranker必須由services.breakout_quality.ranker_cli dispatch，"
            "event trainer不得直接承載daily orchestration"
        )
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
        source_data_end_date=source_data_end(summary, events),
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
    forward_score_ids = resolve_forward_oos_score_group_ids(
        SimpleNamespace(group_table=group_table, outer_policy=outer_policy)
    )
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

    epoch_selection = select_epoch(
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
    model, final_history = fit_final(
        torch,
        feature_bank,
        group_context,
        raw_target,
        percentile_target,
        group_table,
        scoped_split_ids["selection"],
        epochs=selected_epoch,
        args=args,
        plan=plan,
    )

    artifact_paths, output_dir = resolve_training_output_paths(args)
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
            "training_semantics": training_semantics(profile),
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
    split_metrics_by_split: dict[str, Any] = {}
    for name, ids in all_split_ids.items():
        scores = predict_scores(
            torch,
            model,
            feature_bank,
            group_context,
            ids,
            batch_size=int(args.evaluation_batch_size),
            plan=plan,
            training_objective=profile.training_objective,
        )
        score_by_group[ids] = scores
        all_group_split_metrics[name] = split_metrics(
            ids,
            group_table,
            raw_target,
            percentile_target,
            scores,
            include_top_k_quality=True,
            raw_r_regression_loss_name=(
                str(profile.loss_name)
                if profile.training_objective == TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION
                else None
            ),
            raw_r_huber_delta_r=(
                float(profile.raw_r_huber_delta_r)
                if profile.training_objective == TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION
                and profile.raw_r_huber_delta_r is not None
                else None
            ),
        )
        scoped_ids = scoped_split_ids[name]
        split_metrics_by_split[name] = split_metrics(
            scoped_ids,
            group_table,
            raw_target,
            percentile_target,
            score_by_group[scoped_ids],
            include_top_k_quality=True,
            raw_r_regression_loss_name=(
                str(profile.loss_name)
                if profile.training_objective == TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION
                else None
            ),
            raw_r_huber_delta_r=(
                float(profile.raw_r_huber_delta_r)
                if profile.training_objective == TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION
                and profile.raw_r_huber_delta_r is not None
                else None
            ),
        )

    # Forward runtime score availability must depend only on prediction-time
    # information.  OOS metrics above intentionally remain restricted to rows
    # whose future target is complete; inference-only tail rows are scored here
    # after the frozen Selection-fit checkpoint exists.
    missing_forward_ids = forward_score_ids[~np.isfinite(score_by_group[forward_score_ids])]
    if len(missing_forward_ids):
        score_by_group[missing_forward_ids] = predict_scores(
            torch,
            model,
            feature_bank,
            group_context,
            missing_forward_ids,
            batch_size=int(args.evaluation_batch_size),
            plan=plan,
            training_objective=profile.training_objective,
        )

    role_by_group = np.full((group_count,), "selection_other", dtype=object)
    role_by_group[all_split_ids["inner_train"]] = "inner_train"
    role_by_group[all_split_ids["validation"]] = "validation"
    role_by_group[forward_score_ids] = "oos"
    score_frames: list[pd.DataFrame] = []
    for name in ("selection", "oos"):
        ids = all_split_ids[name] if name == "selection" else forward_score_ids
        frame = group_table.iloc[ids][["ticker", "date", "group_index", "label"]].copy()
        frame["split"] = name
        frame["selection_role"] = role_by_group[ids]
        frame["in_training_label_scope"] = np.isin(ids, scoped_split_ids[name])
        if name == "oos":
            evaluable_mask = np.isin(ids, all_split_ids["oos"])
            frame["target_raw_r"] = np.nan
            frame["target_daily_percentile"] = np.nan
            frame.loc[evaluable_mask, "target_raw_r"] = raw_target[ids[evaluable_mask]]
            frame.loc[evaluable_mask, "target_daily_percentile"] = percentile_target[
                ids[evaluable_mask]
            ]
        else:
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
            "model_score": (
                "predicted_r_two_logit_margin"
                if profile.training_objective == TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION
                else "softmax_pass_probability"
            ),
            "batching": training_semantics(profile)["batching"],
            "pairwise_contract": training_semantics(profile)["pairwise_contract"],
            "listwise_contract": training_semantics(profile)["listwise_contract"],
            "raw_r_regression_contract": training_semantics(profile).get("raw_r_regression_contract"),
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
        "split_metrics": split_metrics_by_split,
        "all_group_split_metrics": all_group_split_metrics,
        "trade_alignment": trade_alignment,
        "target_manifest": target_manifest,
        "model_information_cutoff": information_cutoff,
        "oos_used_for_training_or_epoch_selection": False,
        "oos_evaluated_after_checkpoint_write": True,
        "score_eligibility_contract": build_score_eligibility_contract(profile),
        "forward_score_coverage": {
            "inference_eligible_groups": int(len(forward_score_ids)),
            "target_evaluable_groups": int(len(all_split_ids["oos"])),
            "future_target_required_for_score": False,
        },
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
        "training_semantics": training_semantics(profile),
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
        "score_eligibility_contract": payload["score_eligibility_contract"],
        "forward_score_coverage": payload["forward_score_coverage"],
        "research_outputs": {
            "scores": build_file_manifest(score_path),
            "daily_percentile_target": build_file_manifest(percentile_path),
            "report_json": build_file_manifest(report_json_path),
            "report_markdown": build_file_manifest(report_markdown_path),
        },
    }
    write_json(artifact_paths.manifest_path, manifest)

    print("\nContinuous ranker完成")
    print(
        f"selected_epoch={selected_epoch} "
        f"epoch_selection_validation_daily_spearman={epoch_selection['best_validation_mean_daily_spearman']:.4f}"
    )
    print("checkpoint後split metrics（原Validation rows已納入完整Selection重訓；以下不再用於選模）")
    for name in ("inner_train", "validation", "selection", "oos"):
        metrics = split_metrics_by_split[name]
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


def main(argv=None) -> int:
    """Compatibility event-ranker CLI; formal profile dispatch lives in ranker_cli."""
    return int(run(parse_args(argv)))


__all__ = [
    "RANKER_SCHEMA_VERSION",
    "build_daily_percentile_targets",
    "build_pareto_component_percentile_targets",
    "pareto_pair_concordance_metrics",
    "main",
    "parse_args",
    "run",
    "validate_args",
]


if __name__ == "__main__":
    raise SystemExit(main())
