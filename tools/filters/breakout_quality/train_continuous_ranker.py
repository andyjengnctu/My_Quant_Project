"""11B research-only daily-percentile ranker using the active InceptionTime model."""

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

from config.breakout_quality_experiments import (
    STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE,
    TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION,
    get_breakout_quality_experiment_profile,
)
from config.breakout_quality_policy import (
    BREAKOUT_QUALITY_ALLOW_TF32,
    BREAKOUT_QUALITY_DEFAULT_BATCH_SIZE,
    BREAKOUT_QUALITY_DEFAULT_EPOCHS,
    BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    BREAKOUT_QUALITY_DEFAULT_GRADIENT_CLIP_NORM,
    BREAKOUT_QUALITY_DEFAULT_LEARNING_RATE,
    BREAKOUT_QUALITY_DEFAULT_RANDOM_SEED,
    BREAKOUT_QUALITY_DEFAULT_WEIGHT_DECAY,
    BREAKOUT_QUALITY_DETERMINISTIC_ALGORITHMS,
    BREAKOUT_QUALITY_EARLY_STOPPING_MIN_DELTA,
    BREAKOUT_QUALITY_EARLY_STOPPING_PATIENCE,
    BREAKOUT_QUALITY_EVALUATION_BATCH_SIZE,
    BREAKOUT_QUALITY_INNER_VALIDATION_MONTHS,
    BREAKOUT_QUALITY_MIXED_PRECISION_DTYPE,
    BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
    BREAKOUT_QUALITY_PRELOAD_FEATURE_BANK,
    BREAKOUT_QUALITY_TORCH_DEVICE,
    BREAKOUT_QUALITY_USE_INNER_VALIDATION,
    BREAKOUT_QUALITY_USE_MIXED_PRECISION,
)
from filters.breakout_quality.artifacts import build_file_manifest
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
from tools.filters.breakout_quality.common import (
    PROJECT_ROOT,
    load_validated_dataset_bundle,
    write_json,
)

RANKER_SCHEMA_VERSION = 1
RANKER_SCORE_FILENAME = "continuous_ranker_scores.csv"
RANKER_REPORT_JSON_FILENAME = "continuous_ranker_report.json"
RANKER_REPORT_MARKDOWN_FILENAME = "continuous_ranker_report.md"
RANKER_TARGET_FILENAME = "group_target_daily_percentile.npy"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "11B research-only同日percentile regression；保留9A InceptionTime與兩logit head，"
            "以softmax PASS probability對11A同日target percentile做MSE"
        )
    )
    parser.add_argument("--filter-id", default=BREAKOUT_QUALITY_DEFAULT_FILTER_ID)
    parser.add_argument(
        "--experiment-profile",
        default=STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE,
        choices=(STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE,),
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
    parser.add_argument("--seed", type=int, default=BREAKOUT_QUALITY_DEFAULT_RANDOM_SEED)
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
    return parser.parse_args(argv)


def _validate_args(args) -> None:
    profile = get_breakout_quality_experiment_profile(args.experiment_profile)
    if profile.training_objective != TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION:
        raise ValueError("11B命令只接受daily percentile regression profile")
    if profile.continuous_target_id != STRATEGY_ALIGNED_TARGET_ID:
        raise ValueError("11B profile continuous target id不一致")
    if not bool(args.use_inner_validation):
        raise ValueError("11B第一版必須使用inner validation選epoch")
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
        raise ValueError("11B要求group_index連續完整")
    if not np.array_equal(
        np.asarray(event_group_index[group["event_row"].to_numpy(dtype=np.int64)], dtype=np.int64),
        expected,
    ):
        raise ValueError("11B group representative與event_group_index不一致")
    mixed = frame.groupby("group_index", sort=False)["label"].nunique()
    if bool((mixed != 1).any()):
        raise ValueError("11B發現同group混合binary label")
    return group.reset_index(drop=True)


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
    return {
        "group_count": int(len(ids)),
        "mse_vs_daily_percentile": float(np.mean((score_values - pct_values) ** 2)),
        "global_spearman_vs_raw_target": _spearman(score_values, raw_values),
        "global_spearman_vs_daily_percentile": _spearman(score_values, pct_values),
        **daily,
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
        architecture=BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
    ).to(plan.device)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    return model, optimizer


def _train_epoch(
    torch,
    model,
    optimizer,
    feature_bank: np.ndarray,
    group_context: np.ndarray,
    group_ids: np.ndarray,
    percentile_target: np.ndarray,
    *,
    batch_size: int,
    seed: int,
    gradient_clip_norm: float,
    plan,
    grad_scaler,
) -> float:
    model.train()
    rng = np.random.default_rng(int(seed))
    order = np.asarray(group_ids, dtype=np.int64).copy()
    rng.shuffle(order)
    losses: list[float] = []
    import torch.nn.functional as F

    for start in range(0, len(order), int(batch_size)):
        ids = order[start:start + int(batch_size)]
        xb = torch.from_numpy(np.asarray(feature_bank[ids], dtype=np.float32)).to(plan.device)
        cb = torch.from_numpy(np.asarray(group_context[ids], dtype=np.float32)).to(plan.device)
        target = torch.from_numpy(np.asarray(percentile_target[ids], dtype=np.float32)).to(plan.device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(torch, plan):
            logits = model(xb, cb)
            score = torch.softmax(logits.float(), dim=1)[:, LABEL_PASS]
            loss = F.mse_loss(score, target, reduction="mean")
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("11B training loss非有限值")
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
        losses.append(float(loss.detach().cpu().item()))
    if not losses:
        raise ValueError("11B training沒有任何batch")
    return float(np.mean(losses))


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
            raise ValueError("11B validation mean daily Spearman不可用")
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
        marker = " ★新最佳" if improved else ""
        print(
            f"  Epoch {epoch:>3}/{int(args.epochs)} | Train MSE {train_metrics['mse_vs_daily_percentile']:.6f} "
            f"| Val MSE {validation_mse:.6f} | Val Daily Spearman {float(metric):.4f} "
            f"| {elapsed:.1f}s{marker}"
        )
        if int(args.early_stopping_patience) > 0 and epochs_without_improvement >= int(args.early_stopping_patience):
            break
    if best_epoch < 1:
        raise ValueError("11B無法選出best epoch")
    return {
        "best_epoch": int(best_epoch),
        "best_validation_mean_daily_spearman": float(best_metric),
        "best_validation_mse": float(best_validation_mse),
        "completed_epochs": int(len(history)),
        "history": history,
    }


def _fit_final(torch, feature_bank: np.ndarray, group_context: np.ndarray, percentile_target: np.ndarray, final_ids: np.ndarray, *, epochs: int, args, plan):
    model, optimizer = _new_model_and_optimizer(
        torch,
        feature_count=int(feature_bank.shape[2]),
        context_count=int(group_context.shape[1]),
        args=args,
        plan=plan,
    )
    grad_scaler = build_grad_scaler(torch, plan)
    history: list[dict[str, Any]] = []
    print(f"\n完整Selection重訓（{int(epochs)} Epoch）")
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
            batch_size=int(args.batch_size),
            seed=int(args.seed) + epoch,
            gradient_clip_norm=float(args.gradient_clip_norm),
            plan=plan,
            grad_scaler=grad_scaler,
        )
        elapsed = time.perf_counter() - started
        history.append({"epoch": int(epoch), "batch_loss": float(loss), "elapsed_sec": round(float(elapsed), 3)})
        print(f"  Epoch {epoch:>3}/{int(epochs)} | Batch MSE {loss:.6f} | {elapsed:.1f}s")
    return model.eval(), history


def _trade_alignment_metrics(score_frame: pd.DataFrame, target_dir: Path) -> dict[str, Any]:
    path = target_dir / TARGET_TRADE_MATCHES_CSV_FILENAME
    if not path.is_file():
        return {"available": False, "reason": f"not found: {path}"}
    trades = pd.read_csv(path, encoding="utf-8-sig")
    missing = sorted({"ticker", "target_date", "r_multiple"} - set(trades.columns))
    if missing:
        return {"available": False, "reason": f"trade matches missing columns: {missing}", "path": str(path)}
    lookup = score_frame[["ticker", "date", "model_score"]].copy()
    lookup["ticker"] = lookup["ticker"].astype(str)
    lookup["target_date"] = pd.to_datetime(lookup["date"], errors="raise").dt.strftime("%Y-%m-%d")
    lookup = lookup.drop(columns=["date"])
    matched = trades.copy()
    matched["ticker"] = matched["ticker"].astype(str)
    matched["target_date"] = matched["target_date"].astype(str)
    matched["r_multiple"] = pd.to_numeric(matched["r_multiple"], errors="coerce")
    matched = matched.merge(lookup, how="left", on=["ticker", "target_date"], validate="many_to_one")
    valid = matched[np.isfinite(matched["r_multiple"]) & np.isfinite(matched["model_score"])].copy()
    if valid.empty:
        return {"available": True, "path": str(path), "trade_count": int(len(matched)), "matched_trade_count": 0}
    valid = valid.sort_values("model_score", kind="mergesort")
    count = max(1, int(math.ceil(len(valid) * 0.10)))
    scores = valid["model_score"].to_numpy(dtype=np.float64)
    r_values = valid["r_multiple"].to_numpy(dtype=np.float64)
    large = r_values >= 2.0
    median_score = float(np.median(scores))
    return {
        "available": True,
        "path": str(path),
        "trade_count": int(len(matched)),
        "matched_trade_count": int(len(valid)),
        "coverage_rate": float(len(valid) / len(matched)) if len(matched) else None,
        "spearman_model_score_vs_r_multiple": _spearman(scores, r_values),
        "top_model_score_decile_average_r": float(valid.tail(count)["r_multiple"].mean()),
        "bottom_model_score_decile_average_r": float(valid.head(count)["r_multiple"].mean()),
        "large_winner_count_r_ge_2": int(large.sum()),
        "large_winner_top_half_score_retention": (
            float((scores[large] >= median_score).mean()) if bool(large.any()) else None
        ),
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    def fmt(value, digits=4):
        return "-" if value is None else f"{float(value):.{digits}f}"

    lines = [
        "# 11B Strategy-aligned Daily Percentile Ranker",
        "",
        f"- Profile：`{payload['experiment_profile']}`",
        f"- Architecture：`{payload['model_architecture']}`",
        "- Objective：同日11A target percentile的MSE；score為softmax PASS probability。",
        "- Runtime：research-only；不得匯出forward-OOS runtime scores。",
        f"- Selected epoch：`{payload['training']['selected_epoch']}`",
        "",
        "## 1. Split metrics",
        "",
        "| Split | Groups | MSE | Global Spearman | Mean Daily Spearman | Pair Concordance | Binary PR-AUC | P@50% | P@60% | P@70% | Top10 Target | Bottom10 Target |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("inner_train", "validation", "selection", "oos"):
        row = payload["split_metrics"][name]
        lines.append(
            f"| {name} | {int(row['group_count']):,} | {fmt(row['mse_vs_daily_percentile'], 6)} "
            f"| {fmt(row['global_spearman_vs_raw_target'])} | {fmt(row['mean_daily_spearman'])} "
            f"| {fmt(row['pairwise_concordance'])} | {fmt(row['binary_pr_auc'])} "
            f"| {fmt(row['p_at_50pct'])} | {fmt(row['p_at_60pct'])} | {fmt(row['p_at_70pct'])} "
            f"| {fmt(row['top_score_decile_raw_target_mean'])} | {fmt(row['bottom_score_decile_raw_target_mean'])} |"
        )
    lines.extend(["", "## 2. Actual Round-trip R", ""])
    trade = payload.get("trade_alignment") or {}
    if trade.get("available"):
        coverage_rate = trade.get("coverage_rate")
        large_winner_retention = trade.get("large_winner_top_half_score_retention")
        coverage_percent = None if coverage_rate is None else float(coverage_rate) * 100.0
        large_winner_percent = (
            None if large_winner_retention is None else float(large_winner_retention) * 100.0
        )
        lines.extend([
            f"- 配對：`{trade.get('matched_trade_count')}` / `{trade.get('trade_count')}`；coverage `{fmt(coverage_percent, 2)}%`。",
            f"- Spearman(model score, realized R)：`{fmt(trade.get('spearman_model_score_vs_r_multiple'))}`。",
            f"- Top score decile平均R：`{fmt(trade.get('top_model_score_decile_average_r'))}`；Bottom decile：`{fmt(trade.get('bottom_model_score_decile_average_r'))}`。",
            f"- ≥2R大贏家位於score上半部比例：`{fmt(large_winner_percent, 2)}%`。",
        ])
    else:
        lines.append(f"- 未取得實際R診斷：`{trade.get('reason')}`。")
    lines.extend([
        "",
        "## 3. Boundary",
        "",
        "- Epoch、loss與gradient只使用Selection內Inner Train／Validation。",
        "- OOS只在完整Selection重訓、模型checkpoint寫入後評估。",
        "- 本模型不設threshold、不提供runtime gate、不覆蓋9A正式模型。",
    ])
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    args = parse_args(argv)
    _validate_args(args)
    started = time.perf_counter()
    profile = get_breakout_quality_experiment_profile(args.experiment_profile)
    model_spec = get_model_spec(BREAKOUT_QUALITY_MODEL_ARCHITECTURE)
    if bool(model_spec.requires_market_set) or bool(model_spec.use_dataset_context) or bool(model_spec.derived_context_features):
        raise ValueError("11B第一版只允許active sequence-only 9A architecture")

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
    )
    # Before checkpoint freeze, only Selection-date targets may participate in the
    # derived percentile objective. OOS percentiles are built later for reporting.
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
    inner_train_ids = _group_ids_from_event_rows(event_group_index, inner_train_rows, target_valid)
    validation_ids = _group_ids_from_event_rows(event_group_index, validation_rows, target_valid)
    selection_ids = _group_ids_from_event_rows(event_group_index, selection_rows, target_valid)
    oos_ids = _group_ids_from_event_rows(event_group_index, oos_rows, target_valid)
    selection_target_mask = np.zeros(raw_target.shape, dtype=bool)
    selection_target_mask[selection_ids] = True
    selection_percentiles = build_daily_percentile_targets(
        raw_target,
        selection_target_mask,
        group_table["date"],
    )
    percentile_target[selection_ids] = selection_percentiles[selection_ids]
    for name, ids in {
        "inner_train": inner_train_ids,
        "validation": validation_ids,
        "selection": selection_ids,
        "oos": oos_ids,
    }.items():
        if len(ids) < 20:
            raise ValueError(f"11B {name}有效group不足: {len(ids)}")

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

    epoch_selection = _select_epoch(
        torch,
        feature_bank,
        group_context,
        group_table,
        raw_target,
        percentile_target,
        inner_train_ids,
        validation_ids,
        args=args,
        plan=plan,
    )
    selected_epoch = int(epoch_selection["best_epoch"])
    model, final_history = _fit_final(
        torch,
        feature_bank,
        group_context,
        percentile_target,
        selection_ids,
        epochs=selected_epoch,
        args=args,
        plan=plan,
    )

    artifact_paths = resolve_filter_artifact_paths(
        PROJECT_ROOT,
        args.filter_id,
        BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
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
            "continuous_target_contract": target_manifest.get("target_contract"),
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
    oos_target_mask[oos_ids] = True
    oos_percentiles = build_daily_percentile_targets(
        raw_target,
        oos_target_mask,
        group_table["date"],
    )
    percentile_target[oos_ids] = oos_percentiles[oos_ids]
    split_ids = {
        "inner_train": inner_train_ids,
        "validation": validation_ids,
        "selection": selection_ids,
        "oos": oos_ids,
    }
    split_scores: dict[str, np.ndarray] = {}
    split_metrics: dict[str, Any] = {}
    for name, ids in split_ids.items():
        scores = _predict_scores(
            torch,
            model,
            feature_bank,
            group_context,
            ids,
            batch_size=int(args.evaluation_batch_size),
            plan=plan,
        )
        split_scores[name] = scores
        split_metrics[name] = _split_metrics(
            ids, group_table, raw_target, percentile_target, scores,
        )

    role_by_group = np.full((group_count,), "selection_other", dtype=object)
    role_by_group[inner_train_ids] = "inner_train"
    role_by_group[validation_ids] = "validation"
    role_by_group[oos_ids] = "oos"
    score_frames: list[pd.DataFrame] = []
    for name, ids in (("selection", selection_ids), ("oos", oos_ids)):
        frame = group_table.iloc[ids][["ticker", "date", "group_index", "label"]].copy()
        frame["split"] = name
        frame["selection_role"] = role_by_group[ids]
        frame["target_raw_r"] = raw_target[ids]
        frame["target_daily_percentile"] = percentile_target[ids]
        frame["model_score"] = split_scores[name]
        score_frames.append(frame)
    score_frame = pd.concat(score_frames, ignore_index=True)
    if bool(score_frame["group_index"].duplicated().any()):
        raise ValueError("11B research scores每個group必須唯一")
    score_frame = score_frame.sort_values(
        ["date", "ticker", "group_index"],
        kind="mergesort",
    ).reset_index(drop=True)
    output_dir = resolve_filter_model_output_dir(
        PROJECT_ROOT,
        args.filter_id,
        BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
        args.experiment_profile,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / RANKER_SCORE_FILENAME
    report_json_path = output_dir / RANKER_REPORT_JSON_FILENAME
    report_markdown_path = output_dir / RANKER_REPORT_MARKDOWN_FILENAME
    percentile_path = output_dir / RANKER_TARGET_FILENAME
    score_frame.to_csv(score_path, index=False, encoding="utf-8-sig")
    np.save(percentile_path, percentile_target, allow_pickle=False)

    target_dir = resolve_continuous_target_dir(PROJECT_ROOT, args.filter_id)
    trade_alignment = _trade_alignment_metrics(
        score_frame[score_frame["split"] == "oos"].copy(),
        target_dir,
    )
    information_cutoff = str(
        pd.to_datetime(events.iloc[selection_rows]["label_eval_end_date"], errors="raise").max().date()
    )
    payload = {
        "schema_version": RANKER_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "11B Strategy-aligned Daily Percentile Ranker",
        "status": "RESULT_AVAILABLE_PENDING_REVIEW",
        "filter_id": args.filter_id,
        "model_architecture": BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
        "experiment_profile": args.experiment_profile,
        "experiment_settings": profile.as_manifest_payload(),
        "training": {
            "objective": profile.training_objective,
            "loss": profile.loss_name,
            "model_score": "softmax_pass_probability",
            "target": "same_date_rank_percentile_of_strategy_aligned_opportunity_r_v1",
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
        "trade_alignment": trade_alignment,
        "target_manifest": target_manifest,
        "model_information_cutoff": information_cutoff,
        "oos_used_for_training_or_epoch_selection": False,
        "oos_evaluated_after_checkpoint_write": True,
        "runtime_eligibility": {
            "eligible": False,
            "scope": "research_only",
            "reason": "11B is a ranking research objective without a deployment threshold or runtime score contract",
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
        "model_architecture": BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
        "experiment_profile": args.experiment_profile,
        "experiment_settings": profile.as_manifest_payload(),
        "model_spec": model_spec.as_manifest_payload(),
        "training_objective": profile.training_objective,
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
        "selected_epoch": selected_epoch,
        "epoch_selection_source": "inner_validation_mean_daily_spearman",
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

    print("\n11B daily percentile ranker完成")
    print(
        f"selected_epoch={selected_epoch} "
        f"validation_daily_spearman={epoch_selection['best_validation_mean_daily_spearman']:.4f}"
    )
    for name in ("inner_train", "validation", "selection", "oos"):
        metrics = split_metrics[name]
        print(
            f"- {name:<11} groups={metrics['group_count']:,} "
            f"daily_spearman={metrics['mean_daily_spearman']:.4f} "
            f"global_spearman={metrics['global_spearman_vs_raw_target']:.4f} "
            f"PR-AUC={metrics['binary_pr_auc']:.4f}"
        )
    if trade_alignment.get("available") and trade_alignment.get("matched_trade_count", 0):
        print(
            "- trade R: "
            f"matched={trade_alignment['matched_trade_count']}/{trade_alignment['trade_count']} "
            f"spearman={trade_alignment['spearman_model_score_vs_r_multiple']:.4f}"
        )
    else:
        print(f"- trade R: {trade_alignment.get('reason', 'unavailable')}")
    print(f"已輸出: {artifact_paths.model_path}")
    print(f"已輸出: {report_markdown_path}")
    return 0


__all__ = [
    "RANKER_SCHEMA_VERSION",
    "build_daily_percentile_targets",
    "main",
    "parse_args",
]


if __name__ == "__main__":
    raise SystemExit(main())
