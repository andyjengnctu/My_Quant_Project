"""Build rolling/cross-fitted Selection point-in-time continuous-ranker scores."""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import json
import math
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.display_common import InlineProgress, format_elapsed

from config.breakout_quality import (
    BREAKOUT_QUALITY_ALLOW_TF32,
    BREAKOUT_QUALITY_CONTINUOUS_RANKER_TRAIN_PREFETCH_BATCHES,
    BREAKOUT_QUALITY_CONTINUOUS_RANKER_PREFETCH_WORKERS,
    BREAKOUT_QUALITY_POINT_IN_TIME_FOLD_WORKERS,
    BREAKOUT_QUALITY_DEFAULT_BATCH_SIZE,
    BREAKOUT_QUALITY_DEFAULT_EPOCHS,
    BREAKOUT_QUALITY_DEFAULT_GRADIENT_CLIP_NORM,
    BREAKOUT_QUALITY_DEFAULT_LEARNING_RATE,
    BREAKOUT_QUALITY_DEFAULT_WEIGHT_DECAY,
    BREAKOUT_QUALITY_DETERMINISTIC_ALGORITHMS,
    BREAKOUT_QUALITY_EARLY_STOPPING_MIN_DELTA,
    BREAKOUT_QUALITY_EARLY_STOPPING_PATIENCE,
    BREAKOUT_QUALITY_EVALUATION_BATCH_SIZE,
    BREAKOUT_QUALITY_MIXED_PRECISION_DTYPE,
    BREAKOUT_QUALITY_PRELOAD_FEATURE_BANK,
    BREAKOUT_QUALITY_TORCH_DEVICE,
    BREAKOUT_QUALITY_USE_MIXED_PRECISION,
)
from config.breakout_quality import (
    TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    get_breakout_quality_workflow_settings,
)
from filters.breakout_quality.artifacts import build_file_manifest
from filters.breakout_quality.contract import DEFAULT_MODEL_FILENAME
from filters.breakout_quality.ranker_sample_contract import build_score_eligibility_contract
from filters.breakout_quality.models.factory import build_model
from filters.breakout_quality.paths import (
    SELECTION_POINT_IN_TIME_COVERAGE_FILENAME,
    SELECTION_POINT_IN_TIME_MANIFEST_FILENAME,
    SELECTION_POINT_IN_TIME_SCORE_FILENAME,
    resolve_filter_point_in_time_dir,
)
from filters.breakout_quality.torch_runtime import (
    SUPPORTED_MIXED_PRECISION_DTYPES,
    SUPPORTED_TORCH_DEVICES,
)
from filters.breakout_quality.workflow_io import PROJECT_ROOT, write_json
from core.console_report import (
    compact_console_enabled,
    console_color_enabled,
    paint,
    print_artifact_paths,
    render_key_values,
    render_section,
    render_table,
)
from services.breakout_quality.continuous_ranker_pipeline import (
    build_checkpoint_payload,
    build_percentile_target,
    build_training_scope_mask,
    fit_final,
    load_continuous_ranker_data,
    predict_scores,
    resolve_ranker_execution_plan,
    select_epoch,
)

POINT_IN_TIME_SCHEMA_VERSION = 2
AUTO_SCORE_START_VALUE = "auto"
FOLD_CALENDAR_ANCHOR = pd.Timestamp("2000-01-01")
FOLD_MANIFEST_FILENAME = "manifest.json"
FOLD_SCORE_FILENAME = "scores.csv"
REQUIRED_SCORE_COLUMNS = (
    "ticker",
    "date",
    "group_index",
    "breakout_quality_score",
    "fold_id",
    "model_information_cutoff",
)


def parse_args(argv=None) -> argparse.Namespace:
    active_settings = get_breakout_quality_workflow_settings()
    profile_parser = argparse.ArgumentParser(add_help=False)
    profile_parser.add_argument(
        "--experiment-profile", default=active_settings.experiment_profile
    )
    profile_args, _ = profile_parser.parse_known_args(argv)
    settings = get_breakout_quality_workflow_settings(
        experiment_profile=str(profile_args.experiment_profile)
    )
    parser = argparse.ArgumentParser(
        description=(
            "以rolling point-in-time folds建立continuous-ranker scores；"
            "Operational使用expanding history，Stability可指定fixed history；"
            "每個fold只用score period以前且label已完成的資料訓練。"
        )
    )
    parser.add_argument("--filter-id", default=settings.filter_id)
    parser.add_argument("--model-architecture", default=settings.model_architecture)
    parser.add_argument("--experiment-profile", default=settings.experiment_profile)
    parser.add_argument(
        "--score-start-date",
        default=settings.point_in_time_score_start_date,
        help=(
            "PIT Score起始日；使用auto時由Dataset／Target與目前最小group契約"
            "自動找出最早合法月份"
        ),
    )
    parser.add_argument("--score-end-date", default=settings.point_in_time_score_end_date)
    parser.add_argument("--fold-months", type=int, default=settings.point_in_time_fold_months)
    parser.add_argument(
        "--inner-validation-months",
        type=int,
        default=settings.point_in_time_inner_validation_months,
    )
    parser.add_argument(
        "--train-window-months",
        type=int,
        default=settings.point_in_time_train_window_months,
        help=(
            "完整fit history長度（月）；省略表示expanding history。"
            "固定窗口必須大於inner-validation-months。"
        ),
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
        help="預先materialize後續訓練batches；0表示關閉，不改batch順序",
    )
    parser.add_argument(
        "--train-prefetch-workers",
        type=int,
        default=BREAKOUT_QUALITY_CONTINUOUS_RANKER_PREFETCH_WORKERS,
        help="CPU feature materialization workers；只影響feeding效能",
    )
    parser.add_argument(
        "--fold-workers",
        type=int,
        default=BREAKOUT_QUALITY_POINT_IN_TIME_FOLD_WORKERS,
        help="Rolling缺少fold的獨立process平行數；1表示串行，只影響execution",
    )
    parser.add_argument("--lr", type=float, default=BREAKOUT_QUALITY_DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--weight-decay", type=float, default=BREAKOUT_QUALITY_DEFAULT_WEIGHT_DECAY
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=BREAKOUT_QUALITY_DEFAULT_GRADIENT_CLIP_NORM,
    )
    parser.add_argument("--seed", type=int, default=settings.seed)
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
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=settings.point_in_time_resume,
        help="重用fingerprint與hash均符合的既有fold工件",
    )
    parser.add_argument(
        "--checkpoint-only",
        action="store_true",
        help=(
            "只允許重用既有fold score／checkpoint重建PIT工件；"
            "任何fold若需要模型訓練就立即停止"
        ),
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="只建立並驗證fold計畫，不訓練或寫入正式score工件",
    )
    parser.add_argument(
        "--point-in-time-dir-override",
        default=None,
        help=(
            "隔離研究輸出用PIT根目錄；只改fold/checkpoint/score/manifest寫入位置，"
            "Dataset／Target／feature來源仍讀canonical truth"
        ),
    )
    parser.add_argument(
        "--allow-stale-source",
        action="store_true",
        help="只供離線重現；預設要求來源CSV inventory與dataset一致",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    # Explicit CLI overrides are supported for reproducible research.  The loaded experiment
    # profile and model spec are validated by the shared continuous-ranker pipeline.
    if int(args.fold_months) < 1 or int(args.inner_validation_months) < 1:
        raise ValueError("fold-months與inner-validation-months必須>=1")
    if args.train_window_months is not None:
        if int(args.train_window_months) <= int(args.inner_validation_months):
            raise ValueError("fixed train-window-months必須大於inner-validation-months")
    if int(args.epochs) < 1 or int(args.batch_size) < 2 or int(args.evaluation_batch_size) < 1:
        raise ValueError("epochs>=1、batch-size>=2、evaluation-batch-size>=1")
    if int(args.train_prefetch_batches) < 0:
        raise ValueError("train-prefetch-batches必須>=0")
    if int(getattr(args, "train_prefetch_workers", BREAKOUT_QUALITY_CONTINUOUS_RANKER_PREFETCH_WORKERS)) < 1:
        raise ValueError("train-prefetch-workers必須>=1")
    if int(getattr(args, "fold_workers", BREAKOUT_QUALITY_POINT_IN_TIME_FOLD_WORKERS)) < 1:
        raise ValueError("fold-workers必須>=1")
    if float(args.lr) <= 0.0 or float(args.weight_decay) < 0.0:
        raise ValueError("learning rate必須>0，weight decay必須>=0")
    if float(args.gradient_clip_norm) < 0.0:
        raise ValueError("gradient clip norm必須>=0")
    if int(args.seed) < 0:
        raise ValueError("seed必須>=0")
    if int(args.early_stopping_patience) < 0 or float(args.early_stopping_min_delta) < 0.0:
        raise ValueError("early stopping patience/min delta不可為負")


def _iso_timestamp(value: Any, *, field_name: str) -> pd.Timestamp:
    try:
        result = pd.Timestamp(str(value)).normalize()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 必須是合法日期: {value!r}") from exc
    if pd.isna(result):
        raise ValueError(f"{field_name} 不可為NaT")
    return result


def _resolve_training_universe_start(bundle, *, selection_start: pd.Timestamp) -> pd.Timestamp:
    """Resolve the actual model-history lower bound independently of optimizer selection policy."""

    raw = dict(getattr(bundle, "summary", {}) or {}).get("training_universe_start_date")
    if raw in {None, ""}:
        return pd.Timestamp(selection_start).normalize()
    return _iso_timestamp(raw, field_name="training_universe_start_date")


def _json_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_fold_id(score_start: pd.Timestamp, score_end: pd.Timestamp) -> str:
    return f"fold_{score_start:%Y%m%d}_{score_end:%Y%m%d}"


def _build_fold_periods(
    score_start: pd.Timestamp,
    score_end: pd.Timestamp,
    *,
    fold_months: int,
) -> list[dict[str, Any]]:
    """Build calendar-anchored folds whose later boundaries never shift.

    The first fold may be partial when the resolved earliest legal score date falls
    inside an anchored bucket.  Every later fold follows the fixed 2000-01-01
    calendar anchor, so prepending older history does not rename or retrain existing
    periods such as the original 2014-01-01~2014-12-31 fold.
    """

    start = pd.Timestamp(score_start).normalize()
    end = pd.Timestamp(score_end).normalize()
    months = int(fold_months)
    if months < 1:
        raise ValueError("fold_months必須>=1")
    if end < start:
        raise ValueError("point-in-time score期間不合法")

    month_delta = (start.year - FOLD_CALENDAR_ANCHOR.year) * 12 + (
        start.month - FOLD_CALENDAR_ANCHOR.month
    )
    bucket_index = month_delta // months
    bucket_start = (
        FOLD_CALENDAR_ANCHOR + pd.DateOffset(months=bucket_index * months)
    ).normalize()
    bucket_end = (
        bucket_start + pd.DateOffset(months=months) - pd.Timedelta(days=1)
    ).normalize()

    folds: list[dict[str, Any]] = []
    cursor = start
    while cursor <= end:
        fold_end = min(end, bucket_end)
        folds.append(
            {
                "fold_id": _stable_fold_id(cursor, fold_end),
                "score_start": cursor,
                "score_end": fold_end,
            }
        )
        cursor = (bucket_end + pd.Timedelta(days=1)).normalize()
        bucket_start = cursor
        bucket_end = (
            bucket_start + pd.DateOffset(months=months) - pd.Timedelta(days=1)
        ).normalize()
    if not folds:
        raise ValueError("point-in-time score期間沒有任何fold")
    return folds


def _group_event_count(event_group_index: np.ndarray, group_ids: np.ndarray) -> int:
    return int(np.isin(event_group_index, np.asarray(group_ids, dtype=np.int64)).sum())


def _minimum_count_failures(settings, ids: dict[str, Any]) -> list[str]:
    required = {
        "inner_train": (len(ids["train_ids"]), settings.point_in_time_min_train_groups),
        "validation": (
            len(ids["validation_ids"]),
            settings.point_in_time_min_validation_groups,
        ),
        "score": (len(ids["score_ids"]), settings.point_in_time_min_score_groups),
    }
    return [
        f"{name}={actual}<{minimum}"
        for name, (actual, minimum) in required.items()
        if int(actual) < int(minimum)
    ]


def _candidate_month_starts(
    selection_start: pd.Timestamp, selection_end: pd.Timestamp
) -> list[pd.Timestamp]:
    first = selection_start.normalize()
    candidates = [first]
    next_month = (first + pd.offsets.MonthBegin(1)).normalize()
    while next_month <= selection_end:
        if next_month != candidates[-1]:
            candidates.append(next_month)
        next_month = (next_month + pd.offsets.MonthBegin(1)).normalize()
    return candidates


def _resolve_score_start(
    raw_value: Any,
    *,
    bundle,
    settings,
    selection_start: pd.Timestamp,
    score_end: pd.Timestamp,
    fold_months: int,
    validation_months: int,
) -> tuple[pd.Timestamp, dict[str, Any]]:
    raw_text = str(raw_value or "").strip().lower()
    if raw_text != AUTO_SCORE_START_VALUE:
        configured = _iso_timestamp(raw_value, field_name="score_start_date")
        return configured, {
            "mode": "configured",
            "configured_value": str(raw_value),
            "resolved_score_start": str(configured.date()),
        }

    group_dates = pd.to_datetime(bundle.group_table["date"], errors="raise").dt.normalize()
    label_end_dates = pd.to_datetime(
        bundle.group_table["label_eval_end_date"], errors="raise"
    ).dt.normalize()
    scoped_target = np.asarray(build_training_scope_mask(bundle), dtype=bool)

    # ``auto`` may have to reject many early calendar months before enough legal
    # target history exists (MR-13I/J start only when historical Min ROOS risk
    # calibration is available).  Rebuilding four full-size masks through
    # ``_fold_group_ids`` for every rejected month made the CLI look hung on the
    # ~1M-row daily universe.  Precompute the immutable date arrays once and count
    # the first-fold partitions with search/small-window comparisons.  The formulas
    # below are exactly the same strict date/cutoff predicates as _fold_group_ids.
    all_date_ns = group_dates.to_numpy(dtype="datetime64[ns]").view("i8")
    sorted_all_date_ns = np.sort(all_date_ns)
    scoped_date_ns = all_date_ns[scoped_target]
    scoped_label_ns = (
        label_end_dates.to_numpy(dtype="datetime64[ns]").view("i8")[scoped_target]
    )
    nat_ns = np.datetime64("NaT", "ns").view("i8")
    valid_label_end = scoped_label_ns != nat_ns
    ready_ns = np.maximum(scoped_date_ns[valid_label_end], scoped_label_ns[valid_label_end])
    sorted_ready_ns = np.sort(ready_ns)
    scoped_order = np.argsort(scoped_date_ns, kind="mergesort")
    sorted_scoped_date_ns = scoped_date_ns[scoped_order]
    sorted_scoped_label_ns = scoped_label_ns[scoped_order]

    def first_fold_counts(candidate: pd.Timestamp) -> dict[str, int]:
        fold = _build_fold_periods(
            candidate,
            score_end,
            fold_months=int(fold_months),
        )[0]
        score_start_ns = np.int64(pd.Timestamp(candidate).value)
        score_end_ns = np.int64(pd.Timestamp(fold["score_end"]).value)
        validation_start = (
            pd.Timestamp(candidate) - pd.DateOffset(months=int(validation_months))
        ).normalize()
        validation_start_ns = np.int64(validation_start.value)

        train_count = int(
            np.searchsorted(sorted_ready_ns, validation_start_ns, side="left")
        )
        validation_left = int(
            np.searchsorted(
                sorted_scoped_date_ns, validation_start_ns, side="left"
            )
        )
        validation_right = int(
            np.searchsorted(sorted_scoped_date_ns, score_start_ns, side="left")
        )
        validation_label_ns = sorted_scoped_label_ns[
            validation_left:validation_right
        ]
        validation_count = int(
            (
                (validation_label_ns != nat_ns)
                & (validation_label_ns < score_start_ns)
            ).sum()
        )
        score_count = int(
            np.searchsorted(sorted_all_date_ns, score_end_ns, side="right")
            - np.searchsorted(sorted_all_date_ns, score_start_ns, side="left")
        )
        return {
            "inner_train": train_count,
            "validation": validation_count,
            "score": score_count,
        }

    rejected: list[dict[str, Any]] = []
    for candidate in _candidate_month_starts(selection_start, score_end):
        counts = first_fold_counts(candidate)
        required = {
            "inner_train": (
                counts["inner_train"],
                settings.point_in_time_min_train_groups,
            ),
            "validation": (
                counts["validation"],
                settings.point_in_time_min_validation_groups,
            ),
            "score": (counts["score"], settings.point_in_time_min_score_groups),
        }
        failures = [
            f"{name}={actual}<{minimum}"
            for name, (actual, minimum) in required.items()
            if int(actual) < int(minimum)
        ]
        if not failures:
            scoped_dates = group_dates[scoped_target]
            scoped_label_ends = label_end_dates[scoped_target]
            return candidate, {
                "mode": "auto_earliest_legal",
                "configured_value": AUTO_SCORE_START_VALUE,
                "resolved_score_start": str(candidate.date()),
                "selection_start": str(selection_start.date()),
                "selection_end": str(score_end.date()),
                "earliest_dataset_group_date": str(group_dates.min().date()),
                "earliest_scoped_target_group_date": (
                    str(scoped_dates.min().date()) if len(scoped_dates) else None
                ),
                "earliest_scoped_label_completion_date": (
                    str(scoped_label_ends.min().date()) if len(scoped_label_ends) else None
                ),
                "candidate_months_checked": int(len(rejected) + 1),
                "first_fold_group_counts": counts,
                "rejected_candidate_count": int(len(rejected)),
                "recent_rejected_candidates": rejected[-6:],
            }
        rejected.append(
            {
                "score_start": str(candidate.date()),
                "failures": failures,
            }
        )
    raise ValueError(
        "無法在可用歷史內找到符合目前PIT最小group契約的score start；"
        f"available={selection_start.date()}~{score_end.date()}, "
        f"last_failures={rejected[-1] if rejected else None}"
    )


def _fold_group_ids(
    bundle,
    fold: dict[str, Any],
    *,
    validation_months: int,
    train_window_months: int | None = None,
) -> dict[str, Any]:
    group_dates = pd.to_datetime(bundle.group_table["date"], errors="raise").dt.normalize()
    label_end_dates = pd.to_datetime(
        bundle.group_table["label_eval_end_date"], errors="raise"
    ).dt.normalize()
    score_start = pd.Timestamp(fold["score_start"])
    score_end = pd.Timestamp(fold["score_end"])
    validation_start = (score_start - pd.DateOffset(months=int(validation_months))).normalize()
    history_start = (
        None
        if train_window_months is None
        else (score_start - pd.DateOffset(months=int(train_window_months))).normalize()
    )

    scoped_target = build_training_scope_mask(bundle)
    history_mask = np.ones(len(group_dates), dtype=bool)
    if history_start is not None:
        history_mask &= (group_dates >= history_start).to_numpy(dtype=bool)

    train_mask = (
        scoped_target
        & history_mask
        & (group_dates < validation_start).to_numpy(dtype=bool)
        & (label_end_dates < validation_start).to_numpy(dtype=bool)
    )
    validation_mask = (
        scoped_target
        & history_mask
        & (group_dates >= validation_start).to_numpy(dtype=bool)
        & (group_dates < score_start).to_numpy(dtype=bool)
        & (label_end_dates < score_start).to_numpy(dtype=bool)
    )
    final_mask = (
        scoped_target
        & history_mask
        & (group_dates < score_start).to_numpy(dtype=bool)
        & (label_end_dates < score_start).to_numpy(dtype=bool)
    )
    score_mask = (
        (group_dates >= score_start).to_numpy(dtype=bool)
        & (group_dates <= score_end).to_numpy(dtype=bool)
    )
    ids = {
        "train_ids": np.flatnonzero(train_mask).astype(np.int64),
        "validation_ids": np.flatnonzero(validation_mask).astype(np.int64),
        "final_ids": np.flatnonzero(final_mask).astype(np.int64),
        "score_ids": np.flatnonzero(score_mask).astype(np.int64),
        "validation_start": validation_start,
        "history_start": history_start,
    }
    if np.intersect1d(ids["train_ids"], ids["validation_ids"]).size:
        raise ValueError(f"{fold['fold_id']} train/validation group重疊")
    if np.intersect1d(ids["final_ids"], ids["score_ids"]).size:
        raise ValueError(f"{fold['fold_id']} final train/score group重疊")
    if len(ids["final_ids"]):
        cutoff = label_end_dates.iloc[ids["final_ids"]].max()
        if not cutoff < score_start:
            raise ValueError(
                f"{fold['fold_id']} model information cutoff未早於score start: "
                f"cutoff={cutoff.date()}, score_start={score_start.date()}"
            )
    return ids


def _fold_contract_payload(args, bundle, fold, ids: dict[str, Any]) -> dict[str, Any]:
    group_dates = pd.to_datetime(bundle.group_table["date"], errors="raise").dt.normalize()
    label_end_dates = pd.to_datetime(
        bundle.group_table["label_eval_end_date"], errors="raise"
    ).dt.normalize()
    final_ids = ids["final_ids"]
    train_ids = ids["train_ids"]
    validation_ids = ids["validation_ids"]
    score_ids = ids["score_ids"]
    cutoff = label_end_dates.iloc[final_ids].max()
    outer_selection_start = _iso_timestamp(
        bundle.outer_policy.get("selection_start_date"), field_name="selection_start_date"
    )
    training_universe_start = _resolve_training_universe_start(
        bundle, selection_start=outer_selection_start
    )

    def observed_range(group_ids: np.ndarray) -> dict[str, str | None]:
        if len(group_ids) == 0:
            return {"start": None, "end": None}
        values = group_dates.iloc[group_ids]
        return {"start": str(values.min().date()), "end": str(values.max().date())}

    payload = {
        "schema_version": POINT_IN_TIME_SCHEMA_VERSION,
        "fold_id": str(fold["fold_id"]),
        "filter_id": str(args.filter_id),
        "model_architecture": str(args.model_architecture),
        "experiment_profile": str(args.experiment_profile),
        "continuous_target_id": str(bundle.profile.continuous_target_id),
        "training_label_scope": str(bundle.profile.training_label_scope),
        "training_sample_scope": str(bundle.profile.training_sample_scope),
        "seed": int(args.seed),
        "planned_periods": {
            "validation_start": str(pd.Timestamp(ids["validation_start"]).date()),
            "validation_end": str((pd.Timestamp(fold["score_start"]) - pd.Timedelta(days=1)).date()),
            "score_start": str(pd.Timestamp(fold["score_start"]).date()),
            "score_end": str(pd.Timestamp(fold["score_end"]).date()),
        },
        "observed_periods": {
            "inner_train": observed_range(train_ids),
            "validation": observed_range(validation_ids),
            "final_refit": observed_range(final_ids),
            "score": observed_range(score_ids),
        },
        "model_information_cutoff": str(cutoff.date()),
        "group_counts": {
            "inner_train": int(len(train_ids)),
            "validation": int(len(validation_ids)),
            "final_refit": int(len(final_ids)),
            "score": int(len(score_ids)),
        },
        "event_row_counts": {
            "inner_train": _group_event_count(bundle.event_group_index, train_ids),
            "validation": _group_event_count(bundle.event_group_index, validation_ids),
            "final_refit": _group_event_count(bundle.event_group_index, final_ids),
            "score": _group_event_count(bundle.event_group_index, score_ids),
        },
        "model_spec": bundle.model_spec.as_manifest_payload(),
        "experiment_settings": bundle.profile.as_manifest_payload(),
        "training_settings": {
            "epochs_max": int(args.epochs),
            "batch_size": int(args.batch_size),
            "evaluation_batch_size": int(args.evaluation_batch_size),
            "train_prefetch_batches": int(args.train_prefetch_batches),
            "train_prefetch_workers": int(getattr(args, "train_prefetch_workers", BREAKOUT_QUALITY_CONTINUOUS_RANKER_PREFETCH_WORKERS)),
            "learning_rate": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "gradient_clip_norm": float(args.gradient_clip_norm),
            "early_stopping_patience": int(args.early_stopping_patience),
            "early_stopping_min_delta": float(args.early_stopping_min_delta),
            "device": str(args.device),
            "mixed_precision": bool(args.mixed_precision),
            "mixed_precision_dtype": str(args.mixed_precision_dtype),
            "deterministic_algorithms": bool(args.deterministic_algorithms),
            "allow_tf32": bool(args.allow_tf32),
        },
        "source_contract": {
            "dataset_policy": bundle.summary.get("policy"),
            "dataset_storage_schema_version": bundle.summary.get(
                "dataset_storage_schema_version"
            ),
            "source_data_inventory": bundle.summary.get("source_data_inventory"),
            "dataset_artifacts": bundle.summary.get("dataset_artifacts"),
            "target_schema_version": bundle.target_manifest.get("schema_version"),
            "target_contract": bundle.target_manifest.get("target_contract"),
            "target_artifacts": bundle.target_manifest.get("artifacts"),
        },
        "lookahead_contract": {
            "training_requires_label_eval_end_before_score_start": True,
            "validation_requires_label_eval_end_before_score_start": True,
            "score_period_used_for_training_or_epoch_selection": False,
            "oos_used_for_training_or_epoch_selection": False,
        },
    }

    if (
        str(bundle.profile.training_sample_scope)
        == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
    ):
        payload["source_contract"]["training_universe_start_date"] = str(
            training_universe_start.date()
        )

    if args.train_window_months is not None:
        payload["training_settings"]["train_window_months"] = int(args.train_window_months)
        payload["planned_periods"]["history_start"] = str(
            pd.Timestamp(ids["history_start"]).date()
        )
    if (
        str(bundle.profile.training_sample_scope)
        == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
    ):
        payload["score_eligibility_contract"] = build_score_eligibility_contract(bundle.profile)
    return payload


def _validate_minimum_counts(settings, fold_id: str, ids: dict[str, Any]) -> None:
    failed = _minimum_count_failures(settings, ids)
    if failed:
        raise ValueError(f"{fold_id} group coverage不足: {', '.join(failed)}")


def _validate_score_frame(frame: pd.DataFrame, *, fold_contract: dict[str, Any]) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_SCORE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"point-in-time fold score缺少欄位: {missing}")
    work = frame[list(REQUIRED_SCORE_COLUMNS)].copy()
    work["ticker"] = work["ticker"].astype(str)
    work["date"] = pd.to_datetime(work["date"], errors="raise").dt.strftime("%Y-%m-%d")
    work["group_index"] = pd.to_numeric(work["group_index"], errors="raise").astype(np.int64)
    work["breakout_quality_score"] = pd.to_numeric(
        work["breakout_quality_score"], errors="raise"
    ).astype(np.float64)
    if not np.isfinite(work["breakout_quality_score"]).all():
        raise ValueError("point-in-time fold score含非有限值")
    if bool(((work["breakout_quality_score"] < 0.0) | (work["breakout_quality_score"] > 1.0)).any()):
        raise ValueError("point-in-time fold score超出[0,1]")
    if bool(work["group_index"].duplicated().any()):
        raise ValueError("point-in-time fold score group_index重複")
    expected_fold_id = str(fold_contract["fold_id"])
    if set(work["fold_id"].astype(str).unique()) != {expected_fold_id}:
        raise ValueError("point-in-time fold score fold_id不一致")
    expected_cutoff = str(fold_contract["model_information_cutoff"])
    if set(work["model_information_cutoff"].astype(str).unique()) != {expected_cutoff}:
        raise ValueError("point-in-time fold score model_information_cutoff不一致")
    start = pd.Timestamp(fold_contract["planned_periods"]["score_start"])
    end = pd.Timestamp(fold_contract["planned_periods"]["score_end"])
    dates = pd.to_datetime(work["date"], errors="raise")
    if bool(((dates < start) | (dates > end)).any()):
        raise ValueError("point-in-time fold score日期超出fold期間")
    if not pd.Timestamp(expected_cutoff) < start:
        raise ValueError("point-in-time fold cutoff未早於score start")
    return work.sort_values(["date", "ticker", "group_index"], kind="mergesort").reset_index(drop=True)


def _load_reusable_fold(
    *,
    fold_dir: Path,
    expected_fingerprint: str,
    fold_contract: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    manifest_path = fold_dir / FOLD_MANIFEST_FILENAME
    model_path = fold_dir / DEFAULT_MODEL_FILENAME
    score_path = fold_dir / FOLD_SCORE_FILENAME
    if not (manifest_path.is_file() and model_path.is_file() and score_path.is_file()):
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            return None
        if str(manifest.get("contract_fingerprint")) != expected_fingerprint:
            return None
        artifacts = manifest.get("artifacts") or {}
        if build_file_manifest(model_path) != artifacts.get("checkpoint"):
            return None
        if build_file_manifest(score_path) != artifacts.get("scores"):
            return None
        frame = pd.read_csv(
            score_path,
            encoding="utf-8-sig",
            dtype={
                "ticker": "string",
                "fold_id": "string",
                "model_information_cutoff": "string",
            },
        )
        validated = _validate_score_frame(frame, fold_contract=fold_contract)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
        return None
    return validated, manifest


def _fold_contract_is_compatible(
    manifest: dict[str, Any], *, expected_contract: dict[str, Any]
) -> bool:
    comparable_fields = (
        "filter_id",
        "model_architecture",
        "experiment_profile",
        "continuous_target_id",
        "training_label_scope",
        "seed",
        "planned_periods",
        "observed_periods",
        "model_information_cutoff",
        "group_counts",
        "event_row_counts",
        "model_spec",
        "experiment_settings",
        "training_settings",
        "source_contract",
        "lookahead_contract",
    )
    if not all(manifest.get(field) == expected_contract.get(field) for field in comparable_fields):
        return False
    if "score_eligibility_contract" in expected_contract:
        return manifest.get("score_eligibility_contract") == expected_contract.get(
            "score_eligibility_contract"
        )
    return True


def _fold_training_contract_is_compatible(
    candidate: dict[str, Any], *, expected_contract: dict[str, Any]
) -> bool:
    """Check that model-fitting semantics are unchanged while ignoring score-only expansion."""

    exact_fields = (
        "filter_id",
        "model_architecture",
        "experiment_profile",
        "continuous_target_id",
        "training_label_scope",
        "training_sample_scope",
        "seed",
        "planned_periods",
        "model_information_cutoff",
        "model_spec",
        "experiment_settings",
        "training_settings",
        "source_contract",
        "lookahead_contract",
    )
    if not all(candidate.get(field) == expected_contract.get(field) for field in exact_fields):
        return False
    for field in ("observed_periods", "group_counts", "event_row_counts"):
        candidate_values = candidate.get(field)
        expected_values = expected_contract.get(field)
        if not isinstance(candidate_values, dict) or not isinstance(expected_values, dict):
            return False
        for phase in ("inner_train", "validation", "final_refit"):
            if candidate_values.get(phase) != expected_values.get(phase):
                return False
    return True


def _rescore_fold_from_compatible_checkpoint(
    *,
    fold_dir: Path,
    bundle,
    ids: dict[str, Any],
    fold_contract: dict[str, Any],
    expected_fingerprint: str,
    args,
    torch_module,
    plan,
) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    """Reuse an unchanged fitted fold model and regenerate its score rows only."""

    manifest_path = fold_dir / FOLD_MANIFEST_FILENAME
    model_path = fold_dir / DEFAULT_MODEL_FILENAME
    if not (manifest_path.is_file() and model_path.is_file()):
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            return None
        artifacts = dict(manifest.get("artifacts") or {})
        if build_file_manifest(model_path) != artifacts.get("checkpoint"):
            return None
        if not _fold_training_contract_is_compatible(
            manifest, expected_contract=fold_contract
        ):
            return None

        checkpoint = torch_module.load(model_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            return None
        checkpoint_contract = checkpoint.get("fold_contract")
        if not isinstance(checkpoint_contract, dict) or not _fold_training_contract_is_compatible(
            checkpoint_contract, expected_contract=fold_contract
        ):
            return None
        selected_epoch = int(checkpoint.get("selected_epoch") or 0)
        if selected_epoch < 1 or selected_epoch != int(manifest.get("selected_epoch") or 0):
            return None
        if checkpoint.get("model_spec") != fold_contract.get("model_spec"):
            return None
        expected_shape = (
            int(bundle.feature_bank.shape[1]),
            int(bundle.feature_bank.shape[2]),
            int(bundle.group_context.shape[1]),
        )
        checkpoint_shape = tuple(
            int(checkpoint.get(field)) if checkpoint.get(field) is not None else -1
            for field in ("sequence_length", "feature_count", "context_count")
        )
        if checkpoint_shape != expected_shape:
            return None

        model = build_model(
            expected_shape[1],
            expected_shape[2],
            model_spec=checkpoint["model_spec"],
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(plan.device)
        model.eval()
        scores = predict_scores(
            torch_module,
            model,
            bundle,
            ids["score_ids"],
            batch_size=int(args.evaluation_batch_size),
            plan=plan,
        )
        if len(scores) != len(ids["score_ids"]):
            return None
        frame = bundle.group_table.iloc[ids["score_ids"]][
            ["ticker", "date", "group_index"]
        ].copy()
        frame["breakout_quality_score"] = scores
        frame["fold_id"] = str(fold_contract["fold_id"])
        frame["model_information_cutoff"] = fold_contract["model_information_cutoff"]
        frame = _validate_score_frame(frame, fold_contract=fold_contract)
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
        TypeError,
        KeyError,
        RuntimeError,
    ):
        return None

    migrated_checkpoint = {**checkpoint, "fold_contract": dict(fold_contract)}
    score_path = fold_dir / FOLD_SCORE_FILENAME
    torch_module.save(migrated_checkpoint, model_path)
    frame.to_csv(score_path, index=False, encoding="utf-8-sig")
    rescored_manifest = {
        **manifest,
        **fold_contract,
        "schema_version": POINT_IN_TIME_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_fingerprint": expected_fingerprint,
        "score_coverage": {
            "expected_groups": int(len(ids["score_ids"])),
            "scored_groups": int(len(frame)),
            "coverage_rate": 1.0 if len(ids["score_ids"]) else None,
        },
        "migration": {
            "kind": (
                "expanded_daily_score_universe_checkpoint_reuse"
                if str(bundle.profile.training_sample_scope)
                == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
                else "checkpoint_score_regeneration"
            ),
            "training_contract_unchanged": True,
            **(
                {"future_target_required_for_score": False}
                if str(bundle.profile.training_sample_scope)
                == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
                else {}
            ),
        },
        "artifacts": {
            "checkpoint": build_file_manifest(model_path),
            "scores": build_file_manifest(score_path),
        },
    }
    write_json(manifest_path, rescored_manifest)
    return frame, rescored_manifest


# Backward-compatible helper name retained for existing focused contract tests.
_rescore_daily_fold_from_compatible_checkpoint = _rescore_fold_from_compatible_checkpoint


def _migrate_compatible_legacy_fold(
    *,
    point_in_time_dir: Path,
    target_fold_dir: Path,
    expected_fingerprint: str,
    fold_contract: dict[str, Any],
    torch_module,
) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    folds_root = point_in_time_dir / "folds"
    if not folds_root.is_dir():
        return None
    expected_fold_id = str(fold_contract["fold_id"])
    for candidate_dir in sorted(path for path in folds_root.iterdir() if path.is_dir()):
        if candidate_dir.resolve() == target_fold_dir.resolve():
            continue
        manifest_path = candidate_dir / FOLD_MANIFEST_FILENAME
        model_path = candidate_dir / DEFAULT_MODEL_FILENAME
        score_path = candidate_dir / FOLD_SCORE_FILENAME
        if not (manifest_path.is_file() and model_path.is_file() and score_path.is_file()):
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict) or not _fold_contract_is_compatible(
                manifest, expected_contract=fold_contract
            ):
                continue
            artifacts = dict(manifest.get("artifacts") or {})
            if build_file_manifest(model_path) != artifacts.get("checkpoint"):
                continue
            if build_file_manifest(score_path) != artifacts.get("scores"):
                continue
            checkpoint = torch_module.load(
                model_path, map_location="cpu", weights_only=True
            )
            if not isinstance(checkpoint, dict):
                continue
            checkpoint_contract = dict(checkpoint.get("fold_contract") or {})
            if str(checkpoint_contract.get("fold_id") or "") != str(
                manifest.get("fold_id") or ""
            ):
                continue
            if not _fold_contract_is_compatible(
                checkpoint_contract, expected_contract=fold_contract
            ):
                continue
            migrated_checkpoint = {
                **checkpoint,
                "fold_contract": dict(fold_contract),
            }
            frame = pd.read_csv(
                score_path,
                encoding="utf-8-sig",
                dtype={
                    "ticker": "string",
                    "fold_id": "string",
                    "model_information_cutoff": "string",
                },
            )
            source_fold_ids = set(frame.get("fold_id", pd.Series(dtype=str)).astype(str).unique())
            if source_fold_ids != {str(manifest.get("fold_id") or "")}:
                continue
            frame = frame.copy()
            frame["fold_id"] = expected_fold_id
            frame["model_information_cutoff"] = fold_contract["model_information_cutoff"]
            frame = _validate_score_frame(frame, fold_contract=fold_contract)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
            continue

        target_fold_dir.mkdir(parents=True, exist_ok=True)
        target_model_path = target_fold_dir / DEFAULT_MODEL_FILENAME
        target_score_path = target_fold_dir / FOLD_SCORE_FILENAME
        torch_module.save(migrated_checkpoint, target_model_path)
        frame.to_csv(target_score_path, index=False, encoding="utf-8-sig")
        migrated_manifest = {
            **manifest,
            **fold_contract,
            "schema_version": POINT_IN_TIME_SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "contract_fingerprint": expected_fingerprint,
            "migration": {
                "kind": "stable_date_fold_id",
                "source_fold_id": str(manifest.get("fold_id") or candidate_dir.name),
                "source_manifest": build_file_manifest(manifest_path),
            },
            "artifacts": {
                "checkpoint": build_file_manifest(target_model_path),
                "scores": build_file_manifest(target_score_path),
            },
        }
        write_json(target_fold_dir / FOLD_MANIFEST_FILENAME, migrated_manifest)
        return frame, migrated_manifest
    return None


def _point_in_time_dir_for_args(args) -> Path:
    override = str(getattr(args, "point_in_time_dir_override", "") or "").strip()
    if override:
        return Path(override).resolve()
    return resolve_filter_point_in_time_dir(
        PROJECT_ROOT,
        args.filter_id,
        args.model_architecture,
        args.experiment_profile,
    )


def _point_in_time_fold_dir_for_args(args, fold_id: str) -> Path:
    normalized = str(fold_id).strip()
    if not normalized or Path(normalized).name != normalized:
        raise ValueError("point-in-time fold_id 必須是安全的單一資料夾名稱")
    return _point_in_time_dir_for_args(args) / "folds" / normalized


def _selection_point_in_time_score_path_for_args(args) -> Path:
    return _point_in_time_dir_for_args(args) / SELECTION_POINT_IN_TIME_SCORE_FILENAME


def _selection_point_in_time_coverage_path_for_args(args) -> Path:
    return _point_in_time_dir_for_args(args) / SELECTION_POINT_IN_TIME_COVERAGE_FILENAME


def _selection_point_in_time_manifest_path_for_args(args) -> Path:
    return _point_in_time_dir_for_args(args) / SELECTION_POINT_IN_TIME_MANIFEST_FILENAME


def _train_fold(
    args, bundle, fold, ids, fold_contract, contract_fingerprint, *, torch, plan
):
    fold_id = str(fold["fold_id"])
    percentile_target = build_percentile_target(bundle, ids["final_ids"])
    epoch_selection = select_epoch(
        torch,
        bundle,
        percentile_target,
        ids["train_ids"],
        ids["validation_ids"],
        args=args,
        plan=plan,
    )
    selected_epoch = int(epoch_selection["best_epoch"])
    model, final_history = fit_final(
        torch,
        bundle,
        percentile_target,
        ids["final_ids"],
        epochs=selected_epoch,
        args=args,
        plan=plan,
    )
    scores = predict_scores(
        torch,
        model,
        bundle,
        ids["score_ids"],
        batch_size=int(args.evaluation_batch_size),
        plan=plan,
    )
    if len(scores) != len(ids["score_ids"]):
        raise ValueError(f"{fold_id} inference score count不一致")
    frame = bundle.group_table.iloc[ids["score_ids"]][
        ["ticker", "date", "group_index"]
    ].copy()
    frame["breakout_quality_score"] = scores
    frame["fold_id"] = fold_id
    frame["model_information_cutoff"] = fold_contract["model_information_cutoff"]
    frame = _validate_score_frame(frame, fold_contract=fold_contract)

    fold_dir = _point_in_time_fold_dir_for_args(args, fold_id)
    fold_dir.mkdir(parents=True, exist_ok=True)
    model_path = fold_dir / DEFAULT_MODEL_FILENAME
    score_path = fold_dir / FOLD_SCORE_FILENAME
    checkpoint_payload = build_checkpoint_payload(
        model,
        bundle,
        args=args,
        plan=plan,
        selected_epoch=selected_epoch,
        fold_contract=fold_contract,
    )
    torch.save(checkpoint_payload, model_path)
    frame.to_csv(score_path, index=False, encoding="utf-8-sig")
    manifest = {
        **fold_contract,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_fingerprint": contract_fingerprint,
        "selected_epoch": selected_epoch,
        "epoch_selection": epoch_selection,
        "final_refit_history": final_history,
        "score_coverage": {
            "expected_groups": int(len(ids["score_ids"])),
            "scored_groups": int(len(frame)),
            "coverage_rate": 1.0 if len(ids["score_ids"]) else None,
        },
        "torch_execution": plan.as_manifest_payload(),
        "artifacts": {
            "checkpoint": build_file_manifest(model_path),
            "scores": build_file_manifest(score_path),
        },
    }
    write_json(fold_dir / FOLD_MANIFEST_FILENAME, manifest)
    return frame, manifest


_PARALLEL_FOLD_WORKER_STATE: dict[str, Any] | None = None


def _init_parallel_fold_worker(args_payload: dict[str, Any]) -> None:
    """Initialize one isolated Rolling fold process and its CUDA/runtime state."""

    args = argparse.Namespace(**dict(args_payload))
    _validate_args(args)
    settings = get_breakout_quality_workflow_settings(
        experiment_profile=str(args.experiment_profile)
    )
    if not settings.rolling_authorized:
        raise ValueError(
            f"目前profile未授權Rolling PIT scores: {args.experiment_profile}"
        )
    bundle = load_continuous_ranker_data(
        filter_id=args.filter_id,
        model_architecture=args.model_architecture,
        experiment_profile=args.experiment_profile,
        preload_feature_bank=bool(args.preload_feature_bank),
        allow_stale_source=bool(args.allow_stale_source),
        project_root=PROJECT_ROOT,
        progress_callback=None,
    )
    torch, plan = resolve_ranker_execution_plan(args)
    global _PARALLEL_FOLD_WORKER_STATE
    _PARALLEL_FOLD_WORKER_STATE = {
        "args": args,
        "settings": settings,
        "bundle": bundle,
        "torch": torch,
        "plan": plan,
    }


def _run_parallel_fold_task(task: dict[str, Any]) -> dict[str, Any]:
    """Train exactly one fold inside an initialized process and persist its fold artifacts."""

    state = _PARALLEL_FOLD_WORKER_STATE
    if state is None:
        raise RuntimeError("parallel fold worker尚未初始化")
    args = state["args"]
    settings = state["settings"]
    bundle = state["bundle"]
    torch = state["torch"]
    plan = state["plan"]
    fold = {
        "fold_id": str(task["fold_id"]),
        "score_start": pd.Timestamp(str(task["score_start"])).normalize(),
        "score_end": pd.Timestamp(str(task["score_end"])).normalize(),
    }
    ids = _fold_group_ids(
        bundle,
        fold,
        validation_months=int(args.inner_validation_months),
        train_window_months=(
            None if args.train_window_months is None else int(args.train_window_months)
        ),
    )
    _validate_minimum_counts(settings, str(fold["fold_id"]), ids)
    fold_contract = _fold_contract_payload(args, bundle, fold, ids)
    fingerprint = _json_fingerprint(fold_contract)
    expected_fingerprint = str(task["expected_fingerprint"])
    if fingerprint != expected_fingerprint:
        raise RuntimeError(
            "parallel fold reconstructed contract fingerprint不一致: "
            f"fold={fold['fold_id']} expected={expected_fingerprint} actual={fingerprint}"
        )
    started = time.perf_counter()
    frame, manifest = _train_fold(
        args,
        bundle,
        fold,
        ids,
        fold_contract,
        fingerprint,
        torch=torch,
        plan=plan,
    )
    epoch_selection = dict(manifest.get("epoch_selection") or {})
    return {
        "fold_id": str(fold["fold_id"]),
        "selected_epoch": int(manifest["selected_epoch"]),
        "validation_mean_daily_spearman": float(
            epoch_selection.get("best_validation_mean_daily_spearman")
        ),
        "score_count": int(len(frame)),
        "elapsed_sec": float(time.perf_counter() - started),
    }


def _combined_validation(
    combined: pd.DataFrame,
    bundle,
    *,
    score_start: pd.Timestamp,
    score_end: pd.Timestamp,
) -> dict[str, Any]:
    combined = combined.copy()
    if bool(combined["group_index"].duplicated().any()):
        duplicates = int(combined["group_index"].duplicated(keep=False).sum())
        raise ValueError(f"Selection PIT scores有重複group: rows={duplicates}")
    group_dates = pd.to_datetime(bundle.group_table["date"], errors="raise").dt.normalize()
    expected_ids = np.flatnonzero(
        ((group_dates >= score_start) & (group_dates <= score_end)).to_numpy(dtype=bool)
    ).astype(np.int64)
    actual_ids = np.sort(combined["group_index"].to_numpy(dtype=np.int64))
    expected_sorted = np.sort(expected_ids)
    missing_ids = np.setdiff1d(expected_sorted, actual_ids, assume_unique=True)
    extra_ids = np.setdiff1d(actual_ids, expected_sorted, assume_unique=True)
    if len(missing_ids) or len(extra_ids):
        raise ValueError(
            "Selection PIT scores coverage不完整: "
            f"missing={len(missing_ids)}, extra={len(extra_ids)}"
        )
    if len(combined) != len(expected_ids):
        raise ValueError("Selection PIT scores row count與expected groups不一致")

    expected_identity = bundle.group_table.iloc[expected_ids][
        ["group_index", "ticker", "date"]
    ].copy()
    expected_identity["ticker"] = expected_identity["ticker"].astype(str)
    expected_identity["date"] = pd.to_datetime(
        expected_identity["date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    actual_identity = combined[["group_index", "ticker", "date"]].copy()
    actual_identity["ticker"] = actual_identity["ticker"].astype(str)
    actual_identity["date"] = pd.to_datetime(
        actual_identity["date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    identity_check = actual_identity.merge(
        expected_identity,
        how="left",
        on="group_index",
        suffixes=("", "_expected"),
        validate="one_to_one",
    )
    mismatch = (
        identity_check["ticker"] != identity_check["ticker_expected"]
    ) | (identity_check["date"] != identity_check["date_expected"])
    if bool(mismatch.any()):
        raise ValueError(
            "Selection PIT scores ticker/date與dataset group identity不一致: "
            f"mismatch_groups={int(mismatch.sum())}"
        )
    return {
        "expected_group_count": int(len(expected_ids)),
        "scored_group_count": int(len(combined)),
        "coverage_rate": float(len(combined) / len(expected_ids)) if len(expected_ids) else None,
        "duplicate_group_count": 0,
        "missing_group_count": 0,
        "extra_group_count": 0,
        "expected_event_row_count": _group_event_count(bundle.event_group_index, expected_ids),
    }


def _print_plan(
    folds: list[dict[str, Any]],
    fold_details: list[dict[str, Any]],
    *,
    color: bool = False,
) -> None:
    if compact_console_enabled():
        total_score_groups = sum(len(detail["score_ids"]) for detail in fold_details)
        print(
            paint("PIT 計畫", "cyan", enabled=color, bold=True)
            + f" | folds={len(folds)}"
            + f" | period={folds[0]['score_start'].date()}～{folds[-1]['score_end'].date()}"
            + f" | score groups={total_score_groups:,}"
        )
        return

    print(
        render_section(
            paint(
                "Selection point-in-time fold plan",
                "cyan",
                enabled=color,
                bold=True,
            )
        )
    )
    rows = []
    for fold, detail in zip(folds, fold_details):
        rows.append(
            (
                paint(str(fold["fold_id"]), "cyan", enabled=color, bold=True),
                f"{fold['score_start'].date()} ～ {fold['score_end'].date()}",
                f"{len(detail['train_ids']):,}",
                f"{len(detail['validation_ids']):,}",
                f"{len(detail['final_ids']):,}",
                f"{len(detail['score_ids']):,}",
            )
        )
    print(
        render_table(
            ("Fold", "Score period", "Train", "Validation", "Refit", "Score"),
            rows,
            alignments=("left", "left", "right", "right", "right", "right"),
        )
    )


def _combined_fold_record(args, item: dict[str, Any]) -> dict[str, Any]:
    fold_id = str(item["fold_id"])
    fold_dir = _point_in_time_fold_dir_for_args(args, fold_id)
    manifest_path = fold_dir / FOLD_MANIFEST_FILENAME
    return {
        "fold_id": fold_id,
        "planned_periods": item["planned_periods"],
        "observed_periods": item["observed_periods"],
        "model_information_cutoff": item["model_information_cutoff"],
        "group_counts": item["group_counts"],
        "event_row_counts": item["event_row_counts"],
        "selected_epoch": item["selected_epoch"],
        "contract_fingerprint": item["contract_fingerprint"],
        "artifacts": item["artifacts"],
        "fold_manifest": build_file_manifest(manifest_path),
    }


def _run_point_in_time_scores(args: argparse.Namespace) -> int:
    _validate_args(args)
    settings = get_breakout_quality_workflow_settings(
        experiment_profile=str(args.experiment_profile)
    )
    if not settings.rolling_authorized:
        raise ValueError(
            f"目前profile未授權Rolling PIT scores: {args.experiment_profile}"
        )
    started = time.perf_counter()
    color_enabled = console_color_enabled()
    data_started = time.perf_counter()
    data_progress_state = {"last_bucket": -1}

    def _daily_data_progress(processed, total, target_valid, skipped):
        total = max(int(total), 1)
        processed = int(processed)
        bucket = min(20, int(processed * 20 / total))
        if processed not in {0, total} and bucket <= int(data_progress_state["last_bucket"]):
            return
        data_progress_state["last_bucket"] = bucket
        pct = min(100.0, 100.0 * processed / total)
        print(
            "[PIT data] "
            f"{processed}/{total} tickers ({pct:5.1f}%)｜"
            f"target_valid={int(target_valid):,}｜skipped={int(skipped)}｜"
            f"elapsed={format_elapsed(time.perf_counter() - data_started)}",
            flush=True,
        )

    daily_scope = (
        settings.training_sample_scope
        == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
    )
    if daily_scope:
        print("[PIT data] 建立daily stock-day index／40D target...", flush=True)
    bundle = load_continuous_ranker_data(
        filter_id=args.filter_id,
        model_architecture=args.model_architecture,
        experiment_profile=args.experiment_profile,
        preload_feature_bank=bool(args.preload_feature_bank),
        allow_stale_source=bool(args.allow_stale_source),
        project_root=PROJECT_ROOT,
        progress_callback=_daily_data_progress if daily_scope else None,
    )
    selection_start = _iso_timestamp(
        bundle.outer_policy.get("selection_start_date"), field_name="selection_start_date"
    )
    training_universe_start = _resolve_training_universe_start(
        bundle, selection_start=selection_start
    )
    selection_end = _iso_timestamp(
        bundle.outer_policy.get("selection_end_date"), field_name="selection_end_date"
    )
    outer_oos_end_raw = bundle.outer_policy.get("effective_oos_end_date") or bundle.outer_policy.get("oos_end_date")
    available_end = (
        selection_end
        if outer_oos_end_raw in {None, ""}
        else _iso_timestamp(outer_oos_end_raw, field_name="oos_end_date")
    )
    raw_score_end = str(args.score_end_date or "").strip().lower()
    if raw_score_end in {"", "none"}:
        # Legacy reproduction: historical callers that omitted score_end retain the
        # former Selection-bound behavior.  Operational config explicitly uses auto.
        score_end = selection_end
        score_end_mode = "legacy_selection_end"
    elif raw_score_end == AUTO_SCORE_START_VALUE:
        score_end = available_end
        score_end_mode = "auto_available_end"
    else:
        score_end = _iso_timestamp(args.score_end_date, field_name="score_end_date")
        score_end_mode = "configured"
    if not selection_start <= score_end <= available_end:
        raise ValueError(
            "PIT score end必須位於目前可驗證歷史內: "
            f"available={selection_start.date()}~{available_end.date()}, "
            f"score_end={score_end.date()}"
        )
    if str(args.score_start_date or "").strip().lower() == AUTO_SCORE_START_VALUE:
        print("[PIT start] 自動解析最早合法PIT Score日期...", flush=True)
    score_start, score_start_resolution = _resolve_score_start(
        args.score_start_date,
        bundle=bundle,
        settings=settings,
        selection_start=selection_start,
        score_end=score_end,
        fold_months=int(args.fold_months),
        validation_months=int(args.inner_validation_months),
    )
    if not selection_start <= score_start <= score_end <= available_end:
        raise ValueError(
            "PIT score期間必須完整位於目前可驗證歷史內: "
            f"available={selection_start.date()}~{available_end.date()}, "
            f"score={score_start.date()}~{score_end.date()}"
        )

    if score_start_resolution.get("mode") == "auto_earliest_legal":
        print(
            paint("最早合法 PIT Score 日期", "cyan", enabled=color_enabled, bold=True)
            + f"：{score_start_resolution['resolved_score_start']}"
            + f"｜檢查月份={score_start_resolution['candidate_months_checked']}"
            + f"｜首fold train/val/score="
            + "/".join(
                f"{int(score_start_resolution['first_fold_group_counts'][key]):,}"
                for key in ("inner_train", "validation", "score")
            )
        )
    folds = _build_fold_periods(score_start, score_end, fold_months=int(args.fold_months))
    print(f"[PIT plan] 建立 {len(folds)} 個fold的合法 train/validation/score partitions...", flush=True)
    fold_details = [
        _fold_group_ids(
            bundle,
            fold,
            validation_months=int(args.inner_validation_months),
            train_window_months=(
                None if args.train_window_months is None else int(args.train_window_months)
            ),
        )
        for fold in folds
    ]
    for fold, ids in zip(folds, fold_details):
        _validate_minimum_counts(settings, str(fold["fold_id"]), ids)
    _print_plan(folds, fold_details, color=color_enabled)
    if bool(args.plan_only):
        print(
            paint(
                "plan-only完成；未訓練、未寫入正式PIT工件。",
                "yellow",
                enabled=color_enabled,
                bold=True,
            )
        )
        return 0

    torch, plan = resolve_ranker_execution_plan(args)
    compact_console = compact_console_enabled()
    if compact_console:
        precision = (
            plan.autocast_dtype_name
            if plan.mixed_precision_enabled
            else "float32"
        )
        print(
            paint("執行環境", "cyan", enabled=color_enabled, bold=True)
            + f" | device={plan.device_type}"
            + f" | dtype={precision}"
            + f" | deterministic={plan.deterministic_algorithms}"
            + f" | TF32={plan.allow_tf32}"
        )
    else:
        print(
            render_section(
                paint("執行環境", "cyan", enabled=color_enabled, bold=True)
            )
        )
        print(
            render_key_values(
                (
                    ("Torch device", plan.device_type),
                    ("Mixed precision", plan.mixed_precision_enabled),
                    ("Compute dtype", plan.autocast_dtype_name),
                    ("Deterministic", plan.deterministic_algorithms),
                    ("TF32", plan.allow_tf32),
                )
            )
        )
    point_in_time_dir = _point_in_time_dir_for_args(args)
    point_in_time_dir.mkdir(parents=True, exist_ok=True)

    score_frames: list[pd.DataFrame] = []
    fold_manifests: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    reused_fold_count = 0
    migrated_fold_count = 0
    rescored_checkpoint_fold_count = 0
    built_fold_count = 0
    fold_progress = InlineProgress()
    fold_results: list[tuple[pd.DataFrame, dict[str, Any]] | None] = [None] * len(folds)
    fold_contracts: list[dict[str, Any]] = []
    fold_fingerprints: list[str] = []
    missing_fold_indexes: list[int] = []

    # Resolve all reuse/migration decisions in the parent first.  Parallel workers are
    # only allowed to train folds that genuinely remain missing, so completed artifacts
    # retain the exact same resume semantics as serial execution.
    for fold_index, (fold, ids) in enumerate(zip(folds, fold_details), start=1):
        fold_contract = _fold_contract_payload(args, bundle, fold, ids)
        fingerprint = _json_fingerprint(fold_contract)
        fold_contracts.append(fold_contract)
        fold_fingerprints.append(fingerprint)
        fold_dir = _point_in_time_fold_dir_for_args(args, str(fold["fold_id"]))
        reused = (
            _load_reusable_fold(
                fold_dir=fold_dir,
                expected_fingerprint=fingerprint,
                fold_contract=fold_contract,
            )
            if bool(args.resume)
            else None
        )
        migrated = False
        rescored_checkpoint = False
        if reused is None and bool(args.resume):
            reused = _rescore_fold_from_compatible_checkpoint(
                fold_dir=fold_dir,
                bundle=bundle,
                ids=ids,
                fold_contract=fold_contract,
                expected_fingerprint=fingerprint,
                args=args,
                torch_module=torch,
                plan=plan,
            )
            rescored_checkpoint = reused is not None
        if reused is None and bool(args.resume):
            reused = _migrate_compatible_legacy_fold(
                point_in_time_dir=point_in_time_dir,
                target_fold_dir=fold_dir,
                expected_fingerprint=fingerprint,
                fold_contract=fold_contract,
                torch_module=torch,
            )
            migrated = reused is not None
        if reused is not None:
            fold_results[fold_index - 1] = reused
            if rescored_checkpoint:
                rescored_checkpoint_fold_count += 1
            elif migrated:
                migrated_fold_count += 1
            else:
                reused_fold_count += 1
            if not compact_console:
                if rescored_checkpoint:
                    reuse_label = "重用既有checkpoint並重評score universe"
                elif migrated:
                    reuse_label = "遷移舊fold並重用"
                else:
                    reuse_label = "重用既有 fold 工件"
                print(
                    "\n"
                    + paint(str(fold["fold_id"]), "cyan", enabled=color_enabled, bold=True)
                    + "："
                    + paint(
                        reuse_label,
                        "green",
                        enabled=color_enabled,
                        bold=True,
                    )
                )
            continue
        if bool(args.checkpoint_only):
            raise RuntimeError(
                "checkpoint-only PIT重建無法完成；fold缺少可重用score/checkpoint或identity不相容: "
                f"{fold['fold_id']}。Strategy Compare不得因此訓練模型；"
                "請由模型訓練工作類型的『準備策略比較所需模型工件』補齊PIT fold。"
            )
        missing_fold_indexes.append(fold_index - 1)

    built_fold_count = int(len(missing_fold_indexes))
    requested_fold_workers = int(
        getattr(args, "fold_workers", BREAKOUT_QUALITY_POINT_IN_TIME_FOLD_WORKERS)
    )
    effective_fold_workers = min(requested_fold_workers, max(1, len(missing_fold_indexes)))

    if missing_fold_indexes and effective_fold_workers > 1:
        print(
            paint("PIT fold parallel", "cyan", enabled=color_enabled, bold=True)
            + f" | workers={effective_fold_workers}"
            + f" | missing_folds={len(missing_fold_indexes)}"
            + " | mode=spawn-process",
            flush=True,
        )
        args_payload = dict(vars(args))
        task_queue = iter(missing_fold_indexes)
        mp_context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=effective_fold_workers,
            mp_context=mp_context,
            initializer=_init_parallel_fold_worker,
            initargs=(args_payload,),
        ) as executor:
            pending: dict[Any, int] = {}

            def submit_next() -> bool:
                try:
                    index = next(task_queue)
                except StopIteration:
                    return False
                fold = folds[index]
                task = {
                    "fold_id": str(fold["fold_id"]),
                    "score_start": str(pd.Timestamp(fold["score_start"]).date()),
                    "score_end": str(pd.Timestamp(fold["score_end"]).date()),
                    "expected_fingerprint": fold_fingerprints[index],
                }
                future = executor.submit(_run_parallel_fold_task, task)
                pending[future] = index
                return True

            for _ in range(effective_fold_workers):
                if not submit_next():
                    break

            try:
                while pending:
                    done, _not_done = wait(tuple(pending), return_when=FIRST_COMPLETED)
                    for future in done:
                        index = pending.pop(future)
                        fold = folds[index]
                        try:
                            summary = future.result()
                        except Exception as exc:
                            for queued in pending:
                                queued.cancel()
                            raise RuntimeError(
                                f"Parallel PIT fold訓練失敗: {fold['fold_id']}"
                            ) from exc
                        fold_dir = _point_in_time_fold_dir_for_args(
                            args, str(fold["fold_id"])
                        )
                        built = _load_reusable_fold(
                            fold_dir=fold_dir,
                            expected_fingerprint=fold_fingerprints[index],
                            fold_contract=fold_contracts[index],
                        )
                        if built is None:
                            raise RuntimeError(
                                "Parallel PIT fold完成後工件未通過reuse/hash驗證: "
                                f"{fold['fold_id']}"
                            )
                        fold_results[index] = built
                        fold_progress.print_line(
                            paint(str(fold["fold_id"]), "cyan", enabled=color_enabled, bold=True)
                            + " "
                            + paint("完成", "green", enabled=color_enabled, bold=True)
                            + f" | best epoch={int(summary['selected_epoch'])}"
                            + f" | val rho={float(summary['validation_mean_daily_spearman']):.4f}"
                            + f" | score={int(summary['score_count']):,}"
                            + f" | {format_elapsed(float(summary['elapsed_sec']))}"
                        )
                        submit_next()
            except Exception:
                for queued in pending:
                    queued.cancel()
                raise
    elif missing_fold_indexes:
        for index in missing_fold_indexes:
            fold = folds[index]
            ids = fold_details[index]
            fold_started = time.perf_counter()
            if compact_console:
                fold_progress.update(
                    f"PIT fold {index + 1}/{len(folds)} | {fold['fold_id']} | 訓練並評分"
                )
            else:
                print(
                    "\n"
                    + paint(str(fold["fold_id"]), "cyan", enabled=color_enabled, bold=True)
                    + "："
                    + paint("訓練並評分", "yellow", enabled=color_enabled, bold=True)
                    + f" {fold['score_start'].date()} ～ {fold['score_end'].date()}"
                )
            frame, manifest = _train_fold(
                args,
                bundle,
                fold,
                ids,
                fold_contracts[index],
                fold_fingerprints[index],
                torch=torch,
                plan=plan,
            )
            fold_results[index] = (frame, manifest)
            if compact_console:
                epoch_selection = dict(manifest.get("epoch_selection") or {})
                fold_progress.print_line(
                    paint(str(fold["fold_id"]), "cyan", enabled=color_enabled, bold=True)
                    + " "
                    + paint("完成", "green", enabled=color_enabled, bold=True)
                    + f" | best epoch={int(manifest['selected_epoch'])}"
                    + f" | val rho={float(epoch_selection.get('best_validation_mean_daily_spearman')):.4f}"
                    + f" | score={len(frame):,}"
                    + f" | {format_elapsed(time.perf_counter() - fold_started)}"
                )

    for index, (fold, ids) in enumerate(zip(folds, fold_details)):
        result = fold_results[index]
        if result is None:
            raise RuntimeError(f"PIT fold缺少完成結果: {fold['fold_id']}")
        frame, manifest = result
        fold_contract = fold_contracts[index]
        score_frames.append(frame)
        fold_manifests.append(manifest)
        coverage_rows.append(
            {
                "fold_id": str(fold["fold_id"]),
                "score_start": str(pd.Timestamp(fold["score_start"]).date()),
                "score_end": str(pd.Timestamp(fold["score_end"]).date()),
                "model_information_cutoff": str(fold_contract["model_information_cutoff"]),
                "train_groups": int(len(ids["train_ids"])),
                "validation_groups": int(len(ids["validation_ids"])),
                "final_refit_groups": int(len(ids["final_ids"])),
                "expected_score_groups": int(len(ids["score_ids"])),
                "scored_groups": int(len(frame)),
                "coverage_rate": float(len(frame) / len(ids["score_ids"]))
                if len(ids["score_ids"])
                else math.nan,
                "selected_epoch": int(manifest["selected_epoch"]),
                "checkpoint_sha256": str(manifest["artifacts"]["checkpoint"]["sha256"]),
            }
        )

    combined = pd.concat(score_frames, ignore_index=True)
    combined = combined.sort_values(
        ["date", "ticker", "group_index"], kind="mergesort"
    ).reset_index(drop=True)
    validation = _combined_validation(
        combined,
        bundle,
        score_start=score_start,
        score_end=score_end,
    )
    score_path = _selection_point_in_time_score_path_for_args(args)
    coverage_path = _selection_point_in_time_coverage_path_for_args(args)
    manifest_path = _selection_point_in_time_manifest_path_for_args(args)

    # Evaluation-framework migration: when the canonical expanding PIT aggregate is
    # first extended beyond the historical Selection boundary, preserve the old
    # combined aggregate before refreshing it. Individual fold directories are
    # already immutable/reusable, but this snapshot keeps the pre-migration
    # top-level score/coverage/manifest directly inspectable as historical evidence.
    if (
        not str(getattr(args, "point_in_time_dir_override", "") or "").strip()
        and args.train_window_months is None
        and score_end > selection_end
        and manifest_path.exists()
    ):
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing_manifest = {}
        existing_end_raw = str(
            dict(existing_manifest.get("score_period") or {}).get("end") or ""
        ).strip()
        existing_end = pd.Timestamp(existing_end_raw) if existing_end_raw else None
        if existing_end is not None and existing_end <= selection_end:
            snapshot_dir = point_in_time_dir / "legacy_selection_snapshot"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            for source_path in (score_path, coverage_path, manifest_path):
                if source_path.exists():
                    target_path = snapshot_dir / source_path.name
                    if not target_path.exists():
                        shutil.copy2(source_path, target_path)

    combined.to_csv(score_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(coverage_rows).to_csv(coverage_path, index=False, encoding="utf-8-sig")

    manifest = {
        "schema_version": POINT_IN_TIME_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "BUILT",
        "filter_id": str(args.filter_id),
        "model_architecture": str(args.model_architecture),
        "experiment_profile": str(args.experiment_profile),
        "continuous_target_id": str(bundle.profile.continuous_target_id),
        "training_label_scope": str(bundle.profile.training_label_scope),
        "training_sample_scope": str(bundle.profile.training_sample_scope),
        "score_column": "breakout_quality_score",
        "score_period": {
            "start": str(score_start.date()),
            "end": str(score_end.date()),
        },
        "score_start_resolution": score_start_resolution,
        "evaluation_policy": {
            "mode": (
                "expanding_annual_refit"
                if args.train_window_months is None
                else "fixed_window_annual_refit"
            ),
            "train_window_months": (
                None if args.train_window_months is None else int(args.train_window_months)
            ),
            "refit_score_fold_months": int(args.fold_months),
            "score_end_resolution": score_end_mode,
            "information_cutoff_is_per_fold": True,
        },
        "fold_identity": {
            "schema": "score_period_dates",
            "format": "fold_YYYYMMDD_YYYYMMDD",
            "stable_when_earlier_folds_are_added": True,
            "legacy_fold_migration_supported": True,
        },
        # Kept for old readers; no longer defines the active OOS boundary.
        "selection_period": {
            "start": str(selection_start.date()),
            "end": str(selection_end.date()),
        },
        "available_history_period": {
            "start": str(training_universe_start.date()),
            "end": str(available_end.date()),
        },
        "training_universe_start_date": str(training_universe_start.date()),
        "fold_months": int(args.fold_months),
        "inner_validation_months": int(args.inner_validation_months),
        "seed": int(args.seed),
        "fold_count": int(len(fold_manifests)),
        "coverage": validation,
        "folds": [_combined_fold_record(args, item) for item in fold_manifests],
        "source_dataset": bundle.summary,
        "source_continuous_target": bundle.target_manifest,
        "lookahead_contract": {
            "every_score_uses_model_not_trained_on_scored_event": True,
            "training_requires_label_eval_end_before_score_start": True,
            "oos_rows_or_target_statistics_used_for_training_or_epoch_selection": False,
            "future_target_in_score_table": False,
        },
        "runtime_eligibility": {
            "eligible_scope": "point_in_time_strategy_replay",
            "eligible": True,
            "strategy_use_requires_model_validation": True,
            "per_score_fold_information_cutoff_required": True,
            "frozen_forward_required": False,
        },
        "fold_execution": {
            "requested_workers": int(requested_fold_workers),
            "effective_workers": int(effective_fold_workers),
            "process_start_method": (
                "spawn" if missing_fold_indexes and effective_fold_workers > 1 else "serial"
            ),
            "execution_only_not_in_fold_fingerprint": True,
        },
        "artifacts": {
            "scores": build_file_manifest(score_path),
            "coverage": build_file_manifest(coverage_path),
        },
        "torch_execution": plan.as_manifest_payload(),
        "elapsed_sec": round(time.perf_counter() - started, 3),
    }
    if (
        str(bundle.profile.training_sample_scope)
        == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
    ):
        manifest["score_eligibility_contract"] = build_score_eligibility_contract(bundle.profile)
    write_json(manifest_path, manifest)
    if compact_console:
        print(
            paint("PIT Scores 完成", "green", enabled=color_enabled, bold=True)
            + f" | folds={len(folds)}"
            + f" | 重用={reused_fold_count}"
            + f" | 遷移重用={migrated_fold_count}"
            + f" | checkpoint重評={rescored_checkpoint_fold_count}"
            + f" | 新建={built_fold_count}"
            + f" | groups={validation['scored_group_count']:,}"
            + f" | coverage={validation['coverage_rate']:.2%}"
            + f" | {format_elapsed(time.perf_counter() - started)}"
        )
    else:
        print(
            render_section(
                paint(
                    "Selection point-in-time scores 完成",
                    "green",
                    enabled=color_enabled,
                    bold=True,
                )
            )
        )
        print(
            render_key_values(
                (
                    ("Folds", len(folds)),
                    ("Reused folds", reused_fold_count),
                    ("Migrated legacy folds", migrated_fold_count),
                    ("Checkpoint-only rescored folds", rescored_checkpoint_fold_count),
                    ("Built folds", built_fold_count),
                    ("Scored groups", f"{validation['scored_group_count']:,}"),
                    (
                        "Coverage",
                        paint(
                            f"{validation['coverage_rate']:.2%}",
                            "green" if float(validation["coverage_rate"]) >= 1.0 - 1e-12 else "yellow",
                            enabled=color_enabled,
                            bold=True,
                        ),
                    ),
                )
            )
        )
    print_artifact_paths(
        (("PIT Scores", score_path), ("PIT manifest", manifest_path), ("PIT coverage", coverage_path)),
        project_root=PROJECT_ROOT,
    )
    return 0


def build_selection_point_in_time_scores(
    *,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
    score_start_date: str | None = None,
    score_end_date: str | None = None,
    fold_months: int | None = None,
    inner_validation_months: int | None = None,
    train_window_months: int | None = None,
    seed: int | None = None,
    resume: bool | None = None,
    fold_workers: int | None = None,
    checkpoint_only: bool = False,
    plan_only: bool = False,
    point_in_time_dir_override: str | None = None,
    allow_stale_source: bool = False,
) -> int:
    """Programmatic PIT producer used by formal workflows.

    CLI parsing remains an adapter owned by this module; callers no longer construct
    command lines or invoke ``main(argv)`` to request a deterministic PIT build.
    Omitted training/runtime knobs continue to come from the canonical workflow/config.
    """

    argv = [
        "--filter-id", str(filter_id),
        "--model-architecture", str(model_architecture),
        "--experiment-profile", str(experiment_profile),
    ]
    if score_start_date is not None:
        argv.extend(["--score-start-date", str(score_start_date)])
    if score_end_date is not None:
        argv.extend(["--score-end-date", str(score_end_date)])
    if fold_months is not None:
        argv.extend(["--fold-months", str(int(fold_months))])
    if inner_validation_months is not None:
        argv.extend(["--inner-validation-months", str(int(inner_validation_months))])
    if train_window_months is not None:
        argv.extend(["--train-window-months", str(int(train_window_months))])
    if seed is not None:
        argv.extend(["--seed", str(int(seed))])
    if resume is not None:
        argv.append("--resume" if bool(resume) else "--no-resume")
    if fold_workers is not None:
        argv.extend(["--fold-workers", str(int(fold_workers))])
    if checkpoint_only:
        argv.append("--checkpoint-only")
    if plan_only:
        argv.append("--plan-only")
    if point_in_time_dir_override is not None:
        argv.extend(["--point-in-time-dir-override", str(point_in_time_dir_override)])
    if allow_stale_source:
        argv.append("--allow-stale-source")
    return _run_point_in_time_scores(parse_args(argv))


def main(argv=None) -> int:
    return _run_point_in_time_scores(parse_args(argv))


__all__ = [
    "FOLD_MANIFEST_FILENAME",
    "FOLD_SCORE_FILENAME",
    "POINT_IN_TIME_SCHEMA_VERSION",
    "build_selection_point_in_time_scores",
    "main",
    "parse_args",
]


if __name__ == "__main__":
    raise SystemExit(main())
