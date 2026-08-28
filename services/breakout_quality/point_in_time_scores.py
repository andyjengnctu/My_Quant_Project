"""Build rolling/cross-fitted Selection point-in-time continuous-ranker scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.display_common import InlineProgress, format_elapsed
from core.training_progress import (
    clear_trainer_pit_progress_context,
    set_trainer_pit_progress_context,
)

from config.breakout_quality import (
    BREAKOUT_QUALITY_ALLOW_TF32,
    BREAKOUT_QUALITY_CONTINUOUS_RANKER_TRAIN_PREFETCH_BATCHES,
    BREAKOUT_QUALITY_CONTINUOUS_RANKER_PREFETCH_WORKERS,
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
    TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_SAFETY_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SAFETY_CONDITIONAL_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_JOINT_MIN_PAIRWISE_RANKING,
    TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    get_breakout_quality_workflow_settings,
)
from filters.breakout_quality.artifacts import build_file_manifest
from filters.breakout_quality.contract import DEFAULT_MODEL_FILENAME
from filters.breakout_quality.ranker_sample_contract import build_score_eligibility_contract
from filters.breakout_quality.models.factory import build_model
from filters.breakout_quality.point_in_time_schedule import (
    FOLD_CALENDAR_ANCHOR,
    build_point_in_time_fold_periods,
    stable_point_in_time_fold_id,
)
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
    predict_conditional_mfe_safety_scores,
    predict_safety_conditional_mfe_scores,
    predict_safety_raw_mfe_joint_min_scores,
    predict_scores,
    resolve_ranker_execution_plan,
    select_epoch,
)

POINT_IN_TIME_SCHEMA_VERSION = 2
AUTO_SCORE_START_VALUE = "auto"
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
OPTIONAL_SCORE_COLUMNS = (
    "primary_mfe_score",
    "conditional_safety_score",
    "raw_safety_score",
    "raw_mfe_score",
    "joint_min_score",
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
        "--single-score-block",
        action="store_true",
        help="將resolved score_start～score_end視為單一PIT fold；用於固定information-cutoff的OOS Test",
    )
    parser.add_argument(
        "--fold-anchor-date",
        default=None,
        help=(
            "可選的fold calendar anchor（YYYY-MM-DD）；省略時沿用canonical 2000-01-01。"
            "OOS Test可指定2021-01-01並搭配--single-score-block形成單一2021→最新forward fold。"
        ),
    )
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
        "--checkpoint-cache-root",
        default=None,
        help=(
            "可選的fitting-identity checkpoint cache根目錄；每個fold只在模型fitting contract完全相同時"
            "重用權重並依目前fold重新評分，不重用來源score。"
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
    if args.fold_anchor_date not in (None, ""):
        _iso_timestamp(args.fold_anchor_date, field_name="fold_anchor_date")
    if args.checkpoint_cache_root not in (None, ""):
        cache_root = Path(str(args.checkpoint_cache_root)).resolve()
        if cache_root.exists() and not cache_root.is_dir():
            raise ValueError(
                "checkpoint-cache-root存在但不是資料夾: "
                f"{cache_root}"
            )
    if args.train_window_months is not None:
        if int(args.train_window_months) <= int(args.inner_validation_months):
            raise ValueError("fixed train-window-months必須大於inner-validation-months")
    if int(args.epochs) < 1 or int(args.batch_size) < 2 or int(args.evaluation_batch_size) < 1:
        raise ValueError("epochs>=1、batch-size>=2、evaluation-batch-size>=1")
    if int(args.train_prefetch_batches) < 0:
        raise ValueError("train-prefetch-batches必須>=0")
    if int(getattr(args, "train_prefetch_workers", BREAKOUT_QUALITY_CONTINUOUS_RANKER_PREFETCH_WORKERS)) < 1:
        raise ValueError("train-prefetch-workers必須>=1")
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
    """Resolve model-history lower bound without leaking optimizer period policy into Daily Universal."""

    summary = dict(getattr(bundle, "summary", {}) or {})
    raw = summary.get("training_universe_start_date")
    daily_scope = (
        str(getattr(getattr(bundle, "profile", None), "training_sample_scope", ""))
        == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
    )
    if raw in {None, ""}:
        if daily_scope:
            raise ValueError(
                "daily ranker summary缺少training_universe_start_date；"
                "Daily Universal不得fallback到optimizer selection_start_date"
            )
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


# Historical private names remain compatibility aliases; the implementation lives in
# filters.breakout_quality.point_in_time_schedule so every caller uses one schedule.
_stable_fold_id = stable_point_in_time_fold_id
_build_fold_periods = build_point_in_time_fold_periods

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
    fold_anchor: pd.Timestamp | None = None,
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
            fold_anchor=fold_anchor,
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
    if (
        str(bundle.profile.training_objective)
        == TRAINING_OBJECTIVE_DAILY_SAFETY_CONDITIONAL_MFE_PAIRWISE_RANKING
    ):
        # Score-output-only identity.  Deliberately excluded from the fitting identity
        # so existing MR-13R checkpoints can be reused and rescored without training.
        payload["score_output_contract"] = {
            "primary": "breakout_quality_score",
            "conditional_mfe": "breakout_quality_score",
            "raw_safety": "raw_safety_score",
        }
    elif (
        str(bundle.profile.training_objective)
        == TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_JOINT_MIN_PAIRWISE_RANKING
    ):
        payload["score_output_contract"] = {
            "primary": "breakout_quality_score",
            "raw_safety": "raw_safety_score",
            "raw_mfe": "raw_mfe_score",
            "joint_min": "joint_min_score",
        }
    return payload


def _validate_minimum_counts(settings, fold_id: str, ids: dict[str, Any]) -> None:
    failed = _minimum_count_failures(settings, ids)
    if failed:
        raise ValueError(f"{fold_id} group coverage不足: {', '.join(failed)}")


def _validate_score_frame(frame: pd.DataFrame, *, fold_contract: dict[str, Any]) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_SCORE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"point-in-time fold score缺少欄位: {missing}")
    retained_columns = list(REQUIRED_SCORE_COLUMNS) + [
        column for column in OPTIONAL_SCORE_COLUMNS if column in frame.columns
    ]
    work = frame[retained_columns].copy()
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
    for score_column in OPTIONAL_SCORE_COLUMNS:
        if score_column not in work.columns:
            continue
        work[score_column] = pd.to_numeric(work[score_column], errors="raise").astype(np.float64)
        values = work[score_column].to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(values).all() or bool(((values < 0.0) | (values > 1.0)).any()):
            raise ValueError(f"point-in-time fold {score_column}超出[0,1]或含非有限值")
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
        "score_output_contract",
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
    """Check model-fitting identity while intentionally ignoring score-only horizon differences."""

    exact_fields = (
        "filter_id",
        "model_architecture",
        "experiment_profile",
        "continuous_target_id",
        "training_label_scope",
        "training_sample_scope",
        "seed",
        "model_information_cutoff",
        "model_spec",
        "experiment_settings",
        "training_settings",
        "source_contract",
        "lookahead_contract",
    )
    if not all(candidate.get(field) == expected_contract.get(field) for field in exact_fields):
        return False

    candidate_planned = candidate.get("planned_periods")
    expected_planned = expected_contract.get("planned_periods")
    if not isinstance(candidate_planned, dict) or not isinstance(expected_planned, dict):
        return False
    # score_end only controls how far an already-fitted model is evaluated.  Any OOS / Rolling
    # folds therefore share one model whenever all fitting periods plus the common
    # score_start/information cutoff are identical; no calendar year is special-cased.
    for field in ("validation_start", "validation_end", "score_start", "history_start"):
        if candidate_planned.get(field) != expected_planned.get(field):
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


def _fold_training_identity_payload(contract: dict[str, Any]) -> dict[str, Any]:
    """Canonical model-fitting identity; score-only horizon fields are excluded."""

    planned = dict(contract.get("planned_periods") or {})
    observed = dict(contract.get("observed_periods") or {})
    group_counts = dict(contract.get("group_counts") or {})
    event_row_counts = dict(contract.get("event_row_counts") or {})
    payload = {
        key: contract.get(key)
        for key in (
            "filter_id",
            "model_architecture",
            "experiment_profile",
            "continuous_target_id",
            "training_label_scope",
            "training_sample_scope",
            "seed",
            "model_information_cutoff",
            "model_spec",
            "experiment_settings",
            "training_settings",
            "source_contract",
            "lookahead_contract",
        )
    }
    payload["planned_periods"] = {
        key: planned.get(key)
        for key in ("validation_start", "validation_end", "score_start", "history_start")
    }
    payload["observed_periods"] = {
        phase: observed.get(phase) for phase in ("inner_train", "validation", "final_refit")
    }
    payload["group_counts"] = {
        phase: group_counts.get(phase) for phase in ("inner_train", "validation", "final_refit")
    }
    payload["event_row_counts"] = {
        phase: event_row_counts.get(phase) for phase in ("inner_train", "validation", "final_refit")
    }
    return payload


def fold_training_identity(contract: dict[str, Any]) -> str:
    return _json_fingerprint(_fold_training_identity_payload(contract))


def _checkpoint_cache_entry_dir(cache_root: Path, fold_contract: dict[str, Any]) -> Path:
    return Path(cache_root).resolve() / fold_training_identity(fold_contract)


_SCORE_ONLY_CHECKPOINT_REUSE_MIGRATION_KINDS = frozenset({
    "expanded_daily_score_universe_checkpoint_reuse",
    "checkpoint_score_regeneration",
    "fitting_identity_checkpoint_reuse",
})


def _is_score_only_checkpoint_reuse_manifest(manifest: dict[str, Any]) -> bool:
    migration = manifest.get("migration")
    return (
        isinstance(migration, dict)
        and migration.get("training_contract_unchanged") is True
        and str(migration.get("kind") or "") in _SCORE_ONLY_CHECKPOINT_REUSE_MIGRATION_KINDS
    )


def _publish_fold_checkpoint_to_cache(
    *, fold_dir: Path, cache_root: Path | None, fold_contract: dict[str, Any]
) -> dict[str, Any] | None:
    if cache_root is None:
        return None
    source_fold_dir = Path(fold_dir).resolve()
    manifest_path = source_fold_dir / FOLD_MANIFEST_FILENAME
    model_path = source_fold_dir / DEFAULT_MODEL_FILENAME
    if not (manifest_path.is_file() and model_path.is_file()):
        raise RuntimeError(f"PIT fold缺少可發布checkpoint: {source_fold_dir.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not _fold_training_contract_is_compatible(
        manifest, expected_contract=fold_contract
    ):
        raise RuntimeError(f"PIT fold fitting identity與目前contract不一致: {source_fold_dir.name}")
    checkpoint_manifest = build_file_manifest(model_path)
    if checkpoint_manifest != dict(manifest.get("artifacts") or {}).get("checkpoint"):
        raise RuntimeError(f"PIT fold checkpoint hash/size與manifest不一致: {source_fold_dir.name}")

    identity = fold_training_identity(fold_contract)
    entry_dir = _checkpoint_cache_entry_dir(Path(cache_root), fold_contract)
    cached_manifest_path = entry_dir / FOLD_MANIFEST_FILENAME
    cached_model_path = entry_dir / DEFAULT_MODEL_FILENAME
    if cached_manifest_path.is_file() and cached_model_path.is_file():
        try:
            cached_manifest = json.loads(cached_manifest_path.read_text(encoding="utf-8"))
            cached_checkpoint = build_file_manifest(cached_model_path)
            cached_identity = fold_training_identity(cached_manifest)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
            cached_checkpoint = {}
            cached_identity = ""
        if cached_identity == identity and cached_checkpoint == checkpoint_manifest:
            return {
                "training_identity": identity,
                "cache_entry": str(entry_dir),
                "reused_existing_cache": True,
            }
        if cached_identity == identity and cached_checkpoint != checkpoint_manifest:
            if _is_score_only_checkpoint_reuse_manifest(manifest):
                # A score-only rescore must never create a second fitted checkpoint.
                # Older code rewrote model.pt only to refresh its embedded fold_contract,
                # which changed the file SHA despite identical model weights.  Canonicalize
                # such resumable folds back to the existing fitting-identity cache bytes.
                restore_path = source_fold_dir / (
                    f".{DEFAULT_MODEL_FILENAME}.cache_restore_{os.getpid()}_{time.time_ns()}"
                )
                try:
                    shutil.copy2(cached_model_path, restore_path)
                    restore_path.replace(model_path)
                finally:
                    if restore_path.exists():
                        restore_path.unlink()
                restored_checkpoint = build_file_manifest(model_path)
                if restored_checkpoint != cached_checkpoint:
                    raise RuntimeError(
                        "fitting identity cache checkpoint還原後hash不一致；"
                        f"identity={identity}, cached={cached_checkpoint.get('sha256')}, "
                        f"restored={restored_checkpoint.get('sha256')}"
                    )
                migration = dict(manifest.get("migration") or {})
                manifest = {
                    **manifest,
                    "migration": {
                        **migration,
                        "canonical_checkpoint_cache_restore": True,
                        "source_checkpoint_sha_preserved": True,
                        "superseded_local_checkpoint": checkpoint_manifest,
                    },
                    "artifacts": {
                        **dict(manifest.get("artifacts") or {}),
                        "checkpoint": restored_checkpoint,
                    },
                }
                write_json(manifest_path, manifest)
                return {
                    "training_identity": identity,
                    "cache_entry": str(entry_dir),
                    "reused_existing_cache": True,
                    "restored_source_fold_checkpoint": True,
                }
            raise RuntimeError(
                "同一fitting identity產生不同checkpoint；"
                f"identity={identity}, cached={cached_checkpoint.get('sha256')}, "
                f"current={checkpoint_manifest.get('sha256')}"
            )

    entry_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = entry_dir.parent / f".{entry_dir.name}.publish_{os.getpid()}_{time.time_ns()}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=False)
    try:
        shutil.copy2(model_path, staging / DEFAULT_MODEL_FILENAME)
        shutil.copy2(manifest_path, staging / FOLD_MANIFEST_FILENAME)
        write_json(
            staging / "cache_manifest.json",
            {
                "schema_version": 1,
                "training_identity": identity,
                "source_fold_id": str(manifest.get("fold_id") or source_fold_dir.name),
                "checkpoint": checkpoint_manifest,
                "published_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        shutil.rmtree(entry_dir, ignore_errors=True)
        staging.replace(entry_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return {
        "training_identity": identity,
        "cache_entry": str(entry_dir),
        "reused_existing_cache": False,
    }


def _reload_fold_manifest_after_checkpoint_cache_sync(
    *, fold_dir: Path, fold_contract: dict[str, Any]
) -> dict[str, Any]:
    """Reload the canonical fold manifest after checkpoint-cache synchronization.

    Cache synchronization may repair a score-only metadata rewrite by restoring the
    canonical fitted checkpoint and rewriting the fold manifest.  The caller must
    aggregate this post-sync manifest rather than an earlier in-memory snapshot.
    """

    fold_dir = Path(fold_dir).resolve()
    manifest_path = fold_dir / FOLD_MANIFEST_FILENAME
    model_path = fold_dir / DEFAULT_MODEL_FILENAME
    score_path = fold_dir / FOLD_SCORE_FILENAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"PIT fold cache同步後manifest無法讀取: {fold_dir.name}"
        ) from exc
    if not isinstance(manifest, dict):
        raise RuntimeError(f"PIT fold cache同步後manifest格式無效: {fold_dir.name}")
    if not _fold_training_contract_is_compatible(
        manifest, expected_contract=fold_contract
    ):
        raise RuntimeError(f"PIT fold cache同步後fitting identity不一致: {fold_dir.name}")
    artifacts = dict(manifest.get("artifacts") or {})
    if build_file_manifest(model_path) != artifacts.get("checkpoint"):
        raise RuntimeError(f"PIT fold cache同步後checkpoint hash/size與manifest不一致: {fold_dir.name}")
    if build_file_manifest(score_path) != artifacts.get("scores"):
        raise RuntimeError(f"PIT fold cache同步後score hash/size與manifest不一致: {fold_dir.name}")
    return manifest


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
        conditional_heads = None
        reverse_conditional_heads = None
        joint_min_heads = None
        if (
            bundle.profile.training_objective
            == TRAINING_OBJECTIVE_DAILY_SAFETY_CONDITIONAL_MFE_PAIRWISE_RANKING
        ):
            reverse_conditional_heads = predict_safety_conditional_mfe_scores(
                torch_module,
                model,
                bundle,
                ids["score_ids"],
                batch_size=int(args.evaluation_batch_size),
                plan=plan,
            )
            scores = reverse_conditional_heads["conditional_mfe"]
        elif (
            bundle.profile.training_objective
            == TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_JOINT_MIN_PAIRWISE_RANKING
        ):
            joint_min_heads = predict_safety_raw_mfe_joint_min_scores(
                torch_module,
                model,
                bundle,
                ids["score_ids"],
                batch_size=int(args.evaluation_batch_size),
                plan=plan,
            )
            scores = joint_min_heads["raw_mfe"]
        elif (
            bundle.profile.training_objective
            == TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_SAFETY_PAIRWISE_RANKING
        ):
            conditional_heads = predict_conditional_mfe_safety_scores(
                torch_module,
                model,
                bundle,
                ids["score_ids"],
                batch_size=int(args.evaluation_batch_size),
                plan=plan,
            )
            scores = conditional_heads["primary_mfe"]
        else:
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
        if reverse_conditional_heads is not None:
            frame["raw_safety_score"] = reverse_conditional_heads["raw_safety"]
        if joint_min_heads is not None:
            frame["raw_safety_score"] = joint_min_heads["raw_safety"]
            frame["raw_mfe_score"] = joint_min_heads["raw_mfe"]
            frame["joint_min_score"] = joint_min_heads["joint_min"]
        if conditional_heads is not None:
            frame["primary_mfe_score"] = conditional_heads["primary_mfe"]
            frame["conditional_safety_score"] = conditional_heads["conditional_safety"]
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

    # Score-output changes are not fitting changes.  Preserve model.pt byte-for-byte;
    # only the score rows and fold manifest are refreshed.  The embedded checkpoint
    # fold_contract may therefore be older in score-only fields, which is intentional
    # because fitting compatibility excludes those fields.
    source_checkpoint_manifest = build_file_manifest(model_path)
    score_path = fold_dir / FOLD_SCORE_FILENAME
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
            "source_checkpoint_sha_preserved": True,
            **(
                {"future_target_required_for_score": False}
                if str(bundle.profile.training_sample_scope)
                == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
                else {}
            ),
        },
        "artifacts": {
            "checkpoint": source_checkpoint_manifest,
            "scores": build_file_manifest(score_path),
        },
    }
    write_json(manifest_path, rescored_manifest)
    return frame, rescored_manifest


def _rescore_fold_from_fitting_checkpoint(
    *,
    source_fold_dir: Path,
    target_fold_dir: Path,
    bundle,
    ids: dict[str, Any],
    fold_contract: dict[str, Any],
    expected_fingerprint: str,
    args,
    torch_module,
    plan,
) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    """Materialize a compatible checkpoint from the shared fitting-identity cache, then rescore locally.

    Only ``model.pt`` and its fold manifest are imported.  The source score file is never
    copied, so OOS single-block scores cannot leak into Rolling annual score artifacts (or
    vice versa).  A staging directory keeps an incompatible source from damaging any
    partially resumable target fold.
    """

    source_fold_dir = Path(source_fold_dir).resolve()
    target_fold_dir = Path(target_fold_dir).resolve()
    if source_fold_dir == target_fold_dir:
        return None
    source_manifest_path = source_fold_dir / FOLD_MANIFEST_FILENAME
    source_model_path = source_fold_dir / DEFAULT_MODEL_FILENAME
    if not (source_manifest_path.is_file() and source_model_path.is_file()):
        return None
    try:
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if not isinstance(source_manifest, dict):
            return None
        if not _fold_training_contract_is_compatible(
            source_manifest, expected_contract=fold_contract
        ):
            return None
        source_checkpoint_manifest = build_file_manifest(source_model_path)
        if source_checkpoint_manifest != dict(source_manifest.get("artifacts") or {}).get(
            "checkpoint"
        ):
            return None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
        return None

    target_fold_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = target_fold_dir.parent / (
        f".{target_fold_dir.name}.fitting_checkpoint_reuse_{os.getpid()}_{time.time_ns()}"
    )
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=False)
    try:
        shutil.copy2(source_model_path, staging / DEFAULT_MODEL_FILENAME)
        shutil.copy2(source_manifest_path, staging / FOLD_MANIFEST_FILENAME)
        rescored = _rescore_fold_from_compatible_checkpoint(
            fold_dir=staging,
            bundle=bundle,
            ids=ids,
            fold_contract=fold_contract,
            expected_fingerprint=expected_fingerprint,
            args=args,
            torch_module=torch_module,
            plan=plan,
        )
        if rescored is None:
            return None
        frame, manifest = rescored
        # The mode-local rescore helper preserves fitted checkpoint bytes.
        # Fitting-identity reuse likewise restores the exact source checkpoint bytes so
        # OOS and Rolling share one identical fitted artifact SHA.
        shutil.copy2(source_model_path, staging / DEFAULT_MODEL_FILENAME)
        manifest = {
            **manifest,
            "migration": {
                "kind": "fitting_identity_checkpoint_reuse",
                "training_contract_unchanged": True,
                "source_fold_id": str(source_manifest.get("fold_id") or ""),
                "source_checkpoint": source_checkpoint_manifest,
                "source_score_reused": False,
                "source_checkpoint_sha_preserved": True,
            },
            "artifacts": {
                **dict(manifest.get("artifacts") or {}),
                "checkpoint": build_file_manifest(staging / DEFAULT_MODEL_FILENAME),
            },
        }
        write_json(staging / FOLD_MANIFEST_FILENAME, manifest)
        shutil.rmtree(target_fold_dir, ignore_errors=True)
        staging.replace(target_fold_dir)
        return frame, manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


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
    conditional_heads = None
    reverse_conditional_heads = None
    joint_min_heads = None
    if (
        bundle.profile.training_objective
        == TRAINING_OBJECTIVE_DAILY_SAFETY_CONDITIONAL_MFE_PAIRWISE_RANKING
    ):
        reverse_conditional_heads = predict_safety_conditional_mfe_scores(
            torch,
            model,
            bundle,
            ids["score_ids"],
            batch_size=int(args.evaluation_batch_size),
            plan=plan,
        )
        scores = reverse_conditional_heads["conditional_mfe"]
    elif (
        bundle.profile.training_objective
        == TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_JOINT_MIN_PAIRWISE_RANKING
    ):
        joint_min_heads = predict_safety_raw_mfe_joint_min_scores(
            torch,
            model,
            bundle,
            ids["score_ids"],
            batch_size=int(args.evaluation_batch_size),
            plan=plan,
        )
        scores = joint_min_heads["raw_mfe"]
    elif (
        bundle.profile.training_objective
        == TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_SAFETY_PAIRWISE_RANKING
    ):
        conditional_heads = predict_conditional_mfe_safety_scores(
            torch,
            model,
            bundle,
            ids["score_ids"],
            batch_size=int(args.evaluation_batch_size),
            plan=plan,
        )
        scores = conditional_heads["primary_mfe"]
    else:
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
    if reverse_conditional_heads is not None:
        frame["raw_safety_score"] = reverse_conditional_heads["raw_safety"]
    if joint_min_heads is not None:
        frame["raw_safety_score"] = joint_min_heads["raw_safety"]
        frame["raw_mfe_score"] = joint_min_heads["raw_mfe"]
        frame["joint_min_score"] = joint_min_heads["joint_min"]
    if conditional_heads is not None:
        frame["primary_mfe_score"] = conditional_heads["primary_mfe"]
        frame["conditional_safety_score"] = conditional_heads["conditional_safety"]
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


def _snapshot_superseded_point_in_time_aggregate(
    *,
    point_in_time_dir: Path,
    manifest_path: Path,
    score_start: pd.Timestamp,
    score_end: pd.Timestamp,
    selection_end: pd.Timestamp,
) -> Path | None:
    """Preserve a combined PIT aggregate before its time semantics narrow.

    Fold directories are immutable/reusable and stay in place.  Only top-level
    combined score/coverage/manifest/audit files are snapshotted.  The helper is
    intentionally path-agnostic so Extending and Fixed-Window stores get the same
    historical-evidence protection.
    """

    if not manifest_path.exists():
        return None
    try:
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    existing_period = dict(existing_manifest.get("score_period") or {})
    existing_start_raw = str(existing_period.get("start") or "").strip()
    existing_end_raw = str(existing_period.get("end") or "").strip()
    existing_start = pd.Timestamp(existing_start_raw) if existing_start_raw else None
    existing_end = pd.Timestamp(existing_end_raw) if existing_end_raw else None

    snapshot_dir = None
    if existing_end is not None and existing_end <= selection_end and score_end > selection_end:
        snapshot_dir = point_in_time_dir / "legacy_selection_snapshot"
    elif (
        existing_start is not None
        and existing_end is not None
        and (existing_start < score_start or existing_end > score_end)
    ):
        snapshot_dir = point_in_time_dir / (
            "historical_aggregate_"
            + existing_start.strftime("%Y%m%d")
            + "_"
            + existing_end.strftime("%Y%m%d")
        )
    if snapshot_dir is None:
        return None

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for source_path in sorted(point_in_time_dir.glob("selection_point_in_time_*")):
        if not source_path.is_file():
            continue
        target_path = snapshot_dir / source_path.name
        if not target_path.exists():
            shutil.copy2(source_path, target_path)
    return snapshot_dir


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
    fold_anchor = (
        None
        if args.fold_anchor_date in (None, "")
        else _iso_timestamp(args.fold_anchor_date, field_name="fold_anchor_date")
    )
    score_start, score_start_resolution = _resolve_score_start(
        args.score_start_date,
        bundle=bundle,
        settings=settings,
        selection_start=selection_start,
        score_end=score_end,
        fold_months=int(args.fold_months),
        validation_months=int(args.inner_validation_months),
        fold_anchor=fold_anchor,
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
    folds = _build_fold_periods(
        score_start, score_end,
        fold_months=int(args.fold_months),
        fold_anchor=fold_anchor,
        single_score_block=bool(args.single_score_block),
    )
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
    fitting_checkpoint_reuse_fold_count = 0
    built_fold_count = 0
    fold_progress = InlineProgress()
    if folds:
        set_trainer_pit_progress_context(
            completed=0,
            total=len(folds),
            active=1,
            emit=True,
        )
    for fold_index, (fold, ids) in enumerate(zip(folds, fold_details), start=1):
        set_trainer_pit_progress_context(
            completed=fold_index - 1,
            total=len(folds),
            active=fold_index,
            emit=True,
        )
        fold_contract = _fold_contract_payload(args, bundle, fold, ids)
        fingerprint = _json_fingerprint(fold_contract)
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
        fitting_checkpoint_reuse = False
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
        if (
            reused is None
            and bool(args.resume)
            and args.checkpoint_cache_root not in (None, "")
        ):
            cache_entry = _checkpoint_cache_entry_dir(
                Path(str(args.checkpoint_cache_root)), fold_contract
            )
            reused = _rescore_fold_from_fitting_checkpoint(
                source_fold_dir=cache_entry,
                target_fold_dir=fold_dir,
                bundle=bundle,
                ids=ids,
                fold_contract=fold_contract,
                expected_fingerprint=fingerprint,
                args=args,
                torch_module=torch,
                plan=plan,
            )
            fitting_checkpoint_reuse = reused is not None
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
            frame, manifest = reused
            if fitting_checkpoint_reuse:
                fitting_checkpoint_reuse_fold_count += 1
            elif rescored_checkpoint:
                rescored_checkpoint_fold_count += 1
            elif migrated:
                migrated_fold_count += 1
            else:
                reused_fold_count += 1
            if not compact_console:
                if fitting_checkpoint_reuse:
                    reuse_label = "重用fitting identity checkpoint並依目前fold重評score"
                elif rescored_checkpoint:
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
        else:
            if bool(args.checkpoint_only):
                raise RuntimeError(
                    "checkpoint-only PIT重建無法完成；fold缺少可重用score/checkpoint或identity不相容: "
                    f"{fold['fold_id']}。請由canonical model-training service以resume模式補齊PIT fold。"
                )
            built_fold_count += 1
            fold_started = time.perf_counter()
            if compact_console:
                fold_progress.update(
                    f"PIT fold {fold_index}/{len(folds)} | {fold['fold_id']} | 訓練並評分"
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
                fold_contract,
                fingerprint,
                torch=torch,
                plan=plan,
            )
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
        if args.checkpoint_cache_root not in (None, ""):
            _publish_fold_checkpoint_to_cache(
                fold_dir=fold_dir,
                cache_root=Path(str(args.checkpoint_cache_root)),
                fold_contract=fold_contract,
            )
            # Cache synchronization may repair model.pt + fold manifest in place.
            # Reload the post-sync canonical manifest before top-level aggregation;
            # otherwise the combined PIT manifest can retain a stale pre-repair SHA.
            manifest = _reload_fold_manifest_after_checkpoint_cache_sync(
                fold_dir=fold_dir,
                fold_contract=fold_contract,
            )
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
        set_trainer_pit_progress_context(
            completed=fold_index,
            total=len(folds),
            active=(fold_index + 1 if fold_index < len(folds) else None),
            emit=True,
        )
    clear_trainer_pit_progress_context()

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

    # Preserve superseded top-level aggregates while reusing immutable annual folds.
    _snapshot_superseded_point_in_time_aggregate(
        point_in_time_dir=point_in_time_dir,
        manifest_path=manifest_path,
        score_start=score_start,
        score_end=score_end,
        selection_end=selection_end,
    )

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
        "score_columns": (
            {
                "primary": "breakout_quality_score",
                "raw_safety": "raw_safety_score",
                "raw_mfe": "raw_mfe_score",
                "joint_min": "joint_min_score",
            }
            if bundle.profile.training_objective
            == TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_JOINT_MIN_PAIRWISE_RANKING
            else
            {
                "primary": "breakout_quality_score",
                "conditional_mfe": "breakout_quality_score",
                "raw_safety": "raw_safety_score",
            }
            if bundle.profile.training_objective
            == TRAINING_OBJECTIVE_DAILY_SAFETY_CONDITIONAL_MFE_PAIRWISE_RANKING
            else {
                "primary": "breakout_quality_score",
                "primary_mfe": "primary_mfe_score",
                "conditional_safety": "conditional_safety_score",
            }
            if bundle.profile.training_objective
            == TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_SAFETY_PAIRWISE_RANKING
            else {"primary": "breakout_quality_score"}
        ),
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
            "fold_calendar_anchor": str(
                (FOLD_CALENDAR_ANCHOR if fold_anchor is None else fold_anchor).date()
            ),
            "score_end_resolution": score_end_mode,
            "information_cutoff_is_per_fold": True,
        },
        "fold_identity": {
            "schema": "score_period_dates",
            "format": "fold_YYYYMMDD_YYYYMMDD",
            "stable_when_earlier_folds_are_added": bool(fold_anchor is None),
            "calendar_anchor": str(
                (FOLD_CALENDAR_ANCHOR if fold_anchor is None else fold_anchor).date()
            ),
            "legacy_fold_migration_supported": bool(fold_anchor is None),
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
        "fold_anchor_date": str(
            (FOLD_CALENDAR_ANCHOR if fold_anchor is None else fold_anchor).date()
        ),
        "single_score_block": bool(args.single_score_block),
        "inner_validation_months": int(args.inner_validation_months),
        "seed": int(args.seed),
        "fold_count": int(len(fold_manifests)),
        "coverage": validation,
        "fold_reuse_summary": {
            "exact_reuse": int(reused_fold_count),
            "legacy_migration": int(migrated_fold_count),
            "local_checkpoint_rescore": int(rescored_checkpoint_fold_count),
            "fitting_identity_checkpoint_reuse": int(fitting_checkpoint_reuse_fold_count),
            "built": int(built_fold_count),
        },
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
            + f" | fitting identity checkpoint={fitting_checkpoint_reuse_fold_count}"
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
                    ("Fitting-identity checkpoint folds", fitting_checkpoint_reuse_fold_count),
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
    fold_anchor_date: str | None = None,
    inner_validation_months: int | None = None,
    train_window_months: int | None = None,
    seed: int | None = None,
    resume: bool | None = None,
    checkpoint_only: bool = False,
    plan_only: bool = False,
    point_in_time_dir_override: str | None = None,
    checkpoint_cache_root: str | None = None,
    allow_stale_source: bool = False,
    single_score_block: bool = False,
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
    if fold_anchor_date is not None:
        argv.extend(["--fold-anchor-date", str(fold_anchor_date)])
    if single_score_block:
        argv.append("--single-score-block")
    if inner_validation_months is not None:
        argv.extend(["--inner-validation-months", str(int(inner_validation_months))])
    if train_window_months is not None:
        argv.extend(["--train-window-months", str(int(train_window_months))])
    if seed is not None:
        argv.extend(["--seed", str(int(seed))])
    if resume is not None:
        argv.append("--resume" if bool(resume) else "--no-resume")
    if checkpoint_only:
        argv.append("--checkpoint-only")
    if plan_only:
        argv.append("--plan-only")
    if point_in_time_dir_override is not None:
        argv.extend(["--point-in-time-dir-override", str(point_in_time_dir_override)])
    if checkpoint_cache_root is not None:
        argv.extend(["--checkpoint-cache-root", str(checkpoint_cache_root)])
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
    "fold_training_identity",
    "main",
    "parse_args",
]


if __name__ == "__main__":
    raise SystemExit(main())
