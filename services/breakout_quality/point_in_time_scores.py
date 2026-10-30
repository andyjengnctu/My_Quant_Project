"""Build rolling/cross-fitted Selection point-in-time continuous-ranker scores."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
import pickle
import tempfile
import shutil
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.display_common import InlineProgress, format_elapsed
from core.file_integrity import (
    atomic_replace_with_retry,
    compute_file_sha256,
    exclusive_file_lock,
    sibling_publication_lock_path,
)
from core.training_progress import (
    clear_trainer_pit_progress_context,
    render_training_unit_progress,
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
    BREAKOUT_QUALITY_PIT_EPOCH_SELECTION_LIGHTWEIGHT_METRICS,
    BREAKOUT_QUALITY_PRELOAD_FEATURE_BANK,
    BREAKOUT_QUALITY_TORCH_DEVICE,
    BREAKOUT_QUALITY_USE_MIXED_PRECISION,
)
from core.breakout_quality_registry import (
    TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    get_continuous_ranker_research_spec,
)
from core.breakout_quality_policy import (
    get_breakout_quality_workflow_settings,
    get_breakout_quality_rolling_test_mode,
)
from core.breakout_quality_runtime import (
    get_continuous_ranker_persisted_score_columns,
    get_continuous_ranker_score_output_columns,
    get_continuous_ranker_training_policy,
)
from filters.breakout_quality.artifacts import build_file_manifest
from filters.breakout_quality.model_source_provenance import source_inventory_matches_current
from filters.breakout_quality.contract import DEFAULT_MODEL_FILENAME
from filters.breakout_quality.pit_fold_provenance import (
    FOLD_MANIFEST_FILENAME, FOLD_SCORE_FILENAME, FOLD_VALIDATION_SCORE_FILENAME,
)
from filters.breakout_quality.ranker_sample_contract import build_score_eligibility_contract
from filters.breakout_quality.ranking_score_store import (
    clear_selection_point_in_time_ranking_contract_cache,
    selection_pit_lookahead_contract,
    validate_continuous_ranker_score_values,
)
from filters.breakout_quality.models.factory import build_model
from filters.breakout_quality.pit_group_selection import (
    AUTO_SCORE_START_VALUE,
    _iso_timestamp,
    _group_event_count,
    _minimum_count_failures,
    _candidate_month_starts,
    _resolve_score_start,
    _fold_group_ids,
)
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
    resolve_filter_model_output_dir,
    resolve_filter_artifact_paths,
)
from filters.breakout_quality.torch_runtime import (
    add_torch_execution_cli_arguments,
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
from services.breakout_quality.fit_repair_transaction import (
    archived_rolling_fit_fold_candidates, selection_pit_build_lock_path,
)
from services.breakout_quality.fitting_data_dependencies import (
    FitDataDependencies, mintrade40_fit_sample_sha256, build_fitting_context_origin, preserve_fitting_migration_chain,
)
from services.breakout_quality.fitted_model_artifacts import (
    FittedModelContract,
    load_fitted_model_training_evidence,
    scientific_torch_execution,
    _fold_training_contract_is_compatible,
    _forward_checkpoint_import_issues,
    _checkpoint_matches_rescore_contract,
    _fold_training_identity_payload,
    fold_training_identity,
    validate_rolling_fold_fitted_checkpoint,
    select_unique_fitted_contract,
    resolve_rolling_fitted_folds_root,
    unresolved_forward_fit_candidate_issue,
)
from services.breakout_quality.continuous_ranker_pipeline import (
    build_checkpoint_payload,
    build_percentile_target,
    build_training_scope_mask,
    fit_final,
    load_continuous_ranker_data,
    predict_score_output_payload,
    resolve_ranker_execution_plan,
    select_epoch,
)

POINT_IN_TIME_SCHEMA_VERSION = 2
REQUIRED_SCORE_COLUMNS = (
    "ticker",
    "date",
    "group_index",
    "breakout_quality_score",
    "fold_id",
    "model_information_cutoff",
)
OPTIONAL_SCORE_COLUMNS = get_continuous_ranker_persisted_score_columns()


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
    add_torch_execution_cli_arguments(
        parser,
        device=BREAKOUT_QUALITY_TORCH_DEVICE,
        mixed_precision=BREAKOUT_QUALITY_USE_MIXED_PRECISION,
        mixed_precision_dtype=BREAKOUT_QUALITY_MIXED_PRECISION_DTYPE,
        deterministic_algorithms=BREAKOUT_QUALITY_DETERMINISTIC_ALGORITHMS,
        allow_tf32=BREAKOUT_QUALITY_ALLOW_TF32,
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
        "--fit-source-pit-dir", action="append", default=[],
        help=(
            "已發布的其他 OOS／Rolling PIT 根目錄；僅用作逐 Fold FIT 來源候選，"
            "通過 canonical fitting provenance 驗證才載入 checkpoint，從不複製來源 Scores。"
        ),
    )
    parser.add_argument(
        "--checkpoint-import-model-dir",
        default=None,
        help=(
            "可選的既有Forward model目錄；只在其canonical training evidence與目前fold fitting identity"
            "完全相容時，將原checkpoint位元組登錄至shared fitting cache供本fold重評score。"
        ),
    )
    parser.add_argument(
        "--forward-fit-candidate-model-dir",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--checkpoint-import-report-path",
        default=None,
        help="checkpoint-import-model-dir對應的canonical Forward report JSON。",
    )
    parser.add_argument(
        "--allow-stale-source",
        action="store_true",
        help="只供離線重現；預設要求來源CSV inventory與dataset一致",
    )
    parser.add_argument(
        "--strategy-compare-profile-id",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--strategy-compare-source-id",
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)



def build_rolling_first_fold_fit_proof(*, bundle, forward_args, score_start: str):
    """Supply PIT-native fold facts to the FIT owner without a reverse import.

    This is a proof provider, not a checkpoint locator or REUSE authorization.
    Capturing one PIT argument contract also avoids recomputing mode defaults
    for each candidate. The original fold ID/membership/manifest formulas remain
    unchanged and are shared with the ordinary Rolling producer.
    """
    pit_args = parse_args([
        "--filter-id", str(forward_args.filter_id),
        "--model-architecture", str(forward_args.model_architecture),
        "--experiment-profile", str(forward_args.experiment_profile),
        "--seed", str(int(forward_args.seed)),
        "--score-start-date", str(score_start),
        "--inner-validation-months", str(int(forward_args.inner_validation_months)),
    ])

    def native_fold_proof(fold):
        ids = _fold_group_ids(
            bundle, fold,
            validation_months=int(pit_args.inner_validation_months),
            train_window_months=None,
        )
        return ids, _fold_contract_payload(pit_args, bundle, fold, ids)

    return pit_args, native_fold_proof


def _validate_args(args: argparse.Namespace) -> None:
    # Explicit CLI overrides are supported for reproducible research.  The loaded experiment
    # profile and model spec are validated by the shared continuous-ranker pipeline.
    strategy_profile_id = str(
        getattr(args, "strategy_compare_profile_id", "") or ""
    ).strip()
    strategy_source_id = str(
        getattr(args, "strategy_compare_source_id", "") or ""
    ).strip()
    if bool(strategy_profile_id) != bool(strategy_source_id):
        raise ValueError(
            "strategy-compare-profile-id與strategy-compare-source-id必須同時設定"
        )
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
    import_model_dir = str(getattr(args, "checkpoint_import_model_dir", "") or "").strip()
    import_report_path = str(getattr(args, "checkpoint_import_report_path", "") or "").strip()
    if bool(import_model_dir) != bool(import_report_path):
        raise ValueError(
            "checkpoint-import-model-dir與checkpoint-import-report-path必須同時設定"
        )
    if import_model_dir and args.checkpoint_cache_root in (None, ""):
        raise ValueError("checkpoint import需要同時設定checkpoint-cache-root")
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











def _score_output_policy(bundle):
    """Resolve the runtime-owned score-output capability for this profile."""

    training_policy = get_continuous_ranker_training_policy(
        str(bundle.profile.training_objective)
    )
    return training_policy.score_output_policy


def _score_output_capability(bundle) -> dict[str, Any] | None:
    """Compatibility view of the canonical runtime score-output contract."""

    policy = _score_output_policy(bundle)
    if policy is None:
        return None
    return {
        "output_head": str(policy.output_head),
        "columns": policy.manifest_columns(),
    }


def _score_output_columns(bundle) -> dict[str, str]:
    return get_continuous_ranker_score_output_columns(
        str(bundle.profile.training_objective)
    )


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
        "source_contract": FitDataDependencies.from_bundle(bundle).rolling_source(
            fit_cutoff=cutoff,
            daily_training_universe_start_date=(
                str(training_universe_start.date())
                if str(bundle.profile.training_sample_scope)
                == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS else None
            ),
        ),
        "lookahead_contract": {
            "training_requires_label_eval_end_before_score_start": True,
            "validation_requires_label_eval_end_before_score_start": True,
            "score_period_used_for_training_or_epoch_selection": False,
            "oos_used_for_training_or_epoch_selection": False,
        },
    }

    fit_samples = mintrade40_fit_sample_sha256(bundle, ids['final_ids'])
    if fit_samples is not None:
        payload['source_contract']['formal_mintrade40_fit_samples_sha256'] = fit_samples

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
    score_output_capability = _score_output_capability(bundle)
    if score_output_capability is not None:
        # Score-output-only identity. Deliberately excluded from fitting identity so
        # compatible checkpoints are reused and rescored without model training.
        payload["score_output_contract"] = dict(score_output_capability["columns"])
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
    validate_continuous_ranker_score_values(
        work["breakout_quality_score"].to_numpy(dtype=np.float64, copy=False),
        experiment_profile=fold_contract.get("experiment_profile"),
        column="breakout_quality_score", label="point-in-time fold score",
    )
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
    validation_score_path = fold_dir / FOLD_VALIDATION_SCORE_FILENAME
    if not (
        manifest_path.is_file()
        and model_path.is_file()
        and score_path.is_file()
        and validation_score_path.is_file()
    ):
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
        if build_file_manifest(validation_score_path) != artifacts.get("validation_scores"):
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


def _checkpoint_cache_entry_dir(cache_root: Path, fold_contract: dict[str, Any]) -> Path:
    return Path(cache_root).resolve() / fold_training_identity(fold_contract)



def _unverified_fit_checkpoint_reasons(
    *, fold_dir: Path, fold_contract: dict[str, Any], args, torch_module,
) -> tuple[str, ...]:
    """Explain existing checkpoint blockers without treating them as permission to refit.

    This is a read-only diagnostic after all canonical REUSE/score-refresh routes
    failed. The FIT owner alone decides reuse; differences here never waive its
    hash, training identity, source lineage, or epoch proof.
    """
    candidates = [("Rolling fold", Path(fold_dir))]
    if args.checkpoint_cache_root not in (None, ""):
        candidates.append((
            "shared FIT cache",
            _checkpoint_cache_entry_dir(Path(str(args.checkpoint_cache_root)), fold_contract),
        ))
    def differing_fit_fields(actual: dict[str, Any], expected: dict[str, Any]) -> list[str]:
        # Only explain differences in FIT-owner-selected identity fields.
        different: list[str] = []
        for key, expected_value in expected.items():
            actual_value = actual.get(key)
            if actual_value == expected_value:
                continue
            if isinstance(actual_value, dict) and isinstance(expected_value, dict):
                different.extend(
                    f"{key}.{child}" for child in expected_value
                    if actual_value.get(child) != expected_value[child]
                )
            else:
                different.append(key)
        return different

    reasons: list[str] = []
    for label, directory in candidates:
        model_path = directory / DEFAULT_MODEL_FILENAME
        if not model_path.is_file():
            continue
        manifest_path = directory / FOLD_MANIFEST_FILENAME
        if not manifest_path.is_file():
            reasons.append(f"{label}: manifest missing")
            continue
        try:
            candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(candidate, dict):
                reasons.append(f"{label}: manifest is not a mapping")
                continue
            recorded = dict(candidate.get("artifacts") or {}).get("checkpoint")
            if build_file_manifest(model_path) != recorded:
                reasons.append(f"{label}: checkpoint hash/size differs from manifest")
                continue
            # Owner-derived identity payload, not an independent fitting policy.
            # Exclude score-only horizon and score-output capability just as
            # fold_training_identity does, to avoid falsely blaming SCORE drift.
            actual_fit = _fold_training_identity_payload(candidate)
            expected_fit = _fold_training_identity_payload(fold_contract)
            different = differing_fit_fields(actual_fit, expected_fit)
            if different:
                reasons.append(f"{label}: fitting identity fields differ ({', '.join(different[:8])})")
                continue
            checkpoint = torch_module.load(model_path, map_location="cpu", weights_only=True)
            if not isinstance(checkpoint, dict):
                reasons.append(f"{label}: checkpoint payload is not a mapping")
                continue
            embedded = checkpoint.get("fold_contract")
            if isinstance(embedded, dict):
                embedded_fit = _fold_training_identity_payload(embedded)
                different = differing_fit_fields(embedded_fit, expected_fit)
                if different:
                    reasons.append(
                        f"{label}: embedded fitting identity differs ({', '.join(different[:8])})"
                    )
                    continue
            elif not isinstance(candidate.get("fitting_origin"), dict):
                reasons.append(f"{label}: embedded FIT contract/origin proof missing")
                continue
            if checkpoint.get("model_spec") != fold_contract.get("model_spec"):
                reasons.append(f"{label}: checkpoint model_spec differs")
            elif (checkpoint.get("selected_epoch") is not None
                  and int(checkpoint["selected_epoch"]) != int(candidate.get("selected_epoch") or 0)):
                reasons.append(f"{label}: checkpoint selected_epoch differs")
            else:
                reasons.append(
                    f"{label}: FIT metadata matches; checkpoint scoring/provenance verification rejected"
                )
        except (
            OSError, ValueError, TypeError, KeyError, RuntimeError, UnicodeError,
            pickle.UnpicklingError, EOFError, zipfile.BadZipFile,
        ) as exc:
            reasons.append(f"{label}: {type(exc).__name__} while checking checkpoint evidence")
    return tuple(reasons)



def _import_forward_checkpoint_to_fitting_cache_unlocked(
    *,
    source_model_dir: Path,
    source_report_path: Path,
    cache_root: Path,
    fold_contract: dict[str, Any],
    bundle,
    torch_module,
    execution_plan,
) -> dict[str, Any] | None:
    """Publish an already-fitted Forward checkpoint into the canonical fitting cache.

    Returns ``None`` when the Forward artifact is simply not the fit required by this fold.
    Corruption or a conflicting checkpoint for the same fitting identity fails fast.
    """

    source_model_dir = Path(source_model_dir).resolve()
    source_report_path = Path(source_report_path).resolve()
    source_manifest_path = source_model_dir / FOLD_MANIFEST_FILENAME
    source_model_path = source_model_dir / DEFAULT_MODEL_FILENAME
    if not (source_manifest_path.is_file() and source_model_path.is_file() and source_report_path.is_file()):
        return None
    try:
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
        if not isinstance(source_manifest, dict) or not isinstance(source_report, dict):
            return None
        checkpoint_manifest = build_file_manifest(source_model_path)
        checkpoint = torch_module.load(source_model_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            return None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError, RuntimeError):
        return None

    source_fitting_evidence, _fitting_evidence_reason = load_fitted_model_training_evidence(
        model_dir=source_model_dir,
        report_path=source_report_path,
    )
    if source_fitting_evidence is None:
        return None

    issues = _forward_checkpoint_import_issues(
        source_manifest=source_manifest,
        source_report=source_report,
        source_fitting_evidence=source_fitting_evidence,
        checkpoint=checkpoint,
        checkpoint_manifest=checkpoint_manifest,
        fold_contract=fold_contract,
        bundle=bundle,
        execution_plan=execution_plan,
        project_root=PROJECT_ROOT,
        source_model_dir=source_model_dir,
    )
    if issues:
        # For the canonical first Rolling fold, Forward and Rolling share the
        # same score start and therefore are expected to share the exact fit.
        # Never silently refit that fold after advertising a validated Forward
        # checkpoint candidate; surface the violated identity field instead.
        source_oos_start = str(dict(source_report.get("split_report") or {}).get("oos_start_date") or "")
        fold_score_start = str(dict(fold_contract.get("planned_periods") or {}).get("score_start") or "")
        if source_oos_start and source_oos_start == fold_score_start:
            raise RuntimeError(
                "Forward OOS checkpoint與第一個Rolling fold應為同一fitting identity但驗證失敗: "
                + "; ".join(issues[:12])
            )
        return None

    # Source publication, execution and checkpoint bytes were all verified above.
    # Capture its original weight-affecting context before returning an existing
    # cache entry, too: prior importers left otherwise valid caches unwitnessed.
    forward_training = dict(source_report.get("training") or {})
    source_origin = build_fitting_context_origin(
        profile=str(fold_contract["experiment_profile"]),
        target=dict(source_manifest.get("source_continuous_target") or {}),
        checkpoint=checkpoint_manifest, kind="forward_import",
        forward_training=forward_training,
        forward_evidence={
            "training_settings": dict(source_fitting_evidence.training_settings),
            "torch_execution": dict(source_fitting_evidence.torch_execution),
            "selected_epoch": int(source_fitting_evidence.selected_epoch),
        },
    )
    forward_cache_metadata = {
        "selected_epoch": int(source_manifest["selected_epoch"]),
        "epoch_selection": dict(forward_training.get("epoch_selection") or {}),
        "final_refit_history": list(forward_training.get("final_refit_history") or []),
        "torch_execution": dict(source_fitting_evidence.torch_execution),
    }

    identity = fold_training_identity(fold_contract)
    entry_dir = _checkpoint_cache_entry_dir(Path(cache_root), fold_contract)
    cached_manifest_path = entry_dir / FOLD_MANIFEST_FILENAME
    cached_model_path = entry_dir / DEFAULT_MODEL_FILENAME
    if entry_dir.exists():
        if entry_dir.is_symlink() or not entry_dir.is_dir():
            raise RuntimeError(f"fitting identity cache entry is not a plain directory: identity={identity}")
        if not (cached_manifest_path.is_file() and cached_model_path.is_file()):
            raise RuntimeError(f"fitting identity cache entry不完整: identity={identity}")
        try:
            cached_manifest = json.loads(cached_manifest_path.read_text(encoding="utf-8"))
            cached_checkpoint = build_file_manifest(cached_model_path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
            raise RuntimeError(f"fitting identity cache entry無法驗證: identity={identity}") from exc
        if not isinstance(cached_manifest, dict) or not _fold_training_contract_is_compatible(
            cached_manifest, expected_contract=fold_contract
        ):
            raise RuntimeError(f"fitting identity cache contract衝突: identity={identity}")
        if cached_checkpoint != dict(dict(cached_manifest.get("artifacts") or {}).get("checkpoint") or {}):
            raise RuntimeError(f"fitting identity cache checkpoint manifest mismatch: identity={identity}")
        if cached_checkpoint != checkpoint_manifest:
            raise RuntimeError(
                "同一fitting identity已有不同checkpoint；"
                f"identity={identity}, cached={cached_checkpoint.get('sha256')}, "
                f"forward={checkpoint_manifest.get('sha256')}"
            )
        if source_origin is not None:
            existing_origin = cached_manifest.get("fitting_origin")
            migration = cached_manifest.get("migration")
            imported_cache = (
                isinstance(migration, dict)
                and migration.get("kind") == "forward_model_fitting_identity_import"
            )
            if imported_cache and existing_origin is not None and existing_origin != source_origin:
                raise RuntimeError(
                    "same FIT/cache checkpoint has conflicting Forward context origin: "
                    f"identity={identity}"
                )
            if imported_cache and existing_origin is None:
                # Upgrade only an *existing Forward-import cache* whose bytes
                # match the fully verified original Forward checkpoint. This is
                # not a historical inference from the cache alone. Keep native
                # Rolling caches, original weights and the FitKey unchanged.
                for key, value in forward_cache_metadata.items():
                    if key in cached_manifest and cached_manifest[key] is not None and cached_manifest[key] != value:
                        raise RuntimeError(
                            "same FIT/cache has conflicting original Forward evidence: "
                            f"identity={identity}, field={key}"
                        )
                upgraded = {**cached_manifest, **forward_cache_metadata, "fitting_origin": source_origin}
                temporary = cached_manifest_path.with_name(
                    f".{cached_manifest_path.name}.verified_origin_{os.getpid()}_{time.time_ns()}"
                )
                try:
                    write_json(temporary, upgraded)
                    temporary.replace(cached_manifest_path)
                finally:
                    if temporary.exists():
                        temporary.unlink()
        return {
            "training_identity": identity,
            "cache_entry": str(entry_dir),
            "reused_existing_cache": True,
        }

    cache_manifest = {
        **fold_contract,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        **forward_cache_metadata,
        **({"fitting_origin": source_origin} if source_origin is not None else {}),
        "migration": {
            "kind": "forward_model_fitting_identity_import",
            "training_contract_unchanged": True,
            "source_score_reused": False,
            "source_checkpoint_sha_preserved": True,
            "source_forward_checkpoint": checkpoint_manifest,
        },
        "artifacts": {"checkpoint": checkpoint_manifest},
    }
    entry_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = entry_dir.parent / f".{entry_dir.name}.forward_import_{os.getpid()}_{time.time_ns()}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=False)
    try:
        shutil.copy2(source_model_path, staging / DEFAULT_MODEL_FILENAME)
        write_json(staging / FOLD_MANIFEST_FILENAME, cache_manifest)
        if build_file_manifest(staging / DEFAULT_MODEL_FILENAME) != checkpoint_manifest:
            raise RuntimeError(f"Forward cache staged checkpoint changed: identity={identity}")
        write_json(
            staging / "cache_manifest.json",
            {
                "schema_version": 1,
                "training_identity": identity,
                "source_kind": "forward_model_fitting_identity_import",
                "checkpoint": checkpoint_manifest,
                "published_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        if entry_dir.exists():
            raise RuntimeError(f"fitting identity cache entry appeared during Forward import: identity={identity}")
        staging.replace(entry_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return {
        "training_identity": identity,
        "cache_entry": str(entry_dir),
        "reused_existing_cache": False,
    }


def _import_forward_checkpoint_to_fitting_cache(
    *, source_model_dir: Path, source_report_path: Path, cache_root: Path,
    fold_contract: dict[str, Any], bundle, torch_module, execution_plan,
) -> dict[str, Any] | None:
    """One FitKey writer at a time; source/evidence checks stay in the sole FIT owner."""
    entry = _checkpoint_cache_entry_dir(Path(cache_root), fold_contract)
    with exclusive_file_lock(
        sibling_publication_lock_path(entry),
        conflict_message="Another shared FIT checkpoint publication is active",
    ):
        return _import_forward_checkpoint_to_fitting_cache_unlocked(
            source_model_dir=source_model_dir,
            source_report_path=source_report_path,
            cache_root=cache_root,
            fold_contract=fold_contract,
            bundle=bundle, torch_module=torch_module,
            execution_plan=execution_plan,
        )


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


def _publish_fold_checkpoint_to_cache_unlocked(
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
    if entry_dir.exists():
        if entry_dir.is_symlink() or not entry_dir.is_dir():
            raise RuntimeError(f"fitting identity cache entry is not a plain directory: identity={identity}")
        if not (cached_manifest_path.is_file() and cached_model_path.is_file()):
            raise RuntimeError(f"fitting identity cache entry不完整: identity={identity}")
        try:
            cached_manifest = json.loads(cached_manifest_path.read_text(encoding="utf-8"))
            cached_checkpoint = build_file_manifest(cached_model_path)
            cached_identity = fold_training_identity(cached_manifest)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
            cached_checkpoint = {}
            cached_identity = ""
        if cached_identity == identity and cached_checkpoint != dict(dict(cached_manifest.get("artifacts") or {}).get("checkpoint") or {}):
            raise RuntimeError(f"fitting identity cache checkpoint manifest mismatch: identity={identity}")
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
        if build_file_manifest(staging / DEFAULT_MODEL_FILENAME) != checkpoint_manifest:
            raise RuntimeError(f"PIT cache staged checkpoint changed: identity={identity}")
        staged_manifest = json.loads((staging / FOLD_MANIFEST_FILENAME).read_text(encoding="utf-8"))
        if (not isinstance(staged_manifest, dict)
            or not _fold_training_contract_is_compatible(staged_manifest, expected_contract=fold_contract)
            or dict(dict(staged_manifest.get("artifacts") or {}).get("checkpoint") or {}) != checkpoint_manifest):
            raise RuntimeError(f"PIT cache staged manifest/weights changed: identity={identity}")
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
        # Do not destroy an existing FIT identity, including incomplete or
        # concurrently published evidence. The per-FitKey OS lock serializes
        # cooperating producers; rename is the first visible publication.
        if entry_dir.exists():
            raise RuntimeError(f"fitting identity cache entry appeared during publication: identity={identity}")
        staging.replace(entry_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return {
        "training_identity": identity,
        "cache_entry": str(entry_dir),
        "reused_existing_cache": False,
    }


def _publish_fold_checkpoint_to_cache(
    *, fold_dir: Path, cache_root: Path | None, fold_contract: dict[str, Any],
) -> dict[str, Any] | None:
    """Single FIT cache publisher for native fold checkpoints and score-only reuse."""
    if cache_root is None:
        return None
    entry = _checkpoint_cache_entry_dir(Path(cache_root), fold_contract)
    with exclusive_file_lock(
        sibling_publication_lock_path(entry),
        conflict_message="Another shared FIT checkpoint publication is active",
    ):
        return _publish_fold_checkpoint_to_cache_unlocked(
            fold_dir=fold_dir, cache_root=cache_root, fold_contract=fold_contract,
        )


def publish_verified_native_first_fold_cache(
    *, rolling_dir: Path, score_start: str, cache_root: Path,
    expected_checkpoint_sha256: str,
) -> str:
    """PIT producer publication after an isolated canonical FIT rebuild.

    The old first-fold shared cache must already be transactionally isolated.
    Normal force-rebuild uses a temporary cache to prevent accidental mutation;
    after complete Rolling/Forward identity verification the repairing caller
    restores this one new native FitKey entry through the canonical publisher.
    """
    from services.breakout_quality.fitted_model_artifacts import _fit_repair_first_fold

    path, manifest = _fit_repair_first_fold(rolling_dir=rolling_dir, score_start=score_start)
    fold_dir = path.parent
    actual = build_file_manifest(fold_dir / DEFAULT_MODEL_FILENAME)
    origin = manifest.get("fitting_origin")
    if (not isinstance(origin, dict) or origin.get("kind") != "rolling_native"
        or dict(origin.get("checkpoint") or {}) != actual
        or dict(dict(manifest.get("artifacts") or {}).get("checkpoint") or {}) != actual
        or actual.get("sha256") != str(expected_checkpoint_sha256)):
        raise RuntimeError("Rebuilt native first-fold FIT has no verified checkpoint origin")
    identity = fold_training_identity(manifest)
    entry = Path(cache_root).resolve() / identity
    # The repair-specific "must be absent" rule and canonical publication
    # must be one lock transaction; a late concurrent writer is not a REUSE.
    with exclusive_file_lock(
        sibling_publication_lock_path(entry),
        conflict_message="Another shared FIT checkpoint publication is active",
    ):
        if entry.exists():
            raise RuntimeError("New FIT cache identity already occupied during transaction")
        # All score, target, and sample semantics are from the actual PIT Fold
        # contract; no Research-side cache writer or synthetic FitKey is introduced.
        _publish_fold_checkpoint_to_cache_unlocked(
            fold_dir=fold_dir, cache_root=Path(cache_root), fold_contract=manifest,
        )
        cached = entry / FOLD_MANIFEST_FILENAME
        cached_model = entry / DEFAULT_MODEL_FILENAME
        if (not cached.is_file() or not cached_model.is_file()
            or build_file_manifest(cached_model) != actual
            or fold_training_identity(json.loads(cached.read_text(encoding="utf-8"))) != identity):
            raise RuntimeError("Rebuilt native shared FIT cache publication failed verification")
    return identity


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
    validation_score_path = fold_dir / FOLD_VALIDATION_SCORE_FILENAME
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
    if build_file_manifest(validation_score_path) != artifacts.get("validation_scores"):
        raise RuntimeError(f"PIT fold cache同步後validation score hash/size與manifest不一致: {fold_dir.name}")
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
    verified_archived_fit: bool = False,
    fit_validation_errors: list[str] | None = None,
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
        exact_fitting_schema = _fold_training_contract_is_compatible(
            manifest, expected_contract=fold_contract
        )
        if not exact_fitting_schema and not verified_archived_fit:
            # FIT source records may have a proven, non-fitting-only schema/cache
            # difference. The old strict raw-JSON equality is not the FIT
            # authority: validate *all* local (native or archived) checkpoint
            # candidates with the canonical FIT owner before score-only reuse.
            # Missing/changed consumed inputs, provenance or checkpoint bytes
            # continue to fail closed; no optimiser refit is authorised here.
            contract, reason = validate_rolling_fold_fitted_checkpoint(
                manifest_path=manifest_path, manifest=manifest,
                expected=fold_contract,
                expected_execution=scientific_torch_execution(plan),
                bundle=bundle, torch_module=torch_module,
                project_root=PROJECT_ROOT,
            )
            verified_archived_fit = contract is not None
            if not verified_archived_fit:
                if fit_validation_errors is not None and reason:
                    fit_validation_errors.append(str(reason))
                return None

        checkpoint = torch_module.load(model_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            return None
        checkpoint_compatible = _checkpoint_matches_rescore_contract(
            checkpoint=checkpoint,
            manifest=manifest,
            fold_contract=fold_contract,
            bundle=bundle,
            verified_archived_fit=verified_archived_fit,
            project_root=PROJECT_ROOT,
        )
        if not checkpoint_compatible and not verified_archived_fit:
            # A prior FIT-owner-approved score refresh updates the *manifest*
            # source snapshot but intentionally preserves the original Fold
            # contract embedded in model.pt. A later SCORE refresh therefore
            # needs the same FIT proof even when the outer manifest now matches.
            contract, reason = validate_rolling_fold_fitted_checkpoint(
                manifest_path=manifest_path, manifest=manifest,
                expected=fold_contract,
                expected_execution=scientific_torch_execution(plan),
                bundle=bundle, torch_module=torch_module,
                project_root=PROJECT_ROOT,
            )
            if contract is None:
                if fit_validation_errors is not None and reason:
                    fit_validation_errors.append(str(reason))
                return None
            checkpoint_compatible = _checkpoint_matches_rescore_contract(
                checkpoint=checkpoint,
                manifest=manifest,
                fold_contract=fold_contract,
                bundle=bundle,
                verified_archived_fit=True,
                project_root=PROJECT_ROOT,
            )
        if not checkpoint_compatible:
            return None
        expected_shape = (
            int(bundle.feature_bank.shape[1]),
            int(bundle.feature_bank.shape[2]),
            int(bundle.group_context.shape[1]),
        )

        model = build_model(
            expected_shape[1],
            expected_shape[2],
            model_spec=checkpoint["model_spec"],
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(plan.device)
        model.eval()
        scores, score_extras = _predict_primary_score_payload(
            torch_module, model, bundle, ids["score_ids"], args=args, plan=plan
        )
        validation_scores, validation_extras = _predict_primary_score_payload(
            torch_module, model, bundle, ids["validation_ids"], args=args, plan=plan
        )
        frame = _build_fold_score_frame(
            bundle, ids["score_ids"], scores, score_extras,
            fold_id=str(fold_contract["fold_id"]),
            cutoff=fold_contract["model_information_cutoff"],
        )
        validation_frame = _build_fold_score_frame(
            bundle, ids["validation_ids"], validation_scores, validation_extras,
            fold_id=str(fold_contract["fold_id"]),
            cutoff=fold_contract["model_information_cutoff"],
        )
        frame["fold_id"] = str(fold_contract["fold_id"])
        frame["model_information_cutoff"] = fold_contract["model_information_cutoff"]
        frame = _validate_score_frame(frame, fold_contract=fold_contract)
        validation_frame = _validate_validation_score_frame(
            validation_frame, ids=ids["validation_ids"], fold_id=str(fold_contract["fold_id"])
        )
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
    validation_score_path = fold_dir / FOLD_VALIDATION_SCORE_FILENAME
    frame.to_csv(score_path, index=False, encoding="utf-8-sig")
    validation_frame.to_csv(validation_score_path, index=False, encoding="utf-8-sig")
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
        "migration": preserve_fitting_migration_chain(
            manifest, {
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
        }, checkpoint=source_checkpoint_manifest),
        "artifacts": {
            "checkpoint": source_checkpoint_manifest,
            "scores": build_file_manifest(score_path),
            "validation_scores": build_file_manifest(validation_score_path),
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
    verified_archive_binding: dict[str, Any] | None = None,
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
    target_fold_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = target_fold_dir.parent / (
        f".{target_fold_dir.name}.fitting_checkpoint_reuse_{os.getpid()}_{time.time_ns()}"
    )
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=False)

    # A shared FitKey can be concurrently archived by FIT Repair, or published
    # by another PIT mode. Snapshot source manifest and exact model bytes under
    # the same short OS lock as those writers, then release before slow rescore.
    shared_root = getattr(args, "checkpoint_cache_root", None)
    is_shared_cache = (
        shared_root not in (None, "")
        and source_fold_dir == _checkpoint_cache_entry_dir(Path(shared_root), fold_contract)
    )
    source_guard = (
        exclusive_file_lock(
            sibling_publication_lock_path(source_fold_dir),
            conflict_message="Another shared FIT checkpoint publication is active",
        )
        if is_shared_cache else nullcontext()
    )
    try:
        with source_guard:
            if not (source_manifest_path.is_file() and source_model_path.is_file()):
                return None
            try:
                source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
                if not isinstance(source_manifest, dict):
                    return None
                if (verified_archive_binding is None and not _fold_training_contract_is_compatible(
                    source_manifest, expected_contract=fold_contract
                )):
                    return None
                source_checkpoint_manifest = build_file_manifest(source_model_path)
                if source_checkpoint_manifest != dict(source_manifest.get("artifacts") or {}).get(
                    "checkpoint"
                ):
                    return None
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
                return None
            shutil.copy2(source_model_path, staging / DEFAULT_MODEL_FILENAME)
            if verified_archive_binding is None:
                shutil.copy2(source_manifest_path, staging / FOLD_MANIFEST_FILENAME)
            else:
                # Verified archived FITs may project evaluation-specific fields,
                # but preserve their source model's exact checkpoint bytes.
                write_json(staging / FOLD_MANIFEST_FILENAME, {
                    **source_manifest, **fold_contract,
                    "archived_rolling_fit_source": dict(verified_archive_binding),
                    "artifacts": dict(source_manifest.get("artifacts") or {}),
                })
            if build_file_manifest(staging / DEFAULT_MODEL_FILENAME) != source_checkpoint_manifest:
                raise RuntimeError("Fitting checkpoint changed while snapshotting its source")
        # Source data is immutable for this local rescore from now on.
        rescored = _rescore_fold_from_compatible_checkpoint(
            fold_dir=staging,
            bundle=bundle,
            ids=ids,
            fold_contract=fold_contract,
            expected_fingerprint=expected_fingerprint,
            args=args,
            torch_module=torch_module,
            plan=plan,
            verified_archived_fit=verified_archive_binding is not None,
        )
        if rescored is None:
            return None
        frame, manifest = rescored
        # The mode-local rescore helper preserves fitted checkpoint bytes.
        # Fitting-identity reuse likewise restores the exact source checkpoint bytes so
        # OOS and Rolling share one identical fitted artifact SHA.
        manifest = {
            **manifest,
            "migration": preserve_fitting_migration_chain(
                source_manifest, {
                "kind": "fitting_identity_checkpoint_reuse",
                "training_contract_unchanged": True,
                "source_fold_id": str(source_manifest.get("fold_id") or ""),
                "source_checkpoint": source_checkpoint_manifest,
                "source_score_reused": False,
                "source_checkpoint_sha_preserved": True,
            }, checkpoint=source_checkpoint_manifest),
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


def _verified_archived_rolling_fit(
    *, point_in_time_dir: Path, fold_contract: dict[str, Any],
    bundle, torch_module, plan, project_root: Path,
) -> tuple[Path, dict[str, Any]] | None:
    """Resolve an archived fold as FIT evidence, never as permission to retrain.

    A Forward-origin repair can move the *entire* Rolling six-fold tree into
    historical_fit_repairs. Each fold is independently verified using the
    canonical FIT owner and its preserved aggregate/context binding. The old
    score rows are not used, and a conflicting or unprovable archived FIT must
    fail closed instead of triggering another optimizer run.
    """
    fold_id = str(fold_contract["fold_id"])
    locations = archived_rolling_fit_fold_candidates(
        point_in_time_dir=point_in_time_dir, fold_id=fold_id,
        project_root=project_root,
    )
    if not locations:
        return None
    verified: list[tuple[Path, dict[str, Any]]] = []
    problems: list[str] = []
    for location in locations:
        path = location / FOLD_MANIFEST_FILENAME
        model = location / DEFAULT_MODEL_FILENAME
        if not path.is_file() or not model.is_file():
            problems.append(f"{location.name}: incomplete historical FIT")
            continue
        try:
            original = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(original, dict):
                raise ValueError("manifest is not a mapping")
            contract, issue = validate_rolling_fold_fitted_checkpoint(
                manifest_path=path, manifest=original,
                expected=fold_contract,
                expected_execution=scientific_torch_execution(plan),
                bundle=bundle, torch_module=torch_module,
                project_root=project_root,
            )
            if contract is None:
                problems.append(f"{location.name}: {issue}")
                continue
            # The FIT owner's verification already covers the actual fitting
            # dependencies and historical context; a second raw-manifest
            # equality check would incorrectly reject event-only cache drift.
            rolling_aggregate = location.parent.parent / SELECTION_POINT_IN_TIME_MANIFEST_FILENAME
            if not rolling_aggregate.is_file():
                problems.append(f"{location.name}: historical Rolling aggregate missing")
                continue
            verified.append((location, {
                "source_fold_manifest": build_file_manifest(path),
                "source_aggregate_manifest": build_file_manifest(rolling_aggregate),
                "source_checkpoint": build_file_manifest(model),
                "source_fold_path": location.resolve().relative_to(project_root.resolve()).as_posix(),
            }))
        except (OSError, UnicodeError, ValueError, TypeError, RuntimeError) as exc:
            problems.append(f"{location.name}: {type(exc).__name__}: {exc}")
    if not verified:
        raise RuntimeError(
            f"Historical Rolling FIT found for {fold_id} but provenance is unverified; "
            "automatic training blocked: " + "; ".join(problems[:5])
        )
    records = {item[1]["source_checkpoint"]["sha256"] for item in verified}
    if len(records) != 1:
        raise RuntimeError(
            f"Historical Rolling FIT conflict for {fold_id}: matching fitting identity "
            f"has {len(records)} different checkpoint SHA values"
        )
    return verified[0]


def _resolve_verified_cross_mode_fit_source(
    *, fit_source_pit_dirs: tuple[Path, ...], target_fold_dir: Path,
    fold_contract: dict[str, Any], bundle, torch_module, plan,
    project_root: Path = PROJECT_ROOT,
) -> tuple[Path, dict[str, Any]] | None:
    """Find a previously published FIT with the same training identity.

    Only the canonical FIT validator can authorize a source; scores and fold
    calendar ends are NOT part of fitting identity.  A matching candidate with
    broken evidence or conflicting checkpoint SHA must never trigger refitting.
    """
    expected_start = str(dict(fold_contract.get("planned_periods") or {}).get("score_start") or "")
    verified: list[tuple[Path, dict[str, Any]]] = []
    problems: list[str] = []
    for root in dict.fromkeys(Path(value).resolve() for value in fit_source_pit_dirs):
        if root == target_fold_dir.parent.parent.resolve():
            continue
        fold_root = root / "folds"
        if not fold_root.is_dir():
            continue
        for manifest_path in sorted(fold_root.glob("*/" + FOLD_MANIFEST_FILENAME)):
            try:
                source_manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeError, ValueError, TypeError) as exc:
                problems.append(f"{manifest_path.parent.name}: unreadable FIT manifest ({exc})")
                continue
            if not isinstance(source_manifest, dict):
                problems.append(f"{manifest_path.parent.name}: FIT manifest not a mapping")
                continue
            # Filter genuinely distinct experiments before applying the single FIT owner.
            if any(source_manifest.get(field) != fold_contract.get(field) for field in (
                "filter_id", "model_architecture", "experiment_profile", "seed",
            )):
                continue
            source_start = str(dict(source_manifest.get("planned_periods") or {}).get("score_start") or "")
            if source_start != expected_start:
                continue
            model_path = manifest_path.parent / DEFAULT_MODEL_FILENAME
            aggregate = root / SELECTION_POINT_IN_TIME_MANIFEST_FILENAME
            if not model_path.is_file() or not aggregate.is_file():
                problems.append(f"{manifest_path.parent.name}: checkpoint or aggregate missing")
                continue
            try:
                fitted, issue = validate_rolling_fold_fitted_checkpoint(
                    manifest_path=manifest_path, manifest=source_manifest,
                    expected=fold_contract,
                    expected_execution=scientific_torch_execution(plan),
                    bundle=bundle, torch_module=torch_module,
                    project_root=project_root,
                )
                if fitted is None:
                    problems.append(f"{manifest_path.parent.name}: {issue}")
                    continue
                verified.append((manifest_path.parent, {
                    "source_fold_manifest": build_file_manifest(manifest_path),
                    "source_aggregate_manifest": build_file_manifest(aggregate),
                    "source_checkpoint": build_file_manifest(model_path),
                    "source_fold_path": manifest_path.parent.resolve().relative_to(
                        project_root.resolve()
                    ).as_posix(),
                }))
            except (OSError, UnicodeError, ValueError, TypeError, RuntimeError) as exc:
                problems.append(f"{manifest_path.parent.name}: {type(exc).__name__}: {exc}")
    if problems:
        raise RuntimeError(
            "Cross-mode FIT candidate exists but provenance is unverified; "
            "automatic training blocked: " + "; ".join(problems[:5])
        )
    if not verified:
        return None
    checkpoint_hashes = {item[1]["source_checkpoint"]["sha256"] for item in verified}
    if len(checkpoint_hashes) != 1:
        raise RuntimeError(
            "Cross-mode FIT identity conflict: multiple verified checkpoints with different SHA; "
            "automatic training blocked"
        )
    return verified[0]


def _materialize_verified_archived_rolling_fit(
    *, archived_fold: Path, archive_binding: dict[str, Any],
    fold_dir: Path, cache_root: Path | None,
    bundle, ids: dict[str, Any], fold_contract: dict[str, Any],
    expected_fingerprint: str, args, torch_module, plan,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Re-score a proven historical FIT, preserving the original archive bytes.

    The archive was verified by the canonical FIT owner in an all-fold preflight.
    Recheck its SHA immediately before materializing (TOCTOU); incomplete live
    folds/caches are never deleted in order to make a report look READY.
    """
    fold_id = str(fold_contract["fold_id"])
    if fold_dir.exists() and (not fold_dir.is_dir() or any(fold_dir.iterdir())):
        raise RuntimeError(
            f"Rolling FIT recovery refuses to overwrite existing publication: {fold_id}"
        )
    if cache_root is not None:
        cache_entry = _checkpoint_cache_entry_dir(cache_root, fold_contract)
        if cache_entry.exists():
            raise RuntimeError(
                f"Rolling FIT recovery refuses to overwrite existing shared cache: {fold_id}"
            )
    if any((
        build_file_manifest(archived_fold / FOLD_MANIFEST_FILENAME)
        != archive_binding["source_fold_manifest"],
        build_file_manifest(archived_fold / DEFAULT_MODEL_FILENAME)
        != archive_binding["source_checkpoint"],
        build_file_manifest(archived_fold.parent.parent / SELECTION_POINT_IN_TIME_MANIFEST_FILENAME)
        != archive_binding["source_aggregate_manifest"],
    )):
        raise RuntimeError(f"Historical Rolling FIT changed during recovery: {fold_id}")
    # Prefer zero-GPU restoration when the archive contains the *exact* PIT
    # score/evaluation contract, including complete score and validation SHA.
    # Different score horizons may share FIT but must re-score independently.
    exact = _load_reusable_fold(
        fold_dir=archived_fold, expected_fingerprint=expected_fingerprint,
        fold_contract=fold_contract,
    )
    if exact is not None:
        frame, manifest = exact
        staging = fold_dir.parent / (
            f".{fold_dir.name}.archived_reuse_{os.getpid()}_{time.time_ns()}"
        )
        staging.mkdir(parents=True, exist_ok=False)
        try:
            for filename, field in (
                (DEFAULT_MODEL_FILENAME, "checkpoint"),
                (FOLD_SCORE_FILENAME, "scores"),
                (FOLD_VALIDATION_SCORE_FILENAME, "validation_scores"),
            ):
                shutil.copy2(archived_fold / filename, staging / filename)
                if build_file_manifest(staging / filename) != dict(manifest.get("artifacts") or {}).get(field):
                    raise RuntimeError(f"Historical Rolling {field} changed during copy: {fold_id}")
            restored = {**manifest, "archived_rolling_fit_source": dict(archive_binding)}
            write_json(staging / FOLD_MANIFEST_FILENAME, restored)
            if fold_dir.exists():
                fold_dir.rmdir()  # only an empty prechecked directory may exist
            staging.replace(fold_dir)
            return frame, restored
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    reused = _rescore_fold_from_fitting_checkpoint(
        source_fold_dir=archived_fold, target_fold_dir=fold_dir,
        bundle=bundle, ids=ids, fold_contract=fold_contract,
        expected_fingerprint=expected_fingerprint, args=args,
        torch_module=torch_module, plan=plan,
        verified_archive_binding=archive_binding,
    )
    if reused is None:
        raise RuntimeError(
            f"Verified historical Rolling FIT cannot be re-scored: {fold_id}; training blocked"
        )
    frame, manifest = reused
    if build_file_manifest(fold_dir / DEFAULT_MODEL_FILENAME) != archive_binding["source_checkpoint"]:
        raise RuntimeError(
            f"Historical Rolling FIT checkpoint SHA changed during rescore: {fold_id}"
        )
    restored = {**manifest, "archived_rolling_fit_source": dict(archive_binding)}
    write_json(fold_dir / FOLD_MANIFEST_FILENAME, restored)
    return frame, restored


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


def _predict_primary_score_payload(torch_module, model, bundle, group_ids, *, args, plan):
    """Predict the canonical primary score plus runtime-owned sidecar columns."""

    ids = np.asarray(group_ids, dtype=np.int64)
    scores, extras = predict_score_output_payload(
        torch_module,
        model,
        bundle,
        ids,
        batch_size=int(args.evaluation_batch_size),
        plan=plan,
    )
    values = np.asarray(scores, dtype=np.float64)
    if len(values) != len(ids):
        raise ValueError("PIT fold inference score count不一致")
    return values, dict(extras)


def _build_fold_score_frame(bundle, group_ids, scores, extras, *, fold_id: str, cutoff: str) -> pd.DataFrame:
    ids = np.asarray(group_ids, dtype=np.int64)
    frame = bundle.group_table.iloc[ids][["ticker", "date", "group_index"]].copy()
    frame["breakout_quality_score"] = np.asarray(scores, dtype=np.float64)
    for column, values in dict(extras or {}).items():
        frame[str(column)] = np.asarray(values)
    frame["fold_id"] = str(fold_id)
    frame["model_information_cutoff"] = str(cutoff)
    return frame


def _validate_validation_score_frame(frame: pd.DataFrame, *, ids: np.ndarray, fold_id: str) -> pd.DataFrame:
    work = frame.copy()
    if len(work) != len(ids):
        raise ValueError(f"{fold_id} validation score count不一致")
    expected = np.asarray(ids, dtype=np.int64)
    observed = pd.to_numeric(work["group_index"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(observed, expected):
        raise ValueError(f"{fold_id} validation group order不一致")
    values = pd.to_numeric(work["breakout_quality_score"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{fold_id} validation score含非有限值")
    if set(work["fold_id"].astype(str).unique()) != {str(fold_id)}:
        raise ValueError(f"{fold_id} validation fold identity不一致")
    return work.reset_index(drop=True)


def _train_fold(
    args,
    bundle,
    fold,
    ids,
    fold_contract,
    contract_fingerprint,
    *,
    torch,
    plan,
    progress_callback=None,
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
        selection_metrics_only=bool(
            BREAKOUT_QUALITY_PIT_EPOCH_SELECTION_LIGHTWEIGHT_METRICS
        ),
        progress_callback=progress_callback,
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
        progress_callback=progress_callback,
    )
    scores, score_extras = _predict_primary_score_payload(
        torch, model, bundle, ids["score_ids"], args=args, plan=plan
    )
    validation_scores, validation_extras = _predict_primary_score_payload(
        torch, model, bundle, ids["validation_ids"], args=args, plan=plan
    )
    frame = _build_fold_score_frame(
        bundle, ids["score_ids"], scores, score_extras,
        fold_id=fold_id, cutoff=fold_contract["model_information_cutoff"],
    )
    validation_frame = _build_fold_score_frame(
        bundle, ids["validation_ids"], validation_scores, validation_extras,
        fold_id=fold_id, cutoff=fold_contract["model_information_cutoff"],
    )
    frame["fold_id"] = fold_id
    frame["model_information_cutoff"] = fold_contract["model_information_cutoff"]
    frame = _validate_score_frame(frame, fold_contract=fold_contract)
    validation_frame = _validate_validation_score_frame(
        validation_frame, ids=ids["validation_ids"], fold_id=fold_id
    )

    fold_dir = _point_in_time_fold_dir_for_args(args, fold_id)
    fold_dir.mkdir(parents=True, exist_ok=True)
    model_path = fold_dir / DEFAULT_MODEL_FILENAME
    score_path = fold_dir / FOLD_SCORE_FILENAME
    validation_score_path = fold_dir / FOLD_VALIDATION_SCORE_FILENAME
    checkpoint_payload = build_checkpoint_payload(
        model,
        bundle,
        args=args,
        plan=plan,
        selected_epoch=selected_epoch,
        fold_contract=fold_contract,
    )
    final_encoder_pretraining = getattr(model, "_encoder_pretraining_summary", None)
    encoder_pretraining_audit = None
    if final_encoder_pretraining is not None:
        encoder_pretraining_audit = {
            "epoch_selection": epoch_selection.get("encoder_pretraining"),
            "final_refit": final_encoder_pretraining,
        }
        checkpoint_payload["encoder_pretraining"] = encoder_pretraining_audit
    torch.save(checkpoint_payload, model_path)
    frame.to_csv(score_path, index=False, encoding="utf-8-sig")
    validation_frame.to_csv(validation_score_path, index=False, encoding="utf-8-sig")
    native_origin = build_fitting_context_origin(
        profile=str(fold_contract["experiment_profile"]),
        target=dict(bundle.target_manifest or {}),
        checkpoint=build_file_manifest(model_path), kind="rolling_native",
    )
    manifest = {
        **fold_contract,
        **({"fitting_origin": native_origin} if native_origin is not None else {}),
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
            "validation_scores": build_file_manifest(validation_score_path),
        },
    }
    if encoder_pretraining_audit is not None:
        manifest["encoder_pretraining"] = encoder_pretraining_audit
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



def _pit_aggregate_baseline_sha256(manifest_path: Path) -> str | None:
    """Capture a publisher's initial top-level commit marker, not FIT identity."""
    return compute_file_sha256(manifest_path) if manifest_path.is_file() else None


def _stage_pit_aggregate_csv(path: Path, frame: pd.DataFrame) -> Path:
    """Create an fsynced sibling CSV without truncating an active publication."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    staged = Path(name)
    try:
        with staged.open("w", encoding="utf-8-sig", newline="") as stream:
            frame.to_csv(stream, index=False)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return staged


def _publish_selection_pit_aggregate(
    *,
    score_path: Path,
    coverage_path: Path,
    manifest_path: Path,
    score_frame: pd.DataFrame,
    coverage_frame: pd.DataFrame,
    manifest: dict[str, Any],
    baseline_manifest_sha256: str | None,
    point_in_time_dir: Path,
    score_start: pd.Timestamp,
    score_end: pd.Timestamp,
    selection_end: pd.Timestamp,
    started_at: float | None = None,
) -> None:
    """Publish a PIT aggregate through one writer owner and manifest-last commit.

    Staging is outside the publication lock. A second publisher that completed
    after this run started cannot be silently overwritten: the initial commit
    marker must still match while we own the OS lock. Readers validate the
    content SHA, so interrupted multi-file swaps fail closed, not as READY.
    """
    staged: list[Path] = []
    try:
        staged_score = _stage_pit_aggregate_csv(score_path, score_frame)
        staged.append(staged_score)
        staged_coverage = _stage_pit_aggregate_csv(coverage_path, coverage_frame)
        staged.append(staged_coverage)
        records = {
            "scores": build_file_manifest(staged_score),
            "coverage": build_file_manifest(staged_coverage),
        }
        records["scores"]["filename"] = score_path.name
        records["coverage"]["filename"] = coverage_path.name
        manifest["artifacts"] = records
        lock_path = manifest_path.with_name("." + manifest_path.name + ".publish.lock")
        with exclusive_file_lock(
            lock_path, conflict_message="Another Selection PIT aggregate publication is active",
        ):
            if _pit_aggregate_baseline_sha256(manifest_path) != baseline_manifest_sha256:
                raise RuntimeError(
                    "Selection PIT manifest changed during this run; "
                    "refusing to overwrite a concurrent publication"
                )
            _snapshot_superseded_point_in_time_aggregate(
                point_in_time_dir=point_in_time_dir,
                manifest_path=manifest_path,
                score_start=score_start,
                score_end=score_end,
                selection_end=selection_end,
            )
            atomic_replace_with_retry(staged_score, score_path)
            atomic_replace_with_retry(staged_coverage, coverage_path)
            # Legacy/uncoordinated writers are not expected, but verify exact
            # staged bytes before making a valid top-level manifest visible.
            for name, path in (("scores", score_path), ("coverage", coverage_path)):
                if compute_file_sha256(path).lower() != str(records[name]["sha256"]).lower():
                    raise RuntimeError(f"Selection PIT {name} changed while publishing")
            if started_at is not None:
                # Observed wall time includes staging/snapshot/commit I/O, as it
                # did before the publisher became transactional.
                manifest["elapsed_sec"] = round(time.perf_counter() - started_at, 3)
                manifest["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(manifest_path, manifest)
    finally:
        for path in staged:
            path.unlink(missing_ok=True)


def _run_point_in_time_scores(
    args: argparse.Namespace,
    *,
    authorization_mode: str = "rolling",
) -> int:
    _validate_args(args)
    settings = get_breakout_quality_workflow_settings(
        experiment_profile=str(args.experiment_profile)
    )
    mode = str(authorization_mode).strip().lower()
    if mode == "rolling":
        if not settings.rolling_authorized:
            raise ValueError(
                f"目前profile未授權Rolling PIT scores: {args.experiment_profile}"
            )
    elif mode == "strategy_compare":
        from core.strategy_compare_policy import (
            validate_single_seed_strategy_conversion_pit_scope,
        )

        validate_single_seed_strategy_conversion_pit_scope(
            strategy_profile_id=str(args.strategy_compare_profile_id),
            dl_id=str(args.strategy_compare_source_id),
            filter_id=str(args.filter_id),
            model_architecture=str(args.model_architecture),
            experiment_profile=str(args.experiment_profile),
            seed=int(args.seed),
            score_start_date=str(args.score_start_date),
            score_end_date=(
                None if args.score_end_date in (None, "") else str(args.score_end_date)
            ),
            fold_months=int(args.fold_months),
            fold_anchor_date=(
                None if args.fold_anchor_date in (None, "") else str(args.fold_anchor_date)
            ),
            single_score_block=bool(args.single_score_block),
            inner_validation_months=int(args.inner_validation_months),
            train_window_months=(
                None if args.train_window_months is None else int(args.train_window_months)
            ),
        )
    elif mode == "stacking_context":
        research_spec = get_continuous_ranker_research_spec(
            str(args.experiment_profile)
        )
        if not bool(research_spec.selection_pit_authorized):
            raise ValueError(
                "stacking context只允許已有historical PIT authorization的Stage-1 profile"
            )
        if not str(getattr(args, "point_in_time_dir_override", "") or "").strip():
            raise ValueError("stacking context必須使用isolated point_in_time_dir_override")
    else:
        raise ValueError(f"未知PIT authorization mode: {authorization_mode!r}")
    # Authorization must precede all publication-side effects. Serialize the
    # entire authorized build (including Fold FIT imports and rescores), not
    # only the final manifest, so two processes cannot mutate the same Fold.
    if bool(args.plan_only):
        return _run_authorized_point_in_time_scores(args, settings=settings)
    pit_dir = _point_in_time_dir_for_args(args)
    with exclusive_file_lock(
        selection_pit_build_lock_path(pit_dir),
        conflict_message="Another Selection PIT build or FIT repair is active for this output directory",
    ):
        return _run_authorized_point_in_time_scores(args, settings=settings)


def _run_authorized_point_in_time_scores(args, *, settings) -> int:
    started = time.perf_counter()
    color_enabled = console_color_enabled()
    data_started = time.perf_counter()
    data_progress_state = {"last_bucket": -1}
    daily_data_progress = InlineProgress()

    def _daily_data_progress(processed, total, target_valid, skipped):
        total = max(int(total), 1)
        processed = int(processed)
        bucket = min(20, int(processed * 20 / total))
        if processed not in {0, total} and bucket <= int(data_progress_state["last_bucket"]):
            return
        data_progress_state["last_bucket"] = bucket
        pct = min(100.0, 100.0 * processed / total)
        daily_data_progress.update(
            "[PIT data] "
            f"{processed}/{total} tickers ({pct:5.1f}%)｜"
            f"target_valid={int(target_valid):,}｜skipped={int(skipped)}｜"
            f"elapsed={format_elapsed(time.perf_counter() - data_started)}"
        )

    daily_scope = (
        settings.training_sample_scope
        == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
    )
    if daily_scope:
        daily_data_progress.update("[PIT data] 建立daily stock-day index／40D target...")
    try:
        bundle = load_continuous_ranker_data(
            filter_id=args.filter_id,
            model_architecture=args.model_architecture,
            experiment_profile=args.experiment_profile,
            preload_feature_bank=bool(args.preload_feature_bank),
            allow_stale_source=bool(args.allow_stale_source),
            project_root=PROJECT_ROOT,
            progress_callback=_daily_data_progress if daily_scope else None,
            seed=int(args.seed),
            authorized_target_dependency_build=False,
        )
    finally:
        if daily_scope:
            daily_data_progress.finish()
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
    # Detect an aggregate published by another process while folds are in flight.
    aggregate_baseline_sha256 = _pit_aggregate_baseline_sha256(
        _selection_point_in_time_manifest_path_for_args(args)
    )

    # A prior Forward FIT repair may have committed the *whole* six-fold Rolling
    # publication to historical_fit_repairs. Inspect every missing fold's saved
    # FIT before touching any score publication. An unprovable historical FIT
    # must stop the entire refresh before fold 1 is imported/re-scored; it is
    # never interpreted as permission to repeat epochs 1..N.
    archived_fits: dict[str, tuple[Path, dict[str, Any]]] = {}
    if bool(args.resume):
        for preview_index, (preview_fold, preview_ids) in enumerate(zip(folds, fold_details), start=1):
            preview_contract = _fold_contract_payload(args, bundle, preview_fold, preview_ids)
            fold_id = str(preview_fold["fold_id"])
            local_fold = _point_in_time_fold_dir_for_args(args, fold_id)
            cache_model = None
            if args.checkpoint_cache_root not in (None, ""):
                cache_model = (
                    _checkpoint_cache_entry_dir(Path(str(args.checkpoint_cache_root)), preview_contract)
                    / DEFAULT_MODEL_FILENAME
                )
            # Live fold/cache candidates are owned by the normal identity
            # checks. The canonical Forward bridge owns the matching first
            # fold, including its intentionally superseded historical copy.
            if ((local_fold / DEFAULT_MODEL_FILENAME).is_file()
                or (cache_model is not None and cache_model.is_file())
                or (preview_index == 1
                    and getattr(args, "checkpoint_import_model_dir", None) not in (None, "")
                    and getattr(args, "checkpoint_import_report_path", None) not in (None, ""))):
                continue
            archived = _verified_archived_rolling_fit(
                point_in_time_dir=point_in_time_dir, fold_contract=preview_contract,
                bundle=bundle, torch_module=torch, plan=plan,
                project_root=PROJECT_ROOT,
            )
            if archived is not None:
                archived_fits[fold_id] = archived

    score_frames: list[pd.DataFrame] = []
    fold_manifests: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    reused_fold_count = 0
    migrated_fold_count = 0
    rescored_checkpoint_fold_count = 0
    fitting_checkpoint_reuse_fold_count = 0
    forward_checkpoint_import_count = 0
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
        forward_checkpoint_verified = False
        forward_checkpoint_imported = False
        fit_validation_errors: list[str] = []
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
                fit_validation_errors=fit_validation_errors,
            )
            rescored_checkpoint = reused is not None
        if (
            reused is None
            and bool(args.resume)
            and args.checkpoint_cache_root not in (None, "")
            and getattr(args, "checkpoint_import_model_dir", None) not in (None, "")
            and getattr(args, "checkpoint_import_report_path", None) not in (None, "")
        ):
            imported = _import_forward_checkpoint_to_fitting_cache(
                source_model_dir=Path(str(args.checkpoint_import_model_dir)),
                source_report_path=Path(str(args.checkpoint_import_report_path)),
                cache_root=Path(str(args.checkpoint_cache_root)),
                fold_contract=fold_contract,
                bundle=bundle,
                torch_module=torch,
                execution_plan=plan,
            )
            if imported is not None:
                forward_checkpoint_verified = True
                forward_checkpoint_imported = not bool(imported.get("reused_existing_cache"))
                if forward_checkpoint_imported:
                    forward_checkpoint_import_count += 1
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
        cross_mode_fit_reuse = False
        if reused is None and bool(args.resume) and getattr(args, "fit_source_pit_dir", ()):
            # Never overwrite a failed/partially-resumable target Fold while
            # repairing only the score publication from another evaluation mode.
            if any((fold_dir / name).exists() for name in (
                DEFAULT_MODEL_FILENAME, FOLD_MANIFEST_FILENAME,
            )):
                raise RuntimeError(
                    f"Cross-mode FIT destination {fold['fold_id']} already has unverified "
                    "checkpoint/manifest; automatic overwrite and training blocked"
                )
            if args.checkpoint_cache_root not in (None, ""):
                existing_cache = _checkpoint_cache_entry_dir(
                    Path(str(args.checkpoint_cache_root)), fold_contract
                )
                if (existing_cache / DEFAULT_MODEL_FILENAME).exists():
                    raise RuntimeError(
                        f"Cross-mode FIT shared cache for {fold['fold_id']} is unverified; "
                        "automatic overwrite and training blocked"
                    )
            external_source = _resolve_verified_cross_mode_fit_source(
                fit_source_pit_dirs=tuple(Path(x) for x in args.fit_source_pit_dir),
                target_fold_dir=fold_dir, fold_contract=fold_contract,
                bundle=bundle, torch_module=torch, plan=plan, project_root=PROJECT_ROOT,
            )
            if external_source is not None:
                source_fold, verified_binding = external_source
                reused = _rescore_fold_from_fitting_checkpoint(
                    source_fold_dir=source_fold, target_fold_dir=fold_dir,
                    bundle=bundle, ids=ids, fold_contract=fold_contract,
                    expected_fingerprint=fingerprint, args=args,
                    torch_module=torch, plan=plan,
                    verified_archive_binding=verified_binding,
                )
                if reused is None:
                    raise RuntimeError(
                        f"Cross-mode FIT verified for {fold['fold_id']}, but score regeneration "
                        "failed; automatic training blocked"
                    )
                cross_mode_fit_reuse = True
        archived_checkpoint_reuse = False
        if reused is None and bool(args.resume) and str(fold["fold_id"]) in archived_fits:
            archived_fold, archive_binding = archived_fits[str(fold["fold_id"])]
            reused = _materialize_verified_archived_rolling_fit(
                archived_fold=archived_fold, archive_binding=archive_binding,
                fold_dir=fold_dir,
                cache_root=(Path(str(args.checkpoint_cache_root))
                            if args.checkpoint_cache_root not in (None, "") else None),
                bundle=bundle, ids=ids, fold_contract=fold_contract,
                expected_fingerprint=fingerprint, args=args,
                torch_module=torch, plan=plan,
            )
            archived_checkpoint_reuse = True
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
            if fitting_checkpoint_reuse or archived_checkpoint_reuse or cross_mode_fit_reuse:
                fitting_checkpoint_reuse_fold_count += 1
            elif rescored_checkpoint:
                rescored_checkpoint_fold_count += 1
            elif migrated:
                migrated_fold_count += 1
            else:
                reused_fold_count += 1
            if cross_mode_fit_reuse:
                reuse_source = "verified cross-mode published Fold → score refresh"
                reuse_label = "重用跨模式已發布 FIT 並重新評分"
            elif archived_checkpoint_reuse:
                reuse_source = "verified historical Rolling fold → score refresh"
                reuse_label = "重用歷史封存的Rolling FIT並重新評分"
            elif fitting_checkpoint_reuse:
                if forward_checkpoint_verified:
                    reuse_source = (
                        "Forward OOS fitting checkpoint → shared cache"
                        if forward_checkpoint_imported
                        else "shared fitting cache（Forward OOS identity verified）"
                    )
                else:
                    reuse_source = "shared fitting cache"
                reuse_label = "重用fitting identity checkpoint並依目前fold重評score"
            elif rescored_checkpoint:
                reuse_source = "Rolling fold checkpoint"
                reuse_label = "重用既有checkpoint並重評score universe"
            elif migrated:
                reuse_source = "compatible legacy fold"
                reuse_label = "遷移舊fold並重用"
            else:
                reuse_source = "Rolling fold artifact"
                reuse_label = "重用既有 fold 工件"
            if compact_console:
                fold_progress.print_line(
                    f"PIT fold {fold_index}/{len(folds)} | "
                    + paint(str(fold["fold_id"]), "cyan", enabled=color_enabled, bold=True)
                    + " | "
                    + paint("REUSE", "green", enabled=color_enabled, bold=True)
                    + f" | source={reuse_source}"
                    + f" | score={len(frame):,}"
                )
            else:
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
                    + f" | source={reuse_source}"
                )
        else:
            if bool(args.checkpoint_only):
                raise RuntimeError(
                    "checkpoint-only PIT重建無法完成；fold缺少可重用score/checkpoint或identity不相容: "
                    f"{fold['fold_id']}。請由canonical model-training service以resume模式補齊PIT fold。"
                )
            if bool(args.resume):
                # Rebuild only a genuinely absent FIT. An existing fold/cache
                # checkpoint whose contract failed validation is NOT an absent
                # checkpoint and must never be overwritten by a report refresh.
                unverified = [fold_dir / DEFAULT_MODEL_FILENAME]
                if args.checkpoint_cache_root not in (None, ""):
                    cache_entry = _checkpoint_cache_entry_dir(
                        Path(str(args.checkpoint_cache_root)), fold_contract
                    )
                    unverified.append(cache_entry / DEFAULT_MODEL_FILENAME)
                if any(path.is_file() for path in unverified):
                    reasons = _unverified_fit_checkpoint_reasons(
                        fold_dir=fold_dir, fold_contract=fold_contract, args=args,
                        torch_module=torch,
                    )
                    evidence = [f"canonical FIT owner: {reason}" for reason in fit_validation_errors]
                    evidence.extend(reasons)
                    detail = "; ".join(evidence) if evidence else "checkpoint identity proof unavailable"
                    raise RuntimeError(
                        "PIT FIT identity unverified: existing fold/shared-cache checkpoint "
                        f"for {fold['fold_id']} cannot be reused; automatic refit is blocked; "
                        f"read-only evidence: {detail}"
                    )
                candidate_override = str(
                    getattr(args, "forward_fit_candidate_model_dir", "") or ""
                ).strip()
                if candidate_override:
                    forward_model_dir = Path(candidate_override)
                elif str(getattr(args, "point_in_time_dir_override", "") or "").strip():
                    forward_model_dir = None  # isolated scope needs explicit same-seed locator
                else:
                    forward_model_dir = resolve_filter_artifact_paths(
                        PROJECT_ROOT, str(args.filter_id),
                        str(args.model_architecture), str(args.experiment_profile),
                    ).model_dir
                if forward_model_dir is not None:
                    issue = unresolved_forward_fit_candidate_issue(
                        model_dir=Path(forward_model_dir), fold_contract=fold_contract,
                        potentially_same_cutoff=(fold_index == 1),
                    )
                    if issue is not None:
                        raise RuntimeError(
                            f"PIT FIT identity unverified: {fold['fold_id']}: {issue}; "
                            "automatic refit is blocked"
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
            def _fold_training_progress(phase: str, epoch: int, total_epochs: int) -> None:
                if not compact_console:
                    return
                fold_progress.update(
                    render_training_unit_progress(
                        unit_id=f"PIT fold {fold_index}/{len(folds)} | {fold['fold_id']}",
                        elapsed_seconds=time.perf_counter() - fold_started,
                        epoch_progress=(str(phase), int(epoch), int(total_epochs)),
                    )
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
                progress_callback=_fold_training_progress if compact_console else None,
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
        "score_columns": _score_output_columns(bundle),
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
            "forward_checkpoint_import": int(forward_checkpoint_import_count),
            "built": int(built_fold_count),
        },
        "folds": [_combined_fold_record(args, item) for item in fold_manifests],
        "source_dataset": bundle.summary,
        "source_continuous_target": bundle.target_manifest,
        "lookahead_contract": selection_pit_lookahead_contract(),
        "runtime_eligibility": {
            "eligible_scope": "point_in_time_strategy_replay",
            "eligible": True,
            "strategy_use_requires_model_validation": True,
            "per_score_fold_information_cutoff_required": True,
            "frozen_forward_required": False,
        },
        "artifacts": {},  # filled from fully staged CSVs by the publication owner
        "torch_execution": plan.as_manifest_payload(),
        "elapsed_sec": round(time.perf_counter() - started, 3),
    }
    if (
        str(bundle.profile.training_sample_scope)
        == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
    ):
        manifest["score_eligibility_contract"] = build_score_eligibility_contract(bundle.profile)
    _publish_selection_pit_aggregate(
        score_path=score_path,
        coverage_path=coverage_path,
        manifest_path=manifest_path,
        score_frame=combined,
        coverage_frame=pd.DataFrame(coverage_rows),
        manifest=manifest,
        baseline_manifest_sha256=aggregate_baseline_sha256,
        point_in_time_dir=point_in_time_dir,
        score_start=score_start,
        score_end=score_end,
        selection_end=selection_end,
        started_at=started,
    )
    clear_selection_point_in_time_ranking_contract_cache()
    if compact_console:
        print(
            paint("PIT Scores 完成", "green", enabled=color_enabled, bold=True)
            + f" | folds={len(folds)}"
            + f" | 重用={reused_fold_count}"
            + f" | 遷移重用={migrated_fold_count}"
            + f" | checkpoint重評={rescored_checkpoint_fold_count}"
            + f" | fitting identity checkpoint={fitting_checkpoint_reuse_fold_count}"
            + f" | Forward import={forward_checkpoint_import_count}"
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
    _authorization_mode: str = "rolling",
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
    return _run_point_in_time_scores(
        parse_args(argv), authorization_mode=str(_authorization_mode)
    )


def build_cross_fitted_context_scores(**kwargs) -> int:
    """Build isolated Stage-1 scores for stacked research without authorizing current Rolling.

    The source profile must already carry historical/selection PIT authorization.
    Callers must provide an isolated ``point_in_time_dir_override``; this producer
    never changes the source profile's UI/current-time validation authorization.
    """

    if not str(kwargs.get("point_in_time_dir_override") or "").strip():
        raise ValueError("stacking context必須指定point_in_time_dir_override")
    kwargs["_authorization_mode"] = "stacking_context"
    return build_selection_point_in_time_scores(**kwargs)


def main(argv=None) -> int:
    args = parse_args(argv)
    strategy_scoped = bool(
        str(getattr(args, "strategy_compare_profile_id", "") or "").strip()
        and str(getattr(args, "strategy_compare_source_id", "") or "").strip()
    )
    return _run_point_in_time_scores(
        args,
        authorization_mode=("strategy_compare" if strategy_scoped else "rolling"),
    )


__all__ = [
    "FOLD_MANIFEST_FILENAME",
    "FOLD_SCORE_FILENAME",
    "FOLD_VALIDATION_SCORE_FILENAME",
    "POINT_IN_TIME_SCHEMA_VERSION",
    "build_cross_fitted_context_scores",
    "build_selection_point_in_time_scores",
    "build_rolling_first_fold_fit_proof",
    "fold_training_identity",
    "main",
    "parse_args",
]


if __name__ == "__main__":
    raise SystemExit(main())
