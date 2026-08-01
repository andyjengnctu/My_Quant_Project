"""Build rolling/cross-fitted Selection point-in-time continuous-ranker scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.breakout_quality_policy import (
    BREAKOUT_QUALITY_ALLOW_TF32,
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
from config.breakout_quality_experiments import (
    TRAINING_LABEL_SCOPE_ALL,
    TRAINING_LABEL_SCOPE_PASS_ONLY,
)
from config.breakout_quality_workflow import get_breakout_quality_workflow_settings
from filters.breakout_quality.artifacts import build_file_manifest
from filters.breakout_quality.contract import (
    DEFAULT_MODEL_FILENAME,
    LABEL_PASS,
    LABEL_REJECT,
)
from filters.breakout_quality.paths import (
    resolve_filter_point_in_time_dir,
    resolve_filter_point_in_time_fold_dir,
    resolve_selection_point_in_time_coverage_path,
    resolve_selection_point_in_time_manifest_path,
    resolve_selection_point_in_time_score_path,
)
from filters.breakout_quality.torch_runtime import (
    SUPPORTED_MIXED_PRECISION_DTYPES,
    SUPPORTED_TORCH_DEVICES,
)
from tools.filters.breakout_quality.common import PROJECT_ROOT, write_json
from tools.filters.breakout_quality.continuous_ranker_pipeline import (
    build_checkpoint_payload,
    build_percentile_target,
    fit_final,
    load_continuous_ranker_data,
    predict_scores,
    resolve_ranker_execution_plan,
    select_epoch,
)

POINT_IN_TIME_SCHEMA_VERSION = 1
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
    settings = get_breakout_quality_workflow_settings()
    parser = argparse.ArgumentParser(
        description=(
            "以expanding-window folds建立Selection point-in-time continuous-ranker scores；"
            "每個fold只用score period以前且label已完成的資料訓練。"
        )
    )
    parser.add_argument("--filter-id", default=settings.filter_id)
    parser.add_argument("--model-architecture", default=settings.model_architecture)
    parser.add_argument("--experiment-profile", default=settings.experiment_profile)
    parser.add_argument("--score-start-date", default=settings.point_in_time_score_start_date)
    parser.add_argument("--score-end-date", default=settings.point_in_time_score_end_date)
    parser.add_argument("--fold-months", type=int, default=settings.point_in_time_fold_months)
    parser.add_argument(
        "--inner-validation-months",
        type=int,
        default=settings.point_in_time_inner_validation_months,
    )
    parser.add_argument("--epochs", type=int, default=BREAKOUT_QUALITY_DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BREAKOUT_QUALITY_DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--evaluation-batch-size",
        type=int,
        default=BREAKOUT_QUALITY_EVALUATION_BATCH_SIZE,
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
        "--plan-only",
        action="store_true",
        help="只建立並驗證fold計畫，不訓練或寫入正式score工件",
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
    if int(args.epochs) < 1 or int(args.batch_size) < 2 or int(args.evaluation_batch_size) < 1:
        raise ValueError("epochs>=1、batch-size>=2、evaluation-batch-size>=1")
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


def _json_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_fold_periods(
    score_start: pd.Timestamp,
    score_end: pd.Timestamp,
    *,
    fold_months: int,
) -> list[dict[str, Any]]:
    folds: list[dict[str, Any]] = []
    cursor = score_start
    index = 0
    while cursor <= score_end:
        next_start = (cursor + pd.DateOffset(months=int(fold_months))).normalize()
        fold_end = min(score_end, next_start - pd.Timedelta(days=1))
        folds.append(
            {
                "fold_id": f"fold_{index:03d}",
                "score_start": cursor,
                "score_end": fold_end,
            }
        )
        cursor = next_start
        index += 1
    if not folds:
        raise ValueError("point-in-time score期間沒有任何fold")
    return folds


def _group_event_count(event_group_index: np.ndarray, group_ids: np.ndarray) -> int:
    return int(np.isin(event_group_index, np.asarray(group_ids, dtype=np.int64)).sum())


def _fold_group_ids(bundle, fold: dict[str, Any], *, validation_months: int) -> dict[str, Any]:
    group_dates = pd.to_datetime(bundle.group_table["date"], errors="raise").dt.normalize()
    label_end_dates = pd.to_datetime(
        bundle.group_table["label_eval_end_date"], errors="raise"
    ).dt.normalize()
    labels = bundle.group_table["label"].to_numpy(dtype=np.int64)
    target_valid = np.asarray(bundle.target_valid, dtype=bool)
    score_start = pd.Timestamp(fold["score_start"])
    score_end = pd.Timestamp(fold["score_end"])
    validation_start = (score_start - pd.DateOffset(months=int(validation_months))).normalize()

    if bundle.profile.training_label_scope == TRAINING_LABEL_SCOPE_PASS_ONLY:
        scoped_target = target_valid & (labels == LABEL_PASS)
    elif bundle.profile.training_label_scope == TRAINING_LABEL_SCOPE_ALL:
        scoped_target = target_valid & np.isin(labels, [LABEL_REJECT, LABEL_PASS])
    else:
        raise ValueError(
            f"不支援的continuous ranker training scope: {bundle.profile.training_label_scope}"
        )
    train_mask = (
        scoped_target
        & (group_dates < validation_start).to_numpy(dtype=bool)
        & (label_end_dates < validation_start).to_numpy(dtype=bool)
    )
    validation_mask = (
        scoped_target
        & (group_dates >= validation_start).to_numpy(dtype=bool)
        & (group_dates < score_start).to_numpy(dtype=bool)
        & (label_end_dates < score_start).to_numpy(dtype=bool)
    )
    final_mask = (
        scoped_target
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

    def observed_range(group_ids: np.ndarray) -> dict[str, str | None]:
        if len(group_ids) == 0:
            return {"start": None, "end": None}
        values = group_dates.iloc[group_ids]
        return {"start": str(values.min().date()), "end": str(values.max().date())}

    return {
        "schema_version": POINT_IN_TIME_SCHEMA_VERSION,
        "fold_id": str(fold["fold_id"]),
        "filter_id": str(args.filter_id),
        "model_architecture": str(args.model_architecture),
        "experiment_profile": str(args.experiment_profile),
        "continuous_target_id": str(bundle.profile.continuous_target_id),
        "training_label_scope": str(bundle.profile.training_label_scope),
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


def _validate_minimum_counts(settings, fold_id: str, ids: dict[str, Any]) -> None:
    required = {
        "inner_train": (len(ids["train_ids"]), settings.point_in_time_min_train_groups),
        "validation": (
            len(ids["validation_ids"]),
            settings.point_in_time_min_validation_groups,
        ),
        "score": (len(ids["score_ids"]), settings.point_in_time_min_score_groups),
    }
    failed = [
        f"{name}={actual}<{minimum}"
        for name, (actual, minimum) in required.items()
        if int(actual) < int(minimum)
    ]
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


def _train_fold(args, bundle, fold, ids, fold_contract, contract_fingerprint, *, torch, plan):
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

    fold_dir = resolve_filter_point_in_time_fold_dir(
        PROJECT_ROOT,
        args.filter_id,
        fold_id,
        args.model_architecture,
        args.experiment_profile,
    )
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


def _print_plan(folds: list[dict[str, Any]], fold_details: list[dict[str, Any]]) -> None:
    print("\nSelection point-in-time fold plan")
    for fold, detail in zip(folds, fold_details):
        print(
            f"- {fold['fold_id']} score={fold['score_start'].date()}~{fold['score_end'].date()} "
            f"train={len(detail['train_ids']):,} validation={len(detail['validation_ids']):,} "
            f"final={len(detail['final_ids']):,} score={len(detail['score_ids']):,}"
        )


def _combined_fold_record(args, item: dict[str, Any]) -> dict[str, Any]:
    fold_id = str(item["fold_id"])
    fold_dir = resolve_filter_point_in_time_fold_dir(
        PROJECT_ROOT,
        args.filter_id,
        fold_id,
        args.model_architecture,
        args.experiment_profile,
    )
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


def main(argv=None) -> int:
    args = parse_args(argv)
    _validate_args(args)
    settings = get_breakout_quality_workflow_settings()
    started = time.perf_counter()
    bundle = load_continuous_ranker_data(
        filter_id=args.filter_id,
        model_architecture=args.model_architecture,
        experiment_profile=args.experiment_profile,
        preload_feature_bank=bool(args.preload_feature_bank),
        allow_stale_source=bool(args.allow_stale_source),
        project_root=PROJECT_ROOT,
    )
    selection_start = _iso_timestamp(
        bundle.outer_policy.get("selection_start_date"), field_name="selection_start_date"
    )
    selection_end = _iso_timestamp(
        bundle.outer_policy.get("selection_end_date"), field_name="selection_end_date"
    )
    score_start = _iso_timestamp(args.score_start_date, field_name="score_start_date")
    score_end = (
        selection_end
        if args.score_end_date is None or str(args.score_end_date).strip() == ""
        else _iso_timestamp(args.score_end_date, field_name="score_end_date")
    )
    if not selection_start <= score_start <= score_end <= selection_end:
        raise ValueError(
            "PIT score期間必須完整位於Selection內: "
            f"selection={selection_start.date()}~{selection_end.date()}, "
            f"score={score_start.date()}~{score_end.date()}"
        )

    folds = _build_fold_periods(score_start, score_end, fold_months=int(args.fold_months))
    fold_details = [
        _fold_group_ids(bundle, fold, validation_months=int(args.inner_validation_months))
        for fold in folds
    ]
    for fold, ids in zip(folds, fold_details):
        _validate_minimum_counts(settings, str(fold["fold_id"]), ids)
    _print_plan(folds, fold_details)
    if bool(args.plan_only):
        print("plan-only完成；未訓練、未寫入正式PIT工件。")
        return 0

    torch, plan = resolve_ranker_execution_plan(args)
    print(
        f"torch=device={plan.device_type}, mixed_precision={plan.mixed_precision_enabled}, "
        f"dtype={plan.autocast_dtype_name}, deterministic={plan.deterministic_algorithms}, "
        f"tf32={plan.allow_tf32}"
    )
    point_in_time_dir = resolve_filter_point_in_time_dir(
        PROJECT_ROOT,
        args.filter_id,
        args.model_architecture,
        args.experiment_profile,
    )
    point_in_time_dir.mkdir(parents=True, exist_ok=True)

    score_frames: list[pd.DataFrame] = []
    fold_manifests: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    for fold, ids in zip(folds, fold_details):
        fold_contract = _fold_contract_payload(args, bundle, fold, ids)
        fingerprint = _json_fingerprint(fold_contract)
        fold_dir = resolve_filter_point_in_time_fold_dir(
            PROJECT_ROOT,
            args.filter_id,
            str(fold["fold_id"]),
            args.model_architecture,
            args.experiment_profile,
        )
        reused = (
            _load_reusable_fold(
                fold_dir=fold_dir,
                expected_fingerprint=fingerprint,
                fold_contract=fold_contract,
            )
            if bool(args.resume)
            else None
        )
        if reused is not None:
            frame, manifest = reused
            print(f"\n{fold['fold_id']}：重用既有fold工件")
        else:
            print(
                f"\n{fold['fold_id']}：訓練並評分 "
                f"{fold['score_start'].date()}~{fold['score_end'].date()}"
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
    score_path = resolve_selection_point_in_time_score_path(
        PROJECT_ROOT, args.filter_id, args.model_architecture, args.experiment_profile
    )
    coverage_path = resolve_selection_point_in_time_coverage_path(
        PROJECT_ROOT, args.filter_id, args.model_architecture, args.experiment_profile
    )
    manifest_path = resolve_selection_point_in_time_manifest_path(
        PROJECT_ROOT, args.filter_id, args.model_architecture, args.experiment_profile
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
        "score_column": "breakout_quality_score",
        "score_period": {
            "start": str(score_start.date()),
            "end": str(score_end.date()),
        },
        "selection_period": {
            "start": str(selection_start.date()),
            "end": str(selection_end.date()),
        },
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
            "eligible_scope": "selection_model_validation_only",
            "eligible": False,
            "strategy_use_requires_model_validation": True,
            "not_eligible_for_forward_oos_runtime": True,
        },
        "artifacts": {
            "scores": build_file_manifest(score_path),
            "coverage": build_file_manifest(coverage_path),
        },
        "torch_execution": plan.as_manifest_payload(),
        "elapsed_sec": round(time.perf_counter() - started, 3),
    }
    write_json(manifest_path, manifest)
    print("\nSelection point-in-time scores完成")
    print(
        f"folds={len(folds)} groups={validation['scored_group_count']:,} "
        f"coverage={validation['coverage_rate']:.4f}"
    )
    print(f"已輸出: {score_path}")
    print(f"已輸出: {manifest_path}")
    print(f"已輸出: {coverage_path}")
    return 0


__all__ = [
    "FOLD_MANIFEST_FILENAME",
    "FOLD_SCORE_FILENAME",
    "POINT_IN_TIME_SCHEMA_VERSION",
    "main",
    "parse_args",
]


if __name__ == "__main__":
    raise SystemExit(main())
