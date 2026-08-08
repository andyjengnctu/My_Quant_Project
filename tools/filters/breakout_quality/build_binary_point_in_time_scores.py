"""Build expanding-window point-in-time scores for the binary DL filter."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from config.breakout_quality import (
    BREAKOUT_QUALITY_ALLOW_TF32,
    BREAKOUT_QUALITY_CLASS_WEIGHT_MODE,
    BREAKOUT_QUALITY_DEFAULT_BATCH_SIZE,
    BREAKOUT_QUALITY_DEFAULT_EPOCHS,
    BREAKOUT_QUALITY_DEFAULT_GRADIENT_CLIP_NORM,
    BREAKOUT_QUALITY_DEFAULT_LEARNING_RATE,
    BREAKOUT_QUALITY_DEFAULT_WEIGHT_DECAY,
    BREAKOUT_QUALITY_DETERMINISTIC_ALGORITHMS,
    BREAKOUT_QUALITY_EARLY_STOPPING_MIN_DELTA,
    BREAKOUT_QUALITY_EARLY_STOPPING_PATIENCE,
    BREAKOUT_QUALITY_EVALUATION_BATCH_SIZE,
    BREAKOUT_QUALITY_EVALUATION_WORKERS,
    BREAKOUT_QUALITY_INNER_VALIDATION_MONTHS,
    BREAKOUT_QUALITY_MIXED_PRECISION_DTYPE,
    BREAKOUT_QUALITY_PARALLEL_SPLIT_EVALUATION,
    BREAKOUT_QUALITY_PRELOAD_FEATURE_BANK,
    BREAKOUT_QUALITY_TIME_WEIGHT_MODE,
    BREAKOUT_QUALITY_TORCH_DEVICE,
    BREAKOUT_QUALITY_TRAIN_PREFETCH_BATCHES,
    BREAKOUT_QUALITY_USE_MIXED_PRECISION,
    TRAINING_LABEL_SCOPE_ALL,
    TRAINING_OBJECTIVE_BINARY_CLASSIFICATION,
    get_breakout_quality_experiment_profile,
    get_breakout_quality_workflow_settings,
)
from core.runtime_utils import get_taipei_now
from filters.breakout_quality.augmentation import (
    build_training_augmentation_plan,
    validate_training_augmentation_sequence_length,
)
from filters.breakout_quality.binary_pit_score_store import (
    BINARY_PIT_REQUIRED_COLUMNS,
    load_binary_point_in_time_score_table,
)
from filters.breakout_quality.contract import (
    expected_label_policy_for_filter_id,
    LABEL_PASS,
    LABEL_REJECT,
    SCORE_COLUMN,
)
from filters.breakout_quality.inference import strict_parallel_batched_logits
from filters.breakout_quality.model import require_torch
from filters.breakout_quality.models.spec import (
    get_model_spec,
    validate_model_sequence_length,
)
from filters.breakout_quality.torch_runtime import (
    SUPPORTED_MIXED_PRECISION_DTYPES,
    SUPPORTED_TORCH_DEVICES,
    resolve_torch_execution_plan,
)
from core.console_report import (
    print_artifact_paths,
    project_relative_display_path,
    render_key_values,
    render_section,
    render_table,
    render_title,
)
from tools.filters.breakout_quality.build_point_in_time_scores import (
    AUTO_SCORE_START_VALUE,
    _build_fold_periods,
    _fold_group_ids,
    _minimum_count_failures,
    _resolve_score_start,
)
from filters.breakout_quality.workflow_io import (
    PROJECT_ROOT,
    load_validated_dataset_bundle,
)
from tools.filters.breakout_quality import train as train_impl

SCHEMA_VERSION = 2
OUTPUT_RELATIVE_DIR = Path(
    "models/research/breakout_quality/binary_point_in_time_scores"
)
FOLD_SCORE_FILENAME = "scores.csv"
FOLD_MANIFEST_FILENAME = "manifest.json"


@dataclass
class BinaryPitBundle:
    summary: dict[str, Any]
    features: Any
    context: np.ndarray
    labels: np.ndarray
    events: pd.DataFrame
    event_group_index: np.ndarray
    group_table: pd.DataFrame
    target_valid: np.ndarray
    profile: Any
    model_spec: Any


def _build_binary_group_table(
    events: pd.DataFrame,
    event_group_index: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    """Build one deterministic representative row per feature group.

    Binary datasets may contain multiple event rows for the same ticker/date
    feature group.  Trade-path datasets intentionally keep one teacher-active
    PASS/REJECT row while marking the other high_len rows EXCLUDED.  EXCLUDED
    rows are not binary targets and therefore must not be treated as a mixed
    label conflict.  A group is invalid only when its eligible PASS/REJECT rows
    disagree.
    """

    required = {"ticker", "date", "group_index"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise KeyError(f"Binary PIT group table缺少events欄位: {missing}")

    normalized_group_index = np.asarray(event_group_index, dtype=np.int64)
    normalized_labels = np.asarray(labels, dtype=np.int64)
    if normalized_group_index.ndim != 1 or normalized_labels.ndim != 1:
        raise ValueError("Binary PIT event_group_index與labels必須是1D")
    if len(events) != normalized_group_index.size or len(events) != normalized_labels.size:
        raise ValueError(
            "Binary PIT events／event_group_index／labels長度不一致: "
            f"events={len(events)}, group_index={normalized_group_index.size}, "
            f"labels={normalized_labels.size}"
        )
    if len(events) == 0:
        raise ValueError("Binary PIT dataset沒有event rows")

    frame = events[["ticker", "date", "group_index"]].copy()
    frame["event_row"] = np.arange(len(frame), dtype=np.int64)
    frame["label"] = normalized_labels
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    csv_group_index = pd.to_numeric(frame["group_index"], errors="raise").to_numpy(
        dtype=np.int64
    )
    if not np.array_equal(csv_group_index, normalized_group_index):
        raise ValueError("Binary PIT events group_index與event_group_index不一致")

    representatives: list[pd.Series] = []
    for group_index, group_rows in frame.groupby("group_index", sort=True):
        if group_rows["ticker"].astype(str).nunique() != 1 or group_rows["date"].nunique() != 1:
            raise ValueError(
                "Binary PIT同group的ticker/date必須一致: "
                f"group_index={int(group_index)}"
            )
        eligible = group_rows[group_rows["label"].isin((LABEL_REJECT, LABEL_PASS))]
        eligible_labels = eligible["label"].drop_duplicates().tolist()
        if len(eligible_labels) > 1:
            raise ValueError(
                "Binary PIT發現同group混合eligible binary label: "
                f"group_index={int(group_index)}, labels={eligible_labels}"
            )
        representative = (
            eligible.sort_values("event_row", kind="mergesort").iloc[0]
            if not eligible.empty
            else group_rows.sort_values("event_row", kind="mergesort").iloc[0]
        )
        representatives.append(representative)

    group = pd.DataFrame(representatives).sort_values("group_index", kind="mergesort")
    observed = group["group_index"].to_numpy(dtype=np.int64)
    expected = np.arange(len(group), dtype=np.int64)
    if not np.array_equal(observed, expected):
        raise ValueError("Binary PIT要求group_index連續完整")
    representative_rows = group["event_row"].to_numpy(dtype=np.int64)
    if not np.array_equal(normalized_group_index[representative_rows], expected):
        raise ValueError("Binary PIT representative與event_group_index不一致")
    return group.reset_index(drop=True)


def _parse_args(argv=None):
    settings = get_breakout_quality_workflow_settings()
    parser = argparse.ArgumentParser(
        description=(
            "建立Binary DL Filter expanding-window PIT scores；每個score date只使用"
            "該日前已完成label的歷史資料。"
        )
    )
    parser.add_argument("--filter-id", default=settings.filter_id)
    parser.add_argument("--model-architecture", default=settings.model_architecture)
    parser.add_argument("--experiment-profile", default=settings.experiment_profile)
    parser.add_argument("--score-start-date", default=AUTO_SCORE_START_VALUE)
    parser.add_argument("--score-end-date", default="auto")
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
    parser.add_argument(
        "--evaluation-workers", type=int, default=BREAKOUT_QUALITY_EVALUATION_WORKERS
    )
    parser.add_argument(
        "--parallel-split-evaluation",
        action=argparse.BooleanOptionalAction,
        default=BREAKOUT_QUALITY_PARALLEL_SPLIT_EVALUATION,
    )
    parser.add_argument(
        "--train-prefetch-batches",
        type=int,
        default=BREAKOUT_QUALITY_TRAIN_PREFETCH_BATCHES,
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
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args) -> None:
    if min(int(args.fold_months), int(args.inner_validation_months), int(args.epochs)) < 1:
        raise ValueError("fold／validation months與epochs必須>=1")
    if int(args.batch_size) < 2 or int(args.evaluation_batch_size) < 1:
        raise ValueError("batch-size>=2且evaluation-batch-size>=1")
    if float(args.lr) <= 0 or float(args.weight_decay) < 0:
        raise ValueError("lr必須>0且weight-decay不得為負")


def _load_bundle(args) -> BinaryPitBundle:
    profile = get_breakout_quality_experiment_profile(str(args.experiment_profile))
    if profile.training_objective != TRAINING_OBJECTIVE_BINARY_CLASSIFICATION:
        raise ValueError("Binary PIT只接受binary classification profile")
    if profile.training_label_scope != TRAINING_LABEL_SCOPE_ALL:
        raise ValueError("Binary PIT只接受all-label classification profile")
    model_spec = get_model_spec(str(args.model_architecture))
    if bool(model_spec.requires_market_set) or bool(model_spec.use_dataset_context) or bool(model_spec.derived_context_features):
        raise ValueError("Binary PIT目前只支援sequence-only architecture")
    summary, features, context, labels, events = load_validated_dataset_bundle(
        str(args.filter_id),
        expected_policy=expected_label_policy_for_filter_id(str(args.filter_id)),
        require_current_source=True,
    )
    features, context, labels = train_impl._preload_training_arrays(
        features,
        context,
        labels,
        enabled=bool(args.preload_feature_bank),
    )
    event_group_index = np.asarray(features.event_group_index, dtype=np.int64)
    group_table = _build_binary_group_table(events, event_group_index, labels)
    representative_rows = group_table["event_row"].to_numpy(dtype=np.int64)
    group_table["label_eval_end_date"] = pd.to_datetime(
        events.iloc[representative_rows]["label_eval_end_date"], errors="coerce"
    ).dt.normalize().to_numpy()
    group_labels = group_table["label"].to_numpy(dtype=np.int64)
    label_end_valid = group_table["label_eval_end_date"].notna().to_numpy(dtype=bool)
    target_valid = label_end_valid & np.isin(group_labels, [LABEL_REJECT, LABEL_PASS])
    validate_model_sequence_length(model_spec, int(features.shape[1]))
    augmentation = build_training_augmentation_plan(
        name=profile.augmentation_name,
        parameters=profile.augmentation_parameters(),
    )
    validate_training_augmentation_sequence_length(
        augmentation, sequence_length=int(features.shape[1])
    )
    return BinaryPitBundle(
        summary=dict(summary),
        features=features,
        context=np.asarray(context, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        events=events.copy(),
        event_group_index=event_group_index,
        group_table=group_table,
        target_valid=target_valid,
        profile=profile,
        model_spec=model_spec,
    )


def _bundle_for_fold_helpers(bundle: BinaryPitBundle):
    return SimpleNamespace(
        group_table=bundle.group_table,
        target_valid=bundle.target_valid,
        event_group_index=bundle.event_group_index,
        profile=SimpleNamespace(training_label_scope=TRAINING_LABEL_SCOPE_ALL),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()




def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.name),
        "sha256": _sha256(path),
        "size_bytes": int(path.stat().st_size),
    }

def _fold_fingerprint(args, bundle: BinaryPitBundle, fold, ids) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "filter_id": str(args.filter_id),
        "model_architecture": str(args.model_architecture),
        "experiment_profile": str(args.experiment_profile),
        "fold_id": str(fold["fold_id"]),
        "score_start": str(pd.Timestamp(fold["score_start"]).date()),
        "score_end": str(pd.Timestamp(fold["score_end"]).date()),
        "validation_start": str(pd.Timestamp(ids["validation_start"]).date()),
        "counts": {key: int(len(ids[key])) for key in ("train_ids", "validation_ids", "final_ids", "score_ids")},
        "dataset_artifacts": bundle.summary.get("dataset_artifacts"),
        "policy": bundle.summary.get("policy"),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "seed": int(args.seed),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _representative_rows(bundle: BinaryPitBundle, group_ids: np.ndarray) -> np.ndarray:
    return bundle.group_table.iloc[np.asarray(group_ids, dtype=np.int64)]["event_row"].to_numpy(dtype=np.int64)


def _predict_scores(torch, model, bundle, rows, *, args, execution_plan) -> np.ndarray:
    logits = strict_parallel_batched_logits(
        torch,
        model,
        bundle.features,
        bundle.context,
        indices=np.asarray(rows, dtype=np.int64),
        batch_size=int(args.evaluation_batch_size),
        workers=int(args.evaluation_workers),
        execution_plan=execution_plan,
        market_set_bank=None,
    )
    with torch.no_grad():
        return (
            torch.softmax(torch.from_numpy(logits), dim=1)[:, LABEL_PASS]
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )


def _train_fold(torch, bundle, fold, ids, *, args, execution_plan, fold_dir: Path) -> dict[str, Any]:
    train_rows = _representative_rows(bundle, ids["train_ids"])
    validation_rows = _representative_rows(bundle, ids["validation_ids"])
    final_rows = _representative_rows(bundle, ids["final_ids"])
    score_rows = _representative_rows(bundle, ids["score_ids"])
    train_sampling, train_sampling_summary = train_impl._resolve_training_sampling_indices(
        bundle.events,
        bundle.labels,
        train_rows,
        mode=bundle.profile.training_sampling_mode,
        model_spec=bundle.model_spec,
    )
    final_sampling, final_sampling_summary = train_impl._resolve_training_sampling_indices(
        bundle.events,
        bundle.labels,
        final_rows,
        mode=bundle.profile.training_sampling_mode,
        model_spec=bundle.model_spec,
    )
    augmentation = build_training_augmentation_plan(
        name=bundle.profile.augmentation_name,
        parameters=bundle.profile.augmentation_parameters(),
    )
    lr_params = bundle.profile.lr_schedule_parameters()
    selected = train_impl._select_epoch_with_inner_validation(
        torch,
        X=bundle.features,
        C=bundle.context,
        market_set_bank=None,
        y=bundle.labels,
        events=bundle.events,
        train_idx=train_rows,
        training_sampling_idx=train_sampling,
        training_sampling_summary=train_sampling_summary,
        validation_idx=validation_rows,
        max_epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        optimizer_name=bundle.profile.optimizer_name,
        lr_schedule_name=bundle.profile.lr_schedule_name,
        lr_warmup_fraction=float(lr_params.get("warmup_fraction", 0.0)),
        lr_minimum_ratio=float(lr_params.get("minimum_lr_ratio", 1.0)),
        augmentation_plan=augmentation,
        learning_rate=float(args.lr),
        weight_decay=float(args.weight_decay),
        gradient_clip_norm=float(args.gradient_clip_norm),
        class_weight_mode=BREAKOUT_QUALITY_CLASS_WEIGHT_MODE,
        time_weight_mode=BREAKOUT_QUALITY_TIME_WEIGHT_MODE,
        training_weight_reduction=bundle.profile.training_weight_reduction,
        seed=int(args.seed),
        patience=int(args.early_stopping_patience),
        min_delta=float(args.early_stopping_min_delta),
        evaluation_batch_size=int(args.evaluation_batch_size),
        evaluation_workers=int(args.evaluation_workers),
        parallel_split_evaluation=bool(args.parallel_split_evaluation),
        train_prefetch_batches=int(args.train_prefetch_batches),
        execution_plan=execution_plan,
        pretrained_encoder_state=None,
    )
    final_fit = train_impl._fit_full_selection(
        torch,
        X=bundle.features,
        C=bundle.context,
        market_set_bank=None,
        y=bundle.labels,
        events=bundle.events,
        train_idx=final_rows,
        training_sampling_idx=final_sampling,
        training_sampling_summary=final_sampling_summary,
        epochs=int(selected["best_epoch"]),
        target_optimizer_steps=None,
        batch_size=int(args.batch_size),
        optimizer_name=bundle.profile.optimizer_name,
        lr_schedule_name=bundle.profile.lr_schedule_name,
        lr_warmup_fraction=float(lr_params.get("warmup_fraction", 0.0)),
        lr_minimum_ratio=float(lr_params.get("minimum_lr_ratio", 1.0)),
        augmentation_plan=augmentation,
        learning_rate=float(args.lr),
        weight_decay=float(args.weight_decay),
        gradient_clip_norm=float(args.gradient_clip_norm),
        class_weight_mode=BREAKOUT_QUALITY_CLASS_WEIGHT_MODE,
        time_weight_mode=BREAKOUT_QUALITY_TIME_WEIGHT_MODE,
        training_weight_reduction=bundle.profile.training_weight_reduction,
        seed=int(args.seed),
        phase_name="full_refit",
        evaluation_batch_size=int(args.evaluation_batch_size),
        evaluation_workers=int(args.evaluation_workers),
        train_prefetch_batches=int(args.train_prefetch_batches),
        execution_plan=execution_plan,
        pretrained_encoder_state=None,
    )
    model = final_fit["model"]
    model.eval()
    scores = _predict_scores(
        torch, model, bundle, score_rows, args=args, execution_plan=execution_plan
    )
    groups = bundle.group_table.iloc[np.asarray(ids["score_ids"], dtype=np.int64)]
    cutoff = pd.to_datetime(
        bundle.group_table.iloc[np.asarray(ids["final_ids"], dtype=np.int64)]["label_eval_end_date"],
        errors="raise",
    ).max()
    frame = pd.DataFrame(
        {
            "ticker": groups["ticker"].astype(str).to_numpy(),
            "date": pd.to_datetime(groups["date"], errors="raise").dt.strftime("%Y-%m-%d").to_numpy(),
            "group_index": groups["group_index"].to_numpy(dtype=np.int64),
            SCORE_COLUMN: scores,
            "fold_id": str(fold["fold_id"]),
            "model_information_cutoff": str(pd.Timestamp(cutoff).date()),
        }
    )
    fold_dir.mkdir(parents=True, exist_ok=True)
    score_path = fold_dir / FOLD_SCORE_FILENAME
    frame.to_csv(score_path, index=False, encoding="utf-8-sig")
    checkpoint_path = fold_dir / "model.pt"
    torch.save(
        {
            "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "model_spec": bundle.model_spec.as_manifest_payload(),
            "experiment_profile": str(args.experiment_profile),
            "best_epoch": int(selected["best_epoch"]),
            "fold_id": str(fold["fold_id"]),
        },
        checkpoint_path,
    )
    return {
        "frame": frame,
        "selected_epoch": int(selected["best_epoch"]),
        "best_validation_loss": float(selected["best_validation_loss"]),
        "score_path": score_path,
        "checkpoint_path": checkpoint_path,
        "model_information_cutoff": str(pd.Timestamp(cutoff).date()),
    }


def _validate_combined(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(BINARY_PIT_REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Binary PIT combined scores缺少欄位: {missing}")
    work = frame[list(BINARY_PIT_REQUIRED_COLUMNS)].copy()
    work["date"] = pd.to_datetime(work["date"], errors="raise").dt.strftime("%Y-%m-%d")
    work["model_information_cutoff"] = pd.to_datetime(
        work["model_information_cutoff"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    if work.duplicated(["ticker", "date"]).any() or work.duplicated(["group_index"]).any():
        raise ValueError("Binary PIT combined scores出現重複group")
    scores = pd.to_numeric(work[SCORE_COLUMN], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(scores).all() or bool(((scores < 0) | (scores > 1)).any()):
        raise ValueError("Binary PIT combined scores含不合法分數")
    if bool(
        (
            pd.to_datetime(work["model_information_cutoff"], errors="raise")
            >= pd.to_datetime(work["date"], errors="raise")
        ).any()
    ):
        raise ValueError("Binary PIT cutoff未早於score date")
    return work.sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)


def build_binary_point_in_time_scores(*, project_root=PROJECT_ROOT, argv=None) -> dict[str, Any]:
    args = _parse_args(argv)
    _validate_args(args)
    root = Path(project_root).resolve()
    bundle = _load_bundle(args)
    helper_bundle = _bundle_for_fold_helpers(bundle)
    settings = get_breakout_quality_workflow_settings()
    group_dates = pd.to_datetime(bundle.group_table["date"], errors="raise").dt.normalize()
    selection_start = group_dates.min()
    score_end = group_dates.max() if str(args.score_end_date).strip().lower() == "auto" else pd.Timestamp(args.score_end_date).normalize()
    score_start, start_resolution = _resolve_score_start(
        args.score_start_date,
        bundle=helper_bundle,
        settings=settings,
        selection_start=selection_start,
        score_end=score_end,
        fold_months=int(args.fold_months),
        validation_months=int(args.inner_validation_months),
    )
    folds = _build_fold_periods(score_start, score_end, fold_months=int(args.fold_months))
    fold_details = []
    for fold in folds:
        ids = _fold_group_ids(
            helper_bundle, fold, validation_months=int(args.inner_validation_months)
        )
        failures = _minimum_count_failures(settings, ids)
        if failures:
            raise ValueError(f"{fold['fold_id']} PIT最小group契約失敗: {failures}")
        fold_details.append((fold, ids))
    output_dir = (
        root
        / OUTPUT_RELATIVE_DIR
        / str(args.filter_id)
        / str(args.model_architecture)
        / str(args.experiment_profile)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        render_title("Binary DL Filter Point-in-Time Scores")
        + "\n"
        + render_key_values(
            (
                ("期間", f"{score_start.date()} ～ {score_end.date()}"),
                ("Folds", len(folds)),
                ("Validation", f"{int(args.inner_validation_months)} months"),
                ("Profile", args.experiment_profile),
            )
        )
    )
    print(render_section("Fold plan"))
    print(
        render_table(
            ("Fold", "Score period", "Train", "Validation", "Refit", "Score"),
            [
                (
                    fold["fold_id"],
                    f"{fold['score_start'].date()}～{fold['score_end'].date()}",
                    len(ids["train_ids"]),
                    len(ids["validation_ids"]),
                    len(ids["final_ids"]),
                    len(ids["score_ids"]),
                )
                for fold, ids in fold_details
            ],
        )
    )
    plan_path = output_dir / "plan.json"
    plan_payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "PLAN_ONLY" if args.plan_only else "RUNNING",
        "score_period": {"start": str(score_start.date()), "end": str(score_end.date())},
        "score_start_resolution": start_resolution,
        "fold_count": len(folds),
        "folds": [
            {
                "fold_id": fold["fold_id"],
                "score_start": str(fold["score_start"].date()),
                "score_end": str(fold["score_end"].date()),
                "train_groups": len(ids["train_ids"]),
                "validation_groups": len(ids["validation_ids"]),
                "refit_groups": len(ids["final_ids"]),
                "score_groups": len(ids["score_ids"]),
            }
            for fold, ids in fold_details
        ],
    }
    plan_path.write_text(json.dumps(plan_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.plan_only:
        print_artifact_paths((("Binary PIT plan", plan_path),), project_root=root)
        return plan_payload

    torch, _nn = require_torch()
    execution_plan = resolve_torch_execution_plan(
        torch,
        requested_device=str(args.device),
        mixed_precision=bool(args.mixed_precision),
        mixed_precision_dtype=str(args.mixed_precision_dtype),
        deterministic_algorithms=bool(args.deterministic_algorithms),
        allow_tf32=bool(args.allow_tf32),
    )
    if execution_plan.device_type == "cpu":
        torch.set_num_threads(1)
    if execution_plan.device_type == "cuda" and bool(args.parallel_split_evaluation):
        raise ValueError("CUDA Binary PIT不允許parallel-split-evaluation")
    combined = []
    fold_records = []
    started = time.perf_counter()
    for index, (fold, ids) in enumerate(fold_details, 1):
        fold_dir = output_dir / "folds" / str(fold["fold_id"])
        fingerprint = _fold_fingerprint(args, bundle, fold, ids)
        fold_manifest_path = fold_dir / FOLD_MANIFEST_FILENAME
        reusable = False
        checkpoint_path = fold_dir / "model.pt"
        score_path = fold_dir / FOLD_SCORE_FILENAME
        if (
            bool(args.resume)
            and fold_manifest_path.is_file()
            and score_path.is_file()
            and checkpoint_path.is_file()
        ):
            try:
                prior = json.loads(fold_manifest_path.read_text(encoding="utf-8"))
                artifacts = dict(prior.get("artifacts") or {})
                reusable = (
                    str(prior.get("fingerprint") or "") == fingerprint
                    and artifacts.get("scores") == _file_record(score_path)
                    and artifacts.get("checkpoint") == _file_record(checkpoint_path)
                )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError):
                reusable = False
        print(f"\n[{index}/{len(folds)}] {fold['fold_id']}" + ("｜reuse" if reusable else "｜train"))
        if reusable:
            frame = pd.read_csv(fold_dir / FOLD_SCORE_FILENAME, encoding="utf-8-sig")
            record = dict(prior)
        else:
            result = _train_fold(
                torch,
                bundle,
                fold,
                ids,
                args=args,
                execution_plan=execution_plan,
                fold_dir=fold_dir,
            )
            frame = result.pop("frame")
            record = {
                "schema_version": SCHEMA_VERSION,
                "fold_id": fold["fold_id"],
                "fingerprint": fingerprint,
                "score_period": {
                    "start": str(fold["score_start"].date()),
                    "end": str(fold["score_end"].date()),
                },
                "counts": {
                    "train_groups": len(ids["train_ids"]),
                    "validation_groups": len(ids["validation_ids"]),
                    "refit_groups": len(ids["final_ids"]),
                    "score_groups": len(ids["score_ids"]),
                },
                **result,
                "score_path": FOLD_SCORE_FILENAME,
                "checkpoint_path": "model.pt",
                "artifacts": {
                    "scores": _file_record(fold_dir / FOLD_SCORE_FILENAME),
                    "checkpoint": _file_record(fold_dir / "model.pt"),
                },
            }
            fold_manifest_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
        combined.append(frame)
        fold_records.append(record)
        gc.collect()
        if execution_plan.device_type == "cuda":
            torch.cuda.empty_cache()
    score_frame = _validate_combined(pd.concat(combined, ignore_index=True))
    scores_path = output_dir / "scores.csv"
    score_frame.to_csv(scores_path, index=False, encoding="utf-8-sig")
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "schema_type": "binary_point_in_time_scores",
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE",
        "filter_id": str(args.filter_id),
        "model_architecture": str(args.model_architecture),
        "experiment_profile": str(args.experiment_profile),
        "training_objective": TRAINING_OBJECTIVE_BINARY_CLASSIFICATION,
        "shared_group_score_broadcast": True,
        "group_representative_contract": (
            "prefer_first_eligible_pass_reject_row_else_first_event_row; "
            "excluded_rows_do_not_create_mixed_label_conflicts"
        ),
        "threshold": 0.5,
        "score_period": {
            "start": str(pd.to_datetime(score_frame["date"]).min().date()),
            "end": str(pd.to_datetime(score_frame["date"]).max().date()),
        },
        "score_table": {
            "path": "scores.csv",
            "row_count": len(score_frame),
            "columns": list(score_frame.columns),
            "sha256": _sha256(scores_path),
            "size_bytes": scores_path.stat().st_size,
        },
        "fold_count": len(fold_records),
        "folds": fold_records,
        "dataset_policy": bundle.summary.get("policy"),
        "dataset_artifacts": bundle.summary.get("dataset_artifacts"),
        "information_contract": "each score row uses only labels completed before its fold score_start",
        "created_at": get_taipei_now().isoformat(),
        "elapsed_sec": round(time.perf_counter() - started, 3),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    load_binary_point_in_time_score_table.cache_clear()
    load_binary_point_in_time_score_table(str(manifest_path), str(scores_path))
    plan_payload["status"] = "COMPLETE"
    plan_path.write_text(json.dumps(plan_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_artifact_paths(
        (("Binary PIT Scores", scores_path), ("Binary PIT Manifest", manifest_path)),
        project_root=root,
    )
    return manifest


def main(argv=None):
    build_binary_point_in_time_scores(argv=argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
