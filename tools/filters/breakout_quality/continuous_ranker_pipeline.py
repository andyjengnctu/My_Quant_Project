"""Reusable continuous-ranker data, training, and inference pipeline.

The learning implementation remains centralized in ``train_continuous_ranker``.  This module
provides a public orchestration API for full-Selection training and rolling point-in-time folds
without duplicating loss, epoch-selection, refit, or inference semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.breakout_quality import get_breakout_quality_experiment_profile
from filters.breakout_quality.continuous_target import (
    load_validated_continuous_target_arrays,
)
from filters.breakout_quality.contract import DEFAULT_LABEL_POLICY
from filters.breakout_quality.models.factory import (
    count_trainable_parameters,
    require_torch,
)
from filters.breakout_quality.models.spec import (
    get_model_spec,
    validate_model_sequence_length,
)
from filters.breakout_quality.splits import resolve_breakout_quality_outer_policy
from filters.breakout_quality.torch_runtime import resolve_torch_execution_plan
from tools.filters.breakout_quality.common import (
    PROJECT_ROOT,
    load_validated_dataset_bundle,
)
from tools.filters.breakout_quality import train_continuous_ranker as ranker_impl


@dataclass(frozen=True)
class ContinuousRankerDataBundle:
    summary: dict[str, Any]
    events: pd.DataFrame
    labels: np.ndarray
    feature_bank: np.ndarray
    event_group_index: np.ndarray
    group_table: pd.DataFrame
    group_context: np.ndarray
    raw_target: np.ndarray
    target_valid: np.ndarray
    target_manifest: dict[str, Any]
    outer_policy: dict[str, Any]
    profile: Any
    model_spec: Any


def _validate_group_consistency(events: pd.DataFrame) -> None:
    required = {"ticker", "date", "group_index", "label_eval_end_date"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"continuous ranker dataset 缺少欄位: {missing}")
    work = events[["group_index", "ticker", "date", "label_eval_end_date"]].copy()
    grouped = work.groupby("group_index", sort=False)
    mixed = grouped.agg(
        ticker_count=("ticker", "nunique"),
        date_count=("date", "nunique"),
    )
    # ``nunique`` drops missing values by default.  A terminal group whose label horizon is
    # incomplete therefore produced count=0 even when every row consistently carried the same
    # missing value.  Treat missing as one state, while mixed missing/completed dates still fail.
    mixed["label_end_count"] = grouped["label_eval_end_date"].nunique(dropna=False)
    invalid = mixed[
        (mixed["ticker_count"] != 1)
        | (mixed["date_count"] != 1)
        | (mixed["label_end_count"] != 1)
    ]
    if not invalid.empty:
        raise ValueError(
            "continuous ranker 同group的ticker/date/label_eval_end_date必須一致: "
            f"invalid_groups={len(invalid)}"
        )


def calculate_spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    """Return the canonical continuous-ranker Spearman metric."""

    return ranker_impl._spearman(x, y)


def load_continuous_ranker_data(
    *,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
    preload_feature_bank: bool,
    allow_stale_source: bool,
    project_root: str | Path = PROJECT_ROOT,
) -> ContinuousRankerDataBundle:
    profile = get_breakout_quality_experiment_profile(experiment_profile)
    model_spec = get_model_spec(model_architecture)
    if (
        bool(model_spec.requires_market_set)
        or bool(model_spec.use_dataset_context)
        or bool(model_spec.derived_context_features)
    ):
        raise ValueError("continuous ranker pipeline 只支援 sequence-only architecture")

    summary, indexed_features, context, labels, events = load_validated_dataset_bundle(
        filter_id,
        expected_policy=DEFAULT_LABEL_POLICY.as_manifest_payload(),
        require_current_source=not bool(allow_stale_source),
    )
    _validate_group_consistency(events)
    feature_bank = (
        np.array(indexed_features.feature_bank, dtype=np.float32, copy=True, order="C")
        if bool(preload_feature_bank)
        else indexed_features.feature_bank
    )
    event_group_index = np.asarray(indexed_features.event_group_index, dtype=np.int64)
    labels_array = np.asarray(labels, dtype=np.int64)
    group_table = ranker_impl._group_table(events, event_group_index, labels_array)
    representative_rows = group_table["event_row"].to_numpy(dtype=np.int64)
    group_table = group_table.copy()
    group_table["label_eval_end_date"] = pd.to_datetime(
        events.iloc[representative_rows]["label_eval_end_date"], errors="raise"
    ).dt.normalize().to_numpy()
    group_context = np.asarray(context[representative_rows], dtype=np.float32)
    if bool(preload_feature_bank):
        group_context = np.array(group_context, dtype=np.float32, copy=True, order="C")
    validate_model_sequence_length(model_spec, int(feature_bank.shape[1]))

    target_manifest, raw_target, target_valid = load_validated_continuous_target_arrays(
        Path(project_root),
        filter_id,
        target_id=str(profile.continuous_target_id),
        expected_group_count=int(len(group_table)),
        expected_dataset_policy=summary.get("policy"),
        expected_dataset_artifacts=summary.get("dataset_artifacts"),
    )
    outer_policy = resolve_breakout_quality_outer_policy(
        Path(project_root),
        source_data_end_date=ranker_impl._source_data_end(summary, events),
    )
    return ContinuousRankerDataBundle(
        summary=dict(summary),
        events=events.copy(),
        labels=labels_array,
        feature_bank=feature_bank,
        event_group_index=event_group_index,
        group_table=group_table,
        group_context=group_context,
        raw_target=np.asarray(raw_target, dtype=np.float32),
        target_valid=np.asarray(target_valid, dtype=bool),
        target_manifest=dict(target_manifest),
        outer_policy=dict(outer_policy),
        profile=profile,
        model_spec=model_spec,
    )


def resolve_ranker_execution_plan(args):
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
    return torch, plan


def build_percentile_target(
    bundle: ContinuousRankerDataBundle,
    group_ids: np.ndarray,
) -> np.ndarray:
    ids = np.asarray(group_ids, dtype=np.int64)
    mask = np.zeros(bundle.raw_target.shape, dtype=bool)
    mask[ids] = True
    return ranker_impl.build_daily_percentile_targets(
        bundle.raw_target,
        mask,
        bundle.group_table["date"],
    )


def select_epoch(
    torch,
    bundle: ContinuousRankerDataBundle,
    percentile_target: np.ndarray,
    train_ids: np.ndarray,
    validation_ids: np.ndarray,
    *,
    args,
    plan,
) -> dict[str, Any]:
    return ranker_impl._select_epoch(
        torch,
        bundle.feature_bank,
        bundle.group_context,
        bundle.group_table,
        bundle.raw_target,
        percentile_target,
        np.asarray(train_ids, dtype=np.int64),
        np.asarray(validation_ids, dtype=np.int64),
        args=args,
        plan=plan,
    )


def fit_final(
    torch,
    bundle: ContinuousRankerDataBundle,
    percentile_target: np.ndarray,
    final_ids: np.ndarray,
    *,
    epochs: int,
    args,
    plan,
):
    return ranker_impl._fit_final(
        torch,
        bundle.feature_bank,
        bundle.group_context,
        percentile_target,
        np.asarray(final_ids, dtype=np.int64),
        epochs=int(epochs),
        args=args,
        plan=plan,
        phase_label="Fold歷史資料重訓",
    )


def predict_scores(
    torch,
    model,
    bundle: ContinuousRankerDataBundle,
    group_ids: np.ndarray,
    *,
    batch_size: int,
    plan,
) -> np.ndarray:
    return ranker_impl._predict_scores(
        torch,
        model,
        bundle.feature_bank,
        bundle.group_context,
        np.asarray(group_ids, dtype=np.int64),
        batch_size=int(batch_size),
        plan=plan,
    )


def build_checkpoint_payload(
    model,
    bundle: ContinuousRankerDataBundle,
    *,
    args,
    plan,
    selected_epoch: int,
    fold_contract: dict[str, Any],
) -> dict[str, Any]:
    trainable_parameter_count = count_trainable_parameters(model)
    total_parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())
    return {
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "feature_count": int(bundle.feature_bank.shape[2]),
        "context_count": int(bundle.group_context.shape[1]),
        "sequence_length": int(bundle.feature_bank.shape[1]),
        "model_spec": bundle.model_spec.as_manifest_payload(),
        "experiment_profile": str(args.experiment_profile),
        "experiment_settings": bundle.profile.as_manifest_payload(),
        "training_objective": bundle.profile.training_objective,
        "training_label_scope": bundle.profile.training_label_scope,
        "continuous_target_contract": bundle.target_manifest.get("target_contract"),
        "selected_epoch": int(selected_epoch),
        "fold_contract": dict(fold_contract),
        "torch_execution": plan.as_manifest_payload(),
        "trainable_parameter_count": int(trainable_parameter_count),
        "total_parameter_count": int(total_parameter_count),
    }


__all__ = [
    "calculate_spearman",
    "ContinuousRankerDataBundle",
    "build_checkpoint_payload",
    "build_percentile_target",
    "fit_final",
    "load_continuous_ranker_data",
    "predict_scores",
    "resolve_ranker_execution_plan",
    "select_epoch",
]
