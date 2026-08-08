"""Reusable continuous-ranker data, training, and inference pipeline.

The learning implementation remains centralized in ``train_continuous_ranker``.  This module
provides a public orchestration API for full-Selection training and rolling point-in-time folds
without duplicating loss, epoch-selection, refit, or inference semantics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.breakout_quality import get_breakout_quality_experiment_profile
from filters.breakout_quality.models.factory import (
    count_trainable_parameters,
    require_torch,
)
from filters.breakout_quality.torch_runtime import resolve_torch_execution_plan
from filters.breakout_quality.continuous_ranker_data import (
    ContinuousRankerDataBundle,
    load_continuous_ranker_data,
)
from filters.breakout_quality.workflow_io import PROJECT_ROOT
from tools.filters.breakout_quality import train_continuous_ranker as ranker_impl


def calculate_spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    """Return the canonical continuous-ranker Spearman metric."""

    return ranker_impl._spearman(x, y)


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
