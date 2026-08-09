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

from config.breakout_quality import (
    TRAINING_LABEL_SCOPE_ALL,
    TRAINING_LABEL_SCOPE_PASS_ONLY,
    TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
)
from filters.breakout_quality.models.factory import (
    count_trainable_parameters,
    require_torch,
)
from filters.breakout_quality.torch_runtime import resolve_torch_execution_plan
from filters.breakout_quality.contract import LABEL_PASS, LABEL_REJECT
from filters.breakout_quality.continuous_ranker_data import ContinuousRankerDataBundle
from filters.breakout_quality.profile_ranker_data import load_profile_continuous_ranker_data
from tools.filters.breakout_quality import train_continuous_ranker as ranker_impl




def load_continuous_ranker_data(
    *,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
    preload_feature_bank: bool,
    allow_stale_source: bool,
    project_root: Path | None = None,
) -> ContinuousRankerDataBundle:
    """Compatibility facade for the domain-layer canonical sample provider."""

    kwargs = {
        "filter_id": filter_id,
        "model_architecture": model_architecture,
        "experiment_profile": experiment_profile,
        "preload_feature_bank": bool(preload_feature_bank),
        "allow_stale_source": bool(allow_stale_source),
    }
    if project_root is not None:
        kwargs["project_root"] = Path(project_root)
    return load_profile_continuous_ranker_data(**kwargs)


def build_training_scope_mask(bundle: ContinuousRankerDataBundle) -> np.ndarray:
    """Return the canonical target-eligible mask for this sample universe."""

    target_valid = np.asarray(bundle.target_valid, dtype=bool)
    if bundle.profile.training_sample_scope == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS:
        return target_valid.copy()
    labels = bundle.group_table["label"].to_numpy(dtype=np.int64)
    if bundle.profile.training_label_scope == TRAINING_LABEL_SCOPE_PASS_ONLY:
        return target_valid & (labels == LABEL_PASS)
    if bundle.profile.training_label_scope == TRAINING_LABEL_SCOPE_ALL:
        return target_valid & np.isin(labels, [LABEL_REJECT, LABEL_PASS])
    raise ValueError(
        "不支援的continuous ranker training scope: "
        f"{bundle.profile.training_label_scope}"
    )


def primary_audit_metric_scope(bundle: ContinuousRankerDataBundle) -> str:
    if bundle.profile.training_sample_scope == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS:
        return "all_valid_target"
    return "pass_only_target"




def calculate_descriptive_rank_quality(
    dates: np.ndarray,
    scores: np.ndarray,
    raw_targets: np.ndarray,
    *,
    top_k: int,
    boundary_width: int,
) -> dict[str, Any]:
    """Return canonical same-day rank and Top-K diagnostics for one score scope."""

    date_values = pd.to_datetime(pd.Series(dates), errors="raise").dt.normalize()
    score_values = np.asarray(scores, dtype=np.float64)
    target_values = np.asarray(raw_targets, dtype=np.float64)
    if not (len(date_values) == len(score_values) == len(target_values)):
        raise ValueError("rank quality input長度不一致")
    valid = np.isfinite(score_values) & np.isfinite(target_values)
    date_values = date_values[valid].reset_index(drop=True)
    score_values = score_values[valid]
    target_values = target_values[valid]
    if len(score_values) == 0:
        daily = ranker_impl._daily_rank_metrics(
            np.empty(0, dtype="datetime64[ns]"),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
        return {**daily, "top_k_quality": None}
    percentile = ranker_impl.build_daily_percentile_targets(
        target_values,
        np.ones(len(target_values), dtype=bool),
        date_values,
    )
    daily = ranker_impl._daily_rank_metrics(
        date_values.to_numpy(), score_values, target_values
    )
    top_k_quality = ranker_impl._daily_top_k_metrics(
        date_values.to_numpy(),
        score_values,
        target_values,
        percentile,
        top_k=int(top_k),
        boundary_width=int(boundary_width),
    )
    return {**daily, "top_k_quality": top_k_quality}


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
    evaluate_train_metrics = (
        bundle.profile.training_sample_scope
        != TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
    )
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
        evaluate_train_metrics=evaluate_train_metrics,
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
        bundle.group_table,
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
    "calculate_descriptive_rank_quality",
    "calculate_spearman",
    "ContinuousRankerDataBundle",
    "build_checkpoint_payload",
    "build_percentile_target",
    "build_training_scope_mask",
    "fit_final",
    "load_continuous_ranker_data",
    "predict_scores",
    "primary_audit_metric_scope",
    "resolve_ranker_execution_plan",
    "select_epoch",
]
