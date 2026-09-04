"""PIT-safe predicted-safety context shared by conditional/Pure-MFE consumers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.breakout_quality_runtime import (
    CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_SAFETY,
)
from config.breakout_quality import (
    PREDICTED_SAFETY_CONDITIONAL_MFE_TARGET_ID,
    PREDICTED_SAFETY_CONTEXT_PURE_MFE_TARGET_ID,
)
from filters.breakout_quality.conditional_mfe_safety import (
    build_same_date_residual_percentile,
)
from filters.breakout_quality.continuous_ranker_data import (
    build_same_date_percentile_targets,
)
from filters.breakout_quality.predicted_context_artifact import (
    get_predicted_context_artifact_spec,
    load_validated_predicted_context,
    resolve_predicted_context_dir,
)

_SPEC = get_predicted_context_artifact_spec(
    CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_SAFETY
)
PREDICTED_SAFETY_CONTEXT_FILENAME = _SPEC.filename
PREDICTED_SAFETY_CONTEXT_MANIFEST_FILENAME = _SPEC.manifest_filename
STAGE1_PROFILE = _SPEC.stage1_profile
STAGE1_ARCHITECTURE = _SPEC.stage1_architecture
STAGE1_RESEARCH_ID = _SPEC.stage1_research_id
STAGE1_SEED = _SPEC.stage1_seed
CONTEXT_COLUMN = _SPEC.context_column


@dataclass(frozen=True)
class PredictedSafetyConditionalMfeTargets:
    pure_mfe_percentile: np.ndarray
    residual: np.ndarray
    residual_percentile: np.ndarray

    @property
    def training_target(self) -> np.ndarray:
        return np.asarray(self.residual_percentile, dtype=np.float32)


def resolve_predicted_safety_context_dir(
    project_root: str | Path,
    *,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
) -> Path:
    return resolve_predicted_context_dir(
        CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_SAFETY,
        project_root,
        filter_id=filter_id,
        model_architecture=model_architecture,
        experiment_profile=experiment_profile,
    )


def predicted_safety_context_contract() -> dict[str, Any]:
    return _SPEC.contract()


def load_validated_predicted_safety_context(
    project_root: str | Path,
    *,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
    expected_dataset_policy: dict[str, object] | None = None,
    expected_dataset_artifacts: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    return load_validated_predicted_context(
        CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_SAFETY,
        project_root,
        filter_id=filter_id,
        model_architecture=model_architecture,
        experiment_profile=experiment_profile,
        expected_dataset_policy=expected_dataset_policy,
        expected_dataset_artifacts=expected_dataset_artifacts,
    )


def build_predicted_safety_conditional_mfe_targets(
    group_table: pd.DataFrame,
    valid_mask: np.ndarray,
    predicted_safety_percentile: np.ndarray,
) -> PredictedSafetyConditionalMfeTargets:
    required = {"date", "target_favorable_r"}
    missing = sorted(required - set(group_table.columns))
    if missing:
        raise ValueError(f"MR-13AD target缺少canonical欄位: {missing}")
    valid = np.asarray(valid_mask, dtype=bool)
    context = np.asarray(predicted_safety_percentile, dtype=np.float32)
    if valid.shape != context.shape or len(valid) != len(group_table):
        raise ValueError("MR-13AD target/context shape不一致")
    favorable = pd.to_numeric(
        group_table["target_favorable_r"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    component_valid = valid & np.isfinite(favorable) & np.isfinite(context)
    if not np.array_equal(component_valid, valid):
        raise ValueError(
            "MR-13AD valid rows必須同時有finite Pure-MFE與PIT-safe safety context"
        )
    dates = pd.to_datetime(group_table["date"], errors="raise").dt.normalize()
    pure_mfe = build_same_date_percentile_targets(favorable, valid, dates)
    residual, residual_percentile = build_same_date_residual_percentile(
        context,
        pure_mfe,
        valid,
        dates,
    )
    return PredictedSafetyConditionalMfeTargets(
        pure_mfe_percentile=np.asarray(pure_mfe, dtype=np.float32),
        residual=residual,
        residual_percentile=residual_percentile,
    )


__all__ = [
    "CONTEXT_COLUMN",
    "PREDICTED_SAFETY_CONDITIONAL_MFE_TARGET_ID",
    "PREDICTED_SAFETY_CONTEXT_PURE_MFE_TARGET_ID",
    "PREDICTED_SAFETY_CONTEXT_FILENAME",
    "PREDICTED_SAFETY_CONTEXT_MANIFEST_FILENAME",
    "STAGE1_ARCHITECTURE",
    "STAGE1_PROFILE",
    "STAGE1_RESEARCH_ID",
    "STAGE1_SEED",
    "PredictedSafetyConditionalMfeTargets",
    "build_predicted_safety_conditional_mfe_targets",
    "load_validated_predicted_safety_context",
    "predicted_safety_context_contract",
    "resolve_predicted_safety_context_dir",
]
