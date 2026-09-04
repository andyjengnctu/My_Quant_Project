"""PIT-safe predicted-upside context and conditional low-adverse target contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.breakout_quality_runtime import (
    CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_UPSIDE,
)
from core.breakout_quality_runtime import (
    PREDICTED_UPSIDE_CONDITIONAL_LOW_ADVERSE_TARGET_ID,
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
    CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_UPSIDE
)
PREDICTED_UPSIDE_CONTEXT_FILENAME = _SPEC.filename
PREDICTED_UPSIDE_CONTEXT_MANIFEST_FILENAME = _SPEC.manifest_filename
STAGE1_PROFILE = _SPEC.stage1_profile
STAGE1_ARCHITECTURE = _SPEC.stage1_architecture
STAGE1_RESEARCH_ID = _SPEC.stage1_research_id
STAGE1_SEED = _SPEC.stage1_seed
CONTEXT_COLUMN = _SPEC.context_column


@dataclass(frozen=True)
class PredictedUpsideConditionalSafetyTargets:
    low_adverse_percentile: np.ndarray
    residual: np.ndarray
    residual_percentile: np.ndarray

    @property
    def training_target(self) -> np.ndarray:
        return np.asarray(self.residual_percentile, dtype=np.float32)


def resolve_predicted_upside_context_dir(
    project_root: str | Path,
    *,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
) -> Path:
    return resolve_predicted_context_dir(
        CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_UPSIDE,
        project_root,
        filter_id=filter_id,
        model_architecture=model_architecture,
        experiment_profile=experiment_profile,
    )


def predicted_upside_context_contract() -> dict[str, Any]:
    return _SPEC.contract()


def load_validated_predicted_upside_context(
    project_root: str | Path,
    *,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
    expected_dataset_policy: dict[str, object] | None = None,
    expected_dataset_artifacts: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    return load_validated_predicted_context(
        CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_UPSIDE,
        project_root,
        filter_id=filter_id,
        model_architecture=model_architecture,
        experiment_profile=experiment_profile,
        expected_dataset_policy=expected_dataset_policy,
        expected_dataset_artifacts=expected_dataset_artifacts,
    )


def build_predicted_upside_conditional_low_adverse_targets(
    group_table: pd.DataFrame,
    valid_mask: np.ndarray,
    predicted_upside_percentile: np.ndarray,
) -> PredictedUpsideConditionalSafetyTargets:
    required = {"date", "target_adverse_r"}
    missing = sorted(required - set(group_table.columns))
    if missing:
        raise ValueError(f"MR-13AC target缺少canonical欄位: {missing}")
    valid = np.asarray(valid_mask, dtype=bool)
    context = np.asarray(predicted_upside_percentile, dtype=np.float32)
    if valid.shape != context.shape or len(valid) != len(group_table):
        raise ValueError("MR-13AC target/context shape不一致")
    adverse = pd.to_numeric(group_table["target_adverse_r"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    component_valid = valid & np.isfinite(adverse) & np.isfinite(context)
    if not np.array_equal(component_valid, valid):
        raise ValueError(
            "MR-13AC valid rows必須同時有finite adverse與PIT-safe upside context"
        )
    dates = pd.to_datetime(group_table["date"], errors="raise").dt.normalize()
    low_adverse = build_same_date_percentile_targets(-adverse, valid, dates)
    residual, residual_percentile = build_same_date_residual_percentile(
        context,
        low_adverse,
        valid,
        dates,
    )
    return PredictedUpsideConditionalSafetyTargets(
        low_adverse_percentile=np.asarray(low_adverse, dtype=np.float32),
        residual=residual,
        residual_percentile=residual_percentile,
    )


__all__ = [
    "CONTEXT_COLUMN",
    "PREDICTED_UPSIDE_CONDITIONAL_LOW_ADVERSE_TARGET_ID",
    "PREDICTED_UPSIDE_CONTEXT_FILENAME",
    "PREDICTED_UPSIDE_CONTEXT_MANIFEST_FILENAME",
    "STAGE1_ARCHITECTURE",
    "STAGE1_PROFILE",
    "STAGE1_RESEARCH_ID",
    "STAGE1_SEED",
    "PredictedUpsideConditionalSafetyTargets",
    "build_predicted_upside_conditional_low_adverse_targets",
    "load_validated_predicted_upside_context",
    "predicted_upside_context_contract",
    "resolve_predicted_upside_context_dir",
]
