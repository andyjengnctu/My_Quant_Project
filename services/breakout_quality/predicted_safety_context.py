"""Build the reusable PIT-safe Stage-1 safety context for MR-13AD/MR-13AE."""

from __future__ import annotations

from pathlib import Path

from config.breakout_quality_runtime import (
    CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_SAFETY,
)
from filters.breakout_quality.predicted_context_artifact import (
    get_predicted_context_artifact_spec,
)
from services.breakout_quality.predicted_context import (
    build_registered_predicted_context,
    load_stage1_context_scores,
)

_SPEC = get_predicted_context_artifact_spec(
    CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_SAFETY
)


def _load_stage1_scores(path: Path, *, phase: str):
    return load_stage1_context_scores(
        path,
        phase=phase,
        predicted_score_column=_SPEC.predicted_score_column,
        context_column=_SPEC.context_column,
        context_label="predicted-safety",
    )


def build_predicted_safety_context(
    *,
    project_root: str | Path,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
    dataset: str,
    max_tickers: int = 0,
) -> Path:
    """Build/rebuild canonical Selection cross-fit + fixed pre-OOS safety context."""

    return build_registered_predicted_context(
        CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_SAFETY,
        project_root=project_root,
        filter_id=filter_id,
        model_architecture=model_architecture,
        experiment_profile=experiment_profile,
        dataset=dataset,
        max_tickers=max_tickers,
    )


__all__ = ["_load_stage1_scores", "build_predicted_safety_context"]
