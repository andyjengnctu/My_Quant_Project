"""Build the isolated PIT-safe Stage-1 safety context required by MR-13AD."""

from __future__ import annotations

from pathlib import Path

from filters.breakout_quality.predicted_safety_context import (
    CONTEXT_COLUMN,
    PREDICTED_SAFETY_CONTEXT_FILENAME,
    PREDICTED_SAFETY_CONTEXT_MANIFEST_FILENAME,
    STAGE1_ARCHITECTURE,
    STAGE1_PROFILE,
    STAGE1_RESEARCH_ID,
    STAGE1_SEED,
    predicted_safety_context_contract,
    resolve_predicted_safety_context_dir,
)
from services.breakout_quality.predicted_context import (
    build_predicted_context,
    load_stage1_context_scores,
)


def _load_stage1_scores(path: Path, *, phase: str):
    return load_stage1_context_scores(
        path,
        phase=phase,
        predicted_score_column="predicted_safety_score",
        context_column=CONTEXT_COLUMN,
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
    """Build/rebuild MR-13AD Selection cross-fit + fixed pre-OOS safety context."""

    return build_predicted_context(
        project_root=project_root,
        filter_id=filter_id,
        model_architecture=model_architecture,
        experiment_profile=experiment_profile,
        dataset=dataset,
        max_tickers=max_tickers,
        stage1_profile=STAGE1_PROFILE,
        stage1_architecture=STAGE1_ARCHITECTURE,
        stage1_research_id=STAGE1_RESEARCH_ID,
        stage1_seed=STAGE1_SEED,
        context_dir_resolver=resolve_predicted_safety_context_dir,
        context_filename=PREDICTED_SAFETY_CONTEXT_FILENAME,
        context_manifest_filename=PREDICTED_SAFETY_CONTEXT_MANIFEST_FILENAME,
        predicted_score_column="predicted_safety_score",
        context_column=CONTEXT_COLUMN,
        context_contract=predicted_safety_context_contract(),
        context_label="MR-13AD predicted-safety",
    )


__all__ = ["_load_stage1_scores", "build_predicted_safety_context"]
