"""Canonical profile-aware sample-provider dispatch for continuous-ranker consumers."""

from __future__ import annotations

from pathlib import Path

from config.breakout_quality import (
    TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS,
    TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    get_breakout_quality_experiment_profile,
)
from filters.breakout_quality.continuous_ranker_data import (
    ContinuousRankerDataBundle,
    load_continuous_ranker_data,
)
from filters.breakout_quality.daily_ranker_data import load_daily_universal_ranker_data
from filters.breakout_quality.workflow_io import PROJECT_ROOT


def load_profile_continuous_ranker_data(
    *,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
    preload_feature_bank: bool,
    allow_stale_source: bool,
    project_root: str | Path = PROJECT_ROOT,
) -> ContinuousRankerDataBundle:
    """Load the canonical sample provider selected by the experiment profile."""

    profile = get_breakout_quality_experiment_profile(experiment_profile)
    if profile.training_sample_scope == TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS:
        return load_continuous_ranker_data(
            filter_id=filter_id,
            model_architecture=model_architecture,
            experiment_profile=experiment_profile,
            preload_feature_bank=bool(preload_feature_bank),
            allow_stale_source=bool(allow_stale_source),
            project_root=project_root,
        )
    if profile.training_sample_scope == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS:
        return load_daily_universal_ranker_data(
            filter_id=filter_id,
            model_architecture=model_architecture,
            experiment_profile=experiment_profile,
            preload_feature_bank=bool(preload_feature_bank),
            allow_stale_source=bool(allow_stale_source),
            project_root=project_root,
        )
    raise ValueError(
        "不支援的continuous ranker sample scope: "
        f"{profile.training_sample_scope!r}"
    )


__all__ = ["load_profile_continuous_ranker_data"]
