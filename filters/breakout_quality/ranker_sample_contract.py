"""Profile-driven sample and score-eligibility contracts for continuous rankers."""

from __future__ import annotations

from typing import Any

from config.breakout_quality import (
    TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS,
    TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    get_breakout_quality_experiment_profile,
)


def build_score_eligibility_contract(experiment_profile: str | Any) -> dict[str, Any]:
    """Describe which rows may receive runtime/PIT scores without future information."""

    profile = (
        get_breakout_quality_experiment_profile(experiment_profile)
        if isinstance(experiment_profile, str)
        else experiment_profile
    )
    sample_scope = str(profile.training_sample_scope)
    if sample_scope == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS:
        return {
            "schema_version": 1,
            "training_sample_scope": sample_scope,
            "eligibility_basis": "feature_history_only",
            "future_target_required_for_score": False,
            "target_valid_required_for_training": True,
        }
    if sample_scope == TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS:
        return {
            "schema_version": 1,
            "training_sample_scope": sample_scope,
            "eligibility_basis": "canonical_breakout_event_membership",
            "future_target_required_for_score": False,
            "target_valid_required_for_training": True,
        }
    raise ValueError(f"不支援的continuous ranker sample scope: {sample_scope!r}")


__all__ = ["build_score_eligibility_contract"]
