"""Breakout-quality ranking policy and ContextVar source: core-owned runtime contract.

Filter score lookup remains in filters.breakout_quality.runtime; Core only consumes
this lightweight policy/context and a dependency-injected lookup port.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

from core.breakout_quality_registry import (
    TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    get_breakout_quality_experiment_profile,
)
from core.breakout_quality_policy import (
    WORKFLOW_SCORE_SOURCE_CANONICAL_RUNTIME,
    get_breakout_quality_workflow_settings,
)
from core.buy_sort import (
    BREAKOUT_QUALITY_RANKING_POLICY_SCORE,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_CONSTRAINED_OPTIMAL,
    SUPPORTED_BREAKOUT_QUALITY_RANKING_POLICIES,
)

# These score-source identities were previously declared in the filter score store.
# They remain byte-for-byte identical for PIT, reuse, and score manifest contracts.
SCORE_SOURCE_CANONICAL_RUNTIME = WORKFLOW_SCORE_SOURCE_CANONICAL_RUNTIME
SCORE_SOURCE_SELECTION_POINT_IN_TIME = "selection_point_in_time"
SCORE_SOURCE_CONTINUOUS_RANKER_OOS = "continuous_ranker_oos"
SUPPORTED_RANKING_SCORE_SOURCES = (
    SCORE_SOURCE_CANONICAL_RUNTIME,
    SCORE_SOURCE_SELECTION_POINT_IN_TIME,
    SCORE_SOURCE_CONTINUOUS_RANKER_OOS,
)

@dataclass(frozen=True)
class BreakoutQualityRankingSourceContext:
    score_source: str = SCORE_SOURCE_CANONICAL_RUNTIME
    model_architecture: str | None = None
    experiment_profile: str | None = None
    ranking_policy: str = BREAKOUT_QUALITY_RANKING_POLICY_SCORE
    ranking_options: Mapping[str, Any] | None = None
    score_path_override: str | None = None
    score_manifest_path_override: str | None = None


_RANKING_SOURCE_CONTEXT: ContextVar[BreakoutQualityRankingSourceContext | None] = ContextVar(
    "breakout_quality_ranking_source_context",
    default=None,
)


def build_breakout_quality_workflow_ranking_context() -> BreakoutQualityRankingSourceContext:
    """Resolve the formal runtime ranking contract from the workflow SSOT."""

    workflow = get_breakout_quality_workflow_settings()
    if not workflow.runtime_strategy_enabled:
        return BreakoutQualityRankingSourceContext()
    policy = str(workflow.runtime_ranking_policy).strip()
    if policy != BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_CONSTRAINED_OPTIMAL:
        raise ValueError(
            "正式workflow runtime只接受Gate已驗證的MR-13E exact constrained policy: "
            f"{policy!r}"
        )
    if policy not in SUPPORTED_BREAKOUT_QUALITY_RANKING_POLICIES:
        raise ValueError(f"正式workflow runtime ranking policy未由core支援: {policy!r}")
    return BreakoutQualityRankingSourceContext(
        score_source=SCORE_SOURCE_CANONICAL_RUNTIME,
        model_architecture=str(workflow.model_architecture),
        experiment_profile=str(workflow.experiment_profile),
        ranking_policy=policy,
        ranking_options=dict(workflow.runtime_ranking_options),
    )


def get_breakout_quality_ranking_source_context() -> BreakoutQualityRankingSourceContext:
    context = _RANKING_SOURCE_CONTEXT.get()
    return build_breakout_quality_workflow_ranking_context() if context is None else context


def breakout_quality_ranking_uses_daily_information_date() -> bool:
    """Return whether the active research ranker must refresh on each completed bar."""

    context = get_breakout_quality_ranking_source_context()
    if context.score_source not in {
        SCORE_SOURCE_CANONICAL_RUNTIME,
        SCORE_SOURCE_SELECTION_POINT_IN_TIME,
        SCORE_SOURCE_CONTINUOUS_RANKER_OOS,
    }:
        return False
    profile_name = str(context.experiment_profile or "").strip()
    if not profile_name:
        return False
    profile = get_breakout_quality_experiment_profile(profile_name)
    return (
        profile.training_sample_scope
        == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
    )


@contextmanager
def breakout_quality_ranking_source_context(
    *,
    score_source: str,
    model_architecture: str | None = None,
    experiment_profile: str | None = None,
    ranking_policy: str = BREAKOUT_QUALITY_RANKING_POLICY_SCORE,
    ranking_options: Mapping[str, Any] | None = None,
    score_path_override: str | None = None,
    score_manifest_path_override: str | None = None,
) -> Iterator[BreakoutQualityRankingSourceContext]:
    source = str(score_source).strip()
    if source not in SUPPORTED_RANKING_SCORE_SOURCES:
        raise ValueError(f"不支援的breakout-quality ranking score source: {source!r}")
    if source in {SCORE_SOURCE_SELECTION_POINT_IN_TIME, SCORE_SOURCE_CONTINUOUS_RANKER_OOS}:
        if not str(model_architecture or "").strip() or not str(experiment_profile or "").strip():
            raise ValueError("研究ranking source必須指定model architecture與experiment profile")
    resolved_policy = str(ranking_policy).strip()
    if resolved_policy not in SUPPORTED_BREAKOUT_QUALITY_RANKING_POLICIES:
        raise ValueError(f"不支援的breakout-quality ranking policy: {resolved_policy!r}")
    context = BreakoutQualityRankingSourceContext(
        score_source=source,
        model_architecture=None if model_architecture is None else str(model_architecture),
        experiment_profile=None if experiment_profile is None else str(experiment_profile),
        ranking_policy=resolved_policy,
        ranking_options=(None if ranking_options in (None, {}) else dict(ranking_options)),
        score_path_override=(None if score_path_override in (None, "") else str(score_path_override)),
        score_manifest_path_override=(
            None if score_manifest_path_override in (None, "") else str(score_manifest_path_override)
        ),
    )
    token = _RANKING_SOURCE_CONTEXT.set(context)
    try:
        yield context
    finally:
        _RANKING_SOURCE_CONTEXT.reset(token)
