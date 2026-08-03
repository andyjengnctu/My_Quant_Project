"""Runtime bridge used by signal generation and candidate ranking."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from filters.breakout_quality.contract import DEFAULT_FILTER_ID
from core.buy_sort import (
    BREAKOUT_QUALITY_RANKING_POLICY_SCORE,
    SUPPORTED_BREAKOUT_QUALITY_RANKING_POLICIES,
)
from filters.breakout_quality.ranking_score_store import (
    SCORE_SOURCE_CANONICAL_RUNTIME,
    SCORE_SOURCE_SELECTION_POINT_IN_TIME,
    SUPPORTED_RANKING_SCORE_SOURCES,
    lookup_selection_point_in_time_candidate_score,
)
from filters.breakout_quality.score_store import (
    build_pass_condition_from_score_table,
    lookup_breakout_quality_candidate_score,
)


@dataclass(frozen=True)
class BreakoutQualityRankingSourceContext:
    score_source: str = SCORE_SOURCE_CANONICAL_RUNTIME
    model_architecture: str | None = None
    experiment_profile: str | None = None
    ranking_policy: str = BREAKOUT_QUALITY_RANKING_POLICY_SCORE


_RANKING_SOURCE_CONTEXT: ContextVar[BreakoutQualityRankingSourceContext] = ContextVar(
    "breakout_quality_ranking_source_context",
    default=BreakoutQualityRankingSourceContext(),
)


def resolve_project_root_from_runtime() -> str:
    return str(Path(__file__).resolve().parents[2])


def get_breakout_quality_ranking_source_context() -> BreakoutQualityRankingSourceContext:
    return _RANKING_SOURCE_CONTEXT.get()


@contextmanager
def breakout_quality_ranking_source_context(
    *,
    score_source: str,
    model_architecture: str | None = None,
    experiment_profile: str | None = None,
    ranking_policy: str = BREAKOUT_QUALITY_RANKING_POLICY_SCORE,
) -> Iterator[BreakoutQualityRankingSourceContext]:
    source = str(score_source).strip()
    if source not in SUPPORTED_RANKING_SCORE_SOURCES:
        raise ValueError(f"不支援的breakout-quality ranking score source: {source!r}")
    if source == SCORE_SOURCE_SELECTION_POINT_IN_TIME:
        if not str(model_architecture or "").strip() or not str(experiment_profile or "").strip():
            raise ValueError("Selection PIT ranking source必須指定model architecture與experiment profile")
    resolved_policy = str(ranking_policy).strip()
    if resolved_policy not in SUPPORTED_BREAKOUT_QUALITY_RANKING_POLICIES:
        raise ValueError(f"不支援的breakout-quality ranking policy: {resolved_policy!r}")
    context = BreakoutQualityRankingSourceContext(
        score_source=source,
        model_architecture=None if model_architecture is None else str(model_architecture),
        experiment_profile=None if experiment_profile is None else str(experiment_profile),
        ranking_policy=resolved_policy,
    )
    token = _RANKING_SOURCE_CONTEXT.set(context)
    try:
        yield context
    finally:
        _RANKING_SOURCE_CONTEXT.reset(token)


def build_breakout_quality_filter_pass_condition(
    df: pd.DataFrame,
    *,
    ticker: str,
    high_len: int,
    score_threshold: float,
    candidate_condition: np.ndarray,
    filter_id: str = DEFAULT_FILTER_ID,
    project_root: str | None = None,
) -> np.ndarray:
    root = resolve_project_root_from_runtime() if project_root is None else str(project_root)
    return build_pass_condition_from_score_table(
        df,
        ticker=ticker,
        high_len=int(high_len),
        score_threshold=float(score_threshold),
        candidate_condition=candidate_condition,
        project_root=root,
        filter_id=str(filter_id),
    )


def resolve_breakout_quality_candidate_rank(
    *,
    ticker: str,
    signal_date,
    high_len: int,
    filter_id: str = DEFAULT_FILTER_ID,
    project_root: str | None = None,
) -> dict:
    root = resolve_project_root_from_runtime() if project_root is None else str(project_root)
    context = get_breakout_quality_ranking_source_context()
    if context.score_source == SCORE_SOURCE_CANONICAL_RUNTIME:
        payload = lookup_breakout_quality_candidate_score(
            project_root=root,
            ticker=str(ticker),
            signal_date=signal_date,
            high_len=int(high_len),
            filter_id=str(filter_id),
        )
        payload = dict(payload)
        payload.setdefault("score_source", SCORE_SOURCE_CANONICAL_RUNTIME)
        return payload
    if context.score_source == SCORE_SOURCE_SELECTION_POINT_IN_TIME:
        return lookup_selection_point_in_time_candidate_score(
            project_root=root,
            ticker=str(ticker),
            signal_date=signal_date,
            filter_id=str(filter_id),
            model_architecture=str(context.model_architecture),
            experiment_profile=str(context.experiment_profile),
        )
    raise ValueError(f"不支援的breakout-quality ranking score source: {context.score_source!r}")


__all__ = [
    "BreakoutQualityRankingSourceContext",
    "breakout_quality_ranking_source_context",
    "build_breakout_quality_filter_pass_condition",
    "get_breakout_quality_ranking_source_context",
    "resolve_breakout_quality_candidate_rank",
    "resolve_project_root_from_runtime",
]
