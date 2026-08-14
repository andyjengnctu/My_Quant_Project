"""Runtime bridge used by signal generation and candidate ranking."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import pandas as pd

from config.breakout_quality import (
    TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    get_breakout_quality_experiment_profile,
)
from filters.breakout_quality.contract import DEFAULT_FILTER_ID
from core.buy_sort import (
    BREAKOUT_QUALITY_RANKING_POLICY_SCORE,
    SUPPORTED_BREAKOUT_QUALITY_RANKING_POLICIES,
)
from filters.breakout_quality.expected_r_calibration import lookup_expected_r
from filters.breakout_quality.excess_r_calibration import lookup_expected_excess_r
from filters.breakout_quality.ranking_score_store import (
    SCORE_SOURCE_CANONICAL_RUNTIME,
    SCORE_SOURCE_SELECTION_POINT_IN_TIME,
    SCORE_SOURCE_CONTINUOUS_RANKER_OOS,
    SUPPORTED_RANKING_SCORE_SOURCES,
    lookup_continuous_ranker_oos_candidate_score,
    lookup_selection_point_in_time_candidate_score,
)
from filters.breakout_quality.binary_pit_score_store import (
    BINARY_PIT_SCORE_SOURCE,
    build_pass_condition_from_binary_point_in_time_scores,
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
    ranking_options: Mapping[str, Any] | None = None
    score_path_override: str | None = None
    score_manifest_path_override: str | None = None




@dataclass(frozen=True)
class BreakoutQualityFilterSourceContext:
    score_source: str = SCORE_SOURCE_CANONICAL_RUNTIME
    manifest_path: str | None = None
    scores_path: str | None = None


BREAKOUT_QUALITY_FILTER_SCORE_SOURCE_ENV = "BREAKOUT_QUALITY_FILTER_SCORE_SOURCE"
BREAKOUT_QUALITY_BINARY_PIT_MANIFEST_ENV = "BREAKOUT_QUALITY_BINARY_PIT_MANIFEST"
BREAKOUT_QUALITY_BINARY_PIT_SCORES_ENV = "BREAKOUT_QUALITY_BINARY_PIT_SCORES"
_FILTER_SOURCE_ENV_KEYS = (
    BREAKOUT_QUALITY_FILTER_SCORE_SOURCE_ENV,
    BREAKOUT_QUALITY_BINARY_PIT_MANIFEST_ENV,
    BREAKOUT_QUALITY_BINARY_PIT_SCORES_ENV,
)

_FILTER_SOURCE_CONTEXT: ContextVar[BreakoutQualityFilterSourceContext] = ContextVar(
    "breakout_quality_filter_source_context",
    default=BreakoutQualityFilterSourceContext(),
)


def get_breakout_quality_filter_source_context() -> BreakoutQualityFilterSourceContext:
    context = _FILTER_SOURCE_CONTEXT.get()
    if context.score_source != SCORE_SOURCE_CANONICAL_RUNTIME:
        return context
    env_source = str(os.environ.get(BREAKOUT_QUALITY_FILTER_SCORE_SOURCE_ENV) or "").strip()
    if not env_source:
        return context
    if env_source != BINARY_PIT_SCORE_SOURCE:
        raise ValueError(f"不支援的breakout-quality filter score source: {env_source!r}")
    manifest_path = str(os.environ.get(BREAKOUT_QUALITY_BINARY_PIT_MANIFEST_ENV) or "").strip()
    scores_path = str(os.environ.get(BREAKOUT_QUALITY_BINARY_PIT_SCORES_ENV) or "").strip()
    if not manifest_path or not scores_path:
        raise ValueError("Binary PIT filter source缺少manifest／scores環境設定")
    return BreakoutQualityFilterSourceContext(
        score_source=env_source,
        manifest_path=manifest_path,
        scores_path=scores_path,
    )


@contextmanager
def breakout_quality_filter_source_context(
    *,
    score_source: str,
    manifest_path: str | None = None,
    scores_path: str | None = None,
) -> Iterator[BreakoutQualityFilterSourceContext]:
    source = str(score_source).strip()
    if source not in {SCORE_SOURCE_CANONICAL_RUNTIME, BINARY_PIT_SCORE_SOURCE}:
        raise ValueError(f"不支援的breakout-quality filter score source: {source!r}")
    if source == BINARY_PIT_SCORE_SOURCE and (
        not str(manifest_path or "").strip() or not str(scores_path or "").strip()
    ):
        raise ValueError("Binary PIT filter source必須指定manifest_path與scores_path")
    context = BreakoutQualityFilterSourceContext(
        score_source=source,
        manifest_path=None if manifest_path is None else str(manifest_path),
        scores_path=None if scores_path is None else str(scores_path),
    )
    token = _FILTER_SOURCE_CONTEXT.set(context)
    try:
        yield context
    finally:
        _FILTER_SOURCE_CONTEXT.reset(token)


@contextmanager
def breakout_quality_filter_source_execution_context(
    *,
    score_source: str,
    manifest_path: str | None = None,
    scores_path: str | None = None,
) -> Iterator[BreakoutQualityFilterSourceContext]:
    """Propagate one filter source to the current process and spawned prep workers."""
    source = str(score_source).strip()
    manifest_text = str(manifest_path or "").strip()
    scores_text = str(scores_path or "").strip()
    if source == BINARY_PIT_SCORE_SOURCE and (not manifest_text or not scores_text):
        raise ValueError("Binary PIT filter source必須指定manifest_path與scores_path")
    before = {key: os.environ.get(key) for key in _FILTER_SOURCE_ENV_KEYS}
    if source == BINARY_PIT_SCORE_SOURCE:
        os.environ[BREAKOUT_QUALITY_FILTER_SCORE_SOURCE_ENV] = source
        os.environ[BREAKOUT_QUALITY_BINARY_PIT_MANIFEST_ENV] = manifest_text
        os.environ[BREAKOUT_QUALITY_BINARY_PIT_SCORES_ENV] = scores_text
    else:
        for key in _FILTER_SOURCE_ENV_KEYS:
            os.environ.pop(key, None)
    try:
        with breakout_quality_filter_source_context(
            score_source=source,
            manifest_path=manifest_path,
            scores_path=scores_path,
        ) as context:
            yield context
    finally:
        for key, prior in before.items():
            if prior is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prior

_RANKING_SOURCE_CONTEXT: ContextVar[BreakoutQualityRankingSourceContext] = ContextVar(
    "breakout_quality_ranking_source_context",
    default=BreakoutQualityRankingSourceContext(),
)


def resolve_project_root_from_runtime() -> str:
    return str(Path(__file__).resolve().parents[2])


def get_breakout_quality_ranking_source_context() -> BreakoutQualityRankingSourceContext:
    return _RANKING_SOURCE_CONTEXT.get()


def breakout_quality_ranking_uses_daily_information_date() -> bool:
    """Return whether the active research ranker must refresh on each completed bar."""

    context = get_breakout_quality_ranking_source_context()
    if context.score_source not in {
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
    context = get_breakout_quality_filter_source_context()
    if context.score_source == BINARY_PIT_SCORE_SOURCE:
        return build_pass_condition_from_binary_point_in_time_scores(
            df,
            ticker=ticker,
            score_threshold=float(score_threshold),
            candidate_condition=candidate_condition,
            manifest_path=str(context.manifest_path),
            scores_path=str(context.scores_path),
        )
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
    information_date=None,
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
    if context.score_source in {
        SCORE_SOURCE_SELECTION_POINT_IN_TIME,
        SCORE_SOURCE_CONTINUOUS_RANKER_OOS,
    }:
        lookup_date = signal_date
        if breakout_quality_ranking_uses_daily_information_date():
            if information_date is None:
                raise ValueError(
                    "daily-universal ranking source必須提供最新已完成交易日 information_date"
                )
            lookup_date = information_date
        if context.score_source == SCORE_SOURCE_SELECTION_POINT_IN_TIME:
            payload = lookup_selection_point_in_time_candidate_score(
                project_root=root,
                ticker=str(ticker),
                signal_date=lookup_date,
                filter_id=str(filter_id),
                model_architecture=str(context.model_architecture),
                experiment_profile=str(context.experiment_profile),
                score_path_override=context.score_path_override,
                manifest_path_override=context.score_manifest_path_override,
            )
        else:
            payload = lookup_continuous_ranker_oos_candidate_score(
                project_root=root,
                ticker=str(ticker),
                signal_date=lookup_date,
                filter_id=str(filter_id),
                model_architecture=str(context.model_architecture),
                experiment_profile=str(context.experiment_profile),
                score_path_override=context.score_path_override,
            )
        options = dict(context.ranking_options or {})
        calibration_path = str(options.get("expected_r_calibration_path") or "").strip()
        excess_calibration_path = str(
            options.get("expected_excess_r_calibration_path") or ""
        ).strip()
        if calibration_path and excess_calibration_path:
            raise ValueError("ranking options不得同時指定Expected-R與Expected Excess-R calibration")
        if calibration_path:
            enriched = dict(payload)
            if bool(enriched.get("available", False)):
                calibration_lookup_path = Path(calibration_path)
                if not calibration_lookup_path.is_absolute():
                    calibration_lookup_path = Path(root).resolve() / calibration_lookup_path
                enriched.update(lookup_expected_r(
                    lookup_path=calibration_lookup_path,
                    ticker=str(ticker),
                    score_date=lookup_date,
                    expected_score=enriched.get("score"),
                ))
            else:
                enriched.update({
                    "expected_r_available": False,
                    "expected_r_unavailable_reason": "ranking_score_unavailable",
                })
            return enriched
        if excess_calibration_path:
            enriched = dict(payload)
            if bool(enriched.get("available", False)):
                calibration_lookup_path = Path(excess_calibration_path)
                if not calibration_lookup_path.is_absolute():
                    calibration_lookup_path = Path(root).resolve() / calibration_lookup_path
                enriched.update(lookup_expected_excess_r(
                    lookup_path=calibration_lookup_path,
                    ticker=str(ticker),
                    score_date=lookup_date,
                    expected_score=enriched.get("score"),
                ))
            else:
                enriched.update({
                    "expected_excess_r_available": False,
                    "expected_excess_r_unavailable_reason": "ranking_score_unavailable",
                })
            return enriched
        return payload
    raise ValueError(f"不支援的breakout-quality ranking score source: {context.score_source!r}")


__all__ = [
    "BreakoutQualityFilterSourceContext",
    "BreakoutQualityRankingSourceContext",
    "breakout_quality_filter_source_context",
    "breakout_quality_ranking_source_context",
    "build_breakout_quality_filter_pass_condition",
    "get_breakout_quality_filter_source_context",
    "get_breakout_quality_ranking_source_context",
    "breakout_quality_ranking_uses_daily_information_date",
    "resolve_breakout_quality_candidate_rank",
    "resolve_project_root_from_runtime",
]
