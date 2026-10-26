"""Runtime bridge used by signal generation and candidate ranking."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping
import math

import numpy as np
import pandas as pd

from core.breakout_quality_ranking_context import (
    BreakoutQualityRankingSourceContext,
    build_breakout_quality_workflow_ranking_context,
    breakout_quality_ranking_source_context,
    breakout_quality_ranking_uses_daily_information_date,
    get_breakout_quality_ranking_source_context,
    audit_oracle_ranking_policy_context,
    SCORE_SOURCE_AUDIT_ORACLE_FUTURE_TRUTH,
)
from core.breakout_quality_runtime_port import bind_breakout_quality_runtime_port
from filters.breakout_quality.contract import DEFAULT_FILTER_ID
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
from filters.breakout_quality.workflow_runtime_score_store import (
    lookup_workflow_runtime_candidate_score,
)


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

# Private to this adapter: future truth is unavailable to all formal score stores.
_ORACLE_TRUTH_SCORES: ContextVar[Mapping[tuple[str, str], float] | None] = ContextVar(
    "breakout_quality_audit_oracle_truth_scores", default=None,
)
_ORACLE_TRUTH_KEY_MODE: ContextVar[str] = ContextVar(
    "breakout_quality_audit_oracle_truth_key_mode", default="information_date",
)
# Only an explicitly verified daily-universal score source may treat an absent
# feature-ineligible stock-day as an unavailable ranking. Other Oracle truth
# sources still fail closed for missing keys; neither policy imputes a score.
_ORACLE_TRUTH_MISSING_KEY_POLICY: ContextVar[str] = ContextVar(
    "breakout_quality_audit_oracle_missing_key_policy", default="block",
)
_ORACLE_DAILY_UNSCORED_POLICY = "unavailable_outside_mature_feature_universe"
_ORACLE_TRUTH_SCORE_START: ContextVar[str | None] = ContextVar(
    "breakout_quality_audit_oracle_score_start", default=None,
)


def _candidate_score_lookup_date(
    *, signal_date, information_date, use_information_date: bool,
    missing_information_date_message: str,
):
    """One decision-date rule for Oracle truth and formal PIT ranking.

    Score source may differ; the latest completed-bar versus original-signal
    decision is shared.  Event-anchored hindsight labels remain explicitly
    signal-origin based and do not change Daily-universal score semantics.
    """
    if use_information_date:
        if information_date is None:
            raise ValueError(missing_information_date_message)
        return information_date
    return signal_date


@contextmanager
def audit_oracle_truth_score_context(
    scores: Mapping[tuple[str, str], float],
    *, ranking_policy: str, ranking_options: Mapping[str, object] | None = None,
    key_mode: str = "information_date",
    missing_key_policy: str = "block",
    score_start: str | None = None,
) -> Iterator[None]:
    """Bind verified hindsight scores during one isolated Audit replay.

    The daily-universal producer certifies that its score map covers every
    mature feature-eligible stock-day. An absent key therefore means this
    portfolio candidate has no DL-score-eligible Target, exactly as in PIT
    ranking; it remains a normal Min candidate with an unavailable DL rank.
    Event and price Oracle score sources stay strict by default.
    """
    if not scores:
        raise ValueError("Oracle Audit缺少成熟的Ground Truth scores")
    if key_mode not in {"information_date", "signal_origin_date"}:
        raise ValueError("Oracle Audit未知的 score key mode")
    if missing_key_policy not in {"block", _ORACLE_DAILY_UNSCORED_POLICY}:
        raise ValueError("Oracle Audit未知的 missing-key policy")
    if missing_key_policy != "block" and key_mode != "information_date":
        raise ValueError("Only the mature daily-stock-day Oracle may allow unavailable ranks")
    if score_start is not None:
        score_start = pd.Timestamp(score_start).strftime("%Y-%m-%d")
        if key_mode != "information_date":
            raise ValueError("Oracle score start requires an information-date lookup")
    if _ORACLE_TRUTH_SCORES.get() is not None:
        raise RuntimeError("Oracle Audit不得巢狀覆寫未來真值score")
    with audit_oracle_ranking_policy_context(
        ranking_policy=ranking_policy, ranking_options=ranking_options,
    ):
        token = _ORACLE_TRUTH_SCORES.set(scores)
        key_token = _ORACLE_TRUTH_KEY_MODE.set(key_mode)
        missing_token = _ORACLE_TRUTH_MISSING_KEY_POLICY.set(missing_key_policy)
        start_token = _ORACLE_TRUTH_SCORE_START.set(score_start)
        try:
            yield
        finally:
            _ORACLE_TRUTH_SCORE_START.reset(start_token)
            _ORACLE_TRUTH_MISSING_KEY_POLICY.reset(missing_token)
            _ORACLE_TRUTH_KEY_MODE.reset(key_token)
            _ORACLE_TRUTH_SCORES.reset(token)


def resolve_project_root_from_runtime() -> str:
    return str(Path(__file__).resolve().parents[2])


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
    if context.score_source == SCORE_SOURCE_AUDIT_ORACLE_FUTURE_TRUTH:
        truth = _ORACLE_TRUTH_SCORES.get()
        if truth is None or information_date is None:
            raise RuntimeError("Oracle Audit來源或information_date缺失；禁止降級為正式PIT score")
        lookup_day = _candidate_score_lookup_date(
            signal_date=signal_date, information_date=information_date,
            use_information_date=_ORACLE_TRUTH_KEY_MODE.get() != "signal_origin_date",
            missing_information_date_message="Oracle Audit來源或information_date缺失；禁止降級為正式PIT score",
        )
        if lookup_day is None:
            raise RuntimeError("Oracle Audit缺少原始signal date；禁止不明來源fallback")
        date = pd.Timestamp(lookup_day).strftime("%Y-%m-%d")
        score_start = _ORACLE_TRUTH_SCORE_START.get()
        if score_start is not None and date < score_start:
            return {
                "score": None, "available": False,
                "unavailable_reason": "outside_score_period",
                "score_date": date, "filter_id": str(filter_id),
                "score_source": SCORE_SOURCE_AUDIT_ORACLE_FUTURE_TRUTH,
            }
        key = (str(ticker), date)
        if key not in truth:
            if _ORACLE_TRUTH_MISSING_KEY_POLICY.get() == _ORACLE_DAILY_UNSCORED_POLICY:
                return {
                    "score": None, "available": False,
                    "unavailable_reason": "missing_ticker_date_score",
                    "score_date": date, "filter_id": str(filter_id),
                    "score_source": SCORE_SOURCE_AUDIT_ORACLE_FUTURE_TRUTH,
                }
            raise ValueError(f"Oracle Audit未覆蓋可交易候選，停止回測: ticker={ticker}, date={date}")
        value = truth[key]
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"Oracle Audit非有限真值分數，停止回測: ticker={ticker}, date={date}")
        return {
            "score": float(value), "available": True, "unavailable_reason": "",
            "score_date": date, "filter_id": str(filter_id),
            "score_source": SCORE_SOURCE_AUDIT_ORACLE_FUTURE_TRUTH,
        }
    if context.score_source == SCORE_SOURCE_CANONICAL_RUNTIME:
        lookup_date = _candidate_score_lookup_date(
            signal_date=signal_date, information_date=information_date,
            use_information_date=breakout_quality_ranking_uses_daily_information_date(),
            missing_information_date_message=(
                "daily-universal canonical runtime必須提供最新已完成交易日 information_date"
            ),
        )
        if context.model_architecture and context.experiment_profile:
            payload = lookup_workflow_runtime_candidate_score(
                project_root=root,
                ticker=str(ticker),
                information_date=lookup_date,
                high_len=int(high_len),
                filter_id=str(filter_id),
                model_architecture=str(context.model_architecture),
                experiment_profile=str(context.experiment_profile),
            )
        else:
            payload = lookup_breakout_quality_candidate_score(
                project_root=root,
                ticker=str(ticker),
                signal_date=lookup_date,
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
        options = dict(context.ranking_options or {})
        primary_score_column = str(
            options.get("primary_score_column") or ""
        ).strip() or None
        lookup_date = _candidate_score_lookup_date(
            signal_date=signal_date, information_date=information_date,
            use_information_date=breakout_quality_ranking_uses_daily_information_date(),
            missing_information_date_message=(
                "daily-universal ranking source必須提供最新已完成交易日 information_date"
            ),
        )
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
                score_column=primary_score_column,
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
                score_column=primary_score_column,
            )
        safety_dl_id = str(options.get("safety_dl_id") or "").strip()
        safety_filter_id = str(options.get("safety_filter_id") or "").strip()
        safety_profile = str(options.get("safety_experiment_profile") or "").strip()
        safety_architecture = str(options.get("safety_model_architecture") or "").strip()
        safety_source = str(options.get("safety_score_source") or "").strip()
        if safety_dl_id or safety_filter_id or safety_profile or safety_architecture or safety_source:
            if not all((safety_dl_id, safety_filter_id, safety_profile, safety_architecture, safety_source)):
                raise ValueError("safety-constrained ranking options缺少完整secondary source identity")
            safety_score_path_override = str(
                options.get("safety_score_path_override") or ""
            ).strip()
            safety_score_column = str(
                options.get("safety_score_column") or ""
            ).strip() or None
            safety_manifest_path_override = str(
                options.get("safety_score_manifest_path_override") or ""
            ).strip()
            if safety_source == SCORE_SOURCE_CONTINUOUS_RANKER_OOS:
                safety_payload = lookup_continuous_ranker_oos_candidate_score(
                    project_root=root,
                    ticker=str(ticker),
                    signal_date=lookup_date,
                    filter_id=safety_filter_id,
                    model_architecture=safety_architecture,
                    experiment_profile=safety_profile,
                    score_path_override=(
                        None if not safety_score_path_override else safety_score_path_override
                    ),
                    score_column=safety_score_column,
                )
            elif safety_source == SCORE_SOURCE_SELECTION_POINT_IN_TIME:
                safety_payload = lookup_selection_point_in_time_candidate_score(
                    project_root=root,
                    ticker=str(ticker),
                    signal_date=lookup_date,
                    filter_id=safety_filter_id,
                    model_architecture=safety_architecture,
                    experiment_profile=safety_profile,
                    score_path_override=(
                        None if not safety_score_path_override else safety_score_path_override
                    ),
                    manifest_path_override=(
                        None if not safety_manifest_path_override else safety_manifest_path_override
                    ),
                    score_column=safety_score_column,
                )
            else:
                raise ValueError(
                    "safety-constrained secondary source只支援selection_point_in_time或continuous_ranker_oos"
                )
            payload = dict(payload)
            payload.update({
                "safety_dl_id": safety_dl_id,
                "safety_score": safety_payload.get("score"),
                "safety_available": bool(safety_payload.get("available", False)),
                "safety_unavailable_reason": str(safety_payload.get("unavailable_reason") or ""),
                "safety_score_date": str(safety_payload.get("score_date") or ""),
                "safety_score_source": str(safety_payload.get("score_source") or ""),
                "safety_model_architecture": safety_architecture,
                "safety_experiment_profile": safety_profile,
            })
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


# Formal Filter runtime is the adapter for Core's narrow callback port.
# This binding has no score/seed/state payload and is idempotent across consumers.
bind_breakout_quality_runtime_port(
    filter_pass_condition=build_breakout_quality_filter_pass_condition,
    candidate_rank_resolver=resolve_breakout_quality_candidate_rank,
)


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
