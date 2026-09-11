"""Canonical semantic-integrity contract for Market Data V2 local audits.

This contract is deliberately outside the immutable Provider Snapshot identity.
It consumes provider-registry/freshness truth, but local audit semantics must not
mutate bootstrap artifacts merely because verification is strengthened.

Important evidence boundary:
- a completed request + immutable artifact proves local/request completeness;
- an observed date gap against a market calendar is provider-semantic evidence,
  not proof of local corruption, unless the local artifact itself is malformed;
- the stock trading calendar is authoritative only for stock-market datasets,
  not futures/options datasets.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

from core.market_data_dataset_registry import MarketDatasetSpec, get_market_dataset_specs
from core.market_data_freshness_contract import (
    CADENCE_CALENDAR_DAILY,
    CADENCE_CURRENT_VINTAGE,
    CADENCE_EVENT_DRIVEN,
    CADENCE_PERIODIC,
    CADENCE_TRADING_DAILY,
    build_market_data_freshness_contract,
)

DATE_SEMANTIC_AUTHORITATIVE_CALENDAR = "authoritative_calendar"
DATE_SEMANTIC_TRADING_CALENDAR_DENSE = "stock_market_session_dense"
DATE_SEMANTIC_TRADING_CALENDAR_SPARSE = "stock_market_session_sparse"
DATE_SEMANTIC_NOT_APPLICABLE = "not_applicable"
DATE_SEMANTIC_UNVERIFIED = "unverified"

DATE_SEMANTIC_PASS = "PASS"
DATE_SEMANTIC_ATTENTION = "ATTENTION"
DATE_SEMANTIC_PARTIAL = "PARTIAL"
DATE_SEMANTIC_FAIL = "FAIL"
DATE_SEMANTIC_NOT_APPLICABLE_STATUS = "NOT_APPLICABLE"
DATE_SEMANTIC_UNVERIFIED_STATUS = "UNVERIFIED"

AUTHORITATIVE_TRADING_CALENDAR_DATASET = "TaiwanStockTradingDate"

# Local-only semantic audit anchors.  TaiwanStockTradingDate is a schedule/calendar
# candidate feed, but exceptional closures (e.g. typhoon closures) can still appear
# there.  Actual stock-market session evidence therefore requires corroboration by
# at least one independent market-activity feed; do not hard-code exceptional dates.
STOCK_MARKET_ACTIVITY_ANCHOR_DATASETS = (
    "TaiwanStockPrice",
    "TaiwanStockPriceAdj",
    "TaiwanStockTotalReturnIndex",
)

# These feeds are trading-session-scoped but legitimately sparse: no row on a market
# session is not, by itself, evidence of a provider/local gap.
SPARSE_STOCK_SESSION_DATASETS = frozenset({
    "TaiwanStockSecuritiesLending",
    "TaiwanStockDayTradingBorrowingFeeRate",
})


@dataclass(frozen=True)
class DateSemanticResult:
    status: str
    mode: str
    observed_date_count: int
    expected_date_count: int
    first_observed_date: str | None
    last_observed_date: str | None
    compared_start_date: str | None = None
    compared_end_date: str | None = None
    missing_dates: tuple[str, ...] = ()
    unexpected_dates: tuple[str, ...] = ()
    unverified_dates: tuple[str, ...] = ()
    reason: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "mode": self.mode,
            "observed_date_count": int(self.observed_date_count),
            "expected_date_count": int(self.expected_date_count),
            "first_observed_date": self.first_observed_date,
            "last_observed_date": self.last_observed_date,
            "compared_start_date": self.compared_start_date,
            "compared_end_date": self.compared_end_date,
            "missing_dates": list(self.missing_dates),
            "unexpected_dates": list(self.unexpected_dates),
            "unverified_dates": list(self.unverified_dates),
            "reason": self.reason,
        }


def resolve_date_semantic_mode(spec: MarketDatasetSpec) -> str:
    if spec.dataset == AUTHORITATIVE_TRADING_CALENDAR_DATASET:
        return DATE_SEMANTIC_AUTHORITATIVE_CALENDAR
    freshness = build_market_data_freshness_contract(spec)
    if freshness.cadence == CADENCE_TRADING_DAILY:
        # TaiwanStockTradingDate is a stock-market schedule/calendar candidate.
        # Futures/options use different session semantics, so do not project the
        # stock-market calendar onto derivative feeds.
        if spec.category == "derivative_context":
            return DATE_SEMANTIC_UNVERIFIED
        if spec.dataset in SPARSE_STOCK_SESSION_DATASETS:
            return DATE_SEMANTIC_TRADING_CALENDAR_SPARSE
        return DATE_SEMANTIC_TRADING_CALENDAR_DENSE
    if freshness.cadence in {CADENCE_CURRENT_VINTAGE, CADENCE_EVENT_DRIVEN}:
        return DATE_SEMANTIC_NOT_APPLICABLE
    if freshness.cadence in {CADENCE_CALENDAR_DAILY, CADENCE_PERIODIC}:
        return DATE_SEMANTIC_UNVERIFIED
    raise ValueError(f"{spec.dataset} integrity audit 不支援 freshness cadence={freshness.cadence!r}")


def evaluate_date_semantics(
    *,
    mode: str,
    observed_dates: Iterable[str],
    trading_calendar_dates: Iterable[str] = (),
    invalid_date_rows: int = 0,
    missing_date_column: bool = False,
) -> DateSemanticResult:
    observed = tuple(sorted({str(value) for value in observed_dates if str(value)}))
    first_observed = observed[0] if observed else None
    last_observed = observed[-1] if observed else None

    if mode == DATE_SEMANTIC_NOT_APPLICABLE:
        return DateSemanticResult(
            status=DATE_SEMANTIC_NOT_APPLICABLE_STATUS,
            mode=mode,
            observed_date_count=len(observed),
            expected_date_count=0,
            first_observed_date=first_observed,
            last_observed_date=last_observed,
            reason="event_or_static_dataset_has_no_dense_date_presence_contract",
        )
    if mode == DATE_SEMANTIC_UNVERIFIED:
        return DateSemanticResult(
            status=DATE_SEMANTIC_UNVERIFIED_STATUS,
            mode=mode,
            observed_date_count=len(observed),
            expected_date_count=0,
            first_observed_date=first_observed,
            last_observed_date=last_observed,
            reason="no_authoritative_dataset_specific_date_calendar",
        )
    if missing_date_column:
        return DateSemanticResult(
            status=DATE_SEMANTIC_FAIL,
            mode=mode,
            observed_date_count=len(observed),
            expected_date_count=0,
            first_observed_date=first_observed,
            last_observed_date=last_observed,
            reason="nonempty_artifact_missing_date_column",
        )
    if int(invalid_date_rows) > 0:
        return DateSemanticResult(
            status=DATE_SEMANTIC_FAIL,
            mode=mode,
            observed_date_count=len(observed),
            expected_date_count=0,
            first_observed_date=first_observed,
            last_observed_date=last_observed,
            reason=f"invalid_date_rows={int(invalid_date_rows)}",
        )
    if not observed:
        return DateSemanticResult(
            status=DATE_SEMANTIC_ATTENTION,
            mode=mode,
            observed_date_count=0,
            expected_date_count=0,
            first_observed_date=None,
            last_observed_date=None,
            reason="no_observed_dates_in_local_provider_evidence",
        )
    if mode == DATE_SEMANTIC_AUTHORITATIVE_CALENDAR:
        return DateSemanticResult(
            status=DATE_SEMANTIC_PASS,
            mode=mode,
            observed_date_count=len(observed),
            expected_date_count=len(observed),
            first_observed_date=first_observed,
            last_observed_date=last_observed,
            compared_start_date=first_observed,
            compared_end_date=last_observed,
            reason="authoritative_calendar_source",
        )
    if mode not in {DATE_SEMANTIC_TRADING_CALENDAR_DENSE, DATE_SEMANTIC_TRADING_CALENDAR_SPARSE}:
        raise ValueError(f"未支援的 Market Data date semantic mode: {mode}")

    calendar = tuple(sorted({str(value) for value in trading_calendar_dates if str(value)}))
    if not calendar:
        return DateSemanticResult(
            status=DATE_SEMANTIC_UNVERIFIED_STATUS,
            mode=mode,
            observed_date_count=len(observed),
            expected_date_count=0,
            first_observed_date=first_observed,
            last_observed_date=last_observed,
            reason="authoritative_stock_trading_calendar_unavailable",
        )

    compare_start = max(first_observed, calendar[0])
    compare_end = min(last_observed, calendar[-1])
    if compare_start > compare_end:
        return DateSemanticResult(
            status=DATE_SEMANTIC_UNVERIFIED_STATUS,
            mode=mode,
            observed_date_count=len(observed),
            expected_date_count=0,
            first_observed_date=first_observed,
            last_observed_date=last_observed,
            reason=(
                "authoritative_stock_trading_calendar_has_no_overlap: "
                f"calendar={calendar[0]}~{calendar[-1]} observed={first_observed}~{last_observed}"
            ),
        )

    observed_set = set(observed)
    expected = tuple(value for value in calendar if compare_start <= value <= compare_end)
    expected_set = set(expected)
    observed_in_scope = {value for value in observed_set if compare_start <= value <= compare_end}
    missing_candidates = (
        tuple(sorted(expected_set - observed_in_scope))
        if mode == DATE_SEMANTIC_TRADING_CALENDAR_DENSE
        else ()
    )
    # Historical Taiwan stock sessions could occur on Saturdays before the 2019
    # policy change, but a market-session anchor does not prove that every
    # provider dataset promised a row on those weekend sessions.  Treat absence
    # there as an evidence limit rather than a provider gap.  Observed weekend
    # rows remain valid evidence and are never discarded.
    weekend_missing = tuple(
        value
        for value in missing_candidates
        if date.fromisoformat(value).weekday() >= 5
    )
    missing = tuple(value for value in missing_candidates if value not in set(weekend_missing))
    unexpected = tuple(sorted(observed_in_scope - expected_set))
    full_span_covered = first_observed >= calendar[0] and last_observed <= calendar[-1]

    if missing or unexpected:
        status = DATE_SEMANTIC_ATTENTION
        reason = (
            "provider_observed_gap_or_noncorroborated_stock_session"
            if mode == DATE_SEMANTIC_TRADING_CALENDAR_DENSE
            else "provider_observed_noncorroborated_stock_session"
        )
    elif weekend_missing:
        status = DATE_SEMANTIC_PARTIAL
        reason = "historical_weekend_sessions_have_no_authoritative_provider_row_presence_guarantee"
    elif not full_span_covered:
        status = DATE_SEMANTIC_PARTIAL
        reason = (
            "corroborated_stock_session_evidence_only_partially_covers_observed_span: "
            f"sessions={calendar[0]}~{calendar[-1]} observed={first_observed}~{last_observed}"
        )
    else:
        status = DATE_SEMANTIC_PASS
        reason = (
            "complete_corroborated_stock_session_presence_within_observed_span"
            if mode == DATE_SEMANTIC_TRADING_CALENDAR_DENSE
            else "observed_rows_are_within_corroborated_stock_sessions"
        )

    return DateSemanticResult(
        status=status,
        mode=mode,
        observed_date_count=len(observed),
        expected_date_count=len(expected),
        first_observed_date=first_observed,
        last_observed_date=last_observed,
        compared_start_date=compare_start,
        compared_end_date=compare_end,
        missing_dates=missing,
        unexpected_dates=unexpected,
        unverified_dates=weekend_missing,
        reason=reason,
    )


def validate_market_data_integrity_contract() -> dict[str, int]:
    included = tuple(get_market_dataset_specs(included_only=True))
    modes = [resolve_date_semantic_mode(spec) for spec in included]
    included_ids = {spec.dataset for spec in included}
    missing_anchors = sorted(set(STOCK_MARKET_ACTIVITY_ANCHOR_DATASETS) - included_ids)
    if missing_anchors:
        raise ValueError(f"Market Data integrity activity anchors 不在 included registry: {missing_anchors}")
    return {
        "dataset_count": len(included),
        "dense_trading_calendar_count": sum(mode == DATE_SEMANTIC_TRADING_CALENDAR_DENSE for mode in modes),
        "sparse_trading_calendar_count": sum(mode == DATE_SEMANTIC_TRADING_CALENDAR_SPARSE for mode in modes),
        "not_applicable_count": sum(mode == DATE_SEMANTIC_NOT_APPLICABLE for mode in modes),
        "unverified_count": sum(mode == DATE_SEMANTIC_UNVERIFIED for mode in modes),
        "authoritative_calendar_count": sum(mode == DATE_SEMANTIC_AUTHORITATIVE_CALENDAR for mode in modes),
        "activity_anchor_count": len(STOCK_MARKET_ACTIVITY_ANCHOR_DATASETS),
    }


__all__ = [
    "DATE_SEMANTIC_AUTHORITATIVE_CALENDAR",
    "DATE_SEMANTIC_TRADING_CALENDAR_DENSE",
    "DATE_SEMANTIC_TRADING_CALENDAR_SPARSE",
    "DATE_SEMANTIC_NOT_APPLICABLE",
    "DATE_SEMANTIC_UNVERIFIED",
    "DATE_SEMANTIC_PASS",
    "DATE_SEMANTIC_ATTENTION",
    "DATE_SEMANTIC_PARTIAL",
    "DATE_SEMANTIC_FAIL",
    "DATE_SEMANTIC_NOT_APPLICABLE_STATUS",
    "DATE_SEMANTIC_UNVERIFIED_STATUS",
    "AUTHORITATIVE_TRADING_CALENDAR_DATASET",
    "STOCK_MARKET_ACTIVITY_ANCHOR_DATASETS",
    "SPARSE_STOCK_SESSION_DATASETS",
    "DateSemanticResult",
    "resolve_date_semantic_mode",
    "evaluate_date_semantics",
    "validate_market_data_integrity_contract",
]
