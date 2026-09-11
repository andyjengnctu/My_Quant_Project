"""Canonical semantic-integrity contract for Market Data V2 local audits.

This contract is deliberately outside ``MarketDatasetSpec`` so strengthening a
local-only integrity audit cannot mutate the immutable Provider Snapshot /
bootstrap registry fingerprint.  Dataset natural keys remain owned by
``core.market_data_dataset_registry``; this module only declares which datasets
have enough authoritative cadence evidence for a date-semantic audit.
"""
from __future__ import annotations

from dataclasses import dataclass
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
DATE_SEMANTIC_TRADING_CALENDAR_DENSE = "trading_calendar_dense"
DATE_SEMANTIC_NOT_APPLICABLE = "not_applicable"
DATE_SEMANTIC_UNVERIFIED = "unverified"

DATE_SEMANTIC_PASS = "PASS"
DATE_SEMANTIC_FAIL = "FAIL"
DATE_SEMANTIC_NOT_APPLICABLE_STATUS = "NOT_APPLICABLE"
DATE_SEMANTIC_UNVERIFIED_STATUS = "UNVERIFIED"

AUTHORITATIVE_TRADING_CALENDAR_DATASET = "TaiwanStockTradingDate"

# Date cadence ownership remains in ``market_data_freshness_contract``.  This
# audit contract consumes that existing cadence instead of maintaining another
# dataset identity list.



@dataclass(frozen=True)
class DateSemanticResult:
    status: str
    mode: str
    observed_date_count: int
    expected_date_count: int
    first_observed_date: str | None
    last_observed_date: str | None
    missing_dates: tuple[str, ...] = ()
    unexpected_dates: tuple[str, ...] = ()
    reason: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "mode": self.mode,
            "observed_date_count": int(self.observed_date_count),
            "expected_date_count": int(self.expected_date_count),
            "first_observed_date": self.first_observed_date,
            "last_observed_date": self.last_observed_date,
            "missing_dates": list(self.missing_dates),
            "unexpected_dates": list(self.unexpected_dates),
            "reason": self.reason,
        }


def resolve_date_semantic_mode(spec: MarketDatasetSpec) -> str:
    if spec.dataset == AUTHORITATIVE_TRADING_CALENDAR_DATASET:
        return DATE_SEMANTIC_AUTHORITATIVE_CALENDAR
    freshness = build_market_data_freshness_contract(spec)
    if freshness.cadence == CADENCE_TRADING_DAILY:
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
            reason="no_authoritative_dataset_specific_date_cadence_contract",
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
            status=DATE_SEMANTIC_FAIL,
            mode=mode,
            observed_date_count=0,
            expected_date_count=0,
            first_observed_date=None,
            last_observed_date=None,
            reason="no_observed_dates",
        )
    if mode == DATE_SEMANTIC_AUTHORITATIVE_CALENDAR:
        return DateSemanticResult(
            status=DATE_SEMANTIC_PASS,
            mode=mode,
            observed_date_count=len(observed),
            expected_date_count=len(observed),
            first_observed_date=first_observed,
            last_observed_date=last_observed,
            reason="authoritative_calendar_source",
        )
    if mode != DATE_SEMANTIC_TRADING_CALENDAR_DENSE:
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
            reason="authoritative_trading_calendar_unavailable",
        )
    if first_observed < calendar[0] or last_observed > calendar[-1]:
        return DateSemanticResult(
            status=DATE_SEMANTIC_UNVERIFIED_STATUS,
            mode=mode,
            observed_date_count=len(observed),
            expected_date_count=0,
            first_observed_date=first_observed,
            last_observed_date=last_observed,
            reason=(
                "authoritative_trading_calendar_does_not_cover_observed_span: "
                f"calendar={calendar[0]}~{calendar[-1]} observed={first_observed}~{last_observed}"
            ),
        )

    observed_set = set(observed)
    expected = tuple(value for value in calendar if first_observed <= value <= last_observed)
    expected_set = set(expected)
    missing = tuple(sorted(expected_set - observed_set))
    unexpected = tuple(sorted(observed_set - expected_set))
    status = DATE_SEMANTIC_PASS if not missing and not unexpected else DATE_SEMANTIC_FAIL
    reason = "complete_trading_calendar_presence_within_observed_span" if status == DATE_SEMANTIC_PASS else "date_presence_gap_or_nontrading_date"
    return DateSemanticResult(
        status=status,
        mode=mode,
        observed_date_count=len(observed),
        expected_date_count=len(expected),
        first_observed_date=first_observed,
        last_observed_date=last_observed,
        missing_dates=missing,
        unexpected_dates=unexpected,
        reason=reason,
    )


def validate_market_data_integrity_contract() -> dict[str, int]:
    included = tuple(get_market_dataset_specs(included_only=True))
    modes = [resolve_date_semantic_mode(spec) for spec in included]
    return {
        "dataset_count": len(included),
        "dense_trading_calendar_count": sum(mode == DATE_SEMANTIC_TRADING_CALENDAR_DENSE for mode in modes),
        "not_applicable_count": sum(mode == DATE_SEMANTIC_NOT_APPLICABLE for mode in modes),
        "unverified_count": sum(mode == DATE_SEMANTIC_UNVERIFIED for mode in modes),
        "authoritative_calendar_count": sum(mode == DATE_SEMANTIC_AUTHORITATIVE_CALENDAR for mode in modes),
    }



__all__ = [
    "DATE_SEMANTIC_AUTHORITATIVE_CALENDAR",
    "DATE_SEMANTIC_TRADING_CALENDAR_DENSE",
    "DATE_SEMANTIC_NOT_APPLICABLE",
    "DATE_SEMANTIC_UNVERIFIED",
    "DATE_SEMANTIC_PASS",
    "DATE_SEMANTIC_FAIL",
    "DATE_SEMANTIC_NOT_APPLICABLE_STATUS",
    "DATE_SEMANTIC_UNVERIFIED_STATUS",
    "AUTHORITATIVE_TRADING_CALENDAR_DATASET",
    "DateSemanticResult",
    "resolve_date_semantic_mode",
    "evaluate_date_semantics",
    "validate_market_data_integrity_contract",
]
