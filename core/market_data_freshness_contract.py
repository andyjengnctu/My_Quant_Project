"""Canonical dataset-level publication/freshness contract for Market Data V2.

This module is pure policy resolution: no provider calls, no file inspection and
no scheduler side effects.  It gives every included Market Data V2 dataset one
operational freshness meaning so later state/scheduler/UI layers do not invent
independent notions of READY, due dates or publication timing.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Iterable, Mapping

from config.market_data import MARKET_DATA_V2_PUBLICATION_POLICY
from core.market_data_dataset_registry import (
    DAILY_EVENT_REPAIR,
    DAILY_INCREMENTAL,
    DAILY_PERIODIC_REPAIR,
    DAILY_RECENT_REPAIR,
    DAILY_STATIC_REFRESH,
    MarketDatasetSpec,
    get_market_dataset_specs,
    resolve_market_dataset_row_identity,
)

FRESHNESS_STATUS_READY = "READY"
FRESHNESS_STATUS_DUE = "DUE"
FRESHNESS_STATUS_WAIT_PUBLISH = "WAIT_PUBLISH"
FRESHNESS_STATUS_WAIT_QUOTA = "WAIT_QUOTA"
FRESHNESS_STATUS_STALE = "STALE"
FRESHNESS_STATUS_ERROR = "ERROR"
FRESHNESS_STATUS_BLOCKED = "BLOCKED"
FRESHNESS_STATUS_NOT_APPLICABLE = "NOT_APPLICABLE"

FRESHNESS_STATUSES = (
    FRESHNESS_STATUS_READY,
    FRESHNESS_STATUS_DUE,
    FRESHNESS_STATUS_WAIT_PUBLISH,
    FRESHNESS_STATUS_WAIT_QUOTA,
    FRESHNESS_STATUS_STALE,
    FRESHNESS_STATUS_ERROR,
    FRESHNESS_STATUS_BLOCKED,
    FRESHNESS_STATUS_NOT_APPLICABLE,
)

CADENCE_CURRENT_VINTAGE = "current_vintage"
CADENCE_TRADING_DAILY = "trading_daily"
CADENCE_CALENDAR_DAILY = "calendar_daily"
CADENCE_EVENT_DRIVEN = "event_driven"
CADENCE_PERIODIC = "periodic"

EXPECTED_DATE_NONE = "none"
EXPECTED_DATE_TRADING_TARGET = "trading_target_date"
EXPECTED_DATE_LATEST_AVAILABLE = "latest_available_date"
EXPECTED_DATE_PERIOD_DUE = "period_due"

ROW_EXPECTATION_REQUIRED = "required"
ROW_EXPECTATION_OPTIONAL = "optional"

COMPLETENESS_CURRENT_VINTAGE = "current_vintage_schema"
COMPLETENESS_FULL_MARKET_EXACT_DATE = "full_market_exact_date"
COMPLETENESS_RANGE_OR_AGGREGATE = "range_or_aggregate"
COMPLETENESS_EVENT_NO_ROW_VALID = "event_window_no_row_valid"
COMPLETENESS_PERIODIC_WINDOW = "periodic_window"

_MACRO_CATEGORIES = {"macro_context"}


@dataclass(frozen=True)
class MarketDataFreshnessContract:
    dataset: str
    category: str
    cadence: str
    expected_date_mode: str
    row_expectation: str
    completeness_mode: str
    schema_validation_required: bool
    publication_first_check_time: str
    publication_day_offset: int
    publication_schedule_source: str
    publication_schedule_verified: bool
    primary_key_hint: tuple[str, ...]


def _parse_hhmm(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    try:
        hour_text, minute_text = text.split(":", 1)
        parsed = time(hour=int(hour_text), minute=int(minute_text))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} 必須是 HH:MM: {text!r}") from exc
    return parsed.strftime("%H:%M")


def _publication_schedule(dataset: str) -> tuple[str, int, str, bool]:
    raw = MARKET_DATA_V2_PUBLICATION_POLICY
    if not isinstance(raw, Mapping):
        raise ValueError("MARKET_DATA_V2_PUBLICATION_POLICY 必須是 mapping")
    fallback = raw.get("conservative_fallback")
    overrides = raw.get("dataset_overrides")
    if not isinstance(fallback, Mapping) or not isinstance(overrides, Mapping):
        raise ValueError("Market Data publication policy 缺少 fallback/overrides")
    selected = overrides.get(dataset, fallback)
    if not isinstance(selected, Mapping):
        raise ValueError(f"{dataset} publication policy 必須是 mapping")
    first_check = _parse_hhmm(selected.get("first_check_time"), field=f"{dataset}.first_check_time")
    day_offset = int(selected.get("day_offset") or 0)
    if day_offset < 0 or day_offset > 7:
        raise ValueError(f"{dataset}.day_offset 必須落在 0..7")
    source = str(selected.get("source") or "").strip()
    if not source:
        raise ValueError(f"{dataset}.publication source 不可空白")
    verified = dataset in overrides and source.startswith("provider_documentation")
    return first_check, day_offset, source, verified


def _derive_semantics(spec: MarketDatasetSpec) -> tuple[str, str, str, str]:
    if spec.daily_mode == DAILY_STATIC_REFRESH:
        return (
            CADENCE_CURRENT_VINTAGE,
            EXPECTED_DATE_NONE,
            ROW_EXPECTATION_REQUIRED,
            COMPLETENESS_CURRENT_VINTAGE,
        )
    if spec.daily_mode == DAILY_EVENT_REPAIR:
        return (
            CADENCE_EVENT_DRIVEN,
            EXPECTED_DATE_NONE,
            ROW_EXPECTATION_OPTIONAL,
            COMPLETENESS_EVENT_NO_ROW_VALID,
        )
    if spec.daily_mode == DAILY_PERIODIC_REPAIR:
        return (
            CADENCE_PERIODIC,
            EXPECTED_DATE_PERIOD_DUE,
            ROW_EXPECTATION_OPTIONAL,
            COMPLETENESS_PERIODIC_WINDOW,
        )
    if spec.daily_mode in {DAILY_INCREMENTAL, DAILY_RECENT_REPAIR}:
        cadence = CADENCE_CALENDAR_DAILY if spec.category in _MACRO_CATEGORIES else CADENCE_TRADING_DAILY
        expected = EXPECTED_DATE_LATEST_AVAILABLE if cadence == CADENCE_CALENDAR_DAILY else EXPECTED_DATE_TRADING_TARGET
        completeness = (
            COMPLETENESS_FULL_MARKET_EXACT_DATE
            if spec.full_market_exact_date_expected
            else COMPLETENESS_RANGE_OR_AGGREGATE
        )
        return cadence, expected, ROW_EXPECTATION_REQUIRED, completeness
    raise ValueError(f"{spec.dataset} freshness contract 不支援 daily_mode={spec.daily_mode!r}")


def build_market_data_freshness_contract(spec: MarketDatasetSpec) -> MarketDataFreshnessContract:
    if not spec.included:
        raise ValueError(f"excluded dataset 不建立 Trading freshness contract: {spec.dataset}")
    cadence, expected, row_expectation, completeness = _derive_semantics(spec)
    first_check, day_offset, source, verified = _publication_schedule(spec.dataset)
    return MarketDataFreshnessContract(
        dataset=spec.dataset,
        category=spec.category,
        cadence=cadence,
        expected_date_mode=expected,
        row_expectation=row_expectation,
        completeness_mode=completeness,
        schema_validation_required=True,
        publication_first_check_time=first_check,
        publication_day_offset=day_offset,
        publication_schedule_source=source,
        publication_schedule_verified=verified,
        # Operational consumers need the canonical row identity, which may be
        # stricter than the immutable provider/bootstrap merge hint.
        primary_key_hint=tuple(resolve_market_dataset_row_identity(spec)),
    )


def get_market_data_freshness_contracts(
    *, specs: Iterable[MarketDatasetSpec] | None = None,
) -> tuple[MarketDataFreshnessContract, ...]:
    source = tuple(specs) if specs is not None else get_market_dataset_specs(included_only=True)
    included = tuple(spec for spec in source if spec.included)
    contracts = tuple(build_market_data_freshness_contract(spec) for spec in included)
    validate_market_data_freshness_contracts(specs=included, contracts=contracts)
    return contracts


def get_market_data_freshness_contract(dataset: str) -> MarketDataFreshnessContract:
    key = str(dataset or "").strip()
    matches = [item for item in get_market_data_freshness_contracts() if item.dataset == key]
    if len(matches) != 1:
        raise ValueError(f"Market Data freshness dataset identity 無法唯一解析: {key!r}")
    return matches[0]


def validate_market_data_freshness_contracts(
    *,
    specs: Iterable[MarketDatasetSpec] | None = None,
    contracts: Iterable[MarketDataFreshnessContract] | None = None,
) -> dict[str, int]:
    included = tuple(
        spec for spec in (tuple(specs) if specs is not None else get_market_dataset_specs(included_only=True))
        if spec.included
    )
    values = tuple(contracts) if contracts is not None else tuple(build_market_data_freshness_contract(spec) for spec in included)
    spec_ids = [spec.dataset for spec in included]
    contract_ids = [item.dataset for item in values]
    if len(contract_ids) != len(set(contract_ids)):
        raise ValueError("Market Data freshness contract dataset identity 重複")
    if set(contract_ids) != set(spec_ids):
        missing = sorted(set(spec_ids) - set(contract_ids))
        extra = sorted(set(contract_ids) - set(spec_ids))
        raise ValueError(f"Market Data freshness contract coverage drift: missing={missing}, extra={extra}")
    for item in values:
        if item.cadence not in {
            CADENCE_CURRENT_VINTAGE,
            CADENCE_TRADING_DAILY,
            CADENCE_CALENDAR_DAILY,
            CADENCE_EVENT_DRIVEN,
            CADENCE_PERIODIC,
        }:
            raise ValueError(f"{item.dataset} freshness cadence 未支援: {item.cadence}")
        if item.row_expectation not in {ROW_EXPECTATION_REQUIRED, ROW_EXPECTATION_OPTIONAL}:
            raise ValueError(f"{item.dataset} row_expectation 未支援: {item.row_expectation}")
        _parse_hhmm(item.publication_first_check_time, field=f"{item.dataset}.publication_first_check_time")
    return {
        "included_dataset_count": len(included),
        "contract_count": len(values),
        "verified_schedule_count": sum(item.publication_schedule_verified for item in values),
        "fallback_schedule_count": sum(not item.publication_schedule_verified for item in values),
    }


def build_market_data_freshness_contract_summary() -> dict[str, object]:
    contracts = get_market_data_freshness_contracts()
    stats = validate_market_data_freshness_contracts(contracts=contracts)
    return {
        **stats,
        "status_values": FRESHNESS_STATUSES,
        "cadence_counts": {
            cadence: sum(item.cadence == cadence for item in contracts)
            for cadence in (
                CADENCE_CURRENT_VINTAGE,
                CADENCE_TRADING_DAILY,
                CADENCE_CALENDAR_DAILY,
                CADENCE_EVENT_DRIVEN,
                CADENCE_PERIODIC,
            )
        },
    }


__all__ = [
    "FRESHNESS_STATUS_READY",
    "FRESHNESS_STATUS_DUE",
    "FRESHNESS_STATUS_WAIT_PUBLISH",
    "FRESHNESS_STATUS_WAIT_QUOTA",
    "FRESHNESS_STATUS_STALE",
    "FRESHNESS_STATUS_ERROR",
    "FRESHNESS_STATUS_BLOCKED",
    "FRESHNESS_STATUS_NOT_APPLICABLE",
    "FRESHNESS_STATUSES",
    "CADENCE_CURRENT_VINTAGE",
    "CADENCE_TRADING_DAILY",
    "CADENCE_CALENDAR_DAILY",
    "CADENCE_EVENT_DRIVEN",
    "CADENCE_PERIODIC",
    "EXPECTED_DATE_NONE",
    "EXPECTED_DATE_TRADING_TARGET",
    "EXPECTED_DATE_LATEST_AVAILABLE",
    "EXPECTED_DATE_PERIOD_DUE",
    "ROW_EXPECTATION_REQUIRED",
    "ROW_EXPECTATION_OPTIONAL",
    "COMPLETENESS_CURRENT_VINTAGE",
    "COMPLETENESS_FULL_MARKET_EXACT_DATE",
    "COMPLETENESS_RANGE_OR_AGGREGATE",
    "COMPLETENESS_EVENT_NO_ROW_VALID",
    "COMPLETENESS_PERIODIC_WINDOW",
    "MarketDataFreshnessContract",
    "build_market_data_freshness_contract",
    "get_market_data_freshness_contracts",
    "get_market_data_freshness_contract",
    "validate_market_data_freshness_contracts",
    "build_market_data_freshness_contract_summary",
]
