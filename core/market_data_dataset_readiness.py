"""Canonical dataset-level readiness contract for Trading Market Data V2.

Dataset usability and provider-probe scheduling are deliberately separate.
Target-date feeds (for example Price / MarketValue) must prove current-target
freshness before they are READY.  Roll-forward feeds (periodic, event-driven,
current-vintage and latest-available) may remain usable across later Trading
targets once current structural/request-scope validation has been established;
the Due planner independently schedules their next provider observation.
"""
from __future__ import annotations

from typing import Mapping

from core.market_data_freshness_contract import (
    EXPECTED_DATE_TRADING_TARGET,
    FRESHNESS_STATUS_READY,
    MarketDataFreshnessContract,
)

MARKET_DATA_DATASET_VALIDATION_CONTRACT_VERSION = 3
VALIDATION_STATUS_READY = "READY"
VALIDATION_STATUS_NO_ROW_VALID = "NO_ROW_VALID"
VALID_DATASET_VALIDATION_STATUSES = frozenset(
    {VALIDATION_STATUS_READY, VALIDATION_STATUS_NO_ROW_VALID}
)


def has_current_market_data_dataset_validation(row: Mapping[str, object] | None) -> bool:
    item = row if isinstance(row, Mapping) else {}
    try:
        version = int(item.get("validation_contract_version", -1))
    except (TypeError, ValueError):
        return False
    return bool(
        version == MARKET_DATA_DATASET_VALIDATION_CONTRACT_VERSION
        and str(item.get("schema_status") or "") in VALID_DATASET_VALIDATION_STATUSES
        and str(item.get("coverage_status") or "") in VALID_DATASET_VALIDATION_STATUSES
    )


def market_data_contract_requires_target_freshness(
    contract: MarketDataFreshnessContract,
) -> bool:
    """Whether usability requires evidence for the exact Trading target.

    Every other expected-date mode is a roll-forward contract: the currently
    validated provider state remains usable while a separate schedule decides
    when the provider should be checked again.
    """

    return contract.expected_date_mode == EXPECTED_DATE_TRADING_TARGET


def is_market_data_dataset_ready(
    row: Mapping[str, object] | None,
    *,
    target_date: str,
    contract: MarketDataFreshnessContract | None = None,
) -> bool:
    """Return whether one dataset is legally usable for ``target_date``.

    ``contract=None`` preserves the legacy strict target-horizon interpretation
    for callers that do not own a dataset identity.  Canonical Market Data V2
    runtime callers must pass the dataset freshness contract so roll-forward
    feeds are not falsely invalidated merely because their content date or last
    provider observation predates the Trading target.
    """

    item = row if isinstance(row, Mapping) else {}
    ready_target = str(item.get("last_ready_target_date") or "").strip()
    base_ready = bool(
        has_current_market_data_dataset_validation(item)
        and str(item.get("status") or "") == FRESHNESS_STATUS_READY
        and ready_target
    )
    if not base_ready:
        return False
    if contract is not None and not market_data_contract_requires_target_freshness(contract):
        return True
    return ready_target >= str(target_date)


__all__ = [
    "MARKET_DATA_DATASET_VALIDATION_CONTRACT_VERSION",
    "VALIDATION_STATUS_READY",
    "VALIDATION_STATUS_NO_ROW_VALID",
    "VALID_DATASET_VALIDATION_STATUSES",
    "has_current_market_data_dataset_validation",
    "market_data_contract_requires_target_freshness",
    "is_market_data_dataset_ready",
]
