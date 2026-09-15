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

from core.market_data_freshness_contract import MarketDataFreshnessContract
from core.market_data_scan_freshness import market_data_scan_requires_exact_target

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
    row: Mapping[str, object] | None = None,
) -> bool:
    """Whether scan usability requires content for the exact Trading target.

    Dataset source cadence and provider publication timing do not define this
    decision.  Trading-daily feeds may be user-configured as ``latest_synced``
    (for example a slow-publishing input); non-daily feeds are inherently
    latest-synchronized because an exact Trading-day row has no valid meaning.
    """

    item = row if isinstance(row, Mapping) else {}
    return market_data_scan_requires_exact_target(
        contract,
        item.get("scan_freshness_mode"),
    )


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
        and ready_target
    )
    if not base_ready:
        return False
    if contract is not None and not market_data_contract_requires_target_freshness(contract, item):
        # ``latest_synced`` does not mean an old local version is reusable
        # forever.  ``last_ready_target_date`` advances only after that Scan
        # Target has completed a provider observation proving local == the
        # provider's latest available version at that observation time.
        return ready_target >= str(target_date)
    # ``last_ready_target_date`` may be advanced by a latest-synced policy even
    # when provider content has no row dated T.  Exact-target authorization must
    # therefore use its own evidence horizon.  Missing legacy field is safe to
    # fall back because pre-policy dataset state only advanced this horizon from
    # exact-target evidence for Trading-daily feeds.
    exact_raw = (
        item.get("last_exact_ready_target_date")
        if "last_exact_ready_target_date" in item
        else item.get("last_ready_target_date")
    )
    exact_ready_target = str(exact_raw or "").strip()
    return bool(exact_ready_target and exact_ready_target >= str(target_date))


__all__ = [
    "MARKET_DATA_DATASET_VALIDATION_CONTRACT_VERSION",
    "VALIDATION_STATUS_READY",
    "VALIDATION_STATUS_NO_ROW_VALID",
    "VALID_DATASET_VALIDATION_STATUSES",
    "has_current_market_data_dataset_validation",
    "market_data_contract_requires_target_freshness",
    "is_market_data_dataset_ready",
]
