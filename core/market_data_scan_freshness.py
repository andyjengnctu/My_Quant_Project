"""Canonical Trading scan-freshness policy for Market Data V2 datasets.

Source cadence and scan usability are different concerns.  Publication timing
only schedules provider probes; this policy decides whether a Trading scan for
market date ``T`` requires dataset content dated exactly ``T`` or may consume
the latest successfully synchronized local provider version.
"""
from __future__ import annotations

from core.market_data_freshness_contract import (
    CADENCE_TRADING_DAILY,
    MarketDataFreshnessContract,
)

SCAN_FRESHNESS_EXACT_TARGET = "exact_target"
SCAN_FRESHNESS_LATEST_SYNCED = "latest_synced"
SCAN_FRESHNESS_MODES = (
    SCAN_FRESHNESS_EXACT_TARGET,
    SCAN_FRESHNESS_LATEST_SYNCED,
)

# Operational default: MarketValue is a daily provider feed but is commonly
# published too late to gate the same-day post-close scanner.  The setting is
# user-overridable in Workbench and therefore is not a special case in the
# readiness engine itself.
_DEFAULT_LATEST_SYNCED_TRADING_DAILY = frozenset({"TaiwanStockMarketValue"})


def default_market_data_scan_freshness_mode(
    contract: MarketDataFreshnessContract,
) -> str:
    """Return the default scan policy for one canonical freshness contract."""

    if contract.cadence != CADENCE_TRADING_DAILY:
        return SCAN_FRESHNESS_LATEST_SYNCED
    if contract.dataset in _DEFAULT_LATEST_SYNCED_TRADING_DAILY:
        return SCAN_FRESHNESS_LATEST_SYNCED
    return SCAN_FRESHNESS_EXACT_TARGET


def market_data_scan_freshness_is_user_configurable(
    contract: MarketDataFreshnessContract,
) -> bool:
    """Only Trading-daily feeds have a meaningful exact-target/latest choice."""

    return contract.cadence == CADENCE_TRADING_DAILY


def resolve_market_data_scan_freshness_mode(
    contract: MarketDataFreshnessContract,
    configured_mode: object = None,
) -> str:
    """Resolve one dataset policy while validating user-configurable overrides."""

    text = str(configured_mode or "").strip().lower()
    if not text:
        return default_market_data_scan_freshness_mode(contract)
    if text not in SCAN_FRESHNESS_MODES:
        raise ValueError(
            f"{contract.dataset} scan freshness mode 不合法: {configured_mode!r}"
        )
    if not market_data_scan_freshness_is_user_configurable(contract):
        # Non-daily cadence has no exact-date meaning for a Trading-day target.
        # Persisted legacy/manual values must not redefine that semantic.
        return SCAN_FRESHNESS_LATEST_SYNCED
    return text


def market_data_scan_requires_exact_target(
    contract: MarketDataFreshnessContract,
    configured_mode: object = None,
) -> bool:
    return (
        resolve_market_data_scan_freshness_mode(contract, configured_mode)
        == SCAN_FRESHNESS_EXACT_TARGET
    )


__all__ = [
    "SCAN_FRESHNESS_EXACT_TARGET",
    "SCAN_FRESHNESS_LATEST_SYNCED",
    "SCAN_FRESHNESS_MODES",
    "default_market_data_scan_freshness_mode",
    "market_data_scan_freshness_is_user_configurable",
    "resolve_market_data_scan_freshness_mode",
    "market_data_scan_requires_exact_target",
]
