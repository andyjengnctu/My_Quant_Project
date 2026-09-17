"""Workbench-wide state-change domains and view dependency contract.

Mutation producers publish semantic domains after a canonical commit.  The shell
owns fan-out to mounted views; panels never call one another directly.  An unloaded
view needs no notification because its first read is canonical.
"""
from __future__ import annotations

from typing import Iterable

STATE_ACCOUNT = "ACCOUNT"
STATE_POSITIONS = "POSITIONS"
STATE_TRADES = "TRADES"
STATE_PENDING_ENTRIES = "PENDING_ENTRIES"
STATE_RESOURCE_RESERVATION = "RESOURCE_RESERVATION"
STATE_SCANNER_ELIGIBILITY = "SCANNER_ELIGIBILITY"
STATE_ORDERS = "ORDERS"
STATE_PROTECTION = "PROTECTION"
STATE_INDICATOR_EXIT = "INDICATOR_EXIT"
STATE_MARKET_DATA = "MARKET_DATA"
STATE_PARAMS = "PARAMS"
STATE_SCANNER = "SCANNER"

ACCOUNT_MUTATION_DOMAINS = frozenset({
    STATE_ACCOUNT,
    STATE_POSITIONS,
    STATE_TRADES,
    STATE_RESOURCE_RESERVATION,
    STATE_SCANNER_ELIGIBILITY,
})
PENDING_MUTATION_DOMAINS = frozenset({
    STATE_PENDING_ENTRIES,
    STATE_RESOURCE_RESERVATION,
    STATE_SCANNER_ELIGIBILITY,
})
PENDING_FILL_MUTATION_DOMAINS = frozenset(set(ACCOUNT_MUTATION_DOMAINS) | set(PENDING_MUTATION_DOMAINS))

PANEL_DEPENDENCIES = {
    "trading_account": frozenset({
        STATE_ACCOUNT,
        STATE_POSITIONS,
        STATE_TRADES,
        STATE_PENDING_ENTRIES,
        STATE_RESOURCE_RESERVATION,
        STATE_SCANNER_ELIGIBILITY,
        STATE_ORDERS,
        STATE_PROTECTION,
        STATE_INDICATOR_EXIT,
        STATE_MARKET_DATA,
        STATE_PARAMS,
        STATE_SCANNER,
    }),
    "accounting_center": frozenset({STATE_ACCOUNT, STATE_POSITIONS, STATE_TRADES, STATE_MARKET_DATA}),
    "single_stock_backtest_inspector": frozenset({
        STATE_POSITIONS,
        STATE_PENDING_ENTRIES,
        STATE_SCANNER_ELIGIBILITY,
        STATE_SCANNER,
        STATE_MARKET_DATA,
        STATE_PARAMS,
    }),
}


def normalize_state_domains(domains: Iterable[str] | str | None) -> frozenset[str]:
    if domains is None:
        return frozenset()
    if isinstance(domains, str):
        values = [domains]
    else:
        values = list(domains)
    return frozenset(str(value).strip().upper() for value in values if str(value).strip())


def panel_depends_on_state_domains(panel_id: str, domains: Iterable[str] | str | None) -> bool:
    normalized = normalize_state_domains(domains)
    if not normalized:
        return False
    dependencies = PANEL_DEPENDENCIES.get(str(panel_id), frozenset())
    return bool(normalized & dependencies)


__all__ = [
    "STATE_ACCOUNT",
    "STATE_POSITIONS",
    "STATE_TRADES",
    "STATE_PENDING_ENTRIES",
    "STATE_RESOURCE_RESERVATION",
    "STATE_SCANNER_ELIGIBILITY",
    "STATE_ORDERS",
    "STATE_PROTECTION",
    "STATE_INDICATOR_EXIT",
    "STATE_MARKET_DATA",
    "STATE_PARAMS",
    "STATE_SCANNER",
    "ACCOUNT_MUTATION_DOMAINS",
    "PENDING_MUTATION_DOMAINS",
    "PENDING_FILL_MUTATION_DOMAINS",
    "PANEL_DEPENDENCIES",
    "normalize_state_domains",
    "panel_depends_on_state_domains",
]
