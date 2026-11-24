"""Run-local mutable Portfolio state and frozen pre-market spending authority.

Persisted Trading account schemas and the canonical accounting primitives are
unchanged. A reservation freezes membership and spending before any daily sale
and exposes copied candidate records as read-only mappings so neither later
candidate diagnostics nor event owners can reprice an order. Nested runtime
contexts, parameter objects and live shadow positions keep
their canonical owners; this is deliberately not a deep copy of run state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from core.fee_rebate import create_fee_rebate_state


@dataclass
class PortfolioRunState:
    cash: int
    portfolio: dict = field(default_factory=dict)
    fee_rebate_state: dict = field(default_factory=create_fee_rebate_state)
    active_extended_signals: dict = field(default_factory=dict)
    active_extended_signals_by_member: dict = field(default_factory=dict)
    active_reentry_watchlist: dict = field(default_factory=dict)
    active_reentry_watchlists_by_member: dict = field(default_factory=dict)
    trade_history: list = field(default_factory=list)
    closed_trades_stats: list = field(default_factory=list)
    normal_trade_count: int = 0
    extended_trade_count: int = 0
    total_missed_buys: int = 0
    total_missed_sells: int = 0
    portfolio_entry_stats: dict = field(default_factory=lambda: {"filled_buy_count": 0})


@dataclass(frozen=True)
class PremarketReservation:
    available_cash: int
    sizing_equity: float
    action_candidates: tuple[Mapping[str, Any], ...]
    max_positions: int
    no_entry_capacity: bool

    @classmethod
    def capture(cls, *, available_cash, sizing_equity, action_candidates,
                max_positions, no_entry_capacity):
        return cls(available_cash, sizing_equity,
                   tuple(MappingProxyType(dict(candidate)) for candidate in action_candidates),
                   max_positions, no_entry_capacity)
