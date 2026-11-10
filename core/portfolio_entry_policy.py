"""Canonical portfolio entry-source policy; never owns fills, Min exits or cash.

Breakout remains the implicit/default strategy.  A scored Daily strategy
consumes verified (ticker, information-date) score eligibility, while explicitly
unscored Daily optimization and no-DL baselines use all eligible adjacent
market stock-days.  Neither source reads future labels; the Portfolio Engine
executes both using the same position and accounting path.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Mapping, Iterator

import pandas as pd

from core.portfolio_fast_access import get_fast_pos

ENTRY_BREAKOUT = "breakout"
ENTRY_DAILY = "daily"
DAILY_ENTRY_CONTRACT = "daily_stockday_close_limit_next_exchange_session_v1"
DAILY_ATR_ENTRY_CONTRACT = "daily_stockday_atr_limit_next_exchange_session_v1"
DAILY_EXECUTION_LEGACY = "legacy_daily_close_v1"
DAILY_EXECUTION_ATR_LIMIT = "daily_atr_limit_v2"
DAILY_EXECUTION_FULL_FILTERS = "daily_atr_limit_common_filters_v2"
DAILY_EXECUTION_CLOSE_FILTERS = "daily_close_common_filters_v1"
_DAILY_EXECUTION_CONTRACTS = frozenset({
    DAILY_EXECUTION_LEGACY, DAILY_EXECUTION_ATR_LIMIT, DAILY_EXECUTION_FULL_FILTERS,
    DAILY_EXECUTION_CLOSE_FILTERS,
})


def resolve_daily_execution_contract(raw: str | None) -> str:
    contract = str(raw or DAILY_EXECUTION_LEGACY)
    if contract not in _DAILY_EXECUTION_CONTRACTS:
        raise ValueError(f"Unregistered Daily execution contract: {contract!r}")
    return contract


def daily_execution_uses_common_filters(contract: str | None) -> bool:
    return resolve_daily_execution_contract(contract) in {
        DAILY_EXECUTION_FULL_FILTERS, DAILY_EXECUTION_CLOSE_FILTERS,
    }


def daily_execution_uses_atr_limit(contract: str | None) -> bool:
    return resolve_daily_execution_contract(contract) in {
        DAILY_EXECUTION_ATR_LIMIT, DAILY_EXECUTION_FULL_FILTERS,
    }


def daily_execution_contract_from_params(params) -> str:
    """Consumption-time replay semantics travel with the published param artifact."""
    if not isinstance(params, Mapping):
        return DAILY_EXECUTION_LEGACY
    adaptation = params.get("breakout_quality_param_adaptation") or {}
    if not isinstance(adaptation, Mapping):
        raise ValueError("Invalid strategy param adaptation contract")
    return resolve_daily_execution_contract(adaptation.get("daily_execution_contract"))


class DailyEntrySource:
    """One replay-scoped, verified score-universe view of eligible decisions."""
    def __init__(self, score_keys):
        decisions = defaultdict(set)
        for ticker, info_day in score_keys:
            day = pd.Timestamp(info_day).normalize()
            decisions[day].add(str(ticker))
        if not decisions:
            raise ValueError("Daily Entry requires verified eligible stock-day score keys")
        self._decisions = {day: tuple(sorted(tickers)) for day, tickers in decisions.items()}
        self._previous_session = {}
        self._calendar = None
        self.excluded_missing_market_ticker = 0
        self.excluded_missing_adjacent_quote = 0

    def configure_calendar(self, sorted_dates):
        dates = tuple(pd.Timestamp(d).normalize() for d in sorted_dates)
        if len(dates) < 2 or dates != tuple(sorted(set(dates))):
            raise ValueError("Daily Entry exchange-session calendar invalid")
        self._previous_session = {dates[i]: dates[i - 1] for i in range(1, len(dates))}
        self._calendar = dates

    def entries_for(self, today, all_dfs_fast):
        """The same market-loader universe and exchange calendar as Breakout.

        Unavailable securities or non-adjacent quotes are not synthetic trades.
        They remain excluded by canonical portfolio-market eligibility rather
        than raising ticker-specific exceptions or jumping suspended sessions.
        """
        if self._calendar is None:
            raise RuntimeError("Daily Entry calendar has not been configured")
        prior = self._previous_session.get(pd.Timestamp(today).normalize())
        if prior is None:
            return {}
        entries = []
        for ticker in self._decisions.get(prior, ()):
            stock = all_dfs_fast.get(ticker)
            if stock is None:
                self.excluded_missing_market_ticker += 1
                continue
            prev_pos = get_fast_pos(stock, prior)
            today_pos = get_fast_pos(stock, today)
            if prev_pos < 0 or today_pos != prev_pos + 1:
                self.excluded_missing_adjacent_quote += 1
                continue
            entries.append((ticker, prev_pos, today_pos))
        return {today: entries} if entries else {}


class UnscoredDailyEntrySource(DailyEntrySource):
    """DL-independent eligible market-day universe for Daily Min optimization/control.

    Uses the same previous-exchange-session, adjacent-quote rule as scored Daily,
    but never loads model PIT scores or requires a Breakout setup.  Entry
    execution and broker fees remain owned by the ordinary Portfolio Engine.
    """

    def __init__(self):
        self._previous_session = {}
        self._calendar = None
        self._market_ref = None
        self._market_tickers = ()
        self.excluded_missing_market_ticker = 0
        self.excluded_missing_adjacent_quote = 0

    def entries_for(self, today, all_dfs_fast):
        if self._calendar is None:
            raise RuntimeError("Daily Entry calendar has not been configured")
        prior = self._previous_session.get(pd.Timestamp(today).normalize())
        if prior is None:
            return {}
        if all_dfs_fast is not self._market_ref:
            self._market_ref = all_dfs_fast
            self._market_tickers = tuple(sorted(all_dfs_fast))
        entries = []
        for ticker in self._market_tickers:
            stock = all_dfs_fast[ticker]
            previous_pos = get_fast_pos(stock, prior)
            today_pos = get_fast_pos(stock, today)
            if previous_pos < 0 or today_pos != previous_pos + 1:
                self.excluded_missing_adjacent_quote += 1
                continue
            entries.append((ticker, previous_pos, today_pos))
        return {today: entries} if entries else {}


_ACTIVE_ENTRY: ContextVar[DailyEntrySource | None] = ContextVar(
    "portfolio_scored_daily_entry_source", default=None,
)
# Signal preparation can run on a separate thread or spawned process. Workers
# receive the contract explicitly rather than relying on context inheritance.
_SIGNAL_DAILY_EXECUTION: ContextVar[str | None] = ContextVar(
    "portfolio_daily_signal_execution_contract", default=None,
)


@contextmanager
def daily_signal_execution_context(execution_contract: str) -> Iterator[None]:
    token = _SIGNAL_DAILY_EXECUTION.set(resolve_daily_execution_contract(execution_contract))
    try:
        yield
    finally:
        _SIGNAL_DAILY_EXECUTION.reset(token)


def current_daily_entry_source() -> DailyEntrySource | None:
    return _ACTIVE_ENTRY.get()


def current_daily_execution_contract() -> str:
    explicit_worker_contract = _SIGNAL_DAILY_EXECUTION.get()
    if explicit_worker_contract is not None:
        return explicit_worker_contract
    source = _ACTIVE_ENTRY.get()
    return resolve_daily_execution_contract(
        getattr(source, "daily_execution_contract", None) if source is not None else None
    )


@contextmanager
def portfolio_entry_policy_context(
    strategy: str, score_keys=None, *, unscored: bool = False,
    daily_execution_contract: str = DAILY_EXECUTION_LEGACY,
) -> Iterator[None]:
    """Select one mutually exclusive Daily scored/unscored or Breakout entry.

    A caller must explicitly request the DL-independent market-day universe;
    omitting verified score keys does NOT automatically change a scored arm.
    """
    if strategy == "Breakout":
        if score_keys is not None or unscored:
            raise ValueError("Breakout Entry cannot receive Daily candidate inputs")
        yield
        return
    if strategy != "Daily" or (score_keys is None) == (not unscored):
        raise ValueError("Daily Entry requires exactly one verified score universe or unscored mode")
    if _ACTIVE_ENTRY.get() is not None:
        raise RuntimeError("Nested Daily Entry source is prohibited")
    source = UnscoredDailyEntrySource() if unscored else DailyEntrySource(score_keys)
    source.daily_execution_contract = resolve_daily_execution_contract(daily_execution_contract)
    token = _ACTIVE_ENTRY.set(source)
    try:
        yield
    finally:
        _ACTIVE_ENTRY.reset(token)


@contextmanager
def optimizer_unscored_daily_entry_context() -> Iterator[None]:
    """Bind the Daily market-day entries only under an explicit Daily Optimizer science."""
    from core.scientific_runtime_context import get_active_scientific_section

    strategy = get_active_scientific_section("strategy") or {}
    if strategy.get("strategy_id") != ENTRY_DAILY:
        raise ValueError("Unscored Daily Optimizer requires strategy_id=daily")
    param_generation = get_active_scientific_section("parameter_generation") or {}
    execution = resolve_daily_execution_contract(
        param_generation.get("daily_execution_contract")
    )
    with portfolio_entry_policy_context("Daily", unscored=True, daily_execution_contract=execution):
        yield
