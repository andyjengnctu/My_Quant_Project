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


def current_daily_entry_source() -> DailyEntrySource | None:
    return _ACTIVE_ENTRY.get()


@contextmanager
def portfolio_entry_policy_context(
    strategy: str, score_keys=None, *, unscored: bool = False,
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
    with portfolio_entry_policy_context("Daily", unscored=True):
        yield
