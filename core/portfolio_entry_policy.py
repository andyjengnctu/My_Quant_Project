"""Canonical portfolio entry-source policy; never owns fills, Min exits or cash.

Breakout remains the implicit/default strategy.  A Daily strategy consumes only
verified (ticker, information-date) *score eligibility*; future Positive/NetPnL
values are not read here.  The official score/Positive selector decides whether
the resulting candidate is admissible.  The existing Portfolio Engine executes
both modes through the same position and accounting path.
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


_ACTIVE_ENTRY: ContextVar[DailyEntrySource | None] = ContextVar(
    "portfolio_scored_daily_entry_source", default=None,
)


def current_daily_entry_source() -> DailyEntrySource | None:
    return _ACTIVE_ENTRY.get()


@contextmanager
def portfolio_entry_policy_context(strategy: str, score_keys=None) -> Iterator[None]:
    """Explicitly opt in for one authorized Daily replay; default is Breakout."""
    if strategy == "Breakout":
        if score_keys is not None:
            raise ValueError("Breakout Entry cannot receive Daily score candidate inputs")
        yield
        return
    if strategy != "Daily" or score_keys is None:
        raise ValueError("Only authorized Daily Entry with a score universe is supported")
    if _ACTIVE_ENTRY.get() is not None:
        raise RuntimeError("Nested Daily Entry source is prohibited")
    source = DailyEntrySource(score_keys)
    token = _ACTIVE_ENTRY.set(source)
    try:
        yield
    finally:
        _ACTIVE_ENTRY.reset(token)
