"""Every eligible stock-day is an independent hypothetical Min trade decision.

Unlike signal-origin MinTrade targets, admission here never requires a formal
breakout setup or a live continuation. The information-date close is submitted
as the *next-session limit buy*, using that next session's legally effective
per-seed Min parameters. The canonical broker executor decides whether it
fills; after filling, the original Min management/fees/exit path is unchanged.
No future open is used to set the order. No-fill and non-executable closeouts
remain NULL, never synthetic zero/negative NetPnL.
"""
from __future__ import annotations

from functools import lru_cache
import math

import numpy as np
import pandas as pd

from filters.breakout_quality.daily_trade40_target import (
    CounterfactualTrade40Batch, ExecutableMinPeriod, resolve_historical_min_period,
)
from filters.breakout_quality.trade_path_label import (
    build_signal_cache, prepare_trade_path_inputs, simulate_realized_trade_path_label,
)


def compute_true_daily_mintrade_batch(
    frame: pd.DataFrame, positions: np.ndarray, *, ticker: str,
    schedule: tuple[ExecutableMinPeriod, ...], exchange_calendar: pd.DatetimeIndex,
    research_cutoff: str | pd.Timestamp,
) -> CounterfactualTrade40Batch:
    """One canonical broker-ledger result per stock/day, regardless of setup.

    All stock-days use *the same* information-close reference limit and one
    next-executable-ticker-session order policy. Historical signal activation
    is not a precondition. This is a hindsight Target, not an actual Min trade.
    """
    positions = np.asarray(positions, dtype=np.int64)
    dates = pd.DatetimeIndex(frame.index).normalize()
    exchange = pd.DatetimeIndex(exchange_calendar).normalize()
    cutoff = pd.Timestamp(research_cutoff).normalize()
    if (positions.ndim != 1 or (positions < 0).any() or
            (positions >= len(dates)).any() or exchange.empty or
            exchange.has_duplicates or not exchange.is_monotonic_increasing):
        raise ValueError("TrueDailyUniversal stock-day/calendar identity invalid")
    legal_sessions = exchange[exchange <= cutoff]
    # A suspended stock's next observed quote is not the exchange's next
    # executable session. A daily next-session decision may never skip days.
    exchange_positions = exchange.get_indexer(dates.take(positions))
    if (exchange_positions < 0).any():
        raise ValueError("TrueDaily stock-day absent from canonical exchange calendar")
    if legal_sessions.empty or not schedule:
        raise ValueError("TrueDailyUniversal lacks legal Min source/calendar")
    final_session = legal_sessions[-1]
    raw_r = np.full(len(positions), np.nan, dtype=np.float32)
    raw_pnl = np.full(len(positions), np.nan, dtype=np.float64)
    valid = np.zeros(len(positions), dtype=bool)
    maturity = np.full(len(positions), np.datetime64("NaT"), dtype="datetime64[D]")
    forced = np.zeros(len(positions), dtype=bool)
    reasons: list[str] = []
    close = frame["Close"].to_numpy(dtype=np.float64, copy=False)
    prepped = prepare_trade_path_inputs(frame)
    signal_cache: dict[str, tuple] = {}

    @lru_cache(maxsize=None)
    def active_period(day: pd.Timestamp):
        return resolve_historical_min_period(schedule, day)

    @lru_cache(maxsize=None)
    def cache_on(day: pd.Timestamp):
        period = active_period(day)
        if period is None:
            raise ValueError(
                f"no_historical_effective_Min_param during TrueDaily path: "
                f"{ticker} {day.date()}"
            )
        if period.params_signature not in signal_cache:
            signal_cache[period.params_signature] = build_signal_cache(
                frame, period.params, ticker=ticker,
            )
        return signal_cache[period.params_signature]

    forced_count = 0
    for idx, position in enumerate(positions):
        position = int(position)
        if position + 1 >= len(dates) or dates[position] >= final_session:
            reasons.append("no_future_executable_research_session")
            continue
        trade_day = dates[position + 1]
        day_at_exchange = int(exchange_positions[idx])
        if (day_at_exchange + 1 >= len(exchange) or
                trade_day != exchange[day_at_exchange + 1]):
            reasons.append("next_exchange_session_quote_unavailable")
            continue
        if trade_day > final_session:
            reasons.append("no_future_executable_research_session")
            continue
        period = active_period(trade_day)
        if period is None:
            # A missing PIT parameter is not a tradable loss/no-fill. The
            # caller must block if a missing effective P16 is inside OOS.
            reasons.append("no_historical_effective_Min_param")
            continue
        entry_limit = float(close[position])
        if not math.isfinite(entry_limit) or entry_limit <= 0:
            reasons.append("invalid_information_date_close_limit")
            continue
        result = simulate_realized_trade_path_label(
            frame, ticker=ticker, signal_pos=position, params=period.params,
            teacher_effective_date=str(period.start.date()),
            precomputed_signals=cache_on(trade_day),
            counterfactual_entry_limit=entry_limit,
            horizon_end_date=final_session,
            active_signal_cache_resolver=cache_on,
            prepared_inputs=prepped,
        )
        reasons.append(str(result.reason))
        # Only the canonical completed broker-ledger outcome is supervision.
        if result.realized_net_pnl is None:
            continue
        pnl = float(result.realized_net_pnl)
        r = float(result.realized_net_r)
        if not math.isfinite(pnl) or not math.isfinite(r) or not result.exit_date:
            raise ValueError("TrueDailyUniversal completed trade lacks finite settlement")
        if (pnl > 0 > r) or (r > 0 > pnl):
            raise ValueError("TrueDailyUniversal NetPnL / NetR signs differ")
        exit_day = pd.Timestamp(result.exit_date).normalize()
        if not dates[position] < exit_day <= final_session:
            raise ValueError("TrueDailyUniversal trade exits outside information horizon")
        raw_pnl[idx] = pnl
        raw_r[idx] = r
        valid[idx] = True
        maturity[idx] = exit_day.to_datetime64().astype("datetime64[D]")
        if result.forced_closeout:
            forced[idx] = True
            forced_count += 1
    return CounterfactualTrade40Batch(
        raw_r, valid, maturity, tuple(reasons),
        cutoff_truncated_rows=0, research_cutoff_closeout_rows=forced_count,
        target_raw_pnl=raw_pnl, research_cutoff_forced_mask=forced,
    )


__all__ = ["compute_true_daily_mintrade_batch"]
