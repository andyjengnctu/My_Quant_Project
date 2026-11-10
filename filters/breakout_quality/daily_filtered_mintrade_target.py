"""Filtered Daily close-limit supervision composed from canonical primitives.

Admission uses completed information-day filters under the next session's
legally effective parameters. Filled paths, fees, exits and NULL/maturity
come from the existing daily-close broker path; there is no second simulator.
"""

from dataclasses import replace
import numpy as np
import pandas as pd
from core.daily_common_entry_filters import daily_common_filter_signals
from core.portfolio_entry_policy import (
    DAILY_EXECUTION_CLOSE_FILTERS,
    daily_signal_execution_context,
)
from filters.breakout_quality.daily_trade40_target import (
    CounterfactualTrade40Batch,
    resolve_historical_min_period,
)
from filters.breakout_quality.trade_path_label import build_signal_cache
from filters.breakout_quality.true_daily_mintrade_target import (
    compute_true_daily_mintrade_batch,
)


def compute_daily_filtered_mintrade_batch(
    frame, positions, *, ticker, schedule, exchange_calendar, research_cutoff
):
    positions = np.asarray(positions, dtype=np.int64)
    dates = pd.DatetimeIndex(frame.index).normalize()
    if (
        positions.ndim != 1
        or np.any(positions < 0)
        or np.any(positions >= len(dates))
        or np.any(np.diff(positions) <= 0)
        or not schedule
    ):
        raise ValueError("Filtered Daily stock-day/schedule identity invalid")
    calendar = pd.DatetimeIndex(exchange_calendar).normalize()
    if (
        calendar.empty
        or calendar.has_duplicates
        or not calendar.is_monotonic_increasing
        or dates.has_duplicates
        or not dates.is_monotonic_increasing
        or (calendar.get_indexer(dates.take(positions)) < 0).any()
    ):
        raise ValueError("Filtered Daily calendar/information-day identity invalid")
    n = len(positions)
    raw_r, raw_pnl = np.full(n, np.nan, np.float32), np.full(n, np.nan, np.float64)
    valid, forced = np.zeros(n, bool), np.zeros(n, bool)
    maturity = np.full(n, np.datetime64("NaT"), dtype="datetime64[D]")
    reasons = ["daily_filter_rejected"] * n
    admitted_rows, by_signature = [], {}
    with daily_signal_execution_context(DAILY_EXECUTION_CLOSE_FILTERS):
        for row, pos in enumerate(positions):
            if pos + 1 >= len(dates):
                reasons[row] = "no_future_executable_research_session"
                continue
            period = resolve_historical_min_period(schedule, dates[pos + 1])
            if period is None:
                reasons[row] = "no_historical_effective_Min_param"
                continue
            if period.params_signature not in by_signature:
                atr = build_signal_cache(frame, period.params, ticker=ticker)[0]
                by_signature[period.params_signature] = daily_common_filter_signals(
                    frame, period.params, ticker=ticker, atr_main=atr
                )[0]
            if bool(by_signature[period.params_signature][pos]):
                admitted_rows.append(row)
        if not admitted_rows:
            return CounterfactualTrade40Batch(
                raw_r,
                valid,
                maturity,
                tuple(reasons),
                0,
                0,
                target_raw_pnl=raw_pnl,
                research_cutoff_forced_mask=forced,
            )
        selected = np.asarray(admitted_rows, dtype=np.int64)
        batch = compute_true_daily_mintrade_batch(
            frame,
            positions[selected],
            ticker=ticker,
            schedule=schedule,
            exchange_calendar=exchange_calendar,
            research_cutoff=research_cutoff,
        )
    raw_r[selected], raw_pnl[selected] = batch.target_raw_r, batch.target_raw_pnl
    valid[selected], maturity[selected] = batch.valid_mask, batch.maturity_dates
    forced[selected] = batch.research_cutoff_forced_mask
    for row, reason in zip(selected, batch.reasons):
        reasons[row] = reason
    return replace(
        batch,
        target_raw_r=raw_r,
        target_raw_pnl=raw_pnl,
        valid_mask=valid,
        maturity_dates=maturity,
        reasons=tuple(reasons),
        research_cutoff_forced_mask=forced,
    )
