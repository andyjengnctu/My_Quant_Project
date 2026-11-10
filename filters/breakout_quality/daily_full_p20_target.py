"""Every eligible stock-day is an independent hypothetical Daily Full/P20 trade decision.

Unlike signal-origin targets, P20 admission uses the legally effective Daily Full
common filters and ATR buy tolerance on the information day. The candidate
submits one next-session limit order using the signed P20 parameter schedule. The canonical broker executor decides whether it
fills; after filling, the original Min management/fees/exit path is unchanged.
No future open is used to set the order. No-fill and non-executable closeouts
remain NULL, never synthetic zero/negative NetPnL.
"""
from __future__ import annotations

from functools import lru_cache
from core.daily_common_entry_filters import daily_common_filter_signals
from core.portfolio_entry_policy import (
    DAILY_EXECUTION_FULL_FILTERS, daily_signal_execution_context,
)
import math

import numpy as np
import pandas as pd

from filters.breakout_quality.daily_trade40_target import (
    CounterfactualTrade40Batch, ExecutableMinPeriod, resolve_historical_min_period,
)
from filters.breakout_quality.trade_path_label import (
    build_signal_cache, prepare_trade_path_inputs, simulate_realized_trade_path_label,
)


def compute_daily_full_p20_batch(
    frame: pd.DataFrame, positions: np.ndarray, *, ticker: str,
    schedule: tuple[ExecutableMinPeriod, ...], exchange_calendar: pd.DatetimeIndex,
    research_cutoff: str | pd.Timestamp,
) -> CounterfactualTrade40Batch:
    """Full-cycle broker outcomes under the *P20* Daily Full decision policy.

    All feature-eligible days remain in the model cohort. Disabled filters,
    missing adjacent quotes or no fills create NULL outcomes, never a loss.
    One next-exchange-session order only; no continuation or future leakage.
    """
    positions = np.asarray(positions, dtype=np.int64)
    dates = pd.DatetimeIndex(frame.index).normalize()
    exchange = pd.DatetimeIndex(exchange_calendar).normalize()
    cutoff = pd.Timestamp(research_cutoff).normalize()
    if (positions.ndim != 1 or (positions < 0).any() or
            (positions >= len(dates)).any() or exchange.empty or
            exchange.has_duplicates or not exchange.is_monotonic_increasing):
        raise ValueError("Daily Full/P20 stock-day/calendar identity invalid")
    legal_sessions = exchange[exchange <= cutoff]
    # A suspended stock's next observed quote is not the exchange's next
    # executable session. A daily next-session decision may never skip days.
    exchange_positions = exchange.get_indexer(dates.take(positions))
    if (exchange_positions < 0).any():
        raise ValueError("Daily Full/P20 stock-day absent from canonical exchange calendar")
    if legal_sessions.empty or not schedule:
        raise ValueError("Daily Full/P20 lacks legal Min source/calendar")
    final_session = legal_sessions[-1]
    raw_r = np.full(len(positions), np.nan, dtype=np.float32)
    raw_pnl = np.full(len(positions), np.nan, dtype=np.float64)
    valid = np.zeros(len(positions), dtype=bool)
    maturity = np.full(len(positions), np.datetime64("NaT"), dtype="datetime64[D]")
    forced = np.zeros(len(positions), dtype=bool)
    reasons: list[str] = []
    prepped = prepare_trade_path_inputs(frame)
    signal_cache: dict[str, tuple] = {}
    entry_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    @lru_cache(maxsize=None)
    def active_period(day: pd.Timestamp):
        return resolve_historical_min_period(schedule, day)

    @lru_cache(maxsize=None)
    def cache_on(day: pd.Timestamp):
        period = active_period(day)
        if period is None:
            raise ValueError(
                f"no_historical_effective_Min_param during Daily Full/P20 path: "
                f"{ticker} {day.date()}"
            )
        if period.params_signature not in signal_cache:
            # AI: Label/Oracle workers do not inherit portfolio replay context.
            # Bind the declared ATR-limit execution at its numerical owner.
            with daily_signal_execution_context(DAILY_EXECUTION_FULL_FILTERS):
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
        # The *next session's* PIT P20 parameter settings determine today's
        # completed-information-bar admission and canonical ATR buy limit.
        if period.params_signature not in entry_cache:
            with daily_signal_execution_context(DAILY_EXECUTION_FULL_FILTERS):
                entry_cache[period.params_signature] = daily_common_filter_signals(
                    frame, period.params, ticker=ticker,
                    atr_main=cache_on(trade_day)[0],
                )
        admitted, limits = entry_cache[period.params_signature]
        if not bool(admitted[position]):
            reasons.append("daily_full_filter_rejected")
            continue
        entry_limit = float(limits[position])
        if not math.isfinite(entry_limit) or entry_limit <= 0:
            reasons.append("invalid_p20_information_date_atr_limit")
            continue
        # The first evaluation isolates the next *exchange* session.  An
        # unfilled order expires there: never let the canonical Breakout Min
        # continuation logic turn it into a later Daily Full/P20 trade.  Reuse the
        # same broker entry primitive; the second call is only for the genuine
        # initial fills and follows the unchanged full-cycle Min exit/ledger.
        first_session = simulate_realized_trade_path_label(
            frame, ticker=ticker, signal_pos=position, params=period.params,
            teacher_effective_date=str(period.start.date()),
            precomputed_signals=cache_on(trade_day),
            counterfactual_entry_limit=entry_limit,
            horizon_end_date=trade_day,
            active_signal_cache_resolver=cache_on,
            prepared_inputs=prepped,
        )
        if (first_session.fill_type != "INITIAL_FILL" or
                first_session.fill_date != str(trade_day.date())):
            reasons.append(str(first_session.reason))
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
        if (result.fill_date is not None and
                (result.fill_type != "INITIAL_FILL" or
                 result.fill_date != str(trade_day.date()))):
            raise ValueError("Daily Full/P20 full-cycle replay disagrees with first-session fill")
        reasons.append(str(result.reason))
        # Only the canonical completed broker-ledger outcome is supervision.
        if result.realized_net_pnl is None:
            continue
        pnl = float(result.realized_net_pnl)
        r = float(result.realized_net_r)
        if not math.isfinite(pnl) or not math.isfinite(r) or not result.exit_date:
            raise ValueError("Daily Full/P20 completed trade lacks finite settlement")
        if (pnl > 0 > r) or (r > 0 > pnl):
            raise ValueError("Daily Full/P20 NetPnL / NetR signs differ")
        exit_day = pd.Timestamp(result.exit_date).normalize()
        if not dates[position] < exit_day <= final_session:
            raise ValueError("Daily Full/P20 trade exits outside information horizon")
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


__all__ = ["compute_daily_full_p20_batch"]
