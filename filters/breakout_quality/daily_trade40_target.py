"""Daily-universal *counterfactual* Min 40-session net-R target.

For each eligible stock/date, pretend a fresh Min setup originated at that
close.  Its normal limit is the canonical rounded close + ATR*tolerance;
thereafter the official A2 single-stock execution path owns order continuation,
fees, realized risk, stops, exits and the forced 40th exchange-session close.
The hypothetical *originating signal* is NOT a real Min breakout signal and
must never be used to create executable candidates or production orders.

No holding/cash/K/R0/selector state enters a training label. Unfilled,
unexecutable, immature and uncovered dates have a NULL supervised R; they are
not transformed into a fake loss or zero return. The score-eligible universe
remains daily-universal even when a future training label is unavailable.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np
import pandas as pd

from core.price_utils import adjust_long_buy_limit_array
from core.portfolio_param_runtime import build_portfolio_params_signature
from core.strategy_params import V16StrategyParams
from filters.breakout_quality.trade_path_label import (
    build_signal_cache,
    simulate_realized_trade_path_label,
)
from filters.breakout_quality.workflow_io import load_dataset_frame

# The scientific Target/label identity belongs to the Oracle adapter.
# This module owns ONLY the canonical single-stock execution primitive.
HORIZON_SESSIONS = 40


@dataclass(frozen=True)
class ExecutableMinPeriod:
    start: pd.Timestamp
    end: pd.Timestamp | None
    params: V16StrategyParams
    source_id: str
    params_signature: str


@dataclass(frozen=True)
class CounterfactualTrade40Batch:
    target_raw_r: np.ndarray
    valid_mask: np.ndarray
    maturity_dates: np.ndarray
    reasons: tuple[str, ...]


def build_formal_min_execution_schedule(
    records: Iterable[dict[str, Any]], *, seed: int, param_sha256: str,
) -> tuple[tuple[ExecutableMinPeriod, ...], dict[str, Any]]:
    """Adapt the SAME resolved per-seed Min records consumed by Portfolio Replay.

    No independent parameter file, training teacher, optimizer or year switch is
    permitted in the Oracle. Effective dates describe the NEXT executable
    ticker session, not the information bar that produced the ranking.
    """
    if len(param_sha256) != 64 or any(c not in "0123456789abcdef" for c in param_sha256.lower()):
        raise ValueError("Formal Min Oracle missing canonical parameter identity")
    periods = []
    for record in records:
        params = record["params_obj"]
        if bool(params.use_breakout_quality_filter) or bool(params.use_breakout_quality_ranking):
            raise ValueError("Formal Min Oracle must consume DL-off baseline parameters")
        signature = build_portfolio_params_signature(params)
        if signature != str(record["params_signature"]):
            raise ValueError("Formal Min Oracle parameter signature disagrees with executable settings")
        start = pd.Timestamp(record["effective_date"]).normalize()
        end_raw = record.get("effective_end_date")
        end = None if end_raw is None else pd.Timestamp(end_raw).normalize()
        if end is not None and end < start:
            raise ValueError("Formal Min Oracle has reversed parameter period")
        periods.append(ExecutableMinPeriod(
            start=start, end=end, params=params,
            source_id="formal_min_roos_seed", params_signature=str(record["params_signature"]),
        ))
    periods.sort(key=lambda record: record.start)
    if not periods or [item.start for item in periods] != sorted({item.start for item in periods}):
        raise ValueError("Formal Min Oracle requires a unique ordered rolling Min schedule")
    if any(a.end is None or a.end >= b.start for a, b in zip(periods, periods[1:])):
        raise ValueError("Formal Min Oracle rolling periods overlap or lack a close date")
    provenance = {
        "origin": "formal_min_roos_rolling_per_seed",
        "seed": int(seed),
        "param_identity_sha256": str(param_sha256),
        "sources": [{"source_id": "formal_min_roos_seed", "sha256": str(param_sha256)}],
        "periods": [
            {"start": str(item.start.date()),
             "end": None if item.end is None else str(item.end.date()),
             "source_id": item.source_id, "params_signature": item.params_signature}
            for item in periods
        ],
    }
    return tuple(periods), provenance


def resolve_historical_min_period(
    schedule: tuple[ExecutableMinPeriod, ...], date: Any,
) -> ExecutableMinPeriod | None:
    day = pd.Timestamp(date).normalize()
    starts = [period.start for period in schedule]
    index = bisect_right(starts, day) - 1
    if index < 0:
        return None
    # A previous overlapping record may still be active if a more recent
    # time-window ended. Never borrow parameters from a *future* effective date.
    for period in reversed(schedule[:index + 1]):
        if period.start <= day and (period.end is None or day <= period.end):
            return period
    return None


def collect_canonical_exchange_sessions(
    csv_inputs: Iterable[tuple[str, Path]], *, benchmark_ticker: str,
) -> pd.DatetimeIndex:
    """Union the same *sanitized* stock quote sessions used by portfolio replay.

    A benchmark ETF is not an exchange calendar: it can lack a valid day. This
    extra prepass is confined to this new counterfactual training capability.
    """
    calendar: set[pd.Timestamp] = set()
    found_benchmark = False
    for ticker, path in csv_inputs:
        ticker = str(ticker)
        try:
            frame = load_dataset_frame(Path(path), ticker, min_rows=1)
        except (OSError, UnicodeDecodeError, ValueError, KeyError, TypeError):
            if ticker == benchmark_ticker:
                raise ValueError("Counterfactual Trade40 benchmark sanitized source invalid")
            continue
        if ticker == benchmark_ticker:
            found_benchmark = True
        calendar.update(pd.DatetimeIndex(frame.index).normalize().tolist())
    if not found_benchmark or len(calendar) <= HORIZON_SESSIONS:
        raise ValueError("Counterfactual Trade40 missing usable consolidated exchange calendar")
    return pd.DatetimeIndex(sorted(calendar)).normalize()


def compute_counterfactual_trade40_batch(
    frame: pd.DataFrame, positions: np.ndarray, *, ticker: str,
    schedule: tuple[ExecutableMinPeriod, ...], exchange_calendar: pd.DatetimeIndex,
) -> CounterfactualTrade40Batch:
    """Compute continuous net R with *real* execution primitives, never proxy PnL."""
    positions = np.asarray(positions, dtype=np.int64)
    if positions.ndim != 1:
        raise ValueError("counterfactual Trade40 positions must be 1D")
    calendar = pd.DatetimeIndex(exchange_calendar).normalize()
    if calendar.empty or calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("counterfactual Trade40 requires strictly sorted exchange sessions")
    dates = pd.DatetimeIndex(frame.index).normalize()
    if (positions < 0).any() or (positions >= len(dates)).any():
        raise ValueError("counterfactual Trade40 positions outside sanitized price dates")
    calendar_index = calendar.get_indexer(dates.take(positions))
    result = np.full(len(positions), np.nan, dtype=np.float32)
    valid = np.zeros(len(positions), dtype=bool)
    maturity_dates = np.full(len(positions), np.datetime64("NaT"), dtype="datetime64[D]")
    reasons: list[str] = []
    signal_caches: dict[str, Any] = {}
    for row, (position, calendar_pos) in enumerate(zip(positions, calendar_index)):
        if calendar_pos < 0 or calendar_pos + HORIZON_SESSIONS >= len(calendar):
            reasons.append("unmatured_or_missing_exchange_session")
            continue
        expiry = calendar[calendar_pos + HORIZON_SESSIONS]
        maturity_dates[row] = expiry.to_datetime64().astype("datetime64[D]")
        # The Portfolio assigns its Min parameter by the candidate ORDER day.
        # A suspended ticker's next actual bar may be later than the next
        # exchange session; always follow its executable ticker calendar.
        if position + 1 >= len(dates):
            reasons.append("no_next_executable_ticker_session")
            continue
        # The formal Min schedule applies to the order's executable session,
        # not the previous information bar; there is no alternative date mode.
        effective_day = dates[position + 1]
        if effective_day > expiry:
            reasons.append("no_executable_ticker_order_within_40_sessions")
            continue
        period = resolve_historical_min_period(schedule, effective_day)
        if period is None:
            reasons.append("no_historical_effective_Min_param")
            continue
        params = period.params
        cache = signal_caches.get(period.params_signature)
        if cache is None:
            cache = build_signal_cache(frame, params, ticker=ticker)
            signal_caches[period.params_signature] = cache
        atr = float(cache[0][position])
        close = float(frame["Close"].iloc[position])
        if not math.isfinite(atr) or atr <= 0.0 or not math.isfinite(close) or close <= 0.0:
            reasons.append("no_decision_time_ATR_or_close")
            continue
        # Canonical generate_signals defines this same buy-limit formula on
        # *real* setups. We override only the hypothetical initial setup.
        limit = float(adjust_long_buy_limit_array(
            np.asarray([close + atr * float(params.atr_buy_tol)], dtype=np.float64),
            ticker=ticker,
        )[0])
        observation = simulate_realized_trade_path_label(
            frame, ticker=ticker, signal_pos=int(position), params=params,
            teacher_effective_date=str(period.start.date()),
            precomputed_signals=cache, horizon_end_date=expiry,
            counterfactual_entry_limit=limit,
        )
        if observation.realized_net_r is None:
            reasons.append(str(observation.reason))
            continue
        value = float(observation.realized_net_r)
        if not math.isfinite(value):
            raise ValueError("canonical trade path produced nonfinite realized R")
        result[row] = value
        valid[row] = True
        reasons.append("filled_executed_net_R")
    return CounterfactualTrade40Batch(result, valid, maturity_dates, tuple(reasons))


__all__ = [
    "HORIZON_SESSIONS", "ExecutableMinPeriod", "CounterfactualTrade40Batch",
    "build_formal_min_execution_schedule", "resolve_historical_min_period",
    "collect_canonical_exchange_sessions", "compute_counterfactual_trade40_batch",
]
