"""Formal Min signal-event 40/60/full-cycle broker-ledger outcome targets.

At each information date the single-stock *signal* lifecycle uses exactly the
canonical new-setup/continuation/cleanup primitives. Each outcome is then
simulated from that decision's executable signal state, not a fictitious new
order at every close. Portfolio holdings, cash, K/R0 and selector remain outside
this per-stock counterfactual. Missing setups/fills are NULL, never fake zeros.
The DL score-eligible stock-day cohort is unchanged; missing truths rank last.
"""

from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from core.capital_policy import resolve_single_backtest_sizing_capital
from core.extended_signals import create_signal_tracking_state, should_clear_extended_signal
from core.portfolio_param_runtime import (
    build_portfolio_params_signature,
    build_active_param_ensemble_objects_from_payload,
    build_active_param_objects_from_payload,
)
from core.strategy_params import V16StrategyParams
from filters.breakout_quality.trade_path_label import (
    TRADE_PATH_REASON_UNFILLED_SUPERSEDED,
    TRADE_PATH_REASON_UNFILLED_TERMINATED,
    build_signal_cache,
    prepare_trade_path_inputs,
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
    """An outcome's maturity date is its actual resolution date, if earlier.

    cutoff_truncated_rows counts *potential* horizon-over-cutoff rows, not
    forced closes. research_cutoff_closeout_rows counts actual forced closes.
    """
    target_raw_r: np.ndarray
    valid_mask: np.ndarray
    maturity_dates: np.ndarray
    reasons: tuple[str, ...]
    cutoff_truncated_rows: int = 0
    research_cutoff_closeout_rows: int = 0
    # Opt-in canonical broker-ledger net PnL, not a second path/sizing engine.
    target_raw_pnl: np.ndarray | None = None
    # Original simulator evidence: a still-open trade forced closed only by
    # the study-end boundary is NOT a naturally FIT-matured label.
    research_cutoff_forced_mask: np.ndarray | None = None


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



def resolve_formal_min_param_records(base_kind: str, base_params: Any) -> tuple[dict[str, Any], ...]:
    """Resolve the SAME P16 Min active parameter records used in Portfolio Replay.

    The retired event-only Oracle must not remain a dependency of the daily
    counterfactual label. This adapter consumes the canonical parameter owner.
    """
    if base_kind == "rolling_active_param_ensemble":
        records = build_active_param_ensemble_objects_from_payload(base_params)
        for record in records:
            members = record["members"]
            if len(members) != 1:
                raise ValueError("Formal Min P16 Base-Finalist-Best requires one member")
            record["params_obj"] = members[0]["params_obj"]
    elif base_kind == "rolling_oos_param_schedule":
        records = build_active_param_objects_from_payload(base_params)
    else:
        raise ValueError(f"Formal Min needs rolling P16; invalid source: {base_kind}")
    dates = [pd.Timestamp(row["effective_date"]).normalize() for row in records]
    if not dates or dates != sorted(set(dates)):
        raise ValueError("Formal Min effective schedule missing or duplicated")
    return tuple(records)


def rank_trade40_outcome_frame(
    frame: pd.DataFrame, *, value_column: str = "realized_net_r",
) -> dict[tuple[str, str], float]:
    """Canonical daily hindsight ranking without fabricating censored economic outcomes.

    Unfilled outcomes remain NaN in the provider; only their relative selector
    order is below actually executed outcomes. This helper is not an event
    Target producer and cannot publish PIT scores or fit a DL model.
    """
    if value_column not in {"realized_net_r", "realized_net_pnl"}:
        raise ValueError("Daily MinTrade outcome column is not an approved economic metric")
    if frame.empty or not {"ticker", "date", value_column}.issubset(frame):
        raise ValueError("Daily MinTrade missing scoreable stock-day rows")
    if frame.duplicated(["ticker", "date"]).any():
        raise ValueError("Daily Trade40 duplicate ticker/date")
    from filters.breakout_quality.continuous_ranker_data import build_same_date_percentile_targets
    filled = frame[value_column].notna()
    numeric = pd.to_numeric(frame[value_column], errors="raise").astype(float)
    if not np.isfinite(numeric.loc[filled].to_numpy(dtype=float)).all():
        raise ValueError("Daily Trade40 executed economic outcome is nonfinite")
    if not filled.any():
        raise ValueError("Daily Trade40 no executed economic outcome truth")
    daily_min = numeric.groupby(frame["date"]).transform("min").fillna(0.0)
    numeric.loc[~filled] = daily_min.loc[~filled] - (1.0 + daily_min.loc[~filled].abs())
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Daily Trade40 ranking sentinel nonfinite")
    scores = build_same_date_percentile_targets(
        numeric.to_numpy(dtype=float), np.ones(len(frame), dtype=bool), frame["date"],
    ).astype(float)
    if not np.isfinite(scores).all():
        raise ValueError("Daily Trade40 nonfinite daily rank")
    return dict(zip(zip(frame["ticker"].astype(str), frame["date"].astype(str)), scores.tolist()))

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
    horizon_sessions: int | None = HORIZON_SESSIONS,
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
    if horizon_sessions is not None and (
            not isinstance(horizon_sessions, int) or isinstance(horizon_sessions, bool) or horizon_sessions <= 0):
        raise ValueError("Counterfactual Min horizon must be positive exchange sessions")
    if not found_benchmark or len(calendar) <= 1:
        raise ValueError("Counterfactual Min missing usable consolidated exchange calendar")
    return pd.DatetimeIndex(sorted(calendar)).normalize()


def compute_counterfactual_trade40_batch(
    frame: pd.DataFrame, positions: np.ndarray, *, ticker: str,
    schedule: tuple[ExecutableMinPeriod, ...], exchange_calendar: pd.DatetimeIndex,
    horizon_sessions: int | None = HORIZON_SESSIONS,
    research_cutoff: str | pd.Timestamp | None = None,
    include_net_pnl: bool = False,
    signal_state_reset_date: str | pd.Timestamp | None = None,
) -> CounterfactualTrade40Batch:
    """One Min path for 40/60/any H and full-cycle; cutoff is an exit deadline.

    Neither the target horizon nor the study-end boundary changes portfolio
    trades.  Missing executable closing quotes stay censored, never invented.
    """
    if horizon_sessions is not None and (
            not isinstance(horizon_sessions, int) or isinstance(horizon_sessions, bool) or horizon_sessions <= 0):
        raise ValueError("Counterfactual Min horizon must be positive exchange sessions")
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
    cutoff = (calendar[-1] if research_cutoff is None else
              pd.Timestamp(research_cutoff).normalize())
    eligible_sessions = calendar[calendar <= cutoff]
    if not len(eligible_sessions):
        raise ValueError("Counterfactual Min research cutoff has no exchange sessions")
    last_research_session = eligible_sessions[-1]
    result = np.full(len(positions), np.nan, dtype=np.float32)
    # Float64 preserves canonical milli-to-money PnL on large allocations.
    net_pnl = np.full(len(positions), np.nan, dtype=np.float64) if include_net_pnl else None
    valid = np.zeros(len(positions), dtype=bool)
    maturity_dates = np.full(len(positions), np.datetime64("NaT"), dtype="datetime64[D]")
    reasons: list[str] = []
    forced_at_research_cutoff = np.zeros(len(positions), dtype=bool)
    signal_caches: dict[str, Any] = {}
    # Within one stock's canonical replay, the same executable trading day
    # can be visited by the signal-state pass and many overlapping 40D paths.
    # Memoize its already-verified active period, never the simulated outcome.
    active_periods_by_date: dict[pd.Timestamp, ExecutableMinPeriod | None] = {}

    def active_period_for_date(day: pd.Timestamp) -> ExecutableMinPeriod | None:
        key = pd.Timestamp(day).normalize()
        if key not in active_periods_by_date:
            active_periods_by_date[key] = resolve_historical_min_period(schedule, key)
        return active_periods_by_date[key]

    @lru_cache(maxsize=None)
    def active_signal_cache_for_date(day: pd.Timestamp):
        # P16 Base-Finalist-Best has exactly one member. Both supported rolling
        # payload kinds therefore use Portfolio's ordinary identity path:
        # origin params own order/position rules, current params own signals.
        active_period = active_period_for_date(day)
        if active_period is None:
            # A gap during continuation/holding is missing source truth, never
            # an economically censored trade and never permission to keep old
            # or borrow future parameters.
            raise ValueError(
                "no_historical_effective_Min_param during trade path: "
                f"ticker={ticker}, date={day.date()}"
            )
        active_cache = signal_caches.get(active_period.params_signature)
        if active_cache is None:
            active_cache = build_signal_cache(frame, active_period.params, ticker=ticker)
            signal_caches[active_period.params_signature] = active_cache
        return active_cache

    # Reconstruct the *single-stock* pending signal that Portfolio would have
    # before ranking each candidate's next executable session. We deliberately
    # do not assume a fill/selection on earlier dates (those are portfolio
    # resource decisions). Both new setups and shadow continuation use the
    # canonical signal-state owner. No new entry/stop/fee formula lives here.
    requested = set(int(position) for position in positions)
    decisions: dict[int, dict[str, Any]] = {}
    pending: dict[str, Any] | None = None
    open_values = frame["Open"].to_numpy(dtype=np.float64, copy=False)
    high_values = frame["High"].to_numpy(dtype=np.float64, copy=False)
    low_values = frame["Low"].to_numpy(dtype=np.float64, copy=False)
    close_values = frame["Close"].to_numpy(dtype=np.float64, copy=False)
    volume_values = frame["Volume"].to_numpy(dtype=np.float64, copy=False)
    max_info_pos = int(max(requested)) if requested else -1
    reset_date = (None if signal_state_reset_date is None else
                  pd.Timestamp(signal_state_reset_date).normalize())
    if reset_date is not None and pd.isna(reset_date):
        raise ValueError("Min Oracle signal-state reset date invalid")
    previous_trade_day = None
    for information_pos in range(max_info_pos + 1):
        executable_pos = information_pos + 1
        if executable_pos >= len(dates) or dates[information_pos] >= cutoff:
            continue
        trade_day = dates[executable_pos]
        # A historical teacher is allowed before OOS for FIT labels, but its
        # pending signal must not leak into the official evaluation segment.
        # The Oracle starts with no pending state on its first P16 session.
        if (reset_date is not None and trade_day >= reset_date and
                (previous_trade_day is None or previous_trade_day < reset_date)):
            pending = None
        previous_trade_day = trade_day
        active_period = active_period_for_date(trade_day)
        if active_period is None:
            if trade_day >= schedule[0].start and trade_day <= last_research_session:
                raise ValueError(
                    "no_historical_effective_Min_param during signal lifecycle: "
                    f"ticker={ticker}, date={trade_day.date()}"
                )
            pending = None
            continue
        cache = active_signal_cache_for_date(trade_day)
        if bool(cache[1][information_pos]):
            # The next executable date owns the signal formula and the order's
            # originating P16 settings. Do not recalculate a new hypothetical
            # Close+ATR order when only a continuation exists.
            pending = create_signal_tracking_state(
                cache[3][information_pos], cache[0][information_pos],
                active_period.params, ticker=ticker,
                security_profile=frame.attrs.get("security_profile"),
                signal_date=dates[information_pos],
            )
        if information_pos in requested and pending is not None:
            decisions[information_pos] = deepcopy(
                pending, memo={id(pending["_params_obj"]): pending["_params_obj"]},
            )
        if pending is None or trade_day > last_research_session:
            continue
        origin_params = pending["_params_obj"]
        sizing_capital = resolve_single_backtest_sizing_capital(
            origin_params, origin_params.initial_capital,
        )
        if should_clear_extended_signal(
            pending, low_values[executable_pos], high_values[executable_pos],
            t_open=open_values[executable_pos],
            t_close=close_values[executable_pos],
            t_volume=volume_values[executable_pos],
            y_close=close_values[information_pos],
            y_high=high_values[information_pos],
            y_atr=cache[0][information_pos],
            y_ind_sell=cache[2][information_pos],
            sizing_capital=sizing_capital, current_date=trade_day,
            params=origin_params, copy_shadow_position=False,
        ):
            pending = None

    truncated = 0
    cutoff_closeouts = 0
    prepared_inputs = prepare_trade_path_inputs(frame)
    for row, (position, calendar_pos) in enumerate(zip(positions, calendar_index)):
        if calendar_pos < 0 or dates[position] >= last_research_session:
            reasons.append("no_future_executable_research_session")
            continue
        horizon_expiry = (None if horizon_sessions is None or
                          calendar_pos + horizon_sessions >= len(calendar) else
                          calendar[calendar_pos + horizon_sessions])
        # An unavailable H-th session is NOT grounds to delete the stock-day.
        # The single research-cutoff deadline applies to every horizon.
        expiry = min(horizon_expiry, last_research_session) if horizon_expiry is not None else last_research_session
        cutoff_limited = horizon_expiry is None or horizon_expiry > last_research_session
        # This counter means the *declared* horizon lies beyond the Research
        # cutoff, NOT that the actual trade was forcibly closed there. Natural
        # exits can finish earlier; research_cutoff_closeout_rows tracks the
        # separate number of actual executed closeouts at the cutoff.
        if cutoff_limited:
            truncated += 1
        maturity_dates[row] = expiry.to_datetime64().astype("datetime64[D]")
        # An actual normal setup or a still-live continuation is required.
        # All other feature-eligible rows stay in the score cohort but have a
        # censored (NULL) economic outcome; they are NOT artificial new buys.
        if active_period_for_date(dates[position + 1]) is None:
            reasons.append("no_historical_effective_Min_param")
            continue
        state = decisions.get(int(position))
        if state is None:
            reasons.append("no_formal_min_signal_or_continuation")
            continue
        origin_day = pd.Timestamp(state["signal_date"]).normalize()
        origin_pos = int(dates.get_indexer([origin_day])[0])
        if origin_pos < 0 or origin_pos > int(position):
            raise ValueError("Min Oracle signal origin not in the sanitized ticker calendar")
        params = state["_params_obj"]
        active_period = active_period_for_date(dates[position + 1])
        if active_period is None:
            reasons.append("no_historical_effective_Min_param")
            continue
        observation = simulate_realized_trade_path_label(
            frame, ticker=ticker, signal_pos=origin_pos, params=params,
            teacher_effective_date=str(origin_day.date()),
            precomputed_signals=active_signal_cache_for_date(dates[position + 1]),
            horizon_end_date=expiry, decision_pos=int(position),
            starting_signal_state=state,
            active_signal_cache_resolver=active_signal_cache_for_date,
            prepared_inputs=prepared_inputs,
        )
        # The *deadline* and the date a label is actually known are different.
        # A natural stop/indicator exit resolves the full-cycle or H-day label
        # immediately; it must not be recorded as maturing at the later H /
        # Research cutoff. An unfilled order superseded/terminated early is
        # likewise known to be censored then, without fabricating an R value.
        observed_end = (observation.exit_date if observation.realized_net_r is not None
                        else observation.label_eval_end_date if observation.reason in (
                            TRADE_PATH_REASON_UNFILLED_SUPERSEDED,
                            TRADE_PATH_REASON_UNFILLED_TERMINATED,
                        ) else None)
        if observation.realized_net_r is not None and observed_end is None:
            raise ValueError("completed formal Min trade path has no executable exit date")
        if observed_end is not None:
            observed_day = pd.Timestamp(observed_end).normalize()
            if observed_day < dates[position] or observed_day > expiry:
                raise ValueError("formal Min target completed outside its legal observation window")
            maturity_dates[row] = observed_day.to_datetime64().astype("datetime64[D]")
        if observation.realized_net_r is None:
            reasons.append(str(observation.reason))
            continue
        if cutoff_limited and observation.forced_closeout:
            cutoff_closeouts += 1
            forced_at_research_cutoff[row] = True
        value = float(observation.realized_net_r)
        if not math.isfinite(value):
            raise ValueError("canonical trade path produced nonfinite realized R")
        if net_pnl is not None:
            # Reuse the very SAME canonical executed trade; do not estimate PnL
            # using a separate fee, risk denominator, or theoretical fill.
            pnl_value = observation.realized_net_pnl
            if pnl_value is None or not math.isfinite(float(pnl_value)):
                raise ValueError("canonical filled Min path lacks finite broker-ledger NetPnL")
            if (value > 0 and float(pnl_value) <= 0) or (value < 0 and float(pnl_value) >= 0):
                raise ValueError("canonical Min Net-R and NetPnL signs disagree")
            net_pnl[row] = float(pnl_value)
        result[row] = value
        valid[row] = True
        reasons.append("filled_executed_net_R")
    return CounterfactualTrade40Batch(
        result, valid, maturity_dates, tuple(reasons), truncated, cutoff_closeouts,
        target_raw_pnl=net_pnl,
        research_cutoff_forced_mask=forced_at_research_cutoff,
    )


__all__ = [
    "HORIZON_SESSIONS", "ExecutableMinPeriod", "CounterfactualTrade40Batch",
    "build_formal_min_execution_schedule", "resolve_historical_min_period",
    "collect_canonical_exchange_sessions", "compute_counterfactual_trade40_batch",
]
