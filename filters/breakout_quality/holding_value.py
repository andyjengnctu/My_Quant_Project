"""Independent holding-value task. Never an entry-ranking/Oracle score source."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import numpy as np
import pandas as pd

from core.position_management import rollforward_position_management_from_completed_bar
from core.position_step import execute_bar_step, SETTLEMENT_BASIS_LEDGER_NET
from filters.breakout_quality.features import normalize_ohlcv_array_windows
from filters.breakout_quality.daily_trade40_target import resolve_historical_min_period
from filters.breakout_quality.trade_path_label import (
    build_signal_cache, prepare_trade_path_inputs, simulate_realized_trade_path_label,
)

MODEL_ID = "HV1"
CONTRACT = {
    "task": "position_continuation_delta_r_v1", "model_id": MODEL_ID,
    "input": "completed_adjusted_stock_market_ohlcv_300_plus_position_v1",
    "target": "natural_p16_netpnl_minus_next_executable_exit_over_initial_risk",
    "sample": "first_executable_fill_per_original_signal_all_held_closes",
    "split": "whole_mature_episode_before_cutoff_last_calendar_year_validation",
    "loss": "episode_equal_weight_mse", "threshold_delta_r": 0.0,
    "optimizer_sampling": "one_uniform_held_state_per_episode_per_epoch",
    "architecture": "conv16k7_conv32k5_mean_state_mlp32_v1",
    "max_epochs": 12, "patience": 3, "batch_size": 512, "learning_rate": 0.001,
    "window": 300, "state_width": 6,
    "missing": "no_prediction_keep_original_rules", "execution": "next_session_open_persistent_request",
}


def position_context(position, *, frame, information_pos, completed_atr, completed_high, params):
    """Known state only; trailing protection is rolled from the completed bar."""
    # Management updates scalar fields only. A live position can also hold the
    # entire shared market/parameter context: never deep-copy that data graph.
    pos = dict(position)
    rollforward_position_management_from_completed_bar(
        pos, completed_high=completed_high, completed_atr=completed_atr, params=params)
    initial_qty = int(pos["initial_qty"])
    risk_per_share = int(pos["initial_risk_total_milli"]) / 1000.0 / initial_qty
    if risk_per_share <= 0:
        raise ValueError("Holding sample lacks positive canonical initial risk")
    entry_pos = int(frame.index.searchsorted(pd.Timestamp(pos["entry_trade_date"])))
    if entry_pos > information_pos:
        raise ValueError("Holding state predates its entry")
    close = float(frame["Close"].iloc[information_pos])
    entry = float(pos["entry_fill_price"])
    values = np.asarray([
        np.log1p(information_pos - entry_pos + 1),
        (close - entry) / risk_per_share,
        (close - float(pos["trailing_stop"])) / risk_per_share,
        (float(pos["highest_high_since_entry"]) - entry) / risk_per_share,
        float(pos["sold_half"]), float(pos["qty"]) / initial_qty,
    ], dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite completed-bar holding state")
    return values


def next_executable_exit(position, *, frame, information_pos, params, signal_cache):
    """Counterfactual planned exit, using ONLY the canonical settlement primitive.

    A pending protection/half-exit is honored first. A blocked request persists;
    no forced data-end liquidation or invented price supplies a label.
    """
    pos = deepcopy(position)
    dates = frame.index
    values = frame[["Open", "High", "Low", "Close", "Volume"]].to_numpy(copy=False)
    for t in range(information_pos + 1, len(frame)):
        atr = signal_cache(dates[t])[0]
        pos, _, _, _ = execute_bar_step(
            pos, atr[t - 1], True, values[t - 1, 3], *values[t], params,
            current_date=dates[t], y_high=values[t - 1, 1], return_milli=True,
            record_exec_contexts=False, sync_display_fields=False,
            settlement_basis=SETTLEMENT_BASIS_LEDGER_NET,
        )
        if int(pos["qty"]) == 0:
            return int(pos["realized_pnl_milli"]) / 1000.0, dates[t]
    return None


def build_ticker_holding_rows(*, ticker, frame, teacher, cutoff):
    """Observe natural canonical paths; label only fully matured episodes.

    Each original setup starts its canonical first-fill path once. It does not
    create a fresh trade for every day that the same order remains outstanding.
    """
    frame = frame.loc[:pd.Timestamp(cutoff)]
    if len(frame) < CONTRACT["window"] + 2:
        return []
    caches = {}
    periods = {}
    def active(day):
        if day not in periods:
            periods[day] = resolve_historical_min_period(teacher.schedule, day)
        return periods[day]
    def signals(day):
        period = active(day)
        if period is None:
            raise ValueError(f"Missing PIT holding teacher: {ticker}/{day}")
        if period.params_signature not in caches:
            caches[period.params_signature] = build_signal_cache(frame, period.params, ticker=ticker)
        return caches[period.params_signature]
    prepared = prepare_trade_path_inputs(frame)
    result = []
    for i in range(CONTRACT["window"] - 1, len(frame) - 1):
        period = active(frame.index[i + 1])
        if period is None or not bool(signals(frame.index[i + 1])[1][i]):
            continue
        observed = []
        def observe(*, position, information_pos, params, completed_atr, completed_high):
            # Existing natural/protective next-open obligations offer no choice.
            if position.get("pending_exit_action") == "STOP":
                return
            if bool(signals(frame.index[information_pos + 1])[2][information_pos]):
                return
            alternative = next_executable_exit(
                position, frame=frame, information_pos=information_pos,
                params=params, signal_cache=signals)
            if alternative is None:
                return
            context = position_context(
                position, frame=frame, information_pos=information_pos,
                completed_atr=completed_atr, completed_high=completed_high, params=params)
            observed.append((information_pos, context, alternative,
                             int(position["initial_risk_total_milli"]) / 1000.0))
        outcome = simulate_realized_trade_path_label(
            frame, ticker=ticker, signal_pos=i, params=period.params,
            teacher_effective_date=str(frame.index[i].date()),
            precomputed_signals=signals(frame.index[i + 1]),
            active_signal_cache_resolver=signals, prepared_inputs=prepared,
            holding_state_observer=observe,
        )
        if outcome.forced_closeout or outcome.realized_net_pnl is None:
            continue
        maturity = max([pd.Timestamp(outcome.exit_date)] + [row[2][1] for row in observed])
        if maturity > pd.Timestamp(cutoff):
            continue
        episode = f"{ticker}|{frame.index[i].date()}|{outcome.fill_date}"
        for info, context, (early_pnl, _early_date), risk in observed:
            result.append({
                "ticker": str(ticker), "information_date": str(frame.index[info].date()),
                "episode": episode, "entry_date": outcome.fill_date,
                "maturity_date": str(maturity.date()),
                "natural_net_pnl": outcome.realized_net_pnl, "early_net_pnl": early_pnl,
                "target": (outcome.realized_net_pnl - early_pnl) / risk,
                "state": context.tolist(),
            })
    return result


def episode_split(rows, *, cutoff, validation_start):
    """No episode spans train/validation; entire path must mature before cutoff."""
    cutoff, boundary = pd.Timestamp(cutoff), pd.Timestamp(validation_start)
    groups = rows.groupby("episode", sort=False)
    if (groups["maturity_date"].nunique() != 1).any() or (groups["entry_date"].nunique() != 1).any():
        raise ValueError("Inconsistent holding episode chronology")
    mature = pd.to_datetime(rows.maturity_date) < cutoff
    train = mature & (pd.to_datetime(rows.maturity_date) < boundary)
    validation = mature & (pd.to_datetime(rows.entry_date) >= boundary)
    if set(rows.loc[train, "episode"]) & set(rows.loc[validation, "episode"]):
        raise ValueError("Holding episode split leaked")
    return np.flatnonzero(train), np.flatnonzero(validation), np.flatnonzero(mature)


class HoldingFeatures:
    """Lazy 300x10 canonical windows, shared by training and live replay."""
    def __init__(self, frames, benchmark):
        self.frames = frames
        self.benchmark = benchmark
        self.arrays = {key: frame[["Open", "High", "Low", "Close", "Volume"]].to_numpy(dtype=float)
                       for key, frame in frames.items()}
        self.market = benchmark[["Open", "High", "Low", "Close", "Volume"]].to_numpy(dtype=float)

    def coordinates(self, ticker, day):
        frame = self.frames.get(str(ticker))
        if frame is None or day not in frame.index or day not in self.benchmark.index:
            return None
        s, m = int(frame.index.get_loc(day)), int(self.benchmark.index.get_loc(day))
        return None if min(s, m) < CONTRACT["window"] - 1 else (s, m)

    def batch(self, rows):
        width = CONTRACT["window"]
        stock = np.empty((len(rows), width, 5), dtype=float)
        market = np.empty_like(stock)
        for n, (ticker, day) in enumerate(zip(rows.ticker, rows.information_date)):
            coordinate = self.coordinates(str(ticker), pd.Timestamp(day))
            if coordinate is None:
                raise ValueError("Unscorable holding window")
            s, m = coordinate
            stock[n] = self.arrays[str(ticker)][s - width + 1:s + 1]
            market[n] = self.market[m - width + 1:m + 1]
        values = np.concatenate((normalize_ohlcv_array_windows(stock, stock[:, -1, 3]),
                                 normalize_ohlcv_array_windows(market, market[:, -1, 3])), axis=2)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite holding OHLCV")
        return values, np.asarray(rows.state.tolist(), dtype=np.float32)


def build_network():
    import torch
    class HoldingValueNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.sequence = torch.nn.Sequential(
                torch.nn.Conv1d(10, 16, 7, padding=3), torch.nn.ELU(),
                torch.nn.Conv1d(16, 32, 5, padding=2), torch.nn.ELU(),
                torch.nn.AdaptiveAvgPool1d(1))
            self.head = torch.nn.Sequential(torch.nn.Linear(38, 32), torch.nn.ELU(), torch.nn.Linear(32, 1))
        def forward(self, sequence, state):
            encoded = self.sequence(sequence.transpose(1, 2)).squeeze(-1)
            return self.head(torch.cat((encoded, state), dim=1)).squeeze(-1)
    return HoldingValueNetwork()
