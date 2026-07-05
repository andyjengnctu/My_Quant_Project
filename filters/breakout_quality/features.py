"""Raw OHLCV feature and path-based label builders for breakout quality filter."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from core.price_utils import adjust_long_buy_fill_price, adjust_long_buy_limit, adjust_long_stop_price, is_locked_limit_up_bar
from core.signal_utils import tv_atr, tv_true_range
from filters.breakout_quality.contract import (
    CONTEXT_COLUMNS,
    FEATURE_COLUMNS,
    LABEL_IGNORE,
    LABEL_PASS,
    LABEL_REJECT,
    BreakoutQualityLabelPolicy,
)


@dataclass(frozen=True)
class BreakoutQualityDataset:
    features: np.ndarray
    context: np.ndarray
    labels: np.ndarray
    events: pd.DataFrame


def _safe_iqr(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    q75, q25 = np.percentile(finite, [75, 25])
    iqr = float(q75 - q25)
    if not math.isfinite(iqr) or iqr <= 0.0:
        return 1.0
    return iqr


def _normalize_volume_window(volume: np.ndarray) -> np.ndarray:
    log_volume = np.log1p(np.maximum(np.asarray(volume, dtype=np.float64), 0.0))
    finite = log_volume[np.isfinite(log_volume)]
    if finite.size == 0:
        return np.zeros_like(log_volume, dtype=np.float32)
    median = float(np.median(finite))
    iqr = _safe_iqr(log_volume)
    return ((log_volume - median) / iqr).astype(np.float32)


def _normalize_ohlcv_window(window: pd.DataFrame, anchor_close: float, prefix: str = "") -> np.ndarray:
    if not math.isfinite(anchor_close) or anchor_close <= 0.0:
        raise ValueError("anchor_close 必須是有限正數")
    open_norm = window["Open"].to_numpy(dtype=np.float64, copy=False) / anchor_close - 1.0
    high_norm = window["High"].to_numpy(dtype=np.float64, copy=False) / anchor_close - 1.0
    low_norm = window["Low"].to_numpy(dtype=np.float64, copy=False) / anchor_close - 1.0
    close_norm = window["Close"].to_numpy(dtype=np.float64, copy=False) / anchor_close - 1.0
    volume_norm = _normalize_volume_window(window["Volume"].to_numpy(dtype=np.float64, copy=False))
    return np.column_stack([open_norm, high_norm, low_norm, close_norm, volume_norm]).astype(np.float32)


def build_breakout_quality_feature(
    stock_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    *,
    event_pos: int,
    high_len: int,
    breakout_level: float,
    policy: BreakoutQualityLabelPolicy,
) -> tuple[np.ndarray, np.ndarray] | None:
    feature_window = int(policy.feature_window_bars)
    start_pos = int(event_pos) - feature_window + 1
    if start_pos < 0:
        return None

    stock_window = stock_df.iloc[start_pos:int(event_pos) + 1]
    if len(stock_window) != feature_window:
        return None
    event_date = stock_df.index[int(event_pos)]
    if event_date not in benchmark_df.index:
        return None
    benchmark_pos = int(benchmark_df.index.get_loc(event_date))
    benchmark_start = benchmark_pos - feature_window + 1
    if benchmark_start < 0:
        return None
    benchmark_window = benchmark_df.iloc[benchmark_start:benchmark_pos + 1]
    if len(benchmark_window) != feature_window:
        return None

    close_d0 = float(stock_df["Close"].iloc[int(event_pos)])
    benchmark_close_d0 = float(benchmark_df["Close"].iloc[benchmark_pos])
    if close_d0 <= 0.0 or benchmark_close_d0 <= 0.0 or not math.isfinite(breakout_level) or breakout_level <= 0.0:
        return None

    stock_features = _normalize_ohlcv_window(stock_window, close_d0)
    benchmark_features = _normalize_ohlcv_window(benchmark_window, benchmark_close_d0)
    seq_features = np.column_stack([stock_features, benchmark_features]).astype(np.float32)

    high_len_span = max(1, int(policy.high_len_max) - int(policy.high_len_min))
    context = np.asarray([
        (int(high_len) - int(policy.high_len_min)) / high_len_span,
        float(breakout_level) / close_d0 - 1.0,
        close_d0 / float(breakout_level) - 1.0,
        float(stock_df["High"].iloc[int(event_pos)]) / float(breakout_level) - 1.0,
    ], dtype=np.float32)
    return seq_features, context


def build_candidate_event_positions(stock_df: pd.DataFrame, high_lens: Iterable[int]) -> list[dict]:
    high = stock_df["High"].to_numpy(dtype=np.float64, copy=False)
    close = stock_df["Close"].to_numpy(dtype=np.float64, copy=False)
    prev_close = np.empty_like(close)
    prev_close[0] = close[0] if close.size else np.nan
    if close.size > 1:
        prev_close[1:] = close[:-1]

    events: list[dict] = []
    for high_len in high_lens:
        high_len_int = int(high_len)
        rolling_high = pd.Series(high).shift(1).rolling(high_len_int, min_periods=high_len_int).max().to_numpy(dtype=np.float64)
        prev_high = np.empty_like(rolling_high)
        prev_high[0] = rolling_high[0] if rolling_high.size else np.nan
        if rolling_high.size > 1:
            prev_high[1:] = rolling_high[:-1]
        crossover = (close > rolling_high) & (prev_close <= prev_high)
        crossover[0] = False
        for pos in np.flatnonzero(crossover):
            level = float(rolling_high[int(pos)])
            if math.isfinite(level) and level > 0.0:
                events.append({"pos": int(pos), "date": stock_df.index[int(pos)], "high_len": high_len_int, "breakout_level": level})
    return events


def _label_from_path(high_values: np.ndarray, low_values: np.ndarray, *, entry_price: float, risk_r: float, policy: BreakoutQualityLabelPolicy) -> tuple[int, str, float, float]:
    if risk_r <= 0.0 or not math.isfinite(risk_r) or not math.isfinite(entry_price) or entry_price <= 0.0:
        return LABEL_IGNORE, "invalid_risk", np.nan, np.nan

    best_mfe = -np.inf
    worst_mae = np.inf
    for high_price, low_price in zip(high_values, low_values):
        if not (math.isfinite(high_price) and math.isfinite(low_price)):
            continue
        day_mfe = (float(high_price) - entry_price) / risk_r
        day_mae = (float(low_price) - entry_price) / risk_r
        best_mfe = max(best_mfe, day_mfe)
        worst_mae = min(worst_mae, day_mae)

        adverse_first = day_mae <= float(policy.negative_mae_r)
        favorable_first = day_mfe >= float(policy.positive_mfe_r)
        if adverse_first and favorable_first:
            return LABEL_REJECT, "same_bar_adverse_first", float(best_mfe), float(worst_mae)
        if adverse_first and best_mfe < float(policy.reject_confirm_mfe_r):
            return LABEL_REJECT, "mae_before_confirm", float(best_mfe), float(worst_mae)
        if favorable_first and worst_mae > float(policy.negative_mae_r):
            return LABEL_PASS, "mfe_before_mae", float(best_mfe), float(worst_mae)

    if math.isfinite(best_mfe) and best_mfe < float(policy.dead_mfe_r):
        return LABEL_REJECT, "dead_mfe", float(best_mfe), float(worst_mae)
    return LABEL_IGNORE, "ambiguous_path", float(best_mfe), float(worst_mae)


def build_event_label(stock_df: pd.DataFrame, *, event_pos: int, ticker: str, atr_values: np.ndarray, policy: BreakoutQualityLabelPolicy) -> tuple[int, str, float, float, float, float]:
    entry_pos = int(event_pos) + 1
    eval_start = entry_pos + int(policy.evaluate_from_bars_after_entry)
    eval_end = eval_start + int(policy.label_horizon_bars)
    if entry_pos >= len(stock_df) or eval_start >= len(stock_df):
        return LABEL_IGNORE, "insufficient_future", np.nan, np.nan, np.nan, np.nan

    atr = float(atr_values[int(event_pos)]) if int(event_pos) < len(atr_values) else np.nan
    close_d0 = float(stock_df["Close"].iloc[int(event_pos)])
    if not (math.isfinite(atr) and atr > 0.0 and math.isfinite(close_d0) and close_d0 > 0.0):
        return LABEL_IGNORE, "invalid_atr", np.nan, np.nan, np.nan, np.nan

    buy_limit = adjust_long_buy_limit(close_d0 + atr * float(policy.label_atr_buy_tol), ticker=ticker)
    open_d1 = float(stock_df["Open"].iloc[entry_pos])
    high_d1 = float(stock_df["High"].iloc[entry_pos])
    low_d1 = float(stock_df["Low"].iloc[entry_pos])
    close_d1 = float(stock_df["Close"].iloc[entry_pos])
    volume_d1 = float(stock_df["Volume"].iloc[entry_pos])
    y_close = close_d0
    if volume_d1 <= 0.0 or not math.isfinite(open_d1) or not math.isfinite(low_d1):
        return LABEL_IGNORE, "unfilled", buy_limit, np.nan, np.nan, np.nan
    if is_locked_limit_up_bar(open_d1, high_d1, low_d1, close_d1, y_close, ticker=ticker):
        return LABEL_IGNORE, "locked_limit_up", buy_limit, np.nan, np.nan, np.nan
    if low_d1 > buy_limit:
        return LABEL_IGNORE, "unfilled", buy_limit, np.nan, np.nan, np.nan

    entry_price = adjust_long_buy_fill_price(min(open_d1, buy_limit), ticker=ticker)
    stop_price = adjust_long_stop_price(entry_price - atr * float(policy.label_atr_times_init), ticker=ticker)
    risk_r = entry_price - stop_price
    if risk_r <= 0.0:
        return LABEL_IGNORE, "invalid_risk", buy_limit, entry_price, stop_price, risk_r

    path = stock_df.iloc[eval_start:min(eval_end, len(stock_df))]
    if len(path) < int(policy.label_horizon_bars):
        return LABEL_IGNORE, "insufficient_future", buy_limit, entry_price, stop_price, risk_r
    label, reason, mfe_r, mae_r = _label_from_path(
        path["High"].to_numpy(dtype=np.float64, copy=False),
        path["Low"].to_numpy(dtype=np.float64, copy=False),
        entry_price=entry_price,
        risk_r=risk_r,
        policy=policy,
    )
    return label, reason, buy_limit, entry_price, stop_price, risk_r


def build_breakout_quality_dataset_for_frame(
    stock_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    *,
    ticker: str,
    policy: BreakoutQualityLabelPolicy,
) -> BreakoutQualityDataset:
    features = []
    contexts = []
    labels = []
    rows = []
    atr_values = tv_atr(
        stock_df["High"].to_numpy(dtype=np.float64, copy=False),
        stock_df["Low"].to_numpy(dtype=np.float64, copy=False),
        stock_df["Close"].to_numpy(dtype=np.float64, copy=False),
        int(policy.label_atr_len),
    )

    for event in build_candidate_event_positions(stock_df, policy.high_lens()):
        pos = int(event["pos"])
        high_len = int(event["high_len"])
        breakout_level = float(event["breakout_level"])
        feature_payload = build_breakout_quality_feature(
            stock_df,
            benchmark_df,
            event_pos=pos,
            high_len=high_len,
            breakout_level=breakout_level,
            policy=policy,
        )
        if feature_payload is None:
            continue
        label, reason, buy_limit, entry_price, stop_price, risk_r = build_event_label(
            stock_df,
            event_pos=pos,
            ticker=ticker,
            atr_values=atr_values,
            policy=policy,
        )
        seq_features, context = feature_payload
        features.append(seq_features)
        contexts.append(context)
        labels.append(int(label))
        entry_pos = pos + 1
        eval_start_pos = entry_pos + int(policy.evaluate_from_bars_after_entry)
        eval_end_pos = eval_start_pos + int(policy.label_horizon_bars) - 1

        def _format_index_date(index_pos: int) -> str | None:
            if index_pos < 0 or index_pos >= len(stock_df):
                return None
            return pd.Timestamp(stock_df.index[index_pos]).strftime("%Y-%m-%d")

        rows.append({
            "ticker": str(ticker),
            "date": pd.Timestamp(event["date"]).strftime("%Y-%m-%d"),
            "entry_date": _format_index_date(entry_pos),
            "label_eval_start_date": _format_index_date(eval_start_pos),
            "label_eval_end_date": _format_index_date(eval_end_pos),
            "high_len": high_len,
            "breakout_level": breakout_level,
            "label": int(label),
            "label_reason": reason,
            "buy_limit": buy_limit,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "risk_r": risk_r,
        })

    if not features:
        return BreakoutQualityDataset(
            features=np.empty((0, int(policy.feature_window_bars), len(FEATURE_COLUMNS)), dtype=np.float32),
            context=np.empty((0, len(CONTEXT_COLUMNS)), dtype=np.float32),
            labels=np.empty((0,), dtype=np.int64),
            events=pd.DataFrame(columns=[
                "ticker",
                "date",
                "entry_date",
                "label_eval_start_date",
                "label_eval_end_date",
                "high_len",
                "breakout_level",
                "label",
                "label_reason",
            ]),
        )
    return BreakoutQualityDataset(
        features=np.asarray(features, dtype=np.float32),
        context=np.asarray(contexts, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        events=pd.DataFrame(rows),
    )


__all__ = [
    "BreakoutQualityDataset",
    "build_breakout_quality_dataset_for_frame",
    "build_breakout_quality_feature",
    "build_candidate_event_positions",
    "build_event_label",
]
