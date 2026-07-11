"""Raw OHLCV feature and path-based label builders for breakout quality filter."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

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


def _label_from_path(
    high_values: np.ndarray,
    low_values: np.ndarray,
    *,
    anchor_price: float,
    policy: BreakoutQualityLabelPolicy,
) -> tuple[int, str, float, float, float]:
    if not math.isfinite(anchor_price) or anchor_price <= 0.0:
        return LABEL_IGNORE, "invalid_anchor", np.nan, np.nan, np.nan

    highs = np.asarray(high_values, dtype=np.float64)
    lows = np.asarray(low_values, dtype=np.float64)
    if highs.shape != lows.shape or highs.size != int(policy.label_horizon_bars):
        return LABEL_IGNORE, "insufficient_future", np.nan, np.nan, np.nan
    valid = (
        np.isfinite(highs)
        & np.isfinite(lows)
        & (highs > 0.0)
        & (lows > 0.0)
        & (highs >= lows)
    )
    if not bool(np.all(valid)):
        return LABEL_IGNORE, "invalid_future_bar", np.nan, np.nan, np.nan

    upside_returns = highs / anchor_price - 1.0
    downside_returns = lows / anchor_price - 1.0
    max_upside_return = float(np.max(upside_returns))
    max_downside_return = float(np.min(downside_returns))
    pass_barrier_price = anchor_price * (1.0 + float(policy.pass_return_threshold))
    reject_barrier_price = anchor_price * (1.0 + float(policy.reject_return_threshold))

    for bar_offset, (high_price, low_price) in enumerate(
        zip(highs, lows),
        start=1,
    ):
        upside_hit = float(high_price) >= pass_barrier_price
        downside_hit = float(low_price) <= reject_barrier_price
        if upside_hit and downside_hit:
            return (
                LABEL_REJECT,
                "same_bar_adverse_first",
                max_upside_return,
                max_downside_return,
                float(bar_offset),
            )
        if downside_hit:
            return (
                LABEL_REJECT,
                "downside_first",
                max_upside_return,
                max_downside_return,
                float(bar_offset),
            )
        if upside_hit:
            return (
                LABEL_PASS,
                "upside_first",
                max_upside_return,
                max_downside_return,
                float(bar_offset),
            )

    return LABEL_IGNORE, "no_barrier_hit", max_upside_return, max_downside_return, np.nan


def build_event_label(
    stock_df: pd.DataFrame,
    *,
    event_pos: int,
    policy: BreakoutQualityLabelPolicy,
) -> tuple[int, str, float, float, float, float]:
    eval_start = int(event_pos) + 1
    eval_end = eval_start + int(policy.label_horizon_bars)
    if eval_start >= len(stock_df) or eval_end > len(stock_df):
        return LABEL_IGNORE, "insufficient_future", np.nan, np.nan, np.nan, np.nan

    anchor_price = float(stock_df["Close"].iloc[int(event_pos)])
    if not math.isfinite(anchor_price) or anchor_price <= 0.0:
        return LABEL_IGNORE, "invalid_anchor", np.nan, np.nan, np.nan, np.nan

    path = stock_df.iloc[eval_start:eval_end]
    label, reason, max_upside_return, max_downside_return, first_hit_bar = _label_from_path(
        path["High"].to_numpy(dtype=np.float64, copy=False),
        path["Low"].to_numpy(dtype=np.float64, copy=False),
        anchor_price=anchor_price,
        policy=policy,
    )
    return (
        label,
        reason,
        anchor_price,
        max_upside_return,
        max_downside_return,
        first_hit_bar,
    )


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
        (
            label,
            reason,
            anchor_price,
            max_upside_return,
            max_downside_return,
            first_hit_bar,
        ) = build_event_label(
            stock_df,
            event_pos=pos,
            policy=policy,
        )
        seq_features, context = feature_payload
        features.append(seq_features)
        contexts.append(context)
        labels.append(int(label))
        eval_start_pos = pos + 1
        eval_end_pos = eval_start_pos + int(policy.label_horizon_bars) - 1

        def _format_index_date(index_pos: int) -> str | None:
            if index_pos < 0 or index_pos >= len(stock_df):
                return None
            return pd.Timestamp(stock_df.index[index_pos]).strftime("%Y-%m-%d")

        rows.append({
            "ticker": str(ticker),
            "date": pd.Timestamp(event["date"]).strftime("%Y-%m-%d"),
            "label_eval_start_date": _format_index_date(eval_start_pos),
            "label_eval_end_date": _format_index_date(eval_end_pos),
            "high_len": high_len,
            "breakout_level": breakout_level,
            "label": int(label),
            "label_reason": reason,
            "anchor_price": anchor_price,
            "pass_barrier_price": (
                anchor_price * (1.0 + float(policy.pass_return_threshold))
                if math.isfinite(anchor_price)
                else np.nan
            ),
            "reject_barrier_price": (
                anchor_price * (1.0 + float(policy.reject_return_threshold))
                if math.isfinite(anchor_price)
                else np.nan
            ),
            "max_upside_return": max_upside_return,
            "max_downside_return": max_downside_return,
            "first_hit_bar": first_hit_bar,
        })

    if not features:
        return BreakoutQualityDataset(
            features=np.empty((0, int(policy.feature_window_bars), len(FEATURE_COLUMNS)), dtype=np.float32),
            context=np.empty((0, len(CONTEXT_COLUMNS)), dtype=np.float32),
            labels=np.empty((0,), dtype=np.int64),
            events=pd.DataFrame(columns=[
                "ticker",
                "date",
                "label_eval_start_date",
                "label_eval_end_date",
                "high_len",
                "breakout_level",
                "label",
                "label_reason",
                "anchor_price",
                "pass_barrier_price",
                "reject_barrier_price",
                "max_upside_return",
                "max_downside_return",
                "first_hit_bar",
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
