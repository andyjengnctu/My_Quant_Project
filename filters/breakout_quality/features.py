"""Raw OHLCV feature-bank and path-based label builders for breakout quality filter."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from filters.breakout_quality.contract import (
    CONTEXT_COLUMNS,
    FEATURE_COLUMNS,
    LABEL_INVALID,
    LABEL_PASS,
    LABEL_REJECT,
    BreakoutQualityLabelPolicy,
)


EVENT_COLUMNS = (
    "ticker",
    "date",
    "label_eval_start_date",
    "label_eval_end_date",
    "high_len",
    "breakout_level",
    "group_index",
    "label",
    "label_reason",
    "anchor_price",
    "pass_barrier_price",
    "reject_barrier_price",
    "max_upside_return",
    "max_downside_return",
    "first_hit_bar",
)


@dataclass(frozen=True)
class BreakoutQualityDataset:
    feature_bank: np.ndarray
    context: np.ndarray
    labels: np.ndarray
    event_group_index: np.ndarray
    group_anchor_prices: np.ndarray
    future_high_prices: np.ndarray
    future_low_prices: np.ndarray
    future_available_bars: np.ndarray
    future_date_ordinals: np.ndarray
    events: pd.DataFrame


@dataclass(frozen=True)
class BreakoutQualityLabelResult:
    label: int
    reason: str
    max_upside_return: float
    max_downside_return: float
    first_hit_bar: float


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


def _normalize_ohlcv_window(window: pd.DataFrame, anchor_close: float) -> np.ndarray:
    if not math.isfinite(anchor_close) or anchor_close <= 0.0:
        raise ValueError("anchor_close 必須是有限正數")
    open_norm = window["Open"].to_numpy(dtype=np.float64, copy=False) / anchor_close - 1.0
    high_norm = window["High"].to_numpy(dtype=np.float64, copy=False) / anchor_close - 1.0
    low_norm = window["Low"].to_numpy(dtype=np.float64, copy=False) / anchor_close - 1.0
    close_norm = window["Close"].to_numpy(dtype=np.float64, copy=False) / anchor_close - 1.0
    volume_norm = _normalize_volume_window(window["Volume"].to_numpy(dtype=np.float64, copy=False))
    return np.column_stack([open_norm, high_norm, low_norm, close_norm, volume_norm]).astype(np.float32)


def build_breakout_quality_sequence_feature(
    stock_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    *,
    event_pos: int,
    policy: BreakoutQualityLabelPolicy,
) -> np.ndarray | None:
    """Build the ticker/date sequence once; it is shared by all high_len events on that date."""

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
    benchmark_location = benchmark_df.index.get_loc(event_date)
    if not isinstance(benchmark_location, (int, np.integer)):
        return None
    benchmark_pos = int(benchmark_location)
    benchmark_start = benchmark_pos - feature_window + 1
    if benchmark_start < 0:
        return None
    benchmark_window = benchmark_df.iloc[benchmark_start:benchmark_pos + 1]
    if len(benchmark_window) != feature_window:
        return None

    close_d0 = float(stock_df["Close"].iloc[int(event_pos)])
    benchmark_close_d0 = float(benchmark_df["Close"].iloc[benchmark_pos])
    if not math.isfinite(close_d0) or close_d0 <= 0.0:
        return None
    if not math.isfinite(benchmark_close_d0) or benchmark_close_d0 <= 0.0:
        return None

    stock_features = _normalize_ohlcv_window(stock_window, close_d0)
    benchmark_features = _normalize_ohlcv_window(benchmark_window, benchmark_close_d0)
    seq_features = np.column_stack([stock_features, benchmark_features]).astype(np.float32)
    if not np.isfinite(seq_features).all():
        return None
    return seq_features


def build_breakout_quality_context(
    stock_df: pd.DataFrame,
    *,
    event_pos: int,
    high_len: int,
    breakout_level: float,
    policy: BreakoutQualityLabelPolicy,
) -> np.ndarray | None:
    close_d0 = float(stock_df["Close"].iloc[int(event_pos)])
    high_d0 = float(stock_df["High"].iloc[int(event_pos)])
    if (
        not math.isfinite(close_d0)
        or close_d0 <= 0.0
        or not math.isfinite(high_d0)
        or high_d0 <= 0.0
        or not math.isfinite(breakout_level)
        or breakout_level <= 0.0
    ):
        return None

    high_len_span = max(1, int(policy.high_len_max) - int(policy.high_len_min))
    context = np.asarray(
        [
            (int(high_len) - int(policy.high_len_min)) / high_len_span,
            float(breakout_level) / close_d0 - 1.0,
            close_d0 / float(breakout_level) - 1.0,
            high_d0 / float(breakout_level) - 1.0,
        ],
        dtype=np.float32,
    )
    return context if np.isfinite(context).all() else None


def build_breakout_quality_feature(
    stock_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    *,
    event_pos: int,
    high_len: int,
    breakout_level: float,
    policy: BreakoutQualityLabelPolicy,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Backward-compatible combined feature builder used by isolated checks."""

    sequence = build_breakout_quality_sequence_feature(
        stock_df,
        benchmark_df,
        event_pos=event_pos,
        policy=policy,
    )
    if sequence is None:
        return None
    context = build_breakout_quality_context(
        stock_df,
        event_pos=event_pos,
        high_len=high_len,
        breakout_level=breakout_level,
        policy=policy,
    )
    if context is None:
        return None
    return sequence, context


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
        rolling_high = (
            pd.Series(high)
            .shift(1)
            .rolling(high_len_int, min_periods=high_len_int)
            .max()
            .to_numpy(dtype=np.float64)
        )
        prev_high = np.empty_like(rolling_high)
        prev_high[0] = rolling_high[0] if rolling_high.size else np.nan
        if rolling_high.size > 1:
            prev_high[1:] = rolling_high[:-1]
        crossover = (close > rolling_high) & (prev_close <= prev_high)
        if crossover.size:
            crossover[0] = False
        for pos in np.flatnonzero(crossover):
            level = float(rolling_high[int(pos)])
            if math.isfinite(level) and level > 0.0:
                events.append(
                    {
                        "pos": int(pos),
                        "date": stock_df.index[int(pos)],
                        "high_len": high_len_int,
                        "breakout_level": level,
                    }
                )
    return events


def build_future_price_path(
    stock_df: pd.DataFrame,
    *,
    event_pos: int,
    cache_bars: int,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, int]:
    """Cache future High/Low prices once per ticker/date for exact fast relabeling."""

    cache_size = int(cache_bars)
    if cache_size < 1:
        raise ValueError("cache_bars 必須 >= 1")
    high_prices = np.full((cache_size,), np.nan, dtype=np.float64)
    low_prices = np.full((cache_size,), np.nan, dtype=np.float64)
    date_ordinals = np.full((cache_size,), -1, dtype=np.int32)

    anchor_price = float(stock_df["Close"].iloc[int(event_pos)])
    if not math.isfinite(anchor_price) or anchor_price <= 0.0:
        return np.nan, high_prices, low_prices, date_ordinals, 0

    eval_start = int(event_pos) + 1
    available_bars = max(0, min(cache_size, len(stock_df) - eval_start))
    if available_bars == 0:
        return anchor_price, high_prices, low_prices, date_ordinals, 0

    path = stock_df.iloc[eval_start:eval_start + available_bars]
    highs = path["High"].to_numpy(dtype=np.float64, copy=False)
    lows = path["Low"].to_numpy(dtype=np.float64, copy=False)
    high_prices[:available_bars] = highs
    low_prices[:available_bars] = lows
    dates = pd.to_datetime(path.index, errors="raise").to_numpy(dtype="datetime64[D]")
    date_ordinals[:available_bars] = dates.astype(np.int64).astype(np.int32)
    return anchor_price, high_prices, low_prices, date_ordinals, int(available_bars)


def label_from_cached_path(
    high_prices: np.ndarray,
    low_prices: np.ndarray,
    *,
    anchor_price: float,
    available_bars: int,
    policy: BreakoutQualityLabelPolicy,
) -> BreakoutQualityLabelResult:
    horizon = int(policy.label_horizon_bars)
    if int(available_bars) < horizon:
        return BreakoutQualityLabelResult(LABEL_INVALID, "insufficient_future", np.nan, np.nan, np.nan)
    if not math.isfinite(anchor_price) or anchor_price <= 0.0:
        return BreakoutQualityLabelResult(LABEL_INVALID, "invalid_anchor", np.nan, np.nan, np.nan)

    highs = np.asarray(high_prices[:horizon], dtype=np.float64)
    lows = np.asarray(low_prices[:horizon], dtype=np.float64)
    if highs.shape != lows.shape or highs.size != horizon:
        return BreakoutQualityLabelResult(LABEL_INVALID, "insufficient_future", np.nan, np.nan, np.nan)
    valid = np.isfinite(highs) & np.isfinite(lows) & (highs > 0.0) & (lows > 0.0) & (highs >= lows)
    if not bool(np.all(valid)):
        return BreakoutQualityLabelResult(LABEL_INVALID, "invalid_future_bar", np.nan, np.nan, np.nan)

    upside_returns = highs / anchor_price - 1.0
    downside_returns = lows / anchor_price - 1.0
    max_upside_return = float(np.max(upside_returns))
    max_downside_return = float(np.min(downside_returns))
    pass_barrier_price = anchor_price * (1.0 + float(policy.pass_return_threshold))
    reject_barrier_price = anchor_price * (1.0 + float(policy.reject_return_threshold))

    for bar_offset, (high_price, low_price) in enumerate(zip(highs, lows), start=1):
        upside_hit = float(high_price) >= pass_barrier_price
        downside_hit = float(low_price) <= reject_barrier_price
        if upside_hit and downside_hit:
            return BreakoutQualityLabelResult(
                LABEL_REJECT,
                "same_bar_adverse_first",
                max_upside_return,
                max_downside_return,
                float(bar_offset),
            )
        if downside_hit:
            return BreakoutQualityLabelResult(
                LABEL_REJECT,
                "downside_first",
                max_upside_return,
                max_downside_return,
                float(bar_offset),
            )
        if upside_hit:
            return BreakoutQualityLabelResult(
                LABEL_PASS,
                "upside_first",
                max_upside_return,
                max_downside_return,
                float(bar_offset),
            )

    return BreakoutQualityLabelResult(
        LABEL_REJECT,
        "no_upside_target",
        max_upside_return,
        max_downside_return,
        np.nan,
    )


def build_event_label(
    stock_df: pd.DataFrame,
    *,
    event_pos: int,
    policy: BreakoutQualityLabelPolicy,
) -> tuple[int, str, float, float, float, float]:
    anchor_price, high_prices, low_prices, _date_ordinals, available_bars = build_future_price_path(
        stock_df,
        event_pos=event_pos,
        cache_bars=int(policy.label_horizon_bars),
    )
    if not math.isfinite(anchor_price) or anchor_price <= 0.0:
        return LABEL_INVALID, "invalid_anchor", np.nan, np.nan, np.nan, np.nan
    result = label_from_cached_path(
        high_prices,
        low_prices,
        anchor_price=anchor_price,
        available_bars=available_bars,
        policy=policy,
    )
    event_anchor_price = np.nan if result.reason == "insufficient_future" else anchor_price
    return (
        int(result.label),
        result.reason,
        event_anchor_price,
        result.max_upside_return,
        result.max_downside_return,
        result.first_hit_bar,
    )


def _format_index_date(stock_df: pd.DataFrame, index_pos: int) -> str | None:
    if index_pos < 0 or index_pos >= len(stock_df):
        return None
    return pd.Timestamp(stock_df.index[index_pos]).strftime("%Y-%m-%d")


def _empty_dataset(policy: BreakoutQualityLabelPolicy) -> BreakoutQualityDataset:
    return BreakoutQualityDataset(
        feature_bank=np.empty(
            (0, int(policy.feature_window_bars), len(FEATURE_COLUMNS)),
            dtype=np.float32,
        ),
        context=np.empty((0, len(CONTEXT_COLUMNS)), dtype=np.float32),
        labels=np.empty((0,), dtype=np.int8),
        event_group_index=np.empty((0,), dtype=np.int32),
        group_anchor_prices=np.empty((0,), dtype=np.float64),
        future_high_prices=np.empty(
            (0, int(policy.label_path_cache_bars)),
            dtype=np.float64,
        ),
        future_low_prices=np.empty(
            (0, int(policy.label_path_cache_bars)),
            dtype=np.float64,
        ),
        future_available_bars=np.empty((0,), dtype=np.int16),
        future_date_ordinals=np.empty((0, int(policy.label_path_cache_bars)), dtype=np.int32),
        events=pd.DataFrame(columns=list(EVENT_COLUMNS)),
    )


def build_breakout_quality_dataset_for_frame(
    stock_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    *,
    ticker: str,
    policy: BreakoutQualityLabelPolicy,
) -> BreakoutQualityDataset:
    candidate_events = build_candidate_event_positions(stock_df, policy.high_lens())
    if not candidate_events:
        return _empty_dataset(policy)

    ordered_positions: list[int] = []
    first_event_by_pos: dict[int, dict] = {}
    for event in candidate_events:
        pos = int(event["pos"])
        if pos not in first_event_by_pos:
            first_event_by_pos[pos] = event
            ordered_positions.append(pos)

    feature_bank: list[np.ndarray] = []
    group_anchor_prices: list[float] = []
    future_high_prices: list[np.ndarray] = []
    future_low_prices: list[np.ndarray] = []
    future_available_bars: list[int] = []
    future_date_ordinals: list[np.ndarray] = []
    group_payload_by_pos: dict[int, tuple[int, BreakoutQualityLabelResult, float]] = {}

    for pos in ordered_positions:
        sequence = build_breakout_quality_sequence_feature(
            stock_df,
            benchmark_df,
            event_pos=pos,
            policy=policy,
        )
        if sequence is None:
            continue
        first_event = first_event_by_pos[pos]
        if build_breakout_quality_context(
            stock_df,
            event_pos=pos,
            high_len=int(first_event["high_len"]),
            breakout_level=float(first_event["breakout_level"]),
            policy=policy,
        ) is None:
            continue

        anchor_price, high_path, low_path, date_ordinals, available_bars = build_future_price_path(
            stock_df,
            event_pos=pos,
            cache_bars=int(policy.label_path_cache_bars),
        )
        if not math.isfinite(anchor_price) or anchor_price <= 0.0:
            label_result = BreakoutQualityLabelResult(
                LABEL_INVALID,
                "invalid_anchor",
                np.nan,
                np.nan,
                np.nan,
            )
        else:
            label_result = label_from_cached_path(
                high_path,
                low_path,
                anchor_price=anchor_price,
                available_bars=available_bars,
                policy=policy,
            )

        local_group_index = len(feature_bank)
        feature_bank.append(sequence)
        group_anchor_prices.append(anchor_price)
        future_high_prices.append(high_path)
        future_low_prices.append(low_path)
        future_available_bars.append(int(available_bars))
        future_date_ordinals.append(date_ordinals)
        group_payload_by_pos[pos] = (local_group_index, label_result, anchor_price)

    if not feature_bank:
        return _empty_dataset(policy)

    contexts: list[np.ndarray] = []
    labels: list[int] = []
    event_group_index: list[int] = []
    rows: list[dict] = []
    for event in candidate_events:
        pos = int(event["pos"])
        group_payload = group_payload_by_pos.get(pos)
        if group_payload is None:
            continue
        local_group_index, label_result, anchor_price = group_payload
        high_len = int(event["high_len"])
        breakout_level = float(event["breakout_level"])
        context = build_breakout_quality_context(
            stock_df,
            event_pos=pos,
            high_len=high_len,
            breakout_level=breakout_level,
            policy=policy,
        )
        if context is None:
            continue

        contexts.append(context)
        labels.append(int(label_result.label))
        event_group_index.append(local_group_index)
        eval_start_pos = pos + 1
        eval_end_pos = eval_start_pos + int(policy.label_horizon_bars) - 1
        event_anchor_price = (
            np.nan if label_result.reason == "insufficient_future" else anchor_price
        )
        rows.append(
            {
                "ticker": str(ticker),
                "date": pd.Timestamp(event["date"]).strftime("%Y-%m-%d"),
                "label_eval_start_date": _format_index_date(stock_df, eval_start_pos),
                "label_eval_end_date": _format_index_date(stock_df, eval_end_pos),
                "high_len": high_len,
                "breakout_level": breakout_level,
                "group_index": local_group_index,
                "label": int(label_result.label),
                "label_reason": label_result.reason,
                "anchor_price": event_anchor_price,
                "pass_barrier_price": (
                    event_anchor_price * (1.0 + float(policy.pass_return_threshold))
                    if math.isfinite(event_anchor_price)
                    else np.nan
                ),
                "reject_barrier_price": (
                    event_anchor_price * (1.0 + float(policy.reject_return_threshold))
                    if math.isfinite(event_anchor_price)
                    else np.nan
                ),
                "max_upside_return": label_result.max_upside_return,
                "max_downside_return": label_result.max_downside_return,
                "first_hit_bar": label_result.first_hit_bar,
            }
        )

    return BreakoutQualityDataset(
        feature_bank=np.asarray(feature_bank, dtype=np.float32),
        context=np.asarray(contexts, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int8),
        event_group_index=np.asarray(event_group_index, dtype=np.int32),
        group_anchor_prices=np.asarray(group_anchor_prices, dtype=np.float64),
        future_high_prices=np.asarray(future_high_prices, dtype=np.float64),
        future_low_prices=np.asarray(future_low_prices, dtype=np.float64),
        future_available_bars=np.asarray(future_available_bars, dtype=np.int16),
        future_date_ordinals=np.asarray(future_date_ordinals, dtype=np.int32),
        events=pd.DataFrame(rows, columns=list(EVENT_COLUMNS)),
    )


__all__ = [
    "EVENT_COLUMNS",
    "BreakoutQualityDataset",
    "BreakoutQualityLabelResult",
    "build_breakout_quality_context",
    "build_breakout_quality_dataset_for_frame",
    "build_breakout_quality_feature",
    "build_breakout_quality_sequence_feature",
    "build_candidate_event_positions",
    "build_event_label",
    "build_future_price_path",
    "label_from_cached_path",
]
