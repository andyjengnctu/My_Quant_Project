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
    "decision_mfe_return",
    "decision_mae_return",
    "decision_reward_risk_ratio",
    "first_hit_bar",
)


@dataclass(frozen=True)
class BreakoutQualityInferenceDataset:
    feature_bank: np.ndarray
    context: np.ndarray
    event_group_index: np.ndarray
    events: pd.DataFrame
    unavailable_events: pd.DataFrame


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
    decision_mfe_return: float
    decision_mae_return: float
    decision_reward_risk_ratio: float
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
    values = window[["Open", "High", "Low", "Close", "Volume"]].to_numpy(
        dtype=np.float64,
        copy=False,
    )
    return normalize_ohlcv_array_window(values, anchor_close)


def normalize_ohlcv_array_window(values: np.ndarray, anchor_close: float) -> np.ndarray:
    """Normalize one canonical OHLCV window without requiring a DataFrame.

    This is the single numerical implementation shared by the historical
    DataFrame feature builder and the MR-13A lazy stock-day materializer.
    """

    if not math.isfinite(anchor_close) or anchor_close <= 0.0:
        raise ValueError("anchor_close 必須是有限正數")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 5:
        raise ValueError("OHLCV array window必須是 [bars, 5]")
    return normalize_ohlcv_array_windows(
        array[None, :, :],
        np.asarray([anchor_close], dtype=np.float64),
    )[0]


def normalize_ohlcv_array_windows(
    values: np.ndarray,
    anchor_closes: np.ndarray,
) -> np.ndarray:
    """Vectorized canonical OHLCV normalization for a batch of windows.

    The numerical definition is identical to :func:`normalize_ohlcv_array_window`:
    prices are expressed relative to each row's anchor close and volume is
    log1p/median/IQR normalized within that same window.  The batched form exists
    only to remove Python-per-stock normalization overhead from the lazy daily
    feature provider; it does not alter feature membership, ordering, or values.
    """

    array = np.asarray(values, dtype=np.float64)
    anchors = np.asarray(anchor_closes, dtype=np.float64).reshape(-1)
    if array.ndim != 3 or array.shape[2] != 5:
        raise ValueError("OHLCV array windows必須是 [batch, bars, 5]")
    if len(array) != len(anchors):
        raise ValueError("OHLCV array windows與anchor_closes長度不一致")
    if bool(np.any(~np.isfinite(anchors))) or bool(np.any(anchors <= 0.0)):
        raise ValueError("anchor_close 必須是有限正數")
    if len(array) == 0:
        return np.empty((0, int(array.shape[1]), 5), dtype=np.float32)

    output = np.empty(array.shape, dtype=np.float32)
    output[:, :, :4] = array[:, :, :4] / anchors[:, None, None] - 1.0

    volume = np.log1p(np.maximum(array[:, :, 4], 0.0))
    finite_rows = np.all(np.isfinite(volume), axis=1)
    if bool(np.any(finite_rows)):
        finite_volume = volume[finite_rows]
        median = np.median(finite_volume, axis=1)
        q75, q25 = np.percentile(finite_volume, [75, 25], axis=1)
        iqr = q75 - q25
        iqr = np.where(np.isfinite(iqr) & (iqr > 0.0), iqr, 1.0)
        output[finite_rows, :, 4] = (
            (finite_volume - median[:, None]) / iqr[:, None]
        ).astype(np.float32)

    # Historical generic helpers allow non-finite volume rows.  Keep that exact
    # fallback semantics while leaving the canonical sanitized-data hot path
    # fully vectorized.
    for row_index in np.flatnonzero(~finite_rows):
        output[int(row_index), :, 4] = _normalize_volume_window(
            array[int(row_index), :, 4]
        )
    return output


def _build_breakout_quality_sequence_feature_with_reason(
    stock_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    *,
    event_pos: int,
    policy: BreakoutQualityLabelPolicy,
) -> tuple[np.ndarray | None, str | None]:
    """Build one shared sequence and expose why a formal candidate cannot be scored."""

    feature_window = int(policy.feature_window_bars)
    start_pos = int(event_pos) - feature_window + 1
    if start_pos < 0:
        return None, "insufficient_stock_history"

    stock_window = stock_df.iloc[start_pos:int(event_pos) + 1]
    if len(stock_window) != feature_window:
        return None, "insufficient_stock_history"
    event_date = stock_df.index[int(event_pos)]
    if event_date not in benchmark_df.index:
        return None, "benchmark_date_missing"
    benchmark_location = benchmark_df.index.get_loc(event_date)
    if not isinstance(benchmark_location, (int, np.integer)):
        return None, "benchmark_date_ambiguous"
    benchmark_pos = int(benchmark_location)
    benchmark_start = benchmark_pos - feature_window + 1
    if benchmark_start < 0:
        return None, "insufficient_benchmark_history"
    benchmark_window = benchmark_df.iloc[benchmark_start:benchmark_pos + 1]
    if len(benchmark_window) != feature_window:
        return None, "insufficient_benchmark_history"

    close_d0 = float(stock_df["Close"].iloc[int(event_pos)])
    benchmark_close_d0 = float(benchmark_df["Close"].iloc[benchmark_pos])
    if not math.isfinite(close_d0) or close_d0 <= 0.0:
        return None, "invalid_stock_anchor"
    if not math.isfinite(benchmark_close_d0) or benchmark_close_d0 <= 0.0:
        return None, "invalid_benchmark_anchor"

    stock_features = _normalize_ohlcv_window(stock_window, close_d0)
    benchmark_features = _normalize_ohlcv_window(benchmark_window, benchmark_close_d0)
    seq_features = np.column_stack([stock_features, benchmark_features]).astype(np.float32)
    if not np.isfinite(seq_features).all():
        return None, "nonfinite_sequence_feature"
    return seq_features, None


def build_breakout_quality_sequence_feature(
    stock_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    *,
    event_pos: int,
    policy: BreakoutQualityLabelPolicy,
) -> np.ndarray | None:
    """Build the ticker/date sequence once; it is shared by all high_len events on that date."""

    sequence, _reason = _build_breakout_quality_sequence_feature_with_reason(
        stock_df,
        benchmark_df,
        event_pos=event_pos,
        policy=policy,
    )
    return sequence


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



def build_breakout_quality_inference_dataset_for_frame(
    stock_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    *,
    ticker: str,
    policy: BreakoutQualityLabelPolicy,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    require_context: bool = True,
) -> BreakoutQualityInferenceDataset:
    """Build the current runtime candidate universe without using future labels.

    Every formal breakout event is represented either by a model input or by an
    explicit unavailable row.  Exporters may conservatively map unavailable rows
    to REJECT, while an undocumented missing score remains a contract failure.
    """

    candidate_events = build_candidate_event_positions(stock_df, policy.high_lens())
    start_ts = None if start_date is None else pd.Timestamp(start_date).normalize()
    end_ts = None if end_date is None else pd.Timestamp(end_date).normalize()
    if start_ts is not None or end_ts is not None:
        candidate_events = [
            event
            for event in candidate_events
            if (start_ts is None or pd.Timestamp(event["date"]).normalize() >= start_ts)
            and (end_ts is None or pd.Timestamp(event["date"]).normalize() <= end_ts)
        ]
    if not candidate_events:
        return BreakoutQualityInferenceDataset(
            feature_bank=np.empty(
                (0, int(policy.feature_window_bars), len(FEATURE_COLUMNS)),
                dtype=np.float32,
            ),
            context=np.empty((0, len(CONTEXT_COLUMNS)), dtype=np.float32),
            event_group_index=np.empty((0,), dtype=np.int32),
            events=pd.DataFrame(columns=["ticker", "date", "high_len"]),
            unavailable_events=pd.DataFrame(
                columns=["ticker", "date", "high_len", "reason"]
            ),
        )

    events_by_pos: dict[int, list[dict]] = {}
    for event in candidate_events:
        events_by_pos.setdefault(int(event["pos"]), []).append(event)

    feature_bank: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    event_group_index: list[int] = []
    rows: list[dict] = []
    unavailable_rows: list[dict] = []

    for pos, grouped_events in events_by_pos.items():
        sequence, unavailable_reason = _build_breakout_quality_sequence_feature_with_reason(
            stock_df,
            benchmark_df,
            event_pos=pos,
            policy=policy,
        )
        if sequence is None:
            for event in grouped_events:
                unavailable_rows.append(
                    {
                        "ticker": str(ticker),
                        "date": pd.Timestamp(event["date"]).strftime("%Y-%m-%d"),
                        "high_len": int(event["high_len"]),
                        "reason": str(unavailable_reason or "sequence_feature_unavailable"),
                    }
                )
            continue

        local_group_index = len(feature_bank)
        group_rows: list[tuple[dict, np.ndarray]] = []
        for event in grouped_events:
            context = build_breakout_quality_context(
                stock_df,
                event_pos=pos,
                high_len=int(event["high_len"]),
                breakout_level=float(event["breakout_level"]),
                policy=policy,
            )
            if context is None and require_context:
                unavailable_rows.append(
                    {
                        "ticker": str(ticker),
                        "date": pd.Timestamp(event["date"]).strftime("%Y-%m-%d"),
                        "high_len": int(event["high_len"]),
                        "reason": "context_feature_unavailable",
                    }
                )
                continue
            if context is None:
                context = np.zeros((len(CONTEXT_COLUMNS),), dtype=np.float32)
            group_rows.append((event, context))

        if not group_rows:
            continue
        feature_bank.append(sequence)
        for event, context in group_rows:
            contexts.append(context)
            event_group_index.append(local_group_index)
            rows.append(
                {
                    "ticker": str(ticker),
                    "date": pd.Timestamp(event["date"]).strftime("%Y-%m-%d"),
                    "high_len": int(event["high_len"]),
                }
            )

    feature_array = (
        np.asarray(feature_bank, dtype=np.float32)
        if feature_bank
        else np.empty(
            (0, int(policy.feature_window_bars), len(FEATURE_COLUMNS)),
            dtype=np.float32,
        )
    )
    return BreakoutQualityInferenceDataset(
        feature_bank=feature_array,
        context=np.asarray(contexts, dtype=np.float32).reshape(-1, len(CONTEXT_COLUMNS)),
        event_group_index=np.asarray(event_group_index, dtype=np.int32),
        events=pd.DataFrame(rows, columns=["ticker", "date", "high_len"]),
        unavailable_events=pd.DataFrame(
            unavailable_rows,
            columns=["ticker", "date", "high_len", "reason"],
        ),
    )

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
    invalid_metrics = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    if int(available_bars) < horizon:
        return BreakoutQualityLabelResult(LABEL_INVALID, "insufficient_future", *invalid_metrics)
    if not math.isfinite(anchor_price) or anchor_price <= 0.0:
        return BreakoutQualityLabelResult(LABEL_INVALID, "invalid_anchor", *invalid_metrics)

    highs = np.asarray(high_prices[:horizon], dtype=np.float64)
    lows = np.asarray(low_prices[:horizon], dtype=np.float64)
    if highs.shape != lows.shape or highs.size != horizon:
        return BreakoutQualityLabelResult(LABEL_INVALID, "insufficient_future", *invalid_metrics)
    valid = np.isfinite(highs) & np.isfinite(lows) & (highs > 0.0) & (lows > 0.0) & (highs >= lows)
    if not bool(np.all(valid)):
        return BreakoutQualityLabelResult(LABEL_INVALID, "invalid_future_bar", *invalid_metrics)

    upside_returns = highs / anchor_price - 1.0
    downside_returns = lows / anchor_price - 1.0
    max_upside_return = float(np.max(upside_returns))
    max_downside_return = float(np.min(downside_returns))
    min_mfe_price = anchor_price * (1.0 + float(policy.min_mfe_return))
    max_adverse_price = anchor_price * (1.0 + float(policy.max_adverse_return))
    min_ratio = float(policy.min_reward_risk_ratio)

    running_high = float(anchor_price)
    running_low = float(anchor_price)
    for bar_offset, (high_price, low_price) in enumerate(zip(highs, lows), start=1):
        candidate_high = max(running_high, float(high_price))
        candidate_low = min(running_low, float(low_price))
        decision_mfe = float(candidate_high / anchor_price - 1.0)
        decision_downside = float(candidate_low / anchor_price - 1.0)
        decision_mae = max(0.0, -decision_downside)
        decision_ratio = math.inf if decision_mae == 0.0 else float(decision_mfe / decision_mae)

        risk_hit = candidate_low <= max_adverse_price
        mfe_hit = candidate_high > min_mfe_price
        favorable_move = max(0.0, candidate_high - anchor_price)
        adverse_move = max(0.0, anchor_price - candidate_low)
        ratio_hit = adverse_move == 0.0 or favorable_move > min_ratio * adverse_move
        opportunity_hit = mfe_hit and ratio_hit

        # Daily bars do not reveal whether High or Low happened first. When the same
        # bar can both pass and breach the risk limit, use the conservative adverse-first rule.
        if risk_hit and opportunity_hit:
            return BreakoutQualityLabelResult(
                LABEL_REJECT,
                "same_bar_adverse_first",
                max_upside_return,
                max_downside_return,
                decision_mfe,
                decision_mae,
                decision_ratio,
                float(bar_offset),
            )
        if risk_hit:
            return BreakoutQualityLabelResult(
                LABEL_REJECT,
                "downside_first",
                max_upside_return,
                max_downside_return,
                decision_mfe,
                decision_mae,
                decision_ratio,
                float(bar_offset),
            )
        if opportunity_hit:
            return BreakoutQualityLabelResult(
                LABEL_PASS,
                "risk_adjusted_opportunity",
                max_upside_return,
                max_downside_return,
                decision_mfe,
                decision_mae,
                decision_ratio,
                float(bar_offset),
            )

        running_high = candidate_high
        running_low = candidate_low

    final_mfe = float(running_high / anchor_price - 1.0)
    final_downside = float(running_low / anchor_price - 1.0)
    final_mae = max(0.0, -final_downside)
    final_ratio = math.inf if final_mae == 0.0 else float(final_mfe / final_mae)
    return BreakoutQualityLabelResult(
        LABEL_REJECT,
        "no_risk_adjusted_opportunity",
        max_upside_return,
        max_downside_return,
        final_mfe,
        final_mae,
        final_ratio,
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
                    event_anchor_price * (1.0 + float(policy.min_mfe_return))
                    if math.isfinite(event_anchor_price)
                    else np.nan
                ),
                "reject_barrier_price": (
                    event_anchor_price * (1.0 + float(policy.max_adverse_return))
                    if math.isfinite(event_anchor_price)
                    else np.nan
                ),
                "max_upside_return": label_result.max_upside_return,
                "max_downside_return": label_result.max_downside_return,
                "decision_mfe_return": label_result.decision_mfe_return,
                "decision_mae_return": label_result.decision_mae_return,
                "decision_reward_risk_ratio": label_result.decision_reward_risk_ratio,
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
    "BreakoutQualityInferenceDataset",
    "BreakoutQualityLabelResult",
    "build_breakout_quality_context",
    "build_breakout_quality_dataset_for_frame",
    "build_breakout_quality_feature",
    "build_breakout_quality_inference_dataset_for_frame",
    "build_breakout_quality_sequence_feature",
    "build_candidate_event_positions",
    "build_event_label",
    "build_future_price_path",
    "label_from_cached_path",
]
