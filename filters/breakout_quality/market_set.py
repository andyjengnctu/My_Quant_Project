"""Point-in-time full-market inputs for the learned market-set architecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from config.breakout_quality import (
    BREAKOUT_QUALITY_MARKET_SET_BASE_FEATURES,
    BREAKOUT_QUALITY_MARKET_SET_HISTORY_BARS,
    BREAKOUT_QUALITY_MARKET_SET_MAX_STOCKS,
    BREAKOUT_QUALITY_MARKET_SET_MAX_DATES_PER_BATCH,
    BREAKOUT_QUALITY_MARKET_SET_MIN_VALID_HISTORY_RATIO,
)

MARKET_SET_STORAGE_SCHEMA_VERSION = 1
MARKET_SET_STORAGE_FORMAT = "point_in_time_daily_market_bank_v1"
MARKET_SET_FEATURE_COLUMNS = tuple(str(value) for value in BREAKOUT_QUALITY_MARKET_SET_BASE_FEATURES)

@dataclass(frozen=True)
class MarketSetBatch:
    sequences: np.ndarray
    history_mask: np.ndarray
    valid_stock_mask: np.ndarray
    event_to_market: np.ndarray

class IndexedMarketSetBank:
    """Daily market bank with group-date lookup and on-demand history windows."""

    def __init__(
        self,
        daily_features: np.ndarray,
        daily_valid_mask: np.ndarray,
        market_date_ordinals: np.ndarray,
        group_market_date_index: np.ndarray,
        *,
        history_bars: int = BREAKOUT_QUALITY_MARKET_SET_HISTORY_BARS,
        min_valid_history_ratio: float = BREAKOUT_QUALITY_MARKET_SET_MIN_VALID_HISTORY_RATIO,
        max_stocks: int = BREAKOUT_QUALITY_MARKET_SET_MAX_STOCKS,
        max_dates_per_batch: int = BREAKOUT_QUALITY_MARKET_SET_MAX_DATES_PER_BATCH,
    ) -> None:
        if daily_features.ndim != 3:
            raise ValueError(f"market daily_features 必須是 3D: {daily_features.shape}")
        if daily_valid_mask.ndim != 2:
            raise ValueError(f"market daily_valid_mask 必須是 2D: {daily_valid_mask.shape}")
        if daily_features.shape[:2] != daily_valid_mask.shape:
            raise ValueError("market daily feature/mask shape 不一致")
        if market_date_ordinals.ndim != 1 or len(market_date_ordinals) != len(daily_features):
            raise ValueError("market date ordinals shape 不一致")
        if group_market_date_index.ndim != 1:
            raise ValueError("group_market_date_index 必須是 1D")
        if daily_features.shape[2] != len(MARKET_SET_FEATURE_COLUMNS):
            raise ValueError("market feature count 與正式 contract 不一致")
        if not np.isfinite(daily_features).all():
            raise ValueError("market daily features 含 NaN 或 infinite")
        normalized_history = int(history_bars)
        normalized_ratio = float(min_valid_history_ratio)
        normalized_max_stocks = int(max_stocks)
        normalized_max_dates = int(max_dates_per_batch)
        if normalized_history < 2:
            raise ValueError("market history_bars 必須 >= 2")
        if not 0.0 < normalized_ratio <= 1.0:
            raise ValueError("market min_valid_history_ratio 必須介於 0 與 1")
        if normalized_max_stocks < 0:
            raise ValueError("market max_stocks 必須 >= 0")
        if normalized_max_dates < 1:
            raise ValueError("market max_dates_per_batch 必須 >= 1")
        if len(group_market_date_index):
            minimum = int(np.min(group_market_date_index))
            maximum = int(np.max(group_market_date_index))
            if minimum < normalized_history - 1 or maximum >= len(daily_features):
                raise ValueError(
                    "group_market_date_index 無法提供完整 market history window: "
                    f"range=({minimum}, {maximum}), history={normalized_history}, dates={len(daily_features)}"
                )
        self.daily_features = daily_features
        self.daily_valid_mask = daily_valid_mask
        self.market_date_ordinals = market_date_ordinals
        self.group_market_date_index = group_market_date_index
        self.history_bars = normalized_history
        self.min_valid_history_ratio = normalized_ratio
        self.max_stocks = normalized_max_stocks
        self.max_dates_per_batch = normalized_max_dates

    @property
    def feature_count(self) -> int:
        return int(self.daily_features.shape[2])

    @property
    def stock_count(self) -> int:
        available = int(self.daily_features.shape[1])
        if self.max_stocks > 0:
            return min(available, self.max_stocks)
        return available

    def _selected_stock_indices(self) -> np.ndarray:
        count = int(self.daily_features.shape[1])
        if self.max_stocks <= 0 or self.max_stocks >= count:
            return np.arange(count, dtype=np.int64)
        # Deterministic coverage-preserving subset. No labels or OOS metrics are used.
        return np.linspace(0, count - 1, num=self.max_stocks, dtype=np.int64)

    def market_date_indices_for_group_indices(
        self, group_indices: Iterable[int] | np.ndarray
    ) -> np.ndarray:
        groups = np.asarray(
            list(group_indices) if not isinstance(group_indices, np.ndarray) else group_indices,
            dtype=np.int64,
        )
        if groups.ndim != 1:
            raise ValueError("market group indices 必須是 1D")
        if groups.size and (int(groups.min()) < 0 or int(groups.max()) >= len(self.group_market_date_index)):
            raise ValueError("market group index 超出範圍")
        return np.asarray(self.group_market_date_index[groups], dtype=np.int64)

    def market_date_indices_for_event_rows(
        self, event_group_index: np.ndarray, event_rows: Iterable[int] | np.ndarray
    ) -> np.ndarray:
        rows = np.asarray(
            list(event_rows) if not isinstance(event_rows, np.ndarray) else event_rows,
            dtype=np.int64,
        )
        if rows.ndim != 1:
            raise ValueError("market event rows 必須是 1D")
        return self.market_date_indices_for_group_indices(
            np.asarray(event_group_index[rows], dtype=np.int64)
        )

    def materialize_for_group_indices(
        self,
        group_indices: Iterable[int] | np.ndarray,
    ) -> MarketSetBatch:
        groups = np.asarray(
            list(group_indices) if not isinstance(group_indices, np.ndarray) else group_indices,
            dtype=np.int64,
        )
        if groups.ndim != 1:
            raise ValueError("market group indices 必須是 1D")
        if groups.size == 0:
            return MarketSetBatch(
                sequences=np.empty((0, 0, self.history_bars, self.feature_count), dtype=np.float32),
                history_mask=np.empty((0, 0, self.history_bars), dtype=np.bool_),
                valid_stock_mask=np.empty((0, 0), dtype=np.bool_),
                event_to_market=np.empty((0,), dtype=np.int64),
            )
        if int(groups.min()) < 0 or int(groups.max()) >= len(self.group_market_date_index):
            raise ValueError("market group index 超出範圍")
        event_dates = np.asarray(self.group_market_date_index[groups], dtype=np.int64)
        unique_dates, event_to_market = np.unique(event_dates, return_inverse=True)
        offsets = np.arange(-self.history_bars + 1, 1, dtype=np.int64)
        date_matrix = unique_dates[:, None] + offsets[None, :]
        stock_indices = self._selected_stock_indices()
        features = np.asarray(
            self.daily_features[date_matrix][:, :, stock_indices, :],
            dtype=np.float32,
        ).transpose(0, 2, 1, 3)
        history_mask = np.asarray(
            self.daily_valid_mask[date_matrix][:, :, stock_indices],
            dtype=np.bool_,
        ).transpose(0, 2, 1)
        minimum_valid = int(np.ceil(self.history_bars * self.min_valid_history_ratio))
        valid_stock_mask = history_mask[:, :, -1] & (
            history_mask.sum(axis=2) >= minimum_valid
        )
        if np.any(valid_stock_mask.sum(axis=1) == 0):
            bad_dates = unique_dates[valid_stock_mask.sum(axis=1) == 0]
            raise ValueError(f"market set date 沒有可用股票: indices={bad_dates.tolist()[:10]}")
        return MarketSetBatch(
            sequences=np.ascontiguousarray(features),
            history_mask=np.ascontiguousarray(history_mask),
            valid_stock_mask=np.ascontiguousarray(valid_stock_mask),
            event_to_market=np.asarray(event_to_market, dtype=np.int64),
        )

    def materialize_for_event_rows(
        self,
        event_group_index: np.ndarray,
        event_rows: Iterable[int] | np.ndarray,
    ) -> MarketSetBatch:
        rows = np.asarray(
            list(event_rows) if not isinstance(event_rows, np.ndarray) else event_rows,
            dtype=np.int64,
        )
        if rows.ndim != 1:
            raise ValueError("market event rows 必須是 1D")
        return self.materialize_for_group_indices(
            np.asarray(event_group_index[rows], dtype=np.int64)
        )

def build_market_daily_base_features(
    stock_frame: pd.DataFrame,
    market_dates: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray]:
    """Build no-lookahead one-day features and align them to the benchmark calendar."""

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = sorted(required - set(stock_frame.columns))
    if missing:
        raise KeyError(f"market stock frame 缺少欄位: {missing}")
    frame = stock_frame.sort_index()
    open_values = frame["Open"].astype(float)
    high_values = frame["High"].astype(float)
    low_values = frame["Low"].astype(float)
    close_values = frame["Close"].astype(float)
    volume_values = frame["Volume"].astype(float)
    previous_close = close_values.shift(1)
    feature_frame = pd.DataFrame(
        {
            "close_return": close_values / previous_close - 1.0,
            "overnight_return": open_values / previous_close - 1.0,
            "intraday_return": close_values / open_values - 1.0,
            "high_low_range": high_values / low_values - 1.0,
            "log_volume_change": np.log1p(np.maximum(volume_values, 0.0)).diff(),
        },
        index=frame.index,
    )
    feature_frame = feature_frame.reindex(columns=list(MARKET_SET_FEATURE_COLUMNS))
    aligned = feature_frame.reindex(market_dates)
    values = aligned.to_numpy(dtype=np.float64)
    valid = np.isfinite(values).all(axis=1)
    output = np.zeros(values.shape, dtype=np.float32)
    output[valid] = values[valid].astype(np.float32)
    return output, valid.astype(np.bool_)

def market_set_contract_payload() -> dict[str, object]:
    return {
        "storage_schema_version": MARKET_SET_STORAGE_SCHEMA_VERSION,
        "storage_format": MARKET_SET_STORAGE_FORMAT,
        "history_bars": int(BREAKOUT_QUALITY_MARKET_SET_HISTORY_BARS),
        "feature_columns": list(MARKET_SET_FEATURE_COLUMNS),
        "min_valid_history_ratio": float(BREAKOUT_QUALITY_MARKET_SET_MIN_VALID_HISTORY_RATIO),
        "max_stocks": int(BREAKOUT_QUALITY_MARKET_SET_MAX_STOCKS),
        "universe_rule": "point_in_time_rows_present_in_source_csv",
        "calendar": "benchmark_trading_dates",
        "missing_value_rule": "zero_features_with_explicit_history_mask",
    }

__all__ = [
    "IndexedMarketSetBank",
    "MARKET_SET_FEATURE_COLUMNS",
    "MARKET_SET_STORAGE_FORMAT",
    "MARKET_SET_STORAGE_SCHEMA_VERSION",
    "MarketSetBatch",
    "build_market_daily_base_features",
    "market_set_contract_payload",
]
