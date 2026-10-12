"""Pure trade-count representation; source provenance lives in the provider service."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADE_COUNT_FEATURE_NAMES = (
    "Stock_Lagged_Trade_Count_Log_IQR",
    "Benchmark_Lagged_Trade_Count_Log_IQR",
    "Stock_Lagged_Trade_Count_Available",
    "Benchmark_Lagged_Trade_Count_Available",
)


def previous_session_trade_count(
    index: pd.DatetimeIndex,
    *,
    ticker: str,
    calendar: pd.DatetimeIndex,
    observations: dict[str, dict[pd.Timestamp, tuple[float, pd.Timestamp]]],
) -> tuple[np.ndarray, np.ndarray]:
    """Strict T−1 exchange-date join, with no carry-forward of missing stock dates."""
    ordered = pd.DatetimeIndex(calendar).normalize()
    if ordered.has_duplicates or not ordered.is_monotonic_increasing:
        raise ValueError("官方交易日曆需要唯一、嚴格遞增")
    signal_dates = pd.DatetimeIndex(index).normalize()
    result = np.zeros(len(signal_dates), dtype=np.float64)
    valid = np.zeros(len(signal_dates), dtype=bool)
    lookup = observations.get(str(ticker), {})
    positions = ordered.get_indexer(signal_dates)
    for i, (signal, pos) in enumerate(zip(signal_dates, positions)):
        if pos <= 0:
            continue
        prev = ordered[pos - 1]
        observation = lookup.get(prev)
        if observation is None:
            continue
        count, captured = observation
        if not np.isfinite(count) or count < 0 or pd.isna(captured):
            continue
        if pd.Timestamp(captured) >= signal:
            continue
        result[i] = float(count)
        valid[i] = True
    return result, valid


def normalized_trade_count_windows(counts, available, *, placebo: bool = False):
    """Window-only log(count+1) robust z-score; mask never means observed zero.

    Values are learned on *eligible* count observations within the feature window
    at the decision date, never over all future stock history or OOS statistics.
    """
    x = np.asarray(counts, dtype=np.float64)
    m = np.asarray(available, dtype=bool)
    if x.ndim != 2 or x.shape != m.shape or x.shape[1] < 1:
        raise ValueError("trade-count windows/masks shape contract無效")
    if np.any(m & (~np.isfinite(x) | (x < 0))):
        raise ValueError("trade-count source contains invalid observed counts")
    if placebo:
        return np.zeros(x.shape, dtype=np.float32)
    if not np.any(m):
        return np.zeros(x.shape, dtype=np.float32)
    safe = np.where(m, np.log1p(np.where(m, x, 0.0)), np.nan)
    has_valid = m.any(axis=1)
    safe[~has_valid, 0] = 0.0
    q25, median, q75 = np.nanpercentile(safe, (25, 50, 75), axis=1)
    denom = q75 - q25
    denom = np.where(np.isfinite(denom) & (denom > 1e-8), denom, 1.0)
    z = (np.where(m, np.log1p(np.where(m, x, 0.0)), median[:, None]) - median[:, None]) / denom[:, None]
    z[~m] = 0.0
    if not np.isfinite(z).all():
        raise ValueError("trade-count normalized feature contains nonfinite values")
    return z.astype(np.float32)


__all__ = [
    "TRADE_COUNT_FEATURE_NAMES",
    "normalized_trade_count_windows",
    "previous_session_trade_count",
]
