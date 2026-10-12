"""Pure trade-count representation; source provenance lives in the provider service."""
from __future__ import annotations

import numpy as np

TRADE_COUNT_FEATURE_NAMES = (
    "Stock_Lagged_Trade_Count_Log_IQR",
    "Benchmark_Lagged_Trade_Count_Log_IQR",
    "Stock_Lagged_Trade_Count_Available",
    "Benchmark_Lagged_Trade_Count_Available",
)


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


__all__ = ["TRADE_COUNT_FEATURE_NAMES", "normalized_trade_count_windows"]
