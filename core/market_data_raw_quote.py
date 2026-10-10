"""Canonical interpretation of FinMind raw OHLC availability and execution evidence.

This capability is shared by Research raw-price controls, DL diagnostics and
Trading broker-fill validation.  A published all-zero OHLC is *possible* no-quote
evidence, not proof of a bad provider file.  It must never become a synthetic
price or an executable daily high/low.  These functions do not sanitize a
model's time series or decide missing-value policies for research experiments.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _numeric_series(frame: pd.DataFrame, field: str) -> pd.Series:
    if field not in frame.columns:
        raise KeyError(f"Raw market quote 缺少必要欄位: {field}")
    return pd.to_numeric(frame[field], errors="coerce").astype("float64")


def raw_market_positive_volume_mask(
    frame: pd.DataFrame, *, volume_col: str = "Trading_Volume",
) -> pd.Series:
    """Observed positive-volume dates; not evidence that OHLC was published."""
    volume = _numeric_series(frame, volume_col)
    return pd.Series(np.isfinite(volume.to_numpy(dtype=float)) & volume.gt(0).to_numpy(), index=frame.index)


def raw_market_executable_quote_mask(
    frame: pd.DataFrame,
    *,
    high_col: str = "max",
    low_col: str = "min",
    volume_col: str = "Trading_Volume",
) -> pd.Series:
    """Daily low/high and volume suffice for a *possible* historical fill.

    This does not prove that a particular broker execution occurred; it only
    admits a row to exact-date fill price validation.  It deliberately does not
    compare ``open`` with the range: the provider can use different ``open``
    semantics for emerging-market securities.  An all-zero quote is rejected.
    """
    high = _numeric_series(frame, high_col)
    low = _numeric_series(frame, low_col)
    finite = np.isfinite(high.to_numpy(dtype=float)) & np.isfinite(low.to_numpy(dtype=float))
    return pd.Series(
        finite & high.gt(0).to_numpy() & low.gt(0).to_numpy()
        & high.ge(low).to_numpy()
        & raw_market_positive_volume_mask(frame, volume_col=volume_col).to_numpy(),
        index=frame.index,
    )


def raw_research_observed_ohlc_mask(frame: pd.DataFrame) -> pd.Series:
    """Whether a raw OHLC observation can serve as a numerical research feature.

    Independent of turnover/volume and of Trading fill availability. FinMind
    emerging-market Open can legitimately sit outside intraday High/Low; Close
    must still lie inside the reported trading range. No synthetic quote is made.
    A False value is *missing/unusable numerical evidence*, not corruption proof.
    """
    prices = np.column_stack([
        _numeric_series(frame, field).to_numpy(dtype=float)
        for field in ("Open", "High", "Low", "Close")
    ])
    valid = (
        np.all(np.isfinite(prices) & (prices > 0.0), axis=1)
        & (prices[:, 1] >= prices[:, 2])
        & (prices[:, 3] >= prices[:, 2])
        & (prices[:, 3] <= prices[:, 1])
    )
    return pd.Series(valid, index=frame.index)


def raw_ohlc_diagnostic_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Provider observations, NOT a replacement for canonical model sanitization.

    The caller's ``Volume`` might be inherited from an adjusted control frame;
    counts are diagnostic and cannot establish provider quote availability.
    A nonzero open outside high/low is not automatically provider corruption.
    """
    prices = pd.DataFrame({
        field: _numeric_series(frame, field)
        for field in ("Open", "High", "Low", "Close")
    }, index=frame.index)
    all_zero = prices.eq(0.0).all(axis=1)
    nonzero = ~all_zero
    volume = _numeric_series(frame, "Volume")
    return {
        "all_zero_ohlc": all_zero,
        "all_zero_control_volume_zero": all_zero & volume.eq(0.0),
        "all_zero_control_volume_positive": all_zero & volume.gt(0.0),
        "all_zero_control_volume_other": all_zero & ~(volume.eq(0.0) | volume.gt(0.0)),
        "nonzero_open_outside_high_low": nonzero & (prices["Open"].gt(prices["High"]) | prices["Open"].lt(prices["Low"])),
        "nonzero_close_outside_high_low": nonzero & (prices["Close"].gt(prices["High"]) | prices["Close"].lt(prices["Low"])),
        "nonzero_any_nonpositive_price": nonzero & prices.le(0.0).any(axis=1),
    }


__all__ = [
    "raw_market_positive_volume_mask",
    "raw_market_executable_quote_mask",
    "raw_ohlc_diagnostic_masks",
    "raw_research_observed_ohlc_mask",
]
