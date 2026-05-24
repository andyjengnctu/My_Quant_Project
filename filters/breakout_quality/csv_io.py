"""CSV helpers for breakout quality filter artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

_TICKER_COLUMN = "ticker"
_TICKER_DTYPE = {_TICKER_COLUMN: str}


def normalize_ticker_column(df: pd.DataFrame, *, source: str | Path = "") -> pd.DataFrame:
    """Return a copy with ticker preserved as stripped text, including leading zeros."""
    if _TICKER_COLUMN not in df.columns:
        return df
    out = df.copy()
    ticker = out[_TICKER_COLUMN]
    if ticker.isna().any():
        missing_count = int(ticker.isna().sum())
        detail = f"; source={source}" if source else ""
        raise ValueError(f"breakout quality CSV 的 ticker 欄位含空值: {missing_count}{detail}")
    out[_TICKER_COLUMN] = ticker.map(lambda value: str(value).strip())
    empty_count = int((out[_TICKER_COLUMN] == "").sum())
    if empty_count:
        detail = f"; source={source}" if source else ""
        raise ValueError(f"breakout quality CSV 的 ticker 欄位含空字串: {empty_count}{detail}")
    return out


def read_breakout_quality_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read filter CSV artifacts without losing leading-zero tickers such as 0050."""
    kwargs.setdefault("dtype", _TICKER_DTYPE)
    kwargs.setdefault("low_memory", False)
    df = pd.read_csv(path, **kwargs)
    return normalize_ticker_column(df, source=path)


__all__ = ["normalize_ticker_column", "read_breakout_quality_csv"]
