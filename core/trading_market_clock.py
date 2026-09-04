"""Trading daily-bar completion clock contract.

The Trading workflow consumes end-of-day OHLCV.  Before the local safety
cutoff, today's daily bar is not considered complete even if a data provider
already exposes a provisional row.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Iterable

import pandas as pd

from core.runtime_utils import get_taipei_now

TRADING_DAILY_BAR_COMPLETE_HOUR = 14


def latest_allowed_completed_daily_date(*, now=None) -> str:
    current = get_taipei_now() if now is None else now
    date_value = current.date()
    if int(current.hour) < TRADING_DAILY_BAR_COMPLETE_HOUR:
        date_value -= timedelta(days=1)
    return date_value.isoformat()


def select_latest_completed_daily_date(date_values: Iterable[object], *, now=None) -> str | None:
    allowed = pd.Timestamp(latest_allowed_completed_daily_date(now=now))
    series = pd.to_datetime(pd.Series(list(date_values)), errors="coerce").dropna()
    if series.empty:
        return None
    normalized = series.dt.normalize()
    eligible = normalized[normalized <= allowed]
    if eligible.empty:
        return None
    return eligible.max().strftime("%Y-%m-%d")


def assert_completed_daily_information_date(date_value: object, *, now=None) -> str:
    parsed = pd.Timestamp(date_value).normalize()
    allowed = pd.Timestamp(latest_allowed_completed_daily_date(now=now))
    if parsed > allowed:
        raise RuntimeError(
            "Trading daily data 尚未完成："
            f"information_date={parsed.strftime('%Y-%m-%d')} > "
            f"latest_allowed_completed_date={allowed.strftime('%Y-%m-%d')}"
        )
    return parsed.strftime("%Y-%m-%d")


__all__ = [
    "TRADING_DAILY_BAR_COMPLETE_HOUR",
    "latest_allowed_completed_daily_date",
    "select_latest_completed_daily_date",
    "assert_completed_daily_information_date",
]
