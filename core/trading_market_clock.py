"""Trading daily-bar completion clock contract.

The Trading workflow consumes end-of-day OHLCV.  Today's daily bar is not
considered complete before the canonical provider-publication grace time for the
project's adjusted-price source, even if a provider already exposes a provisional
row.  The time is resolved from the Market Data publication-policy SSOT rather
than maintained as a second hard-coded cutoff here.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Iterable

import pandas as pd

from core.market_data_contract import FINMIND_ADJUSTED_PRICE_DATASET
from core.market_data_dataset_registry import get_market_dataset_spec
from core.market_data_freshness_contract import build_market_data_freshness_contract
from core.runtime_utils import get_taipei_now


def trading_daily_bar_complete_time() -> tuple[int, int]:
    """Return the configured fail-closed ``(hour, minute)`` for daily OHLCV.

    The tuple return keeps the established consumer contract while sourcing the
    actual cutoff from the Market Data publication-policy SSOT.
    """

    contract = build_market_data_freshness_contract(
        get_market_dataset_spec(FINMIND_ADJUSTED_PRICE_DATASET)
    )
    if not contract.publication_schedule_verified:
        raise RuntimeError(
            "Trading daily bar completion time 缺 provider-verified publication schedule: "
            f"dataset={FINMIND_ADJUSTED_PRICE_DATASET}"
        )
    hour_text, minute_text = str(contract.publication_first_check_time).split(":", 1)
    return int(hour_text), int(minute_text)


def _completion_hour_minute() -> tuple[int, int]:
    return trading_daily_bar_complete_time()


def latest_allowed_completed_daily_date(*, now=None) -> str:
    current = get_taipei_now() if now is None else now
    date_value = current.date()
    cutoff_hour, cutoff_minute = _completion_hour_minute()
    if (int(current.hour), int(current.minute)) < (cutoff_hour, cutoff_minute):
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
    "trading_daily_bar_complete_time",
    "latest_allowed_completed_daily_date",
    "select_latest_completed_daily_date",
    "assert_completed_daily_information_date",
]
