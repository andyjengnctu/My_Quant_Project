"""Trading Market Data V2 latest completed-day PriceAdj probe.

This module is provider-read-only: it may probe FinMind ``TaiwanStockPriceAdj``
to discover the latest completed Trading market date, but it never materializes
or refreshes Legacy Trading CSV data.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable

import pandas as pd

from core.trading_market_clock import latest_allowed_completed_daily_date, select_latest_completed_daily_date
from services.downloader import runtime as rt
from services.downloader.finmind_http import request_finmind_data_with_retry



class TradingBulkPriceUnsupported(RuntimeError):
    """Provider response does not support the required full-market range contract."""


@dataclass(frozen=True)
class PriceRange:
    start_date: str
    end_date: str


@dataclass
class TradingPriceProbe:
    candidate_date: str
    market_date: str
    current_range: PriceRange
    current_frame: pd.DataFrame


def _iso(value: object) -> str:
    return date.fromisoformat(str(value)).isoformat()


def build_price_history_ranges(
    *,
    start_date: str,
    end_date: str,
    chunk_months: int,
) -> tuple[PriceRange, ...]:
    start = date.fromisoformat(_iso(start_date))
    end = date.fromisoformat(_iso(end_date))
    months = int(chunk_months)
    if months <= 0:
        raise ValueError("Trading canonical price chunk_months 必須 > 0")
    if start > end:
        raise ValueError("Trading canonical price history start 不得晚於 end")
    rows: list[PriceRange] = []
    cursor = start
    while cursor <= end:
        start_index = cursor.year * 12 + (cursor.month - 1)
        next_index = start_index + months
        next_year, next_month0 = divmod(next_index, 12)
        next_boundary = date(next_year, next_month0 + 1, 1)
        chunk_end = min(end, next_boundary - timedelta(days=1))
        rows.append(PriceRange(cursor.isoformat(), chunk_end.isoformat()))
        cursor = chunk_end + timedelta(days=1)
    return tuple(rows)


def _provider_columns(frame: pd.DataFrame) -> dict[str, str]:
    if frame is None or not isinstance(frame, pd.DataFrame):
        raise TradingBulkPriceUnsupported("TaiwanStockPriceAdj provider payload 不是 DataFrame")
    return {str(column).strip().lower(): str(column) for column in frame.columns}


def normalize_adjusted_price_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = _provider_columns(frame)
    aliases = {
        "date": "Date",
        "stock_id": "stock_id",
        "open": "Open",
        "max": "High",
        "min": "Low",
        "close": "Close",
        "trading_volume": "Volume",
    }
    missing = sorted(key for key in aliases if key not in columns)
    if missing:
        raise TradingBulkPriceUnsupported(f"TaiwanStockPriceAdj 缺少必要欄位: {missing}")
    normalized = frame[[columns[key] for key in aliases]].copy()
    normalized.columns = [aliases[key] for key in aliases]
    normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce")
    normalized["stock_id"] = normalized["stock_id"].astype("string").str.strip()
    normalized = normalized.loc[normalized["Date"].notna() & normalized["stock_id"].notna() & (normalized["stock_id"] != "")].copy()
    for column in ("Open", "High", "Low", "Close", "Volume"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    invalid = normalized[["Open", "High", "Low", "Close", "Volume"]].isna().any(axis=1)
    if invalid.any():
        sample = normalized.loc[invalid, ["Date", "stock_id"]].head(10).to_dict("records")
        raise TradingBulkPriceUnsupported(f"TaiwanStockPriceAdj OHLCV 含不合法數值: {sample}")
    if normalized.duplicated(subset=["Date", "stock_id"]).any():
        sample = normalized.loc[normalized.duplicated(subset=["Date", "stock_id"], keep=False), ["Date", "stock_id"]].head(10).to_dict("records")
        raise TradingBulkPriceUnsupported(f"TaiwanStockPriceAdj 同日同 ticker 重複: {sample}")
    return normalized


def _seed_exact_date_cache(
    client,
    *,
    raw_frame: pd.DataFrame,
    dates: Iterable[str],
    coverage_start: str,
    coverage_end: str,
) -> None:
    seed = getattr(client, "seed_data", None)
    if not callable(seed):
        return
    columns = _provider_columns(raw_frame)
    date_column = columns.get("date")
    if date_column is None:
        return
    parsed = pd.to_datetime(raw_frame[date_column], errors="coerce").dt.strftime("%Y-%m-%d")
    start = str(pd.Timestamp(coverage_start).date())
    end = str(pd.Timestamp(coverage_end).date())
    for value in sorted({str(item) for item in dates if start <= str(item) <= end}):
        subset = raw_frame.loc[parsed == value].copy()
        seed(
            dataset=rt.FINMIND_PRICE_DATASET,
            start_date=value,
            end_date=value,
            frame=subset,
        )


def probe_latest_adjusted_price_market_date(*, client, now: datetime | None = None) -> TradingPriceProbe:
    resolved_now = rt.get_taipei_now() if now is None else now
    candidate = latest_allowed_completed_daily_date(now=resolved_now)
    ranges = build_price_history_ranges(
        start_date=rt.PRICE_HISTORY_START_DATE,
        end_date=candidate,
        chunk_months=rt.CANONICAL_PRICE_BULK_CHUNK_MONTHS,
    )
    current_range = ranges[-1]
    frame = request_finmind_data_with_retry(
        client,
        dataset=rt.FINMIND_PRICE_DATASET,
        start_date=current_range.start_date,
        end_date=current_range.end_date,
    )
    normalized = normalize_adjusted_price_frame(frame)
    market_date = select_latest_completed_daily_date(normalized["Date"].tolist(), now=resolved_now)
    if market_date is None:
        raise TradingBulkPriceUnsupported(
            "無法由 FinMind TaiwanStockPriceAdj full-market current range 確認最新完整交易日；"
            f"range={current_range.start_date}~{current_range.end_date}"
        )
    recent_start = date.fromisoformat(market_date) - timedelta(days=6)
    _seed_exact_date_cache(
        client,
        raw_frame=frame,
        dates=((recent_start + timedelta(days=offset)).isoformat() for offset in range(7)),
        coverage_start=current_range.start_date,
        coverage_end=current_range.end_date,
    )
    return TradingPriceProbe(
        candidate_date=str(candidate),
        market_date=str(market_date),
        current_range=current_range,
        current_frame=frame,
    )

__all__ = [
    "TradingBulkPriceUnsupported",
    "PriceRange",
    "TradingPriceProbe",
    "build_price_history_ranges",
    "normalize_adjusted_price_frame",
    "probe_latest_adjusted_price_market_date",
]
