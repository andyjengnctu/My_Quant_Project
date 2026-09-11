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


def _validate_exact_date_probe_frame(frame: pd.DataFrame, *, expected_date: str) -> pd.DataFrame:
    """Validate one full-market exact-date PriceAdj discovery response.

    Discovery must never infer latest-market-date from a wider range because a
    provider may sparsely/truncatedly materialize that range.  A non-empty
    exact-date response is accepted only when every row belongs to the requested
    date; empty responses are legal for weekends/exchange holidays.
    """

    if frame is None or not isinstance(frame, pd.DataFrame):
        raise TradingBulkPriceUnsupported("TaiwanStockPriceAdj provider payload 不是 DataFrame")
    if frame.empty:
        return frame.copy()
    columns = _provider_columns(frame)
    date_column = columns.get("date")
    if date_column is None:
        raise TradingBulkPriceUnsupported("TaiwanStockPriceAdj exact-date probe 缺少 date 欄位")
    parsed = pd.to_datetime(frame[date_column], errors="coerce")
    if parsed.isna().any():
        raise TradingBulkPriceUnsupported("TaiwanStockPriceAdj exact-date probe 含不合法 date")
    observed = set(parsed.dt.strftime("%Y-%m-%d").tolist())
    if observed != {str(expected_date)}:
        raise TradingBulkPriceUnsupported(
            "TaiwanStockPriceAdj exact-date probe scope 不一致："
            f"requested={expected_date}, observed={sorted(observed)}"
        )
    return frame.copy()


def probe_latest_adjusted_price_market_date(
    *,
    client,
    now: datetime | None = None,
    current_market_date: str | None = None,
) -> TradingPriceProbe:
    """Discover the latest completed Trading day using exact-date probes only.

    ``current_market_date`` is already trusted local evidence.  Discovery walks
    backward only through dates newer than that floor and stops at the first
    non-empty exact-date response, so weekends/holidays remain bounded without
    ever issuing the former sparse long-range query.
    """

    resolved_now = rt.get_taipei_now() if now is None else now
    candidate = latest_allowed_completed_daily_date(now=resolved_now)
    floor = None if current_market_date is None else _iso(current_market_date)
    cursor = date.fromisoformat(str(candidate))
    floor_date = None if floor is None else date.fromisoformat(floor)

    if floor_date is not None and cursor <= floor_date:
        empty = pd.DataFrame()
        return TradingPriceProbe(
            candidate_date=str(candidate),
            market_date=floor,
            current_range=PriceRange(str(candidate), str(candidate)),
            current_frame=empty,
        )

    last_range = PriceRange(str(candidate), str(candidate))
    while floor_date is None or cursor > floor_date:
        probe_date = cursor.isoformat()
        last_range = PriceRange(probe_date, probe_date)
        frame = request_finmind_data_with_retry(
            client,
            dataset=rt.FINMIND_PRICE_DATASET,
            start_date=probe_date,
            end_date=probe_date,
        )
        frame = _validate_exact_date_probe_frame(frame, expected_date=probe_date)
        if not frame.empty:
            normalized = normalize_adjusted_price_frame(frame)
            market_date = select_latest_completed_daily_date(normalized["Date"].tolist(), now=resolved_now)
            if market_date != probe_date:
                raise TradingBulkPriceUnsupported(
                    "TaiwanStockPriceAdj exact-date probe 未能確認 requested completed day："
                    f"requested={probe_date}, resolved={market_date}"
                )
            _seed_exact_date_cache(
                client,
                raw_frame=frame,
                dates=(probe_date,),
                coverage_start=probe_date,
                coverage_end=probe_date,
            )
            return TradingPriceProbe(
                candidate_date=str(candidate),
                market_date=probe_date,
                current_range=last_range,
                current_frame=frame,
            )
        cursor -= timedelta(days=1)

    if floor is not None:
        return TradingPriceProbe(
            candidate_date=str(candidate),
            market_date=floor,
            current_range=last_range,
            current_frame=pd.DataFrame(),
        )
    raise TradingBulkPriceUnsupported(
        "無法由 FinMind TaiwanStockPriceAdj full-market exact-date probes 確認最新完整交易日"
    )

__all__ = [
    "TradingBulkPriceUnsupported",
    "PriceRange",
    "TradingPriceProbe",
    "build_price_history_ranges",
    "normalize_adjusted_price_frame",
    "probe_latest_adjusted_price_market_date",
]
