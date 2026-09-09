"""Canonical execution-critical Trading adjusted-price refresh.

FinMind remains the only adjusted-price calculator.  This module changes only
provider request geometry: when many local ticker CSVs are stale, current-vintage
``TaiwanStockPriceAdj`` is fetched in full-market calendar ranges and split into
per-ticker legacy CSVs.  When only a few files are stale, the cheaper per-ticker
full-history path is retained.  Both paths preserve the legacy full-history
refresh semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

from core.file_integrity import atomic_write_text
from core.market_data_execution_policy import get_market_data_execution_policy
from core.trading_dataset_identity import inspect_trading_dataset_member_date_evidence
from core.trading_market_clock import latest_allowed_completed_daily_date, select_latest_completed_daily_date
from services.downloader import runtime as rt
from services.downloader.finmind_http import request_finmind_data_with_retry


_TRADING_CALENDAR_DATASET = "TaiwanStockTradingDate"


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


@dataclass(frozen=True)
class LocalPriceFreshness:
    latest: tuple[str, ...]
    stale: tuple[str, ...]
    unreadable: tuple[str, ...]


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


def _ensure_quota_capacity(client, planned_data_requests: int) -> None:
    planned = int(planned_data_requests)
    if planned <= 0 or not hasattr(client, "get_usage"):
        return
    usage = client.get_usage()
    policy = get_market_data_execution_policy()
    reserve = min(int(policy.quota_reserve_requests), max(0, int(usage.api_request_limit) - 1))
    usable = int(usage.api_request_limit) - int(usage.user_count) - reserve
    if usable < planned:
        raise RuntimeError(
            "Trading execution-critical adjusted-price refresh quota 不足："
            f"planned={planned}, usable={usable}, used={usage.user_count}/{usage.api_request_limit}, reserve={reserve}"
        )


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


def _load_trading_calendar_dates(client) -> tuple[str, ...]:
    frame = request_finmind_data_with_retry(client, dataset=_TRADING_CALENDAR_DATASET)
    if frame is None or not isinstance(frame, pd.DataFrame):
        raise TradingBulkPriceUnsupported("TaiwanStockTradingDate provider payload 不是 DataFrame")
    columns = _provider_columns(frame)
    date_column = columns.get("date")
    if date_column is None:
        raise TradingBulkPriceUnsupported("TaiwanStockTradingDate 缺少 date 欄")
    parsed = pd.to_datetime(frame[date_column], errors="coerce").dropna().dt.strftime("%Y-%m-%d")
    dates = tuple(sorted(set(parsed.tolist())))
    if not dates:
        raise TradingBulkPriceUnsupported("TaiwanStockTradingDate 沒有可用交易日 evidence")
    return dates


def _validate_bulk_range_calendar_coverage(
    normalized: pd.DataFrame,
    *,
    price_range: PriceRange,
    trading_dates: tuple[str, ...],
    market_date: str,
) -> None:
    start = str(pd.Timestamp(price_range.start_date).date())
    range_end = str(pd.Timestamp(price_range.end_date).date())
    proof_end = min(range_end, str(pd.Timestamp(market_date).date()))
    actual_dates = set(normalized["Date"].dt.strftime("%Y-%m-%d").tolist())
    outside = sorted(value for value in actual_dates if value < start or value > range_end)
    if outside:
        raise TradingBulkPriceUnsupported(
            "TaiwanStockPriceAdj bulk range 回傳超出 request boundary 的日期；"
            f"range={start}~{range_end} sample={outside[:20]}"
        )
    expected = {value for value in trading_dates if start <= value <= proof_end}
    if not expected:
        return
    missing = sorted(expected - actual_dates)
    if missing:
        raise TradingBulkPriceUnsupported(
            "TaiwanStockPriceAdj bulk range 缺少 TaiwanStockTradingDate 已確認交易日；"
            f"range={start}~{proof_end} missing={missing[:20]} count={len(missing)}"
        )


def _missing_local_price_files(target_tickers: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        sid for sid in target_tickers
        if not (Path(rt.SAVE_DIR) / f"{sid}.csv").is_file()
    )


def inspect_legacy_price_baseline(sid: str, *, market_date: str):
    """Downloader wrapper around the canonical Trading CSV date-evidence inspector."""

    return inspect_trading_dataset_member_date_evidence(
        Path(rt.SAVE_DIR) / f"{sid}.csv",
        through_date=market_date,
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


def inspect_local_price_freshness(tickers: Iterable[str], *, market_date: str) -> LocalPriceFreshness:
    latest: list[str] = []
    stale: list[str] = []
    unreadable: list[str] = []
    target = str(pd.Timestamp(market_date).date())
    for sid in [str(item) for item in tickers]:
        baseline = inspect_legacy_price_baseline(sid, market_date=target)
        if not baseline.exists:
            stale.append(sid)
            continue
        if not baseline.readable:
            unreadable.append(sid)
            stale.append(sid)
            continue
        if baseline.last_date == target:
            latest.append(sid)
        else:
            stale.append(sid)
    return LocalPriceFreshness(tuple(latest), tuple(stale), tuple(unreadable))


def _legacy_frame_for_ticker(normalized: pd.DataFrame, sid: str, *, market_date: str) -> pd.DataFrame:
    rows = normalized.loc[normalized["stock_id"] == str(sid), ["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    target = pd.Timestamp(market_date).normalize()
    rows = rows.loc[rows["Date"].dt.normalize() <= target].sort_values("Date")
    rows = rows.drop_duplicates(subset=["Date"], keep="last")
    if rows.empty:
        raise ValueError(f"FinMind TaiwanStockPriceAdj 對 {sid} 回傳空歷史")
    rows = rows.set_index("Date")
    return rows[["Open", "High", "Low", "Close", "Volume"]]


def _write_bulk_history(
    *,
    normalized_chunks: list[pd.DataFrame],
    target_tickers: tuple[str, ...],
    market_date: str,
) -> int:
    combined = pd.concat(normalized_chunks, ignore_index=True) if normalized_chunks else pd.DataFrame()
    if combined.empty:
        raise RuntimeError("TaiwanStockPriceAdj full-market bulk refresh 沒有任何可寫入資料")
    target_set = set(target_tickers)
    combined = combined.loc[combined["stock_id"].isin(target_set)].copy()

    # Validate every target before publishing any CSV.  A bulk capability or
    # coverage failure therefore falls back to the legacy per-ticker producer
    # without leaving a partially bulk-refreshed execution dataset behind.
    prepared: dict[str, pd.DataFrame] = {}
    for sid in target_tickers:
        rows = _legacy_frame_for_ticker(combined, sid, market_date=market_date)
        baseline = inspect_legacy_price_baseline(sid, market_date=market_date)
        if not baseline.exists:
            raise TradingBulkPriceUnsupported(
                f"TaiwanStockPriceAdj bulk history 缺少 {sid} 的可讀 legacy baseline；"
                "新檔必須改走 per-ticker full-history producer"
            )
        if not baseline.readable:
            raise TradingBulkPriceUnsupported(
                f"TaiwanStockPriceAdj bulk history 無法驗證 {sid} 的 legacy baseline；"
                f"{baseline.error}"
            )
        refreshed_dates = {item.date().isoformat() for item in rows.index}
        missing_existing = sorted(set(baseline.dates) - refreshed_dates)
        if missing_existing:
            raise TradingBulkPriceUnsupported(
                f"TaiwanStockPriceAdj bulk history 對 {sid} 遺失既有日期；"
                f"missing={missing_existing[:20]} count={len(missing_existing)}"
            )
        prepared[sid] = rows

    for sid in target_tickers:
        atomic_write_text(Path(rt.SAVE_DIR) / f"{sid}.csv", prepared[sid].to_csv())
    return len(prepared)


def refresh_trading_adjusted_price_dataset(
    tickers: Iterable[str],
    market_date: str,
    *,
    client,
    probe: TradingPriceProbe,
    universe_tickers: Iterable[str],
    verbose: bool = True,
) -> dict[str, object]:
    """Refresh legacy CSV truth with the lowest-request exact full-history strategy."""

    rt.ensure_runtime_dirs()
    target_tickers = tuple(dict.fromkeys(str(item) for item in tickers))
    universe_set = {str(item) for item in universe_tickers}
    freshness = inspect_local_price_freshness(target_tickers, market_date=market_date)
    total = len(target_tickers)
    if not freshness.stale:
        return {
            "total": total,
            "count_success": 0,
            "count_skipped_latest": total,
            "last_date_check_error_count": len(freshness.unreadable),
            "download_error_count": 0,
            "trimmed_future_row_count": 0,
            "issue_log_path": None,
            "price_fetch_strategy": "already_latest",
            "planned_price_data_requests": 0,
            "actual_price_provider_requests": 0,
            "bulk_range_count": 0,
            "stale_ticker_count": 0,
        }

    ranges = build_price_history_ranges(
        start_date=rt.PRICE_HISTORY_START_DATE,
        end_date=market_date,
        chunk_months=rt.CANONICAL_PRICE_BULK_CHUNK_MONTHS,
    )
    bulk_request_count = len(ranges)
    per_ticker_request_count = len(freshness.stale)
    before_requests = int(getattr(client, "data_request_count", 0))

    # The current range was already fetched by the market-date probe and is a
    # zero-HTTP cache hit.  Count planned provider work accordingly.
    current_key_cached = bool(
        getattr(client, "has_cached_data", lambda **_kwargs: False)(
            dataset=rt.FINMIND_PRICE_DATASET,
            start_date=probe.current_range.start_date,
            end_date=probe.current_range.end_date,
        )
    )
    calendar_key_cached = bool(
        getattr(client, "has_cached_data", lambda **_kwargs: False)(
            dataset=_TRADING_CALENDAR_DATASET,
        )
    )
    bulk_provider_plan = (
        bulk_request_count
        - (1 if current_key_cached else 0)
        + (0 if calendar_key_cached else 1)
    )
    missing_local = _missing_local_price_files(target_tickers)
    bulk_baseline_unavailable = bool(missing_local or freshness.unreadable)
    raw_evidence_tickers = set(missing_local) | set(freshness.unreadable)
    per_ticker_provider_plan = per_ticker_request_count + len(raw_evidence_tickers)

    if bulk_baseline_unavailable or per_ticker_request_count < bulk_provider_plan:
        _ensure_quota_capacity(client, per_ticker_provider_plan)
        # Reuse the legacy function through its explicit shared-client seam; it
        # refreshes only stale files and preserves current full-history semantics.
        from services.downloader.sync import smart_download_vip_data

        summary = smart_download_vip_data(
            target_tickers,
            market_date,
            verbose=verbose,
            client=client,
            require_target_date_tickers=universe_set,
        )
        return {
            **summary,
            "price_fetch_strategy": "per_ticker_full_history",
            "planned_price_data_requests": per_ticker_provider_plan,
            "actual_price_provider_requests": int(getattr(client, "data_request_count", 0)) - before_requests,
            "bulk_range_count": bulk_request_count,
            "stale_ticker_count": per_ticker_request_count,
        }

    _ensure_quota_capacity(client, bulk_provider_plan)
    trading_dates = _load_trading_calendar_dates(client)
    by_range = {(item.start_date, item.end_date): item for item in ranges}
    current_key = (probe.current_range.start_date, probe.current_range.end_date)
    if current_key not in by_range:
        raise RuntimeError("Trading canonical price current probe range 與 full-history range contract 不一致")

    # Validate that the current full-market payload intersects the Trading
    # universe before spending the rest of the historical requests.  Cached
    # universe members may legitimately lack a target-date row (suspension or
    # temporary halt); full-history existence is validated before publication.
    current_normalized = normalize_adjusted_price_frame(probe.current_frame)
    _validate_bulk_range_calendar_coverage(
        current_normalized,
        price_range=probe.current_range,
        trading_dates=trading_dates,
        market_date=str(market_date),
    )
    target_rows = current_normalized.loc[
        current_normalized["Date"].dt.strftime("%Y-%m-%d") == str(market_date), "stock_id"
    ].astype(str)
    current_target_ids = set(target_rows.tolist())
    if universe_set and not (universe_set & current_target_ids):
        raise TradingBulkPriceUnsupported(
            "TaiwanStockPriceAdj full-market current range 與 Trading universe 無任何交集；"
            "無法證明 bulk payload 可作 canonical producer"
        )
    missing_current = sorted(universe_set - current_target_ids)
    if missing_current:
        raise TradingBulkPriceUnsupported(
            "TaiwanStockPriceAdj full-market current range 缺少已確認 actionable universe ticker；"
            f"missing={missing_current[:20]} count={len(missing_current)}"
        )

    target_set = set(target_tickers)
    normalized_chunks: list[pd.DataFrame] = [
        current_normalized.loc[current_normalized["stock_id"].isin(target_set)].copy()
    ]
    recent_start = date.fromisoformat(str(market_date)) - timedelta(days=6)
    recent_dates = {(recent_start + timedelta(days=offset)).isoformat() for offset in range(7)}

    remaining = [item for item in ranges if (item.start_date, item.end_date) != current_key]
    if verbose:
        print(
            f"\n💎 Canonical PriceAdj bulk refresh | stale={per_ticker_request_count}/{total} | "
            f"range requests={bulk_request_count} (current range REUSE)"
        )
    for index, price_range in enumerate(remaining, start=1):
        raw = request_finmind_data_with_retry(
            client,
            dataset=rt.FINMIND_PRICE_DATASET,
            start_date=price_range.start_date,
            end_date=price_range.end_date,
            retain_cache=False,
        )
        if raw is None or not isinstance(raw, pd.DataFrame):
            raise TradingBulkPriceUnsupported("TaiwanStockPriceAdj historical bulk payload 不是 DataFrame")
        if raw.empty:
            continue
        _seed_exact_date_cache(
            client,
            raw_frame=raw,
            dates=recent_dates,
            coverage_start=price_range.start_date,
            coverage_end=price_range.end_date,
        )
        normalized = normalize_adjusted_price_frame(raw)
        _validate_bulk_range_calendar_coverage(
            normalized,
            price_range=price_range,
            trading_dates=trading_dates,
            market_date=str(market_date),
        )
        normalized_chunks.append(normalized.loc[normalized["stock_id"].isin(target_set)].copy())
        if verbose and (index == len(remaining) or index % 10 == 0):
            print(f"\r[PriceAdj Bulk] {index}/{len(remaining)} historical ranges", end="", flush=True)
    if verbose and remaining:
        print()

    written = _write_bulk_history(
        normalized_chunks=normalized_chunks,
        target_tickers=target_tickers,
        market_date=str(market_date),
    )
    if freshness.unreadable:
        rt.append_downloader_issues(
            "最後日期檢查失敗",
            [f"{sid}: canonical bulk refresh forced" for sid in freshness.unreadable],
        )
    return {
        "total": total,
        "count_success": written,
        "count_skipped_latest": 0,
        "last_date_check_error_count": len(freshness.unreadable),
        "download_error_count": 0,
        "trimmed_future_row_count": 0,
        "issue_log_path": rt.get_downloader_issue_log_path() if freshness.unreadable else None,
        "price_fetch_strategy": "full_market_range_current_vintage",
        "planned_price_data_requests": bulk_provider_plan,
        "actual_price_provider_requests": int(getattr(client, "data_request_count", 0)) - before_requests,
        "bulk_range_count": bulk_request_count,
        "stale_ticker_count": per_ticker_request_count,
        "current_universe_missing_price_row_count": 0,
    }


__all__ = [
    "TradingBulkPriceUnsupported",
    "PriceRange",
    "TradingPriceProbe",
    "LocalPriceFreshness",
    "build_price_history_ranges",
    "normalize_adjusted_price_frame",
    "probe_latest_adjusted_price_market_date",
    "inspect_legacy_price_baseline",
    "inspect_local_price_freshness",
    "refresh_trading_adjusted_price_dataset",
]
