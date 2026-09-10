import pandas as pd
import requests
from datetime import timedelta
from io import StringIO

from core.file_integrity import atomic_write_json, canonical_json_sha256, load_json_strict
from core.market_data_contract import FINMIND_RAW_PRICE_ARCHIVE_DATASET
from core.market_data_pool_contract import screen_daily_trading_execution_pool

from core.console_report import project_relative_display_path
from core.trading_market_clock import (
    latest_allowed_completed_daily_date,
    select_latest_completed_daily_date,
)
from services.downloader import runtime as rt
from services.downloader.finmind_http import FinMindHttpError, request_finmind_data_with_retry


_UNIVERSE_SOURCE_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
}
_UNIVERSE_SOURCE_ENCODING = "cp950"
_UNIVERSE_SOURCE_MAX_ATTEMPTS = 3
_UNIVERSE_SOURCE_RETRY_DELAYS_SEC = (1.0, 3.0)


def _parse_isin_universe_html(html: str) -> list[dict[str, object]]:
    """Parse TWSE ISIN stock/ETF rows without depending on localized headers."""

    tables = pd.read_html(StringIO(str(html)))
    if not tables:
        raise ValueError("TWSE ISIN 回應不含可解析 table")
    frame = tables[0]
    # The ISIN table contract exposes security code/name in column 0 and
    # CFICode in column 5.  Positional parsing intentionally avoids coupling
    # Trading runtime to the MS950-decoded Chinese header labels.
    if frame.shape[1] < 6:
        raise ValueError(f"TWSE ISIN table 欄位不足: columns={frame.shape[1]}")

    code_name = frame.iloc[:, 0].astype("string")
    cfi_code = frame.iloc[:, 5].astype("string").str.strip().str.upper()
    eligible = cfi_code.str.startswith("ES", na=False) | cfi_code.str.startswith("CE", na=False)
    if not bool(eligible.any()):
        raise ValueError("TWSE ISIN table 未解析到任何 ES/CE 股票或 ETF")

    result: list[dict[str, object]] = []
    for raw_code_name, raw_cfi in zip(code_name.loc[eligible], cfi_code.loc[eligible]):
        token = str(raw_code_name or "").strip().split()
        if not token or token[0].lower() in {"nan", "<na>"}:
            continue
        result.append({
            "sid": token[0],
            "is_etf": str(raw_cfi).startswith("CE"),
        })
    if not result:
        raise ValueError("TWSE ISIN ES/CE rows 缺少可用 security id")
    return result


def _fetch_isin_universe_source(url: str) -> list[dict[str, object]]:
    """Fetch one TWSE/TPEX ISIN source with bounded transient retry."""

    failures: list[str] = []
    last_exc: BaseException | None = None
    for attempt in range(1, _UNIVERSE_SOURCE_MAX_ATTEMPTS + 1):
        try:
            response = requests.get(
                url,
                headers=_UNIVERSE_SOURCE_HTTP_HEADERS,
                timeout=rt.REQUEST_TIMEOUT_SEC,
            )
            response.raise_for_status()
            # ISIN pages declare MS950; cp950 is Python's compatible codec.
            response.encoding = _UNIVERSE_SOURCE_ENCODING
            rows = _parse_isin_universe_html(response.text)
            if not rows:
                raise ValueError("TWSE ISIN source 解析結果為空")
            return rows
        except rt.EXPECTED_UNIVERSE_FETCH_EXCEPTIONS as exc:
            last_exc = exc
            failures.append(f"attempt={attempt} {type(exc).__name__}: {exc}")
            if attempt >= _UNIVERSE_SOURCE_MAX_ATTEMPTS:
                break
            delay = _UNIVERSE_SOURCE_RETRY_DELAYS_SEC[
                min(attempt - 1, len(_UNIVERSE_SOURCE_RETRY_DELAYS_SEC) - 1)
            ]
            rt.time.sleep(float(delay))

    detail = " | ".join(failures)
    raise ValueError(
        f"TWSE ISIN source 在 {_UNIVERSE_SOURCE_MAX_ATTEMPTS} 次嘗試後仍失敗: {detail}"
    ) from last_exc


def get_market_last_date(*, client=None):
    print("🕵️‍♂️ 正在確認最新交易日...")
    try:
        search_start = (rt.get_taipei_now() - timedelta(days=15)).strftime("%Y-%m-%d")
        if client is None:
            loader = rt.get_finmind_loader()
            df = loader.get_data(dataset=rt.FINMIND_PRICE_DATASET, data_id='0050', start_date=search_start)
        else:
            df = request_finmind_data_with_retry(
                client, dataset=rt.FINMIND_PRICE_DATASET, data_id='0050', start_date=search_start
            )
        if df is not None and not df.empty:
            df.columns = [c.lower() for c in df.columns]
            actual_date = select_latest_completed_daily_date(df['date'].tolist(), now=rt.get_taipei_now())
            if actual_date:
                print(f"📅 台股最新完整交易日 (FinMind) 為: {actual_date}")
                return actual_date
    except rt.EXPECTED_MARKET_DATE_EXCEPTIONS + (FinMindHttpError,) as e:
        rt.append_downloader_issues("最新交易日(FinMind)失敗", [f"{type(e).__name__}: {e}"])
        print(f"注意：FinMind 日期獲取異常: {type(e).__name__}: {e}")

    print("🔄 啟動備援方案 (YFinance) 獲取交易日...")
    try:
        yf = rt.get_yfinance_module()
        ticker = yf.Ticker("0050.TW")
        hist = ticker.history(period="5d")
        if not hist.empty:
            actual_date = select_latest_completed_daily_date(hist.index.tolist(), now=rt.get_taipei_now())
            if actual_date:
                print(f"📅 台股最新完整交易日 (YF備援) 為: {actual_date}")
                return actual_date
    except rt.EXPECTED_MARKET_DATE_EXCEPTIONS as e:
        rt.append_downloader_issues("最新交易日(YF備援)失敗", [f"{type(e).__name__}: {e}"])
        print(f"注意：YFinance 備援失敗: {type(e).__name__}: {e}")

    allowed_completed = latest_allowed_completed_daily_date(now=rt.get_taipei_now())
    raise RuntimeError(
        "無法由 FinMind 或 YFinance 可靠確認台股最新完整交易日；"
        f"僅能推得 calendar cutoff={allowed_completed}，不能把平日推算當成實際交易日。"
        "依 Trading 保守原則本次資料更新中止。"
    )


UNIVERSE_CACHE_SCHEMA_VERSION = 4
UNIVERSE_SCREENING_CONTRACT = "finmind_backer_bulk_exact_market_date_volume_market_value_v2"


def _universe_contract_identity() -> dict[str, object]:
    payload = {
        "schema_version": UNIVERSE_CACHE_SCHEMA_VERSION,
        "screening_contract": UNIVERSE_SCREENING_CONTRACT,
        "volume_dataset": str(rt.FINMIND_UNIVERSE_VOLUME_DATASET),
        "market_value_dataset": str(rt.FINMIND_UNIVERSE_MARKET_VALUE_DATASET),
        "min_volume": int(rt.MIN_VOLUME),
        "min_market_cap": int(rt.MIN_MARKET_CAP),
    }
    payload["contract_fingerprint"] = canonical_json_sha256(payload)
    return payload


def _load_reusable_universe_cache(path, *, now, market_date: str) -> list[str] | None:
    cache_path = rt.os.fspath(path)
    if not rt.os.path.exists(cache_path):
        return None
    file_mod_time = rt.get_taipei_file_mtime(cache_path)
    if now - file_mod_time >= timedelta(days=rt.RESCAN_DAYS):
        return None
    try:
        payload = load_json_strict(cache_path)
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    expected = _universe_contract_identity()
    for field in (
        "schema_version", "screening_contract", "volume_dataset", "market_value_dataset",
        "min_volume", "min_market_cap", "contract_fingerprint",
    ):
        if payload.get(field) != expected.get(field):
            return None
    tickers = [str(item).strip() for item in list(payload.get("qualified_tickers") or []) if str(item).strip()]
    if not tickers:
        return None
    built_market_date = str(payload.get("built_market_date") or "").strip()
    if not built_market_date:
        return None
    target_market_date = str(pd.Timestamp(market_date).date())
    if built_market_date != target_market_date:
        return None
    core = {key: value for key, value in payload.items() if key != "cache_fingerprint"}
    if str(payload.get("cache_fingerprint") or "") != canonical_json_sha256(core):
        return None
    return tickers


def _publish_universe_cache(path, *, qualified_tickers: list[str], market_date: str) -> None:
    payload = {
        **_universe_contract_identity(),
        "built_market_date": str(pd.Timestamp(market_date).date()),
        "qualified_tickers": list(qualified_tickers),
    }
    payload["cache_fingerprint"] = canonical_json_sha256(payload)
    atomic_write_json(path, payload)


def _normalize_finmind_bulk_screening_frame(
    frame,
    *,
    dataset: str,
    market_date: str,
    value_column: str,
    allow_zero: bool,
) -> pd.DataFrame:
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"FinMind {dataset} 全市場資料為空")
    normalized = frame.copy()
    normalized.columns = [str(col).strip().lower() for col in normalized.columns]
    required = {"date", "stock_id", value_column}
    missing = sorted(required - set(normalized.columns))
    if missing:
        raise ValueError(f"FinMind {dataset} 缺少必要欄位: {missing}")

    target_date = str(pd.Timestamp(market_date).date())
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce").dt.date.astype("string")
    normalized = normalized.loc[normalized["date"] == target_date].copy()
    if normalized.empty:
        raise ValueError(f"FinMind {dataset} 尚無 market_date={target_date} 的全市場資料")

    normalized["stock_id"] = normalized["stock_id"].astype("string").str.strip()
    if normalized["stock_id"].isna().any() or (normalized["stock_id"] == "").any():
        raise ValueError(f"FinMind {dataset} 存在空白 stock_id")
    duplicated = sorted(normalized.loc[normalized["stock_id"].duplicated(keep=False), "stock_id"].astype(str).unique().tolist())
    if duplicated:
        raise ValueError(f"FinMind {dataset} 同一 market_date 存在重複 stock_id: {duplicated[:10]}")

    normalized[value_column] = pd.to_numeric(normalized[value_column], errors="coerce")
    invalid = normalized[value_column].isna()
    if allow_zero:
        invalid = invalid | (normalized[value_column] < 0)
    else:
        invalid = invalid | (normalized[value_column] <= 0)
    if invalid.any():
        bad = normalized.loc[invalid, "stock_id"].astype(str).head(10).tolist()
        raise ValueError(f"FinMind {dataset} {value_column} 存在不合法值: {bad}")
    return normalized[["stock_id", value_column]].copy()


def _normalize_exact_date_ticker_evidence(
    frame,
    *,
    dataset: str,
    market_date: str,
) -> set[str]:
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError(f"FinMind {dataset} 全市場資料為空")
    normalized = frame.copy()
    normalized.columns = [str(col).strip().lower() for col in normalized.columns]
    required = {"date", "stock_id"}
    missing = sorted(required - set(normalized.columns))
    if missing:
        raise ValueError(f"FinMind {dataset} 缺少必要欄位: {missing}")

    target_date = str(pd.Timestamp(market_date).date())
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    normalized = normalized.loc[normalized["date"] == target_date].copy()
    if normalized.empty:
        raise ValueError(f"FinMind {dataset} 尚無 market_date={target_date} 的全市場 ticker evidence")
    normalized["stock_id"] = normalized["stock_id"].astype("string").str.strip()
    if normalized["stock_id"].isna().any() or (normalized["stock_id"] == "").any():
        raise ValueError(f"FinMind {dataset} 存在空白 stock_id")
    duplicated = sorted(
        normalized.loc[normalized["stock_id"].duplicated(keep=False), "stock_id"].astype(str).unique().tolist()
    )
    if duplicated:
        raise ValueError(f"FinMind {dataset} 同一 market_date 存在重複 stock_id: {duplicated[:10]}")
    return set(normalized["stock_id"].astype(str).tolist())


def _load_finmind_bulk_screening_data(
    *,
    market_date: str,
    universe_ticker_ids: set[str] | list[str] | tuple[str, ...],
    client=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_date = str(pd.Timestamp(market_date).date())
    try:
        if client is None:
            loader = rt.get_finmind_loader()
            price_raw = loader.get_data(
                dataset=rt.FINMIND_UNIVERSE_VOLUME_DATASET,
                start_date=target_date,
                timeout=rt.REQUEST_TIMEOUT_SEC,
            )
            raw_price_evidence = loader.get_data(
                dataset=FINMIND_RAW_PRICE_ARCHIVE_DATASET,
                start_date=target_date,
                timeout=rt.REQUEST_TIMEOUT_SEC,
            )
            market_value_raw = loader.get_data(
                dataset=rt.FINMIND_UNIVERSE_MARKET_VALUE_DATASET,
                start_date=target_date,
                timeout=rt.REQUEST_TIMEOUT_SEC,
            )
        else:
            price_raw = request_finmind_data_with_retry(
                client,
                dataset=rt.FINMIND_UNIVERSE_VOLUME_DATASET,
                start_date=target_date,
                end_date=target_date,
            )
            raw_price_evidence = request_finmind_data_with_retry(
                client,
                dataset=FINMIND_RAW_PRICE_ARCHIVE_DATASET,
                start_date=target_date,
                end_date=target_date,
            )
            market_value_raw = request_finmind_data_with_retry(
                client,
                dataset=rt.FINMIND_UNIVERSE_MARKET_VALUE_DATASET,
                start_date=target_date,
                end_date=target_date,
            )
        price = _normalize_finmind_bulk_screening_frame(
            price_raw,
            dataset=rt.FINMIND_UNIVERSE_VOLUME_DATASET,
            market_date=target_date,
            value_column="trading_volume",
            allow_zero=True,
        )
        raw_traded_ids = _normalize_exact_date_ticker_evidence(
            raw_price_evidence,
            dataset=FINMIND_RAW_PRICE_ARCHIVE_DATASET,
            market_date=target_date,
        )
        current_universe_ids = {str(sid).strip() for sid in universe_ticker_ids if str(sid).strip()}
        if not current_universe_ids:
            raise ValueError("TWSE/TPEX current stock/ETF universe 為空，無法驗證 PriceAdj completeness")
        adjusted_ids = set(price["stock_id"].astype(str).tolist())
        required_traded_ids = raw_traded_ids & current_universe_ids
        missing_adjusted = sorted(required_traded_ids - adjusted_ids)
        if missing_adjusted:
            raise ValueError(
                "FinMind TaiwanStockPriceAdj exact-date payload 缺少 current TWSE/TPEX stock/ETF universe 中，"
                "已由 TaiwanStockPrice 確認成交的 ticker；"
                f"count={len(missing_adjusted)} sample={missing_adjusted[:20]}"
            )
        market_value = _normalize_finmind_bulk_screening_frame(
            market_value_raw,
            dataset=rt.FINMIND_UNIVERSE_MARKET_VALUE_DATASET,
            market_date=target_date,
            value_column="market_value",
            allow_zero=True,
        )
        return price, market_value
    except rt.EXPECTED_SCREENING_EXCEPTIONS as exc:
        rt.append_downloader_issues(
            "FinMind bulk快篩失敗",
            [f"market_date={target_date} -> {type(exc).__name__}: {exc}"],
        )
        raise RuntimeError(
            "FinMind Backer 全市場快篩失敗，依 Trading 保守原則中止重掃；"
            f"market_date={target_date} | {type(exc).__name__}: {exc}"
        ) from exc


def _screen_finmind_bulk_universe(
    tickers_info: list[dict[str, object]],
    *,
    price: pd.DataFrame,
    market_value: pd.DataFrame,
) -> tuple[list[str], dict[str, int]]:
    """Legacy current-market adapter to the canonical Trading execution-pool owner."""

    daily_market_members = [
        {
            "stock_id": str(item.get("sid") or "").strip(),
            "is_etf": bool(item.get("is_etf")),
        }
        for item in tickers_info
    ]
    return screen_daily_trading_execution_pool(
        daily_market_members,
        price_rows=price,
        market_value_rows=market_value,
        min_volume=rt.MIN_VOLUME,
        min_market_cap=rt.MIN_MARKET_CAP,
    )


def get_or_update_universe(*, market_date: str, client=None):
    rt.ensure_runtime_dirs()
    market_date_text = str(pd.Timestamp(market_date).date())
    list_file = rt.get_universe_list_file_path()

    cached_tickers = _load_reusable_universe_cache(
        list_file, now=rt.get_taipei_now(), market_date=market_date_text
    )
    if cached_tickers:
        file_mod_time = rt.get_taipei_file_mtime(list_file)
        print(f"✅ 名單有效 (更新於: {file_mod_time.strftime('%Y-%m-%d')})，直接讀取。")
        return cached_tickers

    print(f"🕵️‍♂️ 啟動全市場海選 (市值 > {rt.MIN_MARKET_CAP/1e8:.0f}億 且 成交量 > {rt.MIN_VOLUME/10000:.0f}萬)...")

    urls = [
        "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2",
        "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4",
    ]

    tickers_info = []
    universe_fetch_errors = []

    for url in urls:
        try:
            source_rows = _fetch_isin_universe_source(url)
            if not source_rows:
                raise ValueError("TWSE/TPEX ISIN source 解析結果為空")
            tickers_info.extend(source_rows)
        except rt.EXPECTED_UNIVERSE_FETCH_EXCEPTIONS as e:
            universe_fetch_errors.append(f"{url} -> {type(e).__name__}: {e}")
            if rt.VERBOSE_UNIVERSE_FETCH_ERRORS:
                print(f"\n注意：名單來源抓取失敗: {url} | {type(e).__name__}: {e}")

    if not tickers_info:
        if universe_fetch_errors:
            rt.append_downloader_issues("名單來源失敗", universe_fetch_errors)
        raise RuntimeError(
            "無法取得任何台股股票名單；請檢查網路、TWSE 來源格式或 requests/pandas 解析是否異常。"
        )
    if universe_fetch_errors:
        rt.append_downloader_issues("名單來源失敗", universe_fetch_errors)
        raise RuntimeError(
            "台股 universe 來源不完整，依 Trading 保守原則中止重掃；"
            "不得把部分 TWSE/TPEX 名單發布成新的 universe cache。"
        )

    total_check = len(tickers_info)
    print(
        f"⏳ FinMind Backer bulk 快篩 {total_check} 檔純股與 ETF："
        f"PriceAdj 成交量 + raw current-universe ticker completeness evidence + 全市場市值，共 3 個 dataset requests..."
    )
    price, market_value = _load_finmind_bulk_screening_data(
        market_date=market_date_text,
        universe_ticker_ids={str(item.get("sid") or "").strip() for item in tickers_info},
        client=client,
    )
    try:
        qualified_tickers, stats = _screen_finmind_bulk_universe(
            tickers_info, price=price, market_value=market_value
        )
    except (ValueError, RuntimeError) as exc:
        rt.append_downloader_issues("FinMind bulk快篩失敗", [f"market_date={market_date_text} -> {type(exc).__name__}: {exc}"])
        raise

    _publish_universe_cache(list_file, qualified_tickers=qualified_tickers, market_date=market_date_text)

    print(
        "✅ FinMind bulk 快篩完成 | "
        f"listed={stats['listed_count']} | exact-price={stats['price_exact_date_count']} | "
        f"exact-market-value={stats['market_value_exact_date_count']} | "
        f"non-traded/suspended={stats['listed_without_exact_price_count']} | "
        f"high-volume={stats['high_volume_count']}"
    )
    print(f"🎉 海選完畢！共 {len(qualified_tickers)} 檔入選。")
    return qualified_tickers

