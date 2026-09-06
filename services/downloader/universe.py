import pandas as pd
import requests
from datetime import timedelta
from io import StringIO

from core.file_integrity import atomic_write_json, canonical_json_sha256, load_json_strict

from core.console_report import project_relative_display_path
from core.trading_market_clock import (
    latest_allowed_completed_daily_date,
    select_latest_completed_daily_date,
)
from services.downloader import runtime as rt


def get_market_last_date():
    print("🕵️‍♂️ 正在確認最新交易日...")
    try:
        search_start = (rt.get_taipei_now() - timedelta(days=15)).strftime("%Y-%m-%d")
        loader = rt.get_finmind_loader()
        df = loader.get_data(dataset=rt.FINMIND_PRICE_DATASET, data_id='0050', start_date=search_start)
        if df is not None and not df.empty:
            df.columns = [c.lower() for c in df.columns]
            actual_date = select_latest_completed_daily_date(df['date'].tolist(), now=rt.get_taipei_now())
            if actual_date:
                print(f"📅 台股最新完整交易日 (FinMind) 為: {actual_date}")
                return actual_date
    except rt.EXPECTED_MARKET_DATE_EXCEPTIONS as e:
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


UNIVERSE_CACHE_SCHEMA_VERSION = 3
UNIVERSE_SCREENING_CONTRACT = "finmind_backer_bulk_exact_market_date_volume_market_value_v1"


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


def _load_reusable_universe_cache(path, *, now) -> list[str] | None:
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
    if str(payload.get("built_market_date") or "").strip() == "":
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


def _load_finmind_bulk_screening_data(*, market_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_date = str(pd.Timestamp(market_date).date())
    try:
        loader = rt.get_finmind_loader()
        price_raw = loader.get_data(
            dataset=rt.FINMIND_UNIVERSE_VOLUME_DATASET,
            start_date=target_date,
            timeout=rt.REQUEST_TIMEOUT_SEC,
        )
        market_value_raw = loader.get_data(
            dataset=rt.FINMIND_UNIVERSE_MARKET_VALUE_DATASET,
            start_date=target_date,
            timeout=rt.REQUEST_TIMEOUT_SEC,
        )
        price = _normalize_finmind_bulk_screening_frame(
            price_raw,
            dataset=rt.FINMIND_UNIVERSE_VOLUME_DATASET,
            market_date=target_date,
            value_column="trading_volume",
            allow_zero=True,
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
    universe_by_sid: dict[str, bool] = {}
    for item in tickers_info:
        sid = str(item.get("sid") or "").strip()
        if not sid:
            raise ValueError("TWSE/TPEX universe 存在空白 sid")
        is_etf = bool(item.get("is_etf"))
        prior = universe_by_sid.get(sid)
        if prior is not None and prior != is_etf:
            raise ValueError(f"TWSE/TPEX universe 同一 sid ETF identity 不一致: {sid}")
        universe_by_sid[sid] = is_etf

    volume_by_sid = dict(zip(price["stock_id"].astype(str), price["trading_volume"].astype(float)))
    market_value_by_sid = dict(zip(market_value["stock_id"].astype(str), market_value["market_value"].astype(float)))

    qualified: list[str] = []
    missing_market_value_for_high_volume_stock: list[str] = []
    listed_without_exact_price = 0
    high_volume_count = 0
    for sid, is_etf in universe_by_sid.items():
        volume = volume_by_sid.get(sid)
        if volume is None:
            # Exact-date FinMind price rows define the actually traded daily universe.
            # A listed but non-traded/suspended symbol is conservatively not actionable.
            listed_without_exact_price += 1
            continue
        if volume < float(rt.MIN_VOLUME):
            continue
        high_volume_count += 1
        if is_etf:
            qualified.append(sid)
            continue
        cap = market_value_by_sid.get(sid)
        if cap is None or cap <= 0:
            missing_market_value_for_high_volume_stock.append(sid)
            continue
        if cap >= float(rt.MIN_MARKET_CAP):
            qualified.append(sid)

    if missing_market_value_for_high_volume_stock:
        sample = ",".join(missing_market_value_for_high_volume_stock[:20])
        raise RuntimeError(
            "FinMind bulk快篩對高成交量股票缺少同日 market_value，依 Trading 保守原則中止重掃；"
            f"count={len(missing_market_value_for_high_volume_stock)} sample={sample}"
        )

    qualified = list(dict.fromkeys(qualified))
    stats = {
        "listed_count": len(universe_by_sid),
        "price_exact_date_count": len(volume_by_sid),
        "market_value_exact_date_count": len(market_value_by_sid),
        "listed_without_exact_price_count": int(listed_without_exact_price),
        "high_volume_count": int(high_volume_count),
        "qualified_count": len(qualified),
    }
    return qualified, stats


def get_or_update_universe(*, market_date: str):
    rt.ensure_runtime_dirs()
    market_date_text = str(pd.Timestamp(market_date).date())
    list_file = rt.get_universe_list_file_path()

    cached_tickers = _load_reusable_universe_cache(list_file, now=rt.get_taipei_now())
    if cached_tickers:
        file_mod_time = rt.get_taipei_file_mtime(list_file)
        print(f"✅ 名單有效 (更新於: {file_mod_time.strftime('%Y-%m-%d')})，直接讀取。")
        return cached_tickers

    print(f"🕵️‍♂️ 啟動全市場海選 (市值 > {rt.MIN_MARKET_CAP/1e8:.0f}億 且 成交量 > {rt.MIN_VOLUME/10000:.0f}萬)...")

    urls = [
        "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2",
        "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4"
    ]

    tickers_info = []
    universe_fetch_errors = []

    for url in urls:
        try:
            res = requests.get(url, timeout=rt.REQUEST_TIMEOUT_SEC)
            res.raise_for_status()

            df = pd.read_html(StringIO(res.text))[0]
            df.columns = df.iloc[0]
            df = df.iloc[2:]

            mask = df['CFICode'].str.startswith('ES', na=False) | df['CFICode'].str.startswith('CE', na=False)
            df = df[mask]

            for _, row in df.iterrows():
                code_name = str(row['有價證券代號及名稱']).split()
                if len(code_name) >= 2:
                    sid = code_name[0]
                    is_etf = str(row['CFICode']).startswith('CE')
                    tickers_info.append({"sid": sid, "is_etf": is_etf})

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
        f"全市場成交量 + 全市場市值，共 2 個 dataset requests..."
    )
    price, market_value = _load_finmind_bulk_screening_data(market_date=market_date_text)
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

