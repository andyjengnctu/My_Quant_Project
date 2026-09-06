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


UNIVERSE_CACHE_SCHEMA_VERSION = 2
UNIVERSE_SCREENING_CONTRACT = "completed_daily_volume_close_x_known_shares_v1"


def _universe_contract_identity() -> dict[str, object]:
    payload = {
        "schema_version": UNIVERSE_CACHE_SCHEMA_VERSION,
        "screening_contract": UNIVERSE_SCREENING_CONTRACT,
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
    for field in ("schema_version", "screening_contract", "min_volume", "min_market_cap", "contract_fingerprint"):
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


def _completed_daily_screening_inputs(ticker_obj, *, market_date: str, is_etf: bool) -> tuple[float, float]:
    target = pd.Timestamp(market_date).tz_localize(None).normalize()
    start = (target - pd.Timedelta(days=14)).date().isoformat()
    end = (target + pd.Timedelta(days=1)).date().isoformat()
    hist = ticker_obj.history(start=start, end=end, interval="1d", auto_adjust=False, actions=False)
    if hist is None or hist.empty:
        raise ValueError("yfinance completed daily history 為空")
    frame = hist.copy()
    idx = pd.to_datetime(frame.index, errors="coerce")
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    frame.index = idx.normalize()
    frame = frame.loc[frame.index.notna() & (frame.index <= target)].sort_index()
    if frame.empty:
        raise ValueError(f"yfinance 在 completed market_date={target.date()} 以前無 daily row")
    row = frame.iloc[-1]
    completed_volume = float(pd.to_numeric(row.get("Volume"), errors="coerce"))
    completed_close = float(pd.to_numeric(row.get("Close"), errors="coerce"))
    if not pd.notna(completed_volume) or completed_volume < 0 or not pd.notna(completed_close) or completed_close <= 0:
        raise ValueError("yfinance completed daily Volume/Close 不合法")
    if is_etf:
        return completed_volume, 0.0
    info = ticker_obj.fast_info
    shares = float(info.get("shares", 0) or 0)
    if not pd.notna(shares) or shares <= 0:
        raise ValueError("yfinance fast_info 缺少有效 shares，無法以 completed Close 計算市值")
    return completed_volume, completed_close * shares


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

            suffix = '.TW' if 'strMode=2' in url else '.TWO'
            for _, row in df.iterrows():
                code_name = str(row['有價證券代號及名稱']).split()
                if len(code_name) >= 2:
                    sid = code_name[0]
                    is_etf = str(row['CFICode']).startswith('CE')
                    tickers_info.append({"yf_ticker": f"{sid}{suffix}", "sid": sid, "is_etf": is_etf})

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

    qualified_tickers = []
    screening_errors = []
    total_check = len(tickers_info)
    print(f"⏳ 準備快篩 {total_check} 檔純股與 ETF...\n")

    try:
        yf = rt.get_yfinance_module()
    except rt.EXPECTED_SCREENING_EXCEPTIONS as e:
        screening_errors.append(("__INIT__", "yfinance", f"{type(e).__name__}: {e}"))
        screening_log_lines = [f"{sid} ({yf_t}) -> {err}" for sid, yf_t, err in screening_errors]
        rt.append_downloader_issues("快篩失敗", screening_log_lines)
        raise RuntimeError(f"快篩初始化失敗：{type(e).__name__}: {e}") from e

    for i, item in enumerate(tickers_info):
        pct = ((i + 1) / total_check) * 100
        yf_t, sid, is_etf = item['yf_ticker'], item['sid'], item['is_etf']

        print(f"\r🔍 快篩進度: [{i+1:>4}/{total_check} | {pct:>5.1f}%] {yf_t:<8} ", end="", flush=True)
        try:
            ticker_obj = yf.Ticker(yf_t)
            completed_volume, completed_market_cap = _completed_daily_screening_inputs(
                ticker_obj, market_date=market_date_text, is_etf=bool(is_etf)
            )
            if completed_volume >= rt.MIN_VOLUME:
                if is_etf or completed_market_cap >= rt.MIN_MARKET_CAP:
                    qualified_tickers.append(sid)

        except rt.EXPECTED_SCREENING_EXCEPTIONS as e:
            screening_errors.append((sid, yf_t, f"{type(e).__name__}: {e}"))
        rt.time.sleep(rt.YF_SCREEN_SLEEP_SEC)

    qualified_tickers = list(dict.fromkeys(qualified_tickers))

    if screening_errors:
        screening_log_lines = [f"{sid} ({yf_t}) -> {err}" for sid, yf_t, err in screening_errors]
        rt.append_downloader_issues("快篩失敗", screening_log_lines)
        raise RuntimeError(
            f"台股 universe 快篩有 {len(screening_errors)} 檔無法可靠判定，依 Trading 保守原則中止重掃；"
            "不得把不完整篩選結果發布成新的 universe cache。"
        )

    _publish_universe_cache(list_file, qualified_tickers=qualified_tickers, market_date=market_date_text)

    print(f"\n🎉 海選完畢！共 {len(qualified_tickers)} 檔入選。")
    return qualified_tickers

