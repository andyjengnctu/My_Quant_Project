import pandas as pd
import requests
from datetime import timedelta
from io import StringIO

from tools.downloader import runtime as rt


def get_market_last_date():
    print("🕵️‍♂️ 正在確認最新交易日...")
    try:
        search_start = (rt.get_taipei_now() - timedelta(days=15)).strftime("%Y-%m-%d")
        loader = rt.get_finmind_loader()
        df = loader.get_data(dataset='TaiwanStockPrice', data_id='0050', start_date=search_start)
        if df is not None and not df.empty:
            df.columns = [c.lower() for c in df.columns]
            actual_date = str(df['date'].max()).split(' ')[0]
            print(f"📅 台股最新交易日 (FinMind) 為: {actual_date}")
            return actual_date
    except rt.EXPECTED_MARKET_DATE_EXCEPTIONS as e:
        rt.append_downloader_issues("最新交易日(FinMind)失敗", [f"{type(e).__name__}: {e}"])
        print(f"⚠️ FinMind 日期獲取異常: {type(e).__name__}: {e}")

    print("🔄 啟動備援方案 (YFinance) 獲取交易日...")
    try:
        yf = rt.get_yfinance_module()
        ticker = yf.Ticker("0050.TW")
        hist = ticker.history(period="5d")
        if not hist.empty:
            actual_date = hist.index[-1].strftime("%Y-%m-%d")
            print(f"📅 台股最新交易日 (YF備援) 為: {actual_date}")
            return actual_date
    except rt.EXPECTED_MARKET_DATE_EXCEPTIONS as e:
        rt.append_downloader_issues("最新交易日(YF備援)失敗", [f"{type(e).__name__}: {e}"])
        print(f"⚠️ YFinance 備援失敗: {type(e).__name__}: {e}")

    fallback_date = rt.get_taipei_now()
    if fallback_date.hour < 14:
        fallback_date -= timedelta(days=1)

    while fallback_date.weekday() >= 5:
        fallback_date -= timedelta(days=1)

    fallback_str = fallback_date.strftime("%Y-%m-%d")
    print(f"⚠️ 無法取得精準日期，使用智能推算平日備用日期: {fallback_str}")
    return fallback_str


def get_or_update_universe():
    rt.ensure_runtime_dirs()

    list_file = rt.get_universe_list_file_path()

    if rt.os.path.exists(list_file):
        file_mod_time = rt.get_taipei_file_mtime(list_file)
        if rt.get_taipei_now() - file_mod_time < timedelta(days=rt.RESCAN_DAYS):
            with open(list_file, 'r') as f:
                cached_tickers = [line.strip() for line in f if line.strip()]
            if cached_tickers:
                print(f"✅ 名單有效 (更新於: {file_mod_time.strftime('%Y-%m-%d')})，直接讀取。")
                return cached_tickers
            print("⚠️ universe 快取為空，重新海選。")

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
                print(f"\n⚠️ 名單來源抓取失敗: {url} | {type(e).__name__}: {e}")

    if not tickers_info:
        raise RuntimeError(
            "無法取得任何台股股票名單；請檢查網路、TWSE 來源格式或 requests/pandas 解析是否異常。"
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
            info = yf.Ticker(yf_t).fast_info
            last_volume = info.get('lastVolume', 0)
            market_cap = info.get('marketCap', 0)

            if last_volume >= rt.MIN_VOLUME:
                if is_etf or market_cap >= rt.MIN_MARKET_CAP:
                    qualified_tickers.append(sid)

        except rt.EXPECTED_SCREENING_EXCEPTIONS as e:
            screening_errors.append((sid, yf_t, f"{type(e).__name__}: {e}"))
        rt.time.sleep(rt.YF_SCREEN_SLEEP_SEC)

    qualified_tickers = list(dict.fromkeys(qualified_tickers))

    with open(list_file, 'w') as f:
        for t in qualified_tickers:
            f.write(f"{t}\n")

    print(f"\n🎉 海選完畢！共 {len(qualified_tickers)} 檔入選。")

    if universe_fetch_errors:
        rt.append_downloader_issues("名單來源失敗", universe_fetch_errors)
        print(f"⚠️ 名單來源失敗 {len(universe_fetch_errors)} 筆，詳細已寫入: {rt.get_downloader_issue_log_path()}")

    if screening_errors:
        screening_log_lines = [f"{sid} ({yf_t}) -> {err}" for sid, yf_t, err in screening_errors]
        rt.append_downloader_issues("快篩失敗", screening_log_lines)
        print(f"⚠️ 快篩失敗 {len(screening_errors)} 檔，詳細已寫入: {rt.get_downloader_issue_log_path()}")

    return qualified_tickers
