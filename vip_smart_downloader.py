import pandas as pd
import os
import time
import requests
from datetime import datetime, timedelta
from io import StringIO
from core.v16_log_utils import append_issue_log, build_timestamped_log_path

API_TOKEN = os.getenv("FINMIND_API_TOKEN", "")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "tw_stock_data_vip")
LIST_FILE = os.path.join(SAVE_DIR, "universe_list.txt")

MIN_VOLUME = 1_000_000
MIN_MARKET_CAP = 10_000_000_000
RESCAN_DAYS = 7
REQUEST_TIMEOUT_SEC = 10
YF_SCREEN_SLEEP_SEC = 0.01
FINMIND_DOWNLOAD_SLEEP_SEC = 0.5
FINMIND_PRICE_DATASET = 'TaiwanStockPriceAdj'
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

# # (AI註: 大量批次時避免逐筆錯誤洗板；詳細清單仍保留在摘要與 log)
EXPECTED_MARKET_DATE_EXCEPTIONS = (
    requests.RequestException,
    ValueError,
    KeyError,
    IndexError,
    TypeError,
    ImportError,
    ModuleNotFoundError,
)

EXPECTED_UNIVERSE_FETCH_EXCEPTIONS = (
    requests.RequestException,
    ValueError,
    KeyError,
    IndexError,
    pd.errors.EmptyDataError,
)

EXPECTED_SCREENING_EXCEPTIONS = (
    requests.RequestException,
    ValueError,
    KeyError,
    TypeError,
    AttributeError,
    ImportError,
    ModuleNotFoundError,
)

EXPECTED_LAST_DATE_CHECK_EXCEPTIONS = (
    OSError,
    ValueError,
    KeyError,
    IndexError,
    pd.errors.EmptyDataError,
    pd.errors.ParserError,
)

EXPECTED_DOWNLOAD_EXCEPTIONS = (
    requests.RequestException,
    ValueError,
    KeyError,
    TypeError,
    pd.errors.EmptyDataError,
    pd.errors.ParserError,
    OSError,
    ImportError,
    ModuleNotFoundError,
)

# # (AI註: 大量批次時預設不逐筆洗板；需要時再手動切成 True)
VERBOSE_UNIVERSE_FETCH_ERRORS = False
VERBOSE_LAST_DATE_CHECK_ERRORS = False
VERBOSE_DOWNLOAD_ERRORS = False

dl = None


def get_yfinance_module():
    import yfinance as yf
    return yf


def get_finmind_dataloader_class():
    from FinMind.data import DataLoader
    return DataLoader


def get_finmind_loader():
    global dl
    if dl is None:
        DataLoader = get_finmind_dataloader_class()
        dl = DataLoader()
        if API_TOKEN:
            dl.login_by_token(api_token=API_TOKEN)
    return dl


# # (AI註: 將執行期目錄建立延後到實際執行，避免被 import 時產生副作用)
def ensure_runtime_dirs():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# # (AI註: 防錯透明化 - 將錯誤摘要落檔，避免長時間批次執行後 console 訊息遺失)
DOWNLOADER_SESSION_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
DOWNLOADER_ISSUE_LOG_PATH = None


# # (AI註: session log path 改為 lazy init；只有真的要寫 log 時才建立輸出目錄)
def get_downloader_issue_log_path():
    global DOWNLOADER_ISSUE_LOG_PATH
    if DOWNLOADER_ISSUE_LOG_PATH is None:
        ensure_runtime_dirs()
        DOWNLOADER_ISSUE_LOG_PATH = build_timestamped_log_path(
            "downloader_issues",
            log_dir=OUTPUT_DIR,
            timestamp=DOWNLOADER_SESSION_TS
        )
    return DOWNLOADER_ISSUE_LOG_PATH


# # (AI註: 將下載器的非致命問題統一寫入單一 session log，避免每種類別各自爆檔)
def append_downloader_issues(section, lines):
    if not lines:
        return

    append_issue_log(
        get_downloader_issue_log_path(),
        [f"[{section}] {line}" for line in lines]
    )

def get_market_last_date():
    print("🕵️‍♂️ 正在確認最新交易日...")
    try:
        search_start = (datetime.now() - timedelta(days=15)).strftime("%Y-%m-%d")
        loader = get_finmind_loader()
        df = loader.get_data(dataset='TaiwanStockPrice', data_id='0050', start_date=search_start)
        if df is not None and not df.empty:
            df.columns = [c.lower() for c in df.columns]
            actual_date = str(df['date'].max()).split(' ')[0]
            print(f"📅 台股最新交易日 (FinMind) 為: {actual_date}")
            return actual_date
    except EXPECTED_MARKET_DATE_EXCEPTIONS as e:
        append_downloader_issues("最新交易日(FinMind)失敗", [f"{type(e).__name__}: {e}"])
        print(f"⚠️ FinMind 日期獲取異常: {type(e).__name__}: {e}")

    print("🔄 啟動備援方案 (YFinance) 獲取交易日...")
    try:
        yf = get_yfinance_module()
        ticker = yf.Ticker("0050.TW")
        hist = ticker.history(period="5d")
        if not hist.empty:
            actual_date = hist.index[-1].strftime("%Y-%m-%d")
            print(f"📅 台股最新交易日 (YF備援) 為: {actual_date}")
            return actual_date
    except EXPECTED_MARKET_DATE_EXCEPTIONS as e:
        append_downloader_issues("最新交易日(YF備援)失敗", [f"{type(e).__name__}: {e}"])
        print(f"⚠️ YFinance 備援失敗: {type(e).__name__}: {e}")

    fallback_date = datetime.now()
    if fallback_date.hour < 14:
        fallback_date -= timedelta(days=1)

    while fallback_date.weekday() >= 5:
        fallback_date -= timedelta(days=1)

    fallback_str = fallback_date.strftime("%Y-%m-%d")
    print(f"⚠️ 無法取得精準日期，使用智能推算平日備用日期: {fallback_str}")
    return fallback_str

def get_or_update_universe():
    ensure_runtime_dirs()

    if os.path.exists(LIST_FILE):
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(LIST_FILE))
        if datetime.now() - file_mod_time < timedelta(days=RESCAN_DAYS):
            with open(LIST_FILE, 'r') as f:
                cached_tickers = [line.strip() for line in f if line.strip()]
            if cached_tickers:
                print(f"✅ 名單有效 (更新於: {file_mod_time.strftime('%Y-%m-%d')})，直接讀取。")
                return cached_tickers
            print("⚠️ universe 快取為空，重新海選。")

    print(f"🕵️‍♂️ 啟動全市場海選 (市值 > {MIN_MARKET_CAP/1e8:.0f}億 且 成交量 > {MIN_VOLUME/10000:.0f}萬)...")

    urls = [
        "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2",
        "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4"
    ]

    tickers_info = []
    universe_fetch_errors = []

    for url in urls:
        try:
            res = requests.get(url, timeout=REQUEST_TIMEOUT_SEC)
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

        except EXPECTED_UNIVERSE_FETCH_EXCEPTIONS as e:
            universe_fetch_errors.append(f"{url} -> {type(e).__name__}: {e}")
            if VERBOSE_UNIVERSE_FETCH_ERRORS:
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
        yf = get_yfinance_module()
    except EXPECTED_SCREENING_EXCEPTIONS as e:
        screening_errors.append(("__INIT__", "yfinance", f"{type(e).__name__}: {e}"))
        screening_log_lines = [f"{sid} ({yf_t}) -> {err}" for sid, yf_t, err in screening_errors]
        append_downloader_issues("快篩失敗", screening_log_lines)
        raise RuntimeError(f"快篩初始化失敗：{type(e).__name__}: {e}") from e

    for i, item in enumerate(tickers_info):
        pct = ((i + 1) / total_check) * 100
        yf_t, sid, is_etf = item['yf_ticker'], item['sid'], item['is_etf']

        print(f"\r🔍 快篩進度: [{i+1:>4}/{total_check} | {pct:>5.1f}%] {yf_t:<8} ", end="", flush=True)
        try:
            info = yf.Ticker(yf_t).fast_info
            last_volume = info.get('lastVolume', 0)
            market_cap = info.get('marketCap', 0)

            if last_volume >= MIN_VOLUME:
                if is_etf or market_cap >= MIN_MARKET_CAP:
                    qualified_tickers.append(sid)

        except EXPECTED_SCREENING_EXCEPTIONS as e:
            screening_errors.append((sid, yf_t, f"{type(e).__name__}: {e}"))
        time.sleep(YF_SCREEN_SLEEP_SEC)

    qualified_tickers = list(dict.fromkeys(qualified_tickers))

    with open(LIST_FILE, 'w') as f:
        for t in qualified_tickers:
            f.write(f"{t}\n")

    print(f"\n🎉 海選完畢！共 {len(qualified_tickers)} 檔入選。")

    if universe_fetch_errors:
        append_downloader_issues("名單來源失敗", universe_fetch_errors)
        print(f"⚠️ 名單來源失敗 {len(universe_fetch_errors)} 筆，詳細已寫入: {get_downloader_issue_log_path()}")

    if screening_errors:
        screening_log_lines = [f"{sid} ({yf_t}) -> {err}" for sid, yf_t, err in screening_errors]
        append_downloader_issues("快篩失敗", screening_log_lines)
        print(f"⚠️ 快篩失敗 {len(screening_errors)} 檔，詳細已寫入: {get_downloader_issue_log_path()}")

    return qualified_tickers

def smart_download_vip_data(tickers, market_last_date, verbose=True):
    ensure_runtime_dirs()
    total = len(tickers)

    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    vprint(f"\n💎 啟動 VIP 庫更新 (目標: {total} 檔)")
    vprint("-" * 65)

    download_errors = []
    last_date_check_errors = []
    count_success = 0
    count_skipped_latest = 0

    for i, sid in enumerate(tickers, 1):
        file_path = os.path.join(SAVE_DIR, f"{sid}.csv")

        if os.path.exists(file_path):
            try:
                # # (AI註: 修復 1 - 移除檔案修改時間判斷，嚴格依賴 CSV 內最後一筆日期，避免假更新跳過)
                last_date_in_csv = str(pd.read_csv(file_path, index_col=0).tail(1).index[0]).split(' ')[0]
                if last_date_in_csv == market_last_date:
                    count_skipped_latest += 1
                    vprint(
                        f"\r⏳ [{i:03d}/{total:03d}] 成功:{count_success:>4} | 跳過:{count_skipped_latest:>4} | "
                        f"失敗:{len(download_errors):>4} | {sid:<6} 已最新",
                        end="",
                        flush=True
                    )
                    continue
            except EXPECTED_LAST_DATE_CHECK_EXCEPTIONS as e:
                last_date_check_errors.append(f"{sid}: {type(e).__name__}: {e}")
                if VERBOSE_LAST_DATE_CHECK_ERRORS:
                    vprint(f"\n⚠️ {sid} 檢查最後日期發生錯誤，將強制重抓: {type(e).__name__}: {e}")

        vprint(
            f"\r⚡ [{i:03d}/{total:03d}] 成功:{count_success:>4} | 跳過:{count_skipped_latest:>4} | "
            f"失敗:{len(download_errors):>4} | 正在下載 {sid:<6}",
            end="",
            flush=True
        )
        try:
            loader = get_finmind_loader()
            df = loader.get_data(dataset=FINMIND_PRICE_DATASET, data_id=sid, start_date="1990-01-01")
            if df is None or df.empty:
                raise ValueError("FinMind 回傳空資料")

            df.columns = [c.capitalize() for c in df.columns]
            df = df.rename(columns={"Trading_volume": "Volume", "Max": "High", "Min": "Low"})

            if 'Date' not in df.columns:
                raise KeyError("下載資料缺少 Date 欄位")

            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df = df.sort_index()

            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                raise KeyError(f"缺少必要欄位: {missing_cols}")

            df[required_cols].to_csv(file_path)
            count_success += 1
            time.sleep(FINMIND_DOWNLOAD_SLEEP_SEC)

        except EXPECTED_DOWNLOAD_EXCEPTIONS as e:
            download_errors.append((sid, f"{type(e).__name__}: {e}"))
            if VERBOSE_DOWNLOAD_ERRORS:
                vprint(f"\n❌ {sid} 失敗: {type(e).__name__}: {e}")

    vprint("\n" + "-" * 65)
    vprint(
        f"🏆 本地尊爵資料庫更新完畢！成功 {count_success} 檔 | "
        f"已最新跳過 {count_skipped_latest} 檔 | "
        f"最後日期檢查失敗 {len(last_date_check_errors)} 檔 | "
        f"下載失敗 {len(download_errors)} 檔"
    )

    if last_date_check_errors:
        append_downloader_issues("最後日期檢查失敗", last_date_check_errors)

    if download_errors:
        download_log_lines = [f"{sid} -> {err}" for sid, err in download_errors]
        append_downloader_issues("下載失敗", download_log_lines)

    if last_date_check_errors or download_errors:
        vprint(f"⚠️ 非致命問題詳細已寫入: {get_downloader_issue_log_path()}")

if __name__ == "__main__":
    print(f"🤖 智能量化建庫系統 (VIP版) 啟動 | {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    market_date = get_market_last_date()
    target_tickers = get_or_update_universe()

    if not target_tickers:
        raise RuntimeError("未取得任何可下載標的；請檢查 universe 快篩條件、資料來源或快取內容。")

    smart_download_vip_data(target_tickers, market_date)