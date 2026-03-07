import pandas as pd
import os
import time
import requests
import yfinance as yf
from datetime import datetime, timedelta
from io import StringIO
from FinMind.data import DataLoader

API_TOKEN = os.getenv("FINMIND_API_TOKEN", "")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "tw_stock_data_vip")
LIST_FILE = os.path.join(SAVE_DIR, "universe_list.txt")

MIN_VOLUME = 1_000_000
MIN_MARKET_CAP = 10_000_000_000
RESCAN_DAYS = 7

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

dl = DataLoader()
if API_TOKEN:
    dl.login_by_token(api_token=API_TOKEN)

def get_market_last_date():
    print("🕵️‍♂️ 正在確認最新交易日...")
    try:
        search_start = (datetime.now() - timedelta(days=15)).strftime("%Y-%m-%d")
        df = dl.get_data(dataset='TaiwanStockPrice', data_id='0050', start_date=search_start)
        if df is not None and not df.empty:
            df.columns = [c.lower() for c in df.columns]
            actual_date = str(df['date'].max()).split(' ')[0]
            print(f"📅 台股最新交易日 (FinMind) 為: {actual_date}")
            return actual_date
    except Exception as e:
        print(f"⚠️ FinMind 日期獲取異常: {e}")

    print("🔄 啟動備援方案 (YFinance) 獲取交易日...")
    try:
        ticker = yf.Ticker("0050.TW")
        hist = ticker.history(period="5d")
        if not hist.empty:
            actual_date = hist.index[-1].strftime("%Y-%m-%d")
            print(f"📅 台股最新交易日 (YF備援) 為: {actual_date}")
            return actual_date
    except Exception as e:
        print(f"⚠️ YFinance 備援失敗: {e}")
        
    # 🌟 修復 3：智能推算最近平日，杜絕假日被當成交易日的 Bug
    fallback_date = datetime.now()
    # 若現在早於下午兩點，盤後資料可能還沒出，保守往前退一天
    if fallback_date.hour < 14:
        fallback_date -= timedelta(days=1)
        
    # 自動避開週六(5)與週日(6)，往前尋找最近的平日
    while fallback_date.weekday() >= 5:
        fallback_date -= timedelta(days=1)
        
    fallback_str = fallback_date.strftime("%Y-%m-%d")
    print(f"⚠️ 無法取得精準日期，使用智能推算平日備用日期: {fallback_str}")
    return fallback_str

def get_or_update_universe():
    if os.path.exists(LIST_FILE):
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(LIST_FILE))
        if datetime.now() - file_mod_time < timedelta(days=RESCAN_DAYS):
            print(f"✅ 名單有效 (更新於: {file_mod_time.strftime('%Y-%m-%d')})，直接讀取。")
            with open(LIST_FILE, 'r') as f:
                return [line.strip() for line in f if line.strip()]

    print(f"🕵️‍♂️ 啟動全市場海選 (市值 > {MIN_MARKET_CAP/1e8:.0f}億 且 成交量 > {MIN_VOLUME/10000:.0f}萬)...")
    
    urls = [
        "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2",
        "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4"
    ]
    
    tickers_info = []
    for url in urls:
        try:
            res = requests.get(url, timeout=10) 
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
        except Exception as e:
            print(f"⚠️ 抓取清單時發生錯誤: {e}")

    qualified_tickers = []
    total_check = len(tickers_info)
    print(f"⏳ 準備快篩 {total_check} 檔純股與 ETF...\n")
    
    for i, item in enumerate(tickers_info):
        pct = ((i + 1) / total_check) * 100
        yf_t, sid, is_etf = item['yf_ticker'], item['sid'], item['is_etf']
        
        print(f"\r🔍 快篩進度: [{i+1:>4}/{total_check} | {pct:>5.1f}%] {yf_t:<8} ", end="", flush=True)
        try:
            info = yf.Ticker(yf_t).fast_info
            if info.get('lastVolume', 0) >= MIN_VOLUME:
                if is_etf or info.get('marketCap', 0) >= MIN_MARKET_CAP:
                    qualified_tickers.append(sid)
        except:
            pass
        time.sleep(0.01)

    with open(LIST_FILE, 'w') as f:
        for t in qualified_tickers:
            f.write(f"{t}\n")
    print(f"\n🎉 海選完畢！共 {len(qualified_tickers)} 檔入選。")
    return qualified_tickers

def smart_download_vip_data(tickers, market_last_date):
    total = len(tickers)
    today_date = datetime.today().date()
    print(f"\n💎 啟動 VIP 庫更新 (目標: {total} 檔)")
    print("-" * 65)

    for i, sid in enumerate(tickers, 1):
        file_path = os.path.join(SAVE_DIR, f"{sid}.csv")
        
        if os.path.exists(file_path):
            file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path)).date()
            if file_mtime == today_date:
                print(f"\r⏩ [{i:03d}/{total:03d}] {sid:<6} 檔案今日已更新，跳過。{' '*15}", end="", flush=True)
                continue
                
            try:
                last_date_in_csv = str(pd.read_csv(file_path, index_col=0).tail(1).index[0]).split(' ')[0]
                if last_date_in_csv == market_last_date:
                    print(f"\r⏩ [{i:03d}/{total:03d}] {sid:<6} 資料已是最新 ({market_last_date})，跳過。{' '*15}", end="", flush=True)
                    continue
            except:
                pass 

        print(f"\r⚡ [{i:03d}/{total:03d}] 正在下載 {sid:<6} ...{' '*15}", end="", flush=True)
        try:
            df = dl.get_data(dataset='TaiwanStockPriceAdj', data_id=sid, start_date="1990-01-01")
            if df is not None and not df.empty:
                df.columns = [c.capitalize() for c in df.columns]
                df = df.rename(columns={"Trading_volume": "Volume", "Max": "High", "Min": "Low"})
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                df = df.sort_index()
                df[['Open', 'High', 'Low', 'Close', 'Volume']].to_csv(file_path)
            time.sleep(0.5) 
        except Exception as e:
            print(f"\n❌ {sid} 失敗: {e}")

    print("\n" + "-" * 65)
    print(f"🏆 本地尊爵資料庫更新完畢！")

if __name__ == "__main__":
    print(f"🤖 智能量化建庫系統 (VIP版) 啟動 | {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    market_date = get_market_last_date()
    target_tickers = get_or_update_universe()
    if target_tickers:
        smart_download_vip_data(target_tickers, market_date)