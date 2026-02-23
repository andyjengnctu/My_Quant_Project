import pandas as pd
import os
import time
import requests
import yfinance as yf
from datetime import datetime, timedelta
from io import StringIO
from FinMind.data import DataLoader
import urllib3
import warnings

# 關閉警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore')

# ==========================================
# 0. 參數與 API 設定
# ==========================================
API_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJkYXRlIjoiMjAyNi0wMi0yMSAxMzozOToyMyIsInVzZXJfaWQiOiJhbmR5amVuZ25jdHUiLCJlbWFpbCI6ImFuZHlqZW5nbmN0dUBnbWFpbC5jb20iLCJpcCI6IjM2LjIyOC4xMTMuMTMxIn0.D97Qj43wskbRDXXbESCTO13wnijIhuClsPeobPhYy3s"
SAVE_DIR = "tw_stock_data_vip"
LIST_FILE = os.path.join(SAVE_DIR, "universe_list.txt")

MIN_VOLUME = 1_000_000
MIN_MARKET_CAP = 10_000_000_000
RESCAN_DAYS = 7

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

dl = DataLoader()
dl.login_by_token(api_token=API_TOKEN)

# ==========================================
# 1. 取得台股「真實最後交易日」 (破解假日陷阱)
# ==========================================
def get_market_last_date():
    print("🕵️‍♂️ 正在向證交所確認最新交易日...")
    search_start = (datetime.now() - timedelta(days=15)).strftime("%Y-%m-%d")
    df = dl.get_data(dataset='TaiwanStockPrice', data_id='0050', start_date=search_start)
    if df is not None and not df.empty:
        df.columns = [c.lower() for c in df.columns]
        actual_date = str(df['date'].max()).split(' ')[0]
        print(f"📅 台股最新交易日為: {actual_date}")
        return actual_date
    return datetime.today().strftime("%Y-%m-%d")

# ==========================================
# 2. 智慧海選 (YF 掃描 + CE/ES 精準過濾)
# ==========================================
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
            res = requests.get(url, verify=False) 
            df = pd.read_html(StringIO(res.text))[0] 
            df.columns = df.iloc[0]
            df = df.iloc[2:]
            
            # 🔥 精準過濾：ES=普通股, CE=ETF。徹底排除幾萬檔權證！
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
            pass

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

# ==========================================
# 3. 尊爵下載 (三重防禦跳過)
# ==========================================
def smart_download_vip_data(tickers, market_last_date):
    total = len(tickers)
    today_date = datetime.today().date()
    print(f"\n💎 啟動 VIP 庫更新 (目標: {total} 檔)")
    print("-" * 65)

    for i, sid in enumerate(tickers, 1):
        file_path = os.path.join(SAVE_DIR, f"{sid}.csv")
        
        # 🛡️ 防禦 1 & 2：檢查修改日期 或 CSV 內部日期
        if os.path.exists(file_path):
            file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path)).date()
            if file_mtime == today_date:
                print(f"\r⏩ [{i:03d}/{total:03d}] {sid:<6} 檔案今日已更新，跳過。{' '*15}", end="", flush=True)
                continue
                
            try:
                # 讀取最後一行檢查日期，破除假日陷阱！
                last_date_in_csv = str(pd.read_csv(file_path, index_col=0).tail(1).index[0]).split(' ')[0]
                if last_date_in_csv == market_last_date:
                    print(f"\r⏩ [{i:03d}/{total:03d}] {sid:<6} 資料已是最新 ({market_last_date})，跳過。{' '*15}", end="", flush=True)
                    continue
            except:
                pass # 若檔案損毀則不跳過，重新下載

        # 📥 執行下載
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

# ==========================================
# 4. 主執行區
# ==========================================
if __name__ == "__main__":
    print(f"🤖 智能量化建庫系統 (VIP版) 啟動 | {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    
    # 抓取大盤最後日期
    market_date = get_market_last_date()
    
    # 執行快篩海選
    target_tickers = get_or_update_universe()
    
    # 執行官方還原資料下載
    if target_tickers:
        smart_download_vip_data(target_tickers, market_date)