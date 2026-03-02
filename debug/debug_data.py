import pandas as pd
import os

# ==========================================
# 0. 設定區
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIP_DATA_DIR = "tw_stock_data_vip"  # 你的下載路徑
TV_DATA_DIR = os.path.join(BASE_DIR, "tw_stock_data_vip")        # 你的對帳基準路徑

def verify_local_database():
    ticker = input("\n🔍 請輸入要驗證的股票代號 (例如 0050、2330，輸入 q 離開): ").strip().upper()
    if ticker == 'Q':
        return False

    # 1. 檢查路徑與檔名 (對齊你的 TV 格式)
    vip_path = os.path.join(VIP_DATA_DIR, f"{ticker}.csv")
    tv_path = os.path.join(TV_DATA_DIR, f"TV_Data_Full_{ticker}.csv")

    if not os.path.exists(vip_path):
        print(f"❌ 找不到本地 VIP 資料: {vip_path}")
        return True

    if not os.path.exists(tv_path):
        print(f"❌ 找不到 TV 對帳檔: {tv_path}")
        print(f"💡 提示：請確認檔名是否完全符合 TV_Data_Full_{ticker}.csv")
        return True

    print(f"📥 正在比對 {ticker} 的歷史數據...")

    # 2. 讀取與標準化
    try:
        # 讀取本地 VIP 資料
        df_vip = pd.read_csv(vip_path, index_col=0, parse_dates=True)
        # 讀取 TradingView 對帳資料
        df_tv = pd.read_csv(tv_path, index_col=0, parse_dates=True)
        
        # 統一 TV 欄位大寫
        df_tv.columns = [c.capitalize() for c in df_tv.columns]
    except Exception as e:
        print(f"❌ 讀取 CSV 失敗: {e}")
        return True

    # 3. 合併與計算價差
    df_merged = df_tv.join(df_vip, how='inner', lsuffix='_TV', rsuffix='_VIP')
    
    if df_merged.empty:
        print("⚠️ 雙方日期完全沒有重疊，請檢查資料時間範圍。")
        return True

    # 計算收盤價絕對誤差
    df_merged['Price_Diff'] = abs(df_merged['Close_TV'] - df_merged['Close_VIP'])
    
    # 按年份統計
    df_merged['Year'] = df_merged.index.year
    yearly_report = df_merged.groupby('Year').agg(Max_Err=('Price_Diff', 'max')).sort_index(ascending=False)

    # ==========================================
    # 4. 輸出檢驗報表
    # ==========================================
    print("\n" + "═"*80)
    print(f" 💎 [{ticker}] 本地 VIP 庫 vs TV (ADJ) 數據一致性報告")
    print("═"*80)
    print(f"{'年份':<10} | {'最大誤差 (元)':<20} | {'狀態'}")
    print("-" * 80)

    all_perfect = True
    for year, row in yearly_report.iterrows():
        # 0.5 元以內視為四捨五入正常誤差
        is_ok = row['Max_Err'] <= 0.5
        status = "✅ 完美對齊" if is_ok else f"❌ 誤差 {row['Max_Err']:.2f}"
        if not is_ok: all_perfect = False
        print(f"{year:<10} | {row['Max_Err']:<20.2f} | {status}")

    print("═"*80)
    
    # 詳細數值抽樣
    print(f"📅 [{ticker}] 最新 3 筆數據詳情：")
    print(f"{'日期':<12} | {'TV 收盤':<15} | {'VIP 本地收盤'}")
    print("-" * 80)
    for date, row in df_merged.tail(3).iterrows():
        print(f"{date.strftime('%Y-%m-%d'):<12} | {row['Close_TV']:<15.2f} | {row['Close_VIP']:<15.2f}")
    print("═"*80)

    if all_perfect:
        print(f"🏆 恭喜！[{ticker}] 本地資料庫與 TV CSV 完美契合，可放心回測。")
    else:
        print("💡 提示：若早期有微小價差，通常是不同平台小數點進位方式不同所致。")

    return True

if __name__ == "__main__":
    while True:
        if not verify_local_database():
            break