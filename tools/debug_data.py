import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from core.v16_data_utils import discover_unique_csv_map

# ==========================================
# 0. 設定區
# ==========================================
VIP_DATA_DIR = os.path.join(BASE_DIR, "tw_stock_data_vip")  # 你的下載路徑
TV_DATA_DIR = os.path.join(BASE_DIR, "testing_csv")         # 你的對帳基準路徑


def resolve_compare_csv_path(data_dir, ticker, preferred_file_names):
    for file_name in preferred_file_names:
        file_path = os.path.join(data_dir, file_name)
        if os.path.exists(file_path):
            return file_path

    csv_map, _duplicate_file_issue_lines = discover_unique_csv_map(data_dir)
    if ticker in csv_map:
        return csv_map[ticker]

    raise FileNotFoundError(f"找不到 {ticker} 的 CSV。資料夾={data_dir}")


def verify_local_database():
    ticker = input("\n🔍 請輸入要驗證的股票代號 (例如 0050、2330，輸入 q 離開): ").strip().upper()
    if ticker == 'Q':
        return False

    # 1. 檢查路徑與檔名（優先使用各自慣用命名，若不存在再回退到唯一解析）
    try:
        vip_path = resolve_compare_csv_path(
            VIP_DATA_DIR,
            ticker,
            [f"{ticker}.csv", f"TV_Data_Full_{ticker}.csv"]
        )
    except FileNotFoundError as e:
        print(f"❌ 找不到本地 VIP 資料: {e}")
        return True

    try:
        tv_path = resolve_compare_csv_path(
            TV_DATA_DIR,
            ticker,
            [f"TV_Data_Full_{ticker}.csv", f"{ticker}.csv"]
        )
    except FileNotFoundError as e:
        print(f"❌ 找不到 TV 對帳檔: {e}")
        return True

    print(f"📥 正在比對 {ticker} 的歷史數據...")

    # 2. 讀取與標準化
    try:
        df_vip = pd.read_csv(vip_path, index_col=0, parse_dates=True)
        df_tv = pd.read_csv(tv_path, index_col=0, parse_dates=True)
        df_tv.columns = [c.capitalize() for c in df_tv.columns]
    except Exception as e:
        print(f"❌ 讀取 CSV 失敗: {e}")
        return True

    # 3. 合併與計算價差
    df_merged = df_tv.join(df_vip, how='inner', lsuffix='_TV', rsuffix='_VIP')

    if df_merged.empty:
        print("⚠️ 雙方日期完全沒有重疊，請檢查資料時間範圍。")
        return True

    df_merged['Price_Diff'] = abs(df_merged['Close_TV'] - df_merged['Close_VIP'])
    df_merged['Year'] = df_merged.index.year
    yearly_report = df_merged.groupby('Year').agg(Max_Err=('Price_Diff', 'max')).sort_index(ascending=False)

    print("\n" + "═" * 80)
    print(f" 💎 [{ticker}] 本地 VIP 庫 vs TV (ADJ) 數據一致性報告")
    print("═" * 80)
    print(f"{'年份':<10} | {'最大誤差 (元)':<20} | {'狀態'}")
    print("-" * 80)

    all_perfect = True
    for year, row in yearly_report.iterrows():
        is_ok = row['Max_Err'] <= 0.5
        status = "✅ 完美對齊" if is_ok else f"❌ 誤差 {row['Max_Err']:.2f}"
        if not is_ok:
            all_perfect = False
        print(f"{year:<10} | {row['Max_Err']:<20.2f} | {status}")

    print("═" * 80)
    print(f"📅 [{ticker}] 最新 3 筆數據詳情：")
    print(f"{'日期':<12} | {'TV 收盤':<15} | {'VIP 本地收盤'}")
    print("-" * 80)
    for date, row in df_merged.tail(3).iterrows():
        print(f"{date.strftime('%Y-%m-%d'):<12} | {row['Close_TV']:<15.2f} | {row['Close_VIP']:<15.2f}")
    print("═" * 80)

    if all_perfect:
        print(f"🏆 恭喜！[{ticker}] 本地資料庫與 TV CSV 完美契合，可放心回測。")
    else:
        print("💡 提示：若早期有微小價差，通常是不同平台小數點進位方式不同所致。")

    return True


if __name__ == "__main__":
    while True:
        if not verify_local_database():
            break
