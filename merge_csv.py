import pandas as pd

print("⏳ 啟動歷史數據縫合手術...")

try:
    # 1. 讀取兩份 CSV 檔案
    print("讀取新舊資料中...")
    df_old = pd.read_csv('TV_Data_old_2317.csv')
    df_new = pd.read_csv('TV_Data_new_2317.csv')

    # 2. 將兩份資料上下合併 (垂直拼接)
    df_combined = pd.concat([df_old, df_new], ignore_index=True)
    print(f"合併前總筆數：{len(df_combined)} 筆")

    # 3. 統一轉換時間格式，確保 Python 認得
    df_combined['time'] = pd.to_datetime(df_combined['time'])

    # 4. 剔除重複的 K 線 (如果同一天有兩筆資料，保留最新的)
    df_combined.drop_duplicates(subset=['time'], keep='last', inplace=True)

    # 5. 按照時間先後順序「重新排序」 (從最舊排到最新)
    df_combined.sort_values(by='time', ascending=True, inplace=True)
    print(f"去重並排序後總筆數：{len(df_combined)} 筆")

    # 6. 存成一份全新的終極檔案
    final_filename = 'TV_Data_Full_2820.csv'
    df_combined.to_csv(final_filename, index=False)
    
    print("\n" + "="*40)
    print(f"✅ 縫合成功！已儲存為 {final_filename}")
    print(f"📅 數據起點：{df_combined['time'].iloc[0].strftime('%Y-%m-%d')}")
    print(f"📅 數據終點：{df_combined['time'].iloc[-1].strftime('%Y-%m-%d')}")
    print("="*40)

except FileNotFoundError as e:
    print(f"❌ 找不到檔案，請確認檔名是否正確：{e}")