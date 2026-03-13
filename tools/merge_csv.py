import os
import sys
import pandas as pd


def detect_datetime_column(df):
    for col in ["time", "Time", "date", "Date"]:
        if col in df.columns:
            return col
    raise KeyError("缺少 time/Time/date/Date 欄位")


def merge_csv_files(old_csv_path, new_csv_path, output_csv_path):
    print("⏳ 啟動歷史數據縫合手術...")

    print("讀取新舊資料中...")
    df_old = pd.read_csv(old_csv_path)
    df_new = pd.read_csv(new_csv_path)

    dt_col_old = detect_datetime_column(df_old)
    dt_col_new = detect_datetime_column(df_new)
    if dt_col_old != dt_col_new:
        raise ValueError(f"時間欄位不一致: old={dt_col_old}, new={dt_col_new}")

    dt_col = dt_col_old
    df_combined = pd.concat([df_old, df_new], ignore_index=True)
    print(f"合併前總筆數：{len(df_combined)} 筆")

    df_combined[dt_col] = pd.to_datetime(df_combined[dt_col], errors='raise')
    df_combined.drop_duplicates(subset=[dt_col], keep='last', inplace=True)
    df_combined.sort_values(by=dt_col, ascending=True, inplace=True)
    print(f"去重並排序後總筆數：{len(df_combined)} 筆")

    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)) or ".", exist_ok=True)
    df_combined.to_csv(output_csv_path, index=False)

    print("\n" + "=" * 40)
    print(f"✅ 縫合成功！已儲存為 {output_csv_path}")
    print(f"📅 數據起點：{df_combined[dt_col].iloc[0].strftime('%Y-%m-%d')}")
    print(f"📅 數據終點：{df_combined[dt_col].iloc[-1].strftime('%Y-%m-%d')}")
    print("=" * 40)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        script_name = os.path.basename(sys.argv[0])
        raise SystemExit(
            f"用法: python {script_name} <old_csv_path> <new_csv_path> <output_csv_path>"
        )

    old_csv_path, new_csv_path, output_csv_path = sys.argv[1:4]
    merge_csv_files(old_csv_path, new_csv_path, output_csv_path)
