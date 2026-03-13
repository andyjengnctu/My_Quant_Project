import os
import pandas as pd

LOAD_DATA_MIN_ROWS = 50
LOAD_DATA_REQUIRED_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']
BACKTEST_EXTRA_MIN_ROWS = 10


# # (AI註: 單一真理來源 - 統一各類 lookback 對資料長度的最低要求，
# # (AI註: 避免 optimizer / sim / scanner / tools 只看 high_len，導致其他 rolling 指標長度不足卻靜默通過)
def get_required_min_rows_from_lookbacks(*lookbacks, base_min_rows=LOAD_DATA_MIN_ROWS, extra_rows=BACKTEST_EXTRA_MIN_ROWS):
    resolved_lookbacks = [int(x or 0) for x in lookbacks]
    max_lookback = max(resolved_lookbacks) if resolved_lookbacks else 0
    return max(int(base_min_rows), max_lookback + int(extra_rows))


# # (AI註: 向後相容舊 API；外部模組仍可用舊名稱，實作統一委派到新版 lookbacks 版本)
def get_required_min_rows_from_high_len(high_len, base_min_rows=LOAD_DATA_MIN_ROWS, extra_rows=BACKTEST_EXTRA_MIN_ROWS):
    return get_required_min_rows_from_lookbacks(
        high_len,
        base_min_rows=base_min_rows,
        extra_rows=extra_rows
    )

# # (AI註: 單一真理來源 - 將 CSV 檔名正規化成 ticker，統一支援 2330.csv / TV_Data_Full_2330.csv)
def normalize_ticker_from_csv_filename(file_name):
    base_name = os.path.basename(str(file_name)).strip()
    stem, ext = os.path.splitext(base_name)
    if ext.lower() != '.csv':
        return ''
    if stem.startswith('TV_Data_Full_'):
        stem = stem[len('TV_Data_Full_'):]
    return stem.strip()


# # (AI註: 目前 downloader 輸出是 {ticker}.csv，若同 ticker 同時存在兩種命名，
# # (AI註: 優先保留 {ticker}.csv，並回報重複檔名，禁止 portfolio / scanner 靜默覆寫或重複掃描)
def discover_unique_csv_inputs(data_dir):
    selected_files = {}
    duplicate_issue_lines = []

    for file_name in sorted(os.listdir(data_dir)):
        if not file_name.lower().endswith('.csv'):
            continue

        ticker = normalize_ticker_from_csv_filename(file_name)
        if not ticker:
            continue

        existing_file = selected_files.get(ticker)
        if existing_file is None:
            selected_files[ticker] = file_name
            continue

        preferred_name = f"{ticker}.csv"
        existing_is_preferred = (os.path.basename(existing_file) == preferred_name)
        new_is_preferred = (os.path.basename(file_name) == preferred_name)

        if new_is_preferred and not existing_is_preferred:
            kept_file, dropped_file = file_name, existing_file
            selected_files[ticker] = file_name
        else:
            kept_file, dropped_file = existing_file, file_name

        duplicate_issue_lines.append(
            f"[重複檔名] {ticker}: 同時存在 {existing_file} 與 {file_name}，"
            f"保留 {kept_file}，忽略 {dropped_file}"
        )

    discovered = [
        (ticker, os.path.join(data_dir, file_name))
        for ticker, file_name in sorted(selected_files.items())
    ]
    return discovered, duplicate_issue_lines

# # (AI註: 單一真理來源 - 由策略參數物件推導最低資料長度需求)
def get_required_min_rows(params, base_min_rows=LOAD_DATA_MIN_ROWS, extra_rows=BACKTEST_EXTRA_MIN_ROWS):
    high_len = getattr(params, "high_len", 0)
    atr_len = getattr(params, "atr_len", 0)
    bb_len = getattr(params, "bb_len", 0) if getattr(params, "use_bb", False) else 0
    kc_len = getattr(params, "kc_len", 0) if getattr(params, "use_kc", False) else 0
    vol_short_len = getattr(params, "vol_short_len", 0) if getattr(params, "use_vol", False) else 0
    vol_long_len = getattr(params, "vol_long_len", 0) if getattr(params, "use_vol", False) else 0

    return get_required_min_rows_from_lookbacks(
        high_len,
        atr_len,
        bb_len,
        kc_len,
        vol_short_len,
        vol_long_len,
        base_min_rows=base_min_rows,
        extra_rows=extra_rows
    )


# # (AI註: 單一真理來源 - 批次參數共用同一個最低資料長度門檻，避免預載 raw cache 與 trial / 候選評估口徑分裂)
def get_max_required_min_rows(params_list, base_min_rows=LOAD_DATA_MIN_ROWS, extra_rows=BACKTEST_EXTRA_MIN_ROWS):
    max_needed = int(base_min_rows)
    for params in params_list:
        max_needed = max(
            max_needed,
            get_required_min_rows(params, base_min_rows=base_min_rows, extra_rows=extra_rows)
        )
    return max_needed


# # (AI註: 單一真理來源 - 統一所有模組的 OHLCV 清洗規則，禁止 optimizer / sim / scanner / tools 各自為政)
def sanitize_ohlcv_dataframe(df, ticker, min_rows=LOAD_DATA_MIN_ROWS, required_cols=LOAD_DATA_REQUIRED_COLS):
    working = df.copy()
    working.columns = [c.capitalize() for c in working.columns]

    date_col = 'Time' if 'Time' in working.columns else 'Date'
    if date_col not in working.columns:
        raise KeyError("缺少 Time / Date 欄位")

    missing_cols = [c for c in required_cols if c not in working.columns]
    if missing_cols:
        raise KeyError(f"缺少必要欄位: {missing_cols}")

    for col in required_cols:
        working[col] = pd.to_numeric(working[col], errors='coerce')

    working[date_col] = pd.to_datetime(working[date_col], errors='coerce')

    invalid_mask = (
        working[date_col].isna() |
        working['Open'].isna() |
        working['High'].isna() |
        working['Low'].isna() |
        working['Close'].isna() |
        working['Volume'].isna() |
        (working['Open'] <= 0) |
        (working['High'] <= 0) |
        (working['Low'] <= 0) |
        (working['Close'] <= 0) |
        # # (AI註: 零成交量日不可成交，直接排除，避免回測在不可交易日買入/賣出)
        (working['Volume'] <= 0) |
        (working['High'] < working[['Open', 'Low', 'Close']].max(axis=1)) |
        (working['Low'] > working[['Open', 'High', 'Close']].min(axis=1))
    )

    invalid_row_count = int(invalid_mask.sum())
    if invalid_row_count > 0:
        working = working.loc[~invalid_mask].copy()

    if working.empty:
        raise ValueError(f"{ticker} 清洗後無有效資料")

    working.set_index(date_col, inplace=True)
    working.sort_index(inplace=True)

    duplicate_date_count = int(working.index.duplicated(keep='last').sum())
    if duplicate_date_count > 0:
        working = working[~working.index.duplicated(keep='last')]

    if len(working) < min_rows:
        raise ValueError(f"有效資料不足: 清洗後僅剩 {len(working)} 列")

    stats = {
        'invalid_row_count': invalid_row_count,
        'duplicate_date_count': duplicate_date_count,
        'dropped_row_count': invalid_row_count + duplicate_date_count,
    }
    return working, stats