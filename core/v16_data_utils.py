import pandas as pd

LOAD_DATA_MIN_ROWS = 50
LOAD_DATA_REQUIRED_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']


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
        (working['Volume'] < 0) |
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