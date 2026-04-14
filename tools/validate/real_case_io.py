import os

import pandas as pd

from core.data_utils import get_required_min_rows, sanitize_ohlcv_dataframe


def resolve_csv_path(project_root, data_dir, csv_map_getter, ticker):
    csv_map = csv_map_getter()
    if ticker in csv_map:
        return csv_map[ticker]

    candidates = [
        os.path.join(data_dir, f"TV_Data_Full_{ticker}.csv"),
        os.path.join(data_dir, f"{ticker}.csv"),
        os.path.join(project_root, f"TV_Data_Full_{ticker}.csv"),
        os.path.join(project_root, f"{ticker}.csv"),
    ]
    raise FileNotFoundError(f"找不到 {ticker} 的 CSV。已檢查: {candidates}")


def load_clean_df_from_path(file_path, ticker, params):
    raw_df = pd.read_csv(file_path)
    min_rows_needed = get_required_min_rows(params)
    df, sanitize_stats = sanitize_ohlcv_dataframe(raw_df, ticker, min_rows=min_rows_needed)
    return file_path, df, sanitize_stats



def load_clean_df(project_root, data_dir, csv_map_getter, ticker, params, *, resolved_file_path=None):
    file_path = resolved_file_path or resolve_csv_path(project_root, data_dir, csv_map_getter, ticker)
    return load_clean_df_from_path(file_path, ticker, params)
