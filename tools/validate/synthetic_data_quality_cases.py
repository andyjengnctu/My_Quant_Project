from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from core.data_utils import sanitize_ohlcv_dataframe
from tools.validate.real_case_io import load_clean_df

from .checks import add_check


def validate_sanitize_ohlcv_expected_behavior_case(_base_params):
    case_id = "DATA_QUALITY_SANITIZE_EXPECTED"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    dirty_df = pd.DataFrame(
        [
            {"Date": "2024-01-03", "Open": 13, "High": 15, "Low": 12, "Close": 14, "Volume": 130},
            {"Date": "2024-01-01", "Open": 10, "High": 11, "Low": 9, "Close": 10.5, "Volume": 100},
            {"Date": "2024-01-02", "Open": 11, "High": 12, "Low": 10, "Close": 11.5, "Volume": 110},
            {"Date": "2024-01-02", "Open": 11, "High": 12, "Low": 10, "Close": 11.5, "Volume": 111},
            {"Date": "2024-01-04", "Open": 14, "High": 15, "Low": 13, "Close": 14.5, "Volume": -5},
            {"Date": "2024-01-05", "Open": 15, "High": 16, "Low": 14, "Close": None, "Volume": 150},
            {"Date": "2024-01-06", "Open": 16, "High": 15, "Low": 14, "Close": 16.5, "Volume": 160},
        ]
    )

    cleaned, stats = sanitize_ohlcv_dataframe(dirty_df, "DQ01", min_rows=4)

    add_check(results, "synthetic_data_quality", case_id, "expected_row_count_after_cleaning", 4, len(cleaned))
    add_check(results, "synthetic_data_quality", case_id, "expected_index_sorted", True, cleaned.index.is_monotonic_increasing)
    add_check(results, "synthetic_data_quality", case_id, "expected_duplicate_dates_removed", 1, stats["duplicate_date_count"])
    add_check(results, "synthetic_data_quality", case_id, "expected_invalid_rows_removed", 2, stats["invalid_row_count"])
    add_check(results, "synthetic_data_quality", case_id, "expected_negative_volume_corrected", 1, stats["negative_volume_corrected_count"])
    add_check(results, "synthetic_data_quality", case_id, "expected_zero_volume_retained", 1, stats["zero_volume_row_count"])
    add_check(results, "synthetic_data_quality", case_id, "expected_dropped_row_count", 3, stats["dropped_row_count"])
    add_check(results, "synthetic_data_quality", case_id, "expected_zero_volume_row_still_present", 0.0, float(cleaned.loc[pd.Timestamp("2024-01-04"), "Volume"]), tol=1e-12)
    add_check(results, "synthetic_data_quality", case_id, "expected_columns_preserved", ["Open", "High", "Low", "Close", "Volume"], list(cleaned.columns))

    summary["cleaned_rows"] = len(cleaned)
    return results, summary


def validate_sanitize_ohlcv_failfast_case(_base_params):
    case_id = "DATA_QUALITY_SANITIZE_FAILFAST"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    missing_date_df = pd.DataFrame(
        [{"Open": 10, "High": 11, "Low": 9, "Close": 10.5, "Volume": 100}]
    )
    try:
        sanitize_ohlcv_dataframe(missing_date_df, "DQ02", min_rows=1)
        add_check(results, "synthetic_data_quality", case_id, "missing_date_column_rejected", True, False)
    except KeyError as exc:
        add_check(results, "synthetic_data_quality", case_id, "missing_date_column_marker", True, "缺少 Time / Date 欄位" in str(exc))

    missing_close_df = pd.DataFrame(
        [{"Date": "2024-01-01", "Open": 10, "High": 11, "Low": 9, "Volume": 100}]
    )
    try:
        sanitize_ohlcv_dataframe(missing_close_df, "DQ03", min_rows=1)
        add_check(results, "synthetic_data_quality", case_id, "missing_required_cols_rejected", True, False)
    except KeyError as exc:
        message = str(exc)
        add_check(results, "synthetic_data_quality", case_id, "missing_required_cols_marker", True, "缺少必要欄位" in message)
        add_check(results, "synthetic_data_quality", case_id, "missing_required_cols_lists_close", True, "Close" in message)

    all_invalid_df = pd.DataFrame(
        [
            {"Date": "2024-01-01", "Open": 10, "High": 9, "Low": 8, "Close": 10.5, "Volume": 100},
            {"Date": "2024-01-02", "Open": 11, "High": 12, "Low": 10, "Close": None, "Volume": 100},
        ]
    )
    try:
        sanitize_ohlcv_dataframe(all_invalid_df, "DQ04", min_rows=1)
        add_check(results, "synthetic_data_quality", case_id, "all_invalid_rows_rejected", True, False)
    except ValueError as exc:
        add_check(results, "synthetic_data_quality", case_id, "all_invalid_rows_marker", True, "清洗後無有效資料" in str(exc))

    insufficient_df = pd.DataFrame(
        [
            {"Date": "2024-01-01", "Open": 10, "High": 11, "Low": 9, "Close": 10.5, "Volume": 100},
            {"Date": "2024-01-02", "Open": 11, "High": 12, "Low": 10, "Close": 11.5, "Volume": 110},
        ]
    )
    try:
        sanitize_ohlcv_dataframe(insufficient_df, "DQ05", min_rows=3)
        add_check(results, "synthetic_data_quality", case_id, "insufficient_rows_rejected", True, False)
    except ValueError as exc:
        message = str(exc)
        add_check(results, "synthetic_data_quality", case_id, "insufficient_rows_marker", True, "有效資料不足" in message)
        add_check(results, "synthetic_data_quality", case_id, "insufficient_rows_count_reported", True, "2 列" in message)

    summary["failfast_cases"] = 4
    return results, summary


def validate_load_clean_df_data_quality_case(base_params):
    case_id = "DATA_QUALITY_LOAD_CLEAN_DF"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    rows = [
        {"Date": "2024-01-04", "Open": 13, "High": 14, "Low": 12, "Close": 13.5, "Volume": 130},
        {"Date": "2024-01-02", "Open": 11, "High": 12, "Low": 10, "Close": 11.5, "Volume": 110},
        {"Date": "2024-01-01", "Open": 10, "High": 11, "Low": 9, "Close": 10.5, "Volume": 100},
        {"Date": "2024-01-03", "Open": 12, "High": 13, "Low": 11, "Close": None, "Volume": 120},
        {"Date": "2024-01-05", "Open": 14, "High": 15, "Low": 13, "Close": 14.5, "Volume": -2},
    ]

    with tempfile.TemporaryDirectory(prefix="v16_data_quality_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        data_dir = tmp_root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        csv_path = data_dir / "2330.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        csv_map = {"2330": str(csv_path)}
        with patch("tools.validate.real_case_io.get_required_min_rows", return_value=3):
            file_path, cleaned, stats = load_clean_df(
                project_root=str(tmp_root),
                data_dir=str(data_dir),
                csv_map_getter=lambda: csv_map,
                ticker="2330",
                params=base_params,
            )

    add_check(results, "synthetic_data_quality", case_id, "load_clean_df_returns_selected_path", str(csv_path), file_path)
    add_check(results, "synthetic_data_quality", case_id, "load_clean_df_keeps_valid_rows", 4, len(cleaned))
    add_check(results, "synthetic_data_quality", case_id, "load_clean_df_sorts_dates", True, cleaned.index.is_monotonic_increasing)
    add_check(results, "synthetic_data_quality", case_id, "load_clean_df_invalid_rows_count", 1, stats["invalid_row_count"])
    add_check(results, "synthetic_data_quality", case_id, "load_clean_df_negative_volume_corrected", 1, stats["negative_volume_corrected_count"])
    add_check(results, "synthetic_data_quality", case_id, "load_clean_df_zero_volume_row_present", 0.0, float(cleaned.loc[pd.Timestamp("2024-01-05"), "Volume"]), tol=1e-12)

    summary["loaded_rows"] = len(cleaned)
    return results, summary
