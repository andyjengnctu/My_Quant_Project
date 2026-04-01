from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[2]
from unittest.mock import patch

import pandas as pd

from tools.optimizer.raw_cache import load_all_raw_data
from tools.scanner.stock_processor import process_single_stock
from tools.validate.scanner_expectations import normalize_scanner_result

from .checks import add_check


def _build_repeatable_ohlcv_df() -> pd.DataFrame:
    rows = []
    for idx in range(8):
        base = 100.0 + idx
        rows.append(
            {
                "Date": f"2026-01-{idx + 2:02d}",
                "Open": base,
                "High": base + 1.0,
                "Low": base - 1.0,
                "Close": base + 0.5,
                "Volume": 1000 + idx * 10,
            }
        )
    return pd.DataFrame(rows)


def _normalize_cache_payload(raw_cache: dict[str, pd.DataFrame]) -> dict[str, dict[str, object]]:
    payload: dict[str, dict[str, object]] = {}
    for ticker, df in sorted(raw_cache.items()):
        normalized = df.reset_index().copy()
        normalized[normalized.columns[0]] = normalized[normalized.columns[0]].dt.strftime("%Y-%m-%d")
        records = []
        for row in normalized.to_dict(orient="records"):
            clean_row = {}
            for key, value in row.items():
                if isinstance(value, float):
                    clean_row[key] = round(value, 8)
                else:
                    clean_row[key] = value
            records.append(clean_row)
        payload[ticker] = {
            "columns": list(df.columns),
            "records": records,
        }
    return payload


def _payload_digest(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def validate_scanner_worker_repeatability_case(base_params):
    case_id = "SCANNER_WORKER_REPEATABILITY"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    with TemporaryDirectory() as tmp_dir:
        file_path = Path(tmp_dir) / "2330.csv"
        dummy_df = _build_repeatable_ohlcv_df()
        dummy_df.to_csv(file_path, index=False)
        sanitize_stats = {
            "invalid_row_count": 0,
            "duplicate_date_count": 0,
            "dropped_row_count": 0,
            "negative_volume_corrected_count": 0,
        }
        repeated_stats = {
            "is_candidate": True,
            "is_setup_today": True,
            "buy_limit": 105.0,
            "stop_loss": 99.0,
            "expected_value": 1.1,
            "win_rate": 56.0,
            "trade_count": 9,
            "max_drawdown": 7.5,
            "extended_candidate_today": None,
        }
        with patch("tools.scanner.stock_processor.sanitize_ohlcv_dataframe", return_value=(dummy_df, sanitize_stats)), patch(
            "tools.scanner.stock_processor.run_v16_backtest", return_value=repeated_stats
        ):
            first_result = normalize_scanner_result(process_single_stock(str(file_path), "2330", base_params))
            second_result = normalize_scanner_result(process_single_stock(str(file_path), "2330", base_params))

    add_check(results, "synthetic_regression", case_id, "scanner_worker_repeatable_payload", first_result, second_result)
    add_check(results, "synthetic_regression", case_id, "scanner_worker_repeatable_status", first_result["status"], second_result["status"])
    add_check(results, "synthetic_regression", case_id, "scanner_worker_repeatable_sort_value", first_result["sort_value"], second_result["sort_value"])
    summary["status"] = first_result.get("status")
    return results, summary


def validate_optimizer_raw_cache_rerun_consistency_case(_base_params):
    case_id = "OPTIMIZER_RAW_CACHE_RERUN"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    rows_2330 = [
        {"Date": "2026-01-05", "Open": 13, "High": 14, "Low": 12, "Close": 13.5, "Volume": 130},
        {"Date": "2026-01-03", "Open": 11, "High": 12, "Low": 10, "Close": 11.5, "Volume": 110},
        {"Date": "2026-01-02", "Open": 10, "High": 11, "Low": 9, "Close": 10.5, "Volume": 100},
        {"Date": "2026-01-03", "Open": 11, "High": 12, "Low": 10, "Close": 11.5, "Volume": 111},
        {"Date": "2026-01-04", "Open": 12, "High": 13, "Low": 11, "Close": 12.5, "Volume": -5},
    ]
    rows_2317 = [
        {"Date": "2026-01-02", "Open": 20, "High": 21, "Low": 19, "Close": 20.5, "Volume": 200},
        {"Date": "2026-01-03", "Open": 21, "High": 22, "Low": 20, "Close": 21.5, "Volume": 210},
        {"Date": "2026-01-04", "Open": 22, "High": 23, "Low": 21, "Close": 22.5, "Volume": 220},
    ]

    project_tmp_root = PROJECT_ROOT / "outputs" / "validate" / "_tmp_raw_cache"
    project_tmp_root.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix="v16_raw_cache_", dir=str(project_tmp_root)) as tmp_dir:
        tmp_root = Path(tmp_dir)
        data_dir = tmp_root / "data"
        output_dir = tmp_root / "outputs"
        data_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows_2330).to_csv(data_dir / "2330.csv", index=False)
        pd.DataFrame(rows_2317).to_csv(data_dir / "2317.csv", index=False)

        with contextlib.redirect_stdout(io.StringIO()):
            first_cache = load_all_raw_data(str(data_dir), required_min_rows=3, output_dir=str(output_dir))
        first_payload = _normalize_cache_payload(first_cache)
        first_digest = _payload_digest(first_payload)

        first_cache["2330"].iat[0, 0] = 999.0
        mutated_digest = _payload_digest(_normalize_cache_payload(first_cache))

        with contextlib.redirect_stdout(io.StringIO()):
            second_cache = load_all_raw_data(str(data_dir), required_min_rows=3, output_dir=str(output_dir))
        second_payload = _normalize_cache_payload(second_cache)
        second_digest = _payload_digest(second_payload)

        with contextlib.redirect_stdout(io.StringIO()):
            third_cache = load_all_raw_data(str(data_dir), required_min_rows=3, output_dir=str(output_dir))
        third_payload = _normalize_cache_payload(third_cache)
        third_digest = _payload_digest(third_payload)

        source_rows = pd.read_csv(data_dir / "2330.csv").to_dict(orient="records")

    add_check(results, "synthetic_regression", case_id, "raw_cache_first_and_second_digest_match", first_digest, second_digest)
    add_check(results, "synthetic_regression", case_id, "raw_cache_second_and_third_digest_match", second_digest, third_digest)
    add_check(results, "synthetic_regression", case_id, "raw_cache_mutation_does_not_persist", True, mutated_digest != second_digest)
    add_check(results, "synthetic_regression", case_id, "raw_cache_source_csv_not_mutated", 13.0, float(source_rows[0]["Open"]))
    add_check(results, "synthetic_regression", case_id, "raw_cache_ticker_keys_stable", ["2317", "2330"], sorted(second_cache.keys()))
    add_check(results, "synthetic_regression", case_id, "raw_cache_negative_volume_corrected", 0.0, float(second_cache["2330"].loc[pd.Timestamp("2026-01-04"), "Volume"]))

    summary["ticker_count"] = len(second_cache)
    return results, summary
