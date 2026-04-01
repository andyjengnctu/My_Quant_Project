from __future__ import annotations

import contextlib
import io
import json
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME_CASE_DIR = PROJECT_ROOT / "outputs" / "validate" / "_synthetic_runtime"
from unittest.mock import patch

from core.params_io import load_params_from_json, params_to_json_dict
from tools.validate import module_loader
from tools.validate.preflight_env import (
    _normalize_local_regression_steps,
    load_requirement_names,
    run_preflight,
)

from .checks import add_check


def _write_json(path: Path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_params_io_error_path_case(base_params):
    case_id = "PARAMS_IO_ERROR_PATHS"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    valid_payload = params_to_json_dict(base_params)
    required_fields = list(valid_payload.keys())

    with tempfile.TemporaryDirectory(prefix="v16_params_io_error_") as tmp_dir:
        tmp_root = Path(tmp_dir)

        missing_path = tmp_root / "missing_params.json"
        try:
            load_params_from_json(missing_path)
            add_check(results, "synthetic_error_paths", case_id, "missing_file_rejected", True, False)
        except FileNotFoundError as exc:
            message = str(exc)
            add_check(results, "synthetic_error_paths", case_id, "missing_file_contains_marker", True, "找不到參數檔" in message)
            add_check(results, "synthetic_error_paths", case_id, "missing_file_contains_path", True, str(missing_path) in message)

        invalid_json_path = tmp_root / "invalid_json.json"
        invalid_json_path.write_text('{"a": 1,}', encoding="utf-8")
        try:
            load_params_from_json(invalid_json_path)
            add_check(results, "synthetic_error_paths", case_id, "invalid_json_rejected", True, False)
        except RuntimeError as exc:
            message = str(exc)
            add_check(results, "synthetic_error_paths", case_id, "invalid_json_contains_prefix", True, message.startswith(f"讀取參數檔 {invalid_json_path} 失敗:"))
            add_check(results, "synthetic_error_paths", case_id, "invalid_json_contains_decoder", True, "JSONDecodeError" in message)

        invalid_root_path = tmp_root / "invalid_root.json"
        invalid_root_path.write_text(json.dumps([1, 2, 3], ensure_ascii=False), encoding="utf-8")
        try:
            load_params_from_json(invalid_root_path)
            add_check(results, "synthetic_error_paths", case_id, "invalid_root_rejected", True, False)
        except RuntimeError as exc:
            message = str(exc)
            add_check(results, "synthetic_error_paths", case_id, "invalid_root_contains_marker", True, "根層必須是 object/dict" in message)
            add_check(results, "synthetic_error_paths", case_id, "invalid_root_contains_path", True, str(invalid_root_path) in message)

        missing_key_payload = dict(valid_payload)
        dropped_field = required_fields[0]
        missing_key_payload.pop(dropped_field)
        missing_key_path = tmp_root / "missing_key.json"
        _write_json(missing_key_path, missing_key_payload)
        try:
            load_params_from_json(missing_key_path)
            add_check(results, "synthetic_error_paths", case_id, "missing_required_key_rejected", True, False)
        except RuntimeError as exc:
            message = str(exc)
            add_check(results, "synthetic_error_paths", case_id, "missing_required_key_contains_field", True, dropped_field in message)
            add_check(results, "synthetic_error_paths", case_id, "missing_required_key_contains_marker", True, "缺少必要欄位" in message)

        unknown_key_payload = dict(valid_payload)
        unknown_key_payload["tp_precent"] = 0.3
        unknown_key_path = tmp_root / "unknown_key.json"
        _write_json(unknown_key_path, unknown_key_payload)
        try:
            load_params_from_json(unknown_key_path)
            add_check(results, "synthetic_error_paths", case_id, "unknown_key_rejected", True, False)
        except RuntimeError as exc:
            message = str(exc)
            add_check(results, "synthetic_error_paths", case_id, "unknown_key_contains_field", True, "tp_precent" in message)
            add_check(results, "synthetic_error_paths", case_id, "unknown_key_contains_marker", True, "未知欄位" in message)

    summary["params_error_cases"] = 5
    return results, summary


def validate_module_loader_error_path_case(base_params):
    case_id = "MODULE_LOADER_ERROR_PATHS"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    runtime_root = _RUNTIME_CASE_DIR / "module_loader_error_case"
    runtime_root.mkdir(parents=True, exist_ok=True)
    syntax_file = runtime_root / "broken_module.py"
    syntax_file.write_text("def broken(:\n    pass\n", encoding="utf-8")
    missing_attr_file = runtime_root / "missing_attr.py"
    missing_attr_file.write_text("VALUE = 1\n", encoding="utf-8")

    with patch.object(module_loader, "PROJECT_ROOT", str(runtime_root)):
        module_loader.MODULE_CACHE.clear()
        try:
            module_loader.load_module_from_candidates(
                "test_loader_error_case",
                ["missing_file.py", "broken_module.py", "missing_attr.py"],
                ["run"],
            )
            add_check(results, "synthetic_error_paths", case_id, "module_loader_error_rejected", True, False)
        except FileNotFoundError as exc:
            message = str(exc)
            add_check(results, "synthetic_error_paths", case_id, "module_loader_lists_checked_paths", True, "missing_file.py" in message and "broken_module.py" in message and "missing_attr.py" in message)
            add_check(results, "synthetic_error_paths", case_id, "module_loader_reports_syntax_error", True, "SyntaxError" in message)
            add_check(results, "synthetic_error_paths", case_id, "module_loader_reports_missing_attr", True, "缺少必要屬性" in message and "run" in message)
            add_check(results, "synthetic_error_paths", case_id, "module_loader_preserves_chinese_prefix", True, message.startswith("找不到符合條件的模組"))
        finally:
            module_loader.MODULE_CACHE.clear()

    summary["module_loader_error_cases"] = 1
    return results, summary


def validate_preflight_error_path_case(base_params):
    case_id = "PREFLIGHT_ERROR_PATHS"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    with tempfile.TemporaryDirectory(prefix="v16_preflight_error_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        missing_requirements = tmp_root / "missing_requirements.txt"
        try:
            load_requirement_names(missing_requirements)
            add_check(results, "synthetic_error_paths", case_id, "missing_requirements_file_rejected", True, False)
        except FileNotFoundError as exc:
            message = str(exc)
            add_check(results, "synthetic_error_paths", case_id, "missing_requirements_contains_marker", True, "requirements 檔不存在" in message)
            add_check(results, "synthetic_error_paths", case_id, "missing_requirements_contains_path", True, str(missing_requirements) in message)

        try:
            _normalize_local_regression_steps(["meta_quality", "unknown_step"])
            add_check(results, "synthetic_error_paths", case_id, "invalid_step_rejected", True, False)
        except ValueError as exc:
            message = str(exc)
            add_check(results, "synthetic_error_paths", case_id, "invalid_step_contains_bad_value", True, "unknown_step" in message)
            add_check(results, "synthetic_error_paths", case_id, "invalid_step_lists_valid_steps", True, "quick_gate" in message and "meta_quality" in message)

        requirements_path = tmp_root / "requirements.txt"
        requirements_path.write_text("demo-pkg==1.0\n", encoding="utf-8")
        with patch("tools.validate.preflight_env._check_distribution", return_value=(True, "1.0.0")), patch(
            "tools.validate.preflight_env._check_import",
            return_value=(False, "ImportError: demo import failed"),
        ):
            payload = run_preflight(requirements_path=requirements_path)
        add_check(results, "synthetic_error_paths", case_id, "preflight_import_failure_status", "FAIL", payload["status"])
        add_check(results, "synthetic_error_paths", case_id, "preflight_import_failure_failed_package", ["demo-pkg"], payload["failed_packages"])
        checks = payload.get("checks", [])
        detail = checks[0]["detail"] if checks else ""
        add_check(results, "synthetic_error_paths", case_id, "preflight_import_failure_detail", True, "ImportError: demo import failed" in detail)
        add_check(results, "synthetic_error_paths", case_id, "preflight_import_failure_import_name", "demo_pkg", checks[0]["import_name"] if checks else None)

    summary["preflight_error_cases"] = 3
    return results, summary


def validate_downloader_market_date_fallback_case(base_params):
    from tools.downloader import universe

    case_id = "DOWNLOADER_MARKET_DATE_FALLBACK"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    issue_sections = []

    class _BrokenLoader:
        def get_data(self, *args, **kwargs):
            raise requests.RequestException("finmind down")

    class _BrokenTicker:
        def history(self, period="5d"):
            raise requests.RequestException("yf down")

    class _BrokenYF:
        def Ticker(self, symbol):
            return _BrokenTicker()

    fixed_now = datetime(2026, 4, 6, 13, 0, 0)
    with patch.object(universe.rt, "get_finmind_loader", return_value=_BrokenLoader()), \
         patch.object(universe.rt, "get_yfinance_module", return_value=_BrokenYF()), \
         patch.object(universe.rt, "append_downloader_issues", side_effect=lambda section, lines: issue_sections.append((section, list(lines)))), \
         patch.object(universe.rt, "get_taipei_now", return_value=fixed_now):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            actual_date = universe.get_market_last_date()

    out = stdout.getvalue()
    add_check(results, "synthetic_error_paths", case_id, "fallback_previous_weekday_date", "2026-04-03", actual_date)
    add_check(results, "synthetic_error_paths", case_id, "finmind_failure_logged", True, any(section == "最新交易日(FinMind)失敗" and "RequestException: finmind down" in "\n".join(lines) for section, lines in issue_sections))
    add_check(results, "synthetic_error_paths", case_id, "yf_failure_logged", True, any(section == "最新交易日(YF備援)失敗" and "RequestException: yf down" in "\n".join(lines) for section, lines in issue_sections))
    add_check(results, "synthetic_error_paths", case_id, "stdout_contains_fallback_notice", True, "使用智能推算平日備用日期: 2026-04-03" in out)
    summary["downloader_fallback_cases"] = 1
    return results, summary


def validate_downloader_sync_error_path_case(base_params):
    from tools.downloader import sync

    case_id = "DOWNLOADER_SYNC_ERROR_PATHS"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    issue_sections = []

    class _DummyLoader:
        def get_data(self, dataset, data_id, start_date):
            if data_id == "1101":
                return pd.DataFrame()
            raise requests.RequestException("network timeout")

    dummy_loader = _DummyLoader()
    issue_log_path = str(PROJECT_ROOT / "outputs" / "smart_downloader" / "downloader_issues_test.log")
    with tempfile.TemporaryDirectory(prefix="v16_downloader_sync_error_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        with patch.object(sync.rt, "ensure_runtime_dirs", return_value=None), \
             patch.object(sync.rt, "SAVE_DIR", str(tmp_root)), \
             patch.object(sync.rt.os.path, "exists", return_value=False), \
             patch.object(sync.rt, "get_finmind_loader", return_value=dummy_loader), \
             patch.object(sync.rt.time, "sleep", return_value=None), \
             patch.object(sync.rt, "append_downloader_issues", side_effect=lambda section, lines: issue_sections.append((section, list(lines)))), \
             patch.object(sync.rt, "get_downloader_issue_log_path", return_value=issue_log_path):
            payload = sync.smart_download_vip_data(["1101", "1102"], market_last_date="2024-01-03", verbose=False)

    add_check(results, "synthetic_error_paths", case_id, "all_failed_count_success", 0, payload["count_success"])
    add_check(results, "synthetic_error_paths", case_id, "all_failed_download_error_count", 2, payload["download_error_count"])
    add_check(results, "synthetic_error_paths", case_id, "issue_log_path_exposed", issue_log_path, payload["issue_log_path"])
    combined_issue_lines = "\n".join([f"{section}: {' | '.join(lines)}" for section, lines in issue_sections])
    add_check(results, "synthetic_error_paths", case_id, "empty_df_valueerror_contains_ticker", True, "1101 -> ValueError: FinMind 回傳空資料" in combined_issue_lines)
    add_check(results, "synthetic_error_paths", case_id, "request_exception_contains_ticker", True, "1102 -> RequestException: network timeout" in combined_issue_lines)
    summary["downloader_sync_error_cases"] = 1
    return results, summary


def validate_downloader_main_error_path_case(base_params):
    import importlib
    downloader_main = importlib.import_module("tools.downloader.main")

    case_id = "DOWNLOADER_MAIN_ERROR_PATHS"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    class _FakeRT:
        @staticmethod
        def get_taipei_now():
            return datetime(2026, 4, 2, 9, 30, 0)

    stderr = io.StringIO()
    with patch.object(downloader_main, "_get_downloader_modules", return_value=(_FakeRT(), object(), lambda: "2026-04-01", lambda: ["1101"])), \
         patch.object(downloader_main, "smart_download_vip_data", return_value={
             "count_success": 0,
             "count_skipped_latest": 0,
             "last_date_check_error_count": 1,
             "download_error_count": 2,
             "issue_log_path": "outputs/smart_downloader/downloader_issues_20260402.log",
         }):
        with contextlib.redirect_stderr(stderr):
            rc = downloader_main.main(["tools/downloader/main.py"])

    err = stderr.getvalue()
    add_check(results, "synthetic_error_paths", case_id, "downloader_main_returns_failure", 1, rc)
    add_check(results, "synthetic_error_paths", case_id, "downloader_main_reports_runtimeerror", True, "❌ RuntimeError:" in err)
    add_check(results, "synthetic_error_paths", case_id, "downloader_main_reports_counts", True, "成功 0 檔、已最新跳過 0 檔、最後日期檢查失敗 1 檔、下載失敗 2 檔" in err)
    add_check(results, "synthetic_error_paths", case_id, "downloader_main_reports_issue_log_path", True, "outputs/smart_downloader/downloader_issues_20260402.log" in err)
    summary["downloader_main_error_cases"] = 1
    return results, summary
