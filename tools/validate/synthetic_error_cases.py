from __future__ import annotations

import json
import tempfile
from pathlib import Path

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
