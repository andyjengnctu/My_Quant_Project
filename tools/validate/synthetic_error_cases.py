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
from unittest.mock import patch

from core.output_paths import output_dir_path

_RUNTIME_CASE_DIR = output_dir_path(PROJECT_ROOT, "local_regression") / "_staging" / "validate_runtime" / "_synthetic_runtime"

from core.params_io import load_params_from_json, params_to_json_dict
from tools.validate import module_loader
from tools.validate.preflight_env import (
    _normalize_local_regression_steps,
    load_requirement_names,
    run_preflight,
)

from .checks import bind_synthetic_case, bind_checks, add_check


def _write_json(path: Path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_params_io_error_path_case(base_params):
    case_id = "PARAMS_IO_ERROR_PATHS"
    results, summary, check, check_true = bind_synthetic_case(case_id, 'synthetic_error_paths')

    valid_payload = params_to_json_dict(base_params)
    required_fields = list(valid_payload.keys())

    with tempfile.TemporaryDirectory(prefix="v16_params_io_error_") as tmp_dir:
        tmp_root = Path(tmp_dir)

        missing_path = tmp_root / "missing_params.json"
        try:
            load_params_from_json(missing_path)
            check("missing_file_rejected", True, False)
        except FileNotFoundError as exc:
            message = str(exc)
            check("missing_file_contains_marker", True, "找不到參數檔" in message)
            check("missing_file_contains_path", True, str(missing_path) in message)

        invalid_json_path = tmp_root / "invalid_json.json"
        invalid_json_path.write_text('{"a": 1,}', encoding="utf-8")
        try:
            load_params_from_json(invalid_json_path)
            check("invalid_json_rejected", True, False)
        except RuntimeError as exc:
            message = str(exc)
            check("invalid_json_contains_prefix", True, message.startswith(f"讀取參數檔 {invalid_json_path} 失敗:"))
            check("invalid_json_contains_decoder", True, "JSONDecodeError" in message)

        invalid_root_path = tmp_root / "invalid_root.json"
        invalid_root_path.write_text(json.dumps([1, 2, 3], ensure_ascii=False), encoding="utf-8")
        try:
            load_params_from_json(invalid_root_path)
            check("invalid_root_rejected", True, False)
        except RuntimeError as exc:
            message = str(exc)
            check("invalid_root_contains_marker", True, "根層必須是 object/dict" in message)
            check("invalid_root_contains_path", True, str(invalid_root_path) in message)

        missing_key_payload = dict(valid_payload)
        dropped_field = required_fields[0]
        missing_key_payload.pop(dropped_field)
        missing_key_path = tmp_root / "missing_key.json"
        _write_json(missing_key_path, missing_key_payload)
        try:
            load_params_from_json(missing_key_path)
            check("missing_required_key_rejected", True, False)
        except RuntimeError as exc:
            message = str(exc)
            check("missing_required_key_contains_field", True, dropped_field in message)
            check("missing_required_key_contains_marker", True, "缺少必要欄位" in message)

        unknown_key_payload = dict(valid_payload)
        unknown_key_payload["tp_precent"] = 0.3
        unknown_key_path = tmp_root / "unknown_key.json"
        _write_json(unknown_key_path, unknown_key_payload)
        try:
            load_params_from_json(unknown_key_path)
            check("unknown_key_rejected", True, False)
        except RuntimeError as exc:
            message = str(exc)
            check("unknown_key_contains_field", True, "tp_precent" in message)
            check("unknown_key_contains_marker", True, "未知欄位" in message)

    summary["params_error_cases"] = 5
    return results, summary


def validate_module_loader_error_path_case(base_params):
    case_id = "MODULE_LOADER_ERROR_PATHS"
    results, summary, check, check_true = bind_synthetic_case(case_id, 'synthetic_error_paths')

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
            check("module_loader_error_rejected", True, False)
        except FileNotFoundError as exc:
            message = str(exc)
            check("module_loader_lists_checked_paths", True, "missing_file.py" in message and "broken_module.py" in message and "missing_attr.py" in message)
            check("module_loader_reports_syntax_error", True, "SyntaxError" in message)
            check("module_loader_reports_missing_attr", True, "缺少必要屬性" in message and "run" in message)
            check("module_loader_preserves_chinese_prefix", True, message.startswith("找不到符合條件的模組"))
        finally:
            module_loader.MODULE_CACHE.clear()

    summary["module_loader_error_cases"] = 1
    return results, summary


def validate_preflight_error_path_case(base_params):
    case_id = "PREFLIGHT_ERROR_PATHS"
    results, summary, check, check_true = bind_synthetic_case(case_id, 'synthetic_error_paths')

    with tempfile.TemporaryDirectory(prefix="v16_preflight_error_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        missing_requirements = tmp_root / "missing_requirements.txt"
        try:
            load_requirement_names(missing_requirements)
            check("missing_requirements_file_rejected", True, False)
        except FileNotFoundError as exc:
            message = str(exc)
            check("missing_requirements_contains_marker", True, "requirements 檔不存在" in message)
            check("missing_requirements_contains_path", True, str(missing_requirements) in message)

        try:
            _normalize_local_regression_steps(["meta_quality", "unknown_step"])
            check("invalid_step_rejected", True, False)
        except ValueError as exc:
            message = str(exc)
            check("invalid_step_contains_bad_value", True, "unknown_step" in message)
            check("invalid_step_lists_valid_steps", True, "quick_gate" in message and "meta_quality" in message)

        requirements_path = tmp_root / "requirements.txt"
        requirements_path.write_text("demo-pkg==1.0\n", encoding="utf-8")
        with patch("tools.validate.preflight_env._check_distribution", return_value=(True, "1.0.0")), patch(
            "tools.validate.preflight_env._check_import",
            return_value=(False, "ImportError: demo import failed"),
        ):
            payload = run_preflight(requirements_path=requirements_path)
        check("preflight_import_failure_status", "FAIL", payload["status"])
        check("preflight_import_failure_failed_package", ["demo-pkg"], payload["failed_packages"])
        checks = payload.get("checks", [])
        detail = checks[0]["detail"] if checks else ""
        check("preflight_import_failure_detail", True, "ImportError: demo import failed" in detail)
        check("preflight_import_failure_import_name", "demo_pkg", checks[0]["import_name"] if checks else None)

    summary["preflight_error_cases"] = 3
    return results, summary



def validate_downloader_finmind_token_resolution_case(base_params):
    from services.downloader import runtime

    case_id = "DOWNLOADER_FINMIND_TOKEN_RESOLUTION"
    results, summary, check, check_true = bind_synthetic_case(case_id, 'synthetic_error_paths')

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        token_path = root / "doc" / "FINMIND_API_TOKEN.md"
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(
            "# id:\nsynthetic-user\n\n# passwd:\nsynthetic-pass\n\n# API_TOKEN\nsynthetic-file-token\n",
            encoding="utf-8",
        )

        file_token = runtime.resolve_finmind_api_token(project_root=root, environ={})
        check("markdown_api_token_section_is_used_as_fallback", "synthetic-file-token", file_token)

        env_token = runtime.resolve_finmind_api_token(
            project_root=root,
            environ={"FINMIND_API_TOKEN": "synthetic-env-token"},
        )
        check("environment_variable_has_precedence_over_private_markdown", "synthetic-env-token", env_token)

        token_path.write_text("# id:\nsynthetic-user\n", encoding="utf-8")
        missing_token = runtime.resolve_finmind_api_token(project_root=root, environ={})
        check("missing_api_token_section_resolves_to_empty_without_reading_other_credentials", "", missing_token)

    class _DummyLoader:
        def __init__(self):
            self.logged_token = None

        def login_by_token(self, api_token):
            self.logged_token = api_token

    with patch.object(runtime, "dl", None), patch.object(
        runtime, "get_finmind_dataloader_class", return_value=_DummyLoader
    ), patch.object(runtime, "resolve_finmind_api_token", return_value="synthetic-runtime-token"):
        loader = runtime.get_finmind_loader()
    check("finmind_loader_uses_canonical_token_resolver_at_initialization", "synthetic-runtime-token", loader.logged_token)

    source = (PROJECT_ROOT / "services" / "downloader" / "runtime.py").read_text(encoding="utf-8")
    check("runtime_keeps_env_then_private_markdown_precedence_in_single_owner", True, 'FINMIND_API_TOKEN_ENV_VAR = "FINMIND_API_TOKEN"' in source and 'Path("doc") / "FINMIND_API_TOKEN.md"' in source and "token = resolve_finmind_api_token()" in source)

    summary["finmind_token_resolution_cases"] = 1
    return results, summary

def validate_downloader_main_error_path_case(base_params):
    import importlib
    downloader_main = importlib.import_module("services.downloader.main")

    case_id = "DOWNLOADER_MAIN_ERROR_PATHS"
    results, summary, check, check_true = bind_synthetic_case(case_id, 'synthetic_error_paths')

    market_data_auto_update = importlib.import_module("services.trading.market_data_auto_update")
    stderr = io.StringIO()
    with patch.object(
        market_data_auto_update,
        "run_trading_market_data_auto_update",
        side_effect=RuntimeError(
            "Market Data V2 Daily Update synthetic failure；"
            "target=2026-09-10；dataset=TaiwanStockPriceAdj"
        ),
    ):
        with contextlib.redirect_stderr(stderr):
            rc = downloader_main.main(["services/downloader/main.py"])

    err = stderr.getvalue()
    check("downloader_main_returns_failure", 1, rc)
    check("downloader_main_reports_runtimeerror", True, "❌ RuntimeError:" in err)
    check("downloader_main_reports_v2_target_context", True, "target=2026-09-10" in err)
    check("downloader_main_reports_v2_dataset_context", True, "dataset=TaiwanStockPriceAdj" in err)
    summary["downloader_main_error_cases"] = 1
    return results, summary


