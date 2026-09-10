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

def validate_downloader_market_date_fallback_case(base_params):
    from services.downloader import universe

    case_id = "DOWNLOADER_MARKET_DATE_FALLBACK"
    results, summary, check, check_true = bind_synthetic_case(case_id, 'synthetic_error_paths')

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
            try:
                universe.get_market_last_date()
            except RuntimeError as exc:
                fail_closed_error = str(exc)
            else:
                fail_closed_error = ""

    out = stdout.getvalue()
    check("provider_failure_is_fail_closed", True, bool(fail_closed_error))
    check("provider_failure_rejects_guessed_weekday", True, "不能把平日推算當成實際交易日" in fail_closed_error)
    check("finmind_failure_logged", True, any(section == "最新交易日(FinMind)失敗" and "RequestException: finmind down" in "\n".join(lines) for section, lines in issue_sections))
    check("yf_failure_logged", True, any(section == "最新交易日(YF備援)失敗" and "RequestException: yf down" in "\n".join(lines) for section, lines in issue_sections))
    check("stdout_has_no_guessed_weekday_fallback", False, "使用智能推算平日備用日期" in out)
    summary["downloader_fallback_cases"] = 1
    return results, summary


def validate_downloader_sync_error_path_case(base_params):
    from services.downloader import sync

    case_id = "DOWNLOADER_SYNC_ERROR_PATHS"
    results, summary, check, check_true = bind_synthetic_case(case_id, 'synthetic_error_paths')

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

    check("all_failed_count_success", 0, payload["count_success"])
    check("all_failed_download_error_count", 2, payload["download_error_count"])
    check("issue_log_path_exposed", issue_log_path, payload["issue_log_path"])
    combined_issue_lines = "\n".join([f"{section}: {' | '.join(lines)}" for section, lines in issue_sections])
    check("empty_df_valueerror_contains_ticker", True, "1101 -> ValueError: FinMind 回傳空資料" in combined_issue_lines)
    check("request_exception_contains_ticker", True, "1102 -> RequestException: network timeout" in combined_issue_lines)
    summary["downloader_sync_error_cases"] = 1
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


def validate_downloader_universe_fetch_error_path_case(base_params):
    from services.downloader import universe

    case_id = "DOWNLOADER_UNIVERSE_FETCH_ERROR_PATHS"
    results, summary, check, check_true = bind_synthetic_case(case_id, 'synthetic_error_paths')

    issue_sections = []

    # Parser regression: do not depend on decoded Chinese header names.
    positional_table = pd.DataFrame([
        ["2330 台積電", "x", "x", "x", "x", "ESVUFR", ""],
        ["0050 元大台灣50", "x", "x", "x", "x", "CEOGEU", ""],
        ["030004 權證", "x", "x", "x", "x", "RWXXXX", ""],
    ], columns=["garbled-0", "garbled-1", "garbled-2", "garbled-3", "garbled-4", "garbled-5", "garbled-6"])
    with patch.object(universe.pd, "read_html", return_value=[positional_table]):
        parsed = universe._parse_isin_universe_html("<html></html>")
    check("isin_parser_uses_positional_contract_not_localized_headers", [
        {"sid": "2330", "is_etf": False},
        {"sid": "0050", "is_etf": True},
    ], parsed)

    class _GoodResponse:
        text = "<html></html>"
        encoding = None
        def raise_for_status(self):
            return None

    # A transient request failure must retry with browser-compatible headers and recover.
    transient_calls = []
    good_response = _GoodResponse()
    transient_responses = [requests.RequestException("temporary block"), good_response]
    def _transient_get(url, **kwargs):
        transient_calls.append((url, dict(kwargs)))
        value = transient_responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    with patch.object(universe.requests, "get", side_effect=_transient_get), \
         patch.object(universe, "_parse_isin_universe_html", return_value=[{"sid": "2330", "is_etf": False}]), \
         patch.object(universe.rt.time, "sleep", return_value=None):
        recovered = universe._fetch_isin_universe_source("https://isin.twse.com.tw/isin/C_public.jsp?strMode=2")
    check("isin_transient_failure_is_retried", 2, len(transient_calls))
    check("isin_retry_recovers_membership", [{"sid": "2330", "is_etf": False}], recovered)
    check("isin_fetch_sends_browser_user_agent", True, all(bool(call[1].get("headers", {}).get("User-Agent")) for call in transient_calls))
    check("isin_fetch_sets_cp950_decoding", "cp950", good_response.encoding)

    with tempfile.TemporaryDirectory(prefix="v16_downloader_universe_fetch_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        with patch.object(universe.rt, "ensure_runtime_dirs", return_value=None), \
             patch.object(universe.rt, "SAVE_DIR", str(tmp_root)), \
             patch.object(universe.rt.os.path, "exists", return_value=False), \
             patch.object(universe.requests, "get", side_effect=requests.RequestException("twse down")), \
             patch.object(universe.rt.time, "sleep", return_value=None), \
             patch.object(universe.rt, "append_downloader_issues", side_effect=lambda section, lines: issue_sections.append((section, list(lines)))):
            try:
                universe.get_or_update_universe(market_date="2026-04-03")
                check("universe_fetch_failure_rejected", True, False)
            except RuntimeError as exc:
                message = str(exc)
                check("universe_fetch_failure_reports_runtimeerror", True, "無法取得任何台股股票名單" in message)
                check("universe_fetch_failure_logs_issues", True, any(section == "名單來源失敗" and "twse down" in "\n".join(lines) for section, lines in issue_sections))

        partial_issue_sections = []
        source_rows = [{"sid": "2330", "is_etf": False}]
        with patch.object(universe.rt, "ensure_runtime_dirs", return_value=None), \
             patch.object(universe.rt, "get_universe_list_file_path", return_value=str(tmp_root / "universe_cache_v3.json")), \
             patch.object(universe.rt.os.path, "exists", return_value=False), \
             patch.object(universe, "_fetch_isin_universe_source", side_effect=[source_rows, ValueError("tpex down")]), \
             patch.object(universe.rt, "append_downloader_issues", side_effect=lambda section, lines: partial_issue_sections.append((section, list(lines)))):
            try:
                universe.get_or_update_universe(market_date="2026-04-03")
                partial_rejected = False
                partial_message = ""
            except RuntimeError as exc:
                partial_rejected = True
                partial_message = str(exc)

        check("partial_twse_tpex_source_failure_is_fail_closed", True, partial_rejected)
        check("partial_source_failure_reports_incomplete_universe", True, "universe 來源不完整" in partial_message)
        check("partial_source_failure_never_publishes_cache", False, (tmp_root / "universe_cache_v3.json").exists())
        check("partial_source_failure_logs_failed_source", True, any(section == "名單來源失敗" and "tpex down" in "\n".join(lines) for section, lines in partial_issue_sections))

        empty_issue_sections = []
        with patch.object(universe.rt, "ensure_runtime_dirs", return_value=None), \
             patch.object(universe.rt, "get_universe_list_file_path", return_value=str(tmp_root / "universe_cache_empty.json")), \
             patch.object(universe.rt.os.path, "exists", return_value=False), \
             patch.object(universe, "_fetch_isin_universe_source", side_effect=[[], [{"sid": "0050", "is_etf": True}]]), \
             patch.object(universe.rt, "append_downloader_issues", side_effect=lambda section, lines: empty_issue_sections.append((section, list(lines)))):
            try:
                universe.get_or_update_universe(market_date="2026-04-03")
                empty_source_rejected = False
            except RuntimeError:
                empty_source_rejected = True
        check("empty_one_market_source_is_fail_closed", True, empty_source_rejected)

    summary["issue_section_count"] = len(issue_sections)
    return results, summary

def validate_downloader_universe_screening_init_error_path_case(base_params):
    from services.downloader import universe

    case_id = "DOWNLOADER_UNIVERSE_SCREENING_INIT_ERROR_PATHS"
    results, summary, check, check_true = bind_synthetic_case(case_id, 'synthetic_error_paths')

    issue_sections = []

    twse_rows = [
        {"sid": "2330", "is_etf": False},
        {"sid": "2317", "is_etf": False},
        {"sid": "9999", "is_etf": False},
    ]
    tpex_rows = [{"sid": "0050", "is_etf": True}]

    class _BulkLoader:
        def __init__(self, *, fail_dataset=None, missing_market_value=False):
            self.calls = []
            self.fail_dataset = fail_dataset
            self.missing_market_value = missing_market_value

        def get_data(self, **kwargs):
            self.calls.append(dict(kwargs))
            dataset = kwargs.get("dataset")
            if dataset == self.fail_dataset:
                raise requests.RequestException(f"{dataset} synthetic failure")
            if dataset == universe.rt.FINMIND_UNIVERSE_VOLUME_DATASET:
                return pd.DataFrame({
                    "date": ["2026-04-03", "2026-04-03", "2026-04-03", "2026-04-06"],
                    "stock_id": ["2330", "2317", "0050", "2330"],
                    "Trading_Volume": [2_000.0, 500.0, 3_000.0, 99_000_000.0],
                })
            if dataset == universe.FINMIND_RAW_PRICE_ARCHIVE_DATASET:
                return pd.DataFrame({
                    "date": ["2026-04-03", "2026-04-03", "2026-04-03"],
                    "stock_id": ["2330", "2317", "0050"],
                    "open": [100.0, 80.0, 50.0],
                })
            if dataset == universe.rt.FINMIND_UNIVERSE_MARKET_VALUE_DATASET:
                stock_ids = ["0050"] if self.missing_market_value else ["2330", "0050"]
                values = [100_000_000_000.0] if self.missing_market_value else [2_000_000_000.0, 100_000_000_000.0]
                return pd.DataFrame({
                    "date": ["2026-04-03"] * len(stock_ids),
                    "stock_id": stock_ids,
                    "market_value": values,
                })
            raise AssertionError(f"unexpected dataset {dataset}")

    with tempfile.TemporaryDirectory(prefix="v16_downloader_universe_screening_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        cache_path = tmp_root / "universe_cache_v3.json"

        with patch.object(universe.rt, "ensure_runtime_dirs", return_value=None), \
             patch.object(universe.rt, "get_universe_list_file_path", return_value=str(cache_path)), \
             patch.object(universe.rt.os.path, "exists", return_value=False), \
             patch.object(universe, "_fetch_isin_universe_source", side_effect=[twse_rows, tpex_rows]), \
             patch.object(universe.rt, "get_finmind_loader", side_effect=ModuleNotFoundError("no module named FinMind")), \
             patch.object(universe.rt, "get_yfinance_module", side_effect=AssertionError("YFinance must not be used for universe screening")), \
             patch.object(universe.rt, "append_downloader_issues", side_effect=lambda section, lines: issue_sections.append((section, list(lines)))):
            try:
                universe.get_or_update_universe(market_date="2026-04-03")
                init_rejected = False
                init_message = ""
            except RuntimeError as exc:
                init_rejected = True
                init_message = str(exc)

        check("finmind_bulk_screening_init_failure_is_fail_closed", True, init_rejected)
        check("finmind_bulk_screening_init_failure_reports_provider", True, "FinMind Backer 全市場快篩失敗" in init_message and "ModuleNotFoundError" in init_message)
        check("finmind_bulk_screening_init_failure_logs_issue", True, any(section == "FinMind bulk快篩失敗" and "ModuleNotFoundError" in "\n".join(lines) for section, lines in issue_sections))
        check("finmind_bulk_screening_init_failure_never_publishes_cache", False, cache_path.exists())

        issue_sections.clear()
        failing_loader = _BulkLoader(fail_dataset=universe.rt.FINMIND_UNIVERSE_MARKET_VALUE_DATASET)
        with patch.object(universe.rt, "ensure_runtime_dirs", return_value=None), \
             patch.object(universe.rt, "get_universe_list_file_path", return_value=str(cache_path)), \
             patch.object(universe.rt.os.path, "exists", return_value=False), \
             patch.object(universe, "_fetch_isin_universe_source", side_effect=[twse_rows, tpex_rows]), \
             patch.object(universe.rt, "get_finmind_loader", return_value=failing_loader), \
             patch.object(universe.rt, "get_yfinance_module", side_effect=AssertionError("YFinance must not be used for universe screening")), \
             patch.object(universe.rt, "append_downloader_issues", side_effect=lambda section, lines: issue_sections.append((section, list(lines)))):
            try:
                universe.get_or_update_universe(market_date="2026-04-03")
                dataset_failure_rejected = False
            except RuntimeError:
                dataset_failure_rejected = True
        check("either_finmind_bulk_dataset_failure_is_fail_closed", True, dataset_failure_rejected)
        check("bulk_dataset_failure_never_publishes_cache", False, cache_path.exists())

        good_loader = _BulkLoader()
        with patch.object(universe.rt, "ensure_runtime_dirs", return_value=None), \
             patch.object(universe.rt, "get_universe_list_file_path", return_value=str(cache_path)), \
             patch.object(universe.rt.os.path, "exists", return_value=False), \
             patch.object(universe, "_fetch_isin_universe_source", side_effect=[twse_rows, tpex_rows]), \
             patch.object(universe.rt, "get_finmind_loader", return_value=good_loader), \
             patch.object(universe.rt, "get_yfinance_module", side_effect=AssertionError("YFinance must not be used for universe screening")), \
             patch.object(universe.rt, "MIN_VOLUME", 1_000), \
             patch.object(universe.rt, "MIN_MARKET_CAP", 1_000_000_000):
            membership = universe.get_or_update_universe(market_date="2026-04-03")

        check("finmind_bulk_screening_uses_exact_market_date_not_future_row", ["2330", "0050"], membership)
        check("finmind_bulk_screening_does_not_call_yfinance", True, True)
        check("finmind_bulk_screening_uses_three_dataset_requests_with_raw_ticker_evidence", 3, len(good_loader.calls))
        check("finmind_bulk_price_request_omits_data_id", False, "data_id" in good_loader.calls[0])
        check("finmind_raw_ticker_evidence_request_omits_data_id", False, "data_id" in good_loader.calls[1])
        check("finmind_bulk_market_value_request_omits_data_id", False, "data_id" in good_loader.calls[2])
        check("finmind_bulk_requests_use_requested_market_date", ["2026-04-03", "2026-04-03", "2026-04-03"], [call.get("start_date") for call in good_loader.calls])
        check("etf_qualifies_from_volume_without_market_value_requirement", True, "0050" in membership)
        check("listed_symbol_without_exact_price_is_conservatively_excluded", False, "9999" in membership)
        check("universe_v3_cache_is_machine_readable_and_published", True, cache_path.is_file() and cache_path.read_text(encoding="utf-8").lstrip().startswith("{"))

        with patch.object(universe.rt, "get_taipei_file_mtime", return_value=universe.rt.get_taipei_now()), \
             patch.object(universe.rt, "MIN_VOLUME", 1_000), \
             patch.object(universe.rt, "MIN_MARKET_CAP", 1_000_000_000):
            cache_reused = universe._load_reusable_universe_cache(cache_path, now=universe.rt.get_taipei_now(), market_date="2026-04-03")
        check("matching_universe_v3_contract_can_reuse_cache", ["2330", "0050"], cache_reused)

        with patch.object(universe.rt, "get_taipei_file_mtime", return_value=universe.rt.get_taipei_now()), \
             patch.object(universe.rt, "MIN_VOLUME", 1_000), \
             patch.object(universe.rt, "MIN_MARKET_CAP", 1_000_000_000):
            wrong_date_cache = universe._load_reusable_universe_cache(
                cache_path, now=universe.rt.get_taipei_now(), market_date="2026-04-06"
            )
        check("universe_cache_market_date_mismatch_invalidates_membership",
            None, wrong_date_cache,
        )

        with patch.object(universe.rt, "MIN_VOLUME", 2_000), \
             patch.object(universe.rt, "MIN_MARKET_CAP", 1_000_000_000), \
             patch.object(universe.rt, "get_taipei_file_mtime", return_value=universe.rt.get_taipei_now()):
            stale_threshold_cache = universe._load_reusable_universe_cache(cache_path, now=universe.rt.get_taipei_now(), market_date="2026-04-03")
        check("universe_threshold_change_invalidates_bulk_cache", None, stale_threshold_cache)

        cache_path.unlink(missing_ok=True)
        missing_cap_loader = _BulkLoader(missing_market_value=True)
        with patch.object(universe.rt, "ensure_runtime_dirs", return_value=None), \
             patch.object(universe.rt, "get_universe_list_file_path", return_value=str(cache_path)), \
             patch.object(universe.rt.os.path, "exists", return_value=False), \
             patch.object(universe, "_fetch_isin_universe_source", side_effect=[twse_rows, tpex_rows]), \
             patch.object(universe.rt, "get_finmind_loader", return_value=missing_cap_loader), \
             patch.object(universe.rt, "MIN_VOLUME", 1_000), \
             patch.object(universe.rt, "MIN_MARKET_CAP", 1_000_000_000):
            try:
                universe.get_or_update_universe(market_date="2026-04-03")
                missing_cap_rejected = False
                missing_cap_message = ""
            except RuntimeError as exc:
                missing_cap_rejected = True
                missing_cap_message = str(exc)
        check("high_volume_stock_missing_same_day_market_value_is_fail_closed", True, missing_cap_rejected)
        check("missing_market_value_failure_names_missing_stock", True, "2330" in missing_cap_message)
        check("missing_market_value_failure_never_publishes_cache", False, cache_path.exists())

        duplicate_price = pd.DataFrame({
            "date": ["2026-04-03", "2026-04-03"],
            "stock_id": ["2330", "2330"],
            "Trading_Volume": [2_000, 2_100],
        })
        try:
            universe._normalize_finmind_bulk_screening_frame(
                duplicate_price,
                dataset=universe.rt.FINMIND_UNIVERSE_VOLUME_DATASET,
                market_date="2026-04-03",
                value_column="trading_volume",
                allow_zero=True,
            )
            duplicate_rejected = False
        except ValueError:
            duplicate_rejected = True
        check("duplicate_finmind_bulk_stock_id_is_rejected", True, duplicate_rejected)

    summary["bulk_screening_checks"] = len(results)
    return results, summary
