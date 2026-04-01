from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import importlib
from unittest.mock import patch

from .checks import add_check


def _capture_stdout(func, *args, **kwargs):
    buf = StringIO()
    with redirect_stdout(buf):
        rc = func(*args, **kwargs)
    return rc, buf.getvalue()


def _assert_value_error(results, category, case_id, metric_name, func, expected_substring):
    try:
        func()
    except ValueError as exc:
        add_check(results, category, case_id, metric_name, True, expected_substring in str(exc))
    else:
        add_check(results, category, case_id, metric_name, True, False)


def validate_dataset_cli_contract_case(_base_params):
    case_id = "CLI_DATASET_WRAPPER_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    app_ml_optimizer = importlib.import_module("apps.ml_optimizer")
    app_portfolio_sim = importlib.import_module("apps.portfolio_sim")
    app_vip_scanner = importlib.import_module("apps.vip_scanner")
    scanner_main_module = importlib.import_module("tools.scanner.main")
    validate_cli_module = importlib.import_module("tools.validate.cli")

    wrapper_cases = [
        {
            "program": "apps/ml_optimizer.py",
            "module": app_ml_optimizer,
            "patch_target": "tools.optimizer.main",
            "env_kw": "environ",
        },
        {
            "program": "apps/vip_scanner.py",
            "module": app_vip_scanner,
            "patch_target": "tools.scanner.main",
            "env_kw": "env",
        },
        {
            "program": "apps/portfolio_sim.py",
            "module": app_portfolio_sim,
            "patch_target": "tools.portfolio_sim.main",
            "env_kw": "env",
            "extra_kw": {"env": {"V16_AUTO_OPEN_BROWSER": "0"}},
        },
        {
            "program": "tools/scanner/main.py",
            "module": scanner_main_module,
            "patch_target": "tools.scanner.scan_runner.main",
            "env_kw": "env",
        },
        {
            "program": "tools/validate/cli.py",
            "module": validate_cli_module,
            "patch_target": "tools.validate.main",
            "env_kw": "environ",
        },
    ]

    for case in wrapper_cases:
        program = case["program"]
        module = case["module"]
        kwargs = dict(case.get("extra_kw", {}))
        if case.get("env_kw") and case["env_kw"] not in kwargs:
            kwargs[case["env_kw"]] = {"TEST": "1"}

        rc, help_text = _capture_stdout(module.main, [program, "--help"], **kwargs)
        metric_prefix = program.replace("/", "_").replace(".", "_")
        add_check(results, "cli_contract", case_id, f"{metric_prefix}_help_rc", 0, rc)
        add_check(results, "cli_contract", case_id, f"{metric_prefix}_help_usage", True, f"用法: python {program}" in help_text)
        add_check(results, "cli_contract", case_id, f"{metric_prefix}_help_dataset_choices", True, "reduced|full" in help_text)

        sentinel = 17
        with patch(case["patch_target"], return_value=sentinel) as mocked:
            rc = module.main([program], **kwargs)
            add_check(results, "cli_contract", case_id, f"{metric_prefix}_default_passthrough_rc", sentinel, rc)
            add_check(results, "cli_contract", case_id, f"{metric_prefix}_default_passthrough_called", 1, mocked.call_count)
            called_argv = mocked.call_args.kwargs.get("argv")
            add_check(results, "cli_contract", case_id, f"{metric_prefix}_default_passthrough_argv", [program], called_argv)

        with patch(case["patch_target"], return_value=sentinel) as mocked:
            rc = module.main([program, "--dataset", "reduced"], **kwargs)
            add_check(results, "cli_contract", case_id, f"{metric_prefix}_reduced_passthrough_rc", sentinel, rc)
            called_argv = mocked.call_args.kwargs.get("argv")
            add_check(results, "cli_contract", case_id, f"{metric_prefix}_reduced_passthrough_argv", [program, "--dataset", "reduced"], called_argv)

        with patch(case["patch_target"], return_value=sentinel) as mocked:
            rc = module.main([program, "--dataset=full"], **kwargs)
            add_check(results, "cli_contract", case_id, f"{metric_prefix}_inline_full_passthrough_rc", sentinel, rc)
            called_argv = mocked.call_args.kwargs.get("argv")
            add_check(results, "cli_contract", case_id, f"{metric_prefix}_inline_full_passthrough_argv", [program, "--dataset=full"], called_argv)

        _assert_value_error(
            results,
            "cli_contract",
            case_id,
            f"{metric_prefix}_invalid_flag_rejected",
            lambda module=module, kwargs=kwargs: module.main([program, "--bad"], **kwargs),
            "不支援的參數",
        )
        _assert_value_error(
            results,
            "cli_contract",
            case_id,
            f"{metric_prefix}_missing_dataset_value_rejected",
            lambda module=module, kwargs=kwargs: module.main([program, "--dataset"], **kwargs),
            "缺少值",
        )
        _assert_value_error(
            results,
            "cli_contract",
            case_id,
            f"{metric_prefix}_empty_dataset_value_rejected",
            lambda module=module, kwargs=kwargs: module.main([program, "--dataset="], **kwargs),
            "不能為空",
        )
        _assert_value_error(
            results,
            "cli_contract",
            case_id,
            f"{metric_prefix}_positional_arg_rejected",
            lambda module=module, kwargs=kwargs: module.main([program, "extra"], **kwargs),
            "不支援的位置參數",
        )

    summary["wrapper_count"] = len(wrapper_cases)
    return results, summary


def validate_local_regression_cli_contract_case(_base_params):
    case_id = "CLI_LOCAL_REGRESSION_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    app_smart_downloader = importlib.import_module("apps.smart_downloader")
    export_requirements_lock = importlib.import_module("requirements.export_requirements_lock")
    downloader_main_module = importlib.import_module("tools.downloader.main")
    from tools.local_regression import run_all, run_chain_checks, run_meta_quality, run_ml_smoke, run_quick_gate
    from tools.validate import preflight_env

    rc, help_text = _capture_stdout(run_all.main, ["tools/local_regression/run_all.py", "--help"])
    add_check(results, "cli_contract", case_id, "run_all_help_rc", 0, rc)
    add_check(results, "cli_contract", case_id, "run_all_help_mentions_meta_quality", True, "meta_quality" in help_text)
    add_check(results, "cli_contract", case_id, "run_all_help_mentions_only", True, "--only" in help_text)

    parsed_default = run_all._parse_cli_args(["tools/local_regression/run_all.py"])
    add_check(results, "cli_contract", case_id, "run_all_default_only_steps_none", None, parsed_default.get("only_steps"))
    parsed_dedup = run_all._parse_cli_args(["tools/local_regression/run_all.py", "--only", "quick_gate,quick_gate,meta_quality"])
    add_check(results, "cli_contract", case_id, "run_all_only_dedup_normalized", ["quick_gate", "meta_quality"], parsed_dedup.get("only_steps"))
    parsed_inline = run_all._parse_cli_args(["tools/local_regression/run_all.py", "--only=ml_smoke,meta_quality"])
    add_check(results, "cli_contract", case_id, "run_all_only_inline_normalized", ["ml_smoke", "meta_quality"], parsed_inline.get("only_steps"))

    _assert_value_error(results, "cli_contract", case_id, "run_all_only_missing_value_rejected", lambda: run_all._parse_cli_args(["tools/local_regression/run_all.py", "--only"]), "缺少值")
    _assert_value_error(results, "cli_contract", case_id, "run_all_only_empty_value_rejected", lambda: run_all._parse_cli_args(["tools/local_regression/run_all.py", "--only="]), "不可為空")
    _assert_value_error(results, "cli_contract", case_id, "run_all_unknown_flag_rejected", lambda: run_all._parse_cli_args(["tools/local_regression/run_all.py", "--bad"]), "不支援的參數")

    rc, help_text = _capture_stdout(preflight_env.main, ["tools/validate/preflight_env.py", "--help"])
    add_check(results, "cli_contract", case_id, "preflight_help_rc", 0, rc)
    add_check(results, "cli_contract", case_id, "preflight_help_mentions_meta_quality", True, "meta_quality" in help_text)
    parsed_steps_none = preflight_env._parse_cli_steps(["tools/validate/preflight_env.py"])
    add_check(results, "cli_contract", case_id, "preflight_default_steps_none", None, parsed_steps_none)
    parsed_steps = preflight_env._parse_cli_steps(["tools/validate/preflight_env.py", "--steps", "quick_gate,meta_quality"])
    add_check(results, "cli_contract", case_id, "preflight_steps_parse", ["quick_gate", "meta_quality"], parsed_steps)
    normalized_steps = preflight_env._normalize_local_regression_steps(["quick_gate", "quick_gate", "meta_quality"])
    add_check(results, "cli_contract", case_id, "preflight_steps_dedup_normalized", ["quick_gate", "meta_quality"], normalized_steps)
    _assert_value_error(results, "cli_contract", case_id, "preflight_invalid_step_rejected", lambda: preflight_env._normalize_local_regression_steps(["bad"]), "只接受")

    no_arg_cases = [
        ("requirements/export_requirements_lock.py", export_requirements_lock.main),
        ("tools/local_regression/run_chain_checks.py", run_chain_checks.main),
        ("tools/local_regression/run_ml_smoke.py", run_ml_smoke.main),
        ("tools/local_regression/run_meta_quality.py", run_meta_quality.main),
        ("tools/local_regression/run_quick_gate.py", run_quick_gate.main),
        ("apps/smart_downloader.py", app_smart_downloader.main),
        ("tools/downloader/main.py", downloader_main_module.main),
    ]
    for program, main_func in no_arg_cases:
        metric_prefix = program.replace("/", "_").replace(".", "_")
        rc, help_text = _capture_stdout(main_func, [program, "--help"])
        add_check(results, "cli_contract", case_id, f"{metric_prefix}_help_rc", 0, rc)
        add_check(results, "cli_contract", case_id, f"{metric_prefix}_help_usage", True, f"用法: python {program}" in help_text)
        _assert_value_error(results, "cli_contract", case_id, f"{metric_prefix}_unknown_flag_rejected", lambda main_func=main_func, program=program: main_func([program, "--bad"]), "不支援的參數")
        _assert_value_error(results, "cli_contract", case_id, f"{metric_prefix}_positional_arg_rejected", lambda main_func=main_func, program=program: main_func([program, "extra"]), "不支援的位置參數")

    summary["no_arg_case_count"] = len(no_arg_cases)
    return results, summary
