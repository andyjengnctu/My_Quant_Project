from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import importlib
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

from .checks import add_check


def _capture_stdout(func, *args, **kwargs):
    buf = StringIO()
    with redirect_stdout(buf):
        rc = func(*args, **kwargs)
    return rc, buf.getvalue()


def _capture_stderr(func, *args, **kwargs):
    buf = StringIO()
    with redirect_stderr(buf):
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

    app_package_zip = importlib.import_module("apps.package_zip")
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
        ("apps/package_zip.py", app_package_zip.main),
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


def validate_run_all_cli_error_usage_contract_case(_base_params):
    case_id = "RUN_ALL_CLI_ERROR_USAGE_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from tools.local_regression import run_all

    rc, stderr_text = _capture_stderr(run_all.main, ["tools/local_regression/run_all.py", "--bad"])
    add_check(results, "cli_contract", case_id, "run_all_invalid_flag_main_rc", 2, rc)
    add_check(results, "cli_contract", case_id, "run_all_invalid_flag_error_usage_mentions_meta_quality", True, "meta_quality" in stderr_text)
    add_check(results, "cli_contract", case_id, "run_all_invalid_flag_error_usage_mentions_only", True, "--only" in stderr_text)

    rc, stderr_text = _capture_stderr(run_all.main, ["tools/local_regression/run_all.py", "--only"])
    add_check(results, "cli_contract", case_id, "run_all_missing_only_value_rc", 2, rc)
    add_check(results, "cli_contract", case_id, "run_all_missing_only_value_usage_mentions_meta_quality", True, "meta_quality" in stderr_text)

    return results, summary


def validate_package_zip_runtime_contract_case(_base_params):
    case_id = "PACKAGE_ZIP_RUNTIME_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    app_package_zip = importlib.import_module("apps.package_zip")

    with TemporaryDirectory(prefix="package_zip_contract_") as tmp_dir:
        project_root = Path(tmp_dir)
        (project_root / "apps").mkdir(parents=True)
        (project_root / "arch").mkdir()
        (project_root / "pkg").mkdir()
        (project_root / "pkg" / "module.py").write_text("print('ok')\n", encoding="utf-8")
        (project_root / "README.md").write_text("demo\n", encoding="utf-8")
        (project_root / "pkg" / "__pycache__").mkdir()
        (project_root / "pkg" / "__pycache__" / "module.cpython-312.pyc").write_bytes(b"cache")
        (project_root / "orphan.pyc").write_bytes(b"orphan-cache")
        (project_root / "main_20250101_deadbeef.zip").write_bytes(b"main-old")
        (project_root / "other_branch_20250102_cafebabe.zip").write_bytes(b"other-old")

        fake_now = SimpleNamespace(strftime=lambda fmt: "20260404_123456")

        def _fake_run_git(*args):
            if args == ("rev-parse", "--abbrev-ref", "HEAD"):
                return SimpleNamespace(stdout="feature/runtime-contract\n")
            if args == ("rev-parse", "--short", "HEAD"):
                return SimpleNamespace(stdout="abc1234\n")
            if args == ("ls-files", "--cached", "--others", "--exclude-standard", "-z"):
                return SimpleNamespace(stdout="pkg/module.py\0README.md\0pkg/__pycache__/module.cpython-312.pyc\0orphan.pyc\0")
            raise AssertionError(f"unexpected git args: {args}")

        with patch.object(app_package_zip, "PROJECT_ROOT", project_root), \
             patch.object(app_package_zip, "get_taipei_now", return_value=fake_now), \
             patch.object(app_package_zip, "_run_git", side_effect=_fake_run_git):
            rc, stdout_text = _capture_stdout(app_package_zip.main, ["apps/package_zip.py"])

        new_zip_path = project_root / "feature-runtime-contract_20260404_123456_abc1234.zip"
        archived_root_zips = sorted(path.name for path in (project_root / "arch").glob("*.zip"))
        add_check(results, "cli_contract", case_id, "package_zip_main_rc", 0, rc)
        add_check(results, "cli_contract", case_id, "package_zip_output_exists", True, new_zip_path.exists())
        add_check(results, "cli_contract", case_id, "package_zip_archives_all_root_zips", ["main_20250101_deadbeef.zip", "other_branch_20250102_cafebabe.zip"], archived_root_zips)
        add_check(results, "cli_contract", case_id, "package_zip_root_old_zip_removed", False, (project_root / "other_branch_20250102_cafebabe.zip").exists())
        add_check(results, "cli_contract", case_id, "package_zip_cache_dir_removed", False, (project_root / "pkg" / "__pycache__").exists())
        add_check(results, "cli_contract", case_id, "package_zip_orphan_pyc_removed", False, (project_root / "orphan.pyc").exists())
        add_check(results, "cli_contract", case_id, "package_zip_stdout_reports_archived_count", True, "[package_zip] archived old root zips=2" in stdout_text)

        with zipfile.ZipFile(new_zip_path) as zf:
            member_names = sorted(zf.namelist())
        add_check(results, "cli_contract", case_id, "package_zip_zip_members", ["README.md", "pkg/module.py"], member_names)

    summary["checks"] = len(results)
    return results, summary


def validate_extended_tool_cli_contract_case(_base_params):
    case_id = "CLI_EXTENDED_TOOL_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    app_test_suite = importlib.import_module("apps.test_suite")
    debug_trade_log = importlib.import_module("tools.debug.trade_log")
    optimizer_main = importlib.import_module("tools.optimizer.main")
    portfolio_sim_main = importlib.import_module("tools.portfolio_sim.main")
    scan_runner = importlib.import_module("tools.scanner.scan_runner")
    validate_main = importlib.import_module("tools.validate.main")

    dataset_cases = [
        ("tools/optimizer/main.py", optimizer_main.main, "environ", {}),
        ("tools/portfolio_sim/main.py", portfolio_sim_main.main, "env", {"V16_AUTO_OPEN_BROWSER": "0"}),
        ("tools/scanner/scan_runner.py", scan_runner.main, "env", {"TEST": "1"}),
        ("tools/validate/main.py", validate_main.main, "environ", {}),
        ("tools/debug/trade_log.py", debug_trade_log.main, "environ", {}),
    ]

    for program, main_func, env_kw, env_value in dataset_cases:
        metric_prefix = program.replace("/", "_").replace(".", "_")
        kwargs = {env_kw: env_value}
        rc, help_text = _capture_stdout(main_func, [program, "--help"], **kwargs)
        add_check(results, "cli_contract", case_id, f"{metric_prefix}_help_rc", 0, rc)
        add_check(results, "cli_contract", case_id, f"{metric_prefix}_help_usage", True, f"用法: python {program}" in help_text)
        add_check(results, "cli_contract", case_id, f"{metric_prefix}_help_dataset_choices", True, "reduced|full" in help_text)
        _assert_value_error(results, "cli_contract", case_id, f"{metric_prefix}_invalid_flag_rejected", lambda main_func=main_func, kwargs=kwargs: main_func([program, "--bad"], **kwargs), "不支援的參數")
        _assert_value_error(results, "cli_contract", case_id, f"{metric_prefix}_missing_dataset_value_rejected", lambda main_func=main_func, kwargs=kwargs: main_func([program, "--dataset"], **kwargs), "缺少值")
        _assert_value_error(results, "cli_contract", case_id, f"{metric_prefix}_empty_dataset_value_rejected", lambda main_func=main_func, kwargs=kwargs: main_func([program, "--dataset="], **kwargs), "不能為空")
        _assert_value_error(results, "cli_contract", case_id, f"{metric_prefix}_positional_arg_rejected", lambda main_func=main_func, kwargs=kwargs: main_func([program, "extra"], **kwargs), "不支援的位置參數")

    no_dataset_cases = [
        ("apps/test_suite.py", app_test_suite.main, {}),
    ]

    for program, main_func, kwargs in no_dataset_cases:
        metric_prefix = program.replace("/", "_").replace(".", "_")
        rc, help_text = _capture_stdout(main_func, [program, "--help"], **kwargs)
        add_check(results, "cli_contract", case_id, f"{metric_prefix}_help_rc", 0, rc)
        add_check(results, "cli_contract", case_id, f"{metric_prefix}_help_usage", True, f"用法: python {program}" in help_text)
        _assert_value_error(results, "cli_contract", case_id, f"{metric_prefix}_invalid_flag_rejected", lambda main_func=main_func, kwargs=kwargs: main_func([program, "--bad"], **kwargs), "不支援的參數")
        _assert_value_error(results, "cli_contract", case_id, f"{metric_prefix}_positional_arg_rejected", lambda main_func=main_func, kwargs=kwargs: main_func([program, "extra"], **kwargs), "不支援的位置參數")

    summary["extended_case_count"] = len(dataset_cases) + len(no_dataset_cases)
    return results, summary
