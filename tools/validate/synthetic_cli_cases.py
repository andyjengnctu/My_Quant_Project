from __future__ import annotations

from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from io import StringIO
import importlib
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

from config.breakout_quality import (
    ADAM_WARMUP_COSINE_EXPERIMENT_PROFILE,
    BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
    TRAINING_OBJECTIVE_BINARY_CLASSIFICATION,
    TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
    UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE,
    STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE,
    STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE,
    DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE,
    DAILY_UNIVERSAL_FULL_HORIZON_MFE_ADVERSE_DUAL_MSE_PROFILE,
    DAILY_UNIVERSAL_FULL_HORIZON_PARETO_MFE_LOW_ADVERSE_PAIRWISE_PROFILE,
    DAILY_UNIVERSAL_CONDITIONAL_MFE_SAFETY_FULL_LIST_NDCG_PAIRWISE_PROFILE,
)
from .checks import bind_checks


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
    check, check_true = bind_checks(results, category, case_id)
    try:
        func()
    except ValueError as exc:
        check_true(metric_name, expected_substring in str(exc))
    else:
        check_true(metric_name, False)


def validate_dataset_cli_contract_case(_base_params):
    """Validate generic CLI routing without replaying every historical workflow."""
    case_id = "CLI_DATASET_WRAPPER_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "cli_contract", case_id)

    app = importlib.import_module("tools.filters.breakout_quality.application")
    rc, help_text = _capture_stdout(app.main, ["apps/research.py model", "--help"])
    check("breakout_quality_app_help_rc", 0, rc)
    check_true(
        "breakout_quality_app_help_is_generated_from_command_registry",
        "apps/research.py model" in help_text
        and all(command in help_text for command in app.COMMAND_DESCRIPTIONS),
    )

    with patch.object(app, "is_interactive_console", return_value=False):
        noarg_rc, noarg_text = _capture_stdout(app.main, ["apps/research.py model"])
    check("breakout_quality_noninteractive_no_arg_help_rc", 0, noarg_rc)
    check_true("breakout_quality_noninteractive_no_arg_help", "Breakout quality Dataset" in noarg_text)

    with (
        patch.object(app, "is_interactive_console", return_value=True),
        patch.object(app, "run_model_training_menu", return_value=31) as menu,
    ):
        interactive_rc = app.main(["apps/research.py model"])
    check("breakout_quality_interactive_no_arg_menu_rc", 31, interactive_rc)
    check("breakout_quality_interactive_no_arg_menu_called", 1, menu.call_count)

    command_calls = []
    def _record(command, args, *, program_name):
        command_calls.append((str(command), list(args), str(program_name)))
        return 7

    representative_commands = tuple(app.COMMAND_MODULES)[: min(5, len(app.COMMAND_MODULES))]
    for command in representative_commands:
        with patch.object(app, "_run_command", side_effect=_record):
            dispatch_rc = app.main(
                ["apps/research.py model", command, "--synthetic-flag", "value"]
            )
        check("breakout_quality_registered_command_dispatch_rc", 7, dispatch_rc)
    check_true(
        "breakout_quality_registered_commands_dispatch_without_hardcoded_experiment_ids",
        len(command_calls) == len(representative_commands)
        and all(call[0] in app.COMMAND_MODULES for call in command_calls)
        and all(call[1] == ["--synthetic-flag", "value"] for call in command_calls),
    )

    try:
        app.main(["apps/research.py model", "__unsupported_command__"])
    except ValueError as exc:
        unsupported_rejected = "不支援" in str(exc)
    else:
        unsupported_rejected = False
    check_true("breakout_quality_unknown_command_is_rejected", unsupported_rejected)

    with TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        with patch.object(app, "PROJECT_ROOT", root):
            report_path, report_stdout = _capture_stdout(
                app._emit_breakout_quality_simple_report,
                "evaluate",
                ["--filter-id", "synthetic_quality"],
                returncode=0,
                elapsed_sec=1.25,
            )
        check_true(
            "breakout_quality_simple_report_is_console_and_markdown_with_relative_path",
            report_path.is_file()
            and "Breakout Quality 簡易報表" in report_stdout
            and "outputs/filters/breakout_quality/synthetic_quality/simple_reports/evaluate.md"
            in report_stdout
            and str(root) not in report_stdout,
        )

    from config import breakout_quality as cfg
    current_profiles = [
        profile
        for profile in cfg.SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES
        if cfg.get_continuous_ranker_execution_recipe(profile).current_time_validation_authorized
    ]
    check_true("breakout_quality_cli_has_current_time_validation_profile", bool(current_profiles))

    summary.update(
        {
            "registered_commands": len(app.COMMAND_MODULES),
            "representative_dispatches": len(command_calls),
            "current_time_validation_profiles": current_profiles,
        }
    )
    return results, summary


def validate_local_regression_cli_contract_case(_base_params):
    case_id = "CLI_LOCAL_REGRESSION_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "cli_contract", case_id)

    app_package_zip = importlib.import_module("apps.package_zip")
    app_smart_downloader = importlib.import_module("apps.smart_downloader")
    export_requirements_lock = importlib.import_module("requirements.export_requirements_lock")
    downloader_main_module = importlib.import_module("tools.downloader.main")
    run_all = importlib.import_module("tools.local_regression.run_all")
    run_chain_checks = importlib.import_module("tools.local_regression.run_chain_checks")
    run_meta_quality = importlib.import_module("tools.local_regression.run_meta_quality")
    run_ml_smoke = importlib.import_module("tools.local_regression.run_ml_smoke")
    run_quick_gate = importlib.import_module("tools.local_regression.run_quick_gate")
    preflight_env = importlib.import_module("tools.validate.preflight_env")

    rc, help_text = _capture_stdout(run_all.main, ["tools/local_regression/run_all.py", "--help"])
    check("run_all_help_rc", 0, rc)
    check_true("run_all_help_mentions_meta_quality", "meta_quality" in help_text)
    check_true("run_all_help_mentions_only", "--only" in help_text)

    parsed_default = run_all._parse_cli_args(["tools/local_regression/run_all.py"])
    check("run_all_default_only_steps_none", None, parsed_default.get("only_steps"))
    parsed_dedup = run_all._parse_cli_args(["tools/local_regression/run_all.py", "--only", "quick_gate,quick_gate,meta_quality"])
    check("run_all_only_dedup_normalized", ["quick_gate", "meta_quality"], parsed_dedup.get("only_steps"))
    parsed_inline = run_all._parse_cli_args(["tools/local_regression/run_all.py", "--only=ml_smoke,meta_quality"])
    check("run_all_only_inline_normalized", ["ml_smoke", "meta_quality"], parsed_inline.get("only_steps"))

    _assert_value_error(results, "cli_contract", case_id, "run_all_only_missing_value_rejected", lambda: run_all._parse_cli_args(["tools/local_regression/run_all.py", "--only"]), "缺少值")
    _assert_value_error(results, "cli_contract", case_id, "run_all_only_empty_value_rejected", lambda: run_all._parse_cli_args(["tools/local_regression/run_all.py", "--only="]), "不可為空")
    _assert_value_error(results, "cli_contract", case_id, "run_all_unknown_flag_rejected", lambda: run_all._parse_cli_args(["tools/local_regression/run_all.py", "--bad"]), "不支援的參數")

    rc, help_text = _capture_stdout(preflight_env.main, ["tools/validate/preflight_env.py", "--help"])
    check("preflight_help_rc", 0, rc)
    check_true("preflight_help_mentions_meta_quality", "meta_quality" in help_text)
    parsed_steps_none = preflight_env._parse_cli_steps(["tools/validate/preflight_env.py"])
    check("preflight_default_steps_none", None, parsed_steps_none)
    parsed_steps = preflight_env._parse_cli_steps(["tools/validate/preflight_env.py", "--steps", "quick_gate,meta_quality"])
    check("preflight_steps_parse", ["quick_gate", "meta_quality"], parsed_steps)
    normalized_steps = preflight_env._normalize_local_regression_steps(["quick_gate", "quick_gate", "meta_quality"])
    check("preflight_steps_dedup_normalized", ["quick_gate", "meta_quality"], normalized_steps)
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
        check(f"{metric_prefix}_help_rc", 0, rc)
        check_true(f"{metric_prefix}_help_usage", f"用法: python {program}" in help_text)
        _assert_value_error(results, "cli_contract", case_id, f"{metric_prefix}_unknown_flag_rejected", lambda main_func=main_func, program=program: main_func([program, "--bad"]), "不支援的參數")
        _assert_value_error(results, "cli_contract", case_id, f"{metric_prefix}_positional_arg_rejected", lambda main_func=main_func, program=program: main_func([program, "extra"]), "不支援的位置參數")

    summary["no_arg_case_count"] = len(no_arg_cases)
    return results, summary


def validate_run_all_cli_error_usage_contract_case(_base_params):
    case_id = "RUN_ALL_CLI_ERROR_USAGE_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "cli_contract", case_id)

    run_all = importlib.import_module("tools.local_regression.run_all")

    rc, stderr_text = _capture_stderr(run_all.main, ["tools/local_regression/run_all.py", "--bad"])
    check("run_all_invalid_flag_main_rc", 2, rc)
    check_true("run_all_invalid_flag_error_usage_mentions_meta_quality", "meta_quality" in stderr_text)
    check_true("run_all_invalid_flag_error_usage_mentions_only", "--only" in stderr_text)

    rc, stderr_text = _capture_stderr(run_all.main, ["tools/local_regression/run_all.py", "--only"])
    check("run_all_missing_only_value_rc", 2, rc)
    check_true("run_all_missing_only_value_usage_mentions_meta_quality", "meta_quality" in stderr_text)

    return results, summary


def validate_package_zip_runtime_contract_case(_base_params):
    case_id = "PACKAGE_ZIP_RUNTIME_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "cli_contract", case_id)

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
        (project_root / "busy" / "__pycache__").mkdir(parents=True)
        (project_root / "busy" / "__pycache__" / "busy.cpython-312.pyc").write_bytes(b"busy-cache")
        (project_root / "orphan.pyc").write_bytes(b"orphan-cache")
        (project_root / "main_20250101_deadbeef.zip").write_bytes(b"main-old")
        (project_root / "other_branch_20250102_cafebabe.zip").write_bytes(b"other-old")
        (project_root / "to_chatgpt_bundle_20250103_deadbeef.zip").write_bytes(b"bundle-old")

        fake_now = SimpleNamespace(strftime=lambda fmt: "20260404_123456")

        def _fake_run_git(*args):
            if args == ("rev-parse", "--abbrev-ref", "HEAD"):
                return SimpleNamespace(stdout="feature/runtime-contract\n")
            if args == ("rev-parse", "--short", "HEAD"):
                return SimpleNamespace(stdout="abc1234\n")
            if args == ("ls-files", "--cached", "--others", "--exclude-standard", "-z"):
                return SimpleNamespace(
                    stdout=(
                        "pkg/module.py\0README.md\0"
                        "pkg/__pycache__/module.cpython-312.pyc\0"
                        "busy/__pycache__/busy.cpython-312.pyc\0orphan.pyc\0"
                    )
                )
            raise AssertionError(f"unexpected git args: {args}")

        real_rmtree = app_package_zip.shutil.rmtree
        transient_cache_dir = project_root / "pkg" / "__pycache__"
        busy_cache_dir = project_root / "busy" / "__pycache__"
        rmtree_attempts = Counter()

        def _windows_rmtree_probe(path, ignore_errors=False):
            target = Path(path)
            rmtree_attempts[target] += 1
            if target == transient_cache_dir and rmtree_attempts[target] == 1:
                raise OSError(145, "The directory is not empty")
            if target == busy_cache_dir:
                raise OSError(145, "The directory is not empty")
            return real_rmtree(path, ignore_errors=ignore_errors)

        with patch.object(app_package_zip, "PROJECT_ROOT", project_root), \
             patch.object(app_package_zip, "get_taipei_now", return_value=fake_now), \
             patch.object(app_package_zip, "_run_git", side_effect=_fake_run_git), \
             patch.object(app_package_zip.shutil, "rmtree", side_effect=_windows_rmtree_probe), \
             patch.object(app_package_zip.time, "sleep", return_value=None):
            rc, stdout_text = _capture_stdout(app_package_zip.main, ["apps/package_zip.py"])

        new_zip_path = project_root / "feature-runtime-contract_20260404_123456_abc1234.zip"
        archived_root_zips = sorted(path.name for path in (project_root / "arch").glob("*.zip"))
        check("package_zip_main_rc", 0, rc)
        check_true("package_zip_output_exists", new_zip_path.exists())
        check(
            "package_zip_archives_non_bundle_root_zips_only",
            ["main_20250101_deadbeef.zip", "other_branch_20250102_cafebabe.zip"],
            archived_root_zips,
        )
        check(
            "package_zip_root_old_zip_removed",
            False,
            (project_root / "other_branch_20250102_cafebabe.zip").exists(),
        )
        check_true(
            "package_zip_root_bundle_preserved",
            (project_root / "to_chatgpt_bundle_20250103_deadbeef.zip").exists(),
        )
        check(
            "package_zip_root_bundle_not_archived",
            False,
            (project_root / "arch" / "to_chatgpt_bundle_20250103_deadbeef.zip").exists(),
        )
        check("package_zip_cache_dir_removed", False, transient_cache_dir.exists())
        check_true(
            "package_zip_cache_cleanup_retries_transient_winerror_145",
            rmtree_attempts[transient_cache_dir] >= 2,
        )
        check_true(
            "package_zip_cache_cleanup_persistent_winerror_145_does_not_abort",
            busy_cache_dir.exists()
            and "warning=python_cache_cleanup_incomplete" in stdout_text
            and "package exclusion remains enforced" in stdout_text,
        )
        check("package_zip_orphan_pyc_removed", False, (project_root / "orphan.pyc").exists())
        check_true(
            "package_zip_stdout_reports_archived_count",
            "[package_zip] archived old root zips=2" in stdout_text,
        )

        with zipfile.ZipFile(new_zip_path) as zf:
            member_names = sorted(zf.namelist())
        check("package_zip_zip_members", ["README.md", "pkg/module.py"], member_names)

    summary["checks"] = len(results)
    return results, summary



def validate_package_zip_commit_test_suite_orchestration_case(_base_params):
    case_id = "PACKAGE_ZIP_COMMIT_TEST_SUITE_ORCHESTRATION"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "cli_contract", case_id)

    app_package_zip = importlib.import_module("apps.package_zip")

    with TemporaryDirectory(prefix="package_zip_orchestration_") as tmp_dir:
        project_root = Path(tmp_dir)
        (project_root / "apps").mkdir(parents=True)
        (project_root / "pkg").mkdir()
        (project_root / "pkg" / "module.py").write_text("print('ok')\n", encoding="utf-8")
        (project_root / "README.md").write_text("demo\n", encoding="utf-8")
        (project_root / "legacy_20250101_deadbeef.zip").write_bytes(b"legacy")
        (project_root / "to_chatgpt_bundle_20250103_deadbeef.zip").write_bytes(b"bundle-old")

        fake_now = SimpleNamespace(strftime=lambda fmt: "20260405_120000")
        git_commands = []
        python_commands = []

        def _fake_run_git(*args):
            git_commands.append(args)
            if args == ("status", "--porcelain", "--untracked-files=all"):
                return SimpleNamespace(stdout=" M apps/package_zip.py\n?? README.md\n")
            if args == ("add", "-A"):
                return SimpleNamespace(stdout="")
            if args == ("commit", "-m", "feat: package workflow"):
                return SimpleNamespace(stdout="[feature/workflow fedcba9] feat: package workflow\n")
            if args == ("rev-parse", "--abbrev-ref", "HEAD"):
                return SimpleNamespace(stdout="feature/workflow\n")
            if args == ("rev-parse", "--short", "HEAD"):
                return SimpleNamespace(stdout="fedcba9\n")
            if args == ("ls-files", "--cached", "--others", "--exclude-standard", "-z"):
                return SimpleNamespace(stdout="README.md\0pkg/module.py\0")
            raise AssertionError(f"unexpected git args: {args}")

        def _fake_run_python_command(command):
            python_commands.append(command)
            return SimpleNamespace(returncode=0)

        with patch.object(app_package_zip, "PROJECT_ROOT", project_root),              patch.object(app_package_zip, "get_taipei_now", return_value=fake_now),              patch.object(app_package_zip, "_run_git", side_effect=_fake_run_git),              patch.object(app_package_zip, "_run_python_command", side_effect=_fake_run_python_command):
            rc, stdout_text = _capture_stdout(
                app_package_zip.main,
                ["apps/package_zip.py", "--commit-message", "feat: package workflow", "--run-test-suite"],
            )

        new_zip_path = project_root / "feature-workflow_20260405_120000_fedcba9.zip"
        commit_idx = git_commands.index(("commit", "-m", "feat: package workflow"))
        ls_files_idx = git_commands.index(("ls-files", "--cached", "--others", "--exclude-standard", "-z"))
        check("package_zip_orchestration_main_rc", 0, rc)
        check_true("package_zip_orchestration_add_called", ("add", "-A") in git_commands)
        check_true(
            "package_zip_orchestration_commit_called",
            ("commit", "-m", "feat: package workflow") in git_commands,
        )
        check_true("package_zip_orchestration_commit_precedes_zip_snapshot", commit_idx < ls_files_idx)
        check_true("package_zip_orchestration_output_uses_post_commit_sha", new_zip_path.exists())
        check(
            "package_zip_orchestration_test_suite_called",
            [[sys.executable, "apps/test_suite.py"]],
            python_commands,
        )
        relative_zip_output = f"[package_zip] output={new_zip_path.name}"
        check_true(
            "package_zip_orchestration_output_is_project_relative",
            relative_zip_output in stdout_text and str(project_root).replace("\\", "/") not in stdout_text,
        )
        check_true(
            "package_zip_orchestration_test_suite_after_zip",
            stdout_text.index(relative_zip_output) < stdout_text.index("[package_zip] test_suite=pass"),
        )
        check_true(
            "package_zip_orchestration_commit_headline_reported",
            "[package_zip] commit=[feature/workflow fedcba9] feat: package workflow" in stdout_text,
        )
        check_true(
            "package_zip_orchestration_bundle_preserved",
            (project_root / "to_chatgpt_bundle_20250103_deadbeef.zip").exists(),
        )

    run_bundle_source = (Path(__file__).resolve().parents[2] / "apps" / "run_bundle.py").read_text(encoding="utf-8")
    commit_marker = 'print(f"[3/5] Commit current snapshot: {message}")'
    package_marker = 'print("[4/5] Run package_zip.py before formal tests")'
    test_marker = 'print("[5/5] Run test_suite.py after package")'
    delivery_first_order = (
        commit_marker in run_bundle_source
        and package_marker in run_bundle_source
        and test_marker in run_bundle_source
        and run_bundle_source.index(commit_marker)
        < run_bundle_source.index(package_marker)
        < run_bundle_source.index(test_marker)
    )
    check_true("run_bundle_commits_and_packages_before_formal_test", delivery_first_order)
    check_true(
        "run_bundle_formal_failure_keeps_prebuilt_delivery_snapshot",
        'run_cmd(["git", "commit", "-m", message]' in run_bundle_source
                and 'run_cmd([sys.executable, "apps/package_zip.py"]' in run_bundle_source
                and 'run_cmd([sys.executable, "apps/test_suite.py"]' in run_bundle_source
                and run_bundle_source.index('run_cmd(["git", "commit", "-m", message]')
                < run_bundle_source.index('run_cmd([sys.executable, "apps/package_zip.py"]')
                < run_bundle_source.index('run_cmd([sys.executable, "apps/test_suite.py"]'),
    )

    summary["checks"] = len(results)
    return results, summary


def validate_breakout_quality_app_simple_report_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_APP_SIMPLE_REPORT_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "output_contract", case_id)
    app_breakout_quality = importlib.import_module("tools.filters.breakout_quality.application")

    with TemporaryDirectory() as tmp_dir:
        simple_root = Path(tmp_dir)
        with patch.object(app_breakout_quality, "PROJECT_ROOT", simple_root):
            report_path, console_text = _capture_stdout(
                app_breakout_quality._emit_breakout_quality_simple_report,
                "evaluate",
                ["--filter-id", "synthetic_quality"],
                returncode=0,
                elapsed_sec=1.25,
            )
        markdown = report_path.read_text(encoding="utf-8")
        check_true(
            "breakout_quality_app_console_simple_report",
            "Breakout Quality 簡易報表" in console_text and "狀態" in console_text,
        )
        check_true(
            "breakout_quality_app_persistent_markdown_simple_report",
            report_path.is_file()
            and "Breakout Quality 簡易報表" in markdown
            and "#42A5F5" in markdown,
        )
        check_true(
            "breakout_quality_app_simple_report_paths_are_project_relative",
            "outputs/filters/breakout_quality/synthetic_quality/simple_reports/evaluate.md" in console_text
                        and str(simple_root) not in console_text,
        )
        check_true(
            "breakout_quality_app_simple_report_contains_active_identity",
            "synthetic_quality" in markdown and "Objective" in markdown,
        )

        pit_manifest_path = app_breakout_quality.resolve_selection_point_in_time_manifest_path(
            simple_root,
            "synthetic_quality",
            BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
            STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE,
        )
        pit_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        pit_manifest_path.write_text(
            json.dumps(
                {
                    "coverage": {
                        "scored_group_count": 778532,
                        "expected_group_count": 778532,
                        "coverage_rate": 1.0,
                    }
                }
            ),
            encoding="utf-8",
        )
        with patch.object(app_breakout_quality, "PROJECT_ROOT", simple_root):
            pit_report_path, pit_console = _capture_stdout(
                app_breakout_quality._emit_breakout_quality_simple_report,
                "build-point-in-time-scores",
                [
                    "--filter-id", "synthetic_quality",
                    "--model-architecture", BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
                    "--experiment-profile", STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE,
                ],
                returncode=0,
                elapsed_sec=0.5,
            )
        pit_markdown = pit_report_path.read_text(encoding="utf-8")
        check_true(
            "breakout_quality_pit_build_simple_report_uses_manifest_coverage_before_audit_exists",
            "Score coverage" in pit_console
                        and "100.00%" in pit_console
                        and "778532" in pit_console
                        and "100.00%" in pit_markdown,
        )
        check_true(
            "breakout_quality_pit_build_simple_report_omits_audit_only_metrics",
            "Daily rho" not in pit_console and "Global rho" not in pit_console,
        )

        pit_override = simple_root / "models" / "synthetic_pit_override"
        pit_override.mkdir(parents=True, exist_ok=True)
        (pit_override / "selection_point_in_time_manifest.json").write_text(
            json.dumps(
                {
                    "coverage": {
                        "scored_group_count": 630589,
                        "expected_group_count": 630589,
                        "coverage_rate": 1.0,
                    }
                }
            ),
            encoding="utf-8",
        )
        (pit_override / "selection_point_in_time_audit.json").write_text(
            json.dumps(
                {
                    "score_coverage": {
                        "scored_group_count": 630589,
                        "expected_group_count": 630589,
                        "coverage_rate": 1.0,
                    },
                    "decision_contract": {"primary_metric_scope": "all_valid_target"},
                    "metrics": {
                        "all_valid_target": {
                            "mean_daily_spearman": 0.2119,
                            "global_spearman": 0.1642,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        (pit_override / "selection_point_in_time_audit.md").write_text(
            "# Synthetic PIT audit\n",
            encoding="utf-8",
        )
        with patch.object(app_breakout_quality, "PROJECT_ROOT", simple_root):
            _audit_report_path, audit_console = _capture_stdout(
                app_breakout_quality._emit_breakout_quality_simple_report,
                "audit-point-in-time-scores",
                [
                    "--filter-id", "synthetic_quality",
                    "--model-architecture", BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
                    "--experiment-profile", STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE,
                    "--point-in-time-dir-override", str(pit_override),
                ],
                returncode=0,
                elapsed_sec=0.75,
            )
        check_true(
            "breakout_quality_pit_audit_simple_report_reads_actual_override_artifacts",
            "Score coverage" in audit_console
            and "100.00%" in audit_console
            and "630589" in audit_console
            and "0.2119" in audit_console
            and "0.1642" in audit_console
            and "None" not in audit_console,
        )

        conditional_profile = DAILY_UNIVERSAL_CONDITIONAL_MFE_SAFETY_FULL_LIST_NDCG_PAIRWISE_PROFILE
        conditional_architecture = "inception_time_conditional_mfe_safety_v1"
        conditional_output_dir = app_breakout_quality.resolve_filter_model_output_dir(
            simple_root,
            "synthetic_quality",
            conditional_architecture,
            conditional_profile,
        )
        conditional_output_dir.mkdir(parents=True, exist_ok=True)
        conditional_payload = {
            "training": {
                "selected_epoch": 1,
                "epoch_selection": {
                    "best_validation_mean_daily_spearman": 0.3125,
                },
            },
            "split_metrics": {
                "validation": {
                    "group_count": 203481,
                    "mean_daily_spearman": 0.4348,
                    "global_spearman_vs_raw_target": 0.4335,
                    "pairwise_concordance": 0.6507,
                    "top_score_decile_raw_target_mean": 2.6964,
                    "bottom_score_decile_raw_target_mean": 0.5097,
                },
                "oos": {
                    "group_count": 608204,
                    "mean_daily_spearman": 0.3953,
                    "global_spearman_vs_raw_target": 0.3019,
                    "pairwise_concordance": 0.6364,
                    "top_score_decile_raw_target_mean": 2.2762,
                    "bottom_score_decile_raw_target_mean": 0.6486,
                },
                "breakout_candidate_oos": {
                    "group_count": 17346,
                    "mean_daily_spearman": 0.3736,
                    "global_spearman_vs_raw_target": 0.3771,
                    "pairwise_concordance": 0.6441,
                    "top_score_decile_raw_target_mean": 3.1338,
                    "bottom_score_decile_raw_target_mean": 0.5352,
                },
            },
            "conditional_mfe_safety_evaluation": {
                "validation": {
                    "primary_mfe": {
                        "mean_daily_spearman": 0.4348,
                        "global_spearman_vs_raw_target": 0.4335,
                        "pairwise_concordance": 0.6507,
                    },
                    "conditional_safety": {
                        "mean_daily_spearman": 0.3210,
                        "global_spearman_vs_raw_target": 0.3100,
                        "pairwise_concordance": 0.6120,
                    },
                },
                "oos": {
                    "primary_mfe": {
                        "mean_daily_spearman": 0.3953,
                        "global_spearman_vs_raw_target": 0.3019,
                        "pairwise_concordance": 0.6364,
                    },
                    "conditional_safety": {
                        "mean_daily_spearman": 0.2876,
                        "global_spearman_vs_raw_target": 0.2744,
                        "pairwise_concordance": 0.5988,
                    },
                },
                "breakout_candidate_oos": {
                    "primary_mfe": {
                        "mean_daily_spearman": 0.3736,
                        "global_spearman_vs_raw_target": 0.3771,
                        "pairwise_concordance": 0.6441,
                    },
                    "conditional_safety": {
                        "mean_daily_spearman": 0.3012,
                        "global_spearman_vs_raw_target": 0.2955,
                        "pairwise_concordance": 0.6077,
                    },
                },
            },
        }
        (conditional_output_dir / app_breakout_quality.CONTINUOUS_RANKER_REPORT_FILENAME).write_text(
            json.dumps(conditional_payload),
            encoding="utf-8",
        )
        with patch.object(app_breakout_quality, "PROJECT_ROOT", simple_root):
            conditional_report_path, conditional_console = _capture_stdout(
                app_breakout_quality._emit_breakout_quality_simple_report,
                "train-continuous-ranker",
                [
                    "--filter-id", "synthetic_quality",
                    "--model-architecture", conditional_architecture,
                    "--experiment-profile", conditional_profile,
                ],
                returncode=0,
                elapsed_sec=2.0,
            )
        conditional_markdown = conditional_report_path.read_text(encoding="utf-8")
        check_true(
            "breakout_quality_mr13p_simple_report_uses_standard_model_sop_with_conditional_head_evidence",
            "標準模型 SOP｜1. Learnability" in conditional_console
            and "標準模型 SOP｜2. Generalization" in conditional_console
            and "標準模型 SOP｜3. Multi-head Learnability" in conditional_console
            and "Forward OOS" in conditional_console
            and "Conditional Safety" in conditional_console
            and "0.2876" in conditional_console
            and "59.88%" in conditional_console
            and "標準模型 SOP｜1. Learnability" in conditional_markdown
            and "標準模型 SOP｜3. Multi-head Learnability｜Conditional MFE-Safety" in conditional_markdown
            and "0.2876" in conditional_markdown
            and "59.88%" in conditional_markdown
            and "Actual Round-trip R" not in conditional_markdown,
        )
        with patch.object(app_breakout_quality, "console_color_enabled", return_value=True):
            colored_model_console = app_breakout_quality._render_continuous_ranker_simple_console(
                conditional_payload
            )
        check_true(
            "breakout_quality_model_sop_uses_shared_color_palette_for_sections_generalization_and_evidence",
            "\x1b[96m" in colored_model_console
            and "\x1b[91m" in colored_model_console
            and "\x1b[92m" in colored_model_console
            and "標準模型 SOP｜1. Learnability" in colored_model_console
            and "標準模型 SOP｜2. Generalization" in colored_model_console,
        )
        check_true(
            "breakout_quality_model_sop_markdown_uses_shared_blue_and_semantic_colors",
            "#42A5F5" in conditional_markdown
            and "標準模型 SOP｜1. Learnability" in conditional_markdown,
        )

        compare_dir = (
            simple_root / "outputs" / "filters" / "breakout_quality"
            / "synthetic_quality" / "continuous_ranker_comparison"
        )
        compare_dir.mkdir(parents=True, exist_ok=True)
        (compare_dir / "continuous_ranker_comparison.json").write_text(
            json.dumps(
                {
                    "comparison_settings": {
                        "model_ids": ["CFG-BASE", "CFG-CURRENT"],
                        "reference_arm": "CFG-REF",
                        "summary_pair": ["CFG-CURRENT", "CFG-BASE"],
                    },
                    "fixed_k": {
                        "splits": {
                            "oos": {
                                "competition_date_count": 20,
                                "models": {
                                    "CFG-CURRENT": {
                                        "ndcg_at_k": 0.71,
                                        "boundary_concordance": 0.54,
                                    }
                                },
                            }
                        }
                    },
                    "dynamic_k": {
                        "coverage": {"full_score_coverage_date_count": 18},
                        "evaluation": {
                            "models": {
                                "CFG-CURRENT": {"boundary_concordance": 0.56}
                            },
                            "paired_contrasts": {
                                "CFG-CURRENT_minus_CFG-BASE": {
                                    "metrics": {
                                        "boundary_concordance": {"mean_delta": 0.03}
                                    }
                                }
                            },
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        (compare_dir / "continuous_ranker_comparison.md").write_text(
            "# comparison\n", encoding="utf-8"
        )
        with patch.object(app_breakout_quality, "PROJECT_ROOT", simple_root):
            compare_report_path, compare_console = _capture_stdout(
                app_breakout_quality._emit_breakout_quality_simple_report,
                "compare-continuous-rankers",
                [
                    "--filter-id", "synthetic_quality",
                    "--model-architecture", BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
                ],
                returncode=0,
                elapsed_sec=0.5,
            )
        compare_markdown = compare_report_path.read_text(encoding="utf-8")
        check(
            "breakout_quality_ranker_comparison_simple_report_is_readable_and_not_single_profile_misleading",
            (True, True, True),
            (
                            "CFG-BASE / CFG-CURRENT" in compare_console
                            and "paired ranking-quality comparison (read-only)" in compare_console,
                            "Dynamic Target基準" in compare_console
                            and "原始score-event-date（非trade-date剩餘機會）" in compare_console
                            and "CFG-CURRENT Dynamic raw-score Boundary" in compare_console
                            and "CFG-CURRENT−CFG-BASE Dynamic raw-score Boundary Δ" in compare_console,
                            "continuous_ranker_comparison.md" in compare_markdown
                            and str(simple_root) not in compare_console,
                        ),
        )

    fake_success_module = SimpleNamespace(main=lambda _args: 0)
    with patch.object(
        app_breakout_quality,
        "_load_command_module",
        return_value=fake_success_module,
    ), patch.object(
        app_breakout_quality,
        "_emit_breakout_quality_simple_report",
    ) as emit_mock:
        generic_rc = app_breakout_quality._run_command(
            "evaluate",
            [],
            program_name="apps/research.py model",
        )
        generic_emit_ok = (
            generic_rc == 0
            and emit_mock.call_count == 1
            and emit_mock.call_args.args[0] == "evaluate"
        )
        emit_mock.reset_mock()
        timing_run_rc = app_breakout_quality._run_command(
            "timing-rolling-training",
            ["run"],
            program_name="apps/research.py model",
        )
        timing_run_emit_ok = (
            timing_run_rc == 0
            and emit_mock.call_count == 1
            and emit_mock.call_args.args[0] == "timing-rolling-training"
        )
        emit_mock.reset_mock()
        timing_status_rc = app_breakout_quality._run_command(
            "timing-rolling-training",
            ["status"],
            program_name="apps/research.py model",
        )
        timing_status_does_not_overwrite = (
            timing_status_rc == 0 and emit_mock.call_count == 0
        )

    check_true("breakout_quality_app_successful_subcommands_emit_simple_report", generic_emit_ok)
    check_true(
        "breakout_quality_timing_run_emits_simple_report_without_status_overwrite",
        timing_run_emit_ok and timing_status_does_not_overwrite,
    )
    # Rolling Timing is a separately authorized workflow.  The active model-research
    # profile may intentionally be MODEL_GATE_ONLY (for example MR-13P), so this
    # output-contract test must use an isolated authorized timing profile instead of
    # mutating or implicitly broadening the current research authorization.
    from config import breakout_quality as breakout_quality_config

    authorized_timing_profiles = [
        profile_name
        for profile_name in breakout_quality_config.SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES
        if breakout_quality_config.get_continuous_ranker_execution_recipe(
            profile_name
        ).current_time_validation_authorized
    ]
    if not authorized_timing_profiles:
        raise AssertionError("synthetic timing report需要至少一個已授權current-time profile")
    with patch.object(
        breakout_quality_config,
        "BREAKOUT_QUALITY_ROLLING_TIMING_EXPERIMENT_PROFILE",
        authorized_timing_profiles[0],
    ):
        timing_context = app_breakout_quality._simple_report_context(
            "timing-rolling-training",
            ["run"],
        )
        timing_settings = app_breakout_quality.get_breakout_quality_rolling_timing_settings()
    check_true(
        "breakout_quality_timing_simple_report_uses_timing_profile_identity",
        timing_context[2] == str(timing_settings.experiment_profile)
        and timing_context[2] == str(authorized_timing_profiles[0]),
    )

    source = Path(app_breakout_quality.__file__).read_text(encoding="utf-8")
    check_true(
        "breakout_quality_app_workflow_emits_final_simple_report",
        '"workflow",' in source and "workflow_report_args" in source,
    )

    summary["checks"] = len(results)
    return results, summary


def validate_extended_tool_cli_contract_case(_base_params):
    case_id = "CLI_EXTENDED_TOOL_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "cli_contract", case_id)

    app_test_suite = importlib.import_module("apps.test_suite")
    app_workbench = importlib.import_module("apps.workbench")
    optimizer_main = importlib.import_module("tools.optimizer.main")
    portfolio_sim_main = importlib.import_module("tools.portfolio_sim.main")
    scan_runner = importlib.import_module("tools.scanner.scan_runner")
    validate_main = importlib.import_module("tools.validate.main")

    dataset_cases = [
        ("tools/optimizer/main.py", optimizer_main.main, "environ", {}),
        ("tools/portfolio_sim/main.py", portfolio_sim_main.main, "env", {"V16_AUTO_OPEN_BROWSER": "0"}),
        ("tools/scanner/scan_runner.py", scan_runner.main, "env", {"TEST": "1"}),
        ("tools/validate/main.py", validate_main.main, "environ", {}),
    ]

    for program, main_func, env_kw, env_value in dataset_cases:
        metric_prefix = program.replace("/", "_").replace(".", "_")
        kwargs = {env_kw: env_value}
        rc, help_text = _capture_stdout(main_func, [program, "--help"], **kwargs)
        check(f"{metric_prefix}_help_rc", 0, rc)
        check_true(f"{metric_prefix}_help_usage", f"用法: python {program}" in help_text)
        check_true(f"{metric_prefix}_help_dataset_choices", "reduced|full" in help_text)
        _assert_value_error(results, "cli_contract", case_id, f"{metric_prefix}_invalid_flag_rejected", lambda main_func=main_func, kwargs=kwargs: main_func([program, "--bad"], **kwargs), "不支援的參數")
        _assert_value_error(results, "cli_contract", case_id, f"{metric_prefix}_missing_dataset_value_rejected", lambda main_func=main_func, kwargs=kwargs: main_func([program, "--dataset"], **kwargs), "缺少值")
        _assert_value_error(results, "cli_contract", case_id, f"{metric_prefix}_empty_dataset_value_rejected", lambda main_func=main_func, kwargs=kwargs: main_func([program, "--dataset="], **kwargs), "不能為空")
        _assert_value_error(results, "cli_contract", case_id, f"{metric_prefix}_positional_arg_rejected", lambda main_func=main_func, kwargs=kwargs: main_func([program, "extra"], **kwargs), "不支援的位置參數")

    no_dataset_cases = [
        ("apps/test_suite.py", app_test_suite.main, {}),
        ("apps/workbench.py", app_workbench.main, {}),
    ]

    for program, main_func, kwargs in no_dataset_cases:
        metric_prefix = program.replace("/", "_").replace(".", "_")
        rc, help_text = _capture_stdout(main_func, [program, "--help"], **kwargs)
        check(f"{metric_prefix}_help_rc", 0, rc)
        check_true(f"{metric_prefix}_help_usage", f"用法: python {program}" in help_text)
        _assert_value_error(results, "cli_contract", case_id, f"{metric_prefix}_invalid_flag_rejected", lambda main_func=main_func, kwargs=kwargs: main_func([program, "--bad"], **kwargs), "不支援的參數")
        _assert_value_error(results, "cli_contract", case_id, f"{metric_prefix}_positional_arg_rejected", lambda main_func=main_func, kwargs=kwargs: main_func([program, "extra"], **kwargs), "不支援的位置參數")

    summary["extended_case_count"] = len(dataset_cases) + len(no_dataset_cases)
    return results, summary
