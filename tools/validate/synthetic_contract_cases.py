import csv
import io
import json
import os
import sys
import tempfile
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from openpyxl import load_workbook

from core.output_retention import RetentionRule, apply_retention_rules
from core.data_utils import get_required_min_rows, sanitize_ohlcv_dataframe
from tools.local_regression import common as local_common
from tools.local_regression import run_all as run_all_module
from tools.local_regression import run_meta_quality as run_meta_quality_module
from tools.local_regression import run_quick_gate as run_quick_gate_module
from tools.optimizer.profile import OptimizerProfileRecorder, PROFILE_FIELDS
from tools.local_regression.common import LOCAL_REGRESSION_RUN_DIR_ENV, write_json, write_csv, write_text
from tools.validate.reporting import write_issue_excel_report, write_local_regression_summary
from core.portfolio_fast_data import prep_stock_data_and_trades

from .checks import add_check, make_synthetic_validation_params, run_scanner_reference_check, run_scanner_reference_check_on_clean_df
from .synthetic_case_builders import build_synthetic_competing_candidates_case
from .synthetic_frame_utils import write_synthetic_csv_bundle
from .tool_adapters import (
    run_debug_trade_log_check,
    run_portfolio_sim_tool_check,
    run_portfolio_sim_tool_check_for_dir,
    run_scanner_tool_check,
)


REQUIRED_VALIDATE_SUMMARY_KEYS = {
    "status",
    "dataset",
    "dataset_source",
    "data_dir",
    "csv_path",
    "xlsx_path",
    "output_dir",
    "elapsed_time_sec",
    "real_ticker_count",
    "real_data_coverage_ok",
    "total_checks",
    "pass_count",
    "skip_count",
    "fail_count",
    "peak_traced_memory_mb",
}

REQUIRED_PROFILE_SUMMARY_AVG_KEYS = {
    "objective_wall_sec",
    "prep_wall_sec",
    "portfolio_wall_sec",
    "score_calc_sec",
}

REQUIRED_PREFLIGHT_SUMMARY_KEYS = {
    "status",
    "python_executable",
    "duration_sec",
    "failed_packages",
}

REQUIRED_DATASET_PREP_SUMMARY_KEYS = {
    "status",
    "duration_sec",
    "dataset_dir",
    "source",
    "csv_count",
    "csv_total_bytes",
    "csv_members_sha256",
    "csv_content_sha256",
    "fingerprint_algorithm",
    "reused_existing",
}

REQUIRED_CHAIN_SUMMARY_KEYS = {
    "status",
    "dataset",
    "dataset_info",
    "ticker_count",
    "csv_count",
    "duplicate_issue_count",
    "skipped_ticker_count",
    "detail_count",
    "rows",
    "highlights",
    "portfolio_snapshot",
    "scanner_snapshot",
    "rerun_consistency",
    "failures",
    "peak_traced_memory_mb",
}

CHAIN_SUMMARY_CSV_FIELDS = [
    "ticker",
    "single_trade_count",
    "single_missed_buys",
    "setup_days",
    "pit_pass_days",
    "candidate_days",
    "orderable_days",
    "portfolio_trade_rows",
    "filled_count",
    "portfolio_missed_buy_count",
    "debug_row_count",
    "blocked_by",
    "sanitize_dropped",
]

def _normalize_nan_records(df):
    if df is None:
        return None

    def _normalize_value(value):
        return None if pd.isna(value) else value

    records = []
    for row in df.to_dict("records"):
        records.append({key: _normalize_value(value) for key, value in row.items()})
    return records


REQUIRED_ML_SMOKE_SUMMARY_KEYS = {
    "status",
    "dataset",
    "dataset_info",
    "db_path",
    "db_trial_count",
    "qualified_trial_count",
    "best_trial_value",
    "best_params_path",
    "best_params_required",
    "best_params_keys",
    "optimizer_profile_summary_path",
    "optimizer_profile_trial_count",
    "optimizer_profile_avg_objective_wall_sec",
    "optimizer_repro",
    "failures",
    "peak_traced_memory_mb",
}

REQUIRED_QUICK_GATE_SUMMARY_KEYS = {
    "status",
    "dataset",
    "dataset_info",
    "step_count",
    "failed_count",
    "failed_steps",
    "steps",
    "duration_sec",
    "peak_traced_memory_mb",
}

REQUIRED_META_QUALITY_SUMMARY_KEYS = {
    "status",
    "failures",
    "fail_count",
    "coverage",
    "checklist",
    "formal_entry",
    "performance",
    "results",
}

REQUIRED_MASTER_SUMMARY_KEYS = {
    "overall_status",
    "dataset",
    "dataset_info",
    "timestamp",
    "git_commit",
    "scripts",
    "selected_steps",
    "failures",
    "preflight",
    "dataset_prepare",
    "payload_failures",
    "not_run_step_names",
    "bundle_mode",
    "bundle_entries",
    "failed_step_names",
    "suggested_rerun_command",
}


def _read_csv_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return reader.fieldnames or [], rows


def validate_output_contract_case(_base_params):
    case_id = "OUTPUT_CONTRACT_SCHEMA"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    with tempfile.TemporaryDirectory(prefix="output_contract_") as temp_dir:
        temp_path = Path(temp_dir)

        # JSON contract: validate_consistency_summary.json
        run_dir = temp_path / "run_dir"
        run_dir.mkdir(parents=True, exist_ok=True)
        prev_run_dir = os.environ.get(LOCAL_REGRESSION_RUN_DIR_ENV)
        os.environ[LOCAL_REGRESSION_RUN_DIR_ENV] = str(run_dir)
        try:
            df_results = pd.DataFrame(
                [
                    {"ticker": "1101", "status": "PASS"},
                    {"ticker": "1102", "status": "FAIL"},
                    {"ticker": "1103", "status": "SKIP"},
                ]
            )
            df_failed = pd.DataFrame(
                [
                    {
                        "ticker": "1102",
                        "module": "portfolio",
                        "metric": "final_equity",
                        "passed": False,
                        "expected": 123.0,
                        "actual": 120.0,
                        "note": "mismatch",
                    }
                ]
            )
            write_local_regression_summary(
                dataset_profile_key="reduced",
                dataset_source="direct",
                data_dir="data/tw_stock_data_vip_reduced",
                csv_path="outputs/validate_consistency/consistency_full_scan.csv",
                xlsx_path="outputs/validate_consistency/consistency_issues.xlsx",
                elapsed_time=12.345,
                selected_tickers=["1101", "1102", "1103"],
                df_results=df_results,
                df_failed=df_failed,
                output_dir="outputs/validate_consistency",
                real_data_coverage_ok=True,
                peak_traced_memory_mb=12.345,
            )
        finally:
            if prev_run_dir is None:
                os.environ.pop(LOCAL_REGRESSION_RUN_DIR_ENV, None)
            else:
                os.environ[LOCAL_REGRESSION_RUN_DIR_ENV] = prev_run_dir

        validate_summary_path = run_dir / "validate_consistency_summary.json"
        add_check(results, "output_contract", case_id, "validate_summary_exists", True, validate_summary_path.exists())
        validate_payload = json.loads(validate_summary_path.read_text(encoding="utf-8"))
        add_check(results, "output_contract", case_id, "validate_summary_required_keys_present", [], sorted(REQUIRED_VALIDATE_SUMMARY_KEYS - set(validate_payload.keys())))
        add_check(results, "output_contract", case_id, "validate_summary_status", "FAIL", validate_payload.get("status"))
        add_check(results, "output_contract", case_id, "validate_summary_total_checks", 3, validate_payload.get("total_checks"))
        add_check(results, "output_contract", case_id, "validate_summary_pass_count", 1, validate_payload.get("pass_count"))
        add_check(results, "output_contract", case_id, "validate_summary_skip_count", 1, validate_payload.get("skip_count"))
        add_check(results, "output_contract", case_id, "validate_summary_fail_count", 1, validate_payload.get("fail_count"))
        add_check(results, "output_contract", case_id, "validate_summary_real_ticker_count", 3, validate_payload.get("real_ticker_count"))

        # XLSX contract: issue report sheet names and headers
        df_failed_summary = pd.DataFrame([{"ticker": "1102", "failed_checks": 1}])
        df_failed_module = pd.DataFrame([{"module": "portfolio", "failed_checks": 1}])
        xlsx_path = write_issue_excel_report(
            df_failed,
            df_failed_summary,
            df_failed_module,
            "20260401_000000",
            output_dir=str(temp_path),
            normalize_ticker=lambda value: str(value).zfill(4),
        )
        add_check(results, "output_contract", case_id, "issue_report_xlsx_exists", True, bool(xlsx_path) and Path(xlsx_path).exists())
        workbook = load_workbook(xlsx_path)
        expected_sheets = ["failed_only", "failed_tickers", "failed_modules"]
        add_check(results, "output_contract", case_id, "issue_report_sheet_names", expected_sheets, workbook.sheetnames)
        failed_only_headers = [cell.value for cell in workbook["failed_only"][1]]
        add_check(results, "output_contract", case_id, "issue_report_failed_only_headers", ["ticker", "module", "metric", "passed", "expected", "actual", "note"], failed_only_headers)
        failed_tickers_headers = [cell.value for cell in workbook["failed_tickers"][1]]
        add_check(results, "output_contract", case_id, "issue_report_failed_tickers_headers", ["ticker", "failed_checks"], failed_tickers_headers)
        failed_only_ticker = workbook["failed_only"]["A2"].value
        add_check(results, "output_contract", case_id, "issue_report_ticker_normalized_text", "1102", str(failed_only_ticker))

        # CSV / JSON contract: optimizer profiling outputs
        profiler = OptimizerProfileRecorder(str(temp_path), "20260401_000000", enabled=True, console_print=False)
        profiler.init_output_files()
        profiler.append_row(
            {
                "trial_number": 0,
                "objective_wall_sec": 1.2,
                "prep_wall_sec": 0.3,
                "portfolio_wall_sec": 0.4,
                "score_calc_sec": 0.05,
                "trade_count": 2,
                "trial_value": 0.91,
            }
        )
        with redirect_stdout(io.StringIO()):
            profiler.print_summary()

        profile_csv_path = Path(profiler.csv_path)
        profile_summary_path = Path(profiler.summary_path)
        add_check(results, "output_contract", case_id, "optimizer_profile_csv_exists", True, profile_csv_path.exists())
        add_check(results, "output_contract", case_id, "optimizer_profile_summary_exists", True, profile_summary_path.exists())
        fieldnames, profile_rows = _read_csv_rows(profile_csv_path)
        add_check(results, "output_contract", case_id, "optimizer_profile_csv_header", PROFILE_FIELDS, fieldnames)
        add_check(results, "output_contract", case_id, "optimizer_profile_csv_row_count", 1, len(profile_rows))
        profile_summary = json.loads(profile_summary_path.read_text(encoding="utf-8"))
        add_check(results, "output_contract", case_id, "optimizer_profile_summary_trial_count", 1, profile_summary.get("trial_count"))
        add_check(results, "output_contract", case_id, "optimizer_profile_summary_avg_keys_present", [], sorted(REQUIRED_PROFILE_SUMMARY_AVG_KEYS - set(profile_summary.get("avg", {}).keys())))
        add_check(results, "output_contract", case_id, "optimizer_profile_summary_avg_objective_wall_sec", 1.2, profile_summary.get("avg", {}).get("objective_wall_sec"))

    summary["json_contract_keys"] = sorted(REQUIRED_VALIDATE_SUMMARY_KEYS)
    summary["profile_fields"] = len(PROFILE_FIELDS)
    return results, summary


def validate_local_regression_summary_contract_case(_base_params):
    case_id = "LOCAL_REGRESSION_SUMMARY_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    with tempfile.TemporaryDirectory(prefix="local_regression_summary_contract_") as temp_dir:
        temp_path = Path(temp_dir)
        run_dir = temp_path / "run_dir"
        run_dir.mkdir(parents=True, exist_ok=True)

        preflight_payload = {
            "status": "FAIL",
            "python_executable": "python",
            "duration_sec": 0.321,
            "failed_packages": ["coverage", "optuna"],
            "runtime_error": "",
        }
        preflight_text = run_all_module._safe_format_preflight_summary(preflight_payload)
        add_check(results, "output_contract", case_id, "preflight_summary_contract_lines", True, "status" in preflight_text and "failed_packages : coverage, optuna" in preflight_text)

        dataset_pass = run_all_module._write_dataset_prepare_summary(run_dir, {
            "status": "PASS",
            "duration_sec": 1.234,
            "dataset_dir": "data/tw_stock_data_vip_reduced",
            "source": "existing",
            "csv_count": 24,
            "csv_total_bytes": 2400,
            "csv_members_sha256": "a" * 64,
            "csv_content_sha256": "b" * 64,
            "fingerprint_algorithm": "sha256",
            "reused_existing": True,
            "extracted_files": 0,
        })
        dataset_pass_json = json.loads((run_dir / "dataset_prepare_summary.json").read_text(encoding="utf-8"))
        dataset_pass_text = (run_dir / "dataset_prepare_summary.txt").read_text(encoding="utf-8")
        add_check(results, "output_contract", case_id, "dataset_prepare_pass_required_keys", [], sorted(REQUIRED_DATASET_PREP_SUMMARY_KEYS - set(dataset_pass_json.keys())))
        add_check(results, "output_contract", case_id, "dataset_prepare_pass_text_contract", True, "dataset_dir : data/tw_stock_data_vip_reduced" in dataset_pass_text and "csv_count   : 24" in dataset_pass_text and "members_sha : " in dataset_pass_text and "content_sha : " in dataset_pass_text)

        dataset_fail = run_all_module._write_dataset_prepare_summary(run_dir, {
            "status": "FAIL",
            "duration_sec": 0.456,
            "error_type": "RuntimeError",
            "error_message": "zip missing",
        })
        dataset_fail_text = (run_dir / "dataset_prepare_summary.txt").read_text(encoding="utf-8")
        add_check(results, "output_contract", case_id, "dataset_prepare_fail_status", "FAIL", dataset_fail.get("status"))
        add_check(results, "output_contract", case_id, "dataset_prepare_fail_text_contract", True, "error_type  : RuntimeError" in dataset_fail_text and "error_msg   : zip missing" in dataset_fail_text)

        chain_payload = {
            "status": "PASS",
            "dataset": "reduced",
            "dataset_info": {"csv_count": 24},
            "ticker_count": 24,
            "csv_count": 24,
            "duplicate_issue_count": 0,
            "skipped_ticker_count": 1,
            "detail_count": 2,
            "rows": [{"ticker": "2330"}],
            "highlights": {"blocked_by_counts": {"cash": 1}},
            "portfolio_snapshot": {"trade_rows": 2},
            "scanner_snapshot": {"candidate_count": 3},
            "rerun_consistency": {"enabled": True, "run_count": 2, "all_match": True, "runs": []},
            "failures": [],
            "peak_traced_memory_mb": 45.678,
        }
        write_json(run_dir / "chain_summary.json", chain_payload)
        write_csv(run_dir / "chain_summary.csv", [{field: "" for field in CHAIN_SUMMARY_CSV_FIELDS}], fieldnames=CHAIN_SUMMARY_CSV_FIELDS)
        chain_json = json.loads((run_dir / "chain_summary.json").read_text(encoding="utf-8"))
        chain_csv_fields, _ = _read_csv_rows(run_dir / "chain_summary.csv")
        add_check(results, "output_contract", case_id, "chain_summary_required_keys", [], sorted(REQUIRED_CHAIN_SUMMARY_KEYS - set(chain_json.keys())))
        add_check(results, "output_contract", case_id, "chain_summary_csv_header", CHAIN_SUMMARY_CSV_FIELDS, chain_csv_fields)

        ml_smoke_payload = {
            "status": "PASS",
            "dataset": "reduced",
            "dataset_info": {"csv_count": 24},
            "db_path": "outputs/ml_optimizer/demo.db",
            "db_trial_count": 1,
            "qualified_trial_count": 1,
            "best_trial_value": 1.23,
            "best_params_path": "models/best_params.json",
            "best_params_required": True,
            "best_params_keys": ["high_len", "atr_len"],
            "optimizer_profile_summary_path": "outputs/ml_optimizer/profile.json",
            "optimizer_profile_trial_count": 1,
            "optimizer_profile_avg_objective_wall_sec": 4.5,
            "optimizer_repro": {"enabled": True, "all_match": True},
            "failures": [],
            "peak_traced_memory_mb": 23.456,
        }
        write_json(run_dir / "ml_smoke_summary.json", ml_smoke_payload)
        ml_smoke_json = json.loads((run_dir / "ml_smoke_summary.json").read_text(encoding="utf-8"))
        add_check(results, "output_contract", case_id, "ml_smoke_summary_required_keys", [], sorted(REQUIRED_ML_SMOKE_SUMMARY_KEYS - set(ml_smoke_json.keys())))

        meta_payload = {
            "status": "PASS",
            "failures": [],
            "fail_count": 0,
            "coverage": {
                "ok": True,
                "status": "DONE",
                "totals": {"percent_covered": 50.0, "line_percent_covered": 55.0, "branch_percent_covered": 45.0, "line_min_percent": 50.0, "branch_min_percent": 45.0},
                "line_percent_covered": 55.0,
                "branch_percent_covered": 45.0,
                "line_min_percent": 50.0,
                "branch_min_percent": 45.0,
                "missing_targets": [],
                "zero_covered_targets": [],
            },
            "checklist": {"ok": True, "status": "DONE", "partial_ids": [], "todo_ids": [], "done_ids": ["B11"], "unfinished_d_ids": []},
            "formal_entry": {"ok": True, "project_settings_steps": [], "run_all_steps": [], "preflight_steps": []},
            "performance": {"ok": True, "skipped": False, "step_durations": {}},
            "results": [],
        }
        write_json(run_dir / "meta_quality_summary.json", meta_payload)
        meta_json = json.loads((run_dir / "meta_quality_summary.json").read_text(encoding="utf-8"))
        add_check(results, "output_contract", case_id, "meta_quality_summary_required_keys", [], sorted(REQUIRED_META_QUALITY_SUMMARY_KEYS - set(meta_json.keys())))
        add_check(results, "output_contract", case_id, "meta_quality_summary_coverage_threshold_keys", [], sorted([key for key in ["status", "line_percent_covered", "branch_percent_covered", "line_min_percent", "branch_min_percent", "missing_targets", "zero_covered_targets"] if key not in meta_json.get("coverage", {})]))
        add_check(results, "output_contract", case_id, "meta_quality_summary_checklist_status_key", True, meta_json.get("checklist", {}).get("status") == "DONE")

        selected_steps = ["quick_gate", "consistency", "chain_checks", "ml_smoke", "meta_quality"]
        completed_steps = ["quick_gate", "consistency"]
        failed_steps = ["ml_smoke"]
        not_run = run_all_module._compute_not_run_step_names(
            selected_step_names=selected_steps,
            failed_step_names=failed_steps,
            completed_script_names=completed_steps,
            include_dataset=True,
        )
        master_payload = {
            "overall_status": "FAIL",
            "dataset": "reduced",
            "dataset_info": {"csv_count": 24},
            "timestamp": "2026-04-02T00:00:00+08:00",
            "git_commit": "abc123",
            "scripts": [{"name": "quick_gate", "status": "PASS"}],
            "selected_steps": selected_steps,
            "failures": 1,
            "preflight": preflight_payload,
            "dataset_prepare": dataset_pass_json,
            "payload_failures": [],
            "not_run_step_names": not_run,
            "bundle_mode": "debug_bundle",
            "bundle_entries": run_all_module._build_bundle_entries(run_dir, [run_dir / "master_summary.json", run_dir / "preflight_summary.json"]),
            "failed_step_names": failed_steps,
            "suggested_rerun_command": "python tools/local_regression/run_all.py --only ml_smoke",
        }
        quick_gate_payload = {
            "status": "PASS",
            "dataset": "reduced",
            "dataset_info": {"csv_count": 24},
            "step_count": 3,
            "failed_count": 0,
            "failed_steps": [],
            "steps": [
                {"name": "py_compile", "status": "PASS"},
                {"name": "compileall", "status": "PASS"},
                {"name": "bare_except_scan", "status": "PASS"},
            ],
            "duration_sec": 1.23,
            "peak_traced_memory_mb": 12.34,
        }
        write_json(run_dir / "quick_gate_summary.json", quick_gate_payload)
        quick_gate_json = json.loads((run_dir / "quick_gate_summary.json").read_text(encoding="utf-8"))
        add_check(results, "output_contract", case_id, "quick_gate_summary_required_keys", [], sorted(REQUIRED_QUICK_GATE_SUMMARY_KEYS - set(quick_gate_json.keys())))

        compile_probe_calls = []

        def _fake_compile(source, cfile=None, doraise=False, *args, **kwargs):
            compile_probe_calls.append({"source": source, "cfile": cfile, "doraise": doraise})
            return cfile or source

        with patch.object(run_quick_gate_module, "iter_python_files", return_value=[run_quick_gate_module.PROJECT_ROOT / "apps" / "test_suite.py"]), \
             patch.object(run_quick_gate_module.py_compile, "compile", side_effect=_fake_compile):
            static_results = run_quick_gate_module.run_static_checks()

        compileall_result = next(item for item in static_results if item.get("name") == "compileall")
        compile_probe_path = Path(str(compile_probe_calls[0]["cfile"])) if compile_probe_calls else None
        compile_probe_resolved = str(compile_probe_path.resolve()) if compile_probe_path is not None else ""
        project_root_resolved = str(run_quick_gate_module.PROJECT_ROOT.resolve())
        add_check(results, "output_contract", case_id, "quick_gate_compileall_temp_cfile_present", True, bool(compile_probe_calls and compile_probe_calls[0].get("cfile")))
        add_check(results, "output_contract", case_id, "quick_gate_compileall_temp_cfile_outside_project_root", True, bool(compile_probe_resolved) and not compile_probe_resolved.startswith(project_root_resolved))
        add_check(results, "output_contract", case_id, "quick_gate_compileall_result_stays_pass_with_temp_cfile", "PASS", compileall_result.get("status"))

        runtime_run_dir = temp_path / "quick_gate_runtime_failure"
        runtime_run_dir.mkdir(parents=True, exist_ok=True)
        runtime_stdout = io.StringIO()
        with patch.object(run_quick_gate_module, "resolve_run_dir", return_value=runtime_run_dir):
            with patch.object(run_quick_gate_module, "load_manifest", return_value={**local_common.MANIFEST_DEFAULTS, "dataset": "reduced"}):
                with patch.object(run_quick_gate_module, "ensure_reduced_dataset", return_value={"csv_count": 24}):
                    with patch.object(run_quick_gate_module, "run_static_checks", side_effect=RuntimeError("synthetic quick gate crash")):
                        with patch("builtins.print"):
                            with redirect_stdout(runtime_stdout):
                                runtime_rc = run_quick_gate_module.main(["tools/local_regression/run_quick_gate.py"])

        runtime_json = json.loads((runtime_run_dir / "quick_gate_summary.json").read_text(encoding="utf-8"))
        runtime_console = (runtime_run_dir / "quick_gate_console.log").read_text(encoding="utf-8")
        runtime_stdout_text = runtime_stdout.getvalue()
        add_check(results, "output_contract", case_id, "quick_gate_runtime_failure_return_code", 1, runtime_rc)
        add_check(results, "output_contract", case_id, "quick_gate_runtime_failure_status", "FAIL", runtime_json.get("status"))
        add_check(results, "output_contract", case_id, "quick_gate_runtime_failure_marks_runtime_step", ["__runtime__"], runtime_json.get("failed_steps"))
        add_check(results, "output_contract", case_id, "quick_gate_runtime_failure_includes_error_fields", True, runtime_json.get("error_type") == "RuntimeError" and "synthetic quick gate crash" in runtime_json.get("runtime_error", ""))
        add_check(results, "output_contract", case_id, "quick_gate_runtime_failure_console_has_runtime_error", True, "synthetic quick gate crash" in runtime_console)
        add_check(results, "output_contract", case_id, "quick_gate_runtime_failure_does_not_leak_runtime_payload_to_parent_stdout", "", runtime_stdout_text)

        write_json(run_dir / "preflight_summary.json", preflight_payload)
        write_json(run_dir / "master_summary.json", master_payload)
        master_json = json.loads((run_dir / "master_summary.json").read_text(encoding="utf-8"))
        add_check(results, "output_contract", case_id, "master_summary_required_keys", [], sorted(REQUIRED_MASTER_SUMMARY_KEYS - set(master_json.keys())))
        add_check(results, "output_contract", case_id, "master_summary_not_run_steps_contract", ["chain_checks", "ml_smoke", "meta_quality"], master_json.get("not_run_step_names"))
        add_check(results, "output_contract", case_id, "master_summary_bundle_entries_posix", True, all("\\" not in item for item in master_json.get("bundle_entries", [])))

    summary["dataset_prepare_pass_status"] = dataset_pass_json.get("status")
    summary["master_selected_step_count"] = len(selected_steps)
    return results, summary




def _add_preflight_early_failure_dataset_contract_checks(results, case_id, *, work_dir: Path):
    early_failure_manifest = dict(run_all_module.MANIFEST_DEFAULTS)
    early_failure_manifest["dataset"] = "reduced"
    early_failure_manifest["bundle_name"] = "preflight_failed_bundle.zip"
    early_failure_dir = work_dir / "early_failure"
    early_failure_dir.mkdir(parents=True, exist_ok=True)
    early_preflight_payload = {
        "status": "FAIL",
        "python_executable": "python",
        "python_version": "3.x",
        "requirements_path": "requirements.txt",
        "checked_packages": ["coverage"],
        "failed_packages": ["coverage"],
        "checks": [],
        "duration_sec": 0.11,
    }
    write_json(early_failure_dir / "preflight_summary.json", early_preflight_payload)
    write_text(early_failure_dir / "preflight_summary.txt", "status : FAIL\n")
    with patch.object(run_all_module, "build_bundle_zip", return_value=early_failure_dir / "preflight_failed_bundle.zip"),          patch.object(run_all_module, "archive_bundle_history", side_effect=lambda path: path),          patch.object(run_all_module, "publish_root_bundle_copy", side_effect=lambda path: path),          patch.object(run_all_module, "_apply_output_retention", return_value={"removed_count": 0, "removed_bytes": 0, "removed_entries": []}),          patch.object(run_all_module, "resolve_git_commit", return_value="deadbeef"):
        early_result = run_all_module._finalize_early_failure(
            run_dir=early_failure_dir,
            manifest=early_failure_manifest,
            selected_step_names=["quick_gate", "consistency", "chain_checks", "ml_smoke", "meta_quality"],
            major_index=1,
            major_total=6,
            bundle_mode="preflight_failed",
            failed_step_names=["preflight"],
            preflight_payload=early_preflight_payload,
            include_dataset=True,
        )
    early_master = json.loads((early_failure_dir / "master_summary.json").read_text(encoding="utf-8"))
    expected_not_run = ["dataset_prepare", "quick_gate", "consistency", "chain_checks", "ml_smoke", "meta_quality"]
    add_check(results, "output_contract", case_id, "preflight_failed_early_summary_marks_dataset_prepare_not_run", expected_not_run, early_master.get("not_run_step_names"))
    add_check(results, "output_contract", case_id, "preflight_failed_early_result_marks_dataset_prepare_not_run", expected_not_run, early_result.get("not_run_step_names"))
    add_check(results, "output_contract", case_id, "preflight_failed_early_master_summary_required_keys", [], sorted(REQUIRED_MASTER_SUMMARY_KEYS - set(early_master.keys())))
    add_check(results, "output_contract", case_id, "preflight_failed_early_master_summary_dataset_prepare_status", "NOT_RUN", early_master.get("dataset_prepare", {}).get("status"))
    add_check(results, "output_contract", case_id, "preflight_failed_early_master_summary_dataset_prepare_blocked_by", "preflight", early_master.get("dataset_prepare", {}).get("blocked_by"))
    preflight_payload_failures = early_master.get("payload_failures") or []
    add_check(results, "output_contract", case_id, "preflight_failed_early_master_summary_payload_failure_names", ["preflight"], [item.get("name") for item in preflight_payload_failures])
    preflight_failure_reasons = preflight_payload_failures[0].get("failure_reasons", []) if preflight_payload_failures else []
    add_check(results, "output_contract", case_id, "preflight_failed_early_master_summary_payload_failure_has_status", True, "reported_status=FAIL" in preflight_failure_reasons)
    add_check(results, "output_contract", case_id, "preflight_failed_early_master_summary_payload_failure_has_failed_packages", True, "failed_packages=coverage" in preflight_failure_reasons)

    dataset_failure_dir = work_dir / "dataset_failure"
    dataset_failure_dir.mkdir(parents=True, exist_ok=True)
    passing_preflight_payload = {
        "status": "PASS",
        "python_executable": "python",
        "python_version": "3.x",
        "requirements_path": "requirements.txt",
        "checked_packages": ["coverage"],
        "failed_packages": [],
        "checks": [],
        "duration_sec": 0.05,
    }
    dataset_prepare_payload = {
        "status": "FAIL",
        "duration_sec": 0.23,
        "error_type": "RuntimeError",
        "error_message": "synthetic dataset prepare failure",
    }
    write_json(dataset_failure_dir / "preflight_summary.json", passing_preflight_payload)
    write_text(dataset_failure_dir / "preflight_summary.txt", "status : PASS\n")
    write_json(dataset_failure_dir / "dataset_prepare_summary.json", dataset_prepare_payload)
    write_text(dataset_failure_dir / "dataset_prepare_summary.txt", "status : FAIL\n")
    with patch.object(run_all_module, "build_bundle_zip", return_value=dataset_failure_dir / "dataset_prepare_failed_bundle.zip"),          patch.object(run_all_module, "archive_bundle_history", side_effect=lambda path: path),          patch.object(run_all_module, "publish_root_bundle_copy", side_effect=lambda path: path),          patch.object(run_all_module, "_apply_output_retention", return_value={"removed_count": 0, "removed_bytes": 0, "removed_entries": []}),          patch.object(run_all_module, "resolve_git_commit", return_value="deadbeef"):
        dataset_result = run_all_module._finalize_early_failure(
            run_dir=dataset_failure_dir,
            manifest=early_failure_manifest,
            selected_step_names=["quick_gate", "consistency", "chain_checks", "ml_smoke", "meta_quality"],
            major_index=2,
            major_total=6,
            bundle_mode="dataset_prepare_failed",
            failed_step_names=["dataset_prepare"],
            preflight_payload=passing_preflight_payload,
            include_dataset=True,
            dataset_prepare_payload=dataset_prepare_payload,
        )
    dataset_master = json.loads((dataset_failure_dir / "master_summary.json").read_text(encoding="utf-8"))
    add_check(results, "output_contract", case_id, "dataset_prepare_failed_early_result_marks_scripts_not_run", ["quick_gate", "consistency", "chain_checks", "ml_smoke", "meta_quality"], dataset_result.get("not_run_step_names"))
    dataset_payload_failures = dataset_master.get("payload_failures") or []
    add_check(results, "output_contract", case_id, "dataset_prepare_failed_early_master_summary_payload_failure_names", ["dataset_prepare"], [item.get("name") for item in dataset_payload_failures])
    dataset_failure_reasons = dataset_payload_failures[0].get("failure_reasons", []) if dataset_payload_failures else []
    add_check(results, "output_contract", case_id, "dataset_prepare_failed_early_master_summary_payload_failure_has_status", True, "reported_status=FAIL" in dataset_failure_reasons)
    add_check(results, "output_contract", case_id, "dataset_prepare_failed_early_master_summary_payload_failure_has_error_message", True, "error_message=synthetic dataset prepare failure" in dataset_failure_reasons)
    add_check(results, "output_contract", case_id, "dataset_prepare_failed_early_master_summary_payload_failure_not_marked_unreadable", False, "summary_unreadable" in dataset_failure_reasons)


def validate_run_all_preflight_early_failure_dataset_contract_case(_base_params):
    case_id = "RUN_ALL_PREFLIGHT_EARLY_FAILURE_DATASET_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    with tempfile.TemporaryDirectory(prefix="run_all_preflight_fail_contract_") as temp_dir:
        work_dir = Path(temp_dir)
        _add_preflight_early_failure_dataset_contract_checks(results, case_id, work_dir=work_dir)

    summary["checked_contract"] = "preflight_failed_dataset_prepare_not_run"
    return results, summary


def validate_run_all_manifest_failure_master_summary_contract_case(_base_params):
    case_id = "RUN_ALL_MANIFEST_FAILURE_MASTER_SUMMARY_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    with tempfile.TemporaryDirectory(prefix="run_all_manifest_fail_contract_") as temp_dir:
        run_dir = Path(temp_dir) / "manifest_failure"
        run_dir.mkdir(parents=True, exist_ok=True)
        with patch.object(run_all_module, "build_bundle_zip", return_value=run_dir / "manifest_failed_bundle.zip"),              patch.object(run_all_module, "archive_bundle_history", side_effect=lambda path: path),              patch.object(run_all_module, "publish_root_bundle_copy", side_effect=lambda path: path),              patch.object(run_all_module, "_apply_output_retention", return_value={"removed_count": 0, "removed_bytes": 0, "removed_entries": []}),              patch.object(run_all_module, "resolve_git_commit", return_value="deadbeef"):
            run_all_module._write_manifest_failure_bundle(
                run_dir=run_dir,
                selected_step_names=["quick_gate", "consistency", "chain_checks", "ml_smoke", "meta_quality"],
                include_dataset=True,
                manifest_error=ValueError("synthetic manifest failure"),
            )

        master_json = json.loads((run_dir / "master_summary.json").read_text(encoding="utf-8"))
        expected_not_run = ["preflight", "dataset_prepare", "quick_gate", "consistency", "chain_checks", "ml_smoke", "meta_quality"]
        add_check(results, "output_contract", case_id, "manifest_failed_master_summary_required_keys", [], sorted(REQUIRED_MASTER_SUMMARY_KEYS - set(master_json.keys())))
        add_check(results, "output_contract", case_id, "manifest_failed_master_summary_not_run_steps", expected_not_run, master_json.get("not_run_step_names"))
        add_check(results, "output_contract", case_id, "manifest_failed_master_summary_preflight_status", "NOT_RUN", master_json.get("preflight", {}).get("status"))
        add_check(results, "output_contract", case_id, "manifest_failed_master_summary_preflight_blocked_by", "manifest", master_json.get("preflight", {}).get("blocked_by"))
        add_check(results, "output_contract", case_id, "manifest_failed_master_summary_dataset_prepare_status", "NOT_RUN", master_json.get("dataset_prepare", {}).get("status"))
        add_check(results, "output_contract", case_id, "manifest_failed_master_summary_dataset_prepare_blocked_by", "manifest", master_json.get("dataset_prepare", {}).get("blocked_by"))
        add_check(results, "output_contract", case_id, "manifest_failed_master_summary_payload_failures_empty", [], master_json.get("payload_failures"))

    summary["checked_contract"] = "manifest_failed_master_summary_schema"
    return results, summary

def validate_artifact_lifecycle_contract_case(_base_params):
    case_id = "ARTIFACT_LIFECYCLE_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    with tempfile.TemporaryDirectory(prefix="artifact_lifecycle_") as temp_dir:
        temp_path = Path(temp_dir)
        project_root = temp_path / "project_root"
        run_dir = temp_path / "run_dir"
        output_root = project_root / "outputs" / "local_regression"
        run_dir.mkdir(parents=True, exist_ok=True)
        output_root.mkdir(parents=True, exist_ok=True)
        (project_root / "outputs" / "summary_tools").mkdir(parents=True, exist_ok=True)
        probe_master = run_dir / "master_summary.json"
        probe_console = run_dir / "console_tail.txt"
        probe_preflight = run_dir / "preflight_summary.json"
        probe_preflight_txt = run_dir / "preflight_summary.txt"
        probe_dataset = run_dir / "dataset_prepare_summary.json"
        probe_dataset_txt = run_dir / "dataset_prepare_summary.txt"
        probe_quick = run_dir / "quick_gate_summary.json"
        probe_validate = run_dir / "validate_consistency_summary.json"
        probe_chain_json = run_dir / "chain_summary.json"
        probe_chain_csv = run_dir / "chain_summary.csv"
        probe_ml = run_dir / "ml_smoke_summary.json"
        nested_fail = run_dir / "nested" / "debug.log"
        nested_fail.parent.mkdir(parents=True, exist_ok=True)

        for path, content in [
            (probe_master, '{"status":"PASS"}\n'),
            (probe_console, 'tail\n'),
            (probe_preflight, '{"status":"PASS"}\n'),
            (probe_preflight_txt, 'preflight ok\n'),
            (probe_dataset, '{"status":"PASS"}\n'),
            (probe_dataset_txt, 'dataset ok\n'),
            (probe_quick, '{"status":"PASS"}\n'),
            (probe_validate, '{"status":"PASS"}\n'),
            (probe_chain_json, '{"status":"PASS"}\n'),
            (probe_chain_csv, 'ticker,status\n1101,PASS\n'),
            (probe_ml, '{"status":"PASS"}\n'),
            (nested_fail, 'debug\n'),
        ]:
            path.write_text(content, encoding="utf-8")

        old_project_root = local_common.PROJECT_ROOT
        old_output_root = local_common.OUTPUT_ROOT
        local_common.PROJECT_ROOT = project_root
        local_common.OUTPUT_ROOT = output_root
        try:
            preferred_pass_paths = local_common.select_bundle_paths(run_dir, overall_ok=True)
            preferred_pass_names = [path.name for path in preferred_pass_paths]
            add_check(
                results,
                "artifact_contract",
                case_id,
                "select_bundle_paths_pass_minimum_set",
                [
                    "master_summary.json",
                    "preflight_summary.json",
                    "preflight_summary.txt",
                    "dataset_prepare_summary.json",
                    "dataset_prepare_summary.txt",
                    "quick_gate_summary.json",
                    "validate_consistency_summary.json",
                    "chain_summary.json",
                    "chain_summary.csv",
                    "ml_smoke_summary.json",
                    "console_tail.txt",
                ],
                preferred_pass_names,
            )
            preferred_fail_paths = local_common.select_bundle_paths(run_dir, overall_ok=False)
            preferred_fail_rel = sorted(str(path.relative_to(run_dir)).replace("\\", "/") for path in preferred_fail_paths)
            add_check(results, "artifact_contract", case_id, "select_bundle_paths_fail_includes_nested_debug_files", True, "nested/debug.log" in preferred_fail_rel)

            manifest_payload = local_common.build_artifacts_manifest(run_dir)
            add_check(results, "artifact_contract", case_id, "artifacts_manifest_counts_all_files", 12, manifest_payload.get("artifact_count"))
            manifest_paths = [item.get("relative_path") for item in manifest_payload.get("artifacts", [])]
            add_check(results, "artifact_contract", case_id, "artifacts_manifest_uses_posix_relative_paths", True, all(isinstance(item, str) and "\\" not in item and not item.startswith("/") for item in manifest_paths))
            add_check(results, "artifact_contract", case_id, "artifacts_manifest_includes_nested_debug_file", True, "nested/debug.log" in manifest_paths)

            bundle_path = local_common.build_bundle_zip(run_dir, "to_chatgpt_bundle.zip", include_paths=[probe_master, probe_console])
            add_check(results, "artifact_contract", case_id, "bundle_zip_exists", True, bundle_path.exists())
            with zipfile.ZipFile(bundle_path, "r") as zf:
                zip_members = sorted(zf.namelist())
            add_check(results, "artifact_contract", case_id, "bundle_zip_member_list", ["console_tail.txt", "master_summary.json"], zip_members)

            archived_bundle = local_common.archive_bundle_history(bundle_path)
            add_check(results, "artifact_contract", case_id, "archive_bundle_exists", True, archived_bundle.exists())
            add_check(results, "artifact_contract", case_id, "archive_bundle_removed_from_run_dir", False, bundle_path.exists())
            add_check(results, "artifact_contract", case_id, "archive_bundle_stays_under_output_root", str(output_root.resolve()), str(archived_bundle.parent.resolve()))

            stale_root_bundle = project_root / "to_chatgpt_bundle_old.zip"
            stale_root_bundle.write_text("stale\n", encoding="utf-8")
            root_copy_1 = local_common.publish_root_bundle_copy(archived_bundle)
            add_check(results, "artifact_contract", case_id, "root_bundle_copy_exists", True, root_copy_1.exists())
            root_bundles_after_first = sorted(path.name for path in project_root.glob("to_chatgpt_bundle*.zip"))
            add_check(results, "artifact_contract", case_id, "root_bundle_old_copy_removed_on_publish", [root_copy_1.name], root_bundles_after_first)

            probe_console.write_text("tail updated\n", encoding="utf-8")
            rebundle_path = local_common.build_bundle_zip(run_dir, "to_chatgpt_bundle.zip", include_paths=[probe_master, probe_console])
            with zipfile.ZipFile(rebundle_path, "r") as zf:
                rebundle_members = sorted(zf.namelist())
                rebundle_console = zf.read("console_tail.txt").decode("utf-8")
            rebundle_console_normalized = rebundle_console.replace("\r\n", "\n")
            add_check(results, "artifact_contract", case_id, "rerun_bundle_overwrites_run_dir_zip", ["console_tail.txt", "master_summary.json"], rebundle_members)
            add_check(results, "artifact_contract", case_id, "rerun_bundle_contains_latest_console_tail", "tail updated\n", rebundle_console_normalized)

            archived_bundle_2 = local_common.archive_bundle_history(rebundle_path)
            root_copy_2 = local_common.publish_root_bundle_copy(archived_bundle_2)
            root_bundles_after_second = sorted(path.name for path in project_root.glob("to_chatgpt_bundle*.zip"))
            add_check(results, "artifact_contract", case_id, "root_bundle_only_latest_copy_kept", [root_copy_2.name], root_bundles_after_second)
            with zipfile.ZipFile(root_copy_2, "r") as zf:
                root_console = zf.read("console_tail.txt").decode("utf-8")
            root_console_normalized = root_console.replace("\r\n", "\n")
            add_check(results, "artifact_contract", case_id, "root_bundle_copy_matches_latest_archive_contents", "tail updated\n", root_console_normalized)

            old_archive = output_root / "to_chatgpt_bundle_20260101_000000_old.zip"
            old_archive.write_text("old\n", encoding="utf-8")
            retention_now = local_common.taipei_now()
            old_ts = retention_now.timestamp() - 40 * 86400
            os.utime(old_archive, (old_ts, old_ts))
            new_ts = retention_now.timestamp()
            os.utime(archived_bundle, (new_ts, new_ts))
            os.utime(archived_bundle_2, (new_ts, new_ts))
            retention = apply_retention_rules(
                [
                    RetentionRule(
                        name="bundle_history",
                        target_dir=output_root,
                        patterns=["to_chatgpt_bundle*.zip"],
                        keep_last_n=2,
                        max_age_days=30,
                    )
                ],
                now=retention_now,
            )
            add_check(results, "artifact_contract", case_id, "retention_removed_old_archive_count", 1, retention.get("removed_count"))
            add_check(results, "artifact_contract", case_id, "retention_removed_old_archive_missing", False, old_archive.exists())
            remaining_archives = sorted(path.name for path in output_root.glob("to_chatgpt_bundle*.zip"))
            add_check(results, "artifact_contract", case_id, "retention_keeps_recent_archives", sorted([archived_bundle.name, archived_bundle_2.name]), remaining_archives)
        finally:
            local_common.PROJECT_ROOT = old_project_root
            local_common.OUTPUT_ROOT = old_output_root

    summary["archive_retention_checked"] = True
    summary["root_copy_overwrite_checked"] = True
    summary["pass_fail_bundle_selection_checked"] = True
    summary["artifacts_manifest_checked"] = True
    return results, summary


def validate_dataset_fingerprint_contract_case(_base_params):
    case_id = "DATASET_FINGERPRINT_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    dataset_info = local_common.ensure_reduced_dataset()
    add_check(results, "output_contract", case_id, "dataset_fingerprint_has_required_keys", [], sorted([
        key for key in ["csv_count", "csv_total_bytes", "csv_members_sha256", "csv_content_sha256", "fingerprint_algorithm"]
        if key not in dataset_info
    ]))
    add_check(results, "output_contract", case_id, "dataset_fingerprint_algorithm_is_sha256", "sha256", dataset_info.get("fingerprint_algorithm"))
    add_check(results, "output_contract", case_id, "dataset_member_hash_length_is_64", 64, len(str(dataset_info.get("csv_members_sha256", ""))))
    add_check(results, "output_contract", case_id, "dataset_content_hash_length_is_64", 64, len(str(dataset_info.get("csv_content_sha256", ""))))
    add_check(results, "output_contract", case_id, "dataset_total_bytes_positive", True, int(dataset_info.get("csv_total_bytes", 0)) > 0)

    with tempfile.TemporaryDirectory(prefix="dataset_fingerprint_contract_") as temp_dir:
        run_dir = Path(temp_dir) / "run_dir"
        run_dir.mkdir(parents=True, exist_ok=True)
        written = run_all_module._write_dataset_prepare_summary(run_dir, {
            "status": "PASS",
            "duration_sec": 0.123,
            **dataset_info,
        })
        summary_json = json.loads((run_dir / "dataset_prepare_summary.json").read_text(encoding="utf-8"))
        summary_txt = (run_dir / "dataset_prepare_summary.txt").read_text(encoding="utf-8")

    add_check(results, "output_contract", case_id, "dataset_prepare_summary_preserves_member_hash", dataset_info.get("csv_members_sha256"), summary_json.get("csv_members_sha256"))
    add_check(results, "output_contract", case_id, "dataset_prepare_summary_preserves_content_hash", dataset_info.get("csv_content_sha256"), summary_json.get("csv_content_sha256"))
    add_check(results, "output_contract", case_id, "dataset_prepare_summary_preserves_total_bytes", dataset_info.get("csv_total_bytes"), summary_json.get("csv_total_bytes"))
    add_check(results, "output_contract", case_id, "dataset_prepare_summary_text_displays_hashes", True, "members_sha : " in summary_txt and "content_sha : " in summary_txt)

    summary["csv_count"] = int(dataset_info.get("csv_count", 0))
    summary["csv_total_bytes"] = int(dataset_info.get("csv_total_bytes", 0))
    summary["member_hash_prefix"] = str(dataset_info.get("csv_members_sha256", ""))[:12]
    return results, summary



def validate_run_all_dataset_prepare_pass_main_contract_case(_base_params):
    case_id = "RUN_ALL_DATASET_PREPARE_PASS_MAIN_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    manifest = dict(local_common.MANIFEST_DEFAULTS)
    manifest["bundle_name"] = "to_chatgpt_bundle.zip"
    fixed_now = local_common.taipei_now()

    with tempfile.TemporaryDirectory(prefix="run_all_dataset_prepare_pass_") as temp_dir:
        run_dir = Path(temp_dir) / "run_dir"
        run_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = run_dir / manifest["bundle_name"]
        bundle_path.write_bytes(b"synthetic-bundle")

        dataset_info = {
            "dataset_dir": str(local_common.REDUCED_DATASET_DIR),
            "source": "repo_data_dir",
            "csv_count": 24,
            "reused_existing": True,
            "csv_total_bytes": 4096,
            "csv_members_sha256": "a" * 64,
            "csv_content_sha256": "b" * 64,
            "fingerprint_algorithm": "sha256",
            "extracted_files": [],
        }

        def _fake_run_preflight(run_dir_arg, *, selected_step_names):
            payload = {
                "status": "PASS",
                "python_executable": sys.executable,
                "duration_sec": 0.01,
                "failed_packages": [],
            }
            write_json(run_dir_arg / "preflight_summary.json", payload)
            write_text(run_dir_arg / "preflight_summary.txt", "status : PASS\n")
            return payload

        summary_name_by_script = {script: summary_name for _name, script, summary_name in run_all_module.SCRIPT_ORDER}

        def _fake_run_script(*, name, relative_script, timeout_sec, env, log_path, progress_callback, major_index, major_total):
            payload = {
                "status": "PASS",
                "failures": [],
                "failed_steps": [],
                "failed_count": 0,
                "fail_count": 0,
            }
            summary_name = summary_name_by_script[relative_script]
            write_json(run_dir / summary_name, payload)
            log_path.write_text(f"{name}: PASS\n", encoding="utf-8")
            return {
                "returncode": 0,
                "duration_sec": 0.02,
                "timed_out": False,
                "error_type": "",
                "error_message": "",
                "stdout": "",
                "stderr": "",
            }

        with patch.object(run_all_module, "create_staging_run_dir", return_value=run_dir), \
             patch.object(run_all_module, "load_manifest", return_value=manifest), \
             patch.object(run_all_module, "_run_preflight", side_effect=_fake_run_preflight), \
             patch.object(run_all_module, "ensure_reduced_dataset", return_value=dataset_info), \
             patch.object(run_all_module, "_run_script", side_effect=_fake_run_script), \
             patch.object(run_all_module, "build_bundle_zip", return_value=bundle_path), \
             patch.object(run_all_module, "archive_bundle_history", side_effect=lambda path: path), \
             patch.object(run_all_module, "publish_root_bundle_copy", side_effect=lambda path: path), \
             patch.object(run_all_module, "_apply_output_retention", return_value={"removed_count": 0, "removed_bytes": 0, "removed_entries": []}), \
             patch.object(run_all_module, "cleanup_staging_dir", return_value=None), \
             patch.object(run_all_module, "resolve_git_commit", return_value="deadbeef"), \
             patch.object(run_all_module, "taipei_now", return_value=fixed_now), \
             patch.object(run_all_module, "gather_recent_console_tail", return_value="tail text"):
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                rc = run_all_module.main(["run_all.py"])

        master_summary = json.loads((run_dir / "master_summary.json").read_text(encoding="utf-8"))
        dataset_prepare_summary = json.loads((run_dir / "dataset_prepare_summary.json").read_text(encoding="utf-8"))

    add_check(results, "output_contract", case_id, "run_all_main_dataset_prepare_pass_return_code", 0, rc)
    add_check(results, "output_contract", case_id, "run_all_main_dataset_prepare_pass_overall_status", "PASS", master_summary.get("overall_status"))
    add_check(results, "output_contract", case_id, "run_all_main_dataset_prepare_payload_status", "PASS", master_summary.get("dataset_prepare", {}).get("status"))
    add_check(results, "output_contract", case_id, "run_all_main_dataset_prepare_not_run_steps_empty", [], master_summary.get("not_run_step_names", []))
    add_check(results, "output_contract", case_id, "run_all_main_dataset_prepare_selected_steps", list(run_all_module.STEP_NAMES), master_summary.get("selected_steps"))
    for key in local_common.DATASET_INFO_KEYS:
        if key in dataset_info:
            add_check(results, "output_contract", case_id, f"run_all_main_dataset_prepare_preserves_{key}", dataset_info[key], dataset_prepare_summary.get(key))
            add_check(results, "output_contract", case_id, f"run_all_main_master_summary_dataset_info_preserves_{key}", dataset_info[key], master_summary.get("dataset_info", {}).get(key))

    summary["dataset_keys_checked"] = len([key for key in local_common.DATASET_INFO_KEYS if key in dataset_info])
    summary["selected_step_count"] = len(master_summary.get("selected_steps", []))
    return results, summary


def validate_atomic_write_contract_case(_base_params):
    case_id = "ATOMIC_WRITE_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    with tempfile.TemporaryDirectory(prefix="atomic_write_contract_") as temp_dir:
        run_dir = Path(temp_dir)
        json_path = run_dir / "atomic_summary.json"
        text_path = run_dir / "atomic_summary.txt"

        local_common.write_json(json_path, {"status": "original", "count": 1})
        local_common.write_text(text_path, "original-text\n")
        original_json = json_path.read_text(encoding="utf-8")
        original_text = text_path.read_text(encoding="utf-8")

        with patch.object(local_common.os, "replace", side_effect=OSError("simulated atomic replace failure")):
            try:
                local_common.write_json(json_path, {"status": "new", "count": 2})
            except OSError:
                pass
            else:
                raise AssertionError("write_json should raise when os.replace fails")

            try:
                local_common.write_text(text_path, "new-text\n")
            except OSError:
                pass
            else:
                raise AssertionError("write_text should raise when os.replace fails")

        remaining_temp_files = sorted(path.name for path in run_dir.glob(".*.tmp"))
        add_check(results, "output_contract", case_id, "atomic_write_json_preserves_previous_contents_on_replace_failure", original_json, json_path.read_text(encoding="utf-8"))
        add_check(results, "output_contract", case_id, "atomic_write_text_preserves_previous_contents_on_replace_failure", original_text, text_path.read_text(encoding="utf-8"))
        add_check(results, "output_contract", case_id, "atomic_write_cleanup_removes_temp_files_after_failure", [], remaining_temp_files)

        local_common.write_json(json_path, {"status": "final", "count": 3})
        local_common.write_text(text_path, "final-text\n")
        final_json = json.loads(json_path.read_text(encoding="utf-8"))
        final_text = text_path.read_text(encoding="utf-8")

    add_check(results, "output_contract", case_id, "atomic_write_json_success_after_failure", "final", final_json.get("status"))
    add_check(results, "output_contract", case_id, "atomic_write_text_success_after_failure", "final-text\n", final_text)

    summary["cleanup_checked"] = True
    return results, summary


def validate_atomic_write_retry_contract_case(_base_params):
    case_id = "ATOMIC_WRITE_RETRY_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    with tempfile.TemporaryDirectory(prefix="atomic_write_retry_contract_") as temp_dir:
        run_dir = Path(temp_dir)
        json_path = run_dir / "atomic_retry_summary.json"
        local_common.write_json(json_path, {"status": "original", "count": 1})

        original_replace = local_common.os.replace
        replace_attempts = {"count": 0}

        def flaky_replace(src, dst):
            replace_attempts["count"] += 1
            if replace_attempts["count"] < 3:
                raise PermissionError("simulated transient share violation")
            return original_replace(src, dst)

        with patch.object(local_common.os, "replace", side_effect=flaky_replace),              patch.object(local_common.time, "sleep", side_effect=lambda *_args, **_kwargs: None):
            local_common.write_json(json_path, {"status": "retried", "count": 2})

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        remaining_temp_files = sorted(path.name for path in run_dir.glob(".*.tmp"))

    add_check(results, "output_contract", case_id, "atomic_write_retry_eventually_succeeds_after_transient_permission_error", "retried", payload.get("status"))
    add_check(results, "output_contract", case_id, "atomic_write_retry_attempt_count_matches_transient_failures", 3, replace_attempts["count"])
    add_check(results, "output_contract", case_id, "atomic_write_retry_cleans_temp_files_after_success", [], remaining_temp_files)

    summary["replace_attempts"] = replace_attempts["count"]
    return results, summary


def validate_meta_quality_performance_memory_contract_case(_base_params):
    case_id = "META_QUALITY_PERFORMANCE_MEMORY_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    with tempfile.TemporaryDirectory(prefix="meta_quality_performance_memory_contract_") as temp_dir:
        run_dir = Path(temp_dir) / "run_dir"
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "quick_gate_summary.json", {"status": "PASS", "duration_sec": 1.0, "peak_traced_memory_mb": 12.5})
        write_json(run_dir / "validate_consistency_summary.json", {"status": "PASS", "elapsed_time_sec": 2.0, "peak_traced_memory_mb": 32.5})
        write_json(run_dir / "chain_summary.json", {"status": "PASS", "duration_sec": 3.0, "peak_traced_memory_mb": 64.0})
        write_json(run_dir / "ml_smoke_summary.json", {"status": "PASS", "duration_sec": 4.0, "peak_traced_memory_mb": 48.0, "optimizer_profile_trial_count": 1, "optimizer_profile_avg_objective_wall_sec": 1.5})

        manifest = dict(local_common.MANIFEST_DEFAULTS)
        with patch.dict(os.environ, {LOCAL_REGRESSION_RUN_DIR_ENV: str(run_dir)}):
            perf = run_meta_quality_module._build_performance_summary(
                run_dir,
                manifest,
                current_meta_quality_duration_sec=1.25,
                current_meta_quality_peak_traced_memory_mb=40.0,
            )

        add_check(results, "output_contract", case_id, "performance_memory_contract_ok", True, perf.get("ok"))
        add_check(results, "output_contract", case_id, "performance_step_peak_memory_keys", ["chain_checks", "consistency", "ml_smoke", "quick_gate"], sorted(perf.get("step_peak_traced_memory_mb", {}).keys()))
        add_check(results, "output_contract", case_id, "performance_max_peak_memory_value", 64.0, perf.get("max_step_peak_traced_memory_mb"))
        add_check(results, "output_contract", case_id, "performance_meta_quality_peak_memory_value", 40.0, perf.get("meta_quality_peak_traced_memory_mb"))

    summary["step_peak_memory_count"] = 4
    return results, summary


def validate_portfolio_sim_prepared_tool_contract_case(base_params):
    case_id = "PORTFOLIO_SIM_PREPARED_TOOL_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    case = build_synthetic_competing_candidates_case(base_params, make_synthetic_validation_params)
    ticker = case["primary_ticker"]
    frame = case["frames"][ticker]

    with tempfile.TemporaryDirectory(prefix="portfolio_sim_prepared_tool_") as temp_dir:
        write_synthetic_csv_bundle(temp_dir, {ticker: frame})
        file_path = str(Path(temp_dir) / f"{ticker}.csv")
        legacy_stats = run_portfolio_sim_tool_check_for_dir(
            temp_dir,
            case["params"],
            max_positions=1,
            enable_rotation=False,
            start_year=case["start_year"],
            benchmark_ticker=ticker,
        )
        prepared_stats = run_portfolio_sim_tool_check(ticker, file_path, case["params"])

    metric_names = [
        "trade_count",
        "win_rate",
        "pf_ev",
        "pf_payoff",
        "final_eq",
        "annual_return_pct",
        "reserved_buy_fill_rate",
        "portfolio_buy_rows",
        "portfolio_missed_buy_rows",
        "portfolio_half_take_profit_rows",
    ]
    for metric in metric_names:
        add_check(results, "output_contract", case_id, f"portfolio_sim_prepared_{metric}", legacy_stats[metric], prepared_stats[metric])

    add_check(results, "output_contract", case_id, "portfolio_sim_prepared_module_path", legacy_stats["module_path"], prepared_stats["module_path"])
    return results, summary



def validate_scanner_prepared_tool_contract_case(base_params):
    case_id = "SCANNER_PREPARED_TOOL_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    case = build_synthetic_competing_candidates_case(base_params, make_synthetic_validation_params)
    ticker = case["primary_ticker"]
    frame = case["frames"][ticker]

    with tempfile.TemporaryDirectory(prefix="scanner_prepared_tool_") as temp_dir:
        write_synthetic_csv_bundle(temp_dir, {ticker: frame})
        file_path = str(Path(temp_dir) / f"{ticker}.csv")
        legacy_result, legacy_module_path = run_scanner_tool_check(ticker, file_path, case["params"])
        min_rows_needed = get_required_min_rows(case["params"])
        clean_df, sanitize_stats = sanitize_ohlcv_dataframe(frame.copy(), ticker, min_rows=min_rows_needed)
        prepared_df, _standalone_logs, prepared_stats = prep_stock_data_and_trades(clean_df.copy(), case["params"], return_stats=True)
        prepared_result, prepared_module_path = run_scanner_tool_check(
            ticker,
            file_path,
            case["params"],
            prepared_df=prepared_df,
            sanitize_stats=sanitize_stats,
            precomputed_stats=prepared_stats,
        )

    add_check(results, "output_contract", case_id, "scanner_prepared_module_path", legacy_module_path, prepared_module_path)
    add_check(results, "output_contract", case_id, "scanner_prepared_payload", legacy_result, prepared_result)
    return results, summary


def validate_scanner_reference_clean_df_contract_case(base_params):
    case_id = "SCANNER_REFERENCE_CLEAN_DF_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    case = build_synthetic_competing_candidates_case(base_params, make_synthetic_validation_params)
    ticker = case["primary_ticker"]
    frame = case["frames"][ticker]

    with tempfile.TemporaryDirectory(prefix="scanner_reference_clean_df_") as temp_dir:
        write_synthetic_csv_bundle(temp_dir, {ticker: frame})
        file_path = str(Path(temp_dir) / f"{ticker}.csv")
        legacy_stats = run_scanner_reference_check(ticker, file_path, case["params"])
        min_rows_needed = get_required_min_rows(case["params"])
        clean_df, _sanitize_stats = sanitize_ohlcv_dataframe(frame.copy(), ticker, min_rows=min_rows_needed)
        clean_df_stats = run_scanner_reference_check_on_clean_df(ticker, clean_df, case["params"])

    metric_names = [
        "trade_count",
        "win_rate",
        "expected_value",
        "asset_growth",
        "max_drawdown",
        "missed_buys",
        "missed_sells",
        "is_candidate",
        "is_setup_today",
        "current_position",
    ]
    for metric in metric_names:
        add_check(results, "output_contract", case_id, f"scanner_reference_clean_df_{metric}", legacy_stats[metric], clean_df_stats[metric])

    add_check(results, "output_contract", case_id, "scanner_reference_clean_df_extended_candidate", legacy_stats.get("extended_candidate_today"), clean_df_stats.get("extended_candidate_today"))
    return results, summary


def validate_debug_trade_log_prepared_tool_contract_case(base_params):
    case_id = "DEBUG_TRADE_LOG_PREPARED_TOOL_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    case = build_synthetic_competing_candidates_case(base_params, make_synthetic_validation_params)
    ticker = case["primary_ticker"]
    frame = case["frames"][ticker]
    min_rows_needed = get_required_min_rows(case["params"])
    clean_df, _sanitize_stats = sanitize_ohlcv_dataframe(frame.copy(), ticker, min_rows=min_rows_needed)
    prepared_df, _standalone_logs, _prepared_stats = prep_stock_data_and_trades(clean_df.copy(), case["params"], return_stats=True)

    legacy_df, legacy_module_path = run_debug_trade_log_check(ticker, clean_df, case["params"])
    prepared_result_df, prepared_module_path = run_debug_trade_log_check(ticker, clean_df, case["params"], prepared_df=prepared_df)

    legacy_records = _normalize_nan_records(legacy_df)
    prepared_records = _normalize_nan_records(prepared_result_df)

    add_check(results, "output_contract", case_id, "debug_prepared_module_path", legacy_module_path, prepared_module_path)
    add_check(results, "output_contract", case_id, "debug_prepared_payload", legacy_records, prepared_records)
    return results, summary



def validate_meta_quality_reuses_existing_coverage_artifacts_case(base_params):
    case_id = "META_QUALITY_REUSE_COVERAGE_ARTIFACTS"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from tools.validate import synthetic_cases as synthetic_cases_module

    manifest = local_common.load_manifest()
    overall_line_floor = float(manifest["coverage_line_min_percent"])
    overall_branch_floor = float(manifest["coverage_branch_min_percent"])
    critical_line_floor = float(manifest["coverage_critical_line_min_percent"])
    critical_branch_floor = float(manifest["coverage_critical_branch_min_percent"])
    synthetic_case_count = len(synthetic_cases_module.get_synthetic_validators())

    with tempfile.TemporaryDirectory(prefix="meta_quality_reuse_cov_") as temp_dir:
        run_dir = Path(temp_dir)
        coverage_dir = run_dir / "coverage_artifacts"
        coverage_dir.mkdir(parents=True, exist_ok=True)
        coverage_payload = {
            "totals": {
                "covered_lines": int(max(overall_line_floor, 60.0) * 2),
                "num_statements": 200,
                "covered_branches": int(max(overall_branch_floor, 60.0) * 2),
                "num_branches": 200,
                "percent_covered": float((max(overall_line_floor, 60.0) + max(overall_branch_floor, 60.0)) / 2.0),
            },
            "files": {
                rel_path: {
                    "summary": {
                        "covered_lines": int(critical_line_floor) if rel_path in run_meta_quality_module.CRITICAL_COVERAGE_TARGETS else 1,
                        "num_statements": 100 if rel_path in run_meta_quality_module.CRITICAL_COVERAGE_TARGETS else 1,
                        "percent_covered": float(critical_line_floor) if rel_path in run_meta_quality_module.CRITICAL_COVERAGE_TARGETS else 100.0,
                        "covered_branches": int(critical_branch_floor) if rel_path in run_meta_quality_module.CRITICAL_COVERAGE_TARGETS else 1,
                        "num_branches": 100 if rel_path in run_meta_quality_module.CRITICAL_COVERAGE_TARGETS else 1,
                    }
                }
                for rel_path in run_meta_quality_module.COVERAGE_TARGETS
            },
        }
        write_text(coverage_dir / "coverage_synthetic.json", json.dumps(coverage_payload, ensure_ascii=False, indent=2))
        write_json(coverage_dir / "coverage_run_info.json", {
            "source": "validate_consistency",
            "returncode": 0,
            "stdout": "cached",
            "stderr": "",
            "timed_out": False,
            "synthetic_fail_count": 0,
            "synthetic_case_count": synthetic_case_count,
            "json_generated": True,
        })

        with patch("tools.validate.synthetic_cases.run_synthetic_consistency_suite", side_effect=AssertionError("should reuse existing coverage artifacts")):
            coverage_summary = run_meta_quality_module._build_coverage_summary(run_dir, manifest)

    add_check(results, "output_contract", case_id, "meta_quality_reuse_coverage_ok", True, coverage_summary.get("ok"))
    add_check(results, "output_contract", case_id, "meta_quality_reuse_coverage_flag", True, coverage_summary.get("reused_existing"))
    add_check(results, "output_contract", case_id, "meta_quality_reuse_coverage_returncode", 0, coverage_summary.get("run_result", {}).get("returncode"))
    return results, summary
