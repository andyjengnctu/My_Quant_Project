import csv
import io
import json
import os
import tempfile
import zipfile
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from core.output_retention import RetentionRule, apply_retention_rules
from tools.local_regression import common as local_common
from tools.optimizer.profile import OptimizerProfileRecorder, PROFILE_FIELDS
from tools.validate.main import LOCAL_REGRESSION_RUN_DIR_ENV, write_local_regression_summary
from tools.validate.reporting import write_issue_excel_report

from .checks import add_check


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
}

REQUIRED_PROFILE_SUMMARY_AVG_KEYS = {
    "objective_wall_sec",
    "prep_wall_sec",
    "portfolio_wall_sec",
    "score_calc_sec",
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
        probe_a = run_dir / "master_summary.json"
        probe_b = run_dir / "console_tail.txt"
        probe_a.write_text('{\"status\":\"PASS\"}\n', encoding="utf-8")
        probe_b.write_text('tail\n', encoding="utf-8")

        old_project_root = local_common.PROJECT_ROOT
        old_output_root = local_common.OUTPUT_ROOT
        local_common.PROJECT_ROOT = project_root
        local_common.OUTPUT_ROOT = output_root
        try:
            bundle_path = local_common.build_bundle_zip(run_dir, "to_chatgpt_bundle.zip", include_paths=[probe_a, probe_b])
            add_check(results, "artifact_contract", case_id, "bundle_zip_exists", True, bundle_path.exists())
            with zipfile.ZipFile(bundle_path, 'r') as zf:
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

            rebundle_path = local_common.build_bundle_zip(run_dir, "to_chatgpt_bundle.zip", include_paths=[probe_a])
            archived_bundle_2 = local_common.archive_bundle_history(rebundle_path)
            root_copy_2 = local_common.publish_root_bundle_copy(archived_bundle_2)
            root_bundles_after_second = sorted(path.name for path in project_root.glob("to_chatgpt_bundle*.zip"))
            add_check(results, "artifact_contract", case_id, "root_bundle_only_latest_copy_kept", [root_copy_2.name], root_bundles_after_second)

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
    return results, summary
