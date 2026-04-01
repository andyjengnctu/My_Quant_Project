import io
from contextlib import redirect_stdout

import pandas as pd

from apps import test_suite as test_suite_module
from core.display_common import _strip_ansi
from tools.portfolio_sim import reporting as portfolio_reporting
from tools.validate.reporting import print_console_summary

from .checks import add_check


def _capture_output(callable_obj):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        callable_obj()
    return _strip_ansi(buffer.getvalue())


def validate_validate_console_summary_reporting_case(_base_params):
    case_id = "VALIDATE_CONSOLE_SUMMARY_REPORTING"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    df_results = pd.DataFrame(
        [
            {"ticker": "1101", "module": "stats", "metric": "win_rate", "status": "PASS", "passed": True, "expected": "58.3", "actual": "58.3", "note": ""},
            {"ticker": "1102", "module": "stats", "metric": "ev", "status": "FAIL", "passed": False, "expected": "0.8", "actual": "0.4", "note": "EV mismatch"},
            {"ticker": "SYN_CASE", "module": "synthetic", "metric": "rule", "status": "FAIL", "passed": False, "expected": "PASS", "actual": "FAIL", "note": "Synthetic fail"},
            {"ticker": "REAL_DATA_COVERAGE", "module": "system", "metric": "coverage", "status": "SKIP", "passed": False, "expected": "full", "actual": "synthetic-only", "note": "skip"},
        ]
    )
    df_failed = df_results[df_results["status"] == "FAIL"].copy()
    df_summary = pd.DataFrame(
        [
            {"ticker": "1101", "synthetic": False},
            {"ticker": "1102", "synthetic": False},
            {"ticker": "SYN_CASE", "synthetic": True},
            {"ticker": "REAL_DATA_COVERAGE", "synthetic": False},
        ]
    )

    console_text = _capture_output(
        lambda: print_console_summary(
            df_results=df_results,
            df_failed=df_failed,
            df_summary=df_summary,
            csv_path="outputs/validate/consistency_full_scan_20260402.csv",
            xlsx_path="outputs/validate/consistency_issues_20260402.xlsx",
            elapsed_time=12.34,
            real_summary_count=2,
            real_tickers=["1101", "1102"],
            normalize_ticker_text=lambda value: str(value).strip(),
            max_console_fail_preview=5,
        )
    )

    add_check(results, "reporting_schema", case_id, "console_summary_has_title", True, "一致性回歸摘要" in console_text)
    add_check(results, "reporting_schema", case_id, "console_summary_has_counts", True, "PASS 數: 1" in console_text and "SKIP 數: 1" in console_text and "FAIL 數: 2" in console_text)
    add_check(results, "reporting_schema", case_id, "console_summary_has_problem_counts", True, "有問題真實股票數: 1" in console_text and "有問題 synthetic case 數: 1" in console_text and "有問題 system 項目數: 0" in console_text)
    add_check(results, "reporting_schema", case_id, "console_summary_has_paths", True, "完整 CSV: outputs/validate/consistency_full_scan_20260402.csv" in console_text and "問題 Excel: outputs/validate/consistency_issues_20260402.xlsx" in console_text)
    add_check(results, "reporting_schema", case_id, "console_summary_has_fail_preview_sections", True, "失敗項前覽：" in console_text and "失敗真實股票前覽：" in console_text and "失敗 synthetic/system 前覽：" in console_text)
    add_check(results, "reporting_schema", case_id, "console_summary_has_fail_preview_details", True, "EV mismatch" in console_text and "Synthetic fail" in console_text)

    summary["console_summary_lines"] = len([line for line in console_text.splitlines() if line.strip()])
    return results, summary


def validate_portfolio_yearly_report_schema_case(_base_params):
    case_id = "PORTFOLIO_YEARLY_REPORT_SCHEMA"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    yearly_rows = [
        {"year": 2023, "year_return_pct": 12.345, "is_full_year": True, "start_date": "2023-01-03", "end_date": "2023-12-29"},
        {"year": 2024, "year_return_pct": -3.21, "is_full_year": False, "start_date": "2024-01-02", "end_date": "2024-06-28"},
    ]

    captured_df = None

    def _run_non_empty():
        nonlocal captured_df
        captured_df = portfolio_reporting.print_yearly_return_report(yearly_rows)

    report_text = _capture_output(_run_non_empty)
    add_check(results, "reporting_schema", case_id, "yearly_report_has_title", True, "各年度報酬率" in report_text)
    add_check(results, "reporting_schema", case_id, "yearly_report_has_percent_and_type", True, "12.35%" in report_text and "完整" in report_text and "-3.21%" in report_text and "非完整" in report_text)
    add_check(results, "reporting_schema", case_id, "yearly_report_returns_dataframe_columns", ["year", "year_return_pct", "is_full_year", "start_date", "end_date", "year_label", "year_type"], list(captured_df.columns))
    add_check(results, "reporting_schema", case_id, "yearly_report_year_label_and_type", True, captured_df.loc[0, "year_label"] == "2023" and captured_df.loc[1, "year_type"] == "非完整")

    empty_df = None

    def _run_empty():
        nonlocal empty_df
        empty_df = portfolio_reporting.print_yearly_return_report([])

    empty_text = _capture_output(_run_empty)
    add_check(results, "reporting_schema", case_id, "yearly_report_empty_warning", True, "無年度報酬率資料" in empty_text)
    add_check(results, "reporting_schema", case_id, "yearly_report_empty_schema", ["year", "year_return_pct", "is_full_year", "start_date", "end_date"], list(empty_df.columns))

    summary["yearly_rows"] = int(len(captured_df)) if captured_df is not None else 0
    return results, summary


def validate_test_suite_summary_reporting_case(_base_params):
    case_id = "TEST_SUITE_SUMMARY_REPORTING"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    result_payload = {
        "overall_status": "PASS",
        "failures": 0,
        "selected_steps": ["quick_gate", "consistency", "chain_checks", "ml_smoke", "meta_quality"],
        "failed_step_names": [],
        "not_run_step_names": [],
        "scripts": [
            {"name": "quick_gate", "status": "PASS", "duration_sec": 8.42, "failure_reasons": []},
            {"name": "consistency", "status": "PASS", "duration_sec": 10.23, "failure_reasons": []},
            {"name": "chain_checks", "status": "PASS", "duration_sec": 11.22, "failure_reasons": []},
            {"name": "ml_smoke", "status": "PASS", "duration_sec": 4.01, "failure_reasons": []},
            {"name": "meta_quality", "status": "PASS", "duration_sec": 1.81, "failure_reasons": []},
        ],
        "step_payloads": {
            "quick_gate": {"status": "PASS", "step_count": 88, "failed_count": 0},
            "consistency": {"status": "PASS", "total_checks": 2752, "fail_count": 0, "skip_count": 0, "real_ticker_count": 24},
            "chain_checks": {"status": "PASS", "ticker_count": 24, "highlights": {"traded_ticker_count": 7, "missed_buy_ticker_count": 2, "blocked_by_counts": {"cash": 3, "slots": 1}}, "portfolio_snapshot": {"trade_rows": 14, "reserved_buy_fill_rate": 83.33}},
            "ml_smoke": {"status": "PASS", "db_trial_count": 1},
            "meta_quality": {"status": "PASS", "fail_count": 0, "coverage": {"totals": {"percent_covered": 52.41}}, "checklist": {"todo_ids": []}},
        },
        "preflight": {"status": "PASS", "duration_sec": 0.45, "failed_packages": []},
        "bundle_mode": "minimum_set",
        "archived_bundle": "outputs/local_regression/to_chatgpt_bundle_20260402_123456_abcd1234.zip",
        "root_bundle_copy": "to_chatgpt_bundle_20260402_123456_abcd1234.zip",
        "bundle_entries": ["master_summary.json", "preflight_summary.json", "quick_gate_summary.json"],
        "retention": {"removed_count": 2, "removed_bytes": 4096},
    }

    summary_text = _capture_output(lambda: test_suite_module._print_human_summary(result_payload))
    add_check(results, "reporting_schema", case_id, "test_suite_summary_has_title_and_bundle", True, "Test Suite 結果整理" in summary_text and "bundle 模式 : minimum_set" in summary_text)
    add_check(results, "reporting_schema", case_id, "test_suite_summary_has_step_table", True, "[步驟摘要]" in summary_text and "quick gate" in summary_text and "meta quality" in summary_text)
    add_check(results, "reporting_schema", case_id, "test_suite_summary_has_highlights", True, "step_count=88" in summary_text and "total_checks=2752" in summary_text and "db_trial_count=1" in summary_text)
    add_check(results, "reporting_schema", case_id, "test_suite_summary_has_chain_and_meta_details", True, "blocked_by : cash:3, slots:1" in summary_text and "coverage_percent=52.41" in summary_text)
    add_check(results, "reporting_schema", case_id, "test_suite_summary_has_retention", True, "retention  : removed=2 | bytes=4096" in summary_text)

    summary["test_suite_summary_lines"] = len([line for line in summary_text.splitlines() if line.strip()])
    return results, summary
