from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set
from contextlib import redirect_stdout
import io

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_utils import parse_no_arg_cli, run_cli_entrypoint
from tools.local_regression.common import (
    ensure_reduced_dataset,
    load_manifest,
    resolve_run_dir,
    run_command,
    summarize_result,
    write_json,
    write_text,
)

CHECKLIST_PATH = PROJECT_ROOT / "doc" / "TEST_SUITE_CHECKLIST.md"
PROJECT_SETTINGS_PATH = PROJECT_ROOT / "doc" / "PROJECT_SETTINGS.md"
STATUS_VALUES = {"DONE", "PARTIAL", "TODO", "N/A"}
COVERAGE_TARGETS = [
    "tools/validate/synthetic_cases.py",
    "tools/validate/synthetic_meta_cases.py",
    "tools/validate/synthetic_unit_cases.py",
    "tools/validate/synthetic_history_cases.py",
    "tools/validate/synthetic_flow_cases.py",
    "tools/validate/synthetic_take_profit_cases.py",
    "tools/validate/synthetic_display_cases.py",
    "tools/validate/synthetic_reporting_cases.py",
    "tools/validate/synthetic_error_cases.py",
    "tools/validate/synthetic_data_quality_cases.py",
    "tools/validate/synthetic_cli_cases.py",
    "tools/validate/synthetic_strategy_cases.py",
    "tools/validate/synthetic_regression_cases.py",
    "tools/local_regression/run_chain_checks.py",
    "tools/local_regression/run_ml_smoke.py",
    "apps/test_suite.py",
    "tools/local_regression/run_all.py",
    "tools/validate/reporting.py",
    "tools/portfolio_sim/reporting.py",
    "core/scanner_display.py",
    "core/strategy_dashboard.py",
    "core/display_common.py",
    "core/price_utils.py",
    "core/history_filters.py",
    "core/portfolio_stats.py",
]
REQUIRED_META_IDS = ("B22", "B23", "B24", "B25", "B26")
PERFORMANCE_STEP_FILES = {
    "quick_gate": "quick_gate_summary.json",
    "consistency": "validate_consistency_summary.json",
    "chain_checks": "chain_summary.json",
    "ml_smoke": "ml_smoke_summary.json",
}
PERFORMANCE_MANIFEST_KEYS = {
    "quick_gate": "performance_quick_gate_max_sec",
    "consistency": "performance_consistency_max_sec",
    "chain_checks": "performance_chain_checks_max_sec",
    "ml_smoke": "performance_ml_smoke_max_sec",
}


def _extract_table_rows(text: str, heading: str) -> List[List[str]]:
    pattern = rf"^### {re.escape(heading)}\n\n((?:\|.*\n)+)"
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"找不到表格段落: {heading}")
    lines = [line.strip() for line in match.group(1).splitlines() if line.strip().startswith("|")]
    rows: List[List[str]] = []
    for line in lines[2:]:
        rows.append([part.strip() for part in line.strip("|").split("|")])
    return rows


def _load_checklist_tables() -> Dict[str, List[List[str]]]:
    text = CHECKLIST_PATH.read_text(encoding="utf-8")
    return {
        "B1": _extract_table_rows(text, "B1. 專案設定對應清單（不含暫時特例）"),
        "B2": _extract_table_rows(text, "B2. 未明列於專案設定，但正式 test suite 應納入"),
        "E1": _extract_table_rows(text, "E1. 目前所有 `PARTIAL` 的主表項目摘要"),
        "E2": _extract_table_rows(text, "E2. 目前所有 `TODO` 的主表項目摘要"),
        "F1": _extract_table_rows(text, "F1. 目前所有 `DONE` 的主表項目摘要"),
        "E3": _extract_table_rows(text, "E3. 目前所有未完成的建議測試項目摘要"),
    }


def _load_main_statuses(tables: Dict[str, List[List[str]]]) -> Dict[str, str]:
    statuses: Dict[str, str] = {}
    for key, status_idx in (("B1", 3), ("B2", 4)):
        for cols in tables[key]:
            if len(cols) <= status_idx:
                continue
            statuses[cols[0]] = cols[status_idx]
    return statuses


def _ids_from_table(rows: List[List[str]], idx: int = 1) -> List[str]:
    values: List[str] = []
    for cols in rows:
        if len(cols) > idx:
            values.append(cols[idx])
    return values


def _summarize_checklist_consistency() -> Dict[str, Any]:
    tables = _load_checklist_tables()
    main_statuses = _load_main_statuses(tables)
    results: List[Dict[str, Any]] = []

    invalid_statuses = {key: value for key, value in main_statuses.items() if value not in STATUS_VALUES}
    results.append(
        summarize_result(
            "checklist_main_status_values_valid",
            not invalid_statuses,
            detail=f"invalid={sorted(invalid_statuses.items())}" if invalid_statuses else f"checked={len(main_statuses)}",
            extra={"invalid_statuses": invalid_statuses},
        )
    )

    partial_ids = sorted(key for key, value in main_statuses.items() if value == "PARTIAL")
    todo_ids = sorted(key for key, value in main_statuses.items() if value == "TODO")
    done_ids = sorted(key for key, value in main_statuses.items() if value == "DONE")

    e1_ids = sorted(_ids_from_table(tables["E1"]))
    e2_ids = sorted(_ids_from_table(tables["E2"]))
    f1_ids = sorted(_ids_from_table(tables["F1"]))

    results.append(
        summarize_result(
            "checklist_partial_summary_matches_main_table",
            e1_ids == partial_ids,
            detail=f"summary={e1_ids} | main={partial_ids}",
            extra={"summary_ids": e1_ids, "main_ids": partial_ids},
        )
    )
    results.append(
        summarize_result(
            "checklist_todo_summary_matches_main_table",
            e2_ids == todo_ids,
            detail=f"summary={e2_ids} | main={todo_ids}",
            extra={"summary_ids": e2_ids, "main_ids": todo_ids},
        )
    )
    results.append(
        summarize_result(
            "checklist_done_summary_matches_main_table",
            f1_ids == done_ids,
            detail=f"summary={f1_ids} | main={done_ids}",
            extra={"summary_ids": f1_ids, "main_ids": done_ids},
        )
    )

    for target_id in REQUIRED_META_IDS:
        status = main_statuses.get(target_id, "")
        results.append(
            summarize_result(
                f"checklist_required_meta_id::{target_id}",
                status in {"PARTIAL", "DONE"},
                detail=f"status={status or 'missing'}",
            )
        )

    results.append(
        summarize_result(
            "checklist_done_not_listed_in_todo_summary",
            set(done_ids).isdisjoint(e2_ids),
            detail=f"overlap={sorted(set(done_ids) & set(e2_ids))}",
            extra={"overlap": sorted(set(done_ids) & set(e2_ids))},
        )
    )
    results.append(
        summarize_result(
            "checklist_done_not_listed_in_partial_summary",
            set(done_ids).isdisjoint(e1_ids),
            detail=f"overlap={sorted(set(done_ids) & set(e1_ids))}",
            extra={"overlap": sorted(set(done_ids) & set(e1_ids))},
        )
    )

    unfinished_d_ids = sorted(row[0] for row in tables["E3"] if len(row) > 2 and row[2] != "DONE")
    results.append(
        summarize_result(
            "checklist_unfinished_d_summary_nonempty_when_main_has_gaps",
            (len(partial_ids) + len(todo_ids) == 0) or bool(unfinished_d_ids),
            detail=f"unfinished_d={unfinished_d_ids}",
            extra={"unfinished_d_ids": unfinished_d_ids},
        )
    )

    ok = all(item["status"] == "PASS" for item in results)
    return {
        "ok": ok,
        "results": results,
        "partial_ids": partial_ids,
        "todo_ids": todo_ids,
        "done_ids": done_ids,
        "unfinished_d_ids": unfinished_d_ids,
    }




def _extract_backticked_paths(text: str) -> List[str]:
    return [match.strip() for match in re.findall(r"`([^`]+)`", text) if "/" in match or match.endswith('.py')]


def _extract_project_settings_formal_steps() -> List[str]:
    text = PROJECT_SETTINGS_PATH.read_text(encoding="utf-8")
    pattern = r"apps/test_suite\.py` 必須作為所有已實作測試的單一正式入口；其正式組成步驟目前為：\n((?:\s+`[^`]+`、?\n?)+)"
    match = re.search(pattern, text)
    if not match:
        raise ValueError("找不到 PROJECT_SETTINGS.md 中的正式組成步驟")
    return _extract_backticked_paths(match.group(1))


def _summarize_formal_entry_consistency() -> Dict[str, Any]:
    from apps.test_suite import STEP_LABELS
    from tools.local_regression.run_all import SCRIPT_ORDER, STEP_NAMES
    from tools.validate.preflight_env import _LOCAL_REGRESSION_STEP_ORDER

    results: List[Dict[str, Any]] = []
    project_settings_steps = _extract_project_settings_formal_steps()
    run_all_scripts = [script for _name, script, _summary in SCRIPT_ORDER]
    expected_project_settings_steps = [
        "tools/validate/preflight_env.py",
        "tools/local_regression/run_quick_gate.py",
        "tools/validate/cli.py --dataset reduced",
        "tools/local_regression/run_chain_checks.py",
        "tools/local_regression/run_ml_smoke.py",
        "tools/local_regression/run_meta_quality.py",
    ]

    results.append(
        summarize_result(
            "formal_entry_project_settings_steps_match_run_all",
            project_settings_steps == expected_project_settings_steps,
            detail=f"project_settings={project_settings_steps} | expected={expected_project_settings_steps}",
            extra={"project_settings_steps": project_settings_steps, "expected_steps": expected_project_settings_steps},
        )
    )

    results.append(
        summarize_result(
            "formal_entry_run_all_steps_match_preflight_order",
            list(STEP_NAMES) == list(_LOCAL_REGRESSION_STEP_ORDER),
            detail=f"run_all={list(STEP_NAMES)} | preflight={list(_LOCAL_REGRESSION_STEP_ORDER)}",
            extra={"run_all_steps": list(STEP_NAMES), "preflight_steps": list(_LOCAL_REGRESSION_STEP_ORDER)},
        )
    )

    missing_step_labels = [step for step in STEP_NAMES if step not in STEP_LABELS]
    results.append(
        summarize_result(
            "formal_entry_test_suite_labels_cover_run_all_steps",
            not missing_step_labels,
            detail=f"missing={missing_step_labels}",
            extra={"missing_step_labels": missing_step_labels},
        )
    )

    required_extra_labels = ["preflight", "dataset_prepare", "manifest"]
    missing_extra_labels = [label for label in required_extra_labels if label not in STEP_LABELS]
    results.append(
        summarize_result(
            "formal_entry_test_suite_labels_cover_non_script_stages",
            not missing_extra_labels,
            detail=f"missing={missing_extra_labels}",
            extra={"missing_extra_labels": missing_extra_labels},
        )
    )

    missing_script_files = []
    for script in expected_project_settings_steps:
        script_path = script.split()[0].strip()
        if not (PROJECT_ROOT / script_path).exists():
            missing_script_files.append(script)
    results.append(
        summarize_result(
            "formal_entry_all_declared_scripts_exist",
            not missing_script_files,
            detail=f"missing={missing_script_files}",
            extra={"missing_script_files": missing_script_files},
        )
    )

    ok = all(item["status"] == "PASS" for item in results)
    return {
        "ok": ok,
        "results": results,
        "project_settings_steps": project_settings_steps,
        "run_all_steps": list(STEP_NAMES),
        "preflight_steps": list(_LOCAL_REGRESSION_STEP_ORDER),
    }


def _exercise_coverage_formal_helpers(coverage_dir: Path) -> Dict[str, Any]:
    from apps import test_suite as test_suite_module
    from core.scanner_display import print_scanner_header
    from core.strategy_dashboard import print_strategy_dashboard
    from tools.local_regression import run_all as run_all_module
    from tools.local_regression import run_chain_checks as chain_checks_module
    from tools.local_regression import run_ml_smoke as ml_smoke_module
    from tools.portfolio_sim import reporting as portfolio_reporting
    from tools.validate.reporting import print_console_summary

    probe_dir = coverage_dir / "formal_helper_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)

    def _silent_call(callable_obj):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            callable_obj()
        return buffer.getvalue()

    valid_params_payload = {
        "atr_len": 14,
        "atr_times_init": 2.0,
        "atr_times_trail": 3.0,
        "atr_buy_tol": 1.0,
        "high_len": 20,
        "tp_percent": 0.25,
        "min_history_trades": 5,
        "min_history_ev": 0.1,
        "min_history_win_rate": 0.55,
    }
    params_path = probe_dir / "coverage_best_params.json"
    params_path.write_text(json.dumps(valid_params_payload, ensure_ascii=False), encoding="utf-8")
    profile_path = probe_dir / "optimizer_profile_summary_probe.json"
    profile_path.write_text(json.dumps({"trial_count": 1, "avg": {"objective_wall_sec": 1.234}}, ensure_ascii=False), encoding="utf-8")

    params_info = ml_smoke_module._load_params_payload(params_path)
    profile_info = ml_smoke_module._read_latest_profile_metrics({profile_path.name: profile_path})
    ml_smoke_module._canonical_payload_digest(params_info["payload"])
    probe_run_a = {
        "label": "probe-a",
        "status": "PASS",
        "db_trial_count": 1,
        "qualified_trial_count": 1,
        "best_trial_value": 0.5,
        "best_params_digest": "abc",
        "optimizer_profile_trial_count": 1,
        "optimizer_profile_avg_objective_wall_sec": profile_info["optimizer_profile_avg_objective_wall_sec"],
        "failures": [],
    }
    probe_run_b = {
        "label": "probe-b",
        "status": "PASS",
        "db_trial_count": 1,
        "qualified_trial_count": 1,
        "best_trial_value": 0.5,
        "best_params_digest": "abc",
        "optimizer_profile_trial_count": 1,
        "optimizer_profile_avg_objective_wall_sec": profile_info["optimizer_profile_avg_objective_wall_sec"],
        "failures": [],
    }
    ml_smoke_module._build_repro_summary(probe_run_a, probe_run_b)

    normalized_row = chain_checks_module._normalize_scanner_candidate_row({
        "kind": "buy",
        "ticker": "2330",
        "proj_cost": 12345.6789,
        "expected_value": 1.23456789,
        "sort_value": 2.34567891,
        "text": "probe",
    })
    chain_checks_module._build_highlights([
        {"ticker": "2330", "filled_count": 2, "portfolio_missed_buy_count": 0, "sanitize_dropped": 0, "blocked_by": "filled_or_missed_buy"},
        {"ticker": "2317", "filled_count": 0, "portfolio_missed_buy_count": 1, "sanitize_dropped": 1, "blocked_by": "cash"},
    ])
    canonical_chain_payload = chain_checks_module._canonical_chain_payload({
        "summary_rows": [normalized_row],
        "highlights": {"traded_ticker_count": 1},
        "scanner_snapshot": {"candidate_count": 1},
    })
    chain_checks_module._payload_digest(canonical_chain_payload)

    _silent_call(lambda: print_scanner_header({
        "high_len": 20,
        "atr_len": 14,
        "atr_buy_tol": 1.0,
        "atr_times_init": 2.0,
        "atr_times_trail": 3.0,
        "tp_percent": 0.25,
        "use_bb": False,
        "use_kc": False,
        "use_vol": False,
        "min_history_trades": 5,
        "min_history_win_rate": 0.55,
        "min_history_ev": 0.1,
    }))
    _silent_call(lambda: print_strategy_dashboard(
        {"high_len": 20, "atr_len": 14, "atr_buy_tol": 1.0, "atr_times_init": 2.0, "atr_times_trail": 3.0, "tp_percent": 0.25, "use_bb": False, "use_kc": False, "use_vol": False, "min_history_trades": 5, "min_history_win_rate": 0.55, "min_history_ev": 0.1},
        title="coverage-probe",
        mode_display="投組模式",
        max_pos=3,
        trades=2,
        missed_b=0,
        missed_s=0,
        final_eq=1000000,
        avg_exp=50.0,
        sys_ret=1.0,
        bm_ret=0.5,
        sys_mdd=2.0,
        bm_mdd=3.0,
        win_rate=50.0,
        payoff=1.2,
        ev=0.2,
        benchmark_ticker="0050",
        max_exp=60.0,
        r_sq=0.9,
        m_win_rate=50.0,
        bm_r_sq=0.8,
        bm_m_win_rate=40.0,
        normal_trades=2,
        extended_trades=0,
        annual_trades=2.0,
        reserved_buy_fill_rate=100.0,
        annual_return_pct=1.0,
        bm_annual_return_pct=0.5,
        min_full_year_return_pct=1.0,
        bm_min_full_year_return_pct=0.2,
    ))

    _silent_call(lambda: print_console_summary(
        df_results=__import__("pandas").DataFrame([{"ticker": "2330", "module": "stats", "metric": "ev", "status": "PASS", "passed": True, "expected": "1", "actual": "1", "note": ""}]),
        df_failed=__import__("pandas").DataFrame([], columns=["ticker", "module", "metric", "status", "passed", "expected", "actual", "note"]),
        df_summary=__import__("pandas").DataFrame([{"ticker": "2330", "synthetic": False}]),
        csv_path="outputs/validate/coverage_probe.csv",
        xlsx_path="outputs/validate/coverage_probe.xlsx",
        elapsed_time=1.23,
        real_summary_count=1,
        real_tickers=["2330"],
        normalize_ticker_text=lambda value: str(value).strip(),
        max_console_fail_preview=3,
    ))
    _silent_call(lambda: portfolio_reporting.print_yearly_return_report([{"year": 2024, "year_return_pct": 1.23, "is_full_year": True, "start_date": "2024-01-02", "end_date": "2024-12-31"}]))
    preflight_probe = run_all_module._safe_format_preflight_summary({
        "status": "FAIL",
        "python_executable": sys.executable,
        "duration_sec": 0.123,
        "failed_packages": ["coverage"],
        "runtime_error": "",
    })
    run_all_module._compute_not_run_step_names(
        selected_step_names=["quick_gate", "consistency", "chain_checks", "ml_smoke", "meta_quality"],
        failed_step_names=["ml_smoke"],
        completed_script_names=["quick_gate", "consistency"],
        include_dataset=True,
    )
    run_all_module._build_bundle_entries(probe_dir, [params_path, profile_path])
    dataset_probe = run_all_module._write_dataset_prepare_summary(probe_dir, {
        "status": "PASS",
        "duration_sec": 0.456,
        "dataset_dir": "data/tw_stock_data_vip_reduced",
        "source": "existing",
        "csv_count": 24,
        "reused_existing": True,
    })

    _silent_call(lambda: test_suite_module._print_human_summary({
        "overall_status": "PASS",
        "failures": 0,
        "selected_steps": ["quick_gate", "consistency", "chain_checks", "ml_smoke", "meta_quality"],
        "failed_step_names": [],
        "not_run_step_names": [],
        "scripts": [{"name": "quick_gate", "status": "PASS", "duration_sec": 1.0, "failure_reasons": []}, {"name": "consistency", "status": "PASS", "duration_sec": 1.0, "failure_reasons": []}, {"name": "chain_checks", "status": "PASS", "duration_sec": 1.0, "failure_reasons": []}, {"name": "ml_smoke", "status": "PASS", "duration_sec": 1.0, "failure_reasons": []}, {"name": "meta_quality", "status": "PASS", "duration_sec": 1.0, "failure_reasons": []}],
        "step_payloads": {"quick_gate": {"status": "PASS", "step_count": 1}, "consistency": {"status": "PASS", "total_checks": 1, "fail_count": 0, "skip_count": 0, "real_ticker_count": 1}, "chain_checks": {"status": "PASS", "ticker_count": 1, "highlights": {"blocked_by_counts": {}}, "portfolio_snapshot": {"trade_rows": 1, "reserved_buy_fill_rate": 100.0}}, "ml_smoke": {"status": "PASS", "db_trial_count": 1}, "meta_quality": {"status": "PASS", "fail_count": 0, "coverage": {"totals": {"percent_covered": 1.0}}, "checklist": {"todo_ids": []}}},
        "preflight": {"status": "PASS", "failed_packages": []},
        "bundle_mode": "minimum_set",
        "archived_bundle": "outputs/local_regression/probe.zip",
        "root_bundle_copy": "probe.zip",
        "bundle_entries": ["master_summary.json"],
        "retention": {"removed_count": 0, "removed_bytes": 0},
    }))

    return {
        "ml_params_keys": sorted(params_info["payload"].keys()),
        "profile_trial_count": profile_info["optimizer_profile_trial_count"],
        "normalized_ticker": normalized_row["ticker"],
        "run_all_preflight_lines": len([line for line in preflight_probe.splitlines() if line.strip()]),
        "dataset_probe_status": dataset_probe.get("status"),
    }


def _build_coverage_summary(run_dir: Path) -> Dict[str, Any]:
    import coverage
    from core.params_io import load_params_from_json
    from tools.validate.synthetic_cases import run_synthetic_consistency_suite

    coverage_dir = run_dir / "coverage_artifacts"
    coverage_dir.mkdir(parents=True, exist_ok=True)
    data_file = coverage_dir / ".coverage.synthetic"
    json_file = coverage_dir / "coverage_synthetic.json"

    run_result = {"returncode": 1, "stdout": "", "stderr": "", "timed_out": False}
    payload: Dict[str, Any] = {}
    synthetic_fail_count = 0

    cov = coverage.Coverage(data_file=str(data_file), branch=True)
    try:
        base_params = load_params_from_json(PROJECT_ROOT / "models" / "best_params.json")
        cov.start()
        results, summaries = run_synthetic_consistency_suite(base_params)
        formal_helper_probe = _exercise_coverage_formal_helpers(coverage_dir)
        synthetic_fail_count = sum(1 for row in results if row.get("status") == "FAIL")
        run_result["returncode"] = 0 if synthetic_fail_count == 0 else 1
        run_result["stdout"] = json.dumps({"synthetic_case_count": len(summaries), "synthetic_fail_count": synthetic_fail_count, "formal_helper_probe": formal_helper_probe}, ensure_ascii=False)
    except BaseException as exc:
        run_result["returncode"] = 1
        run_result["stderr"] = f"{type(exc).__name__}: {exc}"
    finally:
        cov.stop()
        cov.save()

    json_ok = False
    if data_file.exists():
        try:
            cov.json_report(outfile=str(json_file), pretty_print=True)
            payload = json.loads(json_file.read_text(encoding="utf-8"))
            json_ok = True
        except BaseException as exc:
            run_result["stderr"] = (run_result["stderr"] + "\n" if run_result["stderr"] else "") + f"json_report: {type(exc).__name__}: {exc}"

    totals = payload.get("totals", {}) if isinstance(payload, dict) else {}
    files_payload = payload.get("files", {}) if isinstance(payload, dict) else {}

    project_root_norm = str(PROJECT_ROOT).replace("\\", "/").rstrip("/")

    def _normalize_coverage_key(raw_path: str) -> str:
        normalized = str(raw_path).replace("\\", "/")
        if project_root_norm and normalized.startswith(project_root_norm + "/"):
            normalized = normalized[len(project_root_norm) + 1 :]
        return normalized.lstrip("/")

    normalized_files_payload = {
        _normalize_coverage_key(raw_key): file_info
        for raw_key, file_info in files_payload.items()
    }

    key_files: Dict[str, Dict[str, Any]] = {}
    missing_targets: List[str] = []
    zero_covered_targets: List[str] = []
    for rel_path in COVERAGE_TARGETS:
        file_info = normalized_files_payload.get(_normalize_coverage_key(rel_path))
        summary = dict(file_info.get("summary", {})) if isinstance(file_info, dict) else {}
        if not summary:
            missing_targets.append(rel_path)
            key_files[rel_path] = {"present": False, "covered_lines": 0}
            continue
        covered_lines = int(summary.get("covered_lines", 0) or 0)
        if covered_lines <= 0:
            zero_covered_targets.append(rel_path)
        key_files[rel_path] = {
            "present": True,
            "covered_lines": covered_lines,
            "num_statements": int(summary.get("num_statements", 0) or 0),
            "percent_covered": float(summary.get("percent_covered", 0.0) or 0.0),
            "covered_branches": int(summary.get("covered_branches", 0) or 0),
            "num_branches": int(summary.get("num_branches", 0) or 0),
        }

    results = [
        summarize_result(
            "coverage_synthetic_suite_runs_successfully",
            run_result["returncode"] == 0,
            detail=(run_result.get("stderr", "") or run_result.get("stdout", "") or "ok").splitlines()[0],
            extra={"returncode": run_result["returncode"], "synthetic_fail_count": synthetic_fail_count},
        ),
        summarize_result(
            "coverage_json_generated",
            json_ok and json_file.exists(),
            detail=run_result.get("stderr", "").splitlines()[0] if run_result.get("stderr") else str(json_file),
            extra={"json_file": str(json_file)},
        ),
        summarize_result(
            "coverage_overall_nonzero",
            float(totals.get("percent_covered", 0.0) or 0.0) > 0.0,
            detail=(
                f"line={totals.get('covered_lines', 0)}/{totals.get('num_statements', 0)} | "
                f"branch={totals.get('covered_branches', 0)}/{totals.get('num_branches', 0)} | "
                f"percent={totals.get('percent_covered', 0.0)}"
            ),
            extra={"totals": totals},
        ),
        summarize_result(
            "coverage_key_targets_present",
            not missing_targets,
            detail=f"missing={missing_targets}",
            extra={"missing_targets": missing_targets},
        ),
        summarize_result(
            "coverage_key_targets_hit",
            not zero_covered_targets,
            detail=f"zero_covered={zero_covered_targets}",
            extra={"zero_covered_targets": zero_covered_targets},
        ),
    ]

    ok = all(item["status"] == "PASS" for item in results)
    return {
        "ok": ok,
        "results": results,
        "run_result": run_result,
        "json_file": str(json_file),
        "totals": {
            "covered_lines": int(totals.get("covered_lines", 0) or 0),
            "num_statements": int(totals.get("num_statements", 0) or 0),
            "covered_branches": int(totals.get("covered_branches", 0) or 0),
            "num_branches": int(totals.get("num_branches", 0) or 0),
            "percent_covered": float(totals.get("percent_covered", 0.0) or 0.0),
            "synthetic_fail_count": synthetic_fail_count,
        },
        "key_files": key_files,
        "missing_targets": missing_targets,
        "zero_covered_targets": zero_covered_targets,
    }


def _build_performance_summary(run_dir: Path, manifest: Dict[str, Any], *, current_meta_quality_duration_sec: float) -> Dict[str, Any]:
    has_shared_run_dir = bool(os.environ.get("V16_LOCAL_REGRESSION_RUN_DIR", "").strip())
    available_step_files = {name: run_dir / file_name for name, file_name in PERFORMANCE_STEP_FILES.items()}
    if not has_shared_run_dir and not any(path.exists() for path in available_step_files.values()):
        results = [
            summarize_result(
                "performance_baseline_skipped_without_shared_run_dir",
                True,
                detail="standalone run_meta_quality 無 shared run_dir；略過 step performance baseline",
            )
        ]
        return {
            "ok": True,
            "skipped": True,
            "results": results,
            "step_durations": {},
            "optimizer_trial_avg_objective_wall_sec": None,
            "optimizer_profile_trial_count": 0,
            "total_duration_sec": current_meta_quality_duration_sec,
        }

    results: List[Dict[str, Any]] = []
    step_durations: Dict[str, float] = {}
    missing_step_files: List[str] = []
    for step_name, path in available_step_files.items():
        if not path.exists():
            missing_step_files.append(step_name)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        step_durations[step_name] = round(float(payload.get("duration_sec", 0.0) or 0.0), 3)

    results.append(
        summarize_result(
            "performance_required_step_summaries_present",
            not missing_step_files,
            detail=f"missing={missing_step_files}",
            extra={"missing_step_files": missing_step_files},
        )
    )

    for step_name, duration_sec in step_durations.items():
        budget_key = PERFORMANCE_MANIFEST_KEYS[step_name]
        budget_sec = float(manifest[budget_key])
        results.append(
            summarize_result(
                f"performance_{step_name}_within_budget",
                duration_sec <= budget_sec,
                detail=f"duration={duration_sec:.3f}s | budget={budget_sec:.3f}s",
                extra={"duration_sec": duration_sec, "budget_sec": budget_sec},
            )
        )

    meta_quality_budget_sec = float(manifest["performance_meta_quality_max_sec"])
    results.append(
        summarize_result(
            "performance_meta_quality_within_budget",
            current_meta_quality_duration_sec <= meta_quality_budget_sec,
            detail=f"duration={current_meta_quality_duration_sec:.3f}s | budget={meta_quality_budget_sec:.3f}s",
            extra={"duration_sec": current_meta_quality_duration_sec, "budget_sec": meta_quality_budget_sec},
        )
    )

    total_duration_sec = round(sum(step_durations.values()) + float(current_meta_quality_duration_sec), 3)
    total_budget_sec = float(manifest["performance_total_max_sec"])
    results.append(
        summarize_result(
            "performance_total_suite_within_budget",
            total_duration_sec <= total_budget_sec,
            detail=f"duration={total_duration_sec:.3f}s | budget={total_budget_sec:.3f}s",
            extra={"duration_sec": total_duration_sec, "budget_sec": total_budget_sec},
        )
    )

    ml_smoke_payload = {}
    ml_smoke_path = available_step_files["ml_smoke"]
    if ml_smoke_path.exists():
        ml_smoke_payload = json.loads(ml_smoke_path.read_text(encoding="utf-8"))
    optimizer_trial_avg = ml_smoke_payload.get("optimizer_profile_avg_objective_wall_sec")
    optimizer_profile_trial_count = int(ml_smoke_payload.get("optimizer_profile_trial_count", 0) or 0)
    optimizer_trial_budget_sec = float(manifest["performance_optimizer_trial_avg_max_sec"])
    optimizer_profile_ready = optimizer_trial_avg not in (None, "")
    results.append(
        summarize_result(
            "performance_optimizer_profile_present",
            optimizer_profile_ready,
            detail=(
                f"avg_objective_wall_sec={optimizer_trial_avg} | trial_count={optimizer_profile_trial_count}"
                if optimizer_profile_ready
                else "optimizer profile summary missing"
            ),
            extra={
                "optimizer_profile_trial_count": optimizer_profile_trial_count,
                "optimizer_profile_avg_objective_wall_sec": optimizer_trial_avg,
            },
        )
    )
    if optimizer_profile_ready:
        optimizer_trial_avg_value = float(optimizer_trial_avg)
        results.append(
            summarize_result(
                "performance_optimizer_avg_trial_within_budget",
                optimizer_trial_avg_value <= optimizer_trial_budget_sec,
                detail=f"avg_objective_wall={optimizer_trial_avg_value:.3f}s | budget={optimizer_trial_budget_sec:.3f}s",
                extra={"duration_sec": optimizer_trial_avg_value, "budget_sec": optimizer_trial_budget_sec},
            )
        )
    else:
        optimizer_trial_avg_value = None

    ok = all(item["status"] == "PASS" for item in results)
    return {
        "ok": ok,
        "skipped": False,
        "results": results,
        "step_durations": step_durations,
        "optimizer_trial_avg_objective_wall_sec": optimizer_trial_avg_value,
        "optimizer_profile_trial_count": optimizer_profile_trial_count,
        "total_duration_sec": total_duration_sec,
    }


def main(argv=None) -> int:
    cli = parse_no_arg_cli(argv, "tools/local_regression/run_meta_quality.py", description="執行 coverage baseline 與 checklist sufficiency formal check")
    if cli["help"]:
        return 0

    manifest = load_manifest()
    ensure_reduced_dataset()
    run_dir = resolve_run_dir("meta_quality")

    started = os.times().elapsed
    coverage_summary = _build_coverage_summary(run_dir)
    checklist_summary = _summarize_checklist_consistency()
    formal_entry_summary = _summarize_formal_entry_consistency()
    current_meta_quality_duration_sec = round(os.times().elapsed - started, 3)
    performance_summary = _build_performance_summary(run_dir, manifest, current_meta_quality_duration_sec=current_meta_quality_duration_sec)

    all_results = [*coverage_summary["results"], *checklist_summary["results"], *formal_entry_summary["results"], *performance_summary["results"]]
    failures = [item["name"] for item in all_results if item["status"] != "PASS"]
    overall_status = "PASS" if not failures else "FAIL"

    summary = {
        "status": overall_status,
        "failures": failures,
        "fail_count": len(failures),
        "coverage": {
            "ok": coverage_summary["ok"],
            "totals": coverage_summary["totals"],
            "json_file": coverage_summary["json_file"],
            "missing_targets": coverage_summary["missing_targets"],
            "zero_covered_targets": coverage_summary["zero_covered_targets"],
        },
        "checklist": {
            "ok": checklist_summary["ok"],
            "partial_ids": checklist_summary["partial_ids"],
            "todo_ids": checklist_summary["todo_ids"],
            "done_ids": checklist_summary["done_ids"],
            "unfinished_d_ids": checklist_summary["unfinished_d_ids"],
        },
        "formal_entry": {
            "ok": formal_entry_summary["ok"],
            "project_settings_steps": formal_entry_summary["project_settings_steps"],
            "run_all_steps": formal_entry_summary["run_all_steps"],
            "preflight_steps": formal_entry_summary["preflight_steps"],
        },
        "performance": {
            "ok": performance_summary["ok"],
            "skipped": performance_summary["skipped"],
            "step_durations": performance_summary["step_durations"],
            "optimizer_profile_trial_count": performance_summary["optimizer_profile_trial_count"],
            "optimizer_trial_avg_objective_wall_sec": performance_summary["optimizer_trial_avg_objective_wall_sec"],
            "total_duration_sec": performance_summary["total_duration_sec"],
        },
        "results": all_results,
    }
    write_json(run_dir / "meta_quality_summary.json", summary)

    lines = [
        f"status        : {overall_status}",
        f"fail_count    : {len(failures)}",
        f"coverage_ok   : {coverage_summary['ok']}",
        f"checklist_ok  : {checklist_summary['ok']}",
        f"formal_entry_ok: {formal_entry_summary['ok']}",
        (
            f"coverage      : line {coverage_summary['totals']['covered_lines']}/"
            f"{coverage_summary['totals']['num_statements']} | branch {coverage_summary['totals']['covered_branches']}/"
            f"{coverage_summary['totals']['num_branches']} | percent={coverage_summary['totals']['percent_covered']:.2f}"
        ),
        f"missing_cov   : {', '.join(coverage_summary['missing_targets']) if coverage_summary['missing_targets'] else '(none)'}",
        f"zero_cov      : {', '.join(coverage_summary['zero_covered_targets']) if coverage_summary['zero_covered_targets'] else '(none)'}",
        f"partial_ids   : {', '.join(checklist_summary['partial_ids']) if checklist_summary['partial_ids'] else '(none)'}",
        f"todo_ids      : {', '.join(checklist_summary['todo_ids']) if checklist_summary['todo_ids'] else '(none)'}",
        f"done_ids      : {', '.join(checklist_summary['done_ids']) if checklist_summary['done_ids'] else '(none)'}",
        f"performance_ok: {performance_summary['ok']}",
        f"perf_total    : {performance_summary['total_duration_sec']:.3f}s",
        (
            f"perf_opt_trial: {performance_summary['optimizer_trial_avg_objective_wall_sec']:.3f}s"
            if performance_summary['optimizer_trial_avg_objective_wall_sec'] is not None
            else "perf_opt_trial: (missing)"
        ),
    ]
    if failures:
        lines.append("failed_checks : " + ", ".join(failures))
    write_text(run_dir / "meta_quality_summary.txt", "\n".join(lines) + "\n")
    print(json.dumps({
        "status": overall_status,
        "fail_count": len(failures),
        "coverage_percent": coverage_summary["totals"]["percent_covered"],
        "checklist_partial_ids": checklist_summary["partial_ids"],
        "checklist_todo_ids": checklist_summary["todo_ids"],
        "performance_total_duration_sec": performance_summary["total_duration_sec"],
        "optimizer_trial_avg_objective_wall_sec": performance_summary["optimizer_trial_avg_objective_wall_sec"],
    }, ensure_ascii=False))
    return 0 if overall_status == "PASS" else 1


if __name__ == "__main__":
    run_cli_entrypoint(main)
