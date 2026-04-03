from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Callable, Dict, List

from tools.local_regression.common import summarize_result
from tools.local_regression.formal_pipeline import (
    DATASET_REQUIRED_STEPS as FORMAL_DATASET_REQUIRED_STEPS,
    FORMAL_STEP_ORDER,
)
from tools.local_regression.meta_quality_targets import (
    CORE_TRADING_COVERAGE_TARGETS,
    COVERAGE_BRANCH_MIN_FLOOR,
    COVERAGE_LINE_MIN_FLOOR,
    COVERAGE_MAX_LINE_BRANCH_GAP,
    COVERAGE_TARGETS,
    CRITICAL_COVERAGE_BRANCH_MIN_FLOOR,
    CRITICAL_COVERAGE_LINE_MIN_FLOOR,
    CRITICAL_COVERAGE_TARGETS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _ratio_percent(numerator: Any, denominator: Any) -> float:
    try:
        numerator_value = float(numerator or 0.0)
        denominator_value = float(denominator or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if denominator_value <= 0.0:
        return 0.0
    return (numerator_value / denominator_value) * 100.0


def _coverage_percent_from_summary(summary: Dict[str, Any], *, branch: bool = False) -> float:
    if branch:
        return _ratio_percent(summary.get("covered_branches", 0), summary.get("num_branches", 0))
    return _ratio_percent(summary.get("covered_lines", 0), summary.get("num_statements", 0))


def _coverage_threshold_policy_ok(line_min_percent: float, branch_min_percent: float) -> tuple[bool, str]:
    branch_priority_gap_ok = (line_min_percent - branch_min_percent) <= COVERAGE_MAX_LINE_BRANCH_GAP
    ok = (
        line_min_percent >= COVERAGE_LINE_MIN_FLOOR
        and branch_min_percent >= COVERAGE_BRANCH_MIN_FLOOR
        and line_min_percent >= branch_min_percent
        and branch_priority_gap_ok
    )
    detail = (
        f"line_min={line_min_percent:.2f} | branch_min={branch_min_percent:.2f} | "
        f"line_floor={COVERAGE_LINE_MIN_FLOOR:.2f} | branch_floor={COVERAGE_BRANCH_MIN_FLOOR:.2f} | "
        f"max_gap={COVERAGE_MAX_LINE_BRANCH_GAP:.2f}"
    )
    return ok, detail


def _critical_coverage_threshold_policy_ok(line_min_percent: float, branch_min_percent: float) -> tuple[bool, str]:
    branch_priority_gap_ok = (line_min_percent - branch_min_percent) <= COVERAGE_MAX_LINE_BRANCH_GAP
    ok = (
        line_min_percent >= CRITICAL_COVERAGE_LINE_MIN_FLOOR
        and branch_min_percent >= CRITICAL_COVERAGE_BRANCH_MIN_FLOOR
        and line_min_percent >= branch_min_percent
        and branch_priority_gap_ok
    )
    detail = (
        f"critical_line_min={line_min_percent:.2f} | critical_branch_min={branch_min_percent:.2f} | "
        f"critical_line_floor={CRITICAL_COVERAGE_LINE_MIN_FLOOR:.2f} | critical_branch_floor={CRITICAL_COVERAGE_BRANCH_MIN_FLOOR:.2f} | "
        f"max_gap={COVERAGE_MAX_LINE_BRANCH_GAP:.2f}"
    )
    return ok, detail


def _exercise_coverage_formal_helpers(coverage_dir: Path) -> Dict[str, Any]:
    from core.test_suite_reporting import TEST_SUITE_STEP_LABELS, print_test_suite_human_summary
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
    probe_run_b = dict(probe_run_a, label="probe-b")
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
        csv_path="outputs/validate_consistency/coverage_probe.csv",
        xlsx_path="outputs/validate_consistency/coverage_probe.xlsx",
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

    _silent_call(lambda: print_test_suite_human_summary(
        {
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
        },
        regression_step_order=FORMAL_STEP_ORDER,
        dataset_required_steps=FORMAL_DATASET_REQUIRED_STEPS,
        step_labels=TEST_SUITE_STEP_LABELS,
    ))

    return {
        "ml_params_keys": sorted(params_info["payload"].keys()),
        "profile_trial_count": profile_info["optimizer_profile_trial_count"],
        "normalized_ticker": normalized_row["ticker"],
        "run_all_preflight_lines": len([line for line in preflight_probe.splitlines() if line.strip()]),
        "dataset_probe_status": dataset_probe.get("status"),
    }


def _load_reusable_coverage_artifacts(coverage_dir: Path) -> tuple[Dict[str, Any] | None, str]:
    def _preview_value(raw_value: Any) -> str:
        preview = repr(raw_value)
        if len(preview) > 100:
            preview = preview[:97] + "..."
        return preview

    def _coerce_int(raw_value: Any, *, field_name: str) -> int:
        try:
            return int(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name}={_preview_value(raw_value)} ({type(exc).__name__})") from exc

    json_file = coverage_dir / "coverage_synthetic.json"
    run_info_file = coverage_dir / "coverage_run_info.json"
    if not json_file.exists() or not run_info_file.exists():
        return None, ""
    try:
        payload = json.loads(json_file.read_text(encoding="utf-8"))
        run_info = json.loads(run_info_file.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        return None, f"coverage_reuse_error={type(exc).__name__}:{exc}"
    if not isinstance(payload, dict) or not isinstance(run_info, dict):
        return None, (
            "coverage_reuse_error=invalid_schema("
            f"payload={type(payload).__name__},run_info={type(run_info).__name__})"
        )
    source = str(run_info.get("source", "") or "")
    if source != "validate_consistency":
        return None, f"coverage_reuse_error=unexpected_source({source or 'missing'})"
    try:
        return {
            "payload": payload,
            "run_result": {
                "returncode": _coerce_int(run_info.get("returncode", 1), field_name="returncode"),
                "stdout": str(run_info.get("stdout", "") or ""),
                "stderr": str(run_info.get("stderr", "") or ""),
                "timed_out": bool(run_info.get("timed_out", False)),
            },
            "synthetic_fail_count": _coerce_int(run_info.get("synthetic_fail_count", 0) or 0, field_name="synthetic_fail_count"),
            "synthetic_case_count": _coerce_int(run_info.get("synthetic_case_count", 0) or 0, field_name="synthetic_case_count"),
            "json_file": str(json_file),
        }, ""
    except ValueError as exc:
        return None, f"coverage_reuse_error=invalid_run_info({exc})"


def build_coverage_summary(
    run_dir: Path,
    manifest: Dict[str, Any],
    *,
    suite_runner: Callable[[Dict[str, Any]], tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] | None = None,
) -> Dict[str, Any]:
    from core.params_io import load_params_from_json

    coverage_dir = run_dir / "coverage_artifacts"
    coverage_dir.mkdir(parents=True, exist_ok=True)
    data_file = coverage_dir / ".coverage.synthetic"
    json_file = coverage_dir / "coverage_synthetic.json"

    run_result = {"returncode": 1, "stdout": "", "stderr": "", "timed_out": False}
    payload: Dict[str, Any] = {}
    synthetic_fail_count = 0
    reused_existing = False
    json_ok = False

    reusable, coverage_reuse_error = _load_reusable_coverage_artifacts(coverage_dir)
    formal_helper_probe = _exercise_coverage_formal_helpers(coverage_dir)
    if reusable is not None:
        payload = reusable["payload"]
        run_result = dict(reusable["run_result"])
        synthetic_fail_count = int(reusable["synthetic_fail_count"])
        if formal_helper_probe and not run_result.get("stderr"):
            run_result["stdout"] = json.dumps(
                {
                    "synthetic_case_count": reusable.get("synthetic_case_count", 0),
                    "synthetic_fail_count": synthetic_fail_count,
                    "formal_helper_probe": formal_helper_probe,
                    "reused_existing": True,
                },
                ensure_ascii=False,
            )
        reused_existing = True
        json_ok = True
    else:
        if suite_runner is None:
            run_result["stderr"] = "coverage_runner_missing"
        else:
            import coverage

            cov = coverage.Coverage(data_file=str(data_file), branch=True)
            try:
                base_params = load_params_from_json(PROJECT_ROOT / "models" / "best_params.json")
                cov.start()
                results, summaries = suite_runner(base_params)
                synthetic_fail_count = sum(1 for row in results if row.get("status") == "FAIL")
                run_result["returncode"] = 0 if synthetic_fail_count == 0 else 1
                run_result["stdout"] = json.dumps({
                    "synthetic_case_count": len(summaries),
                    "synthetic_fail_count": synthetic_fail_count,
                    "formal_helper_probe": formal_helper_probe,
                }, ensure_ascii=False)
            except Exception as exc:
                run_result["returncode"] = 1
                run_result["stderr"] = f"{type(exc).__name__}: {exc}"
            finally:
                cov.stop()
                cov.save()

            if data_file.exists():
                try:
                    cov.json_report(outfile=str(json_file), pretty_print=True)
                    payload = json.loads(json_file.read_text(encoding="utf-8"))
                    json_ok = True
                except Exception as exc:
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
            key_files[rel_path] = {"present": False, "covered_lines": 0, "covered_branches": 0}
            continue
        covered_lines = int(summary.get("covered_lines", 0) or 0)
        covered_branches = int(summary.get("covered_branches", 0) or 0)
        num_statements = int(summary.get("num_statements", 0) or 0)
        num_branches = int(summary.get("num_branches", 0) or 0)
        if covered_lines <= 0:
            zero_covered_targets.append(rel_path)
        key_files[rel_path] = {
            "present": True,
            "covered_lines": covered_lines,
            "num_statements": num_statements,
            "percent_covered": float(summary.get("percent_covered", 0.0) or 0.0),
            "covered_branches": covered_branches,
            "num_branches": num_branches,
            "line_percent_covered": _ratio_percent(covered_lines, num_statements),
            "branch_percent_covered": _ratio_percent(covered_branches, num_branches),
        }

    core_target_missing = sorted(rel_path for rel_path in CORE_TRADING_COVERAGE_TARGETS if rel_path not in COVERAGE_TARGETS)

    line_percent_covered = _ratio_percent(totals.get("covered_lines", 0), totals.get("num_statements", 0))
    branch_percent_covered = _ratio_percent(totals.get("covered_branches", 0), totals.get("num_branches", 0))
    line_min_percent = float(manifest["coverage_line_min_percent"])
    branch_min_percent = float(manifest["coverage_branch_min_percent"])
    critical_line_min_percent = float(manifest["coverage_critical_line_min_percent"])
    critical_branch_min_percent = float(manifest["coverage_critical_branch_min_percent"])
    threshold_policy_ok, threshold_policy_detail = _coverage_threshold_policy_ok(line_min_percent, branch_min_percent)
    critical_threshold_policy_ok, critical_threshold_policy_detail = _critical_coverage_threshold_policy_ok(critical_line_min_percent, critical_branch_min_percent)

    critical_under_line_targets: List[str] = []
    critical_under_branch_targets: List[str] = []
    critical_file_coverage: Dict[str, Dict[str, Any]] = {}
    for rel_path in CRITICAL_COVERAGE_TARGETS:
        summary = dict(key_files.get(rel_path, {}))
        line_percent = _coverage_percent_from_summary(summary)
        branch_percent = _coverage_percent_from_summary(summary, branch=True)
        critical_file_coverage[rel_path] = {
            "present": bool(summary.get("present")),
            "line_percent_covered": line_percent,
            "branch_percent_covered": branch_percent,
            "covered_lines": int(summary.get("covered_lines", 0) or 0),
            "num_statements": int(summary.get("num_statements", 0) or 0),
            "covered_branches": int(summary.get("covered_branches", 0) or 0),
            "num_branches": int(summary.get("num_branches", 0) or 0),
        }
        if line_percent < critical_line_min_percent:
            critical_under_line_targets.append(rel_path)
        if branch_percent < critical_branch_min_percent:
            critical_under_branch_targets.append(rel_path)

    coverage_json_detail = run_result.get("stderr", "").splitlines()[0] if run_result.get("stderr") else str(json_file)
    if coverage_reuse_error:
        coverage_json_detail = f"{coverage_json_detail} | {coverage_reuse_error}" if coverage_json_detail else coverage_reuse_error

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
            detail=coverage_json_detail,
            extra={"json_file": str(json_file), "coverage_reuse_error": coverage_reuse_error},
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
            "coverage_thresholds_respect_formal_floor",
            threshold_policy_ok,
            detail=threshold_policy_detail,
            extra={
                "line_min_percent": line_min_percent,
                "branch_min_percent": branch_min_percent,
                "line_floor": COVERAGE_LINE_MIN_FLOOR,
                "branch_floor": COVERAGE_BRANCH_MIN_FLOOR,
                "max_gap": COVERAGE_MAX_LINE_BRANCH_GAP,
            },
        ),
        summarize_result(
            "coverage_line_percent_within_minimum",
            line_percent_covered >= line_min_percent,
            detail=f"line_percent={line_percent_covered:.2f} | min={line_min_percent:.2f}",
            extra={"line_percent_covered": line_percent_covered, "line_min_percent": line_min_percent},
        ),
        summarize_result(
            "coverage_branch_percent_within_minimum",
            branch_percent_covered >= branch_min_percent,
            detail=f"branch_percent={branch_percent_covered:.2f} | min={branch_min_percent:.2f}",
            extra={"branch_percent_covered": branch_percent_covered, "branch_min_percent": branch_min_percent},
        ),
        summarize_result(
            "coverage_core_trading_targets_declared",
            not core_target_missing,
            detail=f"missing_core_targets={core_target_missing}",
            extra={"missing_core_targets": core_target_missing},
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
        summarize_result(
            "coverage_critical_thresholds_respect_formal_floor",
            critical_threshold_policy_ok,
            detail=critical_threshold_policy_detail,
            extra={
                "critical_line_min_percent": critical_line_min_percent,
                "critical_branch_min_percent": critical_branch_min_percent,
                "critical_line_floor": CRITICAL_COVERAGE_LINE_MIN_FLOOR,
                "critical_branch_floor": CRITICAL_COVERAGE_BRANCH_MIN_FLOOR,
                "max_gap": COVERAGE_MAX_LINE_BRANCH_GAP,
            },
        ),
        summarize_result(
            "coverage_critical_files_line_percent_within_minimum",
            not critical_under_line_targets,
            detail=f"critical_line_under={critical_under_line_targets} | min={critical_line_min_percent:.2f}",
            extra={
                "critical_under_line_targets": critical_under_line_targets,
                "critical_line_min_percent": critical_line_min_percent,
            },
        ),
        summarize_result(
            "coverage_critical_files_branch_percent_within_minimum",
            not critical_under_branch_targets,
            detail=f"critical_branch_under={critical_under_branch_targets} | min={critical_branch_min_percent:.2f}",
            extra={
                "critical_under_branch_targets": critical_under_branch_targets,
                "critical_branch_min_percent": critical_branch_min_percent,
            },
        ),
    ]

    ok = all(item["status"] == "PASS" for item in results)
    return {
        "ok": ok,
        "results": results,
        "run_result": run_result,
        "json_file": str(json_file),
        "coverage_reuse_error": coverage_reuse_error,
        "reused_existing": reused_existing,
        "totals": {
            "covered_lines": int(totals.get("covered_lines", 0) or 0),
            "num_statements": int(totals.get("num_statements", 0) or 0),
            "covered_branches": int(totals.get("covered_branches", 0) or 0),
            "num_branches": int(totals.get("num_branches", 0) or 0),
            "percent_covered": float(totals.get("percent_covered", 0.0) or 0.0),
            "line_percent_covered": line_percent_covered,
            "branch_percent_covered": branch_percent_covered,
            "line_min_percent": line_min_percent,
            "branch_min_percent": branch_min_percent,
            "critical_line_min_percent": critical_line_min_percent,
            "critical_branch_min_percent": critical_branch_min_percent,
            "synthetic_fail_count": synthetic_fail_count,
        },
        "key_files": key_files,
        "critical_file_coverage": critical_file_coverage,
        "missing_core_targets": core_target_missing,
        "missing_targets": missing_targets,
        "zero_covered_targets": zero_covered_targets,
        "critical_under_line_targets": critical_under_line_targets,
        "critical_under_branch_targets": critical_under_branch_targets,
    }


__all__ = [
    "build_coverage_summary",
    "_ratio_percent",
    "_coverage_percent_from_summary",
    "_coverage_threshold_policy_ok",
    "_critical_coverage_threshold_policy_ok",
]
