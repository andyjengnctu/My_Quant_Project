from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

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
STATUS_VALUES = {"DONE", "PARTIAL", "TODO", "N/A"}
COVERAGE_TARGETS = [
    "tools/validate/synthetic_cases.py",
    "tools/validate/synthetic_meta_cases.py",
    "tools/validate/synthetic_unit_cases.py",
    "tools/validate/synthetic_history_cases.py",
    "tools/validate/synthetic_flow_cases.py",
    "tools/validate/synthetic_take_profit_cases.py",
    "tools/validate/synthetic_display_cases.py",
    "core/price_utils.py",
    "core/history_filters.py",
    "core/portfolio_stats.py",
]
REQUIRED_META_IDS = ("B22", "B23", "B24", "B25", "B26")


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
        synthetic_fail_count = sum(1 for row in results if row.get("status") == "FAIL")
        run_result["returncode"] = 0 if synthetic_fail_count == 0 else 1
        run_result["stdout"] = json.dumps({"synthetic_case_count": len(summaries), "synthetic_fail_count": synthetic_fail_count}, ensure_ascii=False)
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


def main(argv=None) -> int:
    cli = parse_no_arg_cli(argv, "tools/local_regression/run_meta_quality.py", description="執行 coverage baseline 與 checklist sufficiency formal check")
    if cli["help"]:
        return 0

    load_manifest()
    ensure_reduced_dataset()
    run_dir = resolve_run_dir("meta_quality")

    coverage_summary = _build_coverage_summary(run_dir)
    checklist_summary = _summarize_checklist_consistency()

    all_results = [*coverage_summary["results"], *checklist_summary["results"]]
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
        "results": all_results,
    }
    write_json(run_dir / "meta_quality_summary.json", summary)

    lines = [
        f"status        : {overall_status}",
        f"fail_count    : {len(failures)}",
        f"coverage_ok   : {coverage_summary['ok']}",
        f"checklist_ok  : {checklist_summary['ok']}",
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
    }, ensure_ascii=False))
    return 0 if overall_status == "PASS" else 1


if __name__ == "__main__":
    run_cli_entrypoint(main)
