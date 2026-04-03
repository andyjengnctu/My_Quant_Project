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

from core.runtime_utils import PeakTracedMemoryTracker, parse_no_arg_cli, run_cli_entrypoint
from tools.local_regression.common import (
    ensure_reduced_dataset,
    load_manifest,
    LOCAL_REGRESSION_RUN_DIR_ENV,
    resolve_run_dir,
    run_command,
    summarize_result,
    write_json,
    write_text,
)

from tools.local_regression.formal_pipeline import (
    DATASET_REQUIRED_STEPS as FORMAL_DATASET_REQUIRED_STEPS,
    FORMAL_COMMAND_ORDER,
    FORMAL_SINGLE_ENTRY,
    FORMAL_STEP_ORDER,
)

CHECKLIST_PATH = PROJECT_ROOT / "doc" / "TEST_SUITE_CHECKLIST.md"
PROJECT_SETTINGS_PATH = PROJECT_ROOT / "doc" / "PROJECT_SETTINGS.md"
CMD_PATH = PROJECT_ROOT / "doc" / "CMD.md"
ARCHITECTURE_PATH = PROJECT_ROOT / "doc" / "ARCHITECTURE.md"
STATUS_VALUES = {"DONE", "PARTIAL", "TODO", "N/A"}
from tools.local_regression.meta_quality_targets import (
    CORE_TRADING_COVERAGE_TARGETS,
    COVERAGE_BRANCH_MIN_FLOOR,
    COVERAGE_LINE_MIN_FLOOR,
    COVERAGE_MAX_LINE_BRANCH_GAP,
    COVERAGE_TARGETS,
    CRITICAL_COVERAGE_BRANCH_MIN_FLOOR,
    CRITICAL_COVERAGE_LINE_MIN_FLOOR,
    CRITICAL_COVERAGE_TARGETS,
    ENTRY_PATH_CRITICAL_COVERAGE_TARGETS,
    TEST_SUITE_ORCHESTRATOR_COVERAGE_TARGETS,
)
from tools.local_regression.meta_quality_coverage import build_coverage_summary as _shared_build_coverage_summary
REQUIRED_META_IDS = ("B22", "B23", "B24", "B25", "B26")
PERFORMANCE_STEP_FILES = {
    "quick_gate": ("quick_gate_summary.json",),
    "consistency": ("validate_consistency_summary.json",),
    "chain_checks": ("chain_summary.json", "chain_checks_summary.json"),
    "ml_smoke": ("ml_smoke_summary.json",),
}
PERFORMANCE_MANIFEST_KEYS = {
    "quick_gate": "performance_quick_gate_max_sec",
    "consistency": "performance_consistency_max_sec",
    "chain_checks": "performance_chain_checks_max_sec",
    "ml_smoke": "performance_ml_smoke_max_sec",
}
PERFORMANCE_MEMORY_MANIFEST_KEY = "performance_peak_traced_memory_mb"



def _load_checklist_tables() -> Dict[str, List[List[str]]]:
    text = CHECKLIST_PATH.read_text(encoding="utf-8")
    return {
        "B1": extract_markdown_table_rows(text, "B1. 專案設定對應清單（不含暫時特例）"),
        "B2": extract_markdown_table_rows(text, "B2. 未明列於專案設定，但正式 test suite 應納入"),
        "E1": extract_markdown_table_rows(text, "E1. 目前所有 `PARTIAL` 的主表項目摘要"),
        "E2": extract_markdown_table_rows(text, "E2. 目前所有 `TODO` 的主表項目摘要"),
        "E3": extract_markdown_table_rows(text, "E3. 目前所有未完成的建議測試項目摘要"),
        "F1": extract_markdown_table_rows(text, "F1. 目前所有 `DONE` 的主表項目摘要"),
        "F2": extract_markdown_table_rows(text, "F2. 目前所有 `DONE` 的建議測試項目摘要"),
        "G": extract_markdown_table_rows(text, "G. 逐項收斂紀錄"),
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


def _sorted_unique(values: List[str]) -> List[str]:
    return sorted(dict.fromkeys(values))


def _latest_statuses_from_convergence_rows(rows: List[List[str]]) -> Dict[str, str]:
    statuses: Dict[str, str] = {}
    for cols in rows:
        if len(cols) <= 3:
            continue
        item_id = cols[1].strip()
        transition = cols[3].strip()
        if not item_id or not transition:
            continue
        statuses[item_id] = transition.split("->")[-1].strip()
    return statuses


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
    e3_ids = sorted(_ids_from_table(tables["E3"], idx=0))
    f1_ids = sorted(_ids_from_table(tables["F1"]))
    f2_rows = tables["F2"]
    f2_ids_raw = _ids_from_table(f2_rows, idx=0)
    f2_ids = _sorted_unique(f2_ids_raw)
    convergence_statuses = _latest_statuses_from_convergence_rows(tables["G"])
    g_done_d_ids = _sorted_unique([item_id for item_id, status in convergence_statuses.items() if item_id.startswith("D") and status == "DONE"])
    g_unfinished_d_ids = sorted(item_id for item_id, status in convergence_statuses.items() if item_id.startswith("D") and status in {"PARTIAL", "TODO", "N/A"})
    g_b_statuses = {item_id: status for item_id, status in convergence_statuses.items() if item_id.startswith("B")}

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
    f2_duplicate_ids = sorted({item_id for item_id in f2_ids_raw if f2_ids_raw.count(item_id) > 1})
    results.append(
        summarize_result(
            "checklist_f2_done_d_ids_unique",
            not f2_duplicate_ids,
            detail=f"duplicates={f2_duplicate_ids}",
            extra={"duplicate_ids": f2_duplicate_ids},
        )
    )

    invalid_f2_entries = []
    for row in f2_rows:
        if len(row) < 2:
            continue
        entry = row[1].replace("`", "").strip()
        is_single_entry = (
            entry.startswith("validate_")
            or entry.endswith(".py")
        ) and (" " not in entry)
        if not is_single_entry:
            invalid_f2_entries.append({"id": row[0] if row else "", "entry": entry})
    results.append(
        summarize_result(
            "checklist_f2_rows_use_single_test_entry",
            not invalid_f2_entries,
            detail=f"invalid={invalid_f2_entries}",
            extra={"invalid_entries": invalid_f2_entries},
        )
    )

    no_op_convergence_rows = []
    for row in tables["G"]:
        if len(row) < 4:
            continue
        transition = row[3].strip()
        if "->" not in transition:
            continue
        from_status, to_status = [part.strip() for part in transition.split("->", 1)]
        if from_status == to_status:
            no_op_convergence_rows.append({"id": row[1].strip(), "transition": transition})
    results.append(
        summarize_result(
            "checklist_g_rows_require_actual_status_change",
            not no_op_convergence_rows,
            detail=f"no_op={no_op_convergence_rows}",
            extra={"no_op_rows": no_op_convergence_rows},
        )
    )

    results.append(
        summarize_result(
            "checklist_done_d_summary_matches_convergence_done_records",
            f2_ids == g_done_d_ids,
            detail=f"f2_unique={f2_ids} | f2_raw={sorted(f2_ids_raw)} | g_done={g_done_d_ids}",
            extra={"f2_ids": f2_ids, "f2_ids_raw": sorted(f2_ids_raw), "g_done_d_ids": g_done_d_ids},
        )
    )
    results.append(
        summarize_result(
            "checklist_unfinished_d_summary_matches_convergence_unfinished_records",
            e3_ids == g_unfinished_d_ids,
            detail=f"e3={e3_ids} | g_unfinished={g_unfinished_d_ids}",
            extra={"e3_ids": e3_ids, "g_unfinished_d_ids": g_unfinished_d_ids},
        )
    )

    done_d_missing_from_f2 = sorted(set(g_done_d_ids) - set(f2_ids))
    done_d_missing_from_g = sorted(set(f2_ids) - set(g_done_d_ids))
    results.append(
        summarize_result(
            "checklist_done_d_summary_has_no_missing_done_records",
            not done_d_missing_from_f2,
            detail=f"missing_from_f2={done_d_missing_from_f2}",
            extra={"missing_from_f2": done_d_missing_from_f2},
        )
    )
    results.append(
        summarize_result(
            "checklist_done_d_summary_all_have_convergence_records",
            not done_d_missing_from_g,
            detail=f"missing_from_g={done_d_missing_from_g}",
            extra={"missing_from_g": done_d_missing_from_g},
        )
    )

    mismatched_b_statuses = {
        item_id: {"main": main_statuses.get(item_id, ""), "g": status}
        for item_id, status in g_b_statuses.items()
        if main_statuses.get(item_id, "") != status
    }
    results.append(
        summarize_result(
            "checklist_main_table_matches_convergence_b_statuses",
            not mismatched_b_statuses,
            detail=f"mismatch={sorted(mismatched_b_statuses.items())}",
            extra={"mismatched_b_statuses": mismatched_b_statuses},
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
        "done_d_ids": f2_ids,
        "g_done_d_ids": g_done_d_ids,
        "g_unfinished_d_ids": g_unfinished_d_ids,
    }


def _extract_backticked_paths(text: str) -> List[str]:
    return [match.strip() for match in re.findall(r"`([^`]+)`", text) if "/" in match or match.endswith('.py')]


from tools.validate.meta_contracts import (
    extract_markdown_table_rows,
    summarize_single_formal_test_entry_contract,
    summarize_test_suite_registry_contract,
)


def _project_settings_declares_formal_pipeline_principles() -> Dict[str, Any]:
    text = PROJECT_SETTINGS_PATH.read_text(encoding="utf-8")
    results: List[Dict[str, Any]] = []

    has_single_entry = "`apps/test_suite.py` 為所有已實作測試的單一正式入口" in text
    results.append(
        summarize_result(
            "formal_entry_project_settings_declares_single_entry",
            has_single_entry,
            detail="expect=`apps/test_suite.py` 為所有已實作測試的單一正式入口",
        )
    )

    has_registry_source = "`tools/local_regression/formal_pipeline.py` 為單一真理來源" in text
    results.append(
        summarize_result(
            "formal_entry_project_settings_declares_registry_source",
            has_registry_source,
            detail="expect=`tools/local_regression/formal_pipeline.py` 為單一真理來源",
        )
    )

    return {
        "ok": all(item["status"] == "PASS" for item in results),
        "results": results,
    }


def _summarize_formal_entry_consistency() -> Dict[str, Any]:
    from tools.local_regression.run_all import DATASET_REQUIRED_STEPS, SCRIPT_ORDER, STEP_NAMES
    from tools.validate.preflight_env import _LOCAL_REGRESSION_STEP_ORDER

    results: List[Dict[str, Any]] = []

    registry_commands = list(FORMAL_COMMAND_ORDER)
    run_all_commands = list(SCRIPT_ORDER)
    test_suite_contract = summarize_test_suite_registry_contract(PROJECT_ROOT, FORMAL_STEP_ORDER)

    results.append(
        summarize_result(
            "formal_entry_run_all_commands_match_registry",
            run_all_commands == registry_commands,
            detail=f"run_all={run_all_commands} | registry={registry_commands}",
            extra={"run_all_commands": run_all_commands, "registry_commands": registry_commands},
        )
    )

    results.append(
        summarize_result(
            "formal_entry_run_all_steps_match_registry",
            list(STEP_NAMES) == list(FORMAL_STEP_ORDER),
            detail=f"run_all={list(STEP_NAMES)} | registry={list(FORMAL_STEP_ORDER)}",
            extra={"run_all_steps": list(STEP_NAMES), "registry_steps": list(FORMAL_STEP_ORDER)},
        )
    )

    results.append(
        summarize_result(
            "formal_entry_preflight_steps_match_registry",
            list(_LOCAL_REGRESSION_STEP_ORDER) == list(FORMAL_STEP_ORDER),
            detail=f"preflight={list(_LOCAL_REGRESSION_STEP_ORDER)} | registry={list(FORMAL_STEP_ORDER)}",
            extra={"preflight_steps": list(_LOCAL_REGRESSION_STEP_ORDER), "registry_steps": list(FORMAL_STEP_ORDER)},
        )
    )

    results.append(
        summarize_result(
            "formal_entry_test_suite_steps_match_registry",
            test_suite_contract["regression_step_order_aliases_registry"],
            detail=f"aliases_registry={test_suite_contract['regression_step_order_aliases_registry']}",
            extra={
                "test_suite_step_order_aliases_registry": test_suite_contract["regression_step_order_aliases_registry"],
                "registry_steps": list(FORMAL_STEP_ORDER),
            },
        )
    )

    results.append(
        summarize_result(
            "formal_entry_dataset_required_steps_match_registry",
            sorted(DATASET_REQUIRED_STEPS) == sorted(FORMAL_DATASET_REQUIRED_STEPS),
            detail=f"run_all={sorted(DATASET_REQUIRED_STEPS)} | registry={sorted(FORMAL_DATASET_REQUIRED_STEPS)}",
            extra={
                "run_all_dataset_required_steps": sorted(DATASET_REQUIRED_STEPS),
                "registry_dataset_required_steps": sorted(FORMAL_DATASET_REQUIRED_STEPS),
            },
        )
    )

    missing_step_labels = test_suite_contract["missing_step_labels"]
    results.append(
        summarize_result(
            "formal_entry_test_suite_labels_cover_registry_steps",
            not missing_step_labels,
            detail=f"missing={missing_step_labels}",
            extra={
                "missing_step_labels": missing_step_labels,
                "test_suite_step_labels": test_suite_contract["step_labels_keys"],
            },
        )
    )

    missing_extra_labels = test_suite_contract["missing_extra_labels"]
    results.append(
        summarize_result(
            "formal_entry_test_suite_labels_cover_non_script_stages",
            not missing_extra_labels,
            detail=f"missing={missing_extra_labels}",
            extra={
                "missing_extra_labels": missing_extra_labels,
                "test_suite_step_labels": test_suite_contract["step_labels_keys"],
            },
        )
    )

    missing_script_files = []
    for _name, command, _summary in FORMAL_COMMAND_ORDER:
        script_path = PROJECT_ROOT / command.split()[0]
        if not script_path.exists():
            missing_script_files.append(command)
    results.append(
        summarize_result(
            "formal_entry_registry_commands_exist",
            not missing_script_files,
            detail=f"missing={missing_script_files}",
            extra={"missing_script_files": missing_script_files},
        )
    )

    principle_contract = _project_settings_declares_formal_pipeline_principles()
    results.extend(principle_contract["results"])

    single_entry_contract = summarize_single_formal_test_entry_contract(PROJECT_ROOT)
    results.append(
        summarize_result(
            "formal_entry_test_suite_file_exists",
            single_entry_contract["test_suite_exists"],
            detail=f"single_entry={FORMAL_SINGLE_ENTRY} | apps={single_entry_contract['app_py_files']}",
            extra={"app_py_files": single_entry_contract["app_py_files"]},
        )
    )
    results.append(
        summarize_result(
            "formal_entry_cmd_declares_single_entry",
            single_entry_contract["cmd_declares_single_entry"],
            detail=f"declared={single_entry_contract['cmd_declares_single_entry']}",
        )
    )
    results.append(
        summarize_result(
            "formal_entry_architecture_declares_single_entry",
            single_entry_contract["architecture_declares_single_entry"],
            detail=f"declared={single_entry_contract['architecture_declares_single_entry']}",
        )
    )
    results.append(
        summarize_result(
            "formal_entry_no_legacy_app_test_entries",
            not single_entry_contract["legacy_entry_paths"],
            detail=f"legacy={single_entry_contract['legacy_entry_paths']}",
            extra={"legacy_entry_paths": single_entry_contract["legacy_entry_paths"]},
        )
    )
    results.append(
        summarize_result(
            "formal_entry_no_suspicious_alternate_app_test_entries",
            not single_entry_contract["suspicious_app_entries"],
            detail=f"suspicious={single_entry_contract['suspicious_app_entries']}",
            extra={"suspicious_app_entries": single_entry_contract["suspicious_app_entries"]},
        )
    )

    ok = all(item["status"] == "PASS" for item in results)
    return {
        "ok": ok,
        "results": results,
        "registry_steps": list(FORMAL_STEP_ORDER),
        "registry_commands": registry_commands,
        "run_all_steps": list(STEP_NAMES),
        "preflight_steps": list(_LOCAL_REGRESSION_STEP_ORDER),
        "test_suite_steps": list(FORMAL_STEP_ORDER),
    }



def _build_coverage_summary(run_dir: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    from tools.validate.synthetic_cases import run_synthetic_consistency_suite

    return _shared_build_coverage_summary(
        run_dir,
        manifest,
        suite_runner=run_synthetic_consistency_suite,
    )


def _build_performance_summary(run_dir: Path, manifest: Dict[str, Any], *, current_meta_quality_duration_sec: float, current_meta_quality_peak_traced_memory_mb: float) -> Dict[str, Any]:
    has_shared_run_dir = bool(os.environ.get(LOCAL_REGRESSION_RUN_DIR_ENV, "").strip())
    available_step_files = {
        name: next((run_dir / file_name for file_name in file_names if (run_dir / file_name).exists()), run_dir / file_names[0])
        for name, file_names in PERFORMANCE_STEP_FILES.items()
    }
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
            "step_peak_traced_memory_mb": {},
            "optimizer_trial_avg_objective_wall_sec": None,
            "optimizer_profile_trial_count": 0,
            "total_duration_sec": current_meta_quality_duration_sec,
            "max_step_peak_traced_memory_mb": round(float(current_meta_quality_peak_traced_memory_mb), 3),
            "meta_quality_peak_traced_memory_mb": round(float(current_meta_quality_peak_traced_memory_mb), 3),
        }

    results: List[Dict[str, Any]] = []
    step_durations: Dict[str, float] = {}
    step_peak_traced_memory_mb: Dict[str, float] = {}
    missing_step_files: List[str] = []
    missing_memory_steps: List[str] = []
    for step_name, path in available_step_files.items():
        if not path.exists():
            missing_step_files.append(step_name)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        duration_value = payload.get("duration_sec")
        if duration_value in (None, ""):
            duration_value = payload.get("elapsed_time_sec")
        if duration_value in (None, ""):
            duration_value = payload.get("duration_seconds")
        step_durations[step_name] = round(float(duration_value or 0.0), 3)
        peak_memory_mb = payload.get("peak_traced_memory_mb")
        if peak_memory_mb in (None, ""):
            missing_memory_steps.append(step_name)
        else:
            step_peak_traced_memory_mb[step_name] = round(float(peak_memory_mb), 3)

    results.append(
        summarize_result(
            "performance_required_step_summaries_present",
            not missing_step_files,
            detail=f"missing={missing_step_files}",
            extra={"missing_step_files": missing_step_files},
        )
    )
    results.append(
        summarize_result(
            "performance_required_step_peak_memory_present",
            not missing_memory_steps,
            detail=f"missing={missing_memory_steps}",
            extra={"missing_memory_steps": missing_memory_steps},
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

    peak_memory_budget_mb = float(manifest[PERFORMANCE_MEMORY_MANIFEST_KEY])
    for step_name, peak_memory_mb in step_peak_traced_memory_mb.items():
        results.append(
            summarize_result(
                f"performance_{step_name}_peak_memory_within_budget",
                peak_memory_mb <= peak_memory_budget_mb,
                detail=f"peak_memory={peak_memory_mb:.3f}MB | budget={peak_memory_budget_mb:.3f}MB",
                extra={"peak_traced_memory_mb": peak_memory_mb, "budget_mb": peak_memory_budget_mb},
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
    results.append(
        summarize_result(
            "performance_meta_quality_peak_memory_within_budget",
            current_meta_quality_peak_traced_memory_mb <= peak_memory_budget_mb,
            detail=f"peak_memory={current_meta_quality_peak_traced_memory_mb:.3f}MB | budget={peak_memory_budget_mb:.3f}MB",
            extra={"peak_traced_memory_mb": current_meta_quality_peak_traced_memory_mb, "budget_mb": peak_memory_budget_mb},
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

    max_step_peak_memory_mb = round(max([float(current_meta_quality_peak_traced_memory_mb), *step_peak_traced_memory_mb.values()]) if step_peak_traced_memory_mb else float(current_meta_quality_peak_traced_memory_mb), 3)
    ok = all(item["status"] == "PASS" for item in results)
    return {
        "ok": ok,
        "skipped": False,
        "results": results,
        "step_durations": step_durations,
        "step_peak_traced_memory_mb": step_peak_traced_memory_mb,
        "optimizer_trial_avg_objective_wall_sec": optimizer_trial_avg_value,
        "optimizer_profile_trial_count": optimizer_profile_trial_count,
        "total_duration_sec": total_duration_sec,
        "max_step_peak_traced_memory_mb": max_step_peak_memory_mb,
        "meta_quality_peak_traced_memory_mb": round(float(current_meta_quality_peak_traced_memory_mb), 3),
    }


def main(argv=None) -> int:
    cli = parse_no_arg_cli(argv, "tools/local_regression/run_meta_quality.py", description="執行 coverage baseline 與 checklist sufficiency formal check")
    if cli["help"]:
        return 0

    tracker = PeakTracedMemoryTracker()
    tracker.__enter__()
    manifest = load_manifest()
    ensure_reduced_dataset()
    run_dir = resolve_run_dir("meta_quality")

    started = os.times().elapsed
    coverage_summary = _build_coverage_summary(run_dir, manifest)
    checklist_summary = _summarize_checklist_consistency()
    formal_entry_summary = _summarize_formal_entry_consistency()
    current_meta_quality_duration_sec = round(os.times().elapsed - started, 3)
    current_meta_quality_peak_traced_memory_mb = tracker.snapshot_peak_mb()
    performance_summary = _build_performance_summary(
        run_dir,
        manifest,
        current_meta_quality_duration_sec=current_meta_quality_duration_sec,
        current_meta_quality_peak_traced_memory_mb=current_meta_quality_peak_traced_memory_mb,
    )

    all_results = [*coverage_summary["results"], *checklist_summary["results"], *formal_entry_summary["results"], *performance_summary["results"]]
    failures = [item["name"] for item in all_results if item["status"] != "PASS"]
    overall_status = "PASS" if not failures else "FAIL"

    summary = {
        "status": overall_status,
        "failures": failures,
        "fail_count": len(failures),
        "coverage": {
            "ok": coverage_summary["ok"],
            "status": "DONE" if coverage_summary["ok"] else "PARTIAL",
            "totals": coverage_summary["totals"],
            "json_file": coverage_summary["json_file"],
            "line_percent_covered": coverage_summary["totals"]["line_percent_covered"],
            "branch_percent_covered": coverage_summary["totals"]["branch_percent_covered"],
            "line_min_percent": coverage_summary["totals"]["line_min_percent"],
            "branch_min_percent": coverage_summary["totals"]["branch_min_percent"],
            "critical_line_min_percent": coverage_summary["totals"]["critical_line_min_percent"],
            "critical_branch_min_percent": coverage_summary["totals"]["critical_branch_min_percent"],
            "missing_core_targets": coverage_summary["missing_core_targets"],
            "missing_targets": coverage_summary["missing_targets"],
            "zero_covered_targets": coverage_summary["zero_covered_targets"],
            "critical_under_line_targets": coverage_summary["critical_under_line_targets"],
            "critical_under_branch_targets": coverage_summary["critical_under_branch_targets"],
        },
        "checklist": {
            "ok": checklist_summary["ok"],
            "status": "TODO" if checklist_summary["todo_ids"] else ("PARTIAL" if checklist_summary["partial_ids"] else "DONE"),
            "partial_ids": checklist_summary["partial_ids"],
            "todo_ids": checklist_summary["todo_ids"],
            "done_ids": checklist_summary["done_ids"],
            "unfinished_d_ids": checklist_summary["unfinished_d_ids"],
        },
        "formal_entry": {
            "ok": formal_entry_summary["ok"],
            "registry_steps": formal_entry_summary["registry_steps"],
            "registry_commands": formal_entry_summary["registry_commands"],
            "run_all_steps": formal_entry_summary["run_all_steps"],
            "preflight_steps": formal_entry_summary["preflight_steps"],
            "test_suite_steps": formal_entry_summary["test_suite_steps"],
        },
        "performance": {
            "ok": performance_summary["ok"],
            "skipped": performance_summary["skipped"],
            "step_durations": performance_summary["step_durations"],
            "optimizer_profile_trial_count": performance_summary["optimizer_profile_trial_count"],
            "optimizer_trial_avg_objective_wall_sec": performance_summary["optimizer_trial_avg_objective_wall_sec"],
            "total_duration_sec": performance_summary["total_duration_sec"],
            "step_peak_traced_memory_mb": performance_summary["step_peak_traced_memory_mb"],
            "max_step_peak_traced_memory_mb": performance_summary["max_step_peak_traced_memory_mb"],
            "meta_quality_peak_traced_memory_mb": performance_summary["meta_quality_peak_traced_memory_mb"],
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
            f"{coverage_summary['totals']['num_statements']} ({coverage_summary['totals']['line_percent_covered']:.2f}%)"
            f" | branch {coverage_summary['totals']['covered_branches']}/"
            f"{coverage_summary['totals']['num_branches']} ({coverage_summary['totals']['branch_percent_covered']:.2f}%)"
            f" | total={coverage_summary['totals']['percent_covered']:.2f}%"
        ),
        f"missing_cov   : {', '.join(coverage_summary['missing_targets']) if coverage_summary['missing_targets'] else '(none)'}",
        f"zero_cov      : {', '.join(coverage_summary['zero_covered_targets']) if coverage_summary['zero_covered_targets'] else '(none)'}",
        f"partial_ids   : {', '.join(checklist_summary['partial_ids']) if checklist_summary['partial_ids'] else '(none)'}",
        f"todo_ids      : {', '.join(checklist_summary['todo_ids']) if checklist_summary['todo_ids'] else '(none)'}",
        f"done_ids      : {', '.join(checklist_summary['done_ids']) if checklist_summary['done_ids'] else '(none)'}",
        f"performance_ok: {performance_summary['ok']}",
        f"perf_total    : {performance_summary['total_duration_sec']:.3f}s",
        f"perf_peak_mem : {performance_summary['max_step_peak_traced_memory_mb']:.3f}MB",
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
        "performance_peak_traced_memory_mb": performance_summary["max_step_peak_traced_memory_mb"],
        "optimizer_trial_avg_objective_wall_sec": performance_summary["optimizer_trial_avg_objective_wall_sec"],
    }, ensure_ascii=False))
    tracker.__exit__(None, None, None)
    return 0 if overall_status == "PASS" else 1


if __name__ == "__main__":
    run_cli_entrypoint(main)
