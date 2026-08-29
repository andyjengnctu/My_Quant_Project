from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

DEFAULT_PERFORMANCE_STEP_FILES = {
    "quick_gate": ("quick_gate_summary.json",),
    "consistency": ("validate_consistency_summary.json",),
    "chain_checks": ("chain_summary.json", "chain_checks_summary.json"),
    "ml_smoke": ("ml_smoke_summary.json",),
}
from tools.local_regression.common import LOCAL_REGRESSION_RUN_DIR_ENV, summarize_blocked_result, summarize_result


def build_performance_summary(
    run_dir: Path,
    manifest: Dict[str, Any],
    *,
    current_meta_quality_duration_sec: float,
    current_meta_quality_peak_process_memory_mb: float,
    performance_step_files: Mapping[str, Sequence[str]] | None = None,
) -> Dict[str, Any]:
    if performance_step_files is None:
        performance_step_files = DEFAULT_PERFORMANCE_STEP_FILES
    has_shared_run_dir = bool(os.environ.get(LOCAL_REGRESSION_RUN_DIR_ENV, "").strip())
    available_step_files = {
        name: next((run_dir / file_name for file_name in file_names if (run_dir / file_name).exists()), run_dir / file_names[0])
        for name, file_names in performance_step_files.items()
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
            "step_peak_process_memory_mb": {},
            "optimizer_trial_avg_objective_wall_sec": None,
            "optimizer_profile_trial_count": 0,
            "total_duration_sec": None,
            "critical_path_duration_sec": None,
            "aggregate_step_duration_sec": None,
            "formal_wall_time_source": None,
            "meta_quality_internal_duration_sec": round(float(current_meta_quality_duration_sec), 3),
            "max_step_peak_process_memory_mb": round(float(current_meta_quality_peak_process_memory_mb), 3),
            "meta_quality_peak_process_memory_mb": round(float(current_meta_quality_peak_process_memory_mb), 3),
        }

    results: List[Dict[str, Any]] = []
    step_peak_process_memory_mb: Dict[str, float] = {}
    missing_step_files: List[str] = []
    missing_memory_steps: List[str] = []
    consistency_synthetic_slowest_cases: List[Dict[str, Any]] = []
    consistency_synthetic_timing_present = False
    for step_name, path in available_step_files.items():
        if not path.exists():
            missing_step_files.append(step_name)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        peak_memory_mb = payload.get("peak_process_memory_mb")
        if peak_memory_mb in (None, ""):
            missing_memory_steps.append(step_name)
        else:
            step_peak_process_memory_mb[step_name] = round(float(peak_memory_mb), 3)
        if step_name == "consistency":
            raw_slowest = payload.get("synthetic_slowest_cases")
            if isinstance(raw_slowest, list):
                consistency_synthetic_timing_present = True
                consistency_synthetic_slowest_cases = [
                    dict(row) for row in raw_slowest if isinstance(row, dict)
                ]

    if missing_step_files:
        results.append(
            summarize_blocked_result(
                "performance_required_step_summaries_present",
                blocked_by="formal_step_summary_unavailable",
                detail=f"missing={missing_step_files}",
                extra={"missing_step_files": missing_step_files},
            )
        )
    else:
        results.append(
            summarize_result(
                "performance_required_step_summaries_present",
                True,
                detail="missing=[]",
                extra={"missing_step_files": []},
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

    if "consistency" not in missing_step_files:
        synthetic_case_budget_sec = float(manifest["performance_synthetic_case_max_sec"])
        if not consistency_synthetic_timing_present:
            results.append(
                summarize_result(
                    "performance_consistency_synthetic_case_timings_present",
                    False,
                    detail="validate_consistency_summary.json 缺少 synthetic_slowest_cases",
                )
            )
        else:
            slowest_case = max(
                consistency_synthetic_slowest_cases,
                key=lambda row: float(row.get("duration_sec", 0.0) or 0.0),
                default={},
            )
            slowest_duration_sec = round(float(slowest_case.get("duration_sec", 0.0) or 0.0), 6)
            slowest_validator_name = str(slowest_case.get("validator_name") or "")
            results.append(
                summarize_result(
                    "performance_consistency_synthetic_case_within_budget",
                    slowest_duration_sec <= synthetic_case_budget_sec,
                    detail=(
                        f"slowest={slowest_validator_name or '-'}:{slowest_duration_sec:.3f}s"
                        f" | budget={synthetic_case_budget_sec:.3f}s"
                    ),
                    extra={
                        "validator_name": slowest_validator_name,
                        "duration_sec": slowest_duration_sec,
                        "budget_sec": synthetic_case_budget_sec,
                    },
                )
            )

    profile_file = run_dir / "optimizer_profile_summary.json"
    optimizer_trial_avg_objective_wall_sec = None
    optimizer_profile_trial_count = 0
    if profile_file.exists():
        try:
            profile_payload = json.loads(profile_file.read_text(encoding="utf-8"))
        except Exception as exc:
            results.append(
                summarize_result(
                    "performance_optimizer_profile_readable",
                    False,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
        else:
            avg_payload = profile_payload.get("avg", {}) if isinstance(profile_payload, dict) else {}
            optimizer_profile_trial_count = int(profile_payload.get("trial_count", 0) or 0)
            raw_avg = avg_payload.get("objective_wall_sec")
            if raw_avg not in (None, ""):
                optimizer_trial_avg_objective_wall_sec = round(float(raw_avg), 3)
            results.append(
                summarize_result(
                    "performance_optimizer_profile_readable",
                    True,
                    detail=f"trial_count={optimizer_profile_trial_count}",
                )
            )
            if optimizer_trial_avg_objective_wall_sec is not None:
                optimizer_budget_sec = float(manifest["performance_optimizer_trial_avg_max_sec"])
                results.append(
                    summarize_result(
                        "performance_optimizer_trial_avg_within_budget",
                        optimizer_trial_avg_objective_wall_sec <= optimizer_budget_sec,
                        detail=f"duration={optimizer_trial_avg_objective_wall_sec:.3f}s | budget={optimizer_budget_sec:.3f}s",
                        extra={"duration_sec": optimizer_trial_avg_objective_wall_sec, "budget_sec": optimizer_budget_sec},
                    )
                )

    max_step_peak_memory_mb = round(
        max([float(current_meta_quality_peak_process_memory_mb)] + [float(value) for value in step_peak_process_memory_mb.values()]),
        3,
    )
    return {
        "ok": all(item["status"] == "PASS" for item in results),
        "skipped": False,
        "results": results,
        "step_durations": {},
        "step_peak_process_memory_mb": step_peak_process_memory_mb,
        "consistency_synthetic_slowest_cases": consistency_synthetic_slowest_cases,
        "optimizer_trial_avg_objective_wall_sec": optimizer_trial_avg_objective_wall_sec,
        "optimizer_profile_trial_count": optimizer_profile_trial_count,
        "total_duration_sec": None,
        "critical_path_duration_sec": None,
        "aggregate_step_duration_sec": None,
        "formal_wall_time_source": None,
        "meta_quality_internal_duration_sec": round(float(current_meta_quality_duration_sec), 3),
        "max_step_peak_process_memory_mb": max_step_peak_memory_mb,
        "meta_quality_peak_process_memory_mb": round(float(current_meta_quality_peak_process_memory_mb), 3),
    }
