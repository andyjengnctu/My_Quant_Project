from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from tools.local_regression.common import (
    partition_result_statuses,
    read_json_if_exists,
    summarize_result,
    write_json,
    write_text,
)

FORMAL_WALL_TIME_MANIFEST_KEYS = {
    "quick_gate": "performance_quick_gate_max_sec",
    "consistency": "performance_consistency_max_sec",
    "chain_checks": "performance_chain_checks_max_sec",
    "ml_smoke": "performance_ml_smoke_max_sec",
    "meta_quality": "performance_meta_quality_max_sec",
}
FORMAL_WALL_TIME_SOURCE = "run_all_parent_subprocess_duration_sec"


def build_parent_wall_time_summary(
    *,
    script_summaries_by_name: Dict[str, Dict[str, Any]],
    selected_step_names: List[str],
    parallel_step_names: Iterable[str],
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    parallel_steps = set(parallel_step_names)
    selected_formal_steps = [name for name in selected_step_names if name in FORMAL_WALL_TIME_MANIFEST_KEYS]
    missing = [name for name in selected_formal_steps if name not in script_summaries_by_name]
    step_durations = {
        name: round(float(script_summaries_by_name[name].get("duration_sec", 0.0) or 0.0), 3)
        for name in selected_formal_steps
        if name in script_summaries_by_name
    }
    results: List[Dict[str, Any]] = [
        summarize_result(
            "performance_parent_wall_timings_present",
            not missing,
            detail=f"source={FORMAL_WALL_TIME_SOURCE} | missing={missing}",
            extra={"source": FORMAL_WALL_TIME_SOURCE, "missing_steps": missing},
        )
    ]
    failed_budget_steps: List[str] = []
    for step_name, duration_sec in step_durations.items():
        budget_sec = float(manifest[FORMAL_WALL_TIME_MANIFEST_KEYS[step_name]])
        within_budget = duration_sec <= budget_sec
        if not within_budget:
            failed_budget_steps.append(step_name)
        results.append(
            summarize_result(
                f"performance_{step_name}_within_budget",
                within_budget,
                detail=(
                    f"parent_wall={duration_sec:.3f}s | budget={budget_sec:.3f}s"
                    f" | source={FORMAL_WALL_TIME_SOURCE}"
                ),
                extra={
                    "duration_sec": duration_sec,
                    "budget_sec": budget_sec,
                    "source": FORMAL_WALL_TIME_SOURCE,
                },
            )
        )

    parallel_duration_sec = max(
        [step_durations[name] for name in selected_formal_steps if name in parallel_steps and name in step_durations],
        default=0.0,
    )
    serial_duration_sec = sum(
        step_durations[name]
        for name in selected_formal_steps
        if name not in parallel_steps and name in step_durations
    )
    critical_path_duration_sec = round(parallel_duration_sec + serial_duration_sec, 3)
    aggregate_step_duration_sec = round(sum(step_durations.values()), 3)
    critical_path_budget_sec = float(manifest["performance_critical_path_max_sec"])
    critical_path_ok = not missing and critical_path_duration_sec <= critical_path_budget_sec
    results.append(
        summarize_result(
            "performance_formal_critical_path_within_budget",
            critical_path_ok,
            detail=(
                f"critical_path={critical_path_duration_sec:.3f}s | budget={critical_path_budget_sec:.3f}s"
                f" | source={FORMAL_WALL_TIME_SOURCE}"
            ),
            extra={
                "duration_sec": critical_path_duration_sec,
                "budget_sec": critical_path_budget_sec,
                "parallel_duration_sec": round(parallel_duration_sec, 3),
                "serial_duration_sec": round(serial_duration_sec, 3),
                "source": FORMAL_WALL_TIME_SOURCE,
            },
        )
    )
    return {
        "ok": all(row.get("status") == "PASS" for row in results),
        "source": FORMAL_WALL_TIME_SOURCE,
        "results": results,
        "step_durations": step_durations,
        "failed_budget_steps": failed_budget_steps,
        "critical_path_duration_sec": critical_path_duration_sec,
        "aggregate_step_duration_sec": aggregate_step_duration_sec,
    }


def merge_parent_wall_time_into_meta_quality_summary(run_dir: Path, wall_summary: Dict[str, Any]) -> None:
    summary_path = run_dir / "meta_quality_summary.json"
    summary = read_json_if_exists(summary_path)
    if not summary:
        return

    wall_result_names = {str(row.get("name") or "") for row in wall_summary.get("results", [])}
    existing_results = [
        row
        for row in summary.get("results", [])
        if isinstance(row, dict) and str(row.get("name") or "") not in wall_result_names
    ]
    merged_results = [*existing_results, *wall_summary.get("results", [])]
    statuses = partition_result_statuses(merged_results)
    failures = statuses["failures"]
    blocked = statuses["blocked"]
    non_pass = statuses["non_pass"]

    performance = dict(summary.get("performance", {}))
    performance.update(
        {
            "ok": bool(performance.get("ok", True)) and bool(wall_summary.get("ok", False)),
            "step_durations": dict(wall_summary.get("step_durations", {})),
            "total_duration_sec": wall_summary.get("aggregate_step_duration_sec"),
            "critical_path_duration_sec": wall_summary.get("critical_path_duration_sec"),
            "aggregate_step_duration_sec": wall_summary.get("aggregate_step_duration_sec"),
            "formal_wall_time_source": wall_summary.get("source"),
        }
    )
    summary.update(
        {
            "status": "PASS" if not non_pass else "FAIL",
            "failures": failures,
            "fail_count": len(failures),
            "blocked": blocked,
            "blocked_count": len(blocked),
            "performance": performance,
            "results": merged_results,
        }
    )
    write_json(summary_path, summary)

    text_path = run_dir / "meta_quality_summary.txt"
    if not text_path.exists():
        return

    lines = text_path.read_text(encoding="utf-8").splitlines()
    replacements = {
        "status        :": f"status        : {summary['status']}",
        "fail_count    :": f"fail_count    : {len(failures)}",
        "blocked_count :": f"blocked_count : {len(blocked)}",
        "performance_ok:": f"performance_ok: {performance['ok']}",
        "perf_critical :": f"perf_critical : {float(wall_summary['critical_path_duration_sec']):.3f}s",
        "perf_aggregate:": f"perf_aggregate: {float(wall_summary['aggregate_step_duration_sec']):.3f}s",
    }
    for index, line in enumerate(lines):
        for prefix, replacement in replacements.items():
            if line.startswith(prefix):
                lines[index] = replacement
                break
    lines.append(f"perf_wall_src : {wall_summary.get('source')}")

    if failures:
        failure_line = "failed_checks : " + ", ".join(failures)
        existing = next((i for i, line in enumerate(lines) if line.startswith("failed_checks :")), None)
        if existing is None:
            lines.append(failure_line)
        else:
            lines[existing] = failure_line
    if blocked:
        blocked_line = "blocked_checks: " + ", ".join(blocked)
        existing = next((i for i, line in enumerate(lines) if line.startswith("blocked_checks:")), None)
        if existing is None:
            lines.append(blocked_line)
        else:
            lines[existing] = blocked_line
    write_text(text_path, "\n".join(lines) + "\n")


__all__ = [
    "FORMAL_WALL_TIME_MANIFEST_KEYS",
    "FORMAL_WALL_TIME_SOURCE",
    "build_parent_wall_time_summary",
    "merge_parent_wall_time_into_meta_quality_summary",
]
