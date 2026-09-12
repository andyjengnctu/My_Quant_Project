"""Outer-rolling seed/live progress event IO and presentation.

This module owns progress-event serialization, seed-progress aggregation, and
progress-line formatting. It does not own fold execution, selection, replay,
or scientific/artifact identity.
"""

from __future__ import annotations

import glob
import json
import math
import os
import time
from threading import Lock

from core.display import C_CYAN, C_GRAY, C_RESET
from services.optimizer.outer_rolling_formatting import (
    _display_short_date_period,
    _fmt_duration,
    _fmt_duration_compact,
)
from services.optimizer.outer_rolling_policy import (
    BASE_RETENTION_COMPARISON_POLICY_NAMES,
    REPORT_POLICY_NAMES,
    _policy_is_available,
)
from services.optimizer.study_utils import is_qualified_trial_value
from services.optimizer.score_display import (
    format_optimizer_score_for_display,
    scale_optimizer_score_for_display,
)


PARALLEL_FOLD_PROGRESS_PREFIX = "FOLD_PROGRESS\t"


def _safe_progress_json_loads(raw_text: str) -> dict:
    try:
        payload = json.loads(str(raw_text))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_parallel_fold_progress_event(*, stage: str, fold_idx: int, fold_count: int, oos_year: int, selection_start, selection_end, **payload) -> None:
    event = {
        "stage": str(stage),
        "fold_idx": int(fold_idx),
        "fold_count": int(fold_count),
        "oos_year": int(oos_year),
        "selection_start": str(selection_start),
        "selection_end": str(selection_end),
        "ts": time.time(),
    }
    event.update(payload)
    print(PARALLEL_FOLD_PROGRESS_PREFIX + json.dumps(event, ensure_ascii=False, sort_keys=True), flush=True)


def write_optimizer_seed_progress_event(*, stage: str, fold_idx: int, fold_count: int, oos_year: int, selection_start, selection_end, **payload) -> None:
    _write_parallel_fold_progress_event(
        stage=stage,
        fold_idx=int(fold_idx),
        fold_count=int(fold_count),
        oos_year=int(oos_year),
        selection_start=selection_start,
        selection_end=selection_end,
        **payload,
    )


def read_optimizer_seed_progresses_from_log_paths(paths) -> dict[int, dict]:
    latest: dict[int, dict] = {}
    for path in list(paths or []):
        for member_index, progress in _read_latest_parallel_fold_seed_progresses(str(path or "")).items():
            current = latest.get(int(member_index))
            if current is None or float(progress.get("ts", 0.0) or 0.0) >= float(current.get("ts", 0.0) or 0.0):
                latest[int(member_index)] = dict(progress)
    return latest


def _fmt_seconds_3(seconds) -> str:
    try:
        value = max(0.0, float(seconds))
    except (TypeError, ValueError):
        value = 0.0
    return f"{value:.3f}s"


def _fmt_score_trunc_2(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    display_number = scale_optimizer_score_for_display(number)
    truncated = math.trunc(display_number * 100.0) / 100.0
    if truncated == 0.0:
        truncated = 0.0
    return f"{truncated:.2f}"


def _safe_progress_ts(progress: dict) -> float | None:
    try:
        value = float(dict(progress or {}).get("ts"))
    except (TypeError, ValueError):
        return None
    return value if value > 0.0 else None


def _safe_progress_float(progress: dict, key: str) -> float | None:
    try:
        value = float(dict(progress or {}).get(str(key)))
    except (TypeError, ValueError):
        return None
    return value if value > 0.0 else None


def _display_short_month_period(value, end_value=None) -> str:
    return _display_short_date_period(value, end_value)


def _count_local_min_completed_from_progress(progress: dict) -> int:
    data = dict(progress or {})
    for key in ("local_min_completed", "completed_local_min_trials"):
        try:
            value = int(data.get(key, 0) or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    if str(data.get("stage") or "").upper() != "LOCAL_MIN_REVIEW":
        return 0
    status = str(data.get("status") or "").strip().upper()
    if status not in {"DONE", "CACHE", "EARLY_STOP", "PASS", "FAIL", "PASS CACHE", "FAIL CACHE"}:
        return 0
    try:
        return max(0, int(data.get("finalist_idx", 0) or 0))
    except (TypeError, ValueError):
        return 0


def _count_local_min_completed_neighbors_from_progress(progress: dict) -> int:
    """Return completed local-min neighbor units for avg_local throughput.

    The display metric avg_local is defined as first local-min neighbor start to
    last completed neighbor divided by completed neighbor count, not finalist
    count.  Older progress events did not carry the cumulative neighbor count,
    so the fallback below uses the current finalist/neighbor position only for
    backward compatibility.
    """
    data = dict(progress or {})
    for key in ("local_min_neighbor_completed", "completed_local_min_neighbors"):
        try:
            value = int(data.get(key, 0) or 0)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            return value
    if str(data.get("stage") or "").upper() != "LOCAL_MIN_REVIEW":
        return 0
    try:
        finalist_idx = max(0, int(data.get("finalist_idx", 0) or 0))
        neighbor_done = max(0, int(data.get("neighbor_done", 0) or 0))
        neighbor_total = max(0, int(data.get("neighbor_total", 0) or 0))
    except (TypeError, ValueError):
        return 0
    if finalist_idx <= 0:
        return max(0, neighbor_done)
    # Approximation for legacy events: most generated neighbor grids are stable
    # across finalists.  New events carry the exact cumulative value above.
    return max(0, (finalist_idx - 1) * neighbor_total + neighbor_done)


def _progress_identity_key(progress: dict) -> tuple[int, int]:
    data = dict(progress or {})
    try:
        fold_idx = int(data.get("fold_idx", 0) or 0)
    except (TypeError, ValueError):
        fold_idx = 0
    try:
        seed_index = int(data.get("seed_ensemble_member_index", 0) or 0)
    except (TypeError, ValueError):
        seed_index = 0
    if seed_index <= 0:
        try:
            seed_index = int(data.get("seed_index", 0) or 0)
        except (TypeError, ValueError):
            seed_index = 0
    return (fold_idx, seed_index)


def _merge_progress_phase_metrics(target: dict, progress: dict) -> None:
    data = dict(progress or {})
    stage = str(data.get("stage") or "").upper()
    ts = _safe_progress_ts(data)
    search_start = _safe_progress_float(data, "search_started_ts")
    try:
        completed = int(data.get("completed", 0) or 0)
    except (TypeError, ValueError):
        completed = 0
    if search_start is None and stage == "OPTIMIZER_SEARCH" and completed <= 0:
        search_start = ts
    if search_start is not None:
        previous = target.get("search_started_ts")
        target["search_started_ts"] = search_start if previous is None else min(float(previous), float(search_start))
    search_end = _safe_progress_float(data, "search_last_done_ts")
    if search_end is None and stage == "OPTIMIZER_SEARCH" and completed > 0:
        search_end = ts
    if search_end is not None:
        target["search_last_done_ts"] = max(float(target.get("search_last_done_ts", 0.0) or 0.0), float(search_end))
    if completed > int(target.get("completed_trials", 0) or 0):
        target["completed_trials"] = int(completed)

    local_start = _safe_progress_float(data, "local_min_started_ts")
    if local_start is not None:
        previous = target.get("local_min_started_ts")
        target["local_min_started_ts"] = local_start if previous is None else min(float(previous), float(local_start))
    local_neighbors = _count_local_min_completed_neighbors_from_progress(data)
    local_finalists = _count_local_min_completed_from_progress(data)
    if local_neighbors > int(target.get("completed_local_min_neighbors", 0) or 0):
        target["completed_local_min_neighbors"] = int(local_neighbors)
    if local_finalists > int(target.get("completed_local_min_finalists", 0) or 0):
        target["completed_local_min_finalists"] = int(local_finalists)
    local_end = _safe_progress_float(data, "local_min_last_neighbor_done_ts")
    if local_end is None:
        local_end = _safe_progress_float(data, "local_min_last_done_ts")
    if local_end is None and local_neighbors > 0:
        local_end = ts
    if local_end is not None:
        target["local_min_last_done_ts"] = max(float(target.get("local_min_last_done_ts", 0.0) or 0.0), float(local_end))


def _collect_seed_progress_phase_metrics(progresses) -> dict:
    merged_by_key: dict[tuple[int, int], dict] = {}
    anonymous_index = 0
    for raw in list(progresses or []):
        progress = dict(raw or {})
        key = _progress_identity_key(progress)
        if key == (0, 0):
            anonymous_index += 1
            key = (-anonymous_index, 0)
        merged = merged_by_key.setdefault(key, {})
        _merge_progress_phase_metrics(merged, progress)
    completed_trials = 0
    completed_local = 0
    completed_local_neighbors = 0
    search_started: list[float] = []
    search_done: list[float] = []
    local_started: list[float] = []
    local_done: list[float] = []
    for metrics in merged_by_key.values():
        completed_trials += int(metrics.get("completed_trials", 0) or 0)
        completed_local += int(metrics.get("completed_local_min_finalists", 0) or 0)
        completed_local_neighbors += int(metrics.get("completed_local_min_neighbors", 0) or 0)
        if metrics.get("search_started_ts") is not None:
            search_started.append(float(metrics["search_started_ts"]))
        if metrics.get("search_last_done_ts") is not None:
            search_done.append(float(metrics["search_last_done_ts"]))
        if metrics.get("local_min_started_ts") is not None:
            local_started.append(float(metrics["local_min_started_ts"]))
        if metrics.get("local_min_last_done_ts") is not None:
            local_done.append(float(metrics["local_min_last_done_ts"]))
    search_span = None
    if search_started and search_done:
        search_span = max(0.0, max(search_done) - min(search_started))
    local_span = None
    if local_started and local_done:
        local_span = max(0.0, max(local_done) - min(local_started))
    return {
        "completed_trials": int(completed_trials),
        "completed_local_min_trials": int(completed_local_neighbors),
        "completed_local_min_finalists": int(completed_local),
        "search_wall_elapsed_sec": search_span,
        "local_min_wall_elapsed_sec": local_span,
    }


def format_optimizer_seed_ensemble_progress_header(
    *,
    folds: int,
    seeds: int,
    min_agree: int | None,
    parallel_workers: int,
    selector: str | None = None,
    backend: str,
    completed_folds: int | None = None,
    pending_folds: int | None = None,
    total_elapsed_sec: float | None = None,
    fold_elapsed_sec: float | None = None,
    setup_elapsed_sec: float | None = None,
    completed_trials: int | None = None,
    search_wall_elapsed_sec: float | None = None,
    completed_local_min_trials: int | None = None,
    local_min_wall_elapsed_sec: float | None = None,
    completed_replays: int | None = None,
    replay_wall_elapsed_sec: float | None = None,
) -> str:
    """Format the seed-ensemble progress header used by rolling and non-rolling paths."""
    _ = (parallel_workers, backend, pending_folds, fold_elapsed_sec, setup_elapsed_sec)
    parts = [
        f"folds={int(folds)}",
        f"seeds={int(seeds)}",
    ]
    if selector:
        parts.append(f"selector={str(selector)}")
    if min_agree is not None:
        parts.append(f"seed_min_agree={int(min_agree)}")
    if completed_folds is not None:
        parts.append(f"completed={int(completed_folds)}/{int(folds)}")
    if total_elapsed_sec is not None:
        parts.append(f"total={_fmt_duration(total_elapsed_sec)}")
    try:
        trial_count = int(completed_trials or 0)
    except (TypeError, ValueError):
        trial_count = 0
    try:
        local_count = int(completed_local_min_trials or 0)
    except (TypeError, ValueError):
        local_count = 0
    if trial_count > 0 and search_wall_elapsed_sec is not None:
        parts.append(f"avg_trial={_fmt_seconds_3(float(search_wall_elapsed_sec) / float(trial_count))}")
    try:
        replay_count = int(completed_replays or 0)
    except (TypeError, ValueError):
        replay_count = 0
    if local_count > 0 and local_min_wall_elapsed_sec is not None:
        parts.append(f"avg_local={_fmt_seconds_3(float(local_min_wall_elapsed_sec) / float(local_count))}")
    if replay_count > 0 and replay_wall_elapsed_sec is not None:
        parts.append(f"avg_replay={_fmt_seconds_3(float(replay_wall_elapsed_sec) / float(replay_count))}")
    all_count = max(0, int(trial_count) + int(local_count) + int(replay_count))
    if all_count > 0 and total_elapsed_sec is not None:
        parts.append(f"avg_all={_fmt_seconds_3(float(total_elapsed_sec) / float(all_count))}")
    return " | ".join(parts)


def _optimizer_resource_usage_suffix(summary: dict | None) -> str:
    payload = dict(summary or {})
    if not bool(payload.get("resource_sampling_available", False)):
        return "CPU avg=N/A | MEM avg=N/A | HD avg=N/A"
    cpu_avg = float(payload.get("cpu_avg_percent", 0.0) or 0.0)
    mem_avg = float(payload.get("memory_avg_percent", 0.0) or 0.0)
    hd_avg = float(payload.get("disk_load_avg_percent", payload.get("disk_busy_avg_percent", 0.0)) or 0.0)
    return f"CPU avg={cpu_avg:.1f}% | MEM avg={mem_avg:.1f}% | HD avg={hd_avg:.1f}%"


def format_optimizer_final_performance_summary(
    *,
    folds: int,
    seeds: int,
    min_agree: int | None,
    completed_folds: int,
    selector: str | None = None,
    finalist_count: int | None = None,
    finalist_min_agree: int | None = None,
    total_elapsed_sec: float,
    completed_trials: int = 0,
    search_wall_elapsed_sec: float | None = None,
    completed_local_min_trials: int = 0,
    local_min_wall_elapsed_sec: float | None = None,
    completed_replays: int = 0,
    replay_wall_elapsed_sec: float | None = None,
    resource_summary: dict | None = None,
    seed_ensemble_enabled: bool = True,
    color: bool = True,
) -> str:
    """Format final performance/resource summary from the same seed-ensemble schema.

    This is intentionally shared by rolling and non-rolling; callers only provide
    different fold counts and collected metrics.
    """
    header = format_optimizer_seed_ensemble_progress_header(
        folds=int(folds),
        seeds=int(seeds),
        min_agree=None if min_agree is None else int(min_agree),
        parallel_workers=0,
        selector=selector,
        backend="",
        completed_folds=int(completed_folds),
        total_elapsed_sec=float(total_elapsed_sec),
        completed_trials=int(completed_trials or 0),
        search_wall_elapsed_sec=search_wall_elapsed_sec,
        completed_local_min_trials=int(completed_local_min_trials or 0),
        local_min_wall_elapsed_sec=local_min_wall_elapsed_sec,
        completed_replays=int(completed_replays or 0),
        replay_wall_elapsed_sec=replay_wall_elapsed_sec,
    )
    ensemble_note = "" if bool(seed_ensemble_enabled) else " | seed_ensemble=off"
    finalist_note = ""
    if finalist_count is not None and finalist_min_agree is not None:
        finalist_note = f" | finalists={int(finalist_count)} | finalist_min_agree={int(finalist_min_agree)}"
    line = f"📏 訓練效能摘要: {header}{finalist_note}{ensemble_note} | {_optimizer_resource_usage_suffix(resource_summary)}"
    return f"{C_CYAN}{line}{C_RESET}" if bool(color) else line


def _format_parallel_fold_progress_line(task: dict, progress: dict, *, log_status: str = "") -> str:
    fold_idx = int(task.get("fold_idx", progress.get("fold_idx", 0)) or 0)
    fold_count = int(task.get("fold_count", progress.get("fold_count", 0)) or 0)
    oos_year = int(task.get("oos_year", progress.get("oos_year", 0)) or 0)
    show_oos = bool(task.get("show_oos", progress.get("show_oos", True)))
    oos_label = _display_short_month_period(task.get("oos_period") or oos_year) if show_oos else ""
    selection_start = str(progress.get("selection_start") or task.get("selection_period") or "").strip()
    selection_end = str(progress.get("selection_end") or "").strip()
    if selection_start and selection_end:
        selection_text = f"train={_display_short_month_period(selection_start, selection_end)}"
    elif selection_start:
        selection_text = f"train={_display_short_month_period(selection_start)}"
    else:
        selection_text = "train=?"
    stage = str(progress.get("stage") or "QUEUED").upper()

    def _progress_prefix() -> str:
        parts = [f"[{fold_idx}/{fold_count}] {selection_text}"]
        if show_oos:
            parts.append(f"OOS {oos_label}")
        return " | ".join(parts)

    elapsed_text = ""
    if progress.get("elapsed_sec") is not None:
        elapsed_text = f" | elapsed={_fmt_duration_compact(progress.get('elapsed_sec'))}"
    if stage == "OPTIMIZER_SEARCH":
        completed = int(progress.get("completed", 0) or 0)
        total = int(progress.get("total", 0) or 0)
        best = progress.get("best_score")
        best_text = format_optimizer_score_for_display(best, decimals=3)
        status = str(progress.get("status") or "").strip()
        status_text = f"{status} | " if status else ""
        local_best = progress.get("best_local_min_score")
        local_best_text = format_optimizer_score_for_display(local_best, decimals=3)
        return (
            f"{_progress_prefix()} | "
            f"{status_text}trial {completed}/{total} | best_base={best_text} | best_local_min={local_best_text}{elapsed_text}"
        )
    if stage == "LOCAL_MIN_REVIEW":
        finalist_idx = int(progress.get("finalist_idx", 0) or 0)
        finalist_total = int(progress.get("finalist_total", 0) or 0)
        neighbor_done = int(progress.get("neighbor_done", 0) or 0)
        neighbor_total = int(progress.get("neighbor_total", 0) or 0)
        current = progress.get("current")
        current_text = format_optimizer_score_for_display(current, decimals=3)
        best = progress.get("best")
        best_text = _fmt_score_trunc_2(best)
        base_best = progress.get("best_base_score")
        base_best_text = _fmt_score_trunc_2(base_best)
        local_min_elapsed_text = ""
        if progress.get("elapsed_sec") is not None:
            local_min_elapsed_text = f" | elapsed={_fmt_duration_compact(progress.get('elapsed_sec'))}"
        return (
            f"{_progress_prefix()} | "
            f"finalist {finalist_idx}/{finalist_total} | local {neighbor_done}/{neighbor_total} | "
            f"current : {current_text} | best_base : {base_best_text} | best_lm : {best_text}{local_min_elapsed_text}"
        )
    if stage == "OOS_DIAGNOSTICS":
        status = str(progress.get("status") or "RUN")
        base_best = progress.get("best_base_score")
        local_best = progress.get("best_local_min_score")
        base_best_text = format_optimizer_score_for_display(base_best, decimals=3)
        local_best_text = format_optimizer_score_for_display(local_best, decimals=3)
        return f"{_progress_prefix()} | diagnostics {status} | best_base={base_best_text} | best_local_min={local_best_text}{elapsed_text}"
    if stage == "ENSEMBLE_REPLAY":
        replay_done = int(progress.get("replay_done", 0) or 0)
        replay_total = int(progress.get("replay_total", 0) or 0)
        policy = str(progress.get("policy") or progress.get("status") or "policy").strip()
        if replay_total > 0:
            return f"{_progress_prefix()} | policy replay {replay_done}/{replay_total} | {policy}{elapsed_text}"
        return f"{_progress_prefix()} | policy replay | {policy}{elapsed_text}"
    if stage == "FOLD_RESULT":
        return f"{_progress_prefix()} | fold result ready{elapsed_text}"
    if stage == "DONE":
        status = _strip_redundant_seed_status(str(progress.get("status") or ""))
        done_label = "DONE" if not status or status.lower() == "done" else f"DONE {status}"
        base_best = progress.get("best_base_score")
        local_best = progress.get("best_local_min_score")
        base_best_text = format_optimizer_score_for_display(base_best, decimals=3)
        local_best_text = format_optimizer_score_for_display(local_best, decimals=3)
        return f"{_progress_prefix()} | {done_label} | best_base={base_best_text} | best_local_min={local_best_text}{elapsed_text}"
    if stage in {"START", "RAW_DATA", "STUDY_CREATE"}:
        status = str(progress.get("status") or stage).replace("_", " ")
        return f"{_progress_prefix()} | {status}{elapsed_text}"
    fallback = str(log_status or "queued").strip()
    return f"{_progress_prefix()} | {fallback}"


def _seed_progress_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _strip_redundant_seed_status(raw_status: str) -> str:
    return _strip_redundant_member_status(raw_status, member_text="")


def _strip_redundant_member_status(raw_status: str, *, member_text: str = "") -> str:
    status_text = str(raw_status or "").strip()
    if not status_text:
        return ""
    candidates = [str(member_text or "").strip()]
    for candidate in candidates:
        if not candidate:
            continue
        if status_text == candidate:
            return ""
        prefix_text = f"{candidate} "
        if status_text.startswith(prefix_text):
            return status_text[len(prefix_text):].strip()
    return status_text


def _format_best_base_value(value) -> str:
    return format_optimizer_score_for_display(value, decimals=3)


def _format_best_local_min_value(value) -> str:
    return format_optimizer_score_for_display(value, decimals=3)


def _seed_progress_label_for_stage(*, stage: str, progress: dict, seed_text: str, log_status: str = "") -> str:
    stage_text = str(stage or "QUEUED").upper()
    status = _strip_redundant_member_status(str(progress.get("status") or log_status or ""), member_text=seed_text)
    status_upper = status.strip().upper()
    if stage_text == "DONE":
        if not status or status.lower() == "done":
            return "DONE"
        return f"DONE {status}"
    if stage_text in {"QUEUED", "START", "RAW_DATA", "STUDY_CREATE"}:
        if status_upper in {"", "QUEUED", "START", "RAW DATA", "STUDY CREATE"}:
            return stage_text.replace("_", " ")
        return status.replace("_", " ")
    if stage_text == "OPTIMIZER_SEARCH":
        completed = _seed_progress_int(progress.get("completed", 0), 0)
        total = _seed_progress_int(progress.get("total", 0), 0)
        return f"trial {completed}/{total}"
    if stage_text == "LOCAL_MIN_REVIEW":
        finalist_idx = _seed_progress_int(progress.get("finalist_idx", 0), 0)
        finalist_total = _seed_progress_int(progress.get("finalist_total", 0), 0)
        neighbor_done = _seed_progress_int(progress.get("neighbor_done", 0), 0)
        neighbor_total = _seed_progress_int(progress.get("neighbor_total", 0), 0)
        return f"finalist {finalist_idx}/{finalist_total} | local {neighbor_done}/{neighbor_total}"
    if stage_text == "OOS_DIAGNOSTICS":
        return f"diagnostics {status or 'RUN'}"
    return status.replace("_", " ") if status else stage_text.replace("_", " ")


def _normalize_seed_progress_display_record(context: dict, progress: dict, *, log_status: str = "") -> dict:
    context = dict(context or {})
    progress = dict(progress or {})
    fold_idx = _seed_progress_int(context.get("fold_idx", progress.get("fold_idx", 0)), 0)
    fold_count = _seed_progress_int(context.get("fold_count", progress.get("fold_count", 0)), 0)
    seed_index = _seed_progress_int(context.get("seed_index", progress.get("seed_ensemble_member_index", 0)), 0)
    seed_count = _seed_progress_int(context.get("seed_count", progress.get("seed_ensemble_member_count", 0)), 0)
    show_oos = bool(context.get("show_oos", True))
    oos_year = _seed_progress_int(context.get("oos_year", progress.get("oos_year", 0)), 0)
    oos_label = _display_short_month_period(context.get("oos_period") or progress.get("oos_period") or oos_year) if show_oos else ""
    selection_start = str(progress.get("selection_start") or context.get("selection_start") or context.get("selection_period") or "").strip()
    selection_end = str(progress.get("selection_end") or context.get("selection_end") or "").strip()
    if selection_start and selection_end:
        selection_text = f"train={_display_short_month_period(selection_start, selection_end)}"
    elif selection_start:
        selection_text = f"train={_display_short_month_period(selection_start)}"
    else:
        selection_text = "train=?"
    seed_text = f"seed {seed_index}/{seed_count}" if seed_index and seed_count else "seed ?/?"
    stage = str(progress.get("stage") or "QUEUED").upper()
    elapsed_sec = progress.get("elapsed_sec")
    base_best = progress.get("best_base_score")
    if base_best is None:
        base_best = progress.get("best_score")
    local_best = progress.get("best_local_min_score")
    if local_best is None:
        local_best = progress.get("best") if stage == "LOCAL_MIN_REVIEW" else None
    return {
        "fold_idx": fold_idx,
        "fold_count": fold_count,
        "selection_text": selection_text,
        "show_oos": show_oos,
        "oos_label": oos_label,
        "seed_text": seed_text,
        "stage": stage,
        "status_label": _seed_progress_label_for_stage(stage=stage, progress=progress, seed_text=seed_text, log_status=log_status),
        "best_base": base_best,
        "best_local_min": local_best,
        "elapsed_sec": elapsed_sec,
        "current": progress.get("current"),
    }


def _render_seed_progress_display_record(record: dict) -> str:
    record = dict(record or {})
    fold_idx = _seed_progress_int(record.get("fold_idx", 0), 0)
    fold_count = _seed_progress_int(record.get("fold_count", 0), 0)
    stage = str(record.get("stage") or "QUEUED").upper()
    status_label = str(record.get("status_label") or stage).strip()
    prefix_parts = [f"[{fold_idx}/{fold_count}] {record.get('selection_text') or 'train=?'}"]
    if bool(record.get("show_oos", True)):
        prefix_parts.append(f"OOS {record.get('oos_label') or '?'}")
    prefix_parts.append(str(record.get('seed_text') or 'seed ?/?'))
    parts = [" | ".join(prefix_parts), status_label]
    if stage == "LOCAL_MIN_REVIEW":
        current = record.get("current")
        current_text = format_optimizer_score_for_display(current, decimals=3)
        parts.append(f"current : {current_text}")
        parts.append(f"best_base : {_fmt_score_trunc_2(record.get('best_base'))}")
        parts.append(f"best_lm : {_fmt_score_trunc_2(record.get('best_local_min'))}")
    else:
        parts.append(f"best_base={_format_best_base_value(record.get('best_base'))}")
        parts.append(f"best_local_min={_format_best_local_min_value(record.get('best_local_min'))}")
    if record.get("elapsed_sec") is not None:
        parts.append(f"elapsed={_fmt_duration_compact(record.get('elapsed_sec'))}")
    return " | ".join(parts)


def _format_seed_ensemble_progress_line(context: dict, progress: dict, *, log_status: str = "") -> str:
    return _render_seed_progress_display_record(
        _normalize_seed_progress_display_record(dict(context or {}), dict(progress or {}), log_status=log_status)
    )


def _seed_progress_context_key(context: dict) -> tuple[int, int]:
    return (
        _seed_progress_int((context or {}).get("fold_idx", 0), 0),
        _seed_progress_int((context or {}).get("seed_index", 0), 0),
    )


def build_optimizer_seed_ensemble_live_lines(
    *,
    header_text: str = "",
    seed_contexts: list[dict] | tuple[dict, ...] = (),
    seed_progress_by_key: dict[tuple[int, int], dict] | None = None,
    fold_progress_lines: list[str] | tuple[str, ...] | None = None,
    table_text: str = "",
    color: bool = True,
) -> list[str]:
    """Build the live seed-ensemble display block from one renderer.

    Rolling and non-rolling callers must only provide different inputs
    (fold_count, oos_periods, and progress events).  Header, seed status lines,
    replay/fold lines, and live table placement are rendered here.
    """
    progress_by_key = dict(seed_progress_by_key or {})
    lines: list[str] = []
    if str(header_text or "").strip():
        header = str(header_text).strip()
        lines.append(f"{C_CYAN}{header}{C_RESET}" if color else header)
    for context in sorted([dict(item) for item in list(seed_contexts or [])], key=lambda item: _seed_progress_context_key(item)):
        key = _seed_progress_context_key(context)
        progress = dict(progress_by_key.get(key) or {"stage": "QUEUED", "status": "queued"})
        line = _format_seed_ensemble_progress_line(context, progress)
        lines.append(f"{C_GRAY}{line}{C_RESET}" if color else line)
    for raw_line in list(fold_progress_lines or []):
        line = str(raw_line or "").strip()
        if line:
            lines.append(f"{C_GRAY}{line}{C_RESET}" if color else line)
    if str(table_text or "").strip():
        if lines:
            lines.append("")
        lines.extend(str(table_text).splitlines())
    return lines


def _read_latest_parallel_fold_seed_progresses(path: str) -> dict[int, dict]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError:
        return {}
    latest: dict[int, dict] = {}
    events_by_member: dict[int, list[dict]] = {}
    completed_by_member: dict[int, int] = {}
    local_completed_by_member: dict[int, int] = {}
    local_neighbor_completed_by_member: dict[int, int] = {}
    for line in reversed(lines[-2000:]):
        raw = str(line).strip()
        if not raw.startswith(PARALLEL_FOLD_PROGRESS_PREFIX):
            continue
        payload = _safe_progress_json_loads(raw.split("\t", 1)[1])
        try:
            member_index = int(payload.get("seed_ensemble_member_index", 0) or 0)
        except (TypeError, ValueError):
            member_index = 0
        if member_index <= 0:
            continue
        events_by_member.setdefault(member_index, []).append(dict(payload))
        try:
            completed = int(payload.get("completed", 0) or 0)
        except (TypeError, ValueError):
            completed = 0
        if completed > int(completed_by_member.get(member_index, 0) or 0):
            completed_by_member[member_index] = int(completed)
        local_completed = _count_local_min_completed_from_progress(payload)
        if local_completed > int(local_completed_by_member.get(member_index, 0) or 0):
            local_completed_by_member[member_index] = int(local_completed)
        local_neighbor_completed = _count_local_min_completed_neighbors_from_progress(payload)
        if local_neighbor_completed > int(local_neighbor_completed_by_member.get(member_index, 0) or 0):
            local_neighbor_completed_by_member[member_index] = int(local_neighbor_completed)
        if member_index not in latest:
            latest[member_index] = payload
    for member_index, payload in list(latest.items()):
        completed = int(completed_by_member.get(member_index, 0) or 0)
        if completed > int(payload.get("completed", 0) or 0):
            payload["completed"] = int(completed)
        local_completed = int(local_completed_by_member.get(member_index, 0) or 0)
        if local_completed > int(payload.get("local_min_completed", 0) or 0):
            payload["local_min_completed"] = int(local_completed)
        local_neighbor_completed = int(local_neighbor_completed_by_member.get(member_index, 0) or 0)
        if local_neighbor_completed > int(payload.get("local_min_neighbor_completed", 0) or 0):
            payload["local_min_neighbor_completed"] = int(local_neighbor_completed)
        events = events_by_member.get(member_index, [])
        search_starts = []
        search_dones = []
        local_starts = []
        local_dones = []
        for event in events:
            stage = str(event.get("stage") or "").upper()
            ts = _safe_progress_ts(event)
            search_start = _safe_progress_float(event, "search_started_ts")
            if search_start is not None:
                search_starts.append(search_start)
            elif stage == "OPTIMIZER_SEARCH" and ts is not None and int(event.get("completed", 0) or 0) <= 0:
                search_starts.append(ts)
            search_done = _safe_progress_float(event, "search_last_done_ts")
            if search_done is not None:
                search_dones.append(search_done)
            elif stage == "OPTIMIZER_SEARCH" and ts is not None and int(event.get("completed", 0) or 0) > 0:
                search_dones.append(ts)
            local_start = _safe_progress_float(event, "local_min_started_ts")
            if local_start is not None:
                local_starts.append(local_start)
            local_done = _safe_progress_float(event, "local_min_last_neighbor_done_ts")
            if local_done is None:
                local_done = _safe_progress_float(event, "local_min_last_done_ts")
            if local_done is not None:
                local_dones.append(local_done)
            elif _count_local_min_completed_neighbors_from_progress(event) > 0 and ts is not None:
                local_dones.append(ts)
        if search_starts:
            payload["search_started_ts"] = min(search_starts)
        if search_dones:
            payload["search_last_done_ts"] = max(search_dones)
        if local_starts:
            payload["local_min_started_ts"] = min(local_starts)
        if local_dones:
            payload["local_min_last_neighbor_done_ts"] = max(local_dones)
            payload["local_min_last_done_ts"] = max(local_dones)
    return latest


def _parallel_fold_seed_log_paths(task: dict) -> list[str]:
    log_path = str((task or {}).get("log_path") or "")
    paths: list[str] = []
    if log_path:
        paths.append(log_path)
        root, ext = os.path.splitext(log_path)
        pattern = f"{root}_seed*{ext or '.log'}"
        paths.extend(sorted(glob.glob(pattern)))
    deduped: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if not path or path in seen:
            continue
        deduped.append(path)
        seen.add(path)
    return deduped


def _read_latest_parallel_fold_seed_progresses_for_task(task: dict) -> dict[int, dict]:
    latest: dict[int, dict] = {}
    for path in _parallel_fold_seed_log_paths(task):
        for member_index, progress in _read_latest_parallel_fold_seed_progresses(path).items():
            current = latest.get(member_index)
            if current is None or float(progress.get("ts", 0.0) or 0.0) >= float(current.get("ts", 0.0) or 0.0):
                latest[int(member_index)] = dict(progress)
    return latest


def render_optimizer_fold_progress_line(
    *,
    fold_idx: int,
    fold_count: int,
    selection_start: str = "",
    selection_end: str = "",
    selection_period: str = "",
    oos_period: str = "",
    oos_year: int = 0,
    show_oos: bool = True,
    stage: str = "START",
    status: str = "",
    completed: int = 0,
    total: int = 0,
    best_score=None,
    best_base_score=None,
    best_local_min_score=None,
    elapsed_sec=None,
    **_unused_context,
) -> str:
    """Render one optimizer fold progress line through the rolling-OOS source.

    Non-rolling uses this with fold_count=1 so progress text and table text are
    both owned by this module instead of being reimplemented in main.py.
    """
    task = {
        "fold_idx": int(fold_idx),
        "fold_count": int(fold_count),
        "oos_year": int(oos_year or 0),
        "oos_period": str(oos_period or ""),
        "selection_period": str(selection_period or selection_start or ""),
        "show_oos": bool(show_oos),
    }
    progress = {
        "fold_idx": int(fold_idx),
        "fold_count": int(fold_count),
        "oos_year": int(oos_year or 0),
        "stage": str(stage or "START"),
        "status": str(status or ""),
        "show_oos": bool(show_oos),
        "selection_start": str(selection_start or selection_period or ""),
        "selection_end": str(selection_end or ""),
        "completed": int(completed or 0),
        "total": int(total or 0),
        "best_score": best_score,
        "best_base_score": best_base_score,
        "best_local_min_score": best_local_min_score,
        "elapsed_sec": elapsed_sec,
    }
    return _format_parallel_fold_progress_line(task, progress)


def render_optimizer_seed_progress_line(*, context: dict, progress: dict) -> str:
    return _format_seed_ensemble_progress_line(dict(context or {}), dict(progress or {}))


def _compact_policy_for_live_result(policy: dict) -> dict:
    payload = dict(policy or {})
    available = _policy_is_available(payload)
    compact = {
        "available": bool(available),
        "unavailable_reason": str(payload.get("unavailable_reason") or payload.get("skip_reason") or ""),
    }
    if available:
        for key in ("rank_1_oos", "rank_1_plain_romd", "rank_1_return_pct", "rank_1_mdd_pct", "benchmark_0050_gap", "benchmark_0050_plain_romd_gap", "best_gap"):
            if key in payload:
                try:
                    compact[key] = float(payload.get(key, 0.0) or 0.0)
                except (TypeError, ValueError):
                    compact[key] = 0.0
        for key in ("rank_1_trial", "rank_1_trades"):
            if key in payload:
                compact[key] = payload.get(key)
    return compact


def _compact_row_for_live_result(row: dict) -> dict:
    source = dict(row or {})
    compact = {
        "fold": source.get("fold"),
        "oos_year": source.get("oos_year"),
        "selection_period": source.get("selection_period", ""),
        "oos_period": source.get("oos_period") or source.get("oos_year", ""),
        "best_finalist_oos_score": float(source.get("best_finalist_oos_score", 0.0) or 0.0),
        "benchmark_oos_score": float(source.get("benchmark_oos_score", 0.0) or 0.0),
        "elapsed_sec": source.get("elapsed_sec"),
    }
    for policy_name in list(REPORT_POLICY_NAMES) + list(BASE_RETENTION_COMPARISON_POLICY_NAMES):
        compact[policy_name] = _compact_policy_for_live_result(source.get(policy_name) or {})
    return compact


def _seed_progress_context_from_task(task: dict) -> dict:
    if not bool((task or {}).get("seed_ensemble_member")):
        return {}
    member_index = int((task or {}).get("seed_ensemble_member_index", 0) or 0)
    member_count = int((task or {}).get("seed_ensemble_member_count", 0) or 0)
    if member_index <= 0 or member_count <= 0:
        return {}
    return {
        "seed_ensemble_member_index": int(member_index),
        "seed_ensemble_member_count": int(member_count),
        "seed": int((task or {}).get("optimizer_seed", 0) or 0),
    }


class _FoldLogSearchProgress:
    def __init__(self, *, fold_idx: int, fold_count: int, oos_year: int, selection_start, selection_end, total_trials: int, seed_context: dict | None = None):
        self.fold_idx = int(fold_idx)
        self.fold_count = int(fold_count)
        self.oos_year = int(oos_year)
        self.selection_start = str(selection_start)
        self.selection_end = str(selection_end)
        self.total_trials = int(total_trials)
        self.seed_context = dict(seed_context or {})
        self.stage_start = time.perf_counter()
        self.stage_start_ts = time.time()
        self.best_score = float("-inf")
        self.last_completed = -1
        self._lock = Lock()

    def emit(self, completed: int, *, force: bool = False) -> None:
        completed = int(completed)
        if not force and completed == self.last_completed:
            return
        self.last_completed = completed
        best_score = None if self.best_score == float("-inf") else float(self.best_score)
        _write_parallel_fold_progress_event(
            stage="OPTIMIZER_SEARCH",
            fold_idx=self.fold_idx,
            fold_count=self.fold_count,
            oos_year=self.oos_year,
            selection_start=self.selection_start,
            selection_end=self.selection_end,
            **self.seed_context,
            completed=completed,
            total=self.total_trials,
            best_score=best_score,
            elapsed_sec=max(0.0, time.perf_counter() - self.stage_start),
            search_started_ts=float(self.stage_start_ts),
            search_last_done_ts=time.time() if int(completed) > 0 else None,
        )

    def callback(self, session):
        def _callback(study, trial):
            with self._lock:
                session.current_session_trial += 1
                if trial.value is not None and is_qualified_trial_value(trial.value):
                    self.best_score = max(self.best_score, float(trial.value))
                completed = int(session.current_session_trial)
            self.emit(completed)
        return _callback

    def done(self, completed: int) -> None:
        self.emit(int(completed), force=True)
