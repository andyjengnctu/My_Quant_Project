"""Canonical execution-only scheduler for rolling-fold seed members.

Scientific seed membership and fold payloads are provided by the producer.
This module owns only progress-derived tail estimates and mutable scheduler
state; ProcessPool submission, failures, and result aggregation stay in the
outer-rolling runtime. The progress files are read at the same decision points.
"""

from __future__ import annotations

import statistics

from services.optimizer.outer_rolling_progress import (
    _read_latest_parallel_fold_seed_progresses_for_task,
    _safe_progress_float,
)


def _seed_scheduler_fold_rate_sec_per_trial(task: dict) -> float | None:
    progresses = dict(_read_latest_parallel_fold_seed_progresses_for_task(task) or {})
    total_trials = 0
    total_elapsed = 0.0
    for progress in progresses.values():
        data = dict(progress or {})
        try:
            completed = int(data.get("completed", 0) or 0)
        except (TypeError, ValueError):
            completed = 0
        if completed <= 0:
            continue
        elapsed = _safe_progress_float(data, "search_active_elapsed_sec")
        if elapsed is None:
            elapsed = _safe_progress_float(data, "elapsed_sec")
        if elapsed is None or float(elapsed) <= 0.0:
            continue
        total_trials += int(completed)
        total_elapsed += float(elapsed)
    if total_trials <= 0 or total_elapsed <= 0.0:
        return None
    return float(total_elapsed) / float(total_trials)


def _seed_scheduler_remaining_trial_units(state: dict) -> int:
    task = dict(state.get("task") or {})
    trials_per_seed = int(dict(task.get("config") or {}).get("trials_per_fold", 0) or 0)
    pending_count = len(list(state.get("pending_members") or []))
    remaining = max(0, int(pending_count) * int(trials_per_seed))
    progresses = dict(_read_latest_parallel_fold_seed_progresses_for_task(task) or {})
    for member_index in set(state.get("active_members") or set()):
        progress = dict(progresses.get(int(member_index)) or {})
        try:
            completed = int(progress.get("completed", 0) or 0)
        except (TypeError, ValueError):
            completed = 0
        remaining += max(0, int(trials_per_seed) - int(completed))
    return int(remaining)


def _seed_scheduler_select_state(states: list[dict]) -> dict | None:
    eligible = [
        state
        for state in list(states or [])
        if list(state.get("pending_members") or [])
        and len(set(state.get("active_members") or set()))
        < int(state.get("per_fold_cap", 1) or 1)
    ]
    if not eligible:
        return None
    known_rates = [
        float(rate)
        for rate in (state.get("rate_sec_per_trial") for state in eligible)
        if rate is not None and float(rate) > 0.0
    ]
    fallback_rate = statistics.median(known_rates) if known_rates else 1.0

    def _priority(state: dict):
        rate = state.get("rate_sec_per_trial")
        effective_rate = (
            float(rate)
            if rate is not None and float(rate) > 0.0
            else float(fallback_rate)
        )
        remaining = int(state.get("remaining_trial_units", 0) or 0)
        active_count = max(1, len(set(state.get("active_members") or set())))
        # Estimate the fold tail after accounting for its current seed concurrency.
        # This prevents two newly free slots from both piling onto the same slow fold
        # before the next-slowest fold receives help.
        estimated_tail_sec = (
            float(effective_rate) * float(max(0, remaining)) / float(active_count)
        )
        return (
            float(estimated_tail_sec),
            -int(dict(state.get("task") or {}).get("fold_idx", 0) or 0),
        )

    return max(eligible, key=_priority)


def _refresh_seed_scheduler_state(states: list[dict]) -> None:
    for state in list(states or []):
        task = dict(state.get("task") or {})
        state["rate_sec_per_trial"] = _seed_scheduler_fold_rate_sec_per_trial(task)
        state["remaining_trial_units"] = _seed_scheduler_remaining_trial_units(state)

def _seed_scheduler_build_state(task: dict, seeds: list[int], *, per_fold_cap: int) -> dict:
    """Create the previous wire-compatible fold state, preserving member order."""
    return {
        "task": task,
        "seeds": seeds,
        "pending_members": [
            (idx, int(seed)) for idx, seed in enumerate(seeds, start=1)
        ],
        "active_members": set(),
        "results": {},
        "aggregated": False,
        "per_fold_cap": int(per_fold_cap),
        "rate_sec_per_trial": None,
        "remaining_trial_units": int(len(seeds))
        * int(dict(task.get("config") or {}).get("trials_per_fold", 0) or 0),
    }


def _seed_scheduler_acquire_member(state: dict) -> tuple[int, int] | None:
    """Reserve the first pending seed before submitting its worker future."""
    pending_members = list(state.get("pending_members") or [])
    if not pending_members:
        return None
    member_index, seed = pending_members.pop(0)
    state["pending_members"] = pending_members
    state.setdefault("active_members", set()).add(int(member_index))
    return int(member_index), int(seed)


def _seed_scheduler_priority_ready(
    states: list[dict], *, warmup_elapsed: float, warmup_max_sec: float
) -> bool:
    """Do not redistribute idle capacity before rates or warmup are available."""
    all_started_folds_have_rate = all(
        state.get("rate_sec_per_trial") is not None
        or not set(state.get("active_members") or set())
        for state in states
    )
    return bool(all_started_folds_have_rate) or warmup_elapsed >= float(warmup_max_sec)


def _seed_scheduler_pending_fold_ids(states: list[dict]) -> set[int]:
    """Keep the original live-board pending-fold semantics and fold identities."""
    return {
        int(dict(state.get("task") or {}).get("fold_idx", 0) or 0)
        for state in states
        if not bool(state.get("aggregated"))
    }
