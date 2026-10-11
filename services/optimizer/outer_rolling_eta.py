"""Presentation-only ETA for a rolling fold x seed optimizer schedule.

All inputs come from existing progress logs and the execution scheduler.
No ETA value participates in fitting, worker placement, artifacts or identity.
"""

from __future__ import annotations

import heapq
import math
import statistics
import time
from datetime import datetime

from core.runtime_utils import TAIPEI_TIMEZONE
from services.optimizer.outer_rolling_formatting import _fmt_duration


_POST_SEARCH_REPLAY_FALLBACK_SEC = 30.0


def _finite_positive(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result > 0.0 else None


def _positive_median(values) -> float | None:
    good = [v for v in (_finite_positive(x) for x in values) if v is not None]
    return float(statistics.median(good)) if good else None


def _fold_parallel_tail(durations: list[float], slots: int) -> float:
    """Lower-bound-ish fold tail when members have a per-fold worker cap."""
    if not durations:
        return 0.0
    workers = max(1, min(int(slots), len(durations)))
    heap = [0.0] * workers
    heapq.heapify(heap)
    for duration in sorted(durations, reverse=True):
        next_free = heapq.heappop(heap)
        heapq.heappush(heap, next_free + max(0.0, duration))
    return max(heap)


def estimate_rolling_seed_eta(
    *,
    tasks: list[dict],
    seed_progress_by_fold: dict[int, dict[int, dict]],
    fold_progress_by_fold: dict[int, dict],
    completed_oos_years: set[int],
    global_workers: int,
    per_fold_cap: int,
    replay_policy_slots: int,
    now_ts: float | None = None,
) -> dict:
    """Estimate *current parameter build* remaining wall time, not Compare ETA.

    Includes queued/active seed search, observed seed post-search overhead, and
    remaining parent-side replay. Missing rate samples return a calibrating
    state rather than a misleading numeric value. Replay without observations
    uses a clearly labelled presentation-only fallback.
    """
    now = float(time.time() if now_ts is None else now_ts)
    all_rates: list[float] = []
    done_overheads: list[float] = []
    folds: list[dict] = []
    completed_oos = {int(value) for value in completed_oos_years}

    for task in tasks:
        idx = int(task.get("fold_idx", 0) or 0)
        year = int(task.get("oos_year", 0) or 0)
        done_fold = year in completed_oos
        trials = max(1, int(dict(task.get("config") or {}).get("trials_per_fold", 0) or 1))
        seed_policy = dict(task.get("seed_ensemble_policy") or {})
        count = max(1, int(seed_policy.get("seed_count", 0) or len(seed_policy.get("resolved_seeds") or []) or 1))
        progresses = dict(seed_progress_by_fold.get(idx) or {})
        fold_rates: list[float] = []
        members: list[dict] = []

        for member in range(1, count + 1):
            progress = dict(progresses.get(member) or {})
            stage = str(progress.get("stage") or "QUEUED").upper()
            finished = done_fold or stage in {"DONE", "FOLD_RESULT"}
            completed = max(0, min(trials, int(progress.get("completed", 0) or 0)))
            if finished or stage in {"LOCAL_MIN_REVIEW", "OOS_DIAGNOSTICS"}:
                completed = trials
            elapsed = _finite_positive(progress.get("search_active_elapsed_sec"))
            if completed >= 8 and elapsed is not None:
                rate = elapsed / completed
                all_rates.append(rate)
                fold_rates.append(rate)
            if finished and elapsed is not None:
                whole = _finite_positive(progress.get("elapsed_sec"))
                if whole is not None and whole > elapsed:
                    done_overheads.append(whole - elapsed)
            members.append({"remaining_trials": 0 if finished else max(0, trials - completed), "finished": finished, "stage": stage})

        folds.append({
            "task": task,
            "members": members,
            "fold_rate": _positive_median(fold_rates),
            "finished": done_fold,
            "replay": dict(fold_progress_by_fold.get(idx) or {}),
        })

    if not folds or all(fold["finished"] for fold in folds):
        return {"status": "done", "remaining_sec": 0.0, "finish_ts": now, "confidence": "measured"}

    fallback_rate = _positive_median(all_rates)
    if fallback_rate is None:
        return {"status": "calibrating", "remaining_sec": None, "finish_ts": None, "confidence": "low"}

    post_search = _positive_median(done_overheads) or 0.0
    fold_tails: list[float] = []
    outstanding_units: list[float] = []
    for fold in folds:
        if fold["finished"]:
            continue
        rate = fold["fold_rate"] or fallback_rate
        durations: list[float] = []
        for item in fold["members"]:
            if item["finished"]:
                continue
            trials_left = item["remaining_trials"]
            stage = item["stage"]
            # A member in post-search still needs its remaining tail, not 300
            # more optimizer trials. No default extra time is invented until
            # completed members supply a real overhead sample.
            overhead = post_search if trials_left > 0 or stage not in {"DONE"} else 0.0
            duration = max(0.0, trials_left * rate + overhead)
            if duration > 0.0:
                durations.append(duration)
                outstanding_units.append(duration)
        fold_tails.append(_fold_parallel_tail(durations, per_fold_cap))

    workers = max(1, int(global_workers))
    search_remaining = max(
        sum(outstanding_units) / workers,
        max(fold_tails, default=0.0),
    )

    replay_samples: list[float] = []
    replay_remaining_count = 0
    for fold in folds:
        if fold["finished"]:
            continue
        progress = fold["replay"]
        try:
            done = max(0, int(progress.get("replay_done", 0) or 0))
            total = int(progress.get("replay_total", 0) or 0)
        except (ValueError, TypeError):
            done, total = 0, 0
        total = total if total > 0 else max(0, int(replay_policy_slots))
        replay_remaining_count += max(0, total - done)
        started = _finite_positive(progress.get("replay_started_ts"))
        ended = _finite_positive(progress.get("replay_last_done_ts"))
        if done > 0 and started is not None and ended is not None and ended > started:
            replay_samples.append((ended - started) / done)

    replay_rate = _positive_median(replay_samples)
    replay_pending = replay_remaining_count * (replay_rate or _POST_SEARCH_REPLAY_FALLBACK_SEC)
    estimate_sec = max(0.0, search_remaining + replay_pending)
    return {
        "status": "estimated",
        "remaining_sec": estimate_sec,
        "finish_ts": now + estimate_sec,
        "search_remaining_sec": search_remaining,
        "replay_remaining_sec": replay_pending,
        "confidence": "dynamic" if replay_rate is not None else "initial",
    }



def estimate_rolling_fold_eta(
    *,
    tasks: list[dict],
    fold_progress_by_fold: dict[int, dict],
    completed_oos_years: set[int],
    fold_workers: int,
    now_ts: float | None = None,
) -> dict:
    """Presentation-only ETA for ordinary parallel folds (one seed each).

    Trials are concurrent across folds, while per-fold post-search diagnostics
    run in the same worker.  This must not use the multi-seed parent Replay
    accounting: single-seed folds do not have an ensemble Replay phase.
    """
    now = float(time.time() if now_ts is None else now_ts)
    done_years = {int(year) for year in completed_oos_years}
    rates: list[float] = []
    post_search_samples: list[float] = []
    states: list[dict] = []
    for task in tasks:
        fold_idx = int(task.get("fold_idx", 0) or 0)
        year = int(task.get("oos_year", 0) or 0)
        progress = dict(fold_progress_by_fold.get(fold_idx) or {})
        stage = str(progress.get("stage") or "QUEUED").upper()
        finished = year in done_years or stage in {"DONE", "FOLD_RESULT"}
        trial_budget = max(1, int(dict(task.get("config") or {}).get("trials_per_fold", 0) or 1))
        completed = max(0, min(trial_budget, int(progress.get("completed", 0) or 0)))
        if finished or stage in {"LOCAL_MIN_REVIEW", "OOS_DIAGNOSTICS"}:
            completed = trial_budget
        search_elapsed = _finite_positive(progress.get("search_active_elapsed_sec"))
        rate = search_elapsed / completed if search_elapsed is not None and completed >= 8 else None
        if rate is not None:
            rates.append(rate)
        if finished:
            whole_elapsed = _finite_positive(progress.get("elapsed_sec"))
            if search_elapsed is not None and whole_elapsed is not None and whole_elapsed > search_elapsed:
                post_search_samples.append(whole_elapsed - search_elapsed)
        states.append({
            "finished": finished,
            "stage": stage,
            "remaining_trials": max(0, trial_budget - completed),
            "rate": rate,
        })

    if not states or all(item["finished"] for item in states):
        return {"status": "done", "remaining_sec": 0.0, "finish_ts": now, "confidence": "measured"}
    fallback_rate = _positive_median(rates)
    if fallback_rate is None:
        return {"status": "calibrating", "remaining_sec": None, "finish_ts": None, "confidence": "low"}

    measured_overhead = _positive_median(post_search_samples)
    overhead = measured_overhead if measured_overhead is not None else _POST_SEARCH_REPLAY_FALLBACK_SEC
    durations = [
        float(item["remaining_trials"]) * (item["rate"] or fallback_rate) + overhead
        for item in states if not item["finished"]
    ]
    remaining = _fold_parallel_tail(durations, max(1, int(fold_workers)))
    return {
        "status": "estimated",
        "remaining_sec": remaining,
        "finish_ts": now + remaining,
        "confidence": "dynamic" if measured_overhead is not None else "initial",
    }


def format_rolling_seed_eta(estimate: dict) -> str:
    """A short, explicitly approximate suffix for the existing progress header."""
    status = str((estimate or {}).get("status") or "calibrating")
    if status == "done":
        return "ETA\u5269\u9918=00:00:00"
    if status != "estimated":
        return "ETA\u5269\u9918=\u6821\u6e96\u4e2d"
    remaining = _finite_positive(estimate.get("remaining_sec"))
    if remaining is None:
        return "ETA\u5269\u9918=\u6821\u6e96\u4e2d"
    rounded = int(math.ceil(remaining / 60.0) * 60)
    finish = datetime.fromtimestamp(float(estimate["finish_ts"]) + rounded - remaining, tz=TAIPEI_TIMEZONE)
    reliability = "\u521d\u4f30" if estimate.get("confidence") == "initial" else "\u52d5\u614b"
    return f"ETA\u5269\u9918~{_fmt_duration(rounded)} ({reliability}) | \u9810\u8a08\u5b8c\u6210~{finish:%m-%d %H:%M} (+08)"
