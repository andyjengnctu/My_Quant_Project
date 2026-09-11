"""Lightweight Market Data V2 bootstrap progress helpers.

This module deliberately has no application/menu imports.  Bootstrap execution,
console rendering and synthetic contracts may share these calculations without
pulling an interactive composition root into their import graph.
"""

from __future__ import annotations


def format_bootstrap_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds == float("inf"):
        return "--:--:--"
    total_seconds = int(round(seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours < 100:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours}h{minutes:02d}m"


def estimate_bootstrap_eta_seconds(
    *,
    done: int,
    initial_done: int,
    total: int,
    elapsed_seconds: float,
    quota_wait_seconds: float = 0.0,
    quota_limit: int | None,
    quota_reserve: int | None,
    observed_sample_floor: int = 1,
) -> float | None:
    """Estimate remaining wall time without treating quota wait as slow I/O.

    Provider quota is an upper bound on sustainable throughput.  Observed active
    throughput is used only after enough jobs have completed in this process;
    quota-wait time is excluded from that active-rate sample.
    """

    remaining = max(0, int(total) - int(done))
    if remaining == 0:
        return 0.0
    process_done = max(0, int(done) - int(initial_done))
    active_elapsed = max(0.0, float(elapsed_seconds) - max(0.0, float(quota_wait_seconds)))
    observed_rate = None
    if process_done >= max(1, int(observed_sample_floor)) and active_elapsed > 0:
        observed_rate = process_done / active_elapsed
    quota_rate = None
    if quota_limit is not None and int(quota_limit) > 0:
        safe_per_hour = max(1, int(quota_limit) - max(0, int(quota_reserve or 0)))
        quota_rate = safe_per_hour / 3600.0
    rates = [rate for rate in (observed_rate, quota_rate) if rate is not None and rate > 0]
    if not rates:
        return None
    return remaining / min(rates)


__all__ = ["estimate_bootstrap_eta_seconds", "format_bootstrap_duration"]
