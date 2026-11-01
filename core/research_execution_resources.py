"""Execution-only CPU lanes inherited by isolated Research producers."""
from __future__ import annotations

import os
from config.runtime import RESEARCH_CPU_WORKER_BUDGET

CPU_LANES_ENV = "MY_QUANT_RESEARCH_CPU_LANES"


def research_cpu_lanes() -> int:
    """One budget for a seed coordinator and its nested ticker processes."""
    budget = RESEARCH_CPU_WORKER_BUDGET
    if type(budget) is not int or budget < 1:
        raise ValueError("Research CPU worker budget must be a positive integer")
    count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    inherited = os.environ.get(CPU_LANES_ENV)
    if inherited is not None:
        limit = int(inherited)
        if limit < 1:
            raise ValueError("Inherited Research CPU lanes must be positive")
        budget = min(budget, limit)
    return min(budget, count)


def research_ticker_workers(requested: int) -> int:
    if type(requested) is not int or requested < 1:
        raise ValueError("Research ticker workers must be a positive integer")
    lanes = research_cpu_lanes()
    # A one-lane producer computes serially. With a pool, retain one lane for
    # its coordinating parent, so nested seed/ticker processes remain bounded.
    return min(requested, max(1, lanes - 1))
