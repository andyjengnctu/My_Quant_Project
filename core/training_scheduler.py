"""Shared scheduling policy for isolated model-training units."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from typing import Any


def pop_next_seed_diverse_unit(
    pending_units: deque[dict[str, Any]],
    active_units: Iterable[Mapping[str, Any]],
    *,
    seed_key: str = "seed",
) -> dict[str, Any]:
    """Pop the next training unit, preferring a seed not already active.

    This is an execution-only policy.  When all pending units belong to an
    already-active seed (for example normal Strategy Compare with one seed),
    the next queued source is returned so spare GPU workers can run different
    DL sources concurrently.  Scientific identity/order inside each training
    unit is unchanged.
    """

    if not pending_units:
        raise IndexError("沒有待排程的training unit")
    active_seeds = {
        int(meta[seed_key])
        for meta in active_units
        if seed_key in meta and meta[seed_key] is not None
    }
    if active_seeds:
        for index, meta in enumerate(pending_units):
            if int(meta[seed_key]) not in active_seeds:
                pending_units.rotate(-index)
                selected = pending_units.popleft()
                pending_units.rotate(index)
                return selected
    return pending_units.popleft()


__all__ = ["pop_next_seed_diverse_unit"]
