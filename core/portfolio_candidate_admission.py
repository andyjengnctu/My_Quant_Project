"""Optional candidate admission at the canonical resource-selection boundary.

No research/target name, future truth or broker rule belongs here. The
ordinary selector runs unchanged when no explicitly scoped predicate is set.
This capability applies *before* resource reservation, not as a score penalty.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator, Mapping

_ADMISSION: ContextVar[Callable[[Mapping], bool] | None] = ContextVar(
    "resource_selection_admission", default=None,
)


@contextmanager
def scoped_candidate_admission(predicate: Callable[[Mapping], bool]) -> Iterator[None]:
    if not callable(predicate):
        raise TypeError("candidate admission policy must be callable")
    if _ADMISSION.get() is not None:
        raise RuntimeError("nested candidate admission policy is forbidden")
    token = _ADMISSION.set(predicate)
    try:
        yield
    finally:
        _ADMISSION.reset(token)


def admitted_resource_candidates(rows):
    candidates = list(rows or [])
    predicate = _ADMISSION.get()
    if predicate is None:
        return candidates
    return [row for row in candidates if predicate(row)]
