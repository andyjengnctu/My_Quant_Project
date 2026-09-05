"""Canonical date-index planning for same-date cross-stock relation models."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


def normalize_relation_date_key(value) -> date:
    """Return a stable Python date key for relational batch planning."""

    return pd.Timestamp(value).normalize().date()


def build_previous_relation_date_indices(
    group_dates: pd.Series | np.ndarray,
    requested_group_ids: np.ndarray,
    *,
    history_steps: int,
) -> dict[date, np.ndarray]:
    """Map each requested decision date to the full prior-date stock universe.

    Relation history is input-only.  The prior universe is resolved from all
    score-eligible group rows, independent of target/split membership, and may
    only move backward in the canonical trading-date sequence.
    """

    steps = int(history_steps)
    if steps < 0:
        raise ValueError("same-date relation history_steps不得為負")
    if steps == 0:
        return {}
    if steps != 1:
        raise ValueError("same-date relation history目前只支援previous trading date (1 step)")

    dates = pd.to_datetime(pd.Series(group_dates), errors="raise").dt.normalize()
    ids = np.asarray(requested_group_ids, dtype=np.int64).reshape(-1)
    if len(ids) == 0:
        return {}
    if bool(np.any(ids < 0)) or int(ids.max()) >= len(dates):
        raise IndexError("same-date relation requested_group_ids超出group_dates範圍")

    values = dates.to_numpy(dtype="datetime64[D]")
    unique_dates = np.unique(values)
    position_by_date = {
        normalize_relation_date_key(value): int(position)
        for position, value in enumerate(unique_dates.tolist())
    }
    groups_by_date: dict[date, np.ndarray] = {}
    for value in unique_dates:
        key = normalize_relation_date_key(value)
        groups_by_date[key] = np.flatnonzero(values == value).astype(np.int64, copy=False)

    result: dict[date, np.ndarray] = {}
    for value in np.unique(values[ids]):
        key = normalize_relation_date_key(value)
        position = position_by_date[key]
        if position < steps:
            result[key] = np.empty(0, dtype=np.int64)
            continue
        previous_key = normalize_relation_date_key(unique_dates[position - steps])
        previous_ids = groups_by_date[previous_key]
        if len(previous_ids) and not bool(np.all(values[previous_ids] < np.datetime64(value, "D"))):
            raise AssertionError("same-date relation history不得包含current/future date")
        result[key] = previous_ids
    return result


__all__ = [
    "build_previous_relation_date_indices",
    "normalize_relation_date_key",
]
