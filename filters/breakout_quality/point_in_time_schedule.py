"""Canonical calendar schedule helpers for Breakout Quality PIT scoring."""

from __future__ import annotations

from typing import Any

import pandas as pd

FOLD_CALENDAR_ANCHOR = pd.Timestamp("2000-01-01")


def stable_point_in_time_fold_id(
    score_start: pd.Timestamp, score_end: pd.Timestamp
) -> str:
    return f"fold_{score_start:%Y%m%d}_{score_end:%Y%m%d}"


def build_point_in_time_fold_periods(
    score_start: pd.Timestamp | str,
    score_end: pd.Timestamp | str,
    *,
    fold_months: int,
    fold_anchor: pd.Timestamp | str | None = None,
    single_score_block: bool = False,
) -> list[dict[str, Any]]:
    """Build the canonical calendar-anchored PIT score folds.

    The first fold may be partial when the requested start falls inside an anchored
    bucket.  Later boundaries stay anchored, so prepending older history never
    renames or retrains an already-defined later fold.
    """

    start = pd.Timestamp(score_start).normalize()
    end = pd.Timestamp(score_end).normalize()
    months = int(fold_months)
    if months < 1:
        raise ValueError("fold_months必須>=1")
    if end < start:
        raise ValueError("point-in-time score期間不合法")
    if bool(single_score_block):
        return [{
            "fold_id": stable_point_in_time_fold_id(start, end),
            "score_start": start,
            "score_end": end,
        }]

    anchor = (
        FOLD_CALENDAR_ANCHOR
        if fold_anchor is None
        else pd.Timestamp(fold_anchor).normalize()
    )
    month_delta = (start.year - anchor.year) * 12 + (start.month - anchor.month)
    bucket_index = month_delta // months
    bucket_start = (anchor + pd.DateOffset(months=bucket_index * months)).normalize()
    bucket_end = (
        bucket_start + pd.DateOffset(months=months) - pd.Timedelta(days=1)
    ).normalize()

    folds: list[dict[str, Any]] = []
    cursor = start
    while cursor <= end:
        fold_end = min(end, bucket_end)
        folds.append(
            {
                "fold_id": stable_point_in_time_fold_id(cursor, fold_end),
                "score_start": cursor,
                "score_end": fold_end,
            }
        )
        cursor = (bucket_end + pd.Timedelta(days=1)).normalize()
        bucket_start = cursor
        bucket_end = (
            bucket_start + pd.DateOffset(months=months) - pd.Timedelta(days=1)
        ).normalize()
    if not folds:
        raise ValueError("point-in-time score期間沒有任何fold")
    return folds


__all__ = [
    "FOLD_CALENDAR_ANCHOR",
    "build_point_in_time_fold_periods",
    "stable_point_in_time_fold_id",
]
