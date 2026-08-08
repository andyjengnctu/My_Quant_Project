"""Shared descriptive ranking-quality metrics for continuous breakout-quality rankers."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def daily_top_k_metrics(
    dates: np.ndarray,
    scores: np.ndarray,
    raw_targets: np.ndarray,
    percentile_targets: np.ndarray,
    *,
    top_k: int,
    boundary_width: int,
) -> dict[str, Any]:
    """Measure same-day Top-K and score-boundary quality with equal date weighting.

    NDCG uses the canonical same-day percentile target as non-negative relevance.
    Economic lift/gap metrics use the raw strategy-aligned R target.  The K-boundary
    compares the score ranks immediately inside K against the same number immediately
    outside K, so it measures the ordering most likely to change a portfolio decision.
    Only candidate_count > K competition days enter the primary Top-K metrics.
    """

    requested_k = int(top_k)
    requested_boundary_width = int(boundary_width)
    if requested_k < 1:
        raise ValueError("continuous ranker report top_k必須>=1")
    if requested_boundary_width < 1:
        raise ValueError("continuous ranker report boundary_width必須>=1")

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "score": np.asarray(scores, dtype=np.float64),
            "raw_target": np.asarray(raw_targets, dtype=np.float64),
            "percentile_target": np.asarray(percentile_targets, dtype=np.float64),
        }
    )
    finite = (
        np.isfinite(frame["score"].to_numpy(dtype=np.float64))
        & np.isfinite(frame["raw_target"].to_numpy(dtype=np.float64))
        & np.isfinite(frame["percentile_target"].to_numpy(dtype=np.float64))
    )
    frame = frame.loc[finite].copy()

    ndcg_values: list[float] = []
    top_k_target_values: list[float] = []
    candidate_target_values: list[float] = []
    top_k_lift_values: list[float] = []
    oracle_overlap_values: list[float] = []
    boundary_concordance_values: list[float] = []
    boundary_inside_target_values: list[float] = []
    boundary_outside_target_values: list[float] = []
    boundary_gap_values: list[float] = []
    all_date_count = 0
    excluded_non_competition_date_count = 0

    for _date, day in frame.groupby("date", sort=True):
        if day.empty:
            continue
        all_date_count += 1
        score_values = day["score"].to_numpy(dtype=np.float64)
        raw_values = day["raw_target"].to_numpy(dtype=np.float64)
        percentile_values = day["percentile_target"].to_numpy(dtype=np.float64)
        count = int(len(day))
        if count <= requested_k:
            excluded_non_competition_date_count += 1
            continue

        k = requested_k
        score_order = np.argsort(-score_values, kind="mergesort")
        target_order = np.argsort(-raw_values, kind="mergesort")
        selected = score_order[:k]
        oracle = target_order[:k]

        discounts = 1.0 / np.log2(np.arange(k, dtype=np.float64) + 2.0)
        actual_dcg = float(np.sum(percentile_values[selected] * discounts))
        ideal_relevance = np.sort(percentile_values)[::-1][:k]
        ideal_dcg = float(np.sum(ideal_relevance * discounts))
        if ideal_dcg > 0.0:
            ndcg_values.append(float(actual_dcg / ideal_dcg))

        selected_mean = float(raw_values[selected].mean())
        candidate_mean = float(raw_values.mean())
        top_k_target_values.append(selected_mean)
        candidate_target_values.append(candidate_mean)
        top_k_lift_values.append(float(selected_mean - candidate_mean))
        oracle_overlap_values.append(
            float(len(set(int(i) for i in selected).intersection(int(i) for i in oracle)) / k)
        )

        width = min(requested_boundary_width, requested_k, count - requested_k)
        if width < 1:
            continue
        inside = score_order[requested_k - width : requested_k]
        outside = score_order[requested_k : requested_k + width]
        inside_targets = raw_values[inside]
        outside_targets = raw_values[outside]
        target_diff = inside_targets[:, None] - outside_targets[None, :]
        comparable_count = int(target_diff.size)
        if comparable_count:
            concordant = float((target_diff > 0.0).sum()) + 0.5 * float((target_diff == 0.0).sum())
            boundary_concordance_values.append(float(concordant / comparable_count))
        inside_mean = float(inside_targets.mean())
        outside_mean = float(outside_targets.mean())
        boundary_inside_target_values.append(inside_mean)
        boundary_outside_target_values.append(outside_mean)
        boundary_gap_values.append(float(inside_mean - outside_mean))

    def mean_or_none(values: list[float]) -> float | None:
        return float(np.mean(values)) if values else None

    competition_date_count = int(len(top_k_target_values))
    return {
        "top_k": requested_k,
        "boundary_width": requested_boundary_width,
        "all_date_count": int(all_date_count),
        "top_k_date_count": competition_date_count,
        "competition_date_count": competition_date_count,
        "excluded_non_competition_date_count": int(excluded_non_competition_date_count),
        "competition_rule": "candidate_count_gt_top_k",
        "ndcg_at_k": mean_or_none(ndcg_values),
        "top_k_raw_target_mean": mean_or_none(top_k_target_values),
        "candidate_raw_target_mean": mean_or_none(candidate_target_values),
        "top_k_raw_target_lift": mean_or_none(top_k_lift_values),
        "oracle_top_k_overlap": mean_or_none(oracle_overlap_values),
        "boundary_date_count": int(len(boundary_gap_values)),
        "boundary_concordance": mean_or_none(boundary_concordance_values),
        "boundary_inside_raw_target_mean": mean_or_none(boundary_inside_target_values),
        "boundary_outside_raw_target_mean": mean_or_none(boundary_outside_target_values),
        "boundary_raw_target_gap": mean_or_none(boundary_gap_values),
        "date_weighting": "equal_date_weight",
        "top_k_scope": "competition_days_only",
        "ndcg_relevance": "same_day_percentile_target",
        "economic_target": "raw_strategy_aligned_r",
    }


def exact_random_top_k_baseline(
    percentile_targets: np.ndarray,
    raw_targets: np.ndarray,
    *,
    top_k: int,
) -> dict[str, float | None]:
    """Return the exact expectation under a uniformly random within-day ordering."""

    relevance = np.asarray(percentile_targets, dtype=np.float64)
    raw = np.asarray(raw_targets, dtype=np.float64)
    valid = np.isfinite(relevance) & np.isfinite(raw)
    relevance = relevance[valid]
    raw = raw[valid]
    count = int(len(raw))
    k = int(top_k)
    if k < 1 or count <= k:
        return {
            "ndcg_at_k": None,
            "top_k_raw_target_mean": None,
            "candidate_raw_target_mean": None,
            "top_k_raw_target_lift": None,
            "oracle_top_k_overlap": None,
            "boundary_concordance": None,
            "boundary_raw_target_gap": None,
        }
    discounts = 1.0 / np.log2(np.arange(k, dtype=np.float64) + 2.0)
    ideal_relevance = np.sort(relevance)[::-1][:k]
    ideal_dcg = float(np.sum(ideal_relevance * discounts))
    expected_dcg = float(np.mean(relevance) * np.sum(discounts))
    candidate_mean = float(np.mean(raw))
    return {
        "ndcg_at_k": (float(expected_dcg / ideal_dcg) if ideal_dcg > 0.0 else None),
        "top_k_raw_target_mean": candidate_mean,
        "candidate_raw_target_mean": candidate_mean,
        "top_k_raw_target_lift": 0.0,
        "oracle_top_k_overlap": float(k / count),
        "boundary_concordance": 0.5,
        "boundary_raw_target_gap": 0.0,
    }


__all__ = ["daily_top_k_metrics", "exact_random_top_k_baseline"]
