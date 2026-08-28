"""Canonical label transform for single-model conditional MFE-safety research.

The transform is supervision-only: future Pure-MFE and adverse-to-peak labels are
used to define a same-date conditional residual target.  No future value becomes
an inference feature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from filters.breakout_quality.continuous_ranker_data import (
    build_same_date_percentile_targets,
)


@dataclass(frozen=True)
class ConditionalMfeSafetyTargets:
    """Aligned target arrays for MR-13P.

    ``primary_mfe_percentile`` remains the Pure-MFE primary ranking target.
    ``conditional_safety_percentile`` is the rank-normalized residual of
    low-adverse percentile after removing its same-date linear expectation given
    the *true* Pure-MFE percentile.  The residual is supervision only.
    """

    primary_mfe_percentile: np.ndarray
    low_adverse_percentile: np.ndarray
    conditional_safety_residual: np.ndarray
    conditional_safety_percentile: np.ndarray

    @property
    def training_target(self) -> np.ndarray:
        return np.column_stack(
            [self.primary_mfe_percentile, self.conditional_safety_percentile]
        ).astype(np.float32, copy=False)


def build_same_date_residual_percentile(
    condition_percentile: np.ndarray,
    response_percentile: np.ndarray,
    valid_mask: np.ndarray,
    dates,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic same-date OLS residual and its average-rank percentile.

    This helper centralizes the exact conditional-ranking transform used by MR-13P
    and newer controlled conditional models.  It is supervision-only; callers own
    the semantics/provenance of both input percentiles.
    """

    condition = np.asarray(condition_percentile, dtype=np.float64)
    response = np.asarray(response_percentile, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if condition.shape != valid.shape or response.shape != valid.shape:
        raise ValueError("same-date residual percentile input shape不一致")
    if bool(np.any(valid & (~np.isfinite(condition) | ~np.isfinite(response)))):
        raise ValueError("same-date residual percentile valid row缺少finite condition/response")
    if bool(np.any(valid & ((condition < 0.0) | (condition > 1.0)))):
        raise ValueError("same-date residual percentile condition超出[0,1]")
    if bool(np.any(valid & ((response < 0.0) | (response > 1.0)))):
        raise ValueError("same-date residual percentile response超出[0,1]")

    normalized_dates = pd.to_datetime(pd.Series(dates), errors="raise").dt.normalize()
    if len(normalized_dates) != len(valid):
        raise ValueError("same-date residual percentile dates長度不一致")
    residual = np.full(len(valid), np.nan, dtype=np.float64)
    work = pd.DataFrame(
        {
            "date": normalized_dates,
            "condition": condition,
            "response": response,
            "group_index": np.arange(len(valid), dtype=np.int64),
        }
    )
    work = work[valid].copy()
    for _date, day in work.groupby("date", sort=True):
        positions = day["group_index"].to_numpy(dtype=np.int64)
        x = day["condition"].to_numpy(dtype=np.float64)
        y = day["response"].to_numpy(dtype=np.float64)
        x_centered = x - float(np.mean(x))
        y_mean = float(np.mean(y))
        denominator = float(np.dot(x_centered, x_centered))
        if denominator <= np.finfo(np.float64).eps:
            fitted = np.full_like(y, y_mean)
        else:
            slope = float(np.dot(x_centered, y - y_mean) / denominator)
            fitted = y_mean + slope * x_centered
        residual[positions] = y - fitted
    if bool(np.any(valid & ~np.isfinite(residual))):
        raise ValueError("same-date residual percentile產生non-finite residual")
    percentile = build_same_date_percentile_targets(residual, valid, normalized_dates)
    return residual.astype(np.float32), np.asarray(percentile, dtype=np.float32)


def build_conditional_mfe_safety_targets(
    group_table: pd.DataFrame,
    valid_mask: np.ndarray,
    *,
    primary_mfe_percentile: np.ndarray | None = None,
) -> ConditionalMfeSafetyTargets:
    """Build same-date continuous conditional safety labels without thresholds.

    For every date, define ``U`` as Pure-MFE percentile and ``S`` as low-adverse
    percentile.  Fit the deterministic same-date OLS relation ``S = a + b U``
    using only rows whose labels are valid for the requested split, then use the
    residual ``e = S - (a + b U)`` as conditional safety.  The residual itself
    is converted to the canonical average-rank same-date percentile so both
    model heads share the same [0,1] pairwise-target scale.

    This is a target transform, not a feature transform.  Future labels never
    enter model inference.
    """

    required = {"date", "target_favorable_r", "target_adverse_r"}
    missing = sorted(required - set(group_table.columns))
    if missing:
        raise ValueError(f"conditional MFE-safety target缺少canonical欄位: {missing}")

    valid = np.asarray(valid_mask, dtype=bool)
    if valid.ndim != 1 or len(valid) != len(group_table):
        raise ValueError("conditional MFE-safety valid_mask shape不一致")

    dates = pd.to_datetime(group_table["date"], errors="raise").dt.normalize()
    favorable = pd.to_numeric(
        group_table["target_favorable_r"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    adverse = pd.to_numeric(
        group_table["target_adverse_r"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    component_valid = valid & np.isfinite(favorable) & np.isfinite(adverse)
    if not np.array_equal(component_valid, valid):
        raise ValueError(
            "conditional MFE-safety valid rows必須同時具有finite favorable/adverse target"
        )

    if primary_mfe_percentile is None:
        mfe_percentile = build_same_date_percentile_targets(
            favorable, valid, dates
        )
    else:
        mfe_percentile = np.asarray(primary_mfe_percentile, dtype=np.float32)
        if mfe_percentile.shape != valid.shape:
            raise ValueError("conditional MFE-safety primary percentile shape不一致")
        if bool(np.any(valid & ~np.isfinite(mfe_percentile))):
            raise ValueError("conditional MFE-safety valid row缺少primary MFE percentile")
        if bool(np.any(np.isfinite(mfe_percentile) & ((mfe_percentile < 0.0) | (mfe_percentile > 1.0)))):
            raise ValueError("conditional MFE-safety primary MFE percentile超出[0,1]")

    low_adverse_percentile = build_same_date_percentile_targets(
        -adverse,
        valid,
        dates,
    )

    residual, conditional_percentile = build_same_date_residual_percentile(
        mfe_percentile,
        low_adverse_percentile,
        valid,
        dates,
    )

    for array in (
        mfe_percentile,
        low_adverse_percentile,
        conditional_percentile,
    ):
        if bool(np.any(valid & ~np.isfinite(array))):
            raise ValueError("conditional MFE-safety valid target不完整")

    return ConditionalMfeSafetyTargets(
        primary_mfe_percentile=np.asarray(mfe_percentile, dtype=np.float32),
        low_adverse_percentile=np.asarray(low_adverse_percentile, dtype=np.float32),
        conditional_safety_residual=residual.astype(np.float32),
        conditional_safety_percentile=np.asarray(conditional_percentile, dtype=np.float32),
    )


__all__ = [
    "ConditionalMfeSafetyTargets",
    "build_conditional_mfe_safety_targets",
    "build_same_date_residual_percentile",
]
