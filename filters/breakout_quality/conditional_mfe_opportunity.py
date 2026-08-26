"""Canonical reverse-conditional MFE target for MR-13Q / MR-13R.

The transform is supervision-only.  It asks how much same-date Pure-MFE rank
remains unexplained after conditioning on same-date low-adverse Safety rank:

    J = U - E(U | S)

where U is Pure-MFE percentile and S is low-adverse percentile.  The OLS
residual is rank-normalized again within date so the final ranking target stays
on the canonical [0, 1] scale.  Future labels never become inference features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from filters.breakout_quality.continuous_ranker_data import (
    build_same_date_percentile_targets,
)

HMHS_HIGH_PERCENTILE_CUTOFF = 0.50


@dataclass(frozen=True)
class ConditionalMfeOpportunityTargets:
    """Aligned reverse-conditional targets shared by both A/B model variants."""

    primary_mfe_percentile: np.ndarray
    low_adverse_safety_percentile: np.ndarray
    conditional_mfe_residual: np.ndarray
    conditional_mfe_percentile: np.ndarray

    @property
    def single_head_training_target(self) -> np.ndarray:
        return np.asarray(self.conditional_mfe_percentile, dtype=np.float32)

    @property
    def duo_head_training_target(self) -> np.ndarray:
        # Head order is explicit: raw Safety condition first, final Conditional-MFE second.
        return np.column_stack(
            [self.low_adverse_safety_percentile, self.conditional_mfe_percentile]
        ).astype(np.float32, copy=False)

    @property
    def safety_raw_mfe_training_target(self) -> np.ndarray:
        # MR-13S controlled contrast keeps the identical Safety head/context but
        # replaces residual J with the absolute same-date Pure-MFE percentile U.
        return np.column_stack(
            [self.low_adverse_safety_percentile, self.primary_mfe_percentile]
        ).astype(np.float32, copy=False)

    @property
    def direct_hmhs_target(self) -> np.ndarray:
        return (
            (self.low_adverse_safety_percentile >= HMHS_HIGH_PERCENTILE_CUTOFF)
            & (self.primary_mfe_percentile >= HMHS_HIGH_PERCENTILE_CUTOFF)
        ).astype(np.float32)

    @property
    def safety_raw_mfe_hmhs_training_target(self) -> np.ndarray:
        # MR-13T keeps MR-13S's two marginal targets intact and adds one direct
        # upper-right intersection target.  The joint head receives no score
        # arithmetic or strategy state; future S/U truth is supervision only.
        return np.column_stack(
            [
                self.low_adverse_safety_percentile,
                self.primary_mfe_percentile,
                self.direct_hmhs_target,
            ]
        ).astype(np.float32, copy=False)


def build_conditional_mfe_opportunity_targets(
    group_table: pd.DataFrame,
    valid_mask: np.ndarray,
    *,
    primary_mfe_percentile: np.ndarray | None = None,
) -> ConditionalMfeOpportunityTargets:
    """Build ``J = U - E(U|S)`` without thresholds, buckets or OOS fitting."""

    required = {"date", "target_favorable_r", "target_adverse_r"}
    missing = sorted(required - set(group_table.columns))
    if missing:
        raise ValueError(f"conditional-MFE target缺少canonical欄位: {missing}")

    valid = np.asarray(valid_mask, dtype=bool)
    if valid.ndim != 1 or len(valid) != len(group_table):
        raise ValueError("conditional-MFE valid_mask shape不一致")

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
            "conditional-MFE valid rows必須同時具有finite favorable/adverse target"
        )

    if primary_mfe_percentile is None:
        mfe_percentile = build_same_date_percentile_targets(favorable, valid, dates)
    else:
        mfe_percentile = np.asarray(primary_mfe_percentile, dtype=np.float32)
        if mfe_percentile.shape != valid.shape:
            raise ValueError("conditional-MFE primary percentile shape不一致")
        if bool(np.any(valid & ~np.isfinite(mfe_percentile))):
            raise ValueError("conditional-MFE valid row缺少primary MFE percentile")
        finite = np.isfinite(mfe_percentile)
        if bool(np.any(finite & ((mfe_percentile < 0.0) | (mfe_percentile > 1.0)))):
            raise ValueError("conditional-MFE primary percentile超出[0,1]")

    safety_percentile = build_same_date_percentile_targets(-adverse, valid, dates)

    residual = np.full(len(group_table), np.nan, dtype=np.float64)
    work = pd.DataFrame(
        {
            "date": dates,
            "u": np.asarray(mfe_percentile, dtype=np.float64),
            "s": np.asarray(safety_percentile, dtype=np.float64),
            "group_index": np.arange(len(group_table), dtype=np.int64),
        }
    )
    work = work[valid].copy()
    for _date, day in work.groupby("date", sort=True):
        positions = day["group_index"].to_numpy(dtype=np.int64)
        u = day["u"].to_numpy(dtype=np.float64)
        s = day["s"].to_numpy(dtype=np.float64)
        s_centered = s - float(np.mean(s))
        u_mean = float(np.mean(u))
        denominator = float(np.dot(s_centered, s_centered))
        if denominator <= np.finfo(np.float64).eps:
            fitted = np.full_like(u, u_mean)
        else:
            slope = float(np.dot(s_centered, u - u_mean) / denominator)
            fitted = u_mean + slope * s_centered
        residual[positions] = u - fitted

    if bool(np.any(valid & ~np.isfinite(residual))):
        raise ValueError("conditional-MFE residual產生non-finite value")

    conditional_percentile = build_same_date_percentile_targets(residual, valid, dates)
    for array in (mfe_percentile, safety_percentile, conditional_percentile):
        if bool(np.any(valid & ~np.isfinite(array))):
            raise ValueError("conditional-MFE valid target不完整")

    return ConditionalMfeOpportunityTargets(
        primary_mfe_percentile=np.asarray(mfe_percentile, dtype=np.float32),
        low_adverse_safety_percentile=np.asarray(safety_percentile, dtype=np.float32),
        conditional_mfe_residual=residual.astype(np.float32),
        conditional_mfe_percentile=np.asarray(conditional_percentile, dtype=np.float32),
    )


__all__ = [
    "ConditionalMfeOpportunityTargets",
    "HMHS_HIGH_PERCENTILE_CUTOFF",
    "build_conditional_mfe_opportunity_targets",
]
