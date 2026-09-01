"""Generic Safety + primary-target supervision for shared two-head rankers.

The primary target is supplied by the experiment profile as a same-date percentile.
Safety remains the canonical low-adverse same-date percentile.  This capability makes no
assumption that the primary target is Pure-MFE, so economic-target controls cannot be
silently reported or selected as Raw-MFE experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from filters.breakout_quality.continuous_ranker_data import (
    build_same_date_percentile_targets,
)


@dataclass(frozen=True)
class SafetyPrimaryTargets:
    low_adverse_safety_percentile: np.ndarray
    primary_target_percentile: np.ndarray

    @property
    def training_target(self) -> np.ndarray:
        return np.column_stack(
            [self.low_adverse_safety_percentile, self.primary_target_percentile]
        ).astype(np.float32, copy=False)


def build_safety_primary_targets(
    group_table: pd.DataFrame,
    primary_target_percentile: np.ndarray,
) -> SafetyPrimaryTargets:
    primary = np.asarray(primary_target_percentile, dtype=np.float32)
    if primary.ndim != 1 or len(primary) != len(group_table):
        raise ValueError("Safety+Primary target percentile shape不一致")
    valid = np.isfinite(primary)
    if bool(np.any(valid & ((primary < 0.0) | (primary > 1.0)))):
        raise ValueError("Safety+Primary target percentile超出[0,1]")
    if "date" not in group_table.columns or "target_adverse_r" not in group_table.columns:
        raise ValueError("Safety+Primary target缺少date/target_adverse_r canonical欄位")

    dates = pd.to_datetime(group_table["date"], errors="raise").dt.normalize()
    adverse = pd.to_numeric(
        group_table["target_adverse_r"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if bool(np.any(valid & ~np.isfinite(adverse))):
        raise ValueError("Safety+Primary valid row缺少finite adverse target")
    safety = build_same_date_percentile_targets(-adverse, valid, dates)
    if bool(np.any(valid & ~np.isfinite(safety))):
        raise ValueError("Safety+Primary valid row缺少Safety percentile")
    return SafetyPrimaryTargets(
        low_adverse_safety_percentile=np.asarray(safety, dtype=np.float32),
        primary_target_percentile=primary,
    )


__all__ = ["SafetyPrimaryTargets", "build_safety_primary_targets"]
