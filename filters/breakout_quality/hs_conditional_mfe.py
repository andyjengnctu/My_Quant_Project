"""True-HS masked conditional-MFE supervision primitives.

The shared encoder and Safety head keep full-universe exposure.  The upside target is
materialized only inside the true High-Safety cohort, so Low-Safety rows never define
MFE ordering or full-list NDCG geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from filters.breakout_quality.continuous_ranker_data import build_same_date_percentile_targets

DEFAULT_TRUE_HS_PERCENTILE_CUTOFF = 0.50


@dataclass(frozen=True)
class HsConditionalMfeTargets:
    low_adverse_safety_percentile: np.ndarray
    primary_mfe_percentile: np.ndarray
    true_hs_mask: np.ndarray
    conditional_mfe_percentile: np.ndarray

    @property
    def training_target(self) -> np.ndarray:
        """Head order: full-universe Safety, then true-HS conditional MFE.

        Conditional MFE is intentionally NaN on LS rows.  The generic pair/list-scope
        capability must remove those items before ranking geometry is constructed.
        """

        return np.column_stack(
            [self.low_adverse_safety_percentile, self.conditional_mfe_percentile]
        ).astype(np.float32, copy=False)


def build_hs_conditional_mfe_targets(
    group_table: pd.DataFrame,
    valid_mask: np.ndarray,
    *,
    true_hs_percentile_cutoff: float = DEFAULT_TRUE_HS_PERCENTILE_CUTOFF,
) -> HsConditionalMfeTargets:
    required = {"date", "target_favorable_r", "target_adverse_r"}
    missing = sorted(required - set(group_table.columns))
    if missing:
        raise ValueError(f"HS-Conditional MFE target缺少canonical欄位: {missing}")

    cutoff = float(true_hs_percentile_cutoff)
    if not np.isfinite(cutoff) or not (0.0 < cutoff < 1.0):
        raise ValueError("true-HS percentile cutoff必須位於(0,1)")

    valid = np.asarray(valid_mask, dtype=bool)
    if valid.ndim != 1 or len(valid) != len(group_table):
        raise ValueError("HS-Conditional MFE valid_mask shape不一致")

    dates = pd.to_datetime(group_table["date"], errors="raise").dt.normalize()
    favorable = pd.to_numeric(group_table["target_favorable_r"], errors="coerce").to_numpy(dtype=np.float64)
    adverse = pd.to_numeric(group_table["target_adverse_r"], errors="coerce").to_numpy(dtype=np.float64)
    component_valid = valid & np.isfinite(favorable) & np.isfinite(adverse)
    if not np.array_equal(component_valid, valid):
        raise ValueError("HS-Conditional MFE valid rows必須同時具有finite favorable/adverse target")

    safety = build_same_date_percentile_targets(-adverse, valid, dates)
    primary_mfe = build_same_date_percentile_targets(favorable, valid, dates)
    true_hs = valid & (np.asarray(safety, dtype=np.float64) >= cutoff)
    conditional_mfe = build_same_date_percentile_targets(favorable, true_hs, dates)

    if not bool(np.any(true_hs)):
        raise ValueError("HS-Conditional MFE沒有任何true-HS sample")
    if bool(np.any(true_hs & ~np.isfinite(conditional_mfe))):
        raise ValueError("true-HS row缺少conditional MFE percentile")
    if bool(np.any((~true_hs) & np.isfinite(conditional_mfe))):
        raise ValueError("Low-Safety row不得取得conditional MFE target")

    return HsConditionalMfeTargets(
        low_adverse_safety_percentile=np.asarray(safety, dtype=np.float32),
        primary_mfe_percentile=np.asarray(primary_mfe, dtype=np.float32),
        true_hs_mask=np.asarray(true_hs, dtype=bool),
        conditional_mfe_percentile=np.asarray(conditional_mfe, dtype=np.float32),
    )


__all__ = [
    "DEFAULT_TRUE_HS_PERCENTILE_CUTOFF",
    "HsConditionalMfeTargets",
    "build_hs_conditional_mfe_targets",
]
