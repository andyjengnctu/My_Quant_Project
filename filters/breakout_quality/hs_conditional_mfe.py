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


@dataclass(frozen=True)
class HsPriorityMfeTargets:
    """Non-compensatory all-universe ranking truth.

    LS rows are tied at the worst relevance (0). HS rows occupy [0.5, 1.0]
    according to their MFE percentile within the same-date true-HS cohort. This
    guarantees every HS row outranks every LS row while preserving upside
    ordering only where Safety is acceptable.
    """

    low_adverse_safety_percentile: np.ndarray
    primary_mfe_percentile: np.ndarray
    true_hs_mask: np.ndarray
    conditional_mfe_percentile: np.ndarray
    hs_priority_mfe_relevance: np.ndarray

    @property
    def training_target(self) -> np.ndarray:
        return np.column_stack(
            [self.low_adverse_safety_percentile, self.hs_priority_mfe_relevance]
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


def build_hs_priority_mfe_targets(
    group_table: pd.DataFrame,
    valid_mask: np.ndarray,
    *,
    true_hs_percentile_cutoff: float = DEFAULT_TRUE_HS_PERCENTILE_CUTOFF,
) -> HsPriorityMfeTargets:
    """Build LS-floor / HS-within-cohort-MFE relevance for all-universe ranking."""

    base = build_hs_conditional_mfe_targets(
        group_table,
        valid_mask,
        true_hs_percentile_cutoff=true_hs_percentile_cutoff,
    )
    relevance = np.zeros(len(group_table), dtype=np.float32)
    hs = np.asarray(base.true_hs_mask, dtype=bool)
    conditional = np.asarray(base.conditional_mfe_percentile, dtype=np.float32)
    relevance[hs] = 0.5 + 0.5 * conditional[hs]
    valid = np.asarray(valid_mask, dtype=bool)
    relevance[~valid] = np.nan

    finite_valid = relevance[valid]
    if len(finite_valid) and (
        float(np.nanmin(finite_valid)) < 0.0 or float(np.nanmax(finite_valid)) > 1.0
    ):
        raise ValueError("HS-Priority MFE relevance必須位於0～1")
    if bool(np.any(relevance[valid & ~hs] != 0.0)):
        raise ValueError("LS row必須固定為最差relevance=0")
    if bool(np.any(relevance[hs] < 0.5)):
        raise ValueError("HS row relevance必須嚴格高於LS floor")

    return HsPriorityMfeTargets(
        low_adverse_safety_percentile=base.low_adverse_safety_percentile,
        primary_mfe_percentile=base.primary_mfe_percentile,
        true_hs_mask=base.true_hs_mask,
        conditional_mfe_percentile=base.conditional_mfe_percentile,
        hs_priority_mfe_relevance=relevance,
    )


__all__ = [
    "DEFAULT_TRUE_HS_PERCENTILE_CUTOFF",
    "HsConditionalMfeTargets",
    "HsPriorityMfeTargets",
    "build_hs_conditional_mfe_targets",
    "build_hs_priority_mfe_targets",
]
