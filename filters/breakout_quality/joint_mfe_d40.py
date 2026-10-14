"""Canonical four-quadrant joint future MFE x full-horizon MAE target.

Truth uses only the existing matured 40-bar daily-universal future paths.
Rows without complete MFE and low paths remain NaN, never training targets.
The same-date P50 cutoffs are fixed research semantics; not OOS-fitted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from filters.breakout_quality.continuous_ranker_data import build_same_date_percentile_targets
from filters.breakout_quality.continuous_target import (
    StrategyAlignedContinuousTargetSpec,
    full_horizon_mae_safety_r_from_minimum_low_return,
)
from filters.breakout_quality.contract import DEFAULT_LABEL_POLICY

JOINT_PERCENTILE_CUTOFF = 0.50
JOINT_CLASS_COUNT = 4
JOINT_FAVORABLE_AND_LOW_MAE_CLASS = 3
JOINT_CLASSES = (
    "low_mfe_high_mae", "low_mfe_low_mae", "high_mfe_high_mae", "high_mfe_low_mae",
)


def build_joint_mfe_d40_quadrants(group_table: pd.DataFrame, valid_mask: np.ndarray) -> np.ndarray:
    required = {"date", "target_favorable_r", "target_minimum_low_return"}
    missing = sorted(required - set(group_table.columns))
    if missing:
        raise ValueError(f"joint MFE-D40 target missing canonical columns: {missing}")
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.ndim != 1 or len(valid) != len(group_table):
        raise ValueError("joint MFE-D40 valid mask must match daily-universal rows")
    favorable = pd.to_numeric(group_table["target_favorable_r"], errors="coerce").to_numpy(dtype=np.float64)
    minimum = pd.to_numeric(group_table["target_minimum_low_return"], errors="coerce").to_numpy(dtype=np.float64)
    if np.any(valid & (~np.isfinite(favorable) | ~np.isfinite(minimum))):
        raise ValueError("matured joint MFE-D40 rows require complete 40-bar MFE and low")
    risk = StrategyAlignedContinuousTargetSpec.from_label_policy(DEFAULT_LABEL_POLICY).risk_budget_return
    safety = full_horizon_mae_safety_r_from_minimum_low_return(minimum, risk_budget_return=risk)
    dates = pd.to_datetime(group_table["date"], errors="raise").dt.normalize()
    favorable_pct = build_same_date_percentile_targets(favorable, valid, dates)
    safety_pct = build_same_date_percentile_targets(safety, valid, dates)
    labels = np.full(len(group_table), np.nan, dtype=np.float32)
    labels[valid] = (
        2 * (favorable_pct[valid] >= JOINT_PERCENTILE_CUTOFF).astype(np.int8)
        + (safety_pct[valid] >= JOINT_PERCENTILE_CUTOFF).astype(np.int8)
    ).astype(np.float32)
    return labels


def joint_probabilistic_metrics(logits: np.ndarray, labels: np.ndarray) -> dict:
    """Proper scoring rules on a held-out scope; class-3 is the joint success event."""
    raw = np.asarray(logits, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != JOINT_CLASS_COUNT or len(raw) != len(truth):
        raise ValueError("joint quadrant logits/target dimension mismatch")
    if not np.isfinite(raw).all() or not np.isfinite(truth).all():
        raise ValueError("joint quadrant logits/target must be finite")
    classes = truth.astype(np.int64)
    if not np.array_equal(truth, classes.astype(np.float64)) or np.any((classes < 0) | (classes >= JOINT_CLASS_COUNT)):
        raise ValueError("joint quadrant target must be integer class 0..3")
    shifted = raw - np.max(raw, axis=1, keepdims=True)
    log_norm = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    log_probs = shifted - log_norm
    probs = np.exp(log_probs)
    success = (classes == JOINT_FAVORABLE_AND_LOW_MAE_CLASS).astype(np.float64)
    return {
        "group_count": int(len(classes)),
        "cross_entropy": float(-np.mean(log_probs[np.arange(len(classes)), classes])),
        "joint_event_brier": float(np.mean((probs[:, JOINT_FAVORABLE_AND_LOW_MAE_CLASS] - success) ** 2)),
        "joint_event_base_rate": float(success.mean()),
        "joint_event_predicted_mean": float(probs[:, JOINT_FAVORABLE_AND_LOW_MAE_CLASS].mean()),
        "class_counts": [int(np.count_nonzero(classes == k)) for k in range(JOINT_CLASS_COUNT)],
    }
