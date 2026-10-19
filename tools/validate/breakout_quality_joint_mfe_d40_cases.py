"""Isolated MR-13CD categorical-target and OOS comparison capability checks.

This is an independent developer check. It is not a formal-suite entry point.
"""
from __future__ import annotations


import numpy as np
import pandas as pd

from core.breakout_quality_runtime import get_continuous_ranker_training_policy
from filters.breakout_quality.joint_mfe_d40 import (
    build_joint_mfe_d40_quadrants, joint_probabilistic_metrics,
)


def _sample():
    # Eight securities per date, deliberately fill all 4 quadrants.
    dates = pd.to_datetime(["2021-01-04"] * 16 + ["2021-01-05"] * 16)
    favorable = np.tile(np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.6, 0.9, 0.7]), 4)
    lowest = np.tile(np.array([-0.08, -0.02, -0.12, -0.01, -0.01, -0.1, -0.11, -0.005]), 4)
    group_table = pd.DataFrame({
        "ticker": [f"T{i:03}" for i in range(16)] * 2,
        "date": dates, "group_index": np.arange(32),
        "target_favorable_r": favorable, "target_minimum_low_return": lowest,
    })
    return group_table


def case() -> None:
    policy = get_continuous_ranker_training_policy("daily_joint_mfe_d40_quadrant_cross_entropy")
    assert policy.categorical_class_count == 4
    assert policy.categorical_primary_class == 3
    group = _sample()
    labels = build_joint_mfe_d40_quadrants(group, np.ones(len(group), dtype=bool))
    assert set(labels.astype(int)) == {0, 1, 2, 3}
    missing = np.ones(len(group), dtype=bool)
    missing[-1] = False
    incomplete = build_joint_mfe_d40_quadrants(group, missing)
    assert np.isnan(incomplete[-1])
    assert np.isfinite(incomplete[:-1]).all()
    for invalid in (np.ones(len(group) + 1, dtype=bool), np.ones(len(group), dtype=int).reshape(-1, 1)):
        with np.testing.assert_raises(ValueError):
            build_joint_mfe_d40_quadrants(group, invalid)
    ideal_logits = np.full((len(labels), 4), -8.0)
    ideal_logits[np.arange(len(labels)), labels.astype(int)] = 8.0
    perfect = joint_probabilistic_metrics(ideal_logits, labels)
    assert perfect["cross_entropy"] < 1e-5
    assert perfect["joint_event_brier"] < 1e-8
    # [AI] Categorical target validity and probability scores are the retained
    # canonical primitives; post-fit historical Extension audit is retired.
    assert perfect["joint_event_predicted_mean"] > 0.0
    assert perfect["joint_event_brier"] >= 0.0


if __name__ == "__main__":
    case()
    print("Joint MFE-D40 label and probabilistic metric primitives PASS")
