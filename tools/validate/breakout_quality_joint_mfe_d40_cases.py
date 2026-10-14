"""Isolated MR-13CD categorical-target and OOS comparison capability checks.

This is an independent developer check. It is not a formal-suite entry point.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.breakout_quality_runtime import get_continuous_ranker_training_policy
from filters.breakout_quality.artifacts import build_file_manifest
from filters.breakout_quality.joint_mfe_d40 import (
    build_joint_mfe_d40_quadrants, joint_probabilistic_metrics,
)
from services.breakout_quality import joint_mfe_d40_comparison as comparison


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
    # Distinct dependence structure: independence-projection is not the joint event probability.
    columns = tuple(policy.categorical_score_columns)
    probs = np.tile(np.array([[.05, .1, .15, .7], [.5, .1, .3, .1]]), (16, 1))
    sample = group[["ticker", "date", "group_index"]].copy()
    for i, col in enumerate(columns):
        sample[col] = probs[:, i]
    sample["joint_truth"] = labels.astype(int)
    reference = comparison._proper_scores(sample, split="Rolling OOS", columns=columns)
    assert reference["stock_days"] == 32
    assert reference["dates"] == 2
    assert reference["multiclass_logloss"] > 0
    assert reference["joint_brier"] != reference["independent_brier"]
    altered = sample.copy()
    altered.loc[0, columns[3]] = 1.1
    with np.testing.assert_raises(ValueError):
        comparison._proper_scores(altered, split="Rolling OOS", columns=columns)

    # Exact same validated canonical OOS samples used by the two workflow paths.
    # Forward uses model_score while rolling stores breakout_quality_score.
    for mode in ("forward_oos", "rolling_oos"):
        with tempfile.TemporaryDirectory(prefix="cd_joint_oos_") as tmp:
            score = sample.drop(columns=["joint_truth"]).copy()
            score["model_score" if mode == "forward_oos" else "breakout_quality_score"] = score[columns[3]]
            if mode == "forward_oos":
                score["target_raw_r"] = group["target_favorable_r"]
            else:
                score["fold_id"] = "F01"
                score["model_information_cutoff"] = "2020-12-31"
            path = Path(tmp) / ("scores.csv.gz" if mode == "forward_oos" else "scores.csv")
            score.to_csv(path, index=False, encoding="utf-8-sig", compression="infer")
            report = {"artifacts": {"oos_scores_gzip": build_file_manifest(path)}} if mode == "forward_oos" else {}
            item = {
                "model_id": "MR-13CD",
                "settings": SimpleNamespace(
                    filter_id="breakout_quality_v1",
                    model_architecture="inception_time_joint_quadrant_v1",
                    experiment_profile="daily_universal_joint_mfe_d40_quadrant_cross_entropy",
                    training_objective="daily_joint_mfe_d40_quadrant_cross_entropy",
                ),
                "contract": SimpleNamespace(score_path=path, report=report),
                "payload": {},
            }
            bundle = SimpleNamespace(group_table=group, target_valid=np.ones(len(group), dtype=bool))
            candidates = {(str(group.iloc[i]["ticker"]), group.iloc[i]["date"]) for i in range(16)}
            with patch.object(comparison, "load_daily_universal_ranker_data", return_value=bundle), patch.object(
                comparison, "load_official_breakout_candidate_keys", return_value=candidates
            ):
                result = comparison.attach_joint_mfe_d40_extension([item], evaluation_mode=mode, project_root=Path(tmp))
                evidence = result[0]["payload"]["comparison_extensions"][0]
                assert evidence["id"] == comparison.EXTENSION_ID
                assert {row["split"] for row in evidence["rows"]} == {
                    "Forward OOS" if mode == "forward_oos" else "Rolling OOS", "Breakout slice"
                }
                assert evidence["rows"][0]["stock_days"] == 32
                bad_score = score.copy()
                bad_score.loc[0, columns[3]] = 1.2
                bad_score.to_csv(path, index=False, encoding="utf-8-sig", compression="infer")
                if mode == "forward_oos":
                    # Forward must reject a changed score-manifest hash.
                    with np.testing.assert_raises_regex(ValueError, "manifest"):
                        comparison.attach_joint_mfe_d40_extension([item], evaluation_mode=mode, project_root=Path(tmp))
                else:
                    with np.testing.assert_raises(ValueError):
                        comparison.attach_joint_mfe_d40_extension([item], evaluation_mode=mode, project_root=Path(tmp))


if __name__ == "__main__":
    case()
    print("MR-13CD isolated target, probability and Forward/Rolling OOS tests PASS")
