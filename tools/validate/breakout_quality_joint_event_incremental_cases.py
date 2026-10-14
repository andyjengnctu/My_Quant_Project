"""Independent paired class-3, Train-only baseline and calibration regression.

Run directly as a developer case.  Not a formal-suite runner.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile

import numpy as np
import pandas as pd

from core.breakout_quality_policy import get_breakout_quality_workflow_settings
from core.breakout_quality_runtime import get_continuous_ranker_training_policy
from core.research_report_contract import validate_approved_persistent_report_contracts
from filters.breakout_quality.artifacts import build_file_manifest
from filters.breakout_quality.joint_mfe_d40 import build_joint_mfe_d40_quadrants
from services.breakout_quality.joint_event_incremental_evidence import (
    _train_class_counts, _prior_skill_rows, _read_compared_model_scores,
    _join_comparison_scores, _calibration_rows,
)
from services.breakout_quality.point_in_time_scores import _fold_group_ids
from services.breakout_quality import joint_mfe_d40_comparison as comparison


PROFILE = {
    "reference": "daily_universal_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise",
    "full_horizon": "daily_universal_shared_safety_hs_conditional_mfe_full_horizon_mae_full_list_ndcg_pairwise",
    "joint": "daily_universal_joint_mfe_d40_quadrant_cross_entropy",
}
MODEL = {"reference": "REF", "full_horizon": "TREAT", "joint": "CATEGORICAL"}
FOLDS = (("fold_20210101_20211231", "2021-01-01", "2021-12-31"),
         ("fold_20220101_20221231", "2022-01-01", "2022-12-31"))


def _fixture():
    # Each date has 24 stock-days with a non-degenerate four-quadrant class truth.
    days = pd.date_range("2019-01-02", "2022-11-02", freq="MS") + pd.Timedelta(days=1)
    records = []
    for d in days:
        for j in range(24):
            records.append({
                "ticker": f"S{j:03d}", "date": d,
                "target_favorable_r": 0.1 + (j % 12) * 0.022,
                "target_minimum_low_return": -(0.01 + ((j * 5) % 24) * 0.005),
                "label_eval_end_date": d + pd.Timedelta(days=40),
            })
    group = pd.DataFrame(records)
    group["group_index"] = np.arange(len(group), dtype=np.int64)
    bundle = SimpleNamespace(
        group_table=group, target_valid=np.ones(len(group), dtype=bool),
        profile=SimpleNamespace(training_sample_scope="daily_eligible_stock_days"),
    )
    labels = build_joint_mfe_d40_quadrants(group, bundle.target_valid)
    assert set(labels.astype(int)) == set(range(4)), set(labels)
    oos = group.loc[(group["date"] >= "2021-03-01") & (group["date"] <= "2022-09-30")].copy()
    oos = oos.loc[oos["date"].dt.month.isin([3, 4, 5, 6, 7, 8, 9])].copy()
    oos["joint_truth"] = labels[oos["group_index"].to_numpy(dtype=int)].astype(int)
    oos["fold_id"] = np.where(oos["date"].dt.year == 2021, FOLDS[0][0], FOLDS[1][0])
    return bundle, oos, labels


def _model_items(tmp, bundle, oos, labels, *, mode):
    policy = get_continuous_ranker_training_policy("daily_joint_mfe_d40_quadrant_cross_entropy")
    columns = tuple(policy.categorical_score_columns)
    items = []
    all_prob = np.full((len(oos), 4), 0.14, dtype=float)
    all_prob[np.arange(len(oos)), oos["joint_truth"].to_numpy(dtype=int)] = 0.58
    for role in PROFILE:
        model_settings = get_breakout_quality_workflow_settings(experiment_profile=PROFILE[role])
        score = oos[["ticker", "date", "group_index"]].copy()
        if role == "joint":
            # Proper probabilities and nontrivial calibration bins.
            for j, col in enumerate(columns):
                score[col] = all_prob[:, j]
            score["model_score" if mode == "forward_oos" else "breakout_quality_score"] = score[columns[3]]
        else:
            # Different ranking signal qualities on identical future truth.
            is_success = oos["joint_truth"].to_numpy(dtype=int) == 3
            score["model_score" if mode == "forward_oos" else "breakout_quality_score"] = np.where(
                is_success, 0.8 if role == "reference" else 0.4, 0.2 if role == "reference" else 0.6
            )
        if mode == "forward_oos":
            score["target_raw_r"] = 0.2
        else:
            score["fold_id"] = oos["fold_id"].to_numpy()
            score["model_information_cutoff"] = np.where(oos["date"].dt.year == 2021, "2020-12-31", "2021-12-31")
        path = Path(tmp) / f"{role}_{mode}.csv.gz"
        score.to_csv(path, index=False, encoding="utf-8-sig", compression="infer")
        report = {"artifacts": {"oos_scores_gzip": build_file_manifest(path)}}
        if role == "joint":
            report["split_report"] = {
                "selection_start_date": "2019-01-01",
                "selection_end_date": "2020-12-31",
                "inner_validation_start_date": "2020-07-01",
                "oos_start_date": "2021-01-01",
                "counts": {"inner_train": int(((bundle.group_table["date"] >= "2019-01-01") &
                    (bundle.group_table["date"] < "2020-07-01") &
                    (bundle.group_table["label_eval_end_date"] < "2020-07-01")).sum())},
            }
        manifest = {}
        if role == "joint" and mode == "rolling_oos":
            folds = []
            for fold_id, start, end in FOLDS:
                ids = _fold_group_ids(bundle, {"fold_id": fold_id, "score_start": pd.Timestamp(start),
                      "score_end": pd.Timestamp(end)}, validation_months=6)
                folds.append({"fold_id": fold_id, "planned_periods": {"score_start": start, "score_end": end},
                              "group_counts": {"inner_train": len(ids["train_ids"])}})
            manifest = {"inner_validation_months": 6, "evaluation_policy": {"train_window_months": None},
                        "folds": folds}
        items.append({"model_id": MODEL[role], "settings": model_settings,
                      "contract": SimpleNamespace(score_path=path, report=report, manifest=manifest,
                                                   continuous_target_id=model_settings.continuous_target_id),
                      "payload": {}})
    return items, columns


def case():
    assert not validate_approved_persistent_report_contracts()
    bundle, oos, labels = _fixture()
    train_original = labels.copy()
    with tempfile.TemporaryDirectory(prefix="joint_incremental_test_") as temp:
        candidate = set(zip(oos.loc[oos["ticker"].str.endswith(("0", "1", "2")), "ticker"],
                            oos.loc[oos["ticker"].str.endswith(("0", "1", "2")), "date"]))
        for mode in ("forward_oos", "rolling_oos"):
            items, columns = _model_items(temp, bundle, oos, labels, mode=mode)
            with patch.object(comparison, "load_daily_universal_ranker_data", return_value=bundle), patch.object(
                    comparison, "load_official_breakout_candidate_keys", return_value=candidate):
                out = comparison.attach_joint_mfe_d40_extension(items, evaluation_mode=mode, project_root=Path(temp))
            extension = {row["model_id"]: {ext["id"]: ext for ext in row["payload"].get("comparison_extensions", [])} for row in out}
            for model in MODEL.values():
                assert "joint_mfe_d40_incremental_skill_v1" in extension[model]
            from services.research.breakout_quality_application import _render_model_comparison_specific_extensions
            views = [{"model_id": model["model_id"],
                      "view": {"extensions": model["payload"].get("comparison_extensions", [])}}
                     for model in out]
            rendered = "\n".join(_render_model_comparison_specific_extensions(views, target="markdown"))
            for section in ("Same Truth", "Train-frozen Skill", "Top10 class-3 hit",
                            "Train-prior CE", "Fixed-bin ECE", "P(Class3) bin"):
                assert section in rendered, section
            source = extension[MODEL["joint"]]["joint_mfe_d40_incremental_skill_v1"]
            ref = extension[MODEL["reference"]]["joint_mfe_d40_incremental_skill_v1"]
            treat = extension[MODEL["full_horizon"]]["joint_mfe_d40_incremental_skill_v1"]
            assert source["cohort_sha256"] == ref["cohort_sha256"] == treat["cohort_sha256"]
            assert source["paired_rows"][0]["stock_days"] == len(oos)
            assert source["paired_rows"][0]["top10_selected"] == ref["paired_rows"][0]["top10_selected"]
            assert ref["paired_rows"][0]["top10_event_hit_pct"] != treat["paired_rows"][0]["top10_event_hit_pct"]
            assert len(source["fold_skill_rows"]) == (4 if mode == "rolling_oos" else 2)
            assert len(source["fold_rank_rows"]) == (4 if mode == "rolling_oos" else 2)
            assert source["calibration_rows"]
            assert all(np.isfinite(row["brier_skill_pct"]) for row in source["fold_skill_rows"])
            assert all(np.isfinite(row["calibration_ece_pp"]) for row in source["fold_skill_rows"])
            assert len(source["fold_calibration_detail"]) == len(source["fold_skill_rows"])
            for group in source["fold_calibration_detail"]:
                assert all(0 <= detail["calibration_gap_pp"] + detail["actual_event_pct"] <= 100
                           for detail in group["bins"])
            assert all(row["train_stock_days"] > 1 for row in source["fold_skill_rows"])
            cd_item = items[2]
            fold = FOLDS[0][0] if mode == "rolling_oos" else "forward"
            count, n = _train_class_counts(bundle, labels, mode=mode, item=cd_item, fold=fold)
            # Hindsight OOS label perturbation cannot alter any Train-only prior.
            disturbed = labels.copy()
            disturbed[oos["group_index"].to_numpy(dtype=int)] = 3
            count_after, n_after = _train_class_counts(bundle, disturbed, mode=mode, item=cd_item, fold=fold)
            assert n == n_after and np.array_equal(count, count_after)
            # Frozen training-count lineage must fail closed if a saved split is tampered with.
            if mode == "forward_oos":
                saved_count = cd_item["contract"].report["split_report"]["counts"]["inner_train"]
                cd_item["contract"].report["split_report"]["counts"]["inner_train"] = saved_count + 1
                with np.testing.assert_raises_regex(ValueError, "counts disagree"):
                    _train_class_counts(bundle, labels, mode=mode, item=cd_item, fold=fold)
                cd_item["contract"].report["split_report"]["counts"]["inner_train"] = saved_count
            else:
                entry = cd_item["contract"].manifest["folds"][0]["group_counts"]
                saved_count = entry["inner_train"]
                entry["inner_train"] = saved_count + 1
                with np.testing.assert_raises_regex(ValueError, "count disagrees"):
                    _train_class_counts(bundle, labels, mode=mode, item=cd_item, fold=fold)
                entry["inner_train"] = saved_count
            # Mutation of historical source score invalidates Forward hash; Rolling PIT cutoff is mandatory.
            before = Path(items[0]["contract"].score_path)
            score = pd.read_csv(before, encoding="utf-8-sig")
            if mode == "forward_oos":
                score.loc[0, "model_score"] = 0.987
                score.to_csv(before, index=False, encoding="utf-8-sig", compression="infer")
                with np.testing.assert_raises_regex(ValueError, "SHA manifest"):
                    _read_compared_model_scores(items[0], mode=mode)
            else:
                score.loc[0, "model_information_cutoff"] = "2022-12-31"
                score.to_csv(before, index=False, encoding="utf-8-sig", compression="infer")
                with np.testing.assert_raises_regex(ValueError, "cutoff"):
                    _read_compared_model_scores(items[0], mode=mode)
            # Deliberate swapped fold of a correctly scored reference must be rejected.
            if mode == "rolling_oos":
                other = oos[["ticker", "date", "group_index"]].copy()
                other["model_score"] = 0.3
                other["fold_id"] = np.where(oos["date"].dt.year == 2021, FOLDS[1][0], FOLDS[0][0])
                joint = oos[["ticker", "date", "group_index", "joint_truth", "fold_id"]].copy()
                joint["joint_event_probability"] = 0.3
                with np.testing.assert_raises_regex(ValueError, "fold differs"):
                    _join_comparison_scores(joint, other, other, mode="rolling_oos")
    assert np.array_equal(labels, train_original)


if __name__ == "__main__":
    case()
    print("Joint same-truth, Train-only climatology, fold skill and calibration: PASS")
