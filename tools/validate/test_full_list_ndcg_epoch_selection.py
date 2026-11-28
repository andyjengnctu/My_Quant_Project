"""Generic full-list-NDCG validation selector primitive contract."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from core.breakout_quality_runtime import (
    TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_PAIRWISE_PROBABILITY,
    get_continuous_ranker_training_policy,
)
from filters.breakout_quality.continuous_ranker_quality import mean_daily_full_list_ndcg
from services.breakout_quality.train_continuous_ranker import _epoch_metric_improved


class FullListNdcgEpochSelectionTests(unittest.TestCase):
    def test_linear_gain_full_list_and_equal_day_weight(self):
        # Date one: perfect NDCG=1; date two: reversed order NDCG<1.
        dates = pd.to_datetime(["2019-01-02"] * 3 + ["2019-01-03"] * 3)
        gains = np.array([0.0, 0.5, 1.0] * 2)
        scores = np.array([0.1, 0.5, 0.9, 0.9, 0.5, 0.1])
        result = mean_daily_full_list_ndcg(dates, scores, gains)
        discount = 1.0 / np.log2(np.arange(3) + 2.0)
        ideal = float(np.sum(np.array([1.0, 0.5, 0.0]) * discount))
        reversed_ndcg = float(np.sum(np.array([0.0, 0.5, 1.0]) * discount) / ideal)
        self.assertEqual(result["rankable_date_count"], 2)
        self.assertAlmostEqual(result["mean_daily_full_list_ndcg"], (1.0 + reversed_ndcg) / 2.0)

    def test_no_top_k_stable_ties_and_nondiscriminating_days(self):
        dates = ["2019-01-01"] * 3 + ["2019-01-02"] * 2
        res = mean_daily_full_list_ndcg(
            dates, [0.4, 0.4, 0.5, 0.0, 1.0], [0.5, 0.0, 1.0, 0.5, 0.5]
        )
        discounts = 1.0 / np.log2(np.arange(3) + 2)
        expected = (1.0 * discounts[0] + 0.5 * discounts[1]) / (
            1.0 * discounts[0] + 0.5 * discounts[1]
        )
        self.assertEqual(res["rankable_date_count"], 1)
        self.assertAlmostEqual(res["mean_daily_full_list_ndcg"], expected)

    def test_bad_values_and_mismatched_lengths_fail_closed(self):
        for scores, gains in [([0.5], [1.0, 0.0]), ([0.5, float("nan")], [0.0, 1.0]),
                              ([0.5, 0.6], [0.0, 1.1])]:
            with self.assertRaises(ValueError):
                mean_daily_full_list_ndcg(["2019-01-01"] * 2, scores, gains)
        none = mean_daily_full_list_ndcg(["2019-01-01"] * 2, [0.5, 0.7], [0.5, 0.5])
        self.assertIsNone(none["mean_daily_full_list_ndcg"])

    def test_policy_selection_is_profile_metric_driven(self):
        policy = get_continuous_ranker_training_policy(TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING)
        policy.validate_profile_loss_metric(loss_name="pairwise_logistic", epoch_selection_metric="mean_daily_spearman")
        policy.validate_profile_loss_metric(loss_name="pairwise_logistic", epoch_selection_metric="mean_daily_full_list_ndcg")
        self.assertEqual(policy.epoch_selection_rule_for("mean_daily_spearman"), "daily_spearman_then_percentile_mse")
        self.assertEqual(policy.epoch_selection_rule_for("mean_daily_full_list_ndcg"), "daily_full_list_ndcg_then_percentile_mse")
        with self.assertRaises(ValueError):
            get_continuous_ranker_training_policy(
                TRAINING_OBJECTIVE_DAILY_PAIRWISE_PROBABILITY
            ).validate_profile_loss_metric(
                loss_name="pairwise_bernoulli_probability",
                epoch_selection_metric="mean_daily_full_list_ndcg",
            )

    def test_existing_min_delta_and_mse_tie_break(self):
        kwargs = dict(
            best_primary=0.8, secondary=0.1, best_secondary=0.2,
            min_delta=0.001, primary_direction="max", secondary_direction="min",
        )
        self.assertTrue(_epoch_metric_improved(primary=0.802, **kwargs))
        self.assertFalse(_epoch_metric_improved(primary=0.8005, **kwargs))
        self.assertTrue(_epoch_metric_improved(primary=0.8, **kwargs))


if __name__ == "__main__":
    unittest.main()
