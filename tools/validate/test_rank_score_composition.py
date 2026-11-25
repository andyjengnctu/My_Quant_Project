"""Runtime rank fusion: independent numeric/cohort and source-admission fixtures."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.rank_score_composition import rank_score_composition
from filters.breakout_quality.rank_score_composition import compose_rank_score_tables, load_rank_composition_scores


def contract(weights=(0.5, 0.5)):
    return {"method": "same_day_percentile_weighted_v1", "universe": "common_score_eligible_stock_days",
            "tie_method": "average_zero_based", "members": tuple(
                {"dl_id": key, "weight": weight} for key, weight in zip(("a", "b"), weights))}


def scores(tickers, values, date="2021-01-04"):
    return pd.Series(values, index=pd.MultiIndex.from_tuples(
        [(ticker, date) for ticker in tickers], names=("ticker", "date")), dtype=float)


class RankScoreCompositionTests(TestCase):
    def test_fixed_fusion_uses_average_zero_based_ties_and_is_unfitted(self):
        tables = {"a": scores(["A", "B", "C", "D"], [9, 9, 0, 3]),
                  "b": scores(["D", "B", "C", "A"], [3, 1, 4, 2])}
        before = deepcopy(tables)
        result = compose_rank_score_tables(tables, contract())
        np.testing.assert_allclose(result.values, [7/12, 5/12, 1/2, 1/2], atol=1e-7)
        for key in tables:
            pd.testing.assert_series_equal(tables[key], before[key])
        scaled = {key: value * 37 + 1000 for key, value in tables.items()}
        pd.testing.assert_series_equal(result, compose_rank_score_tables(scaled, contract()))

    def test_zero_weight_still_requires_common_universe_before_ranking(self):
        tables = {"a": scores(["A", "B", "C"], [0, 1, 2]), "b": scores(["A", "C", "Z"], [5, 4, 0])}
        for weights, expected in (((1., 0.), [0., 1.]), ((0., 1.), [1., 0.]), ((.5, .5), [.5, .5])):
            result = compose_rank_score_tables(tables, contract(weights))
            self.assertEqual(result.index.get_level_values("ticker").tolist(), ["A", "C"])
            np.testing.assert_array_equal(result.values, expected)

    def test_singleton_and_future_days_do_not_change_prior_scores(self):
        tables = {"a": scores(["A"], [9]), "b": scores(["A"], [-4])}
        original = compose_rank_score_tables(tables, contract())
        self.assertEqual(original.iloc[0], .5)
        augmented = {key: pd.concat([value, scores(["A", "B"], [1000, -1000], "2022-02-03")])
                     for key, value in tables.items()}
        pd.testing.assert_series_equal(original, compose_rank_score_tables(augmented, contract()).reindex(original.index))

    def test_missing_nonfinite_duplicates_and_unapproved_contract_fail_closed(self):
        good = {"a": scores(["A", "B"], [1, 2]), "b": scores(["A", "B"], [2, 1])}
        for bad in ({"a": good["a"]}, {**good, "a": scores(["A", "A"], [1, 2])},
                    {**good, "b": scores(["A", "B"], [1, np.nan])},
                    {**good, "b": scores(["Z"], [1])}):
            with self.assertRaises(ValueError):
                compose_rank_score_tables(bad, contract())
        for mutation in ({"tie_method": "first"}, {"universe": "future_mature_targets"},
                         {"members": ({"dl_id": "a", "weight": .7},)},
                         {"members": ({"dl_id": "a", "weight": float("nan")},)}):
            with self.assertRaises(ValueError):
                rank_score_composition({"rank_score_composition": {**contract(), **mutation}})

    def test_every_source_is_bound_and_pit_cutoff_is_enforced(self):
        from filters.breakout_quality import ranking_score_store as owner
        root = Path.cwd()
        members = [dict(dl_id=key, filter_id="fixture", model_architecture="fixture", experiment_profile="fixture",
                        expected_seed=17, score_source="selection_point_in_time", score_path=str(root/f"{key}.csv"),
                        manifest_path=str(root/f"{key}.json"), point_in_time_dir_override=key,
                        score_column="breakout_quality_score") for key in ("a", "b")]
        opts = {"rank_score_composition": contract(), "rank_composition_sources": members}
        seed_by_key = {"a": 17, "b": 17}
        def admit(*args, **kwargs):
            key = kwargs["point_in_time_dir_override"]
            return SimpleNamespace(seed=seed_by_key[key], score_path=root/f"{key}.csv", manifest_path=root/f"{key}.json",
                                   available_from="2021-01-01", available_through="2021-12-31")
        frame = scores(["A", "B"], [1, 2]).to_frame("breakout_quality_score")
        frame["model_information_cutoff"] = "2020-12-31"
        with patch.object(owner, "load_selection_point_in_time_ranking_contract", side_effect=admit), \
             patch.object(owner, "load_selection_point_in_time_score_table_from_path", return_value=frame):
            self.assertEqual(len(load_rank_composition_scores(project_root=root, options=opts, score_bindings={})["scores"]), 2)
            seed_by_key["b"] = 18
            with self.assertRaisesRegex(ValueError, "seed/path"):
                load_rank_composition_scores(project_root=root, options=opts, score_bindings={})
            seed_by_key["b"] = 17
            frame["model_information_cutoff"] = "2021-01-04"
            with self.assertRaisesRegex(ValueError, "non-PIT"):
                load_rank_composition_scores(project_root=root, options=opts, score_bindings={})

    def test_registered_modes_dependencies_overrides_and_projection_are_consistent(self):
        from core.strategy_compare_policy import get_strategy_comparison_settings
        from core.rank_score_composition import rank_composition_source_ids
        from filters.breakout_quality.strategy_compare_dl_artifacts import resolve_arm_runtime_dl_source_ids
        from filters.breakout_quality.strategy_score_projection import runtime_score_projection_requirements
        from services.research.strategy_compare_execution import resolve_strategy_compare_ranking_options
        from core.research_summary import unresolved_oracle_strategy_contracts
        from core.strategy_compare_registry import STRATEGY_COMPARE_ARMS
        unresolved = unresolved_oracle_strategy_contracts(arm_registry=STRATEGY_COMPARE_ARMS)
        for mode in ("oos", "rolling"):
            settings = get_strategy_comparison_settings("broker_cash_dl_conversion_" + mode)
            for arm in settings.enabled_arms:
                ids = rank_composition_source_ids(arm.dl_runtime_options)
                if not ids:
                    continue
                import json
                self.assertEqual(arm.as_dict(), json.loads(json.dumps(arm.as_dict())))
                self.assertIn(arm.arm_id, unresolved)
                self.assertEqual(set(ids), set(resolve_arm_runtime_dl_source_ids(settings, arm)))
                projections = runtime_score_projection_requirements(settings_payload=settings.as_dict(), on_arm_payload=arm.as_dict())
                self.assertEqual(set(ids), set(projections))
                overrides = {key: {"score_path": f"seed_17/{key}/selection_point_in_time_scores.csv",
                                   "manifest_path": f"seed_17/{key}/selection_point_in_time_manifest.json"} for key in ids}
                resolved = resolve_strategy_compare_ranking_options(settings, arm, score_overrides=overrides, expected_seed=17)
                self.assertTrue(all(row["expected_seed"] == 17 and "seed_17" in row["score_path"]
                                    for row in resolved["rank_composition_sources"]))
                with self.assertRaisesRegex(ValueError, "every isolated"):
                    resolve_strategy_compare_ranking_options(settings, arm, score_overrides={ids[0]: overrides[ids[0]]}, expected_seed=17)

    def test_runtime_refreshes_information_date_and_never_fills_missing_member(self):
        from core.strategy_compare_policy import get_strategy_comparison_settings
        from core.breakout_quality_ranking_context import breakout_quality_ranking_source_context
        from filters.breakout_quality.runtime import resolve_breakout_quality_candidate_rank
        settings = get_strategy_comparison_settings("broker_cash_dl_conversion_rolling")
        arm = next(arm for arm in settings.enabled_arms if rank_score_composition(arm.dl_runtime_options))
        source = settings.dl_sources[arm.dl_id]
        options = dict(score_source=source.score_source, model_architecture=source.model_architecture,
                       experiment_profile=source.experiment_profile, ranking_policy=arm.dl_runtime_mode,
                       ranking_options=arm.dl_runtime_options)
        lookup = dict(ticker="A", signal_date="2020-12-31", information_date="2021-01-05", high_len=20)
        with breakout_quality_ranking_source_context(**options):
            with self.assertRaisesRegex(ValueError, "admitted"):
                resolve_breakout_quality_candidate_rank(**lookup)
        options["ranking_options"] = {**options["ranking_options"], "_rank_composition_scores": {
            "scores": {("A", "2021-01-04"): .9, ("A", "2021-01-05"): .2},
            "available_from": "2021-01-04", "available_through": "2021-12-31"}}
        with breakout_quality_ranking_source_context(**options):
            self.assertEqual(resolve_breakout_quality_candidate_rank(**lookup)["score"], .2)
            self.assertFalse(resolve_breakout_quality_candidate_rank(**{**lookup, "ticker": "0050"})["available"])
            self.assertEqual(resolve_breakout_quality_candidate_rank(**{**lookup, "information_date": "2021-01-01"})["unavailable_reason"], "outside_score_period")
