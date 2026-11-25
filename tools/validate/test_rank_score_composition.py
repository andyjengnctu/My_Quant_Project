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
            # Capability contracts remain testable when their historical arms
            # leave the current research roster.
            for arm in settings.arms.values():
                ids = rank_composition_source_ids(arm.dl_runtime_options)
                if not ids:
                    continue
                import json
                self.assertEqual(arm.as_dict(), json.loads(json.dumps(arm.as_dict())))
                self.assertEqual(arm.arm_id in unresolved, arm.enabled)
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
        arm = next(arm for arm in settings.arms.values() if rank_score_composition(arm.dl_runtime_options))
        source = settings.dl_sources[arm.dl_id]
        options = dict(score_source=source.score_source, model_architecture=source.model_architecture,
                       experiment_profile=source.experiment_profile, ranking_policy=arm.dl_runtime_mode,
                       ranking_options=arm.dl_runtime_options)
        lookup = dict(ticker="A", signal_date="2020-12-31", information_date="2021-01-05", high_len=20)
        with breakout_quality_ranking_source_context(**options):
            with self.assertRaisesRegex(ValueError, "admitted"):
                resolve_breakout_quality_candidate_rank(**lookup)
        prepared = {
            "scores": {("A", "2021-01-04"): .9, ("A", "2021-01-05"): .2},
            "available_from": "2021-01-04", "available_through": "2021-12-31"}
        from filters.breakout_quality.rank_score_composition import rank_composition_scores_context, current_rank_composition_scores
        with breakout_quality_ranking_source_context(**options), rank_composition_scores_context(prepared):
            self.assertEqual(resolve_breakout_quality_candidate_rank(**lookup)["score"], .2)
            self.assertFalse(resolve_breakout_quality_candidate_rank(**{**lookup, "ticker": "0050"})["available"])
            self.assertEqual(resolve_breakout_quality_candidate_rank(**{**lookup, "information_date": "2021-01-01"})["unavailable_reason"], "outside_score_period")
        self.assertIsNone(current_rank_composition_scores())
        # AI: Run the actual service context boundary; only compact declarations
        # may flow into candidate ranking_options and the detailed CSV.
        from services.research import strategy_compare_replay as replay
        from core.breakout_quality_ranking_context import get_breakout_quality_ranking_source_context
        def observe(**kwargs):
            self.assertIs(current_rank_composition_scores(), prepared)
            self.assertEqual(get_breakout_quality_ranking_source_context().ranking_options, arm.dl_runtime_options)
            self.assertEqual(resolve_breakout_quality_candidate_rank(**lookup)["score"], .2)
            return {"observed": True}
        with patch.object(replay, "_run_scenario_inside_source_context", side_effect=observe):
            self.assertEqual(replay._run_scenario(
                name="fixture", data_dir="unused", param_source_kind="fixed", params={},
                start_date="2021-01-01", end_date="2021-12-31", max_positions=10,
                enable_rotation=False, quiet=True,
                ranking_source={**options, "rank_composition_scores": prepared}), {"observed": True})
        self.assertIsNone(current_rank_composition_scores())

    def test_robustness_does_not_attribute_primary_model_diagnostics_to_fusion(self):
        import tempfile
        from services.research import strategy_multi_seed_robustness as robust
        from core.strategy_compare_policy import get_strategy_comparison_settings
        settings = get_strategy_comparison_settings("broker_cash_dl_conversion_rolling")
        arm = next(arm for arm in settings.arms.values() if rank_score_composition(arm.dl_runtime_options))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = dict(profile_id=settings.profile_id, arm_id=arm.arm_id, seed=17, arm_order=1, seed_order=1,
                       comparison_start="2021-01-01", comparison_end="2021-12-31", pair_dir=str(root/"pair"),
                       baseline_dir=str(root/"base"), params_path="unused", result_path=str(root/"result.json"),
                       keep_replay_details=True, model_prediction_metrics={key: .8 for key in robust.MODEL_PREDICTION_METRIC_KEYS})
            with patch.object(robust, "PROJECT_ROOT", root), \
                 patch.object(robust, "run_strategy_compare_active_arm", return_value={"score_ranking": {"return_over_max_drawdown": 2.0}}), \
                 patch.object(robust, "_load_direct_selection_r", return_value=0.0):
                result = robust._replay_one_unit(job)
            self.assertEqual(result["return_over_max_drawdown"], 2.0)
            self.assertTrue(all(result[key] is None for key in robust.MODEL_PREDICTION_METRIC_KEYS))

    def test_post_replay_diagnostics_use_full_cohort_fusion_and_reject_raw_score(self):
        from filters.breakout_quality.rank_score_composition import rank_composition_diagnostic_lookup
        from filters.breakout_quality.strategy_compare_diagnostics import _strategy_selection_diagnostics
        lookup = pd.DataFrame([
            dict(ticker="A", signal_date="2021-01-04", breakout_quality_score=8., target_raw_r=2., target_available=True),
            dict(ticker="B", signal_date="2021-01-04", breakout_quality_score=9., target_raw_r=np.nan, target_available=False),
            dict(ticker="Z", signal_date="2021-01-04", breakout_quality_score=7., target_raw_r=1., target_available=True),
        ])
        prepared = {"scores": {("A", "2021-01-04"): .75, ("B", "2021-01-04"): .25}}
        projected = rank_composition_diagnostic_lookup(lookup, prepared)
        pd.testing.assert_frame_equal(projected.drop(columns="breakout_quality_score"), lookup.drop(columns="breakout_quality_score"))
        self.assertEqual(projected.loc[0, "breakout_quality_score"], .75)
        self.assertEqual(projected.loc[1, "breakout_quality_score"], .25)
        self.assertTrue(pd.isna(projected.loc[2, "breakout_quality_score"]))
        orderable = pd.DataFrame([dict(ticker="A", signal_date="2020-12-31", trade_date="2021-01-05",
            breakout_quality_score_date="2021-01-04", breakout_quality_score=.75, breakout_quality_score_available=True)])
        selected = pd.DataFrame([dict(ticker="A", signal_date="2020-12-31", trade_date="2021-01-05", type="買進")])
        metrics, joined, _ = _strategy_selection_diagnostics(orderable=orderable, selected=selected, lookup=projected)
        self.assertEqual(joined.iloc[0]["pit_breakout_quality_score"], .75)
        self.assertFalse(metrics["future_target_used_for_runtime_sort"])
        for invalid in (lookup, projected.assign(breakout_quality_score=.7)):
            with self.assertRaisesRegex(ValueError, "runtime_score"):
                _strategy_selection_diagnostics(orderable=orderable, selected=selected, lookup=invalid)
        # The report-only Forward refresh uses the same admitted witness and guard.
        from tempfile import TemporaryDirectory
        from unittest.mock import Mock
        from filters.breakout_quality import strategy_compare_diagnostics as diagnostic
        with TemporaryDirectory() as folder:
            root = Path(folder)
            score_path = root / "scores.csv"
            score_path.write_text("fixture", encoding="utf-8")
            payload = {"metadata": dict(comparison_mode=diagnostic.COMPARISON_MODE_SCORE_RANKING,
                score_source=diagnostic.SCORE_SOURCE_CONTINUOUS_RANKER_OOS, score_path=str(score_path),
                experiment_profile="fixture", model_architecture="fixture", filter_id="fixture",
                score_ranking_options={"rank_score_composition": contract()})}
            loader = Mock(return_value=prepared)
            with patch.object(diagnostic, "_continuous_forward_target_lookup", return_value=lookup), \
                 patch.object(diagnostic, "detail_exists", return_value=True), \
                 patch.object(diagnostic, "read_detail_csv", return_value=orderable), \
                 patch.object(diagnostic.pd, "read_csv", return_value=selected), \
                 patch.object(diagnostic, "paired_trade_r_conversion_diagnostic", return_value={}), \
                 patch.object(diagnostic, "publish_detail_csv"):
                self.assertTrue(diagnostic.backfill_pair_selection_diagnostics(
                    payload, pair_dir=root, project_root=root, rank_composition_score_loader=loader))
                loader.assert_called_once_with()
                self.assertFalse(payload["selection_diagnostics"]["future_target_used_for_runtime_sort"])
