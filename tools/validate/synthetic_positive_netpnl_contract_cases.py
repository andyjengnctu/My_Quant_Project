"""Isolated reusable single-ranker NetPnL score/selection primitive regressions.

Only isolated synthetic data and a one-epoch CPU model roundtrip are used.
No real market files, strategy optimizer or formal test runner is used.
"""
from __future__ import annotations

import sys
import json
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.positive_netpnl_contract import (positive_labels, positive_score_admitted,
                                           admit_positive_score_rows, POSITIVE_SCORE_CONTRACT, POSITIVE_P_CLASS_COLUMN)
from core.breakout_quality_runtime import (
    get_continuous_ranker_training_policy,
    get_continuous_ranker_score_output_columns,
)
from core.strategy_compare_policy import get_strategy_comparison_settings
from config.strategy_compare import STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE
from services.research.target_oracle_application import get_formal_oracle_definition
from core.audit_policy import get_active_audit_module_id
from core.audit_registry import (resolve_current_oracle_comparisons, oracle_comparison_key,
                                 oracle_target_semantics)
from core.report_contract_versions import load_report_contract_versions
from core.research_report_contract import table_contract
from services.audit.target_oracle_romd import (
    _oracle_comparison_plan, _contract, _resolve_oracle_target_score_contracts,
    _oracle_merge_scoped_reports,
)
from services.research.oracle_artifact_contract import reference_arm_execution_binding
from services.research.target_dl_overview import (
    _oracle_rows, _strategy_bindings, collect_target_dl_overview, render_target_dl_overview,
)
from core.breakout_quality_ranking_context import breakout_quality_ranking_source_context
from filters.breakout_quality.ranking_score_store import SCORE_SOURCE_SELECTION_POINT_IN_TIME
from filters.breakout_quality.runtime import (
    audit_oracle_truth_score_context, resolve_breakout_quality_candidate_rank,
)
from core.portfolio_entry_selection import reorder_candidates_for_resource_aware_quality

class PositiveNetPnlContractCases(TestCase):
    def test_positive_labels_are_strict_and_censor_missing(self):
        labels = positive_labels([22.0, 0.0, -0.001, float('nan'), float('inf')])
        self.assertEqual(labels[:3].tolist(), [1.0, 0.0, 0.0])
        self.assertTrue(np.isnan(labels[3]))
        self.assertTrue(np.isnan(labels[4]))
        with self.assertRaises(ValueError):
            positive_labels([[2.0]])

    def test_scientific_target_and_selector_names_do_not_mutate_signed_receipt_ids(self):
        from core.audit_registry import (oracle_scientific_target_name,
                                         oracle_target_semantics)
        d = get_formal_oracle_definition(get_active_audit_module_id())
        rows = resolve_current_oracle_comparisons(d.source)
        self.assertEqual(len(rows), len({oracle_target_semantics(x) for x in rows}))
        by_legacy = {oracle_comparison_key(x): x for x in rows}
        cases = (
            ("MFE40_Control", "MFE40", "Exact", "Breakout"),
            ("MFE40_Direct", "MFE40", "Direct", "Breakout"),
            ("DailyUniversal_MinTrade_NetPnL", "MinTrade_NetPnL", "Exact", "Breakout"),
            ("DailyUniversal_MinTrade_NetPnL_Direct", "MinTrade_NetPnL", "Direct", "Breakout"),
            ("DailyUniversal_MinTrade_NetPnL_Positive", "MinTrade_NetPnL_Positive", "Exact", "Breakout"),
            ("DailyUniversal_MinTrade_NetPnL_Positive_Direct", "MinTrade_NetPnL_Positive", "Direct", "Breakout"),
            ("TrueDailyUniversal_MinTrade_NetPnL_Positive", "DailyUniversal_MinTrade_NetPnL_Positive", "Exact", "Breakout"),
            ("TrueDailyUniversal_MinTrade_NetPnL_Positive::Daily", "DailyUniversal_MinTrade_NetPnL_Positive", "Exact", "Daily"),
            ("TrueDailyUniversal_MinTrade_NetPnL_Positive_Direct", "DailyUniversal_MinTrade_NetPnL_Positive", "Direct", "Breakout"),
            ("TrueDailyUniversal_MinTrade_NetPnL_Positive_Direct::Daily", "DailyUniversal_MinTrade_NetPnL_Positive", "Direct", "Daily"),
            ("TrueDailyUniversal_MinTrade_NetPnL::Daily", "DailyUniversal_MinTrade_NetPnL", "Exact", "Daily"),
            ("TrueDailyUniversal_MinTrade_NetPnL_Direct::Daily", "DailyUniversal_MinTrade_NetPnL", "Direct", "Daily"),
        )
        for receipt, target, selector, strategy in cases:
            with self.subTest(receipt=receipt):
                self.assertEqual(oracle_target_semantics(by_legacy[receipt]), (target, selector, strategy))
                self.assertEqual(oracle_scientific_target_name(by_legacy[receipt]["target"]), target)
                self.assertEqual(oracle_comparison_key(by_legacy[receipt]), receipt)
        self.assertNotEqual(oracle_target_semantics(by_legacy["DailyUniversal_MinTrade_NetPnL"])[0],
                            oracle_target_semantics(by_legacy["DailyUniversal_MinTrade_NetPnL_Positive"])[0])

    def test_current_positive_models_are_continuous_single_rankers(self):
        from core.breakout_quality_registry import get_breakout_quality_experiment_profile
        from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
        from core.positive_netpnl_contract import POSITIVE_RANK_CALIBRATION
        settings = get_strategy_comparison_settings(STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE)
        arms = [arm for arm in settings.enabled_arms
                if (arm.dl_runtime_options or {}).get("positive_score_column")]
        self.assertEqual(len(arms), 6)
        profiles = {settings.dl_sources[arm.dl_id].experiment_profile for arm in arms}
        self.assertEqual(len(profiles), 2)
        for name in profiles:
            profile = get_breakout_quality_experiment_profile(name)
            recipe = get_continuous_ranker_execution_recipe(name)
            self.assertGreaterEqual(profile.fit_contract_version, 1)
            self.assertEqual(profile.score_calibration, POSITIVE_RANK_CALIBRATION)
            self.assertEqual(recipe.training_policy.target_builder, "scalar_pairwise")
            self.assertTrue(recipe.training_policy.uses_pairwise_loss)
            self.assertIsNone(recipe.training_policy.categorical_class_count)
            self.assertEqual(recipe.training_policy.head_loss_component_count, 1)
            self.assertEqual(recipe.objective_policy.pairwise_reduction, "full_list_delta_ndcg_weighted")
        for arm in arms:
            self.assertNotIn("positive_dl_id", arm.dl_runtime_options)

    def test_no_filter_on_original_netpnl_and_positive_filter_fail_closed(self):
        rank = {'available': True,'score':0.70,'score_date':'2024-10-01',
                'positive_available':True,'positive_score_date':'2024-10-01',
                'positive_probability':0.51}
        options = {'positive_score_column':POSITIVE_P_CLASS_COLUMN, 'positive_prediction_contract':POSITIVE_SCORE_CONTRACT,'positive_probability_threshold':0.5}
        row = {'breakout_quality_rank':rank,'breakout_quality_ranking_options':options}
        self.assertTrue(positive_score_admitted(row))
        for value in (0.5, 0.49, None, float('nan'), 1.1):
            bad = {**rank,'positive_probability':value}
            self.assertFalse(positive_score_admitted({**row,'breakout_quality_rank':bad}))
        self.assertFalse(positive_score_admitted({**row,'breakout_quality_rank':{**rank,'positive_available':False}}))
        self.assertFalse(positive_score_admitted({**row,'breakout_quality_rank':{**rank,'positive_score_date':'2024-09-30'}}))
        self.assertTrue(positive_score_admitted({**row,'breakout_quality_ranking_options':{}}))
        admitted=admit_positive_score_rows([
            row, {**row,'breakout_quality_rank':{**rank,'positive_probability':0.05}},
        ])
        self.assertEqual(admitted,[row])

    def test_gate_runs_before_even_free_slot_short_circuit(self):
        rank={'available': True,'score':0.88,'score_date':'2024-10-01',
              'positive_available':True,'positive_score_date':'2024-10-01','positive_probability':0.1}
        row={'breakout_quality_rank':rank,
             'breakout_quality_ranking_policy':'resource-aware-continuous-score-no-k-no-r0',
             'breakout_quality_ranking_options':{'positive_score_column':POSITIVE_P_CLASS_COLUMN, 'positive_prediction_contract':POSITIVE_SCORE_CONTRACT}}
        order,diag=reorder_candidates_for_resource_aware_quality(
            [row],available_cash=0,sizing_equity=100,
            pre_market_occupied=10,max_positions=10,params=None)
        self.assertEqual(order,[])

    def test_pit_rank_and_positive_use_one_profile_checkpoint_and_date(self):
        settings=get_strategy_comparison_settings(STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE)
        arm=settings.arms['C126']
        source=settings.dl_sources[arm.dl_id]
        options=dict(arm.dl_runtime_options)
        columns=[]
        lookup_calls=[]
        def fake_lookup(**kwargs):
            column=kwargs.get('score_column')
            columns.append(column)
            lookup_calls.append(kwargs)
            return {'score': .82 if column is None else .61,
                    'available':True, 'score_date':'2024-10-01',
                    'score_source':SCORE_SOURCE_SELECTION_POINT_IN_TIME}
        with breakout_quality_ranking_source_context(
            score_source=SCORE_SOURCE_SELECTION_POINT_IN_TIME,
            model_architecture=source.model_architecture,
            experiment_profile=source.experiment_profile,
            ranking_policy=arm.dl_runtime_mode,
            ranking_options=options,
        ), patch('filters.breakout_quality.runtime.lookup_selection_point_in_time_candidate_score',
                 side_effect=fake_lookup):
            predicted=resolve_breakout_quality_candidate_rank(
                ticker='2330',signal_date='2024-10-01',
                information_date='2024-10-01',high_len=40)
        self.assertEqual(columns,[None,'positive_p_positive'])
        self.assertEqual({k:v for k,v in lookup_calls[0].items() if k != 'score_column'},
                         {k:v for k,v in lookup_calls[1].items() if k != 'score_column'})
        self.assertEqual(predicted['score'],.82)
        self.assertEqual(predicted['positive_probability'],.61)
        self.assertTrue(positive_score_admitted({
            'breakout_quality_rank':predicted,
            'breakout_quality_ranking_options':arm.dl_runtime_options,
        }))

    def test_oracle_uses_identical_score_boundary_and_selector(self):
        definition = get_formal_oracle_definition(get_active_audit_module_id())
        plan=_oracle_comparison_plan(definition)
        settings, baseline, exact=_contract(definition)
        contracts=_resolve_oracle_target_score_contracts(definition,settings,plan)
        expected={'DailyUniversal_MinTrade_NetPnL_Positive': 'C125',
                  'DailyUniversal_MinTrade_NetPnL_Positive_Direct':'C126'}
        for target,arm_id in expected.items():
            self.assertEqual(contracts[target]['reference_arm_id'],arm_id)
            self.assertEqual(contracts[target]['perfect_positive_output'],'netpnl_positive_probability_v1')
            arm = settings.arms[arm_id]
            self.assertEqual(arm.dl_id, 'CONT13CP_ROLL')
            self.assertEqual(contracts[target]['positive_score_source_id'], arm.dl_id)
            with audit_oracle_truth_score_context(
                {('2330','2024-10-01'): 0.77, ('2317','2024-10-01'): 0.12},
                ranking_policy=arm.dl_runtime_mode,
                ranking_options=arm.dl_runtime_options,
                positive_scores={('2330','2024-10-01'):1.0,('2317','2024-10-01'):0.0},
                key_mode='information_date',
                missing_key_policy='unavailable_outside_mature_feature_universe',
                score_start='2024-01-01'):
                first=resolve_breakout_quality_candidate_rank(ticker='2330',
                    signal_date='2024-10-01',information_date='2024-10-01', high_len=40)
                second=resolve_breakout_quality_candidate_rank(ticker='2317',
                    signal_date='2024-10-01',information_date='2024-10-01', high_len=40)
                for candidate,expected_pass in ((first,True),(second,False)):
                    self.assertEqual(positive_score_admitted({
                        'breakout_quality_rank':candidate,'breakout_quality_ranking_options':arm.dl_runtime_options,
                    }),expected_pass)
                self.assertEqual(first['score'],.77)
                self.assertEqual(second['score'],.12)
            with self.assertRaises(ValueError):
                with audit_oracle_truth_score_context(
                    {('2330','2024-10-01'): .77},
                    ranking_policy=arm.dl_runtime_mode,
                    ranking_options=arm.dl_runtime_options,
                ):
                    pass

    def test_existing_completed_oracle_rows_survive_new_positive_target(self):
        definition = get_formal_oracle_definition(get_active_audit_module_id())
        comparisons = resolve_current_oracle_comparisons(definition.source)
        versions = load_report_contract_versions(ROOT)
        approved = versions["audit.target_oracle_romd"][-2]  # Historical v11: Breakout-only.
        seeds = [693545351, 2014432738, 104504529, 762252050,
                 9734312, 1020806211, 116293459, 1234683984]
        positive_exact = "DailyUniversal_MinTrade_NetPnL_Positive"
        positive_direct = positive_exact + "_Direct"
        published = [row for row in comparisons if row["target"] != positive_exact
                     and row.get("strategy", "Breakout") == "Breakout"]
        legacy_keys = {"C96", *(oracle_comparison_key(row) for row in published
                                 if not row.get("strategy_arm_id"))}
        old_definitions = {row["target"]: row["description"] for row in published}
        old_definitions[positive_direct] = (
            "NON-PIT diagnostic only: full-cycle Min NetPnL Direct with strictly "
            "positive canonical raw NetPnL required for every order; NULL/zero/negative rejected"
        )
        payload = {
            "status": "COMPLETED", "read_only": True, "hindsight_non_pit": True,
            "report_contract_id": approved["contract_id"],
            "report_contract_fingerprint": approved["fingerprint"],
            "strategy_profile_id": definition.source["strategy_profile_id"],
            "benchmark_id": "end_to_end_v2", "seeds": seeds,
            "start_date": "2021-01-01", "end_date": "2026-03-02",
            "definitions": old_definitions,
            "baseline": {"target": "C96", "seeds": len(seeds),
                         "romd": 9.17, "win_rate_pct": 43.86},
            "oracles": [
                {"target": row["target"], "seeds": len(seeds),
                 "romd": 204.88 if row["target"] == positive_direct else 24.62,
                 "win_rate_pct": 99.71 if row["target"] == positive_direct else 52.68}
                for row in published
            ],
        }
        with TemporaryDirectory() as temp:
            latest = (Path(temp) / "outputs" / "audit" /
                      definition.output_subdir / "latest")
            latest.mkdir(parents=True)
            (latest / "audit.md").write_text("Existing signed result\n", encoding="utf-8")
            def read_rows():
                (latest / "audit.json").write_text(
                    json.dumps(payload), encoding="utf-8")
                return _oracle_rows(Path(temp), definition, versions)

            targets, _, _, values, info = read_rows()
            self.assertEqual(info["benchmark"], "end_to_end_v2")
            self.assertEqual(info["seeds"], tuple(seeds))
            self.assertEqual(values["C96"]["oracle_romd"], 9.17)
            self.assertEqual(values["MFE40_Control"]["oracle_romd"], 24.62)
            self.assertEqual(values["DailyUniversal_MinTrade_NetPnL_Direct"]["oracle_romd"], 24.62)
            self.assertEqual(set(values), legacy_keys)
            self.assertNotIn(positive_exact, values)  # Never backfill a new Target.
            self.assertNotIn(positive_direct, values)  # Old raw Gate != formal dual-head.
            true_daily = {row["target"] for row in comparisons
                          if row.get("strategy") == "Daily"}
            self.assertTrue(true_daily)
            self.assertTrue({name + "::Daily" for name in true_daily}.isdisjoint(values))
            self.assertTrue(true_daily.isdisjoint(values))  # Old unverified binding is not re-certified.
            scientific_daily = {oracle_target_semantics(row)[0] for row in comparisons
                                if row.get("strategy") == "Daily"}
            ledger = Path(temp) / "doc/scientific_contracts/REPORT_CONTRACT_VERSIONS.json"
            ledger.parent.mkdir(parents=True)
            ledger.write_bytes((ROOT / "doc/scientific_contracts/REPORT_CONTRACT_VERSIONS.json").read_bytes())
            overview = collect_target_dl_overview(Path(temp))
            rendered = render_target_dl_overview(overview)
            self.assertIn("Oracle: end_to_end_v2 | N=8 | 2021-01-01 ~ 2026-03-02", rendered)
            self.assertNotIn("no completed matching local report", rendered)
            rows = [row for row in overview["rows"] if row["target"] == "MFE40" and row["selector"] == "Exact"]
            self.assertTrue(rows)
            self.assertTrue(all(row["oracle_romd"] == 24.62 for row in rows))
            self.assertTrue(all(row["oracle_romd"] is None for row in overview["rows"]
                                if row["target"] == "MinTrade_NetPnL_Positive"))
            self.assertTrue(all(row["oracle_romd"] is None for row in overview["rows"]
                                if row["target"] in scientific_daily and row["entry_strategy"] == "Daily"))
            self.assertTrue(all(row["oracle_romd"] is None for row in overview["rows"]
                                if row["target"] in scientific_daily and row["entry_strategy"] == "Breakout"))
            daily_overview_rows = [row for row in overview["rows"]
                                   if row["target"] in scientific_daily]
            self.assertTrue(daily_overview_rows)
            self.assertTrue(all(row["oracle_romd"] is None for row in daily_overview_rows))
            self.assertEqual(overview["rows"][0]["entry_strategy"], "Breakout")
            self.assertEqual({row["entry_strategy"] for row in overview["rows"]
                              if row["target"] in scientific_daily}, {"Breakout", "Daily"})
            self.assertEqual(
                table_contract("research.target_dl_validation_overview", "comparison", "target_dl").headers[:3],
                ("Target", "Strategy", "Selector"),
            )

            # A newly completed, matching report may publish both Positive rows.
            current = versions["audit.target_oracle_romd"][-1]
            payload["report_contract_id"] = current["contract_id"]
            payload["report_contract_fingerprint"] = current["fingerprint"]
            payload["definitions"][positive_direct] = next(
                row["description"] for row in comparisons if row["target"] == positive_direct)
            payload["definitions"][positive_exact] = next(
                row["description"] for row in comparisons if row["target"] == positive_exact)
            payload["oracles"].append({"target": positive_exact, "seeds": len(seeds),
                                       "romd": 25.0, "win_rate_pct": 60.0})
            # Same Target, independent Strategy results: Breakout receipts
            # retain their old identity; Daily may only use its own evidence.
            for target in sorted(true_daily):
                payload["definitions"][target] = next(
                    row["description"] for row in comparisons if row["target"] == target)
                payload["oracles"].append({
                    "target": target, "strategy": "Daily", "seeds": len(seeds),
                    "romd": 41.0, "win_rate_pct": 96.0,
                })
            payload["comparison_inventory"] = list(targets)
            # Legacy aggregate reports have no per-row source-arm receipt.
            # Old Breakout remains readable, but previous C131/C132 Daily
            # results must not be re-labelled as C133/C134 evidence.
            _, _, _, values, info = read_rows()
            self.assertEqual(set(values), legacy_keys)
            self.assertNotIn("TrueDailyUniversal_MinTrade_NetPnL_Positive::Daily", values)
            self.assertNotIn("TrueDailyUniversal_MinTrade_NetPnL_Positive_Direct::Daily", values)
            settings = get_strategy_comparison_settings(STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE)
            payload["comparison_reference_arms"] = {
                oracle_comparison_key(row): reference_arm_execution_binding(row, settings)
                for row in comparisons if row.get("strategy_arm_id")
            }
            _, _, _, values, info = read_rows()
            self.assertEqual(len(values), len(targets))
            self.assertEqual(values[positive_direct]["oracle_romd"], 204.88)
            self.assertEqual(values["TrueDailyUniversal_MinTrade_NetPnL_Positive_Direct"]["oracle_romd"], 24.62)
            self.assertEqual(values["TrueDailyUniversal_MinTrade_NetPnL_Positive_Direct::Daily"]["oracle_romd"], 41.0)
            daily_direct_key = "TrueDailyUniversal_MinTrade_NetPnL_Positive_Direct::Daily"
            original_receipt = payload["comparison_reference_arms"][daily_direct_key]
            payload["comparison_reference_arms"][daily_direct_key] = {
                "arm_id": "C132", "execution_sha256": original_receipt["execution_sha256"],
            }
            _, _, _, values, info = read_rows()
            # A retired dual-source arm is not an equivalent reference name.
            self.assertNotIn(daily_direct_key, values)
            payload["comparison_reference_arms"][daily_direct_key] = {
                "arm_id": "C132", "execution_sha256": "f" * 64,
            }
            self.assertNotIn(daily_direct_key, read_rows()[3])
            self.assertEqual(values["TrueDailyUniversal_MinTrade_NetPnL_Positive_Direct"]["oracle_romd"], 24.62)
            payload["comparison_reference_arms"][daily_direct_key] = original_receipt

            # An incomplete/corrupt original report must not supply partial metrics.
            payload["oracles"].pop()
            _, _, _, values, info = read_rows()
            self.assertEqual(values, {})
            self.assertIsNone(info["path"])

            # A missing published inventory or a bad official fingerprint is invalid.
            payload["oracles"].append({"target": positive_exact, "seeds": len(seeds),
                                       "romd": 25.0, "win_rate_pct": 60.0})
            definitions = payload.pop("definitions")
            self.assertEqual(read_rows()[3], {})
            payload["definitions"] = definitions
            payload["report_contract_fingerprint"] = "incorrect"
            self.assertEqual(read_rows()[3], {})

    def test_scoped_publisher_signs_actual_reference_arm_for_each_strategy(self):
        definition = get_formal_oracle_definition(get_active_audit_module_id())
        comparisons = resolve_current_oracle_comparisons(definition.source)
        baseline = {"target": "C96", "seeds": 8, "romd": 9.17}
        reports = []
        for strategy in ("Breakout", "Daily"):
            subset = [row for row in comparisons if row.get("strategy", "Breakout") == strategy]
            report = {
                "baseline": baseline,
                "seeds": list(range(8)), "benchmark_id": "end_to_end_v2",
                "dataset": "full", "start_date": "2021-01-01", "end_date": "2026-03-02",
                "common_cohort": {"cohort_rows": 630589}, "param_artifacts": {},
                "strategy_profile_id": definition.source["strategy_profile_id"],
                "definitions": {row["target"]: row["description"] for row in subset},
                "oracles": [{"target": row["target"], "strategy": strategy} for row in subset],
            }
            reports.append((strategy, report))
        merged = _oracle_merge_scoped_reports(definition, reports)
        settings = get_strategy_comparison_settings(STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE)
        references = merged["comparison_reference_arms"]
        for row in comparisons:
            if not row.get("strategy_arm_id"):
                continue
            key = oracle_comparison_key(row)
            self.assertEqual(references[key], reference_arm_execution_binding(row, settings))
        true_daily = "TrueDailyUniversal_MinTrade_NetPnL_Positive"
        self.assertEqual(references[true_daily]["arm_id"], "C135")
        self.assertEqual(references[true_daily + "::Daily"]["arm_id"], "C133")
        self.assertNotEqual(references[true_daily]["execution_sha256"],
                            references[true_daily + "::Daily"]["execution_sha256"])

    def test_report_resolves_positive_exact_and_direct_without_ambiguity(self):
        definition = get_formal_oracle_definition(get_active_audit_module_id())
        targets,_,score_targets,_,_=_oracle_rows(ROOT,definition,{})
        bindings=_strategy_bindings(target_names=targets,score_targets=score_targets,definition=definition)
        expected={
            ('DailyUniversal_MinTrade_NetPnL','daily_universal_formal_mintrade_fullcycle_netpnl_per_seed_full_list_ndcg_pairwise'):'C123',
            ('DailyUniversal_MinTrade_NetPnL_Direct','daily_universal_formal_mintrade_fullcycle_netpnl_per_seed_full_list_ndcg_pairwise'):'C124',
            ('DailyUniversal_MinTrade_NetPnL_Positive','daily_universal_formal_mintrade_fullcycle_netpnl_positive_binary'):'C125',
            ('DailyUniversal_MinTrade_NetPnL_Positive_Direct','daily_universal_formal_mintrade_fullcycle_netpnl_positive_binary'):'C126',
        }
        for key,value in expected.items():
            self.assertEqual(bindings.get(key),value)
        # All four TrueDaily arms consume one CT FIT; never the event CP FIT.
        for suffix in ("", "_Direct"):
            name = "TrueDailyUniversal_MinTrade_NetPnL_Positive" + suffix
            self.assertEqual(
                [(profile, arm) for (target, profile), arm in bindings.items() if target == name],
                [("true_daily_universal_mintrade_fullcycle_netpnl_positive_binary",
                  "C136" if suffix else "C135")])
            daily_key = name + "::Daily"
            expected_arm = "C134" if suffix else "C133"
            self.assertEqual(
                [(profile, arm) for (target, profile), arm in bindings.items()
                 if target == daily_key],
                [("true_daily_universal_mintrade_fullcycle_netpnl_positive_binary", expected_arm)],
            )


class PositiveRankCalibrationCases(TestCase):
    """One scalar output, PIT-only calibration, signed checkpoint roundtrip."""

    def _fixture(self):
        from types import SimpleNamespace
        import pandas as pd
        from core.breakout_quality_registry import (
            get_breakout_quality_experiment_profile, DAILY_UNIVERSAL_NETPNL_POSITIVE_BINARY_PROFILE,
        )
        from filters.breakout_quality.positive_rank_calibration import fit_score_calibration
        profile = get_breakout_quality_experiment_profile(DAILY_UNIVERSAL_NETPNL_POSITIVE_BINARY_PROFILE)
        days = pd.to_datetime(['2020-01-01'] * 2 + ['2020-01-06'] * 4 + ['2020-02-01'])
        groups = pd.DataFrame({'ticker': list('ABCDEFG'), 'date': days,
                               'label_eval_end_date': days + pd.Timedelta(days=1)})
        raw = np.array([-2., 3., -30., 0., 5., 200., np.nan])
        args = dict(scores=np.array([.1, .3, .7, .9]), raw_target=raw,
                    group_table=groups, train_ids=np.arange(2), validation_ids=np.arange(2, 6),
                    information_cutoff='2020-01-08', selected_epoch=1, profile=profile)
        return profile, SimpleNamespace(), args, fit_score_calibration(**args)

    def test_monotone_positive_does_not_replace_or_collapse_continuous_rank(self):
        from filters.breakout_quality.positive_rank_calibration import attach_score_calibration, calibrated_score_columns
        profile, model, _, payload = self._fixture()
        attach_score_calibration(model, profile, {'score_calibration': payload}, selected_epoch=1)
        scores = np.array([.1, .3, .7, .9]); before = scores.copy()
        probability = calibrated_score_columns(model, profile, scores)[POSITIVE_P_CLASS_COLUMN]
        np.testing.assert_array_equal(scores, before)
        self.assertEqual(probability.tolist(), [0., 0., 1., 1.])
        self.assertLess(scores[2], scores[3])  # Two profits still have distinct priority.

    def test_only_inner_validation_labels_determine_curve(self):
        from filters.breakout_quality.positive_rank_calibration import fit_score_calibration
        _, _, args, baseline = self._fixture()
        changed = args['raw_target'].copy(); changed[-1] = 1e12
        self.assertEqual(fit_score_calibration(**{**args, 'raw_target': changed}), baseline)
        changed = args['group_table'].copy(); changed.loc[6, 'label_eval_end_date'] = '2099-01-01'
        self.assertEqual(fit_score_calibration(**{**args, 'group_table': changed}), baseline)

    def test_overlap_censor_maturity_and_cutoff_are_rejected(self):
        from filters.breakout_quality.positive_rank_calibration import fit_score_calibration
        _, _, args, _ = self._fixture()
        invalid = [dict(train_ids=[0, 2]), dict(validation_ids=[2, 2, 4, 5]),
                   dict(information_cutoff='2020-01-06'), dict(scores=[.1, .2, .3, np.nan])]
        raw = args['raw_target'].copy(); raw[2] = np.nan
        invalid.append(dict(raw_target=raw))
        groups = args['group_table'].copy(); groups.loc[0, 'label_eval_end_date'] = '2020-01-06'
        invalid.append(dict(group_table=groups))
        for override in invalid:
            with self.subTest(override=list(override)), self.assertRaises(ValueError):
                fit_score_calibration(**{**args, **override})

    def test_constant_score_and_single_class_are_legal_not_invented_labels(self):
        from filters.breakout_quality.positive_rank_calibration import fit_score_calibration, validate_score_calibration
        _, _, args, _ = self._fixture()
        same = fit_score_calibration(**{**args, 'scores': np.full(4, .7)})
        np.testing.assert_array_equal(validate_score_calibration(same).predict([0., 1.]), [.5, .5])
        raw = args['raw_target'].copy(); raw[2:6] = 1.
        all_positive = fit_score_calibration(**{**args, 'raw_target': raw})
        np.testing.assert_array_equal(validate_score_calibration(all_positive).predict([.1, .9]), [1., 1.])
        self.assertEqual(float(positive_labels([1e-300])[0]), 1.)

    def test_missing_corrupt_wrong_epoch_and_wrong_profile_checkpoint_fail(self):
        from copy import deepcopy
        from core.breakout_quality_registry import get_breakout_quality_experiment_profile, TRUE_DAILY_MINTRADE_NETPNL_POSITIVE_PROFILE
        from filters.breakout_quality.positive_rank_calibration import attach_score_calibration, restore_score_calibration
        profile, model, _, payload = self._fixture()
        with self.assertRaises(ValueError):
            restore_score_calibration(model, profile, {'selected_epoch': 1})
        corrupt = deepcopy(payload); corrupt['y'][0] = .99
        for candidate, epoch in ((corrupt, 1), (payload, 2)):
            with self.assertRaises(ValueError):
                attach_score_calibration(model, profile, {'score_calibration': candidate}, selected_epoch=epoch)
        with self.assertRaises(ValueError):
            attach_score_calibration(model, get_breakout_quality_experiment_profile(TRUE_DAILY_MINTRADE_NETPNL_POSITIVE_PROFILE),
                                     {'score_calibration': payload}, selected_epoch=1)

    def test_retired_dual_source_and_unknown_options_cannot_disable_gate(self):
        from core.positive_netpnl_contract import requires_positive_score
        with self.assertRaises(ValueError):
            positive_score_admitted({'breakout_quality_ranking_options': 'invalid'})
        for options in ({'positive_dl_id': 'retired'}, {'positive_dl_id': ''},
                        {'positive_score_column': POSITIVE_P_CLASS_COLUMN}, {'positive_typo': True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                requires_positive_score(options)

    def test_old_well_formed_oracle_schema_is_stale_not_corrupt(self):
        from services.research.oracle_artifact_contract import oracle_ranking_options
        from core.positive_netpnl_contract import requires_positive_score
        old = {'positive_dl_id': 'CONT13CP_ROLL', 'positive_probability_threshold': .5}
        current = {'positive_score_column': POSITIVE_P_CLASS_COLUMN,
                   'positive_prediction_contract': POSITIVE_SCORE_CONTRACT,
                   'positive_probability_threshold': .5}
        self.assertEqual(oracle_ranking_options(old), old)
        self.assertNotEqual(oracle_ranking_options(old), oracle_ranking_options(current))
        with self.assertRaises(ValueError):
            requires_positive_score(old)  # Historical metadata read != permission to execute.

    def test_actual_scalar_fit_checkpoint_reload_for_each_calibrated_profile(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from types import SimpleNamespace
        import pandas as pd
        import torch
        from services.breakout_quality import train_continuous_ranker as trainer, continuous_ranker_pipeline as pipeline
        from services.breakout_quality.fitted_model_artifacts import load_model_from_fitted_checkpoint
        from filters.breakout_quality.torch_runtime import resolve_torch_execution_plan
        from filters.breakout_quality.models.spec import get_model_spec
        from filters.breakout_quality.continuous_ranker_data import build_same_date_percentile_targets
        from core.breakout_quality_registry import get_breakout_quality_experiment_profile
        settings = get_strategy_comparison_settings(STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE)
        profiles = sorted({settings.dl_sources[arm.dl_id].experiment_profile
                           for arm in settings.enabled_arms
                           if (arm.dl_runtime_options or {}).get('positive_score_column')})
        old_threads = torch.get_num_threads(); torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        plan = resolve_torch_execution_plan(torch, requested_device='cpu', mixed_precision=False,
                mixed_precision_dtype='auto', deterministic_algorithms=True, allow_tf32=False)
        features = np.random.default_rng(42).normal(size=(24, 300, 10)).astype(np.float32)
        context = np.empty((24, 0), dtype=np.float32)
        raw = np.tile([-40., -10., 0., 15., 60., 200.], 4).astype(np.float32)
        days = np.repeat(pd.to_datetime(['2020-01-01','2020-01-03','2020-01-06','2020-02-01']).to_numpy(), 6)
        groups = pd.DataFrame({'date': days, 'label_eval_end_date': days + np.timedelta64(1,'D'),
                              'ticker': np.tile([str(1000+i) for i in range(6)], 4), 'label': (raw > 0).astype(int)})
        percentiles = build_same_date_percentile_targets(raw, np.ones(24, dtype=bool), groups['date'])
        for name in profiles:
            with self.subTest(profile=name), redirect_stdout(StringIO()):
                args = trainer.parse_args(['--experiment-profile', name, '--model-architecture', 'inception_time_v1',
                    '--epochs', '1', '--batch-size', '64', '--evaluation-batch-size', '64', '--device', 'cpu',
                    '--no-mixed-precision', '--train-prefetch-batches', '0'])
                args.train_prefetch_workers = 0
                selected = trainer.select_epoch(torch, features, context, groups, raw, percentiles,
                    np.arange(12), np.arange(12,18), args=args, plan=plan, evaluate_train_metrics=False,
                    information_cutoff='2020-01-08')
                model, _ = trainer.fit_final(torch, features, context, raw, percentiles, groups,
                    np.arange(18), epochs=selected['best_epoch'], args=args, plan=plan, epoch_selection=selected)
                bundle = SimpleNamespace(profile=get_breakout_quality_experiment_profile(name),
                    model_spec=get_model_spec('inception_time_v1'), feature_bank=features, group_context=context,
                    group_table=groups, target_manifest={}, raw_target=raw)
                scores, columns = pipeline.predict_score_output_payload(torch, model, bundle,
                    np.arange(18,24), batch_size=64, plan=plan)
                payload = pipeline.build_checkpoint_payload(model, bundle, args=args, plan=plan,
                    selected_epoch=selected['best_epoch'], fold_contract={})
                with TemporaryDirectory() as directory:
                    path = Path(directory) / 'model.pt'; torch.save(payload, path)
                    contract = SimpleNamespace(model_path=path, selected_epoch=selected['best_epoch'], manifest={}, source='synthetic')
                    restored = load_model_from_fitted_checkpoint(torch, contract=contract, bundle=bundle, plan=plan)
                    actual, restored_columns = pipeline.predict_score_output_payload(torch, restored, bundle,
                        np.arange(18,24), batch_size=64, plan=plan)
                    np.testing.assert_array_equal(scores, actual)
                    np.testing.assert_array_equal(columns[POSITIVE_P_CLASS_COLUMN], restored_columns[POSITIVE_P_CLASS_COLUMN])
                    payload.pop('score_calibration'); torch.save(payload, path)
                    with self.assertRaises(ValueError):
                        load_model_from_fitted_checkpoint(torch, contract=contract, bundle=bundle, plan=plan)

class ScoreOutputCompletenessCases(TestCase):
    """Registry-driven persisted-output checks, also used by the PIT contract case."""

    @staticmethod
    def profiles():
        from core.breakout_quality_registry import (
            SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES,
            get_breakout_quality_experiment_profile,
        )
        return [profile for name in SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES
                if (profile := get_breakout_quality_experiment_profile(name)).score_calibration]

    def test_calibrated_output_is_required_by_manifest_and_fold(self):
        from types import SimpleNamespace
        import pandas as pd
        from services.breakout_quality.point_in_time_scores import (
            _score_output_columns, _score_output_capability, _validate_score_frame,
        )
        from filters.breakout_quality.ranking_score_store import _validate_score_output_capability_contract
        self.assertTrue(self.profiles())
        for profile in self.profiles():
            with self.subTest(profile=profile.name):
                columns = get_continuous_ranker_score_output_columns(
                    profile.training_objective, score_calibration=profile.score_calibration,
                )
                self.assertIn(POSITIVE_P_CLASS_COLUMN, columns.values())
                self.assertEqual(_score_output_columns(SimpleNamespace(profile=profile)), columns)
                self.assertEqual(_score_output_capability(SimpleNamespace(profile=profile))["columns"], columns)
                _validate_score_output_capability_contract({"score_columns": columns}, profile=profile)
                with self.assertRaises(ValueError):
                    _validate_score_output_capability_contract(
                        {"score_columns": {"primary": "breakout_quality_score"}}, profile=profile)
                contract = {"experiment_profile": profile.name, "fold_id": "fold",
                            "model_information_cutoff": "2020-12-31", "score_output_contract": columns,
                            "planned_periods": {"score_start": "2021-01-04", "score_end": "2021-01-04"}}
                frame = pd.DataFrame({"ticker": ["2330"], "date": ["2021-01-04"], "group_index": [0],
                                      "breakout_quality_score": [.7], "fold_id": ["fold"],
                                      "model_information_cutoff": ["2020-12-31"]})
                with self.assertRaises(ValueError):
                    _validate_score_frame(frame, fold_contract=contract)
                frame[POSITIVE_P_CLASS_COLUMN] = .6
                self.assertIn(POSITIVE_P_CLASS_COLUMN, _validate_score_frame(frame, fold_contract=contract))

    def test_forward_writer_and_both_readers_require_same_positive_output(self):
        import pandas as pd
        from services.breakout_quality.train_daily_ranker import _write_forward_score_frame
        from filters.breakout_quality.ranking_score_store import (
            load_continuous_ranker_oos_score_table_from_path,
            load_selection_point_in_time_score_table_from_path,
        )
        for profile in self.profiles():
            with self.subTest(profile=profile.name), TemporaryDirectory() as folder:
                path = Path(folder) / "forward.csv.gz"
                frame = pd.DataFrame({"ticker": ["2330", "2317"], "date": ["2021-01-04"] * 2,
                                      "group_index": [0, 1], "model_score": [.2, .8]})
                with self.assertRaises(ValueError):
                    _write_forward_score_frame(frame, path, profile=profile, score_extras={})
                self.assertFalse(path.exists())
                extras = {POSITIVE_P_CLASS_COLUMN: [.1, .9]}
                _write_forward_score_frame(frame, path, profile=profile, score_extras=extras)
                loaded = load_continuous_ranker_oos_score_table_from_path(str(path), profile.name)
                self.assertIn(POSITIVE_P_CLASS_COLUMN, loaded)
                original = pd.read_csv(path, dtype={"ticker": str})
                np.testing.assert_allclose(original["model_score"], frame["model_score"])
                np.testing.assert_allclose(original[POSITIVE_P_CLASS_COLUMN], extras[POSITIVE_P_CLASS_COLUMN])
                frame.to_csv(path, index=False, compression="gzip")
                with self.assertRaises(ValueError):
                    load_continuous_ranker_oos_score_table_from_path(str(path), profile.name)
                pit = frame.rename(columns={"model_score": "breakout_quality_score"}).assign(
                    fold_id="fold", model_information_cutoff="2020-12-31")
                pit.to_csv(path, index=False, compression="gzip")
                with self.assertRaises(ValueError):
                    load_selection_point_in_time_score_table_from_path(str(path), experiment_profile=profile.name)
                pit[POSITIVE_P_CLASS_COLUMN] = extras[POSITIVE_P_CLASS_COLUMN]
                pit.to_csv(path, index=False, compression="gzip")
                self.assertIn(POSITIVE_P_CLASS_COLUMN, load_selection_point_in_time_score_table_from_path(
                    str(path), experiment_profile=profile.name))

    def test_retired_predictors_and_raw_outcome_gate_cannot_execute(self):
        from importlib.util import find_spec
        from core.breakout_quality_registry import get_breakout_quality_experiment_profile
        from core.strategy_compare_registry import STRATEGY_DL_SOURCES
        from services.research.oracle_artifact_contract import execution_context
        with self.assertRaises(ValueError):
            get_breakout_quality_experiment_profile("true_daily_universal_mintrade_fullcycle_netpnl_rank")
        with self.assertRaises(ValueError):
            get_continuous_ranker_training_policy("daily_netpnl_positive_binary_cross_entropy")
        self.assertNotIn("CONT13CS_ROLL", STRATEGY_DL_SOURCES)
        self.assertIsNone(find_spec("core.oracle_positive_gate"))
        self.assertIsNone(find_spec("services.audit.positive_gate_evidence"))
        with self.assertRaisesRegex(ValueError, "Retired raw-outcome"):
            execution_context(dataset="full", price_sha256="a" * 64, param_sha256="b" * 64,
                seed=42, target="test", start="2021-01-01", end="2021-01-04", cutoff="2021-01-04",
                max_positions=10, evaluation_mode="rolling", ranking_policy="test", ranking_options={},
                replay_policy={}, score_contract=None, source_contract=None, dataset_inputs=None,
                execution_gate="fullcycle_reference_net_pnl_strictly_positive_v1")


if __name__ == '__main__':
    main(verbosity=2)
