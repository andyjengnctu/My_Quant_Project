"""Isolated reusable dual-output NetPnL score/selection primitive regressions.

No model FIT, optimizer, real market files, formal test runner or replay is run.
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
                                           admit_positive_score_rows)
from core.breakout_quality_runtime import (
    TRAINING_OBJECTIVE_DAILY_NETPNL_POSITIVE_BINARY,
    get_continuous_ranker_training_policy,
    get_continuous_ranker_score_output_columns,
)
from core.strategy_compare_policy import get_strategy_comparison_settings
from config.strategy_compare import STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE
from services.research.target_oracle_application import get_formal_oracle_definition
from core.audit_policy import get_active_audit_module_id
from core.audit_registry import resolve_current_oracle_comparisons, oracle_comparison_key
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

    def test_binary_score_contract_is_two_class_and_has_pit_column(self):
        policy = get_continuous_ranker_training_policy(TRAINING_OBJECTIVE_DAILY_NETPNL_POSITIVE_BINARY)
        self.assertEqual(policy.categorical_class_count, 2)
        self.assertEqual(policy.categorical_primary_class, 1)
        self.assertEqual(policy.categorical_score_columns,
                         ('positive_p_nonpositive','positive_p_positive'))
        self.assertEqual(get_continuous_ranker_score_output_columns(TRAINING_OBJECTIVE_DAILY_NETPNL_POSITIVE_BINARY)['class_1'],
                         'positive_p_positive')

    def test_no_filter_on_original_netpnl_and_positive_filter_fail_closed(self):
        rank = {'available': True,'score':0.70,'score_date':'2024-10-01',
                'positive_available':True,'positive_score_date':'2024-10-01',
                'positive_probability':0.51}
        options = {'positive_dl_id':'CONT13CP_ROLL','positive_probability_threshold':0.5}
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
             'breakout_quality_ranking_options':{'positive_dl_id':'CONT13CP_ROLL'}}
        order,diag=reorder_candidates_for_resource_aware_quality(
            [row],available_cash=0,sizing_equity=100,
            pre_market_occupied=10,max_positions=10,params=None)
        self.assertEqual(order,[])

    def test_pit_secondary_prediction_is_independent_of_co_rank(self):
        settings=get_strategy_comparison_settings(STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE)
        arm=settings.arms['C126']
        source=settings.dl_sources[arm.dl_id]
        options={**arm.dl_runtime_options,
            'positive_filter_id':source.filter_id,
            'positive_score_source':SCORE_SOURCE_SELECTION_POINT_IN_TIME,
            'positive_model_architecture':'inception_time_v1',
            'positive_experiment_profile':'daily_universal_formal_mintrade_fullcycle_netpnl_positive_binary'}
        columns=[]
        def fake_lookup(**kwargs):
            column=kwargs.get('score_column')
            columns.append(column)
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
            self.assertEqual(arm.dl_id, settings.arms['C123' if arm_id=='C125' else 'C124'].dl_id)
            self.assertEqual(arm.dl_runtime_options['positive_dl_id'],'CONT13CP_ROLL')
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
            self.assertEqual(len(values), len(targets) - 4)
            self.assertNotIn(positive_exact, values)  # Never backfill a new Target.
            self.assertNotIn(positive_direct, values)  # Old raw Gate != formal dual-head.
            true_daily = {row["target"] for row in comparisons
                          if row.get("strategy") == "Daily"}
            self.assertEqual(len(true_daily), 2)
            self.assertTrue({name + "::Daily" for name in true_daily}.isdisjoint(values))
            self.assertTrue(true_daily.issubset(values))  # Old Breakout survives.
            ledger = Path(temp) / "doc/scientific_contracts/REPORT_CONTRACT_VERSIONS.json"
            ledger.parent.mkdir(parents=True)
            ledger.write_bytes((ROOT / "doc/scientific_contracts/REPORT_CONTRACT_VERSIONS.json").read_bytes())
            overview = collect_target_dl_overview(Path(temp))
            rendered = render_target_dl_overview(overview)
            self.assertIn("Oracle: end_to_end_v2 | N=8 | 2021-01-01 ~ 2026-03-02", rendered)
            self.assertNotIn("no completed matching local report", rendered)
            rows = [row for row in overview["rows"] if row["target"] == "MFE40_Control"]
            self.assertTrue(rows)
            self.assertTrue(all(row["oracle_romd"] == 24.62 for row in rows))
            self.assertTrue(all(row["oracle_romd"] is None for row in overview["rows"]
                                if row["target"] in {positive_exact, positive_direct}))
            self.assertTrue(all(row["oracle_romd"] is None for row in overview["rows"]
                                if row["target"] in true_daily and row["entry_strategy"] == "Daily"))
            self.assertTrue(all(row["oracle_romd"] == 24.62 for row in overview["rows"]
                                if row["target"] in true_daily and row["entry_strategy"] == "Breakout"))
            self.assertEqual(overview["rows"][0]["entry_strategy"], "Breakout")
            self.assertEqual({row["entry_strategy"] for row in overview["rows"]
                              if row["target"] in true_daily}, {"Breakout", "Daily"})
            self.assertEqual(
                table_contract("research.target_dl_validation_overview", "comparison", "target_dl").headers[:2],
                ("Target", "Strategy"),
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
                payload["oracles"].append({
                    "target": target, "strategy": "Daily", "seeds": len(seeds),
                    "romd": 41.0, "win_rate_pct": 96.0,
                })
            payload["comparison_inventory"] = list(targets)
            # Legacy aggregate reports have no per-row source-arm receipt.
            # Old Breakout remains readable, but previous C131/C132 Daily
            # results must not be re-labelled as C133/C134 evidence.
            _, _, _, values, info = read_rows()
            self.assertEqual(len(values), len(targets) - 2)
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
            # Equivalent reference names do not change perfect-score results.
            self.assertEqual(values[daily_direct_key]["oracle_romd"], 41.0)
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
        self.assertEqual(references[true_daily]["arm_id"], "C125")
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
        # A strategy-arm reference in the Oracle registry is not proof of
        # matching fitted-model supervision. CO/CP use event Min NetPnL, not
        # TrueDaily's independently-entered stock-day NetPnL. Their existing
        # C125/C126 Rolling/Robustness must never appear as TrueDaily results.
        for suffix in ("", "_Direct"):
            name = "TrueDailyUniversal_MinTrade_NetPnL_Positive" + suffix
            self.assertFalse(any(target == name for target, _ in bindings))
            daily_key = name + "::Daily"
            expected_arm = "C134" if suffix else "C133"
            self.assertEqual(
                [(profile, arm) for (target, profile), arm in bindings.items()
                 if target == daily_key],
                [("true_daily_universal_mintrade_fullcycle_netpnl_positive_binary", expected_arm)],
            )

if __name__ == '__main__':
    main(verbosity=2)
