"""Positive admission topology and selector regressions for current dual-DL arms.

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

from core.positive_netpnl_contract import (positive_labels, POSITIVE_SCORE_CONTRACT, POSITIVE_P_CLASS_COLUMN)
from core.positive_score_admission import (
    positive_score_admitted, admit_positive_score_rows, uses_primary_positive_classifier,
)
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
                                         oracle_target_semantics, oracle_parameter_family)
        d = get_formal_oracle_definition(get_active_audit_module_id())
        rows = resolve_current_oracle_comparisons(d.source)
        self.assertEqual(len(rows), len({(*oracle_target_semantics(x), oracle_parameter_family(x))
                                        for x in rows}))
        by_legacy = {oracle_comparison_key(x): x for x in rows}
        cases = (
            ("MFE40_Control", "MFE40", "Exact", "Breakout"),
            ("MFE40_Direct", "MFE40", "Direct", "Breakout"),
            ("T40", "T40", "Exact", "Breakout"),
            ("T40::Daily", "T40", "Exact", "Daily"),
            ("T40_Direct", "T40", "Direct", "Breakout"),
            ("T40_Direct::Daily", "T40", "Direct", "Daily"),
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

    def test_current_positive_arms_have_independent_ranker_and_classifier(self):
        from core.breakout_quality_registry import get_breakout_quality_experiment_profile
        from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
        settings = get_strategy_comparison_settings(STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE)
        actual = {arm.arm_id: (arm.dl_id, arm.dl_runtime_options.get('positive_dl_id'))
                  for arm in settings.enabled_arms
                  if (arm.dl_runtime_options or {}).get('positive_score_column')
                  and not uses_primary_positive_classifier(arm.dl_runtime_options)}
        self.assertTrue(actual, "Current suite must keep at least one real independent Positive pair")
        # AI: Every registered Positive arm is checked from the single Compare Suite,
        #     without maintaining a second experiment/arm matrix in the validator.
        for arm_id, (rank_id, classification_id) in actual.items():
            self.assertNotEqual(rank_id, classification_id)
            self.assertIn(classification_id, settings.dl_sources)
            self.assertIn(rank_id, settings.dl_sources)
        for arm_id, (rank_id, classification_id) in actual.items():
            with self.subTest(arm=arm_id):
                rank_profile = get_breakout_quality_experiment_profile(
                    settings.dl_sources[rank_id].experiment_profile)
                positive_profile = get_breakout_quality_experiment_profile(
                    settings.dl_sources[classification_id].experiment_profile)
                rank_recipe = get_continuous_ranker_execution_recipe(rank_profile.name)
                classification_recipe = get_continuous_ranker_execution_recipe(positive_profile.name)
                self.assertEqual(rank_profile.continuous_target_id, positive_profile.continuous_target_id)
                self.assertEqual(rank_recipe.training_policy.target_builder, 'scalar_pairwise')
                self.assertTrue(rank_recipe.training_policy.uses_pairwise_loss)
                self.assertEqual(classification_recipe.training_policy.target_builder, 'netpnl_positive_binary')
                self.assertEqual(classification_recipe.training_policy.categorical_class_count, 2)
                self.assertEqual(classification_recipe.training_policy.categorical_primary_class, 1)
                self.assertEqual(classification_recipe.training_policy.categorical_score_columns[1],
                                 POSITIVE_P_CLASS_COLUMN)
                self.assertIsNone(positive_profile.score_calibration)

    def test_retired_t40_positive_is_not_in_any_current_research_inventory(self):
        from config.breakout_quality import (
            BREAKOUT_QUALITY_MODEL_RESEARCH_MODEL_PROFILE,
            BREAKOUT_QUALITY_MODEL_TEST_PROFILES,
        )
        from core.breakout_quality_registry import (
            DAILY_UNIVERSAL_TERMINAL_MAE_GATED_T40_POSITIVE_PROFILE,
            get_continuous_ranker_research_spec,
        )
        from core.strategy_compare_registry import STRATEGY_DL_SOURCES
        from core.audit_registry import resolve_current_oracle_comparisons
        self.assertNotEqual(BREAKOUT_QUALITY_MODEL_RESEARCH_MODEL_PROFILE[0], 'MR-13CW')
        self.assertIn(BREAKOUT_QUALITY_MODEL_RESEARCH_MODEL_PROFILE, BREAKOUT_QUALITY_MODEL_TEST_PROFILES)
        self.assertNotIn('MR-13CW', {model for model, _ in BREAKOUT_QUALITY_MODEL_TEST_PROFILES})
        settings = get_strategy_comparison_settings(STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE)
        active = {arm.arm_id for arm in settings.enabled_arms}
        self.assertTrue({'C96', 'C107', 'C109', 'C125', 'C126', 'C133', 'C134', 'C135', 'C136'}.issubset(active))
        self.assertTrue({'C142', 'C143', 'C144'}.isdisjoint(active))
        self.assertIn('C144', settings.arms)  # Immutable historical strategy ID.
        self.assertIn('CONT13CW_ROLL', STRATEGY_DL_SOURCES)  # Historical PIT source ID.
        self.assertEqual(get_continuous_ranker_research_spec(
            DAILY_UNIVERSAL_TERMINAL_MAE_GATED_T40_POSITIVE_PROFILE).model_research_id, 'MR-13CW')
        definition = get_formal_oracle_definition(get_active_audit_module_id())
        historical = definition.source['historical_only_oracle_targets']
        self.assertIn('T40_Positive', historical)
        self.assertIn('T40_Positive_Direct', historical)
        current = resolve_current_oracle_comparisons(definition.source)
        self.assertTrue(all(row['target'] != 'T40_Positive' for row in current))
        self.assertTrue({'T40', 'T40_Direct'}.issubset({row['target'] for row in current}))
        # Historical name cannot be silently reactivated through an added list row.
        retired_row = {'target': 'T40_Positive', 'selector': 'default',
                       'score_source': 'T40', 'positive_output': 't40_positive_probability_v1',
                       'strategy_arm_id': 'C144'}
        with self.assertRaisesRegex(ValueError, 'Historical-only Oracle target'):
            resolve_current_oracle_comparisons({**definition.source,
                'oracle_comparisons': (*definition.source['oracle_comparisons'], retired_row)})
        overview = collect_target_dl_overview(ROOT)
        self.assertTrue(all(row['target'] != 'T40_Positive' and row['strategy'] != 'C144'
                            for row in overview['rows']))

    def test_target_overview_joins_ranker_and_classifier_by_runtime_roles(self):
        # Truth identity is intentionally shared by NetPnL rank and P(NetPnL>0)
        # binary models.  The report must identify the role from the actual
        # Strategy Registry consumers, not from that shared continuous label.
        from core.breakout_quality_registry import get_continuous_ranker_research_spec, get_breakout_quality_experiment_profile
        from services.research.target_dl_overview import (
            _is_exact_supervision, _is_positive_classification,
        )
        from filters.breakout_quality.strategy_compare_dl_artifacts import resolve_arm_runtime_dl_source_ids
        settings = get_strategy_comparison_settings(STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE)
        cases = (
            ("MinTrade_NetPnL", "Breakout", "Exact", "C123", "MR-13CO", ""),
            ("MinTrade_NetPnL", "Breakout", "Direct", "C124", "MR-13CO", ""),
            ("MinTrade_NetPnL_Positive", "Breakout", "Exact", "C125", "MR-13CO", "MR-13CP"),
            ("MinTrade_NetPnL_Positive", "Breakout", "Direct", "C126", "MR-13CO", "MR-13CP"),
            ("DailyUniversal_MinTrade_NetPnL", "Daily", "Exact", "C137", "MR-13CU", ""),
            ("DailyUniversal_MinTrade_NetPnL", "Daily", "Direct", "C138", "MR-13CU", ""),
            ("DailyUniversal_MinTrade_NetPnL_Positive", "Daily", "Exact", "C133", "MR-13CU", "MR-13CT"),
            ("DailyUniversal_MinTrade_NetPnL_Positive", "Daily", "Direct", "C134", "MR-13CU", "MR-13CT"),
            ("DailyUniversal_MinTrade_NetPnL_Positive", "Breakout", "Exact", "C135", "MR-13CU", "MR-13CT"),
            ("DailyUniversal_MinTrade_NetPnL_Positive", "Breakout", "Direct", "C136", "MR-13CU", "MR-13CT"),
        )
        published = {arm_id: float(i + 1) for i, (*_, arm_id, rank, classifier) in enumerate(cases)}
        with patch('services.research.target_dl_overview._strategy_rolling_metrics', return_value=published):
            report = collect_target_dl_overview(ROOT)
        keys = lambda row: (row['target'], row['entry_strategy'], row['selector'])
        group = {keys(row): [] for row in report['rows']}
        for row in report['rows']:
            group[keys(row)].append(row)
        for target, strategy, selector, arm_id, rank, classifier in cases:
            with self.subTest(arm=arm_id):
                matched = group[(target, strategy, selector)]
                self.assertEqual(len(matched), 1, "one Target/Strategy/Selector is one strategy comparison")
                row = matched[0]
                self.assertEqual((row['strategy'], row['model'], row['positive_model']),
                                 (arm_id, rank, classifier))
                self.assertEqual(row['rolling_romd'], published[arm_id], "never borrow another arm's RoMD")
                arm = settings.arms[arm_id]
                actual_sources = resolve_arm_runtime_dl_source_ids(settings, arm)
                actual_models = tuple(get_continuous_ranker_research_spec(
                    settings.dl_sources[source].experiment_profile).model_research_id
                    for source in actual_sources)
                self.assertEqual(actual_models, (classifier,) if not rank else
                                 (rank, classifier) if classifier else (rank,))
        self.assertTrue(all(row['model'] not in {'MR-13CP', 'MR-13CT'}
                            for row in report['rows']))
        self.assertEqual(table_contract('research.target_dl_validation_overview',
                                       'comparison', 'target_dl').headers[5:7],
                         ('Rank DL', 'Positive DL'))
        rendered = render_target_dl_overview(report)
        self.assertIn('Positive DL', rendered)
        self.assertIn('MR-13CP', rendered)
        self.assertIn('MR-13CT', rendered)
        # A classifier may use the same raw NetPnL truth but cannot provide
        # the rolling Spearman rho as evidence for its ranker.
        for rank, classifier in (("CONT13CO_ROLL", "CONT13CP_ROLL"),
                                  ("CONT13CU_ROLL", "CONT13CT_ROLL")):
            rank_profile = settings.dl_sources[rank].experiment_profile
            binary_profile = settings.dl_sources[classifier].experiment_profile
            target_id = get_breakout_quality_experiment_profile(rank_profile).continuous_target_id
            self.assertTrue(_is_exact_supervision(rank_profile, target_id))
            self.assertFalse(_is_exact_supervision(binary_profile, target_id))
            self.assertFalse(_is_positive_classification(rank_profile, target_id))
            self.assertTrue(_is_positive_classification(binary_profile, target_id))

    def test_target_overview_rejects_positive_arm_without_binary_classifier(self):
        from core.strategy_compare_registry import STRATEGY_DL_SOURCES
        definition = get_formal_oracle_definition(get_active_audit_module_id())
        targets, comparisons, labels, _, _ = _oracle_rows(
            ROOT, definition, load_report_contract_versions(ROOT))
        control = _strategy_bindings(target_names=targets, score_targets=labels, definition=definition)
        target = 'DailyUniversal_MinTrade_NetPnL_Positive'
        rank_profile = STRATEGY_DL_SOURCES['CONT13CO_ROLL']['experiment_profile']
        self.assertEqual(control[(target, rank_profile)], 'C125')
        # Swapping the classifier for a NetPnL ranker must not silently pass,
        # even when their continuous_target_id matches perfectly.
        with patch.dict(STRATEGY_DL_SOURCES, {'CONT13CP_ROLL': STRATEGY_DL_SOURCES['CONT13CO_ROLL']}):
            missing = _strategy_bindings(target_names=targets,score_targets=labels, definition=definition)
        self.assertNotIn((target, rank_profile), missing)
        self.assertEqual(missing[('DailyUniversal_MinTrade_NetPnL', rank_profile)], 'C123')

    def test_no_filter_on_original_netpnl_and_positive_filter_fail_closed(self):
        rank = {'available': True,'score':0.70,'score_date':'2024-10-01',
                'positive_available':True,'positive_score_date':'2024-10-01',
                'positive_probability':0.51}
        options = {'positive_dl_id':'CONT13CP_ROLL', 'positive_score_column':POSITIVE_P_CLASS_COLUMN,'positive_probability_threshold':0.5}
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
             'breakout_quality_ranking_options':{'positive_dl_id':'CONT13CP_ROLL', 'positive_score_column':POSITIVE_P_CLASS_COLUMN, 'positive_probability_threshold':0.5}}
        order,diag=reorder_candidates_for_resource_aware_quality(
            [row],available_cash=0,sizing_equity=100,
            pre_market_occupied=10,max_positions=10,params=None)
        self.assertEqual(order,[])

    def test_pit_rank_and_positive_use_separate_profiles_same_information_date(self):
        settings=get_strategy_comparison_settings(STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE)
        # AI: The runtime must use both independently fitted profiles for every
        #     Positive arm, not only the historically first NetPnL pair.
        for arm in settings.enabled_arms:
            if not (arm.dl_runtime_options or {}).get('positive_score_column'):
                continue
            with self.subTest(arm=arm.arm_id):
                source=settings.dl_sources[arm.dl_id]
                options=dict(arm.dl_runtime_options)
                if uses_primary_positive_classifier(options):
                    columns = []
                    def primary_lookup(**kwargs):
                        columns.append(kwargs.get('score_column'))
                        return {'score': .61, 'available': True,
                                'score_date': '2024-10-01',
                                'score_source': SCORE_SOURCE_SELECTION_POINT_IN_TIME}
                    with breakout_quality_ranking_source_context(
                        score_source=SCORE_SOURCE_SELECTION_POINT_IN_TIME,
                        model_architecture=source.model_architecture,
                        experiment_profile=source.experiment_profile,
                        ranking_policy=arm.dl_runtime_mode,
                        ranking_options=options,
                    ), patch('filters.breakout_quality.runtime.lookup_selection_point_in_time_candidate_score',
                             side_effect=primary_lookup):
                        predicted = resolve_breakout_quality_candidate_rank(
                            ticker='2330', signal_date='2024-10-01',
                            information_date='2024-10-01', high_len=40)
                    self.assertEqual(columns, [POSITIVE_P_CLASS_COLUMN])
                    self.assertEqual(predicted['score'], .61)
                    self.assertNotIn('positive_probability', predicted)
                    self.assertTrue(positive_score_admitted({
                        'breakout_quality_rank': predicted,
                        'breakout_quality_ranking_options': arm.dl_runtime_options,
                    }))
                    continue
                classifier = settings.dl_sources[options['positive_dl_id']]
                options.update(positive_filter_id=classifier.filter_id,
                               positive_model_architecture=classifier.model_architecture,
                               positive_experiment_profile=classifier.experiment_profile,
                               positive_score_source=classifier.score_source)
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
                self.assertNotEqual(lookup_calls[0]['experiment_profile'],
                                    lookup_calls[1]['experiment_profile'])
                self.assertEqual(lookup_calls[0]['experiment_profile'], source.experiment_profile)
                self.assertEqual(lookup_calls[1]['experiment_profile'], classifier.experiment_profile)
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
            self.assertEqual(arm.dl_id, 'CONT13CO_ROLL')
            self.assertEqual(contracts[target]['positive_score_source_id'], 'CONT13CP_ROLL')
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
        from core.audit_registry import oracle_parameter_family
        published = [row for row in comparisons if row["target"] != positive_exact
                     and row.get("strategy", "Breakout") == "Breakout"
                     and oracle_parameter_family(row) == "min"]
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
            # A Target may already have a certified Breakout receipt (e.g. T40)
            # even when the newly authorized Daily receipt is not certified.
            self.assertTrue((true_daily - legacy_keys).isdisjoint(values))
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
            historical_breakout_labels = {oracle_target_semantics(row)[0] for row in published
                                          if oracle_comparison_key(row) in legacy_keys}
            self.assertTrue(all(row["oracle_romd"] is None for row in overview["rows"]
                                if row["target"] in scientific_daily
                                and row["entry_strategy"] == "Breakout"
                                and row["target"] not in historical_breakout_labels))
            daily_overview_rows = [row for row in overview["rows"]
                                   if row["target"] in scientific_daily
                                   and (row["entry_strategy"] == "Daily"
                                        or row["target"] not in historical_breakout_labels)]
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
            # A historical signed report cannot claim newly registered Full/P17
            # Oracle items. It retains only its own completed Min inventory.
            payload["comparison_inventory"] = [key for key in targets
                                                if key not in {"C97", "T40::Full", "T40_Direct::Full"}]
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
            self.assertEqual(set(values), set(payload["comparison_inventory"]))
            self.assertEqual(len(values), len(targets) - 3)  # No Full/P17 receipts in the old report.
            self.assertTrue({"C97", "T40::Full", "T40_Direct::Full"}.isdisjoint(values))
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
        from core.audit_registry import oracle_baseline_arm_ids
        baselines = oracle_baseline_arm_ids(definition.source)
        reports = []
        from services.audit import target_oracle_romd as oracle
        for scope, scoped in oracle._oracle_strategy_scopes(definition):
            strategy = scope.split("::", 1)[0]
            subset = scoped.source["oracle_comparisons"]
            baseline_arm = scoped.source["baseline_arm_id"]
            report = {
                "baseline": {"target": baseline_arm, "seeds": 8,
                             "romd": 9.17 if baseline_arm == baselines["min"] else 15.5},
                "seeds": list(range(8)), "benchmark_id": "end_to_end_v2",
                "dataset": "full", "start_date": "2021-01-01", "end_date": "2026-03-02",
                "common_cohort": {"cohort_rows": 630589}, "param_artifacts": {},
                "strategy_profile_id": definition.source["strategy_profile_id"],
                "definitions": {row["target"]: row["description"] for row in subset},
                "oracles": [{"target": row["target"], "strategy": strategy} for row in subset],
            }
            reports.append((scope, report))
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
            ('DailyUniversal_MinTrade_NetPnL_Positive','daily_universal_formal_mintrade_fullcycle_netpnl_per_seed_full_list_ndcg_pairwise'):'C125',
            ('DailyUniversal_MinTrade_NetPnL_Positive_Direct','daily_universal_formal_mintrade_fullcycle_netpnl_per_seed_full_list_ndcg_pairwise'):'C126',
        }
        for key,value in expected.items():
            self.assertEqual(bindings.get(key),value)
        # All four TrueDaily arms use the CU ranker and independent CT admission.
        for suffix in ("", "_Direct"):
            name = "TrueDailyUniversal_MinTrade_NetPnL_Positive" + suffix
            self.assertEqual(
                [(profile, arm) for (target, profile), arm in bindings.items() if target == name],
                [("true_daily_universal_mintrade_fullcycle_netpnl_full_list_ndcg_pairwise",
                  "C136" if suffix else "C135")])
            daily_key = name + "::Daily"
            expected_arm = "C134" if suffix else "C133"
            self.assertEqual(
                [(profile, arm) for (target, profile), arm in bindings.items()
                 if target == daily_key],
                [("true_daily_universal_mintrade_fullcycle_netpnl_full_list_ndcg_pairwise", expected_arm)],
            )


class IndependentPositiveClassifierCases(TestCase):
    """The Positive gate consumes a distinct PIT-safe binary model, never raw outcomes."""

    def test_no_legacy_single_ranker_contract_or_unknown_secondary_options(self):
        from core.positive_netpnl_contract import requires_positive_score
        with self.assertRaises(ValueError):
            positive_score_admitted({'breakout_quality_ranking_options': 'invalid'})
        for options in ({'positive_dl_id': 'x'}, {'positive_dl_id': ''},
                        {'positive_prediction_contract': 'single_ranker_calibrated_positive_v2'},
                        {'positive_score_column': POSITIVE_P_CLASS_COLUMN}, {'positive_typo': True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                requires_positive_score(options)
        good = {'positive_dl_id': 'CONT13CP_ROLL',
                'positive_score_column': POSITIVE_P_CLASS_COLUMN,
                'positive_probability_threshold': 0.5}
        self.assertTrue(requires_positive_score(good))

    def test_true_two_class_pit_models_and_checkpoint_cpu_roundtrip(self):
        import pandas as pd
        from types import SimpleNamespace
        from contextlib import redirect_stdout
        from io import StringIO
        from core.breakout_quality_registry import (get_breakout_quality_experiment_profile,
                                                     DAILY_UNIVERSAL_NETPNL_POSITIVE_BINARY_PROFILE,
                                                     TRUE_DAILY_MINTRADE_NETPNL_POSITIVE_PROFILE)
        from filters.breakout_quality.models.spec import get_model_spec
        from services.breakout_quality import train_continuous_ranker as trainer
        from services.breakout_quality import continuous_ranker_pipeline as pipeline
        from services.breakout_quality.fitted_model_artifacts import load_model_from_fitted_checkpoint
        from core.breakout_quality_runtime import get_continuous_ranker_training_policy
        import torch
        from filters.breakout_quality.torch_runtime import resolve_torch_execution_plan
        old_threads = torch.get_num_threads(); torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads,old_threads)
        plan = resolve_torch_execution_plan(torch, requested_device='cpu', mixed_precision=False,
            mixed_precision_dtype='auto', deterministic_algorithms=True, allow_tf32=False)
        features = np.random.default_rng(42).normal(size=(24, 300, 10)).astype(np.float32)
        context = np.empty((24, 0), dtype=np.float32)
        raw = np.tile([-40., -10., 0., 15., 60., 200.], 4).astype(np.float32)
        days = np.repeat(pd.to_datetime(['2020-01-01','2020-01-03','2020-01-06','2020-02-01']).to_numpy(), 6)
        groups = pd.DataFrame({'date': days, 'label_eval_end_date': days + np.timedelta64(1,'D'),
                              'ticker': np.tile([str(1000+i) for i in range(6)], 4),
                              'label': (raw > 0).astype(int)})
        percentiles = trainer.build_same_date_percentile_targets(raw, np.ones(24,dtype=bool), groups['date'])
        for name in (DAILY_UNIVERSAL_NETPNL_POSITIVE_BINARY_PROFILE,
                     TRUE_DAILY_MINTRADE_NETPNL_POSITIVE_PROFILE):
            with self.subTest(profile=name), redirect_stdout(StringIO()):
                recipe = get_breakout_quality_experiment_profile(name)
                self.assertEqual(get_continuous_ranker_training_policy(recipe.training_objective).categorical_class_count, 2)
                args = trainer.parse_args(['--experiment-profile',name,'--model-architecture','inception_time_v1',
                    '--epochs','1','--batch-size','64','--evaluation-batch-size','64','--device','cpu',
                    '--no-mixed-precision','--train-prefetch-batches','0'])
                args.train_prefetch_workers=0
                selected=trainer.select_epoch(torch,features,context,groups,raw,percentiles,
                    np.arange(12),np.arange(12,18),args=args,plan=plan,evaluate_train_metrics=False,
                    information_cutoff='2020-01-08')
                model,_=trainer.fit_final(torch,features,context,raw,percentiles,groups,
                    np.arange(18),epochs=selected['best_epoch'],args=args,plan=plan,epoch_selection=selected)
                bundle=SimpleNamespace(profile=recipe,model_spec=get_model_spec('inception_time_v1'),
                    feature_bank=features,group_context=context,group_table=groups,
                    target_manifest={},raw_target=raw)
                scores, columns=pipeline.predict_score_output_payload(torch,model,bundle,
                    np.arange(18,24),batch_size=64,plan=plan)
                self.assertIn(POSITIVE_P_CLASS_COLUMN, columns)
                self.assertTrue(np.all((columns[POSITIVE_P_CLASS_COLUMN] >= 0)&
                                       (columns[POSITIVE_P_CLASS_COLUMN] <= 1)))
                np.testing.assert_allclose(scores,columns[POSITIVE_P_CLASS_COLUMN])
                np.testing.assert_allclose(columns['positive_p_nonpositive'] + columns[POSITIVE_P_CLASS_COLUMN],1.0)
                payload=pipeline.build_checkpoint_payload(model,bundle,args=args,plan=plan,
                    selected_epoch=selected['best_epoch'],fold_contract={})
                with TemporaryDirectory() as directory:
                    path=Path(directory)/'model.pt';torch.save(payload,path)
                    contract=SimpleNamespace(model_path=path,selected_epoch=selected['best_epoch'],
                        manifest={},source='synthetic')
                    restored=load_model_from_fitted_checkpoint(torch,contract=contract,bundle=bundle,plan=plan)
                    actual,restored_columns=pipeline.predict_score_output_payload(torch,restored,bundle,
                        np.arange(18,24),batch_size=64,plan=plan)
                    np.testing.assert_array_equal(scores,actual)
                    np.testing.assert_array_equal(columns[POSITIVE_P_CLASS_COLUMN],
                                                  restored_columns[POSITIVE_P_CLASS_COLUMN])

class ScoreOutputCompletenessCases(TestCase):
    """Registry-driven binary-classifier output checks for PIT and OOS."""

    @staticmethod
    def profiles():
        from core.breakout_quality_registry import (
            SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES,
            get_breakout_quality_experiment_profile,
        )
        return [profile for name in SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES
                if (profile := get_breakout_quality_experiment_profile(name))
                and (policy := get_continuous_ranker_training_policy(profile.training_objective))
                and policy.categorical_class_count == 2
                and POSITIVE_P_CLASS_COLUMN in policy.categorical_score_columns]

    def test_classifier_output_is_required_by_manifest_and_fold(self):
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
                frame['positive_p_nonpositive'] = .4
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
                extras = {POSITIVE_P_CLASS_COLUMN: [.1, .9],
                          'positive_p_nonpositive': [.9,.1]}
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
                pit['positive_p_nonpositive'] = extras['positive_p_nonpositive']
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
        self.assertEqual(get_continuous_ranker_training_policy(
            "daily_mintrade_netpnl_positive_binary_cross_entropy").categorical_class_count, 2)
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
