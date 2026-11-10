"""Independent P18 economic-source, PIT and signed Oracle regressions.

Synthetic prices only; no market fitting or optimizer search is executed.
"""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from core.portfolio_param_runtime import build_portfolio_params_signature
from filters.breakout_quality.daily_min_p18_target import (
    SOURCE_ID,
)
from services.breakout_quality.daily_min_p18_pit_source import materialize_daily_min_p18_pit_labels
from services.breakout_quality.formal_mintrade40_pit_source import FormalMinTrade40PITBlocked
from tools.validate.synthetic_daily_full_p20_fulltrade_cases import _fixture_source

RANK = 'daily_universal_min_p18_mintrade_netpnl_full_list_ndcg_pairwise'
POSITIVE = 'daily_universal_min_p18_mintrade_netpnl_positive_binary'
RECIPE = 'daily_trade_path_daily_min_p18_net_pnl'


def fixture(stop=1.0):
    days, frame, source = _fixture_source()
    params = replace(source.schedule[0].params, atr_times_init=stop, atr_times_trail=stop)
    source = replace(source, source_id=SOURCE_ID, schedule=tuple(
        replace(p, source_id=SOURCE_ID, params=params,
                params_signature=build_portfolio_params_signature(params)) for p in source.schedule))
    return days, frame, source


def labels(source, days, frame, positions=(30, 40, 50, 60, 70, 80, 90), **kwargs):
    return materialize_daily_min_p18_pit_labels(
        source, expected_seed=42, frame=frame, positions=np.array(positions), ticker='2330',
        exchange_calendar=days, fit_cutoff=kwargs.pop('fit_cutoff', days[-1]), **kwargs)


class DailyMinP18Cases(TestCase):
    def test_current_four_arms_and_independent_rank_positive_models(self):
        from core.research_summary import (get_research_summary_strategy_arm_ids,
            get_research_summary_model_profiles, get_research_summary_baseline_arm_ids)
        from core.strategy_compare_policy import get_strategy_comparison_settings, get_strategy_multi_seed_robustness_settings
        from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
        from core.breakout_quality_registry import get_continuous_ranker_research_spec
        from filters.breakout_quality.strategy_compare_dl_artifacts import resolve_arm_runtime_dl_source_ids
        from services.research.target_dl_overview import _is_exact_supervision, _is_positive_classification
        settings = get_strategy_comparison_settings('broker_cash_dl_conversion_rolling')
        active = set(get_research_summary_strategy_arm_ids())
        self.assertTrue({'C160', 'C161', 'C177', 'C178'} <= active)
        self.assertTrue({'C159','C162','C163','C164','C165','C166','C167','C174','C175','C176'}.isdisjoint(active))
        self.assertEqual(get_research_summary_baseline_arm_ids()['daily_min'], 'C161')
        self.assertTrue({('MR-13DF', RANK), ('MR-13DG', POSITIVE)} <= set(get_research_summary_model_profiles()))
        self.assertEqual(resolve_arm_runtime_dl_source_ids(settings, settings.arms['C178']),
                         ('CONT13DF_ROLL', 'CONT13DG_ROLL'))
        self.assertFalse(settings.arms['C161'].dl_enabled)
        self.assertEqual(settings.arms['C160'].dl_id, 'CONT13CJ_ROLL')
        rank, positive = [get_continuous_ranker_execution_recipe(p) for p in (RANK, POSITIVE)]
        self.assertEqual(rank.target_policy, positive.target_policy)
        self.assertNotEqual(rank.training_objective, positive.training_objective)
        self.assertEqual(positive.training_objective, 'daily_mintrade_netpnl_positive_binary_cross_entropy')
        from core.breakout_quality_registry import get_breakout_quality_experiment_profile
        target_id = get_breakout_quality_experiment_profile(RANK).continuous_target_id
        self.assertTrue(_is_exact_supervision(RANK, target_id))
        self.assertFalse(_is_exact_supervision(POSITIVE, target_id))
        self.assertTrue(_is_positive_classification(POSITIVE, target_id))
        for mode in ('broker_cash_dl_conversion_rolling', 'broker_cash_dl_conversion_oos'):
            s = get_strategy_comparison_settings(mode)
            self.assertTrue({'C160','C161','C177','C178'} <= {a.arm_id for a in s.enabled_arms})
        robust = get_strategy_multi_seed_robustness_settings(settings.profile_id)
        self.assertEqual(len(robust.resolved_seeds), 4)
        self.assertTrue({'C160','C161','C177','C178'} <= set(robust.benchmark_strategy_arm_ids))
        self.assertTrue({'C160','C177','C178'} <= set(robust.model_seed_sensitive_arm_ids))
        self.assertNotIn('C161',robust.model_seed_sensitive_arm_ids)

    def test_membership_removal_propagates_without_consumer_exceptions(self):
        from config import research_summary as config
        from core.research_summary import (get_research_summary_strategy_arm_ids,
            get_research_summary_model_profiles, get_research_summary_baseline_arm_ids)
        from core.audit_registry import resolve_current_oracle_comparisons
        from core.audit_policy import get_reusable_audit_definitions
        from core.strategy_compare_policy import get_strategy_comparison_settings
        def is_p18(row):
            return (row.get('oracle_parameter_family') == 'daily_min'
                    or row.get('parameter_family') == 'daily_min'
                    or row.get('oracle', {}).get('parameter_family') == 'daily_min')
        original = set(get_research_summary_strategy_arm_ids())
        replacement = tuple({**row, 'enabled':False} if is_p18(row) else row
                            for row in config.RESEARCH_SUMMARY_ROWS)
        with patch.object(config,'RESEARCH_SUMMARY_ROWS',replacement):
            active=set(get_research_summary_strategy_arm_ids())
            self.assertEqual(original-active, {'C160','C161','C177','C178'})
            self.assertTrue({'C168','C169','C172','C173'} <= active)
            self.assertNotIn('daily_min',get_research_summary_baseline_arm_ids())
            self.assertTrue({'MR-13DF','MR-13DG'}.isdisjoint(
                {mid for mid,p in get_research_summary_model_profiles()}))
            from services.audit.target_oracle_romd import SUPPORTED_AUDIT_TYPE
            definition=next(d for d in get_reusable_audit_definitions('breakout_quality')
                            if d.audit_type == SUPPORTED_AUDIT_TYPE)
            self.assertFalse(any(r.get('parameter_family') == 'daily_min'
                                 for r in resolve_current_oracle_comparisons(definition.source)))
            settings=get_strategy_comparison_settings('broker_cash_dl_conversion_rolling')
            self.assertEqual({a.arm_id for a in settings.enabled_arms}, active)
        self.assertEqual(set(get_research_summary_strategy_arm_ids()),original)

    def test_fit_identity_uses_only_asof_teacher_and_ordered_mature_labels(self):
        from types import SimpleNamespace
        import pandas as pd
        from core.daily_trade_path_capabilities import daily_truth_capability
        from services.breakout_quality.fitting_data_dependencies import (
            FitDataDependencies, mintrade40_fit_sample_sha256)
        cap=daily_truth_capability(RECIPE)
        target={'target_contract':{'target_id':cap['target_id'], 'target_kind':cap['target_kind'],
                'teacher_lineage':{'contract':cap['teacher_contract'], 'seed':42,
                                   'oos_start_date':'2021-01-01'}},
                'formal_mintrade40_teacher_periods':[
                    {'start':'2014-01-01','end':'2020-12-31','published':'2013-12-31','params_signature':'old'},
                    {'start':'2021-01-01','end':None,'published':'2020-12-31','params_signature':'new'}]}
        first=FitDataDependencies({},target).rolling_source(fit_cutoff='2020-01-01')
        updated={**target,'formal_mintrade40_teacher_periods':[
            target['formal_mintrade40_teacher_periods'][0],
            {**target['formal_mintrade40_teacher_periods'][1],'params_signature':'later-change'}]}
        self.assertEqual(first,FitDataDependencies({},updated).rolling_source(fit_cutoff='2020-01-01'))
        table=pd.DataFrame({'ticker':['2330','2330'],'date':['2019-01-01','2019-01-02'],
                            'source_pos':[300,301],'label_eval_end_date':['2019-02-01','2019-02-02']})
        bundle=SimpleNamespace(target_manifest=target, group_table=table, raw_target=np.array([-10.,20.]))
        a=mintrade40_fit_sample_sha256(bundle,np.array([0,1]))
        self.assertIsNotNone(a)
        bundle.raw_target[0]=10.
        self.assertNotEqual(a,mintrade40_fit_sample_sha256(bundle,np.array([0,1])))
        with self.assertRaisesRegex(ValueError,'lineage contract'):
            FitDataDependencies({}, {**target,'target_contract':{
                **target['target_contract'],'teacher_lineage':{'contract':'wrong','seed':42}}}
            ).rolling_source(fit_cutoff='2020-01-01')

    def test_netpnl_and_positive_change_with_actual_p18_exits(self):
        days, frame, tight = fixture(.5)
        _, _, loose = fixture(1.)
        a, b = labels(tight, days, frame), labels(loose, days, frame)
        self.assertTrue(a.completed.all() and b.completed.all())
        self.assertTrue(np.any(a.raw_net_pnl != b.raw_net_pnl))
        self.assertTrue(np.any((a.raw_net_pnl > 0) != (b.raw_net_pnl > 0)))

    def test_future_quotes_do_not_change_pit_labels(self):
        days, frame, source = fixture()
        cutoff = days[110]
        a = labels(source, days, frame, fit_cutoff=cutoff)
        mutated = frame.copy()
        mutated.loc[days[111]:, ['Open','High','Low','Close']] *= 20
        b = labels(source, days, mutated, fit_cutoff=cutoff)
        np.testing.assert_equal(a.raw_net_pnl, b.raw_net_pnl)
        np.testing.assert_equal(a.maturity_dates, b.maturity_dates)

    def test_forced_terminal_closeout_does_not_become_past_fit_label(self):
        days, frame, source = fixture(3.)
        past = labels(source, days, frame)
        terminal = labels(source, days, frame, research_end_is_terminal=True)
        self.assertFalse(past.completed.any())
        self.assertTrue(np.isnan(past.raw_net_pnl).all())
        self.assertTrue(terminal.completed.any())

    def test_failed_next_session_order_is_null_even_if_later_quotes_fill(self):
        days, frame, source = fixture()
        previous_close = frame.loc[days[60], 'Close']
        frame.loc[days[61], ['Open','High','Low','Close']] = previous_close + np.array([21,22,20,21])
        r = labels(source, days, frame, positions=(60,))
        self.assertFalse(r.completed[0])
        self.assertTrue(np.isnan(r.raw_net_pnl[0]))

    def test_foreign_teachers_and_breakout_filters_are_rejected(self):
        days, frame, source = fixture()
        for foreign in ('formal_min_roos_seed', 'daily_min_p19_seed42', 'daily_full_p20_seed42'):
            with self.subTest(source=foreign), self.assertRaises(FormalMinTrade40PITBlocked):
                labels(replace(source, source_id=foreign), days, frame)
        params = replace(source.schedule[0].params, use_vol=True)
        bad = replace(source, schedule=(replace(source.schedule[0], params=params),))
        with self.assertRaises(FormalMinTrade40PITBlocked):
            labels(bad, days, frame)

    def test_historical_builder_uses_three_dimensions_and_correct_work_root(self):
        from services.breakout_quality import daily_min_p18_training_source as teacher
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = teacher.historical_params_path(root, 693545351)
            path.parent.mkdir(parents=True)
            path.write_text("{}")
            with patch.object(teacher, '_runtime_contract', side_effect=lambda **kwargs: kwargs), \
                 patch.object(teacher, '_validate_min_roos_params', return_value={}) as validate:
                contract = teacher._expected_historical_preflight(root, seed=693545351, validate_path=path)
            self.assertEqual(validate.call_args.kwargs['path'].name,path.name)
            self.assertNotEqual(validate.call_args.kwargs['path'],path)
            self.assertEqual(path.read_text(),'{}')
            path.unlink()
            self.assertEqual(validate.call_args.kwargs['arm_id'],teacher.HISTORY_ID)
            reference = contract['baseline_contract']['reference_payload']
            self.assertEqual(reference['search_fields'], ['atr_len','atr_times_init','atr_times_trail'])
            self.assertNotIn('daily_execution_contract', reference)
            self.assertIsNone(contract['args'].daily_search_profile)
            with patch.object(teacher, 'inspect_history', return_value={'action':'BUILD', 'path':path}), \
                 patch.object(teacher, 'prepare_selection_historical_p2_params', return_value={'params_path':path}) as build, \
                 patch.object(teacher, '_validated_records'):
                teacher.execute_history(project_root=root, seed=693545351)
            args = build.call_args.kwargs
            self.assertEqual(args['optimizer_seed'], 693545351)
            self.assertEqual(args['parameter_set_identity'], teacher.HISTORY_ID)
            self.assertIsNone(args['daily_search_profile'])
            self.assertEqual(Path(args['output_relative_dir']) / 'p2_dl_off_trained' / 'active_params' / 'roos_base_best.json', path)
            self.assertEqual(teacher.inspect_history(project_root=root, seed=42)['action'], 'BUILD')

    def test_canonical_current_teacher_is_p18_and_matches_each_fit_seed(self):
        from services.breakout_quality import daily_min_p18_training_source as teacher
        with patch.object(teacher, 'inspect_formal_mintrade40_p16_teacher', return_value={}) as inspect:
            for seed in (42, 693545351, 2014432738):
                teacher.inspect_current(project_root='.', seed=seed)
                self.assertEqual(inspect.call_args.kwargs['seed'], seed)
                self.assertEqual(inspect.call_args.kwargs['baseline_arm'], 'C161')
        from services.breakout_quality.trade_path_target_source import registered_trade_path_source
        source = registered_trade_path_source(RANK)
        self.assertEqual([d.name for d in source.dependencies],
                         ['daily_min_p18_historical_teacher', 'daily_min_p18_current_teacher'])
        self.assertEqual(source, registered_trade_path_source(POSITIVE))

    def test_oracle_uses_the_same_p18_schedule_and_a_separate_signed_witness(self):
        from services.audit import daily_min_p18_oracle_truth as truth
        days, frame, source = fixture()
        values = {('2330',str(days[40].date())): .7}
        parent = {'profile_id':'eligible_provider', 'entry_policy':truth.ENTRY_POLICY,
                  'teacher': {'seed':42,'param_identity_sha256':'a'*64},
                  'rank_sha256':'b'*64, 'raw_label_sha256':'c'*64,
                  'calendar_sha256':'d'*64, 'settled_net_pnl_rows':1,
                  'dataset_summary_sha256':'e'*64}
        with patch.object(truth.daily_min, 'build_true_daily_universal_oracle_scores', return_value=(values,parent)) as build:
            scores, evidence = truth.build_daily_min_p18_oracle_scores(
                output_subdir='audit', canonical_schedule=source.schedule, seed=42)
        self.assertEqual(scores, values)
        self.assertEqual(build.call_args.kwargs['canonical_schedule'], source.schedule)
        self.assertEqual(build.call_args.kwargs['output_subdir'], 'audit/daily_min_p18')
        self.assertEqual(evidence['target_id'], truth.TARGET_ID)
        with patch.object(truth.daily_min, 'load_positive_truth', return_value={next(iter(values)):123.}) as load:
            raw = truth.load_positive_truth(output_subdir='audit', canonical_evidence=evidence)
        self.assertEqual(next(iter(raw.values())),123.)
        self.assertEqual(load.call_args.kwargs['canonical_evidence'], parent)
        with self.assertRaisesRegex(ValueError, 'provenance|witness'):
            truth.load_positive_truth(output_subdir='audit', canonical_evidence={**evidence,'teacher':{'seed':43}})

    def test_oracle_identity_exact_supervision_and_old_source_signatures(self):
        from core.daily_trade_path_capabilities import daily_truth_capability
        from services.audit import target_oracle_romd as oracle
        from services.research.oracle_artifact_contract import source_calculation_contract
        from core.audit_registry import oracle_positive_output_for_recipe, oracle_scientific_target_name
        cap = daily_truth_capability(RECIPE)
        self.assertEqual(oracle._daily_truth_identity(RECIPE,None)[0],cap['target_id'])
        self.assertEqual(oracle_positive_output_for_recipe(RECIPE),'netpnl_positive_probability_v1')
        self.assertEqual(oracle_scientific_target_name('TrueDailyUniversal_MinTrade_P18_NetPnL_Positive_Direct'),
                         'Daily_MinTrade_NetPnL_Positive')
        with self.assertRaises(oracle.OracleBlockedError):
            oracle._daily_truth_identity(RECIPE,40)
        contract=source_calculation_contract(recipe=RECIPE, profile_id=RANK)
        self.assertEqual(contract['calculation_revision'],1)
        self.assertIsNone(contract['horizon_sessions'])
        legacy=source_calculation_contract(recipe='daily_trade_path_true_net_pnl',profile_id=RANK)
        self.assertNotEqual(contract['implementation_sha256'],legacy['implementation_sha256'])
