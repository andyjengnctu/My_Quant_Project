"""Independent MR-13CL Daily-universal Exact/Direct Oracle integration evidence.

This intentionally does not call the forbidden Windows formal test runner.
All synthetic market inputs and source artifacts are mocked; no real market
RoMD, actual four-seed FIT, or PIT model result is asserted here.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch
import tempfile

import numpy as np
import pandas as pd

from core.audit_policy import get_reusable_audit_definitions
from core.report_style import styled_workflow_status
from core.research_report_contract import require_approved_report_contract
from filters.breakout_quality.runtime import (
    resolve_breakout_quality_candidate_rank, audit_oracle_truth_score_context,
)
from core.breakout_quality_ranking_context import get_breakout_quality_ranking_source_context
from services.audit import target_oracle_romd as oracle
from services.audit.daily_trade40_oracle import (
    DailyTrade40OracleBlockedError, build_daily_universal_trade40_oracle_scores,
)
from filters.breakout_quality.daily_trade40_target import LABEL_ID, TARGET_ID

PROFILE = 'daily_universal_counterfactual_mintrade40_netr_full_list_ndcg_pairwise'
EXACT = 'DailyUniversal_MinTrade40_NetR'
DIRECT = 'DailyUniversal_MinTrade40_NetR_Direct'


def definition():
    return next(item for item in get_reusable_audit_definitions('breakout_quality')
                if item.audit_type == oracle.SUPPORTED_AUDIT_TYPE)


def fixture():
    days = pd.bdate_range('2023-01-02', periods=110)
    keys = [
        ('2330', days[10], 1.5, True),
        ('2317', days[10], np.nan, False),
        ('2454', days[10], -0.8, True),
        ('2330', days[11], -0.7, True),
        ('2317', days[11], np.nan, False),
        ('2454', days[11], 0.3, True),
        ('2330', days[79], np.nan, False),  # 40th session is beyond cutoff
    ]
    group = pd.DataFrame({
        'ticker': [x[0] for x in keys],
        'date': [x[1] for x in keys],
        'label_eval_end_date': [days[days.get_loc(x[1])+40] if x[3] else pd.NaT
                                for x in keys],
    })
    raw = np.asarray([x[2] for x in keys], dtype=np.float32)
    valid = np.asarray([x[3] for x in keys], dtype=bool)
    digest = sha256('\n'.join(days.strftime('%Y-%m-%d')).encode('ascii')).hexdigest()
    teacher = {
        'policy': 'base-finalist-best',
        'sources': [{'source_id': 'selection_min_roos', 'sha256': 'a'*64},
                    {'source_id': 'min_best_broker_cash_rolling_experiment', 'sha256': 'b'*64}],
        'periods': [{'start': '2020-01-01', 'end': '2020-12-31', 'source_id': 'selection_min_roos'},
                    {'start': '2021-01-01', 'end': None, 'source_id': 'min_best_broker_cash_rolling_experiment'}],
    }
    contract = {
        'target_id': TARGET_ID, 'label_id': LABEL_ID,
        'horizon_sessions': 40, 'sample_scope': 'daily_eligible_stock_days',
        'score_eligible_sample_count': len(keys),
        'historical_teacher_handoff': 'P2 before 2021-01-01; canonical P16 Min from 2021-01-01',
        'exchange_calendar_sha256': digest, 'portfolio_state_inputs': False,
        'teacher': teacher,
    }
    bundle = SimpleNamespace(
        group_table=group, raw_target=raw, target_valid=valid,
        target_manifest={'target_id':TARGET_ID, 'target_contract':contract,
                         'score_eligible_sample_count':len(keys)},
        summary={'dataset':'full', 'filter_id':'breakout_quality_v1',
                 'sample_scope':'daily_eligible_stock_days',
                 'score_eligible_sample_count': len(keys)},
    )
    return days, bundle


def load_synthetic(bundle, days, *, cutoff=None):
    if cutoff is None:
        cutoff = str(days[60].date())
    with patch('services.audit.daily_trade40_oracle.load_profile_continuous_ranker_data', return_value=bundle), \
         patch('services.audit.daily_trade40_oracle.discover_dataset_csv_inputs', return_value=([('0050', Path('dummy.csv'))], [])), \
         patch('services.audit.daily_trade40_oracle.collect_canonical_exchange_sessions', return_value=days):
        return build_daily_universal_trade40_oracle_scores(
            project_root=Path('/tmp/isolated'), dataset='full', filter_id='breakout_quality_v1',
            profile_id=PROFILE, cutoff=cutoff,
        )


class DailyTrade40OracleSourceCases(TestCase):
    def test_current_inventory_adds_two_rows_but_no_new_report_contract(self):
        current = definition()
        plan = oracle._oracle_comparison_plan(current)
        self.assertEqual(len(plan['targets']), 13)
        self.assertEqual(plan['daily_recipes'], {EXACT: ('daily_trade_path_net_r', 'daily_trade40_profile')})
        self.assertEqual(plan['daily_sources'], {EXACT: EXACT, DIRECT: EXACT})
        self.assertEqual(plan['selectors'][EXACT], 'exact')
        self.assertEqual(plan['selectors'][DIRECT], 'direct')
        self.assertEqual(plan['seed_sources'], {
            'MinTrade40_NetR':'MinTrade40_NetR',
            'MinTrade40_NetR_Direct':'MinTrade40_NetR',
        })
        self.assertEqual(len(plan['recipes']), 5)
        self.assertEqual(require_approved_report_contract('audit.target_oracle_romd').version, 9)
        self.assertEqual(oracle.ORACLE_REPORT_SCHEMA_VERSION, 8)
        self.assertEqual(current.source['daily_trade40_profile'], PROFILE)

    def test_mature_full_feature_universe_rank_sorts_null_last_without_fake_r(self):
        days, bundle = fixture()
        scores, evidence = load_synthetic(bundle, days)
        self.assertEqual(len(scores), 6)
        day = str(days[10].date())
        self.assertEqual(scores[('2330', day)], 1.0)
        self.assertEqual(scores[('2454', day)], 0.5)
        self.assertEqual(scores[('2317', day)], 0.0)
        self.assertEqual(evidence['mature_eligible_rows'], 6)
        self.assertEqual(evidence['settled_net_r_rows'], 4)
        self.assertEqual(evidence['censored_rank_last_rows'], 2)
        self.assertEqual(evidence['unmatured_excluded_rows'], 1)
        self.assertEqual(evidence['key_mode'], 'information_date')
        self.assertTrue(np.isnan(bundle.raw_target[~bundle.target_valid]).all())
        self.assertNotIn(('2330', str(days[79].date())), scores)
        self.assertEqual(len(evidence['rank_sha256']), 64)
        self.assertEqual(evidence['label_id'], LABEL_ID)
        self.assertEqual(evidence['target_id'], TARGET_ID)

    def test_invalid_manifest_finite_censored_or_wrong_maturity_blocks(self):
        days, bundle = fixture()
        for flaw in ('teacher', 'teacher_source', 'teacher_hash', 'teacher_handoff',
                     'calendar', 'censored', 'deadline', 'dataset', 'sample_count',
                     'duplicate'):
            with self.subTest(flaw=flaw):
                changed = SimpleNamespace(**vars(bundle))
                changed.target_manifest = {
                    **bundle.target_manifest,
                    'target_contract': dict(bundle.target_manifest['target_contract'])}
                changed.group_table = bundle.group_table.copy()
                changed.raw_target = bundle.raw_target.copy()
                changed.summary = dict(bundle.summary)
                if flaw == 'teacher':
                    changed.target_manifest['target_contract']['teacher'] = {}
                elif flaw == 'teacher_source':
                    changed.target_manifest['target_contract']['teacher'] = {
                        **bundle.target_manifest['target_contract']['teacher'],
                        'sources':[{'source_id':'selection_min_roos','sha256':'a'*64}]}
                elif flaw == 'teacher_hash':
                    changed.target_manifest['target_contract']['teacher'] = {
                        **bundle.target_manifest['target_contract']['teacher'],
                        'sources':[{'source_id':'selection_min_roos','sha256':'invalid'},
                                   {'source_id':'min_best_broker_cash_rolling_experiment','sha256':'b'*64}]}
                elif flaw == 'teacher_handoff':
                    changed.target_manifest['target_contract']['teacher'] = {
                        **bundle.target_manifest['target_contract']['teacher'],
                        'periods':[{'start':'2021-01-01','end':None,
                                    'source_id':'selection_min_roos'},
                                   {'start':'2020-01-01','end':None,
                                    'source_id':'min_best_broker_cash_rolling_experiment'}]}
                elif flaw == 'calendar':
                    changed.target_manifest['target_contract']['exchange_calendar_sha256'] = '0'*64
                elif flaw == 'censored':
                    changed.raw_target[1] = 0.0
                elif flaw == 'deadline':
                    changed.group_table.loc[0,'label_eval_end_date'] = days[51]
                elif flaw == 'dataset':
                    changed.summary['dataset'] = 'reduced'
                elif flaw == 'sample_count':
                    changed.summary['score_eligible_sample_count'] = 0
                elif flaw == 'duplicate':
                    changed.group_table.loc[1,'ticker'] = '2330'
                with self.assertRaises(DailyTrade40OracleBlockedError):
                    load_synthetic(changed, days)

    def test_no_mature_outcomes_and_all_censored_day_block(self):
        days, bundle = fixture()
        with self.assertRaisesRegex(DailyTrade40OracleBlockedError, 'no 40-session mature'):
            load_synthetic(bundle, days, cutoff=str(days[25].date()))
        changed = SimpleNamespace(**vars(bundle))
        changed.target_valid = bundle.target_valid.copy()
        changed.raw_target = bundle.raw_target.copy()
        changed.target_valid[0] = False
        changed.target_valid[2] = False
        changed.raw_target[0] = np.nan
        changed.raw_target[2] = np.nan
        with self.assertRaisesRegex(DailyTrade40OracleBlockedError, 'no settled Net-R'):
            load_synthetic(changed, days)

    def test_real_canonical_daily_producer_feeds_oracle_adapter_on_synthetic_prices(self):
        # The earlier source tests injected a ready-made label bundle. This
        # test also executes the ACTUAL MR-13CL daily sample/label producer and
        # links its emitted manifest to the new Oracle adapter end to end.
        import filters.breakout_quality.daily_ranker_data as loader
        from tools.validate.synthetic_daily_trade40_cases import fixture as label_fixture
        from filters.breakout_quality.daily_trade40_target import ExecutableMinPeriod

        days = pd.bdate_range('2019-01-02', periods=850)
        def stock(slope):
            close = 100. + slope * np.arange(len(days))
            return pd.DataFrame({
                'Open':close, 'High':close+2., 'Low':close-2.,
                'Close':close, 'Volume':np.full(len(days),1_000_000.)
            },index=days)
        frames = {'0050':stock(.07), '2330':stock(.20), '2317':stock(-.04)}
        params = label_fixture()[2]
        teacher_sources = [
            {'source_id':'selection_min_roos','sha256':'a'*64},
            {'source_id':'min_best_broker_cash_rolling_experiment','sha256':'b'*64},
        ]
        teacher_periods = [
            {'start':str(days[0].date()),'end':'2020-12-31',
             'source_id':'selection_min_roos','params_signature':'p2'},
            {'start':'2021-01-01','end':None,
             'source_id':'min_best_broker_cash_rolling_experiment','params_signature':'p16'},
        ]
        schedule = (
            ExecutableMinPeriod(days[0],pd.Timestamp('2020-12-31'),params,
                                'selection_min_roos','p2'),
            ExecutableMinPeriod(pd.Timestamp('2021-01-01'),None,params,
                                'min_best_broker_cash_rolling_experiment','p16'),
        )
        dataset_summary = {
            'dataset':'full',
            'source_data_date_range':{
                'start':str(days[0].date()),'end':str(days[-1].date())
            },'policy':{},'dataset_artifacts':{},
        }
        outer = {'effective_oos_end_date':str(days[-1].date()),
                 'selection_end_date':str(days[730].date()),
                 'oos_start_date':str(days[731].date()),
                 'oos_end_date':str(days[-1].date())}
        with patch.object(loader,'load_validated_dataset_bundle',
                          return_value=(dataset_summary,None,None,None,pd.DataFrame())), \
             patch.object(loader,'resolve_breakout_quality_outer_policy',return_value=outer), \
             patch.object(loader,'discover_dataset_csv_inputs',
                          return_value=([(ticker,Path(ticker)) for ticker in frames],None)), \
             patch.object(loader,'load_dataset_frame',
                          side_effect=lambda path,ticker,**kwargs:frames[ticker]), \
             patch.object(loader,'load_historical_min_execution_schedule',
                          return_value=(schedule,{'policy':'synthetic',
                           'sources':teacher_sources,'periods':teacher_periods})), \
             patch.object(loader,'collect_canonical_exchange_sessions',return_value=days), \
             patch('services.audit.daily_trade40_oracle.discover_dataset_csv_inputs',
                   return_value=([(ticker,Path(ticker)) for ticker in frames],[])), \
             patch('services.audit.daily_trade40_oracle.collect_canonical_exchange_sessions',
                   return_value=days):
            scores, evidence = build_daily_universal_trade40_oracle_scores(
                project_root=Path('/tmp/isolated'),dataset='full',
                filter_id='breakout_quality_v1',profile_id=PROFILE,
                cutoff=str(days[-1].date()),
            )
        self.assertGreater(len(scores),100)
        self.assertGreater(evidence['settled_net_r_rows'],100)
        self.assertEqual(evidence['mature_eligible_rows'],len(scores))
        self.assertTrue(all(0<=rank<=1 for rank in scores.values()))
        self.assertEqual(evidence['teacher']['sources'],teacher_sources)
        self.assertEqual(evidence['research_cutoff'],str(days[-1].date()))

    def test_missing_candidate_score_fails_closed_not_price_or_pit_fallback(self):
        days, bundle = fixture()
        scores, _ = load_synthetic(bundle, days)
        current=definition()
        exact_arm=oracle._contract(current)[2]
        with audit_oracle_truth_score_context(
            scores, ranking_policy=exact_arm.dl_runtime_mode,
            ranking_options=exact_arm.dl_runtime_options,
            key_mode='information_date',
        ):
            with self.assertRaisesRegex(ValueError, '未覆蓋可交易候選'):
                resolve_breakout_quality_candidate_rank(
                    ticker='9999', signal_date=days[9], information_date=days[10],
                    high_len=260, filter_id='breakout_quality_v1')


class DailyTrade40OracleReplayCases(TestCase):
    def test_exact_direct_4seed_same_daily_scores_and_old_cache_reused(self):
        active = definition()
        earlier = replace(active, source={**active.source, 'oracle_comparisons':tuple(
            item for item in active.source['oracle_comparisons']
            if item['target'] not in {EXACT, DIRECT})})
        plan = oracle._oracle_comparison_plan(earlier)
        price_scores = {name: {('2330','2025-01-03'):0.8} for name in plan['recipes']}
        for name, source in plan['aliases'].items():
            if source in price_scores:
                price_scores[name] = price_scores[source]
        cohort = {'cohort_rows':100, 'first_common_day':'2021-01-01',
                  'last_common_day':'2025-12-24','research_cutoff':'2026-03-02',
                  'providers': {'40d': earlier.source['pure_mfe40_provider']}}
        daily_scores = {('2330','2025-01-03'):0.625}
        daily_evidence = {
            'label_id':LABEL_ID, 'target_id':TARGET_ID, 'profile_id':PROFILE,
            'rank_policy':'average_zero_based_completed_net_r_censored_rank_last_no_r_imputation',
            'rank_sha256':'c'*64, 'raw_label_sha256':'d'*64, 'calendar_sha256':'e'*64,
            'teacher': {'sources':[{'sha256':'f'*64}], 'periods':[{'start':'2021-01-01'}]},
            'mature_eligible_rows':25, 'settled_net_r_rows':20,
            'censored_rank_last_rows':5, 'research_cutoff':'2026-03-02',
        }
        seeds=(693545351, 2014432738, 104504529, 762252050)
        benchmark=SimpleNamespace(benchmark_id='end_to_end_v1')
        event_scores={seed:{('2330','2025-01-02'):0.6} for seed in seeds}
        event_evidence={seed:{'label_id':'LABEL-mintrade40_realized_net_r_v1'} for seed in seeds}
        observed=[]
        def replay(*, name, **kwargs):
            if name.startswith('Oracle DailyUniversal_'):
                score = resolve_breakout_quality_candidate_rank(
                    ticker='2330', signal_date='2025-01-02', information_date='2025-01-03',
                    high_len=260, filter_id='breakout_quality_v1',
                )
                context=get_breakout_quality_ranking_source_context()
                observed.append((name, score['score'],score['score_date'],context.ranking_policy))
            return {'name':name}
        def metrics(result, *, target):
            name=result['name']
            seed=int(name.split('seed=')[1].split()[0])
            romd=(7.0 if target=='C96' else
                  9.0 if target == EXACT else
                  6.0 if target == DIRECT else 8.0)
            return {
                'target':target,'seed':seed, 'total_return_pct':100.0,
                'annualized_return_pct':12.0,'mdd_pct':10.0,'romd':romd,
                'ev_a_r':0.5,'trades':25,'exposure_pct':60.0,
                'yearly':[{'year':2025,'strategy_return_pct':5.0,'is_full_year':True}],
            }
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(oracle,'collect_status',return_value={'status':'READY'}), \
             patch.object(oracle,'_robustness_artifacts',return_value=(benchmark,[(seed,Path(f'seed_{seed}.json')) for seed in seeds])), \
             patch.object(oracle,'_build_oracle_score_panels',return_value=(price_scores,cohort)), \
             patch.object(oracle,'load_strategy_param_evaluation_view',return_value=({'kind':'rolling_active_param_ensemble','payload':{}},'a'*64)), \
             patch.object(oracle,'_validate_requested_param_policy'), \
             patch.object(oracle,'get_active_param_ensemble_date_range',return_value=('2020-01-01','2026-03-02')), \
             patch.object(oracle,'_build_controlled_param_source_pair',return_value=('rolling_active_param_ensemble','base','oracle',None,None,None)), \
             patch.object(oracle,'price_inventory_sha256',return_value='b'*64), \
             patch('services.audit.min_trade40_target.build_seed_oracle_scores',return_value=(event_scores,event_evidence)), \
             patch('services.audit.daily_trade40_oracle.build_daily_universal_trade40_oracle_scores',return_value=(daily_scores,daily_evidence)) as source, \
             patch.object(oracle,'_run_scenario',side_effect=replay) as run, \
             patch.object(oracle,'_metrics',side_effect=metrics):
            root=Path(temp)
            old_result=oracle.run_formal_audit(earlier,project_root=root,quiet=True)
            self.assertEqual(run.call_count,48)
            self.assertEqual(len(old_result['oracles']),11)
            active_result=oracle.run_formal_audit(active,project_root=root,quiet=True)
            self.assertEqual(run.call_count,56,'only 4x2 new daily arms may run')
            self.assertEqual(len(active_result['oracles']),13)
            self.assertEqual(len(observed),8)
            self.assertTrue(all(score==0.625 and date=='2025-01-03' for _,score,date,_ in observed))
            exact_modes={mode for name,score,date,mode in observed if '_Direct' not in name}
            direct_modes={mode for name,score,date,mode in observed if '_Direct' in name}
            self.assertEqual(len(exact_modes),1)
            self.assertEqual(direct_modes,{'resource-aware-continuous-score-no-k-no-r0'})
            self.assertNotEqual(exact_modes,direct_modes)
            self.assertEqual(source.call_count,1)
            new_rows={row['target']:row for row in active_result['oracles'][-2:]}
            self.assertEqual(new_rows[EXACT]['delta_romd_vs_c96'],2.0)
            self.assertEqual(new_rows[DIRECT]['delta_romd_vs_c96'],-1.0)
            second=oracle.run_formal_audit(active,project_root=root,quiet=True)
            self.assertEqual(run.call_count,56,'all existing arms must REUSE with exact same certified score')
            self.assertEqual(second['oracles'],active_result['oracles'])
            self.assertEqual(source.call_count,2)
            # Different raw target lineage must force a different new score
            # identity, without invalidating any legacy arm.
            changed={**daily_evidence,'raw_label_sha256':'0'*64}
            source.return_value=(daily_scores,changed)
            with self.assertRaisesRegex(oracle.OracleBlockedError, 'DIFFERENT identity'):
                oracle.run_formal_audit(active,project_root=root,quiet=True)
            self.assertEqual(run.call_count,56, 'conflicting completed Target identity must BLOCK, not silently REBUILD')

    def test_target_source_blocked_before_any_portfolio_replay(self):
        current=definition()
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(oracle, 'collect_status', return_value={'status':'READY'}), \
             patch.object(oracle, '_robustness_artifacts', return_value=(SimpleNamespace(benchmark_id='end_to_end_v1'),[(1,Path('seed.json'))])), \
             patch.object(oracle, '_build_oracle_score_panels', return_value=({'MFE40_Control':{}},{'last_common_day':'2025-12-24'})), \
             patch('services.audit.daily_trade40_oracle.build_daily_universal_trade40_oracle_scores',side_effect=DailyTrade40OracleBlockedError('missing canonical P16 teacher')), \
             patch.object(oracle, '_run_scenario') as no_replay:
            with self.assertRaisesRegex(oracle.OracleBlockedError,'missing canonical P16 teacher'):
                oracle.build_report(current,project_root=Path(tmp),quiet=True)
            no_replay.assert_not_called()


if __name__ == '__main__':
    main()
