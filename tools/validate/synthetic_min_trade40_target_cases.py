"""Isolated primitive and contract checks for the P16 Min 40-session event truth.

No formal Windows bundle, historical data, or model fitting is executed.
"""
from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.audit_policy import get_reusable_audit_definitions
from core.research_report_contract import require_approved_report_contract
from core.strategy_params import V16StrategyParams
from filters.breakout_quality.trade_path_label import (
    TRADE_PATH_REASON_HORIZON_SELL_BLOCKED,
    TRADE_PATH_REASON_HORIZON_SAME_DAY_FILL,
    TRADE_PATH_REASON_UNFILLED_HORIZON,
    simulate_realized_trade_path_label,
)
from services.audit import min_trade40_target as target


def fixture():
    params = V16StrategyParams(
        use_breakout_ema_filter=False, use_bb=False, use_kc=False, use_vol=False,
        use_breakout_return_filter=False, use_breakout_false_filter=False,
        use_history_threshold=False, use_breakout_reclaim_reentry=False,
        use_breakout_quality_filter=False, use_breakout_quality_ranking=False,
        high_len=3, atr_len=3, atr_buy_tol=1.0, atr_times_init=2.0,
        atr_times_trail=2.0, tp_percent=0.0, initial_capital=1_000_000.0,
        fixed_risk=0.01, max_position_cap_pct=0.3,
    )
    dates = pd.date_range('2020-01-01', periods=9, freq='B')
    data = pd.DataFrame({
        'Open': [95, 98, 100, 103, 101, 103, 104, 105, 1100],
        'High': [97, 100, 102, 103.5, 102, 105, 105, 106, 1101],
        'Low': [94, 97, 99, 102, 100, 102, 103, 104, 1099],
        'Close': [96, 99, 101, 103, 101, 104, 104, 105, 1100],
        'Volume': [1000.0] * 9,
    }, index=dates)
    data.attrs['ticker'] = '2330'
    buy = np.array([False, False, True, False, False, False, False, False, False])
    indicators = (np.full(9, 2.0), buy, np.zeros(9, dtype=bool),
                  np.array([np.nan, np.nan, 101.0] + [np.nan] * 6))
    return params, data, indicators


def evaluate(params, data, signals, deadline=None):
    return simulate_realized_trade_path_label(
        data, ticker='2330', signal_pos=2, params=params,
        teacher_effective_date='2020-01-01', precomputed_signals=signals,
        horizon_end_date=deadline,
    )


class MinTrade40ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.definition = next(d for d in get_reusable_audit_definitions('breakout_quality')
                              if d.audit_type == 'target_ground_truth_strategy_romd')

    def test_new_target_is_normal_oracle_treatment_with_seed_scoped_truth(self):
        spec = target._target_spec(self.definition)
        self.assertEqual(spec['target'], 'MinTrade40_NetR')
        self.assertEqual(spec['horizon_trading_bars'], 40)
        from types import SimpleNamespace
        bad = dict(self.definition.source)
        bad['event_targets'] = [dict(bad['event_targets'][0], horizon_trading_bars=20)]
        with self.assertRaisesRegex(target.MinTrade40BlockedError, '身分'):
            target._target_spec(SimpleNamespace(source=bad))
        self.assertEqual(spec['cohort'], 'p16_min_signal_events')
        self.assertEqual(require_approved_report_contract(target.REPORT_ID).version, 1)
        comparison = next(row for row in self.definition.source['oracle_comparisons']
                          if row['target'] == target.TARGET_NAME)
        self.assertEqual(comparison['score'], 'trade_path_realized_net_r')
        self.assertEqual(comparison['selector'], 'exact')
        self.assertNotEqual(self.definition.output_subdir, target.OUTPUT_SUBDIR)

    def test_excluded_signals_rank_last_without_fabricated_realized_r(self):
        events = [
            {'ticker': '2330', 'signal_date': '2022-01-03', 'realized_net_r': -3.0},
            {'ticker': '2317', 'signal_date': '2022-01-03', 'realized_net_r': 2.0},
            {'ticker': '2303', 'signal_date': '2022-01-03', 'realized_net_r': None},
            {'ticker': '2454', 'signal_date': '2022-01-04', 'realized_net_r': 1.0},
        ]
        scores = target._rank_seed_events(events)
        self.assertLess(scores[('2303', '2022-01-03')], scores[('2330', '2022-01-03')])
        self.assertLess(scores[('2330', '2022-01-03')], scores[('2317', '2022-01-03')])
        self.assertIsNone(events[2]['realized_net_r'])
        self.assertTrue(all(math.isfinite(value) for value in scores.values()))
        with self.assertRaisesRegex(target.MinTrade40BlockedError, '重複'):
            target._rank_seed_events([events[0], events[0]])

    def test_signal_origin_score_survives_continuation_without_leaking_to_next_replay(self):
        from filters.breakout_quality.runtime import (
            audit_oracle_truth_score_context, resolve_breakout_quality_candidate_rank,
        )
        base = dict(ticker='2330', high_len=23, filter_id='breakout_quality_v1')
        with audit_oracle_truth_score_context(
            {('2330', '2022-01-03'): 0.9},
            ranking_policy='resource-aware-continuous-score-constrained-optimal',
            key_mode='signal_origin_date',
        ):
            value = resolve_breakout_quality_candidate_rank(
                **base, signal_date='2022-01-03', information_date='2022-01-05',
            )
            self.assertEqual(value['score'], 0.9)
            self.assertEqual(value['score_date'], '2022-01-03')
            with self.assertRaisesRegex(RuntimeError, '巢狀'):
                with audit_oracle_truth_score_context(
                    {('2330', '2022-01-03'): 0.9},
                    ranking_policy='resource-aware-continuous-score-constrained-optimal',
                ):
                    pass
        # Outside Oracle context an ordinary runtime cannot consume future truth.
        from core.breakout_quality_ranking_context import get_breakout_quality_ranking_source_context
        self.assertNotEqual(get_breakout_quality_ranking_source_context().score_source,
                            'audit_oracle_future_truth')

    def test_same_canonical_accounting_at_cap_and_independence_from_later_prices(self):
        params, data, signals = fixture()
        date = data.index[5]
        capped = evaluate(params, data, signals, deadline=date)
        trimmed = data.iloc[:6].copy()
        reference = evaluate(params, trimmed, tuple(x[:6] for x in signals))
        self.assertEqual(capped.exit_date, date.strftime('%Y-%m-%d'))
        self.assertEqual(capped.exit_price, data.loc[date, 'Close'])
        self.assertEqual(capped.exit_reason, 'HORIZON_CLOSEOUT')
        self.assertTrue(capped.forced_closeout)
        self.assertAlmostEqual(capped.realized_net_r, reference.realized_net_r, places=10)
        self.assertAlmostEqual(capped.realized_net_pnl, reference.realized_net_pnl, places=10)
        self.assertGreater(evaluate(params, data, signals).realized_net_r, 100,
                           'A2 original uncapped path must remain unchanged')
        edited = data.astype(float).copy()
        edited.loc[edited.index[6]:, ['Open', 'High', 'Low', 'Close']] *= 0.1
        self.assertAlmostEqual(evaluate(params, edited, signals, deadline=date).realized_net_r,
                               capped.realized_net_r, places=10)

    def test_legal_earlier_indicator_exit_has_priority_over_horizon(self):
        params, data, signals = fixture()
        indicators = list(signals)
        indicators[2] = indicators[2].copy()
        indicators[2][4] = True
        early = evaluate(params, data, tuple(indicators), deadline=data.index[7])
        self.assertEqual(early.exit_reason, 'IND_SELL')
        self.assertEqual(early.exit_date, data.index[5].strftime('%Y-%m-%d'))
        self.assertFalse(early.forced_closeout)
        self.assertAlmostEqual(early.realized_net_r,
                               evaluate(params, data, tuple(indicators)).realized_net_r)

    def test_unfilled_and_same_day_horizon_never_create_trades(self):
        params, data, signals = fixture()
        no_fill = evaluate(params, data, signals, deadline=data.index[3])
        self.assertEqual(no_fill.reason, TRADE_PATH_REASON_UNFILLED_HORIZON)
        self.assertIsNone(no_fill.realized_net_r)
        self.assertEqual(no_fill.label_eval_end_date, data.index[3].strftime('%Y-%m-%d'))
        same_day = evaluate(params, data, signals, deadline=data.index[4])
        self.assertEqual(same_day.reason, TRADE_PATH_REASON_HORIZON_SAME_DAY_FILL)
        self.assertIsNone(same_day.realized_net_r)
        self.assertEqual(same_day.fill_date, data.index[4].strftime('%Y-%m-%d'))

    def test_suspended_and_missing_deadline_bar_are_excluded(self):
        params, data, signals = fixture()
        suspended = data.copy()
        suspended.loc[suspended.index[5], 'Volume'] = 0
        result = evaluate(params, suspended, signals, deadline=data.index[5])
        self.assertEqual(result.reason, TRADE_PATH_REASON_HORIZON_SELL_BLOCKED)
        self.assertIsNone(result.realized_net_r)
        missing = data.drop(index=data.index[5]).copy()
        use = np.asarray([d != data.index[5] for d in data.index])
        result = evaluate(params, missing, tuple(np.asarray(x)[use] for x in signals),
                          deadline=data.index[5])
        self.assertEqual(result.reason, TRADE_PATH_REASON_HORIZON_SELL_BLOCKED)
        self.assertIsNone(result.exit_date)

    def test_p16_member_count_cannot_silently_switch_to_ensemble(self):
        with patch.object(target, 'build_active_param_ensemble_objects_from_payload',
                          return_value=[{'effective_date': '2020-01-01',
                                         'members': [{'params_obj': object()}, {'params_obj': object()}]}]):
            with self.assertRaisesRegex(target.MinTrade40BlockedError, '多member'):
                target._schedule('rolling_active_param_ensemble', {})
        with self.assertRaisesRegex(target.MinTrade40BlockedError, 'rolling P16'):
            target._schedule('static', {})

    def test_per_seed_cache_identity_reuse_tampering_and_no_cross_seed(self):
        spec = target._target_spec(self.definition)
        source = dict(spec=spec, dataset='full', price_sha='a' * 64,
                      param_sha='b' * 64, seed=11, start='2021-01-01', cutoff='2026-03-02')
        first = target._seed_identity(**source)
        second = target._seed_identity(**{**source, 'seed': 22})
        third = target._seed_identity(**{**source, 'param_sha': 'c' * 64})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = target._cache_path(root, 11, first)
            truth = {'seed': 11, 'mean_net_r': 0.5, 'events': [{'ticker': '2330'}]}
            target._save_seed_cache(path, first, truth)
            self.assertEqual(target._load_seed_cache(path, first), truth)
            self.assertNotEqual(path, target._cache_path(root, 22, second))
            self.assertNotEqual(path, target._cache_path(root, 11, third))
            self.assertIsNone(target._load_seed_cache(target._cache_path(root, 22, second), second))
            with self.assertRaisesRegex(target.MinTrade40BlockedError, '身分／內容衝突'):
                target._load_seed_cache(path, second)
            path.write_text(path.read_text(encoding='utf-8').replace('0.5', '0.9'), encoding='utf-8')
            with self.assertRaisesRegex(target.MinTrade40BlockedError, '身分／內容衝突'):
                target._load_seed_cache(path, first)

    def test_event_report_consumes_injected_p16_context_with_unchanged_cache_identity(self):
        spec = target._target_spec(self.definition)
        evidence = {'seed': 11, 'param_sha256': 'b' * 64, 'param_path': 'test/p16.json'}
        ctx = target.MinTrade40P16Context(
            dataset='full', start_date='2021-01-01', cutoff_date='2026-03-02',
            benchmark_id='end_to_end_v1',
            prepared=((11, 'b' * 64, 'rolling_oos_param_schedule', {}, {}),),
            parameter_evidence=(evidence,),
        )
        identity = target._seed_identity(
            spec=spec, dataset=ctx.dataset, price_sha='a' * 64,
            param_sha='b' * 64, seed=11, start=ctx.start_date,
            cutoff=ctx.cutoff_date,
        )
        truth = {
            'seed': 11, 'signal_events': 1, 'completed_trades': 1,
            'excluded_events': 0, 'horizon_closeouts': 1, 'positive_trades': 1,
            'mean_net_r': 0.5, 'median_net_r': 0.5,
            'p10_net_r': 0.5, 'p90_net_r': 0.5,
            'min_signal_date': '2021-01-04', 'max_signal_date': '2021-01-04',
            'reasons': {'completed': 1}, 'events': [{'ticker': '2330'}],
        }
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target._save_seed_cache(target._cache_path(root, 11, identity), identity, truth)
            with patch.object(target, 'price_inventory_sha256', return_value='a' * 64), \
                 patch.object(target, '_build_seed_events', side_effect=AssertionError('REUSE must not recalculate')):
                report = target.build_report(
                    self.definition, project_root=root,
                    resolve_p16_context=lambda: (calls.append('resolved') or ctx),
                    quiet=True,
                )
                target.run_validation(
                    self.definition, project_root=root,
                    resolve_p16_context=lambda: (calls.append('resolved') or ctx),
                    quiet=True,
                )
                self.assertTrue((root / 'outputs/audit' / target.OUTPUT_SUBDIR / 'latest' / 'audit.md').is_file())
        self.assertEqual(calls, ['resolved', 'resolved'])
        self.assertEqual(report['cache_decisions'], {11: 'REUSE'})
        self.assertEqual(report['benchmark_id'], ctx.benchmark_id)
        self.assertEqual(report['parameter_artifacts'], [evidence])
        self.assertEqual(report['rows'][0]['mean_net_r'], 0.5)
        self.assertNotIn('events', report['rows'][0])
        self.assertTrue(report['hindsight_non_pit'])

    def test_event_builder_counts_actual_net_r_without_fabricated_zeros(self):
        params, data, signals = fixture()
        bench = data.copy()
        schedule = [{'effective_date': data.index[0],
                     'effective_end_date': data.index[-1], 'params_obj': params}]
        paths = [('0050', Path('/tmp/0050.csv')), ('2330', Path('/tmp/2330.csv'))]
        with patch.object(target, 'discover_dataset_csv_inputs', return_value=(paths, [])), \
             patch.object(target, 'load_dataset_frame', side_effect=lambda path,ticker,min_rows: (
                  bench.copy() if ticker == '0050' else data.copy())), \
             patch.object(target, '_schedule', return_value=tuple(schedule)), \
             patch.object(target, 'build_signal_cache', side_effect=lambda frame,p,ticker: (
                 signals if ticker == '2330' else (signals[0], np.zeros(len(frame), dtype=bool),
                                                   signals[2], signals[3]))):
            actual = target._build_seed_events(
                root=ROOT, dataset='full', start='2020-01-01', cutoff='2020-01-13',
                spec={**target._target_spec(self.definition), 'horizon_trading_bars': 3},
                seed=11, kind='rolling_active_param_ensemble', base_params={},
            )
        self.assertEqual(actual['completed_trades'], 1)
        self.assertEqual(actual['excluded_events'], 0)
        self.assertEqual(actual['horizon_closeouts'], 1)
        self.assertEqual(actual['signal_events'], 1)
        self.assertTrue(math.isfinite(actual['mean_net_r']))
        self.assertEqual(actual['events'][0]['horizon_date'], data.index[5].strftime('%Y-%m-%d'))
        self.assertEqual(actual['events'][0]['exit_date'], data.index[5].strftime('%Y-%m-%d'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
