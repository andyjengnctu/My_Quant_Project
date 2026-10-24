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
from core.portfolio_fast_access import get_fast_dates
from core.portfolio_fast_data import (
    pack_static_market_data, merge_static_market_with_dynamic,
    build_normal_setup_index,
)
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


def prepared_stock(frame, signal_flags=None, *, atr=None, buy_limit=None, sell_flags=None):
    """Fixture using the actual packed-market adapter consumed by Portfolio."""
    n = len(frame)
    stock = frame.copy()
    if not {"Open", "High", "Low", "Close", "Volume"}.issubset(stock.columns):
        stock = pd.DataFrame({
            "Open": np.full(n, 100.0), "High": np.full(n, 101.0),
            "Low": np.full(n, 99.0), "Close": np.full(n, 100.0),
            "Volume": np.full(n, 1000.0),
        }, index=frame.index)
    return merge_static_market_with_dynamic(pack_static_market_data(stock), {
        "ATR": np.full(n, 2.0) if atr is None else np.asarray(atr, dtype=float),
        "is_setup": (np.zeros(n, dtype=bool) if signal_flags is None
                     else np.asarray(signal_flags, dtype=bool)),
        "ind_sell_signal": (np.zeros(n, dtype=bool) if sell_flags is None
                            else np.asarray(sell_flags, dtype=bool)),
        "buy_limit": (np.full(n, 100.0) if buy_limit is None
                      else np.asarray(buy_limit, dtype=float)),
    })


def portfolio_context(market):
    # Formal Portfolio timeline is the union of prepared stock-session dates.
    sorted_dates = sorted({pd.Timestamp(day) for packed in market.values()
                           for day in get_fast_dates(packed)})
    return {"all_dfs_fast": market, "sorted_dates": sorted_dates,
            "normal_setup_index": build_normal_setup_index(market)}


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
        self.assertEqual(spec['signal_membership_policy'],
                         'canonical_portfolio_setup_index_p16_v2')
        self.assertEqual(spec['horizon_calendar_policy'],
                         'canonical_portfolio_replay_sorted_dates_v1')
        # AI: One declarative field scopes BOTH seed-label cache and the
        # normal Oracle Replay arm, without a target-specific cache bypass.
        identity = target._seed_identity(
            spec=spec, dataset='full', price_sha='a'*64, param_sha='b'*64,
            seed=11, start='2021-01-01', cutoff='2026-03-02',
        )
        self.assertEqual(identity['signal_membership_policy'],
                         spec['signal_membership_policy'])
        self.assertEqual(identity['horizon_calendar_policy'],
                         spec['horizon_calendar_policy'])
        # Maturity semantics are part of both seed truth and Oracle Replay
        # identities; the previous 0050-quote-calendar cache must not REUSE.
        legacy_spec = dict(spec, horizon_calendar_policy='0050_quote_dates_legacy')
        old_identity = target._seed_identity(
            spec=legacy_spec, dataset='full', price_sha='a'*64, param_sha='b'*64,
            seed=11, start='2021-01-01', cutoff='2026-03-02',
        )
        self.assertNotEqual(identity, old_identity)
        self.assertNotEqual(target._cache_path(ROOT, 11, identity),
                            target._cache_path(ROOT, 11, old_identity))
        from services.audit.target_oracle_romd import (
            _arm_replay_policy, _oracle_comparison_plan,
        )
        arm_policy = _arm_replay_policy(
            self.definition, plan=_oracle_comparison_plan(self.definition),
            target=target.TARGET_NAME, replay_policy={},
        )
        self.assertEqual(arm_policy['hindsight_label_spec']['signal_membership_policy'],
                         spec['signal_membership_policy'])
        self.assertEqual(arm_policy['hindsight_label_spec']['horizon_calendar_policy'],
                         spec['horizon_calendar_policy'])
        self.assertEqual(require_approved_report_contract("audit.target_oracle_romd").report_id,
                         "audit.target_oracle_romd")
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

    def test_reentry_inherits_origin_score_from_formal_parent_state(self):
        # Re-entry is not a separately scanned target event: the formal engine
        # inherits the original normal setup's frozen hindsight ranking.
        from core.breakout_reentry import create_breakout_reentry_signal_state
        from core.extended_signals import resolve_breakout_quality_rank
        params, _, _ = fixture()
        params.use_breakout_reclaim_reentry = True
        rank = {
            'score': 0.9, 'available': True,
            'score_date': '2022-01-03',
            'score_source': 'audit_oracle_future_truth',
            'shared_group_score': True, 'filter_id': 'breakout_quality_v1',
        }
        watch = {'ticker': '2330', 'confirm_price': 100.0,
                 'breakout_quality_rank': rank, '_params_obj': params}
        signal = create_breakout_reentry_signal_state(
            watch, close_price=101.0, atr=2.0, params=params,
            ticker='2330', signal_date='2022-01-13',
        )
        self.assertEqual(signal['source'], 'reentry')
        inherited = resolve_breakout_quality_rank(signal)
        self.assertEqual(inherited['score'], 0.9)
        self.assertEqual(inherited['score_date'], '2022-01-03')
        self.assertEqual(inherited['score_source'], 'audit_oracle_future_truth')

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

    def test_oracle_score_producer_reuses_injected_p16_context_and_never_publishes_separate_report(self):
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
            'reasons': {'completed': 1},
            'events': [{'ticker': '2330', 'signal_date': '2021-01-04', 'realized_net_r': 0.5}],
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
                seed_scores, score_evidence = target.build_seed_oracle_scores(
                    self.definition, project_root=root,
                    resolve_p16_context=lambda: (calls.append('resolved') or ctx),
                    quiet=True,
                )
                self.assertFalse((root / 'outputs/audit' / target.OUTPUT_SUBDIR / 'latest').exists(),
                                 'MinTrade40 is one normal Oracle treatment, not a separate report')
        self.assertEqual(calls, ['resolved', 'resolved'])
        self.assertTrue(math.isfinite(seed_scores[11][('2330', '2021-01-04')]))
        self.assertEqual(score_evidence[11]['completed_trades'], 1)
        self.assertEqual(score_evidence[11]['score_key_mode'], 'signal_origin_date')
        self.assertEqual(report['cache_decisions'], {11: 'REUSE'})
        self.assertEqual(report['benchmark_id'], ctx.benchmark_id)
        self.assertEqual(report['parameter_artifacts'], [evidence])
        self.assertEqual(report['rows'][0]['mean_net_r'], 0.5)
        self.assertNotIn('events', report['rows'][0])
        self.assertTrue(report['hindsight_non_pit'])

    def test_event_cohort_uses_formal_setup_index_across_p16_boundaries(self):
        # The official index uses the ticker's NEXT observed bar; a suspended
        # ticker's candidate can legitimately cross the annual P16 boundary.
        calendar = pd.date_range('2020-12-01', '2022-02-28', freq='B')
        by_ticker = {
            '006205': pd.DataFrame(index=pd.to_datetime([
                '2020-12-30', '2021-01-06', '2021-12-20',
                '2022-01-03', '2022-01-04',
            ])),
            '2330': pd.DataFrame(index=pd.to_datetime([
                '2020-12-31', '2021-01-04', '2021-12-29',
                '2021-12-30', '2022-01-03', '2022-01-04',
            ])),
        }
        positions = {
            '006205': {'2020-12-30', '2021-12-20'},
            '2330': {'2020-12-31', '2021-12-29', '2021-12-30'},
        }
        p2021 = SimpleNamespace(use_breakout_quality_filter=False,
                                use_breakout_quality_ranking=False)
        p2022 = SimpleNamespace(use_breakout_quality_filter=False,
                                use_breakout_quality_ranking=False)
        schedule = (
            {'effective_date': '2021-01-01', 'effective_date_text': '2021-01-01',
             'effective_end_date': '2021-12-31', 'params_obj': p2021},
            {'effective_date': '2022-01-01', 'effective_date_text': '2022-01-01',
             'effective_end_date': '2022-12-31', 'params_obj': p2022},
        )
        packed = {'0050': prepared_stock(pd.DataFrame(index=calendar))}
        for ticker, frame in by_ticker.items():
            flags = np.asarray([day.strftime('%Y-%m-%d') in positions[ticker]
                                for day in frame.index], dtype=bool)
            packed[ticker] = prepared_stock(frame, flags)
        canonical = portfolio_context(packed)
        contexts = {'2021-01-01': canonical, '2022-01-01': canonical}

        def completed_trade(frame, *, ticker, signal_pos, params, **kwargs):
            trade_day = frame.index[signal_pos + 1].strftime('%Y-%m-%d')
            return SimpleNamespace(
                reason='realized_net_profit', status='PASS', fill_date=trade_day,
                entry_price=100.0, exit_date=trade_day, exit_price=101.0,
                exit_reason='HORIZON_CLOSEOUT', realized_net_pnl=100.0,
                realized_net_r=1.0, forced_closeout=True,
            )

        with patch.object(target, '_schedule', return_value=schedule), \
             patch.object(target, '_prepared_min_contexts', return_value=contexts), \
             patch.object(target, 'simulate_realized_trade_path_label',
                          side_effect=completed_trade) as label_fn:
            actual = target._build_seed_events(
                root=ROOT, dataset='full', start='2021-01-01', cutoff='2022-02-28',
                spec={**target._target_spec(self.definition), 'horizon_trading_bars': 1},
                seed=11, kind='rolling_active_param_ensemble', base_params={},
            )
        assigned = {(row['ticker'], row['signal_date']): row['p16_effective_date']
                    for row in actual['events']}
        self.assertEqual(assigned, {
            ('006205', '2020-12-30'): '2021-01-01',
            ('006205', '2021-12-20'): '2022-01-01',
            ('2330', '2020-12-31'): '2021-01-01',
            ('2330', '2021-12-29'): '2021-01-01',
            ('2330', '2021-12-30'): '2022-01-01',
        })
        self.assertEqual(label_fn.call_count, len(assigned))
        self.assertEqual(actual['signal_events'], len(assigned))
        self.assertEqual(actual['completed_trades'], len(assigned))
        self.assertEqual(actual['excluded_events'], 0)
        self.assertLess(actual['min_signal_date'], '2021-01-01')
        scores = target._rank_seed_events(actual['events'])
        self.assertEqual(set(scores), set(assigned))
        self.assertTrue(all(math.isfinite(score) for score in scores.values()))

    def test_event_cohort_respects_canonical_sanitized_setup_for_2404(self):
        # A fresh formal setup (2404 / 2025-06-11) must be included even if
        # an independent raw-price scanner would have disagreed or omitted it.
        dates = pd.date_range('2025-06-09', periods=55, freq='B')
        frame = pd.DataFrame(index=dates)
        flags = np.zeros(len(dates), dtype=bool)
        flags[2] = True  # 2025-06-11: official packed signal
        # 0050 has no quote on the legitimate 2404 signal day; the canonical
        # Portfolio market timeline still contains the session.
        missing_benchmark_day = pd.Timestamp('2025-06-11')
        benchmark = frame.drop(index=missing_benchmark_day)
        market = {'0050': prepared_stock(benchmark), '2404': prepared_stock(frame, flags)}
        context = portfolio_context(market)
        self.assertIn(missing_benchmark_day, context['sorted_dates'])
        self.assertNotIn(missing_benchmark_day,
                         pd.DatetimeIndex(get_fast_dates(market['0050'])))
        schedule = ({'effective_date': '2025-01-01',
                     'effective_date_text': '2025-01-01',
                     'effective_end_date': '2025-12-31',
                     'params_obj': SimpleNamespace(use_breakout_quality_filter=False,
                                                   use_breakout_quality_ranking=False)},)

        def complete(frame, **kwargs):
            return SimpleNamespace(
                reason='realized_net_profit', status='PASS', fill_date='2025-06-12',
                entry_price=100.0, exit_date='2025-08-06', exit_price=105.0,
                exit_reason='HORIZON_CLOSEOUT', realized_net_pnl=500.0,
                realized_net_r=0.5, forced_closeout=True,
            )

        with patch.object(target, '_schedule', return_value=schedule), \
             patch.object(target, '_prepared_min_contexts',
                          return_value={'2025-01-01': context}), \
             patch.object(target, 'simulate_realized_trade_path_label',
                          side_effect=complete):
            actual = target._build_seed_events(
                root=ROOT, dataset='full', start='2025-01-01', cutoff='2025-08-25',
                spec=target._target_spec(self.definition), seed=11,
                kind='rolling_active_param_ensemble', base_params={},
            )
        self.assertEqual(actual['signal_events'], 1)
        self.assertEqual(actual['events'][0]['ticker'], '2404')
        self.assertEqual(actual['events'][0]['signal_date'], '2025-06-11')
        self.assertEqual(actual['events'][0]['p16_effective_date'], '2025-01-01')
        self.assertEqual(actual['events'][0]['horizon_date'], dates[42].strftime('%Y-%m-%d'))
        self.assertIn(('2404', '2025-06-11'), target._rank_seed_events(actual['events']))
        # Missing dates in the real Portfolio timeline must still fail closed.
        broken_context = dict(context, sorted_dates=[d for d in context['sorted_dates']
                                                     if d != missing_benchmark_day])
        with patch.object(target, '_schedule', return_value=schedule), \
             patch.object(target, '_prepared_min_contexts',
                          return_value={'2025-01-01': broken_context}):
            with self.assertRaisesRegex(target.MinTrade40BlockedError, '交易日曆'):
                target._build_seed_events(
                    root=ROOT, dataset='full', start='2025-01-01', cutoff='2025-08-25',
                    spec=target._target_spec(self.definition), seed=11,
                    kind='rolling_active_param_ensemble', base_params={},
                )

    def test_canonical_context_loader_keeps_portfolio_universe_and_single_member(self):
        schedules = ({'effective_date_text': '2021-01-01', 'members':
                      [{'params_obj': object()}]},)
        ctx = {'all_dfs_fast': {}, 'normal_setup_index': {}}
        with patch.object(target, 'get_dataset_dir', return_value=Path('/test/full')), \
             patch.object(target, 'resolve_raw_universe_required_min_rows',
                          return_value=617), \
             patch('services.portfolio_replay._load_contexts_for_active_ensemble_schedule',
                   return_value={'2021-01-01': [ctx]}) as formal_loader:
            actual = target._prepared_min_contexts(
                ROOT, 'full', 'rolling_active_param_ensemble',
                {'raw_universe_required_min_rows': 617}, schedules,
            )
        self.assertIs(actual['2021-01-01'], ctx)
        self.assertEqual(formal_loader.call_args.args[0], '/test/full')
        self.assertEqual(formal_loader.call_args.kwargs['raw_universe_required_min_rows'], 617)
        self.assertFalse(formal_loader.call_args.kwargs['keep_trade_logs'])

    def test_event_builder_counts_actual_net_r_without_fabricated_zeros(self):
        params, data, signals = fixture()
        market = {'0050': prepared_stock(data),
                  '2330': prepared_stock(data, signals[1], atr=signals[0],
                                         sell_flags=signals[2], buy_limit=signals[3])}
        ctx = portfolio_context(market)
        schedule = ({'effective_date': data.index[0],
                     'effective_date_text': '2020-01-01',
                     'effective_end_date': data.index[-1], 'params_obj': params},)
        with patch.object(target, '_schedule', return_value=schedule), \
             patch.object(target, '_prepared_min_contexts',
                          return_value={'2020-01-01': ctx}):
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
