"""Economic and PIT invariants for payoff-sensitive binary decisions."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest import TestCase, main

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from filters.breakout_quality.categorical_decision import (
    payoff_training_targets, payoff_cross_entropy,
    payoff_validation_metrics, payoff_trade_decisions,
)
from core.positive_score_admission import (
    positive_score_admitted, requires_positive_score, uses_model_positive_decision,
)


class PayoffDecisionCases(TestCase):
    def test_high_win_rate_with_negative_expected_payoff_learns_cash(self):
        import torch
        old = torch.get_num_threads(); torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old)
        for outcomes, expected in (([1., 1., 1., -10.], 0.), ([-1., -1., -1., 10.], 1.)):
            target = torch.tensor(payoff_training_targets(outcomes, np.arange(4)))
            bias = torch.nn.Parameter(torch.zeros(1, 2))
            optimizer = torch.optim.SGD([bias], lr=.2)
            for _ in range(100):
                optimizer.zero_grad()
                loss = payoff_cross_entropy(torch, bias.expand(4, -1), target)
                loss.backward(); optimizer.step()
            self.assertEqual(payoff_trade_decisions(bias.detach().numpy()).item(), expected)
            self.assertNotEqual(expected, float(np.mean(np.array(outcomes) > 0) > .5))

    def test_weight_normalization_uses_only_explicit_fit_rows(self):
        raw = np.array([2., -6., 0., np.nan, 999999.])
        before = payoff_training_targets(raw, [0, 1, 2])
        raw[3:] = [-1e20, 1e20]
        np.testing.assert_equal(before, payoff_training_targets(raw, [0, 1, 2]))
        np.testing.assert_allclose(before[:3, 1], [.75, 2.25, 0.])
        self.assertTrue(np.isnan(before[3:]).all())
        # Final refit recomputes normalization for its own newly legal cohort.
        np.testing.assert_allclose(payoff_training_targets(raw, [0, 1])[:2, 1], [.5, 1.5])
        for bad in ([0, 0], [-1], [5]):
            with self.assertRaises(ValueError):
                payoff_training_targets(raw, bad)
        with self.assertRaises(ValueError):
            payoff_training_targets([np.nan, 1.], [0, 1])

    def test_zero_payoff_has_no_gradient_and_tie_retains_cash(self):
        import torch
        logits = torch.tensor([[2., 2.], [1., 1.00000001]], dtype=torch.float64)
        self.assertIsNone(payoff_cross_entropy(torch, logits, torch.zeros(2, 2)))
        np.testing.assert_array_equal(payoff_trade_decisions(logits.numpy()), [0., 1.])
        with self.assertRaises(ValueError):
            payoff_trade_decisions([[0., np.nan]])
        with self.assertRaises(ValueError):
            payoff_training_targets([0., 0.], [0, 1])

    def test_epoch_selection_penalizes_large_loss_and_is_currency_scale_invariant(self):
        cash = np.tile([2., 0.], (4, 1))
        trade = cash[:, ::-1]
        outcomes = np.array([1., 1., 1., -10.])
        a = payoff_validation_metrics(cash, outcomes)
        b = payoff_validation_metrics(trade, outcomes)
        self.assertLess(a['cross_entropy'], b['cross_entropy'])
        self.assertLess(a['joint_event_brier'], b['joint_event_brier'])
        self.assertEqual(a, payoff_validation_metrics(cash, outcomes * 1000))

    def test_declared_strategy_has_two_sources_and_no_threshold(self):
        from core.strategy_compare_policy import get_strategy_comparison_settings
        from config.strategy_compare import STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE
        from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
        from filters.breakout_quality.strategy_compare_dl_artifacts import resolve_arm_runtime_dl_source_ids
        settings = get_strategy_comparison_settings(STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE)
        arms = [arm for arm in settings.enabled_arms if uses_model_positive_decision(arm.dl_runtime_options)]
        self.assertTrue(arms)
        for arm in arms:
            options = dict(arm.dl_runtime_options)
            self.assertTrue(requires_positive_score(options))
            self.assertNotIn('positive_probability_threshold', options)
            self.assertEqual(set(resolve_arm_runtime_dl_source_ids(settings, arm)),
                             {arm.dl_id, options['positive_dl_id']})
            recipe = get_continuous_ranker_execution_recipe(settings.dl_sources[options['positive_dl_id']].experiment_profile)
            self.assertEqual(recipe.model_architecture, 'inception_time_v1')
            self.assertEqual(recipe.training_policy.categorical_decision_column, options['positive_score_column'])
            # Only a model-produced binary action at the rank information date can admit.
            payload = dict(available=True, positive_available=True, score_date='2021-01-04',
                           positive_score_date='2021-01-04', positive_probability=1.)
            row = dict(breakout_quality_ranking_options=options, breakout_quality_rank=payload)
            self.assertTrue(positive_score_admitted(row))
            for action in [0., .61, float('nan'), None, 2., -1.]:
                payload['positive_probability'] = action
                self.assertFalse(positive_score_admitted(row))
            payload.update(positive_probability=1., positive_score_date='2021-01-05')
            self.assertFalse(positive_score_admitted(row))
            with self.assertRaises(ValueError):
                requires_positive_score({**options, 'positive_probability_threshold': .5})

    def test_summary_perfect_output_equivalence_requires_matching_evidence(self):
        from unittest.mock import patch
        from services.research import target_dl_overview as overview
        from services.research.target_dl_overview import collect_target_dl_overview
        from core.strategy_compare_registry import STRATEGY_COMPARE_ARMS, STRATEGY_DL_SOURCES
        from core.research_summary import get_research_summary_rows
        from core.positive_score_admission import perfect_positive_arms_equivalent
        from core.audit_registry import oracle_comparison_key
        from core.report_contract_versions import REGISTRY_RELATIVE_PATH
        groups = get_research_summary_rows()
        equivalent = [group for group in groups if group.get('oracle_equivalent_arm_id')]
        self.assertTrue(equivalent)
        # This contract concerns registered rows with no published evidence.
        # A user's model/Oracle history must not change its inputs or trigger
        # real replay planning and price-inventory scans during the synthetic run.
        with tempfile.TemporaryDirectory(prefix="payoff_summary_contract_") as temp:
            root = Path(temp)
            ledger = root / REGISTRY_RELATIVE_PATH
            ledger.parent.mkdir(parents=True)
            ledger.write_bytes((ROOT / REGISTRY_RELATIVE_PATH).read_bytes())
            report = collect_target_dl_overview(root)
            read_oracle = overview._oracle_rows
            for group in equivalent:
                arm_id = group['bindings'][0]['arm_id']
                reference_id = group['oracle_equivalent_arm_id']
                reference = next(row for row in groups if 'oracle' in row and any(
                    binding.get('arm_id') == reference_id for binding in row.get('bindings', ())))
                key = oracle_comparison_key(reference['oracle'])
                rows = [row for row in report['rows'] if row['strategy'] == arm_id]
                self.assertEqual(len(rows), 1)
                self.assertIsNone(rows[0]['oracle_romd'])
                self.assertIsNone(rows[0]['win_rate_pct'])
                def signed_fixture(*args, **kwargs):
                    values = list(read_oracle(*args, **kwargs))
                    values[3] = {key: {'oracle_romd': 12.5, 'win_rate_pct': 75.0}}
                    return tuple(values)
                with patch.object(overview, '_oracle_rows', side_effect=signed_fixture):
                    published = collect_target_dl_overview(root)
                for selected in (arm_id, reference_id):
                    row, = [row for row in published['rows'] if row['strategy'] == selected]
                    self.assertEqual((row['oracle_romd'], row['win_rate_pct']), (12.5, 75.0))
                # Similar labels alone cannot authorize evidence reuse.
                from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
                arm = STRATEGY_COMPARE_ARMS[arm_id]
                target = get_continuous_ranker_execution_recipe(
                    STRATEGY_DL_SOURCES[arm['dl_id']]['experiment_profile']).continuous_target_id
                for mutation in ({'param_source': 'different_teacher'},
                                 {'entry_strategy': 'Breakout'}, {'dl_id': 'different_ranker'}):
                    self.assertFalse(perfect_positive_arms_equivalent(
                        {**arm, **mutation}, STRATEGY_COMPARE_ARMS[reference_id],
                        sources=STRATEGY_DL_SOURCES, target_id=target))


class PerfectDecisionReplayParityCases(TestCase):
    """Bounded perfect-output equivalence, not equality of fitted DI/DJ models."""

    def _options(self, settings, arm):
        secondary = settings.dl_sources[arm.dl_runtime_options['positive_dl_id']]
        return {**arm.dl_runtime_options,
                'positive_filter_id': secondary.filter_id,
                'positive_model_architecture': secondary.model_architecture,
                'positive_experiment_profile': secondary.experiment_profile,
                'positive_score_source': secondary.score_source}

    def _enter_scores(self, stack, *, settings, arm, scores, positive, oracle,
                      misdated=False, calls=None):
        import pandas as pd
        from unittest.mock import patch
        from filters.breakout_quality.runtime import (
            audit_oracle_truth_score_context, breakout_quality_ranking_source_context,
        )
        from filters.breakout_quality.ranking_score_store import SCORE_SOURCE_SELECTION_POINT_IN_TIME
        primary = settings.dl_sources[arm.dl_id]
        options = self._options(settings, arm)
        if oracle:
            oracle_positive = ({(ticker, '2099-01-01'): value
                                for (ticker, _), value in positive.items()}
                               if misdated else positive)
            stack.enter_context(audit_oracle_truth_score_context(
                scores, ranking_policy=arm.dl_runtime_mode, ranking_options=options,
                positive_scores=oracle_positive, key_mode='information_date',
                missing_key_policy='unavailable_outside_mature_feature_universe'))
            stack.enter_context(patch(
                'filters.breakout_quality.runtime.lookup_selection_point_in_time_candidate_score',
                side_effect=AssertionError('Oracle must not fall back to a PIT lookup')))
        else:
            def lookup(*, ticker, signal_date, filter_id, experiment_profile,
                       score_column=None, **kwargs):
                day = pd.Timestamp(signal_date).strftime('%Y-%m-%d')
                secondary = score_column is not None
                self.assertEqual(experiment_profile,
                                 options['positive_experiment_profile'] if secondary else primary.experiment_profile)
                self.assertEqual(score_column, options['positive_score_column'] if secondary else None)
                if calls is not None:
                    calls.append((str(ticker), day, experiment_profile, score_column))
                value = (positive if secondary else scores).get((str(ticker), day))
                return {'score': value, 'available': value is not None and np.isfinite(value),
                        'score_date': '2099-01-01' if secondary and misdated else day,
                        'filter_id': filter_id, 'unavailable_reason': '' if value is not None else 'missing_ticker_date_score',
                        'score_source': SCORE_SOURCE_SELECTION_POINT_IN_TIME}
            stack.enter_context(breakout_quality_ranking_source_context(
                score_source=SCORE_SOURCE_SELECTION_POINT_IN_TIME,
                model_architecture=primary.model_architecture,
                experiment_profile=primary.experiment_profile,
                ranking_policy=arm.dl_runtime_mode, ranking_options=options))
            stack.enter_context(patch(
                'filters.breakout_quality.runtime.lookup_selection_point_in_time_candidate_score',
                side_effect=lookup))
        return options

    def test_perfect_output_boundary_and_information_date_are_identical(self):
        from contextlib import ExitStack
        from core.strategy_compare_policy import get_strategy_comparison_settings
        from config.strategy_compare import STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE
        from core.positive_netpnl_contract import positive_labels
        from filters.breakout_quality.runtime import resolve_breakout_quality_candidate_rank
        settings = get_strategy_comparison_settings(STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE)
        raw = np.array([-7., -0., 0., .001, 9., np.nan, np.inf])
        labels = positive_labels(raw)
        finite = np.isfinite(raw)
        # Perfect utility logits choose cash on an exact-zero tie.
        decisions = payoff_trade_decisions(np.column_stack((np.zeros(finite.sum()), raw[finite])))
        np.testing.assert_array_equal(decisions, labels[finite])
        scores = {(str(i), '2020-12-31'): .75 for i in range(len(raw) + 1)}
        positive = {(str(i), '2020-12-31'): float(value)
                    for i, value in enumerate(labels) if np.isfinite(value)}
        for arm_id in ('C183', 'C186'):
            for oracle in (False, True):
                for misdated in (False, True):
                    with self.subTest(arm=arm_id, oracle=oracle, misdated=misdated), ExitStack() as stack:
                        options = self._enter_scores(stack, settings=settings, arm=settings.arms[arm_id],
                            scores=scores, positive=positive, oracle=oracle, misdated=misdated)
                        admitted = []
                        for i in range(len(raw) + 1):
                            payload = resolve_breakout_quality_candidate_rank(
                                ticker=str(i), signal_date='2020-12-30',
                                information_date='2020-12-31', high_len=10)
                            self.assertEqual(payload['score_date'], '2020-12-31')
                            admitted.append(positive_score_admitted({
                                'breakout_quality_rank': payload, 'breakout_quality_ranking_options': options}))
                        self.assertEqual(admitted, [False] * 8 if misdated else
                                         [False, False, False, True, True, False, False, False])
                        # A previous-year score cannot authorize the next information day.
                        absent = resolve_breakout_quality_candidate_rank(ticker='4',
                            signal_date='2020-12-31', information_date='2021-01-04', high_len=10)
                        self.assertFalse(positive_score_admitted({
                            'breakout_quality_rank': absent, 'breakout_quality_ranking_options': options}))

    def test_real_daily_portfolio_all_accounts_match_across_source_and_arm(self):
        from contextlib import ExitStack
        from dataclasses import replace
        import pandas as pd
        from core.strategy_compare_policy import get_strategy_comparison_settings
        from config.strategy_compare import (
            STRATEGY_COMPARE_CURRENT_OOS_PROFILE, STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE,
        )
        from core.portfolio_entry_policy import (
            DAILY_EXECUTION_CLOSE_FILTERS, daily_signal_execution_context, portfolio_entry_policy_context,
        )
        from core.portfolio_fast_data import (
            prep_optimizer_stock_data_bundle, pack_static_market_data, merge_static_market_with_dynamic,
        )
        from core.portfolio_engine import run_portfolio_timeline
        from tools.validate.synthetic_daily_trade40_oracle_cases import source_fixture
        days, frames, _, schedule = source_fixture()
        days = days[days <= pd.Timestamp('2021-01-29')]
        params = replace(schedule[0].params, use_breakout_buy=False,
            use_breakout_quality_ranking=True, use_breakout_ema_filter=True,
            breakout_ema_len=5, use_vol=True, vol_long_len=5, vol_breakout_mult=.5)
        market, logs = {}, {}
        with daily_signal_execution_context(DAILY_EXECUTION_CLOSE_FILTERS):
            for ticker, scale in (('2330', 1.), ('2317', .85), ('2454', 1.15)):
                frame = frames['2330'].loc[days].copy()
                frame.loc[:, ['Open', 'High', 'Low', 'Close']] *= scale
                frame.attrs['ticker'] = ticker
                prepared, standalone, _ = prep_optimizer_stock_data_bundle(
                    frame, params, ticker=ticker, include_trade_logs=True, include_pit_stats_index=False)
                market[ticker] = merge_static_market_with_dynamic(pack_static_market_data(frame), prepared)
                logs[ticker] = standalone
        keys = {(ticker, str(day.date())) for ticker in market for day in days
                if day >= pd.Timestamp('2020-12-21')}
        raw = {key: (8., 2., 0., -4., None)[(days.get_loc(pd.Timestamp(key[1])) + i) % 5]
               for i, ticker in enumerate(market) for key in keys if key[0] == ticker}
        scores = {key: .95 if value is None else (value + 4.) / 12. for key, value in raw.items()}
        positive = {key: float(value > 0.) for key, value in raw.items() if value is not None}
        self.assertEqual({v for v in raw.values()}, {8., 2., 0., -4., None})

        def assert_same(left, right):
            self.assertEqual(len(left), len(right))
            for index, (a, b) in enumerate(zip(left, right)):
                if isinstance(a, pd.DataFrame):
                    pd.testing.assert_frame_equal(a, b, check_exact=True)
                elif isinstance(a, float) and np.isnan(a) and np.isnan(b):
                    continue
                else:
                    self.assertEqual(a, b, f'portfolio output {index}')

        def economic_rows(value):
            # Only source provenance differs; keep every candidate/order/trade value.
            if isinstance(value, dict):
                return {key: economic_rows(item) for key, item in value.items()
                        if key not in {'breakout_quality_score_source', 'breakout_quality_ranking_options'}}
            if isinstance(value, list):
                return [economic_rows(item) for item in value]
            return value

        for profile in (STRATEGY_COMPARE_CURRENT_OOS_PROFILE, STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE):
            settings = get_strategy_comparison_settings(profile)
            a, b = (settings.arms[key] for key in ('C183', 'C186'))
            self.assertEqual((a.param_source, a.param_policy, a.dl_id, a.rule_policy, a.entry_strategy, a.dl_runtime_mode),
                             (b.param_source, b.param_policy, b.dl_id, b.rule_policy, b.entry_strategy, b.dl_runtime_mode))
            for scenario in ('mixed', 'date_mismatch', 'missing_rank', 'no_positive'):
                primary = ({key: value for key, value in scores.items() if positive.get(key) != 1.}
                           if scenario == 'missing_rank' else scores)
                companion = ({key: 0. for key in positive} if scenario == 'no_positive' else positive)
                reference = None
                for arm in (a, b):
                    for oracle in (False, True):
                        with self.subTest(profile=profile, scenario=scenario, arm=arm.arm_id, oracle=oracle), ExitStack() as stack:
                            calls, executions, counts = [], [], {}
                            self._enter_scores(stack, settings=settings, arm=arm, scores=primary,
                                positive=companion, oracle=oracle, misdated=scenario == 'date_mismatch', calls=calls)
                            stack.enter_context(portfolio_entry_policy_context('Daily', score_keys=keys,
                                daily_execution_contract=DAILY_EXECUTION_CLOSE_FILTERS))
                            result = run_portfolio_timeline(
                                all_dfs_fast=market, all_standalone_logs=logs, sorted_dates=list(days),
                                start_year=2020, params=params, max_positions=2, enable_rotation=False,
                                benchmark_ticker='2330', benchmark_data=market['2330'],
                                is_training=False, verbose=False,
                                profile_stats={'_timing_enabled': False, 'capture_equity_curve': True},
                                replay_execution_rows=executions, replay_counts=counts)
                            if not oracle:
                                self.assertTrue(calls)
                                self.assertTrue(any(call[-1] == arm.dl_runtime_options['positive_score_column'] for call in calls))
                            # Complete daily equity/accounting and every trade column, plus all aggregate outputs.
                            orders = [{key: value for key, value in row.items()
                                       if not key.startswith('_') and key != 'params_obj'} for row in executions]
                            if reference is None:
                                reference = (result, economic_rows(counts), orders)
                            else:
                                assert_same(result, reference[0])
                                self.assertEqual(economic_rows(counts), reference[1])
                                self.assertEqual(orders, reference[2])
                            if scenario == 'mixed':
                                self.assertGreater(int(result[4]), 0)
                                self.assertGreater(len(executions), 0)
                                self.assertEqual(set(pd.to_datetime(result[0]['Date']).dt.year), {2020, 2021})
                                buys = result[1].loc[result[1]['Type'].str.startswith('買進')]
                                self.assertEqual(set(pd.to_datetime(buys['Date']).dt.year), {2020, 2021})
                                self.assertTrue(result[1]['Type'].str.contains('半倉停利').any())
                                self.assertTrue(result[1]['Type'].str.contains('停損').any())
                                self.assertTrue(result[1]['Type'].str.contains('期末強制結算').any())
                                self.assertTrue(all(positive.get((row['ticker'], str(row['signal_date'])[:10])) == 1.
                                                    for row in executions))
                            else:
                                self.assertEqual(int(result[4]), 0)


if __name__ == '__main__':
    main()
