"""Same-trial PIT materialization must preserve signal and ledger observations."""

import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.portfolio_engine import run_portfolio_timeline
from core.portfolio_entry_policy import (
    DAILY_EXECUTION_FULL_FILTERS, daily_signal_execution_context,
)
from core.portfolio_fast_data import (
    build_neutral_candidate_stats_index, prep_optimizer_stock_data_bundle,
    pack_static_market_data,
)
from services.optimizer.trial_inputs import prepare_trial_inputs
from tools.validate.synthetic_daily_trade40_cases import fixture


class OptimizerDeferredSignalReuseTests(unittest.TestCase):
    def test_packed_dates_keep_lookup_semantics_and_share_timestamps(self):
        indexes = (
            pd.date_range('2024-01-01', periods=5),
            pd.date_range('2024-11-03', periods=5, freq='h', tz='America/New_York'),
            pd.DatetimeIndex(['2024-01-01', '2024-01-01', None]),
            pd.Index([1.0, float('nan'), 3.0]),
        )
        for index in indexes:
            with self.subTest(index=str(index.dtype)):
                frame = pd.DataFrame({key: np.ones(len(index)) for key in
                                      ('Open', 'High', 'Low', 'Close', 'Volume')}, index=index)
                old_dates = tuple(index.tolist())
                old_lookup = {value: i for i, value in enumerate(index)}
                actual = pack_static_market_data(frame)
                self.assertEqual(len(old_lookup), len(actual['date_to_pos']))
                self.assertEqual([old_lookup.get(value, -1) for value in old_dates],
                                 [actual['date_to_pos'].get(value, -1) for value in actual['dates']])
                if isinstance(index, pd.DatetimeIndex):
                    for key in actual['date_to_pos']:
                        self.assertTrue(any(key is value for value in actual['dates']))

    def assert_tree_equal(self, left, right):
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right, strict=True)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(left, right)

    def compare_materialization(self, frame, params):
        dynamic, _, _ = prep_optimizer_stock_data_bundle(
            frame, params, include_trade_logs=False, include_pit_stats_index=False,
        )
        saved = {key: value.copy() for key, value in dynamic.items()}
        expected = prep_optimizer_stock_data_bundle(
            frame, params, include_trade_logs=True, include_pit_stats_index=True,
        )
        with patch('core.portfolio_fast_data.generate_signals',
                   side_effect=AssertionError('same-trial signals were regenerated')):
            actual = prep_optimizer_stock_data_bundle(
                frame, params, include_trade_logs=True, include_pit_stats_index=True,
                prepared_dynamic_data=dynamic,
            )
        self.assert_tree_equal(expected, actual)
        self.assert_tree_equal(saved, dynamic)
        return expected

    def test_breakout_pit_and_trade_ledgers_are_exact(self):
        _, frame, original, _ = fixture(count=180)
        observed_trades = 0
        for reentry in (False, True):
            for capital in (15000.0, 1000000.0):
                with self.subTest(reentry=reentry, capital=capital):
                    params = replace(original, initial_capital=capital,
                                     use_breakout_reclaim_reentry=reentry)
                    _, logs, _ = self.compare_materialization(frame, params)
                    observed_trades += len(logs)
        self.assertGreater(observed_trades, 0)

    def test_daily_full_does_not_apply_signal_filter_twice(self):
        _, frame, original, _ = fixture(count=180)
        params = replace(original, use_breakout_buy=False)
        with daily_signal_execution_context(DAILY_EXECUTION_FULL_FILTERS):
            self.compare_materialization(frame, params)

    def test_inline_preparation_retains_failed_ticker_and_pit_outputs(self):
        _, frame, params, _ = fixture(count=180)
        raw = {'2330': frame, 'short': frame.iloc[:1].copy()}
        kwargs = dict(raw_data_cache=raw, params=params, default_max_workers=1,
                      executor_bundle={'executor_kind': 'inline'},
                      include_trade_logs=False, profile_enabled=False)
        initial = prepare_trial_inputs(**kwargs, include_pit_stats_index=False)
        expected = prepare_trial_inputs(**kwargs, include_pit_stats_index=True)
        with patch('core.portfolio_fast_data.generate_signals',
                   side_effect=AssertionError('same-trial signals were regenerated')):
            actual = prepare_trial_inputs(
                **kwargs, include_pit_stats_index=True,
                prepared_fast_cache=initial['all_dfs_fast'],
            )
        self.assert_tree_equal(expected['all_pit_stats_index'], actual['all_pit_stats_index'])
        self.assertEqual(expected['prep_failures'], actual['prep_failures'])
        self.assertEqual(len(actual['prep_failures']), 1)
        self.assertEqual(expected['master_dates'], actual['master_dates'])

    def test_portfolio_replay_keeps_reused_signal_arrays_readonly_and_bit_exact(self):
        _, frame, params, _ = fixture(count=180)
        kwargs = dict(raw_data_cache={'2330': frame}, params=params,
                      default_max_workers=1, executor_bundle={'executor_kind': 'inline'},
                      include_trade_logs=False, profile_enabled=False)
        initial = prepare_trial_inputs(**kwargs, include_pit_stats_index=False)
        packed = initial['all_dfs_fast']['2330']
        fields = ('ATR', 'is_setup', 'ind_sell_signal', 'buy_limit')
        saved = {key: packed[key].tobytes() for key in fields}
        for key in fields:
            packed[key].flags.writeable = False
        run_portfolio_timeline(
            initial['all_dfs_fast'], {}, sorted(initial['master_dates']),
            int(frame.index[0].year), params, 5, False,
            benchmark_ticker='2330', benchmark_data=packed,
            is_training=True, verbose=False,
            pit_stats_index={'2330': build_neutral_candidate_stats_index(params)},
        )
        with patch('core.portfolio_fast_data.generate_signals',
                   side_effect=AssertionError('post-replay signals were regenerated')):
            actual = prepare_trial_inputs(
                **kwargs, include_pit_stats_index=True,
                prepared_fast_cache=initial['all_dfs_fast'],
            )
        expected = prepare_trial_inputs(**kwargs, include_pit_stats_index=True)
        self.assert_tree_equal(actual['all_pit_stats_index'], expected['all_pit_stats_index'])
        self.assertGreater(len(actual['all_pit_stats_index']['2330']['exit_dates']), 0)
        for key in fields:
            self.assertFalse(packed[key].flags.writeable)
            self.assertEqual(packed[key].tobytes(), saved[key])


if __name__ == '__main__':
    unittest.main()
