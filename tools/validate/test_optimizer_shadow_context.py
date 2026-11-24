"""Exact state parity for shared counterfactual sizing inputs and fill geometry."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import inspect
import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from core import entry_plans, extended_signals, portfolio_entries, price_utils
from core.exact_accounting import infer_security_profile
from core.portfolio_fast_data import pack_prepared_stock_data
from core.strategy_params import V16StrategyParams


STOP_REUSE = '''        # The usual target reference is the same fill used for the initial stop.
        # Reuse that exact result; explicit distinct references keep their owner.
        target_reference_stop = (
            actual_init_sl if resolved_target_reference is buy_price
            else calc_initial_stop_from_reference(resolved_target_reference, entry_atr, params, ticker=ticker, security_profile=security_profile)
        )'''
OLD_STOP = '        target_reference_stop = calc_initial_stop_from_reference(resolved_target_reference, entry_atr, params, ticker=ticker, security_profile=security_profile)'
PREDECESSOR_LEVELS_SHA = '4c1b8a46d3b16792fe19b9f9d9dca05015db15874c2e595e9b246051b807e912'


def predecessor_fill_levels():
    """Recover and pin the pre-optimization function, not a second formula."""
    source = inspect.getsource(entry_plans._resolve_entry_fill_levels).strip()
    if source.count(STOP_REUSE) != 1:
        raise AssertionError('fill-level optimization no longer matches its reference')
    source = source.replace(STOP_REUSE, OLD_STOP)
    if sha256(source.encode()).hexdigest() != PREDECESSOR_LEVELS_SHA:
        raise AssertionError('fill-level predecessor source pin changed')
    namespace = dict(vars(entry_plans))
    exec(compile(source, '<pinned-predecessor-fill-levels>', 'exec'), namespace)
    return namespace['_resolve_entry_fill_levels']


def exact_state(value):
    """Preserve float bits, NaNs, container shape and every position field."""
    if isinstance(value, float):
        return ('float', value.hex())
    if isinstance(value, np.generic):
        return (str(value.dtype), value.tobytes())
    if isinstance(value, dict):
        return {key: exact_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, [exact_state(item) for item in value])
    if isinstance(value, V16StrategyParams):
        return ('params', exact_state(vars(value)))
    return value


class OptimizerShadowContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference_levels = staticmethod(predecessor_fill_levels())

    def setUp(self):
        self.params = V16StrategyParams(
            atr_times_init=2.0, atr_times_trail=1.5,
            use_breakout_quality_filter=False, use_breakout_quality_ranking=False,
        )

    def assert_exact(self, actual, expected):
        self.assertEqual(exact_state(actual), exact_state(expected))

    def test_full_position_matches_pinned_levels_at_tick_boundaries(self):
        for ticker in ('2330', '0050', '00679B'):
            profile = infer_security_profile(ticker)
            for price in (0.999, 1.0, 9.99, 10.0, 49.99, 50.0, 99.99, 100.0, 499.9, 500., 999., 1000.):
                for reference in (None, float('nan'), price, price * 1.03):
                    kwargs = dict(
                        buy_price=price, qty=999, entry_atr=price * 0.03,
                        params=self.params, limit_price=price,
                        ticker=ticker, security_profile=profile,
                        trade_date='2026-12-31', target_reference_price=reference,
                    )
                    with patch.object(entry_plans, '_resolve_entry_fill_levels', self.reference_levels):
                        expected = entry_plans.build_position_from_entry_fill(**kwargs)
                    self.assert_exact(entry_plans.build_position_from_entry_fill(**kwargs), expected)

    def test_contexts_preserve_all_anchor_fields_across_capital_params_and_tax_dates(self):
        cache = {}
        other = replace(self.params, fixed_risk=self.params.fixed_risk * 0.5,
                        max_position_cap_pct=0.4)
        for params in (self.params, other):
            for capital in (0.0, 1000.0, 50000.0, 1000000.0):
                for ticker in ('2330', '0050', '00679B'):
                    for day in ('2026-12-31', '2027-01-01'):
                        for low, high in ((98.0, 101.0), (85.0, 101.0), (98.0, 120.0)):
                            seed = extended_signals.create_signal_tracking_state(
                                100.0, 3.0, params, ticker=ticker,
                                security_profile=infer_security_profile(ticker),
                                signal_date='2026-12-30',
                            )
                            kwargs = dict(t_low=low, t_high=high, t_open=99.0,
                                          sizing_capital=capital, current_date=day,
                                          params=params, copy_shadow_position=False)
                            expected = deepcopy(seed)
                            with patch.object(entry_plans, '_resolve_entry_fill_levels', self.reference_levels):
                                clear_expected = extended_signals.should_clear_extended_signal(expected, **kwargs)
                            actual = deepcopy(seed)
                            clear_actual = extended_signals.should_clear_extended_signal(
                                actual, **kwargs, sizing_context_cache=cache,
                            )
                            self.assertEqual(clear_actual, clear_expected)
                            self.assert_exact(actual, expected)

    def test_cleanup_keeps_terminal_states_and_missing_candidate_rows_equivalent(self):
        dates = pd.to_datetime(['2026-11-19', '2026-11-20'])
        all_fast, seeds = {}, {}
        for i in range(24):
            ticker = str(2300 + i)
            profile = infer_security_profile(ticker)
            low, high = ((98.0, 101.0), (85.0, 101.0), (98.0, 120.0))[i % 3]
            frame = pd.DataFrame({
                'Open': [100., 99.], 'High': [101., high], 'Low': [99., low],
                'Close': [100., 100.], 'Volume': [10000., 10000.],
                'ATR': [3., 3.], 'buy_limit': [100., 100.],
                'is_setup': [False, False], 'ind_sell_signal': [False, False],
            }, index=dates)
            frame.attrs['security_profile'] = profile
            all_fast[ticker] = pack_prepared_stock_data(frame)
            seeds[ticker] = extended_signals.create_signal_tracking_state(
                100., 3., self.params, ticker=ticker,
                security_profile=profile, signal_date=dates[0],
            )
        # A held ticker is removed and a missing market series remains untouched.
        seeds['MISSING'] = deepcopy(next(iter(seeds.values())))
        common = dict(portfolio={'2300': {}}, all_dfs_fast=all_fast,
                      today=dates[1], params=self.params, sizing_capital=1000000.)
        expected = deepcopy(seeds)
        with patch.object(entry_plans, '_resolve_entry_fill_levels', self.reference_levels):
            portfolio_entries.cleanup_extended_signals_for_day(expected, **common)
        actual = deepcopy(seeds)
        with ExitStack() as stack:
            stack.enter_context(patch.object(portfolio_entries, '_NATIVE_LIVE_SHADOW_BATCH_ENABLED', False))
            stack.enter_context(patch.object(portfolio_entries, '_NATIVE_LIVE_SHADOW_CACHED_BATCH_ENABLED', False))
            capital = stack.enter_context(patch.object(extended_signals, 'build_position_sizing_capital_context',
                                                       wraps=price_utils.build_position_sizing_capital_context))
            fee = stack.enter_context(patch.object(price_utils, 'build_position_sizing_fee_context',
                                                   wraps=price_utils.build_position_sizing_fee_context))
            portfolio_entries.cleanup_extended_signals_for_day(actual, **common, training_fast_path=True)
        self.assert_exact(actual, expected)
        self.assertEqual(capital.call_count, 1)
        self.assertEqual(fee.call_count, 1)

    def test_missing_geometry_does_not_prepare_sizing_contexts(self):
        for limit, atr, opening in ((np.nan, 3., 99.), (100., np.nan, 99.),
                                    (100., 3., np.nan), (-1., 3., 99.)):
            state = dict(orig_limit=limit, orig_atr=atr, shadow_position=None)
            with patch.object(extended_signals, 'build_position_sizing_capital_context',
                              side_effect=AssertionError('invalid geometry must not size')):
                extended_signals.ensure_extended_signal_counterfactual_anchor(
                    state, t_open=opening, t_low=98., params=self.params,
                    sizing_capital=1000000., sizing_context_cache={},
                )

    def test_reused_stop_does_not_suppress_distinct_reference(self):
        kwargs = dict(buy_price=100., entry_atr=3., init_sl=None, init_trail=None,
                      target_price=None, limit_price=100., params=self.params)
        original = entry_plans.calc_initial_stop_from_reference
        for reference, calls in ((None, 1), (kwargs['buy_price'], 1),
                                 (float('nan'), 1), (float('100'), 2), (99., 2)):
            with patch.object(entry_plans, 'calc_initial_stop_from_reference', wraps=original) as stop:
                result = entry_plans._resolve_entry_fill_levels(**kwargs, target_reference_price=reference)
            self.assertEqual(stop.call_count, calls)
            self.assertTrue(all(math.isfinite(value) for value in result))

    def test_equal_prices_with_different_scalar_dtypes_keep_original_arithmetic(self):
        for buy in (100.0, np.float32(100.0), np.float64(100.0)):
            for reference in (None, 100.0, np.float32(100.0), np.float64(100.0)):
                kwargs = dict(buy_price=buy, entry_atr=0.05, init_sl=None,
                              init_trail=None, target_price=None, limit_price=100.,
                              params=self.params, target_reference_price=reference)
                self.assert_exact(entry_plans._resolve_entry_fill_levels(**kwargs),
                                  self.reference_levels(**kwargs))


if __name__ == '__main__':
    unittest.main()
