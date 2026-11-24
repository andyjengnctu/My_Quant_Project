"""OOS preparation reuse keeps the canonical shared universe and replay."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import pickle
import unittest
from unittest.mock import patch

from core.data_utils import get_required_min_rows
from core.params_io import params_to_json_dict
from core.portfolio_fast_data import pack_static_market_data
from services.optimizer import outer_rolling_evaluation as owner
from services.optimizer.trial_inputs import prepare_trial_inputs
from tools.validate.synthetic_daily_trade40_cases import fixture


class OptimizerOOSContextReuseTests(unittest.TestCase):
    def test_finalist_reuse_matches_canonical_ensemble_and_universe(self):
        dates, frame, base, _ = fixture(count=250)
        for column in ('Open', 'High', 'Low', 'Close'):
            frame.iloc[169, frame.columns.get_loc(column)] -= 8.0
            frame.iloc[170:, frame.columns.get_loc(column)] += 30.0
        longer = replace(base, high_len=120)
        low = get_required_min_rows(base)
        high = get_required_min_rows(longer)
        self.assertGreater(high, low)
        raw = {'0050': frame.copy(), '2330': frame, 'SHORT': frame.iloc[:(low+high)//2].copy()}
        with TemporaryDirectory() as temporary:
            for ticker, data in raw.items():
                data.to_csv(Path(temporary)/f'{ticker}.csv', index_label='Date')
            session = SimpleNamespace(
                raw_data_cache=raw, static_fast_cache={key: pack_static_market_data(value)
                                                     for key, value in raw.items()},
                master_dates=set(dates), raw_data_cache_required_min_rows=low,
                raw_data_cache_data_dir=temporary, runtime_cache_identity='test-runtime',
                default_max_workers=1, train_max_positions=2, train_enable_rotation=False,
                fixed_strategy_param_overrides=None,
                get_trial_prep_executor_bundle=lambda _: {'executor_kind': 'inline'},
                get_prepared_trial_inputs_from_cache=lambda _: None,
                cache_prepared_trial_inputs=lambda *_: None,
            )
            members = [{'member_index': i, 'params': params_to_json_dict(params)}
                       for i, params in enumerate((base, longer), 1)]
            kwargs = dict(session=session, item={'params_ensemble': members},
                          policy_name='base_finalists_agree', oos_year=2023,
                          oos_start_date='2023-01-02', oos_end_date='2023-12-31')
            def loader(_directory, required, *_args, **_kwargs):
                return {key: value for key, value in raw.items() if len(value) >= required}
            def inline(*args, **values):
                values['executor_bundle'] = {'executor_kind': 'inline'}
                return prepare_trial_inputs(*args, **values)
            with patch('services.portfolio_replay.load_all_raw_data', side_effect=loader), \
                 patch('services.portfolio_replay.prepare_trial_inputs', side_effect=inline):
                expected = owner._evaluate_finalist_ensemble_oos_metrics(**kwargs)
            cache = {}
            with patch('services.portfolio_replay._build_in_memory_raw_context_source',
                       side_effect=AssertionError('redundant raw universe load')):
                actual = owner._evaluate_finalist_ensemble_oos_metrics(**kwargs, prepared_cache=cache)
            self.assertEqual(pickle.dumps(expected), pickle.dumps(actual))
            self.assertGreater(actual['trade_count'], 0)
            self.assertEqual(len(cache), 2)
            with patch.object(owner, 'prepare_trial_inputs', side_effect=AssertionError('repeated finalist prep')):
                owner._get_or_prepare_oos_inputs(session=session, params=base, prepared_cache=cache)

    def test_neutral_deferred_pit_is_not_admitted(self):
        prep = {'all_pit_stats_index': {'2330': {}}, 'pit_stats_deferred': True}
        self.assertFalse(owner._prep_result_has_pit_index(prep))
        prep['pit_stats_deferred'] = False
        self.assertTrue(owner._prep_result_has_pit_index(prep))


if __name__ == '__main__':
    unittest.main()
