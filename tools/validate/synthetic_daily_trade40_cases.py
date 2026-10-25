"""New Daily-universal counterfactual MinTrade40 target primitive and PIT tests.

Synthetic-only; does not run formal Windows suite, optimizer, real training or Replay.
"""
from __future__ import annotations

import json
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

from core.strategy_params import V16StrategyParams
from core.breakout_quality_registry import get_breakout_quality_experiment_profile
from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
from core.price_utils import adjust_long_buy_limit_array
from filters.breakout_quality.daily_trade40_target import (
    TARGET_ID, HORIZON_SESSIONS, ExecutableMinPeriod,
    load_historical_min_execution_schedule,
    resolve_historical_min_period,
    compute_counterfactual_trade40_batch,
    collect_canonical_exchange_sessions,
)
from filters.breakout_quality.ranking_score_store import normalize_portfolio_score_scale
from filters.breakout_quality.trade_path_label import (
    build_signal_cache, simulate_realized_trade_path_label,
)

PROFILE = 'daily_universal_counterfactual_mintrade40_netr_full_list_ndcg_pairwise'


def fixture(*, count=130, rising=True):
    dates = pd.bdate_range('2023-01-02', periods=count)
    price = 100.0 + (0.22 if rising else -0.15) * np.arange(count)
    frame = pd.DataFrame({
        'Open': price, 'High': price + 2, 'Low': price - 2,
        'Close': price, 'Volume': np.full(count, 1_000_000.),
    }, index=dates)
    frame.attrs['ticker'] = '2330'
    params = V16StrategyParams(
        high_len=10, atr_len=5, use_breakout_ema_filter=False, use_bb=False,
        use_kc=False, use_vol=False, use_breakout_return_filter=False,
        use_breakout_false_filter=False, use_history_threshold=False,
        use_breakout_quality_filter=False, use_breakout_quality_ranking=False,
        atr_buy_tol=1.5,
    )
    period = ExecutableMinPeriod(dates[0], None, params, 'synthetic', 'param_sha')
    return dates, frame, params, period


class DailyTrade40SyntheticCases(unittest.TestCase):
    def test_research_recipe_reuses_cj_ck_model_without_legacy_changes(self):
        profile = get_breakout_quality_experiment_profile(PROFILE)
        recipe = get_continuous_ranker_execution_recipe(PROFILE)
        self.assertEqual(recipe.target_policy.materialization_mode, 'daily_trade_path_target')
        self.assertEqual(recipe.target_policy.horizon_bars, HORIZON_SESSIONS)
        self.assertEqual(profile.continuous_target_id, TARGET_ID)
        self.assertEqual(profile.model_architecture, 'inception_time_v1')
        self.assertEqual(profile.portfolio_score_scale, 'daily_percentile')
        for legacy in ('daily_universal_terminal_mae_gated_t40_full_list_ndcg_pairwise',
                       'daily_universal_full_horizon_mae_gated_mfe_t40_full_list_ndcg_pairwise'):
            historical = get_breakout_quality_experiment_profile(legacy)
            self.assertEqual(historical.portfolio_score_scale, 'raw')
            self.assertNotIn('portfolio_score_scale', historical.as_manifest_payload())

    def test_hypothetical_entry_creates_completed_net_r_without_real_breakout(self):
        days, frame, params, period = fixture()
        local_pos = np.array([48, 49, 50], dtype=int)
        batch = compute_counterfactual_trade40_batch(
            frame, local_pos, ticker='2330', schedule=(period,), exchange_calendar=days,
        )
        self.assertTrue(batch.valid_mask.all())
        self.assertTrue(np.isfinite(batch.target_raw_r).all())
        self.assertTrue((batch.target_raw_r > 0).all())
        self.assertEqual(batch.maturity_dates[0], days[88].to_datetime64().astype('datetime64[D]'))
        # A non-setup origin is correctly NOT executable under the original rule.
        atr, buy, sell, limits = build_signal_cache(frame, params, ticker='2330')
        hypothetical = next((p for p in local_pos if not buy[p]), None)
        self.assertIsNotNone(hypothetical)
        original = simulate_realized_trade_path_label(
            frame, ticker='2330', signal_pos=int(hypothetical), params=params,
            teacher_effective_date='2023-01-02', precomputed_signals=(atr,buy,sell,limits),
            horizon_end_date=days[int(hypothetical)+40],
        )
        self.assertIsNone(original.realized_net_r)
        self.assertEqual(original.reason, 'not_a2_formal_setup')

    def test_real_setup_override_with_same_limit_is_ledger_equivalent(self):
        # No new execution engine: the optional counterfactual initial intent
        # leaves the canonical Broker-Cash trade accounting untouched.
        from tools.validate.synthetic_min_trade40_target_cases import fixture as event_fixture
        params, frame, signals = event_fixture()
        original = simulate_realized_trade_path_label(
            frame, ticker='2330', signal_pos=2, params=params,
            teacher_effective_date='2020-01-01', precomputed_signals=signals,
            horizon_end_date=frame.index[5],
        )
        counterfactual = simulate_realized_trade_path_label(
            frame, ticker='2330', signal_pos=2, params=params,
            teacher_effective_date='2020-01-01', precomputed_signals=signals,
            horizon_end_date=frame.index[5], counterfactual_entry_limit=101.0,
        )
        self.assertEqual(original.realized_net_r, counterfactual.realized_net_r)
        self.assertEqual(original.realized_net_pnl, counterfactual.realized_net_pnl)
        self.assertEqual(original.exit_date, counterfactual.exit_date)

    def test_stop_loss_and_forced_40th_session_both_use_ledger_net_r(self):
        dates, rising_frame, params, period = fixture()
        pos = 48
        cache = build_signal_cache(rising_frame, params, ticker='2330')
        atr = float(cache[0][pos])
        limit = float(adjust_long_buy_limit_array(np.asarray([
            float(rising_frame['Close'].iloc[pos]) + atr * params.atr_buy_tol,
        ]), ticker='2330')[0])
        expiry = dates[pos + HORIZON_SESSIONS]
        # A rising, steadily tradable path closes on the exact 40th exchange day.
        forced = simulate_realized_trade_path_label(
            rising_frame, ticker='2330', signal_pos=pos, params=params,
            teacher_effective_date='2023-01-02', precomputed_signals=cache,
            horizon_end_date=expiry, counterfactual_entry_limit=limit,
        )
        self.assertTrue(forced.forced_closeout)
        self.assertEqual(forced.exit_reason, 'HORIZON_CLOSEOUT')
        self.assertEqual(forced.exit_date, expiry.strftime('%Y-%m-%d'))
        self.assertGreater(forced.realized_net_r, 0.)
        # Trigger a *legally executable* stop; it must precede expiry and be
        # negative, rather than being overwritten by terminal return.
        falling = rising_frame.copy()
        for column in ('Open','High','Low','Close'):
            falling.loc[dates[50]:, column] -= 8.0
        sell_cache = build_signal_cache(falling, params, ticker='2330')
        stopped = simulate_realized_trade_path_label(
            falling, ticker='2330', signal_pos=pos, params=params,
            teacher_effective_date='2023-01-02', precomputed_signals=sell_cache,
            horizon_end_date=expiry, counterfactual_entry_limit=limit,
        )
        batch = compute_counterfactual_trade40_batch(
            falling, np.asarray([pos]), ticker='2330',
            schedule=(period,), exchange_calendar=dates,
        )
        self.assertEqual(stopped.exit_reason, 'STOP')
        self.assertLess(pd.Timestamp(stopped.exit_date), expiry)
        self.assertLess(stopped.realized_net_r, 0.)
        self.assertTrue(batch.valid_mask[0])
        self.assertAlmostEqual(float(batch.target_raw_r[0]), float(stopped.realized_net_r), places=6)

    def test_missing_params_and_immature_horizon_censor_not_zero(self):
        days, frame, params, period = fixture()
        first = compute_counterfactual_trade40_batch(
            frame, np.array([40, 115]), ticker='2330',
            schedule=(ExecutableMinPeriod(days[60], None, params, 'test', 'param_sha'),),
            exchange_calendar=days,
        )
        self.assertFalse(first.valid_mask.any())
        self.assertTrue(np.isnan(first.target_raw_r).all())
        self.assertEqual(first.reasons[0], 'no_historical_effective_Min_param')
        self.assertEqual(first.reasons[1], 'unmatured_or_missing_exchange_session')
        self.assertTrue(np.isnat(first.maturity_dates[1]))

    def test_calendar_missing_day_does_not_shorten_40_exchange_sessions(self):
        days, frame, params, period = fixture()
        missing = days[69]
        # Stock is suspended on one of the exchange sessions; deadline remains
        # the 40th MARKET day, and no fictional trade may occur in suspension.
        suspended = frame.drop(index=missing)
        batch = compute_counterfactual_trade40_batch(
            suspended, np.array([48]), ticker='2330',
            schedule=(period,), exchange_calendar=days,
        )
        self.assertEqual(batch.maturity_dates[0], days[88].to_datetime64().astype('datetime64[D]'))

    def test_missing_horizon_trade_or_broker_unfilled_is_censored(self):
        days, frame, params, period = fixture()
        # Artificial no-volume sessions after signal cannot manufacture a fill.
        frame.loc[days[48:90], 'Volume'] = 0.0
        batch = compute_counterfactual_trade40_batch(
            frame, np.array([47]), ticker='2330', schedule=(period,), exchange_calendar=days,
        )
        self.assertFalse(batch.valid_mask[0])
        self.assertTrue(np.isnan(batch.target_raw_r[0]))
        self.assertNotEqual(batch.reasons[0], 'filled_executed_net_R')

    def test_missing_closeout_session_censors_open_position(self):
        days, frame, params, period = fixture()
        origin = 48
        # The exchange was open on day 40, but this stock has no executable
        # quote; a close on the previous session would be false execution.
        suspended = frame.drop(index=days[origin + HORIZON_SESSIONS])
        result = compute_counterfactual_trade40_batch(
            suspended, np.asarray([origin]), ticker='2330',
            schedule=(period,), exchange_calendar=days,
        )
        self.assertFalse(result.valid_mask[0])
        self.assertTrue(np.isnan(result.target_raw_r[0]))
        self.assertEqual(result.reasons[0], 'horizon_sell_not_executable')

    def test_real_inceptiontime_delta_ndcg_optimizer_step_on_synthetic_net_r(self):
        try:
            import torch
        except ImportError:
            self.skipTest('No PyTorch in this local validation environment')
        from filters.breakout_quality.models.active import build_active_model
        from filters.breakout_quality.continuous_ranker_data import build_same_date_percentile_targets
        from services.breakout_quality.train_continuous_ranker import _pairwise_logistic_loss
        from core.breakout_quality_runtime import (
            CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        )
        raw = []
        for rising in (True, True, False, False):
            days, frame, params, period = fixture(rising=rising)
            result = compute_counterfactual_trade40_batch(
                frame, np.asarray([48]), ticker='2330',
                schedule=(period,), exchange_calendar=days,
            )
            self.assertTrue(result.valid_mask[0])
            raw.append(float(result.target_raw_r[0]))
        target = build_same_date_percentile_targets(
            np.asarray(raw), np.ones(len(raw), dtype=bool),
            pd.Series([days[48]] * len(raw)),
        )
        model = build_active_model(10, 0, architecture='inception_time_v1')
        model.train()
        generator = torch.Generator().manual_seed(123)
        logits = model(torch.randn(4, 300, 10, generator=generator), None)
        margin = logits[:, 1] - logits[:, 0]
        loss, pair_count = _pairwise_logistic_loss(
            torch, margin, torch.as_tensor(target), [days[48]] * len(raw),
            reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        )
        self.assertGreater(pair_count, 0)
        self.assertTrue(torch.isfinite(loss).item())
        first_weight = next(model.parameters())
        before = first_weight.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, first_weight.detach()))

    def test_historical_effective_params_are_not_backfilled(self):
        days, frame, params, period = fixture()
        self.assertIsNone(resolve_historical_min_period((period,), days[0] - pd.Timedelta(days=3)))
        self.assertIs(resolve_historical_min_period((period,), days[60]), period)
        expiring = ExecutableMinPeriod(days[20], days[40], params, 'a', 'a')
        self.assertIsNone(resolve_historical_min_period((expiring,), days[60]))

    def test_historical_source_json_requires_both_formal_param_sources(self):
        from core.strategy_compare_policy import get_strategy_comparison_settings
        from core.params_io import params_to_json_dict
        days, frame, params, period = fixture()
        settings = get_strategy_comparison_settings('selection_pit')
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for sid, year in [('selection_min_roos', '2019'), ('min_best_broker_cash_rolling_experiment', '2021')]:
                source = settings.parameter_sources[sid]
                path = root / source.path_template.format(param_filename='roos_base_best.json')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({'schema_type': 'rolling_oos_param_set', 'params_by_oos_year': {year: params_to_json_dict(params)}}))
            schedule, lineage = load_historical_min_execution_schedule(root, settings=settings)
            self.assertEqual(len(schedule), 2)
            self.assertIsNotNone(resolve_historical_min_period(schedule, '2019-06-15'))
            self.assertIsNone(resolve_historical_min_period(schedule, '2020-06-15'))
            self.assertIsNotNone(resolve_historical_min_period(schedule, '2021-06-15'))
            self.assertEqual(len(lineage['sources']), 2)
            self.assertEqual(resolve_historical_min_period(schedule, '2021-06-15').source_id, 'min_best_broker_cash_rolling_experiment')
            self.assertEqual(resolve_historical_min_period(schedule, '2019-06-15').source_id, 'selection_min_roos')
            (root / settings.parameter_sources['min_best_broker_cash_rolling_experiment'].path_template.format(param_filename='roos_base_best.json')).unlink()
            with self.assertRaises(FileNotFoundError):
                load_historical_min_execution_schedule(root, settings=settings)

    def test_formal_pit_and_oos_score_loaders_respect_new_only_scale(self):
        from filters.breakout_quality.ranking_score_store import (
            load_selection_point_in_time_score_table_from_path,
            load_continuous_ranker_oos_score_table_from_path,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = pd.DataFrame({
                'ticker': ['A','B','C'], 'date': ['2021-01-04']*3,
                'group_index': [0,1,2], 'breakout_quality_score': [.1,.9,.5],
                'fold_id': ['fold_0']*3,
                'model_information_cutoff': ['2020-12-31']*3,
            })
            pit_path = root/'selection_point_in_time_scores.csv'
            rows.to_csv(pit_path,index=False)
            pit = load_selection_point_in_time_score_table_from_path(
                str(pit_path), experiment_profile=PROFILE,
            )
            self.assertAlmostEqual(pit.loc[('A','2021-01-04'),'breakout_quality_score'], 0.)
            self.assertAlmostEqual(pit.loc[('B','2021-01-04'),'breakout_quality_score'], 1.)
            historical = load_selection_point_in_time_score_table_from_path(
                str(pit_path), experiment_profile='daily_universal_terminal_mae_gated_t40_full_list_ndcg_pairwise',
            )
            self.assertAlmostEqual(historical.loc[('A','2021-01-04'),'breakout_quality_score'], .1)
            oos_path = root/'daily_ranker_oos_scores.csv'
            rows[['ticker','date','group_index']].assign(model_score=[.1,.9,.5]).to_csv(oos_path,index=False)
            oos = load_continuous_ranker_oos_score_table_from_path(str(oos_path),PROFILE)
            self.assertAlmostEqual(oos.loc[('A','2021-01-04'),'model_score'], 0.)
            self.assertAlmostEqual(oos.loc[('B','2021-01-04'),'model_score'], 1.)

    def test_real_daily_data_bundle_pit_split_and_censored_score_coverage(self):
        import filters.breakout_quality.daily_ranker_data as loader
        count = 850
        dates = pd.bdate_range('2019-01-02', periods=count)
        def sample(slope):
            price = 100. + slope * np.arange(count)
            return pd.DataFrame({
                'Open':price, 'High':price+2., 'Low':price-2.,
                'Close':price, 'Volume':np.full(count, 1_000_000.),
            }, index=dates)
        frames = {'0050':sample(.07), '2330':sample(.22), '2317':sample(-.04)}
        params = fixture()[2]
        dataset_summary = {
            'dataset':'full',
            'source_data_date_range':{'start':str(dates[0].date()),
                                      'end':str(dates[-1].date())},
            'policy':{},'dataset_artifacts':{},
        }
        outer = {
            'effective_oos_end_date':str(dates[-1].date()),
            'selection_end_date':str(dates[730].date()),
            'oos_start_date':str(dates[731].date()),
            'oos_end_date':str(dates[-1].date()),
        }
        model_dir = ROOT
        with patch.object(loader,'load_validated_dataset_bundle',
                          return_value=(dataset_summary,None,None,None,pd.DataFrame())), \
             patch.object(loader,'resolve_breakout_quality_outer_policy',return_value=outer), \
             patch.object(loader,'discover_dataset_csv_inputs',
                          return_value=([(t,Path(t)) for t in frames],None)), \
             patch.object(loader,'load_dataset_frame',
                          side_effect=lambda path,ticker,**kwargs:frames[ticker]), \
             patch.object(loader,'load_historical_min_execution_schedule',
                          return_value=((ExecutableMinPeriod(dates[0],None,params,'test','SHA'),),
                                        {'policy':'test','sources':[]})), \
             patch.object(loader,'collect_canonical_exchange_sessions',return_value=dates):
            bundle = loader.load_daily_universal_ranker_data(
                filter_id='breakout_quality_v1',
                model_architecture='inception_time_v1',
                experiment_profile=PROFILE, preload_feature_bank=False,
                allow_stale_source=True, project_root=model_dir,
            )
        self.assertGreater(int(bundle.target_valid.sum()),500)
        self.assertGreater(int((~bundle.target_valid).sum()),0)
        self.assertTrue(np.isnan(bundle.raw_target[~bundle.target_valid]).all())
        self.assertEqual(bundle.target_manifest['target_contract']['horizon_sessions'],40)
        valid_rows = bundle.group_table.loc[bundle.target_valid]
        # 40 *exchange* sessions, not a ticker-specific or benchmark-only count.
        first = valid_rows.iloc[0]
        signal_session = dates.get_loc(first['date'])
        self.assertEqual(pd.Timestamp(first['label_eval_end_date']),dates[signal_session+40])
        split=loader.build_daily_ranker_split(bundle,inner_validation_months=3)
        self.assertGreater(len(split.inner_train_ids),20)
        self.assertGreater(len(split.validation_ids),20)
        self.assertGreater(len(split.oos_ids),20)
        self.assertTrue((pd.to_datetime(bundle.group_table.iloc[split.inner_train_ids]['label_eval_end_date']) <
                         pd.Timestamp(split.report['inner_validation_start_date'])).all())

    def test_percentile_score_on_all_scored_universe_not_min_candidates(self):
        data = pd.DataFrame({'ticker':['A','B','C','A','B'],
                             'date':['2021-01-01']*3 + ['2021-01-04']*2,
                             'model_score':[100.,120.,130.,-100.,-200.]})
        scaled = normalize_portfolio_score_scale(
            data, experiment_profile=PROFILE, score_column='model_score')
        np.testing.assert_allclose(scaled['model_score'], [0,0.5,1,1,0])
        original = normalize_portfolio_score_scale(
            data, experiment_profile='daily_universal_terminal_mae_gated_t40_full_list_ndcg_pairwise',
            score_column='model_score')
        np.testing.assert_allclose(original['model_score'], data['model_score'])
        self.assertEqual(scaled['ticker'].tolist(), data['ticker'].tolist())
        from filters.breakout_quality.mfe_safety_geometry import ensure_daily_percentile
        same, _ = ensure_daily_percentile(
            pd.DataFrame({'date': data['date'], 'value': data['model_score']}),
            method='average_zero_based',
        )
        np.testing.assert_allclose(scaled['model_score'], same['percentile'])
        with self.assertRaises(ValueError):
            normalize_portfolio_score_scale(
                data.assign(model_score=[1., float('nan'), 2., 3., 4.]),
                experiment_profile=PROFILE, score_column='model_score',
            )


if __name__ == '__main__':
    unittest.main(verbosity=2)
