"""Controlled invariants for Daily T40 encoder pretraining and CP gate composition.

This tests registered recipes, PIT maturity, exact T40 SSOT values, sampling
and a one-epoch CPU pretraining primitive; it does not fit a formal research
model or run broker-cash Replay.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.breakout_quality_registry import (
    DAILY_UNIVERSAL_NETPNL_POSITIVE_BINARY_PROFILE,
    DAILY_UNIVERSAL_T40_PRETRAINED_NETPNL_POSITIVE_BINARY_PROFILE,
    get_breakout_quality_experiment_profile,
    get_breakout_quality_encoder_pretraining_profile,
    get_continuous_ranker_research_spec,
)
from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
from core.strategy_compare_policy import get_strategy_comparison_settings
from config.strategy_compare import STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE
from filters.breakout_quality.continuous_target import (
    DAILY_TERMINAL_MAE_GATED_T40_TARGET_ID,
    StrategyAlignedContinuousTargetSpec,
)
from filters.breakout_quality.contract import DEFAULT_LABEL_POLICY
from filters.breakout_quality.daily_ranker_data import (
    LazyDailyFeatureBank,
    compute_daily_opportunity_target_batch,
)
from filters.breakout_quality.encoder_pretraining import (
    build_ticker_balanced_epoch_ids,
    build_ticker_balanced_epoch_schedules,
    pretrain_encoder_state,
)


class T40PositivePretrainingCases(TestCase):
    def test_cp_cq_same_supervised_contract_with_only_pretraining_added(self):
        cp = get_breakout_quality_experiment_profile(DAILY_UNIVERSAL_NETPNL_POSITIVE_BINARY_PROFILE)
        cq = get_breakout_quality_experiment_profile(DAILY_UNIVERSAL_T40_PRETRAINED_NETPNL_POSITIVE_BINARY_PROFILE)
        for key, value in vars(cp).items():
            if key not in {'name', 'encoder_pretraining_profile', 'fit_contract_version'}:
                self.assertEqual(getattr(cq, key), value, key)
        self.assertIsNone(cp.encoder_pretraining_profile)
        recipe = get_continuous_ranker_execution_recipe(cq.name)
        self.assertEqual(recipe.training_policy.target_builder,
                         get_continuous_ranker_execution_recipe(cp.name).training_policy.target_builder)
        self.assertEqual(get_continuous_ranker_research_spec(cq.name).model_research_id, 'MR-13CQ')
        pre = get_breakout_quality_encoder_pretraining_profile(cq.encoder_pretraining_profile)
        self.assertEqual(pre.target_names, ('terminal_mae_gated_t40_40d',))
        self.assertEqual(pre.source_scope, 'matured_daily_universal_asof_fitting_cutoff')
        self.assertEqual(pre.downstream_finetune, 'full_unfrozen')
        self.assertEqual(cq.fit_contract_version, 2)
        self.assertEqual(pre.sampling_mode, 'complete_daily_train_rows_per_epoch')
        self.assertGreater(pre.early_stopping_patience, 0)

    def test_new_strategy_exact_direct_only_swap_positive_source(self):
        settings = get_strategy_comparison_settings(STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE)
        for old, new, expected_mode in (
            ('C125', 'C127', 'resource-aware-continuous-score-constrained-optimal'),
            ('C126', 'C128', 'resource-aware-continuous-score-no-k-no-r0'),
        ):
            baseline, treatment = settings.arms[old], settings.arms[new]
            self.assertEqual(baseline.dl_id, treatment.dl_id)
            self.assertEqual(baseline.dl_runtime_mode, treatment.dl_runtime_mode)
            self.assertEqual(treatment.dl_runtime_mode, expected_mode)
            self.assertEqual(baseline.param_source, treatment.param_source)
            self.assertEqual(baseline.param_policy, treatment.param_policy)
            self.assertEqual(baseline.rule_policy, treatment.rule_policy)
            bo, to = dict(baseline.dl_runtime_options), dict(treatment.dl_runtime_options)
            self.assertEqual(bo.pop('positive_dl_id'), 'CONT13CP_ROLL')
            self.assertEqual(to.pop('positive_dl_id'), 'CONT13CQ_ROLL')
            self.assertEqual(bo, to)
            self.assertEqual(bo['positive_probability_threshold'], 0.5)
        source = settings.dl_sources['CONT13CQ_ROLL']
        self.assertEqual(source.experiment_profile, DAILY_UNIVERSAL_T40_PRETRAINED_NETPNL_POSITIVE_BINARY_PROFILE)
        self.assertTrue({'C127-C125', 'C128-C126'}.issubset({contrast.contrast_id for contrast in settings.enabled_contrasts}))

    def test_cq_score_consumption_uses_independent_pit_positive_lookup(self):
        from core.breakout_quality_ranking_context import breakout_quality_ranking_source_context
        from filters.breakout_quality.ranking_score_store import SCORE_SOURCE_SELECTION_POINT_IN_TIME
        from filters.breakout_quality.runtime import resolve_breakout_quality_candidate_rank
        from core.positive_netpnl_contract import positive_score_admitted

        settings = get_strategy_comparison_settings(STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE)
        arm = settings.arms['C128']
        co_source = settings.dl_sources[arm.dl_id]
        cq_source = settings.dl_sources[arm.dl_runtime_options['positive_dl_id']]
        self.assertEqual(cq_source.experiment_profile,
                         DAILY_UNIVERSAL_T40_PRETRAINED_NETPNL_POSITIVE_BINARY_PROFILE)
        options = {
            **arm.dl_runtime_options,
            'positive_filter_id': cq_source.filter_id,
            'positive_score_source': SCORE_SOURCE_SELECTION_POINT_IN_TIME,
            'positive_model_architecture': cq_source.model_architecture,
            'positive_experiment_profile': cq_source.experiment_profile,
        }
        columns = []
        def fake_lookup(**kwargs):
            column = kwargs.get('score_column')
            columns.append(column)
            return {'score': 0.91 if column is None else 0.63, 'available': True,
                    'score_date': '2024-10-01',
                    'score_source': SCORE_SOURCE_SELECTION_POINT_IN_TIME}
        with breakout_quality_ranking_source_context(
            score_source=SCORE_SOURCE_SELECTION_POINT_IN_TIME,
            model_architecture=co_source.model_architecture,
            experiment_profile=co_source.experiment_profile,
            ranking_policy=arm.dl_runtime_mode,
            ranking_options=options,
        ), patch('filters.breakout_quality.runtime.lookup_selection_point_in_time_candidate_score',
                  side_effect=fake_lookup):
            rank = resolve_breakout_quality_candidate_rank(
                ticker='2330', signal_date='2024-10-01',
                information_date='2024-10-01', high_len=40)
        self.assertEqual(columns, [None, 'positive_p_positive'])
        self.assertAlmostEqual(rank['score'], 0.91)
        self.assertAlmostEqual(rank['positive_probability'], 0.63)
        self.assertTrue(positive_score_admitted({
            'breakout_quality_rank': rank,
            'breakout_quality_ranking_options': options,
        }))

    def test_t40_price_provider_identical_to_canonical_target_and_pit_cutoff(self):
        dates = pd.date_range('2020-01-01', periods=90, freq='D')
        bars = np.arange(90, dtype=np.float64)
        series = (
            100 + bars * 0.16,
            100 - bars * 0.45,
            90 + bars * 0.04 + 4 * np.sin(bars / 3),
        )
        frames = []
        for close in series:
            frames.append(pd.DataFrame({
                'Open': close,
                'High': close * 1.013,
                'Low': close * 0.98,
                'Close': close,
                'Volume': np.full(90, 1000.),
            }, index=dates))
        positions = np.asarray([3, 5, 12, 21, 34], dtype=np.int32)
        ticker_ids = np.repeat(np.arange(3, dtype=np.int32), len(positions))
        source_positions = np.tile(positions, 3)
        bank = LazyDailyFeatureBank(
            frames=tuple(frames), benchmark=frames[0],
            ticker_ids=ticker_ids, source_positions=source_positions,
            benchmark_positions=source_positions,
            policy=SimpleNamespace(feature_window_bars=3),
        )
        all_ids = np.arange(len(source_positions), dtype=np.int64)
        self.assertTrue(bank.future_path_maturity_mask(all_ids, horizon_bars=40,
                                                       cutoff_date=dates[74]).all())
        self.assertFalse(bank.future_path_maturity_mask(all_ids, horizon_bars=40,
                                                        cutoff_date=dates[50]).all())
        with self.assertRaises(ValueError):
            bank.future_path_statistics(all_ids, cutoff_date=dates[50],
                target_names=('terminal_mae_gated_t40_40d',), horizon_bars=40)
        out, names = bank.future_path_statistics(all_ids, cutoff_date=dates[74],
            target_names=('terminal_mae_gated_t40_40d',), horizon_bars=40)
        self.assertEqual(names, ('terminal_mae_gated_t40_40d',))
        spec = StrategyAlignedContinuousTargetSpec.from_label_policy(DEFAULT_LABEL_POLICY)
        for ticker_id, frame in enumerate(frames):
            canonical = compute_daily_opportunity_target_batch(
                frame, positions, spec=spec, target_id=DAILY_TERMINAL_MAE_GATED_T40_TARGET_ID)
            self.assertTrue(canonical.valid_mask.all())
            start, stop = ticker_id * len(positions), (ticker_id + 1) * len(positions)
            np.testing.assert_allclose(out[start:stop, 0], canonical.target_raw_r,
                atol=1e-5, rtol=1e-5)
        self.assertTrue(bool(np.isclose(out[len(positions), 0], -1.0)))

    def test_ticker_balanced_sampler_preserves_old_draws_and_order(self):
        table = pd.DataFrame({'ticker': ['Z', 'A', 'B', 'A', 'Z', 'B', 'A', 'B', 'Z']})
        ids = np.asarray([5, 0, 2, 7, 1, 4, 3, 8, 6], dtype=np.int64)
        schedules, names = build_ticker_balanced_epoch_schedules(table, ids, seed=693545351, epochs=8)
        self.assertEqual(names, ('A', 'B', 'Z'))
        for epoch, actual in enumerate(schedules, 1):
            expected, expected_names = build_ticker_balanced_epoch_ids(table, ids, seed=693545351, epoch=epoch)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(names, expected_names)

    def test_real_encoder_training_uses_mature_daily_rows_not_sparse_cp_events(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch not installed')
        # The model is trained for one synthetic epoch only. No research fitting.
        original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            group_dates = pd.date_range('2020-01-01', periods=180, freq='D')
            table = pd.DataFrame({
                'date': list(group_dates) * 2,
                'ticker': ['A'] * 180 + ['B'] * 180,
            })
            fake_features = np.zeros((360, 300, 10), dtype=np.float32)
            for j in range(360):
                fake_features[j, :, 3] = (j % 13) * 0.0005
            class ControlledBank:
                shape = fake_features.shape
                def __getitem__(self, ids):
                    return fake_features[ids]
                def future_path_maturity_mask(self, ids, *, horizon_bars, cutoff_date):
                    self_last = pd.Timestamp(cutoff_date)
                    self_rows = table.iloc[ids].date
                    assert horizon_bars == 40
                    return (self_rows + pd.Timedelta(days=40) <= self_last).to_numpy(dtype=bool)
                def future_path_statistics(self, ids, *, cutoff_date, target_names, horizon_bars):
                    assert len(ids) and horizon_bars == 40
                    self_mask = self.future_path_maturity_mask(ids,
                        horizon_bars=horizon_bars, cutoff_date=cutoff_date)
                    assert self_mask.all()
                    return (np.sin(np.asarray(ids, dtype=np.float32) * 0.07)[:, None], target_names)
            bank = ControlledBank()
            profile = get_breakout_quality_encoder_pretraining_profile(
                get_breakout_quality_experiment_profile(
                    DAILY_UNIVERSAL_T40_PRETRAINED_NETPNL_POSITIVE_BINARY_PROFILE
                ).encoder_pretraining_profile
            )
            plan = SimpleNamespace(device=torch.device('cpu'), device_type='cpu',
                mixed_precision_enabled=False)
            # Two original CP-event rows from different dates and tickers.
            sparse_ids = np.asarray([170, 350], dtype=np.int64)
            with patch('filters.breakout_quality.encoder_pretraining.get_breakout_quality_encoder_pretraining_profile',
                       return_value=replace(profile, epochs=2, validation_months=1)):
                state, audit = pretrain_encoder_state(
                    torch, feature_bank=bank, group_table=table, fit_ids=sparse_ids,
                    feature_count=10, context_count=0,
                    model_architecture='inception_time_v1',
                    pretraining_profile_name=profile.name,
                    seed=42, plan=plan, stage='inner_train_epoch_selection')
            self.assertTrue(state)
            self.assertEqual(audit['fitting_group_count'], 2)
            self.assertGreater(audit['candidate_pretraining_group_count'], 100)
            self.assertGreater(audit['matured_fitting_group_count'], 40)
            self.assertEqual(audit['samples_per_epoch'], audit['matured_fitting_group_count'])
            self.assertGreater(audit['pretrain_inner_train_groups'], 2)
            self.assertGreater(audit['pretrain_validation_groups'], 2)
            self.assertGreater(audit['pretrain_best_epoch'], 0)
            self.assertTrue(audit['pretraining_temporal_validation_used'])
            self.assertTrue(audit['downstream_validation_used'] is False)
            self.assertLessEqual(audit['future_path_maturity_cutoff'], '2020-06-19')
            # Test the actual downstream initialization handoff, not just the
            # temporary pretraining model: the binary head remains freshly built.
            from services.breakout_quality.train_continuous_ranker import _new_model_and_optimizer
            model, optimizer = _new_model_and_optimizer(
                torch, feature_count=10, context_count=0,
                args=SimpleNamespace(seed=42, model_architecture='inception_time_v1',
                    lr=0.001, weight_decay=0.0001),
                plan=plan, pretrained_encoder_state=state)
            loaded_state = model.export_encoder_state_dict()
            self.assertTrue(all(torch.equal(state[key], loaded_state[key]) for key in state))
            self.assertEqual(model.classifier.out_features, 2)
            del optimizer, model
        finally:
            torch.set_num_threads(original_threads)


if __name__ == '__main__':
    main(verbosity=2)
