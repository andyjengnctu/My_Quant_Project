"""Primitive invariants for PIT numeric strategy conditions, independent of MR IDs."""
from __future__ import annotations
from dataclasses import replace
from copy import deepcopy
from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd
from core.portfolio_param_runtime import build_portfolio_params_signature
from filters.breakout_quality.trade_decision_context import (
    CONTEXT_FEATURES, CONTEXT_CONTRACT, build_trade_decision_context,
)
from services.breakout_quality.formal_mintrade40_pit_source import FormalMinTrade40PITSource
from services.breakout_quality.fitting_data_dependencies import mintrade40_fit_sample_sha256
from tools.validate.synthetic_daily_trade40_cases import fixture


class TradeDecisionContextCases(unittest.TestCase):
    def setUp(self):
        self.days, self.frame, self.params, self.period = fixture()
        self.calendar = self.days.append(pd.DatetimeIndex([self.days[-1] + pd.offsets.BDay()]))
        self.source = FormalMinTrade40PITSource(
            42, 'a' * 64, (self.period,),
            (str((self.days[0] - pd.Timedelta(days=1)).date()),),
        )
        self.positions = np.array([40, 48, 75, 90, 129])

    def build(self, **override):
        kwargs = dict(source=self.source, frame=self.frame, positions=self.positions,
                      ticker='2330', exchange_calendar=self.calendar)
        kwargs.update(override)
        return build_trade_decision_context(**kwargs)

    def test_context_is_finite_and_all_requested_stockdays_remain(self):
        result = self.build()
        self.assertEqual(result.shape, (len(self.positions), len(CONTEXT_FEATURES)))
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue((result[:, 0] == 1).all())
        self.assertEqual(result[-1, 0], 1)  # Last close has no future ticker quote.

    def test_future_price_mutation_cannot_change_earlier_context(self):
        positions = self.positions[:3]
        expected = self.build(positions=positions)
        changed = self.frame.copy()
        changed.iloc[76:, :4] *= 9.
        changed.iloc[76:, 4] = 0.
        np.testing.assert_array_equal(expected, self.build(frame=changed, positions=positions))
        np.testing.assert_array_equal(self.build()[:3], self.build(frame=changed)[:3])

    def test_all_price_ratios_and_signal_features_are_scale_invariant(self):
        expected = self.build()
        for factor in (0.1, 2., 10.):
            changed = self.frame.copy()
            changed.loc[:, ['Open', 'High', 'Low', 'Close']] *= factor
            np.testing.assert_allclose(expected, self.build(frame=changed), rtol=2e-6, atol=2e-7)

    def test_unpublished_future_teacher_does_not_backfill(self):
        late = replace(self.source, publication_dates=(str(self.days[-1].date()),))
        np.testing.assert_array_equal(self.build(source=late)[:-1], 0.)
        np.testing.assert_array_equal(self.build(source=late)[-1:], self.build()[-1:])

    def test_effective_teacher_changes_only_when_published_and_effective(self):
        params = replace(self.params, high_len=20, atr_times_init=3.)
        boundary = self.days[80]
        first = replace(self.period, end=boundary - pd.Timedelta(days=1))
        second = replace(self.period, start=boundary, params=params,
                         params_signature=build_portfolio_params_signature(params))
        source = replace(self.source, schedule=(first, second), publication_dates=(
            self.source.publication_dates[0], str(self.days[79].date())))
        result = self.build(source=source, positions=np.array([78,79,80]))
        self.assertEqual(result[0, 5], self.params.atr_times_init)
        np.testing.assert_array_equal(result[1:, 5], 3.)

    def test_missing_teacher_is_masked_without_outcome_based_eligibility(self):
        later = replace(self.source, schedule=(replace(self.period, start=self.days[60]),))
        result = self.build(source=later)
        np.testing.assert_array_equal(result[:2], 0.)
        self.assertEqual(result.shape[0], len(self.positions))

    def test_invalid_or_ambiguous_source_fails_closed(self):
        with self.assertRaises(ValueError):
            self.build(source=replace(self.source, source_id='another_teacher'))
        with self.assertRaises(ValueError):
            self.build(source=replace(self.source, schedule=(self.period,self.period),
                                     publication_dates=self.source.publication_dates*2))
        with self.assertRaises(ValueError):
            self.build(positions=np.array([-1]))

    def test_fit_digest_hashes_only_declared_ordered_fit_context(self):
        context = self.build()
        table = pd.DataFrame(dict(ticker=['2330']*len(context), date=self.days.take(self.positions),
            source_pos=self.positions, label_eval_end_date=self.days.take(self.positions)))
        bundle = SimpleNamespace(summary={'input_decision_context': deepcopy(CONTEXT_CONTRACT)},
            target_manifest={'target_contract': {
                'target_kind':'formal_seed_mintrade_fullcycle_broker_net_pnl',
                'teacher_lineage': {'contract':'formal_mintrade40_seed_historical_plus_p16_v1'}}},
            group_table=table, group_context=context.copy(), raw_target=np.arange(len(context),dtype=float))
        ids = np.array([0,1])
        original = mintrade40_fit_sample_sha256(bundle, ids)
        self.assertIsNotNone(original)
        bundle.group_context[-1] += 1.
        self.assertEqual(original, mintrade40_fit_sample_sha256(bundle, ids))
        bundle.group_context[0, 0] += 1.
        self.assertNotEqual(original, mintrade40_fit_sample_sha256(bundle, ids))
        bundle.summary = {}
        legacy = mintrade40_fit_sample_sha256(bundle, ids)
        bundle.group_context[:] = 0.
        self.assertEqual(legacy, mintrade40_fit_sample_sha256(bundle, ids))

    def test_numeric_input_extension_preserves_target_maturity_and_score_universe(self):
        from unittest.mock import patch
        from functools import partial
        from pathlib import Path
        from core.breakout_quality_registry import FORMAL_MINTRADE_NETPNL_DECISION_CONTEXT_PROFILE
        from filters.breakout_quality import daily_ranker_data as loader
        from filters.breakout_quality import daily_model_input_context as model_inputs
        from filters.breakout_quality import daily_trade40_target as target
        from filters.breakout_quality import trade_decision_context

        days, frame, _, period = fixture(count=520, start="2018-01-01")
        source = FormalMinTrade40PITSource(42, "a" * 64, (period,), ("2017-12-31",))

        def outcomes(*, source, frame, positions, **kwargs):
            valid = positions % 2 == 0
            return SimpleNamespace(
                raw_net_r=np.where(valid, positions / 100.0, np.nan),
                raw_net_pnl=np.where(valid, positions * 10.0, np.nan),
                completed=valid,
                maturity_dates=frame.index.take(positions + 1).to_numpy(dtype="datetime64[D]"),
            )

        kwargs = dict(
            filter_id="breakout_quality_v1", preload_feature_bank=False, allow_stale_source=False,
            project_root=Path.cwd(), target_materializer=partial(outcomes, source=source),
            target_lineage={
                "contract": "formal_mintrade40_seed_historical_plus_p16_v1", "seed": 42,
                "source_sha256": "a" * 64, "first_effective_date": str(days[0].date()),
                "teacher_periods": [], "value_field": "raw_net_pnl",
                "target_kind": "formal_seed_mintrade_fullcycle_broker_net_pnl",
            },
        )
        dataset = {"dataset": "full", "source_data_date_range": {"end": str(days[-1].date())}}
        policy = {"effective_oos_end_date": str(days[-1].date()), "oos_start_date": "2019-01-01"}
        csv_inputs = ((("0050", Path("0050.csv")), ("2330", Path("2330.csv"))), [])

        def load_context():
            return loader.load_daily_universal_ranker_data(
                model_architecture="inception_time_trade_decision_context_v1",
                experiment_profile=FORMAL_MINTRADE_NETPNL_DECISION_CONTEXT_PROFILE,
                **kwargs,
            )

        def changed_input(*, positions, **kwargs):
            return np.full((len(positions), len(CONTEXT_FEATURES)), 2.0, dtype=np.float32)

        with (
            patch.object(loader, "load_validated_dataset_bundle",
                         return_value=(dataset, None, None, None, pd.DataFrame())),
            patch.object(loader, "resolve_breakout_quality_outer_policy", return_value=policy),
            patch.object(loader, "discover_dataset_csv_inputs", return_value=csv_inputs),
            patch.object(loader, "load_dataset_frame", return_value=frame),
            patch.object(target, "collect_canonical_exchange_sessions", return_value=days),
            patch.object(model_inputs, "bind_daily_model_input_context",
                         wraps=model_inputs.bind_daily_model_input_context) as bind,
        ):
            original = loader.load_daily_universal_ranker_data(
                model_architecture="inception_time_v1",
                experiment_profile="daily_universal_formal_mintrade_fullcycle_netpnl_per_seed_full_list_ndcg_pairwise",
                **kwargs,
            )
            bind.assert_not_called()
            context = load_context()
            bind.assert_called_once()
            with patch.object(trade_decision_context, "build_trade_decision_context", side_effect=changed_input):
                changed = load_context()

        for bundle in (context, changed):
            np.testing.assert_array_equal(original.target_valid, bundle.target_valid)
            np.testing.assert_array_equal(original.raw_target, bundle.raw_target)
            pd.testing.assert_frame_equal(original.group_table, bundle.group_table)
            self.assertEqual(original.target_manifest["target_contract"], bundle.target_manifest["target_contract"])
            self.assertEqual(original.feature_bank.shape, bundle.feature_bank.shape)
            self.assertEqual(bundle.group_context.shape, (len(original.group_table), len(CONTEXT_FEATURES)))
            self.assertEqual(bundle.summary["input_decision_context"], CONTEXT_CONTRACT)
            np.testing.assert_array_equal(original.feature_bank[:3], bundle.feature_bank[:3])
        np.testing.assert_array_equal(changed.group_context, 2.0)
        self.assertFalse(np.array_equal(context.group_context, changed.group_context))

    def test_common_same_date_policy_can_change_pairwise_stock_ordering(self):
        from filters.breakout_quality.models.inception_time import fuse_decision_context
        backend = SimpleNamespace(cat=lambda values, dim: np.concatenate(values, axis=dim))
        encoded = np.array([[1.,0.],[0.,1.]], dtype=np.float32)
        low = fuse_decision_context(encoded, np.ones((2,1),dtype=np.float32), backend)
        high = fuse_decision_context(encoded, np.full((2,1),3.,dtype=np.float32), backend)
        # score = z0-z1 - .5*(z0*c-z1*c), same policy for both stocks.
        coefficients = np.array([1.,-1.,0.,-.5,.5],dtype=np.float32)
        self.assertGreater((low @ coefficients)[0], (low @ coefficients)[1])
        self.assertLess((high @ coefficients)[0], (high @ coefficients)[1])


if __name__ == '__main__':
    unittest.main()
