"""Stage-local weighting, canonical episodes and numerical RankNet invariants."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.breakout_quality_fit_weighting import EPISODE_COLUMN, ORIGINAL_SIGNAL_INVERSE_FREQUENCY as POLICY
from core.breakout_quality_runtime import CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG as REDUCTION
from filters.breakout_quality.fit_sample_weighting import (
    attach_fit_episodes, build_fit_sample_weights, ordered_fit_episode_sha256, PairMassAudit,
)


def sample_table():
    dates = pd.to_datetime(["2020-01-02"] * 3 + ["2020-01-03"] * 3 + ["2020-01-06"] * 3)
    return pd.DataFrame({
        "ticker": ["A", "B", "C"] * 3, "date": dates,
        "label_eval_end_date": dates + pd.Timedelta(days=1),
        EPISODE_COLUMN: [f"42|{ticker}|{origin}" for ticker, origin in zip(
            ["A", "B", "C"] * 3,
            ["2020-01-02"] * 3 + ["2020-01-02", "2020-01-03", "2020-01-02"]
            + ["2020-01-02", "2020-01-06", "2020-01-02"],
        )],
    })


class FitSampleWeightingTests(unittest.TestCase):
    def weights(self, table, ids, cutoff="2020-01-08"):
        return build_fit_sample_weights(table, np.asarray(ids), policy=POLICY,
                                        cutoff=cutoff, raw_target=np.arange(len(table), dtype=float), expected_seed=42)

    def test_counts_are_stage_local_and_final_recomputed(self):
        table = sample_table()
        inner = self.weights(table, range(6), "2020-01-06")
        np.testing.assert_array_equal(inner.values[:6], [.5, 1, .5, .5, 1, .5])
        # Validation/OOS continuations of A/C cannot alter earlier FIT counts.
        altered = pd.concat([table, table.iloc[6:].assign(date=pd.Timestamp("2021-01-01"),
                              label_eval_end_date=pd.Timestamp("2021-01-02"))], ignore_index=True)
        again = self.weights(altered, range(6), "2020-01-06")
        np.testing.assert_array_equal(again.values[:6], inner.values[:6])
        self.assertEqual(again.summary, inner.summary)
        final = self.weights(table, range(9))
        np.testing.assert_allclose(final.values[[0, 3, 6]], 1 / 3)
        self.assertEqual(final.summary["episodes"], 5)
        # A label from an earlier information date may mature only for refit.
        table.loc[4, "label_eval_end_date"] = pd.Timestamp("2020-01-06")
        inner_ids = [0, 1, 2, 3, 5]
        self.assertEqual(self.weights(table, inner_ids, "2020-01-06").summary["rows"], 5)
        self.assertEqual(self.weights(table, range(9)).summary["rows"], 9)
        self.assertTrue(np.isnan(inner.values[6:]).all())
        self.assertFalse(inner.values.flags.writeable)

    def test_missing_ambiguous_future_mixed_seed_and_null_rows_fail(self):
        for kind in ("missing", "other_seed", "ticker", "future_origin", "immature", "duplicate"):
            table = sample_table()
            if kind == "missing": table.loc[0, EPISODE_COLUMN] = None
            if kind == "other_seed": table.loc[0, EPISODE_COLUMN] = "7|A|2020-01-02"
            if kind == "ticker": table.loc[0, EPISODE_COLUMN] = "42|B|2020-01-02"
            if kind == "future_origin": table.loc[0, EPISODE_COLUMN] = "42|A|2020-01-03"
            if kind == "immature": table.loc[0, "label_eval_end_date"] = pd.Timestamp("2020-01-08")
            if kind == "duplicate": table.loc[3, "date"] = table.loc[0, "date"]
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.weights(table, range(9))
        raw = np.arange(9, dtype=float); raw[0] = np.nan
        with self.assertRaises(ValueError):
            build_fit_sample_weights(sample_table(), np.arange(9), policy=POLICY,
                                     cutoff="2020-01-08", raw_target=raw, expected_seed=42)
        with self.assertRaises(ValueError):
            self.weights(sample_table(), range(9), None)
        table = sample_table()
        table[EPISODE_COLUMN] = table[EPISODE_COLUMN].str.replace("42|", "7|", regex=False)
        with self.assertRaises(ValueError):
            self.weights(table, range(9))

    def test_fit_identity_binds_only_ordered_selected_episode_metadata(self):
        table = sample_table()
        ids = np.arange(6)
        before = ordered_fit_episode_sha256(table, ids, POLICY)
        table.loc[8, EPISODE_COLUMN] = "42|C|2020-01-06"
        self.assertEqual(ordered_fit_episode_sha256(table, ids, POLICY), before)
        table.loc[5, EPISODE_COLUMN] = "42|C|2020-01-03"
        self.assertNotEqual(ordered_fit_episode_sha256(table, ids, POLICY), before)
        self.assertIsNone(ordered_fit_episode_sha256(pd.DataFrame(), [], None))

    def test_canonical_lifecycle_replacement_rollover_reset_and_future_suffix(self):
        from tools.validate.synthetic_daily_trade40_cases import fixture
        from core.portfolio_param_runtime import build_portfolio_params_signature
        from filters.breakout_quality.daily_trade40_target import compute_counterfactual_trade40_batch
        days, frame, params, period = fixture(count=150)
        later = replace(params, high_len=12)
        schedule = (replace(period, end=days[59]), replace(period, start=days[60],
                    params=later, params_signature=build_portfolio_params_signature(later)))
        def project(data, positions, *, reset=None, periods=schedule):
            keys = {}
            compute_counterfactual_trade40_batch(data, np.asarray(positions), ticker="2330",
                schedule=periods, exchange_calendar=days, horizon_sessions=None,
                signal_state_reset_date=reset, episode_keys_out=keys, episode_keys_only=True)
            return keys
        keys = project(frame, range(45, 115))
        self.assertNotIn(47, keys)
        self.assertEqual(keys[48], keys[74])
        self.assertEqual(keys[59], keys[60])
        self.assertEqual(keys[75], f"2330|{days[75].date()}")
        self.assertEqual(keys[90], f"2330|{days[90].date()}")
        self.assertNotEqual(keys[74], keys[75])
        prefix = project(frame.iloc[:81], range(45, 80))
        self.assertEqual(prefix, {k: v for k, v in keys.items() if k < 80})
        changed = frame.copy(); changed.iloc[81:, :4] *= 2
        self.assertEqual(project(changed, range(45, 80)), prefix)
        reset = project(frame, range(45, 80), reset=days[60])
        self.assertNotIn(60, reset)
        self.assertEqual(reset[75], keys[75])
        # Genuine lifecycle exit clears pending state before the next setup.
        falling = frame.copy(); falling.iloc[63:70, :4] = 20
        self.assertNotIn(64, project(falling, range(45, 70)))

    def test_metadata_join_preserves_order_seed_and_nonfit_rows(self):
        from tools.validate.synthetic_daily_trade40_cases import fixture
        days, frame, _, period = fixture(count=150)
        positions = [75, 49, 60, 47]
        table = pd.DataFrame({"ticker": ["2330"] * 4, "ticker_id": [0] * 4,
                              "source_pos": positions, "date": days.take(positions)})
        old = table.copy()
        source = SimpleNamespace(seed=42, schedule=(period,), oos_start_date=None)
        kwargs = dict(frames=(frame,), tickers=("2330",), source=source,
                      exchange_calendar=days, policy=POLICY)
        attach_fit_episodes(table, target_valid=[True, True, True, False], **kwargs)
        pd.testing.assert_frame_equal(table.drop(columns=EPISODE_COLUMN), old)
        self.assertEqual(table.loc[0, EPISODE_COLUMN], f"42|2330|{days[75].date()}")
        self.assertEqual(table.loc[1, EPISODE_COLUMN], table.loc[2, EPISODE_COLUMN])
        self.assertTrue(pd.isna(table.loc[3, EPISODE_COLUMN]))
        with self.assertRaises(ValueError):
            attach_fit_episodes(old, target_valid=[True] * 4, **kwargs)

    def test_weighted_loader_preserves_co_inputs_targets_and_order(self):
        from tools.validate import test_daily_sample_pipeline as fixture_owner
        from tools.validate.synthetic_daily_trade40_cases import fixture
        from core.breakout_quality_registry import CO_PROFILE, SIGNAL_FREQUENCY_PROFILE
        from services.breakout_quality.formal_mintrade40_pit_source import (
            FormalMinTrade40PITSource, materialize_formal_mintrade40_pit_labels,
        )
        days, frame, params, period = fixture(count=520, start="2018-01-01")
        for index in (310, 350, 390, 430, 470):
            frame.iloc[index - 1, :4] -= 7
            frame.iloc[index:, :4] += 8
        source = FormalMinTrade40PITSource(seed=42, param_sha256="a" * 64,
            schedule=(period,), publication_dates=("2017-12-31",))
        def labels(**kwargs):
            return materialize_formal_mintrade40_pit_labels(**kwargs, expected_seed=42,
                horizon_sessions=None, outcome_metric="net_pnl")
        with patch.object(fixture_owner, "fixture", return_value=(days, frame, params, period)), \
                patch.object(fixture_owner, "_outcomes", side_effect=labels):
            original = fixture_owner.build_isolated_daily_bundle(profile=CO_PROFILE, teacher=source)
            weighted = fixture_owner.build_isolated_daily_bundle(profile=SIGNAL_FREQUENCY_PROFILE, teacher=source)
        self.assertGreater(int(weighted.target_valid.sum()), 0)
        pd.testing.assert_frame_equal(original.group_table, weighted.group_table.drop(columns=EPISODE_COLUMN))
        np.testing.assert_array_equal(original.raw_target, weighted.raw_target)
        np.testing.assert_array_equal(original.target_valid, weighted.target_valid)
        np.testing.assert_array_equal(original.feature_bank[:], weighted.feature_bank[:])
        np.testing.assert_array_equal(original.group_context, weighted.group_context)
        self.assertEqual(original.target_manifest, weighted.target_manifest)
        ids = np.flatnonzero(weighted.target_valid)
        self.assertFalse(weighted.group_table.iloc[ids][EPISODE_COLUMN].isna().any())
        self.assertTrue(weighted.group_table.loc[~weighted.target_valid, EPISODE_COLUMN].isna().all())

    def test_all_ones_loss_gradient_parity_and_detached_metadata(self):
        import torch
        from services.breakout_quality.train_continuous_ranker import _pairwise_logistic_loss
        for dtype in (torch.float32, torch.float64):
            margins = torch.tensor([.3, -.5, 1.2, .4, -.2, .1], dtype=dtype, requires_grad=True)
            target = torch.tensor([0, .5, 1, 1, .5, .5], dtype=dtype)
            dates = ["2020-01-02"] * 3 + ["2020-01-03"] * 3
            options = dict(reduction=REDUCTION)
            original, count = _pairwise_logistic_loss(torch, margins, target, dates, **options)
            gradient = torch.autograd.grad(original, margins)[0]
            weights = torch.ones_like(margins, requires_grad=True)
            actual, actual_count = _pairwise_logistic_loss(torch, margins, target, dates,
                                                          item_weights=weights, **options)
            new_gradient, metadata_gradient = torch.autograd.grad(actual, (margins, weights), allow_unused=True)
            self.assertEqual(count, actual_count)
            self.assertTrue(torch.equal(original, actual))
            self.assertTrue(torch.equal(gradient, new_gradient))
            self.assertIsNone(metadata_gradient)

    def test_product_formula_multidate_normalization_ties_and_pair_mass(self):
        import torch
        import torch.nn.functional as F
        from services.breakout_quality.train_continuous_ranker import _pairwise_logistic_loss
        table = sample_table()
        stage = self.weights(table, range(9))
        audit = PairMassAudit(stage)
        margins = torch.tensor([.3, -.5, 1.2, .4, -.2, .1, 1, .2, .7], dtype=torch.float64, requires_grad=True)
        target = torch.tensor([0, .5, 1, 1, .5, .5, .5, .5, .5], dtype=torch.float64)
        numerator = denominator = 0
        before_mass = np.zeros(9); after_mass = np.zeros(9)
        for ids in ([0, 1, 2], [3, 4, 5], [6, 7, 8]):
            scores = margins.detach()[ids]; truth = target[ids]
            # The unchanged CO primitive computes ranking discounts / IDCG in FP32.
            discounts = 1 / torch.log2(torch.arange(2, 5, dtype=torch.float32))
            ranked = torch.empty_like(discounts); ranked[torch.argsort(scores, descending=True, stable=True)] = discounts
            idcg = torch.sum(torch.sort(truth.float(), descending=True).values * discounts)
            for i in range(3):
                for j in range(i + 1, 3):
                    if truth[i] == truth[j]: continue
                    delta = abs(truth[i] - truth[j]) * abs(ranked[i] - ranked[j]) / idcg
                    weight = delta * float(stage.values[ids[i]]) * float(stage.values[ids[j]])
                    numerator += weight * F.softplus(-torch.sign(truth[i] - truth[j]) * (margins[ids[i]] - margins[ids[j]]))
                    denominator += weight
                    before_mass[[ids[i], ids[j]]] += float(delta)
                    after_mass[[ids[i], ids[j]]] += float(weight)
        actual, count = _pairwise_logistic_loss(torch, margins, target, table.date,
            reduction=REDUCTION, item_weights=stage.values.copy(),
            pair_mass_audit=audit.batch(np.arange(9)))
        expected = numerator / denominator
        self.assertEqual(count, 5)
        torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(torch.autograd.grad(actual, margins)[0],
                                   torch.autograd.grad(expected, margins)[0], rtol=1e-14, atol=1e-14)
        for raw, observed in ((before_mass, audit.before), (after_mass, audit.after)):
            expected_mass = np.bincount(stage.episode_codes, weights=raw / raw.sum(), minlength=len(stage.counts))
            np.testing.assert_allclose(observed, expected_mass, rtol=1e-12, atol=1e-12)
        self.assertEqual(audit.summary()["optimizer_batches"], 1)
        empty, count = _pairwise_logistic_loss(torch, margins[6:], target[6:], table.date.iloc[6:],
            reduction=REDUCTION, item_weights=stage.values[6:].copy())
        self.assertIsNone(empty); self.assertEqual(count, 0)

    def test_real_epoch_selection_and_final_refit_use_separate_fixed_weights(self):
        import torch
        from services.breakout_quality import train_continuous_ranker as trainer
        from core.breakout_quality_registry import (
            SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES, get_breakout_quality_experiment_profile,
        )
        profiles = [get_breakout_quality_experiment_profile(p) for p in SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES]
        weighted = [p for p in profiles if p.fit_sample_weight_policy == POLICY]
        self.assertTrue(weighted)
        torch.set_num_threads(1)
        for profile in weighted:
            args = trainer.parse_args(["--experiment-profile", profile.name,
                                       "--model-architecture", profile.model_architecture])
            args.epochs = 2; args.early_stopping_patience = 0
            args.batch_size = 128; args.train_prefetch_batches = 0
            args.inference_prefetch_batches = 0; args.seed = 42
            plan = trainer.resolve_torch_execution_plan(torch, requested_device="cpu",
                mixed_precision=False, mixed_precision_dtype="float16", deterministic_algorithms=True, allow_tf32=False)
            table = sample_table(); table["label"] = -1
            inputs = np.random.default_rng(10).normal(size=(9, 300, 10)).astype(np.float32)
            contexts = np.empty((9, 0), np.float32)
            raw = np.tile(np.asarray([1, 2, 3], np.float32), 3)
            target = np.tile(np.asarray([0, .5, 1], np.float32), 3)
            with patch.object(trainer, "_train_ranker_fitting_epoch", wraps=trainer._train_ranker_fitting_epoch) as epochs:
                selection = trainer.select_epoch(torch, inputs, contexts, table, raw, target,
                    np.arange(6), np.arange(6, 9), args=args, plan=plan,
                    evaluate_train_metrics=False, selection_metrics_only=True, fit_cutoff="2020-01-06")
                self.assertEqual(selection["fit_sample_weighting"]["rows"], 6)
                self.assertIs(epochs.call_args_list[0].kwargs["fit_sample_weights"],
                              epochs.call_args_list[1].kwargs["fit_sample_weights"])
                _, history = trainer.fit_final(torch, inputs, contexts, raw, target, table, np.arange(9),
                    epochs=2, args=args, plan=plan, epoch_selection=selection, fit_cutoff="2020-01-08")
                self.assertEqual(history[0]["fit_sample_weighting"]["rows"], 9)
                self.assertIsNot(epochs.call_args_list[1].kwargs["fit_sample_weights"],
                                 epochs.call_args_list[2].kwargs["fit_sample_weights"])
                self.assertIs(epochs.call_args_list[2].kwargs["fit_sample_weights"],
                              epochs.call_args_list[3].kwargs["fit_sample_weights"])
                self.assertTrue(all(row["pair_mass_audit"]["optimizer_batches"] == 1 for row in history))


def check_fit_sample_weighting(check_true):
    for name in unittest.defaultTestLoader.getTestCaseNames(FitSampleWeightingTests):
        result = unittest.TestResult()
        FitSampleWeightingTests(name).run(result)
        detail = "\n".join(text for _, text in result.errors + result.failures)
        check_true("fit_sample_weighting_" + name + ("\n" + detail if detail else ""), result.wasSuccessful())


if __name__ == "__main__":
    unittest.main()
