"""One reusable top-heavy listwise capability and its registered consumers.

No market FIT, Oracle simulation, artifact writes or report-version changes.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from core.breakout_quality_runtime import (
    TRAINING_OBJECTIVE_DAILY_HEAD_WEIGHTED_LISTWISE,
    CONTINUOUS_RANKER_BATCH_MODE_DATE_COHERENT,
    get_continuous_ranker_training_policy,
)
from core.breakout_quality_registry import (
    get_breakout_quality_experiment_profile,
    get_continuous_ranker_research_spec,
    DAILY_UNIVERSAL_TERMINAL_MAE_GATED_T40_HEAD_WEIGHTED_LISTWISE_PROFILE,
)
from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
from filters.breakout_quality.head_weighted_listwise import (
    head_weighted_sequential_choice_loss,
    mean_daily_head_discounted_gain,
)
from services.breakout_quality.train_continuous_ranker import _date_coherent_batches, _train_epoch



class HeadWeightedListwiseCases(unittest.TestCase):
    def test_top_choice_loss_prefers_true_top(self):
        truth = torch.tensor([1.0, 0.75, 0.5, 0.25])
        dates = ["2020-01-02"] * 4
        ideal = torch.tensor([4.0, 2.0, 1.0, 0.0], requires_grad=True)
        reverse = torch.tensor([0.0, 1.0, 2.0, 4.0], requires_grad=True)
        best_loss, days = head_weighted_sequential_choice_loss(torch, ideal, truth, dates)
        worst_loss, worst_days = head_weighted_sequential_choice_loss(torch, reverse, truth, dates)
        self.assertEqual((days, worst_days), (1, 1))
        self.assertLess(best_loss.item(), worst_loss.item())
        best_loss.backward()
        self.assertTrue(torch.isfinite(ideal.grad).all().item())
        self.assertNotAlmostEqual(float(ideal.grad[0]), 0.0, places=6)

    def test_stronger_early_position_gradient_than_tail(self):
        truth = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
        base = torch.zeros(5, requires_grad=True)
        loss, days = head_weighted_sequential_choice_loss(torch, base, truth, ['2020-01-02'] * 5)
        self.assertEqual(days, 1)
        loss.backward()
        self.assertGreater(abs(float(base.grad[0])), abs(float(base.grad[-1])))

    def test_ties_have_no_arbitrary_order(self):
        truth = torch.tensor([1.0, 1.0, 0.4, 0.4, 0.0])
        scores = torch.tensor([2.0, 0.5, -0.1, -0.8, -1.2])
        date = ['2020-01-02'] * 5
        loss, days = head_weighted_sequential_choice_loss(torch, scores, truth, date)
        swapped, swap_days = head_weighted_sequential_choice_loss(
            torch, scores[[1, 0, 3, 2, 4]], truth, date
        )
        self.assertEqual((days, swap_days), (1, 1))
        self.assertAlmostEqual(loss.item(), swapped.item(), places=6)
        flat, n = head_weighted_sequential_choice_loss(torch, scores, torch.ones(5), date)
        self.assertIsNone(flat)
        self.assertEqual(n, 0)

    def test_maturity_dates_and_missing_values(self):
        y = torch.tensor([1.0, 0.0, float('nan'), 0.25, 0.75])
        values = torch.tensor([1., -1., 999., 0., 2.], requires_grad=True)
        dates = ['2020-01-02']*3 + ['2020-01-03']*2
        loss, count = head_weighted_sequential_choice_loss(torch, values, y, dates)
        self.assertEqual(count, 2)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertEqual(float(values.grad[2]), 0.0)
        perm = [4, 2, 0, 3, 1]
        permuted, _ = head_weighted_sequential_choice_loss(
            torch, values.detach()[perm], y[perm], [dates[k] for k in perm]
        )
        self.assertAlmostEqual(loss.item(), permuted.item(), places=6)

    def test_no_fixed_k_normalized_front_quality(self):
        days = ['2020-01-02'] * 5
        target = np.array([1.0, 0.8, 0.6, 0.4, 0.2], dtype=float)
        self.assertAlmostEqual(mean_daily_head_discounted_gain(days, target, target), 1.0)
        self.assertAlmostEqual(mean_daily_head_discounted_gain(days, target, -target), 0.0)
        early = np.array([0.8, 1., 0.6, .4, .2], dtype=float)
        late = np.array([1., 0.8, .6, .2, .4], dtype=float)
        self.assertLess(mean_daily_head_discounted_gain(days, target, early),
                        mean_daily_head_discounted_gain(days, target, late))
        self.assertIsNone(mean_daily_head_discounted_gain(days, np.ones(5), target))

    def test_full_date_batching_and_training_identity(self):
        dates = ['2020-01-02']*3 + ['2020-01-03']*4
        batches = _date_coherent_batches(np.arange(7), dates, batch_size=2, seed=42)
        groups = [{dates[i] for i in ids} for ids in batches]
        self.assertTrue(all(len(group) == 1 for group in groups))
        self.assertEqual(sorted(np.concatenate(batches).tolist()), list(range(7)))
        name = DAILY_UNIVERSAL_TERMINAL_MAE_GATED_T40_HEAD_WEIGHTED_LISTWISE_PROFILE
        profile = get_breakout_quality_experiment_profile(name)
        recipe = get_continuous_ranker_execution_recipe(name)
        spec = get_continuous_ranker_research_spec(name)
        self.assertEqual(profile.continuous_target_id, 'daily_terminal_mae_gated_t40_v1')
        self.assertEqual(recipe.training_policy.batch_mode, CONTINUOUS_RANKER_BATCH_MODE_DATE_COHERENT)
        self.assertEqual(recipe.training_policy, get_continuous_ranker_training_policy(
            TRAINING_OBJECTIVE_DAILY_HEAD_WEIGHTED_LISTWISE))
        self.assertEqual(recipe.training_policy.epoch_selection_rule, 'head_discounted_gain_then_spearman')
        self.assertEqual(spec.model_research_id, 'MR-13CY')
        self.assertEqual(spec.pairwise_reduction, None)
        self.assertTrue(recipe.historical_pit_authorized)

    def test_generic_one_epoch_backpropagates_across_whole_dates(self):
        torch.manual_seed(101)
        n = 24
        rng = np.random.default_rng(101)
        features = rng.standard_normal((n, 6, 3), dtype=np.float32)
        contexts = rng.standard_normal((n, 1), dtype=np.float32)
        dates = ['2020-01-02']*6 + ['2020-01-03']*6 + ['2020-01-06']*6 + ['2020-01-07']*6
        targets = np.tile(np.linspace(0., 1., 6, dtype=np.float32), 4)

        class MinimalRanker(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Linear(19, 2)

            def forward(self, xb, cb):
                return self.head(torch.cat((xb.flatten(1), cb), dim=1))

        model = MinimalRanker()
        before = [param.detach().clone() for param in model.parameters()]
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss = _train_epoch(
            torch, model, optimizer, features, contexts, np.arange(n), targets, dates,
            training_objective=TRAINING_OBJECTIVE_DAILY_HEAD_WEIGHTED_LISTWISE,
            batch_size=8, seed=42, gradient_clip_norm=1.0,
            plan=SimpleNamespace(
                device=torch.device('cpu'), device_type='cpu',
                mixed_precision_enabled=False, autocast_dtype_name=None,
            ),
            grad_scaler=None,
        )
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())))


if __name__ == '__main__':
    unittest.main()
