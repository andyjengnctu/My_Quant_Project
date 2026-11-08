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

    # AI: Preserve an approved historical model's membership across Training Model switches.
    def test_registered_model_and_rolling_strategy_arms_remain_in_compare_suite(self):
        """Historical regression: switching Training Model must not hide CY/C151/C152."""
        from config.breakout_quality import (
            BREAKOUT_QUALITY_MODEL_RESEARCH_MODEL_PROFILE,
            BREAKOUT_QUALITY_MODEL_TEST_PROFILES,
            BREAKOUT_QUALITY_MODEL_TEST_REFERENCE_PROFILES,
        )
        from core.breakout_quality_policy import get_breakout_quality_model_test_settings
        from core.strategy_compare_policy import get_strategy_compare_suite
        from config.strategy_compare import STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE
        from core.strategy_compare_registry import STRATEGY_COMPARE_PROFILES
        from core.strategy_compare_registry import STRATEGY_COMPARE_ARMS, STRATEGY_DL_SOURCES
        from services.research.target_dl_overview import _strategy_bindings
        from services.research.target_oracle_application import get_formal_oracle_definition
        from core.audit_policy import get_active_audit_module_id
        from core.audit_registry import (
            resolve_current_oracle_comparisons,
            oracle_comparison_key,
            oracle_supervised_target_id,
        )

        profile_name = DAILY_UNIVERSAL_TERMINAL_MAE_GATED_T40_HEAD_WEIGHTED_LISTWISE_PROFILE
        model_id = get_continuous_ranker_research_spec(profile_name).model_research_id
        model_pair = (model_id, profile_name)
        self.assertIn(model_pair, BREAKOUT_QUALITY_MODEL_TEST_REFERENCE_PROFILES)
        self.assertIn(model_pair, BREAKOUT_QUALITY_MODEL_TEST_PROFILES)
        self.assertIn(model_pair, get_breakout_quality_model_test_settings().model_profiles)
        self.assertIn(BREAKOUT_QUALITY_MODEL_RESEARCH_MODEL_PROFILE, BREAKOUT_QUALITY_MODEL_TEST_PROFILES)

        suite_name = STRATEGY_COMPARE_PROFILES[STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE]['suite_id']
        active_arms = set(get_strategy_compare_suite(suite_name)['arm_ids'])
        expected_arms = {'C151', 'C152'}
        self.assertTrue(expected_arms.issubset(active_arms))
        self.assertEqual({STRATEGY_DL_SOURCES[STRATEGY_COMPARE_ARMS[arm]['dl_id']]['experiment_profile']
                          for arm in expected_arms}, {profile_name})

        definition = get_formal_oracle_definition(get_active_audit_module_id())
        comparisons = resolve_current_oracle_comparisons(definition.source)
        targets = (str(definition.source['baseline_arm_id']),
                   *(oracle_comparison_key(item) for item in comparisons))
        score_targets = {}
        for item in comparisons:
            source = item
            if item.get('score_source'):
                source = next(other for other in comparisons
                              if other['target'] == item['score_source'])
            score_targets[oracle_comparison_key(item)] = (
                None if item.get('positive_gate') else oracle_supervised_target_id(source)
            )
        bindings = _strategy_bindings(target_names=targets,
                                      score_targets=score_targets, definition=definition)
        for selector, arm_id in (('exact', 'C151'), ('direct', 'C152')):
            matching = [oracle_comparison_key(item) for item in comparisons
                        if item['target'] == ('T40' if selector == 'exact' else 'T40_Direct')
                        and item.get('strategy', 'Breakout') == 'Breakout'
                        and item.get('parameter_family', 'min') == 'min'
                        and item['selector'] == selector]
            self.assertEqual(len(matching), 1)
            self.assertEqual(bindings[(matching[0], profile_name)], arm_id)

    # AI: Regression fixture supplies values through normal readers, never runtime literals.
    def test_cy_published_metrics_reappear_on_overview_without_cross_arm_inheritance(self):
        """The overview joins existing canonical evidence; it never invents Robustness."""
        from pathlib import Path
        from unittest.mock import patch
        from services.research.target_dl_overview import collect_target_dl_overview

        profile_name = DAILY_UNIVERSAL_TERMINAL_MAE_GATED_T40_HEAD_WEIGHTED_LISTWISE_PROFILE
        def published_rho(_path, _markdown, **kwargs):
            return 0.1234 if kwargs['profile'] == profile_name else None

        with patch('services.research.target_dl_overview._single_model_rho',
                   side_effect=published_rho), \
             patch('services.research.target_dl_overview._rolling_fit_version_current',
                   return_value=True), \
             patch('services.research.target_dl_overview._strategy_rolling_metrics',
                   return_value={'C151': 9.68, 'C152': 7.00}), \
             patch('services.research.target_dl_overview._strategy_robustness_metrics',
                   return_value={}):
            report = collect_target_dl_overview(Path(__file__).resolve().parents[2])

        selected = {row['strategy']: row for row in report['rows']
                    if row['model'] == 'MR-13CY'}
        self.assertEqual(set(selected), {'C151', 'C152'})
        self.assertEqual(selected['C151']['rolling_romd'], 9.68)
        self.assertEqual(selected['C152']['rolling_romd'], 7.00)
        for row in selected.values():
            self.assertEqual(row['rolling_rho'], 0.1234)
            self.assertIsNone(row['robustness_romd'])
            self.assertIsNone(row['robustness_rho'])

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
