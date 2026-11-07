"""Focused synthetic checks for the reusable probabilistic pairwise ranker."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch

from core.breakout_quality_registry import (
    DAILY_UNIVERSAL_TERMINAL_MAE_GATED_T40_PROBABILISTIC_PROFILE,
    get_continuous_ranker_research_spec,
)
from core.breakout_quality_runtime import TRAINING_OBJECTIVE_DAILY_PAIRWISE_PROBABILITY
from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
from core.strategy_compare_registry import STRATEGY_COMPARE_ARMS, STRATEGY_DL_SOURCES
from filters.breakout_quality.pairwise_confidence_ranking import (
    expected_pairwise_win_fraction,
    probabilistic_pairwise_loss,
)
from services.breakout_quality.train_continuous_ranker import predict_scores
from services.research.target_dl_overview import _strategy_bindings
from services.research.target_oracle_application import get_formal_oracle_definition
from core.audit_policy import get_active_audit_module_id
from core.audit_registry import resolve_current_oracle_comparisons, oracle_comparison_key, oracle_supervised_target_id


class PairwiseConfidenceCases(unittest.TestCase):
    def test_antisymmetric_probability_and_no_magic_top_k(self):
        z = np.array([[-2.688, -0.915], [-3.802, -2.579],
                      [-3.683, -0.47], [-2.535, 0.543]], dtype=np.float64)
        u, v = z[:, 0], z[:, 1]
        logits = (u[:, None] - u[None, :]) + u[:, None] * v[None, :] - v[:, None] * u[None, :]
        probs = 1.0 / (1.0 + np.exp(-logits))
        np.testing.assert_allclose(probs + probs.T, np.ones_like(probs), atol=1e-12)
        scores = expected_pairwise_win_fraction(z, ['2020-01-02'] * 4)
        self.assertTrue(np.all((scores >= 0) & (scores <= 1)))
        self.assertAlmostEqual(float(scores.mean()), 0.5, places=6)
        # Cross-stock interactions must be capable of changing the initial u order.
        self.assertNotEqual(np.argsort(-u).tolist(), np.argsort(-scores).tolist())
        self.assertEqual(int(np.argmax(scores)), 3)
        reverse = expected_pairwise_win_fraction(z[::-1], ['2020-01-02'] * 4)
        np.testing.assert_allclose(scores, reverse[::-1], atol=1e-7)

    def test_dates_isolate_all_comparisons_and_singleton(self):
        z = np.array([[3.0, 1], [1.0, -1], [-2.0, 3]], dtype=np.float64)
        dates = ['2020-01-01', '2020-01-01', '2020-01-02']
        scores = expected_pairwise_win_fraction(z, dates)
        self.assertEqual(scores[2], np.float32(0.5))
        self.assertAlmostEqual(float(scores[0] + scores[1]), 1.0, places=6)
        with self.assertRaises(ValueError):
            expected_pairwise_win_fraction(np.array([[np.nan, 1]]), ['2020-01-01'])

    def test_pair_loss_excludes_ties_and_immature_truth_and_backpropagates(self):
        z = torch.tensor([[1.0, 0.3], [0.1, 0.5], [0.0, 1.0], [0.2, -0.3]], requires_grad=True)
        t = torch.tensor([3.0, 1.0, 1.0, float('nan')])
        loss, n = probabilistic_pairwise_loss(torch, z, t, ['2020-01-01'] * 4)
        self.assertEqual(n, 2)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(z.grad)
        self.assertGreater(float(z.grad.abs().sum()), 0)
        self.assertEqual(float(z.grad[3].abs().sum()), 0.0)
        self.assertEqual(probabilistic_pairwise_loss(torch, z[:2], t[:2],
                         ['2020-01-01', '2020-01-02']), (None, 0))

    def test_partial_inference_uses_complete_date_eligible_roster(self):
        z = np.array([[1.4, 1], [0.0, 1.7], [-0.5, -1], [1.0, -2], [-1, 1]], dtype=np.float32)
        dates = ['2020-01-01'] * 3 + ['2020-01-02'] * 2
        features = np.zeros((5, 2, 1), dtype=np.float32)
        context = np.zeros((5, 1), dtype=np.float32)
        def predict_mock(_torch, _model, _bank, _context, ids, **_kwargs):
            return z[np.asarray(ids, dtype=int)]
        with patch('services.breakout_quality.train_continuous_ranker._predict_logits', side_effect=predict_mock):
            full = predict_scores(torch, None, features, context, np.arange(5),
                                  batch_size=8, plan=None,
                                  training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_PROBABILITY,
                                  group_dates=dates)
            subset = predict_scores(torch, None, features, context, np.array([2, 0, 4]),
                                    batch_size=8, plan=None,
                                    training_objective=TRAINING_OBJECTIVE_DAILY_PAIRWISE_PROBABILITY,
                                    group_dates=dates)
        np.testing.assert_allclose(subset, full[[2, 0, 4]], atol=0)

    def test_scientific_registration_and_retired_daily_strategies(self):
        spec = get_continuous_ranker_research_spec(DAILY_UNIVERSAL_TERMINAL_MAE_GATED_T40_PROBABILISTIC_PROFILE)
        self.assertEqual(spec.model_research_id, 'MR-13CX')
        self.assertEqual(spec.pairwise_reduction, 'equal_pair_weight')
        recipe = get_continuous_ranker_execution_recipe(spec.profile_name)
        self.assertEqual(recipe.training_policy.loss_handler, 'pairwise_bernoulli_probability')
        self.assertEqual(recipe.target_policy.component_target_id, 'daily_terminal_mae_gated_t40_v1')
        self.assertEqual(STRATEGY_DL_SOURCES['CONT13CX_ROLL']['experiment_profile'], spec.profile_name)
        from core.strategy_compare_policy import (
            get_strategy_comparison_settings, get_strategy_multi_seed_robustness_settings,
            get_strategy_compare_model_bindings,
        )
        from config.strategy_compare import (
            STRATEGY_COMPARE_CURRENT_OOS_PROFILE, STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE,
            STRATEGY_COMPARE_CURRENT_OOS_ROBUSTNESS,
            STRATEGY_COMPARE_CURRENT_ROLLING_ROBUSTNESS,
        )
        for arm_id, mode, entry in [
            ('C147', 'resource-aware-continuous-score-constrained-optimal', 'Breakout'),
            ('C148', 'resource-aware-continuous-score-no-k-no-r0', 'Breakout'),
            ('C149', 'resource-aware-continuous-score-constrained-optimal', 'Daily'),
            ('C150', 'resource-aware-continuous-score-no-k-no-r0', 'Daily')]:
            arm = STRATEGY_COMPARE_ARMS[arm_id]
            self.assertEqual(arm['dl_id'], 'CONT13CX_ROLL')
            self.assertEqual(arm['dl_runtime_mode'], mode)
            self.assertEqual(arm.get('entry_strategy', 'Breakout'), entry)
            self.assertNotIn('positive_dl_id', arm['dl_runtime_options'])
        for profile in (STRATEGY_COMPARE_CURRENT_OOS_PROFILE, STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE):
            settings = get_strategy_comparison_settings(profile)
            current = {arm.arm_id for arm in settings.enabled_arms}
            self.assertTrue({'C147', 'C148', 'C145', 'C146'}.issubset(current))
            self.assertTrue({'C149', 'C150'}.isdisjoint(current))
            self.assertTrue(all('C149' not in c.contrast_id and 'C150' not in c.contrast_id
                                for c in settings.enabled_contrasts))
        for profile in (STRATEGY_COMPARE_CURRENT_OOS_ROBUSTNESS,
                        STRATEGY_COMPARE_CURRENT_ROLLING_ROBUSTNESS):
            settings = get_strategy_multi_seed_robustness_settings(profile)
            self.assertTrue({'C149', 'C150'}.isdisjoint(set(settings.stochastic_arm_ids)))
        cx_bindings = next(row for row in get_strategy_compare_model_bindings()
                           if row['model_id'] == 'MR-13CX')
        self.assertEqual(set(cx_bindings['strategy_arm_ids']), {'C147', 'C148'})

    def test_target_overview_maps_both_models_to_same_t40_truth(self):
        definition = get_formal_oracle_definition(get_active_audit_module_id())
        comparisons = resolve_current_oracle_comparisons(definition.source)
        targets = (str(definition.source['baseline_arm_id']), *(oracle_comparison_key(i) for i in comparisons))
        score_targets = {}
        for item in comparisons:
            source = item
            if item.get('score_source'):
                source = next(i for i in comparisons if i['target'] == item['score_source'])
            score_targets[oracle_comparison_key(item)] = (
                None if item.get('positive_gate') else oracle_supervised_target_id(source)
            )
        bindings = _strategy_bindings(target_names=targets, score_targets=score_targets, definition=definition)
        profile_cx = DAILY_UNIVERSAL_TERMINAL_MAE_GATED_T40_PROBABILISTIC_PROFILE
        for strategy, selector, cx, cj in [
            ('Breakout', 'exact', 'C147', 'C107'),
            ('Breakout', 'direct', 'C148', 'C109'),
        ]:
            key = next(oracle_comparison_key(i) for i in comparisons if
                       i['target'] == ('T40' if selector == 'exact' else 'T40_Direct') and str(i.get('strategy', 'Breakout')) == strategy
                       and i['selector'] == selector)
            self.assertEqual(bindings[(key, profile_cx)], cx)
            self.assertIn(cj, bindings.values())
        for selector, cj in (('exact', 'C145'), ('direct', 'C146')):
            key = next(oracle_comparison_key(i) for i in comparisons if
                       i['target'] == ('T40' if selector == 'exact' else 'T40_Direct')
                       and str(i.get('strategy', 'Breakout')) == 'Daily'
                       and i['selector'] == selector)
            self.assertNotIn((key, profile_cx), bindings)
            self.assertIn(cj, bindings.values())
        from pathlib import Path
        from services.research.target_dl_overview import collect_target_dl_overview
        rows = collect_target_dl_overview(Path(__file__).resolve().parents[2])['rows']
        self.assertTrue(all(not (row['model'] == 'MR-13CX' and row['entry_strategy'] == 'Daily')
                            for row in rows))


if __name__ == '__main__':
    unittest.main()
