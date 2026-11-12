"""Economic and PIT invariants for payoff-sensitive binary decisions."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import TestCase, main

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from filters.breakout_quality.categorical_decision import (
    payoff_training_targets, payoff_cross_entropy,
    payoff_validation_metrics, payoff_trade_decisions,
)
from core.positive_score_admission import (
    positive_score_admitted, requires_positive_score, uses_model_positive_decision,
)


class PayoffDecisionCases(TestCase):
    def test_high_win_rate_with_negative_expected_payoff_learns_cash(self):
        import torch
        old = torch.get_num_threads(); torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old)
        for outcomes, expected in (([1., 1., 1., -10.], 0.), ([-1., -1., -1., 10.], 1.)):
            target = torch.tensor(payoff_training_targets(outcomes, np.arange(4)))
            bias = torch.nn.Parameter(torch.zeros(1, 2))
            optimizer = torch.optim.SGD([bias], lr=.2)
            for _ in range(100):
                optimizer.zero_grad()
                loss = payoff_cross_entropy(torch, bias.expand(4, -1), target)
                loss.backward(); optimizer.step()
            self.assertEqual(payoff_trade_decisions(bias.detach().numpy()).item(), expected)
            self.assertNotEqual(expected, float(np.mean(np.array(outcomes) > 0) > .5))

    def test_weight_normalization_uses_only_explicit_fit_rows(self):
        raw = np.array([2., -6., 0., np.nan, 999999.])
        before = payoff_training_targets(raw, [0, 1, 2])
        raw[3:] = [-1e20, 1e20]
        np.testing.assert_equal(before, payoff_training_targets(raw, [0, 1, 2]))
        np.testing.assert_allclose(before[:3, 1], [.75, 2.25, 0.])
        self.assertTrue(np.isnan(before[3:]).all())
        # Final refit recomputes normalization for its own newly legal cohort.
        np.testing.assert_allclose(payoff_training_targets(raw, [0, 1])[:2, 1], [.5, 1.5])
        for bad in ([0, 0], [-1], [5]):
            with self.assertRaises(ValueError):
                payoff_training_targets(raw, bad)
        with self.assertRaises(ValueError):
            payoff_training_targets([np.nan, 1.], [0, 1])

    def test_zero_payoff_has_no_gradient_and_tie_retains_cash(self):
        import torch
        logits = torch.tensor([[2., 2.], [1., 1.00000001]], dtype=torch.float64)
        self.assertIsNone(payoff_cross_entropy(torch, logits, torch.zeros(2, 2)))
        np.testing.assert_array_equal(payoff_trade_decisions(logits.numpy()), [0., 1.])
        with self.assertRaises(ValueError):
            payoff_trade_decisions([[0., np.nan]])
        with self.assertRaises(ValueError):
            payoff_training_targets([0., 0.], [0, 1])

    def test_epoch_selection_penalizes_large_loss_and_is_currency_scale_invariant(self):
        cash = np.tile([2., 0.], (4, 1))
        trade = cash[:, ::-1]
        outcomes = np.array([1., 1., 1., -10.])
        a = payoff_validation_metrics(cash, outcomes)
        b = payoff_validation_metrics(trade, outcomes)
        self.assertLess(a['cross_entropy'], b['cross_entropy'])
        self.assertLess(a['joint_event_brier'], b['joint_event_brier'])
        self.assertEqual(a, payoff_validation_metrics(cash, outcomes * 1000))

    def test_declared_strategy_has_two_sources_and_no_threshold(self):
        from core.strategy_compare_policy import get_strategy_comparison_settings
        from config.strategy_compare import STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE
        from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
        from filters.breakout_quality.strategy_compare_dl_artifacts import resolve_arm_runtime_dl_source_ids
        settings = get_strategy_comparison_settings(STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE)
        arms = [arm for arm in settings.enabled_arms if uses_model_positive_decision(arm.dl_runtime_options)]
        self.assertTrue(arms)
        for arm in arms:
            options = dict(arm.dl_runtime_options)
            self.assertTrue(requires_positive_score(options))
            self.assertNotIn('positive_probability_threshold', options)
            self.assertEqual(set(resolve_arm_runtime_dl_source_ids(settings, arm)),
                             {arm.dl_id, options['positive_dl_id']})
            recipe = get_continuous_ranker_execution_recipe(settings.dl_sources[options['positive_dl_id']].experiment_profile)
            self.assertEqual(recipe.model_architecture, 'inception_time_v1')
            self.assertEqual(recipe.training_policy.categorical_decision_column, options['positive_score_column'])
            # Only a model-produced binary action at the rank information date can admit.
            payload = dict(available=True, positive_available=True, score_date='2021-01-04',
                           positive_score_date='2021-01-04', positive_probability=1.)
            row = dict(breakout_quality_ranking_options=options, breakout_quality_rank=payload)
            self.assertTrue(positive_score_admitted(row))
            for action in [0., .61, float('nan'), None, 2., -1.]:
                payload['positive_probability'] = action
                self.assertFalse(positive_score_admitted(row))
            payload.update(positive_probability=1., positive_score_date='2021-01-05')
            self.assertFalse(positive_score_admitted(row))
            with self.assertRaises(ValueError):
                requires_positive_score({**options, 'positive_probability_threshold': .5})

    def test_summary_keeps_control_and_does_not_borrow_oracle(self):
        from services.research.target_dl_overview import collect_target_dl_overview
        from core.strategy_compare_registry import STRATEGY_COMPARE_ARMS
        from core.research_summary import unresolved_oracle_strategy_contracts
        pending = unresolved_oracle_strategy_contracts(arm_registry=STRATEGY_COMPARE_ARMS)
        report = collect_target_dl_overview(ROOT)
        for arm_id in pending:
            rows = [row for row in report['rows'] if row['strategy'] == arm_id]
            self.assertEqual(len(rows), 1)
            self.assertIsNone(rows[0]['oracle_romd'])
            self.assertTrue(rows[0]['positive_model'])
            from core.research_summary import research_summary_parameter_dimensions, research_summary_target_label
            from core.strategy_compare_registry import STRATEGY_EXPERIMENTAL_PARAM_SOURCES
            _filter, version, _daily = research_summary_parameter_dimensions(
                STRATEGY_COMPARE_ARMS[arm_id], parameter_sources=STRATEGY_EXPERIMENTAL_PARAM_SOURCES)
            self.assertEqual(rows[0]['target'], research_summary_target_label(
                'Daily_MinTrade_NetPnL_Positive', parameter_version=version))
        self.assertEqual(len([r for r in report['rows'] if r['strategy'] == 'C183']), 1)


if __name__ == '__main__':
    main()
