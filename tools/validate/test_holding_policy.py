"""Whole-trade objective, PIT, first-stop and conditional authorization checks."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch

from filters.breakout_quality.holding_policy import (
    CONTRACT, expected_episode_gain, episode_outcomes, development_gate, build_policy_network,
)


def episode_rows():
    rows = []
    for i in range(30):
        year = 2016 if i < 20 else 2017
        for age in (0, 1):
            rows.append({"ticker": "2330", "episode": str(i), "entry_date": f"{year}-05-01",
                "information_date": f"{year}-05-0{age+1}", "decision_execution_date": f"{year}-05-0{age+2}",
                "maturity_date": f"{year}-06-01", "natural_exit_date": f"{year}-06-01",
                "early_exit_date": f"{year}-05-0{age+2}", "natural_net_pnl": 200., "early_net_pnl": 100.,
                "initial_risk_money": 100., "target": 1., "decision_allowed": True,
                "state": [float(age), 0., 0., 0., 0., 1.], "feature_index": 1})
    return pd.DataFrame(rows)


class HoldingPolicyTests(TestCase):
    def test_whole_payoff_has_no_inverse_duration_weight(self):
        logits = torch.zeros(101, requires_grad=True)
        payoff = torch.zeros(101); payoff[0] = -10.; payoff[-1] = 1.
        gains = expected_episode_gain(logits, payoff, torch.ones(101, dtype=torch.bool), [100, 1])
        torch.testing.assert_close(gains, torch.tensor([-5., .5]))
        self.assertAlmostEqual(float(gains.mean().detach()), -2.25)
        (-gains.mean()).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad[0]), 0)  # suppress selling the large winner

    def test_survival_and_mask_match_hand_computed_payoff(self):
        logits = torch.zeros(3, requires_grad=True)
        gains = expected_episode_gain(logits, torch.tensor([2., 999., 8.]),
                                      torch.tensor([True, False, True]), [3])
        torch.testing.assert_close(gains, torch.tensor([3.]))  # .5*2 + .5*.5*8
        gains.sum().backward()
        self.assertEqual(float(logits.grad[1]), 0.)

    def test_hard_policy_settles_only_first_request(self):
        rows = episode_rows().iloc[:2].copy()
        rows.loc[rows.index[1], "target"] = -20.
        chosen = episode_outcomes(rows, np.array([0., 8.])).iloc[0]
        self.assertEqual(chosen.gain_r, -1.)
        self.assertEqual(chosen.policy_r, 1.)
        self.assertEqual(chosen.holding_days, 1)
        rows.loc[rows.index[0], "decision_allowed"] = False
        self.assertEqual(episode_outcomes(rows, np.array([0., 8.])).iloc[0].gain_r, 20.)
        self.assertEqual(episode_outcomes(rows, np.array([-10., -10.])).iloc[0].gain_r, 0.)

    def test_same_input_linear_baseline_and_initial_continue(self):
        x, state = torch.randn(3, 300, 10), torch.randn(3, 6)
        for kind in ("dl", "linear"):
            model = build_policy_network(kind)
            p = torch.sigmoid(model(x, state))
            torch.testing.assert_close(p, torch.full((3,), CONTRACT["initial_stop_probability"]))
        self.assertEqual(build_policy_network("linear").head.in_features, x[0].numel() + state.shape[1])

    def test_gate_requires_all_years_economic_gain_and_linear_improvement(self):
        years = {str(y): {"dl": {"mean_gain_r": d}, "linear": {"mean_gain_r": .01}}
                 for y, d in zip((2018, 2019, 2020), (.2, .1, -.1))}
        self.assertTrue(development_gate(years)["passed"])
        years["2018"]["linear"]["mean_gain_r"] = 1.
        self.assertFalse(development_gate(years)["passed"])
        for values in years.values():
            values["dl"]["mean_gain_r"] = 0.
        self.assertFalse(development_gate(years)["passed"])
        with self.assertRaises(ValueError):
            development_gate({"2018": years["2018"]})

    def test_failed_development_cannot_build_formal_fits(self):
        from services.breakout_quality import holding_policy_model as owner
        with TemporaryDirectory() as folder, patch.object(owner, "inspect_development", return_value={
                "complete": True, "passed": False, "reason": "development REJECT"}), patch.object(owner, "sources") as source:
            status = owner.inspect_holding_model(Path(folder), seed=12, config={"mode": "rolling", "dataset": "full"})
            self.assertFalse(status["ready"]); self.assertFalse(status["build_allowed"])
            with self.assertRaisesRegex(ValueError, "development REJECT"):
                owner.ensure_holding_model(Path(folder), seed=12, config={"mode": "rolling", "dataset": "full"})
            source.assert_not_called()

    def test_probability_request_uses_existing_persistent_execution(self):
        from dataclasses import replace
        from core.position_exit_policy import PositionExitDecision, bind_position_exit_provider, request_position_early_exit
        from tools.validate.test_holding_value import position_fixture
        days, _, params, pos = position_fixture()
        params = replace(params, use_breakout_quality_ranking=True)
        with bind_position_exit_provider(lambda **kwargs: PositionExitDecision(True, .5, "exit_probability")):
            self.assertTrue(request_position_early_exit(ticker="2330", position=pos, information_date=days[301],
                execution_date=days[302], completed_atr=5., completed_high=101., params=params))
        self.assertEqual(pos["_model_exit_exit_probability"], .5)
        self.assertNotIn("_model_exit_delta_r", pos)

    def test_complete_samples_preserve_canonical_payoffs_and_future_invariance(self):
        from filters.breakout_quality.holding_value import build_ticker_holding_rows
        from tools.validate.synthetic_daily_trade40_cases import fixture
        days, frame, _, period = fixture(count=390, rising=True)
        frame.iloc[309, :4] -= 7.
        frame.iloc[310:335, :4] += 30.
        frame.iloc[335:, :4] -= 20.
        teacher = SimpleNamespace(schedule=(period,))
        args = {"ticker": "2330", "teacher": teacher, "cutoff": days[365]}
        old = pd.DataFrame(build_ticker_holding_rows(frame=frame, **args))
        new = pd.DataFrame(build_ticker_holding_rows(frame=frame, complete_episode=True, **args))
        self.assertGreater(len(new), 0)
        known = new[new.decision_allowed].merge(old, on=["episode", "information_date"], suffixes=("_new", "_old"))
        self.assertEqual(len(known), len(old))
        np.testing.assert_array_equal(known.target_new, known.target_old)
        np.testing.assert_allclose(new.target, (new.natural_net_pnl-new.early_net_pnl)/new.initial_risk_money)
        self.assertTrue((pd.to_datetime(new.information_date) < pd.to_datetime(new.decision_execution_date)).all())
        changed = frame.copy(); changed.iloc[366:, :4] *= 2
        fresh = pd.DataFrame(build_ticker_holding_rows(frame=changed, complete_episode=True, **args))
        self.assertEqual(new.to_json(), fresh.to_json())

    def test_fit_identity_excludes_future_prices_labels_and_feature_indices(self):
        from services.breakout_quality import holding_policy_model as owner
        rows = episode_rows()
        future = rows.iloc[:2].copy(); future["episode"] = "future"; future["maturity_date"] = "2019-01-01"
        rows = pd.concat((rows, future), ignore_index=True)
        frames = {t: pd.DataFrame(np.ones((5,5)), index=pd.to_datetime([
            "2016-01-01", "2017-01-01", "2017-12-31", "2018-12-31", "2019-01-01"]),
            columns=["Open", "High", "Low", "Close", "Volume"]) for t in ("2330", "0050")}
        def fake_fit(rows, features, **kwargs):
            result = {"fit_identity": kwargs["fit_identity"]}
            owner._write_json(kwargs["directory"] / "fold.json", result)
            return result
        with TemporaryDirectory() as folder, patch.object(owner, "_fit_fold", side_effect=fake_fit), patch.object(
                owner, "implementation_identity", return_value="fixed"), patch.object(
                owner, "slice_formal_mintrade40_pit_source_for_fold", return_value=(None, "prefix")):
            args = dict(seed=42, start=pd.Timestamp("2018-01-01"), rows=rows, features=SimpleNamespace(frames=frames), teacher=None)
            a = owner.fit_for_cutoff(Path(folder), **args)
            rows.loc[rows.episode == "future", "target"] = 999.
            rows["feature_index"] = 97
            for frame in frames.values(): frame.loc["2018-01-01":] *= 4
            b = owner.fit_for_cutoff(Path(folder), **args)
            self.assertEqual(a["fit_identity"], b["fit_identity"])
            rows.loc[0, "target"] += 1
            self.assertNotEqual(a["fit_identity"], owner.fit_for_cutoff(Path(folder), **args)["fit_identity"])

    def test_real_fit_selects_complete_trade_validation_and_retains_linear(self):
        from services.breakout_quality.holding_policy_training import _fit_fold
        class Features:
            def batch(self, rows):
                return np.zeros((len(rows),300,10),np.float32), np.asarray(rows.state.tolist(),np.float32)
        with TemporaryDirectory() as folder, patch.dict(CONTRACT, {
                "max_epochs":1, "min_train_episodes":4, "min_validation_episodes":4}):
            result = _fit_fold(episode_rows(), Features(), start=pd.Timestamp("2018-01-01"), seed=42,
                               directory=Path(folder), fit_identity="synthetic")
            self.assertEqual(result["train_episodes"],20); self.assertEqual(result["validation_episodes"],10)
            self.assertEqual(result["refit_episodes"],30)
            self.assertEqual(set(result["models"]),{"dl","linear"})
            for kind,item in result["models"].items():
                self.assertEqual(item["selected_epochs"],0)  # all early exits lose 1R
                self.assertEqual(item["validation"]["mean_gain_r"],0.)
                self.assertTrue((Path(folder)/item["weights"]).exists())
