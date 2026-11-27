"""Economic, causal and Research integration regressions for holding decisions."""
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
import numpy as np
import pandas as pd

from core.entry_plans import build_position_from_entry_fill
from core.position_exit_policy import bind_position_exit_provider, request_position_early_exit
from core.position_step import execute_bar_step, SETTLEMENT_BASIS_LEDGER_NET
from filters.breakout_quality.holding_value import (
    HoldingFeatures, episode_split, next_executable_exit, position_context,
)
from tools.validate.synthetic_daily_trade40_cases import fixture


def position_fixture():
    days, frame, params, _ = fixture(count=360, start="2018-01-01", rising=False)
    frame.loc[:, ["Open", "High", "Low", "Close"]] = [100., 101., 99., 100.]
    position = build_position_from_entry_fill(
        100., 1000, 80., 80., params, target_price=150., entry_atr=5.,
        ticker="2330", trade_date=days[300])
    return days, frame, params, position


@contextmanager
def holding_research_fixture():
    """Exercise the retained capability without requiring an active experiment."""
    from config import research_summary
    rows = tuple(
        {**row, "enabled": True} if any(
            binding.get("exit_model") == "HV1" for binding in row.get("bindings", ())
        ) else ({**row, "enabled": False} if any(binding.get("exit_model")
            for binding in row.get("bindings", ())) else row) for row in research_summary.RESEARCH_SUMMARY_ROWS
    )
    with patch.object(research_summary, "RESEARCH_SUMMARY_ROWS", rows):
        yield


class HoldingValueTests(TestCase):
    def test_artifact_publication_recovers_from_transient_windows_lock(self):
        import json
        import os
        from services.breakout_quality.holding_value_model import _write_json
        original_replace = os.replace
        attempts = []
        def temporarily_locked(source, target):
            attempts.append((source, target))
            if len(attempts) <= 2:
                raise PermissionError("temporary scanner lock")
            return original_replace(source, target)
        with TemporaryDirectory() as folder:
            path = Path(folder) / "receipt.json"
            _write_json(path, {"completed": 1})
            with patch("core.file_integrity.os.replace", side_effect=temporarily_locked), patch("core.file_integrity.time.sleep"):
                _write_json(path, {"completed": 2})
            self.assertEqual(len(attempts), 3)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"completed": 2})
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_rolling_fallbacks_require_current_position_fit_and_decisions(self):
        import json
        from core.strategy_compare_policy import get_strategy_comparison_settings
        from services.research.strategy_compare_reuse import (
            _pair_fit_lineages_match, _position_model_pair_evidence_matches,
        )
        settings = get_strategy_comparison_settings("broker_cash_dl_conversion_rolling")
        arm = settings.arms["C205"]
        current = {"position_model:HV1:42": {"sha256": "a" * 64}}
        old = {"position_model:HV1:42": {"sha256": "b" * 64}}
        self.assertFalse(_pair_fit_lineages_match(settings=settings, on_arm=arm, current=current, published=old))
        self.assertFalse(_pair_fit_lineages_match(settings=settings, on_arm=arm, current={}, published={}))
        self.assertTrue(_pair_fit_lineages_match(settings=settings, on_arm=arm, current=current, published=current))
        self.assertTrue(_pair_fit_lineages_match(settings=settings, on_arm=settings.arms["C123"], current=current, published=old))
        with TemporaryDirectory() as folder:
            path = Path(folder) / "position_model_decisions.json"
            def matches():
                return _position_model_pair_evidence_matches(pair_dir=Path(folder), on_arm=arm, artifact_identities=current)
            evidence = {"model_manifest_sha256": "b" * 64, "seed": 42, "contract": {"model_id": "HV1"}}
            path.write_text(json.dumps(evidence), encoding="utf-8")
            self.assertFalse(matches())  # rejected even if a parent report claims the new FIT
            evidence["model_manifest_sha256"] = "a" * 64
            path.write_text(json.dumps(evidence), encoding="utf-8")
            self.assertTrue(matches())

    def test_counterfactual_uses_canonical_fees_and_next_open(self):
        days, frame, params, pos = position_fixture()
        frame.loc[days[302], ["Open", "High", "Low", "Close"]] = [103., 105., 102., 104.]
        atr = np.full(len(frame), 5.)
        original = deepcopy(pos)
        pnl, date = next_executable_exit(pos, frame=frame, information_pos=301, params=params,
                                        signal_cache=lambda day: (atr,))
        expected, _, _, events = execute_bar_step(
            deepcopy(pos), 5., True, 100., 103., 105., 102., 104., 1_000_000., params,
            current_date=days[302], y_high=101., return_milli=True,
            settlement_basis=SETTLEMENT_BASIS_LEDGER_NET)
        self.assertIn("IND_SELL", events)
        self.assertEqual(pnl, expected["realized_pnl_milli"] / 1000.)
        self.assertEqual(date, days[302])
        self.assertEqual(pos["qty"], original["qty"])
        self.assertNotEqual(pnl, 3000.)  # fees/tax are not ignored

    def test_blocked_exit_waits_without_fictitious_end_fill(self):
        days, frame, params, pos = position_fixture()
        frame.loc[days[302], "Volume"] = 0.
        atr = np.full(len(frame), 5.)
        result = next_executable_exit(pos, frame=frame, information_pos=301, params=params,
                                      signal_cache=lambda day: (atr,))
        self.assertEqual(result[1], days[303])
        frame.loc[days[302]:, "Volume"] = 0.
        self.assertIsNone(next_executable_exit(pos, frame=frame, information_pos=301, params=params,
                                              signal_cache=lambda day: (atr,)))

    def test_pending_stop_has_priority_over_discretionary_exit(self):
        days, frame, params, pos = position_fixture()
        pos["pending_exit_action"] = "STOP"
        pos["pending_exit_trigger_price"] = 80.
        observed, _, _, events = execute_bar_step(
            deepcopy(pos), 5., True, 100., 100., 101., 99., 100., 1_000_000., params,
            current_date=days[302], y_high=101., return_milli=True,
            settlement_basis=SETTLEMENT_BASIS_LEDGER_NET)
        self.assertIn("STOP", events); self.assertNotIn("IND_SELL", events)
        self.assertEqual(observed["qty"], 0)

    def test_provider_has_no_current_prices_and_request_persists(self):
        days, _, params, pos = position_fixture()
        params = replace(params, use_breakout_quality_ranking=True)
        calls = []
        def prediction(**kwargs):
            calls.append(kwargs)
            self.assertNotIn("t_open", kwargs)
            kwargs["position"]["qty"] = 0
            return -0.1
        arguments = dict(ticker="2330", position=pos, information_date=days[301], execution_date=days[302],
                         completed_atr=5., completed_high=101., params=params)
        with bind_position_exit_provider(prediction):
            self.assertTrue(request_position_early_exit(**arguments))
            self.assertTrue(request_position_early_exit(**{**arguments, "execution_date": days[303]}))
        self.assertEqual(len(calls), 1)
        self.assertEqual(pos["qty"], 1000)
        self.assertFalse(request_position_early_exit(**arguments))
        with bind_position_exit_provider(prediction), self.assertRaises(ValueError):
            request_position_early_exit(**{**arguments, "execution_date": days[301]})

    def test_baseline_and_natural_exit_never_consume_model(self):
        days, _, params, pos = position_fixture()
        def forbidden(**kwargs):
            raise AssertionError("Baseline/natural exit must not consult learned policy")
        kwargs = dict(ticker="2330", position=pos, information_date=days[301], execution_date=days[302],
                      completed_atr=5., completed_high=101., params=params)
        with bind_position_exit_provider(forbidden):
            self.assertFalse(request_position_early_exit(**kwargs))
            self.assertFalse(request_position_early_exit(**{**kwargs,
                "params": replace(params, use_breakout_quality_ranking=True), "natural_exit": True}))

    def test_features_use_only_completed_prices_and_known_state(self):
        days, frame, params, pos = position_fixture()
        market = frame.copy()
        rows = pd.DataFrame([{"ticker": "2330", "information_date": days[301], "state": [0.] * 6}])
        original = HoldingFeatures({"2330": frame}, market).batch(rows)[0]
        frame.loc[days[302]:, :] *= 17
        market.loc[days[302]:, :] *= 31
        changed = HoldingFeatures({"2330": frame}, market).batch(rows)[0]
        np.testing.assert_array_equal(original, changed)
        self.assertIsNone(HoldingFeatures({"2330": frame}, market).coordinates("2330", days[298]))
        before = deepcopy(pos)
        class SharedMarketContext:
            def __deepcopy__(self, memo):
                raise AssertionError("Position inputs must not copy the shared market graph")
        pos["_entry_context"] = SharedMarketContext()
        state = position_context(pos, frame=frame, information_pos=301, completed_atr=5., completed_high=101., params=params)
        self.assertEqual(state.shape, (6,))
        self.assertEqual(pos["trailing_stop"], before["trailing_stop"])

    def test_position_rank_binding_can_activate_without_mutating_origin_params(self):
        days, _, params, pos = position_fixture()
        pos["use_breakout_quality_ranking"] = True
        with bind_position_exit_provider(lambda **kwargs: -0.3):
            self.assertTrue(request_position_early_exit(
                ticker="2330", position=pos, information_date=days[301], execution_date=days[302],
                completed_atr=5., completed_high=101., params=params))
        self.assertFalse(params.use_breakout_quality_ranking)

    def test_holding_source_change_invalidates_only_its_own_seed(self):
        from services.research.strategy_robustness_units import scientific_unit_signature
        from tools.validate.synthetic_strategy_robustness_unit_cases import _registered_fixture
        with holding_research_fixture():
            _, _, arms, seeds, contract = _registered_fixture()
        modified = deepcopy(contract)
        affected = next(a for a in modified["stochastic_arms"] if a.get("position_model_sources"))
        affected["position_model_sources"][str(seeds[0])]["manifest_sha256"] = "different-weights"
        differences = {(a.arm_id, s) for a in arms for s in seeds
                       if scientific_unit_signature(contract=contract, arm_id=a.arm_id, seed=s) !=
                          scientific_unit_signature(contract=modified, arm_id=a.arm_id, seed=s)}
        self.assertEqual(differences, {(affected["arm_id"], seeds[0])})

    def test_missing_position_dependencies_are_planned_once_per_seed(self):
        from core.strategy_compare_policy import get_strategy_comparison_settings
        from core.research_orchestration import ResearchArtifactPlan
        from services.research.position_model_dependency import append_position_model_actions
        with holding_research_fixture():
            settings = get_strategy_comparison_settings("broker_cash_dl_conversion_rolling")
        with TemporaryDirectory() as directory, patch(
            "services.research.position_model_dependency.inspect_holding_model",
            side_effect=lambda root, seed, config: {"ready": False,
                "path": str(Path(root) / str(seed) / "manifest.json"), "reason": "unbuilt"},
        ):
            actions, identities = [], {}
            append_position_model_actions(root=Path(directory), settings=settings,
                                          actions=actions, identities=identities, seeds=(11, 12, 13, 14))
            self.assertEqual(len(actions), 4)  # Exact and Direct share each fit
            self.assertTrue(all(a.action == "BUILD" for a in actions))
            self.assertEqual(ResearchArtifactPlan.from_actions(actions).overall_status, "PREPARABLE")
            inactive = replace(settings, arms={key: replace(arm, enabled=False)
                if (arm.dl_runtime_options or {}).get("position_exit_model") else arm
                for key, arm in settings.arms.items()})
            actions, identities = [], {}
            append_position_model_actions(root=Path(directory), settings=inactive,
                                          actions=actions, identities=identities, seeds=(11, 12, 13, 14))
            self.assertEqual(actions, [])
            self.assertEqual(identities, {})

    def test_whole_episode_purge_and_natural_maturity(self):
        rows = pd.DataFrame([
            {"episode": "old", "entry_date": "2019-01-01", "maturity_date": "2019-05-01"},
            {"episode": "old", "entry_date": "2019-01-01", "maturity_date": "2019-05-01"},
            {"episode": "cross", "entry_date": "2019-12-01", "maturity_date": "2020-02-01"},
            {"episode": "val", "entry_date": "2020-03-01", "maturity_date": "2020-06-01"},
            {"episode": "open", "entry_date": "2020-03-01", "maturity_date": "2021-01-01"},
        ])
        train, valid, final = episode_split(rows, cutoff="2021-01-01", validation_start="2020-01-01")
        self.assertEqual(train.tolist(), [0, 1]); self.assertEqual(valid.tolist(), [3])
        self.assertEqual(final.tolist(), [0, 1, 2, 3])

    def test_observer_cannot_change_canonical_natural_outcome(self):
        from filters.breakout_quality.trade_path_label import build_signal_cache, simulate_realized_trade_path_label
        days, frame, params, _ = fixture(count=150)
        args = dict(ticker="2330", signal_pos=48, params=params,
                    teacher_effective_date=str(days[48].date()), precomputed_signals=build_signal_cache(frame, params, ticker="2330"))
        original = simulate_realized_trade_path_label(frame, **args)
        seen = []
        def observe(**kwargs):
            seen.append(kwargs["information_pos"]); kwargs["position"]["qty"] = 0
        changed = simulate_realized_trade_path_label(frame, **args, holding_state_observer=observe)
        self.assertEqual(original, changed)
        self.assertGreater(len(seen), 0)
        self.assertGreaterEqual(min(seen), 49)

    def test_research_two_arms_keep_entry_controls_and_no_oracle(self):
        from core.strategy_compare_policy import get_strategy_comparison_settings
        from core.research_summary import unresolved_oracle_strategy_contracts
        from core.strategy_compare_registry import STRATEGY_COMPARE_ARMS
        from services.research.position_model_dependency import position_model_bindings
        with holding_research_fixture():
            settings = get_strategy_comparison_settings("broker_cash_dl_conversion_rolling")
            unresolved = unresolved_oracle_strategy_contracts(arm_registry=STRATEGY_COMPARE_ARMS)
        self.assertEqual(set(position_model_bindings(settings)), {"HV1"})
        for treatment, control in (("C205", "C123"), ("C206", "C124")):
            left, right = settings.arms[treatment], settings.arms[control]
            self.assertTrue(left.enabled and right.enabled)
            self.assertEqual(left.dl_id, right.dl_id)
            self.assertEqual(left.param_source, right.param_source)
            self.assertEqual(left.dl_runtime_mode, right.dl_runtime_mode)
            self.assertEqual({k: v for k, v in left.dl_runtime_options.items() if k != "position_exit_model"}, right.dl_runtime_options)
        self.assertTrue(settings.arms["C106"].enabled)
        self.assertTrue({"C205", "C206"} <= set(unresolved))
        self.assertNotIn("C203", STRATEGY_COMPARE_ARMS)

    def test_real_fit_exports_only_mature_states_and_validates_before_refit(self):
        from services.breakout_quality.holding_value_model import _fit_fold
        from filters.breakout_quality.holding_value import CONTRACT
        rows = pd.DataFrame([{
            "ticker": "2330", "information_date": "2019-05-01" if i < 260 else "2020-05-01",
            "entry_date": "2019-01-01" if i < 260 else "2020-01-01",
            "maturity_date": "2019-06-01" if i < 260 else "2020-06-01",
            "episode": str(i), "target": float((i % 3) - 1), "state": [float(i % 3)] * 6,
        } for i in range(330)])
        class Features:
            def batch(self, batch):
                return np.zeros((len(batch), 300, 10), np.float32), np.asarray(batch.state.tolist(), np.float32)
        with TemporaryDirectory() as folder, patch.dict(CONTRACT, {"max_epochs": 1}):
            result = _fit_fold(rows, Features(), start=pd.Timestamp("2021-01-01"), end=pd.Timestamp("2021-12-31"),
                               seed=42, directory=Path(folder), fit_identity="fixture")
            self.assertEqual(result["train_states"], 260)
            self.assertEqual(result["validation_states"], 70)
            self.assertEqual(result["refit_states"], 330)
            self.assertEqual(result["selected_epochs"], 1)
            self.assertTrue((Path(folder) / result["weights"]).is_file())
            self.assertEqual(set(result["validation"]), {"dl", "linear", "always_hold"})
