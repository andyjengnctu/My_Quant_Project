"""Canonical upstream planning across independent teacher seeds."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from core.research_orchestration import ResearchArtifactAction, ResearchArtifactPlan
from services.breakout_quality.trade_path_target_source import TeacherDependency, TradePathSource


class ResearchSeedDependencyQueueTests(unittest.TestCase):
    def _run(self, *, ready_seeds=(), blocked_seed=None, fail_batch=False, incomplete=False):
        from services.research import breakout_quality_application as app
        from services.breakout_quality import trade_path_target_source as gateway

        seeds = (202, 101)
        state = {"dataset": False, "current": set(), "history": set(), "early": set(ready_seeds)}
        events = []
        observed = []
        settings = SimpleNamespace(experiment_profile="fixture", seed=42)

        def plan(_settings, *, seed, **_kwargs):
            rows = []
            for name, priority in (("dataset", 10), ("current", 30), ("history", 40), ("early", 50)):
                ready = state[name] if name == "dataset" else seed in state[name]
                action = "REUSE" if ready else "BUILD"
                if seed == blocked_seed and name == "early":
                    action = "BLOCKED"
                rows.append(ResearchArtifactAction(
                    action_id=name, artifact_key=name, action=action,
                    builder_type=None if action in {"REUSE", "BLOCKED"} else name,
                    description=name, path=f"seed/{seed}/{name}",
                    execution_priority=priority,
                    dependencies=() if name == "dataset" else ("dataset",),
                ))
            return ResearchArtifactPlan.from_actions(rows)

        def single(*, builder_type, seed, **_kwargs):
            self.assertNotEqual(builder_type, "early")
            events.append((builder_type, seed))
            if builder_type == "dataset":
                state[builder_type] = True
            else:
                state[builder_type].add(seed)
            return True

        def batch(*, seeds, **_kwargs):
            self.assertTrue(state["dataset"])
            self.assertEqual(state["current"], {101, 202})
            self.assertEqual(state["history"], {101, 202})
            events.append(("batch", seeds))
            if fail_batch:
                raise RuntimeError("canonical producer failure")
            if not incomplete:
                state["early"].update(seeds)

        source = TradePathSource("fixture", lambda **kwargs: {}, (
            TeacherDependency("early", "early", 50, lambda **kwargs: {},
                              lambda **kwargs: None, batch),
        ))
        with (
            patch.object(app, "_collect_continuous_research_input_plan", side_effect=plan),
            patch.object(gateway, "execute_trade_path_dependency", side_effect=single),
            patch.object(gateway, "registered_trade_path_source", return_value=source),
            patch.object(app, "get_breakout_quality_experiment_profile",
                         return_value=SimpleNamespace(training_sample_scope="fixture")),
            redirect_stdout(StringIO()),
        ):
            code = app._prepare_continuous_research_inputs(
                "research", settings, seeds=seeds, execution_observer=observed.append,
            )
        return code, events, state, observed

    def test_shared_prerequisites_finish_before_batch_and_seed_order_is_preserved(self):
        code, events, state, observed = self._run()
        self.assertEqual(code, 0)
        self.assertEqual(sum(name == "dataset" for name, _ in events), 1)
        self.assertEqual(events[-1], ("batch", (202, 101)))
        self.assertEqual(state["early"], {202, 101})
        self.assertEqual(observed, [True])

    def test_verified_seed_is_not_rebuilt_by_batch(self):
        code, events, _state, _observed = self._run(ready_seeds=(202,))
        self.assertEqual(code, 0)
        self.assertEqual(events[-1], ("batch", (101,)))

    def test_any_blocked_seed_stops_all_producers(self):
        code, events, _state, observed = self._run(blocked_seed=101)
        self.assertEqual(code, 2)
        self.assertEqual(events, [])
        self.assertEqual(observed, [])

    def test_batch_failure_never_certifies_preparation(self):
        code, events, state, observed = self._run(fail_batch=True)
        self.assertEqual(code, 2)
        self.assertEqual(events[-1], ("batch", (202, 101)))
        self.assertEqual(state["early"], set())
        self.assertEqual(observed, [])

    def test_successful_process_return_still_requires_canonical_reuse(self):
        code, _events, state, observed = self._run(incomplete=True)
        self.assertEqual(code, 2)
        self.assertEqual(state["early"], set())
        self.assertEqual(observed, [])

    def test_empty_or_single_seed_preserves_existing_producer_path(self):
        from services.research import breakout_quality_application as app
        with patch.object(app, "_prepare_continuous_research_inputs") as prepare:
            observed = []
            for seeds in ((), (101,)):
                self.assertEqual(app._prepare_model_robustness_seed_batch(
                    "research", SimpleNamespace(), seeds=seeds,
                    execution_observer=observed.append,
                ), 0)
        prepare.assert_not_called()
        self.assertEqual(observed, [False, False])

    def test_source_without_batch_capability_keeps_existing_preparation_path(self):
        from services.research import breakout_quality_application as app
        from services.breakout_quality import trade_path_target_source as gateway
        source = TradePathSource("fixture", lambda **kwargs: {}, (
            TeacherDependency("history", "history", 40, lambda **kwargs: {},
                              lambda **kwargs: None),
        ))
        observed = []
        with (
            patch.object(app, "_requires_formal_trade_path_inputs", return_value=True),
            patch.object(gateway, "registered_trade_path_source", side_effect=(None, source)),
            patch.object(app, "_prepare_continuous_research_inputs") as prepare,
        ):
            for _ in range(2):
                self.assertEqual(app._prepare_model_robustness_seed_batch(
                    "research", SimpleNamespace(experiment_profile="fixture"), seeds=(101, 202),
                    execution_observer=observed.append,
                ), 0)
        prepare.assert_not_called()
        self.assertEqual(observed, [False, False])

    def test_invalid_batch_seed_contract_fails_before_planning(self):
        from services.research import breakout_quality_application as app
        with patch.object(app, "_collect_continuous_research_input_plan") as inspect:
            for seeds in ((), (101, 101), (0, 101), (True, 101)):
                with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                    app._prepare_continuous_research_inputs(
                        "research", SimpleNamespace(seed=42), seeds=seeds,
                    )
        inspect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
