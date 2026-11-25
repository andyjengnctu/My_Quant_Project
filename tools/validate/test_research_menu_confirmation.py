"""Interactive confirmation precedes expensive inspection and producer work."""
from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import patch

from core.strategy_compare_policy import get_strategy_rolling_test_modes
from filters.breakout_quality import source_inventory
from services.research import breakout_quality_application as model
from services.research import strategy_multi_seed_robustness as strategy


class ResearchMenuConfirmationTests(unittest.TestCase):
    def test_model_test_cancel_never_enters_inspection_or_execution(self):
        for choice in ("3", "4", "5", "6"):
            with self.subTest(choice=choice), redirect_stdout(io.StringIO()) as output, \
                 patch.object(model, "_recover_pending_model_fit_repairs", return_value=0), \
                 patch.object(model, "_run_configured_model_comparison") as compare, \
                 patch.object(model, "_run_configured_model_robustness") as robust, \
                 patch("builtins.input", side_effect=[choice, "0", "0"]):
                self.assertEqual(model.run_model_training_menu(), 0)
                compare.assert_not_called()
                robust.assert_not_called()
                self.assertIn("執行確認", output.getvalue())
                if choice in {"5", "6"}:
                    self.assertIn("Benchmark seeds", output.getvalue())

    def test_model_test_confirmation_dispatches_once_to_the_selected_work(self):
        for choice in ("3", "4", "5", "6"):
            events = []
            answers = iter((choice, "", "0"))

            def answer(prompt):
                events.append("confirm" if "開始檢查" in prompt else "menu")
                return next(answers)

            def execute(*args, **kwargs):
                self.assertEqual(events, ["menu", "confirm"])
                events.append("execute")
                self.assertEqual(kwargs["rolling"], choice in {"4", "6"})
                self.assertIsNone(source_inventory._READ_SCOPE.get())
                return 0

            with self.subTest(choice=choice), redirect_stdout(io.StringIO()), \
                 patch.object(model, "_recover_pending_model_fit_repairs", return_value=0), \
                 patch.object(model, "_run_configured_model_comparison", side_effect=execute) as compare, \
                 patch.object(model, "_run_configured_model_robustness", side_effect=execute) as robust, \
                 patch("builtins.input", side_effect=answer):
                self.assertEqual(model.run_model_training_menu(), 0)
                self.assertEqual(events, ["menu", "confirm", "execute", "menu"])
                self.assertEqual(compare.call_count, int(choice in {"3", "4"}))
                self.assertEqual(robust.call_count, int(choice in {"5", "6"}))

    def test_model_confirmation_eof_or_invalid_input_does_not_authorize_work(self):
        for response in (EOFError(), "invalid"):
            with self.subTest(response=response), redirect_stdout(io.StringIO()), \
                 patch("builtins.input", side_effect=response if isinstance(response, Exception) else None,
                       return_value=response):
                self.assertFalse(model._confirm_model_test_workflow(rolling=True, robustness=True))

    def test_strategy_robustness_cancel_precedes_all_artifact_inspection(self):
        for mode in get_strategy_rolling_test_modes():
            with self.subTest(mode=mode), redirect_stdout(io.StringIO()), \
                 patch.object(strategy, "collect_artifact_status") as inspect, \
                 patch.object(strategy, "_render_robustness_execution_plan") as plan, \
                 patch.object(strategy, "_execute_resolved_robustness_plan") as execute, \
                 patch("builtins.input", return_value="0") as prompt:
                self.assertEqual(strategy.run_multi_seed_robustness(
                    robustness_id=str(mode["robustness_id"]), confirm=True), {})
                self.assertEqual(prompt.call_count, 1)
                inspect.assert_not_called()
                plan.assert_not_called()
                execute.assert_not_called()

    def test_strategy_confirmation_checks_current_plan_before_leaving_read_scope_to_execute(self):
        identifier = str(get_strategy_rolling_test_modes()[0]["robustness_id"])
        events = []

        def answer(_prompt):
            self.assertEqual(events, [])
            self.assertIsNone(source_inventory._READ_SCOPE.get())
            events.append("confirm")
            return ""

        def inspect(**kwargs):
            self.assertEqual(events, ["confirm"])
            self.assertIsNotNone(source_inventory._READ_SCOPE.get())
            events.append("inspect")
            return {}

        def execute(**kwargs):
            self.assertEqual(events, ["confirm", "inspect"])
            self.assertIsNone(source_inventory._READ_SCOPE.get())
            events.append("execute")
            return {"accepted": True}

        with redirect_stdout(io.StringIO()), patch("builtins.input", side_effect=answer), \
             patch.object(strategy, "collect_artifact_status", side_effect=inspect), \
             patch.object(strategy, "_render_robustness_execution_plan",
                          return_value=("plan", {"overall_status": "READY", "work_plan": object()})), \
             patch.object(strategy, "_execute_resolved_robustness_plan", side_effect=execute):
            self.assertEqual(strategy.run_multi_seed_robustness(robustness_id=identifier), {"accepted": True})
        self.assertEqual(events, ["confirm", "inspect", "execute"])


if __name__ == "__main__":
    unittest.main()
