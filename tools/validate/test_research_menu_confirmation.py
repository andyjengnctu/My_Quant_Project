"""Show canonical status before confirmation; replan after the user accepts."""
from contextlib import redirect_stdout, ExitStack
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from core.strategy_compare_policy import get_strategy_rolling_test_modes
from filters.breakout_quality import source_inventory
from services.research import breakout_quality_application as model
from services.research import strategy_multi_seed_robustness as strategy


class ResearchMenuConfirmationTests(unittest.TestCase):
    def test_menu_delegates_confirmation_to_the_selected_workflow(self):
        for choice in ('3','4','5','6'):
            with self.subTest(choice=choice), redirect_stdout(io.StringIO()), \
                 patch.object(model,'_recover_pending_model_fit_repairs',return_value=0), \
                 patch.object(model,'_run_configured_model_comparison',return_value=0) as compare, \
                 patch.object(model,'_run_configured_model_robustness',return_value=0) as robust, \
                 patch('builtins.input',side_effect=[choice,'0']):
                self.assertEqual(model.run_model_training_menu(),0)
                selected = compare if choice in {'3','4'} else robust
                selected.assert_called_once_with('apps/research.py model',rolling=choice in {'4','6'},confirm=True)

    def test_model_status_precedes_cancel_and_accept_replans_after_confirmation(self):
        profile = model.get_breakout_quality_model_test_settings().model_profiles[:1]
        ready = SimpleNamespace(source_contract=object(),evidence_payload={},display_action='REUSE',
                                display_reason='READY',evidence_ready=True)
        for robust in (False,True):
            for rolling in (False,True):
                for answer in ('0',''):
                    name = '_run_configured_model_robustness' if robust else '_run_configured_model_comparison'
                    original = getattr(model,name)
                    with self.subTest(robust=robust,rolling=rolling,answer=answer), ExitStack() as stack:
                        output=stack.enter_context(redirect_stdout(io.StringIO()))
                        stack.enter_context(patch.object(model,'get_breakout_quality_model_test_settings',
                                                       return_value=SimpleNamespace(model_profiles=profile)))
                        stack.enter_context(patch.object(model,'_recover_pending_model_fit_repairs',return_value=0))
                        stack.enter_context(patch.object(model,'_resolve_continuous_model_readiness',return_value=ready))
                        stack.enter_context(patch.object(model,'_plan_model_robustness_snapshot',return_value=((),ready)))
                        stack.enter_context(patch.object(model,'plan_fit_origin_repairs',return_value=((),())))
                        stack.enter_context(patch.object(model,'_print_model_action_status',side_effect=lambda rows: print('REUSE status')))
                        render=stack.enter_context(patch.object(model,'_render_standard_model_comparison'))
                        producer=stack.enter_context(patch.object(model,'_prepare_model_robustness_seed_batch'))
                        def confirm(_prompt):
                            self.assertIn('REUSE status',output.getvalue())
                            self.assertIsNone(source_inventory._READ_SCOPE.get())
                            return answer
                        stack.enter_context(patch('builtins.input',side_effect=confirm))
                        # Observe the accepted call's fresh resolution boundary. The first
                        # invocation uses the real owner through the captured reference.
                        rerun=stack.enter_context(patch.object(model,name,return_value=7))
                        self.assertEqual(original('model',rolling=rolling,confirm=True),7 if answer=='' else 0)
                        if answer=='':
                            rerun.assert_called_once_with('model',rolling=rolling,summary_output=None,_show_plan=False)
                        else:
                            rerun.assert_not_called()
                        render.assert_not_called()
                        producer.assert_not_called()

    def test_eof_and_invalid_input_do_not_authorize_model_work(self):
        for response in (EOFError(),'invalid'):
            with self.subTest(response=response),patch('builtins.input',
                    side_effect=response if isinstance(response,Exception) else None,return_value=response):
                self.assertFalse(model._confirm_model_test_workflow())

    def test_strategy_robustness_cancel_follows_plan_and_never_executes(self):
        for mode in get_strategy_rolling_test_modes():
            with self.subTest(mode=mode),redirect_stdout(io.StringIO()) as output, \
                 patch.object(strategy,'collect_artifact_status',return_value={}) as inspect, \
                 patch.object(strategy,'_render_robustness_execution_plan',return_value=('REUSE status',{'overall_status':'READY'})) as plan, \
                 patch.object(strategy,'_execute_resolved_robustness_plan') as execute:
                def confirm(_prompt):
                    self.assertIn('REUSE status',output.getvalue())
                    self.assertIsNone(source_inventory._READ_SCOPE.get())
                    return '0'
                with patch('builtins.input',side_effect=confirm):
                    self.assertEqual(strategy.run_multi_seed_robustness(robustness_id=str(mode['robustness_id'])),{})
                inspect.assert_called_once()
                plan.assert_called_once()
                execute.assert_not_called()

    def test_strategy_acceptance_rechecks_changed_inputs_before_execution(self):
        events=[]; generation='before'
        def inspect(**kwargs):
            self.assertIsNotNone(source_inventory._READ_SCOPE.get())
            events.append('inspect '+generation)
            return {'generation':generation}
        def confirm(_prompt):
            nonlocal generation
            self.assertEqual(events,['inspect before'])
            self.assertIsNone(source_inventory._READ_SCOPE.get())
            generation='after'; events.append('confirm'); return ''
        def execute(**kwargs):
            self.assertIsNone(source_inventory._READ_SCOPE.get())
            self.assertEqual(kwargs['status'],{'generation':'after'})
            events.append('execute'); return {'accepted':True}
        with redirect_stdout(io.StringIO()),patch('builtins.input',side_effect=confirm), \
             patch.object(strategy,'collect_artifact_status',side_effect=inspect), \
             patch.object(strategy,'_render_robustness_execution_plan',return_value=('REUSE',{'overall_status':'READY','work_plan':object()})), \
             patch.object(strategy,'_execute_resolved_robustness_plan',side_effect=execute):
            self.assertEqual(strategy.run_multi_seed_robustness(),{'accepted':True})
        self.assertEqual(events,['inspect before','confirm','inspect after','execute'])


if __name__=='__main__':
    unittest.main()
