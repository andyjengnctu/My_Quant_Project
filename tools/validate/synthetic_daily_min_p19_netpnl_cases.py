"""Independent synthetic invariants for the Daily Min/P19 economic Target.

No optimizer/model training or actual Research OOS is executed here.
The P19 value is a real hypothetical Min trade result, never a renamed P18.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from core.audit_registry import oracle_scientific_target_name
from core.breakout_quality_registry import get_continuous_ranker_research_spec
from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
from core.research_summary import (
    get_research_summary_baseline_arm_ids,
    get_research_summary_model_profiles,
    get_research_summary_strategy_arm_ids,
)
from core.strategy_compare_policy import get_strategy_comparison_settings
from core.strategy_compare_registry import STRATEGY_COMPARE_ARMS
from filters.breakout_quality.daily_min_p19_target import compute_daily_min_p19_batch
from services.breakout_quality.daily_min_p19_source_identity import SOURCE_ID
from services.breakout_quality.daily_min_p19_pit_source import materialize_daily_min_p19_pit_labels
from services.breakout_quality.daily_min_p19_training_source import (
    BASELINE_ARM, PROFILE, TEACHER_SEED, historical_params_path, inspect_history,
)
from services.breakout_quality.formal_mintrade40_pit_source import FormalMinTrade40PITBlocked
from services.research.oracle_artifact_contract import source_calculation_contract
from tools.validate.synthetic_daily_full_p20_fulltrade_cases import _fixture_source

RANK = 'daily_universal_min_p19_mintrade_netpnl_full_list_ndcg_pairwise'
POSITIVE = 'daily_universal_min_p19_mintrade_netpnl_positive_binary'
RECIPE = 'daily_trade_path_daily_min_p19_net_pnl'


def _fixture():
    dates, frame, previous = _fixture_source()
    source = replace(
        previous, source_id=SOURCE_ID, param_sha256='b' * 64,
        schedule=tuple(replace(p, source_id=SOURCE_ID) for p in previous.schedule),
    )
    return dates, frame, source


class DailyMinP19NetPnLCases(TestCase):
    def test_retired_p19_absent_from_current_research_but_history_preserved(self):
        from config import research_summary as roster
        from config.strategy_params import STRATEGY_PARAM_PROFILES, ACTIVE_STRATEGY_PARAM_PROFILE
        from core.audit_registry import AUDIT_REUSABLE_REPORTS, resolve_current_oracle_comparisons
        from core.strategy_compare_policy import get_strategy_multi_seed_robustness_settings
        from services.research.target_dl_overview import collect_target_dl_overview

        p19 = {'C166', 'C167', 'C174', 'C175', 'C176'}
        active = set(get_research_summary_strategy_arm_ids())
        self.assertTrue(p19.isdisjoint(active))
        self.assertIn('C166', STRATEGY_COMPARE_ARMS)
        self.assertNotIn('daily_min_p19', get_research_summary_baseline_arm_ids())
        self.assertTrue(all(row.get('enabled') is False for row in roster.RESEARCH_SUMMARY_ROWS
                            if row.get('arm_id') in p19 or any(
                                binding.get('arm_id') in p19
                                for binding in row.get('bindings', ()))))
        self.assertTrue(all('P19' not in row['target'] for row in
                            resolve_current_oracle_comparisons(
                                AUDIT_REUSABLE_REPORTS['breakout_quality']['target_oracle_romd']['source'])))
        active_profiles = get_research_summary_model_profiles()
        self.assertTrue(all(model_id not in {'MR-13DD', 'MR-13DE'}
                            for model_id, profile in active_profiles))
        for mode in ('broker_cash_dl_conversion_oos', 'broker_cash_dl_conversion_rolling'):
            with self.subTest(compare_mode=mode):
                settings = get_strategy_comparison_settings(mode)
                self.assertTrue(all(not settings.arms[arm].enabled for arm in p19))
                robustness = get_strategy_multi_seed_robustness_settings(mode)
                self.assertTrue(p19.isdisjoint(robustness.benchmark_strategy_arm_ids))
                self.assertTrue(p19.isdisjoint(robustness.model_seed_sensitive_arm_ids))
                self.assertTrue(all(c.left not in p19 and c.right not in p19
                                    for c in settings.enabled_contrasts))
        self.assertNotEqual(ACTIVE_STRATEGY_PARAM_PROFILE,
                            'broker_cash_daily_min_four_seed42_current')
        self.assertEqual(STRATEGY_PARAM_PROFILES[
            'broker_cash_daily_min_four_seed42_current']['parameter_set_identity'], 'PARAM-P19')
        # Historical scientific sources and strategy Replay IDs must not be deleted.
        self.assertFalse(STRATEGY_COMPARE_ARMS['C174']['dl_enabled'])
        self.assertTrue(all(STRATEGY_COMPARE_ARMS[arm]['dl_enabled'] for arm in ('C175','C176')))
        for arm in ('C174','C175','C176'):
            self.assertEqual(STRATEGY_COMPARE_ARMS[arm]['param_source'],
                             'daily_min_four_best_broker_cash_rolling_experiment')
            self.assertEqual(STRATEGY_COMPARE_ARMS[arm]['entry_strategy'], 'Daily')
        for arm in ('C174', 'C175', 'C176'):
            self.assertEqual(STRATEGY_COMPARE_ARMS[arm]['param_source'],
                             'daily_min_four_best_broker_cash_rolling_experiment')
            self.assertEqual(STRATEGY_COMPARE_ARMS[arm]['entry_strategy'], 'Daily')
        for arm in ('C175', 'C176'):
            self.assertFalse(STRATEGY_COMPARE_ARMS[arm]['dl_runtime_options']['preserve_k'])
            self.assertFalse(STRATEGY_COMPARE_ARMS[arm]['dl_runtime_options']['preserve_r0'])
        self.assertNotIn('positive_dl_id', STRATEGY_COMPARE_ARMS['C175']['dl_runtime_options'])
        self.assertEqual(STRATEGY_COMPARE_ARMS['C176']['dl_runtime_options']['positive_dl_id'],
                         'CONT13DE_ROLL')

    def test_compare_plan_lists_signed_p19_teachers_before_training(self):
        from filters.breakout_quality import strategy_compare_dl_artifacts as dl
        from services.research import strategy_compare_preparation_status as planning
        from filters.breakout_quality.artifact_dependency_registry import ArtifactReadiness
        from services.breakout_quality import daily_min_p19_training_source as teacher

        settings = get_strategy_comparison_settings('broker_cash_dl_conversion_rolling')
        source = settings.dl_sources['CONT13DD_ROLL']
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / 'outputs' / 'dataset_summary.json'
            status = ArtifactReadiness(
                artifact_type='dataset_core', ready=True, status='READY', path=dataset,
                dependencies=(), producer_work_type='model_training', description='Dataset READY',
            )
            def history(*, project_root, seed):
                self.assertEqual(seed, 42)
                return {'action': 'BUILD', 'path': historical_params_path(project_root),
                        'reason': 'Independent P19 history required'}
            def current(*, project_root, seed):
                self.assertEqual(seed, 42)
                return {'action': 'REUSE', 'path': root / 'current.json',
                        'reason': 'Verified P19 current teacher'}
            actions = []
            with patch.object(dl, 'collect_model_upstream_readiness', return_value=(status,)), \
                 patch.object(teacher, 'inspect_history', side_effect=history), \
                 patch.object(teacher, 'inspect_current', side_effect=current), \
                 patch.object(teacher, 'execute_history', side_effect=AssertionError('Plan must be read-only')):
                keys = dl._collect_model_upstream_dependencies(
                    root=root, settings=settings, source=source, artifact_identities={},
                    actions=actions, cache={},
                )
                teacher_keys = planning._collect_trade_path_teacher_dependencies(
                    root=root, settings=settings, source=source, actions=actions,
                    cache={}, dataset_dependencies=keys,
                )
            self.assertEqual(len(keys), 1)
            self.assertEqual(len(teacher_keys), 2)
            self.assertTrue(any(':daily_min_p19_historical_teacher:seed_42' in key for key in teacher_keys))
            self.assertTrue(any(':daily_min_p19_current_teacher:seed_42' in key for key in teacher_keys))
            by_key = {item.artifact_key: item for item in actions}
            history_node = next(row for row in actions if 'historical_teacher' in row.artifact_key)
            current_node = next(row for row in actions if 'current_teacher' in row.artifact_key)
            self.assertEqual(history_node.action, 'BUILD')
            self.assertEqual(history_node.builder_type, 'canonical_model_upstream')
            self.assertEqual(history_node.producer_work_type, 'model_training')
            self.assertEqual(history_node.dependencies, (keys[0],))
            self.assertEqual(current_node.action, 'REUSE')
            self.assertEqual(current_node.builder_type, None)
            self.assertEqual(len(by_key), len(actions))
            self.assertFalse((root / 'outputs' / 'optimizer').exists())

    def test_compare_teacher_identity_invalid_blocks_instead_of_rebuilding(self):
        from filters.breakout_quality import strategy_compare_dl_artifacts as dl
        from services.research import strategy_compare_preparation_status as planning
        from filters.breakout_quality.artifact_dependency_registry import ArtifactReadiness
        from services.breakout_quality import daily_min_p19_training_source as teacher
        settings = get_strategy_comparison_settings('broker_cash_dl_conversion_rolling')
        source = settings.dl_sources['CONT13DE_ROLL']
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = ArtifactReadiness(
                artifact_type='dataset_core', ready=True, status='READY',
                path=root / 'dataset.json', dependencies=(),
                producer_work_type='model_training', description='READY',
            )
            def state(action):
                return {'action': action, 'path': root / 'teacher.json',
                        'reason': 'Signed teacher mismatch' if action == 'BLOCKED' else 'READY'}
            for expected, actual in (('BLOCKED', 'BLOCKED'), ('REUSE', 'REUSE')):
                actions = []
                with patch.object(dl, 'collect_model_upstream_readiness', return_value=(dataset,)), \
                     patch.object(teacher, 'inspect_history', return_value=state(actual)), \
                     patch.object(teacher, 'inspect_current', return_value=state('REUSE')):
                    keys = dl._collect_model_upstream_dependencies(
                        root=root, settings=settings, source=source,
                        artifact_identities={}, actions=actions, cache={},
                    )
                    planning._collect_trade_path_teacher_dependencies(
                        root=root, settings=settings, source=source, actions=actions,
                        cache={}, dataset_dependencies=keys,
                    )
                node = next(item for item in actions if 'historical_teacher' in item.artifact_key)
                self.assertEqual(node.action, expected)
                self.assertIsNone(node.builder_type)

    def test_compare_valid_pit_reuse_skips_missing_optimizer_teacher(self):
        from core.research_orchestration import ResearchArtifactAction
        from filters.breakout_quality import strategy_compare_dl_artifacts as dl
        from services.research import strategy_compare_preparation_status as planning
        from filters.breakout_quality.artifact_dependency_registry import ArtifactReadiness
        from services.breakout_quality import daily_min_p19_training_source as teacher

        settings = get_strategy_comparison_settings('broker_cash_dl_conversion_rolling')
        source = settings.dl_sources['CONT13DD_ROLL']
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = ArtifactReadiness(
                artifact_type='dataset_core', ready=True, status='READY',
                path=root / 'dataset.json', dependencies=(),
                producer_work_type='model_training', description='READY',
            )
            def fake_model_status(*, dl_id, actions, source_upstream_dependencies, **kwargs):
                actions.append(ResearchArtifactAction(
                    action_id=f'dl:{dl_id}:forward_scores',
                    artifact_key=f'dl:{dl_id}:forward_scores',
                    action='REUSE', builder_type=None, description='verified PIT',
                    path='model.csv', dependencies=source_upstream_dependencies,
                ))
                return {'ready': True}, True
            with patch.object(dl, 'collect_model_upstream_readiness', return_value=(dataset,)), \
                 patch.object(dl, '_collect_selection_pit_source_status', side_effect=fake_model_status), \
                 patch.object(teacher, 'inspect_history', side_effect=AssertionError('verified PIT must REUSE')), \
                 patch.object(teacher, 'inspect_current', side_effect=AssertionError('verified PIT must REUSE')):
                rows, identities, actions, ready, periods = dl.collect_dl_artifact_status(
                    root=root, settings=settings, required_dl_sources={'CONT13DD_ROLL'},
                    runtime_required_dl_sources={'CONT13DD_ROLL'},
                )
                actions = planning._attach_trade_path_teacher_dependencies(
                    root=root, settings=settings, required_dl_sources={'CONT13DD_ROLL'},
                    actions=actions,
                )
            self.assertTrue(ready['CONT13DD_ROLL'])
            self.assertFalse(any('teacher' in item.artifact_key for item in actions))
            self.assertTrue(all(item.action == 'REUSE' for item in actions))

    def test_service_plan_attaches_teacher_once_after_model_actions(self):
        from core.research_orchestration import ResearchArtifactAction, ResearchArtifactPlan
        from services.research import strategy_compare_preparation_status as planning
        from services.breakout_quality import daily_min_p19_training_source as teacher
        settings = get_strategy_comparison_settings('broker_cash_dl_conversion_rolling')
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_key = 'model-upstream:fixture:dataset_core'
            model_key = 'dl:CONT13DD_ROLL:forward_scores'
            actions = [
                ResearchArtifactAction(
                    'dataset', dataset_key, 'REUSE', None, 'Verified dataset',
                    'dataset.json', producer_work_type='existing_artifact',
                ),
                ResearchArtifactAction(
                    'score', model_key, 'BUILD', 'canonical_model_artifacts',
                    'Missing PIT score', 'scores.csv', dependencies=(dataset_key,),
                    producer_work_type='model_training',
                ),
            ]
            with patch.object(teacher, 'inspect_history', return_value={
                'action': 'BUILD', 'path': root / 'p19_history.json',
                'reason': 'Signed historical P19 teacher missing',
            }), patch.object(teacher, 'inspect_current', return_value={
                'action': 'REUSE', 'path': root / 'p19_current.json',
                'reason': 'Signed current P19 teacher verified',
            }):
                result = planning._attach_trade_path_teacher_dependencies(
                    root=root, settings=settings, required_dl_sources={'CONT13DD_ROLL'},
                    actions=actions,
                )
            self.assertIs(result, actions)
            self.assertEqual(len(actions), 4)
            self.assertEqual(actions[0].artifact_key, dataset_key)
            self.assertEqual(actions[1].artifact_key, model_key)
            teacher_keys = tuple(item.artifact_key for item in actions[2:])
            self.assertTrue(teacher_keys[0].endswith(':daily_min_p19_historical_teacher:seed_42'))
            self.assertTrue(teacher_keys[1].endswith(':daily_min_p19_current_teacher:seed_42'))
            self.assertEqual(actions[1].dependencies, (dataset_key, *teacher_keys))
            self.assertEqual(actions[2].dependencies, (dataset_key,))
            self.assertEqual(actions[3].dependencies, (dataset_key,))
            ResearchArtifactPlan.from_actions(actions)

    def test_shared_param_source_resolver_keeps_legacy_callers(self):
        from filters.breakout_quality.strategy_compare_sources import resolve_param_source_path as owner
        from services.research.strategy_compare_preparation_status import resolve_param_source_path as facade
        self.assertIs(owner, facade)
        settings = get_strategy_comparison_settings('broker_cash_dl_conversion_rolling')
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for source_id in (
                'min_best_broker_cash_rolling_experiment',
                'daily_min_four_best_broker_cash_rolling_experiment',
                'daily_full_nineteen_best_broker_cash_rolling_experiment',
            ):
                self.assertEqual(
                    owner(root, settings, source_id), facade(root, settings, source_id)
                )

    def test_trade_path_teacher_planning_respects_service_filter_boundary(self):
        import ast
        from filters.breakout_quality import strategy_compare_dl_artifacts as dl
        from services.research import strategy_compare_preparation_status as planning
        lower_source = Path(dl.__file__).read_text(encoding='utf-8')
        for item in ast.walk(ast.parse(lower_source)):
            if isinstance(item, ast.ImportFrom):
                self.assertFalse((item.module or '').startswith('services.'),
                                 f'filter must not import service: {item.module}')
            if isinstance(item, ast.Import):
                self.assertFalse(any(alias.name.startswith('services.') for alias in item.names))
        self.assertTrue(hasattr(planning, '_attach_trade_path_teacher_dependencies'))
        self.assertFalse(hasattr(dl, '_collect_trade_path_teacher_dependencies'))

    def test_compare_replans_teacher_before_model_producer(self):
        from core.research_orchestration import ResearchArtifactAction, ResearchArtifactPlan
        from services.research import strategy_compare_preparation as preparation
        settings = get_strategy_comparison_settings('broker_cash_dl_conversion_rolling')
        state = {'teacher': False, 'model': False}
        executions = []
        def refreshed():
            teacher_action = 'REUSE' if state['teacher'] else 'BUILD'
            model_action = 'REUSE' if state['model'] else 'BUILD'
            plan = ResearchArtifactPlan.from_actions((
                ResearchArtifactAction('dataset', 'model-upstream:dataset', 'REUSE',
                                       None, 'READY', 'dataset.json'),
                ResearchArtifactAction('teacher', 'model-upstream:p19_teacher',
                                       teacher_action,
                                       None if state['teacher'] else 'canonical_model_upstream',
                                       'signed teacher', 'teacher.json',
                                       dependencies=('model-upstream:dataset',),
                                       producer_work_type=(
                                           'existing_artifact' if state['teacher'] else 'model_training'
                                       ), execution_priority=30),
                ResearchArtifactAction('scores', 'dl:CONT13DD_ROLL:forward_scores',
                                       model_action,
                                       None if state['model'] else 'canonical_model_artifacts',
                                       'P19 model', 'scores.csv',
                                       dependencies=('model-upstream:p19_teacher',),
                                       producer_work_type=(
                                           'existing_artifact' if state['model'] else 'model_training'
                                       ), execution_priority=100),
            ))
            return {'preparation_plan': plan, 'dl_sources': {'CONT13DD_ROLL': {}}}
        def producer(action):
            executions.append(action.builder_type)
            if action.builder_type == 'canonical_model_upstream':
                self.assertFalse(state['model'])
                state['teacher'] = True
            elif action.builder_type == 'canonical_model_artifacts':
                self.assertTrue(state['teacher'])
                state['model'] = True
            else:
                self.fail('unexpected producer')
            return 0
        with TemporaryDirectory() as tmp:
            outcome = preparation._run_preparation_plan(
                project_root=Path(tmp), settings=settings,
                status=refreshed(), status_refresher=refreshed,
                required_artifact_keys=None, failure_prefix='P19 prerequisite',
                producer_handlers={'model_training': producer},
            )
        self.assertEqual(executions, ['canonical_model_upstream', 'canonical_model_artifacts'])
        self.assertEqual(outcome['preparation_plan'].overall_status, 'READY')

    def test_compare_model_worker_requires_optimizer_teacher_ready(self):
        from core.research_orchestration import ResearchArtifactAction, ResearchArtifactPlan
        from services.research import breakout_quality_application as app
        settings = get_strategy_comparison_settings('broker_cash_dl_conversion_rolling')
        source = settings.dl_sources['CONT13DD_ROLL']
        graph = ResearchArtifactPlan.from_actions((
            ResearchArtifactAction('dataset', 'model-upstream:dataset_core', 'REUSE',
                                   None, 'Dataset READY', 'dataset.json'),
            ResearchArtifactAction('p19', 'model-upstream:daily_min_p19_historical_teacher_seed_42',
                                   'BUILD', 'optimizer_daily_min_p19_historical_teacher',
                                   'P19 history missing', 'missing.json',
                                   dependencies=('model-upstream:dataset_core',)),
        ))
        with patch.object(app, '_strategy_compare_required_model_sources',
                          return_value=((settings,), (('CONT13DD_ROLL', source),))), \
             patch.object(app, '_collect_continuous_research_input_plan', return_value=graph) as lookup, \
             patch.object(app, '_run_strategy_compare_model_build_subprocess',
                          side_effect=AssertionError('No worker may start')):
            with self.assertRaisesRegex(RuntimeError, 'upstream尚未READY'):
                app._prepare_strategy_compare_model_artifacts('synthetic',
                    profile_ids=(settings.profile_id,))
        self.assertEqual(lookup.call_count, 1)
        self.assertEqual(lookup.call_args.kwargs['seed'], 42)

    def test_p19_teacher_is_four_dimensional_and_independent(self):
        self.assertEqual((PROFILE, BASELINE_ARM, TEACHER_SEED), ('min_atr_limit_v2', 'C166', 42))
        self.assertTrue(all(model_id not in {'MR-13DD', 'MR-13DE'}
                            for model_id, profile in get_research_summary_model_profiles()))
        self.assertEqual(get_continuous_ranker_research_spec(RANK).model_research_id, 'MR-13DD')
        self.assertEqual(get_continuous_ranker_research_spec(POSITIVE).model_research_id, 'MR-13DE')
        from strategies.daily.search_space import resolve_daily_search_profile
        fields = tuple(resolve_daily_search_profile(PROFILE)['fields'])
        self.assertEqual(fields, ('atr_len', 'atr_buy_tol', 'atr_times_init', 'atr_times_trail'))
        self.assertNotEqual(PROFILE, 'full_common_filters_v2')
        with TemporaryDirectory() as tmp:
            self.assertIn('daily_min_p19_label_history', historical_params_path(tmp).as_posix())
            self.assertEqual(inspect_history(project_root=tmp, seed=42)['action'], 'BUILD')

    def test_two_models_share_p19_truth_without_reusing_older_p18_or_p20(self):
        rank, pos = (get_continuous_ranker_research_spec(x) for x in (RANK, POSITIVE))
        self.assertEqual((rank.model_research_id, pos.model_research_id), ('MR-13DD', 'MR-13DE'))
        r, p = (get_continuous_ranker_execution_recipe(x) for x in (RANK, POSITIVE))
        self.assertEqual(r.continuous_target_id, p.continuous_target_id)
        self.assertEqual(r.target_policy.contract_kind, 'daily_min_p19_seed42_fullcycle_netpnl')
        self.assertEqual(p.target_policy.contract_kind, r.target_policy.contract_kind)
        self.assertEqual(r.training_policy.target_builder, 'scalar_pairwise')
        self.assertEqual(p.training_policy.target_builder, 'netpnl_positive_binary')
        self.assertEqual(p.training_policy.categorical_class_count, 2)
        for unrelated in ('true_daily_universal_mintrade_fullcycle_netpnl_full_list_ndcg_pairwise',
                          'daily_universal_full_p20_fulltrade_netpnl_full_list_ndcg_pairwise'):
            self.assertNotEqual(get_continuous_ranker_execution_recipe(unrelated).continuous_target_id,
                                r.continuous_target_id)

    def test_oracle_recipe_and_display_are_unique_but_source_immutable(self):
        from core.audit_policy import get_reusable_audit_definitions
        from services.audit import target_oracle_romd as oracle
        definitions = get_reusable_audit_definitions('breakout_quality')
        d = next(item for item in definitions if item.audit_type == oracle.SUPPORTED_AUDIT_TYPE)
        plan = oracle._oracle_comparison_plan(d)
        # Retired P19 must not be a current Oracle computation, but the
        # scientific recipe remains addressable for historical evidence.
        self.assertNotIn(RECIPE, (row[0] for row in plan['daily_recipes'].values()))
        from core.audit_registry import oracle_comparison_key, resolve_current_oracle_comparisons
        inactive = d.source['historical_only_oracle_targets']
        for label in (
            'TrueDailyUniversal_MinTrade_P19_NetPnL',
            'TrueDailyUniversal_MinTrade_P19_NetPnL_Direct',
            'TrueDailyUniversal_MinTrade_P19_NetPnL_Positive_Direct',
        ):
            historical = {
                'target': label, 'selector': 'direct', 'strategy': 'Daily',
                'parameter_family': 'daily_min_p19'
            }
            self.assertIn(oracle_comparison_key(historical), inactive)
            with self.assertRaisesRegex(ValueError, 'Historical-only'):
                resolve_current_oracle_comparisons({
                    **d.source, 'oracle_comparisons': [historical],
                })
        self.assertEqual(oracle_scientific_target_name(
            'TrueDailyUniversal_MinTrade_P19_NetPnL_Direct'), 'Daily_MinTrade_NetPnL')
        self.assertEqual(oracle_scientific_target_name(
            'TrueDailyUniversal_MinTrade_P19_NetPnL_Positive_Direct'),
            'Daily_MinTrade_NetPnL_Positive')
        p19 = source_calculation_contract(recipe=RECIPE, profile_id=RANK)
        p20 = source_calculation_contract(recipe='daily_trade_path_daily_full_p20_net_pnl',
                                          profile_id=RANK)
        old = source_calculation_contract(recipe='daily_trade_path_true_net_pnl', profile_id=RANK)
        self.assertEqual(p19['calculation_revision'], 2)
        self.assertNotEqual(p19['implementation_sha256'], p20['implementation_sha256'])
        self.assertNotEqual(p19['implementation_sha256'], old['implementation_sha256'])

    def test_source_dependency_owner_and_full_execution_fingerprint(self):
        from filters.breakout_quality import daily_min_p19_target as target
        from services.breakout_quality.trade_path_target_source import registered_trade_path_source
        from services.audit import daily_min_p19_oracle_truth as oracle_truth

        source = registered_trade_path_source(RANK)
        self.assertEqual(tuple(dep.name for dep in source.dependencies),
                         ('daily_min_p19_historical_teacher', 'daily_min_p19_current_teacher'))
        self.assertEqual(source.contract_kind, 'daily_min_p19_seed42_fullcycle_netpnl')
        baseline = target.daily_min_p19_execution_sha256()
        witness = oracle_truth.truth_producer_fingerprint()
        original = target.compute_file_sha256

        def changed(path):
            value = original(path)
            if str(path).endswith('daily_full_p20_target.py'):
                return 'f' * 64
            return value

        with patch.object(target, 'compute_file_sha256', side_effect=changed):
            self.assertNotEqual(target.daily_min_p19_execution_sha256(), baseline)
            oracle_truth.truth_producer_fingerprint.cache_clear()
            self.assertNotEqual(oracle_truth.truth_producer_fingerprint(), witness)
        oracle_truth.truth_producer_fingerprint.cache_clear()

    def test_actual_p19_min_path_produces_profit_loss_and_null(self):
        dates, frame, source = _fixture()
        batch = compute_daily_min_p19_batch(
            frame, np.arange(40, 105, dtype=np.int64), ticker='2330',
            schedule=source.schedule, exchange_calendar=dates, research_cutoff=dates[-1],
        )
        valid = np.asarray(batch.valid_mask, dtype=bool)
        raw = np.asarray(batch.target_raw_pnl, dtype=float)
        self.assertGreater(int(valid.sum()), 10)
        self.assertTrue(np.isfinite(raw[valid]).all())
        self.assertTrue(np.isnan(raw[~valid]).all())
        self.assertTrue((raw[valid] > 0).any())
        self.assertTrue((raw[valid] < 0).any())
        self.assertTrue((~valid).any())

    def test_p19_rejects_p20_filters_and_foreign_source(self):
        dates, frame, source = _fixture()
        changed = replace(source.schedule[0],
                          params=replace(source.schedule[0].params, use_bb=True))
        with self.assertRaisesRegex(ValueError, 'Full filters'):
            compute_daily_min_p19_batch(
                frame, np.arange(20, 40, dtype=np.int64), ticker='2330',
                schedule=(changed,), exchange_calendar=dates, research_cutoff=dates[-1],
            )
        foreign = replace(source, source_id='daily_full_p20_seed42')
        with self.assertRaises(FormalMinTrade40PITBlocked):
            materialize_daily_min_p19_pit_labels(
                foreign, expected_seed=42, frame=frame,
                positions=np.arange(30, 93, dtype=np.int64), ticker='2330',
                exchange_calendar=dates, fit_cutoff=dates[100], cache_project_root=None,
            )

    def test_p19_maturity_no_leakage(self):
        dates, frame, source = _fixture()
        labels = materialize_daily_min_p19_pit_labels(
            source, expected_seed=42, frame=frame,
            positions=np.arange(30, 93, dtype=np.int64), ticker='2330',
            exchange_calendar=dates, fit_cutoff=dates[100], cache_project_root=None,
        )
        valid = np.asarray(labels.completed, dtype=bool)
        self.assertGreater(int(valid.sum()), 0)
        self.assertEqual(int(valid.sum()), int(np.isfinite(labels.raw_net_pnl).sum()))
        self.assertTrue(np.isnan(np.asarray(labels.raw_net_pnl)[~valid]).all())
        self.assertTrue((labels.maturity_dates[valid] <=
                         np.datetime64(dates[100].date(), 'D')).all())
        self.assertEqual(labels.preflight['entry_policy'],
                         'next_exchange_session_daily_min_p19_atr_limit_v1')
