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
    def test_registry_uses_new_live_ids_and_preserves_historical_arms(self):
        active = set(get_research_summary_strategy_arm_ids())
        self.assertTrue({'C174', 'C175', 'C176'}.issubset(active))
        self.assertTrue({'C166', 'C167'}.isdisjoint(active))
        self.assertIn('C166', STRATEGY_COMPARE_ARMS)
        self.assertEqual(get_research_summary_baseline_arm_ids()['daily_min_p19'], 'C174')
        settings = get_strategy_comparison_settings('broker_cash_dl_conversion_rolling')
        self.assertTrue(all(settings.arms[a].enabled for a in ('C174', 'C175', 'C176')))
        self.assertTrue(all(not settings.arms[a].enabled for a in ('C166', 'C167')))
        self.assertFalse(settings.arms['C174'].dl_enabled)
        self.assertTrue(all(settings.arms[a].dl_enabled for a in ('C175', 'C176')))
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

    def test_p19_teacher_is_four_dimensional_and_independent(self):
        self.assertEqual((PROFILE, BASELINE_ARM, TEACHER_SEED), ('min_atr_limit_v2', 'C166', 42))
        self.assertTrue(any(profile == RANK and name == 'MR-13DD'
                            for name, profile in get_research_summary_model_profiles()))
        self.assertTrue(any(profile == POSITIVE and name == 'MR-13DE'
                            for name, profile in get_research_summary_model_profiles()))
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
        self.assertIn(RECIPE, (row[0] for row in plan['daily_recipes'].values()))
        self.assertEqual(oracle_scientific_target_name(
            'TrueDailyUniversal_MinTrade_P19_NetPnL_Direct'), 'Daily_MinTrade_NetPnL')
        self.assertEqual(oracle_scientific_target_name(
            'TrueDailyUniversal_MinTrade_P19_NetPnL_Positive_Direct'),
            'Daily_MinTrade_NetPnL_Positive')
        p19 = source_calculation_contract(recipe=RECIPE, profile_id=RANK)
        p20 = source_calculation_contract(recipe='daily_trade_path_daily_full_p20_net_pnl',
                                          profile_id=RANK)
        old = source_calculation_contract(recipe='daily_trade_path_true_net_pnl', profile_id=RANK)
        self.assertEqual(p19['calculation_revision'], 1)
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
