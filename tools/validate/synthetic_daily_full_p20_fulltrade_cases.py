"""Independent invariants for the P20 Daily Full full-cycle NetPnL primitive.

P20 has its own economic teacher and model FIT identities.  These tests do not
run or emulate the formal suite and never create research results.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.breakout_quality_registry import get_continuous_ranker_research_spec
from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
from core.portfolio_param_runtime import build_portfolio_params_signature
from core.research_summary import get_research_summary_strategy_arm_ids
from core.strategy_compare_registry import STRATEGY_COMPARE_ARMS
from filters.breakout_quality.daily_trade40_target import ExecutableMinPeriod
from filters.breakout_quality.daily_full_p20_target import compute_daily_full_p20_batch
from services.breakout_quality.daily_full_p20_pit_source import materialize_daily_full_p20_pit_labels
from services.breakout_quality.daily_full_p20_training_source import SOURCE_ID, inspect_history
from services.breakout_quality.formal_mintrade40_pit_source import (
    FormalMinTrade40PITBlocked, FormalMinTrade40PITSource,
)
from tools.validate.synthetic_daily_trade40_cases import fixture


RANK_PROFILE = "daily_universal_full_p20_fulltrade_netpnl_full_list_ndcg_pairwise"
POS_PROFILE = "daily_universal_full_p20_fulltrade_netpnl_positive_binary"


def _fixture_source():
    days, frame, original, _ = fixture(count=160)
    params = replace(original, use_breakout_buy=False)
    period = ExecutableMinPeriod(
        start=days[0], end=None, params=params, source_id=SOURCE_ID,
        params_signature=build_portfolio_params_signature(params),
    )
    source = FormalMinTrade40PITSource(
        seed=42, param_sha256="a" * 64, schedule=(period,),
        publication_dates=(str((days[0] - pd.Timedelta(days=1)).date()),),
        source_id=SOURCE_ID, oos_start_date=str(days[0].date()),
    )
    return days, frame, source


class DailyFullP20NetPnLContractCases(TestCase):
    def test_new_rank_and_binary_head_share_p20_economic_truth_not_old_models(self):
        rank = get_continuous_ranker_research_spec(RANK_PROFILE)
        positive = get_continuous_ranker_research_spec(POS_PROFILE)
        self.assertEqual((rank.model_research_id, positive.model_research_id),
                         ("MR-13DB", "MR-13DC"))
        rank_runtime = get_continuous_ranker_execution_recipe(RANK_PROFILE)
        pos_runtime = get_continuous_ranker_execution_recipe(POS_PROFILE)
        self.assertEqual(rank_runtime.continuous_target_id, pos_runtime.continuous_target_id)
        self.assertEqual(rank_runtime.target_policy.contract_kind,
                         "daily_full_p20_seed42_fullcycle_netpnl")
        self.assertEqual(pos_runtime.target_policy.contract_kind,
                         rank_runtime.target_policy.contract_kind)
        self.assertEqual(rank_runtime.training_policy.target_builder, "scalar_pairwise")
        self.assertEqual(pos_runtime.training_policy.target_builder, "netpnl_positive_binary")
        self.assertEqual(pos_runtime.training_policy.categorical_class_count, 2)
        self.assertEqual(rank_runtime.target_policy.horizon_bars, None)

    def test_current_strategy_registry_retirement_and_positive_only_on_c172(self):
        current = set(get_research_summary_strategy_arm_ids())
        self.assertTrue({"C168", "C169", "C172", "C173"}.issubset(current))
        self.assertTrue({"C166", "C167"}.isdisjoint(current))
        a, b = STRATEGY_COMPARE_ARMS["C172"], STRATEGY_COMPARE_ARMS["C173"]
        for item in (a, b):
            self.assertEqual(item["entry_strategy"], "Daily")
            self.assertEqual(item["param_source"],
                             "daily_full_nineteen_best_broker_cash_rolling_experiment")
            self.assertEqual(item["dl_id"], "CONT13DB_ROLL")
            self.assertFalse(item["dl_runtime_options"]["preserve_k"])
            self.assertFalse(item["dl_runtime_options"]["preserve_r0"])
        self.assertEqual(a["dl_runtime_options"]["positive_dl_id"], "CONT13DC_ROLL")
        self.assertEqual(a["dl_runtime_options"]["positive_probability_threshold"], 0.5)
        self.assertNotIn("positive_dl_id", b["dl_runtime_options"])

    def test_real_next_session_broker_label_has_both_signs_and_nulls(self):
        days, frame, source = _fixture_source()
        values = compute_daily_full_p20_batch(
            frame, np.arange(40, 105, dtype=np.int64), ticker="2330",
            schedule=source.schedule, exchange_calendar=days, research_cutoff=days[-1],
        )
        valid = np.asarray(values.valid_mask, dtype=bool)
        raw = np.asarray(values.target_raw_pnl, dtype=float)
        self.assertGreater(int(valid.sum()), 0)
        self.assertTrue(np.isnan(raw[~valid]).all())
        self.assertTrue(np.isfinite(raw[valid]).all())
        self.assertTrue(np.any(raw[valid] > 0))
        self.assertTrue(np.any(raw[valid] < 0))
        self.assertEqual(len(values.reasons), len(raw))

    def test_pit_maturity_gates_full_cycle_and_rejects_wrong_teacher(self):
        days, frame, source = _fixture_source()
        positions = np.arange(30, 93, dtype=np.int64)
        args = dict(source=source, expected_seed=42, frame=frame,
                    positions=positions, ticker="2330", exchange_calendar=days,
                    fit_cutoff=days[100], cache_project_root=None)
        labels = materialize_daily_full_p20_pit_labels(**args)
        valid = np.asarray(labels.completed, dtype=bool)
        self.assertGreater(int(valid.sum()), 0)
        self.assertTrue(np.isnan(np.asarray(labels.raw_net_pnl)[~valid]).all())
        self.assertTrue(np.isfinite(np.asarray(labels.raw_net_pnl)[valid]).all())
        self.assertTrue((labels.maturity_dates[valid] <=
                         np.datetime64(days[100].date(), "D")).all())
        self.assertEqual(labels.preflight["entry_policy"],
                         "next_exchange_session_daily_full_p20_atr_limit_v1")
        with self.assertRaisesRegex(FormalMinTrade40PITBlocked,
                                    "Daily Full/P20 source identity mismatch"):
            materialize_daily_full_p20_pit_labels(**{
                **args, "source": replace(source, source_id="formal_full_roos_seed")
            })

    def test_teacher_missing_can_build_but_incompatible_artifact_blocks(self):
        with TemporaryDirectory() as tmp:
            state = inspect_history(project_root=tmp, seed=42)
            self.assertEqual(state["action"], "BUILD")
            path = Path(state["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
            state = inspect_history(project_root=tmp, seed=42)
            self.assertEqual(state["action"], "BLOCKED")

    def test_unrelated_model_registration_keeps_existing_oracle_price_identity(self):
        from services.audit import oracle_score_artifacts as owner
        original = owner.compute_file_sha256
        before = owner.oracle_daily_source_fingerprint()
        def unrelated_model_registration(path):
            if Path(path).name in {"breakout_quality_registry.py", "breakout_quality_runtime.py"}:
                return "f" * 64
            return original(Path(path))
        with patch.object(owner, "compute_file_sha256", side_effect=unrelated_model_registration):
            self.assertEqual(owner.oracle_daily_source_fingerprint(), before)
        def changed_market_features(path):
            if Path(path).name == "features.py":
                return "e" * 64
            return original(Path(path))
        with patch.object(owner, "compute_file_sha256", side_effect=changed_market_features):
            self.assertNotEqual(owner.oracle_daily_source_fingerprint(), before)


if __name__ == "__main__":
    import unittest
    unittest.main()
