"""Isolated contract regression for one-time Oracle target → RoMD Audit.

Run directly; requires no historical market data and does not touch formal
Test Suite frozen entry checks, PIT artifacts, or portfolio model fitting.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import sys
import tempfile
import pandas as pd
from unittest import TestCase, main
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.audit_policy import get_audit_definitions
from core.breakout_quality_ranking_context import (
    SCORE_SOURCE_AUDIT_ORACLE_FUTURE_TRUTH,
    SUPPORTED_RANKING_SCORE_SOURCES,
    breakout_quality_ranking_source_context,
    get_breakout_quality_ranking_source_context,
)
from filters.breakout_quality.runtime import (
    audit_oracle_truth_score_context,
    resolve_breakout_quality_candidate_rank,
)
from services.audit import target_oracle_romd as oracle
from services.audit.runner import collect_audit_menu_state, collect_audit_status
from services.audit.catalog import get_audit_methods


class OracleAuditContractCases(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.definition = next(
            definition for definition in get_audit_definitions("breakout_quality")
            if definition.audit_type == oracle.SUPPORTED_AUDIT_TYPE
        )
        cls.exact_policy = oracle._contract(cls.definition)[2].dl_runtime_mode
        cls.exact_options = oracle._contract(cls.definition)[2].dl_runtime_options

    def test_audit_is_one_time_only_and_not_a_reusable_menu_item(self):
        self.assertEqual(collect_audit_menu_state("breakout_quality")["enabled_count"], 1)
        self.assertNotIn("target_ground_truth_strategy_romd", [x.method_id for x in get_audit_methods()])
        self.assertEqual(len(oracle.ORACLE_TARGETS), 7)
        with tempfile.TemporaryDirectory() as tmp:
            status = collect_audit_status("breakout_quality", project_root=Path(tmp))
            self.assertEqual(status["overall_status"], "BLOCKED")
            self.assertIn("full dataset", status["rows"][0]["reason"])

    def test_future_source_cannot_be_registered_as_formal_pit_score(self):
        self.assertNotIn(SCORE_SOURCE_AUDIT_ORACLE_FUTURE_TRUTH, SUPPORTED_RANKING_SCORE_SOURCES)
        with self.assertRaises(ValueError):
            with breakout_quality_ranking_source_context(score_source=SCORE_SOURCE_AUDIT_ORACLE_FUTURE_TRUTH):
                self.fail("future truth must never enter normal score registry")

    def test_future_context_returns_exact_score_and_resets(self):
        before = get_breakout_quality_ranking_source_context()
        scores = {("2330", "2025-02-03"): 0.8125}
        with audit_oracle_truth_score_context(
            scores, ranking_policy=self.exact_policy, ranking_options=self.exact_options,
        ):
            rank = resolve_breakout_quality_candidate_rank(
                ticker="2330", signal_date="2025-02-03", information_date="2025-02-03", high_len=260,
            )
            self.assertEqual(rank["score"], 0.8125)
            self.assertEqual(rank["score_source"], SCORE_SOURCE_AUDIT_ORACLE_FUTURE_TRUTH)
            self.assertEqual(rank["score_date"], "2025-02-03")
            with self.assertRaisesRegex(ValueError, "未覆蓋"):
                resolve_breakout_quality_candidate_rank(
                    ticker="2317", signal_date="2025-02-03", information_date="2025-02-03", high_len=260,
                )
            with self.assertRaisesRegex(RuntimeError, "巢狀"):
                with audit_oracle_truth_score_context(
                    scores, ranking_policy=self.exact_policy, ranking_options=self.exact_options,
                ):
                    pass
        after = get_breakout_quality_ranking_source_context()
        self.assertEqual(after, before)

    def test_score_context_restores_after_replay_failure(self):
        before = get_breakout_quality_ranking_source_context()
        with self.assertRaisesRegex(RuntimeError, "forced abort"):
            with audit_oracle_truth_score_context(
                {("2330", "2025-02-03"): 0.5}, ranking_policy=self.exact_policy,
                ranking_options=self.exact_options,
            ):
                raise RuntimeError("forced abort")
        self.assertEqual(get_breakout_quality_ranking_source_context(), before)

    def _truth(self, provider: str) -> pd.DataFrame:
        values = {
            "pure_mfe40_provider": {
                "mfe": [3.0, 2.0, 1.0], "peak_adverse": [0.8, 0.1, 0.2],
                "min_low": [-0.1, -0.3, -0.2], "raw": [3.0, 2.0, 1.0],
            },
            "mfe20_d20_provider": {
                "mfe": [1.0, 3.0, 2.0], "peak_adverse": [0.1, 0.2, 0.3],
                "min_low": [-0.2, -0.1, -0.3], "raw": [1.0, 3.0, 2.0],
            },
            "first_breach_provider": {
                "mfe": [0.9, 0.6, 0.4], "peak_adverse": [0.1, 0.5, 0.05],
                "min_low": [-0.1, -0.1, -0.1], "raw": [0.9, 0.6, 0.4],
            },
        }[provider]
        return pd.DataFrame({"ticker": ["2330", "2317", "2454"],
                             "date": ["2025-01-02"] * 3, **values})

    def test_oracle_target_scores_use_matched_cohort_and_correct_directions(self):
        with patch.object(oracle, "_load_truth", side_effect=lambda definition, root, source_field, cutoff: self._truth(source_field)):
            scores, evidence = oracle.build_oracle_score_tables(
                self.definition, project_root=ROOT, cutoff="2026-03-02",
            )
        self.assertEqual(evidence["cohort_rows"], 3)
        self.assertEqual(set(scores), {label for label, _ in oracle.ORACLE_TARGETS})
        key = lambda ticker: (ticker, "2025-01-02")
        # MFE40 ranks 2330 high, but peak-safety composite favors 2317.
        self.assertEqual(scores["MFE40_Control"][key("2330")], 1.0)
        self.assertEqual(scores["AO40_EqualRank"][key("2317")], 1.0)
        self.assertEqual(scores["CG40_EqualRank"][key("2330")], 1.0)
        self.assertEqual(scores["CH20_EqualRank"][key("2317")], 1.0)
        self.assertEqual(scores["U40"][key("2330")], 1.0)
        self.assertEqual(scores["U20"][key("2317")], 1.0)
        self.assertEqual(scores["FirstBreach_NetR"][key("2330")], 1.0)
        self.assertTrue(all(0.0 <= value <= 1.0 for table in scores.values() for value in table.values()))

    def test_mismatched_stock_days_cannot_be_silently_filled(self):
        def short_truth(definition, root, source_field, cutoff):
            table = self._truth(source_field)
            return table.iloc[:2] if source_field == "first_breach_provider" else table
        with patch.object(oracle, "_load_truth", side_effect=short_truth):
            scores, evidence = oracle.build_oracle_score_tables(
                self.definition, project_root=ROOT, cutoff="2026-03-02",
            )
        self.assertEqual(evidence["cohort_rows"], 2)
        self.assertTrue(all(len(table) == 2 for table in scores.values()))
        self.assertNotIn(("2454", "2025-01-02"), scores["U40"])
        self.assertTrue(evidence["oracle_only"])

    def test_tradable_benchmark_has_canonical_oracle_truth_at_common_40d_maturity(self):
        # The daily-universal model TRAINING universe excludes benchmark 0050,
        # but the portfolio still treats it as a tradable candidate.
        dates = pd.bdate_range("2020-01-01", periods=520)
        close = np.arange(520, dtype=float) / 50.0 + 100.0
        frame = pd.DataFrame({
            "Open": close, "High": close * 1.01, "Low": close * 0.99,
            "Close": close, "Volume": 1_000_000.0,
        }, index=dates)
        cutoff = str(dates[450].date())
        from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
        from filters.breakout_quality.daily_ranker_data import compute_daily_opportunity_target_batch
        from filters.breakout_quality.continuous_target import StrategyAlignedContinuousTargetSpec
        from filters.breakout_quality.contract import DEFAULT_LABEL_POLICY
        from dataclasses import replace
        with patch.object(oracle, "discover_dataset_csv_inputs", return_value=([("0050", Path("0050.csv"))], [])), \
             patch.object(oracle, "load_dataset_frame", return_value=frame):
            for provider in ("pure_mfe40_provider", "mfe20_d20_provider", "first_breach_provider"):
                result = oracle._benchmark_truth_for_oracle(
                    self.definition, ROOT, provider, cutoff=cutoff,
                )
                self.assertFalse(result.empty)
                self.assertEqual(set(result["ticker"]), {"0050"})
                self.assertTrue(result["date"].is_unique)
                recipe = get_continuous_ranker_execution_recipe(str(self.definition.source[provider]))
                policy = recipe.target_policy
                horizon = int(policy.horizon_bars or DEFAULT_LABEL_POLICY.label_horizon_bars)
                maturity_bars = int(policy.sample_universe_horizon_bars or horizon)
                # The earliest legal 40D last bar is strictly before/equal to cutoff.
                position = dates.get_indexer(pd.to_datetime(result["date"]))
                self.assertTrue(np.all(position + maturity_bars <= 450))
                spec = StrategyAlignedContinuousTargetSpec.from_label_policy(
                    replace(DEFAULT_LABEL_POLICY, label_horizon_bars=horizon)
                )
                canonical = compute_daily_opportunity_target_batch(
                    frame, position, spec=spec, target_id=policy.component_target_id,
                )
                self.assertTrue(canonical.valid_mask.all())
                np.testing.assert_allclose(result["raw"], canonical.target_raw_r, rtol=1e-6)
                np.testing.assert_allclose(result["mfe"], canonical.favorable_return / spec.risk_budget_return, rtol=1e-6)
                np.testing.assert_allclose(result["peak_adverse"], canonical.adverse_return_to_peak / spec.risk_budget_return, rtol=1e-6)
                np.testing.assert_allclose(result["min_low"], canonical.minimum_low_return, rtol=1e-6)

    def test_missing_benchmark_price_is_blocked_not_neutral_filled(self):
        with patch.object(oracle, "discover_dataset_csv_inputs", return_value=([("2330", Path("2330.csv"))], [])):
            with self.assertRaisesRegex(oracle.OracleBlockedError, "Benchmark價格"):
                oracle._benchmark_truth_for_oracle(
                    self.definition, ROOT, "pure_mfe40_provider", cutoff="2026-03-02",
                )

    def test_truth_loader_augments_only_excluded_benchmark(self):
        provider = "pure_mfe40_provider"
        from core.breakout_quality_registry import get_breakout_quality_experiment_profile
        profile = get_breakout_quality_experiment_profile(str(self.definition.source[provider]))
        fake_bundle = SimpleNamespace(
            target_manifest={"target_contract": {"target_id": profile.continuous_target_id}},
            group_table=pd.DataFrame({
                "ticker": ["2330"], "date": ["2025-01-02"],
                "label_eval_end_date": ["2025-03-10"],
                "target_favorable_r": [1.5], "target_adverse_r": [0.2],
                "target_minimum_low_return": [-0.08],
            }),
            raw_target=np.array([1.5]), target_valid=np.array([True]),
        )
        benchmark = pd.DataFrame({
            "ticker": ["0050"], "date": ["2025-01-02"],
            "mfe": [1.1], "peak_adverse": [0.3],
            "min_low": [-0.09], "raw": [1.1],
        })
        with patch.object(oracle, "load_profile_continuous_ranker_data", return_value=fake_bundle), \
             patch.object(oracle, "_benchmark_truth_for_oracle", return_value=benchmark) as fill:
            result = oracle._load_truth(self.definition, ROOT, provider, cutoff="2026-03-02")
        self.assertEqual(set(result["ticker"]), {"2330", "0050"})
        self.assertEqual(len(result), 2)
        fill.assert_called_once()
        # The same code must NOT override an already canonical provider-sourced benchmark.
        with patch.object(oracle, "load_profile_continuous_ranker_data", return_value=SimpleNamespace(
            target_manifest=fake_bundle.target_manifest,
            group_table=pd.concat([fake_bundle.group_table, fake_bundle.group_table.assign(ticker="0050")]),
            raw_target=np.array([1.5, 1.5]), target_valid=np.array([True, True]),
        )), patch.object(oracle, "_benchmark_truth_for_oracle") as fill:
            result = oracle._load_truth(self.definition, ROOT, provider, cutoff="2026-03-02")
        self.assertEqual(len(result), 2)
        fill.assert_not_called()

    def test_four_seed_paired_replay_and_report_without_real_market(self):
        class Benchmark:
            benchmark_id = "end_to_end_v1"
        seeds = (693545351, 2014432738, 104504529, 762252050)
        self.assertEqual(len(seeds), 4)
        seen = []
        def fake_replay(*, name, **kwargs):
            if name.startswith("C96"):
                self.assertNotEqual(get_breakout_quality_ranking_source_context().score_source,
                                    SCORE_SOURCE_AUDIT_ORACLE_FUTURE_TRUTH)
            else:
                self.assertEqual(get_breakout_quality_ranking_source_context().score_source,
                                 SCORE_SOURCE_AUDIT_ORACLE_FUTURE_TRUTH)
            seen.append(name)
            return {"name": name}
        def fake_metrics(replay, *, target):
            seed = int(replay["name"].split("seed=")[1].split()[0])
            base_romd = 8.0 + seeds.index(seed) * 0.3
            return {
                "target": target, "total_return_pct": 100.0, "annualized_return_pct": 15.0,
                "mdd_pct": 12.0, "romd": base_romd if target == "C96" else base_romd + 1.5,
                "ev_a_r": 0.3, "trades": 50, "exposure_pct": 75.0,
                "yearly": [{"year": 2021, "strategy_return_pct": 4.0, "is_full_year": True}],
            }
        with patch.object(oracle, "collect_status", return_value={"status": "READY"}), \
             patch.object(oracle, "_robustness_artifacts", return_value=(Benchmark(), [(seed, Path(f"seed_{seed}.json")) for seed in seeds])), \
             patch.object(oracle, "build_oracle_score_tables", return_value=({name: {("2330", "2025-01-02"): 0.8} for name, _ in oracle.ORACLE_TARGETS}, {"cohort_rows": 4, "last_common_day": "2025-12-31"})), \
             patch.object(oracle, "load_strategy_param_evaluation_view", return_value=({"kind": "rolling_active_param_ensemble", "payload": {}}, "test-source-sha")), \
             patch.object(oracle, "_validate_requested_param_policy"), \
             patch.object(oracle, "get_active_param_ensemble_date_range", return_value=("2020-01-01", "2026-03-02")), \
             patch.object(oracle, "_build_controlled_param_source_pair", return_value=("rolling_active_param_ensemble", "base", "oracle", None, None, None)), \
             patch.object(oracle, "_run_scenario", side_effect=fake_replay), \
             patch.object(oracle, "_metrics", side_effect=fake_metrics):
            report = oracle.build_report(self.definition, project_root=ROOT, quiet=True)
        self.assertEqual(len(seen), len(seeds) * (1 + len(oracle.ORACLE_TARGETS)))
        self.assertEqual(report["seeds"], list(seeds))
        self.assertEqual(report["end_date"], "2025-12-31")
        self.assertEqual(report["baseline"]["romd_std"], report["oracles"][0]["romd_std"])
        self.assertEqual(report["oracles"][0]["delta_romd_vs_c96"], 1.5)
        self.assertEqual(report["oracles"][0]["positive_seed_count"], len(seeds))
        self.assertIn("Oracle Performance", oracle.render_result(report, target="markdown"))
        self.assertIn("RoMD Mean", oracle.render_result(report, target="console"))
        json.dumps(report, ensure_ascii=False)
        self.assertNotEqual(get_breakout_quality_ranking_source_context().score_source,
                            SCORE_SOURCE_AUDIT_ORACLE_FUTURE_TRUTH)


if __name__ == "__main__":
    main(verbosity=2)
