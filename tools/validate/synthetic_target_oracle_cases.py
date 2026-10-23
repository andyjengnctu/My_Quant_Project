"""Isolated contract regression for reusable Oracle target → RoMD Audit.

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

from core.audit_policy import get_audit_definitions, get_reusable_audit_definitions
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
            definition for definition in get_reusable_audit_definitions("breakout_quality")
            if definition.audit_type == oracle.SUPPORTED_AUDIT_TYPE
        )
        cls.exact_policy = oracle._contract(cls.definition)[2].dl_runtime_mode
        cls.exact_options = oracle._contract(cls.definition)[2].dl_runtime_options

    def test_oracle_is_reusable_only_with_a_single_registered_run_route(self):
        self.assertEqual(collect_audit_menu_state("breakout_quality")["enabled_count"], 0)
        self.assertEqual(
            collect_audit_menu_state("breakout_quality", method_id=oracle.SUPPORTED_AUDIT_TYPE)["enabled_count"], 1
        )
        self.assertEqual(
            [x.method_id for x in get_audit_methods()].count(oracle.SUPPORTED_AUDIT_TYPE), 1
        )
        self.assertFalse(any(
            definition.audit_type == oracle.SUPPORTED_AUDIT_TYPE
            for definition in get_audit_definitions("breakout_quality")
        ))
        self.assertEqual(
            {name for name, _ in oracle.ORACLE_TARGETS},
            {"AO40_EqualRank", "CG40_EqualRank", "CH20_EqualRank", "U40", "U20",
             "FirstBreach_NetR", "MFE40_Control", "MFE40_over_D40", "MFE20", "MAE40", "MAE20"},
        )
        self.assertEqual(self.definition.output_subdir, "breakout_quality/reusable/target_oracle_romd")
        with tempfile.TemporaryDirectory() as tmp:
            status = collect_audit_status(
                "breakout_quality", project_root=Path(tmp), method_id=oracle.SUPPORTED_AUDIT_TYPE,
            )
            self.assertEqual(status["overall_status"], "BLOCKED")
            self.assertIn("full dataset", status["rows"][0]["reason"])

    def test_persistent_contract_must_be_approved_before_replay(self):
        with patch.object(oracle, "require_approved_report_contract", side_effect=ValueError("UNAUTHORIZED")), \
             patch.object(oracle, "build_report") as builder:
            with self.assertRaisesRegex(ValueError, "UNAUTHORIZED"):
                oracle.run_formal_audit(self.definition, project_root=ROOT, quiet=True)
            builder.assert_not_called()

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

    def _mock_market(self, omit_first_breach_last: bool = False):
        market = {field: self._truth(field) for field in oracle._ORACLE_PROVIDER_FIELDS}
        if omit_first_breach_last:
            market["first_breach_provider"] = market["first_breach_provider"].iloc[:2]
        return market, {"score_start": "2025-01-01", "truth_source": "test"}

    def test_oracle_target_scores_use_matched_cohort_and_correct_directions(self):
        with patch.object(oracle, "_load_truth", side_effect=lambda definition, root, source_field, cutoff: self._truth(source_field)), \
             patch.object(oracle, "_market_truth_for_oracle", return_value=self._mock_market()):
            scores, evidence = oracle.build_oracle_score_tables(
                self.definition, project_root=ROOT, cutoff="2026-03-02",
            )
        self.assertEqual(evidence["cohort_rows"], 3)
        self.assertEqual(evidence["mfe40_over_d40_zero_downside_positive_upside_count"], 0)
        self.assertEqual(set(scores), {label for label, _ in oracle.ORACLE_TARGETS})
        key = lambda ticker: (ticker, "2025-01-02")
        self.assertEqual(scores["MFE40_Control"][key("2330")], 1.0)
        self.assertEqual(scores["AO40_EqualRank"][key("2317")], 1.0)
        self.assertEqual(scores["CG40_EqualRank"][key("2330")], 1.0)
        self.assertEqual(scores["CH20_EqualRank"][key("2317")], 1.0)
        self.assertEqual(scores["U40"][key("2330")], 1.0)
        self.assertEqual(scores["U20"][key("2317")], 1.0)
        self.assertEqual(scores["FirstBreach_NetR"][key("2330")], 1.0)
        self.assertEqual(scores["MFE40_over_D40"][key("2330")], 1.0)
        self.assertEqual(scores["MFE20"][key("2317")], 1.0)
        self.assertEqual(scores["MAE40"][key("2330")], 1.0)
        self.assertEqual(scores["MAE20"][key("2317")], 1.0)
        self.assertTrue(all(0.0 <= v <= 1.0 for table in scores.values() for v in table.values()))

    def test_raw_ratio_zero_downside_uses_explicit_order_not_pseudocount(self):
        index = pd.RangeIndex(4)
        ratio = oracle._mfe40_over_d40_ratio(
            pd.Series([1.0, 0.0, 4.0, 0.5], index=index),
            pd.Series([0.0, 0.0, -1.0, -0.01], index=index),
        )
        self.assertTrue(np.isposinf(ratio.iloc[0]))
        self.assertEqual(float(ratio.iloc[1]), 0.0)
        self.assertEqual(float(ratio.iloc[2]), 4.0)
        self.assertEqual(float(ratio.iloc[3]), 50.0)
        self.assertEqual(
            float(oracle._mfe40_over_d40_ratio(pd.Series([-0.4]), pd.Series([-0.2])).iloc[0]),
            -2.0,
        )  # A 40-bar path that never recovers above the anchor is legal.
        ranked, _ = oracle.ensure_daily_percentile(
            pd.DataFrame({"date": ["2025-01-02"] * 4, "value": ratio}),
            method="average_zero_based",
        )
        self.assertTrue(np.isfinite(ranked["percentile"]).all())
        self.assertEqual(float(ranked.iloc[0]["percentile"]), 1.0)
        with self.assertRaisesRegex(oracle.OracleBlockedError, "非負"):
            oracle._mfe40_over_d40_ratio(pd.Series([1.0]), pd.Series([0.1]))
        with self.assertRaisesRegex(oracle.OracleBlockedError, "非負"):
            oracle._mfe40_over_d40_ratio(pd.Series([float("nan")]), pd.Series([-0.1]))

    def test_mismatched_stock_days_cannot_be_silently_filled(self):
        # If an official source claims a valid row but Research prices cannot
        # produce the same 40D-mature future path, stop instead of imputing it.
        with patch.object(oracle, "_load_truth", side_effect=lambda definition, root, source_field, cutoff: self._truth(source_field)), \
             patch.object(oracle, "_market_truth_for_oracle", return_value=self._mock_market(omit_first_breach_last=True)):
            with self.assertRaisesRegex(oracle.OracleBlockedError, "官方真值找不到相應"):
                oracle.build_oracle_score_tables(
                    self.definition, project_root=ROOT, cutoff="2026-03-02",
                )

    def test_full_price_universe_includes_0050_and_early_listed_2211(self):
        # 2211 can be a replay ranking candidate before the DL sample has a
        # complete input feature window. Future truth is independent of that
        # training eligibility; the *canonical* full-path calculator owns it.
        benchmark_dates = pd.bdate_range("2020-01-01", periods=700)
        listing_dates = pd.bdate_range("2021-01-04", periods=180)
        def frame(dates):
            close = np.arange(len(dates), dtype=float) / 50.0 + 100.0
            return pd.DataFrame({
                "Open": close, "High": close * 1.01, "Low": close * 0.99,
                "Close": close, "Volume": 1_000_000.0,
            }, index=dates)
        prices = {"0050": frame(benchmark_dates), "2211": frame(listing_dates)}
        paths = [(ticker, Path(ticker + ".csv")) for ticker in prices]
        settings = SimpleNamespace(start_date="2021-01-01", end_date="2021-06-01", dataset="full")
        with patch.object(oracle, "_contract", return_value=(settings, None, None)), \
             patch.object(oracle, "discover_dataset_csv_inputs", return_value=(paths, [])), \
             patch.object(oracle, "load_dataset_frame", side_effect=lambda path, ticker, **kw: prices[ticker]):
            truth, evidence = oracle._market_truth_for_oracle(self.definition, ROOT, cutoff="2021-06-01")
        self.assertEqual(evidence["price_ticker_count"], 2)
        self.assertEqual(evidence["score_start"], "2020-12-31")
        for provider in oracle._ORACLE_PROVIDER_FIELDS:
            table = truth[provider]
            self.assertEqual(set(table["ticker"]), {"0050", "2211"})
            self.assertFalse(table.duplicated(["ticker", "date"]).any())
            self.assertIn(("2211", "2021-02-26"), set(zip(table.ticker, table.date)))
            self.assertIn(("0050", "2021-01-13"), set(zip(table.ticker, table.date)))
            self.assertTrue(np.isfinite(table[["mfe", "peak_adverse", "min_low", "raw"]].to_numpy()).all())

        # A ticker with only 20 observed bars cannot have a fabricated 40D
        # target. It remains absent; the runtime fail-closed guard still acts.
        short = prices["2211"].iloc[:20]
        with patch.object(oracle, "_contract", return_value=(settings, None, None)), \
             patch.object(oracle, "discover_dataset_csv_inputs", return_value=(paths, [])), \
             patch.object(oracle, "load_dataset_frame", side_effect=lambda path, ticker, **kw: short if ticker == "2211" else prices[ticker]):
            truth, _ = oracle._market_truth_for_oracle(self.definition, ROOT, cutoff="2021-06-01")
        self.assertTrue(all("2211" not in set(table.ticker) for table in truth.values()))

    def test_missing_benchmark_price_is_blocked_not_neutral_filled(self):
        with patch.object(oracle, "discover_dataset_csv_inputs", return_value=([("2330", Path("2330.csv"))], [])):
            with self.assertRaisesRegex(oracle.OracleBlockedError, "Benchmark價格"):
                oracle._market_truth_for_oracle(self.definition, ROOT, cutoff="2026-03-02")

    def test_reconcile_extends_tradable_universe_without_changing_official_truth(self):
        official = pd.DataFrame({
            "ticker": ["2330"], "date": ["2025-01-02"],
            "mfe": [1.5], "peak_adverse": [0.2],
            "min_low": [-0.08], "raw": [1.5],
        })
        complete = pd.concat([official, official.assign(ticker="2211", mfe=2.0, raw=2.0)], ignore_index=True)
        expanded, extra = oracle._reconcile_oracle_truth(
            official, complete, field="pure_mfe40_provider", score_start="2025-01-01",
        )
        self.assertEqual(extra, 1)
        self.assertEqual(set(expanded.ticker), {"2330", "2211"})
        self.assertEqual(float(expanded.loc[expanded.ticker == "2211", "mfe"].iloc[0]), 2.0)
        incompatible = complete.copy()
        incompatible.loc[incompatible.ticker == "2330", "mfe"] += 0.5
        with self.assertRaisesRegex(oracle.OracleBlockedError, "不一致"):
            oracle._reconcile_oracle_truth(
                official, incompatible, field="pure_mfe40_provider", score_start="2025-01-01",
            )

    def test_listing_omitted_by_dl_provider_reaches_oracle_runtime_rank(self):
        provider = pd.DataFrame({
            "ticker": ["0050"], "date": ["2021-02-26"],
            "mfe": [1.0], "peak_adverse": [0.3],
            "min_low": [-0.07], "raw": [1.0],
        })
        price_truth = pd.concat([
            provider,
            provider.assign(ticker="2211", mfe=2.0, peak_adverse=0.2, min_low=-0.03, raw=2.0),
        ], ignore_index=True)
        market = ({field: price_truth for field in oracle._ORACLE_PROVIDER_FIELDS},
                  {"score_start": "2021-01-01", "truth_source": "synthetic"})
        with patch.object(oracle, "_market_truth_for_oracle", return_value=market), \
             patch.object(oracle, "_load_truth", return_value=provider):
            scores, evidence = oracle.build_oracle_score_tables(
                self.definition, project_root=ROOT, cutoff="2026-03-02",
            )
        self.assertEqual(evidence["cohort_rows"], 2)
        self.assertEqual(evidence["rows_outside_dl_training_provider_by_source"],
                         {field: 1 for field in oracle._ORACLE_PROVIDER_FIELDS})
        for target, _ in oracle.ORACLE_TARGETS:
            self.assertIn(("2211", "2021-02-26"), scores[target])
            with audit_oracle_truth_score_context(
                scores[target], ranking_policy=self.exact_policy,
                ranking_options=self.exact_options,
            ):
                rank = resolve_breakout_quality_candidate_rank(
                    ticker="2211", signal_date="2021-02-26",
                    information_date="2021-02-26", high_len=260,
                )
                self.assertTrue(rank["available"])
                self.assertTrue(np.isfinite(rank["score"]))
                with self.assertRaisesRegex(ValueError, "未覆蓋"):
                    resolve_breakout_quality_candidate_rank(
                        ticker="2211", signal_date="2021-02-25",
                        information_date="2021-02-25", high_len=260,
                    )

    def test_canonical_provider_loader_keeps_official_rows(self):
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
        with patch.object(oracle, "load_profile_continuous_ranker_data", return_value=fake_bundle):
            result = oracle._load_truth(self.definition, ROOT, provider, cutoff="2026-03-02")
        self.assertEqual(len(result), 1)
        self.assertEqual(result.ticker.iloc[0], "2330")

    def test_reusable_research_report_publishes_schema2_11_targets(self):
        def summary(name: str):
            return {
                "target": name, "total_return_pct": 100.0, "annualized_return_pct": 14.0,
                "mdd_pct": 12.0, "romd": 8.0, "romd_std": 0.0,
                "ev_a_r": 0.5, "seeds": 1, "yearly": [],
                "seed_observations": [{"seed": 123, "romd": 8.0, "delta_romd_vs_c96": 0.0}],
                "delta_romd_vs_c96": 0.0, "positive_seed_count": 0,
            }
        result = {
            "schema_version": 2, "status": "COMPLETED",
            "start_date": "2021-01-01", "end_date": "2025-12-22",
            "common_cohort": {
                "cohort_rows": 3,
                "mfe40_over_d40_zero_downside_positive_upside_count": 1,
                "mfe40_over_d40_zero_zero_count": 0,
            },
            "benchmark_id": "test", "seeds": [123], "limitations": ["NOT PIT"],
            "baseline": summary("C96"),
            "oracles": [summary(name) for name, _ in oracle.ORACLE_TARGETS],
        }
        from services.audit.runner import render_latest_audit_summary
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            with patch.object(oracle, "build_report", return_value=result):
                published = oracle.run_formal_audit(self.definition, project_root=base, quiet=True)
            self.assertEqual(published["schema_version"], 2)
            self.assertEqual(published["report_contract_id"], "audit.target_oracle_romd@v1")
            self.assertEqual(published["report_id"], self.definition.audit_id)
            latest = base / "outputs/audit" / self.definition.output_subdir / "latest"
            body = (latest / "audit.md").read_text(encoding="utf-8")
            self.assertIn("MFE40_over_D40", body)
            self.assertIn("MAE20", body)
            self.assertIn("MFE40/D40 zero denominator", body)
            self.assertIn("Delta RoMD", body)
            json_data = json.loads((latest / "audit.json").read_text(encoding="utf-8"))
            self.assertEqual(len(json_data["oracles"]), len(oracle.ORACLE_TARGETS))
            listing = render_latest_audit_summary(
                "breakout_quality", project_root=base, method_id=oracle.SUPPORTED_AUDIT_TYPE,
            )
            self.assertIn("存在", listing)
            self.assertIn(self.definition.report_id, listing)
            self.assertIn(self.definition.report_id, render_latest_audit_summary(
                "breakout_quality", project_root=base,
            ))

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
             patch.object(oracle, "build_oracle_score_tables", return_value=({name: {("2330", "2025-01-02"): 0.8} for name, _ in oracle.ORACLE_TARGETS}, {"cohort_rows": 4, "last_common_day": "2025-12-31", "mfe40_over_d40_zero_downside_positive_upside_count": 0, "mfe40_over_d40_zero_zero_count": 0})), \
             patch.object(oracle, "load_strategy_param_evaluation_view", return_value=({"kind": "rolling_active_param_ensemble", "payload": {}}, "a" * 64)), \
             patch.object(oracle, "_validate_requested_param_policy"), \
             patch.object(oracle, "get_active_param_ensemble_date_range", return_value=("2020-01-01", "2026-03-02")), \
             patch.object(oracle, "_build_controlled_param_source_pair", return_value=("rolling_active_param_ensemble", "base", "oracle", None, None, None)), \
             patch.object(oracle, "_run_scenario", side_effect=fake_replay), \
             patch.object(oracle, "price_inventory_sha256", return_value="b" * 64), \
             patch.object(oracle, "_metrics", side_effect=fake_metrics):
            with tempfile.TemporaryDirectory() as tmp:
                report = oracle.build_report(self.definition, project_root=Path(tmp), quiet=True)
                original_calls = len(seen)
                reused = oracle.build_report(self.definition, project_root=Path(tmp), quiet=True)
                self.assertEqual(len(seen), original_calls, "identical replay must REUSE all arms")
                self.assertEqual(reused["baseline"], report["baseline"])
                self.assertEqual(reused["oracles"], report["oracles"])
        self.assertEqual(len(seen), len(seeds) * (1 + len(oracle.ORACLE_TARGETS)))
        self.assertEqual(report["seeds"], list(seeds))
        self.assertEqual(report["schema_version"], 2)
        self.assertEqual(len(report["oracles"]), 11)
        self.assertEqual(report["end_date"], "2025-12-31")
        self.assertEqual(report["baseline"]["romd_std"], report["oracles"][0]["romd_std"])
        self.assertEqual(report["oracles"][0]["delta_romd_vs_c96"], 1.5)
        self.assertEqual(report["oracles"][0]["positive_seed_count"], len(seeds))
        self.assertIn("Oracle Performance", oracle.render_result(report, target="markdown"))
        self.assertIn("MFE40_over_D40", oracle.render_result(report, target="markdown"))
        self.assertIn("MAE20", oracle.render_result(report, target="console"))
        self.assertIn("MFE40/D40 zero denominator", oracle.render_result(report, target="console"))
        self.assertIn("RoMD Mean", oracle.render_result(report, target="console"))
        json.dumps(report, ensure_ascii=False)
        self.assertNotEqual(get_breakout_quality_ranking_source_context().score_source,
                            SCORE_SOURCE_AUDIT_ORACLE_FUTURE_TRUTH)


class OracleReplayEvidenceReuseCases(TestCase):
    """Exact-identity per-arm REUSE and conservative historical-V1 compatibility."""

    @staticmethod
    def _observation(target, seed):
        romd = float(seed) + (0.0 if target == "C96" else 2.0)
        observation = {
            "target": target, "seed": seed,
            "total_return_pct": 123.0, "annualized_return_pct": 16.0,
            "mdd_pct": 12.0, "romd": romd, "ev_a_r": 0.7,
            "trades": 100, "exposure_pct": 75.0,
            "yearly": [{"year": 2021, "strategy_return_pct": 14.0, "is_full_year": True}],
        }
        if target != "C96":
            observation["delta_romd_vs_c96"] = 2.0
        return observation

    @staticmethod
    def _identity(seed=1, target="U40", dataset_sha="a" * 64, param_sha="b" * 64):
        from services.audit.oracle_replay_artifacts import replay_identity
        return replay_identity(
            dataset="full", dataset_sha256=dataset_sha,
            target=target, definition=dict(oracle.ORACLE_TARGETS),
            settings_fingerprint="fixture", start="2021-01-01", end="2025-12-22",
            benchmark="end_to_end_v1", param_sha256=param_sha, seed=seed,
            cohort={"cohort_rows": 123, "research_cutoff": "2026-03-02"},
            ranking_policy="score", ranking_options={"preserve_k_r0": True},
            evaluation_mode="rolling", max_positions=10,
            replay_policy={"exact_k_r0": True},
        )

    def test_content_sha_changes_even_when_csv_length_is_unchanged(self):
        from services.audit import oracle_replay_artifacts as artifacts
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / "2330.csv"
            csv.write_text("100,200\n", encoding="utf-8")
            with patch.object(artifacts, "discover_dataset_csv_inputs", return_value=([("2330", csv)], [])):
                before = artifacts.price_inventory_sha256(Path(tmp), "full")
                csv.write_text("100,201\n", encoding="utf-8")
                after = artifacts.price_inventory_sha256(Path(tmp), "full")
                self.assertNotEqual(before, after)
                self.assertEqual(after, artifacts.price_inventory_sha256(Path(tmp), "full"))

    def test_partial_unit_cache_reuses_only_identical_input_identity(self):
        from services.audit.oracle_replay_artifacts import OracleReplayStore
        with tempfile.TemporaryDirectory() as tmp:
            cache = OracleReplayStore(Path(tmp), "breakout_quality/reusable/target_oracle_romd")
            identity = self._identity()
            record = self._observation("U40", 1)
            self.assertIsNone(cache.read(identity))
            cache.write(identity, record, origin="canonical_replay")
            self.assertEqual(cache.read(identity), record)
            cache.write(identity, record, origin="canonical_replay")  # idempotent
            self.assertIsNone(cache.read(self._identity(dataset_sha="c" * 64)))
            self.assertTrue(cache.has_other_identity(self._identity(dataset_sha="c" * 64)))
            self.assertIsNone(cache.read(self._identity(param_sha="d" * 64)))
            with self.assertRaisesRegex(ValueError, "conflict"):
                cache.write(identity, {**record, "romd": 999}, origin="canonical_replay")
            location = cache._path(identity)
            raw = json.loads(location.read_text(encoding="utf-8"))
            raw["observation"]["romd"] = 999
            location.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "corrupt"):
                cache.read(identity)

    def test_history_reuses_only_completed_matching_seven_target_seed_records(self):
        from services.audit.oracle_replay_artifacts import HistoricalOracleIndex
        seeds = [1, 2, 3, 4]
        description = dict(oracle.ORACLE_TARGETS)
        cohort = {
            "cohort_rows": 643617,
            "first_common_day": "2020-12-31", "last_common_day": "2025-12-22",
            "research_cutoff": "2026-03-02", "risk_budget_return": 0.03,
            "providers": {"40d": "provider-a"}, "scoring": "shared percentile",
            "market_truth": {"price_ticker_count": 556},
            "rows_outside_dl_training_provider_by_source": {"40d": 5},
        }
        context = {
            "benchmark": "end_to_end_v1", "strategy_profile_id": "broker_cash_dl_conversion_rolling",
            "dataset": "full", "start_date": "2021-01-01", "end_date": "2025-12-22",
            "seeds": seeds, "param_sha_by_seed": {seed: "a" * 64 for seed in seeds},
            "cohort": cohort, "definitions": description,
        }
        historical = {
            "schema_version": 1, "status": "COMPLETED", "read_only": True,
            "hindsight_non_pit": True, "report_id": "AUD-target-ground-truth-romd",
            "report_type": "target_ground_truth_strategy_romd",
            "strategy_profile_id": context["strategy_profile_id"],
            "benchmark_id": context["benchmark"], "dataset": "full",
            "start_date": context["start_date"], "end_date": context["end_date"],
            "seeds": seeds,
            "param_artifacts": [{"seed": seed, "param_sha256": "a" * 64} for seed in seeds],
            "common_cohort": cohort,
            "definitions": {key: description[key] for key in list(description)[:7]},
            "baseline": oracle._aggregate([self._observation("C96", seed) for seed in seeds], target="C96"),
            "oracles": [oracle._aggregate([self._observation(key, seed) for seed in seeds], target=key)
                        for key in list(description)[:7]],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs/audit/breakout_quality/one_time/target_oracle_romd/latest"
            root.mkdir(parents=True)
            path = root / "audit.json"
            path.write_text(json.dumps(historical), encoding="utf-8")
            index = HistoricalOracleIndex(Path(tmp))
            self.assertEqual(index.lookup(target="C96", seed=3, context=context)["romd"], 3.0)
            self.assertEqual(index.lookup(target="U40", seed=3, context=context)["romd"], 5.0)
            self.assertIsNone(index.lookup(target="MFE20", seed=3, context=context))
            mutated = {**context, "param_sha_by_seed": {seed: "b" * 64 for seed in seeds}}
            self.assertIsNone(index.lookup(target="U40", seed=3, context=mutated))
            mutated = {**context, "cohort": {**cohort, "cohort_rows": 999}}
            self.assertIsNone(index.lookup(target="U40", seed=3, context=mutated))
            altered = {**historical, "oracles": [dict(item) for item in historical["oracles"]]}
            altered["oracles"][0]["romd"] += 1.0
            path.write_text(json.dumps(altered), encoding="utf-8")
            index = HistoricalOracleIndex(Path(tmp))
            with self.assertRaisesRegex(ValueError, "incomplete / corrupted"):
                index.lookup(target="AO40_EqualRank", seed=3, context=context)

    def test_history_seven_targets_runs_only_four_new_targets_and_then_full_reuse(self):
        seeds = (11, 22, 33, 44)
        definitions = dict(oracle.ORACLE_TARGETS)
        common = {"cohort_rows": 4, "first_common_day": "2021-01-01",
                  "last_common_day": "2025-12-22", "research_cutoff": "2026-03-02",
                  "risk_budget_return": 0.03, "providers": {}, "scoring": "test",
                  "market_truth": {}, "rows_outside_dl_training_provider_by_source": {},
                  "mfe40_over_d40_zero_downside_positive_upside_count": 0,
                  "mfe40_over_d40_zero_zero_count": 0}
        class Benchmark:
            benchmark_id = "end_to_end_v1"
        archived = {
            "schema_version": 1, "status": "COMPLETED", "read_only": True,
            "hindsight_non_pit": True, "report_id": "AUD-target-ground-truth-romd",
            "report_type": oracle.SUPPORTED_AUDIT_TYPE,
            "strategy_profile_id": "broker_cash_dl_conversion_rolling",
            "benchmark_id": Benchmark.benchmark_id, "dataset": "full",
            "start_date": "2021-01-01", "end_date": "2025-12-22",
            "seeds": list(seeds), "common_cohort": common,
            "param_artifacts": [{"seed": seed, "param_sha256": "a" * 64} for seed in seeds],
            "definitions": {k: definitions[k] for k in list(definitions)[:7]},
            "baseline": oracle._aggregate([self._observation("C96", seed) for seed in seeds], target="C96"),
            "oracles": [oracle._aggregate([self._observation(k, seed) for seed in seeds], target=k)
                        for k in list(definitions)[:7]],
        }
        seen = []
        def replay(*, name, **kwargs):
            seen.append(name)
            return {"name": name}
        def metrics(result, *, target):
            seed = int(result["name"].split("seed=")[1].split()[0])
            return self._observation(target, seed)
        definition = next(d for d in get_reusable_audit_definitions("breakout_quality")
                          if d.audit_type == oracle.SUPPORTED_AUDIT_TYPE)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            previous = root / "outputs/audit/breakout_quality/one_time/target_oracle_romd/latest"
            previous.mkdir(parents=True)
            (previous / "audit.json").write_text(json.dumps(archived), encoding="utf-8")
            with patch.object(oracle, "collect_status", return_value={"status": "READY"}), \
                 patch.object(oracle, "_robustness_artifacts", return_value=(Benchmark(), [(seed, Path(f"seed_{seed}.json")) for seed in seeds])), \
                 patch.object(oracle, "build_oracle_score_tables", return_value=({key: {("2330", "2025-01-02"): 0.8} for key in definitions}, common)), \
                 patch.object(oracle, "load_strategy_param_evaluation_view", return_value=({"kind": "rolling_active_param_ensemble", "payload": {}}, "a" * 64)), \
                 patch.object(oracle, "_validate_requested_param_policy"), \
                 patch.object(oracle, "get_active_param_ensemble_date_range", return_value=("2020-01-01", "2026-03-02")), \
                 patch.object(oracle, "_build_controlled_param_source_pair", return_value=("rolling_active_param_ensemble", "base", "oracle", None, None, None)), \
                 patch.object(oracle, "price_inventory_sha256", return_value="b" * 64), \
                 patch.object(oracle, "_run_scenario", side_effect=replay), \
                 patch.object(oracle, "_metrics", side_effect=metrics):
                result = oracle.build_report(definition, project_root=root, quiet=True)
                self.assertEqual(len(seen), len(seeds) * 4, "V1 C96 + 7 targets must not replay")
                self.assertEqual(len(result["oracles"]), 11)
                self.assertEqual(result["baseline"]["seed_observations"][0]["romd"], 11)
                result_again = oracle.build_report(definition, project_root=root, quiet=True)
                self.assertEqual(len(seen), 16, "completed 11-target rerun must reuse all 48")
                self.assertEqual(result_again["oracles"], result["oracles"])

    def test_partial_replay_is_saved_before_later_failure_and_resumes(self):
        seeds = (11, 22, 33, 44)
        class Benchmark:
            benchmark_id = "end_to_end_v1"
        definition = next(d for d in get_reusable_audit_definitions("breakout_quality")
                          if d.audit_type == oracle.SUPPORTED_AUDIT_TYPE)
        common = {"cohort_rows": 4, "last_common_day": "2025-12-22"}
        state = {"calls": 0, "crash": True}
        def replay(*, name, **kwargs):
            state["calls"] += 1
            if state["crash"] and state["calls"] == 10:
                raise RuntimeError("injected replay interruption")
            return {"name": name}
        def metrics(result, *, target):
            seed = int(result["name"].split("seed=")[1].split()[0])
            return self._observation(target, seed)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(oracle, "collect_status", return_value={"status": "READY"}), \
                 patch.object(oracle, "_robustness_artifacts", return_value=(Benchmark(), [(seed, Path(f"seed_{seed}.json")) for seed in seeds])), \
                 patch.object(oracle, "build_oracle_score_tables", return_value=({key: {("2330", "2025-01-02"): 0.8} for key, _ in oracle.ORACLE_TARGETS}, common)), \
                 patch.object(oracle, "load_strategy_param_evaluation_view", return_value=({"kind": "rolling_active_param_ensemble", "payload": {}}, "a" * 64)), \
                 patch.object(oracle, "_validate_requested_param_policy"), \
                 patch.object(oracle, "get_active_param_ensemble_date_range", return_value=("2020-01-01", "2026-03-02")), \
                 patch.object(oracle, "_build_controlled_param_source_pair", return_value=("rolling_active_param_ensemble", "base", "oracle", None, None, None)), \
                 patch.object(oracle, "price_inventory_sha256", return_value="b" * 64), \
                 patch.object(oracle, "_run_scenario", side_effect=replay), \
                 patch.object(oracle, "_metrics", side_effect=metrics):
                with self.assertRaisesRegex(RuntimeError, "injected replay interruption"):
                    oracle.build_report(definition, project_root=Path(tmp), quiet=True)
                self.assertEqual(state["calls"], 10)
                state["crash"] = False
                resumed = oracle.build_report(definition, project_root=Path(tmp), quiet=True)
                self.assertEqual(state["calls"], 49, "the first nine completed arms must not replay")
                self.assertEqual(len(resumed["oracles"]), 11)
                oracle.build_report(definition, project_root=Path(tmp), quiet=True)
                self.assertEqual(state["calls"], 49)


if __name__ == "__main__":
    main(verbosity=2)
