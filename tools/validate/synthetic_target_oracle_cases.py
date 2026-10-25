"""Isolated contract regression for reusable Oracle target → RoMD Audit.

Run directly; requires no historical market data and does not touch formal
Test Suite frozen entry checks, PIT artifacts, or portfolio model fitting.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from dataclasses import replace
import shutil
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

# Historical stock-day Oracle regressions must not instantiate a P16 path
# provider in a temporary no-market fixture. The active suite is tested below.
_ACTIVE_ORACLE_DEFINITION = next(
    d for d in get_reusable_audit_definitions("breakout_quality")
    if d.audit_type == oracle.SUPPORTED_AUDIT_TYPE
)
_NON_PRICE_SCORE_TARGETS = frozenset(
    set(oracle._oracle_comparison_plan(_ACTIVE_ORACLE_DEFINITION)["seed_sources"])
    | set(oracle._oracle_comparison_plan(_ACTIVE_ORACLE_DEFINITION)["daily_sources"])
)
LEGACY_ORACLE_TARGETS = tuple(
    item for item in oracle.ORACLE_TARGETS if item[0] not in _NON_PRICE_SCORE_TARGETS
)
from services.audit.runner import collect_audit_menu_state, collect_audit_status
from services.audit.catalog import get_audit_methods


def _legacy_definition():
    """Historical fixed-nine fixtures, independent of currently active Target rows."""
    active = next(d for d in get_reusable_audit_definitions("breakout_quality")
                  if d.audit_type == oracle.SUPPORTED_AUDIT_TYPE)
    return replace(active, source={**active.source, "oracle_comparisons": tuple(
        item for item in active.source["oracle_comparisons"]
        if item["target"] not in _NON_PRICE_SCORE_TARGETS)})


def _mock_panels(common40):
    """Canonical one-window fixture: Exact and Direct share the identical daily truth."""
    return ({name: {("2330", "2025-01-02"): 0.8}
             for name, _ in LEGACY_ORACLE_TARGETS}, common40)


class OracleAuditContractCases(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.active_definition = next(
            definition for definition in get_reusable_audit_definitions("breakout_quality")
            if definition.audit_type == oracle.SUPPORTED_AUDIT_TYPE
        )
        cls.definition = _legacy_definition()
        cls.exact_policy = oracle._contract(cls.definition)[2].dl_runtime_mode
        cls.exact_options = oracle._contract(cls.definition)[2].dl_runtime_options

    def test_oracle_is_reusable_only_with_a_single_registered_run_route(self):
        self.assertEqual(collect_audit_menu_state("breakout_quality")["enabled_count"], 0)
        self.assertEqual(
            collect_audit_menu_state("breakout_quality", method_id=oracle.SUPPORTED_AUDIT_TYPE)["enabled_count"], 1
        )
        self.assertEqual(
            [x.method_id for x in get_audit_methods(menu_location="research_root")],
            [oracle.SUPPORTED_AUDIT_TYPE],
        )
        self.assertNotIn(oracle.SUPPORTED_AUDIT_TYPE,
                         [x.method_id for x in get_audit_methods(menu_location="audit")])
        self.assertFalse(any(
            definition.audit_type == oracle.SUPPORTED_AUDIT_TYPE
            for definition in get_audit_definitions("breakout_quality")
        ))
        self.assertEqual(
            {name for name, _ in LEGACY_ORACLE_TARGETS},
            {"CG40_EqualRank", "MFE40_Control", "R40", "T40",
             "T40-MFE", "MFE40_Direct", "R40_Direct", "T40_Direct", "T40-MFE_Direct"},
        )
        self.assertEqual(
            oracle._oracle_comparison_plan(self.active_definition)["seed_recipes"],
            {"MinTrade40_NetR": ("trade_path_realized_net_r", "event_targets")},
        )
        self.assertEqual(oracle.ORACLE_DIRECT_SCORE_SOURCES, {
            "MFE40_Direct": "MFE40_Control", "R40_Direct": "R40", "T40_Direct": "T40",
            "T40-MFE_Direct": "T40-MFE",
            "MinTrade40_NetR_Direct": "MinTrade40_NetR",
            "DailyUniversal_MinTrade40_NetR_Direct": "DailyUniversal_MinTrade40_NetR",
        })
        self.assertEqual(self.definition.output_subdir, "breakout_quality/reusable/target_oracle_romd")
        with tempfile.TemporaryDirectory() as tmp:
            status = collect_audit_status(
                "breakout_quality", project_root=Path(tmp), method_id=oracle.SUPPORTED_AUDIT_TYPE,
            )
            self.assertEqual(status["overall_status"], "BLOCKED")
            self.assertIn("full dataset", status["rows"][0]["reason"])

    def test_min_trade40_context_uses_oracle_authorized_p16_sources(self):
        settings = SimpleNamespace(
            dataset='full', start_date='2021-01-01', end_date='2026-03-02',
        )
        baseline = SimpleNamespace(param_policy='base-finalist-best')
        benchmark = SimpleNamespace(benchmark_id='end_to_end_v1')
        parameter_paths = [(11, Path('test/p16.json'))]
        prepared = [(11, 'b' * 64, 'rolling_oos_param_schedule', {}, {})]
        evidence = [{'seed': 11, 'param_sha256': 'b' * 64}]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(oracle, '_contract', return_value=(settings, baseline, object())) as contract, \
                 patch.object(oracle, '_robustness_artifacts', return_value=(benchmark, parameter_paths)) as robustness, \
                 patch.object(oracle, '_prepare_oracle_parameters', return_value=(prepared, evidence)) as prepare:
                ctx = oracle._resolve_min_trade40_p16_context(
                    self.definition, project_root=root,
                )
            contract.assert_called_once_with(self.definition)
            robustness.assert_called_once_with(self.definition, root.resolve())
            prepare.assert_called_once_with(
                self.definition, root=root.resolve(), parameter_paths=parameter_paths,
                start=settings.start_date, end=settings.end_date,
                base_arm=baseline,
            )
        self.assertEqual(ctx.prepared, tuple(prepared))
        self.assertEqual(ctx.parameter_evidence, tuple(evidence))
        self.assertEqual(ctx.benchmark_id, benchmark.benchmark_id)
        self.assertEqual(ctx.start_date, settings.start_date)
        self.assertEqual(ctx.cutoff_date, settings.end_date)

    def test_oracle_cache_accepts_safe_hyphenated_target_without_path_escape(self):
        from services.audit.oracle_replay_artifacts import OracleReplayStore
        with tempfile.TemporaryDirectory() as tmp:
            store = OracleReplayStore(Path(tmp), self.definition.output_subdir)
            identity = {"seed": 11, "target": "T40-MFE"}
            self.assertEqual(store._path(identity).parent.name, "11")
            self.assertTrue(store._path(identity).name.startswith("T40-MFE_"))
            direct = {**identity, "target": "T40-MFE_Direct"}
            self.assertTrue(store._path(direct).name.startswith("T40-MFE_Direct_"))
            self.assertNotEqual(store._path(identity), store._path(direct))
            for unsafe in ("../T40-MFE", "T40/MFE", "T40.MFE", "T40 MFE"):
                with self.assertRaisesRegex(ValueError, "invalid cache target"):
                    store._path({**identity, "target": unsafe})

    def test_summary_display_is_forwarded_only_to_declared_formal_oracle(self):
        from services.audit import runner
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(runner, "collect_audit_status", return_value={"overall_status": "READY"}), \
             patch.object(oracle, "run_formal_audit", return_value={"status": "COMPLETED"}) as execute:
            runner.run_enabled_audits(
                "breakout_quality", project_root=Path(tmp),
                method_id=oracle.SUPPORTED_AUDIT_TYPE, preflight_display="summary")
            self.assertEqual(execute.call_args.kwargs["preflight_display"], "summary")
            with self.assertRaisesRegex(ValueError, "簡要 Replay preflight"):
                runner.run_enabled_audits(
                    "breakout_quality", project_root=Path(tmp),
                    method_id="opportunity_selection_attribution", preflight_display="summary")

    def test_formal_runner_does_not_advertise_secondary_permission_capability(self):
        from inspect import signature
        from services.audit import runner
        self.assertNotIn("before_replay", signature(runner.run_enabled_audits).parameters)
        self.assertNotIn("before_replay", signature(oracle.run_formal_audit).parameters)
        self.assertNotIn("before_replay", signature(oracle.build_report).parameters)

    def test_preflight_action_matrix_and_canonical_best_worst_metric_colors(self):
        from core.research_report_contract import report_contract
        from services.audit.reusable_report import format_comparison_rows
        from core.report_style import best_worst_signals
        from core.display_common import _strip_ansi
        self.assertEqual(best_worst_signals({"a": 2, "b": 2}, preference="higher"), {})
        perf = report_contract("audit.target_oracle_romd").sections[0].tables[0]
        values = [
            {"target": "X", "romd": 5.0, "mdd_pct": 20.0, "total_return_pct": 80},
            {"target": "Y", "romd": 9.0, "mdd_pct": 8.0, "total_return_pct": 120},
            {"target": "Z", "romd": 7.0, "mdd_pct": 12.0, "total_return_pct": 100},
        ]
        cells = format_comparison_rows(perf, values, target="markdown")
        columns = {col.key: i for i, col in enumerate(perf.columns)}
        self.assertIn("#188038", cells[1][columns["romd"]])
        self.assertIn("#C62828", cells[0][columns["romd"]])
        self.assertIn("#188038", cells[1][columns["mdd_pct"]])
        self.assertIn("#C62828", cells[0][columns["mdd_pct"]])
        self.assertNotIn("<span", cells[2][columns["romd"]])
        self.assertNotIn("<span", cells[0][columns["target"]])
        # Year-group values must not be compared to results from other years.
        grouped = format_comparison_rows(perf, values, target="markdown", groups=["2022", "2022", "2023"])
        self.assertNotIn("<span", grouped[2][columns["romd"]])
        plan = oracle.render_replay_preflight(
            label="40D", seeds=[11, 22], targets=["C96", "R40", "T40"],
            decisions={
                (11, "C96"): "REUSE", (22, "C96"): "REUSE",
                (11, "R40"): "RUN", (22, "R40"): "REUSE",
                (11, "T40"): "BLOCKED", (22, "T40"): "RUN",
            },
        )
        plain = _strip_ansi(plan)
        self.assertIn("S1=11", plain)
        self.assertIn("S2=22", plain)
        self.assertIn("REUSE 3 / RUN（需計算）2 / BLOCKED 1", plain)
        self.assertLess(plain.index("C96"), plain.index("R40"))
        self.assertLess(plain.index("R40"), plain.index("T40"))

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

    def test_direct_selector_is_same_canonical_c101_and_audit_only(self):
        from core.breakout_quality_ranking_context import audit_oracle_ranking_policy_context
        settings, base, exact = oracle._contract(self.definition)
        direct = oracle._direct_arm(self.definition, settings, base, exact)
        self.assertEqual(direct.dl_runtime_mode, "resource-aware-continuous-score-no-k-no-r0")
        self.assertIs(direct.dl_runtime_options["preserve_k"], False)
        self.assertIs(direct.dl_runtime_options["preserve_r0"], False)
        with audit_oracle_truth_score_context(
            {("2330", "2025-01-02"): 0.8}, ranking_policy=direct.dl_runtime_mode,
            ranking_options=direct.dl_runtime_options,
        ):
            state = get_breakout_quality_ranking_source_context()
            self.assertEqual(state.score_source, SCORE_SOURCE_AUDIT_ORACLE_FUTURE_TRUTH)
            self.assertEqual(state.ranking_policy, direct.dl_runtime_mode)
        with self.assertRaisesRegex(ValueError, "只允許"):
            with audit_oracle_ranking_policy_context(ranking_policy="score"):
                pass
        self.assertNotIn(SCORE_SOURCE_AUDIT_ORACLE_FUTURE_TRUTH, SUPPORTED_RANKING_SCORE_SOURCES)

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
        market = {field: self._truth(field) for field in oracle._oracle_comparison_plan(self.definition)["providers"]}
        market["pure_mfe40_provider"] = market["pure_mfe40_provider"].assign(terminal_r=[1.5, 2.0, 0.5])
        if omit_first_breach_last:
            market["pure_mfe40_provider"] = market["pure_mfe40_provider"].iloc[:2]
        return market, {"score_start": "2025-01-01", "truth_source": "test"}

    def test_oracle_target_scores_use_matched_cohort_and_correct_directions(self):
        with patch.object(oracle, "_load_truth", side_effect=lambda definition, root, source_field, cutoff: self._truth(source_field)), \
             patch.object(oracle, "_market_truth_for_oracle", return_value=self._mock_market()):
            scores, evidence = oracle.build_oracle_score_tables(
                self.definition, project_root=ROOT, cutoff="2026-03-02",
            )
        self.assertEqual(evidence["cohort_rows"], 3)
        self.assertEqual(evidence["mfe40_over_d40_zero_downside_positive_upside_count"], 0)
        self.assertEqual(set(scores), {label for label, _ in LEGACY_ORACLE_TARGETS})
        key = lambda ticker: (ticker, "2025-01-02")
        self.assertEqual(scores["MFE40_Control"][key("2330")], 1.0)
        self.assertEqual(scores["CG40_EqualRank"][key("2330")], 1.0)
        self.assertNotIn("CH20_EqualRank", scores)
        self.assertNotIn("AO40_EqualRank", scores)
        self.assertNotIn("U40", scores)
        self.assertNotIn("FirstBreach_NetR", scores)
        self.assertEqual(scores["R40"][key("2317")], 1.0)
        self.assertEqual(scores["T40"][key("2330")], 0.5)  # All three reach MAE>=1R.
        for direct_name, exact_name in oracle.ORACLE_DIRECT_SCORE_SOURCES.items():
            if exact_name in scores:
                self.assertIs(scores[direct_name], scores[exact_name], "Direct must reuse identical truth")
        self.assertTrue(all(0.0 <= v <= 1.0 for table in scores.values() for v in table.values()))

    def test_selected_score_providers_do_not_run_retired_truth_dependencies(self):
        market, evidence = self._mock_market()
        evidence.update({"price_ticker_count": 3, "unreadable_tickers": {}})
        with patch.object(oracle, "_load_truth", side_effect=lambda definition, root, source_field, cutoff: self._truth(source_field)) as loader, \
             patch.object(oracle, "_market_truth_for_oracle", return_value=(market, evidence)):
            scores40, cohort40 = oracle._build_oracle_score_panels(
                self.definition, project_root=ROOT, cutoff="2026-03-02",
            )
        self.assertEqual(cohort40["cohort_rows"], 3)
        self.assertEqual(set(cohort40["providers"]), {"40d"})
        self.assertEqual(loader.call_count, 1)
        self.assertFalse(any(name.endswith("60") or name.startswith("CH20") for name in scores40))
        self.assertEqual(len(scores40), len(LEGACY_ORACLE_TARGETS))

    def test_provider_dependencies_follow_selected_comparisons_not_hardcoded(self):
        from dataclasses import replace
        original = list(self.definition.source["oracle_comparisons"])
        mfe40 = next(item for item in original if item["target"] == "MFE40_Control")
        # Remove all other Exact and Direct arms; only 40D remains.
        subset = replace(self.definition, source={**self.definition.source,
                                                 "oracle_comparisons": (mfe40,)})
        p1 = oracle._oracle_comparison_plan(subset)
        self.assertEqual(p1["providers"], ("pure_mfe40_provider",))
        self.assertEqual([name for name, _ in p1["targets"]], ["MFE40_Control"])
        toggled = replace(self.definition, source={**self.definition.source,
            "oracle_comparisons": tuple({**item, "enabled": item["target"] == "MFE40_Control"}
                                        for item in original)})
        self.assertEqual(oracle._oracle_comparison_plan(toggled)["targets"], p1["targets"])
        # Add a test-only score referencing the catalogued 20D provider.
        # Same code path discovers the new dependency without a new allowlist.
        addition = {"target": "MFE20_Test", "description": "20D canonical test",
                    "score": "mfe", "provider": "mfe20_d20_provider", "selector": "exact"}
        extended = replace(self.definition, source={**self.definition.source,
                                                   "oracle_comparisons": (mfe40, addition)})
        p2 = oracle._oracle_comparison_plan(extended)
        self.assertEqual(p2["providers"], ("pure_mfe40_provider", "mfe20_d20_provider"))
        with patch.object(oracle, "_load_truth", side_effect=lambda d, root, field, cutoff: self._truth(field)) as loader, \
             patch.object(oracle, "_market_truth_for_oracle", return_value=({
                 **self._mock_market()[0], "mfe20_d20_provider": self._truth("mfe20_d20_provider")},
                 self._mock_market()[1])):
            scores, cohort = oracle._build_oracle_score_panels(extended, project_root=ROOT,
                                                              cutoff="2026-03-02")
        self.assertEqual(set(scores), {"MFE40_Control", "MFE20_Test"})
        self.assertEqual(loader.call_count, 2)
        self.assertEqual(set(cohort["providers"]), {"40d", "20d"})
        self.assertNotIn("first_breach", cohort["providers"])

    def test_complete_verified_report_skips_all_truth_provider_recalculation(self):
        seeds = (11, 22, 33, 44)
        class Benchmark:
            benchmark_id = "end_to_end_v1"

        source = self.definition.source
        full = {
            "cohort_rows": 3, "first_common_day": "2021-01-04",
            "last_common_day": "2025-12-22", "research_cutoff": "2026-03-02",
            "risk_budget_return": 0.1,
            "providers": {"40d": source["pure_mfe40_provider"]},
            "scoring": "All targets final-ranked by average_zero_based percentile within common-day universe",
            "oracle_only": True,
            "market_truth": {"score_start": "2020-12-31", "mature_rows_by_provider": {
                "pure_mfe40_provider": 3}},
            "rows_outside_dl_training_provider_by_source": {"pure_mfe40_provider": 0},
        }
        outcomes = []
        def fake_replay(*, name, **kwargs):
            outcomes.append(name)
            return {"name": name}
        def fake_metrics(replay, *, target):
            seed = int(replay["name"].split("seed=")[1].split()[0])
            return {"target": target, "total_return_pct": 120.0,
                    "annualized_return_pct": 15.0, "mdd_pct": 12.0,
                    "romd": (8.0 if target == "C96" else 12.0) + seeds.index(seed) * 0.2,
                    "ev_a_r": 0.4, "trades": 50, "exposure_pct": 45.0,
                    "yearly": [{"year": 2021, "strategy_return_pct": 4.0,
                                "is_full_year": True}]}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(oracle, "collect_status", return_value={"status": "READY"}), \
                 patch.object(oracle, "_robustness_artifacts", return_value=(Benchmark(), [
                     (seed, Path(f"seed_{seed}.json")) for seed in seeds])), \
                 patch.object(oracle, "_build_oracle_score_panels", return_value=_mock_panels(full)) as panels, \
                 patch.object(oracle, "load_strategy_param_evaluation_view", return_value=(
                     {"kind": "rolling_active_param_ensemble", "payload": {}}, "a" * 64)), \
                 patch.object(oracle, "_validate_requested_param_policy"), \
                 patch.object(oracle, "get_active_param_ensemble_date_range", return_value=(
                     "2020-01-01", "2026-03-02")), \
                 patch.object(oracle, "_build_controlled_param_source_pair", return_value=(
                     "rolling_active_param_ensemble", "base", "oracle", None, None, None)), \
                 patch.object(oracle, "price_inventory_sha256", return_value="b" * 64), \
                 patch.object(oracle, "_run_scenario", side_effect=fake_replay), \
                 patch.object(oracle, "_metrics", side_effect=fake_metrics), \
                 patch("services.audit.min_trade40_target.build_seed_oracle_scores",
                       side_effect=AssertionError("Legacy Oracle REUSE must not build seed-event truth")) as event_validation:
                first = oracle.run_formal_audit(self.definition, project_root=root, quiet=True)
                self.assertEqual(len(outcomes), 40)
                event_validation.assert_not_called()
                self.assertEqual(panels.call_count, 1)
                second = oracle.build_report(self.definition, project_root=root, quiet=True)
                self.assertEqual(len(outcomes), 40)
                self.assertEqual(panels.call_count, 1, "all REUSE must skip even 40D Truth")
                self.assertEqual(second["oracles"], first["oracles"])
                # A reduced comparison list must reuse the SAME certified
                # historical seed-level replays without any provider work.
                from dataclasses import replace
                selected = tuple(item for item in self.definition.source["oracle_comparisons"]
                                 if item["target"] in {"MFE40_Control", "R40"})
                subset = replace(self.definition, source={**self.definition.source,
                                                          "oracle_comparisons": selected})
                reduced = oracle.build_report(subset, project_root=root, quiet=True)
                self.assertEqual([row["target"] for row in reduced["oracles"]],
                                 ["MFE40_Control", "R40"])
                self.assertEqual(len(reduced["definitions"]), 2)
                self.assertEqual(len(outcomes), 40)
                self.assertEqual(panels.call_count, 1)
                with patch.object(oracle, "price_inventory_sha256", return_value="c" * 64):
                    with self.assertRaises(oracle.OracleBlockedError):
                        oracle.build_report(self.definition, project_root=root, quiet=True)
                self.assertEqual(len(outcomes), 40)

    def test_retired_provider_cohort_reuse_requires_mathematical_proof(self):
        from services.audit.oracle_replay_artifacts import oracle_cohorts_compatible
        new = {"cohort_rows": 3, "first_common_day": "2021-01-01", "last_common_day": "2021-05-01",
               "research_cutoff": "2021-06-01", "risk_budget_return": 0.1,
               "providers": {"40d": "mfe_40"}, "scoring": "percentile",
               "market_truth": {"score_start": "2020-12-31", "mature_rows_by_provider": {"pure_mfe40_provider": 3}},
               "rows_outside_dl_training_provider_by_source": {"pure_mfe40_provider": 1}}
        old = {**new, "providers": {"40d": "mfe_40", "20d": "mfe_20", "first_breach": "breach"},
               "market_truth": {"score_start": "2020-12-31", "mature_rows_by_provider": {
                   "pure_mfe40_provider": 3, "mfe20_d20_provider": 3, "first_breach_provider": 3}},
               "rows_outside_dl_training_provider_by_source": {
                   "pure_mfe40_provider": 1, "mfe20_d20_provider": 1, "first_breach_provider": 1}}
        self.assertTrue(oracle_cohorts_compatible(old, new))
        not_proven = {**old, "market_truth": {"score_start": "2020-12-31", "mature_rows_by_provider": {
            "pure_mfe40_provider": 4, "mfe20_d20_provider": 3, "first_breach_provider": 3}}}
        self.assertFalse(oracle_cohorts_compatible(not_proven, new))
        self.assertFalse(oracle_cohorts_compatible({**old, "providers": {"40d": "changed"}}, new))

    def test_t40_terminal_return_with_mae_clamp_and_negative_return(self):
        # The actual 40D stop risk is a *research* fixed 1R, not an ATR stop.
        # At MAE>=1R, the result must equal -1R irrespective of positive R40.
        r40 = pd.Series([1.5, 1.5, 1.5, 1.5, 1.5, -0.4])
        safety = pd.Series([0.0, -0.2, -0.5, -0.9, -1.2, -0.5])
        target = oracle._terminal_mae_gate_r(r40, safety)
        np.testing.assert_allclose(target.to_numpy(), [1.5, 1.0, 0.25, -0.75, -1.0, -0.7])
        with self.assertRaisesRegex(oracle.OracleBlockedError, "nonnegative MAE40"):
            oracle._terminal_mae_gate_r(pd.Series([1.0]), pd.Series([0.2]))
        with self.assertRaisesRegex(oracle.OracleBlockedError, "finite R40"):
            oracle._terminal_mae_gate_r(pd.Series([float("nan")]), pd.Series([-0.1]))
        np.testing.assert_allclose(
            oracle._terminal_mae_gate_r(r40, safety, horizon=60).to_numpy(), target.to_numpy(),
        )
        with self.assertRaisesRegex(oracle.OracleBlockedError, "nonnegative MAE60"):
            oracle._terminal_mae_gate_r(pd.Series([1.0]), pd.Series([0.2]), horizon=60)

    def test_registered_r40_t40_t40mfe_training_targets_match_oracle_economics(self):
        # One canonical 40-bar price path drives both supervised DL truth and the
        # hindsight Oracle T40/T40-MFE formulas; no alternative target engine.
        from core.breakout_quality_registry import (
            SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES,
            get_breakout_quality_experiment_profile,
            get_continuous_ranker_research_spec,
        )
        from filters.breakout_quality.continuous_target import StrategyAlignedContinuousTargetSpec
        from filters.breakout_quality.daily_ranker_data import compute_daily_opportunity_target_batch
        spec = StrategyAlignedContinuousTargetSpec.from_label_policy(oracle.DEFAULT_LABEL_POLICY)
        self.assertEqual(spec.horizon_bars, 40)
        # Test registered historical Target contracts; do not force retired CI into
        # the mutable current Model Compare/Test list merely to keep regression coverage.
        by_model = {
            research.model_research_id: profile_name
            for profile_name in SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES
            for research in (get_continuous_ranker_research_spec(profile_name),)
        }
        self.assertTrue({"MR-13CI", "MR-13CJ", "MR-13CK"}.issubset(by_model))
        by_target = {
            "MR-13CI": "daily_terminal_r40_v1",
            "MR-13CJ": "daily_terminal_mae_gated_t40_v1",
            "MR-13CK": "daily_full_horizon_mae_gated_mfe_t40_v1",
        }
        policies = []
        for mr, expected_id in by_target.items():
            profile = get_breakout_quality_experiment_profile(by_model[mr])
            research = get_continuous_ranker_research_spec(by_model[mr])
            self.assertEqual(profile.continuous_target_id, expected_id)
            self.assertEqual(profile.model_architecture, "inception_time_v1")
            self.assertEqual(research.model_research_id, mr)
            policies.append((profile.training_objective, profile.loss_name,
                             research.pairwise_reduction, research.trainer_family))
        self.assertEqual(policies[0], policies[1], "R40/T40 use identical learner and Loss")
        self.assertEqual(policies[1], policies[2], "T40 and T40-MFE change only the target")

        def price_frame(close_40, lowest_40, future_peak=None):
            close = np.full(41, 100.0)
            close[40] = close_40
            high = np.maximum(close, 101.0)
            if future_peak is not None:
                high[10] = future_peak
            low = np.minimum(close, 99.0)
            low[10] = lowest_40
            low[40] = min(low[40], close_40)
            return pd.DataFrame({"Close": close, "High": high, "Low": low})

        for terminal_close, min_low, peak, expected_r, expected_t, expected_ck in [
            (115.0, 95.0, None, 1.5, 0.25, 0.25),
            (115.0, 89.0, None, 1.5, -1.0, -1.0),
            (96.0, 95.0, None, -0.4, -0.7, -0.45),
            (105.0, 95.0, 130.0, 0.5, -0.25, 1.0),
        ]:
            frame = price_frame(terminal_close, min_low, peak)
            scored = {}
            for mr, target_id in by_target.items():
                batch = compute_daily_opportunity_target_batch(
                    frame, np.array([0], dtype=np.int64), spec=spec, target_id=target_id,
                )
                self.assertTrue(batch.valid_mask[0], mr)
                scored[mr] = float(batch.target_raw_r[0])
            self.assertAlmostEqual(scored["MR-13CI"], expected_r, places=5)
            self.assertAlmostEqual(scored["MR-13CJ"], expected_t, places=5)
            self.assertAlmostEqual(scored["MR-13CK"], expected_ck, places=5)
            mae_r = max(0, 1.0 - min_low / 100.0) / spec.risk_budget_return
            oracle_t = float(oracle._terminal_mae_gate_r(
                pd.Series([expected_r]), pd.Series([-mae_r]),
            ).iloc[0])
            self.assertAlmostEqual(scored["MR-13CJ"], oracle_t, places=5)
            mfe_r = (float(frame.High.iloc[1:41].max()) / 100.0 - 1.0) / spec.risk_budget_return
            oracle_ck = float(oracle._terminal_mae_gate_r(
                pd.Series([mfe_r]), pd.Series([-mae_r]),
            ).iloc[0])
            self.assertAlmostEqual(scored["MR-13CK"], oracle_ck, places=5)

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
            truth, evidence = oracle._market_truth_for_oracle(
                self.definition, ROOT, cutoff="2021-06-01",
            )
        self.assertEqual(evidence["price_ticker_count"], 2)
        self.assertEqual(evidence["score_start"], "2020-12-31")
        for provider in oracle._oracle_comparison_plan(self.definition)["providers"]:
            table = truth[provider]
            self.assertEqual(set(table["ticker"]), {"0050", "2211"})
            self.assertFalse(table.duplicated(["ticker", "date"]).any())
            self.assertIn(("2211", "2021-02-26"), set(zip(table.ticker, table.date)))
            self.assertIn(("0050", "2021-01-13"), set(zip(table.ticker, table.date)))
            self.assertTrue(np.isfinite(table[["mfe", "peak_adverse", "min_low", "raw"]].to_numpy()).all())
            if provider == "pure_mfe40_provider":
                self.assertTrue(np.isfinite(table["terminal_r"]).all())
                # Close(T+40) / Close(T) - 1 in canonical risk R.
                first = table.loc[(table.ticker == "0050") & (table.date == "2021-01-13")].iloc[0]
                index = prices["0050"].index.get_loc(pd.Timestamp("2021-01-13"))
                budget = oracle.StrategyAlignedContinuousTargetSpec.from_label_policy(oracle.DEFAULT_LABEL_POLICY).risk_budget_return
                expected = (prices["0050"].iloc[index + 40]["Close"] / prices["0050"].iloc[index]["Close"] - 1) / budget
                self.assertAlmostEqual(float(first["terminal_r"]), float(expected), places=6)
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
        market = ({field: (price_truth.assign(terminal_r=[0.8, -0.3]) if field == "pure_mfe40_provider" else price_truth)
                   for field in oracle._oracle_comparison_plan(self.definition)["providers"]},
                  {"score_start": "2021-01-01", "truth_source": "synthetic"})
        with patch.object(oracle, "_market_truth_for_oracle", return_value=market), \
             patch.object(oracle, "_load_truth", return_value=provider):
            scores, evidence = oracle.build_oracle_score_tables(
                self.definition, project_root=ROOT, cutoff="2026-03-02",
            )
        self.assertEqual(evidence["cohort_rows"], 2)
        self.assertEqual(evidence["rows_outside_dl_training_provider_by_source"],
                         {field: 1 for field in oracle._oracle_comparison_plan(self.definition)["providers"]})
        for target, _ in LEGACY_ORACLE_TARGETS:
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

    def test_ssot_golden_is_strict_across_windows_line_endings(self):
        from tools.validate.ssot_regression_guard import (
            APPROVED_GOLDEN_SHA256, read_approved_baseline,
        )
        import hashlib

        original = (ROOT / "tools/validate/baselines/ssot_regression.json").read_bytes()
        lf = original.replace(b"\r\n", b"\n")
        crlf = lf.replace(b"\n", b"\r\n")
        self.assertEqual(hashlib.sha256(lf).hexdigest(), APPROVED_GOLDEN_SHA256)
        self.assertEqual(
            read_approved_baseline(ROOT, raw_bytes=lf),
            read_approved_baseline(ROOT, raw_bytes=crlf),
        )
        for changed in (lf + b" ", crlf + b" "):
            with self.assertRaisesRegex(ValueError, "Golden hash mismatch"):
                read_approved_baseline(ROOT, raw_bytes=changed)
        # A real JSON value mutation must still fail, regardless of line ending.
        changed = lf.replace(b'"schema_version": 1', b'"schema_version": 2', 1)
        self.assertNotEqual(changed, lf)
        with self.assertRaisesRegex(ValueError, "Golden hash mismatch"):
            read_approved_baseline(ROOT, raw_bytes=changed)

    def test_retired_history_never_required_for_current_report_or_oracle_reuse(self):
        from core.report_contract_versions import (
            load_report_contract_versions, validate_active_report_contract_versions,
        )
        from core.research_report_contract import (
            PERSISTENT_REPORT_CONTRACTS, persistent_report_contract_fingerprints,
        )
        from tools.validate.ssot_regression_guard import (
            read_approved_baseline, report_contract_version_history_violations,
        )

        with tempfile.TemporaryDirectory(prefix="standalone_report_contract_") as tmp:
            project = Path(tmp)
            for name in (
                "doc/scientific_contracts/REPORT_CONTRACT_VERSIONS.json",
                "tools/validate/baselines/ssot_regression.json",
            ):
                target = project / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, target)
            self.assertFalse((project / "doc/scientific_contracts/CONTRACT_EVIDENCE_HISTORY.zip").exists())
            registry = load_report_contract_versions(project)
            current = persistent_report_contract_fingerprints()
            self.assertEqual(validate_active_report_contract_versions({
                report_id: (contract.version, current[report_id])
                for report_id, contract in PERSISTENT_REPORT_CONTRACTS.items()
            }, project_root=project), ())
            approved = read_approved_baseline(project)
            self.assertEqual(report_contract_version_history_violations(project, approved), [])
            # All past report IDs/fingerprints needed for completed Oracle
            # payload compatibility remain available in one small ledger.
            oracle_history = registry["audit.target_oracle_romd"]
            self.assertEqual(tuple(row["version"] for row in oracle_history), tuple(range(1, 10)))
            self.assertEqual(oracle_history[-1]["fingerprint"], current["audit.target_oracle_romd"])
            self.assertTrue(all(set(row) == {"contract_id", "version", "fingerprint"}
                                for rows in registry.values() for row in rows))
            # Mutation rejection is independent of the retired file inventory.
            ledger_path = project / "doc/scientific_contracts/REPORT_CONTRACT_VERSIONS.json"
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            row = next(x for x in ledger["reports"] if x["report_id"] == "audit.target_oracle_romd")
            row["versions"][0]["fingerprint"] = "0" * 16
            ledger_path.write_text(json.dumps(ledger, ensure_ascii=False), encoding="utf-8")
            self.assertTrue(report_contract_version_history_violations(project, approved))
            fixture = project / "tools/validate/baselines/ssot_regression.json"
            fixture.write_bytes(fixture.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "Golden hash mismatch"):
                read_approved_baseline(project)

    def test_future_report_versions_need_only_compact_ledger_row(self):
        from core.report_contract_versions import load_report_contract_versions
        from tools.validate.ssot_regression_guard import (
            read_approved_baseline, report_contract_version_history_violations,
        )
        approved = read_approved_baseline(ROOT)
        self.assertEqual(approved["schema_version"], 1)
        with tempfile.TemporaryDirectory(prefix="report_ledger_append_") as tmp:
            project = Path(tmp)
            folder = project / "doc/scientific_contracts"
            folder.mkdir(parents=True)
            ledger_path = folder / "REPORT_CONTRACT_VERSIONS.json"
            shutil.copy2(ROOT / "doc/scientific_contracts/REPORT_CONTRACT_VERSIONS.json", ledger_path)
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            entry = next(x for x in ledger["reports"] if x["report_id"] == "audit.target_oracle_romd")
            self.assertEqual(entry["versions"][-1]["version"], 9)
            entry["versions"].append({
                "contract_id": "audit.target_oracle_romd@v10",
                "version": 10,
                "fingerprint": "1" * 16,
            })
            ledger_path.write_text(json.dumps(ledger, ensure_ascii=False), encoding="utf-8")
            self.assertEqual(load_report_contract_versions(project)["audit.target_oracle_romd"][-1]["version"], 10)
            self.assertEqual(report_contract_version_history_violations(project, approved), [])
            entry["versions"][7]["fingerprint"] = "0" * 16
            ledger_path.write_text(json.dumps(ledger, ensure_ascii=False), encoding="utf-8")
            self.assertTrue(report_contract_version_history_violations(project, approved))

    def test_reusable_research_report_publishes_schema7_exact_and_direct(self):
        def summary(name: str):
            return {
                "target": name, "total_return_pct": 100.0, "annualized_return_pct": 14.0,
                "mdd_pct": 12.0, "romd": 8.0, "romd_std": 0.0,
                "ev_a_r": 0.5, "seeds": 1, "yearly": [],
                "seed_observations": [{"seed": 123, "romd": 8.0, "delta_romd_vs_c96": 0.0}],
                "delta_romd_vs_c96": 0.0, "positive_seed_count": 0,
            }
        result = {
            "schema_version": 8, "status": "COMPLETED",
            "start_date": "2021-01-01", "end_date": "2025-12-22",
            "common_cohort": {"cohort_rows": 3},
            "benchmark_id": "test", "seeds": [123], "limitations": ["NOT PIT"],
            "baseline": summary("C96"),
            "oracles": [summary(name) for name, _ in LEGACY_ORACLE_TARGETS],
        }
        from services.audit.runner import render_latest_audit_summary
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            with patch.object(oracle, "build_report", return_value=result):
                published = oracle.run_formal_audit(self.definition, project_root=base, quiet=True)
            self.assertEqual(published["schema_version"], 8)
            self.assertEqual(published["report_contract_id"], "audit.target_oracle_romd@v9")
            latest = base / "outputs/audit" / self.definition.output_subdir / "latest"
            body = (latest / "audit.md").read_text(encoding="utf-8")
            for name in oracle.ORACLE_DIRECT_SCORE_SOURCES:
                if name in dict(LEGACY_ORACLE_TARGETS):
                    self.assertIn("**" + name + "**", body)
            self.assertIn("**R40**", body)
            self.assertIn("**T40**", body)
            self.assertIn("**T40-MFE**", body)
            self.assertIn("**T40-MFE_Direct**", body)
            self.assertNotIn("60D Mature Oracle", body)
            self.assertNotIn("**CH20_EqualRank**", body)
            self.assertIn("MAE40 >= 1R", body)
            self.assertIn("Delta RoMD", body)
            # Display titles describe stable schema, not today's selected inventory.
            self.assertIn("## 2. Target Definitions\n", body)
            self.assertIn(f"Oracle treatments: {len(LEGACY_ORACLE_TARGETS)} ", body)
            self.assertNotIn("5 × Exact / 4 × Direct", body)
            # A new comparison row changes rendered data but not report identity.
            expanded = {**result, "oracles": [*result["oracles"], summary("FUTURE_TEST_ONLY")]}
            expanded_body = oracle.render_result(expanded, target="markdown")
            self.assertIn(f"Oracle treatments: {len(LEGACY_ORACLE_TARGETS) + 1} ", expanded_body)
            self.assertIn("## 2. Target Definitions\n", expanded_body)
            self.assertEqual(published["report_contract_id"], "audit.target_oracle_romd@v9")
            json_data = json.loads((latest / "audit.json").read_text(encoding="utf-8"))
            self.assertEqual(len(json_data["oracles"]), len(LEGACY_ORACLE_TARGETS))
            self.assertIn("**MFE40**", body)
            self.assertNotIn("**MFE40_Control**", body)
            for old_name in ("AO40_EqualRank", "U40", "FirstBreach_NetR"):
                self.assertNotIn("**" + old_name + "**", body)
            self.assertIn("MFE40_Control", json.dumps(json_data))
            self.assertNotIn("additional_panels", json_data)
            listing = render_latest_audit_summary(
                "breakout_quality", project_root=base, method_id=oracle.SUPPORTED_AUDIT_TYPE,
            )
            self.assertIn("存在", listing)
            self.assertIn(self.definition.report_id, listing)

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
                context = get_breakout_quality_ranking_source_context()
                self.assertEqual(context.score_source, SCORE_SOURCE_AUDIT_ORACLE_FUTURE_TRUTH)
                expected_policy = (
                    "resource-aware-continuous-score-no-k-no-r0"
                    if "_Direct" in name else self.exact_policy
                )
                self.assertEqual(context.ranking_policy, expected_policy)
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
             patch.object(oracle, "_build_oracle_score_panels", return_value=_mock_panels({"cohort_rows": 4, "last_common_day": "2025-12-31", "mfe40_over_d40_zero_downside_positive_upside_count": 0, "mfe40_over_d40_zero_zero_count": 0})), \
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
                # A scientifically different replay identity is NOT permission
                # to silently run it again when matching-panel cache exists.
                with patch.object(oracle, "price_inventory_sha256", return_value="c" * 64):
                    with self.assertRaisesRegex(oracle.OracleBlockedError, "dataset_sha256.*cache="):
                        oracle.build_report(self.definition, project_root=Path(tmp), quiet=True)
                self.assertEqual(len(seen), original_calls)
        self.assertEqual(len(seen), len(seeds) * (1 + len(LEGACY_ORACLE_TARGETS)))
        self.assertEqual(report["seeds"], list(seeds))
        self.assertEqual(report["schema_version"], 8)
        self.assertEqual(len(report["oracles"]), len(LEGACY_ORACLE_TARGETS))
        self.assertEqual(report["end_date"], "2025-12-31")
        self.assertNotIn("additional_panels", report)
        self.assertEqual(report["baseline"]["romd_std"], report["oracles"][0]["romd_std"])
        self.assertEqual(report["oracles"][0]["delta_romd_vs_c96"], 1.5)
        self.assertEqual(report["oracles"][0]["positive_seed_count"], len(seeds))
        self.assertIn("Oracle Performance", oracle.render_result(report, target="markdown"))
        self.assertIn("**R40**", oracle.render_result(report, target="markdown"))
        self.assertIn("**T40**", oracle.render_result(report, target="markdown"))
        self.assertIn("**T40_Direct**", oracle.render_result(report, target="markdown"))
        self.assertNotIn("MFE40/D40 zero denominator", oracle.render_result(report, target="console"))
        self.assertIn("RoMD Mean", oracle.render_result(report, target="console"))
        json.dumps(report, ensure_ascii=False)
        self.assertNotEqual(get_breakout_quality_ranking_source_context().score_source,
                            SCORE_SOURCE_AUDIT_ORACLE_FUTURE_TRUTH)



    def test_oracle_and_daily_pit_reuse_identical_portfolio_rank_input_contract(self):
        """Verify the same candidate/selector contract in every configured time mode.

        Source replacement alone must not change the signal-date policy, daily
        refresh, score ordering or constrained-selector numeric objective. This
        is NOT a claim that archived Oracle percentiles equal raw DL scores.
        """
        from core.portfolio_candidates import _resolve_candidate_quality_ranking
        from core.portfolio_entry_selection_max_dl import (
            _max_dl_score_order, _max_dl_basket_quality_key,
        )
        from core.strategy_compare_policy import (
            get_strategy_comparison_settings, get_strategy_multi_seed_robustness_settings,
        )
        from core.strategy_compare_registry import STRATEGY_COMPARE_ROLLING_TEST_MODES
        from core.extended_signals import attach_breakout_quality_rank
        from filters.breakout_quality.ranking_score_store import SCORE_SOURCE_SELECTION_POINT_IN_TIME

        oracle_exact = oracle._contract(self.active_definition)[2]
        oracle_benchmark = get_strategy_multi_seed_robustness_settings(
            str(self.active_definition.source["strategy_profile_id"])
        )
        modes_checked = set()
        checked_arms = set()
        for mode in STRATEGY_COMPARE_ROLLING_TEST_MODES:
            profile_id = str(mode["profile_id"])
            settings = get_strategy_comparison_settings(profile_id)
            benchmark = get_strategy_multi_seed_robustness_settings(profile_id)
            self.assertEqual(benchmark.benchmark_id, oracle_benchmark.benchmark_id)
            self.assertEqual(tuple(benchmark.resolved_seeds), tuple(oracle_benchmark.resolved_seeds))
            base = settings.arms[str(self.active_definition.source["baseline_arm_id"])]
            matched = [
                arm for arm in settings.enabled_arms
                if (arm.dl_enabled and arm.dl_id
                    and arm.param_source == base.param_source
                    and arm.param_policy == base.param_policy
                    and arm.rule_policy == "all_off"
                    and arm.dl_runtime_mode == oracle_exact.dl_runtime_mode
                    and dict(arm.dl_runtime_options or {}) == dict(oracle_exact.dl_runtime_options or {}))
            ]
            self.assertTrue(matched, f"{profile_id}: no current Daily-universal Exact arm")
            modes_checked.add(str(mode["mode_id"]))
            for arm in matched:
                with self.subTest(mode=profile_id, arm_id=arm.arm_id):
                    checked_arms.add((profile_id, arm.arm_id))
                    source = settings.dl_sources[str(arm.dl_id)]
                    self.assertEqual(source.score_source, SCORE_SOURCE_SELECTION_POINT_IN_TIME)
                    self.assertEqual(oracle_exact.dl_runtime_options, arm.dl_runtime_options)
                    params = SimpleNamespace(
                        use_breakout_quality_ranking=True,
                        breakout_quality_filter_id=str(source.filter_id), high_len=260,
                    )
                    score_map = {
                        (ticker, day): score
                        for ticker, first, second in (
                            ("2330", 0.21, 0.91), ("2317", 0.85, 0.16),
                        )
                        for day, score in (
                            ("2025-02-03", first), ("2025-02-04", second),
                        )
                    }

                    def fake_pit_lookup(*, ticker, signal_date, filter_id, **_kwargs):
                        day = pd.Timestamp(signal_date).strftime("%Y-%m-%d")
                        return {
                            "score": score_map[(ticker, day)], "score_date": day,
                            "available": True, "unavailable_reason": "",
                            "filter_id": filter_id, "score_source": SCORE_SOURCE_SELECTION_POINT_IN_TIME,
                        }

                    def run_source(use_oracle):
                        context = (
                            audit_oracle_truth_score_context(
                                score_map, ranking_policy=oracle_exact.dl_runtime_mode,
                                ranking_options=oracle_exact.dl_runtime_options,
                                key_mode="information_date",
                            ) if use_oracle else breakout_quality_ranking_source_context(
                                score_source=SCORE_SOURCE_SELECTION_POINT_IN_TIME,
                                model_architecture=str(source.model_architecture),
                                experiment_profile=str(source.experiment_profile),
                                ranking_policy=arm.dl_runtime_mode,
                                ranking_options=arm.dl_runtime_options,
                            )
                        )
                        with context, patch(
                            "filters.breakout_quality.runtime.lookup_selection_point_in_time_candidate_score",
                            side_effect=fake_pit_lookup,
                        ) as pit_lookup:
                            signals = {}
                            history = []
                            for info_date in ("2025-02-03", "2025-02-04"):
                                candidates = []
                                for ticker in ("2330", "2317"):
                                    signal_state = signals.setdefault(ticker, {})
                                    rank = _resolve_candidate_quality_ranking(
                                        params=params, ticker=ticker,
                                        signal_date="2025-02-03", information_date=info_date,
                                        signal_state=signal_state,
                                    )
                                    attach_breakout_quality_rank(signal_state, rank)
                                    self.assertEqual(rank["score_date"], info_date)
                                    candidates.append({
                                        "ticker": ticker, "breakout_quality_score": rank["score"],
                                        "breakout_quality_rank": rank,
                                    })
                                base_rank = {id(row): i for i, row in enumerate(candidates)}
                                order = _max_dl_score_order(candidates, base_rank=base_rank)
                                objective = _max_dl_basket_quality_key(order, base_rank=base_rank)
                                ordered_scores = tuple(
                                    (r["ticker"], r["breakout_quality_score"]) for r in order
                                )
                                history.append((ordered_scores, objective))
                            if use_oracle:
                                pit_lookup.assert_not_called()
                            else:
                                self.assertEqual(pit_lookup.call_count, 4)
                            return history

                    oracle_history, dl_history = run_source(True), run_source(False)
                    self.assertEqual(oracle_history, dl_history)
                    self.assertNotEqual(
                        oracle_history[0][0], oracle_history[1][0],
                        "Daily-universal continued signals must refresh rankings",
                    )
        self.assertGreaterEqual(len(modes_checked), 2)
        self.assertGreaterEqual(len(checked_arms), len(modes_checked))

    def test_exact_score_sum_is_not_merely_a_rank_order(self):
        """Retain the historical numeric objective; never silently change its scale.

        Two score sets can have exactly the same stock order but different
        optimal fixed-K baskets. A new common-percentile research comparison
        would therefore require its own scientific identity and evidence.
        """
        from core.portfolio_entry_selection_max_dl import _max_dl_basket_quality_key

        def quality(scores, selected):
            rows = [
                {"ticker": str(i), "breakout_quality_score": float(score),
                 "breakout_quality_rank": {"available": True, "score": float(score)}}
                for i, score in enumerate(scores)
            ]
            rank = {id(row): i for i, row in enumerate(rows)}
            return _max_dl_basket_quality_key([rows[i] for i in selected], base_rank=rank)

        tight = (0.99, 0.70, 0.69, 0.10)
        spread = (0.99, 0.20, 0.19, 0.10)
        self.assertTrue(quality(tight, (1, 2)) > quality(tight, (0, 3)))
        self.assertTrue(quality(spread, (0, 3)) > quality(spread, (1, 2)))

    def test_oracle_and_pit_identical_scores_produce_identical_broker_cash_replay(self):
        """One actual Broker-Cash engine for both score sources, Exact AND Direct.

        This is behavioral evidence, not a mock _run_scenario. Reversing two
        stock rankings must change the actual filled ticker; with identical
        scores each policy must agree on all canonical trade/equity outputs.
        """
        from core.strategy_params import V16StrategyParams
        from core.strategy_compare_policy import get_strategy_comparison_settings
        from core.strategy_compare_registry import STRATEGY_COMPARE_ROLLING_TEST_MODES
        from filters.breakout_quality.ranking_score_store import SCORE_SOURCE_SELECTION_POINT_IN_TIME
        from tools.validate.synthetic_portfolio_common import (
            build_synthetic_competing_candidates_case, run_portfolio_core_check_for_dir,
        )
        from tools.validate.synthetic_frame_utils import write_synthetic_csv_bundle

        case = build_synthetic_competing_candidates_case(V16StrategyParams())
        params = case["params"]
        params.use_breakout_quality_ranking = True
        params.breakout_quality_filter_id = str(self.active_definition.source["filter_id"])
        settings, baseline, exact = oracle._contract(self.active_definition)
        direct = oracle._direct_arm(self.active_definition, settings, baseline, exact)
        policies = {"exact": exact, "direct": direct}
        with tempfile.TemporaryDirectory() as tmp:
            write_synthetic_csv_bundle(tmp, case["frames"])
            for mode in STRATEGY_COMPARE_ROLLING_TEST_MODES:
                settings = get_strategy_comparison_settings(str(mode["profile_id"]))
                base = settings.arms[str(self.active_definition.source["baseline_arm_id"])]
                arms = [
                    arm for arm in settings.enabled_arms
                    if (arm.dl_enabled and arm.dl_id
                        and arm.param_source == base.param_source
                        and arm.param_policy == base.param_policy
                        and arm.rule_policy == "all_off"
                        and arm.dl_runtime_mode == exact.dl_runtime_mode
                        and dict(arm.dl_runtime_options or {}) == dict(exact.dl_runtime_options or {}))
                ]
                self.assertTrue(arms)
                # Pick one real registered PIT score provider per evaluation
                # mode. The separate Direct policy comes from canonical config,
                # not an Audit-specific selector or an invented new DL arm.
                source = settings.dl_sources[str(arms[0].dl_id)]
                for policy_name, policy in policies.items():
                    selected_tickers = set()
                    for winner in case["frames"]:
                        score_map = {
                            (ticker, day): (0.9 if ticker == winner else 0.1)
                            for ticker, frame in case["frames"].items()
                            for day in frame["Date"].to_list()
                        }

                        def fake_pit_lookup(*, ticker, signal_date, filter_id, **_kwargs):
                            date = pd.Timestamp(signal_date).strftime("%Y-%m-%d")
                            return {
                                "score": score_map[(ticker, date)], "score_date": date,
                                "available": True, "unavailable_reason": "",
                                "filter_id": filter_id,
                                "score_source": SCORE_SOURCE_SELECTION_POINT_IN_TIME,
                            }

                        def execute(use_oracle):
                            context = (
                                audit_oracle_truth_score_context(
                                    score_map, ranking_policy=policy.dl_runtime_mode,
                                    ranking_options=policy.dl_runtime_options,
                                    key_mode="information_date",
                                ) if use_oracle else breakout_quality_ranking_source_context(
                                    score_source=SCORE_SOURCE_SELECTION_POINT_IN_TIME,
                                    model_architecture=str(source.model_architecture),
                                    experiment_profile=str(source.experiment_profile),
                                    ranking_policy=policy.dl_runtime_mode,
                                    ranking_options=policy.dl_runtime_options,
                                )
                            )
                            with context, patch(
                                "filters.breakout_quality.runtime.lookup_selection_point_in_time_candidate_score",
                                side_effect=fake_pit_lookup,
                            ):
                                return run_portfolio_core_check_for_dir(
                                    tmp, params, max_positions=case["max_positions"],
                                    enable_rotation=case["enable_rotation"],
                                    start_year=case["start_year"],
                                    benchmark_ticker=case["benchmark_ticker"],
                                )

                        oracle_result, dl_result = execute(True), execute(False)
                        with self.subTest(mode=mode["mode_id"], policy=policy_name, winner=winner):
                            for field in (
                                "total_return", "mdd", "final_eq", "trade_count",
                                "win_rate", "pf_ev", "pf_payoff", "filled_buy_count",
                                "avg_exp", "max_exp", "reserved_buy_fill_rate",
                            ):
                                self.assertEqual(oracle_result[field], dl_result[field], field)
                            pd.testing.assert_frame_equal(
                                oracle_result["df_trades"], dl_result["df_trades"],
                            )
                            self.assertEqual(
                                oracle_result["portfolio_completed_trades"],
                                dl_result["portfolio_completed_trades"],
                            )
                            bought = oracle_result["df_trades"].loc[
                                oracle_result["df_trades"]["Type"].str.startswith("買進"), "Ticker"
                            ].tolist()
                            self.assertEqual(bought, [winner])
                            selected_tickers.add(winner)
                    self.assertEqual(selected_tickers, set(case["frames"]))


class ActiveMinTrade40OracleComparisonCases(TestCase):
    """Current Target is a regular Broker-Cash comparison, not a second report."""

    def test_current_seed_path_ranks_origin_on_same_seed_p16_and_reuses_arm(self):
        definition = next(d for d in get_reusable_audit_definitions("breakout_quality")
                          if d.audit_type == oracle.SUPPORTED_AUDIT_TYPE)
        definition = replace(definition, source={**definition.source,
            "oracle_comparisons": tuple(row for row in definition.source["oracle_comparisons"]
                if row["target"] not in oracle._oracle_comparison_plan(definition)["daily_sources"])})
        plan = oracle._oracle_comparison_plan(definition)
        self.assertIn("MinTrade40_NetR", dict(plan["targets"]))
        self.assertIn("MinTrade40_NetR", plan["seed_recipes"])
        self.assertEqual(plan["seed_sources"], {
            "MinTrade40_NetR": "MinTrade40_NetR",
            "MinTrade40_NetR_Direct": "MinTrade40_NetR",
        })
        self.assertEqual(plan["aliases"]["MinTrade40_NetR_Direct"], "MinTrade40_NetR")
        self.assertEqual(len(plan["targets"]), 11)
        exact_policy = oracle._contract(definition)[2].dl_runtime_mode
        seeds = (693545351, 2014432738, 104504529, 762252050)
        class Benchmark:
            benchmark_id = "end_to_end_v1"
        origin = "2025-01-02"
        future_info = "2025-01-03"
        score_by_seed = {
            seed: {("2330", origin): 0.2 + index * 0.1}
            for index, seed in enumerate(seeds)
        }
        evidence = {seed: {"signal_events": 10, "completed_trades": 9,
                           "excluded_events": 1, "score_key_mode": "signal_origin_date"}
                    for seed in seeds}
        actual_source = []
        def scenario(*, name, **kwargs):
            if "MinTrade40_NetR" in name:
                seed = int(name.split("seed=")[1])
                ranked = resolve_breakout_quality_candidate_rank(
                    ticker="2330", signal_date=origin, information_date=future_info,
                    high_len=260, filter_id="breakout_quality_v1",
                )
                self.assertAlmostEqual(ranked["score"], score_by_seed[seed][("2330", origin)])
                self.assertEqual(ranked["score_date"], origin)
                context = get_breakout_quality_ranking_source_context()
                self.assertEqual(context.ranking_policy, (
                    "resource-aware-continuous-score-no-k-no-r0"
                    if "_Direct" in name else exact_policy
                ))
                actual_source.append((seed, "_Direct" in name, ranked["score"]))
            elif name.startswith("Oracle"):
                ranked = resolve_breakout_quality_candidate_rank(
                    ticker="2330", signal_date=origin, information_date=origin,
                    high_len=260, filter_id="breakout_quality_v1",
                )
                self.assertEqual(ranked["score_date"], origin)
            return {"name": name}
        def metrics(replay, *, target):
            seed = int(replay["name"].split("seed=")[1].split()[0])
            return {"target": target, "total_return_pct": 100.0,
                    "annualized_return_pct": 15.0, "mdd_pct": 12.0,
                    "romd": 8.0 + (2.0 if target.startswith("MinTrade40_NetR") else 0.0),
                    "ev_a_r": 0.4, "trades": 50, "exposure_pct": 70.0,
                    "yearly": [{"year": 2025, "strategy_return_pct": 5.0, "is_full_year": True}]}
        common = {"cohort_rows": 100, "first_common_day": "2021-01-01",
                  "last_common_day": "2025-12-22", "research_cutoff": "2026-03-02",
                  "providers": {"40d": definition.source["pure_mfe40_provider"]},
                  "market_truth": {"score_start": "2020-12-31", "mature_rows_by_provider": {
                      "pure_mfe40_provider": 100}},
                  "rows_outside_dl_training_provider_by_source": {"pure_mfe40_provider": 0}}
        score_fixture = {name: {("2330", origin): 0.8}
                         for name in plan["recipes"]}
        for name, source in plan["aliases"].items():
            if source in score_fixture:
                score_fixture[name] = score_fixture[source]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(oracle, "collect_status", return_value={"status": "READY"}), \
                 patch.object(oracle, "_robustness_artifacts", return_value=(Benchmark(), [
                     (seed, Path(f"seed_{seed}.json")) for seed in seeds])), \
                 patch.object(oracle, "_build_oracle_score_panels", return_value=(score_fixture, common)), \
                 patch.object(oracle, "load_strategy_param_evaluation_view", return_value=(
                     {"kind": "rolling_active_param_ensemble", "payload": {}}, "a" * 64)), \
                 patch.object(oracle, "_validate_requested_param_policy"), \
                 patch.object(oracle, "get_active_param_ensemble_date_range", return_value=(
                     "2020-01-01", "2026-03-02")), \
                 patch.object(oracle, "_build_controlled_param_source_pair", return_value=(
                     "rolling_active_param_ensemble", "base", "oracle", None, None, None)), \
                 patch.object(oracle, "price_inventory_sha256", return_value="b" * 64), \
                 patch("services.audit.min_trade40_target.build_seed_oracle_scores",
                       return_value=(score_by_seed, evidence)) as build_labels, \
                 patch.object(oracle, "_run_scenario", side_effect=scenario) as replay, \
                 patch.object(oracle, "_metrics", side_effect=metrics):
                # First freeze the existing Exact + stock-day panel. Adding a
                # Direct selector later must only RUN four new seed replays.
                old_definition = replace(definition, source={
                    **definition.source, "oracle_comparisons": tuple(
                        item for item in definition.source["oracle_comparisons"]
                        if item["target"] != "MinTrade40_NetR_Direct"
                    ),
                })
                old_plan = oracle._oracle_comparison_plan(old_definition)
                self.assertEqual(
                    oracle._arm_replay_policy(old_definition, plan=old_plan,
                                              target="MinTrade40_NetR", replay_policy={}),
                    oracle._arm_replay_policy(definition, plan=plan,
                                              target="MinTrade40_NetR", replay_policy={}),
                )
                old_report = oracle.run_formal_audit(old_definition, project_root=root, quiet=True)
                self.assertEqual(replay.call_count, 44)
                self.assertEqual(len(old_report["oracles"]), 10)
                published = oracle.run_formal_audit(definition, project_root=root, quiet=True)
                self.assertEqual(len(actual_source), 8)
                for seed in seeds:
                    exact = next(v for s, direct, v in actual_source if s == seed and not direct)
                    other = next(v for s, direct, v in actual_source if s == seed and direct)
                    self.assertEqual(exact, other, "Exact/Direct must reuse the SAME seed score")
                self.assertEqual(replay.call_count, 48, "only the four new Direct units may RUN")
                self.assertEqual(build_labels.call_count, 2,
                                 "both visits consume the same one-label producer")
                self.assertEqual(len(published["oracles"]), 11)
                self.assertEqual([row["target"] for row in published["oracles"]][-2:],
                                 ["MinTrade40_NetR", "MinTrade40_NetR_Direct"])
                for row in published["oracles"][-2:]:
                    self.assertEqual(row["positive_seed_count"], 4)
                    self.assertEqual(row["delta_romd_vs_c96"], 2.0)
                self.assertEqual(len(published["seed_target_evidence"]), 4)
                self.assertIn("MinTrade40_NetR", (root / "outputs/audit" / definition.output_subdir / "latest" / "audit.md").read_text(encoding="utf-8"))
                self.assertFalse((root / "outputs/audit/breakout_quality/reusable/min_trade40_netr/latest").exists())
                second = oracle.run_formal_audit(definition, project_root=root, quiet=True)
                self.assertEqual(second["oracles"], published["oracles"])
                self.assertEqual(replay.call_count, 48, "all finished arms must REUSE")
                self.assertEqual(build_labels.call_count, 2, "completed report skips re-materialization")
            self.assertNotEqual(get_breakout_quality_ranking_source_context().score_source,
                                SCORE_SOURCE_AUDIT_ORACLE_FUTURE_TRUTH)


class OracleReplayEvidenceReuseCases(TestCase):
    """Exact-identity per-arm REUSE and data-driven local report discovery."""

    @staticmethod
    def _old_definitions():
        active = dict(LEGACY_ORACLE_TARGETS)
        # Frozen *test fixture* from the historical eleven-treatment evidence.
        # Production reuse must not have any historical target-name allowlist.
        retired = {
            "AO40_EqualRank": "40D MFE + adverse-to-peak Safety, 50/50 daily rank",
            "U40": "40D MFE - D40, canonical R",
            "FirstBreach_NetR": "MFE before first risk breach - adverse to that peak, R",
            "CH20_EqualRank": "20D MFE + full D20 Safety, 50/50 daily rank",
            "U20": "20D MFE - D20, canonical R",
            "MFE40_over_D40": "Exact MFE40 / D40 in canonical R; if D40=0 and MFE>0 rank as +infinity, 0/0 ranks as zero",
            "MFE20": "20D maximum favorable excursion, in canonical R",
            "MAE40": "Negative full-horizon 40D MAE (lower adverse excursion ranks higher), in canonical R",
            "MAE20": "Negative full-horizon 20D MAE (lower adverse excursion ranks higher), in canonical R",
        }
        names = ("AO40_EqualRank", "CG40_EqualRank", "CH20_EqualRank", "U40", "U20",
                 "FirstBreach_NetR", "MFE40_Control", "MFE40_over_D40", "MFE20", "MAE40", "MAE20")
        source = {**retired, **active}
        return {name: source[name] for name in names}

    def test_discovers_local_completed_reports_without_known_target_inventory_or_folder(self):
        from services.audit.oracle_replay_artifacts import HistoricalOracleIndex
        from core.report_contract_versions import load_report_contract_versions
        seeds = [1, 2, 3, 4]
        cohort = {"cohort_rows": 16, "first_common_day": "2021-01-01",
                  "last_common_day": "2025-12-22", "research_cutoff": "2026-03-02",
                  "risk_budget_return": 0.1, "providers": {}, "scoring": "percentile",
                  "market_truth": {}, "rows_outside_dl_training_provider_by_source": {}}
        definitions = dict(LEGACY_ORACLE_TARGETS)
        unknown_retired = "RETIRED_UNKNOWN_TO_ENGINE"
        local_definitions = {"U40": self._old_definitions()["U40"], unknown_retired: "historical test only"}
        payload = {
            "schema_version": 3, "status": "COMPLETED", "read_only": True,
            "hindsight_non_pit": True,
            "report_id": "target_oracle_romd", "report_type": oracle.SUPPORTED_AUDIT_TYPE,
            "report_contract_id": "audit.target_oracle_romd@v2",
            "report_contract_fingerprint": load_report_contract_versions()["audit.target_oracle_romd"][1]["fingerprint"],
            "strategy_profile_id": "broker_cash_dl_conversion_rolling",
            "benchmark_id": "end_to_end_v1", "dataset": "full",
            "start_date": "2021-01-01", "end_date": "2025-12-22",
            "seeds": seeds, "common_cohort": cohort,
            "param_artifacts": [{"seed": seed, "param_sha256": "a" * 64} for seed in seeds],
            "definitions": local_definitions,
            "baseline": oracle._aggregate([self._observation("C96", seed) for seed in seeds], target="C96"),
            "oracles": [oracle._aggregate([self._observation(name, seed) for seed in seeds], target=name)
                        for name in local_definitions],
        }
        context = {
            "benchmark": "end_to_end_v1", "strategy_profile_id": "broker_cash_dl_conversion_rolling",
            "dataset": "full", "start_date": "2021-01-01", "end_date": "2025-12-22",
            "seeds": seeds, "param_sha_by_seed": {seed: "a" * 64 for seed in seeds},
            "cohort": cohort, "definitions": {**definitions, "U40": self._old_definitions()["U40"]},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs/audit/breakout_quality"
            location = root / "archived_elsewhere/20261023"  # No fixed one_time/reusable folder.
            location.mkdir(parents=True)
            path = location / "audit.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            unrelated = root / "unrelated/latest"
            unrelated.mkdir(parents=True)
            (unrelated / "audit.json").write_text("broken unrelated audit", encoding="utf-8")
            index = HistoricalOracleIndex(Path(tmp))
            self.assertEqual(index.lookup(target="U40", seed=3, context=context),
                             self._observation("U40", 3))
            self.assertEqual(index.lookup(target="C96", seed=3, context=context),
                             self._observation("C96", 3))
            self.assertIsNone(index.lookup(target="T40", seed=3, context=context))
            # A target-specific definition drift blocks only that target.
            changed = dict(payload, definitions={**local_definitions, "U40": "different science"})
            path.write_text(json.dumps(changed), encoding="utf-8")
            index = HistoricalOracleIndex(Path(tmp))
            self.assertIsNone(index.lookup(target="U40", seed=3, context=context))
            self.assertIsNotNone(index.lookup(target="C96", seed=3, context=context))
            # An unapproved report version must never be reused.
            changed["report_contract_id"] = "audit.target_oracle_romd@v999"
            path.write_text(json.dumps(changed), encoding="utf-8")
            index = HistoricalOracleIndex(Path(tmp))
            self.assertIsNone(index.lookup(target="C96", seed=3, context=context))

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
    def _identity(seed=1, target="CG40_EqualRank", dataset_sha="a" * 64, param_sha="b" * 64):
        from services.audit.oracle_replay_artifacts import replay_identity
        return replay_identity(
            dataset="full", dataset_sha256=dataset_sha,
            target=target, definition={
                "C96": "Canonical same-window Min P16 Base-Finalist-Best",
                **dict(LEGACY_ORACLE_TARGETS),
            },
            start="2021-01-01", end="2025-12-22",
            benchmark="end_to_end_v1", param_sha256=param_sha, seed=seed,
            cohort={"cohort_rows": 123, "research_cutoff": "2026-03-02"},
            ranking_policy="score", ranking_options={"preserve_k_r0": True},
            evaluation_mode="rolling", max_positions=10,
            replay_policy={"exact_k_r0": True},
        )

    @staticmethod
    def _write_legacy_cache(store, identity, observation):
        from core.file_integrity import canonical_json_sha256
        path = store._path(identity)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cache_schema": 1, "identity": identity,
            "identity_sha256": canonical_json_sha256(identity),
            "observation": observation,
            "observation_sha256": canonical_json_sha256(observation),
            "origin": "historical_fixture",
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        return path

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
            record = self._observation("CG40_EqualRank", 1)
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

    def test_read_only_existing_identity_inventory_rejects_corrupted_record(self):
        from services.audit.oracle_replay_artifacts import OracleReplayStore
        with tempfile.TemporaryDirectory() as tmp:
            store=OracleReplayStore(Path(tmp), "breakout_quality/reusable/target_oracle_romd")
            identity=self._identity(target="C96")
            observation=self._observation("C96", 1)
            store.write(identity, observation, origin="canonical_replay")
            self.assertEqual(
                store.existing_completed_identities(seed=1,target="C96"), (identity,)
            )
            self.assertEqual(store.existing_completed_identities(seed=2,target="C96"), ())
            path=store._path(identity)
            payload=json.loads(path.read_text(encoding="utf-8"))
            payload["observation"]["romd"] += 1.0
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"inventory is corrupt"):
                store.existing_completed_identities(seed=1,target="C96")

    def test_conflict_reports_exact_fields_and_relative_cache_path_without_replay(self):
        from services.audit.oracle_replay_artifacts import OracleReplayStore
        with tempfile.TemporaryDirectory() as tmp:
            store = OracleReplayStore(Path(tmp), "breakout_quality/reusable/target_oracle_romd")
            cached = self._identity(target="C96", dataset_sha="a" * 64, param_sha="b" * 64)
            # Same dates and panel, but inputs and policy changed; never REUSE.
            requested = {
                **cached,
                "dataset_sha256": "c" * 64,
                "param_sha256": "d" * 64,
                "cohort": {**cached["cohort"], "cohort_rows": 124},
                "replay_policy": {**cached["replay_policy"], "exact_k_r0": False},
            }
            store.write(cached, self._observation("C96", 1), origin="canonical_replay")
            message = store.explain_other_identity(requested)
            self.assertIsNotNone(message)
            for field in ("dataset_sha256", "param_sha256", "cohort.cohort_rows",
                          "replay_policy.exact_k_r0"):
                self.assertIn(field, message)
            self.assertIn("outputs/audit/breakout_quality/reusable/target_oracle_romd/replay_cache/1/", message)
            self.assertIn("cache=aaaaaaaaaaaaaaaa...", message)
            self.assertIn("current=cccccccccccccccc...", message)
            self.assertIsNone(store.read(requested))
            self.assertEqual(store.read(cached), self._observation("C96", 1))
            self.assertEqual(store.explain_other_identity(requested), message)

    def test_legacy_comparison_digest_only_can_reuse_and_preserve_old_evidence(self):
        from services.audit.oracle_replay_artifacts import OracleReplayStore
        with tempfile.TemporaryDirectory() as tmp:
            store = OracleReplayStore(Path(tmp), "breakout_quality/reusable/target_oracle_romd")
            current = self._identity(target="C96")
            old = {**current, "cache_schema": 1, "strategy_settings_fingerprint": "7" * 64}
            record = self._observation("C96", 1)
            self._write_legacy_cache(store, old, record)
            legacy_path = store._path(old)
            original_bytes = legacy_path.read_bytes()
            with self.assertRaisesRegex(ValueError, "read-only"):
                store.write(old, record, origin="illegal_legacy_rewrite")
            self.assertIsNone(store.read(current))
            self.assertIsNone(store.explain_other_identity(current))
            reused, origin = store.resolve_completed(current)
            self.assertEqual(reused, record)
            self.assertIn("verified legacy replay cache", origin)
            store.write(current, reused, origin="verified_legacy_replay_cache")
            self.assertEqual(store.resolve_completed(current)[1], "verified per-arm cache")
            self.assertEqual(store.read(old), record)
            self.assertEqual(legacy_path.read_bytes(), original_bytes)

    def test_legacy_migration_rejects_any_replay_semantic_change(self):
        from services.audit.oracle_replay_artifacts import OracleReplayStore
        modifications = (
            {"dataset_sha256": "d" * 64},
            {"param_sha256": "e" * 64},
            {"target_definition": "different mathematical truth"},
            {"ranking_options": {"preserve_k_r0": False}},
            {"replay_policy": {"exact_k_r0": False}},
            {"cohort": {"cohort_rows": 124, "research_cutoff": "2026-03-02"}},
            {"max_positions": 9},
            {"end_date": "2025-12-23"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = OracleReplayStore(Path(tmp), "breakout_quality/reusable/target_oracle_romd")
            base = self._identity(target="C96")
            old = {**base, "cache_schema": 1, "strategy_settings_fingerprint": "8" * 64}
            self._write_legacy_cache(store, old, self._observation("C96", 1))
            for changed in modifications:
                candidate = dict(base, **changed)
                if candidate == base:
                    continue
                # AI: A different time window is a separate cache cohort, not
                # AI: a same-window conflict, but must never be reused.
                if candidate["end_date"] != base["end_date"]:
                    self.assertEqual(store.resolve_completed(candidate), (None, "none"))
                else:
                    with self.subTest(changed=changed), self.assertRaisesRegex(
                        ValueError, "DIFFERENT identity"
                    ):
                        store.resolve_completed(candidate)
            self.assertFalse(store.has_other_identity(base))

    def test_equivalent_legacy_caches_with_different_observations_fail_closed(self):
        from services.audit.oracle_replay_artifacts import OracleReplayStore
        with tempfile.TemporaryDirectory() as tmp:
            store = OracleReplayStore(Path(tmp), "breakout_quality/reusable/target_oracle_romd")
            identity = self._identity(target="C96")
            for digest, romd in (("a" * 64, 1.0), ("b" * 64, 99.0)):
                old = {**identity, "cache_schema": 1, "strategy_settings_fingerprint": digest}
                self._write_legacy_cache(store, old, {**self._observation("C96", 1), "romd": romd})
            with self.assertRaisesRegex(ValueError, "disagree on observations"):
                store.resolve_completed(identity)

    def test_incomplete_legacy_identity_is_not_admitted(self):
        from services.audit.oracle_replay_artifacts import OracleReplayStore
        with tempfile.TemporaryDirectory() as tmp:
            store = OracleReplayStore(Path(tmp), "breakout_quality/reusable/target_oracle_romd")
            current = self._identity(target="C96")
            old = {**current, "cache_schema": 1, "strategy_settings_fingerprint": "a" * 64}
            # AI: A missing broker/score policy is not an innocuous display change.
            old.pop("replay_policy")
            self._write_legacy_cache(store, old, self._observation("C96", 1))
            with self.assertRaisesRegex(ValueError, "DIFFERENT identity"):
                store.resolve_completed(current)

    def test_unrelated_maturity_panel_does_not_block_baseline_reuse(self):
        from services.audit.oracle_replay_artifacts import OracleReplayStore
        with tempfile.TemporaryDirectory() as tmp:
            store = OracleReplayStore(Path(tmp), "breakout_quality/reusable/target_oracle_romd")
            identity = self._identity(target="C96")
            different_panel = {
                **identity,
                "cohort": {**identity["cohort"], "providers": {"horizon": "60D"}},
            }
            store.write(different_panel, self._observation("C96", 1), origin="canonical_replay")
            self.assertIsNone(store.explain_other_identity(identity))
            self.assertFalse(store.has_other_identity(identity))

    def test_prefix_distinct_targets_are_not_cache_conflicts(self):
        from services.audit.oracle_replay_artifacts import OracleReplayStore
        with tempfile.TemporaryDirectory() as tmp:
            store = OracleReplayStore(Path(tmp), "breakout_quality/reusable/target_oracle_romd")
            direct = self._identity(target="R40_Direct")
            store.write(direct, self._observation("R40_Direct", 1), origin="canonical_replay")
            self.assertFalse(store.has_other_identity(self._identity(target="R40")))

    def test_history_reuses_only_completed_matching_seven_target_seed_records(self):
        from services.audit.oracle_replay_artifacts import HistoricalOracleIndex
        seeds = [1, 2, 3, 4]
        description = dict(LEGACY_ORACLE_TARGETS)
        historical_descriptions = self._old_definitions()
        first_seven = list(historical_descriptions)[:7]
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
            "cohort": cohort, "definitions": {**description, **{name: historical_descriptions[name]
                                        for name in ("AO40_EqualRank", "U40", "FirstBreach_NetR")}},
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
            "definitions": {key: historical_descriptions[key] for key in first_seven},
            "baseline": oracle._aggregate([self._observation("C96", seed) for seed in seeds], target="C96"),
            "oracles": [oracle._aggregate([self._observation(key, seed) for seed in seeds], target=key)
                        for key in first_seven],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs/audit/breakout_quality/one_time/target_oracle_romd/latest"
            root.mkdir(parents=True)
            path = root / "audit.json"
            path.write_text(json.dumps(historical), encoding="utf-8")
            index = HistoricalOracleIndex(Path(tmp))
            self.assertEqual(index.lookup(target="C96", seed=3, context=context)["romd"], 3.0)
            self.assertEqual(index.lookup(target="U40", seed=3, context=context)["romd"], 5.0)
            self.assertIsNone(index.lookup(target="R40", seed=3, context=context))
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

    def test_completed_v3_report_reuses_matching_exact_and_runs_only_missing_direct(self):
        from services.audit.oracle_replay_artifacts import HistoricalOracleIndex
        from core.report_contract_versions import load_report_contract_versions
        seeds = (11, 22, 33, 44)
        definitions = dict(LEGACY_ORACLE_TARGETS)
        old_definitions = dict(definitions)
        old_definitions.pop("T40-MFE")  # two new treatments did not exist in v3
        for name in oracle.ORACLE_DIRECT_SCORE_SOURCES:
            if name in old_definitions:
                old_definitions.pop(name)
        old_definitions.update({name: self._old_definitions()[name]
                                for name in ("AO40_EqualRank", "U40", "FirstBreach_NetR", "CH20_EqualRank")})
        common = {"cohort_rows": 4, "first_common_day": "2021-01-01",
                  "last_common_day": "2025-12-22", "research_cutoff": "2026-03-02",
                  "risk_budget_return": 0.03, "providers": {}, "scoring": "test",
                  "market_truth": {}, "rows_outside_dl_training_provider_by_source": {},
                  "ratio_zero_denominator_policy": "D40=0,MFE40>0 -> +infinity (highest rank); 0/0 -> 0; final score = finite daily percentile",
                  "mfe40_over_d40_zero_downside_positive_upside_count": 0,
                  "mfe40_over_d40_zero_zero_count": 0}
        class Benchmark:
            benchmark_id = "end_to_end_v1"
        old_report = {
            "schema_version": 4, "status": "COMPLETED", "read_only": True,
            "hindsight_non_pit": True, "report_id": "target_oracle_romd",
            "report_type": oracle.SUPPORTED_AUDIT_TYPE,
            "report_contract_id": "audit.target_oracle_romd@v3",
            "report_contract_fingerprint": load_report_contract_versions()["audit.target_oracle_romd"][2]["fingerprint"],
            "strategy_profile_id": "broker_cash_dl_conversion_rolling",
            "benchmark_id": Benchmark.benchmark_id, "dataset": "full",
            "start_date": "2021-01-01", "end_date": "2025-12-22",
            "seeds": list(seeds), "common_cohort": common,
            "param_artifacts": [{"seed": seed, "param_sha256": "a" * 64} for seed in seeds],
            "definitions": old_definitions,
            "baseline": oracle._aggregate([self._observation("C96", seed) for seed in seeds], target="C96"),
            "oracles": [oracle._aggregate([self._observation(name, seed) for seed in seeds], target=name)
                        for name in old_definitions],
            "additional_panels": {"60d": {
                "start_date": "2021-01-01", "end_date": "2025-11-24",
                "common_cohort": {"cohort_rows": 3, "providers": {"60d": "test"}},
                "definitions": {"MFE60": "old"},
                "param_artifacts": [{"seed": seed, "param_sha256": "a" * 64} for seed in seeds],
                "baseline": self._observation("C96", 11),
                "oracles": [self._observation("MFE60", 11)],
            }},
        }
        seen = []
        def replay(*, name, **kwargs):
            seen.append(name)
            return {"name": name}
        def metrics(result, *, target):
            seed = int(result["name"].split("seed=")[1].split()[0])
            return self._observation(target, seed)
        definition = _legacy_definition()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            previous = root / "outputs/audit/breakout_quality/reusable/target_oracle_romd/20261023T223000Z"
            previous.mkdir(parents=True)
            (previous / "audit.json").write_text(json.dumps(old_report), encoding="utf-8")
            with patch.object(oracle, "collect_status", return_value={"status": "READY"}), \
                 patch.object(oracle, "_robustness_artifacts", return_value=(Benchmark(), [(seed, Path(f"seed_{seed}.json")) for seed in seeds])), \
                 patch.object(oracle, "_build_oracle_score_panels", return_value=_mock_panels(common)), \
                 patch.object(oracle, "load_strategy_param_evaluation_view", return_value=({"kind": "rolling_active_param_ensemble", "payload": {}}, "a" * 64)), \
                 patch.object(oracle, "_validate_requested_param_policy"), \
                 patch.object(oracle, "get_active_param_ensemble_date_range", return_value=("2020-01-01", "2026-03-02")), \
                 patch.object(oracle, "_build_controlled_param_source_pair", return_value=("rolling_active_param_ensemble", "base", "oracle", None, None, None)), \
                 patch.object(oracle, "price_inventory_sha256", return_value="b" * 64), \
                 patch.object(oracle, "_run_scenario", side_effect=replay), \
                 patch.object(oracle, "_metrics", side_effect=metrics):
                result = oracle.build_report(definition, project_root=root, quiet=True)
                self.assertEqual(len(seen), 20, "v3 reuses C96 + four matching Exact arms; five treatments need Replay")
                self.assertTrue(all("_Direct" in name or "T40-MFE" in name for name in seen))
                self.assertEqual([r["target"] for r in result["oracles"]], list(definitions))
                second = oracle.build_report(definition, project_root=root, quiet=True)
                self.assertEqual(len(seen), 20, "Direct and new T40-MFE must reuse their per-arm caches")
                self.assertEqual(second["oracles"], result["oracles"])
                # A wrong source definition must not be substituted for Exact.
                index = HistoricalOracleIndex(root)
                context = {
                    "benchmark": Benchmark.benchmark_id,
                    "strategy_profile_id": "broker_cash_dl_conversion_rolling",
                    "dataset": "full", "start_date": "2021-01-01", "end_date": "2025-12-22",
                    "seeds": list(seeds), "param_sha_by_seed": {seed: "a" * 64 for seed in seeds},
                    "cohort": common, "definitions": definitions,
                }
                self.assertIsNotNone(index.lookup(target="CG40_EqualRank", seed=11, context=context))
                self.assertIsNone(index.lookup(target="R40_Direct", seed=11, context=context))
                self.assertIsNone(index.lookup(target="CG40_EqualRank", seed=11, context={
                    **context, "param_sha_by_seed": {seed: "c" * 64 for seed in seeds},
                }))
                self.assertIsNone(index.lookup(target="CG40_EqualRank", seed=11, context={
                    **context, "cohort": {**common, "cohort_rows": 9},
                }))

    def test_completed_v4_ten_target_report_reuses_all_seven_current_arms(self):
        # Resizing the current study must never repeat seven scientifically
        # identical historical observations or the same-window C96 control.
        from core.report_contract_versions import load_report_contract_versions
        seeds = (11, 22, 33, 44)
        class Benchmark:
            benchmark_id = "end_to_end_v1"
        common = {"cohort_rows": 4, "last_common_day": "2025-12-22"}
        definitions = dict(LEGACY_ORACLE_TARGETS)
        old_definitions = {**self._old_definitions(), **definitions}
        # Old V4 had the ten active controls, not all 11 legacy controls.
        old_order = ("AO40_EqualRank", "CG40_EqualRank", "U40", "FirstBreach_NetR",
                     "MFE40_Control", "R40", "T40", "MFE40_Direct", "R40_Direct", "T40_Direct")
        old_definitions = {name: old_definitions[name] for name in old_order}
        old_report = {
            "schema_version": 5, "status": "COMPLETED", "read_only": True,
            "hindsight_non_pit": True, "report_id": "target_oracle_romd",
            "report_type": oracle.SUPPORTED_AUDIT_TYPE,
            "report_contract_id": "audit.target_oracle_romd@v4",
            "report_contract_fingerprint": load_report_contract_versions()["audit.target_oracle_romd"][3]["fingerprint"],
            "strategy_profile_id": "broker_cash_dl_conversion_rolling",
            "benchmark_id": Benchmark.benchmark_id, "dataset": "full",
            "start_date": "2021-01-01", "end_date": "2025-12-22",
            "seeds": list(seeds), "common_cohort": common,
            "param_artifacts": [{"seed": seed, "param_sha256": "a" * 64} for seed in seeds],
            "definitions": old_definitions,
            "baseline": oracle._aggregate([self._observation("C96", seed) for seed in seeds], target="C96"),
            "oracles": [oracle._aggregate([self._observation(name, seed) for seed in seeds], target=name)
                        for name in old_order],
        }
        from core.audit_policy import get_reusable_audit_definitions
        definition = _legacy_definition()
        seen = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            previous = root / "outputs/audit/breakout_quality/reusable/target_oracle_romd/20261024T110500Z"
            previous.mkdir(parents=True)
            (previous / "audit.json").write_text(json.dumps(old_report), encoding="utf-8")
            with patch.object(oracle, "collect_status", return_value={"status": "READY"}), \
                 patch.object(oracle, "_robustness_artifacts", return_value=(Benchmark(), [(seed, Path(f"seed_{seed}.json")) for seed in seeds])), \
                 patch.object(oracle, "_build_oracle_score_panels", return_value=_mock_panels(common)), \
                 patch.object(oracle, "load_strategy_param_evaluation_view", return_value=({"kind": "rolling_active_param_ensemble", "payload": {}}, "a" * 64)), \
                 patch.object(oracle, "_validate_requested_param_policy"), \
                 patch.object(oracle, "get_active_param_ensemble_date_range", return_value=("2020-01-01", "2026-03-02")), \
                 patch.object(oracle, "_build_controlled_param_source_pair", return_value=("rolling_active_param_ensemble", "base", "oracle", None, None, None)), \
                 patch.object(oracle, "price_inventory_sha256", return_value="b" * 64), \
                 patch.object(oracle, "_run_scenario", side_effect=lambda *, name, **kwargs: (seen.append(name), {"name": name})[1]), \
                 patch.object(oracle, "_metrics", side_effect=lambda replay, *, target: self._observation(target, int(replay["name"].split("seed=")[1].split()[0]))):
                rebuilt = oracle.build_report(definition, project_root=root, quiet=True)
        self.assertEqual(len(seen), 8, "historical v4 matching observations must REUSE; both T40-MFE selectors need Replay")
        self.assertTrue(all("T40-MFE" in name for name in seen))
        self.assertEqual([x["target"] for x in rebuilt["oracles"]], list(definitions))
        self.assertEqual(rebuilt["schema_version"], 8)

    def test_incompatible_completed_v4_report_blocks_before_any_replay(self):
        from core.report_contract_versions import load_report_contract_versions
        from core.audit_policy import get_reusable_audit_definitions
        seeds = (11, 22, 33, 44)
        class Benchmark:
            benchmark_id = "end_to_end_v1"
        common = {"cohort_rows": 4, "last_common_day": "2025-12-22"}
        definitions = dict(LEGACY_ORACLE_TARGETS)
        report = {
            "schema_version": 5, "status": "COMPLETED", "read_only": True,
            "hindsight_non_pit": True, "report_id": "target_oracle_romd",
            "report_type": oracle.SUPPORTED_AUDIT_TYPE,
            "report_contract_id": "audit.target_oracle_romd@v4",
            "report_contract_fingerprint": load_report_contract_versions()["audit.target_oracle_romd"][3]["fingerprint"],
            "strategy_profile_id": "broker_cash_dl_conversion_rolling",
            "benchmark_id": Benchmark.benchmark_id, "dataset": "full",
            "start_date": "2021-01-01", "end_date": "2025-12-22",
            "seeds": list(seeds), "common_cohort": common,
            "param_artifacts": [{"seed": seed, "param_sha256": "c" * 64} for seed in seeds],
            "definitions": definitions,
            "baseline": oracle._aggregate([self._observation("C96", seed) for seed in seeds], target="C96"),
            "oracles": [oracle._aggregate([self._observation(name, seed) for seed in seeds], target=name)
                        for name in definitions],
        }
        definition = _legacy_definition()
        seen = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "outputs/audit/breakout_quality/reusable/target_oracle_romd/20261024T110500Z"
            directory.mkdir(parents=True)
            (directory / "audit.json").write_text(json.dumps(report), encoding="utf-8")
            with patch.object(oracle, "collect_status", return_value={"status": "READY"}), \
                 patch.object(oracle, "_robustness_artifacts", return_value=(Benchmark(), [(seed, Path(f"seed_{seed}.json")) for seed in seeds])), \
                 patch.object(oracle, "_build_oracle_score_panels", return_value=_mock_panels(common)), \
                 patch.object(oracle, "load_strategy_param_evaluation_view", return_value=({"kind": "rolling_active_param_ensemble", "payload": {}}, "a" * 64)), \
                 patch.object(oracle, "_validate_requested_param_policy"), \
                 patch.object(oracle, "get_active_param_ensemble_date_range", return_value=("2020-01-01", "2026-03-02")), \
                 patch.object(oracle, "_build_controlled_param_source_pair", return_value=("rolling_active_param_ensemble", "base", "oracle", None, None, None)), \
                 patch.object(oracle, "price_inventory_sha256", return_value="b" * 64), \
                 patch.object(oracle, "_run_scenario", side_effect=lambda *, name, **kwargs: seen.append(name)):
                with self.assertRaisesRegex(oracle.OracleBlockedError, "param_artifacts.param_sha256"):
                    oracle.build_report(definition, project_root=root, quiet=True)
                # A later Target-definition mismatch must also abort before
                # starting the earlier, otherwise runnable C96 arm.
                report["param_artifacts"] = [
                    {"seed": seed, "param_sha256": "a" * 64} for seed in seeds
                ]
                report["definitions"] = {**definitions, "R40": "changed science"}
                (directory / "audit.json").write_text(json.dumps(report), encoding="utf-8")
                with self.assertRaisesRegex(oracle.OracleBlockedError, "definitions.R40"):
                    oracle.build_report(definition, project_root=root, quiet=True)
        self.assertEqual(seen, [], "a mismatched completed arm must never trigger any replay")

    def test_partial_replay_is_saved_before_later_failure_and_resumes(self):
        seeds = (11, 22, 33, 44)
        class Benchmark:
            benchmark_id = "end_to_end_v1"
        definition = _legacy_definition()
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
                 patch.object(oracle, "_build_oracle_score_panels", return_value=_mock_panels(common)), \
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
                self.assertEqual(state["calls"], 41, "the first nine completed arms must not replay")
                self.assertEqual(len(resumed["oracles"]), len(LEGACY_ORACLE_TARGETS))
                oracle.build_report(definition, project_root=Path(tmp), quiet=True)
                self.assertEqual(state["calls"], 41)

    def test_compare_matrix_display_changes_do_not_replay_oracle(self):
        from core.strategy_compare_policy import get_strategy_comparison_settings
        seeds = (11, 22, 33, 44)
        class Benchmark:
            benchmark_id = "end_to_end_v1"
        common = {"cohort_rows": 4, "last_common_day": "2025-12-22"}
        definition = _legacy_definition()
        settings = get_strategy_comparison_settings(str(definition.source["strategy_profile_id"]))
        modified_arms = dict(settings.arms)
        modified_arms["C100"] = replace(settings.arms["C100"], enabled=True,
                                        name="new display title", description="new comparison-only description")
        modified_arms["C96"] = replace(settings.arms["C96"], name="renamed control")
        modified_sources = dict(settings.parameter_sources)
        baseline_source = settings.arms["C96"].param_source
        modified_sources[baseline_source] = replace(settings.parameter_sources[baseline_source],
                                                    description="new catalog documentation")
        display_changed = replace(settings, arms=modified_arms, parameter_sources=modified_sources)
        calls = []
        def run(*, name, **kwargs):
            calls.append(name)
            return {"name": name}
        def metrics(result, *, target):
            seed = int(result["name"].split("seed=")[1].split()[0])
            return self._observation(target, seed)
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(oracle, "collect_status", return_value={"status": "READY"}), \
                 patch.object(oracle, "_robustness_artifacts", return_value=(Benchmark(), [(seed, Path(f"seed_{seed}.json")) for seed in seeds])), \
                 patch.object(oracle, "_build_oracle_score_panels", return_value=_mock_panels(common)), \
                 patch.object(oracle, "load_strategy_param_evaluation_view", return_value=({"kind": "rolling_active_param_ensemble", "payload": {}}, "a" * 64)), \
                 patch.object(oracle, "_validate_requested_param_policy"), \
                 patch.object(oracle, "get_active_param_ensemble_date_range", return_value=("2020-01-01", "2026-03-02")), \
                 patch.object(oracle, "_build_controlled_param_source_pair", return_value=("rolling_active_param_ensemble", "base", "oracle", None, None, None)), \
                 patch.object(oracle, "price_inventory_sha256", return_value="b" * 64), \
                 patch.object(oracle, "_run_scenario", side_effect=run), \
                 patch.object(oracle, "_metrics", side_effect=metrics):
                report = oracle.build_report(definition, project_root=Path(tmp), quiet=True)
                self.assertEqual(len(calls), 40)
                with patch.object(oracle, "get_strategy_comparison_settings", return_value=display_changed):
                    reused = oracle.build_report(definition, project_root=Path(tmp), quiet=True)
                self.assertEqual(len(calls), 40, "presentation/membership must not cause new replay")
                self.assertEqual(reused["oracles"], report["oracles"])

    def test_legacy_c96_reuse_in_complete_four_seed_report_without_replay(self):
        from core.file_integrity import canonical_json_sha256
        from services.audit.oracle_replay_artifacts import OracleReplayStore
        seeds = (11, 22, 33, 44)
        class Benchmark:
            benchmark_id = "end_to_end_v1"
        definition = _legacy_definition()
        common = {"cohort_rows": 4, "last_common_day": "2025-12-22"}
        runs = []
        def replay(*, name, **kwargs):
            runs.append(name)
            return {"name": name}
        def metrics(result, *, target):
            seed = int(result["name"].split("seed=")[1].split()[0])
            return self._observation(target, seed)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(oracle, "collect_status", return_value={"status": "READY"}), \
                 patch.object(oracle, "_robustness_artifacts", return_value=(Benchmark(), [(seed, Path(f"seed_{seed}.json")) for seed in seeds])), \
                 patch.object(oracle, "_build_oracle_score_panels", return_value=_mock_panels(common)), \
                 patch.object(oracle, "load_strategy_param_evaluation_view", return_value=({"kind": "rolling_active_param_ensemble", "payload": {}}, "a" * 64)), \
                 patch.object(oracle, "_validate_requested_param_policy"), \
                 patch.object(oracle, "get_active_param_ensemble_date_range", return_value=("2020-01-01", "2026-03-02")), \
                 patch.object(oracle, "_build_controlled_param_source_pair", return_value=("rolling_active_param_ensemble", "base", "oracle", None, None, None)), \
                 patch.object(oracle, "price_inventory_sha256", return_value="b" * 64), \
                 patch.object(oracle, "_run_scenario", side_effect=replay), \
                 patch.object(oracle, "_metrics", side_effect=metrics):
                original = oracle.build_report(definition, project_root=root, quiet=True)
                self.assertEqual(len(runs), 40)
                store = OracleReplayStore(root, definition.output_subdir)
                canonical_path, = tuple((store.cache_root / "11").glob("C96_*.json"))
                current_payload = json.loads(canonical_path.read_text(encoding="utf-8"))
                legacy_identity = {
                    **current_payload["identity"], "cache_schema": 1,
                    "strategy_settings_fingerprint": "7e7058c39ad23498" + "0" * 48,
                }
                legacy_payload = {
                    **current_payload, "cache_schema": 1, "identity": legacy_identity,
                    "identity_sha256": canonical_json_sha256(legacy_identity),
                }
                legacy_path = store._path(legacy_identity)
                legacy_path.write_text(json.dumps(legacy_payload), encoding="utf-8")
                original_legacy_bytes = legacy_path.read_bytes()
                canonical_path.unlink()  # AI: Fixture: old cache exists, new key does not.
                rebuilt = oracle.build_report(definition, project_root=root, quiet=True)
                self.assertEqual(len(runs), 40, "verified old C96 must not run")
                self.assertEqual(rebuilt["oracles"], original["oracles"])
                self.assertEqual(rebuilt["baseline"], original["baseline"])
                self.assertEqual(legacy_path.read_bytes(), original_legacy_bytes)
                self.assertTrue(canonical_path.exists(), "v2 alias must be persistently reusable")
                self.assertEqual(json.loads(canonical_path.read_text(encoding="utf-8"))["origin"],
                                 "verified_legacy_replay_cache")

    def test_canonical_missing_results_run_then_reuse_without_permission_stage(self):
        seeds = (11, 22, 33, 44)
        class Benchmark:
            benchmark_id = "end_to_end_v1"
        definition = _legacy_definition()
        runs = []
        def replay(*, name, **_kwargs):
            runs.append(name)
            return {"name": name}
        def metrics(result, *, target):
            seed = int(result["name"].split("seed=")[1].split()[0])
            return self._observation(target, seed)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(oracle, "collect_status", return_value={"status": "READY"}), \
                 patch.object(oracle, "_robustness_artifacts", return_value=(Benchmark(), [(seed, Path(f"seed_{seed}.json")) for seed in seeds])), \
                 patch.object(oracle, "_build_oracle_score_panels", return_value=_mock_panels({"cohort_rows": 4, "last_common_day": "2025-12-22"})) as panels, \
                 patch.object(oracle, "load_strategy_param_evaluation_view", return_value=({"kind": "rolling_active_param_ensemble", "payload": {}}, "a" * 64)), \
                 patch.object(oracle, "_validate_requested_param_policy"), \
                 patch.object(oracle, "get_active_param_ensemble_date_range", return_value=("2020-01-01", "2026-03-02")), \
                 patch.object(oracle, "_build_controlled_param_source_pair", return_value=("rolling_active_param_ensemble", "base", "oracle", None, None, None)), \
                 patch.object(oracle, "price_inventory_sha256", return_value="b" * 64), \
                 patch.object(oracle, "_run_scenario", side_effect=replay), \
                 patch.object(oracle, "_metrics", side_effect=metrics):
                first = oracle.build_report(definition, project_root=root, quiet=True)
                self.assertEqual(len(runs), 40)
                second = oracle.build_report(definition, project_root=root, quiet=True)
                self.assertEqual(len(runs), 40)
                self.assertEqual(panels.call_count, 2)  # build_report alone does not persist the report fast path
                self.assertEqual(first["oracles"], second["oracles"])
                self.assertFalse(hasattr(oracle, "_existing_artifact_inventory"))

    def test_completed_schema6_report_restores_32_old_units_and_only_runs_new_target(self):
        from core.report_contract_versions import load_report_contract_versions
        seeds = (11, 22, 33, 44)
        class Benchmark:
            benchmark_id = "end_to_end_v1"
        common40 = {"cohort_rows": 4, "last_common_day": "2025-12-22"}
        recorded = []
        def replay(*, name, **kwargs):
            recorded.append(name)
            return {"name": name}
        def metrics(result, *, target):
            seed = int(result["name"].split("seed=")[1].split()[0])
            return self._observation(target, seed)
        definition = _legacy_definition()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(oracle, "collect_status", return_value={"status": "READY"}), \
                 patch.object(oracle, "_robustness_artifacts", return_value=(Benchmark(), [(seed, Path(f"seed_{seed}.json")) for seed in seeds])), \
                 patch.object(oracle, "_build_oracle_score_panels", return_value=_mock_panels(common40)), \
                 patch.object(oracle, "load_strategy_param_evaluation_view", return_value=({"kind": "rolling_active_param_ensemble", "payload": {}}, "a" * 64)), \
                 patch.object(oracle, "_validate_requested_param_policy"), \
                 patch.object(oracle, "get_active_param_ensemble_date_range", return_value=("2020-01-01", "2026-03-02")), \
                 patch.object(oracle, "_build_controlled_param_source_pair", return_value=("rolling_active_param_ensemble", "base", "oracle", None, None, None)), \
                 patch.object(oracle, "price_inventory_sha256", return_value="b" * 64), \
                 patch.object(oracle, "_run_scenario", side_effect=replay), \
                 patch.object(oracle, "_metrics", side_effect=metrics):
                report = oracle.build_report(definition, project_root=root, quiet=True)
                self.assertEqual(len(recorded), 40)
                # Emulate a genuine historical v6 report with seven treatments.
                new_targets = {"T40-MFE", "T40-MFE_Direct"}
                for name in new_targets:
                    report["definitions"].pop(name)
                report["oracles"] = [row for row in report["oracles"] if row["target"] not in new_targets]
                report.update({
                    "report_id": "target_oracle_romd", "report_type": oracle.SUPPORTED_AUDIT_TYPE,
                    "schema_version": 6,
                    "report_contract_id": "audit.target_oracle_romd@v6",
                    "report_contract_fingerprint": load_report_contract_versions()["audit.target_oracle_romd"][5]["fingerprint"],
                })
                directory = root / "outputs/audit/breakout_quality/reusable/target_oracle_romd"
                snapshot = directory / "20261024T104500Z"
                snapshot.mkdir(parents=True)
                (snapshot / "audit.json").write_text(json.dumps(report), encoding="utf-8")
                shutil.rmtree(directory / "replay_cache")
                rebuilt = oracle.build_report(definition, project_root=root, quiet=True)
                self.assertEqual(len(recorded), 48, "completed v6 report must REUSE 32 and calculate eight new units")
                self.assertEqual([row for row in rebuilt["oracles"] if row["target"] not in new_targets], report["oracles"])
                self.assertTrue(new_targets <= {row["target"] for row in rebuilt["oracles"]})
                console = io.StringIO()
                with redirect_stdout(console):
                    oracle.build_report(definition, project_root=root, quiet=False,
                                        preflight_display="summary")
                self.assertEqual(len(recorded), 48)
                self.assertIn("REUSE 40 / RUN（需計算）0 / BLOCKED 0", console.getvalue())
                self.assertIn("Replay preflight（執行前）", console.getvalue())
                self.assertNotIn("[REUSE]", console.getvalue())

    def test_completed_schema7_reuses_all_36_old_units_and_runs_only_mfe_direct(self):
        from core.report_contract_versions import load_report_contract_versions
        seeds = (11, 22, 33, 44)
        class Benchmark:
            benchmark_id = "end_to_end_v1"
        common = {"cohort_rows": 4, "last_common_day": "2025-12-22"}
        recorded = []
        def replay(*, name, **kwargs):
            recorded.append(name)
            return {"name": name}
        def metrics(result, *, target):
            seed = int(result["name"].split("seed=")[1].split()[0])
            return self._observation(target, seed)
        definition = _legacy_definition()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(oracle, "collect_status", return_value={"status": "READY"}), \
                 patch.object(oracle, "_robustness_artifacts", return_value=(Benchmark(), [(seed, Path(f"seed_{seed}.json")) for seed in seeds])), \
                 patch.object(oracle, "_build_oracle_score_panels", return_value=_mock_panels(common)), \
                 patch.object(oracle, "load_strategy_param_evaluation_view", return_value=({"kind": "rolling_active_param_ensemble", "payload": {}}, "a" * 64)), \
                 patch.object(oracle, "_validate_requested_param_policy"), \
                 patch.object(oracle, "get_active_param_ensemble_date_range", return_value=("2020-01-01", "2026-03-02")), \
                 patch.object(oracle, "_build_controlled_param_source_pair", return_value=("rolling_active_param_ensemble", "base", "oracle", None, None, None)), \
                 patch.object(oracle, "price_inventory_sha256", return_value="b" * 64), \
                 patch.object(oracle, "_run_scenario", side_effect=replay), \
                 patch.object(oracle, "_metrics", side_effect=metrics):
                current = oracle.build_report(definition, project_root=root, quiet=True)
                self.assertEqual(len(recorded), 40)
                # The prior completed report is a physically separate historical
                # snapshot; only its missing current Direct Replay is calculated.
                old = {**current,
                       "definitions": {name: desc for name, desc in current["definitions"].items() if name != "T40-MFE_Direct"},
                       "oracles": [row for row in current["oracles"] if row["target"] != "T40-MFE_Direct"],
                       "report_id": "target_oracle_romd",
                       "report_type": oracle.SUPPORTED_AUDIT_TYPE,
                       "schema_version": 7,
                       "report_contract_id": "audit.target_oracle_romd@v7",
                       "report_contract_fingerprint": load_report_contract_versions()["audit.target_oracle_romd"][6]["fingerprint"]}
                old_dir = root / "outputs/audit/breakout_quality/reusable/target_oracle_romd/20261024T154600Z"
                old_dir.mkdir(parents=True)
                (old_dir / "audit.json").write_text(json.dumps(old), encoding="utf-8")
                cache_root = root / "outputs/audit/breakout_quality/reusable/target_oracle_romd/replay_cache"
                shutil.rmtree(cache_root)
                previous = len(recorded)
                report = oracle.build_report(definition, project_root=root, quiet=True)
                self.assertEqual(len(recorded) - previous, 4)
                self.assertTrue(all("T40-MFE_Direct" in name for name in recorded[previous:]))
                self.assertEqual([x for x in report["oracles"] if x["target"] != "T40-MFE_Direct"], old["oracles"])
                previous = len(recorded)
                again = oracle.build_report(definition, project_root=root, quiet=True)
                self.assertEqual(len(recorded), previous)
                self.assertEqual(again["oracles"], report["oracles"])



if __name__ == "__main__":
    main(verbosity=2)
