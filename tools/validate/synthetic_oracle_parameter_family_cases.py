"""Regression tests for shared [1][1]/[1][2] Min and Full Oracle inventory.

Synthetic only: no market data, fitted models, Portfolio replays or Optimizer.
Run with python -m unittest tools.validate.synthetic_oracle_parameter_family_cases -q.
"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from core.audit_policy import get_reusable_audit_definitions
from core.audit_registry import (
    oracle_baseline_arm_ids, oracle_comparison_key, oracle_parameter_family,
    oracle_display_strategy, oracle_target_semantics,
    oracle_target_work_inventory, oracle_control_dependency_inventory,
    resolve_current_oracle_comparisons, validate_oracle_compare_family_coverage,
)
from core.strategy_compare_registry import STRATEGY_COMPARE_ARMS
from services.audit import target_oracle_romd as oracle
from services.research.target_dl_overview import _oracle_rows, collect_target_dl_overview


def definition():
    return next(d for d in get_reusable_audit_definitions("breakout_quality")
                if d.audit_type == oracle.SUPPORTED_AUDIT_TYPE)


class OracleParameterFamilyCases(TestCase):
    def test_event_netpnl_daily_p18_does_not_reactivate_legacy_p16(self):
        from config import research_summary as roster
        from core.research_summary import (
            get_research_summary_rows, get_research_summary_strategy_arm_ids,
            get_research_summary_oracle_source_catalog,
        )
        from core.strategy_compare_policy import get_strategy_comparison_settings
        from core.strategy_compare_registry import STRATEGY_COMPARE_ARMS

        assert_roster = get_research_summary_rows()
        by_arm = {binding["arm_id"]: group
                  for group in assert_roster
                  for binding in group.get("bindings", ()) if binding.get("arm_id")}
        archived = {binding["arm_id"]: group
                    for group in roster.RESEARCH_SUMMARY_ROWS
                    if not group.get("enabled", True)
                    for binding in group.get("bindings", ()) if binding.get("arm_id")}
        d = definition()
        current = {r.get("strategy_arm_id"): r
                   for r in resolve_current_oracle_comparisons(d.source)}
        self.assertTrue({"C163", "C164"}.issubset(current))
        self.assertTrue({"C162", "C159"}.isdisjoint(current))
        self.assertIn("C162", archived)
        self.assertTrue({"C123", "C124"}.isdisjoint(get_research_summary_strategy_arm_ids()))
        settings = get_strategy_comparison_settings("broker_cash_dl_conversion_rolling")
        active = {arm.arm_id for arm in settings.enabled_arms}
        self.assertTrue({"C163", "C164"}.issubset(active))
        self.assertTrue({"C162", "C159"}.isdisjoint(active))
        self.assertTrue({"C123", "C124"}.isdisjoint(active))
        for arm_id, historical, selector in (("C163", "C124", "direct"),):
            with self.subTest(arm=arm_id):
                new = by_arm[arm_id]["oracle"]
                frozen = archived[historical]["oracle"]
                self.assertEqual({k: new.get(k) for k in ("target", "selector", "score", "provider",
                                                            "horizon_mode", "score_source")},
                                 {k: frozen.get(k) for k in ("target", "selector", "score", "provider",
                                                                "horizon_mode", "score_source")})
                self.assertEqual(new["strategy_arm_id"], arm_id)
                self.assertEqual((new["strategy"], new["parameter_family"]), ("Daily", "daily_min"))
                self.assertEqual(new["selector"], selector)
                self.assertEqual(oracle_comparison_key(new),
                                 f'{new["target"]}::Daily::Daily_min')
                arm = STRATEGY_COMPARE_ARMS[arm_id]
                self.assertEqual((arm["dl_id"], arm["entry_strategy"], arm["param_source"]),
                                 ("CONT13CO_ROLL", "Daily", "daily_min_best_broker_cash_rolling_experiment"))
                self.assertEqual(arm["replay_contract_version"], 1)
                self.assertNotIn("positive_dl_id", arm.get("dl_runtime_options", {}))
        producers = {row["target"]: row for row in get_research_summary_oracle_source_catalog()}
        self.assertEqual(producers["DailyUniversal_MinTrade_NetPnL"]["score"],
                         archived["C123"]["oracle"]["score"])
        scope = next(sc for name, sc in oracle._oracle_strategy_scopes(d)
                     if name == "Daily::Daily_min")
        plan = oracle._oracle_comparison_plan(scope)
        self.assertEqual(plan["daily_recipes"]["DailyUniversal_MinTrade_NetPnL"],
                         ("daily_trade_path_net_pnl", "daily_trade40_profile", None))
        self.assertEqual(plan["aliases"]["DailyUniversal_MinTrade_NetPnL_Direct"],
                         "DailyUniversal_MinTrade_NetPnL")
        self.assertTrue({"C163-C161", "C164-C161", "C164-C134"}.issubset(
            {contrast.contrast_id for contrast in settings.enabled_contrasts}))
        # Same old Breakout/min receipt key can never become current again.
        with self.assertRaisesRegex(ValueError, "Historical-only Oracle target"):
            resolve_current_oracle_comparisons({**d.source,
                "oracle_comparisons": (*d.source["oracle_comparisons"],
                                      dict(archived["C123"]["oracle"]))})
        # Even a frozen historical template may not re-open its original Min family.
        modified = tuple({**row, "parameter_family": "min"}
                         if row.get("oracle_template_arm_id") == "C123" else row
                         for row in roster.RESEARCH_SUMMARY_ROWS)
        with patch.object(roster, "RESEARCH_SUMMARY_ROWS", modified):
            with self.assertRaisesRegex(ValueError, "Invalid Summary Oracle template"):
                get_research_summary_rows()

    def test_parameter_family_is_a_strategy_display_qualifier_not_a_target(self):
        d = definition()
        comparisons = resolve_current_oracle_comparisons(d.source)
        min_row = next(r for r in comparisons if r["target"] == "T40"
                       and oracle_parameter_family(r) == "min"
                       and r.get("strategy", "Breakout") == "Breakout")
        full_row = next(r for r in comparisons if r["target"] == "T40"
                        and oracle_parameter_family(r) == "full")
        self.assertEqual(oracle_target_semantics(min_row)[0], "T40")
        self.assertEqual(oracle_target_semantics(full_row)[0], "T40")
        self.assertEqual(oracle_display_strategy(strategy="Breakout", parameter_family="min"),
                         "Breakout")
        self.assertEqual(oracle_display_strategy(strategy="Breakout", parameter_family="full"),
                         "Breakout (Full)")
        # Neither historical key may be renamed by a presentation change.
        self.assertEqual(oracle_comparison_key(min_row), "T40")
        self.assertEqual(oracle_comparison_key(full_row), "T40::Full")
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "doc/scientific_contracts/REPORT_CONTRACT_VERSIONS.json"
            ledger.parent.mkdir(parents=True)
            ledger.write_bytes((Path(__file__).resolve().parents[2] /
                                "doc/scientific_contracts/REPORT_CONTRACT_VERSIONS.json").read_bytes())
            rows = collect_target_dl_overview(root)["rows"]
        by_arm = {r["strategy"]: r for r in rows}
        for arm in ("C153", "C154"):
            self.assertEqual(by_arm[arm]["target"], "T40")
            self.assertEqual(by_arm[arm]["entry_strategy"], "Breakout (Full)")
        self.assertEqual(by_arm["C97"]["target"], "")
        self.assertEqual(by_arm["C97"]["entry_strategy"], "Breakout (Full)")
        self.assertEqual(by_arm["C107"]["entry_strategy"], "Breakout")
        self.assertEqual(by_arm["C109"]["entry_strategy"], "Breakout")
        for arm, selector in (("C160", "Direct"),):
            self.assertEqual((by_arm[arm]["target"], by_arm[arm]["entry_strategy"],
                              by_arm[arm]["selector"]), ("T40", "Daily (Min/P18)", selector))
        self.assertEqual((by_arm["C161"]["target"], by_arm["C161"]["entry_strategy"],
                          by_arm["C161"]["selector"]), ("", "Daily (Min/P18)", "Default"))
        self.assertFalse(any("(Full)" in r["target"] or "P18" in r["target"] for r in rows))

    def test_every_oracle_arm_has_the_same_columns_in_both_menus(self):
        """Future families need no separate [1]/[2] column mapping."""
        from core.audit_registry import oracle_work_display_dimensions, oracle_work_inventory
        d = definition()
        dimensions = oracle_work_display_dimensions(d.source)
        self.assertEqual(tuple(dimensions), oracle_work_inventory(d.source))
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "doc/scientific_contracts/REPORT_CONTRACT_VERSIONS.json"
            ledger.parent.mkdir(parents=True)
            ledger.write_bytes((Path(__file__).resolve().parents[2] /
                                "doc/scientific_contracts/REPORT_CONTRACT_VERSIONS.json").read_bytes())
            rows = {row["strategy"]: row for row in collect_target_dl_overview(root)["rows"]
                    if row["strategy"]}
        referenced_arms = {
            oracle_comparison_key(row): row["strategy_arm_id"]
            for row in resolve_current_oracle_comparisons(d.source)
            if row.get("strategy_arm_id")
        }
        controls = set(oracle_baseline_arm_ids(d.source).values())
        by_arm = {
            (key if key in controls else referenced_arms[key]): value
            for key, value in dimensions.items()
            if key in controls or key in referenced_arms
        }
        for arm_id, expected in by_arm.items():
            with self.subTest(arm=arm_id):
                row = rows[arm_id]
                self.assertEqual((row["target"], row["entry_strategy"], row["selector"]),
                                 expected)
        # No-DL has no Target regardless of when its economic parameter
        # family was registered or whether it is an Oracle baseline.
        for arm_id, arm in STRATEGY_COMPARE_ARMS.items():
            if arm_id in rows and not arm.get("dl_enabled"):
                with self.subTest(control=arm_id):
                    self.assertEqual((rows[arm_id]["target"], rows[arm_id]["selector"]),
                                     ("", "Default"))
        # A future accidental conversion of a control to a scored arm fails
        # before either UI can render it as a baseline.
        control_id = next(iter(oracle_baseline_arm_ids(d.source).values()))
        with patch.dict(STRATEGY_COMPARE_ARMS, {
            control_id: {**STRATEGY_COMPARE_ARMS[control_id], "dl_enabled": True}
        }):
            with self.assertRaisesRegex(ValueError, "No-DL control"):
                oracle_work_display_dimensions(d.source)

    def test_formal_four_seed_plan_shows_full_in_strategy_only(self):
        import re
        d = definition()
        seeds = (11, 22, 33, 44)

        def prepared(scope, **_kwargs):
            base = scope.source["baseline_arm_id"]
            targets = (base, *(row["target"] for row in scope.source["oracle_comparisons"]))
            return {
                "seed_set": seeds, "targets": targets,
                "benchmark_id": "end_to_end_v1",
                "start_date": "2021-01-01", "end_date": "2026-03-02",
                "missing_p16": (), "producer_kwargs": {"baseline": base},
                "parameter_scientific_sha256_by_seed": {},
                "preflight": {(seed, target): "REUSE" for seed in seeds for target in targets},
                "completed_observations": {}, "completed_identities": {},
            }

        with TemporaryDirectory() as tmp, patch.object(
                oracle, "_run_single_stage_oracle", side_effect=prepared):
            result, _scopes = oracle._oracle_combined_preview(d, project_root=Path(tmp))
        visible = result["text"]
        self.assertNotIn("T40 (Full)", visible)
        self.assertGreaterEqual(len(re.findall(r"T40\s+Breakout \(Full\)\s+(Exact|Direct)", visible)), 2)
        target_table = visible.split("Portfolio REUSE", 1)[0]
        self.assertNotIn("Default", target_table)
        self.assertNotIn("C96", target_table)
        self.assertNotIn("C97", target_table)
        self.assertNotIn("C161", target_table)
        self.assertIn("C161", visible.split("Portfolio REUSE", 1)[1])
        self.assertEqual(tuple(result["targets"]), oracle_target_work_inventory(d.source))
        self.assertEqual(tuple(result["control_dependencies"]),
                         oracle_control_dependency_inventory(d.source))
        self.assertRegex(visible, r"T40\s+Breakout\s+Exact")
        self.assertRegex(visible, r"T40\s+Daily \(Min/P18\)\s+Direct")
        self.assertRegex(visible, r"T40\s+Daily \(Min/P18\)\s+Direct")
        self.assertIn("Daily (Min/P18)", visible)
        self.assertNotIn("T40 × Daily", visible)
        self.assertEqual(len(result["seed_set"]), 4)

    def test_no_dl_is_never_an_oracle_target_or_counted_as_target_run(self):
        d = definition()
        targets = oracle_target_work_inventory(d.source)
        controls = oracle_control_dependency_inventory(d.source)
        self.assertEqual(len(targets), len(resolve_current_oracle_comparisons(d.source)))
        self.assertEqual(set(controls), {"C96", "C97", "C161"})
        self.assertFalse(set(controls).intersection(targets))
        seed_set = (693545351, 2014432738, 104504529, 762252050)
        decisions = {(seed, target): "REUSE" for seed in seed_set for target in targets}
        decisions.update({(seed, control): "RUN" for seed in seed_set for control in controls})
        out = oracle.render_replay_preflight(
            label="Fixed", seeds=list(seed_set), targets=list(targets),
            control_dependencies=controls, decisions=decisions,
        )
        shown_targets = out.split("Portfolio REUSE", 1)[0]
        for control in controls:
            self.assertNotIn(control, shown_targets)
            self.assertIn(control, out.split("Portfolio REUSE", 1)[1])
        self.assertIn(f"Portfolio REUSE {len(targets) * len(seed_set)} / RUN", out)
        self.assertIn("RUN 4", out)
        self.assertIn("No-DL Portfolio", out)
        with self.assertRaisesRegex(ValueError, "mixed into Target"):
            oracle.render_replay_preflight(
                label="Fixed", seeds=list(seed_set), targets=[*targets, controls[0]],
                control_dependencies=controls, decisions=decisions,
            )
        row = next(r for r in resolve_current_oracle_comparisons(d.source)
                   if r.get("strategy_arm_id") == "C160")
        altered = {**d.source, "oracle_comparisons": tuple(
            ({**item, "strategy_arm_id": "C161"} if item.get("strategy_arm_id") == "C160" else item)
            for item in resolve_current_oracle_comparisons(d.source)
        )}
        with self.assertRaisesRegex(ValueError, "must be a scored strategy"):
            oracle_target_work_inventory(altered)

    def test_oracle_human_report_shows_target_not_family_suffix(self):
        import re
        seeds = (11, 22, 33, 44)
        def perf(target, strategy="Breakout", family="min", romd=10.0):
            return {
                "target": target, "strategy": strategy, "parameter_family": family,
                "total_return_pct": 12.0, "annualized_return_pct": 8.0,
                "mdd_pct": 10.0, "romd": romd, "romd_std": 0.0,
                "win_rate_pct": 60.0, "ev_a_r": 0.5,
                "seeds": len(seeds), "positive_seed_count": 2,
                "yearly": [], "seed_observations": [{"seed": seed, "romd": romd} for seed in seeds],
            }
        report = {
            "baseline": perf("C96"),
            "control_baselines": [perf("C97", family="full")],
            "oracles": [perf("T40", romd=21.0), perf("T40", family="full", romd=44.0)],
            "definitions": {"T40": "Canonical target"},
            "start_date": "2021-01-01", "end_date": "2026-03-02",
            "common_cohort": {"cohort_rows": 123},
            "benchmark_id": "end_to_end_v1", "seeds": seeds,
            "schema_version": 8, "limitations": [],
        }
        for kind in ("console", "markdown"):
            with self.subTest(kind=kind):
                visible = oracle.render_result(report, target=kind)
                self.assertNotIn("T40 (Full)", visible)
                self.assertIn("Breakout (Full)", visible)
                self.assertNotIn("|  | Breakout (Full) | Default |", visible)
                self.assertNotIn("|  | Breakout | Default |", visible)
                self.assertIn("T40", visible)
                if kind == "markdown":
                    self.assertIn("| T40 | Breakout (Full) | Exact |", visible)
                    self.assertIn("| T40 | Breakout | Exact |", visible)
                    self.assertIn("### T40 (Breakout (Full) / Exact)", visible)
                    # The reference No-DL control has a blank display Target,
                    # but must not be paired against itself in the seed table.
                    paired_heading = oracle.report_contract(
                        "audit.target_oracle_romd").sections[3].display_title
                    paired_section = visible.split(
                        f"## {paired_heading}\n", 1)[1].split("\n## ", 1)[0]
                    paired_data_lines = [line for line in paired_section.splitlines()
                                         if line.startswith("| ")]
                    self.assertEqual(len(paired_data_lines),
                                     1 + len(report["oracles"]) * len(seeds))
                else:
                    self.assertRegex(visible, r"T40\s+Breakout \(Full\)\s+Exact")

    def test_one_registry_drives_both_menu_inventories(self):
        d = definition()
        comparisons = resolve_current_oracle_comparisons(d.source)
        baselines = oracle_baseline_arm_ids(d.source)
        self.assertEqual(baselines, {"min": "C96", "full": "C97", "daily_min": "C161"})
        scopes = oracle._oracle_strategy_scopes(d)
        self.assertEqual([(name, scope.source["baseline_arm_id"],
                           len(scope.source["oracle_comparisons"])) for name, scope in scopes],
                         [("Breakout", "C96", len([row for row in comparisons
                            if oracle_parameter_family(row) == "min" and
                            row.get("strategy", "Breakout") == "Breakout"])),
                          ("Daily", "C96", len([row for row in comparisons
                            if oracle_parameter_family(row) == "min" and
                            row.get("strategy", "Breakout") == "Daily"])),
                          ("Breakout::Full", "C97", len([row for row in comparisons
                            if oracle_parameter_family(row) == "full"])),
                          ("Daily::Daily_min", "C161", len([row for row in comparisons
                            if oracle_parameter_family(row) == "daily_min"]))])
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = root / "doc/scientific_contracts/REPORT_CONTRACT_VERSIONS.json"
            ledger.parent.mkdir(parents=True)
            ledger.write_bytes((Path(__file__).resolve().parents[2] /
                                "doc/scientific_contracts/REPORT_CONTRACT_VERSIONS.json").read_bytes())
            targets, *_ = _oracle_rows(root, d, {})
            overview = collect_target_dl_overview(root)
        self.assertEqual(len(targets), len(comparisons) + len(baselines))
        self.assertEqual(targets, (
            *(key for family, control_id in baselines.items()
              for key in (control_id, *(
                  oracle_comparison_key(row) for row in comparisons
                  if oracle_parameter_family(row) == family))),
        ))
        self.assertEqual(set(targets), {"C96", "C97", "C161", *(
            oracle_comparison_key(row) for row in comparisons)})
        full_arms = {row["strategy"]: (row["target"], row["oracle_romd"])
                     for row in overview["rows"]
                     if row["strategy"] in {"C97", "C153", "C154"}}
        self.assertEqual(set(full_arms), {"C97", "C153", "C154"})
        self.assertTrue(all(metric is None for _target, metric in full_arms.values()))
        p18 = {row["strategy"]: row for row in overview["rows"]
               if row["strategy"] in {"C160", "C161", "C163", "C164"}}
        self.assertEqual(set(p18), {"C160", "C161", "C163", "C164"})
        self.assertEqual(p18["C160"]["target"], "T40")
        self.assertEqual(p18["C164"]["target"], "DailyUniversal_MinTrade_NetPnL_Positive")
        self.assertEqual({p18[arm]["entry_strategy"] for arm in p18}, {"Daily (Min/P18)"})
        self.assertEqual(p18["C161"]["target"], "")
        self.assertEqual(p18["C161"]["selector"], "Default")
        self.assertTrue(all(row["oracle_romd"] is None for row in p18.values()))

    def test_reference_arms_use_full_formal_parameters_and_same_t40_truth(self):
        d = definition()
        scopes = {name: scope for name, scope in oracle._oracle_strategy_scopes(d)}
        min_scoped, daily_scoped, full_scoped, p18_scoped = (
            scopes[name] for name in ("Breakout", "Daily", "Breakout::Full", "Daily::Daily_min"))
        settings, min_base, min_exact = oracle._contract(min_scoped)
        _settings, full_base, full_exact = oracle._contract(full_scoped)
        min_direct = oracle._direct_arm(min_scoped, settings, min_base, min_exact)
        full_direct = oracle._direct_arm(full_scoped, settings, full_base, full_exact)
        self.assertEqual((min_base.arm_id, full_base.arm_id), ("C96", "C97"))
        self.assertEqual((min_exact.arm_id, min_direct.arm_id), ("C103", "C101"))
        self.assertEqual((full_exact.arm_id, full_direct.arm_id), ("C153", "C154"))
        self.assertEqual((full_base.param_policy, full_base.rule_policy),
                         (min_base.param_policy, "formal"))
        self.assertNotEqual(min_base.param_source, full_base.param_source)
        min_plan = oracle._oracle_comparison_plan(min_scoped)
        full_plan = oracle._oracle_comparison_plan(full_scoped)
        for source, name in ((min_plan, "T40"), (full_plan, "T40")):
            self.assertEqual(source["recipes"][name], ("terminal_mae_gate", "pure_mfe40_provider"))
        self.assertEqual(full_plan["aliases"]["T40_Direct"], "T40")
        self.assertEqual(daily_scoped.source["baseline_arm_id"], "C96")
        self.assertNotEqual(full_scoped.output_subdir, min_scoped.output_subdir)
        _settings, p18_base, p18_exact = oracle._contract(p18_scoped)
        p18_direct = oracle._direct_arm(p18_scoped, settings, p18_base, p18_exact)
        self.assertIsNone(p18_exact)  # No active P18 Exact; historic C159 is signature-only.
        self.assertEqual((p18_base.arm_id, p18_direct.arm_id, p18_scoped.source["exact_arm_id"]),
                         ("C161", "C160", "C159"))
        self.assertNotEqual(p18_base.param_source, min_base.param_source)
        self.assertNotEqual(p18_base.param_source, full_base.param_source)
        self.assertEqual(oracle._oracle_comparison_plan(p18_scoped)["recipes"]["T40"],
                         min_plan["recipes"]["T40"])
        self.assertEqual(oracle._oracle_comparison_plan(p18_scoped)["aliases"]["T40_Direct"], "T40")
        self.assertNotEqual(p18_scoped.output_subdir, full_scoped.output_subdir)

    def test_min_and_full_have_distinct_canonical_optimizer_sources_for_four_seeds(self):
        d = definition()
        source_kwargs = {}

        def resolved(root, *, replication_seed, **kwargs):
            source_kwargs.setdefault(kwargs["training_profile_id"], set()).add(replication_seed)
            return {"ready": False, "action": "BUILD", "status": "MISSING",
                    "path": Path(root) / "pending.json",
                    "manifest_path": Path(root) / "pending_manifest.json"}

        with TemporaryDirectory() as temp, patch.object(
                oracle, "resolve_fixed_multiseed_robustness_strategy_parameter_artifact",
                side_effect=resolved):
            root = Path(temp)
            plans = {}
            for _name, scope in oracle._oracle_strategy_scopes(d):
                base = scope.source["baseline_arm_id"]
                plans[base] = oracle._oracle_parameter_reuse_plan(scope, root)
        self.assertEqual(set(plans), {"C96", "C97", "C161"})
        profiles = {base: plan["producer_kwargs"]["training_profile_id"]
                    for base, plan in plans.items()}
        self.assertEqual(len(set(profiles.values())), 3)
        self.assertEqual(profiles["C161"], "broker_cash_daily_min_best_seed42_current")
        self.assertTrue(all(len(plan["rows"]) == 4 for plan in plans.values()))
        self.assertEqual(set(source_kwargs), set(profiles.values()))
        self.assertTrue(all(not row["ready"] and row["action"] == "BUILD"
                            for plan in plans.values() for row in plan["rows"]))

    def test_full_parameter_view_keeps_formal_filters_and_min_keeps_all_off(self):
        d = definition()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            args = dict(root=root, parameter_paths=((1, root / "params.json"),),
                        start="2021-01-01", end="2026-03-02")
            observed = []

            def controlled(source, **kwargs):
                observed.append(kwargs)
                return "rolling_oos_param_schedule", {}, {}, None, None, None

            with patch.object(oracle, "load_strategy_param_evaluation_view", return_value=(
                    {"kind": "rolling_oos_param_schedule", "payload": {}}, "a" * 64)), \
                 patch.object(oracle, "get_active_param_date_range", return_value=(
                    "2020-01-01", "2026-03-02")), \
                 patch.object(oracle, "_validate_requested_param_policy"), \
                 patch.object(oracle, "_build_controlled_param_source_pair", side_effect=controlled):
                scopes = {name: scope for name, scope in oracle._oracle_strategy_scopes(d)}
                for name in ("Breakout", "Breakout::Full"):
                    scope = scopes[name]
                    base = oracle._contract(scope)[1]
                    oracle._prepare_oracle_parameters(scope, base_arm=base, **args)
            self.assertEqual(len(observed), 2)
            self.assertEqual(observed[0]["optional_entry_filter_policy"],
                             oracle.OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF)
            self.assertEqual(observed[0]["shared_param_overrides"],
                             oracle.ALL_RULE_FILTERS_OFF_OVERRIDES)
            self.assertEqual(observed[1]["optional_entry_filter_policy"],
                             oracle.OPTIONAL_ENTRY_FILTER_POLICY_CURRENT)
            self.assertIsNone(observed[1]["shared_param_overrides"])

    def test_unscored_control_native_reuse_matches_run_identity_for_each_parameter_family(self):
        """C96/C97 skip only perfect-score parity, not an effective execution change."""
        from services.audit.oracle_replay_artifacts import OracleReplayStore, replay_identity
        from services.research.oracle_artifact_contract import (
            OracleExecutionIndex, execution_context,
        )
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for _scope_name, scoped in oracle._oracle_strategy_scopes(definition()):
                baseline = scoped.source["baseline_arm_id"]
                settings, base_arm, exact_arm = oracle._contract(scoped)
                arm = exact_arm or settings.arms[str(scoped.source["exact_arm_id"])]
                policy = oracle._canonical_oracle_replay_policy(
                    scoped, base_arm=base_arm, exact_arm=arm,
                )
                context = execution_context(
                    dataset=settings.dataset, price_sha256="a" * 64,
                    param_sha256="b" * 64, seed=693545351, target=baseline,
                    start=str(settings.start_date),
                    end=str(scoped.dimensions["oracle_evaluation_end_date"]),
                    cutoff=str(settings.end_date), max_positions=settings.max_positions,
                    evaluation_mode=str(scoped.dimensions["evaluation_mode"]),
                    ranking_policy=str(arm.dl_runtime_mode),
                    ranking_options=dict(arm.dl_runtime_options),
                    replay_policy={**policy,
                        "oracle_parity_implementation": oracle.ORACLE_PARITY_IMPLEMENTATION,
                        "period_stats_implementation": oracle.PERIOD_STATS_IMPLEMENTATION},
                    score_contract=None, source_contract=None,
                    dataset_inputs=None, execution_gate=None,
                )
                physical_policy = oracle._arm_replay_policy(
                    scoped, plan={"selectors": {}, "daily_sources": {}}, target=baseline,
                    replay_policy=policy, effective_context=context,
                )
                self.assertNotIn("oracle_parity_implementation", physical_policy)
                cohort = {"cohort_rows": 2, "first_common_day": "2021-01-04",
                          "last_common_day": str(scoped.dimensions["oracle_evaluation_end_date"]),
                          "research_cutoff": str(settings.end_date)}
                identity = replay_identity(
                    dataset=settings.dataset, dataset_sha256="a" * 64,
                    target=baseline, definition={baseline: baseline},
                    start=str(settings.start_date),
                    end=str(scoped.dimensions["oracle_evaluation_end_date"]),
                    benchmark="end_to_end_v1", param_sha256="b" * 64,
                    seed=693545351, cohort=cohort,
                    ranking_policy=str(arm.dl_runtime_mode),
                    ranking_options=dict(arm.dl_runtime_options),
                    evaluation_mode=str(scoped.dimensions["evaluation_mode"]),
                    max_positions=settings.max_positions, replay_policy=physical_policy,
                )
                observation = {
                    "target": baseline, "seed": 693545351,
                    "total_return_pct": 100.0, "annualized_return_pct": 20.0,
                    "mdd_pct": 10.0, "romd": 10.0, "ev_a_r": 1.0,
                    "trades": 10, "exposure_pct": 40.0, "win_rate_pct": 50.0,
                    "yearly": [{"year": 2021, "strategy_return_pct": 10.0,
                                "is_full_year": True}],
                }
                store = OracleReplayStore(root, scoped.output_subdir)
                store.write(identity, observation, origin="unscored_control_test")
                index = OracleExecutionIndex(store)
                with self.subTest(baseline=baseline, output=scoped.output_subdir):
                    native = index.native_completed(context)
                    self.assertIsNotNone(native)
                    self.assertEqual(native[0], identity)
                    self.assertEqual(native[1], observation)
                    self.assertEqual(store.plan_execution({(693545351, baseline): identity}).action(
                        693545351, baseline), "REUSE")
                    index.publish(context, identity, observation)
                    self.assertEqual(index.read(context), (identity, observation))
                    # A score-bearing recipient never inherits the control exemption.
                    scored = {**context, "target": "T40", "score_contract": {"scope": "score"},
                              "source_contract": {"recipe": "terminal_mae_gate"}}
                    self.assertIn("oracle_parity_implementation",
                        index._economic_policy_differences(scored, identity))
                    # Changing an actual economic input still fails closed.
                    altered = {**context, "replay_policy": {
                        **context["replay_policy"], "score_threshold": 0.12345}}
                    self.assertIn("score_threshold",
                        index._economic_policy_differences(altered, identity))

    def test_full_replay_store_requires_its_own_c97_contrast(self):
        """Exercise actual signed read/write, not only a mocked Full report."""
        from services.audit.oracle_replay_artifacts import OracleReplayStore, replay_identity
        with TemporaryDirectory() as temp:
            root = Path(temp)
            store = OracleReplayStore(root, "breakout_quality/reusable/target_oracle_romd/parameter_family_full")

            def identity(target):
                return replay_identity(
                    dataset="full", dataset_sha256="a" * 64, target=target,
                    definition={target: target}, start="2021-01-01", end="2026-03-02",
                    benchmark="end_to_end_v2", param_sha256="b" * 64, seed=1,
                    cohort={"cohort_rows": 2, "first_common_day": "2021-01-04",
                            "last_common_day": "2026-03-02"},
                    ranking_policy="resource-aware-continuous-score-constrained-optimal",
                    ranking_options={"preserve_k_r0": True}, evaluation_mode="rolling",
                    max_positions=10, replay_policy={"baseline_arm": "C97"},
                )

            def observation(target, romd, **extra):
                return {"target": target, "seed": 1, "romd": romd,
                        "total_return_pct": 100.0, "annualized_return_pct": 20.0,
                        "mdd_pct": 10.0, "ev_a_r": 1.0, "trades": 2,
                        "exposure_pct": 40.0, "win_rate_pct": 50.0,
                        "yearly": [{"year": 2021, "strategy_return_pct": 10.0,
                                    "is_full_year": True}], **extra}

            base = identity("C97")
            scored = identity("T40")
            store.write(base, observation("C97", 8.0), origin="controlled_full_fixture")
            self.assertEqual(store.read(base)["romd"], 8.0)
            with self.assertRaises(KeyError):
                store.write(scored, observation("T40", 10.0,
                            delta_romd_vs_c96=2.0), origin="invalid_mislabeled_full")
            store.write(scored, observation("T40", 10.0,
                        delta_romd_vs_c97=2.0), origin="controlled_full_fixture")
            self.assertEqual(store.read(scored)["delta_romd_vs_c97"], 2.0)

    def test_published_full_report_requires_own_control_and_consumer_proofs(self):
        import json
        from core.training_policy import get_robustness_benchmark_policy_snapshot
        from core.report_contract_versions import load_report_contract_versions
        from core.strategy_compare_policy import get_strategy_comparison_settings
        from services.research.oracle_artifact_contract import reference_arm_execution_binding

        d = definition()
        comparisons = resolve_current_oracle_comparisons(d.source)
        baselines = oracle_baseline_arm_ids(d.source)
        settings = get_strategy_comparison_settings(d.source["strategy_profile_id"])
        versions = load_report_contract_versions(Path(__file__).resolve().parents[2])
        current = versions["audit.target_oracle_romd"][-1]
        keys = [baseline for family, arm_id in baselines.items()
                for baseline in [arm_id] + [oracle_comparison_key(row) for row in comparisons
                                              if oracle_parameter_family(row) == family]]
        seeds = list(get_robustness_benchmark_policy_snapshot()["resolved_seeds"])
        report = {
            "status": "COMPLETED", "read_only": True, "hindsight_non_pit": True,
            "report_contract_id": current["contract_id"],
            "report_contract_fingerprint": current["fingerprint"],
            "strategy_profile_id": d.source["strategy_profile_id"],
            "benchmark_id": "end_to_end_v1", "seeds": seeds,
            "start_date": "2021-01-01", "end_date": "2026-03-02",
            "baseline": {"target": "C96", "strategy": "Breakout", "seeds": len(seeds),
                         "romd": 9.0, "win_rate_pct": 40.0},
            "control_baselines": [
                {"target": "C97", "strategy": "Breakout", "parameter_family": "full",
                 "seeds": len(seeds), "romd": 14.5, "win_rate_pct": 44.0},
                {"target": "C161", "strategy": "Daily", "parameter_family": "daily_min",
                 "seeds": len(seeds), "romd": 12.5, "win_rate_pct": 47.0},
            ],
            "oracles": [{"target": row["target"], "strategy": row.get("strategy", "Breakout"),
                         "parameter_family": oracle_parameter_family(row),
                         "comparison_key": oracle_comparison_key(row), "seeds": len(seeds),
                         "romd": 42.0 if oracle_parameter_family(row) == "full" else 24.0,
                         "win_rate_pct": 61.0} for row in comparisons],
            "definitions": {row["target"]: row["description"] for row in comparisons},
            "comparison_inventory": keys,
            "comparison_reference_arms": {
                oracle_comparison_key(row): reference_arm_execution_binding(row, settings)
                for row in comparisons if row.get("strategy_arm_id")},
        }
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_dir = root / "outputs/audit" / d.output_subdir / "latest"
            result_dir.mkdir(parents=True)
            (result_dir / "audit.md").write_text("Published full Oracle report", encoding="utf-8")
            def inspect():
                (result_dir / "audit.json").write_text(json.dumps(report), encoding="utf-8")
                return _oracle_rows(root, d, versions)[3]
            values = inspect()
            self.assertEqual(set(values), set(keys))
            # An old 8S aggregate is historical-only; it cannot supply current 4S rows.
            report["benchmark_id"] = "end_to_end_v2"
            report["seeds"] = [*seeds, 9734312, 1020806211, 116293459, 1234683984]
            self.assertEqual(inspect(), {})
            report["benchmark_id"] = "end_to_end_v1"
            report["seeds"] = seeds
            self.assertEqual(values["C96"]["oracle_romd"], 9.0)
            self.assertEqual(values["C97"]["oracle_romd"], 14.5)
            self.assertEqual(values["T40::Full"]["oracle_romd"], 42.0)
            self.assertEqual(values["T40_Direct::Full"]["oracle_romd"], 42.0)
            # Declaring Full success without an independently completed C97
            # control is not a completed report, not an implicit C96 alias.
            report["control_baselines"] = []
            self.assertEqual(inspect(), {})
            report["control_baselines"] = [
                {"target": "C97", "strategy": "Breakout", "parameter_family": "full",
                 "seeds": len(seeds), "romd": 14.5, "win_rate_pct": 44.0},
                {"target": "C161", "strategy": "Daily", "parameter_family": "daily_min",
                 "seeds": len(seeds), "romd": 12.5, "win_rate_pct": 47.0},
            ]
            report["control_baselines"].pop()
            self.assertEqual(inspect(), {})  # New family also needs its own control.
            report["control_baselines"].append(
                {"target": "C161", "strategy": "Daily", "parameter_family": "daily_min",
                 "seeds": len(seeds), "romd": 12.5, "win_rate_pct": 47.0})
            report["comparison_reference_arms"].pop("T40_Direct::Full")
            values = inspect()
            self.assertNotIn("T40_Direct::Full", values)
            self.assertIn("T40::Full", values)

    def test_new_full_strategy_without_oracle_fails_but_min_suite_is_independent(self):
        d = definition()
        from core.strategy_compare_policy import get_strategy_comparison_settings
        settings = get_strategy_comparison_settings(d.source["strategy_profile_id"])
        arms = {key: dict(value) for key, value in STRATEGY_COMPARE_ARMS.items()}
        active = tuple(arm.arm_id for arm in settings.enabled_arms)
        validate_oracle_compare_family_coverage(d.source, active_arm_ids=active, arm_registry=arms)
        invalid = {**d.source, "oracle_comparisons": tuple(
            row for row in d.source["oracle_comparisons"]
            if oracle_comparison_key(row) != "T40_Direct::Full")}
        with self.assertRaisesRegex(ValueError, "no executable matching comparison"):
            validate_oracle_compare_family_coverage(invalid, active_arm_ids=active, arm_registry=arms)
        # Other [4] arms are deliberately NOT required to have a one-to-one
        # counterpart; shared Target/Selector rows remain supported.
        self.assertGreater(len(active), len(d.source["oracle_comparisons"]))

    def test_roster_scored_family_cannot_be_summary_only_without_oracle(self):
        from config import research_summary as cfg
        from services.research.target_oracle_application import get_formal_oracle_definition
        original = cfg.RESEARCH_SUMMARY_ROWS
        broken = tuple({key: value for key, value in row.items()
                        if key != "oracle_template_arm_id"}
                       if row.get("oracle_template_arm_id") == "C146" else row
                       for row in original)
        with patch.object(cfg, "RESEARCH_SUMMARY_ROWS", broken):
            with self.assertRaisesRegex(ValueError, "no executable matching comparison"):
                get_formal_oracle_definition("breakout_quality")

    def test_oracle_authorized_missing_full_parameters_use_own_canonical_producer(self):
        """Missing Full/P17 is built on its own source; Min/P16 remains reused."""
        from services.research import target_oracle_application as app
        from services.optimizer import strategy_param_service
        d = definition()
        from core.training_policy import get_robustness_benchmark_policy_snapshot
        seeds = tuple(get_robustness_benchmark_policy_snapshot()["resolved_seeds"])
        profiles = {"C96": "broker_cash_min_best_seed42_current",
                    "C97": "broker_cash_full_best_seed42_current",
                    "C161": "broker_cash_daily_min_best_seed42_current"}
        done = {"value": False}
        def inspect(scoped, *, project_root):
            baseline = scoped.source["baseline_arm_id"]
            return {"rows": tuple({
                "seed": seed,
                "ready": baseline != "C97" or seed != seeds[-1] or done["value"],
                "action": "REUSE" if (baseline != "C97" or seed != seeds[-1] or done["value"]) else "BUILD",
                "status": "READY" if (baseline != "C97" or seed != seeds[-1] or done["value"]) else "MISSING",
            } for seed in seeds),
                "producer_kwargs": {"training_profile_id": profiles[baseline]}}
        plans = {
            "C96": {"seed_set": seeds, "missing": (),
                    "producer_kwargs": {"training_profile_id": profiles["C96"]}},
            "C97": {"seed_set": seeds, "missing": (seeds[-1],),
                    "producer_kwargs": {"training_profile_id": profiles["C97"]}},
            "C161": {"seed_set": seeds, "missing": (),
                     "producer_kwargs": {"training_profile_id": profiles["C161"]}},
        }
        with TemporaryDirectory() as temp, \
             patch.object(app, "inspect_oracle_parameter_prerequisites", side_effect=inspect), \
             patch.object(strategy_param_service,
                          "ensure_fixed_multiseed_robustness_strategy_parameter_artifact") as producer:
            # Missing an entire family from approval must be rejected without fitting.
            with self.assertRaisesRegex(RuntimeError, "缺少 C97"):
                app.ensure_authorized_oracle_parameters(
                    project_root=Path(temp), definition=d,
                    approved_plan={"parameter_plans": {
                        "C96": plans["C96"], "C161": plans["C161"]}},
                )
            producer.assert_not_called()
            def build(_root, *, replication_seed, training_profile_id, **kwargs):
                self.assertEqual(training_profile_id, profiles["C97"])
                self.assertEqual(replication_seed, seeds[-1])
                self.assertTrue(kwargs["resume_parameter_training"])
                done["value"] = True
                return {"action": "BUILD"}
            producer.side_effect = build
            result = app.ensure_authorized_oracle_parameters(
                project_root=Path(temp), definition=d,
                approved_plan={"parameter_plans": plans},
            )
            self.assertEqual(result["built_or_repaired"], 1)
            producer.assert_called_once()

    def test_full_baseline_receipts_have_different_reuse_keys_and_plan(self):
        d = definition()
        from core.training_policy import get_robustness_benchmark_policy_snapshot
        seeds = tuple(get_robustness_benchmark_policy_snapshot()["resolved_seeds"])
        self.assertEqual(len(seeds), 4)
        comparisons = resolve_current_oracle_comparisons(d.source)
        def fake_run(scoped, *, preview_only=False, **_kwargs):
            self.assertTrue(preview_only)
            family = next(family for family, arm in oracle_baseline_arm_ids(d.source).items()
                          if arm == scoped.source["baseline_arm_id"])
            base = scoped.source["baseline_arm_id"]
            targets = (base, *(row["target"] for row in scoped.source["oracle_comparisons"]))
            return {"status": "PREVIEW", "preflight": {
                (seed, target): ("REUSE" if family == "min" else "RUN")
                for seed in seeds for target in targets},
                "benchmark_id": "end_to_end_v1", "start_date": "2021-01-01",
                "end_date": "2026-03-02", "seed_set": seeds,
                "completed_observations": {}, "missing_p16": (),
                "producer_kwargs": {"training_profile_id": family}, "targets": targets}
        with patch.object(oracle, "_run_single_stage_oracle", side_effect=fake_run):
            preview = oracle.preview_oracle_replay_plan(d, project_root=Path("/tmp"))
        keys = preview["targets"]
        self.assertEqual(tuple(keys), oracle_target_work_inventory(d.source))
        self.assertEqual(tuple(preview["control_dependencies"]),
                         oracle_control_dependency_inventory(d.source))
        self.assertEqual(len(keys), len(comparisons))
        self.assertEqual(len(preview["preflight"]),
                         (len(keys) + len(preview["control_dependencies"])) * len(seeds))
        min_keys = {oracle_comparison_key(row) for row in comparisons
                    if oracle_parameter_family(row) == "min"}
        new_keys = {"T40_Direct::Daily::Daily_min",
                    "TrueDailyUniversal_MinTrade_NetPnL_Positive_Direct::Daily::Daily_min"}
        self.assertTrue(all(preview["preflight"][(seed, key)] == "REUSE"
                            for seed in seeds for key in min_keys))
        self.assertTrue(all(preview["preflight"][(seed, key)] == "RUN"
                            for seed in seeds for key in keys if key not in min_keys))
        self.assertTrue(new_keys.issubset(set(keys)))
        self.assertFalse(set(preview["control_dependencies"]).intersection(keys))
        self.assertEqual(set(preview["parameter_plans"]), {"C96", "C97", "C161"})
        self.assertNotEqual(preview["parameter_plans"]["C161"]["producer_kwargs"],
                            preview["parameter_plans"]["C96"]["producer_kwargs"])
        self.assertNotEqual(preview["parameter_plans"]["C96"]["producer_kwargs"],
                            preview["parameter_plans"]["C97"]["producer_kwargs"])
        self.assertEqual(tuple(row["target"] for row in comparisons if
                               oracle_parameter_family(row) == "full"),
                         ("T40", "T40_Direct", "FullTrade_NetPnL",
                          "FullTrade_NetPnL_Direct"))


class BaselineRobustnessReuseCases(TestCase):
    """Projection is optional, read-only, and never promotes model scores."""

    @staticmethod
    def _fixture(root: Path, arm_id: str):
        import csv
        import hashlib
        from core.strategy_compare_policy import get_strategy_comparison_settings
        from core.training_policy import get_robustness_benchmark_policy_snapshot
        from services.research.strategy_multi_seed_robustness import (
            SEED_RESULTS_FILENAME, SEED_YEARLY_RESULTS_FILENAME,
        )

        settings = get_strategy_comparison_settings("broker_cash_dl_conversion_rolling")
        arm = settings.arms[arm_id]
        four_seeds = tuple(get_robustness_benchmark_policy_snapshot(
            benchmark_id="end_to_end_v1")["resolved_seeds"])
        all_seeds = tuple(get_robustness_benchmark_policy_snapshot(
            benchmark_id="end_to_end_v2")["resolved_seeds"])
        assert len(four_seeds) == 4 and all_seeds[:4] == four_seeds
        sha = "a" * 64
        observations = []
        rows = []
        for number, seed in enumerate(all_seeds, start=1):
            observations.append({
                "seed": seed, "romd": float(number + 5),
                "total_return_pct": float(number + 50),
                "mdd_pct": float(number + 10),
                "param_scientific_sha256": sha, "param_sha256": "b" * 64,
                "dataset": settings.dataset,
                "max_positions": settings.max_positions, "rotation": settings.rotation,
            })
            if seed in four_seeds:
                rows.append({
                    "arm_id": arm_id, "seed": str(seed),
                    "return_over_max_drawdown": str(float(number + 5)),
                    "total_return_pct": str(float(number + 50)),
                    "max_drawdown_pct": str(float(number + 10)),
                    "strategy_param_sha256": sha,
                })
        with (root / SEED_RESULTS_FILENAME).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        (root / SEED_YEARLY_RESULTS_FILENAME).write_text("seed,year\n", encoding="utf-8")
        artifacts = {}
        for key, filename in (("seed_results", SEED_RESULTS_FILENAME),
                              ("seed_yearly_results", SEED_YEARLY_RESULTS_FILENAME)):
            path = root / filename
            artifacts[key] = {
                "filename": filename, "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        contract = {
            "benchmark_id": "end_to_end_v1",
            "comparison_period": {"start": "2021-01-01", "end": "2026-03-02"},
            "dataset": settings.dataset, "max_positions": settings.max_positions,
            "rotation": settings.rotation,
            "stochastic_arms": [{
                "arm_id": arm_id, "param_source": arm.param_source,
                "param_policy": arm.param_policy, "rule_policy": arm.rule_policy,
                "entry_strategy": arm.entry_strategy,
                "replay_contract_version": arm.replay_contract_version,
                "model_seed_sensitive": False,
            }],
            "benchmark_parameter_artifact_identities": {
                f"{arm_id}:seed={seed}": {"sha256": sha} for seed in four_seeds
            },
        }
        oracle_info = {
            "benchmark": "end_to_end_v2", "seeds": all_seeds,
            "start_date": "2021-01-01", "end_date": "2026-03-02",
            "baseline_evidence": {arm_id: tuple(observations)},
        }
        mean = sum(float(row["return_over_max_drawdown"]) for row in rows) / 4
        return settings, four_seeds, all_seeds, contract, oracle_info, {"durable_artifacts": artifacts}, mean

    def test_no_dl_signed_8s_reuses_only_matching_first_four(self):
        from statistics import fmean
        from services.research.target_dl_overview import _certified_no_dl_oracle_8s
        for arm_id in ("C96", "C97"):
            with self.subTest(arm=arm_id), TemporaryDirectory() as tmp:
                root = Path(tmp)
                settings, four, eight, contract, oracle_info, manifest, mean = self._fixture(root, arm_id)
                kwargs = dict(root=root, run=root, manifest=manifest, contract=contract,
                              oracle_info=oracle_info, arm_id=arm_id, settings=settings,
                              four_seeds=four, published_four_mean=mean)
                self.assertEqual(_certified_no_dl_oracle_8s(**kwargs),
                                 (fmean(float(x["romd"]) for x in oracle_info["baseline_evidence"][arm_id]),
                                  eight, "end_to_end_v2"))
                incomplete = {**oracle_info, "baseline_evidence": {
                    arm_id: oracle_info["baseline_evidence"][arm_id][:-1]}}
                self.assertIsNone(_certified_no_dl_oracle_8s(**{**kwargs, "oracle_info": incomplete}))
                mismatch = {**oracle_info, "baseline_evidence": {
                    arm_id: [{**item, **({"romd": 999.0} if i == 0 else {})}
                             for i, item in enumerate(oracle_info["baseline_evidence"][arm_id])]}}
                self.assertIsNone(_certified_no_dl_oracle_8s(**{**kwargs, "oracle_info": mismatch}))
                self.assertIsNone(_certified_no_dl_oracle_8s(**{**kwargs, "published_four_mean": mean + 1}))
                self.assertIsNone(_certified_no_dl_oracle_8s(**{**kwargs, "contract": {
                    **contract, "comparison_period": {"start": "2022-01-01", "end": "2026-03-02"}}}))
                self.assertIsNone(_certified_no_dl_oracle_8s(**{**kwargs, "contract": {
                    **contract, "benchmark_parameter_artifact_identities": {}}}))
                self.assertIsNone(_certified_no_dl_oracle_8s(**{**kwargs, "manifest": {
                    "durable_artifacts": {}}}))
                self.assertIsNone(_certified_no_dl_oracle_8s(**{**kwargs, "oracle_info": {
                    **oracle_info, "benchmark": "unknown"}}))

    def test_current_four_seed_overview_never_promotes_historical_eight_seed_aggregate(self):
        import json
        from config.strategy_compare import STRATEGY_COMPARE_CURRENT_ROLLING_ROBUSTNESS
        from core.report_contract_versions import load_report_contract_versions
        from core.strategy_compare_policy import get_strategy_multi_seed_robustness_settings
        from services.research import target_dl_overview as overview

        with TemporaryDirectory() as temp:
            root = Path(temp)
            settings, four, eight, contract, oracle_info, manifest, mean = self._fixture(root, "C96")
            cfg = get_strategy_multi_seed_robustness_settings(STRATEGY_COMPARE_CURRENT_ROLLING_ROBUSTNESS)
            folder = root / cfg.output_root / "fake-four-seed-fingerprint"
            folder.mkdir(parents=True)
            for name in ("seed_results.csv", "seed_yearly_returns.csv"):
                (folder / name).write_bytes((root / name).read_bytes())
            contract.update({"fingerprint": folder.name,
                             "profile_id": cfg.profile_id,
                             "resolved_seeds": list(four), "seed_count": len(four),
                             "stochastic_arms": [contract["stochastic_arms"][0],
                                                 {"arm_id": "C97"}, {"arm_id": "C107"}]})
            versions = load_report_contract_versions(Path(__file__).resolve().parents[2])
            approved = versions["strategy.standard_sop"][-1]
            summary = {
                "status": "RESULT_AVAILABLE", "contract": contract,
                "report_contract_id": approved["contract_id"],
                "report_contract_fingerprint": approved["fingerprint"],
                "romd_statistics": [
                    {"arm_id": "C96", "mean": mean, "n": len(four)},
                    {"arm_id": "C97", "mean": 6.44, "n": len(four)},
                    {"arm_id": "C107", "mean": 10.45, "n": len(four)},
                ],
            }
            manifest.update({"status": "COMPLETED", "contract": contract})
            (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (folder / "robustness_summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (folder / "robustness_report.md").write_text("Approved synthetic 4S", encoding="utf-8")
            # Only the verified baseline may use the Oracle 8S evidence;
            # the extra DL arm's published four-seed number stays untouched.
            with patch.object(overview, "_robust_arm_fit_lineage_current", return_value=True):
                result = overview._strategy_robustness_metrics(
                    root, versions, oracle_info=oracle_info,
                    oracle_source=definition().source,
                )
            self.assertEqual(result["C96"], (mean, four, "end_to_end_v1"))
            self.assertEqual(result["C97"], (6.44, four, "end_to_end_v1"))
            self.assertEqual(result["C107"], (10.45, four, "end_to_end_v1"))

    def test_controls_display_same_names_and_no_oracle_metrics(self):
        from core.audit_registry import oracle_baseline_display_name
        d = definition()
        for arm_id in oracle_baseline_arm_ids(d.source).values():
            self.assertEqual(oracle_baseline_display_name(d.source, arm_id), "")
        with self.assertRaisesRegex(ValueError, "Unknown Oracle no-DL control"):
            oracle_baseline_display_name(d.source, "C153")
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = root / "doc/scientific_contracts/REPORT_CONTRACT_VERSIONS.json"
            ledger.parent.mkdir(parents=True)
            ledger.write_bytes((Path(__file__).resolve().parents[2] /
                                "doc/scientific_contracts/REPORT_CONTRACT_VERSIONS.json").read_bytes())
            result = collect_target_dl_overview(root)
            rows = {row["strategy"]: row for row in result["rows"]}
            for arm_id in ("C96", "C97", "C98", "C161"):
                self.assertEqual(rows[arm_id]["target"], "")
                self.assertEqual(rows[arm_id]["selector"], "Default")
                self.assertIsNone(rows[arm_id]["oracle_romd"])
                self.assertIsNone(rows[arm_id]["win_rate_pct"])


if __name__ == "__main__":
    from unittest import main
    main()
