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
    resolve_current_oracle_comparisons, validate_oracle_compare_family_coverage,
)
from core.strategy_compare_registry import STRATEGY_COMPARE_ARMS
from services.audit import target_oracle_romd as oracle
from services.research.target_dl_overview import _oracle_rows, collect_target_dl_overview


def definition():
    return next(d for d in get_reusable_audit_definitions("breakout_quality")
                if d.audit_type == oracle.SUPPORTED_AUDIT_TYPE)


class OracleParameterFamilyCases(TestCase):
    def test_one_registry_drives_both_menu_inventories(self):
        d = definition()
        comparisons = resolve_current_oracle_comparisons(d.source)
        baselines = oracle_baseline_arm_ids(d.source)
        self.assertEqual(baselines, {"min": "C96", "full": "C97"})
        scopes = oracle._oracle_strategy_scopes(d)
        self.assertEqual([(name, scope.source["baseline_arm_id"],
                           len(scope.source["oracle_comparisons"])) for name, scope in scopes],
                         [("Breakout", "C96", 18), ("Daily", "C96", 6),
                          ("Breakout::Full", "C97", 2)])
        with TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = root / "doc/scientific_contracts/REPORT_CONTRACT_VERSIONS.json"
            ledger.parent.mkdir(parents=True)
            ledger.write_bytes((Path(__file__).resolve().parents[2] /
                                "doc/scientific_contracts/REPORT_CONTRACT_VERSIONS.json").read_bytes())
            targets, *_ = _oracle_rows(root, d, {})
            overview = collect_target_dl_overview(root)
        self.assertEqual(len(targets), 28)
        self.assertEqual(targets, (
            "C96", *(oracle_comparison_key(row) for row in comparisons
                     if oracle_parameter_family(row) == "min"),
            "C97", *(oracle_comparison_key(row) for row in comparisons
                     if oracle_parameter_family(row) == "full"),
        ))
        self.assertEqual(set(targets), {"C96", "C97", *(
            oracle_comparison_key(row) for row in comparisons)})
        full_arms = {row["strategy"]: (row["target"], row["oracle_romd"])
                     for row in overview["rows"]
                     if row["strategy"] in {"C97", "C153", "C154"}}
        self.assertEqual(set(full_arms), {"C97", "C153", "C154"})
        self.assertTrue(all(metric is None for _target, metric in full_arms.values()))

    def test_reference_arms_use_full_formal_parameters_and_same_t40_truth(self):
        d = definition()
        min_scoped, daily_scoped, full_scoped = (scope for _name, scope
                                                  in oracle._oracle_strategy_scopes(d))
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

    def test_min_and_full_have_distinct_canonical_optimizer_sources_for_all_eight_seeds(self):
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
        self.assertEqual(set(plans), {"C96", "C97"})
        self.assertNotEqual(plans["C96"]["producer_kwargs"]["training_profile_id"],
                            plans["C97"]["producer_kwargs"]["training_profile_id"])
        self.assertEqual(len(plans["C96"]["rows"]), 8)
        self.assertEqual(len(plans["C97"]["rows"]), 8)
        self.assertEqual(set(source_kwargs), {plans["C96"]["producer_kwargs"]["training_profile_id"],
                                             plans["C97"]["producer_kwargs"]["training_profile_id"]})
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
        from core.report_contract_versions import load_report_contract_versions
        from core.strategy_compare_policy import get_strategy_comparison_settings
        from services.research.oracle_artifact_contract import reference_arm_execution_binding

        d = definition()
        comparisons = resolve_current_oracle_comparisons(d.source)
        baselines = oracle_baseline_arm_ids(d.source)
        settings = get_strategy_comparison_settings(d.source["strategy_profile_id"])
        versions = load_report_contract_versions(Path(__file__).resolve().parents[2])
        current = versions["audit.target_oracle_romd"][-1]
        keys = ["C96", *(oracle_comparison_key(row) for row in comparisons
                         if oracle_parameter_family(row) == "min"), "C97",
                *(oracle_comparison_key(row) for row in comparisons
                  if oracle_parameter_family(row) == "full")]
        report = {
            "status": "COMPLETED", "read_only": True, "hindsight_non_pit": True,
            "report_contract_id": current["contract_id"],
            "report_contract_fingerprint": current["fingerprint"],
            "strategy_profile_id": d.source["strategy_profile_id"],
            "benchmark_id": "end_to_end_v2", "seeds": list(range(1, 9)),
            "start_date": "2021-01-01", "end_date": "2026-03-02",
            "baseline": {"target": "C96", "strategy": "Breakout", "seeds": 8,
                         "romd": 9.0, "win_rate_pct": 40.0},
            "control_baselines": [{"target": "C97", "strategy": "Breakout",
                                   "parameter_family": "full", "seeds": 8,
                                   "romd": 14.5, "win_rate_pct": 44.0}],
            "oracles": [{"target": row["target"], "strategy": row.get("strategy", "Breakout"),
                         "parameter_family": oracle_parameter_family(row),
                         "comparison_key": oracle_comparison_key(row), "seeds": 8,
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
            self.assertEqual(values["C96"]["oracle_romd"], 9.0)
            self.assertEqual(values["C97"]["oracle_romd"], 14.5)
            self.assertEqual(values["T40::Full"]["oracle_romd"], 42.0)
            self.assertEqual(values["T40_Direct::Full"]["oracle_romd"], 42.0)
            # Declaring Full success without an independently completed C97
            # control is not a completed report, not an implicit C96 alias.
            report["control_baselines"] = []
            self.assertEqual(inspect(), {})
            report["control_baselines"] = [{"target": "C97", "strategy": "Breakout",
                                            "parameter_family": "full", "seeds": 8,
                                            "romd": 14.5, "win_rate_pct": 44.0}]
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

    def test_oracle_never_builds_missing_full_parameters(self):
        """[1] cannot authorize missing Full/P17 Optimizer seeds or borrow Min/P16."""
        from services.research import target_oracle_application as app
        from services.optimizer import strategy_param_service
        d = definition()
        seeds = (693545351, 2014432738)
        profiles = {"C96": "broker_cash_min_best_seed42_current",
                    "C97": "broker_cash_full_best_seed42_current"}
        def inspect(scoped, *, project_root):
            baseline = scoped.source["baseline_arm_id"]
            return {"rows": tuple({
                "seed": seed, "ready": baseline == "C96" or seed == seeds[0],
                "action": "REUSE" if baseline == "C96" or seed == seeds[0] else "BUILD",
                "status": "READY" if baseline == "C96" or seed == seeds[0] else "MISSING",
            } for seed in seeds),
                "producer_kwargs": {"training_profile_id": profiles[baseline]}}
        plans = {
            "C96": {"seed_set": seeds, "missing": (),
                    "producer_kwargs": {"training_profile_id": profiles["C96"]}},
            "C97": {"seed_set": seeds, "missing": (seeds[1],),
                    "producer_kwargs": {"training_profile_id": profiles["C97"]}},
        }
        with TemporaryDirectory() as temp, \
             patch.object(app, "inspect_oracle_parameter_prerequisites", side_effect=inspect), \
             patch.object(strategy_param_service, "ensure_fixed_multiseed_robustness_strategy_parameter_artifact",
                          side_effect=AssertionError("Optimizer must not run")) as producer:
            with self.assertRaisesRegex(RuntimeError, "缺少正式參數家族"):
                app.ensure_authorized_oracle_parameters(
                    project_root=Path(temp), definition=d,
                    approved_plan={"missing_p16": (), "seed_set": seeds},
                )
            with self.assertRaisesRegex(RuntimeError, "不執行 Optimizer"):
                app.ensure_authorized_oracle_parameters(
                    project_root=Path(temp), definition=d,
                    approved_plan={"parameter_plans": plans},
                )
            producer.assert_not_called()

    def test_full_baseline_receipts_have_different_reuse_keys_and_plan(self):
        d = definition()
        seeds = (1, 2)
        comparisons = resolve_current_oracle_comparisons(d.source)
        def fake_run(scoped, *, preview_only=False, **_kwargs):
            self.assertTrue(preview_only)
            family = "full" if scoped.source["baseline_arm_id"] == "C97" else "min"
            base = scoped.source["baseline_arm_id"]
            targets = (base, *(row["target"] for row in scoped.source["oracle_comparisons"]))
            return {"status": "PREVIEW", "preflight": {
                (seed, target): ("RUN" if family == "full" else "REUSE")
                for seed in seeds for target in targets},
                "benchmark_id": "end_to_end_v2", "start_date": "2021-01-01",
                "end_date": "2026-03-02", "seed_set": seeds,
                "completed_observations": {}, "missing_p16": (),
                "producer_kwargs": {"training_profile_id": family}, "targets": targets}
        with patch.object(oracle, "_run_single_stage_oracle", side_effect=fake_run):
            preview = oracle.preview_oracle_replay_plan(d, project_root=Path("/tmp"))
        keys = preview["targets"]
        self.assertEqual(len(keys), 28)
        self.assertEqual(len(preview["preflight"]), len(keys) * len(seeds))
        self.assertTrue(all(preview["preflight"][(seed, key)] == "REUSE"
                            for seed in seeds for key in keys[:25]))
        self.assertTrue(all(preview["preflight"][(seed, key)] == "RUN"
                            for seed in seeds for key in keys[25:]))
        self.assertEqual(set(preview["parameter_plans"]), {"C96", "C97"})
        self.assertNotEqual(preview["parameter_plans"]["C96"]["producer_kwargs"],
                            preview["parameter_plans"]["C97"]["producer_kwargs"])
        self.assertEqual(tuple(row["target"] for row in comparisons if
                               oracle_parameter_family(row) == "full"), ("T40", "T40_Direct"))


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

    def test_overview_upgrades_only_certified_no_dl_arm_and_leaves_dl_four_seed(self):
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
            self.assertEqual(result["C96"][1], eight)
            self.assertAlmostEqual(result["C96"][0], 9.5)
            self.assertEqual(result["C97"], (6.44, four, "end_to_end_v1"))
            self.assertEqual(result["C107"], (10.45, four, "end_to_end_v1"))

    def test_controls_display_same_names_and_no_oracle_metrics(self):
        from core.audit_registry import oracle_baseline_display_name
        d = definition()
        for arm_id in ("C96", "C97"):
            self.assertEqual(oracle_baseline_display_name(d.source, arm_id), arm_id)
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
            for arm_id in ("C96", "C97"):
                self.assertEqual(rows[arm_id]["target"], arm_id)
                self.assertIsNone(rows[arm_id]["oracle_romd"])
                self.assertIsNone(rows[arm_id]["win_rate_pct"])


if __name__ == "__main__":
    from unittest import main
    main()
