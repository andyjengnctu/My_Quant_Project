"""Generic delta-planning, immutable observation and execution regression cases."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from core.file_integrity import canonical_json_sha256
from core.strategy_compare_policy import (
    get_strategy_comparison_settings, get_strategy_multi_seed_robustness_settings,
)
from filters.breakout_quality.strategy_compare_dl_artifacts import resolve_arm_runtime_dl_source_ids
from services.research.strategy_robustness_units import (
    RobustnessWorkPlan, ScientificUnitStore, scientific_unit_signature,
    model_source_signature, model_observation_provenance_matches,
    pin_files, require_pinned_files,
)


def _registered_fixture():
    settings = get_strategy_comparison_settings("broker_cash_dl_conversion_rolling")
    cfg = get_strategy_multi_seed_robustness_settings(settings.profile_id)
    arms = tuple(settings.arms[x] for x in cfg.stochastic_arm_ids)
    seeds = tuple(int(x) for x in cfg.resolved_seeds)
    contracts = []
    for arm in arms:
        contracts.append({
            "arm_id": arm.arm_id, "param_source": arm.param_source,
            "param_policy": arm.param_policy, "rule_policy": arm.rule_policy,
            "model_seed_sensitive": arm.dl_enabled, "strategy_seed_sensitive": True,
            "seed_pairing": "same_seed", "dl_id": arm.dl_id,
            "dl_runtime_mode": arm.dl_runtime_mode,
            "dl_runtime_options": dict(arm.dl_runtime_options or {}),
            "runtime_dl_sources": [
                {"dl_id": dl_id, "score_source": settings.dl_sources[dl_id].score_source,
                 "point_in_time_training_policy": {"fold_months": 12, "seed": "per_replication"}}
                for dl_id in resolve_arm_runtime_dl_source_ids(settings, arm)
            ] if arm.dl_enabled else [],
        })
    contract = {
        "fingerprint": "fixture", "scientific_contract_version": 1,
        "robustness_id": cfg.robustness_id, "profile_id": settings.profile_id,
        "dataset": settings.dataset, "dataset_identity": {"snapshot": "fixed"},
        "max_positions": settings.max_positions, "rotation": settings.rotation,
        "comparison_period": {"start": "2021-01-01", "end": "2026-03-02"},
        "benchmark_id": cfg.benchmark_id, "resolved_seeds": list(seeds),
        "strategy_trials_per_fold": cfg.strategy_trials_per_fold,
        "training_defaults": {"epochs": 3, "learning_rate": 0.001},
        "stochastic_arms": contracts,
        "benchmark_parameter_artifact_identities": {
            f"{a.arm_id}:seed={s}": {"sha256": "parameter", "family": "min",
                "evaluation_mode": "rolling", "param_policy": "base-finalist-best"}
            for a in arms for s in seeds},
    }
    return settings, cfg, arms, seeds, contract


class StrategyRobustnessUnitCases(unittest.TestCase):
    def test_membership_order_display_and_seed_extensions_preserve_old_units(self):
        _, _, arms, seeds, contract = _registered_fixture()
        old = deepcopy(contract)
        old["stochastic_arms"] = old["stochastic_arms"][:-2]
        expanded = deepcopy(contract)
        expanded.update(fingerprint="other_view", robustness_id="other_title", resolved_seeds=[*seeds, 17])
        expanded["stochastic_arms"].reverse()
        for arm in arms[:-2]:
            for seed in seeds:
                with self.subTest(arm=arm.arm_id, seed=seed):
                    self.assertEqual(
                        scientific_unit_signature(contract=old, arm_id=arm.arm_id, seed=seed),
                        scientific_unit_signature(contract=expanded, arm_id=arm.arm_id, seed=seed),
                    )

    def test_rule_and_parameter_changes_only_invalidate_own_arm(self):
        _, _, arms, seeds, contract = _registered_fixture()
        changed = deepcopy(contract)
        changed["stochastic_arms"][-1]["dl_runtime_options"]["fixture_threshold"] = 0.6
        seed = seeds[0]
        changed["benchmark_parameter_artifact_identities"][f"{arms[0].arm_id}:seed={seed}"]["sha256"] = "new"
        differences = {(a.arm_id, s) for a in arms for s in seeds
                       if scientific_unit_signature(contract=contract, arm_id=a.arm_id, seed=s)
                       != scientific_unit_signature(contract=changed, arm_id=a.arm_id, seed=s)}
        self.assertEqual(differences, {(arms[-1].arm_id, s) for s in seeds} | {(arms[0].arm_id, seed)})

    def test_true_input_changes_cannot_reuse_completed_model_units(self):
        _, _, arms, seeds, contract = _registered_fixture()
        arm = next(a for a in arms if a.dl_enabled)
        original = scientific_unit_signature(contract=contract, arm_id=arm.arm_id, seed=seeds[0])
        for key, value in (("dataset_identity", {"snapshot": "different"}),
                           ("training_defaults", {"epochs": 4}),
                           ("comparison_period", {"start": "2022-01-01", "end": "2026-03-02"})):
            with self.subTest(key=key):
                changed = dict(contract, **{key: value})
                self.assertNotEqual(original, scientific_unit_signature(contract=changed, arm_id=arm.arm_id, seed=seeds[0]))

    def test_model_workspace_depends_on_own_source_not_comparison(self):
        _, _, arms, _, contract = _registered_fixture()
        dl = next(a.dl_id for a in arms if a.dl_enabled)
        changed = deepcopy(contract)
        changed["stochastic_arms"] = changed["stochastic_arms"][:-2]
        changed["fingerprint"] = "new-container"
        changed["resolved_seeds"] = [90]
        self.assertEqual(model_source_signature(contract, dl), model_source_signature(changed, dl))
        changed["training_defaults"]["epochs"] += 1
        self.assertNotEqual(model_source_signature(contract, dl), model_source_signature(changed, dl))

    def test_delta_closure_authorizes_only_new_arms_and_missing_sources(self):
        settings, _, arms, seeds, _ = _registered_fixture()
        units = {(a.arm_id, s) for a in arms for s in seeds}
        old = {(a.arm_id, s) for a in arms[:-2] for s in seeds}
        deps = {a.arm_id: resolve_arm_runtime_dl_source_ids(settings, a) if a.dl_enabled else () for a in arms}
        plan = RobustnessWorkPlan.build(fingerprint="view", units=units, reuse_units=old, dependencies=deps)
        self.assertEqual(plan.replay_units, {(a.arm_id, s) for a in arms[-2:] for s in seeds})
        self.assertEqual(plan.source_units, {(dl, s) for a in arms[-2:] for dl in deps[a.arm_id] for s in seeds})
        for unit in old:
            with self.assertRaises(RuntimeError):
                plan.require_replay(unit)
        with self.assertRaises(RuntimeError):
            plan.require_source(("unrelated", seeds[0]))
        with self.assertRaises(RuntimeError):
            plan.verify_completed(old - {next(iter(old))})
        with self.assertRaises(RuntimeError):
            plan.verify_completed(old | {next(iter(plan.replay_units))})

    def test_receipt_is_atomic_immutable_and_survives_incomplete_outer_run(self):
        signature = {"arm": "fixture", "seed": 42}
        observation = {"row": {"romd": 4.5}, "yearly": [], "source": "incomplete_view"}
        def require_same(old, new):
            if old != new:
                raise ValueError("conflicting observation")
        with tempfile.TemporaryDirectory() as temp:
            store = ScientificUnitStore(temp)
            self.assertIsNone(store.load(signature))
            store.publish(signature=signature, observation=observation, validate_equivalent=require_same)
            path = store.path(signature)
            original = path.read_bytes(), path.stat().st_mtime_ns
            store.publish(signature=signature, observation=observation, validate_equivalent=require_same)
            self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), original)
            self.assertEqual(ScientificUnitStore(temp).load(signature), observation)
            with self.assertRaises(ValueError):
                store.publish(signature=signature, observation={"row": {"romd": 7}}, validate_equivalent=require_same)
            self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), original)
            payload = json.loads(path.read_text())
            payload["body"]["observation"]["row"]["romd"] = 8
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "SHA"):
                store.load(signature)
            with self.assertRaises(ValueError):
                store.publish(signature=signature, observation=observation, validate_equivalent=require_same)

    def test_original_multisource_provenance_not_mutable_latest(self):
        signature = {"arm": {"dl_id": "rank", "runtime_dl_sources": [{"dl_id": "rank"}, {"dl_id": "gate"}]}}
        runtime = {"rank": {"score_sha256": "original-rank"}, "gate": {"score_sha256": "original-gate"}}
        row = {"score_sha256": "original-rank", "runtime_source_identity_sha256": canonical_json_sha256(runtime)}
        attribution = {"runtime_training_identities": runtime, "training_identity": dict(row)}
        self.assertTrue(model_observation_provenance_matches(signature, row, attribution))
        for corrupt in (dict(row, score_sha256="new-latest"), dict(row, runtime_source_identity_sha256="wrong")):
            self.assertFalse(model_observation_provenance_matches(signature, corrupt, attribution))
        broken = deepcopy(attribution)
        broken["runtime_training_identities"].pop("gate")
        self.assertFalse(model_observation_provenance_matches(signature, row, broken))
        broken = deepcopy(attribution)
        broken["runtime_training_identities"]["gate"]["score_sha256"] = "changed"
        self.assertFalse(model_observation_provenance_matches(signature, row, broken))

    def test_pending_shared_source_cannot_drift_from_completed_control(self):
        plan = RobustnessWorkPlan.build(
            fingerprint="view", units={("old", 42), ("new", 42)},
            reuse_units={("old", 42)}, dependencies={"new": ("rank", "gate")},
        )
        identity = {"score_sha256": "original", "fold_count": 6}
        plan = plan.bind_sources([(("rank", 42), identity), (("unrelated", 42), {"score_sha256": "unused"})])
        self.assertEqual(len(plan.source_identities), 1)
        plan.require_source_identity(("rank", 42), identity)
        plan.require_source_identity(("gate", 42), {"score_sha256": "newly-fitted"})
        with self.assertRaises(ValueError):
            plan.require_source_identity(("rank", 42), {"score_sha256": "latest-retrained", "fold_count": 6})
        with self.assertRaises(ValueError):
            plan.bind_sources([(("rank", 42), {"score_sha256": "conflicting-control"})])
        self.assertEqual(plan.reuse_units, {("old", 42)})

    def test_receipt_discovery_uses_original_evidence_and_rejects_corruption(self):
        import gzip
        from services.research import strategy_multi_seed_robustness as robust
        settings, _, arms, seeds, contract = _registered_fixture()
        arm, seed = next(a for a in arms if a.dl_enabled), seeds[0]
        signature = scientific_unit_signature(contract=contract, arm_id=arm.arm_id, seed=seed)
        sources = {x["dl_id"]: {"score_sha256": "original-" + x["dl_id"]}
                   for x in signature["arm"]["runtime_dl_sources"]}
        row = {"arm_id": arm.arm_id, "seed": seed,
               "score_sha256": sources[arm.dl_id]["score_sha256"],
               "runtime_source_identity_sha256": canonical_json_sha256(sources)}
        cfg = SimpleNamespace(reuse_completed=True, benchmark_id=contract["benchmark_id"],
                              model_seed_sensitive_arm_ids=(arm.arm_id,), resolved_seeds=(seed,),
                              output_root="outputs/robustness")
        with tempfile.TemporaryDirectory() as temp, patch.object(robust, "PROJECT_ROOT", Path(temp)):
            source = Path(temp) / "outputs" / "incomplete_comparison"
            destination = robust._attribution_unit_dir(source, arm.arm_id, seed)
            destination.mkdir(parents=True)
            files = {}
            for role in ("trades", "equity", "daily_capacity", "selected_buys", "execution"):
                path = destination / (role + ".csv.gz")
                with gzip.open(path, "wb") as stream:
                    stream.write(b"value\n1\n")
                files[role] = {"path": str(path.relative_to(temp)),
                               "sha256": robust.compute_file_sha256(path), "row_count": 1}
            attr = {"schema_version": robust.ATTRIBUTION_SOURCE_SCHEMA_VERSION,
                    "scientific_fingerprint": "old", "arm_id": arm.arm_id, "seed": seed,
                    "scientific_observation_validation": {"status": "VERIFIED"},
                    "training_identity": row, "runtime_training_identities": sources, "files": files}
            (destination / robust.MANIFEST_FILENAME).write_text(json.dumps(attr))
            observation = {"row": row, "yearly": [{"year": 2021}],
                           "source_run_root": str(source.relative_to(temp)),
                           "source_fingerprint": "old", "source_attribution": attr}
            store = robust._unit_observation_store(cfg)
            store.publish(signature=signature, observation=observation,
                          validate_equivalent=lambda a, b: self.assertEqual(a, b))
            before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in source.rglob("*") if p.is_file()}
            with patch.object(robust, "inspect_reusable_model_research_rolling_score", side_effect=AssertionError("Latest lookup")), \
                 patch.object(robust, "_iter_registered_robustness_scientific_runs", side_effect=AssertionError("legacy scan")):
                result = robust._discover_registered_model_scientific_reuse(
                    settings=settings, robustness=cfg, current_contract=contract, stochastic_arms=(arm,))
                self.assertEqual(set(result), {(arm.arm_id, seed)})
                self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before})
                (destination / "trades.csv.gz").write_bytes(b"corrupted")
                with self.assertRaisesRegex(ValueError, "corrupted"):
                    robust._discover_registered_model_scientific_reuse(
                        settings=settings, robustness=cfg, current_contract=contract, stochastic_arms=(arm,))

    def test_existing_source_workspace_reuses_without_fit_and_corruption_never_rebuilds(self):
        from services.research import strategy_multi_seed_robustness as robust
        settings, _, arms, seeds, _ = _registered_fixture()
        dl = next(a.dl_id for a in arms if a.dl_enabled)
        with tempfile.TemporaryDirectory() as temp:
            legacy = Path(temp) / "legacy"
            legacy.mkdir()
            evidence = legacy / "scores.csv"
            evidence.write_text("original-score-proof")
            job = {"settings": settings, "seed": seeds[0], "model_dir": Path(temp) / "stable",
                   "legacy_model_dir": legacy, "research_dir": Path(temp) / "research", "dl_id": dl,
                   "reuse_completed": True, "comparison_start": "2021-01-01", "comparison_end": "2026-03-02"}
            with patch.object(robust, "run_strategy_compare_training_unit", side_effect=AssertionError("FIT")), \
                 patch.object(robust, "inspect_reusable_model_research_rolling_score", side_effect=AssertionError("Latest")), \
                 patch.object(robust, "_model_prediction_metrics_from_training_artifacts", return_value={}), \
                 patch.object(robust, "validate_strategy_compare_training_artifacts", return_value={"score_sha256": "original"}) as validator:
                result = robust._train_one_unit(dict(job))
                self.assertTrue(result["reused"])
                self.assertEqual(result["artifacts"]["current_training_elapsed_sec"], 0)
                self.assertEqual(validator.call_args.kwargs["model_dir"], legacy.resolve())
                validator.side_effect = ValueError("score identity mismatch")
                with self.assertRaisesRegex(RuntimeError, "no replacement FIT"):
                    robust._train_one_unit(dict(job))
                self.assertEqual(evidence.read_text(), "original-score-proof")

    def test_plan_dependency_revision_is_not_silent_expansion(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "proof.json"
            path.write_text("proof-a")
            pinned = pin_files([path])
            require_pinned_files(pinned)
            # Canonical producers publish through atomic replacement.  A same-size
            # in-place write is not a reliable stat-change fixture on Windows,
            # where timestamp resolution differs across filesystems.
            staged = path.with_name(path.name + ".new")
            staged.write_text("proof-b")
            staged.replace(path)
            with self.assertRaises(RuntimeError):
                require_pinned_files(pinned)

    def test_real_executor_only_submits_new_two_and_does_no_fit_or_baseline_replay(self):
        """Real queue/harvest/guards and CSV writes; expensive producers are spies."""
        from services.research import strategy_multi_seed_robustness as r
        settings, cfg, arms, seeds, contract = _registered_fixture()
        cfg = replace(cfg, cpu_replay_workers=1, gpu_train_workers=1, keep_attribution_source=False)
        old_units = {(a.arm_id, s) for a in arms[:-2] for s in seeds}
        all_units = {(a.arm_id, s) for a in arms for s in seeds}
        dependencies = {a.arm_id: resolve_arm_runtime_dl_source_ids(settings, a) if a.dl_enabled else () for a in arms}
        plan = RobustnessWorkPlan.build(fingerprint=contract["fingerprint"], units=all_units,
                                       reuse_units=old_units, dependencies=dependencies)
        arm_order = {a.arm_id: i for i, a in enumerate(arms, 1)}
        seed_order = {s: i for i, s in enumerate(seeds, 1)}
        old_rows = [{"arm_id": a, "seed": s, "return_over_max_drawdown": 6.0,
                     "arm_order": arm_order[a], "seed_order": seed_order[s]} for a, s in sorted(old_units)]
        old_yearly = [{"arm_id": a, "seed": s, "year": 2021, "return_pct": 3.0, "is_complete_year": True,
                       "arm_order": arm_order[a], "seed_order": seed_order[s]}
                      for a, s in sorted(old_units)]
        current_scores = {(dl, s): {"score": f"score-{dl}-{s}", "manifest": f"manifest-{dl}-{s}",
                           "score_sha256": f"sha-{dl}-{s}", "selected_epoch": 2, "fold_count": 6,
                           "score_execution_start": "2021-01-01", "training_elapsed_sec": 10000,
                           "fold_reuse_summary": {"built": 6}}
                          for dl, s in plan.source_units}
        bindings = {unit: {"path": "params.json", "sha256": "parameter", "manifest_sha256": "manifest",
                           "evaluation_mode": "rolling"} for unit in all_units}
        seen = []
        def replay(job):
            unit = (job["arm_id"], job["seed"])
            self.assertIn(unit, plan.replay_units)
            self.assertTrue(job["replay_active_only"])
            self.assertIsNone(job["baseline_dir"])
            self.assertEqual(job["training_elapsed_sec"], 0)
            self.assertTrue(Path(job["baseline_trade_path"]).is_file())
            self.assertTrue(job["baseline_trade_sha256"])
            seen.append(unit)
            return {"arm_id": unit[0], "seed": unit[1], "return_over_max_drawdown": 9.0,
                    "arm_order": job["arm_order"], "seed_order": job["seed_order"],
                    "yearly": [{"arm_id": unit[0], "seed": unit[1], "year": 2021,
                                "arm_order": job["arm_order"], "seed_order": job["seed_order"],
                                "return_pct": 5.0, "is_complete_year": True}]}
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack, redirect_stdout(io.StringIO()) as out:
            root = Path(temp)
            def mock(name, **kwargs):
                return stack.enter_context(patch.object(r, name, **kwargs))
            mock("PROJECT_ROOT", new=root)
            mock("_run_root", return_value=root / "view")
            mock("_model_work_root", return_value=root / "models" / "suite")
            mock("_reuse_completed_robustness_run", return_value=None)
            mock("_check_resumable_robustness_manifest", return_value=None)
            mock("_materialize_registered_strategy_only_scientific_reuse", return_value={})
            mock("_resolve_existing_scientific_unit_states", return_value=(pd.DataFrame(old_rows), pd.DataFrame(old_yearly),
                 {unit: {"action": r.RESULT_ACTION_REUSE} for unit in old_units}))
            mock("_run_fixed_baselines", return_value={})
            no_fit = mock("_train_one_unit", side_effect=AssertionError("unexpected FIT"))
            def retained_context(**kwargs):
                # A present, verified transient baseline remains REUSE even if
                # new CV arms require its trades for direct selection-R.
                candidate = Path(kwargs["output_dir"])
                candidate.mkdir(parents=True, exist_ok=True)
                (candidate / "no_filter_trades.csv").write_text("trade_id\n", encoding="utf-8")
                return True
            mock("_strategy_only_baseline_context_available", side_effect=retained_context)
            no_baseline = mock("_run_strategy_only_benchmark_unit", side_effect=AssertionError("unexpected baseline context rebuild"))
            no_inspection = mock("inspect_reusable_model_research_rolling_score", side_effect=AssertionError("plan was already resolved"))
            mock("_model_prediction_metrics_from_training_artifacts", return_value={})
            wake_on_completion = mock("wait", wraps=r.wait)
            mock("_replay_one_unit", side_effect=replay)
            def commit(**kwargs):
                frame = r._load_seed_results(kwargs["seed_results_path"])
                self.assertEqual(set(zip(frame.arm_id, frame.seed)), all_units)
                for a, s in old_units:
                    self.assertEqual(float(frame[(frame.arm_id == a) & (frame.seed == s)].iloc[0].return_over_max_drawdown), 6.0)
                return {"fixture_complete": True}
            mock("_commit_robustness_scientific_result", side_effect=commit)
            cleanup = mock("_cleanup_completed_robustness_transients", return_value=None)
            mock("_display_completed_robustness_result", return_value=None)
            result = r._execute_robustness_under_lock(
                cfg=cfg, settings=settings, status={}, contract=contract, fixed_arms=(), stochastic_arms=arms,
                model_arms=tuple(a for a in arms if a.dl_enabled), strategy_only_arms=tuple(a for a in arms if not a.dl_enabled),
                registered_scientific_reuse={}, benchmark_bindings=bindings,
                comparison_start="2021-01-01", comparison_end="2026-03-02", work_plan=plan,
                planned_model_artifacts=current_scores,
            )
            self.assertTrue(result["fixture_complete"])
            self.assertEqual(set(seen), plan.replay_units)
            self.assertEqual(len(seen), len(plan.replay_units))
            no_fit.assert_not_called(); no_baseline.assert_not_called(); no_inspection.assert_not_called()
            self.assertTrue(wake_on_completion.called, "replay must wait for completion, not a fixed polling sleep")
            self.assertEqual(cleanup.call_args.kwargs["training_cleanup_targets"], [])
            self.assertNotIn("built:6", out.getvalue())
            self.assertNotIn("elapsed=", out.getvalue())


    def test_active_only_selection_r_uses_verified_external_baseline_trades(self):
        """Same metric owner, no fake baseline inside the model-only directory."""
        from services.research import strategy_comparison as compare
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pair_dir = root / "active_only"
            baseline_dir = root / "baseline"
            pair_dir.mkdir(); baseline_dir.mkdir()
            (baseline_dir / "no_filter_trades.csv").write_text("dummy\n", encoding="utf-8")
            (pair_dir / "score_ranking_trades.csv").write_text("dummy\n", encoding="utf-8")
            def rounds(_frame, *, scenario):
                if scenario == "no_filter":
                    return pd.DataFrame({"match_key": ["A", "B"], "r_multiple": [1.0, 2.0]})
                return pd.DataFrame({"match_key": ["B", "C"], "r_multiple": [2.0, 4.0]})
            with patch.object(compare, "reconstruct_round_trips", side_effect=rounds):
                with self.assertRaisesRegex(FileNotFoundError, "no_filter_trades"):
                    compare._load_direct_selection_r(
                        pair_dir, root=root,
                        active_trades_filename="score_ranking_trades.csv",
                    )
                selection_r = compare._load_direct_selection_r(
                    pair_dir, root=root,
                    active_trades_filename="score_ranking_trades.csv",
                    baseline_trades_path=baseline_dir / "no_filter_trades.csv",
                )
                self.assertEqual(selection_r, 3.0)  # exclusive C +4 minus A +1

    def test_empty_trade_csv_is_legal_only_for_verified_zero_trades(self):
        from services.research import strategy_comparison as compare
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            baseline = root / "baseline" / "no_filter_trades.csv"
            active_dir = root / "active"
            baseline.parent.mkdir()
            active_dir.mkdir()
            # The canonical engine writes a BOM plus a newline for a zero-row,
            # zero-column pandas trade history (not literally zero bytes).
            baseline.write_bytes(b"\xef\xbb\xbf\n")
            (active_dir / "score_ranking_trades.csv").write_bytes(b"\xef\xbb\xbf\n")
            self.assertEqual(compare._load_direct_selection_r(
                active_dir, root=root, active_trades_filename="score_ranking_trades.csv",
                baseline_trades_path=baseline,
                baseline_trade_count=0, active_trade_count=0,
            ), 0.0)
            with self.assertRaisesRegex(ValueError, "零交易證據"):
                compare._load_direct_selection_r(
                    active_dir, root=root, active_trades_filename="score_ranking_trades.csv",
                    baseline_trades_path=baseline,
                    baseline_trade_count=1, active_trade_count=0,
                )
            with self.assertRaisesRegex(ValueError, "零交易證據"):
                compare._load_direct_selection_r(
                    active_dir, root=root, active_trades_filename="score_ranking_trades.csv",
                    baseline_trades_path=baseline,
                    baseline_trade_count=0, active_trade_count=5,
                )
            with self.assertRaisesRegex(ValueError, "零交易證據"):
                compare._load_direct_selection_r(
                    active_dir, root=root, active_trades_filename="score_ranking_trades.csv",
                    baseline_trades_path=baseline,
                )

    def test_active_only_rejects_changed_same_seed_baseline_bytes(self):
        from services.research import strategy_multi_seed_robustness as robust
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            baseline = root / "baseline" / "no_filter_trades.csv"
            baseline.parent.mkdir()
            baseline.write_text("trade_id\n", encoding="utf-8")
            job = {
                "profile_id": "broker_cash_dl_conversion_rolling",
                "arm_id": "C140", "seed": 693545351,
                "arm_order": 32, "seed_order": 1,
                "comparison_start": "2021-01-01", "comparison_end": "2026-03-02",
                "pair_dir": str(root / "replay"),
                "result_path": str(root / "result.json"),
                "baseline_trade_path": str(baseline),
                "baseline_trade_sha256": "0" * 64,
                "replay_active_only": True,
                "keep_attribution_source": False,
                "keep_replay_details": True,
                "params_path": "unused.json", "score_overrides": {},
            }
            with (patch.object(robust, "PROJECT_ROOT", new=root),
                  patch.object(robust, "run_strategy_compare_active_arm", return_value={"score_ranking": {}}),
                  patch.object(robust, "_load_direct_selection_r", side_effect=AssertionError("unverified baseline must not be read"))):
                with self.assertRaisesRegex(RuntimeError, "SHA"):
                    robust._replay_one_unit(job)
            self.assertFalse((root / "result.json").exists())

    def test_signed_baseline_context_rebuild_is_not_scientific_replay(self):
        from services.research import strategy_multi_seed_robustness as robust
        settings, cfg, _, seeds, _ = _registered_fixture()
        seed = seeds[0]
        baseline, model = settings.arms["C139"], settings.arms["C140"]
        plan = RobustnessWorkPlan.build(
            fingerprint="fixture", units={(baseline.arm_id, seed), (model.arm_id, seed)},
            reuse_units={(baseline.arm_id, seed)}, dependencies={model.arm_id: (model.dl_id,)},
        )
        saved = {"arm_id": baseline.arm_id, "seed": seed, "return_over_max_drawdown": 2.29,
                 "selected_epoch": 0, "fold_count": None}
        years = [{"arm_id": baseline.arm_id, "seed": seed, "year": 2021,
                  "return_pct": 4.0, "is_complete_year": True}]
        calls = []
        def rebuild(**kwargs):
            calls.append(kwargs)
            path = Path(kwargs["output_dir"])
            path.mkdir(parents=True, exist_ok=True)
            (path / "no_filter_trades.csv").write_text("trade_id\n", encoding="utf-8")
            return {**saved, "yearly": [{"year": 2021, "return_pct": 4.0,
                                        "is_complete_year": True}]}
        with tempfile.TemporaryDirectory() as temp:
            with (patch.object(robust, "_strategy_only_baseline_context_available", return_value=False),
                  patch.object(robust, "_run_strategy_only_benchmark_unit", side_effect=rebuild)):
                paths, actions, rows, yearly = robust._prepare_strategy_only_baseline_units(
                    cfg=replace(cfg, cpu_replay_workers=1), settings=settings,
                    run_root=Path(temp), stochastic_arms=(baseline, model),
                    model_arms=(model,), strategy_only_arms=(baseline,),
                    seeds=(seed,), benchmark_bindings={(baseline.arm_id, seed): {
                        "path": "unused.json", "sha256": "param", "manifest_sha256": "manifest",
                        "evaluation_mode": "rolling", "parameter_payload_ready": True,
                    }}, comparison_start="2021-01-01", comparison_end="2026-03-02",
                    scientific_completed={(baseline.arm_id, seed)},
                    completed={(baseline.arm_id, seed)},
                    rows=[dict(saved)], yearly_rows=list(years), work_plan=plan,
                )
            self.assertEqual(actions[(baseline.arm_id, seed)], "REBUILD_CONTEXT")
            self.assertEqual(len(calls), 1)
            self.assertEqual(rows, [saved], "signed seed observation cannot be overwritten")
            self.assertEqual(yearly, years)
            self.assertTrue((Path(paths[(baseline.arm_id, seed)]) / "no_filter_trades.csv").exists())

    def test_real_robustness_registry_maps_every_model_to_one_same_param_baseline(self):
        from services.research import strategy_multi_seed_robustness as robust
        settings, cfg, arms, _, _ = _registered_fixture()
        baselines = tuple(arm for arm in arms if not arm.dl_enabled)
        for model in (arm for arm in arms if arm.dl_enabled):
            with self.subTest(arm=model.arm_id):
                baseline = robust._resolve_same_seed_baseline_arm(
                    settings=settings, strategy_only_arms=baselines, model_arm=model,
                )
                self.assertEqual(baseline.param_source, model.param_source)
                self.assertEqual(baseline.rule_policy, model.rule_policy)

    def test_active_only_matches_paired_real_broker_cash_replay(self):
        """Run the real engine; only fixture score/parameter I/O is injected."""
        from services.research import strategy_compare_engine as engine
        from core.strategy_params import V16StrategyParams
        from filters.breakout_quality.ranking_score_store import SCORE_SOURCE_SELECTION_POINT_IN_TIME
        from filters.breakout_quality.strategy_compare_contracts import comparison_labels
        from tools.validate.synthetic_portfolio_common import build_synthetic_competing_candidates_case
        from tools.validate.synthetic_frame_utils import write_synthetic_csv_bundle

        settings, _, arms, _, _ = _registered_fixture()
        positive_arms = [arm for arm in arms if arm.dl_enabled
                         and (arm.dl_runtime_options or {}).get("positive_score_column")]
        self.assertGreaterEqual(len(positive_arms), 2)
        # Registered same-source Positive arms exercise both selector modes.
        fixture = build_synthetic_competing_candidates_case(V16StrategyParams())
        dates = sorted({str(day)[:10] for frame in fixture["frames"].values() for day in frame["Date"]})
        start, end = dates[0], dates[-1]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "prices"
            data.mkdir()
            write_synthetic_csv_bundle(str(data), fixture["frames"])
            params_path = root / "parameters.json"
            params_path.write_text("{}")
            score_file = root / "scores.csv"
            score_file.write_text("fixture")
            for arm in positive_arms[-2:]:
                source = settings.dl_sources[arm.dl_id]
                options = dict(arm.dl_runtime_options or {})
                secondary_id = options['positive_dl_id']
                secondary = settings.dl_sources[secondary_id]
                self.assertEqual(resolve_arm_runtime_dl_source_ids(settings, arm), (arm.dl_id, secondary_id))
                options.update(positive_filter_id=secondary.filter_id,
                    positive_score_source=secondary.score_source,
                    positive_model_architecture=secondary.model_architecture,
                    positive_experiment_profile=secondary.experiment_profile)
                for gate_allows in (True, False):
                    with self.subTest(arm=arm.arm_id, gate_allows=gate_allows):
                        pit = SimpleNamespace(available_from=start, available_through=end,
                                              manifest_path=score_file, score_path=score_file,
                                              experiment_profile=source.experiment_profile,
                                              audit_path=None, model_validation_gate={}, manifest={})
                        runtime = SimpleNamespace(
                            comparison_mode=engine.COMPARISON_MODE_SCORE_RANKING,
                            ranking_policy=arm.dl_runtime_mode, ranking_options=options,
                            optional_entry_filter_policy=engine.OPTIONAL_ENTRY_FILTER_POLICY_CURRENT,
                            score_source=SCORE_SOURCE_SELECTION_POINT_IN_TIME,
                            filter_id=source.filter_id, model_architecture=source.model_architecture,
                            experiment_profile=source.experiment_profile, configured_threshold=0.5,
                            labels=comparison_labels(engine.COMPARISON_MODE_SCORE_RANKING),
                            runtime_contract=None, pit_contract=pit, continuous_contract=None,
                            continuous_score_override=None,
                            ranking_source={"score_source": source.score_source,
                                "model_architecture": source.model_architecture,
                                "experiment_profile": source.experiment_profile,
                                "ranking_policy": arm.dl_runtime_mode, "ranking_options": options},
                            filter_source=None, binary_filter_manifest=None,
                            binary_filter_manifest_path=None, binary_filter_scores_path=None,
                            effective_score_source=source.score_source,
                            manifest_architecture=source.model_architecture,
                            manifest_profile=source.experiment_profile, start_date=start, end_date=end,
                        )
                        def score(**kwargs):
                            positive = kwargs.get("score_column") == "positive_p_positive"
                            value = (0.8 if gate_allows else 0.1) if positive else (
                                0.9 if str(kwargs["ticker"]) == sorted(fixture["frames"])[0] else 0.3)
                            return {"score": value, "available": True,
                                    "score_date": pd.Timestamp(kwargs["signal_date"]).strftime("%Y-%m-%d"),
                                    "score_source": SCORE_SOURCE_SELECTION_POINT_IN_TIME}
                        calls = []
                        real_scenario = engine._run_scenario
                        def observe(**kwargs):
                            calls.append(kwargs["name"])
                            return real_scenario(**kwargs)
                        with (
                            patch.object(engine, "resolve_strategy_compare_runtime_contract", return_value=runtime),
                            patch.object(engine, "load_strategy_param_evaluation_view", return_value=(
                                {"kind": "single_param", "params": fixture["params"]}, "fixture-param")),
                            patch.object(engine, "get_dataset_dir", return_value=str(data)),
                            patch.object(engine, "_run_scenario", side_effect=observe),
                            patch("filters.breakout_quality.runtime.lookup_selection_point_in_time_candidate_score", side_effect=score),
                            patch("filters.breakout_quality.ranking_score_store.load_selection_point_in_time_score_table_from_path",
                                  return_value=pd.DataFrame(index=pd.MultiIndex.from_tuples(
                                      [(ticker, str(day)[:10]) for ticker, frame in fixture["frames"].items()
                                       for day in frame["Date"]], names=["ticker", "date"]))),
                            patch.object(engine, "materialize_strategy_pair_readable_report"),
                        ):
                            paired_dir = root / f"paired_{arm.arm_id}_{gate_allows}"
                            active_dir = root / f"active_{arm.arm_id}_{gate_allows}"
                            common = dict(project_root=root, params_path=str(params_path),
                                          entry_strategy=arm.entry_strategy,
                                          allow_static_diagnostic=True, quiet=True,
                                          max_positions=fixture["max_positions"],
                                          enable_rotation=fixture["enable_rotation"],
                                          capture_execution_diagnostics=True,
                                          capture_selection_target_diagnostics=False)
                            paired = engine.run_comparison(**common, output_dir_override=paired_dir)
                            active = engine.run_comparison(**common, output_dir_override=active_dir, replay_active_only=True)
                        self.assertEqual(calls, ["no_filter", "score_ranking", "score_ranking"])
                        # Wall-clock telemetry is not an economic result (F9).
                        economic = lambda values: {k: v for k, v in values.items()
                                                   if not k.startswith("resource_aware_selector_timing_")}
                        self.assertEqual(economic(active["score_ranking"]), economic(paired["score_ranking"]))
                        annual_keys = ("year", "score_ranking_return_pct", "is_full_year", "start_date", "end_date")
                        self.assertEqual([{k: row[k] for k in annual_keys} for row in active["yearly"]],
                                         [{k: row[k] for k in annual_keys} for row in paired["yearly"]])
                        for filename in ("score_ranking_trades.csv", "score_ranking_equity.csv",
                                         "score_ranking_daily_capacity.csv", "score_ranking_selected_buys.csv",
                                         "score_ranking_execution.csv"):
                            if filename == "score_ranking_daily_capacity.csv":
                                left = pd.read_csv(active_dir / filename)
                                right = pd.read_csv(paired_dir / filename)
                                telemetry = ["Resource_Aware_Selector_Elapsed_Ns"]
                                pd.testing.assert_frame_equal(left.drop(columns=telemetry, errors="ignore"),
                                                              right.drop(columns=telemetry, errors="ignore"))
                            else:
                                self.assertEqual((active_dir / filename).read_bytes(), (paired_dir / filename).read_bytes(), filename)
                        self.assertFalse((active_dir / "no_filter_trades.csv").exists())
                        self.assertFalse((active_dir / "strategy_comparison.json").exists())
                        self.assertTrue((active_dir / "strategy_observation.json").exists())
                        # The real pair and Active-only observation must yield
                        # identical selection R from the same base/active trades.
                        from services.research.strategy_comparison import _load_direct_selection_r
                        self.assertAlmostEqual(
                            _load_direct_selection_r(
                                paired_dir, root=root,
                                active_trades_filename="score_ranking_trades.csv",
                                baseline_trade_count=paired["no_filter"]["trade_count"],
                                active_trade_count=paired["score_ranking"]["trade_count"],
                            ),
                            _load_direct_selection_r(
                                active_dir, root=root,
                                active_trades_filename="score_ranking_trades.csv",
                                baseline_trades_path=paired_dir / "no_filter_trades.csv",
                                baseline_trade_count=paired["no_filter"]["trade_count"],
                                active_trade_count=active["score_ranking"]["trade_count"],
                            ), places=10,
                        )



    def test_historical_score_revisions_do_not_invalidate_unmodified_arms(self):
        """Real legacy discovery must not mix separate signed score versions."""
        from services.research import strategy_multi_seed_robustness as robust
        settings, cfg, arms, seeds, contract = _registered_fixture()
        arm = next(a for a in arms if a.dl_enabled and
                   len(resolve_arm_runtime_dl_source_ids(settings, a)) == 1)
        dl_id = resolve_arm_runtime_dl_source_ids(settings, arm)[0]
        seed = seeds[0]
        cfg = SimpleNamespace(
            reuse_completed=True, benchmark_id=cfg.benchmark_id,
            model_seed_sensitive_arm_ids=(arm.arm_id,),
            resolved_seeds=(seed,), output_root="outputs/test_score_revision",
        )
        current = {"selected_epoch": 2, "fold_count": 6,
                   "model_sha256": "model-current", "score_sha256": "score-current",
                   "score_available_from": "2021-01-01",
                   "score_available_through": "2026-03-02"}
        old = dict(current, model_sha256="model-old", score_sha256="score-old")
        sources = []
        attributions = {}
        for label, artifact in (("current", current), ("old", old)):
            root = Path("/tmp") / ("signed_robustness_" + label)
            contract_copy = deepcopy(contract)
            contract_copy["fingerprint"] = label
            runtime_sources = {dl_id: artifact}
            row = {
                "arm_id": arm.arm_id, "seed": seed,
                "score_sha256": artifact["score_sha256"],
                "runtime_source_identity_sha256": canonical_json_sha256(runtime_sources),
                "strategy_param_sha256": "parameter",
                "return_over_max_drawdown": 10.0,
            }
            attributions[str(root)] = {
                "runtime_training_identities": runtime_sources,
                "training_identity": dict(row),
                "files": {"trades": {"sha256": label}},
            }
            sources.append((root, {
                "contract": contract_copy,
                "seed_frame": pd.DataFrame([row]),
                "seed_yearly_frame": pd.DataFrame([{
                    "arm_id": arm.arm_id, "seed": seed, "year": 2021,
                    "return_pct": 2.0, "is_complete_year": True,
                }]),
                "evidence": {"artifacts": {}},
            }))
        unit = (arm.arm_id, seed)
        for source_order in (sources, list(reversed(sources))):
            with self.subTest(first_source=source_order[0][0].name):
                with (
                    patch.object(robust, "_iter_registered_robustness_scientific_runs",
                                 return_value=iter(source_order)),
                    patch.object(robust, "_read_attribution_unit_manifest",
                                 side_effect=lambda root, **kw: attributions[str(root)]),
                    patch.object(robust, "inspect_reusable_model_research_rolling_score",
                                 return_value=(current, "READY")),
                ):
                    found = robust._discover_registered_model_scientific_reuse(
                        settings=settings, robustness=cfg,
                        current_contract=contract, stochastic_arms=(arm,),
                    )
                self.assertEqual(found[unit]["row"]["score_sha256"], "score-current")
        # Ambiguous historical provenance without a verified tie-break is not REUSE.
        with (
            patch.object(robust, "_iter_registered_robustness_scientific_runs",
                         return_value=iter(sources)),
            patch.object(robust, "_read_attribution_unit_manifest",
                         side_effect=lambda root, **kw: attributions[str(root)]),
            patch.object(robust, "inspect_reusable_model_research_rolling_score",
                         return_value=(None, "NOT_READY")),
        ):
            with self.assertRaisesRegex(ValueError, "Ambiguous historical model score provenance"):
                robust._discover_registered_model_scientific_reuse(
                    settings=settings, robustness=cfg,
                    current_contract=contract, stochastic_arms=(arm,),
                )
        # With only one verified source, no newest-model lookup or FIT is needed.
        with (
            patch.object(robust, "_iter_registered_robustness_scientific_runs",
                         return_value=iter(sources[:1])),
            patch.object(robust, "_read_attribution_unit_manifest",
                         side_effect=lambda root, **kw: attributions[str(root)]),
            patch.object(robust, "inspect_reusable_model_research_rolling_score",
                         side_effect=AssertionError("model looked up for completed result")),
        ):
            found = robust._discover_registered_model_scientific_reuse(
                settings=settings, robustness=cfg,
                current_contract=contract, stochastic_arms=(arm,),
            )
        self.assertEqual(found[unit]["row"]["score_sha256"], "score-current")

    def test_positive_dual_source_collision_requires_matching_both_scores(self):
        """An equal-rank score cannot reuse an unrelated Positive classifier result."""
        from services.research import strategy_multi_seed_robustness as robust
        settings, cfg, arms, seeds, contract = _registered_fixture()
        arm = next(a for a in arms if a.dl_enabled and
                   (a.dl_runtime_options or {}).get("positive_score_column"))
        dl_ids = resolve_arm_runtime_dl_source_ids(settings, arm)
        self.assertEqual(dl_ids, (arm.dl_id, arm.dl_runtime_options['positive_dl_id']))
        seed = seeds[0]
        cfg = SimpleNamespace(
            reuse_completed=True, benchmark_id=cfg.benchmark_id,
            model_seed_sensitive_arm_ids=(arm.arm_id,), resolved_seeds=(seed,),
            output_root="outputs/test_inline_positive_score_revision",
        )
        rank = {"selected_epoch": 2, "fold_count": 6, "model_sha256": "same-rank-model",
                "score_sha256": "same-rank-score", "score_available_from": "2021-01-01",
                "score_available_through": "2026-03-02"}
        classifier_current = dict(rank, score_sha256='binary-classifier-current')
        classifier_previous = dict(rank, score_sha256='binary-classifier-previous')
        sources = []
        attributions = {}
        for label, classified in (("current", classifier_current), ("old", classifier_previous)):
            root = Path("/tmp") / ("signed_inline_" + label)
            source_contract = deepcopy(contract)
            source_contract["fingerprint"] = label
            runtime_sources = {dl_ids[0]: rank, dl_ids[1]: classified}
            row = {"arm_id": arm.arm_id, "seed": seed,
                   "score_sha256": rank["score_sha256"],
                   "runtime_source_identity_sha256": canonical_json_sha256(runtime_sources)}
            attributions[str(root)] = {"runtime_training_identities": runtime_sources,
                                       "training_identity": dict(row),
                                       "files": {"trades": {"sha256": label}}}
            sources.append((root, {"contract": source_contract,
                                   "seed_frame": pd.DataFrame([row]),
                                   "seed_yearly_frame": pd.DataFrame([{
                                       "arm_id": arm.arm_id, "seed": seed, "year": 2021,
                                       "return_pct": 2.0, "is_complete_year": True,
                                   }]), "evidence": {"artifacts": {}}}))
        def canonical_score(*, source, **_kw):
            return (rank if source.experiment_profile == settings.dl_sources[dl_ids[0]].experiment_profile
                    else classifier_current, "READY")
        with (
            patch.object(robust, "_iter_registered_robustness_scientific_runs",
                         return_value=iter(reversed(sources))),
            patch.object(robust, "_read_attribution_unit_manifest",
                         side_effect=lambda root, **kw: attributions[str(root)]),
            patch.object(robust, "inspect_reusable_model_research_rolling_score",
                         side_effect=canonical_score),
        ):
            found = robust._discover_registered_model_scientific_reuse(
                settings=settings, robustness=cfg,
                current_contract=contract, stochastic_arms=(arm,),
            )
        self.assertEqual(found[(arm.arm_id, seed)]["source_fingerprint"], "current")

    def test_same_provenance_with_different_economic_results_still_blocks(self):
        """Only different score versions are independent, not contradictory PnL."""
        from services.research import strategy_multi_seed_robustness as robust
        before = {"row": {"score_sha256": "same", "runtime_source_identity_sha256": "same",
                          "strategy_param_sha256": "same", "return_over_max_drawdown": 3.0},
                  "yearly": [{"year": 2021, "return_pct": 10.0, "is_complete_year": True}]}
        after = deepcopy(before)
        after["row"]["return_over_max_drawdown"] = 4.0
        with self.assertRaisesRegex(RuntimeError, "scientific observation"):
            robust._observation_equivalence(before, after)


def append_strategy_robustness_unit_checks(*, results, case_id):
    """Join the existing Strategy Compare capability case, not a new gate."""
    from tools.validate.check_result_utils import add_check
    for method in unittest.defaultTestLoader.getTestCaseNames(StrategyRobustnessUnitCases):
        result = unittest.TestResult()
        StrategyRobustnessUnitCases(method).run(result)
        notes = [detail for _test, detail in (*result.failures, *result.errors)]
        add_check(results, "synthetic_breakout_quality", case_id,
                  "immutable_unit_" + method.removeprefix("test_"), True,
                  result.wasSuccessful(), note="\n".join(notes))


if __name__ == "__main__":
    unittest.main()
