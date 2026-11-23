"""Regression checks for destructive boundaries of the standalone cleaner."""
import contextlib
import csv
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import py_compile
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from apps import cleaner
from core.compare_csv_storage import copy_reused_detail, publish_detail_csv
from core.regression_fixture_access import fixture_access
from services import cleanup_rules as rules, project_cleaner as service
from services.research.strategy_compare_pair_integrity import (
    build_completed_pair_file_integrity, completed_strategy_run_publication_matches,
)


class CleanupTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.write("doc/PROJECT_SETTINGS.md", b"test")
        self.git("init", "-q")
        self.base = Path("outputs/strategy_compare/preview/rolling/runs")
        self.output = "score_ranking_orderable_candidates.csv"
        self.payload = b"name,value\n" + b"test,123\n" * 1000

    def git(self, *args):
        subprocess.run(["git", "-C", str(self.root), *args], capture_output=True, check=True)

    def write(self, rel, data):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data if isinstance(data, bytes) else json.dumps(data).encode())
        return path

    def publish(self, run, *, pinned=False, data=None):
        rel = self.base / run
        pair = rel / "pairs/arm"
        value = {"metadata": {"name": "arm"}}
        self.write(pair / "strategy_comparison.json", value)
        self.write(pair / "strategy_comparison.md", b"md")
        p = self.write(pair / self.output, self.payload if data is None else data)
        os.utime(p, ns=(1710000000000000000, 1710000000000000000))
        info = {"status": "COMPLETED", "config_fingerprint": "0123456789ab", "pairs": {"arm": value}}
        manifest = {"config_fingerprint": info["config_fingerprint"], "run_dir": rel.as_posix()}
        if pinned:
            record = build_completed_pair_file_integrity({"arm": p.parent})
            info["pair_file_integrity"] = manifest["pair_file_integrity"] = record
        self.write(rel / "strategy_comparison.json", info)
        self.write(rel / "manifest.json", manifest)
        return p

    def menu(self, answers):
        text = io.StringIO()
        with patch.object(cleaner, "ROOT", self.root), patch("builtins.input", side_effect=answers), contextlib.redirect_stdout(text):
            cleaner.main()
        return text.getvalue()

    def summary(self, mode):
        with ZipFile(self.root / "outputs/project_cleanup" / f"last_{mode}.zip") as result:
            self.assertIsNone(result.testzip())
            return json.loads(result.read("summary.json")), list(csv.DictReader(
                io.StringIO(result.read("decisions.csv").decode("utf-8-sig"))))

    def scanned_cleaner(self):
        rows = []
        worker = service.Cleaner(self.root, SimpleNamespace(row=lambda *args, **kwargs: rows.append(args)), True)
        with contextlib.redirect_stdout(io.StringIO()):
            worker.scan()
        return worker, rows

    def migrated_parameter(self, name="run_best_params.json"):
        from core.file_integrity import compute_file_sha256
        from core.strategy_param_artifacts import resolve_strategy_param_manifest_path
        source = self.write(Path("models") / name, {"params": {"value": 1}})
        target = self.write(Path("models/strategy_params/canonical") / name, source.read_bytes())
        state = {"run_best_params.json": "active", "candidate_best_params.json": "candidate_best",
                 "candidate_retention_best_params.json": "candidate_retention_best",
                 "candidate_val_score_best_params.json": "candidate_val_score_best"}[name]
        self.write(resolve_strategy_param_manifest_path(self.root, family="full", evaluation_mode="trade"), {
            "producer": "optimizer", "family": "full", "evaluation_mode": "trade",
            "state_artifacts": {state: {"path": target.relative_to(self.root).as_posix(),
                                         "sha256": compute_file_sha256(target)}},
        })
        return source, target

    def diagnostic_projection(self, rel):
        import numpy as np
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = ("identity", "ticker", "dates", "labels", "raw_r", "valid", "payload_sha256")
        values = {key: np.array([0]) for key in fields}
        values.update(identity=np.asarray("a" * 64), payload_sha256=np.asarray("b" * 64))
        np.savez(path, **values)
        return path

    def test_current_parameter_overrides_survive_migration_cleanup(self):
        from core import model_paths
        from services.optimizer.strategy_param_service import collect_legacy_root_strategy_parameter_cleanup_plan
        cases = (
            ("run_best_params.json", model_paths.RUN_BEST_PARAMS_PATH_ENV_VAR, model_paths.resolve_run_best_params_path),
            ("candidate_best_params.json", model_paths.CANDIDATE_BEST_PARAMS_PATH_ENV_VAR, model_paths.resolve_candidate_best_params_path),
            ("candidate_retention_best_params.json", model_paths.CANDIDATE_RETENTION_BEST_PARAMS_PATH_ENV_VAR, model_paths.resolve_candidate_retention_best_params_path),
            ("candidate_val_score_best_params.json", model_paths.CANDIDATE_VAL_SCORE_BEST_PARAMS_PATH_ENV_VAR, model_paths.resolve_candidate_val_score_best_params_path),
        )
        for name, variable, resolver in cases:
            with self.subTest(name=name):
                source, target = self.migrated_parameter(name)
                self.assertEqual(collect_legacy_root_strategy_parameter_cleanup_plan(self.root)["status"], "READY")
                with patch.dict(os.environ, {variable: str(source)}):
                    self.menu(["2", "0"])
                    self.assertEqual(Path(resolver(str(self.root))).read_bytes(), target.read_bytes())
                self.menu(["2", "0"])
                self.assertFalse(source.exists())

    def test_duplicate_link_groups_merge_without_changing_reuse_bytes(self):
        from core.compare_csv_storage import disk_bytes
        a, b, c, d = [self.publish(name, pinned=True) for name in ("a", "b", "c", "d")]
        b.unlink(); os.link(a, b)
        d.unlink(); os.link(c, d)
        saving = disk_bytes(c, c.stat())
        self.menu(["1", "0"])
        self.assertEqual(sum(self.summary("preview")[0]["saved"].values()), saving)
        self.assertFalse(a.samefile(c))
        self.menu(["2", "0"])
        self.assertTrue(all(a.samefile(p) for p in (b, c, d)))
        self.assertEqual(sum(self.summary("apply")[0]["saved"].values()), saving)
        for p in (a, b, c, d):
            run = p.parents[2]
            self.assertTrue(completed_strategy_run_publication_matches(
                run_dir=run, project_root=self.root,
                run_payload=json.loads((run / "strategy_comparison.json").read_text())))
        self.menu(["2", "0"])
        self.assertEqual(sum(self.summary("apply")[0]["saved"].values()), 0)

    def test_active_strategy_template_protects_migrated_parameter(self):
        from dataclasses import replace
        from core import strategy_compare_policy as policy
        from filters.breakout_quality.strategy_compare_sources import resolve_param_source_path
        source, target = self.migrated_parameter()
        settings = policy.get_strategy_comparison_settings()
        arm = settings.enabled_arms[0]
        sources = dict(settings.parameter_sources)
        sources[arm.param_source] = replace(sources[arm.param_source],
            canonical_family=None, canonical_evaluation_mode=None,
            path_template=source.relative_to(self.root).as_posix())
        settings = replace(settings, parameter_sources=sources)
        with patch.object(policy, "get_strategy_comparison_settings", return_value=settings):
            self.assertEqual(resolve_param_source_path(self.root, settings, arm.param_source), source)
            self.menu(["2", "0"])
            self.assertEqual(source.read_bytes(), target.read_bytes())
        self.menu(["2", "0"])
        self.assertFalse(source.exists())

    def test_transitive_parameter_and_model_dependencies_are_retained(self):
        from dataclasses import replace
        from core import strategy_compare_policy as policy
        from core.strategy_comparison import StrategyArtifactBuilder
        from filters.breakout_quality.strategy_compare_dl_artifacts import resolve_required_artifact_sources
        source, target = self.migrated_parameter()
        slots = rules.retired_diagnostics(self.root)
        retired_slot, retired_profile = next(iter(slots.items()))
        cache = self.diagnostic_projection(retired_slot)
        settings = policy.get_strategy_comparison_settings()
        arm = settings.enabled_arms[0]
        sources = dict(settings.parameter_sources)
        models = dict(settings.dl_sources)
        models["CLEANUP_TEACHER"] = replace(next(iter(models.values())),
            dl_id="CLEANUP_TEACHER", experiment_profile=retired_profile)
        sources["CLEANUP_UPSTREAM"] = replace(sources[arm.param_source],
            source_id="CLEANUP_UPSTREAM", builder=None, trained_with_dl_id="CLEANUP_TEACHER",
            canonical_family=None, canonical_evaluation_mode=None,
            path_template=source.relative_to(self.root).as_posix())
        sources["CLEANUP_MIDDLE"] = replace(sources[arm.param_source], source_id="CLEANUP_MIDDLE",
            builder=StrategyArtifactBuilder(builder_type="frozen_oos_from_rolling", enabled=True,
                                             options={"source_param_source_id": "CLEANUP_UPSTREAM"}))
        sources[arm.param_source] = replace(sources[arm.param_source],
            builder=StrategyArtifactBuilder(builder_type="frozen_oos_from_rolling", enabled=True,
                                             options={"source_param_source_id": "CLEANUP_MIDDLE"}))
        settings = replace(settings, parameter_sources=sources, dl_sources=models)
        with patch.object(policy, "get_strategy_comparison_settings", return_value=settings):
            params, dl, _ = resolve_required_artifact_sources(settings)
            self.assertIn("CLEANUP_UPSTREAM", params)
            self.assertIn("CLEANUP_TEACHER", dl)
            self.assertFalse(retired_slot in rules.retired_diagnostics(self.root))
            self.menu(["2", "0"])
            self.assertEqual(source.read_bytes(), target.read_bytes())
            self.assertTrue(cache.is_file())
        self.menu(["2", "0"])
        self.assertFalse(source.exists())
        self.assertFalse(cache.exists())

    def test_partial_group_merge_can_resume_without_overstating_savings(self):
        a, b, c, d = [self.publish(name, pinned=True) for name in ("a", "b", "c", "d")]
        b.unlink(); os.link(a, b)
        d.unlink(); os.link(c, d)
        original_replace = os.replace
        def fail_last(src, dst):
            if Path(dst) == d:
                raise PermissionError("injected final alias busy")
            return original_replace(src, dst)
        with patch("services.project_cleaner.os.replace", side_effect=fail_last):
            self.menu(["2", "0"])
        self.assertTrue(a.samefile(c))
        self.assertFalse(a.samefile(d))
        self.assertTrue(all(p.read_bytes() == self.payload for p in (a, b, c, d)))
        self.assertEqual(sum(self.summary("apply")[0]["saved"].values()), 0)
        self.menu(["2", "0"])
        self.assertTrue(a.samefile(d))
        self.assertGreater(sum(self.summary("apply")[0]["saved"].values()), 0)

    def test_group_merge_preserves_replaced_alias(self):
        a, b, c, d = [self.publish(name) for name in ("a", "b", "c", "d")]
        b.unlink(); os.link(a, b)
        d.unlink(); os.link(c, d)
        replacement = self.write("new.csv", b"updated independent content")
        original_replace = os.replace
        def replace_next_alias(src, dst):
            original_replace(src, dst)
            if Path(dst) == c:
                original_replace(replacement, d)
        with patch("services.project_cleaner.os.replace", side_effect=replace_next_alias):
            self.menu(["2", "0"])
        self.assertEqual(d.read_bytes(), b"updated independent content")
        self.assertTrue(a.samefile(c))
        self.assertFalse(a.samefile(d))

    def completed_robustness(self, *, benchmark=True, namespace="suite/rolling"):
        import pandas as pd
        from services.research import strategy_multi_seed_robustness as owner
        from core.strategy_compare_policy import get_strategy_comparison_settings
        settings = get_strategy_comparison_settings()
        arm = next(iter(settings.enabled_arms))
        fixed = next(a for a in settings.arms.values() if not a.dl_enabled)
        run = self.root / rules.ROBUST / namespace / "1234567890abcdef"
        period = {"start": "2021-01-01", "end": "2021-12-31"}
        contract = {"fingerprint": run.name, "profile_id": settings.profile_id,
                    "resolved_seeds": [42], "stochastic_arms": [{"arm_id": arm.arm_id}],
                    "fixed_arms": [{"arm_id": fixed.arm_id}], "comparison_period": period,
                    "retention": {"keep_scores": False, "keep_replay_details": False}}
        metrics = {key: 1.0 for _, key, _ in owner.MEAN_METRICS}
        self.write(run / owner.SEED_RESULTS_FILENAME, pd.DataFrame([{
            "arm_id": arm.arm_id, "seed": 42, "arm_order": 1, "seed_order": 1, **metrics,
        }]).to_csv(index=False).encode())
        self.write(run / owner.SEED_YEARLY_RESULTS_FILENAME, pd.DataFrame([{
            "arm_id": arm.arm_id, "seed": 42, "arm_order": 1, "seed_order": 1,
            "year": 2021, "return_pct": 1.0, "is_complete_year": True,
        }]).to_csv(index=False).encode())
        summary = {"schema_version": owner.ROBUSTNESS_SCHEMA_VERSION, "contract": contract,
                   "paired_comparisons": [], "common_strategy_report": [],
                   "mean_strategy_metrics": [{"arm_id": arm.arm_id, "type": "Multi-seed", "n": 1, **metrics}]}
        if benchmark:
            summary["benchmark_yearly"] = [{"year": 2021, "year_return_pct": 1.0}]
        self.write(run / owner.SUMMARY_FILENAME, summary)
        self.write(run / owner.REPORT_FILENAME, b"historical report")
        with patch.object(owner, "PROJECT_ROOT", self.root):
            self.write(run / owner.MANIFEST_FILENAME, {"status": "COMPLETED", "contract": contract,
                       "durable_artifacts": owner._build_durable_result_artifacts(run)})
            self.assertIsNotNone(owner._validate_completed_run(run, backfill_integrity=False))
        return owner, settings, fixed, run, contract

    def test_legacy_robustness_report_keeps_baseline_needed_by_reuse(self):
        from core.strategy_comparison import resolve_strategy_comparison_arm_param_policy, strategy_comparison_param_artifact_key
        owner, settings, arm, run, contract = self.completed_robustness(benchmark=False)
        policy = resolve_strategy_comparison_arm_param_policy(settings, arm)
        all_off = arm.rule_policy == "all_off"
        metadata = {"dataset": settings.dataset, "params_file_sha256": "a" * 64,
                    "requested_param_policy": policy,
                    "param_evaluation_mode": settings.parameter_sources[arm.param_source].canonical_evaluation_mode or "rolling",
                    "optional_entry_filter_policy": owner.OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF if all_off else owner.OPTIONAL_ENTRY_FILTER_POLICY_CURRENT,
                    "shared_param_overrides": dict(owner.ALL_RULE_FILTERS_OFF_OVERRIDES) if all_off else {},
                    "max_positions": settings.max_positions, "enable_rotation": settings.rotation == "on",
                    "comparison_period": contract["comparison_period"]}
        folder = owner._baseline_work_dir(run, arm)
        self.write(folder / "strategy_comparison.json", {"metadata": metadata, "no_filter": {"return_pct": 1.0},
                   "benchmark_yearly": [{"year": 2021, "year_return_pct": 1.0}]})
        for name in ("yearly_returns_comparison.csv", "no_filter_equity.csv", "no_filter_trades.csv",
                     "no_filter_daily_capacity.csv", "no_filter_orderable_candidates.csv"):
            self.write(folder / name, b"col\n1\n")
        status = {"comparison_period": contract["comparison_period"],
                  "resolved_arm_parameter_paths": {arm.arm_id: "params.json"},
                  "artifact_identities": {strategy_comparison_param_artifact_key(arm.param_source, policy): {"sha256": "a" * 64}}}
        with patch.object(owner, "PROJECT_ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            before = owner._run_fixed_baselines(settings=settings, status=status, run_root=run, fixed_arms=(arm,), allow_replay=False)
            self.menu(["2", "0"])
            after = owner._run_fixed_baselines(settings=settings, status=status, run_root=run, fixed_arms=(arm,), allow_replay=False)
        self.assertEqual(before, after)

    def test_completed_robustness_cleans_all_disposable_work_and_remains_reusable(self):
        owner, settings, arm, run, contract = self.completed_robustness()
        trash = [self.write(run / "work" / scope / "leftover.json", b"transient")
                 for scope in ("results", "baselines", "replay", "training", "benchmark_baselines")]
        keep = [self.write(run / "work/custom/unknown.txt", b"unknown"),
                self.write("models/research/shared_fitting_checkpoints/fit/model.pt", b"fit"),
                self.write(run.parent / "scientific_units/unit/manifest.json", b"evidence")]
        durable = {p: p.read_bytes() for p in run.iterdir() if p.is_file()}
        with patch.object(owner, "PROJECT_ROOT", self.root):
            self.menu(["2", "0"])
            result = owner._validate_completed_run(run, expected_contract=contract, backfill_integrity=False)
            with patch.object(owner, "_print_report_tables"), \
                 patch.object(owner, "_train_one_unit", side_effect=AssertionError("unexpected FIT")), \
                 patch.object(owner, "_replay_one_unit", side_effect=AssertionError("unexpected replay")), \
                 contextlib.redirect_stdout(io.StringIO()):
                reused = owner._reuse_completed_robustness_run(
                    cfg=SimpleNamespace(reuse_completed=True), settings=settings,
                    run_root=run, contract=contract, status={}, fixed_arms=())
        self.assertIsNotNone(result)
        self.assertEqual(reused, result["summary"])
        self.assertTrue(all(not p.exists() for p in trash))
        self.assertTrue(all(p.is_file() for p in keep))
        self.assertTrue(all(p.read_bytes() == data for p, data in durable.items()))

    def test_robustness_retention_and_incomplete_work_are_preserved(self):
        owner, settings, arm, run, contract = self.completed_robustness()
        for state in ("RUNNING", "FAILED", "COMPLETED"):
            with self.subTest(state=state):
                contract["retention"].update(keep_scores=True, keep_replay_details=True)
                summary = json.loads((run / owner.SUMMARY_FILENAME).read_text())
                summary["contract"] = contract
                self.write(run / owner.SUMMARY_FILENAME, summary)
                with patch.object(owner, "PROJECT_ROOT", self.root):
                    self.write(run / owner.MANIFEST_FILENAME, {"status": state, "contract": contract,
                        "durable_artifacts": owner._build_durable_result_artifacts(run)})
                    files = {scope: self.write(run / "work" / scope / "keep.json", b"preserve")
                             for scope in ("results", "baselines", "replay", "training", "benchmark_baselines")}
                    self.menu(["2", "0"])
                for scope, file in files.items():
                    self.assertEqual(file.exists(), state != "COMPLETED" or scope not in {"results", "baselines"})

    def test_robustness_work_referenced_by_provenance_is_preserved(self):
        owner, settings, arm, run, contract = self.completed_robustness()
        protected = self.write(run / "work/replay/evidence.csv", b"historical backing bytes")
        self.write(run / owner.ATTRIBUTION_SOURCE_DIRNAME / "unit/manifest.json", {
            "files": {"trades": {"path": protected.relative_to(self.root).as_posix()}}})
        disposable = self.write(run / "work/results/result.json", b"transient")
        with patch.object(owner, "PROJECT_ROOT", self.root):
            self.menu(["2", "0"])
        self.assertTrue(protected.exists())
        self.assertFalse(disposable.exists())

    def test_registered_legacy_robustness_depths_are_cleaned(self):
        for namespace in ("", "selection_pit", "suite/rolling"):
            with self.subTest(namespace=namespace):
                owner, settings, arm, run, contract = self.completed_robustness(namespace=namespace)
                disposable = self.write(run / "work/results/result.json", b"transient")
                with patch.object(owner, "PROJECT_ROOT", self.root):
                    self.menu(["2", "0"])
                    self.assertIsNotNone(owner._validate_completed_run(run, backfill_integrity=False))
                self.assertFalse(disposable.exists())

    def test_empty_completed_work_is_removed_but_unknown_and_active_folders_stay(self):
        owner, settings, arm, run, contract = self.completed_robustness()
        empty = run / "work/replay/seed/pair"
        empty.mkdir(parents=True)
        unknown = run / "work/custom/notes"
        unknown.mkdir(parents=True)
        active = run.parent / "incomplete/work/replay/seed"
        active.mkdir(parents=True)
        lookalike = self.write(rules.ROBUST / "unrecognized/work", b"a file named work")
        with patch.object(owner, "PROJECT_ROOT", self.root):
            self.menu(["1", "0"])
            self.assertTrue(empty.is_dir())
            self.menu(["2", "0"])
        self.assertFalse((run / "work/replay").exists())
        self.assertTrue(unknown.is_dir() and active.is_dir())
        self.assertEqual(lookalike.read_bytes(), b"a file named work")

    def test_empty_work_respects_lease_and_replaced_links(self):
        from core.file_integrity import exclusive_file_lock, sibling_publication_lock_path
        owner, settings, arm, run, contract = self.completed_robustness()
        empty = run / "work/replay/seed/pair"
        empty.mkdir(parents=True)
        unrelated = self.root / "unrelated/empty"
        unrelated.mkdir(parents=True)
        with patch.object(owner, "PROJECT_ROOT", self.root):
            with exclusive_file_lock(sibling_publication_lock_path(run)):
                self.menu(["2", "0"])
                self.assertTrue(empty.is_dir())
            worker, rows = self.scanned_cleaner()
            empty.rmdir()
            empty.symlink_to(unrelated, target_is_directory=True)
            worker.clear_unused()
        self.assertTrue(empty.is_symlink())
        self.assertTrue(unrelated.is_dir())

    def test_newly_tracked_candidate_is_preserved(self):
        a, b, c = [self.publish(name) for name in ("a", "b", "c")]
        worker, rows = self.scanned_cleaner()
        self.git("add", str(b))
        worker.clear_unused()
        worker.share()
        self.assertFalse(a.samefile(b))
        self.assertTrue(a.samefile(c))
        self.assertTrue(any(row[0] == "KEEP" and row[1] == b.relative_to(self.root) for row in rows))

    def test_git_changes_during_hashing_are_seen_before_replacement(self):
        a, b = self.publish("a"), self.publish("b")
        worker, rows = self.scanned_cleaner()
        original_hash = worker.hash_file
        def stage_after_hash(path):
            digest = original_hash(path)
            if path == b:
                self.git("add", str(b))
            return digest
        with patch.object(worker, "hash_file", side_effect=stage_after_hash):
            worker.share()
        self.assertFalse(a.samefile(b))
        self.assertFalse(any(row[0] == "LINK" for row in rows))

    def test_missing_file_does_not_discard_other_candidates(self):
        a, b, c = [self.publish(name) for name in ("a", "b", "c")]
        worker, rows = self.scanned_cleaner()
        a.unlink()
        worker.clear_unused()
        worker.share()
        self.assertTrue(b.samefile(c))
        self.assertTrue(any(row[0] == "SKIP" and row[1] == a.relative_to(self.root) for row in rows))

    def test_updated_legacy_source_requires_fresh_migration_proof(self):
        rel = Path("models/run_best_params.json")
        source = self.write(rel, b"old")
        worker, rows = self.scanned_cleaner()
        worker.legacy = {rel: "content_sha_match"}
        source.write_bytes(b"new")
        worker.remove_one("legacy", rel, 3, set())
        self.assertEqual(source.read_bytes(), b"new")
        self.assertEqual(rows[-1][0], "KEEP")

    def test_interrupted_link_syscall_can_be_retried(self):
        a, b = self.publish("a"), self.publish("b")
        original_link = os.link
        def link_then_interrupt(src, dst):
            original_link(src, dst)
            raise KeyboardInterrupt
        with patch("services.project_cleaner.os.link", side_effect=link_then_interrupt):
            self.menu(["2", "0"])
        self.assertFalse((self.root / "outputs/project_cleanup/pending.link").exists())
        self.menu(["2", "0"])
        self.assertTrue(a.samefile(b))

    def test_failed_stage_removal_preserves_recovery_intent(self):
        a, b = self.publish("a"), self.publish("b")
        stage = self.root / "outputs/project_cleanup/pending.link"
        original_unlink, original_replace = Path.unlink, os.replace
        def reject_stage_unlink(path, *args, **kwargs):
            if path == stage:
                raise PermissionError("injected stage is busy")
            return original_unlink(path, *args, **kwargs)
        def reject_stage_replace(src, dst):
            if Path(src) == stage:
                raise PermissionError("injected destination is busy")
            return original_replace(src, dst)
        with patch.object(Path, "unlink", reject_stage_unlink), \
             patch("services.project_cleaner.os.replace", side_effect=reject_stage_replace):
            self.menu(["2", "0"])
        self.assertTrue(stage.exists())
        self.assertTrue((stage.parent / "apply.pending.csv").exists())
        self.assertIn("略過", self.summary("apply")[0]["status"])
        self.menu(["2", "0"])
        self.assertFalse(stage.exists())
        self.assertTrue(a.samefile(b))

    def test_malformed_pair_mapping_is_retained(self):
        a, b = self.publish("a"), self.publish("b")
        for p in (a, b):
            result = p.parents[2] / "strategy_comparison.json"
            payload = json.loads(result.read_text())
            payload["pairs"] = ["bad shape"]
            result.write_text(json.dumps(payload))
        self.menu(["2", "0"])
        self.assertFalse(a.samefile(b))

    def test_corrupt_history_returns_to_menu(self):
        target = self.root / "outputs/project_cleanup/last_apply.zip"
        target.parent.mkdir(parents=True)
        for contents in (None, {}, {"mode": "apply", "status": "已完成", "counts": []}):
            with self.subTest(contents=contents):
                with ZipFile(target, "w") as archive:
                    archive.writestr("decisions.csv", "")
                    if contents is not None:
                        archive.writestr("summary.json", json.dumps(contents))
                self.assertIn("本次操作未完成", self.menu(["3", "0"]))

    def test_cached_publication_rejects_later_changes(self):
        for changed in ("csv", "manifest", "pair_json"):
            with self.subTest(changed=changed):
                a, b = self.publish("a", pinned=True), self.publish("b", pinned=True)
                worker, rows = self.scanned_cleaner()
                for p in (a, b):
                    self.assertTrue(worker.published(p.relative_to(self.root)))
                    if changed == "csv":
                        stamp = p.stat().st_mtime_ns
                        p.write_bytes(b"!" * len(self.payload))
                        os.utime(p, ns=(stamp, stamp))
                    elif changed == "manifest":
                        (p.parents[2] / "manifest.json").write_text("{}")
                    else:
                        (p.parent / "strategy_comparison.json").write_text("{}")
                worker.share()
                self.assertFalse(a.samefile(b))
                self.assertFalse(any(row[0] == "LINK" for row in rows))

    def test_publication_and_dedup_read_each_csv_once(self):
        from core import file_integrity
        a, b = self.publish("a", pinned=True), self.publish("b", pinned=True)
        file_integrity._observed_file_sha256.cache_clear()
        with patch.object(file_integrity, "compute_file_sha256", wraps=file_integrity.compute_file_sha256) as hashed:
            self.menu(["1", "0"])
        observed = [Path(call.args[0]) for call in hashed.call_args_list]
        self.assertEqual(observed.count(a), 1)
        self.assertEqual(observed.count(b), 1)
        self.assertEqual(self.summary("preview")[0]["counts"]["PLAN_LINK"], 1)

    def test_menu_proofs_dedup_retry_and_independent_rewrites(self):
        a, b = self.publish("a", pinned=True), self.publish("b")
        unequal = self.publish("c", data=b"Z" * len(self.payload))
        partial = self.publish("d")
        (partial.parents[2] / "manifest.json").unlink()
        src = self.write("services/module.py", b"v=42")
        self.git("add", "services/module.py")
        pyc = Path(py_compile.compile(str(src), doraise=True))
        stale = self.write(self.base / "a/pairs/arm/strategy_comparison.html", b"retired")
        fixture = self.write(rules.STAGING / "quick_gate_broken_run_best_params_12345678/models/run_best_params.json",
                             b"{not-valid-json")
        custom = self.write(rules.STAGING / "quick_gate_broken_run_best_params_xxxxxxxx/models/run_best_params.json",
                            b"important")
        protected = [self.write(x, b"keep") for x in ("data/trading/raw", "state/trading/open",
                     "models/research/checkpoint.pt", "outputs/optimizer/study.db", "outputs/unclassified/draft")]
        before, report = self.menu(["", "0"]), self.summary("preview")[0]
        self.assertIn("預估可釋放", before)
        self.assertEqual(report["counts"]["PLAN_LINK"], 1)
        self.assertTrue(pyc.exists() and stale.exists() and fixture.exists())
        self.assertFalse(os.path.samefile(a, b))
        self.menu(["2", "3", self.output, "0"])
        stats, rows = self.summary("apply")
        self.assertEqual(stats["counts"]["LINK"], 1)
        self.assertEqual(stats["counts"]["DELETE"], 3)
        self.assertTrue(os.path.samefile(a, b))
        self.assertTrue(all(item.exists() for item in [custom, partial, unequal, *protected]))
        self.assertFalse(os.path.samefile(a, unequal))
        self.assertFalse(os.path.samefile(a, partial))
        self.assertFalse(pyc.exists() or stale.exists() or fixture.exists())
        for p in (a,b):
            directory=p.parents[2]
            self.assertTrue(completed_strategy_run_publication_matches(
                run_dir=directory, project_root=self.root,
                run_payload=json.loads((directory / "strategy_comparison.json").read_text())))
        self.menu(["2","0"])
        self.assertEqual(sum(self.summary("apply")[0]["saved"].values()), 0)
        replay = self.root / self.base / "e/pairs/arm" / self.output
        replay.parent.mkdir(parents=True)
        copy_reused_detail(a, replay)
        self.assertTrue(os.path.samefile(a,replay))
        import pandas as pd
        publish_detail_csv(pd.DataFrame({"new":[1]}), replay, index=False, encoding="utf-8-sig")
        self.assertFalse(os.path.samefile(a,replay))
        self.assertEqual(a.read_bytes(), self.payload)
        self.assertEqual(b.read_bytes(), self.payload)

    def test_reuse_copies_independently_and_traces_why_link_was_skipped(self):
        source = self.publish("original")
        for run, metadata, link_error, reason in (
            ("link_error", [True, True], OSError("link unavailable"), "link unavailable"),
            ("different_acl", [True, False], None, "不同檔案安全屬性"),
        ):
            with self.subTest(run=run):
                pair = self.root / self.base / run / "pairs/arm"
                dest = pair / source.name
                with patch("core.compare_csv_storage.file_metadata", side_effect=metadata), \
                     patch("core.compare_csv_storage.os.link", side_effect=link_error) as link, \
                     self.assertLogs("core.compare_csv_storage", level="DEBUG") as log:
                    shutil.copytree(source.parent, pair, copy_function=copy_reused_detail)
                self.assertEqual(link.call_count, int(link_error is not None))
                message, = log.output
                for fragment in (reason, source.relative_to(self.root).as_posix(), dest.relative_to(self.root).as_posix()):
                    self.assertIn(fragment, message)
                self.assertEqual(dest.read_bytes(), self.payload)
                self.assertEqual(dest.stat().st_mtime_ns, source.stat().st_mtime_ns)
                self.assertFalse(dest.samefile(source))
                dest.write_bytes(b"changed")
                self.assertEqual(source.read_bytes(), self.payload)

    def test_refuse_unmanaged_links_corruption_active_work_and_tracked(self):
        a, b = self.publish("a", pinned=True), self.publish("b", pinned=True)
        a.write_bytes(b"!" * len(self.payload))
        os.utime(a, ns=(b.stat().st_mtime_ns, b.stat().st_mtime_ns))
        self.write(self.base / "a/pairs/arm/strategy_comparison.html", b"retired")
        alias=self.root / "private.csv"
        os.link(b,alias)
        src=self.write("services/tracked.py", b"x=1")
        pyc=Path(py_compile.compile(str(src),doraise=True))
        self.git("add","-f",str(src),str(pyc))
        stub=self.write(rules.STAGING / "quick_gate_broken_run_best_params_12345678/models/run_best_params.json", b"{not-valid-json")
        with fixture_access(self.root / rules.STAGING, cleanup=False):
            self.menu(["2","0"])
        self.assertTrue(stub.is_file() and pyc.is_file() and (a.parent/"strategy_comparison.html").is_file())
        self.assertFalse(os.path.samefile(a,b))
        self.assertTrue(os.path.samefile(b,alias))
        self.assertEqual(self.summary("apply")[0]["counts"].get("LINK",0),0)

    def test_change_in_model_roster_preserves_cache_and_fit(self):
        from core.strategy_compare_policy import _compatibility_catalogs
        from config import breakout_quality
        slots = rules.retired_diagnostics(self.root)
        self.assertTrue(slots)
        rel, profile = next(iter(slots.items()))
        eligible = self.diagnostic_projection(rel)
        fit = self.write(Path("models/research") / profile / "model.pt", b"unique historical model")
        self.assertIn(profile, {v.get("experiment_profile") for v in _compatibility_catalogs()[1].values()})
        with patch.object(breakout_quality, "BREAKOUT_QUALITY_MODEL_RESEARCH_EXPERIMENT_PROFILE", profile):
            self.assertNotIn(rel, rules.retired_diagnostics(self.root))
            self.menu(["2", "0"])
            self.assertTrue(eligible.exists())
        self.menu(["2", "0"])
        self.assertFalse(eligible.exists())
        self.assertTrue(fit.is_file())

    def test_robust_work_requires_committed_evidence_and_lease(self):
        run = rules.ROBUST / "suite/rolling/1234567890abcdef"
        work = self.write(run / "work/results/result.json", b"temp")
        replay = self.write(run / "work/replay/seed_result.json", b"durable if policy requires")
        proof = self.write(run / "robustness_summary.json", b"permanent")
        from core.file_integrity import exclusive_file_lock, sibling_publication_lock_path
        lock_path = sibling_publication_lock_path(self.root / run)
        self.menu(["1", "0"])
        self.assertFalse(lock_path.exists())
        self.menu(["2", "0"])
        self.assertTrue(work.is_file())
        from services.research import strategy_multi_seed_robustness as owner
        with patch.object(owner, "_validate_completed_run", return_value={"contract": {"retention": {"keep_replay_details": True}}, "summary": {}}):
            with exclusive_file_lock(sibling_publication_lock_path(self.root / run)):
                self.menu(["2", "0"])
                self.assertTrue(work.is_file())
            self.menu(["2", "0"])
        self.assertFalse(work.is_file())
        self.assertTrue(proof.is_file() and replay.is_file())

    def test_recover_abrupt_interruption_between_link_and_replace(self):
        a, b = self.publish("a"), self.publish("b")
        from services.cleanup_report import FIELDS
        target = self.root / "outputs/project_cleanup"
        target.mkdir(parents=True)
        with (target / "apply.pending.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(FIELDS)
            writer.writerow(["INTENT", b.relative_to(self.root).as_posix(), len(self.payload), 0, "link",
                             a.relative_to(self.root).as_posix()])
        os.link(a, target / "pending.link")
        self.menu(["3", "", "0"])
        self.assertFalse((target / "pending.link").exists())
        self.assertFalse(os.path.samefile(a,b))
        self.assertIn("中斷", self.summary("apply")[0]["status"])
        self.menu(["2","0"])
        self.assertTrue(os.path.samefile(a,b))

    def test_interrupt_and_recover(self):
        a,b=self.publish("a"),self.publish("b")
        with patch.object(service.Cleaner,"link_one",side_effect=KeyboardInterrupt):
            self.menu(["2","0"])
        self.assertIn("中斷",self.summary("apply")[0]["status"])
        self.assertFalse(os.path.samefile(a,b))
        self.menu(["2","0"])
        self.assertTrue(os.path.samefile(a,b))


if __name__ == "__main__":
    unittest.main()
