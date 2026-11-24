"""Deletion-boundary checks for independently superseded robustness work."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
import json
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from core.file_integrity import compute_observed_file_sha256, exclusive_file_lock, sibling_publication_lock_path
from core.artifact_inspection import ArtifactInspection, InspectionStatus
from services import cleanup_superseded_robustness as cleanup


@dataclass
class Config:
    robustness_id: str = "generic"
    profile_id: str = "generic"
    output_root: str = "outputs/strategy_compare/robustness/generic/rolling"
    model_work_root: str = "models/research/generic_work"
    benchmark_id: str = "generic"


@dataclass
class Settings:
    profile_id: str = "generic"
    dl_sources: dict = field(default_factory=dict)


@dataclass
class Source:
    experiment_profile: str = "generic_model"


class SupersededRobustnessTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.cfg = Config()
        self.settings = Settings()
        self.old = Path(self.cfg.output_root) / ("a" * 16)
        self.new = Path(self.cfg.output_root) / ("b" * 16)
        self.contract = {
            "robustness_id": "generic", "fingerprint": self.old.name,
            "model_artifact_fingerprint": "c" * 16,
            "retention": {"keep_scores": False, "keep_checkpoints": False, "keep_replay_details": False},
            "scientific_contract_version": 6, "dataset": "full", "dataset_identity": {"sha": "same"},
            "comparison_period": {"start": "2021-01-01", "end": "2021-12-31"},
            "resolved_seeds": [7, 11], "stochastic_arms": [{"arm_id": "A", "model_seed_sensitive": False}],
            "benchmark_parameter_artifact_identities": {
                "A:seed=7": {"sha256": "d" * 64}, "A:seed=11": {"sha256": "e" * 64}},
        }
        self.old_payload = {"status": "FAILED", "resumable": True, "contract": self.contract,
                            "failed_at_utc": "2001-01-01T00:00:00+00:00"}
        new_contract = deepcopy(self.contract)
        new_contract["fingerprint"] = self.new.name
        new_contract["model_artifact_fingerprint"] = "d" * 16
        self.new_payload = {"status": "COMPLETED", "contract": new_contract}
        self.rows = pd.DataFrame([{"arm_id": "A", "seed": n, "total_return_pct": 5, "selected_epoch": 0,
                                  "strategy_param_sha256": h * 64} for n, h in ((7, "d"), (11, "e"))])
        self.years = pd.DataFrame([{"arm_id": "A", "seed": n, "year": 2021,
                                   "return_pct": 5, "is_complete_year": True} for n in (7, 11)])
        self.write(self.old / "manifest.json", self.old_payload)
        self.write(self.new / "manifest.json", self.new_payload)
        self.rows.iloc[:1].to_csv(self.root / self.old / cleanup.owner.SEED_RESULTS_FILENAME, index=False)
        self.years.iloc[:1].to_csv(self.root / self.old / cleanup.owner.SEED_YEARLY_RESULTS_FILENAME, index=False)
        self.target = self.old / "work/benchmark_baselines/A__seed_7/no_filter_orderable_candidates.csv"
        self.write(self.target, b"ticker,value\n0050,1\n")
        self.pair = {"metadata": {"output_dir": self.target.parent.as_posix(), "output_scope": "caller_override",
                     "score_source": "dl_off_baseline_only", "comparison_design": "standalone_dl_off_active_param_replay",
                     "dataset": "full", "comparison_period": self.contract["comparison_period"],
                     "params_file_sha256": "d" * 64}, "no_filter": {"total_return_pct": 5},
                     "yearly": [{"year": 2021, "no_filter_return_pct": 5, "is_full_year": True}]}
        self.write(self.target.parent / "strategy_comparison.json", self.pair)
        self.write(self.old / "work/benchmark_baselines/A__seed_7/user_notes.csv", b"unique")
        self.write(self.old / "work/training/train.log", b"original failure")
        self.write(self.new / "report.md", b"complete report")
        self.validated = {"contract": new_contract, "seed_frame": self.rows, "seed_yearly_frame": self.years}
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(cleanup.owner, "PROJECT_ROOT", self.root))
        stack.enter_context(patch.object(cleanup, "get_strategy_multi_seed_robustness_profiles", return_value=({"robustness_id": "generic"},)))
        stack.enter_context(patch.object(cleanup, "get_strategy_multi_seed_robustness_settings", return_value=self.cfg))
        stack.enter_context(patch.object(cleanup, "get_strategy_comparison_settings", return_value=self.settings))
        stack.enter_context(patch.object(
            cleanup.owner, "inspect_completed_robustness_run",
            side_effect=lambda path, **kw: (
                ArtifactInspection(InspectionStatus.READY, "", self.validated)
                if path == self.root / self.new else
                ArtifactInspection(InspectionStatus.MISSING, "synthetic source missing")
            ),
        ))
        self.worker = SimpleNamespace(apply=True, hash_file=compute_observed_file_sha256,
                                      lifecycle=SimpleNamespace(removed=lambda path: None))

    def write(self, relative, data):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data if isinstance(data, bytes) else json.dumps(data).encode())
        return path

    def prepare(self):
        adapter = cleanup.SupersededRobustnessCleanup(self.root)
        adapter.prepare({p.relative_to(self.root): p.stat().st_size for p in self.root.rglob("*") if p.is_file()})
        return adapter

    def test_completed_superset_releases_only_owned_work_without_changing_failed_json(self):
        marker = self.root / self.old / "manifest.json"
        original = marker.read_bytes()
        adapter = self.prepare()
        self.assertEqual(adapter.candidate_paths, {self.target})
        self.assertEqual(adapter.ownership_references, {
            self.target.parent / "strategy_comparison.json": {("metadata", "output_dir"): self.target.parent.as_posix()}})
        with adapter.deletion_lease(self.worker, self.target):
            self.assertTrue(adapter.proof(self.worker, self.target))
            adapter.verify_before_delete(self.worker, self.target)
            (self.root / self.target).unlink()
        self.assertEqual(marker.read_bytes(), original)
        self.assertTrue((self.root / self.old / "work/benchmark_baselines/A__seed_7/user_notes.csv").exists())
        self.assertTrue((self.root / self.old / "work/benchmark_baselines/A__seed_7/strategy_comparison.json").exists())

    def test_age_without_complete_unit_coverage_never_authorizes_delete(self):
        self.validated["contract"]["resolved_seeds"] = [7]
        self.assertFalse(self.prepare().candidate_paths)

    def test_foreign_project_with_matching_manifest_cannot_use_global_owner(self):
        with patch.object(cleanup.owner, "PROJECT_ROOT", self.root / "different"):
            adapter = self.prepare()
        self.assertFalse(adapter.candidates)
        self.assertIn(self.old, adapter.errors)

    def test_same_seed_with_different_scientific_contract_is_not_replacement(self):
        self.validated["contract"]["dataset_identity"] = {"sha": "different"}
        self.assertFalse(self.prepare().candidate_paths)

    def test_observation_or_score_revision_conflict_keeps_old_work(self):
        for column, value in (("total_return_pct", 9), ("score_sha256", "f" * 64)):
            with self.subTest(column=column):
                self.validated["seed_frame"] = self.rows.copy()
                self.validated["seed_frame"].loc[0, column] = value
                self.assertFalse(self.prepare().candidate_paths)

    def test_unmerged_late_observation_must_also_match(self):
        late = {"arm_id": "A", "seed": 11, "total_return_pct": 99, "strategy_param_sha256": "e" * 64,
                "yearly": self.years.iloc[1:].to_dict("records")}
        self.write(self.old / "work/results/A__seed_11.json", late)
        self.assertFalse(self.prepare().candidate_paths)

    def test_original_retention_and_completed_publication_are_required(self):
        self.contract["retention"]["keep_scores"] = True
        self.write(self.old / "manifest.json", self.old_payload)
        self.assertFalse(self.prepare().candidate_paths)
        self.contract["retention"]["keep_scores"] = False
        self.write(self.old / "manifest.json", self.old_payload)
        self.validated = None
        self.assertFalse(self.prepare().candidate_paths)

    def test_old_row_mutation_between_validation_and_commit_is_detected(self):
        real = cleanup.owner._observation_equivalence
        def change_after_comparison(left, right):
            real(left, right)
            self.write(self.old / cleanup.owner.SEED_RESULTS_FILENAME, b"changed\n")
        with patch.object(cleanup.owner, "_observation_equivalence", side_effect=change_after_comparison):
            self.assertFalse(self.prepare().candidate_paths)

    def test_replacement_mutation_after_prepare_and_after_proof_is_detected(self):
        for after_proof in (False, True):
            self.write(self.new / "report.md", b"complete report")
            adapter = self.prepare()
            with adapter.deletion_lease(self.worker, self.target):
                if after_proof:
                    self.assertTrue(adapter.proof(self.worker, self.target))
                self.write(self.new / "report.md", b"modified replacement")
                with self.assertRaisesRegex(ValueError, "變動"):
                    if after_proof:
                        adapter.verify_before_delete(self.worker, self.target)
                    else:
                        adapter.proof(self.worker, self.target)

    def test_active_run_lease_prevents_cleaner_entry(self):
        adapter = self.prepare()
        with exclusive_file_lock(sibling_publication_lock_path(self.root / self.old)):
            with self.assertRaises(RuntimeError):
                with adapter.deletion_lease(self.worker, self.target):
                    self.fail("active writer lease was ignored")
        self.assertTrue((self.root / self.target).exists())

    def test_bare_shared_model_fingerprint_blocks_model_candidates(self):
        other = Path(self.cfg.output_root) / ("f" * 16)
        payload = deepcopy(self.old_payload)
        payload["contract"]["fingerprint"] = other.name
        payload["contract"]["resolved_seeds"] = [99]
        self.write(other / "manifest.json", payload)
        with patch.object(cleanup.SupersededRobustnessCleanup, "_model_candidates") as model:
            adapter = self.prepare()
            self.assertIn(self.target, adapter.candidates)
            model.assert_not_called()

    def test_unknown_contract_still_owns_its_bare_model_fingerprint(self):
        other = Path(self.cfg.output_root) / ("f" * 16)
        payload = deepcopy(self.old_payload)
        payload["contract"].update(fingerprint=other.name, robustness_id="retired_unknown_suite")
        self.write(other / "manifest.json", payload)
        with patch.object(cleanup.SupersededRobustnessCleanup, "_model_candidates") as model:
            self.assertIn(self.target, self.prepare().candidates)
            model.assert_not_called()

    def test_mismatched_work_owner_never_gets_reference_exclusion(self):
        self.pair["metadata"]["params_file_sha256"] = "0" * 64
        self.write(self.target.parent / "strategy_comparison.json", self.pair)
        adapter = self.prepare()
        self.assertFalse(adapter.candidates)
        self.assertFalse(adapter.ownership_references)

    def test_replacement_receipt_backing_file_is_kept_even_in_old_work_scope(self):
        self.write(self.new / "attribution_source/unit/manifest.json",
                   {"files": {"raw": {"path": self.target.as_posix()}}})
        self.assertNotIn(self.target, self.prepare().candidates)

    def test_incremental_proof_properties_pin_science_and_cover_new_late_results(self):
        config = self.write(Path("config/owned_settings.py"), b"CONFIG = 1\n")
        late = self.old / "work/results/A__seed_11.json"
        self.write(late, {**self.rows.iloc[1].to_dict(), "yearly": self.years.iloc[1:].to_dict("records")})
        adapter = self.prepare()
        original = adapter.proof_revisions
        expected = {config, self.root / late, self.root / self.old / "manifest.json",
                    self.root / self.new / "manifest.json", self.root / self.new / "report.md",
                    self.root / self.old / cleanup.owner.SEED_RESULTS_FILENAME,
                    self.root / self.target.parent / "strategy_comparison.json"}
        self.assertTrue(expected.issubset(original))
        self.assertTrue({self.old, self.new, Path(self.cfg.model_work_root) / self.contract["model_artifact_fingerprint"]}
                        .issubset(adapter.proof_scopes))
        future_late_result = self.old / "work/results/new_unmerged_observation.json"
        self.assertTrue(any(future_late_result.is_relative_to(scope) for scope in adapter.proof_scopes))
        with patch.object(cleanup, "file_change_signature", side_effect=AssertionError("properties must not refresh authority")):
            adapter.proof_revisions.clear()
            self.assertEqual(adapter.proof_revisions, original)
        self.write(self.new / "report.md", b"new scientific revision")
        self.assertEqual(adapter.proof_revisions, original)
        with adapter.deletion_lease(self.worker, self.target):
            with self.assertRaisesRegex(ValueError, "變動"):
                adapter.proof(self.worker, self.target)

    def test_real_cleaner_preview_apply_and_reference_veto_across_multiple_files(self):
        from services.project_cleaner import Cleaner
        from services.cleanup_rules import robust_unit
        second = self.target.with_name("no_filter_daily_capacity.csv")
        external = self.target.with_name("no_filter_trades.csv")
        own_field = self.old / "work/benchmark_baselines/A__seed_11/no_filter_orderable_candidates.csv"
        for relative in (second, external, own_field):
            self.write(relative, b"known producer work\n")
        pair = deepcopy(self.pair)
        pair["metadata"].update(output_dir=own_field.parent.as_posix(), params_file_sha256="e" * 64)
        # An exact owner field exemption must not hide a different consumer
        # field even when both fields occur in the same original marker.
        pair["audit_input"] = own_field.as_posix()
        self.write(own_field.parent / "strategy_comparison.json", pair)
        self.write(Path("outputs/consumer/manifest.json"), {"input": external.as_posix()})
        # The real cleaner entrypoint creates its receipt directory before scan.
        (self.root / "outputs/project_cleanup").mkdir()
        originals = {p: p.read_bytes() for p in self.root.rglob("*.json")}
        targets = {self.target, second, external, own_field}
        expected = {self.target, second}
        self.assertTrue(all(robust_unit(p) == self.old for p in targets))
        for apply in (False, True):
            rows = []
            report = SimpleNamespace(row=lambda *args, **kwargs: rows.append(args))
            worker = Cleaner(self.root, report, apply)
            with redirect_stdout(io.StringIO()):
                worker.scan()
                self.assertEqual({p for p, _ in worker.storage_candidates["superseded_robustness"]}, targets)
                self.assertFalse(any(p in targets for entries in worker.deletes.values() for _, p, _ in entries))
                worker.clear_unused()
                # A later cleanup phase re-enters the real lifecycle session;
                # our own unlinks and producer lease files must not stale it.
                changed = [p for p, stamp in worker.lifecycle.directory_revisions.items()
                           if cleanup.file_change_signature(self.root / p) != stamp]
                self.assertFalse(changed, (apply, changed, rows))
                with worker.lifecycle.session(worker):
                    pass
            action = "DELETE" if apply else "PLAN_DELETE"
            self.assertEqual({row[1] for row in rows if row[0] == action}, expected, rows)
            self.assertTrue(all((self.root / p).exists() for p in targets - expected))
            self.assertTrue(all((self.root / p).exists() is (not apply) for p in expected))
            self.assertFalse(any(row[0] == "SKIP" and row[1] in targets for row in rows), rows)
        self.assertTrue(all(path.read_bytes() == value for path, value in originals.items()))

    def model_fixture(self):
        from services.research import strategy_compare_training as training
        self.settings.dl_sources["MODEL"] = Source()
        old = Path(self.cfg.model_work_root) / self.contract["model_artifact_fingerprint"] / "MODEL__seed_7/model"
        source = Path("models/research/model_robustness/rolling_oos/generic_model/seed_7")
        fold = Path("folds/fold_20210101_20211231")
        data = b"ticker,score\n0050,1\n"
        digest = sha256(data).hexdigest()
        receipt = {"filename": "scores.csv", "sha256": digest, "size_bytes": len(data)}
        metadata = {"contract_fingerprint": "same-fit", "seed": 7, "planned_periods": {"end": "2020-12-31"},
                    "artifacts": {"scores": receipt}}
        for prefix in (old, source):
            self.write(prefix / fold / "scores.csv", data)
            self.write(prefix / fold / "model.pt", b"original weights")
            self.write(prefix / fold / "validation_scores.csv", b"unique unpinned scores")
            self.write(prefix / fold / "manifest.json", metadata)
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch("core.breakout_quality_policy.get_breakout_quality_workflow_settings", return_value=object()))
        stack.enter_context(patch.object(training, "resolve_shared_model_rolling_robustness_dirs",
                                         return_value=(self.root / source, self.root / source)))
        stack.enter_context(patch.object(training, "validate_strategy_compare_training_artifacts", return_value={}))
        return old / fold / "scores.csv", source / fold / "scores.csv", old, source

    def test_model_scope_preserves_weights_unique_scores_and_requires_actual_equal_bytes(self):
        target, source, old_root, canonical = self.model_fixture()
        adapter = self.prepare()
        self.assertEqual(adapter.sources, {target: source})
        self.assertTrue({old_root, canonical}.issubset(adapter.proof_scopes))
        self.assertIn(self.root / source, adapter.proof_revisions)
        self.assertNotIn(target.with_name("model.pt"), adapter.candidates)
        self.assertNotIn(target.with_name("validation_scores.csv"), adapter.candidates)
        with adapter.deletion_lease(self.worker, target):
            self.assertTrue(adapter.proof(self.worker, target))
            adapter.verify_before_delete(self.worker, target)
        # Keep the same length, but change bytes after publication. Metadata and
        # the existence of another file are insufficient substitution proof.
        self.write(source, b"!" * (self.root / source).stat().st_size)
        adapter = self.prepare()
        with adapter.deletion_lease(self.worker, target):
            self.assertFalse(adapter.proof(self.worker, target))

    def test_model_fold_identity_and_source_writer_lease_are_required(self):
        from services.breakout_quality.fit_repair_transaction import selection_pit_build_lock_path
        target, source, _, canonical = self.model_fixture()
        adapter = self.prepare()
        with exclusive_file_lock(selection_pit_build_lock_path(self.root / canonical)):
            with self.assertRaises(RuntimeError):
                with adapter.deletion_lease(self.worker, target):
                    self.fail("canonical score writer was not excluded")
        marker = self.root / source.parent / "manifest.json"
        payload = json.loads(marker.read_text())
        payload["planned_periods"] = {"end": "2021-12-31"}
        self.write(marker, payload)
        self.assertNotIn(target, self.prepare().candidates)

    def test_new_shared_model_owner_after_prepare_blocks_score_deletion(self):
        target, _, _, _ = self.model_fixture()
        adapter = self.prepare()
        other = Path(self.cfg.output_root) / ("f" * 16)
        payload = deepcopy(self.old_payload)
        payload["contract"]["fingerprint"] = other.name
        self.write(other / "manifest.json", payload)
        with adapter.deletion_lease(self.worker, target):
            with self.assertRaisesRegex(ValueError, "owner"):
                adapter.proof(self.worker, target)


class EmptyProjectTest(unittest.TestCase):
    def test_unrelated_empty_project_needs_no_global_owner_patch(self):
        with tempfile.TemporaryDirectory() as temporary:
            adapter = cleanup.SupersededRobustnessCleanup(temporary)
            with patch.object(cleanup, "get_strategy_multi_seed_robustness_profiles", side_effect=AssertionError("unrelated config was queried")):
                adapter.prepare({})
            self.assertFalse(adapter.candidates)
            self.assertFalse(adapter.errors)


if __name__ == "__main__":
    unittest.main()
