"""Destructive-boundary checks for producer-declared, identity-independent cleanup."""
import contextlib
from datetime import datetime, timedelta, timezone
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from core.artifact_lifecycle import (
    MANIFEST_NAME, lifecycle_access, publish_lifecycle, read_lifecycle,
)
from core.file_integrity import compute_file_sha256
from core.output_retention import RetentionRule, expired_paths
from services import cleanup_rules as rules
from tools.validate import test_project_cleaner as previous


class LifecycleCleanupTest(unittest.TestCase):
    setUp = previous.CleanupTest.setUp
    git = previous.CleanupTest.git
    write = previous.CleanupTest.write
    menu = previous.CleanupTest.menu
    summary = previous.CleanupTest.summary
    scanned_cleaner = previous.CleanupTest.scanned_cleaner

    def storage(self, name="arbitrary-research", *, state="COMPLETED", roles=None,
                requires=None, retain=(), subject="", replacements=None, active_pid=None):
        workspace = Path("outputs/research/experiments") / name
        roles = roles or {"scratch.csv": "work", "scores.csv": "evidence", "model.pt": "resume"}
        files = {}
        for filename, role in roles.items():
            files[filename] = self.write(workspace / filename, {"value": filename} if filename.endswith(".json") else b"value\n1\n")
        result = self.write(Path("outputs/research/published") / (name + ".json"), {"result": 1})
        paths = {path.relative_to(self.root): roles[name] for name, path in files.items()}
        edges = {workspace / name: tuple(workspace / p for p in children)
                 for name, children in (requires or {}).items()}
        replacement_paths = {workspace / name: Path(value) for name, value in (replacements or {}).items()}
        marker = publish_lifecycle(
            self.root, workspace=workspace, producer="generic_research_owner", state=state,
            roles=paths, publication=(result.relative_to(self.root),), requires=edges,
            replacements=replacement_paths, retain=tuple(workspace / p for p in retain),
            subject=subject, active_pid=active_pid,
        )
        return workspace, files, result, marker

    def test_completed_work_is_deleted_but_evidence_and_resume_are_unchanged(self):
        workspace, files, result, marker = self.storage()
        before = {p: p.read_bytes() for p in (files["scores.csv"], files["model.pt"], result, marker)}
        self.menu(["1", "0"])
        self.assertTrue(files["scratch.csv"].exists())
        self.menu(["2", "0"])
        self.assertFalse(files["scratch.csv"].exists())
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        self.menu(["2", "0"])
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_transitive_and_directory_dependencies_are_kept(self):
        _, files, _, _ = self.storage(
            roles={"result.json": "result", "a.csv": "work", "tree/b.csv": "work", "junk.csv": "work"},
            requires={"result.json": ["a.csv"], "a.csv": ["tree"]},
        )
        self.menu(["2", "0"])
        self.assertTrue(files["a.csv"].exists() and files["tree/b.csv"].exists())
        self.assertFalse(files["junk.csv"].exists())

    def test_cross_workspace_dependency_and_cycles_do_not_lose_backing_files(self):
        root_a, files_a, _, marker_a = self.storage("one", roles={"a.csv": "work", "b.csv": "work"}, requires={"a.csv": ["b.csv"], "b.csv": ["a.csv"]})
        root_b, files_b, _, marker_b = self.storage("two", roles={"evidence.json": "evidence"})
        payload = json.loads(marker_b.read_text())
        payload["artifacts"][0]["requires"] = [(root_a / "a.csv").as_posix()]
        marker_b.write_text(json.dumps(payload))
        self.menu(["2", "0"])
        self.assertTrue(all(p.exists() for p in files_a.values()))

    def test_existing_json_reference_protects_the_whole_referenced_directory(self):
        workspace, files, _, _ = self.storage()
        self.write("outputs/another_owner/manifest.json", {"source_dir": workspace.as_posix()})
        self.menu(["2", "0"])
        self.assertTrue(all(p.exists() for p in files.values()))

    def test_running_and_resumable_work_never_becomes_garbage_by_age(self):
        for state in ("RUNNING", "RESUMABLE"):
            with self.subTest(state=state):
                _, files, _, _ = self.storage(state, state=state)
                for path in files.values():
                    os.utime(path, (1, 1))
                self.menu(["2", "0"])
                self.assertTrue(all(p.exists() for p in files.values()))

    def test_superseded_cache_uses_the_live_subjects_not_a_fixed_model_list(self):
        for subject in ("future-profile-alpha", "unseen-profile-beta"):
            with self.subTest(subject=subject):
                _, files, _, _ = self.storage(subject, state="SUPERSEDED", roles={"old.npz": "cache"}, subject=subject)
                with patch.object(rules, "active_artifact_subjects", return_value={subject}):
                    self.menu(["2", "0"])
                self.assertTrue(files["old.npz"].exists())
                with patch.object(rules, "active_artifact_subjects", return_value=set()):
                    self.menu(["2", "0"])
                self.assertFalse(files["old.npz"].exists())

    def test_roster_change_between_scan_and_apply_protects_a_retired_cache(self):
        _, files, _, _ = self.storage(state="SUPERSEDED", roles={"old.npz": "cache"}, subject="new-live")
        with patch.object(rules, "active_artifact_subjects", return_value=set()):
            worker, _ = self.scanned_cleaner()
        with patch.object(rules, "active_artifact_subjects", return_value={"new-live"}):
            worker.clear_unused()
        self.assertTrue(files["old.npz"].exists())

    def test_changed_published_result_keeps_work_files(self):
        _, files, result, _ = self.storage()
        result.write_bytes(b"changed")
        self.menu(["2", "0"])
        self.assertTrue(files["scratch.csv"].exists())

    def test_modified_work_file_and_newly_tracked_work_are_kept(self):
        _, files, _, _ = self.storage(roles={"changed.csv": "work", "tracked.csv": "work", "unused.csv": "work"})
        worker, _ = self.scanned_cleaner()
        files["changed.csv"].write_bytes(b"changed bytes")
        self.git("add", str(files["tracked.csv"]))
        worker.clear_unused()
        self.assertTrue(files["changed.csv"].exists() and files["tracked.csv"].exists())
        self.assertFalse(files["unused.csv"].exists())

    def test_changed_reference_record_after_scan_blocks_deletion(self):
        workspace, files, _, _ = self.storage()
        ref = self.write("outputs/other/manifest.json", {"source": "unrelated"})
        worker, _ = self.scanned_cleaner()
        ref.write_text(json.dumps({"source": (workspace / "scratch.csv").as_posix()}))
        worker.clear_unused()
        self.assertTrue(files["scratch.csv"].exists())

    def test_new_reference_after_scan_blocks_deletion(self):
        workspace, files, _, _ = self.storage()
        worker, _ = self.scanned_cleaner()
        self.write("state/new_consumer/manifest.json", {"inputs": {(workspace / "scratch.csv").as_posix(): "sha"}})
        worker.clear_unused()
        self.assertTrue(files["scratch.csv"].exists())
        self.menu(["2", "0"])
        self.assertTrue(files["scratch.csv"].exists())

    def test_dependency_protection_also_applies_to_old_parameter_and_reuse_rules(self):
        old, _ = previous.CleanupTest.migrated_parameter(self)
        original = previous.CleanupTest.publish(self, "first", pinned=True)
        copy = previous.CleanupTest.publish(self, "second", pinned=True, source=original)
        workspace, files, result, marker = self.storage()
        payload = json.loads(marker.read_text())
        payload["retain"] = [old.relative_to(self.root).as_posix(), copy.relative_to(self.root).as_posix()]
        marker.write_text(json.dumps(payload))
        self.menu(["2", "0"])
        self.assertTrue(old.exists() and copy.exists() and original.exists())
        self.assertFalse(files["scratch.csv"].exists())

    def test_broad_output_destination_does_not_pin_every_workspace(self):
        _, files, _, _ = self.storage()
        self.write("config/output_destination.json", {"output_root": "outputs"})
        self.menu(["2", "0"])
        self.assertFalse(files["scratch.csv"].exists())

    def test_replica_requires_an_equal_retained_replacement(self):
        original = self.write("outputs/research/published/original.csv", b"value\n1\n")
        _, files, _, _ = self.storage(roles={"copy.csv": "replica"}, replacements={"copy.csv": original.relative_to(self.root)})
        self.menu(["2", "0"])
        self.assertFalse(files["copy.csv"].exists())
        self.assertEqual(original.read_bytes(), b"value\n1\n")
        _, files, _, _ = self.storage("changed-source", roles={"copy.csv": "replica"}, replacements={"copy.csv": original.relative_to(self.root)})
        original.write_bytes(b"changed")
        self.menu(["2", "0"])
        self.assertTrue(files["copy.csv"].exists())

    def test_abandoned_work_needs_a_terminal_record_and_no_active_writer(self):
        _, files, _, _ = self.storage(state="ABANDONED", roles={"temp.bin": "work"}, active_pid=os.getpid())
        self.menu(["2", "0"])
        self.assertTrue(files["temp.bin"].exists())
        with patch("services.cleanup_lifecycle.process_is_absent", return_value=True):
            self.menu(["2", "0"])
        self.assertFalse(files["temp.bin"].exists())

    def test_lifecycle_writer_lease_prevents_cleanup(self):
        _, files, _, _ = self.storage()
        with lifecycle_access(self.root):
            self.menu(["2", "0"])
        self.assertTrue(files["scratch.csv"].exists())

    def test_unknown_files_and_invalid_declarations_are_preserved(self):
        unknown = self.write("outputs/research/experiments/unclassified/model.pt", b"original")
        _, files, _, marker = self.storage()
        payload = json.loads(marker.read_text())
        payload["artifacts"][0]["path"] = "../outside"
        marker.write_text(json.dumps(payload))
        self.menu(["2", "0"])
        self.assertTrue(unknown.exists() and all(p.exists() for p in files.values()))

    def test_invalid_publish_does_not_replace_an_existing_record(self):
        workspace, files, result, marker = self.storage()
        before = marker.read_bytes()
        with self.assertRaises(ValueError):
            publish_lifecycle(self.root, workspace=workspace, producer="owner", state="INVALID",
                              roles={files["scratch.csv"].relative_to(self.root): "work"},
                              publication=(result.relative_to(self.root),))
        self.assertEqual(marker.read_bytes(), before)

    def test_symlink_and_shared_work_file_are_not_unlinked(self):
        _, files, _, _ = self.storage(roles={"linked.csv": "work", "shared.csv": "work"})
        original = self.write("data/real.csv", b"value\n1\n")
        files["linked.csv"].unlink()
        files["linked.csv"].symlink_to(original)
        files["shared.csv"].unlink()
        os.link(original, files["shared.csv"])
        self.menu(["2", "0"])
        self.assertTrue(files["linked.csv"].is_symlink())
        self.assertTrue(files["shared.csv"].exists() and original.exists())

    def bundle(self, name, timestamp):
        path = self.root / "outputs/local_regression" / f"to_chatgpt_bundle_{name}.zip"
        path.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(path, "w") as z:
            z.writestr("master_summary.json", json.dumps({"overall_status": "PASS", "scripts": [], "dataset": "reduced"}))
        os.utime(path, (timestamp, timestamp))
        return path

    def test_bundle_count_policy_is_config_driven_and_rejects_unrecognized_archives(self):
        self.write("tools/local_regression/manifest.json", {
            "retention_enabled": True, "local_regression_keep_last_n": 2,
            "local_regression_max_age_days": 30, "local_regression_retention_mode": "count",
        })
        timestamp = datetime.now(timezone.utc).timestamp()
        bundles = [self.bundle(str(i), timestamp+i) for i in range(5)]
        unknown = self.write("outputs/local_regression/to_chatgpt_bundle_unknown.zip", b"not a diagnostic archive")
        malformed = self.bundle("malformed", timestamp+10)
        with ZipFile(malformed, "w") as archive:
            archive.writestr("master_summary.json", json.dumps({"overall_status": [], "scripts": [], "dataset": "reduced"}))
        self.menu(["1", "0"])
        self.assertTrue(all(p.exists() for p in bundles))
        self.menu(["2", "0"])
        self.assertEqual([p.exists() for p in bundles], [False, False, False, True, True])
        self.assertTrue(unknown.exists() and malformed.exists())

    def test_bundle_and_workspace_publication_dependencies_are_not_deleted_together(self):
        self.write("tools/local_regression/manifest.json", {
            "retention_enabled": True, "local_regression_keep_last_n": 0,
            "local_regression_max_age_days": 0, "local_regression_retention_mode": "count",
        })
        bundle = self.bundle("pinned", 1)
        workspace, files, _, marker = self.storage()
        payload = json.loads(marker.read_text())
        payload["publication"] = {bundle.relative_to(self.root).as_posix(): compute_file_sha256(bundle)}
        marker.write_text(json.dumps(payload))
        self.menu(["2", "0"])
        self.assertTrue(bundle.exists())
        self.assertFalse(files["scratch.csv"].exists())

    def test_finished_disposable_record_does_not_pin_archive_forever(self):
        self.write("tools/local_regression/manifest.json", {
            "local_regression_keep_last_n": 0, "local_regression_max_age_days": 0,
            "local_regression_retention_mode": "count",
        })
        archive = self.bundle("finished", 1)
        _, files, _, marker = self.storage(roles={"temporary.csv": "work"})
        payload = json.loads(marker.read_text())
        payload["publication"] = {archive.relative_to(self.root).as_posix(): compute_file_sha256(archive)}
        marker.write_text(json.dumps(payload))
        self.menu(["2", "0"])
        self.assertFalse(files["temporary.csv"].exists())
        self.assertTrue(archive.exists())
        self.menu(["2", "0"])
        self.assertFalse(archive.exists())
        self.assertTrue(marker.exists())

    def test_referenced_bundle_and_changed_retention_order_are_preserved(self):
        self.write("tools/local_regression/manifest.json", {
            "local_regression_keep_last_n": 1, "local_regression_max_age_days": 30,
        })
        old, newest = self.bundle("old", 1), self.bundle("new", 2)
        self.write("state/diagnostic_evidence.json", {"evidence": old.relative_to(self.root).as_posix()})
        self.menu(["2", "0"])
        self.assertTrue(old.exists() and newest.exists())
        (self.root / "state/diagnostic_evidence.json").unlink()
        worker, _ = self.scanned_cleaner()
        os.utime(old, (3, 3))
        worker.clear_unused()
        self.assertTrue(old.exists() and newest.exists())

    def test_bundle_validation_is_linear_in_archive_count(self):
        from core import output_retention as retention
        self.write("tools/local_regression/manifest.json", {
            "local_regression_keep_last_n": 3, "local_regression_max_age_days": 30,
        })
        bundles = [self.bundle(str(i), i+1) for i in range(15)]
        with patch.object(retention, "is_local_regression_bundle", wraps=retention.is_local_regression_bundle) as validate:
            self.menu(["2", "0"])
        self.assertEqual(validate.call_count, len(bundles))
        self.assertEqual(sum(path.exists() for path in bundles), 3)

    def test_diagnostic_producer_registers_only_its_verified_inventory(self):
        from tools.local_regression import common
        workspace = self.root / "outputs/local_regression/_staging/completed"
        summary = self.write(workspace.relative_to(self.root) / "master_summary.json",
                             {"overall_status": "PASS", "scripts": [], "dataset": "reduced"})
        work = self.write(workspace.relative_to(self.root) / "worker.log", b"completed\n")
        inventory = self.write(workspace.relative_to(self.root) / common.ARTIFACTS_MANIFEST_FILENAME,
                               common.build_artifacts_manifest(workspace))
        extra = self.write(workspace.relative_to(self.root) / "user_added.csv", b"keep\n")
        bundle = common.build_bundle_zip(workspace, "to_chatgpt_bundle.zip", include_paths=[summary, inventory])
        with patch.object(common, "PROJECT_ROOT", self.root), patch.object(common, "OUTPUT_ROOT", workspace.parents[1]):
            archive = common.archive_bundle_history(bundle)
        record = read_lifecycle(self.root, workspace.relative_to(self.root) / MANIFEST_NAME)
        self.assertNotIn(extra.relative_to(self.root), {a.path for a in record.artifacts})
        self.menu(["2", "0"])
        self.assertTrue(work.exists())  # The publishing process is still alive.
        with patch("services.cleanup_lifecycle.process_is_absent", return_value=True):
            self.menu(["2", "0"])
        self.assertFalse(work.exists() or summary.exists() or inventory.exists())
        self.assertTrue(extra.exists() and archive.exists())

    def test_changed_inventory_never_authorizes_diagnostic_cleanup(self):
        from tools.local_regression import common
        workspace = self.root / "outputs/local_regression/_staging/changed"
        summary = self.write(workspace.relative_to(self.root) / "master_summary.json",
                             {"overall_status": "PASS", "scripts": [], "dataset": "reduced"})
        work = self.write(workspace.relative_to(self.root) / "worker.log", b"old\n")
        inventory = self.write(workspace.relative_to(self.root) / common.ARTIFACTS_MANIFEST_FILENAME,
                               common.build_artifacts_manifest(workspace))
        bundle = common.build_bundle_zip(workspace, "to_chatgpt_bundle.zip", include_paths=[summary, inventory])
        work.write_bytes(b"new\n")
        warning = io.StringIO()
        with patch.object(common, "PROJECT_ROOT", self.root), patch.object(common, "OUTPUT_ROOT", workspace.parents[1]), contextlib.redirect_stderr(warning):
            archive = common.archive_bundle_history(bundle)
        self.assertTrue(archive.exists() and work.exists())
        self.assertFalse((workspace / MANIFEST_NAME).exists())
        self.assertIn("清理依據未完成", warning.getvalue())

    def test_retention_modes_preserve_existing_both_semantics(self):
        now = datetime.now(timezone.utc)
        bundles = [self.bundle(str(i), (now-timedelta(days=40-i)).timestamp()) for i in range(4)]
        newer = self.bundle("new", now.timestamp())
        target = newer.parent
        legacy = RetentionRule("history", target, ("*.zip",), 2, 30)
        self.assertEqual(set(expired_paths(legacy, now=now)), set(bundles[:3]))
        count = RetentionRule("history", target, ("*.zip",), 1, 300, mode="count")
        self.assertEqual(set(expired_paths(count, now=now)), set(bundles))
        with self.assertRaises(ValueError):
            expired_paths(RetentionRule("bad", target, ("*",), -1, 30))


if __name__ == "__main__":
    unittest.main()
