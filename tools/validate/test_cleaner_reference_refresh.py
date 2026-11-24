"""Concurrent reference updates never become new cleanup authority."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from core.artifact_lifecycle import MANIFEST_NAME
from core.file_integrity import file_change_signature
from services import cleanup_lifecycle
from tools.validate import test_artifact_lifecycle as lifecycle_tests


class ReferenceRefreshTest(unittest.TestCase):
    setUp = lifecycle_tests.LifecycleCleanupTest.setUp
    git = lifecycle_tests.LifecycleCleanupTest.git
    write = lifecycle_tests.LifecycleCleanupTest.write
    scanned_cleaner = lifecycle_tests.LifecycleCleanupTest.scanned_cleaner
    storage = lifecycle_tests.LifecycleCleanupTest.storage

    def clear(self, worker):
        with contextlib.redirect_stdout(io.StringIO()):
            worker.clear_unused()

    def test_cleaner_refreshes_unrelated_json_and_nested_batch_inventory(self):
        _, files, result, marker = self.storage(roles={"one.csv": "work", "two.csv": "work"})
        status = self.write("state/another_application/status.json", {"requests": 1})
        stable = {p: p.read_bytes() for p in (result, marker)}
        worker, _ = self.scanned_cleaner()
        status.write_text(json.dumps({"requests": 123, "result": "ready"}))
        batch = self.write("data/another_application/batches/new/batch_manifest.json",
                           {"schema_version": 7, "request_ids": ["unrelated"]})
        database = self.write("data/another_application/batches/new/events.sqlite", b"database bytes")
        self.clear(worker)
        self.assertTrue(all(not p.exists() for p in files.values()))
        self.assertEqual(stable, {p: p.read_bytes() for p in stable})
        self.assertTrue(batch.exists() and database.exists())
        # A later phase accepts the revised external inventory and our own unlinks.
        with worker.lifecycle.session(worker):
            pass

    def test_changed_external_reference_preserves_candidate(self):
        workspace, files, _, _ = self.storage()
        status = self.write("state/consumer/status.json", {"inputs": []})
        worker, _ = self.scanned_cleaner()
        status.write_text(json.dumps({"inputs": [(workspace / "scratch.csv").as_posix()]}))
        self.clear(worker)
        self.assertTrue(files["scratch.csv"].exists())
        self.assertTrue(worker.lifecycle.protected(workspace / "scratch.csv"))

    def test_new_nested_json_reference_preserves_candidate(self):
        workspace, files, _, _ = self.storage()
        worker, _ = self.scanned_cleaner()
        self.write("data/new_consumer/batches/new/batch_manifest.json",
                   {"inputs": {workspace.as_posix(): "used"}})
        self.clear(worker)
        self.assertTrue(files["scratch.csv"].exists())
        self.assertTrue(worker.lifecycle.protected(workspace / "scratch.csv"))

    def test_removing_external_reference_never_releases_old_dependency(self):
        workspace, files, _, _ = self.storage()
        status = self.write("state/consumer/status.json", {"input": workspace.as_posix()})
        worker, _ = self.scanned_cleaner()
        status.write_text(json.dumps({"inputs": []}))
        self.clear(worker)
        self.assertTrue(files["scratch.csv"].exists())
        status.unlink()
        with worker.lifecycle.session(worker):
            self.assertTrue(worker.lifecycle.protected(workspace / "scratch.csv"))

    def test_publication_and_lifecycle_marker_changes_stay_strict(self):
        for change in ("publication", "marker"):
            with self.subTest(change=change):
                _, files, result, marker = self.storage(change)
                worker, _ = self.scanned_cleaner()
                target = result if change == "publication" else marker
                target.write_text(target.read_text() + " ")
                self.clear(worker)
                self.assertTrue(files["scratch.csv"].exists())

    def test_adapter_proof_snapshot_is_not_refreshed(self):
        _, files, _, _ = self.storage()
        source = self.write("state/owner/source.json", {"version": 1})
        worker, _ = self.scanned_cleaner()
        saved = file_change_signature(source)
        worker.storage_adapters["fixture"] = SimpleNamespace(
            proof_revisions={source: saved}, proof_scopes=set())
        source.write_text(json.dumps({"version": 2}))
        self.clear(worker)
        self.assertTrue(files["scratch.csv"].exists())
        self.assertEqual(worker.storage_adapters["fixture"].proof_revisions[source], saved)

    def test_late_observation_in_adapter_scope_stays_strict(self):
        _, files, _, _ = self.storage()
        source = self.write("outputs/owner/oldrun/work/results/existing.json", {"metric": 1})
        worker, _ = self.scanned_cleaner()
        worker.storage_adapters["fixture"] = SimpleNamespace(
            proof_revisions={source: file_change_signature(source)},
            proof_scopes={Path("outputs/owner/oldrun")})
        self.write("outputs/owner/oldrun/work/results/arm_seed.json", {"metric": 2})
        self.clear(worker)
        self.assertTrue(files["scratch.csv"].exists())

    def test_changed_configuration_identity_with_no_path_stays_strict(self):
        for extension, initial, updated in (
                ("json", {"active": "profile-a"}, {"active": "profile-b"}),
                ("py", b"ACTIVE = 'profile-a'", b"ACTIVE = 'profile-b'")):
            with self.subTest(extension=extension):
                _, files, _, _ = self.storage(extension)
                config = self.write("config/current." + extension, initial)
                worker, _ = self.scanned_cleaner()
                self.write(config.relative_to(self.root), updated)
                self.clear(worker)
                self.assertTrue(files["scratch.csv"].exists())

    def test_new_and_modified_incomplete_json_fail_closed(self):
        for existing in (False, True):
            with self.subTest(existing=existing):
                _, files, _, _ = self.storage(str(existing))
                path = Path("state/worker") / str(existing) / "status.json"
                if existing:
                    self.write(path, {"requests": 1})
                worker, _ = self.scanned_cleaner()
                self.write(path, b'{"incomplete":')
                self.clear(worker)
                self.assertTrue(files["scratch.csv"].exists())
                # Leave a valid external record for the next independent case.
                self.write(path, {"requests": 2})

    def test_new_unknown_owner_or_lifecycle_declaration_is_not_authority(self):
        for name, payload in ((MANIFEST_NAME, {"schema": "unknown"}),
                              ("manifest.json", {"unknown_owner": True}),
                              ("state.json", {"contract": {"model_artifact_fingerprint": "old"}})):
            with self.subTest(name=name):
                _, files, _, _ = self.storage(name)
                worker, _ = self.scanned_cleaner()
                added = self.write(Path("outputs/new_owner") / name, payload)
                self.clear(worker)
                self.assertTrue(files["scratch.csv"].exists())
                added.unlink()

    def test_reference_changing_during_read_fails_closed(self):
        _, files, _, _ = self.storage()
        status = self.write("state/worker/status.json", {"requests": 1})
        worker, _ = self.scanned_cleaner()
        status.write_text(json.dumps({"requests": 2}))
        original = Path.read_text

        def changing(path, *args, **kwargs):
            value = original(path, *args, **kwargs)
            if path == status:
                path.write_text(json.dumps({"requests": 3333}))
            return value

        with patch.object(Path, "read_text", changing):
            self.clear(worker)
        self.assertTrue(files["scratch.csv"].exists())

    def test_later_json_read_cannot_hide_change_to_already_read_reference(self):
        workspace, files, _, _ = self.storage()
        first = self.write("state/worker/first.json", {"requests": 1})
        second = self.write("state/worker/second.json", {"requests": 1})
        worker, _ = self.scanned_cleaner()
        first.write_text(json.dumps({"requests": 2}))
        second.write_text(json.dumps({"requests": 2}))
        original = Path.read_text
        read = []

        def changing(path, *args, **kwargs):
            value = original(path, *args, **kwargs)
            if path in (first, second):
                if read:
                    read[0].write_text(json.dumps({"input": workspace.as_posix()}))
                read.append(path)
            return value

        with patch.object(Path, "read_text", changing):
            self.clear(worker)
        self.assertEqual(len(read), 2)
        self.assertTrue(files["scratch.csv"].exists())
        # The next phase must reread the changed record and retain its target.
        self.clear(worker)
        self.assertTrue(worker.lifecycle.protected(workspace / "scratch.csv"))

    def test_new_json_added_after_directory_scan_is_detected_by_final_pass(self):
        workspace, files, _, _ = self.storage()
        status = self.write("state/worker/status.json", {"requests": 1})
        worker, _ = self.scanned_cleaner()
        status.write_text(json.dumps({"requests": 2}))
        original = Path.read_text

        def changing(path, *args, **kwargs):
            value = original(path, *args, **kwargs)
            if path == status:
                self.write("state/worker/late.json", {"input": workspace.as_posix()})
            return value

        with patch.object(Path, "read_text", changing):
            self.clear(worker)
        self.assertTrue(files["scratch.csv"].exists())

    def test_failed_inventory_refresh_cannot_commit_unparsed_new_json(self):
        workspace, files, _, _ = self.storage()
        self.write("data/first/existing.txt", b"one")
        self.write("data/second/existing.txt", b"two")
        worker, _ = self.scanned_cleaner()
        first = Path("data/first")
        second = Path("data/second")
        new_reference = first / "batch_manifest.json"
        self.write(new_reference, {"input": workspace.as_posix()})
        self.write(second / "new.txt", b"new")
        # Process first before the deliberately failing second directory.
        for relative in (second, first):
            stamp = worker.lifecycle.directory_revisions.pop(relative)
            worker.lifecycle.directory_revisions[relative] = stamp
        original = os.scandir

        def failing(path):
            if Path(path) == self.root / second:
                raise OSError("temporary listing failure")
            return original(path)

        with patch.object(os, "scandir", failing):
            self.clear(worker)
        self.assertNotIn(new_reference, worker.lifecycle.files)
        self.clear(worker)
        self.assertTrue(files["scratch.csv"].exists())
        self.assertTrue(worker.lifecycle.protected(workspace / "scratch.csv"))

    def test_interrupted_reference_stat_preserves_unparsed_new_json_for_retry(self):
        for references_candidate in (True, False):
            with self.subTest(references_candidate=references_candidate):
                workspace, files, _, _ = self.storage("pending-" + str(references_candidate))
                status = self.write("state/pending/status.json", {"requests": 0})
                worker, _ = self.scanned_cleaner()
                relative = Path("data/new_consumer") / str(references_candidate) / "batch_manifest.json"
                new_record = self.write(relative, {"input": workspace.as_posix()
                                                  if references_candidate else "unrelated"})

                def interrupted(path):
                    if Path(path) == status:
                        raise OSError("temporary stat failure after directory commit")
                    return file_change_signature(path)

                with patch.object(cleanup_lifecycle, "file_change_signature", interrupted):
                    self.clear(worker)
                self.assertIn(relative, worker.lifecycle.files)
                self.assertNotIn(relative, worker.lifecycle.reference_revisions)
                self.assertIn(relative, worker.lifecycle.pending_reference_paths)
                self.assertTrue(files["scratch.csv"].exists())
                self.clear(worker)
                self.assertEqual(files["scratch.csv"].exists(), references_candidate)
                self.assertEqual(worker.lifecycle.protected(workspace / "scratch.csv"), references_candidate)
                self.assertIn(relative, worker.lifecycle.reference_revisions)
                self.assertFalse(worker.lifecycle.pending_reference_paths)
                self.assertTrue(new_record.exists())

    def test_new_directory_link_is_not_traversed(self):
        _, files, _, _ = self.storage()
        target = self.root / "external_target"
        target.mkdir()
        (self.root / "data").mkdir()
        worker, _ = self.scanned_cleaner()
        link = self.root / "data" / "new_consumer"
        if os.name == "nt":
            result = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                                    capture_output=True)
            if result.returncode:
                self.skipTest("junction unavailable")
        else:
            link.symlink_to(target, target_is_directory=True)
        try:
            self.clear(worker)
            self.assertTrue(files["scratch.csv"].exists())
        finally:
            if os.name == "nt":
                link.rmdir()
            else:
                link.unlink()


if __name__ == "__main__":
    unittest.main()
