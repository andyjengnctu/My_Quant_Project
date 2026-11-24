"""Opaque raw-cache duplicate proofs and shared writer lease boundaries."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from core.file_integrity import compute_observed_file_sha256
from services.cleanup_raw_cache import RawCacheCleanup
from services.optimizer.raw_cache import _RawCacheBuildLock, _build_named_raw_cache_signature, _build_raw_cache_paths


class RawCacheCleanupTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.changed_directories = []
        self.worker = SimpleNamespace(
            apply=True, hash_file=compute_observed_file_sha256,
            lifecycle=SimpleNamespace(removed=self.changed_directories.append),
        )

    def cache(self, output, *, data=b"opaque cache bytes", source="same"):
        paths = _build_raw_cache_paths(self.root / output, "external", 360)
        paths["cache_dir"].mkdir(parents=True, exist_ok=True)
        signature, payload = _build_named_raw_cache_signature(
            {"data_root": str(self.root), "source": source}, 360, cache_namespace="external",
        )
        meta = {"schema_version": 2, "signature": signature, "profile_key": "external",
                "data_dir": str(self.root), "required_min_rows": 360, "ticker_count": 3,
                "signature_payload": payload}
        paths["payload_path"].write_bytes(data)
        paths["meta_path"].write_text(json.dumps(meta))
        return paths

    def prepared(self, *paths):
        cleanup = RawCacheCleanup(self.root)
        cleanup.prepare({item["payload_path"].relative_to(self.root): item["payload_path"].stat().st_size
                         for item in paths})
        return cleanup

    def test_identical_execution_copy_can_be_removed_without_loading_pickle(self):
        ordinary = self.cache("outputs/portfolio_sim")
        retained = self.cache("outputs/trading/optimizer/promotion_releases/any_release")
        cleanup = self.prepared(ordinary, retained)
        relative = ordinary["payload_path"].relative_to(self.root)
        self.assertTrue(cleanup.candidate(relative))
        self.assertFalse(cleanup.candidate(retained["payload_path"].relative_to(self.root)))
        with patch("pickle.load", side_effect=AssertionError("unpickle forbidden")), \
                patch("pickle.loads", side_effect=AssertionError("unpickle forbidden")):
            with cleanup.deletion_lease(self.worker, relative):
                self.assertTrue(cleanup.proof(self.worker, relative))
                cleanup.verify_before_delete(self.worker, relative)
                ordinary["payload_path"].unlink()
        self.assertEqual(retained["payload_path"].read_bytes(), b"opaque cache bytes")
        self.assertTrue(ordinary["meta_path"].exists())
        self.assertFalse(ordinary["lock_dir"].exists() or retained["lock_dir"].exists())
        self.assertEqual(len(self.changed_directories), 2)

    def test_generic_groups_keep_one_original_and_never_clear_unique_history(self):
        first = self.cache("outputs/example_a")
        second = self.cache("outputs/example_b")
        unique = self.cache("outputs/trading/optimizer", source="older immutable source")
        cleanup = self.prepared(first, second, unique)
        self.assertEqual(cleanup.sources, {
            second["payload_path"].relative_to(self.root): first["payload_path"].relative_to(self.root),
        })

    def test_metadata_alone_cannot_authorize_different_payload_bytes(self):
        ordinary = self.cache("outputs/portfolio_sim", data=b"different payload")
        retained = self.cache("outputs/trading/optimizer", data=b"original payload!")
        cleanup = self.prepared(ordinary, retained)
        relative = ordinary["payload_path"].relative_to(self.root)
        with cleanup.deletion_lease(self.worker, relative):
            self.assertEqual(cleanup.proof(self.worker, relative), "")
            with self.assertRaises(ValueError):
                cleanup.verify_before_delete(self.worker, relative)
        self.assertTrue(ordinary["payload_path"].exists())

    def test_incremental_proof_properties_keep_original_revisions_and_both_cache_scopes(self):
        ordinary = self.cache("outputs/portfolio_sim")
        retained = self.cache("outputs/trading/optimizer/promotion_releases/any_release")
        unrelated = self.cache("outputs/trading/optimizer", source="different source")
        cleanup = self.prepared(ordinary, retained, unrelated)
        relative = ordinary["payload_path"].relative_to(self.root)
        original = cleanup.proof_revisions
        self.assertEqual(set(original), {ordinary["meta_path"], retained["meta_path"]})
        self.assertEqual(cleanup.proof_scopes, {
            ordinary["cache_dir"].relative_to(self.root), retained["cache_dir"].relative_to(self.root)})
        with patch("services.cleanup_raw_cache.file_change_signature", side_effect=AssertionError("properties must not refresh authority")):
            cleanup.proof_revisions.clear()
            self.assertEqual(cleanup.proof_revisions, original)
        retained["meta_path"].write_text("{}")
        self.assertEqual(cleanup.proof_revisions, original)
        with cleanup.deletion_lease(self.worker, relative):
            with self.assertRaises(ValueError):
                cleanup.proof(self.worker, relative)
        self.assertTrue(ordinary["payload_path"].exists())

    def test_unrecognized_schema_or_signature_keeps_the_cache(self):
        ordinary = self.cache("outputs/portfolio_sim")
        retained = self.cache("outputs/trading/optimizer")
        payload = json.loads(ordinary["meta_path"].read_text())
        payload["signature_payload"]["source_identity"]["source"] = "changed"
        ordinary["meta_path"].write_text(json.dumps(payload))
        cleanup = self.prepared(ordinary, retained)
        self.assertFalse(cleanup.candidate(ordinary["payload_path"].relative_to(self.root)))
        self.assertIn(ordinary["payload_path"].relative_to(self.root), cleanup.errors)

    def test_source_or_metadata_changes_after_proof_block_removal(self):
        for changed in ("payload_path", "meta_path"):
            with self.subTest(changed=changed):
                ordinary = self.cache("outputs/portfolio_sim")
                retained = self.cache("outputs/trading/optimizer")
                cleanup = self.prepared(ordinary, retained)
                relative = ordinary["payload_path"].relative_to(self.root)
                with cleanup.deletion_lease(self.worker, relative):
                    self.assertTrue(cleanup.proof(self.worker, relative))
                    retained[changed].write_bytes(b"changed")
                    with self.assertRaises(ValueError):
                        cleanup.verify_before_delete(self.worker, relative)
                self.assertTrue(ordinary["payload_path"].exists())

    def test_active_old_writer_is_never_stolen_by_cleaner_or_writer(self):
        ordinary = self.cache("outputs/portfolio_sim")
        retained = self.cache("outputs/trading/optimizer")
        cleanup = self.prepared(ordinary, retained)
        relative = ordinary["payload_path"].relative_to(self.root)
        with _RawCacheBuildLock(retained) as owner:
            os.utime(retained["lock_dir"], (1, 1))
            owner._remove_stale_lock_if_needed(0)
            self.assertTrue(owner.owner_path.exists())
            with self.assertRaises(RuntimeError):
                with cleanup.deletion_lease(self.worker, relative):
                    self.fail("active writer lock was stolen")
            self.assertTrue(owner.owner_path.exists())
        self.assertTrue(ordinary["payload_path"].exists())

    def test_stale_recovery_requires_positive_process_absence(self):
        paths = self.cache("outputs/portfolio_sim")
        lock = _RawCacheBuildLock(paths)
        paths["lock_dir"].mkdir()
        lock.owner_path.write_text(json.dumps({"pid": os.getpid(), "created_at": 1}))
        os.utime(paths["lock_dir"], (1, 1))
        with patch("core.artifact_lifecycle.process_is_absent", return_value=False):
            lock._remove_stale_lock_if_needed(0)
        self.assertTrue(paths["lock_dir"].exists())
        with patch("core.artifact_lifecycle.process_is_absent", return_value=True):
            lock._remove_stale_lock_if_needed(0)
        self.assertFalse(paths["lock_dir"].exists())

    def test_preview_has_no_lock_filesystem_mutations(self):
        ordinary = self.cache("outputs/portfolio_sim")
        retained = self.cache("outputs/trading/optimizer")
        cleanup = self.prepared(ordinary, retained)
        relative = ordinary["payload_path"].relative_to(self.root)
        self.worker.apply = False
        before = set(self.root.rglob("*"))
        with cleanup.deletion_lease(self.worker, relative):
            self.assertTrue(cleanup.proof(self.worker, relative))
        self.assertEqual(set(self.root.rglob("*")), before)
        self.assertEqual(self.changed_directories, [])

    def test_real_cache_miss_rebuilds_through_the_producer_after_duplicate_cleanup(self):
        import pandas as pd
        from services.optimizer.raw_cache import load_or_build_optimizer_persisted_raw_cache
        frame = pd.DataFrame({"Date": ["2020-01-02"], "Close": [10.0]})
        built = []

        def builder():
            built.append(True)
            return {"1234": frame}, [], {}

        options = dict(cache_namespace="external", required_min_rows=360,
                       source_identity={"data_root": str(self.root), "source": "stable fixture"},
                       builder=builder, verbose=False)
        paths = []
        for output in ("outputs/portfolio_sim", "outputs/trading/optimizer"):
            directory = self.root / output
            load_or_build_optimizer_persisted_raw_cache(output_dir=directory, **options)
            paths.append(_build_raw_cache_paths(directory, "external", 360))
        ordinary, retained = paths
        cleanup = self.prepared(*paths)
        relative = ordinary["payload_path"].relative_to(self.root)
        with cleanup.deletion_lease(self.worker, relative):
            self.assertTrue(cleanup.proof(self.worker, relative))
            ordinary["payload_path"].unlink()
        after = load_or_build_optimizer_persisted_raw_cache(
            output_dir=self.root / "outputs/portfolio_sim", **options,
        )
        self.assertEqual(len(built), 3)
        pd.testing.assert_frame_equal(after["1234"], frame)
        self.assertTrue(retained["payload_path"].exists())

    def lifecycle_scope(self, cleanup):
        from services.cleanup_lifecycle import LifecycleCleanup
        from services.project_cleaner import walk
        scope = LifecycleCleanup(self.root)
        for relative, size in walk(self.root, lambda *args: None, scope.observe_directory):
            scope.observe(relative, size)
        scope.extra_targets.update(cleanup.reference_targets)
        scope.prepare(set())
        return scope

    def test_external_exact_file_or_cache_directory_reference_pins_duplicates(self):
        ordinary = self.cache("outputs/portfolio_sim")
        retained = self.cache("outputs/trading/optimizer")
        cleanup = self.prepared(ordinary, retained)
        relative = ordinary["payload_path"].relative_to(self.root)
        reference = self.root / "state/consumer.json"
        reference.parent.mkdir()
        for target in (relative, relative.parent):
            with self.subTest(target=target):
                reference.write_text(json.dumps({"input": target.as_posix()}))
                self.assertTrue(self.lifecycle_scope(cleanup).protected(relative))
        reference.write_text(json.dumps({"output_root": "outputs"}))
        self.assertFalse(self.lifecycle_scope(cleanup).protected(relative))

    def test_new_reference_or_malformed_evidence_blocks_extra_target_session(self):
        ordinary = self.cache("outputs/portfolio_sim")
        retained = self.cache("outputs/trading/optimizer")
        cleanup = self.prepared(ordinary, retained)
        relative = ordinary["payload_path"].relative_to(self.root)
        reference = self.root / "state/consumer.json"
        reference.parent.mkdir()
        scope = self.lifecycle_scope(cleanup)
        reference.write_text(json.dumps({"input": relative.as_posix()}))
        self.worker.apply = False
        with self.assertRaises(ValueError):
            with scope.session(self.worker):
                self.fail("new reference was ignored")
        reference.write_text("{broken json")
        scope = self.lifecycle_scope(cleanup)
        with self.assertRaises(ValueError):
            with scope.session(self.worker):
                self.fail("malformed dependency evidence was ignored")


if __name__ == "__main__":
    unittest.main()
