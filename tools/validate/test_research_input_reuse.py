"""Focused input-reuse regressions; runnable independently with unittest."""
from __future__ import annotations

import ctypes
import gc
import json
import os
from pathlib import Path
import tempfile
import unittest
import weakref
from unittest.mock import patch

import numpy as np
import pandas as pd

from core import file_integrity as integrity
from filters.breakout_quality import dataset_store, source_inventory, workflow_io as io


def make_dataset(directory: Path, *, groups: int = 12, window: int = 8):
    directory.mkdir(parents=True, exist_ok=True)
    paths = dataset_store.resolve_dataset_paths(directory)
    policy = io.DEFAULT_LABEL_POLICY.as_manifest_payload()
    policy.update(feature_window_bars=window, label_path_cache_bars=4)
    arrays = {
        "feature_bank": np.ones((groups, window, len(io.FEATURE_COLUMNS)), dtype=np.float32),
        "event_context": np.zeros((groups, len(io.CONTEXT_COLUMNS)), dtype=np.float32),
        "event_labels": np.full(groups, io.LABEL_PASS, dtype=np.int8),
        "event_group_index": np.arange(groups, dtype=np.int64),
        "group_anchor_prices": np.full(groups, 10.0),
        "future_high_prices": np.full((groups, 4), 12.0),
        "future_low_prices": np.full((groups, 4), 9.0),
        "future_available_bars": np.full(groups, 4, dtype=np.int32),
        "future_date_ordinals": np.full((groups, 4), 20000, dtype=np.int32),
    }
    for name, target in paths.array_paths().items():
        np.save(target, arrays[name], allow_pickle=False)
    events = pd.DataFrame({
        "ticker": ["0050"] * groups,
        "date": pd.date_range("2013-04-01", periods=groups).strftime("%Y-%m-%d"),
        "high_len": policy["high_len_values"][0],
        "group_index": np.arange(groups), "label": io.LABEL_PASS,
        "label_eval_end_date": "2020-12-31",
    })
    events.to_csv(paths.events, index=False)
    summary = {
        "filter_id": "fixture", "dataset": "reduced",
        "dataset_storage_schema_version": dataset_store.DATASET_STORAGE_SCHEMA_VERSION,
        "dataset_storage_format": dataset_store.DATASET_STORAGE_FORMAT,
        "feature_columns": list(io.FEATURE_COLUMNS), "context_columns": list(io.CONTEXT_COLUMNS),
        "policy": policy, "label_policy": io.label_manifest_payload_from_policy_manifest(policy),
        "dataset_artifacts": {
            name: {"filename": path.name, "size_bytes": path.stat().st_size,
                   "sha256": integrity.compute_file_sha256(path)}
            for name, path in paths.artifact_paths().items()
        },
        "event_count": groups, "feature_group_count": groups,
        "label_counts": io.label_counts(arrays["event_labels"]),
        "source_data_inventory": {"fixture_generation": 1},
    }
    paths.summary.write_text(json.dumps(summary), encoding="utf-8")
    return paths, summary


class DatasetReuseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.paths, self.summary = make_dataset(Path(self.tmp.name))
        io._load_validated_dataset_payload.cache_clear()
        dataset_store._revision_verified_sha256.cache_clear()
        self.addCleanup(io._load_validated_dataset_payload.cache_clear)
        self.addCleanup(dataset_store._revision_verified_sha256.cache_clear)
        self.mapping = patch.object(io, "dataset_paths", return_value=self.paths)
        self.mapping.start()
        self.addCleanup(self.mapping.stop)

    def load(self, **kwargs):
        return io.load_validated_dataset_bundle("fixture", **kwargs)

    def rewrite_summary(self, summary=None):
        self.paths.summary.write_text(json.dumps(summary or self.summary), encoding="utf-8")

    def test_unchanged_load_reuses_byte_checks_and_row_validation(self):
        with patch.object(dataset_store, "compute_file_sha256", wraps=integrity.compute_file_sha256) as hashes, \
             patch.object(io, "_validate_future_paths", wraps=io._validate_future_paths) as paths, \
             patch.object(io, "read_breakout_quality_csv", wraps=io.read_breakout_quality_csv) as csv:
            first = self.load()
            for _ in range(5):
                repeat = self.load()
                self.assertEqual(first[0], repeat[0])
                np.testing.assert_array_equal(first[1][:], repeat[1][:])
                np.testing.assert_array_equal(first[2], repeat[2])
                np.testing.assert_array_equal(first[3], repeat[3])
                pd.testing.assert_frame_equal(first[4], repeat[4])
            self.assertEqual(hashes.call_count, len(self.paths.artifact_paths()))
            self.assertEqual(paths.call_count, 1)
            self.assertEqual(csv.call_count, 1)

    def test_mutable_returns_are_isolated_and_arrays_are_readonly(self):
        result = self.load()
        result[0]["policy"]["feature_window_bars"] = 999
        result[4].loc[0, "ticker"] = "changed"
        for array in (result[1].feature_bank, result[1].event_group_index, result[2], result[3]):
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array.flat[0] = 99
        again = self.load()
        self.assertEqual(again[0], self.summary)
        self.assertEqual(again[4].iloc[0]["ticker"], "0050")

    def test_cache_does_not_keep_mapped_files_open(self):
        references = []
        original = io.load_npy

        def observed(path):
            array = original(path)
            references.append(weakref.ref(array))
            return array

        with patch.object(io, "load_npy", side_effect=observed):
            data = self.load()
        del data
        gc.collect()
        self.assertTrue(all(ref() is None for ref in references))
        # Windows mapped files cannot be deleted while an extra cache holds them.
        self.paths.feature_bank.unlink()

    def test_source_freshness_is_checked_on_every_request(self):
        with patch.object(io, "build_source_data_inventory", return_value=self.summary["source_data_inventory"]) as source:
            self.load(require_current_source=True)
            self.load(require_current_source=True)
            self.assertEqual(source.call_count, 2)
            source.return_value = {"fixture_generation": 2}
            with self.assertRaisesRegex(ValueError, "來源 CSV"):
                self.load(require_current_source=True)

    def test_changed_policy_is_rejected_on_warm_cache(self):
        self.load()
        policy = dict(self.summary["policy"], feature_window_bars=999)
        with self.assertRaisesRegex(ValueError, "policy"):
            self.load(expected_policy=policy)

    def test_same_size_restored_mtime_corruption_is_rejected(self):
        self.load()
        path = self.paths.feature_bank
        before = path.stat()
        with path.open("r+b") as handle:
            handle.seek(-4, os.SEEK_END)
            handle.write(np.array([2], dtype=np.float32).tobytes())
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.load()

    def test_deleted_artifact_does_not_hit_warm_cache(self):
        self.load()
        self.paths.future_high_prices.unlink()
        with self.assertRaisesRegex(ValueError, "不存在"):
            self.load()

    def test_summary_claims_do_not_skip_validation(self):
        self.load()
        self.summary["event_count"] += 1
        self.rewrite_summary()
        with self.assertRaisesRegex(ValueError, "event_count"):
            self.load()

    def test_changed_published_hash_is_rejected(self):
        self.load()
        self.summary["dataset_artifacts"]["feature_bank"]["sha256"] = "0" * 64
        self.rewrite_summary()
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.load()

    def test_republished_valid_payload_is_reloaded(self):
        self.load()
        changed = np.full((12, len(io.CONTEXT_COLUMNS)), 3.0, dtype=np.float32)
        np.save(self.paths.event_context, changed)
        self.summary["dataset_artifacts"]["event_context"]["sha256"] = integrity.compute_file_sha256(self.paths.event_context)
        self.rewrite_summary()
        result = self.load()
        np.testing.assert_array_equal(result[2], changed)

    def test_failed_validation_is_not_cached(self):
        self.summary["event_count"] = 13
        self.rewrite_summary()
        with self.assertRaises(ValueError):
            self.load()
        self.summary["event_count"] = 12
        self.rewrite_summary()
        self.assertEqual(len(self.load()[4]), 12)

    def test_concurrent_publication_is_rejected(self):
        original = io._validate_future_paths

        def changed(*args):
            original(*args)
            self.paths.summary.write_text(self.paths.summary.read_text() + " ", encoding="utf-8")

        with patch.object(io, "_validate_future_paths", side_effect=changed):
            with self.assertRaisesRegex(RuntimeError, "changed"):
                self.load()
        self.assertEqual(io._load_validated_dataset_payload.cache_info().currsize, 0)

    def test_unknown_filter_id_is_rejected_on_warm_cache(self):
        self.load()
        with self.assertRaisesRegex(ValueError, "filter_id"):
            io.load_validated_dataset_bundle("other")


class FileObservationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        integrity._observed_file_sha256.cache_clear()
        self.addCleanup(integrity._observed_file_sha256.cache_clear)
        self.path = Path(self.tmp.name) / "input.csv"
        self.path.write_bytes(b"original")

    def test_identical_files_are_hashed_once_per_process_revision(self):
        expected = integrity.compute_file_sha256(self.path)
        with patch.object(integrity, "compute_file_sha256", wraps=integrity.compute_file_sha256) as raw:
            for _ in range(5):
                self.assertEqual(integrity.compute_observed_file_sha256(self.path), expected)
            self.assertEqual(raw.call_count, 1)

    def test_restored_mtime_and_atomic_replacement_invalidate(self):
        before = self.path.stat()
        original = integrity.compute_observed_file_sha256(self.path)
        self.path.write_bytes(b"modified")
        os.utime(self.path, ns=(before.st_atime_ns, before.st_mtime_ns))
        changed = integrity.compute_observed_file_sha256(self.path)
        self.assertNotEqual(original, changed)
        sibling = self.path.with_suffix(".new")
        sibling.write_bytes(b"replaced")
        os.utime(sibling, ns=(before.st_atime_ns, before.st_mtime_ns))
        os.replace(sibling, self.path)
        self.assertNotEqual(changed, integrity.compute_observed_file_sha256(self.path))

    def test_deleted_file_cannot_use_digest_cache(self):
        integrity.compute_observed_file_sha256(self.path)
        self.path.unlink()
        with self.assertRaises(FileNotFoundError):
            integrity.compute_observed_file_sha256(self.path)

    def test_hash_race_is_rejected_and_not_cached(self):
        original = integrity.compute_file_sha256

        def changed(path):
            digest = original(path)
            self.path.write_bytes(b"modified")
            return digest

        with patch.object(integrity, "compute_file_sha256", side_effect=changed):
            with self.assertRaisesRegex(RuntimeError, "changed"):
                integrity.compute_observed_file_sha256(self.path)
        self.assertEqual(integrity._observed_file_sha256.cache_info().currsize, 0)

    def test_windows_uses_change_time_and_closes_handle(self):
        class BasicInfo(ctypes.Structure):
            _fields_ = [("ChangeTime", ctypes.c_longlong)]

        def query(handle, level, pointer, size):
            pointer._obj.ChangeTime = 123456
            return True

        with patch.object(integrity, "_windows_file_information_api",
                          return_value=(ctypes, BasicInfo, lambda *args: 123, query, lambda handle: closed.append(handle))):
            closed = []
            self.assertEqual(integrity._windows_file_change_time_ns(self.path), 12345600)
            self.assertEqual(closed, [123])

    def test_source_witness_still_rejects_same_metadata_different_bytes(self):
        root = Path(self.tmp.name)
        inventory = {"csv_inventory_sha256": "a" * 64}
        with patch.object(source_inventory, "get_dataset_dir", return_value=str(root)), \
             patch.object(source_inventory, "discover_unique_csv_inputs", return_value=([("0050", self.path)], [])), \
             patch.object(source_inventory, "_build_source_data_inventory", return_value=inventory):
            first = source_inventory._source_content_witness(root, "reduced", inventory)
            self.assertEqual(source_inventory._source_content_witness(root, "reduced", inventory), first)
            before = self.path.stat()
            self.path.write_bytes(b"modified")
            os.utime(self.path, ns=(before.st_atime_ns, before.st_mtime_ns))
            with self.assertRaisesRegex(ValueError, "內容已變更"):
                source_inventory._source_content_witness(root, "reduced", inventory)


class FuturePathTests(unittest.TestCase):
    def test_available_prefix_nan_and_block_boundaries_preserve_first_bad_row(self):
        count = 4100
        anchor = np.ones(count)
        high = np.full((count, 4), 2.0)
        low = np.ones((count, 4))
        available = np.full(count, 2)
        high[:, 2:] = 0.0  # Invalid outside the available prefix is ignored.
        high[2, 0] = np.nan  # Original finite-pair check permits NaN.
        io._validate_future_paths(anchor, high, low, available)
        high[4097, 0] = 0.0
        with self.assertRaisesRegex(ValueError, "group_index=4097"):
            io._validate_future_paths(anchor, high, low, available)

    def test_infinity_and_out_of_range_availability_still_fail(self):
        for anchor, high, low, available in (
            (np.array([np.inf]), np.ones((1, 2)), np.ones((1, 2)), np.array([2])),
            (np.array([1.]), np.array([[np.inf, 1]]), np.ones((1, 2)), np.array([0])),
            (np.array([1.]), np.ones((1, 2)), np.ones((1, 2)), np.array([3])),
            (np.array([1.]), np.ones((1, 2)), np.ones((1, 2)), np.array([np.nan])),
        ):
            with self.subTest(available=available), self.assertRaises(ValueError):
                io._validate_future_paths(anchor, high, low, available)


if __name__ == "__main__":
    unittest.main()
