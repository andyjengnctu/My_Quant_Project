"""Unfinished copies may lose duplicate bytes without acquiring scientific status."""
from contextlib import contextmanager
import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from core import reused_copy_storage as storage
from core.compare_csv_storage import comparison_publication_lock, detail_read_path
from core.file_integrity import compute_observed_file_sha256, file_change_signature
from services.research.strategy_compare_pair_integrity import (
    build_completed_pair_file_integrity, completed_strategy_run_publication_matches,
)


class ReusedCopyStorageTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.base = self.root / "outputs/strategy_compare/future/profile/runs"
        self.filename = "score_ranking_orderable_candidates.csv"
        self.data = b"ticker,value\nTEST,1\n"

    def write(self, path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8"))
        return path

    def pair(self, run, source=None):
        pair = self.base / run / "pairs/arm"
        if source is None:
            payload = {"metadata": {"policy": "same scientific identity", "output_scope": "comparison"},
                       "no_filter": {"return_pct": 1}, "score_ranking": {"return_pct": 2}}
        else:
            payload = json.loads((source.parent / "strategy_comparison.json").read_text())
            payload["metadata"].update(output_scope="reused_pair_cache",
                cache_reused_from=source.parent.relative_to(self.root).as_posix())
        payload["metadata"]["output_dir"] = pair.relative_to(self.root).as_posix()
        self.write(pair / "strategy_comparison.json", payload)
        self.write(pair / "strategy_comparison.md", b"report\n")
        return self.write(pair / self.filename, self.data)

    def complete(self, path, *, pinned=True):
        pair = json.loads((path.parent / "strategy_comparison.json").read_text())
        run = path.parents[2]
        payload = {"status": "COMPLETED", "config_fingerprint": "0123456789ab", "pairs": {"arm": pair}}
        manifest = {"config_fingerprint": payload["config_fingerprint"], "run_dir": run.relative_to(self.root).as_posix()}
        if pinned:
            record = build_completed_pair_file_integrity({"arm": path.parent})
            payload["pair_file_integrity"] = manifest["pair_file_integrity"] = record
        self.write(run / "strategy_comparison.json", payload)
        self.write(run / "manifest.json", manifest)
        return path

    def fixture(self, *, pinned=True):
        source = self.complete(self.pair("original"), pinned=pinned)
        return source, self.pair("unfinished", source)

    def test_legacy_and_pinned_sources_get_storage_receipt_without_completion(self):
        for pinned in (False, True):
            with self.subTest(pinned=pinned):
                source = self.complete(self.pair(f"source-{pinned}"), pinned=pinned)
                target = self.pair(f"copy-{pinned}", source)
                scientific = {p: p.read_bytes() for p in self.root.rglob("*.json")}
                revisions, digest = storage.publication_evidence(target)
                self.assertEqual(revisions[target.parents[2] / "manifest.json"], ("missing",))
                self.assertEqual(digest, compute_observed_file_sha256(target))
                self.assertFalse(storage.receipt_path(target).exists())
                storage.record_alias(target, source, digest)
                self.assertTrue(all(path.read_bytes() == data for path, data in scientific.items()))
                self.assertFalse((target.parents[2] / "manifest.json").exists())
                self.assertFalse((target.parents[2] / "strategy_comparison.json").exists())
                self.assertNotEqual(storage.receipt_path(target).parent, target.parent)
                target.unlink()
                self.assertEqual(detail_read_path(target).read_bytes(), self.data)
                source.write_bytes(b"!" * len(self.data))
                with self.assertRaises(ValueError):
                    detail_read_path(target)

    def test_partial_or_malformed_target_publication_never_uses_unfinished_proof(self):
        source, target = self.fixture()
        for name in ("manifest.json", "strategy_comparison.json"):
            marker = target.parents[2] / name
            for value in (b"not json", {"status": "COMPLETED"}):
                with self.subTest(marker=name, value=value):
                    self.write(marker, value)
                    with self.assertRaisesRegex(ValueError, "發布紀錄"):
                        storage.publication_evidence(target)
                    marker.unlink()
        self.assertTrue(target.exists())
        self.assertFalse(storage.receipt_path(target).exists())

    def test_only_three_producer_location_fields_can_differ(self):
        _, target = self.fixture()
        marker = target.parent / "strategy_comparison.json"
        original = json.loads(marker.read_text())
        mutations = [
            lambda p: p["metadata"].update(output_dir="wrong"),
            lambda p: p["metadata"].update(output_scope="caller_override"),
            lambda p: p["metadata"].update(cache_reused_from="../outside"),
            lambda p: p["metadata"].update(cache_reused_from=str(self.base)),
            lambda p: p["metadata"].update(new_unknown_identity="changed"),
            lambda p: p["score_ranking"].update(return_pct=999),
        ]
        for mutate in mutations:
            payload = deepcopy(original)
            mutate(payload)
            self.write(marker, payload)
            with self.assertRaises(ValueError):
                storage.publication_evidence(target)
        self.write(marker, original)
        storage.publication_evidence(target)

    def test_source_manifest_conflict_and_changed_pinned_bytes_preserve_target(self):
        source, target = self.fixture()
        marker = source.parents[2] / "manifest.json"
        original = marker.read_bytes()
        payload = json.loads(original)
        payload["config_fingerprint"] = "different123"
        self.write(marker, payload)
        with self.assertRaisesRegex(ValueError, "config_fingerprint"):
            storage.publication_evidence(target)
        marker.write_bytes(original)
        source.write_bytes(b"changed but equal\n")
        target.write_bytes(source.read_bytes())
        with self.assertRaisesRegex(ValueError, "SHA256"):
            storage.publication_evidence(target)
        self.assertTrue(target.exists())

    def test_already_removed_intermediate_source_resolves_to_committed_original(self):
        source = self.complete(self.pair("original"), pinned=True)
        middle = self.complete(self.pair("middle", source), pinned=True)
        target = self.pair("unfinished", middle)
        middle.unlink()
        revisions, digest = storage.publication_evidence(target)
        self.assertIn(middle.parent / "strategy_comparison.json", revisions)
        storage.record_alias(target, source, digest)
        target.unlink()
        self.assertEqual(detail_read_path(target), source)

    def test_unrelated_equal_file_cannot_be_recorded_as_source(self):
        source, target = self.fixture()
        unrelated = self.complete(self.pair("unrelated"))
        digest = compute_observed_file_sha256(source)
        with self.assertRaisesRegex(ValueError, "來源鏈"):
            storage.record_alias(target, unrelated, digest)
        self.assertFalse(storage.receipt_path(target).exists())

    def test_marker_change_after_hash_and_partial_commit_race_are_detected(self):
        _, target = self.fixture()
        marker = target.parents[2] / "manifest.json"
        def mutate_after_hash(path):
            digest = compute_observed_file_sha256(path)
            if path == target:
                self.write(marker, {"in_progress": True})
            return digest
        with self.assertRaisesRegex(ValueError, "驗證後變動"):
            storage.publication_evidence(target, file_hasher=mutate_after_hash)
        self.assertTrue(target.exists())

    def test_later_real_completion_hashes_inherited_details_without_new_scientific_fields(self):
        source, target = self.fixture(pinned=False)
        _, digest = storage.publication_evidence(target)
        storage.record_alias(target, source, digest)
        target.unlink()
        marker = target.parent / "strategy_comparison.json"
        payload = json.loads(marker.read_text())
        payload["benchmark_yearly"] = [{"year": 2020, "return_pct": 3}]
        self.write(marker, payload)
        self.complete(target)
        run = target.parents[2]
        result = json.loads((run / "strategy_comparison.json").read_text())
        self.assertEqual(result["pair_file_integrity"]["pairs"]["arm"][self.filename], digest)
        self.assertTrue(completed_strategy_run_publication_matches(
            run_dir=run, run_payload=result, project_root=self.root))
        self.assertEqual(detail_read_path(target).read_bytes(), self.data)

    def test_corrupt_or_retargeted_storage_receipt_is_not_silently_ignored(self):
        source, target = self.fixture(pinned=False)
        _, digest = storage.publication_evidence(target)
        storage.record_alias(target, source, digest)
        target.unlink()
        receipt = storage.receipt_path(target)
        saved = receipt.read_bytes()
        for value in (b"bad json", {**json.loads(saved), "target": "wrong"},
                      {**json.loads(saved), "source": "../outside.csv"},
                      {**json.loads(saved), "sha256": "f" * 64}):
            self.write(receipt, value)
            with self.assertRaises(ValueError):
                detail_read_path(target)
        receipt.write_bytes(saved)
        self.assertEqual(detail_read_path(target), source)

    def test_pair_and_run_marker_commits_share_csv_cleanup_lock(self):
        entered = []
        @contextmanager
        def lock(root):
            entered.append(root)
            yield
        for marker in (self.base / "new/pairs/arm/strategy_comparison.json",
                       self.base / "new/strategy_comparison.json", self.base / "new/manifest.json"):
            with patch("core.compare_csv_storage.storage_lock", lock):
                with comparison_publication_lock(marker):
                    self.assertEqual(entered[-1], self.root)
        self.assertEqual(len(entered), 3)

    def test_cleaner_preview_is_read_only_then_apply_records_alias_and_preserves_unfinished_state(self):
        from apps import cleaner
        source, target = self.fixture(pinned=False)
        self.write(self.root / "doc/PROJECT_SETTINGS.md", b"fixture")
        subprocess.run(["git", "-C", str(self.root), "init", "-q"], check=True, capture_output=True)
        scientific = {p: p.read_bytes() for p in self.base.rglob("*.json")}
        for answer in ("1", "2"):
            with patch.object(cleaner, "ROOT", self.root), patch("builtins.input", side_effect=[answer, "0"]), contextlib.redirect_stdout(io.StringIO()):
                cleaner.main()
            self.assertTrue(source.exists())
            self.assertTrue(all(p.read_bytes() == data for p, data in scientific.items()))
            self.assertFalse((target.parents[2] / "manifest.json").exists())
            self.assertFalse((target.parents[2] / "strategy_comparison.json").exists())
            self.assertEqual(target.exists(), answer == "1")
            self.assertEqual(storage.receipt_path(target).exists(), answer == "2")
        self.assertEqual(detail_read_path(target).read_bytes(), self.data)

    def test_actual_json_owner_publishes_inside_shared_lock_with_identical_bytes(self):
        from services.research.strategy_comparison import _write_json
        from core.file_integrity import text_publication_bytes
        marker = self.base / "new/manifest.json"
        payload = {"unicode": "保留原始內容", "value": [1, 2]}
        active = []
        @contextmanager
        def lock(root):
            self.assertEqual(root, self.root)
            active.append(True)
            yield
            active.pop()
        from services.research import strategy_comparison as owner
        real_write = owner.atomic_write_json
        def observed_write(path, value):
            self.assertEqual(active, [True])
            return real_write(path, value)
        with patch("core.compare_csv_storage.storage_lock", lock), patch.object(owner, "atomic_write_json", observed_write):
            _write_json(marker, payload)
        self.assertEqual(marker.read_bytes(), text_publication_bytes(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"))


if __name__ == "__main__":
    unittest.main()
