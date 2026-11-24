"""Baseline candidate provenance shares the existing detail storage contract."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from core.compare_csv_storage import (
    detail_read_path, inherited_detail_files, reused_detail_source,
)
from core.file_integrity import load_json_object_or_none
from services.research.strategy_compare_pair_integrity import (
    build_completed_pair_file_integrity, completed_pair_file_integrity_matches,
)
from tools.validate import test_project_cleaner as previous


class BaselineStorageTest(unittest.TestCase):
    git = previous.CleanupTest.git
    write = previous.CleanupTest.write
    publish = previous.CleanupTest.publish
    menu = previous.CleanupTest.menu
    scanned_cleaner = previous.CleanupTest.scanned_cleaner

    def setUp(self):
        previous.CleanupTest.setUp(self)
        self.output = "no_filter_orderable_candidates.csv"

    def baseline_copy(self, run, source, *, flag=True, data=None):
        path = self.publish(run, data=data)
        run_dir = path.parents[2]
        pair = load_json_object_or_none(path.parent / "strategy_comparison.json")
        pair["metadata"].update(
            baseline_reused=flag,
            baseline_reuse_source=source.parent.relative_to(self.root).as_posix(),
        )
        self.write(path.parent.relative_to(self.root) / "strategy_comparison.json", pair)
        payload = load_json_object_or_none(run_dir / "strategy_comparison.json")
        manifest = load_json_object_or_none(run_dir / "manifest.json")
        payload["pairs"][path.parent.name] = pair
        record = build_completed_pair_file_integrity({path.parent.name: path.parent})
        payload["pair_file_integrity"] = manifest["pair_file_integrity"] = record
        self.write(run_dir.relative_to(self.root) / "strategy_comparison.json", payload)
        self.write(run_dir.relative_to(self.root) / "manifest.json", manifest)
        return path

    def assert_published(self, path):
        self.assertTrue(completed_pair_file_integrity_matches(
            run_dir=path.parents[2],
            run_payload=load_json_object_or_none(path.parents[2] / "strategy_comparison.json"),
            pair_id=path.parent.name, pair_dir=path.parent, required_paths=(path,),
        ))

    def test_completed_baseline_copy_is_removed_and_readers_keep_original_bytes(self):
        original = self.publish("a", pinned=True)
        copy = self.baseline_copy("b", original)
        expected = original.read_bytes()
        self.menu(["2", "0"])
        self.assertFalse(copy.exists())
        self.assertEqual(detail_read_path(copy), original)
        self.assertEqual(detail_read_path(copy).read_bytes(), expected)
        self.assertEqual(inherited_detail_files(copy.parent, present_names=set()), {self.output: copy})
        self.assert_published(copy)

    def test_only_explicit_boolean_baseline_provenance_is_accepted(self):
        original = self.publish("source", pinned=True)
        copies = [self.baseline_copy(str(index), original, flag=flag)
                  for index, flag in enumerate((False, 1, "true", None))]
        self.menu(["2", "0"])
        for copy in copies:
            self.assertIsNone(reused_detail_source(copy))
            self.assertTrue(copy.exists())

    def test_baseline_never_supplies_target_diagnostics_or_score_ranking(self):
        for name in ("no_filter_orderable_target_diagnostics.csv",
                     "score_ranking_orderable_candidates.csv",
                     "score_ranking_orderable_target_diagnostics.csv"):
            with self.subTest(filename=name):
                self.output = name
                original = self.publish("source_" + name, pinned=True)
                copy = self.baseline_copy("copy_" + name, original)
                self.assertIsNone(reused_detail_source(copy))
                self.menu(["2", "0"])
                self.assertTrue(copy.exists())
                copy.unlink()
                self.assertEqual(detail_read_path(copy), copy)
                self.assertEqual(inherited_detail_files(copy.parent, present_names=set()), {})

    def test_changed_and_unpublished_copies_are_preserved(self):
        original = self.publish("source", pinned=True)
        changed = self.baseline_copy("changed", original, data=self.payload.replace(b"123", b"456"))
        unfinished = self.baseline_copy("unfinished", original)
        (unfinished.parents[2] / "manifest.json").unlink()
        (unfinished.parents[2] / "strategy_comparison.json").unlink()
        self.menu(["2", "0"])
        self.assertEqual(changed.read_bytes(), self.payload.replace(b"123", b"456"))
        self.assertEqual(unfinished.read_bytes(), self.payload)

    def test_unpublished_source_never_authorizes_removal(self):
        original = self.publish("source", pinned=True)
        copy = self.baseline_copy("copy", original)
        (original.parents[2] / "manifest.json").unlink()
        self.menu(["2", "0"])
        self.assertEqual(copy.read_bytes(), self.payload)

    def test_mixed_pair_and_baseline_chain_remains_readable_after_cleanup(self):
        original = self.publish("a", pinned=True)
        full_pair = self.publish("b", pinned=True, source=original)
        baseline = self.baseline_copy("c", full_pair)
        self.menu(["2", "0"])
        self.assertFalse(full_pair.exists())
        self.assertFalse(baseline.exists())
        self.assertEqual(detail_read_path(baseline), original)
        self.assert_published(full_pair)
        self.assert_published(baseline)

    def test_real_baseline_planner_replay_and_audit_use_baseline_lineage(self):
        original_write = self.write
        converted = []

        def publish_baseline_lineage(relative, data):
            metadata = data.get("metadata") if isinstance(data, dict) else None
            if isinstance(metadata, dict) and "cache_reused_from" in metadata:
                metadata["baseline_reused"] = True
                metadata["baseline_reuse_source"] = metadata.pop("cache_reused_from")
                converted.append(relative)
            return original_write(relative, data)

        with patch.object(self, "write", side_effect=publish_baseline_lineage):
            previous.CleanupTest.test_completed_baseline_planner_and_real_readers_reuse_after_deletion(self)
        self.assertEqual(len(converted), 1)

    def test_cycles_and_nonlocal_paths_do_not_resolve(self):
        original = self.publish("source", pinned=True)
        copy = self.baseline_copy("copy", original)
        marker = copy.parent / "strategy_comparison.json"
        payload = load_json_object_or_none(marker)
        for source in (copy.parent.relative_to(self.root).as_posix(),
                       "../outputs/strategy_compare/other/runs/run/pairs/arm",
                       "C:outputs/strategy_compare/other/runs/run/pairs/arm",
                       "C:/outside/outputs/strategy_compare/other/runs/run/pairs/arm"):
            with self.subTest(source=source):
                payload["metadata"]["baseline_reuse_source"] = source
                marker.write_text(json.dumps(payload))
                with self.assertRaises(ValueError):
                    reused_detail_source(copy)
                self.assertTrue(copy.exists())

    def test_provenance_change_during_hash_preserves_the_copy(self):
        original = self.publish("source", pinned=True)
        copy = self.baseline_copy("copy", original)
        worker, _ = self.scanned_cleaner()
        original_hash = worker.hash_file
        changed = []

        def change_after_hash(path):
            digest = original_hash(path)
            if path == copy and not changed:
                marker = copy.parent / "strategy_comparison.json"
                payload = load_json_object_or_none(marker)
                payload["metadata"]["baseline_reused"] = False
                marker.write_text(json.dumps(payload))
                changed.append(path)
            return digest

        with patch.object(worker, "hash_file", side_effect=change_after_hash):
            worker.clear_reused_details()
        self.assertEqual(changed, [copy])
        self.assertEqual(copy.read_bytes(), self.payload)


if __name__ == "__main__":
    unittest.main()
