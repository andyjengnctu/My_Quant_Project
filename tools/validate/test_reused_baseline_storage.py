"""Only producer-reused candidate baselines may leave unfinished comparisons."""
import contextlib
from copy import deepcopy
import io
import json
import subprocess
import unittest
from unittest.mock import patch

from core import reused_copy_storage as storage
from core.compare_csv_storage import detail_read_path
from services.research.strategy_compare_pair_integrity import completed_strategy_run_publication_matches
from tools.validate import test_reused_copy_storage as fixtures


class ReusedBaselineStorageTest(unittest.TestCase):
    write = fixtures.ReusedCopyStorageTest.write
    pair = fixtures.ReusedCopyStorageTest.pair
    complete = fixtures.ReusedCopyStorageTest.complete

    def setUp(self):
        fixtures.ReusedCopyStorageTest.setUp(self)
        self.filename = "no_filter_orderable_candidates.csv"

    def fixture(self, *, pinned=True):
        source = self.complete(self.pair("original"), pinned=pinned)
        target = self.pair("unfinished")
        marker = target.parent / "strategy_comparison.json"
        pair = json.loads(marker.read_text())
        pair["metadata"].update(output_scope="caller_override", comparison_mode="score-ranking",
            baseline_reused=True, baseline_reuse_source=source.parent.relative_to(self.root).as_posix(),
            model="another independently executed arm")
        pair["score_ranking"]["return_pct"] = 999
        self.write(marker, pair)
        return source, target

    def test_baseline_copy_pins_bytes_without_equating_independent_arm_results(self):
        for pinned in (True, False):
            with self.subTest(pinned=pinned):
                source, target = self.fixture(pinned=pinned)
                scientific = {p: p.read_bytes() for p in self.base.rglob("*.json")}
                _, digest = storage.publication_evidence(target)
                storage.record_alias(target, source, digest)
                target.unlink()
                self.assertEqual(detail_read_path(target).read_bytes(), self.data)
                self.assertTrue(all(p.read_bytes() == value for p, value in scientific.items()))
                self.assertFalse((target.parents[2] / "strategy_comparison.json").exists())
                self.assertFalse((target.parents[2] / "manifest.json").exists())
                source.write_bytes(b"!" * len(self.data))
                with self.assertRaises(ValueError):
                    detail_read_path(target)

    def test_baseline_boolean_and_single_filename_contract_are_strict(self):
        _, target = self.fixture()
        marker = target.parent / "strategy_comparison.json"
        saved = json.loads(marker.read_text())
        for value in (1, "true", False, None):
            payload = deepcopy(saved)
            payload["metadata"]["baseline_reused"] = value
            self.write(marker, payload)
            with self.assertRaises(ValueError):
                storage.publication_evidence(target)
        self.write(marker, saved)
        for name in ("score_ranking_orderable_candidates.csv", "no_filter_orderable_target_diagnostics.csv",
                     "score_ranking_orderable_target_diagnostics.csv", "no_filter_trades.csv"):
            other = self.write(target.with_name(name), self.data)
            with self.assertRaises(ValueError):
                storage.publication_evidence(other)

    def test_misplaced_or_ambiguous_provenance_never_uses_baseline_fallback(self):
        _, target = self.fixture()
        marker = target.parent / "strategy_comparison.json"
        original = json.loads(marker.read_text())
        mutations = [
            {"output_scope": "canonical"}, {"output_dir": "other/place"},
            {"comparison_mode": "hard-filter"}, {"baseline_reuse_source": "../outside"},
            {"baseline_reuse_source": str(self.base)}, {"cache_reused_from": ""},
            {"cache_reused_from": original["metadata"]["baseline_reuse_source"]},
        ]
        for mutation in mutations:
            payload = deepcopy(original)
            payload["metadata"].update(mutation)
            self.write(marker, payload)
            with self.assertRaises(ValueError):
                storage.publication_evidence(target)

    def test_partial_target_and_incomplete_direct_source_stay_protected(self):
        source, target = self.fixture()
        for name in ("manifest.json", "strategy_comparison.json"):
            marker = target.parents[2] / name
            self.write(marker, {"status": "COMPLETED"})
            with self.assertRaisesRegex(ValueError, "發布紀錄"):
                storage.publication_evidence(target)
            marker.unlink()
        (source.parents[2] / "manifest.json").unlink()
        with self.assertRaises((OSError, ValueError)):
            storage.publication_evidence(target)
        self.assertTrue(target.exists())
        self.assertFalse(storage.receipt_path(target).exists())

    def test_baseline_receipt_allows_normal_benchmark_refresh_and_real_completion(self):
        source, target = self.fixture(pinned=False)
        _, digest = storage.publication_evidence(target)
        storage.record_alias(target, source, digest)
        target.unlink()
        marker = target.parent / "strategy_comparison.json"
        pair = json.loads(marker.read_text())
        pair["benchmark_yearly"] = [{"year": 2020, "return_pct": 3}]
        self.write(marker, pair)
        self.complete(target)
        result = json.loads((target.parents[2] / "strategy_comparison.json").read_text())
        self.assertEqual(result["pair_file_integrity"]["pairs"]["arm"][self.filename], digest)
        self.assertTrue(completed_strategy_run_publication_matches(
            run_dir=target.parents[2], run_payload=result, project_root=self.root))
        self.assertEqual(detail_read_path(target).read_bytes(), self.data)

    def test_cleaner_preview_then_apply_baseline_copy_keeps_scientific_json(self):
        from apps import cleaner
        source, target = self.fixture(pinned=False)
        self.write(self.root / "doc/PROJECT_SETTINGS.md", b"fixture")
        subprocess.run(["git", "-C", str(self.root), "init", "-q"], check=True, capture_output=True)
        scientific = {p: p.read_bytes() for p in self.base.rglob("*.json")}
        for answer in ("1", "2"):
            with patch.object(cleaner, "ROOT", self.root), patch("builtins.input", side_effect=[answer, "0"]), contextlib.redirect_stdout(io.StringIO()):
                cleaner.main()
            self.assertTrue(source.exists())
            self.assertEqual(target.exists(), answer == "1")
            self.assertEqual(storage.receipt_path(target).exists(), answer == "2")
            self.assertTrue(all(p.read_bytes() == value for p, value in scientific.items()))
            self.assertFalse((target.parents[2] / "manifest.json").exists())
        self.assertEqual(detail_read_path(target).read_bytes(), self.data)


if __name__ == "__main__":
    unittest.main()
