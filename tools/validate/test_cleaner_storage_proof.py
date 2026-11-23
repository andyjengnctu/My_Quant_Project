"""Storage cleanup must preserve bytes without relabelling invalid research as READY."""
import json
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from core.compare_csv_storage import detail_read_path
from services.research.strategy_compare_pair_integrity import (
    build_completed_pair_file_integrity, completed_strategy_run_publication_matches,
)
from tools.validate import test_project_cleaner as existing


class CleanerStorageProofTest(unittest.TestCase):
    setUp = existing.CleanupTest.setUp
    git = existing.CleanupTest.git
    write = existing.CleanupTest.write
    publish = existing.CleanupTest.publish
    menu = existing.CleanupTest.menu
    summary = existing.CleanupTest.summary
    scanned_cleaner = existing.CleanupTest.scanned_cleaner

    def full_run_valid(self, path):
        run = path.parents[2]
        return completed_strategy_run_publication_matches(
            run_dir=run, run_payload=json.loads((run / "strategy_comparison.json").read_text()),
            project_root=self.root,
        )

    def add_pair(self, path, name):
        run = path.parents[2]
        payload = json.loads((run / "strategy_comparison.json").read_text())
        pair = run / "pairs" / name
        entry = {"metadata": {"name": name}}
        self.write(pair / "strategy_comparison.json", entry)
        self.write(pair / "strategy_comparison.md", b"other report")
        csv = self.write(pair / self.output, b"other bytes\n")
        payload["pairs"][name] = entry
        record = build_completed_pair_file_integrity({key: run / "pairs" / key for key in payload["pairs"]})
        payload["pair_file_integrity"] = record
        manifest = json.loads((run / "manifest.json").read_text())
        manifest["pair_file_integrity"] = record
        self.write(run / "strategy_comparison.json", payload)
        self.write(run / "manifest.json", manifest)
        return csv

    def test_changed_report_does_not_block_identical_csv_or_become_valid(self):
        source = self.publish("original", pinned=True)
        copy = self.publish("reused", pinned=True, source=source)
        before = copy.read_bytes()
        reports = [p.parent / "strategy_comparison.md" for p in (source, copy)]
        for report in reports:
            report.write_bytes(b"modified report retained for investigation\n")
        self.assertFalse(self.full_run_valid(source))
        self.assertFalse(self.full_run_valid(copy))
        self.menu(["1", "0"])
        self.assertTrue(copy.exists())
        self.assertEqual(self.summary("preview")[0]["counts"].get("PLAN_DELETE"), 1)
        self.menu(["2", "0"])
        self.assertFalse(copy.exists())
        self.assertEqual(detail_read_path(copy).read_bytes(), before)
        self.assertEqual(source.read_bytes(), before)
        self.assertTrue(all(p.read_bytes() == b"modified report retained for investigation\n" for p in reports))
        self.assertFalse(self.full_run_valid(source))
        self.assertFalse(self.full_run_valid(copy))

    def test_unrelated_pair_corruption_does_not_block_intact_detail(self):
        source = self.publish("original", pinned=True)
        copy = self.publish("reused", pinned=True, source=source)
        other = self.add_pair(copy, "unrelated-future-arm")
        other.write_bytes(b"corrupted other pair\n")
        self.assertFalse(self.full_run_valid(copy))
        self.menu(["2", "0"])
        self.assertFalse(copy.exists())
        self.assertEqual(detail_read_path(copy).read_bytes(), self.payload)
        self.assertEqual(other.read_bytes(), b"corrupted other pair\n")
        self.assertFalse(self.full_run_valid(copy))

    def test_missing_unrelated_pair_does_not_delete_its_remaining_evidence(self):
        source = self.publish("original", pinned=True)
        copy = self.publish("reused", pinned=True, source=source)
        other = self.add_pair(source, "other")
        other.unlink()
        self.menu(["2", "0"])
        self.assertFalse(copy.exists())
        self.assertTrue((other.parent / "strategy_comparison.json").exists())
        self.assertEqual(detail_read_path(copy).read_bytes(), self.payload)

    def test_manifest_mismatch_still_blocks_and_exports_only_json_evidence(self):
        source = self.publish("original", pinned=True)
        copy = self.publish("reused", pinned=True, source=source)
        marker = copy.parents[2] / "manifest.json"
        payload = json.loads(marker.read_text())
        payload["config_fingerprint"] = "different123"
        self.write(marker, payload)
        self.menu(["2", "0"])
        self.assertTrue(copy.exists())
        _, decisions = self.summary("apply")
        self.assertTrue(any("config_fingerprint" in row["reason"] for row in decisions if row["action"] == "SKIP"))
        with ZipFile(self.root / "outputs/project_cleanup/last_apply.zip") as archive:
            evidence = [name for name in archive.namelist() if name.startswith("evidence/")]
            self.assertTrue(evidence)
            self.assertTrue(all(name.endswith(".json") for name in evidence))
            self.assertEqual(archive.read("evidence/" + marker.relative_to(self.root).as_posix()), marker.read_bytes())
            self.assertFalse(any(name.endswith(".csv") for name in evidence))

    def test_own_pair_metadata_mismatch_is_not_bypassed(self):
        source = self.publish("original", pinned=True)
        copy = self.publish("reused", pinned=True, source=source)
        marker = copy.parent / "strategy_comparison.json"
        payload = json.loads(marker.read_text())
        payload["metadata"]["changed"] = True
        self.write(marker, payload)
        self.menu(["2", "0"])
        self.assertTrue(copy.exists())
        self.assertTrue(any("run.pairs" in row["reason"] for row in self.summary("apply")[1]))

    def test_equal_modified_bytes_do_not_bypass_existing_csv_hash(self):
        source = self.publish("original", pinned=True)
        copy = self.publish("reused", pinned=True, source=source)
        source.write_bytes(b"changed same\n")
        copy.write_bytes(source.read_bytes())
        self.menu(["2", "0"])
        self.assertTrue(copy.exists())
        self.assertTrue(any("SHA256" in row["reason"] for row in self.summary("apply")[1]))

    def test_source_manifest_failure_is_visible_as_skip_with_detail(self):
        source = self.publish("original", pinned=True)
        copy = self.publish("reused", pinned=True, source=source)
        self.write(source.parents[2] / "manifest.json", {})
        self.menu(["2", "0"])
        self.assertTrue(copy.exists())
        self.assertTrue(any(row["action"] == "SKIP" and "config_fingerprint" in row["reason"]
                            for row in self.summary("apply")[1]))

    def test_manifest_change_after_storage_check_preserves_copy(self):
        source = self.publish("original", pinned=True)
        copy = self.publish("reused", pinned=True, source=source)
        worker, _ = self.scanned_cleaner()
        self.assertTrue(worker.detail_published(copy.relative_to(self.root)))
        self.write(copy.parents[2] / "manifest.json", {})
        worker.clear_reused_details()
        self.assertTrue(copy.exists())

    def test_source_change_immediately_after_hash_does_not_delete_only_old_bytes(self):
        source = self.publish("original", pinned=True)
        copy = self.publish("reused", pinned=True, source=source)
        worker, _ = self.scanned_cleaner()
        real_hash = worker.hash_file
        calls = 0
        def change_after_comparison(path):
            nonlocal calls
            digest = real_hash(path)
            if path == source:
                calls += 1
                if calls == 2:
                    source.write_bytes(b"!" * len(self.payload))
            return digest
        with patch.object(worker, "hash_file", side_effect=change_after_comparison):
            worker.clear_reused_details()
        self.assertGreaterEqual(calls, 2)
        self.assertTrue(copy.exists())
        self.assertEqual(copy.read_bytes(), self.payload)

    def test_review_report_collects_existing_owner_evidence_without_deleting_data(self):
        workspace = Path("outputs/other_research/unknown-future-study")
        result = self.write(workspace / "result.json", {"status": "finished", "choice": "unseen"})
        manifest = self.write(workspace / "manifest.json", {"producer": "unknown"})
        data = self.write(workspace / "large_input.bin", b"irreplaceable raw evidence\n")
        trading = self.write("outputs/trading/account/result.json", {"private": "trading"})
        self.menu(["2", "0"])
        self.assertTrue(all(p.exists() for p in (result, manifest, data, trading)))
        with ZipFile(self.root / "outputs/project_cleanup/last_apply.zip") as archive:
            self.assertEqual(archive.read("evidence/" + result.relative_to(self.root).as_posix()), result.read_bytes())
            self.assertEqual(archive.read("evidence/" + manifest.relative_to(self.root).as_posix()), manifest.read_bytes())
            self.assertFalse(any(name.endswith("large_input.bin") or "/trading/" in name for name in archive.namelist()))
            context = json.loads(archive.read("evidence/review_context.json"))
            self.assertTrue(context["selected"])

    def test_review_export_budget_is_reported_and_does_not_change_retention(self):
        workspace = Path("outputs/other_research/study")
        result = self.write(workspace / "result.json", {"payload": "x"*100})
        with patch("services.cleanup_report.REVIEW_EVIDENCE_BUDGET_BYTES", 10):
            self.menu(["2", "0"])
        self.assertTrue(result.exists())
        with ZipFile(self.root / "outputs/project_cleanup/last_apply.zip") as archive:
            context = json.loads(archive.read("evidence/review_context.json"))
            self.assertFalse(context["selected"])
            self.assertTrue(context["skipped"])


if __name__ == "__main__":
    unittest.main()
