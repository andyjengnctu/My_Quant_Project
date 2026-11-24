"""Frozen caller results can lose duplicate bytes without losing provenance."""
import json
from pathlib import Path
import tempfile
import unittest

from core import caller_baseline_storage as storage
from core.file_integrity import canonical_json_sha256, compute_observed_file_sha256


class CallerBaselineStorageTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.subject = Path("outputs/optimizer/positive_threshold/GENERIC_SUBJECT")
        self.identity = {"spec": {"output_subdir": self.subject.as_posix()}, "seed": 42,
                         "start": "2020-01-01", "end": "2020-12-31", "params_sha256": "a" * 64}
        self.selection = self.root / self.subject / "seed_42/selection" / canonical_json_sha256(self.identity, length=16)
        self.bytes = b"ticker,trade_date\nTEST,2020-01-02\n"

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, bytes):
            path.write_bytes(value)
        else:
            path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def publish(self, count=3):
        candidates, files = [], []
        for i in range(1, count + 1):
            pair = self.selection / f"mask_{i}/replay"
            metadata = {"output_scope": "caller_override", "output_dir": pair.relative_to(self.root).as_posix(),
                        "comparison_mode": "score-ranking", "comparison_period": {"start": "2020-01-01", "end": "2020-12-31"},
                        "params_source_file_sha256": "a" * 64,
                        "score_ranking_options": {"positive_probability_threshold": i / 10},
                        "baseline_reused": bool(files),
                        "baseline_reuse_source": files[-1].parent.relative_to(self.root).as_posix() if files else None}
            payload = {"metadata": metadata, "no_filter": {"total_return_pct": 1},
                       "score_ranking": {"total_return_pct": i}, "yearly": [{"year": 2020, "return_pct": i}]}
            self.write(pair / "strategy_comparison.json", payload)
            self.write(pair.parent / "result.json", payload)
            files.append(self.write(pair / storage.BASELINE_NAME, self.bytes))
            candidates.append({"threshold": i / 10, "admission_mask_key": i,
                               "metrics": payload["score_ranking"], "yearly": payload["yearly"]})
        frozen = {"status": "FROZEN", "selection_identity": self.identity,
                  "selected_threshold": 0.1, "candidates": candidates}
        frozen["receipt_sha256"] = canonical_json_sha256(frozen)
        self.write(self.selection / "frozen_threshold.json", frozen)
        return files

    def test_frozen_receipt_matches_result_identity_metrics_and_period(self):
        original, copied, _ = self.publish()
        evidence, digest = storage.publication_evidence(copied)
        self.assertEqual(len(evidence), 3)
        self.assertIsNone(digest)
        self.assertEqual(storage.recorded_source(copied), original)
        self.assertIsNone(storage.recorded_source(original))
        markers = (copied.parent.parent / "result.json", self.selection / "frozen_threshold.json")
        saved = {path: path.read_bytes() for path in markers}
        for marker in markers:
            with self.subTest(marker=marker.name):
                self.write(marker, {})
                with self.assertRaises(ValueError):
                    storage.publication_evidence(copied)
                marker.write_bytes(saved[marker])
        frozen = json.loads(markers[1].read_text(encoding="utf-8"))
        frozen["candidates"][1]["metrics"]["total_return_pct"] = -999
        frozen["receipt_sha256"] = canonical_json_sha256({k: v for k, v in frozen.items() if k != "receipt_sha256"})
        self.write(markers[1], frozen)
        with self.assertRaisesRegex(ValueError, "候選"):
            storage.publication_evidence(copied)

    def test_only_supported_namespace_and_same_selection_sources_are_accepted(self):
        original, copied, _ = self.publish()
        self.assertEqual(storage.root_for(copied), self.root)
        for path in (self.root / "outputs/arbitrary/replay" / storage.BASELINE_NAME,
                     copied.with_name("score_ranking_orderable_candidates.csv"),
                     self.selection / "evaluation/treatment" / storage.BASELINE_NAME):
            self.assertIsNone(storage.root_for(path))
        marker = copied.parent / "strategy_comparison.json"
        payload = json.loads(marker.read_text(encoding="utf-8"))
        payload["metadata"]["baseline_reuse_source"] = "../outside/replay"
        self.write(marker, payload)
        self.write(copied.parents[1] / "result.json", payload)
        with self.assertRaises(ValueError):
            storage.recorded_source(copied)

    def test_storage_receipt_pins_bytes_without_changing_scientific_records(self):
        original, copied, _ = self.publish()
        scientific = {path: path.read_bytes() for path in (
            self.selection / "frozen_threshold.json", copied.parent / "strategy_comparison.json",
            copied.parents[1] / "result.json")}
        digest = compute_observed_file_sha256(copied)
        storage.record_alias(copied, original, digest)
        self.assertEqual(storage.expected_sha256(copied), digest)
        self.assertTrue(all(path.read_bytes() == content for path, content in scientific.items()))
        copied.unlink()
        self.assertEqual(storage.recorded_source(copied), original)
        self.assertEqual(storage.expected_sha256(copied), digest)
        copied.parent.joinpath(storage.STORAGE_NAME).unlink()
        with self.assertRaisesRegex(ValueError, "儲存收據"):
            storage.recorded_source(copied)

    def test_byte_difference_or_republished_pair_never_reuses_old_receipt(self):
        original, copied, _ = self.publish()
        digest = compute_observed_file_sha256(copied)
        original.write_bytes(b"different bytes")
        with self.assertRaisesRegex(ValueError, "相同 bytes"):
            storage.record_alias(copied, original, digest)
        self.assertFalse((copied.parent / storage.STORAGE_NAME).exists())
        original.write_bytes(self.bytes)
        storage.record_alias(copied, original, digest)
        marker = copied.parent / "strategy_comparison.json"
        payload = json.loads(marker.read_text(encoding="utf-8"))
        payload["metadata"]["new_publication"] = True
        self.write(marker, payload)
        self.write(copied.parents[1] / "result.json", payload)
        with self.assertRaisesRegex(ValueError, "原發布內容"):
            storage.expected_sha256(copied)

    def test_missing_copy_reads_original_chain_and_rejects_changed_bytes(self):
        from core.compare_csv_storage import detail_read_path
        original, first, second = self.publish()
        digest = compute_observed_file_sha256(original)
        storage.record_alias(first, original, digest)
        storage.record_alias(second, original, digest)
        first.unlink()
        second.unlink()
        self.assertEqual(detail_read_path(second).read_bytes(), self.bytes)
        original.write_bytes(b"changed source")
        with self.assertRaises(ValueError):
            detail_read_path(second)


if __name__ == "__main__":
    unittest.main()
