"""Only producer-owned, self-verified optional projections can be discarded."""
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from services.breakout_quality import real_path_diagnostic_cache as cache
from tools.validate import test_project_cleaner as existing


class DiagnosticCacheCleanupTest(unittest.TestCase):
    setUp = existing.CleanupTest.setUp
    git = existing.CleanupTest.git
    write = existing.CleanupTest.write
    menu = existing.CleanupTest.menu
    summary = existing.CleanupTest.summary
    scanned_cleaner = existing.CleanupTest.scanned_cleaner

    def projection(self, *, seed=123, digest=None, extra=None):
        path = cache._slot(self.root, "future-filter", "unregistered-architecture", "unregistered-profile", seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        values = {
            "ticker": np.array(["0050", "2330"]),
            "dates": np.array(["2026-01-02", "2026-01-02"], dtype="datetime64[ns]"),
            "labels": np.array([-1, 1], dtype=np.int64),
            "raw_r": np.array([np.nan, 1.25], dtype=np.float64),
            "valid": np.array([False, True], dtype=bool),
        }
        checksum = cache._payload_digest(*(values[key] for key in ("ticker", "dates", "labels", "raw_r", "valid")))
        values.update(identity=np.asarray("a" * 64), payload_sha256=np.asarray(digest or checksum))
        values.update(extra or {})
        np.savez_compressed(path, **values)
        return path

    def test_cleanup_is_independent_of_current_profile_and_preserves_sources(self):
        projection = self.projection()
        keep = [self.write("data/canonical.csv", b"source"), self.write("models/fit/model.pt", b"model"),
                self.write("outputs/research/result.json", {"published": True})]
        before = {path: path.read_bytes() for path in keep}
        self.menu(["1", "0"])
        self.assertTrue(projection.exists())
        self.assertTrue(any(row["action"] == "PLAN_DELETE" and row["path"].endswith(projection.name)
                            for row in self.summary("preview")[1]))
        self.menu(["2", "0"])
        self.assertFalse(projection.exists())
        self.assertEqual(before, {path: path.read_bytes() for path in keep})
        self.menu(["2", "0"])
        self.assertFalse(self.summary("apply")[0]["counts"].get("DELETE"))

    def test_unknown_layout_schema_and_bad_payload_are_kept(self):
        valid = self.projection()
        corrupted = self.projection(seed=124, digest="b" * 64)
        unknown = self.projection(seed=125, extra={"evidence": np.asarray("keep")})
        other = self.write(Path("outputs/other_cache") / valid.name, valid.read_bytes())
        shard = self.write(cache._CACHE_SUBDIR / "zz" / valid.name, valid.read_bytes())
        for path in (corrupted, unknown):
            with self.assertRaises(ValueError):
                cache.real_path_diagnostic_cache_proof(self.root, path.relative_to(self.root))
        self.menu(["2", "0"])
        self.assertFalse(valid.exists())
        self.assertTrue(all(path.exists() for path in (corrupted, unknown, other, shard)))

    def test_held_reader_publication_lease_prevents_cleanup(self):
        path = self.projection()
        with cache.real_path_diagnostic_cache_access(self.root):
            self.menu(["2", "0"])
        self.assertTrue(path.exists())
        self.assertTrue(any(row["action"] == "SKIP" and row["path"].endswith(path.name)
                            for row in self.summary("apply")[1]))

    def test_reader_preserves_payload_and_treats_absence_or_busy_cache_as_miss(self):
        path = self.projection()
        args = dict(root=self.root, filter_id="future-filter", architecture="unregistered-architecture",
                    profile="unregistered-profile", seed=123, target_lineage={})
        with patch.object(cache, "_identity", return_value="a" * 64):
            loaded = cache.load_verified_real_path_diagnostic(**args)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.group_table["ticker"].tolist(), ["0050", "2330"])
            np.testing.assert_array_equal(loaded.target_valid, [False, True])
            with cache.real_path_diagnostic_cache_access(self.root):
                self.assertIsNone(cache.load_verified_real_path_diagnostic(**args))
            path.unlink()
            self.assertIsNone(cache.load_verified_real_path_diagnostic(**args))

    def test_checksum_change_during_verification_is_rejected(self):
        path = self.projection()
        real_read = cache._read_projection
        def mutate(file):
            result = real_read(file)
            path.write_bytes(b"changed")
            return result
        with patch.object(cache, "_read_projection", side_effect=mutate):
            with self.assertRaises(ValueError):
                cache.real_path_diagnostic_cache_proof(self.root, path.relative_to(self.root))

    def test_preview_does_not_create_a_projection_lock(self):
        with cache.real_path_diagnostic_cache_access(self.root, preview=True):
            pass
        self.assertFalse((self.root / "outputs/project_cleanup/real_path_diagnostic.lock").exists())


if __name__ == "__main__":
    unittest.main()
