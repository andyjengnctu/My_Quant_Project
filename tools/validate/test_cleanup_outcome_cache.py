"""Disposable Min execution cache cleanup preserves canonical computation."""
from pathlib import Path
from hashlib import sha256
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from filters.breakout_quality.daily_trade40_target import CounterfactualTrade40Batch
from services import cleanup_outcome_cache as cleanup
from services.breakout_quality import formal_mintrade_outcome_cache as owner


class OutcomeCacheCleanupTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.key = "b" * 64
        self.relative = cleanup.CACHE_ROOT / "aa" / ("a" * 64 + ".npz")
        self.batch = CounterfactualTrade40Batch(
            target_raw_r=np.array([1.0, np.nan], dtype=np.float32),
            target_raw_pnl=np.array([100.0, np.nan], dtype=np.float64),
            valid_mask=np.array([True, False]),
            maturity_dates=np.array(["2020-01-03", "NaT"], dtype="datetime64[D]"),
            reasons=("complete", "unavailable"), cutoff_truncated_rows=0,
            research_cutoff_closeout_rows=0, research_cutoff_forced_mask=np.array([False, False]),
        )
        self.kwargs = dict(
            project_root=self.root, frame=pd.DataFrame(), positions=np.array([0, 1]),
            ticker="TEST", exchange_calendar=pd.DatetimeIndex([]), source_sha256="c" * 64,
            seed=123, schedule_signatures=(), horizon_sessions=40,
            research_cutoff=pd.Timestamp("2020-01-03"), signal_state_reset_date=None,
        )

    def write_cache(self):
        owner._write_cache(self.root / self.relative, self.key, self.batch)
        return self.root / self.relative

    def test_only_valid_producer_shards_have_disposal_proof(self):
        path = self.write_cache()
        self.assertTrue(cleanup.candidate(self.relative))
        self.assertTrue(cleanup.proof(self.root, self.relative))
        for relative in (
            Path("data") / self.relative.name,
            cleanup.CACHE_ROOT / "bb" / self.relative.name,
            cleanup.CACHE_ROOT / "aa" / "extra" / self.relative.name,
            cleanup.CACHE_ROOT / "aa" / ".temporary.npz",
            cleanup.CACHE_ROOT / "aa" / "unknown.npz",
            path,
        ):
            with self.subTest(path=relative):
                self.assertFalse(cleanup.candidate(relative))
                self.assertFalse(cleanup.proof(self.root, relative))
        path.write_bytes(b"not an npz")
        self.assertFalse(cleanup.proof(self.root, self.relative))
        np.savez(path, unrelated=np.array([1]))
        self.assertFalse(cleanup.proof(self.root, self.relative))

    def test_producer_invariants_are_required_and_mutation_is_rejected(self):
        path = self.write_cache()
        with np.load(path, allow_pickle=False) as stored:
            payload = {name: stored[name] for name in stored.files}
        payload["forced"] = np.array([False, True])
        np.savez(path, **payload)
        self.assertFalse(cleanup.proof(self.root, self.relative))
        self.write_cache()
        read = cleanup.inspect_formal_mintrade_cache_shard

        def changing_read(*args):
            result = read(*args)
            path.write_bytes(b"changed after validation")
            return result

        with patch.object(cleanup, "inspect_formal_mintrade_cache_shard", side_effect=changing_read):
            with self.assertRaisesRegex(ValueError, "驗證期間變動"):
                cleanup.proof(self.root, self.relative)

    def test_cleaner_and_producer_use_the_same_exclusive_lease(self):
        with cleanup.deletion_lease(self.root):
            with self.assertRaises(RuntimeError):
                with owner.formal_mintrade_cache_access(self.root):
                    self.fail("a live cleanup lease must exclude cache I/O")

    def test_cache_hit_and_missing_cache_return_identical_canonical_batch(self):
        with patch.object(owner, "_cache_identity", return_value=self.key), \
                patch.object(owner, "_read_cache", wraps=owner._read_cache) as read:
            calls = []

            def compute():
                calls.append(True)
                return self.batch

            first = owner.load_or_compute_formal_mintrade_outcomes(**self.kwargs, compute=compute)
            cold_reads = read.call_count
            cached = owner.load_or_compute_formal_mintrade_outcomes(**self.kwargs, compute=compute)
            self.assertEqual(len(calls), 1)
            self.assertEqual(read.call_count - cold_reads, 1)
            shard = next((self.root / cleanup.CACHE_ROOT).rglob("*.npz"))
            relative = shard.relative_to(self.root)
            self.assertTrue(cleanup.proof(self.root, relative))
            # Use the cleaner's unified deletion boundary, retaining its Git,
            # physical-path, regular-file, link-count and revision protection.
            from types import SimpleNamespace
            from services.project_cleaner import Cleaner
            worker = Cleaner(self.root, SimpleNamespace(row=lambda *a, **kw: None), True)
            with cleanup.deletion_lease(self.root), patch.object(
                    worker, "proof", side_effect=lambda kind, rel, scope: cleanup.proof(self.root, rel)):
                worker.remove_one("outcome_cache", relative, shard.stat().st_size, None)
            self.assertFalse(shard.exists())
            restored = owner.load_or_compute_formal_mintrade_outcomes(**self.kwargs, compute=compute)
            self.assertEqual(len(calls), 2)
            for field in ("target_raw_r", "target_raw_pnl", "valid_mask", "maturity_dates",
                          "research_cutoff_forced_mask"):
                np.testing.assert_array_equal(getattr(first, field), getattr(cached, field))
                np.testing.assert_array_equal(getattr(first, field), getattr(restored, field))

    def test_busy_cleanup_lock_skips_optional_cache_without_blocking_compute(self):
        with patch.object(owner, "_cache_identity", return_value=self.key), \
                cleanup.deletion_lease(self.root), \
                patch.object(owner, "_read_cache", side_effect=AssertionError("read while locked")), \
                patch.object(owner, "_write_cache", side_effect=AssertionError("write while locked")):
            result = owner.load_or_compute_formal_mintrade_outcomes(
                **self.kwargs, compute=lambda: self.batch,
            )
        self.assertIs(result, self.batch)
        self.assertFalse((self.root / cleanup.CACHE_ROOT).exists())

    def test_distinct_pending_signal_lifecycles_do_not_evict_each_other(self):
        calls = []
        def compute():
            calls.append(True)
            return self.batch
        for reset in (None, '2014-01-01', None, '2014-01-01'):
            result = owner.load_or_compute_formal_mintrade_outcomes(
                **{**self.kwargs, 'signal_state_reset_date': reset}, compute=compute)
            np.testing.assert_array_equal(result.target_raw_pnl, self.batch.target_raw_pnl)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(list((self.root / cleanup.CACHE_ROOT).rglob('*.npz'))), 2)

    def test_legacy_slot_requires_the_complete_current_content_identity(self):
        legacy = sha256(json.dumps({
            'ticker': 'TEST', 'seed': 123, 'horizon_sessions': 40, 'cutoff': '2020-01-03',
        }, sort_keys=True).encode('utf-8')).hexdigest()
        owner._write_cache(self.root / cleanup.CACHE_ROOT / legacy[:2] / (legacy+'.npz'), self.key, self.batch)
        with patch.object(owner, '_cache_identity', return_value=self.key):
            result = owner.load_or_compute_formal_mintrade_outcomes(
                **self.kwargs, compute=lambda: self.fail('verified legacy hit must not recompute'))
        np.testing.assert_array_equal(result.target_raw_pnl, self.batch.target_raw_pnl)
        calls = []
        with patch.object(owner, '_cache_identity', return_value='d'*64):
            owner.load_or_compute_formal_mintrade_outcomes(
                **self.kwargs, compute=lambda: calls.append(True) or self.batch)
        self.assertEqual(len(calls), 1)

    def test_transient_windows_lock_uses_shared_atomic_publication_retry(self):
        from core import file_integrity
        original = file_integrity.os.replace
        attempts = []
        def transient(*args):
            attempts.append(True)
            if len(attempts) == 1:
                raise PermissionError('Windows transient file lock')
            return original(*args)
        with patch.object(file_integrity.os, 'replace', side_effect=transient), \
                patch.object(file_integrity.time, 'sleep'):
            path = self.write_cache()
        self.assertEqual(len(attempts), 2)
        self.assertTrue(cleanup.proof(self.root, path.relative_to(self.root)))

    def test_optional_cache_write_failure_does_not_hide_the_canonical_outcome(self):
        with patch.object(owner, '_cache_identity', return_value=self.key), \
                patch.object(owner, '_write_cache', side_effect=PermissionError('locked')):
            result = owner.load_or_compute_formal_mintrade_outcomes(
                **self.kwargs, compute=lambda: self.batch)
        self.assertIs(result, self.batch)
        def failed_compute():
            raise RuntimeError('canonical computation failed')
        with patch.object(owner, '_cache_identity', return_value=self.key), \
                self.assertRaisesRegex(RuntimeError, 'canonical computation failed'):
            owner.load_or_compute_formal_mintrade_outcomes(**self.kwargs, compute=failed_compute)


if __name__ == "__main__":
    unittest.main()
