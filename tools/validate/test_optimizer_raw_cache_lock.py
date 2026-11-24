"""Raw-cache owner transitions must remain exclusive across real processes."""
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from core.file_integrity import exclusive_file_lock
from services.optimizer.raw_cache import _RawCacheBuildLock, _build_raw_cache_paths


def _paused_reaper(paths, started, before_unlink, proceed, results):
    from services.optimizer import raw_cache
    original = raw_cache._safe_unlink

    def pause(path):
        before_unlink.set()
        if not proceed.wait(10):
            raise RuntimeError("test did not release stale-owner transition")
        original(path)

    try:
        started.set()
        with patch.object(raw_cache, "_safe_unlink", side_effect=pause):
            _RawCacheBuildLock(paths)._remove_stale_lock_if_needed(0)
        results.put("ok")
    except Exception as exc:
        results.put(f"{type(exc).__name__}: {exc}")


def _hold_guard(paths, ready):
    with _RawCacheBuildLock(paths)._state_guard():
        ready.set()
        time.sleep(20)


class OptimizerRawCacheLockTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.paths = _build_raw_cache_paths(Path(self.temp.name), "fixture", 360)
        self.paths["cache_dir"].mkdir()
        self.context = multiprocessing.get_context("spawn")

    def test_two_process_reapers_cannot_keep_the_same_stale_owner_observation(self):
        # A real terminated PID proves absence without teaching the test a
        # fake liveness policy. All payloads and locks live in a temporary root.
        exited = subprocess.Popen([sys.executable, "-B", "-c", "pass"])
        exited.wait(timeout=10)
        dead_pid = exited.pid
        # Windows keeps a terminated PID observable until Popen's process
        # handle closes; release it before asking the production liveness API.
        del exited
        from core.artifact_lifecycle import process_is_absent
        self.assertTrue(process_is_absent(dead_pid))
        self.paths["lock_dir"].mkdir()
        owner_path = self.paths["lock_dir"] / "owner.json"
        owner_path.write_text(json.dumps({"pid": dead_pid, "created_at": 1}))
        os.utime(self.paths["lock_dir"], (1, 1))
        queues, events, workers = [], [], []
        for _ in range(2):
            started, paused, proceed = (self.context.Event() for _ in range(3))
            result = self.context.Queue()
            worker = self.context.Process(
                target=_paused_reaper,
                args=(self.paths, started, paused, proceed, result),
            )
            events.append((started, paused, proceed))
            queues.append(result)
            workers.append(worker)
        try:
            workers[0].start()
            self.assertTrue(events[0][1].wait(10), "first reaper did not validate the dead owner")
            workers[1].start()
            self.assertTrue(events[1][0].wait(10))
            self.assertFalse(events[1][1].wait(0.3), "second reaper retained a stale deletion decision")
            events[0][2].set()
            events[1][2].set()
            for worker, result in zip(workers, queues):
                worker.join(10)
                self.assertFalse(worker.is_alive())
                self.assertEqual(result.get(timeout=2), "ok")
            self.assertFalse(self.paths["lock_dir"].exists())
            # A newly acquired owner remains live even when it is made old
            # enough to trigger another process's stale-recovery inspection.
            with _RawCacheBuildLock(self.paths) as active:
                os.utime(self.paths["lock_dir"], (1, 1))
                token = json.loads(active.owner_path.read_text())["token"]
                started, paused, proceed = (self.context.Event() for _ in range(3))
                proceed.set()
                result = self.context.Queue()
                newer = self.context.Process(
                    target=_paused_reaper, args=(self.paths, started, paused, proceed, result),
                )
                workers.append(newer)
                queues.append(result)
                newer.start()
                newer.join(10)
                self.assertFalse(newer.is_alive())
                self.assertEqual(result.get(timeout=2), "ok")
                self.assertFalse(paused.is_set())
                self.assertEqual(json.loads(active.owner_path.read_text())["token"], token)
        finally:
            for _, _, proceed in events:
                proceed.set()
            for worker in workers:
                if worker.pid is not None:
                    if worker.is_alive():
                        worker.terminate()
                    worker.join(10)
            for result in queues:
                result.close()

    def test_nonblocking_acquisition_respects_an_inflight_transition(self):
        lock = _RawCacheBuildLock(self.paths, blocking=False)
        with exclusive_file_lock(lock.guard_path):
            started = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, "transition"):
                lock.__enter__()
            self.assertLess(time.monotonic() - started, 1.0)
            self.assertFalse(lock.lock_dir.exists())
        with lock:
            self.assertTrue(lock.owner_path.exists())
        self.assertTrue(lock.guard_path.is_file())
        self.assertFalse(lock.lock_dir.exists())

    def test_release_never_removes_a_replaced_owner(self):
        with _RawCacheBuildLock(self.paths) as lock:
            replacement = {"pid": os.getpid(), "token": "different owner"}
            lock.owner_path.write_text(json.dumps(replacement))
            with patch("builtins.print"):
                lock.__exit__(None, None, None)
        self.assertEqual(json.loads(lock.owner_path.read_text()), replacement)

    def test_terminated_process_does_not_leave_the_guard_locked(self):
        ready = self.context.Event()
        worker = self.context.Process(target=_hold_guard, args=(self.paths, ready))
        worker.start()
        try:
            self.assertTrue(ready.wait(10))
            worker.terminate()
            worker.join(10)
            self.assertFalse(worker.is_alive())
            with _RawCacheBuildLock(self.paths, blocking=False) as lock:
                self.assertTrue(lock.owner_path.exists())
        finally:
            if worker.is_alive():
                worker.terminate()
            worker.join(10)


if __name__ == "__main__":
    unittest.main()
