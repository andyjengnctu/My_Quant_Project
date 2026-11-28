"""Dataset preference and scheduler evidence must survive concurrent writers."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import multiprocessing
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import patch

from services.trading import market_data_dataset_state as datasets
from core.market_data_trading_storage_contract import resolve_trading_market_data_v2_dataset_state_path


_DATASET = "TaiwanStockMarketValue"
_NOW = datetime(2026, 11, 20, 21, tzinfo=timezone(timedelta(hours=8)))


def _set_mode_in_process(root, started, finished, errors):
    try:
        started.set()
        datasets.set_market_data_scan_freshness_mode(root, dataset=_DATASET, mode="exact_target")
    except Exception as exc:
        errors.put(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        finished.set()


class DatasetStateConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "account"

    def initialize(self, root):
        datasets.publish_market_data_dataset_state(root, datasets.build_initial_market_data_dataset_state(updated_at=_NOW))
        datasets.set_market_data_scan_freshness_mode(root, dataset=_DATASET, mode="latest_synced")

    def set_mode(self, root):
        return datasets.set_market_data_scan_freshness_mode(root, dataset=_DATASET, mode="exact_target")

    def background_operations(self):
        return {
            "sync": lambda root: datasets.record_market_data_sync_success(
                root, target_date="2026-11-20", finished_at=_NOW,
                attempted_datasets=(_DATASET,),
                observations={_DATASET: {"row_count": 1, "observed_max_date": "2026-11-20"}}),
            "outcome": lambda root: datasets.schedule_market_data_auto_update_outcomes(
                root, target_date="2026-11-20", now=_NOW,
                publication_retry_minutes=(5, 10), max_publication_retries=2,
                quota_defer_minutes=5, error_defer_minutes=5,
                error_datasets=(_DATASET,), error_message="provider unavailable"),
            "due": lambda root: datasets.refresh_market_data_due_state(
                root, target_date="2026-11-20", now=_NOW),
        }

    def delayed_first_read(self, entered, release):
        original = datasets.load_market_data_dataset_state
        first = True
        def load(*args, **kwargs):
            nonlocal first
            value = original(*args, **kwargs)
            if first:
                first = False
                entered.set()
                if not release.wait(10):
                    raise TimeoutError("dataset mutation test barrier")
            return value
        return load

    def test_thread_interleavings_preserve_preferences_and_all_scheduler_evidence(self):
        for name, background in self.background_operations().items():
            for preference_first in (False, True):
                with self.subTest(operation=name, preference_first=preference_first):
                    self.initialize(self.root)
                    serial_root = self.root.parent / "serial"
                    self.initialize(serial_root)
                    first, second = (self.set_mode, background) if preference_first else (background, self.set_mode)
                    first(serial_root)
                    second(serial_root)
                    expected = datasets.load_market_data_dataset_state(serial_root)["datasets"]
                    entered, release, second_started, second_finished = Event(), Event(), Event(), Event()
                    def run_second():
                        second_started.set()
                        try:
                            return second(self.root / "unused" / "..")
                        finally:
                            second_finished.set()
                    with patch.object(datasets, "load_market_data_dataset_state", self.delayed_first_read(entered, release)):
                        with ThreadPoolExecutor(max_workers=2) as pool:
                            first_result = pool.submit(first, self.root)
                            try:
                                self.assertTrue(entered.wait(5))
                                second_result = pool.submit(run_second)
                                self.assertTrue(second_started.wait(5))
                                self.assertFalse(second_finished.wait(0.1))
                            finally:
                                release.set()
                            first_result.result(timeout=10)
                            second_result.result(timeout=10)
                    actual = datasets.load_market_data_dataset_state(self.root)["datasets"]
                    self.assertEqual(expected, actual)
                    self.assertEqual("exact_target", actual[_DATASET]["scan_freshness_mode"])

    def test_process_writer_waits_for_read_modify_publish_and_keeps_both_results(self):
        self.initialize(self.root)
        context = multiprocessing.get_context("spawn")
        started, finished, errors = context.Event(), context.Event(), context.Queue()
        entered, release = Event(), Event()
        child = context.Process(target=_set_mode_in_process, args=(str(self.root), started, finished, errors))
        with patch.object(datasets, "load_market_data_dataset_state", self.delayed_first_read(entered, release)):
            with ThreadPoolExecutor(max_workers=1) as pool:
                first = pool.submit(self.background_operations()["outcome"], self.root)
                try:
                    self.assertTrue(entered.wait(5))
                    child.start()
                    self.assertTrue(started.wait(15))
                    self.assertFalse(finished.wait(0.15))
                finally:
                    release.set()
                    if child.pid is not None:
                        child.join(15)
                        if child.is_alive():
                            child.terminate()
                            child.join(5)
                first.result(timeout=10)
        self.assertEqual(0, child.exitcode)
        errors.close()
        errors.join_thread()
        row = datasets.load_market_data_dataset_state(self.root)["datasets"][_DATASET]
        self.assertEqual("exact_target", row["scan_freshness_mode"])
        self.assertEqual("provider unavailable", row["last_error"])
        self.assertEqual(1, row["error_retry_count"])

    def test_different_project_roots_do_not_block_or_share_state(self):
        other = self.root.parent / "other"
        self.initialize(self.root)
        self.initialize(other)
        entered, release = Event(), Event()
        with patch.object(datasets, "load_market_data_dataset_state", self.delayed_first_read(entered, release)):
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(self.background_operations()["outcome"], self.root)
                try:
                    self.assertTrue(entered.wait(5))
                    pool.submit(self.set_mode, other).result(timeout=5)
                finally:
                    release.set()
                first.result(timeout=10)
        root_row = datasets.load_market_data_dataset_state(self.root)["datasets"][_DATASET]
        other_row = datasets.load_market_data_dataset_state(other)["datasets"][_DATASET]
        self.assertEqual("latest_synced", root_row["scan_freshness_mode"])
        self.assertEqual("exact_target", other_row["scan_freshness_mode"])
        self.assertEqual("provider unavailable", root_row["last_error"])
        self.assertIsNone(other_row["last_error"])

    def test_validation_and_write_failures_release_both_lock_layers(self):
        self.initialize(self.root)
        path = resolve_trading_market_data_v2_dataset_state_path(self.root)
        before = path.read_bytes()
        with self.assertRaises(ValueError):
            datasets.set_market_data_scan_freshness_mode(self.root, dataset=_DATASET, mode="invalid")
        with patch.object(datasets, "atomic_write_json", side_effect=OSError("injected disk failure")):
            with self.assertRaisesRegex(OSError, "injected disk failure"):
                self.set_mode(self.root)
        self.assertEqual(before, path.read_bytes())
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(self.set_mode, self.root).result(timeout=5)
        self.assertEqual("exact_target", datasets.load_market_data_dataset_state(self.root)["datasets"][_DATASET]["scan_freshness_mode"])


if __name__ == "__main__":
    unittest.main()
