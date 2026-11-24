"""Real publication-lock races must preserve the bytes a reader consumes."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import pandas as pd

from core import compare_csv_reader as reader
from core.compare_csv_storage import storage_lock
from filters.breakout_quality.csv_io import read_breakout_quality_csv


class CompareCsvReaderTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        runs = self.root / "outputs/strategy_compare/example/rolling/runs"
        self.source = runs / "original/pairs/baseline/no_filter_orderable_candidates.csv"
        self.copy = runs / "reused/pairs/baseline/no_filter_orderable_candidates.csv"
        self.bytes = "\ufeffticker,trade_date,value\n0050,2020-01-02,1.25\n2330,2020-01-03,\n".encode("utf-8")
        for path in (self.source, self.copy):
            path.parent.mkdir(parents=True)
            path.write_bytes(self.bytes)
            metadata = {} if path == self.source else {
                "output_scope": "reused_pair_cache",
                "cache_reused_from": self.source.parent.relative_to(self.root).as_posix(),
            }
            path.parent.joinpath("strategy_comparison.json").write_text(json.dumps({"metadata": metadata}))
        self.options = {"encoding": "utf-8-sig", "dtype": {"ticker": "string"}, "parse_dates": ["trade_date"]}

    def test_reader_pins_descriptor_before_waiting_cleaner_can_unlink(self):
        resolved, attempted, finished = (threading.Event() for _ in range(3))
        outcomes, errors = [], []
        original_resolve = reader.detail_read_path
        original_read = pd.read_csv

        def cleanup():
            try:
                if not resolved.wait(5):
                    raise AssertionError("reader did not reach path resolution")
                attempted.set()
                with storage_lock(self.root):
                    try:
                        self.copy.unlink()
                        outcomes.append("unlinked")
                    except PermissionError:
                        outcomes.append("open descriptor protected the Windows file")
            except (OSError, AssertionError, RuntimeError) as exc:
                errors.append(exc)
            finally:
                finished.set()

        def resolve(path):
            result = original_resolve(path)
            resolved.set()
            self.assertTrue(attempted.wait(5))
            self.assertFalse(finished.wait(0.1), "cleanup crossed the resolve/open lock boundary")
            return result

        def parse(stream, **options):
            # Parsing must not keep the publication lock. The cleaner can now
            # attempt unlink, while the already-open descriptor stays readable.
            self.assertTrue(finished.wait(5), "CSV parsing held the publication lock")
            return original_read(stream, **options)

        worker = threading.Thread(target=cleanup, daemon=True)
        worker.start()
        try:
            with patch.object(reader, "detail_read_path", side_effect=resolve), \
                    patch.object(pd, "read_csv", side_effect=parse):
                actual = reader.read_detail_csv(self.copy, **self.options)
        finally:
            resolved.set()
            worker.join(6)
        self.assertFalse(worker.is_alive())
        self.assertFalse(errors)
        self.assertEqual(len(outcomes), 1)
        pd.testing.assert_frame_equal(actual, original_read(self.source, **self.options))

    def test_cleaner_winning_the_lock_resolves_the_retained_source(self):
        started, finished = threading.Event(), threading.Event()
        results, errors = [], []

        def read():
            started.set()
            try:
                results.append(reader.read_detail_csv(self.copy, **self.options))
            except (OSError, ValueError, RuntimeError) as exc:
                errors.append(exc)
            finally:
                finished.set()

        with storage_lock(self.root):
            worker = threading.Thread(target=read, daemon=True)
            worker.start()
            self.assertTrue(started.wait(5))
            self.assertFalse(finished.wait(0.1))
            self.copy.unlink()
        worker.join(6)
        self.assertFalse(worker.is_alive())
        self.assertFalse(errors)
        self.assertEqual(len(results), 1)
        pd.testing.assert_frame_equal(results[0], pd.read_csv(self.source, **self.options))

    def test_dataframe_options_and_ticker_normalization_are_unchanged(self):
        pd.testing.assert_frame_equal(
            reader.read_detail_csv(self.copy, **self.options), pd.read_csv(self.copy, **self.options),
        )
        expected = read_breakout_quality_csv(self.copy, encoding="utf-8-sig")
        with reader.open_detail(self.copy) as stream:
            actual = read_breakout_quality_csv(stream, encoding="utf-8-sig")
        self.assertTrue(stream.closed)
        self.assertEqual(actual["ticker"].tolist(), ["0050", "2330"])
        pd.testing.assert_frame_equal(actual, expected)

    def test_descriptor_closes_when_the_consumer_raises(self):
        with self.assertRaisesRegex(ValueError, "parser failed"):
            with reader.open_detail(self.copy) as stream:
                raise ValueError("parser failed")
        self.assertTrue(stream.closed)


if __name__ == "__main__":
    unittest.main()
