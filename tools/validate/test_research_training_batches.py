"""Scientific equivalence of optimized whole-date training batch planning."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from services.breakout_quality.train_continuous_ranker import _date_coherent_batches
from tools.benchmark_research_training_batches import reference_date_coherent_batches


class BinaryTrainingBatchTests(unittest.TestCase):
    def test_prefetch_preserves_sample_order_weights_and_tail(self):
        from services.breakout_quality.train import _iter_training_batches

        features = np.arange(60, dtype=np.float32).reshape(10, 3, 2)
        context = np.arange(10, dtype=np.float32).reshape(10, 1)
        labels = np.arange(10) % 2
        weights = np.linspace(0.25, 2.5, 10, dtype=np.float32)
        for ids in (np.array([7, 1, 7, 9, 0, 4, 2]), np.array([], dtype=np.int64)):
            for prefetch in (0, 1, 4):
                with self.subTest(prefetch=prefetch, count=len(ids)):
                    batches = list(_iter_training_batches(
                        features, context, labels, weights, ids,
                        batch_size=3, prefetch_batches=prefetch,
                    ))
                    self.assertEqual([len(batch[0]) for batch in batches],
                                     [3, 3, 1] if len(ids) else [])
                    for column, original in enumerate((features, context, labels, weights)):
                        actual = np.concatenate([batch[column] for batch in batches]) if batches else original[:0]
                        np.testing.assert_array_equal(actual, original[ids])

    def test_retired_market_input_is_rejected_before_forward(self):
        from filters.breakout_quality.inference import forward_breakout_quality_model

        class HistoricalModel:
            requires_market_set = True

            def __call__(self, *_args):
                raise AssertionError("A retired model must not execute")

        with self.assertRaisesRegex(ValueError, "Retired Market Set"):
            forward_breakout_quality_model(HistoricalModel(), None, None)


class ResearchTrainingBatchTests(unittest.TestCase):
    def assert_same_batches(self, ids, dates, *, batch_size, seed, single=False):
        options = dict(batch_size=batch_size, seed=seed, single_date_per_batch=single)
        expected = reference_date_coherent_batches(ids, dates, **options)
        actual = _date_coherent_batches(ids, dates, **options)
        self.assertEqual(len(expected), len(actual))
        for before, after in zip(expected, actual):
            np.testing.assert_array_equal(before, after)
            self.assertEqual(after.dtype, np.dtype("int64"))

    def test_randomized_dates_seeds_and_batch_boundaries(self):
        # Date counts straddle the logical batch limit; input rows deliberately
        # do not arrive in date order. Complete-date supervision must not split.
        rng = np.random.default_rng(391)
        for trial in range(32):
            counts = rng.integers(1, 90, size=25)
            dates = np.repeat(pd.bdate_range("2017-01-02", periods=len(counts)), counts)
            ids = rng.permutation(len(dates))[:rng.integers(1, len(dates) + 1)]
            for single in (False, True):
                with self.subTest(trial=trial, single=single):
                    self.assert_same_batches(ids, dates, batch_size=int(rng.integers(1, 150)),
                                             seed=trial, single=single)

    def test_duplicate_negative_ids_and_nondefault_series_index(self):
        dates = pd.Series(pd.to_datetime([
            "2021-01-04 15:30", "2021-01-05 09:00", "2021-01-04 12:00",
            "2021-01-06 12:00", "2021-01-05 14:00",
        ]), index=[80, 40, 20, 10, 90])
        ids = np.array([4, -5, 1, 2, 4, -1, 3, 0])
        for batch_size in (1, 3, 5, 100):
            for single in (False, True):
                self.assert_same_batches(ids, dates, batch_size=batch_size, seed=42, single=single)

    def test_timezone_and_empty_inputs(self):
        dates = pd.date_range("2021-03-13", periods=7, freq="12h", tz="America/New_York")
        for ids in (np.array([], dtype=np.int64), np.array([6, 2, 1, 3, 5, 0])):
            self.assert_same_batches(ids, dates, batch_size=3, seed=17)
        self.assert_same_batches(np.array([], dtype=np.int64), pd.DatetimeIndex([]),
                                 batch_size=3, seed=17)

    def test_nat_preserves_historical_dictionary_sort_order(self):
        dates = pd.to_datetime(["2021-01-05", None, "2021-01-04", None, "2021-01-06"])
        for ids in (np.arange(5), np.array([1, 4, 2, 3, 0]), np.array([1, 3])):
            for single in (False, True):
                self.assert_same_batches(ids, dates, batch_size=2, seed=31, single=single)

    def test_invalid_inputs_keep_exception_types(self):
        dates = pd.date_range("2021-01-01", periods=3)
        for ids, values in ((np.array([[0, 1]]), dates), (np.array([3]), dates),
                            (np.array([-4]), dates), (np.array([0]), ["invalid"])):
            with self.subTest(ids=ids.tolist()):
                with self.assertRaises(Exception) as expected:
                    reference_date_coherent_batches(ids, values, batch_size=2, seed=1)
                with self.assertRaises(type(expected.exception)):
                    _date_coherent_batches(ids, values, batch_size=2, seed=1)


if __name__ == "__main__":
    unittest.main()
