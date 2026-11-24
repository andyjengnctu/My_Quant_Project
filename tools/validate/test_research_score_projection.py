"""Byte-level regressions for research projection hashing and bounded reuse."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from filters.breakout_quality import strategy_score_projection as projection
from tools.benchmark_research_score_projection import reference_projection_digest


def table_for(values):
    size = len(values)
    return pd.DataFrame({"score": values}, index=pd.MultiIndex.from_arrays([
        np.resize(["0050", "2330", "台積", "quoted\n\0ticker"], size),
        np.resize(["2026-01-02", "2026-01-05", "2026-01-06"], size),
    ], names=["ticker", "date"]))


class ProjectionDigestTests(unittest.TestCase):
    def assert_reference(self, table):
        expected = reference_projection_digest(table=table, score_column="score")
        self.assertEqual(projection._projection_digest(table=table, score_column="score"), expected)

    def test_float64_byte_contract_including_special_values(self):
        values = np.array([
            0.0, -0.0, 1.0, -1.0, np.nextafter(0.0, 1.0),
            np.finfo(np.float64).tiny, np.finfo(np.float64).max,
            np.nan, -np.nan, np.inf, -np.inf,
        ])
        self.assert_reference(table_for(values))
        positive_zero, negative_zero = table_for([0.0]), table_for([-0.0])
        self.assertNotEqual(
            projection._projection_digest(table=positive_zero, score_column="score"),
            projection._projection_digest(table=negative_zero, score_column="score"),
        )

    def test_random_bits_and_partial_final_block_preserve_order(self):
        rng = np.random.default_rng(20261124)
        bits = rng.integers(0, np.iinfo(np.uint64).max, size=17001, dtype=np.uint64)
        table = table_for(bits.view(np.float64))
        self.assert_reference(table)
        self.assert_reference(table.iloc[::-1])
        before = projection._projection_digest(table=table, score_column="score")
        changed = table.copy()
        changed.iloc[8192, 0] = 3.14159
        self.assertNotEqual(before, projection._projection_digest(table=changed, score_column="score"))

    def test_float32_conversion_and_empty_table(self):
        self.assert_reference(table_for(np.array([0.1, -0.0, np.nan], dtype=np.float32)))
        self.assert_reference(table_for(np.empty(0)))
        self.assert_reference(pd.DataFrame({"score": []}))

    def test_non_ascii_date_still_fails_but_unused_level_does_not(self):
        table = pd.DataFrame({"score": [1.]}, index=pd.MultiIndex.from_tuples([("0050", "二〇二六")]))
        with self.assertRaises(UnicodeEncodeError):
            projection._projection_digest(table=table, score_column="score")
        table = table_for([1.])
        table.index = table.index.set_levels(["2026-01-02", "unused非ASCII"], level=1)
        self.assert_reference(table)

    def test_missing_column_fails(self):
        with self.assertRaises(KeyError):
            projection._projection_digest(table=table_for([1.]), score_column="absent")

    def test_projection_cache_retains_multi_source_pass_and_invalidates_revision(self):
        projection.clear_selection_point_in_time_projection_cache()
        self.addCleanup(projection.clear_selection_point_in_time_projection_cache)
        table = table_for([1.])
        with patch.object(projection, "load_selection_point_in_time_score_table_from_path", return_value=table) as reader:
            first = [projection._selection_pit_projection_cached(
                f"source_{i}.csv", (1, 42, "a" * 64), ("score",),
            ) for i in range(20)]
            repeat = [projection._selection_pit_projection_cached(
                f"source_{i}.csv", (1, 42, "a" * 64), ("score",),
            ) for i in range(20)]
            self.assertEqual(first, repeat)
            self.assertEqual(reader.call_count, 20)
            reader.return_value = table_for([2.])
            changed = projection._selection_pit_projection_cached(
                "source_0.csv", (1, 42, "b" * 64), ("score",),
            )
            self.assertNotEqual(first[0], changed)
            self.assertEqual(reader.call_count, 21)
            projection.clear_selection_point_in_time_projection_cache()
            projection._selection_pit_projection_cached("source_0.csv", (1, 42, "b" * 64), ("score",))
            self.assertEqual(reader.call_count, 22)


if __name__ == "__main__":
    unittest.main()
