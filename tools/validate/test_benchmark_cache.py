"""Benchmark memoization must preserve the uncached numerical contract."""
import unittest
from unittest.mock import patch

import pandas as pd

from core import portfolio_benchmark as owner


class BenchmarkCacheTests(unittest.TestCase):
    def setUp(self):
        cache = patch.object(owner, "_BENCHMARK_PERIOD_STATS_CACHE", owner.OrderedDict())
        cache.start()
        self.addCleanup(cache.stop)
        self.dates = list(pd.date_range("2021-01-01", periods=4))

    def compare(self, data, dates=None, start=0):
        kwargs = dict(benchmark_data=data, sorted_dates=dates or self.dates, start_idx=start)
        actual = owner._get_benchmark_period_stats(**kwargs)
        self.assertEqual(actual, owner._build_benchmark_period_stats(**kwargs))
        return actual

    def test_recycled_object_identity_never_reuses_other_prices(self):
        # Deterministically simulate Python reassigning a released dict's ID.
        with patch.object(owner, "id", return_value=17, create=True):
            for last in (110.0, 112.0, 90.0):
                data = {self.dates[0]: {"Close": 100.0}, self.dates[-1]: {"Close": last}}
                self.compare(data)

    def test_changed_values_and_returned_rows_cannot_poison_cache(self):
        data = {day: {"Close": 100.0 + index} for index, day in enumerate(self.dates)}
        result = self.compare(data)
        result["bm_ret_pct"] = 999
        result["bm_monthly_return_rows"][0]["month_return_pct"] = 999
        self.compare(data)
        data[self.dates[-1]]["Close"] = 120.0
        self.compare(data)

    def test_interior_calendar_and_preperiod_anchor_are_effective_inputs(self):
        data = {day: {"Close": price} for day, price in zip(self.dates, (100.0, 50.0, 110.0, 120.0))}
        first = self.compare(data, self.dates[:2] + self.dates[-1:])
        second = self.compare(data, self.dates[:1] + self.dates[2:])
        self.assertNotEqual(first["bm_max_drawdown"], second["bm_max_drawdown"])
        del data[self.dates[1]]
        self.compare(data, start=1)
        data[self.dates[0]]["Close"] = 90.0
        self.compare(data, start=1)

    def test_identical_effective_inputs_reuse_bounded_cache(self):
        data = {day: {"Close": 100.0 + index} for index, day in enumerate(self.dates)}
        build = owner._build_benchmark_period_stats
        with patch.object(owner, "_build_benchmark_period_stats", wraps=build) as counted:
            first = owner._get_benchmark_period_stats(benchmark_data=data, sorted_dates=self.dates, start_idx=0)
            second = owner._get_benchmark_period_stats(benchmark_data=dict(data), sorted_dates=list(self.dates), start_idx=0)
        self.assertEqual(counted.call_count, 1)
        self.assertEqual(first, second)
        with patch.object(owner, "BENCHMARK_PERIOD_STATS_CACHE_MAX_ITEMS", 2):
            for end in (110.0, 112.0, 115.0):
                data[self.dates[-1]]["Close"] = end
                self.compare(data)
        self.assertEqual(len(owner._BENCHMARK_PERIOD_STATS_CACHE), 2)


if __name__ == "__main__":
    unittest.main()
