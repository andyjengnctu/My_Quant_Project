"""Oracle persists strategy evidence, not the cached benchmark report sidecar.

These tests exercise the real Portfolio/Oracle consumers with deliberately wrong
benchmark cache results. They do not certify complete Portfolio reports or any
future benchmark implementation as equivalent.
"""
from collections import OrderedDict
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from core import portfolio_benchmark, portfolio_engine
from core.file_integrity import canonical_json_sha256
from core.portfolio_entry_policy import (
    DAILY_EXECUTION_ATR_LIMIT, DAILY_EXECUTION_FULL_FILTERS, DAILY_EXECUTION_LEGACY,
    PortfolioExecutionContract,
)
from core.portfolio_fast_data import pack_prepared_stock_data, prep_stock_data_and_trades
from core.serialization_utils import json_native_value
from services.audit.target_oracle_romd import _metrics
from services.research.strategy_compare_replay import _unpack_result
from tools.validate.synthetic_daily_trade40_oracle_cases import source_fixture


def _native(value):
    if isinstance(value, np.ndarray):
        return _native(value.tolist())
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_native(item) for item in value]
    return json_native_value(value)


def _digest(value):
    return canonical_json_sha256(_native(value))


class OracleBenchmarkScopeTests(unittest.TestCase):
    def setUp(self):
        self.exchange, frames, _, schedule = source_fixture()
        self.params = schedule[0].params
        self.market, self.logs = {}, {}
        for ticker, frame in frames.items():
            prepared, self.logs[ticker] = prep_stock_data_and_trades(frame, self.params)
            self.market[ticker] = pack_prepared_stock_data(prepared)
        cache = patch.object(portfolio_benchmark, "_BENCHMARK_PERIOD_STATS_CACHE", OrderedDict())
        cache.start()
        self.addCleanup(cache.stop)

    def _run(self, contract):
        profile = {"_timing_enabled": False}
        counts, executions, selectors = {}, [], []
        result = portfolio_engine.run_portfolio_timeline(
            self.market, self.logs, list(self.exchange), 2021, self.params, 10, False,
            benchmark_ticker="2330", benchmark_data=self.market["2330"],
            is_training=False, verbose=False, profile_stats=profile,
            execution_contract=contract, replay_counts=counts,
            replay_execution_rows=executions, replay_selector_trace_rows=selectors,
        )
        return _unpack_result((*result, profile)), counts, executions, selectors

    def _assert_oracle_isolated(self, contract):
        baseline, base_counts, base_execution, base_selectors = self._run(contract)
        original = portfolio_engine._get_benchmark_period_stats

        def wrong_benchmark(**kwargs):
            stats = deepcopy(original(**kwargs))
            for key, value in stats.items():
                if key.startswith("bm_") and isinstance(value, (float, int)):
                    stats[key] = 987.123
                if isinstance(value, list):
                    for row in value:
                        for field in row:
                            if field.endswith("return_pct"):
                                row[field] = 987.123
            return stats

        with patch.object(portfolio_engine, "_get_benchmark_period_stats", side_effect=wrong_benchmark) as changed_cache:
            changed, counts, execution, selectors = self._run(contract)
        self.assertGreater(changed_cache.call_count, 0)
        self.assertGreater(baseline["trade_count"], 0)
        # Compare the complete persisted Oracle payload, including its strategy
        # yearly evidence, Win Rate and RoMD; no hand-selected output subset.
        expected = _metrics(baseline, target="benchmark-scope-probe")
        actual = _metrics(changed, target="benchmark-scope-probe")
        self.assertEqual(_digest(expected), _digest(actual))
        self.assertTrue(expected["yearly"])
        for field in baseline:
            if field != "profile":
                self.assertEqual(_digest(baseline[field]), _digest(changed[field]), field)
        for field in baseline["profile"]:
            if field != "bm_report_yearly_return_rows":
                self.assertEqual(_digest(baseline["profile"][field]),
                                 _digest(changed["profile"][field]), field)
        self.assertEqual(_digest(base_counts), _digest(counts))
        self.assertEqual(_digest(base_execution), _digest(execution))
        self.assertEqual(_digest(base_selectors), _digest(selectors))
        self.assertNotEqual(_digest(baseline["profile"]["bm_report_yearly_return_rows"]),
                            _digest(changed["profile"]["bm_report_yearly_return_rows"]))

    def test_breakout_oracle_evidence_ignores_changed_benchmark_sidecar(self):
        self._assert_oracle_isolated(PortfolioExecutionContract())

    def test_daily_legacy_oracle_evidence_ignores_changed_benchmark_sidecar(self):
        self._assert_oracle_isolated(PortfolioExecutionContract(
            "Daily", DAILY_EXECUTION_LEGACY, "daily_unscored"))

    def test_daily_atr_oracle_evidence_ignores_changed_benchmark_sidecar(self):
        self._assert_oracle_isolated(PortfolioExecutionContract(
            "Daily", DAILY_EXECUTION_ATR_LIMIT, "daily_unscored"))

    def test_daily_full_oracle_evidence_ignores_changed_benchmark_sidecar(self):
        self._assert_oracle_isolated(PortfolioExecutionContract(
            "Daily", DAILY_EXECUTION_FULL_FILTERS, "daily_unscored"))


if __name__ == "__main__":
    unittest.main()
