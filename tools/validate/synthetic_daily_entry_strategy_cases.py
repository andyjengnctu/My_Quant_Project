"""New Daily entry primitive and one-owner Oracle Strategy regression guards.

Execute: python -m unittest tools.validate.synthetic_daily_entry_strategy_cases -q
No real market data, formal optimizer or signed replay is executed here.
"""
from __future__ import annotations

from unittest import TestCase, main
from unittest.mock import patch

import pandas as pd

from core.audit_policy import get_reusable_audit_definitions
from core.portfolio_candidates import _resolve_qualified_normal_setup
from core.portfolio_entry_policy import (
    DAILY_ENTRY_CONTRACT, DailyEntrySource, current_daily_entry_source,
    portfolio_entry_policy_context,
)
from core.research_report_contract import require_approved_report_contract
from services.audit import target_oracle_romd as oracle


class DailyEntryStrategyCases(TestCase):
    def test_true_daily_only_two_strategy_records(self):
        definition = next(d for d in get_reusable_audit_definitions("breakout_quality")
                          if d.audit_type == oracle.SUPPORTED_AUDIT_TYPE)
        rows = tuple(definition.source["oracle_comparisons"])
        expected = {
            "TrueDailyUniversal_MinTrade_NetPnL_Positive",
            "TrueDailyUniversal_MinTrade_NetPnL_Positive_Direct",
        }
        self.assertEqual({r["target"] for r in rows if oracle._oracle_target_strategy(definition, r["target"]) == "Daily"}, expected)
        self.assertEqual(len(rows), 18)
        for r in rows:
            if r["target"] not in expected:
                self.assertEqual(oracle._oracle_target_strategy(definition, r["target"]), "Breakout")
        self.assertEqual(oracle._oracle_target_strategy(definition, "C96"), "Breakout")
        contract = require_approved_report_contract("audit.target_oracle_romd")
        self.assertEqual(contract.version, 12)
        self.assertEqual(tuple(c.label for c in contract.sections[0].tables[0].columns[:3]),
                         ("Target", "Strategy", "Return Mean"))
        self.assertTrue(DAILY_ENTRY_CONTRACT.startswith("daily_stockday_close_limit_"))

    def test_scoped_daily_scores_do_not_leak_or_read_positive_truth(self):
        days = tuple(pd.Timestamp(t) for t in
                     ("2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07"))
        market = {
            "2330": {d: {"Close": 100.0} for d in days},
            "2317": {days[0]: {"Close": 90.0}, days[2]: {"Close": 92.0}},
        }
        keys = {
            ("2330", "2021-01-04"), ("2330", "2021-01-06"),
            ("2317", "2021-01-04"), ("00958B", "2021-01-04"),
        }
        self.assertIsNone(current_daily_entry_source())
        with portfolio_entry_policy_context("Breakout"):
            self.assertIsNone(current_daily_entry_source())
        with portfolio_entry_policy_context("Daily", score_keys=keys):
            source = current_daily_entry_source()
            self.assertIsInstance(source, DailyEntrySource)
            source.configure_calendar(days)
            # Market-covered score stocks only; today must follow information day.
            self.assertEqual(source.entries_for(days[1], market), {days[1]: [("2330", 0, 1)]})
            self.assertEqual(source.excluded_missing_market_ticker, 1)
            self.assertEqual(source.excluded_missing_adjacent_quote, 1)
            self.assertEqual(source.entries_for(days[2], market), {})
            self.assertEqual(source.entries_for(days[3], market), {days[3]: [("2330", 2, 3)]})
            with self.assertRaisesRegex(RuntimeError, "Nested Daily"):
                with portfolio_entry_policy_context("Daily", score_keys=keys):
                    pass
        self.assertIsNone(current_daily_entry_source())
        with self.assertRaises(ValueError):
            with portfolio_entry_policy_context("Daily"):
                pass
        self.assertIsNone(current_daily_entry_source())

    def test_entry_limit_prior_close_and_history_gate_is_breakout_only(self):
        d1, d2 = pd.Timestamp("2021-01-04"), pd.Timestamp("2021-01-05")
        # The old Breakout setup carries 110 limit, but Daily must use Close=100.
        fast = {d1: {"buy_limit": 110.0, "ATR": 3.0, "Close": 100.0},
                d2: {"buy_limit": 115.0, "ATR": 3.0, "Close": 105.0}}
        kwargs = dict(
            ticker="2330", y_pos=0, portfolio={}, sold_today=set(),
            all_dfs_fast={"2330": fast}, active_extended_signals={},
            pit_stats_index={"2330": []}, pit_stats_cursor={},
            today=d2, params=object(), profile_stats=None,
        )
        with patch("core.portfolio_candidates.get_pit_stats_from_index",
                   side_effect=AssertionError("Breakout history gate must not run for Daily")):
            daily = _resolve_qualified_normal_setup(**kwargs, entry_policy="daily")
        self.assertEqual(daily[0], 100.0)
        self.assertEqual(daily[1:4], (3.0, 100.0, d1))
        with patch("core.portfolio_candidates.get_pit_stats_from_index",
                   return_value=(True, 1.0, .5, 2, 10.0)) as historical:
            breakout = _resolve_qualified_normal_setup(**kwargs)
            historical.assert_called_once()
        self.assertEqual(breakout[0], 110.0)

    def test_real_portfolio_engine_changes_only_daily_buy_admission(self):
        # A small real canonical Portfolio Engine integration (not a fake
        # replay): the descending-price no-Breakout stock is score eligible.
        from tools.validate.synthetic_daily_trade40_oracle_cases import source_fixture
        from core.portfolio_fast_data import prep_stock_data_and_trades, pack_prepared_stock_data
        from core.portfolio_engine import run_portfolio_timeline
        exchange, frames, _bundle, schedule = source_fixture()
        params = schedule[0].params
        market, trade_logs = {}, {}
        for ticker, frame in frames.items():
            ready, logs = prep_stock_data_and_trades(frame, params)
            market[ticker] = pack_prepared_stock_data(ready)
            trade_logs[ticker] = logs
        keys = {(ticker, str(day.date())) for ticker in market
                for day in exchange if day.year == 2021}
        kwargs = dict(
            all_dfs_fast=market, all_standalone_logs=trade_logs,
            sorted_dates=list(exchange), start_year=2021,
            params=params, max_positions=10, enable_rotation=False,
            benchmark_ticker="2330", benchmark_data=market["2330"],
            is_training=False, verbose=False, profile_stats={},
        )
        breakout = run_portfolio_timeline(**kwargs)
        with portfolio_entry_policy_context("Daily", score_keys=keys):
            daily = run_portfolio_timeline(**kwargs)
        self.assertEqual(int(breakout[4]), 1)
        self.assertGreater(int(daily[4]), int(breakout[4]))
        self.assertIsNone(current_daily_entry_source())

    def test_declared_strategy_rejects_wrong_scientific_target(self):
        definition = next(d for d in get_reusable_audit_definitions("breakout_quality")
                          if d.audit_type == oracle.SUPPORTED_AUDIT_TYPE)
        # Fail-closed: a new unrelated Daily target must not inherit this policy.
        from dataclasses import replace
        source = dict(definition.source)
        source["oracle_comparisons"] = tuple(
            dict(row, strategy="Daily") if row["target"] == "T40" else row
            for row in definition.source["oracle_comparisons"]
        )
        corrupted = replace(definition, source=source)
        with self.assertRaisesRegex(oracle.OracleBlockedError, "signed TrueDaily"):
            oracle._oracle_target_strategy(corrupted, "T40")


if __name__ == "__main__":
    main()
