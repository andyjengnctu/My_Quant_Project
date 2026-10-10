"""Cross-domain, independent synthetic regression for raw market quote semantics.

Run directly (not the formal suite):
PYTHONPATH=. python -m unittest tools.validate.synthetic_market_raw_quote_cases -v
"""
from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd

from core.market_data_raw_quote import (
    raw_market_executable_quote_mask,
    raw_market_positive_volume_mask,
    raw_ohlc_diagnostic_masks,
)
from services.research.raw_price_strategy_dataset import _raw_quote_diagnostics, _raw_price_quality_failures
from services.trading.actual_fill_validation import (
    list_trading_actual_fill_dates, list_trading_positive_volume_dates,
    load_trading_actual_fill_market_evidence,
)
from services.trading.order_form_constraints import list_trading_pending_order_dates


class _FixtureMarketView:
    def __init__(self, raw_rows):
        self.rows = list(raw_rows)

    def read_dataset_frame(self, dataset, *, columns, data_id=None, start_date=None, end_date=None, **_kwargs):
        if dataset == "TaiwanStockTradingDate":
            return pd.DataFrame(
                [{"date": row["date"]} for row in self.rows if not end_date or row["date"] <= end_date],
                columns=columns,
            )
        if dataset != "TaiwanStockPrice":
            raise AssertionError(f"unexpected dataset {dataset}")
        chosen = [
            row for row in self.rows
            if (not data_id or row["stock_id"] == data_id)
            and (not start_date or row["date"] >= start_date)
            and (not end_date or row["date"] <= end_date)
        ]
        return pd.DataFrame(chosen, columns=columns)


class SharedRawMarketQuoteTests(unittest.TestCase):
    def test_quote_availability_is_distinct_from_positive_volume(self):
        rows = pd.DataFrame([
            {"max": 101.0, "min": 99.0, "Trading_Volume": 100},
            {"max": 0.0, "min": 0.0, "Trading_Volume": 100},
            {"max": 101.0, "min": 99.0, "Trading_Volume": 0},
            {"max": float("inf"), "min": 99.0, "Trading_Volume": 100},
            {"max": 90.0, "min": 100.0, "Trading_Volume": 100},
        ])
        self.assertEqual(raw_market_positive_volume_mask(rows).tolist(), [True, True, False, True, True])
        self.assertEqual(raw_market_executable_quote_mask(rows).tolist(), [True, False, False, False, False])

    def test_controlled_source_diagnostics_use_the_common_owner(self):
        rows = pd.DataFrame([
            {"Open": 0, "High": 0, "Low": 0, "Close": 0, "Volume": 50},
            {"Open": 105, "High": 104, "Low": 99, "Close": 101, "Volume": 100},
            {"Open": 100, "High": 101, "Low": 98, "Close": 100, "Volume": 100},
        ])
        masks = raw_ohlc_diagnostic_masks(rows)
        self.assertEqual(_raw_quote_diagnostics(rows), {name: int(mask.sum()) for name, mask in masks.items()})
        self.assertEqual(_raw_quote_diagnostics(rows)["all_zero_control_volume_positive"], 1)
        self.assertEqual(_raw_quote_diagnostics(rows)["nonzero_open_outside_high_low"], 1)
        # Reporting valid emerging-market Open semantics does NOT relax the
        # existing canonical model sanitizer or rewrite source OHLC.
        dated = rows.copy()
        dated["Date"] = pd.date_range("2021-01-01", periods=3, freq="B")
        invalid_count, _ = _raw_price_quality_failures(dated, ticker="1234")
        self.assertEqual(invalid_count, 2)

    def test_trading_fill_options_match_validation_but_order_dates_remain(self):
        dates = ["2026-09-10", "2026-09-11", "2026-09-12", "2026-09-13"]
        view = _FixtureMarketView([
            {"date": dates[0], "stock_id": "2330", "max": 101, "min": 99, "Trading_Volume": 100},
            {"date": dates[1], "stock_id": "2330", "max": 0, "min": 0, "Trading_Volume": 100},
            {"date": dates[2], "stock_id": "2330", "max": 101, "min": 99, "Trading_Volume": 0},
            {"date": dates[3], "stock_id": "2330", "max": 110, "min": 109, "Trading_Volume": 100},
        ])
        args = {"project_root": Path("."), "ticker": "2330", "market_view": view, "today": "2026-09-15"}
        self.assertEqual(list_trading_actual_fill_dates(**args), (dates[0], dates[3]))
        self.assertEqual(list_trading_positive_volume_dates(**args), (dates[0], dates[1], dates[3]))
        self.assertEqual(list_trading_pending_order_dates(
            Path("."), ticker="2330", market_view=view, latest_finalized_date="2026-09-15",
        ), (dates[0], dates[1], dates[3]))
        with self.assertRaisesRegex(ValueError, "raw 高低價資料不合法"):
            load_trading_actual_fill_market_evidence(
                Path("."), ticker="2330", trade_date=dates[1], market_view=view,
            )
        actual = load_trading_actual_fill_market_evidence(
            Path("."), ticker="2330", trade_date=dates[0], market_view=view,
        )
        self.assertEqual(actual["market_low"], 99.0)

    def test_duplicate_raw_rows_cannot_be_selected_for_fill(self):
        repeated = {"date": "2026-09-10", "stock_id": "2330", "max": 101, "min": 99, "Trading_Volume": 100}
        view = _FixtureMarketView([repeated, repeated.copy()])
        self.assertEqual(list_trading_actual_fill_dates(
            Path("."), ticker="2330", market_view=view, today="2026-09-15",
        ), ())
        self.assertEqual(list_trading_positive_volume_dates(
            Path("."), ticker="2330", market_view=view, today="2026-09-15",
        ), ("2026-09-10",))


if __name__ == "__main__":
    unittest.main()
