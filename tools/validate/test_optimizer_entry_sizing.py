"""Canonical sizing survives optimizer cash caps without being recomputed."""
from copy import copy
import random
import unittest
from unittest.mock import patch

from core import entry_plans, portfolio_entry_plans
from core.exact_accounting import calc_cash_buy_total_milli_from_price, milli_to_money
from core.strategy_params import V16StrategyParams


def _candidate(params, *, price, stop, capital, max_qty=None, ticker="2330", date="2026-03-02"):
    seed = dict(limit_price=price, init_sl=stop, init_trail=stop,
                target_price=price + (price - stop) * 3, entry_atr=price - stop,
                ticker=ticker, security_profile=None, trade_date=date,
                sizing_capital=capital, max_qty=max_qty)
    sized = entry_plans.resize_candidate_plan_to_capital(seed, capital, params)
    qty = sized["qty"]
    row = dict(seed, limit_px=price, qty=qty, params_obj=params,
               use_breakout_quality_ranking=False, is_orderable=qty > 0,
               proj_cost_milli=calc_cash_buy_total_milli_from_price(price, qty) if qty > 0 else 0)
    return row


class OptimizerEntrySizingTests(unittest.TestCase):
    def _compare_cash_caps(self, params, row):
        if row["qty"] <= 0:
            return 0
        cost = row["proj_cost_milli"]
        budgets = {0, 1, cost // 2, cost - 1}
        for qty in (1, 999, 1000, 1001):
            boundary = calc_cash_buy_total_milli_from_price(row["limit_px"], qty)
            budgets.update(value for value in (boundary - 1, boundary, boundary + 1) if 0 <= value < cost)
        for budget in sorted(budgets):
            with self.subTest(price=row["limit_px"], qty=row["qty"], budget=budget):
                expected = portfolio_entry_plans._build_cash_capped_entry_plan_for_candidate(
                    row, None, budget, params, row["sizing_capital"], copy_shadow_position=False)
                before = dict(row)
                with patch.object(entry_plans, "resize_candidate_plan_to_capital", side_effect=AssertionError("repeated sizing")):
                    actual = portfolio_entry_plans._resolve_training_entry_plan(
                        row, budget, params, row["sizing_capital"])
                self.assertEqual(actual, expected)
                self.assertEqual(row, before)
                if actual is not None:
                    self.assertLessEqual(actual["reserved_cost_milli"], budget)
                    self.assertEqual(actual["reserved_cost"], milli_to_money(actual["reserved_cost_milli"]))
        self.assertIs(portfolio_entry_plans._resolve_training_entry_plan(
            row, cost, params, row["sizing_capital"]), row)
        return len(budgets)

    def test_cash_fee_lot_and_minimum_notional_boundaries(self):
        checks = 0
        for minimum in (0., 50_126.):
            params = V16StrategyParams(min_entry_notional=minimum, fixed_risk=.1, max_position_cap_pct=1.)
            for price, stop in ((5., 4.999), (31.45, 29.95), (500., 499.), (1000., 900.)):
                for max_qty in (None, 1, 999, 1000, 1001, 2459):
                    row = _candidate(params, price=price, stop=stop, capital=2_000_000.003, max_qty=max_qty)
                    checks += self._compare_cash_caps(params, row)
        self.assertGreater(checks, 100)

    def test_random_contexts_preserve_exact_plans(self):
        rng = random.Random(69211)
        for _ in range(120):
            params = V16StrategyParams(min_entry_notional=rng.choice((0., 50_126.)),
                                      fixed_risk=rng.choice((.005, .01, .04)),
                                      max_position_cap_pct=rng.choice((.1, .3, 1.)))
            price = rng.randint(1000, 1_000_000) / 1000
            row = _candidate(params, price=price, stop=price * rng.uniform(.7, .999),
                             capital=rng.uniform(10_000, 20_000_000),
                             max_qty=rng.choice((None, 999, 1000, 1001, 9876)),
                             ticker=rng.choice(("2330", "0050", "00679B", "01001T")),
                             date=rng.choice(("2026-03-02", "2027-01-02")))
            self._compare_cash_caps(params, row)

    def test_noncanonical_contexts_keep_generic_sizing(self):
        params = V16StrategyParams(min_entry_notional=0.)
        row = _candidate(params, price=10., stop=9.9, capital=1_000_000.)
        variants = [dict(row, params_obj=copy(params)),
                    dict(row, sizing_capital=None),
                    dict(row, sizing_capital=float('nan')),
                    dict(row, use_breakout_quality_ranking=True),
                    dict(row, _ensemble_context={})]
        for variant in variants:
            with patch.object(entry_plans, "resize_candidate_plan_to_capital",
                              wraps=entry_plans.resize_candidate_plan_to_capital) as resize:
                portfolio_entry_plans._resolve_training_entry_plan(variant, 1, params, 1_000_000.)
                resize.assert_called_once()


if __name__ == "__main__":
    unittest.main()
