import copy

import numpy as np
import pandas as pd

from core.backtest_finalize import finalize_open_position_at_end
from core.config import V16StrategyParams
from core.entry_plans import build_cash_capped_entry_plan, build_position_from_entry_fill
from core.exact_accounting import (
    calc_reconciled_exit_display_pnl,
    build_buy_ledger_from_price,
    build_sell_ledger_from_price,
    calc_entry_total_cost,
    calc_exit_net_total,
    milli_to_money,
    money_to_milli,
    price_to_milli,
    sync_position_display_fields,
)
from core.history_filters import evaluate_history_candidate_metrics
from core.portfolio_stats import (
    build_full_year_return_stats,
    calc_annual_return_pct,
    calc_curve_stats,
    calc_sim_years,
    find_sim_start_idx,
)
from core.portfolio_exits import closeout_open_positions
from core.position_step import execute_bar_step
from core.price_utils import (
    adjust_long_buy_limit,
    adjust_long_sell_fill_price,
    calc_entry_price,
    calc_half_take_profit_sell_qty,
    calc_limit_down_price,
    calc_limit_up_price,
    calc_net_sell_price,
    calc_position_size,
    can_execute_half_take_profit,
    is_limit_down_bar,
    is_limit_up_bar,
)

from .checks import add_check


def validate_price_utils_unit_case(_base_params):
    params = V16StrategyParams()
    case_id = "UNIT_PRICE_UTILS"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    sized_qty = calc_position_size(100.0, 95.0, 10_000.0, 0.02, params)
    entry_cost = 100.0 * sized_qty + max(100.0 * sized_qty * params.buy_fee, params.min_fee)
    exit_net = 95.0 * sized_qty - max(95.0 * sized_qty * params.sell_fee, params.min_fee) - (95.0 * sized_qty * params.tax_rate)
    actual_risk = entry_cost - exit_net

    add_check(results, "unit_price_utils", case_id, "buy_limit_rounds_down_to_tick", 10.0, adjust_long_buy_limit(10.03), tol=1e-9)
    add_check(results, "unit_price_utils", case_id, "sell_fill_rounds_down_to_tick", 10.0, adjust_long_sell_fill_price(10.03), tol=1e-9)
    add_check(results, "unit_price_utils", case_id, "entry_price_uses_min_fee", 102.0, calc_entry_price(100.0, 10, params), tol=1e-9)
    add_check(results, "unit_price_utils", case_id, "net_sell_price_includes_fee_and_tax", 97.7, calc_net_sell_price(100.0, 10, params), tol=1e-9)
    add_check(results, "unit_price_utils", case_id, "position_size_expected_qty", 30, sized_qty)
    add_check(results, "unit_price_utils", case_id, "position_size_entry_cost_respects_cap", True, entry_cost <= 10_000.0)
    add_check(results, "unit_price_utils", case_id, "position_size_actual_risk_respects_limit", True, actual_risk <= 200.0 + 1e-9)
    add_check(results, "unit_price_utils", case_id, "invalid_stop_returns_zero_qty", 0, calc_position_size(100.0, 101.0, 10_000.0, 0.02, params))
    add_check(results, "unit_price_utils", case_id, "half_take_profit_qty_for_odd_lot", 1, calc_half_take_profit_sell_qty(3, 0.5))
    add_check(results, "unit_price_utils", case_id, "half_take_profit_qty_blocks_full_liquidation", 0, calc_half_take_profit_sell_qty(1, 0.5))
    add_check(results, "unit_price_utils", case_id, "can_execute_half_tp_false_for_single_share", False, can_execute_half_take_profit(1, 0.5))
    add_check(results, "unit_price_utils", case_id, "can_execute_half_tp_true_for_three_shares", True, can_execute_half_take_profit(3, 0.5))

    summary["checked_qty"] = sized_qty
    return results, summary


def validate_history_filters_unit_case(_base_params):
    import config.training_policy as training_policy

    case_id = "UNIT_HISTORY_FILTERS"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    params = V16StrategyParams()

    params.min_history_trades = 0
    params.min_history_ev = -1e9
    params.min_history_win_rate = 0.0

    original_ev_method = training_policy.EV_CALC_METHOD
    try:
        params.min_history_trades = 0
        params.min_history_ev = 0.0
        params.min_history_win_rate = 0.0
        zero_allowed = evaluate_history_candidate_metrics(0, 0, 0.0, 0.0, 0.0, params)
        add_check(results, "unit_history_filters", case_id, "zero_history_can_be_allowed", (True, 0.0, 0.0, 0), zero_allowed)

        params.min_history_trades = 5
        insufficient = evaluate_history_candidate_metrics(4, 3, 2.0, 3.0, -1.0, params)
        add_check(results, "unit_history_filters", case_id, "insufficient_trade_count_rejected", False, insufficient[0])
        add_check(results, "unit_history_filters", case_id, "insufficient_trade_count_preserved", 4, insufficient[3])

        training_policy.EV_CALC_METHOD = "A"
        params.min_history_trades = 1
        params.min_history_ev = 0.4
        params.min_history_win_rate = 0.7
        method_a = evaluate_history_candidate_metrics(4, 3, 2.0, 3.0, -1.0, params)
        add_check(results, "unit_history_filters", case_id, "method_a_candidate_true", True, method_a[0])
        add_check(results, "unit_history_filters", case_id, "method_a_expected_value", 0.5, method_a[1], tol=1e-9)
        add_check(results, "unit_history_filters", case_id, "method_a_win_rate", 0.75, method_a[2], tol=1e-9)

        training_policy.EV_CALC_METHOD = "B"
        params.min_history_ev = 1.4
        params.min_history_win_rate = 0.5
        method_b = evaluate_history_candidate_metrics(4, 2, 0.0, 4.0, -1.0, params)
        add_check(results, "unit_history_filters", case_id, "method_b_candidate_true", True, method_b[0])
        add_check(results, "unit_history_filters", case_id, "method_b_expected_value", 1.5, method_b[1], tol=1e-9)
        add_check(results, "unit_history_filters", case_id, "method_b_win_rate", 0.5, method_b[2], tol=1e-9)

        params.min_history_ev = 90.0
        params.min_history_win_rate = 1.0
        all_win = evaluate_history_candidate_metrics(3, 3, 0.0, 6.0, 0.0, params)
        add_check(results, "unit_history_filters", case_id, "method_b_all_win_candidate_true", True, all_win[0])
        add_check(results, "unit_history_filters", case_id, "method_b_all_win_payoff_cap_fallback", 99.9, all_win[1], tol=1e-9)

        params.min_history_ev = -0.5
        params.min_history_win_rate = 0.1
        all_loss = evaluate_history_candidate_metrics(3, 0, -3.0, 0.0, -3.0, params)
        add_check(results, "unit_history_filters", case_id, "method_b_all_loss_expected_value", -1.0, all_loss[1], tol=1e-9)
        add_check(results, "unit_history_filters", case_id, "method_b_all_loss_win_rate", 0.0, all_loss[2], tol=1e-9)
    finally:
        training_policy.EV_CALC_METHOD = original_ev_method

    summary["ev_methods_checked"] = ["A", "B"]
    return results, summary


def validate_portfolio_stats_unit_case(_base_params):
    case_id = "UNIT_PORTFOLIO_STATS"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    empty_r_sq, empty_monthly_win = calc_curve_stats([])
    growth_r_sq, growth_monthly_win = calc_curve_stats([100.0, 110.0, 121.0, 133.1])

    sorted_dates = list(pd.to_datetime([
        "2024-01-02",
        "2024-12-31",
        "2025-01-02",
        "2025-12-31",
    ]))
    full_year_stats = build_full_year_return_stats(
        sorted_dates,
        year_start_equity={2024: 100.0, 2025: 120.0},
        year_end_equity={2024: 110.0, 2025: 90.0},
        year_first_sim_date={2024: sorted_dates[1], 2025: sorted_dates[2]},
        year_last_sim_date={2024: sorted_dates[1], 2025: sorted_dates[3]},
    )
    sim_years = calc_sim_years(sorted_dates, start_idx=1)

    add_check(results, "unit_portfolio_stats", case_id, "empty_curve_r_squared", 0.0, empty_r_sq, tol=1e-12)
    add_check(results, "unit_portfolio_stats", case_id, "empty_curve_monthly_win_rate", 0.0, empty_monthly_win, tol=1e-12)
    add_check(results, "unit_portfolio_stats", case_id, "growth_curve_r_squared_near_one", True, growth_r_sq > 0.9999)
    add_check(results, "unit_portfolio_stats", case_id, "growth_curve_monthly_win_rate", 100.0, growth_monthly_win, tol=1e-9)
    add_check(results, "unit_portfolio_stats", case_id, "partial_year_excluded_from_full_year_count", 1, full_year_stats["full_year_count"])
    add_check(results, "unit_portfolio_stats", case_id, "partial_year_still_kept_in_rows", 2, len(full_year_stats["yearly_return_rows"]))
    add_check(results, "unit_portfolio_stats", case_id, "full_year_min_return_uses_only_complete_years", -25.0, full_year_stats["min_full_year_return_pct"], tol=1e-9)
    add_check(results, "unit_portfolio_stats", case_id, "find_sim_start_idx_hits_first_date_ge_start_year", 2, find_sim_start_idx(sorted_dates, 2025))
    add_check(results, "unit_portfolio_stats", case_id, "calc_sim_years_shared_period_basis", 366.0 / 365.25, sim_years, tol=1e-9)
    add_check(results, "unit_portfolio_stats", case_id, "calc_annual_return_pct_cagr", 10.0, calc_annual_return_pct(100.0, 121.0, 2.0), tol=1e-9)
    add_check(results, "unit_portfolio_stats", case_id, "calc_annual_return_pct_end_value_non_positive", -100.0, calc_annual_return_pct(100.0, 0.0, 2.0), tol=1e-9)

    summary["full_year_count"] = full_year_stats["full_year_count"]
    return results, summary



def validate_exact_accounting_ledger_conservation_case(_base_params):
    params = V16StrategyParams()
    case_id = "UNIT_EXACT_LEDGER"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    buy_ledger = build_buy_ledger_from_price(10.05, 3000, params)
    sell_ledger = build_sell_ledger_from_price(10.95, 3000, params)

    add_check(results, "unit_exact_accounting", case_id, "buy_ledger_gross_plus_fee_equals_net", buy_ledger["gross_buy_milli"] + buy_ledger["buy_fee_milli"], buy_ledger["net_buy_total_milli"])
    add_check(results, "unit_exact_accounting", case_id, "sell_ledger_gross_minus_fee_minus_tax_equals_net", sell_ledger["gross_sell_milli"] - sell_ledger["sell_fee_milli"] - sell_ledger["tax_milli"], sell_ledger["net_sell_total_milli"])
    add_check(results, "unit_exact_accounting", case_id, "entry_total_helper_matches_buy_ledger", milli_to_money(buy_ledger["net_buy_total_milli"]), calc_entry_total_cost(10.05, 3000, params), tol=1e-12)
    add_check(results, "unit_exact_accounting", case_id, "exit_total_helper_matches_sell_ledger", milli_to_money(sell_ledger["net_sell_total_milli"]), calc_exit_net_total(10.95, 3000, params), tol=1e-12)

    summary["buy_net_total_milli"] = buy_ledger["net_buy_total_milli"]
    return results, summary


def validate_exact_accounting_cost_basis_allocation_case(_base_params):
    params = V16StrategyParams()
    params.tp_percent = 0.5
    case_id = "UNIT_EXACT_COST_BASIS"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    position = build_position_from_entry_fill(100.0, 3, init_sl=95.0, init_trail=95.0, params=params, target_price=110.0)
    original_cost_basis_milli = position["remaining_cost_basis_milli"]

    position, _tp_freed_cash, _tp_pnl_realized, tp_events = execute_bar_step(
        position,
        y_atr=1.0,
        y_ind_sell=False,
        y_close=100.0,
        t_open=100.0,
        t_high=110.0,
        t_low=99.0,
        t_close=109.0,
        t_volume=1000.0,
        params=params,
    )
    tp_context = position.get("_last_exec_contexts", [])[0]

    position, _stop_freed_cash, _stop_pnl_realized, stop_events = execute_bar_step(
        position,
        y_atr=1.0,
        y_ind_sell=False,
        y_close=109.0,
        t_open=94.0,
        t_high=96.0,
        t_low=90.0,
        t_close=92.0,
        t_volume=1000.0,
        params=params,
    )
    stop_context = position.get("_last_exec_contexts", [])[0]

    add_check(results, "unit_exact_accounting", case_id, "tp_half_event_fired", True, "TP_HALF" in tp_events)
    add_check(results, "unit_exact_accounting", case_id, "stop_event_fired", True, "STOP" in stop_events)
    add_check(results, "unit_exact_accounting", case_id, "allocated_cost_basis_sums_back_to_original", original_cost_basis_milli, int(tp_context["allocated_cost_milli"]) + int(stop_context["allocated_cost_milli"]))
    add_check(results, "unit_exact_accounting", case_id, "remaining_cost_basis_zero_after_tail_exit", 0, position["remaining_cost_basis_milli"])
    add_check(results, "unit_exact_accounting", case_id, "realized_pnl_tracks_sum_of_legs", int(tp_context["pnl_milli"]) + int(stop_context["pnl_milli"]), position["realized_pnl_milli"])

    summary["original_cost_basis_milli"] = original_cost_basis_milli
    return results, summary


def validate_exact_accounting_tick_limit_integer_case(_base_params):
    case_id = "UNIT_EXACT_TICK_LIMIT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    up_limit = calc_limit_up_price(95.1)
    down_limit = calc_limit_down_price(95.1)
    near_up = np.nextafter(up_limit, 0.0)
    near_down = np.nextafter(down_limit, np.inf)

    add_check(results, "unit_exact_accounting", case_id, "limit_up_price_rounds_to_expected_tick", 104.6, up_limit, tol=1e-12)
    add_check(results, "unit_exact_accounting", case_id, "limit_down_price_rounds_to_expected_tick", 85.6, down_limit, tol=1e-12)
    add_check(results, "unit_exact_accounting", case_id, "nearby_float_normalizes_to_same_limit_up_milli", price_to_milli(up_limit), price_to_milli(near_up))
    add_check(results, "unit_exact_accounting", case_id, "nearby_float_normalizes_to_same_limit_down_milli", price_to_milli(down_limit), price_to_milli(near_down))
    add_check(results, "unit_exact_accounting", case_id, "limit_up_bar_uses_integer_price_comparison", True, is_limit_up_bar(near_up, near_up, near_up, near_up, 95.1))
    add_check(results, "unit_exact_accounting", case_id, "limit_down_bar_uses_integer_price_comparison", True, is_limit_down_bar(near_down, near_down, near_down, near_down, 95.1))

    summary["limit_up"] = up_limit
    return results, summary


def validate_exact_accounting_cash_risk_boundary_case(_base_params):
    params = V16StrategyParams()
    case_id = "UNIT_EXACT_CASH_RISK"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    candidate_plan = {
        "limit_price": 100.0,
        "init_sl": 95.0,
        "init_trail": 95.0,
        "target_price": 110.0,
        "entry_atr": 1.0,
    }
    resized = build_cash_capped_entry_plan(candidate_plan, 10_000.0, params)
    exact_cash = milli_to_money(resized["reserved_cost_milli"])
    below_cash = milli_to_money(resized["reserved_cost_milli"] - 1)

    accepted = build_cash_capped_entry_plan(candidate_plan, exact_cash, params)
    rejected = build_cash_capped_entry_plan(candidate_plan, below_cash, params)

    add_check(results, "unit_exact_accounting", case_id, "cash_cap_accepts_exact_reserved_total", True, accepted is not None)
    add_check(results, "unit_exact_accounting", case_id, "cash_cap_rejects_one_milli_shortfall", True, rejected is None)
    add_check(results, "unit_exact_accounting", case_id, "reserved_cost_stored_as_integer_total", resized["reserved_cost_milli"], money_to_milli(resized["reserved_cost"]))

    summary["reserved_cost_milli"] = resized["reserved_cost_milli"]
    return results, summary


def validate_exact_accounting_display_leg_reconciliation_case(_base_params):
    case_id = "UNIT_EXACT_DISPLAY_RECONCILIATION"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    position = {"display_realized_pnl_sum": 12.34}
    total_trade_pnl = 29.33
    reconciled_exit_pnl = calc_reconciled_exit_display_pnl(position, total_trade_pnl)

    add_check(results, "unit_exact_accounting", case_id, "reconciled_exit_leg_absorbs_rounding_residual", 16.99, reconciled_exit_pnl, tol=1e-12)
    add_check(results, "unit_exact_accounting", case_id, "display_leg_sum_matches_total_trade_pnl", 29.33, round(position["display_realized_pnl_sum"] + reconciled_exit_pnl, 2), tol=1e-12)

    summary["reconciled_exit_pnl"] = reconciled_exit_pnl
    return results, summary


def validate_exact_accounting_single_vs_portfolio_parity_case(_base_params):
    params = V16StrategyParams()
    case_id = "UNIT_EXACT_SINGLE_PORTFOLIO_PARITY"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    base_position = build_position_from_entry_fill(100.0, 10, init_sl=95.0, init_trail=95.0, params=params, target_price=110.0, entry_type="normal")
    single_position = copy.deepcopy(base_position)
    portfolio_position = copy.deepcopy(base_position)
    portfolio_position["last_px"] = 110.0
    initial_capital_milli = money_to_milli(params.initial_capital)
    starting_cash_milli = initial_capital_milli - base_position["net_buy_total_milli"]

    single_state = finalize_open_position_at_end(
        position=single_position,
        final_close=110.0,
        final_date=pd.Timestamp("2025-01-03"),
        current_capital_milli=starting_cash_milli,
        current_equity_milli=starting_cash_milli,
        peak_capital_milli=initial_capital_milli,
        max_drawdown_pct=0.0,
        trade_count=0,
        full_wins=0,
        total_profit_milli=0,
        total_loss_milli=0,
        total_r_multiple=0.0,
        total_r_win=0.0,
        total_r_loss=0.0,
        trade_logs=[],
        return_logs=False,
        params=params,
    )
    closed_trades_stats = []
    portfolio_cash_milli, normal_trade_count, extended_trade_count = closeout_open_positions(
        portfolio={"2330": portfolio_position},
        cash=starting_cash_milli,
        params=params,
        trade_history=[],
        is_training=True,
        closed_trades_stats=closed_trades_stats,
        normal_trade_count=0,
        extended_trade_count=0,
        last_date=pd.Timestamp("2025-01-03"),
    )

    add_check(results, "unit_exact_accounting", case_id, "single_and_portfolio_closeout_final_cash_match", single_state["current_capital_milli"], portfolio_cash_milli)
    add_check(results, "unit_exact_accounting", case_id, "single_and_portfolio_closeout_trade_pnl_match", milli_to_money(single_state["total_profit_milli"]), closed_trades_stats[0]["pnl"], tol=1e-12)
    add_check(results, "unit_exact_accounting", case_id, "portfolio_closeout_counts_normal_trade", 1, normal_trade_count)
    add_check(results, "unit_exact_accounting", case_id, "portfolio_closeout_keeps_extended_trade_count_zero", 0, extended_trade_count)

    summary["final_cash_milli"] = portfolio_cash_milli
    return results, summary


def validate_exact_accounting_display_derived_case(_base_params):
    params = V16StrategyParams()
    case_id = "UNIT_EXACT_DISPLAY_DERIVED"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    position = build_position_from_entry_fill(100.0, 10, init_sl=95.0, init_trail=95.0, params=params, target_price=110.0)
    position["entry"] = -1.0
    position["entry_capital_total"] = -1.0
    position["realized_pnl"] = -1.0
    sync_position_display_fields(position)

    add_check(results, "unit_exact_accounting", case_id, "entry_display_is_derived_from_integer_ledger", milli_to_money(position["net_buy_total_milli"]) / position["initial_qty"], position["entry"], tol=1e-12)
    add_check(results, "unit_exact_accounting", case_id, "entry_capital_display_is_derived_from_integer_ledger", milli_to_money(position["net_buy_total_milli"]), position["entry_capital_total"], tol=1e-12)
    add_check(results, "unit_exact_accounting", case_id, "realized_pnl_display_is_derived_from_integer_ledger", milli_to_money(position["realized_pnl_milli"]), position["realized_pnl"], tol=1e-12)

    summary["entry_capital_total"] = position["entry_capital_total"]
    return results, summary


def _oracle_net_sell_price(price: float, qty: int, params) -> float:
    gross = float(price) * int(qty)
    fee_total = max(gross * float(params.sell_fee), float(params.min_fee))
    tax_total = gross * float(params.tax_rate)
    return (gross - fee_total - tax_total) / max(int(qty), 1)


def _oracle_position_size(buy_price: float, stop_price: float, capital: float, risk_fraction: float, params) -> int:
    if stop_price >= buy_price or capital <= 0 or risk_fraction <= 0:
        return 0
    risk_budget = capital * risk_fraction
    max_qty = int(capital // buy_price)
    best_qty = 0
    for qty in range(max_qty, 0, -1):
        entry_total = buy_price * qty + max(buy_price * qty * params.buy_fee, params.min_fee)
        if entry_total > capital + 1e-9:
            continue
        exit_total = stop_price * qty - max(stop_price * qty * params.sell_fee, params.min_fee) - (stop_price * qty * params.tax_rate)
        if (entry_total - exit_total) <= risk_budget + 1e-9:
            best_qty = qty
            break
    return best_qty


def _oracle_history_expected_value(method: str, trade_count: int, win_count: int, total_r_sum: float, avg_win_r: float, avg_loss_r: float) -> float:
    if trade_count <= 0:
        return 0.0
    win_rate = win_count / trade_count
    if method == 'A':
        return total_r_sum / trade_count
    if avg_loss_r < 0:
        payoff = abs(avg_win_r / avg_loss_r) if avg_loss_r != 0 else 99.9
    elif win_count == trade_count:
        payoff = 99.9
    else:
        payoff = 0.0
    return win_rate * payoff - (1.0 - win_rate)


def validate_independent_oracle_golden_case(_base_params):
    import config.training_policy as training_policy

    case_id = 'UNIT_INDEPENDENT_ORACLE_GOLDEN'
    results = []
    summary = {'ticker': case_id, 'synthetic': True}
    params = V16StrategyParams()

    params.min_history_trades = 0
    params.min_history_ev = -1e9
    params.min_history_win_rate = 0.0

    original_ev_method = training_policy.EV_CALC_METHOD
    try:
        oracle_net = _oracle_net_sell_price(100.0, 10, params)
        prod_net = calc_net_sell_price(100.0, 10, params)
        add_check(results, 'unit_independent_oracle', case_id, 'oracle_net_sell_price_matches_production', oracle_net, prod_net, tol=1e-9)

        oracle_qty = _oracle_position_size(100.0, 95.0, 10_000.0, 0.02, params)
        prod_qty = calc_position_size(100.0, 95.0, 10_000.0, 0.02, params)
        add_check(results, 'unit_independent_oracle', case_id, 'oracle_position_size_matches_production', oracle_qty, prod_qty)

        training_policy.EV_CALC_METHOD = 'A'
        method_a = evaluate_history_candidate_metrics(4, 3, 2.0, 3.0, -1.0, params)
        add_check(results, 'unit_independent_oracle', case_id, 'oracle_history_ev_method_a_matches_production', _oracle_history_expected_value('A', 4, 3, 2.0, 3.0, -1.0), method_a[1], tol=1e-9)

        training_policy.EV_CALC_METHOD = 'B'
        method_b = evaluate_history_candidate_metrics(4, 2, 0.0, 4.0, -1.0, params)
        add_check(results, 'unit_independent_oracle', case_id, 'oracle_history_ev_method_b_matches_production', _oracle_history_expected_value('B', 4, 2, 0.0, 4.0, -1.0), method_b[1], tol=1e-9)

        add_check(results, 'unit_independent_oracle', case_id, 'oracle_calc_sim_years_matches_production', 366.0 / 365.25, calc_sim_years(list(pd.to_datetime(['2024-12-31', '2025-12-31'])), start_idx=0), tol=1e-9)
        add_check(results, 'unit_independent_oracle', case_id, 'oracle_annual_return_pct_matches_production', 10.0, calc_annual_return_pct(100.0, 121.0, 2.0), tol=1e-9)
    finally:
        training_policy.EV_CALC_METHOD = original_ev_method

    summary['oracle_checks'] = 6
    return results, summary
