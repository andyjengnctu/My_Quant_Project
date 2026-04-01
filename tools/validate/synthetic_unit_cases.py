import pandas as pd

from core.config import V16StrategyParams
from core.history_filters import evaluate_history_candidate_metrics
from core.portfolio_stats import (
    build_full_year_return_stats,
    calc_annual_return_pct,
    calc_curve_stats,
    calc_sim_years,
    find_sim_start_idx,
)
from core.price_utils import (
    adjust_long_buy_limit,
    adjust_long_sell_fill_price,
    calc_entry_price,
    calc_half_take_profit_sell_qty,
    calc_net_sell_price,
    calc_position_size,
    can_execute_half_take_profit,
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
    import core.history_filters as history_filters

    case_id = "UNIT_HISTORY_FILTERS"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    params = V16StrategyParams()

    params.min_history_trades = 0
    params.min_history_ev = -1e9
    params.min_history_win_rate = 0.0

    original_ev_method = history_filters.EV_CALC_METHOD
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

        history_filters.EV_CALC_METHOD = "A"
        params.min_history_trades = 1
        params.min_history_ev = 0.4
        params.min_history_win_rate = 0.7
        method_a = evaluate_history_candidate_metrics(4, 3, 2.0, 3.0, -1.0, params)
        add_check(results, "unit_history_filters", case_id, "method_a_candidate_true", True, method_a[0])
        add_check(results, "unit_history_filters", case_id, "method_a_expected_value", 0.5, method_a[1], tol=1e-9)
        add_check(results, "unit_history_filters", case_id, "method_a_win_rate", 0.75, method_a[2], tol=1e-9)

        history_filters.EV_CALC_METHOD = "B"
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
        history_filters.EV_CALC_METHOD = original_ev_method

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
    import core.history_filters as history_filters

    case_id = 'UNIT_INDEPENDENT_ORACLE_GOLDEN'
    results = []
    summary = {'ticker': case_id, 'synthetic': True}
    params = V16StrategyParams()

    params.min_history_trades = 0
    params.min_history_ev = -1e9
    params.min_history_win_rate = 0.0

    original_ev_method = history_filters.EV_CALC_METHOD
    try:
        oracle_net = _oracle_net_sell_price(100.0, 10, params)
        prod_net = calc_net_sell_price(100.0, 10, params)
        add_check(results, 'unit_independent_oracle', case_id, 'oracle_net_sell_price_matches_production', oracle_net, prod_net, tol=1e-9)

        oracle_qty = _oracle_position_size(100.0, 95.0, 10_000.0, 0.02, params)
        prod_qty = calc_position_size(100.0, 95.0, 10_000.0, 0.02, params)
        add_check(results, 'unit_independent_oracle', case_id, 'oracle_position_size_matches_production', oracle_qty, prod_qty)

        history_filters.EV_CALC_METHOD = 'A'
        method_a = evaluate_history_candidate_metrics(4, 3, 2.0, 3.0, -1.0, params)
        add_check(results, 'unit_independent_oracle', case_id, 'oracle_history_ev_method_a_matches_production', _oracle_history_expected_value('A', 4, 3, 2.0, 3.0, -1.0), method_a[1], tol=1e-9)

        history_filters.EV_CALC_METHOD = 'B'
        method_b = evaluate_history_candidate_metrics(4, 2, 0.0, 4.0, -1.0, params)
        add_check(results, 'unit_independent_oracle', case_id, 'oracle_history_ev_method_b_matches_production', _oracle_history_expected_value('B', 4, 2, 0.0, 4.0, -1.0), method_b[1], tol=1e-9)

        add_check(results, 'unit_independent_oracle', case_id, 'oracle_calc_sim_years_matches_production', 366.0 / 365.25, calc_sim_years(list(pd.to_datetime(['2024-12-31', '2025-12-31'])), start_idx=0), tol=1e-9)
        add_check(results, 'unit_independent_oracle', case_id, 'oracle_annual_return_pct_matches_production', 10.0, calc_annual_return_pct(100.0, 121.0, 2.0), tol=1e-9)
    finally:
        history_filters.EV_CALC_METHOD = original_ev_method

    summary['oracle_checks'] = 6
    return results, summary
