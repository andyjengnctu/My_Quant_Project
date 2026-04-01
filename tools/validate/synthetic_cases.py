from .synthetic_param_cases import (
    validate_synthetic_history_ev_threshold_case,
    validate_synthetic_param_guardrail_case,
    validate_synthetic_pit_same_day_exit_excluded_case,
    validate_synthetic_proj_cost_cash_capped_case,
    validate_synthetic_single_backtest_not_gated_by_own_history_case,
)
from .synthetic_portfolio_cases import (
    validate_synthetic_competing_candidates_case,
    validate_synthetic_exit_orders_only_for_held_positions_case,
    validate_synthetic_extended_miss_buy_case,
    validate_synthetic_fee_tax_net_equity_case,
    validate_synthetic_half_tp_full_year_case,
    validate_synthetic_intraday_reprice_forbidden_case,
    validate_synthetic_no_intraday_switch_after_failed_fill_case,
    validate_synthetic_rotation_t_plus_one_case,
    validate_synthetic_same_bar_stop_priority_case,
    validate_synthetic_same_day_buy_sell_forbidden_case,
    validate_synthetic_same_day_sell_block_case,
    validate_synthetic_unexecutable_half_tp_case,
)


def run_synthetic_consistency_suite(base_params):
    all_results = []
    summaries = []
    validators = [
        validate_synthetic_same_day_buy_sell_forbidden_case,
        validate_synthetic_intraday_reprice_forbidden_case,
        validate_synthetic_no_intraday_switch_after_failed_fill_case,
        validate_synthetic_exit_orders_only_for_held_positions_case,
        validate_synthetic_fee_tax_net_equity_case,
        validate_synthetic_half_tp_full_year_case,
        validate_synthetic_same_bar_stop_priority_case,
        validate_synthetic_extended_miss_buy_case,
        validate_synthetic_competing_candidates_case,
        validate_synthetic_same_day_sell_block_case,
        validate_synthetic_unexecutable_half_tp_case,
        validate_synthetic_rotation_t_plus_one_case,
        validate_synthetic_proj_cost_cash_capped_case,
        validate_synthetic_history_ev_threshold_case,
        validate_synthetic_pit_same_day_exit_excluded_case,
        validate_synthetic_single_backtest_not_gated_by_own_history_case,
        validate_synthetic_param_guardrail_case,
    ]

    for validator in validators:
        results, summary = validator(base_params)
        all_results.extend(results)
        summaries.append(summary)

    return all_results, summaries
