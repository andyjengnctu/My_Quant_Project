from .synthetic_param_cases import (
    validate_synthetic_history_ev_threshold_case,
    validate_synthetic_param_guardrail_case,
    validate_synthetic_pit_same_day_exit_excluded_case,
    validate_synthetic_proj_cost_cash_capped_case,
    validate_synthetic_single_backtest_not_gated_by_own_history_case,
)
from .synthetic_portfolio_cases import (
    validate_synthetic_competing_candidates_case,
    validate_synthetic_extended_miss_buy_case,
    validate_synthetic_half_tp_full_year_case,
    validate_synthetic_missed_buy_no_replacement_case,
    validate_synthetic_rotation_t_plus_one_case,
    validate_synthetic_same_bar_stop_priority_case,
    validate_synthetic_same_day_sell_block_case,
    validate_synthetic_unexecutable_half_tp_case,
)


def run_synthetic_consistency_suite(base_params):
    all_results = []
    summaries = []
    validators = [
        validate_synthetic_half_tp_full_year_case,
        validate_synthetic_same_bar_stop_priority_case,
        validate_synthetic_extended_miss_buy_case,
        validate_synthetic_missed_buy_no_replacement_case,
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
