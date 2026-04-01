from .synthetic_param_cases import (
    validate_synthetic_history_ev_threshold_case,
    validate_synthetic_lookahead_prev_day_only_case,
    validate_synthetic_param_guardrail_case,
    validate_synthetic_pit_same_day_exit_excluded_case,
    validate_synthetic_portfolio_history_filter_only_case,
    validate_synthetic_proj_cost_cash_capped_case,
    validate_synthetic_single_backtest_not_gated_by_own_history_case,
)
from .synthetic_portfolio_cases import (
    validate_synthetic_competing_candidates_case,
    validate_synthetic_candidate_order_fill_layer_separation_case,
    validate_synthetic_exit_orders_only_for_held_positions_case,
    validate_synthetic_extended_miss_buy_case,
    validate_synthetic_fee_tax_net_equity_case,
    validate_synthetic_missed_sell_accounting_case,
    validate_synthetic_half_tp_full_year_case,
    validate_synthetic_round_trip_pnl_only_on_tail_exit_case,
    validate_synthetic_intraday_reprice_forbidden_case,
    validate_synthetic_no_intraday_switch_after_failed_fill_case,
    validate_synthetic_rotation_t_plus_one_case,
    validate_synthetic_same_bar_stop_priority_case,
    validate_synthetic_same_day_buy_sell_forbidden_case,
    validate_synthetic_same_day_sell_block_case,
    validate_synthetic_unexecutable_half_tp_case,
)
from .synthetic_unit_cases import (
    validate_history_filters_unit_case,
    validate_independent_oracle_golden_case,
    validate_portfolio_stats_unit_case,
    validate_price_utils_unit_case,
)
from .synthetic_meta_cases import (
    validate_cmd_document_contract_case,
    validate_known_bad_fault_injection_case,
    validate_registry_checklist_entry_consistency_case,
)
from .synthetic_display_cases import (
    validate_display_reporting_sanity_case,
)
from .synthetic_reporting_cases import (
    validate_portfolio_yearly_report_schema_case,
    validate_test_suite_summary_reporting_case,
    validate_validate_console_summary_reporting_case,
)
from .synthetic_contract_cases import (
    validate_artifact_lifecycle_contract_case,
    validate_output_contract_case,
)
from .synthetic_error_cases import (
    validate_module_loader_error_path_case,
    validate_params_io_error_path_case,
    validate_preflight_error_path_case,
)
from .synthetic_data_quality_cases import (
    validate_load_clean_df_data_quality_case,
    validate_sanitize_ohlcv_expected_behavior_case,
    validate_sanitize_ohlcv_failfast_case,
)
from .synthetic_cli_cases import (
    validate_dataset_cli_contract_case,
    validate_local_regression_cli_contract_case,
)


def get_synthetic_validators():
    return [
        validate_synthetic_same_day_buy_sell_forbidden_case,
        validate_synthetic_intraday_reprice_forbidden_case,
        validate_synthetic_no_intraday_switch_after_failed_fill_case,
        validate_synthetic_exit_orders_only_for_held_positions_case,
        validate_synthetic_fee_tax_net_equity_case,
        validate_synthetic_missed_sell_accounting_case,
        validate_synthetic_round_trip_pnl_only_on_tail_exit_case,
        validate_synthetic_half_tp_full_year_case,
        validate_synthetic_same_bar_stop_priority_case,
        validate_synthetic_candidate_order_fill_layer_separation_case,
        validate_synthetic_extended_miss_buy_case,
        validate_synthetic_competing_candidates_case,
        validate_synthetic_same_day_sell_block_case,
        validate_synthetic_unexecutable_half_tp_case,
        validate_synthetic_rotation_t_plus_one_case,
        validate_synthetic_proj_cost_cash_capped_case,
        validate_synthetic_history_ev_threshold_case,
        validate_synthetic_portfolio_history_filter_only_case,
        validate_synthetic_lookahead_prev_day_only_case,
        validate_synthetic_pit_same_day_exit_excluded_case,
        validate_synthetic_single_backtest_not_gated_by_own_history_case,
        validate_synthetic_param_guardrail_case,
        validate_price_utils_unit_case,
        validate_history_filters_unit_case,
        validate_portfolio_stats_unit_case,
        validate_independent_oracle_golden_case,
        validate_registry_checklist_entry_consistency_case,
        validate_known_bad_fault_injection_case,
        validate_cmd_document_contract_case,
        validate_display_reporting_sanity_case,
        validate_validate_console_summary_reporting_case,
        validate_portfolio_yearly_report_schema_case,
        validate_test_suite_summary_reporting_case,
        validate_output_contract_case,
        validate_artifact_lifecycle_contract_case,
        validate_params_io_error_path_case,
        validate_module_loader_error_path_case,
        validate_preflight_error_path_case,
        validate_sanitize_ohlcv_expected_behavior_case,
        validate_sanitize_ohlcv_failfast_case,
        validate_load_clean_df_data_quality_case,
        validate_dataset_cli_contract_case,
        validate_local_regression_cli_contract_case,
    ]


def run_synthetic_consistency_suite(base_params):
    all_results = []
    summaries = []
    validators = get_synthetic_validators()

    for validator in validators:
        results, summary = validator(base_params)
        all_results.extend(results)
        summaries.append(summary)

    return all_results, summaries
