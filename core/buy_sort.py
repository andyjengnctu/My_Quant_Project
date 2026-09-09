import math

from core.config import get_buy_sort_method


BUY_LIMIT_OVERAGE_SORT_METHOD = 'BUY_LIMIT_OVERAGE_THEN_PROJ_COST'
ENTRY_TYPE_THEN_PROJ_COST_SORT_METHOD = 'ENTRY_TYPE_THEN_PROJ_COST'

BREAKOUT_QUALITY_RANKING_POLICY_SCORE = 'score'
BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_ADJUSTED = 'capital-adjusted-score'
BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_BUCKET = 'capital-bucket-then-score'
BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_BINARY = 'resource-aware-binary'
BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_BINARY_BASKET = 'resource-aware-binary-basket'
BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS = 'resource-aware-continuous'
BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_CAPITAL_PRESERVING = (
    'resource-aware-continuous-capital-preserving'
)
BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL = (
    'resource-aware-continuous-max-dl'
)
BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT = (
    'resource-aware-continuous-max-dl-feasible-ascent'
)
BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_MATCHED_FEASIBLE_ASCENT = (
    'resource-aware-continuous-max-dl-matched-feasible-ascent'
)
BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_K_FLEX_R0_FEASIBLE_ASCENT = (
    'resource-aware-continuous-max-dl-k-flex-r0-feasible-ascent'
)
BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT_STALE_GUARD = (
    'resource-aware-continuous-max-dl-feasible-ascent-stale-score-guard'
)
BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXPECTED_PNL_FEASIBLE_ASCENT = (
    'resource-aware-continuous-expected-pnl-feasible-ascent'
)
BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_FEASIBLE_ASCENT = (
    'resource-aware-continuous-excess-alpha-feasible-ascent'
)
BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_NO_R0_FEASIBLE_ASCENT = (
    'resource-aware-continuous-excess-alpha-no-r0-feasible-ascent'
)
BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_CONSTRAINED_OPTIMAL = (
    'resource-aware-continuous-excess-alpha-constrained-optimal'
)
BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_CONSTRAINED_OPTIMAL = (
    'resource-aware-continuous-score-constrained-optimal'
)
BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_K_FLEX_R0_CONSTRAINED_OPTIMAL = (
    'resource-aware-continuous-score-k-flex-r0-constrained-optimal'
)
BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_SAFETY_CONSTRAINED_OPTIMAL = (
    'resource-aware-continuous-score-safety-constrained-optimal'
)
BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_RESIDUAL_SAFETY_CONSTRAINED_OPTIMAL = (
    'resource-aware-continuous-score-residual-safety-constrained-optimal'
)
BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_R0_CONSTRAINED_OPTIMAL = (
    'resource-aware-continuous-score-no-r0-constrained-optimal'
)
BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_K_NO_R0 = (
    'resource-aware-continuous-score-no-k-no-r0'
)
BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_K_NO_R0_RAW_SAFETY_GATE = (
    'resource-aware-continuous-score-no-k-no-r0-raw-safety-gate'
)
BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_K_NO_R0_SAFETY_MFE_PRODUCT = (
    'resource-aware-continuous-score-no-k-no-r0-safety-mfe-product'
)
BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_CAPITAL_NO_R0_CONSTRAINED_OPTIMAL = (
    'resource-aware-continuous-score-capital-no-r0-constrained-optimal'
)
BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_CAPITAL_PARETO_NO_R0_CONSTRAINED_OPTIMAL = (
    'resource-aware-continuous-score-capital-pareto-no-r0-constrained-optimal'
)
RESOURCE_AWARE_BREAKOUT_QUALITY_RANKING_POLICY_ORDER = (
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_BINARY,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_BINARY_BASKET,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_CAPITAL_PRESERVING,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_MATCHED_FEASIBLE_ASCENT,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_K_FLEX_R0_FEASIBLE_ASCENT,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT_STALE_GUARD,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXPECTED_PNL_FEASIBLE_ASCENT,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_FEASIBLE_ASCENT,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_NO_R0_FEASIBLE_ASCENT,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_CONSTRAINED_OPTIMAL,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_CONSTRAINED_OPTIMAL,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_K_FLEX_R0_CONSTRAINED_OPTIMAL,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_SAFETY_CONSTRAINED_OPTIMAL,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_RESIDUAL_SAFETY_CONSTRAINED_OPTIMAL,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_R0_CONSTRAINED_OPTIMAL,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_K_NO_R0,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_K_NO_R0_RAW_SAFETY_GATE,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_K_NO_R0_SAFETY_MFE_PRODUCT,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_CAPITAL_NO_R0_CONSTRAINED_OPTIMAL,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_CAPITAL_PARETO_NO_R0_CONSTRAINED_OPTIMAL,
)
RESOURCE_AWARE_BREAKOUT_QUALITY_RANKING_POLICIES = frozenset(
    RESOURCE_AWARE_BREAKOUT_QUALITY_RANKING_POLICY_ORDER
)
CONTINUOUS_SCORE_SOURCE_BREAKOUT_QUALITY_RANKING_POLICIES = (
    RESOURCE_AWARE_BREAKOUT_QUALITY_RANKING_POLICIES
    - {
        BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_BINARY,
        BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_BINARY_BASKET,
        BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_K_FLEX_R0_FEASIBLE_ASCENT,
        BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXPECTED_PNL_FEASIBLE_ASCENT,
        BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_FEASIBLE_ASCENT,
        BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_NO_R0_FEASIBLE_ASCENT,
        BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_CONSTRAINED_OPTIMAL,
    }
)

SUPPORTED_BREAKOUT_QUALITY_RANKING_POLICIES = (
    BREAKOUT_QUALITY_RANKING_POLICY_SCORE,
    BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_ADJUSTED,
    BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_BUCKET,
    *sorted(RESOURCE_AWARE_BREAKOUT_QUALITY_RANKING_POLICIES),
)
BREAKOUT_QUALITY_CAPITAL_BUCKET_COUNT = 3

# Canonical ranking-policy execution-description contract consumed by Strategy Compare metadata.
_DEFAULT_BREAKOUT_QUALITY_SCORE_RANKING_ORDER = (
    'capital_deployment_bucket_desc',
    'breakout_quality_score_desc',
    'existing_buy_sort',
    'ticker_deterministic',
)
_BREAKOUT_QUALITY_NON_RESOURCE_SCORE_ORDER_OVERRIDES = {
    BREAKOUT_QUALITY_RANKING_POLICY_SCORE: ('breakout_quality_score_desc', 'existing_buy_sort', 'ticker_deterministic'),
    BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_ADJUSTED: ('capital_adjusted_score_desc', 'existing_buy_sort', 'ticker_deterministic'),
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_RESIDUAL_SAFETY_CONSTRAINED_OPTIMAL: ('same_param_exact_resource_baseline', 'fixed_baseline_order_count_and_r0', 'frozen_primary_continuous_dl_score', 'frozen_secondary_low_adverse_score',
     'same_day_primary_and_safety_rank_percentiles', 'ols_expected_safety_rank_given_primary_rank',
     'baseline_relative_residual_safety_coverage_and_score_sum_floor', 'exact_branch_and_bound_full_candidate_universe',
     'canonical_cash_capped_plan_on_every_included_state', 'maximize_primary_continuous_dl_score_sum', 'global_optimum_certificate', 'ticker_deterministic'),
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_SAFETY_CONSTRAINED_OPTIMAL: ('same_param_exact_resource_baseline', 'fixed_baseline_order_count_and_r0', 'frozen_primary_continuous_dl_score', 'frozen_secondary_low_adverse_score',
     'baseline_relative_secondary_coverage_and_score_sum_floor', 'exact_branch_and_bound_full_candidate_universe',
     'canonical_cash_capped_plan_on_every_included_state', 'maximize_primary_continuous_dl_score_sum', 'global_optimum_certificate', 'ticker_deterministic'),
}

# value = (score_order, dl_intervention, quality_objective, resource_feasibility, selection_objective, fallback)
_BREAKOUT_QUALITY_RESOURCE_AWARE_REPORT_SPECS = {
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_BINARY: (('resource_bottleneck_gate', 'binary_pass_promotions', 'existing_buy_sort', 'ticker_deterministic'),
     'only_when_baseline_stops_before_free_slots_with_unselected_candidates', 'increase_reserved_capital_assigned_to_pass_candidates',
     'cash remains the binding pre-market resource after the selected basket', 'first_improving_pass_reserved_promotion', 'canonical_same_param_baseline_order'),
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_BINARY_BASKET: (('resource_bottleneck_gate', 'best_improvement_pass_basket', 'existing_buy_sort', 'ticker_deterministic'),
     'only_when_baseline_stops_before_free_slots_with_unselected_candidates', 'increase_reserved_capital_assigned_to_pass_candidates',
     'cash remains the binding pre-market resource after the selected basket', 'best_improvement_pass_reserved_then_pass_count_then_min_roos_rank',
     'canonical_same_param_baseline_order'),
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS: (('resource_bottleneck_gate', 'continuous_score_desc_if_cash_binding', 'existing_buy_sort_fallback', 'ticker_deterministic'),
     'only_when_baseline_stops_before_free_slots_with_unselected_candidates', 'maximize_selected_continuous_score_without_breaking_cash_binding',
     'cash remains the binding pre-market resource after the selected basket', 'continuous_score_desc_with_cash_binding_constrained_promotions',
     'canonical_same_param_baseline_order'),
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_CAPITAL_PRESERVING: (('same_param_exact_resource_baseline', 'continuous_score_desc', 'selected_count_not_below_baseline', 'reserved_capital_not_below_baseline',
      'existing_buy_sort_fallback', 'ticker_deterministic'),
     'cash_or_slot_binding_with_exact_baseline_resource_preservation', 'maximize_selected_continuous_score_subject_to_same_param_baseline_resource_floor',
     'selected_count>=baseline_selected_count and reserved_cost>=baseline_reserved_cost', 'best_improvement_continuous_score_with_exact_resource_floor',
     'canonical_same_param_baseline_order'),
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_CONSTRAINED_OPTIMAL: (('same_param_exact_resource_baseline', 'fixed_baseline_order_count_and_r0', 'frozen_mr13e_daily_percentile_to_pit_expected_excess_r',
      'exact_branch_and_bound_full_candidate_universe', 'canonical_cash_capped_plan_on_every_included_state',
      'maximize_expected_excess_r_times_canonical_planned_initial_risk', 'global_optimum_certificate', 'ticker_deterministic'),
     'all_days_with_feasible_alternative_baskets_under_fixed_baseline_order_count',
     'maximize_sum_expected_excess_r_times_canonical_planned_initial_risk_subject_to_same_k_r0_exact_global_search',
     'selected_count==baseline_selected_count and reserved_cost>=baseline_reserved_cost', 'exact_branch_and_bound_full_universe_subject_to_k_r0_and_canonical_cash',
     'c39_feasible_ascent_is_incumbent_lower_bound_only; exact_search_space_is_not_restricted'),
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_FEASIBLE_ASCENT: (('same_param_exact_resource_baseline', 'fixed_baseline_order_count', 'frozen_mr13e_daily_percentile_to_pit_expected_excess_r',
      'expected_excess_r_times_canonical_planned_initial_risk_top_k', 'deterministic_minimum_repair_seed', 'best_feasible_single_swap_excess_alpha_ascent',
      'one_swap_local_optimum', 'ticker_deterministic'),
     'all_days_with_feasible_alternative_baskets_under_fixed_baseline_order_count',
     'maximize_sum_expected_excess_r_times_canonical_planned_initial_risk_subject_to_same_k_r0',
     'selected_count==baseline_selected_count and reserved_cost>=baseline_reserved_cost',
     'expected_excess_alpha_top_k_repair_seed_then_best_feasible_single_swap_excess_alpha_ascent',
     'same_param_baseline_only_if_excess_alpha_minimum_repair_cannot_reach_k_r0_then_excess_alpha_ascent_continues'),
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_NO_R0_FEASIBLE_ASCENT: (('same_param_exact_resource_baseline', 'fixed_baseline_order_count', 'frozen_mr13e_daily_percentile_to_pit_expected_excess_r',
      'expected_excess_r_times_canonical_planned_initial_risk_top_k', 'no_baseline_reserved_capital_floor', 'no_r0_minimum_repair',
      'k_only_cash_feasible_seed_if_raw_top_k_cannot_place_k_orders', 'best_k_only_cash_feasible_single_swap_excess_alpha_ascent', 'one_swap_local_optimum',
      'ticker_deterministic'),
     'all_days_with_feasible_alternative_baskets_under_fixed_baseline_order_count',
     'maximize_sum_expected_excess_r_times_canonical_planned_initial_risk_subject_to_same_k_and_true_cash_only',
     'selected_count==baseline_selected_count; no baseline R0 floor; canonical cash-capped reservation remains binding',
     'expected_excess_alpha_top_k_then_k_only_cash_feasible_seed_if_needed_then_best_single_swap_ascent',
     'same_param_baseline_is_only_a_k_cash_feasible_seed_when_raw_top_k_cannot_place_k_orders; no_r0_repair'),
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXPECTED_PNL_FEASIBLE_ASCENT: (('same_param_exact_resource_baseline', 'fixed_baseline_order_count', 'frozen_mr13e_daily_percentile_to_expected_r',
      'expected_r_times_canonical_planned_initial_risk_top_k', 'deterministic_minimum_repair_seed', 'best_feasible_single_swap_expected_pnl_ascent',
      'one_swap_local_optimum', 'ticker_deterministic'),
     'all_days_with_feasible_alternative_baskets_under_fixed_baseline_order_count',
     'maximize_sum_expected_r_times_canonical_planned_initial_risk_subject_to_same_k_r0',
     'selected_count==baseline_selected_count and reserved_cost>=baseline_reserved_cost',
     'expected_pnl_top_k_repair_seed_then_best_feasible_single_swap_expected_pnl_ascent',
     'same_param_baseline_only_if_expected_pnl_minimum_repair_cannot_reach_k_r0_then_expected_pnl_ascent_continues'),
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL: (('same_param_exact_resource_baseline', 'fixed_baseline_order_count', 'continuous_score_top_k', 'deterministic_minimum_repair_if_needed',
      'reserved_capital_not_below_baseline', 'baseline_fallback_only_if_repair_fails', 'ticker_deterministic'),
     'all_days_with_feasible_alternative_baskets_under_fixed_baseline_order_count',
     'maximize_fixed_k_continuous_score_subject_to_same_param_baseline_reserved_capital_floor',
     'selected_count==baseline_selected_count and reserved_cost>=baseline_reserved_cost', 'dl_top_k_then_deterministic_minimum_repair',
     'canonical_same_param_baseline_order'),
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT: (('same_param_exact_resource_baseline', 'fixed_baseline_order_count', 'continuous_score_top_k', 'deterministic_minimum_repair_seed',
      'best_feasible_single_swap_ascent', 'one_swap_local_optimum', 'ticker_deterministic'),
     'all_days_with_feasible_alternative_baskets_under_fixed_baseline_order_count',
     'maximize_fixed_k_continuous_score_subject_to_same_param_baseline_reserved_capital_floor',
     'selected_count==baseline_selected_count and reserved_cost>=baseline_reserved_cost', 'dl_top_k_repair_seed_then_best_feasible_single_swap_ascent',
     'minimum_repair_seed_may_use_same_param_baseline_then_feasible_ascent_continues'),
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT_STALE_GUARD: (('same_param_exact_resource_baseline', 'fixed_baseline_order_count', 'continuous_score_top_k', 'deterministic_minimum_repair_seed',
      'stale_score_membership_guard', 'fresh_only_best_feasible_single_swap_ascent', 'one_swap_local_optimum', 'ticker_deterministic'),
     'all_days_with_feasible_alternative_baskets_under_fixed_baseline_order_count',
     'maximize_fixed_k_continuous_score_subject_to_same_param_baseline_reserved_capital_floor',
     'selected_count==baseline_selected_count and reserved_cost>=baseline_reserved_cost',
     'dl_top_k_repair_seed_then_stale_membership_guard_then_fresh_only_best_feasible_single_swap_ascent',
     'stale_score_membership_change_blocked_to_same_param_baseline_or_fresh_only_ascent'),
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_K_FLEX_R0_FEASIBLE_ASCENT: (('capital_deployment_bucket_desc', 'breakout_quality_score_desc', 'existing_buy_sort', 'ticker_deterministic'),
     'only_when_baseline_stops_before_free_slots_with_unselected_candidates',
     'maximize_local_feasible_count_then_continuous_score_subject_to_baseline_k_to_physical_free_slots_and_r0',
     'baseline_selected_count<=selected_count<=physical_free_slots and reserved_cost>=baseline_reserved_cost',
     'fixed_k_feasible_seed_then_best_feasible_single_add_or_swap_ascent_count_first_then_score',
     'same_fixed_k_feasible_seed_then_deterministic_local_add_swap_search; no_global_optimum_claim'),
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_MATCHED_FEASIBLE_ASCENT: (('same_param_exact_resource_baseline', 'fixed_baseline_order_count', 'continuous_score_top_k', 'deterministic_minimum_repair_seed',
      'best_feasible_single_swap_ascent', 'one_swap_local_optimum', 'ticker_deterministic'),
     'all_days_with_feasible_alternative_baskets_under_fixed_baseline_order_count',
     'maximize_fixed_k_continuous_score_subject_to_same_param_baseline_reserved_capital_floor',
     'selected_count==baseline_selected_count and reserved_cost>=baseline_reserved_cost', 'dl_top_k_repair_seed_then_best_feasible_single_swap_ascent',
     'minimum_repair_seed_may_use_same_param_baseline_then_feasible_ascent_continues'),
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_CAPITAL_NO_R0_CONSTRAINED_OPTIMAL: (('same_param_exact_resource_baseline', 'fixed_baseline_order_count_only_no_r0', 'frozen_continuous_dl_score', 'exact_branch_and_bound_full_candidate_universe',
      'canonical_cash_capped_plan_on_every_included_state', 'maximize_continuous_dl_score_times_reserved_capital_sum', 'global_optimum_certificate',
      'ticker_deterministic'),
     'all_days_with_feasible_alternative_baskets_under_fixed_baseline_order_count',
     'maximize_sum_continuous_dl_score_times_canonical_reserved_capital_subject_to_same_k_and_true_cash_only_exact_global_search',
     'selected_count==baseline_selected_count; no baseline R0 floor; canonical cash-capped reservation remains binding',
     'exact_branch_and_bound_full_universe_subject_to_k_and_canonical_cash_no_r0',
     'objective_top_k_if_cash_feasible_else_same_param_baseline_is_incumbent_only; no_r0_repair; exact_search_space_is_not_restricted'),
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_CAPITAL_PARETO_NO_R0_CONSTRAINED_OPTIMAL: (('same_param_exact_resource_baseline', 'fixed_baseline_order_count_only_no_r0', 'frozen_continuous_dl_score', 'exact_branch_and_bound_full_candidate_universe',
      'canonical_cash_capped_plan_on_every_included_state', 'maximize_score_coverage_first', 'derive_exact_score_and_reserved_capital_pareto_extremes',
      'normalize_quality_and_capital_by_pareto_extremes', 'maximize_normalized_quality_times_capital_product', 'global_optimum_certificate',
      'ticker_deterministic'),
     'all_days_with_feasible_alternative_baskets_under_fixed_baseline_order_count',
     'maximize_normalized_basket_score_quality_times_canonical_reserved_capital_using_exact_pareto_extremes_subject_to_same_k_and_true_cash_only',
     'selected_count==baseline_selected_count; no baseline R0 floor; canonical cash-capped reservation remains binding',
     'three_exact_passes_score_endpoint_capital_endpoint_then_normalized_product_global_optimum',
     'same_param_baseline_is_feasible_incumbent_only; score_and_capital_exact_endpoints_seed_final_product_pass; no_r0_repair'),
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_CONSTRAINED_OPTIMAL: (('same_param_exact_resource_baseline', 'fixed_baseline_order_count_and_r0', 'frozen_continuous_dl_score', 'exact_branch_and_bound_full_candidate_universe',
      'canonical_cash_capped_plan_on_every_included_state', 'maximize_continuous_dl_score_sum', 'global_optimum_certificate', 'ticker_deterministic'),
     'all_days_with_feasible_alternative_baskets_under_fixed_baseline_order_count', 'maximize_sum_continuous_dl_score_subject_to_same_k_r0_exact_global_search',
     'selected_count==baseline_selected_count and reserved_cost>=baseline_reserved_cost', 'exact_branch_and_bound_full_universe_subject_to_k_r0_and_canonical_cash',
     'c35_feasible_ascent_is_incumbent_lower_bound_only; exact_search_space_is_not_restricted'),
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_K_FLEX_R0_CONSTRAINED_OPTIMAL: (('same_param_exact_resource_baseline', 'baseline_order_count_is_minimum', 'physical_free_slots_is_maximum', 'baseline_r0_floor_preserved',
      'maximize_feasible_planned_order_count_first', 'frozen_continuous_dl_score', 'exact_branch_and_bound_full_candidate_universe',
      'canonical_cash_capped_plan_on_every_included_state', 'maximize_continuous_dl_score_sum_at_selected_count', 'global_optimum_certificate',
      'ticker_deterministic'),
     'all_days_with_feasible_baskets_between_baseline_k_and_physical_free_slots_while_preserving_r0',
     'maximize_feasible_count_then_sum_continuous_dl_score_subject_to_baseline_k_to_physical_free_slots_r0_exact_global_search',
     'baseline_selected_count<=selected_count<=physical_free_slots and reserved_cost>=baseline_reserved_cost',
     'descending_count_search_then_exact_branch_and_bound_maximize_score_subject_to_r0_and_canonical_cash',
     'try_counts_from_physical_free_slots_down_to_baseline_k; objective_top_count_and_deterministic_reserve_orders_are_incumbents_only; '
     'exact_search_space_is_not_restricted'),
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_K_NO_R0: (('capital_deployment_bucket_desc', 'breakout_quality_score_desc', 'existing_buy_sort', 'ticker_deterministic'),
     'all_days; score order may select any canonically cash-feasible count from zero to physical free slots',
     'existing_frozen_continuous_model_score_descending_then_original_deterministic_rank_tie',
     '0<=selected_count<=physical_free_slots; no baseline K/R0 floor; canonical cash-capped sizing/orderability remains authoritative',
     'deterministic_frozen_model_score_descending_then_canonical_cash_simulation; no combinatorial search',
     'none; same-param baseline is diagnostic only and never repairs/replaces score-order membership'),
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_K_NO_R0_RAW_SAFETY_GATE: (('capital_deployment_bucket_desc', 'breakout_quality_score_desc', 'existing_buy_sort', 'ticker_deterministic'),
     'all_days; score order may select any canonically cash-feasible count from zero to physical free slots',
     'raw_safety_same_day_orderable_percentile_gate_then_existing_frozen_conditional_mfe_score_descending',
     '0<=selected_count<=physical_free_slots; no baseline K/R0 floor; canonical cash-capped sizing/orderability remains authoritative',
     'raw_safety_gate_then_deterministic_frozen_model_score_descending_then_canonical_cash_simulation; no combinatorial search',
     'none; same-param baseline is diagnostic only and never repairs/replaces score-order membership'),
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_K_NO_R0_SAFETY_MFE_PRODUCT: (('capital_deployment_bucket_desc', 'breakout_quality_score_desc', 'existing_buy_sort', 'ticker_deterministic'),
     'all_days; score order may select any canonically cash-feasible count from zero to physical free slots',
     'same_day_orderable_raw_safety_percentile_times_conditional_mfe_percentile_descending',
     '0<=selected_count<=physical_free_slots; no baseline K/R0 floor; canonical cash-capped sizing/orderability remains authoritative',
     'same_day_raw_safety_percentile_x_conditional_mfe_percentile_descending_then_canonical_cash_simulation; no threshold; no fitted weight; no combinatorial '
     'search',
     'none; same-param baseline is diagnostic only and never repairs/replaces score-order membership'),
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_R0_CONSTRAINED_OPTIMAL: (('same_param_exact_resource_baseline', 'fixed_baseline_order_count_only_no_r0', 'frozen_continuous_dl_score', 'exact_branch_and_bound_full_candidate_universe',
      'canonical_cash_capped_plan_on_every_included_state', 'maximize_continuous_dl_score_sum', 'global_optimum_certificate', 'ticker_deterministic'),
     'all_days_with_feasible_alternative_baskets_under_fixed_baseline_order_count',
     'maximize_sum_continuous_dl_score_subject_to_same_k_and_true_cash_only_exact_global_search',
     'selected_count==baseline_selected_count; no baseline R0 floor; canonical cash-capped reservation remains binding',
     'exact_branch_and_bound_full_universe_subject_to_k_and_canonical_cash_no_r0',
     'objective_top_k_if_cash_feasible_else_same_param_baseline_is_incumbent_only; no_r0_repair; exact_search_space_is_not_restricted'),
}

_BREAKOUT_QUALITY_NO_K_NO_R0_REPORT_POLICIES = frozenset({
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_K_NO_R0,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_K_NO_R0_RAW_SAFETY_GATE,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_K_NO_R0_SAFETY_MFE_PRODUCT,
})


def build_breakout_quality_score_ranking_order(ranking_policy, *, include_runtime_member_vote_count=False):
    resource_spec = _BREAKOUT_QUALITY_RESOURCE_AWARE_REPORT_SPECS.get(ranking_policy)
    order = list(
        resource_spec[0]
        if resource_spec is not None
        else _BREAKOUT_QUALITY_NON_RESOURCE_SCORE_ORDER_OVERRIDES.get(
            ranking_policy,
            _DEFAULT_BREAKOUT_QUALITY_SCORE_RANKING_ORDER,
        )
    )
    if include_runtime_member_vote_count:
        order.insert(0, 'runtime_member_vote_count_desc')
    return order


def build_breakout_quality_capital_aware_ranking_contract(ranking_policy, *, ranking_options):
    resource_spec = _BREAKOUT_QUALITY_RESOURCE_AWARE_REPORT_SPECS.get(ranking_policy)
    if resource_spec is not None:
        _, dl_intervention, quality_objective, resource_feasibility, selection_objective, fallback = resource_spec
        contract = {
            'resource_gate': (
                'canonical_cash_sizing_orderability_and_physical_slots_only; baseline K/R0 diagnostic_only'
                if ranking_policy in _BREAKOUT_QUALITY_NO_K_NO_R0_REPORT_POLICIES
                else 'canonical_same_param_exact_cash_cap_baseline'
            ),
            'dl_intervention': dl_intervention,
            'quality_objective': quality_objective,
            'resource_feasibility': resource_feasibility,
            'selection_objective': selection_objective,
            'fallback': fallback,
            'future_target_used': False,
            'additional_numeric_thresholds': [],
        }
        if ranking_policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT_STALE_GUARD:
            contract['additional_numeric_thresholds'] = [{
                'name': 'stale_score_membership_guard_max_age_days',
                'value': ranking_options.get('stale_score_membership_guard_max_age_days'),
                'unit': 'calendar_days',
                'source': 'config/strategy_compare.py',
            }]
        return contract
    if ranking_policy == BREAKOUT_QUALITY_RANKING_POLICY_SCORE:
        return None
    return {
        'projected_capital_fraction_source': 'canonical_pretrade_proj_cost_div_sizing_capital',
        'deployment_rate': 'min(1, projected_capital_fraction / max_position_cap_pct)',
        'capital_bucket_count': BREAKOUT_QUALITY_CAPITAL_BUCKET_COUNT,
        'capital_bucket_scope': 'same_day_score_available_orderable_candidates',
        'future_target_used': False,
    }

def _as_finite_float(value, *, default):
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric_value):
        return default
    return numeric_value


def calc_buy_limit_overage_pct(prev_close, limit_price):
    resolved_prev_close = _as_finite_float(prev_close, default=math.inf)
    resolved_limit_price = _as_finite_float(limit_price, default=math.nan)
    if not math.isfinite(resolved_prev_close) or not math.isfinite(resolved_limit_price) or resolved_limit_price <= 0:
        return math.inf
    return max(0.0, (resolved_prev_close / resolved_limit_price - 1.0) * 100.0)


def calc_buy_limit_overage_pct_from_row(row):
    if row is None:
        return math.inf
    explicit_value = row.get('buy_limit_overage_pct')
    if explicit_value is not None:
        return _as_finite_float(explicit_value, default=math.inf)
    if row.get('prev_close') is not None or row.get('limit_price') is not None or row.get('limit_px') is not None:
        return calc_buy_limit_overage_pct(
            row.get('prev_close'),
            row.get('limit_price') if row.get('limit_price') is not None else row.get('limit_px'),
        )
    return _as_finite_float(row.get('sort_value'), default=math.inf)


def calc_entry_type_priority(candidate_type):
    normalized = str(candidate_type or '').strip().lower()
    if normalized in {'normal', 'buy', 'reentry'}:
        return 0
    if normalized in {'extended', 'extended_tbd', 'continuation'}:
        return 1
    return 2


def calc_entry_type_priority_from_row(row):
    if row is None:
        return 2
    return calc_entry_type_priority(
        row.get('type')
        or row.get('entry_source')
        or row.get('kind')
    )


def _descending_text_key(value):
    text = str(value or '')
    return tuple(-ord(ch) for ch in text)


def is_buy_limit_overage_sort(method=None):
    active_method = get_buy_sort_method() if method is None else method
    return active_method == BUY_LIMIT_OVERAGE_SORT_METHOD


def calc_buy_sort_value(method, ev, proj_cost, win_rate, trade_count, asset_growth_pct=0.0, prev_close=None, limit_price=None):
    if method == 'EV':
        return float(ev)
    if method == 'PROJ_COST':
        return float(proj_cost)
    if method == 'HIST_WIN_X_TRADES':
        return float(win_rate) * float(trade_count)
    if method == 'ASSET_GROWTH':
        return float(asset_growth_pct)
    if method == BUY_LIMIT_OVERAGE_SORT_METHOD:
        return calc_buy_limit_overage_pct(prev_close, limit_price)
    if method == ENTRY_TYPE_THEN_PROJ_COST_SORT_METHOD:
        return calc_buy_limit_overage_pct(prev_close, limit_price)
    raise ValueError(f"未知的 BUY_SORT_METHOD: {method}")


def calc_active_buy_sort_value(ev, proj_cost, win_rate, trade_count, asset_growth_pct=0.0, prev_close=None, limit_price=None):
    return calc_buy_sort_value(
        get_buy_sort_method(),
        ev,
        proj_cost,
        win_rate,
        trade_count,
        asset_growth_pct,
        prev_close=prev_close,
        limit_price=limit_price,
    )


def get_buy_sort_title(method=None):
    active_method = get_buy_sort_method() if method is None else method
    if active_method == 'EV':
        return '按期望值 (EV) 由大到小排序'
    if active_method == 'PROJ_COST':
        return '按預估投入資金由大到小排序'
    if active_method == 'HIST_WIN_X_TRADES':
        return '按歷史勝率 × 交易次數由大到小排序'
    if active_method == 'ASSET_GROWTH':
        return '按資產成長由大到小排序'
    if active_method == BUY_LIMIT_OVERAGE_SORT_METHOD:
        return '按買入限價超出幅度由小到大，再按預估投入資金排序'
    if active_method == ENTRY_TYPE_THEN_PROJ_COST_SORT_METHOD:
        return '按新突破/Re-entry優先，再按買入限價超出幅度由小到大排序'
    raise ValueError(f"未知的 BUY_SORT_METHOD: {active_method}")


def get_buy_sort_metric_label(method=None):
    active_method = get_buy_sort_method() if method is None else method
    if active_method == 'EV':
        return 'EV'
    if active_method == 'PROJ_COST':
        return '預估投入'
    if active_method == 'HIST_WIN_X_TRADES':
        return '勝率×次數'
    if active_method == 'ASSET_GROWTH':
        return '資產成長'
    if active_method == BUY_LIMIT_OVERAGE_SORT_METHOD:
        return '超限幅'
    if active_method == ENTRY_TYPE_THEN_PROJ_COST_SORT_METHOD:
        return '超限幅'
    raise ValueError(f"未知的 BUY_SORT_METHOD: {active_method}")


def format_buy_sort_metric_value(value, method=None):
    active_method = get_buy_sort_method() if method is None else method
    numeric_value = float(value)
    if active_method in {BUY_LIMIT_OVERAGE_SORT_METHOD, ENTRY_TYPE_THEN_PROJ_COST_SORT_METHOD} and not math.isfinite(numeric_value):
        return 'N/A'
    if active_method == 'EV':
        return f'{numeric_value:.2f}R'
    if active_method == 'PROJ_COST':
        return f'{numeric_value:,.0f}'
    if active_method == 'HIST_WIN_X_TRADES':
        return f'{numeric_value:.2f}'
    if active_method == 'ASSET_GROWTH':
        return f'{numeric_value:.2f}%'
    if active_method == BUY_LIMIT_OVERAGE_SORT_METHOD:
        return f'{numeric_value:.2f}%'
    if active_method == ENTRY_TYPE_THEN_PROJ_COST_SORT_METHOD:
        return f'{numeric_value:.2f}%'
    raise ValueError(f"未知的 BUY_SORT_METHOD: {active_method}")


def calc_projected_capital_metrics(*, proj_cost, sizing_capital, max_position_cap_pct):
    """Derive capital deployment from the canonical pre-trade sizing result."""

    cost = _as_finite_float(proj_cost, default=math.nan)
    capital = _as_finite_float(sizing_capital, default=math.nan)
    cap_fraction = _as_finite_float(max_position_cap_pct, default=math.nan)
    if (
        not math.isfinite(cost)
        or not math.isfinite(capital)
        or not math.isfinite(cap_fraction)
        or cost < 0.0
        or capital <= 0.0
        or cap_fraction <= 0.0
    ):
        return None, None
    projected_fraction = cost / capital
    deployment_rate = min(1.0, max(0.0, projected_fraction / cap_fraction))
    return float(projected_fraction), float(deployment_rate)


def _ranking_enabled_for_rows(rows):
    flags = {bool(item.get("use_breakout_quality_ranking", False)) for item in rows}
    if len(flags) > 1:
        raise ValueError("同一候選集合的 use_breakout_quality_ranking 不一致")
    return bool(flags and True in flags)


def resolve_breakout_quality_ranking_policy(rows):
    policies = {
        str(
            item.get("breakout_quality_ranking_policy")
            or BREAKOUT_QUALITY_RANKING_POLICY_SCORE
        ).strip()
        for item in rows
    }
    if len(policies) > 1:
        raise ValueError("同一候選集合的 breakout_quality_ranking_policy 不一致")
    policy = next(iter(policies), BREAKOUT_QUALITY_RANKING_POLICY_SCORE)
    if policy not in SUPPORTED_BREAKOUT_QUALITY_RANKING_POLICIES:
        raise ValueError(f"不支援的 breakout-quality ranking policy: {policy!r}")
    return policy


def _quality_score_parts(item):
    rank_payload = item.get("breakout_quality_rank")
    explicitly_unavailable = (
        isinstance(rank_payload, dict)
        and not bool(rank_payload.get("available", False))
    )
    score = _as_finite_float(item.get("breakout_quality_score"), default=math.nan)
    if explicitly_unavailable or not math.isfinite(score):
        return False, 0.0
    return True, float(score)


def _quality_score_desc_key(item):
    available, score = _quality_score_parts(item)
    if not available:
        # (AI註: 缺少PIT Score不是REJECT，也不得填0；排在有效Score後再完整沿用原buy-sort。)
        return (1, 0.0)
    return (0, -score)


def _capital_deployment_rate(item):
    explicit = _as_finite_float(
        item.get("projected_capital_deployment_rate"), default=math.nan
    )
    if math.isfinite(explicit) and 0.0 <= explicit <= 1.0:
        return float(explicit)
    projected_fraction, deployment_rate = calc_projected_capital_metrics(
        proj_cost=item.get("proj_cost"),
        sizing_capital=item.get("sizing_capital"),
        max_position_cap_pct=item.get("max_position_cap_pct"),
    )
    if projected_fraction is None or deployment_rate is None:
        raise ValueError(
            "capital-aware ranking候選缺少正式sizing資本欄位："
            f"ticker={item.get('ticker')}, proj_cost={item.get('proj_cost')}, "
            f"sizing_capital={item.get('sizing_capital')}, "
            f"max_position_cap_pct={item.get('max_position_cap_pct')}"
        )
    return float(deployment_rate)


def _linear_quantile(values, quantile):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("quantile values不可為空")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(quantile)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def build_breakout_quality_ranking_prefixes(rows):
    """Build deterministic quality-ranking prefixes for one candidate set."""

    rows = list(rows or [])
    if not rows or not _ranking_enabled_for_rows(rows):
        return {id(item): () for item in rows}
    policy = resolve_breakout_quality_ranking_policy(rows)
    if policy == BREAKOUT_QUALITY_RANKING_POLICY_SCORE:
        return {id(item): _quality_score_desc_key(item) for item in rows}
    if policy == BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_ADJUSTED:
        prefixes = {}
        for item in rows:
            available, score = _quality_score_parts(item)
            prefixes[id(item)] = (
                (1, 0.0)
                if not available
                else (0, -(score * _capital_deployment_rate(item)))
            )
        return prefixes
    if policy in RESOURCE_AWARE_BREAKOUT_QUALITY_RANKING_POLICIES:
        # (AI註: Resource-aware quality只在盤前資源瓶頸判定後介入；候選建立階段必須完整保留原Min ROOS順序。)
        return {id(item): () for item in rows}
    if policy == BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_BUCKET:
        available_rows = [item for item in rows if _quality_score_parts(item)[0]]
        rates = [_capital_deployment_rate(item) for item in available_rows]
        if rates:
            lower_boundary = _linear_quantile(rates, 1.0 / BREAKOUT_QUALITY_CAPITAL_BUCKET_COUNT)
            upper_boundary = _linear_quantile(rates, 2.0 / BREAKOUT_QUALITY_CAPITAL_BUCKET_COUNT)
        else:
            lower_boundary = upper_boundary = 0.0
        prefixes = {}
        for item in rows:
            available, score = _quality_score_parts(item)
            if not available:
                prefixes[id(item)] = (1, 0, 0.0)
                continue
            rate = _capital_deployment_rate(item)
            bucket_priority = 0 if rate >= upper_boundary else 1 if rate >= lower_boundary else 2
            prefixes[id(item)] = (0, bucket_priority, -score)
        return prefixes
    raise ValueError(f"不支援的 breakout-quality ranking policy: {policy!r}")


def sort_candidate_rows(rows, method=None):
    active_method = get_buy_sort_method() if method is None else method
    quality_ranking = _ranking_enabled_for_rows(rows) if rows else False
    ranking_policy = (
        resolve_breakout_quality_ranking_policy(rows) if quality_ranking else None
    )
    resource_aware_quality = (
        ranking_policy in RESOURCE_AWARE_BREAKOUT_QUALITY_RANKING_POLICIES
    )
    quality_prefixes = (
        build_breakout_quality_ranking_prefixes(rows)
        if quality_ranking and not resource_aware_quality
        else {}
    )
    quality_ranking_for_sort = bool(quality_ranking and not resource_aware_quality)

    def quality_prefix(item):
        return quality_prefixes.get(id(item), ())

    if active_method == BUY_LIMIT_OVERAGE_SORT_METHOD:
        if quality_ranking_for_sort:
            rows.sort(
                key=lambda item: (
                    *quality_prefix(item),
                    _as_finite_float(item.get('sort_value'), default=math.inf),
                    -_as_finite_float(item.get('proj_cost'), default=0.0),
                    _descending_text_key(item.get('ticker')),
                )
            )
        else:
            rows.sort(
                key=lambda item: (
                    _as_finite_float(item.get('sort_value'), default=math.inf),
                    -_as_finite_float(item.get('proj_cost'), default=0.0),
                    _descending_text_key(item.get('ticker')),
                )
            )
        return rows
    if active_method == ENTRY_TYPE_THEN_PROJ_COST_SORT_METHOD:
        if quality_ranking_for_sort:
            rows.sort(
                key=lambda item: (
                    *quality_prefix(item),
                    calc_entry_type_priority_from_row(item),
                    calc_buy_limit_overage_pct_from_row(item),
                    str(item.get('ticker') or ''),
                )
            )
        else:
            rows.sort(
                key=lambda item: (
                    calc_entry_type_priority_from_row(item),
                    calc_buy_limit_overage_pct_from_row(item),
                    str(item.get('ticker') or ''),
                )
            )
        return rows
    if quality_ranking_for_sort:
        rows.sort(
            key=lambda item: (
                *quality_prefix(item),
                -_as_finite_float(item.get('sort_value'), default=-math.inf),
                _descending_text_key(item.get('ticker')),
            )
        )
    else:
        rows.sort(
            key=lambda item: (
                _as_finite_float(item.get('sort_value'), default=-math.inf),
                str(item.get('ticker') or ''),
            ),
            reverse=True,
        )
    return rows


def is_sort_value_better(candidate_value, incumbent_value, method=None):
    active_method = get_buy_sort_method() if method is None else method
    smaller_is_better = active_method in {BUY_LIMIT_OVERAGE_SORT_METHOD, ENTRY_TYPE_THEN_PROJ_COST_SORT_METHOD}
    candidate_numeric = _as_finite_float(candidate_value, default=math.inf if smaller_is_better else -math.inf)
    incumbent_numeric = _as_finite_float(incumbent_value, default=math.inf if smaller_is_better else -math.inf)
    if smaller_is_better:
        return candidate_numeric < incumbent_numeric
    return candidate_numeric > incumbent_numeric
