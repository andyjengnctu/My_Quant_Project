"""Declarative ranking policy contracts shared by validation and execution.

This leaf imports no runtime owner. Selector/decorator callables are bound by
Portfolio; report, dependency and validation consumers use the same descriptor.
Historical policy IDs and serialized report fields remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

BREAKOUT_QUALITY_RANKING_POLICY_SCORE = 'score'
BREAKOUT_QUALITY_RANKING_POLICY_POSITIVE_ONLY_DEFAULT = 'positive-only-default'
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

@dataclass(frozen=True)
class RankingPolicySpec:
    policy_id: str
    resource_aware: bool = False
    continuous_score_source: bool = False
    prefix_key: str = "baseline"
    selector_key: str | None = None
    selector_label: str | None = None
    static_options: tuple[tuple[str, Any], ...] = ()
    score_transform_key: str | None = None
    dynamic_options_key: str | None = None
    validation_capabilities: tuple[str, ...] = ()
    dependency_roles: tuple[str, ...] = ()
    ranking_order: tuple[str, ...] = ()
    score_kind: str = "continuous"
    report_spec: tuple | None = None
    capital_contract_enabled: bool = True
    no_baseline_resource_floor: bool = False

    @property
    def score_order(self) -> tuple[str, ...]:
        return self.report_spec[0] if self.report_spec is not None else self.ranking_order

    def has_validation(self, capability: str) -> bool:
        return capability in self.validation_capabilities


RANKING_POLICY_SPECS = (
    RankingPolicySpec(
        policy_id=BREAKOUT_QUALITY_RANKING_POLICY_SCORE,
        prefix_key='score',
        capital_contract_enabled=False,
        ranking_order=('breakout_quality_score_desc', 'existing_buy_sort', 'ticker_deterministic'),
    ),
    RankingPolicySpec(
        policy_id=BREAKOUT_QUALITY_RANKING_POLICY_POSITIVE_ONLY_DEFAULT,
        prefix_key='baseline',
        capital_contract_enabled=False,
        validation_capabilities=('primary_positive',),
        ranking_order=('existing_buy_sort', 'ticker_deterministic'),
    ),
    RankingPolicySpec(
        policy_id=BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_ADJUSTED,
        prefix_key='capital_adjusted',
        ranking_order=('capital_adjusted_score_desc', 'existing_buy_sort', 'ticker_deterministic'),
    ),
    RankingPolicySpec(
        policy_id=BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_BUCKET,
        prefix_key='capital_bucket',
        ranking_order=('capital_deployment_bucket_desc', 'breakout_quality_score_desc', 'existing_buy_sort', 'ticker_deterministic'),
    ),
    RankingPolicySpec(
        policy_id=BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_BINARY,
        resource_aware=True,
        prefix_key='baseline',
        selector_key='binary_greedy',
        score_kind='binary',
        selector_label='greedy-first-improvement',
        report_spec=(('resource_bottleneck_gate', 'binary_pass_promotions', 'existing_buy_sort', 'ticker_deterministic'),
         'only_when_baseline_stops_before_free_slots_with_unselected_candidates',
         'increase_reserved_capital_assigned_to_pass_candidates',
         'cash remains the binding pre-market resource after the selected basket',
         'first_improving_pass_reserved_promotion', 'canonical_same_param_baseline_order'),
    ),
    RankingPolicySpec(
        policy_id=BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_BINARY_BASKET,
        resource_aware=True,
        prefix_key='baseline',
        selector_key='binary_basket',
        score_kind='binary',
        selector_label='best-improvement-basket',
        report_spec=(('resource_bottleneck_gate', 'best_improvement_pass_basket', 'existing_buy_sort', 'ticker_deterministic'),
         'only_when_baseline_stops_before_free_slots_with_unselected_candidates',
         'increase_reserved_capital_assigned_to_pass_candidates',
         'cash remains the binding pre-market resource after the selected basket',
         'best_improvement_pass_reserved_then_pass_count_then_min_roos_rank', 'canonical_same_param_baseline_order'),
    ),
    RankingPolicySpec(
        policy_id=BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS,
        resource_aware=True,
        continuous_score_source=True,
        prefix_key='baseline',
        selector_key='continuous',
        selector_label='continuous-score-constrained',
        report_spec=(('resource_bottleneck_gate', 'continuous_score_desc_if_cash_binding', 'existing_buy_sort_fallback',
          'ticker_deterministic'),
         'only_when_baseline_stops_before_free_slots_with_unselected_candidates',
         'maximize_selected_continuous_score_without_breaking_cash_binding',
         'cash remains the binding pre-market resource after the selected basket',
         'continuous_score_desc_with_cash_binding_constrained_promotions', 'canonical_same_param_baseline_order'),
    ),
    RankingPolicySpec(
        policy_id=BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_CAPITAL_PRESERVING,
        resource_aware=True,
        continuous_score_source=True,
        prefix_key='baseline',
        selector_key='continuous_capital_preserving',
        selector_label='continuous-score-capital-preserving',
        report_spec=(('same_param_exact_resource_baseline', 'continuous_score_desc', 'selected_count_not_below_baseline',
          'reserved_capital_not_below_baseline', 'existing_buy_sort_fallback', 'ticker_deterministic'),
         'cash_or_slot_binding_with_exact_baseline_resource_preservation',
         'maximize_selected_continuous_score_subject_to_same_param_baseline_resource_floor',
         'selected_count>=baseline_selected_count and reserved_cost>=baseline_reserved_cost',
         'best_improvement_continuous_score_with_exact_resource_floor', 'canonical_same_param_baseline_order'),
    ),
    RankingPolicySpec(
        policy_id=BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL,
        resource_aware=True,
        continuous_score_source=True,
        prefix_key='baseline',
        selector_key='continuous_max_dl',
        selector_label='continuous-score-max-dl',
        report_spec=(('same_param_exact_resource_baseline', 'fixed_baseline_order_count', 'continuous_score_top_k',
          'deterministic_minimum_repair_if_needed', 'reserved_capital_not_below_baseline',
          'baseline_fallback_only_if_repair_fails', 'ticker_deterministic'),
         'all_days_with_feasible_alternative_baskets_under_fixed_baseline_order_count',
         'maximize_fixed_k_continuous_score_subject_to_same_param_baseline_reserved_capital_floor',
         'selected_count==baseline_selected_count and reserved_cost>=baseline_reserved_cost',
         'dl_top_k_then_deterministic_minimum_repair', 'canonical_same_param_baseline_order'),
    ),
    RankingPolicySpec(
        policy_id=BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT,
        resource_aware=True,
        continuous_score_source=True,
        prefix_key='baseline',
        selector_key='continuous_max_dl_feasible_ascent',
        selector_label='continuous-score-max-dl-feasible-ascent',
        report_spec=(('same_param_exact_resource_baseline', 'fixed_baseline_order_count', 'continuous_score_top_k',
          'deterministic_minimum_repair_seed', 'best_feasible_single_swap_ascent', 'one_swap_local_optimum',
          'ticker_deterministic'),
         'all_days_with_feasible_alternative_baskets_under_fixed_baseline_order_count',
         'maximize_fixed_k_continuous_score_subject_to_same_param_baseline_reserved_capital_floor',
         'selected_count==baseline_selected_count and reserved_cost>=baseline_reserved_cost',
         'dl_top_k_repair_seed_then_best_feasible_single_swap_ascent',
         'minimum_repair_seed_may_use_same_param_baseline_then_feasible_ascent_continues'),
    ),
    RankingPolicySpec(
        policy_id=BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_MATCHED_FEASIBLE_ASCENT,
        resource_aware=True,
        continuous_score_source=True,
        prefix_key='baseline',
        selector_key='continuous_max_dl_feasible_ascent',
        selector_label='greedy-first-improvement',
        static_options=(('matched_score_proposal', True),),
        report_spec=(('same_param_exact_resource_baseline', 'fixed_baseline_order_count', 'continuous_score_top_k',
          'deterministic_minimum_repair_seed', 'best_feasible_single_swap_ascent', 'one_swap_local_optimum',
          'ticker_deterministic'),
         'all_days_with_feasible_alternative_baskets_under_fixed_baseline_order_count',
         'maximize_fixed_k_continuous_score_subject_to_same_param_baseline_reserved_capital_floor',
         'selected_count==baseline_selected_count and reserved_cost>=baseline_reserved_cost',
         'dl_top_k_repair_seed_then_best_feasible_single_swap_ascent',
         'minimum_repair_seed_may_use_same_param_baseline_then_feasible_ascent_continues'),
    ),
    RankingPolicySpec(
        policy_id=BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_K_FLEX_R0_FEASIBLE_ASCENT,
        resource_aware=True,
        prefix_key='baseline',
        selector_key='continuous_max_dl_feasible_ascent',
        selector_label='greedy-first-improvement',
        static_options=(('allow_count_growth', True), ('matched_score_proposal', True)),
        validation_capabilities=('local_count_growth',),
        report_spec=(('capital_deployment_bucket_desc', 'breakout_quality_score_desc', 'existing_buy_sort',
          'ticker_deterministic'),
         'only_when_baseline_stops_before_free_slots_with_unselected_candidates',
         'maximize_local_feasible_count_then_continuous_score_subject_to_baseline_k_to_physical_free_slots_and_r0',
         'baseline_selected_count<=selected_count<=physical_free_slots and reserved_cost>=baseline_reserved_cost',
         'fixed_k_feasible_seed_then_best_feasible_single_add_or_swap_ascent_count_first_then_score',
         'same_fixed_k_feasible_seed_then_deterministic_local_add_swap_search; no_global_optimum_claim'),
    ),
    RankingPolicySpec(
        policy_id=BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT_STALE_GUARD,
        resource_aware=True,
        continuous_score_source=True,
        prefix_key='baseline',
        selector_key='continuous_max_dl_feasible_ascent',
        selector_label='continuous-score-max-dl-feasible-ascent-stale-score-guard',
        dynamic_options_key='stale_guard',
        validation_capabilities=('stale_guard',),
        report_spec=(('same_param_exact_resource_baseline', 'fixed_baseline_order_count', 'continuous_score_top_k',
          'deterministic_minimum_repair_seed', 'stale_score_membership_guard',
          'fresh_only_best_feasible_single_swap_ascent', 'one_swap_local_optimum', 'ticker_deterministic'),
         'all_days_with_feasible_alternative_baskets_under_fixed_baseline_order_count',
         'maximize_fixed_k_continuous_score_subject_to_same_param_baseline_reserved_capital_floor',
         'selected_count==baseline_selected_count and reserved_cost>=baseline_reserved_cost',
         'dl_top_k_repair_seed_then_stale_membership_guard_then_fresh_only_best_feasible_single_swap_ascent',
         'stale_score_membership_change_blocked_to_same_param_baseline_or_fresh_only_ascent'),
    ),
    RankingPolicySpec(
        policy_id=BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXPECTED_PNL_FEASIBLE_ASCENT,
        resource_aware=True,
        prefix_key='baseline',
        selector_key='continuous_max_dl_feasible_ascent',
        selector_label='continuous-expected-pnl-feasible-ascent',
        static_options=(('objective_mode', 'expected_pnl'),),
        validation_capabilities=('expected_pnl',),
        dependency_roles=('expected_r_calibration',),
        report_spec=(('same_param_exact_resource_baseline', 'fixed_baseline_order_count',
          'frozen_mr13e_daily_percentile_to_expected_r', 'expected_r_times_canonical_planned_initial_risk_top_k',
          'deterministic_minimum_repair_seed', 'best_feasible_single_swap_expected_pnl_ascent',
          'one_swap_local_optimum', 'ticker_deterministic'),
         'all_days_with_feasible_alternative_baskets_under_fixed_baseline_order_count',
         'maximize_sum_expected_r_times_canonical_planned_initial_risk_subject_to_same_k_r0',
         'selected_count==baseline_selected_count and reserved_cost>=baseline_reserved_cost',
         'expected_pnl_top_k_repair_seed_then_best_feasible_single_swap_expected_pnl_ascent',
         'same_param_baseline_only_if_expected_pnl_minimum_repair_cannot_reach_k_r0_then_expected_pnl_ascent_continues'),
    ),
    RankingPolicySpec(
        policy_id=BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_FEASIBLE_ASCENT,
        resource_aware=True,
        prefix_key='baseline',
        selector_key='continuous_max_dl_feasible_ascent',
        selector_label='continuous-excess-alpha-feasible-ascent',
        static_options=(('objective_mode', 'excess_alpha'),),
        validation_capabilities=('excess_alpha',),
        dependency_roles=('expected_excess_r_calibration',),
        report_spec=(('same_param_exact_resource_baseline', 'fixed_baseline_order_count',
          'frozen_mr13e_daily_percentile_to_pit_expected_excess_r',
          'expected_excess_r_times_canonical_planned_initial_risk_top_k', 'deterministic_minimum_repair_seed',
          'best_feasible_single_swap_excess_alpha_ascent', 'one_swap_local_optimum', 'ticker_deterministic'),
         'all_days_with_feasible_alternative_baskets_under_fixed_baseline_order_count',
         'maximize_sum_expected_excess_r_times_canonical_planned_initial_risk_subject_to_same_k_r0',
         'selected_count==baseline_selected_count and reserved_cost>=baseline_reserved_cost',
         'expected_excess_alpha_top_k_repair_seed_then_best_feasible_single_swap_excess_alpha_ascent',
         'same_param_baseline_only_if_excess_alpha_minimum_repair_cannot_reach_k_r0_then_excess_alpha_ascent_continues'),
    ),
    RankingPolicySpec(
        policy_id=BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_NO_R0_FEASIBLE_ASCENT,
        resource_aware=True,
        prefix_key='baseline',
        selector_key='continuous_max_dl_feasible_ascent',
        selector_label='continuous-excess-alpha-no-r0-feasible-ascent',
        static_options=(('objective_mode', 'excess_alpha'), ('preserve_reserve_floor', False), ('minimum_repair_enabled', False)),
        validation_capabilities=('excess_alpha_no_reserve_floor',),
        dependency_roles=('expected_excess_r_calibration',),
        report_spec=(('same_param_exact_resource_baseline', 'fixed_baseline_order_count',
          'frozen_mr13e_daily_percentile_to_pit_expected_excess_r',
          'expected_excess_r_times_canonical_planned_initial_risk_top_k', 'no_baseline_reserved_capital_floor',
          'no_r0_minimum_repair', 'k_only_cash_feasible_seed_if_raw_top_k_cannot_place_k_orders',
          'best_k_only_cash_feasible_single_swap_excess_alpha_ascent', 'one_swap_local_optimum',
          'ticker_deterministic'),
         'all_days_with_feasible_alternative_baskets_under_fixed_baseline_order_count',
         'maximize_sum_expected_excess_r_times_canonical_planned_initial_risk_subject_to_same_k_and_true_cash_only',
         'selected_count==baseline_selected_count; no baseline R0 floor; canonical cash-capped reservation remains '
         'binding',
         'expected_excess_alpha_top_k_then_k_only_cash_feasible_seed_if_needed_then_best_single_swap_ascent',
         'same_param_baseline_is_only_a_k_cash_feasible_seed_when_raw_top_k_cannot_place_k_orders; no_r0_repair'),
    ),
    RankingPolicySpec(
        policy_id=BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_CONSTRAINED_OPTIMAL,
        resource_aware=True,
        prefix_key='baseline',
        selector_key='continuous_excess_alpha_constrained_optimal',
        selector_label='continuous-excess-alpha-constrained-optimal',
        validation_capabilities=('exact_excess_alpha',),
        dependency_roles=('expected_excess_r_calibration',),
        report_spec=(('same_param_exact_resource_baseline', 'fixed_baseline_order_count_and_r0',
          'frozen_mr13e_daily_percentile_to_pit_expected_excess_r', 'exact_branch_and_bound_full_candidate_universe',
          'canonical_cash_capped_plan_on_every_included_state',
          'maximize_expected_excess_r_times_canonical_planned_initial_risk', 'global_optimum_certificate',
          'ticker_deterministic'),
         'all_days_with_feasible_alternative_baskets_under_fixed_baseline_order_count',
         'maximize_sum_expected_excess_r_times_canonical_planned_initial_risk_subject_to_same_k_r0_exact_global_search',
         'selected_count==baseline_selected_count and reserved_cost>=baseline_reserved_cost',
         'exact_branch_and_bound_full_universe_subject_to_k_r0_and_canonical_cash',
         'c39_feasible_ascent_is_incumbent_lower_bound_only; exact_search_space_is_not_restricted'),
    ),
    RankingPolicySpec(
        policy_id=BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_CONSTRAINED_OPTIMAL,
        resource_aware=True,
        continuous_score_source=True,
        prefix_key='baseline',
        selector_key='continuous_score_constrained_optimal',
        selector_label='continuous-score-constrained-optimal',
        validation_capabilities=('exact_score', 'positive_filter'),
        report_spec=(('same_param_exact_resource_baseline', 'fixed_baseline_order_count_and_r0', 'frozen_continuous_dl_score',
          'exact_branch_and_bound_full_candidate_universe', 'canonical_cash_capped_plan_on_every_included_state',
          'maximize_continuous_dl_score_sum', 'global_optimum_certificate', 'ticker_deterministic'),
         'all_days_with_feasible_alternative_baskets_under_fixed_baseline_order_count',
         'maximize_sum_continuous_dl_score_subject_to_same_k_r0_exact_global_search',
         'selected_count==baseline_selected_count and reserved_cost>=baseline_reserved_cost',
         'exact_branch_and_bound_full_universe_subject_to_k_r0_and_canonical_cash',
         'c35_feasible_ascent_is_incumbent_lower_bound_only; exact_search_space_is_not_restricted'),
    ),
    RankingPolicySpec(
        policy_id=BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_K_FLEX_R0_CONSTRAINED_OPTIMAL,
        resource_aware=True,
        continuous_score_source=True,
        prefix_key='baseline',
        selector_key='continuous_score_k_flex_r0_constrained_optimal',
        selector_label='continuous-score-k-flex-r0-constrained-optimal',
        validation_capabilities=('exact_count_growth',),
        report_spec=(('same_param_exact_resource_baseline', 'baseline_order_count_is_minimum', 'physical_free_slots_is_maximum',
          'baseline_r0_floor_preserved', 'maximize_feasible_planned_order_count_first', 'frozen_continuous_dl_score',
          'exact_branch_and_bound_full_candidate_universe', 'canonical_cash_capped_plan_on_every_included_state',
          'maximize_continuous_dl_score_sum_at_selected_count', 'global_optimum_certificate',
          'ticker_deterministic'),
         'all_days_with_feasible_baskets_between_baseline_k_and_physical_free_slots_while_preserving_r0',
         'maximize_feasible_count_then_sum_continuous_dl_score_subject_to_baseline_k_to_physical_free_slots_r0_exact_global_search',
         'baseline_selected_count<=selected_count<=physical_free_slots and reserved_cost>=baseline_reserved_cost',
         'descending_count_search_then_exact_branch_and_bound_maximize_score_subject_to_r0_and_canonical_cash',
         'try_counts_from_physical_free_slots_down_to_baseline_k; '
         'objective_top_count_and_deterministic_reserve_orders_are_incumbents_only; '
         'exact_search_space_is_not_restricted'),
    ),
    RankingPolicySpec(
        policy_id=BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_SAFETY_CONSTRAINED_OPTIMAL,
        resource_aware=True,
        continuous_score_source=True,
        prefix_key='baseline',
        selector_key='continuous_score_safety_constrained_optimal',
        selector_label='continuous-score-safety-constrained-optimal',
        validation_capabilities=('safety_floor',),
        dependency_roles=('safety',),
        ranking_order=('same_param_exact_resource_baseline', 'fixed_baseline_order_count_and_r0',
         'frozen_primary_continuous_dl_score', 'frozen_secondary_low_adverse_score',
         'baseline_relative_secondary_coverage_and_score_sum_floor', 'exact_branch_and_bound_full_candidate_universe',
         'canonical_cash_capped_plan_on_every_included_state', 'maximize_primary_continuous_dl_score_sum',
         'global_optimum_certificate', 'ticker_deterministic'),
    ),
    RankingPolicySpec(
        policy_id=BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_RESIDUAL_SAFETY_CONSTRAINED_OPTIMAL,
        resource_aware=True,
        continuous_score_source=True,
        prefix_key='baseline',
        selector_key='continuous_score_residual_safety_constrained_optimal',
        selector_label='continuous-score-residual-safety-constrained-optimal',
        score_transform_key='same_day_rank_residual_safety_scores',
        validation_capabilities=('safety_floor', 'residual_safety'),
        dependency_roles=('safety',),
        ranking_order=('same_param_exact_resource_baseline', 'fixed_baseline_order_count_and_r0',
         'frozen_primary_continuous_dl_score', 'frozen_secondary_low_adverse_score',
         'same_day_primary_and_safety_rank_percentiles', 'ols_expected_safety_rank_given_primary_rank',
         'baseline_relative_residual_safety_coverage_and_score_sum_floor',
         'exact_branch_and_bound_full_candidate_universe', 'canonical_cash_capped_plan_on_every_included_state',
         'maximize_primary_continuous_dl_score_sum', 'global_optimum_certificate', 'ticker_deterministic'),
    ),
    RankingPolicySpec(
        policy_id=BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_R0_CONSTRAINED_OPTIMAL,
        resource_aware=True,
        continuous_score_source=True,
        prefix_key='baseline',
        selector_key='continuous_score_no_r0_constrained_optimal',
        selector_label='continuous-score-no-r0-constrained-optimal',
        validation_capabilities=('exact_no_reserve_floor',),
        report_spec=(('same_param_exact_resource_baseline', 'fixed_baseline_order_count_only_no_r0', 'frozen_continuous_dl_score',
          'exact_branch_and_bound_full_candidate_universe', 'canonical_cash_capped_plan_on_every_included_state',
          'maximize_continuous_dl_score_sum', 'global_optimum_certificate', 'ticker_deterministic'),
         'all_days_with_feasible_alternative_baskets_under_fixed_baseline_order_count',
         'maximize_sum_continuous_dl_score_subject_to_same_k_and_true_cash_only_exact_global_search',
         'selected_count==baseline_selected_count; no baseline R0 floor; canonical cash-capped reservation remains '
         'binding',
         'exact_branch_and_bound_full_universe_subject_to_k_and_canonical_cash_no_r0',
         'objective_top_k_if_cash_feasible_else_same_param_baseline_is_incumbent_only; no_r0_repair; '
         'exact_search_space_is_not_restricted'),
    ),
    RankingPolicySpec(
        policy_id=BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_K_NO_R0,
        resource_aware=True,
        continuous_score_source=True,
        prefix_key='baseline',
        selector_key='continuous_score_no_k_no_r0',
        selector_label='continuous-score-no-k-no-r0',
        validation_capabilities=('direct_score', 'positive_filter'),
        report_spec=(('capital_deployment_bucket_desc', 'breakout_quality_score_desc', 'existing_buy_sort',
          'ticker_deterministic'),
         'all_days; score order may select any canonically cash-feasible count from zero to physical free slots',
         'existing_frozen_continuous_model_score_descending_then_original_deterministic_rank_tie',
         '0<=selected_count<=physical_free_slots; no baseline K/R0 floor; canonical cash-capped sizing/orderability '
         'remains authoritative',
         'deterministic_frozen_model_score_descending_then_canonical_cash_simulation; no combinatorial search',
         'none; same-param baseline is diagnostic only and never repairs/replaces score-order membership'),
        no_baseline_resource_floor=True,
    ),
    RankingPolicySpec(
        policy_id=BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_K_NO_R0_RAW_SAFETY_GATE,
        resource_aware=True,
        continuous_score_source=True,
        prefix_key='baseline',
        selector_key='continuous_score_no_k_no_r0_raw_safety_gate',
        selector_label='continuous-score-no-k-no-r0-raw-safety-gate',
        score_transform_key='same_day_rank_safety_scores',
        dynamic_options_key='safety_percentile',
        validation_capabilities=('raw_safety_gate',),
        dependency_roles=('safety',),
        report_spec=(('capital_deployment_bucket_desc', 'breakout_quality_score_desc', 'existing_buy_sort',
          'ticker_deterministic'),
         'all_days; score order may select any canonically cash-feasible count from zero to physical free slots',
         'raw_safety_same_day_orderable_percentile_gate_then_existing_frozen_conditional_mfe_score_descending',
         '0<=selected_count<=physical_free_slots; no baseline K/R0 floor; canonical cash-capped sizing/orderability '
         'remains authoritative',
         'raw_safety_gate_then_deterministic_frozen_model_score_descending_then_canonical_cash_simulation; no '
         'combinatorial search',
         'none; same-param baseline is diagnostic only and never repairs/replaces score-order membership'),
        no_baseline_resource_floor=True,
    ),
    RankingPolicySpec(
        policy_id=BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_K_NO_R0_SAFETY_MFE_PRODUCT,
        resource_aware=True,
        continuous_score_source=True,
        prefix_key='baseline',
        selector_key='continuous_score_no_k_no_r0_safety_mfe_product',
        selector_label='continuous-score-no-k-no-r0-safety-mfe-product',
        score_transform_key='same_day_rank_safety_mfe_product_scores',
        validation_capabilities=('safety_mfe_product',),
        dependency_roles=('safety',),
        report_spec=(('capital_deployment_bucket_desc', 'breakout_quality_score_desc', 'existing_buy_sort',
          'ticker_deterministic'),
         'all_days; score order may select any canonically cash-feasible count from zero to physical free slots',
         'same_day_orderable_raw_safety_percentile_times_conditional_mfe_percentile_descending',
         '0<=selected_count<=physical_free_slots; no baseline K/R0 floor; canonical cash-capped sizing/orderability '
         'remains authoritative',
         'same_day_raw_safety_percentile_x_conditional_mfe_percentile_descending_then_canonical_cash_simulation; no '
         'threshold; no fitted weight; no combinatorial search',
         'none; same-param baseline is diagnostic only and never repairs/replaces score-order membership'),
        no_baseline_resource_floor=True,
    ),
    RankingPolicySpec(
        policy_id=BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_CAPITAL_NO_R0_CONSTRAINED_OPTIMAL,
        resource_aware=True,
        continuous_score_source=True,
        prefix_key='baseline',
        selector_key='continuous_score_capital_no_r0_constrained_optimal',
        selector_label='continuous-score-capital-no-r0-constrained-optimal',
        validation_capabilities=('exact_no_reserve_floor',),
        report_spec=(('same_param_exact_resource_baseline', 'fixed_baseline_order_count_only_no_r0', 'frozen_continuous_dl_score',
          'exact_branch_and_bound_full_candidate_universe', 'canonical_cash_capped_plan_on_every_included_state',
          'maximize_continuous_dl_score_times_reserved_capital_sum', 'global_optimum_certificate',
          'ticker_deterministic'),
         'all_days_with_feasible_alternative_baskets_under_fixed_baseline_order_count',
         'maximize_sum_continuous_dl_score_times_canonical_reserved_capital_subject_to_same_k_and_true_cash_only_exact_global_search',
         'selected_count==baseline_selected_count; no baseline R0 floor; canonical cash-capped reservation remains '
         'binding',
         'exact_branch_and_bound_full_universe_subject_to_k_and_canonical_cash_no_r0',
         'objective_top_k_if_cash_feasible_else_same_param_baseline_is_incumbent_only; no_r0_repair; '
         'exact_search_space_is_not_restricted'),
    ),
    RankingPolicySpec(
        policy_id=BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_CAPITAL_PARETO_NO_R0_CONSTRAINED_OPTIMAL,
        resource_aware=True,
        continuous_score_source=True,
        prefix_key='baseline',
        selector_key='continuous_score_capital_pareto_no_r0_constrained_optimal',
        selector_label='continuous-score-capital-pareto-no-r0-constrained-optimal',
        validation_capabilities=('exact_no_reserve_floor', 'pareto_product'),
        report_spec=(('same_param_exact_resource_baseline', 'fixed_baseline_order_count_only_no_r0', 'frozen_continuous_dl_score',
          'exact_branch_and_bound_full_candidate_universe', 'canonical_cash_capped_plan_on_every_included_state',
          'maximize_score_coverage_first', 'derive_exact_score_and_reserved_capital_pareto_extremes',
          'normalize_quality_and_capital_by_pareto_extremes', 'maximize_normalized_quality_times_capital_product',
          'global_optimum_certificate', 'ticker_deterministic'),
         'all_days_with_feasible_alternative_baskets_under_fixed_baseline_order_count',
         'maximize_normalized_basket_score_quality_times_canonical_reserved_capital_using_exact_pareto_extremes_subject_to_same_k_and_true_cash_only',
         'selected_count==baseline_selected_count; no baseline R0 floor; canonical cash-capped reservation remains '
         'binding',
         'three_exact_passes_score_endpoint_capital_endpoint_then_normalized_product_global_optimum',
         'same_param_baseline_is_feasible_incumbent_only; score_and_capital_exact_endpoints_seed_final_product_pass; '
         'no_r0_repair'),
    ),
)
_POLICY_BY_ID = MappingProxyType({spec.policy_id: spec for spec in RANKING_POLICY_SPECS})
if len(_POLICY_BY_ID) != len(RANKING_POLICY_SPECS):
    raise ValueError("Duplicate ranking policy descriptor")


def get_ranking_policy_spec(policy_id: str | None) -> RankingPolicySpec | None:
    return _POLICY_BY_ID.get(policy_id)


def ranking_policy_has_validation(policy_id: str | None, capability: str) -> bool:
    spec = get_ranking_policy_spec(policy_id)
    return spec is not None and spec.has_validation(capability)


def ranking_policy_requires(policy_id: str | None, role: str) -> bool:
    spec = get_ranking_policy_spec(policy_id)
    return spec is not None and role in spec.dependency_roles


RESOURCE_AWARE_BREAKOUT_QUALITY_RANKING_POLICY_ORDER = tuple(
    spec.policy_id for spec in RANKING_POLICY_SPECS if spec.resource_aware
)
RESOURCE_AWARE_BREAKOUT_QUALITY_RANKING_POLICIES = frozenset(RESOURCE_AWARE_BREAKOUT_QUALITY_RANKING_POLICY_ORDER)
CONTINUOUS_SCORE_SOURCE_BREAKOUT_QUALITY_RANKING_POLICIES = frozenset(
    spec.policy_id for spec in RANKING_POLICY_SPECS if spec.continuous_score_source
)
SUPPORTED_BREAKOUT_QUALITY_RANKING_POLICIES = (
    *(spec.policy_id for spec in RANKING_POLICY_SPECS if not spec.resource_aware),
    *sorted(RESOURCE_AWARE_BREAKOUT_QUALITY_RANKING_POLICIES),
)
BREAKOUT_QUALITY_CAPITAL_BUCKET_COUNT = 3
_DEFAULT_BREAKOUT_QUALITY_SCORE_RANKING_ORDER = (
    'capital_deployment_bucket_desc', 'breakout_quality_score_desc', 'existing_buy_sort', 'ticker_deterministic',
)
_BREAKOUT_QUALITY_RESOURCE_AWARE_REPORT_SPECS = {
    spec.policy_id: spec.report_spec for spec in RANKING_POLICY_SPECS if spec.report_spec is not None
}
_BREAKOUT_QUALITY_NO_K_NO_R0_REPORT_POLICIES = frozenset(
    spec.policy_id for spec in RANKING_POLICY_SPECS if spec.no_baseline_resource_floor
)

_BREAKOUT_QUALITY_NON_RESOURCE_SCORE_ORDER_OVERRIDES = {
    BREAKOUT_QUALITY_RANKING_POLICY_POSITIVE_ONLY_DEFAULT: _POLICY_BY_ID[BREAKOUT_QUALITY_RANKING_POLICY_POSITIVE_ONLY_DEFAULT].score_order,
    BREAKOUT_QUALITY_RANKING_POLICY_SCORE: _POLICY_BY_ID[BREAKOUT_QUALITY_RANKING_POLICY_SCORE].score_order,
    BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_ADJUSTED: _POLICY_BY_ID[BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_ADJUSTED].score_order,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_RESIDUAL_SAFETY_CONSTRAINED_OPTIMAL: _POLICY_BY_ID[BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_RESIDUAL_SAFETY_CONSTRAINED_OPTIMAL].score_order,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_SAFETY_CONSTRAINED_OPTIMAL: _POLICY_BY_ID[BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_SAFETY_CONSTRAINED_OPTIMAL].score_order,
}
