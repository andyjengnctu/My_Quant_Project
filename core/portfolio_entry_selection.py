import time

from core.buy_sort import (
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
from core.portfolio_entry_selection_common import (
    _candidate_binary_pass,
    _candidate_continuous_score,
    _selected_continuous_score_metrics,
    _continuous_selected_quality_key,
    _simulate_reserved_candidate_order,
    _resource_aware_quality_policy,
    _resource_aware_default_diag,
    _resource_aware_diag_from_result,
    _decorate_same_day_rank_safety_scores,
    _decorate_same_day_rank_safety_mfe_product_scores,
    _decorate_same_day_rank_residual_safety_scores,
    _reorder_resource_aware_binary_greedy,
    _resource_aware_trial_rank_key,
    _reorder_resource_aware_binary_basket,
    _reorder_resource_aware_continuous,
    _capital_preserving_continuous_quality_key,
    _preserves_min_roos_resource_geometry,
    _reorder_resource_aware_continuous_capital_preserving,
)
from core.portfolio_entry_selection_max_dl import (
    _max_dl_score_order,
    _max_dl_execution_order,
    _max_dl_basket_quality_key,
    _max_dl_resource_deficit,
    _max_dl_basket_is_feasible,
    _reorder_resource_aware_continuous_max_dl,
    _candidate_date_value,
    _stale_score_guard_max_age_days,
    _candidate_has_stale_scored_signal,
    _reorder_resource_aware_continuous_max_dl_feasible_ascent,
    _reorder_resource_aware_continuous_excess_alpha_constrained_optimal,
    _reorder_resource_aware_continuous_score_constrained_optimal,
    _reorder_resource_aware_continuous_score_k_flex_r0_constrained_optimal,
    _reorder_resource_aware_continuous_score_safety_constrained_optimal,
    _reorder_resource_aware_continuous_score_residual_safety_constrained_optimal,
    _reorder_resource_aware_continuous_score_no_r0_constrained_optimal,
    _reorder_resource_aware_continuous_score_no_k_no_r0,
    _reorder_resource_aware_continuous_score_no_k_no_r0_raw_safety_gate,
    _reorder_resource_aware_continuous_score_no_k_no_r0_safety_mfe_product,
    _reorder_resource_aware_continuous_score_capital_no_r0_constrained_optimal,
    _reorder_resource_aware_continuous_score_capital_pareto_no_r0_constrained_optimal,
)


_RESOURCE_AWARE_POLICY_SPECS = {
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_BINARY_BASKET: (
        'best-improvement-basket', _reorder_resource_aware_binary_basket, {},
    ),
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS: (
        'continuous-score-constrained', _reorder_resource_aware_continuous, {},
    ),
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_CAPITAL_PRESERVING: (
        'continuous-score-capital-preserving', _reorder_resource_aware_continuous_capital_preserving, {},
    ),
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL: (
        'continuous-score-max-dl', _reorder_resource_aware_continuous_max_dl, {},
    ),
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT: (
        'continuous-score-max-dl-feasible-ascent', _reorder_resource_aware_continuous_max_dl_feasible_ascent, {},
    ),
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT_STALE_GUARD: (
        'continuous-score-max-dl-feasible-ascent-stale-score-guard',
        _reorder_resource_aware_continuous_max_dl_feasible_ascent,
        {},
    ),
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXPECTED_PNL_FEASIBLE_ASCENT: (
        'continuous-expected-pnl-feasible-ascent',
        _reorder_resource_aware_continuous_max_dl_feasible_ascent,
        {'objective_mode': 'expected_pnl'},
    ),
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_FEASIBLE_ASCENT: (
        'continuous-excess-alpha-feasible-ascent',
        _reorder_resource_aware_continuous_max_dl_feasible_ascent,
        {'objective_mode': 'excess_alpha'},
    ),
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_NO_R0_FEASIBLE_ASCENT: (
        'continuous-excess-alpha-no-r0-feasible-ascent',
        _reorder_resource_aware_continuous_max_dl_feasible_ascent,
        {'objective_mode': 'excess_alpha', 'preserve_reserve_floor': False, 'minimum_repair_enabled': False},
    ),
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXCESS_ALPHA_CONSTRAINED_OPTIMAL: (
        'continuous-excess-alpha-constrained-optimal', _reorder_resource_aware_continuous_excess_alpha_constrained_optimal, {},
    ),
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_CONSTRAINED_OPTIMAL: (
        'continuous-score-constrained-optimal', _reorder_resource_aware_continuous_score_constrained_optimal, {},
    ),
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_K_FLEX_R0_CONSTRAINED_OPTIMAL: (
        'continuous-score-k-flex-r0-constrained-optimal',
        _reorder_resource_aware_continuous_score_k_flex_r0_constrained_optimal,
        {},
    ),
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_SAFETY_CONSTRAINED_OPTIMAL: (
        'continuous-score-safety-constrained-optimal',
        _reorder_resource_aware_continuous_score_safety_constrained_optimal,
        {},
    ),
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_RESIDUAL_SAFETY_CONSTRAINED_OPTIMAL: (
        'continuous-score-residual-safety-constrained-optimal',
        _reorder_resource_aware_continuous_score_residual_safety_constrained_optimal,
        {},
    ),
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_R0_CONSTRAINED_OPTIMAL: (
        'continuous-score-no-r0-constrained-optimal',
        _reorder_resource_aware_continuous_score_no_r0_constrained_optimal,
        {},
    ),
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_K_NO_R0: (
        'continuous-score-no-k-no-r0', _reorder_resource_aware_continuous_score_no_k_no_r0, {},
    ),
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_K_NO_R0_RAW_SAFETY_GATE: (
        'continuous-score-no-k-no-r0-raw-safety-gate',
        _reorder_resource_aware_continuous_score_no_k_no_r0_raw_safety_gate,
        {},
    ),
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_K_NO_R0_SAFETY_MFE_PRODUCT: (
        'continuous-score-no-k-no-r0-safety-mfe-product',
        _reorder_resource_aware_continuous_score_no_k_no_r0_safety_mfe_product,
        {},
    ),
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_CAPITAL_NO_R0_CONSTRAINED_OPTIMAL: (
        'continuous-score-capital-no-r0-constrained-optimal',
        _reorder_resource_aware_continuous_score_capital_no_r0_constrained_optimal,
        {},
    ),
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_CAPITAL_PARETO_NO_R0_CONSTRAINED_OPTIMAL: (
        'continuous-score-capital-pareto-no-r0-constrained-optimal',
        _reorder_resource_aware_continuous_score_capital_pareto_no_r0_constrained_optimal,
        {},
    ),
    # Preserve the historical diagnostic selector label for these two variants.
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_MATCHED_FEASIBLE_ASCENT: (
        'greedy-first-improvement',
        _reorder_resource_aware_continuous_max_dl_feasible_ascent,
        {'matched_score_proposal': True},
    ),
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_K_FLEX_R0_FEASIBLE_ASCENT: (
        'greedy-first-improvement',
        _reorder_resource_aware_continuous_max_dl_feasible_ascent,
        {'allow_count_growth': True, 'matched_score_proposal': True},
    ),
}

_RESOURCE_AWARE_SCORE_DECORATORS = {
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_K_NO_R0_RAW_SAFETY_GATE:
        _decorate_same_day_rank_safety_scores,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_K_NO_R0_SAFETY_MFE_PRODUCT:
        _decorate_same_day_rank_safety_mfe_product_scores,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_RESIDUAL_SAFETY_CONSTRAINED_OPTIMAL:
        _decorate_same_day_rank_residual_safety_scores,
}


def _resource_aware_dynamic_kwargs(policy, rows):
    if policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT_STALE_GUARD:
        return {'stale_score_membership_guard_max_age_days': _stale_score_guard_max_age_days(rows)}
    if policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_K_NO_R0_RAW_SAFETY_GATE:
        options = dict(rows[0].get('breakout_quality_ranking_options') or {}) if rows else {}
        return {'safety_percentile_cutoff': float(options.get('safety_percentile_cutoff', 0.5))}
    return {}


def select_resource_aware_action_candidates(orderable_candidates_today, resource_selection_diag):
    """Return the candidates that may create pre-market orders, preserving full diagnostics input."""

    rows = list(orderable_candidates_today or [])
    raw_limit = (resource_selection_diag or {}).get('pre_market_order_limit')
    if raw_limit is None:
        return rows
    limit = int(raw_limit)
    if limit < 0:
        raise ValueError('pre_market_order_limit不得小於0')
    return rows[:limit]



def build_reserved_candidate_order_plan(
    orderable_candidates_today,
    *,
    available_cash,
    sizing_equity,
    free_slots,
    params,
):
    """Pure pre-market reservation plan using the canonical portfolio allocator semantics."""
    return _simulate_reserved_candidate_order(
        list(orderable_candidates_today or []),
        available_cash=available_cash,
        sizing_equity=sizing_equity,
        free_slots=int(free_slots),
        params=params,
    )

def reorder_candidates_for_resource_aware_quality(
    orderable_candidates_today,
    *,
    available_cash,
    sizing_equity,
    pre_market_occupied,
    max_positions,
    params,
):
    """Apply the configured resource-aware quality selector before reservation.

    The same-parameter DL-off ordering owns the exact cash-capped resource baseline.
    Binary variants optimize PASS use only on cash-binding days.  The original
    continuous variant uses frozen event-level quality scores on those same days.
    The capital-preserving continuous variant may also act on slot-binding days,
    but it must preserve the same-parameter DL-off selected-count and exact reserved capital.
    The max-DL variants fix the same-parameter DL-off planned-order count and reserved-capital
    floor.  The score variant maximizes frozen DL score; the frozen Expected-PnL variant
    maximizes calibrated Expected R times canonical planned initial risk; the frozen Excess-Alpha
    variant maximizes PIT Expected Excess-R times canonical planned initial risk.  The no-R0
    Excess-Alpha ablation keeps the same baseline K and true cash/sizing feasibility but removes
    the baseline reserved-capital floor and its minimum-repair path. Exact constrained modes reuse
    one branch-and-bound owner and differ only by frozen score vs Excess-Alpha objective.
    """

    started_ns = time.perf_counter_ns()
    rows = list(orderable_candidates_today or [])
    free_slots = max(0, int(max_positions) - int(pre_market_occupied))
    policy = _resource_aware_quality_policy(rows)
    selector, handler, static_kwargs = _RESOURCE_AWARE_POLICY_SPECS.get(
        policy,
        ('greedy-first-improvement', _reorder_resource_aware_binary_greedy, {}),
    )
    residual_fit_diag = {}
    decorator = _RESOURCE_AWARE_SCORE_DECORATORS.get(policy)
    if decorator is not None:
        rows, residual_fit_diag = decorator(rows)
    default_diag = _resource_aware_default_diag(rows, free_slots, selector=selector)

    def finish(order, diag):
        out = dict(diag or {})
        out.update(residual_fit_diag)
        out['selector_elapsed_ns'] = int(max(0, time.perf_counter_ns() - started_ns))
        return order, out

    if policy is None:
        # DL-off / non-resource-aware baseline does not execute a selector.
        # Keep selector timing at the canonical zero instead of measuring wrapper jitter.
        return rows, default_diag
    if free_slots <= 0 or not rows:
        return finish(rows, default_diag)

    baseline = _simulate_reserved_candidate_order(
        rows,
        available_cash=available_cash,
        sizing_equity=sizing_equity,
        free_slots=free_slots,
        params=params,
    )
    extra_kwargs = dict(static_kwargs)
    extra_kwargs.update(_resource_aware_dynamic_kwargs(policy, rows))
    order, diag = handler(
        rows,
        available_cash=available_cash,
        sizing_equity=sizing_equity,
        free_slots=free_slots,
        params=params,
        baseline=baseline,
        default_diag=default_diag,
        **extra_kwargs,
    )
    return finish(order, diag)

