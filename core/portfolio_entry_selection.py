import time

from core.buy_sort import (
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_BINARY_BASKET,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_CAPITAL_PRESERVING,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT_STALE_GUARD,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXPECTED_PNL_FEASIBLE_ASCENT,
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
)


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
    maximizes calibrated Expected R times canonical planned initial risk.  Both remain subject
    to the same hard K/R0 resource constraints and introduce no utilization threshold.
    """

    started_ns = time.perf_counter_ns()
    rows = list(orderable_candidates_today or [])
    free_slots = max(0, int(max_positions) - int(pre_market_occupied))
    policy = _resource_aware_quality_policy(rows)
    selector = (
        'best-improvement-basket'
        if policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_BINARY_BASKET
        else 'continuous-score-max-dl-feasible-ascent-stale-score-guard'
        if policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT_STALE_GUARD
        else 'continuous-expected-pnl-feasible-ascent'
        if policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXPECTED_PNL_FEASIBLE_ASCENT
        else 'continuous-score-max-dl-feasible-ascent'
        if policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT
        else 'continuous-score-max-dl'
        if policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL
        else 'continuous-score-capital-preserving'
        if policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_CAPITAL_PRESERVING
        else 'continuous-score-constrained'
        if policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS
        else 'greedy-first-improvement'
    )
    default_diag = _resource_aware_default_diag(rows, free_slots, selector=selector)

    def finish(order, diag):
        out = dict(diag or {})
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
    if policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT_STALE_GUARD:
        max_age_days = _stale_score_guard_max_age_days(rows)
        order, diag = _reorder_resource_aware_continuous_max_dl_feasible_ascent(
            rows,
            available_cash=available_cash,
            sizing_equity=sizing_equity,
            free_slots=free_slots,
            params=params,
            baseline=baseline,
            default_diag=default_diag,
            stale_score_membership_guard_max_age_days=max_age_days,
        )
        return finish(order, diag)
    if policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_EXPECTED_PNL_FEASIBLE_ASCENT:
        order, diag = _reorder_resource_aware_continuous_max_dl_feasible_ascent(
            rows,
            available_cash=available_cash,
            sizing_equity=sizing_equity,
            free_slots=free_slots,
            params=params,
            baseline=baseline,
            default_diag=default_diag,
            objective_mode='expected_pnl',
        )
        return finish(order, diag)
    if policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT:
        order, diag = _reorder_resource_aware_continuous_max_dl_feasible_ascent(
            rows,
            available_cash=available_cash,
            sizing_equity=sizing_equity,
            free_slots=free_slots,
            params=params,
            baseline=baseline,
            default_diag=default_diag,
        )
        return finish(order, diag)
    if policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL:
        order, diag = _reorder_resource_aware_continuous_max_dl(
            rows,
            available_cash=available_cash,
            sizing_equity=sizing_equity,
            free_slots=free_slots,
            params=params,
            baseline=baseline,
            default_diag=default_diag,
        )
        return finish(order, diag)
    if policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_CAPITAL_PRESERVING:
        order, diag = _reorder_resource_aware_continuous_capital_preserving(
            rows,
            available_cash=available_cash,
            sizing_equity=sizing_equity,
            free_slots=free_slots,
            params=params,
            baseline=baseline,
            default_diag=default_diag,
        )
        return finish(order, diag)
    if policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS:
        order, diag = _reorder_resource_aware_continuous(
            rows,
            available_cash=available_cash,
            sizing_equity=sizing_equity,
            free_slots=free_slots,
            params=params,
            baseline=baseline,
            default_diag=default_diag,
        )
        return finish(order, diag)
    if policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_BINARY_BASKET:
        order, diag = _reorder_resource_aware_binary_basket(
            rows,
            available_cash=available_cash,
            sizing_equity=sizing_equity,
            free_slots=free_slots,
            params=params,
            baseline=baseline,
            default_diag=default_diag,
        )
        return finish(order, diag)
    order, diag = _reorder_resource_aware_binary_greedy(
        rows,
        available_cash=available_cash,
        sizing_equity=sizing_equity,
        free_slots=free_slots,
        params=params,
        baseline=baseline,
        default_diag=default_diag,
    )
    return finish(order, diag)


