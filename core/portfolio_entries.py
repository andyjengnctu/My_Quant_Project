import math
import time
from datetime import date, datetime
from core.capital_policy import resolve_portfolio_entry_budget
from core.buy_sort import (
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_BINARY,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_BINARY_BASKET,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_CAPITAL_PRESERVING,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_MAX_DL_FEASIBLE_ASCENT_STALE_GUARD,
    RESOURCE_AWARE_BREAKOUT_QUALITY_RANKING_POLICIES,
    resolve_breakout_quality_ranking_policy,
)
from core.exact_accounting import (
    coerce_money_like_to_milli,
    milli_to_money,
    restore_money_like_from_milli,
)
from core.trade_plans import (
    build_cash_capped_entry_plan,
    clone_shadow_position,
    entry_notional_meets_minimum,
    execute_pre_market_entry_plan,
    resolve_signal_tracking_params,
    should_clear_extended_signal,
)
from core.portfolio_fast_data import get_fast_close, get_fast_pos, get_fast_value


def _format_candidate_date(value):
    if value is None or value != value:
        return ""
    if hasattr(value, "strftime"):
        return value.strftime('%Y-%m-%d')
    return str(value)


def _candidate_kind_label(candidate_type):
    normalized = str(candidate_type or '')
    if normalized == 'extended':
        return '延續候選'
    if normalized == 'reentry':
        return 'Re-entry'
    return '新訊號'


def build_candidate_plan_seed(candidate_row, sizing_equity=None):
    sizing_capital = candidate_row.get('sizing_capital')
    if (sizing_capital is None or sizing_capital != sizing_capital) and sizing_equity is not None:
        sizing_capital = sizing_equity
    plan = {
        'limit_price': candidate_row['limit_px'],
        'init_sl': candidate_row['init_sl'],
        'init_trail': candidate_row['init_trail'],
        'target_price': candidate_row.get('target_price'),
        'entry_atr': candidate_row.get('entry_atr'),
        'ticker': candidate_row.get('ticker'),
        'security_profile': candidate_row.get('security_profile'),
        'trade_date': candidate_row.get('trade_date'),
        'sizing_capital': sizing_capital,
        'orig_limit': candidate_row.get('orig_limit'),
        'orig_atr': candidate_row.get('orig_atr'),
        'max_qty': candidate_row.get('max_qty'),
    }
    if candidate_row.get('entry_source') is not None:
        plan['entry_source'] = candidate_row.get('entry_source')

    shadow_position_state = candidate_row.get('shadow_position_state')
    if shadow_position_state is None:
        signal_state = candidate_row.get('signal_state') or {}
        shadow_position_state = signal_state.get('shadow_position')
    if shadow_position_state is not None:
        plan['shadow_position_state'] = clone_shadow_position(shadow_position_state)
    return plan


def _build_candidate_full_entry_plan_if_affordable(candidate_row, available_cash_milli, params, sizing_equity=None):
    qty = int(candidate_row.get('qty', 0) or 0)
    reserved_cost_milli = int(candidate_row.get('proj_cost_milli', 0) or 0)
    if qty <= 0 or reserved_cost_milli <= 0 or reserved_cost_milli > int(available_cash_milli):
        return None
    if not entry_notional_meets_minimum(candidate_row.get('limit_px'), qty, params):
        return None

    entry_plan = build_candidate_plan_seed(candidate_row, sizing_equity=sizing_equity)
    entry_plan['qty'] = qty
    entry_plan['is_orderable'] = True
    entry_plan['reserved_cost_milli'] = reserved_cost_milli
    entry_plan['reserved_cost'] = milli_to_money(reserved_cost_milli)
    return entry_plan


def _build_cash_capped_entry_plan_for_candidate(candidate_row, effective_entry_budget, effective_entry_budget_milli, params, sizing_equity):
    full_entry_plan = _build_candidate_full_entry_plan_if_affordable(
        candidate_row,
        effective_entry_budget_milli,
        params,
        sizing_equity=sizing_equity,
    )
    if full_entry_plan is not None:
        return full_entry_plan
    return build_cash_capped_entry_plan(
        build_candidate_plan_seed(candidate_row, sizing_equity=sizing_equity),
        effective_entry_budget,
        params,
    )



def _candidate_binary_pass(candidate_row):
    if candidate_row.get('breakout_quality_ranking_policy') not in {
        BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_BINARY,
        BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_BINARY_BASKET,
    }:
        return False
    rank_payload = candidate_row.get('breakout_quality_rank')
    if not isinstance(rank_payload, dict) or not bool(rank_payload.get('available', False)):
        return False
    score = candidate_row.get('breakout_quality_score')
    try:
        numeric_score = float(score)
    except (TypeError, ValueError):
        return False
    if numeric_score != numeric_score:
        return False
    candidate_params = candidate_row.get('params_obj')
    if candidate_params is None:
        raise ValueError('resource-aware Binary候選缺少params_obj，無法解析正式threshold')
    threshold = float(getattr(candidate_params, 'breakout_quality_score_threshold'))
    return numeric_score >= threshold


def _candidate_continuous_score(candidate_row):
    rank_payload = candidate_row.get('breakout_quality_rank')
    if not isinstance(rank_payload, dict) or not bool(rank_payload.get('available', False)):
        return None
    score = candidate_row.get('breakout_quality_score')
    try:
        numeric_score = float(score)
    except (TypeError, ValueError):
        return None
    if numeric_score != numeric_score or not math.isfinite(numeric_score):
        return None
    return numeric_score


def _selected_continuous_score_metrics(result):
    scores = [
        score
        for row in result.get('selected_rows', [])
        if (score := _candidate_continuous_score(row)) is not None
    ]
    return {
        'scored_count': int(len(scores)),
        'score_sum': float(sum(scores)),
        'score_mean': (None if not scores else float(sum(scores) / len(scores))),
    }


def _continuous_selected_quality_key(result, *, free_slots):
    scores = sorted(
        [
            -1.0 if (score := _candidate_continuous_score(row)) is None else float(score)
            for row in result.get('selected_rows', [])
        ],
        reverse=True,
    )
    padded = scores + [-1.0] * max(0, int(free_slots) - len(scores))
    return tuple(padded[: int(free_slots)])


def _simulate_reserved_candidate_order(
    candidate_rows,
    *,
    available_cash,
    sizing_equity,
    free_slots,
    params,
):
    remaining_cash_milli = coerce_money_like_to_milli(available_cash)
    initial_cash_milli = int(remaining_cash_milli)
    selected_rows = []
    selected_plans = []
    for cand in list(candidate_rows or []):
        if len(selected_rows) >= int(free_slots):
            break
        if cand.get('is_orderable') is False:
            continue
        candidate_params = cand.get('params_obj') or params
        effective_entry_budget = resolve_portfolio_entry_budget(
            milli_to_money(remaining_cash_milli),
            candidate_params.initial_capital,
            candidate_params,
        )
        effective_entry_budget_milli = coerce_money_like_to_milli(effective_entry_budget)
        plan = _build_cash_capped_entry_plan_for_candidate(
            cand,
            effective_entry_budget,
            effective_entry_budget_milli,
            candidate_params,
            sizing_equity,
        )
        if plan is None:
            continue
        reserved_cost_milli = int(plan.get('reserved_cost_milli', 0) or 0)
        if reserved_cost_milli <= 0 or reserved_cost_milli > int(remaining_cash_milli):
            continue
        selected_rows.append(cand)
        selected_plans.append(plan)
        remaining_cash_milli -= reserved_cost_milli
    selected_pass_flags = [_candidate_binary_pass(row) for row in selected_rows]
    pass_reserved_cost_milli = sum(
        int(plan.get('reserved_cost_milli', 0) or 0)
        for row, plan, is_pass in zip(selected_rows, selected_plans, selected_pass_flags)
        if is_pass
    )
    selected_ids = {id(row) for row in selected_rows}
    has_unselected_candidates = any(id(row) not in selected_ids for row in list(candidate_rows or []))
    cash_is_binding = bool(
        len(selected_rows) < int(free_slots)
        and has_unselected_candidates
        and int(initial_cash_milli - remaining_cash_milli) > 0
    )
    return {
        'selected_rows': selected_rows,
        'selected_plans': selected_plans,
        'selected_count': int(len(selected_rows)),
        'pass_count': int(sum(selected_pass_flags)),
        'pass_reserved_cost_milli': int(pass_reserved_cost_milli),
        'reserved_cost_milli': int(initial_cash_milli - remaining_cash_milli),
        'remaining_cash_milli': int(remaining_cash_milli),
        'has_unselected_candidates': bool(has_unselected_candidates),
        'cash_is_binding': bool(cash_is_binding),
    }


def _resource_aware_quality_policy(candidate_rows):
    rows = list(candidate_rows or [])
    if not rows:
        return None
    flags = {bool(row.get('use_breakout_quality_ranking', False)) for row in rows}
    if flags != {True}:
        return None
    policy = resolve_breakout_quality_ranking_policy(rows)
    if policy not in RESOURCE_AWARE_BREAKOUT_QUALITY_RANKING_POLICIES:
        return None
    return policy


def _resource_aware_default_diag(rows, free_slots, *, selector):
    return {
        'enabled': False,
        'mode': 'inactive',
        'selector': str(selector),
        'free_slots': int(free_slots),
        'candidate_count': int(len(rows)),
        'baseline_selected_count': 0,
        'baseline_pass_count': 0,
        'baseline_reserved_cost_milli': 0,
        'baseline_pass_reserved_cost_milli': 0,
        'selected_count': 0,
        'selected_pass_count': 0,
        'reserved_cost_milli': 0,
        'pass_reserved_cost_milli': 0,
        'promoted_pass_count': 0,
        'changed': False,
        'basket_search_states': 0,
        'basket_search_pruned': 0,
        'basket_feasible_count': 0,
        'baseline_scored_selected_count': 0,
        'selected_scored_count': 0,
        'baseline_selected_score_sum': 0.0,
        'selected_score_sum': 0.0,
        'baseline_selected_score_mean': None,
        'selected_score_mean': None,
        'promoted_score_orders': 0,
        'direct_score_order_feasible': False,
        'resource_preservation_required': False,
        'selected_count_preserved': True,
        'reserved_capital_preserved': True,
        'pre_market_order_limit': None,
        'max_dl_eligible': False,
        'max_dl_repair_steps': 0,
        'max_dl_repair_evaluations': 0,
        'max_dl_fallback_to_baseline': False,
        'max_dl_seed_fallback': False,
        'max_dl_feasible_ascent_steps': 0,
        'max_dl_feasible_ascent_evaluations': 0,
        'max_dl_feasible_ascent_local_optimum': False,
        'stale_score_membership_guard_enabled': False,
        'stale_score_membership_guard_max_age_days': None,
        'stale_score_candidate_count': 0,
        'stale_score_guard_triggered': False,
        'stale_score_guard_seed_blocked': False,
        'stale_score_guard_blocked_swaps': 0,
        'selector_elapsed_ns': 0,
    }


def _resource_aware_diag_from_result(default_diag, baseline, selected, *, changed, promoted_pass_count, selector):
    baseline_score = _selected_continuous_score_metrics(baseline)
    selected_score = _selected_continuous_score_metrics(selected)
    return {
        **default_diag,
        'enabled': True,
        'selector': str(selector),
        'baseline_selected_count': int(baseline['selected_count']),
        'baseline_pass_count': int(baseline['pass_count']),
        'baseline_reserved_cost_milli': int(baseline['reserved_cost_milli']),
        'baseline_pass_reserved_cost_milli': int(baseline['pass_reserved_cost_milli']),
        'selected_count': int(selected['selected_count']),
        'selected_pass_count': int(selected['pass_count']),
        'reserved_cost_milli': int(selected['reserved_cost_milli']),
        'pass_reserved_cost_milli': int(selected['pass_reserved_cost_milli']),
        'promoted_pass_count': int(max(0, promoted_pass_count)),
        'changed': bool(changed),
        'baseline_scored_selected_count': int(baseline_score['scored_count']),
        'selected_scored_count': int(selected_score['scored_count']),
        'baseline_selected_score_sum': float(baseline_score['score_sum']),
        'selected_score_sum': float(selected_score['score_sum']),
        'baseline_selected_score_mean': baseline_score['score_mean'],
        'selected_score_mean': selected_score['score_mean'],
        'selected_count_preserved': bool(
            int(selected['selected_count']) >= int(baseline['selected_count'])
        ),
        'reserved_capital_preserved': bool(
            int(selected['reserved_cost_milli']) >= int(baseline['reserved_cost_milli'])
        ),
    }


def _reorder_resource_aware_binary_greedy(
    rows,
    *,
    available_cash,
    sizing_equity,
    free_slots,
    params,
    baseline,
    default_diag,
):
    diag = _resource_aware_diag_from_result(
        default_diag,
        baseline,
        baseline,
        changed=False,
        promoted_pass_count=0,
        selector='greedy-first-improvement',
    )
    if not bool(baseline['cash_is_binding']):
        diag['mode'] = 'capital-utilization'
        return list(rows), diag

    diag['mode'] = 'dl-selection'
    base_rank = {id(row): idx for idx, row in enumerate(rows)}
    promoted_ids = set()
    current_order = list(rows)
    current = baseline

    while True:
        current_selected_ids = {id(row) for row in current['selected_rows']}
        improving_trial = None
        for candidate in rows:
            candidate_id = id(candidate)
            if candidate_id in current_selected_ids or candidate_id in promoted_ids:
                continue
            if not _candidate_binary_pass(candidate):
                continue
            trial_promoted = set(promoted_ids)
            trial_promoted.add(candidate_id)
            trial_order = sorted(
                rows,
                key=lambda row: (
                    0 if id(row) in trial_promoted else 1,
                    base_rank[id(row)],
                ),
            )
            trial = _simulate_reserved_candidate_order(
                trial_order,
                available_cash=available_cash,
                sizing_equity=sizing_equity,
                free_slots=free_slots,
                params=params,
            )
            if not bool(trial['cash_is_binding']):
                continue
            if int(trial['pass_reserved_cost_milli']) <= int(current['pass_reserved_cost_milli']):
                continue
            improving_trial = (trial_promoted, trial_order, trial)
            break

        if improving_trial is None:
            break
        promoted_ids, current_order, current = improving_trial

    if not bool(current['cash_is_binding']):
        raise RuntimeError('resource-aware Binary違反盤前cash-bottleneck資源契約')

    diag.update(_resource_aware_diag_from_result(
        default_diag,
        baseline,
        current,
        changed=bool([id(row) for row in current_order] != [id(row) for row in rows]),
        promoted_pass_count=int(current['pass_count'] - baseline['pass_count']),
        selector='greedy-first-improvement',
    ))
    diag['mode'] = 'dl-selection'
    return current_order, diag


def _resource_aware_trial_rank_key(trial_order, trial_result, base_rank):
    promoted_pass_ranks = tuple(
        base_rank[id(row)]
        for row in trial_result['selected_rows']
        if _candidate_binary_pass(row)
    )
    return (
        -int(trial_result['pass_reserved_cost_milli']),
        -int(trial_result['pass_count']),
        promoted_pass_ranks,
        tuple(base_rank[id(row)] for row in trial_order),
    )


def _reorder_resource_aware_binary_basket(
    rows,
    *,
    available_cash,
    sizing_equity,
    free_slots,
    params,
    baseline,
    default_diag,
):
    """Use best-improvement PASS promotion under the C11 resource contract.

    C11 accepts the first improving PASS promotion in Min ROOS order.  C12
    evaluates every not-yet-promoted PASS candidate at each step, chooses the
    best exact cash-capped improvement, then repeats from that new basket.  It
    removes first-candidate path dependence without an exponential exhaustive
    subset search or any additional numeric threshold.
    """

    diag = _resource_aware_diag_from_result(
        default_diag,
        baseline,
        baseline,
        changed=False,
        promoted_pass_count=0,
        selector='best-improvement-basket',
    )
    if not bool(baseline['cash_is_binding']):
        diag['mode'] = 'capital-utilization'
        return list(rows), diag

    diag['mode'] = 'dl-selection'
    base_rank = {id(row): idx for idx, row in enumerate(rows)}
    promoted_ids = set()
    current_order = list(rows)
    current = baseline
    evaluated_trials = 0
    feasible_trials = 0

    while True:
        best_trial = None
        best_key = None
        for candidate in rows:
            candidate_id = id(candidate)
            if candidate_id in promoted_ids or not _candidate_binary_pass(candidate):
                continue
            trial_promoted = set(promoted_ids)
            trial_promoted.add(candidate_id)
            trial_order = sorted(
                rows,
                key=lambda row: (
                    0 if id(row) in trial_promoted else 1,
                    base_rank[id(row)],
                ),
            )
            if [id(row) for row in trial_order] == [id(row) for row in current_order]:
                continue
            evaluated_trials += 1
            trial = _simulate_reserved_candidate_order(
                trial_order,
                available_cash=available_cash,
                sizing_equity=sizing_equity,
                free_slots=free_slots,
                params=params,
            )
            if not bool(trial['cash_is_binding']):
                continue
            feasible_trials += 1
            current_pass_reserved = int(current['pass_reserved_cost_milli'])
            trial_pass_reserved = int(trial['pass_reserved_cost_milli'])
            current_pass_count = int(current['pass_count'])
            trial_pass_count = int(trial['pass_count'])
            if trial_pass_reserved < current_pass_reserved:
                continue
            if (
                trial_pass_reserved == current_pass_reserved
                and trial_pass_count <= current_pass_count
            ):
                continue
            trial_key = _resource_aware_trial_rank_key(trial_order, trial, base_rank)
            if best_key is None or trial_key < best_key:
                best_key = trial_key
                best_trial = (trial_promoted, trial_order, trial)

        if best_trial is None:
            break
        promoted_ids, current_order, current = best_trial

    if not bool(current['cash_is_binding']):
        raise RuntimeError('resource-aware Binary basket違反盤前cash-bottleneck資源契約')

    diag.update(_resource_aware_diag_from_result(
        default_diag,
        baseline,
        current,
        changed=bool([id(row) for row in current_order] != [id(row) for row in rows]),
        promoted_pass_count=int(current['pass_count'] - baseline['pass_count']),
        selector='best-improvement-basket',
    ))
    diag.update({
        'mode': 'dl-selection',
        'basket_search_states': int(evaluated_trials),
        'basket_search_pruned': 0,
        'basket_feasible_count': int(feasible_trials),
    })
    return current_order, diag

def _reorder_resource_aware_continuous(
    rows,
    *,
    available_cash,
    sizing_equity,
    free_slots,
    params,
    baseline,
    default_diag,
):
    """Use frozen continuous quality only when cash is already the binding resource.

    The Min ROOS order owns the resource bottleneck decision.  If Min ROOS needs
    all available position slots, quality ranking is not allowed to intervene.
    When cash binds first, the selector first tries the pure continuous-score order.
    If that order would turn the day into a slot-bottleneck day, it falls back to
    score-descending promotions and accepts only promotions whose exact cash-capped
    replay remains cash-binding.  No score threshold or Min-ROOS/score weight exists.
    """
    diag = _resource_aware_diag_from_result(
        default_diag, baseline, baseline, changed=False, promoted_pass_count=0,
        selector='continuous-score-constrained',
    )
    if not bool(baseline['cash_is_binding']):
        diag['mode'] = 'capital-utilization'
        return list(rows), diag

    diag['mode'] = 'dl-selection'
    base_rank = {id(row): idx for idx, row in enumerate(rows)}
    score_rows = [row for row in rows if _candidate_continuous_score(row) is not None]
    score_rows.sort(key=lambda row: (-float(_candidate_continuous_score(row)), base_rank[id(row)]))
    scored_ids = {id(row) for row in score_rows}
    direct_order = score_rows + [row for row in rows if id(row) not in scored_ids]
    direct = _simulate_reserved_candidate_order(
        direct_order,
        available_cash=available_cash,
        sizing_equity=sizing_equity,
        free_slots=free_slots,
        params=params,
    )
    if bool(direct['cash_is_binding']):
        baseline_selected_ids = {id(row) for row in baseline['selected_rows']}
        promoted_count = len([row for row in direct['selected_rows'] if id(row) not in baseline_selected_ids])
        diag.update(_resource_aware_diag_from_result(
            default_diag, baseline, direct,
            changed=bool([id(row) for row in direct_order] != [id(row) for row in rows]),
            promoted_pass_count=0,
            selector='continuous-score-direct',
        ))
        diag.update({
            'mode': 'dl-selection',
            'promoted_score_orders': int(promoted_count),
            'direct_score_order_feasible': True,
            'basket_search_states': 1,
            'basket_feasible_count': 1,
        })
        return direct_order, diag

    promoted_ids = set()
    current_order = list(rows)
    current = baseline
    current_key = _continuous_selected_quality_key(current, free_slots=free_slots)
    evaluated = 1
    feasible = 0
    for candidate in score_rows:
        candidate_id = id(candidate)
        trial_promoted = set(promoted_ids)
        trial_promoted.add(candidate_id)
        trial_order = sorted(
            rows,
            key=lambda row: (
                0 if id(row) in trial_promoted else 1,
                -(
                    _candidate_continuous_score(row)
                    if _candidate_continuous_score(row) is not None
                    else -1.0
                ) if id(row) in trial_promoted else 0.0,
                base_rank[id(row)],
            ),
        )
        if [id(row) for row in trial_order] == [id(row) for row in current_order]:
            continue
        evaluated += 1
        trial = _simulate_reserved_candidate_order(
            trial_order,
            available_cash=available_cash,
            sizing_equity=sizing_equity,
            free_slots=free_slots,
            params=params,
        )
        if not bool(trial['cash_is_binding']):
            continue
        if candidate_id not in {id(row) for row in trial['selected_rows']}:
            continue
        feasible += 1
        trial_key = _continuous_selected_quality_key(trial, free_slots=free_slots)
        if trial_key <= current_key:
            continue
        promoted_ids = trial_promoted
        current_order = trial_order
        current = trial
        current_key = trial_key

    if not bool(current['cash_is_binding']):
        raise RuntimeError('resource-aware Continuous違反盤前cash-bottleneck資源契約')
    baseline_selected_ids = {id(row) for row in baseline['selected_rows']}
    promoted_count = len([row for row in current['selected_rows'] if id(row) not in baseline_selected_ids])
    diag.update(_resource_aware_diag_from_result(
        default_diag, baseline, current,
        changed=bool([id(row) for row in current_order] != [id(row) for row in rows]),
        promoted_pass_count=0,
        selector='continuous-score-constrained',
    ))
    diag.update({
        'mode': 'dl-selection',
        'promoted_score_orders': int(promoted_count),
        'direct_score_order_feasible': False,
        'basket_search_states': int(evaluated),
        'basket_feasible_count': int(feasible),
    })
    return current_order, diag



def _capital_preserving_continuous_quality_key(result, *, free_slots):
    """Rank a selected basket by frozen continuous quality without rewarding empty slots."""

    scores = sorted(
        [
            float(score)
            for row in result.get('selected_rows', [])
            if (score := _candidate_continuous_score(row)) is not None
        ],
        reverse=True,
    )
    padded = scores + [-math.inf] * max(0, int(free_slots) - len(scores))
    return tuple(padded[: int(free_slots)])


def _preserves_min_roos_resource_geometry(baseline, trial):
    """Require no loss of planned position count or exact pre-market reserved capital."""

    return bool(
        int(trial['selected_count']) >= int(baseline['selected_count'])
        and int(trial['reserved_cost_milli']) >= int(baseline['reserved_cost_milli'])
    )


def _reorder_resource_aware_continuous_capital_preserving(
    rows,
    *,
    available_cash,
    sizing_equity,
    free_slots,
    params,
    baseline,
    default_diag,
):
    """Improve frozen continuous quality without degrading Min ROOS resource geometry.

    Unlike C15, this selector is not limited to days where cash is the baseline
    bottleneck.  It may also intervene on slot-binding days, but every accepted
    basket must preserve both Min ROOS planned selected-count and exact reserved
    capital.  The direct score order is tried first.  If it violates either
    invariant, deterministic best-improvement score promotions are searched; no
    future fill information, score threshold, utilization tolerance, or blend
    weight is introduced.
    """

    diag = _resource_aware_diag_from_result(
        default_diag,
        baseline,
        baseline,
        changed=False,
        promoted_pass_count=0,
        selector='continuous-score-capital-preserving',
    )
    diag['resource_preservation_required'] = True

    score_rows = [row for row in rows if _candidate_continuous_score(row) is not None]
    if not score_rows or not bool(baseline['has_unselected_candidates']):
        diag['mode'] = 'capital-utilization'
        return list(rows), diag

    diag['mode'] = 'dl-selection'
    base_rank = {id(row): idx for idx, row in enumerate(rows)}
    baseline_key = _capital_preserving_continuous_quality_key(
        baseline, free_slots=free_slots
    )

    score_rows.sort(
        key=lambda row: (
            -float(_candidate_continuous_score(row)),
            base_rank[id(row)],
        )
    )
    scored_ids = {id(row) for row in score_rows}
    direct_order = score_rows + [row for row in rows if id(row) not in scored_ids]
    direct = _simulate_reserved_candidate_order(
        direct_order,
        available_cash=available_cash,
        sizing_equity=sizing_equity,
        free_slots=free_slots,
        params=params,
    )
    direct_feasible = _preserves_min_roos_resource_geometry(baseline, direct)
    direct_key = _capital_preserving_continuous_quality_key(
        direct, free_slots=free_slots
    )
    if direct_feasible and direct_key > baseline_key:
        baseline_selected_ids = {id(row) for row in baseline['selected_rows']}
        promoted_count = sum(
            1 for row in direct['selected_rows'] if id(row) not in baseline_selected_ids
        )
        diag.update(
            _resource_aware_diag_from_result(
                default_diag,
                baseline,
                direct,
                changed=bool(
                    [id(row) for row in direct_order] != [id(row) for row in rows]
                ),
                promoted_pass_count=0,
                selector='continuous-score-capital-preserving-direct',
            )
        )
        diag.update({
            'mode': 'dl-selection',
            'promoted_score_orders': int(promoted_count),
            'direct_score_order_feasible': True,
            'basket_search_states': 1,
            'basket_feasible_count': 1,
            'resource_preservation_required': True,
        })
        if not diag['selected_count_preserved'] or not diag['reserved_capital_preserved']:
            raise RuntimeError('capital-preserving Continuous直接排序違反Min ROOS資源契約')
        return direct_order, diag

    promoted_ids = set()
    current_order = list(rows)
    current = baseline
    current_key = baseline_key
    evaluated = 1
    feasible = 0

    while True:
        best_trial = None
        best_trial_key = None
        best_tie_key = None
        for candidate in score_rows:
            candidate_id = id(candidate)
            if candidate_id in promoted_ids:
                continue
            trial_promoted = set(promoted_ids)
            trial_promoted.add(candidate_id)
            trial_order = sorted(
                rows,
                key=lambda row: (
                    0 if id(row) in trial_promoted else 1,
                    -(
                        float(_candidate_continuous_score(row))
                        if id(row) in trial_promoted
                        and _candidate_continuous_score(row) is not None
                        else 0.0
                    ),
                    base_rank[id(row)],
                ),
            )
            if [id(row) for row in trial_order] == [id(row) for row in current_order]:
                continue
            evaluated += 1
            trial = _simulate_reserved_candidate_order(
                trial_order,
                available_cash=available_cash,
                sizing_equity=sizing_equity,
                free_slots=free_slots,
                params=params,
            )
            if not _preserves_min_roos_resource_geometry(baseline, trial):
                continue
            if candidate_id not in {id(row) for row in trial['selected_rows']}:
                continue
            feasible += 1
            trial_key = _capital_preserving_continuous_quality_key(
                trial, free_slots=free_slots
            )
            if trial_key <= current_key:
                continue
            selected_base_ranks = tuple(
                -base_rank[id(row)] for row in trial['selected_rows']
            )
            tie_key = (
                -len(trial_promoted),
                selected_base_ranks,
            )
            if (
                best_trial_key is None
                or trial_key > best_trial_key
                or (trial_key == best_trial_key and tie_key > best_tie_key)
            ):
                best_trial_key = trial_key
                best_tie_key = tie_key
                best_trial = (trial_promoted, trial_order, trial)

        if best_trial is None:
            break
        promoted_ids, current_order, current = best_trial
        current_key = best_trial_key

    if not _preserves_min_roos_resource_geometry(baseline, current):
        raise RuntimeError('capital-preserving Continuous違反Min ROOS資源契約')

    baseline_selected_ids = {id(row) for row in baseline['selected_rows']}
    promoted_count = sum(
        1 for row in current['selected_rows'] if id(row) not in baseline_selected_ids
    )
    diag.update(
        _resource_aware_diag_from_result(
            default_diag,
            baseline,
            current,
            changed=bool(
                [id(row) for row in current_order] != [id(row) for row in rows]
            ),
            promoted_pass_count=0,
            selector='continuous-score-capital-preserving',
        )
    )
    diag.update({
        'mode': 'dl-selection',
        'promoted_score_orders': int(promoted_count),
        'direct_score_order_feasible': bool(direct_feasible),
        'basket_search_states': int(evaluated),
        'basket_feasible_count': int(feasible),
        'resource_preservation_required': True,
    })
    if not diag['selected_count_preserved'] or not diag['reserved_capital_preserved']:
        raise RuntimeError('capital-preserving Continuous輸出違反Min ROOS資源契約')
    return current_order, diag


def _max_dl_score_order(rows, *, base_rank):
    """Rank candidate membership only by frozen DL score, then original deterministic rank."""

    return sorted(
        list(rows or []),
        key=lambda row: (
            0 if _candidate_continuous_score(row) is not None else 1,
            -(
                float(_candidate_continuous_score(row))
                if _candidate_continuous_score(row) is not None
                else 0.0
            ),
            base_rank[id(row)],
        ),
    )


def _max_dl_execution_order(rows, *, base_rank):
    """Keep the same-parameter DL-off baseline priority inside an already chosen DL basket."""

    return sorted(list(rows or []), key=lambda row: base_rank[id(row)])


def _max_dl_basket_quality_key(rows, *, base_rank):
    """Higher is better; score coverage comes before score sum, with deterministic ties."""

    basket = list(rows or [])
    scores = [
        float(score)
        for row in basket
        if (score := _candidate_continuous_score(row)) is not None
    ]
    sorted_scores = tuple(sorted(scores, reverse=True))
    stable_base_ranks = tuple(
        -base_rank[id(row)]
        for row in sorted(basket, key=lambda item: base_rank[id(item)])
    )
    return (
        int(len(scores)),
        float(sum(scores)),
        sorted_scores,
        stable_base_ranks,
    )


def _max_dl_resource_deficit(result, *, target_count, reserve_floor_milli):
    """Lexicographic hard-constraint deficit: planned order count first, then reserve floor."""

    return (
        max(0, int(target_count) - int(result['selected_count'])),
        max(0, int(reserve_floor_milli) - int(result['reserved_cost_milli'])),
    )


def _max_dl_basket_is_feasible(result, *, target_count, reserve_floor_milli):
    return bool(
        int(result['selected_count']) == int(target_count)
        and int(result['reserved_cost_milli']) >= int(reserve_floor_milli)
    )


def _reorder_resource_aware_continuous_max_dl(
    rows,
    *,
    available_cash,
    sizing_equity,
    free_slots,
    params,
    baseline,
    default_diag,
):
    """Let frozen DL own stock choice; the same-parameter DL-off baseline supplies exact pre-market resource floors.

    The baseline fixes K (planned order count) and R0 (exact reserved capital).  The
    unconstrained DL Top-K basket is tried first.  Basket membership is owned by DL,
    while the execution order inside a chosen basket stays on the same-parameter
    DL-off baseline rank so stock selection is not mixed with a second allocation-priority change.
    If Top-K violates K/R0, deterministic minimum-repair replaces original Top-K
    members one at a time.  Every trial uses the canonical cash-capped reservation
    simulator.  No score threshold, blend
    weight, utilization ratio, future fill data, or OOS-tuned numeric tolerance is
    introduced.
    """

    target_count = int(baseline['selected_count'])
    reserve_floor_milli = int(baseline['reserved_cost_milli'])
    diag = _resource_aware_diag_from_result(
        default_diag,
        baseline,
        baseline,
        changed=False,
        promoted_pass_count=0,
        selector='continuous-score-max-dl',
    )
    diag.update({
        'resource_preservation_required': True,
        'pre_market_order_limit': int(target_count),
        'max_dl_eligible': False,
        'max_dl_repair_steps': 0,
        'max_dl_repair_evaluations': 0,
        'max_dl_fallback_to_baseline': False,
    })

    if target_count <= 0 or len(rows) <= target_count:
        diag['mode'] = 'capital-utilization'
        return list(rows), diag

    score_available = any(_candidate_continuous_score(row) is not None for row in rows)
    if not score_available:
        diag['mode'] = 'capital-utilization'
        return list(rows), diag

    diag['mode'] = 'dl-selection'
    diag['max_dl_eligible'] = True
    base_rank = {id(row): idx for idx, row in enumerate(rows)}
    dl_order = _max_dl_score_order(rows, base_rank=base_rank)
    pure_basket = list(dl_order[:target_count])
    pure_ids = {id(row) for row in pure_basket}

    def evaluate_basket(basket):
        ordered_basket = _max_dl_execution_order(basket, base_rank=base_rank)
        result = _simulate_reserved_candidate_order(
            ordered_basket,
            available_cash=available_cash,
            sizing_equity=sizing_equity,
            free_slots=target_count,
            params=params,
        )
        return ordered_basket, result

    pure_ordered, pure_result = evaluate_basket(pure_basket)
    evaluated = 1
    feasible_count = int(
        _max_dl_basket_is_feasible(
            pure_result,
            target_count=target_count,
            reserve_floor_milli=reserve_floor_milli,
        )
    )
    direct_feasible = bool(feasible_count)
    baseline_selected_ids = {id(row) for row in baseline['selected_rows']}

    def final_order_for_basket(basket):
        basket_order = _max_dl_execution_order(basket, base_rank=base_rank)
        basket_ids = {id(row) for row in basket_order}
        return basket_order + [row for row in rows if id(row) not in basket_ids]

    def finalize(
        basket,
        result,
        *,
        selector,
        repair_steps,
        fallback,
        preserve_basket_order=False,
    ):
        if preserve_basket_order:
            basket_order = list(basket)
            basket_ids = {id(row) for row in basket_order}
            final_order = basket_order + [
                row for row in rows if id(row) not in basket_ids
            ]
        else:
            final_order = final_order_for_basket(basket)
        promoted_count = sum(
            1 for row in result['selected_rows'] if id(row) not in baseline_selected_ids
        )
        selected_ids = {id(row) for row in result['selected_rows']}
        out = _resource_aware_diag_from_result(
            default_diag,
            baseline,
            result,
            changed=bool(selected_ids != baseline_selected_ids),
            promoted_pass_count=0,
            selector=selector,
        )
        out.update({
            'mode': 'dl-selection',
            'promoted_score_orders': int(promoted_count),
            'direct_score_order_feasible': bool(direct_feasible),
            'basket_search_states': int(evaluated),
            'basket_feasible_count': int(feasible_count),
            'resource_preservation_required': True,
            'pre_market_order_limit': int(target_count),
            'max_dl_eligible': True,
            'max_dl_repair_steps': int(repair_steps),
            'max_dl_repair_evaluations': int(max(0, evaluated - 1)),
            'max_dl_fallback_to_baseline': bool(fallback),
        })
        if int(result['selected_count']) != target_count:
            raise RuntimeError('max-DL Continuous輸出未維持同參數DL-off baseline預留單數')
        if int(result['reserved_cost_milli']) < reserve_floor_milli:
            raise RuntimeError('max-DL Continuous輸出低於同參數DL-off baseline reserved-capital floor')
        return final_order, out

    if direct_feasible:
        return finalize(
            pure_ordered,
            pure_result,
            selector='continuous-score-max-dl-direct',
            repair_steps=0,
            fallback=False,
        )

    current_basket = list(pure_basket)
    current_result = pure_result
    current_deficit = _max_dl_resource_deficit(
        current_result,
        target_count=target_count,
        reserve_floor_milli=reserve_floor_milli,
    )
    repaired_out_ids = set()
    repaired_in_ids = set()

    for repair_step in range(1, target_count + 1):
        current_ids = {id(row) for row in current_basket}
        out_rows = [
            row
            for row in pure_basket
            if id(row) in current_ids and id(row) not in repaired_out_ids
        ]
        in_rows = [
            row
            for row in dl_order[target_count:]
            if id(row) not in current_ids and id(row) not in repaired_in_ids
        ]
        if not out_rows or not in_rows:
            break

        best_progress = None
        best_progress_quality = None
        best_progress_tie = None
        best_feasible = None
        best_feasible_quality = None
        best_feasible_tie = None

        for out_row in out_rows:
            for in_row in in_rows:
                trial_basket = [
                    row for row in current_basket if id(row) != id(out_row)
                ] + [in_row]
                trial_ordered, trial_result = evaluate_basket(trial_basket)
                evaluated += 1
                trial_deficit = _max_dl_resource_deficit(
                    trial_result,
                    target_count=target_count,
                    reserve_floor_milli=reserve_floor_milli,
                )
                if trial_deficit >= current_deficit:
                    continue

                quality_key = _max_dl_basket_quality_key(
                    trial_ordered, base_rank=base_rank
                )
                tie_key = (
                    -base_rank[id(out_row)],
                    -base_rank[id(in_row)],
                )
                if _max_dl_basket_is_feasible(
                    trial_result,
                    target_count=target_count,
                    reserve_floor_milli=reserve_floor_milli,
                ):
                    feasible_count += 1
                    if (
                        best_feasible_quality is None
                        or quality_key > best_feasible_quality
                        or (
                            quality_key == best_feasible_quality
                            and tie_key > best_feasible_tie
                        )
                    ):
                        best_feasible_quality = quality_key
                        best_feasible_tie = tie_key
                        best_feasible = (
                            out_row,
                            in_row,
                            trial_ordered,
                            trial_result,
                        )

                if (
                    best_progress_quality is None
                    or quality_key > best_progress_quality
                    or (
                        quality_key == best_progress_quality
                        and tie_key > best_progress_tie
                    )
                ):
                    best_progress_quality = quality_key
                    best_progress_tie = tie_key
                    best_progress = (
                        out_row,
                        in_row,
                        trial_ordered,
                        trial_result,
                        trial_deficit,
                    )

        if best_feasible is not None:
            out_row, in_row, trial_ordered, trial_result = best_feasible
            repaired_out_ids.add(id(out_row))
            repaired_in_ids.add(id(in_row))
            return finalize(
                trial_ordered,
                trial_result,
                selector='continuous-score-max-dl-minimum-repair',
                repair_steps=repair_step,
                fallback=False,
            )

        if best_progress is None:
            break

        out_row, in_row, current_basket, current_result, current_deficit = best_progress
        repaired_out_ids.add(id(out_row))
        repaired_in_ids.add(id(in_row))

    baseline_basket = list(baseline['selected_rows'])
    baseline_ordered = list(baseline_basket)
    baseline_result = _simulate_reserved_candidate_order(
        baseline_ordered,
        available_cash=available_cash,
        sizing_equity=sizing_equity,
        free_slots=target_count,
        params=params,
    )
    if not _max_dl_basket_is_feasible(
        baseline_result,
        target_count=target_count,
        reserve_floor_milli=reserve_floor_milli,
    ):
        raise RuntimeError('max-DL Continuous無法重建同參數DL-off baseline resource floor')
    return finalize(
        baseline_ordered,
        baseline_result,
        selector='continuous-score-max-dl-baseline-fallback',
        repair_steps=len(repaired_out_ids),
        fallback=True,
        preserve_basket_order=True,
    )


def _candidate_date_value(value):
    if value is None:
        return None
    try:
        is_missing = bool(value != value)
    except (TypeError, ValueError):
        is_missing = False
    if is_missing:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if hasattr(value, "date"):
        try:
            resolved = value.date()
        except (TypeError, ValueError, AttributeError):
            resolved = None
        if isinstance(resolved, date):
            return resolved
    text = str(value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _stale_score_guard_max_age_days(rows):
    values = set()
    for row in list(rows or []):
        options = row.get('breakout_quality_ranking_options') or {}
        raw = options.get('stale_score_membership_guard_max_age_days')
        if raw is None:
            continue
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise ValueError('stale-score membership guard max age必須是非負整數')
        values.add(int(raw))
    if not values:
        raise ValueError('stale-score membership guard缺少max age設定')
    if len(values) != 1:
        raise ValueError('同一候選集合的stale-score membership guard max age不一致')
    return next(iter(values))


def _candidate_has_stale_scored_signal(row, *, max_age_days):
    if _candidate_continuous_score(row) is None:
        return False
    trade_date = _candidate_date_value(row.get('trade_date') or row.get('candidate_date'))
    score_date = _candidate_date_value(
        row.get('breakout_quality_score_date') or row.get('signal_date')
    )
    if trade_date is None or score_date is None:
        # 有有效score卻沒有可稽核日期時採保守解讀：不得讓它改變membership。
        return True
    return int((trade_date - score_date).days) > int(max_age_days)


def _reorder_resource_aware_continuous_max_dl_feasible_ascent(
    rows,
    *,
    available_cash,
    sizing_equity,
    free_slots,
    params,
    baseline,
    default_diag,
    stale_score_membership_guard_max_age_days=None,
):
    """Strengthen C17 with feasible best-improvement swaps, without exact combinatorial search.

    C17 first supplies a guaranteed-feasible K-basket.  If its Top-K repair had to
    fall back to the same-parameter DL-off baseline, that basket is still a valid seed rather than a terminal
    failure.  From the seed, every single membership swap is evaluated with canonical
    exact reservation; the highest-DL-quality feasible improvement is accepted and the
    process repeats until no improving single swap remains.  Capital never contributes
    to the objective: it is only the K/R0 feasibility contract.
    """

    target_count = int(baseline['selected_count'])
    reserve_floor_milli = int(baseline['reserved_cost_milli'])
    base_rank = {id(row): idx for idx, row in enumerate(rows)}
    guard_enabled = stale_score_membership_guard_max_age_days is not None
    guard_max_age = (
        None if not guard_enabled else int(stale_score_membership_guard_max_age_days)
    )
    stale_ids = (
        set()
        if not guard_enabled
        else {
            id(row) for row in rows
            if _candidate_has_stale_scored_signal(row, max_age_days=guard_max_age)
        }
    )
    seed_order, seed_diag = _reorder_resource_aware_continuous_max_dl(
        rows,
        available_cash=available_cash,
        sizing_equity=sizing_equity,
        free_slots=free_slots,
        params=params,
        baseline=baseline,
        default_diag=default_diag,
    )
    if not bool(seed_diag.get('max_dl_eligible', False)):
        out = dict(seed_diag)
        out.update({
            'selector': (
                'continuous-score-max-dl-feasible-ascent-stale-score-guard'
                if guard_enabled else 'continuous-score-max-dl-feasible-ascent'
            ),
            'stale_score_membership_guard_enabled': bool(guard_enabled),
            'stale_score_membership_guard_max_age_days': guard_max_age,
            'stale_score_candidate_count': int(len(stale_ids)),
            'stale_score_guard_triggered': False,
            'stale_score_guard_seed_blocked': False,
            'stale_score_guard_blocked_swaps': 0,
            'max_dl_seed_fallback': bool(seed_diag.get('max_dl_fallback_to_baseline', False)),
            'max_dl_fallback_to_baseline': False,
            'max_dl_feasible_ascent_local_optimum': True,
        })
        return seed_order, out

    def evaluate_basket(basket):
        ordered = _max_dl_execution_order(basket, base_rank=base_rank)
        result = _simulate_reserved_candidate_order(
            ordered,
            available_cash=available_cash,
            sizing_equity=sizing_equity,
            free_slots=target_count,
            params=params,
        )
        return ordered, result

    baseline_ids = {id(row) for row in baseline['selected_rows']}
    seed_basket = list(seed_order[:target_count])
    seed_ids = {id(row) for row in seed_basket}
    seed_guard_blocked = bool(
        guard_enabled
        and any(row_id in stale_ids for row_id in (baseline_ids ^ seed_ids))
    )
    current_basket = (
        list(baseline['selected_rows']) if seed_guard_blocked else seed_basket
    )
    current_order, current_result = evaluate_basket(current_basket)
    if not _max_dl_basket_is_feasible(
        current_result,
        target_count=target_count,
        reserve_floor_milli=reserve_floor_milli,
    ):
        raise RuntimeError('max-DL feasible-ascent seed不符合同參數DL-off baseline資源契約')
    current_key = _max_dl_basket_quality_key(current_order, base_rank=base_rank)
    evaluations = 0
    blocked_swaps = 0
    ascent_steps = 0
    local_optimum = False

    while True:
        current_ids = {id(row) for row in current_basket}
        in_rows = [row for row in rows if id(row) not in current_ids]
        best = None
        best_key = current_key
        best_tie = None
        for out_row in list(current_basket):
            for in_row in in_rows:
                trial_basket = [
                    row for row in current_basket if id(row) != id(out_row)
                ] + [in_row]
                trial_order, trial_result = evaluate_basket(trial_basket)
                evaluations += 1
                if not _max_dl_basket_is_feasible(
                    trial_result,
                    target_count=target_count,
                    reserve_floor_milli=reserve_floor_milli,
                ):
                    continue
                quality_key = _max_dl_basket_quality_key(
                    trial_order, base_rank=base_rank
                )
                if quality_key <= current_key:
                    continue
                if guard_enabled and (id(out_row) in stale_ids or id(in_row) in stale_ids):
                    # 只計數C25本來會接受的hard-feasible score改善；
                    # 單純存在stale候選不應被誤報成guard真正阻擋。
                    blocked_swaps += 1
                    continue
                tie_key = (
                    -base_rank[id(out_row)],
                    -base_rank[id(in_row)],
                )
                if (
                    best is None
                    or quality_key > best_key
                    or (quality_key == best_key and tie_key > best_tie)
                ):
                    best = (trial_basket, trial_order, trial_result)
                    best_key = quality_key
                    best_tie = tie_key
        if best is None:
            local_optimum = True
            break
        current_basket, current_order, current_result = best
        current_key = best_key
        ascent_steps += 1

    baseline_selected_ids = {id(row) for row in baseline['selected_rows']}
    selected_ids = {id(row) for row in current_result['selected_rows']}
    final_ids = {id(row) for row in current_order}
    final_order = list(current_order) + [
        row for row in rows if id(row) not in final_ids
    ]
    promoted_count = sum(
        1 for row in current_result['selected_rows']
        if id(row) not in baseline_selected_ids
    )
    out = _resource_aware_diag_from_result(
        default_diag,
        baseline,
        current_result,
        changed=bool(selected_ids != baseline_selected_ids),
        promoted_pass_count=0,
        selector=(
            'continuous-score-max-dl-feasible-ascent-stale-score-guard'
            if guard_enabled else 'continuous-score-max-dl-feasible-ascent'
        ),
    )
    out.update({
        'mode': 'dl-selection',
        'promoted_score_orders': int(promoted_count),
        'direct_score_order_feasible': bool(seed_diag.get('direct_score_order_feasible', False)),
        'basket_search_states': int(seed_diag.get('basket_search_states', 0) or 0) + int(evaluations),
        'basket_search_pruned': 0,
        'basket_feasible_count': int(seed_diag.get('basket_feasible_count', 0) or 0),
        'resource_preservation_required': True,
        'pre_market_order_limit': int(target_count),
        'max_dl_eligible': True,
        'max_dl_repair_steps': int(seed_diag.get('max_dl_repair_steps', 0) or 0),
        'max_dl_repair_evaluations': int(seed_diag.get('max_dl_repair_evaluations', 0) or 0),
        'max_dl_seed_fallback': bool(seed_diag.get('max_dl_fallback_to_baseline', False)),
        'max_dl_fallback_to_baseline': False,
        'max_dl_feasible_ascent_steps': int(ascent_steps),
        'max_dl_feasible_ascent_evaluations': int(evaluations),
        'max_dl_feasible_ascent_local_optimum': bool(local_optimum),
        'stale_score_membership_guard_enabled': bool(guard_enabled),
        'stale_score_membership_guard_max_age_days': guard_max_age,
        'stale_score_candidate_count': int(len(stale_ids)),
        'stale_score_guard_triggered': bool(seed_guard_blocked or blocked_swaps > 0),
        'stale_score_guard_seed_blocked': bool(seed_guard_blocked),
        'stale_score_guard_blocked_swaps': int(blocked_swaps),
    })
    if int(current_result['selected_count']) != target_count:
        raise RuntimeError('max-DL feasible-ascent輸出未維持同參數DL-off baseline預留單數')
    if int(current_result['reserved_cost_milli']) < reserve_floor_milli:
        raise RuntimeError('max-DL feasible-ascent輸出低於同參數DL-off baseline reserved-capital floor')
    return final_order, out


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
    The max-DL variant fixes the same-parameter DL-off planned-order count and reserved-capital
    floor, then lets frozen DL score own stock choice subject only to those hard
    resource constraints.  No selector introduces a numeric utilization threshold.
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



def execute_reserved_entries_for_day(
    portfolio,
    active_extended_signals,
    orderable_candidates_today,
    sold_today,
    all_dfs_fast,
    today,
    params,
    cash,
    available_cash,
    sizing_equity,
    max_positions,
    trade_history,
    is_training,
    total_missed_buys,
    entry_stats=None,
):
    pre_market_occupied = len(portfolio) + len(sold_today)
    remaining_orderable_candidates = list(orderable_candidates_today)
    cash_template = cash
    cash_milli = coerce_money_like_to_milli(cash)
    available_cash_milli = coerce_money_like_to_milli(available_cash)

    while remaining_orderable_candidates and pre_market_occupied < max_positions:
        cand = remaining_orderable_candidates.pop(0)
        candidate_params = cand.get('params_obj') or params
        effective_entry_budget = resolve_portfolio_entry_budget(
            milli_to_money(available_cash_milli),
            candidate_params.initial_capital,
            candidate_params,
        )
        effective_entry_budget_milli = coerce_money_like_to_milli(effective_entry_budget)

        if cand.get('is_orderable') is False:
            continue
        candidate_kind_label = _candidate_kind_label(cand.get('type'))
        signal_date_text = _format_candidate_date(cand.get('signal_date'))
        candidate_date_text = _format_candidate_date(cand.get('candidate_date') or cand.get('trade_date') or today)

        chosen_entry_plan = _build_cash_capped_entry_plan_for_candidate(
            cand,
            effective_entry_budget,
            effective_entry_budget_milli,
            candidate_params,
            sizing_equity,
        )
        if chosen_entry_plan is None:
            continue

        candidate_context = cand.get('_ensemble_context') if isinstance(cand.get('_ensemble_context'), dict) else {}
        candidate_all_dfs_fast = candidate_context.get('all_dfs_fast') or all_dfs_fast
        fast_df = candidate_all_dfs_fast[cand['ticker']]
        t_pos = cand['today_pos']
        y_pos = cand['yesterday_pos']
        t_open = get_fast_value(fast_df, 'Open', pos=t_pos)
        t_high = get_fast_value(fast_df, 'High', pos=t_pos)
        t_low = get_fast_value(fast_df, 'Low', pos=t_pos)
        t_close = get_fast_close(fast_df, pos=t_pos)
        t_volume = get_fast_value(fast_df, 'Volume', pos=t_pos)
        y_close = get_fast_close(fast_df, pos=y_pos)

        reserved_cost_milli = chosen_entry_plan['reserved_cost_milli']
        available_cash_milli -= reserved_cost_milli
        pre_market_occupied += 1

        entry_result = execute_pre_market_entry_plan(
            entry_plan=chosen_entry_plan,
            t_open=t_open,
            t_high=t_high,
            t_low=t_low,
            t_close=t_close,
            t_volume=t_volume,
            y_close=y_close,
            params=candidate_params,
            entry_type=cand['type'],
            ticker=cand['ticker'],
            trade_date=today,
        )

        if entry_result['filled']:
            actual_total_cost_milli = entry_result['position']['net_buy_total_milli']
            cash_milli -= actual_total_cost_milli
            entry_result['position']['_entry_params_obj'] = candidate_params
            entry_result['position']['_entry_params_signature'] = str(cand.get('params_signature') or '')
            entry_result['position']['_ensemble_vote_count'] = cand.get('ensemble_vote_count')
            entry_result['position']['_ensemble_member_key'] = cand.get('ensemble_member_key')
            entry_result['position']['_ensemble_min_agree'] = cand.get('ensemble_min_agree')
            ensemble_member_keys = cand.get('ensemble_member_keys')
            if isinstance(ensemble_member_keys, (list, tuple, set)):
                entry_result['position']['_ensemble_member_keys'] = sorted({str(key) for key in ensemble_member_keys if str(key).strip()})
            ensemble_member_params_by_key = cand.get('ensemble_member_params_by_key')
            if isinstance(ensemble_member_params_by_key, dict):
                # # (AI註: 持倉需承接本次共識的全部 member 參數，讓 STOP 後各 member 能獨立進入 reclaim watchlist。)
                entry_result['position']['_ensemble_member_params_by_key'] = dict(ensemble_member_params_by_key)
            ensemble_member_quality_rank_by_key = cand.get('ensemble_member_quality_rank_by_key')
            if isinstance(ensemble_member_quality_rank_by_key, dict):
                entry_result['position']['_ensemble_member_quality_rank_by_key'] = {
                    str(key): dict(value)
                    for key, value in ensemble_member_quality_rank_by_key.items()
                    if str(key).strip() and isinstance(value, dict)
                }
            if candidate_context:
                entry_result['position']['_entry_context'] = candidate_context
            # # (AI註: 保存本次實際進場對應的訊號／候選日期，只供 round-trip 歸因與稽核，不介入交易決策。)
            entry_result['position']['signal_date'] = signal_date_text
            entry_result['position']['candidate_date'] = candidate_date_text
            entry_result['position']['candidate_type'] = candidate_kind_label
            # (AI註: 保存進場時的 quality ranking 診斷欄位；只供事後歸因，不介入持倉管理。)
            entry_result['position']['breakout_quality_score'] = cand.get('breakout_quality_score')
            entry_result['position']['breakout_quality_score_date'] = cand.get('breakout_quality_score_date')
            entry_result['position']['use_breakout_quality_ranking'] = bool(cand.get('use_breakout_quality_ranking', False))
            quality_rank_payload = cand.get('breakout_quality_rank')
            if isinstance(quality_rank_payload, dict):
                entry_result['position']['breakout_quality_rank'] = dict(quality_rank_payload)
            portfolio[cand['ticker']] = entry_result['position']
            if entry_stats is not None:
                entry_stats['filled_buy_count'] = int(entry_stats.get('filled_buy_count', 0) or 0) + 1

            if cand['ticker'] in active_extended_signals:
                del active_extended_signals[cand['ticker']]
            if not is_training:
                buy_position = entry_result['position']
                tp_half_price = buy_position.get('tp_half')
                if bool(buy_position.get('sold_half', False)):
                    tp_half_price = None
                trade_history.append(
                    {
                        'Date': today.strftime('%Y-%m-%d'),
                        'Ticker': cand['ticker'],
                        'Type': f"買進 ({candidate_kind_label}, EV:{cand['ev']:.2f}R)",
                        '買訊日': signal_date_text,
                        '候選日': candidate_date_text,
                        '候選類型': candidate_kind_label,
                        '買入限價': chosen_entry_plan['limit_price'],
                        '成交價': entry_result.get('entry_fill_price', entry_result['buy_price']),
                        '成本均價': entry_result.get('cost_basis_price', entry_result['entry_price']),
                        '停損價': buy_position.get('sl'),
                        '半倉停利價': tp_half_price,
                        'Shadow買進價': buy_position.get('shadow_entry_fill_price'),
                        '股數': buy_position['initial_qty'],
                        '預留總金額': milli_to_money(reserved_cost_milli),
                        '投入總金額': milli_to_money(actual_total_cost_milli),
                        '進場類型': cand['type'],
                        '單筆損益': 0.0,
                        '該筆總損益': 0.0,
                        'R_Multiple': 0.0,
                        'Risk': candidate_params.fixed_risk,
                        'Quality Score': cand.get('breakout_quality_score'),
                        'Quality Score Date': cand.get('breakout_quality_score_date'),
                        'Quality Ranking': bool(cand.get('use_breakout_quality_ranking', False)),
                        'Ensemble Vote Count': cand.get('ensemble_vote_count'),
                    }
                )
        elif entry_result['count_as_missed_buy']:
            total_missed_buys += 1
            if not is_training:
                miss_buy_type = '錯失買進(Re-entry)' if cand['type'] == 'reentry' else ('錯失買進(延續候選)' if cand['type'] == 'extended' else '錯失買進(新訊號)')
                trade_history.append(
                    {
                        'Date': today.strftime('%Y-%m-%d'),
                        'Ticker': cand['ticker'],
                        'Type': miss_buy_type,
                        '買訊日': signal_date_text,
                        '候選日': candidate_date_text,
                        '候選類型': candidate_kind_label,
                        '進場類型': cand['type'],
                        '單筆損益': 0.0,
                        '該筆總損益': 0.0,
                        'R_Multiple': 0.0,
                        'Risk': candidate_params.fixed_risk,
                        '買入限價': chosen_entry_plan['limit_price'],
                        '成交價': None,
                        '成本均價': None,
                        '股數': chosen_entry_plan['qty'],
                        '預留總金額': milli_to_money(reserved_cost_milli),
                        '投入總金額': 0.0,
                        'Quality Score': cand.get('breakout_quality_score'),
                        'Quality Score Date': cand.get('breakout_quality_score_date'),
                        'Quality Ranking': bool(cand.get('use_breakout_quality_ranking', False)),
                        'Ensemble Vote Count': cand.get('ensemble_vote_count'),
                        '備註': f"預掛限價 {chosen_entry_plan['limit_price']:.2f} 未成交",
                    }
                )

    return restore_money_like_from_milli(cash_milli, cash_template), total_missed_buys


def cleanup_extended_signals_for_day(active_extended_signals, portfolio, all_dfs_fast, today, params, sizing_capital):
    for ticker in sorted(list(active_extended_signals.keys())):
        signal_state = active_extended_signals.get(ticker)
        signal_params = resolve_signal_tracking_params(signal_state, params)

        if ticker in portfolio:
            del active_extended_signals[ticker]
            continue

        fast_df = all_dfs_fast.get(ticker)
        if fast_df is None:
            continue

        t_pos = get_fast_pos(fast_df, today)
        if t_pos < 0:
            continue

        if t_pos <= 0:
            continue
        y_pos = t_pos - 1
        t_open = get_fast_value(fast_df, 'Open', pos=t_pos)
        t_low = get_fast_value(fast_df, 'Low', pos=t_pos)
        t_high = get_fast_value(fast_df, 'High', pos=t_pos)
        t_close = get_fast_close(fast_df, pos=t_pos)
        t_volume = get_fast_value(fast_df, 'Volume', pos=t_pos)
        y_close = get_fast_close(fast_df, pos=y_pos)
        y_high = get_fast_value(fast_df, 'High', pos=y_pos)
        y_atr = get_fast_value(fast_df, 'ATR', pos=y_pos)
        y_ind_sell = bool(get_fast_value(fast_df, 'ind_sell_signal', pos=y_pos))
        if should_clear_extended_signal(
            signal_state,
            t_low,
            t_high,
            t_open=t_open,
            t_close=t_close,
            t_volume=t_volume,
            y_close=y_close,
            y_high=y_high,
            y_atr=y_atr,
            y_ind_sell=y_ind_sell,
            sizing_capital=sizing_capital,
            current_date=today,
            params=signal_params,
        ):
            del active_extended_signals[ticker]
