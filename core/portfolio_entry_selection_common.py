import math

from core.capital_policy import resolve_portfolio_entry_budget
from core.buy_sort import (
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_BINARY,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_BINARY_BASKET,
    RESOURCE_AWARE_BREAKOUT_QUALITY_RANKING_POLICIES,
    resolve_breakout_quality_ranking_policy,
)
from core.exact_accounting import coerce_money_like_to_milli, milli_to_money
from core.portfolio_entry_plans import _build_cash_capped_entry_plan_for_candidate


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


