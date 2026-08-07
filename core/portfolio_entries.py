import math
from core.capital_policy import resolve_portfolio_entry_budget
from core.buy_sort import (
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_BINARY,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_BINARY_BASKET,
    BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS,
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

    Min ROOS always owns the exact cash-capped resource-bottleneck decision.
    Binary variants optimize PASS use only on cash-binding days.  The continuous
    variant instead uses frozen event-level quality scores only on those same
    DL-selection days.  No selector may turn a cash-binding baseline into a
    slot-binding order plan or introduce a numeric utilization threshold.
    """

    rows = list(orderable_candidates_today or [])
    free_slots = max(0, int(max_positions) - int(pre_market_occupied))
    policy = _resource_aware_quality_policy(rows)
    selector = (
        'best-improvement-basket'
        if policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_BINARY_BASKET
        else 'continuous-score-constrained'
        if policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS
        else 'greedy-first-improvement'
    )
    default_diag = _resource_aware_default_diag(rows, free_slots, selector=selector)
    if free_slots <= 0 or not rows or policy is None:
        return rows, default_diag

    baseline = _simulate_reserved_candidate_order(
        rows,
        available_cash=available_cash,
        sizing_equity=sizing_equity,
        free_slots=free_slots,
        params=params,
    )
    if policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS:
        return _reorder_resource_aware_continuous(
            rows,
            available_cash=available_cash,
            sizing_equity=sizing_equity,
            free_slots=free_slots,
            params=params,
            baseline=baseline,
            default_diag=default_diag,
        )
    if policy == BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_BINARY_BASKET:
        return _reorder_resource_aware_binary_basket(
            rows,
            available_cash=available_cash,
            sizing_equity=sizing_equity,
            free_slots=free_slots,
            params=params,
            baseline=baseline,
            default_diag=default_diag,
        )
    return _reorder_resource_aware_binary_greedy(
        rows,
        available_cash=available_cash,
        sizing_equity=sizing_equity,
        free_slots=free_slots,
        params=params,
        baseline=baseline,
        default_diag=default_diag,
    )

# Backward-compatible import alias; all logic lives in the generic quality selector.
reorder_candidates_for_resource_aware_binary = reorder_candidates_for_resource_aware_quality


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
