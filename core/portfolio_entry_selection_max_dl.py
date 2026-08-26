from datetime import date, datetime
import math

from core.exact_accounting import (
    calc_planned_initial_risk_from_prices_milli,
    coerce_money_like_to_milli,
)
from core.order_lot_policy import get_min_entry_notional_milli

from core.portfolio_entry_selection_common import (
    _candidate_continuous_score,
    _candidate_continuous_safety_score,
    _candidate_continuous_residual_safety_score,
    _selected_continuous_safety_score_metrics,
    _selected_continuous_residual_safety_score_metrics,
    _resource_aware_diag_from_result,
    _simulate_reserved_candidate_order,
)


class _NoFeasibleConstrainedBasket(RuntimeError):
    """Internal exact-search signal used only by variable-count wrappers."""


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




def _candidate_expected_r(candidate_row):
    rank_payload = candidate_row.get('breakout_quality_rank')
    if not isinstance(rank_payload, dict):
        return None
    if not bool(rank_payload.get('expected_r_available', False)):
        return None
    try:
        value = float(rank_payload.get('expected_r'))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _candidate_expected_excess_r(candidate_row):
    rank_payload = candidate_row.get('breakout_quality_rank')
    if not isinstance(rank_payload, dict):
        return None
    if not bool(rank_payload.get('expected_excess_r_available', False)):
        return None
    try:
        value = float(rank_payload.get('expected_excess_r'))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _planned_initial_risk_milli(*, row, plan):
    qty = int(plan.get('qty', 0) or 0)
    if qty <= 0:
        return 0
    params = row.get('params_obj')
    if params is None:
        raise ValueError('resource-aware economic objective候選缺少params_obj')
    limit_price = plan.get('limit_price')
    stop_price = plan.get('init_sl')
    if limit_price is None or stop_price is None:
        return None
    return calc_planned_initial_risk_from_prices_milli(
        float(limit_price),
        float(stop_price),
        qty,
        params,
        ticker=row.get('ticker'),
        security_profile=row.get('security_profile'),
        trade_date=row.get('trade_date'),
    )


def _candidate_standalone_expected_pnl_milli(row):
    expected_r = _candidate_expected_r(row)
    if expected_r is None:
        return None
    plan = {
        'qty': int(row.get('qty', 0) or 0),
        'limit_price': row.get('limit_px'),
        'init_sl': row.get('init_sl'),
    }
    try:
        risk_milli = _planned_initial_risk_milli(row=row, plan=plan)
    except (TypeError, ValueError, KeyError, AttributeError):
        return None
    if risk_milli is None:
        return None
    return float(expected_r * float(risk_milli))


def _expected_pnl_order(rows, *, base_rank):
    """Rank raw membership by frozen Expected-R times canonical standalone planned risk."""

    return sorted(
        list(rows or []),
        key=lambda row: (
            0 if _candidate_standalone_expected_pnl_milli(row) is not None else 1,
            -(
                float(_candidate_standalone_expected_pnl_milli(row))
                if _candidate_standalone_expected_pnl_milli(row) is not None
                else 0.0
            ),
            -(
                float(_candidate_expected_r(row))
                if _candidate_expected_r(row) is not None
                else 0.0
            ),
            base_rank[id(row)],
        ),
    )


def _expected_pnl_basket_quality_key(result, *, base_rank):
    """Higher is better: calibration coverage, total Expected $PnL, deterministic ties."""

    rows = list(result.get('selected_rows') or [])
    plans = list(result.get('selected_plans') or [])
    if len(rows) != len(plans):
        raise ValueError('Expected-PnL selector selected_rows/plans長度不一致')
    values = []
    expected_rs = []
    for row, plan in zip(rows, plans):
        expected_r = _candidate_expected_r(row)
        if expected_r is None:
            continue
        risk_milli = _planned_initial_risk_milli(row=row, plan=plan)
        if risk_milli is None:
            continue
        values.append(float(expected_r * float(risk_milli)))
        expected_rs.append(float(expected_r))
    stable_base_ranks = tuple(
        -base_rank[id(row)]
        for row in sorted(rows, key=lambda item: base_rank[id(item)])
    )
    return (
        int(len(values)),
        float(sum(values)),
        tuple(sorted(values, reverse=True)),
        float(sum(expected_rs)),
        stable_base_ranks,
    )


def _candidate_standalone_excess_alpha_milli(row):
    expected_excess_r = _candidate_expected_excess_r(row)
    if expected_excess_r is None:
        return None
    plan = {
        'qty': int(row.get('qty', 0) or 0),
        'limit_price': row.get('limit_px'),
        'init_sl': row.get('init_sl'),
    }
    try:
        risk_milli = _planned_initial_risk_milli(row=row, plan=plan)
    except (TypeError, ValueError, KeyError, AttributeError):
        return None
    if risk_milli is None:
        return None
    return float(expected_excess_r * float(risk_milli))


def _excess_alpha_order(rows, *, base_rank):
    """Rank raw membership by Expected Excess-R times canonical standalone planned risk."""

    return sorted(
        list(rows or []),
        key=lambda row: (
            0 if _candidate_standalone_excess_alpha_milli(row) is not None else 1,
            -(
                float(_candidate_standalone_excess_alpha_milli(row))
                if _candidate_standalone_excess_alpha_milli(row) is not None
                else 0.0
            ),
            -(
                float(_candidate_expected_excess_r(row))
                if _candidate_expected_excess_r(row) is not None
                else 0.0
            ),
            base_rank[id(row)],
        ),
    )


def _excess_alpha_basket_quality_key(result, *, base_rank):
    """Higher is better: calibration coverage, total expected alpha dollars, deterministic ties."""

    rows = list(result.get('selected_rows') or [])
    plans = list(result.get('selected_plans') or [])
    if len(rows) != len(plans):
        raise ValueError('Excess-Alpha selector selected_rows/plans長度不一致')
    values = []
    expected_excess_rs = []
    for row, plan in zip(rows, plans):
        expected_excess_r = _candidate_expected_excess_r(row)
        if expected_excess_r is None:
            continue
        risk_milli = _planned_initial_risk_milli(row=row, plan=plan)
        if risk_milli is None:
            continue
        values.append(float(expected_excess_r * float(risk_milli)))
        expected_excess_rs.append(float(expected_excess_r))
    stable_base_ranks = tuple(
        -base_rank[id(row)]
        for row in sorted(rows, key=lambda item: base_rank[id(item)])
    )
    return (
        int(len(values)),
        float(sum(values)),
        tuple(sorted(values, reverse=True)),
        float(sum(expected_excess_rs)),
        stable_base_ranks,
    )


def _objective_selector_name(objective_mode, *, suffix, no_r0=False):
    if objective_mode == 'expected_pnl':
        prefix = 'continuous-expected-pnl'
    elif objective_mode == 'excess_alpha':
        prefix = 'continuous-excess-alpha'
    elif objective_mode == 'score_capital':
        prefix = 'continuous-score-capital-product'
    else:
        prefix = 'continuous-score-max-dl'
    if no_r0:
        return f'{prefix}-no-r0-{suffix}'
    return f'{prefix}-{suffix}'

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


def _score_capital_basket_quality_key(result, *, base_rank):
    """Higher is better: score coverage, then Σ(score × canonical reserved capital)."""

    rows = list(result.get('selected_rows') or [])
    plans = list(result.get('selected_plans') or [])
    if len(rows) != len(plans):
        raise ValueError('Score×Capital selector selected_rows/plans長度不一致')
    values = []
    for row, plan in zip(rows, plans):
        score = _candidate_continuous_score(row)
        if score is None:
            continue
        reserved_cost_milli = int(plan.get('reserved_cost_milli', 0) or 0)
        values.append(float(score) * float(reserved_cost_milli))
    stable_base_ranks = tuple(
        -base_rank[id(row)]
        for row in sorted(rows, key=lambda item: base_rank[id(item)])
    )
    return (
        int(len(values)),
        float(sum(values)),
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


def build_max_dl_repair_mechanism_diagnostic(
    rows,
    *,
    available_cash,
    sizing_equity,
    params,
    resource_selection_diag,
):
    """Build a scalable research-only minimum-repair search certificate.

    Production minimum-repair already exhaustively evaluates every single swap at
    each accepted step.  Reuse that exact local-search trace instead of performing
    a combinatorial all-K-basket oracle.  A one-step repair is therefore an exact
    certificate for minimum replacement distance=1 and the best frozen-score
    feasible one-swap basket.  Multi-step paths remain explicitly unresolved beyond
    the production greedy path; feasible-ascent separately certifies a one-swap
    local optimum.  No target/outcome information participates in this diagnostic.
    """

    del available_cash, sizing_equity, params  # diagnostic uses already-completed production trace only
    diag = dict(resource_selection_diag or {})
    if not bool(diag.get('max_dl_eligible', False)):
        return None
    if str(diag.get('basket_objective') or 'score') != 'score':
        return None
    repair_steps = int(diag.get('max_dl_repair_steps', 0) or 0)
    if repair_steps <= 0:
        return None

    candidates = list(rows or [])
    target_count = int(diag.get('pre_market_order_limit', 0) or 0)
    reserve_floor_milli = int(diag.get('baseline_reserved_cost_milli', 0) or 0)
    trace = dict(diag.get('_selector_trace_baskets') or {})
    raw_rows = list(trace.get('raw_top_n') or [])
    repair_rows = list(trace.get('minimum_repair_seed') or [])
    final_rows = list(trace.get('feasible_ascent_final') or repair_rows)
    repair_trace = list(diag.get('_selector_repair_steps') or [])
    if target_count <= 0 or len(raw_rows) != target_count:
        return {
            'status': 'UNAVAILABLE_TRACE_CONTRACT',
            'reason': 'raw Top-N membership缺少或與pre-market order limit不一致',
        }

    base_rank = {id(row): idx for idx, row in enumerate(candidates)}

    def quality(rows_):
        ordered = _max_dl_execution_order(rows_, base_rank=base_rank)
        return _max_dl_basket_quality_key(ordered, base_rank=base_rank)

    def replacement_distance(rows_):
        raw_ids = {id(row) for row in raw_rows}
        candidate_ids = {id(row) for row in list(rows_ or [])}
        return int(len(raw_ids - candidate_ids))

    raw_quality = quality(raw_rows)
    repair_quality = quality(repair_rows)
    final_quality = quality(final_rows)
    raw_selected_count = None
    raw_reserved_cost_milli = None
    raw_count_deficit = None
    raw_reserve_deficit_milli = None
    if repair_trace:
        first = dict(repair_trace[0])
        raw_selected_count = int(first.get('before_selected_count', 0) or 0)
        raw_reserved_cost_milli = int(first.get('before_reserved_cost_milli', 0) or 0)
        raw_count_deficit = int(first.get('before_count_deficit', 0) or 0)
        raw_reserve_deficit_milli = int(first.get('before_reserve_deficit_milli', 0) or 0)

    fallback = bool(diag.get('max_dl_seed_fallback', False) or diag.get('max_dl_fallback_to_baseline', False))
    if fallback:
        classification = 'BASELINE_FALLBACK_AFTER_GREEDY_REPAIR'
    elif repair_steps == 1:
        classification = 'EXACT_ONE_SWAP_RESOURCE_CONSTRAINT'
    else:
        classification = 'MULTI_STEP_GREEDY_PATH_UNRESOLVED'

    return {
        'status': 'AVAILABLE',
        'classification': classification,
        'target_count': int(target_count),
        'candidate_count': int(len(candidates)),
        'reserve_floor_milli': int(reserve_floor_milli),
        'raw_selected_count': raw_selected_count,
        'raw_reserved_cost_milli': raw_reserved_cost_milli,
        'raw_count_deficit': raw_count_deficit,
        'raw_reserve_deficit_milli': raw_reserve_deficit_milli,
        'actual_repair_steps': int(repair_steps),
        'actual_repair_replacement_distance': int(replacement_distance(repair_rows)),
        'repair_seed_score_sum': float(repair_quality[1]),
        'raw_score_sum': float(raw_quality[1]),
        'final_score_sum': float(final_quality[1]),
        'repair_seed_score_loss_from_raw': float(repair_quality[1] - raw_quality[1]),
        'final_score_change_from_repair': float(final_quality[1] - repair_quality[1]),
        'repair_search_evaluations': int(diag.get('max_dl_repair_evaluations', 0) or 0),
        'ascent_search_evaluations': int(diag.get('max_dl_feasible_ascent_evaluations', 0) or 0),
        'ascent_local_optimum': bool(diag.get('max_dl_feasible_ascent_local_optimum', False)),
        'repair_steps_trace': repair_trace,
    }


def _reorder_resource_aware_continuous_max_dl(
    rows,
    *,
    available_cash,
    sizing_equity,
    free_slots,
    params,
    baseline,
    default_diag,
    objective_mode='score',
    preserve_reserve_floor=True,
    minimum_repair_enabled=True,
):
    """Let a frozen continuous objective own stock choice; the same-parameter DL-off baseline supplies exact pre-market resource floors.

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

    if objective_mode not in {'score', 'expected_pnl', 'excess_alpha'}:
        raise ValueError(f'不支援的max-DL objective_mode: {objective_mode!r}')

    target_count = int(baseline['selected_count'])
    reserve_floor_milli = (
        int(baseline['reserved_cost_milli']) if preserve_reserve_floor else 0
    )
    diag = _resource_aware_diag_from_result(
        default_diag,
        baseline,
        baseline,
        changed=False,
        promoted_pass_count=0,
        selector=(
            'continuous-expected-pnl'
            if objective_mode == 'expected_pnl'
            else 'continuous-excess-alpha'
            if objective_mode == 'excess_alpha'
            else 'continuous-score-max-dl'
        ),
    )
    diag.update({
        'resource_preservation_required': bool(preserve_reserve_floor),
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
    dl_order = (
        _expected_pnl_order(rows, base_rank=base_rank)
        if objective_mode == 'expected_pnl'
        else _excess_alpha_order(rows, base_rank=base_rank)
        if objective_mode == 'excess_alpha'
        else _max_dl_score_order(rows, base_rank=base_rank)
    )
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

    def quality_key(rows_, result_):
        if objective_mode == 'expected_pnl':
            return _expected_pnl_basket_quality_key(result_, base_rank=base_rank)
        if objective_mode == 'excess_alpha':
            return _excess_alpha_basket_quality_key(result_, base_rank=base_rank)
        return _max_dl_basket_quality_key(rows_, base_rank=base_rank)

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
        repair_trace_steps=None,
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
            'resource_preservation_required': bool(preserve_reserve_floor),
            'pre_market_order_limit': int(target_count),
            'max_dl_eligible': True,
            'max_dl_repair_steps': int(repair_steps),
            'max_dl_repair_evaluations': int(max(0, evaluated - 1)),
            'max_dl_fallback_to_baseline': bool(fallback),
            'basket_objective': str(objective_mode),
            # Transient research-only membership trace.  Candidate objects remain
            # in-memory and are serialized only by an explicit replay trace sink.
            '_selector_trace_baskets': {
                'raw_top_n': list(pure_basket),
                'minimum_repair_seed': list(result.get('selected_rows') or []),
            },
            '_selector_repair_steps': list(repair_trace_steps or []),
        })
        if int(result['selected_count']) != target_count:
            raise RuntimeError('max-DL Continuous輸出未維持同參數DL-off baseline預留單數')
        if preserve_reserve_floor and int(result['reserved_cost_milli']) < reserve_floor_milli:
            raise RuntimeError('max-DL Continuous輸出低於同參數DL-off baseline reserved-capital floor')
        return final_order, out

    if direct_feasible:
        return finalize(
            pure_ordered,
            pure_result,
            selector=_objective_selector_name(
                objective_mode,
                suffix='direct',
                no_r0=not preserve_reserve_floor,
            ),
            repair_steps=0,
            fallback=False,
            repair_trace_steps=[],
        )

    if not minimum_repair_enabled:
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
            reserve_floor_milli=0,
        ):
            raise RuntimeError('max-DL no-R0無法重建同參數DL-off K-only cash-feasible seed')
        feasible_count += 1
        return finalize(
            baseline_ordered,
            baseline_result,
            selector=_objective_selector_name(
                objective_mode,
                suffix='cash-feasible-seed',
                no_r0=True,
            ),
            repair_steps=0,
            fallback=True,
            preserve_basket_order=True,
            repair_trace_steps=[],
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
    repair_trace_steps = []

    def repair_step_payload(
        *, step, out_row, in_row, before_result, after_result,
        evaluated_swap_count, progress_swap_count, feasible_swap_count,
    ):
        before_deficit = _max_dl_resource_deficit(
            before_result,
            target_count=target_count,
            reserve_floor_milli=reserve_floor_milli,
        )
        after_deficit = _max_dl_resource_deficit(
            after_result,
            target_count=target_count,
            reserve_floor_milli=reserve_floor_milli,
        )
        return {
            'step': int(step),
            'out_row': out_row,
            'in_row': in_row,
            'before_selected_count': int(before_result['selected_count']),
            'after_selected_count': int(after_result['selected_count']),
            'before_reserved_cost_milli': int(before_result['reserved_cost_milli']),
            'after_reserved_cost_milli': int(after_result['reserved_cost_milli']),
            'before_count_deficit': int(before_deficit[0]),
            'after_count_deficit': int(after_deficit[0]),
            'before_reserve_deficit_milli': int(before_deficit[1]),
            'after_reserve_deficit_milli': int(after_deficit[1]),
            'before_quality_key': quality_key(before_result.get('selected_rows') or [], before_result),
            'after_quality_key': quality_key(after_result.get('selected_rows') or [], after_result),
            'after_feasible': bool(
                _max_dl_basket_is_feasible(
                    after_result,
                    target_count=target_count,
                    reserve_floor_milli=reserve_floor_milli,
                )
            ),
            'evaluated_swap_count': int(evaluated_swap_count),
            'progress_swap_count': int(progress_swap_count),
            'feasible_swap_count': int(feasible_swap_count),
        }

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
        step_evaluated = 0
        step_progress = 0
        step_feasible = 0

        for out_row in out_rows:
            for in_row in in_rows:
                step_evaluated += 1
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
                step_progress += 1

                quality_key_value = quality_key(trial_ordered, trial_result)
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
                    step_feasible += 1
                    if (
                        best_feasible_quality is None
                        or quality_key_value > best_feasible_quality
                        or (
                            quality_key_value == best_feasible_quality
                            and tie_key > best_feasible_tie
                        )
                    ):
                        best_feasible_quality = quality_key_value
                        best_feasible_tie = tie_key
                        best_feasible = (
                            out_row,
                            in_row,
                            trial_ordered,
                            trial_result,
                        )

                if (
                    best_progress_quality is None
                    or quality_key_value > best_progress_quality
                    or (
                        quality_key_value == best_progress_quality
                        and tie_key > best_progress_tie
                    )
                ):
                    best_progress_quality = quality_key_value
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
            repair_trace_steps.append(repair_step_payload(
                step=repair_step,
                out_row=out_row,
                in_row=in_row,
                before_result=current_result,
                after_result=trial_result,
                evaluated_swap_count=step_evaluated,
                progress_swap_count=step_progress,
                feasible_swap_count=step_feasible,
            ))
            repaired_out_ids.add(id(out_row))
            repaired_in_ids.add(id(in_row))
            return finalize(
                trial_ordered,
                trial_result,
                selector=_objective_selector_name(
                    objective_mode,
                    suffix='minimum-repair',
                    no_r0=not preserve_reserve_floor,
                ),
                repair_steps=repair_step,
                fallback=False,
                repair_trace_steps=repair_trace_steps,
            )

        if best_progress is None:
            break

        out_row, in_row, next_basket, next_result, next_deficit = best_progress
        repair_trace_steps.append(repair_step_payload(
            step=repair_step,
            out_row=out_row,
            in_row=in_row,
            before_result=current_result,
            after_result=next_result,
            evaluated_swap_count=step_evaluated,
            progress_swap_count=step_progress,
            feasible_swap_count=step_feasible,
        ))
        current_basket, current_result, current_deficit = next_basket, next_result, next_deficit
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
        selector=_objective_selector_name(
            objective_mode,
            suffix='baseline-fallback',
            no_r0=not preserve_reserve_floor,
        ),
        repair_steps=len(repaired_out_ids),
        fallback=True,
        preserve_basket_order=True,
        repair_trace_steps=repair_trace_steps,
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


def _reorder_resource_aware_continuous_constrained_optimal(
    rows,
    *,
    available_cash,
    sizing_equity,
    free_slots,
    params,
    baseline,
    default_diag,
    objective_mode,
    preserve_reserve_floor=True,
    preserve_baseline_safety_floor=False,
    safety_score_mode='raw',
    safety_floor_contract=None,
    target_count_override=None,
    selector_suffix='constrained-optimal',
):
    """Exactly maximize one frozen DL objective under canonical K/cash feasibility.

    The existing C42/C41 paths keep the same-parameter baseline R0 floor.  No-R0
    research paths keep only baseline K plus canonical sizing/cash/orderability and use
    the baseline solely as a feasible incumbent; no minimum-repair or feasible-ascent
    membership restriction is applied. Branch-and-bound visits the full candidate
    include/exclude tree in canonical baseline execution order. Every included state is
    rebuilt through canonical cash-capped reservation. Pruning uses only admissible
    optimistic bounds. A normal return certifies the exact lexicographic optimum for
    the selected objective under the requested constraints.
    """

    if objective_mode not in {'score', 'score_capital', 'excess_alpha'}:
        raise ValueError(f'constrained optimal不支援objective_mode={objective_mode!r}')

    candidates = list(rows or [])
    baseline_target_count = int(baseline['selected_count'])
    target_count = (
        int(baseline_target_count)
        if target_count_override is None
        else int(target_count_override)
    )
    if target_count < baseline_target_count:
        raise ValueError('constrained solver target_count不得低於同參數baseline K')
    if target_count > int(free_slots):
        raise ValueError('constrained solver target_count不得高於physical free slots')
    reserve_floor_milli = (
        int(baseline['reserved_cost_milli']) if preserve_reserve_floor else 0
    )
    min_entry_notional_milli = min(
        (
            int(get_min_entry_notional_milli(row.get('params_obj') or params))
            for row in candidates
        ),
        default=int(get_min_entry_notional_milli(params)),
    )
    if safety_score_mode not in {'raw', 'residual'}:
        raise ValueError(f'不支援的safety_score_mode={safety_score_mode!r}')
    safety_metric_fn = (
        _selected_continuous_residual_safety_score_metrics
        if safety_score_mode == 'residual'
        else _selected_continuous_safety_score_metrics
    )
    safety_candidate_fn = (
        _candidate_continuous_residual_safety_score
        if safety_score_mode == 'residual'
        else _candidate_continuous_safety_score
    )
    baseline_safety = safety_metric_fn(baseline)
    safety_floor_coverage = (
        int(baseline_safety['scored_count']) if preserve_baseline_safety_floor else 0
    )
    safety_floor_score_sum = (
        float(baseline_safety['score_sum']) if preserve_baseline_safety_floor else 0.0
    )

    def safety_metrics(result):
        return safety_metric_fn(result)

    def safety_floor_satisfied(result):
        if not preserve_baseline_safety_floor:
            return True
        metrics = safety_metrics(result)
        return (
            int(metrics['scored_count']) >= int(safety_floor_coverage)
            and float(metrics['score_sum']) + 1e-12 >= float(safety_floor_score_sum)
        )

    base_rank = {id(row): idx for idx, row in enumerate(candidates)}
    # Exact search must traverse candidates in canonical execution order.  The
    # recursive cash-reservation state is a prefix state, so reordering the search
    # frontier by score would invalidate otherwise-admissible cash/reserve pruning.
    search_candidates = list(candidates)
    use_baseline_k_seed = bool(
        preserve_reserve_floor and target_count == baseline_target_count
    )
    seed_source = (
        'c35-feasible-ascent'
        if use_baseline_k_seed and objective_mode == 'score'
        else 'c39-feasible-ascent'
        if use_baseline_k_seed and objective_mode == 'excess_alpha'
        else 'objective-top-count-or-baseline-order'
    )
    selector_name = _objective_selector_name(
        objective_mode,
        suffix=str(selector_suffix),
        no_r0=not preserve_reserve_floor,
    )

    seed_order = None
    seed_diag = {}
    if use_baseline_k_seed:
        seed_order, seed_diag = _reorder_resource_aware_continuous_max_dl_feasible_ascent(
            candidates,
            available_cash=available_cash,
            sizing_equity=sizing_equity,
            free_slots=free_slots,
            params=params,
            baseline=baseline,
            default_diag=default_diag,
            objective_mode=objective_mode,
        )
        if (
            not bool(seed_diag.get('max_dl_eligible', False))
            and not preserve_baseline_safety_floor
        ):
            out = dict(seed_diag)
            out.update({
                'selector': selector_name,
                'basket_objective': str(objective_mode),
                'constrained_solver_optimality_certified': True,
                'constrained_solver_search_states': 0,
                'constrained_solver_pruned_states': 0,
                'constrained_solver_feasible_baskets': 0,
                'constrained_solver_seed_source': seed_source,
                'max_dl_repair_steps': 0,
                'max_dl_repair_evaluations': 0,
                'max_dl_feasible_ascent_steps': 0,
                'max_dl_feasible_ascent_evaluations': 0,
                'max_dl_feasible_ascent_local_optimum': False,
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

    def basket_quality_key(result):
        if objective_mode == 'score':
            return _max_dl_basket_quality_key(
                result.get('selected_rows') or [],
                base_rank=base_rank,
            )
        if objective_mode == 'score_capital':
            return _score_capital_basket_quality_key(result, base_rank=base_rank)
        return _excess_alpha_basket_quality_key(result, base_rank=base_rank)

    standalone = []
    for row in search_candidates:
        single_order, single_result = evaluate_basket([row])
        del single_order
        if int(single_result['selected_count']) != 1:
            standalone.append({
                'orderable': False,
                'reserve_upper_milli': 0,
                'coverage_upper': 0,
                'objective_upper': 0.0,
                'safety_coverage_upper': 0,
                'safety_score_upper': 0.0,
            })
            continue

        if objective_mode == 'score':
            score = _candidate_continuous_score(row)
            coverage = int(score is not None)
            raw_value = 0.0 if score is None else float(score)
        elif objective_mode == 'score_capital':
            score = _candidate_continuous_score(row)
            plan = single_result['selected_plans'][0]
            reserved_cost_milli = int(plan.get('reserved_cost_milli', 0) or 0)
            coverage = int(score is not None)
            raw_value = (
                0.0
                if not coverage
                else float(score) * float(reserved_cost_milli)
            )
        else:
            plan = single_result['selected_plans'][0]
            expected_excess_r = _candidate_expected_excess_r(row)
            risk_milli = _planned_initial_risk_milli(row=row, plan=plan)
            coverage = int(expected_excess_r is not None and risk_milli is not None)
            raw_value = (
                0.0
                if not coverage
                else float(expected_excess_r) * float(risk_milli)
            )

        safety_score = safety_candidate_fn(row)
        standalone.append({
            'orderable': True,
            'reserve_upper_milli': int(single_result['reserved_cost_milli']),
            'coverage_upper': coverage,
            # Zero is a valid optimistic replacement for a negative contribution:
            # exact K may force negatives, so ignoring them only loosens the bound.
            'objective_upper': max(0.0, float(raw_value)),
            'safety_coverage_upper': int(safety_score is not None),
            'safety_score_upper': (0.0 if safety_score is None else float(safety_score)),
        })

    n = len(search_candidates)
    suffix_orderable = [0] * (n + 1)
    suffix_coverage = [0] * (n + 1)
    suffix_safety_coverage = [0] * (n + 1)
    # The old implementation sorted the entire remaining suffix at every search
    # state to obtain top-N optimistic bounds.  K-Flex increases target_count,
    # making that O(states * n log n) overhead dominant.  Cache the exact same
    # top values once per suffix; lookup below is then O(K), with K<=free_slots.
    suffix_reserve_tops = [()] * (n + 1)
    suffix_objective_tops = [()] * (n + 1)
    suffix_safety_tops = [()] * (n + 1)

    def _prepend_top(previous, value):
        values = list(previous)
        values.append(value)
        values.sort(reverse=True)
        return tuple(values[:target_count])

    for idx in range(n - 1, -1, -1):
        item = standalone[idx]
        suffix_orderable[idx] = suffix_orderable[idx + 1] + int(item['orderable'])
        suffix_coverage[idx] = suffix_coverage[idx + 1] + int(item['coverage_upper'])
        suffix_safety_coverage[idx] = (
            suffix_safety_coverage[idx + 1] + int(item['safety_coverage_upper'])
        )
        if item['orderable']:
            suffix_reserve_tops[idx] = _prepend_top(
                suffix_reserve_tops[idx + 1],
                int(item['reserve_upper_milli']),
            )
            suffix_objective_tops[idx] = _prepend_top(
                suffix_objective_tops[idx + 1],
                float(item['objective_upper']),
            )
            suffix_safety_tops[idx] = _prepend_top(
                suffix_safety_tops[idx + 1],
                float(item['safety_score_upper']),
            )
        else:
            suffix_reserve_tops[idx] = suffix_reserve_tops[idx + 1]
            suffix_objective_tops[idx] = suffix_objective_tops[idx + 1]
            suffix_safety_tops[idx] = suffix_safety_tops[idx + 1]

    if use_baseline_k_seed:
        seed_basket = list(seed_order[:target_count])
    else:
        if objective_mode == 'score':
            seed_ranked = _max_dl_score_order(candidates, base_rank=base_rank)
        else:
            seed_ranked = sorted(
                candidates,
                key=lambda row: (
                    0 if standalone[base_rank[id(row)]]['coverage_upper'] else 1,
                    -float(standalone[base_rank[id(row)]]['objective_upper']),
                    base_rank[id(row)],
                ),
            )
        seed_basket = list(seed_ranked[:target_count])

    incumbent_candidates = [list(seed_basket)]
    if target_count == baseline_target_count:
        incumbent_candidates.append(list(baseline.get('selected_rows') or []))
    else:
        # K-Flex may need a different membership simply to fit more physical slots.
        # These are only deterministic incumbents; exact search still owns the result.
        incumbent_candidates.extend(
            (
                list(candidates),
                [
                    search_candidates[pos]
                    for pos in sorted(
                        range(n),
                        key=lambda idx: (
                            int(standalone[idx]['reserve_upper_milli']),
                            base_rank[id(search_candidates[idx])],
                        ),
                    )
                ][:target_count],
                [
                    search_candidates[pos]
                    for pos in sorted(
                        range(n),
                        key=lambda idx: (
                            -int(standalone[idx]['reserve_upper_milli']),
                            base_rank[id(search_candidates[idx])],
                        ),
                    )
                ][:target_count],
            )
        )

    if target_count > baseline_target_count:
        baseline_rows = list(baseline.get('selected_rows') or [])
        baseline_ids = {id(row) for row in baseline_rows}
        extra_needed = max(0, int(target_count - len(baseline_rows)))
        if extra_needed:
            score_extras = [
                row for row in _max_dl_score_order(candidates, base_rank=base_rank)
                if id(row) not in baseline_ids
            ][:extra_needed]
            reserve_extras = [
                search_candidates[pos]
                for pos in sorted(
                    range(n),
                    key=lambda idx: (
                        int(standalone[idx]['reserve_upper_milli']),
                        base_rank[id(search_candidates[idx])],
                    ),
                )
                if id(search_candidates[pos]) not in baseline_ids
            ][:extra_needed]
            incumbent_candidates.extend((
                baseline_rows + score_extras,
                baseline_rows + reserve_extras,
            ))

    best_basket = None
    best_result = None
    best_key = None
    direct_score_order_feasible = False
    for seed_idx, incumbent_basket in enumerate(incumbent_candidates):
        incumbent_order, incumbent_result = evaluate_basket(incumbent_basket)
        feasible = (
            _max_dl_basket_is_feasible(
                incumbent_result,
                target_count=target_count,
                reserve_floor_milli=reserve_floor_milli,
            )
            and safety_floor_satisfied(incumbent_result)
        )
        if seed_idx == 0:
            direct_score_order_feasible = bool(feasible)
        if not feasible:
            continue
        incumbent_key = basket_quality_key(incumbent_result)
        if best_key is None or incumbent_key > best_key:
            best_basket = list(incumbent_result.get('selected_rows') or [])
            best_result = incumbent_result
            best_key = incumbent_key

    if not use_baseline_k_seed:
        seed_diag = {
            'direct_score_order_feasible': bool(direct_score_order_feasible),
            '_selector_trace_baskets': {
                'raw_top_n': list(seed_basket),
            },
        }

    search_states = 0
    pruned_states = 0
    safety_pruned_states = 0
    feasible_baskets = int(best_result is not None)

    def current_objective_summary(result):
        if objective_mode == 'score':
            values = [
                float(score)
                for row in list(result.get('selected_rows') or [])
                if (score := _candidate_continuous_score(row)) is not None
            ]
            return int(len(values)), float(sum(values))
        if objective_mode == 'score_capital':
            values = []
            for row, plan in zip(
                list(result.get('selected_rows') or []),
                list(result.get('selected_plans') or []),
            ):
                score = _candidate_continuous_score(row)
                if score is None:
                    continue
                values.append(
                    float(score) * float(int(plan.get('reserved_cost_milli', 0) or 0))
                )
            return int(len(values)), float(sum(values))

        coverage = 0
        total = 0.0
        for row, plan in zip(
            list(result.get('selected_rows') or []),
            list(result.get('selected_plans') or []),
        ):
            expected_excess_r = _candidate_expected_excess_r(row)
            if expected_excess_r is None:
                continue
            risk_milli = _planned_initial_risk_milli(row=row, plan=plan)
            if risk_milli is None:
                continue
            coverage += 1
            total += float(expected_excess_r) * float(risk_milli)
        return int(coverage), float(total)

    def optimistic_reserve_upper(idx, need, current_reserved):
        if need <= 0:
            return int(current_reserved)
        return int(current_reserved) + int(sum(suffix_reserve_tops[idx][:need]))

    def optimistic_objective_upper(idx, need, current_total):
        if need <= 0:
            return float(current_total)
        return float(current_total) + float(sum(suffix_objective_tops[idx][:need]))

    def optimistic_safety_upper(idx, need, current_total):
        if need <= 0:
            return float(current_total)
        return float(current_total) + float(sum(suffix_safety_tops[idx][:need]))

    empty_result = _simulate_reserved_candidate_order(
        [],
        available_cash=available_cash,
        sizing_equity=sizing_equity,
        free_slots=target_count,
        params=params,
    )

    def visit(idx, basket, result, coverage, objective_total):
        nonlocal best_basket, best_result, best_key
        nonlocal search_states, pruned_states, safety_pruned_states, feasible_baskets
        search_states += 1
        selected_count = int(result['selected_count'])
        need = int(target_count - selected_count)

        if need == 0:
            if int(result['reserved_cost_milli']) < reserve_floor_milli:
                pruned_states += 1
                return
            if not safety_floor_satisfied(result):
                pruned_states += 1
                safety_pruned_states += 1
                return
            feasible_baskets += 1
            quality = basket_quality_key(result)
            if best_key is None or quality > best_key:
                best_key = quality
                best_basket = list(result.get('selected_rows') or basket)
                best_result = result
            return

        if idx >= n or suffix_orderable[idx] < need:
            pruned_states += 1
            return
        # Every additional canonical order must satisfy the shared minimum entry
        # notional.  Reserved buy cost is never below notional, so this is an
        # admissible cash-feasibility lower bound and can prove high-count states
        # impossible without enumerating their membership combinations.
        if (
            min_entry_notional_milli > 0
            and int(result['remaining_cash_milli'])
            < int(need) * int(min_entry_notional_milli)
        ):
            pruned_states += 1
            return
        if optimistic_reserve_upper(
            idx, need, result['reserved_cost_milli']
        ) < reserve_floor_milli:
            pruned_states += 1
            return

        if preserve_baseline_safety_floor:
            current_safety = safety_metrics(result)
            max_safety_coverage = int(current_safety['scored_count']) + min(
                int(need), int(suffix_safety_coverage[idx])
            )
            if max_safety_coverage < int(safety_floor_coverage):
                pruned_states += 1
                safety_pruned_states += 1
                return
            if optimistic_safety_upper(
                idx, need, float(current_safety['score_sum'])
            ) + 1e-12 < float(safety_floor_score_sum):
                pruned_states += 1
                safety_pruned_states += 1
                return

        if best_key is not None:
            max_coverage = int(coverage) + min(int(need), int(suffix_coverage[idx]))
            if max_coverage < int(best_key[0]):
                pruned_states += 1
                return
            if max_coverage == int(best_key[0]):
                objective_upper = optimistic_objective_upper(idx, need, objective_total)
                if objective_upper < float(best_key[1]):
                    pruned_states += 1
                    return

        row = search_candidates[idx]

        if standalone[idx]['orderable']:
            include_basket = list(basket) + [row]
            include_order, include_result = evaluate_basket(include_basket)
            if int(include_result['selected_count']) == selected_count + 1:
                new_coverage, new_total = current_objective_summary(include_result)
                visit(
                    idx + 1,
                    include_order,
                    include_result,
                    new_coverage,
                    new_total,
                )

        visit(idx + 1, basket, result, coverage, objective_total)

    direct_score_optimum_shortcut = bool(
        objective_mode == 'score' and direct_score_order_feasible
    )
    if not direct_score_optimum_shortcut:
        empty_coverage, empty_total = current_objective_summary(empty_result)
        visit(0, [], empty_result, empty_coverage, empty_total)

    if best_result is None:
        raise _NoFeasibleConstrainedBasket(
            f'constrained solver找不到target_count={target_count}的合法R0/cash basket'
        )
    if (
        not _max_dl_basket_is_feasible(
            best_result,
            target_count=target_count,
            reserve_floor_milli=reserve_floor_milli,
        )
        or not safety_floor_satisfied(best_result)
    ):
        raise RuntimeError('constrained solver結束後沒有合法K/R0/safety basket')

    basket_order = _max_dl_execution_order(best_basket, base_rank=base_rank)
    basket_ids = {id(row) for row in basket_order}
    final_result = _simulate_reserved_candidate_order(
        basket_order,
        available_cash=available_cash,
        sizing_equity=sizing_equity,
        free_slots=target_count,
        params=params,
    )
    if basket_quality_key(final_result) != best_key:
        raise RuntimeError('constrained solver final execution order改變objective optimum')
    final_order = basket_order + [
        row for row in candidates if id(row) not in basket_ids
    ]

    baseline_ids = {id(row) for row in baseline['selected_rows']}
    selected_ids = {id(row) for row in best_result['selected_rows']}
    promoted_count = sum(
        1 for row in best_result['selected_rows'] if id(row) not in baseline_ids
    )
    seed_trace = dict(seed_diag.get('_selector_trace_baskets') or {})
    selected_safety = safety_metrics(best_result)
    safety_floor_binding = bool(
        preserve_baseline_safety_floor
        and int(selected_safety['scored_count']) == int(safety_floor_coverage)
        and math.isclose(
            float(selected_safety['score_sum']),
            float(safety_floor_score_sum),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )
    out = _resource_aware_diag_from_result(
        default_diag,
        baseline,
        best_result,
        changed=bool(selected_ids != baseline_ids),
        promoted_pass_count=0,
        selector=selector_name,
    )
    out.update({
        'mode': 'dl-selection',
        'promoted_score_orders': int(promoted_count),
        'direct_score_order_feasible': bool(
            seed_diag.get('direct_score_order_feasible', False)
        ),
        'basket_search_states': int(search_states),
        'basket_search_pruned': int(pruned_states),
        'basket_feasible_count': int(feasible_baskets),
        'resource_preservation_required': bool(preserve_reserve_floor),
        'pre_market_order_limit': int(target_count),
        'max_dl_eligible': True,
        'max_dl_repair_steps': 0,
        'max_dl_repair_evaluations': 0,
        'max_dl_seed_fallback': False,
        'max_dl_fallback_to_baseline': False,
        'max_dl_feasible_ascent_steps': 0,
        'max_dl_feasible_ascent_evaluations': 0,
        'max_dl_feasible_ascent_local_optimum': False,
        'basket_objective': str(objective_mode),
        'constrained_solver_optimality_certified': True,
        'constrained_solver_direct_score_optimum_shortcut': bool(
            direct_score_optimum_shortcut
        ),
        'constrained_solver_search_states': int(search_states),
        'constrained_solver_pruned_states': int(pruned_states),
        'constrained_solver_feasible_baskets': int(feasible_baskets),
        'constrained_solver_safety_pruned_states': int(safety_pruned_states),
        'safety_floor_enabled': bool(preserve_baseline_safety_floor),
        'safety_floor_contract': (
            str(safety_floor_contract or 'baseline_coverage_and_score_sum_floor_v1')
            if preserve_baseline_safety_floor else None
        ),
        'safety_score_mode': str(safety_score_mode),
        'baseline_safety_scored_count': int(baseline_safety['scored_count']),
        'selected_safety_scored_count': int(selected_safety['scored_count']),
        'baseline_safety_score_sum': float(baseline_safety['score_sum']),
        'selected_safety_score_sum': float(selected_safety['score_sum']),
        'baseline_safety_score_mean': baseline_safety['score_mean'],
        'selected_safety_score_mean': selected_safety['score_mean'],
        'safety_floor_binding': bool(safety_floor_binding),
        'safety_floor_violation': bool(not safety_floor_satisfied(best_result)),
        'constrained_solver_seed_source': seed_source,
        'constrained_solver_seed_repair_evaluations': int(
            seed_diag.get('max_dl_repair_evaluations', 0) or 0
        ),
        'constrained_solver_seed_ascent_evaluations': int(
            seed_diag.get('max_dl_feasible_ascent_evaluations', 0) or 0
        ),
        '_selector_trace_baskets': {
            'raw_top_n': list(seed_trace.get('raw_top_n') or []),
            'minimum_repair_seed': [],
            'feasible_ascent_final': [],
            'constrained_optimal_final': list(best_result.get('selected_rows') or []),
        },
        '_selector_repair_steps': [],
    })
    if int(best_result['selected_count']) != target_count:
        raise RuntimeError('constrained solver未維持同參數baseline K')
    if preserve_reserve_floor and int(best_result['reserved_cost_milli']) < reserve_floor_milli:
        raise RuntimeError('constrained solver輸出低於同參數baseline R0')
    if preserve_baseline_safety_floor and not safety_floor_satisfied(best_result):
        raise RuntimeError('constrained solver輸出低於同參數baseline safety floor')
    return final_order, out


def _reorder_resource_aware_continuous_excess_alpha_constrained_optimal(
    rows,
    *,
    available_cash,
    sizing_equity,
    free_slots,
    params,
    baseline,
    default_diag,
):
    return _reorder_resource_aware_continuous_constrained_optimal(
        rows,
        available_cash=available_cash,
        sizing_equity=sizing_equity,
        free_slots=free_slots,
        params=params,
        baseline=baseline,
        default_diag=default_diag,
        objective_mode='excess_alpha',
    )


def _reorder_resource_aware_continuous_score_constrained_optimal(
    rows,
    *,
    available_cash,
    sizing_equity,
    free_slots,
    params,
    baseline,
    default_diag,
):
    return _reorder_resource_aware_continuous_constrained_optimal(
        rows,
        available_cash=available_cash,
        sizing_equity=sizing_equity,
        free_slots=free_slots,
        params=params,
        baseline=baseline,
        default_diag=default_diag,
        objective_mode='score',
    )

def _reorder_resource_aware_continuous_score_k_flex_r0_constrained_optimal(
    rows,
    *,
    available_cash,
    sizing_equity,
    free_slots,
    params,
    baseline,
    default_diag,
):
    """Maximize feasible planned-order count, then the unchanged frozen score objective.

    SR-C67 changes only the C59 count contract: baseline K becomes a minimum and
    physical free slots become the upper bound.  The baseline R0 floor, canonical
    sizing/cash reservation, candidate universe and raw score objective are unchanged.
    Counts are tried from the physical cap downward; the first feasible count is then
    solved exactly by the same branch-and-bound owner used by C59.
    """

    candidates = list(rows or [])
    baseline_k = int(baseline['selected_count'])
    physical_free_slots = max(0, int(free_slots))
    if baseline_k < 0:
        raise ValueError('K-Flex baseline K不得小於0')
    if baseline_k > physical_free_slots:
        raise RuntimeError('K-Flex baseline K高於physical free slots')
    min_entry_notional_milli = min(
        (
            int(get_min_entry_notional_milli(row.get('params_obj') or params))
            for row in candidates
        ),
        default=int(get_min_entry_notional_milli(params)),
    )
    cash_count_cap = (
        int(coerce_money_like_to_milli(available_cash)) // min_entry_notional_milli
        if min_entry_notional_milli > 0
        else physical_free_slots
    )
    max_count = min(physical_free_slots, len(candidates), int(cash_count_cap))
    if max_count < baseline_k:
        raise RuntimeError('K-Flex候選數不足以維持baseline K')

    attempted_counts = []
    for target_count in range(max_count, baseline_k - 1, -1):
        attempted_counts.append(int(target_count))
        try:
            order, diag = _reorder_resource_aware_continuous_constrained_optimal(
                candidates,
                available_cash=available_cash,
                sizing_equity=sizing_equity,
                free_slots=physical_free_slots,
                params=params,
                baseline=baseline,
                default_diag=default_diag,
                objective_mode='score',
                preserve_reserve_floor=True,
                target_count_override=target_count,
                selector_suffix='k-flex-r0-constrained-optimal',
            )
        except _NoFeasibleConstrainedBasket:
            continue

        out = dict(diag)
        selected_count = int(out.get('selected_count', 0) or 0)
        reserved_milli = int(out.get('reserved_cost_milli', 0) or 0)
        baseline_r0_milli = int(baseline['reserved_cost_milli'])
        if selected_count != int(target_count):
            raise RuntimeError('K-Flex exact solver輸出count與target_count不一致')
        if not (baseline_k <= selected_count <= physical_free_slots):
            raise RuntimeError('K-Flex輸出違反baseline K至physical free-slot範圍')
        if reserved_milli < baseline_r0_milli:
            raise RuntimeError('K-Flex輸出低於同參數baseline R0')

        out.update({
            'selector': 'continuous-score-k-flex-r0-constrained-optimal',
            'count_constraint': 'baseline_k_to_physical_free_slots_v1',
            'baseline_k': int(baseline_k),
            'physical_free_slots': int(physical_free_slots),
            'k_flex_max_count': int(max_count),
            'k_flex_cash_count_cap': int(cash_count_cap),
            'k_flex_selected_count': int(selected_count),
            'k_flex_extra_positions': int(selected_count - baseline_k),
            'k_flex_target_counts_attempted': tuple(attempted_counts),
            'k_flex_r0_preserved': True,
            'pre_market_order_limit': int(selected_count),
        })
        return order, out

    raise RuntimeError('K-Flex找不到任何可維持baseline K/R0的canonical cash-feasible basket')


def _reorder_resource_aware_continuous_score_safety_constrained_optimal(
    rows,
    *,
    available_cash,
    sizing_equity,
    free_slots,
    params,
    baseline,
    default_diag,
):
    order, diag = _reorder_resource_aware_continuous_constrained_optimal(
        rows,
        available_cash=available_cash,
        sizing_equity=sizing_equity,
        free_slots=free_slots,
        params=params,
        baseline=baseline,
        default_diag=default_diag,
        objective_mode='score',
        preserve_baseline_safety_floor=True,
    )
    diag = dict(diag)
    diag['selector'] = 'continuous-score-safety-constrained-optimal'
    return order, diag

def _reorder_resource_aware_continuous_score_residual_safety_constrained_optimal(
    rows,
    *,
    available_cash,
    sizing_equity,
    free_slots,
    params,
    baseline,
    default_diag,
):
    order, diag = _reorder_resource_aware_continuous_constrained_optimal(
        rows,
        available_cash=available_cash,
        sizing_equity=sizing_equity,
        free_slots=free_slots,
        params=params,
        baseline=baseline,
        default_diag=default_diag,
        objective_mode='score',
        preserve_baseline_safety_floor=True,
        safety_score_mode='residual',
        safety_floor_contract='baseline_residual_coverage_and_score_sum_floor_v1',
    )
    diag = dict(diag)
    diag['selector'] = 'continuous-score-residual-safety-constrained-optimal'
    return order, diag

def _reorder_resource_aware_continuous_score_no_k_no_r0(
    rows,
    *,
    available_cash,
    sizing_equity,
    free_slots,
    params,
    baseline,
    default_diag,
):
    """SR-C70: keep the C66 score order, remove only C58-derived K and R0.

    Membership is proposed by the unchanged frozen ``model_score`` descending
    order with the original deterministic candidate rank as the tie-break.  The
    canonical reservation simulator remains the sole sizing/cash/orderability
    feasibility owner and may select any count from zero through physical free
    slots.  There is no baseline-K minimum/target and no baseline reserved-capital
    floor or repair path.
    """

    candidates = list(rows or [])
    physical_free_slots = max(0, int(free_slots))
    base_rank = {id(row): idx for idx, row in enumerate(candidates)}
    score_order = _max_dl_score_order(candidates, base_rank=base_rank)
    selected = _simulate_reserved_candidate_order(
        score_order,
        available_cash=available_cash,
        sizing_equity=sizing_equity,
        free_slots=physical_free_slots,
        params=params,
    )
    diag = _resource_aware_diag_from_result(
        default_diag,
        baseline,
        selected,
        changed=bool([id(row) for row in score_order] != [id(row) for row in candidates]),
        promoted_pass_count=0,
        selector='continuous-score-no-k-no-r0',
    )
    selected_count = int(selected.get('selected_count', 0) or 0)
    if not (0 <= selected_count <= physical_free_slots):
        raise RuntimeError('No-K/No-R0 selector輸出超過physical free-slot contract')
    diag.update({
        'mode': 'dl-selection',
        'resource_preservation_required': False,
        'count_constraint': 'zero_to_physical_free_slots_v1',
        'baseline_k': int(baseline.get('selected_count', 0) or 0),
        'physical_free_slots': int(physical_free_slots),
        'no_k_no_r0': True,
        'preserve_k': False,
        'preserve_r0': False,
        'pre_market_order_limit': int(selected_count),
        'direct_score_order_feasible': True,
        'basket_search_states': 1,
        'basket_feasible_count': 1,
        'constrained_solver_optimality_certified': False,
    })
    return score_order, diag


def _reorder_resource_aware_continuous_score_no_k_no_r0_raw_safety_gate(
    rows,
    *,
    available_cash,
    sizing_equity,
    free_slots,
    params,
    baseline,
    default_diag,
    safety_percentile_cutoff,
):
    """SR-C71: C70 membership mechanics plus a Raw Safety percentile gate only.

    K and R0 remain absent exactly as in C70.  Eligibility is restricted to
    candidates with an available MR-13R Raw Safety head score whose same-day
    orderable-candidate average-rank percentile is at least the configured cutoff.
    Eligible names retain the unchanged Conditional-MFE model-score order and are
    passed through the same canonical reservation simulator.
    """

    candidates = list(rows or [])
    physical_free_slots = max(0, int(free_slots))
    cutoff = float(safety_percentile_cutoff)
    if not (0.0 <= cutoff <= 1.0):
        raise ValueError('Raw Safety percentile cutoff必須介於0與1')
    eligible = []
    for row in candidates:
        value = row.get('breakout_quality_safety_score_percentile')
        try:
            percentile = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(percentile) and percentile >= cutoff:
            eligible.append(row)
    base_rank = {id(row): idx for idx, row in enumerate(candidates)}
    score_order = _max_dl_score_order(eligible, base_rank=base_rank)
    selected = _simulate_reserved_candidate_order(
        score_order,
        available_cash=available_cash,
        sizing_equity=sizing_equity,
        free_slots=physical_free_slots,
        params=params,
    )
    diag = _resource_aware_diag_from_result(
        default_diag,
        baseline,
        selected,
        changed=True,
        promoted_pass_count=0,
        selector='continuous-score-no-k-no-r0-raw-safety-gate',
    )
    selected_count = int(selected.get('selected_count', 0) or 0)
    if not (0 <= selected_count <= physical_free_slots):
        raise RuntimeError('No-K/No-R0 Raw Safety gate輸出超過physical free-slot contract')
    # Keep ineligible rows after the eligible score order for full diagnostics while
    # pre_market_order_limit prevents them from creating orders.
    eligible_ids = {id(row) for row in score_order}
    final_order = list(score_order) + [row for row in candidates if id(row) not in eligible_ids]
    diag.update({
        'mode': 'dl-selection',
        'resource_preservation_required': False,
        'count_constraint': 'zero_to_physical_free_slots_v1',
        'baseline_k': int(baseline.get('selected_count', 0) or 0),
        'physical_free_slots': int(physical_free_slots),
        'no_k_no_r0': True,
        'preserve_k': False,
        'preserve_r0': False,
        'raw_safety_gate_enabled': True,
        'raw_safety_gate_percentile_cutoff': float(cutoff),
        'raw_safety_gate_eligible_count': int(len(score_order)),
        'pre_market_order_limit': int(selected_count),
        'direct_score_order_feasible': True,
        'basket_search_states': 1,
        'basket_feasible_count': 1,
        'constrained_solver_optimality_certified': False,
    })
    return final_order, diag


def _reorder_resource_aware_continuous_score_no_k_no_r0_safety_mfe_product(
    rows,
    *,
    available_cash,
    sizing_equity,
    free_slots,
    params,
    baseline,
    default_diag,
):
    """SR-C74: rank No-K/No-R0 candidates by Safety×Conditional-MFE percentiles.

    There is no Safety threshold and no fitted weight.  Candidate membership is the
    same orderable cross-section, except rows missing either decision-time head score
    cannot receive a joint score and therefore cannot create an order.  The product
    score is precomputed by the shared same-day percentile decorator; ties fall back
    only to the canonical incoming rank.
    """

    candidates = list(rows or [])
    physical_free_slots = max(0, int(free_slots))
    base_rank = {id(row): idx for idx, row in enumerate(candidates)}
    eligible = []
    for row in candidates:
        if not bool(row.get('breakout_quality_safety_mfe_product_available', False)):
            continue
        try:
            score = float(row.get('breakout_quality_safety_mfe_product_score'))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(score):
            continue
        eligible.append(row)
    joint_order = sorted(
        eligible,
        key=lambda row: (
            -float(row['breakout_quality_safety_mfe_product_score']),
            base_rank[id(row)],
        ),
    )
    selected = _simulate_reserved_candidate_order(
        joint_order,
        available_cash=available_cash,
        sizing_equity=sizing_equity,
        free_slots=physical_free_slots,
        params=params,
    )
    diag = _resource_aware_diag_from_result(
        default_diag,
        baseline,
        selected,
        changed=True,
        promoted_pass_count=0,
        selector='continuous-score-no-k-no-r0-safety-mfe-product',
    )
    selected_count = int(selected.get('selected_count', 0) or 0)
    if not (0 <= selected_count <= physical_free_slots):
        raise RuntimeError('No-K/No-R0 Safety×MFE product輸出超過physical free-slot contract')
    eligible_ids = {id(row) for row in joint_order}
    final_order = list(joint_order) + [row for row in candidates if id(row) not in eligible_ids]
    selected_rows = list(selected.get('selected_rows') or [])
    selected_products = [
        float(row['breakout_quality_safety_mfe_product_score'])
        for row in selected_rows
        if bool(row.get('breakout_quality_safety_mfe_product_available', False))
    ]
    diag.update({
        'mode': 'dl-selection',
        'resource_preservation_required': False,
        'count_constraint': 'zero_to_physical_free_slots_v1',
        'baseline_k': int(baseline.get('selected_count', 0) or 0),
        'physical_free_slots': int(physical_free_slots),
        'no_k_no_r0': True,
        'preserve_k': False,
        'preserve_r0': False,
        'safety_mfe_product_enabled': True,
        'safety_mfe_product_eligible_count': int(len(joint_order)),
        'safety_mfe_product_selected_score_mean': (
            None
            if not selected_products
            else float(sum(selected_products) / len(selected_products))
        ),
        'pre_market_order_limit': int(selected_count),
        'direct_score_order_feasible': True,
        'basket_search_states': 1,
        'basket_feasible_count': 1,
        'constrained_solver_optimality_certified': False,
    })
    return final_order, diag


def _reorder_resource_aware_continuous_score_no_r0_constrained_optimal(
    rows,
    *,
    available_cash,
    sizing_equity,
    free_slots,
    params,
    baseline,
    default_diag,
):
    return _reorder_resource_aware_continuous_constrained_optimal(
        rows,
        available_cash=available_cash,
        sizing_equity=sizing_equity,
        free_slots=free_slots,
        params=params,
        baseline=baseline,
        default_diag=default_diag,
        objective_mode='score',
        preserve_reserve_floor=False,
    )


def _reorder_resource_aware_continuous_score_capital_no_r0_constrained_optimal(
    rows,
    *,
    available_cash,
    sizing_equity,
    free_slots,
    params,
    baseline,
    default_diag,
):
    return _reorder_resource_aware_continuous_constrained_optimal(
        rows,
        available_cash=available_cash,
        sizing_equity=sizing_equity,
        free_slots=free_slots,
        params=params,
        baseline=baseline,
        default_diag=default_diag,
        objective_mode='score_capital',
        preserve_reserve_floor=False,
    )


def _score_capital_pareto_summary(result, *, base_rank):
    rows = list(result.get('selected_rows') or [])
    scores = [
        float(score)
        for row in rows
        if (score := _candidate_continuous_score(row)) is not None
    ]
    stable_base_ranks = tuple(
        -base_rank[id(row)]
        for row in sorted(rows, key=lambda item: base_rank[id(item)])
    )
    return (
        int(len(scores)),
        float(sum(scores)),
        int(result.get('reserved_cost_milli', 0) or 0),
        stable_base_ranks,
    )


def _normalize_pareto_axis(value, *, low, high):
    value = float(value)
    low = float(low)
    high = float(high)
    if high <= low + 1e-12:
        return 1.0
    return min(1.0, max(0.0, (value - low) / (high - low)))


def _score_capital_pareto_basket_quality_key(
    result,
    *,
    base_rank,
    quality_min,
    quality_max,
    capital_min_milli,
    capital_max_milli,
):
    coverage, quality, capital_milli, stable = _score_capital_pareto_summary(
        result, base_rank=base_rank
    )
    quality_norm = _normalize_pareto_axis(
        quality, low=quality_min, high=quality_max
    )
    capital_norm = _normalize_pareto_axis(
        capital_milli, low=capital_min_milli, high=capital_max_milli
    )
    return (
        int(coverage),
        float(quality_norm * capital_norm),
        float(quality),
        int(capital_milli),
        stable,
    )


def _reorder_resource_aware_continuous_score_capital_pareto_no_r0_constrained_optimal(
    rows,
    *,
    available_cash,
    sizing_equity,
    free_slots,
    params,
    baseline,
    default_diag,
):
    """Exact C48 basket-level Pareto selection under K/canonical-cash and no R0.

    Missing scores never gain an advantage over scored rows: every pass first
    maximizes score coverage.  Within that maximal-coverage universe, pass 1
    finds the score-sum endpoint and pass 2 the canonical-reserved-capital
    endpoint.  These two efficient endpoints define the no-lambda normalization
    range.  Pass 3 exactly maximizes normalized quality * normalized capital.
    Because the scalar is monotone in both axes and deterministic ties prefer
    higher quality then higher capital, the selected basket is Pareto-efficient.
    """

    candidates = list(rows or [])
    target_count = int(baseline['selected_count'])
    base_rank = {id(row): idx for idx, row in enumerate(candidates)}
    selector_name = 'continuous-score-capital-pareto-no-r0-constrained-optimal'

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

    baseline_order, baseline_result = evaluate_basket(
        list(baseline.get('selected_rows') or [])
    )
    if int(baseline_result['selected_count']) != target_count:
        raise RuntimeError('C48 Pareto exact無法取得同參數baseline K可行incumbent')

    standalone = []
    for row in candidates:
        _single_order, single_result = evaluate_basket([row])
        if int(single_result['selected_count']) != 1:
            standalone.append({
                'orderable': False,
                'reserve_upper_milli': 0,
                'coverage_upper': 0,
                'score_upper': 0.0,
            })
            continue
        score = _candidate_continuous_score(row)
        standalone.append({
            'orderable': True,
            'reserve_upper_milli': int(single_result['reserved_cost_milli']),
            'coverage_upper': int(score is not None),
            'score_upper': max(0.0, 0.0 if score is None else float(score)),
        })

    n = len(candidates)
    suffix_orderable = [0] * (n + 1)
    suffix_coverage = [0] * (n + 1)
    for idx in range(n - 1, -1, -1):
        suffix_orderable[idx] = suffix_orderable[idx + 1] + int(
            standalone[idx]['orderable']
        )
        suffix_coverage[idx] = suffix_coverage[idx + 1] + int(
            standalone[idx]['coverage_upper']
        )

    def optimistic_quality_upper(idx, need, current_quality):
        if need <= 0:
            return float(current_quality)
        values = sorted(
            (
                float(standalone[pos]['score_upper'])
                for pos in range(idx, n)
                if standalone[pos]['orderable']
            ),
            reverse=True,
        )
        return float(current_quality) + float(sum(values[:need]))

    def optimistic_capital_upper(idx, need, current_capital_milli):
        if need <= 0:
            return int(current_capital_milli)
        values = sorted(
            (
                int(standalone[pos]['reserve_upper_milli'])
                for pos in range(idx, n)
                if standalone[pos]['orderable']
            ),
            reverse=True,
        )
        return int(current_capital_milli) + int(sum(values[:need]))

    empty_result = _simulate_reserved_candidate_order(
        [],
        available_cash=available_cash,
        sizing_equity=sizing_equity,
        free_slots=target_count,
        params=params,
    )

    def run_search(mode, *, normalization=None, seed_results=()):
        if mode not in {'score', 'capital', 'product'}:
            raise ValueError(f'C48不支援search mode={mode!r}')

        def key_for(result):
            coverage, quality, capital_milli, stable = _score_capital_pareto_summary(
                result, base_rank=base_rank
            )
            if mode == 'score':
                return (coverage, quality, capital_milli, stable)
            if mode == 'capital':
                return (coverage, capital_milli, quality, stable)
            return _score_capital_pareto_basket_quality_key(
                result,
                base_rank=base_rank,
                quality_min=normalization['quality_min'],
                quality_max=normalization['quality_max'],
                capital_min_milli=normalization['capital_min_milli'],
                capital_max_milli=normalization['capital_max_milli'],
            )

        initial = [baseline_result, *list(seed_results or [])]
        feasible_initial = [
            result
            for result in initial
            if int(result.get('selected_count', 0) or 0) == target_count
        ]
        if not feasible_initial:
            raise RuntimeError('C48 exact search缺少K可行初始解')
        best_result = max(feasible_initial, key=key_for)
        best_basket = list(best_result.get('selected_rows') or [])
        best_key = key_for(best_result)
        search_states = 0
        pruned_states = 0
        feasible_baskets = 0

        def visit(idx, basket, result):
            nonlocal best_result, best_basket, best_key
            nonlocal search_states, pruned_states, feasible_baskets
            search_states += 1
            coverage, quality, capital_milli, _stable = _score_capital_pareto_summary(
                result, base_rank=base_rank
            )
            selected_count = int(result['selected_count'])
            need = int(target_count - selected_count)

            if need == 0:
                feasible_baskets += 1
                candidate_key = key_for(result)
                if candidate_key > best_key:
                    best_key = candidate_key
                    best_result = result
                    best_basket = list(basket)
                return

            if idx >= n or suffix_orderable[idx] < need:
                pruned_states += 1
                return

            max_coverage = int(coverage) + min(int(need), int(suffix_coverage[idx]))
            if max_coverage < int(best_key[0]):
                pruned_states += 1
                return

            quality_upper = optimistic_quality_upper(idx, need, quality)
            capital_upper = optimistic_capital_upper(idx, need, capital_milli)
            if max_coverage == int(best_key[0]):
                if mode == 'score':
                    if quality_upper < float(best_key[1]) - 1e-12:
                        pruned_states += 1
                        return
                    if (
                        math.isclose(quality_upper, float(best_key[1]), rel_tol=0.0, abs_tol=1e-12)
                        and capital_upper < int(best_key[2])
                    ):
                        pruned_states += 1
                        return
                elif mode == 'capital':
                    if capital_upper < int(best_key[1]):
                        pruned_states += 1
                        return
                    if (
                        capital_upper == int(best_key[1])
                        and quality_upper < float(best_key[2]) - 1e-12
                    ):
                        pruned_states += 1
                        return
                else:
                    quality_norm_upper = _normalize_pareto_axis(
                        quality_upper,
                        low=normalization['quality_min'],
                        high=normalization['quality_max'],
                    )
                    capital_norm_upper = _normalize_pareto_axis(
                        capital_upper,
                        low=normalization['capital_min_milli'],
                        high=normalization['capital_max_milli'],
                    )
                    product_upper = float(quality_norm_upper * capital_norm_upper)
                    if product_upper < float(best_key[1]) - 1e-12:
                        pruned_states += 1
                        return

            row = candidates[idx]
            if standalone[idx]['orderable']:
                include_basket = list(basket) + [row]
                include_order, include_result = evaluate_basket(include_basket)
                if int(include_result['selected_count']) == selected_count + 1:
                    visit(idx + 1, include_order, include_result)
            visit(idx + 1, basket, result)

        visit(0, [], empty_result)
        if int(best_result['selected_count']) != target_count:
            raise RuntimeError('C48 exact search未維持同參數baseline K')
        return {
            'basket': list(best_basket),
            'result': best_result,
            'key': best_key,
            'search_states': int(search_states),
            'pruned_states': int(pruned_states),
            'feasible_baskets': int(feasible_baskets),
        }

    score_pass = run_search('score')
    capital_pass = run_search('capital', seed_results=(score_pass['result'],))
    score_coverage, quality_max, capital_min_milli, _ = _score_capital_pareto_summary(
        score_pass['result'], base_rank=base_rank
    )
    capital_coverage, quality_min, capital_max_milli, _ = _score_capital_pareto_summary(
        capital_pass['result'], base_rank=base_rank
    )
    if score_coverage != capital_coverage:
        raise RuntimeError('C48 Pareto兩個端點的最大Score coverage不一致')
    if quality_min > quality_max + 1e-12:
        raise RuntimeError('C48 Pareto quality normalization範圍反向')
    if capital_min_milli > capital_max_milli:
        raise RuntimeError('C48 Pareto capital normalization範圍反向')

    normalization = {
        'quality_min': float(quality_min),
        'quality_max': float(quality_max),
        'capital_min_milli': int(capital_min_milli),
        'capital_max_milli': int(capital_max_milli),
    }
    product_pass = run_search(
        'product',
        normalization=normalization,
        seed_results=(score_pass['result'], capital_pass['result']),
    )
    best_result = product_pass['result']
    best_basket = list(product_pass['basket'])
    final_ordered = _max_dl_execution_order(best_basket, base_rank=base_rank)
    final_result = _simulate_reserved_candidate_order(
        final_ordered,
        available_cash=available_cash,
        sizing_equity=sizing_equity,
        free_slots=target_count,
        params=params,
    )
    final_key = _score_capital_pareto_basket_quality_key(
        final_result,
        base_rank=base_rank,
        quality_min=quality_min,
        quality_max=quality_max,
        capital_min_milli=capital_min_milli,
        capital_max_milli=capital_max_milli,
    )
    if final_key != product_pass['key']:
        raise RuntimeError('C48 final canonical execution改變Pareto optimum')

    selected_ids = {id(row) for row in final_result['selected_rows']}
    baseline_ids = {id(row) for row in baseline['selected_rows']}
    final_order = final_ordered + [row for row in candidates if id(row) not in selected_ids]
    final_coverage, final_quality, final_capital_milli, _ = _score_capital_pareto_summary(
        final_result, base_rank=base_rank
    )
    quality_norm = _normalize_pareto_axis(
        final_quality, low=quality_min, high=quality_max
    )
    capital_norm = _normalize_pareto_axis(
        final_capital_milli, low=capital_min_milli, high=capital_max_milli
    )
    total_states = sum(
        int(item['search_states']) for item in (score_pass, capital_pass, product_pass)
    )
    total_pruned = sum(
        int(item['pruned_states']) for item in (score_pass, capital_pass, product_pass)
    )
    total_feasible = sum(
        int(item['feasible_baskets']) for item in (score_pass, capital_pass, product_pass)
    )
    out = _resource_aware_diag_from_result(
        default_diag,
        baseline,
        final_result,
        changed=bool(selected_ids != baseline_ids),
        promoted_pass_count=0,
        selector=selector_name,
    )
    out.update({
        'mode': 'dl-selection',
        'promoted_score_orders': int(
            sum(1 for row in final_result['selected_rows'] if id(row) not in baseline_ids)
        ),
        'direct_score_order_feasible': False,
        'basket_search_states': int(total_states),
        'basket_search_pruned': int(total_pruned),
        'basket_feasible_count': int(total_feasible),
        'resource_preservation_required': False,
        'pre_market_order_limit': int(target_count),
        'max_dl_eligible': True,
        'max_dl_repair_steps': 0,
        'max_dl_repair_evaluations': 0,
        'max_dl_seed_fallback': False,
        'max_dl_fallback_to_baseline': False,
        'max_dl_feasible_ascent_steps': 0,
        'max_dl_feasible_ascent_evaluations': 0,
        'max_dl_feasible_ascent_local_optimum': False,
        'basket_objective': 'score_capital_pareto',
        'constrained_solver_optimality_certified': True,
        'constrained_solver_search_states': int(total_states),
        'constrained_solver_pruned_states': int(total_pruned),
        'constrained_solver_feasible_baskets': int(total_feasible),
        'constrained_solver_seed_source': 'baseline-plus-exact-pareto-extremes',
        'constrained_solver_seed_repair_evaluations': 0,
        'constrained_solver_seed_ascent_evaluations': 0,
        'pareto_selection_method': 'normalized_product_v1',
        'pareto_max_score_coverage': int(score_coverage),
        'pareto_quality_min': float(quality_min),
        'pareto_quality_max': float(quality_max),
        'pareto_capital_min_milli': int(capital_min_milli),
        'pareto_capital_max_milli': int(capital_max_milli),
        'pareto_selected_quality': float(final_quality),
        'pareto_selected_capital_milli': int(final_capital_milli),
        'pareto_selected_quality_norm': float(quality_norm),
        'pareto_selected_capital_norm': float(capital_norm),
        'pareto_selected_product': float(quality_norm * capital_norm),
        'pareto_quality_axis_degenerate': bool(quality_max <= quality_min + 1e-12),
        'pareto_capital_axis_degenerate': bool(capital_max_milli <= capital_min_milli),
        'pareto_score_pass_states': int(score_pass['search_states']),
        'pareto_capital_pass_states': int(capital_pass['search_states']),
        'pareto_product_pass_states': int(product_pass['search_states']),
        '_selector_trace_baskets': {
            'raw_top_n': [],
            'minimum_repair_seed': [],
            'feasible_ascent_final': [],
            'constrained_optimal_final': list(final_result.get('selected_rows') or []),
        },
        '_selector_repair_steps': [],
    })
    if int(final_result['selected_count']) != target_count:
        raise RuntimeError('C48 Pareto exact未維持同參數baseline K')
    return final_order, out


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
    objective_mode='score',
    preserve_reserve_floor=True,
    minimum_repair_enabled=True,
    allow_count_growth=False,
    matched_score_proposal=False,
):
    """Strengthen one feasible seed with deterministic single-add/swap local search.

    The default fixed-K mode preserves the historical feasible-ascent contract.
    ``matched_score_proposal`` is opt-in and owned only by the SR-C68/C69 matched
    research pair; historical feasible-ascent callers do not use it.  K-Flex mode
    is the matched local-search treatment: it starts from the same fixed-K feasible
    seed and may accept one additional canonical planned order at a time up to the
    physical free-slot cap.  R0 remains a hard floor.  Count is lexicographically
    primary only in K-Flex mode; within a count, the unchanged frozen score objective
    owns membership quality.  This is deliberately a deterministic local optimum,
    not an exact/global-optimum certificate.
    """

    baseline_k = int(baseline['selected_count'])
    physical_free_slots = max(0, int(free_slots))
    if baseline_k < 0 or baseline_k > physical_free_slots:
        raise RuntimeError('feasible-ascent baseline K超出physical free-slot範圍')
    reserve_floor_milli = (
        int(baseline['reserved_cost_milli']) if preserve_reserve_floor else 0
    )
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
        objective_mode=objective_mode,
        preserve_reserve_floor=preserve_reserve_floor,
        minimum_repair_enabled=minimum_repair_enabled,
    )
    if not bool(seed_diag.get('max_dl_eligible', False)):
        out = dict(seed_diag)
        out.update({
            'selector': (
                'continuous-score-max-dl-feasible-ascent-stale-score-guard'
                if guard_enabled and objective_mode == 'score' and not allow_count_growth
                else 'continuous-score-max-dl-matched-feasible-ascent'
                if matched_score_proposal and objective_mode == 'score' and not allow_count_growth
                else 'continuous-score-max-dl-k-flex-r0-feasible-ascent'
                if allow_count_growth and objective_mode == 'score' and preserve_reserve_floor
                else _objective_selector_name(
                    objective_mode,
                    suffix='feasible-ascent',
                    no_r0=not preserve_reserve_floor,
                )
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
            'max_dl_feasible_ascent_additions': 0,
            'k_flex_r0_preserved': bool(allow_count_growth and preserve_reserve_floor),
            'baseline_k': int(baseline_k),
            'physical_free_slots': int(physical_free_slots),
        })
        return seed_order, out

    def evaluate_basket(basket):
        expected_count = len(basket) if allow_count_growth else baseline_k
        ordered = _max_dl_execution_order(basket, base_rank=base_rank)
        result = _simulate_reserved_candidate_order(
            ordered,
            available_cash=available_cash,
            sizing_equity=sizing_equity,
            free_slots=expected_count,
            params=params,
        )
        return ordered, result, expected_count

    def basket_is_feasible(result_, expected_count):
        return _max_dl_basket_is_feasible(
            result_,
            target_count=expected_count,
            reserve_floor_milli=reserve_floor_milli,
        )

    def quality_key(rows_, result_):
        if objective_mode == 'expected_pnl':
            base_key = _expected_pnl_basket_quality_key(result_, base_rank=base_rank)
        elif objective_mode == 'excess_alpha':
            base_key = _excess_alpha_basket_quality_key(result_, base_rank=base_rank)
        else:
            base_key = _max_dl_basket_quality_key(rows_, base_rank=base_rank)
        if allow_count_growth:
            return (int(result_['selected_count']), *tuple(base_key))
        return base_key

    baseline_ids = {id(row) for row in baseline['selected_rows']}
    seed_basket = list(seed_order[:baseline_k])
    seed_ids = {id(row) for row in seed_basket}
    seed_guard_blocked = bool(
        guard_enabled
        and any(row_id in stale_ids for row_id in (baseline_ids ^ seed_ids))
    )
    current_basket = (
        list(baseline['selected_rows']) if seed_guard_blocked else seed_basket
    )
    current_order, current_result, current_expected_count = evaluate_basket(current_basket)
    if not basket_is_feasible(current_result, current_expected_count):
        raise RuntimeError('max-DL feasible-ascent seed不符合同參數DL-off baseline資源契約')
    current_key = quality_key(current_order, current_result)

    direct_proposal_feasible = False
    if matched_score_proposal:
        # Matched deterministic score-priority proposal.  Score order is used only to
        # propose membership; accepted membership is always re-executed in canonical
        # baseline order before feasibility/quality comparison.  C68/C69 therefore
        # share the same proposal algorithm and differ only in the count cap.
        proposal_cap = physical_free_slots if allow_count_growth else baseline_k
        score_proposal_order = _max_dl_score_order(rows, base_rank=base_rank)
        score_proposal_result = _simulate_reserved_candidate_order(
            score_proposal_order,
            available_cash=available_cash,
            sizing_equity=sizing_equity,
            free_slots=proposal_cap,
            params=params,
        )
        proposed_membership = list(score_proposal_result.get('selected_rows') or [])
        if len(proposed_membership) >= baseline_k:
            proposal_order, proposal_result, proposal_expected_count = evaluate_basket(proposed_membership)
            if basket_is_feasible(proposal_result, proposal_expected_count):
                proposal_key = quality_key(proposal_order, proposal_result)
                direct_proposal_feasible = True
                if proposal_key > current_key:
                    current_basket = list(proposed_membership)
                    current_order = proposal_order
                    current_result = proposal_result
                    current_key = proposal_key

    evaluations = 0
    blocked_swaps = 0
    ascent_steps = 0
    addition_steps = 0
    local_optimum = False

    while True:
        current_ids = {id(row) for row in current_basket}
        in_rows = [row for row in rows if id(row) not in current_ids]
        best = None
        best_key = current_key
        best_tie = None

        # K-Flex treatment: count is primary, so test every legal one-position
        # expansion before same-count swaps.  Each proposal is re-executed in
        # canonical baseline order; no score-order execution semantics leak in.
        if allow_count_growth and len(current_basket) < physical_free_slots:
            for in_row in in_rows:
                trial_basket = list(current_basket) + [in_row]
                trial_order, trial_result, expected_count = evaluate_basket(trial_basket)
                evaluations += 1
                if not basket_is_feasible(trial_result, expected_count):
                    continue
                quality_key_value = quality_key(trial_order, trial_result)
                if quality_key_value <= current_key:
                    continue
                if guard_enabled and id(in_row) in stale_ids:
                    blocked_swaps += 1
                    continue
                tie_key = (1, -base_rank[id(in_row)], 0)
                if (
                    best is None
                    or quality_key_value > best_key
                    or (quality_key_value == best_key and tie_key > best_tie)
                ):
                    best = ('add', trial_basket, trial_order, trial_result)
                    best_key = quality_key_value
                    best_tie = tie_key

        for out_row in list(current_basket):
            for in_row in in_rows:
                trial_basket = [
                    row for row in current_basket if id(row) != id(out_row)
                ] + [in_row]
                trial_order, trial_result, expected_count = evaluate_basket(trial_basket)
                evaluations += 1
                if not basket_is_feasible(trial_result, expected_count):
                    continue
                quality_key_value = quality_key(trial_order, trial_result)
                if quality_key_value <= current_key:
                    continue
                if guard_enabled and (id(out_row) in stale_ids or id(in_row) in stale_ids):
                    blocked_swaps += 1
                    continue
                tie_key = (0, -base_rank[id(out_row)], -base_rank[id(in_row)])
                if (
                    best is None
                    or quality_key_value > best_key
                    or (quality_key_value == best_key and tie_key > best_tie)
                ):
                    best = ('swap', trial_basket, trial_order, trial_result)
                    best_key = quality_key_value
                    best_tie = tie_key
        if best is None:
            local_optimum = True
            break
        operation, current_basket, current_order, current_result = best
        current_key = best_key
        ascent_steps += 1
        if operation == 'add':
            addition_steps += 1

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
    selector_name = (
        'continuous-score-max-dl-feasible-ascent-stale-score-guard'
        if guard_enabled and objective_mode == 'score' and not allow_count_growth
        else 'continuous-score-max-dl-matched-feasible-ascent'
        if matched_score_proposal and objective_mode == 'score' and not allow_count_growth
        else 'continuous-score-max-dl-k-flex-r0-feasible-ascent'
        if allow_count_growth and objective_mode == 'score' and preserve_reserve_floor
        else _objective_selector_name(
            objective_mode,
            suffix='feasible-ascent',
            no_r0=not preserve_reserve_floor,
        )
    )
    out = _resource_aware_diag_from_result(
        default_diag,
        baseline,
        current_result,
        changed=bool(selected_ids != baseline_selected_ids),
        promoted_pass_count=0,
        selector=selector_name,
    )
    seed_trace = dict(seed_diag.get('_selector_trace_baskets') or {})
    final_count = int(current_result['selected_count'])
    out.update({
        'mode': 'dl-selection',
        'promoted_score_orders': int(promoted_count),
        'direct_score_order_feasible': bool(direct_proposal_feasible or seed_diag.get('direct_score_order_feasible', False)),
        'basket_search_states': int(seed_diag.get('basket_search_states', 0) or 0) + int(evaluations),
        'basket_search_pruned': 0,
        'basket_feasible_count': int(seed_diag.get('basket_feasible_count', 0) or 0),
        'resource_preservation_required': bool(preserve_reserve_floor),
        'pre_market_order_limit': int(final_count if allow_count_growth else baseline_k),
        'max_dl_eligible': True,
        'max_dl_repair_steps': int(seed_diag.get('max_dl_repair_steps', 0) or 0),
        'max_dl_repair_evaluations': int(seed_diag.get('max_dl_repair_evaluations', 0) or 0),
        'max_dl_seed_fallback': bool(seed_diag.get('max_dl_fallback_to_baseline', False)),
        'max_dl_fallback_to_baseline': False,
        'max_dl_feasible_ascent_steps': int(ascent_steps),
        'max_dl_feasible_ascent_additions': int(addition_steps),
        'max_dl_feasible_ascent_evaluations': int(evaluations),
        'max_dl_feasible_ascent_local_optimum': bool(local_optimum),
        'basket_objective': str(objective_mode),
        'stale_score_membership_guard_enabled': bool(guard_enabled),
        'stale_score_membership_guard_max_age_days': guard_max_age,
        'stale_score_candidate_count': int(len(stale_ids)),
        'stale_score_guard_triggered': bool(seed_guard_blocked or blocked_swaps > 0),
        'stale_score_guard_seed_blocked': bool(seed_guard_blocked),
        'stale_score_guard_blocked_swaps': int(blocked_swaps),
        'k_flex_r0_preserved': bool(allow_count_growth and preserve_reserve_floor),
        'count_constraint': (
            'baseline_k_to_physical_free_slots_local_v1'
            if allow_count_growth
            else 'baseline_k_exact_local_v1'
        ),
        'baseline_k': int(baseline_k),
        'physical_free_slots': int(physical_free_slots),
        'k_flex_extra_positions': int(max(0, final_count - baseline_k)),
        'constrained_solver_optimality_certified': False,
        '_selector_trace_baskets': {
            'raw_top_n': list(seed_trace.get('raw_top_n') or []),
            'minimum_repair_seed': list(seed_trace.get('minimum_repair_seed') or seed_basket),
            'feasible_ascent_final': list(current_result.get('selected_rows') or []),
        },
        '_selector_repair_steps': list(seed_diag.get('_selector_repair_steps') or []),
    })
    if allow_count_growth:
        if not (baseline_k <= final_count <= physical_free_slots):
            raise RuntimeError('K-Flex feasible-ascent輸出違反baseline K至physical free-slot範圍')
    elif final_count != baseline_k:
        raise RuntimeError('max-DL feasible-ascent輸出未維持同參數DL-off baseline預留單數')
    if preserve_reserve_floor and int(current_result['reserved_cost_milli']) < reserve_floor_milli:
        raise RuntimeError('max-DL feasible-ascent輸出低於同參數DL-off baseline reserved-capital floor')
    return final_order, out
