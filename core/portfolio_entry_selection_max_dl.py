from datetime import date, datetime
import math

from core.exact_accounting import calc_planned_initial_risk_from_prices_milli

from core.portfolio_entry_selection_common import (
    _candidate_continuous_score,
    _resource_aware_diag_from_result,
    _simulate_reserved_candidate_order,
)


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
    """Exactly maximize frozen Excess-Alpha under the existing K/R0 resource contract.

    C39's repair/ascent path is used only to obtain a strong feasible incumbent.  It
    never limits the search space.  Branch-and-bound then visits the complete
    candidate membership decision tree in the canonical baseline execution order.
    Every included state is rebuilt by the canonical cash-capped reservation
    simulator, so planned risk and reserved capital keep the production sizing/cash
    semantics.  Pruning uses only admissible optimistic bounds: remaining candidate
    count, standalone maximum reserve, and standalone positive Excess-Alpha value.
    There is no candidate truncation, beam width, time budget, lambda, or tuned
    tolerance.  Returning from this function therefore means the final feasible
    basket is globally optimal under the exact lexicographic C39 Excess-Alpha key.
    """

    candidates = list(rows or [])
    target_count = int(baseline['selected_count'])
    reserve_floor_milli = int(baseline['reserved_cost_milli'])
    base_rank = {id(row): idx for idx, row in enumerate(candidates)}

    seed_order, seed_diag = _reorder_resource_aware_continuous_max_dl_feasible_ascent(
        candidates,
        available_cash=available_cash,
        sizing_equity=sizing_equity,
        free_slots=free_slots,
        params=params,
        baseline=baseline,
        default_diag=default_diag,
        objective_mode='excess_alpha',
    )
    if not bool(seed_diag.get('max_dl_eligible', False)):
        out = dict(seed_diag)
        out.update({
            'selector': 'continuous-excess-alpha-constrained-optimal',
            'constrained_solver_optimality_certified': True,
            'constrained_solver_search_states': 0,
            'constrained_solver_pruned_states': 0,
            'constrained_solver_feasible_baskets': 0,
            'constrained_solver_seed_source': 'c39-feasible-ascent',
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

    seed_basket = list(seed_order[:target_count])
    incumbent_order, incumbent_result = evaluate_basket(seed_basket)
    if not _max_dl_basket_is_feasible(
        incumbent_result,
        target_count=target_count,
        reserve_floor_milli=reserve_floor_milli,
    ):
        raise RuntimeError('C41 constrained solver無法取得C39可行incumbent')
    best_basket = list(incumbent_order)
    best_result = incumbent_result
    best_key = _excess_alpha_basket_quality_key(best_result, base_rank=base_rank)

    standalone = []
    for row in candidates:
        single_order, single_result = evaluate_basket([row])
        if int(single_result['selected_count']) != 1:
            standalone.append({
                'orderable': False,
                'reserve_upper_milli': 0,
                'coverage_upper': 0,
                'alpha_upper': 0.0,
            })
            continue
        plan = single_result['selected_plans'][0]
        expected_excess_r = _candidate_expected_excess_r(row)
        risk_milli = _planned_initial_risk_milli(row=row, plan=plan)
        coverage = int(expected_excess_r is not None and risk_milli is not None)
        raw_value = (
            0.0
            if not coverage
            else float(expected_excess_r) * float(risk_milli)
        )
        standalone.append({
            'orderable': True,
            'reserve_upper_milli': int(single_result['reserved_cost_milli']),
            'coverage_upper': coverage,
            # With less remaining cash, planned risk cannot exceed the standalone
            # full-cash plan.  For negative excess-R, zero is the admissible upper
            # bound because cash-capping can only shrink the negative magnitude.
            'alpha_upper': max(0.0, float(raw_value)),
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

    search_states = 0
    pruned_states = 0
    feasible_baskets = 1

    def current_objective_summary(result):
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
        values = sorted(
            (
                int(standalone[pos]['reserve_upper_milli'])
                for pos in range(idx, n)
                if standalone[pos]['orderable']
            ),
            reverse=True,
        )
        return int(current_reserved) + int(sum(values[:need]))

    def optimistic_alpha_upper(idx, need, current_total):
        if need <= 0:
            return float(current_total)
        values = sorted(
            (
                float(standalone[pos]['alpha_upper'])
                for pos in range(idx, n)
                if standalone[pos]['orderable']
            ),
            reverse=True,
        )
        return float(current_total) + float(sum(values[:need]))

    empty_result = _simulate_reserved_candidate_order(
        [],
        available_cash=available_cash,
        sizing_equity=sizing_equity,
        free_slots=target_count,
        params=params,
    )

    def visit(idx, basket, result, coverage, alpha_total):
        nonlocal best_basket, best_result, best_key
        nonlocal search_states, pruned_states, feasible_baskets
        search_states += 1
        selected_count = int(result['selected_count'])
        need = int(target_count - selected_count)

        if need == 0:
            if int(result['reserved_cost_milli']) < reserve_floor_milli:
                pruned_states += 1
                return
            feasible_baskets += 1
            quality = _excess_alpha_basket_quality_key(result, base_rank=base_rank)
            if quality > best_key:
                best_key = quality
                best_basket = list(basket)
                best_result = result
            return

        if idx >= n or suffix_orderable[idx] < need:
            pruned_states += 1
            return
        if optimistic_reserve_upper(idx, need, result['reserved_cost_milli']) < reserve_floor_milli:
            pruned_states += 1
            return

        max_coverage = int(coverage) + min(int(need), int(suffix_coverage[idx]))
        if max_coverage < int(best_key[0]):
            pruned_states += 1
            return
        if max_coverage == int(best_key[0]):
            alpha_upper = optimistic_alpha_upper(idx, need, alpha_total)
            if alpha_upper < float(best_key[1]):
                pruned_states += 1
                return

        row = candidates[idx]

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

        # Excluding a row is always part of the complete membership decision tree.
        visit(idx + 1, basket, result, coverage, alpha_total)

    visit(0, [], empty_result, 0, 0.0)

    if not _max_dl_basket_is_feasible(
        best_result,
        target_count=target_count,
        reserve_floor_milli=reserve_floor_milli,
    ):
        raise RuntimeError('C41 constrained solver結束後沒有合法K/R0 basket')

    final_order = _max_dl_execution_order(best_basket, base_rank=base_rank)
    final_ids = {id(row) for row in final_order}
    final_order.extend(row for row in candidates if id(row) not in final_ids)
    baseline_ids = {id(row) for row in baseline['selected_rows']}
    selected_ids = {id(row) for row in best_result['selected_rows']}
    promoted_count = sum(
        1 for row in best_result['selected_rows'] if id(row) not in baseline_ids
    )
    seed_trace = dict(seed_diag.get('_selector_trace_baskets') or {})
    out = _resource_aware_diag_from_result(
        default_diag,
        baseline,
        best_result,
        changed=bool(selected_ids != baseline_ids),
        promoted_pass_count=0,
        selector='continuous-excess-alpha-constrained-optimal',
    )
    out.update({
        'mode': 'dl-selection',
        'promoted_score_orders': int(promoted_count),
        'direct_score_order_feasible': bool(seed_diag.get('direct_score_order_feasible', False)),
        'basket_search_states': int(search_states),
        'basket_search_pruned': int(pruned_states),
        'basket_feasible_count': int(feasible_baskets),
        'resource_preservation_required': True,
        'pre_market_order_limit': int(target_count),
        'max_dl_eligible': True,
        # C39 repair/ascent is only an incumbent producer, not C41's selection path.
        'max_dl_repair_steps': 0,
        'max_dl_repair_evaluations': 0,
        'max_dl_seed_fallback': False,
        'max_dl_fallback_to_baseline': False,
        'max_dl_feasible_ascent_steps': 0,
        'max_dl_feasible_ascent_evaluations': 0,
        'max_dl_feasible_ascent_local_optimum': False,
        'basket_objective': 'excess_alpha',
        'constrained_solver_optimality_certified': True,
        'constrained_solver_search_states': int(search_states),
        'constrained_solver_pruned_states': int(pruned_states),
        'constrained_solver_feasible_baskets': int(feasible_baskets),
        'constrained_solver_seed_source': 'c39-feasible-ascent',
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
        raise RuntimeError('C41 constrained solver未維持同參數baseline K')
    if int(best_result['reserved_cost_milli']) < reserve_floor_milli:
        raise RuntimeError('C41 constrained solver輸出低於同參數baseline R0')
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
):
    """Strengthen one feasible seed with feasible best-improvement swaps, without exact combinatorial search.

    C17 first supplies a guaranteed-feasible K-basket.  If its Top-K repair had to
    fall back to the same-parameter DL-off baseline, that basket is still a valid seed rather than a terminal
    failure.  From the seed, every single membership swap is evaluated with canonical
    exact reservation; the highest-DL-quality feasible improvement is accepted and the
    process repeats until no improving single swap remains.  Capital never contributes
    to the objective: it is only the K/R0 feasibility contract.
    """

    target_count = int(baseline['selected_count'])
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
                if guard_enabled and objective_mode == 'score'
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

    def quality_key(rows_, result_):
        if objective_mode == 'expected_pnl':
            return _expected_pnl_basket_quality_key(result_, base_rank=base_rank)
        if objective_mode == 'excess_alpha':
            return _excess_alpha_basket_quality_key(result_, base_rank=base_rank)
        return _max_dl_basket_quality_key(rows_, base_rank=base_rank)

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
    current_key = quality_key(current_order, current_result)
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
                quality_key_value = quality_key(trial_order, trial_result)
                if quality_key_value <= current_key:
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
                    or quality_key_value > best_key
                    or (quality_key_value == best_key and tie_key > best_tie)
                ):
                    best = (trial_basket, trial_order, trial_result)
                    best_key = quality_key_value
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
            if guard_enabled and objective_mode == 'score'
            else _objective_selector_name(
                objective_mode,
                suffix='feasible-ascent',
                no_r0=not preserve_reserve_floor,
            )
        ),
    )
    seed_trace = dict(seed_diag.get('_selector_trace_baskets') or {})
    out.update({
        'mode': 'dl-selection',
        'promoted_score_orders': int(promoted_count),
        'direct_score_order_feasible': bool(seed_diag.get('direct_score_order_feasible', False)),
        'basket_search_states': int(seed_diag.get('basket_search_states', 0) or 0) + int(evaluations),
        'basket_search_pruned': 0,
        'basket_feasible_count': int(seed_diag.get('basket_feasible_count', 0) or 0),
        'resource_preservation_required': bool(preserve_reserve_floor),
        'pre_market_order_limit': int(target_count),
        'max_dl_eligible': True,
        'max_dl_repair_steps': int(seed_diag.get('max_dl_repair_steps', 0) or 0),
        'max_dl_repair_evaluations': int(seed_diag.get('max_dl_repair_evaluations', 0) or 0),
        'max_dl_seed_fallback': bool(seed_diag.get('max_dl_fallback_to_baseline', False)),
        'max_dl_fallback_to_baseline': False,
        'max_dl_feasible_ascent_steps': int(ascent_steps),
        'max_dl_feasible_ascent_evaluations': int(evaluations),
        'max_dl_feasible_ascent_local_optimum': bool(local_optimum),
        'basket_objective': str(objective_mode),
        'stale_score_membership_guard_enabled': bool(guard_enabled),
        'stale_score_membership_guard_max_age_days': guard_max_age,
        'stale_score_candidate_count': int(len(stale_ids)),
        'stale_score_guard_triggered': bool(seed_guard_blocked or blocked_swaps > 0),
        'stale_score_guard_seed_blocked': bool(seed_guard_blocked),
        'stale_score_guard_blocked_swaps': int(blocked_swaps),
        '_selector_trace_baskets': {
            'raw_top_n': list(seed_trace.get('raw_top_n') or []),
            'minimum_repair_seed': list(seed_trace.get('minimum_repair_seed') or seed_basket),
            'feasible_ascent_final': list(current_result.get('selected_rows') or []),
        },
        '_selector_repair_steps': list(seed_diag.get('_selector_repair_steps') or []),
    })
    if int(current_result['selected_count']) != target_count:
        raise RuntimeError('max-DL feasible-ascent輸出未維持同參數DL-off baseline預留單數')
    if preserve_reserve_floor and int(current_result['reserved_cost_milli']) < reserve_floor_milli:
        raise RuntimeError('max-DL feasible-ascent輸出低於同參數DL-off baseline reserved-capital floor')
    return final_order, out


