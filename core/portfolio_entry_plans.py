from core.exact_accounting import milli_to_money
from core.trade_plans import (
    build_cash_capped_entry_plan,
    clone_shadow_position,
    entry_notional_meets_minimum,
)


def build_candidate_plan_seed(candidate_row, sizing_equity=None, *, copy_shadow_position=True):
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
        plan['shadow_position_state'] = (
            clone_shadow_position(shadow_position_state)
            if copy_shadow_position
            else shadow_position_state
        )
    return plan


def _build_candidate_full_entry_plan_if_affordable(candidate_row, available_cash_milli, params, sizing_equity=None, *, copy_shadow_position=True):
    qty = int(candidate_row.get('qty', 0) or 0)
    reserved_cost_milli = int(candidate_row.get('proj_cost_milli', 0) or 0)
    if qty <= 0 or reserved_cost_milli <= 0 or reserved_cost_milli > int(available_cash_milli):
        return None
    if not entry_notional_meets_minimum(candidate_row.get('limit_px'), qty, params):
        return None

    entry_plan = build_candidate_plan_seed(candidate_row, sizing_equity=sizing_equity, copy_shadow_position=copy_shadow_position)
    entry_plan['qty'] = qty
    entry_plan['is_orderable'] = True
    entry_plan['reserved_cost_milli'] = reserved_cost_milli
    entry_plan['reserved_cost'] = milli_to_money(reserved_cost_milli)
    return entry_plan


def _build_cash_capped_entry_plan_for_candidate(candidate_row, effective_entry_budget, effective_entry_budget_milli, params, sizing_equity, *, copy_shadow_position=True):
    full_entry_plan = _build_candidate_full_entry_plan_if_affordable(
        candidate_row,
        effective_entry_budget_milli,
        params,
        sizing_equity=sizing_equity,
        copy_shadow_position=copy_shadow_position,
    )
    if full_entry_plan is not None:
        return full_entry_plan
    if effective_entry_budget is None:
        effective_entry_budget = milli_to_money(effective_entry_budget_milli)
    return build_cash_capped_entry_plan(
        build_candidate_plan_seed(
            candidate_row,
            sizing_equity=sizing_equity,
            copy_shadow_position=copy_shadow_position,
        ),
        effective_entry_budget,
        params,
    )

def _resolve_training_entry_plan(candidate_row, available_cash_milli, params, sizing_equity):
    """Return the pre-sized candidate directly when its reservation still fits.

    Optimizer training already materialized canonical geometry, quantity and exact
    reserved cash in the candidate row. Rebuilding the same plan dict before every
    attempted order is pure allocation overhead. The rare cash-capped case keeps
    the canonical builder as the fallback.
    """
    qty = int(candidate_row.get('qty', 0) or 0)
    reserved_cost_milli = int(candidate_row.get('proj_cost_milli', 0) or 0)
    if (
        qty > 0
        and reserved_cost_milli > 0
        and reserved_cost_milli <= int(available_cash_milli)
        and entry_notional_meets_minimum(candidate_row.get('limit_px'), qty, params)
    ):
        return candidate_row
    return _build_cash_capped_entry_plan_for_candidate(
        candidate_row,
        None,
        int(available_cash_milli),
        params,
        sizing_equity,
        copy_shadow_position=False,
    )

