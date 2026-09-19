import pandas as pd

from core.fee_rebate import (
    SETTLEMENT_BASIS_BROKER_CASH,
    SETTLEMENT_BASIS_LEDGER_NET,
    accrue_fee_rebate,
    validate_fee_rebate_settlement_basis,
)
from core.exact_accounting import (
    allocate_cost_basis_milli,
    build_sell_ledger_from_price,
    calc_average_price_from_total_milli,
    milli_to_money,
    milli_to_price,
    price_to_milli,
    sync_position_display_fields,
)



def _reset_exec_contexts(position, *, enabled=True):
    if enabled:
        position['_last_exec_contexts'] = []
    else:
        position.pop('_last_exec_contexts', None)


def _record_exec_context(
    position,
    *,
    event,
    exec_price,
    net_price,
    qty,
    pnl,
    deferred=False,
    trigger_price=None,
    net_total_milli=0,
    cash_total_milli=None,
    fee_rebate_receivable_milli=0,
    allocated_cost_milli=0,
    pnl_milli=0,
    enabled=True,
):
    if not enabled:
        return
    position.setdefault('_last_exec_contexts', []).append(
        {
            'event': event,
            'exec_price': float(exec_price),
            'net_price': float(net_price),
            'qty': int(qty),
            'pnl': float(pnl),
            'deferred': bool(deferred),
            'trigger_price': None if pd.isna(trigger_price) else float(trigger_price),
            'net_total_milli': int(net_total_milli),
            'cash_total_milli': int(net_total_milli if cash_total_milli is None else cash_total_milli),
            'fee_rebate_receivable_milli': int(fee_rebate_receivable_milli or 0),
            'allocated_cost_milli': int(allocated_cost_milli),
            'pnl_milli': int(pnl_milli),
        }
    )



def first_exec_context(position, event_name):
    for context in position.get("_last_exec_contexts", []):
        if context.get("event") == event_name:
            return context
    return None

def sum_last_exec_contexts_milli(position):
    contexts = position.get('_last_exec_contexts', [])
    freed_cash_milli = sum(int(ctx.get('net_total_milli', 0)) for ctx in contexts)
    pnl_realized_milli = sum(int(ctx.get('pnl_milli', 0)) for ctx in contexts)
    return freed_cash_milli, pnl_realized_milli


def sum_last_exec_context_cash_milli(position):
    contexts = position.get('_last_exec_contexts', [])
    return sum(int(ctx['cash_total_milli']) for ctx in contexts)


def _execute_sell_leg(position, *, event, exec_price, sell_qty, params, deferred=False, trigger_price=None, trade_date=None, record_exec_contexts=True, sync_display_fields=True, fee_rebate_state=None, settlement_basis=None):
    sell_ledger = build_sell_ledger_from_price(
        exec_price,
        sell_qty,
        params,
        ticker=position.get('ticker'),
        security_profile=position.get('security_profile'),
        trade_date=trade_date,
    )
    allocated_cost_milli = allocate_cost_basis_milli(position['remaining_cost_basis_milli'], position['qty'], sell_qty)
    economic_freed_cash_milli = int(sell_ledger['net_sell_total_milli'])
    cash_freed_milli = int(sell_ledger['cash_sell_total_milli'])
    fee_rebate_receivable_milli = int(sell_ledger['sell_fee_rebate_receivable_milli'])
    pnl_milli = economic_freed_cash_milli - allocated_cost_milli
    validate_fee_rebate_settlement_basis(settlement_basis, fee_rebate_state)
    if settlement_basis == SETTLEMENT_BASIS_BROKER_CASH:
        accrue_fee_rebate(fee_rebate_state, fee_rebate_receivable_milli)
    else:
        cash_freed_milli = economic_freed_cash_milli

    position['realized_pnl_milli'] += pnl_milli
    position['remaining_cost_basis_milli'] -= allocated_cost_milli
    position['qty'] -= sell_qty
    if position['qty'] <= 0:
        position['qty'] = 0
        position['remaining_cost_basis_milli'] = 0
    if sync_display_fields:
        sync_position_display_fields(position)

    if record_exec_contexts:
        avg_net_price = calc_average_price_from_total_milli(economic_freed_cash_milli, sell_qty)
        _record_exec_context(
            position,
            event=event,
            exec_price=exec_price,
            net_price=avg_net_price,
            qty=sell_qty,
            pnl=milli_to_money(pnl_milli),
            deferred=deferred,
            trigger_price=trigger_price,
            net_total_milli=economic_freed_cash_milli,
            cash_total_milli=cash_freed_milli,
            fee_rebate_receivable_milli=fee_rebate_receivable_milli,
            allocated_cost_milli=allocated_cost_milli,
            pnl_milli=pnl_milli,
            enabled=True,
        )
    return cash_freed_milli, pnl_milli














def execute_confirmed_position_sell_fill(
    position,
    *,
    exec_price,
    sell_qty,
    params,
    trade_date=None,
    event="MANUAL_CONFIRMED_SELL",
):
    """Apply one externally confirmed sell fill through canonical exact accounting."""
    qty = int(position.get("qty", 0) or 0)
    sell_qty = int(sell_qty)
    if qty <= 0:
        raise ValueError("position 沒有可賣持股")
    if sell_qty <= 0 or sell_qty > qty:
        raise ValueError(f"sell_qty 必須介於 1..{qty}，收到 {sell_qty}")
    return _execute_sell_leg(
        position,
        event=str(event),
        exec_price=exec_price,
        sell_qty=sell_qty,
        params=params,
        deferred=False,
        trade_date=trade_date,
        record_exec_contexts=True,
        sync_display_fields=True,
        settlement_basis=SETTLEMENT_BASIS_LEDGER_NET,
    )


# AI: Compatibility imports, not a second implementation of management rules.
from core.position_management import (
    PositionExitDecision, _update_trailing_stop,
    rollforward_position_management_from_completed_bar,
    resolve_position_intraday_exit_hits, step_position_management,
)


def execute_bar_step(position, y_atr, y_ind_sell, y_close, t_open, t_high, t_low, t_close, t_volume, params, current_date=None, y_high=None, return_milli=False, record_exec_contexts=True, sync_display_fields=True, fee_rebate_state=None, settlement_basis=None):
    """Research execution adapter over the canonical management transition."""
    validate_fee_rebate_settlement_basis(settlement_basis, fee_rebate_state)
    freed_cash_milli, pnl_realized_milli = 0, 0
    _reset_exec_contexts(position, enabled=record_exec_contexts)

    def execute(decision: PositionExitDecision) -> bool:
        nonlocal freed_cash_milli, pnl_realized_milli
        if not decision.executable:
            return False
        cash, pnl = _execute_sell_leg(
            position, event=decision.event, exec_price=decision.reference_price,
            sell_qty=decision.qty, params=params, deferred=decision.deferred,
            trigger_price=decision.trigger_price, trade_date=decision.trade_date,
            record_exec_contexts=record_exec_contexts, sync_display_fields=sync_display_fields,
            fee_rebate_state=fee_rebate_state, settlement_basis=settlement_basis,
        )
        freed_cash_milli += cash
        pnl_realized_milli += pnl
        return True

    events = step_position_management(
        position, y_atr=y_atr, y_ind_sell=y_ind_sell, y_close=y_close,
        t_open=t_open, t_high=t_high, t_low=t_low, t_close=t_close, t_volume=t_volume,
        params=params, on_decision=execute, current_date=current_date, y_high=y_high,
        sync_display_fields=sync_display_fields,
    )
    if return_milli:
        return position, freed_cash_milli, pnl_realized_milli, events
    return position, milli_to_money(freed_cash_milli), milli_to_money(pnl_realized_milli), events
