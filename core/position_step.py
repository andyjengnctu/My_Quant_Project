import pandas as pd

from core.exact_accounting import (
    allocate_cost_basis_milli,
    build_sell_ledger_from_price,
    calc_average_price_from_total_milli,
    milli_to_money,
    price_to_milli,
    sync_position_display_fields,
)
from core.price_utils import (
    adjust_long_sell_fill_price,
    adjust_long_stop_price,
    calc_half_take_profit_sell_qty,
    get_exit_sell_block_reason,
)


def _reset_exec_contexts(position):
    position['_last_exec_contexts'] = []


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
    allocated_cost_milli=0,
    pnl_milli=0,
):
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
            'allocated_cost_milli': int(allocated_cost_milli),
            'pnl_milli': int(pnl_milli),
        }
    )


def sum_last_exec_contexts_milli(position):
    contexts = position.get('_last_exec_contexts', [])
    freed_cash_milli = sum(int(ctx.get('net_total_milli', 0)) for ctx in contexts)
    pnl_realized_milli = sum(int(ctx.get('pnl_milli', 0)) for ctx in contexts)
    return freed_cash_milli, pnl_realized_milli


def _execute_sell_leg(position, *, event, exec_price, sell_qty, params, deferred=False, trigger_price=None, trade_date=None):
    sell_ledger = build_sell_ledger_from_price(
        exec_price,
        sell_qty,
        params,
        ticker=position.get('ticker'),
        security_profile=position.get('security_profile'),
        trade_date=trade_date,
    )
    allocated_cost_milli = allocate_cost_basis_milli(position['remaining_cost_basis_milli'], position['qty'], sell_qty)
    freed_cash_milli = sell_ledger['net_sell_total_milli']
    pnl_milli = freed_cash_milli - allocated_cost_milli

    position['realized_pnl_milli'] += pnl_milli
    position['remaining_cost_basis_milli'] -= allocated_cost_milli
    position['qty'] -= sell_qty
    if position['qty'] <= 0:
        position['qty'] = 0
        position['remaining_cost_basis_milli'] = 0
    sync_position_display_fields(position)

    avg_net_price = calc_average_price_from_total_milli(freed_cash_milli, sell_qty)
    _record_exec_context(
        position,
        event=event,
        exec_price=exec_price,
        net_price=avg_net_price,
        qty=sell_qty,
        pnl=milli_to_money(pnl_milli),
        deferred=deferred,
        trigger_price=trigger_price,
        net_total_milli=freed_cash_milli,
        allocated_cost_milli=allocated_cost_milli,
        pnl_milli=pnl_milli,
    )
    return freed_cash_milli, pnl_milli


def _update_trailing_stop(position, *, y_high, y_atr, params):
    if pd.isna(y_atr):
        return

    highest_high_milli = int(position.get('highest_high_since_entry_milli', position['entry_fill_price_milli']))
    if not pd.isna(y_high):
        highest_high_milli = max(highest_high_milli, price_to_milli(y_high))
    position['highest_high_since_entry_milli'] = highest_high_milli
    position['highest_high_since_entry'] = milli_to_money(highest_high_milli)

    trail_reference = milli_to_money(highest_high_milli)
    candidate_trail = adjust_long_stop_price(
        trail_reference - (y_atr * params.atr_times_trail),
        ticker=position.get('ticker'),
    )
    candidate_trail_milli = price_to_milli(candidate_trail)
    position['trailing_stop_milli'] = max(position.get('trailing_stop_milli', 0), candidate_trail_milli)
    position['sl_milli'] = max(position['initial_stop_milli'], position['trailing_stop_milli'])
    position['trailing_stop'] = milli_to_money(position['trailing_stop_milli'])
    position['sl'] = milli_to_money(position['sl_milli'])


def execute_bar_step(position, y_atr, y_ind_sell, y_close, t_open, t_high, t_low, t_close, t_volume, params, current_date=None, y_high=None):
    freed_cash_milli, pnl_realized_milli = 0, 0
    events = []
    _reset_exec_contexts(position)

    if position['qty'] <= 0:
        return position, 0.0, 0.0, events

    _update_trailing_stop(position, y_high=y_high, y_atr=y_atr, params=params)

    if y_ind_sell:
        sell_block_reason = get_exit_sell_block_reason(t_open, t_high, t_low, t_close, t_volume, y_close, ticker=position.get("ticker"))
        if sell_block_reason is None:
            exec_price = adjust_long_sell_fill_price(t_open, ticker=position.get("ticker"))
            leg_freed_cash_milli, leg_pnl_milli = _execute_sell_leg(
                position,
                event='IND_SELL',
                exec_price=exec_price,
                sell_qty=position['qty'],
                params=params,
                deferred=False,
                trade_date=current_date,
            )
            freed_cash_milli += leg_freed_cash_milli
            pnl_realized_milli += leg_pnl_milli
            events.append('IND_SELL')
        else:
            events.extend(['MISSED_SELL', sell_block_reason])

        return position, milli_to_money(freed_cash_milli), milli_to_money(pnl_realized_milli), events

    is_stop_hit = price_to_milli(t_low) <= position['sl_milli']
    half_sell_qty = calc_half_take_profit_sell_qty(position['qty'], params.tp_percent)
    is_tp_hit = price_to_milli(t_high) >= position['tp_half_milli'] and not position['sold_half'] and half_sell_qty > 0

    if is_stop_hit and is_tp_hit:
        is_tp_hit = False

    if is_tp_hit and not (pd.isna(t_volume) or t_volume <= 0):
        exec_price = adjust_long_sell_fill_price(max(position['tp_half'], t_open), ticker=position.get('ticker'))
        leg_freed_cash_milli, leg_pnl_milli = _execute_sell_leg(
            position,
            event='TP_HALF',
            exec_price=exec_price,
            sell_qty=half_sell_qty,
            params=params,
            deferred=False,
            trigger_price=position['tp_half'],
            trade_date=current_date,
        )
        freed_cash_milli += leg_freed_cash_milli
        pnl_realized_milli += leg_pnl_milli
        position['sold_half'] = True
        events.append('TP_HALF')

    if is_stop_hit and position['qty'] > 0:
        sell_block_reason = get_exit_sell_block_reason(t_open, t_high, t_low, t_close, t_volume, y_close, ticker=position.get("ticker"))
        if sell_block_reason is None:
            exec_price = adjust_long_sell_fill_price(min(position['sl'], t_open), ticker=position.get('ticker'))
            leg_freed_cash_milli, leg_pnl_milli = _execute_sell_leg(
                position,
                event='STOP',
                exec_price=exec_price,
                sell_qty=position['qty'],
                params=params,
                deferred=False,
                trigger_price=position['sl'],
                trade_date=current_date,
            )
            freed_cash_milli += leg_freed_cash_milli
            pnl_realized_milli += leg_pnl_milli
            events.append('STOP')
        else:
            events.extend(['MISSED_SELL', sell_block_reason])

    return position, milli_to_money(freed_cash_milli), milli_to_money(pnl_realized_milli), events
