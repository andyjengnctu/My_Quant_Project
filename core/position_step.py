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


def _clear_pending_exit(position):
    position['pending_exit_action'] = None
    position['pending_exit_trigger_price'] = pd.NA


def _execute_sell_leg(position, *, event, exec_price, sell_qty, params, deferred=False, trigger_price=None):
    sell_ledger = build_sell_ledger_from_price(exec_price, sell_qty, params)
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


def _process_pending_exit_action(position, *, t_open, t_high, t_low, t_close, t_volume, y_close, params, events):
    pending_action = position.get('pending_exit_action')
    resolved_ticker = position.get('ticker')
    if pending_action not in {'STOP', 'TP_HALF'}:
        return False, 0, 0

    sell_block_reason = get_exit_sell_block_reason(t_open, t_high, t_low, t_close, t_volume, y_close, ticker=resolved_ticker)
    if sell_block_reason is not None:
        events.extend(['MISSED_SELL', sell_block_reason])
        return True, 0, 0

    trigger_price = position.get('pending_exit_trigger_price', pd.NA)
    exec_price = adjust_long_sell_fill_price(t_open, ticker=resolved_ticker)
    if pending_action == 'STOP':
        freed_cash_milli, pnl_milli = _execute_sell_leg(
            position,
            event='STOP',
            exec_price=exec_price,
            sell_qty=position['qty'],
            params=params,
            deferred=True,
            trigger_price=trigger_price,
        )
        _clear_pending_exit(position)
        events.extend(['STOP', 'DEFERRED_STOP_ON_OPEN'])
        return True, freed_cash_milli, pnl_milli

    sell_qty = calc_half_take_profit_sell_qty(position['qty'], params.tp_percent)
    if sell_qty <= 0:
        _clear_pending_exit(position)
        return False, 0, 0

    freed_cash_milli, pnl_milli = _execute_sell_leg(
        position,
        event='TP_HALF',
        exec_price=exec_price,
        sell_qty=sell_qty,
        params=params,
        deferred=True,
        trigger_price=trigger_price,
    )
    position['sold_half'] = True
    _clear_pending_exit(position)
    events.extend(['TP_HALF', 'DEFERRED_TP_HALF_ON_OPEN'])
    return False, freed_cash_milli, pnl_milli


def execute_bar_step(position, y_atr, y_ind_sell, y_close, t_open, t_high, t_low, t_close, t_volume, params):
    freed_cash_milli, pnl_realized_milli = 0, 0
    events = []
    _reset_exec_contexts(position)

    if position['qty'] <= 0:
        return position, 0.0, 0.0, events

    should_return, pending_freed_cash_milli, pending_pnl_milli = _process_pending_exit_action(
        position,
        t_open=t_open,
        t_high=t_high,
        t_low=t_low,
        t_close=t_close,
        t_volume=t_volume,
        y_close=y_close,
        params=params,
        events=events,
    )
    freed_cash_milli += pending_freed_cash_milli
    pnl_realized_milli += pending_pnl_milli
    if should_return:
        return position, milli_to_money(freed_cash_milli), milli_to_money(pnl_realized_milli), events
    if position['qty'] <= 0:
        return position, milli_to_money(freed_cash_milli), milli_to_money(pnl_realized_milli), events

    if y_close > position['pure_buy_price'] + (y_atr * params.atr_times_trail):
        new_trail = adjust_long_stop_price(y_close - (y_atr * params.atr_times_trail), ticker=position.get("ticker"))
        new_trail_milli = price_to_milli(new_trail)
        position['trailing_stop_milli'] = max(position.get('trailing_stop_milli', 0), new_trail_milli)
        position['sl_milli'] = max(position['initial_stop_milli'], position['trailing_stop_milli'])
        position['trailing_stop'] = milli_to_money(position['trailing_stop_milli'])
        position['sl'] = milli_to_money(position['sl_milli'])

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
            )
            freed_cash_milli += leg_freed_cash_milli
            pnl_realized_milli += leg_pnl_milli
            events.append('STOP')
        else:
            events.extend(['MISSED_SELL', sell_block_reason])

    return position, milli_to_money(freed_cash_milli), milli_to_money(pnl_realized_milli), events
