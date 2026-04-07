import pandas as pd

from core.price_utils import (
    adjust_long_sell_fill_price,
    adjust_long_stop_price,
    calc_half_take_profit_sell_qty,
    calc_net_sell_price,
    get_exit_sell_block_reason,
)


def _reset_exec_contexts(position):
    position['_last_exec_contexts'] = []


def _record_exec_context(position, *, event, exec_price, net_price, qty, pnl, deferred=False, trigger_price=None):
    position.setdefault('_last_exec_contexts', []).append(
        {
            'event': event,
            'exec_price': float(exec_price),
            'net_price': float(net_price),
            'qty': int(qty),
            'pnl': float(pnl),
            'deferred': bool(deferred),
            'trigger_price': None if pd.isna(trigger_price) else float(trigger_price),
        }
    )


def _clear_pending_exit(position):
    position['pending_exit_action'] = None
    position['pending_exit_trigger_price'] = pd.NA


def _execute_sell_leg(position, *, event, exec_price, sell_qty, params, deferred=False, trigger_price=None):
    net_price = calc_net_sell_price(exec_price, sell_qty, params)
    freed_cash = net_price * sell_qty
    pnl = (net_price - position['entry']) * sell_qty
    position['realized_pnl'] += pnl
    position['qty'] -= sell_qty
    _record_exec_context(
        position,
        event=event,
        exec_price=exec_price,
        net_price=net_price,
        qty=sell_qty,
        pnl=pnl,
        deferred=deferred,
        trigger_price=trigger_price,
    )
    return freed_cash, pnl


def _process_pending_exit_action(position, *, t_open, t_high, t_low, t_close, t_volume, y_close, params, events):
    pending_action = position.get('pending_exit_action')
    if pending_action not in {'STOP', 'TP_HALF'}:
        return False, 0.0, 0.0

    sell_block_reason = get_exit_sell_block_reason(t_open, t_high, t_low, t_close, t_volume, y_close)
    if sell_block_reason is not None:
        events.extend(['MISSED_SELL', sell_block_reason])
        return True, 0.0, 0.0

    trigger_price = position.get('pending_exit_trigger_price', pd.NA)
    exec_price = adjust_long_sell_fill_price(t_open)
    if pending_action == 'STOP':
        freed_cash, pnl = _execute_sell_leg(
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
        return True, freed_cash, pnl

    sell_qty = calc_half_take_profit_sell_qty(position['qty'], params.tp_percent)
    if sell_qty <= 0:
        _clear_pending_exit(position)
        return False, 0.0, 0.0

    freed_cash, pnl = _execute_sell_leg(
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
    return False, freed_cash, pnl


def execute_bar_step(position, y_atr, y_ind_sell, y_close, t_open, t_high, t_low, t_close, t_volume, params):
    freed_cash, pnl_realized = 0.0, 0.0
    events = []
    _reset_exec_contexts(position)

    if position['qty'] <= 0:
        return position, freed_cash, pnl_realized, events

    should_return, pending_freed_cash, pending_pnl = _process_pending_exit_action(
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
    freed_cash += pending_freed_cash
    pnl_realized += pending_pnl
    if should_return:
        return position, freed_cash, pnl_realized, events
    if position['qty'] <= 0:
        return position, freed_cash, pnl_realized, events

    if y_close > position['pure_buy_price'] + (y_atr * params.atr_times_trail):
        new_trail = adjust_long_stop_price(y_close - (y_atr * params.atr_times_trail))
        position['trailing_stop'] = max(position.get('trailing_stop', 0.0), new_trail)
        position['sl'] = max(position['initial_stop'], position['trailing_stop'])

    if y_ind_sell:
        sell_block_reason = get_exit_sell_block_reason(t_open, t_high, t_low, t_close, t_volume, y_close)
        if sell_block_reason is None:
            exec_price = adjust_long_sell_fill_price(t_open)
            leg_freed_cash, leg_pnl = _execute_sell_leg(
                position,
                event='IND_SELL',
                exec_price=exec_price,
                sell_qty=position['qty'],
                params=params,
                deferred=False,
            )
            freed_cash += leg_freed_cash
            pnl_realized += leg_pnl
            events.append('IND_SELL')
        else:
            events.extend(['MISSED_SELL', sell_block_reason])

        return position, freed_cash, pnl_realized, events

    is_stop_hit = t_low <= position['sl']
    half_sell_qty = calc_half_take_profit_sell_qty(position['qty'], params.tp_percent)
    is_tp_hit = t_high >= position['tp_half'] and not position['sold_half'] and half_sell_qty > 0

    if is_stop_hit and is_tp_hit:
        is_tp_hit = False

    if is_tp_hit and not (pd.isna(t_volume) or t_volume <= 0):
        exec_price = adjust_long_sell_fill_price(max(position['tp_half'], t_open))
        leg_freed_cash, leg_pnl = _execute_sell_leg(
            position,
            event='TP_HALF',
            exec_price=exec_price,
            sell_qty=half_sell_qty,
            params=params,
            deferred=False,
            trigger_price=position['tp_half'],
        )
        freed_cash += leg_freed_cash
        pnl_realized += leg_pnl
        position['sold_half'] = True
        events.append('TP_HALF')

    if is_stop_hit and position['qty'] > 0:
        sell_block_reason = get_exit_sell_block_reason(t_open, t_high, t_low, t_close, t_volume, y_close)
        if sell_block_reason is None:
            exec_price = adjust_long_sell_fill_price(min(position['sl'], t_open))
            leg_freed_cash, leg_pnl = _execute_sell_leg(
                position,
                event='STOP',
                exec_price=exec_price,
                sell_qty=position['qty'],
                params=params,
                deferred=False,
                trigger_price=position['sl'],
            )
            freed_cash += leg_freed_cash
            pnl_realized += leg_pnl
            events.append('STOP')
        else:
            events.extend(['MISSED_SELL', sell_block_reason])

    return position, freed_cash, pnl_realized, events
