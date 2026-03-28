import pandas as pd

from core.v16_price_utils import (
    adjust_long_sell_fill_price,
    adjust_long_stop_price,
    calc_half_take_profit_sell_qty,
    calc_net_sell_price,
    get_exit_sell_block_reason,
)


def execute_bar_step(position, y_atr, y_ind_sell, y_close, t_open, t_high, t_low, t_close, t_volume, params):
    freed_cash, pnl_realized = 0.0, 0.0
    events = []

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
            net_price = calc_net_sell_price(exec_price, position['qty'], params)
            freed_cash += net_price * position['qty']
            pnl = (net_price - position['entry']) * position['qty']
            pnl_realized += pnl
            position['realized_pnl'] += pnl
            position['qty'] = 0
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
        sell_qty = half_sell_qty
        net_price = calc_net_sell_price(exec_price, sell_qty, params)
        freed_cash = net_price * sell_qty
        pnl = (net_price - position['entry']) * sell_qty
        pnl_realized += pnl
        position['realized_pnl'] += pnl
        position['qty'] -= sell_qty
        position['sold_half'] = True
        events.append('TP_HALF')

    if is_stop_hit:
        sell_block_reason = get_exit_sell_block_reason(t_open, t_high, t_low, t_close, t_volume, y_close)
        if sell_block_reason is None:
            exec_price = adjust_long_sell_fill_price(min(position['sl'], t_open))
            net_price = calc_net_sell_price(exec_price, position['qty'], params)
            freed_cash += net_price * position['qty']
            pnl = (net_price - position['entry']) * position['qty']
            pnl_realized += pnl
            position['realized_pnl'] += pnl
            position['qty'] = 0
            events.append('STOP')
        else:
            events.extend(['MISSED_SELL', sell_block_reason])

    return position, freed_cash, pnl_realized, events
