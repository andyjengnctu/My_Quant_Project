import numpy as np

from core.position_step import execute_bar_step
from core.price_utils import adjust_long_sell_fill_price, calc_net_sell_price
from tools.debug.charting import record_trade_marker
from tools.debug.log_rows import append_debug_trade_row


def process_debug_position_step(
    *,
    position,
    atr_prev,
    sell_condition_prev,
    close_prev,
    t_open,
    t_high,
    t_low,
    t_close,
    t_volume,
    current_date,
    params,
    trade_logs,
    chart_context=None,
):
    prev_qty = position['qty']
    prev_realized = position.get('realized_pnl', 0.0)
    prev_tp_half = position.get('tp_half', np.nan)

    position, _freed_cash, pnl_realized, events = execute_bar_step(
        position,
        atr_prev,
        sell_condition_prev,
        close_prev,
        t_open,
        t_high,
        t_low,
        t_close,
        t_volume,
        params,
    )

    realized_delta = position.get('realized_pnl', 0.0) - prev_realized
    active_stop_after_update = position.get('sl', np.nan)
    date_str = current_date.strftime('%Y-%m-%d')

    if 'TP_HALF' in events and realized_delta != 0:
        sold_qty = prev_qty - position['qty']
        exec_sell_price_half = adjust_long_sell_fill_price(max(prev_tp_half, t_open))
        sell_net_price_half = calc_net_sell_price(exec_sell_price_half, sold_qty, params)
        append_debug_trade_row(
            trade_logs,
            date_str=date_str,
            action="半倉停利",
            price=exec_sell_price_half,
            net_price=sell_net_price_half,
            qty=sold_qty,
            gross_amount=sell_net_price_half * sold_qty,
            stop_price=active_stop_after_update,
            tp_half_price=np.nan,
            atr_prev=atr_prev,
            pnl=realized_delta,
        )
        record_trade_marker(
            chart_context,
            current_date=current_date,
            action="半倉停利",
            price=exec_sell_price_half,
            qty=sold_qty,
            meta={
                'pnl_value': float(realized_delta),
                'pnl_pct': float(((sell_net_price_half - float(position.get('entry', exec_sell_price_half))) / float(position.get('entry', exec_sell_price_half)) * 100.0) if float(position.get('entry', 0.0) or 0.0) > 0 else 0.0),
            },
        )

    if 'STOP' in events or 'IND_SELL' in events:
        half_qty = prev_qty - position['qty'] if 'TP_HALF' in events else 0
        final_exit_qty = prev_qty - half_qty
        action_str = "停損殺出" if 'STOP' in events else "指標賣出"
        sell_price = (
            adjust_long_sell_fill_price(min(active_stop_after_update, t_open))
            if 'STOP' in events else adjust_long_sell_fill_price(t_open)
        )
        sell_net_price = calc_net_sell_price(sell_price, final_exit_qty, params)
        final_leg_pnl = pnl_realized - realized_delta if 'TP_HALF' in events else pnl_realized
        append_debug_trade_row(
            trade_logs,
            date_str=date_str,
            action=action_str,
            price=sell_price,
            net_price=sell_net_price,
            qty=final_exit_qty,
            gross_amount=sell_net_price * final_exit_qty,
            stop_price=active_stop_after_update,
            tp_half_price=np.nan,
            atr_prev=atr_prev,
            pnl=final_leg_pnl,
        )
        record_trade_marker(
            chart_context,
            current_date=current_date,
            action=action_str,
            price=sell_price,
            qty=final_exit_qty,
            meta={
                'pnl_value': float(final_leg_pnl),
                'pnl_pct': float(((sell_net_price - float(position.get('entry', sell_price))) / float(position.get('entry', sell_price)) * 100.0) if float(position.get('entry', 0.0) or 0.0) > 0 else 0.0),
            },
        )
    elif 'MISSED_SELL' in events:
        sell_block_reason = next((event for event in events if event in {'NO_VOLUME', 'LOCKED_DOWN'}), None)
        reason_note = {
            'NO_VOLUME': '零量，當日無法賣出',
            'LOCKED_DOWN': '一字跌停鎖死，當日無法賣出',
        }.get(sell_block_reason, '賣出受阻，當日無法賣出')
        append_debug_trade_row(
            trade_logs,
            date_str=date_str,
            action="錯失賣出",
            price=np.nan,
            net_price=np.nan,
            qty=prev_qty,
            gross_amount=np.nan,
            stop_price=active_stop_after_update,
            tp_half_price=np.nan,
            atr_prev=atr_prev,
            pnl=0.0,
            note=reason_note,
        )
        marker_price = active_stop_after_update if not np.isnan(active_stop_after_update) else t_close
        record_trade_marker(
            chart_context,
            current_date=current_date,
            action="錯失賣出",
            price=marker_price,
            qty=prev_qty,
            note=reason_note,
        )

    return position, pnl_realized


def append_debug_forced_closeout(*, position, current_date, atr_last, params, trade_logs, chart_context=None):
    exec_sell_price = adjust_long_sell_fill_price(position['close_price'])
    sell_net_price = calc_net_sell_price(exec_sell_price, position['qty'], params)
    final_leg_pnl = (sell_net_price - position['entry']) * position['qty']
    append_debug_trade_row(
        trade_logs,
        date_str=current_date.strftime('%Y-%m-%d'),
        action="期末強制結算",
        price=exec_sell_price,
        net_price=sell_net_price,
        qty=position['qty'],
        gross_amount=sell_net_price * position['qty'],
        stop_price=position.get('sl', np.nan),
        tp_half_price=np.nan,
        atr_prev=atr_last,
        pnl=final_leg_pnl,
    )
    record_trade_marker(
        chart_context,
        current_date=current_date,
        action="期末強制結算",
        price=exec_sell_price,
        qty=position['qty'],
        meta={
            'pnl_value': float(final_leg_pnl),
            'pnl_pct': float(((sell_net_price - float(position.get('entry', exec_sell_price))) / float(position.get('entry', exec_sell_price)) * 100.0) if float(position.get('entry', 0.0) or 0.0) > 0 else 0.0),
        },
    )
