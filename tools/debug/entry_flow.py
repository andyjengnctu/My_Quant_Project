import numpy as np

from core.entry_plans import build_normal_entry_plan, execute_pre_market_entry_plan
from core.extended_signals import (
    build_extended_entry_plan_from_signal,
    create_signal_tracking_state,
    should_clear_extended_signal,
)
from core.price_utils import calc_entry_price
from tools.debug.log_rows import append_debug_trade_row, get_debug_tp_half_price


def process_debug_entry_for_day(
    *,
    position,
    pos_qty_start_of_bar,
    active_extended_signal,
    buy_condition_prev,
    buy_limit_prev,
    atr_prev,
    close_prev,
    sizing_cap,
    t_open,
    t_high,
    t_low,
    t_close,
    t_volume,
    current_date,
    params,
    trade_logs,
):
    buy_triggered = False
    date_str = current_date.strftime('%Y-%m-%d')

    if buy_condition_prev and pos_qty_start_of_bar == 0:
        signal_state = create_signal_tracking_state(buy_limit_prev, atr_prev, params)
        if signal_state is not None:
            active_extended_signal = signal_state

        entry_plan = build_normal_entry_plan(buy_limit_prev, atr_prev, sizing_cap, params)
        entry_result = execute_pre_market_entry_plan(
            entry_plan=entry_plan,
            t_open=t_open,
            t_high=t_high,
            t_low=t_low,
            t_close=t_close,
            t_volume=t_volume,
            y_close=close_prev,
            params=params,
            entry_type='normal',
        )
        if entry_result['filled']:
            position = entry_result['position']
            buy_triggered = True
            active_extended_signal = None
            append_debug_trade_row(
                trade_logs,
                date_str=date_str,
                action="買進",
                price=entry_result['buy_price'],
                net_price=entry_result['entry_price'],
                qty=entry_plan['qty'],
                gross_amount=entry_result['entry_price'] * entry_plan['qty'],
                stop_price=entry_plan['init_sl'],
                tp_half_price=get_debug_tp_half_price(entry_result['tp_half'], entry_plan['qty'], params),
                atr_prev=atr_prev,
                pnl=0.0,
            )
        elif entry_result['count_as_missed_buy']:
            reserved_cost = calc_entry_price(entry_plan['limit_price'], entry_plan['qty'], params) * entry_plan['qty']
            append_debug_trade_row(
                trade_logs,
                date_str=date_str,
                action="錯失買進",
                price=np.nan,
                net_price=np.nan,
                qty=entry_plan['qty'],
                gross_amount=reserved_cost,
                stop_price=entry_plan['init_sl'],
                tp_half_price=np.nan,
                atr_prev=atr_prev,
                pnl=0.0,
                note=f"預掛限價 {entry_plan['limit_price']:.2f} 未成交",
            )
        elif entry_result['is_worse_than_initial_stop']:
            reserved_cost = calc_entry_price(entry_plan['limit_price'], entry_plan['qty'], params) * entry_plan['qty']
            append_debug_trade_row(
                trade_logs,
                date_str=date_str,
                action="放棄進場(先達停損)",
                price=entry_result['buy_price'],
                net_price=np.nan,
                qty=entry_plan['qty'],
                gross_amount=reserved_cost,
                stop_price=entry_plan['init_sl'],
                tp_half_price=np.nan,
                atr_prev=atr_prev,
                pnl=0.0,
                note="不計 miss buy",
            )

    elif active_extended_signal is not None and pos_qty_start_of_bar == 0:
        entry_plan = build_extended_entry_plan_from_signal(active_extended_signal, close_prev, sizing_cap, params)
        entry_result = execute_pre_market_entry_plan(
            entry_plan=entry_plan,
            t_open=t_open,
            t_high=t_high,
            t_low=t_low,
            t_close=t_close,
            t_volume=t_volume,
            y_close=close_prev,
            params=params,
            entry_type='extended',
        )
        if entry_result['filled']:
            position = entry_result['position']
            buy_triggered = True
            active_extended_signal = None
            append_debug_trade_row(
                trade_logs,
                date_str=date_str,
                action="買進(延續候選)",
                price=entry_result['buy_price'],
                net_price=entry_result['entry_price'],
                qty=entry_plan['qty'],
                gross_amount=entry_result['entry_price'] * entry_plan['qty'],
                stop_price=entry_plan['init_sl'],
                tp_half_price=get_debug_tp_half_price(entry_result['tp_half'], entry_plan['qty'], params),
                atr_prev=atr_prev,
                pnl=0.0,
            )
        elif entry_result['count_as_missed_buy']:
            reserved_cost = calc_entry_price(entry_plan['limit_price'], entry_plan['qty'], params) * entry_plan['qty']
            append_debug_trade_row(
                trade_logs,
                date_str=date_str,
                action="錯失買進(延續候選)",
                price=np.nan,
                net_price=np.nan,
                qty=entry_plan['qty'],
                gross_amount=reserved_cost,
                stop_price=entry_plan['init_sl'],
                tp_half_price=np.nan,
                atr_prev=atr_prev,
                pnl=0.0,
                note=f"預掛限價 {entry_plan['limit_price']:.2f} 未成交",
            )
        elif entry_result['is_worse_than_initial_stop']:
            reserved_cost = calc_entry_price(entry_plan['limit_price'], entry_plan['qty'], params) * entry_plan['qty']
            append_debug_trade_row(
                trade_logs,
                date_str=date_str,
                action="放棄進場(延續先達停損)",
                price=entry_result['buy_price'],
                net_price=np.nan,
                qty=entry_plan['qty'],
                gross_amount=reserved_cost,
                stop_price=entry_plan['init_sl'],
                tp_half_price=np.nan,
                atr_prev=atr_prev,
                pnl=0.0,
                note="不計 miss buy",
            )

    if not buy_triggered and position['qty'] == 0 and should_clear_extended_signal(active_extended_signal, t_low):
        active_extended_signal = None

    return position, active_extended_signal
