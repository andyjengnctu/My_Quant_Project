import numpy as np

from core.entry_plans import build_normal_candidate_plan, build_normal_entry_plan, execute_pre_market_entry_plan
from core.price_utils import calc_frozen_target_price
from core.extended_signals import (
    build_extended_candidate_plan_from_signal,
    build_extended_entry_plan_from_signal,
    create_signal_tracking_state,
    should_clear_extended_signal,
)
from core.price_utils import calc_entry_price
from tools.debug.charting import record_active_levels, record_limit_order, record_trade_marker
from tools.debug.log_rows import append_debug_trade_row, get_debug_tp_half_price


def _record_entry_plan_marker(chart_context, *, current_date, entry_plan, entry_type, entry_result, note=""):
    if chart_context is None or entry_plan is None:
        return

    status = "abandoned"
    if entry_result['filled']:
        status = "filled"
    elif entry_result['count_as_missed_buy']:
        status = "missed"

    record_limit_order(
        chart_context,
        current_date=current_date,
        limit_price=entry_plan['limit_price'],
        qty=entry_plan['qty'],
        entry_type=entry_type,
        status=status,
        note=note,
    )


def _record_entry_plan_preview_levels(chart_context, *, current_date, entry_plan):
    if chart_context is None or entry_plan is None:
        return
    tp_price = float(entry_plan.get('target_price', calc_frozen_target_price(entry_plan['limit_price'], entry_plan['init_sl'])))
    record_active_levels(
        chart_context,
        current_date=current_date,
        stop_price=entry_plan['init_sl'],
        tp_half_price=tp_price,
        limit_price=entry_plan['limit_price'],
        entry_price=np.nan,
    )


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
    chart_context=None,
    current_capital=None,
):
    buy_triggered = False
    date_str = current_date.strftime('%Y-%m-%d')

    if buy_condition_prev and pos_qty_start_of_bar == 0:
        signal_state = create_signal_tracking_state(buy_limit_prev, atr_prev, params)
        if signal_state is not None:
            active_extended_signal = signal_state

        preview_candidate_plan = build_normal_candidate_plan(buy_limit_prev, atr_prev, sizing_cap, params)
        _record_entry_plan_preview_levels(chart_context, current_date=current_date, entry_plan=preview_candidate_plan)
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
        marker_note = ""
        if entry_result['count_as_missed_buy']:
            marker_note = f"預掛限價 {entry_plan['limit_price']:.2f} 未成交"
        elif entry_result['is_worse_than_initial_stop']:
            marker_note = "先達停損，放棄進場"
        _record_entry_plan_marker(
            chart_context,
            current_date=current_date,
            entry_plan=entry_plan,
            entry_type='normal',
            entry_result=entry_result,
            note=marker_note,
        )
        if entry_result['filled']:
            position = entry_result['position']
            position['limit_price'] = entry_plan['limit_price']
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
                stop_price=position['initial_stop'],
                tp_half_price=get_debug_tp_half_price(position['tp_half'], entry_plan['qty'], params),
                atr_prev=atr_prev,
                pnl=0.0,
            )
            record_trade_marker(
                chart_context,
                current_date=current_date,
                action="買進",
                price=entry_result['buy_price'],
                qty=entry_plan['qty'],
                meta={
                    'current_capital': None if current_capital is None else float(current_capital),
                    'limit_price': float(entry_plan['limit_price']),
                    'entry_price': float(entry_result['buy_price']),
                    'stop_price': float(position['initial_stop']),
                    'tp_price': float(position['tp_half']),
                    'buy_capital': float(entry_result['entry_price'] * entry_plan['qty']),
                },
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
        preview_candidate_plan = build_extended_candidate_plan_from_signal(active_extended_signal, sizing_cap, params)
        _record_entry_plan_preview_levels(chart_context, current_date=current_date, entry_plan=preview_candidate_plan)
        entry_plan = build_extended_entry_plan_from_signal(
            active_extended_signal,
            sizing_cap,
            params,
            y_close=close_prev,
        )
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
        marker_note = ""
        if entry_result['count_as_missed_buy']:
            marker_note = f"預掛限價 {entry_plan['limit_price']:.2f} 未成交"
        elif entry_result['is_worse_than_initial_stop']:
            marker_note = "延續候選先達停損，放棄進場"
        _record_entry_plan_marker(
            chart_context,
            current_date=current_date,
            entry_plan=entry_plan,
            entry_type='extended',
            entry_result=entry_result,
            note=marker_note,
        )
        if entry_result['filled']:
            position = entry_result['position']
            position['limit_price'] = entry_plan['limit_price']
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
                stop_price=position['initial_stop'],
                tp_half_price=get_debug_tp_half_price(position['tp_half'], entry_plan['qty'], params),
                atr_prev=atr_prev,
                pnl=0.0,
            )
            record_trade_marker(
                chart_context,
                current_date=current_date,
                action="買進(延續候選)",
                price=entry_result['buy_price'],
                qty=entry_plan['qty'],
                meta={
                    'current_capital': None if current_capital is None else float(current_capital),
                    'limit_price': float(entry_plan['limit_price']),
                    'entry_price': float(entry_result['buy_price']),
                    'stop_price': float(position['initial_stop']),
                    'tp_price': float(position['tp_half']),
                    'buy_capital': float(entry_result['entry_price'] * entry_plan['qty']),
                },
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

    if not buy_triggered and position['qty'] == 0 and should_clear_extended_signal(active_extended_signal, t_low, t_high):
        active_extended_signal = None

    return position, active_extended_signal
