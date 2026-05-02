import numpy as np

from core.entry_plans import build_normal_candidate_plan, build_normal_entry_plan, execute_pre_market_entry_plan
from core.extended_signals import (
    build_extended_candidate_plan_from_signal,
    build_extended_entry_plan_from_signal,
    create_signal_tracking_state,
    should_clear_extended_signal,
)
from core.exact_accounting import calc_entry_total_cost, milli_to_money, round_money_for_display
from tools.trade_analysis.charting import record_active_levels, record_shadow_active_levels, record_limit_order, record_trade_marker
from tools.trade_analysis.log_rows import append_debug_trade_row, get_debug_tp_half_price


def _resolve_display_entry_total(entry_result, *, qty, params):
    position = entry_result.get("position") or {}
    exact_entry_total_milli = int(position.get("net_buy_total_milli", 0) or 0)
    if exact_entry_total_milli > 0:
        return milli_to_money(exact_entry_total_milli)
    display_entry_cost = float(entry_result.get("entry_cost", 0.0) or 0.0)
    if display_entry_cost > 0:
        return round_money_for_display(display_entry_cost)
    buy_price = float(entry_result.get("buy_price", 0.0) or 0.0)
    return calc_entry_total_cost(buy_price, int(qty or 0), params)


def _resolve_entry_display_stop_price(position):
    if position is None:
        return np.nan
    active_stop = position.get('sl', np.nan)
    if not _is_missing_price(active_stop):
        return active_stop
    return position.get('initial_stop', np.nan)


def _resolve_entry_display_tp_half_price(position, *, qty, params):
    if position is None or bool(position.get('sold_half', False)):
        return np.nan
    return get_debug_tp_half_price(position.get('tp_half', np.nan), qty, params)


def _is_missing_price(value):
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return value is None


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

    has_shadow_state = entry_plan.get('shadow_position_state') is not None
    if has_shadow_state:
        record_shadow_active_levels(
            chart_context,
            current_date=current_date,
            stop_price=entry_plan.get('continuation_invalidation_barrier', np.nan),
            tp_half_price=entry_plan.get('continuation_completion_barrier', np.nan),
            limit_price=entry_plan['limit_price'],
            entry_price=entry_plan.get('entry_ref_price', np.nan),
        )
        return

    record_active_levels(
        chart_context,
        current_date=current_date,
        stop_price=np.nan,
        tp_half_price=np.nan,
        limit_price=entry_plan['limit_price'],
        entry_price=np.nan,
    )


def _record_active_extended_shadow_levels(chart_context, *, current_date, signal_state):
    if chart_context is None or signal_state is None:
        return

    shadow_position = signal_state.get('shadow_position') or {}
    if int(shadow_position.get('qty', 0) or 0) <= 0:
        return

    tp_half = shadow_position.get('tp_half', np.nan)
    if bool(shadow_position.get('sold_half', False)) or shadow_position.get('pending_exit_action') == 'TP_HALF':
        tp_half = np.nan

    limit_price = signal_state.get('orig_limit', np.nan)
    if limit_price is None or limit_price != limit_price:
        limit_price = shadow_position.get('limit_price', np.nan)

    record_shadow_active_levels(
        chart_context,
        current_date=current_date,
        stop_price=shadow_position.get('sl', np.nan),
        tp_half_price=tp_half,
        limit_price=limit_price,
        entry_price=shadow_position.get('entry_fill_price', np.nan),
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
    high_prev=np.nan,
    sell_condition_prev=False,
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
    ticker=None,
    security_profile=None,
    trade_date=None,
):
    buy_triggered = False
    spent_cash = 0.0
    effective_trade_date = current_date if trade_date is None else trade_date
    date_str = current_date.strftime('%Y-%m-%d')

    if buy_condition_prev and pos_qty_start_of_bar == 0:
        signal_state = create_signal_tracking_state(buy_limit_prev, atr_prev, params, ticker=ticker, security_profile=security_profile)
        if signal_state is not None:
            active_extended_signal = signal_state

        preview_candidate_plan = build_normal_candidate_plan(buy_limit_prev, atr_prev, sizing_cap, params, ticker=ticker, security_profile=security_profile, trade_date=effective_trade_date)
        _record_entry_plan_preview_levels(chart_context, current_date=current_date, entry_plan=preview_candidate_plan)
        entry_plan = build_normal_entry_plan(buy_limit_prev, atr_prev, sizing_cap, params, ticker=ticker, security_profile=security_profile, trade_date=effective_trade_date)
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
            ticker=ticker,
            security_profile=security_profile,
            trade_date=effective_trade_date,
        )
        marker_note = ""
        if entry_result['count_as_missed_buy']:
            marker_note = f"預掛限價 {entry_plan['limit_price']:.2f} 未成交"
        elif entry_result['is_worse_than_initial_stop']:
            marker_note = "放棄進場"
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
            spent_cash = _resolve_display_entry_total(entry_result, qty=entry_plan['qty'], params=params)
            reserved_cost = calc_entry_total_cost(entry_plan['limit_price'], entry_plan['qty'], params)
            append_debug_trade_row(
                trade_logs,
                date_str=date_str,
                action="買進",
                price=entry_result['buy_price'],
                net_price=entry_result['entry_price'],
                qty=entry_plan['qty'],
                gross_amount=spent_cash,
                stop_price=_resolve_entry_display_stop_price(position),
                tp_half_price=_resolve_entry_display_tp_half_price(position, qty=entry_plan['qty'], params=params),
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
                    'limit_price': float(entry_plan['limit_price']),
                    'entry_price': float(entry_result['buy_price']),
                    'stop_price': float(_resolve_entry_display_stop_price(position)),
                    'tp_price': float(_resolve_entry_display_tp_half_price(position, qty=entry_plan['qty'], params=params)),
                    'reserved_capital': reserved_cost,
                    'buy_capital': spent_cash,
                    'current_capital': None if current_capital is None else float(current_capital),
                    'entry_type': 'normal',
                    'result': '成交',
                },
            )
        elif entry_result['count_as_missed_buy']:
            reserved_cost = calc_entry_total_cost(entry_plan['limit_price'], entry_plan['qty'], params)
            append_debug_trade_row(
                trade_logs,
                date_str=date_str,
                action="錯失買進",
                price=np.nan,
                net_price=np.nan,
                qty=entry_plan['qty'],
                gross_amount=reserved_cost,
                stop_price=np.nan,
                tp_half_price=np.nan,
                atr_prev=atr_prev,
                pnl=0.0,
                note=f"預掛限價 {entry_plan['limit_price']:.2f} 未成交",
            )
        elif entry_result['is_worse_than_initial_stop']:
            reserved_cost = calc_entry_total_cost(entry_plan['limit_price'], entry_plan['qty'], params)
            append_debug_trade_row(
                trade_logs,
                date_str=date_str,
                action="放棄進場",
                price=entry_result['buy_price'],
                net_price=np.nan,
                qty=entry_plan['qty'],
                gross_amount=reserved_cost,
                stop_price=np.nan,
                tp_half_price=np.nan,
                atr_prev=atr_prev,
                pnl=0.0,
                note="不計 miss buy",
            )

    elif active_extended_signal is not None and pos_qty_start_of_bar == 0:
        preview_candidate_plan = build_extended_candidate_plan_from_signal(active_extended_signal, sizing_cap, params, ticker=ticker, security_profile=security_profile, trade_date=effective_trade_date)
        _record_entry_plan_preview_levels(chart_context, current_date=current_date, entry_plan=preview_candidate_plan)
        entry_plan = build_extended_entry_plan_from_signal(
            active_extended_signal,
            sizing_cap,
            params,
            y_close=close_prev,
            ticker=ticker,
            security_profile=security_profile,
            trade_date=effective_trade_date,
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
            ticker=ticker,
            security_profile=security_profile,
            trade_date=effective_trade_date,
        )
        marker_note = ""
        if entry_result['count_as_missed_buy']:
            marker_note = f"預掛限價 {entry_plan['limit_price']:.2f} 未成交"
        elif entry_result['is_worse_than_initial_stop']:
            marker_note = "放棄進場(延續候選)"
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
            spent_cash = _resolve_display_entry_total(entry_result, qty=entry_plan['qty'], params=params)
            reserved_cost = calc_entry_total_cost(entry_plan['limit_price'], entry_plan['qty'], params)
            append_debug_trade_row(
                trade_logs,
                date_str=date_str,
                action="買進(延續候選)",
                price=entry_result['buy_price'],
                net_price=entry_result['entry_price'],
                qty=entry_plan['qty'],
                gross_amount=spent_cash,
                stop_price=_resolve_entry_display_stop_price(position),
                tp_half_price=_resolve_entry_display_tp_half_price(position, qty=entry_plan['qty'], params=params),
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
                    'limit_price': float(entry_plan['limit_price']),
                    'entry_price': float(entry_result['buy_price']),
                    'stop_price': float(_resolve_entry_display_stop_price(position)),
                    'tp_price': float(_resolve_entry_display_tp_half_price(position, qty=entry_plan['qty'], params=params)),
                    'reserved_capital': reserved_cost,
                    'buy_capital': spent_cash,
                    'current_capital': None if current_capital is None else float(current_capital),
                    'entry_type': 'extended',
                    'result': '成交',
                },
            )
        elif entry_result['count_as_missed_buy']:
            reserved_cost = calc_entry_total_cost(entry_plan['limit_price'], entry_plan['qty'], params)
            append_debug_trade_row(
                trade_logs,
                date_str=date_str,
                action="錯失買進(延續候選)",
                price=np.nan,
                net_price=np.nan,
                qty=entry_plan['qty'],
                gross_amount=reserved_cost,
                stop_price=np.nan,
                tp_half_price=np.nan,
                atr_prev=atr_prev,
                pnl=0.0,
                note=f"預掛限價 {entry_plan['limit_price']:.2f} 未成交",
            )
        elif entry_result['is_worse_than_initial_stop']:
            reserved_cost = calc_entry_total_cost(entry_plan['limit_price'], entry_plan['qty'], params)
            append_debug_trade_row(
                trade_logs,
                date_str=date_str,
                action="放棄進場(延續候選)",
                price=entry_result['buy_price'],
                net_price=np.nan,
                qty=entry_plan['qty'],
                gross_amount=reserved_cost,
                stop_price=np.nan,
                tp_half_price=np.nan,
                atr_prev=atr_prev,
                pnl=0.0,
                note="不計 miss buy",
            )

    if not buy_triggered and position['qty'] == 0 and active_extended_signal is not None:
        should_clear_signal = should_clear_extended_signal(
            active_extended_signal,
            t_low,
            t_high,
            t_open=t_open,
            t_close=t_close,
            t_volume=t_volume,
            y_close=close_prev,
            y_high=high_prev,
            y_atr=atr_prev,
            y_ind_sell=sell_condition_prev,
            sizing_capital=sizing_cap,
            current_date=current_date,
            params=params,
        )
        if should_clear_signal:
            active_extended_signal = None
        else:
            _record_active_extended_shadow_levels(
                chart_context,
                current_date=current_date,
                signal_state=active_extended_signal,
            )

    return position, active_extended_signal, spent_cash
