import os
import numpy as np
import pandas as pd

from core.v16_core import (
    adjust_long_sell_fill_price,
    build_extended_entry_plan_from_signal,
    build_normal_entry_plan,
    calc_entry_price,
    calc_net_sell_price,
    can_execute_half_take_profit,
    create_signal_tracking_state,
    execute_bar_step,
    execute_pre_market_entry_plan,
    generate_signals,
    should_clear_extended_signal,
)
from tools.debug.reporting import finalize_debug_trade_logs


def get_debug_tp_half_price(tp_half, qty, params):
    return tp_half if can_execute_half_take_profit(qty, params.tp_percent) else np.nan


def _append_row(trade_logs, *, date_str, action, price, net_price, qty, gross_amount,
                stop_price, tp_half_price, atr_prev, pnl, note=""):
    row = {
        "日期": date_str,
        "動作": action,
        "成交價": price,
        "含息成本價": net_price,
        "股數": qty,
        "投入總金額": gross_amount,
        "設定停損價": stop_price,
        "半倉停利價": tp_half_price,
        "ATR(前日)": atr_prev,
        "單筆實質損益": pnl,
    }
    if note:
        row["備註"] = note
    trade_logs.append(row)


def run_debug_backtest(df, ticker, params, output_dir, colors, export_excel=True, verbose=True):
    """以正式核心邏輯為準，輸出可讀交易明細的除錯工具"""
    h = df['High'].to_numpy(dtype=np.float64, copy=False)
    l = df['Low'].to_numpy(dtype=np.float64, copy=False)
    c = df['Close'].to_numpy(dtype=np.float64, copy=False)
    o = df['Open'].to_numpy(dtype=np.float64, copy=False)
    v = df['Volume'].to_numpy(dtype=np.float64, copy=False)
    dates = df.index

    atr_main, buy_condition, sell_condition, buy_limits = generate_signals(df, params)

    position = {'qty': 0}
    active_extended_signal = None
    current_capital = params.initial_capital
    trade_logs = []

    for j in range(1, len(c)):
        if np.isnan(atr_main[j - 1]):
            continue

        pos_start_of_current_bar = position['qty']

        if pos_start_of_current_bar > 0:
            prev_qty = position['qty']
            prev_realized = position.get('realized_pnl', 0.0)
            prev_tp_half = position.get('tp_half', np.nan)

            position, _freed_cash, pnl_realized, events = execute_bar_step(
                position,
                atr_main[j - 1],
                sell_condition[j - 1],
                c[j - 1],
                o[j],
                h[j],
                l[j],
                c[j],
                v[j],
                params,
            )

            realized_delta = position.get('realized_pnl', 0.0) - prev_realized
            active_stop_after_update = position.get('sl', np.nan)
            date_str = dates[j].strftime('%Y-%m-%d')

            if 'TP_HALF' in events and realized_delta != 0:
                sold_qty = prev_qty - position['qty']
                exec_sell_price_half = adjust_long_sell_fill_price(max(prev_tp_half, o[j]))
                sell_net_price_half = calc_net_sell_price(exec_sell_price_half, sold_qty, params)
                _append_row(
                    trade_logs,
                    date_str=date_str,
                    action="半倉停利",
                    price=exec_sell_price_half,
                    net_price=sell_net_price_half,
                    qty=sold_qty,
                    gross_amount=sell_net_price_half * sold_qty,
                    stop_price=active_stop_after_update,
                    tp_half_price=np.nan,
                    atr_prev=atr_main[j - 1],
                    pnl=realized_delta,
                )

            if 'STOP' in events or 'IND_SELL' in events:
                half_qty = prev_qty - position['qty'] if 'TP_HALF' in events else 0
                final_exit_qty = prev_qty - half_qty
                action_str = "停損殺出" if 'STOP' in events else "指標賣出"
                sell_price = (
                    adjust_long_sell_fill_price(min(active_stop_after_update, o[j]))
                    if 'STOP' in events else adjust_long_sell_fill_price(o[j])
                )
                sell_net_price = calc_net_sell_price(sell_price, final_exit_qty, params)
                final_leg_pnl = pnl_realized - realized_delta if 'TP_HALF' in events else pnl_realized
                _append_row(
                    trade_logs,
                    date_str=date_str,
                    action=action_str,
                    price=sell_price,
                    net_price=sell_net_price,
                    qty=final_exit_qty,
                    gross_amount=sell_net_price * final_exit_qty,
                    stop_price=active_stop_after_update,
                    tp_half_price=np.nan,
                    atr_prev=atr_main[j - 1],
                    pnl=final_leg_pnl,
                )
            elif 'MISSED_SELL' in events:
                sell_block_reason = next((event for event in events if event in {'NO_VOLUME', 'LOCKED_DOWN'}), None)
                reason_note = {
                    'NO_VOLUME': '零量，當日無法賣出',
                    'LOCKED_DOWN': '一字跌停鎖死，當日無法賣出',
                }.get(sell_block_reason, '賣出受阻，當日無法賣出')
                _append_row(
                    trade_logs,
                    date_str=date_str,
                    action="錯失賣出",
                    price=np.nan,
                    net_price=np.nan,
                    qty=prev_qty,
                    gross_amount=np.nan,
                    stop_price=active_stop_after_update,
                    tp_half_price=np.nan,
                    atr_prev=atr_main[j - 1],
                    pnl=0.0,
                    note=reason_note,
                )

            current_capital += pnl_realized

        is_setup_prev = buy_condition[j - 1] and (pos_start_of_current_bar == 0)
        buy_triggered = False
        sizing_cap = current_capital if getattr(params, 'use_compounding', True) else params.initial_capital
        date_str = dates[j].strftime('%Y-%m-%d')

        if is_setup_prev:
            signal_state = create_signal_tracking_state(buy_limits[j - 1], atr_main[j - 1], params)
            if signal_state is not None:
                active_extended_signal = signal_state

            entry_plan = build_normal_entry_plan(buy_limits[j - 1], atr_main[j - 1], sizing_cap, params)
            entry_result = execute_pre_market_entry_plan(
                entry_plan=entry_plan,
                t_open=o[j], t_high=h[j], t_low=l[j], t_close=c[j], t_volume=v[j], y_close=c[j - 1],
                params=params, entry_type='normal',
            )
            if entry_result['filled']:
                position = entry_result['position']
                buy_triggered = True
                active_extended_signal = None
                _append_row(
                    trade_logs,
                    date_str=date_str,
                    action="買進",
                    price=entry_result['buy_price'],
                    net_price=entry_result['entry_price'],
                    qty=entry_plan['qty'],
                    gross_amount=entry_result['entry_price'] * entry_plan['qty'],
                    stop_price=entry_plan['init_sl'],
                    tp_half_price=get_debug_tp_half_price(entry_result['tp_half'], entry_plan['qty'], params),
                    atr_prev=atr_main[j - 1],
                    pnl=0.0,
                )
            elif entry_result['count_as_missed_buy']:
                reserved_cost = calc_entry_price(entry_plan['limit_price'], entry_plan['qty'], params) * entry_plan['qty']
                _append_row(
                    trade_logs,
                    date_str=date_str,
                    action="錯失買進",
                    price=np.nan,
                    net_price=np.nan,
                    qty=entry_plan['qty'],
                    gross_amount=reserved_cost,
                    stop_price=entry_plan['init_sl'],
                    tp_half_price=np.nan,
                    atr_prev=atr_main[j - 1],
                    pnl=0.0,
                    note=f"預掛限價 {entry_plan['limit_price']:.2f} 未成交",
                )
            elif entry_result['is_worse_than_initial_stop']:
                reserved_cost = calc_entry_price(entry_plan['limit_price'], entry_plan['qty'], params) * entry_plan['qty']
                _append_row(
                    trade_logs,
                    date_str=date_str,
                    action="放棄進場(先達停損)",
                    price=entry_result['buy_price'],
                    net_price=np.nan,
                    qty=entry_plan['qty'],
                    gross_amount=reserved_cost,
                    stop_price=entry_plan['init_sl'],
                    tp_half_price=np.nan,
                    atr_prev=atr_main[j - 1],
                    pnl=0.0,
                    note="不計 miss buy",
                )

        elif active_extended_signal is not None and pos_start_of_current_bar == 0:
            entry_plan = build_extended_entry_plan_from_signal(active_extended_signal, c[j - 1], sizing_cap, params)
            entry_result = execute_pre_market_entry_plan(
                entry_plan=entry_plan,
                t_open=o[j], t_high=h[j], t_low=l[j], t_close=c[j], t_volume=v[j], y_close=c[j - 1],
                params=params, entry_type='extended',
            )
            if entry_result['filled']:
                position = entry_result['position']
                buy_triggered = True
                active_extended_signal = None
                _append_row(
                    trade_logs,
                    date_str=date_str,
                    action="買進(延續候選)",
                    price=entry_result['buy_price'],
                    net_price=entry_result['entry_price'],
                    qty=entry_plan['qty'],
                    gross_amount=entry_result['entry_price'] * entry_plan['qty'],
                    stop_price=entry_plan['init_sl'],
                    tp_half_price=get_debug_tp_half_price(entry_result['tp_half'], entry_plan['qty'], params),
                    atr_prev=atr_main[j - 1],
                    pnl=0.0,
                )
            elif entry_result['count_as_missed_buy']:
                reserved_cost = calc_entry_price(entry_plan['limit_price'], entry_plan['qty'], params) * entry_plan['qty']
                _append_row(
                    trade_logs,
                    date_str=date_str,
                    action="錯失買進(延續候選)",
                    price=np.nan,
                    net_price=np.nan,
                    qty=entry_plan['qty'],
                    gross_amount=reserved_cost,
                    stop_price=entry_plan['init_sl'],
                    tp_half_price=np.nan,
                    atr_prev=atr_main[j - 1],
                    pnl=0.0,
                    note=f"預掛限價 {entry_plan['limit_price']:.2f} 未成交",
                )
            elif entry_result['is_worse_than_initial_stop']:
                reserved_cost = calc_entry_price(entry_plan['limit_price'], entry_plan['qty'], params) * entry_plan['qty']
                _append_row(
                    trade_logs,
                    date_str=date_str,
                    action="放棄進場(延續先達停損)",
                    price=entry_result['buy_price'],
                    net_price=np.nan,
                    qty=entry_plan['qty'],
                    gross_amount=reserved_cost,
                    stop_price=entry_plan['init_sl'],
                    tp_half_price=np.nan,
                    atr_prev=atr_main[j - 1],
                    pnl=0.0,
                    note="不計 miss buy",
                )

        if not buy_triggered and position['qty'] == 0 and should_clear_extended_signal(active_extended_signal, l[j]):
            active_extended_signal = None

    if position['qty'] > 0:
        exec_sell_price = adjust_long_sell_fill_price(c[-1])
        sell_net_price = calc_net_sell_price(exec_sell_price, position['qty'], params)
        final_leg_pnl = (sell_net_price - position['entry']) * position['qty']
        _append_row(
            trade_logs,
            date_str=dates[-1].strftime('%Y-%m-%d'),
            action="期末強制結算",
            price=exec_sell_price,
            net_price=sell_net_price,
            qty=position['qty'],
            gross_amount=sell_net_price * position['qty'],
            stop_price=position.get('sl', np.nan),
            tp_half_price=np.nan,
            atr_prev=atr_main[-1] if len(atr_main) > 0 else np.nan,
            pnl=final_leg_pnl,
        )

    return finalize_debug_trade_logs(
        trade_logs=trade_logs,
        ticker=ticker,
        output_dir=output_dir,
        colors=colors,
        export_excel=export_excel,
        verbose=verbose,
    )
