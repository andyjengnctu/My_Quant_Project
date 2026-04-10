import numpy as np

from core.exact_accounting import (
    build_sell_ledger_from_price,
    calc_reconciled_exit_display_pnl,
    milli_to_money,
    register_display_realized_pnl,
    round_money_for_display,
)
from core.position_step import execute_bar_step, sum_last_exec_contexts_milli
from core.price_utils import adjust_long_sell_fill_price, calc_net_sell_price
from tools.debug.charting import record_trade_marker
from tools.debug.history_snapshot import build_pit_history_snapshot
from tools.debug.log_rows import append_debug_trade_row


def _first_exec_context(position, event_name):
    for ctx in position.get('_last_exec_contexts', []):
        if ctx.get('event') == event_name:
            return ctx
    return None


def _resolve_completed_trade_count(history_snapshot, *, include_current_round_trip):
    if history_snapshot is None:
        return None
    base_trade_count = int(history_snapshot.get('trade_count', 0) or 0)
    return base_trade_count + 1 if include_current_round_trip else base_trade_count


def _resolve_full_entry_capital(position, fallback_qty):
    exact_entry_total_milli = int(position.get('net_buy_total_milli', 0) or 0)
    if exact_entry_total_milli > 0:
        return milli_to_money(exact_entry_total_milli)
    display_entry_capital = float(position.get('entry_capital_total', 0.0) or 0.0)
    if display_entry_capital > 0:
        return round_money_for_display(display_entry_capital)
    entry_price = float(position.get('entry', 0.0) or 0.0)
    initial_qty = int(position.get('initial_qty', fallback_qty) or fallback_qty or 0)
    return round_money_for_display(entry_price * initial_qty)


def _resolve_display_sell_total(exit_context, *, sell_price, qty, params):
    if exit_context is not None and int(exit_context.get('net_total_milli', 0) or 0) > 0:
        return milli_to_money(int(exit_context.get('net_total_milli', 0) or 0))
    sell_ledger = build_sell_ledger_from_price(sell_price, qty, params)
    return milli_to_money(sell_ledger['net_sell_total_milli'])


def _resolve_display_leg_return_pct(position, exit_context, *, fallback_entry_price, fallback_net_price):
    allocated_cost_milli = 0 if exit_context is None else int(exit_context.get('allocated_cost_milli', 0) or 0)
    pnl_milli = 0 if exit_context is None else int(exit_context.get('pnl_milli', 0) or 0)
    if allocated_cost_milli > 0:
        return float(milli_to_money(pnl_milli) / milli_to_money(allocated_cost_milli) * 100.0)
    entry_price = float(position.get('entry', fallback_entry_price) or fallback_entry_price or 0.0)
    if entry_price <= 0:
        return 0.0
    return float((float(fallback_net_price) - entry_price) / entry_price * 100.0)


def _build_completed_trade_snapshot(stats_index, current_date, params, current_capital_after_event, overall_max_drawdown):
    if stats_index is None or current_capital_after_event is None:
        return None
    return build_pit_history_snapshot(
        stats_index,
        current_date,
        params,
        current_capital_after_event,
        overall_max_drawdown,
        include_current_date_exits=True,
    )


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
    history_snapshot=None,
    stats_index=None,
    current_capital_before_event=None,
    overall_max_drawdown=0.0,
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

    freed_cash_milli, pnl_realized_milli = sum_last_exec_contexts_milli(position)
    freed_cash = milli_to_money(freed_cash_milli)
    realized_delta = milli_to_money(pnl_realized_milli)
    active_stop_after_update = position.get('sl', np.nan)
    date_str = current_date.strftime('%Y-%m-%d')
    tp_context = _first_exec_context(position, 'TP_HALF')
    stop_context = _first_exec_context(position, 'STOP')
    ind_sell_context = _first_exec_context(position, 'IND_SELL')

    if 'TP_HALF' in events and tp_context is not None:
        sold_qty = int(tp_context['qty'])
        exec_sell_price_half = float(tp_context['exec_price'])
        sell_net_price_half = float(tp_context['net_price'])
        tp_leg_pnl = round_money_for_display(tp_context['pnl'])
        tp_sell_total = _resolve_display_sell_total(tp_context, sell_price=exec_sell_price_half, qty=sold_qty, params=params)
        current_capital_after_tp = None if current_capital_before_event is None else float(current_capital_before_event) + tp_sell_total
        append_debug_trade_row(
            trade_logs,
            date_str=date_str,
            action="半倉停利",
            price=exec_sell_price_half,
            net_price=sell_net_price_half,
            qty=sold_qty,
            gross_amount=tp_sell_total,
            stop_price=active_stop_after_update,
            tp_half_price=np.nan,
            atr_prev=atr_prev,
            pnl=tp_leg_pnl,
        )
        register_display_realized_pnl(position, tp_leg_pnl)
        record_trade_marker(
            chart_context,
            current_date=current_date,
            action="半倉停利",
            price=exec_sell_price_half,
            qty=sold_qty,
            meta={
                'current_capital': current_capital_after_tp,
                'pnl_value': tp_leg_pnl,
                'pnl_pct': _resolve_display_leg_return_pct(
                    position,
                    tp_context,
                    fallback_entry_price=exec_sell_price_half,
                    fallback_net_price=sell_net_price_half,
                ),
                'sell_capital': float(tp_sell_total),
                'payoff_ratio': None if history_snapshot is None else float(history_snapshot.get('payoff_ratio', 0.0)),
                'win_rate': None if history_snapshot is None else float(history_snapshot.get('win_rate', 0.0)),
                'expected_value': None if history_snapshot is None else float(history_snapshot.get('expected_value', 0.0)),
                'trade_count': None if history_snapshot is None else int(history_snapshot.get('trade_count', 0) or 0),
                'max_drawdown': None if history_snapshot is None else float(history_snapshot.get('max_drawdown', 0.0)),
            },
        )

    if 'STOP' in events or 'IND_SELL' in events:
        exit_context = stop_context if 'STOP' in events else ind_sell_context
        final_exit_qty = prev_qty if exit_context is None else int(exit_context['qty'])
        action_str = "停損殺出" if 'STOP' in events else "指標賣出"
        sell_price = adjust_long_sell_fill_price(t_open) if exit_context is None else float(exit_context['exec_price'])
        sell_net_price = calc_net_sell_price(sell_price, final_exit_qty, params) if exit_context is None else float(exit_context['net_price'])
        sell_total_amount = _resolve_display_sell_total(exit_context, sell_price=sell_price, qty=final_exit_qty, params=params)
        current_capital_after_exit = None if current_capital_before_event is None else float(current_capital_before_event) + float(freed_cash)
        completed_trade_snapshot = _build_completed_trade_snapshot(
            stats_index,
            current_date,
            params,
            current_capital_after_exit,
            overall_max_drawdown,
        )
        total_pnl = float(position.get('realized_pnl', pnl_realized))
        final_leg_pnl = calc_reconciled_exit_display_pnl(position, total_pnl)
        full_entry_capital = _resolve_full_entry_capital(position, prev_qty)
        total_return_pct = (total_pnl / full_entry_capital * 100.0) if full_entry_capital > 0 else 0.0
        append_debug_trade_row(
            trade_logs,
            date_str=date_str,
            action=action_str,
            price=sell_price,
            net_price=sell_net_price,
            qty=final_exit_qty,
            gross_amount=sell_total_amount,
            stop_price=active_stop_after_update,
            tp_half_price=np.nan,
            atr_prev=atr_prev,
            pnl=final_leg_pnl,
        )
        register_display_realized_pnl(position, final_leg_pnl)
        record_trade_marker(
            chart_context,
            current_date=current_date,
            action=action_str,
            price=sell_price,
            qty=final_exit_qty,
            meta={
                'current_capital': current_capital_after_exit,
                'pnl_value': float(final_leg_pnl),
                'total_pnl': float(total_pnl),
                'pnl_pct': float(total_return_pct),
                'sell_capital': float(sell_total_amount),
                'payoff_ratio': None if completed_trade_snapshot is None else float(completed_trade_snapshot.get('payoff_ratio', 0.0)),
                'win_rate': None if completed_trade_snapshot is None else float(completed_trade_snapshot.get('win_rate', 0.0)),
                'expected_value': None if completed_trade_snapshot is None else float(completed_trade_snapshot.get('expected_value', 0.0)),
                'trade_count': _resolve_completed_trade_count(completed_trade_snapshot, include_current_round_trip=False),
                'max_drawdown': None if completed_trade_snapshot is None else float(completed_trade_snapshot.get('max_drawdown', 0.0)),
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

    return position, freed_cash


def append_debug_forced_closeout(
    *,
    position,
    current_date,
    atr_last,
    params,
    trade_logs,
    chart_context=None,
    current_capital_before_event=None,
    stats_index=None,
    overall_max_drawdown=0.0,
):
    exec_sell_price = adjust_long_sell_fill_price(position['close_price'])
    sell_net_price = calc_net_sell_price(exec_sell_price, position['qty'], params)
    sell_ledger = build_sell_ledger_from_price(exec_sell_price, position['qty'], params)
    final_leg_actual_pnl_milli = sell_ledger['net_sell_total_milli'] - position.get('remaining_cost_basis_milli', 0)
    total_pnl_milli = int(position.get('realized_pnl_milli', 0) or 0) + int(final_leg_actual_pnl_milli)
    total_pnl = milli_to_money(total_pnl_milli)
    final_leg_pnl = calc_reconciled_exit_display_pnl(position, total_pnl)
    current_capital_after_exit = None if current_capital_before_event is None else float(current_capital_before_event) + milli_to_money(sell_ledger['net_sell_total_milli'])
    completed_trade_snapshot = _build_completed_trade_snapshot(
        stats_index,
        current_date,
        params,
        current_capital_after_exit,
        overall_max_drawdown,
    )
    full_entry_capital = _resolve_full_entry_capital(position, position['qty'])
    total_return_pct = (total_pnl / full_entry_capital * 100.0) if full_entry_capital > 0 else 0.0
    append_debug_trade_row(
        trade_logs,
        date_str=current_date.strftime('%Y-%m-%d'),
        action="期末強制結算",
        price=exec_sell_price,
        net_price=sell_net_price,
        qty=position['qty'],
        gross_amount=milli_to_money(sell_ledger['net_sell_total_milli']),
        stop_price=position.get('sl', np.nan),
        tp_half_price=np.nan,
        atr_prev=atr_last,
        pnl=final_leg_pnl,
    )
    register_display_realized_pnl(position, final_leg_pnl)
    record_trade_marker(
        chart_context,
        current_date=current_date,
        action="期末強制結算",
        price=exec_sell_price,
        qty=position['qty'],
        meta={
            'current_capital': current_capital_after_exit,
            'pnl_value': float(final_leg_pnl),
            'total_pnl': float(total_pnl),
            'pnl_pct': float(total_return_pct),
            'sell_capital': float(milli_to_money(sell_ledger['net_sell_total_milli'])),
            'payoff_ratio': None if completed_trade_snapshot is None else float(completed_trade_snapshot.get('payoff_ratio', 0.0)),
            'win_rate': None if completed_trade_snapshot is None else float(completed_trade_snapshot.get('win_rate', 0.0)),
            'expected_value': None if completed_trade_snapshot is None else float(completed_trade_snapshot.get('expected_value', 0.0)),
            'trade_count': _resolve_completed_trade_count(completed_trade_snapshot, include_current_round_trip=False),
            'max_drawdown': None if completed_trade_snapshot is None else float(completed_trade_snapshot.get('max_drawdown', 0.0)),
        },
    )
