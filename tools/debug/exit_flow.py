import numpy as np

from core.position_step import execute_bar_step
from core.price_utils import adjust_long_sell_fill_price, calc_net_sell_price
from tools.debug.charting import record_trade_marker
from tools.debug.history_snapshot import build_pit_history_snapshot
from tools.debug.log_rows import append_debug_trade_row


def _resolve_completed_trade_count(history_snapshot, *, include_current_round_trip):
    if history_snapshot is None:
        return None
    base_trade_count = int(history_snapshot.get('trade_count', 0) or 0)
    return base_trade_count + 1 if include_current_round_trip else base_trade_count


def _resolve_full_entry_capital(position, fallback_qty):
    entry_price = float(position.get('entry', 0.0) or 0.0)
    initial_qty = int(position.get('initial_qty', fallback_qty) or fallback_qty or 0)
    return entry_price * initial_qty


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

    realized_delta = position.get('realized_pnl', 0.0) - prev_realized
    active_stop_after_update = position.get('sl', np.nan)
    date_str = current_date.strftime('%Y-%m-%d')

    if 'TP_HALF' in events and realized_delta != 0:
        sold_qty = prev_qty - position['qty']
        exec_sell_price_half = adjust_long_sell_fill_price(max(prev_tp_half, t_open))
        sell_net_price_half = calc_net_sell_price(exec_sell_price_half, sold_qty, params)
        current_capital_after_tp = None if current_capital_before_event is None else float(current_capital_before_event) + float(realized_delta)
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
                'current_capital': current_capital_after_tp,
                'pnl_value': float(realized_delta),
                'pnl_pct': float(((sell_net_price_half - float(position.get('entry', exec_sell_price_half))) / float(position.get('entry', exec_sell_price_half)) * 100.0) if float(position.get('entry', 0.0) or 0.0) > 0 else 0.0),
                'sell_capital': float(sell_net_price_half * sold_qty),
                'payoff_ratio': None if history_snapshot is None else float(history_snapshot.get('payoff_ratio', 0.0)),
                'win_rate': None if history_snapshot is None else float(history_snapshot.get('win_rate', 0.0)),
                'expected_value': None if history_snapshot is None else float(history_snapshot.get('expected_value', 0.0)),
                'trade_count': None if history_snapshot is None else int(history_snapshot.get('trade_count', 0) or 0),
                'max_drawdown': None if history_snapshot is None else float(history_snapshot.get('max_drawdown', 0.0)),
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
        current_capital_after_exit = None if current_capital_before_event is None else float(current_capital_before_event) + float(pnl_realized)
        completed_trade_snapshot = _build_completed_trade_snapshot(
            stats_index,
            current_date,
            params,
            current_capital_after_exit,
            overall_max_drawdown,
        )
        total_pnl = float(position.get('realized_pnl', pnl_realized))
        full_entry_capital = _resolve_full_entry_capital(position, prev_qty)
        total_return_pct = (total_pnl / full_entry_capital * 100.0) if full_entry_capital > 0 else 0.0
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
                'current_capital': current_capital_after_exit,
                'pnl_value': float(final_leg_pnl),
                'total_pnl': float(total_pnl),
                'pnl_pct': float(total_return_pct),
                'sell_capital': float(sell_net_price * final_exit_qty),
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

    return position, pnl_realized


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
    final_leg_pnl = (sell_net_price - position['entry']) * position['qty']
    total_pnl = float(position.get('realized_pnl', 0.0) + final_leg_pnl)
    current_capital_after_exit = None if current_capital_before_event is None else float(current_capital_before_event) + float(final_leg_pnl)
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
            'current_capital': current_capital_after_exit,
            'pnl_value': float(final_leg_pnl),
            'total_pnl': float(total_pnl),
            'pnl_pct': float(total_return_pct),
            'sell_capital': float(sell_net_price * position['qty']),
            'payoff_ratio': None if completed_trade_snapshot is None else float(completed_trade_snapshot.get('payoff_ratio', 0.0)),
            'win_rate': None if completed_trade_snapshot is None else float(completed_trade_snapshot.get('win_rate', 0.0)),
            'expected_value': None if completed_trade_snapshot is None else float(completed_trade_snapshot.get('expected_value', 0.0)),
            'trade_count': _resolve_completed_trade_count(completed_trade_snapshot, include_current_round_trip=False),
            'max_drawdown': None if completed_trade_snapshot is None else float(completed_trade_snapshot.get('max_drawdown', 0.0)),
        },
    )
