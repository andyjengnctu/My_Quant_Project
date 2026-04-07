import numpy as np
import pandas as pd

from core.backtest_core import run_v16_backtest
from core.capital_policy import resolve_single_backtest_sizing_capital
from core.entry_plans import build_normal_candidate_plan, build_normal_entry_plan
from core.extended_signals import build_extended_candidate_plan_from_signal
from core.history_filters import evaluate_history_candidate_metrics
from core.price_utils import calc_entry_price, calc_frozen_target_price, calc_net_sell_price
from core.portfolio_fast_data import build_trade_stats_index
from core.signal_utils import generate_signals
from tools.debug.charting import (
    create_debug_chart_context,
    record_active_levels,
    record_signal_annotation,
    set_chart_status_box,
    set_chart_summary_box,
    set_chart_future_preview,
)
from tools.debug.entry_flow import process_debug_entry_for_day
from tools.debug.exit_flow import append_debug_forced_closeout, process_debug_position_step
from tools.debug.history_snapshot import build_pit_history_snapshot
from tools.debug.reporting import finalize_debug_analysis


# 保留 stable patch seam 供 synthetic contract / GUI coverage 路徑覆寫 PIT history snapshot。
_build_pit_history_snapshot = build_pit_history_snapshot


def _extract_precomputed_signals(df):
    required_columns = {'ATR', 'is_setup', 'ind_sell_signal', 'buy_limit'}
    if not required_columns.issubset(df.columns):
        return None
    return (
        df['ATR'].to_numpy(copy=False),
        df['is_setup'].to_numpy(copy=False),
        df['ind_sell_signal'].to_numpy(copy=False),
        df['buy_limit'].to_numpy(copy=False),
    )


def _resolve_active_tp_half(position):
    if position.get('qty', 0) <= 0:
        return np.nan
    if position.get('sold_half', False):
        return np.nan
    return position.get('tp_half', np.nan)


def _resolve_chart_tp_line(position):
    if position.get('qty', 0) <= 0:
        return np.nan
    if position.get('_debug_tp_preview_done', False):
        return np.nan
    return position.get('tp_half', np.nan)


def _apply_chart_future_preview_from_plan(chart_context, preview_plan):
    if chart_context is None or preview_plan is None:
        return False
    preview_tp = preview_plan.get('target_price', calc_frozen_target_price(preview_plan['limit_price'], preview_plan['init_sl']))
    set_chart_future_preview(
        chart_context,
        stop_price=preview_plan['init_sl'],
        tp_half_price=preview_tp,
        limit_price=preview_plan['limit_price'],
        entry_price=np.nan,
    )
    return True


def _record_buy_signal_annotation(*, chart_context, signal_date, signal_low, entry_plan, history_snapshot, params):
    if chart_context is None:
        return
    meta = dict(history_snapshot or {})
    current_capital = meta.get('current_capital')
    detail_lines = []
    if current_capital is not None:
        detail_lines.append(f"資金: {float(current_capital):,.0f}")
    if entry_plan is None:
        detail_lines.append('本次資金不足，無法掛單')
    else:
        tp_line = calc_frozen_target_price(entry_plan['limit_price'], entry_plan['init_sl'])
        buy_capital = calc_entry_price(entry_plan['limit_price'], entry_plan['qty'], params) * entry_plan['qty']
        detail_lines.extend([
            f"股數: {int(entry_plan['qty']):,}",
            f"金額: {buy_capital:,.0f}",
            f"停利: {tp_line:.2f}",
            f"限價: {entry_plan['limit_price']:.2f}",
            f"停損: {entry_plan['init_sl']:.2f}",
        ])
        meta.update({
            'current_capital': None if current_capital is None else float(current_capital),
            'tp_price': float(tp_line),
            'limit_price': float(entry_plan['limit_price']),
            'stop_price': float(entry_plan['init_sl']),
            'entry_price': float(entry_plan['limit_price']),
            'buy_capital': float(buy_capital),
            'qty': int(entry_plan['qty']),
        })
    record_signal_annotation(
        chart_context,
        current_date=signal_date,
        signal_type='buy',
        anchor_price=signal_low,
        title='買訊',
        detail_lines=detail_lines,
        meta=meta,
    )


def _record_sell_signal_annotation(*, chart_context, signal_date, signal_low, signal_close, position, history_snapshot, params):
    if chart_context is None or position.get('qty', 0) <= 0:
        return
    entry_price = float(position.get('entry', signal_close))
    signal_trade_pct = ((float(signal_close) - entry_price) / entry_price * 100.0) if entry_price > 0 else 0.0
    record_signal_annotation(
        chart_context,
        current_date=signal_date,
        signal_type='sell',
        anchor_price=signal_low,
        title='賣訊',
        detail_lines=[],
        meta={'profit_pct': float(signal_trade_pct), 'max_drawdown': float(history_snapshot.get('max_drawdown', 0.0))},
    )


def _apply_chart_sidebars(*, chart_context, stats_dict, sell_condition):
    if chart_context is None or stats_dict is None:
        return
    summary_lines = [
        f"資產成長: {stats_dict.get('asset_growth', 0.0):.1f}%",
        f"交易次數: {int(stats_dict.get('trade_count', 0) or 0)}",
        f"錯失買點: {int(stats_dict.get('missed_buys', 0) or 0)}次",
        f"單筆報酬: {stats_dict.get('score', 0.0):.2f}%",
        '',
        f"勝率: {stats_dict.get('win_rate', 0.0):.2f}%",
        f"風報比: {stats_dict.get('payoff_ratio', 0.0):.2f}",
        f"期望值: {stats_dict.get('expected_value', 0.0):.2f} R",
        f"最大回撤: {stats_dict.get('max_drawdown', 0.0):.2f}%",
    ]
    set_chart_summary_box(chart_context, summary_lines=summary_lines)
    has_raw_buy_signal = bool(stats_dict.get('is_setup_today')) or stats_dict.get('extended_candidate_today') is not None
    sell_signal_today = bool(sell_condition[-1]) if len(sell_condition) > 0 else False
    history_gate_ok = bool(stats_dict.get('is_candidate', False))
    if sell_signal_today:
        primary_signal_line = "出現賣訊"
    elif has_raw_buy_signal:
        primary_signal_line = "出現買入訊號"
    else:
        primary_signal_line = "無買入訊號"
    status_lines = [
        primary_signal_line,
        '符合歷史績效' if history_gate_ok else '未符合歷史績效',
    ]
    set_chart_status_box(chart_context, status_lines=status_lines, ok=history_gate_ok)


def run_debug_analysis(df, ticker, params, output_dir, colors, export_excel=True, export_chart=True, return_chart_payload=False, verbose=True, precomputed_signals=None):
    """以正式核心邏輯為準，輸出可讀交易明細與 K 線圖 artifact。"""
    h = df['High'].to_numpy(dtype=np.float64, copy=False)
    l = df['Low'].to_numpy(dtype=np.float64, copy=False)
    c = df['Close'].to_numpy(dtype=np.float64, copy=False)
    o = df['Open'].to_numpy(dtype=np.float64, copy=False)
    v = df['Volume'].to_numpy(dtype=np.float64, copy=False)
    dates = df.index
    if precomputed_signals is None:
        precomputed_signals = _extract_precomputed_signals(df)
    if precomputed_signals is None:
        precomputed_signals = generate_signals(df, params)
    atr_main, buy_condition, sell_condition, buy_limits = precomputed_signals
    stats_dict, standalone_logs = run_v16_backtest(df.copy(), params, return_logs=True, precomputed_signals=precomputed_signals)
    stats_index = build_trade_stats_index(standalone_logs)
    position = {'qty': 0}
    active_extended_signal = None
    current_capital = params.initial_capital
    trade_logs = []
    chart_context = create_debug_chart_context(df) if (export_chart or return_chart_payload) else None
    for j in range(1, len(c)):
        if np.isnan(atr_main[j - 1]):
            continue
        pos_qty_start_of_bar = position['qty']
        signal_date = dates[j - 1]
        signal_history_snapshot = _build_pit_history_snapshot(stats_index, signal_date, params, current_capital, stats_dict.get('max_drawdown', 0.0))
        if chart_context is not None and buy_condition[j - 1] and pos_qty_start_of_bar == 0:
            sizing_cap = resolve_single_backtest_sizing_capital(params, current_capital)
            entry_plan_preview = build_normal_entry_plan(buy_limits[j - 1], atr_main[j - 1], sizing_cap, params)
            _record_buy_signal_annotation(
                chart_context=chart_context,
                signal_date=signal_date,
                signal_low=l[j - 1],
                entry_plan=entry_plan_preview,
                history_snapshot=signal_history_snapshot,
                params=params,
            )
        if chart_context is not None and pos_qty_start_of_bar > 0 and sell_condition[j - 1]:
            _record_sell_signal_annotation(
                chart_context=chart_context,
                signal_date=signal_date,
                signal_low=l[j - 1],
                signal_close=c[j - 1],
                position=position,
                history_snapshot=signal_history_snapshot,
                params=params,
            )
        if pos_qty_start_of_bar > 0:
            position, pnl_realized = process_debug_position_step(
                position=position,
                atr_prev=atr_main[j - 1],
                sell_condition_prev=sell_condition[j - 1],
                close_prev=c[j - 1],
                t_open=o[j],
                t_high=h[j],
                t_low=l[j],
                t_close=c[j],
                t_volume=v[j],
                current_date=dates[j],
                params=params,
                trade_logs=trade_logs,
                chart_context=chart_context,
                history_snapshot=signal_history_snapshot,
                stats_index=stats_index,
                current_capital_before_event=current_capital,
                overall_max_drawdown=stats_dict.get('max_drawdown', 0.0),
            )
            current_capital += pnl_realized
        sizing_cap = resolve_single_backtest_sizing_capital(params, current_capital)
        position, active_extended_signal = process_debug_entry_for_day(
            position=position,
            pos_qty_start_of_bar=pos_qty_start_of_bar,
            active_extended_signal=active_extended_signal,
            buy_condition_prev=buy_condition[j - 1],
            buy_limit_prev=buy_limits[j - 1],
            atr_prev=atr_main[j - 1],
            close_prev=c[j - 1],
            sizing_cap=sizing_cap,
            t_open=o[j],
            t_high=h[j],
            t_low=l[j],
            t_close=c[j],
            t_volume=v[j],
            current_date=dates[j],
            params=params,
            trade_logs=trade_logs,
            chart_context=chart_context,
            current_capital=current_capital,
        )
        if chart_context is not None and position['qty'] > 0:
            record_active_levels(
                chart_context,
                current_date=dates[j],
                stop_price=position.get('sl', np.nan),
                tp_half_price=_resolve_chart_tp_line(position),
                limit_price=position.get('limit_price', np.nan),
                entry_price=position.get('pure_buy_price', np.nan),
            )
            tp_half_value = position.get('tp_half', np.nan)
            if position.get('qty', 0) > 0 and not pd.isna(tp_half_value) and h[j] >= tp_half_value:
                position['_debug_tp_preview_done'] = True
    if len(c) > 0:
        latest_history_snapshot = _build_pit_history_snapshot(stats_index, dates[-1], params, current_capital, stats_dict.get('max_drawdown', 0.0))
        if chart_context is not None and position.get('qty', 0) == 0 and bool(buy_condition[-1]):
            latest_sizing_cap = resolve_single_backtest_sizing_capital(params, current_capital)
            latest_entry_plan_preview = build_normal_candidate_plan(buy_limits[-1], atr_main[-1], latest_sizing_cap, params) if not np.isnan(atr_main[-1]) else None
            _record_buy_signal_annotation(
                chart_context=chart_context,
                signal_date=dates[-1],
                signal_low=l[-1],
                entry_plan=latest_entry_plan_preview,
                history_snapshot=latest_history_snapshot,
                params=params,
            )
            if latest_history_snapshot.get('is_candidate', False):
                _apply_chart_future_preview_from_plan(chart_context, latest_entry_plan_preview)
        elif chart_context is not None and position.get('qty', 0) == 0 and active_extended_signal is not None:
            latest_sizing_cap = resolve_single_backtest_sizing_capital(params, current_capital)
            latest_extended_preview = build_extended_candidate_plan_from_signal(active_extended_signal, latest_sizing_cap, params)
            if latest_history_snapshot.get('is_candidate', False):
                _apply_chart_future_preview_from_plan(chart_context, latest_extended_preview)
        if chart_context is not None and position.get('qty', 0) > 0 and bool(sell_condition[-1]):
            _record_sell_signal_annotation(
                chart_context=chart_context,
                signal_date=dates[-1],
                signal_low=l[-1],
                signal_close=c[-1],
                position=position,
                history_snapshot=latest_history_snapshot,
                params=params,
            )
    if position['qty'] > 0:
        position['close_price'] = c[-1]
        append_debug_forced_closeout(
            position=position,
            current_date=dates[-1],
            atr_last=atr_main[-1] if len(atr_main) > 0 else np.nan,
            params=params,
            trade_logs=trade_logs,
            chart_context=chart_context,
            current_capital_before_event=current_capital,
            stats_index=stats_index,
            overall_max_drawdown=stats_dict.get('max_drawdown', 0.0),
        )
    _apply_chart_sidebars(chart_context=chart_context, stats_dict=stats_dict, sell_condition=sell_condition)
    return finalize_debug_analysis(
        trade_logs=trade_logs,
        ticker=ticker,
        output_dir=output_dir,
        colors=colors,
        export_excel=export_excel,
        export_chart=export_chart,
        return_chart_payload=return_chart_payload,
        verbose=verbose,
        price_df=df,
        chart_context=chart_context,
    )


def run_debug_backtest(df, ticker, params, output_dir, colors, export_excel=True, verbose=True, precomputed_signals=None):
    result = run_debug_analysis(
        df=df,
        ticker=ticker,
        params=params,
        output_dir=output_dir,
        colors=colors,
        export_excel=export_excel,
        export_chart=False,
        verbose=verbose,
        precomputed_signals=precomputed_signals,
    )
    return result['trade_logs_df']
