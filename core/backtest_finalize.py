from core.exact_accounting import build_sell_ledger_from_price, calc_ratio_from_milli, milli_to_money
from core.price_utils import (
    adjust_long_buy_limit,
    adjust_long_sell_fill_price,
    calc_frozen_target_price,
    calc_initial_stop_from_reference,
)
from core.capital_policy import resolve_single_backtest_sizing_capital
from core.trade_plans import (
    build_extended_candidate_plan_from_signal,
    build_extended_tbd_candidate_plan_from_state,
    evaluate_history_candidate_metrics,
    has_extended_signal_orderable_context_for_day,
    has_extended_tbd_orderable_context_for_day,
)
from core.extended_signals import is_extended_tbd_display_day, is_extended_tbd_shadow_alive


def finalize_open_position_at_end(
    *,
    position,
    final_close,
    final_date,
    current_capital_milli,
    current_equity_milli,
    peak_capital_milli,
    max_drawdown_pct,
    trade_count,
    full_wins,
    total_profit_milli,
    total_loss_milli,
    total_r_multiple,
    total_r_win,
    total_r_loss,
    trade_logs,
    return_logs,
    params,
    ticker=None,
    collect_stats=True,
):
    end_position_qty = position['qty']
    had_open_position_at_end = end_position_qty > 0

    if not had_open_position_at_end:
        return {
            'position': position,
            'current_capital_milli': current_capital_milli,
            'current_equity_milli': current_equity_milli,
            'peak_capital_milli': peak_capital_milli,
            'max_drawdown_pct': max_drawdown_pct,
            'trade_count': trade_count,
            'full_wins': full_wins,
            'total_profit_milli': total_profit_milli,
            'total_loss_milli': total_loss_milli,
            'total_r_multiple': total_r_multiple,
            'total_r_win': total_r_win,
            'total_r_loss': total_r_loss,
            'had_open_position_at_end': had_open_position_at_end,
            'end_position_qty': end_position_qty,
            'trade_logs': trade_logs,
            'final_trade_exit_date': None,
            'final_trade_pnl': None,
            'final_trade_r_mult': None,
        }

    resolved_ticker = ticker or position.get("ticker")
    exec_price = adjust_long_sell_fill_price(final_close, ticker=resolved_ticker)
    sell_ledger = build_sell_ledger_from_price(
        exec_price,
        position['qty'],
        params,
        ticker=resolved_ticker,
        security_profile=position.get('security_profile'),
        trade_date=final_date,
    )
    pnl_milli = sell_ledger['net_sell_total_milli'] - position['remaining_cost_basis_milli']
    total_pnl_milli = position['realized_pnl_milli'] + pnl_milli
    total_pnl = milli_to_money(total_pnl_milli)
    trade_r_mult = calc_ratio_from_milli(total_pnl_milli, position.get('initial_risk_total_milli', 0))

    total_r_multiple += trade_r_mult
    trade_count += 1

    if return_logs:
        trade_logs.append({'exit_date': final_date, 'pnl': total_pnl, 'r_mult': trade_r_mult})

    if collect_stats:
        if total_pnl_milli > 0:
            full_wins += 1
            total_profit_milli += total_pnl_milli
            total_r_win += trade_r_mult
        else:
            total_loss_milli += abs(total_pnl_milli)
            total_r_loss += abs(trade_r_mult)

    current_capital_milli += sell_ledger['net_sell_total_milli']
    current_equity_milli = current_capital_milli
    if collect_stats:
        peak_capital_milli = max(peak_capital_milli, current_equity_milli)
        current_drawdown_pct = ((peak_capital_milli - current_equity_milli) / peak_capital_milli) * 100 if peak_capital_milli > 0 else 0.0
        max_drawdown_pct = max(max_drawdown_pct, current_drawdown_pct)
    position['qty'] = 0
    position['remaining_cost_basis_milli'] = 0

    return {
        'position': position,
        'current_capital_milli': current_capital_milli,
        'current_equity_milli': current_equity_milli,
        'peak_capital_milli': peak_capital_milli,
        'max_drawdown_pct': max_drawdown_pct,
        'trade_count': trade_count,
        'full_wins': full_wins,
        'total_profit_milli': total_profit_milli,
        'total_loss_milli': total_loss_milli,
        'total_r_multiple': total_r_multiple,
        'total_r_win': total_r_win,
        'total_r_loss': total_r_loss,
        'had_open_position_at_end': had_open_position_at_end,
        'end_position_qty': end_position_qty,
        'trade_logs': trade_logs,
        'final_trade_exit_date': final_date,
        'final_trade_pnl': total_pnl,
        'final_trade_r_mult': trade_r_mult,
    }


def _is_valid_preview_price(value):
    try:
        return value is not None and value == value and value > 0
    except TypeError:
        return False


def build_backtest_stats(
    *,
    params,
    current_capital,
    current_equity,
    max_drawdown_pct,
    trade_count,
    full_wins,
    total_profit,
    total_loss,
    total_r_multiple,
    total_r_win,
    total_r_loss,
    missed_buy_count,
    missed_sell_count,
    buy_condition_last,
    atr_last,
    close_last,
    buy_limit_last=None,
    low_last=None,
    had_open_position_at_end=False,
    active_extended_signal=None,
    active_extended_signal_tbd=None,
    end_position_qty=0,
    avg_bars_held=0,
    final_date=None,
    ticker=None,
    security_profile=None,
):
    win_rate = (full_wins / trade_count * 100) if trade_count > 0 else 0
    avg_win = total_profit / full_wins if full_wins > 0 else 0
    loss_count = trade_count - full_wins
    avg_loss = total_loss / loss_count if loss_count > 0 else 0
    payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else (99.9 if avg_win > 0 else 0.0)

    is_candidate, expected_value, _history_win_rate, _history_trade_count = evaluate_history_candidate_metrics(
        trade_count,
        full_wins,
        total_r_multiple,
        total_r_win,
        total_r_loss,
        params,
    )

    total_net_profit = current_equity - params.initial_capital
    total_net_profit_pct = ((current_equity / params.initial_capital) - 1) * 100 if params.initial_capital > 0 else 0.0
    buy_next_day = bool(buy_condition_last)

    resolved_ticker = ticker or (active_extended_signal_tbd or {}).get("ticker") or (active_extended_signal or {}).get("ticker")
    resolved_security_profile = security_profile or (active_extended_signal or {}).get("security_profile")
    if resolved_security_profile is None:
        resolved_security_profile = (active_extended_signal_tbd or {}).get("security_profile")
    buy_limit = 0.0
    if buy_next_day:
        if _is_valid_preview_price(buy_limit_last):
            buy_limit = float(buy_limit_last)
        else:
            fallback_raw_limit = close_last + atr_last * params.atr_buy_tol
            buy_limit = adjust_long_buy_limit(fallback_raw_limit, ticker=resolved_ticker, security_profile=resolved_security_profile)
    stop_loss = calc_initial_stop_from_reference(buy_limit, atr_last, params, ticker=resolved_ticker, security_profile=resolved_security_profile) if buy_next_day else 0.0
    tp_price = calc_frozen_target_price(buy_limit, stop_loss, ticker=resolved_ticker, security_profile=resolved_security_profile) if buy_next_day else 0.0
    current_position = int(end_position_qty)
    score = (total_net_profit_pct / trade_count) if trade_count > 0 else 0.0

    # # (AI註: scanner / workbench 的 TBD 用途是讓使用者確認『實際投組是否已持有』，
    # # 因此不可用單股路徑的 current_position / hasOpenPositionAtEnd 直接整體封鎖；
    # # 否則凡是單股已買進且仍在倉的情境，TBD 會被全部吞掉。
    has_terminal_position = bool(had_open_position_at_end) or current_position > 0
    display_extended_signal = active_extended_signal_tbd if active_extended_signal_tbd is not None else active_extended_signal
    display_extended_tbd_signal = display_extended_signal
    shadow_active_today = is_extended_tbd_shadow_alive(display_extended_tbd_signal)
    tbd_display_today = display_extended_tbd_signal is not None and is_extended_tbd_display_day(display_extended_tbd_signal, low_last)
    extended_candidate_today = None
    extended_orderable_today = False
    sizing_capital = resolve_single_backtest_sizing_capital(params, current_capital)
    if display_extended_signal is not None and not tbd_display_today:
        resolved_ticker = resolved_ticker or display_extended_signal.get("ticker")
        extended_candidate_today = build_extended_candidate_plan_from_signal(
            display_extended_signal,
            sizing_capital,
            params,
            ticker=resolved_ticker,
            security_profile=resolved_security_profile,
            trade_date=final_date,
        )
        extended_orderable_today = has_extended_signal_orderable_context_for_day(
            display_extended_signal,
            extended_candidate_today,
            close_last,
            ticker=resolved_ticker,
            security_profile=resolved_security_profile,
        )

    extended_candidate_tbd_today = None
    extended_tbd_orderable_today = False
    if display_extended_tbd_signal is not None and tbd_display_today:
        resolved_ticker = resolved_ticker or display_extended_tbd_signal.get("ticker")
        if shadow_active_today:
            extended_candidate_tbd_today = build_extended_tbd_candidate_plan_from_state(
                display_extended_tbd_signal,
                sizing_capital,
                params,
                ticker=resolved_ticker,
                security_profile=resolved_security_profile,
                trade_date=final_date,
            )
        else:
            extended_candidate_tbd_today = build_extended_candidate_plan_from_signal(
                display_extended_tbd_signal,
                sizing_capital,
                params,
                ticker=resolved_ticker,
                security_profile=resolved_security_profile,
                trade_date=final_date,
            )
        extended_tbd_orderable_today = has_extended_tbd_orderable_context_for_day(
            display_extended_tbd_signal,
            extended_candidate_tbd_today,
            close_last,
            ticker=resolved_ticker,
            security_profile=resolved_security_profile,
        ) if shadow_active_today else has_extended_signal_orderable_context_for_day(
            display_extended_tbd_signal,
            extended_candidate_tbd_today,
            close_last,
            ticker=resolved_ticker,
            security_profile=resolved_security_profile,
        )

    stats_dict = {
        'currentCapital': current_capital,
        'currentEquity': current_equity,
        'totalNetProfit': total_net_profit,
        'totalNetProfitPct': total_net_profit_pct,
        'maxDrawdownPct': max_drawdown_pct,
        'tradeCount': trade_count,
        'winRate': win_rate,
        'profitFactor': expected_value,
        'payoffRatio': payoff_ratio,
        'missedBuyCount': missed_buy_count,
        'missedSellCount': missed_sell_count,
        'buyNextDay': buy_next_day,
        'buyPrice': buy_limit,
        'sellPrice': stop_loss,
        'tpPrice': tp_price,
        'is_candidate': is_candidate,
        'history_ev': expected_value,
        'history_win_rate': _history_win_rate,
        'history_trade_count': _history_trade_count,
        'hasOpenPositionAtEnd': had_open_position_at_end,
        'activeExtendedSignal': display_extended_signal is not None,
        'activeExtendedSignalTBD': display_extended_tbd_signal is not None and tbd_display_today,
        'endPositionQty': end_position_qty,
        'avgBarsHeld': avg_bars_held,
        'trade_count': trade_count,
        'win_rate': win_rate,
        'expected_value': expected_value,
        'asset_growth': total_net_profit_pct,
        'max_drawdown': max_drawdown_pct,
        'payoff_ratio': payoff_ratio,
        'missed_buys': missed_buy_count,
        'missed_sells': missed_sell_count,
        'is_setup_today': buy_next_day,
        'buy_limit': buy_limit,
        'stop_loss': stop_loss,
        'tp_price': tp_price,
        'extended_candidate_today': extended_candidate_today,
        'extended_orderable_today': extended_orderable_today,
        'extended_candidate_tbd_today': extended_candidate_tbd_today,
        'extended_tbd_orderable_today': extended_tbd_orderable_today,
        'current_position': current_position,
        'score': score,
    }
    return stats_dict
