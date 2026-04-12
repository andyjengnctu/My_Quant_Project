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
    evaluate_history_candidate_metrics,
    is_extended_signal_orderable_for_day,
)


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

    if total_pnl_milli > 0:
        full_wins += 1
        total_profit_milli += total_pnl_milli
        total_r_win += trade_r_mult
    else:
        total_loss_milli += abs(total_pnl_milli)
        total_r_loss += abs(trade_r_mult)

    current_capital_milli += sell_ledger['net_sell_total_milli']
    current_equity_milli = current_capital_milli
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
    }


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
    had_open_position_at_end,
    active_extended_signal,
    end_position_qty,
    avg_bars_held,
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
    resolved_ticker = ticker or (active_extended_signal or {}).get("ticker")
    resolved_security_profile = security_profile or (active_extended_signal or {}).get("security_profile")
    buy_limit = adjust_long_buy_limit(close_last, ticker=resolved_ticker, security_profile=resolved_security_profile) if buy_next_day else 0.0
    stop_loss = calc_initial_stop_from_reference(close_last, atr_last, params, ticker=resolved_ticker, security_profile=resolved_security_profile) if buy_next_day else 0.0
    tp_price = calc_frozen_target_price(close_last, stop_loss, ticker=resolved_ticker, security_profile=resolved_security_profile) if buy_next_day else 0.0
    current_position = int(end_position_qty)
    score = (total_net_profit_pct / trade_count) if trade_count > 0 else 0.0

    extended_candidate_today = None
    extended_orderable_today = False
    if active_extended_signal is not None:
        sizing_capital = resolve_single_backtest_sizing_capital(params, current_capital)
        resolved_ticker = resolved_ticker or active_extended_signal.get("ticker")
        extended_candidate_today = build_extended_candidate_plan_from_signal(
            active_extended_signal,
            sizing_capital,
            params,
            ticker=resolved_ticker,
            security_profile=resolved_security_profile,
            trade_date=final_date,
        )
        extended_orderable_today = is_extended_signal_orderable_for_day(
            active_extended_signal,
            extended_candidate_today,
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
        'activeExtendedSignal': active_extended_signal is not None,
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
        'current_position': current_position,
        'score': score,
    }
    return stats_dict
