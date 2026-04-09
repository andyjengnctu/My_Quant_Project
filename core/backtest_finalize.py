from core.exact_accounting import build_sell_ledger_from_price, milli_to_money
from core.price_utils import (
    adjust_long_buy_limit,
    adjust_long_sell_fill_price,
    adjust_long_stop_price,
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

    exec_price = adjust_long_sell_fill_price(final_close)
    sell_ledger = build_sell_ledger_from_price(exec_price, position['qty'], params)
    pnl_milli = sell_ledger['net_sell_total_milli'] - position['remaining_cost_basis_milli']
    total_pnl_milli = position['realized_pnl_milli'] + pnl_milli
    total_pnl = milli_to_money(total_pnl_milli)
    trade_r_mult = total_pnl / position['initial_risk_total'] if position['initial_risk_total'] > 0 else 0.0

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

    current_capital_milli += pnl_milli
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

    stats_dict = {
        'currentCapital': current_capital,
        'currentEquity': current_equity,
        'totalNetProfit': current_capital - params.initial_capital,
        'totalNetProfitPct': ((current_equity / params.initial_capital) - 1) * 100 if params.initial_capital > 0 else 0.0,
        'maxDrawdownPct': max_drawdown_pct,
        'tradeCount': trade_count,
        'winRate': win_rate,
        'profitFactor': expected_value,
        'payoffRatio': payoff_ratio,
        'missedBuyCount': missed_buy_count,
        'missedSellCount': missed_sell_count,
        'buyNextDay': bool(buy_condition_last),
        'buyPrice': adjust_long_buy_limit(close_last) if buy_condition_last else 0.0,
        'sellPrice': adjust_long_stop_price(close_last - atr_last * params.atr_times_init) if buy_condition_last else 0.0,
        'tpPrice': close_last + (close_last - (close_last - atr_last * params.atr_times_init)) if buy_condition_last else 0.0,
        'is_candidate': is_candidate,
        'history_ev': expected_value,
        'history_win_rate': _history_win_rate,
        'history_trade_count': _history_trade_count,
        'hasOpenPositionAtEnd': had_open_position_at_end,
        'activeExtendedSignal': active_extended_signal is not None,
        'endPositionQty': end_position_qty,
        'avgBarsHeld': avg_bars_held,
    }
    return stats_dict
