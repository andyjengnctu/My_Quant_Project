from core.price_utils import (
    adjust_long_buy_limit,
    adjust_long_sell_fill_price,
    adjust_long_stop_price,
    calc_net_sell_price,
)
from core.trade_plans import (
    build_extended_candidate_plan_from_signal,
    evaluate_history_candidate_metrics,
)


def finalize_open_position_at_end(
    *,
    position,
    final_close,
    final_date,
    current_capital,
    current_equity,
    peak_capital,
    max_drawdown_pct,
    trade_count,
    full_wins,
    total_profit,
    total_loss,
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
            'current_capital': current_capital,
            'current_equity': current_equity,
            'peak_capital': peak_capital,
            'max_drawdown_pct': max_drawdown_pct,
            'trade_count': trade_count,
            'full_wins': full_wins,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'total_r_multiple': total_r_multiple,
            'total_r_win': total_r_win,
            'total_r_loss': total_r_loss,
            'had_open_position_at_end': had_open_position_at_end,
            'end_position_qty': end_position_qty,
            'trade_logs': trade_logs,
        }

    exec_price = adjust_long_sell_fill_price(final_close)
    net_price = calc_net_sell_price(exec_price, position['qty'], params)
    pnl = (net_price - position['entry']) * position['qty']
    total_pnl = position['realized_pnl'] + pnl
    trade_r_mult = total_pnl / position['initial_risk_total'] if position['initial_risk_total'] > 0 else 0.0

    total_r_multiple += trade_r_mult
    trade_count += 1

    if return_logs:
        trade_logs.append({'exit_date': final_date, 'pnl': total_pnl, 'r_mult': trade_r_mult})

    if total_pnl > 0:
        full_wins += 1
        total_profit += total_pnl
        total_r_win += trade_r_mult
    else:
        total_loss += abs(total_pnl)
        total_r_loss += abs(trade_r_mult)

    current_capital += pnl
    current_equity = current_capital
    peak_capital = max(peak_capital, current_equity)
    current_drawdown_pct = ((peak_capital - current_equity) / peak_capital) * 100 if peak_capital > 0 else 0.0
    max_drawdown_pct = max(max_drawdown_pct, current_drawdown_pct)
    position['qty'] = 0

    return {
        'position': position,
        'current_capital': current_capital,
        'current_equity': current_equity,
        'peak_capital': peak_capital,
        'max_drawdown_pct': max_drawdown_pct,
        'trade_count': trade_count,
        'full_wins': full_wins,
        'total_profit': total_profit,
        'total_loss': total_loss,
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
    peak_capital,
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
    del peak_capital
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

    final_equity = current_equity
    total_net_profit_pct = ((final_equity - params.initial_capital) / params.initial_capital) * 100
    score = total_net_profit_pct / trade_count if trade_count > 0 else 0

    is_setup_today = buy_condition_last and (not had_open_position_at_end)
    buy_limit_today = adjust_long_buy_limit(close_last + atr_last * params.atr_buy_tol) if is_setup_today else float('nan')
    stop_loss_today = adjust_long_stop_price(buy_limit_today - atr_last * params.atr_times_init) if is_setup_today else float('nan')

    extended_candidate_today = None
    if (not had_open_position_at_end) and active_extended_signal is not None:
        sizing_cap = current_capital if getattr(params, 'use_compounding', True) else params.initial_capital
        extended_candidate_today = build_extended_candidate_plan_from_signal(active_extended_signal, close_last, sizing_cap, params)

    return {
        'asset_growth': total_net_profit_pct,
        'trade_count': trade_count,
        'missed_buys': missed_buy_count,
        'missed_sells': missed_sell_count,
        'score': score,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'payoff_ratio': payoff_ratio,
        'expected_value': expected_value,
        'max_drawdown': max_drawdown_pct,
        'is_candidate': is_candidate,
        'is_setup_today': is_setup_today,
        'buy_limit': buy_limit_today,
        'stop_loss': stop_loss_today,
        'extended_candidate_today': extended_candidate_today,
        'current_position': end_position_qty,
        'avg_bars_held': avg_bars_held,
    }
