import numpy as np
import pandas as pd

from core.v16_config import EV_CALC_METHOD, V16StrategyParams
from core.v16_price_utils import (
    adjust_long_buy_limit,
    adjust_long_sell_fill_price,
    adjust_long_stop_price,
    adjust_long_target_price,
    calc_entry_price,
    calc_half_take_profit_sell_qty,
    calc_net_sell_price,
    calc_reference_candidate_qty,
    can_execute_half_take_profit,
    get_exit_sell_block_reason,
)
from core.v16_signal_utils import generate_signals
from core.v16_trade_plans import (
    build_cash_capped_entry_plan,
    build_extended_candidate_plan_from_signal,
    build_extended_entry_plan_from_signal,
    build_normal_candidate_plan,
    build_normal_entry_plan,
    create_signal_tracking_state,
    evaluate_history_candidate_metrics,
    execute_pre_market_entry_plan,
    resize_candidate_plan_to_capital,
    should_clear_extended_signal,
)


def execute_bar_step(position, y_atr, y_ind_sell, y_close, t_open, t_high, t_low, t_close, t_volume, params):
    freed_cash, pnl_realized = 0.0, 0.0
    events = []

    if position['qty'] <= 0:
        return position, freed_cash, pnl_realized, events

    if y_close > position['pure_buy_price'] + (y_atr * params.atr_times_trail):
        new_trail = adjust_long_stop_price(y_close - (y_atr * params.atr_times_trail))
        position['trailing_stop'] = max(position.get('trailing_stop', 0.0), new_trail)
        position['sl'] = max(position['initial_stop'], position['trailing_stop'])

    # # (AI註: y_ind_sell 是 T-1 收盤後已知、T 日盤前即可決定的賣出指令，
    # # (AI註: 必須優先於 T 日盤中的 TP_HALF / STOP 判斷，避免出現不可能的事件序列)
    if y_ind_sell:
        sell_block_reason = get_exit_sell_block_reason(t_open, t_high, t_low, t_close, t_volume, y_close)
        if sell_block_reason is None:
            exec_price = adjust_long_sell_fill_price(t_open)
            net_price = calc_net_sell_price(exec_price, position['qty'], params)
            freed_cash += net_price * position['qty']
            pnl = (net_price - position['entry']) * position['qty']
            pnl_realized += pnl
            position['realized_pnl'] += pnl
            position['qty'] = 0
            events.append('IND_SELL')
        else:
            events.extend(['MISSED_SELL', sell_block_reason])

        return position, freed_cash, pnl_realized, events

    is_stop_hit = t_low <= position['sl']
    half_sell_qty = calc_half_take_profit_sell_qty(position['qty'], params.tp_percent)
    is_tp_hit = t_high >= position['tp_half'] and not position['sold_half'] and half_sell_qty > 0

    # # (AI註: 同棒同時碰停損與半倉停利時，維持最壞情境，優先視為停損)
    if is_stop_hit and is_tp_hit:
        is_tp_hit = False

    if is_tp_hit and not (pd.isna(t_volume) or t_volume <= 0):
        exec_price = adjust_long_sell_fill_price(max(position['tp_half'], t_open))
        sell_qty = half_sell_qty
        net_price = calc_net_sell_price(exec_price, sell_qty, params)
        freed_cash = net_price * sell_qty
        pnl = (net_price - position['entry']) * sell_qty
        pnl_realized += pnl
        position['realized_pnl'] += pnl
        position['qty'] -= sell_qty
        position['sold_half'] = True
        events.append('TP_HALF')

    if is_stop_hit:
        sell_block_reason = get_exit_sell_block_reason(t_open, t_high, t_low, t_close, t_volume, y_close)
        if sell_block_reason is None:
            exec_price = adjust_long_sell_fill_price(min(position['sl'], t_open))
            net_price = calc_net_sell_price(exec_price, position['qty'], params)
            freed_cash += net_price * position['qty']
            pnl = (net_price - position['entry']) * position['qty']
            pnl_realized += pnl
            position['realized_pnl'] += pnl
            position['qty'] = 0
            events.append('STOP')
        else:
            events.extend(['MISSED_SELL', sell_block_reason])

    return position, freed_cash, pnl_realized, events

def run_v16_backtest(df, params=None, return_logs=False, precomputed_signals=None):
    if params is None:
        params = V16StrategyParams()
    H = df['High'].to_numpy(dtype=np.float64, copy=False)
    L = df['Low'].to_numpy(dtype=np.float64, copy=False)
    C = df['Close'].to_numpy(dtype=np.float64, copy=False)
    O = df['Open'].to_numpy(dtype=np.float64, copy=False)
    V = df['Volume'].to_numpy(dtype=np.float64, copy=False)
    Dates = df.index
    if precomputed_signals is None:
        ATR_main, buyCondition, sellCondition, buy_limits = generate_signals(df, params)
    else:
        ATR_main, buyCondition, sellCondition, buy_limits = precomputed_signals

    position = {'qty': 0}
    active_extended_signal = None
    currentCapital = params.initial_capital
    tradeCount, fullWins, missedBuyCount, missedSellCount = 0, 0, 0, 0
    totalProfit, totalLoss = 0.0, 0.0
    peakCapital, maxDrawdownPct = currentCapital, 0.0
    total_r_multiple, total_r_win, total_r_loss, total_bars_held = 0.0, 0.0, 0.0, 0
    trade_logs = []
    currentEquity = currentCapital

    for j in range(1, len(C)):
        if np.isnan(ATR_main[j-1]):
            continue
        pos_start_of_current_bar = position['qty']

        if pos_start_of_current_bar > 0:
            total_bars_held += 1
            position, freed_cash, pnl_realized, events = execute_bar_step(
                position, ATR_main[j-1], sellCondition[j-1], C[j-1],
                O[j], H[j], L[j], C[j], V[j], params
            )
            if 'STOP' in events or 'IND_SELL' in events:
                total_pnl = position['realized_pnl']
                trade_r_mult = total_pnl / position['initial_risk_total'] if position['initial_risk_total'] > 0 else 0
                total_r_multiple += trade_r_mult
                tradeCount += 1
                if return_logs:
                    trade_logs.append({'exit_date': Dates[j], 'pnl': total_pnl, 'r_mult': trade_r_mult})
                if total_pnl > 0:
                    fullWins += 1
                    totalProfit += total_pnl
                    total_r_win += trade_r_mult
                else:
                    totalLoss += abs(total_pnl)
                    total_r_loss += abs(trade_r_mult)
            elif 'MISSED_SELL' in events:
                missedSellCount += 1
            currentCapital += pnl_realized

        isSetup_prev = buyCondition[j-1] and (pos_start_of_current_bar == 0)
        buyTriggered = False
        sizing_cap = currentCapital if getattr(params, 'use_compounding', True) else params.initial_capital

        if isSetup_prev:
            signal_state = create_signal_tracking_state(buy_limits[j-1], ATR_main[j-1], params)
            if signal_state is not None:
                active_extended_signal = signal_state

            entry_plan = build_normal_entry_plan(buy_limits[j-1], ATR_main[j-1], sizing_cap, params)
            entry_result = execute_pre_market_entry_plan(
                entry_plan=entry_plan,
                t_open=O[j],
                t_high=H[j],
                t_low=L[j],
                t_close=C[j],
                t_volume=V[j],
                y_close=C[j-1],
                params=params,
                entry_type='normal',
            )
            if entry_result['filled']:
                position = entry_result['position']
                buyTriggered = True
                active_extended_signal = None
            elif entry_result['count_as_missed_buy']:
                missedBuyCount += 1

        elif active_extended_signal is not None and pos_start_of_current_bar == 0:
            entry_plan = build_extended_entry_plan_from_signal(active_extended_signal, C[j-1], sizing_cap, params)
            entry_result = execute_pre_market_entry_plan(
                entry_plan=entry_plan,
                t_open=O[j],
                t_high=H[j],
                t_low=L[j],
                t_close=C[j],
                t_volume=V[j],
                y_close=C[j-1],
                params=params,
                entry_type='extended',
            )
            if entry_result['filled']:
                position = entry_result['position']
                buyTriggered = True
                active_extended_signal = None
            elif entry_result['count_as_missed_buy']:
                missedBuyCount += 1

        if not buyTriggered and position['qty'] == 0 and should_clear_extended_signal(active_extended_signal, L[j]):
            active_extended_signal = None

        currentEquity = currentCapital
        if position['qty'] > 0:
            floating_exec_price = adjust_long_sell_fill_price(C[j])
            floatingSellNet = calc_net_sell_price(floating_exec_price, position['qty'], params)
            floatingPnL = (floatingSellNet - position['entry']) * position['qty']
            currentEquity = currentCapital + floatingPnL

        peakCapital = max(peakCapital, currentEquity)
        currentDrawdownPct = ((peakCapital - currentEquity) / peakCapital) * 100 if peakCapital > 0 else 0.0
        maxDrawdownPct = max(maxDrawdownPct, currentDrawdownPct)

    end_position_qty = position['qty']
    had_open_position_at_end = end_position_qty > 0

    if had_open_position_at_end:
        exec_price = adjust_long_sell_fill_price(C[-1])
        net_price = calc_net_sell_price(exec_price, position['qty'], params)
        pnl = (net_price - position['entry']) * position['qty']
        total_pnl = position['realized_pnl'] + pnl
        trade_r_mult = total_pnl / position['initial_risk_total'] if position['initial_risk_total'] > 0 else 0.0

        total_r_multiple += trade_r_mult
        tradeCount += 1

        if return_logs:
            trade_logs.append({
                'exit_date': Dates[-1],
                'pnl': total_pnl,
                'r_mult': trade_r_mult
            })

        if total_pnl > 0:
            fullWins += 1
            totalProfit += total_pnl
            total_r_win += trade_r_mult
        else:
            totalLoss += abs(total_pnl)
            total_r_loss += abs(trade_r_mult)

        currentCapital += pnl
        currentEquity = currentCapital

        peakCapital = max(peakCapital, currentEquity)
        currentDrawdownPct = ((peakCapital - currentEquity) / peakCapital) * 100 if peakCapital > 0 else 0.0
        maxDrawdownPct = max(maxDrawdownPct, currentDrawdownPct)

        position['qty'] = 0

    winRate = (fullWins / tradeCount * 100) if tradeCount > 0 else 0
    avgWin = totalProfit / fullWins if fullWins > 0 else 0
    lossCount = tradeCount - fullWins
    avgLoss = totalLoss / lossCount if lossCount > 0 else 0
    payoffRatio = (avgWin / avgLoss) if avgLoss > 0 else (99.9 if avgWin > 0 else 0.0)

    if EV_CALC_METHOD == 'B':
        win_rate_dec = fullWins / tradeCount if tradeCount > 0 else 0
        avg_win_r = total_r_win / fullWins if fullWins > 0 else 0
        avg_loss_r = total_r_loss / lossCount if lossCount > 0 else 0
        payoff_for_ev = min(10.0, (avg_win_r / avg_loss_r)) if avg_loss_r > 0 else (99.9 if avg_win_r > 0 else 0.0)
        expectedValue = (win_rate_dec * payoff_for_ev) - (1 - win_rate_dec)
    else:
        expectedValue = (total_r_multiple / tradeCount) if tradeCount > 0 else 0.0

    finalEquity = currentEquity
    totalNetProfitPct = ((finalEquity - params.initial_capital) / params.initial_capital) * 100
    score = totalNetProfitPct / tradeCount if tradeCount > 0 else 0

    isSetup_today = buyCondition[-1] and (not had_open_position_at_end)
    buyLimit_today = adjust_long_buy_limit(C[-1] + ATR_main[-1] * params.atr_buy_tol) if isSetup_today else np.nan
    stopLoss_today = adjust_long_stop_price(buyLimit_today - ATR_main[-1] * params.atr_times_init) if isSetup_today else np.nan

    isCandidate, _, _, _ = evaluate_history_candidate_metrics(
        tradeCount,
        fullWins,
        total_r_multiple,
        total_r_win,
        total_r_loss,
        params,
    )

    extended_candidate_today = None
    if (not had_open_position_at_end) and active_extended_signal is not None:
        sizing_cap = currentCapital if getattr(params, 'use_compounding', True) else params.initial_capital
        extended_candidate_today = build_extended_candidate_plan_from_signal(active_extended_signal, C[-1], sizing_cap, params)

    avg_bars_held = total_bars_held / tradeCount if tradeCount > 0 else 0

    stats_dict = {
        "asset_growth": totalNetProfitPct,
        "trade_count": tradeCount,
        "missed_buys": missedBuyCount,
        "missed_sells": missedSellCount,
        "score": score,
        "win_rate": winRate,
        "avg_win": avgWin,
        "avg_loss": avgLoss,
        "payoff_ratio": payoffRatio,
        "expected_value": expectedValue,
        "max_drawdown": maxDrawdownPct,
        "is_candidate": isCandidate,
        "is_setup_today": isSetup_today,
        "buy_limit": buyLimit_today,
        "stop_loss": stopLoss_today,
        "extended_candidate_today": extended_candidate_today,
        "current_position": end_position_qty,
        "avg_bars_held": avg_bars_held
    }

    if return_logs: return stats_dict, trade_logs
    return stats_dict