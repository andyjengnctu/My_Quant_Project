import numpy as np

from core.backtest_finalize import build_backtest_stats, finalize_open_position_at_end
from core.capital_policy import resolve_single_backtest_sizing_capital
from core.exact_accounting import build_sell_ledger_from_price, milli_to_money, money_to_milli
from core.strategy_params import V16StrategyParams
from core.position_step import execute_bar_step
from core.price_utils import adjust_long_sell_fill_price
from core.signal_utils import generate_signals
from core.trade_plans import (
    build_extended_entry_plan_from_signal,
    build_normal_entry_plan,
    create_signal_tracking_state,
    execute_pre_market_entry_plan,
    should_clear_extended_signal,
)


def run_v16_backtest(df, params=None, return_logs=False, precomputed_signals=None, ticker=None):
    if params is None:
        params = V16StrategyParams()

    resolved_ticker = ticker or df.attrs.get('ticker')

    H = df['High'].to_numpy(dtype=np.float64, copy=False)
    L = df['Low'].to_numpy(dtype=np.float64, copy=False)
    C = df['Close'].to_numpy(dtype=np.float64, copy=False)
    O = df['Open'].to_numpy(dtype=np.float64, copy=False)
    V = df['Volume'].to_numpy(dtype=np.float64, copy=False)
    Dates = df.index

    if precomputed_signals is None:
        ATR_main, buyCondition, sellCondition, buy_limits = generate_signals(df, params, ticker=resolved_ticker)
    else:
        ATR_main, buyCondition, sellCondition, buy_limits = precomputed_signals

    position = {'qty': 0}
    active_extended_signal = None
    currentCapital_milli = money_to_milli(params.initial_capital)
    tradeCount, fullWins, missedBuyCount, missedSellCount = 0, 0, 0, 0
    totalProfit_milli, totalLoss_milli = 0, 0
    peakCapital_milli, maxDrawdownPct = currentCapital_milli, 0.0
    total_r_multiple, total_r_win, total_r_loss, total_bars_held = 0.0, 0.0, 0.0, 0
    trade_logs = []
    currentEquity_milli = currentCapital_milli

    if len(C) == 0:
        stats_dict = build_backtest_stats(
            params=params,
            ticker=resolved_ticker,
            current_capital=milli_to_money(currentCapital_milli),
            current_equity=milli_to_money(currentEquity_milli),
            max_drawdown_pct=maxDrawdownPct,
            trade_count=tradeCount,
            full_wins=fullWins,
            total_profit=0.0,
            total_loss=0.0,
            total_r_multiple=total_r_multiple,
            total_r_win=total_r_win,
            total_r_loss=total_r_loss,
            missed_buy_count=missedBuyCount,
            missed_sell_count=missedSellCount,
            buy_condition_last=False,
            atr_last=float('nan'),
            close_last=float('nan'),
            had_open_position_at_end=False,
            active_extended_signal=None,
            end_position_qty=0,
            avg_bars_held=0,
            final_date=None,
        )
        stats_dict['is_candidate'] = False
        if return_logs:
            return stats_dict, trade_logs
        return stats_dict

    for j in range(1, len(C)):
        if np.isnan(ATR_main[j - 1]):
            currentEquity_milli = currentCapital_milli
            peakCapital_milli = max(peakCapital_milli, currentEquity_milli)
            currentDrawdownPct = ((peakCapital_milli - currentEquity_milli) / peakCapital_milli) * 100 if peakCapital_milli > 0 else 0.0
            maxDrawdownPct = max(maxDrawdownPct, currentDrawdownPct)
            continue

        pos_start_of_current_bar = position['qty']
        if pos_start_of_current_bar > 0:
            total_bars_held += 1
            position, _freed_cash, _pnl_realized, events = execute_bar_step(
                position,
                ATR_main[j - 1],
                sellCondition[j - 1],
                C[j - 1],
                O[j],
                H[j],
                L[j],
                C[j],
                V[j],
                params,
                current_date=Dates[j],
            )
            freed_cash_milli = sum(int(ctx.get('net_total_milli', 0)) for ctx in position.get('_last_exec_contexts', []))
            currentCapital_milli += freed_cash_milli
            if 'STOP' in events or 'IND_SELL' in events:
                total_pnl = position['realized_pnl']
                trade_r_mult = total_pnl / position['initial_risk_total'] if position['initial_risk_total'] > 0 else 0
                total_r_multiple += trade_r_mult
                tradeCount += 1
                if return_logs:
                    trade_logs.append({'exit_date': Dates[j], 'pnl': total_pnl, 'r_mult': trade_r_mult})
                if position['realized_pnl_milli'] > 0:
                    fullWins += 1
                    totalProfit_milli += position['realized_pnl_milli']
                    total_r_win += trade_r_mult
                else:
                    totalLoss_milli += abs(position['realized_pnl_milli'])
                    total_r_loss += abs(trade_r_mult)
            elif 'MISSED_SELL' in events:
                missedSellCount += 1

        isSetup_prev = buyCondition[j - 1] and (pos_start_of_current_bar == 0)
        buyTriggered = False
        sizing_cap = resolve_single_backtest_sizing_capital(params, milli_to_money(currentCapital_milli))

        if isSetup_prev:
            signal_state = create_signal_tracking_state(buy_limits[j - 1], ATR_main[j - 1], params, ticker=resolved_ticker)
            if signal_state is not None:
                active_extended_signal = signal_state

            entry_plan = build_normal_entry_plan(buy_limits[j - 1], ATR_main[j - 1], sizing_cap, params, ticker=resolved_ticker, trade_date=Dates[j])
            entry_result = execute_pre_market_entry_plan(
                entry_plan=entry_plan,
                t_open=O[j],
                t_high=H[j],
                t_low=L[j],
                t_close=C[j],
                t_volume=V[j],
                y_close=C[j - 1],
                params=params,
                entry_type='normal',
                ticker=resolved_ticker,
                trade_date=Dates[j],
            )
            if entry_result['filled']:
                position = entry_result['position']
                currentCapital_milli -= position['net_buy_total_milli']
                buyTriggered = True
                active_extended_signal = None
            elif entry_result['count_as_missed_buy']:
                missedBuyCount += 1

        elif active_extended_signal is not None and pos_start_of_current_bar == 0:
            entry_plan = build_extended_entry_plan_from_signal(
                active_extended_signal,
                sizing_cap,
                params,
                y_close=C[j - 1],
                ticker=resolved_ticker,
                trade_date=Dates[j],
            )
            entry_result = execute_pre_market_entry_plan(
                entry_plan=entry_plan,
                t_open=O[j],
                t_high=H[j],
                t_low=L[j],
                t_close=C[j],
                t_volume=V[j],
                y_close=C[j - 1],
                params=params,
                entry_type='extended',
                ticker=resolved_ticker,
                trade_date=Dates[j],
            )
            if entry_result['filled']:
                position = entry_result['position']
                currentCapital_milli -= position['net_buy_total_milli']
                buyTriggered = True
                active_extended_signal = None
            elif entry_result['count_as_missed_buy']:
                missedBuyCount += 1

        if not buyTriggered and position['qty'] == 0 and should_clear_extended_signal(active_extended_signal, L[j], H[j], t_open=O[j], params=params):
            active_extended_signal = None

        currentEquity_milli = currentCapital_milli
        if position['qty'] > 0:
            floating_exec_price = adjust_long_sell_fill_price(C[j], ticker=resolved_ticker)
            floating_sell_ledger = build_sell_ledger_from_price(
                floating_exec_price,
                position['qty'],
                params,
                ticker=position.get('ticker', resolved_ticker),
                security_profile=position.get('security_profile'),
                trade_date=Dates[j],
            )
            currentEquity_milli = currentCapital_milli + floating_sell_ledger['net_sell_total_milli']

        peakCapital_milli = max(peakCapital_milli, currentEquity_milli)
        currentDrawdownPct = ((peakCapital_milli - currentEquity_milli) / peakCapital_milli) * 100 if peakCapital_milli > 0 else 0.0
        maxDrawdownPct = max(maxDrawdownPct, currentDrawdownPct)

    final_state = finalize_open_position_at_end(
        position=position,
        ticker=resolved_ticker,
        final_close=C[-1],
        final_date=Dates[-1],
        current_capital_milli=currentCapital_milli,
        current_equity_milli=currentEquity_milli,
        peak_capital_milli=peakCapital_milli,
        max_drawdown_pct=maxDrawdownPct,
        trade_count=tradeCount,
        full_wins=fullWins,
        total_profit_milli=totalProfit_milli,
        total_loss_milli=totalLoss_milli,
        total_r_multiple=total_r_multiple,
        total_r_win=total_r_win,
        total_r_loss=total_r_loss,
        trade_logs=trade_logs,
        return_logs=return_logs,
        params=params,
    )
    currentCapital_milli = final_state['current_capital_milli']
    currentEquity_milli = final_state['current_equity_milli']
    maxDrawdownPct = final_state['max_drawdown_pct']
    tradeCount = final_state['trade_count']
    fullWins = final_state['full_wins']
    totalProfit_milli = final_state['total_profit_milli']
    totalLoss_milli = final_state['total_loss_milli']
    total_r_multiple = final_state['total_r_multiple']
    total_r_win = final_state['total_r_win']
    total_r_loss = final_state['total_r_loss']
    had_open_position_at_end = final_state['had_open_position_at_end']
    end_position_qty = final_state['end_position_qty']
    trade_logs = final_state['trade_logs']

    avg_bars_held = total_bars_held / tradeCount if tradeCount > 0 else 0
    stats_dict = build_backtest_stats(
        params=params,
        ticker=resolved_ticker,
        current_capital=milli_to_money(currentCapital_milli),
        current_equity=milli_to_money(currentEquity_milli),
        max_drawdown_pct=maxDrawdownPct,
        trade_count=tradeCount,
        full_wins=fullWins,
        total_profit=milli_to_money(totalProfit_milli),
        total_loss=milli_to_money(totalLoss_milli),
        total_r_multiple=total_r_multiple,
        total_r_win=total_r_win,
        total_r_loss=total_r_loss,
        missed_buy_count=missedBuyCount,
        missed_sell_count=missedSellCount,
        buy_condition_last=buyCondition[-1],
        atr_last=ATR_main[-1],
        close_last=C[-1],
        had_open_position_at_end=had_open_position_at_end,
        active_extended_signal=active_extended_signal,
        end_position_qty=end_position_qty,
        avg_bars_held=avg_bars_held,
        final_date=Dates[-1],
    )

    if return_logs:
        return stats_dict, trade_logs
    return stats_dict
