import numpy as np

from core.backtest_finalize import build_backtest_stats, finalize_open_position_at_end
from core.capital_policy import resolve_single_backtest_sizing_capital
from core.exact_accounting import build_sell_ledger_from_price, calc_ratio_from_milli, milli_to_money, money_to_milli, price_to_milli
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


def _empty_pit_stats_index():
    return {
        'exit_dates': [],
        'cum_trade_count': np.array([], dtype=np.int32),
        'cum_win_count': np.array([], dtype=np.int32),
        'cum_win_r_sum': np.array([], dtype=np.float64),
        'cum_loss_r_sum': np.array([], dtype=np.float64),
        'cum_total_r_sum': np.array([], dtype=np.float64),
        'cum_pnl_sum': np.array([], dtype=np.float64),
    }


def _create_pit_stats_builder():
    return {
        'exit_dates': [],
        'cum_trade_count': [],
        'cum_win_count': [],
        'cum_win_r_sum': [],
        'cum_loss_r_sum': [],
        'cum_total_r_sum': [],
        'cum_pnl_sum': [],
        '_trade_count': 0,
        '_win_count': 0,
        '_win_r_sum': 0.0,
        '_loss_r_sum': 0.0,
        '_total_r_sum': 0.0,
        '_pnl_sum': 0.0,
    }


def _append_pit_trade(builder, *, exit_date, pnl, r_mult):
    if builder is None:
        return

    builder['_trade_count'] += 1
    builder['_total_r_sum'] += float(r_mult)
    builder['_pnl_sum'] += float(pnl)
    if float(pnl) > 0.0:
        builder['_win_count'] += 1
        builder['_win_r_sum'] += float(r_mult)
    else:
        builder['_loss_r_sum'] += float(r_mult)

    builder['exit_dates'].append(exit_date)
    builder['cum_trade_count'].append(builder['_trade_count'])
    builder['cum_win_count'].append(builder['_win_count'])
    builder['cum_win_r_sum'].append(builder['_win_r_sum'])
    builder['cum_loss_r_sum'].append(builder['_loss_r_sum'])
    builder['cum_total_r_sum'].append(builder['_total_r_sum'])
    builder['cum_pnl_sum'].append(builder['_pnl_sum'])


def _finalize_pit_stats_index(builder):
    if builder is None:
        return None
    if not builder['exit_dates']:
        return _empty_pit_stats_index()
    return {
        'exit_dates': list(builder['exit_dates']),
        'cum_trade_count': np.array(builder['cum_trade_count'], dtype=np.int32),
        'cum_win_count': np.array(builder['cum_win_count'], dtype=np.int32),
        'cum_win_r_sum': np.array(builder['cum_win_r_sum'], dtype=np.float64),
        'cum_loss_r_sum': np.array(builder['cum_loss_r_sum'], dtype=np.float64),
        'cum_total_r_sum': np.array(builder['cum_total_r_sum'], dtype=np.float64),
        'cum_pnl_sum': np.array(builder['cum_pnl_sum'], dtype=np.float64),
    }


def _optimizer_limit_reachable_for_entry_day(t_low, t_open, t_volume, limit_price):
    """Cheap no-fill guard for optimizer PIT mode.

    The formal entry function remains the single source of truth for fills.
    This guard only skips building full entry plans on days that cannot fill
    because required bar fields are invalid or the day low never reaches the
    plan limit. It is used only when collect_stats=False, so scanner/display
    miss-buy accounting stays on the original full path.
    """
    if np.isnan(t_low) or np.isnan(t_open) or np.isnan(t_volume) or np.isnan(limit_price):
        return False
    if float(t_volume) <= 0.0:
        return False
    return price_to_milli(t_low) <= price_to_milli(limit_price)


def _optimizer_extended_entry_limit(active_extended_signal):
    if active_extended_signal is None:
        return np.nan
    shadow_position = active_extended_signal.get("shadow_position")
    if shadow_position is not None and int(shadow_position.get("qty", 0) or 0) > 0:
        if shadow_position.get("pending_exit_action") is not None:
            return np.nan
        return shadow_position.get("entry_fill_price", np.nan)
    return active_extended_signal.get("orig_limit", np.nan)


def run_v16_backtest(df, params=None, return_logs=False, precomputed_signals=None, ticker=None, collect_stats=True, return_pit_stats_index=False):
    if params is None:
        params = V16StrategyParams()

    resolved_ticker = ticker or df.attrs.get('ticker')
    resolved_security_profile = df.attrs.get('security_profile')

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
    scanner_extended_signal = None
    pit_stats_builder = _create_pit_stats_builder() if return_pit_stats_index else None
    currentCapital_milli = money_to_milli(params.initial_capital)
    tradeCount, fullWins, missedBuyCount, missedSellCount = 0, 0, 0, 0
    totalProfit_milli, totalLoss_milli = 0, 0
    peakCapital_milli, maxDrawdownPct = currentCapital_milli, 0.0
    total_r_multiple, total_r_win, total_r_loss, total_bars_held = 0.0, 0.0, 0.0, 0
    trade_logs = []
    currentEquity_milli = currentCapital_milli
    collect_stats = bool(collect_stats)
    optimizer_setup_entry_positions = None
    optimizer_setup_entry_cursor = 0
    if not collect_stats:
        valid_setup_mask = np.asarray(buyCondition[:-1], dtype=bool) & ~np.isnan(np.asarray(ATR_main[:-1], dtype=np.float64))
        optimizer_setup_entry_positions = np.flatnonzero(valid_setup_mask) + 1

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
            low_last=float('nan'),
            had_open_position_at_end=False,
            active_extended_signal=None,
            end_position_qty=0,
            avg_bars_held=0,
            final_date=None,
            security_profile=resolved_security_profile,
        )
        stats_dict['is_candidate'] = False
        if return_pit_stats_index:
            pit_stats_index = _empty_pit_stats_index()
            if return_logs:
                return (stats_dict if collect_stats else None), trade_logs, pit_stats_index
            return (stats_dict if collect_stats else None), pit_stats_index
        if return_logs:
            return (stats_dict if collect_stats else None), trade_logs
        return stats_dict if collect_stats else None

    j = 1
    while j < len(C):
        if np.isnan(ATR_main[j - 1]):
            if collect_stats:
                currentEquity_milli = currentCapital_milli
                peakCapital_milli = max(peakCapital_milli, currentEquity_milli)
                currentDrawdownPct = ((peakCapital_milli - currentEquity_milli) / peakCapital_milli) * 100 if peakCapital_milli > 0 else 0.0
                maxDrawdownPct = max(maxDrawdownPct, currentDrawdownPct)
            j += 1
            continue

        pos_start_of_current_bar = position['qty']
        if (
            not collect_stats
            and pos_start_of_current_bar == 0
            and active_extended_signal is None
            and not bool(buyCondition[j - 1])
        ):
            if optimizer_setup_entry_positions is not None:
                while (
                    optimizer_setup_entry_cursor < len(optimizer_setup_entry_positions)
                    and int(optimizer_setup_entry_positions[optimizer_setup_entry_cursor]) <= j
                ):
                    optimizer_setup_entry_cursor += 1
                if optimizer_setup_entry_cursor < len(optimizer_setup_entry_positions):
                    j = int(optimizer_setup_entry_positions[optimizer_setup_entry_cursor])
                    continue
            break

        if pos_start_of_current_bar > 0:
            if collect_stats:
                total_bars_held += 1
            position, freed_cash_milli, _pnl_realized_milli, events = execute_bar_step(
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
                y_high=H[j - 1],
                return_milli=True,
                record_exec_contexts=False,
                sync_display_fields=collect_stats or return_logs,
            )
            currentCapital_milli += freed_cash_milli
            if 'STOP' in events or 'IND_SELL' in events:
                total_pnl_milli = int(position.get('realized_pnl_milli', 0) or 0)
                total_pnl = position['realized_pnl'] if (collect_stats or return_logs) else milli_to_money(total_pnl_milli)
                trade_r_mult = calc_ratio_from_milli(total_pnl_milli, position.get('initial_risk_total_milli', 0))
                if collect_stats:
                    total_r_multiple += trade_r_mult
                    tradeCount += 1
                if return_logs:
                    trade_logs.append({'exit_date': Dates[j], 'pnl': total_pnl, 'r_mult': trade_r_mult})
                if pit_stats_builder is not None:
                    _append_pit_trade(pit_stats_builder, exit_date=Dates[j], pnl=total_pnl, r_mult=trade_r_mult)
                if collect_stats:
                    if position['realized_pnl_milli'] > 0:
                        fullWins += 1
                        totalProfit_milli += position['realized_pnl_milli']
                        total_r_win += trade_r_mult
                    else:
                        totalLoss_milli += abs(position['realized_pnl_milli'])
                        total_r_loss += abs(trade_r_mult)
            elif 'MISSED_SELL' in events and collect_stats:
                missedSellCount += 1

        isSetup_prev = bool(buyCondition[j - 1]) and (pos_start_of_current_bar == 0)
        buyTriggered = False
        sizing_cap = None

        if isSetup_prev:
            sizing_cap = resolve_single_backtest_sizing_capital(params, currentCapital_milli / 1000.0)
            signal_state = create_signal_tracking_state(
                buy_limits[j - 1],
                ATR_main[j - 1],
                params,
                ticker=resolved_ticker,
                security_profile=resolved_security_profile,
            )
            if signal_state is not None:
                active_extended_signal = signal_state
                if collect_stats:
                    scanner_extended_signal = create_signal_tracking_state(
                        buy_limits[j - 1],
                        ATR_main[j - 1],
                        params,
                        ticker=resolved_ticker,
                        security_profile=resolved_security_profile,
                    )

            should_try_normal_entry = collect_stats or _optimizer_limit_reachable_for_entry_day(
                L[j], O[j], V[j], buy_limits[j - 1]
            )
            if should_try_normal_entry:
                entry_plan = build_normal_entry_plan(
                    buy_limits[j - 1],
                    ATR_main[j - 1],
                    sizing_cap,
                    params,
                    ticker=resolved_ticker,
                    security_profile=resolved_security_profile,
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
                    entry_type='normal',
                    ticker=resolved_ticker,
                    trade_date=Dates[j],
                )
                entry_filled = bool(entry_result['filled'])
                entry_count_as_missed_buy = bool(entry_result['count_as_missed_buy'])
            else:
                entry_filled = False
                entry_count_as_missed_buy = False
            if entry_filled:
                filled_signal_state = active_extended_signal
                position = entry_result['position']
                currentCapital_milli -= position['net_buy_total_milli']
                buyTriggered = True
                active_extended_signal = None
            elif entry_count_as_missed_buy and collect_stats:
                missedBuyCount += 1

        elif active_extended_signal is not None and pos_start_of_current_bar == 0:
            sizing_cap = resolve_single_backtest_sizing_capital(params, currentCapital_milli / 1000.0)
            should_try_extended_entry = collect_stats or _optimizer_limit_reachable_for_entry_day(
                L[j], O[j], V[j], _optimizer_extended_entry_limit(active_extended_signal)
            )
            if should_try_extended_entry:
                entry_plan = build_extended_entry_plan_from_signal(
                    active_extended_signal,
                    sizing_cap,
                    params,
                    y_close=C[j - 1],
                    ticker=resolved_ticker,
                    security_profile=active_extended_signal.get('security_profile'),
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
                entry_filled = bool(entry_result['filled'])
                entry_count_as_missed_buy = bool(entry_result['count_as_missed_buy'])
            else:
                entry_filled = False
                entry_count_as_missed_buy = False
            if entry_filled:
                filled_signal_state = active_extended_signal
                position = entry_result['position']
                currentCapital_milli -= position['net_buy_total_milli']
                buyTriggered = True
                active_extended_signal = None
            elif entry_count_as_missed_buy and collect_stats:
                missedBuyCount += 1

        if not buyTriggered and position['qty'] == 0 and active_extended_signal is not None:
            if sizing_cap is None:
                sizing_cap = resolve_single_backtest_sizing_capital(params, currentCapital_milli / 1000.0)
            if should_clear_extended_signal(
                active_extended_signal,
                L[j],
                H[j],
                t_open=O[j],
                t_close=C[j],
                t_volume=V[j],
                y_close=C[j - 1],
                y_high=H[j - 1],
                y_atr=ATR_main[j - 1],
                y_ind_sell=sellCondition[j - 1],
                sizing_capital=sizing_cap,
                current_date=Dates[j],
                params=params,
                copy_shadow_position=collect_stats,
            ):
                active_extended_signal = None

        if collect_stats and scanner_extended_signal is not None:
            if sizing_cap is None:
                sizing_cap = resolve_single_backtest_sizing_capital(params, currentCapital_milli / 1000.0)
            should_clear_scanner_extended = should_clear_extended_signal(
                scanner_extended_signal,
                L[j],
                H[j],
                t_open=O[j],
                t_close=C[j],
                t_volume=V[j],
                y_close=C[j - 1],
                y_high=H[j - 1],
                y_atr=ATR_main[j - 1],
                y_ind_sell=sellCondition[j - 1],
                sizing_capital=sizing_cap,
                current_date=Dates[j],
                params=params,
            )
            if should_clear_scanner_extended:
                scanner_extended_signal = None

        if collect_stats:
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

        j += 1

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
        collect_stats=collect_stats,
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
    if pit_stats_builder is not None and final_state.get('final_trade_exit_date') is not None:
        _append_pit_trade(
            pit_stats_builder,
            exit_date=final_state['final_trade_exit_date'],
            pnl=final_state['final_trade_pnl'],
            r_mult=final_state['final_trade_r_mult'],
        )
    pit_stats_index = _finalize_pit_stats_index(pit_stats_builder) if return_pit_stats_index else None

    if not collect_stats:
        if return_pit_stats_index:
            if return_logs:
                return None, trade_logs, pit_stats_index
            return None, pit_stats_index
        if return_logs:
            return None, trade_logs
        return None

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
        low_last=L[-1],
        had_open_position_at_end=had_open_position_at_end,
        active_extended_signal=active_extended_signal,
        active_extended_signal_tbd=scanner_extended_signal,
        end_position_qty=end_position_qty,
        avg_bars_held=avg_bars_held,
        final_date=Dates[-1],
        security_profile=resolved_security_profile,
    )

    if return_pit_stats_index:
        if return_logs:
            return stats_dict, trade_logs, pit_stats_index
        return stats_dict, pit_stats_index
    if return_logs:
        return stats_dict, trade_logs
    return stats_dict
