from core.buy_sort import calc_buy_sort_value
from core.config import get_buy_sort_method
from core.exact_accounting import (
    build_sell_ledger_from_price,
    calc_ratio_from_milli,
    calc_reconciled_exit_display_pnl,
    calc_total_from_average_price_milli,
    coerce_money_like_to_milli,
    milli_to_money,
    register_display_realized_pnl,
    round_money_for_display,
)
from core.position_step import execute_bar_step
from core.price_utils import (
    adjust_long_sell_fill_price,
    get_exit_sell_block_reason,
)
from core.portfolio_fast_data import (
    get_fast_close,
    get_fast_pos,
    get_fast_value,
    get_pit_stats_from_index,
    is_extended_entry_type,
)


def _first_exec_context(position, event_name):
    for ctx in position.get('_last_exec_contexts', []):
        if ctx.get('event') == event_name:
            return ctx
    return None


def _round_money_for_history(value):
    return round_money_for_display(value)




def _resolve_full_entry_capital_milli(position, fallback_qty):
    exact_total_milli = int(position.get('net_buy_total_milli', 0) or 0)
    if exact_total_milli > 0:
        return exact_total_milli

    display_total = float(position.get('entry_capital_total', 0.0) or 0.0)
    if display_total > 0:
        return coerce_money_like_to_milli(round_money_for_display(display_total))

    avg_entry_price = float(position.get('entry', 0.0) or 0.0)
    initial_qty = int(position.get('initial_qty', fallback_qty) or fallback_qty or 0)
    if avg_entry_price <= 0 or initial_qty <= 0:
        return 0
    return calc_total_from_average_price_milli(avg_entry_price, initial_qty)


def _calc_position_mark_to_market_return(position, mark_price, params, trade_date=None):
    remaining_qty = int(position.get('qty', 0) or 0)
    if remaining_qty <= 0:
        return float('inf')

    full_entry_total_milli = _resolve_full_entry_capital_milli(position, remaining_qty)
    if full_entry_total_milli <= 0:
        return float('inf')

    realized_pnl_milli = int(position.get('realized_pnl_milli', 0) or 0)
    remaining_cost_basis_milli = int(position.get('remaining_cost_basis_milli', 0) or 0)
    sell_ledger = build_sell_ledger_from_price(
        mark_price,
        remaining_qty,
        params,
        ticker=position.get('ticker'),
        security_profile=position.get('security_profile'),
        trade_date=trade_date,
    )

    if remaining_cost_basis_milli > 0:
        floating_pnl_milli = sell_ledger['net_sell_total_milli'] - remaining_cost_basis_milli
        total_trade_pnl_milli = realized_pnl_milli + floating_pnl_milli
    else:
        total_trade_pnl_milli = sell_ledger['net_sell_total_milli'] - full_entry_total_milli

    return total_trade_pnl_milli / full_entry_total_milli


def try_rotate_weakest_position(
    portfolio,
    orderable_candidates_today,
    max_positions,
    enable_rotation,
    sold_today,
    all_dfs_fast,
    today,
    pit_stats_index,
    params,
    cash,
    closed_trades_stats,
    trade_history,
    is_training,
    normal_trade_count,
    extended_trade_count,
):
    if len(portfolio) != max_positions or not enable_rotation or not orderable_candidates_today:
        return cash, normal_trade_count, extended_trade_count

    for cand_idx in range(len(orderable_candidates_today)):
        cand = orderable_candidates_today[cand_idx]
        weakest_ticker = None
        lowest_ret = float('inf')

        for pt in sorted(portfolio.keys()):
            pos = portfolio[pt]
            pt_data = all_dfs_fast[pt]
            pt_pos = get_fast_pos(pt_data, today)
            if pt_pos <= 0:
                continue
            pt_y_pos = pt_pos - 1
            pt_y_close = get_fast_close(pt_data, pos=pt_y_pos)
            ret = _calc_position_mark_to_market_return(pos, pt_y_close, params, trade_date=today)

            holding_cost = milli_to_money(pos.get('remaining_cost_basis_milli', 0))
            _, holding_ev, holding_win_rate, holding_trade_count, holding_asset_growth_pct = get_pit_stats_from_index(
                pit_stats_index[pt],
                today,
                params,
            )
            holding_sort_value = calc_buy_sort_value(
                get_buy_sort_method(),
                holding_ev,
                holding_cost,
                holding_win_rate,
                holding_trade_count,
                holding_asset_growth_pct,
            )
            is_strategically_better = cand['sort_value'] > holding_sort_value

            if is_strategically_better and ret < lowest_ret:
                lowest_ret = ret
                weakest_ticker = pt

        if weakest_ticker is None:
            continue

        w_data = all_dfs_fast[weakest_ticker]
        w_pos = get_fast_pos(w_data, today)
        if w_pos <= 0:
            continue
        w_y_pos = w_pos - 1
        w_open = get_fast_value(w_data, 'Open', pos=w_pos)
        w_high = get_fast_value(w_data, 'High', pos=w_pos)
        w_low = get_fast_value(w_data, 'Low', pos=w_pos)
        w_close = get_fast_close(w_data, pos=w_pos)
        w_y_close = get_fast_close(w_data, pos=w_y_pos)
        sell_block_reason = get_exit_sell_block_reason(
            w_open,
            w_high,
            w_low,
            w_close,
            get_fast_value(w_data, 'Volume', pos=w_pos),
            w_y_close,
        )

        if sell_block_reason is not None:
            continue

        pos = portfolio[weakest_ticker]
        est_sell_px = adjust_long_sell_fill_price(w_open, ticker=weakest_ticker)
        sell_ledger = build_sell_ledger_from_price(
            est_sell_px,
            pos['qty'],
            params,
            ticker=weakest_ticker,
            security_profile=pos.get('security_profile'),
            trade_date=today,
        )
        est_freed_cash_milli = sell_ledger['net_sell_total_milli']
        pnl_milli = est_freed_cash_milli - pos['remaining_cost_basis_milli']
        cash += est_freed_cash_milli

        total_pnl_milli = pos['realized_pnl_milli'] + pnl_milli
        total_pnl = milli_to_money(total_pnl_milli)
        display_tail_pnl = calc_reconciled_exit_display_pnl(pos, total_pnl)
        total_r = calc_ratio_from_milli(total_pnl_milli, pos.get('initial_risk_total_milli', 0))
        closed_trades_stats.append(
            {'pnl': total_pnl, 'r_mult': total_r, 'entry_type': pos.get('entry_type', 'normal')}
        )
        if is_extended_entry_type(pos.get('entry_type', 'normal')):
            extended_trade_count += 1
        else:
            normal_trade_count += 1

        if not is_training:
            trade_history.append(
                {
                    'Date': today.strftime('%Y-%m-%d'),
                    'Ticker': weakest_ticker,
                    'Type': '汰弱賣出(Open, T+1再評估買進)',
                    '成交價': est_sell_px,
                    '股數': pos['qty'],
                    '單筆損益': _round_money_for_history(display_tail_pnl),
                    '該筆總損益': _round_money_for_history(total_pnl),
                    'R_Multiple': total_r,
                    'Risk': params.fixed_risk,
                }
            )
            register_display_realized_pnl(pos, display_tail_pnl)

        del portfolio[weakest_ticker]
        sold_today.add(weakest_ticker)
        break

    return cash, normal_trade_count, extended_trade_count


def settle_portfolio_positions(
    portfolio,
    sold_today,
    all_dfs_fast,
    today,
    params,
    cash,
    closed_trades_stats,
    trade_history,
    is_training,
    total_missed_sells,
    normal_trade_count,
    extended_trade_count,
):
    tickers_to_remove = []
    for ticker in sorted(portfolio.keys()):
        pos = portfolio[ticker]
        fast_df = all_dfs_fast[ticker]
        t_pos = get_fast_pos(fast_df, today)
        if t_pos <= 0:
            continue
        y_pos = t_pos - 1

        pos, freed_cash_milli, _pnl_realized_milli, events = execute_bar_step(
            pos,
            get_fast_value(fast_df, 'ATR', pos=y_pos),
            get_fast_value(fast_df, 'ind_sell_signal', pos=y_pos),
            get_fast_close(fast_df, pos=y_pos),
            get_fast_value(fast_df, 'Open', pos=t_pos),
            get_fast_value(fast_df, 'High', pos=t_pos),
            get_fast_value(fast_df, 'Low', pos=t_pos),
            get_fast_close(fast_df, pos=t_pos),
            get_fast_value(fast_df, 'Volume', pos=t_pos),
            params,
            current_date=today,
            y_high=get_fast_value(fast_df, 'High', pos=y_pos),
            return_milli=is_training,
            record_exec_contexts=not is_training,
            sync_display_fields=not is_training,
        )
        if not is_training:
            freed_cash_milli = sum(int(ctx.get('net_total_milli', 0)) for ctx in pos.get('_last_exec_contexts', []))
        cash += int(freed_cash_milli)

        tp_context = _first_exec_context(pos, 'TP_HALF')
        stop_context = _first_exec_context(pos, 'STOP')
        ind_sell_context = _first_exec_context(pos, 'IND_SELL')

        if 'TP_HALF' in events and not is_training:
            trade_history.append(
                {
                    'Date': today.strftime('%Y-%m-%d'),
                    'Ticker': ticker,
                    'Type': '半倉停利',
                    '成交價': None if tp_context is None else tp_context['exec_price'],
                    '股數': None if tp_context is None else tp_context['qty'],
                    '單筆損益': 0.0 if tp_context is None else _round_money_for_history(round_money_for_display(tp_context['pnl'])),
                    'R_Multiple': 0.0,
                    'Risk': params.fixed_risk,
                }
            )
            if tp_context is not None:
                register_display_realized_pnl(pos, round_money_for_display(tp_context['pnl']))

        if 'STOP' in events or 'IND_SELL' in events:
            total_pnl_milli = int(pos.get('realized_pnl_milli', 0) or 0)
            total_pnl = milli_to_money(total_pnl_milli) if is_training else pos['realized_pnl']
            total_r = calc_ratio_from_milli(pos.get('realized_pnl_milli', 0), pos.get('initial_risk_total_milli', 0))
            closed_trades_stats.append(
                {'pnl': total_pnl, 'r_mult': total_r, 'entry_type': pos.get('entry_type', 'normal')}
            )
            if is_extended_entry_type(pos.get('entry_type', 'normal')):
                extended_trade_count += 1
            else:
                normal_trade_count += 1

            if not is_training:
                t_type = '全倉結算(停損)' if 'STOP' in events else '全倉結算(指標)'
                exit_context = stop_context if 'STOP' in events else ind_sell_context
                display_exit_pnl = calc_reconciled_exit_display_pnl(pos, total_pnl)
                exit_trigger_price = None
                if 'STOP' in events and exit_context is not None:
                    exit_trigger_price = exit_context.get('trigger_price')
                trade_history.append(
                    {
                        'Date': today.strftime('%Y-%m-%d'),
                        'Ticker': ticker,
                        'Type': t_type,
                        '成交價': None if exit_context is None else exit_context['exec_price'],
                        '停損價': exit_trigger_price,
                        '股數': None if exit_context is None else exit_context['qty'],
                        '單筆損益': _round_money_for_history(display_exit_pnl),
                        '該筆總損益': _round_money_for_history(total_pnl),
                        'R_Multiple': total_r,
                        'Risk': params.fixed_risk,
                    }
                )
                register_display_realized_pnl(pos, display_exit_pnl)

            tickers_to_remove.append(ticker)
        elif 'MISSED_SELL' in events:
            total_missed_sells += 1
            if not is_training:
                sell_block_reason = get_exit_sell_block_reason(
                    get_fast_value(fast_df, 'Open', pos=t_pos),
                    get_fast_value(fast_df, 'High', pos=t_pos),
                    get_fast_value(fast_df, 'Low', pos=t_pos),
                    get_fast_close(fast_df, pos=t_pos),
                    get_fast_value(fast_df, 'Volume', pos=t_pos),
                    get_fast_close(fast_df, pos=y_pos),
                    ticker=ticker,
                )
                reason_note = {
                    'NO_VOLUME': '零量，當日無法賣出',
                    'LOCKED_DOWN': '一字跌停鎖死，當日無法賣出',
                }.get(sell_block_reason, '賣出受阻，當日無法賣出')
                trade_history.append(
                    {
                        'Date': today.strftime('%Y-%m-%d'),
                        'Ticker': ticker,
                        'Type': '錯失賣出',
                        '成交價': None,
                        '停損價': pos.get('trailing_stop', pos.get('initial_stop')),
                        '參考收盤價': get_fast_close(fast_df, pos=t_pos),
                        '股數': pos.get('qty'),
                        '單筆損益': 0.0,
                        '該筆總損益': _round_money_for_history(pos['realized_pnl']),
                        'R_Multiple': 0.0,
                        'Risk': params.fixed_risk,
                        '備註': reason_note,
                    }
                )

    for ticker in tickers_to_remove:
        del portfolio[ticker]
        sold_today.add(ticker)

    return cash, total_missed_sells, normal_trade_count, extended_trade_count


def closeout_open_positions(
    portfolio,
    cash,
    params,
    trade_history,
    is_training,
    closed_trades_stats,
    normal_trade_count,
    extended_trade_count,
    last_date,
):
    final_cash = cash

    for ticker in sorted(list(portfolio.keys())):
        pos = portfolio[ticker]
        raw_exit_price = pos.get('last_px', pos.get('pure_buy_price', pos['entry']))
        exec_price = adjust_long_sell_fill_price(raw_exit_price, ticker=ticker)
        sell_ledger = build_sell_ledger_from_price(
            exec_price,
            pos['qty'],
            params,
            ticker=ticker,
            security_profile=pos.get('security_profile'),
            trade_date=last_date,
        )
        final_cash += sell_ledger['net_sell_total_milli']

        pnl_milli = sell_ledger['net_sell_total_milli'] - pos['remaining_cost_basis_milli']
        total_pnl_milli = pos['realized_pnl_milli'] + pnl_milli
        total_pnl = milli_to_money(total_pnl_milli)
        display_tail_pnl = calc_reconciled_exit_display_pnl(pos, total_pnl)
        total_r = calc_ratio_from_milli(total_pnl_milli, pos.get('initial_risk_total_milli', 0))
        closed_trades_stats.append(
            {'pnl': total_pnl, 'r_mult': total_r, 'entry_type': pos.get('entry_type', 'normal')}
        )
        if is_extended_entry_type(pos.get('entry_type', 'normal')):
            extended_trade_count += 1
        else:
            normal_trade_count += 1

        if not is_training:
            trade_history.append(
                {
                    'Date': last_date.strftime('%Y-%m-%d') if last_date else '',
                    'Ticker': ticker,
                    'Type': '期末強制結算',
                    '成交價': exec_price,
                    '股數': pos['qty'],
                    '單筆損益': _round_money_for_history(display_tail_pnl),
                    '該筆總損益': _round_money_for_history(total_pnl),
                    'R_Multiple': total_r,
                    'Risk': params.fixed_risk,
                }
            )
            register_display_realized_pnl(pos, display_tail_pnl)

    return final_cash, normal_trade_count, extended_trade_count
