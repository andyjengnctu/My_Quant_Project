from core.v16_buy_sort import calc_buy_sort_value
from core.v16_config import BUY_SORT_METHOD
from core.v16_core import (
    adjust_long_sell_fill_price,
    calc_net_sell_price,
    execute_bar_step,
    get_exit_sell_block_reason,
)
from core.v16_portfolio_fast_data import (
    get_fast_close,
    get_fast_pos,
    get_fast_value,
    get_pit_stats_from_index,
    is_extended_entry_type,
)


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
            ret = (pt_y_close - pos['entry']) / pos['entry']

            holding_cost = pos['entry'] * pos['qty']
            _, holding_ev, holding_win_rate, holding_trade_count = get_pit_stats_from_index(
                pit_stats_index[pt],
                today,
                params,
            )
            holding_sort_value = calc_buy_sort_value(
                BUY_SORT_METHOD,
                holding_ev,
                holding_cost,
                holding_win_rate,
                holding_trade_count,
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
        est_sell_px = adjust_long_sell_fill_price(w_open)
        est_freed_cash = calc_net_sell_price(est_sell_px, pos['qty'], params) * pos['qty']

        pnl = est_freed_cash - (pos['entry'] * pos['qty'])
        cash += est_freed_cash

        total_pnl = pos['realized_pnl'] + pnl
        total_r = total_pnl / pos['initial_risk_total'] if pos['initial_risk_total'] > 0 else 0
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
                    '單筆損益': pnl,
                    '該筆總損益': total_pnl,
                    'R_Multiple': total_r,
                    'Risk': params.fixed_risk,
                }
            )

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

        pos, freed_cash, pnl_realized, events = execute_bar_step(
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
        )
        cash += freed_cash

        if 'TP_HALF' in events and not is_training:
            trade_history.append(
                {
                    'Date': today.strftime('%Y-%m-%d'),
                    'Ticker': ticker,
                    'Type': '半倉停利',
                    '單筆損益': pnl_realized,
                    'R_Multiple': 0.0,
                    'Risk': params.fixed_risk,
                }
            )

        if 'STOP' in events or 'IND_SELL' in events:
            total_pnl = pos['realized_pnl']
            total_r = total_pnl / pos['initial_risk_total'] if pos['initial_risk_total'] > 0 else 0
            closed_trades_stats.append(
                {'pnl': total_pnl, 'r_mult': total_r, 'entry_type': pos.get('entry_type', 'normal')}
            )
            if is_extended_entry_type(pos.get('entry_type', 'normal')):
                extended_trade_count += 1
            else:
                normal_trade_count += 1

            if not is_training:
                t_type = '全倉結算(停損)' if 'STOP' in events else '全倉結算(指標)'
                trade_history.append(
                    {
                        'Date': today.strftime('%Y-%m-%d'),
                        'Ticker': ticker,
                        'Type': t_type,
                        '單筆損益': pnl_realized,
                        '該筆總損益': total_pnl,
                        'R_Multiple': total_r,
                        'Risk': params.fixed_risk,
                    }
                )

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
                        '單筆損益': 0.0,
                        '該筆總損益': pos['realized_pnl'],
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
        exec_price = adjust_long_sell_fill_price(raw_exit_price)
        net_price = calc_net_sell_price(exec_price, pos['qty'], params)
        final_cash += net_price * pos['qty']

        pnl = (net_price - pos['entry']) * pos['qty']
        total_pnl = pos['realized_pnl'] + pnl
        total_r = total_pnl / pos['initial_risk_total'] if pos['initial_risk_total'] > 0 else 0
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
                    '單筆損益': pnl,
                    '該筆總損益': total_pnl,
                    'R_Multiple': total_r,
                    'Risk': params.fixed_risk,
                }
            )

    return final_cash, normal_trade_count, extended_trade_count
