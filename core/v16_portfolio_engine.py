import pandas as pd
import time
from core.v16_core import adjust_long_sell_fill_price, calc_entry_price, calc_net_sell_price, execute_bar_step, build_normal_candidate_plan, create_signal_tracking_state, build_extended_candidate_plan_from_signal, build_cash_capped_entry_plan, execute_pre_market_entry_plan, should_clear_extended_signal, get_exit_sell_block_reason
from core.v16_config import EV_CALC_METHOD, BUY_SORT_METHOD
from core.v16_buy_sort import calc_buy_sort_value
from core.v16_portfolio_fast_data import (
    build_normal_setup_index,
    build_trade_stats_index,
    calc_mark_to_market_equity,
    get_fast_close,
    get_fast_close_on_or_before,
    get_fast_pos,
    get_fast_value,
    get_pit_stats_from_index,
    has_fast_date,
    is_extended_entry_type,
)
from core.v16_portfolio_stats import (
    build_benchmark_full_year_return_stats,
    build_full_year_return_stats,
    calc_annual_return_pct,
    calc_curve_stats,
    calc_sim_years,
    find_sim_start_idx,
)
from core.v16_portfolio_stats import calc_portfolio_score

def run_portfolio_timeline(all_dfs_fast, all_standalone_logs, sorted_dates, start_year, params, max_positions, enable_rotation, benchmark_ticker="0050", benchmark_data=None, is_training=True, profile_stats=None, verbose=True):
    t_portfolio_start = time.perf_counter() if profile_stats is not None else None
    candidate_scan_sec = 0.0
    day_loop_sec = 0.0
    rotation_sec = 0.0
    settle_sec = 0.0
    buy_sec = 0.0
    equity_mark_sec = 0.0
    build_trade_index_sec = 0.0
    ticker_dates_sec = 0.0
    closeout_sec = 0.0

    t0 = time.perf_counter() if profile_stats is not None else None
    start_idx = find_sim_start_idx(sorted_dates, start_year)
    if profile_stats is not None:
        ticker_dates_sec = time.perf_counter() - t0

    t0 = time.perf_counter() if profile_stats is not None else None
    pit_stats_index = {t: build_trade_stats_index(logs) for t, logs in all_standalone_logs.items()}
    normal_setup_index = build_normal_setup_index(all_dfs_fast)
    if profile_stats is not None:
        build_trade_index_sec = time.perf_counter() - t0

    initial_capital = params.initial_capital
    cash = initial_capital
    portfolio = {}
    active_extended_signals = {}
    trade_history, equity_curve, closed_trades_stats = [], [], []
    normal_trade_count, extended_trade_count = 0, 0
    peak_equity, max_drawdown, current_equity = initial_capital, 0.0, initial_capital
    total_exposure, sim_days, total_missed_buys, total_missed_sells, max_exp = 0.0, 0, 0, 0, 0.0
    monthly_equities, bm_monthly_equities = [initial_capital], []
    yesterday_equity = initial_capital
    year_start_equity, year_end_equity = {}, {}
    year_first_sim_date, year_last_sim_date = {}, {}

    current_month = sorted_dates[start_idx].month if start_idx < len(sorted_dates) else 1

    current_bm_px = None
    benchmark_start_price = None
    bm_peak_price = None
    if benchmark_data and start_idx < len(sorted_dates):
        benchmark_start_price, benchmark_anchor_date = get_fast_close_on_or_before(benchmark_data, sorted_dates[start_idx])
        if benchmark_start_price is not None:
            bm_peak_price = benchmark_start_price
            bm_monthly_equities.append(benchmark_start_price)
            if benchmark_anchor_date == sorted_dates[start_idx]:
                current_bm_px = benchmark_start_price

    yesterday_bm_px, bm_max_drawdown, bm_ret_pct = current_bm_px, 0.0, 0.0

    for i in range(start_idx, len(sorted_dates)):
        t_day_start = time.perf_counter() if profile_stats is not None else None
        sim_days += 1
        today = sorted_dates[i]

        if today.year not in year_start_equity:
            year_start_equity[today.year] = current_equity
            year_first_sim_date[today.year] = pd.Timestamp(today)

        available_cash = cash
        sizing_equity = current_equity if getattr(params, 'use_compounding', True) else initial_capital

        sold_today = set()

        if verbose and (not is_training) and i % 20 == 0:
            exp = ((current_equity - cash) / current_equity) * 100 if current_equity > 0 else 0
            print(f"\033[90m⏳ 推進中: {today.strftime('%Y-%m')} | 資產: {current_equity:,.0f} | 水位: {exp:>5.1f}%...\033[0m", end="\r", flush=True)

        t0 = time.perf_counter() if profile_stats is not None else None
        candidates_today = []
        orderable_candidates_today = []
        normal_setup_tickers_today = set()

        for ticker, y_pos, t_pos in sorted(normal_setup_index.get(today, []), key=lambda x: x[0]):
            normal_setup_tickers_today.add(ticker)
            if ticker in portfolio or ticker in sold_today:
                continue

            fast_df = all_dfs_fast[ticker]
            y_buy_limit = get_fast_value(fast_df, 'buy_limit', pos=y_pos)
            y_atr = get_fast_value(fast_df, 'ATR', pos=y_pos)

            signal_state = create_signal_tracking_state(y_buy_limit, y_atr, params)
            if signal_state is not None:
                active_extended_signals[ticker] = signal_state

            is_candidate, ev, win_rate, trade_count = get_pit_stats_from_index(
                pit_stats_index[ticker], today, params
            )
            if not is_candidate:
                continue

            candidate_plan = build_normal_candidate_plan(y_buy_limit, y_atr, sizing_equity, params)
            if candidate_plan is None:
                continue

            est_limit_px = candidate_plan['limit_price']
            est_init_sl = candidate_plan['init_sl']
            est_init_trail = candidate_plan['init_trail']
            est_qty = candidate_plan['qty']
            est_cost = calc_entry_price(est_limit_px, est_qty, params) * est_qty if est_qty > 0 else 0.0
            sort_value = calc_buy_sort_value(BUY_SORT_METHOD, ev, est_cost, win_rate, trade_count)
            candidate_row = {
                'ticker': ticker,
                'type': 'normal',
                'limit_px': est_limit_px,
                'ev': ev,
                'y_atr': y_atr,
                'today_pos': t_pos,
                'yesterday_pos': y_pos,
                'qty': est_qty,
                'proj_cost': est_cost,
                'sort_value': sort_value,
                'hist_win_rate': win_rate,
                'hist_trade_count': trade_count,
                'init_sl': est_init_sl,
                'init_trail': est_init_trail,
                'is_orderable': candidate_plan['is_orderable'],
            }
            candidates_today.append(candidate_row)
            if candidate_row['is_orderable']:
                orderable_candidates_today.append(candidate_row)

        for ticker in sorted(list(active_extended_signals.keys())):
            if ticker in portfolio or ticker in sold_today or ticker in normal_setup_tickers_today:
                continue

            fast_df = all_dfs_fast.get(ticker)
            if fast_df is None:
                continue

            t_pos = get_fast_pos(fast_df, today)
            if t_pos <= 0:
                continue
            y_pos = t_pos - 1

            is_candidate, ev, win_rate, trade_count = get_pit_stats_from_index(pit_stats_index[ticker], today, params)
            if not is_candidate:
                continue

            reference_price = get_fast_close(fast_df, pos=y_pos)
            candidate_plan = build_extended_candidate_plan_from_signal(
                active_extended_signals[ticker],
                reference_price,
                sizing_equity,
                params,
            )
            if candidate_plan is None:
                continue

            est_limit_px = candidate_plan['limit_price']
            est_init_sl = candidate_plan['init_sl']
            est_init_trail = candidate_plan['init_trail']
            est_qty = candidate_plan['qty']
            est_cost = calc_entry_price(est_limit_px, est_qty, params) * est_qty if est_qty > 0 else 0.0
            sort_value = calc_buy_sort_value(BUY_SORT_METHOD, ev, est_cost, win_rate, trade_count)
            candidate_row = {
                'ticker': ticker,
                'type': 'extended',
                'limit_px': est_limit_px,
                'ev': ev,
                'y_atr': candidate_plan['orig_atr'],
                'today_pos': t_pos,
                'yesterday_pos': y_pos,
                'qty': est_qty,
                'proj_cost': est_cost,
                'sort_value': sort_value,
                'hist_win_rate': win_rate,
                'hist_trade_count': trade_count,
                'signal_state': active_extended_signals[ticker],
                'init_sl': est_init_sl,
                'init_trail': est_init_trail,
                'is_orderable': candidate_plan['is_orderable'],
            }
            candidates_today.append(candidate_row)
            if candidate_row['is_orderable']:
                orderable_candidates_today.append(candidate_row)

        candidates_today.sort(key=lambda x: (x['sort_value'], x['ticker']), reverse=True)
        orderable_candidates_today.sort(key=lambda x: (x['sort_value'], x['ticker']), reverse=True)
        if profile_stats is not None:
            candidate_scan_sec += time.perf_counter() - t0

        t0 = time.perf_counter() if profile_stats is not None else None
        if len(portfolio) == max_positions and enable_rotation and orderable_candidates_today:
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
                    _, holding_ev, holding_win_rate, holding_trade_count = get_pit_stats_from_index(pit_stats_index[pt], today, params)
                    holding_sort_value = calc_buy_sort_value(BUY_SORT_METHOD, holding_ev, holding_cost, holding_win_rate, holding_trade_count)
                    is_strategically_better = cand['sort_value'] > holding_sort_value

                    if is_strategically_better and ret < lowest_ret:
                        lowest_ret = ret
                        weakest_ticker = pt

                if weakest_ticker:
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

                    if sell_block_reason is None:
                        pos = portfolio[weakest_ticker]
                        est_sell_px = adjust_long_sell_fill_price(w_open)
                        est_freed_cash = calc_net_sell_price(est_sell_px, pos['qty'], params) * pos['qty']

                        # # (AI註: 問題3 版本B - rotation 改成 T 日只賣弱股、T+1 才重新評估可買標的，不用今天候選的 proj_cost 綁定是否先賣弱股)
                        pnl = est_freed_cash - (pos['entry'] * pos['qty'])
                        cash += est_freed_cash

                        total_pnl = pos['realized_pnl'] + pnl
                        total_r = total_pnl / pos['initial_risk_total'] if pos['initial_risk_total'] > 0 else 0
                        closed_trades_stats.append({'pnl': total_pnl, 'r_mult': total_r, 'entry_type': pos.get('entry_type', 'normal')})
                        if is_extended_entry_type(pos.get('entry_type', 'normal')):
                            extended_trade_count += 1
                        else:
                            normal_trade_count += 1

                        if not is_training:
                            trade_history.append({
                                "Date": today.strftime('%Y-%m-%d'),
                                "Ticker": weakest_ticker,
                                "Type": "汰弱賣出(Open, T+1再評估買進)",
                                "單筆損益": pnl,
                                "該筆總損益": total_pnl,
                                "R_Multiple": total_r,
                                "Risk": params.fixed_risk
                            })

                        del portfolio[weakest_ticker]
                        sold_today.add(weakest_ticker)
                        break
        if profile_stats is not None:
            rotation_sec += time.perf_counter() - t0

        t0 = time.perf_counter() if profile_stats is not None else None
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
                trade_history.append({
                    "Date": today.strftime('%Y-%m-%d'),
                    "Ticker": ticker,
                    "Type": "半倉停利",
                    "單筆損益": pnl_realized,
                    "R_Multiple": 0.0,
                    "Risk": params.fixed_risk
                })

            if 'STOP' in events or 'IND_SELL' in events:
                total_pnl = pos['realized_pnl']
                total_r = total_pnl / pos['initial_risk_total'] if pos['initial_risk_total'] > 0 else 0
                closed_trades_stats.append({'pnl': total_pnl, 'r_mult': total_r, 'entry_type': pos.get('entry_type', 'normal')})
                if is_extended_entry_type(pos.get('entry_type', 'normal')):
                    extended_trade_count += 1
                else:
                    normal_trade_count += 1

                if not is_training:
                    t_type = "全倉結算(停損)" if 'STOP' in events else "全倉結算(指標)"
                    trade_history.append({
                        "Date": today.strftime('%Y-%m-%d'),
                        "Ticker": ticker,
                        "Type": t_type,
                        "單筆損益": pnl_realized,
                        "該筆總損益": total_pnl,
                        "R_Multiple": total_r,
                        "Risk": params.fixed_risk
                    })

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
                    trade_history.append({
                        "Date": today.strftime('%Y-%m-%d'),
                        "Ticker": ticker,
                        "Type": "錯失賣出",
                        "單筆損益": 0.0,
                        "該筆總損益": pos['realized_pnl'],
                        "R_Multiple": 0.0,
                        "Risk": params.fixed_risk,
                        "備註": reason_note,
                    })

        for t in tickers_to_remove:
            del portfolio[t]
            sold_today.add(t)
        if profile_stats is not None:
            settle_sec += time.perf_counter() - t0

        t0 = time.perf_counter() if profile_stats is not None else None
        # # (AI註: 問題3 版本B - sold_today 也包含 rotation 當日賣出，名額必須凍結到下一交易日)
        pre_market_occupied = len(portfolio) + len(sold_today)
        remaining_orderable_candidates = list(orderable_candidates_today)

        while remaining_orderable_candidates and pre_market_occupied < max_positions:
            chosen_idx = 0
            chosen_entry_plan = None

            if BUY_SORT_METHOD == 'PROJ_COST':
                chosen_key = None
                for cand_idx, probe_cand in enumerate(remaining_orderable_candidates):
                    probe_entry_plan = build_cash_capped_entry_plan(
                        {
                            'limit_price': probe_cand['limit_px'],
                            'init_sl': probe_cand['init_sl'],
                            'init_trail': probe_cand['init_trail'],
                        },
                        available_cash,
                        params,
                    )
                    if probe_entry_plan is None:
                        continue

                    probe_key = (probe_entry_plan['reserved_cost'], probe_cand['ticker'])
                    if chosen_key is None or probe_key > chosen_key:
                        chosen_key = probe_key
                        chosen_idx = cand_idx
                        chosen_entry_plan = probe_entry_plan

                if chosen_entry_plan is None:
                    break

            cand = remaining_orderable_candidates.pop(chosen_idx)
            if chosen_entry_plan is None:
                chosen_entry_plan = build_cash_capped_entry_plan(
                    {
                        'limit_price': cand['limit_px'],
                        'init_sl': cand['init_sl'],
                        'init_trail': cand['init_trail'],
                    },
                    available_cash,
                    params,
                )
                if chosen_entry_plan is None:
                    continue

            fast_df = all_dfs_fast[cand['ticker']]
            t_pos = cand['today_pos']
            y_pos = cand['yesterday_pos']
            t_open = get_fast_value(fast_df, 'Open', pos=t_pos)
            t_high = get_fast_value(fast_df, 'High', pos=t_pos)
            t_low = get_fast_value(fast_df, 'Low', pos=t_pos)
            t_close = get_fast_close(fast_df, pos=t_pos)
            t_volume = get_fast_value(fast_df, 'Volume', pos=t_pos)
            y_close = get_fast_close(fast_df, pos=y_pos)

            reserved_cost = chosen_entry_plan['reserved_cost']
            available_cash -= reserved_cost
            pre_market_occupied += 1

            entry_result = execute_pre_market_entry_plan(
                entry_plan=chosen_entry_plan,
                t_open=t_open,
                t_high=t_high,
                t_low=t_low,
                t_close=t_close,
                t_volume=t_volume,
                y_close=y_close,
                params=params,
                entry_type=cand['type'],
            )

            if entry_result['filled']:
                qty = chosen_entry_plan['qty']
                actual_total_cost = entry_result['entry_price'] * qty
                cash -= actual_total_cost
                portfolio[cand['ticker']] = entry_result['position']

                if cand['ticker'] in active_extended_signals:
                    del active_extended_signals[cand['ticker']]
                if not is_training:
                    trade_history.append({
                        "Date": today.strftime('%Y-%m-%d'),
                        "Ticker": cand['ticker'],
                        "Type": f"買進 (EV:{cand['ev']:.2f}R)",
                        "單筆損益": 0.0,
                        "該筆總損益": 0.0,
                        "R_Multiple": 0.0,
                        "Risk": params.fixed_risk
                    })
            elif entry_result['count_as_missed_buy']:
                total_missed_buys += 1
                if not is_training:
                    miss_buy_type = "錯失買進(延續候選)" if cand['type'] == 'extended' else "錯失買進(新訊號)"
                    trade_history.append({
                        "Date": today.strftime('%Y-%m-%d'),
                        "Ticker": cand['ticker'],
                        "Type": miss_buy_type,
                        "單筆損益": 0.0,
                        "該筆總損益": 0.0,
                        "R_Multiple": 0.0,
                        "Risk": params.fixed_risk,
                        "備註": f"預掛限價 {chosen_entry_plan['limit_price']:.2f} 未成交",
                        "投入總金額": reserved_cost,
                    })
        if profile_stats is not None:
            buy_sec += time.perf_counter() - t0

        for ticker in sorted(list(active_extended_signals.keys())):
            if ticker in portfolio:
                del active_extended_signals[ticker]
                continue

            fast_df = all_dfs_fast.get(ticker)
            if fast_df is None:
                continue

            t_pos = get_fast_pos(fast_df, today)
            if t_pos < 0:
                continue

            t_low = get_fast_value(fast_df, 'Low', pos=t_pos)
            if should_clear_extended_signal(active_extended_signals[ticker], t_low):
                del active_extended_signals[ticker]

        t0 = time.perf_counter() if profile_stats is not None else None
        today_equity = calc_mark_to_market_equity(cash, portfolio, all_dfs_fast, today, params)
        if profile_stats is not None:
            equity_mark_sec += time.perf_counter() - t0

        current_equity = today_equity
        invested_capital = today_equity - cash
        exposure_pct = (invested_capital / today_equity) * 100 if today_equity > 0 else 0
        total_exposure += exposure_pct
        if exposure_pct > max_exp:
            max_exp = exposure_pct

        strategy_ret_pct = (today_equity - initial_capital) / initial_capital * 100

        if benchmark_data and benchmark_start_price is not None and has_fast_date(benchmark_data, today):
            current_bm_px = get_fast_close(benchmark_data, date=today)
            bm_ret_pct = (current_bm_px - benchmark_start_price) / benchmark_start_price * 100 if benchmark_start_price > 0 else 0.0

            if bm_peak_price is not None:
                if current_bm_px > bm_peak_price:
                    bm_peak_price = current_bm_px
                current_bm_drawdown = (bm_peak_price - current_bm_px) / bm_peak_price * 100
                if current_bm_drawdown > bm_max_drawdown:
                    bm_max_drawdown = current_bm_drawdown

        if today.month != current_month:
            monthly_equities.append(yesterday_equity)
            if yesterday_bm_px is not None:
                bm_monthly_equities.append(yesterday_bm_px)
            current_month = today.month

        yesterday_equity = today_equity
        yesterday_bm_px = current_bm_px
        year_end_equity[today.year] = today_equity
        year_last_sim_date[today.year] = pd.Timestamp(today)

        if not is_training:
            equity_curve.append({
                "Date": today.strftime('%Y-%m-%d'),
                "Equity": today_equity,
                "Invested_Amount": invested_capital,
                "Exposure_Pct": exposure_pct,
                "Strategy_Return_Pct": strategy_ret_pct,
                f"Benchmark_{benchmark_ticker}_Pct": bm_ret_pct
            })

        if today_equity > peak_equity:
            peak_equity = today_equity
        drawdown = (peak_equity - today_equity) / peak_equity * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown

        if profile_stats is not None:
            day_loop_sec += time.perf_counter() - t_day_start

    # # (AI註: 月底權益應以期末強制結算後的真實 final equity 為準，故移到 closeout 後再 append)

    t0 = time.perf_counter() if profile_stats is not None else None
    final_cash = cash
    last_date = sorted_dates[-1] if len(sorted_dates) > 0 else None

    for ticker in sorted(list(portfolio.keys())):
        pos = portfolio[ticker]
        raw_exit_price = pos.get('last_px', pos.get('pure_buy_price', pos['entry']))
        exec_price = adjust_long_sell_fill_price(raw_exit_price)
        net_price = calc_net_sell_price(exec_price, pos['qty'], params)
        final_cash += net_price * pos['qty']

        pnl = (net_price - pos['entry']) * pos['qty']
        total_pnl = pos['realized_pnl'] + pnl
        total_r = total_pnl / pos['initial_risk_total'] if pos['initial_risk_total'] > 0 else 0
        closed_trades_stats.append({'pnl': total_pnl, 'r_mult': total_r, 'entry_type': pos.get('entry_type', 'normal')})
        if is_extended_entry_type(pos.get('entry_type', 'normal')):
            extended_trade_count += 1
        else:
            normal_trade_count += 1

        if not is_training:
            trade_history.append({
                "Date": last_date.strftime('%Y-%m-%d') if last_date else "",
                "Ticker": ticker,
                "Type": "期末強制結算",
                "單筆損益": pnl,
                "該筆總損益": total_pnl,
                "R_Multiple": total_r,
                "Risk": params.fixed_risk
            })

    if profile_stats is not None:
        closeout_sec = time.perf_counter() - t0

    today_equity = final_cash
    if last_date is not None and last_date.year in year_end_equity:
        year_end_equity[last_date.year] = today_equity

    # # (AI註: 期末強制結算後補做一次 peak / drawdown 更新，避免 final closeout 對 MDD 漏算)
    if today_equity > peak_equity:
        peak_equity = today_equity
    drawdown = (peak_equity - today_equity) / peak_equity * 100 if peak_equity > 0 else 0.0
    if drawdown > max_drawdown:
        max_drawdown = drawdown

    if len(sorted_dates) > start_idx:
        monthly_equities.append(today_equity)
        if current_bm_px is not None:
            bm_monthly_equities.append(current_bm_px)

    total_return = (today_equity - initial_capital) / initial_capital * 100

    if not is_training and len(equity_curve) > 0:
        equity_curve[-1]['Equity'] = today_equity
        equity_curve[-1]['Strategy_Return_Pct'] = total_return
        equity_curve[-1]['Invested_Amount'] = 0.0
        equity_curve[-1]['Exposure_Pct'] = 0.0

    t0 = time.perf_counter() if profile_stats is not None else None
    r_squared, monthly_win_rate = calc_curve_stats(monthly_equities)
    bm_r_squared, bm_monthly_win_rate = calc_curve_stats(bm_monthly_equities)
    curve_stats_sec = (time.perf_counter() - t0) if profile_stats is not None else 0.0

    trade_count = len(closed_trades_stats)
    if trade_count > 0:
        wins = [t for t in closed_trades_stats if t['pnl'] > 0]
        losses = [t for t in closed_trades_stats if t['pnl'] <= 0]

        win_count = len(wins)
        loss_count = len(losses)
        win_rate = (win_count / trade_count) * 100
        avg_win_amount = sum(t['pnl'] for t in wins) / win_count if win_count > 0 else 0.0
        avg_loss_amount = abs(sum(t['pnl'] for t in losses) / loss_count) if loss_count > 0 else 0.0
        pf_payoff = (avg_win_amount / avg_loss_amount) if avg_loss_amount > 0 else (99.9 if avg_win_amount > 0 else 0.0)

        avg_win_r = sum(t['r_mult'] for t in wins) / win_count if win_count > 0 else 0.0
        avg_loss_r = abs(sum(t['r_mult'] for t in losses) / loss_count) if loss_count > 0 else 0.0
        if EV_CALC_METHOD == 'B':
            payoff_for_ev = min(10.0, (avg_win_r / avg_loss_r)) if avg_loss_r > 0 else (99.9 if avg_win_r > 0 else 0.0)
            pf_ev = (win_rate / 100.0 * payoff_for_ev) - (1 - win_rate / 100.0)
        else:
            pf_ev = sum(t['r_mult'] for t in closed_trades_stats) / trade_count
    else:
        win_rate, pf_ev, pf_payoff = 0.0, 0.0, 0.0

    avg_exp = total_exposure / sim_days if sim_days > 0 else 0.0
    sim_years = calc_sim_years(sorted_dates, start_idx)
    annual_trades = (trade_count / sim_years) if sim_years > 0 else 0.0
    total_reserved_entries = normal_trade_count + extended_trade_count
    reserved_buy_fill_rate = (total_reserved_entries / (total_reserved_entries + total_missed_buys) * 100.0) if (total_reserved_entries + total_missed_buys) > 0 else 0.0
    annual_return_pct = calc_annual_return_pct(initial_capital, today_equity, sim_years)

    bm_start_value = float(benchmark_start_price) if benchmark_start_price is not None else 0.0
    bm_end_value = bm_start_value * (1.0 + bm_ret_pct / 100.0) if bm_start_value > 0 else 0.0
    bm_annual_return_pct = calc_annual_return_pct(bm_start_value, bm_end_value, sim_years)

    yearly_stats = build_full_year_return_stats(
        sorted_dates=sorted_dates,
        year_start_equity=year_start_equity,
        year_end_equity=year_end_equity,
        year_first_sim_date=year_first_sim_date,
        year_last_sim_date=year_last_sim_date,
    )
    bm_yearly_stats = build_benchmark_full_year_return_stats(
        sorted_dates=sorted_dates,
        benchmark_data=benchmark_data,
        yearly_return_rows=yearly_stats['yearly_return_rows'],
    )

    if profile_stats is not None:
        profile_stats['portfolio_wall_sec'] = time.perf_counter() - t_portfolio_start
        profile_stats['portfolio_ticker_dates_sec'] = ticker_dates_sec
        profile_stats['portfolio_build_trade_index_sec'] = build_trade_index_sec
        profile_stats['portfolio_day_loop_sec'] = day_loop_sec
        profile_stats['portfolio_candidate_scan_sec'] = candidate_scan_sec
        profile_stats['portfolio_rotation_sec'] = rotation_sec
        profile_stats['portfolio_settle_sec'] = settle_sec
        profile_stats['portfolio_buy_sec'] = buy_sec
        profile_stats['portfolio_equity_mark_sec'] = equity_mark_sec
        profile_stats['portfolio_closeout_sec'] = closeout_sec
        profile_stats['curve_stats_sec'] = curve_stats_sec
        profile_stats['sim_years'] = sim_years
        profile_stats['annual_return_pct'] = annual_return_pct
        profile_stats['bm_annual_return_pct'] = bm_annual_return_pct
        profile_stats.update(yearly_stats)
        profile_stats.update(bm_yearly_stats)

    if is_training:
        return total_return, max_drawdown, trade_count, today_equity, avg_exp, max_exp, bm_ret_pct, bm_max_drawdown, win_rate, pf_ev, pf_payoff, total_missed_buys, total_missed_sells, r_squared, monthly_win_rate, bm_r_squared, bm_monthly_win_rate, normal_trade_count, extended_trade_count, annual_trades, reserved_buy_fill_rate, annual_return_pct, bm_annual_return_pct

    df_equity = pd.DataFrame(equity_curve)
    df_trades = pd.DataFrame(trade_history)
    final_bm_return = df_equity.iloc[-1][f"Benchmark_{benchmark_ticker}_Pct"] if not df_equity.empty else 0.0
    return df_equity, df_trades, total_return, max_drawdown, trade_count, win_rate, pf_ev, pf_payoff, today_equity, avg_exp, max_exp, final_bm_return, bm_max_drawdown, total_missed_buys, total_missed_sells, r_squared, monthly_win_rate, bm_r_squared, bm_monthly_win_rate, normal_trade_count, extended_trade_count, annual_trades, reserved_buy_fill_rate, annual_return_pct, bm_annual_return_pct
