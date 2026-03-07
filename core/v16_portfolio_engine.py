import pandas as pd
import numpy as np
import bisect
from core.v16_core import generate_signals, adjust_to_tick, calc_net_sell_price, calc_position_size, calc_entry_price, execute_bar_step, evaluate_chase_condition, run_v16_backtest
from core.v16_config import EV_CALC_METHOD, BUY_SORT_METHOD

def calc_portfolio_score(sys_ret, sys_mdd, m_win_rate, r_sq):
    from core.v16_config import SCORE_CALC_METHOD
    romd = sys_ret / (abs(sys_mdd) + 0.0001)
    if SCORE_CALC_METHOD == 'LOG_R2': return romd * (m_win_rate / 100.0) * r_sq
    return romd

def calc_curve_stats(eq_list):
    r_squared, monthly_win_rate = 0.0, 0.0
    if len(eq_list) > 2:
        eq_array = np.array(eq_list)
        x = np.arange(len(eq_array))
        if np.std(eq_array) > 0:
            valid_idx = eq_array > 0
            if np.any(valid_idx):
                log_eq = np.log(eq_array[valid_idx])
                x_valid = x[valid_idx]
                if np.std(log_eq) > 0 and len(x_valid) > 1:
                    r_matrix = np.corrcoef(x_valid, log_eq)
                    r_squared = r_matrix[0, 1] ** 2 if not np.isnan(r_matrix[0, 1]) else 0.0
        eq_series = pd.Series(eq_list)
        monthly_rets = eq_series.pct_change().dropna()
        if len(monthly_rets) > 0: monthly_win_rate = (len(monthly_rets[monthly_rets > 0]) / len(monthly_rets)) * 100
    return r_squared, monthly_win_rate

def prep_stock_data_and_trades(df, params):
    df = df.copy()
    ATR_main, buyCondition, sellCondition, buy_limits = generate_signals(df, params)
    df['ATR'] = ATR_main
    df['is_setup'] = buyCondition
    df['ind_sell_signal'] = sellCondition
    df['buy_limit'] = buy_limits
    
    # # (AI註: 問題1修復 - 強制調用核心引擎，取得真正的 Standalone Trade Logs，拯救 PIT 失真)
    _, standalone_logs = run_v16_backtest(df, params, return_logs=True)
    return df, standalone_logs

def get_pit_stats(trade_logs, current_date, params):
    past_trades = [t for t in trade_logs if t['exit_date'] < current_date]
    trade_count = len(past_trades)
    min_trades_req = getattr(params, 'min_history_trades', 0)
    
    if trade_count < min_trades_req: return False, 0.0, 0.0 
    if trade_count == 0: return True, 0.0, 0.0
    
    wins = [t for t in past_trades if t['pnl'] > 0]
    win_rate = len(wins) / trade_count
    if EV_CALC_METHOD == 'B':
        avg_win_r = sum(t['r_mult'] for t in wins) / len(wins) if len(wins) > 0 else 0
        loss_count = trade_count - len(wins)
        avg_loss_r = abs(sum(t['r_mult'] for t in past_trades if t['pnl'] <= 0) / loss_count) if loss_count > 0 else 0
        payoff_for_ev = min(10.0, (avg_win_r / avg_loss_r)) if avg_loss_r > 0 else (99.9 if avg_win_r > 0 else 0.0)
        ev_to_sort = (win_rate * payoff_for_ev) - (1 - win_rate)
    else: ev_to_sort = sum(t['r_mult'] for t in past_trades) / trade_count

    is_candidate = (ev_to_sort > getattr(params, 'min_history_ev', 0.0)) and (win_rate >= getattr(params, 'min_history_win_rate', 0.30))
    return is_candidate, ev_to_sort, win_rate

def run_portfolio_timeline(all_dfs_fast, all_standalone_logs, sorted_dates, start_year, params, max_positions, enable_rotation, benchmark_ticker="0050", benchmark_data=None, is_training=True):
    start_dt = pd.to_datetime(f"{start_year}-01-01")
    start_idx = next((i for i, d in enumerate(sorted_dates) if d >= start_dt), len(sorted_dates))
    start_idx = max(1, start_idx)

    ticker_dates = {t: sorted(list(df.keys())) for t, df in all_dfs_fast.items()}

    initial_capital = params.initial_capital
    cash = initial_capital
    portfolio = {} 
    pending_chases = {} 
    trade_history, equity_curve, closed_trades_stats = [], [], []
    peak_equity, max_drawdown, current_equity = initial_capital, 0.0, initial_capital 
    total_exposure, sim_days, total_missed_buys, total_missed_sells, max_exp = 0.0, 0, 0, 0, 0.0  
    monthly_equities, bm_monthly_equities = [], []
    yesterday_equity = initial_capital
    
    current_month = sorted_dates[start_idx].month if start_idx < len(sorted_dates) else 1
    current_bm_px = benchmark_data[sorted_dates[start_idx]]['Close'] if benchmark_data and start_idx < len(sorted_dates) and sorted_dates[start_idx] in benchmark_data else None
    yesterday_bm_px, benchmark_start_price, bm_peak_price, bm_max_drawdown, bm_ret_pct = current_bm_px, None, None, 0.0, 0.0

    for i in range(start_idx, len(sorted_dates)):
        sim_days += 1
        today = sorted_dates[i]
        yesterday = sorted_dates[i-1]
        
        available_cash = cash
        sizing_equity = current_equity if getattr(params, 'use_compounding', True) else initial_capital
        
        sold_today = set() # # (AI註: 修復 Bug 2 - 每日重置，記錄今日賣出的標的)
        
        if not is_training and i % 20 == 0:
            exp = ((current_equity - cash) / current_equity) * 100 if current_equity > 0 else 0
            print(f"\033[90m⏳ 推進中: {today.strftime('%Y-%m')} | 資產: {current_equity:,.0f} | 水位: {exp:>5.1f}%...\033[0m", end="\r", flush=True)

        # 1. 執行昨日持倉的盤中結算
        tickers_to_remove = []
        for ticker in sorted(portfolio.keys()):
            pos = portfolio[ticker]
            fast_df = all_dfs_fast[ticker]
            if today not in fast_df: continue 
            
            d_list = ticker_dates[ticker]
            idx = bisect.bisect_left(d_list, today)
            if idx == 0: continue
            y_row = fast_df[d_list[idx - 1]]
            t_row = fast_df[today]
            
            pos, freed_cash, pnl_realized, events = execute_bar_step(
                pos, y_row['ATR'], y_row['ind_sell_signal'], y_row['Close'], 
                t_row['Open'], t_row['High'], t_row['Low'], t_row['Close'], params
            )
            cash += freed_cash
            
            if 'TP_HALF' in events and not is_training:
                trade_history.append({"Date": today.strftime('%Y-%m-%d'), "Ticker": ticker, "Type": "半倉停利", "單筆損益": pnl_realized, "R_Multiple": 0.0, "Risk": params.fixed_risk})
            if 'STOP' in events or 'IND_SELL' in events:
                total_pnl = pos['realized_pnl']
                total_r = total_pnl / pos['initial_risk_total'] if pos['initial_risk_total'] > 0 else 0
                closed_trades_stats.append({'pnl': total_pnl, 'r_mult': total_r})
                if not is_training: 
                    t_type = "全倉結算(停損)" if 'STOP' in events else "全倉結算(指標)"
                    trade_history.append({"Date": today.strftime('%Y-%m-%d'), "Ticker": ticker, "Type": t_type, "單筆損益": pnl_realized, "該筆總損益": total_pnl, "R_Multiple": total_r, "Risk": params.fixed_risk})
                tickers_to_remove.append(ticker)
            elif 'LOCKED_DOWN' in events: total_missed_sells += 1 

        for t in tickers_to_remove: 
            del portfolio[t]
            sold_today.add(t) # # (AI註: 修復 Bug 2 - 將常規賣出的股票加入今日已賣名單)

        # 2. 獲取當日候選標的
        candidates_today = []
        for ticker in sorted(all_dfs_fast.keys()):
            # # (AI註: 修復 Bug 2 - 排除現在持有與今天剛賣掉的股票，嚴禁同日買回)
            if ticker in portfolio or ticker in sold_today: continue 
            fast_df = all_dfs_fast[ticker]
            if today not in fast_df: continue
            
            d_list = ticker_dates[ticker]
            idx = bisect.bisect_left(d_list, today)
            if idx == 0: continue
            y_row = fast_df[d_list[idx - 1]]
            
            if y_row['is_setup']:
                # 使用 all_standalone_logs
                is_candidate, ev, win_rate = get_pit_stats(all_standalone_logs[ticker], today, params)
                if is_candidate:
                    # 嚴守盤前定錨
                    est_init_sl = adjust_to_tick(y_row['buy_limit'] - y_row['ATR'] * params.atr_times_init)
                    est_init_trail = adjust_to_tick(y_row['buy_limit'] - y_row['ATR'] * params.atr_times_trail)
                    est_qty = calc_position_size(y_row['buy_limit'], est_init_sl, sizing_equity, params.fixed_risk, params)
                    if est_qty > 0:
                        est_cost = calc_entry_price(y_row['buy_limit'], est_qty, params) * est_qty
                        candidates_today.append({
                            'ticker': ticker, 'type': 'normal', 'limit_px': y_row['buy_limit'], 'ev': ev, 
                            'y_atr': y_row['ATR'], 't_row': fast_df[today], 'y_row': y_row, 'qty': est_qty, 'proj_cost': est_cost,
                            'init_sl': est_init_sl, 'init_trail': est_init_trail
                        })
            
            elif ticker in pending_chases:
                chase = pending_chases[ticker]
                is_candidate, ev, win_rate = get_pit_stats(all_standalone_logs[ticker], today, params)
                if is_candidate:
                    est_qty = chase['qty'] 
                    if est_qty > 0:
                        est_cost = calc_entry_price(chase['chase_price'], est_qty, params) * est_qty
                        candidates_today.append({
                            'ticker': ticker, 'type': 'chase', 'limit_px': chase['chase_price'], 'ev': ev, 
                            'y_atr': chase['orig_atr'], 't_row': fast_df[today], 'y_row': y_row, 'qty': est_qty, 'proj_cost': est_cost, 
                            'chase_data': chase, 'init_sl': chase['sl'], 'init_trail': chase['sl'], 'orig_limit': chase['orig_limit']
                        })
        
        if BUY_SORT_METHOD == 'EV': candidates_today.sort(key=lambda x: (x['ev'], x['ticker']), reverse=True)
        else: candidates_today.sort(key=lambda x: (x['proj_cost'], x['ticker']), reverse=True)

        # 3. 汰弱換強邏輯 (問題3修復 - 逐一檢查 candidate 並強制綁定)
        if len(portfolio) == max_positions and enable_rotation and candidates_today:
            for i in range(len(candidates_today)):
                cand = candidates_today[i]
                weakest_ticker = None; lowest_ret = float('inf')
                
                for pt in sorted(portfolio.keys()):
                    pos = portfolio[pt]
                    d_list = ticker_dates[pt]
                    idx = bisect.bisect_left(d_list, today)
                    if idx == 0: continue
                    pt_y_row = all_dfs_fast[pt][d_list[idx - 1]]
                    
                    ret = (pt_y_row['Close'] - pos['entry']) / pos['entry']
                    _, holding_ev, _ = get_pit_stats(all_standalone_logs[pt], today, params)
                    if cand['ev'] > holding_ev and ret < lowest_ret:
                        lowest_ret = ret; weakest_ticker = pt

                if weakest_ticker:
                    w_row = all_dfs_fast[weakest_ticker][today]
                    w_y_row = all_dfs_fast[weakest_ticker][ticker_dates[weakest_ticker][bisect.bisect_left(ticker_dates[weakest_ticker], today) - 1]]
                    is_locked_down = (w_row['Open'] == w_row['High']) and (w_row['High'] == w_row['Low']) and (w_row['Low'] == w_row['Close']) and (w_row['Close'] < w_y_row['Close'])
                    
                    if not is_locked_down:
                        pos = portfolio[weakest_ticker]
                        est_sell_px = adjust_to_tick(w_row['Open'])
                        est_freed_cash = calc_net_sell_price(est_sell_px, pos['qty'], params) * pos['qty']
                        
                        # # (AI註: 修復資金不一致 Bug - 嚴守「當天賣出的資金不可當天再投入」，換股僅憑盤前既有 available_cash 判斷)
                        if cand['proj_cost'] <= available_cash:
                            exec_price = est_sell_px
                            pnl = est_freed_cash - (pos['entry'] * pos['qty'])
                            cash += est_freed_cash 
                            total_pnl = pos['realized_pnl'] + pnl
                            total_r = total_pnl / pos['initial_risk_total'] if pos['initial_risk_total'] > 0 else 0
                            closed_trades_stats.append({'pnl': total_pnl, 'r_mult': total_r})
                            if not is_training: trade_history.append({"Date": today.strftime('%Y-%m-%d'), "Ticker": weakest_ticker, "Type": "汰弱賣出(Open)", "單筆損益": pnl, "該筆總損益": total_pnl, "R_Multiple": total_r, "Risk": params.fixed_risk})
                            
                            del portfolio[weakest_ticker]
                            sold_today.add(weakest_ticker) # # (AI註: 修復 Bug 1/2 - 記錄汰弱賣出的股票)
                            
                            # 關鍵綁定: 剔除因資格不符未觸發 rotation 的前序 Candidate
                            candidates_today = candidates_today[i:]
                            break

        # 4. 嘗試買進 (嚴守 11d 資金定錨與 11b 不挪用找零)
        # # (AI註: 修復 Bug 1 - 計算盤前佔用的額度，嚴守「盤前掛單」。今日賣出所空出的額度，要等到明日迴圈才能用)
        pre_market_occupied = len(portfolio) + len(sold_today)

        for cand in candidates_today:
            if pre_market_occupied < max_positions and cand['proj_cost'] <= available_cash:
                available_cash -= cand['proj_cost'] # 圈存資金
                
                t_row = cand['t_row']
                y_row = cand['y_row']
                is_locked_up = (t_row['Open'] == t_row['High']) and (t_row['High'] == t_row['Low']) and (t_row['Low'] == t_row['Close']) and (t_row['Close'] > y_row['Close'])
                buyTriggered = False
                
                if t_row['Low'] <= cand['limit_px'] and not is_locked_up: 
                    buy_price = adjust_to_tick(min(t_row['Open'], cand['limit_px']))
                    
                    # 確保跳空暴跌沒擊穿盤前定好的停損
                    if buy_price > cand['init_sl']:
                        qty = cand['qty']
                        actual_cost_per_share = calc_entry_price(buy_price, qty, params)
                        actual_total_cost = actual_cost_per_share * qty
                        
                        # 問題2修復 - 找零不放回 available_cash
                        cash -= actual_total_cost
                        
                        net_sl_per_share = calc_net_sell_price(cand['init_sl'], qty, params) 
                        if cand['type'] == 'normal': tp_px = adjust_to_tick(buy_price + (actual_cost_per_share - net_sl_per_share))
                        else: tp_px = cand['chase_data']['tp']
                            
                        initial_risk = (actual_cost_per_share - net_sl_per_share) * qty
                        if initial_risk <= 0: initial_risk = actual_total_cost * 0.01
                        
                        portfolio[cand['ticker']] = {
                            'qty': qty, 'entry': actual_cost_per_share, 'sl': max(cand['init_sl'], cand['init_trail']), 
                            'initial_stop': cand['init_sl'], 'trailing_stop': cand['init_trail'],
                            'tp_half': tp_px, 'sold_half': False, 'pure_buy_price': buy_price,
                            'realized_pnl': 0.0, 'initial_risk_total': initial_risk
                        }
                        buyTriggered = True
                        pre_market_occupied += 1 # # (AI註: 修復 Bug 1 - 成功買入，盤前預估額度被扣一檔)
                        
                        if cand['ticker'] in pending_chases: del pending_chases[cand['ticker']]
                        if not is_training: trade_history.append({"Date": today.strftime('%Y-%m-%d'), "Ticker": cand['ticker'], "Type": f"買進 (EV:{cand['ev']:.2f}R)", "單筆損益": 0.0, "該筆總損益": 0.0, "R_Multiple": 0.0, "Risk": params.fixed_risk})

                if not buyTriggered:
                    total_missed_buys += 1
                    # 統一續追邏輯
                    if cand['type'] == 'normal':
                        chase_res = evaluate_chase_condition(t_row['Close'], cand['limit_px'], cand['y_atr'], sizing_equity, params)
                        if chase_res:
                            chase_res['orig_limit'] = cand['limit_px']
                            chase_res['orig_atr'] = cand['y_atr']
                            pending_chases[cand['ticker']] = chase_res
                    elif cand['type'] == 'chase':
                        chase_res = evaluate_chase_condition(t_row['Close'], cand['orig_limit'], cand['y_atr'], sizing_equity, params)
                        if chase_res:
                            chase_res['orig_limit'] = cand['orig_limit']
                            chase_res['orig_atr'] = cand['y_atr']
                            pending_chases[cand['ticker']] = chase_res
                        else: del pending_chases[cand['ticker']]

        today_equity = cash  
        for pt in sorted(portfolio.keys()):
            pos = portfolio[pt]
            px = all_dfs_fast[pt][today]['Close'] if today in all_dfs_fast[pt] else pos.get('last_px', pos.get('pure_buy_price', pos['entry']))
            pos['last_px'] = px # 記錄最後估值
            net_px = calc_net_sell_price(px, pos['qty'], params) 
            today_equity += pos['qty'] * net_px
            
        current_equity = today_equity 
        invested_capital = today_equity - cash
        exposure_pct = (invested_capital / today_equity) * 100 if today_equity > 0 else 0
        total_exposure += exposure_pct
        if exposure_pct > max_exp: max_exp = exposure_pct 

        strategy_ret_pct = (today_equity - initial_capital) / initial_capital * 100
        
        if benchmark_data and today in benchmark_data:
            current_bm_px = benchmark_data[today]['Close']
            if benchmark_start_price is None:
                benchmark_start_price = current_bm_px
                bm_peak_price = current_bm_px
            if benchmark_start_price and benchmark_start_price > 0: bm_ret_pct = (current_bm_px - benchmark_start_price) / benchmark_start_price * 100
            if bm_peak_price is not None:
                if current_bm_px > bm_peak_price: bm_peak_price = current_bm_px
                current_bm_drawdown = (bm_peak_price - current_bm_px) / bm_peak_price * 100
                if current_bm_drawdown > bm_max_drawdown: bm_max_drawdown = current_bm_drawdown
                    
        if today.month != current_month:
            monthly_equities.append(yesterday_equity)
            if yesterday_bm_px is not None: bm_monthly_equities.append(yesterday_bm_px)
            current_month = today.month
            
        yesterday_equity = today_equity
        yesterday_bm_px = current_bm_px
            
        if not is_training:
            equity_curve.append({
                "Date": today.strftime('%Y-%m-%d'), "Equity": today_equity, "Invested_Amount": invested_capital, 
                "Exposure_Pct": exposure_pct, "Strategy_Return_Pct": strategy_ret_pct, f"Benchmark_{benchmark_ticker}_Pct": bm_ret_pct
            })
            
        if today_equity > peak_equity: peak_equity = today_equity
        drawdown = (peak_equity - today_equity) / peak_equity * 100
        if drawdown > max_drawdown: max_drawdown = drawdown

    if len(sorted_dates) > start_idx:
        monthly_equities.append(today_equity)
        if current_bm_px is not None: bm_monthly_equities.append(current_bm_px)

    final_cash = cash
    last_date = sorted_dates[-1] if len(sorted_dates) > 0 else None
    
    # 問題6修復 - 期末強制結算不再使用買價，改用最新一日的收盤價 MTM
    for ticker in sorted(list(portfolio.keys())):
        pos = portfolio[ticker]
        exec_price = pos.get('last_px', pos.get('pure_buy_price', pos['entry']))
        net_price = calc_net_sell_price(exec_price, pos['qty'], params)
        final_cash += net_price * pos['qty']
        
        pnl = (net_price - pos['entry']) * pos['qty']
        total_pnl = pos['realized_pnl'] + pnl
        total_r = total_pnl / pos['initial_risk_total'] if pos['initial_risk_total'] > 0 else 0
        closed_trades_stats.append({'pnl': total_pnl, 'r_mult': total_r})
        
        if not is_training: trade_history.append({"Date": last_date.strftime('%Y-%m-%d') if last_date else "", "Ticker": ticker, "Type": "期末強制結算", "單筆損益": pnl, "該筆總損益": total_pnl, "R_Multiple": total_r, "Risk": params.fixed_risk})

    today_equity = final_cash
    total_return = (today_equity - initial_capital) / initial_capital * 100
    
    if not is_training and len(equity_curve) > 0:
        equity_curve[-1]['Equity'] = today_equity
        equity_curve[-1]['Strategy_Return_Pct'] = total_return
        equity_curve[-1]['Invested_Amount'] = 0.0
        equity_curve[-1]['Exposure_Pct'] = 0.0

    r_squared, monthly_win_rate = calc_curve_stats(monthly_equities)
    bm_r_squared, bm_monthly_win_rate = calc_curve_stats(bm_monthly_equities)

    trade_count = len(closed_trades_stats)
    if trade_count > 0:
        wins = [t for t in closed_trades_stats if t['pnl'] > 0]
        losses = [t for t in closed_trades_stats if t['pnl'] <= 0]
        win_rate = (len(wins) / trade_count) * 100
        avg_win_r = sum(t['r_mult'] for t in wins) / len(wins) if len(wins) > 0 else 0
        avg_loss_r = abs(sum(t['r_mult'] for t in losses) / len(losses)) if len(losses) > 0 else 0
        pf_payoff = (avg_win_r / avg_loss_r) if avg_loss_r > 0 else (99.9 if avg_win_r > 0 else 0.0)

        if EV_CALC_METHOD == 'B':
            payoff_for_ev = min(10.0, (avg_win_r / avg_loss_r)) if avg_loss_r > 0 else (99.9 if avg_win_r > 0 else 0.0)
            pf_ev = (win_rate / 100 * payoff_for_ev) - (1 - win_rate / 100)
        else: pf_ev = sum(t['r_mult'] for t in closed_trades_stats) / trade_count
    else: win_rate, pf_ev, pf_payoff = 0.0, 0.0, 0.0
        
    avg_exp = total_exposure / sim_days if sim_days > 0 else 0.0

    if is_training: return total_return, max_drawdown, trade_count, today_equity, avg_exp, max_exp, bm_ret_pct, bm_max_drawdown, win_rate, pf_ev, pf_payoff, total_missed_buys, total_missed_sells, r_squared, monthly_win_rate, bm_r_squared, bm_monthly_win_rate
    else:
        df_equity = pd.DataFrame(equity_curve)
        df_trades = pd.DataFrame(trade_history)
        final_bm_return = df_equity.iloc[-1][f"Benchmark_{benchmark_ticker}_Pct"] if not df_equity.empty else 0.0
        return df_equity, df_trades, total_return, max_drawdown, trade_count, win_rate, pf_ev, pf_payoff, today_equity, avg_exp, max_exp, final_bm_return, bm_max_drawdown, total_missed_buys, total_missed_sells, r_squared, monthly_win_rate, bm_r_squared, bm_monthly_win_rate