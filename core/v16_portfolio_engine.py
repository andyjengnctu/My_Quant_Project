import pandas as pd
import numpy as np
import math
from core.v16_core import generate_signals, adjust_to_tick, calc_net_sell_price, calc_position_size, calc_entry_price
from core.v16_config import EV_CALC_METHOD, BUY_SORT_METHOD

# (AI註: ==========================================)
# (AI註: 🌟 全域統一：投資組合系統評分公式 (單一真理來源))
# (AI註: ==========================================)
def calc_portfolio_score(sys_ret, sys_mdd, m_win_rate, r_sq):
    from core.v16_config import SCORE_CALC_METHOD
    
    # (AI註: 計算 RoMD，加上 0.0001 避免除以零的數學錯誤)
    romd = sys_ret / (abs(sys_mdd) + 0.0001)
    
    if SCORE_CALC_METHOD == 'LOG_R2':
        # (AI註: 這裡的公式一旦修改，AI 訓練與終端機面板將 100% 同步連動！)
        raw_score = romd * (m_win_rate / 100.0) * r_sq
    else:
        raw_score = romd
        
    return raw_score 

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
        if len(monthly_rets) > 0:
            monthly_win_rate = (len(monthly_rets[monthly_rets > 0]) / len(monthly_rets)) * 100
    return r_squared, monthly_win_rate

def prep_stock_data_and_trades(df, params):
    df = df.copy()
    ATR_main, buyCondition, sellCondition, buy_limits = generate_signals(df, params)
    df['ATR'] = ATR_main
    df['is_setup'] = buyCondition
    df['ind_sell_signal'] = sellCondition
    df['buy_limit'] = buy_limits

    O, H, L, C = df['Open'].values, df['High'].values, df['Low'].values, df['Close'].values
    trade_logs = []
    in_position = False
    entry_price, sl_price, tp_half = 0.0, 0.0, 0.0
    buy_price = 0.0 # (AI註: 修復 1: 新增記錄原始買進觸發價，以對齊 core 單股回測的追蹤停損邏輯)
    qty = 0
    realized_pnl, initial_risk = 0.0, 0.0
    sold_half = False
    dates = df.index.values
    
    # (AI註: 修復 5: 將 dummy_cap 對齊設定檔的初始本金，避免 min_fee 導致 R 倍數在不同資金規模下失真)
    dummy_cap = params.initial_capital  
    
    for i in range(1, len(df)):
        if np.isnan(ATR_main[i-1]): continue
        if in_position:
            is_locked_down = (O[i] == H[i]) and (H[i] == L[i]) and (L[i] == C[i]) and (C[i] < C[i-1])
            is_stop = sellCondition[i-1] or L[i] <= sl_price
            
            if is_stop:
                if not is_locked_down:
                    exec_px = adjust_to_tick(O[i] if sellCondition[i-1] else min(O[i], sl_price))
                    net_px = calc_net_sell_price(exec_px, qty, params)
                    realized_pnl += (net_px - entry_price) * qty
                    total_r = realized_pnl / initial_risk if initial_risk > 0 else 0
                    trade_logs.append({'exit_date': pd.to_datetime(dates[i]), 'pnl': realized_pnl, 'r_mult': total_r})
                    in_position = False
            
            if in_position:
                if not sold_half and H[i] >= tp_half:
                    sell_qty = int(np.floor(qty * params.tp_percent))
                    if sell_qty > 0:
                        exec_px = adjust_to_tick(max(O[i], tp_half))
                        net_px = calc_net_sell_price(exec_px, sell_qty, params)
                        realized_pnl += (net_px - entry_price) * sell_qty
                        qty -= sell_qty
                        sold_half = True
                        
                # (AI註: 修復 1: 追蹤停損邏輯完全對齊 v16_core.py，只有當收盤價大於「買進價+ATR*倍數」時才上移)
                if C[i] > buy_price + (ATR_main[i] * params.atr_times_trail):
                    new_sl = adjust_to_tick(C[i] - (ATR_main[i] * params.atr_times_trail))
                    if new_sl > sl_price: sl_price = new_sl
                
        else:
            is_locked_up = (O[i] == H[i]) and (H[i] == L[i]) and (L[i] == C[i]) and (C[i] > C[i-1])
            if buyCondition[i-1] and L[i] <= buy_limits[i-1] and not is_locked_up:
                exec_px = adjust_to_tick(min(O[i], buy_limits[i-1]))
                buy_price = exec_px # (AI註: 記錄原始買進觸發價)
                init_sl = adjust_to_tick(exec_px - (ATR_main[i-1] * params.atr_times_init))
                qty = calc_position_size(exec_px, init_sl, dummy_cap, params.fixed_risk, params)
                if qty > 0:
                    entry_price = calc_entry_price(exec_px, qty, params)
                    sl_price = init_sl
                    net_sl = calc_net_sell_price(sl_price, qty, params)
                    tp_half = adjust_to_tick(exec_px + (entry_price - net_sl))
                    initial_risk = (entry_price - net_sl) * qty
                    if initial_risk <= 0: initial_risk = dummy_cap * 0.01
                    realized_pnl = 0.0
                    sold_half = False
                    in_position = True
    return df, trade_logs

def get_pit_stats(trade_logs, current_date, params):
    past_trades = [t for t in trade_logs if t['exit_date'] < current_date]
    trade_count = len(past_trades)
    if trade_count < getattr(params, 'min_history_trades', 1): return False, 0.0, 0.0 
    
    wins = [t for t in past_trades if t['pnl'] > 0]
    win_rate = len(wins) / trade_count
    
    if EV_CALC_METHOD == 'B':
        avg_win_pnl = sum(t['pnl'] for t in wins) / len(wins) if len(wins) > 0 else 0
        loss_count = trade_count - len(wins)
        avg_loss_pnl = abs(sum(t['pnl'] for t in past_trades if t['pnl'] <= 0) / loss_count) if loss_count > 0 else 0
        payoff_for_ev = min(10.0, (avg_win_pnl / avg_loss_pnl)) if avg_loss_pnl > 0 else (99.9 if avg_win_pnl > 0 else 0.0)
        ev_to_sort = (win_rate * payoff_for_ev) - (1 - win_rate)
    else:
        ev_to_sort = sum(t['r_mult'] for t in past_trades) / trade_count

    is_candidate = (ev_to_sort > getattr(params, 'min_history_ev', 0.0)) and (win_rate >= getattr(params, 'min_history_win_rate', 0.30))
    return is_candidate, ev_to_sort, win_rate

def run_portfolio_timeline(all_dfs_fast, all_trade_logs, sorted_dates, start_year, params, max_positions, enable_rotation, benchmark_ticker="0050", benchmark_data=None, is_training=True):
    start_dt = pd.to_datetime(f"{start_year}-01-01")
    # (AI註: 修復 4: 找不到對應日期時，回傳 len(sorted_dates) 直接終止迴圈，而非從索引 1 錯誤開跑)
    start_idx = next((i for i, d in enumerate(sorted_dates) if d >= start_dt), len(sorted_dates))

    initial_capital = params.initial_capital
    cash = initial_capital
    portfolio = {} 
    trade_history, equity_curve, closed_trades_stats = [], [], []
    peak_equity = initial_capital
    max_drawdown = 0.0
    current_equity = initial_capital 
    total_exposure = 0.0
    sim_days = 0
    total_missed_buys = 0  
    total_missed_sells = 0 
    
    monthly_equities = []
    bm_monthly_equities = []
    current_month = sorted_dates[start_idx].month if len(sorted_dates) > start_idx else 1
    yesterday_equity = initial_capital
    current_bm_px = benchmark_data[sorted_dates[start_idx]]['Close'] if benchmark_data and sorted_dates[start_idx] in benchmark_data else None
    yesterday_bm_px = current_bm_px
    
    benchmark_start_price = None
    bm_peak_price = None     
    bm_max_drawdown = 0.0    
    bm_ret_pct = 0.0

    for i in range(start_idx, len(sorted_dates)):
        sim_days += 1
        today = sorted_dates[i]
        yesterday = sorted_dates[i-1]
        held_yesterday = set(portfolio.keys())
        
        available_cash = cash
        
        if getattr(params, 'use_compounding', True):
            sizing_equity = current_equity
        else:
            sizing_equity = initial_capital
        
        if not is_training and i % 20 == 0:
            exp = ((current_equity - cash) / current_equity) * 100 if current_equity > 0 else 0
            print(f"\033[90m⏳ 推進中: {today.strftime('%Y-%m')} | 資產: {current_equity:,.0f} | 水位: {exp:>5.1f}%...\033[0m", end="\r", flush=True)

        candidates_today = []
        tickers_to_remove = []

        for ticker, pos in portfolio.items():
            fast_df = all_dfs_fast[ticker]
            if today not in fast_df: continue 
            row, y_row = fast_df[today], fast_df[yesterday] if yesterday in fast_df else fast_df[today]
            is_ind_sell = y_row['ind_sell_signal']
            is_locked_down = (row['Open'] == row['High']) and (row['High'] == row['Low']) and (row['Low'] == row['Close']) and (row['Close'] < y_row['Close'])
            
            if is_ind_sell or row['Low'] <= pos['sl']:
                if not is_locked_down:
                    exec_price = adjust_to_tick(row['Open'] if is_ind_sell else min(row['Open'], pos['sl']))
                    net_price = calc_net_sell_price(exec_price, pos['qty'], params)
                    pnl = (net_price - pos['entry']) * pos['qty']
                    cash += (net_price * pos['qty'])
                    
                    available_cash += (net_price * pos['qty'])
                    
                    total_pnl = pos['realized_pnl'] + pnl
                    total_r = total_pnl / pos['initial_risk_total'] if pos['initial_risk_total'] > 0 else 0
                    closed_trades_stats.append({'pnl': total_pnl, 'r_mult': total_r})
                    
                    if not is_training: 
                        t_type = "全倉結算(指標)" if is_ind_sell else "全倉結算(停損)"
                        trade_history.append({"Date": today.strftime('%Y-%m-%d'), "Ticker": ticker, "Type": t_type, "Price": exec_price, "單筆損益": pnl, "該筆總損益": total_pnl, "R_Multiple": total_r, "Risk": params.fixed_risk})
                    
                    tickers_to_remove.append(ticker)
                    continue
                else:
                    total_missed_sells += 1 
                
            if not pos['sold_half'] and row['High'] >= pos['tp_half']:
                sell_qty = int(np.floor(pos['qty'] * params.tp_percent))
                if sell_qty > 0:
                    exec_price = adjust_to_tick(max(row['Open'], pos['tp_half']))
                    net_price = calc_net_sell_price(exec_price, sell_qty, params)
                    pnl = (net_price - pos['entry']) * sell_qty
                    cash += (net_price * sell_qty)
                    
                    available_cash += (net_price * sell_qty)
                    
                    pos['realized_pnl'] += pnl
                    pos['qty'] -= sell_qty
                    pos['sold_half'] = True
                    if not is_training: 
                        trade_history.append({"Date": today.strftime('%Y-%m-%d'), "Ticker": ticker, "Type": "半倉停利", "Price": exec_price, "單筆損益": pnl, "該筆總損益": pos['realized_pnl'], "R_Multiple": 0.0, "Risk": params.fixed_risk})
            
            new_sl = adjust_to_tick(row['Close'] - (row['ATR'] * params.atr_times_trail))
            if new_sl > pos['sl']: pos['sl'] = new_sl

        for t in tickers_to_remove: del portfolio[t]

        for ticker, fast_df in all_dfs_fast.items():
            if ticker in portfolio or ticker in held_yesterday: continue 
            if today not in fast_df or yesterday not in fast_df: continue
            y_row, t_row = fast_df[yesterday], fast_df[today]
            
            if y_row['is_setup']:
                is_locked_up = (t_row['Open'] == t_row['High']) and (t_row['High'] == t_row['Low']) and (t_row['Low'] == t_row['Close']) and (t_row['Close'] > y_row['Close'])
                if t_row['Low'] <= y_row['buy_limit'] and not is_locked_up:
                    is_candidate, ev, win_rate = get_pit_stats(all_trade_logs[ticker], yesterday, params)
                    if is_candidate:
                        candidates_today.append({
                            'ticker': ticker, 'buy_limit': y_row['buy_limit'], 'ev': ev, 
                            'y_atr': y_row['ATR'], 't_row': t_row
                        })
                else:
                    total_missed_buys += 1
        
        if candidates_today:
            for c in candidates_today:
                c['exec_px'] = adjust_to_tick(min(c['t_row']['Open'], c['buy_limit']))
                c['sl_price'] = adjust_to_tick(c['exec_px'] - (c['y_atr'] * params.atr_times_init))
                c['qty'] = calc_position_size(c['exec_px'], c['sl_price'], sizing_equity, params.fixed_risk, params)
                c['proj_cost'] = calc_entry_price(c['exec_px'], c['qty'], params) * c['qty'] if c['qty'] > 0 else 0.0
            
            if BUY_SORT_METHOD == 'EV': candidates_today.sort(key=lambda x: x['ev'], reverse=True)
            else: candidates_today.sort(key=lambda x: x['proj_cost'], reverse=True)

            if len(portfolio) == max_positions and enable_rotation:
                best_cand = candidates_today[0]
                weakest_ticker, lowest_ret, weakest_ev = None, 0.0, 0.0
                
                for pt, pos in portfolio.items():
                    if yesterday in all_dfs_fast[pt]:
                        ret = (all_dfs_fast[pt][yesterday]['Close'] - pos['entry']) / pos['entry']
                        if ret < lowest_ret:
                            _, holding_ev, _ = get_pit_stats(all_trade_logs[pt], yesterday, params)
                            if holding_ev < best_cand['ev']:
                                lowest_ret = ret
                                weakest_ticker = pt
                                weakest_ev = holding_ev
                
                if weakest_ticker:
                    w_row = all_dfs_fast[weakest_ticker][today]
                    w_y_row = all_dfs_fast[weakest_ticker][yesterday] if yesterday in all_dfs_fast[weakest_ticker] else w_row
                    is_locked_down = (w_row['Open'] == w_row['High']) and (w_row['High'] == w_row['Low']) and (w_row['Low'] == w_row['Close']) and (w_row['Close'] < w_y_row['Close'])
                    
                    if is_locked_down: 
                        total_missed_sells += 1
                    else:
                        pos = portfolio[weakest_ticker]
                        est_sell_px = adjust_to_tick(w_row['Open'])
                        est_cash_freed = calc_net_sell_price(est_sell_px, pos['qty'], params) * pos['qty']
                        
                        test_cash = available_cash + est_cash_freed
                        cand_qty = best_cand['qty']
                        cand_cost = best_cand['proj_cost']
                        
                        if cand_qty > 0 and cand_cost <= test_cash:
                            exec_price = est_sell_px
                            pnl = est_cash_freed - (pos['entry'] * pos['qty'])
                            cash += est_cash_freed
                            available_cash += est_cash_freed
                            
                            total_pnl = pos['realized_pnl'] + pnl
                            total_r = total_pnl / pos['initial_risk_total'] if pos['initial_risk_total'] > 0 else 0
                            closed_trades_stats.append({'pnl': total_pnl, 'r_mult': total_r})
                            
                            if not is_training: trade_history.append({"Date": today.strftime('%Y-%m-%d'), "Ticker": weakest_ticker, "Type": "汰弱賣出(Open)", "Price": exec_price, "單筆損益": pnl, "該筆總損益": total_pnl, "R_Multiple": total_r, "Risk": params.fixed_risk})
                            del portfolio[weakest_ticker]

            for cand in candidates_today:
                if len(portfolio) < max_positions:
                    buy_price = cand['exec_px']
                    sl_price = cand['sl_price']
                    qty = cand['qty'] 
                    
                    if qty > 0:
                        entry_cost_per_share = calc_entry_price(buy_price, qty, params)
                        cost_total = entry_cost_per_share * qty
                        
                        if cost_total <= available_cash:
                            available_cash -= cost_total
                            cash -= cost_total
                            net_sl_per_share = calc_net_sell_price(sl_price, qty, params)
                            tp_target = adjust_to_tick(buy_price + (entry_cost_per_share - net_sl_per_share))
                            initial_risk = (entry_cost_per_share - net_sl_per_share) * qty
                            if initial_risk <= 0: initial_risk = cost_total * 0.01
                            
                            portfolio[cand['ticker']] = {
                                'qty': qty, 'entry': entry_cost_per_share, 'sl': sl_price, 
                                'initial_sl_net': net_sl_per_share, 'tp_half': tp_target, 
                                'sold_half': False, 'last_px': buy_price,
                                'realized_pnl': 0.0, 'initial_risk_total': initial_risk
                            }
                            if not is_training: trade_history.append({"Date": today.strftime('%Y-%m-%d'), "Ticker": cand['ticker'], "Type": f"買進 (EV:{cand['ev']:.2f}R)", "Price": buy_price, "單筆損益": 0.0, "該筆總損益": 0.0, "R_Multiple": 0.0, "Risk": params.fixed_risk})

        today_equity = cash  
        for pt, pos in portfolio.items():
            px = all_dfs_fast[pt][today]['Close'] if today in all_dfs_fast[pt] else pos.get('last_px', pos['entry'])
            pos['last_px'] = px
            net_px = calc_net_sell_price(px, pos['qty'], params) 
            today_equity += pos['qty'] * net_px
            
        current_equity = today_equity 
        invested_capital = today_equity - cash
        exposure_pct = (invested_capital / today_equity) * 100 if today_equity > 0 else 0
        total_exposure += exposure_pct

        strategy_ret_pct = (today_equity - initial_capital) / initial_capital * 100
        
        if benchmark_data and today in benchmark_data:
            current_bm_px = benchmark_data[today]['Close']
            if benchmark_start_price is None:
                benchmark_start_price = current_bm_px
                bm_peak_price = current_bm_px
            if benchmark_start_price and benchmark_start_price > 0:
                bm_ret_pct = (current_bm_px - benchmark_start_price) / benchmark_start_price * 100
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
        if current_bm_px is not None:
            bm_monthly_equities.append(current_bm_px)

    final_cash = cash
    last_date = sorted_dates[-1] if len(sorted_dates) > 0 else None
    for ticker, pos in list(portfolio.items()):
        exec_price = pos.get('last_px', pos['entry'])
        net_price = calc_net_sell_price(exec_price, pos['qty'], params)
        final_cash += net_price * pos['qty']
        
        pnl = (net_price - pos['entry']) * pos['qty']
        total_pnl = pos['realized_pnl'] + pnl
        total_r = total_pnl / pos['initial_risk_total'] if pos['initial_risk_total'] > 0 else 0
        closed_trades_stats.append({'pnl': total_pnl, 'r_mult': total_r})
        
        if not is_training:
            trade_history.append({"Date": last_date.strftime('%Y-%m-%d') if last_date else "", "Ticker": ticker, "Type": "期末強制結算", "Price": exec_price, "單筆損益": pnl, "該筆總損益": total_pnl, "R_Multiple": total_r, "Risk": params.fixed_risk})

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
        pf_payoff = (avg_win_r / avg_loss_r) if avg_loss_r > 0 else 0.0

        if EV_CALC_METHOD == 'B':
            payoff_for_ev = min(10.0, (avg_win_r / avg_loss_r)) if avg_loss_r > 0 else (99.9 if avg_win_r > 0 else 0.0)
            pf_ev = (win_rate / 100 * payoff_for_ev) - (1 - win_rate / 100)
        else:
            pf_ev = sum(t['r_mult'] for t in closed_trades_stats) / trade_count
    else:
        win_rate, pf_ev, pf_payoff = 0.0, 0.0, 0.0
        
    avg_exp = total_exposure / sim_days if sim_days > 0 else 0.0

    if is_training:
        return total_return, max_drawdown, trade_count, today_equity, avg_exp, bm_ret_pct, bm_max_drawdown, win_rate, pf_ev, pf_payoff, total_missed_buys, total_missed_sells, r_squared, monthly_win_rate, bm_r_squared, bm_monthly_win_rate
    else:
        df_equity = pd.DataFrame(equity_curve)
        df_trades = pd.DataFrame(trade_history)
        final_bm_return = df_equity.iloc[-1][f"Benchmark_{benchmark_ticker}_Pct"] if not df_equity.empty else 0.0
        return df_equity, df_trades, total_return, max_drawdown, trade_count, win_rate, pf_ev, pf_payoff, today_equity, final_bm_return, bm_max_drawdown, total_missed_buys, total_missed_sells, r_squared, monthly_win_rate, bm_r_squared, bm_monthly_win_rate