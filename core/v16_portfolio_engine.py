import pandas as pd
import numpy as np
from core.v16_core import generate_signals, adjust_to_tick, calc_net_sell_price, calc_position_size, calc_entry_price
from core.v16_config import EV_CALC_METHOD, BUY_SORT_METHOD

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
    entry_price, sl_price, initial_sl_price = 0.0, 0.0, 0.0
    dates = df.index.values
    
    for i in range(1, len(df)):
        if np.isnan(ATR_main[i-1]): continue
        if in_position:
            is_locked_limit_down = (O[i] == H[i]) and (H[i] == L[i]) and (L[i] == C[i]) and (C[i] < C[i-1])
            
            if sellCondition[i-1] or L[i] <= sl_price:
                if not is_locked_limit_down:
                    exit_price = O[i] if sellCondition[i-1] else min(O[i], sl_price)
                    initial_risk = entry_price - initial_sl_price
                    if initial_risk <= 0: initial_risk = entry_price * 0.01
                    r_multiple = (exit_price - entry_price) / initial_risk
                    
                    trade_logs.append({'exit_date': pd.to_datetime(dates[i]), 'pnl': (exit_price - entry_price) / entry_price, 'r_mult': r_multiple})
                    in_position = False
            else:
                new_sl = C[i] - (ATR_main[i] * params.atr_times_trail)
                if new_sl > sl_price: sl_price = new_sl
        else:
            is_locked_limit_up = (O[i] == H[i]) and (H[i] == L[i]) and (L[i] == C[i]) and (C[i] > C[i-1])
            
            if buyCondition[i-1] and L[i] <= buy_limits[i-1] and not is_locked_limit_up:
                entry_price = min(O[i], buy_limits[i-1]) 
                initial_sl_price = entry_price - (ATR_main[i-1] * params.atr_times_init)
                sl_price = initial_sl_price
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
    start_idx = next((i for i, d in enumerate(sorted_dates) if d >= start_dt), 1)

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
    
    benchmark_start_price = None
    bm_peak_price = None     
    bm_max_drawdown = 0.0    
    bm_ret_pct = 0.0

    for i in range(start_idx, len(sorted_dates)):
        sim_days += 1
        today = sorted_dates[i]
        yesterday = sorted_dates[i-1]
        held_yesterday = set(portfolio.keys())
        overnight_cash = cash
        
        sizing_equity = cash + sum([pos['qty'] * pos.get('last_px', pos['entry']) for pos in portfolio.values()])
        
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
            
            is_locked_limit_down = (row['Open'] == row['High']) and (row['High'] == row['Low']) and (row['Low'] == row['Close']) and (row['Close'] < y_row['Close'])
            
            if is_ind_sell or row['Low'] <= pos['sl']:
                if not is_locked_limit_down:
                    exec_price = adjust_to_tick(row['Open'] if is_ind_sell else min(row['Open'], pos['sl']))
                    net_price = calc_net_sell_price(exec_price, pos['qty'], params)
                    pnl = (net_price - pos['entry']) * pos['qty']
                    cash += (net_price * pos['qty'])
                    
                    risk_per_share = pos['entry'] - pos['initial_sl_net']
                    if risk_per_share <= 0: risk_per_share = pos['entry'] * 0.01
                    r_mult = (net_price - pos['entry']) / risk_per_share
                    
                    if is_training: closed_trades_stats.append({'pnl': pnl, 'r_mult': r_mult})
                    else: 
                        t_type = "指標賣訊" if is_ind_sell else "停損出場"
                        trade_history.append({"Date": today.strftime('%Y-%m-%d'), "Ticker": ticker, "Type": t_type, "Price": exec_price, "PnL": pnl, "R_Multiple": r_mult, "Risk": params.fixed_risk})
                    
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
                    
                    risk_per_share = pos['entry'] - pos['initial_sl_net']
                    if risk_per_share <= 0: risk_per_share = pos['entry'] * 0.01
                    r_mult = (net_price - pos['entry']) / risk_per_share
                    
                    pos['qty'] -= sell_qty
                    pos['sold_half'] = True
                    if is_training: closed_trades_stats.append({'pnl': pnl, 'r_mult': r_mult})
                    else: trade_history.append({"Date": today.strftime('%Y-%m-%d'), "Ticker": ticker, "Type": "半倉停利", "Price": exec_price, "PnL": pnl, "R_Multiple": r_mult, "Risk": params.fixed_risk})
            
            new_sl = adjust_to_tick(row['Close'] - (row['ATR'] * params.atr_times_trail))
            if new_sl > pos['sl']: pos['sl'] = new_sl

        for t in tickers_to_remove: del portfolio[t]

        for ticker, fast_df in all_dfs_fast.items():
            if ticker in portfolio or ticker in held_yesterday: continue 
            if today not in fast_df or yesterday not in fast_df: continue
            
            y_row, t_row = fast_df[yesterday], fast_df[today]
            
            is_locked_limit_up = (t_row['Open'] == t_row['High']) and (t_row['High'] == t_row['Low']) and (t_row['Low'] == t_row['Close']) and (t_row['Close'] > y_row['Close'])
            
            if y_row['is_setup']:
                if t_row['Low'] <= y_row['buy_limit'] and not is_locked_limit_up:
                    is_candidate, ev, win_rate = get_pit_stats(all_trade_logs[ticker], yesterday, params)
                    if is_candidate:
                        buy_price = adjust_to_tick(min(t_row['Open'], y_row['buy_limit']))
                        sl_price = adjust_to_tick(buy_price - (y_row['ATR'] * params.atr_times_init))
                        
                        proj_qty = calc_position_size(buy_price, sl_price, sizing_equity, params.fixed_risk, params)
                        proj_cost = calc_entry_price(buy_price, proj_qty, params) * proj_qty if proj_qty > 0 else 0.0
                        
                        candidates_today.append({'ticker': ticker, 'buy_price': buy_price, 'sl_price': sl_price, 'ev': ev, 'proj_cost': proj_cost})
                else:
                    total_missed_buys += 1
                    
        if candidates_today:
            # 🌟 向 config 拿取全域買入排序方法
            if BUY_SORT_METHOD == 'EV':
                candidates_today.sort(key=lambda x: x['ev'], reverse=True)
            else:
                candidates_today.sort(key=lambda x: x['proj_cost'], reverse=True)
                
            for cand in candidates_today:
                if len(portfolio) < max_positions:
                    qty = calc_position_size(cand['buy_price'], cand['sl_price'], sizing_equity, params.fixed_risk, params)
                    if qty > 0:
                        entry_cost_per_share = calc_entry_price(cand['buy_price'], qty, params)
                        cost_total = entry_cost_per_share * qty
                        if cost_total <= overnight_cash:
                            overnight_cash -= cost_total
                            cash -= cost_total
                            net_sl_per_share = calc_net_sell_price(cand['sl_price'], qty, params)
                            tp_target = adjust_to_tick(cand['buy_price'] + (entry_cost_per_share - net_sl_per_share))
                            portfolio[cand['ticker']] = {'qty': qty, 'entry': entry_cost_per_share, 'sl': cand['sl_price'], 'initial_sl_net': net_sl_per_share, 'tp_half': tp_target, 'sold_half': False, 'last_px': cand['buy_price']}
                            if not is_training: trade_history.append({"Date": today.strftime('%Y-%m-%d'), "Ticker": cand['ticker'], "Type": f"買進 (EV:{cand['ev']:.2f}R)", "Price": cand['buy_price'], "PnL": 0, "R_Multiple": 0.0, "Risk": params.fixed_risk})
                
                elif len(portfolio) == max_positions and enable_rotation:
                    weakest_ticker, lowest_return = None, 0.0 
                    for pt, pos in portfolio.items():
                        if today not in all_dfs_fast[pt]: continue 
                        ret = (all_dfs_fast[pt][today]['Close'] - pos['entry']) / pos['entry']
                        if ret < lowest_return:
                            _, holding_ev, _ = get_pit_stats(all_trade_logs[pt], yesterday, params)
                            if holding_ev < cand['ev']:
                                lowest_return = ret
                                weakest_ticker = pt
                                
                    if weakest_ticker: 
                        w_row = all_dfs_fast[weakest_ticker][today]
                        w_y_row = all_dfs_fast[weakest_ticker][yesterday] if yesterday in all_dfs_fast[weakest_ticker] else w_row
                        w_locked_down = (w_row['Open'] == w_row['High']) and (w_row['High'] == w_row['Low']) and (w_row['Low'] == w_row['Close']) and (w_row['Close'] < w_y_row['Close'])
                        
                        if w_locked_down:
                            total_missed_sells += 1
                        else:
                            pos = portfolio[weakest_ticker]
                            sell_price = adjust_to_tick(all_dfs_fast[weakest_ticker][today]['Close'])
                            net_price = calc_net_sell_price(sell_price, pos['qty'], params)
                            pnl = (net_price - pos['entry']) * pos['qty']
                            cash += (net_price * pos['qty'])
                            overnight_cash += (net_price * pos['qty'])
                            
                            risk_per_share = pos['entry'] - pos['initial_sl_net']
                            if risk_per_share <= 0: risk_per_share = pos['entry'] * 0.01
                            r_mult = (net_price - pos['entry']) / risk_per_share
                            
                            if is_training: closed_trades_stats.append({'pnl': pnl, 'r_mult': r_mult})
                            else: trade_history.append({"Date": today.strftime('%Y-%m-%d'), "Ticker": weakest_ticker, "Type": "汰弱賣出", "Price": sell_price, "PnL": pnl, "R_Multiple": r_mult, "Risk": params.fixed_risk})
                            
                            del portfolio[weakest_ticker]
                            
                            sizing_equity = cash + sum([p['qty'] * p.get('last_px', p['entry']) for p in portfolio.values()])
                            qty = calc_position_size(cand['buy_price'], cand['sl_price'], sizing_equity, params.fixed_risk, params)
                            if qty > 0:
                                entry_cost_per_share = calc_entry_price(cand['buy_price'], qty, params)
                                cost_total = entry_cost_per_share * qty
                                if cost_total <= overnight_cash:
                                    overnight_cash -= cost_total
                                    cash -= cost_total
                                    net_sl_per_share = calc_net_sell_price(cand['sl_price'], qty, params)
                                    tp_target = adjust_to_tick(cand['buy_price'] + (entry_cost_per_share - net_sl_per_share))
                                    portfolio[cand['ticker']] = {'qty': qty, 'entry': entry_cost_per_share, 'sl': cand['sl_price'], 'initial_sl_net': net_sl_per_share, 'tp_half': tp_target, 'sold_half': False, 'last_px': cand['buy_price']}
                                    if not is_training: trade_history.append({"Date": today.strftime('%Y-%m-%d'), "Ticker": cand['ticker'], "Type": f"汰弱換入 (EV:{cand['ev']:.2f}R)", "Price": cand['buy_price'], "PnL": 0, "R_Multiple": 0.0, "Risk": params.fixed_risk})

        today_equity = cash  
        for pt, pos in portfolio.items():
            if today in all_dfs_fast[pt]:
                px = all_dfs_fast[pt][today]['Close']
                pos['last_px'] = px 
            else:
                px = pos.get('last_px', pos['entry']) 
            today_equity += pos['qty'] * px
            
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
                if current_bm_px > bm_peak_price:
                    bm_peak_price = current_bm_px
                current_bm_drawdown = (bm_peak_price - current_bm_px) / bm_peak_price * 100
                if current_bm_drawdown > bm_max_drawdown:
                    bm_max_drawdown = current_bm_drawdown
            
        if not is_training:
            equity_curve.append({
                "Date": today.strftime('%Y-%m-%d'), 
                "Equity": today_equity,
                "Invested_Amount": invested_capital, 
                "Exposure_Pct": exposure_pct,
                "Strategy_Return_Pct": strategy_ret_pct,
                f"Benchmark_{benchmark_ticker}_Pct": bm_ret_pct
            })
            
        if today_equity > peak_equity: peak_equity = today_equity
        drawdown = (peak_equity - today_equity) / peak_equity * 100
        if drawdown > max_drawdown: max_drawdown = drawdown

    total_return = (today_equity - initial_capital) / initial_capital * 100

    if is_training:
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
        return total_return, max_drawdown, trade_count, today_equity, avg_exp, bm_ret_pct, bm_max_drawdown, win_rate, pf_ev, pf_payoff, total_missed_buys, total_missed_sells
    else:
        df_equity = pd.DataFrame(equity_curve)
        df_trades = pd.DataFrame(trade_history)
        
        closed_trades = df_trades[~df_trades['Type'].str.contains('買進|換入', regex=True, na=False)] if not df_trades.empty else pd.DataFrame()
        
        if len(closed_trades) > 0:
            win_rate = len(closed_trades[closed_trades['PnL'] > 0]) / len(closed_trades) * 100
            wins = closed_trades[closed_trades['R_Multiple'] > 0]
            losses = closed_trades[closed_trades['R_Multiple'] <= 0]
            
            avg_win_r = wins['R_Multiple'].mean() if len(wins) > 0 else 0
            avg_loss_r = abs(losses['R_Multiple'].mean()) if len(losses) > 0 else 0
            pf_payoff = avg_win_r / avg_loss_r if avg_loss_r > 0 else 0.0

            if EV_CALC_METHOD == 'B':
                payoff_for_ev = min(10.0, (avg_win_r / avg_loss_r)) if avg_loss_r > 0 else (99.9 if avg_win_r > 0 else 0.0)
                pf_ev = (win_rate / 100 * payoff_for_ev) - (1 - win_rate / 100)
            else:
                pf_ev = closed_trades['R_Multiple'].mean() 
        else:
            win_rate, pf_ev, pf_payoff = 0.0, 0.0, 0.0
            
        final_bm_return = df_equity.iloc[-1][f"Benchmark_{benchmark_ticker}_Pct"] if not df_equity.empty else 0.0
        return df_equity, df_trades, total_return, max_drawdown, win_rate, pf_ev, pf_payoff, today_equity, final_bm_return, bm_max_drawdown, total_missed_buys, total_missed_sells