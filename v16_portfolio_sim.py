import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

from v16_config import V16StrategyParams
from v16_core import (
    tv_atr, tv_ema, tv_supertrend, adjust_to_tick,
    calc_entry_price, calc_net_sell_price, calc_position_size
)

warnings.filterwarnings('ignore')

C_RED = '\033[91m'
C_YELLOW = '\033[93m'
C_CYAN = '\033[96m'
C_GREEN = '\033[92m'
C_GRAY = '\033[90m'
C_RESET = '\033[0m'

def load_dynamic_params(json_file):
    params = V16StrategyParams()
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            for key, value in data.items():
                if hasattr(params, key):
                    setattr(params, key, value)
            return params, True
        except Exception as e:
            pass
    return params, False

def prep_stock_data_and_trades(df, params):
    df = df.copy()
    H, L, C = df['High'].values, df['Low'].values, df['Close'].values
    O, V = df['Open'].values, df['Volume'].values

    ATR_main = tv_atr(H, L, C, params.atr_len)
    ATR_kc = tv_atr(H, L, C, params.kc_len)
    HighN = df['High'].shift(1).rolling(params.high_len, min_periods=1).max().values
    
    KC_Mid = tv_ema(C, params.kc_len)
    KC_Lower = KC_Mid - ATR_kc * params.kc_mult
    
    SuperTrend_Dir = tv_supertrend(H, L, C, ATR_main, params.atr_times_trail)
    isSupertrend_Bearish_Flip = (SuperTrend_Dir == 1) & (np.roll(SuperTrend_Dir, 1) == -1)
    isSupertrend_Bearish_Flip[0] = False
    
    BB_Mid = df['Close'].rolling(params.bb_len).mean().values
    BB_Upper = BB_Mid + params.bb_mult * df['Close'].rolling(params.bb_len).std(ddof=0).values
    VolS = df['Volume'].rolling(params.vol_short_len).mean().values
    VolL = df['Volume'].rolling(params.vol_long_len).mean().values
    
    isPriceCrossover = (C > HighN) & (np.roll(C, 1) <= np.roll(HighN, 1))
    isPriceCrossover[0] = False
    isKcCrossunder = (C < KC_Lower) & (np.roll(C, 1) >= np.roll(KC_Lower, 1))
    isKcCrossunder[0] = False
    
    volCondition = np.isnan(V) | (VolS > VolL)
    buyCondition = (C > O) & (C > BB_Upper) & isPriceCrossover & volCondition
    sellCondition = (isSupertrend_Bearish_Flip & (C <= O)) | (isKcCrossunder & (C < O))

    df['ATR'] = ATR_main
    df['is_setup'] = buyCondition
    df['ind_sell_signal'] = sellCondition
    
    buy_limits = np.full_like(C, np.nan)
    for i in range(len(C)):
        if buyCondition[i]:
            buy_limits[i] = adjust_to_tick(C[i] + ATR_main[i] * params.atr_buy_tol)
    df['buy_limit'] = buy_limits

    trade_logs = []
    in_position = False
    entry_price, sl_price = 0.0, 0.0
    dates = df.index.values
    
    for i in range(1, len(df)):
        if np.isnan(ATR_main[i-1]) or np.isnan(BB_Upper[i-1]): continue
        if in_position:
            if sellCondition[i-1] or L[i] <= sl_price:
                exit_price = O[i] if sellCondition[i-1] else min(O[i], sl_price)
                trade_logs.append({'exit_date': pd.to_datetime(dates[i]), 'pnl': (exit_price - entry_price) / entry_price})
                in_position = False
            else:
                new_sl = C[i] - (ATR_main[i] * params.atr_times_trail)
                if new_sl > sl_price: sl_price = new_sl
        else:
            if buyCondition[i-1] and L[i] <= buy_limits[i-1]:
                entry_price = min(O[i], buy_limits[i-1])
                sl_price = entry_price - (ATR_main[i-1] * params.atr_times_init)
                in_position = True
    return df, trade_logs

def get_pit_stats(trade_logs, current_date):
    past_trades = [t for t in trade_logs if t['exit_date'] < current_date]
    trade_count = len(past_trades)
    if trade_count < 5: return False, 0.0, 0.0 #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<設定歷史績效過濾條件 <<<<<<<<<<<<<<<<
    wins = [t for t in past_trades if t['pnl'] > 0]
    losses = [t for t in past_trades if t['pnl'] <= 0]
    win_rate = len(wins) / trade_count
    avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0.0
    avg_loss = abs(sum(t['pnl'] for t in losses) / len(losses)) if losses else 0.0
    payoff = avg_win / avg_loss if avg_loss > 0 else (5.0 if avg_win > 0 else 0)
    ev = (win_rate * payoff) - (1 - win_rate)
    return (win_rate >= 0.35) and (ev > 0.2), ev, win_rate #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<設定歷史績效過濾條件 <<<<<<<<<<<<<<<<

def run_portfolio_simulation(data_dir, params, max_positions=5, enable_rotation=False):
    print(f"{C_CYAN}📦 正在預載入歷史軌跡，構建真實時間軸...{C_RESET}")
    all_dfs = {}
    all_trade_logs = {}
    master_dates = set()
    
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    for count, file in enumerate(csv_files):
        ticker = file.replace('.csv', '').replace('TV_Data_Full_', '')
        try:
            df = pd.read_csv(os.path.join(data_dir, file))
            if len(df) < params.high_len + 20: continue
            df.columns = [c.capitalize() for c in df.columns]
            df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].replace(0, np.nan).ffill()
            date_col = 'Time' if 'Time' in df.columns else 'Date'
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
            df, logs = prep_stock_data_and_trades(df, params)
            all_dfs[ticker] = df
            all_trade_logs[ticker] = logs
            master_dates.update(df.index)
        except: continue
        if count % 50 == 0: print(f"{C_GRAY}   進度: 已處理 {count} 檔股票資料...{C_RESET}", end="\r")

    sorted_dates = sorted(list(master_dates))
    all_dfs_fast = {ticker: df.to_dict('index') for ticker, df in all_dfs.items()}
    print(f"\n{C_GREEN}✅ 預處理完成！啟動時間軸回測...{C_RESET}\n")

    initial_capital = params.initial_capital
    cash = initial_capital

    portfolio = {} 
    trade_history, equity_curve = [], []
    peak_equity = initial_capital
    max_drawdown = 0.0
    current_equity = initial_capital 

    for i in range(1, len(sorted_dates)):
        today = sorted_dates[i]
        yesterday = sorted_dates[i-1]
        
        held_yesterday = set(portfolio.keys())
        
        if i % 20 == 0:
            exp = ((current_equity - cash) / current_equity) * 100 if current_equity > 0 else 0
            print(f"{C_GRAY}⏳ 推進中: {today.strftime('%Y-%m')} | 資產: {current_equity:,.0f} | 水位: {exp:>5.1f}%...{C_RESET}", end="\r", flush=True)
            
        candidates_today = []
        tickers_to_remove = []

        # ------------------------------------------
        # 階段 A: 處理賣出訊號
        # ------------------------------------------
        for ticker, pos in portfolio.items():
            fast_df = all_dfs_fast[ticker]
            if today not in fast_df: continue 
            row = fast_df[today]
            y_row = fast_df[yesterday] if yesterday in fast_df else row
            
            is_ind_sell = y_row['ind_sell_signal']
            
            if is_ind_sell or row['Low'] <= pos['sl']:
                exec_price = row['Open'] if is_ind_sell else min(row['Open'], pos['sl'])
                exec_price = adjust_to_tick(exec_price)
                net_price = calc_net_sell_price(exec_price, pos['qty'], params)
                pnl = (net_price - pos['entry']) * pos['qty']
                cash += (net_price * pos['qty'])
                
                t_type = "指標賣訊" if is_ind_sell else "停損出場"
                trade_history.append({"Date": today.strftime('%Y-%m-%d'), "Ticker": ticker, "Type": t_type, "Price": exec_price, "PnL": pnl, "Risk": params.fixed_risk})
                tickers_to_remove.append(ticker)
                continue
                
            if not pos['sold_half'] and row['High'] >= pos['tp_half']:
                sell_qty = int(np.floor(pos['qty'] * params.tp_percent))
                if sell_qty > 0:
                    exec_price = adjust_to_tick(max(row['Open'], pos['tp_half']))
                    net_price = calc_net_sell_price(exec_price, sell_qty, params)
                    pnl = (net_price - pos['entry']) * sell_qty
                    cash += (net_price * sell_qty)
                    pos['qty'] -= sell_qty
                    pos['sold_half'] = True
                    trade_history.append({"Date": today.strftime('%Y-%m-%d'), "Ticker": ticker, "Type": "半倉停利", "Price": exec_price, "PnL": pnl, "Risk": params.fixed_risk})
            
            new_sl = adjust_to_tick(row['Close'] - (row['ATR'] * params.atr_times_trail))
            if new_sl > pos['sl']: pos['sl'] = new_sl

        for t in tickers_to_remove: del portfolio[t]

        # ------------------------------------------
        # 階段 B: 尋找買進訊號
        # ------------------------------------------
        for ticker, fast_df in all_dfs_fast.items():
            if ticker in portfolio: continue 
            if ticker in held_yesterday: continue 
            if today not in fast_df or yesterday not in fast_df: continue
            
            y_row = fast_df[yesterday]
            t_row = fast_df[today]
            
            if y_row['is_setup'] and t_row['Low'] <= y_row['buy_limit']:
                is_candidate, ev, win_rate = get_pit_stats(all_trade_logs[ticker], yesterday)
                if is_candidate:
                    buy_price = adjust_to_tick(min(t_row['Open'], y_row['buy_limit']))
                    sl_price = adjust_to_tick(buy_price - (y_row['ATR'] * params.atr_times_init))
                    candidates_today.append({'ticker': ticker, 'buy_price': buy_price, 'sl_price': sl_price, 'ev': ev})

        # ------------------------------------------
        # 階段 C: 資金分配與換股邏輯 (🌟 全面套用 1% 固定風險)
        # ------------------------------------------
        if candidates_today:
            candidates_today.sort(key=lambda x: x['ev'], reverse=True)
            for cand in candidates_today:
                if len(portfolio) < max_positions:
                    qty = calc_position_size(cand['buy_price'], cand['sl_price'], current_equity, params.fixed_risk, params)
                    if qty > 0:
                        entry_cost_per_share = calc_entry_price(cand['buy_price'], qty, params)
                        cost_total = entry_cost_per_share * qty
                        if cost_total <= cash:
                            cash -= cost_total
                            net_sl_per_share = calc_net_sell_price(cand['sl_price'], qty, params)
                            tp_target = adjust_to_tick(cand['buy_price'] + (entry_cost_per_share - net_sl_per_share))
                            portfolio[cand['ticker']] = {'qty': qty, 'entry': entry_cost_per_share, 'sl': cand['sl_price'], 'tp_half': tp_target, 'sold_half': False, 'risk_used': params.fixed_risk, 'last_px': cand['buy_price']}
                            trade_history.append({"Date": today.strftime('%Y-%m-%d'), "Ticker": cand['ticker'], "Type": f"買進 (EV:{cand['ev']:.2f})", "Price": cand['buy_price'], "PnL": 0, "Risk": params.fixed_risk})
                
                elif len(portfolio) == max_positions and enable_rotation:
                    weakest_ticker, lowest_return = None, 0.0 
                    for pt, pos in portfolio.items():
                        if today not in all_dfs_fast[pt]: continue 
                        ret = (all_dfs_fast[pt][today]['Close'] - pos['entry']) / pos['entry']
                        if ret < lowest_return:
                            _, holding_ev, _ = get_pit_stats(all_trade_logs[pt], yesterday)
                            if holding_ev < cand['ev']:
                                lowest_return = ret
                                weakest_ticker = pt
                            
                    if weakest_ticker: 
                        pos = portfolio[weakest_ticker]
                        sell_price = adjust_to_tick(all_dfs_fast[weakest_ticker][today]['Close'])
                        net_price = calc_net_sell_price(sell_price, pos['qty'], params)
                        pnl = (net_price - pos['entry']) * pos['qty']
                        cash += (net_price * pos['qty'])
                        trade_history.append({"Date": today.strftime('%Y-%m-%d'), "Ticker": weakest_ticker, "Type": "汰弱賣出", "Price": sell_price, "PnL": pnl, "Risk": params.fixed_risk})
                        del portfolio[weakest_ticker]
                        
                        qty = calc_position_size(cand['buy_price'], cand['sl_price'], current_equity, params.fixed_risk, params)
                        if qty > 0:
                            entry_cost_per_share = calc_entry_price(cand['buy_price'], qty, params)
                            cost_total = entry_cost_per_share * qty
                            if cost_total <= cash:
                                cash -= cost_total
                                net_sl_per_share = calc_net_sell_price(cand['sl_price'], qty, params)
                                tp_target = adjust_to_tick(cand['buy_price'] + (entry_cost_per_share - net_sl_per_share))
                                portfolio[cand['ticker']] = {'qty': qty, 'entry': entry_cost_per_share, 'sl': cand['sl_price'], 'tp_half': tp_target, 'sold_half': False, 'risk_used': params.fixed_risk, 'last_px': cand['buy_price']}
                                trade_history.append({"Date": today.strftime('%Y-%m-%d'), "Ticker": cand['ticker'], "Type": f"汰弱換入 (EV:{cand['ev']:.2f})", "Price": cand['buy_price'], "PnL": 0, "Risk": params.fixed_risk})

        # ------------------------------------------
        # 盤後結算
        # ------------------------------------------
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

        equity_curve.append({
            "Date": today.strftime('%Y-%m-%d'), 
            "Equity": today_equity,
            "Invested_Amount": invested_capital, 
            "Exposure_Pct": exposure_pct 
        })
        
        if today_equity > peak_equity: peak_equity = today_equity
        drawdown = (peak_equity - today_equity) / peak_equity * 100
        if drawdown > max_drawdown: max_drawdown = drawdown

    print(" " * 120, end="\r") 
    df_equity = pd.DataFrame(equity_curve)
    df_trades = pd.DataFrame(trade_history)
    total_return = (today_equity - params.initial_capital) / params.initial_capital * 100
    win_rate = len(df_trades[df_trades['PnL'] > 0]) / len(df_trades[df_trades['PnL'] != 0]) * 100 if len(df_trades[df_trades['PnL'] != 0]) > 0 else 0

    return df_equity, df_trades, total_return, max_drawdown, win_rate, today_equity

if __name__ == "__main__":
    import time
    DATA_DIR = "tw_stock_data_vip"
    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(f"⚙️ {C_YELLOW}V16 投資組合模擬器：固定 1% 嚴格資金控管版{C_RESET}")
    print(f"{C_CYAN}================================================================================{C_RESET}")
    
    ans_rot = input(f"👉 1. 是否啟用「汰弱換股」模式？ (輸入 Y 啟用，直接按 Enter 預設為 N 鎖倉): ").strip().upper()
    USER_ROTATION = True if ans_rot == 'Y' else False
    ans_pos = input(f"👉 2. 請設定「最大持倉數量」 (直接按 Enter 預設為 10): ").strip()
    USER_MAX_POS = int(ans_pos) if ans_pos.isdigit() else 10

    params, is_loaded = load_dynamic_params("v16_best_params.json")
    if is_loaded: print(f"\n{C_GREEN}✅ 成功載入 AI 訓練大腦！{C_RESET}")
    else: print(f"\n{C_YELLOW}⚠️ 找不到 v16_best_params.json，使用預設參數。{C_RESET}")

    start_time = time.time()
    df_eq, df_tr, tot_ret, mdd, win_rate, final_eq = run_portfolio_simulation(
        DATA_DIR, params, max_positions=USER_MAX_POS, enable_rotation=USER_ROTATION
    )
    end_time = time.time()
    
    avg_exposure = df_eq['Exposure_Pct'].mean()
    max_exposure = df_eq['Exposure_Pct'].max()
    mode_display = "開啟 (強勢輪動)" if USER_ROTATION else "關閉 (穩定鎖倉)"
    
    print(f"\n{C_CYAN}================================================================================{C_RESET}")
    print(f"📊 【投資組合實戰模擬報告 (Point-In-Time 真實重現)】")
    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(f"⚙️ 汰弱換股模式:   {mode_display}")
    print(f"💼 最大持股上限:   {USER_MAX_POS} 檔")
    print(f"⏱️ 回測總耗時:     {end_time - start_time:.2f} 秒")
    print(f"💰 最終總資產:     {final_eq:,.0f} 元")
    print(f"📈 總資產報酬率:   {tot_ret:>.2f} %")
    print(f"📉 最大回撤 (MDD): {mdd:>.2f} %")
    print(f"🎯 實戰勝率:       {win_rate:>.2f} %")
    print(f"🔄 總交易紀錄:     {len(df_tr)} 筆")
    print(f"🌊 平均資金水位:   {avg_exposure:>.2f} % (最高 {max_exposure:>.2f} %)")
    print(f"{C_CYAN}================================================================================{C_RESET}")
    
    with pd.ExcelWriter("V16_Portfolio_Report.xlsx") as writer:
        df_eq.to_excel(writer, sheet_name="Equity Curve", index=False)
        df_tr.to_excel(writer, sheet_name="Trade History", index=False)
    print(f"{C_GREEN}📁 完整資產曲線與交易明細已匯出至: V16_Portfolio_Report.xlsx{C_RESET}")