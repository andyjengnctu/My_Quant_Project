import os
import sys
import time
import json
import optuna
import numpy as np
import pandas as pd
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

from core.v16_config import V16StrategyParams
from v16_portfolio_sim import prep_stock_data_and_trades, get_pit_stats
from core.v16_core import calc_net_sell_price, adjust_to_tick, calc_entry_price, calc_position_size

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

C_RED = '\033[91m' 
C_YELLOW = '\033[93m'
C_CYAN = '\033[96m'
C_GREEN = '\033[92m'
C_RESET = '\033[0m'
C_GRAY = '\033[90m'

# ==========================================
# ⚙️ 訓練專用投資組合設定
# ==========================================
TRAIN_MAX_POSITIONS = 10         
TRAIN_START_YEAR = 2015         
TRAIN_ENABLE_ROTATION = False   
DATA_DIR = "tw_stock_data_vip"
RAW_DATA_CACHE = {}             
# ==========================================

CURRENT_SESSION_TRIAL = 0  
N_TRIALS = 0  

def load_all_raw_data():
    print(f"{C_CYAN}📦 正在將歷史數據載入記憶體快取 (僅需執行一次)...{C_RESET}")
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    for count, file in enumerate(files):
        ticker = file.replace('.csv', '').replace('TV_Data_Full_', '')
        try:
            df = pd.read_csv(os.path.join(DATA_DIR, file))
            if len(df) < 150: continue
            df.columns = [c.capitalize() for c in df.columns]
            df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].replace(0, np.nan).ffill()
            date_col = 'Time' if 'Time' in df.columns else 'Date'
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
            RAW_DATA_CACHE[ticker] = df
        except: continue
        if count % 50 == 0: print(f"{C_GRAY}   進度: 已快取 {count} 檔股票...{C_RESET}", end="\r")
    print(f"\n{C_GREEN}✅ 記憶體快取完成！共載入 {len(RAW_DATA_CACHE)} 檔標的。{C_RESET}\n")

def worker_prep_data(ticker, df, params):
    try:
        df_prepared, logs = prep_stock_data_and_trades(df, params)
        return ticker, df_prepared.to_dict('index'), logs
    except:
        return ticker, None, None

def objective(trial):
    ai_use_bb = trial.suggest_categorical("use_bb", [True, False])
    ai_use_kc = trial.suggest_categorical("use_kc", [True, False])
    ai_use_vol = trial.suggest_categorical("use_vol", [True, False]) 

    ai_params = V16StrategyParams(
        atr_len = trial.suggest_int("atr_len", 3, 20), # orginal 5~20
        atr_times_init = trial.suggest_float("atr_times_init", 1.0, 3.5, step=0.1),
        atr_times_trail = trial.suggest_float("atr_times_trail", 2.0, 4.5, step=0.1), 
        atr_buy_tol = trial.suggest_float("atr_buy_tol", 0.1, 1.5, step=0.1),
        high_len = trial.suggest_int("high_len", 40, 200, step=1), # orginal 40~150, 5
        tp_percent = trial.suggest_float("tp_percent", 0.0, 0.6, step=0.01), 
        use_bb = ai_use_bb, 
        use_kc = ai_use_kc, 
        use_vol = ai_use_vol,
        bb_len = trial.suggest_int("bb_len", 10, 30, step=1) if ai_use_bb else 20,#10~30, 2
        bb_mult = trial.suggest_float("bb_mult", 1.0, 2.5, step=0.1) if ai_use_bb else 2.0,
        kc_len = trial.suggest_int("kc_len", 3, 30, step=1) if ai_use_kc else 20,# original 10~30, 2
        kc_mult = trial.suggest_float("kc_mult", 1.0, 3.0, step=0.1) if ai_use_kc else 2.0, # orginal 1.5 ~3.0
        vol_short_len = trial.suggest_int("vol_short_len", 1, 10) if ai_use_vol else 5,
        vol_long_len = trial.suggest_int("vol_long_len", 5, 30) if ai_use_vol else 19, # orginal 10~30
        use_compounding = True 
    )

    all_dfs_fast = {}
    all_trade_logs = {}
    master_dates = set()

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(worker_prep_data, ticker, df, ai_params) for ticker, df in RAW_DATA_CACHE.items()]
        for future in as_completed(futures):
            ticker, fast_df, logs = future.result()
            if fast_df:
                all_dfs_fast[ticker] = fast_df
                all_trade_logs[ticker] = logs
                master_dates.update(fast_df.keys())

    if not master_dates: return -9999.0

    sorted_dates = sorted(list(master_dates))
    start_dt = pd.to_datetime(f"{TRAIN_START_YEAR}-01-01")
    start_idx = next((i for i, d in enumerate(sorted_dates) if d >= start_dt), 1)

    initial_capital = ai_params.initial_capital
    cash = initial_capital
    portfolio = {} 
    peak_equity = initial_capital
    max_drawdown = 0.0
    trade_count = 0
    
    closed_trades_stats = []
    
    total_exposure = 0.0
    sim_days = 0
    benchmark_data = all_dfs_fast.get("0050", None)
    bm_start_px, bm_peak_px, last_bm_px = None, None, None
    bm_mdd = 0.0

    for i in range(start_idx, len(sorted_dates)):
        sim_days += 1
        today = sorted_dates[i]
        yesterday = sorted_dates[i-1]
        held_yesterday = set(portfolio.keys())
        candidates_today = []
        tickers_to_remove = []

        for ticker, pos in portfolio.items():
            fast_df = all_dfs_fast[ticker]
            if today not in fast_df: continue 
            row = fast_df[today]
            y_row = fast_df[yesterday] if yesterday in fast_df else row
            is_ind_sell = y_row['ind_sell_signal']
            
            if is_ind_sell or row['Low'] <= pos['sl']:
                exec_price = adjust_to_tick(row['Open'] if is_ind_sell else min(row['Open'], pos['sl']))
                net_price = calc_net_sell_price(exec_price, pos['qty'], ai_params)
                cash += (net_price * pos['qty'])
                
                risk_per_share = pos['entry'] - pos['initial_sl_net']
                if risk_per_share <= 0: risk_per_share = pos['entry'] * 0.01
                r_mult = (net_price - pos['entry']) / risk_per_share
                closed_trades_stats.append({'pnl': net_price - pos['entry'], 'r_mult': r_mult})
                
                tickers_to_remove.append(ticker)
                trade_count += 1
                continue
                
            if not pos['sold_half'] and row['High'] >= pos['tp_half']:
                sell_qty = int(np.floor(pos['qty'] * ai_params.tp_percent))
                if sell_qty > 0:
                    exec_price = adjust_to_tick(max(row['Open'], pos['tp_half']))
                    net_price = calc_net_sell_price(exec_price, sell_qty, ai_params)
                    cash += (net_price * sell_qty)
                    
                    risk_per_share = pos['entry'] - pos['initial_sl_net']
                    if risk_per_share <= 0: risk_per_share = pos['entry'] * 0.01
                    r_mult = (net_price - pos['entry']) / risk_per_share
                    closed_trades_stats.append({'pnl': net_price - pos['entry'], 'r_mult': r_mult})
                    
                    pos['qty'] -= sell_qty
                    pos['sold_half'] = True
                    trade_count += 1
            
            new_sl = adjust_to_tick(row['Close'] - (row['ATR'] * ai_params.atr_times_trail))
            if new_sl > pos['sl']: pos['sl'] = new_sl

        for t in tickers_to_remove: del portfolio[t]

        for ticker, fast_df in all_dfs_fast.items():
            if ticker in portfolio or ticker in held_yesterday: continue 
            if today not in fast_df or yesterday not in fast_df: continue
            
            y_row, t_row = fast_df[yesterday], fast_df[today]
            if y_row['is_setup'] and t_row['Low'] <= y_row['buy_limit']:
                is_candidate, ev, win_rate = get_pit_stats(all_trade_logs[ticker], yesterday)
                if is_candidate:
                    buy_price = adjust_to_tick(min(t_row['Open'], y_row['buy_limit']))
                    sl_price = adjust_to_tick(buy_price - (y_row['ATR'] * ai_params.atr_times_init))
                    candidates_today.append({'ticker': ticker, 'buy_price': buy_price, 'sl_price': sl_price, 'ev': ev})

        current_equity_for_sizing = cash + sum([pos['qty'] * pos.get('last_px', pos['entry']) for pos in portfolio.values()])
        
        if candidates_today:
            candidates_today.sort(key=lambda x: x['ev'], reverse=True)
            for cand in candidates_today:
                if len(portfolio) < TRAIN_MAX_POSITIONS:
                    qty = calc_position_size(cand['buy_price'], cand['sl_price'], current_equity_for_sizing, ai_params.fixed_risk, ai_params)
                    if qty > 0:
                        entry_cost_per_share = calc_entry_price(cand['buy_price'], qty, ai_params)
                        if (entry_cost_per_share * qty) <= cash:
                            cash -= (entry_cost_per_share * qty)
                            net_sl_per_share = calc_net_sell_price(cand['sl_price'], qty, ai_params)
                            portfolio[cand['ticker']] = {
                                'qty': qty, 'entry': entry_cost_per_share, 'sl': cand['sl_price'], 
                                'initial_sl_net': net_sl_per_share,
                                'tp_half': adjust_to_tick(cand['buy_price'] + (entry_cost_per_share - net_sl_per_share)), 
                                'sold_half': False, 'last_px': cand['buy_price']
                            }

        today_equity = cash  
        invested_capital = 0.0
        for pt, pos in portfolio.items():
            px = all_dfs_fast[pt][today]['Close'] if today in all_dfs_fast[pt] else pos.get('last_px', pos['entry'])
            pos['last_px'] = px 
            val = pos['qty'] * px
            today_equity += val
            invested_capital += val
            
        total_exposure += (invested_capital / today_equity * 100) if today_equity > 0 else 0.0
            
        if today_equity > peak_equity: peak_equity = today_equity
        drawdown = (peak_equity - today_equity) / peak_equity * 100
        if drawdown > max_drawdown: max_drawdown = drawdown
            
        if benchmark_data and today in benchmark_data:
            curr_bm_px = benchmark_data[today]['Close']
            last_bm_px = curr_bm_px
            if bm_start_px is None:
                bm_start_px = curr_bm_px
                bm_peak_px = curr_bm_px
            if curr_bm_px > bm_peak_px: bm_peak_px = curr_bm_px
            bm_drawdown = (bm_peak_px - curr_bm_px) / bm_peak_px * 100
            if bm_drawdown > bm_mdd: bm_mdd = bm_drawdown

    total_return_pct = ((today_equity - initial_capital) / initial_capital) * 100
    avg_exposure = total_exposure / sim_days if sim_days > 0 else 0.0
    bm_return_pct = ((last_bm_px - bm_start_px) / bm_start_px * 100) if (bm_start_px and last_bm_px) else 0.0

    if closed_trades_stats:
        wins = [t for t in closed_trades_stats if t['pnl'] > 0]
        losses = [t for t in closed_trades_stats if t['pnl'] <= 0]
        win_rate = (len(wins) / len(closed_trades_stats)) * 100
        pf_ev = sum(t['r_mult'] for t in closed_trades_stats) / len(closed_trades_stats)
        
        avg_win_r = sum(t['r_mult'] for t in wins) / len(wins) if wins else 0
        avg_loss_r = abs(sum(t['r_mult'] for t in losses) / len(losses)) if losses else 0
        pf_payoff = (avg_win_r / avg_loss_r) if avg_loss_r > 0 else (99.9 if avg_win_r > 0 else 0)
    else:
        win_rate, pf_ev, pf_payoff = 0.0, 0.0, 0.0

    if max_drawdown > 45.0:
        trial.set_user_attr("fail_reason", f"投資組合回撤過大 ({max_drawdown:.1f}%)")
        return -9999.0
    if trade_count < (len(sorted_dates) / 10): 
        trial.set_user_attr("fail_reason", f"資金利用率/交易次數過低 ({trade_count}次)")
        return -9999.0
    if total_return_pct <= 0:
        trial.set_user_attr("fail_reason", f"最終虧損 ({total_return_pct:.1f}%)")
        return -9999.0

    final_romd = total_return_pct / (max_drawdown + 0.1)

    trial.set_user_attr("pf_return", total_return_pct)
    trial.set_user_attr("pf_mdd", max_drawdown)
    trial.set_user_attr("pf_trades", trade_count)
    trial.set_user_attr("final_equity", today_equity)
    trial.set_user_attr("avg_exposure", avg_exposure)
    trial.set_user_attr("bm_return", bm_return_pct)
    trial.set_user_attr("bm_mdd", bm_mdd)
    trial.set_user_attr("win_rate", win_rate)
    trial.set_user_attr("pf_ev", pf_ev)
    trial.set_user_attr("pf_payoff", pf_payoff)
        
    return final_romd

def monitoring_callback(study, trial):
    global CURRENT_SESSION_TRIAL
    CURRENT_SESSION_TRIAL += 1
    duration = trial.duration.total_seconds() if trial.duration else 0.0
    
    if trial.value and trial.value <= -9000:
        fail_msg = trial.user_attrs.get("fail_reason", "策略無效")
        status_text = f"{C_YELLOW}淘汰 [{fail_msg}]{C_RESET}"
        score_text = "N/A"
    else:
        status_text = f"{C_GREEN}進化中{C_RESET}"
        score_text = f"{trial.value:.3f}"
        
    line_text = f"{C_GRAY}⏳ [累積 {trial.number + 1:>4} | 本輪 {CURRENT_SESSION_TRIAL:>3}/{N_TRIALS}] 耗時: {duration:>5.1f}s | RoMD分數: {score_text:>7} | 狀態: {status_text}{C_RESET}"
    print(f"\r{line_text}\033[K", end="", flush=True)

    if study.best_trial.number == trial.number and trial.value and trial.value > -9000:
        print() 
        attrs = trial.user_attrs
        p = trial.params
        
        bb_str = f"啟用 ({p.get('bb_len', 20)} 日, 寬度 {p.get('bb_mult', 2.0):.1f} 倍)" if p.get('use_bb', False) else "[關閉]"
        kc_str = f"啟用 ({p.get('kc_len', 20)} 日, 寬度 {p.get('kc_mult', 2.0):.1f} 倍)" if p.get('use_kc', False) else "[關閉]"
        vol_str = f"啟用 (短 {p.get('vol_short_len', 5)} 日 > 長 {p.get('vol_long_len', 19)} 日)" if p.get('use_vol', False) else "[關閉]"

        sys_ret, bm_ret = attrs['pf_return'], attrs['bm_return']
        sys_mdd, bm_mdd = attrs['pf_mdd'], attrs['bm_mdd']
        alpha = sys_ret - bm_ret
        mdd_diff = bm_mdd - sys_mdd 
        
        sys_romd = trial.value
        bm_romd = bm_ret / (bm_mdd + 0.1) if bm_mdd != 0 else 0.0
        romd_diff = sys_romd - bm_romd

        alpha_str = f"{'+' if alpha > 0 else ''}{alpha:.2f}%"
        alpha_color = C_GREEN if alpha > 0 else C_RED
        sys_ret_str = f"{'+' if sys_ret > 0 else ''}{sys_ret:.2f}%"
        sys_ret_color = C_GREEN if sys_ret > 0 else C_RED
        bm_ret_str = f"{'+' if bm_ret > 0 else ''}{bm_ret:.2f}%"
        sys_mdd_str = f"-{sys_mdd:.2f}%"
        bm_mdd_str = f"-{bm_mdd:.2f}%"
        mdd_diff_str = f"少跌 {mdd_diff:.2f}%" if mdd_diff > 0 else f"多跌 {abs(mdd_diff):.2f}%"
        mdd_diff_color = C_GREEN if mdd_diff > 0 else C_RED
        romd_diff_str = f"{'+' if romd_diff > 0 else ''}{romd_diff:.2f}"
        romd_diff_color = C_GREEN if romd_diff > 0 else C_RED
        
        mode_display = "啟用 (汰弱換強)" if TRAIN_ENABLE_ROTATION else "關閉 (穩定鎖倉)"

        # 🌟 印出華麗戰情面板
        print(f"\n{C_RED}🏆 破紀錄！發現更強的投資組合參數！ (累積第 {trial.number + 1} 次測試){C_RESET}")
        print(f"{C_GRAY}--------------------------------------------------------------------------------{C_RESET}")
        print(f"模式: {mode_display} | 最大持股: {TRAIN_MAX_POSITIONS} 檔")
        print(f"總交易紀錄: {attrs['pf_trades']} 筆 | 最終資產: {attrs['final_equity']:,.0f} 元")
        print(f"平均資金水位: {attrs['avg_exposure']:.2f} %")
        print(f"{C_GRAY}--------------------------------------------------------------------------------{C_RESET}")
        print(f"🏆 績效與風險對比表")
        print(f"| 指標項目         | V16 尊爵系統   | 同期大盤 (0050) | 差異 (Alpha)   |")
        print(f"|------------------|----------------|-----------------|----------------|")
        print(f"| 總資產報酬率     | {sys_ret_color}{sys_ret_str:<14}{C_RESET} | {bm_ret_str:<15} | {alpha_color}{alpha_str:<14}{C_RESET} |")
        print(f"| 最大回撤 (MDD)   | {C_YELLOW}{sys_mdd_str:<14}{C_RESET} | {bm_mdd_str:<15} | {mdd_diff_color}{mdd_diff_str:<14}{C_RESET} |")
        print(f"| 報酬回撤比(RoMD) | {C_CYAN}{sys_romd:.2f}{' ' * max(0, 14-len(f'{sys_romd:.2f}'))}{C_RESET} | {bm_romd:<15.2f} | {romd_diff_color}{romd_diff_str:<14}{C_RESET} |")
        print(f"| 系統實戰勝率     | {attrs['win_rate']:>6.2f} %       | -               | -              |")
        print(f"| 盈虧風報比       | {attrs['pf_payoff']:>6.2f}         | -               | -              |")
        print(f"| 實戰期望值(EV)   | {attrs['pf_ev']:>6.2f} R       | -               | -              |")
        print(f"{C_GRAY}--------------------------------------------------------------------------------{C_RESET}")
        print(f"⚙️ [核心參數] 突破: {p['high_len']:>3} 日新高 | ATR 週期: {p['atr_len']:>2} 日 | 半倉停利: {p.get('tp_percent', 0.5)*100:>2.0f} %")
        print(f"              掛單: +{p.get('atr_buy_tol', 1.5):.1f} ATR | 停損: -{p['atr_times_init']:.1f} ATR | 追蹤停利: -{p['atr_times_trail']:.1f} ATR")
        print(f"🛡️ [濾網決策]布林通道 (BB) : {bb_str}")
        print(f"                阿肯那(KC) : {kc_str}")
        print(f"                均量 (Vol) : {vol_str}")
        print(f"{C_CYAN}================================================================================{C_RESET}\n")

        # 🌟 新增：將純文字結果寫入日誌檔 (outputs/portfolio_breakthrough_log.txt)
        os.makedirs("outputs", exist_ok=True)
        log_content = (
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🏆 破紀錄！ 累積第 {trial.number + 1} 次測試\n"
            f"   📊 [投資組合表現] 最終總權益: {attrs['final_equity']:,.0f} 元 | 總報酬: {attrs['pf_return']:.2f}% | RoMD: {trial.value:.3f}\n"
            f"   🛡️ [風險與交易] 最大回撤: {attrs['pf_mdd']:.2f}% | 勝率: {attrs['win_rate']:.2f}% | EV: {attrs['pf_ev']:.2f}R | 總交易數: {attrs['pf_trades']} 次\n"
            f"   ⚙️ [核心參數] 突破: {p['high_len']:>3}日 | ATR: {p['atr_len']:>2}日 | 半倉停利: {p.get('tp_percent', 0.5)*100:>2.0f}%\n"
            f"                 掛單: +{p.get('atr_buy_tol', 1.5):.1f} ATR | 停損: -{p['atr_times_init']:.1f} ATR | 追蹤: -{p['atr_times_trail']:.1f} ATR\n"
            f"   🛡️ [濾網決策] 布林通道(BB): {bb_str} | 阿肯那(KC): {kc_str} | 均量(Vol): {vol_str}\n"
            f"--------------------------------------------------------------------------------\n"
        )
        with open("outputs/portfolio_breakthrough_log.txt", "a", encoding="utf-8") as f:
            f.write(log_content)

if __name__ == "__main__":
    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(f"⚙️ {C_YELLOW}V16 端到端 (End-to-End) 投資組合極速 AI 訓練引擎啟動{C_RESET}")
    print(f"{C_CYAN}================================================================================{C_RESET}")
    
    load_all_raw_data()
    
    if len(RAW_DATA_CACHE) == 0:
        print("❌ 快取失敗，沒有可用的股票資料。")
        sys.exit()

    # 🌟 因為是 10 檔持倉環境，建議命名獨立的資料庫與日誌檔
    db_file_name = "models/v16_portfolio_ai_10pos.db"  
    log_file_name = "outputs/portfolio_breakthrough_log.txt"
    DB_NAME = f"sqlite:///{db_file_name}"

    if os.path.exists(db_file_name):
        choice = input(f"\n👉 發現舊有 Portfolio 記憶庫！ [1] 接續訓練  [2] 刪除重來 (預設 1): ").strip()
        if choice == '2':
            os.remove(db_file_name)
            # 同步刪除舊日誌防呆
            if os.path.exists(log_file_name):
                os.remove(log_file_name)
            print(f"{C_RED}🗑️ 已刪除舊記憶與歷史軌跡。{C_RESET}")
    
    trials_input = input(f"👉 請輸入訓練次數 (直接按 Enter 預設 50,000 次): ").strip()
    N_TRIALS = int(trials_input) if trials_input.isdigit() else 50000
    
    study = optuna.create_study(study_name="v16_portfolio_optimization", storage=DB_NAME, load_if_exists=True, direction="maximize")
    
    # 🌟 新增：接續訓練時，印出「歷史突破軌跡紀錄」
    completed_trials = len(study.trials)
    if completed_trials > 0:
        print(f"\n{C_GREEN}✅ 記憶庫載入成功！已累積 {completed_trials} 次經驗。{C_RESET}")
        if os.path.exists(log_file_name):
            print(f"\n{C_CYAN}📜 【歷史突破軌跡紀錄】 (來自 {log_file_name}){C_RESET}")
            with open(log_file_name, "r", encoding="utf-8") as f:
                lines = f.readlines()
                # 顯示倒數約 3 次的破紀錄
                if len(lines) > 21:
                    print(f"{C_GRAY}...(省略早期紀錄，僅顯示最近 3 次進化突破)...\n" + "".join(lines[-21:]) + f"{C_RESET}")
                else:
                    print(f"{C_GRAY}" + "".join(lines) + f"{C_RESET}")
    else:
        print(f"\n{C_YELLOW}🌟 全新記憶庫建立完成！{C_RESET}")

    print(f"\n{C_CYAN}🚀 開始進行投資組合端到端優化...{C_RESET}\n")
    try:
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=1, callbacks=[monitoring_callback])
    except KeyboardInterrupt:
        print(f"\n\n{C_YELLOW}⚠️ 手動中斷，進度已保留。{C_RESET}")
        sys.exit()
    
    try:
        if study.best_value and study.best_value > -9000:
            with open("models/v16_best_params.json", "w") as f:
                json.dump(study.best_params, f, indent=4)
            print(f"{C_GREEN}💾 已成功將最強投資組合參數匯出至 'models/v16_best_params.json'！{C_RESET}")
    except ValueError:
        pass