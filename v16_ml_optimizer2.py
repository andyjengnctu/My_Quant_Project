import os
import sys
import time
import json
import optuna
import numpy as np
import pandas as pd
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

from core.v16_config import V16StrategyParams, SCORE_CALC_METHOD
from core.v16_portfolio_engine import prep_stock_data_and_trades, run_portfolio_timeline, calc_portfolio_score
from core.v16_display import print_strategy_dashboard, C_RED, C_YELLOW, C_CYAN, C_GREEN, C_GRAY, C_RESET

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

TRAIN_MAX_POSITIONS = 10         
TRAIN_START_YEAR = 2015         
TRAIN_ENABLE_ROTATION = False   
DATA_DIR = "tw_stock_data_vip"
RAW_DATA_CACHE = {}             
CURRENT_SESSION_TRIAL, N_TRIALS = 0, 0  

def load_all_raw_data():
    print(f"{C_CYAN}📦 正在將歷史數據載入記憶體快取 (僅需執行一次)...{C_RESET}")
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    for count, file in enumerate(files):
        ticker = file.replace('.csv', '').replace('TV_Data_Full_', '')
        try:
            df = pd.read_csv(os.path.join(DATA_DIR, file))
            # (AI註: 修復 4 - 第一階段: 快取時放寬過濾條件到 50，將嚴格篩選交給 Worker 動態處理)
            if len(df) < 50: continue
            df.columns = [c.capitalize() for c in df.columns]
            df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].replace(0, np.nan).ffill()
            date_col = 'Time' if 'Time' in df.columns else 'Date'
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
            RAW_DATA_CACHE[ticker] = df
        except Exception as e:
            print(f"{C_YELLOW}\n[警告] 載入 {ticker} 發生錯誤: {e}{C_RESET}")
            continue
        if count % 50 == 0: print(f"{C_GRAY}   進度: 已快取 {count} 檔股票...{C_RESET}", end="\r")
    print(f"\n{C_GREEN}✅ 記憶體快取完成！共載入 {len(RAW_DATA_CACHE)} 檔標的。{C_RESET}\n")

def worker_prep_data(ticker, df, params):
    try:
        # (AI註: 修復 4 - 第二階段: AI 動態生成 high_len 後，嚴格執行與 Scanner 一致的股票濾網)
        if len(df) < params.high_len + 10:
            return ticker, None, None
            
        df_prepared, logs = prep_stock_data_and_trades(df, params)
        return ticker, df_prepared.to_dict('index'), logs
    except Exception as e: 
        return ticker, None, None

def objective(trial):
    ai_use_bb = trial.suggest_categorical("use_bb", [True, False])
    ai_use_kc = trial.suggest_categorical("use_kc", [True, False])
    ai_use_vol = trial.suggest_categorical("use_vol", [True, False]) 
    ai_params = V16StrategyParams(
        atr_len = trial.suggest_int("atr_len", 3, 25), 
        atr_times_init = trial.suggest_float("atr_times_init", 1.0, 3.5, step=0.1),
        atr_times_trail = trial.suggest_float("atr_times_trail", 2.0, 4.5, step=0.1), 
        atr_buy_tol = trial.suggest_float("atr_buy_tol", 0.1, 3.5, step=0.1),
        high_len = trial.suggest_int("high_len", 40, 250, step=1), 
        tp_percent = trial.suggest_float("tp_percent", 0.0, 0.6, step=0.01), 
        use_bb = ai_use_bb, use_kc = ai_use_kc, use_vol = ai_use_vol,
        bb_len = trial.suggest_int("bb_len", 10, 30, step=1) if ai_use_bb else 20,
        bb_mult = trial.suggest_float("bb_mult", 1.0, 2.5, step=0.1) if ai_use_bb else 2.0,
        kc_len = trial.suggest_int("kc_len", 3, 30, step=1) if ai_use_kc else 20,
        kc_mult = trial.suggest_float("kc_mult", 1.0, 3.0, step=0.1) if ai_use_kc else 2.0, 
        vol_short_len = trial.suggest_int("vol_short_len", 1, 10) if ai_use_vol else 5,
        vol_long_len = trial.suggest_int("vol_long_len", 5, 30) if ai_use_vol else 19, 
        min_history_trades = trial.suggest_int("min_history_trades", 1, 5),
        min_history_ev = trial.suggest_float("min_history_ev", -1.0, 0.5, step=0.1),
        min_history_win_rate = trial.suggest_float("min_history_win_rate", 0.0, 0.6, step=0.05),
        use_compounding = True 
    )
    all_dfs_fast, all_trade_logs, master_dates = {}, {}, set()
    with ProcessPoolExecutor(max_workers=14) as executor:
        futures = [executor.submit(worker_prep_data, ticker, df, ai_params) for ticker, df in RAW_DATA_CACHE.items()]
        for future in as_completed(futures):
            ticker, fast_df, logs = future.result()
            if fast_df:
                all_dfs_fast[ticker], all_trade_logs[ticker] = fast_df, logs
                master_dates.update(fast_df.keys())
    if not master_dates: return -9999.0
    sorted_dates = sorted(list(master_dates))
    benchmark_data = all_dfs_fast.get("0050", None)
    
    ret_pct, mdd, t_count, final_eq, avg_exp, bm_ret, bm_mdd, win_rate, pf_ev, pf_payoff, total_missed, total_missed_sells, r_sq, m_win_rate, bm_r_sq, bm_m_win_rate = run_portfolio_timeline(
        all_dfs_fast, all_trade_logs, sorted_dates, TRAIN_START_YEAR, ai_params, 
        TRAIN_MAX_POSITIONS, TRAIN_ENABLE_ROTATION, 
        benchmark_ticker="0050", benchmark_data=benchmark_data, is_training=True
    )
    
    if mdd > 45.0: trial.set_user_attr("fail_reason", f"回撤過大 ({mdd:.1f}%)"); return -9999.0
    if t_count < 30: trial.set_user_attr("fail_reason", f"交易次數過低 ({t_count}次)"); return -9999.0
    if ret_pct <= 0: trial.set_user_attr("fail_reason", f"最終虧損 ({ret_pct:.1f}%)"); return -9999.0
    if m_win_rate < 45.0: trial.set_user_attr("fail_reason", f"月勝率偏低 ({m_win_rate:.0f}%)"); return -9999.0
    if r_sq < 0.40: trial.set_user_attr("fail_reason", f"曲線過度震盪 (R²={r_sq:.2f})"); return -9999.0

    final_score = calc_portfolio_score(ret_pct, mdd, m_win_rate, r_sq)
    
    trial.set_user_attr("pf_return", ret_pct); trial.set_user_attr("pf_mdd", mdd); trial.set_user_attr("pf_trades", t_count); trial.set_user_attr("final_equity", final_eq); trial.set_user_attr("avg_exposure", avg_exp); trial.set_user_attr("bm_return", bm_ret); trial.set_user_attr("bm_mdd", bm_mdd); trial.set_user_attr("win_rate", win_rate); trial.set_user_attr("pf_ev", pf_ev); trial.set_user_attr("pf_payoff", pf_payoff); trial.set_user_attr("missed_buys", total_missed); trial.set_user_attr("missed_sells", total_missed_sells)
    trial.set_user_attr("r_squared", r_sq)
    trial.set_user_attr("m_win_rate", m_win_rate)
    trial.set_user_attr("bm_r_squared", bm_r_sq)
    trial.set_user_attr("bm_m_win_rate", bm_m_win_rate)
    
    return final_score

def monitoring_callback(study, trial):
    global CURRENT_SESSION_TRIAL
    CURRENT_SESSION_TRIAL += 1
    duration = trial.duration.total_seconds() if trial.duration else 0.0
    if trial.value and trial.value <= -9000:
        fail_msg = trial.user_attrs.get("fail_reason", "策略無效")
        status_text, score_text = f"{C_YELLOW}淘汰 [{fail_msg}]{C_RESET}", "N/A"
    else:
        status_text, score_text = f"{C_GREEN}進化中{C_RESET}", f"{trial.value:.3f}"
    print(f"\r{C_GRAY}⏳ [累積 {trial.number + 1:>4} | 本輪 {CURRENT_SESSION_TRIAL:>3}/{N_TRIALS}] 耗時: {duration:>5.1f}s | 系統評分: {score_text:>7} | 狀態: {status_text}{C_RESET}\033[K", end="", flush=True)
    
    if study.best_trial.number == trial.number and trial.value and trial.value > -9000:
        print()
        attrs, p = trial.user_attrs, trial.params
        mode_display = "啟用 (汰弱換強)" if TRAIN_ENABLE_ROTATION else "關閉 (穩定鎖倉)"
        print(f"\n{C_RED}🏆 破紀錄！發現更強的投資組合參數！ (累積第 {trial.number + 1} 次測試){C_RESET}")
        
        print_strategy_dashboard(
            params=p, title="績效與風險對比表", mode_display=mode_display, max_pos=TRAIN_MAX_POSITIONS,
            trades=attrs['pf_trades'], missed_b=attrs.get('missed_buys', 0), missed_s=attrs.get('missed_sells', 0),
            final_eq=attrs['final_equity'], avg_exp=attrs['avg_exposure'], sys_ret=attrs['pf_return'], 
            bm_ret=attrs['bm_return'], sys_mdd=attrs['pf_mdd'], bm_mdd=attrs['bm_mdd'], win_rate=attrs['win_rate'], 
            payoff=attrs['pf_payoff'], ev=attrs['pf_ev'],
            r_sq=attrs['r_squared'], m_win_rate=attrs['m_win_rate'], bm_r_sq=attrs.get('bm_r_squared', 0.0), bm_m_win_rate=attrs.get('bm_m_win_rate', 0.0)
        )

if __name__ == "__main__":
    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(f"⚙️ {C_YELLOW}V16 端到端 (End-to-End) 投資組合極速 AI 訓練引擎啟動{C_RESET}")
    print(f"{C_CYAN}================================================================================{C_RESET}")
    load_all_raw_data(); db_file = "models/v16_portfolio_ai_10pos_overnight.db"; DB_NAME = f"sqlite:///{db_file}"
    
    if os.path.exists(db_file):
        choice = input(f"\n👉 發現舊有 Portfolio 記憶庫！ [1] 接續訓練  [2] 刪除重來 (預設 1): ").strip()
        if choice == '2': 
            os.remove(db_file)
            print(f"{C_RED}🗑️ 已刪除舊記憶。{C_RESET}")
            
    user_input = input(f"👉 請輸入訓練次數 (預設 50000，輸入 0 則直接提取匯出參數): ").strip()
    N_TRIALS = int(user_input) if user_input != "" else 50000
    study = optuna.create_study(study_name="v16_portfolio_optimization_overnight", storage=DB_NAME, load_if_exists=True, direction="maximize")
    
    if len(study.trials) > 0:
        print(f"\n{C_GREEN}✅ 已累積 {len(study.trials)} 次經驗。{C_RESET}")
        try:
            best_trial = study.best_trial
            if best_trial.value and best_trial.value > -9000:
                attrs, p = best_trial.user_attrs, best_trial.params
                mode_display = "啟用 (汰弱換強)" if TRAIN_ENABLE_ROTATION else "關閉 (穩定鎖倉)"
                
                print(f"\n{C_CYAN}📜 【歷史突破紀錄還原】{C_RESET}")
                print(f"{C_RED}🏆 目前記憶庫的最強參數！ (來自累積第 {best_trial.number + 1} 次測試){C_RESET}")
                
                print_strategy_dashboard(
                    params=p, title="績效與風險對比表", mode_display=mode_display, max_pos=TRAIN_MAX_POSITIONS,
                    trades=attrs['pf_trades'], missed_b=attrs.get('missed_buys', 0), missed_s=attrs.get('missed_sells', 0),
                    final_eq=attrs['final_equity'], avg_exp=attrs['avg_exposure'], sys_ret=attrs['pf_return'], 
                    bm_ret=attrs['bm_return'], sys_mdd=attrs['pf_mdd'], bm_mdd=attrs['bm_mdd'], win_rate=attrs['win_rate'], 
                    payoff=attrs['pf_payoff'], ev=attrs['pf_ev'],
                    r_sq=attrs.get('r_squared', 0.0), m_win_rate=attrs.get('m_win_rate', 0.0), bm_r_sq=attrs.get('bm_r_squared', 0.0), bm_m_win_rate=attrs.get('bm_m_win_rate', 0.0)
                )
        except ValueError: pass

    if N_TRIALS == 0:
        try:
            if study.best_value and study.best_value > -9000:
                with open("models/v16_best_params.json", "w") as f: json.dump(study.best_params, f, indent=4)
                print(f"\n{C_GREEN}💾 匯出成功！已從記憶庫提取最強參數！{C_RESET}\n")
            else: print(f"\n{C_YELLOW}⚠️ 目前記憶庫中尚無及格的紀錄，無法匯出。{C_RESET}\n")
        except ValueError: print(f"\n{C_YELLOW}⚠️ 記憶庫為空，無法匯出。{C_RESET}\n")
    else:
        print(f"\n{C_CYAN}🚀 開始優化...{C_RESET}\n")
        try: study.optimize(objective, n_trials=N_TRIALS, n_jobs=1, callbacks=[monitoring_callback])
        except KeyboardInterrupt: pass
        print(f"\n\n{C_YELLOW}🛑 訓練階段結束或已中斷。{C_RESET}")