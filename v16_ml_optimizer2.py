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
from core.v16_portfolio_engine import prep_stock_data_and_trades, run_portfolio_timeline

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

C_RED = '\033[91m' 
C_YELLOW = '\033[93m'
C_CYAN = '\033[96m'
C_GREEN = '\033[92m'
C_RESET = '\033[0m'
C_GRAY = '\033[90m'

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
    except: return ticker, None, None

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
    
    ret_pct, mdd, t_count, final_eq, avg_exp, bm_ret, bm_mdd, win_rate, pf_ev, pf_payoff, total_missed = run_portfolio_timeline(
        all_dfs_fast, all_trade_logs, sorted_dates, TRAIN_START_YEAR, ai_params, 
        TRAIN_MAX_POSITIONS, TRAIN_ENABLE_ROTATION, 
        benchmark_ticker="0050", benchmark_data=benchmark_data, is_training=True
    )
    
    if mdd > 45.0: trial.set_user_attr("fail_reason", f"回撤過大 ({mdd:.1f}%)"); return -9999.0
    if t_count < (len(sorted_dates) / 10): trial.set_user_attr("fail_reason", f"次數過低 ({t_count}次)"); return -9999.0
    if ret_pct <= 0: trial.set_user_attr("fail_reason", f"最終虧損 ({ret_pct:.1f}%)"); return -9999.0
    final_romd = (ret_pct ** 0.5) / (mdd + 0.1)
    trial.set_user_attr("pf_return", ret_pct); trial.set_user_attr("pf_mdd", mdd); trial.set_user_attr("pf_trades", t_count); trial.set_user_attr("final_equity", final_eq); trial.set_user_attr("avg_exposure", avg_exp); trial.set_user_attr("bm_return", bm_ret); trial.set_user_attr("bm_mdd", bm_mdd); trial.set_user_attr("win_rate", win_rate); trial.set_user_attr("pf_ev", pf_ev); trial.set_user_attr("pf_payoff", pf_payoff); trial.set_user_attr("missed_buys", total_missed)
    return final_romd

def monitoring_callback(study, trial):
    global CURRENT_SESSION_TRIAL
    CURRENT_SESSION_TRIAL += 1
    duration = trial.duration.total_seconds() if trial.duration else 0.0
    if trial.value and trial.value <= -9000:
        fail_msg = trial.user_attrs.get("fail_reason", "策略無效")
        status_text, score_text = f"{C_YELLOW}淘汰 [{fail_msg}]{C_RESET}", "N/A"
    else:
        status_text, score_text = f"{C_GREEN}進化中{C_RESET}", f"{trial.value:.3f}"
    print(f"\r{C_GRAY}⏳ [累積 {trial.number + 1:>4} | 本輪 {CURRENT_SESSION_TRIAL:>3}/{N_TRIALS}] 耗時: {duration:>5.1f}s | RoMD分數: {score_text:>7} | 狀態: {status_text}{C_RESET}\033[K", end="", flush=True)
    if study.best_trial.number == trial.number and trial.value and trial.value > -9000:
        print(); attrs, p = trial.user_attrs, trial.params
        bb_str = f"啟用 ({p.get('bb_len', 20)} 日, 寬度 {p.get('bb_mult', 2.0):.1f} 倍)" if p.get('use_bb', False) else "[關閉]"
        kc_str = f"啟用 ({p.get('kc_len', 20)} 日, 寬度 {p.get('kc_mult', 2.0):.1f} 倍)" if p.get('use_kc', False) else "[關閉]"
        vol_str = f"啟用 (短 {p.get('vol_short_len', 5)} 日 > 長 {p.get('vol_long_len', 19)} 日)" if p.get('use_vol', False) else "[關閉]"
        
        sys_ret, bm_ret, sys_mdd, bm_mdd = attrs['pf_return'], attrs['bm_return'], attrs['pf_mdd'], attrs['bm_mdd']
        alpha, mdd_diff = sys_ret - bm_ret, bm_mdd - sys_mdd 
        sys_romd_display = (sys_ret / abs(sys_mdd)) if sys_mdd != 0 else 0.0
        bm_romd_display = (bm_ret / abs(bm_mdd)) if bm_mdd != 0 else 0.0
        romd_diff = sys_romd_display - bm_romd_display
        
        alpha_str, alpha_color = f"{'+' if alpha > 0 else ''}{alpha:.2f}%", C_GREEN if alpha > 0 else C_RED
        sys_ret_str, sys_ret_color = f"+{sys_ret:.2f}%", C_GREEN if sys_ret > 0 else C_RED
        bm_ret_str, sys_mdd_str, bm_mdd_str = f"+{bm_ret:.2f}%", f"-{sys_mdd:.2f}%", f"-{bm_mdd:.2f}%"
        mdd_diff_str, mdd_diff_color = f"少跌 {mdd_diff:.2f}%" if mdd_diff > 0 else f"多跌 {abs(mdd_diff):.2f}%", C_GREEN if mdd_diff > 0 else C_RED
        romd_diff_str, romd_diff_color = f"{'+' if romd_diff > 0 else ''}{romd_diff:.2f}", C_GREEN if romd_diff > 0 else C_RED
        
        mode_display = "啟用 (汰弱換強)" if TRAIN_ENABLE_ROTATION else "關閉 (穩定鎖倉)"
        
        print(f"\n{C_RED}🏆 破紀錄！發現更強的投資組合參數！ (累積第 {trial.number + 1} 次測試){C_RESET}")
        print(f"{C_GRAY}--------------------------------------------------------------------------------{C_RESET}")
        print(f"模式: {mode_display} | 最大持股: {TRAIN_MAX_POSITIONS} 檔")
        print(f"總交易紀錄: {attrs['pf_trades']} 筆 (錯失買點: {attrs.get('missed_buys', 0)} 次) | 最終資產: {attrs['final_equity']:,.0f} 元")
        print(f"平均資金水位: {attrs['avg_exposure']:.2f} %")
        print(f"{C_GRAY}--------------------------------------------------------------------------------{C_RESET}")
        print(f"🏆 績效與風險對比表")
        print(f"--------------------------------------------------------------------------------")
        print(f"| 指標項目         | V16 尊爵系統   | 同期大盤 (0050) | 差異 (Alpha)   |")
        print(f"|------------------|----------------|-----------------|----------------|")
        print(f"| 總資產報酬率     | {sys_ret_color}{sys_ret_str:<14}{C_RESET} | {bm_ret_str:<15} | {alpha_color}{alpha_str:<14}{C_RESET} |")
        print(f"| 最大回撤 (MDD)   | {C_YELLOW}{sys_mdd_str:<14}{C_RESET} | {bm_mdd_str:<15} | {mdd_diff_color}{mdd_diff_str:<14}{C_RESET} |")
        print(f"| 報酬回撤比(RoMD) | {C_CYAN}{sys_romd_display:.2f}{' ' * max(0, 14-len(f'{sys_romd_display:.2f}'))}{C_RESET} | {bm_romd_display:<15.2f} | {romd_diff_color}{romd_diff_str:<14}{C_RESET} |")
        print(f"| 系統實戰勝率     | {attrs['win_rate']:>6.2f} %       | -               | -              |")
        print(f"| 盈虧風報比       | {attrs['pf_payoff']:>6.2f}         | -               | -              |")
        print(f"| 實戰期望值(EV)   | {attrs['pf_ev']:>6.2f} R       | -               | -              |")
        print(f"{C_GRAY}--------------------------------------------------------------------------------{C_RESET}")
        print(f"⚙️ [核心參數] 突破: {p['high_len']:>3} 日新高 | ATR 週期: {p['atr_len']:>2} 日 | 半倉停利: {p.get('tp_percent', 0.5)*100:>2.0f} %")
        print(f"              掛單: +{p.get('atr_buy_tol', 1.5):.1f} ATR | 停損: -{p['atr_times_init']:.1f} ATR | 追蹤停利: -{p['atr_times_trail']:.1f} ATR")
        print(f"🛡️ [濾網決策] 布林通道 (BB) : {bb_str}")
        print(f"              阿肯那(KC)    : {kc_str}")
        print(f"              均量 (Vol)    : {vol_str}")
        print(f"{C_CYAN}================================================================================{C_RESET}\n")

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
            
    N_TRIALS = int(input(f"👉 請輸入訓練次數 (預設 50000): ").strip() or 50000)
    study = optuna.create_study(study_name="v16_portfolio_optimization_overnight", storage=DB_NAME, load_if_exists=True, direction="maximize")
    
    if len(study.trials) > 0:
        print(f"\n{C_GREEN}✅ 已累積 {len(study.trials)} 次經驗。{C_RESET}")
        
        # 🌟 接續訓練時：完美重現最後一次突破的最強參數與完整戰情面板
        try:
            best_trial = study.best_trial
            if best_trial.value and best_trial.value > -9000:
                attrs, p = best_trial.user_attrs, best_trial.params
                bb_str = f"啟用 ({p.get('bb_len', 20)} 日, 寬度 {p.get('bb_mult', 2.0):.1f} 倍)" if p.get('use_bb', False) else "[關閉]"
                kc_str = f"啟用 ({p.get('kc_len', 20)} 日, 寬度 {p.get('kc_mult', 2.0):.1f} 倍)" if p.get('use_kc', False) else "[關閉]"
                vol_str = f"啟用 (短 {p.get('vol_short_len', 5)} 日 > 長 {p.get('vol_long_len', 19)} 日)" if p.get('use_vol', False) else "[關閉]"
                
                sys_ret, bm_ret, sys_mdd, bm_mdd = attrs['pf_return'], attrs['bm_return'], attrs['pf_mdd'], attrs['bm_mdd']
                alpha, mdd_diff = sys_ret - bm_ret, bm_mdd - sys_mdd 
                sys_romd_display = (sys_ret / abs(sys_mdd)) if sys_mdd != 0 else 0.0
                bm_romd_display = (bm_ret / abs(bm_mdd)) if bm_mdd != 0 else 0.0
                romd_diff = sys_romd_display - bm_romd_display
                
                alpha_str, alpha_color = f"{'+' if alpha > 0 else ''}{alpha:.2f}%", C_GREEN if alpha > 0 else C_RED
                sys_ret_str, sys_ret_color = f"+{sys_ret:.2f}%", C_GREEN if sys_ret > 0 else C_RED
                bm_ret_str, sys_mdd_str, bm_mdd_str = f"+{bm_ret:.2f}%", f"-{sys_mdd:.2f}%", f"-{bm_mdd:.2f}%"
                mdd_diff_str, mdd_diff_color = f"少跌 {mdd_diff:.2f}%" if mdd_diff > 0 else f"多跌 {abs(mdd_diff):.2f}%", C_GREEN if mdd_diff > 0 else C_RED
                romd_diff_str, romd_diff_color = f"{'+' if romd_diff > 0 else ''}{romd_diff:.2f}", C_GREEN if romd_diff > 0 else C_RED
                
                mode_display = "啟用 (汰弱換強)" if TRAIN_ENABLE_ROTATION else "關閉 (穩定鎖倉)"
                
                print(f"\n{C_CYAN}📜 【歷史突破紀錄還原】{C_RESET}")
                print(f"{C_RED}🏆 目前記憶庫的最強參數！ (來自累積第 {best_trial.number + 1} 次測試){C_RESET}")
                print(f"{C_GRAY}--------------------------------------------------------------------------------{C_RESET}")
                print(f"模式: {mode_display} | 最大持股: {TRAIN_MAX_POSITIONS} 檔")
                print(f"總交易紀錄: {attrs['pf_trades']} 筆 (錯失買點: {attrs.get('missed_buys', 0)} 次) | 最終資產: {attrs['final_equity']:,.0f} 元")
                print(f"平均資金水位: {attrs['avg_exposure']:.2f} %")
                print(f"{C_GRAY}--------------------------------------------------------------------------------{C_RESET}")
                print(f"🏆 績效與風險對比表")
                print(f"--------------------------------------------------------------------------------")
                print(f"| 指標項目         | V16 尊爵系統   | 同期大盤 (0050) | 差異 (Alpha)   |")
                print(f"|------------------|----------------|-----------------|----------------|")
                print(f"| 總資產報酬率     | {sys_ret_color}{sys_ret_str:<14}{C_RESET} | {bm_ret_str:<15} | {alpha_color}{alpha_str:<14}{C_RESET} |")
                print(f"| 最大回撤 (MDD)   | {C_YELLOW}{sys_mdd_str:<14}{C_RESET} | {bm_mdd_str:<15} | {mdd_diff_color}{mdd_diff_str:<14}{C_RESET} |")
                print(f"| 報酬回撤比(RoMD) | {C_CYAN}{sys_romd_display:.2f}{' ' * max(0, 14-len(f'{sys_romd_display:.2f}'))}{C_RESET} | {bm_romd_display:<15.2f} | {romd_diff_color}{romd_diff_str:<14}{C_RESET} |")
                print(f"| 系統實戰勝率     | {attrs['win_rate']:>6.2f} %       | -               | -              |")
                print(f"| 盈虧風報比       | {attrs['pf_payoff']:>6.2f}         | -               | -              |")
                print(f"| 實戰期望值(EV)   | {attrs['pf_ev']:>6.2f} R       | -               | -              |")
                print(f"{C_GRAY}--------------------------------------------------------------------------------{C_RESET}")
                print(f"⚙️ [核心參數] 突破: {p['high_len']:>3} 日新高 | ATR 週期: {p['atr_len']:>2} 日 | 半倉停利: {p.get('tp_percent', 0.5)*100:>2.0f} %")
                print(f"              掛單: +{p.get('atr_buy_tol', 1.5):.1f} ATR | 停損: -{p['atr_times_init']:.1f} ATR | 追蹤停利: -{p['atr_times_trail']:.1f} ATR")
                print(f"🛡️ [濾網決策] 布林通道 (BB) : {bb_str}")
                print(f"              阿肯那(KC)    : {kc_str}")
                print(f"              均量 (Vol)    : {vol_str}")
                print(f"{C_CYAN}================================================================================{C_RESET}\n")
        except ValueError:
            pass

    print(f"\n{C_CYAN}🚀 開始優化...{C_RESET}\n")
    try: study.optimize(objective, n_trials=N_TRIALS, n_jobs=1, callbacks=[monitoring_callback])
    except KeyboardInterrupt: pass
    if study.best_value and study.best_value > -9000:
        with open("models/v16_best_params.json", "w") as f: json.dump(study.best_params, f, indent=4)
        print(f"{C_GREEN}💾 已匯出最強參數！{C_RESET}")