import os
import sys
import time
import math
import json
import optuna
import numpy as np
import pandas as pd
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

from core.v16_config import V16StrategyParams
from core.v16_core import run_v16_backtest

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

C_RED = '\033[91m' 
C_YELLOW = '\033[93m'
C_CYAN = '\033[96m'
C_GREEN = '\033[92m'
C_RESET = '\033[0m'
C_GRAY = '\033[90m'

WORKER_CACHE = {}
DISPLAY_MODE = 1  
CURRENT_SESSION_TRIAL = 0  
N_TRIALS = 0  

# ==========================================
DATA_DIR = "tw_stock_data_vip"
TARGET_FILES = []
if os.path.exists(DATA_DIR):
    TARGET_FILES = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
# ==========================================

def evaluate_single_stock(file_path, params):
    try:
        if file_path not in WORKER_CACHE:
            df = pd.read_csv(file_path)
            if len(df) < 100:
                WORKER_CACHE[file_path] = None
                return None
                
            df.columns = [c.capitalize() for c in df.columns]
            df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].replace(0, np.nan).ffill()
            
            if 'Time' in df.columns:
                df['Time'] = pd.to_datetime(df['Time'])
                df.set_index('Time', inplace=True)
            elif 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                
            WORKER_CACHE[file_path] = df
            
        df = WORKER_CACHE[file_path]
        if df is None: return None
            
        return run_v16_backtest(df, params)
    except Exception:
        return None

def objective(trial): 
    ai_use_bb = trial.suggest_categorical("use_bb", [True, False])
    ai_use_kc = trial.suggest_categorical("use_kc", [True, False])
    ai_use_vol = trial.suggest_categorical("use_vol", [True, False]) 

    ai_params = V16StrategyParams(
        atr_len = trial.suggest_int("atr_len", 5, 20),
        atr_times_init = trial.suggest_float("atr_times_init", 1.0, 3.5, step=0.1),
        atr_times_trail = trial.suggest_float("atr_times_trail", 2.0, 4.5, step=0.1), 
        atr_buy_tol = trial.suggest_float("atr_buy_tol", 0.1, 1.5, step=0.1),
        high_len = trial.suggest_int("high_len", 40, 150, step=5),
        tp_percent = trial.suggest_float("tp_percent", 0.0, 0.6, step=0.01), 
        
        use_bb = ai_use_bb,
        use_kc = ai_use_kc,
        use_vol = ai_use_vol,
        
        bb_len = trial.suggest_int("bb_len", 10, 30, step=2) if ai_use_bb else 20,
        bb_mult = trial.suggest_float("bb_mult", 1.0, 2.5, step=0.1) if ai_use_bb else 2.0,
        
        kc_len = trial.suggest_int("kc_len", 10, 30, step=2) if ai_use_kc else 20,
        kc_mult = trial.suggest_float("kc_mult", 1.5, 3.0, step=0.1) if ai_use_kc else 2.0,
        
        vol_short_len = trial.suggest_int("vol_short_len", 1, 10) if ai_use_vol else 5,
        vol_long_len = trial.suggest_int("vol_long_len", 10, 30) if ai_use_vol else 19,
        
        # 🌟 核心環境控制：強制關閉複利，讓 AI 跑出純淨無瑕的「單利真金白銀」
        use_compounding = False # true為考慮複利、False是單利
    )

    all_stats = []
    max_single_mdd = 0.0

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(evaluate_single_stock, fp, ai_params) for fp in TARGET_FILES]
        for future in as_completed(futures):
            stats = future.result()
            if stats and stats['trade_count'] > 0:
                all_stats.append(stats)
                max_single_mdd = max(max_single_mdd, stats['max_drawdown'])
                
                if max_single_mdd > 85.0:
                    trial.set_user_attr("fail_reason", f"極端風險 ({max_single_mdd:.1f}%)")
                    for f in futures: f.cancel()
                    return -9999.0

    if len(all_stats) < (len(TARGET_FILES) * 0.05) or not all_stats:
        trial.set_user_attr("fail_reason", "無效交易參數")
        return -9999.0

    valid_count = len(all_stats)
    total_trades = sum(s['trade_count'] for s in all_stats)
    
    if total_trades == 0:
        trial.set_user_attr("fail_reason", "交易次數為 0")
        return -9999.0
        
    avg_trades = total_trades / valid_count
    avg_ev = sum(s['expected_value'] * s['trade_count'] for s in all_stats) / total_trades
    avg_winrate = sum(s['win_rate'] * s['trade_count'] for s in all_stats) / total_trades
    avg_payoff = sum(s['payoff_ratio'] * s['trade_count'] for s in all_stats) / total_trades
    avg_mdd = sum(s['max_drawdown'] for s in all_stats) / valid_count 
    total_R_value = total_trades * avg_ev 
    avg_growth = sum(s['asset_growth'] for s in all_stats) / valid_count 
    global_asset_growth = sum(s['asset_growth'] for s in all_stats) / valid_count # orignal / len(TARGET_FILES)
    global_net_profit = sum(s.get('net_profit_value', 0) for s in all_stats)    

    # 🌟 新增：計算全市場的「平均單筆持倉天數」
    avg_holding_days = sum(s['avg_bars_held'] * s['trade_count'] for s in all_stats) / total_trades

    ideal_min_trades = len(TARGET_FILES) * 8
    trade_penalty = min(1.0, total_trades / ideal_min_trades)

    # 詳細記錄淘汰原因
    if avg_mdd > 35.0: 
        trial.set_user_attr("fail_reason", f"回撤過大 ({avg_mdd:.1f}%)")
        return -9999.0
    if avg_ev <= 0: 
        trial.set_user_attr("fail_reason", f"期望值為負 ({avg_ev:.2f}R)")
        return -9999.0
    if avg_winrate < 35.0: 
        trial.set_user_attr("fail_reason", f"勝率過低 ({avg_winrate:.1f}%)")
        return -9999.0
    if avg_growth <= 0.0: 
        trial.set_user_attr("fail_reason", f"沒賺到真金白銀 ({avg_growth:.1f}%)")
        return -9999.0
    
    # =========================================================
    # 🌟 實戰派公式 2.0：RoMD × 期望值 ÷ √(平均持倉天數)
    # =========================================================


    # trade_impact = total_trades ** 1.0
    # ev_impact = avg_ev ** 1.0
    # win_rate_impact = (avg_winrate / 50) ** 1.0 
    # growth_impact = min(1.0, (avg_growth / 10) ** 1.0)
    # mmd_impact = avg_mdd ** 1.0
    # time_impact = avg_holding_days ** 0.5
    # raw_score = (trade_impact * ev_impact * win_rate_impact * growth_impact) / ((mmd_impact + 1.0) * (time_impact+1)) * 1
    # final_score = raw_score * trade_penalty  

    # time_penalty_divisor = math.sqrt(avg_holding_days + 1.0)
    # raw_score = (avg_growth * avg_ev * win_rate_impact) / (avg_mdd + 1.0)    
    # final_score = (raw_score / time_penalty_divisor) * trade_penalty * 10000

    # ===========================================
    # 原本最精簡的
    # ==========================================
    raw_score = total_R_value / (avg_mdd + 1.0)
    final_score = raw_score * trade_penalty 

    # ===========================================
    # Testing
    # ==========================================
    # raw_score = (global_asset_growth) / (max_single_mdd ** 2 + 1.0) * 10000
    # final_score = raw_score * trade_penalty 

    # ===========================================
    # Testing
    # ==========================================
    # raw_score = (global_net_profit ** 0.5) / (avg_mdd + 1.0)
    # final_score = raw_score * trade_penalty 

    # 寫入 Trial 供顯示
    trial.set_user_attr("win_rate", avg_winrate)
    trial.set_user_attr("ev", avg_ev)
    trial.set_user_attr("trades_total", total_trades) 
    trial.set_user_attr("trades_avg", avg_trades) 
    trial.set_user_attr("mdd", avg_mdd)
    trial.set_user_attr("max_mdd", max_single_mdd)  # 🌟 新增：紀錄全市場最慘回撤
    trial.set_user_attr("total_R", total_R_value)
    trial.set_user_attr("growth", global_asset_growth) 
    trial.set_user_attr("payoff", avg_payoff)
    trial.set_user_attr("penalty", trade_penalty)
    trial.set_user_attr("raw_score", raw_score)
    trial.set_user_attr("hold_days", avg_holding_days) # 🌟 記錄平均持倉天數
        
    return final_score

def monitoring_callback(study, trial):
    global CURRENT_SESSION_TRIAL
    CURRENT_SESSION_TRIAL += 1
    
    duration = trial.duration.total_seconds() if trial.duration else 0.0
    
    if trial.value and trial.value <= -9000:
        fail_msg = trial.user_attrs.get("fail_reason", "方向錯誤")
        status_text = f"{C_YELLOW}淘汰 [{fail_msg}]{C_RESET}"
        score_text = "N/A"
    else:
        status_text = f"{C_GREEN}進化中{C_RESET}"
        score_text = f"{trial.value:.2f}"
        
    line_text = f"{C_GRAY}⏳ [累積 {trial.number + 1:>4} | 本輪 {CURRENT_SESSION_TRIAL:>3}/{N_TRIALS}] 耗時: {duration:>5.2f} 秒 | 最終分數: {score_text:>7} | 狀態: {status_text}{C_RESET}"
    
    if DISPLAY_MODE == 1:
        print(f"\r{line_text}{' '*20}", end="", flush=True)
    else:
        print(line_text)

    if study.best_trial.number == trial.number and trial.value and trial.value > -9000:
        
        if DISPLAY_MODE == 1:
            print() 
            
        attrs = trial.user_attrs
        p = trial.params
        
        bb_str = f"啟用 ({p.get('bb_len', 20)} 日, 寬度 {p.get('bb_mult', 2.0):.1f} 倍)" if p.get('use_bb', False) else "[關閉]"
        kc_str = f"啟用 ({p.get('kc_len', 20)} 日, 寬度 {p.get('kc_mult', 2.0):.1f} 倍)" if p.get('use_kc', False) else "[關閉]"
        vol_str = f"啟用 (短 {p.get('vol_short_len', 5)} 日 > 長 {p.get('vol_long_len', 19)} 日)" if p.get('use_vol', False) else "[關閉]"
        
        penalty_val = attrs.get('penalty', 1.0)
        if penalty_val < 1.0:
            penalty_str = f"{C_YELLOW}⚠️ 頻率過低懲罰 (分數打 {penalty_val*100:.1f} 折){C_RESET}"
        else:
            penalty_str = f"{C_GREEN}✅ 交易頻率達標 (無懲罰){C_RESET}"

        print(f"\n{C_RED}🏆 破紀錄！發現更強的參數進化！ (累積第 {trial.number + 1} 次測試) | 最終資金效率: {trial.value:.2f}{C_RESET}")
        print(f"   📊 [全市場平均] 勝率: {attrs['win_rate']:>5.2f}% | 期望值: {attrs['ev']:>5.2f}R | 風報比: {attrs['payoff']:>4.2f}")
        # 🌟 印出平均持倉天數，讓您感受資金效率的變化
        print(f"   📈 [交易頻率]   總交易: {attrs['trades_total']}次 | 單檔平均: {attrs['trades_avg']:.1f}次 | 平均抱牢: {attrs.get('hold_days', 0):.1f} 天 | {penalty_str}")
        print(f"   💰 [獲利與風險] 全市場平均報酬: {attrs.get('growth', 0):>6.1f}% | 平均回撤: {attrs['mdd']:>5.2f}% | 最大回撤: {attrs.get('max_mdd', 0):>5.2f}% | 總R: {attrs.get('total_R', 0):.1f}")
        print(f"   ⚙️ [核心參數] 突破: {p['high_len']:>3} 日新高 | ATR 週期: {p['atr_len']:>2} 日 | 半倉停利: {p.get('tp_percent', 0.5)*100:>2.0f} %")
        print(f"                 掛單: +{p['atr_buy_tol']:.1f} ATR | 停損: -{p['atr_times_init']:.1f} ATR | 追蹤停利: -{p['atr_times_trail']:.1f} ATR")
        print(f"   🛡️ [濾網決策] 布林通道 (BB) : {bb_str}")
        print(f"                 阿肯那通道(KC): {kc_str}")
        print(f"                 均量爆發 (Vol): {vol_str}")
        print(f"{C_CYAN}--------------------------------------------------------------------------------{C_RESET}\n")

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        print("❌ 找不到資料夾，請確認路徑。")
        exit()
        
    if len(TARGET_FILES) == 0:
        print("📂 資料夾內無 CSV 檔案。")
        exit()

    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(f"⚙️ {C_YELLOW}V16 全市場極速 AI 訓練引擎 (法人級 RoMD + 時間平方根折現版){C_RESET}")
    print(f"{C_CYAN}================================================================================{C_RESET}")
    
    print(f"{C_GREEN}✅ 已找到 {len(TARGET_FILES)} 檔股票資料。{C_RESET}")
    
    db_file_name = "v16_ai_memory.db"
    DB_NAME = f"sqlite:///{db_file_name}"
    STUDY_NAME = "v16_global_optimization"

    if os.path.exists(db_file_name):
        print(f"\n{C_YELLOW}⚠️ 系統偵測到已存在的 AI 記憶庫！{C_RESET}")
        choice = input(f"👉 請選擇： [1] 繼承記憶，接續訓練  [2] 刪除記憶，重頭來過 (預設 1): ").strip()
        
        if choice == '2':
            os.remove(db_file_name)
            print(f"{C_RED}🗑️ 已刪除舊有記憶，準備展開新的進化框架！{C_RESET}")
        else:
            print(f"{C_GREEN}🧠 選擇接續訓練，準備喚醒前次大腦...{C_RESET}")
    else:
        print(f"\n{C_GRAY}🧠 未發現舊有記憶，將建立全新 AI 大腦...{C_RESET}")

    display_input = input(f"👉 請選擇顯示模式： [1] 簡潔模式(預設，不洗版)  [2] 詳細模式(顯示每筆): ").strip()
    DISPLAY_MODE = 2 if display_input == '2' else 1

    trials_input = input(f"👉 請輸入本次要進行的 AI 訓練次數 (直接按 Enter 預設為 100,000): ").strip()
    N_TRIALS = int(trials_input) if trials_input.isdigit() else 100000
    
    study = optuna.create_study(
        study_name=STUDY_NAME,               
        storage=DB_NAME,                     
        load_if_exists=True,                 
        direction="maximize", 
        sampler=optuna.samplers.TPESampler()        
    )
    
    completed_trials = len(study.trials)
    if completed_trials > 0:
        print(f"\n{C_GREEN}✅ 記憶庫載入成功！大腦已累積 {completed_trials} 次歷史經驗。將從第 {completed_trials + 1} 次繼續探索！{C_RESET}")
        
        try:
            best_trial = study.best_trial
            if best_trial.value and best_trial.value > -9000:
                attrs = best_trial.user_attrs
                p = best_trial.params
                
                bb_str = f"啟用 ({p.get('bb_len', 20)} 日, 寬度 {p.get('bb_mult', 2.0):.1f} 倍)" if p.get('use_bb', False) else "[關閉]"
                kc_str = f"啟用 ({p.get('kc_len', 20)} 日, 寬度 {p.get('kc_mult', 2.0):.1f} 倍)" if p.get('use_kc', False) else "[關閉]"
                vol_str = f"啟用 (短 {p.get('vol_short_len', 5)} 日 > 長 {p.get('vol_long_len', 19)} 日)" if p.get('use_vol', False) else "[關閉]"
                
                penalty_val = attrs.get('penalty', 1.0)
                if penalty_val < 1.0:
                    penalty_str = f"{C_YELLOW}⚠️ 頻率過低懲罰 (分數打 {penalty_val*100:.1f} 折){C_RESET}"
                else:
                    penalty_str = f"{C_GREEN}✅ 交易頻率達標 (無懲罰){C_RESET}"

                print(f"\n{C_YELLOW}🏆 【目前記憶庫中的最強聖杯紀錄】 | 最終資金效率: {best_trial.value:.2f}{C_RESET}")
                print(f"   📊 [全市場平均] 勝率: {attrs.get('win_rate', 0):>5.2f}% | 期望值: {attrs.get('ev', 0):>5.2f}R | 風報比: {attrs.get('payoff', 0):>4.2f}")
                print(f"   📈 [交易頻率]   總交易: {attrs.get('trades_total', 0)}次 | 單檔平均: {attrs.get('trades_avg', 0):.1f}次 | 平均抱牢: {attrs.get('hold_days', 0):.1f} 天 | {penalty_str}")
                print(f"   💰 [獲利與風險] 全市場平均報酬: {attrs.get('growth', 0):>6.1f}% | 平均回撤: {attrs.get('mdd', 0):>5.2f}% | 最大回撤: {attrs.get('max_mdd', 0):>5.2f}% | 總R: {attrs.get('total_R', 0):.1f}")
                print(f"   ⚙️ [核心參數] 突破: {p.get('high_len', 0):>3} 日新高 | ATR 週期: {p.get('atr_len', 0):>2} 日 | 半倉停利: {p.get('tp_percent', 0.5)*100:>2.0f} %")
                print(f"                 掛單: +{p.get('atr_buy_tol', 0):.1f} ATR | 停損: -{p.get('atr_times_init', 0):.1f} ATR | 追蹤停利: -{p.get('atr_times_trail', 0):.1f} ATR")
                print(f"   🛡️ [濾網決策] 布林通道 (BB) : {bb_str}")
                print(f"                 阿肯那通道(KC): {kc_str}")
                print(f"                 均量爆發 (Vol): {vol_str}")
                print(f"{C_CYAN}--------------------------------------------------------------------------------{C_RESET}")
            else:
                print(f"{C_GRAY}   (目前歷史紀錄中尚未找到符合及格線的參數組合，AI 將繼續努力...){C_RESET}")
        except Exception:
            pass
            
    else:
        print(f"\n{C_YELLOW}🌟 全新記憶庫已建立！準備展開未知的探索...{C_RESET}")
    
    print(f"\n{C_CYAN}🚀 正在全速推進 {N_TRIALS} 次全市場進化...{C_RESET}\n")
    
    start_time = time.time()
    
    try:
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=1, callbacks=[monitoring_callback])
    except KeyboardInterrupt:
        print(f"\n\n{C_YELLOW}⚠️ 偵測到使用者強制中斷 (Ctrl+C)！進度已安全保留在資料庫中。{C_RESET}")
        print(f"{C_GRAY}💡 提示：若需匯出並覆寫 JSON，請重新執行程式並將訓練次數設為 0。{C_RESET}")
        sys.exit(0)
    
    end_time = time.time()
    
    print(f"\n{C_YELLOW}================================================================================{C_RESET}")
    print(f"🏁 訓練結束！總耗時: {end_time - start_time:.2f} 秒。")
    
    try:
        if study.best_value and study.best_value > -9000:
            print(f"✨ AI 記憶庫中目前的最強資金效率分數: {study.best_value:.2f}")
            with open("models/v16_best_params.json", "w") as f:
                json.dump(study.best_params, f, indent=4)
            print(f"{C_GREEN}💾 已成功將最強參數匯出至 'models/v16_best_params.json'！{C_RESET}")
        else:
            print(f"{C_GRAY}⚠️ 目前記憶庫中尚無及格的參數，無法匯出 JSON。{C_RESET}")
    except ValueError:
        print(f"{C_GRAY}⚠️ 記憶庫尚無完成的紀錄。{C_RESET}")