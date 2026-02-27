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

from v16_config import V16StrategyParams
from v16_core import run_v16_backtest

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

C_RED = '\033[91m' 
C_YELLOW = '\033[93m'
C_CYAN = '\033[96m'
C_GREEN = '\033[92m'
C_RESET = '\033[0m'
C_GRAY = '\033[90m'

# ==========================================
# 🌟 AI 機器學習優化器核心設定
# ==========================================
TARGET_MDD_HALF_LIFE = 20.0       
EXPECTED_TRADES_PER_STOCK = 5.0   
MIN_TOTAL_TRADES = 50             
MAX_ALLOWABLE_MDD = 60.0          
# ==========================================

WORKER_CACHE = {}
DISPLAY_MODE = 1  
CURRENT_SESSION_TRIAL = 0  
N_TRIALS = 0  
TRAIN_END_YEAR = None 

DATA_DIR = "tw_stock_data_vip"
TARGET_FILES = []
if os.path.exists(DATA_DIR):
    TARGET_FILES = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.csv')]

def evaluate_single_stock(file_path, params, train_end_year=None):
    try:
        cache_key = f"{file_path}_{train_end_year}" 
        if cache_key not in WORKER_CACHE:
            df = pd.read_csv(file_path)
            
            df.columns = [c.capitalize() for c in df.columns]
            df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].replace(0, np.nan).ffill()
            
            if 'Time' in df.columns:
                df['Time'] = pd.to_datetime(df['Time'])
                df.set_index('Time', inplace=True)
            elif 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)

            if train_end_year is not None:
                df = df[df.index <= pd.to_datetime(f"{train_end_year}-12-31")]
                
            if len(df) < 100:
                WORKER_CACHE[cache_key] = None
                return None
                
            WORKER_CACHE[cache_key] = df
            
        df = WORKER_CACHE[cache_key]
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
        
        # 🌟 實戰派防線設定
        atr_times_init = trial.suggest_float("atr_times_init", 1.5, 3.0, step=0.1),
        atr_times_trail = trial.suggest_float("atr_times_trail", 2.5, 4.5, step=0.1), 
        
        atr_buy_tol = trial.suggest_float("atr_buy_tol", 0.1, 1.5, step=0.1),
        high_len = trial.suggest_int("high_len", 40, 150, step=5),
        
        # 🔥 終極殺手鐧：強迫 AI 必須提早停利！最少賣 30%，最多賣 60%！不准死抱！
        tp_percent = trial.suggest_float("tp_percent", 0.3, 0.6, step=0.05), 
        
        use_bb = ai_use_bb,
        use_kc = ai_use_kc,
        use_vol = ai_use_vol,
        
        bb_len = trial.suggest_int("bb_len", 10, 30, step=2) if ai_use_bb else 20,
        bb_mult = trial.suggest_float("bb_mult", 1.0, 2.5, step=0.1) if ai_use_bb else 2.0,
        
        kc_len = trial.suggest_int("kc_len", 10, 30, step=2) if ai_use_kc else 20,
        kc_mult = trial.suggest_float("kc_mult", 1.5, 3.0, step=0.1) if ai_use_kc else 2.0,
        
        vol_short_len = trial.suggest_int("vol_short_len", 1, 10) if ai_use_vol else 5,
        vol_long_len = trial.suggest_int("vol_long_len", 5, 30) if ai_use_vol else 19 
    )

    all_stats = []
    max_single_mdd = 0.0

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(evaluate_single_stock, fp, ai_params, TRAIN_END_YEAR) for fp in TARGET_FILES]
        for future in as_completed(futures):
            stats = future.result()
            if stats and stats['trade_count'] > 0:
                all_stats.append(stats)
                max_single_mdd = max(max_single_mdd, stats['max_drawdown'])
                
                if max_single_mdd > MAX_ALLOWABLE_MDD:
                    trial.set_user_attr("fail_reason", f"單檔極端回撤 ({max_single_mdd:.1f}%)")
                    for f in futures: f.cancel()
                    return -9999.0

    if len(all_stats) < (len(TARGET_FILES) * 0.05) or not all_stats:
        trial.set_user_attr("fail_reason", "交易過於冷清")
        return -9999.0

    valid_count = len(all_stats)
    total_trades = sum(s['trade_count'] for s in all_stats)
    
    if total_trades < MIN_TOTAL_TRADES:
        trial.set_user_attr("fail_reason", f"樣本數不足 (<{MIN_TOTAL_TRADES}次)")
        return -9999.0
        
    avg_trades = total_trades / valid_count
    avg_ev = sum(s['expected_value'] * s['trade_count'] for s in all_stats) / total_trades
    avg_winrate = sum(s['win_rate'] * s['trade_count'] for s in all_stats) / total_trades
    avg_payoff = sum(s['payoff_ratio'] * s['trade_count'] for s in all_stats) / total_trades
    avg_mdd = sum(s['max_drawdown'] for s in all_stats) / valid_count 

    # =========================================================
    # 🌟 終極平衡計分模型 (高勝率強制引導版)
    # =========================================================
    
    opportunity_factor = math.sqrt(total_trades)
    risk_penalty = TARGET_MDD_HALF_LIFE / (TARGET_MDD_HALF_LIFE + avg_mdd) 
    
    ideal_min_trades = len(TARGET_FILES) * EXPECTED_TRADES_PER_STOCK
    frequency_penalty = min(1.0, total_trades / ideal_min_trades)

    if avg_ev > 0:
        # 🔥 解除 42% 錨定陷阱，改用「50% 基準指數獎勵」
        # 如果勝率 60%，(60/50)^3 = 1.72 倍獎勵！
        # 如果勝率 40%，(40/50)^3 = 0.51 倍懲罰！
        # AI 將無崖可躲，只能乖乖提高勝率來獲取高分！
        win_multiplier = (avg_winrate / 50.0) ** 3
        
        raw_score = avg_ev * win_multiplier * opportunity_factor * risk_penalty 
        final_score = raw_score * frequency_penalty
    else:
        # 虧損時的平滑引導
        raw_score = avg_ev + (avg_winrate / 200.0) - (avg_mdd / 100.0)
        final_score = raw_score

    trial.set_user_attr("win_rate", avg_winrate)
    trial.set_user_attr("ev", avg_ev)
    trial.set_user_attr("trades_total", total_trades) 
    trial.set_user_attr("trades_avg", avg_trades) 
    trial.set_user_attr("mdd", avg_mdd)
    trial.set_user_attr("total_R", total_trades * avg_ev)
    trial.set_user_attr("payoff", avg_payoff)
    trial.set_user_attr("penalty", frequency_penalty)
    trial.set_user_attr("raw_score", raw_score)
        
    return final_score

def monitoring_callback(study, trial):
    global CURRENT_SESSION_TRIAL
    CURRENT_SESSION_TRIAL += 1
    
    duration = trial.duration.total_seconds() if trial.duration else 0.0
    
    if trial.value and trial.value <= -9000:
        fail_msg = trial.user_attrs.get("fail_reason", "方向錯誤")
        status_text = f"{C_YELLOW}淘汰 [{fail_msg}]{C_RESET}"
        score_text = "N/A"
    elif trial.value and trial.value < 0:
        status_text = f"{C_RED}摸索中(虧損){C_RESET}"
        score_text = f"{trial.value:.2f}"
    else:
        status_text = f"{C_GREEN}進化中(獲利){C_RESET}"
        score_text = f"{trial.value:.2f}"
        
    line_text = f"{C_GRAY}⏳ [累積 {trial.number + 1:>4} | 本輪 {CURRENT_SESSION_TRIAL:>3}/{N_TRIALS}] 耗時: {duration:>5.2f} 秒 | 最終分數: {score_text:>7} | 狀態: {status_text}{C_RESET}"
    
    if DISPLAY_MODE == 1:
        print(f"\r{line_text}{' '*20}", end="", flush=True)
    else:
        print(line_text)

    if study.best_trial.number == trial.number and trial.value and trial.value > 0:
        
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

        print(f"\n{C_RED}🏆 破紀錄！發現更強的參數進化！ (累積第 {trial.number + 1} 次測試) | 最終實戰評分: {trial.value:.2f}{C_RESET}")
        print(f"   📊 [全市場平均] 勝率: {attrs['win_rate']:>5.2f}% | 真實期望值(EV): {attrs['ev']:>5.2f} R | 盈虧比參考: {attrs['payoff']:>4.2f}")
        print(f"   📈 [交易頻率]   總交易: {attrs['trades_total']}次 | 單檔平均: {attrs['trades_avg']:.1f}次 | {penalty_str}")
        print(f"   💰 [獲利與風險] 總累積獲利: {attrs['total_R']:>7.1f} R | 平均回撤: {attrs['mdd']:>5.2f}% ")
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
    print(f"⚙️ {C_YELLOW}V16 全市場極速 AI 訓練引擎 (強制停利高勝率版){C_RESET}")
    print(f"{C_CYAN}================================================================================{C_RESET}")
    
    print(f"{C_GREEN}✅ 已找到 {len(TARGET_FILES)} 檔股票資料。{C_RESET}")
    
    db_file_name = "v16_ai_memory.db"
    DB_NAME = f"sqlite:///{db_file_name}"
    STUDY_NAME = "v16_global_optimization"

    # 🌟 時光眼罩設定
    ans_year = input(f"\n👉 請設定【AI 訓練截止年份】\n(例如輸入 2020，AI 將完全看不到 2021 年以後的資料。直接按 Enter 預設為看全歷史): ").strip()
    if ans_year.isdigit():
        TRAIN_END_YEAR = int(ans_year)
        print(f"{C_RED}⚠️ 注意：AI 大腦已被蒙上眼罩，只能使用 【{TRAIN_END_YEAR} 年 12 月 31 日】 以前的數據進行訓練！{C_RESET}")
    else:
        print(f"{C_GRAY}📖 AI 將使用全部歷史數據進行訓練。{C_RESET}")

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
    else:
        print(f"\n{C_YELLOW}🌟 全新記憶庫已建立！準備展開未知的探索...{C_RESET}")
    
    print(f"\n{C_CYAN}🚀 正在全速推進 {N_TRIALS} 次全市場進化...{C_RESET}\n")
    
    start_time = time.time()
    
    try:
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=1, callbacks=[monitoring_callback])
    except KeyboardInterrupt:
        print(f"\n\n{C_YELLOW}⚠️ 偵測到使用者強制中斷 (Ctrl+C)！進度已安全保留在資料庫中。{C_RESET}")
        sys.exit(0)
    
    end_time = time.time()
    
    print(f"\n{C_YELLOW}================================================================================{C_RESET}")
    print(f"🏁 訓練結束！總耗時: {end_time - start_time:.2f} 秒。")
    
    try:
        if study.best_value and study.best_value > 0:
            print(f"✨ AI 記憶庫中目前的最強實戰評分: {study.best_value:.2f}")
            with open("v16_best_params.json", "w") as f:
                json.dump(study.best_params, f, indent=4)
            print(f"{C_GREEN}💾 已成功將最強參數匯出至 'v16_best_params.json'！{C_RESET}")
        else:
            print(f"{C_GRAY}⚠️ 目前記憶庫中尚無獲利的參數，無法匯出 JSON。請增加訓練次數。{C_RESET}")
    except ValueError:
        print(f"{C_GRAY}⚠️ 記憶庫尚無完成的紀錄。{C_RESET}")