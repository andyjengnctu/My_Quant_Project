import os
import sys
import time
import json
import traceback
import optuna
import numpy as np
import pandas as pd
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

from core.v16_config import V16StrategyParams, SCORE_CALC_METHOD
from core.v16_portfolio_engine import prep_stock_data_and_trades, run_portfolio_timeline, calc_portfolio_score
from core.v16_display import print_strategy_dashboard, C_RED, C_YELLOW, C_CYAN, C_GREEN, C_GRAY, C_RESET
from core.v16_data_utils import sanitize_ohlcv_dataframe, LOAD_DATA_MIN_ROWS
from core.v16_log_utils import write_issue_log, append_issue_log, build_timestamped_log_path

# # (AI註: 收窄 warning 範圍；預設保留 warning，可疑資料與數值問題不要被全域吃掉)
warnings.simplefilter("default")

# # (AI註: 第三方套件的重複性 FutureWarning 只顯示一次，避免訓練輸出被洗版)
warnings.filterwarnings(
    "once",
    category=FutureWarning,
    module=r"optuna(\..*)?$"
)

# # (AI註: 相同 RuntimeWarning 只顯示一次；保留可見性，但避免大量重複輸出)
warnings.filterwarnings(
    "once",
    category=RuntimeWarning
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

TRAIN_MAX_POSITIONS = 10         
TRAIN_START_YEAR = 2015         
TRAIN_ENABLE_ROTATION = False   
DATA_DIR = "tw_stock_data_vip"
RAW_DATA_CACHE = {}             
CURRENT_SESSION_TRIAL, N_TRIALS = 0, 0  
DEFAULT_OPTIMIZER_MAX_WORKERS = min(14, os.cpu_count() or 1)

# # (AI註: 防錯透明化 - 將錯誤摘要落檔，避免 console 訊息在長時間訓練後遺失)
OPTIMIZER_SESSION_TS = time.strftime("%Y%m%d_%H%M%S")
OPTIMIZER_PREP_OTHER_LOG_PATH = build_timestamped_log_path(
    "optimizer_prep_other_failures",
    timestamp=OPTIMIZER_SESSION_TS
)
OPTIMIZER_PREP_SUMMARY = {
    "trials_with_insufficient": 0,
    "trials_with_other": 0,
    "insufficient_count_total": 0,
    "other_count_total": 0,
    "unique_other_count": 0,
}
OPTIMIZER_PREP_OTHER_SEEN = set()


# # (AI註: optimizer 的預處理異常改成 session 級彙總：
# # 1. 資料不足不逐 trial 洗板
# # 2. 其他異常寫入單一 log 檔
# # 3. 相同異常去重，避免長時間訓練下 log 爆量)
def record_optimizer_prep_failures(trial_number, insufficient_failures, other_failures):
    insufficient_count = len(insufficient_failures)
    other_count = len(other_failures)

    if insufficient_count > 0:
        OPTIMIZER_PREP_SUMMARY["trials_with_insufficient"] += 1
        OPTIMIZER_PREP_SUMMARY["insufficient_count_total"] += insufficient_count

    if other_count > 0:
        OPTIMIZER_PREP_SUMMARY["trials_with_other"] += 1
        OPTIMIZER_PREP_SUMMARY["other_count_total"] += other_count

        new_lines = []
        for ticker, reason in other_failures:
            signature = (ticker, reason)
            if signature in OPTIMIZER_PREP_OTHER_SEEN:
                continue
            OPTIMIZER_PREP_OTHER_SEEN.add(signature)
            new_lines.append(f"trial={trial_number + 1} | {ticker}: {reason}")

        if new_lines:
            append_issue_log(OPTIMIZER_PREP_OTHER_LOG_PATH, new_lines)

        OPTIMIZER_PREP_SUMMARY["unique_other_count"] = len(OPTIMIZER_PREP_OTHER_SEEN)


# # (AI註: 訓練結束時再印一次摘要，避免訓練過程被黃色警示干擾)
def print_optimizer_prep_summary():
    insufficient_total = OPTIMIZER_PREP_SUMMARY["insufficient_count_total"]
    other_total = OPTIMIZER_PREP_SUMMARY["other_count_total"]

    if insufficient_total == 0 and other_total == 0:
        return

    print(
        f"{C_YELLOW}⚠️ 本輪預處理摘要："
        f"資料不足 trial={OPTIMIZER_PREP_SUMMARY['trials_with_insufficient']} 次 / 累計 {insufficient_total} 檔；"
        f"其他異常 trial={OPTIMIZER_PREP_SUMMARY['trials_with_other']} 次 / 累計 {other_total} 檔 / "
        f"唯一 {OPTIMIZER_PREP_SUMMARY['unique_other_count']} 筆。{C_RESET}"
    )

    if OPTIMIZER_PREP_SUMMARY["unique_other_count"] > 0:
        print(f"{C_YELLOW}⚠️ 其他異常詳細已寫入: {OPTIMIZER_PREP_OTHER_LOG_PATH}{C_RESET}")

# # (AI註: 防止 magic number 散落；平行度預設沿用原本上限 14，但允許由 params 覆蓋)
def resolve_optimizer_max_workers(params):
    configured = getattr(params, 'optimizer_max_workers', DEFAULT_OPTIMIZER_MAX_WORKERS)
    try:
        configured = int(configured)
    except (TypeError, ValueError):
        configured = DEFAULT_OPTIMIZER_MAX_WORKERS
    return max(1, configured)


# # (AI註: 將 trial 內的「有效資料不足」與真正異常分流，避免 optimizer 在高 high_len 區域反覆洗板)
def is_insufficient_data_message(msg):
    return isinstance(msg, str) and ("有效資料不足" in msg)

def load_all_raw_data():
    if not os.path.exists(DATA_DIR):
        print(f"{C_RED}❌ 嚴重錯誤：找不到資料夾 {DATA_DIR}，請先執行 vip_smart_downloader.py！{C_RESET}")
        sys.exit(1)

    print(f"{C_CYAN}📦 正在將歷史數據載入記憶體快取 (僅需執行一次)...{C_RESET}")
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv')])
    load_issues = []
    total_invalid_rows = 0
    total_duplicate_dates = 0
    total_dropped_rows = 0

    for count, file in enumerate(files, start=1):
        ticker = file.replace('.csv', '').replace('TV_Data_Full_', '')
        try:
            raw_df = pd.read_csv(os.path.join(DATA_DIR, file))
            if len(raw_df) < LOAD_DATA_MIN_ROWS:
                load_issues.append(f"{ticker}: 原始資料列數不足 ({len(raw_df)})")
                continue

            clean_df, sanitize_stats = sanitize_ohlcv_dataframe(raw_df, ticker)
            RAW_DATA_CACHE[ticker] = clean_df

            invalid_row_count = sanitize_stats['invalid_row_count']
            duplicate_date_count = sanitize_stats['duplicate_date_count']
            dropped_row_count = sanitize_stats['dropped_row_count']

            total_invalid_rows += invalid_row_count
            total_duplicate_dates += duplicate_date_count
            total_dropped_rows += dropped_row_count

            if dropped_row_count > 0:
                load_issues.append(
                    f"{ticker}: 清洗移除 {dropped_row_count} 列 "
                    f"(異常OHLCV={invalid_row_count}, 重複日期={duplicate_date_count})"
                )

        except Exception as e:
            load_issues.append(f"{ticker}: {type(e).__name__}: {e}")
            continue

        if count % 50 == 0:
            print(f"{C_GRAY}   進度: 已掃描 {count} 檔股票...{C_RESET}", end="\r")

    print(
        f"\n{C_GREEN}✅ 記憶體快取完成！共載入 {len(RAW_DATA_CACHE)} 檔標的，"
        f"移除 {total_dropped_rows} 列資料 "
        f"(異常OHLCV={total_invalid_rows}, 重複日期={total_duplicate_dates})。{C_RESET}\n"
    )

    if load_issues:
        issue_path = write_issue_log("optimizer_load_issues", load_issues)
        print(f"{C_YELLOW}⚠️ 資料載入/清洗摘要共 {len(load_issues)} 筆，已寫入: {issue_path}{C_RESET}")

def worker_prep_data(ticker, df, params):
    try:
        if len(df) < params.high_len + 10:
            return {
                "ticker": ticker,
                "ok": False,
                "reason": f"有效資料不足: 清洗後僅剩 {len(df)} 列，至少需要 {params.high_len + 10} 列",
                "data": None,
                "logs": None,
            }

        df_prepared, logs = prep_stock_data_and_trades(df, params)
        return {
            "ticker": ticker,
            "ok": True,
            "reason": "",
            "data": df_prepared.to_dict('index'),
            "logs": logs,
        }
    except Exception as e:
        tb_lines = traceback.format_exc().strip().splitlines()
        tb_tail = " | ".join(tb_lines[-3:]) if tb_lines else ""
        return {
            "ticker": ticker,
            "ok": False,
            "reason": f"{type(e).__name__}: {e}" + (f" | Traceback: {tb_tail}" if tb_tail else ""),
            "data": None,
            "logs": None,
        }

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
        min_history_trades = trial.suggest_int("min_history_trades", 0, 5),
        min_history_ev = trial.suggest_float("min_history_ev", -1.0, 0.5, step=0.1),
        min_history_win_rate = trial.suggest_float("min_history_win_rate", 0.0, 0.6, step=0.05),
        use_compounding = True 
    )
    all_dfs_fast, all_trade_logs, master_dates = {}, {}, set()
    prep_failures = []
    max_workers = resolve_optimizer_max_workers(ai_params)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker_prep_data, ticker, df, ai_params) for ticker, df in RAW_DATA_CACHE.items()]
        for future in as_completed(futures):
            result = future.result()
            ticker = result["ticker"]
            if result["ok"]:
                all_dfs_fast[ticker] = result["data"]
                all_trade_logs[ticker] = result["logs"]
                master_dates.update(result["data"].keys())
            else:
                prep_failures.append((ticker, result["reason"]))
    if prep_failures:
        insufficient_failures = [(ticker, reason) for ticker, reason in prep_failures if is_insufficient_data_message(reason)]
        other_failures = [(ticker, reason) for ticker, reason in prep_failures if not is_insufficient_data_message(reason)]

        record_optimizer_prep_failures(
            trial_number=trial.number,
            insufficient_failures=insufficient_failures,
            other_failures=other_failures
        )

        if other_failures:
            trial.set_user_attr("prep_other_failures", len(other_failures))
    if not master_dates: return -9999.0
    sorted_dates = sorted(list(master_dates))
    benchmark_data = all_dfs_fast.get("0050", None)
    
    # 精準 17 個變數解包，支援 max_exp
    ret_pct, mdd, t_count, final_eq, avg_exp, max_exp, bm_ret, bm_mdd, win_rate, pf_ev, pf_payoff, total_missed, total_missed_sells, r_sq, m_win_rate, bm_r_sq, bm_m_win_rate = run_portfolio_timeline(
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
    
    trial.set_user_attr("pf_return", ret_pct); trial.set_user_attr("pf_mdd", mdd); trial.set_user_attr("pf_trades", t_count); trial.set_user_attr("final_equity", final_eq); trial.set_user_attr("avg_exposure", avg_exp); trial.set_user_attr("max_exposure", max_exp); trial.set_user_attr("bm_return", bm_ret); trial.set_user_attr("bm_mdd", bm_mdd); trial.set_user_attr("win_rate", win_rate); trial.set_user_attr("pf_ev", pf_ev); trial.set_user_attr("pf_payoff", pf_payoff); trial.set_user_attr("missed_buys", total_missed); trial.set_user_attr("missed_sells", total_missed_sells)
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
            final_eq=attrs['final_equity'], avg_exp=attrs['avg_exposure'], max_exp=attrs.get('max_exposure', None),
            sys_ret=attrs['pf_return'], bm_ret=attrs['bm_return'], sys_mdd=attrs['pf_mdd'], bm_mdd=attrs['bm_mdd'], 
            win_rate=attrs['win_rate'], payoff=attrs['pf_payoff'], ev=attrs['pf_ev'],
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
                    final_eq=attrs['final_equity'], avg_exp=attrs['avg_exposure'], max_exp=attrs.get('max_exposure', None),
                    sys_ret=attrs['pf_return'], bm_ret=attrs['bm_return'], sys_mdd=attrs['pf_mdd'], bm_mdd=attrs['bm_mdd'], 
                    win_rate=attrs['win_rate'], payoff=attrs['pf_payoff'], ev=attrs['pf_ev'],
                    r_sq=attrs.get('r_squared', 0.0), m_win_rate=attrs.get('m_win_rate', 0.0), bm_r_sq=attrs.get('bm_r_squared', 0.0), bm_m_win_rate=attrs.get('bm_m_win_rate', 0.0)
                )
        except ValueError as e:
            print(f"{C_YELLOW}⚠️ 無法還原歷史最佳參數儀表板: {type(e).__name__}: {e}{C_RESET}")

    if N_TRIALS == 0:
        try:
            if study.best_value and study.best_value > -9000:
                with open("models/v16_best_params.json", "w") as f: json.dump(study.best_params, f, indent=4)
                print(f"\n{C_GREEN}💾 匯出成功！已從記憶庫提取最強參數！{C_RESET}\n")
            else: print(f"\n{C_YELLOW}⚠️ 目前記憶庫中尚無及格的紀錄，無法匯出。{C_RESET}\n")
        except ValueError: print(f"\n{C_YELLOW}⚠️ 記憶庫為空，無法匯出。{C_RESET}\n")
    else:
        print(f"\n{C_CYAN}🚀 開始優化...{C_RESET}\n")
        try:
            study.optimize(objective, n_trials=N_TRIALS, n_jobs=1, callbacks=[monitoring_callback])
        except KeyboardInterrupt:
            print(f"\n{C_YELLOW}⚠️ 使用者中斷訓練流程。{C_RESET}")

        print()
        print_optimizer_prep_summary()
        print(f"\n{C_YELLOW}🛑 訓練階段結束或已中斷。{C_RESET}")