import os
import time
import json
import traceback
import csv
import optuna
import pandas as pd
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

from core.v16_config import (
    V16StrategyParams,
    MIN_ANNUAL_TRADES, MIN_BUY_FILL_RATE,
    MIN_TRADE_WIN_RATE, MIN_FULL_YEAR_RETURN_PCT,
    MAX_PORTFOLIO_MDD_PCT, MIN_MONTHLY_WIN_RATE, MIN_EQUITY_CURVE_R_SQUARED,
)
from core.v16_portfolio_engine import prep_stock_data_and_trades, pack_prepared_stock_data, get_fast_dates, run_portfolio_timeline, calc_portfolio_score
from core.v16_display import print_strategy_dashboard, C_RED, C_YELLOW, C_CYAN, C_GREEN, C_GRAY, C_RESET
from core.v16_data_utils import (
    sanitize_ohlcv_dataframe,
    get_required_min_rows,
    get_required_min_rows_from_high_len,
    discover_unique_csv_inputs,
)
from core.v16_log_utils import write_issue_log, format_exception_summary
from core.v16_params_io import build_params_from_mapping, params_to_json_dict

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

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "tw_stock_data_vip")
BEST_PARAMS_PATH = os.path.join(MODELS_DIR, "v16_best_params.json")
DB_FILE_PATH = os.path.join(MODELS_DIR, "v16_portfolio_ai_10pos_overnight.db")


# # (AI註: 將目錄建立延後到實際執行期，避免被 import 時污染呼叫端工作目錄)
def ensure_runtime_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


TRAIN_MAX_POSITIONS = 10         
TRAIN_START_YEAR = 2015         
TRAIN_ENABLE_ROTATION = False   
RAW_DATA_CACHE = {}             
CURRENT_SESSION_TRIAL, N_TRIALS = 0, 0  
# # (AI註: Windows / Python 3.14 下 ProcessPool + 大量 dataframe IPC 容易不穩，先保守降平行度)
DEFAULT_OPTIMIZER_MAX_WORKERS = min(6, max(1, (os.cpu_count() or 1) // 2))

# # (AI註: 預處理摘要只保留「資料不足」統計；其他非預期錯誤已改成直接 raise，不再保留死邏輯分支)
OPTIMIZER_SESSION_TS = time.strftime("%Y%m%d_%H%M%S")
OPTIMIZER_PREP_SUMMARY = {
    "trials_with_insufficient": 0,
    "insufficient_count_total": 0,
}

# # (AI註: profiling 版預設開啟 objective 分段計時；只量測，不改交易邏輯)
ENABLE_OPTIMIZER_PROFILING = True
ENABLE_PROFILE_CONSOLE_PRINT = False
PROFILE_PRINT_EVERY_N_TRIALS = 1
PROFILE_CSV_PATH = os.path.join(OUTPUT_DIR, f"optimizer_profile_{OPTIMIZER_SESSION_TS}.csv")
PROFILE_SUMMARY_PATH = os.path.join(OUTPUT_DIR, f"optimizer_profile_summary_{OPTIMIZER_SESSION_TS}.json")
PROFILE_ROWS = []

# # (AI註: 問題6 - optimizer 搜尋空間與資料最低長度共用單一常數，避免 raw cache 與 trial worker 各自為政)
OPTIMIZER_HIGH_LEN_MIN = 40
OPTIMIZER_HIGH_LEN_MAX = 250
OPTIMIZER_HIGH_LEN_STEP = 5
OPTIMIZER_REQUIRED_MIN_ROWS = get_required_min_rows_from_high_len(OPTIMIZER_HIGH_LEN_MAX)

# # (AI註: 方案B - None=照常搜尋 tp_percent；數值=暫時固定該 tp_percent 並跳過搜尋)
OPTIMIZER_FIXED_TP_PERCENT = None

PROFILE_FIELDS = [
    "trial_number", "objective_wall_sec", "prep_wall_sec", "prep_worker_total_sum_sec",
    "prep_worker_copy_sum_sec", "prep_worker_generate_signals_sum_sec", "prep_worker_assign_sum_sec",
    "prep_worker_run_backtest_sum_sec", "prep_worker_to_dict_sum_sec", "prep_ok_count",
    "prep_fail_count", "prep_avg_per_ok_sec", "sort_dates_sec", "portfolio_wall_sec",
    "portfolio_total_sec", "portfolio_ticker_dates_sec", "portfolio_build_trade_index_sec",
    "portfolio_day_loop_sec", "portfolio_candidate_scan_sec", "portfolio_rotation_sec",
    "portfolio_settle_sec", "portfolio_buy_sec", "portfolio_equity_mark_sec",
    "portfolio_closeout_sec", "portfolio_curve_stats_sec", "filter_rules_sec",
    "score_calc_sec", "ret_pct", "mdd", "trade_count",
    "annual_return_pct", "annual_trades", "reserved_buy_fill_rate",
    "full_year_count", "min_full_year_return_pct",
    "m_win_rate", "r_squared",
    "base_score", "trial_value", "fail_reason"
]


# # (AI註: optimizer 的預處理摘要只保留「資料不足」統計；其他非預期錯誤已改成直接 raise)
def record_optimizer_prep_failures(trial_number, insufficient_failures):
    insufficient_count = len(insufficient_failures)

    if insufficient_count > 0:
        OPTIMIZER_PREP_SUMMARY["trials_with_insufficient"] += 1
        OPTIMIZER_PREP_SUMMARY["insufficient_count_total"] += insufficient_count


# # (AI註: 訓練結束時再印一次資料不足摘要，避免訓練過程被重複警示干擾)
def validate_optimizer_param_overrides(param_mapping):
    if not isinstance(param_mapping, dict):
        raise TypeError(f"optimizer override 必須是 dict，收到 {type(param_mapping).__name__}")

    base_payload = params_to_json_dict(V16StrategyParams())
    merged_payload = dict(base_payload)
    merged_payload.update(param_mapping)
    validated = build_params_from_mapping(merged_payload)
    return {key: getattr(validated, key) for key in param_mapping}


# # (AI註: 方案B固定值也必須共用參數 guardrail，避免直接繞過 JSON 載入驗證)
def resolve_optimizer_tp_percent(trial):
    if OPTIMIZER_FIXED_TP_PERCENT is None:
        return trial.suggest_float("tp_percent", 0.0, 0.6, step=0.01)

    fixed_tp_percent = float(OPTIMIZER_FIXED_TP_PERCENT)
    validated_params = validate_optimizer_param_overrides({"tp_percent": fixed_tp_percent})
    fixed_tp_percent = float(validated_params["tp_percent"])
    trial.set_user_attr("fixed_tp_percent", fixed_tp_percent)
    return fixed_tp_percent


# # (AI註: 單一真理來源 - 顯示/匯出最佳參數時統一補回固定模式缺少的 tp_percent，並套用同一組 guardrail)
def build_optimizer_trial_params(param_mapping, user_attrs=None):
    resolved_params = dict(param_mapping)
    if "tp_percent" in resolved_params:
        resolved_params["tp_percent"] = float(resolved_params["tp_percent"])
    else:
        fixed_tp_percent = None if user_attrs is None else user_attrs.get("fixed_tp_percent")
        if fixed_tp_percent is None:
            fixed_tp_percent = OPTIMIZER_FIXED_TP_PERCENT
        if fixed_tp_percent is None:
            raise ValueError("最佳 trial 缺少 tp_percent，且目前未設定 OPTIMIZER_FIXED_TP_PERCENT，無法還原完整參數。")
        resolved_params["tp_percent"] = float(fixed_tp_percent)

    validated_params = validate_optimizer_param_overrides(resolved_params)
    return {key: validated_params[key] for key in resolved_params}


def build_best_params_payload_from_trial(best_trial):
    resolved_params = build_optimizer_trial_params(best_trial.params, best_trial.user_attrs)
    base_payload = params_to_json_dict(V16StrategyParams())
    base_payload.update(resolved_params)
    return params_to_json_dict(build_params_from_mapping(base_payload))


def print_optimizer_prep_summary():
    insufficient_total = OPTIMIZER_PREP_SUMMARY["insufficient_count_total"]

    if insufficient_total == 0:
        return

    print(
        f"{C_YELLOW}⚠️ 本輪預處理摘要："
        f"資料不足 trial={OPTIMIZER_PREP_SUMMARY['trials_with_insufficient']} 次 / 累計 {insufficient_total} 檔。{C_RESET}"
    )

def init_profile_output_files():
    if not ENABLE_OPTIMIZER_PROFILING:
        return
    ensure_runtime_dirs()
    with open(PROFILE_CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=PROFILE_FIELDS)
        writer.writeheader()


def append_profile_row(row):
    if not ENABLE_OPTIMIZER_PROFILING:
        return

    normalized = {field: row.get(field, "") for field in PROFILE_FIELDS}
    PROFILE_ROWS.append(normalized)
    with open(PROFILE_CSV_PATH, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=PROFILE_FIELDS)
        writer.writerow(normalized)


def print_profile_summary():
    if not ENABLE_OPTIMIZER_PROFILING or not PROFILE_ROWS:
        return

    numeric_fields = [
        "objective_wall_sec", "prep_wall_sec", "prep_worker_total_sum_sec",
        "prep_worker_generate_signals_sum_sec", "prep_worker_run_backtest_sum_sec",
        "prep_worker_to_dict_sum_sec", "prep_avg_per_ok_sec", "sort_dates_sec",
        "portfolio_wall_sec", "portfolio_total_sec", "portfolio_ticker_dates_sec",
        "portfolio_build_trade_index_sec", "portfolio_day_loop_sec",
        "portfolio_candidate_scan_sec", "portfolio_rotation_sec", "portfolio_settle_sec",
        "portfolio_buy_sec", "portfolio_equity_mark_sec", "portfolio_closeout_sec",
        "portfolio_curve_stats_sec", "filter_rules_sec", "score_calc_sec"
    ]
    summary = {
        "trial_count": len(PROFILE_ROWS),
        "avg": {}
    }

    for field in numeric_fields:
        vals = []
        for row_idx, row in enumerate(PROFILE_ROWS):
            val = row.get(field, "")
            if isinstance(val, (int, float)):
                vals.append(float(val))
            elif isinstance(val, str) and val not in ("", None):
                try:
                    vals.append(float(val))
                except ValueError as e:
                    raise ValueError(
                        f"PROFILE_ROWS[{row_idx}]['{field}'] 無法轉成 float: {val!r}"
                    ) from e
        summary["avg"][field] = (sum(vals) / len(vals)) if vals else 0.0

    with open(PROFILE_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    avg = summary["avg"]
    print(f"{C_CYAN}📊 Profiling 平均摘要（{summary['trial_count']} trials）:{C_RESET}")
    print(
        f"{C_GRAY}   objective={avg['objective_wall_sec']:.3f}s | "
        f"prep_wall={avg['prep_wall_sec']:.3f}s | portfolio_wall={avg['portfolio_wall_sec']:.3f}s | "
        f"worker_generate_sum={avg['prep_worker_generate_signals_sum_sec']:.3f}s | "
        f"worker_backtest_sum={avg['prep_worker_run_backtest_sum_sec']:.3f}s | "
        f"to_dict_sum={avg['prep_worker_to_dict_sum_sec']:.3f}s | "
        f"pf_day_loop={avg['portfolio_day_loop_sec']:.3f}s{C_RESET}"
    )
    print(f"{C_GRAY}   CSV: {PROFILE_CSV_PATH}{C_RESET}")
    print(f"{C_GRAY}   JSON: {PROFILE_SUMMARY_PATH}{C_RESET}")


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

def is_insufficient_data_error(exc):
    return isinstance(exc, ValueError) and ("有效資料不足" in str(exc))

def load_all_raw_data():
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"找不到資料夾 {DATA_DIR}，請先執行 vip_smart_downloader.py！")

    print(f"{C_CYAN}📦 正在將歷史數據載入記憶體快取 (僅需執行一次)...{C_RESET}")
    csv_inputs, duplicate_file_issue_lines = discover_unique_csv_inputs(DATA_DIR)
    if not csv_inputs:
        raise FileNotFoundError(f"資料夾 {DATA_DIR} 內沒有任何 CSV 檔案。")

    load_issues = list(duplicate_file_issue_lines)
    total_invalid_rows = 0
    total_duplicate_dates = 0
    total_dropped_rows = 0
    total_files = len(csv_inputs)
    fresh_raw_data_cache = {}

    for count, (ticker, file_path) in enumerate(csv_inputs, start=1):
        try:
            raw_df = pd.read_csv(file_path)
            min_rows_needed = OPTIMIZER_REQUIRED_MIN_ROWS
            if len(raw_df) < min_rows_needed:
                load_issues.append(f"{ticker}: 原始資料列數不足 ({len(raw_df)})，至少需要 {min_rows_needed} 列")
                continue

            clean_df, sanitize_stats = sanitize_ohlcv_dataframe(raw_df, ticker, min_rows=min_rows_needed)
            fresh_raw_data_cache[ticker] = clean_df

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

        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError, KeyError, IndexError, TypeError, RuntimeError) as e:
            if is_insufficient_data_error(e):
                load_issues.append(f"{ticker}: {type(e).__name__}: {e}")
                continue
            raise RuntimeError(
                f"optimizer 原始資料快取失敗: ticker={ticker} | {format_exception_summary(e)}"
            ) from e

        if count % 50 == 0 or count == total_files:
            print(
                f"{C_GRAY}   進度: [{count}/{total_files}] 已掃描股票快取...{C_RESET}",
                end="\r"
            )

    if not fresh_raw_data_cache:
        raise RuntimeError("記憶體快取完成後仍無任何可用標的，無法進行 optimizer。")

    # # (AI註: 原子性更新快取，避免同一個 Python session 重跑時殘留舊 ticker，
    # # (AI註: 也避免載入途中失敗留下半新半舊的污染狀態)
    RAW_DATA_CACHE.clear()
    RAW_DATA_CACHE.update(fresh_raw_data_cache)

    print(
        f"\n{C_GREEN}✅ 記憶體快取完成！共載入 {len(RAW_DATA_CACHE)} 檔標的，"
        f"移除 {total_dropped_rows} 列資料 "
        f"(異常OHLCV={total_invalid_rows}, 重複日期={total_duplicate_dates})。{C_RESET}\n"
    )

    if load_issues:
        issue_path = write_issue_log("optimizer_load_issues", load_issues, log_dir=OUTPUT_DIR)
        print(f"{C_YELLOW}⚠️ 資料載入/清洗摘要共 {len(load_issues)} 筆，已寫入: {issue_path}{C_RESET}")

def worker_prep_data(ticker, df, params):
    t_worker_start = time.perf_counter()
    profile_stats = {}
    try:
        min_rows_needed = get_required_min_rows(params)
        if len(df) < min_rows_needed:
            return {
                "ticker": ticker,
                "ok": False,
                "reason": f"有效資料不足: 清洗後僅剩 {len(df)} 列，至少需要 {min_rows_needed} 列",
                "data": None,
                "logs": None,
                "profile": {
                    "worker_total_sec": time.perf_counter() - t_worker_start,
                    "prep_total_sec": 0.0,
                    "copy_sec": 0.0,
                    "generate_signals_sec": 0.0,
                    "assign_columns_sec": 0.0,
                    "run_backtest_sec": 0.0,
                    "to_dict_sec": 0.0,
                },
            }

        df_prepared, logs = prep_stock_data_and_trades(df, params, profile_stats=profile_stats)
        t0 = time.perf_counter()
        packed_data = pack_prepared_stock_data(df_prepared)
        pack_sec = time.perf_counter() - t0
        return {
            "ticker": ticker,
            "ok": True,
            "reason": "",
            "data": packed_data,
            "logs": logs,
            "profile": {
                "worker_total_sec": time.perf_counter() - t_worker_start,
                "prep_total_sec": float(profile_stats.get('total_sec', 0.0)),
                "copy_sec": float(profile_stats.get('copy_sec', 0.0)),
                "generate_signals_sec": float(profile_stats.get('generate_signals_sec', 0.0)),
                "assign_columns_sec": float(profile_stats.get('assign_columns_sec', 0.0)),
                "run_backtest_sec": float(profile_stats.get('run_backtest_sec', 0.0)),
                "to_dict_sec": float(pack_sec),
            },
        }
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError, KeyError, IndexError, TypeError, RuntimeError) as e:
        if is_insufficient_data_error(e):
            tb_lines = traceback.format_exc().strip().splitlines()
            tb_tail = " | ".join(tb_lines[-3:]) if tb_lines else ""
            return {
                "ticker": ticker,
                "ok": False,
                "reason": f"{type(e).__name__}: {e}" + (f" | Traceback: {tb_tail}" if tb_tail else ""),
                "data": None,
                "logs": None,
                "profile": {
                    "worker_total_sec": time.perf_counter() - t_worker_start,
                    "prep_total_sec": float(profile_stats.get('total_sec', 0.0)),
                    "copy_sec": float(profile_stats.get('copy_sec', 0.0)),
                    "generate_signals_sec": float(profile_stats.get('generate_signals_sec', 0.0)),
                    "assign_columns_sec": float(profile_stats.get('assign_columns_sec', 0.0)),
                    "run_backtest_sec": float(profile_stats.get('run_backtest_sec', 0.0)),
                    "to_dict_sec": 0.0,
                },
            }
        raise RuntimeError(
            f"optimizer 候選資料準備失敗: ticker={ticker} | {format_exception_summary(e)}"
        ) from e

# # (AI註: 單一真理來源 - 統一合併 worker 回傳，避免平行與 fallback 序列邏輯分叉)
def merge_prep_result(result, all_dfs_fast, all_trade_logs, master_dates, prep_failures, prep_profile):
    ticker = result["ticker"]
    result_profile = result.get("profile", {})

    prep_profile['worker_total_sum_sec'] += float(result_profile.get('worker_total_sec', 0.0))
    prep_profile['prep_total_sum_sec'] += float(result_profile.get('prep_total_sec', 0.0))
    prep_profile['copy_sum_sec'] += float(result_profile.get('copy_sec', 0.0))
    prep_profile['generate_signals_sum_sec'] += float(result_profile.get('generate_signals_sec', 0.0))
    prep_profile['assign_sum_sec'] += float(result_profile.get('assign_columns_sec', 0.0))
    prep_profile['run_backtest_sum_sec'] += float(result_profile.get('run_backtest_sec', 0.0))
    prep_profile['to_dict_sum_sec'] += float(result_profile.get('to_dict_sec', 0.0))

    if result["ok"]:
        prep_profile['ok_count'] += 1
        all_dfs_fast[ticker] = result["data"]
        all_trade_logs[ticker] = result["logs"]
        master_dates.update(get_fast_dates(result["data"]))
    else:
        prep_profile['fail_count'] += 1
        prep_failures.append((ticker, result["reason"]))

def objective(trial):
    t_objective_start = time.perf_counter()

    ai_use_bb = trial.suggest_categorical("use_bb", [True, False])
    ai_use_kc = trial.suggest_categorical("use_kc", [True, False])
    ai_use_vol = trial.suggest_categorical("use_vol", [True, False])

    if ai_use_vol:
        vol_short_len = trial.suggest_int("vol_short_len", 1, 10)
        vol_long_len = trial.suggest_int("vol_long_len", vol_short_len, 30)
    else:
        vol_short_len = 5
        vol_long_len = 19

    ai_params = V16StrategyParams(
        atr_len = trial.suggest_int("atr_len", 3, 25),
        atr_times_init = trial.suggest_float("atr_times_init", 1.0, 3.5, step=0.1),
        atr_times_trail = trial.suggest_float("atr_times_trail", 2.0, 4.5, step=0.1),
        atr_buy_tol = trial.suggest_float("atr_buy_tol", 0.1, 3.5, step=0.1),
        high_len = trial.suggest_int("high_len", OPTIMIZER_HIGH_LEN_MIN, OPTIMIZER_HIGH_LEN_MAX, step=OPTIMIZER_HIGH_LEN_STEP),
        tp_percent = resolve_optimizer_tp_percent(trial),
        use_bb = ai_use_bb, use_kc = ai_use_kc, use_vol = ai_use_vol,
        bb_len = trial.suggest_int("bb_len", 10, 30, step=1) if ai_use_bb else 20,
        bb_mult = trial.suggest_float("bb_mult", 1.0, 2.5, step=0.1) if ai_use_bb else 2.0,
        kc_len = trial.suggest_int("kc_len", 3, 30, step=1) if ai_use_kc else 20,
        kc_mult = trial.suggest_float("kc_mult", 1.0, 3.0, step=0.1) if ai_use_kc else 2.0,
        vol_short_len = vol_short_len,
        vol_long_len = vol_long_len,
        min_history_trades = trial.suggest_int("min_history_trades", 0, 5),
        min_history_ev = trial.suggest_float("min_history_ev", -1.0, 0.5, step=0.1),
        min_history_win_rate = trial.suggest_float("min_history_win_rate", 0.0, 0.6, step=0.01),
        use_compounding = True
    )
    all_dfs_fast, all_trade_logs, master_dates = {}, {}, set()
    prep_failures = []
    prep_profile = {
        'worker_total_sum_sec': 0.0,
        'prep_total_sum_sec': 0.0,
        'copy_sum_sec': 0.0,
        'generate_signals_sum_sec': 0.0,
        'assign_sum_sec': 0.0,
        'run_backtest_sum_sec': 0.0,
        'to_dict_sum_sec': 0.0,
        'ok_count': 0,
        'fail_count': 0,
    }
    max_workers = resolve_optimizer_max_workers(ai_params)
    t0 = time.perf_counter()
    prep_mode = "parallel"

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(worker_prep_data, ticker, df, ai_params) for ticker, df in RAW_DATA_CACHE.items()]
            for future in as_completed(futures):
                result = future.result()
                merge_prep_result(
                    result,
                    all_dfs_fast, all_trade_logs, master_dates,
                    prep_failures, prep_profile
                )

    except BrokenProcessPool as e:
        # # (AI註: Windows 多進程池偶發崩潰時，不讓整個 Optuna 中止；退回單進程完成本 trial)
        prep_mode = "sequential_fallback"
        trial.set_user_attr("prep_pool_error", f"{type(e).__name__}: {e}")

        all_dfs_fast, all_trade_logs, master_dates = {}, {}, set()
        prep_failures = []
        prep_profile = {
            'worker_total_sum_sec': 0.0,
            'prep_total_sum_sec': 0.0,
            'copy_sum_sec': 0.0,
            'generate_signals_sum_sec': 0.0,
            'assign_sum_sec': 0.0,
            'run_backtest_sum_sec': 0.0,
            'to_dict_sum_sec': 0.0,
            'ok_count': 0,
            'fail_count': 0,
        }

        for ticker, df in RAW_DATA_CACHE.items():
            result = worker_prep_data(ticker, df, ai_params)
            merge_prep_result(
                result,
                all_dfs_fast, all_trade_logs, master_dates,
                prep_failures, prep_profile
            )

    prep_wall_sec = time.perf_counter() - t0
    trial.set_user_attr("prep_mode", prep_mode)
    if prep_failures:
        insufficient_failures = [(ticker, reason) for ticker, reason in prep_failures if is_insufficient_data_message(reason)]

        record_optimizer_prep_failures(
            trial_number=trial.number,
            insufficient_failures=insufficient_failures
        )

    profile_row = {
        'trial_number': trial.number + 1,
        'objective_wall_sec': 0.0,
        'prep_wall_sec': prep_wall_sec,
        'prep_worker_total_sum_sec': prep_profile['worker_total_sum_sec'],
        'prep_worker_copy_sum_sec': prep_profile['copy_sum_sec'],
        'prep_worker_generate_signals_sum_sec': prep_profile['generate_signals_sum_sec'],
        'prep_worker_assign_sum_sec': prep_profile['assign_sum_sec'],
        'prep_worker_run_backtest_sum_sec': prep_profile['run_backtest_sum_sec'],
        'prep_worker_to_dict_sum_sec': prep_profile['to_dict_sum_sec'],
        'prep_ok_count': prep_profile['ok_count'],
        'prep_fail_count': prep_profile['fail_count'],
        'prep_avg_per_ok_sec': (prep_profile['prep_total_sum_sec'] / prep_profile['ok_count']) if prep_profile['ok_count'] > 0 else 0.0,
        'sort_dates_sec': 0.0,
        'portfolio_wall_sec': 0.0,
        'portfolio_total_sec': 0.0,
        'portfolio_ticker_dates_sec': 0.0,
        'portfolio_build_trade_index_sec': 0.0,
        'portfolio_day_loop_sec': 0.0,
        'portfolio_candidate_scan_sec': 0.0,
        'portfolio_rotation_sec': 0.0,
        'portfolio_settle_sec': 0.0,
        'portfolio_buy_sec': 0.0,
        'portfolio_equity_mark_sec': 0.0,
        'portfolio_closeout_sec': 0.0,
        'portfolio_curve_stats_sec': 0.0,
        'filter_rules_sec': 0.0,
        'score_calc_sec': 0.0,
        'ret_pct': 0.0,
        'mdd': 0.0,
        'trade_count': 0,
        'annual_return_pct': 0.0,
        'annual_trades': 0.0,
        'reserved_buy_fill_rate': 0.0,
        'full_year_count': 0,
        'min_full_year_return_pct': 0.0,
        'm_win_rate': 0.0,
        'r_squared': 0.0,
        'base_score': 0.0,
        'trial_value': -9999.0,
        'fail_reason': '',
    }

    if not master_dates:
        profile_row['objective_wall_sec'] = time.perf_counter() - t_objective_start
        profile_row['fail_reason'] = '無有效資料'
        append_profile_row(profile_row)
        trial.set_user_attr("profile_row", profile_row)
        return -9999.0

    t0 = time.perf_counter()
    sorted_dates = sorted(list(master_dates))
    profile_row['sort_dates_sec'] = time.perf_counter() - t0
    benchmark_data = all_dfs_fast.get("0050", None)

    pf_profile = {}
    t0 = time.perf_counter()
    ret_pct, mdd, t_count, final_eq, avg_exp, max_exp, bm_ret, bm_mdd, win_rate, pf_ev, pf_payoff, total_missed, total_missed_sells, r_sq, m_win_rate, bm_r_sq, bm_m_win_rate, normal_trade_count, extended_trade_count, annual_trades, reserved_buy_fill_rate, annual_return_pct, bm_annual_return_pct = run_portfolio_timeline(
        all_dfs_fast, all_trade_logs, sorted_dates, TRAIN_START_YEAR, ai_params,
        TRAIN_MAX_POSITIONS, TRAIN_ENABLE_ROTATION,
        benchmark_ticker="0050", benchmark_data=benchmark_data, is_training=True, profile_stats=pf_profile, verbose=False
    )
    profile_row['portfolio_wall_sec'] = time.perf_counter() - t0
    profile_row['portfolio_total_sec'] = float(pf_profile.get('portfolio_wall_sec', 0.0))
    profile_row['portfolio_ticker_dates_sec'] = float(pf_profile.get('portfolio_ticker_dates_sec', 0.0))
    profile_row['portfolio_build_trade_index_sec'] = float(pf_profile.get('portfolio_build_trade_index_sec', 0.0))
    profile_row['portfolio_day_loop_sec'] = float(pf_profile.get('portfolio_day_loop_sec', 0.0))
    profile_row['portfolio_candidate_scan_sec'] = float(pf_profile.get('portfolio_candidate_scan_sec', 0.0))
    profile_row['portfolio_rotation_sec'] = float(pf_profile.get('portfolio_rotation_sec', 0.0))
    profile_row['portfolio_settle_sec'] = float(pf_profile.get('portfolio_settle_sec', 0.0))
    profile_row['portfolio_buy_sec'] = float(pf_profile.get('portfolio_buy_sec', 0.0))
    profile_row['portfolio_equity_mark_sec'] = float(pf_profile.get('portfolio_equity_mark_sec', 0.0))
    profile_row['portfolio_closeout_sec'] = float(pf_profile.get('portfolio_closeout_sec', 0.0))
    profile_row['portfolio_curve_stats_sec'] = float(pf_profile.get('curve_stats_sec', 0.0))
    full_year_count = int(pf_profile.get('full_year_count', 0))
    min_full_year_return_pct = float(pf_profile.get('min_full_year_return_pct', 0.0))
    bm_min_full_year_return_pct = float(pf_profile.get('bm_min_full_year_return_pct', 0.0))

    profile_row['ret_pct'] = ret_pct
    profile_row['mdd'] = mdd
    profile_row['trade_count'] = t_count
    profile_row['annual_return_pct'] = annual_return_pct
    profile_row['annual_trades'] = annual_trades
    profile_row['reserved_buy_fill_rate'] = reserved_buy_fill_rate
    profile_row['full_year_count'] = full_year_count
    profile_row['min_full_year_return_pct'] = min_full_year_return_pct
    profile_row['m_win_rate'] = m_win_rate
    profile_row['r_squared'] = r_sq

    t0 = time.perf_counter()
    fail_reason = None
    if mdd > MAX_PORTFOLIO_MDD_PCT:
        fail_reason = f"回撤過大 ({mdd:.1f}%)"
    elif annual_trades < MIN_ANNUAL_TRADES:
        fail_reason = f"年化交易次數過低 ({annual_trades:.2f}次/年)"
    elif reserved_buy_fill_rate < MIN_BUY_FILL_RATE:
        fail_reason = f"保留後買進成交率過低 ({reserved_buy_fill_rate:.2f}%)"
    elif annual_return_pct <= 0:
        fail_reason = f"年化報酬率非正 ({annual_return_pct:.2f}%)"
    elif full_year_count <= 0:
        fail_reason = "無完整年度可驗證 min{r_y}"
    elif min_full_year_return_pct <= MIN_FULL_YEAR_RETURN_PCT:
        fail_reason = (
            f"完整年度最差報酬未大於 {MIN_FULL_YEAR_RETURN_PCT:.2f}% "
            f"({min_full_year_return_pct:.2f}%)"
        )
    elif win_rate < MIN_TRADE_WIN_RATE:
        fail_reason = f"實戰勝率偏低 ({win_rate:.2f}%)"
    elif m_win_rate < MIN_MONTHLY_WIN_RATE:
        fail_reason = f"月勝率偏低 ({m_win_rate:.0f}%)"
    elif r_sq < MIN_EQUITY_CURVE_R_SQUARED:
        fail_reason = f"曲線過度震盪 (R²={r_sq:.2f})"
    profile_row['filter_rules_sec'] = time.perf_counter() - t0

    if fail_reason is not None:
        trial.set_user_attr("fail_reason", fail_reason)
        profile_row['fail_reason'] = fail_reason
        profile_row['trial_value'] = -9999.0
        profile_row['objective_wall_sec'] = time.perf_counter() - t_objective_start
        append_profile_row(profile_row)
        trial.set_user_attr("profile_row", profile_row)
        return -9999.0

    t0 = time.perf_counter()
    base_score = calc_portfolio_score(ret_pct, mdd, m_win_rate, r_sq, annual_return_pct=annual_return_pct)
    final_score = base_score
    profile_row['score_calc_sec'] = time.perf_counter() - t0
    profile_row['base_score'] = base_score

    trial.set_user_attr("pf_return", ret_pct); trial.set_user_attr("pf_mdd", mdd); trial.set_user_attr("pf_trades", t_count); trial.set_user_attr("final_equity", final_eq); trial.set_user_attr("avg_exposure", avg_exp); trial.set_user_attr("max_exposure", max_exp); trial.set_user_attr("bm_return", bm_ret); trial.set_user_attr("bm_mdd", bm_mdd); trial.set_user_attr("win_rate", win_rate); trial.set_user_attr("pf_ev", pf_ev); trial.set_user_attr("pf_payoff", pf_payoff); trial.set_user_attr("missed_buys", total_missed); trial.set_user_attr("missed_sells", total_missed_sells); trial.set_user_attr("normal_trades", normal_trade_count); trial.set_user_attr("extended_trades", extended_trade_count); trial.set_user_attr("annual_trades", annual_trades); trial.set_user_attr("reserved_buy_fill_rate", reserved_buy_fill_rate); trial.set_user_attr("annual_return_pct", annual_return_pct); trial.set_user_attr("bm_annual_return_pct", bm_annual_return_pct); trial.set_user_attr("full_year_count", full_year_count); trial.set_user_attr("min_full_year_return_pct", min_full_year_return_pct); trial.set_user_attr("yearly_return_rows", pf_profile.get("yearly_return_rows", [])); trial.set_user_attr("base_score", base_score)
    trial.set_user_attr("bm_min_full_year_return_pct", bm_min_full_year_return_pct)
    trial.set_user_attr("r_squared", r_sq)
    trial.set_user_attr("m_win_rate", m_win_rate)
    trial.set_user_attr("bm_r_squared", bm_r_sq)
    trial.set_user_attr("bm_m_win_rate", bm_m_win_rate)

    profile_row['trial_value'] = final_score
    profile_row['objective_wall_sec'] = time.perf_counter() - t_objective_start
    append_profile_row(profile_row)
    trial.set_user_attr("profile_row", profile_row)
    return final_score

def monitoring_callback(study, trial):
    global CURRENT_SESSION_TRIAL
    CURRENT_SESSION_TRIAL += 1
    duration = trial.duration.total_seconds() if trial.duration else 0.0
    if trial.value is not None and trial.value <= -9000:
        fail_msg = trial.user_attrs.get("fail_reason", "策略無效")
        status_text, score_text = f"{C_YELLOW}淘汰 [{fail_msg}]{C_RESET}", "N/A"
    else:
        prep_mode = trial.user_attrs.get("prep_mode", "parallel")
        mode_suffix = " [fallback]" if prep_mode == "sequential_fallback" else ""
        status_text, score_text = f"{C_GREEN}進化中{mode_suffix}{C_RESET}", f"{trial.value:.3f}"
    print(f"\r{C_GRAY}⏳ [累積 {trial.number + 1:>4} | 本輪 {CURRENT_SESSION_TRIAL:>3}/{N_TRIALS}] 耗時: {duration:>5.1f}s | 系統評分: {score_text:>7} | 狀態: {status_text}{C_RESET}\033[K", end="", flush=True)

    if ENABLE_OPTIMIZER_PROFILING and ENABLE_PROFILE_CONSOLE_PRINT and (CURRENT_SESSION_TRIAL % PROFILE_PRINT_EVERY_N_TRIALS == 0):
        profile = trial.user_attrs.get("profile_row", {})
        print()
        print(
            f"{C_GRAY}   [Profile] total={float(profile.get('objective_wall_sec', 0.0)):.3f}s | "
            f"prep_wall={float(profile.get('prep_wall_sec', 0.0)):.3f}s | "
            f"pf_wall={float(profile.get('portfolio_wall_sec', 0.0)):.3f}s | "
            f"gen_sum={float(profile.get('prep_worker_generate_signals_sum_sec', 0.0)):.3f}s | "
            f"backtest_sum={float(profile.get('prep_worker_run_backtest_sum_sec', 0.0)):.3f}s | "
            f"to_dict_sum={float(profile.get('prep_worker_to_dict_sum_sec', 0.0)):.3f}s | "
            f"pf_loop={float(profile.get('portfolio_day_loop_sec', 0.0)):.3f}s{C_RESET}"
        )
    
    if study.best_trial.number == trial.number and trial.value is not None and trial.value > -9000:
        print()
        attrs = trial.user_attrs
        p = build_optimizer_trial_params(trial.params, attrs)
        mode_display = "啟用 (汰弱換強)" if TRAIN_ENABLE_ROTATION else "關閉 (穩定鎖倉)"
        print(f"\n{C_RED}🏆 破紀錄！發現更強的投資組合參數！ (累積第 {trial.number + 1} 次測試){C_RESET}")
        
        print_strategy_dashboard(
            params=p, title="績效與風險對比表", mode_display=mode_display, max_pos=TRAIN_MAX_POSITIONS,
            trades=attrs['pf_trades'], missed_b=attrs.get('missed_buys', 0), missed_s=attrs.get('missed_sells', 0),
            final_eq=attrs['final_equity'], avg_exp=attrs['avg_exposure'], max_exp=attrs.get('max_exposure', None),
            sys_ret=attrs['pf_return'], bm_ret=attrs['bm_return'], sys_mdd=attrs['pf_mdd'], bm_mdd=attrs['bm_mdd'], 
            win_rate=attrs['win_rate'], payoff=attrs['pf_payoff'], ev=attrs['pf_ev'],
            r_sq=attrs['r_squared'], m_win_rate=attrs['m_win_rate'], bm_r_sq=attrs.get('bm_r_squared', 0.0), bm_m_win_rate=attrs.get('bm_m_win_rate', 0.0),
            normal_trades=attrs.get('normal_trades', attrs['pf_trades']), extended_trades=attrs.get('extended_trades', 0),
            annual_trades=attrs.get('annual_trades', 0.0), reserved_buy_fill_rate=attrs.get('reserved_buy_fill_rate', 0.0),
            annual_return_pct=attrs.get('annual_return_pct', 0.0), bm_annual_return_pct=attrs.get('bm_annual_return_pct', 0.0),
            min_full_year_return_pct=attrs.get('min_full_year_return_pct', 0.0), bm_min_full_year_return_pct=attrs.get('bm_min_full_year_return_pct', 0.0)
        )
        print(f"{C_GRAY}   年化報酬率: {attrs.get('annual_return_pct', 0.0):.2f}% | 年化交易次數: {attrs.get('annual_trades', 0.0):.1f} 次/年 | 保留後買進成交率: {attrs.get('reserved_buy_fill_rate', 0.0):.1f}% | 完整年度數: {attrs.get('full_year_count', 0)} | 最差完整年度: {attrs.get('min_full_year_return_pct', 0.0):.2f}%{C_RESET}")

if __name__ == "__main__":
    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(f"⚙️ {C_YELLOW}V16 端到端 (End-to-End) 投資組合極速 AI 訓練引擎啟動{C_RESET}")
    print(f"{C_CYAN}================================================================================{C_RESET}")
    ensure_runtime_dirs(); load_all_raw_data(); db_file = DB_FILE_PATH; DB_NAME = f"sqlite:///{db_file}"
    init_profile_output_files()
    if ENABLE_OPTIMIZER_PROFILING:
        print(f"{C_GRAY}🧪 Profiling 已啟用，trial 明細將寫入: {PROFILE_CSV_PATH}{C_RESET}")
    
    if os.path.exists(db_file):
        choice = input("\n👉 發現舊有 Portfolio 記憶庫！ [1] 接續訓練  [2] 刪除重來 (預設 1): ").strip()
        if choice == '2': 
            os.remove(db_file)
            print(f"{C_RED}🗑️ 已刪除舊記憶。{C_RESET}")
            
    user_input = input("👉 請輸入訓練次數 (預設 50000，輸入 0 則直接提取匯出參數): ").strip()
    N_TRIALS = int(user_input) if user_input != "" else 50000
    study = optuna.create_study(study_name="v16_portfolio_optimization_overnight", storage=DB_NAME, load_if_exists=True, direction="maximize")
    
    if len(study.trials) > 0:
        print(f"\n{C_GREEN}✅ 已累積 {len(study.trials)} 次經驗。{C_RESET}")
        try:
            best_trial = study.best_trial
            if best_trial.value is not None and best_trial.value > -9000:
                attrs = best_trial.user_attrs
                p = build_optimizer_trial_params(best_trial.params, attrs)
                mode_display = "啟用 (汰弱換強)" if TRAIN_ENABLE_ROTATION else "關閉 (穩定鎖倉)"
                
                print(f"\n{C_CYAN}📜 【歷史突破紀錄還原】{C_RESET}")
                print(f"{C_RED}🏆 目前記憶庫的最強參數！ (來自累積第 {best_trial.number + 1} 次測試){C_RESET}")
                
                print_strategy_dashboard(
                    params=p, title="績效與風險對比表", mode_display=mode_display, max_pos=TRAIN_MAX_POSITIONS,
                    trades=attrs['pf_trades'], missed_b=attrs.get('missed_buys', 0), missed_s=attrs.get('missed_sells', 0),
                    final_eq=attrs['final_equity'], avg_exp=attrs['avg_exposure'], max_exp=attrs.get('max_exposure', None),
                    sys_ret=attrs['pf_return'], bm_ret=attrs['bm_return'], sys_mdd=attrs['pf_mdd'], bm_mdd=attrs['bm_mdd'], 
                    win_rate=attrs['win_rate'], payoff=attrs['pf_payoff'], ev=attrs['pf_ev'],
                    r_sq=attrs.get('r_squared', 0.0), m_win_rate=attrs.get('m_win_rate', 0.0), bm_r_sq=attrs.get('bm_r_squared', 0.0), bm_m_win_rate=attrs.get('bm_m_win_rate', 0.0),
                    normal_trades=attrs.get('normal_trades', attrs['pf_trades']), extended_trades=attrs.get('extended_trades', 0),
                    annual_trades=attrs.get('annual_trades', 0.0), reserved_buy_fill_rate=attrs.get('reserved_buy_fill_rate', 0.0),
                    annual_return_pct=attrs.get('annual_return_pct', 0.0), bm_annual_return_pct=attrs.get('bm_annual_return_pct', 0.0),
                    min_full_year_return_pct=attrs.get('min_full_year_return_pct', 0.0), bm_min_full_year_return_pct=attrs.get('bm_min_full_year_return_pct', 0.0)
                )
                print(f"{C_GRAY}   年化報酬率: {attrs.get('annual_return_pct', 0.0):.2f}% | 年化交易次數: {attrs.get('annual_trades', 0.0):.1f} 次/年 | 保留後買進成交率: {attrs.get('reserved_buy_fill_rate', 0.0):.1f}% | 完整年度數: {attrs.get('full_year_count', 0)} | 最差完整年度: {attrs.get('min_full_year_return_pct', 0.0):.2f}%{C_RESET}")
        except ValueError as e:
            print(f"{C_YELLOW}⚠️ 無法還原歷史最佳參數儀表板: {type(e).__name__}: {e}{C_RESET}")

    if N_TRIALS == 0:
        completed_trials = [trial for trial in study.trials if trial.value is not None]
        if len(study.trials) == 0:
            print(f"\n{C_YELLOW}⚠️ 記憶庫為空，無法匯出。{C_RESET}\n")
        elif not completed_trials:
            print(f"\n{C_YELLOW}⚠️ 目前記憶庫中尚無已完成紀錄，無法匯出。{C_RESET}\n")
        else:
            try:
                best_trial = study.best_trial
                if best_trial.value is not None and best_trial.value > -9000:
                    best_params_payload = build_best_params_payload_from_trial(best_trial)
                    with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
                        json.dump(best_params_payload, f, indent=4, ensure_ascii=False)
                    print(f"\n{C_GREEN}💾 匯出成功！已從記憶庫提取最強參數！{C_RESET}\n")
                else:
                    print(f"\n{C_YELLOW}⚠️ 目前記憶庫中尚無及格的紀錄，無法匯出。{C_RESET}\n")
            except ValueError as e:
                print(f"\n{C_YELLOW}⚠️ 匯出最佳參數失敗: {type(e).__name__}: {e}{C_RESET}\n")
    else:
        print(f"\n{C_CYAN}🚀 開始優化...{C_RESET}\n")
        try:
            study.optimize(objective, n_trials=N_TRIALS, n_jobs=1, callbacks=[monitoring_callback])
        except KeyboardInterrupt:
            print(f"\n{C_YELLOW}⚠️ 使用者中斷訓練流程。{C_RESET}")

        print()
        print_profile_summary()
        print_optimizer_prep_summary()
        print(f"\n{C_YELLOW}🛑 訓練階段結束或已中斷。{C_RESET}")
