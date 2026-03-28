import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

import pandas as pd

from core.v16_data_utils import (
    discover_unique_csv_inputs,
    get_required_min_rows,
    sanitize_ohlcv_dataframe,
)
from core.v16_display import C_CYAN, C_GRAY, C_GREEN, C_RESET, C_YELLOW
from core.v16_log_utils import format_exception_summary, write_issue_log
from core.v16_portfolio_engine import get_fast_dates, pack_prepared_stock_data, prep_stock_data_and_trades
from core.v16_runtime_utils import get_process_pool_executor_kwargs


def is_insufficient_data_message(message):
    return isinstance(message, str) and ("有效資料不足" in message)


def is_insufficient_data_error(exc):
    return isinstance(exc, ValueError) and ("有效資料不足" in str(exc))


def resolve_optimizer_max_workers(params, default_max_workers):
    configured = getattr(params, "optimizer_max_workers", default_max_workers)
    try:
        configured = int(configured)
    except (TypeError, ValueError):
        configured = default_max_workers
    return max(1, configured)


def load_all_raw_data(data_dir, required_min_rows, output_dir):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"找不到資料夾 {data_dir}，請先執行 apps/smart_downloader.py！")

    print(f"{C_CYAN}📦 正在將歷史數據載入記憶體快取 (僅需執行一次)...{C_RESET}")
    csv_inputs, duplicate_file_issue_lines = discover_unique_csv_inputs(data_dir)
    if not csv_inputs:
        raise FileNotFoundError(f"資料夾 {data_dir} 內沒有任何 CSV 檔案。")

    load_issues = list(duplicate_file_issue_lines)
    total_invalid_rows = 0
    total_duplicate_dates = 0
    total_dropped_rows = 0
    total_files = len(csv_inputs)
    fresh_raw_data_cache = {}

    for count, (ticker, file_path) in enumerate(csv_inputs, start=1):
        try:
            raw_df = pd.read_csv(file_path)
            if len(raw_df) < required_min_rows:
                load_issues.append(f"{ticker}: 原始資料列數不足 ({len(raw_df)})，至少需要 {required_min_rows} 列")
                continue

            clean_df, sanitize_stats = sanitize_ohlcv_dataframe(raw_df, ticker, min_rows=required_min_rows)
            fresh_raw_data_cache[ticker] = clean_df

            invalid_row_count = sanitize_stats["invalid_row_count"]
            duplicate_date_count = sanitize_stats["duplicate_date_count"]
            dropped_row_count = sanitize_stats["dropped_row_count"]

            total_invalid_rows += invalid_row_count
            total_duplicate_dates += duplicate_date_count
            total_dropped_rows += dropped_row_count

            if dropped_row_count > 0:
                load_issues.append(
                    f"{ticker}: 清洗移除 {dropped_row_count} 列 "
                    f"(異常OHLCV={invalid_row_count}, 重複日期={duplicate_date_count})"
                )
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError, KeyError, IndexError, TypeError, RuntimeError) as exc:
            if is_insufficient_data_error(exc):
                load_issues.append(f"{ticker}: {type(exc).__name__}: {exc}")
                continue
            raise RuntimeError(
                f"optimizer 原始資料快取失敗: ticker={ticker} | {format_exception_summary(exc)}"
            ) from exc

        if count % 50 == 0 or count == total_files:
            print(f"{C_GRAY}   進度: [{count}/{total_files}] 已掃描股票快取...{C_RESET}", end="\r")

    if not fresh_raw_data_cache:
        raise RuntimeError("記憶體快取完成後仍無任何可用標的，無法進行 optimizer。")

    print(
        f"\n{C_GREEN}✅ 記憶體快取完成！共載入 {len(fresh_raw_data_cache)} 檔標的，"
        f"移除 {total_dropped_rows} 列資料 "
        f"(異常OHLCV={total_invalid_rows}, 重複日期={total_duplicate_dates})。{C_RESET}\n"
    )

    if load_issues:
        issue_path = write_issue_log("optimizer_load_issues", load_issues, log_dir=output_dir)
        print(f"{C_YELLOW}⚠️ 資料載入/清洗摘要共 {len(load_issues)} 筆，已寫入: {issue_path}{C_RESET}")

    return fresh_raw_data_cache


def worker_prep_data(ticker, df, params):
    worker_start = time.perf_counter()
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
                    "worker_total_sec": time.perf_counter() - worker_start,
                    "prep_total_sec": 0.0,
                    "copy_sec": 0.0,
                    "generate_signals_sec": 0.0,
                    "assign_columns_sec": 0.0,
                    "run_backtest_sec": 0.0,
                    "to_dict_sec": 0.0,
                },
            }

        df_prepared, logs = prep_stock_data_and_trades(df, params, profile_stats=profile_stats)
        pack_start = time.perf_counter()
        packed_data = pack_prepared_stock_data(df_prepared)
        pack_sec = time.perf_counter() - pack_start
        return {
            "ticker": ticker,
            "ok": True,
            "reason": "",
            "data": packed_data,
            "logs": logs,
            "profile": {
                "worker_total_sec": time.perf_counter() - worker_start,
                "prep_total_sec": float(profile_stats.get("total_sec", 0.0)),
                "copy_sec": float(profile_stats.get("copy_sec", 0.0)),
                "generate_signals_sec": float(profile_stats.get("generate_signals_sec", 0.0)),
                "assign_columns_sec": float(profile_stats.get("assign_columns_sec", 0.0)),
                "run_backtest_sec": float(profile_stats.get("run_backtest_sec", 0.0)),
                "to_dict_sec": float(pack_sec),
            },
        }
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError, KeyError, IndexError, TypeError, RuntimeError) as exc:
        if is_insufficient_data_error(exc):
            tb_lines = traceback.format_exc().strip().splitlines()
            tb_tail = " | ".join(tb_lines[-3:]) if tb_lines else ""
            return {
                "ticker": ticker,
                "ok": False,
                "reason": f"{type(exc).__name__}: {exc}" + (f" | Traceback: {tb_tail}" if tb_tail else ""),
                "data": None,
                "logs": None,
                "profile": {
                    "worker_total_sec": time.perf_counter() - worker_start,
                    "prep_total_sec": float(profile_stats.get("total_sec", 0.0)),
                    "copy_sec": float(profile_stats.get("copy_sec", 0.0)),
                    "generate_signals_sec": float(profile_stats.get("generate_signals_sec", 0.0)),
                    "assign_columns_sec": float(profile_stats.get("assign_columns_sec", 0.0)),
                    "run_backtest_sec": float(profile_stats.get("run_backtest_sec", 0.0)),
                    "to_dict_sec": 0.0,
                },
            }
        raise RuntimeError(
            f"optimizer 候選資料準備失敗: ticker={ticker} | {format_exception_summary(exc)}"
        ) from exc


def merge_prep_result(result, all_dfs_fast, all_trade_logs, master_dates, prep_failures, prep_profile):
    ticker = result["ticker"]
    result_profile = result.get("profile", {})

    prep_profile["worker_total_sum_sec"] += float(result_profile.get("worker_total_sec", 0.0))
    prep_profile["prep_total_sum_sec"] += float(result_profile.get("prep_total_sec", 0.0))
    prep_profile["copy_sum_sec"] += float(result_profile.get("copy_sec", 0.0))
    prep_profile["generate_signals_sum_sec"] += float(result_profile.get("generate_signals_sec", 0.0))
    prep_profile["assign_sum_sec"] += float(result_profile.get("assign_columns_sec", 0.0))
    prep_profile["run_backtest_sum_sec"] += float(result_profile.get("run_backtest_sec", 0.0))
    prep_profile["to_dict_sum_sec"] += float(result_profile.get("to_dict_sec", 0.0))

    if result["ok"]:
        prep_profile["ok_count"] += 1
        all_dfs_fast[ticker] = result["data"]
        all_trade_logs[ticker] = result["logs"]
        master_dates.update(get_fast_dates(result["data"]))
        return

    prep_profile["fail_count"] += 1
    prep_failures.append((ticker, result["reason"]))


def prepare_trial_inputs(raw_data_cache, params, default_max_workers):
    all_dfs_fast, all_trade_logs, master_dates = {}, {}, set()
    prep_failures = []
    prep_profile = {
        "worker_total_sum_sec": 0.0,
        "prep_total_sum_sec": 0.0,
        "copy_sum_sec": 0.0,
        "generate_signals_sum_sec": 0.0,
        "assign_sum_sec": 0.0,
        "run_backtest_sum_sec": 0.0,
        "to_dict_sum_sec": 0.0,
        "ok_count": 0,
        "fail_count": 0,
    }
    max_workers = resolve_optimizer_max_workers(params, default_max_workers)
    prep_mode = "parallel"
    pool_start_method = None
    prep_wall_start = time.perf_counter()
    pool_error_text = None

    try:
        process_pool_kwargs, pool_start_method = get_process_pool_executor_kwargs()
        with ProcessPoolExecutor(max_workers=max_workers, **process_pool_kwargs) as executor:
            futures = [executor.submit(worker_prep_data, ticker, df, params) for ticker, df in raw_data_cache.items()]
            for future in as_completed(futures):
                result = future.result()
                merge_prep_result(result, all_dfs_fast, all_trade_logs, master_dates, prep_failures, prep_profile)
    except BrokenProcessPool as exc:
        prep_mode = "sequential_fallback"
        pool_error_text = f"{type(exc).__name__}: {exc}"
        all_dfs_fast, all_trade_logs, master_dates = {}, {}, set()
        prep_failures = []
        prep_profile = {
            "worker_total_sum_sec": 0.0,
            "prep_total_sum_sec": 0.0,
            "copy_sum_sec": 0.0,
            "generate_signals_sum_sec": 0.0,
            "assign_sum_sec": 0.0,
            "run_backtest_sum_sec": 0.0,
            "to_dict_sum_sec": 0.0,
            "ok_count": 0,
            "fail_count": 0,
        }
        for ticker, df in raw_data_cache.items():
            result = worker_prep_data(ticker, df, params)
            merge_prep_result(result, all_dfs_fast, all_trade_logs, master_dates, prep_failures, prep_profile)

    return {
        "all_dfs_fast": all_dfs_fast,
        "all_trade_logs": all_trade_logs,
        "master_dates": master_dates,
        "prep_failures": prep_failures,
        "prep_profile": prep_profile,
        "prep_wall_sec": time.perf_counter() - prep_wall_start,
        "prep_mode": prep_mode,
        "pool_start_method": pool_start_method,
        "pool_error_text": pool_error_text,
    }
