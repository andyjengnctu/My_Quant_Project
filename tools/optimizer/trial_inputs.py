import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

import pandas as pd

from core.v16_data_utils import get_required_min_rows
from core.v16_log_utils import format_exception_summary
from core.v16_portfolio_fast_data import get_fast_dates, pack_prepared_stock_data, prep_stock_data_and_trades
from core.v16_runtime_utils import get_process_pool_executor_kwargs
from tools.optimizer.raw_cache import is_insufficient_data_error, resolve_optimizer_max_workers


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

    prep_wall_sec = time.perf_counter() - prep_wall_start
    return {
        "all_dfs_fast": all_dfs_fast,
        "all_trade_logs": all_trade_logs,
        "master_dates": master_dates,
        "prep_failures": prep_failures,
        "prep_profile": prep_profile,
        "prep_mode": prep_mode,
        "pool_start_method": pool_start_method,
        "pool_error_text": pool_error_text,
        "prep_wall_sec": prep_wall_sec,
    }
