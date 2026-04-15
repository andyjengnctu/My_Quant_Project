import inspect
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_context

import pandas as pd

from core.data_utils import get_required_min_rows
from core.log_utils import format_exception_summary
from core.portfolio_fast_data import merge_static_market_with_dynamic, prep_optimizer_stock_data_bundle, pack_static_market_data
from core.runtime_utils import get_process_pool_executor_kwargs
from tools.optimizer.feature_cache import build_optimizer_feature_cache_for_ticker
from tools.optimizer.raw_cache import is_insufficient_data_error, resolve_optimizer_max_workers

_WORKER_RAW_DATA_CACHE = None
_WORKER_FEATURE_CACHE = None
_WORKER_FEATURE_CONFIG = None


def init_worker_raw_data_cache(raw_data_cache, feature_config):
    global _WORKER_RAW_DATA_CACHE, _WORKER_FEATURE_CACHE, _WORKER_FEATURE_CONFIG
    _WORKER_RAW_DATA_CACHE = raw_data_cache
    _WORKER_FEATURE_CACHE = {}
    _WORKER_FEATURE_CONFIG = feature_config


def _resolve_feature_cache_for_ticker(*, ticker, df, optimizer_feature_cache, optimizer_feature_config):
    if optimizer_feature_cache is not None:
        cached = optimizer_feature_cache.get(ticker)
        if cached is not None:
            return cached
    if optimizer_feature_config is None:
        return None
    built = build_optimizer_feature_cache_for_ticker(df, feature_lengths=optimizer_feature_config)
    if optimizer_feature_cache is not None:
        optimizer_feature_cache[ticker] = built
    return built


def worker_prep_data(ticker, df, params, feature_cache=None):
    worker_start = time.perf_counter()
    profile_stats = {}
    try:
        min_rows_needed = get_required_min_rows(params)
        if len(df) < min_rows_needed:
            return {
                "ticker": ticker,
                "ok": False,
                "reason": f"有效資料不足: 清洗後僅剩 {len(df)} 列，至少需要 {min_rows_needed} 列",
                "dynamic": None,
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

        dynamic_data, logs = prep_optimizer_stock_data_bundle(df, params, profile_stats=profile_stats, ticker=ticker, feature_cache=feature_cache)
        pack_sec = float(profile_stats.get('to_dict_sec', 0.0))
        return {
            "ticker": ticker,
            "ok": True,
            "reason": "",
            "dynamic": dynamic_data,
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
                "dynamic": None,
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


def worker_prep_data_from_cache(ticker, params):
    global _WORKER_RAW_DATA_CACHE, _WORKER_FEATURE_CACHE, _WORKER_FEATURE_CONFIG
    if _WORKER_RAW_DATA_CACHE is None:
        raise RuntimeError("optimizer worker raw_data_cache 尚未初始化")
    try:
        df = _WORKER_RAW_DATA_CACHE[ticker]
    except KeyError as exc:
        raise RuntimeError(f"optimizer worker 找不到 ticker={ticker} 的快取資料") from exc
    feature_cache = _resolve_feature_cache_for_ticker(
        ticker=ticker,
        df=df,
        optimizer_feature_cache=_WORKER_FEATURE_CACHE,
        optimizer_feature_config=_WORKER_FEATURE_CONFIG,
    )
    return worker_prep_data(ticker, df, params, feature_cache=feature_cache)


def merge_prep_result(result, all_dfs_fast, all_trade_logs, master_dates, prep_failures, prep_profile, static_fast_cache, *, use_static_master_dates=False):
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
        all_dfs_fast[ticker] = merge_static_market_with_dynamic(static_fast_cache[ticker], result["dynamic"])
        all_trade_logs[ticker] = result["logs"]
        if not use_static_master_dates:
            master_dates.update(static_fast_cache[ticker]['dates'])
        return

    prep_profile["fail_count"] += 1
    prep_failures.append((ticker, result["reason"]))


def _build_process_pool_executor(max_workers, raw_data_cache, feature_config):
    process_pool_kwargs, pool_start_method = get_process_pool_executor_kwargs()
    executor_kwargs = dict(process_pool_kwargs)
    if os.name != "nt" and "mp_context" in inspect.signature(ProcessPoolExecutor).parameters:
        try:
            executor_kwargs["mp_context"] = get_context("fork")
            pool_start_method = "fork"
        except ValueError as exc:
            pool_start_method = (
                f"{pool_start_method or 'default'}|fork_unavailable:{type(exc).__name__}"
            )
    supports_initializer = "initializer" in inspect.signature(ProcessPoolExecutor).parameters
    if supports_initializer:
        executor_kwargs["initializer"] = init_worker_raw_data_cache
        executor_kwargs["initargs"] = (raw_data_cache, feature_config)
    return ProcessPoolExecutor(max_workers=max_workers, **executor_kwargs), pool_start_method, supports_initializer


def _build_thread_pool_executor(max_workers):
    return ThreadPoolExecutor(max_workers=max_workers), 'thread', False


def worker_prep_batch(raw_data_cache, feature_cache_store, feature_config, tickers, params):
    results = []
    for ticker in tickers:
        df = raw_data_cache[ticker]
        feature_cache = _resolve_feature_cache_for_ticker(
            ticker=ticker,
            df=df,
            optimizer_feature_cache=feature_cache_store,
            optimizer_feature_config=feature_config,
        )
        results.append(worker_prep_data(ticker, df, params, feature_cache=feature_cache))
    return results


def worker_prep_batch_from_cache(tickers, params):
    return [worker_prep_data_from_cache(ticker, params) for ticker in tickers]


def _build_balanced_ticker_batches(raw_data_cache, tickers, max_workers, *, sticky_process_assignment=False):
    worker_count = max(1, int(max_workers))
    if sticky_process_assignment:
        target_batch_count = min(len(tickers), worker_count)
    elif len(tickers) <= worker_count * 4:
        return [[ticker] for ticker in tickers]
    else:
        target_batch_count = min(len(tickers), max(worker_count, worker_count * 4))

    ordered_tickers = sorted(
        tickers,
        key=lambda ticker: len(raw_data_cache.get(ticker, ())),
        reverse=True,
    )
    batch_rows = [0] * target_batch_count
    batches = [[] for _ in range(target_batch_count)]

    for ticker in ordered_tickers:
        row_count = len(raw_data_cache.get(ticker, ()))
        target_idx = min(range(target_batch_count), key=lambda idx: batch_rows[idx])
        batches[target_idx].append(ticker)
        batch_rows[target_idx] += row_count

    return [batch for batch in batches if batch]


def _run_prep_with_executor(executor, tickers, params, all_dfs_fast, all_trade_logs, master_dates, prep_failures, prep_profile, raw_data_cache, static_fast_cache, optimizer_feature_cache, optimizer_feature_config, max_workers, executor_kind, *, use_static_master_dates=False):
    sticky_process_assignment = executor_kind == 'process'
    ticker_batches = _build_balanced_ticker_batches(
        raw_data_cache,
        tickers,
        max_workers,
        sticky_process_assignment=sticky_process_assignment,
    )
    if executor_kind == 'thread':
        futures = [executor.submit(worker_prep_batch, raw_data_cache, optimizer_feature_cache, optimizer_feature_config, batch, params) for batch in ticker_batches]
    else:
        futures = [executor.submit(worker_prep_batch_from_cache, batch, params) for batch in ticker_batches]
    for future in as_completed(futures):
        batch_results = future.result()
        for result in batch_results:
            merge_prep_result(
                result,
                all_dfs_fast,
                all_trade_logs,
                master_dates,
                prep_failures,
                prep_profile,
                static_fast_cache,
                use_static_master_dates=use_static_master_dates,
            )


def prepare_trial_inputs(raw_data_cache, params, default_max_workers, executor_bundle=None, static_fast_cache=None, static_master_dates=None, optimizer_feature_cache=None, optimizer_feature_config=None):
    resolved_static_fast_cache = static_fast_cache or {ticker: pack_static_market_data(df) for ticker, df in raw_data_cache.items()}
    use_static_master_dates = static_master_dates is not None and len(static_master_dates) > 0
    all_dfs_fast, all_trade_logs = {}, {}
    master_dates = set(static_master_dates) if use_static_master_dates else set()
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
    tickers = list(raw_data_cache.keys())
    created_executor = None
    executor_kind = 'process'

    try:
        if executor_bundle is not None and int(executor_bundle.get("max_workers", 0)) == max_workers:
            created_executor = executor_bundle["executor"]
            pool_start_method = executor_bundle.get("pool_start_method")
            executor_kind = executor_bundle.get("executor_kind", 'process')
            _run_prep_with_executor(created_executor, tickers, params, all_dfs_fast, all_trade_logs, master_dates, prep_failures, prep_profile, raw_data_cache, resolved_static_fast_cache, optimizer_feature_cache, optimizer_feature_config, max_workers, executor_kind, use_static_master_dates=use_static_master_dates)
        else:
            created_executor, pool_start_method, supports_initializer = _build_process_pool_executor(max_workers, raw_data_cache, optimizer_feature_config)
            executor_kind = 'process'
            try:
                if executor_kind == 'thread':
                    _run_prep_with_executor(created_executor, tickers, params, all_dfs_fast, all_trade_logs, master_dates, prep_failures, prep_profile, raw_data_cache, resolved_static_fast_cache, optimizer_feature_cache, optimizer_feature_config, max_workers, executor_kind, use_static_master_dates=use_static_master_dates)
                elif supports_initializer:
                    _run_prep_with_executor(created_executor, tickers, params, all_dfs_fast, all_trade_logs, master_dates, prep_failures, prep_profile, raw_data_cache, resolved_static_fast_cache, optimizer_feature_cache, optimizer_feature_config, max_workers, executor_kind, use_static_master_dates=use_static_master_dates)
                else:
                    futures = [created_executor.submit(worker_prep_data, ticker, df, params, _resolve_feature_cache_for_ticker(ticker=ticker, df=df, optimizer_feature_cache=optimizer_feature_cache, optimizer_feature_config=optimizer_feature_config)) for ticker, df in raw_data_cache.items()]
                    for future in as_completed(futures):
                        result = future.result()
                        merge_prep_result(result, all_dfs_fast, all_trade_logs, master_dates, prep_failures, prep_profile, static_fast_cache, use_static_master_dates=use_static_master_dates)
            finally:
                created_executor.shutdown(wait=True, cancel_futures=False)
                created_executor = None
    except BrokenProcessPool as exc:
        prep_mode = "sequential_fallback"
        pool_error_text = f"{type(exc).__name__}: {exc}"
        all_dfs_fast, all_trade_logs = {}, {}
        master_dates = set()
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
            result = worker_prep_data(ticker, df, params, _resolve_feature_cache_for_ticker(ticker=ticker, df=df, optimizer_feature_cache=optimizer_feature_cache, optimizer_feature_config=optimizer_feature_config))
            merge_prep_result(result, all_dfs_fast, all_trade_logs, master_dates, prep_failures, prep_profile, static_fast_cache, use_static_master_dates=use_static_master_dates)

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
