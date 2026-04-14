from concurrent.futures import ProcessPoolExecutor, as_completed
import os

import pandas as pd

from core.portfolio_engine import run_portfolio_timeline
from core.portfolio_fast_data import get_fast_dates, pack_prepared_stock_data, prep_stock_data_and_trades

from tools.validate.checks import (
    add_fail_result,
    add_skip_result,
    build_consistency_parity_params,
    build_scanner_validation_params,
    extract_yearly_profile_fields,
    make_consistency_params,
    run_scanner_reference_check_on_clean_df,
)
from tools.validate.real_case_assertions import append_real_case_checks
from tools.validate.real_case_io import load_clean_df, load_clean_df_from_path
from tools.validate.tool_adapters import (
    VALIDATION_RECOVERABLE_EXCEPTIONS,
    run_debug_trade_log_check,
    run_downloader_tool_check,
    run_portfolio_sim_tool_check,
    run_scanner_tool_check,
)
from tools.local_regression.shared_prep_cache import load_shared_prep_cache_entry


def run_single_backtest_check(df, params):
    prep_df, standalone_logs, stats = prep_stock_data_and_trades(df.copy(), params, return_stats=True)
    return stats, standalone_logs, prep_df




def build_single_ticker_portfolio_context(ticker, prep_df, standalone_logs):
    fast_data = pack_prepared_stock_data(prep_df)
    sorted_dates = sorted(get_fast_dates(fast_data))
    if not sorted_dates:
        raise ValueError(f"{ticker}: pack_prepared_stock_data 後沒有任何有效日期")

    start_year = int(pd.Timestamp(sorted_dates[0]).year)
    return {
        "fast_data": fast_data,
        "sorted_dates": sorted_dates,
        "start_year": start_year,
        "all_dfs_fast": {ticker: fast_data},
        "all_standalone_logs": {ticker: standalone_logs},
    }


def run_single_ticker_portfolio_check(ticker, prep_df, standalone_logs, params, *, portfolio_context=None):
    context = portfolio_context or build_single_ticker_portfolio_context(ticker, prep_df, standalone_logs)
    fast_data = context["fast_data"]
    sorted_dates = context["sorted_dates"]
    start_year = context["start_year"]
    profile_stats = {}

    result = run_portfolio_timeline(
        all_dfs_fast=context["all_dfs_fast"],
        all_standalone_logs=context["all_standalone_logs"],
        sorted_dates=sorted_dates,
        start_year=start_year,
        params=params,
        max_positions=1,
        enable_rotation=False,
        benchmark_ticker=ticker,
        benchmark_data=fast_data,
        is_training=True,
        profile_stats=profile_stats,
        verbose=False,
    )

    expected_result_len = 23
    if len(result) != expected_result_len:
        raise ValueError(
            f"run_portfolio_timeline(is_training=True) 回傳長度異常: {len(result)}，"
            f"預期 {expected_result_len}"
        )

    (
        total_return,
        mdd,
        trade_count,
        final_eq,
        avg_exp,
        max_exp,
        bm_ret,
        bm_mdd,
        win_rate,
        pf_ev,
        pf_payoff,
        total_missed,
        total_missed_sells,
        r_sq,
        m_win_rate,
        bm_r_sq,
        bm_m_win_rate,
        normal_trade_count,
        extended_trade_count,
        annual_trades,
        reserved_buy_fill_rate,
        annual_return_pct,
        bm_annual_return_pct,
    ) = result

    payload = {
        "total_return": total_return,
        "mdd": mdd,
        "trade_count": trade_count,
        "final_eq": final_eq,
        "avg_exp": avg_exp,
        "max_exp": max_exp,
        "bm_ret": bm_ret,
        "bm_mdd": bm_mdd,
        "win_rate": win_rate,
        "pf_ev": pf_ev,
        "pf_payoff": pf_payoff,
        "total_missed": total_missed,
        "total_missed_sells": total_missed_sells,
        "r_sq": r_sq,
        "m_win_rate": m_win_rate,
        "bm_r_sq": bm_r_sq,
        "bm_m_win_rate": bm_m_win_rate,
        "normal_trade_count": normal_trade_count,
        "extended_trade_count": extended_trade_count,
        "annual_trades": annual_trades,
        "reserved_buy_fill_rate": reserved_buy_fill_rate,
        "annual_return_pct": annual_return_pct,
        "bm_annual_return_pct": bm_annual_return_pct,
        "sorted_dates": sorted_dates,
        "start_year": start_year,
        "standalone_logs": standalone_logs,
        "prep_df": prep_df,
    }
    payload.update(extract_yearly_profile_fields(profile_stats))
    return payload


def validate_one_ticker(project_root, data_dir, csv_map_getter, ticker, base_params, *, resolved_file_path=None):
    params = make_consistency_params(base_params)
    scanner_params = build_scanner_validation_params(base_params)
    cache_entry = load_shared_prep_cache_entry(ticker)
    if resolved_file_path is not None:
        file_path, df, sanitize_stats = load_clean_df_from_path(resolved_file_path, ticker, params)
    else:
        file_path, df, sanitize_stats = load_clean_df(project_root, data_dir, csv_map_getter, ticker, params)

    use_shared_prepared = (
        isinstance(cache_entry, dict)
        and cache_entry.get("status") == "ready"
        and str(cache_entry.get("file_path", "")) == str(file_path)
    )

    results = []
    summary = {
        "ticker": ticker,
        "file_path": file_path,
        "sanitize_dropped": sanitize_stats["dropped_row_count"],
        "sanitize_invalid": sanitize_stats["invalid_row_count"],
        "sanitize_duplicate": sanitize_stats["duplicate_date_count"],
    }

    if use_shared_prepared:
        single_stats = dict(cache_entry["single_stats"])
        standalone_logs = list(cache_entry["standalone_logs"])
        prep_df = cache_entry["prepared_df"].copy()
        portfolio_context = {
            "fast_data": cache_entry["fast_data"],
            "sorted_dates": list(cache_entry["sorted_dates"]),
            "start_year": int(cache_entry["start_year"]),
            "all_dfs_fast": {ticker: cache_entry["fast_data"]},
            "all_standalone_logs": {ticker: list(cache_entry["standalone_logs"])},
        }
    else:
        single_stats, standalone_logs, prep_df = run_single_backtest_check(df, params)
        portfolio_context = build_single_ticker_portfolio_context(ticker, prep_df, standalone_logs)
    scanner_ref_stats = run_scanner_reference_check_on_clean_df(ticker, df, scanner_params)
    parity_params = build_consistency_parity_params(params)
    portfolio_stats = run_single_ticker_portfolio_check(ticker, prep_df, standalone_logs, parity_params, portfolio_context=portfolio_context)
    portfolio_sim_stats = run_portfolio_sim_tool_check(
        ticker,
        file_path,
        parity_params,
        prepared_df=prep_df,
        standalone_logs=standalone_logs,
        packed_fast_data=portfolio_context["fast_data"],
        sorted_dates=portfolio_context["sorted_dates"],
        start_year=portfolio_context["start_year"],
    )
    scanner_result, scanner_module_path = run_scanner_tool_check(
        ticker,
        file_path,
        scanner_params,
        sanitize_stats=sanitize_stats,
        precomputed_stats=scanner_ref_stats,
    )
    downloader_df, downloader_module_path, downloader_request, downloader_expected_dataset = None, None, None, None
    downloader_error = None
    try:
        downloader_df, downloader_module_path, downloader_request, downloader_expected_dataset = run_downloader_tool_check(ticker)
    except VALIDATION_RECOVERABLE_EXCEPTIONS as e:
        downloader_error = f"{type(e).__name__}: {e}"
    debug_df, debug_module_path = run_debug_trade_log_check(ticker, df, params, prepared_df=prep_df)

    summary["portfolio_sim_module_path"] = portfolio_sim_stats["module_path"]
    summary["scanner_module_path"] = scanner_module_path
    summary["downloader_module_path"] = downloader_module_path
    summary["downloader_error"] = downloader_error
    summary["debug_module_path"] = debug_module_path
    summary["single_trade_count"] = single_stats["trade_count"]
    summary["portfolio_trade_count"] = portfolio_stats["trade_count"]
    summary["open_position_exists"] = bool(single_stats["current_position"] > 0)
    summary["has_extended_candidate_today"] = bool(scanner_ref_stats.get("extended_candidate_today") is not None)
    summary["has_missed_buy"] = bool(single_stats["missed_buys"] > 0)
    summary["portfolio_half_take_profit_rows"] = int(portfolio_sim_stats["portfolio_half_take_profit_rows"])
    summary["shared_prep_cache_used"] = bool(use_shared_prepared)

    append_real_case_checks(
        results,
        ticker=ticker,
        params=params,
        df=df,
        single_stats=single_stats,
        standalone_logs=standalone_logs,
        scanner_ref_stats=scanner_ref_stats,
        portfolio_stats=portfolio_stats,
        portfolio_sim_stats=portfolio_sim_stats,
        scanner_result=scanner_result,
        downloader_df=downloader_df,
        downloader_request=downloader_request,
        downloader_expected_dataset=downloader_expected_dataset,
        downloader_error=downloader_error,
        debug_df=debug_df,
    )

    return results, summary


def _determine_real_scan_worker_count(total_tickers):
    if total_tickers <= 1:
        return 1

    cpu_count = os.cpu_count() or 1
    if cpu_count <= 2:
        return 1
    if cpu_count <= 4:
        return min(total_tickers, 2)
    if cpu_count <= 8:
        return min(total_tickers, 4)
    if cpu_count <= 12:
        return min(total_tickers, 6)
    return min(total_tickers, 8)


def _validate_one_ticker_worker(task):
    return validate_one_ticker(
        task["project_root"],
        task["data_dir"],
        None,
        task["ticker"],
        task["base_params"],
        resolved_file_path=task["resolved_file_path"],
    )



def run_real_ticker_scan(
    selected_tickers,
    base_params,
    *,
    project_root,
    data_dir,
    csv_map_getter,
    add_fail_result,
    add_skip_result,
    format_exception_summary,
    is_insufficient_data_error,
    progress_printer=None,
):
    total_tickers = len(selected_tickers)
    ticker_pass_count = 0
    ticker_skip_count = 0
    ticker_fail_count = 0
    completed_count = 0

    csv_map = csv_map_getter()
    worker_count = _determine_real_scan_worker_count(total_tickers)
    ordered_results = [None] * total_tickers
    ordered_summaries = [None] * total_tickers

    def _record_outcome(idx, ticker, *, results=None, summary=None, exc=None):
        nonlocal completed_count, ticker_pass_count, ticker_skip_count, ticker_fail_count

        ticker_results = []
        ticker_summary = None
        if exc is None:
            ticker_results = list(results or [])
            ticker_summary = dict(summary or {"ticker": ticker})
        elif is_insufficient_data_error(exc):
            ticker_results = []
            add_skip_result(
                ticker_results,
                "system",
                ticker,
                "validation_runtime",
                f"資料不足，跳過驗證。({type(exc).__name__}: {exc})",
            )
            ticker_summary = {
                "ticker": ticker,
                "validation_runtime": f"SKIP: {format_exception_summary(exc)}",
            }
        else:
            ticker_results = []
            add_fail_result(
                ticker_results,
                "system",
                ticker,
                "validation_runtime",
                "no exception",
                format_exception_summary(exc),
                "單一 ticker 的 runtime / import side effect / 路徑權限問題，不可讓整體 validate 中斷。",
            )
            ticker_summary = {
                "ticker": ticker,
                "validation_runtime": f"FAIL: {format_exception_summary(exc)}",
            }

        ordered_results[idx] = ticker_results
        ordered_summaries[idx] = ticker_summary
        ticker_statuses = {row["status"] for row in ticker_results}
        if "FAIL" in ticker_statuses:
            ticker_fail_count += 1
        elif "PASS" in ticker_statuses:
            ticker_pass_count += 1
        else:
            ticker_skip_count += 1

        completed_count += 1
        if progress_printer is not None:
            progress_printer(completed_count, total_tickers, ticker, ticker_pass_count, ticker_skip_count, ticker_fail_count)

    if worker_count <= 1:
        for idx, ticker in enumerate(selected_tickers):
            try:
                results, summary = validate_one_ticker(
                    project_root,
                    data_dir,
                    csv_map_getter,
                    ticker,
                    base_params,
                    resolved_file_path=csv_map.get(ticker),
                )
                _record_outcome(idx, ticker, results=results, summary=summary)
            except VALIDATION_RECOVERABLE_EXCEPTIONS as exc:
                _record_outcome(idx, ticker, exc=exc)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_task = {}
            for idx, ticker in enumerate(selected_tickers):
                task = {
                    "project_root": project_root,
                    "data_dir": data_dir,
                    "ticker": ticker,
                    "base_params": base_params,
                    "resolved_file_path": csv_map.get(ticker),
                }
                future = executor.submit(_validate_one_ticker_worker, task)
                future_to_task[future] = (idx, ticker)

            for future in as_completed(future_to_task):
                idx, ticker = future_to_task[future]
                try:
                    results, summary = future.result()
                    _record_outcome(idx, ticker, results=results, summary=summary)
                except Exception as exc:
                    _record_outcome(idx, ticker, exc=exc)

    all_results = []
    summaries = []
    for idx in range(total_tickers):
        all_results.extend(ordered_results[idx] or [])
        if ordered_summaries[idx] is not None:
            summaries.append(ordered_summaries[idx])

    return all_results, summaries, {
        "total_tickers": total_tickers,
        "pass_count": ticker_pass_count,
        "skip_count": ticker_skip_count,
        "fail_count": ticker_fail_count,
        "worker_count": worker_count,
    }
