import pandas as pd

from core.portfolio_engine import run_portfolio_timeline
from core.portfolio_fast_data import get_fast_dates, pack_prepared_stock_data, prep_stock_data_and_trades

from tools.validate.checks import (
    add_fail_result,
    add_skip_result,
    build_execution_only_params,
    build_scanner_validation_params,
    extract_yearly_profile_fields,
    make_consistency_params,
    run_scanner_reference_check,
)
from tools.validate.real_case_assertions import append_real_case_checks
from tools.validate.real_case_io import load_clean_df
from tools.validate.tool_adapters import (
    VALIDATION_RECOVERABLE_EXCEPTIONS,
    run_debug_trade_log_check,
    run_downloader_tool_check,
    run_portfolio_sim_tool_check,
    run_scanner_tool_check,
)


def run_single_backtest_check(df, params):
    prep_df, standalone_logs, stats = prep_stock_data_and_trades(df.copy(), params, return_stats=True)
    return stats, standalone_logs, prep_df


def run_single_ticker_portfolio_check(ticker, prep_df, standalone_logs, params):
    execution_params = build_execution_only_params(params)

    fast_data = pack_prepared_stock_data(prep_df)
    all_dfs_fast = {ticker: fast_data}
    all_standalone_logs = {ticker: standalone_logs}

    sorted_dates = sorted(get_fast_dates(fast_data))
    if not sorted_dates:
        raise ValueError(f"{ticker}: pack_prepared_stock_data 後沒有任何有效日期")

    start_year = int(pd.Timestamp(sorted_dates[0]).year)
    profile_stats = {}

    result = run_portfolio_timeline(
        all_dfs_fast=all_dfs_fast,
        all_standalone_logs=all_standalone_logs,
        sorted_dates=sorted_dates,
        start_year=start_year,
        params=execution_params,
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


def validate_one_ticker(project_root, data_dir, csv_map_getter, ticker, base_params):
    params = make_consistency_params(base_params)
    scanner_params = build_scanner_validation_params(base_params)
    file_path, df, sanitize_stats = load_clean_df(project_root, data_dir, csv_map_getter, ticker, params)

    results = []
    summary = {
        "ticker": ticker,
        "file_path": file_path,
        "sanitize_dropped": sanitize_stats["dropped_row_count"],
        "sanitize_invalid": sanitize_stats["invalid_row_count"],
        "sanitize_duplicate": sanitize_stats["duplicate_date_count"],
    }

    single_stats, standalone_logs, prep_df = run_single_backtest_check(df, params)
    scanner_ref_stats = run_scanner_reference_check(ticker, file_path, scanner_params)
    portfolio_stats = run_single_ticker_portfolio_check(ticker, prep_df, standalone_logs, params)
    portfolio_sim_stats = run_portfolio_sim_tool_check(ticker, file_path, params)
    scanner_result, scanner_module_path = run_scanner_tool_check(ticker, file_path, scanner_params)
    downloader_df, downloader_module_path, downloader_request, downloader_expected_dataset = None, None, None, None
    downloader_error = None
    try:
        downloader_df, downloader_module_path, downloader_request, downloader_expected_dataset = run_downloader_tool_check(ticker)
    except VALIDATION_RECOVERABLE_EXCEPTIONS as e:
        downloader_error = f"{type(e).__name__}: {e}"
    debug_df, debug_module_path = run_debug_trade_log_check(ticker, df, params)

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
    all_results = []
    summaries = []
    ticker_pass_count = 0
    ticker_skip_count = 0
    ticker_fail_count = 0

    total_tickers = len(selected_tickers)
    for idx, ticker in enumerate(selected_tickers, start=1):
        ticker_results_before = len(all_results)

        try:
            results, summary = validate_one_ticker(project_root, data_dir, csv_map_getter, ticker, base_params)
            all_results.extend(results)
            summaries.append(summary)
        except VALIDATION_RECOVERABLE_EXCEPTIONS as e:
            if is_insufficient_data_error(e):
                add_skip_result(
                    all_results,
                    "system",
                    ticker,
                    "validation_runtime",
                    f"資料不足，跳過驗證。({type(e).__name__}: {e})",
                )
                summaries.append({
                    "ticker": ticker,
                    "validation_runtime": f"SKIP: {format_exception_summary(e)}",
                })
            else:
                add_fail_result(
                    all_results,
                    "system",
                    ticker,
                    "validation_runtime",
                    "no exception",
                    format_exception_summary(e),
                    "單一 ticker 的 runtime / import side effect / 路徑權限問題，不可讓整體 validate 中斷。",
                )
                summaries.append({
                    "ticker": ticker,
                    "validation_runtime": f"FAIL: {format_exception_summary(e)}",
                })

        ticker_results = all_results[ticker_results_before:]
        ticker_statuses = {row["status"] for row in ticker_results}

        if "FAIL" in ticker_statuses:
            ticker_fail_count += 1
        elif "PASS" in ticker_statuses:
            ticker_pass_count += 1
        else:
            ticker_skip_count += 1

        if progress_printer is not None:
            progress_printer(idx, total_tickers, ticker, ticker_pass_count, ticker_skip_count, ticker_fail_count)

    return all_results, summaries, {
        "total_tickers": total_tickers,
        "pass_count": ticker_pass_count,
        "skip_count": ticker_skip_count,
        "fail_count": ticker_fail_count,
    }
