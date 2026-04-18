import os

import pandas as pd

from core.data_utils import discover_unique_csv_inputs, get_required_min_rows, sanitize_ohlcv_dataframe
from core.dataset_profiles import (
    build_missing_dataset_dir_message,
    infer_dataset_profile_key_from_data_dir,
)
from core.display import C_CYAN, C_GREEN, C_GRAY, C_YELLOW, C_RESET
from core.log_utils import format_exception_summary, write_issue_log
from core.walk_forward_policy import load_walk_forward_policy
from core.portfolio_engine import run_portfolio_timeline
from core.portfolio_fast_data import pack_prepared_stock_data, prep_stock_data_and_trades
from .runtime_common import LOAD_PROGRESS_EVERY, OUTPUT_DIR, PROJECT_ROOT, ensure_runtime_dirs, is_insufficient_data_error

PORTFOLIO_DEFAULT_BENCHMARK_TICKER = "0050"


def resolve_default_portfolio_start_year() -> int:
    policy = load_walk_forward_policy(PROJECT_ROOT)
    return int(policy["train_start_year"])


def load_portfolio_market_context(data_dir, params, *, verbose=True):
    ensure_runtime_dirs()
    if not data_dir:
        profile_key = infer_dataset_profile_key_from_data_dir(data_dir)
        raise FileNotFoundError(build_missing_dataset_dir_message(profile_key, data_dir))
    if not os.path.exists(data_dir):
        profile_key = infer_dataset_profile_key_from_data_dir(data_dir)
        raise FileNotFoundError(build_missing_dataset_dir_message(profile_key, data_dir))

    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    vprint(f"{C_CYAN}📦 正在預載入歷史軌跡，構建真實時間軸...{C_RESET}")
    all_dfs_fast, all_trade_logs, master_dates = {}, {}, set()
    load_issue_lines = []
    total_invalid_rows = 0
    total_duplicate_dates = 0
    total_dropped_rows = 0
    total_skipped_insufficient = 0
    total_sanitize_issue_tickers = 0

    csv_inputs, duplicate_file_issue_lines = discover_unique_csv_inputs(data_dir)
    load_issue_lines.extend(duplicate_file_issue_lines)
    total_files = len(csv_inputs)

    for count, (ticker, file_path) in enumerate(csv_inputs, start=1):
        try:
            raw_df = pd.read_csv(file_path)
            min_rows_needed = get_required_min_rows(params)

            if len(raw_df) < min_rows_needed:
                total_skipped_insufficient += 1
                load_issue_lines.append(
                    f"[資料不足] {ticker}: 原始資料列數不足 ({len(raw_df)})，至少需要 {min_rows_needed} 筆"
                )
                continue

            df, sanitize_stats = sanitize_ohlcv_dataframe(raw_df, ticker, min_rows=min_rows_needed)
            invalid_row_count = sanitize_stats['invalid_row_count']
            duplicate_date_count = sanitize_stats['duplicate_date_count']
            dropped_row_count = sanitize_stats['dropped_row_count']

            total_invalid_rows += invalid_row_count
            total_duplicate_dates += duplicate_date_count
            total_dropped_rows += dropped_row_count

            if dropped_row_count > 0:
                total_sanitize_issue_tickers += 1
                load_issue_lines.append(
                    f"[清洗] {ticker}: 清洗移除 {dropped_row_count} 列 "
                    f"(異常OHLCV={invalid_row_count}, 重複日期={duplicate_date_count})"
                )

            prep_df, logs = prep_stock_data_and_trades(df, params)
            master_dates.update(prep_df.index)
            all_dfs_fast[ticker] = pack_prepared_stock_data(prep_df)
            all_trade_logs[ticker] = logs

        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError, KeyError, IndexError, TypeError, RuntimeError) as e:
            if is_insufficient_data_error(e):
                total_skipped_insufficient += 1
                load_issue_lines.append(f"[資料不足] {ticker}: {type(e).__name__}: {e}")
                continue
            raise RuntimeError(
                f"預載入失敗: ticker={ticker} | {format_exception_summary(e)}"
            ) from e

        if count % LOAD_PROGRESS_EVERY == 0 or count == total_files:
            vprint(
                f"{C_GRAY}   預載入進度: [{count}/{total_files}] "
                f"成功:{len(all_dfs_fast)} | 資料不足:{total_skipped_insufficient}{C_RESET}",
                end="\r",
                flush=True,
            )

    load_log_path = write_issue_log("portfolio_sim_load_issues", load_issue_lines, log_dir=OUTPUT_DIR) if load_issue_lines else None

    vprint(" " * 160, end="\r")
    if load_log_path:
        vprint(f"{C_YELLOW}⚠️ 預載入摘要已寫入: {load_log_path}{C_RESET}")
    if not all_dfs_fast:
        raise RuntimeError("未能成功載入任何股票資料！")

    sorted_dates = sorted(master_dates)
    vprint(
        f"\n{C_GREEN}✅ 預處理完成！共載入 {len(all_dfs_fast)} 檔標的，"
        f"移除 {total_dropped_rows} 列資料 "
        f"(異常OHLCV={total_invalid_rows}, 重複日期={total_duplicate_dates})，"
        f"候選清洗 {total_sanitize_issue_tickers} 檔，"
        f"資料不足跳過 {total_skipped_insufficient} 檔。{C_RESET}\n"
    )
    return {"all_dfs_fast": all_dfs_fast, "all_trade_logs": all_trade_logs, "sorted_dates": sorted_dates}


def run_portfolio_simulation_prepared(all_dfs_fast, all_trade_logs, sorted_dates, params, max_positions=5, enable_rotation=False, start_year=None, benchmark_ticker=PORTFOLIO_DEFAULT_BENCHMARK_TICKER, verbose=True):
    resolved_start_year = resolve_default_portfolio_start_year() if start_year is None else int(start_year)
    benchmark_data = all_dfs_fast.get(benchmark_ticker, None)
    if verbose:
        print(" " * 120, end="\r")

    pf_profile = {}
    result = run_portfolio_timeline(
        all_dfs_fast,
        all_trade_logs,
        sorted_dates,
        resolved_start_year,
        params,
        max_positions,
        enable_rotation,
        benchmark_ticker=benchmark_ticker,
        benchmark_data=benchmark_data,
        is_training=False,
        profile_stats=pf_profile,
        verbose=verbose,
    )
    return (*result, pf_profile)


def run_portfolio_simulation(data_dir, params, max_positions=5, enable_rotation=False, start_year=None, benchmark_ticker=PORTFOLIO_DEFAULT_BENCHMARK_TICKER, verbose=True):
    context = load_portfolio_market_context(data_dir, params, verbose=verbose)
    return run_portfolio_simulation_prepared(
        context["all_dfs_fast"],
        context["all_trade_logs"],
        context["sorted_dates"],
        params,
        max_positions=max_positions,
        enable_rotation=enable_rotation,
        start_year=start_year,
        benchmark_ticker=benchmark_ticker,
        verbose=verbose,
    )
