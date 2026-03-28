import pandas as pd

from core.data_utils import discover_unique_csv_inputs, get_required_min_rows, sanitize_ohlcv_dataframe
from core.dataset_profiles import build_empty_dataset_dir_message, build_missing_dataset_dir_message
from core.display import C_CYAN, C_GREEN, C_GRAY, C_YELLOW, C_RESET
from core.log_utils import format_exception_summary, write_issue_log
from core.portfolio_engine import run_portfolio_timeline
from core.portfolio_fast_data import pack_prepared_stock_data, prep_stock_data_and_trades
from .runtime_common import LOAD_PROGRESS_EVERY, OUTPUT_DIR, ensure_runtime_dirs, is_insufficient_data_error


def run_portfolio_simulation(data_dir, params, max_positions=5, enable_rotation=False, start_year=2015, benchmark_ticker="0050", verbose=True):
    ensure_runtime_dirs()
    if not data_dir:
        profile_key = "reduced" if os.path.basename(os.path.normpath(data_dir)) == "tw_stock_data_vip_reduced" else "full"
        raise FileNotFoundError(build_missing_dataset_dir_message(profile_key, data_dir))
    import os
    if not os.path.exists(data_dir):
        profile_key = "reduced" if os.path.basename(os.path.normpath(data_dir)) == "tw_stock_data_vip_reduced" else "full"
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

            df, logs = prep_stock_data_and_trades(df, params)
            master_dates.update(df.index)
            all_dfs_fast[ticker] = pack_prepared_stock_data(df)
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
                flush=True
            )

    load_log_path = write_issue_log("portfolio_sim_load_issues", load_issue_lines, log_dir=OUTPUT_DIR) if load_issue_lines else None

    vprint(" " * 160, end="\r")

    if load_log_path:
        vprint(f"{C_YELLOW}⚠️ 預載入摘要已寫入: {load_log_path}{C_RESET}")

    if not all_dfs_fast:
        raise RuntimeError("未能成功載入任何股票資料！")

    sorted_dates = sorted(list(master_dates))

    vprint(
        f"\n{C_GREEN}✅ 預處理完成！共載入 {len(all_dfs_fast)} 檔標的，"
        f"移除 {total_dropped_rows} 列資料 "
        f"(異常OHLCV={total_invalid_rows}, 重複日期={total_duplicate_dates})，"
        f"候選清洗 {total_sanitize_issue_tickers} 檔，"
        f"資料不足跳過 {total_skipped_insufficient} 檔。"
        f"自 {start_year} 年開始啟動真實時間軸回測...{C_RESET}\n"
    )

    benchmark_data = all_dfs_fast.get(benchmark_ticker, None)
    vprint(" " * 120, end="\r")

    pf_profile = {}
    result = run_portfolio_timeline(
        all_dfs_fast,
        all_trade_logs,
        sorted_dates,
        start_year,
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
