import os

import pandas as pd

from core.data_utils import discover_unique_csv_inputs, sanitize_ohlcv_dataframe
from core.display import C_CYAN, C_GRAY, C_GREEN, C_RESET, C_YELLOW
from core.log_utils import format_exception_summary, write_issue_log
from core.dataset_profiles import build_empty_dataset_dir_message, build_missing_dataset_dir_message


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
        profile_key = "reduced" if os.path.basename(os.path.normpath(data_dir)) == "tw_stock_data_vip_reduced" else "full"
        raise FileNotFoundError(build_missing_dataset_dir_message(profile_key, data_dir))

    print(f"{C_CYAN}📦 正在將歷史數據載入記憶體快取 (僅需執行一次)...{C_RESET}")
    csv_inputs, duplicate_file_issue_lines = discover_unique_csv_inputs(data_dir)
    if not csv_inputs:
        profile_key = "reduced" if os.path.basename(os.path.normpath(data_dir)) == "tw_stock_data_vip_reduced" else "full"
        raise FileNotFoundError(build_empty_dataset_dir_message(profile_key, data_dir))

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
