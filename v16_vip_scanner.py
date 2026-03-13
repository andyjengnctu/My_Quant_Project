import os
import json
import pandas as pd
import numpy as np 
import warnings
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from core.v16_config import V16StrategyParams, BUY_SORT_METHOD
from core.v16_core import run_v16_backtest, calc_position_size, calc_entry_price, adjust_long_target_price, calc_net_sell_price
from core.v16_display import print_scanner_header, C_RED, C_YELLOW, C_CYAN, C_GREEN, C_GRAY, C_RESET
from core.v16_data_utils import sanitize_ohlcv_dataframe, get_required_min_rows, discover_unique_csv_inputs
from core.v16_log_utils import write_issue_log, format_exception_summary
from core.v16_buy_sort import calc_buy_sort_value, get_buy_sort_title

# # (AI註: 收窄 warning 範圍；預設保留 warning，可疑資料與數值問題不要被全域吃掉)
warnings.simplefilter("default")

# # (AI註: 相同 RuntimeWarning 只顯示一次；保留可見性，但避免掃描輸出被重複洗版)
warnings.filterwarnings("once", category=RuntimeWarning)

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

SCANNER_PROGRESS_EVERY = 25
DEFAULT_SCANNER_MAX_WORKERS = min(8, max(1, (os.cpu_count() or 1) // 2))

def resolve_scanner_max_workers(params):
    configured = getattr(params, 'scanner_max_workers', DEFAULT_SCANNER_MAX_WORKERS)
    try:
        configured = int(configured)
    except (TypeError, ValueError):
        configured = DEFAULT_SCANNER_MAX_WORKERS
    return max(1, configured)

def load_dynamic_params(json_file):
    params = V16StrategyParams()
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"找不到參數檔: {json_file}")

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for key, value in data.items():
            if hasattr(params, key):
                setattr(params, key, value)
        return params, True
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as e:
        raise RuntimeError(f"讀取參數檔 {json_file} 失敗: {format_exception_summary(e)}") from e


# # (AI註: 將「清洗後有效資料不足」與真正異常分流，避免 scanner 被新上市/短歷史標的洗板)
def is_insufficient_data_error(exc):
    return isinstance(exc, ValueError) and ("有效資料不足" in str(exc))

def process_single_stock(file_path, ticker, params):
    try:
        raw_df = pd.read_csv(file_path)
        min_rows_needed = get_required_min_rows(params)
        df, sanitize_stats = sanitize_ohlcv_dataframe(raw_df, ticker, min_rows=min_rows_needed)

        invalid_row_count = sanitize_stats['invalid_row_count']
        duplicate_date_count = sanitize_stats['duplicate_date_count']
        dropped_row_count = sanitize_stats['dropped_row_count']

        stats = run_v16_backtest(df, params)

        if not stats or not stats['is_candidate']:
            return None

        sanitize_issue = None
        if dropped_row_count > 0:
            sanitize_issue = (
                f"{ticker}: 清洗移除 {dropped_row_count} 列 "
                f"(異常OHLCV={invalid_row_count}, 重複日期={duplicate_date_count})"
            )

        stat_str = f"勝率:{stats['win_rate']:>5.1f}% | 期望值:{stats['expected_value']:>5.2f}R | 交易:{stats['trade_count']:>3}次 | MDD:{stats['max_drawdown']:>5.1f}%"
        reference_capital = params.initial_capital

        if stats['is_setup_today']:
            proj_qty = calc_position_size(stats['buy_limit'], stats['stop_loss'], reference_capital, params.fixed_risk, params)
            if proj_qty == 0:
                return ('candidate', None, None, None, None, ticker, sanitize_issue)

            proj_cost = calc_entry_price(stats['buy_limit'], proj_qty, params) * proj_qty

            actual_cost_per_share = calc_entry_price(stats['buy_limit'], proj_qty, params)
            net_sl_per_share = calc_net_sell_price(stats['stop_loss'], proj_qty, params)
            est_target = adjust_long_target_price(stats['buy_limit'] + (actual_cost_per_share - net_sl_per_share))

            buy_str = f"限價買進:{stats['buy_limit']:>6.2f} | 停損:{stats['stop_loss']:>6.2f} | 停利(預估):{est_target:>6.2f} | 參考投入:{proj_cost:>7,.0f}"
            msg = f"[🚨 最新買訊] {ticker:<6} | {stat_str} | {buy_str}"

            sort_value = calc_buy_sort_value(BUY_SORT_METHOD, stats['expected_value'], proj_cost, stats['win_rate'] / 100.0, stats['trade_count'])
            return ('buy', proj_cost, stats['expected_value'], sort_value, msg, ticker, sanitize_issue)

        elif stats.get('chase_today') is not None:
            chase = stats['chase_today']
            proj_qty = chase['qty']
            if proj_qty == 0:
                return ('candidate', None, None, None, None, ticker, sanitize_issue)
            proj_cost = calc_entry_price(chase['chase_price'], proj_qty, params) * proj_qty

            zone_str = f"追買限價:{chase['chase_price']:>6.2f} | 停損:{chase['sl']:>6.2f} | 盈虧比:{chase['rr']:>4.2f} | 參考投入:{proj_cost:>7,.0f}"
            msg = f"[⚠️ 遲到補車 (精準1R縮倉)] {ticker:<5} | {stat_str} | {zone_str}"
            sort_value = calc_buy_sort_value(BUY_SORT_METHOD, stats['expected_value'], proj_cost, stats['win_rate'] / 100.0, stats['trade_count'])
            return ('zone', proj_cost, stats['expected_value'], sort_value, msg, ticker, sanitize_issue)

        return ('candidate', None, None, None, None, ticker, sanitize_issue)

    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError, KeyError, IndexError, TypeError, RuntimeError) as e:
        if is_insufficient_data_error(e):
            return ('skip_insufficient', None, None, None, None, ticker, None)
        raise RuntimeError(
            f"scanner 處理失敗: ticker={ticker} | {format_exception_summary(e)}"
        ) from e
    
def run_daily_scanner(data_dir):
    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(f"{C_CYAN}🚀 啟動【v16 尊爵版】極速平行掃描儀 | 時間: {datetime.now().strftime('%Y-%m-%d %H:%M')}{C_RESET}")
    print(f"{C_CYAN}================================================================================{C_RESET}")
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"找不到資料夾 {data_dir}。")

    csv_inputs, duplicate_file_issue_lines = discover_unique_csv_inputs(data_dir)
    total_files = len(csv_inputs)
    if total_files == 0:
        raise FileNotFoundError(f"資料夾 {data_dir} 內沒有任何 CSV 檔案。")

    params, is_loaded = load_dynamic_params("models/v16_best_params.json")
    if is_loaded:
        print(f"{C_GREEN}✅ 成功載入 AI 聖杯參數大腦！{C_RESET}")
        
    print_scanner_header(params)
    print(f"{C_YELLOW}ℹ️ 本掃描器的投入金額僅以 initial_capital 作為參考估算，非帳戶級真實可下單金額。{C_RESET}")
    print(f"{C_CYAN}--------------------------------------------------------------------------------{C_RESET}")

    count_scanned, count_history_qualified = 0, 0
    count_skipped_insufficient = 0
    count_sanitized_candidates = 0
    buy_list, in_zone_list = [], []
    scanner_issue_lines = list(duplicate_file_issue_lines)
    start_time = time.time()
    max_workers = resolve_scanner_max_workers(params)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_stock,
                file_path,
                ticker,
                params
            ): file_path for ticker, file_path in csv_inputs
        }

        for future in as_completed(futures):
            count_scanned += 1

            result = future.result()
            if result and len(result) == 7:
                status, proj_cost, ev, sort_value, msg, ticker, sanitize_issue = result

                if status in ['buy', 'zone', 'candidate']:
                    count_history_qualified += 1
                    if sanitize_issue is not None:
                        count_sanitized_candidates += 1
                        scanner_issue_lines.append(f"[清洗] {sanitize_issue}")
                elif status == 'skip_insufficient':
                    count_skipped_insufficient += 1

                if status == 'buy':
                    buy_list.append({'proj_cost': proj_cost, 'ev': ev, 'sort_value': sort_value, 'text': msg, 'ticker': ticker})
                elif status == 'zone':
                    in_zone_list.append({'proj_cost': proj_cost, 'ev': ev, 'sort_value': sort_value, 'text': msg, 'ticker': ticker})

            if count_scanned % SCANNER_PROGRESS_EVERY == 0 or count_scanned == total_files:
                print(
                    f"{C_GRAY}⏳ 極速運算中: [{count_scanned}/{total_files}] "
                    f"最新買訊:{len(buy_list)} | 追車:{len(in_zone_list)}{C_RESET}",
                    end="\r",
                    flush=True
                )

    scanner_issue_log_path = write_issue_log("scanner_issues", scanner_issue_lines) if scanner_issue_lines else None
    elapsed_time = time.time() - start_time

    buy_list.sort(key=lambda x: (x['sort_value'], x['ticker']), reverse=True)
    in_zone_list.sort(key=lambda x: (x['sort_value'], x['ticker']), reverse=True)
    sort_title = get_buy_sort_title(BUY_SORT_METHOD)

    print(" " * 160, end="\r")
    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(
        f"⚡ 掃描完畢！共掃描 {count_scanned} 檔標的，耗時 {elapsed_time:.2f} 秒。"
        f"歷史及格候選: {count_history_qualified} 檔 | 資料不足跳過: {count_skipped_insufficient} 檔 | "
        f"候選清洗: {count_sanitized_candidates} 檔 | max_workers: {max_workers}"
    )
        
    if buy_list or in_zone_list:
        print(f"\n{C_RED}🔥 【第一優先：明日掛單清單 (最新買訊)】 {sort_title} 🔥{C_RESET}")
        if buy_list:
            for item in buy_list: print(f"   {C_RED}➤ {item['text']}{C_RESET}")
        else: print(f"   {C_RED}無最新買訊。{C_RESET}")

        print(f"\n{C_YELLOW}⚠️ 【第二優先：遲到安全追車清單 (已精準還原1R且縮倉)】 {sort_title} ⚠️{C_RESET}")
        if in_zone_list:
            for item in in_zone_list: print(f"   {C_YELLOW}➤ {item['text']}{C_RESET}")
        else: print(f"   {C_YELLOW}無符合安全追買條件的標的。{C_RESET}")
    else:
        print(f"\n{C_GREEN}💤 今日無符合實戰買點的標的，保留現金，明日再戰！{C_RESET}")

    if scanner_issue_log_path:
        print(
            f"\n{C_YELLOW}⚠️ 清洗摘要已寫入: {scanner_issue_log_path} "
            f"(候選清洗 {count_sanitized_candidates} 檔){C_RESET}"
        )

    print(f"{C_CYAN}================================================================================{C_RESET}")

if __name__ == "__main__":
    run_daily_scanner("tw_stock_data_vip")