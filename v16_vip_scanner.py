import os
import pandas as pd
import warnings
import time
from core.v16_params_io import load_params_from_json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from core.v16_config import BUY_SORT_METHOD
from core.v16_core import run_v16_backtest, calc_reference_candidate_qty, calc_entry_price, adjust_long_target_price, calc_net_sell_price, can_execute_half_take_profit
from core.v16_display import print_scanner_header, C_RED, C_YELLOW, C_CYAN, C_GREEN, C_GRAY, C_RESET
from core.v16_data_utils import sanitize_ohlcv_dataframe, get_required_min_rows, discover_unique_csv_inputs
from core.v16_log_utils import write_issue_log, format_exception_summary
from core.v16_runtime_utils import get_process_pool_executor_kwargs
from core.v16_buy_sort import calc_buy_sort_value, get_buy_sort_title

# # (AI註: 收窄 warning 範圍；預設保留 warning，可疑資料與數值問題不要被全域吃掉)
warnings.simplefilter("default")

# # (AI註: 相同 RuntimeWarning 只顯示一次；保留可見性，但避免掃描輸出被重複洗版)
warnings.filterwarnings("once", category=RuntimeWarning)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "tw_stock_data_vip")
BEST_PARAMS_PATH = os.path.join(MODELS_DIR, "v16_best_params.json")


# # (AI註: 將目錄建立延後到實際執行期，避免被 import 時污染呼叫端工作目錄)
def ensure_runtime_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


SCANNER_PROGRESS_EVERY = 25
DEFAULT_SCANNER_MAX_WORKERS = min(8, max(1, (os.cpu_count() or 1) // 2))

def resolve_scanner_max_workers(params):
    configured = getattr(params, 'scanner_max_workers', DEFAULT_SCANNER_MAX_WORKERS)
    try:
        configured = int(configured)
    except (TypeError, ValueError):
        configured = DEFAULT_SCANNER_MAX_WORKERS
    return max(1, configured)

def load_strict_params(json_file):
    return load_params_from_json(json_file)

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

        if stats['is_setup_today']:
            proj_qty = calc_reference_candidate_qty(stats['buy_limit'], stats['stop_loss'], params)
            if proj_qty == 0:
                return ('candidate', None, None, None, None, ticker, sanitize_issue)

            proj_cost = calc_entry_price(stats['buy_limit'], proj_qty, params) * proj_qty

            if can_execute_half_take_profit(proj_qty, params.tp_percent):
                actual_cost_per_share = calc_entry_price(stats['buy_limit'], proj_qty, params)
                net_sl_per_share = calc_net_sell_price(stats['stop_loss'], proj_qty, params)
                est_target = adjust_long_target_price(stats['buy_limit'] + (actual_cost_per_share - net_sl_per_share))
                buy_str = f"限價買進:{stats['buy_limit']:>6.2f} | 停損:{stats['stop_loss']:>6.2f} | 停利(預估):{est_target:>6.2f} | 參考投入:{proj_cost:>7,.0f}"
            elif params.tp_percent <= 0:
                buy_str = f"限價買進:{stats['buy_limit']:>6.2f} | 停損:{stats['stop_loss']:>6.2f} | 半倉停利:關閉 | 參考投入:{proj_cost:>7,.0f}"
            else:
                buy_str = f"限價買進:{stats['buy_limit']:>6.2f} | 停損:{stats['stop_loss']:>6.2f} | 半倉停利:股數不足 | 參考投入:{proj_cost:>7,.0f}"
            msg = f"{ticker:<6} | {stat_str} | {buy_str}"

            sort_value = calc_buy_sort_value(BUY_SORT_METHOD, stats['expected_value'], proj_cost, stats['win_rate'] / 100.0, stats['trade_count'])
            return ('buy', proj_cost, stats['expected_value'], sort_value, msg, ticker, sanitize_issue)

        extended_candidate = stats.get('extended_candidate_today')
        if extended_candidate is not None:
            limit_price = extended_candidate.get('limit_price')
            init_sl = extended_candidate.get('init_sl')
            if limit_price is None or init_sl is None:
                raise KeyError("extended candidate 缺少 limit_price/init_sl")

            proj_qty = calc_reference_candidate_qty(limit_price, init_sl, params)
            if proj_qty == 0:
                return ('candidate', None, None, None, None, ticker, sanitize_issue)

            proj_cost = calc_entry_price(limit_price, proj_qty, params) * proj_qty

            extended_str = f"延續限價:{limit_price:>6.2f} | 停損:{init_sl:>6.2f} | 參考投入:{proj_cost:>7,.0f}"
            msg = f"{ticker:<6} | {stat_str} | {extended_str}"
            sort_value = calc_buy_sort_value(
                BUY_SORT_METHOD,
                stats['expected_value'],
                proj_cost,
                stats['win_rate'] / 100.0,
                stats['trade_count']
            )
            return ('extended', proj_cost, stats['expected_value'], sort_value, msg, ticker, sanitize_issue)

        return ('candidate', None, None, None, None, ticker, sanitize_issue)

    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError, KeyError, IndexError, TypeError, RuntimeError) as e:
        if is_insufficient_data_error(e):
            return ('skip_insufficient', None, None, None, None, ticker, None)
        raise RuntimeError(
            f"scanner 處理失敗: ticker={ticker} | {format_exception_summary(e)}"
        ) from e
    
def run_daily_scanner(data_dir=None):
    ensure_runtime_dirs()
    data_dir = DEFAULT_DATA_DIR if data_dir is None else data_dir
    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(f"{C_CYAN}🚀 啟動【v16 尊爵版】極速平行掃描儀 | 時間: {datetime.now().strftime('%Y-%m-%d %H:%M')}{C_RESET}")
    print(f"{C_CYAN}================================================================================{C_RESET}")
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"找不到資料夾 {data_dir}。")

    csv_inputs, duplicate_file_issue_lines = discover_unique_csv_inputs(data_dir)
    total_files = len(csv_inputs)
    if total_files == 0:
        raise FileNotFoundError(f"資料夾 {data_dir} 內沒有任何 CSV 檔案。")

    params = load_strict_params(BEST_PARAMS_PATH)
    print(f"{C_GREEN}✅ 成功載入 AI 聖杯參數大腦！{C_RESET}")
        
    print_scanner_header(params)
    print(f"{C_YELLOW}ℹ️ 本掃描器的投入金額僅以 initial_capital 作為參考估算，非帳戶級真實可下單金額。{C_RESET}")
    print(f"{C_CYAN}--------------------------------------------------------------------------------{C_RESET}")

    count_scanned, count_history_qualified = 0, 0
    count_skipped_insufficient = 0
    count_sanitized_candidates = 0
    candidate_rows = []
    scanner_issue_lines = list(duplicate_file_issue_lines)
    start_time = time.time()
    max_workers = resolve_scanner_max_workers(params)

    process_pool_kwargs, pool_start_method = get_process_pool_executor_kwargs()

    with ProcessPoolExecutor(max_workers=max_workers, **process_pool_kwargs) as executor:
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

                if status in ['buy', 'extended', 'candidate']:
                    count_history_qualified += 1
                    if sanitize_issue is not None:
                        count_sanitized_candidates += 1
                        scanner_issue_lines.append(f"[清洗] {sanitize_issue}")
                elif status == 'skip_insufficient':
                    count_skipped_insufficient += 1

                if status in ['buy', 'extended']:
                    candidate_rows.append({'kind': status, 'proj_cost': proj_cost, 'ev': ev, 'sort_value': sort_value, 'text': msg, 'ticker': ticker})

            if count_scanned % SCANNER_PROGRESS_EVERY == 0 or count_scanned == total_files:
                print(
                    f"{C_GRAY}⏳ 極速運算中: [{count_scanned}/{total_files}] "
                    f"新訊號:{sum(1 for x in candidate_rows if x['kind'] == 'buy')} | 延續:{sum(1 for x in candidate_rows if x['kind'] == 'extended')}{C_RESET}",
                    end="\r",
                    flush=True
                )

    scanner_issue_log_path = write_issue_log("scanner_issues", scanner_issue_lines, log_dir=OUTPUT_DIR) if scanner_issue_lines else None
    elapsed_time = time.time() - start_time

    candidate_rows.sort(key=lambda x: (x['sort_value'], x['ticker']), reverse=True)
    sort_title = get_buy_sort_title(BUY_SORT_METHOD)

    print(" " * 160, end="\r")
    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(
        f"⚡ 掃描完畢！共掃描 {count_scanned} 檔標的，耗時 {elapsed_time:.2f} 秒。"
        f"歷史及格候選: {count_history_qualified} 檔 | 資料不足跳過: {count_skipped_insufficient} 檔 | "
        f"候選清洗: {count_sanitized_candidates} 檔 | max_workers: {max_workers}"
        f" | start_method: {pool_start_method or 'default'}"
    )
        
    if candidate_rows:
        new_count = sum(1 for x in candidate_rows if x['kind'] == 'buy')
        extended_count = sum(1 for x in candidate_rows if x['kind'] == 'extended')
        print(f"\n{C_RED}🔥 【明日候選清單：新訊號 + 延續候選同池排序】 {sort_title} 🔥{C_RESET}")
        print(f"{C_GRAY}   顏色區分：{C_RED}紅色=新訊號{C_GRAY} | {C_YELLOW}黃色=延續候選{C_RESET}")
        print(f"{C_GRAY}   候選統計：新訊號 {new_count} 檔 | 延續候選 {extended_count} 檔{C_RESET}")
        for item in candidate_rows:
            prefix = "[新訊號]" if item['kind'] == 'buy' else "[延續候選]"
            color = C_RED if item['kind'] == 'buy' else C_YELLOW
            print(f"   {color}➤ {prefix} {item['text']}{C_RESET}")
    else:
        print(f"\n{C_GREEN}💤 今日無符合實戰買點的標的，保留現金，明日再戰！{C_RESET}")

    if scanner_issue_log_path:
        print(
            f"\n{C_YELLOW}⚠️ 清洗摘要已寫入: {scanner_issue_log_path} "
            f"(候選清洗 {count_sanitized_candidates} 檔){C_RESET}"
        )

    print(f"{C_CYAN}================================================================================{C_RESET}")

if __name__ == "__main__":
    run_daily_scanner()