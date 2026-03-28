import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

from core.v16_data_utils import discover_unique_csv_inputs
from core.v16_dataset_profiles import DEFAULT_DATASET_PROFILE, get_dataset_dir, get_dataset_profile_label, resolve_dataset_profile_from_cli_env
from core.v16_display import C_CYAN, C_GRAY, C_GREEN, C_RED, C_RESET, C_YELLOW, print_scanner_header
from core.v16_log_utils import write_issue_log
from core.v16_runtime_utils import get_process_pool_executor_kwargs
from .reporting import print_scanner_start_banner, print_scanner_summary
from .worker import BEST_PARAMS_PATH, OUTPUT_DIR, SCANNER_PROGRESS_EVERY, ensure_runtime_dirs, load_strict_params, process_single_stock, resolve_scanner_max_workers

warnings.simplefilter("default")
warnings.filterwarnings("once", category=RuntimeWarning)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_daily_scanner(data_dir):
    ensure_runtime_dirs()
    print_scanner_start_banner(datetime.now().strftime('%Y-%m-%d %H:%M'))

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

    count_scanned = 0
    count_history_qualified = 0
    count_skipped_insufficient = 0
    count_sanitized_candidates = 0
    candidate_rows = []
    scanner_issue_lines = list(duplicate_file_issue_lines)
    start_time = time.time()
    max_workers = resolve_scanner_max_workers(params)
    process_pool_kwargs, pool_start_method = get_process_pool_executor_kwargs()

    with ProcessPoolExecutor(max_workers=max_workers, **process_pool_kwargs) as executor:
        futures = {
            executor.submit(process_single_stock, file_path, ticker, params): file_path
            for ticker, file_path in csv_inputs
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
    print_scanner_summary(
        count_scanned=count_scanned,
        elapsed_time=elapsed_time,
        count_history_qualified=count_history_qualified,
        count_skipped_insufficient=count_skipped_insufficient,
        count_sanitized_candidates=count_sanitized_candidates,
        max_workers=max_workers,
        pool_start_method=pool_start_method,
        candidate_rows=candidate_rows,
        scanner_issue_log_path=scanner_issue_log_path,
    )


def main(argv=None, env=None):
    argv = sys.argv if argv is None else argv
    env = os.environ if env is None else env
    try:
        dataset_profile_key, dataset_source = resolve_dataset_profile_from_cli_env(
            argv,
            env,
            default=DEFAULT_DATASET_PROFILE,
        )
        selected_data_dir = get_dataset_dir(PROJECT_ROOT, dataset_profile_key)
        print(
            f"{C_GRAY}📁 使用資料集: {get_dataset_profile_label(dataset_profile_key)} | "
            f"來源: {dataset_source} | 路徑: {selected_data_dir}{C_RESET}"
        )
    except ValueError as e:
        print(f"{C_RED}❌ {e}{C_RESET}", file=sys.stderr)
        raise SystemExit(1)
    run_daily_scanner(selected_data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
