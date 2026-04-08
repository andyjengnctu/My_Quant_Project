import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from core.dataset_profiles import (
    DEFAULT_DATASET_PROFILE,
    build_empty_dataset_dir_message,
    build_missing_dataset_dir_message,
    get_dataset_dir,
    get_dataset_profile_label,
    infer_dataset_profile_key_from_data_dir,
    resolve_dataset_profile_from_cli_env,
)
from core.display import C_CYAN, C_GRAY, C_GREEN, C_RED, C_RESET, C_YELLOW, print_scanner_header
from core.log_utils import write_issue_log
from core.runtime_utils import enable_line_buffered_stdout, get_process_pool_executor_kwargs, get_taipei_now, has_help_flag, resolve_cli_program_name, validate_cli_args
from .reporting import print_history_qualified_summary, print_scanner_start_banner, print_scanner_summary
from .runtime_common import BEST_PARAMS_PATH, OUTPUT_DIR, PROJECT_ROOT, SCANNER_PROGRESS_EVERY, ensure_runtime_dirs, load_strict_params, resolve_scanner_max_workers


def _prepare_scan_inputs(data_dir, params):
    from core.data_utils import discover_unique_csv_inputs

    if not os.path.exists(data_dir):
        profile_key = infer_dataset_profile_key_from_data_dir(data_dir)
        raise FileNotFoundError(build_missing_dataset_dir_message(profile_key, data_dir))

    csv_inputs, duplicate_file_issue_lines = discover_unique_csv_inputs(data_dir)
    ensure_runtime_dirs()
    print_scanner_start_banner(get_taipei_now().strftime('%Y-%m-%d %H:%M'))
    total_files = len(csv_inputs)
    if total_files == 0:
        profile_key = infer_dataset_profile_key_from_data_dir(data_dir)
        raise FileNotFoundError(build_empty_dataset_dir_message(profile_key, data_dir))

    print(f"{C_GREEN}✅ 成功載入 AI 聖杯參數大腦！{C_RESET}")
    print_scanner_header(params)
    print(f"{C_YELLOW}ℹ️ 本掃描器的投入金額以 scanner_live_capital 作為參考估算，非帳戶級真實可下單金額。{C_RESET}")
    print(f"{C_CYAN}--------------------------------------------------------------------------------{C_RESET}")
    return csv_inputs, duplicate_file_issue_lines, total_files


def _adapt_candidate_scan_result(result):
    if result is None:
        return {"history_qualified": False, "skip_insufficient": False, "row": None, "sanitize_issue": None}

    status, proj_cost, ev, sort_value, msg, ticker, sanitize_issue = result
    history_qualified = status in ['buy', 'extended', 'candidate']
    row = None
    if status in ['buy', 'extended']:
        row = {
            'kind': status,
            'proj_cost': proj_cost,
            'ev': ev,
            'expected_value': ev,
            'sort_value': sort_value,
            'text': msg,
            'ticker': ticker,
            'sanitize_issue': sanitize_issue,
        }
    return {
        "history_qualified": history_qualified,
        "skip_insufficient": status == 'skip_insufficient',
        "row": row,
        "sanitize_issue": sanitize_issue,
    }


def _adapt_history_scan_result(result):
    if result is None:
        return {"history_qualified": False, "skip_insufficient": False, "row": None, "sanitize_issue": None}

    status = result.get('status')
    return {
        "history_qualified": status in ['buy', 'extended', 'candidate'],
        "skip_insufficient": status == 'skip_insufficient',
        "row": result if status in ['buy', 'extended', 'candidate'] else None,
        "sanitize_issue": result.get('sanitize_issue'),
    }


def _format_candidate_progress(count_scanned, total_files, rows):
    return (
        f"{C_GRAY}⏳ 極速運算中: [{count_scanned}/{total_files}] "
        f"新訊號:{sum(1 for x in rows if x['kind'] == 'buy')} | "
        f"延續:{sum(1 for x in rows if x['kind'] == 'extended')}{C_RESET}"
    )


def _format_history_progress(count_scanned, total_files, rows):
    return (
        f"{C_GRAY}⏳ 極速運算中: [{count_scanned}/{total_files}] "
        f"新訊號:{sum(1 for x in rows if x['kind'] == 'buy')} | "
        f"延續:{sum(1 for x in rows if x['kind'] == 'extended')} | "
        f"歷績:{sum(1 for x in rows if x['kind'] == 'candidate')}{C_RESET}"
    )


def _run_parallel_scan(data_dir, params, *, process_single_stock_fn, adapt_result_fn, progress_formatter):
    csv_inputs, duplicate_file_issue_lines, total_files = _prepare_scan_inputs(data_dir, params)

    count_scanned = 0
    count_history_qualified = 0
    count_skipped_insufficient = 0
    count_sanitized_candidates = 0
    rows = []
    scanner_issue_lines = list(duplicate_file_issue_lines)
    start_time = time.time()
    max_workers = resolve_scanner_max_workers(params)
    process_pool_kwargs, pool_start_method = get_process_pool_executor_kwargs()

    with ProcessPoolExecutor(max_workers=max_workers, **process_pool_kwargs) as executor:
        futures = {
            executor.submit(process_single_stock_fn, file_path, ticker, params): file_path
            for ticker, file_path in csv_inputs
        }

        for future in as_completed(futures):
            count_scanned += 1
            adapted = adapt_result_fn(future.result())
            if adapted['history_qualified']:
                count_history_qualified += 1
                if adapted['sanitize_issue'] is not None:
                    count_sanitized_candidates += 1
                    scanner_issue_lines.append(f"[清洗] {adapted['sanitize_issue']}")
            elif adapted['skip_insufficient']:
                count_skipped_insufficient += 1

            if adapted['row'] is not None:
                rows.append(adapted['row'])

            if count_scanned % SCANNER_PROGRESS_EVERY == 0 or count_scanned == total_files:
                print(progress_formatter(count_scanned, total_files, rows), end="\r", flush=True)

    scanner_issue_log_path = write_issue_log("scanner_issues", scanner_issue_lines, log_dir=OUTPUT_DIR) if scanner_issue_lines else None
    elapsed_time = time.time() - start_time
    return {
        'count_scanned': count_scanned,
        'elapsed_time': elapsed_time,
        'count_history_qualified': count_history_qualified,
        'count_skipped_insufficient': count_skipped_insufficient,
        'count_sanitized_candidates': count_sanitized_candidates,
        'max_workers': max_workers,
        'pool_start_method': pool_start_method,
        'rows': rows,
        'scanner_issue_log_path': scanner_issue_log_path,
    }


def run_daily_scanner(data_dir, params):
    from .stock_processor import process_single_stock

    scan_state = _run_parallel_scan(
        data_dir,
        params,
        process_single_stock_fn=process_single_stock,
        adapt_result_fn=_adapt_candidate_scan_result,
        progress_formatter=_format_candidate_progress,
    )
    candidate_rows = list(scan_state['rows'])
    print_scanner_summary(
        count_scanned=scan_state['count_scanned'],
        elapsed_time=scan_state['elapsed_time'],
        count_history_qualified=scan_state['count_history_qualified'],
        count_skipped_insufficient=scan_state['count_skipped_insufficient'],
        count_sanitized_candidates=scan_state['count_sanitized_candidates'],
        max_workers=scan_state['max_workers'],
        pool_start_method=scan_state['pool_start_method'],
        candidate_rows=candidate_rows,
        scanner_issue_log_path=scan_state['scanner_issue_log_path'],
    )
    return {
        'count_scanned': scan_state['count_scanned'],
        'elapsed_time': scan_state['elapsed_time'],
        'count_history_qualified': scan_state['count_history_qualified'],
        'count_skipped_insufficient': scan_state['count_skipped_insufficient'],
        'count_sanitized_candidates': scan_state['count_sanitized_candidates'],
        'max_workers': scan_state['max_workers'],
        'pool_start_method': scan_state['pool_start_method'],
        'candidate_rows': list(candidate_rows),
        'scanner_issue_log_path': scan_state['scanner_issue_log_path'],
    }


def run_history_qualified_scanner(data_dir, params):
    from .stock_processor import process_single_stock_history_qualified

    scan_state = _run_parallel_scan(
        data_dir,
        params,
        process_single_stock_fn=process_single_stock_history_qualified,
        adapt_result_fn=_adapt_history_scan_result,
        progress_formatter=_format_history_progress,
    )
    print_history_qualified_summary(
        count_scanned=scan_state['count_scanned'],
        elapsed_time=scan_state['elapsed_time'],
        count_history_qualified=scan_state['count_history_qualified'],
        count_skipped_insufficient=scan_state['count_skipped_insufficient'],
        count_sanitized_candidates=scan_state['count_sanitized_candidates'],
        max_workers=scan_state['max_workers'],
        pool_start_method=scan_state['pool_start_method'],
        history_qualified_rows=scan_state['rows'],
        scanner_issue_log_path=scan_state['scanner_issue_log_path'],
    )
    return {
        'count_scanned': scan_state['count_scanned'],
        'elapsed_time': scan_state['elapsed_time'],
        'count_history_qualified': scan_state['count_history_qualified'],
        'count_skipped_insufficient': scan_state['count_skipped_insufficient'],
        'count_sanitized_candidates': scan_state['count_sanitized_candidates'],
        'max_workers': scan_state['max_workers'],
        'pool_start_method': scan_state['pool_start_method'],
        'history_qualified_rows': list(scan_state['rows']),
        'scanner_issue_log_path': scan_state['scanner_issue_log_path'],
    }


def main(argv=None, env=None):
    enable_line_buffered_stdout()
    import sys
    argv = sys.argv if argv is None else argv
    env = os.environ if env is None else env
    validate_cli_args(argv, value_options=("--dataset",))
    if has_help_flag(argv):
        program_name = resolve_cli_program_name(argv, "tools/scanner/scan_runner.py")
        print(f"用法: python {program_name} [--dataset reduced|full]")
        print("說明: 預設資料集為完整；縮減資料集路徑為 <repo>/data/tw_stock_data_vip_reduced。")
        return 0
    from core.data_utils import discover_unique_csv_inputs

    try:
        dataset_profile_key, dataset_source = resolve_dataset_profile_from_cli_env(
            argv,
            env,
            default=DEFAULT_DATASET_PROFILE,
        )
        selected_data_dir = get_dataset_dir(PROJECT_ROOT, dataset_profile_key)
        if not os.path.isdir(selected_data_dir):
            raise FileNotFoundError(build_missing_dataset_dir_message(dataset_profile_key, selected_data_dir))
        csv_inputs, _ = discover_unique_csv_inputs(selected_data_dir)
        if not csv_inputs:
            raise FileNotFoundError(build_empty_dataset_dir_message(dataset_profile_key, selected_data_dir))
        params = load_strict_params(BEST_PARAMS_PATH)
        print(
            f"{C_GRAY}📁 使用資料集: {get_dataset_profile_label(dataset_profile_key)} | "
            f"來源: {dataset_source} | 路徑: {selected_data_dir}{C_RESET}"
        )
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        import sys
        print(f"{C_RED}❌ {e}{C_RESET}", file=sys.stderr)
        return 1
    try:
        run_daily_scanner(selected_data_dir, params)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"{C_RED}❌ {exc}{C_RESET}", file=sys.stderr)
        return 1
    return 0
