import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.dataset_profiles import (
    DEFAULT_VALIDATE_DATASET_PROFILE,
    VALIDATE_DATASET_ENV_VAR,
    build_validate_dataset_prompt,
    extract_dataset_cli_value,
    get_dataset_dir,
    get_dataset_profile_label,
    normalize_dataset_profile_key,
    build_missing_dataset_dir_message,
    build_empty_dataset_dir_message,
)
from core.log_utils import format_exception_summary
from core.params_io import load_params_from_json
from core.runtime_utils import PeakTracedMemoryTracker, run_cli_entrypoint, enable_line_buffered_stdout, get_taipei_now, has_help_flag, is_interactive_stdin, resolve_cli_program_name, safe_prompt, validate_cli_args
from core.output_paths import build_output_dir

OUTPUT_DIR = build_output_dir(PROJECT_ROOT, "validate_consistency")
DATA_DIR = get_dataset_dir(PROJECT_ROOT, DEFAULT_VALIDATE_DATASET_PROFILE)
PARAMS_FILE = os.path.join(PROJECT_ROOT, "models", "best_params.json")
MAX_CONSOLE_FAIL_PREVIEW = 20

CSV_PATH_CACHE = None
CSV_DUPLICATE_ISSUES = None
CSV_PATH_CACHE_DATA_DIR = None


def get_data_dir_csv_map():
    global CSV_PATH_CACHE, CSV_DUPLICATE_ISSUES, CSV_PATH_CACHE_DATA_DIR

    resolved_data_dir = os.path.abspath(DATA_DIR)
    if (CSV_PATH_CACHE is None) or (CSV_PATH_CACHE_DATA_DIR != resolved_data_dir):
        from core.data_utils import discover_unique_csv_map

        CSV_PATH_CACHE, CSV_DUPLICATE_ISSUES = discover_unique_csv_map(resolved_data_dir)
        CSV_PATH_CACHE_DATA_DIR = resolved_data_dir
    return CSV_PATH_CACHE


def set_active_data_dir(data_dir):
    global DATA_DIR, CSV_PATH_CACHE, CSV_DUPLICATE_ISSUES, CSV_PATH_CACHE_DATA_DIR

    DATA_DIR = os.path.abspath(data_dir)
    CSV_PATH_CACHE = None
    CSV_DUPLICATE_ISSUES = None
    CSV_PATH_CACHE_DATA_DIR = None


def resolve_validate_dataset_profile_key(argv, environ):
    cli_value = extract_dataset_cli_value(argv)
    if cli_value is not None and str(cli_value).strip() != "":
        return normalize_dataset_profile_key(cli_value), "CLI"

    env_value = environ.get(VALIDATE_DATASET_ENV_VAR)
    if env_value:
        return normalize_dataset_profile_key(env_value), "ENV"

    if not is_interactive_stdin():
        return normalize_dataset_profile_key(DEFAULT_VALIDATE_DATASET_PROFILE), "DEFAULT"

    selected_value = safe_prompt(
        build_validate_dataset_prompt(DEFAULT_VALIDATE_DATASET_PROFILE),
        DEFAULT_VALIDATE_DATASET_PROFILE,
    )
    return normalize_dataset_profile_key(selected_value), "UI"


def load_params():
    return load_params_from_json(PARAMS_FILE)


def discover_available_tickers():
    if not os.path.isdir(DATA_DIR):
        return []

    return sorted(get_data_dir_csv_map().keys())


def print_progress(idx, total_tickers, ticker, ticker_pass_count, ticker_skip_count, ticker_fail_count):
    print(
        f"\r進度: [{idx}/{total_tickers}] 目前: {ticker:<8} | PASS股票:{ticker_pass_count} | SKIP股票:{ticker_skip_count} | FAIL股票:{ticker_fail_count}",
        end="",
        flush=True
    )




def _run_synthetic_suite_with_optional_coverage(run_dir, base_params, validator_runner, recoverable_exceptions):
    if not run_dir:
        return validator_runner(base_params)

    try:
        import coverage
    except ImportError:
        return validator_runner(base_params)

    coverage_dir = os.path.join(run_dir, "coverage_artifacts")
    os.makedirs(coverage_dir, exist_ok=True)
    data_file = os.path.join(coverage_dir, ".coverage.synthetic")
    json_file = os.path.join(coverage_dir, "coverage_synthetic.json")
    run_info_file = os.path.join(coverage_dir, "coverage_run_info.json")

    cov = coverage.Coverage(data_file=data_file, branch=True)
    results = []
    summaries = []
    synthetic_fail_count = 0
    raised_exc = None
    run_info = {
        "source": "validate_consistency",
        "returncode": 1,
        "stdout": "",
        "stderr": "",
        "timed_out": False,
        "synthetic_fail_count": 0,
        "synthetic_case_count": 0,
        "json_generated": False,
    }

    try:
        cov.start()
        results, summaries = validator_runner(base_params)
        synthetic_fail_count = sum(1 for row in results if row.get("status") == "FAIL")
        run_info["returncode"] = 0 if synthetic_fail_count == 0 else 1
        run_info["synthetic_fail_count"] = int(synthetic_fail_count)
        run_info["synthetic_case_count"] = int(len(summaries))
        run_info["stdout"] = json.dumps(
            {
                "synthetic_case_count": len(summaries),
                "synthetic_fail_count": synthetic_fail_count,
            },
            ensure_ascii=False,
        )
    except recoverable_exceptions as exc:
        raised_exc = exc
        run_info["stderr"] = f"{type(exc).__name__}: {exc}"
    finally:
        cov.stop()
        cov.save()
        if os.path.exists(data_file):
            try:
                cov.json_report(outfile=json_file, pretty_print=True)
                run_info["json_generated"] = True
            except Exception as exc:  # pragma: no cover - defensive artifact reporting
                run_info["stderr"] = (run_info["stderr"] + "\n" if run_info["stderr"] else "") + f"json_report: {type(exc).__name__}: {exc}"

        with open(run_info_file, "w", encoding="utf-8") as f:
            json.dump(run_info, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")

    if raised_exc is not None:
        raise raised_exc
    return results, summaries


def main(argv=None, environ=None):
    enable_line_buffered_stdout()
    argv = sys.argv if argv is None else argv
    environ = os.environ if environ is None else environ
    validate_cli_args(argv, value_options=("--dataset",))
    if has_help_flag(argv):
        program_name = resolve_cli_program_name(argv, "tools/validate/main.py")
        print(f"用法: python {program_name} [--dataset reduced|full]")
        print("說明: 預設資料集為縮減；reduced 測試資料路徑為 <repo>/data/tw_stock_data_vip_reduced。")
        return 0

    tracker = PeakTracedMemoryTracker()
    tracker.__enter__()

    import pandas as pd
    from tools.validate.checks import (
        add_fail_result,
        add_skip_result,
        normalize_ticker_text,
        is_insufficient_data_error,
    )
    from tools.validate.real_cases import run_real_ticker_scan
    from tools.local_regression.common import LOCAL_REGRESSION_RUN_DIR_ENV
    from tools.validate.reporting import print_console_summary, write_issue_excel_report, write_local_regression_summary
    from tools.validate.synthetic_cases import run_synthetic_consistency_suite
    from tools.validate.tool_adapters import VALIDATION_RECOVERABLE_EXCEPTIONS

    suite_run_dir = environ.get(LOCAL_REGRESSION_RUN_DIR_ENV, "").strip()
    output_dir = suite_run_dir if suite_run_dir else OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    try:
        dataset_profile_key, dataset_source = resolve_validate_dataset_profile_key(argv, environ)
        selected_data_dir = get_dataset_dir(PROJECT_ROOT, dataset_profile_key)
        set_active_data_dir(selected_data_dir)
        dataset_label = get_dataset_profile_label(dataset_profile_key)
    except ValueError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1

    try:
        base_params = load_params()
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1

    print(
        f"驗證資料集: {dataset_label} | "
        f"來源: {dataset_source} | 路徑: {DATA_DIR}"
    )
    all_results = []
    summaries = []
    start_time = time.time()

    selected_tickers = []
    real_data_unavailable_reason = None
    if not os.path.isdir(DATA_DIR):
        real_data_unavailable_reason = build_missing_dataset_dir_message(dataset_profile_key, DATA_DIR)
        print(real_data_unavailable_reason)
        print("將只執行 synthetic coverage suite，但本次驗證不可視為完整通過。")
    else:
        selected_tickers = discover_available_tickers()
        if not selected_tickers:
            real_data_unavailable_reason = build_empty_dataset_dir_message(dataset_profile_key, DATA_DIR)
            print(real_data_unavailable_reason)
            print("將只執行 synthetic coverage suite，但本次驗證不可視為完整通過。")

    total_tickers = len(selected_tickers)
    if total_tickers > 0:
        print(f"開始自動掃描 {total_tickers} 檔股票...")
        real_results, real_summaries, scan_stats = run_real_ticker_scan(
            selected_tickers,
            base_params,
            project_root=PROJECT_ROOT,
            data_dir=DATA_DIR,
            csv_map_getter=get_data_dir_csv_map,
            add_fail_result=add_fail_result,
            add_skip_result=add_skip_result,
            format_exception_summary=format_exception_summary,
            is_insufficient_data_error=is_insufficient_data_error,
            progress_printer=print_progress,
        )
        all_results.extend(real_results)
        summaries.extend(real_summaries)
        print(" " * 160, end="\r")
        print()
    else:
        scan_stats = {"total_tickers": 0}

    print("開始執行 synthetic coverage suite...")
    try:
        synthetic_results, synthetic_summaries = _run_synthetic_suite_with_optional_coverage(
            suite_run_dir,
            base_params,
            run_synthetic_consistency_suite,
            VALIDATION_RECOVERABLE_EXCEPTIONS,
        )
        all_results.extend(synthetic_results)
        summaries.extend(synthetic_summaries)
    except VALIDATION_RECOVERABLE_EXCEPTIONS as e:
        add_fail_result(
            all_results,
            "synthetic_suite",
            "SYNTHETIC_SUITE",
            "runtime",
            "suite runs successfully",
            format_exception_summary(e),
            "synthetic coverage suite 失敗時不可靜默略過，否則 miss buy / half TP / 多檔互動覆蓋會出現假象。"
        )
        summaries.append({
            "ticker": "SYNTHETIC_SUITE",
            "validation_runtime": f"FAIL: {format_exception_summary(e)}",
            "synthetic": True,
        })

    real_data_coverage_ok = real_data_unavailable_reason is None
    if real_data_unavailable_reason is not None:
        add_skip_result(
            all_results,
            "system",
            "REAL_DATA_COVERAGE",
            "real_data_scan_required",
            f"{real_data_unavailable_reason} 最嚴格檢查不可只靠 synthetic coverage suite；本次結果只能視為工具與合成案例檢查，不可視為完整通過。"
        )
        summaries.append({
            "ticker": "REAL_DATA_COVERAGE",
            "validation_runtime": f"SKIP: {real_data_unavailable_reason}",
            "synthetic": False,
        })

    df_results = pd.DataFrame(all_results)
    df_summary = pd.DataFrame(summaries)
    df_failed = df_results[df_results["status"] == "FAIL"].copy() if not df_results.empty else pd.DataFrame()

    for df_obj in [df_results, df_summary, df_failed]:
        if not df_obj.empty and "ticker" in df_obj.columns:
            df_obj["ticker"] = df_obj["ticker"].map(normalize_ticker_text)

    if not df_failed.empty:
        df_failed = df_failed.sort_values(by=["ticker", "module", "metric"]).reset_index(drop=True)

    timestamp = get_taipei_now().strftime("%Y%m%d_%H%M%S")
    csv_path = None

    if df_failed.empty:
        df_failed_summary = pd.DataFrame(columns=["ticker", "failed_checks"])
        df_failed_module = pd.DataFrame(columns=["module", "failed_checks"])
    else:
        df_failed_summary = (
            df_failed.groupby("ticker", dropna=False)
            .agg(failed_checks=("passed", "size"))
            .reset_index()
            .sort_values(by=["failed_checks", "ticker"], ascending=[False, True])
        )
        df_failed_module = (
            df_failed.groupby("module", dropna=False)
            .agg(failed_checks=("passed", "size"))
            .reset_index()
            .sort_values(by=["failed_checks", "module"], ascending=[False, True])
        )

    xlsx_path = None
    if suite_run_dir:
        if not df_failed.empty:
            csv_path = os.path.join(output_dir, f"consistency_failures_{timestamp}.csv")
            df_failed.to_csv(csv_path, index=False, encoding="utf-8-sig")
            xlsx_path = write_issue_excel_report(
                df_failed=df_failed,
                df_failed_summary=df_failed_summary,
                df_failed_module=df_failed_module,
                timestamp=timestamp,
                output_dir=output_dir,
                normalize_ticker=normalize_ticker_text,
            )
    else:
        csv_path = os.path.join(output_dir, f"consistency_full_scan_{timestamp}.csv")
        df_results.to_csv(csv_path, index=False, encoding="utf-8-sig")
        xlsx_path = write_issue_excel_report(
            df_failed=df_failed,
            df_failed_summary=df_failed_summary,
            df_failed_module=df_failed_module,
            timestamp=timestamp,
            output_dir=output_dir,
            normalize_ticker=normalize_ticker_text,
        )

    elapsed_time = time.time() - start_time

    print_console_summary(
        df_results=df_results,
        df_failed=df_failed,
        df_summary=df_summary,
        csv_path=csv_path,
        xlsx_path=xlsx_path,
        elapsed_time=elapsed_time,
        real_summary_count=scan_stats["total_tickers"],
        real_tickers=selected_tickers,
        normalize_ticker_text=normalize_ticker_text,
        max_console_fail_preview=MAX_CONSOLE_FAIL_PREVIEW,
    )

    write_local_regression_summary(
        dataset_profile_key=dataset_profile_key,
        dataset_source=dataset_source,
        data_dir=DATA_DIR,
        csv_path=csv_path,
        xlsx_path=xlsx_path,
        elapsed_time=elapsed_time,
        selected_tickers=selected_tickers,
        df_results=df_results,
        df_failed=df_failed,
        output_dir=output_dir,
        real_data_coverage_ok=real_data_coverage_ok,
        peak_traced_memory_mb=tracker.snapshot_peak_mb(),
    )

    tracker.__exit__(None, None, None)
    return 1 if ((not df_failed.empty) or (not real_data_coverage_ok)) else 0


if __name__ == "__main__":
    run_cli_entrypoint(main)
