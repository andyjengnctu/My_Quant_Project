import os
import sys
import warnings


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.dataset_profiles import (
    DEFAULT_DATASET_PROFILE,
    build_empty_dataset_dir_message,
    build_missing_dataset_dir_message,
    get_dataset_dir,
    get_dataset_profile_label,
    normalize_dataset_profile_key,
    resolve_dataset_profile_from_cli_env,
)
from core.capital_policy import resolve_scanner_live_capital
from core.params_io import load_params_from_json
from core.strategy_params import V16StrategyParams, strategy_params_to_dict
from core.output_paths import build_output_dir
from core.runtime_utils import enable_line_buffered_stdout, has_help_flag, resolve_cli_program_name, safe_prompt, validate_cli_args

warnings.simplefilter("default")
warnings.filterwarnings("once", category=RuntimeWarning)

C_CYAN = '\033[96m'
C_GREEN = '\033[92m'
C_YELLOW = '\033[93m'
C_RED = '\033[91m'
C_RESET = '\033[0m'

COLOR_MAP = {
    "cyan": C_CYAN,
    "green": C_GREEN,
    "yellow": C_YELLOW,
    "red": C_RED,
    "reset": C_RESET,
}

DATA_DIR = get_dataset_dir(BASE_DIR, DEFAULT_DATASET_PROFILE)
OUTPUT_DIR = build_output_dir(BASE_DIR, "debug_trade_log")


def load_params(json_file=os.path.join(BASE_DIR, "models", "best_params.json"), *, verbose=True):
    params = load_params_from_json(json_file)
    if verbose:
        print(f"{C_GREEN}✅ 成功載入參數大腦: {json_file}{C_RESET}")
    return params




def build_debug_view_params(params):
    param_payload = strategy_params_to_dict(params, include_runtime=True)
    cloned_params = V16StrategyParams(**strategy_params_to_dict(params, include_runtime=False))
    for runtime_field in ("optimizer_max_workers", "scanner_max_workers", "scanner_live_capital"):
        if runtime_field in param_payload:
            setattr(cloned_params, runtime_field, param_payload[runtime_field])
    cloned_params.initial_capital = resolve_scanner_live_capital(cloned_params)
    return cloned_params

def resolve_debug_data_dir(dataset_profile_key=DEFAULT_DATASET_PROFILE):
    normalized_key = normalize_dataset_profile_key(dataset_profile_key, default=DEFAULT_DATASET_PROFILE)
    return get_dataset_dir(BASE_DIR, normalized_key)


def resolve_debug_file_path(ticker, *, dataset_profile_key=DEFAULT_DATASET_PROFILE, data_dir=None):
    from core.data_utils import discover_unique_csv_inputs, resolve_unique_csv_path

    normalized_key = normalize_dataset_profile_key(dataset_profile_key, default=DEFAULT_DATASET_PROFILE)
    resolved_data_dir = resolve_debug_data_dir(normalized_key) if data_dir is None else os.path.abspath(str(data_dir))

    manual_csv_path = os.path.abspath(f"{ticker}.csv")
    if os.path.exists(manual_csv_path):
        return manual_csv_path, resolved_data_dir, "MANUAL"

    if not os.path.isdir(resolved_data_dir):
        raise FileNotFoundError(build_missing_dataset_dir_message(normalized_key, resolved_data_dir))

    csv_inputs, _duplicate_file_issue_lines = discover_unique_csv_inputs(resolved_data_dir)
    if not csv_inputs:
        raise FileNotFoundError(build_empty_dataset_dir_message(normalized_key, resolved_data_dir))

    file_path, _duplicate_file_issue_lines = resolve_unique_csv_path(resolved_data_dir, ticker)
    return file_path, resolved_data_dir, "DATASET"


def load_debug_price_frame(ticker, *, dataset_profile_key=DEFAULT_DATASET_PROFILE, data_dir=None):
    from core.data_utils import get_required_min_rows, sanitize_ohlcv_dataframe
    import pandas as pd

    normalized_key = normalize_dataset_profile_key(dataset_profile_key, default=DEFAULT_DATASET_PROFILE)
    file_path, resolved_data_dir, source = resolve_debug_file_path(
        ticker,
        dataset_profile_key=normalized_key,
        data_dir=data_dir,
    )
    raw_df = pd.read_csv(file_path)
    params = load_params(verbose=False)
    min_rows_needed = get_required_min_rows(params)
    clean_df, sanitize_stats = sanitize_ohlcv_dataframe(raw_df, ticker, min_rows=min_rows_needed)
    return {
        "ticker": ticker,
        "params": params,
        "file_path": file_path,
        "data_dir": resolved_data_dir,
        "source": source,
        "dataset_profile_key": normalized_key,
        "dataset_label": get_dataset_profile_label(normalized_key),
        "raw_df": raw_df,
        "clean_df": clean_df,
        "sanitize_stats": sanitize_stats,
    }


def run_debug_analysis(df, ticker, params, export_excel=True, export_chart=True, return_chart_payload=False, verbose=True, precomputed_signals=None, output_dir=None):
    from tools.trade_analysis.backtest import run_debug_analysis as _run_debug_analysis

    return _run_debug_analysis(
        df=df,
        ticker=ticker,
        params=params,
        output_dir=OUTPUT_DIR if output_dir is None else output_dir,
        colors=COLOR_MAP,
        export_excel=export_excel,
        export_chart=export_chart,
        return_chart_payload=return_chart_payload,
        verbose=verbose,
        precomputed_signals=precomputed_signals,
    )


def run_debug_backtest(df, ticker, params, export_excel=True, verbose=True, precomputed_signals=None):
    result = run_debug_analysis(
        df=df,
        ticker=ticker,
        params=params,
        export_excel=export_excel,
        export_chart=False,
        verbose=verbose,
        precomputed_signals=precomputed_signals,
    )
    return result["trade_logs_df"]



def run_debug_prepared_backtest(prepared_df, ticker, params, export_excel=True, verbose=True):
    return run_debug_backtest(
        prepared_df,
        ticker,
        params,
        export_excel=export_excel,
        verbose=verbose,
        precomputed_signals=None,
    )



def run_debug_ticker_analysis(
    ticker,
    *,
    dataset_profile_key=DEFAULT_DATASET_PROFILE,
    data_dir=None,
    params=None,
    export_excel=True,
    export_chart=True,
    return_chart_payload=False,
    verbose=False,
    output_dir=None,
):
    load_result = load_debug_price_frame(ticker, dataset_profile_key=dataset_profile_key, data_dir=data_dir)
    resolved_params = load_result["params"] if params is None else params
    debug_params = build_debug_view_params(resolved_params)
    analysis_result = run_debug_analysis(
        load_result["clean_df"],
        ticker,
        debug_params,
        export_excel=export_excel,
        export_chart=export_chart,
        return_chart_payload=return_chart_payload,
        verbose=verbose,
        output_dir=output_dir,
    )
    return {
        **load_result,
        **analysis_result,
    }



def main(argv=None, environ=None):
    global DATA_DIR

    enable_line_buffered_stdout()
    argv = sys.argv if argv is None else argv
    environ = os.environ if environ is None else environ
    validate_cli_args(argv, value_options=("--dataset",))
    if has_help_flag(argv):
        program_name = resolve_cli_program_name(argv, "tools/trade_analysis/trade_log.py")
        print(f"用法: python {program_name} [--dataset reduced|full]")
        print("說明: 非互動模式可用 pipe 輸入股票代號；資料集預設為完整。")
        return 0

    try:
        dataset_profile_key, dataset_source = resolve_dataset_profile_from_cli_env(
            argv,
            environ,
            default=DEFAULT_DATASET_PROFILE,
        )
        DATA_DIR = resolve_debug_data_dir(dataset_profile_key)
    except ValueError as e:
        raise ValueError(str(e)) from e

    dataset_dir_exists = os.path.isdir(DATA_DIR)
    csv_inputs = []
    if dataset_dir_exists:
        from core.data_utils import discover_unique_csv_inputs

        csv_inputs, _duplicate_file_issue_lines = discover_unique_csv_inputs(DATA_DIR)

    ticker = safe_prompt("\n👉 請輸入要除錯的股票代號 (例如: 00972): ", "").strip()
    if not ticker:
        if not dataset_dir_exists:
            raise FileNotFoundError(build_missing_dataset_dir_message(dataset_profile_key, DATA_DIR))
        if not csv_inputs:
            raise FileNotFoundError(build_empty_dataset_dir_message(dataset_profile_key, DATA_DIR))
        raise ValueError("未輸入股票代號，工具已取消。")

    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(f"🛠️ {C_YELLOW}V16 放大鏡：單檔股票交易明細除錯工具{C_RESET}")
    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(
        f"📁 使用資料集: {get_dataset_profile_label(dataset_profile_key)} | "
        f"來源: {dataset_source} | 路徑: {DATA_DIR}"
    )

    analysis_result = run_debug_ticker_analysis(
        ticker,
        dataset_profile_key=dataset_profile_key,
        export_excel=True,
        export_chart=True,
        verbose=False,
    )
    sanitize_stats = analysis_result["sanitize_stats"]

    print(f"📥 讀取 {analysis_result['file_path']}...")
    print(f"{C_GREEN}✅ 成功載入參數大腦: {os.path.join(BASE_DIR, 'models', 'best_params.json')}{C_RESET}")

    dropped_row_count = sanitize_stats['dropped_row_count']
    invalid_row_count = sanitize_stats['invalid_row_count']
    duplicate_date_count = sanitize_stats['duplicate_date_count']

    if dropped_row_count > 0:
        print(
            f"{C_YELLOW}⚠️ {ticker} 清洗移除 {dropped_row_count} 列 "
            f"(異常OHLCV={invalid_row_count}, 重複日期={duplicate_date_count}){C_RESET}"
        )

    if analysis_result["trade_logs_df"] is None:
        print(f"{C_YELLOW}⚠️ 這檔股票沒有任何交易紀錄。{C_RESET}")
        return 0

    if analysis_result["excel_path"]:
        print(f"{C_GREEN}📁 交易明細已成功匯出至：{analysis_result['excel_path']}{C_RESET}")
    if analysis_result["chart_path"]:
        print(f"{C_GREEN}📈 K 線交易檢視已成功匯出至：{analysis_result['chart_path']}{C_RESET}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"{C_RED}❌ {e}{C_RESET}", file=sys.stderr)
        raise SystemExit(1)
