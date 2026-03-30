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
    resolve_dataset_profile_from_cli_env,
)
from core.params_io import load_params_from_json
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



def run_debug_backtest(df, ticker, params, export_excel=True, verbose=True):
    from tools.debug.backtest import run_debug_backtest as _run_debug_backtest

    return _run_debug_backtest(
        df=df,
        ticker=ticker,
        params=params,
        output_dir=OUTPUT_DIR,
        colors=COLOR_MAP,
        export_excel=export_excel,
        verbose=verbose,
    )



def main(argv=None, environ=None):
    global DATA_DIR

    enable_line_buffered_stdout()
    argv = sys.argv if argv is None else argv
    environ = os.environ if environ is None else environ
    validate_cli_args(argv, value_options=("--dataset",))
    if has_help_flag(argv):
        program_name = resolve_cli_program_name(argv, "tools/debug/trade_log.py")
        print(f"用法: python {program_name} [--dataset reduced|full]")
        print("說明: 非互動模式可用 pipe 輸入股票代號；資料集預設為完整。")
        return 0

    from core.data_utils import discover_unique_csv_inputs, get_required_min_rows, resolve_unique_csv_path, sanitize_ohlcv_dataframe
    import pandas as pd

    try:
        dataset_profile_key, dataset_source = resolve_dataset_profile_from_cli_env(
            argv,
            environ,
            default=DEFAULT_DATASET_PROFILE,
        )
        DATA_DIR = get_dataset_dir(BASE_DIR, dataset_profile_key)
    except ValueError as e:
        raise ValueError(str(e)) from e

    dataset_dir_exists = os.path.isdir(DATA_DIR)
    csv_inputs = []
    if dataset_dir_exists:
        csv_inputs, _duplicate_file_issue_lines = discover_unique_csv_inputs(DATA_DIR)

    ticker = safe_prompt("\n👉 請輸入要除錯的股票代號 (例如: 00972): ", "").strip()
    if not ticker:
        if not dataset_dir_exists:
            raise FileNotFoundError(build_missing_dataset_dir_message(dataset_profile_key, DATA_DIR))
        if not csv_inputs:
            raise FileNotFoundError(build_empty_dataset_dir_message(dataset_profile_key, DATA_DIR))
        raise ValueError("未輸入股票代號，工具已取消。")

    manual_csv_path = f"{ticker}.csv"
    if os.path.exists(manual_csv_path):
        file_path = manual_csv_path
    else:
        if not dataset_dir_exists:
            raise FileNotFoundError(build_missing_dataset_dir_message(dataset_profile_key, DATA_DIR))
        if not csv_inputs:
            raise FileNotFoundError(build_empty_dataset_dir_message(dataset_profile_key, DATA_DIR))
        try:
            file_path, _duplicate_file_issue_lines = resolve_unique_csv_path(DATA_DIR, ticker)
        except FileNotFoundError as e:
            raise FileNotFoundError(str(e)) from e

    params = load_params(verbose=False)

    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(f"🛠️ {C_YELLOW}V16 放大鏡：單檔股票交易明細除錯工具{C_RESET}")
    print(f"{C_CYAN}================================================================================{C_RESET}")
    print(
        f"📁 使用資料集: {get_dataset_profile_label(dataset_profile_key)} | "
        f"來源: {dataset_source} | 路徑: {DATA_DIR}"
    )
    print(f"📥 讀取 {file_path}...")
    raw_df = pd.read_csv(file_path)
    print(f"{C_GREEN}✅ 成功載入參數大腦: {os.path.join(BASE_DIR, 'models', 'best_params.json')}{C_RESET}")

    min_rows_needed = get_required_min_rows(params)
    df, sanitize_stats = sanitize_ohlcv_dataframe(raw_df, ticker, min_rows=min_rows_needed)

    dropped_row_count = sanitize_stats['dropped_row_count']
    invalid_row_count = sanitize_stats['invalid_row_count']
    duplicate_date_count = sanitize_stats['duplicate_date_count']

    if dropped_row_count > 0:
        print(
            f"{C_YELLOW}⚠️ {ticker} 清洗移除 {dropped_row_count} 列 "
            f"(異常OHLCV={invalid_row_count}, 重複日期={duplicate_date_count}){C_RESET}"
        )

    print("⏳ 正在產生完整交易明細...")
    run_debug_backtest(df, ticker, params)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"{C_RED}❌ {e}{C_RESET}", file=sys.stderr)
        raise SystemExit(1)
