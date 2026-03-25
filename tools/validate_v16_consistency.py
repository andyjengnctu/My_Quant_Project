import os
import sys
import copy
import time
import importlib.util
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.v16_params_io import load_params_from_json
from core.v16_core import run_v16_backtest, calc_position_size, calc_entry_price
from core.v16_buy_sort import calc_buy_sort_value
from core.v16_config import BUY_SORT_METHOD
from core.v16_portfolio_engine import (
    prep_stock_data_and_trades,
    pack_prepared_stock_data,
    get_fast_dates,
    run_portfolio_timeline,
)
from core.v16_data_utils import sanitize_ohlcv_dataframe, get_required_min_rows, discover_unique_csv_map
from core.v16_log_utils import format_exception_summary
from contextlib import redirect_stdout, redirect_stderr
import tempfile
import io


OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
DATA_DIR = os.path.join(PROJECT_ROOT, "tw_stock_data_vip")
PARAMS_FILE = os.path.join(PROJECT_ROOT, "models", "v16_best_params.json")

FLOAT_TOL = 1e-6
VALIDATE_PROGRESS_EVERY = 25
MAX_CONSOLE_FAIL_PREVIEW = 20


CSV_PATH_CACHE = None
CSV_DUPLICATE_ISSUES = None


def get_data_dir_csv_map():
    global CSV_PATH_CACHE, CSV_DUPLICATE_ISSUES
    if CSV_PATH_CACHE is None:
        CSV_PATH_CACHE, CSV_DUPLICATE_ISSUES = discover_unique_csv_map(DATA_DIR)
    return CSV_PATH_CACHE


# # (AI註: consistency test 與 portfolio engine 共用相同口徑：實際模擬起訖日換算年數)
def calc_validation_sim_years(sorted_dates, start_year):
    if not sorted_dates:
        return 0.0

    start_dt = pd.to_datetime(f"{start_year}-01-01")
    start_idx = next((i for i, d in enumerate(sorted_dates) if d >= start_dt), len(sorted_dates))
    start_idx = max(1, start_idx)
    if start_idx >= len(sorted_dates):
        return 0.0

    first_dt = pd.Timestamp(sorted_dates[start_idx])
    last_dt = pd.Timestamp(sorted_dates[-1])
    span_days = (last_dt - first_dt).days + 1
    if span_days <= 0:
        return 0.0
    return span_days / 365.25


# # (AI註: 用 CAGR 驗證 annual return 指標，避免只驗證總報酬卻漏掉年化口徑)
def calc_validation_annual_return_pct(start_value, end_value, years):
    if start_value <= 0 or years <= 0:
        return 0.0
    if end_value <= 0:
        return -100.0
    return ((end_value / start_value) ** (1.0 / years) - 1.0) * 100.0


def load_params():
    return load_params_from_json(PARAMS_FILE)


def normalize_ticker_text(value):
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text[:-2].isdigit() and text.endswith(".0"):
        text = text[:-2]

    if text.isdigit() and len(text) < 4:
        text = text.zfill(4)

    return text

def is_insufficient_data_error(exc):
    return isinstance(exc, ValueError) and ("有效資料不足" in str(exc))

def make_consistency_params(base_params):
    params = copy.deepcopy(base_params)
    params.min_history_trades = 0
    params.min_history_ev = -1e9
    params.min_history_win_rate = 0.0
    return params


def discover_available_tickers():
    if not os.path.isdir(DATA_DIR):
        return []

    return sorted(get_data_dir_csv_map().keys())


def resolve_csv_path(ticker):
    csv_map = get_data_dir_csv_map()
    if ticker in csv_map:
        return csv_map[ticker]

    candidates = [
        os.path.join(DATA_DIR, f"TV_Data_Full_{ticker}.csv"),
        os.path.join(DATA_DIR, f"{ticker}.csv"),
        os.path.join(PROJECT_ROOT, f"TV_Data_Full_{ticker}.csv"),
        os.path.join(PROJECT_ROOT, f"{ticker}.csv"),
    ]
    raise FileNotFoundError(f"找不到 {ticker} 的 CSV。已檢查: {candidates}")


def load_clean_df(ticker, params):
    file_path = resolve_csv_path(ticker)
    raw_df = pd.read_csv(file_path)
    min_rows_needed = get_required_min_rows(params)
    df, sanitize_stats = sanitize_ohlcv_dataframe(raw_df, ticker, min_rows=min_rows_needed)
    return file_path, df, sanitize_stats


def add_check(results, module_name, ticker, metric, expected, actual, tol=FLOAT_TOL, note=""):
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        diff = abs(float(expected) - float(actual))
        passed = diff <= tol
    else:
        diff = None
        passed = expected == actual

    status = "PASS" if passed else "FAIL"

    results.append({
        "ticker": ticker,
        "module": module_name,
        "metric": metric,
        "expected": expected,
        "actual": actual,
        "abs_diff": diff,
        "passed": passed,
        "status": status,
        "note": note
    })

def add_skip_result(results, module_name, ticker, metric, note):
    results.append({
        "ticker": ticker,
        "module": module_name,
        "metric": metric,
        "expected": "SKIP",
        "actual": "SKIP",
        "abs_diff": None,
        "passed": True,
        "status": "SKIP",
        "note": note
    })


def add_fail_result(results, module_name, ticker, metric, expected, actual, note):
    results.append({
        "ticker": ticker,
        "module": module_name,
        "metric": metric,
        "expected": expected,
        "actual": actual,
        "abs_diff": None,
        "passed": False,
        "status": "FAIL",
        "note": note
    })


# # (AI註: validate 的 single_vs_portfolio 要驗證「原始執行邏輯一致性」，
# # (AI註: 不應混入 portfolio 的歷史績效濾網，否則會把設計上的選股層差異誤判成執行層 bug)
def build_execution_only_params(params):
    return make_consistency_params(params)


# # (AI註: scanner 必須用 production 門檻驗證；
# # (AI註: 若套用 consistency 的寬鬆門檻，會驗到實際 scanner 不會出現的候選)
def build_scanner_validation_params(base_params):
    return copy.deepcopy(base_params)


# # (AI註: 將 scanner 的 tuple 輸出正規化成 dict，
# # (AI註: 避免 validate 誤把 tuple 當 dict，導致 vip_scanner 完全沒被驗到)
def normalize_scanner_result(raw_result):
    if raw_result is None:
        return None

    if isinstance(raw_result, dict):
        return raw_result

    if not isinstance(raw_result, tuple) or len(raw_result) != 7:
        raise TypeError(
            f"scanner 回傳格式異常: type={type(raw_result).__name__}, value={raw_result}"
        )

    status, proj_cost, ev, sort_value, msg, ticker, sanitize_issue = raw_result
    return {
        "status": status,
        "proj_cost": proj_cost,
        "expected_value": ev,
        "sort_value": sort_value,
        "message": msg,
        "ticker": ticker,
        "sanitize_issue": sanitize_issue,
    }


# # (AI註: 用 strict scanner 參數重跑一次核心回測，作為 scanner 工具的真實參考口徑)
def run_scanner_reference_check(ticker, file_path, params):
    try:
        raw_df = pd.read_csv(file_path)
        min_rows_needed = get_required_min_rows(params)
        df, _sanitize_stats = sanitize_ohlcv_dataframe(raw_df, ticker, min_rows=min_rows_needed)
        return run_v16_backtest(df.copy(), params)
    except ValueError as e:
        if is_insufficient_data_error(e):
            return {"scanner_expected_status": "skip_insufficient"}
        raise


def derive_expected_scanner_status(scanner_ref_stats, params):
    if scanner_ref_stats.get("scanner_expected_status") == "skip_insufficient":
        return "skip_insufficient"

    if not scanner_ref_stats or not scanner_ref_stats["is_candidate"]:
        return None

    if scanner_ref_stats["is_setup_today"]:
        proj_qty = calc_position_size(
            scanner_ref_stats["buy_limit"],
            scanner_ref_stats["stop_loss"],
            params.initial_capital,
            params.fixed_risk,
            params
        )
        return "buy" if proj_qty > 0 else "candidate"

    extended_candidate_today = scanner_ref_stats.get("extended_candidate_today")
    if extended_candidate_today is not None:
        return "extended" if extended_candidate_today.get("qty", 0) > 0 else "candidate"

    return "candidate"

def build_expected_scanner_payload(scanner_ref_stats, params):
    status = derive_expected_scanner_status(scanner_ref_stats, params)
    payload = {
        "status": status,
        "expected_value": None,
        "proj_cost": None,
        "sort_value": None,
    }

    if status not in ("buy", "extended"):
        return payload

    if status == "buy":
        limit_price = scanner_ref_stats["buy_limit"]
        stop_loss = scanner_ref_stats["stop_loss"]
        proj_qty = calc_position_size(
            limit_price,
            stop_loss,
            params.initial_capital,
            params.fixed_risk,
            params
        )
    else:
        extended_candidate = scanner_ref_stats.get("extended_candidate_today")
        if extended_candidate is None:
            return payload
        limit_price = extended_candidate["limit_price"]
        proj_qty = extended_candidate.get("qty", 0)

    proj_cost = calc_entry_price(limit_price, proj_qty, params) * proj_qty
    sort_value = calc_buy_sort_value(
        BUY_SORT_METHOD,
        scanner_ref_stats["expected_value"],
        proj_cost,
        scanner_ref_stats["win_rate"] / 100.0,
        scanner_ref_stats["trade_count"],
    )
    payload.update({
        "expected_value": scanner_ref_stats["expected_value"],
        "proj_cost": proj_cost,
        "sort_value": sort_value,
    })
    return payload


def suppress_tool_output(func, *args, **kwargs):
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        return func(*args, **kwargs)


def resolve_source_date_column(source_df, ticker):
    if "Time" in source_df.columns:
        return "Time"
    if "Date" in source_df.columns:
        return "Date"
    raise KeyError(f"{ticker}: 找不到 Date/Time 欄位")


def run_portfolio_sim_tool_check(ticker, file_path, params):
    module, module_path = load_module_from_candidates(
        "portfolio_sim_module",
        ["v16_portfolio_sim.py"],
        required_attrs=["run_portfolio_simulation"]
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        source_df = pd.read_csv(file_path)
        temp_csv_path = os.path.join(temp_dir, os.path.basename(file_path))
        source_df.to_csv(temp_csv_path, index=False)

        date_col = resolve_source_date_column(source_df, ticker)

        parsed_dates = pd.to_datetime(source_df[date_col], errors="coerce")
        min_year = parsed_dates.dt.year.min()
        if pd.isna(min_year):
            raise ValueError(f"{ticker}: 日期欄位 {date_col} 無任何可解析日期")

        result = suppress_tool_output(
            module.run_portfolio_simulation,
            data_dir=temp_dir,
            params=copy.deepcopy(params),
            max_positions=1,
            enable_rotation=False,
            start_year=int(min_year),
            benchmark_ticker=ticker,
            verbose=False,
        )

    (
        _df_eq,
        _df_tr,
        tot_ret,
        mdd,
        trade_count,
        win_rate,
        pf_ev,
        pf_payoff,
        final_eq,
        avg_exp,
        max_exp,
        bm_ret,
        bm_mdd,
        total_missed,
        total_missed_sells,
        r_sq,
        m_win_rate,
        bm_r_sq,
        bm_m_win_rate,
        normal_trade_count,
        extended_trade_count,
        annual_trades,
        reserved_buy_fill_rate,
        annual_return_pct,
        bm_annual_return_pct,
        _pf_profile,
    ) = result

    return {
        "module_path": module_path,
        "total_return": tot_ret,
        "mdd": mdd,
        "trade_count": trade_count,
        "win_rate": win_rate,
        "pf_ev": pf_ev,
        "pf_payoff": pf_payoff,
        "final_eq": final_eq,
        "avg_exp": avg_exp,
        "max_exp": max_exp,
        "bm_ret": bm_ret,
        "bm_mdd": bm_mdd,
        "total_missed": total_missed,
        "total_missed_sells": total_missed_sells,
        "r_sq": r_sq,
        "m_win_rate": m_win_rate,
        "bm_r_sq": bm_r_sq,
        "bm_m_win_rate": bm_m_win_rate,
        "normal_trade_count": normal_trade_count,
        "extended_trade_count": extended_trade_count,
        "annual_trades": annual_trades,
        "reserved_buy_fill_rate": reserved_buy_fill_rate,
        "annual_return_pct": annual_return_pct,
        "bm_annual_return_pct": bm_annual_return_pct,
    }

def run_scanner_tool_check(ticker, file_path, params):
    module, module_path = load_module_from_candidates(
        "vip_scanner_module",
        ["v16_vip_scanner.py"],
        required_attrs=["process_single_stock"]
    )

    raw_result = suppress_tool_output(
        module.process_single_stock,
        file_path=file_path,
        ticker=ticker,
        params=copy.deepcopy(params)
    )
    return normalize_scanner_result(raw_result), module_path

def run_downloader_tool_check(ticker):
    module, module_path = load_module_from_candidates(
        "vip_downloader_module",
        ["vip_smart_downloader.py"],
        required_attrs=["smart_download_vip_data"]
    )

    class DummyDL:
        def __init__(self):
            self.calls = []

        def get_data(self, dataset, data_id, start_date):
            self.calls.append({
                "dataset": dataset,
                "data_id": data_id,
                "start_date": start_date,
            })
            if data_id != ticker:
                raise ValueError(f"unexpected ticker: {data_id}")
            return pd.DataFrame({
                'date': ['2024-01-03', '2024-01-02'],
                'open': [11.0, 10.0],
                'max': [12.0, 11.0],
                'min': [10.5, 9.5],
                'close': [11.5, 10.5],
                'trading_volume': [2000, 1000],
            })

    with tempfile.TemporaryDirectory() as temp_dir:
        original_save_dir = module.SAVE_DIR
        original_dl = module.dl
        original_sleep = module.time.sleep
        dummy_loader = DummyDL()
        try:
            module.SAVE_DIR = temp_dir
            module.dl = dummy_loader
            module.time.sleep = lambda *_args, **_kwargs: None
            suppress_tool_output(
                module.smart_download_vip_data,
                [ticker],
                market_last_date='2024-01-03',
                verbose=False
            )
            csv_path = os.path.join(temp_dir, f"{ticker}.csv")
            downloaded_df = pd.read_csv(csv_path, index_col=0)
        finally:
            module.SAVE_DIR = original_save_dir
            module.dl = original_dl
            module.time.sleep = original_sleep

    download_request = dummy_loader.calls[0] if dummy_loader.calls else None
    return downloaded_df, module_path, download_request, module.FINMIND_PRICE_DATASET

# # (AI註: validate 補回工具級檢查，避免只驗核心邏輯而漏掉 portfolio_sim / scanner / downloader)
MODULE_CACHE = {}

def load_module_from_candidates(cache_key, candidate_files, required_attrs):
    if cache_key in MODULE_CACHE:
        return MODULE_CACHE[cache_key]

    checked_paths = []
    rejected_paths = []

    for file_name in candidate_files:
        module_path = os.path.join(PROJECT_ROOT, file_name)
        checked_paths.append(module_path)

        if not os.path.exists(module_path):
            continue

        spec = importlib.util.spec_from_file_location(cache_key, module_path)
        if spec is None or spec.loader is None:
            rejected_paths.append(f"{module_path} -> 無法建立 spec/loader")
            continue

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        missing_attrs = [attr for attr in required_attrs if not hasattr(module, attr)]
        if missing_attrs:
            rejected_paths.append(f"{module_path} -> 缺少必要屬性: {missing_attrs}")
            continue

        MODULE_CACHE[cache_key] = (module, module_path)
        return module, module_path

    detail_msg = "；".join(rejected_paths) if rejected_paths else "沒有任何可用候選檔"
    raise FileNotFoundError(
        f"找不到符合條件的模組。檢查路徑: {checked_paths}。原因: {detail_msg}"
    )

def run_single_backtest_check(ticker, df, params):
    stats, standalone_logs = run_v16_backtest(df.copy(), params, return_logs=True)
    return stats, standalone_logs

def run_single_ticker_portfolio_check(ticker, df, params):
    execution_params = build_execution_only_params(params)
    prep_df, standalone_logs = prep_stock_data_and_trades(df.copy(), execution_params)

    fast_data = pack_prepared_stock_data(prep_df)
    all_dfs_fast = {ticker: fast_data}
    all_standalone_logs = {ticker: standalone_logs}

    sorted_dates = sorted(get_fast_dates(fast_data))
    if not sorted_dates:
        raise ValueError(f"{ticker}: pack_prepared_stock_data 後沒有任何有效日期")

    start_year = int(pd.Timestamp(sorted_dates[0]).year)

    result = run_portfolio_timeline(
        all_dfs_fast=all_dfs_fast,
        all_standalone_logs=all_standalone_logs,
        sorted_dates=sorted_dates,
        start_year=start_year,
        params=execution_params,
        max_positions=1,
        enable_rotation=False,
        benchmark_ticker=ticker,
        benchmark_data=fast_data,
        is_training=True,
        verbose=False
    )

    expected_result_len = 23
    if len(result) != expected_result_len:
        raise ValueError(
            f"run_portfolio_timeline(is_training=True) 回傳長度異常: {len(result)}，"
            f"預期 {expected_result_len}"
        )

    (
        total_return,
        mdd,
        trade_count,
        final_eq,
        avg_exp,
        max_exp,
        bm_ret,
        bm_mdd,
        win_rate,
        pf_ev,
        pf_payoff,
        total_missed,
        total_missed_sells,
        r_sq,
        m_win_rate,
        bm_r_sq,
        bm_m_win_rate,
        normal_trade_count,
        extended_trade_count,
        annual_trades,
        reserved_buy_fill_rate,
        annual_return_pct,
        bm_annual_return_pct,
    ) = result

    return {
        "total_return": total_return,
        "mdd": mdd,
        "trade_count": trade_count,
        "final_eq": final_eq,
        "avg_exp": avg_exp,
        "max_exp": max_exp,
        "bm_ret": bm_ret,
        "bm_mdd": bm_mdd,
        "win_rate": win_rate,
        "pf_ev": pf_ev,
        "pf_payoff": pf_payoff,
        "total_missed": total_missed,
        "total_missed_sells": total_missed_sells,
        "r_sq": r_sq,
        "m_win_rate": m_win_rate,
        "bm_r_sq": bm_r_sq,
        "bm_m_win_rate": bm_m_win_rate,
        "normal_trade_count": normal_trade_count,
        "extended_trade_count": extended_trade_count,
        "annual_trades": annual_trades,
        "reserved_buy_fill_rate": reserved_buy_fill_rate,
        "annual_return_pct": annual_return_pct,
        "bm_annual_return_pct": bm_annual_return_pct,
        "sorted_dates": sorted_dates,
        "start_year": start_year,
        "standalone_logs": standalone_logs,
        "prep_df": prep_df,
    }


def run_all_stock_stats_check(ticker, params):
    module, module_path = load_module_from_candidates(
        "all_stock_stats_module",
        ["tools/all_stock_stats.py", "all_stock_stats.py"],
        required_attrs=["process_single_stock_for_export"]
    )

    file_path = resolve_csv_path(ticker)
    export_row = module.process_single_stock_for_export(ticker, file_path, params)
    return export_row, module_path


def run_debug_trade_log_check(ticker, df, params):
    module, module_path = load_module_from_candidates(
        "debug_trade_log_module",
        ["tools/debug_trade_log.py", "debug_trade_log.py"],
        required_attrs=["run_debug_backtest"]
    )
    debug_df = module.run_debug_backtest(
        df.copy(),
        ticker,
        params,
        export_excel=False,
        verbose=False
    )
    return debug_df, module_path


# # (AI註: 將 debug_trade_log 的逐列事件重建為 completed trades，
# # (AI註: 才能嚴格驗證半倉停利 + 尾倉結算後的總損益口徑是否與核心一致)
def rebuild_completed_trades_from_debug_log(debug_df):
    if debug_df is None or len(debug_df) == 0:
        return []

    required_cols = {"日期", "動作", "單筆實質損益"}
    missing_cols = [col for col in required_cols if col not in debug_df.columns]
    if missing_cols:
        raise KeyError(f"debug_trade_log 缺少必要欄位: {missing_cols}")

    full_exit_actions = {"停損殺出", "指標賣出", "期末強制結算"}
    ignored_actions = {"錯失買進(新訊號)", "錯失買進(延續候選)", "放棄進場(先達停損)", "放棄進場(延續先達停損)"}
    completed_trades = []
    active_trade = None

    for row in debug_df.itertuples(index=False):
        action = getattr(row, "動作")
        trade_date = pd.to_datetime(getattr(row, "日期")).strftime("%Y-%m-%d")
        realized_pnl = round(float(getattr(row, "單筆實質損益")), 2)

        if action.startswith("買進"):
            if active_trade is not None:
                raise ValueError("debug_trade_log 出現連續買進，上一筆交易尚未完整結束。")
            active_trade = {
                "buy_date": trade_date,
                "exit_date": None,
                "total_pnl": 0.0,
                "half_exit_count": 0,
                "full_exit_action": None,
            }
            continue

        if action in ignored_actions:
            continue

        if action == "半倉停利":
            if active_trade is None:
                raise ValueError("debug_trade_log 出現半倉停利，但前面沒有對應買進。")
            active_trade["total_pnl"] = round(active_trade["total_pnl"] + realized_pnl, 2)
            active_trade["half_exit_count"] += 1
            continue

        if action in full_exit_actions:
            if active_trade is None:
                raise ValueError(f"debug_trade_log 出現 {action}，但前面沒有對應買進。")
            active_trade["total_pnl"] = round(active_trade["total_pnl"] + realized_pnl, 2)
            active_trade["exit_date"] = trade_date
            active_trade["full_exit_action"] = action
            completed_trades.append(active_trade)
            active_trade = None
            continue

        raise ValueError(f"debug_trade_log 出現未納入驗證的動作: {action}")

    if active_trade is not None:
        raise ValueError("debug_trade_log 最後仍有未完成交易，缺少完整賣出列。")

    return completed_trades


def validate_one_ticker(ticker, base_params):
    params = make_consistency_params(base_params)
    scanner_params = build_scanner_validation_params(base_params)
    file_path, df, sanitize_stats = load_clean_df(ticker, params)

    results = []
    summary = {
        "ticker": ticker,
        "file_path": file_path,
        "sanitize_dropped": sanitize_stats["dropped_row_count"],
        "sanitize_invalid": sanitize_stats["invalid_row_count"],
        "sanitize_duplicate": sanitize_stats["duplicate_date_count"],
    }

    single_stats, standalone_logs = run_single_backtest_check(ticker, df, params)
    scanner_ref_stats = run_scanner_reference_check(ticker, file_path, scanner_params)
    portfolio_stats = run_single_ticker_portfolio_check(ticker, df, params)
    portfolio_sim_stats = run_portfolio_sim_tool_check(ticker, file_path, params)
    scanner_result, scanner_module_path = run_scanner_tool_check(ticker, file_path, scanner_params)
    downloader_df, downloader_module_path, downloader_request, downloader_expected_dataset = run_downloader_tool_check(ticker)
    export_row, export_module_path = run_all_stock_stats_check(ticker, params)
    debug_df, debug_module_path = run_debug_trade_log_check(ticker, df, params)

    summary["portfolio_sim_module_path"] = portfolio_sim_stats["module_path"]
    summary["scanner_module_path"] = scanner_module_path
    summary["downloader_module_path"] = downloader_module_path
    summary["export_module_path"] = export_module_path
    summary["debug_module_path"] = debug_module_path
    summary["single_trade_count"] = single_stats["trade_count"]
    summary["portfolio_trade_count"] = portfolio_stats["trade_count"]
    summary["open_position_exists"] = bool(single_stats["current_position"] > 0)

    add_check(results, "single_vs_portfolio", ticker, "asset_growth_vs_total_return",
              single_stats["asset_growth"], portfolio_stats["total_return"])

    add_check(results, "single_vs_portfolio", ticker, "max_drawdown_vs_mdd",
              single_stats["max_drawdown"], portfolio_stats["mdd"])

    add_check(results, "single_vs_portfolio", ticker, "missed_buys",
              single_stats["missed_buys"], portfolio_stats["total_missed"])

    add_check(results, "single_vs_portfolio", ticker, "missed_sells",
              single_stats["missed_sells"], portfolio_stats["total_missed_sells"])

    add_check(
        results,
        "single_vs_portfolio",
        ticker,
        "trade_count",
        single_stats["trade_count"],
        portfolio_stats["trade_count"],
        note="單股與投組都已將期末強制結算納入交易統計。"
    )

    add_check(
        results,
        "single_vs_portfolio",
        ticker,
        "normal_plus_extended_trade_count",
        portfolio_stats["trade_count"],
        portfolio_stats["normal_trade_count"] + portfolio_stats["extended_trade_count"],
        note="正常/延續完整交易數總和應等於總交易次數。"
    )

    sim_years = calc_validation_sim_years(portfolio_stats["sorted_dates"], portfolio_stats["start_year"])
    expected_annual_trades = (portfolio_stats["trade_count"] / sim_years) if sim_years > 0 else 0.0
    total_reserved_entries = portfolio_stats["normal_trade_count"] + portfolio_stats["extended_trade_count"]
    expected_reserved_buy_fill_rate = (
        total_reserved_entries / (total_reserved_entries + portfolio_stats["total_missed"]) * 100.0
        if (total_reserved_entries + portfolio_stats["total_missed"]) > 0 else 0.0
    )
    expected_annual_return_pct = calc_validation_annual_return_pct(
        params.initial_capital, portfolio_stats["final_eq"], sim_years
    )
    expected_bm_annual_return_pct = calc_validation_annual_return_pct(
        100.0, 100.0 * (1.0 + portfolio_stats["bm_ret"] / 100.0), sim_years
    )

    add_check(results, "single_vs_portfolio", ticker, "annual_trades", expected_annual_trades, portfolio_stats["annual_trades"])
    add_check(results, "single_vs_portfolio", ticker, "reserved_buy_fill_rate", expected_reserved_buy_fill_rate, portfolio_stats["reserved_buy_fill_rate"])
    add_check(results, "single_vs_portfolio", ticker, "annual_return_pct", expected_annual_return_pct, portfolio_stats["annual_return_pct"])
    add_check(results, "single_vs_portfolio", ticker, "bm_annual_return_pct", expected_bm_annual_return_pct, portfolio_stats["bm_annual_return_pct"])

    add_check(results, "single_vs_portfolio", ticker, "win_rate",
              single_stats["win_rate"], portfolio_stats["win_rate"])
    add_check(results, "single_vs_portfolio", ticker, "payoff_ratio",
              single_stats["payoff_ratio"], portfolio_stats["pf_payoff"])
    add_check(results, "single_vs_portfolio", ticker, "expected_value",
              single_stats["expected_value"], portfolio_stats["pf_ev"])

    add_check(results, "portfolio_sim", ticker, "total_return", portfolio_stats["total_return"], portfolio_sim_stats["total_return"])
    add_check(results, "portfolio_sim", ticker, "mdd", portfolio_stats["mdd"], portfolio_sim_stats["mdd"])
    add_check(results, "portfolio_sim", ticker, "trade_count", portfolio_stats["trade_count"], portfolio_sim_stats["trade_count"])
    add_check(results, "portfolio_sim", ticker, "win_rate", portfolio_stats["win_rate"], portfolio_sim_stats["win_rate"])
    add_check(results, "portfolio_sim", ticker, "pf_ev", portfolio_stats["pf_ev"], portfolio_sim_stats["pf_ev"])
    add_check(results, "portfolio_sim", ticker, "pf_payoff", portfolio_stats["pf_payoff"], portfolio_sim_stats["pf_payoff"])
    add_check(results, "portfolio_sim", ticker, "final_eq", portfolio_stats["final_eq"], portfolio_sim_stats["final_eq"])
    add_check(results, "portfolio_sim", ticker, "avg_exp", portfolio_stats["avg_exp"], portfolio_sim_stats["avg_exp"])
    add_check(results, "portfolio_sim", ticker, "max_exp", portfolio_stats["max_exp"], portfolio_sim_stats["max_exp"])
    add_check(results, "portfolio_sim", ticker, "bm_ret", portfolio_stats["bm_ret"], portfolio_sim_stats["bm_ret"])
    add_check(results, "portfolio_sim", ticker, "bm_mdd", portfolio_stats["bm_mdd"], portfolio_sim_stats["bm_mdd"])
    add_check(results, "portfolio_sim", ticker, "total_missed", portfolio_stats["total_missed"], portfolio_sim_stats["total_missed"])
    add_check(results, "portfolio_sim", ticker, "total_missed_sells", portfolio_stats["total_missed_sells"], portfolio_sim_stats["total_missed_sells"])
    add_check(results, "portfolio_sim", ticker, "r_sq", portfolio_stats["r_sq"], portfolio_sim_stats["r_sq"])
    add_check(results, "portfolio_sim", ticker, "m_win_rate", portfolio_stats["m_win_rate"], portfolio_sim_stats["m_win_rate"])
    add_check(results, "portfolio_sim", ticker, "bm_r_sq", portfolio_stats["bm_r_sq"], portfolio_sim_stats["bm_r_sq"])
    add_check(results, "portfolio_sim", ticker, "bm_m_win_rate", portfolio_stats["bm_m_win_rate"], portfolio_sim_stats["bm_m_win_rate"])
    add_check(results, "portfolio_sim", ticker, "normal_trade_count", portfolio_stats["normal_trade_count"], portfolio_sim_stats["normal_trade_count"])
    add_check(results, "portfolio_sim", ticker, "extended_trade_count", portfolio_stats["extended_trade_count"], portfolio_sim_stats["extended_trade_count"])
    add_check(results, "portfolio_sim", ticker, "annual_trades", portfolio_stats["annual_trades"], portfolio_sim_stats["annual_trades"])
    add_check(results, "portfolio_sim", ticker, "reserved_buy_fill_rate", portfolio_stats["reserved_buy_fill_rate"], portfolio_sim_stats["reserved_buy_fill_rate"])
    add_check(results, "portfolio_sim", ticker, "annual_return_pct", portfolio_stats["annual_return_pct"], portfolio_sim_stats["annual_return_pct"])
    add_check(results, "portfolio_sim", ticker, "bm_annual_return_pct", portfolio_stats["bm_annual_return_pct"], portfolio_sim_stats["bm_annual_return_pct"])

    expected_scanner_payload = build_expected_scanner_payload(scanner_ref_stats, scanner_params)
    expected_scanner_status = expected_scanner_payload["status"]

    if scanner_result is None:
        add_check(
            results,
            "vip_scanner",
            ticker,
            "status",
            expected_scanner_status,
            None,
            note="scanner 已實際執行；None 只在 strict production 門檻下無候選時才屬正確。"
        )
    else:
        add_check(
            results,
            "vip_scanner",
            ticker,
            "ticker",
            str(ticker),
            str(scanner_result["ticker"])
        )

        add_check(
            results,
            "vip_scanner",
            ticker,
            "status",
            expected_scanner_status,
            scanner_result["status"]
        )
        add_check(
            results,
            "vip_scanner",
            ticker,
            "expected_value",
            expected_scanner_payload["expected_value"],
            scanner_result["expected_value"]
        )
        add_check(
            results,
            "vip_scanner",
            ticker,
            "proj_cost",
            expected_scanner_payload["proj_cost"],
            scanner_result["proj_cost"]
        )
        add_check(
            results,
            "vip_scanner",
            ticker,
            "sort_value",
            expected_scanner_payload["sort_value"],
            scanner_result["sort_value"]
        )

    extended_candidate = scanner_ref_stats.get("extended_candidate_today")
    if extended_candidate is None:
        add_skip_result(results, "vip_scanner", ticker, "extended_reference_price_in_range", "今日無延續候選，無需驗 A 版區間定義。")
        add_skip_result(results, "vip_scanner", ticker, "extended_limit_price_in_range", "今日無延續候選，無需驗 A 版區間定義。")
    else:
        reference_price = float(df["Close"].iloc[-1])
        add_check(
            results,
            "vip_scanner",
            ticker,
            "extended_reference_price_in_range",
            True,
            bool(extended_candidate["init_sl"] < reference_price <= extended_candidate["orig_limit"]),
        )
        add_check(
            results,
            "vip_scanner",
            ticker,
            "extended_limit_price_in_range",
            True,
            bool(extended_candidate["init_sl"] < extended_candidate["limit_price"] <= extended_candidate["orig_limit"]),
        )

    expected_download_cols = ["Open", "High", "Low", "Close", "Volume"]
    actual_download_cols = list(downloader_df.columns)
    add_check(results, "vip_downloader", ticker, "columns", expected_download_cols, actual_download_cols)
    add_check(results, "vip_downloader", ticker, "row_count", 2, len(downloader_df))
    add_check(results, "vip_downloader", ticker, "dataset", downloader_expected_dataset, None if downloader_request is None else downloader_request["dataset"])
    add_check(results, "vip_downloader", ticker, "data_id", ticker, None if downloader_request is None else downloader_request["data_id"])
    add_check(results, "vip_downloader", ticker, "start_date", "1990-01-01", None if downloader_request is None else downloader_request["start_date"])
    expected_download_index = ["2024-01-02", "2024-01-03"]
    actual_download_index = [str(idx).split(" ")[0] for idx in downloader_df.index.tolist()]
    add_check(results, "vip_downloader", ticker, "date_index_sorted", expected_download_index, actual_download_index)
    add_check(results, "vip_downloader", ticker, "index_name", "Date", downloader_df.index.name)
    expected_download_rows = [
        {"Open": 10.0, "High": 11.0, "Low": 9.5, "Close": 10.5, "Volume": 1000},
        {"Open": 11.0, "High": 12.0, "Low": 10.5, "Close": 11.5, "Volume": 2000},
    ]
    actual_download_rows = downloader_df.reset_index(drop=True).to_dict("records")
    add_check(results, "vip_downloader", ticker, "ohlcv_values_after_sort", expected_download_rows, actual_download_rows)

    if export_row is None:
        if single_stats["trade_count"] == 0:
            add_skip_result(
                results,
                "all_stock_stats",
                ticker,
                "export_row_exists",
                "無 completed trades，匯出工具回傳 None 屬設計行為。"
            )
        else:
            add_fail_result(
                results,
                "all_stock_stats",
                ticker,
                "export_row_exists",
                "非空",
                "None",
                "有 completed trades，但匯出工具卻回傳 None。"
            )
    else:
        add_check(results, "all_stock_stats", ticker, "交易次數",
                  single_stats["trade_count"], export_row["交易次數"])
        add_check(results, "all_stock_stats", ticker, "勝率 (Win Rate %)",
                  single_stats["win_rate"], export_row["勝率 (Win Rate %)"])
        add_check(results, "all_stock_stats", ticker, "平均獲利金額 (avgWin)",
                  single_stats["avg_win"], export_row["平均獲利金額 (avgWin)"])
        add_check(results, "all_stock_stats", ticker, "平均虧損金額 (avgLoss)",
                  single_stats["avg_loss"], export_row["平均虧損金額 (avgLoss)"])
        add_check(results, "all_stock_stats", ticker, "盈虧比 (payoffRatio)",
                  single_stats["payoff_ratio"], export_row["盈虧比 (payoffRatio)"])
        add_check(results, "all_stock_stats", ticker, "期望值 (expectedValue)",
                  single_stats["expected_value"], export_row["期望值 (expectedValue)"])
        add_check(results, "all_stock_stats", ticker, "平均持倉天數",
                  single_stats["avg_bars_held"], export_row["平均持倉天數"])
        add_check(results, "all_stock_stats", ticker, "總資產報酬率 (%)",
                  single_stats["asset_growth"], export_row["總資產報酬率 (%)"])
        add_check(results, "all_stock_stats", ticker, "最大回撤 MDD (%)",
                  single_stats["max_drawdown"], export_row["最大回撤 MDD (%)"])

    expected_buy_rows = len(standalone_logs)

    if debug_df is None or len(debug_df) == 0:
        if expected_buy_rows == 0:
            add_skip_result(
                results,
                "debug_trade_log",
                ticker,
                "debug_df_exists",
                "無交易紀錄時，debug 工具回傳 None 屬設計行為。"
            )
        else:
            add_fail_result(
                results,
                "debug_trade_log",
                ticker,
                "debug_df_exists",
                "非空",
                "None/Empty",
                "理應有交易明細，但 debug 工具回傳空值。"
            )
    else:
        buy_rows = int(debug_df["動作"].fillna("").str.startswith("買進").sum())
        exit_rows = int(debug_df["動作"].isin(["停損殺出", "指標賣出", "期末強制結算"]).sum())
        half_rows = int((debug_df["動作"] == "半倉停利").sum())
        debug_completed_trades = rebuild_completed_trades_from_debug_log(debug_df)

        expected_exit_rows = len(standalone_logs)
        expected_trade_pnls = [round(float(log["pnl"]), 2) for log in standalone_logs]
        actual_trade_pnls = [trade["total_pnl"] for trade in debug_completed_trades]
        expected_exit_dates = [pd.to_datetime(log["exit_date"]).strftime("%Y-%m-%d") for log in standalone_logs]
        actual_exit_dates = [trade["exit_date"] for trade in debug_completed_trades]
        expected_realized_pnl_sum = round(sum(expected_trade_pnls), 2)
        actual_realized_pnl_sum = round(sum(actual_trade_pnls), 2)

        add_check(
            results,
            "debug_trade_log",
            ticker,
            "buy_rows",
            expected_buy_rows,
            buy_rows,
            note="debug 已將期末強制結算列為完整賣出紀錄，買進筆數應等於 completed trades。"
        )

        add_check(
            results,
            "debug_trade_log",
            ticker,
            "full_exit_rows",
            expected_exit_rows,
            exit_rows
        )

        add_check(
            results,
            "debug_trade_log",
            ticker,
            "completed_trade_count",
            len(standalone_logs),
            len(debug_completed_trades),
            note="debug 明細需能重建為與核心 completed trades 完全相同的筆數。"
        )

        add_check(
            results,
            "debug_trade_log",
            ticker,
            "completed_trade_exit_dates",
            expected_exit_dates,
            actual_exit_dates,
            note="每筆 completed trade 的最終出場日期必須與核心一致，包含期末強制結算。"
        )

        add_check(
            results,
            "debug_trade_log",
            ticker,
            "completed_trade_pnl_sequence",
            expected_trade_pnls,
            actual_trade_pnls,
            note="debug 需將半倉停利 + 尾倉賣出合併後，逐筆總損益與核心 completed trades 一致。"
        )

        add_check(
            results,
            "debug_trade_log",
            ticker,
            "completed_trade_realized_pnl_sum",
            expected_realized_pnl_sum,
            actual_realized_pnl_sum,
            tol=0.01,
            note="逐筆加總後的總已實現損益必須與核心 completed trades 一致。"
        )

        results.append({
            "ticker": ticker,
            "module": "debug_trade_log",
            "metric": "half_take_profit_rows",
            "expected": "資訊欄位",
            "actual": half_rows,
            "abs_diff": None,
            "passed": True,
            "status": "PASS",
            "note": "半倉停利筆數只供人工檢查，不直接對應 completed trades。"
        })

    return results, summary


def write_issue_excel_report(df_failed, df_failed_summary, df_failed_module, timestamp):
    from openpyxl.styles import numbers

    if df_failed.empty:
        return None

    report_path = os.path.join(OUTPUT_DIR, f"v16_consistency_issues_{timestamp}.xlsx")

    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        df_failed.to_excel(writer, sheet_name="failed_only", index=False)
        df_failed_summary.to_excel(writer, sheet_name="failed_tickers", index=False)
        df_failed_module.to_excel(writer, sheet_name="failed_modules", index=False)

        for sheet_name in ["failed_only", "failed_tickers"]:
            ws = writer.book[sheet_name]
            header_map = {cell.value: idx + 1 for idx, cell in enumerate(ws[1])}
            if "ticker" in header_map:
                col_idx = header_map["ticker"]
                for row_idx in range(2, ws.max_row + 1):
                    cell = ws.cell(row=row_idx, column=col_idx)
                    cell.number_format = numbers.FORMAT_TEXT
                    cell.value = normalize_ticker_text(cell.value)

    return report_path


def print_console_summary(df_results, df_failed, df_summary, csv_path, xlsx_path, elapsed_time):
    total_tickers = len(df_summary)
    failed_tickers = int(df_failed["ticker"].nunique()) if not df_failed.empty else 0
    pass_count = int((df_results["status"] == "PASS").sum()) if not df_results.empty else 0
    skip_count = int((df_results["status"] == "SKIP").sum()) if not df_results.empty else 0
    fail_count = int((df_results["status"] == "FAIL").sum()) if not df_results.empty else 0

    print("\n================================================================================")
    print("一致性回歸摘要")
    print("================================================================================")
    print(f"耗時: {elapsed_time:.2f} 秒")
    print(f"成功進入 summary 的股票數: {total_tickers}")
    print(f"總檢查數: {len(df_results)}")
    print(f"PASS 數: {pass_count}")
    print(f"SKIP 數: {skip_count}")
    print(f"FAIL 數: {fail_count}")
    print(f"有問題股票數: {failed_tickers}")
    print(f"完整 CSV: {csv_path}")
    print(f"問題 Excel: {xlsx_path if xlsx_path else '無，因為沒有 failed 項'}")

    if df_failed.empty:
        print("\n失敗項摘要：無")
        return

    print("\n失敗項前覽：")
    show_cols = ["ticker", "module", "metric", "expected", "actual", "note"]
    preview_df = df_failed[show_cols].head(MAX_CONSOLE_FAIL_PREVIEW).copy()
    print(preview_df.to_string(index=False))

    remain_count = len(df_failed) - len(preview_df)
    if remain_count > 0:
        print(f"\n... 尚有 {remain_count} 筆 FAIL 未顯示，請直接查看 CSV / Excel。")

    failed_summary = (
        df_failed.groupby("ticker", dropna=False)
        .agg(failed_checks=("passed", "size"))
        .reset_index()
        .sort_values(by=["failed_checks", "ticker"], ascending=[False, True])
        .head(MAX_CONSOLE_FAIL_PREVIEW)
    )

    if not failed_summary.empty:
        print("\n失敗股票前覽：")
        print(failed_summary.to_string(index=False))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    base_params = load_params()

    if not os.path.isdir(DATA_DIR):
        print(f"找不到資料夾: {DATA_DIR}")
        print("請先準備 tw_stock_data_vip 後再執行一致性驗證。")
        return 2

    selected_tickers = discover_available_tickers()
    if not selected_tickers:
        print(f"資料夾內找不到任何 CSV: {DATA_DIR}")
        print("請先放入 {ticker}.csv 或 TV_Data_Full_{ticker}.csv 後再執行一致性驗證。")
        return 2

    all_results = []
    summaries = []

    ticker_pass_count = 0
    ticker_skip_count = 0
    ticker_fail_count = 0

    total_tickers = len(selected_tickers)
    start_time = time.time()

    print(f"開始自動掃描 {total_tickers} 檔股票...")

    for idx, ticker in enumerate(selected_tickers, start=1):
        ticker_results_before = len(all_results)

        try:
            results, summary = validate_one_ticker(ticker, base_params)
            all_results.extend(results)
            summaries.append(summary)
        except (FileNotFoundError, ImportError, OSError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError, KeyError, IndexError, TypeError, RuntimeError) as e:
            if is_insufficient_data_error(e):
                add_skip_result(
                    all_results,
                    "system",
                    ticker,
                    "validation_runtime",
                    f"資料不足，跳過驗證。({type(e).__name__}: {e})"
                )
            else:
                raise RuntimeError(
                    f"一致性驗證執行失敗: ticker={ticker} | {format_exception_summary(e)}"
                ) from e

        ticker_results = all_results[ticker_results_before:]
        ticker_statuses = {row["status"] for row in ticker_results}

        if "FAIL" in ticker_statuses:
            ticker_fail_count += 1
        elif "PASS" in ticker_statuses:
            ticker_pass_count += 1
        else:
            ticker_skip_count += 1

        print(
            f"\r進度: [{idx}/{total_tickers}] 目前: {ticker:<8} | PASS股票:{ticker_pass_count} | SKIP股票:{ticker_skip_count} | FAIL股票:{ticker_fail_count}",
            end="",
            flush=True
        )

    print(" " * 160, end="\r")
    print()

    df_results = pd.DataFrame(all_results)
    df_summary = pd.DataFrame(summaries)
    df_failed = df_results[df_results["status"] == "FAIL"].copy() if not df_results.empty else pd.DataFrame()

    for df_obj in [df_results, df_summary, df_failed]:
        if not df_obj.empty and "ticker" in df_obj.columns:
            df_obj["ticker"] = df_obj["ticker"].map(normalize_ticker_text)

    if not df_failed.empty:
        df_failed = df_failed.sort_values(by=["ticker", "module", "metric"]).reset_index(drop=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUTPUT_DIR, f"v16_consistency_full_scan_{timestamp}.csv")
    df_results.to_csv(csv_path, index=False, encoding="utf-8-sig")

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

    xlsx_path = write_issue_excel_report(
        df_failed=df_failed,
        df_failed_summary=df_failed_summary,
        df_failed_module=df_failed_module,
        timestamp=timestamp
    )

    elapsed_time = time.time() - start_time

    print_console_summary(
        df_results=df_results,
        df_failed=df_failed,
        df_summary=df_summary,
        csv_path=csv_path,
        xlsx_path=xlsx_path,
        elapsed_time=elapsed_time
    )

    return 1 if not df_failed.empty else 0


if __name__ == "__main__":
    sys.exit(main())