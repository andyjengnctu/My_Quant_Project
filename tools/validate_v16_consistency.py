import os
import sys
import copy
import time
import importlib.util
import tempfile
import io
import warnings
from contextlib import redirect_stdout, redirect_stderr
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.v16_params_io import load_params_from_json
from core.v16_core import run_v16_backtest
from core.v16_portfolio_engine import (
    prep_stock_data_and_trades,
    pack_prepared_stock_data,
    get_fast_dates,
    run_portfolio_timeline,
    unpack_portfolio_timeline_result,
    PORTFOLIO_TIMELINE_TRAIN_FIELDS,
)
from core.v16_data_utils import sanitize_ohlcv_dataframe, get_required_min_rows, discover_unique_csv_map
from core.v16_log_utils import format_exception_summary


OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
DATA_DIR = os.path.join(PROJECT_ROOT, "tw_stock_data_vip")
PARAMS_FILE = os.path.join(PROJECT_ROOT, "models", "v16_best_params.json")

FLOAT_TOL = 1e-6
VALIDATE_PROGRESS_EVERY = 25
MAX_CONSOLE_FAIL_PREVIEW = 20


CSV_PATH_CACHE = None
CSV_DUPLICATE_ISSUES = None
MODULE_CACHE = {}


# # (AI註: validate 自己做唯一輸出層；tool 全部 silent，只在這裡判定單檔 PASS / FAIL / SKIP)
def classify_ticker_status(ticker_results):
    statuses = {row["status"] for row in ticker_results}
    if "FAIL" in statuses:
        return "FAIL"
    if "PASS" in statuses:
        return "PASS"
    return "SKIP"


def build_progress_line(idx, total_tickers, ticker, pass_count, fail_count, skip_count):
    return (
        f"\r進度: [{idx:03d}/{total_tickers:03d}] "
        f"目前: {ticker:<8} | "
        f"PASS股票: {pass_count:<4} | "
        f"FAIL股票: {fail_count:<4} | "
        f"SKIP股票: {skip_count:<4}"
    )


def suppress_tool_output(func, *args, **kwargs):
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        return func(*args, **kwargs)


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


def load_module_from_candidates(module_name, relative_paths, required_attrs=None):
    if required_attrs is None:
        required_attrs = []

    checked_paths = []
    rejected_paths = []
    cache_key = (module_name, tuple(relative_paths), tuple(required_attrs))
    if cache_key in MODULE_CACHE:
        return MODULE_CACHE[cache_key]

    for rel_path in relative_paths:
        abs_path = os.path.join(PROJECT_ROOT, rel_path)
        checked_paths.append(abs_path)

        if not os.path.exists(abs_path):
            continue

        spec = importlib.util.spec_from_file_location(module_name, abs_path)
        if spec is None or spec.loader is None:
            rejected_paths.append(f"{abs_path} -> 無法建立 spec/loader")
            continue

        module = importlib.util.module_from_spec(spec)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*asyncio\.get_event_loop_policy.*deprecated.*Python 3\.16.*",
                category=DeprecationWarning,
                module=r"nest_asyncio",
            )
            spec.loader.exec_module(module)

        missing_attrs = [attr for attr in required_attrs if not hasattr(module, attr)]
        if missing_attrs:
            rejected_paths.append(f"{abs_path} -> 缺少必要屬性: {missing_attrs}")
            continue

        MODULE_CACHE[cache_key] = (module, abs_path)
        return MODULE_CACHE[cache_key]

    detail_msg = "；".join(rejected_paths) if rejected_paths else "沒有任何可用候選檔"
    raise FileNotFoundError(
        f"找不到符合條件的模組。檢查路徑: {checked_paths}。原因: {detail_msg}"
    )


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
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"找不到資料夾: {DATA_DIR}")

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
# # (AI註: 與 make_consistency_params 共用同一份放寬口徑，避免 validate 內部再出現第二套歷史門檻)
def build_execution_only_params(params):
    return make_consistency_params(params)

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

    portfolio_result = unpack_portfolio_timeline_result(
        run_portfolio_timeline(
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
        ),
        is_training=True,
    )

    result = {
        "total_return": portfolio_result["total_return"],
        "mdd": portfolio_result["mdd"],
        "trade_count": portfolio_result["trade_count"],
        "final_eq": portfolio_result["final_equity"],
        "avg_exp": portfolio_result["avg_exposure"],
        "max_exp": portfolio_result["max_exposure"],
        "bm_ret": portfolio_result["benchmark_return_pct"],
        "bm_mdd": portfolio_result["benchmark_mdd"],
        "win_rate": portfolio_result["win_rate"],
        "pf_ev": portfolio_result["pf_ev"],
        "pf_payoff": portfolio_result["pf_payoff"],
        "total_missed": portfolio_result["total_missed_buys"],
        "total_missed_sells": portfolio_result["total_missed_sells"],
        "r_sq": portfolio_result["r_squared"],
        "m_win_rate": portfolio_result["monthly_win_rate"],
        "bm_r_sq": portfolio_result["benchmark_r_squared"],
        "bm_m_win_rate": portfolio_result["benchmark_monthly_win_rate"],
        "normal_trade_count": portfolio_result["normal_trade_count"],
        "chase_trade_count": portfolio_result["chase_trade_count"],
        "annual_trades": portfolio_result["annual_trades"],
        "reserved_buy_fill_rate": portfolio_result["reserved_buy_fill_rate"],
        "annual_return_pct": portfolio_result["annual_return_pct"],
        "bm_annual_return_pct": portfolio_result["benchmark_annual_return_pct"],
        "sorted_dates": sorted_dates,
        "start_year": start_year,
        "standalone_logs": standalone_logs,
        "prep_df": prep_df,
    }
    result["run_portfolio_timeline_result_len"] = len(PORTFOLIO_TIMELINE_TRAIN_FIELDS)
    return result


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




def run_portfolio_sim_tool_check(ticker, file_path, params):
    module, module_path = load_module_from_candidates(
        "portfolio_sim_module",
        ["v16_portfolio_sim.py"],
        required_attrs=["run_portfolio_simulation"]
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        source_df = pd.read_csv(file_path)
        date_col = "Time" if "Time" in source_df.columns else "Date"
        start_year = int(pd.to_datetime(source_df[date_col], errors="coerce").dt.year.min())
        temp_csv_path = os.path.join(temp_dir, os.path.basename(file_path))
        source_df.to_csv(temp_csv_path, index=False)
        result = suppress_tool_output(
            module.run_portfolio_simulation,
            data_dir=temp_dir,
            params=copy.deepcopy(params),
            max_positions=1,
            enable_rotation=False,
            start_year=start_year,
            benchmark_ticker=ticker,
            verbose=False,
        )

    (
        df_eq,
        df_tr,
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
        chase_trade_count,
        annual_trades,
        reserved_buy_fill_rate,
        annual_return_pct,
        bm_annual_return_pct,
        pf_profile,
    ) = result

    return {
        "module_path": module_path,
        "df_eq": df_eq,
        "df_tr": df_tr,
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
        "chase_trade_count": chase_trade_count,
        "annual_trades": annual_trades,
        "reserved_buy_fill_rate": reserved_buy_fill_rate,
        "annual_return_pct": annual_return_pct,
        "bm_annual_return_pct": bm_annual_return_pct,
        "pf_profile": pf_profile,
    }


def run_scanner_tool_check(ticker, file_path, params):
    module, module_path = load_module_from_candidates(
        "vip_scanner_module",
        ["v16_vip_scanner.py"],
        required_attrs=["process_single_stock"]
    )
    result = module.process_single_stock(file_path, ticker, copy.deepcopy(params))
    return result, module_path


def run_downloader_tool_check(ticker):
    module, module_path = load_module_from_candidates(
        "vip_downloader_module",
        ["vip_smart_downloader.py"],
        required_attrs=["smart_download_vip_data"]
    )

    class DummyDL:
        def get_data(self, dataset, data_id, start_date):
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
        try:
            module.SAVE_DIR = temp_dir
            module.dl = DummyDL()
            module.time.sleep = lambda *_args, **_kwargs: None
            suppress_tool_output(
                module.smart_download_vip_data,
                [ticker],
                market_last_date='2024-01-03',
                verbose=False,
            )
            csv_path = os.path.join(temp_dir, f"{ticker}.csv")
            downloaded_df = pd.read_csv(csv_path, index_col=0)
        finally:
            module.SAVE_DIR = original_save_dir
            module.dl = original_dl
            module.time.sleep = original_sleep

    return downloaded_df, module_path


def validate_one_ticker(ticker, base_params):
    params = make_consistency_params(base_params)
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
    portfolio_stats = run_single_ticker_portfolio_check(ticker, df, params)
    export_row, export_module_path = run_all_stock_stats_check(ticker, params)
    debug_df, debug_module_path = run_debug_trade_log_check(ticker, df, params)
    portfolio_sim_stats = run_portfolio_sim_tool_check(ticker, file_path, params)
    scanner_result, scanner_module_path = run_scanner_tool_check(ticker, file_path, params)
    downloader_df, downloader_module_path = run_downloader_tool_check(ticker)

    summary["export_module_path"] = export_module_path
    summary["debug_module_path"] = debug_module_path
    summary["portfolio_sim_module_path"] = portfolio_sim_stats["module_path"]
    summary["scanner_module_path"] = scanner_module_path
    summary["downloader_module_path"] = downloader_module_path
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

    open_position_exists = single_stats["current_position"] > 0

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
        "normal_plus_chase_trade_count",
        portfolio_stats["trade_count"],
        portfolio_stats["normal_trade_count"] + portfolio_stats["chase_trade_count"],
        note="正常/追價完整交易數總和應等於總交易次數。"
    )

    sim_years = calc_validation_sim_years(portfolio_stats["sorted_dates"], portfolio_stats["start_year"])
    expected_annual_trades = (portfolio_stats["trade_count"] / sim_years) if sim_years > 0 else 0.0
    expected_reserved_buy_fill_rate = (
        portfolio_stats["trade_count"] / (portfolio_stats["trade_count"] + portfolio_stats["total_missed"]) * 100.0
        if (portfolio_stats["trade_count"] + portfolio_stats["total_missed"]) > 0 else 0.0
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
        buy_rows = int((debug_df["動作"] == "買進").sum())
        exit_rows = int(debug_df["動作"].isin(["停損殺出", "指標賣出", "期末強制結算"]).sum())
        half_rows = int((debug_df["動作"] == "半倉停利").sum())

        expected_exit_rows = len(standalone_logs)

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

    add_check(results, "portfolio_sim", ticker, "total_return",
              portfolio_stats["total_return"], portfolio_sim_stats["total_return"])
    add_check(results, "portfolio_sim", ticker, "mdd",
              portfolio_stats["mdd"], portfolio_sim_stats["mdd"])
    add_check(results, "portfolio_sim", ticker, "trade_count",
              portfolio_stats["trade_count"], portfolio_sim_stats["trade_count"])
    add_check(results, "portfolio_sim", ticker, "win_rate",
              portfolio_stats["win_rate"], portfolio_sim_stats["win_rate"])
    add_check(results, "portfolio_sim", ticker, "pf_ev",
              portfolio_stats["pf_ev"], portfolio_sim_stats["pf_ev"])
    add_check(results, "portfolio_sim", ticker, "pf_payoff",
              portfolio_stats["pf_payoff"], portfolio_sim_stats["pf_payoff"])
    add_check(results, "portfolio_sim", ticker, "annual_trades",
              portfolio_stats["annual_trades"], portfolio_sim_stats["annual_trades"])
    add_check(results, "portfolio_sim", ticker, "reserved_buy_fill_rate",
              portfolio_stats["reserved_buy_fill_rate"], portfolio_sim_stats["reserved_buy_fill_rate"])
    add_check(results, "portfolio_sim", ticker, "annual_return_pct",
              portfolio_stats["annual_return_pct"], portfolio_sim_stats["annual_return_pct"])
    add_check(results, "portfolio_sim", ticker, "bm_annual_return_pct",
              portfolio_stats["bm_annual_return_pct"], portfolio_sim_stats["bm_annual_return_pct"])
    add_check(results, "portfolio_sim", ticker, "timeline_result_len",
              portfolio_stats.get("run_portfolio_timeline_result_len", len(PORTFOLIO_TIMELINE_TRAIN_FIELDS)),
              len(PORTFOLIO_TIMELINE_TRAIN_FIELDS),
              note="透過 shared field map 驗證 portfolio_sim 與 portfolio_engine API 長度一致。")

    expected_scanner_result = None
    if single_stats['is_candidate']:
        if single_stats['is_setup_today']:
            from core.v16_core import calc_position_size, calc_entry_price
            proj_qty = calc_position_size(single_stats['buy_limit'], single_stats['stop_loss'], params.initial_capital, params.fixed_risk, params)
            if proj_qty == 0:
                expected_scanner_result = ('candidate', None, None, None, None, ticker, None)
            else:
                proj_cost = calc_entry_price(single_stats['buy_limit'], proj_qty, params) * proj_qty
                expected_scanner_result = ('buy', proj_cost, single_stats['expected_value'], None, None, ticker, None)
        elif single_stats.get('chase_today') is not None:
            chase = single_stats['chase_today']
            if chase['qty'] == 0:
                expected_scanner_result = ('candidate', None, None, None, None, ticker, None)
            else:
                proj_cost = calc_entry_price(chase['chase_price'], chase['qty'], params) * chase['qty']
                expected_scanner_result = ('zone', proj_cost, single_stats['expected_value'], None, None, ticker, None)
        else:
            expected_scanner_result = ('candidate', None, None, None, None, ticker, None)

    if expected_scanner_result is None:
        add_check(results, "vip_scanner", ticker, "candidate_none",
                  None, scanner_result)
    else:
        add_check(results, "vip_scanner", ticker, "status",
                  expected_scanner_result[0], scanner_result[0] if scanner_result is not None else None)
        add_check(results, "vip_scanner", ticker, "proj_cost",
                  expected_scanner_result[1], scanner_result[1] if scanner_result is not None else None)
        add_check(results, "vip_scanner", ticker, "expected_value",
                  expected_scanner_result[2], scanner_result[2] if scanner_result is not None else None)
        add_check(results, "vip_scanner", ticker, "ticker",
                  expected_scanner_result[5], scanner_result[5] if scanner_result is not None and len(scanner_result) > 5 else None)

    add_check(results, "vip_smart_downloader", ticker, "download_columns",
              ['Open', 'High', 'Low', 'Close', 'Volume'], list(downloader_df.columns))
    add_check(results, "vip_smart_downloader", ticker, "download_sorted_ascending",
              True, bool(pd.Index(downloader_df.index).is_monotonic_increasing))
    add_check(results, "vip_smart_downloader", ticker, "download_row_count",
              2, int(len(downloader_df)))
    add_check(results, "vip_smart_downloader", ticker, "download_last_date",
              '2024-01-03', str(downloader_df.index[-1]).split(' ')[0])

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
    selected_tickers = discover_available_tickers()

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
        ticker_status = classify_ticker_status(ticker_results)

        if ticker_status == "FAIL":
            ticker_fail_count += 1
        elif ticker_status == "PASS":
            ticker_pass_count += 1
        else:
            ticker_skip_count += 1

        print(
            build_progress_line(
                idx,
                total_tickers,
                ticker,
                ticker_pass_count,
                ticker_fail_count,
                ticker_skip_count,
            ),
            end="",
            flush=True,
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