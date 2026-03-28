import os

import pandas as pd

from core.v16_core import run_v16_backtest
from core.v16_data_utils import sanitize_ohlcv_dataframe, get_required_min_rows
from core.v16_portfolio_engine import run_portfolio_timeline
from core.v16_portfolio_fast_data import get_fast_dates, pack_prepared_stock_data, prep_stock_data_and_trades
from tools.validate.checks import (
    add_check,
    add_fail_result,
    add_skip_result,
    build_execution_only_params,
    build_expected_scanner_payload,
    build_scanner_validation_params,
    calc_expected_full_year_metrics,
    calc_validation_annual_return_pct,
    calc_validation_sim_years,
    extract_yearly_profile_fields,
    make_consistency_params,
    run_scanner_reference_check,
)
from tools.validate.tool_adapters import (
    VALIDATION_RECOVERABLE_EXCEPTIONS,
    run_debug_trade_log_check,
    run_downloader_tool_check,
    run_portfolio_sim_tool_check,
    run_scanner_tool_check,
)
from tools.validate.trade_rebuild import (
    rebuild_completed_trades_from_debug_log,
    rebuild_completed_trades_from_portfolio_trade_log,
)


def resolve_csv_path(project_root, data_dir, csv_map_getter, ticker):
    csv_map = csv_map_getter()
    if ticker in csv_map:
        return csv_map[ticker]

    candidates = [
        os.path.join(data_dir, f"TV_Data_Full_{ticker}.csv"),
        os.path.join(data_dir, f"{ticker}.csv"),
        os.path.join(project_root, f"TV_Data_Full_{ticker}.csv"),
        os.path.join(project_root, f"{ticker}.csv"),
    ]
    raise FileNotFoundError(f"找不到 {ticker} 的 CSV。已檢查: {candidates}")


def load_clean_df(project_root, data_dir, csv_map_getter, ticker, params):
    file_path = resolve_csv_path(project_root, data_dir, csv_map_getter, ticker)
    raw_df = pd.read_csv(file_path)
    min_rows_needed = get_required_min_rows(params)
    df, sanitize_stats = sanitize_ohlcv_dataframe(raw_df, ticker, min_rows=min_rows_needed)
    return file_path, df, sanitize_stats


def run_single_backtest_check(df, params):
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
    profile_stats = {}

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
        profile_stats=profile_stats,
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

    payload = {
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
    payload.update(extract_yearly_profile_fields(profile_stats))
    return payload


def validate_one_ticker(project_root, data_dir, csv_map_getter, ticker, base_params):
    params = make_consistency_params(base_params)
    scanner_params = build_scanner_validation_params(base_params)
    file_path, df, sanitize_stats = load_clean_df(project_root, data_dir, csv_map_getter, ticker, params)

    results = []
    summary = {
        "ticker": ticker,
        "file_path": file_path,
        "sanitize_dropped": sanitize_stats["dropped_row_count"],
        "sanitize_invalid": sanitize_stats["invalid_row_count"],
        "sanitize_duplicate": sanitize_stats["duplicate_date_count"],
    }

    single_stats, standalone_logs = run_single_backtest_check(df, params)
    scanner_ref_stats = run_scanner_reference_check(ticker, file_path, scanner_params)
    portfolio_stats = run_single_ticker_portfolio_check(ticker, df, params)
    portfolio_sim_stats = run_portfolio_sim_tool_check(ticker, file_path, params)
    scanner_result, scanner_module_path = run_scanner_tool_check(ticker, file_path, scanner_params)
    downloader_df, downloader_module_path, downloader_request, downloader_expected_dataset = None, None, None, None
    downloader_error = None
    try:
        downloader_df, downloader_module_path, downloader_request, downloader_expected_dataset = run_downloader_tool_check(ticker)
    except VALIDATION_RECOVERABLE_EXCEPTIONS as e:
        downloader_error = f"{type(e).__name__}: {e}"
    debug_df, debug_module_path = run_debug_trade_log_check(ticker, df, params)

    summary["portfolio_sim_module_path"] = portfolio_sim_stats["module_path"]
    summary["scanner_module_path"] = scanner_module_path
    summary["downloader_module_path"] = downloader_module_path
    summary["downloader_error"] = downloader_error
    summary["debug_module_path"] = debug_module_path
    summary["single_trade_count"] = single_stats["trade_count"]
    summary["portfolio_trade_count"] = portfolio_stats["trade_count"]
    summary["open_position_exists"] = bool(single_stats["current_position"] > 0)
    summary["has_extended_candidate_today"] = bool(scanner_ref_stats.get("extended_candidate_today") is not None)
    summary["has_missed_buy"] = bool(single_stats["missed_buys"] > 0)
    summary["portfolio_half_take_profit_rows"] = int(portfolio_sim_stats["portfolio_half_take_profit_rows"])

    # --- begin generated check block ---
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
    expected_exit_dates = [pd.to_datetime(log["exit_date"]).strftime("%Y-%m-%d") for log in standalone_logs]
    expected_trade_pnls = [round(float(log["pnl"]), 2) for log in standalone_logs]
    expected_realized_pnl_sum = round(sum(expected_trade_pnls), 2)
    expected_full_year_metrics = calc_expected_full_year_metrics(portfolio_stats["yearly_return_rows"])
    expected_bm_full_year_metrics = calc_expected_full_year_metrics(portfolio_stats["bm_yearly_return_rows"])

    add_check(results, "single_vs_portfolio", ticker, "annual_trades", expected_annual_trades, portfolio_stats["annual_trades"])
    add_check(results, "single_vs_portfolio", ticker, "reserved_buy_fill_rate", expected_reserved_buy_fill_rate, portfolio_stats["reserved_buy_fill_rate"])
    add_check(results, "single_vs_portfolio", ticker, "annual_return_pct", expected_annual_return_pct, portfolio_stats["annual_return_pct"])
    add_check(results, "single_vs_portfolio", ticker, "bm_annual_return_pct", expected_bm_annual_return_pct, portfolio_stats["bm_annual_return_pct"])
    add_check(results, "single_vs_portfolio", ticker, "full_year_count", expected_full_year_metrics["full_year_count"], portfolio_stats["full_year_count"])
    add_check(results, "single_vs_portfolio", ticker, "min_full_year_return_pct", expected_full_year_metrics["min_full_year_return_pct"], portfolio_stats["min_full_year_return_pct"])
    add_check(results, "single_vs_portfolio", ticker, "yearly_return_rows", portfolio_stats["yearly_return_rows"], portfolio_stats["yearly_return_rows"], note="年度報酬明細需保留完整列內容，供後續與 portfolio_sim 對照。")
    add_check(results, "single_vs_portfolio", ticker, "bm_full_year_count", expected_bm_full_year_metrics["full_year_count"], portfolio_stats["bm_full_year_count"])
    add_check(results, "single_vs_portfolio", ticker, "bm_min_full_year_return_pct", expected_bm_full_year_metrics["min_full_year_return_pct"], portfolio_stats["bm_min_full_year_return_pct"])
    add_check(results, "single_vs_portfolio", ticker, "bm_yearly_return_rows", portfolio_stats["bm_yearly_return_rows"], portfolio_stats["bm_yearly_return_rows"], note="Benchmark 年度報酬明細需保留完整列內容，供後續與 portfolio_sim 對照。")

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
    add_check(
        results,
        "portfolio_sim",
        ticker,
        "df_trades_missed_buy_rows",
        portfolio_stats["total_missed"],
        portfolio_sim_stats["portfolio_missed_buy_rows"],
        note="portfolio df_trades 中的錯失買進列數，必須與 total_missed 完全一致。"
    )
    add_check(
        results,
        "portfolio_sim",
        ticker,
        "df_trades_missed_sell_rows",
        portfolio_stats["total_missed_sells"],
        portfolio_sim_stats["portfolio_missed_sell_rows"],
        note="portfolio df_trades 中的錯失賣出列數，必須與 total_missed_sells 完全一致。"
    )
    add_check(
        results,
        "portfolio_sim",
        ticker,
        "df_trades_buy_rows",
        len(standalone_logs),
        portfolio_sim_stats["portfolio_buy_rows"],
        note="portfolio df_trades 中的買進列數，必須與核心 completed trades 筆數一致。"
    )
    add_check(
        results,
        "portfolio_sim",
        ticker,
        "df_trades_full_exit_rows",
        len(standalone_logs),
        portfolio_sim_stats["portfolio_full_exit_rows"],
        note="portfolio df_trades 中的完整賣出列數，必須與核心 completed trades 筆數一致，包含期末強制結算。"
    )
    add_check(
        results,
        "portfolio_sim",
        ticker,
        "df_trades_period_closeout_rows",
        1 if single_stats["current_position"] > 0 else 0,
        portfolio_sim_stats["portfolio_period_closeout_rows"],
        note="若單股回測期末仍持有部位，portfolio df_trades 必須恰有一列期末強制結算；否則必須為 0。"
    )
    add_check(
        results,
        "portfolio_sim",
        ticker,
        "df_trades_completed_trade_count",
        len(standalone_logs),
        len(portfolio_sim_stats["portfolio_completed_trades"]),
        note="portfolio df_trades 必須能重建成與核心 completed trades 完全相同的筆數。"
    )
    add_check(
        results,
        "portfolio_sim",
        ticker,
        "df_trades_completed_trade_exit_dates",
        expected_exit_dates,
        [trade["exit_date"] for trade in portfolio_sim_stats["portfolio_completed_trades"]],
        note="portfolio df_trades 重建出的逐筆最終出場日期，必須與核心 completed trades 完全一致。"
    )
    add_check(
        results,
        "portfolio_sim",
        ticker,
        "df_trades_completed_trade_pnl_sequence",
        expected_trade_pnls,
        [trade["total_pnl"] for trade in portfolio_sim_stats["portfolio_completed_trades"]],
        note="portfolio df_trades 必須將半倉停利 + 尾倉賣出正確合併，逐筆總損益 sequence 與核心一致。"
    )
    add_check(
        results,
        "portfolio_sim",
        ticker,
        "df_trades_completed_trade_realized_pnl_sum",
        expected_realized_pnl_sum,
        round(sum(trade["total_pnl"] for trade in portfolio_sim_stats["portfolio_completed_trades"]), 2),
        tol=0.01,
        note="portfolio df_trades 重建後的 completed trades 總已實現損益，必須與核心一致。"
    )
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
    add_check(results, "portfolio_sim", ticker, "full_year_count", portfolio_stats["full_year_count"], portfolio_sim_stats["full_year_count"])
    add_check(results, "portfolio_sim", ticker, "min_full_year_return_pct", portfolio_stats["min_full_year_return_pct"], portfolio_sim_stats["min_full_year_return_pct"])
    add_check(results, "portfolio_sim", ticker, "yearly_return_rows", portfolio_stats["yearly_return_rows"], portfolio_sim_stats["yearly_return_rows"])
    add_check(results, "portfolio_sim", ticker, "bm_full_year_count", portfolio_stats["bm_full_year_count"], portfolio_sim_stats["bm_full_year_count"])
    add_check(results, "portfolio_sim", ticker, "bm_min_full_year_return_pct", portfolio_stats["bm_min_full_year_return_pct"], portfolio_sim_stats["bm_min_full_year_return_pct"])
    add_check(results, "portfolio_sim", ticker, "bm_yearly_return_rows", portfolio_stats["bm_yearly_return_rows"], portfolio_sim_stats["bm_yearly_return_rows"])

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

    if downloader_error is not None:
        add_fail_result(
            results,
            "vip_downloader",
            ticker,
            "tool_runtime",
            "tool loads and runs",
            downloader_error,
            note="downloader 工具失敗時，validate 應保留其他模組結果，不可整體中斷。"
        )
    else:
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
        action_series = debug_df["動作"].fillna("")
        buy_rows = int(action_series.str.startswith("買進").sum())
        exit_rows = int(action_series.isin(["停損殺出", "指標賣出", "期末強制結算"]).sum())
        half_rows = int((action_series == "半倉停利").sum())
        missed_buy_rows = int(action_series.str.startswith("錯失買進").sum())
        missed_sell_rows = int((action_series == "錯失賣出").sum())
        debug_completed_trades = rebuild_completed_trades_from_debug_log(debug_df)

        expected_exit_rows = len(standalone_logs)
        actual_trade_pnls = [trade["total_pnl"] for trade in debug_completed_trades]
        actual_exit_dates = [trade["exit_date"] for trade in debug_completed_trades]
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
            "missed_buy_rows",
            int(single_stats["missed_buys"]),
            missed_buy_rows,
            note="debug 明細中的錯失買進筆數，必須與核心 missed_buys 完全一致。"
        )

        add_check(
            results,
            "debug_trade_log",
            ticker,
            "missed_sell_rows",
            int(single_stats["missed_sells"]),
            missed_sell_rows,
            note="debug 明細中的錯失賣出筆數，必須與核心 missed_sells 完全一致。"
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

        add_check(
            results,
            "debug_trade_log",
            ticker,
            "half_take_profit_rows",
            int(portfolio_sim_stats["portfolio_half_take_profit_rows"]),
            half_rows,
            note="debug 與 portfolio_sim 的半倉停利列數必須一致，避免半倉現金回收口徑漂移。"
        )

    return results, summary

def run_real_ticker_scan(selected_tickers, base_params, *, project_root, data_dir, csv_map_getter, add_fail_result, add_skip_result, format_exception_summary, is_insufficient_data_error, progress_printer=None):
    all_results = []
    summaries = []
    ticker_pass_count = 0
    ticker_skip_count = 0
    ticker_fail_count = 0

    total_tickers = len(selected_tickers)
    for idx, ticker in enumerate(selected_tickers, start=1):
        ticker_results_before = len(all_results)

        try:
            results, summary = validate_one_ticker(project_root, data_dir, csv_map_getter, ticker, base_params)
            all_results.extend(results)
            summaries.append(summary)
        except VALIDATION_RECOVERABLE_EXCEPTIONS as e:
            if is_insufficient_data_error(e):
                add_skip_result(
                    all_results,
                    "system",
                    ticker,
                    "validation_runtime",
                    f"資料不足，跳過驗證。({type(e).__name__}: {e})"
                )
                summaries.append({
                    "ticker": ticker,
                    "validation_runtime": f"SKIP: {format_exception_summary(e)}",
                })
            else:
                add_fail_result(
                    all_results,
                    "system",
                    ticker,
                    "validation_runtime",
                    "no exception",
                    format_exception_summary(e),
                    "單一 ticker 的 runtime / import side effect / 路徑權限問題，不可讓整體 validate 中斷。"
                )
                summaries.append({
                    "ticker": ticker,
                    "validation_runtime": f"FAIL: {format_exception_summary(e)}",
                })

        ticker_results = all_results[ticker_results_before:]
        ticker_statuses = {row["status"] for row in ticker_results}

        if "FAIL" in ticker_statuses:
            ticker_fail_count += 1
        elif "PASS" in ticker_statuses:
            ticker_pass_count += 1
        else:
            ticker_skip_count += 1

        if progress_printer is not None:
            progress_printer(idx, total_tickers, ticker, ticker_pass_count, ticker_skip_count, ticker_fail_count)

    return all_results, summaries, {
        "total_tickers": total_tickers,
        "pass_count": ticker_pass_count,
        "skip_count": ticker_skip_count,
        "fail_count": ticker_fail_count,
    }
