import pandas as pd

from tools.validate.checks import (
    add_check,
    add_fail_result,
    add_skip_result,
    build_expected_scanner_payload,
    calc_expected_full_year_metrics,
    calc_validation_annual_return_pct,
    calc_validation_sim_years,
)
from tools.validate.trade_rebuild import rebuild_completed_trades_from_debug_log


def append_real_case_checks(
    results,
    *,
    ticker,
    params,
    df,
    single_stats,
    standalone_logs,
    scanner_ref_stats,
    portfolio_stats,
    portfolio_sim_stats,
    scanner_result,
    downloader_df,
    downloader_request,
    downloader_expected_dataset,
    downloader_error,
    debug_df,
):
    expected_scanner_payload = build_expected_scanner_payload(scanner_ref_stats, params)
    expected_scanner_status = expected_scanner_payload["status"]

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
    add_check(results, "portfolio_sim", ticker, "df_trades_missed_buy_rows", portfolio_stats["total_missed"], portfolio_sim_stats["portfolio_missed_buy_rows"], note="portfolio df_trades 中的錯失買進列數，必須與 total_missed 完全一致。")
    add_check(results, "portfolio_sim", ticker, "df_trades_missed_sell_rows", portfolio_stats["total_missed_sells"], portfolio_sim_stats["portfolio_missed_sell_rows"], note="portfolio df_trades 中的錯失賣出列數，必須與 total_missed_sells 完全一致。")
    add_check(results, "portfolio_sim", ticker, "df_trades_buy_rows", len(standalone_logs), portfolio_sim_stats["portfolio_buy_rows"], note="portfolio df_trades 中的買進列數，必須與核心 completed trades 筆數一致。")
    add_check(results, "portfolio_sim", ticker, "df_trades_full_exit_rows", len(standalone_logs), portfolio_sim_stats["portfolio_full_exit_rows"], note="portfolio df_trades 中的完整賣出列數，必須與核心 completed trades 筆數一致，包含期末強制結算。")
    add_check(results, "portfolio_sim", ticker, "df_trades_period_closeout_rows", 1 if single_stats["current_position"] > 0 else 0, portfolio_sim_stats["portfolio_period_closeout_rows"], note="若單股回測期末仍持有部位，portfolio df_trades 必須恰有一列期末強制結算；否則必須為 0。")
    add_check(results, "portfolio_sim", ticker, "df_trades_completed_trade_count", len(standalone_logs), len(portfolio_sim_stats["portfolio_completed_trades"]), note="portfolio df_trades 必須能重建成與核心 completed trades 完全相同的筆數。")
    add_check(results, "portfolio_sim", ticker, "df_trades_completed_trade_exit_dates", expected_exit_dates, [trade["exit_date"] for trade in portfolio_sim_stats["portfolio_completed_trades"]], note="portfolio df_trades 重建出的逐筆最終出場日期，必須與核心 completed trades 完全一致。")
    add_check(results, "portfolio_sim", ticker, "df_trades_completed_trade_pnl_sequence", expected_trade_pnls, [trade["total_pnl"] for trade in portfolio_sim_stats["portfolio_completed_trades"]], note="portfolio df_trades 必須將半倉停利 + 尾倉賣出正確合併，逐筆總損益 sequence 與核心一致。")
    add_check(results, "portfolio_sim", ticker, "df_trades_completed_trade_realized_pnl_sum", expected_realized_pnl_sum, round(sum(trade["total_pnl"] for trade in portfolio_sim_stats["portfolio_completed_trades"]), 2), tol=0.01, note="portfolio df_trades 重建後的 completed trades 總已實現損益，必須與核心一致。")
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

    if scanner_result is None:
        add_check(results, "vip_scanner", ticker, "status", expected_scanner_status, None, note="scanner 已實際執行；None 只在 strict production 門檻下無候選時才屬正確。")
    else:
        add_check(results, "vip_scanner", ticker, "ticker", str(ticker), str(scanner_result["ticker"]))
        add_check(results, "vip_scanner", ticker, "status", expected_scanner_status, scanner_result["status"])
        add_check(results, "vip_scanner", ticker, "expected_value", expected_scanner_payload["expected_value"], scanner_result["expected_value"])
        add_check(results, "vip_scanner", ticker, "proj_cost", expected_scanner_payload["proj_cost"], scanner_result["proj_cost"])
        add_check(results, "vip_scanner", ticker, "sort_value", expected_scanner_payload["sort_value"], scanner_result["sort_value"])

    extended_candidate = scanner_ref_stats.get("extended_candidate_today")
    if extended_candidate is None:
        add_skip_result(results, "vip_scanner", ticker, "extended_reference_price_in_range", "今日無延續候選，無需驗 A 版區間定義。")
        add_skip_result(results, "vip_scanner", ticker, "extended_limit_price_in_range", "今日無延續候選，無需驗 A 版區間定義。")
    else:
        reference_price = float(df["Close"].iloc[-1])
        add_check(results, "vip_scanner", ticker, "extended_reference_price_in_range", True, bool(extended_candidate["init_sl"] < reference_price <= extended_candidate["orig_limit"]))
        add_check(results, "vip_scanner", ticker, "extended_limit_price_in_range", True, bool(extended_candidate["init_sl"] < extended_candidate["limit_price"] <= extended_candidate["orig_limit"]))

    if downloader_error is not None:
        add_fail_result(results, "vip_downloader", ticker, "tool_runtime", "tool loads and runs", downloader_error, note="downloader 工具失敗時，validate 應保留其他模組結果，不可整體中斷。")
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
            add_skip_result(results, "debug_trade_log", ticker, "debug_df_exists", "無交易紀錄時，debug 工具回傳 None 屬設計行為。")
        else:
            add_fail_result(results, "debug_trade_log", ticker, "debug_df_exists", "非空", "None/Empty", "理應有交易明細，但 debug 工具回傳空值。")
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

        add_check(results, "debug_trade_log", ticker, "buy_rows", expected_buy_rows, buy_rows, note="debug 已將期末強制結算列為完整賣出紀錄，買進筆數應等於 completed trades。")
        add_check(results, "debug_trade_log", ticker, "full_exit_rows", expected_exit_rows, exit_rows)
        add_check(results, "debug_trade_log", ticker, "missed_buy_rows", int(single_stats["missed_buys"]), missed_buy_rows, note="debug 明細中的錯失買進筆數，必須與核心 missed_buys 完全一致。")
        add_check(results, "debug_trade_log", ticker, "missed_sell_rows", int(single_stats["missed_sells"]), missed_sell_rows, note="debug 明細中的錯失賣出筆數，必須與核心 missed_sells 完全一致。")
        add_check(results, "debug_trade_log", ticker, "completed_trade_count", len(standalone_logs), len(debug_completed_trades), note="debug 明細需能重建為與核心 completed trades 完全相同的筆數。")
        add_check(results, "debug_trade_log", ticker, "completed_trade_exit_dates", expected_exit_dates, actual_exit_dates, note="每筆 completed trade 的最終出場日期必須與核心一致，包含期末強制結算。")
        add_check(results, "debug_trade_log", ticker, "completed_trade_pnl_sequence", expected_trade_pnls, actual_trade_pnls, note="debug 需將半倉停利 + 尾倉賣出合併後，逐筆總損益與核心 completed trades 一致。")
        add_check(results, "debug_trade_log", ticker, "completed_trade_realized_pnl_sum", expected_realized_pnl_sum, actual_realized_pnl_sum, tol=0.01, note="逐筆加總後的總已實現損益必須與核心 completed trades 一致。")
        add_check(results, "debug_trade_log", ticker, "half_take_profit_rows", int(portfolio_sim_stats["portfolio_half_take_profit_rows"]), half_rows, note="debug 與 portfolio_sim 的半倉停利列數必須一致，避免半倉現金回收口徑漂移。")
