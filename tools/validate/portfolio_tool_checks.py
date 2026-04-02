import copy

import pandas as pd

from core.data_utils import get_required_min_rows, sanitize_ohlcv_dataframe
from core.portfolio_fast_data import pack_prepared_stock_data, prep_stock_data_and_trades

from .module_loader import load_module_from_candidates
from .portfolio_payloads import build_portfolio_stats_payload
from .tool_check_common import resolve_source_date_column, suppress_tool_output


def _build_portfolio_stats_from_result(result, module_path):
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
        pf_profile,
    ) = result

    return build_portfolio_stats_payload(
        module_path=module_path,
        df_trades=_df_tr,
        total_return=tot_ret,
        mdd=mdd,
        trade_count=trade_count,
        win_rate=win_rate,
        pf_ev=pf_ev,
        pf_payoff=pf_payoff,
        final_eq=final_eq,
        avg_exp=avg_exp,
        max_exp=max_exp,
        bm_ret=bm_ret,
        bm_mdd=bm_mdd,
        total_missed=total_missed,
        total_missed_sells=total_missed_sells,
        r_sq=r_sq,
        m_win_rate=m_win_rate,
        bm_r_sq=bm_r_sq,
        bm_m_win_rate=bm_m_win_rate,
        normal_trade_count=normal_trade_count,
        extended_trade_count=extended_trade_count,
        annual_trades=annual_trades,
        reserved_buy_fill_rate=reserved_buy_fill_rate,
        annual_return_pct=annual_return_pct,
        bm_annual_return_pct=bm_annual_return_pct,
        profile_stats=pf_profile,
    )


def run_portfolio_sim_tool_check_for_dir(data_dir, params, *, max_positions, enable_rotation, start_year, benchmark_ticker):
    module, module_path = load_module_from_candidates(
        "portfolio_sim_module",
        ["apps/portfolio_sim.py"],
        required_attrs=["run_portfolio_simulation"],
    )

    result = suppress_tool_output(
        module.run_portfolio_simulation,
        data_dir=data_dir,
        params=copy.deepcopy(params),
        max_positions=max_positions,
        enable_rotation=enable_rotation,
        start_year=start_year,
        benchmark_ticker=benchmark_ticker,
        verbose=False,
    )
    return _build_portfolio_stats_from_result(result, module_path)


def run_portfolio_sim_tool_check(ticker, file_path, params):
    module, module_path = load_module_from_candidates(
        "portfolio_sim_module",
        ["apps/portfolio_sim.py"],
        required_attrs=["run_portfolio_simulation_prepared"],
    )

    source_df = pd.read_csv(file_path)
    date_col = resolve_source_date_column(source_df, ticker)
    parsed_dates = pd.to_datetime(source_df[date_col], errors="coerce")
    min_year = parsed_dates.dt.year.min()
    if pd.isna(min_year):
        raise ValueError(f"{ticker}: 日期欄位 {date_col} 無任何可解析日期")

    min_rows_needed = get_required_min_rows(params)
    clean_df, _sanitize_stats = sanitize_ohlcv_dataframe(source_df, ticker, min_rows=min_rows_needed)
    prep_df, standalone_logs = prep_stock_data_and_trades(clean_df, params)

    result = suppress_tool_output(
        module.run_portfolio_simulation_prepared,
        all_dfs_fast={ticker: pack_prepared_stock_data(prep_df)},
        all_trade_logs={ticker: standalone_logs},
        sorted_dates=sorted(prep_df.index),
        params=copy.deepcopy(params),
        max_positions=1,
        enable_rotation=False,
        start_year=int(min_year),
        benchmark_ticker=ticker,
        verbose=False,
    )
    return _build_portfolio_stats_from_result(result, module_path)
