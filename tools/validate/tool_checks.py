import copy
import io
import os
import tempfile
from contextlib import redirect_stderr, redirect_stdout

import pandas as pd

from .module_loader import load_module_from_candidates
from .portfolio_payloads import build_portfolio_stats_payload
from .scanner_expectations import normalize_scanner_result


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


def run_portfolio_sim_tool_check_for_dir(data_dir, params, *, max_positions, enable_rotation, start_year, benchmark_ticker):
    module, module_path = load_module_from_candidates(
        "portfolio_sim_module",
        ["apps/portfolio_sim.py"],
        required_attrs=["run_portfolio_simulation"]
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


def run_portfolio_sim_tool_check(ticker, file_path, params):
    with tempfile.TemporaryDirectory() as temp_dir:
        source_df = pd.read_csv(file_path)
        temp_csv_path = os.path.join(temp_dir, os.path.basename(file_path))
        source_df.to_csv(temp_csv_path, index=False)

        date_col = resolve_source_date_column(source_df, ticker)
        parsed_dates = pd.to_datetime(source_df[date_col], errors="coerce")
        min_year = parsed_dates.dt.year.min()
        if pd.isna(min_year):
            raise ValueError(f"{ticker}: 日期欄位 {date_col} 無任何可解析日期")

        return run_portfolio_sim_tool_check_for_dir(
            temp_dir,
            params,
            max_positions=1,
            enable_rotation=False,
            start_year=int(min_year),
            benchmark_ticker=ticker,
        )


def run_scanner_tool_check(ticker, file_path, params):
    module, module_path = load_module_from_candidates(
        "vip_scanner_module",
        ["apps/vip_scanner.py"],
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
        ["tools/downloader/main.py"],
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


def run_debug_trade_log_check(ticker, df, params):
    module, module_path = load_module_from_candidates(
        "debug_trade_log_module",
        ["tools/debug/trade_log.py"],
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
