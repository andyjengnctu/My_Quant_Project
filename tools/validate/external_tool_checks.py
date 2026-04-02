import os
import tempfile

import pandas as pd

from .module_loader import load_module_from_candidates
from .scanner_expectations import normalize_scanner_result
from .tool_check_common import suppress_tool_output


def run_scanner_tool_check(ticker, file_path, params, *, prepared_df=None, sanitize_stats=None, precomputed_stats=None):
    module, module_path = load_module_from_candidates(
        "vip_scanner_module",
        ["apps/vip_scanner.py"],
        required_attrs=["process_single_stock"],
    )

    if precomputed_stats is not None and hasattr(module, "build_scanner_response_from_stats"):
        raw_result = suppress_tool_output(
            module.build_scanner_response_from_stats,
            ticker=ticker,
            stats=precomputed_stats,
            params=params,
            sanitize_stats=sanitize_stats or {},
        )
    elif prepared_df is not None and hasattr(module, "process_prepared_stock"):
        raw_result = suppress_tool_output(
            module.process_prepared_stock,
            df=prepared_df,
            ticker=ticker,
            params=params,
            sanitize_stats=sanitize_stats or {},
        )
    else:
        raw_result = suppress_tool_output(
            module.process_single_stock,
            file_path=file_path,
            ticker=ticker,
            params=params,
        )
    return normalize_scanner_result(raw_result), module_path


def run_downloader_tool_check(ticker):
    module, module_path = load_module_from_candidates(
        "vip_downloader_module",
        ["tools/downloader/main.py"],
        required_attrs=["smart_download_vip_data"],
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
                "date": ["2024-01-03", "2024-01-02"],
                "open": [11.0, 10.0],
                "max": [12.0, 11.0],
                "min": [10.5, 9.5],
                "close": [11.5, 10.5],
                "trading_volume": [2000, 1000],
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
                market_last_date="2024-01-03",
                verbose=False,
            )
            csv_path = os.path.join(temp_dir, f"{ticker}.csv")
            downloaded_df = pd.read_csv(csv_path, index_col=0)
        finally:
            module.SAVE_DIR = original_save_dir
            module.dl = original_dl
            module.time.sleep = original_sleep

    download_request = dummy_loader.calls[0] if dummy_loader.calls else None
    return downloaded_df, module_path, download_request, module.FINMIND_PRICE_DATASET


def run_debug_trade_log_check(ticker, df, params, *, prepared_df=None):
    module, module_path = load_module_from_candidates(
        "debug_trade_log_module",
        ["tools/debug/trade_log.py"],
        required_attrs=["run_debug_backtest"],
    )
    runner = getattr(module, "run_debug_prepared_backtest", None) if prepared_df is not None else None
    if runner is not None:
        debug_df = runner(
            prepared_df.copy(),
            ticker,
            params,
            export_excel=False,
            verbose=False,
        )
    else:
        debug_df = module.run_debug_backtest(
            df.copy(),
            ticker,
            params,
            export_excel=False,
            verbose=False,
        )
    return debug_df, module_path
