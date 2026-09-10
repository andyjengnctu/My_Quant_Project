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



def run_debug_trade_log_check(ticker, df, params, *, prepared_df=None):
    module, module_path = load_module_from_candidates(
        "debug_trade_log_module",
        ["services/trade_analysis/trade_log.py"],
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
