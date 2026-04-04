from core.strategy_params import build_runtime_param_raw_value, normalize_runtime_param_value


def _resolve_compounding_capital(value, fallback_value):
    if value is None:
        return max(0.0, float(fallback_value))
    return max(0.0, float(value))


def resolve_single_backtest_sizing_capital(params, current_capital=None):
    return _resolve_compounding_capital(current_capital, params.initial_capital)


def resolve_portfolio_sizing_equity(current_equity, initial_capital, params):
    return _resolve_compounding_capital(current_equity, initial_capital)


def resolve_portfolio_entry_budget(available_cash, initial_capital, params):
    return _resolve_compounding_capital(available_cash, initial_capital)


def resolve_scanner_live_capital(params):
    raw_value = build_runtime_param_raw_value(params, 'scanner_live_capital')
    return normalize_runtime_param_value('scanner_live_capital', raw_value)
