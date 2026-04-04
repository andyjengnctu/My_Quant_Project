"""相容 façade：保留既有匯入路徑，實際 source of truth 已拆分。"""

from config.runtime_defaults import (  # noqa: F401
    BUY_SORT_METHOD,
    EV_CALC_METHOD,
    MAX_PORTFOLIO_MDD_PCT,
    MIN_ANNUAL_TRADES,
    MIN_BUY_FILL_RATE,
    MIN_EQUITY_CURVE_R_SQUARED,
    MIN_FULL_YEAR_RETURN_PCT,
    MIN_MONTHLY_WIN_RATE,
    MIN_TRADE_WIN_RATE,
    RUNTIME_PARAM_DEFAULTS,
    RUNTIME_PARAM_SPECS,
    RUNTIME_PARAM_TYPES,
    SCORE_CALC_METHOD,
    SYSTEM_SCORE_DISPLAY_MULTIPLIER,
)
from core.capital_policy import (  # noqa: F401
    resolve_portfolio_entry_budget,
    resolve_portfolio_sizing_equity,
    resolve_scanner_live_capital,
    resolve_single_backtest_sizing_capital,
)
from core.strategy_params import (  # noqa: F401
    V16StrategyParams,
    build_runtime_param_raw_value,
    normalize_runtime_param_value,
    normalize_strategy_param_value,
    strategy_params_to_dict,
    validate_strategy_param_ranges,
)
