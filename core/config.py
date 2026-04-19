"""相容 façade：保留既有匯入路徑，實際 source of truth 已拆分。"""

import config.training_policy as _training_policy


def get_buy_sort_method():
    return _training_policy.BUY_SORT_METHOD


def get_ev_calc_method():
    return _training_policy.EV_CALC_METHOD


def get_score_calc_method():
    return _training_policy.SCORE_CALC_METHOD


def get_score_numerator_method():
    return _training_policy.SCORE_NUMERATOR_METHOD


from config.execution_policy import (  # noqa: F401
    EXECUTION_POLICY_PARAM_SPECS,
    RUNTIME_PARAM_DEFAULTS,
    RUNTIME_PARAM_SPECS,
    RUNTIME_PARAM_TYPES,
    build_execution_policy_snapshot,
    build_runtime_param_snapshot,
)
from config.training_policy import (  # noqa: F401
    SELECTION_POLICY_PARAM_SPECS,
    MAX_PORTFOLIO_MDD_PCT,
    MIN_ANNUAL_TRADES,
    MIN_BUY_FILL_RATE,
    MIN_EQUITY_CURVE_R_SQUARED,
    MIN_FULL_YEAR_RETURN_PCT,
    MIN_MONTHLY_WIN_RATE,
    MIN_TRADE_WIN_RATE,
    BUY_SORT_METHOD,
    EV_CALC_METHOD,
    SCORE_CALC_METHOD,
    SCORE_NUMERATOR_METHOD,
    SYSTEM_SCORE_DISPLAY_MULTIPLIER,
    OPTIMIZER_FIXED_TP_PERCENT,
    DEFAULT_OPTIMIZER_MODEL_MODE,
    PREDEPLOY_SELECTION_START_YEAR,
    OOS_EVALUATION_START_YEAR,
    build_selection_policy_snapshot,
    build_training_score_policy_snapshot,
    build_training_threshold_snapshot,
)
from core.capital_policy import (  # noqa: F401
    resolve_portfolio_entry_budget,
    resolve_portfolio_sizing_equity,
    resolve_scanner_live_capital,
    resolve_single_backtest_sizing_capital,
)
from core.strategy_params import (  # noqa: F401
    STRATEGY_PARAM_SPECS,
    V16StrategyParams,
    build_runtime_param_raw_value,
    normalize_runtime_param_value,
    normalize_strategy_param_value,
    strategy_params_to_dict,
    validate_strategy_param_ranges,
)
from strategies.breakout.schema import BREAKOUT_PARAM_SPECS  # noqa: F401
