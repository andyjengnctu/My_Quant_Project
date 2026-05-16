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
    OPTIMIZER_FIXED_TP_PERCENT,
    DEFAULT_OPTIMIZER_MODEL_MODE,
    PREDEPLOY_SELECTION_START_YEAR,
    OOS_EVALUATION_START_YEAR,
    build_selection_policy_snapshot,
    build_training_score_policy_snapshot,
    build_training_threshold_snapshot,
)
from config.training_display_policy import (  # noqa: F401
    SYSTEM_SCORE_DISPLAY_MULTIPLIER,
    build_display_policy_snapshot,
)
from config.training_performance_policy import (  # noqa: F401
    OPTIMIZER_FEATURE_BANK_MAX_ITEMS,
    OPTIMIZER_ROLLING_FOLD_WORKERS,
    OPTIMIZER_ROLLING_PARALLEL_PREP_CACHE_MAX_ITEMS,
    build_training_performance_policy_snapshot,
    resolve_optimizer_feature_bank_max_items_default,
    resolve_optimizer_rolling_fold_workers_default,
    resolve_optimizer_rolling_parallel_prep_cache_max_items_default,
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
