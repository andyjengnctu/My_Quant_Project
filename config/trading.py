"""Trading declarative configuration.

Trading state/artifacts are physically isolated from Research while consuming the
same canonical strategy/execution semantics.
"""

TRADING_ACTIVE_STRATEGY_ID = "full_rule_based_no_dl"
TRADING_DATASET_PROFILE = "full"
TRADING_PARAM_FAMILY = "full"
TRADING_PARAM_SELECTOR = "base_finalist_best"
TRADING_OPTIMIZER_MULTI_SEED_REQUIRED = True
TRADING_DL_FILTER_ENABLED = False
TRADING_DL_RANKING_ENABLED = False

__all__ = [
    "TRADING_ACTIVE_STRATEGY_ID",
    "TRADING_DATASET_PROFILE",
    "TRADING_PARAM_FAMILY",
    "TRADING_PARAM_SELECTOR",
    "TRADING_OPTIMIZER_MULTI_SEED_REQUIRED",
    "TRADING_DL_FILTER_ENABLED",
    "TRADING_DL_RANKING_ENABLED",
]
