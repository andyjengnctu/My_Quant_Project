"""Trading declarative product configuration.

Trading state/artifacts are physically isolated from Research while consuming the
same canonical Strategy Optimizer training semantics from ``config.training_policy``
and execution/performance semantics from the canonical performance policy.
"""

TRADING_ACTIVE_STRATEGY_ID = "full_rule_based_no_dl"
TRADING_DATASET_PROFILE = "full"
TRADING_PARAM_FAMILY = "full"
TRADING_DL_FILTER_ENABLED = False
TRADING_DL_RANKING_ENABLED = False

__all__ = [
    "TRADING_ACTIVE_STRATEGY_ID",
    "TRADING_DATASET_PROFILE",
    "TRADING_PARAM_FAMILY",
    "TRADING_DL_FILTER_ENABLED",
    "TRADING_DL_RANKING_ENABLED",
]
