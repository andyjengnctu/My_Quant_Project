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

# Candidate Scan Target switches to the just-closed Taiwan session at this
# local market time. Provider publication times are a separate Market Data
# scheduler concern and must not delay the target/readiness question itself.
TRADING_MARKET_SESSION_CLOSE_TIME = "13:30"

# Workbench Trading Center initial-state reads are independent read-only I/O
# after local consumer reconciliation. Parallelism only changes execution
# strategy; canonical state ownership and derivation remain unchanged.
TRADING_WORKBENCH_INITIAL_READ_WORKERS = 6

# Single-stock Workbench cache warming is intentionally bounded so background
# work never dominates the foreground ticker the user explicitly selected.
# These knobs only change execution scheduling/cache warmth; Trading membership,
# ordering, Params identity, and analysis semantics remain unchanged.
TRADING_WORKBENCH_SINGLE_STOCK_OHLCV_PREFETCH_TICKERS = 16
TRADING_WORKBENCH_SINGLE_STOCK_ANALYSIS_PREFETCH_TICKERS = 3

__all__ = [
    "TRADING_ACTIVE_STRATEGY_ID",
    "TRADING_DATASET_PROFILE",
    "TRADING_PARAM_FAMILY",
    "TRADING_DL_FILTER_ENABLED",
    "TRADING_DL_RANKING_ENABLED",
    "TRADING_MARKET_SESSION_CLOSE_TIME",
    "TRADING_WORKBENCH_INITIAL_READ_WORKERS",
    "TRADING_WORKBENCH_SINGLE_STOCK_OHLCV_PREFETCH_TICKERS",
    "TRADING_WORKBENCH_SINGLE_STOCK_ANALYSIS_PREFETCH_TICKERS",
]
