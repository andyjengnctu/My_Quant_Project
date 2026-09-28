"""Trading declarative product configuration.

Trading state/artifacts are physically isolated from Research.  Workbench production
parameter training owns its product-level seed/trial contract here while reusing the
canonical Optimizer engine/search semantics.  Research may replay or scale the same
policy for controlled PIT experiments, but must not redefine Trading production.
"""

TRADING_ACTIVE_STRATEGY_ID = "full_rule_based_no_dl"
TRADING_DATASET_PROFILE = "full"
TRADING_PARAM_FAMILY = "full"
TRADING_PARAM_SELECTOR = "base_finalists_agree"

# Trading production parameter training is a product-level contract, not a
# Research comparison knob.  Each explicit retrain uses a fresh random seed
# bundle, matching the pre-2026-09-19 Workbench production semantics.
TRADING_OPTIMIZER_TRIALS_PER_SEED = 1000
TRADING_OPTIMIZER_SEED_COUNT = 8
TRADING_OPTIMIZER_SEED_MODE = "random_n_each_training_run"
TRADING_OPTIMIZER_SEED_MIN_AGREE = "auto"
TRADING_OPTIMIZER_TRAIN_WINDOW_MONTHS = 120

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
    "TRADING_PARAM_SELECTOR",
    "TRADING_OPTIMIZER_TRIALS_PER_SEED",
    "TRADING_OPTIMIZER_SEED_COUNT",
    "TRADING_OPTIMIZER_SEED_MODE",
    "TRADING_OPTIMIZER_SEED_MIN_AGREE",
    "TRADING_OPTIMIZER_TRAIN_WINDOW_MONTHS",
    "TRADING_DL_FILTER_ENABLED",
    "TRADING_DL_RANKING_ENABLED",
    "TRADING_MARKET_SESSION_CLOSE_TIME",
    "TRADING_WORKBENCH_INITIAL_READ_WORKERS",
    "TRADING_WORKBENCH_SINGLE_STOCK_OHLCV_PREFETCH_TICKERS",
    "TRADING_WORKBENCH_SINGLE_STOCK_ANALYSIS_PREFETCH_TICKERS",
]
