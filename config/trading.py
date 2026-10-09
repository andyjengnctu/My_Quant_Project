"""Trading declarative product configuration.

Trading state/artifacts remain physically isolated from Research.  The active
strategy-parameter semantics are selected by one registered Strategy Parameter
profile so production cannot silently drift from the promoted Research contract.
"""

from config.strategy_params import STRATEGY_PARAM_FITTING_SEED_PROTOCOLS, STRATEGY_PARAM_PROFILES
from config.training_policy import OUTER_ROLLING_TRAIN_WINDOW_MONTHS


TRADING_ACTIVE_STRATEGY_ID = "full_rule_based_no_dl"
TRADING_DATASET_PROFILE = "full"

# 2026-10-06 user promotion decision: Trading adopts SR-C85 / PARAM-P6.
# Runtime resolves the registered profile for family/selector/budget, while the
# live seed sampling mode is a Trading-only product setting. Research retains
# the fixed seed bundle for controlled, reproducible C85 experiments.
TRADING_STRATEGY_PARAM_PROFILE_ID = "production_multiseed_current"

_TRADING_PARAM_PROFILE = STRATEGY_PARAM_PROFILES[TRADING_STRATEGY_PARAM_PROFILE_ID]
_TRADING_FITTING_PROTOCOL = STRATEGY_PARAM_FITTING_SEED_PROTOCOLS[
    str(_TRADING_PARAM_PROFILE["fitting_seed_protocol"])
]

TRADING_PARAM_FAMILY = str(_TRADING_PARAM_PROFILE["filter_profile"])
TRADING_PARAM_SELECTOR = str(_TRADING_PARAM_PROFILE["parameter_policy"]).replace("-", "_")
TRADING_PARAMETER_SET_IDENTITY = str(_TRADING_PARAM_PROFILE["parameter_set_identity"])
TRADING_OPTIMIZER_TRIALS_PER_SEED = int(_TRADING_FITTING_PROTOCOL["trials_per_seed"])
TRADING_OPTIMIZER_SEED_COUNT = int(_TRADING_FITTING_PROTOCOL["seed_count"])
# Trading restores its original random-on-each-explicit-retrain behavior.
# Research P6 still uses its immutable fixed Research generator seed=42.
TRADING_OPTIMIZER_SEED_MODE = "random_n_each_training_run"
TRADING_OPTIMIZER_SEED_GENERATOR_SEED = (
    int(_TRADING_FITTING_PROTOCOL["generator_seed"])
    if TRADING_OPTIMIZER_SEED_MODE == "fixed_generated_bundle"
    else None
)
TRADING_OPTIMIZER_SEED_MIN_AGREE = _TRADING_FITTING_PROTOCOL["min_agree"]
TRADING_OPTIMIZER_TRAIN_WINDOW_MONTHS = int(OUTER_ROLLING_TRAIN_WINDOW_MONTHS)

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
    "TRADING_STRATEGY_PARAM_PROFILE_ID",
    "TRADING_PARAM_FAMILY",
    "TRADING_PARAM_SELECTOR",
    "TRADING_PARAMETER_SET_IDENTITY",
    "TRADING_OPTIMIZER_TRIALS_PER_SEED",
    "TRADING_OPTIMIZER_SEED_COUNT",
    "TRADING_OPTIMIZER_SEED_MODE",
    "TRADING_OPTIMIZER_SEED_GENERATOR_SEED",
    "TRADING_OPTIMIZER_SEED_MIN_AGREE",
    "TRADING_OPTIMIZER_TRAIN_WINDOW_MONTHS",
    "TRADING_DL_FILTER_ENABLED",
    "TRADING_DL_RANKING_ENABLED",
    "TRADING_MARKET_SESSION_CLOSE_TIME",
    "TRADING_WORKBENCH_INITIAL_READ_WORKERS",
    "TRADING_WORKBENCH_SINGLE_STOCK_OHLCV_PREFETCH_TICKERS",
    "TRADING_WORKBENCH_SINGLE_STOCK_ANALYSIS_PREFETCH_TICKERS",
]
