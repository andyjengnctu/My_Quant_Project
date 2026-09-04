"""User-adjustable optimizer execution/performance policy values.

These knobs may change execution strategy or resource use, but must not change
sample membership/order, seeds, optimizer updates, loss, determinism, or other
scientific semantics. Runtime resolution belongs to ``core.training_performance``.
"""

# Rolling fold parallelism.
OPTIMIZER_ROLLING_FOLD_WORKERS = "fold_count"  # "fold_count" / "auto" or a positive integer
OPTIMIZER_ROLLING_PARALLEL_PREP_CACHE_MAX_ITEMS = 0
OPTIMIZER_FEATURE_BANK_MAX_ITEMS = 1024

# Random-seed ensemble parallelism.
OPTIMIZER_RANDOM_SEED_ENSEMBLE_PARALLEL_WORKERS = "auto"
OPTIMIZER_RANDOM_SEED_ENSEMBLE_PARALLEL_BACKEND = "process"

# Parallelism inside one optimizer search unit.
OPTIMIZER_SINGLE_FOLD_SEARCH_PARALLEL_TRIALS = 8
OPTIMIZER_SINGLE_FOLD_ALLOW_TPE_PARALLEL_SEARCH = False
OPTIMIZER_SINGLE_FOLD_LOCAL_MIN_PARALLEL_WORKERS = 4
OPTIMIZER_SINGLE_FOLD_LOCAL_MIN_PROCESS_WORKERS = 0

# Policy replay execution strategy.
OPTIMIZER_POLICY_REPLAY_PARALLEL_WORKERS = 2
OPTIMIZER_POLICY_REPLAY_PARALLEL_BACKEND = "process"
OPTIMIZER_POLICY_REPLAY_DEDUP_BY_SIGNATURE = True
OPTIMIZER_POLICY_REPLAY_CONTEXT_REUSE_ENABLED = True

# Local-min diagnostics / safe evaluation ordering.
OPTIMIZER_LOCAL_MIN_DEPENDENCY_STATS_ENABLED = True
OPTIMIZER_LOCAL_MIN_PORTFOLIO_DEPENDENCY_ORDER = "last"
OPTIMIZER_LOCAL_MIN_SIGNAL_DEPENDENCY_FIELD_ORDER = "hard_fail_first"
