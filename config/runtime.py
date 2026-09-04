"""User-adjustable application runtime / console policy values.

These settings may change execution strategy or verbosity, but must not change
strategy/scientific semantics. Resolution against CPU count or platform belongs
to the consuming runtime module rather than config.
"""

# Scanner runtime / progress display.
SCANNER_PROGRESS_EVERY = 25
SCANNER_AUTO_MAX_WORKERS_CAP = 8

# Optimizer session runtime / diagnostics.
OPTIMIZER_WINDOWS_AUTO_MAX_WORKERS_CAP = 8
OPTIMIZER_OTHER_AUTO_MAX_WORKERS_CAP = 6
ENABLE_OPTIMIZER_PROFILING = True
ENABLE_OPTIMIZER_PROFILE_CONSOLE_PRINT = False
OPTIMIZER_PROFILE_PRINT_EVERY_N_TRIALS = 1
