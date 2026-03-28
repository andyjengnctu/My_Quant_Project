import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.portfolio_sim import ensure_runtime_dirs, is_insufficient_data_error, load_strict_params, main, print_yearly_return_report, run_portfolio_simulation

__all__ = [
    "ensure_runtime_dirs",
    "is_insufficient_data_error",
    "load_strict_params",
    "print_yearly_return_report",
    "run_portfolio_simulation",
    "main",
]

if __name__ == "__main__":
    raise SystemExit(main())
