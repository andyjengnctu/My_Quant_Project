import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.scanner import ensure_runtime_dirs, is_insufficient_data_error, load_strict_params, main, process_single_stock, resolve_scanner_max_workers, run_daily_scanner

__all__ = [
    "ensure_runtime_dirs",
    "is_insufficient_data_error",
    "load_strict_params",
    "process_single_stock",
    "resolve_scanner_max_workers",
    "run_daily_scanner",
    "main",
]

if __name__ == "__main__":
    raise SystemExit(main())
