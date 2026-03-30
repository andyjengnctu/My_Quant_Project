import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.scanner.scan_runner import main, run_daily_scanner

__all__ = [
    "run_daily_scanner",
    "main",
]

if __name__ == "__main__":
    raise SystemExit(main())
