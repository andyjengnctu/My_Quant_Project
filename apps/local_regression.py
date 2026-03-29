import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.local_regression.run_all import main

__all__ = ["main"]

if __name__ == "__main__":
    raise SystemExit(main())
