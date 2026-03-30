import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main(argv=None, env=None):
    from tools.scanner.scan_runner import main as scanner_main

    return scanner_main(argv=argv, env=env)


def run_daily_scanner(*args, **kwargs):
    from tools.scanner.scan_runner import run_daily_scanner as _run_daily_scanner

    return _run_daily_scanner(*args, **kwargs)


__all__ = [
    "run_daily_scanner",
    "main",
]

if __name__ == "__main__":
    raise SystemExit(main())
