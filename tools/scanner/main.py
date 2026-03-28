from .scan_runner import main, run_daily_scanner

__all__ = [
    "run_daily_scanner",
    "main",
]

if __name__ == "__main__":
    raise SystemExit(main())
