from __future__ import annotations


def main(*args, **kwargs):
    from .main import main as _main

    return _main(*args, **kwargs)


def run_daily_scanner(*args, **kwargs):
    from .main import run_daily_scanner as _run_daily_scanner

    return _run_daily_scanner(*args, **kwargs)


def __getattr__(name):
    if name in {"ensure_runtime_dirs", "is_insufficient_data_error", "load_strict_params", "process_single_stock", "resolve_scanner_max_workers"}:
        from . import worker as worker_module

        value = getattr(worker_module, name)
        globals()[name] = value
        return value
    if name in {"process_prepared_stock", "build_scanner_response_from_stats"}:
        from . import stock_processor as stock_processor_module

        value = getattr(stock_processor_module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ensure_runtime_dirs",
    "is_insufficient_data_error",
    "load_strict_params",
    "process_single_stock",
    "process_prepared_stock",
    "build_scanner_response_from_stats",
    "resolve_scanner_max_workers",
    "run_daily_scanner",
    "main",
]
