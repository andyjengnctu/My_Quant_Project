from __future__ import annotations


def main(*args, **kwargs):
    from .main import main as _main

    return _main(*args, **kwargs)


def __getattr__(name):
    if name in {"export_portfolio_reports", "print_yearly_return_report"}:
        from . import reporting as reporting_module

        value = getattr(reporting_module, name)
        globals()[name] = value
        return value
    if name in {"ensure_runtime_dirs", "is_insufficient_data_error", "load_strict_params", "run_portfolio_simulation"}:
        from . import runtime as runtime_module

        value = getattr(runtime_module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ensure_runtime_dirs",
    "is_insufficient_data_error",
    "load_strict_params",
    "run_portfolio_simulation",
    "print_yearly_return_report",
    "export_portfolio_reports",
    "main",
]
