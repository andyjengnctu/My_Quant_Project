from __future__ import annotations


def ml_optimizer_main(*args, **kwargs):
    from .ml_optimizer import main as _main

    return _main(*args, **kwargs)


def portfolio_sim_main(*args, **kwargs):
    from .portfolio_sim import main as _main

    return _main(*args, **kwargs)


def smart_downloader_main(*args, **kwargs):
    from .smart_downloader import main as _main

    return _main(*args, **kwargs)


def test_suite_main(*args, **kwargs):
    from .test_suite import main as _main

    return _main(*args, **kwargs)


def vip_scanner_main(*args, **kwargs):
    from .vip_scanner import main as _main

    return _main(*args, **kwargs)


def __getattr__(name):
    if name in {"print_yearly_return_report", "run_portfolio_simulation"}:
        from . import portfolio_sim as portfolio_sim_module

        value = getattr(portfolio_sim_module, name)
        globals()[name] = value
        return value
    if name in {"process_single_stock", "run_daily_scanner"}:
        from . import vip_scanner as vip_scanner_module

        value = getattr(vip_scanner_module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ml_optimizer_main",
    "portfolio_sim_main",
    "print_yearly_return_report",
    "run_portfolio_simulation",
    "smart_downloader_main",
    "test_suite_main",
    "vip_scanner_main",
    "process_single_stock",
    "run_daily_scanner",
]
