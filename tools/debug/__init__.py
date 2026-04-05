from __future__ import annotations


def main(*args, **kwargs):
    from .trade_log import main as _main

    return _main(*args, **kwargs)


def run_debug_analysis(*args, **kwargs):
    from .trade_log import run_debug_analysis as _run_debug_analysis

    return _run_debug_analysis(*args, **kwargs)


def run_debug_backtest(*args, **kwargs):
    from .trade_log import run_debug_backtest as _run_debug_backtest

    return _run_debug_backtest(*args, **kwargs)


def run_debug_prepared_backtest(*args, **kwargs):
    from .trade_log import run_debug_prepared_backtest as _run_debug_prepared_backtest

    return _run_debug_prepared_backtest(*args, **kwargs)


def run_debug_ticker_analysis(*args, **kwargs):
    from .trade_log import run_debug_ticker_analysis as _run_debug_ticker_analysis

    return _run_debug_ticker_analysis(*args, **kwargs)


__all__ = [
    "run_debug_analysis",
    "run_debug_backtest",
    "run_debug_prepared_backtest",
    "run_debug_ticker_analysis",
    "main",
]
