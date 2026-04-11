from __future__ import annotations


def main(*args, **kwargs):
    from .trade_log import main as _main

    return _main(*args, **kwargs)


def run_trade_analysis(*args, **kwargs):
    from .trade_log import run_trade_analysis as _run_trade_analysis

    return _run_trade_analysis(*args, **kwargs)


def run_trade_backtest(*args, **kwargs):
    from .trade_log import run_trade_backtest as _run_trade_backtest

    return _run_trade_backtest(*args, **kwargs)


def run_prepared_trade_backtest(*args, **kwargs):
    from .trade_log import run_prepared_trade_backtest as _run_prepared_trade_backtest

    return _run_prepared_trade_backtest(*args, **kwargs)


def run_ticker_analysis(*args, **kwargs):
    from .trade_log import run_ticker_analysis as _run_ticker_analysis

    return _run_ticker_analysis(*args, **kwargs)


def run_debug_analysis(*args, **kwargs):
    return run_trade_analysis(*args, **kwargs)


def run_debug_backtest(*args, **kwargs):
    return run_trade_backtest(*args, **kwargs)


def run_debug_prepared_backtest(*args, **kwargs):
    return run_prepared_trade_backtest(*args, **kwargs)


def run_debug_ticker_analysis(*args, **kwargs):
    return run_ticker_analysis(*args, **kwargs)


__all__ = [
    "run_trade_analysis",
    "run_trade_backtest",
    "run_prepared_trade_backtest",
    "run_ticker_analysis",
    "run_debug_analysis",
    "run_debug_backtest",
    "run_debug_prepared_backtest",
    "run_debug_ticker_analysis",
    "main",
]
