"""Canonical physical paths for live Trading current-state files."""
from __future__ import annotations

from pathlib import Path

from core.runtime_domains import RUNTIME_DOMAIN_TRADING, resolve_runtime_domain_paths
from core.trading_account_state import TRADING_ACCOUNT_STATE_FILENAME
from core.trading_fill_transaction import TRADING_FILL_TRANSACTION_FILENAME
from core.trading_order_state import TRADING_ORDER_STATE_FILENAME


def resolve_trading_state_root(project_root) -> Path:
    paths = resolve_runtime_domain_paths(project_root, domain=RUNTIME_DOMAIN_TRADING)
    if paths.state_root is None:
        raise RuntimeError("Trading runtime domain 缺少 state_root")
    return Path(paths.state_root)


def resolve_trading_account_state_path(project_root) -> Path:
    return resolve_trading_state_root(project_root) / TRADING_ACCOUNT_STATE_FILENAME


def resolve_trading_order_state_path(project_root) -> Path:
    return resolve_trading_state_root(project_root) / TRADING_ORDER_STATE_FILENAME


def resolve_trading_fill_transaction_path(project_root) -> Path:
    return resolve_trading_state_root(project_root) / TRADING_FILL_TRANSACTION_FILENAME


__all__ = [
    "resolve_trading_state_root",
    "resolve_trading_account_state_path",
    "resolve_trading_order_state_path",
    "resolve_trading_fill_transaction_path",
]
