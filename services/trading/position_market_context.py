"""Shared completed-daily market context for live Trading strategy positions."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from core.data_utils import discover_unique_csv_map, get_required_min_rows, sanitize_ohlcv_dataframe
from core.runtime_domains import RUNTIME_DOMAIN_TRADING, resolve_runtime_domain_paths
from core.trading_account_state import POSITION_SOURCE_STRATEGY_FILL
from core.trading_identity import normalize_trading_date
from services.trading.account_state import load_trading_account_state
from services.trading.order_state import load_trading_order_state


def load_trading_position_market_frame(*, file_path: str, ticker: str, params, allowed_date: str) -> pd.DataFrame:
    raw = pd.read_csv(file_path)
    df, _stats = sanitize_ohlcv_dataframe(
        raw,
        ticker,
        min_rows=get_required_min_rows(params),
    )
    allowed_ts = pd.Timestamp(allowed_date).normalize()
    if pd.Timestamp(df.index.max()).normalize() > allowed_ts:
        raise RuntimeError(
            f"Trading position data 含尚未完成日K: {ticker} "
            f"latest={pd.Timestamp(df.index.max()).strftime('%Y-%m-%d')} > allowed={allowed_date}"
        )
    return df


def resolve_trading_strategy_position_sources(project_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    account = load_trading_account_state(project_root, required=False)
    if account is None:
        return {}, {}, {}
    orders = load_trading_order_state(project_root, required=False)
    if orders is None:
        orders = {"orders": {}}
    has_strategy_positions = any(
        isinstance(record, dict) and record.get("source") == POSITION_SOURCE_STRATEGY_FILL
        for record in (account.get("positions") or {}).values()
    )
    if not has_strategy_positions:
        return account, orders, {}
    paths = resolve_runtime_domain_paths(project_root, domain=RUNTIME_DOMAIN_TRADING)
    csv_map, duplicate_issues = discover_unique_csv_map(paths.data_dir)
    if duplicate_issues:
        raise RuntimeError("Trading dataset 存在重複ticker CSV：" + "；".join(duplicate_issues[:5]))
    return account, orders, csv_map


__all__ = [
    "normalize_trading_date",
    "load_trading_position_market_frame",
    "resolve_trading_strategy_position_sources",
]
