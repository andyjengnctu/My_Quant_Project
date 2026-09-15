"""Low-level persistence reader for Trading Market Data V2 state.

This module owns only physical state loading/validation so higher-level target
resolution and V2 read views can depend on it without importing each other.
"""
from __future__ import annotations

from typing import Any

from core.file_integrity import canonical_json_sha256, load_json_strict
from core.market_data_trading_storage_contract import (
    TRADING_MARKET_DATA_V2_SCHEMA_VERSION,
    resolve_trading_market_data_v2_state_path,
)


def load_trading_market_data_v2_state(project_root, *, required: bool = False) -> dict[str, Any] | None:
    path = resolve_trading_market_data_v2_state_path(project_root)
    if not path.is_file():
        if required:
            raise FileNotFoundError("Trading Market Data V2 archive state 尚未建立")
        return None
    payload = load_json_strict(path)
    if not isinstance(payload, dict):
        raise ValueError("Trading Market Data V2 archive state 必須是 object")
    if int(payload.get("schema_version", -1)) != TRADING_MARKET_DATA_V2_SCHEMA_VERSION:
        raise ValueError("Trading Market Data V2 archive state schema 不相容")
    core = {key: value for key, value in payload.items() if key != "state_fingerprint"}
    if str(payload.get("state_fingerprint") or "") != canonical_json_sha256(core):
        raise ValueError("Trading Market Data V2 archive state fingerprint 不一致")
    return payload


__all__ = ["load_trading_market_data_v2_state"]
