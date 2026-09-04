"""Canonical recoverable Trading fill transaction journal contract."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from core.file_integrity import canonical_json_sha256
from core.runtime_domains import RUNTIME_DOMAIN_TRADING

TRADING_FILL_TRANSACTION_SCHEMA_VERSION = 1
TRADING_FILL_TRANSACTION_FILENAME = "fill_transaction.json"


def build_trading_fill_transaction_journal(
    *,
    transaction_id: str,
    created_at: str,
    account_original: dict[str, Any],
    account_target: dict[str, Any],
    order_original: dict[str, Any],
    order_target: dict[str, Any],
) -> dict[str, Any]:
    txid = str(transaction_id or "").strip()
    if not txid:
        raise ValueError("Trading fill transaction_id 不可為空")
    payload = {
        "schema_version": TRADING_FILL_TRANSACTION_SCHEMA_VERSION,
        "runtime_domain": RUNTIME_DOMAIN_TRADING,
        "transaction_id": txid,
        "created_at": str(created_at),
        "account_original_sha256": canonical_json_sha256(account_original),
        "account_target_sha256": canonical_json_sha256(account_target),
        "order_original_sha256": canonical_json_sha256(order_original),
        "order_target_sha256": canonical_json_sha256(order_target),
        "account_target": deepcopy(account_target),
        "order_target": deepcopy(order_target),
    }
    validate_trading_fill_transaction_journal(payload)
    return payload


def validate_trading_fill_transaction_journal(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise TypeError("Trading fill transaction journal 必須是 object")
    if int(payload.get("schema_version", -1)) != TRADING_FILL_TRANSACTION_SCHEMA_VERSION:
        raise ValueError("Trading fill transaction schema_version 不相容")
    if str(payload.get("runtime_domain") or "") != RUNTIME_DOMAIN_TRADING:
        raise ValueError("Trading fill transaction runtime_domain 必須是 trading")
    if not str(payload.get("transaction_id") or "").strip():
        raise ValueError("Trading fill transaction_id 不可為空")
    if not str(payload.get("created_at") or "").strip():
        raise ValueError("Trading fill transaction created_at 不可為空")
    for state_name in ("account", "order"):
        target = payload.get(f"{state_name}_target")
        if not isinstance(target, dict):
            raise ValueError(f"Trading fill transaction {state_name}_target 必須是 object")
        target_sha = str(payload.get(f"{state_name}_target_sha256") or "")
        if target_sha != canonical_json_sha256(target):
            raise ValueError(f"Trading fill transaction {state_name}_target hash 不一致")
        original_sha = str(payload.get(f"{state_name}_original_sha256") or "")
        if not original_sha:
            raise ValueError(f"Trading fill transaction 缺少 {state_name}_original_sha256")


__all__ = [
    "TRADING_FILL_TRANSACTION_SCHEMA_VERSION",
    "TRADING_FILL_TRANSACTION_FILENAME",
    "build_trading_fill_transaction_journal",
    "validate_trading_fill_transaction_journal",
]
