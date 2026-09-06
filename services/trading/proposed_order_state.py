"""Canonical persisted Trading proposed-order artifact contract.

The producer lives in :mod:`services.trading.order_planning`.  This module owns
only the stable schema/path/load/read-model contract so operational consumers
can inspect the artifact without depending on the allocation producer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from core.console_report import project_relative_display_path
from core.file_integrity import canonical_json_sha256, compute_file_sha256, load_json_strict
from core.runtime_domains import RUNTIME_DOMAIN_TRADING, resolve_runtime_output_dir
from services.trading.account_state import load_trading_account_state
from services.trading.scanner_state import (
    load_trading_scanner_runtime,
    resolve_trading_candidate_snapshot_path,
)

PROPOSED_ORDER_SCHEMA_VERSION = 2
PROPOSED_ORDER_STATUS = "PROPOSED"


def resolve_trading_proposed_orders_dir(project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    return Path(resolve_runtime_output_dir(root, domain=RUNTIME_DOMAIN_TRADING, category="proposed_orders"))


def resolve_trading_proposed_orders_json_path(project_root: str | Path) -> Path:
    return resolve_trading_proposed_orders_dir(project_root) / "proposed_orders.json"


def resolve_trading_proposed_orders_text_path(project_root: str | Path) -> Path:
    return resolve_trading_proposed_orders_dir(project_root) / "proposed_orders.txt"


def load_current_trading_proposed_order_plan(
    project_root: str | Path,
    *,
    require_current: bool = True,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    path = resolve_trading_proposed_orders_json_path(root)
    if not path.is_file():
        raise FileNotFoundError("Trading 建議掛單尚未產生；請先執行「4 建議掛單」。")
    payload = load_json_strict(path)
    if not isinstance(payload, dict):
        raise RuntimeError("Trading proposed-order payload 不合法")
    if int(payload.get("schema_version", -1)) != PROPOSED_ORDER_SCHEMA_VERSION:
        raise RuntimeError("Trading proposed-order schema_version 不相容，請重新產生建議掛單")
    if str(payload.get("status") or "") != PROPOSED_ORDER_STATUS:
        raise RuntimeError("Trading proposed-order status 不合法")
    if str(payload.get("runtime_domain") or "") != RUNTIME_DOMAIN_TRADING:
        raise RuntimeError("Trading proposed-order runtime domain 不合法")
    payload_core = {key: value for key, value in payload.items() if key != "plan_fingerprint"}
    expected_fingerprint = canonical_json_sha256(payload_core)
    if str(payload.get("plan_fingerprint") or "") != expected_fingerprint:
        raise RuntimeError("Trading proposed-order plan fingerprint 不一致")
    if not require_current:
        return payload

    runtime = load_trading_scanner_runtime(root, verify_dataset_content=True)
    if str(payload.get("information_date") or "") != str(runtime["latest_data_date"]):
        raise RuntimeError("Trading 建議掛單資料日期已過期；請重新執行 3 Scanner 與 4 建議掛單")
    if str(payload.get("strategy_id") or "") != str(runtime["profile"].strategy_id):
        raise RuntimeError("Trading 建議掛單策略與目前設定不一致")
    if str(payload.get("param_selector") or "") != str(runtime["profile"].param_selector):
        raise RuntimeError("Trading 建議掛單 selector 與目前設定不一致")
    if str(payload.get("selected_params_sha256") or "") != compute_file_sha256(runtime["selected_path"]):
        raise RuntimeError("Trading 建議掛單對應的 params 已改變；請重新執行 Scanner／建議掛單")
    snapshot_path = resolve_trading_candidate_snapshot_path(root)
    if not snapshot_path.is_file() or str(payload.get("candidate_snapshot_sha256") or "") != compute_file_sha256(snapshot_path):
        raise RuntimeError("Trading 建議掛單對應的 Scanner snapshot 已改變；請重新執行建議掛單")
    account = load_trading_account_state(root, required=True)
    if int(payload.get("account_revision", -1)) != int(account["revision"]):
        raise RuntimeError("Trading account 已與建議掛單使用的 revision 不一致；請重新產生建議掛單")
    return payload


def get_trading_proposed_order_plan_read_model(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    json_path = resolve_trading_proposed_orders_json_path(root)
    text_path = resolve_trading_proposed_orders_text_path(root)
    if not json_path.is_file():
        return {
            "exists": False,
            "valid": False,
            "fresh": False,
            "status": None,
            "information_date": None,
            "order_count": 0,
            "account_revision": None,
            "plan_fingerprint": None,
            "error": None,
            "json_path": project_relative_display_path(json_path, project_root=root),
            "text_path": project_relative_display_path(text_path, project_root=root),
        }
    try:
        payload = load_current_trading_proposed_order_plan(root, require_current=False)
    except (OSError, FileNotFoundError, TypeError, ValueError, RuntimeError) as exc:
        return {
            "exists": True,
            "valid": False,
            "fresh": False,
            "status": None,
            "information_date": None,
            "order_count": 0,
            "account_revision": None,
            "plan_fingerprint": None,
            "error": f"{type(exc).__name__}: {exc}",
            "json_path": project_relative_display_path(json_path, project_root=root),
            "text_path": project_relative_display_path(text_path, project_root=root),
        }
    freshness_error = None
    try:
        load_current_trading_proposed_order_plan(root, require_current=True)
    except (OSError, FileNotFoundError, TypeError, ValueError, RuntimeError) as exc:
        freshness_error = f"{type(exc).__name__}: {exc}"
    return {
        "exists": True,
        "valid": True,
        "fresh": freshness_error is None,
        "status": payload.get("status"),
        "information_date": payload.get("information_date"),
        "order_count": len(payload.get("orders") or []),
        "account_revision": payload.get("account_revision"),
        "plan_fingerprint": payload.get("plan_fingerprint"),
        "reserved_total": payload.get("reserved_total"),
        "error": freshness_error,
        "json_path": project_relative_display_path(json_path, project_root=root),
        "text_path": project_relative_display_path(text_path, project_root=root),
    }


__all__ = [
    "PROPOSED_ORDER_SCHEMA_VERSION",
    "PROPOSED_ORDER_STATUS",
    "resolve_trading_proposed_orders_dir",
    "resolve_trading_proposed_orders_json_path",
    "resolve_trading_proposed_orders_text_path",
    "load_current_trading_proposed_order_plan",
    "get_trading_proposed_order_plan_read_model",
]
