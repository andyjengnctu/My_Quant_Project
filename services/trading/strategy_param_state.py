"""Trading-only lineage binding for canonical Optimizer strategy parameters."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from core.console_report import project_relative_display_path
from core.file_integrity import atomic_write_json, canonical_json_sha256, compute_file_sha256, load_json_strict
from core.runtime_domains import RUNTIME_DOMAIN_TRADING, resolve_runtime_domain_paths
from core.trading_policy import get_trading_strategy_profile, resolve_trading_selected_strategy_param_path
from services.trading.market_data_state import (
    get_trading_market_data_snapshot_sha256,
    load_trading_market_data_snapshot,
)

TRADING_STRATEGY_PARAM_BINDING_SCHEMA_VERSION = 1
TRADING_STRATEGY_PARAM_BINDING_FILENAME = "trading_param_binding.json"


def resolve_trading_strategy_param_binding_path(project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    profile = get_trading_strategy_profile()
    paths = resolve_runtime_domain_paths(root, domain=RUNTIME_DOMAIN_TRADING, dataset_profile=profile.dataset_profile)
    return Path(paths.strategy_params_root) / TRADING_STRATEGY_PARAM_BINDING_FILENAME


def _validate_binding(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise TypeError("Trading param binding 必須是 object")
    if int(payload.get("schema_version", -1)) != TRADING_STRATEGY_PARAM_BINDING_SCHEMA_VERSION:
        raise ValueError("Trading param binding schema 不相容；請重新更新 Trading Params")
    if str(payload.get("runtime_domain") or "") != RUNTIME_DOMAIN_TRADING:
        raise ValueError("Trading param binding runtime domain 不合法")
    for field in ("strategy_id", "param_selector", "selected_params_sha256", "market_data_snapshot_sha256", "dataset_content_sha256", "latest_data_date"):
        if not str(payload.get(field) or ""):
            raise ValueError(f"Trading param binding 缺少 {field}")
    core = {key: value for key, value in payload.items() if key != "binding_fingerprint"}
    if str(payload.get("binding_fingerprint") or "") != canonical_json_sha256(core):
        raise ValueError("Trading param binding fingerprint 不一致")


def publish_trading_strategy_param_binding(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    profile = get_trading_strategy_profile()
    selected_path = Path(resolve_trading_selected_strategy_param_path(root))
    if not selected_path.is_file():
        raise FileNotFoundError("Trading selected strategy params 尚未產生")
    market_snapshot = load_trading_market_data_snapshot(root, required=True, verify_dataset_content=True)
    payload: dict[str, Any] = {
        "schema_version": TRADING_STRATEGY_PARAM_BINDING_SCHEMA_VERSION,
        "runtime_domain": RUNTIME_DOMAIN_TRADING,
        "strategy_id": profile.strategy_id,
        "param_selector": profile.param_selector,
        "selected_params_path": project_relative_display_path(selected_path, project_root=root),
        "selected_params_sha256": compute_file_sha256(selected_path),
        "market_data_snapshot_sha256": get_trading_market_data_snapshot_sha256(root),
        "dataset_content_sha256": str(market_snapshot["dataset_fingerprint"]["csv_content_sha256"]),
        "latest_data_date": str(market_snapshot["market_date"]),
    }
    payload["binding_fingerprint"] = canonical_json_sha256(payload)
    path = resolve_trading_strategy_param_binding_path(root)
    atomic_write_json(path, payload)
    return payload


def load_trading_strategy_param_binding(
    project_root: str | Path,
    *,
    required: bool = True,
    verify_current: bool = True,
    verify_dataset_content: bool = False,
) -> dict[str, Any] | None:
    root = Path(project_root).resolve()
    path = resolve_trading_strategy_param_binding_path(root)
    if not path.is_file():
        if required:
            raise FileNotFoundError("Trading param binding 尚未建立；請重新執行「2 更新 Trading Params」")
        return None
    payload = load_json_strict(path)
    _validate_binding(payload)
    if not verify_current:
        return payload

    profile = get_trading_strategy_profile()
    selected_path = Path(resolve_trading_selected_strategy_param_path(root))
    if str(payload.get("strategy_id") or "") != profile.strategy_id:
        raise RuntimeError("Trading param binding strategy 已過期")
    if str(payload.get("param_selector") or "") != profile.param_selector:
        raise RuntimeError("Trading param binding selector 已過期")
    if not selected_path.is_file() or str(payload.get("selected_params_sha256") or "") != compute_file_sha256(selected_path):
        raise RuntimeError("Trading selected params 已與 param binding 不一致；請重新更新 Trading Params")
    market_snapshot = load_trading_market_data_snapshot(
        root, required=True, verify_dataset_content=verify_dataset_content
    )
    if str(payload.get("dataset_content_sha256") or "") != str(market_snapshot["dataset_fingerprint"]["csv_content_sha256"]):
        raise RuntimeError("Trading dataset content identity 與 params binding 不一致")
    if str(payload.get("latest_data_date") or "") != str(market_snapshot.get("market_date") or ""):
        raise RuntimeError("Trading params binding data date 已過期")
    return payload


def get_trading_strategy_param_binding_sha256(project_root: str | Path) -> str:
    path = resolve_trading_strategy_param_binding_path(project_root)
    if not path.is_file():
        raise FileNotFoundError("Trading param binding 尚未建立")
    return compute_file_sha256(path)


__all__ = [
    "TRADING_STRATEGY_PARAM_BINDING_SCHEMA_VERSION",
    "TRADING_STRATEGY_PARAM_BINDING_FILENAME",
    "resolve_trading_strategy_param_binding_path",
    "publish_trading_strategy_param_binding",
    "load_trading_strategy_param_binding",
    "get_trading_strategy_param_binding_sha256",
]
