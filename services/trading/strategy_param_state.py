"""Trading-only lineage binding for canonical Optimizer strategy parameters."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from core.console_report import project_relative_display_path
from core.file_integrity import atomic_write_json, canonical_json_sha256, compute_file_sha256, load_json_strict
from core.runtime_domains import RUNTIME_DOMAIN_TRADING, resolve_runtime_domain_paths
from core.trading_policy import get_trading_strategy_profile, resolve_trading_selected_strategy_param_path
from services.trading.strategy_param_runtime import load_trading_strategy_param_runtime
from services.trading.market_data_consumer import (
    get_trading_v2_consumer_state_sha256,
    load_trading_v2_consumer_state,
)

TRADING_STRATEGY_PARAM_BINDING_SCHEMA_VERSION = 5
TRADING_STRATEGY_PARAM_BINDING_FILENAME = "trading_param_binding.json"
TRADING_PARAM_USAGE_TRAINED_CURRENT = "trained_current"
TRADING_PARAM_USAGE_REUSE_EXISTING = "reuse_existing"
TRADING_PARAM_USAGE_MODES = frozenset({TRADING_PARAM_USAGE_TRAINED_CURRENT, TRADING_PARAM_USAGE_REUSE_EXISTING})


def resolve_trading_strategy_param_binding_path(project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    profile = get_trading_strategy_profile()
    paths = resolve_runtime_domain_paths(root, domain=RUNTIME_DOMAIN_TRADING, dataset_profile=profile.dataset_profile)
    return Path(paths.strategy_params_root) / TRADING_STRATEGY_PARAM_BINDING_FILENAME


def _validate_binding(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise TypeError("Trading param binding 必須是 object")
    if int(payload.get("schema_version", -1)) != TRADING_STRATEGY_PARAM_BINDING_SCHEMA_VERSION:
        raise ValueError("Trading param binding schema 不相容；請重新套用 Trading Params")
    if str(payload.get("runtime_domain") or "") != RUNTIME_DOMAIN_TRADING:
        raise ValueError("Trading param binding runtime domain 不合法")
    for field in (
        "strategy_id",
        "param_selector",
        "selected_params_sha256",
        "market_data_consumer_state_sha256",
        "market_data_source_view_fingerprint",
        "latest_data_date",
        "param_training_data_date",
        "usage_mode",
        "param_member_count",
        "param_min_agree",
    ):
        if not str(payload.get(field) or ""):
            raise ValueError(f"Trading param binding 缺少 {field}")
    if str(payload.get("usage_mode")) not in TRADING_PARAM_USAGE_MODES:
        raise ValueError("Trading param binding usage_mode 不合法")
    member_count = int(payload.get("param_member_count") or 0)
    min_agree = int(payload.get("param_min_agree") or 0)
    if member_count < 1 or min_agree < 1 or min_agree > member_count:
        raise ValueError("Trading param binding Params ensemble metadata 不合法")
    if str(payload.get("param_training_data_date")) > str(payload.get("latest_data_date")):
        raise ValueError("Trading param binding 的 Params 訓練資料日晚於使用資料日")
    if (
        str(payload.get("usage_mode")) == TRADING_PARAM_USAGE_TRAINED_CURRENT
        and str(payload.get("param_training_data_date")) != str(payload.get("latest_data_date"))
    ):
        raise ValueError("trained_current binding 要求 Params 訓練資料日與使用資料日相同")
    core = {key: value for key, value in payload.items() if key != "binding_fingerprint"}
    if str(payload.get("binding_fingerprint") or "") != canonical_json_sha256(core):
        raise ValueError("Trading param binding fingerprint 不一致")


def _selected_param_metadata(selected_path: Path, *, expected_selector: str) -> dict[str, Any]:
    selected_payload = load_json_strict(selected_path)
    if not isinstance(selected_payload, dict):
        raise TypeError("Trading selected strategy params payload 必須是 object")
    if str(selected_payload.get("selector") or "").strip() != str(expected_selector):
        raise RuntimeError("Trading selected strategy params selector 與目前 Trading 設定不一致")
    param_runtime = load_trading_strategy_param_runtime(selected_path)
    meta = dict(selected_payload.get("meta") or {})
    walk_forward_policy = dict(meta.get("walk_forward_policy") or {})
    training_date = str(walk_forward_policy.get("latest_data_date") or "").strip()
    if not training_date:
        raise ValueError("Trading selected strategy params 缺少訓練資料日")
    return {
        "training_date": training_date,
        "member_count": int(param_runtime["member_count"]),
        "min_agree": int(param_runtime["min_agree"]),
    }


def publish_trading_strategy_param_binding(
    project_root: str | Path,
    *,
    usage_mode: str = TRADING_PARAM_USAGE_TRAINED_CURRENT,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    profile = get_trading_strategy_profile()
    normalized_usage_mode = str(usage_mode or "").strip()
    if normalized_usage_mode not in TRADING_PARAM_USAGE_MODES:
        raise ValueError(f"不支援的 Trading Params usage mode: {usage_mode!r}")
    selected_path = Path(resolve_trading_selected_strategy_param_path(root))
    if not selected_path.is_file():
        raise FileNotFoundError("Trading selected strategy params 尚未產生")
    consumer_state = load_trading_v2_consumer_state(root, required=True, verify_current_view=True)
    current_data_date = str(consumer_state["market_date"])
    param_metadata = _selected_param_metadata(
        selected_path,
        expected_selector=profile.param_selector,
    )
    param_training_data_date = str(param_metadata["training_date"])
    if param_training_data_date > current_data_date:
        raise RuntimeError(
            "Trading selected strategy params 訓練資料日晚於目前 Trading data；禁止使用未來 Params"
        )
    if normalized_usage_mode == TRADING_PARAM_USAGE_TRAINED_CURRENT and param_training_data_date != current_data_date:
        raise RuntimeError(
            "本次 Params 並非由目前 Trading data 重新訓練；若要沿用既有 Params，請選擇沿用模式"
        )
    payload: dict[str, Any] = {
        "schema_version": TRADING_STRATEGY_PARAM_BINDING_SCHEMA_VERSION,
        "runtime_domain": RUNTIME_DOMAIN_TRADING,
        "strategy_id": profile.strategy_id,
        "param_selector": profile.param_selector,
        "selected_params_path": project_relative_display_path(selected_path, project_root=root),
        "selected_params_sha256": compute_file_sha256(selected_path),
        "market_data_source": "trading_market_data_v2_historical_latest_view",
        "market_data_consumer_state_sha256": get_trading_v2_consumer_state_sha256(root),
        "market_data_source_view_fingerprint": str(consumer_state["source_view_fingerprint"]),
        "latest_data_date": current_data_date,
        "param_training_data_date": param_training_data_date,
        "usage_mode": normalized_usage_mode,
        "param_member_count": int(param_metadata["member_count"]),
        "param_min_agree": int(param_metadata["min_agree"]),
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
            raise FileNotFoundError("Trading param binding 尚未建立；請重新執行「2 套用 Trading Params」")
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
        raise RuntimeError("Trading selected params 已與 param binding 不一致；請重新套用 Trading Params")
    consumer_state = load_trading_v2_consumer_state(
        root, required=True, verify_current_view=bool(verify_dataset_content)
    )
    if str(payload.get("market_data_consumer_state_sha256") or "") != get_trading_v2_consumer_state_sha256(root):
        raise RuntimeError("Trading V2 consumer state identity 與 params binding 不一致")
    if str(payload.get("market_data_source_view_fingerprint") or "") != str(consumer_state["source_view_fingerprint"]):
        raise RuntimeError("Trading V2 view identity 與 params binding 不一致")
    if str(payload.get("latest_data_date") or "") != str(consumer_state.get("market_date") or ""):
        raise RuntimeError("Trading params binding data date 已過期")
    param_metadata = _selected_param_metadata(
        selected_path,
        expected_selector=profile.param_selector,
    )
    if str(payload.get("param_training_data_date") or "") != str(param_metadata["training_date"]):
        raise RuntimeError("Trading Params artifact 訓練資料日已與 param binding 不一致")
    if int(payload.get("param_member_count") or 0) != int(param_metadata["member_count"]):
        raise RuntimeError("Trading Params artifact member_count 已與 param binding 不一致")
    if int(payload.get("param_min_agree") or 0) != int(param_metadata["min_agree"]):
        raise RuntimeError("Trading Params artifact min_agree 已與 param binding 不一致")
    return payload


def get_trading_strategy_param_binding_sha256(project_root: str | Path) -> str:
    path = resolve_trading_strategy_param_binding_path(project_root)
    if not path.is_file():
        raise FileNotFoundError("Trading param binding 尚未建立")
    return compute_file_sha256(path)


__all__ = [
    "TRADING_STRATEGY_PARAM_BINDING_SCHEMA_VERSION",
    "TRADING_STRATEGY_PARAM_BINDING_FILENAME",
    "TRADING_PARAM_USAGE_TRAINED_CURRENT",
    "TRADING_PARAM_USAGE_REUSE_EXISTING",
    "TRADING_PARAM_USAGE_MODES",
    "resolve_trading_strategy_param_binding_path",
    "publish_trading_strategy_param_binding",
    "load_trading_strategy_param_binding",
    "get_trading_strategy_param_binding_sha256",
]
