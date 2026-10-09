"""Trading promotion boundary: stage safely, bind Params, explicitly activate.

Trading risk/position management is NOT gated here.  Guards apply only to NEW
signals and new broker BUY orders, and preserve the pre-promotion legacy state.
"""
from __future__ import annotations

from pathlib import Path

from core.file_integrity import compute_file_sha256
from core.strategy_param_artifacts import resolve_strategy_param_manifest_path
from core.strategy_promotion import (
    commit_trading_release,
    get_trading_release,
    load_trading_release_state,
    stage_trading_strategy_rule, record_active_release_fitting,
)
from services.trading.state_lock import serialized_trading_state_mutation
from core.trading_policy import (
    get_trading_strategy_profile, resolve_trading_strategy_params_root,
    resolve_trading_selected_strategy_param_path,
)
from services.trading.strategy_param_state import (
    get_trading_strategy_param_binding_sha256,
    load_trading_strategy_param_binding,
)


def _param_manifest_path(root: str | Path, *, purpose: str = "active") -> Path:
    profile = get_trading_strategy_profile(root, purpose=purpose)
    return resolve_strategy_param_manifest_path(
        root, family=profile.param_family, evaluation_mode="trade",
        strategy_params_root=resolve_trading_strategy_params_root(root, purpose=purpose),
    )


@serialized_trading_state_mutation
def stage_trading_release(root: str | Path, arm_id: str) -> dict:
    """Prepare a new isolated fitting namespace, never switch active."""
    return stage_trading_strategy_rule(root, arm_id)


@serialized_trading_state_mutation
def activate_staged_trading_release(root: str | Path) -> dict:
    """Explicitly switch after the staged Parameters and V2 binding are ready."""
    root = Path(root).resolve()
    state = load_trading_release_state(root)
    if not isinstance(state.get("pending"), dict):
        raise RuntimeError("尚無待啟用 Trading 策略；請先選擇策略並重新訓練 Params")
    pending = get_trading_release(root, purpose="training", require_current=True)
    assert pending is not None
    selected_path = Path(resolve_trading_selected_strategy_param_path(root, purpose="training"))
    if not selected_path.is_file():
        raise RuntimeError("待啟用策略尚未重新訓練獨立的 Trading Params")
    manifest = _param_manifest_path(root, purpose="training")
    if not manifest.is_file():
        raise RuntimeError("待啟用策略尚缺 Optimizer 原生 manifest；禁止升格")
    binding = load_trading_strategy_param_binding(root, purpose="training", required=True, verify_current=True)
    if str(binding.get("promotion_release_id")) != str(pending["release_id"]):
        raise RuntimeError("待啟用 Params binding 的 Release ID 不一致")
    return commit_trading_release(
        root,
        str(pending["release_id"]),
        selected_params_sha256=compute_file_sha256(selected_path),
        binding_sha256=get_trading_strategy_param_binding_sha256(root, purpose="training"),
        manifest_sha256=compute_file_sha256(manifest),
    )


@serialized_trading_state_mutation
def record_active_trading_release_fitting(
    root: str | Path, *, expected_release_id: str | None,
    selected_params_sha256: str, binding_sha256: str, manifest_sha256: str,
) -> None:
    """Stage fitting may not alter the current active release."""
    if expected_release_id is None:
        return
    state = load_trading_release_state(root)
    if state.get("pending"):
        raise RuntimeError("仍有待啟用策略；不可改寫 ACTIVE Release fitting")
    record_active_release_fitting(
        root, expected_release_id=expected_release_id,
        selected_params_sha256=selected_params_sha256, binding_sha256=binding_sha256,
        manifest_sha256=manifest_sha256,
    )


def assert_active_trading_release_ready(root: str | Path) -> dict | None:
    """Fail closed for new Scanner BUYs; do not guard exits or held positions."""
    root = Path(root).resolve()
    state = load_trading_release_state(root)
    if state.get("active") is None:
        # Existing pre-promotion Trading remains untouched until user explicitly
        # opts into this workflow.  All promoted releases are strictly guarded.
        return None
    active = get_trading_release(root, purpose="active", require_current=True)
    assert active is not None
    selected_path = Path(resolve_trading_selected_strategy_param_path(root))
    if not selected_path.is_file() or compute_file_sha256(selected_path) != active.get("selected_params_sha256"):
        raise RuntimeError("Trading ACTIVE Release Params 已改變；禁止產生新候選或新買單")
    manifest = _param_manifest_path(root)
    if not manifest.is_file() or compute_file_sha256(manifest) != active.get("param_manifest_sha256"):
        raise RuntimeError("Trading ACTIVE Release Optimizer manifest 已改變；禁止新買單")
    binding = load_trading_strategy_param_binding(root, required=True, verify_current=True)
    binding_sha = get_trading_strategy_param_binding_sha256(root)
    if binding_sha != active.get("binding_sha256_at_activation"):
        raise RuntimeError("Trading ACTIVE Release binding 已改變；禁止新買單")
    if binding.get("promotion_release_id") != active["release_id"] or binding.get("strategy_rule_sha256") != active["rule_sha256"]:
        raise RuntimeError("Trading ACTIVE Release 規則身分與 Params 不一致")
    return active


__all__ = [
    "stage_trading_release", "activate_staged_trading_release", "record_active_trading_release_fitting",
    "assert_active_trading_release_ready",
]
