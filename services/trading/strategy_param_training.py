"""Trading strategy-parameter production via the canonical Optimizer engine."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from core.dataset_profiles import get_dataset_profile_label
from core.path_utils import project_relative_display_path
from core.trading_policy import build_trading_strategy_param_training_plan
from core.file_integrity import compute_file_sha256
from core.strategy_promotion import get_trading_release, load_trading_release_state
from services.optimizer.application import run_static_strategy_parameter_training
from services.trading.market_data_consumer import (
    get_trading_v2_consumer_state_sha256,
    load_trading_v2_consumer_state,
    load_trading_v2_optimizer_raw_data,
)
from services.trading.strategy_param_state import publish_trading_strategy_param_binding, get_trading_strategy_param_binding_sha256
from services.trading.strategy_promotion import record_active_trading_release_fitting, _param_manifest_path
from services.trading.strategy_param_state import (
    TRADING_PARAM_USAGE_REUSE_EXISTING,
    TRADING_PARAM_USAGE_TRAINED_CURRENT,
)


def _display_path(project_root: str | os.PathLike[str], path: str | os.PathLike[str]) -> str:
    return project_relative_display_path(path, project_root=project_root)


def run_trading_strategy_param_training(
    *,
    project_root: str | os.PathLike[str],
    environ=None,
) -> dict[str, Any]:
    """Train the configured Trading strategy params without touching Research artifacts."""
    root = Path(project_root).resolve()
    plan = build_trading_strategy_param_training_plan(root)
    expected_release_id = plan.get("release_id")
    if expected_release_id:
        get_trading_release(root, purpose="training", require_current=True)

    market_before = load_trading_v2_consumer_state(root, required=True, verify_current_view=True)
    consumer_state_sha_before = get_trading_v2_consumer_state_sha256(root)
    source_view_fingerprint_before = str(market_before["source_view_fingerprint"])

    if not bool(plan["optimizer_requires_multi_seed"]):
        raise RuntimeError("目前Trading strategy要求multi-seed producer；設定卻未啟用")

    # Producer owns the per-TRAIN random draw and records the resolved bundle.
    # Workbench orchestration must not independently sample or pin seed truth.
    result = run_static_strategy_parameter_training(
        project_root=root,
        selected_data_dir=str(root),
        output_dir=str(plan["output_dir"]),
        models_dir=str(plan["models_root"]),
        strategy_params_root=str(plan["strategy_params_root"]),
        dataset_profile_key=str(plan["dataset_profile"]),
        dataset_label=f"{get_dataset_profile_label(str(plan['dataset_profile']))} / Trading",
        param_family=str(plan["param_family"]),
        selected_policy=str(plan["param_selector"]),
        trials_per_seed=int(plan["optimizer_trials_per_seed"]),
        seed_count=int(plan["optimizer_seed_count"]),
        seed_min_agree=int(plan["optimizer_seed_min_agree"]),
        trade_train_window_months=int(plan["optimizer_train_window_months"]),
        seed_ensemble_policy_override=dict(plan["optimizer_seed_ensemble_policy"]),
        environ=environ,
        raw_data_loader=load_trading_v2_optimizer_raw_data,
        latest_data_date_override=str(market_before["market_date"]),
    )
    market_after = load_trading_v2_consumer_state(root, required=True, verify_current_view=True)
    if str(market_after["source_view_fingerprint"]) != source_view_fingerprint_before:
        raise RuntimeError("Trading V2 view identity 在 Params 訓練期間已變更；本次 Params 不得投入 Scanner")
    if get_trading_v2_consumer_state_sha256(root) != consumer_state_sha_before:
        raise RuntimeError("Trading V2 consumer state 在 Params 訓練期間已變更；本次 Params 不得投入 Scanner")
    if str(market_after.get("market_date") or "") != str(market_before.get("market_date") or ""):
        raise RuntimeError("Trading V2 latest date 在 Params 訓練期間已變更；本次 Params 不得投入 Scanner")
    current_training = get_trading_release(root, purpose="training")
    if (current_training or {}).get("release_id") != expected_release_id:
        raise RuntimeError("Trading 待啟用策略在訓練期間變更；不得發布 Params")
    selected_path = Path(str(result["selected_params_path"]))
    selected_sha = compute_file_sha256(selected_path)
    binding = publish_trading_strategy_param_binding(
        root,
        usage_mode=TRADING_PARAM_USAGE_TRAINED_CURRENT,
        purpose="training",
    )
    if str(binding.get("selected_params_sha256") or "") != selected_sha:
        raise RuntimeError("Trading Params binding 與剛產生的 selected params 不一致")
    if expected_release_id and not load_trading_release_state(root).get("pending"):
        record_active_trading_release_fitting(root, expected_release_id=expected_release_id, selected_params_sha256=selected_sha, binding_sha256=get_trading_strategy_param_binding_sha256(root, purpose="training"), manifest_sha256=compute_file_sha256(_param_manifest_path(root, purpose="training")))

    return {
        **dict(result),
        "runtime_domain": "trading",
        "strategy_id": str(plan["strategy_id"]),
        "strategy_param_profile_id": str(plan["strategy_param_profile_id"]),
        "parameter_set_identity": str(plan["parameter_set_identity"]),
        "registered_scientific_fingerprint_sha256": str(plan["registered_scientific_fingerprint_sha256"]),
        "evaluation_mode": "trade",
        "optimizer_trials_per_seed": int(plan["optimizer_trials_per_seed"]),
        "optimizer_seed_count": int(plan["optimizer_seed_count"]),
        "optimizer_seed_mode": str(plan["optimizer_seed_mode"]),
        "optimizer_seed_generator_seed": plan["optimizer_seed_generator_seed"],
        "optimizer_resolved_seeds": [int(value) for value in result["resolved_optimizer_seeds"]],
        "optimizer_train_window_months": int(plan["optimizer_train_window_months"]),
        "data_dir": "data/trading/market_data_v2",
        "market_data_source": "trading_market_data_v2_historical_latest_view",
        "models_root": _display_path(root, str(plan["models_root"])),
        "strategy_params_root": _display_path(root, str(plan["strategy_params_root"])),
        "output_dir": _display_path(root, str(plan["output_dir"])),
        "selected_params_path": _display_path(root, str(result["selected_params_path"])),
        "manifest_path": _display_path(root, str(result["manifest_path"])),
        "market_data_consumer_state_sha256": consumer_state_sha_before,
        "market_data_source_view_fingerprint": source_view_fingerprint_before,
        "param_binding_fingerprint": binding["binding_fingerprint"],
        "param_usage_mode": binding["usage_mode"],
        "param_training_data_date": binding["param_training_data_date"],
        "param_member_count": int(binding["param_member_count"]),
        "param_min_agree": int(binding["param_min_agree"]),
    }


def reuse_trading_strategy_params(
    *,
    project_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Explicitly bind the existing canonical Trading Params to current Trading data.

    The selected Params artifact is not rewritten.  Its training-date lineage stays
    intact; only the Trading usage binding advances to the current information date.
    """

    started = time.perf_counter()
    print("⏳ 套用既有 Trading Params：綁定 finalized consumer identity...", flush=True)
    root = Path(project_root).resolve()
    plan = build_trading_strategy_param_training_plan(root)
    if plan.get("release_id"):
        get_trading_release(root, purpose="training", require_current=True)
    selected_path = Path(str(plan["selected_params_path"]))
    if not selected_path.is_file():
        raise FileNotFoundError(
            "目前沒有可沿用的 Trading Params；請先選擇「重新訓練 Params」建立第一份正式參數。"
        )
    # Reuse binds an already-finalized immutable consumer identity; it does not
    # consume market frames.  Re-auditing every pinned V2 batch here made a
    # metadata-only binding action take ~minutes. Scanner preflight still
    # performs the full finalized-view verification before consuming data.
    binding = publish_trading_strategy_param_binding(
        root,
        usage_mode=TRADING_PARAM_USAGE_REUSE_EXISTING,
        verify_consumer_view=False,
        purpose="training",
    )
    plan_release = plan.get("release_id")
    if plan_release and not load_trading_release_state(root).get("pending"):
        record_active_trading_release_fitting(root, expected_release_id=str(plan_release), selected_params_sha256=compute_file_sha256(selected_path), binding_sha256=get_trading_strategy_param_binding_sha256(root, purpose="training"), manifest_sha256=compute_file_sha256(_param_manifest_path(root, purpose="training")))
    elapsed = time.perf_counter() - started
    print(f"✅ Trading Params 綁定完成 | elapsed={elapsed:.2f}s", flush=True)
    return {
        "status": "READY",
        "runtime_domain": "trading",
        "strategy_id": str(plan["strategy_id"]),
        "strategy_param_profile_id": str(plan["strategy_param_profile_id"]),
        "parameter_set_identity": str(plan["parameter_set_identity"]),
        "param_selector": str(plan["param_selector"]),
        "selected_params_path": _display_path(root, selected_path),
        "selected_params_sha256": compute_file_sha256(selected_path),
        "latest_data_date": str(binding["latest_data_date"]),
        "param_training_data_date": str(binding["param_training_data_date"]),
        "param_usage_mode": str(binding["usage_mode"]),
        "param_member_count": int(binding["param_member_count"]),
        "param_min_agree": int(binding["param_min_agree"]),
        "market_data_consumer_state_sha256": str(binding["market_data_consumer_state_sha256"]),
        "market_data_source_view_fingerprint": str(binding["market_data_source_view_fingerprint"]),
        "param_binding_fingerprint": str(binding["binding_fingerprint"]),
    }


__all__ = ["run_trading_strategy_param_training", "reuse_trading_strategy_params"]
