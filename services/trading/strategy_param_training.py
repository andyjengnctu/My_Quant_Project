"""Trading strategy-parameter production via the canonical Optimizer engine."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from core.dataset_profiles import get_dataset_profile_label
from core.path_utils import project_relative_display_path
from core.trading_policy import build_trading_strategy_param_training_plan
from core.file_integrity import compute_file_sha256
from services.optimizer.application import run_static_strategy_parameter_training
from services.trading.market_data_state import (
    get_trading_market_data_snapshot_sha256,
    load_trading_market_data_snapshot,
)
from services.trading.strategy_param_state import publish_trading_strategy_param_binding


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
    if not bool(plan["optimizer_requires_multi_seed"]):
        raise RuntimeError("目前Trading strategy要求multi-seed producer；設定卻未啟用")

    market_before = load_trading_market_data_snapshot(root, required=True, verify_dataset_content=True)
    market_snapshot_sha_before = get_trading_market_data_snapshot_sha256(root)
    dataset_content_sha_before = str(market_before["dataset_fingerprint"]["csv_content_sha256"])

    result = run_static_strategy_parameter_training(
        project_root=root,
        selected_data_dir=str(plan["data_dir"]),
        output_dir=str(plan["output_dir"]),
        models_dir=str(plan["models_root"]),
        strategy_params_root=str(plan["strategy_params_root"]),
        dataset_profile_key=str(plan["dataset_profile"]),
        dataset_label=f"{get_dataset_profile_label(str(plan['dataset_profile']))} / Trading",
        param_family=str(plan["param_family"]),
        selected_policy=str(plan["param_selector"]),
        trials_per_seed=int(plan["optimizer_trials_per_seed"]),
        seed_count=int(plan["optimizer_seed_count"]),
        seed_min_agree=plan["optimizer_seed_min_agree"],
        trade_train_window_months=int(plan["optimizer_train_window_months"]),
        environ=environ,
    )
    market_after = load_trading_market_data_snapshot(root, required=True, verify_dataset_content=True)
    if str(market_after["dataset_fingerprint"]["csv_content_sha256"]) != dataset_content_sha_before:
        raise RuntimeError("Trading dataset content 在 Params 訓練期間已變更；本次 Params 不得投入 Scanner")
    if str(market_after.get("market_date") or "") != str(market_before.get("market_date") or ""):
        raise RuntimeError("Trading dataset latest date 在 Params 訓練期間已變更；本次 Params 不得投入 Scanner")
    selected_path = Path(str(result["selected_params_path"]))
    selected_sha = compute_file_sha256(selected_path)
    binding = publish_trading_strategy_param_binding(root)
    if str(binding.get("selected_params_sha256") or "") != selected_sha:
        raise RuntimeError("Trading Params binding 與剛產生的 selected params 不一致")

    return {
        **dict(result),
        "runtime_domain": "trading",
        "strategy_id": str(plan["strategy_id"]),
        "data_dir": _display_path(root, str(plan["data_dir"])),
        "models_root": _display_path(root, str(plan["models_root"])),
        "strategy_params_root": _display_path(root, str(plan["strategy_params_root"])),
        "output_dir": _display_path(root, str(plan["output_dir"])),
        "selected_params_path": _display_path(root, str(result["selected_params_path"])),
        "manifest_path": _display_path(root, str(result["manifest_path"])),
        "market_data_snapshot_sha256": market_snapshot_sha_before,
        "dataset_content_sha256": dataset_content_sha_before,
        "param_binding_fingerprint": binding["binding_fingerprint"],
    }


__all__ = ["run_trading_strategy_param_training"]
