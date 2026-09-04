"""Trading strategy-parameter production via the canonical Optimizer engine."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from core.dataset_profiles import get_dataset_profile_label
from core.path_utils import project_relative_display_path
from core.trading_policy import build_trading_strategy_param_training_plan
from services.optimizer.application import run_static_strategy_parameter_training


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
    }


__all__ = ["run_trading_strategy_param_training"]
