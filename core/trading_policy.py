"""Trading policy validation and resolved snapshots."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import config.training_policy as _training_settings
from config.trading import (
    TRADING_ACTIVE_STRATEGY_ID,
    TRADING_DATASET_PROFILE,
    TRADING_DL_FILTER_ENABLED,
    TRADING_DL_RANKING_ENABLED,
    TRADING_PARAM_FAMILY,
)
from core.training_policy import get_strategy_parameter_training_policy_snapshot
from core.dataset_profiles import DATASET_PROFILE_SPECS, normalize_dataset_profile_key
from core.strategy_param_artifacts import (
    POLICY_FILENAME_BY_NAME,
    normalize_strategy_param_family,
    resolve_strategy_param_artifact_path,
)
from core.runtime_domains import RUNTIME_DOMAIN_TRADING, resolve_runtime_domain_paths
from core.trading_data_dependencies import get_trading_data_dependency_spec


SUPPORTED_TRADING_STRATEGY_IDS = ("full_rule_based_no_dl",)


@dataclass(frozen=True)
class TradingStrategyProfile:
    strategy_id: str
    dataset_profile: str
    param_family: str
    param_selector: str
    optimizer_trials_per_fold: int
    optimizer_seed: int
    optimizer_train_window_months: int
    optimizer_oos_horizon_months: int
    optimizer_schedule_start_date: str
    dl_filter_enabled: bool
    dl_ranking_enabled: bool


def get_trading_strategy_profile() -> TradingStrategyProfile:
    strategy_id = str(TRADING_ACTIVE_STRATEGY_ID).strip()
    if strategy_id not in SUPPORTED_TRADING_STRATEGY_IDS:
        raise ValueError(f"不支援的Trading strategy: {strategy_id!r}")
    # Every executable strategy must have exactly one canonical data dependency contract.
    get_trading_data_dependency_spec(strategy_id)

    dataset_profile = normalize_dataset_profile_key(TRADING_DATASET_PROFILE)
    if dataset_profile not in DATASET_PROFILE_SPECS:
        raise ValueError(f"不支援的Trading dataset profile: {dataset_profile!r}")

    family = normalize_strategy_param_family(TRADING_PARAM_FAMILY)
    optimizer_policy = get_strategy_parameter_training_policy_snapshot(evaluation_mode="rolling")
    selector = str(_training_settings.TRADE_MODE_RUN_BEST_SELECTOR).strip()
    if selector not in POLICY_FILENAME_BY_NAME:
        raise ValueError(f"不支援的Trading strategy param selector: {selector!r}")

    trials_per_fold = int(optimizer_policy["trials_per_fold"])
    optimizer_seed = int(optimizer_policy["optimizer_seed"])
    train_window_months = int(optimizer_policy["train_window_months"])
    oos_horizon_months = int(optimizer_policy["oos_horizon_months"])
    schedule_start_date = f"{int(_training_settings.OOS_EVALUATION_START_YEAR):04d}-01-01"
    if trials_per_fold < 1:
        raise ValueError("Trading optimizer trials_per_fold必須>=1")
    if optimizer_seed < 0:
        raise ValueError("Trading optimizer seed必須>=0")
    if train_window_months < 1 or oos_horizon_months < 1:
        raise ValueError("Trading optimizer rolling window設定必須>=1")

    dl_filter = bool(TRADING_DL_FILTER_ENABLED)
    dl_ranking = bool(TRADING_DL_RANKING_ENABLED)
    if strategy_id == "full_rule_based_no_dl" and (dl_filter or dl_ranking):
        raise ValueError("full_rule_based_no_dl禁止啟用DL filter/ranking")

    return TradingStrategyProfile(
        strategy_id=strategy_id,
        dataset_profile=dataset_profile,
        param_family=family,
        param_selector=selector,
        optimizer_trials_per_fold=trials_per_fold,
        optimizer_seed=optimizer_seed,
        optimizer_train_window_months=train_window_months,
        optimizer_oos_horizon_months=oos_horizon_months,
        optimizer_schedule_start_date=schedule_start_date,
        dl_filter_enabled=dl_filter,
        dl_ranking_enabled=dl_ranking,
    )


def resolve_trading_selected_strategy_param_path(project_root) -> str:
    profile = get_trading_strategy_profile()
    paths = resolve_runtime_domain_paths(
        project_root, domain=RUNTIME_DOMAIN_TRADING, dataset_profile=profile.dataset_profile
    )
    return str(
        resolve_strategy_param_artifact_path(
            project_root,
            family=profile.param_family,
            evaluation_mode="rolling",
            policy=profile.param_selector,
            strategy_params_root=paths.strategy_params_root,
        )
    )


def build_trading_strategy_param_training_plan(project_root) -> dict[str, object]:
    profile = get_trading_strategy_profile()
    paths = resolve_runtime_domain_paths(
        project_root, domain=RUNTIME_DOMAIN_TRADING, dataset_profile=profile.dataset_profile
    )
    return {
        "runtime_domain": RUNTIME_DOMAIN_TRADING,
        "strategy_id": profile.strategy_id,
        "dataset_profile": profile.dataset_profile,
        "data_dir": paths.data_dir,
        "models_root": paths.models_root,
        "strategy_params_root": paths.strategy_params_root,
        "output_dir": str(Path(paths.outputs_root) / "optimizer"),
        "param_family": profile.param_family,
        "param_selector": profile.param_selector,
        "evaluation_mode": "rolling",
        "optimizer_trials_per_fold": profile.optimizer_trials_per_fold,
        "optimizer_seed": profile.optimizer_seed,
        "optimizer_train_window_months": profile.optimizer_train_window_months,
        "optimizer_oos_horizon_months": profile.optimizer_oos_horizon_months,
        "optimizer_schedule_start_date": profile.optimizer_schedule_start_date,
        "selected_params_path": resolve_trading_selected_strategy_param_path(project_root),
    }


def get_trading_policy_snapshot() -> dict[str, object]:
    profile = get_trading_strategy_profile()
    dependency = get_trading_data_dependency_spec(profile.strategy_id)
    return {
        "strategy_id": profile.strategy_id,
        "dataset_profile": profile.dataset_profile,
        "param_family": profile.param_family,
        "param_selector": profile.param_selector,
        "optimizer_trials_per_fold": profile.optimizer_trials_per_fold,
        "optimizer_seed": profile.optimizer_seed,
        "optimizer_train_window_months": profile.optimizer_train_window_months,
        "optimizer_oos_horizon_months": profile.optimizer_oos_horizon_months,
        "optimizer_schedule_start_date": profile.optimizer_schedule_start_date,
        "dl_filter_enabled": profile.dl_filter_enabled,
        "dl_ranking_enabled": profile.dl_ranking_enabled,
        "data_dependency_fingerprint": dependency.fingerprint,
        "required_v2_datasets": list(dependency.required_v2_datasets),
    }


__all__ = [
    "SUPPORTED_TRADING_STRATEGY_IDS",
    "TradingStrategyProfile",
    "get_trading_strategy_profile",
    "resolve_trading_selected_strategy_param_path",
    "build_trading_strategy_param_training_plan",
    "get_trading_policy_snapshot",
]
