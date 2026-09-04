"""Trading policy validation and resolved snapshots."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config.trading import (
    TRADING_ACTIVE_STRATEGY_ID,
    TRADING_DATASET_PROFILE,
    TRADING_DL_FILTER_ENABLED,
    TRADING_DL_RANKING_ENABLED,
    TRADING_OPTIMIZER_MULTI_SEED_REQUIRED,
    TRADING_OPTIMIZER_TRIALS_PER_SEED,
    TRADING_OPTIMIZER_SEED_COUNT,
    TRADING_OPTIMIZER_SEED_MIN_AGREE,
    TRADING_OPTIMIZER_TRAIN_WINDOW_MONTHS,
    TRADING_PARAM_FAMILY,
    TRADING_PARAM_SELECTOR,
)
from core.dataset_profiles import DATASET_PROFILE_SPECS, normalize_dataset_profile_key
from core.strategy_param_artifacts import (
    POLICY_FILENAME_BY_NAME,
    normalize_strategy_param_family,
    resolve_strategy_param_artifact_path,
)
from core.runtime_domains import RUNTIME_DOMAIN_TRADING, resolve_runtime_domain_paths


SUPPORTED_TRADING_STRATEGY_IDS = ("full_rule_based_no_dl",)


@dataclass(frozen=True)
class TradingStrategyProfile:
    strategy_id: str
    dataset_profile: str
    param_family: str
    param_selector: str
    optimizer_requires_multi_seed: bool
    optimizer_trials_per_seed: int
    optimizer_seed_count: int
    optimizer_seed_min_agree: object
    optimizer_train_window_months: int
    dl_filter_enabled: bool
    dl_ranking_enabled: bool


def get_trading_strategy_profile() -> TradingStrategyProfile:
    strategy_id = str(TRADING_ACTIVE_STRATEGY_ID).strip()
    if strategy_id not in SUPPORTED_TRADING_STRATEGY_IDS:
        raise ValueError(f"不支援的Trading strategy: {strategy_id!r}")

    dataset_profile = normalize_dataset_profile_key(TRADING_DATASET_PROFILE)
    if dataset_profile not in DATASET_PROFILE_SPECS:
        raise ValueError(f"不支援的Trading dataset profile: {dataset_profile!r}")

    family = normalize_strategy_param_family(TRADING_PARAM_FAMILY)
    selector = str(TRADING_PARAM_SELECTOR).strip()
    if selector not in POLICY_FILENAME_BY_NAME:
        raise ValueError(f"不支援的Trading strategy param selector: {selector!r}")

    trials_per_seed = int(TRADING_OPTIMIZER_TRIALS_PER_SEED)
    seed_count = int(TRADING_OPTIMIZER_SEED_COUNT)
    train_window_months = int(TRADING_OPTIMIZER_TRAIN_WINDOW_MONTHS)
    if trials_per_seed < 1:
        raise ValueError("Trading optimizer trials_per_seed必須>=1")
    if seed_count < 2 and bool(TRADING_OPTIMIZER_MULTI_SEED_REQUIRED):
        raise ValueError("Trading multi-seed策略要求seed_count>=2")
    if train_window_months < 1:
        raise ValueError("Trading optimizer train_window_months必須>=1")

    dl_filter = bool(TRADING_DL_FILTER_ENABLED)
    dl_ranking = bool(TRADING_DL_RANKING_ENABLED)
    if strategy_id == "full_rule_based_no_dl" and (dl_filter or dl_ranking):
        raise ValueError("full_rule_based_no_dl禁止啟用DL filter/ranking")

    return TradingStrategyProfile(
        strategy_id=strategy_id,
        dataset_profile=dataset_profile,
        param_family=family,
        param_selector=selector,
        optimizer_requires_multi_seed=bool(TRADING_OPTIMIZER_MULTI_SEED_REQUIRED),
        optimizer_trials_per_seed=trials_per_seed,
        optimizer_seed_count=seed_count,
        optimizer_seed_min_agree=TRADING_OPTIMIZER_SEED_MIN_AGREE,
        optimizer_train_window_months=train_window_months,
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
            evaluation_mode="trade",
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
        "evaluation_mode": "trade",
        "optimizer_requires_multi_seed": profile.optimizer_requires_multi_seed,
        "optimizer_trials_per_seed": profile.optimizer_trials_per_seed,
        "optimizer_seed_count": profile.optimizer_seed_count,
        "optimizer_seed_min_agree": profile.optimizer_seed_min_agree,
        "optimizer_train_window_months": profile.optimizer_train_window_months,
        "selected_params_path": resolve_trading_selected_strategy_param_path(project_root),
    }


def get_trading_policy_snapshot() -> dict[str, object]:
    profile = get_trading_strategy_profile()
    return {
        "strategy_id": profile.strategy_id,
        "dataset_profile": profile.dataset_profile,
        "param_family": profile.param_family,
        "param_selector": profile.param_selector,
        "optimizer_requires_multi_seed": profile.optimizer_requires_multi_seed,
        "optimizer_trials_per_seed": profile.optimizer_trials_per_seed,
        "optimizer_seed_count": profile.optimizer_seed_count,
        "optimizer_seed_min_agree": profile.optimizer_seed_min_agree,
        "optimizer_train_window_months": profile.optimizer_train_window_months,
        "dl_filter_enabled": profile.dl_filter_enabled,
        "dl_ranking_enabled": profile.dl_ranking_enabled,
    }


__all__ = [
    "SUPPORTED_TRADING_STRATEGY_IDS",
    "TradingStrategyProfile",
    "get_trading_strategy_profile",
    "resolve_trading_selected_strategy_param_path",
    "build_trading_strategy_param_training_plan",
    "get_trading_policy_snapshot",
]
