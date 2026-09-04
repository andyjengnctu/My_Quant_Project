"""Trading policy validation and resolved snapshots."""
from __future__ import annotations

from dataclasses import dataclass

from config.trading import (
    TRADING_ACTIVE_STRATEGY_ID,
    TRADING_DATASET_PROFILE,
    TRADING_DL_FILTER_ENABLED,
    TRADING_DL_RANKING_ENABLED,
    TRADING_OPTIMIZER_MULTI_SEED_REQUIRED,
    TRADING_PARAM_FAMILY,
    TRADING_PARAM_SELECTOR,
)
from core.dataset_profiles import DATASET_PROFILE_SPECS, normalize_dataset_profile_key
from core.strategy_param_artifacts import POLICY_FILENAME_BY_NAME, normalize_strategy_param_family


SUPPORTED_TRADING_STRATEGY_IDS = ("full_rule_based_no_dl",)


@dataclass(frozen=True)
class TradingStrategyProfile:
    strategy_id: str
    dataset_profile: str
    param_family: str
    param_selector: str
    optimizer_requires_multi_seed: bool
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
        dl_filter_enabled=dl_filter,
        dl_ranking_enabled=dl_ranking,
    )


def get_trading_policy_snapshot() -> dict[str, object]:
    profile = get_trading_strategy_profile()
    return {
        "strategy_id": profile.strategy_id,
        "dataset_profile": profile.dataset_profile,
        "param_family": profile.param_family,
        "param_selector": profile.param_selector,
        "optimizer_requires_multi_seed": profile.optimizer_requires_multi_seed,
        "dl_filter_enabled": profile.dl_filter_enabled,
        "dl_ranking_enabled": profile.dl_ranking_enabled,
    }


__all__ = [
    "SUPPORTED_TRADING_STRATEGY_IDS",
    "TradingStrategyProfile",
    "get_trading_strategy_profile",
    "get_trading_policy_snapshot",
]
