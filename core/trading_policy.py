"""Trading policy validation and resolved snapshots."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config.trading import (
    TRADING_ACTIVE_STRATEGY_ID,
    TRADING_DATASET_PROFILE,
    TRADING_DL_FILTER_ENABLED,
    TRADING_DL_RANKING_ENABLED,
    TRADING_STRATEGY_PARAM_PROFILE_ID,
    TRADING_OPTIMIZER_SEED_MODE,
)
from config.training_policy import OUTER_ROLLING_TRAIN_WINDOW_MONTHS
from core.dataset_profiles import DATASET_PROFILE_SPECS, normalize_dataset_profile_key
from core.strategy_param_artifacts import (
    normalize_strategy_param_family,
    normalize_strategy_param_policy,
    resolve_strategy_param_artifact_path,
)
from core.seed_ensemble_policy import build_seed_ensemble_policy_snapshot
from core.strategy_param_policy import (
    get_strategy_param_fitting_seed_protocol,
    get_strategy_param_training_profile,
)
from core.runtime_domains import RUNTIME_DOMAIN_TRADING, resolve_runtime_domain_paths
from core.trading_data_dependencies import get_trading_data_dependency_spec


SUPPORTED_TRADING_STRATEGY_IDS = ("full_rule_based_no_dl",)
SUPPORTED_TRADING_OPTIMIZER_SEED_MODES = ("random_n_each_training_run", "fixed_generated_bundle")


@dataclass(frozen=True)
class TradingStrategyProfile:
    strategy_id: str
    dataset_profile: str
    strategy_param_profile_id: str
    parameter_set_identity: str
    registered_scientific_fingerprint_sha256: str
    param_family: str
    param_selector: str
    optimizer_requires_multi_seed: bool
    optimizer_trials_per_seed: int
    optimizer_seed_count: int
    optimizer_seed_mode: str
    optimizer_seed_generator_seed: int | None
    optimizer_seed_min_agree: int
    optimizer_resolved_seeds: tuple[int, ...]
    optimizer_train_window_months: int
    dl_filter_enabled: bool
    dl_ranking_enabled: bool

    def optimizer_seed_ensemble_policy(self) -> dict[str, object]:
        # The canonical seed-policy builder owns min-agree and random fitting
        # semantics; fixed Research seeds are not product defaults.
        policy = build_seed_ensemble_policy_snapshot(
            enabled=True,
            seed_count=self.optimizer_seed_count,
            min_agree=self.optimizer_seed_min_agree,
        )
        if self.optimizer_seed_mode == "fixed_generated_bundle":
            policy.update({
                "seed_mode": "fixed_generated_bundle",
                "generator_seed": int(self.optimizer_seed_generator_seed),
                "resolved_seeds": [int(seed) for seed in self.optimizer_resolved_seeds],
            })
        return policy


def get_trading_strategy_profile() -> TradingStrategyProfile:
    strategy_id = str(TRADING_ACTIVE_STRATEGY_ID).strip()
    if strategy_id not in SUPPORTED_TRADING_STRATEGY_IDS:
        raise ValueError(f"不支援的Trading strategy: {strategy_id!r}")
    get_trading_data_dependency_spec(strategy_id)

    promoted_profile = get_strategy_param_training_profile(TRADING_STRATEGY_PARAM_PROFILE_ID)
    if promoted_profile.strategy_id != "breakout":
        raise ValueError(
            "Trading目前只支援breakout Strategy Parameter profile: "
            f"{promoted_profile.profile_id} -> {promoted_profile.strategy_id!r}"
        )
    if promoted_profile.validation_enabled:
        raise ValueError("Trading production不得直接採用啟用validation的Strategy Parameter profile")
    if promoted_profile.price_basis != "adjusted":
        raise ValueError("Trading production目前只允許canonical adjusted-price Strategy Parameter profile")
    if not promoted_profile.registered_scientific_fingerprint_sha256:
        raise ValueError("Trading promoted Strategy Parameter profile必須已有registered scientific fingerprint")

    dataset_profile = normalize_dataset_profile_key(promoted_profile.dataset)
    configured_dataset = normalize_dataset_profile_key(TRADING_DATASET_PROFILE)
    if dataset_profile not in DATASET_PROFILE_SPECS:
        raise ValueError(f"不支援的Trading dataset profile: {dataset_profile!r}")
    if configured_dataset != dataset_profile:
        raise ValueError(
            "Trading dataset與promoted Strategy Parameter profile不一致: "
            f"trading={configured_dataset}, profile={dataset_profile}"
        )

    family = normalize_strategy_param_family(promoted_profile.filter_profile)
    selector = normalize_strategy_param_policy(promoted_profile.parameter_policy)
    fitting = get_strategy_param_fitting_seed_protocol(promoted_profile.fitting_seed_protocol)
    if not fitting.enabled or fitting.seed_count < 2:
        raise ValueError("Trading promoted Strategy Parameter profile必須使用multi-seed fitting")
    trading_seed_mode = str(TRADING_OPTIMIZER_SEED_MODE).strip()
    if trading_seed_mode not in SUPPORTED_TRADING_OPTIMIZER_SEED_MODES:
        raise ValueError(f"不支援的Trading optimizer seed mode: {trading_seed_mode!r}")
    if trading_seed_mode == "fixed_generated_bundle" and len(fitting.resolved_seeds) != fitting.seed_count:
        raise ValueError("Trading fixed fitting profile缺少完整resolved seed bundle")

    train_window_months = int(OUTER_ROLLING_TRAIN_WINDOW_MONTHS)
    if train_window_months < 1:
        raise ValueError("Trading optimizer train_window_months必須>=1")

    dl_filter = bool(TRADING_DL_FILTER_ENABLED)
    dl_ranking = bool(TRADING_DL_RANKING_ENABLED)
    if strategy_id == "full_rule_based_no_dl" and (dl_filter or dl_ranking):
        raise ValueError("full_rule_based_no_dl禁止啟用DL filter/ranking")

    return TradingStrategyProfile(
        strategy_id=strategy_id,
        dataset_profile=dataset_profile,
        strategy_param_profile_id=promoted_profile.profile_id,
        parameter_set_identity=promoted_profile.parameter_set_identity,
        registered_scientific_fingerprint_sha256=str(
            promoted_profile.registered_scientific_fingerprint_sha256
        ),
        param_family=family,
        param_selector=selector,
        optimizer_requires_multi_seed=True,
        optimizer_trials_per_seed=int(fitting.trials_per_seed),
        optimizer_seed_count=int(fitting.seed_count),
        optimizer_seed_mode=trading_seed_mode,
        optimizer_seed_generator_seed=(int(fitting.generator_seed) if trading_seed_mode == "fixed_generated_bundle" else None),
        optimizer_seed_min_agree=int(fitting.min_agree),
        optimizer_resolved_seeds=(
            tuple(int(value) for value in fitting.resolved_seeds)
            if trading_seed_mode == "fixed_generated_bundle" else ()
        ),
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
        "strategy_param_profile_id": profile.strategy_param_profile_id,
        "parameter_set_identity": profile.parameter_set_identity,
        "registered_scientific_fingerprint_sha256": profile.registered_scientific_fingerprint_sha256,
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
        "optimizer_seed_mode": profile.optimizer_seed_mode,
        "optimizer_seed_generator_seed": profile.optimizer_seed_generator_seed,
        "optimizer_seed_min_agree": profile.optimizer_seed_min_agree,
        "optimizer_resolved_seeds": list(profile.optimizer_resolved_seeds),
        "optimizer_seed_ensemble_policy": profile.optimizer_seed_ensemble_policy(),
        "optimizer_train_window_months": profile.optimizer_train_window_months,
        "selected_params_path": resolve_trading_selected_strategy_param_path(project_root),
    }


def get_trading_policy_snapshot() -> dict[str, object]:
    profile = get_trading_strategy_profile()
    dependency = get_trading_data_dependency_spec(profile.strategy_id)
    return {
        "strategy_id": profile.strategy_id,
        "dataset_profile": profile.dataset_profile,
        "strategy_param_profile_id": profile.strategy_param_profile_id,
        "parameter_set_identity": profile.parameter_set_identity,
        "registered_scientific_fingerprint_sha256": profile.registered_scientific_fingerprint_sha256,
        "param_family": profile.param_family,
        "param_selector": profile.param_selector,
        "optimizer_requires_multi_seed": profile.optimizer_requires_multi_seed,
        "optimizer_trials_per_seed": profile.optimizer_trials_per_seed,
        "optimizer_seed_count": profile.optimizer_seed_count,
        "optimizer_seed_mode": profile.optimizer_seed_mode,
        "optimizer_seed_generator_seed": profile.optimizer_seed_generator_seed,
        "optimizer_seed_min_agree": profile.optimizer_seed_min_agree,
        "optimizer_resolved_seeds": list(profile.optimizer_resolved_seeds),
        "optimizer_train_window_months": profile.optimizer_train_window_months,
        "dl_filter_enabled": profile.dl_filter_enabled,
        "dl_ranking_enabled": profile.dl_ranking_enabled,
        "data_dependency_fingerprint": dependency.fingerprint,
        "required_v2_datasets": list(dependency.required_v2_datasets),
    }


__all__ = [
    "SUPPORTED_TRADING_STRATEGY_IDS",
    "SUPPORTED_TRADING_OPTIMIZER_SEED_MODES",
    "TradingStrategyProfile",
    "get_trading_strategy_profile",
    "resolve_trading_selected_strategy_param_path",
    "build_trading_strategy_param_training_plan",
    "get_trading_policy_snapshot",
]
