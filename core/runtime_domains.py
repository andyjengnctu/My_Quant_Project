"""Research/Trading runtime-domain path ownership.

Research keeps the historical project paths so existing scientific/artifact
identities do not move. Trading uses an isolated namespace while sharing the
same strategy/execution code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from config.research import RESEARCH_MARKET_DATA_CUTOFF
from core.dataset_profiles import (
    DATASET_PROFILE_FULL,
    DATASET_PROFILE_SPECS,
    get_dataset_dir,
    normalize_dataset_profile_key,
)
from core.output_paths import normalize_output_category
from core.market_data_contract import get_active_research_data_generation, get_trading_market_data_lifecycle

RUNTIME_DOMAIN_RESEARCH = "research"
RUNTIME_DOMAIN_TRADING = "trading"
RUNTIME_DOMAINS = (RUNTIME_DOMAIN_RESEARCH, RUNTIME_DOMAIN_TRADING)


@dataclass(frozen=True)
class RuntimeDomainPaths:
    domain: str
    data_dir: str
    models_root: str
    strategy_params_root: str
    outputs_root: str
    state_root: str | None


def normalize_runtime_domain(value: str) -> str:
    domain = str(value or "").strip().lower()
    if domain not in RUNTIME_DOMAINS:
        raise ValueError(f"不支援的runtime domain: {value!r}")
    return domain


def resolve_runtime_domain_paths(
    project_root: str | os.PathLike[str],
    *,
    domain: str,
    dataset_profile: str = DATASET_PROFILE_FULL,
) -> RuntimeDomainPaths:
    root = os.path.abspath(os.fspath(project_root))
    normalized_domain = normalize_runtime_domain(domain)
    profile = normalize_dataset_profile_key(dataset_profile)

    if normalized_domain == RUNTIME_DOMAIN_RESEARCH:
        return RuntimeDomainPaths(
            domain=normalized_domain,
            data_dir=os.path.abspath(get_dataset_dir(root, profile)),
            models_root=os.path.join(root, "models"),
            strategy_params_root=os.path.join(root, "models", "strategy_params"),
            outputs_root=os.path.join(root, "outputs"),
            state_root=None,
        )

    dir_name = DATASET_PROFILE_SPECS[profile]["dir_name"]
    return RuntimeDomainPaths(
        domain=normalized_domain,
        data_dir=os.path.join(root, "data", "trading", dir_name),
        models_root=os.path.join(root, "models", "trading"),
        strategy_params_root=os.path.join(root, "models", "trading", "strategy_params"),
        outputs_root=os.path.join(root, "outputs", "trading"),
        state_root=os.path.join(root, "state", "trading"),
    )


def resolve_runtime_output_dir(
    project_root: str | os.PathLike[str], *, domain: str, category: str
) -> str:
    normalized_category = normalize_output_category(category)
    paths = resolve_runtime_domain_paths(project_root, domain=domain)
    return os.path.join(paths.outputs_root, normalized_category)


def assert_runtime_write_path_is_not_research_dataset(
    project_root: str | os.PathLike[str], path: str | os.PathLike[str]
) -> None:
    root = os.path.abspath(os.fspath(project_root))
    candidate = os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(path))))
    protected = {
        os.path.normcase(os.path.realpath(os.path.abspath(get_dataset_dir(root, profile))))
        for profile in DATASET_PROFILE_SPECS
    }
    for protected_root in protected:
        try:
            inside_research = os.path.commonpath([candidate, protected_root]) == protected_root
        except ValueError:
            inside_research = False
        if inside_research:
            raise RuntimeError(
                "禁止寫入Research dataset；正式downloader只能寫Trading data namespace"
            )


def build_runtime_domain_contract_snapshot(project_root: str | os.PathLike[str]) -> dict[str, object]:
    root = os.path.abspath(os.fspath(project_root))
    research = resolve_runtime_domain_paths(root, domain=RUNTIME_DOMAIN_RESEARCH)
    trading = resolve_runtime_domain_paths(root, domain=RUNTIME_DOMAIN_TRADING)
    research_generation = get_active_research_data_generation(root)
    trading_lifecycle = get_trading_market_data_lifecycle()
    return {
        "research": {
            "data_dir": research.data_dir,
            "models_root": research.models_root,
            "strategy_params_root": research.strategy_params_root,
            "outputs_root": research.outputs_root,
            "market_data_cutoff": str(RESEARCH_MARKET_DATA_CUTOFF),
            "market_data_generation": research_generation.generation_id,
            "market_data_universe_mode": research_generation.universe_mode,
        },
        "trading": {
            "data_dir": trading.data_dir,
            "models_root": trading.models_root,
            "strategy_params_root": trading.strategy_params_root,
            "outputs_root": trading.outputs_root,
            "state_root": trading.state_root,
            "market_data_mode": trading_lifecycle.mode,
            "market_data_bootstrap_source_generation": trading_lifecycle.bootstrap_source_generation,
        },
    }


__all__ = [
    "RUNTIME_DOMAIN_RESEARCH",
    "RUNTIME_DOMAIN_TRADING",
    "RUNTIME_DOMAINS",
    "RuntimeDomainPaths",
    "normalize_runtime_domain",
    "resolve_runtime_domain_paths",
    "resolve_runtime_output_dir",
    "assert_runtime_write_path_is_not_research_dataset",
    "build_runtime_domain_contract_snapshot",
]
