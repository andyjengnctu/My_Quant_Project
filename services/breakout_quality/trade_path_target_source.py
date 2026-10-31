"""Canonical capability gateway for supervised, real-execution-path DL targets.

The runtime recipe owns target selection.  This adapter owns *only* the
reusable mechanics of finding a declared producer, obtaining its seed-scoped
label materializer, and planning its canonical parameter dependencies.

Adding another real-path primitive registers one capability here, rather than
adding MR/profile-specific branches to the Research app, score loader and
training entry points.  Historical model/Target identities are not rewritten.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any

from core.breakout_quality_runtime import (
    CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_TRADE_PATH,
)
from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe


@dataclass(frozen=True)
class TeacherDependency:
    name: str
    builder_type: str
    priority: int
    inspect: Callable[..., dict[str, object]]
    execute: Callable[..., object]


@dataclass(frozen=True)
class TradePathSource:
    contract_kind: str
    loader: Callable[..., dict[str, object]]
    dependencies: tuple[TeacherDependency, ...]


def _min40_loader(*, project_root: str | Path, seed: int, authorized_build: bool):
    from services.breakout_quality.formal_mintrade40_training_source import (
        resolve_formal_mintrade40_loader_kwargs,
    )
    return resolve_formal_mintrade40_loader_kwargs(
        project_root=project_root, seed=seed, authorized_build=authorized_build,
    )


def _min40_p16_inspect(*, project_root: str | Path, seed: int):
    from services.breakout_quality.formal_mintrade40_training_source import (
        inspect_formal_mintrade40_p16_teacher,
    )
    return inspect_formal_mintrade40_p16_teacher(project_root=project_root, seed=seed)


def _min40_p16_execute(*, project_root: str | Path, seed: int):
    from services.breakout_quality.formal_mintrade40_training_source import (
        _current_p16_seed_artifact,
    )
    return _current_p16_seed_artifact(Path(project_root), seed, authorized_build=True)


def _min40_historical_inspect(*, project_root: str | Path, seed: int):
    from services.optimizer.formal_mintrade40_teacher_source import (
        inspect_formal_mintrade40_optimizer_teacher,
    )
    return inspect_formal_mintrade40_optimizer_teacher(
        project_root=project_root, seed=seed,
    )


def _min40_historical_execute(*, project_root: str | Path, seed: int):
    from services.optimizer.formal_mintrade40_teacher_source import (
        prepare_formal_mintrade40_optimizer_teacher,
    )
    return prepare_formal_mintrade40_optimizer_teacher(
        project_root=project_root, seed=seed, authorized=True,
    )


# The only currently authorized primitive.  A 60D/NetPnL/Full-cycle target
# may share these same producer dependencies; its *new* canonical materializer
# and target semantics must first be registered by their owner.
_TRADE_PATH_SOURCES: dict[str, TradePathSource] = {
    "formal_seed_mintrade40_netr": TradePathSource(
        contract_kind="formal_seed_mintrade40_netr",
        loader=_min40_loader,
        dependencies=(
            TeacherDependency(
                "formal_mintrade40_p16", "optimizer_formal_mintrade40_p16", 30,
                _min40_p16_inspect, _min40_p16_execute,
            ),
            TeacherDependency(
                "formal_mintrade40_history", "optimizer_formal_mintrade40_history", 40,
                _min40_historical_inspect, _min40_historical_execute,
            ),
        ),
    ),
}


_RETIRED_TRADE_PATH_CONTRACTS = frozenset({"daily_counterfactual_mintrade40_netr"})


def registered_trade_path_source(experiment_profile: str) -> TradePathSource | None:
    """Only the canonical runtime policy may select the source capability."""
    recipe = get_continuous_ranker_execution_recipe(str(experiment_profile))
    policy = recipe.target_policy
    if policy.materialization_mode != CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_TRADE_PATH:
        return None
    kind = str(policy.contract_kind)
    if kind in _RETIRED_TRADE_PATH_CONTRACTS:
        return None  # Retired historical identity is not an executable producer.
    source = _TRADE_PATH_SOURCES.get(kind)
    if source is None:
        raise ValueError(f"Unregistered real-path target capability: {kind!r}")
    return source


def loader_kwargs_for_trade_path(
    *, experiment_profile: str, project_root: str | Path,
    seed: int | None, authorized_build: bool = False,
) -> dict[str, object]:
    source = registered_trade_path_source(experiment_profile)
    if source is None:
        return {}
    if type(seed) is not int or seed <= 0:
        raise ValueError("Real-path model source requires an explicit positive seed")
    return source.loader(
        project_root=project_root, seed=seed, authorized_build=authorized_build,
    )


def execute_trade_path_dependency(
    *, builder_type: str, project_root: str | Path,
    seed: int, experiment_profile: str,
) -> bool:
    source = registered_trade_path_source(experiment_profile)
    if source is None:
        return False
    for requirement in source.dependencies:
        if builder_type == requirement.builder_type:
            requirement.execute(project_root=project_root, seed=seed)
            return True
    return False


__all__ = [
    "TeacherDependency", "TradePathSource", "registered_trade_path_source",
    "loader_kwargs_for_trade_path", "execute_trade_path_dependency",
]
