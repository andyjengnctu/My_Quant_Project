"""Strategy Compare application/service boundary for Research.

User-facing Research entry points call this module rather than importing
Strategy Compare implementation modules from ``filters/`` directly.  The
service owns application-level dispatch and the fail-closed boundary for the
historical runtime-integration gate; scientific settings remain in ``config/``
and comparison/model/optimizer implementations remain in their canonical
owners.
"""

from __future__ import annotations

from typing import Callable

from config.strategy_compare import (
    get_strategy_comparison_menu_profiles,
    get_strategy_comparison_settings,
    get_strategy_multi_seed_robustness_profiles,
    get_strategy_multi_seed_robustness_settings,
    get_strategy_runtime_integration_settings,
)


_RUNTIME_INTEGRATION_MUTATING_ACTIONS = frozenset({"run", "gate", "promote", "apply", "refresh"})
_RUNTIME_INTEGRATION_READ_ONLY_ACTIONS = frozenset({"status", "show", "latest"})


def resolve_strategy_comparison_execution(profile_id: str):
    """Resolve one comparison profile and its auditable dependency plan."""

    from filters.breakout_quality.strategy_comparison import (
        render_execution_plan,
        resolve_comparison_plan,
    )

    settings = get_strategy_comparison_settings(str(profile_id))
    resolved_plan = resolve_comparison_plan(settings=settings)
    rendered_plan = render_execution_plan(
        settings=settings,
        status=resolved_plan.status_dict(),
    )
    return settings, resolved_plan, rendered_plan


def execute_strategy_comparison(
    *,
    resolved_plan,
    settings,
    producer_handlers: dict[str, Callable] | None = None,
) -> dict:
    """Execute an already-resolved comparison through the canonical engine."""

    from filters.breakout_quality.strategy_comparison import run_strategy_comparison

    return run_strategy_comparison(
        resolved_plan=resolved_plan,
        auto_prepare=True,
        settings=settings,
        quiet=True,
        producer_handlers=dict(producer_handlers or {}),
    )


def show_strategy_comparison_profile_status(*, profile_id: str) -> None:
    """Render readiness for one configured Strategy Compare profile."""

    from filters.breakout_quality.strategy_comparison import show_strategy_comparison_status

    settings = get_strategy_comparison_settings(str(profile_id))
    show_strategy_comparison_status(settings=settings)


def run_strategy_multi_seed_robustness(
    *,
    robustness_id: str,
    confirm: bool,
    model_upstream_preparer: Callable[[], int] | None = None,
) -> dict:
    """Run one configured robustness profile through its canonical implementation."""

    from filters.breakout_quality.strategy_multi_seed_robustness import run_multi_seed_robustness

    return run_multi_seed_robustness(
        robustness_id=str(robustness_id),
        confirm=bool(confirm),
        model_upstream_preparer=model_upstream_preparer,
    )


def show_strategy_multi_seed_robustness_status(*, robustness_id: str) -> None:
    from filters.breakout_quality.strategy_multi_seed_robustness import (
        show_multi_seed_robustness_status,
    )

    show_multi_seed_robustness_status(robustness_id=str(robustness_id))


def show_latest_strategy_multi_seed_robustness_report(*, robustness_id: str) -> None:
    from filters.breakout_quality.strategy_multi_seed_robustness import (
        show_latest_multi_seed_robustness_report,
    )

    show_latest_multi_seed_robustness_report(robustness_id=str(robustness_id))


def runtime_integration_execution_enabled() -> bool:
    """Return whether the gate is both enabled and bound only to current profiles."""

    cfg = get_strategy_runtime_integration_settings()
    current_profiles = {row["profile_id"] for row in get_strategy_comparison_menu_profiles()}
    current_robustness = {row["robustness_id"] for row in get_strategy_multi_seed_robustness_profiles()}
    return bool(
        cfg.enabled
        and cfg.selection_profile_id in current_profiles
        and cfg.forward_profile_id in current_profiles
        and cfg.selection_robustness_id in current_robustness
        and cfg.forward_robustness_id in current_robustness
    )


def runtime_integration_allowed_actions() -> tuple[str, ...]:
    """Return the CLI-visible actions allowed by current config and profile reachability."""

    if runtime_integration_execution_enabled():
        return ("run", "promote", "status", "latest")
    return ("status", "latest")


def dispatch_runtime_integration_action(action: str) -> None:
    """Dispatch the legacy runtime gate with a config-authoritative fail-closed guard.

    Read-only historical inspection remains available for compatibility.  Any
    action that can create gate output or change runtime state is rejected before
    importing the producer when the integration is disabled in config.
    """

    normalized = str(action).strip().lower()
    supported = _RUNTIME_INTEGRATION_MUTATING_ACTIONS | _RUNTIME_INTEGRATION_READ_ONLY_ACTIONS
    if normalized not in supported:
        raise ValueError(f"compare integration不支援的命令: {action}")

    cfg = get_strategy_runtime_integration_settings()
    if normalized in _RUNTIME_INTEGRATION_MUTATING_ACTIONS and not runtime_integration_execution_enabled():
        raise RuntimeError(
            f"{cfg.label}目前未綁定可執行的current Strategy Compare profiles；"
            "historical compatibility只允許status/latest唯讀檢視。"
        )

    from filters.breakout_quality.runtime_integration_gate import (
        run_runtime_integration_gate,
        show_latest_runtime_integration_report,
        show_runtime_integration_status,
    )

    if normalized in {"run", "gate"}:
        run_runtime_integration_gate()
        return
    if normalized in {"promote", "apply", "refresh"}:
        from services.breakout_quality.runtime_promotion import (
            apply_or_refresh_runtime_promotion,
        )

        apply_or_refresh_runtime_promotion()
        return
    if normalized in {"status", "show"}:
        show_runtime_integration_status()
        return
    show_latest_runtime_integration_report()


__all__ = [
    "dispatch_runtime_integration_action",
    "execute_strategy_comparison",
    "resolve_strategy_comparison_execution",
    "run_strategy_multi_seed_robustness",
    "runtime_integration_allowed_actions",
    "runtime_integration_execution_enabled",
    "show_latest_strategy_multi_seed_robustness_report",
    "show_strategy_comparison_profile_status",
    "show_strategy_multi_seed_robustness_status",
]
