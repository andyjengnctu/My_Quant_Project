"""Generic dependency-aware artifact preparation loop for Research workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from core.research_orchestration import ResearchArtifactAction, ResearchArtifactPlan


@dataclass(frozen=True)
class ResearchPreparationOutcome:
    plan: ResearchArtifactPlan
    executed: tuple[tuple[str, str, str | None, str], ...]
    waves: int


def _selected_plan(
    plan: ResearchArtifactPlan,
    required_artifact_keys: tuple[str, ...] | None,
) -> ResearchArtifactPlan:
    return plan if required_artifact_keys is None else plan.select(required_artifact_keys)


def run_research_artifact_preparation(
    *,
    plan_refresher: Callable[[], ResearchArtifactPlan],
    action_executor: Callable[[ResearchArtifactAction], None],
    required_artifact_keys: Iterable[str] | None = None,
    failure_prefix: str = "Research前置",
    max_waves: int | None = None,
    on_action: Callable[[ResearchArtifactAction], None] | None = None,
) -> ResearchPreparationOutcome:
    """Prepare deterministic artifact dependencies until READY.

    The plan is *always refreshed before the first action* and after every action.  This
    prevents stale downstream metadata (for example an unknown comparison end before a
    Dataset rebuild) from leaking across producer boundaries.  Partial/resumable work is
    represented by RESUME and is ordered exactly like BUILD/REBUILD.
    """

    requested = (
        None
        if required_artifact_keys is None
        else tuple(dict.fromkeys(str(value) for value in required_artifact_keys))
    )
    current = plan_refresher()
    selected = _selected_plan(current, requested)
    if selected.blocked:
        blocked = [item.artifact_key for item in selected.actions if item.action == "BLOCKED"]
        raise RuntimeError(f"{failure_prefix}包含BLOCKED工件: " + ", ".join(blocked))
    if selected.overall_status == "READY":
        return ResearchPreparationOutcome(plan=current, executed=(), waves=0)

    wave_limit = int(max_waves or max(2, len(selected.actions) * 3 + 3))
    executed_signatures: set[tuple[str, str, str | None, str]] = set()
    executed_order: list[tuple[str, str, str | None, str]] = []

    for wave in range(1, wave_limit + 1):
        # ``current`` is fresh on entry and is refreshed immediately after every producer.
        # This gives exactly one truth snapshot per dependency wave while guaranteeing that
        # downstream work never consumes metadata captured before an upstream mutation.
        selected = _selected_plan(current, requested)
        if selected.overall_status == "READY":
            return ResearchPreparationOutcome(
                plan=current,
                executed=tuple(executed_order),
                waves=wave - 1,
            )
        if selected.blocked:
            blocked = [item.artifact_key for item in selected.actions if item.action == "BLOCKED"]
            raise RuntimeError(
                f"{failure_prefix}重新規劃後變成BLOCKED: " + ", ".join(blocked)
            )

        action = selected.next_runnable_action(
            executed_signatures=executed_signatures
        )
        if action is None:
            remaining = [
                f"{item.artifact_key}:{item.action}"
                for item in selected.actions
                if item.action != "REUSE"
            ]
            raise RuntimeError(
                f"{failure_prefix}沒有可執行且依賴已READY的動作: "
                + ", ".join(remaining)
            )

        if on_action is not None:
            on_action(action)
        try:
            action_executor(action)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(
                f"{failure_prefix}失敗: {action.artifact_key} | action={action.action} | "
                f"path={action.path} | {type(exc).__name__}: {exc}"
            ) from exc
        executed_signatures.add(action.execution_signature)
        executed_order.append(action.execution_signature)
        # Mandatory re-plan boundary after every producer invocation.
        current = plan_refresher()

    current = plan_refresher()
    selected = _selected_plan(current, requested)
    remaining = [
        f"{item.artifact_key}:{item.action}"
        for item in selected.actions
        if item.action != "REUSE"
    ]
    raise RuntimeError(
        f"{failure_prefix}超過最大依賴波次仍未READY: " + ", ".join(remaining)
    )


__all__ = ["ResearchPreparationOutcome", "run_research_artifact_preparation"]
