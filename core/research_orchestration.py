"""Generic Research artifact dependency planning contracts.

Every Research work type may declare artifact readiness independently, but orchestration
semantics (dependency closure, runnable ordering, re-plan, and BLOCKED rules) live here.
Domain modules remain the single producers of their own artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

RESEARCH_ARTIFACT_READY_ACTIONS = frozenset({"REUSE", "NOT_REQUIRED"})
RESEARCH_ARTIFACT_BUILD_ACTIONS = frozenset({"BUILD", "REBUILD", "RESUME", "MIGRATE", "DERIVE"})
RESEARCH_ARTIFACT_TERMINAL_ACTIONS = frozenset({"REUSE", "NOT_REQUIRED", "BLOCKED"})
RESEARCH_ARTIFACT_SUPPORTED_ACTIONS = (
    RESEARCH_ARTIFACT_READY_ACTIONS
    | RESEARCH_ARTIFACT_BUILD_ACTIONS
    | {"BLOCKED"}
)


def resolve_research_artifact_action(
    *,
    ready: bool,
    artifact_exists: bool,
    has_builder: bool,
    auto_prepare: bool,
    reuse_ready_artifacts: bool,
    rebuild_stale_artifacts: bool,
    resume_partial_artifacts: bool,
    resumable: bool = False,
) -> str:
    """Resolve REUSE/BUILD/REBUILD/RESUME/BLOCKED from one project-wide policy.

    Domain readiness code supplies facts only (ready/existence/resumability/builder).
    The generic Research layer owns how those facts map to an execution action, so
    model, optimizer, compare and audit workflows cannot drift into different
    missing/stale/partial semantics.
    """

    if ready and reuse_ready_artifacts:
        return "REUSE"
    if not auto_prepare or not has_builder:
        return "BLOCKED"
    if ready:
        return "REBUILD" if rebuild_stale_artifacts else "BLOCKED"
    if resumable and resume_partial_artifacts:
        return "RESUME"
    if artifact_exists:
        return "REBUILD" if rebuild_stale_artifacts else "BLOCKED"
    return "BUILD"


@dataclass(frozen=True)
class ResearchArtifactAction:
    action_id: str
    artifact_key: str
    action: str
    builder_type: str | None
    description: str
    path: str
    dependencies: tuple[str, ...] = ()
    producer_work_type: str | None = None
    execution_priority: int = 100

    def __post_init__(self) -> None:
        normalized = str(self.action).strip().upper()
        if normalized not in RESEARCH_ARTIFACT_SUPPORTED_ACTIONS:
            raise ValueError(
                f"Research artifact action不支援: {self.action!r}; "
                f"allowed={sorted(RESEARCH_ARTIFACT_SUPPORTED_ACTIONS)}"
            )
        object.__setattr__(self, "action", normalized)
        object.__setattr__(
            self,
            "dependencies",
            tuple(str(value) for value in self.dependencies),
        )

    @property
    def execution_signature(self) -> tuple[str, str, str | None, str]:
        return (self.artifact_key, self.action, self.builder_type, self.path)

    @property
    def runnable_work(self) -> bool:
        return self.action in RESEARCH_ARTIFACT_BUILD_ACTIONS

    def as_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "artifact_key": self.artifact_key,
            "action": self.action,
            "builder_type": self.builder_type,
            "description": self.description,
            "path": self.path,
            "dependencies": list(self.dependencies),
            "producer_work_type": self.producer_work_type,
            "execution_priority": int(self.execution_priority),
        }


@dataclass(frozen=True)
class ResearchArtifactPlan:
    overall_status: str
    actions: tuple[ResearchArtifactAction, ...]

    @classmethod
    def from_actions(
        cls, actions: Iterable[ResearchArtifactAction]
    ) -> "ResearchArtifactPlan":
        normalized = tuple(actions)
        cls._validate_dependencies(normalized)
        action_names = {item.action for item in normalized}
        overall_status = (
            "BLOCKED"
            if "BLOCKED" in action_names
            else "PREPARABLE"
            if action_names & RESEARCH_ARTIFACT_BUILD_ACTIONS
            else "READY"
        )
        return cls(overall_status=overall_status, actions=normalized)

    @staticmethod
    def _validate_dependencies(actions: tuple[ResearchArtifactAction, ...]) -> None:
        action_by_key = {item.artifact_key: item for item in actions}
        if len(action_by_key) != len(actions):
            raise ValueError("Research artifact plan的artifact_key不得重複")
        keys = set(action_by_key)
        for item in actions:
            unknown = sorted(set(item.dependencies) - keys)
            if unknown:
                raise ValueError(
                    f"Research artifact {item.artifact_key}依賴未知工件: "
                    + ", ".join(unknown)
                )

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(key: str) -> None:
            if key in visited:
                return
            if key in visiting:
                raise ValueError(f"Research artifact依賴形成循環: {key}")
            visiting.add(key)
            for dependency in action_by_key[key].dependencies:
                visit(dependency)
            visiting.remove(key)
            visited.add(key)

        for key in action_by_key:
            visit(key)

    @property
    def blocked(self) -> bool:
        return self.overall_status == "BLOCKED"

    @property
    def preparable(self) -> bool:
        return self.overall_status == "PREPARABLE"

    @property
    def action_by_key(self) -> dict[str, ResearchArtifactAction]:
        return {item.artifact_key: item for item in self.actions}

    def select(
        self, required_artifact_keys: Iterable[str]
    ) -> "ResearchArtifactPlan":
        required = {str(value) for value in required_artifact_keys}
        action_by_key = self.action_by_key
        unknown = sorted(required - set(action_by_key))
        if unknown:
            raise ValueError("Research artifact plan缺少要求工件: " + ", ".join(unknown))

        selected: set[str] = set()

        def include(key: str) -> None:
            if key in selected:
                return
            selected.add(key)
            for dependency in action_by_key[key].dependencies:
                include(dependency)

        for key in required:
            include(key)
        return ResearchArtifactPlan.from_actions(
            item for item in self.actions if item.artifact_key in selected
        )

    def next_runnable_action(
        self,
        *,
        executed_signatures: set[tuple[str, str, str | None, str]] | None = None,
    ) -> ResearchArtifactAction | None:
        action_by_key = self.action_by_key
        executed = executed_signatures or set()
        candidates: list[ResearchArtifactAction] = []
        for item in self.actions:
            if not item.runnable_work or item.execution_signature in executed:
                continue
            dependency_actions = [action_by_key[key].action for key in item.dependencies]
            if all(value in RESEARCH_ARTIFACT_READY_ACTIONS for value in dependency_actions):
                candidates.append(item)
        if not candidates:
            return None
        return sorted(
            candidates,
            key=lambda item: (int(item.execution_priority), item.artifact_key),
        )[0]

    def as_dict(self) -> dict[str, Any]:
        return {
            "overall_status": self.overall_status,
            "actions": [item.as_dict() for item in self.actions],
        }


# Backward-compatible names for Strategy Compare callers.  The generic Research
# contract is the implementation SSOT; strategy modules only retain their public names.
StrategyPreparationAction = ResearchArtifactAction
StrategyPreparationPlan = ResearchArtifactPlan


__all__ = [
    "RESEARCH_ARTIFACT_BUILD_ACTIONS",
    "RESEARCH_ARTIFACT_READY_ACTIONS",
    "RESEARCH_ARTIFACT_SUPPORTED_ACTIONS",
    "ResearchArtifactAction",
    "resolve_research_artifact_action",
    "ResearchArtifactPlan",
    "StrategyPreparationAction",
    "StrategyPreparationPlan",
]
