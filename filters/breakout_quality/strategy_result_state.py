"""Canonical Strategy Compare scientific-result state and execution-action mapping.

Artifact/domain adapters are responsible for proving that an existing result matches the
full scientific/evaluation identity.  Once that proof is available, every Strategy
Compare entry point must use this module to map the verified evidence to execution
state.  UI renderers and executors must not infer RUN/REUSE independently from paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

RESULT_ACTION_REUSE = "REUSE"
RESULT_ACTION_RUN = "RUN"
RESULT_ACTION_REBUILD_CONTEXT = "REBUILD_CONTEXT"


@dataclass(frozen=True)
class StrategyResultState:
    """Resolved state for one scientific strategy observation.

    ``evidence`` must already have passed the domain-specific identity/integrity
    validator.  ``context`` is explicitly non-scientific transient replay material
    (for example a retained DL-off baseline directory needed only to execute another
    arm).  Missing transient context may require rebuilding context without invalidating
    the durable scientific result.
    """

    action: str
    scientific_result_ready: bool
    context_required: bool = False
    context_ready: bool = False
    source_dir: Path | None = None
    reason: str = ""
    evidence: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "scientific_result_ready": bool(self.scientific_result_ready),
            "context_required": bool(self.context_required),
            "context_ready": bool(self.context_ready),
            "source_dir": self.source_dir,
            "reason": self.reason,
            "evidence": None if self.evidence is None else dict(self.evidence),
        }


def resolve_strategy_result_state(
    *,
    verified_evidence: Mapping[str, Any] | None,
    source_dir: str | Path | None = None,
    context_required: bool = False,
    context_ready: bool = False,
    missing_reason: str = "no identity-compatible completed scientific result",
) -> StrategyResultState:
    """Map verified scientific evidence to the one canonical execution action."""

    scientific_ready = isinstance(verified_evidence, Mapping)
    resolved_source = None if source_dir in (None, "") else Path(source_dir).resolve()
    if not scientific_ready:
        return StrategyResultState(
            action=RESULT_ACTION_RUN,
            scientific_result_ready=False,
            context_required=bool(context_required),
            context_ready=bool(context_ready),
            source_dir=resolved_source,
            reason=str(missing_reason),
            evidence=None,
        )
    if context_required and not context_ready:
        return StrategyResultState(
            action=RESULT_ACTION_REBUILD_CONTEXT,
            scientific_result_ready=True,
            context_required=True,
            context_ready=False,
            source_dir=resolved_source,
            reason="scientific result reusable; transient execution context missing",
            evidence=verified_evidence,
        )
    return StrategyResultState(
        action=RESULT_ACTION_REUSE,
        scientific_result_ready=True,
        context_required=bool(context_required),
        context_ready=bool(context_ready),
        source_dir=resolved_source,
        reason="identity-compatible completed scientific result",
        evidence=verified_evidence,
    )


def result_state_from_mapping(raw: Mapping[str, Any]) -> StrategyResultState:
    """Rehydrate a serialized result state at immutable plan/execution boundaries."""

    action = str(raw.get("action") or "")
    if action not in {
        RESULT_ACTION_REUSE,
        RESULT_ACTION_RUN,
        RESULT_ACTION_REBUILD_CONTEXT,
    }:
        raise ValueError(f"不支援的Strategy result action: {action!r}")
    evidence = raw.get("evidence")
    if evidence is not None and not isinstance(evidence, Mapping):
        raise TypeError("Strategy result state evidence必須是mapping或None")
    source_dir = raw.get("source_dir")
    return StrategyResultState(
        action=action,
        scientific_result_ready=bool(raw.get("scientific_result_ready")),
        context_required=bool(raw.get("context_required")),
        context_ready=bool(raw.get("context_ready")),
        source_dir=None if source_dir in (None, "") else Path(str(source_dir)).resolve(),
        reason=str(raw.get("reason") or ""),
        evidence=evidence,
    )


__all__ = [
    "RESULT_ACTION_REBUILD_CONTEXT",
    "RESULT_ACTION_REUSE",
    "RESULT_ACTION_RUN",
    "StrategyResultState",
    "resolve_strategy_result_state",
    "result_state_from_mapping",
]
