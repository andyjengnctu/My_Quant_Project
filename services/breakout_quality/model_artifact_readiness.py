"""Single lifecycle projection for continuous Model FIT/SCORE/EVIDENCE/REPORT.

Canonical FIT and score producers still own scientific validation.  This module
consumes their *validated* contracts, together with the single Standard SOP
evidence validator, and owns the dependency/action projection shared by Forward,
Rolling and Model Comparison.  A candidate path never authorizes checkpoint REUSE.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from core.research_orchestration import ResearchArtifactAction, ResearchArtifactPlan
from core.research_report_contract import (
    persistent_report_contract_fingerprint,
    require_approved_report_contract,
)
from services.breakout_quality.standard_model_sop import standard_model_sop_completeness_issues


@dataclass(frozen=True)
class ModelArtifactReadiness:
    """One immutable decision snapshot shared by menu, comparison and execution."""

    mode: str
    source_contract: Any | None
    evidence_payload: dict[str, Any] | None
    source_reason: str | None
    plan: ResearchArtifactPlan
    display_action: str
    display_reason: str

    @property
    def source_ready(self) -> bool:
        return self.source_contract is not None

    @property
    def evidence_ready(self) -> bool:
        return self.evidence_payload is not None

    @property
    def audit_only_refresh(self) -> bool:
        return self.mode == "rolling" and self.source_ready and not self.evidence_ready

    @property
    def report_only_refresh(self) -> bool:
        return self.evidence_ready and self.plan.action_by_key["REPORT"].action == "DERIVE"


def standard_model_report_matches_source(
    source_payload: Mapping[str, Any], source_path: Path | None,
    *, expected_identity: Mapping[str, str] | None = None,
) -> bool:
    """Check current publication without accepting a stale/missing report as evidence.

    A missing renderer artifact only schedules REPORT materialization; it never
    invalidates FIT, SCORE or canonical evidence and therefore cannot retrain.
    """
    if source_path is None:
        return False
    source = dict(source_payload or {})
    standard = dict(source.get("standard_model_sop") or {})
    if not standard and str(source.get("schema") or "").startswith("standard_model_sop_v"):
        standard = source
    if not standard or standard_model_sop_completeness_issues(source):
        return False
    base = Path(source_path).resolve().parent
    machine_file = base / "standard_model_report.json"
    markdown_file = base / "standard_model_report.md"
    if not machine_file.is_file() or not markdown_file.is_file():
        return False
    try:
        saved = json.loads(machine_file.read_text(encoding="utf-8"))
        contract = require_approved_report_contract("model.standard_sop")
        fingerprint = persistent_report_contract_fingerprint(contract.report_id)
        if not isinstance(saved, dict):
            return False
        if any((
            saved.get("report_id") != contract.report_id,
            saved.get("report_version") != int(contract.version),
            saved.get("report_contract_fingerprint") != fingerprint,
            saved.get("standard_model_sop") != standard,
        )):
            return False
        if expected_identity and any(
            str(saved.get(key) or "") != str(expected)
            for key, expected in expected_identity.items()
        ):
            return False
        document = markdown_file.read_text(encoding="utf-8")
        return (
            f"`{contract.report_id}` v`{contract.version}`" in document
            and fingerprint in document
        )
    except (OSError, ValueError, TypeError, UnicodeError):
        return False


def resolve_model_artifact_readiness(
    *,
    mode: str,
    source_contract: Any | None,
    evidence_payload: Mapping[str, Any] | None,
    source_reason: str | None,
    source_path: Path | None,
    forward_fit_candidate: bool = False,
    rolling_fit_candidate: bool = False,
    report_identity: Mapping[str, str] | None = None,
) -> ModelArtifactReadiness:
    """Derive one stage dependency plan from canonical producer facts.

    FIT candidates are *not* considered verified.  They only select VERIFY FIT
    (owner validation) instead of an unconditional BUILD.  A Rolling source
    contract with incomplete SOP evidence means audit-only REFRESH, never FIT.
    """
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"forward", "rolling"}:
        raise ValueError(f"Unsupported Model readiness mode: {mode!r}")
    source_ready = source_contract is not None
    payload = dict(evidence_payload or {})
    issues = standard_model_sop_completeness_issues(payload) if payload else ("Standard SOP evidence missing",)
    evidence_ready = source_ready and not issues
    if not evidence_ready:
        payload = {}

    report_ready = bool(
        evidence_ready and standard_model_report_matches_source(
            payload, source_path, expected_identity=report_identity,
        )
    )
    candidate = bool(forward_fit_candidate or rolling_fit_candidate)
    # Rolling scored-source readiness proves that existing scores can be used;
    # it does NOT establish that a historical FIT checkpoint is still present
    # or fitting-identity compatible with another execution mode.  No fitting
    # is necessary on this path, but checkpoint REUSE remains the FIT owner's
    # independent responsibility.
    if source_ready and normalized_mode == "rolling":
        fit_action = "NOT_REQUIRED"
        fit_reason = "validated Rolling score source; checkpoint REUSE not asserted"
    elif source_ready:
        fit_action = "REUSE"
        fit_reason = "canonical Forward fitted source validated"
    else:
        fit_action = "RESUME" if candidate else "BUILD"
        fit_reason = "canonical FIT owner must inspect identity before training"
    plan = ResearchArtifactPlan.from_actions((
        ResearchArtifactAction(
            "model:fit", "FIT", fit_action,
            "canonical_fit_owner", fit_reason, "",
        ),
        ResearchArtifactAction(
            "model:score", "SCORE", "REUSE" if source_ready else "BUILD",
            "canonical_score_producer", "mode-specific canonical score/lineage", "",
            dependencies=("FIT",),
        ),
        ResearchArtifactAction(
            "model:evidence", "EVIDENCE", "REUSE" if evidence_ready else "DERIVE",
            "canonical_evaluation_owner", "Standard Model SOP evidence", "",
            dependencies=("SCORE",),
        ),
        ResearchArtifactAction(
            "model:report", "REPORT", "REUSE" if report_ready else "DERIVE",
            "canonical_report_renderer", "approved SOP publication", "",
            dependencies=("EVIDENCE",),
        ),
    ))
    if evidence_ready:
        action = "REUSE"
        reason = "READY" if report_ready else "READY; report artifact refresh only"
    elif source_ready:
        action = "REFRESH"
        reason = str(source_reason or "Standard SOP evidence requires audit-only refresh")
    elif candidate:
        action = "VERIFY FIT" if rolling_fit_candidate else "REFRESH/VERIFY"
        reason = (
            "Rolling first-fold FIT candidate; canonical identity verification required"
            if rolling_fit_candidate else str(source_reason or "canonical FIT verification required")
        )
    else:
        action = "BUILD/REFRESH" if normalized_mode == "rolling" else "BUILD"
        reason = str(source_reason or "canonical source missing")
    return ModelArtifactReadiness(
        mode=normalized_mode,
        source_contract=source_contract,
        evidence_payload=payload if evidence_ready else None,
        source_reason=source_reason,
        plan=plan,
        display_action=action,
        display_reason=reason,
    )
