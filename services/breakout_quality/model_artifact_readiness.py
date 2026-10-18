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
from typing import Any, Mapping, Sequence

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
    verified_source_ready: bool
    evidence_payload: dict[str, Any] | None
    source_reason: str | None
    plan: ResearchArtifactPlan
    display_action: str
    display_reason: str

    @property
    def source_ready(self) -> bool:
        return self.verified_source_ready

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
    verified_source_ready: bool = False,
) -> ModelArtifactReadiness:
    """Derive one stage dependency plan from canonical producer facts.

    FIT candidates are *not* considered verified.  They only select VERIFY FIT
    (owner validation) instead of an unconditional BUILD.  A Rolling source
    contract with incomplete SOP evidence means audit-only REFRESH, never FIT.
    """
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"forward", "rolling"}:
        raise ValueError(f"Unsupported Model readiness mode: {mode!r}")
    # Robustness may retain only a verified full source's SOP payload in its
    # historical consumer API. The caller may explicitly attest that the full
    # canonical owner validated it; this does not provide a checkpoint handle.
    source_ready = source_contract is not None or bool(verified_source_ready)
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
        verified_source_ready=source_ready,
        evidence_payload=payload if evidence_ready else None,
        source_reason=source_reason,
        plan=plan,
        display_action=action,
        display_reason=reason,
    )


@dataclass(frozen=True)
class ModelRobustnessSeedSource:
    """Canonical producer observations for one isolated benchmark seed.

    ``verified_evidence`` must originate from a successful *complete* source
    validator; it must not be an unverified JSON read or a file existence test.
    SCORE-only contracts are supplied separately so a missing Audit never
    implies invalid FIT/SCORE. The source owner's original reason is preserved.
    """

    seed: int
    verified_evidence: Mapping[str, Any] | None
    evidence_path: Path | None
    reason: str | None
    score_contract: Any | None = None


@dataclass(frozen=True)
class ModelRobustnessSeedReadiness:
    seed: int
    readiness: ModelArtifactReadiness


@dataclass(frozen=True)
class ModelRobustnessReadiness:
    """Canonical multi-seed projection; REPORT is a single aggregate action."""

    mode: str
    seeds: tuple[ModelRobustnessSeedReadiness, ...]
    plan: ResearchArtifactPlan
    display_action: str
    display_reason: str

    @property
    def evidence_ready(self) -> bool:
        return all(item.readiness.evidence_ready for item in self.seeds)


def resolve_model_robustness_readiness(
    *, mode: str, sources: Sequence[ModelRobustnessSeedSource],
) -> ModelRobustnessReadiness:
    """Project Forward/Rolling benchmark seeds onto the *same* stage policy.

    The report in Model Robustness is the cross-seed comparison, not a per-seed
    standalone Model publication. It is DERIVEd only after every seed's
    EVIDENCE is ready. No seed can authorize another seed's SCORE or FIT reuse.
    This pure projection does not build artifacts, select seeds or change SOP.
    """
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"forward", "rolling"}:
        raise ValueError(f"Unsupported Model robustness mode: {mode!r}")
    if not sources:
        raise ValueError("Model robustness requires benchmark seeds")
    seeds = tuple(int(item.seed) for item in sources)
    if len(set(seeds)) != len(seeds):
        raise ValueError("Model robustness benchmark seeds must be unique")

    stages: list[ModelRobustnessSeedReadiness] = []
    actions: list[ResearchArtifactAction] = []
    for item in sources:
        # A complete evidence payload is passed here only after the canonical
        # complete source loader has validated FIT/SCORE and SOP publication.
        # Never synthesize a placeholder model/score contract (e.g. object()).
        stage = resolve_model_artifact_readiness(
            mode=normalized_mode,
            source_contract=item.score_contract,
            verified_source_ready=item.verified_evidence is not None,
            evidence_payload=item.verified_evidence,
            source_reason=item.reason,
            source_path=None,  # No per-seed standalone report publication.
        )
        stages.append(ModelRobustnessSeedReadiness(seed=int(item.seed), readiness=stage))
        for key in ("FIT", "SCORE", "EVIDENCE"):
            step = stage.plan.action_by_key[key]
            actions.append(ResearchArtifactAction(
                action_id=f"model:seed:{int(item.seed)}:{key.lower()}",
                artifact_key=f"seed:{int(item.seed)}:{key}",
                action=step.action, builder_type=step.builder_type,
                description=step.description, path="",
                dependencies=tuple(
                    f"seed:{int(item.seed)}:{dependency}" for dependency in step.dependencies
                ),
            ))
    actions.append(ResearchArtifactAction(
        action_id="model:robustness:report", artifact_key="REPORT",
        action="DERIVE", builder_type="canonical_comparison_renderer",
        description="approved cross-seed Standard Model Comparison publication",
        path="", dependencies=tuple(f"seed:{seed}:EVIDENCE" for seed in seeds),
    ))
    plan = ResearchArtifactPlan.from_actions(actions)
    evidence_count = sum(stage.readiness.evidence_ready for stage in stages)
    score_count = sum(stage.readiness.source_ready for stage in stages)
    total = len(stages)
    if evidence_count == total:
        action, reason = "REUSE", "READY"
    elif normalized_mode == "rolling" and score_count == total:
        action = "REFRESH EVIDENCE"
        reason = f"{evidence_count}/{total} seeds EVIDENCE READY; all SCORE sources valid"
    elif normalized_mode == "rolling":
        action = "BUILD/REFRESH"
        reason = f"{evidence_count}/{total} seeds EVIDENCE READY; {score_count}/{total} SCORE READY"
    else:
        action = "BUILD/REFRESH"
        reason = f"{evidence_count}/{total} seeds READY"
    return ModelRobustnessReadiness(
        mode=normalized_mode, seeds=tuple(stages), plan=plan,
        display_action=action, display_reason=reason,
    )
