"""Research Market Data promotion and effective-active state contract.

Configuration keeps the bootstrap fallback generation declarative.  A successful
Research V2 promotion is instead represented by one immutable promotion manifest
plus one small atomic active pointer.  Consumers never infer promotion from file
presence and never mutate the freeze candidate itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from config.market_data import RESEARCH_DATA_GENERATION_V2, RESEARCH_REQUIRED_CUTOFF
from core.file_integrity import canonical_json_sha256, compute_file_sha256, load_json_strict
from core.market_data_contract import (
    RESEARCH_STATUS_ACTIVE_FROZEN,
    ResearchDataGenerationContract,
    build_market_data_contract_snapshot,
    get_active_research_data_generation,
    get_research_data_generation,
)
from core.market_data_adjusted_price_invariance import adjusted_price_representation_contract_fingerprint
from core.market_data_research_materialization import (
    RESEARCH_V2_COMPAT_MATERIALIZATION_STATUS_READY,
    materialization_identity_from_payload,
)
from core.market_data_research_scope import research_v2_dataset_scope_contract_fingerprint
from core.market_data_research_storage_contract import (
    resolve_active_research_generation_path,
    resolve_research_v2_compatibility_dataset_dir,
    resolve_research_v2_freeze_candidate_manifest_path,
    resolve_research_v2_materialization_manifest_path,
    resolve_research_v2_promotion_manifest_path,
)

RESEARCH_V2_PROMOTION_SCHEMA_VERSION = 1
RESEARCH_V2_PROMOTION_STATUS_ACTIVE = "ACTIVE_FROZEN"
RESEARCH_ACTIVE_POINTER_SCHEMA_VERSION = 1

RESEARCH_V2_PROMOTION_IDENTITY_FIELDS = (
    "schema_version",
    "generation_id",
    "status",
    "prior_generation_id",
    "freeze_candidate_fingerprint",
    "freeze_candidate_manifest_sha256",
    "provider_snapshot_fingerprint",
    "candidate_fingerprint",
    "required_cutoff",
    "frozen_cutoff",
    "research_scope_contract_fingerprint",
    "adjusted_price_representation_contract_fingerprint",
    "required_common_complete_fingerprint",
    "materialization_fingerprint",
    "materialization_manifest_sha256",
    "materialized_dataset_inventory_sha256",
    "promotion_authorized",
    "active_research_generation_changed",
)


@dataclass(frozen=True)
class ActiveResearchV2Promotion:
    promotion_fingerprint: str
    freeze_candidate_fingerprint: str
    materialization_fingerprint: str
    frozen_cutoff: str
    full_dataset_dir: Path
    payload: dict[str, object]


def _require_hex64(value: object, *, field: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{field} 必須是 64-char SHA256")
    return text


def promotion_identity_from_payload(payload: Mapping[str, object]) -> dict[str, object]:
    return {field: payload.get(field) for field in RESEARCH_V2_PROMOTION_IDENTITY_FIELDS}


def build_research_v2_promotion_identity_payload(
    *,
    freeze_candidate: Mapping[str, object],
    freeze_candidate_manifest_sha256: str,
    materialization: Mapping[str, object],
    materialization_manifest_sha256: str,
    prior_generation_id: str,
) -> dict[str, object]:
    if str(freeze_candidate.get("generation_id") or "") != RESEARCH_DATA_GENERATION_V2:
        raise ValueError("Research V2 promotion 只能消費 research_v2 freeze candidate")
    if str(freeze_candidate.get("status") or "") != "FREEZE_CANDIDATE_READY":
        raise ValueError("Research V2 promotion 需要 READY freeze candidate")
    if bool(freeze_candidate.get("promotion_authorized")):
        raise ValueError("freeze candidate 本身不得被改寫為 promotion-authorized")
    if str(materialization.get("status") or "") != RESEARCH_V2_COMPAT_MATERIALIZATION_STATUS_READY:
        raise ValueError("Research V2 promotion 需要 READY compatibility materialization")
    freeze_fp = _require_hex64(freeze_candidate.get("freeze_candidate_fingerprint"), field="freeze_candidate_fingerprint")
    if str(materialization.get("freeze_candidate_fingerprint") or "") != freeze_fp:
        raise ValueError("Research V2 promotion materialization / freeze candidate lineage drift")
    required_cutoff = str(freeze_candidate.get("required_cutoff") or "")
    frozen_cutoff = str(freeze_candidate.get("frozen_cutoff") or "")
    if required_cutoff != RESEARCH_REQUIRED_CUTOFF or frozen_cutoff != RESEARCH_REQUIRED_CUTOFF:
        raise ValueError("Research V2 promotion cutoff 不等於 fixed Research required cutoff")
    scope_fp = _require_hex64(
        freeze_candidate.get("research_scope_contract_fingerprint"), field="research_scope_contract_fingerprint"
    )
    if scope_fp != research_v2_dataset_scope_contract_fingerprint():
        raise ValueError("Research V2 promotion scope contract drift")
    adjusted_fp = _require_hex64(
        freeze_candidate.get("adjusted_price_representation_contract_fingerprint"),
        field="adjusted_price_representation_contract_fingerprint",
    )
    if adjusted_fp != adjusted_price_representation_contract_fingerprint():
        raise ValueError("Research V2 promotion adjusted-price representation contract drift")
    return {
        "schema_version": RESEARCH_V2_PROMOTION_SCHEMA_VERSION,
        "generation_id": RESEARCH_DATA_GENERATION_V2,
        "status": RESEARCH_V2_PROMOTION_STATUS_ACTIVE,
        "prior_generation_id": str(prior_generation_id),
        "freeze_candidate_fingerprint": freeze_fp,
        "freeze_candidate_manifest_sha256": _require_hex64(
            freeze_candidate_manifest_sha256, field="freeze_candidate_manifest_sha256"
        ),
        "provider_snapshot_fingerprint": _require_hex64(
            freeze_candidate.get("provider_snapshot_fingerprint"), field="provider_snapshot_fingerprint"
        ),
        "candidate_fingerprint": _require_hex64(
            freeze_candidate.get("candidate_fingerprint"), field="candidate_fingerprint"
        ),
        "required_cutoff": required_cutoff,
        "frozen_cutoff": frozen_cutoff,
        "research_scope_contract_fingerprint": scope_fp,
        "adjusted_price_representation_contract_fingerprint": adjusted_fp,
        "required_common_complete_fingerprint": _require_hex64(
            freeze_candidate.get("required_common_complete_fingerprint"), field="required_common_complete_fingerprint"
        ),
        "materialization_fingerprint": _require_hex64(
            materialization.get("materialization_fingerprint"), field="materialization_fingerprint"
        ),
        "materialization_manifest_sha256": _require_hex64(
            materialization_manifest_sha256, field="materialization_manifest_sha256"
        ),
        "materialized_dataset_inventory_sha256": _require_hex64(
            materialization.get("dataset_inventory_sha256"), field="materialized_dataset_inventory_sha256"
        ),
        "promotion_authorized": True,
        "active_research_generation_changed": True,
    }


def _validate_materialization_manifest_shallow(
    project_root: Path,
    *,
    materialization_fingerprint: str,
    expected_manifest_sha256: str,
    expected_inventory_sha256: str,
) -> Path:
    manifest_path = resolve_research_v2_materialization_manifest_path(project_root, materialization_fingerprint)
    if not manifest_path.is_file():
        raise FileNotFoundError("active Research V2 compatibility materialization manifest 不存在")
    if compute_file_sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("active Research V2 materialization manifest SHA256 drift")
    payload = load_json_strict(manifest_path)
    if not isinstance(payload, dict):
        raise ValueError("Research V2 materialization manifest 必須是 object")
    identity = materialization_identity_from_payload(payload)
    if canonical_json_sha256(identity) != materialization_fingerprint:
        raise ValueError("Research V2 materialization fingerprint drift")
    if str(payload.get("status") or "") != RESEARCH_V2_COMPAT_MATERIALIZATION_STATUS_READY:
        raise ValueError("Research V2 materialization 尚未 READY")
    if str(payload.get("dataset_inventory_sha256") or "") != expected_inventory_sha256:
        raise ValueError("Research V2 materialization inventory identity drift")
    dataset_dir = resolve_research_v2_compatibility_dataset_dir(project_root, materialization_fingerprint)
    if not dataset_dir.is_dir():
        raise FileNotFoundError("active Research V2 compatibility dataset directory 不存在")
    return dataset_dir


def load_active_research_v2_promotion(
    project_root,
    *,
    required: bool = False,
) -> ActiveResearchV2Promotion | None:
    root = Path(project_root).resolve()
    pointer_path = resolve_active_research_generation_path(root)
    if not pointer_path.is_file():
        if required:
            raise FileNotFoundError("active Research generation pointer 尚未建立")
        return None
    pointer = load_json_strict(pointer_path)
    if not isinstance(pointer, dict):
        raise ValueError("active Research generation pointer 必須是 object")
    if int(pointer.get("schema_version") or 0) != RESEARCH_ACTIVE_POINTER_SCHEMA_VERSION:
        raise ValueError("active Research generation pointer schema 不相容")
    if str(pointer.get("generation_id") or "") != RESEARCH_DATA_GENERATION_V2:
        raise ValueError("active Research generation pointer generation_id 不合法")
    promotion_fp = _require_hex64(pointer.get("promotion_fingerprint"), field="promotion_fingerprint")
    manifest_sha = _require_hex64(pointer.get("promotion_manifest_sha256"), field="promotion_manifest_sha256")
    manifest_path = resolve_research_v2_promotion_manifest_path(root, promotion_fp)
    if not manifest_path.is_file():
        raise FileNotFoundError("active Research V2 promotion manifest 不存在")
    if compute_file_sha256(manifest_path) != manifest_sha:
        raise ValueError("active Research V2 promotion manifest SHA256 drift")
    payload = load_json_strict(manifest_path)
    if not isinstance(payload, dict):
        raise ValueError("Research V2 promotion manifest 必須是 object")
    identity = promotion_identity_from_payload(payload)
    if canonical_json_sha256(identity) != promotion_fp:
        raise ValueError("Research V2 promotion fingerprint drift")
    if str(payload.get("generation_id") or "") != RESEARCH_DATA_GENERATION_V2:
        raise ValueError("Research V2 promotion generation drift")
    if str(payload.get("status") or "") != RESEARCH_V2_PROMOTION_STATUS_ACTIVE:
        raise ValueError("Research V2 promotion status 不合法")
    if not bool(payload.get("promotion_authorized")) or not bool(payload.get("active_research_generation_changed")):
        raise ValueError("Research V2 promotion authorization flags 不合法")
    if str(payload.get("required_cutoff") or "") != RESEARCH_REQUIRED_CUTOFF:
        raise ValueError("Research V2 promotion required cutoff drift")
    if str(payload.get("frozen_cutoff") or "") != RESEARCH_REQUIRED_CUTOFF:
        raise ValueError("Research V2 promotion frozen cutoff drift")
    if str(payload.get("research_scope_contract_fingerprint") or "") != research_v2_dataset_scope_contract_fingerprint():
        raise ValueError("active Research V2 scope contract drift")
    if str(payload.get("adjusted_price_representation_contract_fingerprint") or "") != adjusted_price_representation_contract_fingerprint():
        raise ValueError("active Research V2 adjusted-price contract drift")
    freeze_fp = _require_hex64(payload.get("freeze_candidate_fingerprint"), field="freeze_candidate_fingerprint")
    freeze_manifest_path = resolve_research_v2_freeze_candidate_manifest_path(root, freeze_fp)
    if not freeze_manifest_path.is_file():
        raise FileNotFoundError("active Research V2 freeze candidate manifest 不存在")
    expected_freeze_sha = _require_hex64(
        payload.get("freeze_candidate_manifest_sha256"), field="freeze_candidate_manifest_sha256"
    )
    if compute_file_sha256(freeze_manifest_path) != expected_freeze_sha:
        raise ValueError("active Research V2 freeze candidate manifest SHA256 drift")
    materialization_fp = _require_hex64(payload.get("materialization_fingerprint"), field="materialization_fingerprint")
    dataset_dir = _validate_materialization_manifest_shallow(
        root,
        materialization_fingerprint=materialization_fp,
        expected_manifest_sha256=_require_hex64(
            payload.get("materialization_manifest_sha256"), field="materialization_manifest_sha256"
        ),
        expected_inventory_sha256=_require_hex64(
            payload.get("materialized_dataset_inventory_sha256"), field="materialized_dataset_inventory_sha256"
        ),
    )
    return ActiveResearchV2Promotion(
        promotion_fingerprint=promotion_fp,
        freeze_candidate_fingerprint=freeze_fp,
        materialization_fingerprint=materialization_fp,
        frozen_cutoff=str(payload["frozen_cutoff"]),
        full_dataset_dir=dataset_dir,
        payload=dict(payload),
    )


def get_effective_research_data_generation(project_root) -> ResearchDataGenerationContract:
    """Resolve validated promotion state first, then the declarative fallback."""

    root = Path(project_root).resolve()
    promoted = load_active_research_v2_promotion(root, required=False)
    if promoted is not None:
        blueprint = get_research_data_generation(RESEARCH_DATA_GENERATION_V2)
        return ResearchDataGenerationContract(
            generation_id=RESEARCH_DATA_GENERATION_V2,
            status=RESEARCH_STATUS_ACTIVE_FROZEN,
            lifecycle="immutable",
            cutoff_mode=blueprint.cutoff_mode,
            cutoff=promoted.frozen_cutoff,
            required_cutoff=blueprint.required_cutoff,
            universe_mode=blueprint.universe_mode,
        )

    return get_active_research_data_generation()


def build_effective_market_data_contract_snapshot(project_root) -> dict[str, object]:
    """Overlay validated effective Research state on the canonical base snapshot."""

    snapshot = build_market_data_contract_snapshot()
    active = get_effective_research_data_generation(project_root)
    snapshot["active_research_generation"] = active.__dict__
    generations = dict(snapshot["research_generations"])
    if active.generation_id in generations:
        generations[active.generation_id] = active.__dict__
    snapshot["research_generations"] = generations
    return snapshot


def resolve_promoted_research_full_dataset_dir(project_root) -> Path | None:
    active = load_active_research_v2_promotion(project_root, required=False)
    return None if active is None else active.full_dataset_dir


__all__ = [
    "RESEARCH_V2_PROMOTION_SCHEMA_VERSION",
    "RESEARCH_V2_PROMOTION_STATUS_ACTIVE",
    "RESEARCH_ACTIVE_POINTER_SCHEMA_VERSION",
    "RESEARCH_V2_PROMOTION_IDENTITY_FIELDS",
    "ActiveResearchV2Promotion",
    "promotion_identity_from_payload",
    "build_research_v2_promotion_identity_payload",
    "load_active_research_v2_promotion",
    "get_effective_research_data_generation",
    "build_effective_market_data_contract_snapshot",
    "resolve_promoted_research_full_dataset_dir",
]
