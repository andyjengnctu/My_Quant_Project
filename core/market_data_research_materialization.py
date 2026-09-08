"""Frozen Research V2 compatibility materialization contract.

The active Research generation must remain consumable by existing OHLCV CSV
consumers without introducing a second adjusted-price engine.  This contract
therefore maps FinMind canonical adjusted OHLC plus the separately-authorized
raw Trading_Volume field into one immutable six-column compatibility view.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Mapping

from core.file_integrity import canonical_json_sha256
from core.market_data_adjusted_price_invariance import PROVIDER_VOLUME_FIELD
from core.market_data_research_scope import (
    RESEARCH_V2_ADJUSTED_PRICE_DATASET,
    RESEARCH_V2_RAW_VOLUME_DATASET,
)

RESEARCH_V2_COMPAT_MATERIALIZATION_SCHEMA_VERSION = 2
RESEARCH_V2_COMPAT_MATERIALIZATION_CONTRACT_ID = "research_v2_legacy_ohlcv_compatibility_v1"
RESEARCH_V2_COMPAT_MATERIALIZATION_STATUS_READY = "MATERIALIZATION_READY"
RESEARCH_V2_COMPAT_OUTPUT_COLUMNS = ("Date", "Open", "High", "Low", "Close", "Volume")
RESEARCH_V2_COMPAT_PRICE_FIELD_MAPPING = {
    "Open": "open",
    "High": "max",
    "Low": "min",
    "Close": "close",
}
RESEARCH_V2_COMPAT_VOLUME_FIELD_MAPPING = {"Volume": PROVIDER_VOLUME_FIELD}

RESEARCH_V2_COMPAT_MATERIALIZATION_IDENTITY_FIELDS = (
    "schema_version",
    "contract_id",
    "generation_id",
    "freeze_candidate_fingerprint",
    "candidate_fingerprint",
    "required_source_projection_fingerprint",
    "daily_universe_file_sha256",
    "required_cutoff",
    "frozen_cutoff",
    "research_scope_contract_fingerprint",
    "adjusted_price_representation_contract_fingerprint",
    "adjusted_price_revision_proof_fingerprint",
    "required_common_complete_fingerprint",
    "price_dataset",
    "volume_dataset",
    "price_field_mapping",
    "volume_field_mapping",
    "output_columns",
)


def _require_hex64(value: object, *, field: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{field} 必須是 64-char SHA256")
    return text


def build_research_v2_compatibility_materialization_identity_payload(
    freeze_candidate: Mapping[str, object],
) -> dict[str, object]:
    if str(freeze_candidate.get("generation_id") or "") != "research_v2":
        raise ValueError("Research V2 compatibility materialization 只能消費 research_v2 freeze candidate")
    if str(freeze_candidate.get("status") or "") != "FREEZE_CANDIDATE_READY":
        raise ValueError("Research V2 compatibility materialization 需要 READY freeze candidate")
    if bool(freeze_candidate.get("promotion_authorized")):
        raise ValueError("freeze candidate 不得先行改寫 promotion_authorized")
    required_cutoff = str(freeze_candidate.get("required_cutoff") or "").strip()
    frozen_cutoff = str(freeze_candidate.get("frozen_cutoff") or "").strip()
    if not required_cutoff or frozen_cutoff != required_cutoff:
        raise ValueError("Research V2 compatibility materialization frozen cutoff drift")
    try:
        date.fromisoformat(required_cutoff)
    except ValueError as exc:
        raise ValueError("Research V2 compatibility materialization cutoff 必須是 YYYY-MM-DD") from exc

    return {
        "schema_version": RESEARCH_V2_COMPAT_MATERIALIZATION_SCHEMA_VERSION,
        "contract_id": RESEARCH_V2_COMPAT_MATERIALIZATION_CONTRACT_ID,
        "generation_id": "research_v2",
        "freeze_candidate_fingerprint": _require_hex64(
            freeze_candidate.get("freeze_candidate_fingerprint"), field="freeze_candidate_fingerprint"
        ),
        "candidate_fingerprint": _require_hex64(
            freeze_candidate.get("candidate_fingerprint"), field="candidate_fingerprint"
        ),
        "required_source_projection_fingerprint": _require_hex64(
            freeze_candidate.get("required_source_projection_fingerprint"), field="required_source_projection_fingerprint"
        ),
        "daily_universe_file_sha256": _require_hex64(
            freeze_candidate.get("daily_universe_file_sha256"), field="daily_universe_file_sha256"
        ),
        "required_cutoff": required_cutoff,
        "frozen_cutoff": frozen_cutoff,
        "research_scope_contract_fingerprint": _require_hex64(
            freeze_candidate.get("research_scope_contract_fingerprint"), field="research_scope_contract_fingerprint"
        ),
        "adjusted_price_representation_contract_fingerprint": _require_hex64(
            freeze_candidate.get("adjusted_price_representation_contract_fingerprint"),
            field="adjusted_price_representation_contract_fingerprint",
        ),
        "adjusted_price_revision_proof_fingerprint": _require_hex64(
            freeze_candidate.get("adjusted_price_revision_proof_fingerprint"),
            field="adjusted_price_revision_proof_fingerprint",
        ),
        "required_common_complete_fingerprint": _require_hex64(
            freeze_candidate.get("required_common_complete_fingerprint"), field="required_common_complete_fingerprint"
        ),
        "price_dataset": RESEARCH_V2_ADJUSTED_PRICE_DATASET,
        "volume_dataset": RESEARCH_V2_RAW_VOLUME_DATASET,
        "price_field_mapping": dict(RESEARCH_V2_COMPAT_PRICE_FIELD_MAPPING),
        "volume_field_mapping": dict(RESEARCH_V2_COMPAT_VOLUME_FIELD_MAPPING),
        "output_columns": list(RESEARCH_V2_COMPAT_OUTPUT_COLUMNS),
    }


def research_v2_compatibility_materialization_fingerprint(
    freeze_candidate: Mapping[str, object],
) -> str:
    return canonical_json_sha256(build_research_v2_compatibility_materialization_identity_payload(freeze_candidate))


def deterministic_compatibility_file_mtime_ns(required_cutoff: str) -> int:
    """Return a stable mtime so deterministic rematerialization keeps source inventory stable."""

    parsed = date.fromisoformat(str(required_cutoff))
    stamp = datetime(parsed.year, parsed.month, parsed.day, tzinfo=timezone.utc).timestamp()
    return int(stamp * 1_000_000_000)


def materialization_identity_from_payload(payload: Mapping[str, object]) -> dict[str, object]:
    return {field: payload.get(field) for field in RESEARCH_V2_COMPAT_MATERIALIZATION_IDENTITY_FIELDS}


__all__ = [
    "RESEARCH_V2_COMPAT_MATERIALIZATION_SCHEMA_VERSION",
    "RESEARCH_V2_COMPAT_MATERIALIZATION_CONTRACT_ID",
    "RESEARCH_V2_COMPAT_MATERIALIZATION_STATUS_READY",
    "RESEARCH_V2_COMPAT_OUTPUT_COLUMNS",
    "RESEARCH_V2_COMPAT_PRICE_FIELD_MAPPING",
    "RESEARCH_V2_COMPAT_VOLUME_FIELD_MAPPING",
    "RESEARCH_V2_COMPAT_MATERIALIZATION_IDENTITY_FIELDS",
    "build_research_v2_compatibility_materialization_identity_payload",
    "research_v2_compatibility_materialization_fingerprint",
    "deterministic_compatibility_file_mtime_ns",
    "materialization_identity_from_payload",
]
