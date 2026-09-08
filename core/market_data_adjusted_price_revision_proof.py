"""PIT proof contract for post-cutoff FinMind adjusted-price historical rebuilds.

Round 13 proved invariance only for one documented corporate-action operation:
a positive common scalar applied to an already-mature historical prefix.  FinMind
announced a provider algorithm correction on 2026-09-01 and rebuilt the full
``TaiwanStockPriceAdj`` history.  Snapshot immutability alone cannot prove that
such a rebuild belongs to the Round-13 invariance class.

This module therefore defines a separate evidence contract.  A current-vintage
PriceAdj source may enter Research V2 only when every required frozen Research
stock-day can be compared with the legacy frozen Research V1 FinMind adjusted
price and the two OHLC histories are exactly related by one positive scalar per
instrument.  No adjusted-price calculator is implemented here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from core.file_integrity import canonical_json_sha256
from core.market_data_adjusted_price_invariance import (
    adjusted_price_representation_contract_fingerprint,
)
from core.market_data_contract import FINMIND_ADJUSTED_PRICE_DATASET

ADJUSTED_PRICE_REVISION_PROOF_SCHEMA_VERSION = 1
ADJUSTED_PRICE_REVISION_PROOF_CONTRACT_ID = "finmind_price_adj_post_cutoff_revision_equivalence_v1"
ADJUSTED_PRICE_PROVIDER_CORRECTION_DATE = "2026-09-01"
ADJUSTED_PRICE_PROVIDER_CORRECTION_ID = "finmind_20260901_taiwan_stock_price_adj_full_history_rebuild"
ADJUSTED_PRICE_PROVIDER_CORRECTION_EVIDENCE = (
    "FinMind WhatIsNew/2026-09-01: TaiwanStockPriceAdj adjustment logic corrected for "
    "holiday-postponed ex-right/dividend events and capital-reduction/ex-right-dividend coexistence; full history rebuilt"
)

ADJUSTED_PRICE_REVISION_STATUS_READY = "REVISION_EQUIVALENCE_PROVEN"
ADJUSTED_PRICE_REVISION_STATUS_BLOCKED = "REVISION_EQUIVALENCE_NOT_PROVEN"

ADJUSTED_PRICE_REVISION_PROOF_IDENTITY_FIELDS = (
    "schema_version",
    "contract_id",
    "dataset",
    "provider_correction_id",
    "provider_correction_date",
    "provider_correction_evidence",
    "required_cutoff",
    "status",
    "required_stock_count",
    "proven_stock_count",
    "required_stock_day_count",
    "proven_stock_day_count",
    "missing_legacy_stock_count",
    "missing_legacy_stock_day_count",
    "missing_current_stock_day_count",
    "non_scalar_mismatch_count",
    "legacy_required_source_fingerprint",
    "current_required_source_fingerprint",
    "daily_universe_file_sha256",
    "representation_contract_fingerprint",
)


@dataclass(frozen=True)
class AdjustedPriceRevisionProofSummary:
    schema_version: int
    contract_id: str
    dataset: str
    provider_correction_id: str
    provider_correction_date: str
    required_cutoff: str
    status: str
    required_stock_count: int
    proven_stock_count: int
    required_stock_day_count: int
    proven_stock_day_count: int
    missing_legacy_stock_count: int
    missing_legacy_stock_day_count: int
    missing_current_stock_day_count: int
    non_scalar_mismatch_count: int
    legacy_required_source_fingerprint: str
    current_required_source_fingerprint: str
    daily_universe_file_sha256: str
    representation_contract_fingerprint: str


def build_adjusted_price_revision_proof_identity_payload(
    *,
    required_cutoff: str,
    status: str,
    required_stock_count: int,
    proven_stock_count: int,
    required_stock_day_count: int,
    proven_stock_day_count: int,
    missing_legacy_stock_count: int,
    missing_legacy_stock_day_count: int,
    missing_current_stock_day_count: int,
    non_scalar_mismatch_count: int,
    legacy_required_source_fingerprint: str,
    current_required_source_fingerprint: str,
    daily_universe_file_sha256: str,
) -> dict[str, object]:
    representation_fp = adjusted_price_representation_contract_fingerprint()
    return {
        "schema_version": ADJUSTED_PRICE_REVISION_PROOF_SCHEMA_VERSION,
        "contract_id": ADJUSTED_PRICE_REVISION_PROOF_CONTRACT_ID,
        "dataset": FINMIND_ADJUSTED_PRICE_DATASET,
        "provider_correction_id": ADJUSTED_PRICE_PROVIDER_CORRECTION_ID,
        "provider_correction_date": ADJUSTED_PRICE_PROVIDER_CORRECTION_DATE,
        "provider_correction_evidence": ADJUSTED_PRICE_PROVIDER_CORRECTION_EVIDENCE,
        "required_cutoff": str(required_cutoff),
        "status": str(status),
        "required_stock_count": int(required_stock_count),
        "proven_stock_count": int(proven_stock_count),
        "required_stock_day_count": int(required_stock_day_count),
        "proven_stock_day_count": int(proven_stock_day_count),
        "missing_legacy_stock_count": int(missing_legacy_stock_count),
        "missing_legacy_stock_day_count": int(missing_legacy_stock_day_count),
        "missing_current_stock_day_count": int(missing_current_stock_day_count),
        "non_scalar_mismatch_count": int(non_scalar_mismatch_count),
        "legacy_required_source_fingerprint": str(legacy_required_source_fingerprint),
        "current_required_source_fingerprint": str(current_required_source_fingerprint),
        "daily_universe_file_sha256": str(daily_universe_file_sha256),
        "representation_contract_fingerprint": representation_fp,
    }


def adjusted_price_revision_proof_fingerprint(payload: dict[str, object]) -> str:
    return canonical_json_sha256(payload)


def validate_adjusted_price_revision_proof_payload(payload: dict[str, object]) -> dict[str, object]:
    if int(payload.get("schema_version") or 0) != ADJUSTED_PRICE_REVISION_PROOF_SCHEMA_VERSION:
        raise ValueError("adjusted-price revision proof schema drift")
    if str(payload.get("contract_id") or "") != ADJUSTED_PRICE_REVISION_PROOF_CONTRACT_ID:
        raise ValueError("adjusted-price revision proof contract drift")
    if str(payload.get("dataset") or "") != FINMIND_ADJUSTED_PRICE_DATASET:
        raise ValueError("adjusted-price revision proof dataset drift")
    if str(payload.get("provider_correction_id") or "") != ADJUSTED_PRICE_PROVIDER_CORRECTION_ID:
        raise ValueError("adjusted-price revision provider correction identity drift")
    if str(payload.get("provider_correction_date") or "") != ADJUSTED_PRICE_PROVIDER_CORRECTION_DATE:
        raise ValueError("adjusted-price revision provider correction date drift")
    if str(payload.get("representation_contract_fingerprint") or "") != adjusted_price_representation_contract_fingerprint():
        raise ValueError("adjusted-price revision representation contract drift")
    for key in (
        "legacy_required_source_fingerprint",
        "current_required_source_fingerprint",
        "daily_universe_file_sha256",
    ):
        digest = str(payload.get(key) or "").strip().lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError(f"adjusted-price revision proof {key} invalid")
    status = str(payload.get("status") or "")
    if status not in {ADJUSTED_PRICE_REVISION_STATUS_READY, ADJUSTED_PRICE_REVISION_STATUS_BLOCKED}:
        raise ValueError("adjusted-price revision proof status invalid")
    required_stocks = int(payload.get("required_stock_count") or 0)
    proven_stocks = int(payload.get("proven_stock_count") or 0)
    required_days = int(payload.get("required_stock_day_count") or 0)
    proven_days = int(payload.get("proven_stock_day_count") or 0)
    missing_stocks = int(payload.get("missing_legacy_stock_count") or 0)
    missing_days = int(payload.get("missing_legacy_stock_day_count") or 0)
    missing_current_days = int(payload.get("missing_current_stock_day_count") or 0)
    mismatches = int(payload.get("non_scalar_mismatch_count") or 0)
    if min(required_stocks, proven_stocks, required_days, proven_days, missing_stocks, missing_days, missing_current_days, mismatches) < 0:
        raise ValueError("adjusted-price revision proof counts cannot be negative")
    ready = (
        required_stocks > 0
        and required_days > 0
        and proven_stocks == required_stocks
        and proven_days == required_days
        and missing_stocks == 0
        and missing_days == 0
        and missing_current_days == 0
        and mismatches == 0
    )
    if (status == ADJUSTED_PRICE_REVISION_STATUS_READY) != ready:
        raise ValueError("adjusted-price revision proof READY/count contract drift")
    return dict(payload)


def adjusted_price_revision_proof_summary(payload: dict[str, object]) -> AdjustedPriceRevisionProofSummary:
    validated = validate_adjusted_price_revision_proof_payload(payload)
    keys = AdjustedPriceRevisionProofSummary.__dataclass_fields__.keys()
    return AdjustedPriceRevisionProofSummary(**{key: validated[key] for key in keys})


__all__ = [
    "ADJUSTED_PRICE_REVISION_PROOF_SCHEMA_VERSION",
    "ADJUSTED_PRICE_REVISION_PROOF_CONTRACT_ID",
    "ADJUSTED_PRICE_PROVIDER_CORRECTION_DATE",
    "ADJUSTED_PRICE_PROVIDER_CORRECTION_ID",
    "ADJUSTED_PRICE_PROVIDER_CORRECTION_EVIDENCE",
    "ADJUSTED_PRICE_REVISION_STATUS_READY",
    "ADJUSTED_PRICE_REVISION_STATUS_BLOCKED",
    "ADJUSTED_PRICE_REVISION_PROOF_IDENTITY_FIELDS",
    "AdjustedPriceRevisionProofSummary",
    "build_adjusted_price_revision_proof_identity_payload",
    "adjusted_price_revision_proof_fingerprint",
    "validate_adjusted_price_revision_proof_payload",
    "adjusted_price_revision_proof_summary",
]
