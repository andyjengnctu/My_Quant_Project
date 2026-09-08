"""Research V2 required-scope common-complete and immutable freeze-candidate contract.

Round 15 is deliberately narrower than promotion.  It computes completeness only
for the Round-14 authorized foundation scope, pins the resulting immutable
candidate identity, and leaves the configured active Research generation
unchanged.  Archive-wide optional datasets never enter this identity.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

from core.file_integrity import canonical_json_sha256
from core.market_data_research_scope import (
    RESEARCH_V2_COMMON_COMPLETE_DATASETS,
    RESEARCH_V2_REQUIRED_DATASETS,
)

RESEARCH_V2_REQUIRED_COMMON_COMPLETE_SCHEMA_VERSION = 1
RESEARCH_V2_REQUIRED_COMMON_COMPLETE_CONTRACT_ID = "research_v2_required_scope_common_complete_v1"
RESEARCH_V2_FREEZE_CANDIDATE_SCHEMA_VERSION = 3
RESEARCH_V2_FREEZE_CANDIDATE_STATUS_READY = "FREEZE_CANDIDATE_READY"

RESEARCH_V2_FREEZE_CANDIDATE_IDENTITY_FIELDS = (
    "schema_version",
    "generation_id",
    "status",
    "required_cutoff",
    "candidate_fingerprint",
    "required_source_projection_fingerprint",
    "daily_universe_file_sha256",
    "research_scope_contract_fingerprint",
    "adjusted_price_representation_contract_fingerprint",
    "adjusted_price_revision_proof_fingerprint",
    "required_dataset_scope",
    "common_complete_dataset_scope",
    "required_common_complete_start_date",
    "required_common_complete_tail_date_count",
    "required_common_complete_fingerprint",
    "research_common_complete_cutoff",
    "frozen_cutoff",
    "promotion_authorized",
    "active_research_generation_changed",
)


@dataclass(frozen=True)
class ResearchV2RequiredCommonCompleteSummary:
    schema_version: int
    contract_id: str
    required_cutoff: str
    participating_datasets: tuple[str, ...]
    trading_date_count: int
    common_complete_date_count: int
    common_complete_tail_start: str | None
    common_complete_cutoff: str | None
    common_complete_tail_date_count: int
    required_cutoff_complete: bool
    coverage_fingerprint: str


def build_research_v2_required_common_complete(
    *,
    trading_dates: Iterable[str],
    complete_dates_by_dataset: Mapping[str, Iterable[str]],
    required_cutoff: str,
    research_scope_contract_fingerprint: str,
    participating_datasets: Iterable[str] | None = None,
) -> tuple[tuple[str, ...], ResearchV2RequiredCommonCompleteSummary]:
    cutoff = str(required_cutoff or "").strip()
    scope_fp = str(research_scope_contract_fingerprint or "").strip().lower()
    if not cutoff:
        raise ValueError("Research V2 required common-complete cutoff 不可空白")
    if len(scope_fp) != 64:
        raise ValueError("Research V2 required common-complete scope fingerprint 不合法")

    participants = tuple(participating_datasets or RESEARCH_V2_COMMON_COMPLETE_DATASETS)
    if len(participants) != len(set(participants)):
        raise ValueError("Research V2 common-complete participant identity 重複")
    if set(participants) != set(RESEARCH_V2_COMMON_COMPLETE_DATASETS):
        raise ValueError(
            "Research V2 common-complete participant scope drift: "
            f"actual={sorted(participants)} expected={sorted(RESEARCH_V2_COMMON_COMPLETE_DATASETS)}"
        )

    calendar = tuple(sorted({str(value) for value in trading_dates if str(value) and str(value) <= cutoff}))
    normalized_by_dataset: dict[str, tuple[str, ...]] = {}
    for dataset in participants:
        if dataset not in complete_dates_by_dataset:
            raise ValueError(f"Research V2 common-complete 缺 dataset evidence: {dataset}")
        normalized_by_dataset[dataset] = tuple(
            sorted({str(value) for value in complete_dates_by_dataset[dataset] if str(value) and str(value) <= cutoff})
        )

    common = set(calendar)
    for dataset in participants:
        common.intersection_update(normalized_by_dataset[dataset])
    common_dates = tuple(sorted(common))

    runs: list[list[str]] = []
    active: list[str] = []
    for date_value in calendar:
        if date_value in common:
            active.append(date_value)
        elif active:
            runs.append(active)
            active = []
    if active:
        runs.append(active)
    latest_run = max(runs, key=lambda row: row[-1], default=[])
    cutoff_complete = bool(latest_run and latest_run[-1] == cutoff)
    common_cutoff = cutoff if cutoff_complete else None

    fingerprint_payload = {
        "schema_version": RESEARCH_V2_REQUIRED_COMMON_COMPLETE_SCHEMA_VERSION,
        "contract_id": RESEARCH_V2_REQUIRED_COMMON_COMPLETE_CONTRACT_ID,
        "required_cutoff": cutoff,
        "research_scope_contract_fingerprint": scope_fp,
        "participants": list(participants),
        "calendar": list(calendar),
        "complete_dates_by_dataset": {
            dataset: list(normalized_by_dataset[dataset])
            for dataset in participants
        },
        "common_complete_dates": list(common_dates),
        "latest_contiguous_tail": list(latest_run),
    }
    summary = ResearchV2RequiredCommonCompleteSummary(
        schema_version=RESEARCH_V2_REQUIRED_COMMON_COMPLETE_SCHEMA_VERSION,
        contract_id=RESEARCH_V2_REQUIRED_COMMON_COMPLETE_CONTRACT_ID,
        required_cutoff=cutoff,
        participating_datasets=participants,
        trading_date_count=len(calendar),
        common_complete_date_count=len(common_dates),
        common_complete_tail_start=latest_run[0] if latest_run else None,
        common_complete_cutoff=common_cutoff,
        common_complete_tail_date_count=len(latest_run),
        required_cutoff_complete=cutoff_complete,
        coverage_fingerprint=canonical_json_sha256(fingerprint_payload),
    )
    return common_dates, summary


def required_common_complete_summary_payload(
    summary: ResearchV2RequiredCommonCompleteSummary,
) -> dict[str, object]:
    payload = asdict(summary)
    payload["participating_datasets"] = list(summary.participating_datasets)
    return payload


def build_research_v2_freeze_candidate_identity_payload(
    *,
    required_cutoff: str,
    candidate_fingerprint: str,
    required_source_projection_fingerprint: str,
    daily_universe_file_sha256: str,
    research_scope_contract_fingerprint: str,
    adjusted_price_representation_contract_fingerprint: str,
    adjusted_price_revision_proof_fingerprint: str,
    required_common_complete_summary: ResearchV2RequiredCommonCompleteSummary,
) -> dict[str, object]:
    summary = required_common_complete_summary
    if not summary.required_cutoff_complete or summary.common_complete_cutoff != str(required_cutoff):
        raise ValueError("Research V2 required common-complete 尚未完整覆蓋 required cutoff，不得建立 freeze candidate")
    if set(summary.participating_datasets) != set(RESEARCH_V2_COMMON_COMPLETE_DATASETS):
        raise ValueError("Research V2 freeze candidate common-complete scope drift")
    return {
        "schema_version": RESEARCH_V2_FREEZE_CANDIDATE_SCHEMA_VERSION,
        "generation_id": "research_v2",
        "status": RESEARCH_V2_FREEZE_CANDIDATE_STATUS_READY,
        "required_cutoff": str(required_cutoff),
        "candidate_fingerprint": str(candidate_fingerprint),
        "required_source_projection_fingerprint": str(required_source_projection_fingerprint),
        "daily_universe_file_sha256": str(daily_universe_file_sha256),
        "research_scope_contract_fingerprint": str(research_scope_contract_fingerprint),
        "adjusted_price_representation_contract_fingerprint": str(adjusted_price_representation_contract_fingerprint),
        "adjusted_price_revision_proof_fingerprint": str(adjusted_price_revision_proof_fingerprint),
        "required_dataset_scope": list(RESEARCH_V2_REQUIRED_DATASETS),
        "common_complete_dataset_scope": list(RESEARCH_V2_COMMON_COMPLETE_DATASETS),
        "required_common_complete_start_date": summary.common_complete_tail_start,
        "required_common_complete_tail_date_count": int(summary.common_complete_tail_date_count),
        "required_common_complete_fingerprint": str(summary.coverage_fingerprint),
        "research_common_complete_cutoff": str(summary.common_complete_cutoff),
        "frozen_cutoff": str(required_cutoff),
        "promotion_authorized": False,
        "active_research_generation_changed": False,
    }


__all__ = [
    "RESEARCH_V2_REQUIRED_COMMON_COMPLETE_SCHEMA_VERSION",
    "RESEARCH_V2_REQUIRED_COMMON_COMPLETE_CONTRACT_ID",
    "RESEARCH_V2_FREEZE_CANDIDATE_SCHEMA_VERSION",
    "RESEARCH_V2_FREEZE_CANDIDATE_STATUS_READY",
    "RESEARCH_V2_FREEZE_CANDIDATE_IDENTITY_FIELDS",
    "ResearchV2RequiredCommonCompleteSummary",
    "build_research_v2_required_common_complete",
    "required_common_complete_summary_payload",
    "build_research_v2_freeze_candidate_identity_payload",
]
