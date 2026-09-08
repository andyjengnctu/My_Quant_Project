"""Research V2 dataset PIT-review and mechanical completeness contracts.

This layer classifies *how* an archived dataset can be audited without granting
scientific authorization to consume it.  In particular, a date-presence audit
only proves that dated provider rows exist; it does not prove publication-time,
revision-history, or model-input PIT legality.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

from core.file_integrity import canonical_json_sha256
from core.market_data_dataset_registry import (
    PIT_CURRENT_VINTAGE,
    PIT_EXACT_CANDIDATE,
    PIT_REVIEW_REQUIRED,
    MarketDatasetSpec,
    get_market_dataset_specs,
)
from core.market_data_freshness_contract import (
    CADENCE_CALENDAR_DAILY,
    CADENCE_CURRENT_VINTAGE,
    CADENCE_EVENT_DRIVEN,
    CADENCE_PERIODIC,
    CADENCE_TRADING_DAILY,
    MarketDataFreshnessContract,
    get_market_data_freshness_contracts,
)

RESEARCH_PIT_REVIEW_SCHEMA_VERSION = 1

AUDIT_MODE_EXACT_CANDIDATE = "exact_candidate"
AUDIT_MODE_TRADING_DAILY_DATE_PRESENCE = "trading_daily_date_presence_review"
AUDIT_MODE_CALENDAR_DAILY_DATE_PRESENCE = "calendar_daily_date_presence_review"
AUDIT_MODE_EVENT_INFORMATION_TIME = "event_information_time_review"
AUDIT_MODE_PERIODIC_PUBLICATION = "periodic_publication_revision_review"
AUDIT_MODE_STATIC_CURRENT_VINTAGE = "static_current_vintage_review"
AUDIT_MODE_CURRENT_VINTAGE_BLOCKED = "current_vintage_blocked"

PIT_REVIEW_STATUS_AUTOMATIC_EXACT_AUDIT = "AUTOMATIC_EXACT_AUDIT"
PIT_REVIEW_STATUS_REVIEW_REQUIRED = "REVIEW_REQUIRED"
PIT_REVIEW_STATUS_CURRENT_VINTAGE_BLOCKED = "CURRENT_VINTAGE_BLOCKED"

DATE_AUDIT_STATUS_NOT_APPLICABLE = "NOT_APPLICABLE"
DATE_AUDIT_STATUS_READY = "READY"
DATE_AUDIT_STATUS_NO_ARTIFACTS = "NO_ARTIFACTS"
DATE_AUDIT_STATUS_NO_DATED_ROWS = "NO_DATED_ROWS"
DATE_AUDIT_STATUS_SCHEMA_BLOCKED = "SCHEMA_BLOCKED"
DATE_AUDIT_STATUS_ERROR = "ERROR"


@dataclass(frozen=True)
class ResearchV2PitReviewContract:
    dataset: str
    pit_class: str
    cadence: str
    audit_mode: str
    review_status: str
    automatic_date_presence_audit: bool
    contributes_mechanical_trading_daily_ceiling: bool
    scientific_input_authorized: bool
    reason: str


@dataclass(frozen=True)
class ResearchV2DatasetDateAudit:
    dataset: str
    audit_mode: str
    audit_status: str
    observed_date_count: int
    first_observed_date: str | None
    latest_observed_date: str | None
    reason: str


@dataclass(frozen=True)
class ResearchV2MechanicalCommonCompleteSummary:
    participating_dataset_count: int
    successful_dataset_count: int
    trading_date_count: int
    common_complete_date_count: int
    common_complete_tail_start: str | None
    common_complete_ceiling_date: str | None
    common_complete_tail_date_count: int
    audit_complete: bool
    coverage_fingerprint: str


def _resolve_contract(
    spec: MarketDatasetSpec,
    freshness: MarketDataFreshnessContract,
) -> ResearchV2PitReviewContract:
    if spec.dataset != freshness.dataset:
        raise ValueError("Research V2 PIT review spec/freshness dataset identity drift")

    if spec.pit_class == PIT_CURRENT_VINTAGE:
        return ResearchV2PitReviewContract(
            dataset=spec.dataset,
            pit_class=spec.pit_class,
            cadence=freshness.cadence,
            audit_mode=AUDIT_MODE_CURRENT_VINTAGE_BLOCKED,
            review_status=PIT_REVIEW_STATUS_CURRENT_VINTAGE_BLOCKED,
            automatic_date_presence_audit=False,
            contributes_mechanical_trading_daily_ceiling=False,
            scientific_input_authorized=False,
            reason="current-vintage representation requires retrospective invariance/PIT proof before Research use",
        )

    if spec.pit_class == PIT_EXACT_CANDIDATE:
        return ResearchV2PitReviewContract(
            dataset=spec.dataset,
            pit_class=spec.pit_class,
            cadence=freshness.cadence,
            audit_mode=AUDIT_MODE_EXACT_CANDIDATE,
            review_status=PIT_REVIEW_STATUS_AUTOMATIC_EXACT_AUDIT,
            automatic_date_presence_audit=False,
            contributes_mechanical_trading_daily_ceiling=(freshness.cadence == CADENCE_TRADING_DAILY),
            scientific_input_authorized=False,
            reason="provider registry permits automatic exact-candidate audit; model-input authorization remains separate",
        )

    if spec.pit_class != PIT_REVIEW_REQUIRED:
        raise ValueError(f"Research V2 PIT review 不支援 pit_class: {spec.dataset} -> {spec.pit_class}")

    if freshness.cadence == CADENCE_TRADING_DAILY:
        audit_mode = AUDIT_MODE_TRADING_DAILY_DATE_PRESENCE
        automatic = True
        contributes = True
        reason = (
            "dated trading-day rows can be mechanically checked for date presence only; "
            "publication/revision semantics still require dataset-specific PIT authorization"
        )
    elif freshness.cadence == CADENCE_CALENDAR_DAILY:
        audit_mode = AUDIT_MODE_CALENDAR_DAILY_DATE_PRESENCE
        automatic = True
        contributes = False
        reason = (
            "dated calendar-day rows can be mechanically summarized, but they do not define the Taiwan trading-day common cutoff"
        )
    elif freshness.cadence == CADENCE_EVENT_DRIVEN:
        audit_mode = AUDIT_MODE_EVENT_INFORMATION_TIME
        automatic = False
        contributes = False
        reason = "event effective date is not automatically equivalent to decision-time information availability"
    elif freshness.cadence == CADENCE_PERIODIC:
        audit_mode = AUDIT_MODE_PERIODIC_PUBLICATION
        automatic = False
        contributes = False
        reason = "period end/date is not automatically the publication/revision information time"
    elif freshness.cadence == CADENCE_CURRENT_VINTAGE:
        audit_mode = AUDIT_MODE_STATIC_CURRENT_VINTAGE
        automatic = False
        contributes = False
        reason = "static/current-vintage metadata needs historical transition or as-of evidence before retrospective use"
    else:
        raise ValueError(f"Research V2 PIT review 不支援 cadence: {spec.dataset} -> {freshness.cadence}")

    return ResearchV2PitReviewContract(
        dataset=spec.dataset,
        pit_class=spec.pit_class,
        cadence=freshness.cadence,
        audit_mode=audit_mode,
        review_status=PIT_REVIEW_STATUS_REVIEW_REQUIRED,
        automatic_date_presence_audit=automatic,
        contributes_mechanical_trading_daily_ceiling=contributes,
        scientific_input_authorized=False,
        reason=reason,
    )


def build_research_v2_pit_review_contracts(
    *,
    specs: Iterable[MarketDatasetSpec] | None = None,
    freshness_contracts: Iterable[MarketDataFreshnessContract] | None = None,
) -> tuple[ResearchV2PitReviewContract, ...]:
    spec_rows = tuple(specs or get_market_dataset_specs(included_only=True))
    freshness_rows = tuple(freshness_contracts or get_market_data_freshness_contracts())
    freshness_by_dataset = {row.dataset: row for row in freshness_rows}
    if len(freshness_by_dataset) != len(freshness_rows):
        raise ValueError("Research V2 PIT review freshness dataset identity 重複")
    rows: list[ResearchV2PitReviewContract] = []
    for spec in spec_rows:
        if not spec.included:
            continue
        freshness = freshness_by_dataset.get(spec.dataset)
        if freshness is None:
            raise ValueError(f"Research V2 PIT review 缺 freshness contract: {spec.dataset}")
        rows.append(_resolve_contract(spec, freshness))
    validate_research_v2_pit_review_contracts(rows)
    return tuple(rows)


def research_v2_pit_review_contract_fingerprint(
    contracts: Iterable[ResearchV2PitReviewContract] | None = None,
) -> str:
    rows = tuple(contracts or build_research_v2_pit_review_contracts())
    return canonical_json_sha256(
        {
            "schema_version": RESEARCH_PIT_REVIEW_SCHEMA_VERSION,
            "contracts": [asdict(row) for row in rows],
        }
    )


def validate_research_v2_pit_review_contracts(
    contracts: Iterable[ResearchV2PitReviewContract] | None = None,
) -> dict[str, object]:
    rows = tuple(contracts or build_research_v2_pit_review_contracts())
    names = [row.dataset for row in rows]
    expected = [spec.dataset for spec in get_market_dataset_specs(included_only=True)]
    if len(names) != len(set(names)):
        raise ValueError("Research V2 PIT review contract dataset identity 重複")
    if set(names) != set(expected):
        raise ValueError(
            "Research V2 PIT review contract coverage drift: "
            f"missing={sorted(set(expected) - set(names))}, extra={sorted(set(names) - set(expected))}"
        )
    if any(row.scientific_input_authorized for row in rows):
        raise ValueError("Round 10 PIT review contract 不得自行授權任何 Research model input")

    mode_counts: dict[str, int] = {}
    for row in rows:
        mode_counts[row.audit_mode] = mode_counts.get(row.audit_mode, 0) + 1
    return {
        "dataset_count": len(rows),
        "mode_counts": mode_counts,
        "automatic_date_audit_count": sum(row.automatic_date_presence_audit for row in rows),
        "mechanical_trading_daily_constraint_count": sum(
            row.contributes_mechanical_trading_daily_ceiling for row in rows
        ),
        "review_required_count": sum(row.review_status == PIT_REVIEW_STATUS_REVIEW_REQUIRED for row in rows),
        "current_vintage_blocked_count": sum(
            row.review_status == PIT_REVIEW_STATUS_CURRENT_VINTAGE_BLOCKED for row in rows
        ),
        "contract_fingerprint": research_v2_pit_review_contract_fingerprint(rows),
    }


def summarize_mechanical_common_complete_tail(
    *,
    trading_dates: Iterable[str],
    exact_complete_dates: Iterable[str],
    dataset_audits: Iterable[ResearchV2DatasetDateAudit],
    observed_dates_by_dataset: Mapping[str, Iterable[str]],
    participating_datasets: Iterable[str],
) -> ResearchV2MechanicalCommonCompleteSummary:
    calendar = tuple(sorted({str(value) for value in trading_dates if str(value)}))
    exact = {str(value) for value in exact_complete_dates if str(value)}
    audit_by_dataset = {row.dataset: row for row in dataset_audits}
    participants = tuple(sorted({str(value) for value in participating_datasets if str(value)}))
    successful = tuple(
        dataset
        for dataset in participants
        if dataset in audit_by_dataset and audit_by_dataset[dataset].audit_status == DATE_AUDIT_STATUS_READY
    )
    audit_complete = len(successful) == len(participants)

    common = set(calendar).intersection(exact)
    if audit_complete:
        for dataset in participants:
            common.intersection_update({str(value) for value in observed_dates_by_dataset.get(dataset, ()) if str(value)})
    else:
        common.clear()

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
    payload = {
        "participants": participants,
        "successful": successful,
        "calendar": calendar,
        "common_complete_dates": tuple(sorted(common)),
        "tail": tuple(latest_run),
        "audit_complete": audit_complete,
    }
    return ResearchV2MechanicalCommonCompleteSummary(
        participating_dataset_count=len(participants),
        successful_dataset_count=len(successful),
        trading_date_count=len(calendar),
        common_complete_date_count=len(common),
        common_complete_tail_start=latest_run[0] if latest_run else None,
        common_complete_ceiling_date=latest_run[-1] if latest_run else None,
        common_complete_tail_date_count=len(latest_run),
        audit_complete=audit_complete,
        coverage_fingerprint=canonical_json_sha256(payload),
    )


__all__ = [
    "RESEARCH_PIT_REVIEW_SCHEMA_VERSION",
    "AUDIT_MODE_EXACT_CANDIDATE",
    "AUDIT_MODE_TRADING_DAILY_DATE_PRESENCE",
    "AUDIT_MODE_CALENDAR_DAILY_DATE_PRESENCE",
    "AUDIT_MODE_EVENT_INFORMATION_TIME",
    "AUDIT_MODE_PERIODIC_PUBLICATION",
    "AUDIT_MODE_STATIC_CURRENT_VINTAGE",
    "AUDIT_MODE_CURRENT_VINTAGE_BLOCKED",
    "PIT_REVIEW_STATUS_AUTOMATIC_EXACT_AUDIT",
    "PIT_REVIEW_STATUS_REVIEW_REQUIRED",
    "PIT_REVIEW_STATUS_CURRENT_VINTAGE_BLOCKED",
    "DATE_AUDIT_STATUS_NOT_APPLICABLE",
    "DATE_AUDIT_STATUS_READY",
    "DATE_AUDIT_STATUS_NO_ARTIFACTS",
    "DATE_AUDIT_STATUS_NO_DATED_ROWS",
    "DATE_AUDIT_STATUS_SCHEMA_BLOCKED",
    "DATE_AUDIT_STATUS_ERROR",
    "ResearchV2PitReviewContract",
    "ResearchV2DatasetDateAudit",
    "ResearchV2MechanicalCommonCompleteSummary",
    "build_research_v2_pit_review_contracts",
    "research_v2_pit_review_contract_fingerprint",
    "validate_research_v2_pit_review_contracts",
    "summarize_mechanical_common_complete_tail",
]
