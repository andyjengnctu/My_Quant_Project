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

RESEARCH_PIT_REVIEW_SCHEMA_VERSION = 2

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

PIT_LEGALITY_STATUS_OUTSIDE_ROUND12_SCOPE = "OUTSIDE_ROUND12_SCOPE"
PIT_LEGALITY_STATUS_EVENT_ANCHOR_READY = "EVENT_INFORMATION_TIME_ANCHOR_READY"
PIT_LEGALITY_STATUS_HISTORICAL_VINTAGE_BLOCKED = "HISTORICAL_PUBLICATION_VINTAGE_BLOCKED"
PIT_LEGALITY_STATUS_CURRENT_VINTAGE_BLOCKED = "CURRENT_VINTAGE_BLOCKED"


@dataclass(frozen=True)
class _ResearchV2NonDailyPitPolicy:
    pit_legality_status: str
    information_time_anchor_ready: bool
    information_time_rule: str
    information_time_columns: tuple[str, ...]
    revision_rule: str
    evidence_source: str
    field_scope_required: bool
    reason: str


def _event_policy(
    *,
    information_time_rule: str,
    information_time_columns: tuple[str, ...],
    revision_rule: str,
    evidence_source: str,
    reason: str,
) -> _ResearchV2NonDailyPitPolicy:
    return _ResearchV2NonDailyPitPolicy(
        pit_legality_status=PIT_LEGALITY_STATUS_EVENT_ANCHOR_READY,
        information_time_anchor_ready=True,
        information_time_rule=information_time_rule,
        information_time_columns=information_time_columns,
        revision_rule=revision_rule,
        evidence_source=evidence_source,
        field_scope_required=True,
        reason=reason,
    )


def _blocked_vintage_policy(*, evidence_source: str, reason: str) -> _ResearchV2NonDailyPitPolicy:
    return _ResearchV2NonDailyPitPolicy(
        pit_legality_status=PIT_LEGALITY_STATUS_HISTORICAL_VINTAGE_BLOCKED,
        information_time_anchor_ready=False,
        information_time_rule="blocked_missing_historical_publication_or_as_of_vintage",
        information_time_columns=(),
        revision_rule="blocked_without_historical_publication_and_revision_vintage",
        evidence_source=evidence_source,
        field_scope_required=True,
        reason=reason,
    )


# Canonical Research-only scientific evidence.  Do not reuse Trading publication
# schedules here: those describe when the current provider endpoint is expected to
# refresh, not when a historical row first became knowable.
_NON_DAILY_RESEARCH_PIT_POLICIES: dict[str, _ResearchV2NonDailyPitPolicy] = {
    # Event datasets: establish only a conservative information-time anchor.
    # This does not authorize the whole row: later-populated/revised fields remain
    # blocked until a future required feature field set is explicitly scoped.
    "TaiwanStockSuspended": _event_policy(
        information_time_rule="next_taiwan_trading_session_after_event_date",
        information_time_columns=("date",),
        revision_rule="event_occurrence_only; resumption_or_later_fields_require_separate_as_of_proof",
        evidence_source="provider_schema:TaiwanStockSuspended.date=suspension_date",
        reason="suspension start is observable after the event date; later resumption fields must not be back-projected",
    ),
    "TaiwanStockDayTradingSuspension": _event_policy(
        information_time_rule="next_taiwan_trading_session_after_event_date",
        information_time_columns=("date",),
        revision_rule="event_occurrence_only; end_date_or_later_fields_require_separate_as_of_proof",
        evidence_source="provider_schema:TaiwanStockDayTradingSuspension.date=start_date",
        reason="day-trading suspension occurrence can be anchored after its start date without assuming same-day publication",
    ),
    "TaiwanStockMarginShortSaleSuspension": _event_policy(
        information_time_rule="next_taiwan_trading_session_after_event_date",
        information_time_columns=("date",),
        revision_rule="event_occurrence_only; end_date_or_later_fields_require_separate_as_of_proof",
        evidence_source="provider_schema:TaiwanStockMarginShortSaleSuspension.date=start_date",
        reason="margin/short-sale suspension occurrence can be anchored after its start date only",
    ),
    "TaiwanStockDispositionSecuritiesPeriod": _event_policy(
        information_time_rule="next_taiwan_trading_session_after_announcement_date",
        information_time_columns=("date",),
        revision_rule="announcement_payload_only; later corrections require separately versioned evidence",
        evidence_source="provider_schema:TaiwanStockDispositionSecuritiesPeriod.date=announcement_date",
        reason="provider schema identifies date as announcement date; absent an intraday timestamp, same-day use remains disallowed",
    ),
    "TaiwanStockDividend": _event_policy(
        information_time_rule="next_taiwan_trading_session_after_announcement_timestamp",
        information_time_columns=("AnnouncementDate", "AnnouncementTime"),
        revision_rule="announcement_time_field_scope_only; later_populated_or_revised_fields_require_separate_as_of_proof",
        evidence_source="provider_schema:TaiwanStockDividend.AnnouncementDate+AnnouncementTime",
        reason="provider rows expose an announcement date/time anchor, but the full row contains fields whose historical availability must be scoped separately",
    ),
    "TaiwanStockDividendResult": _event_policy(
        information_time_rule="next_taiwan_trading_session_after_event_date",
        information_time_columns=("date",),
        revision_rule="realized_result_only_after_event; no_pre_event_or_unversioned_revision_use",
        evidence_source="provider_schema:TaiwanStockDividendResult.date=event_date",
        reason="realized ex-right/ex-dividend result may only be treated as known after the dated event",
    ),
    "TaiwanStockCapitalReductionReferencePrice": _event_policy(
        information_time_rule="next_taiwan_trading_session_after_event_date",
        information_time_columns=("date",),
        revision_rule="dated_reference_event_only; later corrections require separately versioned evidence",
        evidence_source="provider_schema:TaiwanStockCapitalReductionReferencePrice.date=reference_date",
        reason="reference-price event is conservatively anchored only after its dated event",
    ),
    "TaiwanStockSplitPrice": _event_policy(
        information_time_rule="next_taiwan_trading_session_after_event_date",
        information_time_columns=("date",),
        revision_rule="dated_split_event_only; later corrections require separately versioned evidence",
        evidence_source="provider_schema:TaiwanStockSplitPrice.date=split_date",
        reason="split event is conservatively anchored only after the split date",
    ),
    "TaiwanStockParValueChange": _event_policy(
        information_time_rule="next_taiwan_trading_session_after_event_date",
        information_time_columns=("date",),
        revision_rule="dated_par_value_event_only; later corrections require separately versioned evidence",
        evidence_source="provider_schema:TaiwanStockParValueChange.date=event_date",
        reason="par-value change is conservatively anchored only after the dated event",
    ),

    # Periodic datasets: the archived row's period/effective date is not a
    # historical publication-vintage timestamp.  Current endpoint refresh timing
    # therefore cannot legalize retrospective Research use.
    "TaiwanStockHoldingSharesPer": _blocked_vintage_policy(
        evidence_source="provider_schema:TaiwanStockHoldingSharesPer.period_date_without_publication_vintage",
        reason="holding-distribution period date does not prove when that historical row first became available or whether it was revised",
    ),
    "TaiwanStockFinancialStatements": _blocked_vintage_policy(
        evidence_source="provider_schema:TaiwanStockFinancialStatements.period_date_without_publication_vintage",
        reason="financial-statement period end is not publication time and the archive does not preserve historical revision vintages",
    ),
    "TaiwanStockBalanceSheet": _blocked_vintage_policy(
        evidence_source="provider_schema:TaiwanStockBalanceSheet.period_date_without_publication_vintage",
        reason="balance-sheet period end is not publication time and the archive does not preserve historical revision vintages",
    ),
    "TaiwanStockCashFlowsStatement": _blocked_vintage_policy(
        evidence_source="provider_schema:TaiwanStockCashFlowsStatement.period_date_without_publication_vintage",
        reason="cash-flow period end is not publication time and the archive does not preserve historical revision vintages",
    ),
    "TaiwanStockMonthRevenue": _blocked_vintage_policy(
        evidence_source="provider_schema:TaiwanStockMonthRevenue.create_time_not_historically_available_before_2026-04-21",
        reason="provider create_time is unavailable for the historical Research horizon, so publication/revision time cannot be reconstructed for prior rows",
    ),
    "TaiwanStockMarketValueWeight": _blocked_vintage_policy(
        evidence_source="provider_schema:TaiwanStockMarketValueWeight.period_date_without_historical_publication_vintage",
        reason="provider operational update schedule is not row-level historical publication evidence and cannot define Research PIT identity",
    ),
    "TaiwanBusinessIndicator": _blocked_vintage_policy(
        evidence_source="provider_schema:TaiwanBusinessIndicator.period_date_without_publication_vintage",
        reason="macro period date does not preserve first-publication or revision vintage",
    ),
    "InterestRate": _blocked_vintage_policy(
        evidence_source="provider_schema:InterestRate.effective_date_without_publication_vintage",
        reason="rate effective date does not prove historical announcement availability and revision vintage",
    ),

    # Static/current-vintage review datasets: present-day rows cannot reconstruct
    # historical state transitions unless the provider archive itself carries an
    # explicit as-of chain.
    "TaiwanStockInfo": _blocked_vintage_policy(
        evidence_source="provider_schema:TaiwanStockInfo.current_update_rows_with_partial_transition_history",
        reason="current security-master rows plus partial transition rows do not prove every attribute's historical as-of value",
    ),
    "TaiwanStockIndustryChain": _blocked_vintage_policy(
        evidence_source="provider_schema:TaiwanStockIndustryChain.current_update_date_without_transition_vintage",
        reason="current industry-chain mapping does not preserve a complete historical transition/as-of chain",
    ),
    "TaiwanStockActiveETFInfo": _blocked_vintage_policy(
        evidence_source="provider_schema:TaiwanStockActiveETFInfo.current_active_list_without_historical_state_chain",
        reason="current active-ETF list cannot be back-projected without historical membership transitions",
    ),
    "TaiwanFutOptDailyInfo": _blocked_vintage_policy(
        evidence_source="provider_schema:TaiwanFutOptDailyInfo.no_historical_as_of_column",
        reason="derivative product master has no historical as-of column or transition chain",
    ),
}


@dataclass(frozen=True)
class ResearchV2PitReviewContract:
    dataset: str
    pit_class: str
    cadence: str
    audit_mode: str
    review_status: str
    automatic_date_presence_audit: bool
    contributes_mechanical_trading_daily_ceiling: bool
    pit_legality_status: str
    information_time_anchor_ready: bool
    information_time_rule: str
    information_time_columns: tuple[str, ...]
    revision_rule: str
    evidence_source: str
    field_scope_required: bool
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


def _base_contract_kwargs(*, spec: MarketDatasetSpec, freshness: MarketDataFreshnessContract) -> dict[str, object]:
    return {
        "dataset": spec.dataset,
        "pit_class": spec.pit_class,
        "cadence": freshness.cadence,
        "scientific_input_authorized": False,
    }


def _resolve_contract(
    spec: MarketDatasetSpec,
    freshness: MarketDataFreshnessContract,
) -> ResearchV2PitReviewContract:
    if spec.dataset != freshness.dataset:
        raise ValueError("Research V2 PIT review spec/freshness dataset identity drift")

    base = _base_contract_kwargs(spec=spec, freshness=freshness)
    if spec.pit_class == PIT_CURRENT_VINTAGE:
        return ResearchV2PitReviewContract(
            **base,
            audit_mode=AUDIT_MODE_CURRENT_VINTAGE_BLOCKED,
            review_status=PIT_REVIEW_STATUS_CURRENT_VINTAGE_BLOCKED,
            automatic_date_presence_audit=False,
            contributes_mechanical_trading_daily_ceiling=False,
            pit_legality_status=PIT_LEGALITY_STATUS_CURRENT_VINTAGE_BLOCKED,
            information_time_anchor_ready=False,
            information_time_rule="blocked_current_vintage_without_retrospective_representation_invariance",
            information_time_columns=(),
            revision_rule="blocked_current_vintage_representation",
            evidence_source="research_contract:current_vintage_representation_requires_retrospective_proof",
            field_scope_required=True,
            reason="current-vintage representation requires retrospective invariance/PIT proof before Research use",
        )

    if spec.pit_class == PIT_EXACT_CANDIDATE:
        return ResearchV2PitReviewContract(
            **base,
            audit_mode=AUDIT_MODE_EXACT_CANDIDATE,
            review_status=PIT_REVIEW_STATUS_AUTOMATIC_EXACT_AUDIT,
            automatic_date_presence_audit=False,
            contributes_mechanical_trading_daily_ceiling=(freshness.cadence == CADENCE_TRADING_DAILY),
            pit_legality_status=PIT_LEGALITY_STATUS_OUTSIDE_ROUND12_SCOPE,
            information_time_anchor_ready=False,
            information_time_rule="exact_candidate_audit_is_owned_by_round9_exact_coverage_contract",
            information_time_columns=(),
            revision_rule="separate_exact_candidate_contract",
            evidence_source="research_contract:round9_exact_candidate",
            field_scope_required=False,
            reason="provider registry permits automatic exact-candidate audit; model-input authorization remains separate",
        )

    if spec.pit_class != PIT_REVIEW_REQUIRED:
        raise ValueError(f"Research V2 PIT review 不支援 pit_class: {spec.dataset} -> {spec.pit_class}")

    if freshness.cadence == CADENCE_TRADING_DAILY:
        return ResearchV2PitReviewContract(
            **base,
            audit_mode=AUDIT_MODE_TRADING_DAILY_DATE_PRESENCE,
            review_status=PIT_REVIEW_STATUS_REVIEW_REQUIRED,
            automatic_date_presence_audit=True,
            contributes_mechanical_trading_daily_ceiling=True,
            pit_legality_status=PIT_LEGALITY_STATUS_OUTSIDE_ROUND12_SCOPE,
            information_time_anchor_ready=False,
            information_time_rule="daily_date_presence_only; scientific_availability_rule_deferred",
            information_time_columns=("date",),
            revision_rule="daily_publication_and_revision_semantics_not_authorized_by_date_presence",
            evidence_source="research_contract:round10_trading_daily_date_presence",
            field_scope_required=True,
            reason=(
                "dated trading-day rows can be mechanically checked for date presence only; "
                "publication/revision semantics still require dataset-specific PIT authorization"
            ),
        )
    if freshness.cadence == CADENCE_CALENDAR_DAILY:
        return ResearchV2PitReviewContract(
            **base,
            audit_mode=AUDIT_MODE_CALENDAR_DAILY_DATE_PRESENCE,
            review_status=PIT_REVIEW_STATUS_REVIEW_REQUIRED,
            automatic_date_presence_audit=True,
            contributes_mechanical_trading_daily_ceiling=False,
            pit_legality_status=PIT_LEGALITY_STATUS_OUTSIDE_ROUND12_SCOPE,
            information_time_anchor_ready=False,
            information_time_rule="daily_date_presence_only; scientific_availability_rule_deferred",
            information_time_columns=("date",),
            revision_rule="daily_publication_and_revision_semantics_not_authorized_by_date_presence",
            evidence_source="research_contract:round10_calendar_daily_date_presence",
            field_scope_required=True,
            reason="dated calendar-day rows can be mechanically summarized, but they do not define the Taiwan trading-day common cutoff",
        )

    if freshness.cadence == CADENCE_EVENT_DRIVEN:
        audit_mode = AUDIT_MODE_EVENT_INFORMATION_TIME
    elif freshness.cadence == CADENCE_PERIODIC:
        audit_mode = AUDIT_MODE_PERIODIC_PUBLICATION
    elif freshness.cadence == CADENCE_CURRENT_VINTAGE:
        audit_mode = AUDIT_MODE_STATIC_CURRENT_VINTAGE
    else:
        raise ValueError(f"Research V2 PIT review 不支援 cadence: {spec.dataset} -> {freshness.cadence}")

    policy = _NON_DAILY_RESEARCH_PIT_POLICIES.get(spec.dataset)
    if policy is None:
        raise ValueError(f"Research V2 non-daily PIT legality 缺 canonical policy: {spec.dataset}")
    return ResearchV2PitReviewContract(
        **base,
        audit_mode=audit_mode,
        review_status=PIT_REVIEW_STATUS_REVIEW_REQUIRED,
        automatic_date_presence_audit=False,
        contributes_mechanical_trading_daily_ceiling=False,
        pit_legality_status=policy.pit_legality_status,
        information_time_anchor_ready=policy.information_time_anchor_ready,
        information_time_rule=policy.information_time_rule,
        information_time_columns=policy.information_time_columns,
        revision_rule=policy.revision_rule,
        evidence_source=policy.evidence_source,
        field_scope_required=policy.field_scope_required,
        reason=policy.reason,
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


def research_v2_pit_review_contract_payloads(
    contracts: Iterable[ResearchV2PitReviewContract] | None = None,
) -> list[dict[str, object]]:
    rows = tuple(contracts or build_research_v2_pit_review_contracts())
    payloads: list[dict[str, object]] = []
    for row in rows:
        payload = asdict(row)
        payload["information_time_columns"] = list(row.information_time_columns)
        payloads.append(payload)
    return payloads


def research_v2_pit_review_contract_fingerprint(
    contracts: Iterable[ResearchV2PitReviewContract] | None = None,
) -> str:
    rows = tuple(contracts or build_research_v2_pit_review_contracts())
    return canonical_json_sha256(
        {
            "schema_version": RESEARCH_PIT_REVIEW_SCHEMA_VERSION,
            "contracts": research_v2_pit_review_contract_payloads(rows),
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
        raise ValueError("Research V2 PIT review contract 不得自行授權任何 Research model input")

    freshness_by_dataset = {row.dataset: row for row in get_market_data_freshness_contracts()}
    expected_non_daily = {
        row.dataset
        for row in rows
        if row.pit_class == PIT_REVIEW_REQUIRED
        and freshness_by_dataset[row.dataset].cadence in {CADENCE_EVENT_DRIVEN, CADENCE_PERIODIC, CADENCE_CURRENT_VINTAGE}
    }
    policy_ids = set(_NON_DAILY_RESEARCH_PIT_POLICIES)
    if policy_ids != expected_non_daily:
        raise ValueError(
            "Research V2 non-daily PIT policy coverage drift: "
            f"missing={sorted(expected_non_daily - policy_ids)}, extra={sorted(policy_ids - expected_non_daily)}"
        )

    non_daily_rows = [row for row in rows if row.dataset in expected_non_daily]
    event_rows = [row for row in non_daily_rows if row.cadence == CADENCE_EVENT_DRIVEN]
    blocked_vintage_rows = [
        row for row in non_daily_rows if row.cadence in {CADENCE_PERIODIC, CADENCE_CURRENT_VINTAGE}
    ]
    if any(
        row.pit_legality_status != PIT_LEGALITY_STATUS_EVENT_ANCHOR_READY
        or not row.information_time_anchor_ready
        or not row.information_time_columns
        or not row.field_scope_required
        for row in event_rows
    ):
        raise ValueError("Research V2 event PIT policy 必須只有 conservative information-time anchor，且不得省略 field-scope gate")
    if any(
        row.pit_legality_status != PIT_LEGALITY_STATUS_HISTORICAL_VINTAGE_BLOCKED
        or row.information_time_anchor_ready
        or not row.field_scope_required
        for row in blocked_vintage_rows
    ):
        raise ValueError("Research V2 periodic/static PIT policy 必須在缺 historical publication/as-of vintage 時 fail-closed")
    if any(not row.evidence_source or not row.revision_rule or not row.information_time_rule for row in rows):
        raise ValueError("Research V2 PIT review contract 缺 legality evidence/rule identity")

    mode_counts: dict[str, int] = {}
    legality_status_counts: dict[str, int] = {}
    for row in rows:
        mode_counts[row.audit_mode] = mode_counts.get(row.audit_mode, 0) + 1
        legality_status_counts[row.pit_legality_status] = legality_status_counts.get(row.pit_legality_status, 0) + 1
    return {
        "dataset_count": len(rows),
        "mode_counts": mode_counts,
        "pit_legality_status_counts": legality_status_counts,
        "automatic_date_audit_count": sum(row.automatic_date_presence_audit for row in rows),
        "mechanical_trading_daily_constraint_count": sum(
            row.contributes_mechanical_trading_daily_ceiling for row in rows
        ),
        "review_required_count": sum(row.review_status == PIT_REVIEW_STATUS_REVIEW_REQUIRED for row in rows),
        "current_vintage_blocked_count": sum(
            row.review_status == PIT_REVIEW_STATUS_CURRENT_VINTAGE_BLOCKED for row in rows
        ),
        "non_daily_policy_count": len(non_daily_rows),
        "event_information_time_anchor_ready_count": len(event_rows),
        "historical_publication_vintage_blocked_count": len(blocked_vintage_rows),
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
    "PIT_LEGALITY_STATUS_OUTSIDE_ROUND12_SCOPE",
    "PIT_LEGALITY_STATUS_EVENT_ANCHOR_READY",
    "PIT_LEGALITY_STATUS_HISTORICAL_VINTAGE_BLOCKED",
    "PIT_LEGALITY_STATUS_CURRENT_VINTAGE_BLOCKED",
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
    "research_v2_pit_review_contract_payloads",
    "research_v2_pit_review_contract_fingerprint",
    "validate_research_v2_pit_review_contracts",
    "summarize_mechanical_common_complete_tail",
]
