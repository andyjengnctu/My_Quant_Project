"""Canonical Research V2 dataset scope and field-level scientific authorization.

The neutral Market Data V2 archive intentionally contains more datasets than one
Research generation requires.  This module freezes the narrower Research V2
foundation scope without turning every archived dataset into a model input.

Required foundation semantics:
* TaiwanStockTradingDate: trading-calendar evidence only.
* TaiwanStockDelisting: historical universe event evidence only; no-row is legal.
* TaiwanStockPrice: (date, stock_id) universe evidence plus Trading_Volume only.
  Raw OHLC remains prohibited as model price input by project policy.
* TaiwanStockPriceLimit: exact daily coverage evidence only.
* TaiwanStockPriceAdj: price source only through the Round-13 invariant
  representations; raw/absolute adjusted levels and Trading_Volume are not
  directly authorized here.

Every other archived dataset stays optional/not-selected for direct field/model
input authorization.  That designation does not prevent a separate universe/PIT
guard from consuming neutral metadata evidence (for example TaiwanStockInfo
market-transition rows) when sample membership itself must be historically legal.
Such guard evidence is pinned by the candidate identity, not promoted into the
foundation field-input matrix.  Optional field PIT review state is preserved for
future experiment-specific authorization and cannot block Research V2 foundation
freeze merely because the dataset exists in the archive.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from core.file_integrity import canonical_json_sha256
from core.market_data_contract import (
    FINMIND_ADJUSTED_PRICE_DATASET,
    FINMIND_RAW_PRICE_ARCHIVE_DATASET,
)
from core.market_data_adjusted_price_invariance import (
    PRICE_REPRESENTATION_STATUS_INVARIANT,
    PROVIDER_PRICE_FIELDS,
    PROVIDER_VOLUME_FIELD,
    get_adjusted_price_representation_contract,
    validate_adjusted_price_representation_contract,
)
from core.market_data_dataset_registry import get_market_dataset_specs
from core.market_data_research_pit_contract import (
    build_research_v2_pit_review_contracts,
    research_v2_pit_review_contract_payloads,
    validate_research_v2_pit_review_contracts,
)

RESEARCH_V2_SCOPE_SCHEMA_VERSION = 1
RESEARCH_V2_SCOPE_CONTRACT_ID = "research_v2_foundation_scope_field_authorization_v1"

SCOPE_STATUS_REQUIRED_AUTHORIZED = "REQUIRED_AUTHORIZED"
SCOPE_STATUS_OPTIONAL_NOT_SELECTED = "OPTIONAL_NOT_SELECTED"

FIELD_USE_EVIDENCE_ONLY = "EVIDENCE_ONLY"
FIELD_USE_DIRECT_INPUT = "DIRECT_INPUT"
FIELD_USE_TRANSFORM_SOURCE_ONLY = "TRANSFORM_SOURCE_ONLY"

RESEARCH_V2_REQUIRED_DATASETS = (
    "TaiwanStockTradingDate",
    "TaiwanStockDelisting",
    FINMIND_RAW_PRICE_ARCHIVE_DATASET,
    "TaiwanStockPriceLimit",
    FINMIND_ADJUSTED_PRICE_DATASET,
)

RESEARCH_V2_COMMON_COMPLETE_DATASETS = (
    "TaiwanStockTradingDate",
    FINMIND_RAW_PRICE_ARCHIVE_DATASET,
    "TaiwanStockPriceLimit",
    FINMIND_ADJUSTED_PRICE_DATASET,
)

RESEARCH_V2_RAW_VOLUME_DATASET = FINMIND_RAW_PRICE_ARCHIVE_DATASET
RESEARCH_V2_ADJUSTED_PRICE_DATASET = FINMIND_ADJUSTED_PRICE_DATASET


@dataclass(frozen=True)
class ResearchV2FieldAuthorization:
    fields: tuple[str, ...]
    use_mode: str
    scientific_role: str
    availability_rule: str
    revision_rule: str
    direct_scientific_use_authorized: bool
    reason: str


@dataclass(frozen=True)
class ResearchV2DatasetScopeContract:
    dataset: str
    required_for_generation: bool
    contributes_to_common_complete: bool
    scope_status: str
    evidence_fields: tuple[str, ...]
    field_authorizations: tuple[ResearchV2FieldAuthorization, ...]
    authorized_representation_ids: tuple[str, ...]
    unlisted_fields_authorized: bool
    pit_contract_status: str
    pit_contract_fingerprint: str
    authorization_basis: str
    reason: str


def _field_rule(
    fields: tuple[str, ...],
    *,
    use_mode: str,
    scientific_role: str,
    availability_rule: str,
    revision_rule: str,
    direct: bool,
    reason: str,
) -> ResearchV2FieldAuthorization:
    return ResearchV2FieldAuthorization(
        fields=tuple(fields),
        use_mode=use_mode,
        scientific_role=scientific_role,
        availability_rule=availability_rule,
        revision_rule=revision_rule,
        direct_scientific_use_authorized=bool(direct),
        reason=reason,
    )


def _required_rules(pit_fingerprint_by_dataset: dict[str, str], pit_status_by_dataset: dict[str, str]) -> dict[str, ResearchV2DatasetScopeContract]:
    representation = get_adjusted_price_representation_contract()
    validate_adjusted_price_representation_contract(representation)
    invariant_ids = tuple(
        rule.representation_id
        for rule in representation.rules
        if rule.status == PRICE_REPRESENTATION_STATUS_INVARIANT
    )
    if not invariant_ids:
        raise ValueError("Research V2 adjusted-price required scope 缺 invariant representation")

    common_revision = (
        "consume only the pinned immutable Provider Snapshot; arbitrary provider historical corrections are not "
        "claimed PIT-invariant and therefore change the snapshot/scientific identity"
    )
    required = set(RESEARCH_V2_REQUIRED_DATASETS)
    common_complete = set(RESEARCH_V2_COMMON_COMPLETE_DATASETS)

    def contract(
        dataset: str,
        *,
        evidence_fields: tuple[str, ...],
        field_authorizations: tuple[ResearchV2FieldAuthorization, ...] = (),
        representation_ids: tuple[str, ...] = (),
        basis: str,
        reason: str,
    ) -> ResearchV2DatasetScopeContract:
        return ResearchV2DatasetScopeContract(
            dataset=dataset,
            required_for_generation=dataset in required,
            contributes_to_common_complete=dataset in common_complete,
            scope_status=SCOPE_STATUS_REQUIRED_AUTHORIZED,
            evidence_fields=tuple(evidence_fields),
            field_authorizations=tuple(field_authorizations),
            authorized_representation_ids=tuple(representation_ids),
            unlisted_fields_authorized=False,
            pit_contract_status=str(pit_status_by_dataset[dataset]),
            pit_contract_fingerprint=str(pit_fingerprint_by_dataset[dataset]),
            authorization_basis=basis,
            reason=reason,
        )

    return {
        "TaiwanStockTradingDate": contract(
            "TaiwanStockTradingDate",
            evidence_fields=("date",),
            basis="round9_exact_candidate_trading_calendar",
            reason="required Research calendar evidence; no model feature is authorized from this table",
        ),
        "TaiwanStockDelisting": contract(
            "TaiwanStockDelisting",
            evidence_fields=("date", "stock_id"),
            basis="round9_exact_event_universe_evidence",
            reason="required historical-universe event evidence; event no-row remains legal and is not a daily denominator",
        ),
        RESEARCH_V2_RAW_VOLUME_DATASET: contract(
            RESEARCH_V2_RAW_VOLUME_DATASET,
            evidence_fields=("date", "stock_id"),
            field_authorizations=(
                _field_rule(
                    (PROVIDER_VOLUME_FIELD,),
                    use_mode=FIELD_USE_DIRECT_INPUT,
                    scientific_role="canonical_daily_share_volume",
                    availability_rule="next_taiwan_trading_session_after_market_date",
                    revision_rule=common_revision,
                    direct=True,
                    reason=(
                        "FinMind documents Trading_Volume as the completed daily share quantity across TWSE/TPEX/emerging; "
                        "Research uses it only from the next Taiwan trading session. Raw OHLC from TaiwanStockPrice remains prohibited."
                    ),
                ),
            ),
            basis="round9_exact_raw_market_evidence_plus_round14_volume_field_authorization",
            reason="required for daily universe presence and canonical unadjusted share-volume input only",
        ),
        "TaiwanStockPriceLimit": contract(
            "TaiwanStockPriceLimit",
            evidence_fields=("date", "stock_id"),
            basis="round9_exact_daily_coverage_evidence",
            reason="required exact daily coverage evidence; price-limit payload fields are not model inputs in the foundation scope",
        ),
        RESEARCH_V2_ADJUSTED_PRICE_DATASET: contract(
            RESEARCH_V2_ADJUSTED_PRICE_DATASET,
            evidence_fields=("date", "stock_id"),
            field_authorizations=(
                _field_rule(
                    tuple(PROVIDER_PRICE_FIELDS),
                    use_mode=FIELD_USE_TRANSFORM_SOURCE_ONLY,
                    scientific_role="canonical_adjusted_price_invariant_representation_source",
                    availability_rule="next_taiwan_trading_session_after_market_date; matured supervision additionally obeys label_maturity_cutoff",
                    revision_rule=(
                        "direct raw/absolute adjusted-price levels are prohibited; only Round-13 invariant representations may consume OHLC; "
                        + common_revision
                    ),
                    direct=False,
                    reason="OHLC are authorized only as source operands of the explicit Round-13 scale-invariant representations",
                ),
            ),
            representation_ids=invariant_ids,
            basis="round13_adjusted_price_representation_invariance_proof_plus_round14_scope_authorization",
            reason=(
                "required canonical price source through invariant representations only; Trading_Volume is intentionally not authorized "
                "from this current-vintage table because Research V2 volume truth is TaiwanStockPrice.Trading_Volume"
            ),
        ),
    }


def build_research_v2_dataset_scope_contracts() -> tuple[ResearchV2DatasetScopeContract, ...]:
    specs = tuple(get_market_dataset_specs(included_only=True))
    pit_rows = tuple(build_research_v2_pit_review_contracts())
    pit_stats = validate_research_v2_pit_review_contracts(pit_rows)
    pit_by_dataset = {row.dataset: row for row in pit_rows}
    if len(pit_by_dataset) != len(pit_rows):
        raise ValueError("Research V2 scope PIT contract dataset identity 重複")
    pit_status_by_dataset = {name: row.pit_legality_status for name, row in pit_by_dataset.items()}
    pit_fingerprint_by_dataset = {
        name: canonical_json_sha256(research_v2_pit_review_contract_payloads((row,))[0])
        for name, row in pit_by_dataset.items()
    }
    required_rules = _required_rules(pit_fingerprint_by_dataset, pit_status_by_dataset)

    rows: list[ResearchV2DatasetScopeContract] = []
    for spec in specs:
        required = required_rules.get(spec.dataset)
        if required is not None:
            rows.append(required)
            continue
        pit = pit_by_dataset[spec.dataset]
        rows.append(
            ResearchV2DatasetScopeContract(
                dataset=spec.dataset,
                required_for_generation=False,
                contributes_to_common_complete=False,
                scope_status=SCOPE_STATUS_OPTIONAL_NOT_SELECTED,
                evidence_fields=(),
                field_authorizations=(),
                authorized_representation_ids=(),
                unlisted_fields_authorized=False,
                pit_contract_status=pit.pit_legality_status,
                pit_contract_fingerprint="",
                authorization_basis="archive_presence_does_not_imply_research_input_selection",
                reason=(
                    "archived for future experiment-specific Research decisions but not selected by the Research V2 foundation scope; "
                    "its current PIT review status therefore cannot block foundation freeze"
                ),
            )
        )
    validate_research_v2_dataset_scope_contracts(rows)
    return tuple(rows)


def research_v2_dataset_scope_contract_payloads(
    contracts: Iterable[ResearchV2DatasetScopeContract] | None = None,
) -> list[dict[str, object]]:
    rows = tuple(contracts or build_research_v2_dataset_scope_contracts())
    payloads: list[dict[str, object]] = []
    for row in rows:
        payload = asdict(row)
        payload["evidence_fields"] = list(row.evidence_fields)
        payload["authorized_representation_ids"] = list(row.authorized_representation_ids)
        payload["field_authorizations"] = [
            {
                **asdict(field_rule),
                "fields": list(field_rule.fields),
            }
            for field_rule in row.field_authorizations
        ]
        payloads.append(payload)
    return payloads


def research_v2_dataset_scope_contract_fingerprint(
    contracts: Iterable[ResearchV2DatasetScopeContract] | None = None,
) -> str:
    rows = tuple(contracts or build_research_v2_dataset_scope_contracts())
    required_rows = tuple(row for row in rows if row.required_for_generation)
    return canonical_json_sha256(
        {
            "schema_version": RESEARCH_V2_SCOPE_SCHEMA_VERSION,
            "contract_id": RESEARCH_V2_SCOPE_CONTRACT_ID,
            "required_dataset_scope": list(RESEARCH_V2_REQUIRED_DATASETS),
            "common_complete_dataset_scope": list(RESEARCH_V2_COMMON_COMPLETE_DATASETS),
            "required_contracts": research_v2_dataset_scope_contract_payloads(required_rows),
        }
    )


def validate_research_v2_dataset_scope_contracts(
    contracts: Iterable[ResearchV2DatasetScopeContract] | None = None,
) -> dict[str, object]:
    rows = tuple(contracts or build_research_v2_dataset_scope_contracts())
    specs = tuple(get_market_dataset_specs(included_only=True))
    expected_ids = {spec.dataset for spec in specs}
    names = [row.dataset for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("Research V2 dataset scope identity 重複")
    if set(names) != expected_ids:
        raise ValueError(
            "Research V2 dataset scope coverage drift: "
            f"missing={sorted(expected_ids - set(names))}, extra={sorted(set(names) - expected_ids)}"
        )

    required = tuple(row for row in rows if row.required_for_generation)
    required_names = tuple(row.dataset for row in required)
    if set(required_names) != set(RESEARCH_V2_REQUIRED_DATASETS):
        raise ValueError(
            "Research V2 required dataset scope drift: "
            f"actual={sorted(required_names)} expected={sorted(RESEARCH_V2_REQUIRED_DATASETS)}"
        )
    common_complete_names = {row.dataset for row in rows if row.contributes_to_common_complete}
    if common_complete_names != set(RESEARCH_V2_COMMON_COMPLETE_DATASETS):
        raise ValueError("Research V2 common-complete participant scope drift")
    if any(row.scope_status != SCOPE_STATUS_REQUIRED_AUTHORIZED for row in required):
        raise ValueError("Research V2 required dataset scope 必須全部取得明確 authorization")
    if any(row.unlisted_fields_authorized for row in rows):
        raise ValueError("Research V2 field authorization 必須 fail-closed；未列欄位不得自動合法")
    if any(not row.pit_contract_fingerprint for row in required):
        raise ValueError("Research V2 required dataset 必須 pin dataset-specific PIT contract identity")

    by_dataset = {row.dataset: row for row in rows}
    raw = by_dataset[RESEARCH_V2_RAW_VOLUME_DATASET]
    raw_direct_fields = {
        field
        for rule in raw.field_authorizations
        if rule.direct_scientific_use_authorized
        for field in rule.fields
    }
    if raw_direct_fields != {PROVIDER_VOLUME_FIELD}:
        raise ValueError("Research V2 TaiwanStockPrice direct scientific input 必須只授權 Trading_Volume")
    if set(PROVIDER_PRICE_FIELDS).intersection(raw_direct_fields):
        raise ValueError("Research V2 raw TaiwanStockPrice OHLC 不得成為 model price input")

    adjusted = by_dataset[RESEARCH_V2_ADJUSTED_PRICE_DATASET]
    adjusted_direct_fields = {
        field
        for rule in adjusted.field_authorizations
        if rule.direct_scientific_use_authorized
        for field in rule.fields
    }
    if adjusted_direct_fields:
        raise ValueError("Research V2 adjusted-price raw fields 不得 direct scientific use")
    if any(PROVIDER_VOLUME_FIELD in rule.fields for rule in adjusted.field_authorizations):
        raise ValueError("Research V2 volume truth 不得從 TaiwanStockPriceAdj current-vintage field 授權")
    representation = get_adjusted_price_representation_contract()
    invariant_ids = {
        rule.representation_id
        for rule in representation.rules
        if rule.status == PRICE_REPRESENTATION_STATUS_INVARIANT
    }
    if set(adjusted.authorized_representation_ids) != invariant_ids:
        raise ValueError("Research V2 adjusted-price authorized representation set 必須精確等於 Round-13 invariant set")

    optional = tuple(row for row in rows if not row.required_for_generation)
    if any(row.scope_status != SCOPE_STATUS_OPTIONAL_NOT_SELECTED for row in optional):
        raise ValueError("Research V2 optional archive dataset 不得被 foundation scope 隱式選入")
    if any(row.field_authorizations or row.authorized_representation_ids or row.evidence_fields for row in optional):
        raise ValueError("Research V2 optional-not-selected dataset 不得取得隱式 field/model authorization")
    if any(row.pit_contract_fingerprint for row in optional):
        raise ValueError("Research V2 optional-not-selected PIT diagnostics 不得污染 foundation scientific fingerprint")

    direct_input_dataset_count = sum(
        any(rule.direct_scientific_use_authorized for rule in row.field_authorizations)
        for row in rows
    )
    representation_input_dataset_count = sum(bool(row.authorized_representation_ids) for row in rows)
    return {
        "schema_version": RESEARCH_V2_SCOPE_SCHEMA_VERSION,
        "contract_id": RESEARCH_V2_SCOPE_CONTRACT_ID,
        "dataset_count": len(rows),
        "required_dataset_count": len(required),
        "optional_not_selected_count": len(optional),
        "required_dataset_scope": list(RESEARCH_V2_REQUIRED_DATASETS),
        "common_complete_dataset_scope": list(RESEARCH_V2_COMMON_COMPLETE_DATASETS),
        "direct_input_dataset_count": direct_input_dataset_count,
        "representation_input_dataset_count": representation_input_dataset_count,
        "required_scope_authorization_complete": True,
        "contract_fingerprint": research_v2_dataset_scope_contract_fingerprint(rows),
    }


__all__ = [
    "RESEARCH_V2_SCOPE_SCHEMA_VERSION",
    "RESEARCH_V2_SCOPE_CONTRACT_ID",
    "SCOPE_STATUS_REQUIRED_AUTHORIZED",
    "SCOPE_STATUS_OPTIONAL_NOT_SELECTED",
    "FIELD_USE_EVIDENCE_ONLY",
    "FIELD_USE_DIRECT_INPUT",
    "FIELD_USE_TRANSFORM_SOURCE_ONLY",
    "RESEARCH_V2_REQUIRED_DATASETS",
    "RESEARCH_V2_COMMON_COMPLETE_DATASETS",
    "RESEARCH_V2_RAW_VOLUME_DATASET",
    "RESEARCH_V2_ADJUSTED_PRICE_DATASET",
    "ResearchV2FieldAuthorization",
    "ResearchV2DatasetScopeContract",
    "build_research_v2_dataset_scope_contracts",
    "research_v2_dataset_scope_contract_payloads",
    "research_v2_dataset_scope_contract_fingerprint",
    "validate_research_v2_dataset_scope_contracts",
]
