"""Research-only adjusted-price representation invariance contract.

FinMind ``TaiwanStockPriceAdj`` is a current-vintage, backward-adjusted price
series.  The provider documents corporate-action adjustment as a backward
cumulative factor from the latest trading day: the event-day adjusted price
remains the raw event-day price while earlier history is restated.

This module does *not* authorize the raw adjusted-price table as a Research
input.  It records the narrower mathematical result needed by Research V2:
which price representations are invariant to a later positive scalar restatement
of an already-mature historical prefix, and which fields/uses remain blocked.
Arbitrary provider data corrections or historical algorithm rebuilds are not
claimed to be invariant; immutable Provider Snapshot identity remains required.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from core.file_integrity import canonical_json_sha256
from core.market_data_contract import FINMIND_ADJUSTED_PRICE_DATASET

ADJUSTED_PRICE_REPRESENTATION_SCHEMA_VERSION = 1
ADJUSTED_PRICE_REPRESENTATION_CONTRACT_ID = "finmind_adjusted_price_relative_scale_invariance_v1"
ADJUSTED_PRICE_RESTATEMENT_MODEL = "documented_positive_scalar_backward_prefix_restatement"
ADJUSTED_PRICE_PROOF_STATUS = "CORPORATE_ACTION_RESTATEMENT_INVARIANCE_PROVEN"
ADJUSTED_PRICE_VENDOR_CORRECTION_STATUS = "NOT_CLAIMED_INVARIANT_SNAPSHOT_PIN_REQUIRED"
ADJUSTED_PRICE_SCOPE_STATUS = "SCIENTIFIC_SCOPE_AUTHORIZATION_PENDING"

PRICE_REPRESENTATION_STATUS_INVARIANT = "INVARIANT_UNDER_DOCUMENTED_PREFIX_SCALE_RESTATEMENT"
PRICE_REPRESENTATION_STATUS_BLOCKED = "BLOCKED_NOT_SCALE_INVARIANT"
PRICE_REPRESENTATION_STATUS_DEFERRED = "DEFERRED_OUTSIDE_PRICE_RESTATEMENT_PROOF"

PROVIDER_PRICE_FIELDS = ("open", "max", "min", "close")
CANONICAL_PRICE_FIELDS = ("Open", "High", "Low", "Close")
PROVIDER_VOLUME_FIELD = "Trading_Volume"


@dataclass(frozen=True)
class AdjustedPriceRepresentationRule:
    representation_id: str
    source_fields: tuple[str, ...]
    transform_rule: str
    scientific_role: str
    status: str
    maturity_rule: str
    direct_scientific_use_authorized: bool
    reason: str


@dataclass(frozen=True)
class AdjustedPriceRepresentationContract:
    schema_version: int
    contract_id: str
    dataset: str
    provider_evidence_checked_date: str
    provider_evidence_source: str
    restatement_model: str
    proof_status: str
    vendor_correction_status: str
    scope_status: str
    provider_price_fields: tuple[str, ...]
    canonical_price_fields: tuple[str, ...]
    provider_volume_field: str
    raw_absolute_price_levels_authorized: bool
    scientific_input_authorized: bool
    rules: tuple[AdjustedPriceRepresentationRule, ...]


_RULES: tuple[AdjustedPriceRepresentationRule, ...] = (
    AdjustedPriceRepresentationRule(
        representation_id="predecision_price_order_relations_v1",
        source_fields=CANONICAL_PRICE_FIELDS,
        transform_rule="comparisons_and_rolling_extrema_using_only_rows_at_or_before_decision_time",
        scientific_role="candidate_membership_and_predecision_price_geometry",
        status=PRICE_REPRESENTATION_STATUS_INVARIANT,
        maturity_rule="decision_time_only; all consumed price rows must be at_or_before_decision_time",
        direct_scientific_use_authorized=False,
        reason="positive common scaling preserves ordering, crossover truth, and rolling extrema relations",
    ),
    AdjustedPriceRepresentationRule(
        representation_id="anchor_relative_ohlc_v1",
        source_fields=CANONICAL_PRICE_FIELDS,
        transform_rule="price_t / decision_anchor_close - 1",
        scientific_role="normalized_historical_price_sequence",
        status=PRICE_REPRESENTATION_STATUS_INVARIANT,
        maturity_rule="decision_time_only; numerator and anchor belong to the same instrument prefix",
        direct_scientific_use_authorized=False,
        reason="a later positive scalar applied to both numerator and anchor cancels exactly",
    ),
    AdjustedPriceRepresentationRule(
        representation_id="relative_price_context_v1",
        source_fields=("High", "Close", "breakout_level"),
        transform_rule="ratios among same-prefix price levels, including level/close, close/level, and high/level",
        scientific_role="scale_free_price_context",
        status=PRICE_REPRESENTATION_STATUS_INVARIANT,
        maturity_rule="decision_time_only; every price operand must share the same historical prefix restatement",
        direct_scientific_use_authorized=False,
        reason="all ratio operands receive the same positive scalar and therefore cancel",
    ),
    AdjustedPriceRepresentationRule(
        representation_id="matured_anchor_relative_future_path_v1",
        source_fields=("High", "Low", "Close"),
        transform_rule="matured future high/low path divided by decision anchor; barriers and path labels are ratio-derived",
        scientific_role="matured_supervision_path_and_ratio_target",
        status=PRICE_REPRESENTATION_STATUS_INVARIANT,
        maturity_rule="label maturity must be at_or_before Research information cutoff before the sample may train",
        direct_scientific_use_authorized=False,
        reason=(
            "after label maturity, any later documented corporate-action restatement multiplies the anchor "
            "and every matured path price by the same positive scalar; returns, barrier ordering, and hit bar remain unchanged"
        ),
    ),
    AdjustedPriceRepresentationRule(
        representation_id="absolute_adjusted_price_level_v1",
        source_fields=CANONICAL_PRICE_FIELDS,
        transform_rule="raw adjusted OHLC or derived absolute price levels",
        scientific_role="absolute_price_level",
        status=PRICE_REPRESENTATION_STATUS_BLOCKED,
        maturity_rule="not_applicable",
        direct_scientific_use_authorized=False,
        reason="absolute levels change when a later corporate action rescales the historical prefix",
    ),
    AdjustedPriceRepresentationRule(
        representation_id="adjusted_dataset_volume_field_v1",
        source_fields=("Volume",),
        transform_rule="volume feature semantics",
        scientific_role="volume_input",
        status=PRICE_REPRESENTATION_STATUS_DEFERRED,
        maturity_rule="requires independent field-level PIT/publication/revision authorization",
        direct_scientific_use_authorized=False,
        reason="price-scale invariance proves nothing about historical volume revisions or publication semantics",
    ),
)


def get_adjusted_price_representation_contract() -> AdjustedPriceRepresentationContract:
    return AdjustedPriceRepresentationContract(
        schema_version=ADJUSTED_PRICE_REPRESENTATION_SCHEMA_VERSION,
        contract_id=ADJUSTED_PRICE_REPRESENTATION_CONTRACT_ID,
        dataset=FINMIND_ADJUSTED_PRICE_DATASET,
        provider_evidence_checked_date="2026-09-08",
        provider_evidence_source=(
            "FinMind Technical/TaiwanStockPriceAdj: latest-trading-day base with backward cumulative adjustment factors; "
            "event-day adjusted price equals raw event-day price and adjustment is reflected in earlier history"
        ),
        restatement_model=ADJUSTED_PRICE_RESTATEMENT_MODEL,
        proof_status=ADJUSTED_PRICE_PROOF_STATUS,
        vendor_correction_status=ADJUSTED_PRICE_VENDOR_CORRECTION_STATUS,
        scope_status=ADJUSTED_PRICE_SCOPE_STATUS,
        provider_price_fields=PROVIDER_PRICE_FIELDS,
        canonical_price_fields=CANONICAL_PRICE_FIELDS,
        provider_volume_field=PROVIDER_VOLUME_FIELD,
        raw_absolute_price_levels_authorized=False,
        scientific_input_authorized=False,
        rules=_RULES,
    )


def adjusted_price_representation_contract_payload(
    contract: AdjustedPriceRepresentationContract | None = None,
) -> dict[str, object]:
    row = contract or get_adjusted_price_representation_contract()
    payload = asdict(row)
    payload["provider_price_fields"] = list(row.provider_price_fields)
    payload["canonical_price_fields"] = list(row.canonical_price_fields)
    payload["rules"] = [
        {
            **asdict(rule),
            "source_fields": list(rule.source_fields),
        }
        for rule in row.rules
    ]
    return payload


def adjusted_price_representation_contract_fingerprint(
    contract: AdjustedPriceRepresentationContract | None = None,
) -> str:
    return canonical_json_sha256(adjusted_price_representation_contract_payload(contract))


def validate_adjusted_price_representation_contract(
    contract: AdjustedPriceRepresentationContract | None = None,
) -> dict[str, object]:
    row = contract or get_adjusted_price_representation_contract()
    if row.dataset != FINMIND_ADJUSTED_PRICE_DATASET:
        raise ValueError("adjusted-price representation contract dataset identity drift")
    if row.restatement_model != ADJUSTED_PRICE_RESTATEMENT_MODEL:
        raise ValueError("adjusted-price restatement model drift")
    if row.proof_status != ADJUSTED_PRICE_PROOF_STATUS:
        raise ValueError("adjusted-price representation proof status drift")
    if row.raw_absolute_price_levels_authorized or row.scientific_input_authorized:
        raise ValueError("Round 13 不得授權 raw adjusted-price levels 或 Research dataset scope")
    if tuple(row.provider_price_fields) != PROVIDER_PRICE_FIELDS or tuple(row.canonical_price_fields) != CANONICAL_PRICE_FIELDS:
        raise ValueError("adjusted-price proof field scope drift")

    rules = tuple(row.rules)
    ids = [item.representation_id for item in rules]
    if len(ids) != len(set(ids)):
        raise ValueError("adjusted-price representation rule identity 重複")
    if any(item.direct_scientific_use_authorized for item in rules):
        raise ValueError("Round 13 representation proof 不得自行授權 scientific input")

    invariant = tuple(item for item in rules if item.status == PRICE_REPRESENTATION_STATUS_INVARIANT)
    blocked = tuple(item for item in rules if item.status == PRICE_REPRESENTATION_STATUS_BLOCKED)
    deferred = tuple(item for item in rules if item.status == PRICE_REPRESENTATION_STATUS_DEFERRED)
    required_ids = {
        "predecision_price_order_relations_v1",
        "anchor_relative_ohlc_v1",
        "relative_price_context_v1",
        "matured_anchor_relative_future_path_v1",
        "absolute_adjusted_price_level_v1",
        "adjusted_dataset_volume_field_v1",
    }
    if set(ids) != required_ids:
        raise ValueError(
            "adjusted-price representation rule coverage drift: "
            f"missing={sorted(required_ids - set(ids))}, extra={sorted(set(ids) - required_ids)}"
        )
    return {
        "schema_version": int(row.schema_version),
        "rule_count": len(rules),
        "invariant_rule_count": len(invariant),
        "blocked_rule_count": len(blocked),
        "deferred_rule_count": len(deferred),
        "proof_status": row.proof_status,
        "vendor_correction_status": row.vendor_correction_status,
        "scientific_input_authorized": bool(row.scientific_input_authorized),
        "contract_fingerprint": adjusted_price_representation_contract_fingerprint(row),
    }


__all__ = [
    "ADJUSTED_PRICE_REPRESENTATION_SCHEMA_VERSION",
    "ADJUSTED_PRICE_REPRESENTATION_CONTRACT_ID",
    "ADJUSTED_PRICE_RESTATEMENT_MODEL",
    "ADJUSTED_PRICE_PROOF_STATUS",
    "ADJUSTED_PRICE_VENDOR_CORRECTION_STATUS",
    "ADJUSTED_PRICE_SCOPE_STATUS",
    "PRICE_REPRESENTATION_STATUS_INVARIANT",
    "PRICE_REPRESENTATION_STATUS_BLOCKED",
    "PRICE_REPRESENTATION_STATUS_DEFERRED",
    "PROVIDER_PRICE_FIELDS",
    "CANONICAL_PRICE_FIELDS",
    "PROVIDER_VOLUME_FIELD",
    "AdjustedPriceRepresentationRule",
    "AdjustedPriceRepresentationContract",
    "get_adjusted_price_representation_contract",
    "adjusted_price_representation_contract_payload",
    "adjusted_price_representation_contract_fingerprint",
    "validate_adjusted_price_representation_contract",
]
