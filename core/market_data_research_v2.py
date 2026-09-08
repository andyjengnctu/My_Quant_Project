"""Research V2 market-data candidate contracts and PIT/completeness primitives.

This layer deliberately stops short of Research V2 promotion. Provider-facing
``pit_class`` values are review inputs, not blanket scientific approval. The
archive-wide exact/date-presence audits remain diagnostics; the narrower
Research V2 required dataset/field scope is authorized separately by
``core.market_data_research_scope`` and is what later common-complete/freeze
logic must consume.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

import pandas as pd

from core.file_integrity import canonical_json_sha256
from core.market_data_research_freeze import ResearchV2RequiredCommonCompleteSummary
from core.market_data_dataset_registry import (
    PIT_ARCHIVE_ONLY,
    PIT_CURRENT_VINTAGE,
    PIT_EXACT_CANDIDATE,
    PIT_REVIEW_REQUIRED,
    MarketDatasetSpec,
    get_market_dataset_specs,
)

RESEARCH_V2_CANDIDATE_SCHEMA_VERSION = 7
RESEARCH_V2_CANDIDATE_STATUS_NOT_READY = "CANDIDATE_NOT_READY"
RESEARCH_V2_DATASET_STATUS_EXACT_CANDIDATE = "EXACT_CANDIDATE"
RESEARCH_V2_DATASET_STATUS_REVIEW_REQUIRED = "REVIEW_REQUIRED"
RESEARCH_V2_DATASET_STATUS_CURRENT_VINTAGE_BLOCKED = "CURRENT_VINTAGE_BLOCKED"
RESEARCH_V2_DATASET_STATUS_ARCHIVE_ONLY = "ARCHIVE_ONLY"

RESEARCH_V2_DAILY_UNIVERSE_SOURCE_DATASET = "TaiwanStockPrice"
RESEARCH_V2_TRADING_CALENDAR_DATASET = "TaiwanStockTradingDate"
RESEARCH_V2_DAILY_COVERAGE_DATASET = "TaiwanStockPriceLimit"
RESEARCH_V2_EVENT_EVIDENCE_DATASET = "TaiwanStockDelisting"
RESEARCH_V2_ADJUSTED_PRICE_DATASET = "TaiwanStockPriceAdj"

RESEARCH_V2_EXACT_AUDIT_DATASETS = (
    RESEARCH_V2_TRADING_CALENDAR_DATASET,
    RESEARCH_V2_EVENT_EVIDENCE_DATASET,
    RESEARCH_V2_DAILY_UNIVERSE_SOURCE_DATASET,
    RESEARCH_V2_DAILY_COVERAGE_DATASET,
)


RESEARCH_V2_CANDIDATE_IDENTITY_FIELDS = (
    "schema_version",
    "generation_id",
    "status",
    "provider_snapshot_fingerprint",
    "provider_manifest_fingerprint",
    "provider_as_of_date",
    "required_cutoff",
    "historical_instrument_count",
    "daily_universe_source_dataset",
    "daily_universe_fingerprint",
    "exact_audit_datasets",
    "exact_candidate_ceiling_date",
    "latest_complete_tail_start",
    "latest_complete_tail_date_count",
    "exact_coverage_fingerprint",
    "adjusted_price_representation_contract_fingerprint",
    "research_scope_contract_fingerprint",
    "required_dataset_scope",
    "required_common_complete_start_date",
    "required_common_complete_tail_date_count",
    "required_common_complete_fingerprint",
    "research_common_complete_cutoff",
    "frozen_cutoff",
    "active_research_generation_changed",
)


@dataclass(frozen=True)
class ResearchV2DatasetAssessment:
    dataset: str
    pit_class: str
    candidate_status: str
    automatically_usable_for_pit_audit: bool
    reason: str


@dataclass(frozen=True)
class ResearchV2ExactCoverageSummary:
    provider_as_of_date: str
    trading_date_count: int
    daily_universe_date_count: int
    daily_universe_row_count: int
    exact_complete_date_count: int
    incomplete_date_count: int
    first_universe_date: str | None
    latest_universe_date: str | None
    latest_exact_complete_date: str | None
    latest_complete_tail_start: str | None
    latest_complete_tail_date_count: int
    coverage_fingerprint: str


def build_research_v2_dataset_assessments(
    specs: Iterable[MarketDatasetSpec] | None = None,
) -> tuple[ResearchV2DatasetAssessment, ...]:
    rows: list[ResearchV2DatasetAssessment] = []
    for spec in tuple(specs or get_market_dataset_specs(included_only=True)):
        if spec.pit_class == PIT_EXACT_CANDIDATE:
            status = RESEARCH_V2_DATASET_STATUS_EXACT_CANDIDATE
            automatic = True
            reason = "provider registry marks exact-date candidate; Research audit must still prove completeness"
        elif spec.pit_class == PIT_CURRENT_VINTAGE:
            status = RESEARCH_V2_DATASET_STATUS_CURRENT_VINTAGE_BLOCKED
            automatic = False
            reason = "current-vintage representation is not automatically PIT-legal for retrospective Research"
        elif spec.pit_class == PIT_REVIEW_REQUIRED:
            status = RESEARCH_V2_DATASET_STATUS_REVIEW_REQUIRED
            automatic = False
            reason = "dataset requires dataset-specific publication/PIT review before Research use"
        elif spec.pit_class == PIT_ARCHIVE_ONLY:
            status = RESEARCH_V2_DATASET_STATUS_ARCHIVE_ONLY
            automatic = False
            reason = "archive-only dataset is not a Research input candidate"
        else:
            raise ValueError(f"未知 Market Data pit_class: {spec.dataset} -> {spec.pit_class}")
        rows.append(
            ResearchV2DatasetAssessment(
                dataset=spec.dataset,
                pit_class=spec.pit_class,
                candidate_status=status,
                automatically_usable_for_pit_audit=automatic,
                reason=reason,
            )
        )
    names = [row.dataset for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("Research V2 dataset assessment 含重複 dataset")
    return tuple(rows)


def research_v2_dataset_assessment_fingerprint(
    assessments: Iterable[ResearchV2DatasetAssessment] | None = None,
) -> str:
    rows = tuple(assessments or build_research_v2_dataset_assessments())
    return canonical_json_sha256([asdict(row) for row in rows])


def validate_research_v2_candidate_contract() -> dict[str, int | str]:
    rows = build_research_v2_dataset_assessments()
    exact = [row for row in rows if row.candidate_status == RESEARCH_V2_DATASET_STATUS_EXACT_CANDIDATE]
    review = [row for row in rows if row.candidate_status == RESEARCH_V2_DATASET_STATUS_REVIEW_REQUIRED]
    current = [row for row in rows if row.candidate_status == RESEARCH_V2_DATASET_STATUS_CURRENT_VINTAGE_BLOCKED]
    archive = [row for row in rows if row.candidate_status == RESEARCH_V2_DATASET_STATUS_ARCHIVE_ONLY]
    exact_names = {row.dataset for row in exact}
    if exact_names != set(RESEARCH_V2_EXACT_AUDIT_DATASETS):
        raise ValueError(
            "Research V2 exact-candidate audit dataset set 與 provider registry drift: "
            f"actual={sorted(exact_names)} expected={sorted(RESEARCH_V2_EXACT_AUDIT_DATASETS)}"
        )
    adjusted = next((row for row in rows if row.dataset == RESEARCH_V2_ADJUSTED_PRICE_DATASET), None)
    if adjusted is None or adjusted.candidate_status != RESEARCH_V2_DATASET_STATUS_CURRENT_VINTAGE_BLOCKED:
        raise ValueError("Research V2 必須對 current-vintage adjusted price fail-closed")
    return {
        "dataset_count": len(rows),
        "exact_candidate_count": len(exact),
        "review_required_count": len(review),
        "current_vintage_blocked_count": len(current),
        "archive_only_count": len(archive),
        "assessment_fingerprint": research_v2_dataset_assessment_fingerprint(rows),
    }


def validate_research_v2_required_cutoff(*, provider_as_of_date: str, required_cutoff: str) -> str:
    provider_date = pd.Timestamp(str(provider_as_of_date)).date()
    cutoff_date = pd.Timestamp(str(required_cutoff)).date()
    if provider_date < cutoff_date:
        raise ValueError(
            "Research V2 Provider Snapshot as-of 早於 required cutoff: "
            f"provider_as_of={provider_date.isoformat()} required_cutoff={cutoff_date.isoformat()}"
        )
    return cutoff_date.isoformat()


def _normalize_date_series(values) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d")


def build_daily_pit_universe(
    raw_price_rows: pd.DataFrame,
    *,
    historical_instruments: Iterable[str],
    trading_dates: Iterable[str] | None = None,
    provider_as_of_date: str | None = None,
) -> pd.DataFrame:
    required = {"date", "stock_id"}
    missing = required.difference(raw_price_rows.columns)
    if missing:
        raise ValueError(f"TaiwanStockPrice PIT universe evidence 缺少欄位: {sorted(missing)}")
    pool = {str(value or "").strip() for value in historical_instruments if str(value or "").strip()}
    if not pool:
        raise ValueError("Research V2 historical instrument pool 不可為空")
    frame = raw_price_rows.loc[:, ["date", "stock_id"]].copy()
    frame["date"] = _normalize_date_series(frame["date"])
    frame["stock_id"] = frame["stock_id"].astype(str).str.strip()
    frame = frame.loc[frame["date"].notna() & frame["stock_id"].isin(pool)]
    if provider_as_of_date:
        frame = frame.loc[frame["date"] <= str(provider_as_of_date)]
    if trading_dates is not None:
        allowed_dates = {str(value) for value in trading_dates}
        frame = frame.loc[frame["date"].isin(allowed_dates)]
    frame = frame.drop_duplicates(["date", "stock_id"]).sort_values(["date", "stock_id"], kind="stable")
    return frame.reset_index(drop=True)



def summarize_exact_candidate_coverage_table(
    coverage: pd.DataFrame,
    *,
    provider_as_of_date: str,
    daily_universe_row_count: int,
) -> ResearchV2ExactCoverageSummary:
    required = {
        "date",
        "universe_count",
        "price_limit_count",
        "missing_price_limit_count",
        "extra_price_limit_count",
        "exact_complete",
    }
    missing = required.difference(coverage.columns)
    if missing:
        raise ValueError(f"exact coverage table 缺少欄位: {sorted(missing)}")
    normalized = coverage.loc[:, [
        "date",
        "universe_count",
        "price_limit_count",
        "missing_price_limit_count",
        "extra_price_limit_count",
        "exact_complete",
    ]].copy()
    normalized["date"] = _normalize_date_series(normalized["date"])
    normalized = normalized.loc[normalized["date"].notna() & (normalized["date"] <= str(provider_as_of_date))]
    normalized = normalized.sort_values("date", kind="stable").reset_index(drop=True)
    universe_dates = normalized.loc[normalized["universe_count"] > 0, "date"].tolist()
    complete_dates = normalized.loc[normalized["exact_complete"].astype(bool), "date"].tolist()
    latest_complete = complete_dates[-1] if complete_dates else None
    tail_start = None
    tail_count = 0
    if latest_complete is not None:
        upto = normalized.loc[normalized["date"] <= latest_complete].reset_index(drop=True)
        cursor = len(upto) - 1
        while cursor >= 0 and bool(upto.loc[cursor, "exact_complete"]):
            tail_start = str(upto.loc[cursor, "date"])
            tail_count += 1
            cursor -= 1
    logical_rows = normalized.to_dict(orient="records")
    return ResearchV2ExactCoverageSummary(
        provider_as_of_date=str(provider_as_of_date),
        trading_date_count=int(len(normalized)),
        daily_universe_date_count=int(len(universe_dates)),
        daily_universe_row_count=int(daily_universe_row_count),
        exact_complete_date_count=int(normalized["exact_complete"].astype(bool).sum()) if not normalized.empty else 0,
        incomplete_date_count=int((~normalized["exact_complete"].astype(bool)).sum()) if not normalized.empty else 0,
        first_universe_date=universe_dates[0] if universe_dates else None,
        latest_universe_date=universe_dates[-1] if universe_dates else None,
        latest_exact_complete_date=latest_complete,
        latest_complete_tail_start=tail_start,
        latest_complete_tail_date_count=tail_count,
        coverage_fingerprint=canonical_json_sha256(logical_rows),
    )

def build_exact_candidate_daily_coverage(
    *,
    trading_dates: Iterable[str],
    daily_universe: pd.DataFrame,
    price_limit_rows: pd.DataFrame,
    provider_as_of_date: str,
) -> tuple[pd.DataFrame, ResearchV2ExactCoverageSummary]:
    if set(daily_universe.columns) != {"date", "stock_id"}:
        if not {"date", "stock_id"}.issubset(daily_universe.columns):
            raise ValueError("daily_universe 必須包含 date / stock_id")
        universe = daily_universe.loc[:, ["date", "stock_id"]].copy()
    else:
        universe = daily_universe.copy()
    required = {"date", "stock_id"}
    missing = required.difference(price_limit_rows.columns)
    if missing:
        raise ValueError(f"TaiwanStockPriceLimit completeness evidence 缺少欄位: {sorted(missing)}")

    calendar = sorted(
        {
            str(value)
            for value in trading_dates
            if str(value) and str(value) <= str(provider_as_of_date)
        }
    )
    universe["date"] = _normalize_date_series(universe["date"])
    universe["stock_id"] = universe["stock_id"].astype(str).str.strip()
    universe = universe.loc[universe["date"].isin(calendar)].drop_duplicates(["date", "stock_id"])

    limits = price_limit_rows.loc[:, ["date", "stock_id"]].copy()
    limits["date"] = _normalize_date_series(limits["date"])
    limits["stock_id"] = limits["stock_id"].astype(str).str.strip()
    limits = limits.loc[limits["date"].isin(calendar)].drop_duplicates(["date", "stock_id"])

    universe_groups = {date: set(group["stock_id"]) for date, group in universe.groupby("date", sort=False)}
    limit_groups = {date: set(group["stock_id"]) for date, group in limits.groupby("date", sort=False)}
    rows: list[dict[str, object]] = []
    for date_value in calendar:
        members = universe_groups.get(date_value, set())
        covered = limit_groups.get(date_value, set())
        missing_members = members.difference(covered)
        extra_members = covered.difference(members)
        complete = bool(members) and not missing_members
        rows.append(
            {
                "date": date_value,
                "universe_count": len(members),
                "price_limit_count": len(covered),
                "missing_price_limit_count": len(missing_members),
                "extra_price_limit_count": len(extra_members),
                "exact_complete": bool(complete),
            }
        )
    coverage = pd.DataFrame(rows)
    summary = summarize_exact_candidate_coverage_table(
        coverage,
        provider_as_of_date=str(provider_as_of_date),
        daily_universe_row_count=int(len(universe)),
    )
    return coverage, summary


def build_research_v2_candidate_identity_payload(
    *,
    provider_snapshot_fingerprint: str,
    provider_manifest_fingerprint: str,
    provider_as_of_date: str,
    required_cutoff: str,
    historical_instrument_count: int,
    daily_universe_fingerprint: str,
    coverage_summary: ResearchV2ExactCoverageSummary,
    adjusted_price_representation_contract_fingerprint: str,
    research_scope_contract_fingerprint: str,
    required_dataset_scope: Iterable[str],
    required_common_complete_summary: ResearchV2RequiredCommonCompleteSummary,
) -> dict[str, object]:
    return {
        "schema_version": RESEARCH_V2_CANDIDATE_SCHEMA_VERSION,
        "generation_id": "research_v2",
        "status": RESEARCH_V2_CANDIDATE_STATUS_NOT_READY,
        "provider_snapshot_fingerprint": str(provider_snapshot_fingerprint),
        "provider_manifest_fingerprint": str(provider_manifest_fingerprint),
        "provider_as_of_date": str(provider_as_of_date),
        "required_cutoff": str(required_cutoff),
        "historical_instrument_count": int(historical_instrument_count),
        "daily_universe_source_dataset": RESEARCH_V2_DAILY_UNIVERSE_SOURCE_DATASET,
        "daily_universe_fingerprint": str(daily_universe_fingerprint),
        "exact_audit_datasets": list(RESEARCH_V2_EXACT_AUDIT_DATASETS),
        "exact_candidate_ceiling_date": coverage_summary.latest_exact_complete_date,
        "latest_complete_tail_start": coverage_summary.latest_complete_tail_start,
        "latest_complete_tail_date_count": int(coverage_summary.latest_complete_tail_date_count),
        "exact_coverage_fingerprint": coverage_summary.coverage_fingerprint,
        "adjusted_price_representation_contract_fingerprint": str(adjusted_price_representation_contract_fingerprint),
        "research_scope_contract_fingerprint": str(research_scope_contract_fingerprint),
        "required_dataset_scope": [str(value) for value in required_dataset_scope],
        "required_common_complete_start_date": required_common_complete_summary.common_complete_tail_start,
        "required_common_complete_tail_date_count": int(required_common_complete_summary.common_complete_tail_date_count),
        "required_common_complete_fingerprint": str(required_common_complete_summary.coverage_fingerprint),
        "research_common_complete_cutoff": required_common_complete_summary.common_complete_cutoff,
        "frozen_cutoff": None,
        "active_research_generation_changed": False,
    }


__all__ = [
    "RESEARCH_V2_CANDIDATE_SCHEMA_VERSION",
    "RESEARCH_V2_CANDIDATE_STATUS_NOT_READY",
    "RESEARCH_V2_DATASET_STATUS_EXACT_CANDIDATE",
    "RESEARCH_V2_DATASET_STATUS_REVIEW_REQUIRED",
    "RESEARCH_V2_DATASET_STATUS_CURRENT_VINTAGE_BLOCKED",
    "RESEARCH_V2_DATASET_STATUS_ARCHIVE_ONLY",
    "RESEARCH_V2_DAILY_UNIVERSE_SOURCE_DATASET",
    "RESEARCH_V2_TRADING_CALENDAR_DATASET",
    "RESEARCH_V2_DAILY_COVERAGE_DATASET",
    "RESEARCH_V2_EVENT_EVIDENCE_DATASET",
    "RESEARCH_V2_ADJUSTED_PRICE_DATASET",
    "RESEARCH_V2_EXACT_AUDIT_DATASETS",
    "RESEARCH_V2_CANDIDATE_IDENTITY_FIELDS",
    "ResearchV2DatasetAssessment",
    "ResearchV2ExactCoverageSummary",
    "build_research_v2_dataset_assessments",
    "research_v2_dataset_assessment_fingerprint",
    "validate_research_v2_candidate_contract",
    "validate_research_v2_required_cutoff",
    "build_daily_pit_universe",
    "summarize_exact_candidate_coverage_table",
    "build_exact_candidate_daily_coverage",
    "build_research_v2_candidate_identity_payload",
]
