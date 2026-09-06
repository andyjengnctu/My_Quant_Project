"""Pure bootstrap request planning for the Market Data V2 archive."""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterable, Mapping

from core.market_data_dataset_registry import (
    BOOTSTRAP_BULK_REFERENCE_DATES,
    BOOTSTRAP_FIXED_DATA_ID_FULL_RANGE,
    BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE,
    BOOTSTRAP_SINGLE_FULL_RANGE,
    BOOTSTRAP_SINGLE_NO_DATES,
    MarketDatasetSpec,
)


@dataclass(frozen=True)
class DatasetProbeEvidence:
    dataset: str
    status: str
    request_count: int
    row_count: int
    columns: tuple[str, ...]
    observed_dates: tuple[str, ...] = ()
    earliest_date: str | None = None
    latest_date: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class BootstrapPlanRow:
    dataset: str
    category: str
    bootstrap_mode: str
    request_count: int
    basis: str


@dataclass(frozen=True)
class BootstrapRequestPlan:
    as_of_date: str
    historical_instrument_count: int
    included_dataset_count: int
    per_instrument_dataset_count: int
    total_requests: int
    quota_limit: int
    minimum_quota_hours: float
    minimum_quota_windows: int
    rows: tuple[BootstrapPlanRow, ...]


def _normalize_instruments(instruments: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(sorted({str(item or "").strip() for item in instruments if str(item or "").strip()}))
    if not normalized:
        raise ValueError("Historical instrument universe 不可為空")
    return normalized


def _require_probe(evidence_by_dataset: Mapping[str, DatasetProbeEvidence], dataset: str) -> DatasetProbeEvidence:
    evidence = evidence_by_dataset.get(dataset)
    if evidence is None:
        raise ValueError(f"缺少 dataset preflight evidence: {dataset}")
    if evidence.status != "PASS":
        raise ValueError(f"dataset preflight 尚未 PASS: {dataset} | {evidence.error or evidence.status}")
    return evidence


def build_bootstrap_request_plan(
    *,
    specs: Iterable[MarketDatasetSpec],
    historical_instruments: Iterable[str],
    evidence_by_dataset: Mapping[str, DatasetProbeEvidence],
    as_of_date: str,
    quota_limit: int,
) -> BootstrapRequestPlan:
    instruments = _normalize_instruments(historical_instruments)
    limit = int(quota_limit)
    if limit <= 0:
        raise ValueError(f"api_request_limit 必須 > 0: {quota_limit!r}")
    as_of = str(as_of_date or "").strip()
    if not as_of:
        raise ValueError("as_of_date 不可空白")

    included_specs = tuple(spec for spec in specs if spec.included)
    rows: list[BootstrapPlanRow] = []

    for spec in included_specs:
        evidence = _require_probe(evidence_by_dataset, spec.dataset)
        mode = spec.bootstrap_mode
        if mode in {BOOTSTRAP_SINGLE_NO_DATES, BOOTSTRAP_SINGLE_FULL_RANGE}:
            request_count = 1
            basis = "single canonical request"
        elif mode == BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE:
            request_count = len(instruments)
            basis = f"{len(instruments)} historical instruments × 1 full-range request"
        elif mode == BOOTSTRAP_BULK_REFERENCE_DATES:
            dates = tuple(sorted({date for date in evidence.observed_dates if date and date <= as_of}))
            if not dates:
                raise ValueError(f"{spec.dataset} bulk planner 缺少 reference observed_dates")
            request_count = len(dates)
            basis = f"{len(dates)} observed reference dates × 1 all-market exact-date request"
        elif mode == BOOTSTRAP_FIXED_DATA_ID_FULL_RANGE:
            if not spec.fixed_data_ids:
                raise ValueError(f"{spec.dataset} fixed_data_ids 不可為空")
            request_count = len(spec.fixed_data_ids)
            basis = f"{len(spec.fixed_data_ids)} fixed data_id × 1 full-range request"
        else:
            raise ValueError(f"不支援的 bootstrap_mode: {spec.dataset} -> {mode}")
        rows.append(
            BootstrapPlanRow(
                dataset=spec.dataset,
                category=spec.category,
                bootstrap_mode=mode,
                request_count=int(request_count),
                basis=basis,
            )
        )

    total = sum(row.request_count for row in rows)
    per_instrument_dataset_count = sum(
        spec.bootstrap_mode == BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE for spec in included_specs
    )
    return BootstrapRequestPlan(
        as_of_date=as_of,
        historical_instrument_count=len(instruments),
        included_dataset_count=len(included_specs),
        per_instrument_dataset_count=int(per_instrument_dataset_count),
        total_requests=int(total),
        quota_limit=limit,
        minimum_quota_hours=float(total / limit),
        minimum_quota_windows=int(ceil(total / limit)),
        rows=tuple(rows),
    )


__all__ = [
    "DatasetProbeEvidence",
    "BootstrapPlanRow",
    "BootstrapRequestPlan",
    "build_bootstrap_request_plan",
]
