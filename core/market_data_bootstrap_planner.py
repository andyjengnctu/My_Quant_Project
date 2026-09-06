"""Pure bootstrap request planning for the Market Data V2 archive."""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterable, Mapping

from core.market_data_bootstrap_requests import build_bootstrap_request_manifest
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
    registry_fingerprint: str
    manifest_fingerprint: str
    rows: tuple[BootstrapPlanRow, ...]


def _basis_for_mode(mode: str, *, historical_instrument_count: int, request_count: int, spec: MarketDatasetSpec) -> str:
    if mode in {BOOTSTRAP_SINGLE_NO_DATES, BOOTSTRAP_SINGLE_FULL_RANGE}:
        return "single canonical request"
    if mode == BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE:
        return f"{historical_instrument_count} historical instruments × 1 full-range request"
    if mode == BOOTSTRAP_BULK_REFERENCE_DATES:
        return f"{request_count} observed reference dates × 1 all-market exact-date request"
    if mode == BOOTSTRAP_FIXED_DATA_ID_FULL_RANGE:
        if spec.bootstrap_chunk_months > 0:
            return f"{len(spec.fixed_data_ids)} fixed data_id × calendar-{spec.bootstrap_chunk_months}mo chunks ({request_count} requests)"
        if spec.bootstrap_chunk_years > 0:
            return f"{len(spec.fixed_data_ids)} fixed data_id × calendar-{spec.bootstrap_chunk_years}y chunks ({request_count} requests)"
        return f"{request_count} fixed data_id × 1 full-range request"
    raise ValueError(f"不支援的 bootstrap_mode: {mode}")


def build_bootstrap_request_plan(
    *,
    specs: Iterable[MarketDatasetSpec],
    historical_instruments: Iterable[str],
    evidence_by_dataset: Mapping[str, DatasetProbeEvidence],
    as_of_date: str,
    quota_limit: int,
) -> BootstrapRequestPlan:
    limit = int(quota_limit)
    if limit <= 0:
        raise ValueError(f"api_request_limit 必須 > 0: {quota_limit!r}")

    included_specs = tuple(spec for spec in specs if spec.included)
    manifest = build_bootstrap_request_manifest(
        specs=included_specs,
        historical_instruments=historical_instruments,
        evidence_by_dataset=evidence_by_dataset,
        as_of_date=as_of_date,
    )
    counts: dict[str, int] = {}
    for request in manifest.requests:
        counts[request.dataset] = counts.get(request.dataset, 0) + 1

    rows = tuple(
        BootstrapPlanRow(
            dataset=spec.dataset,
            category=spec.category,
            bootstrap_mode=spec.bootstrap_mode,
            request_count=counts.get(spec.dataset, 0),
            basis=_basis_for_mode(
                spec.bootstrap_mode,
                historical_instrument_count=manifest.historical_instrument_count,
                request_count=counts.get(spec.dataset, 0),
                spec=spec,
            ),
        )
        for spec in included_specs
    )
    total = manifest.total_requests
    per_instrument_dataset_count = sum(
        spec.bootstrap_mode == BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE for spec in included_specs
    )
    return BootstrapRequestPlan(
        as_of_date=manifest.as_of_date,
        historical_instrument_count=manifest.historical_instrument_count,
        included_dataset_count=len(included_specs),
        per_instrument_dataset_count=int(per_instrument_dataset_count),
        total_requests=int(total),
        quota_limit=limit,
        minimum_quota_hours=float(total / limit),
        minimum_quota_windows=int(ceil(total / limit)),
        registry_fingerprint=manifest.registry_fingerprint,
        manifest_fingerprint=manifest.manifest_fingerprint,
        rows=rows,
    )


__all__ = [
    "DatasetProbeEvidence",
    "BootstrapPlanRow",
    "BootstrapRequestPlan",
    "build_bootstrap_request_plan",
]
