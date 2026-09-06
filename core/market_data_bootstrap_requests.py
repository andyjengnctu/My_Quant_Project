"""Canonical Market Data V2 bootstrap request manifest construction.

This module expands the provider-facing dataset registry plus preflight evidence
into deterministic logical HTTP requests.  It owns request identity so the
planner, persistent ledger and later storage executor cannot drift apart.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Iterable, Mapping

from core.market_data_dataset_registry import (
    BOOTSTRAP_BULK_REFERENCE_DATES,
    BOOTSTRAP_FIXED_DATA_ID_FULL_RANGE,
    BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE,
    BOOTSTRAP_SINGLE_FULL_RANGE,
    BOOTSTRAP_SINGLE_NO_DATES,
    MarketDatasetSpec,
)

BOOTSTRAP_FULL_RANGE_START = "1900-01-01"


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def _normalize_instruments(instruments: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(sorted({str(item or "").strip() for item in instruments if str(item or "").strip()}))
    if not normalized:
        raise ValueError("Historical instrument universe 不可為空")
    return normalized


def _evidence_status(evidence: object) -> str:
    if isinstance(evidence, Mapping):
        return str(evidence.get("status") or "").strip()
    return str(getattr(evidence, "status", "") or "").strip()


def _evidence_error(evidence: object) -> str | None:
    if isinstance(evidence, Mapping):
        value = evidence.get("error")
    else:
        value = getattr(evidence, "error", None)
    resolved = str(value or "").strip()
    return resolved or None


def _evidence_observed_dates(evidence: object) -> tuple[str, ...]:
    if isinstance(evidence, Mapping):
        raw = evidence.get("observed_dates") or ()
    else:
        raw = getattr(evidence, "observed_dates", ()) or ()
    return tuple(sorted({str(value or "").strip() for value in raw if str(value or "").strip()}))


def _require_probe(evidence_by_dataset: Mapping[str, object], dataset: str) -> object:
    evidence = evidence_by_dataset.get(dataset)
    if evidence is None:
        raise ValueError(f"缺少 dataset preflight evidence: {dataset}")
    status = _evidence_status(evidence)
    if status != "PASS":
        raise ValueError(f"dataset preflight 尚未 PASS: {dataset} | {_evidence_error(evidence) or status}")
    return evidence


@dataclass(frozen=True)
class BootstrapHttpRequest:
    dataset: str
    bootstrap_mode: str
    data_id: str | None
    start_date: str | None
    end_date: str | None

    @property
    def request_id(self) -> str:
        return _canonical_sha256(
            {
                "dataset": self.dataset,
                "bootstrap_mode": self.bootstrap_mode,
                "data_id": self.data_id,
                "start_date": self.start_date,
                "end_date": self.end_date,
            }
        )


@dataclass(frozen=True)
class BootstrapRequestManifest:
    as_of_date: str
    full_range_start: str
    registry_fingerprint: str
    manifest_fingerprint: str
    historical_instrument_count: int
    requests: tuple[BootstrapHttpRequest, ...]

    @property
    def total_requests(self) -> int:
        return len(self.requests)


def build_registry_fingerprint(specs: Iterable[MarketDatasetSpec]) -> str:
    ordered = sorted(
        (asdict(spec) for spec in specs),
        key=lambda item: (str(item.get("dataset") or ""), str(item.get("archive_policy") or "")),
    )
    return _canonical_sha256(ordered)


def build_bootstrap_request_manifest(
    *,
    specs: Iterable[MarketDatasetSpec],
    historical_instruments: Iterable[str],
    evidence_by_dataset: Mapping[str, object],
    as_of_date: str,
    full_range_start: str = BOOTSTRAP_FULL_RANGE_START,
) -> BootstrapRequestManifest:
    instruments = _normalize_instruments(historical_instruments)
    as_of = str(as_of_date or "").strip()
    if not as_of:
        raise ValueError("as_of_date 不可空白")
    range_start = str(full_range_start or "").strip()
    if not range_start:
        raise ValueError("full_range_start 不可空白")
    if range_start > as_of:
        raise ValueError(f"full_range_start 不得晚於 as_of_date: {range_start} > {as_of}")

    included_specs = tuple(spec for spec in specs if spec.included)
    if not included_specs:
        raise ValueError("Market Data bootstrap 沒有 included dataset")

    requests: list[BootstrapHttpRequest] = []
    for spec in included_specs:
        evidence = _require_probe(evidence_by_dataset, spec.dataset)
        mode = spec.bootstrap_mode
        if mode == BOOTSTRAP_SINGLE_NO_DATES:
            requests.append(BootstrapHttpRequest(spec.dataset, mode, None, None, None))
        elif mode == BOOTSTRAP_SINGLE_FULL_RANGE:
            requests.append(BootstrapHttpRequest(spec.dataset, mode, None, range_start, as_of))
        elif mode == BOOTSTRAP_PER_INSTRUMENT_FULL_RANGE:
            requests.extend(
                BootstrapHttpRequest(spec.dataset, mode, instrument, range_start, as_of)
                for instrument in instruments
            )
        elif mode == BOOTSTRAP_BULK_REFERENCE_DATES:
            dates = tuple(date for date in _evidence_observed_dates(evidence) if date <= as_of)
            if not dates:
                raise ValueError(f"{spec.dataset} bulk request manifest 缺少 reference observed_dates")
            requests.extend(BootstrapHttpRequest(spec.dataset, mode, None, date, date) for date in dates)
        elif mode == BOOTSTRAP_FIXED_DATA_ID_FULL_RANGE:
            if not spec.fixed_data_ids:
                raise ValueError(f"{spec.dataset} fixed_data_ids 不可為空")
            requests.extend(
                BootstrapHttpRequest(spec.dataset, mode, data_id, range_start, as_of)
                for data_id in spec.fixed_data_ids
            )
        else:
            raise ValueError(f"不支援的 bootstrap_mode: {spec.dataset} -> {mode}")

    request_ids = tuple(request.request_id for request in requests)
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("Bootstrap request manifest 含重複 logical request identity")

    registry_fingerprint = build_registry_fingerprint(included_specs)
    manifest_fingerprint = _canonical_sha256(
        {
            "as_of_date": as_of,
            "full_range_start": range_start,
            "registry_fingerprint": registry_fingerprint,
            "historical_instruments": instruments,
            "request_ids": request_ids,
        }
    )
    return BootstrapRequestManifest(
        as_of_date=as_of,
        full_range_start=range_start,
        registry_fingerprint=registry_fingerprint,
        manifest_fingerprint=manifest_fingerprint,
        historical_instrument_count=len(instruments),
        requests=tuple(requests),
    )


__all__ = [
    "BOOTSTRAP_FULL_RANGE_START",
    "BootstrapHttpRequest",
    "BootstrapRequestManifest",
    "build_registry_fingerprint",
    "build_bootstrap_request_manifest",
]
