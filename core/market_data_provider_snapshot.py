"""Canonical identity contract for a completed neutral Market Data V2 provider snapshot."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable

from core.file_integrity import canonical_json_sha256
from core.market_data_bootstrap_requests import BootstrapRequestManifest

MARKET_DATA_PROVIDER_SNAPSHOT_SCHEMA_VERSION = 1
MARKET_DATA_PROVIDER_NAME = "FinMind"
MARKET_DATA_PROVIDER_SNAPSHOT_ROLE = "neutral_provider_bootstrap"


@dataclass(frozen=True)
class ProviderArtifactEvidence:
    request_id: str
    dataset: str
    row_count: int
    content_sha256: str


@dataclass(frozen=True)
class ProviderDatasetSummary:
    dataset: str
    request_count: int
    row_count: int
    artifact_fingerprint: str


def _normalize_artifacts(
    manifest: BootstrapRequestManifest,
    artifacts: Iterable[ProviderArtifactEvidence],
) -> tuple[ProviderArtifactEvidence, ...]:
    ordered = tuple(artifacts)
    if len(ordered) != manifest.total_requests:
        raise ValueError(
            f"Provider snapshot artifact 數與 manifest 不一致: actual={len(ordered)}, expected={manifest.total_requests}"
        )
    expected_ids = tuple(request.request_id for request in manifest.requests)
    actual_ids = tuple(item.request_id for item in ordered)
    if actual_ids != expected_ids:
        raise ValueError("Provider snapshot artifact request order/identity 與 manifest 不一致")
    for item in ordered:
        if int(item.row_count) < 0:
            raise ValueError(f"Provider snapshot row_count 不得為負數: {item.request_id}")
        digest = str(item.content_sha256 or "").strip().lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError(f"Provider snapshot content_sha256 不合法: {item.request_id}")
    return ordered


def build_provider_dataset_summaries(
    manifest: BootstrapRequestManifest,
    artifacts: Iterable[ProviderArtifactEvidence],
) -> tuple[ProviderDatasetSummary, ...]:
    ordered = _normalize_artifacts(manifest, artifacts)
    grouped: dict[str, list[ProviderArtifactEvidence]] = defaultdict(list)
    for item in ordered:
        grouped[item.dataset].append(item)
    summaries: list[ProviderDatasetSummary] = []
    for dataset in sorted(grouped):
        items = grouped[dataset]
        summaries.append(
            ProviderDatasetSummary(
                dataset=dataset,
                request_count=len(items),
                row_count=sum(int(item.row_count) for item in items),
                artifact_fingerprint=canonical_json_sha256(
                    [
                        {
                            "request_id": item.request_id,
                            "row_count": int(item.row_count),
                            "content_sha256": item.content_sha256,
                        }
                        for item in items
                    ]
                ),
            )
        )
    return tuple(summaries)


def build_provider_snapshot_identity_payload(
    *,
    manifest: BootstrapRequestManifest,
    artifacts: Iterable[ProviderArtifactEvidence],
) -> dict[str, object]:
    ordered = _normalize_artifacts(manifest, artifacts)
    dataset_summaries = build_provider_dataset_summaries(manifest, ordered)
    artifacts_fingerprint = canonical_json_sha256(
        [
            {
                "request_id": item.request_id,
                "dataset": item.dataset,
                "row_count": int(item.row_count),
                "content_sha256": item.content_sha256,
            }
            for item in ordered
        ]
    )
    identity = {
        "schema_version": MARKET_DATA_PROVIDER_SNAPSHOT_SCHEMA_VERSION,
        "provider": MARKET_DATA_PROVIDER_NAME,
        "snapshot_role": MARKET_DATA_PROVIDER_SNAPSHOT_ROLE,
        "status": "READY",
        "as_of_date": manifest.as_of_date,
        "registry_fingerprint": manifest.registry_fingerprint,
        "manifest_fingerprint": manifest.manifest_fingerprint,
        "historical_instrument_count": manifest.historical_instrument_count,
        "total_requests": manifest.total_requests,
        "total_rows": sum(int(item.row_count) for item in ordered),
        "artifacts_fingerprint": artifacts_fingerprint,
        "datasets": [asdict(item) for item in dataset_summaries],
    }
    return identity


def build_provider_snapshot_payload(
    *,
    manifest: BootstrapRequestManifest,
    artifacts: Iterable[ProviderArtifactEvidence],
    finalized_at: str,
) -> dict[str, object]:
    identity = build_provider_snapshot_identity_payload(manifest=manifest, artifacts=artifacts)
    snapshot_fingerprint = canonical_json_sha256(identity)
    return {
        **identity,
        "snapshot_fingerprint": snapshot_fingerprint,
        "finalized_at": str(finalized_at),
    }


def provider_snapshot_identity_from_payload(payload: dict[str, object]) -> dict[str, object]:
    return {
        key: payload.get(key)
        for key in (
            "schema_version",
            "provider",
            "snapshot_role",
            "status",
            "as_of_date",
            "registry_fingerprint",
            "manifest_fingerprint",
            "historical_instrument_count",
            "total_requests",
            "total_rows",
            "artifacts_fingerprint",
            "datasets",
        )
    }


__all__ = [
    "MARKET_DATA_PROVIDER_SNAPSHOT_SCHEMA_VERSION",
    "MARKET_DATA_PROVIDER_NAME",
    "MARKET_DATA_PROVIDER_SNAPSHOT_ROLE",
    "ProviderArtifactEvidence",
    "ProviderDatasetSummary",
    "build_provider_dataset_summaries",
    "build_provider_snapshot_identity_payload",
    "build_provider_snapshot_payload",
    "provider_snapshot_identity_from_payload",
]
