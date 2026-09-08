"""Neutral Market Data V2 provider-snapshot repository.

This module is the cross-domain owner for locating and validating immutable
Provider Snapshots.  Trading and Research may consume the same neutral snapshot,
but neither domain may mutate it or silently substitute its own lifecycle state.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.file_integrity import canonical_json_sha256, load_json_strict
from core.market_data_bootstrap_requests import BootstrapRequestManifest, build_registry_fingerprint
from core.market_data_dataset_registry import get_market_dataset_specs
from core.market_data_provider_snapshot import (
    ProviderArtifactEvidence,
    build_provider_snapshot_identity_payload,
    provider_snapshot_identity_from_payload,
)
from core.market_data_storage_contract import (
    MARKET_DATA_BOOTSTRAP_RELATIVE_ROOT,
    MARKET_DATA_PROVIDER_SNAPSHOT_FILENAME,
    resolve_market_data_bootstrap_ledger_path,
)
from services.downloader.market_data_ledger import (
    LedgerCommittedArtifact,
    MarketDataJobLedger,
    WORKLOAD_DONE,
)


@dataclass(frozen=True)
class ReadyProviderSnapshotArchive:
    path: Path
    payload: dict[str, Any]
    ledger_path: Path
    workload_id: str
    artifacts: tuple[LedgerCommittedArtifact, ...]

    @property
    def snapshot_fingerprint(self) -> str:
        return str(self.payload["snapshot_fingerprint"])

    @property
    def manifest_fingerprint(self) -> str:
        return str(self.payload["manifest_fingerprint"])

    @property
    def as_of_date(self) -> str:
        return str(self.payload["as_of_date"])


def validate_provider_snapshot_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict) or str(payload.get("status") or "") != "READY":
        raise ValueError("Market Data V2 provider snapshot 尚未 READY")
    identity = provider_snapshot_identity_from_payload(payload)
    if str(payload.get("snapshot_fingerprint") or "") != canonical_json_sha256(identity):
        raise ValueError("Market Data V2 provider snapshot fingerprint 不一致")
    current_registry = build_registry_fingerprint(get_market_dataset_specs(included_only=True))
    if str(payload.get("registry_fingerprint") or "") != current_registry:
        raise ValueError("Market Data V2 provider snapshot registry 與 current registry 已 drift")
    return payload


def find_latest_ready_provider_snapshot(project_root) -> tuple[Path, dict[str, Any]] | None:
    root = Path(project_root).resolve()
    base = root / MARKET_DATA_BOOTSTRAP_RELATIVE_ROOT
    candidates: list[tuple[str, str, Path, dict[str, Any]]] = []
    if not base.is_dir():
        return None
    for path in base.glob(f"*/{MARKET_DATA_PROVIDER_SNAPSHOT_FILENAME}"):
        try:
            payload = validate_provider_snapshot_payload(load_json_strict(path))
        except (OSError, ValueError, TypeError):
            continue
        candidates.append(
            (
                str(payload.get("as_of_date") or ""),
                str(payload.get("finalized_at") or ""),
                path,
                payload,
            )
        )
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], str(item[2])))
    _as_of, _finalized, path, payload = candidates[-1]
    return path, payload


def find_ready_provider_snapshot_by_fingerprint(
    project_root,
    snapshot_fingerprint: str,
) -> tuple[Path, dict[str, Any]] | None:
    wanted = str(snapshot_fingerprint or "").strip()
    if len(wanted) != 64:
        raise ValueError("provider snapshot fingerprint 不合法")
    root = Path(project_root).resolve()
    base = root / MARKET_DATA_BOOTSTRAP_RELATIVE_ROOT
    if not base.is_dir():
        return None
    for path in base.glob(f"*/{MARKET_DATA_PROVIDER_SNAPSHOT_FILENAME}"):
        try:
            payload = validate_provider_snapshot_payload(load_json_strict(path))
        except (OSError, ValueError, TypeError):
            continue
        if str(payload.get("snapshot_fingerprint") or "") == wanted:
            return path, payload
    return None


def _rebuild_snapshot_identity_from_ledger(
    payload: dict[str, Any],
    artifacts: tuple[LedgerCommittedArtifact, ...],
) -> dict[str, object]:
    requests = tuple(item.to_request() for item in artifacts)
    for item, request in zip(artifacts, requests):
        if request.request_id != item.request_id:
            raise ValueError(f"Provider Snapshot ledger request identity drift: {item.request_id}")
    manifest = BootstrapRequestManifest(
        as_of_date=str(payload.get("as_of_date") or ""),
        # Provider Snapshot identity does not include full_range_start; the ledger
        # contains the exact committed requests that define this immutable source.
        full_range_start="",
        registry_fingerprint=str(payload.get("registry_fingerprint") or ""),
        manifest_fingerprint=str(payload.get("manifest_fingerprint") or ""),
        historical_instrument_count=int(payload.get("historical_instrument_count") or 0),
        requests=requests,
    )
    evidence = tuple(
        ProviderArtifactEvidence(
            request_id=item.request_id,
            dataset=item.dataset,
            row_count=int(item.row_count),
            content_sha256=item.content_sha256,
        )
        for item in artifacts
    )
    return build_provider_snapshot_identity_payload(manifest=manifest, artifacts=evidence)


def load_ready_provider_snapshot_archive(
    project_root,
    *,
    snapshot_fingerprint: str | None = None,
) -> ReadyProviderSnapshotArchive:
    root = Path(project_root).resolve()
    if snapshot_fingerprint is None:
        found = find_latest_ready_provider_snapshot(root)
    else:
        found = find_ready_provider_snapshot_by_fingerprint(root, snapshot_fingerprint)
    if found is None:
        raise FileNotFoundError("Market Data V2 READY Provider Snapshot 不存在")
    path, payload = found
    manifest_fingerprint = str(payload.get("manifest_fingerprint") or "")
    ledger_path = resolve_market_data_bootstrap_ledger_path(root, manifest_fingerprint)
    if not ledger_path.is_file():
        raise FileNotFoundError("Provider Snapshot 對應 bootstrap ledger 不存在")
    ledger = MarketDataJobLedger(ledger_path)
    workload_id = f"market_data_v2_bootstrap:{manifest_fingerprint}"
    summary = ledger.get_summary(workload_id)
    expected_total = int(payload.get("total_requests") or 0)
    if summary.workload_status != WORKLOAD_DONE or summary.done != expected_total or summary.total != expected_total:
        raise ValueError(
            "Provider Snapshot ledger completeness drift: "
            f"status={summary.workload_status}, done={summary.done}, total={summary.total}, expected={expected_total}"
        )
    artifacts = ledger.list_committed_artifacts(workload_id)
    if len(artifacts) != expected_total:
        raise ValueError("Provider Snapshot ledger committed artifact count drift")
    rebuilt = _rebuild_snapshot_identity_from_ledger(payload, artifacts)
    expected_identity = provider_snapshot_identity_from_payload(payload)
    if rebuilt != expected_identity:
        raise ValueError("Provider Snapshot 與 immutable bootstrap ledger identity 不一致")
    return ReadyProviderSnapshotArchive(
        path=path,
        payload=payload,
        ledger_path=ledger_path,
        workload_id=workload_id,
        artifacts=artifacts,
    )


__all__ = [
    "ReadyProviderSnapshotArchive",
    "validate_provider_snapshot_payload",
    "find_latest_ready_provider_snapshot",
    "find_ready_provider_snapshot_by_fingerprint",
    "load_ready_provider_snapshot_archive",
]
