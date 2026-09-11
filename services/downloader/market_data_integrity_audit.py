"""Local-only Market Data V2 physical/request integrity audit.

This audit proves integrity of the canonical local archive evidence without
calling the provider.  It intentionally does not claim absolute provider row
completeness when a dataset has no authoritative dataset-specific expected
instrument universe.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from core.file_integrity import compute_file_sha256, load_json_strict
from core.market_data_dataset_readiness import (
    MARKET_DATA_DATASET_VALIDATION_CONTRACT_VERSION,
    VALID_DATASET_VALIDATION_STATUSES,
    is_market_data_dataset_ready,
)
from core.market_data_dataset_registry import get_market_dataset_specs
from core.market_data_storage_contract import resolve_market_data_request_parquet_path
from core.market_data_trading_storage_contract import (
    TRADING_MARKET_DATA_V2_BATCH_MANIFEST_FILENAME,
    resolve_trading_market_data_v2_ledger_path,
    resolve_trading_market_data_v2_request_path,
    resolve_trading_market_data_v2_root,
    validate_trading_sync_batch_manifest_payload_for_read,
)
from services.downloader.market_data_ledger import MarketDataJobLedger, WORKLOAD_DONE
from services.downloader.market_data_storage import PyArrowParquetCodec
from services.market_data.provider_snapshot_repository import load_ready_provider_snapshot_archive
from services.trading.market_data_dataset_state import load_market_data_dataset_state
from services.trading.market_data_v2_state import load_trading_market_data_v2_state


ProgressFn = Callable[[dict[str, object]], None]


def _verify_artifact(*, path: Path, ledger_item, codec: PyArrowParquetCodec):
    if not path.is_file():
        raise FileNotFoundError(f"Market Data V2 artifact 不存在: request={ledger_item.request_id}")
    inspection = codec.inspect(path)
    if int(inspection.row_count) != int(ledger_item.row_count):
        raise ValueError(
            "Market Data V2 artifact row_count 與 ledger 不一致: "
            f"request={ledger_item.request_id}, artifact={inspection.row_count}, ledger={ledger_item.row_count}"
        )
    if int(dict(inspection.metadata or {}).get("row_count", -1)) != int(inspection.row_count):
        raise ValueError(
            "Market Data V2 artifact row_count metadata 與實體不一致: "
            f"request={ledger_item.request_id}"
        )
    actual_sha = str(compute_file_sha256(path) or "").strip().lower()
    if actual_sha != str(ledger_item.content_sha256 or "").strip().lower():
        raise ValueError(f"Market Data V2 artifact SHA256 drift: request={ledger_item.request_id}")
    return inspection



def _verify_provider_metadata(*, inspection, archive, ledger_item) -> None:
    request = ledger_item.to_request()
    expected = {
        "manifest_fingerprint": archive.manifest_fingerprint,
        "registry_fingerprint": str(archive.payload.get("registry_fingerprint") or ""),
        "request_id": str(request.request_id),
        "dataset": str(request.dataset),
        "bootstrap_mode": str(request.bootstrap_mode),
        "data_id": request.data_id,
        "start_date": request.start_date,
        "end_date": request.end_date,
    }
    metadata = dict(inspection.metadata or {})
    actual = {key: metadata.get(key) for key in expected}
    if actual != expected:
        raise ValueError(
            "Provider Snapshot artifact request identity metadata drift: "
            f"request={request.request_id}, actual={actual}, expected={expected}"
        )

def _verify_overlay_metadata(*, inspection, payload: dict[str, object], ledger_item) -> None:
    request = ledger_item.to_request()
    expected = {
        "batch_fingerprint": str(payload["batch_fingerprint"]),
        "validation_contract_version": int(payload["validation_contract_version"]),
        "registry_fingerprint": str(payload["registry_fingerprint"]),
        "base_provider_snapshot_fingerprint": str(payload["base_provider_snapshot_fingerprint"]),
        "base_provider_manifest_fingerprint": str(payload["base_provider_manifest_fingerprint"]),
        "request_id": str(request.request_id),
        "dataset": str(request.dataset),
        "bootstrap_mode": str(request.bootstrap_mode),
        "data_id": request.data_id,
        "start_date": request.start_date,
        "end_date": request.end_date,
    }
    if payload.get("refresh_token") is not None:
        expected["refresh_token"] = payload.get("refresh_token")
    metadata = dict(inspection.metadata or {})
    actual = {key: metadata.get(key) for key in expected}
    if actual != expected:
        raise ValueError(
            "Trading V2 artifact request identity metadata drift: "
            f"request={request.request_id}, actual={actual}, expected={expected}"
        )


def _dataset_validation_summary(project_root: Path, *, target_date: str) -> dict[str, object]:
    specs = tuple(get_market_dataset_specs(included_only=True))
    expected = {spec.dataset for spec in specs}
    state = load_market_data_dataset_state(project_root, required=False)
    rows = dict((state or {}).get("datasets") or {})
    actual = set(rows)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    current_validation = 0
    schema_valid = 0
    coverage_valid = 0
    ready = 0
    for dataset in sorted(expected):
        row = dict(rows.get(dataset) or {})
        try:
            current = int(row.get("validation_contract_version", -1)) == MARKET_DATA_DATASET_VALIDATION_CONTRACT_VERSION
        except (TypeError, ValueError):
            current = False
        if current:
            current_validation += 1
            if str(row.get("schema_status") or "") in VALID_DATASET_VALIDATION_STATUSES:
                schema_valid += 1
            if str(row.get("coverage_status") or "") in VALID_DATASET_VALIDATION_STATUSES:
                coverage_valid += 1
        if is_market_data_dataset_ready(row, target_date=str(target_date)):
            ready += 1
    total = len(expected)
    status = "PASS" if (
        not missing
        and not unexpected
        and current_validation == total
        and schema_valid == total
        and coverage_valid == total
        and ready == total
    ) else "FAIL"
    return {
        "status": status,
        "dataset_count": total,
        "state_dataset_count": len(rows),
        "current_validation_count": current_validation,
        "schema_valid_count": schema_valid,
        "coverage_valid_count": coverage_valid,
        "ready_count": ready,
        "missing_datasets": missing,
        "unexpected_datasets": unexpected,
    }


def run_market_data_v2_full_integrity_audit(
    *,
    project_root,
    progress_fn: ProgressFn | None = None,
) -> dict[str, object]:
    """Verify canonical local Market Data V2 evidence without provider I/O."""

    root = Path(project_root).resolve()
    codec = PyArrowParquetCodec()
    archive = load_ready_provider_snapshot_archive(root)

    provider_total = len(archive.artifacts)
    provider_rows = 0
    for index, item in enumerate(archive.artifacts, start=1):
        request = item.to_request()
        path = resolve_market_data_request_parquet_path(root, archive.manifest_fingerprint, request)
        inspection = _verify_artifact(path=path, ledger_item=item, codec=codec)
        _verify_provider_metadata(inspection=inspection, archive=archive, ledger_item=item)
        provider_rows += int(item.row_count)
        if progress_fn is not None and (index == provider_total or index % 500 == 0):
            progress_fn({
                "phase": "PROVIDER",
                "verified": index,
                "total": provider_total,
                "dataset": item.dataset,
                "data_id": item.data_id,
            })

    batch_root = resolve_trading_market_data_v2_root(root) / "batches"
    done_batches = 0
    incomplete_batches = 0
    overlay_requests = 0
    overlay_rows = 0
    overlay_artifacts_verified = 0
    if batch_root.is_dir():
        for manifest_path in sorted(batch_root.glob(f"*/{TRADING_MARKET_DATA_V2_BATCH_MANIFEST_FILENAME}")):
            payload = validate_trading_sync_batch_manifest_payload_for_read(load_json_strict(manifest_path))
            batch_fp = str(payload["batch_fingerprint"])
            if manifest_path.parent.name != batch_fp:
                raise ValueError(f"Trading V2 batch directory 與 manifest fingerprint 不一致: {batch_fp}")
            ledger_path = resolve_trading_market_data_v2_ledger_path(root, batch_fp)
            if not ledger_path.is_file():
                raise FileNotFoundError(f"Trading V2 batch ledger 不存在: {batch_fp}")
            ledger = MarketDataJobLedger(ledger_path, read_only=True)
            workload_id = ledger.find_unique_workload_id_by_manifest_fingerprint(batch_fp)
            if workload_id is None:
                raise ValueError(f"Trading V2 batch ledger 找不到 manifest workload: {batch_fp}")
            summary = ledger.get_summary(workload_id)
            expected = int(payload.get("request_count") or 0)
            if summary.workload_status != WORKLOAD_DONE:
                incomplete_batches += 1
                continue
            if (
                summary.total != expected
                or summary.done != expected
                or summary.pending
                or summary.running
                or summary.retryable
                or summary.blocked
            ):
                raise ValueError(
                    "Trading V2 DONE batch ledger completeness drift: "
                    f"batch={batch_fp}, done={summary.done}, total={summary.total}, expected={expected}"
                )
            artifacts = ledger.list_committed_artifacts(workload_id)
            request_ids = tuple(str(value) for value in payload.get("request_ids") or ())
            if tuple(item.request_id for item in artifacts) != request_ids:
                raise ValueError(f"Trading V2 batch committed request identity drift: {batch_fp}")
            done_batches += 1
            overlay_requests += len(artifacts)
            for item in artifacts:
                request = item.to_request()
                path = resolve_trading_market_data_v2_request_path(root, batch_fp, request)
                inspection = _verify_artifact(path=path, ledger_item=item, codec=codec)
                _verify_overlay_metadata(inspection=inspection, payload=payload, ledger_item=item)
                overlay_rows += int(item.row_count)
                overlay_artifacts_verified += 1
                if progress_fn is not None and overlay_artifacts_verified % 250 == 0:
                    progress_fn({
                        "phase": "OVERLAY",
                        "verified": overlay_artifacts_verified,
                        "dataset": item.dataset,
                        "data_id": item.data_id,
                    })

    archive_state = load_trading_market_data_v2_state(root, required=False)
    target_date = str((archive_state or {}).get("latest_sync_target_date") or archive.as_of_date)
    dataset_validation = _dataset_validation_summary(root, target_date=target_date)
    local_integrity = "PASS" if dataset_validation["status"] == "PASS" else "FAIL"
    return {
        "status": local_integrity,
        "provider_snapshot_status": "PASS",
        "provider_snapshot_fingerprint": archive.snapshot_fingerprint,
        "provider_as_of_date": archive.as_of_date,
        "provider_requests_verified": provider_total,
        "provider_rows_verified": provider_rows,
        "overlay_status": "PASS",
        "overlay_done_batches_verified": done_batches,
        "overlay_incomplete_batches_ignored": incomplete_batches,
        "overlay_requests_verified": overlay_requests,
        "overlay_rows_verified": overlay_rows,
        "target_date": target_date,
        "dataset_validation": dataset_validation,
        "absolute_instrument_completeness": "UNVERIFIED",
        "absolute_instrument_completeness_reason": "no_authoritative_dataset_specific_expected_universe",
        "provider_requests_made": 0,
    }


__all__ = ["run_market_data_v2_full_integrity_audit"]
