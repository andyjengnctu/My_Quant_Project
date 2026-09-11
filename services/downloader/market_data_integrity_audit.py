"""Local-only Market Data V2 physical/request integrity audit.

This audit proves integrity of the canonical local archive evidence without
calling the provider.  It intentionally does not claim absolute provider row
completeness when a dataset has no authoritative dataset-specific expected
instrument universe.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable

import pandas as pd

from core.file_integrity import compute_file_sha256, load_json_strict
from core.market_data_dataset_readiness import (
    MARKET_DATA_DATASET_VALIDATION_CONTRACT_VERSION,
    VALID_DATASET_VALIDATION_STATUSES,
    is_market_data_dataset_ready,
)
from core.market_data_dataset_registry import get_market_dataset_display_name_zh, get_market_dataset_specs
from core.market_data_integrity_contract import (
    AUTHORITATIVE_TRADING_CALENDAR_DATASET,
    DATE_SEMANTIC_FAIL,
    DATE_SEMANTIC_NOT_APPLICABLE,
    DATE_SEMANTIC_NOT_APPLICABLE_STATUS,
    DATE_SEMANTIC_PASS,
    DATE_SEMANTIC_UNVERIFIED,
    DATE_SEMANTIC_UNVERIFIED_STATUS,
    evaluate_date_semantics,
    resolve_date_semantic_mode,
    validate_market_data_integrity_contract,
)
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


def _new_semantic_row(spec) -> dict[str, object]:
    keys = tuple(str(value) for value in spec.primary_key_hint)
    return {
        "dataset": spec.dataset,
        "display_name_zh": get_market_dataset_display_name_zh(spec.dataset),
        "primary_key": keys,
        "natural_key_status": "UNVERIFIED" if not keys else "PASS",
        "natural_key_artifacts_checked": 0,
        "natural_key_null_rows": 0,
        "natural_key_duplicate_rows": 0,
        "natural_key_missing_column_artifacts": 0,
        "date_semantic_mode": resolve_date_semantic_mode(spec),
        "observed_dates": set(),
        "invalid_date_rows": 0,
        "missing_date_column_artifacts": 0,
    }


def _natural_key_issue_counts(frame: pd.DataFrame, primary_key: tuple[str, ...]) -> tuple[int, int]:
    keys = tuple(str(value) for value in primary_key if str(value))
    if not keys:
        raise ValueError("natural-key audit 需要非空 primary key")
    missing = [key for key in keys if key not in frame.columns]
    if missing:
        raise ValueError(f"natural-key audit frame 缺 key 欄位: {missing}")
    null_rows = int(frame.loc[:, list(keys)].isna().any(axis=1).sum())
    duplicate_rows = int(frame.duplicated(list(keys), keep=False).sum())
    return null_rows, duplicate_rows


def _observe_semantic_artifact(*, path: Path, inspection, spec, semantic_row: dict[str, object]) -> None:
    if int(inspection.row_count) <= 0:
        return

    keys = tuple(str(value) for value in spec.primary_key_hint)
    date_mode = str(semantic_row["date_semantic_mode"])
    needs_date = date_mode not in {DATE_SEMANTIC_NOT_APPLICABLE, DATE_SEMANTIC_UNVERIFIED}
    available = set(str(value) for value in inspection.columns)
    missing_keys = [key for key in keys if key not in available]
    if keys and missing_keys:
        semantic_row["natural_key_status"] = "FAIL"
        semantic_row["natural_key_missing_column_artifacts"] = int(semantic_row["natural_key_missing_column_artifacts"]) + 1
    if needs_date and "date" not in available:
        semantic_row["missing_date_column_artifacts"] = int(semantic_row["missing_date_column_artifacts"]) + 1

    requested_columns = tuple(dict.fromkeys((*(key for key in keys if key in available), *(("date",) if needs_date and "date" in available else ()))))
    if not requested_columns:
        return
    frame = pd.read_parquet(path, columns=list(requested_columns))

    if keys and not missing_keys:
        semantic_row["natural_key_artifacts_checked"] = int(semantic_row["natural_key_artifacts_checked"]) + 1
        null_rows, duplicate_rows = _natural_key_issue_counts(frame, keys)
        semantic_row["natural_key_null_rows"] = int(semantic_row["natural_key_null_rows"]) + null_rows
        semantic_row["natural_key_duplicate_rows"] = int(semantic_row["natural_key_duplicate_rows"]) + duplicate_rows
        if null_rows or duplicate_rows:
            semantic_row["natural_key_status"] = "FAIL"

    if needs_date and "date" in frame.columns:
        raw = frame["date"]
        parsed = pd.to_datetime(raw, errors="coerce")
        invalid_rows = int((raw.notna() & parsed.isna()).sum())
        semantic_row["invalid_date_rows"] = int(semantic_row["invalid_date_rows"]) + invalid_rows
        normalized = parsed.dropna().dt.strftime("%Y-%m-%d")
        semantic_row["observed_dates"].update(str(value) for value in normalized.unique())


def _finalize_semantic_integrity(semantic_rows: dict[str, dict[str, object]]) -> dict[str, object]:
    calendar_row = semantic_rows[AUTHORITATIVE_TRADING_CALENDAR_DATASET]
    trading_calendar_dates = tuple(sorted(str(value) for value in calendar_row["observed_dates"]))
    rows: list[dict[str, object]] = []
    date_counts = defaultdict(int)
    key_counts = defaultdict(int)
    for dataset in sorted(semantic_rows):
        row = semantic_rows[dataset]
        date_result = evaluate_date_semantics(
            mode=str(row["date_semantic_mode"]),
            observed_dates=tuple(row["observed_dates"]),
            trading_calendar_dates=trading_calendar_dates,
            invalid_date_rows=int(row["invalid_date_rows"]),
            missing_date_column=bool(int(row["missing_date_column_artifacts"])),
        )
        natural_key_status = str(row["natural_key_status"] or "UNVERIFIED")
        date_counts[date_result.status] += 1
        key_counts[natural_key_status] += 1
        rows.append({
            "dataset": dataset,
            "display_name_zh": row["display_name_zh"],
            "date_semantic": date_result.as_dict(),
            "primary_key": list(row["primary_key"]),
            "natural_key_status": natural_key_status,
            "natural_key_artifacts_checked": int(row["natural_key_artifacts_checked"]),
            "natural_key_null_rows": int(row["natural_key_null_rows"]),
            "natural_key_duplicate_rows": int(row["natural_key_duplicate_rows"]),
            "natural_key_missing_column_artifacts": int(row["natural_key_missing_column_artifacts"]),
        })

    status = "PASS" if date_counts[DATE_SEMANTIC_FAIL] == 0 and key_counts["FAIL"] == 0 else "FAIL"
    return {
        "status": status,
        "date_semantic": {
            "pass_count": int(date_counts[DATE_SEMANTIC_PASS]),
            "fail_count": int(date_counts[DATE_SEMANTIC_FAIL]),
            "unverified_count": int(date_counts[DATE_SEMANTIC_UNVERIFIED_STATUS]),
            "not_applicable_count": int(date_counts[DATE_SEMANTIC_NOT_APPLICABLE_STATUS]),
        },
        "natural_key": {
            "pass_count": int(key_counts["PASS"]),
            "fail_count": int(key_counts["FAIL"]),
            "unverified_count": int(key_counts["UNVERIFIED"]),
        },
        "rows": rows,
    }


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
    metadata = dict(inspection.metadata or {})
    if "validation_contract_version" in payload:
        expected["validation_contract_version"] = int(payload["validation_contract_version"])
    elif metadata.get("validation_contract_version") not in (None, ""):
        raise ValueError(
            "Trading V2 pre-versioned artifact 不應含 validation contract metadata: "
            f"request={request.request_id}"
        )
    if payload.get("refresh_token") is not None:
        expected["refresh_token"] = payload.get("refresh_token")
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
    validate_market_data_integrity_contract()
    specs = tuple(get_market_dataset_specs(included_only=True))
    spec_by_dataset = {spec.dataset: spec for spec in specs}
    semantic_rows = {spec.dataset: _new_semantic_row(spec) for spec in specs}

    provider_total = len(archive.artifacts)
    provider_rows = 0
    for index, item in enumerate(archive.artifacts, start=1):
        request = item.to_request()
        path = resolve_market_data_request_parquet_path(root, archive.manifest_fingerprint, request)
        inspection = _verify_artifact(path=path, ledger_item=item, codec=codec)
        _verify_provider_metadata(inspection=inspection, archive=archive, ledger_item=item)
        spec = spec_by_dataset.get(str(item.dataset))
        if spec is None:
            raise ValueError(f"Provider Snapshot artifact dataset 不在 current included registry: {item.dataset}")
        _observe_semantic_artifact(path=path, inspection=inspection, spec=spec, semantic_row=semantic_rows[spec.dataset])
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
                spec = spec_by_dataset.get(str(item.dataset))
                if spec is None:
                    raise ValueError(f"Trading V2 overlay dataset 不在 current included registry: {item.dataset}")
                _observe_semantic_artifact(path=path, inspection=inspection, spec=spec, semantic_row=semantic_rows[spec.dataset])
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
    semantic_integrity = _finalize_semantic_integrity(semantic_rows)
    local_integrity = "PASS" if (dataset_validation["status"] == "PASS" and semantic_integrity["status"] == "PASS") else "FAIL"
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
        "semantic_integrity": semantic_integrity,
        "absolute_instrument_completeness": "UNVERIFIED",
        "absolute_instrument_completeness_reason": "no_authoritative_dataset_specific_expected_universe",
        "provider_requests_made": 0,
    }


__all__ = ["run_market_data_v2_full_integrity_audit"]
