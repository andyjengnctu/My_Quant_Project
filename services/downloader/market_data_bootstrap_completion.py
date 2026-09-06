"""Integrity finalization for a completed neutral Market Data V2 bootstrap archive."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Callable

from core.file_integrity import atomic_write_json, atomic_write_text, canonical_json_sha256, load_json_strict
from core.market_data_provider_snapshot import (
    ProviderArtifactEvidence,
    build_provider_snapshot_identity_payload,
    build_provider_snapshot_payload,
    provider_snapshot_identity_from_payload,
)
from core.market_data_storage_contract import (
    MARKET_DATA_BOOTSTRAP_MANIFEST_FILENAME,
    resolve_market_data_bootstrap_archive_dir,
    resolve_market_data_bootstrap_ledger_path,
    resolve_market_data_provider_snapshot_path,
)
from core.path_utils import project_relative_display_path
from services.downloader.market_data_bootstrap_activation import MarketDataBootstrapActivation
from services.downloader.market_data_ledger import MarketDataJobLedger, WORKLOAD_DONE
from services.downloader.market_data_storage import MarketDataBootstrapStorageSink

PROVIDER_SNAPSHOT_REPORT_PREFIX = "market_data_v2_provider_snapshot_"


class MarketDataBootstrapCompletionError(RuntimeError):
    pass


def _provider_snapshot_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Market Data V2 Provider Snapshot",
        "",
        f"- status: **{payload.get('status')}**",
        f"- as-of date: **{payload.get('as_of_date')}**",
        f"- snapshot fingerprint: `{payload.get('snapshot_fingerprint')}`",
        f"- manifest fingerprint: `{payload.get('manifest_fingerprint')}`",
        f"- datasets: **{payload.get('dataset_count')}**",
        f"- historical instruments: **{payload.get('historical_instrument_count')}**",
        f"- requests verified: **{payload.get('verified_requests')} / {payload.get('total_requests')}**",
        f"- total rows: **{payload.get('total_rows')}**",
        f"- provider snapshot: `{payload.get('provider_snapshot_path')}`",
        f"- archive: `{payload.get('archive_dir')}`",
        f"- ledger: `{payload.get('ledger_path')}`",
        "",
        "此 snapshot 只是 Research V2 / Trading 後續分叉的 neutral provider source；不代表 Research V2 已建立或 Trading runtime 已切換。",
        "",
    ]
    return "\n".join(lines)


def _assert_existing_snapshot_identity(
    *,
    path: Path,
    expected_identity: dict[str, object],
) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        raw = load_json_strict(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise MarketDataBootstrapCompletionError(
            f"既有 provider snapshot manifest 無法讀取: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise MarketDataBootstrapCompletionError("既有 provider snapshot manifest root 必須為 object")
    actual_identity = provider_snapshot_identity_from_payload(raw)
    if actual_identity != expected_identity:
        raise MarketDataBootstrapCompletionError(
            "既有 provider snapshot identity 與目前已驗證 archive 不一致，禁止覆寫 immutable snapshot"
        )
    expected_fingerprint = canonical_json_sha256(expected_identity)
    if str(raw.get("snapshot_fingerprint") or "") != expected_fingerprint:
        raise MarketDataBootstrapCompletionError("既有 provider snapshot fingerprint 驗證失敗")
    return raw


def finalize_market_data_v2_provider_snapshot(
    *,
    activation: MarketDataBootstrapActivation,
    project_root,
    output_dir,
    now_fn: Callable[[], datetime] | None = None,
    storage=None,
    progress_fn: Callable[[dict[str, object]], None] | None = None,
    progress_every: int = 500,
) -> dict[str, object]:
    """Verify every committed artifact and publish one immutable provider snapshot manifest.

    This function performs no provider HTTP request.  It requires a fully DONE
    ledger, validates exact request identity parity, re-opens each committed
    Parquet artifact through the storage owner, recomputes SHA256, and only then
    publishes the neutral provider snapshot manifest.
    """

    manifest = activation.manifest
    root = Path(project_root).resolve()
    archive_dir = resolve_market_data_bootstrap_archive_dir(root, manifest.manifest_fingerprint)
    bootstrap_manifest_path = archive_dir / MARKET_DATA_BOOTSTRAP_MANIFEST_FILENAME
    if not bootstrap_manifest_path.is_file():
        raise MarketDataBootstrapCompletionError(
            "找不到 Bootstrap storage manifest；禁止在缺少 storage identity evidence 時建立 provider snapshot"
        )
    ledger_path = resolve_market_data_bootstrap_ledger_path(root, manifest.manifest_fingerprint)
    if not ledger_path.is_file():
        raise MarketDataBootstrapCompletionError(
            "找不到 Bootstrap ledger；完整下載尚未開始或 manifest identity 不一致"
        )

    ledger = MarketDataJobLedger(ledger_path)
    workload_id = ledger.workload_id_for_manifest(manifest)
    try:
        summary = ledger.get_summary(workload_id)
    except ValueError as exc:
        raise MarketDataBootstrapCompletionError("Bootstrap ledger 不含目前 manifest workload") from exc
    if summary.workload_status != WORKLOAD_DONE:
        raise MarketDataBootstrapCompletionError(
            f"Bootstrap workload 尚未 DONE: status={summary.workload_status}, done={summary.done}/{summary.total}, blocked={summary.blocked}"
        )
    if summary.total != manifest.total_requests or summary.done != manifest.total_requests:
        raise MarketDataBootstrapCompletionError(
            f"Bootstrap ledger completeness 不符: done={summary.done}, total={summary.total}, expected={manifest.total_requests}"
        )
    if summary.pending or summary.running or summary.retryable or summary.blocked:
        raise MarketDataBootstrapCompletionError("Bootstrap ledger 含非 DONE job，禁止建立 provider snapshot")

    try:
        committed = ledger.list_committed_artifacts(workload_id)
    except ValueError as exc:
        raise MarketDataBootstrapCompletionError(str(exc)) from exc
    if len(committed) != manifest.total_requests:
        raise MarketDataBootstrapCompletionError(
            f"DONE artifact ledger 數與 manifest 不一致: {len(committed)} != {manifest.total_requests}"
        )

    expected_request_ids = tuple(request.request_id for request in manifest.requests)
    ledger_request_ids = tuple(item.request_id for item in committed)
    if ledger_request_ids != expected_request_ids:
        raise MarketDataBootstrapCompletionError("DONE artifact ledger request order/identity 與 manifest 不一致")

    sink = storage or MarketDataBootstrapStorageSink(project_root=root, manifest=manifest)
    evidence: list[ProviderArtifactEvidence] = []
    every = max(1, int(progress_every))
    for index, (request, ledger_item) in enumerate(zip(manifest.requests, committed), start=1):
        if ledger_item.to_request() != request:
            raise MarketDataBootstrapCompletionError(
                f"Ledger request payload 與 manifest 不一致: {request.request_id}"
            )
        try:
            receipt = sink.recover_committed(request)
        except (RuntimeError, ValueError, OSError) as exc:
            raise MarketDataBootstrapCompletionError(
                f"Provider artifact verify 失敗: {request.dataset}/{request.data_id or '-'} | {type(exc).__name__}: {exc}"
            ) from exc
        if receipt is None or not receipt.committed:
            raise MarketDataBootstrapCompletionError(
                f"Provider artifact 缺失或未 committed: {request.dataset}/{request.data_id or '-'}"
            )
        if receipt.row_count != ledger_item.row_count:
            raise MarketDataBootstrapCompletionError(
                f"Provider artifact row_count 與 ledger 不一致: request={request.request_id}, artifact={receipt.row_count}, ledger={ledger_item.row_count}"
            )
        if receipt.content_sha256 != ledger_item.content_sha256:
            raise MarketDataBootstrapCompletionError(
                f"Provider artifact SHA256 與 ledger 不一致: request={request.request_id}"
            )
        evidence.append(
            ProviderArtifactEvidence(
                request_id=request.request_id,
                dataset=request.dataset,
                row_count=receipt.row_count,
                content_sha256=str(receipt.content_sha256),
            )
        )
        if progress_fn is not None and (index == manifest.total_requests or index % every == 0):
            progress_fn(
                {
                    "verified": index,
                    "total": manifest.total_requests,
                    "dataset": request.dataset,
                    "data_id": request.data_id,
                }
            )

    identity = build_provider_snapshot_identity_payload(manifest=manifest, artifacts=evidence)
    snapshot_path = resolve_market_data_provider_snapshot_path(root, manifest.manifest_fingerprint)
    existing = _assert_existing_snapshot_identity(path=snapshot_path, expected_identity=identity)

    finalized_at = (now_fn() if now_fn is not None else datetime.now().astimezone()).isoformat()
    if existing is None:
        snapshot_payload = build_provider_snapshot_payload(
            manifest=manifest,
            artifacts=evidence,
            finalized_at=finalized_at,
        )
        atomic_write_json(snapshot_path, snapshot_payload)
        reused = False
    else:
        snapshot_payload = existing
        reused = True

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = (now_fn() if now_fn is not None else datetime.now().astimezone()).strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"{PROVIDER_SNAPSHOT_REPORT_PREFIX}{stamp}.json"
    md_path = out_dir / f"{PROVIDER_SNAPSHOT_REPORT_PREFIX}{stamp}.md"
    report_payload: dict[str, object] = {
        **snapshot_payload,
        "verified_at": (now_fn() if now_fn is not None else datetime.now().astimezone()).isoformat(),
        "reused_existing_snapshot": reused,
        "verified_requests": manifest.total_requests,
        "dataset_count": len(snapshot_payload.get("datasets") or []),
        "provider_snapshot_path": project_relative_display_path(snapshot_path, project_root=root),
        "archive_dir": project_relative_display_path(archive_dir, project_root=root),
        "ledger_path": project_relative_display_path(ledger_path, project_root=root),
    }
    atomic_write_json(json_path, report_payload)
    atomic_write_text(md_path, _provider_snapshot_markdown(report_payload))
    return {
        **report_payload,
        "json_path": str(json_path),
        "markdown_path": str(md_path),
    }


__all__ = [
    "PROVIDER_SNAPSHOT_REPORT_PREFIX",
    "MarketDataBootstrapCompletionError",
    "finalize_market_data_v2_provider_snapshot",
]
