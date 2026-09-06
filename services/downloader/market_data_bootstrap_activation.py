"""Explicit activation seam for the Market Data V2 full bootstrap.

This module is the only runtime bridge from a persisted READY Backer preflight
artifact to the deterministic request manifest, persistent SQLite ledger,
quota-aware executor and atomic Parquet storage sink.  Merely importing this
module or running the normal Trading downloader never starts the full bootstrap.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Callable

from core.file_integrity import atomic_write_json, atomic_write_text, load_json_strict
from core.market_data_bootstrap_requests import (
    BootstrapRequestManifest,
    build_bootstrap_request_manifest,
)
from core.market_data_dataset_registry import get_market_dataset_specs, validate_market_dataset_registry
from core.market_data_execution_policy import MarketDataExecutionPolicy, get_market_data_execution_policy
from core.market_data_instrument_universe import historical_stock_etf_universe_contract_fingerprint
from core.market_data_storage_contract import (
    resolve_market_data_bootstrap_archive_dir,
    resolve_market_data_bootstrap_ledger_path,
)
from core.path_utils import project_relative_display_path
from services.downloader.finmind_http import FinMindHttpClient
from services.downloader.market_data_executor import MarketDataBootstrapExecutor
from services.downloader.market_data_ledger import LedgerSummary, MarketDataJobLedger
from services.downloader.market_data_storage import MarketDataBootstrapStorageSink

PREFLIGHT_REPORT_GLOB = "market_data_v2_preflight_plan_*.json"
BOOTSTRAP_STATUS_PREFIX = "market_data_v2_bootstrap_status_"
SUPPORTED_PREFLIGHT_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class MarketDataBootstrapActivation:
    preflight_path: Path
    manifest: BootstrapRequestManifest
    preflight_quota_limit: int
    planned_total_requests: int


class MarketDataBootstrapActivationError(RuntimeError):
    pass


def _require_dict(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise MarketDataBootstrapActivationError(f"{field} 必須是 object")
    return value


def _require_list(value: object, *, field: str) -> list[object]:
    if not isinstance(value, list):
        raise MarketDataBootstrapActivationError(f"{field} 必須是 list")
    return value


def _latest_preflight_path(output_dir) -> Path:
    root = Path(output_dir)
    paths = sorted(path for path in root.glob(PREFLIGHT_REPORT_GLOB) if path.is_file())
    if not paths:
        raise MarketDataBootstrapActivationError(
            "找不到 Market Data V2 Preflight evidence；請先由 Smart Downloader 執行 Preflight。"
        )
    return paths[-1]


def _probe_index(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    probes = _require_list(payload.get("probes"), field="preflight.probes")
    index: dict[str, dict[str, object]] = {}
    for raw in probes:
        item = _require_dict(raw, field="preflight.probes[]")
        dataset = str(item.get("dataset") or "").strip()
        if not dataset:
            raise MarketDataBootstrapActivationError("preflight probe 缺少 dataset")
        if dataset in index:
            raise MarketDataBootstrapActivationError(f"preflight probe dataset 重複: {dataset}")
        index[dataset] = item
    return index


def prepare_market_data_v2_bootstrap_activation(*, output_dir) -> MarketDataBootstrapActivation:
    """Resolve the latest preflight and re-derive its manifest from current SSOT.

    The newest preflight report is authoritative for activation intent.  A newer
    BLOCKED report therefore cannot be bypassed by silently reusing an older READY
    report.  Registry or request-manifest drift requires a new preflight.
    """

    validate_market_dataset_registry()
    path = _latest_preflight_path(output_dir)
    try:
        raw_payload = load_json_strict(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise MarketDataBootstrapActivationError(
            f"最新 Market Data Preflight 無法讀取: {type(exc).__name__}: {exc}"
        ) from exc
    payload = _require_dict(raw_payload, field="preflight")

    if int(payload.get("schema_version", -1)) != SUPPORTED_PREFLIGHT_SCHEMA_VERSION:
        raise MarketDataBootstrapActivationError(
            f"Preflight schema_version 不支援: {payload.get('schema_version')!r}"
        )
    if str(payload.get("status") or "").strip() != "READY":
        raise MarketDataBootstrapActivationError(
            "最新 Market Data Preflight 不是 READY；請先修正 blocking probes 並重新執行 Preflight。"
        )
    failures = _require_list(payload.get("probe_failures"), field="preflight.probe_failures")
    if failures:
        raise MarketDataBootstrapActivationError("READY Preflight 不得同時含 blocking probe failures")
    if payload.get("plan_error") not in (None, ""):
        raise MarketDataBootstrapActivationError("READY Preflight 不得同時含 plan_error")

    current_universe_contract = historical_stock_etf_universe_contract_fingerprint()
    preflight_universe_contract = str(
        payload.get("historical_universe_contract_fingerprint") or ""
    ).strip()
    if preflight_universe_contract != current_universe_contract:
        raise MarketDataBootstrapActivationError(
            "Preflight historical universe contract 已 drift；請重新執行 Preflight。"
        )

    plan = _require_dict(payload.get("plan"), field="preflight.plan")
    instruments_raw = _require_list(
        payload.get("historical_instruments"), field="preflight.historical_instruments"
    )
    instruments = tuple(str(value or "").strip() for value in instruments_raw if str(value or "").strip())
    if not instruments or len(set(instruments)) != len(instruments):
        raise MarketDataBootstrapActivationError("Preflight historical instrument universe 空白或重複")
    as_of_date = str(payload.get("as_of_date") or "").strip()
    if not as_of_date:
        raise MarketDataBootstrapActivationError("Preflight 缺少 as_of_date")

    specs = get_market_dataset_specs(included_only=True)
    probes = _probe_index(payload)
    expected_datasets = {spec.dataset for spec in specs}
    missing = sorted(expected_datasets.difference(probes))
    extra = sorted(set(probes).difference(expected_datasets))
    if missing or extra:
        raise MarketDataBootstrapActivationError(
            f"Preflight dataset roster 與 current registry 不一致: missing={missing}, extra={extra}"
        )

    manifest = build_bootstrap_request_manifest(
        specs=specs,
        historical_instruments=instruments,
        evidence_by_dataset=probes,
        as_of_date=as_of_date,
    )

    checks = {
        "registry_fingerprint": (str(plan.get("registry_fingerprint") or ""), manifest.registry_fingerprint),
        "manifest_fingerprint": (str(plan.get("manifest_fingerprint") or ""), manifest.manifest_fingerprint),
        "total_requests": (int(plan.get("total_requests") or -1), manifest.total_requests),
        "historical_instrument_count": (
            int(payload.get("historical_instrument_count") or -1),
            manifest.historical_instrument_count,
        ),
        "included_dataset_count": (int(payload.get("included_dataset_count") or -1), len(specs)),
    }
    mismatches = [
        f"{name}: preflight={actual!r}, current={expected!r}"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    if mismatches:
        raise MarketDataBootstrapActivationError(
            "Preflight/registry/request manifest 已 drift，禁止啟動 bootstrap；請重新 Preflight。 "
            + " | ".join(mismatches)
        )

    quota = _require_dict(payload.get("quota"), field="preflight.quota")
    try:
        quota_limit = int(quota["api_request_limit"])
        plan_quota_limit = int(plan["quota_limit"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MarketDataBootstrapActivationError("Preflight 缺少合法 api_request_limit") from exc
    if quota_limit <= 0 or plan_quota_limit != quota_limit:
        raise MarketDataBootstrapActivationError(
            f"Preflight quota identity 不一致: quota={quota_limit}, plan={plan_quota_limit}"
        )

    return MarketDataBootstrapActivation(
        preflight_path=path,
        manifest=manifest,
        preflight_quota_limit=quota_limit,
        planned_total_requests=manifest.total_requests,
    )


def get_existing_bootstrap_summary(*, project_root, activation: MarketDataBootstrapActivation) -> LedgerSummary | None:
    ledger_path = resolve_market_data_bootstrap_ledger_path(
        project_root, activation.manifest.manifest_fingerprint
    )
    if not ledger_path.is_file():
        return None
    ledger = MarketDataJobLedger(ledger_path)
    workload_id = ledger.workload_id_for_manifest(activation.manifest)
    try:
        return ledger.get_summary(workload_id)
    except ValueError:
        # A ledger file without this manifest is not valid resume evidence.  Do
        # not seed or mutate it just for a read-only status preview.
        return None


class _ProgressStorageSink:
    def __init__(
        self,
        *,
        sink,
        initial_done: int,
        total: int,
        every: int,
        progress_fn: Callable[[dict[str, object]], None] | None,
        progress_context_fn: Callable[[], dict[str, object]] | None = None,
    ):
        self._sink = sink
        self._done = int(initial_done)
        self._total = int(total)
        self._every = max(1, int(every))
        self._progress_fn = progress_fn
        self._progress_context_fn = progress_context_fn

    def _record(self, request, *, recovered: bool) -> None:
        self._done += 1
        if self._progress_fn is None:
            return
        if self._done == self._total or self._done % self._every == 0:
            event: dict[str, object] = {
                "done": self._done,
                "total": self._total,
                "dataset": request.dataset,
                "data_id": request.data_id,
                "recovered": bool(recovered),
            }
            if self._progress_context_fn is not None:
                event.update(self._progress_context_fn())
            self._progress_fn(event)

    def recover_committed(self, request):
        recover_fn = getattr(self._sink, "recover_committed", None)
        if not callable(recover_fn):
            return None
        receipt = recover_fn(request)
        if receipt is not None:
            self._record(request, recovered=True)
        return receipt

    def __call__(self, request, frame):
        receipt = self._sink(request, frame)
        self._record(request, recovered=False)
        return receipt


def _bootstrap_status_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Market Data V2 Bootstrap Status",
        "",
        f"- status: `{payload.get('status')}`",
        f"- as_of_date: `{payload.get('as_of_date')}`",
        f"- manifest_fingerprint: `{payload.get('manifest_fingerprint')}`",
        f"- total requests: **{payload.get('total')}**",
        f"- done: **{payload.get('done')}**",
        f"- unfinished: **{payload.get('unfinished')}**",
        f"- blocked: **{payload.get('blocked')}**",
        f"- cumulative data HTTP attempts: **{payload.get('http_attempts')}**",
        f"- this-process data requests: **{payload.get('process_data_requests')}**",
        f"- preflight: `{payload.get('preflight_path')}`",
        f"- archive: `{payload.get('archive_dir')}`",
        f"- ledger: `{payload.get('ledger_path')}`",
        "",
    ]
    return "\n".join(lines)


def execute_market_data_v2_bootstrap(
    *,
    activation: MarketDataBootstrapActivation,
    token: str,
    project_root,
    output_dir,
    timeout_sec: float = 30.0,
    client: FinMindHttpClient | None = None,
    sink=None,
    policy: MarketDataExecutionPolicy | None = None,
    now_fn: Callable[[], datetime] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
    progress_fn: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    """Execute/resume one explicitly prepared manifest until DONE or BLOCKED."""

    manifest = activation.manifest
    resolved_policy = policy or get_market_data_execution_policy()
    root = Path(project_root).resolve()
    archive_dir = resolve_market_data_bootstrap_archive_dir(root, manifest.manifest_fingerprint)
    ledger_path = resolve_market_data_bootstrap_ledger_path(root, manifest.manifest_fingerprint)
    ledger = MarketDataJobLedger(ledger_path)
    workload_id = ledger.seed_manifest(manifest, now=now_fn() if now_fn is not None else None)
    before = ledger.get_summary(workload_id)

    http = client or FinMindHttpClient(token=token, timeout_sec=timeout_sec)
    storage = sink or MarketDataBootstrapStorageSink(project_root=root, manifest=manifest)
    readiness_fn = getattr(storage, "validate_activation_readiness", None)
    if callable(readiness_fn):
        readiness_fn()
    executor = MarketDataBootstrapExecutor(
        ledger=ledger,
        client=http,
        policy=resolved_policy,
        now_fn=now_fn,
        sleep_fn=sleep_fn,
    )
    progress_sink = _ProgressStorageSink(
        sink=storage,
        initial_done=before.done,
        total=manifest.total_requests,
        every=resolved_policy.progress_every_committed_requests,
        progress_fn=progress_fn,
        progress_context_fn=executor.quota_progress_snapshot,
    )
    summary = executor.run(manifest=manifest, sink=progress_sink)

    finished_at = now_fn() if now_fn is not None else datetime.now().astimezone()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = finished_at.strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"{BOOTSTRAP_STATUS_PREFIX}{stamp}.json"
    md_path = out_dir / f"{BOOTSTRAP_STATUS_PREFIX}{stamp}.md"
    payload: dict[str, object] = {
        "schema_version": 1,
        "status": summary.workload_status,
        "generated_at": finished_at.isoformat(),
        "as_of_date": manifest.as_of_date,
        "registry_fingerprint": manifest.registry_fingerprint,
        "manifest_fingerprint": manifest.manifest_fingerprint,
        "workload_id": summary.workload_id,
        "total": summary.total,
        "pending": summary.pending,
        "running": summary.running,
        "retryable": summary.retryable,
        "done": summary.done,
        "unfinished": summary.unfinished,
        "blocked": summary.blocked,
        "http_attempts": summary.http_attempts,
        "process_data_requests": int(http.data_request_count),
        "process_usage_requests": int(http.usage_request_count),
        "preflight_path": project_relative_display_path(activation.preflight_path, project_root=root),
        "archive_dir": project_relative_display_path(archive_dir, project_root=root),
        "ledger_path": project_relative_display_path(ledger_path, project_root=root),
    }
    atomic_write_json(json_path, payload)
    atomic_write_text(md_path, _bootstrap_status_markdown(payload))
    return {
        **payload,
        "json_path": str(json_path),
        "markdown_path": str(md_path),
    }


__all__ = [
    "PREFLIGHT_REPORT_GLOB",
    "MarketDataBootstrapActivation",
    "MarketDataBootstrapActivationError",
    "prepare_market_data_v2_bootstrap_activation",
    "get_existing_bootstrap_summary",
    "execute_market_data_v2_bootstrap",
]
