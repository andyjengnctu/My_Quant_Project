"""Canonical Trading market-data update orchestration.

Workbench and Smart Downloader call this single service.  The execution-critical
legacy CSV dataset remains the current ``full_rule_based_no_dl`` truth, while one
process-local FinMind request cache lets that producer and the V2 sidecar reuse
identical provider responses without creating another persistent data truth.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from core.runtime_domains import RUNTIME_DOMAIN_TRADING, resolve_runtime_domain_paths
from core.trading_identity import normalize_trading_ticker
from services.downloader.application import run_trading_dataset_update
from services.trading.account_state import load_trading_account_state
from services.trading.market_data_state import publish_trading_market_data_snapshot


def _build_shared_finmind_client(*, token: str):
    resolved = str(token or "").strip()
    if not resolved:
        return None
    from config.market_data import MARKET_DATA_V2_HTTP_TIMEOUT_SEC
    from services.downloader.finmind_http import FinMindHttpClient
    from services.downloader.finmind_shared_client import SharedFinMindRequestClient

    return SharedFinMindRequestClient(
        FinMindHttpClient(token=resolved, timeout_sec=MARKET_DATA_V2_HTTP_TIMEOUT_SEC)
    )


def run_trading_market_data_update(
    *,
    project_root: str | Path,
    provider_client=None,
    sync_v2_archive: bool = True,
) -> dict[str, Any]:
    """Run the single canonical Trading data update path.

    Order remains execution-semantic:
      1) refresh the execution-critical canonical CSV dataset;
      2) publish the canonical Trading market-data snapshot;
      3) maintain the optional V2 Trading archive sidecar.

    One process-local provider client/cache spans steps 1 and 3.  This changes
    request geometry only; the CSV output remains execution truth and V2 remains
    non-blocking for the active rule-based strategy.
    """

    root = Path(project_root).resolve()
    paths = resolve_runtime_domain_paths(root, domain=RUNTIME_DOMAIN_TRADING)
    from services.downloader import runtime as downloader_runtime

    if Path(downloader_runtime.SAVE_DIR).resolve() != Path(paths.data_dir).resolve():
        raise RuntimeError("Trading downloader SAVE_DIR與runtime-domain data truth不一致")

    account = load_trading_account_state(root, required=False)
    required_position_tickers = sorted(
        normalize_trading_ticker(ticker)
        for ticker, record in ((account or {}).get("positions") or {}).items()
        if int(((record or {}).get("broker") or {}).get("qty") or 0) > 0
    )

    token = None
    if provider_client is None or sync_v2_archive:
        token = downloader_runtime.resolve_finmind_api_token(project_root=root)
    shared_client = provider_client if provider_client is not None else _build_shared_finmind_client(token=str(token or ""))
    data_before_legacy = int(getattr(shared_client, "data_request_count", 0)) if shared_client is not None else 0
    usage_before_legacy = int(getattr(shared_client, "usage_request_count", 0)) if shared_client is not None else 0

    result = dict(
        run_trading_dataset_update(
            required_tickers=required_position_tickers,
            provider_client=shared_client,
        )
    )
    if str(result.get("runtime_domain") or "") != RUNTIME_DOMAIN_TRADING:
        raise RuntimeError("Trading downloader回傳的runtime domain不合法")

    data_after_legacy = int(getattr(shared_client, "data_request_count", 0)) if shared_client is not None else 0
    usage_after_legacy = int(getattr(shared_client, "usage_request_count", 0)) if shared_client is not None else 0

    snapshot = publish_trading_market_data_snapshot(
        root,
        market_date=result.get("market_date"),
        required_position_tickers=required_position_tickers,
    )

    if sync_v2_archive:
        from core.market_data_trading_sync_policy import get_market_data_trading_sync_policy
        from services.downloader.market_data_trading_sync import sync_market_data_v2_trading_archive

        v2_output_dir = Path(downloader_runtime.OUTPUT_DIR) / "market_data_v2" / "trading_sync"
        v2_policy = get_market_data_trading_sync_policy()
        try:
            v2_archive = dict(
                sync_market_data_v2_trading_archive(
                    project_root=root,
                    target_date=str(result.get("market_date") or ""),
                    token=token,
                    output_dir=v2_output_dir,
                    client=shared_client,
                )
            )
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            if v2_policy.execution_fail_closed:
                raise
            error = f"{type(exc).__name__}: {exc}"
            try:
                from services.trading.market_data_v2_state import publish_trading_market_data_v2_failure

                publish_trading_market_data_v2_failure(
                    root,
                    target_date=str(result.get("market_date") or ""),
                    error=error,
                )
            except (OSError, ValueError, RuntimeError, TypeError) as state_exc:
                error = f"{error}; V2 state persist failed: {type(state_exc).__name__}: {state_exc}"
            v2_archive = {
                "status": "STALE",
                "execution_blocking": False,
                "target_date": str(result.get("market_date") or ""),
                "error": error,
            }
    else:
        v2_archive = {
            "status": "DEFERRED_TO_DUE_PLANNER",
            "execution_blocking": False,
            "target_date": str(result.get("market_date") or ""),
            "process_data_requests": 0,
            "process_usage_requests": 0,
        }

    snapshot_fn = getattr(shared_client, "snapshot", None) if shared_client is not None else None
    if callable(snapshot_fn):
        shared_snapshot = snapshot_fn()
    else:
        shared_snapshot = {
            "provider_data_requests": data_after_legacy if shared_client is not None else None,
            "provider_usage_requests": usage_after_legacy if shared_client is not None else None,
            "cache_hits": 0,
            "cache_misses": 0,
            "uncached_fetches": 0,
            "seeded_entries": 0,
            "cached_entries": 0,
        }
    return {
        **result,
        "market_data_snapshot_fingerprint": snapshot["snapshot_fingerprint"],
        "dataset_content_sha256": snapshot["dataset_fingerprint"]["csv_content_sha256"],
        "market_data_v2_archive": v2_archive,
        "provider_request_session": {
            **shared_snapshot,
            "legacy_stage_data_requests": data_after_legacy - data_before_legacy if shared_client is not None else None,
            "legacy_stage_usage_requests": usage_after_legacy - usage_before_legacy if shared_client is not None else None,
            "v2_stage_data_requests": v2_archive.get("process_data_requests"),
            "v2_stage_usage_requests": v2_archive.get("process_usage_requests"),
        },
    }


def plan_trading_market_data_due_update(
    *,
    project_root: str | Path,
    target_date: str,
    now,
):
    """Build the local-only dataset due plan; this function performs no provider calls."""

    from services.trading.market_data_dataset_state import refresh_market_data_due_state

    plan, _state = refresh_market_data_due_state(
        Path(project_root).resolve(),
        target_date=str(target_date),
        now=now,
    )
    return plan


__all__ = ["run_trading_market_data_update", "plan_trading_market_data_due_update"]
