"""Canonical Trading market-data update orchestration.

Both Workbench and Smart Downloader must call this single service.  The legacy
execution-critical CSV producer remains authoritative for the current
``full_rule_based_no_dl`` path; the Market Data V2 archive is maintained as an
isolated non-blocking Trading sidecar after the canonical CSV snapshot is
published.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from core.runtime_domains import RUNTIME_DOMAIN_TRADING, resolve_runtime_domain_paths
from core.trading_identity import normalize_trading_ticker
from services.downloader.application import run_trading_dataset_update
from services.trading.account_state import load_trading_account_state
from services.trading.market_data_state import publish_trading_market_data_snapshot


def run_trading_market_data_update(*, project_root: str | Path) -> dict[str, Any]:
    """Run the single canonical Trading data update path.

    Order is intentional and execution-semantic:
      1) update the execution-critical canonical CSV dataset;
      2) publish the canonical Trading market-data snapshot;
      3) maintain the optional V2 Trading archive sidecar.

    A V2 sidecar failure remains non-blocking while the active Trading strategy
    is ``full_rule_based_no_dl``.  Future strategies may only change that
    behavior through an explicit V2 dependency contract.
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
    result = dict(run_trading_dataset_update(required_tickers=required_position_tickers))
    if str(result.get("runtime_domain") or "") != RUNTIME_DOMAIN_TRADING:
        raise RuntimeError("Trading downloader回傳的runtime domain不合法")

    snapshot = publish_trading_market_data_snapshot(
        root,
        market_date=result.get("market_date"),
        required_position_tickers=required_position_tickers,
    )

    from core.market_data_trading_sync_policy import get_market_data_trading_sync_policy
    from services.downloader.market_data_trading_sync import sync_market_data_v2_trading_archive

    v2_output_dir = Path(downloader_runtime.OUTPUT_DIR) / "market_data_v2" / "trading_sync"
    token = downloader_runtime.resolve_finmind_api_token(project_root=root)
    v2_policy = get_market_data_trading_sync_policy()
    try:
        v2_archive = dict(
            sync_market_data_v2_trading_archive(
                project_root=root,
                target_date=str(result.get("market_date") or ""),
                token=token,
                output_dir=v2_output_dir,
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

    return {
        **result,
        "market_data_snapshot_fingerprint": snapshot["snapshot_fingerprint"],
        "dataset_content_sha256": snapshot["dataset_fingerprint"]["csv_content_sha256"],
        "market_data_v2_archive": v2_archive,
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
