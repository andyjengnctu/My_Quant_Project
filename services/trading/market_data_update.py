"""Canonical Trading market-data update orchestration during V2 cutover.

Market Data V2 is provider-authoritative.  Existing rule-based consumers may
still read the six-column Trading CSV dataset, but that dataset is now only a
local compatibility materialization from the verified V2 historical/latest view.
No Legacy CSV producer is allowed to call FinMind on this path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from core.runtime_domains import RUNTIME_DOMAIN_TRADING, resolve_runtime_domain_paths
from core.trading_identity import normalize_trading_ticker
from services.trading.account_state import load_trading_account_state
from services.trading.market_data_compatibility import materialize_trading_v2_compatibility_dataset
from services.trading.market_data_state import publish_trading_market_data_snapshot


def run_trading_market_data_update(
    *,
    project_root: str | Path,
    provider_client=None,
    sync_v2_archive: bool = True,
) -> dict[str, Any]:
    """Refresh V2 first, then rebuild the transitional CSV compatibility view.

    ``sync_v2_archive=False`` is retained for local/test callers that explicitly
    want to materialize only from already-verified V2 state.
    """

    root = Path(project_root).resolve()
    paths = resolve_runtime_domain_paths(root, domain=RUNTIME_DOMAIN_TRADING)
    account = load_trading_account_state(root, required=False)
    required_position_tickers = sorted(
        normalize_trading_ticker(ticker)
        for ticker, record in ((account or {}).get("positions") or {}).items()
        if int(((record or {}).get("broker") or {}).get("qty") or 0) > 0
    )

    if sync_v2_archive:
        from services.trading.market_data_auto_update import run_trading_market_data_auto_update

        v2_update = dict(
            run_trading_market_data_auto_update(
                project_root=root,
                client=provider_client,
                force_market_date_discovery=True,
            )
        )
        if str(v2_update.get("status") or "") in {"NO_TARGET", "BLOCKED", "DISABLED"}:
            raise RuntimeError(
                "Trading V2 update 尚未提供可 materialize 的 canonical market-data truth；"
                f"status={v2_update.get('status')} error={v2_update.get('error') or '-'}"
            )
    else:
        v2_update = {
            "status": "LOCAL_V2_ONLY",
            "provider_requests_required": False,
            "data_requests": 0,
            "usage_requests": 0,
        }

    target_date = str(v2_update.get("target_date") or "").strip() or None
    result = dict(
        materialize_trading_v2_compatibility_dataset(
            root,
            required_tickers=required_position_tickers,
            market_date=target_date,
        )
    )
    if str(result.get("runtime_domain") or "") != RUNTIME_DOMAIN_TRADING:
        raise RuntimeError("Trading V2 compatibility materialization runtime domain 不合法")
    if Path(result.get("data_dir") or "").is_absolute():
        raise RuntimeError("Trading V2 compatibility data_dir user-facing path 必須為 project-relative")
    if Path(paths.data_dir).resolve() == root.resolve():
        raise RuntimeError("Trading runtime data_dir 不可解析為 project root")

    snapshot = publish_trading_market_data_snapshot(
        root,
        market_date=result["market_date"],
        required_position_tickers=required_position_tickers,
        current_universe_tickers=list(result["current_execution_pool_tickers"]),
    )
    return {
        **result,
        "status": "READY",
        "market_data_snapshot_fingerprint": snapshot["snapshot_fingerprint"],
        "dataset_content_sha256": snapshot["dataset_fingerprint"]["csv_content_sha256"],
        "market_data_v2_archive": v2_update,
        "provider_request_session": {
            "legacy_stage_data_requests": 0,
            "legacy_stage_usage_requests": 0,
            "compatibility_materialization_provider_calls": 0,
            "v2_stage_data_requests": int(v2_update.get("data_requests") or 0),
            "v2_stage_usage_requests": int(v2_update.get("usage_requests") or 0),
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
