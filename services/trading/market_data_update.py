"""Canonical Trading Market Data V2 update orchestration.

The updater owns the production write sequence after the V2-only cutover:
refresh/verify the Trading V2 archive and then publish the V2 execution consumer
state. Legacy six-column Trading CSV/snapshot artifacts are not produced here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from core.trading_data_dependencies import get_trading_data_dependency_spec
from core.trading_identity import normalize_trading_ticker
from core.trading_policy import get_trading_strategy_profile
from services.trading.account_state import load_trading_account_state
from services.trading.live_reentry import resolve_trading_live_reentry_required_tickers
from services.trading.market_data_consumer import publish_trading_v2_consumer_state
from services.trading.market_data_v2_view import TradingMarketDataV2View


def _resolve_consumer_market_date(project_root: Path, update_result: dict[str, Any]) -> str:
    """Resolve the latest execution-safe date from the exact Trading dependencies.

    A newly discovered provider target may be ahead of one or more required
    datasets while they are still waiting for publication.  Consumer state must
    therefore never advance past the common READY horizon merely because the
    archive target advanced.
    """

    profile = get_trading_strategy_profile()
    spec = get_trading_data_dependency_spec(profile.strategy_id)
    view = TradingMarketDataV2View.open(project_root)
    horizon = view.training_horizon(required_datasets=spec.required_v2_datasets)
    ready_through = str(horizon.training_through_date or "").strip()
    if not ready_through:
        raise RuntimeError("Trading V2 required datasets 尚未形成可發布 consumer state 的共同 READY horizon")
    target_date = str(update_result.get("target_date") or "").strip()
    if not target_date:
        return ready_through
    return min(target_date, ready_through)


def run_trading_market_data_update(
    *,
    project_root: str | Path,
    provider_client=None,
    sync_v2_archive: bool = True,
    progress_fn=None,
    quota_wait_fn=None,
) -> dict[str, Any]:
    """Refresh V2 if requested, then publish canonical execution consumer state."""

    root = Path(project_root).resolve()
    account = load_trading_account_state(root, required=False)
    required_position_tickers = sorted(
        normalize_trading_ticker(ticker)
        for ticker, record in ((account or {}).get("positions") or {}).items()
        if int(((record or {}).get("broker") or {}).get("qty") or 0) > 0
    )

    required_reentry_tickers = resolve_trading_live_reentry_required_tickers(root)

    if sync_v2_archive:
        from services.trading.market_data_auto_update import run_trading_market_data_auto_update

        v2_update = dict(
            run_trading_market_data_auto_update(
                project_root=root,
                client=provider_client,
                force_market_date_discovery=True,
                refresh_provider_quota=True,
                progress_fn=progress_fn,
                quota_wait_fn=quota_wait_fn,
            )
        )
        if str(v2_update.get("status") or "") in {"NO_TARGET", "BLOCKED", "DISABLED"}:
            raise RuntimeError(
                "Trading V2 update 尚未提供可發布 consumer state 的 canonical market-data truth；"
                f"status={v2_update.get('status')} error={v2_update.get('error') or '-'}"
            )
    else:
        v2_update = {
            "status": "LOCAL_V2_ONLY",
            "provider_requests_required": False,
            "data_requests": 0,
            "usage_requests": 0,
        }

    market_date = _resolve_consumer_market_date(root, v2_update)
    consumer_state = publish_trading_v2_consumer_state(
        root,
        market_date=market_date,
        required_position_tickers=required_position_tickers,
        required_reentry_tickers=required_reentry_tickers,
    )
    return {
        "status": "READY",
        "runtime_domain": "trading",
        "market_date": str(consumer_state["market_date"]),
        "market_data_source": str(consumer_state["source"]),
        "market_data_consumer_state_fingerprint": str(consumer_state["state_fingerprint"]),
        "market_data_consumer_source_view_fingerprint": str(consumer_state["source_view_fingerprint"]),
        "current_execution_pool_tickers": list(consumer_state["current_execution_pool_tickers"]),
        "current_execution_pool_ticker_count": int(consumer_state["current_execution_pool_ticker_count"]),
        "required_position_tickers": list(consumer_state["required_position_tickers"]),
        "required_position_ticker_count": len(consumer_state["required_position_tickers"]),
        "required_reentry_tickers": list(consumer_state.get("required_reentry_tickers") or []),
        "required_reentry_ticker_count": len(consumer_state.get("required_reentry_tickers") or []),
        "training_tickers": list(consumer_state["training_tickers"]),
        "training_ticker_count": int(consumer_state["training_ticker_count"]),
        "market_data_v2_archive": v2_update,
        "provider_request_session": {
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
