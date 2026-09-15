"""Provider-free promotion/reconciliation for Trading V2 execution consumer state."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from core.trading_identity import normalize_trading_ticker
from core.trading_policy import get_trading_strategy_profile
from services.market_data.provider_snapshot_repository import find_latest_ready_provider_snapshot
from services.trading.account_state import load_trading_account_state
from services.trading.data_readiness import build_trading_data_readiness_from_evidence
from services.trading.live_reentry import resolve_trading_live_reentry_required_tickers
from services.trading.market_data_consumer import (
    load_trading_v2_consumer_state,
    publish_trading_v2_consumer_state,
)
from services.trading.market_data_dataset_state import load_market_data_dataset_state
from services.trading.market_data_v2_state import resolve_trading_market_data_update_target_date


def reconcile_trading_v2_consumer_state_from_local_evidence(
    project_root: str | Path,
    *,
    now=None,
) -> dict[str, Any]:
    """Reconcile candidate target and finalized consumer without provider calls."""

    root = Path(project_root).resolve()
    target_date = resolve_trading_market_data_update_target_date(root, now=now)
    if not target_date:
        prior = load_trading_v2_consumer_state(root, required=False, verify_current_view=False)
        return {
            "promoted": False,
            "reason": "NO_TARGET",
            "target_date": None,
            "market_date": None if prior is None else prior.get("market_date"),
            "provider_calls": 0,
        }
    result = promote_trading_v2_consumer_state_if_ready(
        root,
        target_date=str(target_date),
    )
    return {
        **dict(result),
        "target_date": str(target_date),
        "provider_calls": 0,
    }


def promote_trading_v2_consumer_state_if_ready(
    project_root: str | Path,
    *,
    target_date: str,
) -> dict[str, Any]:
    """Finalize a newer execution target once active dependencies are READY."""

    root = Path(project_root).resolve()
    candidate = str(target_date or "").strip()
    if not candidate:
        return {"promoted": False, "reason": "NO_TARGET", "market_date": None}
    if find_latest_ready_provider_snapshot(root) is None:
        return {"promoted": False, "reason": "NO_PROVIDER_SNAPSHOT", "market_date": None}

    prior = load_trading_v2_consumer_state(root, required=False, verify_current_view=False)
    prior_date = None if prior is None else str(prior.get("market_date") or "").strip() or None
    if prior_date is not None and prior_date >= candidate:
        return {"promoted": False, "reason": "ALREADY_FINALIZED", "market_date": prior_date}

    profile = get_trading_strategy_profile()
    dataset_state = load_market_data_dataset_state(root, required=False)
    readiness = build_trading_data_readiness_from_evidence(
        strategy_id=profile.strategy_id,
        target_date=candidate,
        consumer_state_ready=True,
        dataset_state=dataset_state,
        consumer_state_required=False,
    )
    if not bool(readiness.get("ready")):
        return {
            "promoted": False,
            "reason": "DEPENDENCIES_NOT_READY",
            "market_date": prior_date,
            "blocking_dependencies": list(readiness.get("blocking_dependencies") or []),
        }

    account = load_trading_account_state(root, required=False)
    required_position_tickers = sorted(
        normalize_trading_ticker(ticker)
        for ticker, record in ((account or {}).get("positions") or {}).items()
        if int(((record or {}).get("broker") or {}).get("qty") or 0) > 0
    )
    required_reentry_tickers = resolve_trading_live_reentry_required_tickers(root)
    state = publish_trading_v2_consumer_state(
        root,
        market_date=candidate,
        required_position_tickers=required_position_tickers,
        required_reentry_tickers=required_reentry_tickers,
    )
    return {
        "promoted": True,
        "reason": "PROMOTED",
        "market_date": str(state["market_date"]),
        "consumer_state_fingerprint": str(state["state_fingerprint"]),
        "source_view_fingerprint": str(state["source_view_fingerprint"]),
    }


__all__ = [
    "reconcile_trading_v2_consumer_state_from_local_evidence",
    "promote_trading_v2_consumer_state_if_ready",
]
