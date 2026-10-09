"""Canonical Trading Market Data V2 update orchestration.

The updater owns the production write sequence after the V2-only cutover:
refresh/verify the Trading V2 archive and then publish the V2 execution consumer
state. Legacy six-column Trading CSV/snapshot artifacts are not produced here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from core.trading_data_dependencies import get_trading_data_dependency_spec
from core.trading_policy import get_trading_strategy_profile
from services.trading.data_readiness import build_trading_data_readiness_from_evidence
from services.trading.market_data_consumer import load_trading_v2_consumer_state
from services.trading.market_data_consumer_promotion import promote_trading_v2_consumer_state_if_ready
from services.trading.market_data_dataset_state import load_market_data_dataset_state
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


def _archive_blocker_detail(update: dict[str, Any]) -> str:
    """Explain incomplete archive evidence without changing Trading eligibility."""

    required = tuple(str(item) for item in (update.get("trading_blocking_datasets") or ()))
    incomplete = tuple(str(item) for item in (update.get("incomplete_datasets") or ()))
    rows = tuple(
        dict(item) for item in (update.get("archive_incomplete_datasets") or ())
        if isinstance(item, dict)
    )
    blocked_rows = [item for item in rows if str(item.get("status") or "") == "BLOCKED"]
    details = [
        f"{row.get('dataset')} ({row.get('last_error') or row.get('status') or 'NOT_READY'})"
        for row in blocked_rows[:3]
    ]
    parts = [f"archive_status={update.get('status') or '-'}"]
    if required:
        parts.append("required_not_ready=" + ",".join(required))
    if incomplete:
        parts.append("incomplete_datasets=" + ",".join(incomplete[:5]))
    if details:
        parts.append("archive_blocked=" + "; ".join(details))
    if update.get("error"):
        parts.append("error=" + str(update["error"]))
    if update.get("ledger_path"):
        parts.append("ledger=" + str(update["ledger_path"]))
    return " | ".join(parts)


def run_trading_market_data_update(
    *,
    project_root: str | Path,
    provider_client=None,
    sync_v2_archive: bool = True,
    progress_fn=None,
    quota_wait_fn=None,
) -> dict[str, Any]:
    """Force-probe the archive; publish only a verified Trading consumer horizon.

    An archive workload can be BLOCKED by one of 51 datasets unrelated to the
    active strategy.  The archive status is preserved as diagnostic evidence,
    but only the six canonical dependencies control execution eligibility.
    """

    root = Path(project_root).resolve()
    if sync_v2_archive:
        from services.trading.market_data_auto_update import run_trading_market_data_auto_update

        v2_update = dict(
            run_trading_market_data_auto_update(
                project_root=root,
                client=provider_client,
                force_market_date_discovery=True,
                force_refresh_current_target=True,
                refresh_provider_quota=True,
                progress_fn=progress_fn,
                quota_wait_fn=quota_wait_fn,
            )
        )
        v2_status = str(v2_update.get("status") or "").strip().upper()
        if v2_status == "BUSY":
            raise RuntimeError(
                "Trading V2 Full Update 未執行：另一個 Market Data worker 正在執行；"
                "本次沒有送出 provider request，也不會沿用舊 consumer state 冒充更新完成。"
            )
        if v2_status in {"NO_TARGET", "DISABLED"}:
            raise RuntimeError(
                "Trading V2 update 無法建立有效的更新 target；"
                + _archive_blocker_detail(v2_update)
            )
        # BLOCKED/DEFERRED describe the *entire* 51-dataset archive batch, not
        # the six required Trading dependencies.  A verified prior common READY
        # horizon may still be safely consumed, but never represented as T READY.
    else:
        v2_update = {
            "status": "LOCAL_V2_ONLY",
            "provider_requests_required": False,
            "data_requests": 0,
            "usage_requests": 0,
        }

    try:
        market_date = _resolve_consumer_market_date(root, v2_update)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise RuntimeError(
            "Trading V2 無法解析共同 READY horizon："
            f"{type(exc).__name__}: {exc} | {_archive_blocker_detail(v2_update)}"
        ) from exc

    # The promotion service is the one canonical owner for both a newer target
    # and a same-target required-dataset source revision.  It checks current
    # dataset validation and immutable view evidence before publishing.
    promotion = promote_trading_v2_consumer_state_if_ready(root, target_date=market_date)
    if not bool(promotion.get("promoted")) and str(promotion.get("reason") or "") != "ALREADY_FINALIZED":
        blockers = "; ".join(str(item) for item in (promotion.get("blocking_dependencies") or ()))
        raise RuntimeError(
            "Trading V2 consumer state 尚無可發布的有效日期；"
            f"market_date={market_date} reason={promotion.get('reason') or '-'} "
            f"{blockers} | {_archive_blocker_detail(v2_update)}"
        )

    consumer_state = load_trading_v2_consumer_state(root, required=True, verify_current_view=True)
    if str(consumer_state.get("market_date") or "") != market_date:
        raise RuntimeError(
            "Trading V2 finalized consumer 日期與共同 READY horizon 不一致；"
            f"expected={market_date}, finalized={consumer_state.get('market_date')}"
        )

    profile = get_trading_strategy_profile()
    dataset_state = load_market_data_dataset_state(root, required=False)
    valid_consumer = build_trading_data_readiness_from_evidence(
        strategy_id=profile.strategy_id,
        target_date=market_date,
        consumer_state_ready=True,
        dataset_state=dataset_state,
        consumer_state_required=False,
    )
    if not bool(valid_consumer["ready"]):
        raise RuntimeError(
            "Trading V2 finalized consumer 已存在，但其必要資料未通過驗證："
            + "; ".join(str(item) for item in (valid_consumer.get("blocking_dependencies") or ()))
        )

    requested_target = str(v2_update.get("target_date") or market_date)
    target_readiness = build_trading_data_readiness_from_evidence(
        strategy_id=profile.strategy_id,
        target_date=requested_target,
        consumer_state_ready=True,
        dataset_state=dataset_state,
        consumer_state_required=False,
    )
    target_ready = bool(target_readiness["ready"] and market_date == requested_target)
    target_blocking = [
        str(item["dataset"])
        for item in target_readiness["v2_dependencies"]
        if not bool(item["ready"])
    ]
    return {
        "status": "READY" if target_ready else "PARTIAL",
        "runtime_domain": "trading",
        "target_date": requested_target,
        "market_date": str(consumer_state["market_date"]),
        "trading_target_ready": target_ready,
        "trading_target_blocking_datasets": target_blocking,
        "trading_target_blocking_dependencies": list(target_readiness["blocking_dependencies"]),
        "trading_target_ready_dataset_count": int(target_readiness["ready_v2_dataset_count"]),
        "trading_target_required_dataset_count": int(target_readiness["required_v2_dataset_count"]),
        "archive_status": str(v2_update.get("status") or ""),
        "consumer_promotion": dict(promotion),
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
