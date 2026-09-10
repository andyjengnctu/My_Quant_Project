"""Canonical Trading Market Data V2 readiness gate.

This service is provider-free. It composes the active Trading dependency
contract with the canonical V2 execution consumer state and per-dataset
operational readiness. Legacy Trading CSV/snapshot state is intentionally not
accepted as readiness evidence.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from core.market_data_freshness_contract import FRESHNESS_STATUS_READY
from core.trading_data_dependencies import get_trading_data_dependency_spec
from core.trading_policy import get_trading_strategy_profile
from services.trading.market_data_consumer import load_trading_v2_consumer_state
from services.trading.market_data_dataset_state import (
    VALIDATION_STATUS_NO_ROW_VALID,
    VALIDATION_STATUS_READY,
    load_market_data_dataset_state,
)

READINESS_STATUS_READY = "READY"
READINESS_STATUS_BLOCKED = "BLOCKED"
_VALID_REQUIRED_VALIDATION_STATUSES = {
    VALIDATION_STATUS_READY,
    VALIDATION_STATUS_NO_ROW_VALID,
}


def _evaluate_v2_dependency(
    *,
    dataset: str,
    target_date: str | None,
    dynamic: Mapping[str, object] | None,
) -> dict[str, Any]:
    row = dict(dynamic or {})
    status = str(row.get("status") or "MISSING")
    ready_target = str(row.get("last_ready_target_date") or "") or None
    schema_status = str(row.get("schema_status") or "UNKNOWN")
    coverage_status = str(row.get("coverage_status") or "UNKNOWN")
    reasons: list[str] = []
    if not target_date:
        reasons.append("Trading V2 consumer target date 尚未建立")
    if dynamic is None:
        reasons.append("dataset operational state 尚未建立")
    else:
        if not ready_target or (target_date and ready_target < target_date):
            reasons.append(f"last_ready_target_date={ready_target or '-'} 未達 target={target_date or '-'}")
        if status != FRESHNESS_STATUS_READY:
            reasons.append(f"status={status}")
        if schema_status not in _VALID_REQUIRED_VALIDATION_STATUSES:
            reasons.append(f"schema={schema_status}")
        if coverage_status not in _VALID_REQUIRED_VALIDATION_STATUSES:
            reasons.append(f"coverage={coverage_status}")
    return {
        "dataset": dataset,
        "ready": not reasons,
        "status": status,
        "last_ready_target_date": ready_target,
        "latest_data_date": row.get("latest_data_date"),
        "schema_status": schema_status,
        "coverage_status": coverage_status,
        "reason": None if not reasons else "; ".join(reasons),
    }


def build_trading_data_readiness_from_evidence(
    *,
    strategy_id: str,
    target_date: str | None,
    consumer_state_ready: bool,
    consumer_state_reason: str | None = None,
    dataset_state: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    spec = get_trading_data_dependency_spec(strategy_id)
    blockers: list[str] = []
    consumer_ready = bool(consumer_state_ready and target_date)
    if not consumer_ready:
        blockers.append(consumer_state_reason or "Trading V2 execution consumer state 尚未就緒")

    state_rows = {}
    if isinstance(dataset_state, Mapping):
        raw_rows = dataset_state.get("datasets")
        if isinstance(raw_rows, Mapping):
            state_rows = {str(key): value for key, value in raw_rows.items() if isinstance(value, Mapping)}

    v2_rows = [
        _evaluate_v2_dependency(dataset=dataset, target_date=target_date, dynamic=state_rows.get(dataset))
        for dataset in spec.required_v2_datasets
    ]
    for row in v2_rows:
        if not row["ready"]:
            blockers.append(f"V2 {row['dataset']}: {row['reason']}")

    ready_v2_count = sum(bool(row["ready"]) for row in v2_rows)
    ready = bool(consumer_ready and ready_v2_count == len(v2_rows))
    return {
        "status": READINESS_STATUS_READY if ready else READINESS_STATUS_BLOCKED,
        "ready": ready,
        "strategy_id": spec.strategy_id,
        "target_date": target_date,
        "dependency_fingerprint": spec.fingerprint,
        "consumer_state_required": True,
        "consumer_state_ready": consumer_ready,
        "required_v2_dataset_count": len(v2_rows),
        "ready_v2_dataset_count": ready_v2_count,
        "blocking_v2_dataset_count": len(v2_rows) - ready_v2_count,
        "required_v2_datasets": list(spec.required_v2_datasets),
        "v2_dependencies": v2_rows,
        "blocking_dependencies": blockers,
    }


def build_trading_data_readiness_for_consumer_evidence(
    project_root: str | Path,
    *,
    strategy_id: str,
    target_date: str | None,
    consumer_state_ready: bool,
    consumer_state_reason: str | None = None,
) -> dict[str, Any]:
    """Join already-validated V2 consumer evidence with required V2 dataset state."""

    state = None
    state_error = None
    try:
        state = load_market_data_dataset_state(project_root, required=False)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        state_error = f"{type(exc).__name__}: {exc}"
    result = build_trading_data_readiness_from_evidence(
        strategy_id=strategy_id,
        target_date=target_date,
        consumer_state_ready=consumer_state_ready,
        consumer_state_reason=consumer_state_reason,
        dataset_state=state,
    )
    if state_error:
        result["state_error"] = state_error
        if result.get("ready"):
            result["ready"] = False
            result["status"] = READINESS_STATUS_BLOCKED
            result["blocking_dependencies"] = list(result.get("blocking_dependencies") or []) + [
                "V2 dataset state 無法驗證: " + state_error
            ]
    return result


def build_trading_data_readiness(
    project_root: str | Path,
    *,
    verify_consumer_view: bool = False,
) -> dict[str, Any]:
    """Resolve active-strategy readiness using only canonical Market Data V2 state."""

    root = Path(project_root).resolve()
    profile = get_trading_strategy_profile()
    target_date: str | None = None
    consumer_ready = False
    consumer_reason: str | None = None
    try:
        consumer_state = load_trading_v2_consumer_state(
            root,
            required=True,
            verify_current_view=bool(verify_consumer_view),
        )
        target_date = str(consumer_state.get("market_date") or "") or None
        consumer_ready = bool(target_date)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        consumer_reason = f"{type(exc).__name__}: {exc}"

    state = None
    state_error = None
    try:
        state = load_market_data_dataset_state(root, required=False)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        state_error = f"{type(exc).__name__}: {exc}"

    result = build_trading_data_readiness_from_evidence(
        strategy_id=profile.strategy_id,
        target_date=target_date,
        consumer_state_ready=consumer_ready,
        consumer_state_reason=consumer_reason,
        dataset_state=state,
    )
    if state_error:
        result["state_error"] = state_error
        if result.get("ready"):
            result["ready"] = False
            result["status"] = READINESS_STATUS_BLOCKED
            result["blocking_dependencies"] = list(result.get("blocking_dependencies") or []) + [
                "V2 dataset state 無法驗證: " + state_error
            ]
    return result


def assert_trading_data_readiness(readiness: Mapping[str, object]) -> None:
    if bool(readiness.get("ready")):
        return
    blockers = [str(item) for item in list(readiness.get("blocking_dependencies") or []) if str(item)]
    if not blockers:
        blockers = ["Trading data dependency readiness 尚未就緒"]
    raise RuntimeError("Trading data readiness BLOCKED：" + "；".join(blockers))


__all__ = [
    "READINESS_STATUS_READY",
    "READINESS_STATUS_BLOCKED",
    "build_trading_data_readiness_from_evidence",
    "build_trading_data_readiness_for_consumer_evidence",
    "build_trading_data_readiness",
    "assert_trading_data_readiness",
]
