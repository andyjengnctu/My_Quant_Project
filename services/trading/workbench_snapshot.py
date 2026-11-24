"""Trading refresh application service; UI adapters only render its projections."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from config.trading import TRADING_WORKBENCH_INITIAL_READ_WORKERS
from core.console_report import project_relative_display_path
from core.trading_policy import get_trading_policy_snapshot
from services.trading.daily_workflow import build_trading_daily_workflow_snapshot
from services.trading.proposed_order_state import get_trading_proposed_order_plan_read_model
from services.trading.scanner_state import get_trading_candidate_snapshot_read_model, load_trading_candidate_snapshot_for_account
from services.trading.position_rollforward import build_trading_position_rollforward_snapshot, reconcile_trading_manual_position_management
from services.trading.lifecycle_sync import run_trading_lifecycle_sync
from services.trading.pending_entry_service import get_trading_pending_entry_read_model, recover_trading_pending_entry_transaction
from services.trading.pending_entry_state import load_trading_pending_entry_state
from services.trading.operations_status import derive_trading_operations_status_from_preloaded
from services.trading.market_data_consumer_promotion import reconcile_trading_v2_consumer_state_from_local_evidence
from services.trading.protection_planning import get_trading_protection_plan_read_model
from services.trading.indicator_exit_planning import get_trading_indicator_exit_plan_read_model
from services.trading.fill_reconciliation import resolve_trading_fill_transaction_path, recover_trading_fill_transaction
from services.trading.order_state import get_trading_order_read_model, load_trading_order_state
from services.trading.account_dashboard import build_trading_account_dashboard_read_model, publish_trading_account_dashboard_snapshot
from services.trading.account_state import get_trading_account_read_model, load_trading_account_state, resolve_trading_account_state_path

import threading
from services.trading.state_lock import serialized_trading_state_mutation
from services.trading.state_snapshot import TradingStateSnapshot, trading_snapshot_reads

_LOCAL_RECONCILE_LOCK = threading.Lock()


def run_trading_local_reconcile_serialized(callback):
    with _LOCAL_RECONCILE_LOCK:
        return callback()


@serialized_trading_state_mutation
def capture_trading_state_snapshot(project_root):
    return TradingStateSnapshot.capture(project_root, {
        "account": lambda: load_trading_account_state(project_root, required=False),
        "orders": lambda: load_trading_order_state(project_root, required=False),
        "pending": lambda: load_trading_pending_entry_state(project_root, required=False),
    }, fill_transaction_pending=Path(resolve_trading_fill_transaction_path(project_root)).is_file())


def _project_snapshot(snapshot, loader):
    with trading_snapshot_reads(snapshot):
        return loader()


def build_trading_accounting_snapshot(project_root):
    snapshot = capture_trading_state_snapshot(project_root)
    with trading_snapshot_reads(snapshot):
        account = get_trading_account_read_model(project_root)
        dashboard = build_trading_account_dashboard_read_model(project_root)
    publish_trading_account_dashboard_snapshot(project_root, dashboard)
    return account, dashboard


def _build_trading_account_panel_projection(project_root) -> dict[str, object]:
    policy = get_trading_policy_snapshot(project_root)
    state = load_trading_account_state(project_root, required=False)
    account_path = resolve_trading_account_state_path(project_root)
    if state is None:
        return {
            "initialized": False,
            "state_path": project_relative_display_path(account_path, project_root=project_root),
            "policy": policy,
            "revision": None,
            "cash": None,
            "position_count": 0,
            "positions": [],
            "updated_at": None,
        }
    read_model = get_trading_account_read_model(project_root)
    return {
        "initialized": True,
        "state_path": project_relative_display_path(account_path, project_root=project_root),
        "policy": policy,
        **read_model,
    }


def _capture_initial_panel_value(loader):
    try:
        return True, loader()
    except Exception as exc:
        return False, exc


def build_trading_account_panel_initial_bundle(project_root) -> dict[str, object]:
    """Read one coherent initial Trading page bundle without duplicate state I/O."""

    root = Path(project_root).resolve()
    bundle: dict[str, object] = {}
    # Consumer promotion can change the finalized market-data identity, so it
    # runs before lifecycle/recovery and the common account/order/pending capture.
    def _reconcile_initial_state():
        market_data = run_trading_local_reconcile_serialized(
            lambda: reconcile_trading_v2_consumer_state_from_local_evidence(root)
        )
        manual_management = reconcile_trading_manual_position_management(root)
        return {"market_data": market_data, "manual_management": manual_management}

    bundle["reconcile"] = _capture_initial_panel_value(_reconcile_initial_state)
    # Lifecycle state must be synchronized before any Trading read model is
    # rendered.  This is the canonical automatic replacement for the old
    # user-driven position rollforward step and also advances ACTIVE pending
    # plans with their own frozen lineage.
    bundle["lifecycle_sync"] = _capture_initial_panel_value(
        lambda: run_trading_lifecycle_sync(root)
    )
    # Recovery precedes the short capture lock. Projection never mutates state.
    recovery = _capture_initial_panel_value(lambda: recover_trading_fill_transaction(root))
    pending_recovery = _capture_initial_panel_value(lambda: recover_trading_pending_entry_transaction(root))
    snapshot = capture_trading_state_snapshot(root)
    def project(loader):
        return _capture_initial_panel_value(lambda: _project_snapshot(snapshot, loader))
    bundle["account"] = project(lambda: _build_trading_account_panel_projection(root)) if recovery[0] else recovery

    loaders = {
        # Display-only stock names are warmed independently by the panel and must
        # never gate the canonical Trading first paint.  A cold TaiwanStockInfo
        # verification can be much slower than the account/scanner read models.
        "dashboard": lambda: build_trading_account_dashboard_read_model(root),
        "candidate_read": lambda: get_trading_candidate_snapshot_read_model(root),
        "protection": lambda: get_trading_protection_plan_read_model(root, recover_pending_fill=False),
        "indicator": lambda: get_trading_indicator_exit_plan_read_model(root, recover_pending_fill=False),
        "workflow": lambda: build_trading_daily_workflow_snapshot(root),
        "orders": lambda: get_trading_order_read_model(root),
        "proposed": lambda: get_trading_proposed_order_plan_read_model(root),
        "position_rollforward": lambda: build_trading_position_rollforward_snapshot(root),
        "pending": lambda: get_trading_pending_entry_read_model(root, recover_pending=False),
    }
    worker_count = max(1, min(int(TRADING_WORKBENCH_INITIAL_READ_WORKERS), len(loaders)))
    parallel_results: dict[str, tuple[bool, object]] = {}
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="workbench-trading-read") as executor:
        futures = {
            key: executor.submit(project, loader)
            for key, loader in loaders.items()
        }
        for key, future in futures.items():
            parallel_results[key] = future.result()

    if not pending_recovery[0]:
        parallel_results["pending"] = pending_recovery
    if parallel_results["dashboard"][0]:
        publication = _capture_initial_panel_value(
            lambda: publish_trading_account_dashboard_snapshot(root, parallel_results["dashboard"][1])
        )
        if not publication[0]:
            parallel_results["dashboard"] = publication

    for key in ("dashboard", "candidate_read", "protection", "indicator", "workflow", "pending"):
        bundle[key] = parallel_results[key]

    candidate = bundle["candidate_read"]
    if candidate[0] and bool(candidate[1].get("fresh")):
        bundle["candidate_payload"] = project(
            lambda: load_trading_candidate_snapshot_for_account(root, require_current=False)
        )
    else:
        bundle["candidate_payload"] = (True, None)

    component_errors: dict[str, str] = {}

    def _preloaded_value(bundle_key: str, operations_key: str):
        if bundle_key == "account":
            ok, value = bundle["account"]
        else:
            ok, value = parallel_results[bundle_key]
        if ok:
            return value
        component_errors[operations_key] = f"{type(value).__name__}: {value}"
        return None

    fill_transaction_pending = snapshot.fill_transaction_pending
    preloaded_operations = {
        "fill_transaction_pending": fill_transaction_pending,
        "workflow": _preloaded_value("workflow", "workflow"),
        "candidate": _preloaded_value("candidate_read", "candidate"),
        "account": _preloaded_value("account", "account"),
        "position_rollforward": _preloaded_value("position_rollforward", "position_rollforward"),
        "orders": _preloaded_value("orders", "orders"),
        "proposed": _preloaded_value("proposed", "proposed"),
        "protection": _preloaded_value("protection", "protection"),
        "indicator_exit": _preloaded_value("indicator", "indicator_exit"),
    }
    bundle["operations"] = _capture_initial_panel_value(
        lambda: derive_trading_operations_status_from_preloaded(
            workflow=preloaded_operations["workflow"],
            account=preloaded_operations["account"],
            orders=preloaded_operations["orders"],
            candidate=preloaded_operations["candidate"],
            proposed=preloaded_operations["proposed"],
            protection=preloaded_operations["protection"],
            indicator_exit=preloaded_operations["indicator_exit"],
            position_rollforward=preloaded_operations["position_rollforward"],
            pending_entries=_preloaded_value("pending", "pending_entries"),
            fill_transaction_pending=bool(preloaded_operations["fill_transaction_pending"]),
            component_errors=component_errors,
        )
    )
    return bundle


def build_trading_account_panel_snapshot(project_root):
    recover_trading_fill_transaction(project_root)
    snapshot = capture_trading_state_snapshot(project_root)
    return _project_snapshot(snapshot, lambda: _build_trading_account_panel_projection(project_root))


