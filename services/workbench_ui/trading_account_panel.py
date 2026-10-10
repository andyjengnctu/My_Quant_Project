"""Trading Account Workbench facade: stable public panel and method API.

The view, read models, background lifecycle, and user-action controllers are
composed from responsibility-specific mixins. This module owns only the public
class and its construction state; domain rules remain in services.trading.
"""
from __future__ import annotations

import queue
from tkinter import ttk, messagebox
from services.workbench_ui.workbench import WORKBENCH_FRAME_STYLE

from services.workbench_ui.trading_account_panel_runtime import TradingAccountRuntimeMixin
from services.workbench_ui.trading_account_panel_layout import TradingAccountLayoutMixin
from services.workbench_ui.trading_account_panel_protection import TradingAccountProtectionMixin
from services.workbench_ui.trading_account_panel_dashboard import TradingAccountDashboardMixin
from services.workbench_ui.trading_account_panel_promotion import TradingAccountPromotionMixin
from services.workbench_ui.trading_account_panel_scanner import TradingAccountScannerMixin
from services.workbench_ui.trading_account_panel_pending import TradingAccountPendingMixin
from services.workbench_ui.trading_account_panel_orders import TradingAccountOrdersMixin
from services.workbench_ui.trading_account_panel_account import TradingAccountAccountMixin
from services.workbench_ui.trading_account_panel_data import (
    WORKBENCH_PROJECT_ROOT,
    MANUAL_SOURCE,
    UNMANAGED_STATUS,
    PARAM_MODE_REUSE_LABEL,
    PARAM_MODE_TRAIN_LABEL,
    PARAM_MODE_BY_LABEL,
    WORKFLOW_HINT,
    CASH_HINT,
    MANUAL_POSITION_HINT,
    FILL_HINT,
    PROPOSED_ORDER_HINT,
    PROTECTION_HINT,
    OCO_HINT,
    INDICATOR_HINT,
    POSITION_DECISION_HINT,
    SCANNER_HINT,
    BUY_ENTRY_HINT,
    PENDING_ENTRY_HINT,
    holding_order_date_by_lineage,
    _pending_sync_failure_at_cell,
    parse_trading_money_text,
    parse_trading_qty_text,
    format_trading_money,
    build_trading_account_panel_snapshot,
    _capture_initial_panel_value,
    build_trading_account_panel_initial_bundle,
    _load_panel_value,
    _parse_candidate_date,
    _candidate_signal_age_days,
    _candidate_signal_date_text,
)
from services.workbench_ui.trading_account_panel_status import (
    build_trading_status_segments,
    _TradingStatusLine,
)

class TradingAccountPanel(
    TradingAccountRuntimeMixin,
    TradingAccountLayoutMixin,
    TradingAccountProtectionMixin,
    TradingAccountDashboardMixin,
    TradingAccountPromotionMixin,
    TradingAccountScannerMixin,
    TradingAccountPendingMixin,
    TradingAccountOrdersMixin,
    TradingAccountAccountMixin,
    ttk.Frame,
):
    def __init__(self, master):
        super().__init__(master, padding=10, style=WORKBENCH_FRAME_STYLE)
        self._snapshot: dict[str, object] = {}
        self._account_dashboard_snapshot: dict[str, object] = {}
        self._performance_rows: list[dict[str, object]] = []
        self._position_rows: dict[str, dict[str, object]] = {}
        self._candidate_rows: list[dict[str, object]] = []
        self._candidate_payload: dict[str, object] = {}
        self._pending_snapshot: dict[str, object] = {}
        self._pending_rows: dict[str, dict[str, object]] = {}
        self._pending_draft_origin: str | None = None
        self._pending_draft_candidate: dict[str, object] | None = None
        self._pending_edit_entry_id: str | None = None
        self._pending_edit_snapshot: dict[str, object] = {}
        self._pending_draft_programmatic_update = False
        self._pending_manual_preview_after_id = None
        self._pending_preview_thread = None
        self._pending_preview_token = 0
        self._pending_preview_inflight = 0
        self._pending_preview_results: queue.Queue = queue.Queue()
        self._pending_preview_poll_after_id = None
        self._pending_order_form_constraints: dict[str, object] = {}
        self._pending_fill_form_constraints: dict[str, object] = {}
        self._pending_price_option_set: frozenset[str] = frozenset()
        self._table_sort_titles: dict[ttk.Treeview, dict[str, str]] = {}
        self._table_sort_kinds: dict[ttk.Treeview, dict[str, str]] = {}
        self._table_sort_state: dict[ttk.Treeview, tuple[str, bool]] = {}
        self._table_sort_values: dict[ttk.Treeview, dict[str, dict[str, object]]] = {}
        self._trade_fill_constraints: dict[str, object] = {}
        self._trade_price_option_set: frozenset[str] = frozenset()
        self._trade_constraint_after_id = None
        self._trade_constraint_programmatic_update = False
        self._trade_constraint_token = 0
        self._trade_constraint_inflight = 0
        self._trade_constraint_results: queue.Queue = queue.Queue()
        self._trade_constraint_poll_after_id = None
        self._proposed_order_rows: list[dict[str, object]] = []
        self._order_snapshot: dict[str, object] = {}
        self._order_rows: dict[str, dict[str, object]] = {}
        self._protection_snapshot: dict[str, object] = {}
        self._protection_rows: list[dict[str, object]] = []
        self._indicator_snapshot: dict[str, object] = {}
        self._indicator_rows: list[dict[str, object]] = []
        self._command_thread = None
        self._command_token = 0
        self._command_results: queue.Queue = queue.Queue()
        self._command_poll_after_id = None
        self._command_context: dict[int, dict[str, object]] = {}
        self._command_saved_button_states: dict[object, str] = {}
        self._param_mode_user_selected = False
        self._workflow_buttons = []
        self._workflow_action_buttons: dict[str, object] = {}
        self._operations_snapshot: dict[str, object] = {}
        self._lifecycle_sync_snapshot: dict[str, object] = {}
        self._latest_workflow_snapshot: dict[str, object] = {}
        self._footer_hint_after_id = None
        self._initial_state_thread = None
        self._initial_state_token = 0
        self._initial_state_results: queue.Queue = queue.Queue()
        self._initial_state_poll_after_id = None
        self._initial_state_start_after_id = None
        self._initial_state_started = False
        self._initial_state_applied = False
        self._deferred_initial_bundle: dict[str, object] | None = None
        self._initial_apply_steps: list[object] = []
        self._initial_apply_after_id = None
        self._initial_apply_bundle_active = False
        self._workbench_active = False
        self._initial_preloaded: dict[str, object] = {}
        self._suspend_operations_refresh = False
        self._external_state_refresh_pending_domains: set[str] = set()
        self._stock_name_warm_thread = None
        self._stock_name_warm_after_id = None
        self._stock_name_warm_error: str | None = None
        self._build_ui()
        self._set_initial_loading_state()
        # AI: The Workbench shell notifies visibility after construction.  Keep a
        # Map/Unmap fallback for standalone panel use and defer the heavyweight
        # first read until this page remains visible for a short dwell interval.
        self.bind("<Map>", lambda _event: self.on_workbench_visibility_changed(True), add="+")
        self.bind("<Unmap>", lambda _event: self.on_workbench_visibility_changed(False), add="+")

__all__ = [
    "TradingAccountPanel",
    "build_trading_account_panel_snapshot",
    "format_trading_money",
    "parse_trading_money_text",
    "parse_trading_qty_text",
]
