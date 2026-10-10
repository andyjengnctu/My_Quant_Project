from __future__ import annotations

from tkinter import messagebox
from services.trading.daily_workflow import TRADING_PARAM_MODE_REUSE, build_trading_daily_workflow_snapshot, run_trading_candidate_scan, run_trading_param_step
from services.trading.strategy_param_state import TradingParamContractMismatchError
from services.trading.order_planning import build_trading_proposed_order_plan
from services.trading.proposed_order_state import get_trading_proposed_order_plan_read_model, load_current_trading_proposed_order_plan
from services.trading.scanner_state import get_trading_candidate_snapshot_read_model, load_trading_candidate_snapshot_for_account
from services.trading.lifecycle_sync import SYNC_STATUS_FAILED, SYNC_STATUS_LATEST, SYNC_STATUS_PENDING
from services.trading.pending_entry_service import get_trading_pending_entry_read_model
from services.workbench_ui.state_sync import STATE_ORDERS, STATE_PARAMS, STATE_SCANNER, STATE_SCANNER_ELIGIBILITY
from services.workbench_ui.paged_table import TableColumn
from services.workbench_ui.stock_names import workbench_stock_name
from services.workbench_ui.trading_source_labels import trading_source_display_label
from services.workbench_ui.selection_behavior import clear_treeview_selection
from services.workbench_ui.trading_account_panel_data import (
    WORKBENCH_PROJECT_ROOT,
    PARAM_MODE_REUSE_LABEL,
    PARAM_MODE_TRAIN_LABEL,
    format_trading_money,
    _load_panel_value,
    _candidate_signal_age_days,
    _candidate_signal_date_text,
)

class TradingAccountScannerMixin:
    """Trading Workbench TradingAccountScanner responsibilities; composed by TradingAccountPanel."""

    def refresh_candidate_snapshot_rows(self):
        try:
            snapshot = _load_panel_value(
                self, "candidate_read", lambda: get_trading_candidate_snapshot_read_model(WORKBENCH_PROJECT_ROOT)
            )
        except (OSError, ValueError, RuntimeError) as exc:
            self._reload_candidate_rows([], candidate_payload=None)
            self._operations_next_var.set(f"FAIL | Scanner snapshot 讀取失敗：{exc} | 下一步：重新執行 Scanner")
            return
        if not bool(snapshot.get("valid")):
            self._reload_candidate_rows([], candidate_payload=None)
            self._operations_next_var.set(
                f"WAIT | Scanner snapshot 無效：{snapshot.get('error') or 'schema 不相容'} | 下一步：重新執行 Scanner"
            )
            return
        if not bool(snapshot.get("fresh")):
            self._reload_candidate_rows([], candidate_payload=None)
            self._operations_next_var.set(
                f"WAIT | Scanner snapshot 已過期：{snapshot.get('error') or 'inputs 已變更'} | 下一步：重新執行 Scanner"
            )
            return
        try:
            payload = _load_panel_value(
                self,
                "candidate_payload",
                lambda: load_trading_candidate_snapshot_for_account(
                    WORKBENCH_PROJECT_ROOT, require_current=False
                ),
            )
        except (OSError, FileNotFoundError, TypeError, ValueError, RuntimeError) as exc:
            self._reload_candidate_rows([], candidate_payload=None)
            self._operations_next_var.set(f"FAIL | Scanner snapshot 讀取失敗：{exc} | 下一步：重新執行 Scanner")
            return
        self._reload_candidate_rows(payload.get("candidate_rows") or [], candidate_payload=payload)

    def refresh_pending_entries(self):
        try:
            snapshot = _load_panel_value(
                self, "pending", lambda: get_trading_pending_entry_read_model(WORKBENCH_PROJECT_ROOT)
            )
        except (OSError, FileNotFoundError, TypeError, ValueError, RuntimeError) as exc:
            self._pending_snapshot = {}
            self._pending_rows = {}
            if hasattr(self, "_pending_tree"):
                self._clear_table_sort_values(self._pending_tree)
                for item in self._pending_tree.get_children():
                    self._pending_tree.delete(item)
            self._operations_next_var.set(f"FAIL | 掛單讀取失敗：{exc} | 下一步：全狀態刷新")
            return
        self._pending_snapshot = dict(snapshot or {})
        selected = self._selected_pending_entry_id()
        for item in self._pending_tree.get_children():
            self._pending_tree.delete(item)
        self._clear_table_sort_values(self._pending_tree)
        self._pending_rows = {}
        pending_sync_errors = {
            str(key): str(value)
            for key, value in dict(self._lifecycle_sync_snapshot.get("pending_errors") or {}).items()
        }
        active_rows = [
            dict(row) for row in list(snapshot.get("entries") or [])
            if str(row.get("status") or "") == "ACTIVE"
        ]
        for row in active_rows:
            entry_id = str(row.get("pending_entry_id") or "")
            if not entry_id:
                continue
            self._pending_rows[entry_id] = row
            source = trading_source_display_label(origin=row.get("origin"))
            # AI: Preserve the canonical failure on the read-only display row.
            # A service-wide failure must not leave individual orders looking current.
            row["sync_error"] = pending_sync_errors.get(entry_id) or self._lifecycle_sync_snapshot.get("error")
            if row["sync_error"]:
                status = SYNC_STATUS_FAILED
            else:
                status = str(row.get("sync_status") or (SYNC_STATUS_PENDING if bool(row.get("stale")) else SYNC_STATUS_LATEST))
            planned_date = row.get("planned_trade_date") or row.get("information_date") or "-"
            qty = int(row.get("planned_qty") or 0)
            reserved_cost = row.get("reserved_cost")
            limit_price = row.get("limit_price")
            init_sl = row.get("init_sl")
            target_price = row.get("target_price")
            self._pending_tree.insert(
                "", "end", iid=entry_id,
                values=(
                    "▣", source, row.get("ticker") or "-", workbench_stock_name(WORKBENCH_PROJECT_ROOT, row.get("ticker")), planned_date,
                    f"{qty:,}",
                    self._format_candidate_number(reserved_cost, digits=0),
                    self._format_candidate_number(limit_price, digits=2),
                    self._format_candidate_number(init_sl, digits=2),
                    self._format_candidate_number(target_price, digits=2),
                    status,
                ),
            )
            self._set_table_sort_values(
                self._pending_tree,
                entry_id,
                {
                    "source": source,
                    "ticker": row.get("ticker"),
                    "stock_name": workbench_stock_name(WORKBENCH_PROJECT_ROOT, row.get("ticker")),
                    "date": planned_date,
                    "qty": qty,
                    "reserved": reserved_cost,
                    "limit": limit_price,
                    "stop": init_sl,
                    "target": target_price,
                    "status": status,
                },
            )
        self._apply_current_table_sort(self._pending_tree)
        self._fit_tree_rows(self._pending_tree, len(active_rows))
        edit_id = str(self._pending_edit_entry_id or "").strip()
        if edit_id and edit_id in self._pending_rows:
            # A pending-row selection is meaningful only while the form is bound
            # to that exact ACTIVE pending_entry_id.  Never restore a visual
            # selection without edit identity; that creates a row which looks
            # selected but whose form still runs the create-new path.
            self._pending_tree.selection_set(edit_id)
            self._pending_tree.focus(edit_id)
            self._pending_edit_snapshot = dict(self._pending_rows[edit_id])
        elif edit_id:
            self._reset_pending_draft_form(
                message="原選取掛單已不存在或已結案；目前已回到新增掛單模式。"
            )
        else:
            clear_treeview_selection(self._pending_tree)

    def refresh_proposed_order_plan(self):
        try:
            snapshot = _load_panel_value(
                self, "proposed_read", lambda: get_trading_proposed_order_plan_read_model(WORKBENCH_PROJECT_ROOT)
            )
        except (OSError, ValueError, RuntimeError) as exc:
            self._reload_proposed_order_rows([])
            self._proposed_status_var.set(f"建議掛單讀取失敗：{exc}")
            return
        if not snapshot.get("exists"):
            self._reload_proposed_order_rows([])
            self._proposed_status_var.set("尚未產生建議掛單。")
            return
        if not snapshot.get("valid"):
            self._reload_proposed_order_rows([])
            self._proposed_status_var.set(f"建議掛單 INVALID：{snapshot.get('error') or 'schema 不合法'}")
            return
        if not snapshot.get("fresh"):
            self._reload_proposed_order_rows([])
            self._proposed_status_var.set("PROPOSED STALE｜請重新執行 3 Scanner 與 5 建議掛單。")
            return
        try:
            payload = _load_panel_value(
                self, "proposed_payload", lambda: load_current_trading_proposed_order_plan(WORKBENCH_PROJECT_ROOT, require_current=False)
            )
        except (OSError, FileNotFoundError, TypeError, ValueError, RuntimeError) as exc:
            self._reload_proposed_order_rows([])
            self._proposed_status_var.set(f"建議掛單讀取失敗：{exc}")
            return
        self._reload_proposed_order_rows(payload.get("orders") or [])
        self._proposed_status_var.set(
            f"PROPOSED | account rev {payload.get('account_revision')} | equity {format_trading_money(payload.get('sizing_equity'))} | "
            f"預留 {format_trading_money(payload.get('reserved_total'))} | 餘額 {format_trading_money(payload.get('cash_after_reservation'))}"
        )

    def _refresh_workflow_views(self):
        self.refresh_daily_workflow()
        self.refresh_candidate_snapshot_rows()
        self.refresh_pending_entries()
        self.refresh_proposed_order_plan()

    def refresh_daily_workflow(self):
        try:
            snapshot = _load_panel_value(
                self, "workflow", lambda: build_trading_daily_workflow_snapshot(WORKBENCH_PROJECT_ROOT)
            )
        except (OSError, ValueError, RuntimeError) as exc:
            self._latest_workflow_snapshot = {}
            self._operations_next_var.set(f"FAIL | Workflow 狀態讀取失敗：{exc} | 下一步：全狀態刷新")
            return
        self._latest_workflow_snapshot = dict(snapshot)
        reusable = bool(snapshot.get("params_reusable"))
        if not self._param_mode_user_selected:
            self._param_mode_var.set(PARAM_MODE_REUSE_LABEL if reusable else PARAM_MODE_TRAIN_LABEL)
        self.refresh_operations_status()

    @staticmethod
    def _workflow_state_domains(action: str):
        mapping = {
            "params": {STATE_PARAMS, STATE_SCANNER_ELIGIBILITY},
            "scanner": {STATE_SCANNER, STATE_SCANNER_ELIGIBILITY},
            "orders": {STATE_ORDERS},
        }
        return mapping.get(str(action), set())

    def _start_workflow_action(self, action: str):
        param_mode = self._selected_param_mode()
        param_label = "沿用既有 Trading Params" if param_mode == TRADING_PARAM_MODE_REUSE else "重新訓練 Trading Params"
        labels = {
            "params": param_label,
            "scanner": "Scanner 候選",
            "orders": "產生建議掛單",
        }
        if action not in labels:
            messagebox.showerror("Trading workflow", f"未知 workflow action: {action}", parent=self)
            return
        self._operations_next_var.set(f"BUSY | {labels[action]} 執行中… | 下一步：等待完成")

        def on_success(result):
            self._finish_workflow_success(action, result)

        def on_error(exc):
            if isinstance(exc, TradingParamContractMismatchError):
                # AI: Never silently reinterpret legacy fixed seed Params as
                # random, or start a new retrain without explicit user action.
                self._param_mode_var.set(PARAM_MODE_TRAIN_LABEL)
                self._param_mode_user_selected = False
                self._operations_next_var.set(
                    "需重新訓練 Trading Params | 舊 Params 與目前的 seed/producer 契約不相容"
                )
                messagebox.showwarning(
                    "Trading Params 需要重新訓練",
                    "目前 Trading Params 與現行策略契約不相容，不能沿用。\n"
                    "已切換至「重新訓練 Params」；請重新執行「2 訓練／套用 Params」。",
                    parent=self,
                )
                return
            self._operations_next_var.set(f"FAIL | {type(exc).__name__}: {exc} | 下一步：修正錯誤後重試")
            messagebox.showerror("Trading workflow 失敗", f"{type(exc).__name__}: {exc}", parent=self)

        self._submit_trading_command(
            labels[action],
            lambda: self._run_workflow_command(action, param_mode),
            on_success=on_success,
            on_error=on_error,
            refresh_state=True,
            state_domains=self._workflow_state_domains(action),
        )

    @staticmethod
    def _run_workflow_command(action: str, param_mode: str):
        if action == "params":
            return run_trading_param_step(project_root=WORKBENCH_PROJECT_ROOT, mode=param_mode)
        if action == "scanner":
            return run_trading_candidate_scan(project_root=WORKBENCH_PROJECT_ROOT)
        if action == "orders":
            return build_trading_proposed_order_plan(project_root=WORKBENCH_PROJECT_ROOT)
        raise ValueError(f"不支援的 Trading workflow action: {action!r}")

    @staticmethod
    def _format_candidate_number(value, *, digits=2):
        if value is None:
            return "-"
        try:
            return f"{float(value):,.{digits}f}"
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _candidate_static_columns_before_dynamic():
        return (
            TableColumn("rank", "順位", 5, sort_kind="numeric"),
            TableColumn("ticker", "股票", 7),
            TableColumn("stock_name", "名稱", 12),
            TableColumn("kind_label", "類型", 10),
        )

    def _candidate_static_columns_after_dynamic(self):
        return (
            TableColumn("signal_date", "買訊日", 10),
            TableColumn("signal_age_days", "新鮮度", 7, sort_kind="numeric", formatter=lambda v, _r: "-" if v is None else f"{int(v)}天"),
            TableColumn("market_price", "市價", 9, sort_kind="numeric", formatter=lambda v, _r: self._format_candidate_number(v, digits=2)),
            TableColumn("limit_price", "買入限價", 9, sort_kind="numeric", formatter=lambda v, _r: self._format_candidate_number(v, digits=2)),
            TableColumn("stop_price", "初始Stop", 9, sort_kind="numeric", formatter=lambda v, _r: self._format_candidate_number(v, digits=2)),
            TableColumn("target_price", "停利線", 9, sort_kind="numeric", formatter=lambda v, _r: self._format_candidate_number(v, digits=2)),
            TableColumn("proj_qty", "參考股數", 9, sort_kind="numeric", formatter=lambda v, _r: "-" if v is None else f"{int(v):,}"),
            TableColumn("proj_cost", "參考投入", 10, sort_kind="numeric", formatter=lambda v, _r: self._format_candidate_number(v, digits=0)),
            TableColumn("ev_value", "EV", 7, sort_kind="numeric", formatter=lambda v, _r: self._format_candidate_number(v, digits=3)),
            TableColumn("win_rate", "歷史勝率", 8, sort_kind="numeric", formatter=lambda v, _r: self._format_candidate_number(v, digits=1) + "%" if v is not None else "-"),
            TableColumn("trade_count", "交易次數", 8, sort_kind="numeric", formatter=lambda v, _r: f"{int(v or 0):,}"),
            TableColumn("asset_growth", "資產成長", 8, sort_kind="numeric", formatter=lambda v, _r: self._format_candidate_number(v, digits=1) + "%" if v is not None else "-"),
        )

    @staticmethod
    def _candidate_metric_formatter(metric):
        spec = dict(metric or {})
        format_kind = str(spec.get("format_kind") or "")
        if format_kind == "fraction":
            denominator = int(spec.get("denominator") or 0)
            return lambda value, _row, total=denominator: "-" if value is None or total < 1 else f"{int(value)}/{total}"
        if format_kind == "integer":
            return lambda value, _row: "-" if value is None else str(int(value))
        if format_kind == "integer_grouped":
            return lambda value, _row: "-" if value is None else f"{int(round(float(value))):,}"
        if format_kind == "number":
            return lambda value, _row: "-" if value is None else f"{float(value):.2f}"
        if format_kind == "percent":
            return lambda value, _row: "-" if value is None else f"{float(value):.2f}%"
        if format_kind == "r":
            return lambda value, _row: "-" if value is None else f"{float(value):.2f}R"
        return lambda value, _row: "-" if value is None else str(value)

    def _build_candidate_table_columns(self, display_metrics):
        dynamic_columns = tuple(
            TableColumn(
                str(metric.get("key") or ""),
                str(metric.get("label") or ""),
                int(metric.get("width") or 8),
                sort_kind=str(metric.get("sort_kind") or "text"),
                formatter=self._candidate_metric_formatter(metric),
            )
            for metric in list(display_metrics or [])
            if str(metric.get("key") or "").strip() and str(metric.get("label") or "").strip()
        )
        return self._candidate_static_columns_before_dynamic() + dynamic_columns + self._candidate_static_columns_after_dynamic()

    @staticmethod
    def _candidate_display_metrics(candidate_payload):
        payload = dict(candidate_payload or {})
        metrics = payload.get("candidate_display_metrics")
        if isinstance(metrics, list):
            return [dict(item) for item in metrics if isinstance(item, dict)]
        return []

    def _reload_candidate_rows(self, rows, *, candidate_payload=None):
        self._candidate_payload = dict(candidate_payload or {})
        display_metrics = self._candidate_display_metrics(candidate_payload)
        self._candidate_tree.set_columns(self._build_candidate_table_columns(display_metrics))
        self._candidate_rows = [dict(row) for row in list(rows or [])]
        self._candidate_by_ticker = {
            str(row.get("ticker") or "").strip().upper(): row
            for row in self._candidate_rows
            if str(row.get("ticker") or "").strip()
        }
        kind_labels = {"buy": "新訊號", "extended": "延續", "extended_tbd": "延續(TBD)", "reentry": "再進場"}
        display_rows = []
        for idx, row in enumerate(self._candidate_rows, 1):
            seed = dict(row.get("execution_plan_seed") or {})
            ticker = str(row.get("ticker") or "-").strip().upper()
            candidate_date = (
                row.get("candidate_date")
                or row.get("trade_date")
                or self._candidate_payload.get("latest_data_date")
            )
            display_row = {
                "_table_id": f"candidate:{ticker}",
                "rank": idx,
                "ticker": ticker,
                "stock_name": workbench_stock_name(WORKBENCH_PROJECT_ROOT, ticker),
                "kind_label": kind_labels.get(str(row.get("kind") or ""), str(row.get("kind") or "-")),
                "signal_date": _candidate_signal_date_text(row.get("signal_date")),
                "signal_age_days": _candidate_signal_age_days(row.get("signal_date"), candidate_date),
                "market_price": row.get("prev_close"),
                "limit_price": row.get("limit_price") if row.get("limit_price") is not None else seed.get("limit_price"),
                "stop_price": seed.get("init_sl"),
                "target_price": seed.get("target_price"),
                "proj_qty": row.get("proj_qty"),
                "proj_cost": row.get("proj_cost"),
                "ev_value": row.get("expected_value", row.get("ev")),
                "win_rate": row.get("win_rate"),
                "trade_count": row.get("trade_count"),
                "asset_growth": row.get("asset_growth"),
            }
            for metric in display_metrics:
                metric_key = str(metric.get("key") or "").strip()
                value_field = str(metric.get("value_field") or "").strip()
                if metric_key and value_field:
                    display_row[metric_key] = row.get(value_field)
            display_rows.append(display_row)
        self._candidate_tree.set_rows(display_rows, preserve_selection=True)

    def _selected_candidate_ticker(self):
        row = self._candidate_tree.selected_row()
        return str(row.get("ticker") or "").strip().upper() if row else None

    def _selected_candidate_row(self):
        ticker = self._selected_candidate_ticker()
        return dict(getattr(self, "_candidate_by_ticker", {}).get(ticker) or {}) if ticker else None
