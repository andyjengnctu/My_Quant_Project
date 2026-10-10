from __future__ import annotations

from datetime import datetime, timedelta, timezone
import queue
import threading
import tkinter as tk
from tkinter import messagebox
from services.trading.scanner_state import load_trading_scanner_runtime
from services.trading.order_form_constraints import ORDER_FORM_DATE_KIND_PENDING, build_trading_actual_fill_form_constraints, build_trading_pending_order_form_constraints
from services.trading.pending_entry_service import create_manual_trading_pending_entry, create_scanner_trading_pending_entry, delete_pending_entry, fill_trading_pending_entry, preview_manual_trading_pending_entry, preview_scanner_trading_pending_entry, preview_trading_pending_entry_fill, preview_trading_pending_entry_update, update_trading_pending_entry_intent
from services.workbench_ui.state_sync import PENDING_FILL_MUTATION_DOMAINS, PENDING_MUTATION_DOMAINS
from services.workbench_ui.trading_source_labels import TRADING_SOURCE_CUSTOM_LABEL, TRADING_SOURCE_STRATEGY_LABEL, trading_source_display_label
from services.workbench_ui.selection_behavior import clear_treeview_selection
from services.workbench_ui.workbench import _warn_gui_fallback
from services.workbench_ui.trading_account_panel_data import (
    WORKBENCH_PROJECT_ROOT,
    _pending_sync_failure_at_cell,
    parse_trading_money_text,
    parse_trading_qty_text,
)

class TradingAccountPendingMixin:
    """Trading Workbench TradingAccountPending responsibilities; composed by TradingAccountPanel."""

    def _pending_fill_mode_active(self) -> bool:
        return bool(
            self._pending_edit_entry_id is not None
            and self._pending_fill_date_var.get().strip()
        )

    def _configure_pending_mode_buttons(self) -> None:
        editing = self._pending_edit_entry_id is not None
        fill_mode = self._pending_fill_mode_active()
        if hasattr(self, "_pending_order_ticker_entry"):
            self._pending_order_ticker_entry.configure(state="disabled" if editing else "normal")
        if hasattr(self, "_pending_order_qty_entry"):
            self._pending_order_qty_entry.configure(state="disabled" if fill_mode else "normal")
        if hasattr(self, "_pending_order_date_field"):
            allowed_dates = (
                ()
                if fill_mode
                else tuple((self._pending_order_form_constraints or {}).get("allowed_dates") or ())
            )
            self._pending_order_date_field.set_allowed_dates(allowed_dates)
            self._pending_order_date_field.entry.configure(state="disabled" if fill_mode else "normal")
        if hasattr(self, "_pending_submit_button"):
            selected_kind = str((self._pending_order_form_constraints or {}).get("selected_date_kind") or "")
            order_ready = bool(
                not fill_mode
                and selected_kind == ORDER_FORM_DATE_KIND_PENDING
                and self._pending_order_limit_var.get().strip()
            )
            self._pending_submit_button.configure(
                text="更新掛單" if editing else "確認掛單",
                state="normal" if order_ready else "disabled",
            )
        if hasattr(self, "_pending_fill_button"):
            self._pending_fill_button.configure(
                text="確認成交",
                state="disabled",
            )
        if hasattr(self, "_pending_delete_button"):
            self._pending_delete_button.configure(
                state="normal" if editing and not fill_mode else "disabled"
            )


    def _set_pending_edit_mode(
        self, pending_entry_id: str | None, *, row: dict[str, object] | None = None
    ) -> None:
        self._pending_edit_entry_id = None if not pending_entry_id else str(pending_entry_id)
        self._pending_edit_snapshot = {} if self._pending_edit_entry_id is None else dict(row or {})
        self._configure_pending_mode_buttons()

    def _reset_pending_draft_form(self, *, message: str | None = None) -> None:
        self._invalidate_pending_preview()
        if self._pending_manual_preview_after_id is not None:
            try:
                self.after_cancel(self._pending_manual_preview_after_id)
            except tk.TclError as exc:
                _warn_gui_fallback("Trading pending preview debounce", exc)
            self._pending_manual_preview_after_id = None
        self._pending_draft_origin = "manual"
        self._pending_draft_candidate = None
        self._set_pending_edit_mode(None)
        self._pending_draft_programmatic_update = True
        try:
            self._pending_order_source_var.set(TRADING_SOURCE_CUSTOM_LABEL)
            self._pending_order_ticker_var.set("")
            self._pending_order_qty_var.set("")
            self._pending_order_limit_var.set("")
            self._pending_fill_price_var.set("")
            self._pending_order_date_var.set("")
            self._pending_fill_date_var.set("")
        finally:
            self._pending_draft_programmatic_update = False
        self._pending_order_form_constraints = {}
        self._pending_fill_form_constraints = {}
        self._pending_price_option_set = frozenset()
        if hasattr(self, "_pending_fill_price_combo"):
            self._pending_fill_price_combo.configure(values=(), state="disabled")
        if hasattr(self, "_pending_order_date_field"):
            self._pending_order_date_field.set_allowed_dates(())
        if hasattr(self, "_pending_fill_date_field"):
            self._pending_fill_date_field.set_allowed_dates(())
        self._pending_draft_preview_var.set(message or "")

    def _on_candidate_selected(self, row):
        ticker = str((row or {}).get("ticker") or "").strip().upper()
        self._trade_ticker_var.set(ticker)
        candidate = self._selected_candidate_row()
        if not candidate:
            self._reset_pending_draft_form(message="已取消 Scanner 選取；目前為手動新增掛單模式。")
            return
        clear_treeview_selection(self._pending_tree)
        self._invalidate_pending_preview()
        self._set_pending_edit_mode(None)
        self._pending_draft_origin = "scanner"
        self._pending_draft_candidate = dict(candidate)
        seed = dict(candidate.get("execution_plan_seed") or {})
        candidate_qty = candidate.get("proj_qty")
        if candidate_qty is None:
            candidate_qty = seed.get("qty")
        candidate_price = candidate.get("limit_price")
        if candidate_price is None:
            candidate_price = seed.get("limit_price")
        candidate_date = (
            candidate.get("trade_date")
            or candidate.get("candidate_date")
            or self._candidate_payload.get("latest_data_date")
            or ""
        )
        self._pending_draft_programmatic_update = True
        try:
            self._pending_order_source_var.set(TRADING_SOURCE_STRATEGY_LABEL)
            self._pending_order_ticker_var.set(ticker)
            self._pending_order_qty_var.set("" if candidate_qty is None else str(int(candidate_qty)))
            self._pending_order_limit_var.set("" if candidate_price is None else str(candidate_price))
            self._pending_fill_price_var.set("")
            self._pending_order_date_var.set(str(candidate_date or ""))
            self._pending_fill_date_var.set("")
        finally:
            self._pending_draft_programmatic_update = False
        self._pending_draft_preview_var.set(
            f"{ticker}｜Scanner 資料已帶入；正在驗證目前帳戶資源、掛單日期與自動買入限價…"
        )
        self._preview_pending_draft(use_current_overrides=False, silent=True)

    def _selected_pending_entry_id(self):
        if not hasattr(self, "_pending_tree"):
            return None
        selected = self._pending_tree.selection()
        return str(selected[0]) if selected else None

    def _selected_pending_row(self):
        entry_id = self._selected_pending_entry_id()
        return None if not entry_id else self._pending_rows.get(entry_id)

    def _bind_existing_pending_for_ticker(self, ticker: str, *, preserve_overrides: bool) -> bool:
        """Bind the form to the unique ACTIVE pending row for ``ticker``.

        Duplicate ACTIVE intent for one ticker is not a valid create-new state.
        This guard runs before *every* preview path, so changing the date/qty
        immediately after typing a ticker cannot cancel a delayed ticker-commit
        callback and accidentally fall back to the create-new resource check.
        """

        ticker_key = str(ticker or "").strip().upper()
        if not ticker_key or self._pending_edit_entry_id is not None:
            return False
        matches = [
            (entry_id, dict(row))
            for entry_id, row in self._pending_rows.items()
            if str(row.get("ticker") or "").strip().upper() == ticker_key
        ]
        if len(matches) != 1:
            return False
        entry_id, row = matches[0]
        current_qty = self._pending_order_qty_var.get().strip() if preserve_overrides else ""
        current_date = self._pending_order_date_var.get().strip() if preserve_overrides else ""
        current_fill_date = self._pending_fill_date_var.get().strip() if preserve_overrides else ""
        current_fill_price = self._pending_fill_price_var.get().strip() if preserve_overrides else ""
        origin = "scanner" if str(row.get("origin") or "") == "scanner_strategy" else "manual"
        self._pending_tree.selection_set(entry_id)
        self._pending_tree.focus(entry_id)
        self._candidate_tree.clear_selection(notify=False)
        self._pending_draft_origin = origin
        self._pending_draft_candidate = None
        self._set_pending_edit_mode(entry_id, row=row)
        self._pending_draft_programmatic_update = True
        try:
            self._pending_order_source_var.set(trading_source_display_label(origin=origin))
            self._pending_order_ticker_var.set(ticker_key)
            self._pending_order_qty_var.set(current_qty or str(int(row.get("planned_qty") or 0)))
            self._pending_order_limit_var.set(str(row.get("limit_price") or ""))
            self._pending_order_date_var.set(
                current_date or str(row.get("planned_trade_date") or row.get("information_date") or "")
            )
            self._pending_fill_date_var.set(current_fill_date)
            self._pending_fill_price_var.set(current_fill_price)
        finally:
            self._pending_draft_programmatic_update = False
        return True

    def _on_pending_selected(self, _event=None):
        row = self._selected_pending_row()
        if not row:
            return
        if self._pending_manual_preview_after_id is not None:
            try:
                self.after_cancel(self._pending_manual_preview_after_id)
            except tk.TclError as exc:
                _warn_gui_fallback("Trading pending selection preview debounce", exc)
            self._pending_manual_preview_after_id = None
        self._invalidate_pending_preview()
        self._candidate_tree.clear_selection(notify=False)
        entry_id = str(row.get("pending_entry_id") or "")
        origin = "scanner" if str(row.get("origin") or "") == "scanner_strategy" else "manual"
        self._pending_draft_origin = origin
        self._pending_draft_candidate = None
        self._set_pending_edit_mode(entry_id, row=dict(row))
        planned_date = str(row.get("planned_trade_date") or "").strip()
        self._pending_draft_programmatic_update = True
        try:
            self._pending_order_source_var.set(trading_source_display_label(origin=origin))
            self._pending_order_ticker_var.set(str(row.get("ticker") or ""))
            self._pending_order_qty_var.set(str(int(row.get("planned_qty") or 0)))
            self._pending_order_limit_var.set(str(row.get("limit_price") or ""))
            self._pending_fill_price_var.set("")
            self._pending_order_date_var.set(
                planned_date or datetime.now(timezone(timedelta(hours=8))).date().isoformat()
            )
            self._pending_fill_date_var.set("")
        finally:
            self._pending_draft_programmatic_update = False
        self._pending_draft_preview_var.set(
            f"編輯 {row.get('ticker')}｜載入合法掛單日期與自動買入限價…"
        )
        self._preview_pending_draft(use_current_overrides=True, silent=True)

    def _on_pending_tree_click(self, event):
        failure_row = _pending_sync_failure_at_cell(self._pending_tree, self._pending_rows, event)
        if failure_row is not None:
            # AI: This is a diagnostic read, not a selection/edit/retry command.
            messagebox.showerror(
                "掛單同步失敗",
                f"{failure_row.get('ticker') or '-'}\n\n{failure_row['sync_error']}",
                parent=self,
            )
            return "break"
        region = str(self._pending_tree.identify_region(event.x, event.y) or "")
        if region == "heading":
            return None
        entry_id = str(self._pending_tree.identify_row(event.y) or "")
        column = self._pending_tree.identify_column(event.x)
        if region == "cell" and column == "#1":
            row = self._pending_rows.get(entry_id)
            if row:
                self._open_ticker_in_inspector(str(row.get("ticker") or ""))
            return "break"
        if not entry_id:
            if self._pending_tree.selection() or self._pending_edit_entry_id is not None:
                clear_treeview_selection(self._pending_tree)
                self._reset_pending_draft_form(
                    message="已取消掛單選取；目前為新增掛單模式。"
                )
                return "break"
            return None

        # Drive the pending form from the canonical edit identity rather than
        # Treeview's transient visual selection.  A row can be visually selected
        # by a refresh without a <<TreeviewSelect>> callback; treating that as a
        # second-click deselect was the cause of a create-new preview against its
        # own ACTIVE reservation.
        if str(self._pending_edit_entry_id or "") == entry_id:
            clear_treeview_selection(self._pending_tree)
            self._reset_pending_draft_form(
                message="已取消掛單選取；目前為新增掛單模式。"
            )
            return "break"
        self._pending_tree.selection_set(entry_id)
        self._pending_tree.focus(entry_id)
        self._on_pending_selected()
        return "break"

    def _open_selected_pending_in_inspector(self, _event=None):
        if _pending_sync_failure_at_cell(self._pending_tree, self._pending_rows, _event) is not None:
            return "break"
        row = self._selected_pending_row()
        if row:
            self._open_ticker_in_inspector(str(row.get("ticker") or ""))

    def _schedule_manual_pending_ticker_preview(self, _event=None):
        if self._pending_draft_programmatic_update:
            return None
        self._invalidate_pending_preview()
        if hasattr(self, "_pending_submit_button"):
            self._pending_submit_button.configure(state="disabled")
        if hasattr(self, "_pending_fill_button"):
            self._pending_fill_button.configure(state="disabled")
        if self._pending_manual_preview_after_id is not None:
            try:
                self.after_cancel(self._pending_manual_preview_after_id)
            except tk.TclError as exc:
                _warn_gui_fallback("Trading pending ticker preview debounce", exc)
        ticker = self._pending_order_ticker_var.get().strip().upper()
        if len(ticker) < 4:
            self._pending_manual_preview_after_id = None
            return None
        self._pending_manual_preview_after_id = self.after(350, self._auto_preview_manual_pending_ticker)
        return None

    def _schedule_pending_value_preview(self, *_args):
        if self._pending_draft_programmatic_update:
            return None
        self._invalidate_pending_preview()
        if hasattr(self, "_pending_submit_button"):
            self._pending_submit_button.configure(state="disabled")
        if hasattr(self, "_pending_fill_button"):
            self._pending_fill_button.configure(state="disabled")
        ticker = self._pending_order_ticker_var.get().strip().upper()
        if len(ticker) < 4:
            return None
        if self._pending_manual_preview_after_id is not None:
            try:
                self.after_cancel(self._pending_manual_preview_after_id)
            except tk.TclError as exc:
                _warn_gui_fallback("Trading pending value preview debounce", exc)
        self._pending_manual_preview_after_id = self.after(250, self._auto_preview_pending_values)
        return None

    def _auto_preview_pending_values(self):
        self._pending_manual_preview_after_id = None
        self._preview_pending_draft(use_current_overrides=True, silent=True)

    def _auto_preview_manual_pending_ticker(self):
        self._pending_manual_preview_after_id = None
        self._on_manual_pending_ticker_commit()

    def _on_manual_pending_ticker_commit(self, _event=None):
        if self._pending_draft_programmatic_update:
            return None
        if self._pending_manual_preview_after_id is not None:
            try:
                self.after_cancel(self._pending_manual_preview_after_id)
            except tk.TclError as exc:
                _warn_gui_fallback("Trading pending ticker preview commit", exc)
            self._pending_manual_preview_after_id = None
        ticker = self._pending_order_ticker_var.get().strip().upper()
        if not ticker:
            return None
        scanner_ticker = str((self._pending_draft_candidate or {}).get("ticker") or "").strip().upper()
        edit_row = self._pending_rows.get(str(self._pending_edit_entry_id or "")) or {}
        edit_ticker = str(edit_row.get("ticker") or "").strip().upper()
        if self._pending_draft_origin == "scanner" and ticker == scanner_ticker:
            return self._preview_pending_draft(use_current_overrides=True, silent=True)
        if self._pending_edit_entry_id and ticker == edit_ticker:
            return self._preview_pending_draft(use_current_overrides=True, silent=True)

        # Duplicate ACTIVE pending intent is never a valid create-new action.
        # If the user types the ticker of the one existing ACTIVE row, bind the
        # form to that pending_entry_id and hydrate the full edit form instead of
        # running the create path and reporting that the ticker blocks itself.
        if self._bind_existing_pending_for_ticker(ticker, preserve_overrides=True):
            return self._preview_pending_draft(use_current_overrides=True, silent=True)

        self._pending_draft_origin = "manual"
        self._pending_draft_candidate = None
        self._pending_draft_programmatic_update = True
        try:
            self._pending_order_source_var.set(TRADING_SOURCE_CUSTOM_LABEL)
            self._pending_order_ticker_var.set(ticker)
            self._pending_order_qty_var.set("")
            self._pending_order_limit_var.set("")
            self._pending_fill_price_var.set("")
            self._pending_order_date_var.set("")
        finally:
            self._pending_draft_programmatic_update = False
        self._preview_pending_draft(use_current_overrides=False, silent=True)
        return None

    def _pending_draft_override_values(self, *, use_current_overrides: bool):
        ticker = self._pending_order_ticker_var.get().strip().upper()
        if not ticker:
            raise ValueError("股票代號必填")
        if not use_current_overrides:
            return ticker, None, None, self._pending_order_date_var.get().strip() or None
        qty_text = self._pending_order_qty_var.get().strip()
        qty = None if not qty_text else parse_trading_qty_text(qty_text, "掛單股數")
        # 買入限價是策略/手動計畫的自動計算結果；Workbench 不提供人工 override。
        limit_price = None
        planned_date = self._pending_order_date_var.get().strip() or None
        return ticker, qty, limit_price, planned_date

    def _build_pending_draft_request(self, *, use_current_overrides: bool):
        ticker, qty, limit_price, planned_date = self._pending_draft_override_values(
            use_current_overrides=use_current_overrides
        )
        origin = str(self._pending_draft_origin or "manual")
        candidate = dict(self._pending_draft_candidate or {}) if origin == "scanner" else None
        if origin == "scanner" and self._pending_edit_entry_id is None:
            if not candidate or str(candidate.get("ticker") or "").strip().upper() != ticker:
                raise ValueError("Scanner 掛單來源已改變；請重新選取 Scanner 股票或改用手動輸入")
        edit_id = str(self._pending_edit_entry_id or "").strip()
        existing_entry = dict(self._pending_rows.get(edit_id) or self._pending_edit_snapshot or {})
        if edit_id:
            if not existing_entry or str(existing_entry.get("pending_entry_id") or "") != edit_id:
                raise RuntimeError("目前選取的掛單已不存在或已更新；請重新選取掛單。")
            existing_ticker = str(existing_entry.get("ticker") or "").strip().upper()
            if ticker != existing_ticker:
                raise ValueError("更新掛單不可變更股票；請先取消選取，再以新增掛單建立另一支股票。")
            origin = "scanner" if str(existing_entry.get("origin") or "") == "scanner_strategy" else "manual"
            candidate = None
        return {
            "pending_entry_id": edit_id or None,
            "origin": origin,
            "candidate": candidate,
            "existing_entry": existing_entry,
            "ticker": ticker,
            "qty": qty,
            "limit_price": limit_price,
            "planned_date": planned_date,
            "fill_date": self._pending_fill_date_var.get().strip() or None,
            "latest_finalized_date": str(
                self._candidate_payload.get("latest_data_date")
                or self._snapshot.get("latest_data_date")
                or ""
            ).strip() or None,
        }

    @staticmethod
    def _pending_draft_worker(request):
        payload = dict(request or {})
        pending_entry_id = str(payload.get("pending_entry_id") or "").strip()
        existing_entry = dict(payload.get("existing_entry") or {})

        if pending_entry_id:
            row = preview_trading_pending_entry_update(
                WORKBENCH_PROJECT_ROOT,
                pending_entry_id=pending_entry_id,
                origin=str(payload.get("origin") or "manual"),
                ticker=payload.get("ticker"),
                qty=payload.get("qty"),
                limit_price=payload.get("limit_price"),
                planned_trade_date=payload.get("planned_date"),
                candidate_reference=dict(payload.get("candidate") or {}) or None,
            )
        elif str(payload.get("origin") or "manual") == "scanner":
            row = preview_scanner_trading_pending_entry(
                WORKBENCH_PROJECT_ROOT,
                candidate_reference=dict(payload.get("candidate") or {}),
                qty=payload.get("qty"),
                limit_price=payload.get("limit_price"),
                planned_trade_date=payload.get("planned_date"),
            )
        else:
            row = preview_manual_trading_pending_entry(
                WORKBENCH_PROJECT_ROOT,
                ticker=payload.get("ticker"),
                qty=payload.get("qty"),
                limit_price=payload.get("limit_price"),
                planned_trade_date=payload.get("planned_date"),
            )

        row = dict(row or {})
        runtime = load_trading_scanner_runtime(WORKBENCH_PROJECT_ROOT)
        latest_finalized_date = (
            str(payload.get("latest_finalized_date") or "").strip()
            or str(runtime.get("latest_data_date") or "").strip()
            or str(row.get("information_date") or "")
        )
        order_constraints = build_trading_pending_order_form_constraints(
            WORKBENCH_PROJECT_ROOT,
            ticker=row.get("ticker"),
            information_date=row.get("information_date"),
            selected_date=row.get("planned_trade_date"),
            include_fill_dates=False,
            pending_limit_price=row.get("limit_price"),
            pending_order_date=None,
            latest_finalized_date=latest_finalized_date,
        )
        row["_order_form_context"] = ORDER_FORM_DATE_KIND_PENDING
        row["_order_form_constraints"] = order_constraints

        fill_constraints = {}
        if pending_entry_id and existing_entry:
            fill_constraints = build_trading_actual_fill_form_constraints(
                WORKBENCH_PROJECT_ROOT,
                ticker=row.get("ticker"),
                selected_date=payload.get("fill_date"),
                latest_finalized_date=latest_finalized_date,
                earliest_exclusive_date=row.get("planned_trade_date"),
                max_price=row.get("limit_price"),
                auto_select_preferred=False,
            )
        row["_fill_form_constraints"] = fill_constraints
        row["_form_fill_date"] = payload.get("fill_date")
        return row

    @staticmethod
    def _normalize_price_option_text(value) -> str:
        try:
            return f"{float(value):.3f}".rstrip("0").rstrip(".")
        except (TypeError, ValueError):
            return str(value or "").strip()

    def _apply_pending_order_form_constraints(self, constraints, *, editing: bool) -> None:
        payload = dict(constraints or {})
        self._pending_order_form_constraints = payload
        self._configure_pending_mode_buttons()

    def _apply_pending_fill_constraints(self, constraints, *, editing: bool) -> None:
        payload = dict(constraints or {})
        self._pending_fill_form_constraints = payload
        allowed_dates = tuple(payload.get("allowed_dates") or ())
        price_options = tuple(str(value) for value in (payload.get("price_options") or ()))
        self._pending_price_option_set = frozenset(price_options)
        self._pending_fill_date_field.set_allowed_dates(allowed_dates if editing else ())
        self._pending_fill_price_combo.configure(
            values=price_options,
            state="readonly" if editing and price_options else "disabled",
        )
        self._refresh_pending_fill_button_state()

    def _refresh_pending_fill_button_state(self, *_args) -> None:
        if not hasattr(self, "_pending_fill_button"):
            return
        editing = self._pending_edit_entry_id is not None
        constraints = dict(self._pending_fill_form_constraints or {})
        fill_date = self._pending_fill_date_var.get().strip()
        fill_price_text = self._normalize_price_option_text(self._pending_fill_price_var.get())
        valid_date = bool(editing and fill_date and fill_date in set(constraints.get("allowed_dates") or ()))
        valid_price = bool(fill_price_text and fill_price_text in self._pending_price_option_set)
        self._configure_pending_mode_buttons()
        self._pending_fill_button.configure(
            text="確認成交",
            state="normal" if valid_date and valid_price else "disabled",
        )
        if fill_date and self._pending_fill_price_var.get().strip() and not valid_price:
            self._pending_draft_preview_var.set(
                f"成交價 {self._pending_fill_price_var.get().strip()} 不在成交日 {fill_date} 的合法 tick 範圍；請重新選擇。"
            )

    def _schedule_pending_fill_preview(self, *_args):
        if self._pending_draft_programmatic_update or self._pending_edit_entry_id is None:
            return None
        # Selecting a fill date switches the shared form into fill mode
        # immediately. Order-intent fields/actions stay locked until the fill
        # date is cleared, even while the market-evidence preview is loading.
        self._configure_pending_mode_buttons()
        self._invalidate_pending_preview()
        return self._preview_pending_draft(use_current_overrides=True, silent=True)

    def _apply_pending_draft_preview(self, result):
        row = dict(result or {})
        if not row:
            return
        order_constraints = dict(row.get("_order_form_constraints") or {})
        fill_constraints = dict(row.get("_fill_form_constraints") or {})
        editing = self._pending_edit_entry_id is not None
        self._pending_draft_programmatic_update = True
        try:
            source = trading_source_display_label(origin=row.get("origin"))
            self._pending_order_source_var.set(source)
            self._pending_order_ticker_var.set(str(row.get("ticker") or ""))
            self._pending_order_limit_var.set(str(row.get("limit_price") or ""))
            self._pending_order_qty_var.set(str(int(row.get("planned_qty") or 0)))
            self._pending_order_date_var.set(str(row.get("planned_trade_date") or ""))
            if not editing:
                self._pending_fill_price_var.set("")
                self._pending_fill_date_var.set("")
            elif row.get("_form_fill_date"):
                self._pending_fill_date_var.set(str(row.get("_form_fill_date")))
        finally:
            self._pending_draft_programmatic_update = False

        self._apply_pending_order_form_constraints(order_constraints, editing=editing)
        self._apply_pending_fill_constraints(fill_constraints, editing=editing)
        self._pending_draft_preview_var.set(
            f"預留成本 {float(row.get('reserved_cost') or 0):,.0f}｜"
            f"買入限價 {row.get('limit_price')}｜停損 {row.get('init_sl')}｜停利 {row.get('target_price')}"
        )

    def _invalidate_pending_preview(self) -> None:
        # Invalidate already-running background work immediately.  This is called
        # on selection/field changes, not only when the next worker eventually starts.
        self._pending_preview_token += 1

    def _clear_pending_constraint_options(self) -> None:
        self._pending_order_form_constraints = {}
        self._pending_fill_form_constraints = {}
        self._pending_price_option_set = frozenset()
        if hasattr(self, "_pending_fill_price_combo"):
            self._pending_fill_price_combo.configure(values=(), state="disabled")
        if hasattr(self, "_pending_order_date_field"):
            self._pending_order_date_field.set_allowed_dates(())
        if hasattr(self, "_pending_fill_date_field"):
            self._pending_fill_date_field.set_allowed_dates(())

    @staticmethod
    def _pending_constraint_worker(request: dict) -> dict:
        """Resolve order/fill selectors even when account sizing preview fails."""

        payload = dict(request or {})
        existing_entry = dict(payload.get("existing_entry") or {})
        ticker = str(payload.get("ticker") or existing_entry.get("ticker") or "").strip().upper()
        if not ticker:
            raise ValueError("股票代號必填")

        latest_finalized_date = str(payload.get("latest_finalized_date") or "").strip()
        if not latest_finalized_date:
            runtime = load_trading_scanner_runtime(WORKBENCH_PROJECT_ROOT)
            latest_finalized_date = str(runtime.get("latest_data_date") or "").strip()
        if not latest_finalized_date:
            raise RuntimeError("Trading 最新 finalized 日期不可用")

        candidate = dict(payload.get("candidate") or {})
        information_date = str(
            existing_entry.get("information_date")
            or candidate.get("trade_date")
            or candidate.get("candidate_date")
            or latest_finalized_date
        ).strip()
        order_constraints = build_trading_pending_order_form_constraints(
            WORKBENCH_PROJECT_ROOT,
            ticker=ticker,
            information_date=information_date,
            selected_date=payload.get("planned_date"),
            include_fill_dates=False,
            pending_limit_price=(existing_entry.get("limit_price") if existing_entry else None),
            pending_order_date=None,
            latest_finalized_date=latest_finalized_date,
        )
        fill_constraints = {}
        if existing_entry:
            fill_constraints = build_trading_actual_fill_form_constraints(
                WORKBENCH_PROJECT_ROOT,
                ticker=ticker,
                selected_date=payload.get("fill_date"),
                latest_finalized_date=latest_finalized_date,
                earliest_exclusive_date=payload.get("planned_date") or existing_entry.get("planned_trade_date"),
                max_price=existing_entry.get("limit_price"),
                auto_select_preferred=False,
            )
        return {"order": order_constraints, "fill": fill_constraints}

    @staticmethod
    def _pending_preview_error_message(error: Exception) -> str:
        text = str(error or "").strip() or "未知錯誤"
        lowered = text.casefold()
        if any(token in text for token in ("可用現金", "股數", "資金", "slot", "超過")):
            prefix = "需調整"
        elif any(token in lowered for token in ("stale", "candidate")) and any(token in text for token in ("過期", "不一致", "來源已改變", "snapshot")):
            prefix = "資料已過期"
        elif any(token in text for token in ("市場資料", "交易資料", "OHLC", "價格區間", "Trading adjusted", "TaiwanStockPrice")):
            prefix = "缺少市場資料"
        else:
            prefix = "無法試算"
        return f"{prefix}：{text}"

    def _pending_preview_worker(self, token: int, request: dict, silent: bool) -> None:
        result = None
        error = None
        fallback_constraints = None
        try:
            result = self._pending_draft_worker(request)
        except Exception as exc:
            error = exc
            try:
                fallback_constraints = self._pending_constraint_worker(request)
            except Exception:
                fallback_constraints = None
        self._pending_preview_results.put((token, result, error, bool(silent), fallback_constraints))

    def _schedule_pending_preview_poll(self) -> None:
        if self._pending_preview_poll_after_id is None:
            self._pending_preview_poll_after_id = self.after(30, self._drain_pending_preview_results)

    def _drain_pending_preview_results(self) -> None:
        self._pending_preview_poll_after_id = None
        while True:
            try:
                token, result, error, silent, fallback_constraints = self._pending_preview_results.get_nowait()
            except queue.Empty:
                break
            self._pending_preview_inflight = max(0, int(self._pending_preview_inflight) - 1)
            if int(token) != int(self._pending_preview_token):
                continue
            if error is not None:
                if fallback_constraints:
                    self._apply_pending_order_form_constraints(
                        dict(fallback_constraints.get("order") or {}),
                        editing=self._pending_edit_entry_id is not None,
                    )
                    self._apply_pending_fill_constraints(
                        dict(fallback_constraints.get("fill") or {}),
                        editing=self._pending_edit_entry_id is not None,
                    )
                message = self._pending_preview_error_message(error)
                if silent:
                    # Selection stays visible even when account-level preview fails.
                    self._pending_draft_preview_var.set(message)
                else:
                    messagebox.showerror("掛單試算失敗", str(error), parent=self)
            else:
                self._apply_pending_draft_preview(result)
        if self._pending_preview_inflight > 0:
            self._schedule_pending_preview_poll()

    def _preview_pending_draft(self, *, use_current_overrides: bool, silent: bool = False):
        ticker = self._pending_order_ticker_var.get().strip().upper()
        if not ticker:
            return False
        # Run this synchronously before any preview request.  Date/qty edits can
        # cancel the delayed ticker-commit callback, so relying on that callback
        # alone leaves the form on the create-new path and makes an existing
        # ACTIVE pending order conflict with itself.
        self._bind_existing_pending_for_ticker(
            ticker, preserve_overrides=bool(use_current_overrides)
        )
        try:
            request = self._build_pending_draft_request(use_current_overrides=use_current_overrides)
        except (ValueError, RuntimeError) as exc:
            message = self._pending_preview_error_message(exc)
            if silent:
                self._pending_draft_preview_var.set(message)
            else:
                messagebox.showerror("掛單試算失敗", str(exc), parent=self)
            return False
        self._pending_preview_token += 1
        token = int(self._pending_preview_token)
        self._pending_preview_inflight += 1
        self._clear_pending_constraint_options()
        self._pending_submit_button.configure(state="disabled")
        self._pending_fill_button.configure(state="disabled")
        self._pending_draft_preview_var.set(f"{ticker}｜載入中…")
        thread = threading.Thread(
            target=self._pending_preview_worker,
            args=(token, request, bool(silent)),
            name=f"workbench-pending-preview-{token}",
            daemon=True,
        )
        self._pending_preview_thread = thread
        thread.start()
        self._schedule_pending_preview_poll()
        return True

    def _preview_current_pending_draft(self):
        return self._preview_pending_draft(use_current_overrides=True, silent=False)

    def _confirm_submit_pending_draft(self):
        if self._pending_fill_mode_active():
            messagebox.showerror(
                "掛單",
                "已選擇成交日；請先清除成交日，才能修改或更新掛單。",
                parent=self,
            )
            return
        ticker = self._pending_order_ticker_var.get().strip().upper()
        if not ticker:
            messagebox.showerror("掛單", "股票代號必填。", parent=self)
            return
        try:
            request = self._build_pending_draft_request(use_current_overrides=True)
        except (ValueError, RuntimeError) as exc:
            messagebox.showerror("掛單確認失敗", str(exc), parent=self)
            return

        editing = bool(request.get("pending_entry_id"))

        def preview_success(result):
            row = dict(result or {})
            self._apply_pending_draft_preview(row)
            action_text = "更新掛單" if editing else "確認掛單"
            if not messagebox.askyesno(
                action_text,
                f"{row.get('ticker')}｜{int(row.get('planned_qty') or 0):,} 股｜限價 {row.get('limit_price')}｜{row.get('planned_trade_date')}\n"
                f"預留成本 {float(row.get('reserved_cost') or 0):,.0f}｜停損 {row.get('init_sl')}｜停利 {row.get('target_price')}\n\n"
                + ("確認後會原子取代原掛單，不會重複計算 reservation。" if editing else "確認後才會建立正式掛單並鎖定資金／持股 slot。"),
                parent=self,
            ):
                return
            qty = int(row.get("planned_qty") or 0)
            planned_date = str(row.get("planned_trade_date") or "")
            request_origin = str(request.get("origin") or "manual")
            if editing:
                worker = lambda: update_trading_pending_entry_intent(
                    WORKBENCH_PROJECT_ROOT,
                    pending_entry_id=str(request.get("pending_entry_id") or ""),
                    origin=request_origin,
                    ticker=str(row.get("ticker") or request.get("ticker") or ticker),
                    qty=qty,
                    limit_price=None,
                    planned_trade_date=planned_date,
                    candidate_reference=dict(request.get("candidate") or {}) or None,
                )
            elif request_origin == "scanner":
                candidate = dict(request.get("candidate") or {})
                worker = lambda: create_scanner_trading_pending_entry(
                    WORKBENCH_PROJECT_ROOT,
                    candidate_reference=candidate,
                    qty=qty,
                    limit_price=None,
                    planned_trade_date=planned_date,
                )
            else:
                worker = lambda: create_manual_trading_pending_entry(
                    WORKBENCH_PROJECT_ROOT,
                    ticker=str(row.get("ticker") or request.get("ticker") or ticker),
                    qty=qty,
                    limit_price=None,
                    planned_trade_date=planned_date,
                )

            def save_success(saved):
                saved_row = dict(saved or {})
                verb = "已更新" if editing else "已加入掛單區"
                self._reset_pending_draft_form(
                    message=f"{saved_row.get('ticker')} {verb}；資源與 Scanner Pool 已同步。"
                )
                messagebox.showinfo(action_text, f"{saved_row.get('ticker')} {verb}。", parent=self)

            self._submit_trading_command(
                f"{row.get('ticker')} {action_text}",
                worker,
                on_success=save_success,
                error_title=f"{action_text}失敗",
                refresh_state=True,
                state_domains=PENDING_MUTATION_DOMAINS,
            )

        self._submit_trading_command(
            f"{ticker} 掛單送出前確認",
            lambda request=request: self._pending_draft_worker(request),
            on_success=preview_success,
            error_title="掛單確認失敗",
            refresh_state=False,
        )

    def _pending_fill_values(self):
        row = self._selected_pending_row()
        if not row:
            raise ValueError("請先選取掛單")
        constraints = dict(self._pending_fill_form_constraints or {})
        qty = parse_trading_qty_text(self._pending_order_qty_var.get(), "實際成交股數")
        price = parse_trading_money_text(self._pending_fill_price_var.get(), "實際成交價", allow_zero=False)
        trade_date = self._pending_fill_date_var.get().strip()
        if not trade_date:
            raise ValueError("成交日必填")
        if trade_date not in set(constraints.get("allowed_dates") or ()):
            raise ValueError("成交日必須晚於掛單日、不得超過最新 finalized 日期，且該股當日需有正式交易證據")
        normalized_price = self._normalize_price_option_text(price)
        if normalized_price not in self._pending_price_option_set:
            raise ValueError("實際成交價不在所選成交日的合法市場價格 ticks 內")
        return row, qty, price, trade_date

    def _confirm_pending_fill(self):
        try:
            row, qty, price, trade_date = self._pending_fill_values()
            preview = preview_trading_pending_entry_fill(
                WORKBENCH_PROJECT_ROOT,
                pending_entry_id=str(row.get("pending_entry_id") or ""),
                qty=qty,
                price=price,
                trade_date=trade_date,
            )
        except (ValueError, RuntimeError, OSError, FileNotFoundError) as exc:
            messagebox.showerror("掛單成交確認失敗", str(exc), parent=self)
            return
        evidence = dict(preview.get("market_evidence") or {})
        warning_rows = [str(item) for item in list(preview.get("warnings") or []) if str(item).strip()]
        warning_text = "" if not warning_rows else "\n\n注意：\n- " + "\n- ".join(warning_rows)
        if not messagebox.askyesno(
            "確認掛單成交",
            f"{row.get('ticker')}｜{qty:,} 股 @ {price}｜{trade_date}\n"
            f"掛單限價 {row.get('limit_price')}｜規劃 {int(row.get('planned_qty') or 0):,} 股\n"
            f"市場證據 Low {evidence.get('market_low')} / High {evidence.get('market_high')}"
            f"{warning_text}\n\n確認後將轉入持股區。",
            parent=self,
        ):
            return
        entry_id = str(row.get("pending_entry_id") or "")
        ticker = str(row.get("ticker") or "")

        def on_success(_result):
            self._reset_pending_draft_form(message=f"{ticker} 已成交並轉入持股區；掛單與帳務狀態已同步。")
            messagebox.showinfo("掛單成交", f"{ticker} 已轉入持股區。", parent=self)

        self._submit_trading_command(
            f"{ticker} 掛單成交轉持股",
            lambda: fill_trading_pending_entry(
                WORKBENCH_PROJECT_ROOT, pending_entry_id=entry_id, qty=qty, price=price, trade_date=trade_date,
                expected_account_revision=None,
            ),
            on_success=on_success,
            error_title="掛單成交失敗",
            refresh_state=True,
            state_domains=PENDING_FILL_MUTATION_DOMAINS,
        )

    def _delete_selected_pending(self):
        if self._pending_fill_mode_active():
            messagebox.showerror(
                "刪除掛單",
                "已選擇成交日；請先清除成交日，才能刪除掛單。",
                parent=self,
            )
            return
        row = self._selected_pending_row()
        if not row:
            messagebox.showerror("刪除掛單", "請先選取掛單。", parent=self)
            return
        ticker = str(row.get("ticker") or "")
        if not messagebox.askyesno(
            "確認刪除掛單",
            f"刪除 {ticker} 尚未執行的掛單？\n\n刪除後會立即釋放此掛單的預留資金與持股 slot，之後可重新建立掛單。",
            parent=self,
        ):
            return
        entry_id = str(row.get("pending_entry_id") or "")
        self._submit_trading_command(
            f"{ticker} 刪除掛單",
            lambda: delete_pending_entry(
                WORKBENCH_PROJECT_ROOT, pending_entry_id=entry_id, note="Workbench user delete"
            ),
            on_success=lambda _result: (
                self._reset_pending_draft_form(message=f"{ticker} 掛單已刪除；預留資源與 Scanner Pool 已同步。"),
                messagebox.showinfo("刪除掛單", f"{ticker} 掛單已刪除，預留資源已釋放。", parent=self),
            ),
            error_title="刪除掛單失敗",
            refresh_state=True,
            state_domains=PENDING_MUTATION_DOMAINS,
        )

    def _open_ticker_in_inspector(self, ticker, *, candidate_row=None):
        ticker = str(ticker or "").strip().upper()
        if not ticker:
            return
        callback = getattr(self.winfo_toplevel(), "_open_single_stock_inspector", None)
        if not callable(callback):
            messagebox.showerror("單股回測檢視", "Workbench 尚未提供單股回測導覽。", parent=self)
            return
        callback(
            ticker,
            runtime_domain="trading",
            auto_run=True,
            candidate_row=dict(candidate_row or {}),
            candidate_rows=[dict(row) for row in list(self._candidate_rows or [])],
            candidate_latest_data_date=self._candidate_payload.get("latest_data_date"),
        )

    def _open_candidate_ticker_in_inspector(self, ticker):
        ticker = str(ticker or "").strip().upper()
        candidate_row = dict(getattr(self, "_candidate_by_ticker", {}).get(ticker) or {})
        self._open_ticker_in_inspector(ticker, candidate_row=candidate_row)

    def _open_selected_candidate_in_inspector(self, _event=None):
        ticker = self._selected_candidate_ticker()
        if ticker:
            candidate_row = dict(getattr(self, "_candidate_by_ticker", {}).get(ticker) or {})
            self._open_ticker_in_inspector(ticker, candidate_row=candidate_row)

    def _open_selected_position_in_inspector(self, _event=None):
        ticker = self._selected_ticker()
        if ticker:
            self._open_ticker_in_inspector(ticker)
