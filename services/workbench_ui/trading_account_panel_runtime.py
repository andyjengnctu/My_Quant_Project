from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from services.trading.lifecycle_sync import SYNC_STATUS_FAILED
from services.workbench_ui.state_sync import ACCOUNT_MUTATION_DOMAINS, STATE_MARKET_DATA, normalize_state_domains
from services.workbench_ui.paged_table import PagedTable
from services.workbench_ui.stock_names import load_workbench_stock_name_map, workbench_stock_name, workbench_stock_name_cache_ready
from services.workbench_ui.workbench import WORKBENCH_PANEL_ACTIVATION_DELAY_MS, _warn_gui_fallback
from services.workbench_ui.trading_account_panel_data import (
    WORKBENCH_PROJECT_ROOT,
    build_trading_account_panel_initial_bundle,
)

class TradingAccountRuntimeMixin:
    """Trading Workbench TradingAccountRuntime responsibilities; composed by TradingAccountPanel."""

    def on_workbench_visibility_changed(self, active: bool) -> None:
        """Start/apply Trading startup work only while this notebook page is visible."""

        self._workbench_active = bool(active)
        if self._workbench_active:
            self._ensure_stock_name_warm()
        if not self._workbench_active:
            if self._initial_state_start_after_id is not None:
                try:
                    self.after_cancel(self._initial_state_start_after_id)
                except tk.TclError as exc:
                    _warn_gui_fallback("Trading initial load after_cancel", exc)
                self._initial_state_start_after_id = None
            if self._initial_apply_after_id is not None:
                try:
                    self.after_cancel(self._initial_apply_after_id)
                except tk.TclError as exc:
                    _warn_gui_fallback("Trading initial render after_cancel", exc)
                self._initial_apply_after_id = None
            return

        if self._deferred_initial_bundle is not None and not self._initial_apply_bundle_active:
            bundle = self._deferred_initial_bundle
            self._deferred_initial_bundle = None
            self._begin_initial_state_bundle_apply(bundle)
            return
        if self._initial_apply_bundle_active:
            self._schedule_initial_apply_step()
            return
        if self._initial_state_started or self._initial_state_applied:
            return
        if self._initial_state_start_after_id is None:
            self._initial_state_start_after_id = self.after(
                WORKBENCH_PANEL_ACTIVATION_DELAY_MS,
                self._start_initial_state_load,
            )

    def _ensure_stock_name_warm(self) -> None:
        """Warm display-only names without delaying canonical Trading data."""

        if workbench_stock_name_cache_ready(WORKBENCH_PROJECT_ROOT):
            if self._initial_state_applied and self._workbench_active:
                self._refresh_rendered_stock_names()
            return
        if self._stock_name_warm_thread is None or not self._stock_name_warm_thread.is_alive():
            self._stock_name_warm_error = None

            def _worker():
                try:
                    load_workbench_stock_name_map(WORKBENCH_PROJECT_ROOT)
                except Exception as exc:
                    self._stock_name_warm_error = f"{type(exc).__name__}: {exc}"

            self._stock_name_warm_thread = threading.Thread(
                target=_worker,
                name="workbench-trading-stock-names",
                daemon=True,
            )
            self._stock_name_warm_thread.start()
        if self._stock_name_warm_after_id is None:
            self._stock_name_warm_after_id = self.after(100, self._poll_stock_name_warm)

    def _poll_stock_name_warm(self) -> None:
        self._stock_name_warm_after_id = None
        thread = self._stock_name_warm_thread
        if thread is not None and thread.is_alive():
            self._stock_name_warm_after_id = self.after(100, self._poll_stock_name_warm)
            return
        self._stock_name_warm_thread = None
        if self._stock_name_warm_error is not None:
            return
        if self._workbench_active:
            self._refresh_rendered_stock_names()

    def _refresh_rendered_stock_names(self) -> None:
        """Patch visible name cells from cache only; perform no domain I/O."""

        candidate = getattr(self, "_candidate_tree", None)
        if isinstance(candidate, PagedTable):
            rows = [dict(row) for row in list(getattr(candidate, "_rows", []))]
            for row in rows:
                row["stock_name"] = workbench_stock_name(WORKBENCH_PROJECT_ROOT, row.get("ticker"))
            candidate.set_rows(rows, preserve_selection=True)

        for attr_name in (
            "_tree", "_pending_tree", "_proposed_tree", "_order_tree",
            "_protection_tree", "_indicator_tree",
        ):
            tree = getattr(self, attr_name, None)
            if not isinstance(tree, ttk.Treeview):
                continue
            columns = tuple(str(value) for value in tree.cget("columns"))
            if "ticker" not in columns or "stock_name" not in columns:
                continue
            ticker_index = columns.index("ticker")
            name_index = columns.index("stock_name")
            for item_id in tree.get_children():
                values = list(tree.item(item_id, "values"))
                if ticker_index >= len(values) or name_index >= len(values):
                    continue
                values[name_index] = workbench_stock_name(
                    WORKBENCH_PROJECT_ROOT, values[ticker_index]
                )
                tree.item(item_id, values=values)

        for ticker_var_name, name_var_name in (
            ("_ticker_var", "_ticker_name_var"),
            ("_pending_order_ticker_var", "_pending_order_stock_name_var"),
            ("_trade_ticker_var", "_trade_stock_name_var"),
        ):
            ticker_var = getattr(self, ticker_var_name, None)
            name_var = getattr(self, name_var_name, None)
            if ticker_var is not None and name_var is not None:
                name_var.set(workbench_stock_name(WORKBENCH_PROJECT_ROOT, ticker_var.get()))

    def _set_initial_loading_state(self):
        self._operations_next_var.set("LOADING | Trading 狀態載入中… | 下一步：等待載入完成")
        self._set_workflow_buttons_state("disabled")
        for button_name in ("_initialize_button", "_set_cash_button", "_add_button", "_correct_button", "_remove_button", "_confirm_ordered_button"):
            button = getattr(self, button_name, None)
            if button is not None:
                button.configure(state="disabled")

    def _start_initial_state_load(self):
        self._initial_state_start_after_id = None
        if not self._workbench_active:
            return
        if self._initial_state_thread is not None and self._initial_state_thread.is_alive():
            return
        if self._initial_state_started or self._initial_state_applied:
            return
        self._initial_state_started = True
        self._initial_state_token += 1
        token = int(self._initial_state_token)
        thread = threading.Thread(
            target=self._initial_state_load_worker,
            args=(token,),
            name="workbench-trading-initial-state",
            daemon=True,
        )
        self._initial_state_thread = thread
        thread.start()
        self._schedule_initial_state_poll()

    def _initial_state_load_worker(self, token: int):
        bundle = None
        error = None
        try:
            # Read-only first-open work stays on this background thread.  Do not
            # queue it behind unrelated center startup I/O; shared V2 verification
            # already serializes/cache-shares at its canonical owner.
            bundle = build_trading_account_panel_initial_bundle(WORKBENCH_PROJECT_ROOT)
        except Exception as exc:
            error = exc
        self._initial_state_results.put((int(token), bundle, error))

    def _schedule_initial_state_poll(self):
        if self._initial_state_poll_after_id is None:
            self._initial_state_poll_after_id = self.after(25, self._drain_initial_state_results)

    def _drain_initial_state_results(self):
        self._initial_state_poll_after_id = None
        while True:
            try:
                token, bundle, error = self._initial_state_results.get_nowait()
            except queue.Empty:
                break
            self._finish_initial_state_load(token, bundle, error)
        if self._initial_state_thread is not None and self._initial_state_thread.is_alive():
            self._schedule_initial_state_poll()

    def _prepare_state_bundle_for_render(self, bundle: dict[str, object]) -> None:
        lifecycle_ok, lifecycle_value = dict(bundle or {}).get("lifecycle_sync", (True, {}))
        if lifecycle_ok:
            self._lifecycle_sync_snapshot = dict(lifecycle_value or {})
        else:
            self._lifecycle_sync_snapshot = {
                "status": SYNC_STATUS_FAILED,
                "error": f"{type(lifecycle_value).__name__}: {lifecycle_value}",
                "pending_errors": {},
                "position_errors": {},
                "position_status_by_ticker": {},
            }
        self._initial_preloaded = dict(bundle or {})

    def _apply_state_bundle(self, bundle: dict[str, object]) -> None:
        """Apply an already-read canonical Trading state bundle on the Tk thread only."""
        self._prepare_state_bundle_for_render(bundle)
        self._suspend_operations_refresh = True
        try:
            self.refresh_account()
            self.refresh_account_dashboard()
            self.refresh_candidate_snapshot_rows()
            self.refresh_pending_entries()
            self.refresh_protection_plan()
            self.refresh_indicator_exit_plan()
            self.refresh_daily_workflow()
        finally:
            self._suspend_operations_refresh = False
        self.refresh_operations_status()
        self._initial_preloaded.clear()

    def _begin_initial_state_bundle_apply(self, bundle: dict[str, object]) -> None:
        """Render first-open Trading state in short Tk turns and pause while hidden."""

        if self._initial_apply_bundle_active:
            self._deferred_initial_bundle = dict(bundle or {})
            return
        self._prepare_state_bundle_for_render(bundle)
        self._suspend_operations_refresh = True
        self._initial_apply_steps = [
            self.refresh_account,
            self.refresh_account_dashboard,
            self.refresh_candidate_snapshot_rows,
            self.refresh_pending_entries,
            self.refresh_protection_plan,
            self.refresh_indicator_exit_plan,
            self.refresh_daily_workflow,
        ]
        self._initial_apply_bundle_active = True
        self._schedule_initial_apply_step()

    def _schedule_initial_apply_step(self) -> None:
        if not self._workbench_active or not self._initial_apply_bundle_active:
            return
        if self._initial_apply_after_id is None:
            self._initial_apply_after_id = self.after(1, self._run_initial_apply_step)

    def _run_initial_apply_step(self) -> None:
        self._initial_apply_after_id = None
        if not self._workbench_active or not self._initial_apply_bundle_active:
            return
        if self._initial_apply_steps:
            step = self._initial_apply_steps.pop(0)
            try:
                step()
            except Exception as exc:
                self._suspend_operations_refresh = False
                self._initial_apply_steps.clear()
                self._initial_preloaded.clear()
                self._initial_apply_bundle_active = False
                self._initial_state_started = False
                self._operations_next_var.set(
                    f"FAIL | Trading 初始畫面建立失敗：{type(exc).__name__}: {exc}"
                )
                return
            self._schedule_initial_apply_step()
            return

        self._suspend_operations_refresh = False
        try:
            self.refresh_operations_status()
        except Exception as exc:
            self._initial_state_started = False
            self._operations_next_var.set(
                f"FAIL | Trading 初始畫面建立失敗：{type(exc).__name__}: {exc}"
            )
            return
        finally:
            self._initial_preloaded.clear()
            self._initial_apply_bundle_active = False
        self._initial_state_applied = True
        self._schedule_pending_external_account_refresh()

    def _finish_initial_state_load(self, token: int, bundle, error=None):
        if int(token) != int(self._initial_state_token):
            return
        self._initial_state_thread = None
        if error is not None:
            self._initial_state_started = False
            self._operations_next_var.set(
                f"FAIL | Trading 初始狀態載入失敗：{type(error).__name__}: {error}"
            )
            return
        if not isinstance(bundle, dict):
            self._initial_state_started = False
            self._operations_next_var.set("FAIL | Trading 初始狀態載入失敗：bundle 格式不合法")
            return
        if not self._workbench_active:
            self._deferred_initial_bundle = dict(bundle)
            return
        self._begin_initial_state_bundle_apply(bundle)

    def _iter_action_buttons(self):
        stack = [self]
        while stack:
            parent = stack.pop()
            for child in parent.winfo_children():
                stack.append(child)
                if isinstance(child, ttk.Button):
                    yield child

    def _set_command_busy(self, busy: bool, *, label: str = "") -> None:
        if busy:
            self._command_saved_button_states = {}
            for button in self._iter_action_buttons():
                try:
                    state = str(button.cget("state") or "normal")
                    self._command_saved_button_states[button] = state
                    button.configure(state="disabled")
                except tk.TclError as exc:
                    _warn_gui_fallback("Trading command disable button", exc)
            self._operations_next_var.set(f"BUSY | {label} 執行中… | 下一步：等待完成")
            return
        saved = dict(self._command_saved_button_states)
        self._command_saved_button_states.clear()
        for button, state in saved.items():
            try:
                if button.winfo_exists():
                    button.configure(state=state)
            except tk.TclError as exc:
                _warn_gui_fallback("Trading command restore button", exc)

    def _submit_trading_command(
        self,
        label: str,
        worker,
        *,
        on_success=None,
        on_error=None,
        error_title: str = "Trading 操作失敗",
        refresh_state: bool = True,
        state_domains=None,
    ) -> bool:
        if (
            (self._initial_state_thread is not None and self._initial_state_thread.is_alive())
            or bool(getattr(self, "_initial_apply_bundle_active", False))
        ):
            self._operations_next_var.set("WAIT | Trading 初始狀態仍在載入中…")
            return False
        if self._command_thread is not None and self._command_thread.is_alive():
            self._operations_next_var.set(f"BUSY | {label} 尚未執行；目前已有 Trading command 執行中。")
            return False
        self._command_token += 1
        token = int(self._command_token)
        self._command_context[token] = {
            "label": str(label),
            "on_success": on_success,
            "on_error": on_error,
            "error_title": str(error_title),
            "state_domains": normalize_state_domains(state_domains),
        }
        self._set_command_busy(True, label=str(label))
        thread = threading.Thread(
            target=self._trading_command_worker,
            args=(token, worker, bool(refresh_state)),
            name=f"workbench-trading-command-{token}",
            daemon=True,
        )
        self._command_thread = thread
        thread.start()
        self._schedule_command_poll()
        return True

    def _trading_command_worker(self, token: int, worker, refresh_state: bool) -> None:
        result = None
        command_error = None
        try:
            result = worker()
        except Exception as exc:  # surfaced on Tk thread with the command context
            command_error = exc

        bundle = None
        bundle_error = None
        if refresh_state:
            try:
                bundle = build_trading_account_panel_initial_bundle(WORKBENCH_PROJECT_ROOT)
            except Exception as exc:
                bundle_error = exc
        self._command_results.put((int(token), result, command_error, bundle, bundle_error))

    def _schedule_command_poll(self) -> None:
        if self._command_poll_after_id is None:
            self._command_poll_after_id = self.after(25, self._drain_command_results)

    def _drain_command_results(self) -> None:
        self._command_poll_after_id = None
        while True:
            try:
                payload = self._command_results.get_nowait()
            except queue.Empty:
                break
            self._finish_trading_command(*payload)
        if self._command_thread is not None and self._command_thread.is_alive():
            self._schedule_command_poll()

    def _finish_trading_command(self, token: int, result, command_error, bundle, bundle_error) -> None:
        context = self._command_context.pop(int(token), {})
        if int(token) != int(self._command_token):
            return
        self._command_thread = None
        self._set_command_busy(False)
        if isinstance(bundle, dict):
            self._apply_state_bundle(bundle)

        on_error = context.get("on_error")
        on_success = context.get("on_success")
        if command_error is not None:
            if callable(on_error):
                on_error(command_error)
            else:
                messagebox.showerror(str(context.get("error_title") or "Trading 操作失敗"), str(command_error), parent=self)
            self._schedule_pending_external_account_refresh()
            return
        if bundle_error is not None:
            messagebox.showwarning(
                "Trading 狀態刷新失敗",
                f"{context.get('label') or 'Trading command'} 已完成，但 canonical state 背景刷新失敗：{bundle_error}",
                parent=self,
            )
        if callable(on_success):
            on_success(result)
        domains = normalize_state_domains(context.get("state_domains"))
        if domains:
            callback = getattr(self.winfo_toplevel(), "_notify_workbench_state_changed", None)
            if callable(callback):
                callback(domains, source_panel_id="trading_account")
        self._schedule_pending_external_account_refresh()

    def _request_state_refresh(self, label: str = "Trading 狀態刷新") -> None:
        self._submit_trading_command(label, lambda: None, refresh_state=True)

    def refresh_for_state_domains(self, domains) -> bool:
        """Coalesce external mutations into one canonical Trading bundle reload."""
        normalized = normalize_state_domains(domains)
        if not normalized:
            return False
        if STATE_MARKET_DATA in normalized:
            self._ensure_stock_name_warm()
        initial_busy = bool(
            (self._initial_state_thread is not None and self._initial_state_thread.is_alive())
            or bool(getattr(self, "_initial_apply_bundle_active", False))
            or getattr(self, "_deferred_initial_bundle", None) is not None
            or (
                bool(getattr(self, "_initial_state_started", False))
                and not bool(getattr(self, "_initial_state_applied", False))
            )
        )
        command_busy = self._command_thread is not None and self._command_thread.is_alive()
        if initial_busy or command_busy:
            self._external_state_refresh_pending_domains.update(normalized)
            return False
        self._external_state_refresh_pending_domains.clear()
        return self._submit_trading_command(
            "同步狀態變更",
            lambda: None,
            refresh_state=True,
        )

    def refresh_external_account_state(self) -> bool:
        """Backward-compatible account refresh through the state-domain coordinator."""
        return self.refresh_for_state_domains(ACCOUNT_MUTATION_DOMAINS)

    def _schedule_pending_external_account_refresh(self) -> None:
        if not self._external_state_refresh_pending_domains:
            return
        domains = frozenset(self._external_state_refresh_pending_domains)
        self.after_idle(lambda domains=domains: self.refresh_for_state_domains(domains))

    def destroy(self):
        if self._pending_manual_preview_after_id is not None:
            try:
                self.after_cancel(self._pending_manual_preview_after_id)
            except tk.TclError as exc:
                _warn_gui_fallback("Trading pending manual preview after_cancel", exc)
            self._pending_manual_preview_after_id = None
        if self._trade_constraint_after_id is not None:
            try:
                self.after_cancel(self._trade_constraint_after_id)
            except tk.TclError as exc:
                _warn_gui_fallback("Trading direct-fill constraint after_cancel", exc)
            self._trade_constraint_after_id = None
        for attr_name, label in (
            ("_initial_state_start_after_id", "initial-state-start"),
            ("_initial_state_poll_after_id", "initial-state"),
            ("_initial_apply_after_id", "initial-render"),
            ("_command_poll_after_id", "command"),
            ("_pending_preview_poll_after_id", "pending-preview"),
            ("_trade_constraint_poll_after_id", "direct-fill-constraint"),
            ("_stock_name_warm_after_id", "stock-name-warm"),
        ):
            after_id = getattr(self, attr_name, None)
            if after_id is not None:
                try:
                    self.after_cancel(after_id)
                except tk.TclError as exc:
                    _warn_gui_fallback(f"Trading {label} poll after_cancel", exc)
                setattr(self, attr_name, None)
        super().destroy()
