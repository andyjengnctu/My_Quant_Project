from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import messagebox
from services.trading.scanner_state import load_trading_scanner_runtime
from services.trading.lifecycle_sync import SYNC_STATUS_FAILED, SYNC_STATUS_LATEST
from services.trading.order_form_constraints import build_trading_actual_fill_form_constraints
from services.trading.account_dashboard import build_trading_account_dashboard_read_model
from services.trading.account_state import TradingAccountRevisionConflict, adopt_existing_trading_position, correct_existing_trading_position, initialize_trading_account_state, remove_existing_trading_position, set_trading_cash_balance
from services.workbench_ui.state_sync import ACCOUNT_MUTATION_DOMAINS
from services.workbench_ui.stock_names import workbench_stock_name
from services.workbench_ui.trading_source_labels import trading_source_display_label
from services.workbench_ui.selection_behavior import handle_treeview_toggle_click
from services.trading.account_trade_entry import preview_trading_account_buy, record_trading_account_buy, resolve_trading_direct_buy_source
from services.workbench_ui.workbench import _warn_gui_fallback
from services.workbench_ui.trading_account_panel_data import (
    WORKBENCH_PROJECT_ROOT,
    MANUAL_SOURCE,
    UNMANAGED_STATUS,
    holding_order_date_by_lineage,
    parse_trading_money_text,
    parse_trading_qty_text,
    format_trading_money,
    build_trading_account_panel_snapshot,
    _load_panel_value,
)

class TradingAccountAccountMixin:
    """Trading Workbench TradingAccountAccount responsibilities; composed by TradingAccountPanel."""

    def _current_revision(self) -> int:
        revision = self._snapshot.get("revision")
        if revision is None:
            raise RuntimeError("Trading account 尚未初始化")
        return int(revision)

    def _submit_account_mutation(self, label: str, worker, *, success_message: str, clear_form: bool = False) -> None:
        def on_success(_result):
            if clear_form:
                self._clear_position_form()
            messagebox.showinfo("Trading 帳戶", success_message, parent=self)

        def on_error(exc):
            if isinstance(exc, TradingAccountRevisionConflict):
                messagebox.showerror(
                    "Trading 帳戶已更新",
                    f"{exc}\n\n已重新讀取最新狀態，請確認後再操作。",
                    parent=self,
                )
            else:
                messagebox.showerror("Trading 帳戶操作失敗", str(exc), parent=self)

        self._submit_trading_command(
            label,
            worker,
            on_success=on_success,
            on_error=on_error,
            refresh_state=True,
            state_domains=ACCOUNT_MUTATION_DOMAINS,
        )

    def refresh_account_dashboard(self):
        try:
            snapshot = _load_panel_value(
                self, "dashboard", lambda: build_trading_account_dashboard_read_model(WORKBENCH_PROJECT_ROOT)
            )
        except (FileNotFoundError, ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
            self._account_dashboard_snapshot = {}
            for variable in self._dashboard_metric_vars.values():
                variable.set("-")
            self._dashboard_detail_var.set(f"帳戶儀表板讀取失敗：{type(exc).__name__}: {exc}")
            self._reload_performance_rows([])
            self._reload_positions()
            return
        self._account_dashboard_snapshot = dict(snapshot or {})
        summary = dict(self._account_dashboard_snapshot.get("summary") or {})

        def money_text(value):
            return "-" if value is None else f"{float(value):,.0f}"

        self._dashboard_metric_vars["cash"].set(money_text(summary.get("cash")))
        self._dashboard_metric_vars["market_value"].set(money_text(summary.get("holdings_market_value")))
        self._dashboard_metric_vars["liquidation"].set(money_text(summary.get("holdings_net_liquidation")))
        self._dashboard_metric_vars["equity"].set(money_text(summary.get("equity")))
        warnings = list(self._account_dashboard_snapshot.get("warnings") or [])
        revision = self._account_dashboard_snapshot.get("source_account_revision")
        market_date = self._account_dashboard_snapshot.get("market_date") or "-"
        self._dashboard_detail_var.set(
            f"Trading 市價日 {market_date} | account revision {revision if revision is not None else '-'} | "
            f"持股 {int(summary.get('position_count') or 0)} | 已平倉 {int(summary.get('closed_trade_count') or 0)} | "
            f"唯讀衍生資料 outputs/trading/account/"
            + (f" | 注意 {len(warnings)} 項" if warnings else "")
        )
        self._reload_performance_rows(self._account_dashboard_snapshot.get("performance") or [])
        self._reload_positions()

    def _reload_performance_rows(self, rows):
        for item in self._performance_tree.get_children():
            self._performance_tree.delete(item)
        self._performance_rows = [dict(row) for row in list(rows or [])]
        self._fit_tree_rows(self._performance_tree, len(self._performance_rows))
        for row in self._performance_rows:
            return_pct = row.get("return_pct")
            profitable_rate = row.get("profitable_rate_pct")
            pnl = row.get("pnl")
            tag = "gain" if pnl is not None and float(pnl) > 0 else "loss" if pnl is not None and float(pnl) < 0 else "flat"
            self._performance_tree.insert(
                "",
                "end",
                tags=(tag,),
                values=(
                    row.get("scope") or "-",
                    f"{int(row.get('position_or_trade_count') or 0):,}",
                    self._format_candidate_number(row.get("cost_basis"), digits=0),
                    self._format_candidate_number(row.get("pnl"), digits=0),
                    "-" if return_pct is None else f"{float(return_pct):.2f}%",
                    f"{int(row.get('profitable_count') or 0):,}",
                    "-" if profitable_rate is None else f"{float(profitable_rate):.1f}%",
                ),
            )

    def refresh_account(self):
        try:
            snapshot = _load_panel_value(
                self, "account", lambda: build_trading_account_panel_snapshot(WORKBENCH_PROJECT_ROOT)
            )
        except (ValueError, RuntimeError, OSError) as exc:
            self._snapshot = {}
            self.refresh_operations_status()
            self._operations_next_var.set(f"FAIL | Trading account 狀態讀取失敗：{exc} | 下一步：全狀態刷新")
            return
        self._snapshot = snapshot
        initialized = bool(snapshot.get("initialized"))
        self._initialize_button.configure(state="disabled" if initialized else "normal")
        self._set_cash_button.configure(state="normal" if initialized else "disabled")
        self._add_button.configure(state="normal" if initialized else "disabled")
        if initialized and snapshot.get("cash") is not None:
            self._cash_var.set(format_trading_money(snapshot.get("cash")))
        elif not initialized:
            self._cash_var.set("")
        self._reload_positions()
        self._apply_order_lock_to_account_controls()
        self.refresh_operations_status()

    def _reload_positions(self):
        if not hasattr(self, "_tree"):
            return
        selected = self._selected_ticker()
        for item in self._tree.get_children():
            self._tree.delete(item)
        self._clear_table_sort_values(self._tree)
        self._position_rows = {}
        dashboard_positions = {
            str(row.get("ticker") or ""): dict(row)
            for row in list(self._account_dashboard_snapshot.get("positions") or [])
        }
        order_dates_by_lineage = holding_order_date_by_lineage(
            list(self._pending_snapshot.get("entries") or [])
        )
        position_status_by_ticker = {
            str(key): str(value)
            for key, value in dict(self._lifecycle_sync_snapshot.get("position_status_by_ticker") or {}).items()
        }
        position_errors = {
            str(key): str(value)
            for key, value in dict(self._lifecycle_sync_snapshot.get("position_errors") or {}).items()
        }
        for base_row in list(self._snapshot.get("positions") or []):
            ticker = str(base_row.get("ticker") or "")
            row = dict(base_row)
            row.update(dashboard_positions.get(ticker, {}))
            # Editing eligibility belongs to the canonical account read model.
            row["has_sell_history"] = bool(base_row.get("has_sell_history"))
            self._position_rows[ticker] = row
            sell_signal = "賣出訊號" if row.get("sell_signal") else "-"
            source_text = trading_source_display_label(source=row.get("source"))
            lineage_id = str(row.get("management_lineage_id") or row.get("strategy_lineage_id") or "").strip()
            entry_date = row.get("entry_date") or "-"
            # AI: Prefer the effective-account join, which survives BUY corrections.
            order_date = (row.get("order_date") or "-") if "order_date" in row else order_dates_by_lineage.get((lineage_id, str(entry_date)), "-")
            qty = int(row.get("qty") or 0)
            average_cost = row.get("average_cost")
            current_price = row.get("current_price")
            effective_stop = row.get("effective_stop")
            target_price = row.get("target_price")
            if ticker in position_errors:
                sync_status = SYNC_STATUS_FAILED
            else:
                sync_status = position_status_by_ticker.get(ticker, SYNC_STATUS_LATEST)
            self._tree.insert(
                "", "end", iid=ticker,
                values=(
                    "▣", source_text, ticker, workbench_stock_name(WORKBENCH_PROJECT_ROOT, ticker), order_date, entry_date, f"{qty:,}",
                    format_trading_money(average_cost),
                    format_trading_money(current_price),
                    format_trading_money(effective_stop),
                    format_trading_money(target_price),
                    sell_signal,
                    sync_status,
                ),
            )
            self._set_table_sort_values(
                self._tree,
                ticker,
                {
                    "source": source_text,
                    "ticker": ticker,
                    "stock_name": workbench_stock_name(WORKBENCH_PROJECT_ROOT, ticker),
                    "order_date": order_date,
                    "entry_date": entry_date,
                    "qty": qty,
                    "avg_cost": average_cost,
                    "current": current_price,
                    "stop": effective_stop,
                    "target": target_price,
                    "sell_signal": sell_signal,
                    "status": sync_status,
                },
            )
        self._apply_current_table_sort(self._tree)
        self._fit_tree_rows(self._tree, len(self._position_rows))
        if selected and selected in self._position_rows:
            self._tree.selection_set(selected)
            self._tree.focus(selected)
        else:
            self._set_position_action_state(None)

    def _on_position_tree_click(self, event):
        region = self._tree.identify_region(event.x, event.y)
        if region == "cell" and self._tree.identify_column(event.x) == "#1":
            item_id = str(self._tree.identify_row(event.y) or "")
            if not item_id:
                return "break"
            self._open_ticker_in_inspector(item_id)
            return "break"
        return handle_treeview_toggle_click(
            self._tree,
            event,
            ignore_columns=("#1",),
            on_clear=self._clear_position_form,
        )

    def _selected_ticker(self):
        selected = self._tree.selection()
        return str(selected[0]) if selected else None

    def _set_position_action_state(self, row):
        editable = bool(
            row
            and row.get("source") == MANUAL_SOURCE
            and row.get("management_status") == UNMANAGED_STATUS
            and not bool(row.get("has_sell_history"))
            and not self._has_active_orders()
        )
        state = "normal" if editable else "disabled"
        self._correct_button.configure(state=state)
        self._remove_button.configure(state=state)

    def _on_position_selected(self, _event=None):
        ticker = self._selected_ticker()
        row = self._position_rows.get(ticker or "")
        self._set_position_action_state(row)
        if not row:
            return
        self._ticker_var.set(str(row.get("ticker") or ""))
        self._qty_var.set(str(int(row.get("qty") or 0)))
        self._cost_var.set(format_trading_money(row.get("remaining_cost_basis")))
        self._entry_date_var.set(str(row.get("entry_date") or ""))
        self._note_var.set("")

    def _clear_position_form(self):
        for variable in (self._ticker_var, self._qty_var, self._cost_var, self._entry_date_var, self._note_var):
            variable.set("")
        self._tree.selection_remove(*self._tree.selection())
        self._set_position_action_state(None)

    def _initialize_account(self):
        try:
            cash = parse_trading_money_text(self._cash_var.get(), "帳戶現金", allow_blank=True, allow_zero=True)
        except ValueError as exc:
            messagebox.showerror("Trading 帳戶操作失敗", str(exc), parent=self)
            return
        self._submit_account_mutation(
            "初始化 Trading account",
            lambda: initialize_trading_account_state(WORKBENCH_PROJECT_ROOT, cash=cash),
            success_message="Trading account 已初始化。",
        )

    def _set_cash(self):
        try:
            cash = parse_trading_money_text(self._cash_var.get(), "帳戶現金", allow_blank=False, allow_zero=True)
            expected_revision = self._current_revision()
        except (ValueError, RuntimeError) as exc:
            messagebox.showerror("Trading 帳戶操作失敗", str(exc), parent=self)
            return
        self._submit_account_mutation(
            "更新 Trading 現金",
            lambda: set_trading_cash_balance(
                WORKBENCH_PROJECT_ROOT,
                cash=cash,
                expected_revision=None,
                note="Workbench cash reconciliation",
            ),
            success_message="現金餘額已更新。",
        )

    def _schedule_direct_buy_ticker_constraints(self, *_args):
        if hasattr(self, "_trade_source_var"):
            self._trade_source_var.set("-")
        return self._schedule_direct_fill_constraints(*_args)

    def _schedule_direct_fill_constraints(self, *_args):
        if self._trade_constraint_programmatic_update:
            return None
        self._trade_constraint_token += 1
        if self._trade_constraint_after_id is not None:
            try:
                self.after_cancel(self._trade_constraint_after_id)
            except tk.TclError as exc:
                _warn_gui_fallback("Trading direct-fill constraint debounce", exc)
        self._trade_constraint_after_id = self.after(220, self._start_direct_fill_constraints)
        return None

    @staticmethod
    def _direct_fill_constraint_worker(token: int, ticker: str, selected_date: str | None):
        source_resolution = None
        source_error = None
        try:
            source_resolution = resolve_trading_direct_buy_source(
                WORKBENCH_PROJECT_ROOT, ticker=ticker
            )
        except Exception as exc:
            source_error = exc

        result = None
        error = None
        try:
            runtime = load_trading_scanner_runtime(WORKBENCH_PROJECT_ROOT)
            latest = str(runtime.get("latest_data_date") or "").strip()
            result = build_trading_actual_fill_form_constraints(
                WORKBENCH_PROJECT_ROOT,
                ticker=ticker,
                selected_date=selected_date or None,
                latest_finalized_date=latest or None,
            )
        except Exception as exc:
            error = exc
        return token, result, error, source_resolution, source_error

    def _start_direct_fill_constraints(self):
        self._trade_constraint_after_id = None
        ticker = self._trade_ticker_var.get().strip().upper()
        if len(ticker) < 4:
            self._trade_fill_constraints = {}
            self._trade_price_option_set = frozenset()
            self._trade_price_combo.configure(values=())
            self._trade_date_field.set_allowed_dates(())
            self._trade_source_var.set("-")
            return
        selected_date = self._trade_date_var.get().strip() or None
        self._trade_constraint_token += 1
        token = int(self._trade_constraint_token)
        self._trade_constraint_inflight += 1

        def worker():
            self._trade_constraint_results.put(
                self._direct_fill_constraint_worker(token, ticker, selected_date)
            )

        threading.Thread(
            target=worker,
            name=f"workbench-direct-fill-constraint-{token}",
            daemon=True,
        ).start()
        if self._trade_constraint_poll_after_id is None:
            self._trade_constraint_poll_after_id = self.after(30, self._drain_direct_fill_constraints)

    def _drain_direct_fill_constraints(self):
        self._trade_constraint_poll_after_id = None
        while True:
            try:
                token, result, error, source_resolution, source_error = self._trade_constraint_results.get_nowait()
            except queue.Empty:
                break
            self._trade_constraint_inflight = max(0, int(self._trade_constraint_inflight) - 1)
            if int(token) != int(self._trade_constraint_token):
                continue
            if source_error is not None:
                self._trade_source_var.set("-")
            else:
                self._trade_source_var.set(
                    trading_source_display_label(source=(source_resolution or {}).get("source"))
                )
            if error is not None:
                self._trade_fill_constraints = {}
                self._trade_price_option_set = frozenset()
                self._trade_price_combo.configure(values=())
                self._trade_date_field.set_allowed_dates(())
                continue
            payload = dict(result or {})
            self._trade_fill_constraints = payload
            allowed = tuple(payload.get("allowed_dates") or ())
            prices = tuple(str(value) for value in (payload.get("price_options") or ()))
            self._trade_price_option_set = frozenset(prices)
            self._trade_date_field.set_allowed_dates(allowed)
            self._trade_price_combo.configure(values=prices)
            if not self._trade_date_var.get().strip() and payload.get("preferred_fill_date"):
                self._trade_constraint_programmatic_update = True
                try:
                    self._trade_date_var.set(str(payload.get("preferred_fill_date")))
                finally:
                    self._trade_constraint_programmatic_update = False
                self._schedule_direct_fill_constraints()
        if self._trade_constraint_inflight > 0:
            self._trade_constraint_poll_after_id = self.after(30, self._drain_direct_fill_constraints)

    def _simple_trade_values(self):
        ticker = self._trade_ticker_var.get().strip().upper()
        if not ticker:
            ticker = self._selected_candidate_ticker() or ""
        if not ticker:
            raise ValueError("股票代號必填")
        qty = parse_trading_qty_text(self._trade_qty_var.get(), "成交股數")
        price = parse_trading_money_text(self._trade_price_var.get(), "成交價", allow_zero=False)
        trade_date = self._trade_date_var.get().strip()
        if not trade_date:
            raise ValueError("成交日必填")
        constraints = dict(self._trade_fill_constraints or {})
        allowed_dates = set(constraints.get("allowed_dates") or ())
        if allowed_dates and trade_date not in allowed_dates:
            raise ValueError("成交日不是目前最新 finalized 範圍內、且有正式市場證據的可選日期")
        normalized_price = self._normalize_price_option_text(price)
        if self._trade_price_option_set and normalized_price not in self._trade_price_option_set:
            raise ValueError("成交價不在所選成交日的合法市場價格 ticks 內")
        return ticker, qty, price, trade_date

    def _record_simple_trade(self, side: str):
        try:
            ticker, qty, price, trade_date = self._simple_trade_values()
        except (ValueError, RuntimeError) as exc:
            messagebox.showerror("成交登錄", str(exc), parent=self)
            return
        if str(side).upper() != "BUY":
            messagebox.showerror("成交登錄", "賣出成交請到帳務中心點選庫存後操作。", parent=self)
            return

        candidate = self._selected_candidate_row()
        if candidate and str(candidate.get("ticker") or "").strip().upper() != ticker:
            candidate = None
        try:
            preview = preview_trading_account_buy(
                WORKBENCH_PROJECT_ROOT,
                ticker=ticker,
                qty=qty,
                price=price,
                trade_date=trade_date,
                candidate=candidate,
            )
        except (ValueError, RuntimeError, OSError, FileNotFoundError) as exc:
            messagebox.showerror("成交登錄失敗", str(exc), parent=self)
            return
        source_text = trading_source_display_label(source=preview.get("source"))
        evidence = dict(preview.get("market_evidence") or {})
        evidence_text = (
            f"市場證據：{evidence.get('trade_date') or trade_date} "
            f"Low {evidence.get('market_low')} / High {evidence.get('market_high')}"
        )
        warning_rows = [str(item) for item in list(preview.get("warnings") or []) if str(item).strip()]
        warning_text = "" if not warning_rows else "\n\n注意：\n- " + "\n- ".join(warning_rows)
        if not messagebox.askyesno(
            "確認買入成交",
            f"{ticker}｜{qty:,} 股 @ {price}｜{trade_date}\n\n來源：{source_text}\n{evidence_text}"
            f"{warning_text}\n\n價金、買入手續費與持有成本由帳務 SSOT 自動計算。",
            parent=self,
        ):
            return

        worker = lambda: record_trading_account_buy(
            WORKBENCH_PROJECT_ROOT,
            ticker=ticker,
            qty=qty,
            price=price,
            trade_date=trade_date,
            expected_account_revision=None,
            # AI: Confirm the provenance shown in the preview, not a transient
            # table selection. The service revalidates this reference under lock.
            candidate=preview.get("candidate"),
            expected_route=preview["route"],
        )

        def on_success(_result):
            self._trade_ticker_var.set("")
            self._trade_qty_var.set("")
            self._trade_price_var.set("")
            self._trade_date_var.set("")
            self._trade_source_var.set("-")
            self._trade_fill_constraints = {}
            self._trade_price_option_set = frozenset()
            self._trade_price_combo.configure(values=())
            self._trade_date_field.set_allowed_dates(())
            messagebox.showinfo(
                "成交登錄",
                f"{ticker} 買入成交已寫入帳戶 SSOT；持股區已同步重新整理。",
                parent=self.winfo_toplevel(),
            )

        self._submit_trading_command(
            f"{ticker} 買入成交登錄",
            worker,
            on_success=on_success,
            error_title="成交登錄失敗",
            # The buy mutates account truth used throughout Trading Center; refresh
            # the canonical bundle before reporting success.
            refresh_state=True,
            state_domains=ACCOUNT_MUTATION_DOMAINS,
        )

    def _position_form_values(self):
        ticker = self._ticker_var.get().strip().upper()
        if not ticker:
            raise ValueError("股票代號必填")
        return {
            "ticker": ticker,
            "qty": parse_trading_qty_text(self._qty_var.get()),
            "cost_basis_total": parse_trading_money_text(self._cost_var.get(), "剩餘成本總額", allow_zero=False),
            "entry_date": self._entry_date_var.get().strip() or None,
            "note": self._note_var.get().strip() or None,
        }

    def _adopt_position(self):
        try:
            values = self._position_form_values()
            expected_revision = self._current_revision()
        except (ValueError, RuntimeError) as exc:
            messagebox.showerror("Trading 帳戶操作失敗", str(exc), parent=self)
            return
        self._submit_account_mutation(
            f"登記既有持股 {values['ticker']}",
            lambda: adopt_existing_trading_position(
                WORKBENCH_PROJECT_ROOT,
                expected_revision=None,
                **values,
            ),
            success_message="既有持股已登記；現金未變更。",
            clear_form=True,
        )

    def _correct_position(self):
        selected = self._selected_ticker()
        if not selected:
            messagebox.showerror("Trading 帳戶操作失敗", "請先選取要修正的持股。", parent=self)
            return
        try:
            values = self._position_form_values()
            if values["ticker"] != selected:
                raise ValueError("修正持股時不可變更股票代號；如代號輸入錯誤，請移除後重新新增。")
            expected_revision = self._current_revision()
        except (ValueError, RuntimeError) as exc:
            messagebox.showerror("Trading 帳戶操作失敗", str(exc), parent=self)
            return
        self._submit_account_mutation(
            f"修正既有持股 {selected}",
            lambda: correct_existing_trading_position(
                WORKBENCH_PROJECT_ROOT,
                expected_revision=None,
                **values,
            ),
            success_message="既有持股 broker truth 已修正；現金未變更。",
        )

    def _remove_position(self):
        selected = self._selected_ticker()
        if not selected:
            messagebox.showerror("Trading 帳戶操作失敗", "請先選取要移除的持股。", parent=self)
            return
        row = self._position_rows.get(selected) or {}
        if not messagebox.askyesno(
            "移除既有持股",
            f"確定要移除 {selected}（{int(row.get('qty') or 0):,} 股）？\n\n這是 broker truth 修正，不會產生賣出交易，也不會改變 cash。",
            parent=self,
        ):
            return
        try:
            expected_revision = self._current_revision()
        except RuntimeError as exc:
            messagebox.showerror("Trading 帳戶操作失敗", str(exc), parent=self)
            return
        note = self._note_var.get().strip() or "Workbench manual position removal"
        self._submit_account_mutation(
            f"移除既有持股 {selected}",
            lambda: remove_existing_trading_position(
                WORKBENCH_PROJECT_ROOT,
                ticker=selected,
                expected_revision=None,
                note=note,
            ),
            success_message="既有持股已移除；現金未變更。",
            clear_form=True,
        )
