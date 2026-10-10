from __future__ import annotations

from tkinter import messagebox
from core.trading_order_state import TRADING_ACTIVE_ORDER_STATUSES, TRADING_ORDER_PURPOSE_INDICATOR_EXIT
from services.trading.protection_planning import build_trading_protection_plan
from services.trading.indicator_exit_planning import build_trading_indicator_exit_plan
from services.trading.entry_order_submission import confirm_trading_order_submission
from services.trading.fill_reconciliation import TradingFillRevisionConflict, confirm_trading_buy_order_fill, confirm_trading_protection_sell_order_fill, confirm_trading_indicator_sell_order_fill
from services.trading.order_state import TradingOrderRevisionConflict, confirm_trading_order_cancellation, get_trading_order_read_model
from services.workbench_ui.state_sync import ACCOUNT_MUTATION_DOMAINS, STATE_INDICATOR_EXIT, STATE_ORDERS, STATE_PROTECTION
from services.workbench_ui.stock_names import workbench_stock_name
from services.workbench_ui.trading_account_panel_data import (
    WORKBENCH_PROJECT_ROOT,
    parse_trading_money_text,
    parse_trading_qty_text,
    _load_panel_value,
)

class TradingAccountOrdersMixin:
    """Trading Workbench TradingAccountOrders responsibilities; composed by TradingAccountPanel."""

    def _reload_proposed_order_rows(self, rows):
        for item in self._proposed_tree.get_children():
            self._proposed_tree.delete(item)
        self._proposed_order_rows = [dict(row) for row in list(rows or [])]
        self._fit_tree_rows(self._proposed_tree, len(self._proposed_order_rows))
        for row in self._proposed_order_rows:
            iid = f"{int(row.get('rank') or 0)}:{str(row.get('ticker') or '')}"
            self._proposed_tree.insert(
                "",
                "end",
                iid=iid,
                values=(
                    int(row.get("rank") or 0),
                    row.get("ticker") or "-",
                    workbench_stock_name(WORKBENCH_PROJECT_ROOT, row.get("ticker")),
                    {"buy": "新訊號", "extended": "延續", "extended_tbd": "延續(TBD)", "reentry": "再進場"}.get(
                        str(row.get("kind") or ""), str(row.get("kind") or "-")
                    ),
                    f"{int(row.get('ensemble_vote_count') or 0)}/{int(row.get('ensemble_member_count') or 0)} (門檻{int(row.get('ensemble_min_agree') or 0)})",
                    self._format_candidate_number(row.get("limit_price"), digits=2),
                    f"{int(row.get('qty') or 0):,}",
                    self._format_candidate_number(row.get("reserved_cost"), digits=0),
                    self._format_candidate_number(row.get("init_sl"), digits=2),
                    (
                        "停利線 " + self._format_candidate_number(row.get("target_price"), digits=2)
                    ),
                ),
            )

    def _selected_proposed_order(self):
        selected = self._proposed_tree.selection()
        if not selected:
            return None
        iid = str(selected[0])
        for row in self._proposed_order_rows:
            if f"{int(row.get('rank') or 0)}:{str(row.get('ticker') or '')}" == iid:
                return dict(row)
        return None

    def _current_order_revision(self):
        revision = self._order_snapshot.get("revision")
        return None if revision is None else int(revision)

    def _has_active_orders(self) -> bool:
        return int(self._order_snapshot.get("active_order_count") or 0) > 0

    def refresh_order_state(self):
        try:
            snapshot = _load_panel_value(
                self, "orders", lambda: get_trading_order_read_model(WORKBENCH_PROJECT_ROOT)
            )
        except (ValueError, RuntimeError, OSError) as exc:
            self._order_snapshot = {}
            self._order_rows = {}
            self._order_status_var.set(f"掛單狀態讀取失敗：{exc}")
            return
        self._order_snapshot = snapshot
        self._order_rows = {}
        for item in self._order_tree.get_children():
            self._order_tree.delete(item)
        for row in list(snapshot.get("orders") or []):
            order_id = str(row.get("order_id") or "")
            self._order_rows[order_id] = dict(row)
            self._order_tree.insert(
                "",
                "end",
                iid=order_id,
                values=(
                    row.get("ticker") or "-",
                    workbench_stock_name(WORKBENCH_PROJECT_ROOT, row.get("ticker")),
                    row.get("side") or "-",
                    row.get("purpose") or "-",
                    row.get("status") or "-",
                    row.get("order_type") or "-",
                    self._format_candidate_number(row.get("trigger_price") if row.get("trigger_price") is not None else row.get("limit_price"), digits=2),
                    f"{int(row.get('qty') or 0):,}",
                    f"{int(row.get('filled_qty') or 0):,}",
                    f"{int(row.get('remaining_qty') or 0):,}",
                    self._format_candidate_number(row.get("average_fill_price"), digits=2),
                    row.get("broker_order_id") or "-",
                    row.get("information_date") or "-",
                    row.get("ordered_at") or "-",
                    row.get("filled_at") or "-",
                    row.get("cancelled_at") or "-",
                ),
            )
        self._fit_tree_rows(self._order_tree, len(self._order_rows))
        active = int(snapshot.get("active_order_count") or 0)
        revision = snapshot.get("revision")
        self._order_status_var.set(
            f"{'ACTIVE' if active else 'CLEAR'} | revision {revision if revision is not None else '-'} | active {active} "
            f"(BUY {int(snapshot.get('active_entry_order_count') or 0)} / protection SELL {int(snapshot.get('active_protection_order_count') or 0)}) | "
            f"總紀錄 {int(snapshot.get('order_count') or 0)}"
        )
        self._apply_order_lock_to_account_controls()
        self.refresh_operations_status()

    def _apply_order_lock_to_account_controls(self):
        active = self._has_active_orders()
        initialized = bool(self._snapshot.get("initialized"))
        self._set_cash_button.configure(state="disabled" if active or not initialized else "normal")
        self._add_button.configure(state="disabled" if active or not initialized else "normal")
        selected = self._selected_ticker()
        self._set_position_action_state(self._position_rows.get(selected or ""))

    def _confirm_selected_ordered(self):
        row = self._selected_proposed_order()
        if not row:
            messagebox.showerror("Trading 掛單", "請先選取一筆建議掛單。", parent=self)
            return
        ticker = str(row.get("ticker") or "")
        qty = int(row.get("qty") or 0)
        limit_price = self._format_candidate_number(row.get("limit_price"), digits=2)
        if not messagebox.askyesno(
            "確認實際送單",
            f"確認已在券商送出 {ticker} 買單？\n\n股數：{qty:,}\n限價：{limit_price}\n\n此動作只建立 ORDERED broker truth，不會扣 cash、也不會建立持股。",
            parent=self,
        ):
            return
        rank = int(row.get("rank") or 0)
        expected_revision = self._current_order_revision()
        broker_order_id = self._broker_order_id_var.get().strip() or None
        note = self._order_note_var.get().strip() or None

        def worker():
            return confirm_trading_order_submission(
                WORKBENCH_PROJECT_ROOT,
                rank=rank,
                ticker=ticker,
                expected_revision=expected_revision,
                broker_order_id=broker_order_id,
                note=note,
            )

        def on_success(_result):
            self._broker_order_id_var.set("")
            self._order_note_var.set("")
            messagebox.showinfo("Trading 掛單", f"{ticker} 已記錄為 ORDERED；account cash／positions 未變更。", parent=self)

        def on_error(exc):
            title = "Trading 掛單已更新" if isinstance(exc, TradingOrderRevisionConflict) else "Trading 掛單失敗"
            suffix = "\n\n已重新讀取最新掛單狀態。" if isinstance(exc, TradingOrderRevisionConflict) else ""
            messagebox.showerror(title, f"{exc}{suffix}", parent=self)

        self._submit_trading_command(
            f"{ticker} BUY 送單確認",
            worker,
            on_success=on_success,
            on_error=on_error,
            state_domains={STATE_ORDERS},
        )

    def _confirm_selected_fill(self):
        selected = self._order_tree.selection()
        if not selected:
            messagebox.showerror("Trading 成交", "請先選取一筆 ORDERED / PARTIAL 掛單。", parent=self)
            return
        order_id = str(selected[0])
        row = dict(self._order_rows.get(order_id) or {})
        if str(row.get("status")) not in TRADING_ACTIVE_ORDER_STATUSES:
            messagebox.showerror("Trading 成交", "只有 ORDERED / PARTIAL 掛單可確認成交。", parent=self)
            return
        try:
            fill_qty = parse_trading_qty_text(self._fill_qty_var.get(), "本次成交股數")
            fill_price = parse_trading_money_text(self._fill_price_var.get(), "本次成交價", allow_zero=False)
            trade_date = self._fill_date_var.get().strip()
            if not trade_date:
                raise ValueError("成交日必填")
        except ValueError as exc:
            messagebox.showerror("Trading 成交", str(exc), parent=self)
            return
        remaining = int(row.get("remaining_qty") or 0)
        if fill_qty > remaining:
            messagebox.showerror("Trading 成交", f"本次成交股數不可超過未成交股數 {remaining:,}", parent=self)
            return
        ticker = str(row.get("ticker") or "")
        if not messagebox.askyesno(
            "確認券商成交",
            f"確認券商實際成交 {ticker} {row.get('side') or 'BUY'}？\n\n本次成交：{fill_qty:,} 股 @ {fill_price}\n成交日：{trade_date}\n\n只有實際券商成交才可確認；此動作會以 canonical exact accounting 更新 cash／持股。",
            parent=self,
        ):
            return
        if str(row.get("side") or "BUY") == "BUY":
            fill_fn = confirm_trading_buy_order_fill
        elif str(row.get("purpose") or "") == TRADING_ORDER_PURPOSE_INDICATOR_EXIT:
            fill_fn = confirm_trading_indicator_sell_order_fill
        else:
            fill_fn = confirm_trading_protection_sell_order_fill
        expected_order_revision = int(self._current_order_revision())
        expected_account_revision = int(self._current_revision())

        def worker():
            result = fill_fn(
                WORKBENCH_PROJECT_ROOT,
                order_id=order_id,
                fill_qty=fill_qty,
                fill_price=fill_price,
                trade_date=trade_date,
                expected_order_revision=expected_order_revision,
                expected_account_revision=expected_account_revision,
            )
            protection_error = None
            try:
                build_trading_protection_plan(WORKBENCH_PROJECT_ROOT)
            except (ValueError, RuntimeError, OSError, FileNotFoundError) as exc:
                protection_error = str(exc)
            indicator_error = None
            try:
                build_trading_indicator_exit_plan(WORKBENCH_PROJECT_ROOT)
            except (ValueError, RuntimeError, OSError, FileNotFoundError) as exc:
                indicator_error = str(exc)
            return {
                "fill_result": result,
                "protection_error": protection_error,
                "indicator_error": indicator_error,
            }

        def on_success(payload):
            payload = dict(payload or {})
            result = dict(payload.get("fill_result") or {})
            protection_error = payload.get("protection_error")
            indicator_error = payload.get("indicator_error")
            self._fill_qty_var.set("")
            self._fill_price_var.set("")
            self._fill_date_var.set("")
            message = (
                f"{ticker} 已更新為 {result.get('status')}；累計成交 {int(result.get('filled_qty') or 0):,}，"
                f"未成交 {int(result.get('remaining_qty') or 0):,}。"
            )
            if protection_error:
                message += f"\n\n成交已入帳，但保護單計畫建立失敗：{protection_error}"
            else:
                message += "\n\n已依實際成交後 canonical position state 機械刷新 Stop / TP 保護單計畫；尚未送券商。"
            if indicator_error:
                message += f"\n\nIndicator 計畫刷新失敗：{indicator_error}"
            messagebox.showinfo("Trading 成交", message, parent=self)

        def on_error(exc):
            title = "Trading 成交狀態已更新" if isinstance(exc, TradingFillRevisionConflict) else "Trading 成交失敗"
            suffix = "\n\n已重新讀取最新 account／order state。" if isinstance(exc, TradingFillRevisionConflict) else ""
            messagebox.showerror(title, f"{exc}{suffix}", parent=self)

        self._submit_trading_command(
            f"{ticker} 成交確認",
            worker,
            on_success=on_success,
            on_error=on_error,
            state_domains=set(ACCOUNT_MUTATION_DOMAINS) | {STATE_ORDERS, STATE_PROTECTION, STATE_INDICATOR_EXIT},
        )

    def _cancel_selected_order(self):
        selected = self._order_tree.selection()
        if not selected:
            messagebox.showerror("Trading 掛單", "請先選取一筆 ORDERED / PARTIAL 掛單。", parent=self)
            return
        order_id = str(selected[0])
        row = dict(self._order_rows.get(order_id) or {})
        if str(row.get("status")) not in TRADING_ACTIVE_ORDER_STATUSES:
            messagebox.showerror("Trading 掛單", "只有 ORDERED / PARTIAL 掛單可確認取消剩餘委託。", parent=self)
            return
        ticker = str(row.get("ticker") or "")
        if not messagebox.askyesno(
            "確認券商取消",
            f"確認券商端 {ticker} 未成交剩餘委託已取消？\n\n已成交部位不回滾；此動作只結束未成交餘額。",
            parent=self,
        ):
            return
        expected_revision = int(self._current_order_revision())
        note = self._order_note_var.get().strip() or "Workbench confirmed broker cancellation"
        filled_qty = int(row.get("filled_qty") or 0)

        def worker():
            result = confirm_trading_order_cancellation(
                WORKBENCH_PROJECT_ROOT,
                order_id=order_id,
                expected_revision=expected_revision,
                note=note,
            )
            protection_refresh_error = None
            if filled_qty > 0:
                try:
                    build_trading_protection_plan(WORKBENCH_PROJECT_ROOT)
                except (ValueError, RuntimeError, OSError, FileNotFoundError) as exc:
                    protection_refresh_error = str(exc)
            return {"cancel_result": result, "protection_error": protection_refresh_error}

        def on_success(payload):
            self._order_note_var.set("")
            protection_error = dict(payload or {}).get("protection_error")
            message = f"{ticker} 已記錄為 CANCELLED。"
            if protection_error:
                message += f"\n\n取消已記錄，但保護單計畫刷新失敗：{protection_error}"
            messagebox.showinfo("Trading 掛單", message, parent=self)

        def on_error(exc):
            title = "Trading 掛單已更新" if isinstance(exc, TradingOrderRevisionConflict) else "Trading 掛單失敗"
            suffix = "\n\n已重新讀取最新掛單狀態。" if isinstance(exc, TradingOrderRevisionConflict) else ""
            messagebox.showerror(title, f"{exc}{suffix}", parent=self)

        self._submit_trading_command(
            f"{ticker} 取消委託確認",
            worker,
            on_success=on_success,
            on_error=on_error,
            state_domains={STATE_ORDERS, STATE_PROTECTION},
        )

    def _finish_workflow_success(self, action: str, result):
        labels = {
            "params": ("Params 套用完成", "3 Scanner 候選"),
            "orders": ("建議掛單完成", "檢查掛單計畫"),
            "scanner": ("Scanner 完成", "查看 Scanner Pool；自行至券商交易，成交後回 Workbench 登錄"),
        }
        if str(action) not in labels:
            raise ValueError(f"不支援的 Trading workflow completion: {action!r}")
        done_text, next_text = labels[str(action)]
        self._operations_next_var.set(f"DONE | {done_text} | 下一步：{next_text}")
