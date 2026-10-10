from __future__ import annotations

from tkinter import messagebox
from services.trading.protection_planning import PROTECTION_STOP_REMAINDER_ACTION, build_trading_protection_plan, get_trading_protection_plan_read_model
from services.trading.indicator_exit_planning import build_trading_indicator_exit_plan, get_trading_indicator_exit_plan_read_model
from services.trading.indicator_exit_order_submission import confirm_trading_indicator_exit_submission
from services.trading.protection_order_submission import confirm_trading_protection_leg_submission, confirm_trading_protection_oco_submission
from services.workbench_ui.state_sync import STATE_INDICATOR_EXIT, STATE_ORDERS, STATE_PROTECTION
from services.workbench_ui.stock_names import workbench_stock_name
from services.workbench_ui.trading_account_panel_data import (
    WORKBENCH_PROJECT_ROOT,
    _load_panel_value,
)

class TradingAccountProtectionMixin:
    """Trading Workbench TradingAccountProtection responsibilities; composed by TradingAccountPanel."""

    @staticmethod
    def _fit_tree_rows(tree, count):
        count = int(count or 0)
        if count <= 0:
            tree.grid_remove()
            return
        tree.grid()
        tree.configure(height=count)

    def _reload_protection_rows(self, rows):
        for item in self._protection_tree.get_children():
            self._protection_tree.delete(item)
        self._protection_rows = [dict(row) for row in list(rows or [])]
        self._fit_tree_rows(self._protection_tree, len(self._protection_rows))
        for row in self._protection_rows:
            self._protection_tree.insert(
                "",
                "end",
                iid=str(row.get("ticker") or ""),
                values=(
                    row.get("ticker") or "-",
                    workbench_stock_name(WORKBENCH_PROJECT_ROOT, row.get("ticker")),
                    f"{int(row.get('position_qty') or 0):,}",
                    self._format_candidate_number(row.get("entry_fill_price"), digits=2),
                    f"{int(row.get('stop_qty') or 0):,}",
                    ("MARKET" if bool(row.get("stop_forced_exit")) else self._format_candidate_number(row.get("stop_price"), digits=2)),
                    f"{int(row.get('tp_qty') or 0):,}",
                    self._format_candidate_number(row.get("target_price"), digits=2),
                    row.get("entry_order_status") or "-",
                    row.get("same_bar_priority") or "-",
                ),
            )

    def _selected_protection_row(self):
        selected = self._protection_tree.selection()
        if not selected:
            return None
        ticker = str(selected[0])
        for row in self._protection_rows:
            if str(row.get("ticker") or "") == ticker:
                return dict(row)
        return None

    def _confirm_protection_leg_submitted(self, action: str):
        row = self._selected_protection_row()
        if not row:
            messagebox.showerror("Trading 保護 SELL", "請先選取一筆成交後保護單計畫。", parent=self)
            return
        ticker = str(row.get("ticker") or "")
        label = (
            "Stop" if action == "STOP_FULL"
            else "Stop 剩餘 MARKET forced-exit" if action == PROTECTION_STOP_REMAINDER_ACTION
            else "TP"
        )
        if not messagebox.askyesno(
            "確認保護 SELL 已送券商",
            f"確認已在券商實際送出 {ticker} 的 {label} SELL？\n\n此動作只建立 ORDERED broker truth，不代表成交。若同時存在其他 SELL 委託，系統會依實際持股上限檢查。",
            parent=self,
        ):
            return
        expected_revision = int(self._current_order_revision())
        broker_order_id = self._protection_broker_id_var.get().strip() or None

        def worker():
            return confirm_trading_protection_leg_submission(
                WORKBENCH_PROJECT_ROOT,
                ticker=ticker,
                action=action,
                expected_order_revision=expected_revision,
                broker_order_id=broker_order_id,
                note="Workbench confirmed protection SELL submission",
            )

        def on_success(_result):
            self._protection_broker_id_var.set("")
            messagebox.showinfo("Trading 保護 SELL", f"{ticker} {label} 已記錄為 ORDERED；account 未修改。", parent=self)

        self._submit_trading_command(
            f"{ticker} {label} 送單確認",
            worker,
            on_success=on_success,
            error_title="Trading 保護 SELL 送單失敗",
            state_domains={STATE_ORDERS, STATE_PROTECTION},
        )

    def _confirm_protection_oco_submitted(self):
        row = self._selected_protection_row()
        if not row:
            messagebox.showerror("Trading 保護 OCO", "請先選取一筆成交後保護單計畫。", parent=self)
            return
        ticker = str(row.get("ticker") or "")
        group_id = self._protection_oco_group_var.get().strip()
        if not group_id:
            messagebox.showerror("Trading 保護 OCO", "必須輸入券商實際 OCO/互斥群組 ID；系統不會自行假設券商支援 OCO。", parent=self)
            return
        if not messagebox.askyesno(
            "確認券商 OCO 送單",
            f"確認券商已將 {ticker} Stop + TP 以 native OCO/互斥群組送出？\n\n群組 ID：{group_id}\n\n只有實際券商具備互斥/共享持股語意時才可確認。此動作不代表成交。",
            parent=self,
        ):
            return
        expected_revision = int(self._current_order_revision())
        stop_broker_order_id = self._protection_stop_broker_id_var.get().strip() or None
        tp_broker_order_id = self._protection_tp_broker_id_var.get().strip() or None

        def worker():
            return confirm_trading_protection_oco_submission(
                WORKBENCH_PROJECT_ROOT,
                ticker=ticker,
                expected_order_revision=expected_revision,
                broker_oco_group_id=group_id,
                stop_broker_order_id=stop_broker_order_id,
                tp_broker_order_id=tp_broker_order_id,
                note="Workbench confirmed broker-native OCO protection submission",
            )

        def on_success(_result):
            self._protection_oco_group_var.set("")
            self._protection_stop_broker_id_var.set("")
            self._protection_tp_broker_id_var.set("")
            messagebox.showinfo("Trading 保護 OCO", f"{ticker} Stop+TP 已依使用者確認記錄為券商 OCO ORDERED；account 未修改。", parent=self)

        self._submit_trading_command(
            f"{ticker} OCO 送單確認",
            worker,
            on_success=on_success,
            error_title="Trading 保護 OCO 送單失敗",
            state_domains={STATE_ORDERS, STATE_PROTECTION},
        )

    def refresh_protection_plan(self):
        try:
            snapshot = _load_panel_value(
                self, "protection", lambda: get_trading_protection_plan_read_model(WORKBENCH_PROJECT_ROOT)
            )
        except (ValueError, RuntimeError, OSError) as exc:
            self._protection_snapshot = {}
            self._reload_protection_rows([])
            self._protection_status_var.set(f"保護單計畫讀取失敗：{exc}")
            self._reload_positions()
            return
        self._protection_snapshot = snapshot
        self._reload_protection_rows(snapshot.get("positions") or [])
        self._reload_positions()
        if not snapshot.get("exists"):
            self._protection_status_var.set(
                "尚未建立保護單計畫"
            )
            self.refresh_operations_status()
            return
        freshness = "FRESH" if snapshot.get("fresh") else "STALE"
        skipped = list(snapshot.get("manual_positions_skipped") or [])
        suffix = f" | manual未接管 {','.join(skipped)}" if skipped else ""
        self._protection_status_var.set(
            f"{freshness} | {snapshot.get('status') or '-'} / {snapshot.get('broker_status') or '-'} | "
            f"positions {int(snapshot.get('position_count') or 0)}{suffix}"
        )
        self.refresh_operations_status()

    def _rebuild_protection_plan(self):
        def on_success(result):
            result = dict(result or {})
            self._protection_status_var.set(
                f"FRESH | {result.get('status')} / {result.get('broker_status')} | positions {len(result.get('positions') or [])}"
            )

        self._submit_trading_command(
            "建立／刷新保護單計畫",
            lambda: build_trading_protection_plan(WORKBENCH_PROJECT_ROOT),
            on_success=on_success,
            error_title="Trading 保護單計畫",
            state_domains={STATE_PROTECTION},
        )

    def _reload_indicator_rows(self, rows):
        for item in self._indicator_tree.get_children():
            self._indicator_tree.delete(item)
        self._indicator_rows=[dict(row) for row in list(rows or [])]
        self._fit_tree_rows(self._indicator_tree, len(self._indicator_rows))
        for row in self._indicator_rows:
            signal_key=str(row.get("signal_key") or "")
            iid=signal_key or f"{row.get('ticker')}:{row.get('signal_information_date')}"
            self._indicator_tree.insert("", "end", iid=iid, values=(row.get("ticker") or "-", workbench_stock_name(WORKBENCH_PROJECT_ROOT, row.get("ticker")), row.get("signal_information_date") or "-", f"{int(row.get('qty') or 0):,}", row.get("entry_trade_date") or "-", row.get("order_type") or "MARKET", "CARRIED" if row.get("carried_forward") else "NEW"))

    def _selected_indicator_row(self):
        selected=self._indicator_tree.selection()
        if not selected: return None
        key=str(selected[0])
        for row in self._indicator_rows:
            if str(row.get("signal_key") or "")==key: return dict(row)
        return None

    def refresh_indicator_exit_plan(self):
        try:
            snapshot=_load_panel_value(
                self, "indicator", lambda: get_trading_indicator_exit_plan_read_model(WORKBENCH_PROJECT_ROOT)
            )
        except (ValueError,RuntimeError,OSError) as exc:
            self._indicator_snapshot={}; self._reload_indicator_rows([]); self._indicator_status_var.set(f"Indicator SELL 計畫讀取失敗：{exc}"); self._reload_positions(); return
        self._indicator_snapshot=snapshot; self._reload_indicator_rows(snapshot.get("exits") or []); self._reload_positions()
        if not snapshot.get("exists"):
            self._indicator_status_var.set("尚未建立 Indicator SELL 計畫"); return
        self._indicator_status_var.set(f"{'FRESH' if snapshot.get('fresh') else 'STALE'} | exits {int(snapshot.get('exit_count') or 0)} | active broker Indicator SELL {int(snapshot.get('active_indicator_exit_order_count') or 0)}")

    def _rebuild_indicator_exit_plan(self):
        self._submit_trading_command(
            "建立／刷新 Indicator SELL 計畫",
            lambda: build_trading_indicator_exit_plan(WORKBENCH_PROJECT_ROOT),
            error_title="Trading Indicator SELL 計畫",
            state_domains={STATE_INDICATOR_EXIT},
        )

    def _confirm_indicator_exit_submitted(self):
        row = self._selected_indicator_row()
        if not row:
            messagebox.showerror("Trading Indicator SELL", "請先選取一筆 Indicator SELL 計畫。", parent=self)
            return
        ticker = str(row.get("ticker") or "")
        if not messagebox.askyesno(
            "確認 Indicator MARKET SELL 已送券商",
            f"確認已在券商實際送出 {ticker} 全倉 MARKET SELL？\n\n此動作只建立 ORDERED broker truth，不代表成交；若仍有 active Stop/TP 必須先在券商取消並於掛單表確認。",
            parent=self,
        ):
            return
        signal_key = str(row.get("signal_key") or "")
        expected_revision = int(self._current_order_revision())
        broker_order_id = self._indicator_broker_id_var.get().strip() or None

        def worker():
            return confirm_trading_indicator_exit_submission(
                WORKBENCH_PROJECT_ROOT,
                signal_key=signal_key,
                expected_order_revision=expected_revision,
                broker_order_id=broker_order_id,
                note="Workbench confirmed Indicator MARKET SELL submission",
            )

        def on_success(_result):
            self._indicator_broker_id_var.set("")
            messagebox.showinfo("Trading Indicator SELL", f"{ticker} Indicator MARKET SELL 已記錄為 ORDERED；account 未修改。", parent=self)

        self._submit_trading_command(
            f"{ticker} Indicator SELL 送單確認",
            worker,
            on_success=on_success,
            error_title="Trading Indicator SELL 送單失敗",
            state_domains={STATE_ORDERS, STATE_INDICATOR_EXIT},
        )
