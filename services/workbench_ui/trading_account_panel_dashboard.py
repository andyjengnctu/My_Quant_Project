from __future__ import annotations

import tkinter as tk
from tkinter import messagebox
from core.trading_policy import get_trading_policy_snapshot
from core.strategy_promotion import load_trading_release_state
from services.trading.strategy_param_state import TRADING_PARAM_USAGE_REUSE_EXISTING, TRADING_PARAM_USAGE_TRAINED_CURRENT
from services.trading.operations_status import build_trading_operations_status
from services.trading.operational_audit import run_trading_operational_audit
from services.workbench_ui.workbench import WORKBENCH_ERROR, WORKBENCH_INFO, WORKBENCH_MUTED, WORKBENCH_SUCCESS, WORKBENCH_TEXT, WORKBENCH_WARNING, _warn_gui_fallback
from services.workbench_ui.trading_account_panel_data import (
    WORKBENCH_PROJECT_ROOT,
    _load_panel_value,
)

class TradingAccountDashboardMixin:
    """Trading Workbench TradingAccountDashboard responsibilities; composed by TradingAccountPanel."""

    def _set_overview_card(
        self, key: str, primary: object, detail: object = "", *,
        tone: str = "text", detail_tone: str = "muted",
    ) -> None:
        self._overview_vars[key].set(str(primary if primary not in (None, "") else "-"))
        self._overview_detail_vars[key].set(str(detail or ""))
        colors = {
            "text": WORKBENCH_TEXT,
            "muted": WORKBENCH_MUTED,
            "info": WORKBENCH_INFO,
            "success": WORKBENCH_SUCCESS,
            "warning": WORKBENCH_WARNING,
            "error": WORKBENCH_ERROR,
        }
        self._overview_primary_labels[key].configure(foreground=colors.get(str(tone), WORKBENCH_TEXT))
        self._overview_detail_labels[key].configure(foreground=colors.get(str(detail_tone), WORKBENCH_MUTED))

    def _render_strategy_params_overview(self, snapshot: dict[str, object], *, release_state: dict | None = None) -> None:
        """Show canonical Trading Release and bound Params in one dashboard card."""
        policy = get_trading_policy_snapshot(WORKBENCH_PROJECT_ROOT)
        dl_state = "OFF" if not policy.get("dl_filter_enabled") and not policy.get("dl_ranking_enabled") else "ON"
        training_date = snapshot.get("param_training_data_date") or "尚無 Params"
        usage_mode = {
            TRADING_PARAM_USAGE_REUSE_EXISTING: "沿用既有",
            TRADING_PARAM_USAGE_TRAINED_CURRENT: "重新訓練",
        }.get(snapshot.get("param_usage_mode"), "未綁定")
        detail = (
            f"{snapshot.get('param_selector') or '-'} | {usage_mode} | "
            f"訓練至 {training_date} | DL {dl_state}"
        )
        tone = "success" if snapshot.get("params_ready_for_scan") else (
            "info" if snapshot.get("params_reusable") else "warning"
        )
        try:
            # ACTIVE identity comes from the Release SSOT, not the engine strategy_id.
            state = release_state if release_state is not None else load_trading_release_state(WORKBENCH_PROJECT_ROOT)
            if not isinstance(state, dict):
                raise ValueError("Trading Release 狀態格式不合法")
            active = state.get("active")
            pending = state.get("pending")
            if active is not None and (
                not isinstance(active, dict)
                or active.get("status") != "ACTIVE"
                or not str(active.get("arm_id") or "").strip()
            ):
                raise ValueError("ACTIVE Release 記錄不合法")
            if pending is not None and (
                not isinstance(pending, dict)
                or not str(pending.get("arm_id") or "").strip()
            ):
                raise ValueError("待啟用 Release 記錄不合法")
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            self._set_overview_card(
                "strategy_params", "Release 狀態異常",
                f"{type(exc).__name__}: {exc}", tone="error", detail_tone="error",
            )
            return

        primary = f"ACTIVE {active['arm_id']}" if active else "既有 Trading 策略"
        if pending:
            detail = f"{detail} | 待啟用 {pending['arm_id']}"
        self._set_overview_card("strategy_params", primary, detail, tone=tone)

    def refresh_operations_status(self):
        if bool(getattr(self, "_suspend_operations_refresh", False)):
            return
        try:
            snapshot = _load_panel_value(
                self, "operations", lambda: build_trading_operations_status(WORKBENCH_PROJECT_ROOT)
            )
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            self._operations_snapshot = {}
            for key in self._overview_vars:
                self._set_overview_card(
                    key, "-", "狀態讀取失敗", tone="muted", detail_tone="error"
                )
            self._operations_next_var.set(f"FAIL | Trading 狀態讀取失敗：{exc} | 下一步：修正 Trading 整體狀態")
            self._set_workflow_buttons_state("disabled")
            return
        self._operations_snapshot = snapshot

        overall = str(snapshot.get("overall_status") or "-")
        latest_data_date = str(snapshot.get("latest_data_date") or "-")
        data_ready = bool(snapshot.get("trading_data_ready"))

        def count_text(value: object) -> str:
            return "-" if value is None else f"{int(value):,} 檔"

        def amount_text(value: object) -> str:
            return "-" if value is None else f"{float(value):,.0f}"

        listed_count = snapshot.get("listed_ticker_count")
        quick_count = snapshot.get("quick_filter_qualified_count")
        candidate_count = snapshot.get("scanner_candidate_ticker_count")
        scanner_fresh = bool(snapshot.get("candidate_snapshot_fresh"))
        scanner_date = str(snapshot.get("scanner_information_date") or "-")
        latest_date_tone = "text" if latest_data_date != "-" else "muted"
        candidate_date_tone = (
            "text"
            if scanner_fresh and scanner_date != "-" and scanner_date == latest_data_date
            else ("warning" if scanner_date != "-" else "muted")
        )
        self._set_overview_card(
            "total",
            count_text(listed_count),
            latest_data_date,
            tone="success" if listed_count is not None else "muted",
            detail_tone=latest_date_tone,
        )
        self._set_overview_card(
            "quick",
            count_text(quick_count),
            latest_data_date,
            tone="success" if quick_count is not None else "muted",
            detail_tone=latest_date_tone,
        )
        self._set_overview_card(
            "candidates",
            count_text(candidate_count),
            scanner_date,
            tone="success" if scanner_fresh else ("warning" if candidate_count is not None else "muted"),
            detail_tone=candidate_date_tone,
        )

        free_slots = snapshot.get("portfolio_free_slot_count")
        held_count = snapshot.get("portfolio_open_position_count")
        max_positions = snapshot.get("portfolio_max_positions")
        locked_slots = snapshot.get("pending_locked_slot_count")
        if None not in (free_slots, held_count, max_positions, locked_slots):
            slot_detail = (
                f"掛單: {int(locked_slots):,} | "
                f"持有: {int(held_count):,} | 上限: {int(max_positions):,}"
            )
        else:
            slot_detail = "掛單: - | 持有: - | 上限: -"
        self._set_overview_card(
            "slots",
            "-" if free_slots is None else f"{int(free_slots):,}",
            slot_detail,
            tone="success" if free_slots is not None else "muted",
            detail_tone="text",
        )

        available_cash = snapshot.get("available_cash_after_pending")
        reserved_milli = snapshot.get("pending_reserved_total_milli")
        cash = snapshot.get("cash")
        reserved_cash = None if reserved_milli is None else float(reserved_milli) / 1000.0
        fund_detail = f"掛單: {amount_text(reserved_cash)} | 現金餘額: {amount_text(cash)}"
        self._set_overview_card(
            "funds",
            amount_text(available_cash),
            fund_detail,
            tone="success" if available_cash is not None else "muted",
            detail_tone="text",
        )

        self._render_strategy_params_overview(snapshot)

        rollforward_due = list(snapshot.get("rollforward_due_tickers") or [])
        if not data_ready:
            next_text = "至資料中心確認 Trading 資料"
        elif rollforward_due:
            next_text = "等待 lifecycle 自動同步／修正同步失敗"
        elif not snapshot.get("params_ready_for_scan"):
            next_text = "2 訓練／套用 Params"
        elif not scanner_fresh:
            next_text = "3 Scanner 候選"
        else:
            next_text = "查看 Scanner Pool；自行至券商交易，成交後回 Workbench 登錄"
        display_status = (
            "READY"
            if data_ready and not rollforward_due and bool(snapshot.get("params_ready_for_scan")) and scanner_fresh
            else "WAIT"
        )
        self._operations_next_var.set(f"{display_status} | 下一步：{next_text}")
        self._apply_workflow_action_availability()

    def _run_operational_audit(self):
        def on_success(audit):
            blockers = list((audit or {}).get("blockers") or [])
            warnings = list((audit or {}).get("warnings") or [])
            if blockers:
                summary = "；".join(blockers[:2])
                self._operations_next_var.set("LIVE_BLOCKED | 實盤就緒檢查未通過 | 下一步：修正阻擋項目")
                messagebox.showwarning("Trading 尚不可實盤", summary, parent=self)
            else:
                suffix = f"｜警告 {len(warnings)} 項" if warnings else ""
                self._operations_next_var.set(f"DONE | 實盤就緒檢查 {audit.get('status')}{suffix} | 下一步：依目前 Trading 狀態操作")
                messagebox.showinfo("Trading 實盤就緒檢查", f"{audit.get('status')}{suffix}", parent=self)

        def on_error(exc):
            self._operations_next_var.set(f"FAIL | 實盤就緒檢查失敗：{exc} | 下一步：修正 Audit 錯誤")
            messagebox.showerror("Trading 實盤就緒檢查失敗", str(exc), parent=self)

        self._submit_trading_command(
            "實盤就緒檢查",
            lambda: run_trading_operational_audit(WORKBENCH_PROJECT_ROOT),
            on_success=on_success,
            on_error=on_error,
            refresh_state=True,
        )

    def _refresh_all_trading_state(self):
        # Refresh decision/accounting read models only; broker OMS recovery is no
        # longer part of the user-facing Trading Center workflow.
        self._submit_trading_command(
            "全狀態刷新",
            lambda: {"status": "REFRESH"},
            error_title="Trading 狀態刷新失敗",
            refresh_state=True,
        )

    def _apply_workflow_action_availability(self):
        if self._command_thread is not None and self._command_thread.is_alive():
            self._set_workflow_buttons_state("disabled")
            return
        for _action, button in self._workflow_action_buttons.items():
            button.configure(state="normal")
        # Legacy broker-order controls are hidden from the simplified UI.
        self._confirm_ordered_button.configure(state="disabled")

    @staticmethod
    def _set_wraplength(labels, width: int) -> None:
        wraplength = max(320, int(width) - 32)
        for label in labels:
            label.configure(wraplength=wraplength)

    def _show_footer_hint(self, text: str) -> None:
        if self._footer_hint_after_id is not None:
            try:
                self.after_cancel(self._footer_hint_after_id)
            except tk.TclError as exc:
                _warn_gui_fallback("Trading footer hint after_cancel", exc)
            self._footer_hint_after_id = None
        self._footer_hint_var.set(str(text or ""))

    def _schedule_footer_hint_clear(self) -> None:
        if self._footer_hint_after_id is not None:
            try:
                self.after_cancel(self._footer_hint_after_id)
            except tk.TclError as exc:
                _warn_gui_fallback("Trading footer hint after_cancel", exc)
        self._footer_hint_after_id = self.after(80, self._clear_footer_hint)

    def _clear_footer_hint(self) -> None:
        self._footer_hint_after_id = None
        self._footer_hint_var.set("")

    def _bind_footer_hint(self, widget, text: str) -> None:
        targets = [widget]
        try:
            targets.extend(widget.winfo_children())
        except tk.TclError as exc:
            _warn_gui_fallback("Trading footer hint winfo_children", exc)
        index = 0
        while index < len(targets):
            target = targets[index]
            index += 1
            try:
                for child in target.winfo_children():
                    if child not in targets:
                        targets.append(child)
                target.bind("<Enter>", lambda _event, value=text: self._show_footer_hint(value), add="+")
                target.bind("<Leave>", lambda _event: self._schedule_footer_hint_clear(), add="+")
            except tk.TclError as exc:
                _warn_gui_fallback("Trading footer hint bind", exc)
