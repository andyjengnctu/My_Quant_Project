from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from core.strategy_promotion import current_strategy_rules, get_approved_rule, load_promotion_approvals, load_trading_release_state
from services.trading.strategy_promotion import stage_trading_release, activate_staged_trading_release
from services.workbench_ui.state_sync import STATE_PARAMS, STATE_SCANNER_ELIGIBILITY
from services.workbench_ui.workbench import WORKBENCH_BG, WORKBENCH_BUTTON_STYLE, WORKBENCH_FRAME_STYLE, WORKBENCH_LABEL_STYLE, WORKBENCH_TREE_STYLE
from services.workbench_ui.trading_account_panel_data import (
    WORKBENCH_PROJECT_ROOT,
    PARAM_MODE_TRAIN_LABEL,
    PARAM_MODE_BY_LABEL,
)

class TradingAccountPromotionMixin:
    """Trading Workbench TradingAccountPromotion responsibilities; composed by TradingAccountPanel."""

    def _open_strategy_promotion_dialog(self):
        """Same Compare Suite as Research [3]/[8], never a Trading allowlist."""
        if self._command_thread is not None and self._command_thread.is_alive():
            messagebox.showwarning("策略升格", "Trading 工作進行中，請稍後再操作策略升格。", parent=self)
            return
        win = tk.Toplevel(self)
        win.title("Trading 策略選擇／切換")
        win.transient(self.winfo_toplevel())
        win.geometry("850x420")
        win.configure(background=WORKBENCH_BG)
        outer = ttk.Frame(win, padding=12, style=WORKBENCH_FRAME_STYLE)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="與 Research [3] 策略組合比較使用同一份策略清單", style=WORKBENCH_LABEL_STYLE).pack(anchor="w")
        cols = ("name", "approval", "capability", "rule")
        # The default native Treeview has white rows; Workbench's inherited
        # foreground is white.  Use the existing dark Treeview contract.
        tree = ttk.Treeview(
            outer, columns=cols, show="headings", height=7,
            style=WORKBENCH_TREE_STYLE, selectmode="browse",
        )
        for col, title, width in (("name", "策略", 390), ("approval", "Research 核准", 100),
                                  ("capability", "Trading", 150), ("rule", "Rule Fingerprint", 150)):
            tree.heading(col, text=title)
            tree.column(col, width=width, stretch=col == "name")
        tree.pack(fill="both", expand=True, pady=(12, 8))
        info = tk.StringVar(value="選擇一個策略；未核准／未支援的策略不可準備")
        ttk.Label(outer, textvariable=info, style=WORKBENCH_LABEL_STYLE, wraplength=800).pack(anchor="w")
        status_var = tk.StringVar(value="")
        ttk.Label(outer, textvariable=status_var, style=WORKBENCH_LABEL_STYLE).pack(anchor="w", pady=(6, 0))
        controls = ttk.Frame(outer, style=WORKBENCH_FRAME_STYLE)
        controls.pack(fill="x", pady=(10, 0))
        selected_ids = {}

        def refresh():
            # Read canonical approval/release state on every refresh.  Research
            # can approve in a separate process while this dialog stays open.
            try:
                rules = current_strategy_rules()
                state = load_trading_release_state(WORKBENCH_PROJECT_ROOT)
                approval_records = load_promotion_approvals(WORKBENCH_PROJECT_ROOT).get("approvals") or {}
                rows = []
                for arm in rules:
                    try:
                        get_approved_rule(WORKBENCH_PROJECT_ROOT, arm["arm_id"])
                        approved, issue = True, ""
                    except RuntimeError as exc:
                        approved, issue = False, str(exc)
                    rows.append((arm, approved, issue))
            except (OSError, RuntimeError, ValueError) as exc:
                messagebox.showerror("策略升格", f"無法讀取策略升格狀態：{exc}", parent=win)
                return
            old_selection = tree.selection()
            tree.delete(*tree.get_children())
            selected_ids.clear()
            for arm, approved, issue in rows:
                arm_id = arm["arm_id"]
                has_approval_record = arm_id in approval_records
                selected_ids[arm_id] = (arm, approved, issue, has_approval_record)
                if approved:
                    approval_status = "已核准"
                elif has_approval_record:
                    approval_status = "核准失效"
                else:
                    approval_status = "待核准"
                tree.insert("", "end", iid=arm_id, values=(
                    f"{arm_id} — {arm['name']}", approval_status,
                    "支援" if arm["trading_supported"] else "暫不支援",
                    arm["rule_sha256"][:12],
                ))
            active = state.get("active") or {}
            pending = state.get("pending") or {}
            self._render_strategy_params_overview(self._operations_snapshot, release_state=state)
            status_var.set(
                f"ACTIVE：{active.get('arm_id') or '既有 Trading 策略'}  |  "
                f"待啟用：{pending.get('arm_id') or '無'}  |  切換前持股與掛單不會自動換參數"
            )
            if old_selection and old_selection[0] in selected_ids:
                tree.selection_set(old_selection[0])
            elif rules:
                # Prefer a ready-to-stage entry over an unsupported first row.
                preferred = next(
                    (arm["arm_id"] for arm, approved, _ in rows if approved and arm["trading_supported"]),
                    None,
                ) or next((arm["arm_id"] for arm in rules if arm["trading_supported"]), rules[0]["arm_id"])
                tree.selection_set(preferred)
            on_selected()

        def on_selected(_event=None):
            selection = tree.selection()
            if not selection or selection[0] not in selected_ids:
                stage_button.configure(state="disabled")
                return
            arm, approved, issue, has_approval_record = selected_ids[selection[0]]
            blockers = "；".join(arm["trading_blockers"]) if arm["trading_blockers"] else "已由共用 Trading producer 支援"
            approval_hint = "Research 已核准" if approved else (
                issue if has_approval_record else "請先到 Research [8] 核准"
            )
            info.set(f"{arm['arm_id']} | {approval_hint} | {blockers}")
            stage_button.configure(state="normal" if approved and arm["trading_supported"] else "disabled")

        def stage():
            selection = tree.selection()
            if not selection:
                return
            arm_id = selection[0]
            if not messagebox.askyesno(
                "準備 Trading 策略",
                f"準備 {arm_id} 的獨立 Trading Params？\n目前 ACTIVE 不會切換；完成重訓後必須再次明確啟用。",
                parent=win,
            ):
                return
            try:
                stage_trading_release(WORKBENCH_PROJECT_ROOT, arm_id)
                self._param_mode_var.set(PARAM_MODE_TRAIN_LABEL)
                self._param_mode_user_selected = True
            except (OSError, RuntimeError, ValueError) as exc:
                messagebox.showerror("準備失敗", str(exc), parent=win)
                return
            refresh()
            messagebox.showinfo("策略已準備", "待啟用策略已建立。請執行「2 訓練／套用 Params」（選擇重新訓練）；完成後回到「1 策略選擇」啟用。", parent=win)

        def activate():
            try:
                pending = load_trading_release_state(WORKBENCH_PROJECT_ROOT).get("pending")
            except (OSError, RuntimeError, ValueError) as exc:
                messagebox.showerror("策略升格", str(exc), parent=win)
                return
            if not pending:
                messagebox.showwarning("策略升格", "尚無待啟用策略。", parent=win)
                return
            if not messagebox.askyesno("啟用 Trading Release", f"確認啟用 {pending.get('arm_id')}？需要獨立 Params 與有效 binding，既有持股維持 frozen lineage。", parent=win):
                return

            def success(result):
                messagebox.showinfo("策略已啟用", f"{result['arm_id']} 已啟用，請重新執行 Scanner；舊候選快照不得沿用。", parent=self)
                if win.winfo_exists():
                    refresh()
                self._request_state_refresh("Trading 策略升格")

            def failure(exc):
                messagebox.showerror("策略升格遭阻擋", str(exc), parent=self)

            self._submit_trading_command(
                "啟用 Trading Release",
                lambda: activate_staged_trading_release(WORKBENCH_PROJECT_ROOT),
                on_success=success, on_error=failure, refresh_state=True,
                state_domains={STATE_PARAMS, STATE_SCANNER_ELIGIBILITY},
            )

        stage_button = ttk.Button(controls, text="準備選取策略", command=stage, style=WORKBENCH_BUTTON_STYLE)
        stage_button.pack(side="left")
        ttk.Button(controls, text="啟用已訓練策略", command=activate, style=WORKBENCH_BUTTON_STYLE).pack(side="left", padx=(10, 0))
        ttk.Button(controls, text="重新整理核准狀態", command=refresh, style=WORKBENCH_BUTTON_STYLE).pack(side="left", padx=(10, 0))
        ttk.Button(controls, text="關閉", command=win.destroy, style=WORKBENCH_BUTTON_STYLE).pack(side="right")
        tree.bind("<<TreeviewSelect>>", on_selected)
        # Returning to Workbench after approving via Research [8] must re-read
        # the shared approval artifact, without a polling loop or auto-activation.
        win.bind("<FocusIn>", lambda event: refresh() if event.widget is win else None, add="+")
        refresh()

    def _on_param_mode_selected(self, _event=None) -> None:
        self._param_mode_user_selected = True

    def _selected_param_mode(self) -> str:
        label = str(self._param_mode_var.get() or "").strip()
        try:
            return PARAM_MODE_BY_LABEL[label]
        except KeyError as exc:
            raise ValueError(f"不支援的 Params 模式：{label or '-'}") from exc

    def _set_workflow_buttons_state(self, state: str):
        for button in self._workflow_buttons:
            button.configure(state=state)
