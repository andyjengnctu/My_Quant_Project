from __future__ import annotations

from pathlib import Path
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from config.market_data import MARKET_DATA_V2_PUBLICATION_POLICY
from core.console_report import format_datetime_in_timezone
from services.downloader.daily_console_progress import MarketDataDailyConsoleProgress
from services.trading.market_data_auto_update import run_trading_market_data_auto_update
from services.trading.market_data_ops import build_market_data_ops_read_model
from services.trading.market_data_scheduler import (
    install_or_update_market_data_scheduler,
    remove_market_data_scheduler,
    set_market_data_scheduler_enabled,
)
from services.trading.market_data_update import run_trading_market_data_update
from services.workbench_ui.workbench import (
    WORKBENCH_BG,
    WORKBENCH_BUTTON_STYLE,
    WORKBENCH_ERROR,
    WORKBENCH_ERROR_LABEL_STYLE,
    WORKBENCH_FRAME_STYLE,
    WORKBENCH_HSCROLL_STYLE,
    WORKBENCH_INFO,
    WORKBENCH_INFO_LABEL_STYLE,
    WORKBENCH_LABEL_STYLE,
    WORKBENCH_LABELLF_STYLE,
    WORKBENCH_MUTED,
    WORKBENCH_MUTED_LABEL_STYLE,
    WORKBENCH_NOTEBOOK_STYLE,
    WORKBENCH_SUCCESS,
    WORKBENCH_SUCCESS_LABEL_STYLE,
    WORKBENCH_TREE_STYLE,
    WORKBENCH_UI_FONT,
    WORKBENCH_VSCROLL_STYLE,
    WORKBENCH_WARNING,
    WORKBENCH_WARNING_LABEL_STYLE,
    WORKBENCH_TEXT,
)

WORKBENCH_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _fmt_datetime(value) -> str:
    return format_datetime_in_timezone(
        value,
        timezone_name=str(MARKET_DATA_V2_PUBLICATION_POLICY.get("timezone") or "Asia/Taipei"),
        output_format="%m-%d %H:%M",
    )


def _fmt_date(value) -> str:
    text = str(value or "").strip()
    return text or "-"


def _retry_text(row: dict[str, object]) -> str:
    return "/".join(
        str(int(row.get(key) or 0))
        for key in ("publication_retry_count", "quota_defer_count", "error_retry_count")
    )


def _status_tag(value: object) -> str:
    status = str(value or "").strip().upper()
    if status in {"READY", "SUCCESS", "SYNCED", "UPDATED", "CURRENT", "INSTALLED_ENABLED", "DONE"}:
        return "success"
    if status in {"DUE", "WAIT_PUBLISH", "WAIT_QUOTA", "DEFERRED", "PROBE", "NOT_INSTALLED", "INSTALLED_DISABLED"}:
        return "warning"
    if status in {"BLOCKED", "ERROR", "FAIL", "STALE", "UNAVAILABLE"}:
        return "error"
    if status in {"NO_DUE", "NO_NEW_DATE", "NOT_APPLICABLE", "NO_ROW_VALID"}:
        return "muted"
    return "info"


def _color_for_tag(tag: str) -> str:
    return {
        "success": WORKBENCH_SUCCESS,
        "warning": WORKBENCH_WARNING,
        "error": WORKBENCH_ERROR,
        "muted": WORKBENCH_MUTED,
        "info": WORKBENCH_INFO,
    }.get(str(tag), WORKBENCH_TEXT)


def _attention_tags(value: object) -> tuple[str, ...]:
    """Color only rows that require attention; routine healthy rows stay neutral."""

    status = str(value or "").strip().upper()
    if status in {"BLOCKED", "ERROR", "FAIL", "UNAVAILABLE"}:
        return ("error",)
    if status in {"DUE", "WAIT_QUOTA", "DEFERRED", "NOT_INSTALLED", "INSTALLED_DISABLED", "STALE"}:
        return ("warning",)
    return ()


class MarketDataOpsPanel(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=6, style=WORKBENCH_FRAME_STYLE)
        self._action_thread: threading.Thread | None = None
        self._snapshot: dict[str, object] = {}
        self._dataset_by_iid: dict[str, dict[str, object]] = {}
        self._status_var = tk.StringVar(value="讀取 Market Data 狀態…")
        self._detail_var = tk.StringVar(value="選取 dataset 查看詳細狀態。")
        self._kpi_labels: dict[str, tk.Label] = {}
        self._kpi_vars = {key: tk.StringVar(value="-") for key in (
            "trading", "v2", "datasets", "next", "quota", "auto"
        )}
        self._kpi_detail_vars = {key: tk.StringVar(value="-") for key in self._kpi_vars}
        self._build_ui()
        self.refresh_local_status()

    def _build_ui(self):
        controls = ttk.Frame(self, style=WORKBENCH_FRAME_STYLE)
        controls.pack(fill="x", pady=(0, 6))
        ttk.Button(controls, text="刷新狀態（0 quota）", command=self.refresh_local_status, style=WORKBENCH_BUTTON_STYLE).pack(side="left", padx=(0, 6))
        self._due_button = ttk.Button(controls, text="立即檢查 Due", command=lambda: self._start_action("due"), style=WORKBENCH_BUTTON_STYLE)
        self._due_button.pack(side="left", padx=(0, 6))
        self._full_button = ttk.Button(controls, text="完整更新 Trading 資料", command=lambda: self._start_action("full"), style=WORKBENCH_BUTTON_STYLE)
        self._full_button.pack(side="left", padx=(0, 10))
        self._scheduler_install_button = ttk.Button(controls, text="安裝 / 更新 Auto Sync", command=lambda: self._confirm_scheduler_action("scheduler_install"), style=WORKBENCH_BUTTON_STYLE)
        self._scheduler_install_button.pack(side="left", padx=(0, 6))
        self._scheduler_toggle_button = ttk.Button(controls, text="停用 Auto Sync", command=lambda: self._confirm_scheduler_action("scheduler_toggle"), style=WORKBENCH_BUTTON_STYLE)
        self._scheduler_toggle_button.pack(side="left", padx=(0, 6))
        self._scheduler_remove_button = ttk.Button(controls, text="移除 Auto Sync", command=lambda: self._confirm_scheduler_action("scheduler_remove"), style=WORKBENCH_BUTTON_STYLE)
        self._scheduler_remove_button.pack(side="left", padx=(0, 10))
        self._status_label = ttk.Label(controls, textvariable=self._status_var, style=WORKBENCH_INFO_LABEL_STYLE)
        self._status_label.pack(side="left", fill="x", expand=True)

        kpi = ttk.Frame(self, style=WORKBENCH_FRAME_STYLE)
        kpi.pack(fill="x", pady=(0, 6))
        labels = (
            ("Trading Ready", "trading"),
            ("V2 Archive", "v2"),
            ("Datasets", "datasets"),
            ("Next Check", "next"),
            ("Quota", "quota"),
            ("Auto Worker", "auto"),
        )
        for col, (title, key) in enumerate(labels):
            box = ttk.LabelFrame(kpi, text=title, padding=(8, 4), style=WORKBENCH_LABELLF_STYLE)
            box.grid(row=0, column=col, padx=(0, 6), sticky="nsew")
            primary = tk.Label(
                box,
                textvariable=self._kpi_vars[key],
                background=WORKBENCH_BG,
                foreground=WORKBENCH_TEXT,
                font=(WORKBENCH_UI_FONT[0], WORKBENCH_UI_FONT[1], "bold"),
                justify="center",
            )
            primary.pack(fill="x")
            ttk.Label(
                box,
                textvariable=self._kpi_detail_vars[key],
                style=WORKBENCH_MUTED_LABEL_STYLE,
                justify="center",
            ).pack(fill="x", pady=(1, 0))
            self._kpi_labels[key] = primary
            kpi.columnconfigure(col, weight=1)

        meters = ttk.Frame(self, style=WORKBENCH_FRAME_STYLE)
        meters.pack(fill="x", pady=(0, 6))
        readiness_box = ttk.LabelFrame(meters, text="V2 Target Freshness", padding=(8, 4), style=WORKBENCH_LABELLF_STYLE)
        readiness_box.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self._readiness_progress = ttk.Progressbar(readiness_box, maximum=100.0, mode="determinate")
        self._readiness_progress.pack(fill="x")
        self._readiness_text = tk.StringVar(value="-")
        self._readiness_label = ttk.Label(readiness_box, textvariable=self._readiness_text, style=WORKBENCH_INFO_LABEL_STYLE)
        self._readiness_label.pack(anchor="w", pady=(2, 0))

        quota_box = ttk.LabelFrame(meters, text="Last Observed Provider Quota", padding=(8, 4), style=WORKBENCH_LABELLF_STYLE)
        quota_box.pack(side="left", fill="x", expand=True)
        self._quota_progress = ttk.Progressbar(quota_box, maximum=100.0, mode="determinate")
        self._quota_progress.pack(fill="x")
        self._quota_text = tk.StringVar(value="尚無 quota evidence；本頁刷新不會查 provider。")
        self._quota_label = ttk.Label(quota_box, textvariable=self._quota_text, style=WORKBENCH_MUTED_LABEL_STYLE)
        self._quota_label.pack(anchor="w", pady=(2, 0))

        notebook = ttk.Notebook(self, style=WORKBENCH_NOTEBOOK_STYLE)
        notebook.pack(fill="both", expand=True)
        dataset_tab = ttk.Frame(notebook, padding=4, style=WORKBENCH_FRAME_STYLE)
        schedule_tab = ttk.Frame(notebook, padding=4, style=WORKBENCH_FRAME_STYLE)
        notebook.add(dataset_tab, text="Dataset Status")
        notebook.add(schedule_tab, text="Schedule / Activity")
        self._build_dataset_tab(dataset_tab)
        self._build_schedule_tab(schedule_tab)

    def _build_dataset_tab(self, master):
        columns = ("dataset", "status", "latest", "expected", "success", "next", "schedule", "schema", "coverage", "retries")
        frame = ttk.Frame(master, style=WORKBENCH_FRAME_STYLE)
        frame.pack(fill="both", expand=True)
        self._dataset_tree = ttk.Treeview(frame, columns=columns, show="headings", style=WORKBENCH_TREE_STYLE, height=22)
        self._configure_tree_tags(self._dataset_tree)
        headers = {
            "dataset": "Dataset", "status": "Status", "latest": "Latest", "expected": "Expected",
            "success": "Last Success", "next": "Next Check", "schedule": "Publish", "schema": "Schema",
            "coverage": "Coverage", "retries": "Retry P/Q/E",
        }
        widths = {"dataset": 245, "status": 110, "latest": 145, "expected": 165, "success": 125, "next": 180, "schedule": 115, "schema": 100, "coverage": 120, "retries": 90}
        for col in columns:
            self._dataset_tree.heading(col, text=headers[col])
            self._dataset_tree.column(col, width=widths[col], minwidth=70, anchor="w")
        y = ttk.Scrollbar(frame, orient="vertical", command=self._dataset_tree.yview, style=WORKBENCH_VSCROLL_STYLE)
        x = ttk.Scrollbar(frame, orient="horizontal", command=self._dataset_tree.xview, style=WORKBENCH_HSCROLL_STYLE)
        self._dataset_tree.configure(yscrollcommand=y.set, xscrollcommand=x.set)
        self._dataset_tree.grid(row=0, column=0, sticky="nsew")
        y.grid(row=0, column=1, sticky="ns")
        x.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        self._dataset_tree.bind("<<TreeviewSelect>>", self._on_dataset_select)
        ttk.Label(master, textvariable=self._detail_var, style=WORKBENCH_LABEL_STYLE, foreground=WORKBENCH_MUTED, justify="left", wraplength=1700).pack(fill="x", pady=(6, 0))

    def _build_schedule_tab(self, master):
        top = ttk.Frame(master, style=WORKBENCH_FRAME_STYLE)
        top.pack(fill="both", expand=True)
        self._schedule_tree = ttk.Treeview(top, columns=("next", "dataset", "status", "publish", "source"), show="headings", style=WORKBENCH_TREE_STYLE, height=16)
        self._configure_tree_tags(self._schedule_tree)
        for col, title, width in (
            ("next", "Next Check", 190), ("dataset", "Dataset", 280), ("status", "Status", 120),
            ("publish", "Expected Publish", 160), ("source", "Schedule Source", 420),
        ):
            self._schedule_tree.heading(col, text=title)
            self._schedule_tree.column(col, width=width, anchor="w")
        sy = ttk.Scrollbar(top, orient="vertical", command=self._schedule_tree.yview, style=WORKBENCH_VSCROLL_STYLE)
        self._schedule_tree.configure(yscrollcommand=sy.set)
        self._schedule_tree.pack(side="left", fill="both", expand=True)
        sy.pack(side="left", fill="y")

        activity_box = ttk.LabelFrame(master, text="Recent Activity", padding=4, style=WORKBENCH_LABELLF_STYLE)
        activity_box.pack(fill="both", expand=True, pady=(6, 0))
        self._activity_tree = ttk.Treeview(activity_box, columns=("at", "dataset", "result", "error"), show="headings", style=WORKBENCH_TREE_STYLE, height=8)
        self._configure_tree_tags(self._activity_tree)
        for col, title, width in (("at", "At", 150), ("dataset", "Dataset", 280), ("result", "Result", 160), ("error", "Error", 700)):
            self._activity_tree.heading(col, text=title)
            self._activity_tree.column(col, width=width, anchor="w")
        ay = ttk.Scrollbar(activity_box, orient="vertical", command=self._activity_tree.yview, style=WORKBENCH_VSCROLL_STYLE)
        self._activity_tree.configure(yscrollcommand=ay.set)
        self._activity_tree.pack(side="left", fill="both", expand=True)
        ay.pack(side="left", fill="y")

    @staticmethod
    def _configure_tree_tags(tree: ttk.Treeview):
        tree.tag_configure("warning", foreground=WORKBENCH_WARNING)
        tree.tag_configure("error", foreground=WORKBENCH_ERROR)

    def refresh_local_status(self):
        try:
            snapshot = build_market_data_ops_read_model(WORKBENCH_PROJECT_ROOT)
        except Exception as exc:
            self._status_var.set(f"狀態讀取失敗：{type(exc).__name__}: {exc}")
            self._status_label.configure(style=WORKBENCH_ERROR_LABEL_STYLE)
            return
        self._snapshot = snapshot
        self._render(snapshot)
        blockers = [str(item) for item in list(snapshot.get("trading_blocking_dependencies") or []) if str(item)]
        suffix = "" if not blockers else " | Trading BLOCKED: " + "；".join(blockers[:2])
        self._status_var.set(f"本地狀態已刷新（provider calls={snapshot.get('provider_calls', 0)}）{suffix}")
        self._status_label.configure(style=WORKBENCH_ERROR_LABEL_STYLE if blockers else WORKBENCH_INFO_LABEL_STYLE)

    def _render(self, snapshot: dict[str, object]):
        target = snapshot.get("trading_target_date") or "-"
        strategy_id = snapshot.get("trading_strategy_id") or "-"
        required_v2 = int(snapshot.get("trading_required_v2_count") or 0)
        ready_v2 = int(snapshot.get("trading_ready_v2_count") or 0)
        trading_tag = "success" if snapshot.get("trading_ready") else "error"
        self._kpi_vars["trading"].set("READY" if snapshot.get("trading_ready") else "BLOCKED")
        self._kpi_detail_vars["trading"].set(f"{target}\n{strategy_id} | required V2 {ready_v2}/{required_v2}")
        self._kpi_labels["trading"].configure(foreground=_color_for_tag(trading_tag))

        v2_status = snapshot.get("v2_status") or "-"
        self._kpi_vars["v2"].set(v2_status)
        self._kpi_detail_vars["v2"].set(str(snapshot.get("v2_latest_sync_target_date") or "-"))
        self._kpi_labels["v2"].configure(foreground=_color_for_tag(_status_tag(v2_status)))

        ready_count = int(snapshot.get("ready_count") or 0)
        dataset_count = int(snapshot.get("dataset_count") or 0)
        due_count = int(snapshot.get("due_count") or 0)
        self._kpi_vars["datasets"].set(f"{ready_count}/{dataset_count} READY")
        self._kpi_detail_vars["datasets"].set(f"Due {due_count}")
        datasets_ready = int(snapshot.get("ready_count") or 0) == int(snapshot.get("dataset_count") or 0) and int(snapshot.get("due_count") or 0) == 0
        self._kpi_labels["datasets"].configure(foreground=_color_for_tag("success" if datasets_ready else "warning"))

        self._kpi_vars["next"].set(_fmt_datetime(snapshot.get("next_check_at")))
        self._kpi_detail_vars["next"].set("next global check")
        self._kpi_labels["next"].configure(foreground=WORKBENCH_INFO)

        quota_used, quota_limit = snapshot.get("quota_user_count"), snapshot.get("quota_limit")
        quota_state = str(snapshot.get("quota_observation_status") or "NONE")
        self._kpi_vars["quota"].set("NO EVIDENCE" if quota_limit is None else f"{quota_used}/{quota_limit}")
        self._kpi_detail_vars["quota"].set(
            "provider usage unknown"
            if quota_limit is None
            else f"{quota_state} · {_fmt_datetime(snapshot.get('quota_observed_at'))}"
        )
        quota_tag = "success" if quota_state == "CURRENT" else "warning" if quota_state == "STALE" else "muted"
        self._kpi_labels["quota"].configure(foreground=_color_for_tag(quota_tag))

        auto_text = "ON" if snapshot.get("auto_sync_active") else "OFF"
        scheduler_status = str(snapshot.get("scheduler_registration_status") or "-")
        scheduler_next = _fmt_datetime(snapshot.get("scheduler_next_run_at"))
        self._kpi_vars["auto"].set(auto_text)
        self._kpi_detail_vars["auto"].set(
            f"wake {snapshot.get('scheduler_wake_minutes') or '-'}m\nTask: {scheduler_status} | next {scheduler_next}"
        )
        self._kpi_labels["auto"].configure(
            foreground=_color_for_tag("success" if snapshot.get("auto_sync_active") else _status_tag(scheduler_status))
        )
        self._render_scheduler_controls(snapshot)

        total = int(snapshot.get("dataset_count") or 0)
        ready = int(snapshot.get("ready_count") or 0)
        readiness_pct = 100.0 * ready / total if total else 0.0
        self._readiness_progress["value"] = readiness_pct
        counts = dict(snapshot.get("status_counts") or {})
        count_text = " | ".join(f"{key} {counts[key]}" for key in sorted(counts))
        self._readiness_text.set(f"target {snapshot.get('update_target_date') or '-'} | {ready}/{total} ({readiness_pct:.1f}%) | {count_text or '-'}")
        self._readiness_label.configure(style=WORKBENCH_LABEL_STYLE)

        quota_pct = snapshot.get("quota_percent")
        self._quota_progress["value"] = float(quota_pct or 0.0)
        if quota_limit is None:
            self._quota_text.set("尚無 quota evidence；刷新此頁不會查 provider。")
            self._quota_label.configure(style=WORKBENCH_MUTED_LABEL_STYLE)
        else:
            self._quota_text.set(
                f"{quota_state} | quota={quota_used}/{quota_limit} ({float(quota_pct or 0):.1f}%) | observed {_fmt_datetime(snapshot.get('quota_observed_at'))} | "
                f"last auto data/usage={snapshot.get('latest_auto_data_requests') or 0}/{snapshot.get('latest_auto_usage_requests') or 0}"
            )
            self._quota_label.configure(style=WORKBENCH_LABEL_STYLE)

        for tree in (self._dataset_tree, self._schedule_tree, self._activity_tree):
            for iid in tree.get_children():
                tree.delete(iid)
        self._dataset_by_iid.clear()

        datasets = list(snapshot.get("datasets") or [])
        for index, row in enumerate(datasets):
            iid = f"d{index}"
            self._dataset_by_iid[iid] = dict(row)
            schedule = f"D+{int(row.get('publication_day_offset') or 0)} {row.get('publication_first_check_time') or '-'}"
            projected_status = row.get("projected_status") or row.get("status")
            self._dataset_tree.insert("", "end", iid=iid, tags=_attention_tags(projected_status), values=(
                row.get("dataset"), projected_status, row.get("latest_display") or "UNKNOWN",
                row.get("expected_display") or "UNKNOWN", _fmt_datetime(row.get("last_success_at")), _fmt_datetime(row.get("next_check_display")),
                schedule, row.get("schema_status") or "-", row.get("coverage_status") or "-", _retry_text(row),
            ))

        scheduler_next = snapshot.get("scheduler_next_run_at")
        scheduler_source = f"{snapshot.get('scheduler_app_path') or 'apps/market_data_auto_update.py'} | {snapshot.get('scheduler_wake_minutes') or '-'}m"
        drift = tuple(snapshot.get("scheduler_drift_reasons") or ())
        if drift:
            scheduler_source += " | drift=" + ",".join(str(item) for item in drift)
        if snapshot.get("scheduler_error"):
            scheduler_source += " | error=" + str(snapshot.get("scheduler_error"))[:180]
        self._schedule_tree.insert("", "end", tags=_attention_tags(snapshot.get("scheduler_registration_status")), values=(
            _fmt_datetime(scheduler_next),
            "Windows Auto Sync Worker",
            snapshot.get("scheduler_registration_status") or "-",
            _fmt_datetime(scheduler_next),
            scheduler_source,
        ))
        discovery_next = snapshot.get("market_date_discovery_next_check_at")
        if discovery_next:
            self._schedule_tree.insert("", "end", values=(
                _fmt_datetime(discovery_next),
                "Trading Market Date Discovery",
                "PROBE",
                _fmt_datetime(discovery_next),
                "TaiwanStockPriceAdj canonical completed-day probe",
            ))
        schedule_rows = sorted(datasets, key=lambda row: (str(row.get("next_check_at") or "9999"), str(row.get("dataset") or "")))
        for row in schedule_rows:
            projected_status = row.get("projected_status") or row.get("status")
            self._schedule_tree.insert("", "end", tags=_attention_tags(projected_status), values=(
                _fmt_datetime(row.get("next_check_display")), row.get("dataset"), projected_status,
                _fmt_datetime(row.get("expected_publish_at")), row.get("publication_schedule_source") or "-",
            ))
        if snapshot.get("market_date_discovery_last_probe_at"):
            discovery_result = snapshot.get("market_date_discovery_last_result") or "-"
            self._activity_tree.insert("", "end", tags=_attention_tags(discovery_result), values=(
                _fmt_datetime(snapshot.get("market_date_discovery_last_probe_at")),
                "Trading Market Date Discovery",
                discovery_result,
                "",
            ))
        for row in snapshot.get("recent_activity") or []:
            activity_result = row.get("result") or "-"
            self._activity_tree.insert("", "end", tags=_attention_tags(activity_result), values=(
                _fmt_datetime(row.get("at")), row.get("dataset"), activity_result, row.get("error") or "",
            ))

    def _on_dataset_select(self, _event=None):
        selected = self._dataset_tree.selection()
        if not selected:
            return
        row = self._dataset_by_iid.get(selected[0], {})
        verified = "provider-verified" if row.get("publication_schedule_verified") else "fallback"
        self._detail_var.set(
            f"{row.get('dataset')} | category={row.get('category')} cadence={row.get('cadence')} | "
            f"publication={row.get('publication_first_check_time')} D+{row.get('publication_day_offset')} ({verified}) | "
            f"PK={row.get('primary_key_hint') or '-'} | last_result={row.get('last_attempt_result') or '-'} | "
            f"error={row.get('last_error') or '-'}"
        )

    def _render_scheduler_controls(self, snapshot: dict[str, object]):
        if self._action_thread is not None and self._action_thread.is_alive():
            return
        supported = bool(snapshot.get("scheduler_supported"))
        installed = bool(snapshot.get("scheduler_installed"))
        enabled = bool(snapshot.get("scheduler_enabled"))
        self._scheduler_install_button.configure(state="normal" if supported else "disabled")
        self._scheduler_remove_button.configure(state="normal" if supported and installed else "disabled")
        self._scheduler_toggle_button.configure(
            text="停用 Auto Sync" if enabled else "啟用 Auto Sync",
            state="normal" if supported and installed else "disabled",
        )

    def _set_action_state(self, state: str):
        self._due_button.configure(state=state)
        self._full_button.configure(state=state)
        self._scheduler_install_button.configure(state=state)
        self._scheduler_toggle_button.configure(state=state)
        self._scheduler_remove_button.configure(state=state)

    def _confirm_scheduler_action(self, action: str):
        if action == "scheduler_install":
            prompt = f"建立或更新 Windows Task Scheduler Auto Sync？\n\n每 {self._snapshot.get('scheduler_wake_minutes') or 15} 分鐘喚醒一次，並在登入 Windows 時立即喚醒；沒有 due data 時不使用 FinMind quota。"
        elif action == "scheduler_remove":
            prompt = "移除 Windows Task Scheduler 的 Market Data Auto Sync？\n\n這只移除 OS 排程，不會刪除 Market Data、state 或 Provider Snapshot。"
        else:
            enabled = bool(self._snapshot.get("scheduler_enabled"))
            prompt = "停用 Auto Sync 排程？" if enabled else "啟用 Auto Sync 排程？"
        if not messagebox.askyesno("Market Data Auto Sync", prompt, parent=self):
            return
        self._start_action(action)

    def _start_action(self, action: str):
        if self._action_thread is not None and self._action_thread.is_alive():
            self._status_var.set("Market Data operation 執行中。")
            return
        self._set_action_state("disabled")
        labels = {
            "due": "Due 檢查",
            "full": "完整 Trading 更新",
            "scheduler_install": "安裝 / 更新 Auto Sync",
            "scheduler_toggle": "切換 Auto Sync",
            "scheduler_remove": "移除 Auto Sync",
        }
        self._status_var.set(f"執行中：{labels.get(action, action)}")
        self._status_label.configure(style=WORKBENCH_WARNING_LABEL_STYLE)
        self._action_thread = threading.Thread(target=self._run_action, args=(action,), name=f"workbench-data-ops-{action}", daemon=True)
        self._action_thread.start()

    def _run_action(self, action: str):
        console_progress = MarketDataDailyConsoleProgress() if action in {"due", "full"} else None
        try:
            if action == "due":
                result = run_trading_market_data_auto_update(
                    project_root=WORKBENCH_PROJECT_ROOT,
                    progress_fn=console_progress.progress,
                    quota_wait_fn=console_progress.quota_wait,
                )
            elif action == "full":
                result = run_trading_market_data_update(
                    project_root=WORKBENCH_PROJECT_ROOT,
                    progress_fn=console_progress.progress,
                    quota_wait_fn=console_progress.quota_wait,
                )
            elif action == "scheduler_install":
                result = install_or_update_market_data_scheduler(WORKBENCH_PROJECT_ROOT)
            elif action == "scheduler_toggle":
                result = set_market_data_scheduler_enabled(
                    WORKBENCH_PROJECT_ROOT,
                    enabled=not bool(self._snapshot.get("scheduler_enabled")),
                )
            elif action == "scheduler_remove":
                result = remove_market_data_scheduler(WORKBENCH_PROJECT_ROOT)
            else:
                raise ValueError(f"未知 Market Data Ops action: {action}")
        except Exception as exc:
            if console_progress is not None:
                console_progress.close()
            self.after(0, self._finish_action_error, exc)
            return
        if console_progress is not None:
            canonical_result = result.get("market_data_v2_archive") if action == "full" and isinstance(result, dict) else result
            if isinstance(canonical_result, dict):
                console_progress.result(canonical_result, label="Workbench Full Update" if action == "full" else "Workbench Due Check")
            else:
                console_progress.close()
        self.after(0, self._finish_action_success, result)

    def _finish_action_success(self, result):
        self._action_thread = None
        self._set_action_state("normal")
        self.refresh_local_status()
        status = result.get("status") if isinstance(result, dict) else None
        canonical_result = (
            result.get("market_data_v2_archive")
            if isinstance(result, dict) and isinstance(result.get("market_data_v2_archive"), dict)
            else result
        )
        quota_error = canonical_result.get("quota_refresh_error") if isinstance(canonical_result, dict) else None
        if quota_error:
            self._status_var.set(f"完成：{status or 'OK'} | Quota refresh WARN：{quota_error}")
            self._status_label.configure(style=WORKBENCH_WARNING_LABEL_STYLE)
        else:
            self._status_var.set(f"完成：{status or 'OK'}")
            self._status_label.configure(style=WORKBENCH_SUCCESS_LABEL_STYLE)

    def _finish_action_error(self, exc: Exception):
        self._action_thread = None
        self._set_action_state("normal")
        self.refresh_local_status()
        self._status_var.set(f"FAIL：{type(exc).__name__}: {exc}")
        self._status_label.configure(style=WORKBENCH_ERROR_LABEL_STYLE)
        messagebox.showerror("Market Data Ops", f"{type(exc).__name__}: {exc}", parent=self)


__all__ = ["MarketDataOpsPanel"]
