from __future__ import annotations

from datetime import datetime
from pathlib import Path
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from services.trading.market_data_auto_update import run_trading_market_data_auto_update
from services.trading.market_data_ops import build_market_data_ops_read_model
from services.trading.market_data_update import run_trading_market_data_update
from services.workbench_ui.workbench import (
    WORKBENCH_BUTTON_STYLE,
    WORKBENCH_FRAME_STYLE,
    WORKBENCH_HSCROLL_STYLE,
    WORKBENCH_LABEL_STYLE,
    WORKBENCH_LABELLF_STYLE,
    WORKBENCH_MUTED,
    WORKBENCH_TREE_STYLE,
    WORKBENCH_VSCROLL_STYLE,
)

WORKBENCH_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _fmt_datetime(value) -> str:
    text = str(value or "").strip()
    if not text:
        return "-"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return text
    return parsed.astimezone().strftime("%m-%d %H:%M") if parsed.tzinfo else parsed.strftime("%m-%d %H:%M")


def _fmt_date(value) -> str:
    text = str(value or "").strip()
    return text or "-"


def _retry_text(row: dict[str, object]) -> str:
    return "/".join(
        str(int(row.get(key) or 0))
        for key in ("publication_retry_count", "quota_defer_count", "error_retry_count")
    )


class MarketDataOpsPanel(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=6, style=WORKBENCH_FRAME_STYLE)
        self._action_thread: threading.Thread | None = None
        self._snapshot: dict[str, object] = {}
        self._dataset_by_iid: dict[str, dict[str, object]] = {}
        self._status_var = tk.StringVar(value="讀取 Market Data 狀態…")
        self._detail_var = tk.StringVar(value="選取 dataset 查看詳細狀態。")
        self._kpi_vars = {key: tk.StringVar(value="-") for key in (
            "trading", "v2", "datasets", "next", "quota", "auto"
        )}
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
        ttk.Label(controls, textvariable=self._status_var, style=WORKBENCH_LABEL_STYLE).pack(side="left", fill="x", expand=True)

        kpi = ttk.Frame(self, style=WORKBENCH_FRAME_STYLE)
        kpi.pack(fill="x", pady=(0, 6))
        labels = (
            ("Trading Data", "trading"),
            ("V2 Archive", "v2"),
            ("Datasets", "datasets"),
            ("Next Check", "next"),
            ("Quota", "quota"),
            ("Auto Worker", "auto"),
        )
        for col, (title, key) in enumerate(labels):
            box = ttk.LabelFrame(kpi, text=title, padding=(8, 4), style=WORKBENCH_LABELLF_STYLE)
            box.grid(row=0, column=col, padx=(0, 6), sticky="nsew")
            ttk.Label(box, textvariable=self._kpi_vars[key], style=WORKBENCH_LABEL_STYLE, justify="center").pack(fill="x")
            kpi.columnconfigure(col, weight=1)

        meters = ttk.Frame(self, style=WORKBENCH_FRAME_STYLE)
        meters.pack(fill="x", pady=(0, 6))
        readiness_box = ttk.LabelFrame(meters, text="V2 Readiness", padding=(8, 4), style=WORKBENCH_LABELLF_STYLE)
        readiness_box.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self._readiness_progress = ttk.Progressbar(readiness_box, maximum=100.0, mode="determinate")
        self._readiness_progress.pack(fill="x")
        self._readiness_text = tk.StringVar(value="-")
        ttk.Label(readiness_box, textvariable=self._readiness_text, style=WORKBENCH_LABEL_STYLE).pack(anchor="w", pady=(2, 0))

        quota_box = ttk.LabelFrame(meters, text="Last Observed Provider Quota", padding=(8, 4), style=WORKBENCH_LABELLF_STYLE)
        quota_box.pack(side="left", fill="x", expand=True)
        self._quota_progress = ttk.Progressbar(quota_box, maximum=100.0, mode="determinate")
        self._quota_progress.pack(fill="x")
        self._quota_text = tk.StringVar(value="尚無 quota evidence；本頁刷新不會查 provider。")
        ttk.Label(quota_box, textvariable=self._quota_text, style=WORKBENCH_LABEL_STYLE).pack(anchor="w", pady=(2, 0))

        notebook = ttk.Notebook(self)
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
        headers = {
            "dataset": "Dataset", "status": "Status", "latest": "Latest", "expected": "Expected",
            "success": "Last Success", "next": "Next Check", "schedule": "Publish", "schema": "Schema",
            "coverage": "Coverage", "retries": "Retry P/Q/E",
        }
        widths = {"dataset": 245, "status": 110, "latest": 95, "expected": 95, "success": 125, "next": 125, "schedule": 115, "schema": 100, "coverage": 120, "retries": 90}
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
        for col, title, width in (
            ("next", "Next Check", 145), ("dataset", "Dataset", 280), ("status", "Status", 120),
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
        for col, title, width in (("at", "At", 150), ("dataset", "Dataset", 280), ("result", "Result", 160), ("error", "Error", 700)):
            self._activity_tree.heading(col, text=title)
            self._activity_tree.column(col, width=width, anchor="w")
        ay = ttk.Scrollbar(activity_box, orient="vertical", command=self._activity_tree.yview, style=WORKBENCH_VSCROLL_STYLE)
        self._activity_tree.configure(yscrollcommand=ay.set)
        self._activity_tree.pack(side="left", fill="both", expand=True)
        ay.pack(side="left", fill="y")

    def refresh_local_status(self):
        try:
            snapshot = build_market_data_ops_read_model(WORKBENCH_PROJECT_ROOT)
        except Exception as exc:
            self._status_var.set(f"狀態讀取失敗：{type(exc).__name__}: {exc}")
            return
        self._snapshot = snapshot
        self._render(snapshot)
        self._status_var.set(f"本地狀態已刷新（provider calls={snapshot.get('provider_calls', 0)}）")

    def _render(self, snapshot: dict[str, object]):
        target = snapshot.get("trading_target_date") or "-"
        self._kpi_vars["trading"].set(f"{'READY' if snapshot.get('execution_data_ready') else 'NOT READY'}\n{target}")
        self._kpi_vars["v2"].set(f"{snapshot.get('v2_status') or '-'}\n{snapshot.get('v2_latest_sync_target_date') or '-'}")
        self._kpi_vars["datasets"].set(f"{int(snapshot.get('ready_count') or 0)}/{int(snapshot.get('dataset_count') or 0)} READY\nDue {int(snapshot.get('due_count') or 0)}")
        self._kpi_vars["next"].set(_fmt_datetime(snapshot.get("next_check_at")))
        quota_used, quota_limit = snapshot.get("quota_user_count"), snapshot.get("quota_limit")
        self._kpi_vars["quota"].set("-" if quota_limit is None else f"{quota_used}/{quota_limit}\n{_fmt_datetime(snapshot.get('quota_observed_at'))}")
        auto_text = "ON" if snapshot.get("auto_worker_enabled") else "OFF"
        self._kpi_vars["auto"].set(f"{auto_text} | wake {snapshot.get('scheduler_wake_minutes') or '-'}m\nTask Scheduler: 尚未由程式管理")

        total = int(snapshot.get("dataset_count") or 0)
        ready = int(snapshot.get("ready_count") or 0)
        readiness_pct = 100.0 * ready / total if total else 0.0
        self._readiness_progress["value"] = readiness_pct
        counts = dict(snapshot.get("status_counts") or {})
        count_text = " | ".join(f"{key} {counts[key]}" for key in sorted(counts))
        self._readiness_text.set(f"{ready}/{total} ({readiness_pct:.1f}%) | {count_text or '-'}")

        quota_pct = snapshot.get("quota_percent")
        self._quota_progress["value"] = float(quota_pct or 0.0)
        if quota_limit is None:
            self._quota_text.set("尚無 quota evidence；刷新此頁不會查 provider。")
        else:
            self._quota_text.set(
                f"quota={quota_used}/{quota_limit} ({float(quota_pct or 0):.1f}%) | observed {_fmt_datetime(snapshot.get('quota_observed_at'))} | "
                f"last auto data/usage={snapshot.get('latest_auto_data_requests') or 0}/{snapshot.get('latest_auto_usage_requests') or 0}"
            )

        for tree in (self._dataset_tree, self._schedule_tree, self._activity_tree):
            for iid in tree.get_children():
                tree.delete(iid)
        self._dataset_by_iid.clear()

        datasets = list(snapshot.get("datasets") or [])
        for index, row in enumerate(datasets):
            iid = f"d{index}"
            self._dataset_by_iid[iid] = dict(row)
            schedule = f"D+{int(row.get('publication_day_offset') or 0)} {row.get('publication_first_check_time') or '-'}"
            self._dataset_tree.insert("", "end", iid=iid, values=(
                row.get("dataset"), row.get("projected_status") or row.get("status"), _fmt_date(row.get("latest_data_date")),
                _fmt_date(row.get("latest_expected_date")), _fmt_datetime(row.get("last_success_at")), _fmt_datetime(row.get("next_check_at")),
                schedule, row.get("schema_status") or "-", row.get("coverage_status") or "-", _retry_text(row),
            ))

        schedule_rows = sorted(datasets, key=lambda row: (str(row.get("next_check_at") or "9999"), str(row.get("dataset") or "")))
        for row in schedule_rows:
            self._schedule_tree.insert("", "end", values=(
                _fmt_datetime(row.get("next_check_at")), row.get("dataset"), row.get("projected_status") or row.get("status"),
                _fmt_datetime(row.get("expected_publish_at")), row.get("publication_schedule_source") or "-",
            ))
        for row in snapshot.get("recent_activity") or []:
            self._activity_tree.insert("", "end", values=(
                _fmt_datetime(row.get("at")), row.get("dataset"), row.get("result") or "-", row.get("error") or "",
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

    def _set_action_state(self, state: str):
        self._due_button.configure(state=state)
        self._full_button.configure(state=state)

    def _start_action(self, action: str):
        if self._action_thread is not None and self._action_thread.is_alive():
            self._status_var.set("Market Data operation 執行中。")
            return
        self._set_action_state("disabled")
        label = "Due 檢查" if action == "due" else "完整 Trading 更新"
        self._status_var.set(f"執行中：{label}")
        self._action_thread = threading.Thread(target=self._run_action, args=(action,), name=f"workbench-data-ops-{action}", daemon=True)
        self._action_thread.start()

    def _run_action(self, action: str):
        try:
            if action == "due":
                result = run_trading_market_data_auto_update(project_root=WORKBENCH_PROJECT_ROOT)
            else:
                result = run_trading_market_data_update(project_root=WORKBENCH_PROJECT_ROOT)
        except Exception as exc:
            self.after(0, self._finish_action_error, exc)
            return
        self.after(0, self._finish_action_success, result)

    def _finish_action_success(self, result):
        self._action_thread = None
        self._set_action_state("normal")
        self.refresh_local_status()
        status = result.get("status") if isinstance(result, dict) else None
        self._status_var.set(f"完成：{status or 'OK'}")

    def _finish_action_error(self, exc: Exception):
        self._action_thread = None
        self._set_action_state("normal")
        self.refresh_local_status()
        self._status_var.set(f"FAIL：{type(exc).__name__}: {exc}")
        messagebox.showerror("Market Data Ops", f"{type(exc).__name__}: {exc}", parent=self)


__all__ = ["MarketDataOpsPanel"]
