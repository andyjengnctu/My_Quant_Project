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


_STATUS_MARKERS = {
    "success": "🟢",
    "warning": "🟡",
    "error": "🔴",
    "muted": "⚪",
    "info": "🔵",
}


def _status_display(value: object) -> str:
    text = str(value or "-").strip() or "-"
    if text == "-":
        return text
    return f"{_STATUS_MARKERS[_status_tag(text)]} {text}"


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
        self._required_dataset_box = ttk.LabelFrame(
            master,
            text="Trading 必要資料",
            padding=4,
            style=WORKBENCH_LABELLF_STYLE,
        )
        self._required_dataset_box.pack(fill="x")
        self._required_dataset_group = self._build_dataset_table(self._required_dataset_box, height=6)

        self._other_dataset_box = ttk.LabelFrame(
            master,
            text="其它完整資料",
            padding=4,
            style=WORKBENCH_LABELLF_STYLE,
        )
        self._other_dataset_box.pack(fill="both", expand=True, pady=(6, 0))
        self._other_dataset_group = self._build_dataset_table(self._other_dataset_box, height=15)

        ttk.Label(
            master,
            textvariable=self._detail_var,
            style=WORKBENCH_LABEL_STYLE,
            foreground=WORKBENCH_MUTED,
            justify="left",
            wraplength=1700,
        ).pack(fill="x", pady=(6, 0))

    def _build_dataset_table(self, master, *, height: int) -> ttk.Treeview:
        frame = ttk.Frame(master, style=WORKBENCH_FRAME_STYLE)
        frame.pack(fill="both", expand=True)
        columns = (
            ("dataset", "Dataset", 205, 135),
            ("name_zh", "中文名稱", 190, 120),
            ("status", "Status", 115, 90),
            ("latest", "Latest", 110, 85),
            ("expected_publish", "Expected Publish", 140, 110),
            ("success", "Last Success", 120, 95),
            ("next", "Next Check", 120, 95),
            ("schema", "Schema", 115, 90),
            ("coverage", "Coverage", 120, 90),
            ("retries", "Retry P/Q/E", 95, 80),
        )
        tree = ttk.Treeview(
            frame,
            columns=tuple(item[0] for item in columns),
            show="headings",
            style=WORKBENCH_TREE_STYLE,
            height=height,
            selectmode="browse",
        )
        for column, title, width, minwidth in columns:
            tree.heading(column, text=title)
            tree.column(column, width=width, minwidth=minwidth, anchor="w", stretch=True)
        sy = ttk.Scrollbar(frame, orient="vertical", command=tree.yview, style=WORKBENCH_VSCROLL_STYLE)
        sx = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        tree.grid(row=0, column=0, sticky="nsew")
        sy.grid(row=0, column=1, sticky="ns")
        sx.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        tree.bind("<<TreeviewSelect>>", self._on_dataset_select)
        return tree

    def _build_schedule_tab(self, master):
        top = ttk.Frame(master, style=WORKBENCH_FRAME_STYLE)
        top.pack(fill="both", expand=True)
        columns = (
            ("dataset", "Dataset", 285, 180),
            ("name_zh", "中文名稱", 215, 140),
            ("status", "Status", 150, 105),
            ("publish", "Expected Publish", 175, 125),
            ("source", "Schedule Source", 650, 300),
        )
        self._schedule_tree = ttk.Treeview(
            top,
            columns=tuple(item[0] for item in columns),
            show="headings",
            style=WORKBENCH_TREE_STYLE,
            height=16,
            selectmode="browse",
        )
        for column, title, width, minwidth in columns:
            self._schedule_tree.heading(column, text=title)
            self._schedule_tree.column(column, width=width, minwidth=minwidth, anchor="w", stretch=True)
        sy = ttk.Scrollbar(top, orient="vertical", command=self._schedule_tree.yview, style=WORKBENCH_VSCROLL_STYLE)
        sx = ttk.Scrollbar(top, orient="horizontal", command=self._schedule_tree.xview)
        self._schedule_tree.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        self._schedule_tree.grid(row=0, column=0, sticky="nsew")
        sy.grid(row=0, column=1, sticky="ns")
        sx.grid(row=1, column=0, sticky="ew")
        top.rowconfigure(0, weight=1)
        top.columnconfigure(0, weight=1)

        activity_box = ttk.LabelFrame(master, text="Recent Activity", padding=4, style=WORKBENCH_LABELLF_STYLE)
        activity_box.pack(fill="both", expand=True, pady=(6, 0))
        self._activity_tree = ttk.Treeview(
            activity_box,
            columns=("at", "dataset", "name_zh", "result", "detail"),
            show="headings",
            style=WORKBENCH_TREE_STYLE,
            height=8,
        )
        for col, title, width, minwidth in (
            ("at", "At", 145, 110),
            ("dataset", "Dataset", 260, 170),
            ("name_zh", "中文名稱", 220, 140),
            ("result", "Result", 150, 105),
            ("detail", "Detail", 650, 300),
        ):
            self._activity_tree.heading(col, text=title)
            self._activity_tree.column(col, width=width, minwidth=minwidth, anchor="w", stretch=True)
        ay = ttk.Scrollbar(activity_box, orient="vertical", command=self._activity_tree.yview, style=WORKBENCH_VSCROLL_STYLE)
        ax = ttk.Scrollbar(activity_box, orient="horizontal", command=self._activity_tree.xview)
        self._activity_tree.configure(yscrollcommand=ay.set, xscrollcommand=ax.set)
        self._activity_tree.grid(row=0, column=0, sticky="nsew")
        ay.grid(row=0, column=1, sticky="ns")
        ax.grid(row=1, column=0, sticky="ew")
        activity_box.rowconfigure(0, weight=1)
        activity_box.columnconfigure(0, weight=1)

    def _on_dataset_select(self, event) -> None:
        selected = event.widget.selection()
        if selected:
            self._show_dataset_detail(selected[0])

    def _show_dataset_detail(self, iid: str) -> None:
        row = self._dataset_by_iid.get(iid, {})
        if not row:
            return
        verified = "provider-verified" if row.get("publication_schedule_verified") else "fallback"
        self._detail_var.set(
            f"{row.get('dataset')}｜{row.get('display_name_zh') or '-'} | category={row.get('category')} cadence={row.get('cadence')} | "
            f"publication={row.get('publication_first_check_time')} D+{row.get('publication_day_offset')} ({verified}) | "
            f"PK={row.get('primary_key_hint') or '-'} | last_result={row.get('last_attempt_result') or '-'} | "
            f"error={row.get('last_error') or '-'}"
        )

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

    @staticmethod
    def _clear_tree(tree: ttk.Treeview) -> None:
        for iid in tree.get_children():
            tree.delete(iid)

    @staticmethod
    def _expected_publish_sort_key(row: dict[str, object]) -> tuple[int, str, str]:
        value = str(row.get("expected_publish_at") or "").strip()
        return (0 if value else 1, value, str(row.get("dataset") or ""))

    def _insert_dataset_row(
        self,
        tree: ttk.Treeview,
        *,
        iid: str,
        row: dict[str, object],
    ) -> None:
        projected_status = row.get("projected_status") or row.get("status") or "-"
        schema_status = row.get("schema_status") or "-"
        coverage_status = row.get("coverage_status") or "-"
        tree.insert(
            "",
            "end",
            iid=iid,
            values=(
                row.get("dataset") or "-",
                row.get("display_name_zh") or "-",
                _status_display(projected_status),
                row.get("latest_data_date") or "-",
                _fmt_datetime(row.get("expected_publish_at")),
                _fmt_datetime(row.get("last_success_at")),
                _fmt_datetime(row.get("next_check_at")),
                _status_display(schema_status),
                _status_display(coverage_status),
                _retry_text(row),
            ),
        )

    def _insert_schedule_row(
        self,
        *,
        iid: str,
        dataset: object,
        display_name_zh: object,
        status: object,
        expected_publish_at: object,
        source: object,
    ) -> None:
        self._schedule_tree.insert(
            "",
            "end",
            iid=iid,
            values=(
                dataset or "-",
                display_name_zh or "-",
                _status_display(status),
                _fmt_datetime(expected_publish_at),
                source or "-",
            ),
        )

    def _render(self, snapshot: dict[str, object]):
        target = snapshot.get("trading_target_date") or "-"
        strategy_id = snapshot.get("trading_strategy_id") or "-"
        required_v2 = int(snapshot.get("trading_required_v2_count") or 0)
        ready_v2 = int(snapshot.get("trading_ready_v2_count") or 0)
        trading_tag = "success" if snapshot.get("trading_ready") else "error"
        trading_primary = f"{ready_v2}/{required_v2} READY"
        if not snapshot.get("trading_ready"):
            trading_primary += " · BLOCKED"
        self._kpi_vars["trading"].set(trading_primary)
        self._kpi_detail_vars["trading"].set(f"{target} | {strategy_id}")
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
        datasets_ready = ready_count == dataset_count and due_count == 0
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
            f"wake {snapshot.get('scheduler_wake_minutes') or '-'}m | Task: {scheduler_status} | next {scheduler_next}"
        )
        self._kpi_labels["auto"].configure(
            foreground=_color_for_tag("success" if snapshot.get("auto_sync_active") else _status_tag(scheduler_status))
        )
        self._render_scheduler_controls(snapshot)

        total = dataset_count
        ready = ready_count
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

        self._clear_tree(self._required_dataset_group)
        self._clear_tree(self._other_dataset_group)
        self._clear_tree(self._schedule_tree)
        for iid in self._activity_tree.get_children():
            self._activity_tree.delete(iid)
        self._dataset_by_iid.clear()

        datasets = [dict(row) for row in snapshot.get("datasets") or []]
        required_names = {str(item) for item in snapshot.get("trading_required_v2_datasets") or []}
        required_rows = sorted(
            (row for row in datasets if str(row.get("dataset") or "") in required_names),
            key=self._expected_publish_sort_key,
        )
        other_rows = sorted(
            (row for row in datasets if str(row.get("dataset") or "") not in required_names),
            key=self._expected_publish_sort_key,
        )
        self._required_dataset_box.configure(text=f"Trading 必要資料（{len(required_rows)}）")
        self._other_dataset_box.configure(text=f"其它完整資料（{len(other_rows)}）")
        for section, group, rows in (
            ("required", self._required_dataset_group, required_rows),
            ("other", self._other_dataset_group, other_rows),
        ):
            for index, row in enumerate(rows):
                iid = f"{section}:{index}"
                self._dataset_by_iid[iid] = row
                self._insert_dataset_row(group, iid=iid, row=row)

        scheduler_next = snapshot.get("scheduler_next_run_at")
        scheduler_source = f"{snapshot.get('scheduler_app_path') or 'apps/market_data_auto_update.py'} | {snapshot.get('scheduler_wake_minutes') or '-'}m"
        drift = tuple(snapshot.get("scheduler_drift_reasons") or ())
        if drift:
            scheduler_source += " | drift=" + ",".join(str(item) for item in drift)
        if snapshot.get("scheduler_error"):
            scheduler_source += " | error=" + str(snapshot.get("scheduler_error"))[:180]
        self._insert_schedule_row(
            iid="system:auto_worker",
            dataset="Windows Auto Sync Worker",
            display_name_zh="Windows 自動同步排程",
            status=snapshot.get("scheduler_registration_status") or "-",
            expected_publish_at=scheduler_next,
            source=scheduler_source,
        )
        discovery_next = snapshot.get("market_date_discovery_next_check_at")
        if discovery_next:
            self._insert_schedule_row(
                iid="system:market_date_discovery",
                dataset="Trading Market Date Discovery",
                display_name_zh="交易日探索",
                status="PROBE",
                expected_publish_at=discovery_next,
                source="TaiwanStockPriceAdj canonical completed-day probe",
            )
        schedule_rows = sorted(datasets, key=self._expected_publish_sort_key)
        for index, row in enumerate(schedule_rows):
            self._insert_schedule_row(
                iid=f"schedule:{index}",
                dataset=row.get("dataset"),
                display_name_zh=row.get("display_name_zh"),
                status=row.get("projected_status") or row.get("status"),
                expected_publish_at=row.get("expected_publish_at"),
                source=row.get("publication_schedule_source") or "-",
            )

        activity_rows: list[dict[str, object]] = []
        if snapshot.get("market_date_discovery_last_probe_at"):
            activity_rows.append(
                {
                    "at": snapshot.get("market_date_discovery_last_probe_at"),
                    "dataset": "Trading Market Date Discovery",
                    "display_name_zh": "交易日探索",
                    "result": snapshot.get("market_date_discovery_last_result") or "-",
                    "detail": "-",
                }
            )
        activity_rows.extend(dict(row) for row in snapshot.get("recent_activity") or [])
        activity_rows.sort(
            key=lambda row: (
                str(row.get("at") or ""),
                1 if row.get("activity_type") == "quota" else 0,
            ),
            reverse=True,
        )
        visible_activity = activity_rows[:20]
        quota_activity = next(
            (row for row in activity_rows if row.get("activity_type") == "quota"),
            None,
        )
        if quota_activity is not None and not any(row.get("activity_type") == "quota" for row in visible_activity):
            visible_activity = visible_activity[:19] + [quota_activity]
        for row in visible_activity:
            self._activity_tree.insert(
                "",
                "end",
                values=(
                    _fmt_datetime(row.get("at")),
                    row.get("dataset") or "-",
                    row.get("display_name_zh") or "-",
                    row.get("result") or "-",
                    row.get("detail") or row.get("error") or "-",
                ),
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
