"""Config-driven formal audit runner for Breakout Quality."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config.audit import AUDIT_OUTPUT_ROOT, get_audit_definitions, get_enabled_audit_definitions
from filters.breakout_quality.console_report import (
    project_relative_display_path,
    render_key_values,
    render_section,
    render_table,
    render_title,
)
from tools.filters.breakout_quality.audit_pass_quality import (
    collect_pass_quality_status,
    run_pass_quality_audit,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODULE_ID = "breakout_quality"


def _status_for_definition(definition, *, project_root: Path) -> dict[str, Any]:
    if definition.audit_type == "pass_quality":
        return collect_pass_quality_status(definition, project_root=project_root)
    raise ValueError(f"不支援的Breakout Quality audit type: {definition.audit_type}")


def collect_audit_status(*, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    root = Path(project_root).resolve()
    definitions = get_audit_definitions(MODULE_ID)
    rows = []
    statuses = {}
    for definition in definitions:
        status = _status_for_definition(definition, project_root=root)
        statuses[definition.audit_id] = status
        rows.append(
            {
                "enabled": bool(definition.enabled),
                "audit_id": definition.audit_id,
                "audit_type": definition.audit_type,
                "description": definition.description,
                "status": status["status"],
                "reason": status.get("reason", ""),
                "source": dict(status.get("source") or {}),
            }
        )
    enabled_rows = [row for row in rows if row["enabled"]]
    overall = (
        "READY"
        if enabled_rows and all(row["status"] == "READY" for row in enabled_rows)
        else "BLOCKED"
    )
    return {"overall_status": overall, "rows": rows, "statuses": statuses}


def render_audit_status(*, project_root: Path = PROJECT_ROOT) -> str:
    root = Path(project_root).resolve()
    status = collect_audit_status(project_root=root)
    rows = []
    for row in status["rows"]:
        source = row["source"]
        source_text = (
            f"{source.get('arm_id') or '-'}/{source.get('dl_id') or '-'}"
            if source
            else "-"
        )
        action = (
            "SKIP" if not row["enabled"]
            else "READ" if row["status"] == "READY"
            else "BLOCKED"
        )
        rows.append(
            (
                "ON" if row["enabled"] else "OFF",
                action,
                row["audit_id"],
                row["audit_type"],
                source_text,
                row["status"],
                row["description"],
                row["reason"],
            )
        )
    return "\n\n".join(
        (
            render_title("Breakout Quality Audit 設定與工件狀態"),
            render_key_values(
                (
                    ("設定檔", "config/audit.py"),
                    ("Module", MODULE_ID),
                    ("整體狀態", status["overall_status"]),
                    ("輸出根目錄", AUDIT_OUTPUT_ROOT),
                    ("語意", "Strategy validity / DL quality / Selector allocation 分層"),
                )
            ),
            render_table(
                ("開關", "動作", "Audit", "類型", "來源", "狀態", "用途", "阻擋原因"),
                rows,
            ),
        )
    )


def run_enabled_audits(
    *, project_root: Path = PROJECT_ROOT, quiet: bool = False
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    definitions = get_enabled_audit_definitions(MODULE_ID)
    if not definitions:
        raise RuntimeError("config/audit.py沒有啟用任何Breakout Quality Audit")
    status = collect_audit_status(project_root=root)
    if status["overall_status"] != "READY":
        raise RuntimeError("目前Audit缺少只讀來源工件；請先查看Audit設定與工件狀態")
    outputs = {}
    for definition in definitions:
        if definition.audit_type == "pass_quality":
            outputs[definition.audit_id] = run_pass_quality_audit(
                definition, project_root=root, quiet=quiet
            )
        else:
            raise ValueError(f"不支援的Audit type: {definition.audit_type}")
    return outputs


def render_latest_audit_summary(*, project_root: Path = PROJECT_ROOT) -> str:
    root = Path(project_root).resolve()
    rows = []
    for definition in get_audit_definitions(MODULE_ID):
        latest = root / Path(AUDIT_OUTPUT_ROOT) / Path(definition.output_subdir) / "latest" / "audit.md"
        rows.append(
            (
                definition.audit_id,
                "存在" if latest.is_file() else "尚無",
                project_relative_display_path(latest, project_root=root),
            )
        )
    return "\n\n".join(
        (
            render_title("Breakout Quality 最近 Audit 結果"),
            render_table(("Audit", "狀態", "Markdown"), rows),
        )
    )


__all__ = [
    "collect_audit_status",
    "render_audit_status",
    "render_latest_audit_summary",
    "run_enabled_audits",
]
