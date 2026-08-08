"""Project-wide config-driven formal Audit runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config.audit import (
    AUDIT_OUTPUT_ROOT,
    get_audit_definitions,
    get_audit_module_ids,
    get_enabled_audit_definitions,
)
from core.console_report import (
    project_relative_display_path,
    render_key_values,
    render_table,
    render_title,
)
from tools.audit.catalog import get_audit_handler, validate_audit_catalog

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _module_title(module_id: str) -> str:
    return str(module_id).replace("_", " ").title()


def collect_audit_status(
    module_id: str,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    definitions = get_audit_definitions(module_id)
    validate_audit_catalog(definitions)
    rows: list[dict[str, Any]] = []
    statuses: dict[str, dict[str, Any]] = {}
    for definition in definitions:
        spec = get_audit_handler(definition)
        status = spec.load_status_handler()(definition, project_root=root)
        statuses[definition.audit_id] = status
        rows.append(
            {
                "enabled": bool(definition.enabled),
                "audit_id": definition.audit_id,
                "audit_type": definition.audit_type,
                "description": definition.description,
                "status": str(status.get("status") or "BLOCKED"),
                "reason": str(status.get("reason") or ""),
                "source": dict(status.get("source") or {}),
            }
        )
    enabled_rows = [row for row in rows if row["enabled"]]
    overall = (
        "READY"
        if enabled_rows and all(row["status"] == "READY" for row in enabled_rows)
        else "BLOCKED"
        if enabled_rows
        else "DISABLED"
    )
    return {
        "module_id": module_id,
        "overall_status": overall,
        "rows": rows,
        "statuses": statuses,
    }


def render_audit_status(
    module_id: str,
    *,
    project_root: Path = PROJECT_ROOT,
) -> str:
    status = collect_audit_status(module_id, project_root=project_root)
    rows = []
    for row in status["rows"]:
        source = row["source"]
        source_text = str(
            source.get("display")
            or source.get("arm_id")
            or source.get("candidate_arm_id")
            or "-"
        )
        action = (
            "SKIP"
            if not row["enabled"]
            else "READ"
            if row["status"] == "READY"
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
            render_title(f"{_module_title(module_id)} Audit 設定與工件狀態"),
            render_key_values(
                (
                    ("設定檔", "config/audit.py"),
                    ("Catalog", "tools/audit/catalog.py"),
                    ("Module", module_id),
                    ("整體狀態", status["overall_status"]),
                    ("輸出根目錄", AUDIT_OUTPUT_ROOT),
                    ("契約", "只讀既有正式工件；不得修改runtime／training"),
                )
            ),
            render_table(
                ("開關", "動作", "Audit", "類型", "來源", "狀態", "用途", "阻擋原因"),
                rows,
            ),
        )
    )


def run_enabled_audits(
    module_id: str,
    *,
    project_root: Path = PROJECT_ROOT,
    quiet: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    definitions = get_enabled_audit_definitions(module_id)
    if not definitions:
        raise RuntimeError(f"config/audit.py沒有啟用任何{module_id} Audit")
    validate_audit_catalog(get_audit_definitions(module_id))
    status = collect_audit_status(module_id, project_root=root)
    if status["overall_status"] != "READY":
        raise RuntimeError("目前Audit缺少只讀來源工件；請先查看Audit設定與工件狀態")
    outputs: dict[str, Any] = {}
    for definition in definitions:
        spec = get_audit_handler(definition)
        outputs[definition.audit_id] = spec.load_run_handler()(
            definition,
            project_root=root,
            quiet=quiet,
        )
    return outputs


def render_latest_audit_summary(
    module_id: str,
    *,
    project_root: Path = PROJECT_ROOT,
) -> str:
    root = Path(project_root).resolve()
    rows = []
    for definition in get_audit_definitions(module_id):
        latest = (
            root
            / Path(AUDIT_OUTPUT_ROOT)
            / Path(definition.output_subdir)
            / "latest"
            / "audit.md"
        )
        rows.append(
            (
                definition.audit_id,
                "存在" if latest.is_file() else "尚無",
                project_relative_display_path(latest, project_root=root),
            )
        )
    return "\n\n".join(
        (
            render_title(f"{_module_title(module_id)} 最近 Audit 結果"),
            render_table(("Audit", "狀態", "Markdown"), rows),
        )
    )


def collect_project_audit_status(*, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    return {
        module_id: collect_audit_status(module_id, project_root=project_root)
        for module_id in get_audit_module_ids(enabled_only=True)
    }


__all__ = [
    "collect_audit_status",
    "collect_project_audit_status",
    "render_audit_status",
    "render_latest_audit_summary",
    "run_enabled_audits",
]
