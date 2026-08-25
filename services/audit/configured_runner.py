"""Config-driven formal Audit runner used by the Research application."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config.audit import (
    AUDIT_OUTPUT_ROOT,
    AuditDefinition,
    get_audit_definitions,
    get_enabled_audit_definitions,
)
from core.path_utils import project_relative_display_path
from services.audit.mfe_safety_quadrants import (
    AuditBlockedError,
    SUPPORTED_AUDIT_TYPE,
    load_latest_result,
    preflight,
    render_result,
    run_audit,
)

_SUPPORTED_TYPES = {SUPPORTED_AUDIT_TYPE}


def _definition_preflight(definition: AuditDefinition, *, project_root: Path) -> dict[str, Any]:
    if definition.audit_type == SUPPORTED_AUDIT_TYPE:
        return preflight(definition, project_root=project_root)
    return {
        "status": "BLOCKED",
        "blockers": [f"configured Audit runner尚未支援audit_type={definition.audit_type}"],
        "target_paths": [],
        "strategy_paths": [],
    }


def render_audit_status(module_id: str, *, project_root: str | Path) -> str:
    from core.console_report import render_key_values, render_section, render_table, render_title

    root = Path(project_root)
    definitions = get_audit_definitions(module_id)
    rows = []
    overall_ready = True
    for definition in definitions:
        state = _definition_preflight(definition, project_root=root)
        status = str(state["status"])
        if definition.enabled and status != "READY":
            overall_ready = False
        blockers = "；".join(str(item) for item in state.get("blockers", ())) or "-"
        rows.append(
            [
                "ON" if definition.enabled else "OFF",
                "RUN" if definition.enabled else "SKIP",
                definition.audit_id,
                definition.audit_type,
                status,
                definition.description,
                blockers,
            ]
        )
    overall_status = "READY" if overall_ready and any(item.enabled for item in definitions) else (
        "IDLE" if not any(item.enabled for item in definitions) else "BLOCKED"
    )
    lines = [render_title("Breakout Quality Audit 設定與工件狀態")]
    lines.append(
        render_key_values(
            [
                ("設定檔", "config/audit.py"),
                ("Module", module_id),
                ("整體狀態", overall_status),
                ("輸出根目錄", AUDIT_OUTPUT_ROOT),
                ("契約", "只讀既有正式工件；不得修改runtime／training"),
            ]
        )
    )
    if rows:
        lines.append(render_section("目前設定"))
        lines.append(
            render_table(
                ["開關", "動作", "Audit", "類型", "狀態", "用途", "阻擋原因"],
                rows,
            )
        )
    else:
        lines.append("目前沒有設定中的正式 Audit。")
    return "\n".join(lines)


def run_enabled_audits(module_id: str, *, project_root: str | Path) -> list[dict[str, Any]]:
    root = Path(project_root)
    definitions = get_enabled_audit_definitions(module_id)
    if not definitions:
        print("[SKIP] 目前沒有啟用的正式 Audit")
        return []
    unsupported = [item for item in definitions if item.audit_type not in _SUPPORTED_TYPES]
    if unsupported:
        raise ValueError(
            "configured Audit runner遇到尚未支援的audit_type: "
            + ", ".join(f"{item.audit_id}={item.audit_type}" for item in unsupported)
        )
    results: list[dict[str, Any]] = []
    for definition in definitions:
        state = _definition_preflight(definition, project_root=root)
        if state["status"] != "READY":
            raise AuditBlockedError(
                f"{definition.audit_id} BLOCKED: " + "；".join(state["blockers"])
            )
        print(f"[RUN] {definition.audit_id}")
        result = run_audit(definition, project_root=root)
        results.append(result)
        print(f"[DONE] {definition.audit_id} | fingerprint={result['config_fingerprint']}")
    return results


def render_latest_audit_summary(module_id: str, *, project_root: str | Path) -> str:
    from core.console_report import render_key_values, render_title

    root = Path(project_root)
    definitions = get_audit_definitions(module_id)
    rendered: list[str] = []
    for definition in definitions:
        if definition.audit_type != SUPPORTED_AUDIT_TYPE:
            continue
        result = load_latest_result(definition, project_root=root)
        if result is not None:
            rendered.append(render_result(result))
    if rendered:
        return "\n\n".join(rendered)
    output_root = root / AUDIT_OUTPUT_ROOT
    return (
        render_title("Breakout Quality Audit 最近結果")
        + "\n"
        + render_key_values(
            [
                ("狀態", "尚無可讀取的正式 Audit 結果"),
                ("輸出根目錄", project_relative_display_path(output_root, project_root=root)),
            ]
        )
    )


__all__ = [
    "render_audit_status",
    "render_latest_audit_summary",
    "run_enabled_audits",
]
