"""Project-wide config-driven formal Audit runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config.audit import (
    AUDIT_OUTPUT_ROOT,
    get_audit_definitions,
    get_audit_module_ids,
)
from config.research import get_research_artifact_preparation_policy
from core.research_orchestration import (
    ResearchArtifactAction,
    ResearchArtifactPlan,
    resolve_research_artifact_action,
)
from services.research.artifact_orchestrator import run_research_artifact_preparation
from core.console_report import (
    project_relative_display_path,
    render_key_values,
    render_table,
    render_title,
)
from services.audit.catalog import (
    get_audit_handler,
    get_audit_method,
    get_definition_method_id,
    validate_audit_catalog,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _module_title(module_id: str) -> str:
    return str(module_id).replace("_", " ").title()


def _definitions_for_method(
    module_id: str,
    method_id: str | None,
    audit_id: str | None = None,
):
    definitions = get_audit_definitions(module_id)
    if method_id is not None:
        method_key = get_audit_method(method_id).method_id
        definitions = tuple(
            definition
            for definition in definitions
            if get_definition_method_id(definition) == method_key
        )
    if audit_id is not None:
        audit_key = str(audit_id).strip()
        definitions = tuple(
            definition for definition in definitions if definition.audit_id == audit_key
        )
        if not definitions:
            raise ValueError(f"Audit ID不存在於目前scope: {module_id}/{audit_key}")
    return definitions


def collect_audit_menu_state(
    module_id: str,
    *,
    method_id: str | None = None,
    audit_id: str | None = None,
) -> dict[str, Any]:
    """Return config/catalog-only state for interactive menu rendering.

    Menu navigation must stay lightweight.  Full status handlers intentionally run
    deep read-only preflight checks over Audit artifacts, so invoking them merely to
    draw a menu causes large repeated I/O and DataFrame work.
    """

    definitions = _definitions_for_method(module_id, method_id, audit_id)
    rows = [
        {
            "enabled": bool(definition.enabled),
            "audit_id": definition.audit_id,
            "method_id": get_definition_method_id(definition),
        }
        for definition in definitions
    ]
    enabled_count = sum(1 for row in rows if row["enabled"])
    return {
        "module_id": module_id,
        "method_id": method_id,
        "audit_id": audit_id,
        "configured": bool(enabled_count),
        "enabled_count": enabled_count,
        "rows": rows,
    }


def collect_audit_status(
    module_id: str,
    *,
    project_root: Path = PROJECT_ROOT,
    method_id: str | None = None,
    audit_id: str | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    definitions = _definitions_for_method(module_id, method_id, audit_id)
    validate_audit_catalog(get_audit_definitions(module_id))
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
                "method_id": get_definition_method_id(definition),
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
        "method_id": method_id,
        "audit_id": audit_id,
        "overall_status": overall,
        "rows": rows,
        "statuses": statuses,
    }


def collect_audit_preparation_plan(
    module_id: str,
    *,
    project_root: Path = PROJECT_ROOT,
    method_id: str | None = None,
    audit_id: str | None = None,
    status_snapshot: dict[str, Any] | None = None,
) -> ResearchArtifactPlan:
    """Map enabled read-only Audit sources to the shared Research dependency contract.

    Audit implementations remain read-only.  If an Audit catalog entry declares a
    canonical source preparer, orchestration may invoke that producer before the Audit;
    otherwise a missing source is a true BLOCKED research dependency.
    """

    root = Path(project_root).resolve()
    status = (
        collect_audit_status(
            module_id,
            project_root=root,
            method_id=method_id,
            audit_id=audit_id,
        )
        if status_snapshot is None
        else status_snapshot
    )
    if (
        str(status.get("module_id") or "") != str(module_id)
        or status.get("method_id") != method_id
        or status.get("audit_id") != audit_id
    ):
        raise ValueError("Audit status snapshot與requested module/method不一致")
    definitions = {
        item.audit_id: item
        for item in _definitions_for_method(module_id, method_id, audit_id)
        if item.enabled
    }
    actions: list[ResearchArtifactAction] = []
    policy = get_research_artifact_preparation_policy()
    for row in status["rows"]:
        if not row["enabled"]:
            continue
        definition = definitions[str(row["audit_id"])]
        spec = get_audit_handler(definition)
        ready = str(row.get("status") or "").upper() == "READY"
        has_preparer = bool(spec.preparation_function)
        source = dict(row.get("source") or {})
        raw_path = source.get("path")
        source_path = None if raw_path in (None, "") else Path(str(raw_path))
        if source_path is not None and not source_path.is_absolute():
            source_path = root / source_path
        path_exists = bool(source_path is not None and source_path.exists())
        action = resolve_research_artifact_action(
            ready=ready,
            artifact_exists=path_exists,
            has_builder=has_preparer,
            auto_prepare=bool(policy.auto_prepare),
            reuse_ready_artifacts=bool(policy.reuse_ready_artifacts),
            rebuild_stale_artifacts=bool(policy.rebuild_stale_artifacts),
            resume_partial_artifacts=bool(policy.resume_partial_artifacts),
            resumable=False,
        )
        path = str(source.get("path") or source.get("display") or "-")
        actions.append(
            ResearchArtifactAction(
                action_id=f"audit-source:{definition.audit_id}",
                artifact_key=f"audit-source:{definition.audit_id}",
                action=action,
                builder_type=(
                    None if ready else f"audit_source:{definition.audit_type}" if has_preparer else None
                ),
                description=(
                    "重用Audit只讀來源工件"
                    if ready
                    else str(row.get("reason") or "Audit來源工件尚未就緒")
                ),
                path=path,
                dependencies=(),
                producer_work_type=(
                    "existing_artifact" if ready else "audit_source_provider" if has_preparer else None
                ),
                execution_priority=10,
            )
        )
    return ResearchArtifactPlan.from_actions(actions)


def render_audit_status(
    module_id: str,
    *,
    project_root: Path = PROJECT_ROOT,
    method_id: str | None = None,
    audit_id: str | None = None,
) -> str:
    status = collect_audit_status(
        module_id,
        project_root=project_root,
        method_id=method_id,
        audit_id=audit_id,
    )
    preparation_plan = collect_audit_preparation_plan(
        module_id,
        project_root=project_root,
        method_id=method_id,
        audit_id=audit_id,
        status_snapshot=status,
    )
    action_by_audit_id = {
        item.artifact_key.split(":", 1)[1]: item
        for item in preparation_plan.actions
        if item.artifact_key.startswith("audit-source:")
    }
    rows = []
    for row in status["rows"]:
        source = row["source"]
        source_text = str(
            source.get("display")
            or source.get("arm_id")
            or source.get("candidate_arm_id")
            or "-"
        )
        planned = action_by_audit_id.get(str(row["audit_id"]))
        action = (
            "SKIP"
            if not row["enabled"]
            else "READ"
            if planned is not None and planned.action == "REUSE"
            else planned.action
            if planned is not None
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
            render_title(
                f"{audit_id} 設定與工件狀態"
                if audit_id is not None
                else f"{_module_title(module_id)} Audit 設定與工件狀態"
                if method_id is None
                else f"{get_audit_method(method_id).menu_label} 設定與工件狀態"
            ),
            render_key_values(
                (
                    ("設定檔", "config/audit.py"),
                    ("Catalog", "services/audit/catalog.py"),
                    ("Module", module_id),
                    ("整體狀態", preparation_plan.overall_status if preparation_plan.actions else status["overall_status"]),
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
    method_id: str | None = None,
    audit_id: str | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    definitions = tuple(
        item
        for item in _definitions_for_method(module_id, method_id, audit_id)
        if item.enabled
    )
    if not definitions:
        raise RuntimeError(f"config/audit.py沒有啟用任何{module_id} Audit")
    validate_audit_catalog(get_audit_definitions(module_id))
    status = collect_audit_status(
        module_id,
        project_root=root,
        method_id=method_id,
        audit_id=audit_id,
    )
    if status["overall_status"] != "READY":
        policy = get_research_artifact_preparation_policy()
        if not policy.auto_prepare:
            raise RuntimeError("目前Research artifact auto_prepare已關閉，Audit來源尚未就緒")
        definitions_by_id = {item.audit_id: item for item in definitions}

        def _refresh_plan() -> ResearchArtifactPlan:
            return collect_audit_preparation_plan(
                module_id,
                project_root=root,
                method_id=method_id,
                audit_id=audit_id,
            )

        def _execute(action: ResearchArtifactAction) -> None:
            audit_id = action.artifact_key.split(":", 1)[1]
            definition = definitions_by_id[audit_id]
            spec = get_audit_handler(definition)
            result = spec.load_preparation_handler()(definition, project_root=root)
            if isinstance(result, int) and int(result) != 0:
                raise RuntimeError(f"Audit source preparer失敗: returncode={result}")

        run_research_artifact_preparation(
            plan_refresher=_refresh_plan,
            action_executor=_execute,
            failure_prefix="Audit前置",
        )
        status = collect_audit_status(
            module_id,
            project_root=root,
            method_id=method_id,
            audit_id=audit_id,
        )
        if status["overall_status"] != "READY":
            raise RuntimeError("Audit canonical source preparer完成後狀態仍非READY")
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
    method_id: str | None = None,
    audit_id: str | None = None,
) -> str:
    root = Path(project_root).resolve()
    rows = []
    for definition in _definitions_for_method(module_id, method_id, audit_id):
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
            render_title(
                f"{audit_id} 最近結果"
                if audit_id is not None
                else f"{_module_title(module_id)} 最近 Audit 結果"
                if method_id is None
                else f"{get_audit_method(method_id).menu_label} 最近結果"
            ),
            render_table(("Audit", "狀態", "Markdown"), rows),
        )
    )


def collect_project_audit_status(*, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    return {
        module_id: collect_audit_status(module_id, project_root=project_root)
        for module_id in get_audit_module_ids(enabled_only=True)
    }


__all__ = [
    "collect_audit_menu_state",
    "collect_audit_preparation_plan",
    "collect_audit_status",
    "collect_project_audit_status",
    "render_audit_status",
    "render_latest_audit_summary",
    "run_enabled_audits",
]
