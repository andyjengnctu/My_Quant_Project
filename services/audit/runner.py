"""Project-wide config-driven formal Audit runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from config.audit import AUDIT_OUTPUT_ROOT
from core.audit_policy import (
    get_audit_definitions,
    get_audit_module_ids,
    get_reusable_audit_definitions,
)
from core.research_orchestration import (
    ResearchArtifactAction,
    ResearchArtifactPlan,
    resolve_research_artifact_action,
)
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
    if method_id is not None and audit_id is None:
        method_key = get_audit_method(method_id).method_id
        definitions = tuple(
            definition
            for definition in get_reusable_audit_definitions(module_id)
            if get_definition_method_id(definition) == method_key
        )
    else:
        definitions = get_audit_definitions(module_id)
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
    validate_audit_catalog(
        (*get_audit_definitions(module_id), *get_reusable_audit_definitions(module_id))
    )
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
    """Plan read-only Audit consumption using the shared Research action vocabulary.

    Unlike model/parameter training workflows, an Audit has no authorization
    to BUILD, REBUILD or RESUME its upstream sources. Missing or incompatible
    upstream evidence must be BLOCKED, never silently passed to an Optimizer.
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
    for row in status["rows"]:
        if not row["enabled"]:
            continue
        definition = definitions[str(row["audit_id"])]
        ready = str(row.get("status") or "").upper() == "READY"
        source = dict(row.get("source") or {})
        raw_path = source.get("path")
        source_path = None if raw_path in (None, "") else Path(str(raw_path))
        if source_path is not None and not source_path.is_absolute():
            source_path = root / source_path
        path_exists = bool(source_path is not None and source_path.exists())
        action = resolve_research_artifact_action(
            ready=ready,
            artifact_exists=path_exists,
            has_builder=False,
            auto_prepare=False,
            reuse_ready_artifacts=True,
            rebuild_stale_artifacts=False,
            resume_partial_artifacts=False,
            resumable=False,
        )
        path = str(source.get("path") or source.get("display") or "-")
        actions.append(
            ResearchArtifactAction(
                action_id=f"audit-source:{definition.audit_id}",
                artifact_key=f"audit-source:{definition.audit_id}",
                action=action,
                builder_type=None,
                description=(
                    "重用Audit只讀來源工件"
                    if ready
                    else str(row.get("reason") or "Audit來源工件尚未就緒")
                ),
                path=path,
                dependencies=(),
                producer_work_type="existing_artifact" if ready else None,
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


def render_audit_artifact_inventory(
    module_id: str,
    *,
    project_root: Path = PROJECT_ROOT,
    method_id: str | None = None,
    audit_id: str | None = None,
) -> str:
    """Lightweight cache-presence preview; never computes Target or grants REUSE.

    The catalog advertises the single canonical inventory provider. Full
    identity checks and Replay authorization remain with that Audit owner.
    """
    definitions = tuple(
        item for item in _definitions_for_method(module_id, method_id, audit_id)
        if item.enabled
    )
    if len(definitions) != 1:
        raise ValueError("快速 Replay 工件盤點只支援單一已啟用的正式 Audit")
    definition = definitions[0]
    handler = get_audit_handler(definition).load_inventory_handler()
    return str(handler(definition, project_root=Path(project_root).resolve()))


def run_enabled_audits(
    module_id: str,
    *,
    project_root: Path = PROJECT_ROOT,
    quiet: bool = False,
    method_id: str | None = None,
    audit_id: str | None = None,
    before_replay: Callable[[str], None] | None = None,
    preflight_display: str = "full",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    definitions = tuple(
        item
        for item in _definitions_for_method(module_id, method_id, audit_id)
        if item.enabled
    )
    if not definitions:
        raise RuntimeError(f"config/audit.py沒有啟用任何{module_id} Audit")
    if preflight_display not in {"full", "summary"}:
        raise ValueError("未知 Replay preflight 顯示方式")
    if preflight_display != "full" and (
        len(definitions) != 1
        or not get_audit_handler(definitions[0]).preflight_summary
    ):
        raise ValueError("簡要 Replay preflight 僅供正式 Target 驗證使用")
    if before_replay is not None and (
        len(definitions) != 1
        or not get_audit_handler(definitions[0]).preflight_summary
    ):
        raise ValueError("執行前 Replay 確認只允許正式 Target 驗證的唯一 owner")
    validate_audit_catalog(
        (*get_audit_definitions(module_id), *get_reusable_audit_definitions(module_id))
    )
    status = collect_audit_status(
        module_id,
        project_root=root,
        method_id=method_id,
        audit_id=audit_id,
    )
    if status["overall_status"] != "READY":
        # The runner must have no callable producer path. A Target audit is
        # authorized to compute its own hindsight Replay, not fit missing
        # historical parameter truth or repair upstream model artifacts.
        blocked = [
            f"{row['audit_id']}: {row.get('reason') or 'canonical source not READY'}"
            for row in status["rows"] if row["enabled"] and row["status"] != "READY"
        ]
        raise RuntimeError(
            "Audit 來源 BLOCKED（只讀驗證，不自動訓練／最佳化）："
            + " | ".join(blocked)
        )
    outputs: dict[str, Any] = {}
    for definition in definitions:
        spec = get_audit_handler(definition)
        kwargs: dict[str, Any] = {"project_root": root, "quiet": quiet}
        if preflight_display != "full":
            kwargs["preflight_display"] = preflight_display
        if before_replay is not None:
            kwargs["before_replay"] = before_replay
        outputs[definition.audit_id] = spec.load_run_handler()(definition, **kwargs)
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
    # The history page shows both retained one-time studies and reusable reports.
    # The menu/run scopes stay separate; no second execution route is created.
    definitions = (
        (*get_audit_definitions(module_id), *get_reusable_audit_definitions(module_id))
        if method_id is None and audit_id is None
        else _definitions_for_method(module_id, method_id, audit_id)
    )
    for definition in definitions:
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
    "render_audit_artifact_inventory",
    "render_latest_audit_summary",
    "run_enabled_audits",
]
