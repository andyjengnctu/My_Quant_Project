"""Project-wide single inventory for Audit implementations and CLI exposure."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable

from core.audit_policy import AuditDefinition

StatusHandler = Callable[..., dict[str, Any]]
RunHandler = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class AuditCatalogEntry:
    audit_type: str
    domain: str
    module: str
    mode: str
    description: str
    read_only: bool
    # Formal Audits have NO upstream artifact producer/optimizer capability.
    # READ-ONLY status is REUSE or BLOCKED, never automatic BUILD/RESUME.
    method_id: str | None = None
    status_function: str | None = None
    run_function: str | None = None
    cli_command: str | None = None
    preflight_summary: bool = False
    # Optional canonical read-only preview used before costly source checks.
    inventory_function: str | None = None

    @property
    def formal(self) -> bool:
        return self.mode == "formal"

    def load_status_handler(self) -> StatusHandler:
        if not self.formal or not self.status_function:
            raise ValueError(f"Audit不是config-driven formal handler: {self.audit_type}")
        handler = getattr(import_module(self.module), self.status_function)
        if not callable(handler):
            raise TypeError(f"Audit status handler不可呼叫: {self.module}.{self.status_function}")
        return handler

    def load_run_handler(self) -> RunHandler:
        if not self.formal or not self.run_function:
            raise ValueError(f"Audit不是config-driven formal handler: {self.audit_type}")
        handler = getattr(import_module(self.module), self.run_function)
        if not callable(handler):
            raise TypeError(f"Audit run handler不可呼叫: {self.module}.{self.run_function}")
        return handler

    def load_inventory_handler(self) -> Callable[..., str]:
        if not self.formal or not self.preflight_summary or not self.inventory_function:
            raise ValueError(f"Audit沒有正式快速工件盤點能力: {self.audit_type}")
        handler = getattr(import_module(self.module), self.inventory_function, None)
        if not callable(handler):
            raise TypeError(f"Audit inventory handler不可呼叫: {self.module}.{self.inventory_function}")
        return handler

@dataclass(frozen=True)
class AuditMethodEntry:
    method_id: str
    menu_label: str
    description: str
    order: int
    one_time_only: bool = False
    menu_location: str = "audit"


AUDIT_METHOD_CATALOG: dict[str, AuditMethodEntry] = {
    "opportunity_selection_attribution": AuditMethodEntry(
        method_id="opportunity_selection_attribution",
        menu_label="Opportunity／Selection Attribution",
        description="比較Daily→Breakout→Orderable→Planned的actual opportunity與selection轉化",
        order=10,
    ),
    "trade_outcome_path_attribution": AuditMethodEntry(
        method_id="trade_outcome_path_attribution",
        menu_label="Trade Outcome／Path Attribution",
        description="比較Planned→Filled→Realized的MFE/adverse/first-passage/path conversion",
        order=20,
    ),
    "portfolio_drawdown_attribution": AuditMethodEntry(
        method_id="portfolio_drawdown_attribution",
        menu_label="Portfolio／Drawdown Attribution",
        description="比較Realized trades→portfolio capital/exposure與exact MTM drawdown attribution",
        order=30,
    ),
    "target_ground_truth_strategy_romd": AuditMethodEntry(
        method_id="target_ground_truth_strategy_romd",
        menu_label="Target 驗證",
        description="Reusable hindsight-only Oracle RoMD comparison (never PIT or Trading)",
        order=100,
        menu_location="research_root",
    ),
}

def get_audit_methods(*, menu_location: str | None = None) -> tuple[AuditMethodEntry, ...]:
    """One registry; each formal work type has exactly one interactive menu home."""
    return tuple(sorted(
        (m for m in AUDIT_METHOD_CATALOG.values()
         if not m.one_time_only and (menu_location is None or m.menu_location == menu_location)),
        key=lambda item: item.order,
    ))

def get_audit_method(method_id: str) -> AuditMethodEntry:
    key = str(method_id).strip()
    method = AUDIT_METHOD_CATALOG.get(key)
    if method is None:
        raise ValueError(f"Audit method未註冊: {method_id}")
    return method


AUDIT_CATALOG: dict[str, AuditCatalogEntry] = {
    # Only genuinely supported diagnostic commands belong here.  Canonical
    # Dataset/Target/PIT builders live in services/ and are not research Audits.
    "point_in_time_scores": AuditCatalogEntry(
        audit_type="point_in_time_scores",
        domain="breakout_quality",
        module="services.breakout_quality.point_in_time_audit",
        mode="research",
        description="驗證point-in-time Score的Target排序能力與fold穩定性",
        read_only=True,
        cli_command="audit-point-in-time-scores",
    ),
    "opportunity_selection_attribution": AuditCatalogEntry(
        audit_type="opportunity_selection_attribution",
        domain="breakout_quality",
        module="services.audit.opportunity_selection",
        mode="formal",
        description="Reusable Daily/Breakout opportunity → final planned selection attribution",
        read_only=True,
        method_id="opportunity_selection_attribution",
        status_function="collect_status",
        run_function="run_formal_audit",
    ),
    "trade_outcome_path_attribution": AuditCatalogEntry(
        audit_type="trade_outcome_path_attribution",
        domain="breakout_quality",
        module="services.audit.trade_outcome_path",
        mode="formal",
        description="Reusable planned→filled→realized trade outcome/path attribution",
        read_only=True,
        method_id="trade_outcome_path_attribution",
        status_function="collect_status",
        run_function="run_formal_audit",
    ),
    "target_ground_truth_strategy_romd": AuditCatalogEntry(
        audit_type="target_ground_truth_strategy_romd",
        domain="breakout_quality",
        module="services.audit.target_oracle_romd",
        mode="formal",
        description="Reusable hindsight Oracle Ground Truth -> Broker-Cash RoMD (NON-PIT)",
        read_only=True, method_id="target_ground_truth_strategy_romd",
        status_function="collect_status", run_function="run_formal_audit",
        preflight_summary=True, inventory_function="render_existing_artifact_inventory",
    ),
    "portfolio_drawdown_attribution": AuditCatalogEntry(
        audit_type="portfolio_drawdown_attribution",
        domain="breakout_quality",
        module="services.audit.portfolio_drawdown",
        mode="formal",
        description="Reusable capital/exposure + exact peak→trough MTM portfolio attribution",
        read_only=True,
        method_id="portfolio_drawdown_attribution",
        status_function="collect_status",
        run_function="run_formal_audit",
    ),
}



def get_audit_entry(audit_type: str) -> AuditCatalogEntry:
    entry = AUDIT_CATALOG.get(str(audit_type))
    if entry is None:
        raise ValueError(f"Audit catalog未註冊audit_type: {audit_type}")
    return entry


def get_audit_handler(definition: AuditDefinition) -> AuditCatalogEntry:
    entry = get_audit_entry(definition.audit_type)
    if entry.domain != str(definition.module_id):
        raise ValueError(
            "Audit catalog domain與config不一致: "
            f"audit={definition.audit_id}, config={definition.module_id}, catalog={entry.domain}"
        )
    if not entry.formal:
        raise ValueError(f"config/audit.py只能使用formal Audit: {definition.audit_id}")
    if not entry.read_only:
        raise ValueError(f"Formal Audit handler必須read-only: {definition.audit_id}")
    return entry


def get_definition_method_id(definition: AuditDefinition) -> str:
    entry = get_audit_handler(definition)
    if not entry.method_id:
        raise ValueError(f"Formal Audit缺少method_id: {definition.audit_id}")
    return str(entry.method_id)


def get_domain_cli_commands(domain: str) -> dict[str, AuditCatalogEntry]:
    domain_key = str(domain).strip()
    entries = {
        str(entry.cli_command): entry
        for entry in AUDIT_CATALOG.values()
        if entry.domain == domain_key and entry.cli_command
    }
    if len(entries) != sum(
        1
        for entry in AUDIT_CATALOG.values()
        if entry.domain == domain_key and entry.cli_command
    ):
        raise ValueError(f"Audit CLI command重複: domain={domain_key}")
    return entries


def validate_audit_catalog(definitions: tuple[AuditDefinition, ...] = ()) -> None:
    seen_modules: set[tuple[str, str]] = set()
    seen_commands: set[tuple[str, str]] = set()
    for method in AUDIT_METHOD_CATALOG.values():
        if method.menu_location not in {"audit", "research_root"}:
            raise ValueError(f"Audit method menu_location不合法: {method.method_id}")
    for key, entry in AUDIT_CATALOG.items():
        if key != entry.audit_type:
            raise ValueError(f"Audit catalog key與audit_type不一致: {key} != {entry.audit_type}")
        if not entry.domain or not entry.module or not entry.mode or not entry.description:
            raise ValueError(f"Audit catalog欄位不可空白: {key}")
        if entry.formal and (not entry.status_function or not entry.run_function or not entry.read_only):
            raise ValueError(f"Formal Audit catalog contract不完整: {key}")
        if entry.preflight_summary and (not entry.formal or not entry.method_id):
            raise ValueError(f"簡要Replay狀態只能宣告於正式Audit: {key}")
        if entry.inventory_function and (not entry.formal or not entry.preflight_summary):
            raise ValueError(f"快速工件盤點必須由正式Replay能力宣告: {key}")
        if entry.preflight_summary and not entry.inventory_function:
            raise ValueError(f"正式Replay前置確認缺少快速工件盤點: {key}")
        if entry.formal:
            if not entry.method_id:
                raise ValueError(f"Formal Audit必須綁定method_id: {key}")
            get_audit_method(entry.method_id)
        identity = (entry.domain, entry.module)
        if identity in seen_modules:
            raise ValueError(f"Audit module重複登記: {entry.module}")
        seen_modules.add(identity)
        if entry.cli_command:
            command_identity = (entry.domain, entry.cli_command)
            if command_identity in seen_commands:
                raise ValueError(f"Audit CLI command重複: {entry.cli_command}")
            seen_commands.add(command_identity)
    seen_ids: set[str] = set()
    for definition in definitions:
        if definition.audit_id in seen_ids:
            raise ValueError(f"Audit ID重複: {definition.audit_id}")
        seen_ids.add(definition.audit_id)
        get_audit_handler(definition)


validate_audit_catalog()

__all__ = [
    "AUDIT_CATALOG",
    "AUDIT_METHOD_CATALOG",
    "AuditCatalogEntry",
    "AuditMethodEntry",
    "get_audit_entry",
    "get_audit_handler",
    "get_audit_method",
    "get_audit_methods",
    "get_definition_method_id",
    "get_domain_cli_commands",
    "validate_audit_catalog",
]
