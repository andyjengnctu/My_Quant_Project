"""Project-wide single inventory for Audit implementations and CLI exposure."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable

from config.audit import AuditDefinition

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
    method_id: str | None = None
    status_function: str | None = None
    run_function: str | None = None
    cli_command: str | None = None
    preparation_function: str | None = None

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

    def load_preparation_handler(self) -> RunHandler:
        if not self.preparation_function:
            raise ValueError(f"Audit沒有canonical source preparer: {self.audit_type}")
        handler = getattr(import_module(self.module), self.preparation_function)
        if not callable(handler):
            raise TypeError(
                f"Audit source preparer不可呼叫: {self.module}.{self.preparation_function}"
            )
        return handler


@dataclass(frozen=True)
class AuditMethodEntry:
    method_id: str
    menu_label: str
    description: str
    order: int


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
}

def get_audit_methods() -> tuple[AuditMethodEntry, ...]:
    return tuple(sorted(AUDIT_METHOD_CATALOG.values(), key=lambda item: item.order))

def get_audit_method(method_id: str) -> AuditMethodEntry:
    key = str(method_id).strip()
    method = AUDIT_METHOD_CATALOG.get(key)
    if method is None:
        raise ValueError(f"Audit method未註冊: {method_id}")
    return method


AUDIT_CATALOG: dict[str, AuditCatalogEntry] = {
    "mr13ab_survival_increment": AuditCatalogEntry(
        audit_type="mr13ab_survival_increment",
        domain="breakout_quality",
        module="services.audit.mr13ab_survival_increment",
        mode="formal",
        description="只讀MR-13AB/MR-13K frozen OOS changed-row與conflict-pair survival increment",
        read_only=True,
        method_id="opportunity_selection_attribution",
        status_function="collect_status",
        run_function="run_formal_audit",
        preparation_function="prepare_reference_frozen_scores",
    ),
    # Only genuinely supported diagnostic commands belong here.  Canonical
    # Dataset/Target/PIT builders live in services/ and are not research Audits.
    "regime": AuditCatalogEntry(
        audit_type="regime",
        domain="breakout_quality",
        module="tools.audit.breakout_quality.regime",
        mode="research",
        description="稽核Selection／OOS市場狀態與breakout event覆蓋",
        read_only=True,
        cli_command="regime-audit",
    ),
    "point_in_time_scores": AuditCatalogEntry(
        audit_type="point_in_time_scores",
        domain="breakout_quality",
        module="services.breakout_quality.point_in_time_audit",
        mode="research",
        description="驗證point-in-time Score的Target排序能力與fold穩定性",
        read_only=True,
        cli_command="audit-point-in-time-scores",
    ),
    "selection_resource_constraint_attribution": AuditCatalogEntry(
        audit_type="selection_resource_constraint_attribution",
        domain="breakout_quality",
        module="services.audit.selection_resource_constraints",
        mode="formal",
        description="只讀C58-derived K/R0 resource conversion與MFE/Safety stage attribution",
        read_only=True,
        method_id="opportunity_selection_attribution",
        status_function="collect_status",
        run_function="run_formal_audit",
    ),
    "mr13r_joint_capital_drawdown": AuditCatalogEntry(
        audit_type="mr13r_joint_capital_drawdown",
        domain="breakout_quality",
        module="services.audit.mr13r_joint_capital_drawdown",
        mode="formal",
        description="只讀MR-13R joint signal、Safety→capital conversion與portfolio drawdown attribution",
        read_only=True,
        method_id="portfolio_drawdown_attribution",
        status_function="collect_status",
        run_function="run_formal_audit",
    ),
    "c69_marginal_position_attribution": AuditCatalogEntry(
        audit_type="c69_marginal_position_attribution",
        domain="breakout_quality",
        module="services.audit.c69_marginal_positions",
        mode="formal",
        description="只讀C68/C69 matched marginal positions、truth geometry與path conversion attribution",
        read_only=True,
        method_id="opportunity_selection_attribution",
        status_function="collect_status",
        run_function="run_formal_audit",
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
    for key, entry in AUDIT_CATALOG.items():
        if key != entry.audit_type:
            raise ValueError(f"Audit catalog key與audit_type不一致: {key} != {entry.audit_type}")
        if not entry.domain or not entry.module or not entry.mode or not entry.description:
            raise ValueError(f"Audit catalog欄位不可空白: {key}")
        if entry.formal and (not entry.status_function or not entry.run_function or not entry.read_only):
            raise ValueError(f"Formal Audit catalog contract不完整: {key}")
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
