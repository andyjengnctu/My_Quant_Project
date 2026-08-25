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


AUDIT_CATALOG: dict[str, AuditCatalogEntry] = {
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
    "continuous_truth_strategy_quadrants": AuditCatalogEntry(
        audit_type="continuous_truth_strategy_quadrants",
        domain="breakout_quality",
        module="services.audit.mfe_safety_quadrants",
        mode="formal",
        description="只讀MFE × Safety truth與Strategy Compare row-level evidence的四象限Audit",
        read_only=True,
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
    "AuditCatalogEntry",
    "get_audit_entry",
    "get_audit_handler",
    "get_domain_cli_commands",
    "validate_audit_catalog",
]
