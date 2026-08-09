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


AUDIT_CATALOG: dict[str, AuditCatalogEntry] = {
    # Config-driven formal Audit handlers.
    "pass_quality": AuditCatalogEntry(
        audit_type="pass_quality",
        domain="breakout_quality",
        module="tools.audit.breakout_quality.pass_quality",
        mode="formal",
        description="A9 PASS內部品質與實際Realized R診斷",
        read_only=True,
        status_function="collect_pass_quality_status",
        run_function="run_pass_quality_audit",
    ),
    "pass_persistence": AuditCatalogEntry(
        audit_type="pass_persistence",
        domain="breakout_quality",
        module="tools.audit.breakout_quality.pass_persistence",
        mode="formal",
        description="A9 PASS persistence與false-positive重複權重診斷",
        read_only=True,
        status_function="collect_pass_persistence_status",
        run_function="run_pass_persistence_audit",
    ),
    "selection_confidence": AuditCatalogEntry(
        audit_type="selection_confidence",
        domain="breakout_quality",
        module="tools.audit.breakout_quality.selection_confidence",
        mode="formal",
        description="A9多PASS競爭日confidence排序力診斷",
        read_only=True,
        status_function="collect_selection_confidence_status",
        run_function="run_selection_confidence_audit",
    ),
    "strategy_attribution": AuditCatalogEntry(
        audit_type="strategy_attribution",
        domain="breakout_quality",
        module="tools.audit.breakout_quality.c15_strategy_attribution",
        mode="formal",
        description="跨arm wealth-path、selection、capital geometry與slot occupancy歸因",
        read_only=True,
        status_function="collect_strategy_attribution_status",
        run_function="run_strategy_attribution_audit",
    ),
    "strategy_realization_capture": AuditCatalogEntry(
        audit_type="strategy_realization_capture",
        domain="breakout_quality",
        module="tools.audit.breakout_quality.strategy_realization_capture",
        mode="formal",
        description="既有score-ranking replay的exclusive trade、資金幾何、slot occupancy與Target→Realized capture整合歸因",
        read_only=True,
        status_function="collect_strategy_realization_capture_status",
        run_function="run_strategy_realization_capture_audit",
    ),
    "pit_fold_runtime_attribution": AuditCatalogEntry(
        audit_type="pit_fold_runtime_attribution",
        domain="breakout_quality",
        module="tools.audit.breakout_quality.pit_fold_runtime_attribution",
        mode="formal",
        description="Selection PIT orderable pool的cross-fold score mixing、fold transition與exclusive winner capture歸因",
        read_only=True,
        status_function="collect_pit_fold_runtime_attribution_status",
        run_function="run_pit_fold_runtime_attribution_audit",
    ),
    "pit_target_realization_attribution": AuditCatalogEntry(
        audit_type="pit_target_realization_attribution",
        domain="breakout_quality",
        module="tools.audit.breakout_quality.pit_target_realization_attribution",
        mode="formal",
        description="Selection PIT exclusive trades的Target／Score對realized R與signal→entry age歸因",
        read_only=True,
        status_function="collect_pit_target_realization_attribution_status",
        run_function="run_pit_target_realization_attribution_audit",
    ),
    # Research / historical Audit commands. They share this inventory but are not
    # eligible for the formal config runner unless promoted to mode=formal later.
    "regime": AuditCatalogEntry(
        audit_type="regime",
        domain="breakout_quality",
        module="tools.audit.breakout_quality.regime",
        mode="research",
        description="稽核Selection／OOS市場狀態與breakout event覆蓋",
        read_only=True,
        cli_command="regime-audit",
    ),
    "continuous_target": AuditCatalogEntry(
        audit_type="continuous_target",
        domain="breakout_quality",
        module="tools.audit.breakout_quality.continuous_target",
        mode="research",
        description="建立11A連續target arrays並稽核分布、同日排序與實際R方向",
        read_only=False,
        cli_command="audit-continuous-target",
    ),
    "qualified_candidate_set": AuditCatalogEntry(
        audit_type="qualified_candidate_set",
        domain="breakout_quality",
        module="tools.audit.breakout_quality.qualified_candidate_set",
        mode="research",
        description="執行11C策略qualified candidate-set失敗歸因；research-only",
        read_only=True,
        cli_command="audit-qualified-candidate-set",
    ),
    "target_component_attribution": AuditCatalogEntry(
        audit_type="target_component_attribution",
        domain="breakout_quality",
        module="tools.audit.breakout_quality.target_component_attribution",
        mode="research",
        description="執行11D Target成分與Label條件失敗歸因；research-only",
        read_only=True,
        cli_command="audit-target-attribution",
    ),
    "target_time_penalty_ablation": AuditCatalogEntry(
        audit_type="target_time_penalty_ablation",
        domain="breakout_quality",
        module="tools.audit.breakout_quality.target_time_penalty_ablation",
        mode="research",
        description="執行11E固定移除time penalty的Target稽核；research-only",
        read_only=True,
        cli_command="audit-target-time-ablation",
    ),
    "no_time_continuous_target": AuditCatalogEntry(
        audit_type="no_time_continuous_target",
        domain="breakout_quality",
        module="tools.audit.breakout_quality.no_time_continuous_target",
        mode="research",
        description="建立11F No-time Target arrays並做Selection-only可學性稽核；research-only",
        read_only=False,
        cli_command="audit-no-time-target",
    ),
    "pass_realization_gap": AuditCatalogEntry(
        audit_type="pass_realization_gap",
        domain="breakout_quality",
        module="tools.audit.breakout_quality.pass_realization_gap",
        mode="research",
        description="執行11H PASS-only實現落差歸因；research-only、CLI-only",
        read_only=True,
        cli_command="audit-pass-realization-gap",
    ),
    "selection_strategy_realization": AuditCatalogEntry(
        audit_type="selection_strategy_realization",
        domain="breakout_quality",
        module="tools.audit.breakout_quality.selection_strategy_realization",
        mode="research",
        description="執行11I Selection nested-OOS策略實現覆蓋稽核；research-only、CLI-only",
        read_only=False,
        cli_command="audit-selection-strategy-realization",
    ),
    "candidate_counterfactual_execution": AuditCatalogEntry(
        audit_type="candidate_counterfactual_execution",
        domain="breakout_quality",
        module="tools.audit.breakout_quality.candidate_counterfactual_execution",
        mode="historical",
        description="執行11J per-candidate counterfactual execution稽核；已停止、僅供歷史追溯",
        read_only=False,
        cli_command="audit-candidate-counterfactual",
    ),
    "portfolio_selection_pressure": AuditCatalogEntry(
        audit_type="portfolio_selection_pressure",
        domain="breakout_quality",
        module="tools.audit.breakout_quality.portfolio_selection_pressure",
        mode="research",
        description="執行11K portfolio selection-pressure歸因；read-only、CLI-only",
        read_only=True,
        cli_command="audit-selection-pressure",
    ),
    "point_in_time_scores": AuditCatalogEntry(
        audit_type="point_in_time_scores",
        domain="breakout_quality",
        module="tools.audit.breakout_quality.point_in_time_scores",
        mode="research",
        description="驗證point-in-time Score的Target排序能力與fold穩定性",
        read_only=True,
        cli_command="audit-point-in-time-scores",
    ),
    "score_ranking_capture": AuditCatalogEntry(
        audit_type="score_ranking_capture",
        domain="portfolio",
        module="tools.audit.portfolio.score_ranking_capture",
        mode="library",
        description="既有score-ranking replay的資本效率與Target capture attribution primitives",
        read_only=True,
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
