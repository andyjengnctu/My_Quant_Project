"""Canonical Market Data governance contract.

This owner separates stable provider/price semantics from declarative lifecycle
state in ``config.market_data``.  It intentionally does not perform network I/O,
storage, PIT normalization, universe construction, or downloader orchestration.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Mapping

from config.market_data import (
    ACTIVE_RESEARCH_DATA_GENERATION,
    RESEARCH_DATA_GENERATIONS,
    TRADING_MARKET_DATA_LIFECYCLE,
    MARKET_DATA_V2_LIFECYCLE,
)

MARKET_DATA_PROVIDER_ID = "finmind"
FINMIND_ADJUSTED_PRICE_DATASET = "TaiwanStockPriceAdj"
FINMIND_RAW_PRICE_ARCHIVE_DATASET = "TaiwanStockPrice"

# Current project price semantics: FinMind owns adjusted-price calculation.
# Raw price may be archived, but is not an authorized current model/strategy
# price input and the project must not maintain a second adjustment engine.
RAW_PRICE_DIRECT_CONSUMPTION_ENABLED = False
PROJECT_ADJUSTED_PRICE_ENGINE_ENABLED = False
ADJUSTED_PRICE_RETROSPECTIVE_INVARIANCE_REQUIRED = True
RESEARCH_DAILY_BAR_CLOCK = "finmind_adjusted_price_row_calendar"
ADJUSTED_PRICE_NO_PRICE_DAY_POLICY = "preserve_provider_carry_forward_row_if_canonical_ohlc_is_valid"

RESEARCH_CUTOFF_MODE_FIXED = "fixed"
RESEARCH_CUTOFF_MODE_BOOTSTRAP_COMMON_COMPLETE_MANIFEST = "bootstrap_common_complete_manifest"
RESEARCH_STATUS_ACTIVE_FROZEN = "active_frozen"
RESEARCH_STATUS_AUTHORIZED_NOT_READY = "authorized_not_ready"
RESEARCH_STATUS_READY_FROZEN = "ready_frozen"

MARKET_DATA_V2_CANONICAL_DATA_PLANE = "market_data_v2"
MARKET_DATA_V2_PROVIDER_ARCHIVE_ROLE = "neutral_provider_ssot"
MARKET_DATA_V2_HISTORICAL_PIT_UNIVERSE_ROLE = "neutral_derived_ssot"
MARKET_DATA_V2_PROVIDER_SNAPSHOT_SOURCE = "market_data_v2_provider_snapshot"
MARKET_DATA_V2_RESEARCH_VIEW_ROLE = "frozen_scientific_view"
MARKET_DATA_V2_TRADING_VIEW_ROLE = "latest_operational_view"
MARKET_DATA_V2_LEGACY_TARGET_ROLE = "v2_materialized_compatibility_only"
MARKET_DATA_V2_MIGRATION_PHASE_LEGACY_EXECUTION_TRANSITION = "legacy_execution_transition"
MARKET_DATA_V2_MIGRATION_PHASE_V2_EXECUTION_CUTOVER = "v2_execution_cutover"
MARKET_DATA_V2_MIGRATION_PHASE_V2_ONLY = "v2_only"
MARKET_DATA_V2_MIGRATION_PHASES = (
    MARKET_DATA_V2_MIGRATION_PHASE_LEGACY_EXECUTION_TRANSITION,
    MARKET_DATA_V2_MIGRATION_PHASE_V2_EXECUTION_CUTOVER,
    MARKET_DATA_V2_MIGRATION_PHASE_V2_ONLY,
)


@dataclass(frozen=True)
class MarketPriceSourceContract:
    provider_id: str
    adjusted_dataset: str
    raw_archive_dataset: str
    raw_direct_consumption_enabled: bool
    project_adjusted_price_engine_enabled: bool
    retrospective_adjustment_invariance_required: bool
    research_daily_bar_clock: str
    adjusted_price_no_price_day_policy: str


@dataclass(frozen=True)
class ResearchDataGenerationContract:
    generation_id: str
    status: str
    lifecycle: str
    cutoff_mode: str
    cutoff: str | None
    required_cutoff: str
    universe_mode: str


@dataclass(frozen=True)
class TradingMarketDataLifecycleContract:
    mode: str
    provider_archive_source: str
    bootstrap_source_generation: str
    status: str


@dataclass(frozen=True)
class MarketDataV2LifecycleContract:
    canonical_data_plane: str
    provider_archive_role: str
    historical_pit_universe_role: str
    historical_pit_universe_shared_across_domains: bool
    domain_specific_historical_membership_rebuild_allowed: bool
    research_view_role: str
    trading_view_role: str
    shared_provider_archive_required: bool
    independent_domain_provider_redownload_allowed: bool
    legacy_target_role: str
    legacy_target_must_derive_from_v2: bool
    migration_phase: str
    legacy_direct_provider_download_allowed_during_transition: bool


def _require_nonempty_text(value: object, *, field: str) -> str:
    resolved = str(value or "").strip()
    if not resolved:
        raise ValueError(f"{field} 不可空白")
    return resolved


def _validate_iso_date(value: str, *, field: str) -> str:
    resolved = _require_nonempty_text(value, field=field)
    try:
        date.fromisoformat(resolved)
    except ValueError as exc:
        raise ValueError(f"{field} 必須是 YYYY-MM-DD: {resolved!r}") from exc
    return resolved


def get_market_price_source_contract() -> MarketPriceSourceContract:
    if FINMIND_ADJUSTED_PRICE_DATASET == FINMIND_RAW_PRICE_ARCHIVE_DATASET:
        raise ValueError("adjusted/raw price dataset 不得共用同一 identity")
    if RAW_PRICE_DIRECT_CONSUMPTION_ENABLED:
        raise ValueError("目前治理不允許 raw price 直接成為模型/策略價格 input")
    if PROJECT_ADJUSTED_PRICE_ENGINE_ENABLED:
        raise ValueError("目前治理禁止專案自行維護第二套台股還原價 engine")
    return MarketPriceSourceContract(
        provider_id=MARKET_DATA_PROVIDER_ID,
        adjusted_dataset=FINMIND_ADJUSTED_PRICE_DATASET,
        raw_archive_dataset=FINMIND_RAW_PRICE_ARCHIVE_DATASET,
        raw_direct_consumption_enabled=RAW_PRICE_DIRECT_CONSUMPTION_ENABLED,
        project_adjusted_price_engine_enabled=PROJECT_ADJUSTED_PRICE_ENGINE_ENABLED,
        retrospective_adjustment_invariance_required=ADJUSTED_PRICE_RETROSPECTIVE_INVARIANCE_REQUIRED,
        research_daily_bar_clock=RESEARCH_DAILY_BAR_CLOCK,
        adjusted_price_no_price_day_policy=ADJUSTED_PRICE_NO_PRICE_DAY_POLICY,
    )


def get_research_data_generation(generation_id: str) -> ResearchDataGenerationContract:
    resolved_id = _require_nonempty_text(generation_id, field="research generation_id")
    raw = RESEARCH_DATA_GENERATIONS.get(resolved_id)
    if not isinstance(raw, Mapping):
        raise ValueError(f"未登記 Research market-data generation: {resolved_id}")

    status = _require_nonempty_text(raw.get("status"), field=f"{resolved_id}.status")
    lifecycle = _require_nonempty_text(raw.get("lifecycle"), field=f"{resolved_id}.lifecycle")
    cutoff_mode = _require_nonempty_text(raw.get("cutoff_mode"), field=f"{resolved_id}.cutoff_mode")
    universe_mode = _require_nonempty_text(raw.get("universe_mode"), field=f"{resolved_id}.universe_mode")
    cutoff_raw = raw.get("cutoff")
    required_cutoff_raw = raw.get("required_cutoff", cutoff_raw)
    required_cutoff = _validate_iso_date(required_cutoff_raw, field=f"{resolved_id}.required_cutoff")

    if cutoff_mode == RESEARCH_CUTOFF_MODE_FIXED:
        cutoff = _validate_iso_date(cutoff_raw, field=f"{resolved_id}.cutoff")
        if cutoff != required_cutoff:
            raise ValueError(f"{resolved_id} fixed cutoff 必須等於 required_cutoff")
    elif cutoff_mode == RESEARCH_CUTOFF_MODE_BOOTSTRAP_COMMON_COMPLETE_MANIFEST:
        if status == RESEARCH_STATUS_AUTHORIZED_NOT_READY:
            if cutoff_raw is not None:
                raise ValueError(f"{resolved_id} 尚未 READY 前不得預先硬編 cutoff")
            cutoff = None
        else:
            cutoff = _validate_iso_date(cutoff_raw, field=f"{resolved_id}.cutoff")
            if cutoff != required_cutoff:
                raise ValueError(f"{resolved_id} READY frozen cutoff 必須等於 required_cutoff")
    else:
        raise ValueError(f"不支援的 Research cutoff_mode: {cutoff_mode}")

    if status not in {
        RESEARCH_STATUS_ACTIVE_FROZEN,
        RESEARCH_STATUS_AUTHORIZED_NOT_READY,
        RESEARCH_STATUS_READY_FROZEN,
    }:
        raise ValueError(f"不支援的 Research generation status: {status}")

    return ResearchDataGenerationContract(
        generation_id=resolved_id,
        status=status,
        lifecycle=lifecycle,
        cutoff_mode=cutoff_mode,
        cutoff=cutoff,
        required_cutoff=required_cutoff,
        universe_mode=universe_mode,
    )


def get_active_research_data_generation() -> ResearchDataGenerationContract:
    """Resolve the declarative source-code fallback Research generation."""

    contract = get_research_data_generation(ACTIVE_RESEARCH_DATA_GENERATION)
    if contract.cutoff is None:
        raise ValueError("ACTIVE Research market-data generation 必須已有 frozen cutoff")
    if contract.status not in {RESEARCH_STATUS_ACTIVE_FROZEN, RESEARCH_STATUS_READY_FROZEN}:
        raise ValueError(f"ACTIVE Research generation 尚未 READY: {contract.status}")
    return contract


def get_market_data_v2_lifecycle() -> MarketDataV2LifecycleContract:
    raw = MARKET_DATA_V2_LIFECYCLE
    if not isinstance(raw, Mapping):
        raise ValueError("MARKET_DATA_V2_LIFECYCLE 必須是 mapping")

    canonical_data_plane = _require_nonempty_text(
        raw.get("canonical_data_plane"), field="market_data_v2.canonical_data_plane"
    )
    provider_archive_role = _require_nonempty_text(
        raw.get("provider_archive_role"), field="market_data_v2.provider_archive_role"
    )
    historical_pit_universe_role = _require_nonempty_text(
        raw.get("historical_pit_universe_role"), field="market_data_v2.historical_pit_universe_role"
    )
    research_view_role = _require_nonempty_text(
        raw.get("research_view_role"), field="market_data_v2.research_view_role"
    )
    trading_view_role = _require_nonempty_text(
        raw.get("trading_view_role"), field="market_data_v2.trading_view_role"
    )
    legacy_target_role = _require_nonempty_text(
        raw.get("legacy_target_role"), field="market_data_v2.legacy_target_role"
    )
    migration_phase = _require_nonempty_text(
        raw.get("migration_phase"), field="market_data_v2.migration_phase"
    )

    shared_provider_archive_required = bool(raw.get("shared_provider_archive_required"))
    historical_pit_universe_shared_across_domains = bool(
        raw.get("historical_pit_universe_shared_across_domains")
    )
    domain_specific_historical_membership_rebuild_allowed = bool(
        raw.get("domain_specific_historical_membership_rebuild_allowed")
    )
    independent_domain_provider_redownload_allowed = bool(
        raw.get("independent_domain_provider_redownload_allowed")
    )
    legacy_target_must_derive_from_v2 = bool(raw.get("legacy_target_must_derive_from_v2"))
    legacy_direct_provider_download_allowed_during_transition = bool(
        raw.get("legacy_direct_provider_download_allowed_during_transition")
    )

    if canonical_data_plane != MARKET_DATA_V2_CANONICAL_DATA_PLANE:
        raise ValueError(f"Market Data canonical data plane 必須為 {MARKET_DATA_V2_CANONICAL_DATA_PLANE}")
    if provider_archive_role != MARKET_DATA_V2_PROVIDER_ARCHIVE_ROLE:
        raise ValueError(f"Market Data V2 provider archive role 必須為 {MARKET_DATA_V2_PROVIDER_ARCHIVE_ROLE}")
    if historical_pit_universe_role != MARKET_DATA_V2_HISTORICAL_PIT_UNIVERSE_ROLE:
        raise ValueError(
            f"Market Data V2 historical PIT universe role 必須為 {MARKET_DATA_V2_HISTORICAL_PIT_UNIVERSE_ROLE}"
        )
    if not historical_pit_universe_shared_across_domains:
        raise ValueError("Market Data V2 historical PIT universe 必須由 Research / Trading 共用")
    if domain_specific_historical_membership_rebuild_allowed:
        raise ValueError("Research / Trading 不得各自重建第二套 historical PIT membership truth")
    if research_view_role != MARKET_DATA_V2_RESEARCH_VIEW_ROLE:
        raise ValueError(f"Market Data V2 Research view role 必須為 {MARKET_DATA_V2_RESEARCH_VIEW_ROLE}")
    if trading_view_role != MARKET_DATA_V2_TRADING_VIEW_ROLE:
        raise ValueError(f"Market Data V2 Trading view role 必須為 {MARKET_DATA_V2_TRADING_VIEW_ROLE}")
    if not shared_provider_archive_required:
        raise ValueError("Market Data V2 必須由 Research / Trading 共用 neutral Provider Archive")
    if independent_domain_provider_redownload_allowed:
        raise ValueError("Market Data V2 禁止 Research / Trading 為相同 provider history 建立獨立下載真理")
    if legacy_target_role != MARKET_DATA_V2_LEGACY_TARGET_ROLE:
        raise ValueError(f"Legacy target role 必須為 {MARKET_DATA_V2_LEGACY_TARGET_ROLE}")
    if not legacy_target_must_derive_from_v2:
        raise ValueError("Legacy compatibility target 必須由 V2 materialize，不得維持獨立 provider truth")
    if migration_phase not in MARKET_DATA_V2_MIGRATION_PHASES:
        raise ValueError(f"不支援的 Market Data V2 migration phase: {migration_phase}")
    if migration_phase == MARKET_DATA_V2_MIGRATION_PHASE_LEGACY_EXECUTION_TRANSITION:
        if not legacy_direct_provider_download_allowed_during_transition:
            raise ValueError("legacy execution transition 必須如實允許尚未退役的 direct-provider producer")
    elif legacy_direct_provider_download_allowed_during_transition:
        raise ValueError("V2 execution cutover 後不得再允許 legacy direct-provider producer")

    return MarketDataV2LifecycleContract(
        canonical_data_plane=canonical_data_plane,
        provider_archive_role=provider_archive_role,
        historical_pit_universe_role=historical_pit_universe_role,
        historical_pit_universe_shared_across_domains=historical_pit_universe_shared_across_domains,
        domain_specific_historical_membership_rebuild_allowed=domain_specific_historical_membership_rebuild_allowed,
        research_view_role=research_view_role,
        trading_view_role=trading_view_role,
        shared_provider_archive_required=shared_provider_archive_required,
        independent_domain_provider_redownload_allowed=independent_domain_provider_redownload_allowed,
        legacy_target_role=legacy_target_role,
        legacy_target_must_derive_from_v2=legacy_target_must_derive_from_v2,
        migration_phase=migration_phase,
        legacy_direct_provider_download_allowed_during_transition=legacy_direct_provider_download_allowed_during_transition,
    )


def get_trading_market_data_lifecycle() -> TradingMarketDataLifecycleContract:
    raw = TRADING_MARKET_DATA_LIFECYCLE
    if not isinstance(raw, Mapping):
        raise ValueError("TRADING_MARKET_DATA_LIFECYCLE 必須是 mapping")
    mode = _require_nonempty_text(raw.get("mode"), field="trading.mode")
    provider_archive_source = _require_nonempty_text(
        raw.get("provider_archive_source"), field="trading.provider_archive_source"
    )
    bootstrap_source_generation = _require_nonempty_text(
        raw.get("bootstrap_source_generation"),
        field="trading.bootstrap_source_generation",
    )
    status = _require_nonempty_text(raw.get("status"), field="trading.status")
    if mode != "incremental_latest":
        raise ValueError(f"不支援的 Trading market-data mode: {mode}")
    if provider_archive_source != MARKET_DATA_V2_PROVIDER_SNAPSHOT_SOURCE:
        raise ValueError(
            f"Trading V2 provider archive source 必須為 {MARKET_DATA_V2_PROVIDER_SNAPSHOT_SOURCE}"
        )
    # Compatibility metadata remains resolvable during migration, but it does
    # not own provider/archive truth; both Research and Trading pin the neutral
    # Market Data V2 Provider Snapshot.
    get_research_data_generation(bootstrap_source_generation)
    return TradingMarketDataLifecycleContract(
        mode=mode,
        provider_archive_source=provider_archive_source,
        bootstrap_source_generation=bootstrap_source_generation,
        status=status,
    )


def build_market_data_contract_snapshot() -> dict[str, object]:
    price = get_market_price_source_contract()
    active_research = get_active_research_data_generation()
    research_generations = {
        generation_id: get_research_data_generation(generation_id).__dict__
        for generation_id in RESEARCH_DATA_GENERATIONS
    }
    trading = get_trading_market_data_lifecycle()
    market_data_v2 = get_market_data_v2_lifecycle()
    return {
        "price_source": price.__dict__,
        "active_research_generation": active_research.__dict__,
        "research_generations": research_generations,
        "trading": trading.__dict__,
        "market_data_v2": market_data_v2.__dict__,
    }


__all__ = [
    "MARKET_DATA_PROVIDER_ID",
    "FINMIND_ADJUSTED_PRICE_DATASET",
    "FINMIND_RAW_PRICE_ARCHIVE_DATASET",
    "RAW_PRICE_DIRECT_CONSUMPTION_ENABLED",
    "PROJECT_ADJUSTED_PRICE_ENGINE_ENABLED",
    "ADJUSTED_PRICE_RETROSPECTIVE_INVARIANCE_REQUIRED",
    "RESEARCH_DAILY_BAR_CLOCK",
    "ADJUSTED_PRICE_NO_PRICE_DAY_POLICY",
    "RESEARCH_CUTOFF_MODE_FIXED",
    "RESEARCH_CUTOFF_MODE_BOOTSTRAP_COMMON_COMPLETE_MANIFEST",
    "RESEARCH_STATUS_ACTIVE_FROZEN",
    "RESEARCH_STATUS_AUTHORIZED_NOT_READY",
    "RESEARCH_STATUS_READY_FROZEN",
    "MARKET_DATA_V2_CANONICAL_DATA_PLANE",
    "MARKET_DATA_V2_PROVIDER_ARCHIVE_ROLE",
    "MARKET_DATA_V2_HISTORICAL_PIT_UNIVERSE_ROLE",
    "MARKET_DATA_V2_PROVIDER_SNAPSHOT_SOURCE",
    "MARKET_DATA_V2_RESEARCH_VIEW_ROLE",
    "MARKET_DATA_V2_TRADING_VIEW_ROLE",
    "MARKET_DATA_V2_LEGACY_TARGET_ROLE",
    "MARKET_DATA_V2_MIGRATION_PHASE_LEGACY_EXECUTION_TRANSITION",
    "MARKET_DATA_V2_MIGRATION_PHASE_V2_EXECUTION_CUTOVER",
    "MARKET_DATA_V2_MIGRATION_PHASE_V2_ONLY",
    "MARKET_DATA_V2_MIGRATION_PHASES",
    "MarketPriceSourceContract",
    "ResearchDataGenerationContract",
    "TradingMarketDataLifecycleContract",
    "MarketDataV2LifecycleContract",
    "get_market_price_source_contract",
    "get_research_data_generation",
    "get_active_research_data_generation",
    "get_market_data_v2_lifecycle",
    "get_trading_market_data_lifecycle",
    "build_market_data_contract_snapshot",
]
