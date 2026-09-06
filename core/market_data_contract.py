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

RESEARCH_CUTOFF_MODE_FIXED = "fixed"
RESEARCH_CUTOFF_MODE_BOOTSTRAP_COMMON_COMPLETE_MANIFEST = "bootstrap_common_complete_manifest"
RESEARCH_STATUS_ACTIVE_FROZEN = "active_frozen"
RESEARCH_STATUS_AUTHORIZED_NOT_READY = "authorized_not_ready"
RESEARCH_STATUS_READY_FROZEN = "ready_frozen"


@dataclass(frozen=True)
class MarketPriceSourceContract:
    provider_id: str
    adjusted_dataset: str
    raw_archive_dataset: str
    raw_direct_consumption_enabled: bool
    project_adjusted_price_engine_enabled: bool
    retrospective_adjustment_invariance_required: bool


@dataclass(frozen=True)
class ResearchDataGenerationContract:
    generation_id: str
    status: str
    lifecycle: str
    cutoff_mode: str
    cutoff: str | None
    universe_mode: str


@dataclass(frozen=True)
class TradingMarketDataLifecycleContract:
    mode: str
    bootstrap_source_generation: str
    status: str


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

    if cutoff_mode == RESEARCH_CUTOFF_MODE_FIXED:
        cutoff = _validate_iso_date(cutoff_raw, field=f"{resolved_id}.cutoff")
    elif cutoff_mode == RESEARCH_CUTOFF_MODE_BOOTSTRAP_COMMON_COMPLETE_MANIFEST:
        if status == RESEARCH_STATUS_AUTHORIZED_NOT_READY:
            if cutoff_raw is not None:
                raise ValueError(f"{resolved_id} 尚未 READY 前不得預先硬編 cutoff")
            cutoff = None
        else:
            cutoff = _validate_iso_date(cutoff_raw, field=f"{resolved_id}.cutoff")
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
        universe_mode=universe_mode,
    )


def get_active_research_data_generation() -> ResearchDataGenerationContract:
    contract = get_research_data_generation(ACTIVE_RESEARCH_DATA_GENERATION)
    if contract.cutoff is None:
        raise ValueError("ACTIVE Research market-data generation 必須已有 frozen cutoff")
    if contract.status not in {RESEARCH_STATUS_ACTIVE_FROZEN, RESEARCH_STATUS_READY_FROZEN}:
        raise ValueError(f"ACTIVE Research generation 尚未 READY: {contract.status}")
    return contract


def get_trading_market_data_lifecycle() -> TradingMarketDataLifecycleContract:
    raw = TRADING_MARKET_DATA_LIFECYCLE
    if not isinstance(raw, Mapping):
        raise ValueError("TRADING_MARKET_DATA_LIFECYCLE 必須是 mapping")
    mode = _require_nonempty_text(raw.get("mode"), field="trading.mode")
    bootstrap_source_generation = _require_nonempty_text(
        raw.get("bootstrap_source_generation"),
        field="trading.bootstrap_source_generation",
    )
    status = _require_nonempty_text(raw.get("status"), field="trading.status")
    if mode != "incremental_latest":
        raise ValueError(f"不支援的 Trading market-data mode: {mode}")
    # The target bootstrap source must be a declared Research data generation;
    # it may remain AUTHORIZED_NOT_READY until the bootstrap round completes.
    get_research_data_generation(bootstrap_source_generation)
    return TradingMarketDataLifecycleContract(
        mode=mode,
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
    return {
        "price_source": price.__dict__,
        "active_research_generation": active_research.__dict__,
        "research_generations": research_generations,
        "trading": trading.__dict__,
    }


__all__ = [
    "MARKET_DATA_PROVIDER_ID",
    "FINMIND_ADJUSTED_PRICE_DATASET",
    "FINMIND_RAW_PRICE_ARCHIVE_DATASET",
    "RAW_PRICE_DIRECT_CONSUMPTION_ENABLED",
    "PROJECT_ADJUSTED_PRICE_ENGINE_ENABLED",
    "ADJUSTED_PRICE_RETROSPECTIVE_INVARIANCE_REQUIRED",
    "RESEARCH_CUTOFF_MODE_FIXED",
    "RESEARCH_CUTOFF_MODE_BOOTSTRAP_COMMON_COMPLETE_MANIFEST",
    "RESEARCH_STATUS_ACTIVE_FROZEN",
    "RESEARCH_STATUS_AUTHORIZED_NOT_READY",
    "RESEARCH_STATUS_READY_FROZEN",
    "MarketPriceSourceContract",
    "ResearchDataGenerationContract",
    "TradingMarketDataLifecycleContract",
    "get_market_price_source_contract",
    "get_research_data_generation",
    "get_active_research_data_generation",
    "get_trading_market_data_lifecycle",
    "build_market_data_contract_snapshot",
]
