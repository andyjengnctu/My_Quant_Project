"""Declarative Market Data lifecycle configuration.

Stable provider/price semantics belong to ``core.market_data_contract``.  This
module only declares which Research data generation is active and the planned
Research/Trading lifecycle states.
"""

RESEARCH_DATA_GENERATION_V1 = "research_v1"
RESEARCH_DATA_GENERATION_V2 = "research_v2"

RESEARCH_DATA_GENERATIONS = {
    RESEARCH_DATA_GENERATION_V1: {
        "status": "active_frozen",
        "lifecycle": "immutable",
        "cutoff_mode": "fixed",
        "cutoff": "2026-03-02",
        "universe_mode": "static_20260302_screen_pool",
    },
    RESEARCH_DATA_GENERATION_V2: {
        "status": "authorized_not_ready",
        "lifecycle": "freeze_after_bootstrap",
        "cutoff_mode": "bootstrap_common_complete_manifest",
        "cutoff": None,
        "universe_mode": "daily_pit_eligibility",
    },
}

# V2 is authorized but must not become current Research truth before the
# bootstrap/common-complete snapshot is built and frozen.
ACTIVE_RESEARCH_DATA_GENERATION = RESEARCH_DATA_GENERATION_V1

TRADING_MARKET_DATA_LIFECYCLE = {
    "mode": "incremental_latest",
    "bootstrap_source_generation": RESEARCH_DATA_GENERATION_V2,
    "status": "legacy_downloader_until_v2_ready",
}

__all__ = [
    "RESEARCH_DATA_GENERATION_V1",
    "RESEARCH_DATA_GENERATION_V2",
    "RESEARCH_DATA_GENERATIONS",
    "ACTIVE_RESEARCH_DATA_GENERATION",
    "TRADING_MARKET_DATA_LIFECYCLE",
]
