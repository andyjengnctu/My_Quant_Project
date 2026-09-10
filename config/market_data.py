"""Declarative Market Data lifecycle configuration.

Stable provider/price semantics belong to ``core.market_data_contract``.  This
module only declares which Research data generation is active and the planned
Research/Trading lifecycle states.
"""

RESEARCH_DATA_GENERATION_V1 = "research_v1"
RESEARCH_DATA_GENERATION_V2 = "research_v2"
RESEARCH_REQUIRED_CUTOFF = "2026-03-02"

RESEARCH_DATA_GENERATIONS = {
    RESEARCH_DATA_GENERATION_V1: {
        "status": "active_frozen",
        "lifecycle": "immutable",
        "cutoff_mode": "fixed",
        "cutoff": RESEARCH_REQUIRED_CUTOFF,
        "required_cutoff": RESEARCH_REQUIRED_CUTOFF,
        "universe_mode": "static_20260302_screen_pool",
    },
    RESEARCH_DATA_GENERATION_V2: {
        "status": "authorized_not_ready",
        "lifecycle": "freeze_after_bootstrap",
        "cutoff_mode": "bootstrap_common_complete_manifest",
        "cutoff": None,
        # Research and Trading share a neutral provider archive, but Research V2
        # must never extend its scientific horizon beyond the fixed Research date.
        "required_cutoff": RESEARCH_REQUIRED_CUTOFF,
        "universe_mode": "daily_pit_eligibility",
    },
}

# V2 is authorized but must not become current Research truth before the
# bootstrap/common-complete snapshot is built and frozen.
ACTIVE_RESEARCH_DATA_GENERATION = RESEARCH_DATA_GENERATION_V1


# Market Data V2 bootstrap executor knobs are execution-only policy; they do not
# participate in Research scientific identity.  Quota values remain live-provider
# driven; these settings only control safety margin, polling and bounded retries.
MARKET_DATA_V2_HTTP_TIMEOUT_SEC = 30.0
MARKET_DATA_V2_PREFLIGHT_RETRY_POLICY = {
    "retryable_attempts": 2,
    "retry_backoff_seconds": (2.0,),
}

MARKET_DATA_V2_EXECUTION_POLICY = {
    "quota_reserve_requests": 50,
    "quota_resume_headroom_requests": 1000,
    "quota_refresh_every_requests": 25,
    "quota_poll_seconds": 30.0,
    "max_retryable_attempts": 4,
    "retry_backoff_seconds": (5.0, 30.0, 120.0),
    "job_lease_seconds": 300.0,
    "executor_lock_seconds": 300.0,
    "progress_every_committed_requests": 100,
}

# Market Data V2 storage knobs are operational only. Bootstrap data is first
# published into the neutral provider archive. Research and Trading may hold
# isolated lifecycle views/state, but they must not fork provider-history truth.
MARKET_DATA_V2_STORAGE_POLICY = {
    "format": "parquet",
    "compression": "zstd",
    "minimum_free_bytes": 10 * 1024**3,
    "staging_headroom_multiplier": 2.0,
    "minimum_staging_headroom_bytes": 64 * 1024**2,
}

# Final V2 data-plane target plus the truthful current migration phase.  The
# neutral Provider Archive is the one long-term provider-data SSOT shared by
# Research and Trading.  Domain-specific consumers may expose different views
# (frozen scientific vs latest operational), but must not create independent
# provider-history truths.  Legacy direct-provider CSV remains temporarily
# allowed only because current execution consumers have not yet migrated.
MARKET_DATA_V2_LIFECYCLE = {
    "canonical_data_plane": "market_data_v2",
    "provider_archive_role": "neutral_provider_ssot",
    "historical_pit_universe_role": "neutral_derived_ssot",
    "historical_pit_universe_shared_across_domains": True,
    "domain_specific_historical_membership_rebuild_allowed": False,
    # Pool layering is date-local. Model context may be a model-specific
    # projection of the same-day PIT market universe, but today's Trading
    # execution pool may never define historical model/training membership.
    "model_context_pool_role": "date_local_pit_projection",
    "model_context_pool_source": "neutral_daily_pit_market_universe",
    "model_context_may_depend_on_current_execution_pool": False,
    "trading_execution_pool_role": "date_local_new_entry_eligibility",
    "trading_execution_pool_source": "neutral_daily_pit_market_universe",
    "current_execution_pool_may_define_historical_training_universe": False,
    "research_view_role": "frozen_scientific_view",
    "trading_view_role": "latest_operational_view",
    "shared_provider_archive_required": True,
    "independent_domain_provider_redownload_allowed": False,
    "legacy_target_role": "v2_materialized_compatibility_only",
    "legacy_target_must_derive_from_v2": True,
    "migration_phase": "legacy_execution_transition",
    "legacy_direct_provider_download_allowed_during_transition": True,
}

TRADING_MARKET_DATA_LIFECYCLE = {
    "mode": "incremental_latest",
    "provider_archive_source": "market_data_v2_provider_snapshot",
    # Compatibility metadata retained during migration; provider/archive truth
    # is no longer semantically owned by a Research generation.
    "bootstrap_source_generation": RESEARCH_DATA_GENERATION_V2,
    # Execution-critical CSV truth remains in place while the V2 archive is
    # maintained as a non-blocking sidecar.  Future DL Trading must explicitly
    # declare V2 dataset dependencies before V2 may become execution-critical.
    "status": "legacy_execution_with_v2_archive_sidecar",
}

# Trading Market Data V2 archive synchronization is operational only; these
# repair windows do not change the active rule-based strategy/scientific truth.
# A sidecar failure must not block current full_rule_based_no_dl execution.
MARKET_DATA_V2_TRADING_SYNC_POLICY = {
    "enabled": True,
    "execution_fail_closed": False,
    "recent_repair_calendar_days": 7,
    "event_repair_calendar_days": 30,
}


# One-shot Trading Market Data V2 automatic updater policy.  Scheduler wake-ups
# are local-only; provider calls are allowed only when the due planner returns
# at least one dataset.
MARKET_DATA_V2_AUTO_UPDATE_POLICY = {
    "enabled": True,
    "scheduler_wake_minutes": 15,
    "scheduler_task_name": "My_Quant_Project Market Data Auto Update",
    "scheduler_initial_delay_minutes": 1,
    "scheduler_execution_time_limit_minutes": 60,
    "publication_retry_minutes": (15, 30, 60),
    "max_publication_retries": 3,
    "quota_defer_minutes": 15,
    "error_defer_minutes": 60,
    "worker_lock_minutes": 30,
    # Latest completed Trading day discovery is deliberately sparse.  The
    # scheduler may wake every 15 minutes, but this probe is only due after
    # the provider-documented Price publication window.  Weekends are skipped
    # locally; exchange holidays cost at most the bounded retry probes below.
    "market_date_discovery_first_check_time": "17:45",
    "market_date_discovery_retry_minutes": (15, 30, 60),
    "market_date_discovery_max_retries": 3,
}

# Dataset publication/freshness scheduling is Trading-operations metadata only.
# It does not participate in the neutral Provider Snapshot / Research scientific
# identity.  Dataset-specific overrides below are provider-documented publication
# times with a small observation grace; datasets without a verified time use the
# conservative next-day fallback until later evidence replaces it.
MARKET_DATA_V2_PUBLICATION_POLICY = {
    "timezone": "Asia/Taipei",
    "conservative_fallback": {
        "first_check_time": "01:45",
        "day_offset": 1,
        "source": "conservative_next_day_fallback",
    },
    "dataset_overrides": {
        "TaiwanStockInfo": {"first_check_time": "01:45", "day_offset": 0, "source": "provider_documentation:technical"},
        "TaiwanStockTradingDate": {"first_check_time": "18:15", "day_offset": 0, "source": "provider_documentation:technical"},
        "TaiwanStockPrice": {"first_check_time": "17:45", "day_offset": 0, "source": "provider_documentation:technical"},
        "TaiwanStockPriceAdj": {"first_check_time": "17:45", "day_offset": 0, "source": "provider_documentation:technical"},
        "TaiwanStockPER": {"first_check_time": "18:15", "day_offset": 0, "source": "provider_documentation:technical"},
        "TaiwanStockDayTrading": {"first_check_time": "21:45", "day_offset": 0, "source": "provider_documentation:technical"},
        "TaiwanStockPriceLimit": {"first_check_time": "18:15", "day_offset": 0, "source": "provider_documentation:technical"},
        "TaiwanStockTotalReturnIndex": {"first_check_time": "17:05", "day_offset": 0, "source": "provider_documentation:technical"},
        "TaiwanStockInstitutionalInvestorsBuySellWide": {"first_check_time": "20:15", "day_offset": 0, "source": "provider_documentation:chip"},
        "TaiwanStockTotalInstitutionalInvestors": {"first_check_time": "15:15", "day_offset": 0, "source": "provider_documentation:chip"},
        "TaiwanStockMarginPurchaseShortSale": {"first_check_time": "21:15", "day_offset": 0, "source": "provider_documentation:chip"},
        "TaiwanStockTotalMarginPurchaseShortSale": {"first_check_time": "21:15", "day_offset": 0, "source": "provider_documentation:chip"},
        "TaiwanStockShareholding": {"first_check_time": "21:15", "day_offset": 0, "source": "provider_documentation:chip"},
        "TaiwanStockSecuritiesLending": {"first_check_time": "15:15", "day_offset": 0, "source": "provider_documentation:chip"},
        "TaiwanDailyShortSaleBalances": {"first_check_time": "21:15", "day_offset": 0, "source": "provider_documentation:chip"},
        "TaiwanTotalExchangeMarginMaintenance": {"first_check_time": "21:15", "day_offset": 0, "source": "provider_documentation:chip"},
        "TaiwanStockDelisting": {"first_check_time": "23:45", "day_offset": 0, "source": "provider_documentation:fundamental"},
        "TaiwanStockDispositionSecuritiesPeriod": {"first_check_time": "23:15", "day_offset": 0, "source": "provider_documentation:chip_window_end"},
        "TaiwanStockDayTradingBorrowingFeeRate": {"first_check_time": "22:15", "day_offset": 0, "source": "provider_documentation:chip_window_end"},
        "TaiwanStockSplitPrice": {"first_check_time": "18:15", "day_offset": 0, "source": "provider_documentation:fundamental"},
        "TaiwanStockMarketValue": {"first_check_time": "23:45", "day_offset": 0, "source": "provider_documentation:fundamental"},
        "TaiwanStockMarketValueWeight": {"first_check_time": "23:55", "day_offset": 0, "source": "provider_documentation:fundamental"},
        "TaiwanFuturesDaily": {"first_check_time": "16:45", "day_offset": 0, "source": "provider_documentation:derivatives"},
        "TaiwanFuturesInstitutionalInvestors": {"first_check_time": "18:15", "day_offset": 0, "source": "provider_documentation:derivatives"},
        "TaiwanOptionInstitutionalInvestors": {"first_check_time": "16:15", "day_offset": 0, "source": "provider_documentation:derivatives"},
        "TaiwanOptionVix": {"first_check_time": "18:15", "day_offset": 0, "source": "provider_documentation:derivatives"},
    },
}

__all__ = [
    "RESEARCH_DATA_GENERATION_V1",
    "RESEARCH_DATA_GENERATION_V2",
    "RESEARCH_REQUIRED_CUTOFF",
    "RESEARCH_DATA_GENERATIONS",
    "ACTIVE_RESEARCH_DATA_GENERATION",
    "TRADING_MARKET_DATA_LIFECYCLE",
    "MARKET_DATA_V2_HTTP_TIMEOUT_SEC",
    "MARKET_DATA_V2_PREFLIGHT_RETRY_POLICY",
    "MARKET_DATA_V2_EXECUTION_POLICY",
    "MARKET_DATA_V2_STORAGE_POLICY",
    "MARKET_DATA_V2_LIFECYCLE",
    "MARKET_DATA_V2_TRADING_SYNC_POLICY",
    "MARKET_DATA_V2_AUTO_UPDATE_POLICY",
    "MARKET_DATA_V2_PUBLICATION_POLICY",
]
