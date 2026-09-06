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

# Market Data V2 storage knobs are operational only.  Bootstrap data is first
# published into a neutral provider archive; Research V2 and Trading will branch
# into isolated lifecycle snapshots in later rounds.
MARKET_DATA_V2_STORAGE_POLICY = {
    "format": "parquet",
    "compression": "zstd",
    "minimum_free_bytes": 10 * 1024**3,
    "staging_headroom_multiplier": 2.0,
    "minimum_staging_headroom_bytes": 64 * 1024**2,
}

TRADING_MARKET_DATA_LIFECYCLE = {
    "mode": "incremental_latest",
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

__all__ = [
    "RESEARCH_DATA_GENERATION_V1",
    "RESEARCH_DATA_GENERATION_V2",
    "RESEARCH_DATA_GENERATIONS",
    "ACTIVE_RESEARCH_DATA_GENERATION",
    "TRADING_MARKET_DATA_LIFECYCLE",
    "MARKET_DATA_V2_HTTP_TIMEOUT_SEC",
    "MARKET_DATA_V2_PREFLIGHT_RETRY_POLICY",
    "MARKET_DATA_V2_EXECUTION_POLICY",
    "MARKET_DATA_V2_STORAGE_POLICY",
    "MARKET_DATA_V2_TRADING_SYNC_POLICY",
]
