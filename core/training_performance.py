"""Runtime resolution for optimizer performance-policy config.

Config owns only declarative execution knobs.  This module owns coercion,
environment overrides, derived worker counts, and snapshots.
"""

from __future__ import annotations

import os

from config.training_performance_policy import (
    OPTIMIZER_FEATURE_BANK_MAX_ITEMS,
    OPTIMIZER_LOCAL_MIN_DEPENDENCY_STATS_ENABLED,
    OPTIMIZER_LOCAL_MIN_PORTFOLIO_DEPENDENCY_ORDER,
    OPTIMIZER_LOCAL_MIN_SIGNAL_DEPENDENCY_FIELD_ORDER,
    OPTIMIZER_POLICY_REPLAY_CONTEXT_REUSE_ENABLED,
    OPTIMIZER_POLICY_REPLAY_DEDUP_BY_SIGNATURE,
    OPTIMIZER_POLICY_REPLAY_PARALLEL_BACKEND,
    OPTIMIZER_POLICY_REPLAY_PARALLEL_WORKERS,
    OPTIMIZER_RANDOM_SEED_ENSEMBLE_PARALLEL_BACKEND,
    OPTIMIZER_RANDOM_SEED_ENSEMBLE_PARALLEL_WORKERS,
    OPTIMIZER_ROLLING_FOLD_WORKERS,
    OPTIMIZER_ROLLING_PARALLEL_PREP_CACHE_MAX_ITEMS,
    OPTIMIZER_SINGLE_FOLD_ALLOW_TPE_PARALLEL_SEARCH,
    OPTIMIZER_SINGLE_FOLD_LOCAL_MIN_PARALLEL_WORKERS,
    OPTIMIZER_SINGLE_FOLD_LOCAL_MIN_PARALLEL_MAX_WORKERS,
    OPTIMIZER_SINGLE_FOLD_LOCAL_MIN_PROCESS_WORKERS,
    OPTIMIZER_SINGLE_FOLD_SEARCH_PARALLEL_TRIALS,
    OPTIMIZER_ACTIVE_REPLAY_INCLUDE_PIT_STATS_INDEX,
    OPTIMIZER_ACTIVE_REPLAY_INCLUDE_TRADE_LOGS,
    OPTIMIZER_ACTIVE_REPLAY_PREP_AUTO_MAX_WORKERS,
    OPTIMIZER_ACTIVE_REPLAY_PREP_WORKERS,
    OPTIMIZER_ACTIVE_REPLAY_USE_PREPARED_CACHE,
    OPTIMIZER_ACTIVE_REPLAY_WRITE_PREPARED_CACHE,
    OPTIMIZER_FULL_EVAL_CACHE_MAX_ITEMS,
    OPTIMIZER_LOCAL_MIN_PROGRESS_MIN_INTERVAL_SEC,
    OPTIMIZER_OUTER_ROLLING_PROFILE_WRITE_FILES,
    OPTIMIZER_OUTER_ROLLING_STUDY_STORAGE,
    OPTIMIZER_PORTFOLIO_PREP_AUTO_MAX_WORKERS,
    OPTIMIZER_PORTFOLIO_PREP_PARALLEL_MIN_TICKERS,
    OPTIMIZER_PORTFOLIO_PREP_WORKERS,
    OPTIMIZER_PROFILE_WRITE_FILES,
    OPTIMIZER_RAW_CACHE_LOCK_STALE_SEC,
    OPTIMIZER_RESOURCE_DISK_MBPS_CAP,
    OPTIMIZER_RESOURCE_SAMPLE_INTERVAL_SEC,
    OPTIMIZER_RESOURCE_WRITE_CSV,
    OPTIMIZER_ROLLING_SHARED_PREP_CACHE_MAX_ITEMS,
)
from core.seed_ensemble_policy import (
    resolve_seed_ensemble_parallel_backend,
    resolve_seed_ensemble_parallel_workers,
)


_LOCAL_MIN_SIGNAL_DEPENDENCY_FIELD_ORDER_HARD_FAIL_FIRST = (
    "high_len",
    "atr_times_trail",
    "atr_buy_tol",
    "bb_len",
    "atr_len",
    "bb_mult",
    "kc_len",
    "kc_mult",
    "vol_long_len",
    "vol_breakout_mult",
    "breakout_return_min",
)


def _coerce_int(value, *, default: int, min_value: int = 0, max_value: int | None = None) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        resolved = int(default)
    resolved = max(int(min_value), resolved)
    if max_value is not None:
        resolved = min(int(max_value), resolved)
    return resolved


def _coerce_bool(value, *, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "y"}:
        return True
    if text in {"0", "false", "no", "off", "n"}:
        return False
    return bool(default)


def resolve_optimizer_rolling_fold_workers_default(fold_count):
    resolved_fold_count = _coerce_int(fold_count, default=1, min_value=1)
    raw_value = OPTIMIZER_ROLLING_FOLD_WORKERS
    if raw_value is None:
        return resolved_fold_count
    text = str(raw_value).strip().lower()
    if text in {"", "fold_count", "folds", "auto"}:
        return resolved_fold_count
    return _coerce_int(raw_value, default=resolved_fold_count, min_value=1)


def resolve_optimizer_rolling_parallel_prep_cache_max_items_default():
    return _coerce_int(
        OPTIMIZER_ROLLING_PARALLEL_PREP_CACHE_MAX_ITEMS,
        default=0,
        min_value=0,
        max_value=256,
    )


def resolve_optimizer_feature_bank_max_items_default():
    return _coerce_int(
        OPTIMIZER_FEATURE_BANK_MAX_ITEMS,
        default=1024,
        min_value=0,
        max_value=65536,
    )


def resolve_optimizer_local_min_portfolio_dependency_order():
    raw_value = os.environ.get(
        "OPTIMIZER_LOCAL_MIN_PORTFOLIO_DEPENDENCY_ORDER",
        OPTIMIZER_LOCAL_MIN_PORTFOLIO_DEPENDENCY_ORDER,
    )
    text = str(raw_value or "").strip().lower()
    if text in {"last", "deprioritize", "deprioritized", "tail"}:
        return "last"
    if text in {"original", "off", "0", "false", "no"}:
        return "original"
    return "last"


def resolve_optimizer_local_min_signal_dependency_field_order() -> tuple[str, ...]:
    raw_value = os.environ.get(
        "OPTIMIZER_LOCAL_MIN_SIGNAL_DEPENDENCY_FIELD_ORDER",
        OPTIMIZER_LOCAL_MIN_SIGNAL_DEPENDENCY_FIELD_ORDER,
    )
    text = str(raw_value or "").strip()
    normalized = text.lower()
    if normalized in {"", "hard_fail_first", "default", "on", "1", "true", "yes"}:
        return tuple(_LOCAL_MIN_SIGNAL_DEPENDENCY_FIELD_ORDER_HARD_FAIL_FIRST)
    if normalized in {"original", "off", "0", "false", "no"}:
        return tuple()
    fields = []
    seen = set()
    for part in text.split(","):
        field = str(part or "").strip()
        if not field or field in seen:
            continue
        fields.append(field)
        seen.add(field)
    return tuple(fields)


def resolve_optimizer_single_fold_search_parallel_trials_default():
    return _coerce_int(
        OPTIMIZER_SINGLE_FOLD_SEARCH_PARALLEL_TRIALS,
        default=1,
        min_value=1,
        max_value=16,
    )


def is_optimizer_single_fold_tpe_parallel_search_allowed_default():
    return _coerce_bool(
        OPTIMIZER_SINGLE_FOLD_ALLOW_TPE_PARALLEL_SEARCH,
        default=False,
    )


def resolve_optimizer_single_fold_local_min_parallel_workers_default():
    return _coerce_int(
        OPTIMIZER_SINGLE_FOLD_LOCAL_MIN_PARALLEL_WORKERS,
        default=1,
        min_value=1,
    )


def resolve_optimizer_single_fold_local_min_parallel_workers(environ=None) -> int:
    default_workers = resolve_optimizer_single_fold_local_min_parallel_workers_default()
    return _resolve_env_int(
        "OPTIMIZER_LOCAL_MIN_PARALLEL_WORKERS",
        config_default=default_workers,
        environ=environ,
        min_value=1,
        max_value=_coerce_int(OPTIMIZER_SINGLE_FOLD_LOCAL_MIN_PARALLEL_MAX_WORKERS, default=4, min_value=1),
    )


def resolve_optimizer_single_fold_local_min_process_workers_default():
    return _coerce_int(
        OPTIMIZER_SINGLE_FOLD_LOCAL_MIN_PROCESS_WORKERS,
        default=0,
        min_value=0,
    )


def resolve_optimizer_random_seed_ensemble_parallel_workers_default(seed_count):
    return resolve_seed_ensemble_parallel_workers(
        seed_count,
        OPTIMIZER_RANDOM_SEED_ENSEMBLE_PARALLEL_WORKERS,
    )


def resolve_optimizer_random_seed_ensemble_parallel_backend_default():
    return resolve_seed_ensemble_parallel_backend(
        OPTIMIZER_RANDOM_SEED_ENSEMBLE_PARALLEL_BACKEND
    )


def resolve_optimizer_policy_replay_parallel_workers_default(policy_count):
    resolved_policy_count = _coerce_int(policy_count, default=1, min_value=1)
    raw_value = os.environ.get(
        "OPTIMIZER_POLICY_REPLAY_PARALLEL_WORKERS",
        OPTIMIZER_POLICY_REPLAY_PARALLEL_WORKERS,
    )
    text = str(raw_value or "").strip().lower()
    if text in {"", "auto", "policy_count", "policies", "replay_count"}:
        return resolved_policy_count
    return _coerce_int(
        raw_value,
        default=resolved_policy_count,
        min_value=1,
        max_value=resolved_policy_count,
    )


def resolve_optimizer_policy_replay_parallel_backend_default():
    raw_value = os.environ.get(
        "OPTIMIZER_POLICY_REPLAY_PARALLEL_BACKEND",
        OPTIMIZER_POLICY_REPLAY_PARALLEL_BACKEND,
    )
    text = str(raw_value or "").strip().lower()
    if text in {"thread", "threads", "threadpool"}:
        return "thread"
    return "process"


def is_optimizer_policy_replay_dedup_by_signature_enabled_default():
    raw_value = os.environ.get(
        "OPTIMIZER_POLICY_REPLAY_DEDUP_BY_SIGNATURE",
        OPTIMIZER_POLICY_REPLAY_DEDUP_BY_SIGNATURE,
    )
    return _coerce_bool(raw_value, default=True)


def is_optimizer_policy_replay_context_reuse_enabled_default():
    raw_value = os.environ.get(
        "OPTIMIZER_POLICY_REPLAY_CONTEXT_REUSE_ENABLED",
        OPTIMIZER_POLICY_REPLAY_CONTEXT_REUSE_ENABLED,
    )
    return _coerce_bool(raw_value, default=True)


def is_optimizer_local_min_dependency_stats_enabled():
    raw_value = os.environ.get(
        "OPTIMIZER_LOCAL_MIN_DEPENDENCY_STATS_ENABLED",
        OPTIMIZER_LOCAL_MIN_DEPENDENCY_STATS_ENABLED,
    )
    return _coerce_bool(raw_value, default=True)



def _environment_value(name: str, *, environ=None):
    if isinstance(environ, dict) and name in environ:
        return environ.get(name)
    return os.environ.get(name)


def _resolve_env_int(name: str, *, config_default, environ=None, min_value: int = 0, max_value: int | None = None) -> int:
    raw_value = _environment_value(name, environ=environ)
    if raw_value is None or str(raw_value).strip() == "":
        raw_value = config_default
    return _coerce_int(raw_value, default=int(config_default), min_value=min_value, max_value=max_value)


def _resolve_env_float(name: str, *, config_default, environ=None, min_value: float = 0.0, max_value: float | None = None) -> float:
    raw_value = _environment_value(name, environ=environ)
    if raw_value is None or str(raw_value).strip() == "":
        raw_value = config_default
    try:
        resolved = float(raw_value)
    except (TypeError, ValueError):
        resolved = float(config_default)
    resolved = max(float(min_value), resolved)
    if max_value is not None:
        resolved = min(float(max_value), resolved)
    return float(resolved)


def _resolve_env_bool(name: str, *, config_default: bool, environ=None) -> bool:
    raw_value = _environment_value(name, environ=environ)
    if raw_value is None or str(raw_value).strip() == "":
        raw_value = config_default
    return _coerce_bool(raw_value, default=bool(config_default))


def _resolve_env_permissive_bool(name: str, *, config_default: bool, environ=None) -> bool:
    raw_value = _environment_value(name, environ=environ)
    if raw_value is None or str(raw_value).strip() == "":
        return bool(config_default)
    return str(raw_value).strip().lower() not in {"0", "false", "no", "off", "n"}


def resolve_optimizer_rolling_shared_prep_cache_max_items(environ=None) -> int:
    return _resolve_env_int(
        "OPTIMIZER_ROLLING_SHARED_PREP_CACHE_MAX_ITEMS",
        config_default=OPTIMIZER_ROLLING_SHARED_PREP_CACHE_MAX_ITEMS,
        environ=environ,
        min_value=0,
        max_value=4096,
    )


def resolve_optimizer_full_evaluation_cache_max_items(environ=None) -> int:
    return _resolve_env_int(
        "OPTIMIZER_FULL_EVAL_CACHE_MAX_ITEMS",
        config_default=OPTIMIZER_FULL_EVAL_CACHE_MAX_ITEMS,
        environ=environ,
        min_value=0,
        max_value=4096,
    )


def resolve_optimizer_portfolio_prep_parallel_min_tickers_default() -> int:
    return _coerce_int(OPTIMIZER_PORTFOLIO_PREP_PARALLEL_MIN_TICKERS, default=30, min_value=1)


def resolve_optimizer_portfolio_prep_workers(raw_data_count: int, environ=None) -> int:
    count = _coerce_int(raw_data_count, default=1, min_value=1)
    raw_value = _environment_value("V16_PORTFOLIO_MAX_WORKERS", environ=environ)
    if raw_value is None or str(raw_value).strip() == "":
        raw_value = OPTIMIZER_PORTFOLIO_PREP_WORKERS
    text = str(raw_value or "").strip().lower()
    if text not in {"", "auto"}:
        try:
            requested = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"V16_PORTFOLIO_MAX_WORKERS 必須是整數，收到: {raw_value}") from exc
        return max(1, min(requested, count))
    if count < resolve_optimizer_portfolio_prep_parallel_min_tickers_default():
        return 1
    auto_cap = _coerce_int(OPTIMIZER_PORTFOLIO_PREP_AUTO_MAX_WORKERS, default=8, min_value=1)
    return max(1, min(os.cpu_count() or 1, auto_cap, count))


def resolve_optimizer_active_replay_prep_workers(raw_data_count: int, environ=None) -> int:
    count = _coerce_int(raw_data_count, default=1, min_value=1)
    raw_value = _environment_value("OPTIMIZER_ACTIVE_REPLAY_PREP_WORKERS", environ=environ)
    if raw_value is None or str(raw_value).strip() == "":
        raw_value = OPTIMIZER_ACTIVE_REPLAY_PREP_WORKERS
    text = str(raw_value or "").strip().lower()
    if text not in {"", "auto"}:
        try:
            requested = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"OPTIMIZER_ACTIVE_REPLAY_PREP_WORKERS 必須是整數，收到: {raw_value}") from exc
        return max(1, min(requested, count))
    auto_cap = _coerce_int(OPTIMIZER_ACTIVE_REPLAY_PREP_AUTO_MAX_WORKERS, default=8, min_value=1)
    return max(1, min(os.cpu_count() or 1, auto_cap, count))


def resolve_optimizer_raw_cache_lock_stale_sec(environ=None) -> float:
    raw_value = _environment_value("OPTIMIZER_RAW_CACHE_LOCK_STALE_SEC", environ=environ)
    if raw_value is None or str(raw_value).strip() == "":
        return max(60.0, float(OPTIMIZER_RAW_CACHE_LOCK_STALE_SEC))
    try:
        return max(60.0, float(raw_value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"OPTIMIZER_RAW_CACHE_LOCK_STALE_SEC 必須是數字秒數，收到: {raw_value}") from exc


def resolve_optimizer_local_min_progress_min_interval_sec(environ=None) -> float:
    return _resolve_env_float(
        "OPTIMIZER_LOCAL_MIN_PROGRESS_MIN_INTERVAL_SEC",
        config_default=OPTIMIZER_LOCAL_MIN_PROGRESS_MIN_INTERVAL_SEC,
        environ=environ,
        min_value=0.0,
        max_value=10.0,
    )


def is_optimizer_profile_write_files_enabled(environ=None, *, outer_rolling: bool = False) -> bool:
    default = OPTIMIZER_OUTER_ROLLING_PROFILE_WRITE_FILES if outer_rolling else OPTIMIZER_PROFILE_WRITE_FILES
    return _resolve_env_permissive_bool("OPTIMIZER_PROFILE_WRITE_FILES", config_default=default, environ=environ)


def resolve_optimizer_outer_rolling_study_storage_default() -> str:
    text = str(OPTIMIZER_OUTER_ROLLING_STUDY_STORAGE or "memory").strip().lower()
    if text in {"1", "true", "yes", "on", "sqlite", "sqlite_db", "db"}:
        return "sqlite"
    return "memory"


def is_optimizer_resource_write_csv_enabled(environ=None) -> bool:
    return _resolve_env_permissive_bool("OPTIMIZER_RESOURCE_WRITE_CSV", config_default=OPTIMIZER_RESOURCE_WRITE_CSV, environ=environ)


def resolve_optimizer_resource_sample_interval_sec(environ=None) -> float:
    return _resolve_env_float(
        "OPTIMIZER_RESOURCE_SAMPLE_INTERVAL_SEC",
        config_default=OPTIMIZER_RESOURCE_SAMPLE_INTERVAL_SEC,
        environ=environ,
        min_value=0.5,
        max_value=60.0,
    )


def resolve_optimizer_resource_disk_mbps_cap(environ=None) -> float:
    return _resolve_env_float(
        "OPTIMIZER_RESOURCE_DISK_MBPS_CAP",
        config_default=OPTIMIZER_RESOURCE_DISK_MBPS_CAP,
        environ=environ,
        min_value=1.0,
        max_value=10000.0,
    )


def is_optimizer_active_replay_include_trade_logs_enabled(environ=None) -> bool:
    return _resolve_env_permissive_bool(
        "OPTIMIZER_ACTIVE_REPLAY_INCLUDE_TRADE_LOGS",
        config_default=OPTIMIZER_ACTIVE_REPLAY_INCLUDE_TRADE_LOGS,
        environ=environ,
    )


def is_optimizer_active_replay_include_pit_stats_index_enabled(environ=None) -> bool:
    return _resolve_env_permissive_bool(
        "OPTIMIZER_ACTIVE_REPLAY_INCLUDE_PIT_STATS_INDEX",
        config_default=OPTIMIZER_ACTIVE_REPLAY_INCLUDE_PIT_STATS_INDEX,
        environ=environ,
    )


def is_optimizer_active_replay_use_prepared_cache_enabled(environ=None) -> bool:
    return _resolve_env_permissive_bool(
        "OPTIMIZER_ACTIVE_REPLAY_USE_PREPARED_CACHE",
        config_default=OPTIMIZER_ACTIVE_REPLAY_USE_PREPARED_CACHE,
        environ=environ,
    )


def is_optimizer_active_replay_write_prepared_cache_enabled(environ=None) -> bool:
    return _resolve_env_permissive_bool(
        "OPTIMIZER_ACTIVE_REPLAY_WRITE_PREPARED_CACHE",
        config_default=OPTIMIZER_ACTIVE_REPLAY_WRITE_PREPARED_CACHE,
        environ=environ,
    )


def build_optimizer_outer_rolling_env_defaults(fold_count: int) -> dict[str, str]:
    return {
        "OPTIMIZER_PROFILE_WRITE_FILES": "1" if bool(OPTIMIZER_OUTER_ROLLING_PROFILE_WRITE_FILES) else "0",
        "OPTIMIZER_OUTER_ROLLING_STUDY_STORAGE": resolve_optimizer_outer_rolling_study_storage_default(),
        "OPTIMIZER_RESOURCE_WRITE_CSV": "1" if bool(OPTIMIZER_RESOURCE_WRITE_CSV) else "0",
        "OPTIMIZER_ACTIVE_REPLAY_INCLUDE_TRADE_LOGS": "1" if bool(OPTIMIZER_ACTIVE_REPLAY_INCLUDE_TRADE_LOGS) else "0",
        "OPTIMIZER_ACTIVE_REPLAY_INCLUDE_PIT_STATS_INDEX": "1" if bool(OPTIMIZER_ACTIVE_REPLAY_INCLUDE_PIT_STATS_INDEX) else "0",
        "OPTIMIZER_ACTIVE_REPLAY_USE_PREPARED_CACHE": "1" if bool(OPTIMIZER_ACTIVE_REPLAY_USE_PREPARED_CACHE) else "0",
        "OPTIMIZER_ACTIVE_REPLAY_WRITE_PREPARED_CACHE": "1" if bool(OPTIMIZER_ACTIVE_REPLAY_WRITE_PREPARED_CACHE) else "0",
        "OPTIMIZER_ROLLING_FOLD_WORKERS": str(resolve_optimizer_rolling_fold_workers_default(fold_count)),
        "OPTIMIZER_SINGLE_FOLD_SEARCH_PARALLEL_TRIALS": str(resolve_optimizer_single_fold_search_parallel_trials_default()),
        "OPTIMIZER_SINGLE_FOLD_ALLOW_TPE_PARALLEL_SEARCH": "1" if is_optimizer_single_fold_tpe_parallel_search_allowed_default() else "0",
        "OPTIMIZER_LOCAL_MIN_PARALLEL_WORKERS": str(resolve_optimizer_single_fold_local_min_parallel_workers_default()),
        "OPTIMIZER_LOCAL_MIN_PROCESS_WORKERS": str(resolve_optimizer_single_fold_local_min_process_workers_default()),
        "OPTIMIZER_ROLLING_PARALLEL_PREP_CACHE_MAX_ITEMS": str(resolve_optimizer_rolling_parallel_prep_cache_max_items_default()),
        "OPTIMIZER_FEATURE_BANK_MAX_ITEMS": str(resolve_optimizer_feature_bank_max_items_default()),
    }

def build_training_performance_policy_snapshot(fold_count=None, seed_ensemble_size=None):
    return {
        "OPTIMIZER_ROLLING_FOLD_WORKERS": OPTIMIZER_ROLLING_FOLD_WORKERS,
        "OPTIMIZER_ROLLING_FOLD_WORKERS_RESOLVED": (
            None
            if fold_count is None
            else resolve_optimizer_rolling_fold_workers_default(fold_count)
        ),
        "OPTIMIZER_ROLLING_PARALLEL_PREP_CACHE_MAX_ITEMS": resolve_optimizer_rolling_parallel_prep_cache_max_items_default(),
        "OPTIMIZER_FEATURE_BANK_MAX_ITEMS": resolve_optimizer_feature_bank_max_items_default(),
        "OPTIMIZER_SINGLE_FOLD_SEARCH_PARALLEL_TRIALS": resolve_optimizer_single_fold_search_parallel_trials_default(),
        "OPTIMIZER_SINGLE_FOLD_ALLOW_TPE_PARALLEL_SEARCH": is_optimizer_single_fold_tpe_parallel_search_allowed_default(),
        "OPTIMIZER_SINGLE_FOLD_LOCAL_MIN_PARALLEL_WORKERS": resolve_optimizer_single_fold_local_min_parallel_workers_default(),
        "OPTIMIZER_SINGLE_FOLD_LOCAL_MIN_PARALLEL_MAX_WORKERS": int(OPTIMIZER_SINGLE_FOLD_LOCAL_MIN_PARALLEL_MAX_WORKERS),
        "OPTIMIZER_SINGLE_FOLD_LOCAL_MIN_PROCESS_WORKERS": resolve_optimizer_single_fold_local_min_process_workers_default(),
        "OPTIMIZER_RANDOM_SEED_ENSEMBLE_PARALLEL_WORKERS": OPTIMIZER_RANDOM_SEED_ENSEMBLE_PARALLEL_WORKERS,
        "OPTIMIZER_RANDOM_SEED_ENSEMBLE_PARALLEL_WORKERS_RESOLVED": (
            None
            if seed_ensemble_size is None
            else resolve_optimizer_random_seed_ensemble_parallel_workers_default(seed_ensemble_size)
        ),
        "OPTIMIZER_RANDOM_SEED_ENSEMBLE_PARALLEL_BACKEND": OPTIMIZER_RANDOM_SEED_ENSEMBLE_PARALLEL_BACKEND,
        "OPTIMIZER_RANDOM_SEED_ENSEMBLE_PARALLEL_BACKEND_RESOLVED": resolve_optimizer_random_seed_ensemble_parallel_backend_default(),
        "OPTIMIZER_POLICY_REPLAY_PARALLEL_WORKERS": OPTIMIZER_POLICY_REPLAY_PARALLEL_WORKERS,
        "OPTIMIZER_POLICY_REPLAY_PARALLEL_BACKEND": OPTIMIZER_POLICY_REPLAY_PARALLEL_BACKEND,
        "OPTIMIZER_POLICY_REPLAY_PARALLEL_BACKEND_RESOLVED": resolve_optimizer_policy_replay_parallel_backend_default(),
        "OPTIMIZER_POLICY_REPLAY_DEDUP_BY_SIGNATURE": is_optimizer_policy_replay_dedup_by_signature_enabled_default(),
        "OPTIMIZER_POLICY_REPLAY_CONTEXT_REUSE_ENABLED": is_optimizer_policy_replay_context_reuse_enabled_default(),
        "OPTIMIZER_LOCAL_MIN_DEPENDENCY_STATS_ENABLED": is_optimizer_local_min_dependency_stats_enabled(),
        "OPTIMIZER_LOCAL_MIN_PORTFOLIO_DEPENDENCY_ORDER": resolve_optimizer_local_min_portfolio_dependency_order(),
        "OPTIMIZER_LOCAL_MIN_SIGNAL_DEPENDENCY_FIELD_ORDER": ",".join(
            resolve_optimizer_local_min_signal_dependency_field_order()
        ),
        "OPTIMIZER_ROLLING_SHARED_PREP_CACHE_MAX_ITEMS": resolve_optimizer_rolling_shared_prep_cache_max_items(),
        "OPTIMIZER_FULL_EVAL_CACHE_MAX_ITEMS": resolve_optimizer_full_evaluation_cache_max_items(),
        "OPTIMIZER_PORTFOLIO_PREP_WORKERS": OPTIMIZER_PORTFOLIO_PREP_WORKERS,
        "OPTIMIZER_PORTFOLIO_PREP_PARALLEL_MIN_TICKERS": int(OPTIMIZER_PORTFOLIO_PREP_PARALLEL_MIN_TICKERS),
        "OPTIMIZER_PORTFOLIO_PREP_AUTO_MAX_WORKERS": int(OPTIMIZER_PORTFOLIO_PREP_AUTO_MAX_WORKERS),
        "OPTIMIZER_ACTIVE_REPLAY_PREP_WORKERS": OPTIMIZER_ACTIVE_REPLAY_PREP_WORKERS,
        "OPTIMIZER_ACTIVE_REPLAY_PREP_AUTO_MAX_WORKERS": int(OPTIMIZER_ACTIVE_REPLAY_PREP_AUTO_MAX_WORKERS),
        "OPTIMIZER_RAW_CACHE_LOCK_STALE_SEC": resolve_optimizer_raw_cache_lock_stale_sec(),
        "OPTIMIZER_LOCAL_MIN_PROGRESS_MIN_INTERVAL_SEC": resolve_optimizer_local_min_progress_min_interval_sec(),
        "OPTIMIZER_PROFILE_WRITE_FILES": is_optimizer_profile_write_files_enabled(),
        "OPTIMIZER_OUTER_ROLLING_PROFILE_WRITE_FILES": bool(OPTIMIZER_OUTER_ROLLING_PROFILE_WRITE_FILES),
        "OPTIMIZER_OUTER_ROLLING_STUDY_STORAGE": resolve_optimizer_outer_rolling_study_storage_default(),
        "OPTIMIZER_RESOURCE_WRITE_CSV": is_optimizer_resource_write_csv_enabled(),
        "OPTIMIZER_RESOURCE_SAMPLE_INTERVAL_SEC": resolve_optimizer_resource_sample_interval_sec(),
        "OPTIMIZER_RESOURCE_DISK_MBPS_CAP": resolve_optimizer_resource_disk_mbps_cap(),
        "OPTIMIZER_ACTIVE_REPLAY_INCLUDE_TRADE_LOGS": is_optimizer_active_replay_include_trade_logs_enabled(),
        "OPTIMIZER_ACTIVE_REPLAY_INCLUDE_PIT_STATS_INDEX": is_optimizer_active_replay_include_pit_stats_index_enabled(),
        "OPTIMIZER_ACTIVE_REPLAY_USE_PREPARED_CACHE": is_optimizer_active_replay_use_prepared_cache_enabled(),
        "OPTIMIZER_ACTIVE_REPLAY_WRITE_PREPARED_CACHE": is_optimizer_active_replay_write_prepared_cache_enabled(),
    }
