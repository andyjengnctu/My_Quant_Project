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
    OPTIMIZER_SINGLE_FOLD_LOCAL_MIN_PROCESS_WORKERS,
    OPTIMIZER_SINGLE_FOLD_SEARCH_PARALLEL_TRIALS,
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
    }
