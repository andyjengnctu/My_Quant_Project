"""One mutation lock for Trading account, broker orders and fill recovery."""
from __future__ import annotations

from functools import wraps
import os
import threading

from core.process_lock import try_process_lock
from core.trading_state_paths import resolve_trading_state_mutation_lock_path


_OWNED_LOCKS = threading.local()
_THREAD_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS: dict[str, tuple[threading.Lock, int]] = {}


class TradingStateBusyError(RuntimeError):
    pass


def _acquire_thread_lock(path_key: str) -> tuple[threading.Lock, bool]:
    # AI: SQLite owns cross-process exclusion, but its same-process / multi-thread
    # lock behavior is platform-dependent.  Keep a per-path in-process mutex so a
    # competing thread is rejected before it can read a stale Trading revision.
    with _THREAD_LOCKS_GUARD:
        entry = _THREAD_LOCKS.get(path_key)
        if entry is None:
            lock = threading.Lock()
            refs = 0
        else:
            lock, refs = entry
        _THREAD_LOCKS[path_key] = (lock, refs + 1)

    acquired = lock.acquire(blocking=False)
    if not acquired:
        _release_thread_lock_ref(path_key, lock, release_owned=False)
    return lock, acquired


def _release_thread_lock_ref(path_key: str, lock: threading.Lock, *, release_owned: bool) -> None:
    if release_owned:
        lock.release()
    with _THREAD_LOCKS_GUARD:
        entry = _THREAD_LOCKS.get(path_key)
        if entry is None:
            return
        current_lock, refs = entry
        if current_lock is not lock:
            return
        remaining = refs - 1
        if remaining <= 0:
            _THREAD_LOCKS.pop(path_key, None)
        else:
            _THREAD_LOCKS[path_key] = (lock, remaining)


def serialized_trading_state_mutation(function):
    @wraps(function)
    def wrapped(project_root, *args, **kwargs):
        path = resolve_trading_state_mutation_lock_path(project_root).resolve()
        path_key = str(path)
        key = (os.getpid(), path_key)
        owned = getattr(_OWNED_LOCKS, "paths", None)
        if owned is None:
            owned = _OWNED_LOCKS.paths = set()
        # AI: A fill commit calls recovery in the same thread. Only that owner
        # may re-enter; another thread/process must fail before reading state.
        if key in owned:
            return function(project_root, *args, **kwargs)

        thread_lock, thread_acquired = _acquire_thread_lock(path_key)
        if not thread_acquired:
            raise TradingStateBusyError("Trading 帳務／掛單正在更新；請等待完成並刷新狀態後重試")
        try:
            with try_process_lock(path) as acquired:
                if not acquired:
                    raise TradingStateBusyError("Trading 帳務／掛單正在更新；請等待完成並刷新狀態後重試")
                owned.add(key)
                try:
                    return function(project_root, *args, **kwargs)
                finally:
                    owned.remove(key)
        finally:
            _release_thread_lock_ref(path_key, thread_lock, release_owned=True)

    return wrapped


__all__ = ["TradingStateBusyError", "serialized_trading_state_mutation"]
