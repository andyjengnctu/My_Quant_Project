"""One mutation lock for Trading account, broker orders and fill recovery."""
from __future__ import annotations

from functools import wraps
import os
import threading

from core.process_lock import try_process_lock
from core.trading_state_paths import resolve_trading_state_mutation_lock_path


_OWNED_LOCKS = threading.local()


class TradingStateBusyError(RuntimeError):
    pass


def serialized_trading_state_mutation(function):
    @wraps(function)
    def wrapped(project_root, *args, **kwargs):
        path = resolve_trading_state_mutation_lock_path(project_root).resolve()
        key = (os.getpid(), str(path))
        owned = getattr(_OWNED_LOCKS, "paths", None)
        if owned is None:
            owned = _OWNED_LOCKS.paths = set()
        # AI: A fill commit calls recovery in the same thread. Only that owner
        # may re-enter; another thread/process must fail before reading state.
        if key in owned:
            return function(project_root, *args, **kwargs)
        with try_process_lock(path) as acquired:
            if not acquired:
                raise TradingStateBusyError("Trading 帳務／掛單正在更新；請等待完成並刷新狀態後重試")
            owned.add(key)
            try:
                return function(project_root, *args, **kwargs)
            finally:
                owned.remove(key)

    return wrapped


__all__ = ["TradingStateBusyError", "serialized_trading_state_mutation"]
