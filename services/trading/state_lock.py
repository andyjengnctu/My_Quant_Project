"""One mutation lock for Trading account, broker orders and fill recovery."""
from __future__ import annotations

from functools import wraps
import os
import threading

from core.process_lock import try_process_lock
from core.trading_state_paths import resolve_trading_state_mutation_lock_path


_OWNED_LOCKS = threading.local()
# AI: Account, order and fill-recovery state form one Trading mutation domain.
# Use one process-wide mutex so Windows path aliases/casing cannot let two
# threads enter the read-check-write window at the same time.  SQLite remains
# the cross-process exclusion layer.
_IN_PROCESS_TRADING_MUTATION_LOCK = threading.Lock()


class TradingStateBusyError(RuntimeError):
    pass


def _canonical_lock_key(project_root) -> str:
    path = resolve_trading_state_mutation_lock_path(project_root).resolve()
    return os.path.normcase(os.path.realpath(os.fspath(path)))


def serialized_trading_state_mutation(function):
    @wraps(function)
    def wrapped(project_root, *args, **kwargs):
        path = resolve_trading_state_mutation_lock_path(project_root).resolve()
        path_key = _canonical_lock_key(project_root)
        owned_path_key = getattr(_OWNED_LOCKS, "path_key", None)

        # AI: Fill commit/recovery can legitimately re-enter in the same thread,
        # but only for the exact same Trading state domain.  A nested mutation of
        # another project root must never bypass either lock layer.
        if owned_path_key is not None:
            if owned_path_key == path_key:
                return function(project_root, *args, **kwargs)
            raise TradingStateBusyError(
                "Trading 帳務／掛單正在更新；同一執行緒不可同時修改另一個 Trading state"
            )

        if not _IN_PROCESS_TRADING_MUTATION_LOCK.acquire(blocking=False):
            raise TradingStateBusyError("Trading 帳務／掛單正在更新；請等待完成並刷新狀態後重試")
        try:
            with try_process_lock(path) as acquired:
                if not acquired:
                    raise TradingStateBusyError("Trading 帳務／掛單正在更新；請等待完成並刷新狀態後重試")
                _OWNED_LOCKS.path_key = path_key
                try:
                    return function(project_root, *args, **kwargs)
                finally:
                    del _OWNED_LOCKS.path_key
        finally:
            _IN_PROCESS_TRADING_MUTATION_LOCK.release()

    return wrapped


__all__ = ["TradingStateBusyError", "serialized_trading_state_mutation"]
