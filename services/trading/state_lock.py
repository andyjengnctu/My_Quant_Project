"""One mutation lock for Trading account, broker orders and fill recovery."""
from __future__ import annotations

from functools import wraps
from contextlib import contextmanager
from hashlib import sha256
import os
import threading

from core.process_lock import try_process_lock
from services.trading.state_snapshot import assert_live_trading_mutation
from core.trading_state_paths import (
    resolve_trading_state_mutation_lock_path,
    resolve_trading_fill_transaction_path,
    resolve_trading_pending_entry_transaction_path,
)


_OWNED_LOCKS = threading.local()
# AI: Account, order and fill-recovery state form one Trading mutation domain.
# Use one process-wide mutex so Windows path aliases/casing cannot let two
# threads enter the read-check-write window at the same time.  SQLite remains
# the cross-process exclusion layer.
_IN_PROCESS_TRADING_MUTATION_LOCK = threading.Lock()


class TradingStateBusyError(RuntimeError):
    pass


class TradingTransactionPendingError(RuntimeError):
    pass


def assert_no_trading_fill_transaction(project_root):
    """Readers and writers share the unfinished broker-fill boundary."""
    if resolve_trading_fill_transaction_path(project_root).exists():
        raise TradingTransactionPendingError(
            "Trading 尚有未完成 fill transaction；請先由 Workbench 重新整理以完成 recovery"
        )


def assert_trading_state_writable(project_root):
    """Transaction integrity is independent of active broker-order policy.

    Only the pending transfer that owns the exact durable journal may perform
    its account/pending writes. An unrelated public mutation always fails first.
    """
    assert_no_trading_fill_transaction(project_root)
    path = resolve_trading_pending_entry_transaction_path(project_root)
    if path.exists():
        authority = getattr(_OWNED_LOCKS, "pending_transaction", None)
        if authority != (_canonical_lock_key(project_root), sha256(path.read_bytes()).hexdigest()):
            raise TradingTransactionPendingError(
                "Trading 尚有未完成掛單轉持股 transaction；請先重新整理完成 recovery"
            )


@contextmanager
def pending_entry_transaction_access(project_root):
    """Internal commit/recovery capability, scoped to an already-owned lock."""
    key = _canonical_lock_key(project_root)
    if getattr(_OWNED_LOCKS, "path_key", None) != key:
        raise TradingStateBusyError("Pending transaction access requires the Trading mutation lock")
    assert_no_trading_fill_transaction(project_root)
    path = resolve_trading_pending_entry_transaction_path(project_root)
    authority = (key, sha256(path.read_bytes()).hexdigest())
    previous = getattr(_OWNED_LOCKS, "pending_transaction", None)
    _OWNED_LOCKS.pending_transaction = authority
    try:
        yield
    finally:
        if previous is None:
            del _OWNED_LOCKS.pending_transaction
        else:
            _OWNED_LOCKS.pending_transaction = previous


def _canonical_lock_key(project_root) -> str:
    path = resolve_trading_state_mutation_lock_path(project_root).resolve()
    return os.path.normcase(os.path.realpath(os.fspath(path)))


def serialized_trading_state_mutation(function):
    @wraps(function)
    def wrapped(project_root, *args, **kwargs):
        assert_live_trading_mutation()
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


__all__ = [
    "TradingStateBusyError", "TradingTransactionPendingError",
    "serialized_trading_state_mutation", "assert_trading_state_writable",
    "assert_no_trading_fill_transaction", "pending_entry_transaction_access",
]
