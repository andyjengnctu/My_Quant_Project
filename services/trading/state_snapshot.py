"""Scoped, immutable account/order/pending inputs for read-model projection.

The application captures these inputs under the Trading mutation lock. Each
worker binds that captured value explicitly; ordinary callers keep live reads.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType


_READ_SNAPSHOT = ContextVar("trading_state_read_snapshot", default=None)


@dataclass(frozen=True)
class TradingStateSnapshot:
    project_root: Path
    values: object
    fill_transaction_pending: bool

    @classmethod
    def capture(cls, project_root, loaders, *, fill_transaction_pending=False):
        values = {}
        for key, loader in loaders.items():
            try:
                values[key] = (True, deepcopy(loader()))
            except Exception as exc:
                values[key] = (False, exc)
        return cls(Path(project_root).resolve(), MappingProxyType(values), bool(fill_transaction_pending))


@contextmanager
def trading_snapshot_reads(snapshot):
    token = _READ_SNAPSHOT.set(snapshot)
    try:
        yield
    finally:
        _READ_SNAPSHOT.reset(token)


def snapshot_state_value(project_root, key, *, required=False):
    snapshot = _READ_SNAPSHOT.get()
    if snapshot is None or snapshot.project_root != Path(project_root).resolve():
        return False, None
    ok, value = snapshot.values[key]
    if not ok:
        raise value
    if value is None and required:
        raise FileNotFoundError(f"Trading {key} state 尚未初始化")
    return True, deepcopy(value)


def assert_live_trading_mutation():
    if _READ_SNAPSHOT.get() is not None:
        raise RuntimeError("Trading snapshot projection 不得修改持久狀態")
