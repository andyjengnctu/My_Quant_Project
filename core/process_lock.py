"""Non-blocking, process-owned local mutexes backed by SQLite transactions."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sqlite3


@contextmanager
def try_process_lock(path: str | Path):
    """Yield whether this caller exclusively owns the lock until context exit.

    AI: Keep the lock database at a stable path. Never unlink it: a second inode
    would let two processes own different locks with the same filename. SQLite
    releases its transaction lock on close or process death, including Windows.
    No application records are stored in this database.
    """

    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(target), timeout=0, isolation_level=None)
    acquired = False
    try:
        try:
            connection.execute("BEGIN IMMEDIATE")
            acquired = True
        except sqlite3.OperationalError as exc:
            code = getattr(exc, "sqlite_errorcode", None)
            if code is None or (int(code) & 0xFF) not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                raise
        yield acquired
    finally:
        try:
            if acquired:
                connection.rollback()
        finally:
            connection.close()


__all__ = ["try_process_lock"]
