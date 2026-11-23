"""Concurrent regression probes may run together. Cleanup alone needs an exclusive lease."""
from contextlib import contextmanager
import sqlite3


@contextmanager
def fixture_access(staging, *, cleanup):
    staging.mkdir(parents=True, exist_ok=True)
    lock = staging.parent.parent / "project_cleanup/fixture_use.sqlite"
    if lock.parent.resolve() != lock.parent.absolute():
        raise OSError("測試暫存鎖目錄異常")
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.resolve() != lock.absolute() or (lock.exists() and lock.stat().st_nlink != 1):
        raise OSError("測試暫存鎖路徑異常")
    db = sqlite3.connect(lock, timeout=0 if cleanup else 30, isolation_level=None)
    try:
        db.execute("BEGIN EXCLUSIVE" if cleanup else "BEGIN")
        db.execute("SELECT count(*) FROM sqlite_schema").fetchone()
        yield
    finally:
        db.close()
