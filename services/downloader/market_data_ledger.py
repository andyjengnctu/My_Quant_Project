"""Persistent SQLite execution ledger for Market Data V2 bootstrap jobs."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Iterable

from core.market_data_bootstrap_requests import BootstrapHttpRequest, BootstrapRequestManifest

JOB_PENDING = "PENDING"
JOB_RUNNING = "RUNNING"
JOB_RETRYABLE = "RETRYABLE"
JOB_DONE = "DONE"
JOB_BLOCKED = "BLOCKED"

WORKLOAD_READY = "READY"
WORKLOAD_RUNNING = "RUNNING"
WORKLOAD_WAIT_QUOTA = "WAIT_QUOTA"
WORKLOAD_BLOCKED = "BLOCKED"
WORKLOAD_DONE = "DONE"


@dataclass(frozen=True)
class LedgerJob:
    workload_id: str
    request_id: str
    ordinal: int
    dataset: str
    bootstrap_mode: str
    data_id: str | None
    start_date: str | None
    end_date: str | None
    status: str
    attempt_count: int
    http_attempt_count: int
    retryable_failure_count: int

    def to_request(self) -> BootstrapHttpRequest:
        return BootstrapHttpRequest(
            dataset=self.dataset,
            bootstrap_mode=self.bootstrap_mode,
            data_id=self.data_id,
            start_date=self.start_date,
            end_date=self.end_date,
        )


@dataclass(frozen=True)
class LedgerSummary:
    workload_id: str
    workload_status: str
    total: int
    pending: int
    running: int
    retryable: int
    done: int
    blocked: int
    http_attempts: int

    @property
    def unfinished(self) -> int:
        return self.pending + self.running + self.retryable


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None = None) -> str:
    resolved = _utc_now() if value is None else value
    if resolved.tzinfo is None:
        resolved = resolved.replace(tzinfo=timezone.utc)
    return resolved.astimezone(timezone.utc).isoformat()


def _row_to_job(row: sqlite3.Row) -> LedgerJob:
    return LedgerJob(
        workload_id=str(row["workload_id"]),
        request_id=str(row["request_id"]),
        ordinal=int(row["ordinal"]),
        dataset=str(row["dataset"]),
        bootstrap_mode=str(row["bootstrap_mode"]),
        data_id=row["data_id"],
        start_date=row["start_date"],
        end_date=row["end_date"],
        status=str(row["status"]),
        attempt_count=int(row["attempt_count"]),
        http_attempt_count=int(row["http_attempt_count"]),
        retryable_failure_count=int(row["retryable_failure_count"]),
    )


class MarketDataJobLedger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn


    @contextmanager
    def _connection(self):
        conn = self._connect()
        try:
            yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS workloads (
                    workload_id TEXT PRIMARY KEY,
                    manifest_fingerprint TEXT NOT NULL,
                    registry_fingerprint TEXT NOT NULL,
                    as_of_date TEXT NOT NULL,
                    expected_jobs INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    blocked_reason TEXT
                );

                CREATE TABLE IF NOT EXISTS jobs (
                    workload_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    dataset TEXT NOT NULL,
                    bootstrap_mode TEXT NOT NULL,
                    data_id TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    status TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    http_attempt_count INTEGER NOT NULL DEFAULT 0,
                    retryable_failure_count INTEGER NOT NULL DEFAULT 0,
                    row_count INTEGER,
                    content_sha256 TEXT,
                    error_kind TEXT,
                    error_message TEXT,
                    not_before TEXT,
                    lease_owner TEXT,
                    lease_until TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (workload_id, request_id),
                    UNIQUE (workload_id, ordinal),
                    FOREIGN KEY (workload_id) REFERENCES workloads(workload_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS executor_locks (
                    workload_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    lease_until TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (workload_id) REFERENCES workloads(workload_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_market_data_jobs_ready
                    ON jobs(workload_id, status, not_before, ordinal);
                """
            )

    @staticmethod
    def workload_id_for_manifest(manifest: BootstrapRequestManifest) -> str:
        return f"market_data_v2_bootstrap:{manifest.manifest_fingerprint}"

    def seed_manifest(self, manifest: BootstrapRequestManifest, *, now: datetime | None = None) -> str:
        workload_id = self.workload_id_for_manifest(manifest)
        now_iso = _iso(now)
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT * FROM workloads WHERE workload_id = ?",
                (workload_id,),
            ).fetchone()
            if existing is None:
                conn.execute(
                    """
                    INSERT INTO workloads(
                        workload_id, manifest_fingerprint, registry_fingerprint, as_of_date,
                        expected_jobs, status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        workload_id,
                        manifest.manifest_fingerprint,
                        manifest.registry_fingerprint,
                        manifest.as_of_date,
                        manifest.total_requests,
                        WORKLOAD_READY,
                        now_iso,
                        now_iso,
                    ),
                )
            else:
                if str(existing["manifest_fingerprint"]) != manifest.manifest_fingerprint:
                    raise ValueError("既有 workload manifest fingerprint 不一致")
                if int(existing["expected_jobs"]) != manifest.total_requests:
                    raise ValueError("既有 workload expected_jobs 與 request manifest 不一致")

            for ordinal, request in enumerate(manifest.requests):
                conn.execute(
                    """
                    INSERT OR IGNORE INTO jobs(
                        workload_id, request_id, ordinal, dataset, bootstrap_mode,
                        data_id, start_date, end_date, status, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        workload_id,
                        request.request_id,
                        ordinal,
                        request.dataset,
                        request.bootstrap_mode,
                        request.data_id,
                        request.start_date,
                        request.end_date,
                        JOB_PENDING,
                        now_iso,
                    ),
                )
            actual_count = int(
                conn.execute("SELECT COUNT(*) FROM jobs WHERE workload_id = ?", (workload_id,)).fetchone()[0]
            )
            if actual_count != manifest.total_requests:
                raise ValueError(
                    f"Ledger job count 與 manifest 不一致: actual={actual_count}, expected={manifest.total_requests}"
                )
            conn.execute("UPDATE workloads SET updated_at = ? WHERE workload_id = ?", (now_iso, workload_id))
            conn.execute("COMMIT")
        return workload_id

    def acquire_executor_lock(
        self,
        workload_id: str,
        *,
        owner_id: str,
        now: datetime,
        lease_until: datetime,
    ) -> None:
        now_iso = _iso(now)
        lease_iso = _iso(lease_until)
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT owner_id, lease_until FROM executor_locks WHERE workload_id = ?",
                (workload_id,),
            ).fetchone()
            if row is not None and str(row["owner_id"]) != owner_id and str(row["lease_until"]) > now_iso:
                raise RuntimeError(
                    f"Market Data workload 已有 active executor: owner={row['owner_id']}, lease_until={row['lease_until']}"
                )
            conn.execute(
                """
                INSERT INTO executor_locks(workload_id, owner_id, lease_until, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(workload_id) DO UPDATE SET
                    owner_id=excluded.owner_id,
                    lease_until=excluded.lease_until,
                    updated_at=excluded.updated_at
                """,
                (workload_id, owner_id, lease_iso, now_iso),
            )
            conn.execute("COMMIT")

    def renew_executor_lock(
        self,
        workload_id: str,
        *,
        owner_id: str,
        now: datetime,
        lease_until: datetime,
    ) -> None:
        now_iso = _iso(now)
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            updated = conn.execute(
                """
                UPDATE executor_locks
                SET lease_until = ?, updated_at = ?
                WHERE workload_id = ? AND owner_id = ?
                """,
                (_iso(lease_until), now_iso, workload_id, owner_id),
            ).rowcount
            if updated != 1:
                raise RuntimeError("Market Data executor lock 已遺失或被其他 executor 取代")
            conn.execute("COMMIT")

    def release_executor_lock(self, workload_id: str, *, owner_id: str) -> None:
        with self._connection() as conn:
            conn.execute(
                "DELETE FROM executor_locks WHERE workload_id = ? AND owner_id = ?",
                (workload_id, owner_id),
            )

    def recover_orphaned_running_jobs(
        self,
        workload_id: str,
        *,
        owner_id: str,
        now: datetime,
    ) -> int:
        """Recover RUNNING jobs after this executor legitimately owns the workload lock.

        Once ``acquire_executor_lock`` succeeds, any RUNNING job leased by a
        different executor belongs to a crashed/expired owner.  Reset it
        immediately instead of waiting for the longer per-job lease to expire.
        """
        now_iso = _iso(now)
        with self._connection() as conn:
            updated = conn.execute(
                """
                UPDATE jobs
                SET status = ?, lease_owner = NULL, lease_until = NULL, updated_at = ?
                WHERE workload_id = ? AND status = ?
                  AND (lease_owner IS NULL OR lease_owner != ?)
                """,
                (JOB_PENDING, now_iso, workload_id, JOB_RUNNING, owner_id),
            ).rowcount
        return int(updated)

    def claim_next_job(
        self,
        workload_id: str,
        *,
        owner_id: str,
        now: datetime,
        lease_until: datetime,
    ) -> LedgerJob | None:
        now_iso = _iso(now)
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, lease_owner = NULL, lease_until = NULL, updated_at = ?
                WHERE workload_id = ? AND status = ? AND lease_until IS NOT NULL AND lease_until <= ?
                """,
                (JOB_PENDING, now_iso, workload_id, JOB_RUNNING, now_iso),
            )
            row = conn.execute(
                """
                SELECT * FROM jobs
                WHERE workload_id = ?
                  AND (
                    status = ? OR
                    (status = ? AND (not_before IS NULL OR not_before <= ?))
                  )
                ORDER BY ordinal
                LIMIT 1
                """,
                (workload_id, JOB_PENDING, JOB_RETRYABLE, now_iso),
            ).fetchone()
            if row is None:
                conn.execute("COMMIT")
                return None
            request_id = str(row["request_id"])
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, attempt_count = attempt_count + 1,
                    lease_owner = ?, lease_until = ?, started_at = COALESCE(started_at, ?),
                    not_before = NULL, updated_at = ?
                WHERE workload_id = ? AND request_id = ?
                """,
                (
                    JOB_RUNNING,
                    owner_id,
                    _iso(lease_until),
                    now_iso,
                    now_iso,
                    workload_id,
                    request_id,
                ),
            )
            conn.execute(
                "UPDATE workloads SET status = ?, updated_at = ? WHERE workload_id = ? AND status != ?",
                (WORKLOAD_RUNNING, now_iso, workload_id, WORKLOAD_BLOCKED),
            )
            claimed = conn.execute(
                "SELECT * FROM jobs WHERE workload_id = ? AND request_id = ?",
                (workload_id, request_id),
            ).fetchone()
            conn.execute("COMMIT")
            return _row_to_job(claimed)

    def record_http_attempt(self, workload_id: str, request_id: str, *, now: datetime) -> None:
        with self._connection() as conn:
            updated = conn.execute(
                """
                UPDATE jobs
                SET http_attempt_count = http_attempt_count + 1, updated_at = ?
                WHERE workload_id = ? AND request_id = ? AND status = ?
                """,
                (_iso(now), workload_id, request_id, JOB_RUNNING),
            ).rowcount
            if updated != 1:
                raise RuntimeError("只能對 RUNNING Market Data job 記錄 HTTP attempt")

    def mark_done(
        self,
        workload_id: str,
        request_id: str,
        *,
        row_count: int,
        content_sha256: str | None,
        now: datetime,
    ) -> None:
        now_iso = _iso(now)
        with self._connection() as conn:
            updated = conn.execute(
                """
                UPDATE jobs
                SET status = ?, row_count = ?, content_sha256 = ?, error_kind = NULL,
                    error_message = NULL, lease_owner = NULL, lease_until = NULL,
                    completed_at = ?, updated_at = ?
                WHERE workload_id = ? AND request_id = ? AND status = ?
                """,
                (
                    JOB_DONE,
                    int(row_count),
                    content_sha256,
                    now_iso,
                    now_iso,
                    workload_id,
                    request_id,
                    JOB_RUNNING,
                ),
            ).rowcount
            if updated != 1:
                raise RuntimeError("只能完成 RUNNING Market Data job")

    def mark_pending_after_quota(
        self,
        workload_id: str,
        request_id: str,
        *,
        message: str,
        now: datetime,
    ) -> None:
        now_iso = _iso(now)
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            updated = conn.execute(
                """
                UPDATE jobs
                SET status = ?, error_kind = 'quota', error_message = ?, not_before = NULL,
                    lease_owner = NULL, lease_until = NULL, updated_at = ?
                WHERE workload_id = ? AND request_id = ? AND status = ?
                """,
                (JOB_PENDING, str(message), now_iso, workload_id, request_id, JOB_RUNNING),
            ).rowcount
            if updated != 1:
                raise RuntimeError("只能將 RUNNING job 退回 quota pending")
            conn.execute(
                "UPDATE workloads SET status = ?, updated_at = ? WHERE workload_id = ?",
                (WORKLOAD_WAIT_QUOTA, now_iso, workload_id),
            )
            conn.execute("COMMIT")

    def mark_retryable(
        self,
        workload_id: str,
        request_id: str,
        *,
        error_kind: str,
        message: str,
        not_before: datetime,
        now: datetime,
    ) -> None:
        with self._connection() as conn:
            updated = conn.execute(
                """
                UPDATE jobs
                SET status = ?, retryable_failure_count = retryable_failure_count + 1,
                    error_kind = ?, error_message = ?, not_before = ?,
                    lease_owner = NULL, lease_until = NULL, updated_at = ?
                WHERE workload_id = ? AND request_id = ? AND status = ?
                """,
                (
                    JOB_RETRYABLE,
                    str(error_kind),
                    str(message),
                    _iso(not_before),
                    _iso(now),
                    workload_id,
                    request_id,
                    JOB_RUNNING,
                ),
            ).rowcount
            if updated != 1:
                raise RuntimeError("只能將 RUNNING Market Data job 標記 RETRYABLE")

    def mark_blocked(
        self,
        workload_id: str,
        request_id: str,
        *,
        error_kind: str,
        message: str,
        now: datetime,
    ) -> None:
        now_iso = _iso(now)
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            updated = conn.execute(
                """
                UPDATE jobs
                SET status = ?, error_kind = ?, error_message = ?, lease_owner = NULL,
                    lease_until = NULL, updated_at = ?
                WHERE workload_id = ? AND request_id = ? AND status = ?
                """,
                (
                    JOB_BLOCKED,
                    str(error_kind),
                    str(message),
                    now_iso,
                    workload_id,
                    request_id,
                    JOB_RUNNING,
                ),
            ).rowcount
            if updated != 1:
                raise RuntimeError("只能將 RUNNING Market Data job 標記 BLOCKED")
            conn.execute(
                """
                UPDATE workloads
                SET status = ?, blocked_reason = ?, updated_at = ?
                WHERE workload_id = ?
                """,
                (WORKLOAD_BLOCKED, str(message), now_iso, workload_id),
            )
            conn.execute("COMMIT")

    def set_workload_status(self, workload_id: str, *, status: str, now: datetime) -> None:
        with self._connection() as conn:
            conn.execute(
                "UPDATE workloads SET status = ?, updated_at = ? WHERE workload_id = ?",
                (str(status), _iso(now), workload_id),
            )

    def next_retry_at(self, workload_id: str) -> str | None:
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT MIN(not_before) AS next_retry
                FROM jobs
                WHERE workload_id = ? AND status = ? AND not_before IS NOT NULL
                """,
                (workload_id, JOB_RETRYABLE),
            ).fetchone()
        return None if row is None else row["next_retry"]

    def get_summary(self, workload_id: str) -> LedgerSummary:
        with self._connection() as conn:
            workload = conn.execute(
                "SELECT status FROM workloads WHERE workload_id = ?",
                (workload_id,),
            ).fetchone()
            if workload is None:
                raise ValueError(f"未知 Market Data workload: {workload_id}")
            rows = conn.execute(
                """
                SELECT status, COUNT(*) AS n, COALESCE(SUM(http_attempt_count), 0) AS http_attempts
                FROM jobs WHERE workload_id = ? GROUP BY status
                """,
                (workload_id,),
            ).fetchall()
        counts = {str(row["status"]): int(row["n"]) for row in rows}
        total_http_attempts = sum(int(row["http_attempts"]) for row in rows)
        total = sum(counts.values())
        return LedgerSummary(
            workload_id=workload_id,
            workload_status=str(workload["status"]),
            total=total,
            pending=counts.get(JOB_PENDING, 0),
            running=counts.get(JOB_RUNNING, 0),
            retryable=counts.get(JOB_RETRYABLE, 0),
            done=counts.get(JOB_DONE, 0),
            blocked=counts.get(JOB_BLOCKED, 0),
            http_attempts=total_http_attempts,
        )

    def get_job(self, workload_id: str, request_id: str) -> LedgerJob:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE workload_id = ? AND request_id = ?",
                (workload_id, request_id),
            ).fetchone()
        if row is None:
            raise ValueError(f"未知 Market Data job: {request_id}")
        return _row_to_job(row)


__all__ = [
    "JOB_PENDING",
    "JOB_RUNNING",
    "JOB_RETRYABLE",
    "JOB_DONE",
    "JOB_BLOCKED",
    "WORKLOAD_READY",
    "WORKLOAD_RUNNING",
    "WORKLOAD_WAIT_QUOTA",
    "WORKLOAD_BLOCKED",
    "WORKLOAD_DONE",
    "LedgerJob",
    "LedgerSummary",
    "MarketDataJobLedger",
]
