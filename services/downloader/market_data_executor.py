"""Quota-aware resumable execution core for Market Data V2 bootstrap jobs.

Execution remains storage-agnostic, but a sink may expose ``recover_committed``
so a crash after atomic publication and before ledger DONE can resume without
issuing the same provider data request again.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import time
from typing import Callable
from uuid import uuid4

import pandas as pd

from core.market_data_bootstrap_requests import BootstrapHttpRequest, BootstrapRequestManifest
from core.market_data_execution_policy import MarketDataExecutionPolicy, get_market_data_execution_policy
from core.market_data_storage_contract import MarketDataCommitError, MarketDataCommitReceipt
from services.downloader.finmind_http import FinMindHttpClient, FinMindHttpError, FinMindUsage
from services.downloader.market_data_ledger import (
    JOB_BLOCKED,
    WORKLOAD_BLOCKED,
    WORKLOAD_DONE,
    WORKLOAD_RUNNING,
    WORKLOAD_WAIT_QUOTA,
    LedgerSummary,
    MarketDataJobLedger,
)


@dataclass
class _QuotaState:
    usage: FinMindUsage | None = None
    local_data_attempts_since_refresh: int = 0


class MarketDataBootstrapExecutor:
    def __init__(
        self,
        *,
        ledger: MarketDataJobLedger,
        client: FinMindHttpClient,
        policy: MarketDataExecutionPolicy | None = None,
        now_fn: Callable[[], datetime] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        owner_id: str | None = None,
        blocking_waits: bool = True,
        quota_wait_observer: Callable[[dict[str, object]], None] | None = None,
    ):
        self.ledger = ledger
        self.client = client
        self.policy = policy or get_market_data_execution_policy()
        self.now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        self.sleep_fn = sleep_fn or time.sleep
        self.owner_id = str(owner_id or f"executor-{uuid4().hex}")
        self.blocking_waits = bool(blocking_waits)
        self.quota_wait_observer = quota_wait_observer
        self._quota = _QuotaState()
        self._quota_wait_started_at: datetime | None = None
        self._quota_wait_accumulated_seconds = 0.0

    def _now(self) -> datetime:
        value = self.now_fn()
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _lock_until(self, now: datetime) -> datetime:
        return now + timedelta(seconds=self.policy.executor_lock_seconds)

    def _job_lease_until(self, now: datetime) -> datetime:
        return now + timedelta(seconds=self.policy.job_lease_seconds)

    def _renew_lock(self, workload_id: str) -> None:
        now = self._now()
        self.ledger.renew_executor_lock(
            workload_id,
            owner_id=self.owner_id,
            now=now,
            lease_until=self._lock_until(now),
        )

    def _refresh_usage(self) -> FinMindUsage:
        usage = self.client.get_usage()
        self._quota.usage = usage
        self._quota.local_data_attempts_since_refresh = 0
        return usage

    def _quota_wait_seconds(self) -> float:
        total = float(self._quota_wait_accumulated_seconds)
        if self._quota_wait_started_at is not None:
            total += max(0.0, (self._now() - self._quota_wait_started_at).total_seconds())
        return total

    def _begin_quota_wait(self) -> None:
        if self._quota_wait_started_at is None:
            self._quota_wait_started_at = self._now()

    def _end_quota_wait(self) -> None:
        if self._quota_wait_started_at is None:
            return
        self._quota_wait_accumulated_seconds += max(
            0.0,
            (self._now() - self._quota_wait_started_at).total_seconds(),
        )
        self._quota_wait_started_at = None

    def _emit_quota_wait(
        self,
        workload_id: str,
        *,
        reason: str,
        error: str | None = None,
    ) -> None:
        observer = self.quota_wait_observer
        if observer is None:
            return
        usage = self._quota.usage
        summary = self.ledger.get_summary(workload_id)
        event: dict[str, object] = {
            "kind": "WAIT_QUOTA",
            "reason": str(reason),
            "done": int(summary.done),
            "total": int(summary.total),
            "waited_seconds": self._quota_wait_seconds(),
            "poll_seconds": float(self.policy.quota_poll_seconds),
            "error": error,
        }
        if usage is not None:
            limit = int(usage.api_request_limit)
            reserve = self._effective_reserve(limit)
            effective_used = int(usage.user_count) + int(self._quota.local_data_attempts_since_refresh)
            stop_used_max = max(0, limit - reserve - 1)
            resume_headroom = self._effective_resume_headroom(limit)
            resume_used_max = max(0, limit - reserve - resume_headroom)
            remaining = int(limit - effective_used)
            event.update(
                {
                    "quota_user_count": effective_used,
                    "quota_limit": limit,
                    "quota_remaining": remaining,
                    "quota_usable_remaining": max(0, remaining - reserve),
                    "quota_reserve": reserve,
                    "quota_safe_used_max": stop_used_max,
                    "quota_resume_headroom": resume_headroom,
                    "quota_resume_used_max": resume_used_max,
                    "quota_needed_drop": max(0, effective_used - resume_used_max),
                }
            )
        observer(event)

    def _effective_reserve(self, limit: int) -> int:
        return min(self.policy.quota_reserve_requests, max(0, int(limit) - 1))

    def _effective_resume_headroom(self, limit: int) -> int:
        reserve = self._effective_reserve(limit)
        return min(
            self.policy.quota_resume_headroom_requests,
            max(1, int(limit) - reserve),
        )

    def _estimated_remaining(self) -> int:
        usage = self._quota.usage
        if usage is None:
            return 0
        return int(usage.api_request_limit - usage.user_count - self._quota.local_data_attempts_since_refresh)

    def _quota_has_capacity(self) -> bool:
        usage = self._quota.usage
        if usage is None:
            return False
        return self._estimated_remaining() > self._effective_reserve(usage.api_request_limit)

    def _quota_has_resume_capacity(self) -> bool:
        usage = self._quota.usage
        if usage is None:
            return False
        limit = int(usage.api_request_limit)
        reserve = self._effective_reserve(limit)
        headroom = self._effective_resume_headroom(limit)
        usable = self._estimated_remaining() - reserve
        return usable >= headroom

    def quota_progress_snapshot(self) -> dict[str, int | None]:
        """Return the executor's best current quota estimate for UI progress only.

        This is deliberately observational: it does not refresh provider usage and
        therefore cannot change execution cadence or consume an extra usage request.
        Local data attempts since the last live refresh are deducted so progress
        output remains conservative between refreshes.
        """

        usage = self._quota.usage
        if usage is None:
            return {
                "quota_limit": None,
                "quota_remaining": None,
                "quota_usable_remaining": None,
                "quota_reserve": None,
                "quota_wait_seconds": self._quota_wait_seconds(),
            }
        limit = int(usage.api_request_limit)
        remaining = max(0, self._estimated_remaining())
        reserve = self._effective_reserve(limit)
        return {
            "quota_limit": limit,
            "quota_remaining": remaining,
            "quota_usable_remaining": max(0, remaining - reserve),
            "quota_reserve": reserve,
            "quota_wait_seconds": self._quota_wait_seconds(),
        }

    def _refresh_usage_with_transient_wait(self, workload_id: str) -> FinMindUsage:
        while True:
            try:
                return self._refresh_usage()
            except FinMindHttpError as exc:
                if not exc.retryable:
                    raise
                self._begin_quota_wait()
                self.ledger.set_workload_status(workload_id, status=WORKLOAD_WAIT_QUOTA, now=self._now())
                self._renew_lock(workload_id)
                self._emit_quota_wait(
                    workload_id,
                    reason="usage_refresh_retry",
                    error=f"{type(exc).__name__}: {exc}",
                )
                self.sleep_fn(self.policy.quota_poll_seconds)

    def _ensure_quota_capacity(self, workload_id: str) -> bool:
        should_refresh = (
            self._quota.usage is None
            or self._quota.local_data_attempts_since_refresh >= self.policy.quota_refresh_every_requests
            or not self._quota_has_capacity()
        )
        if should_refresh:
            if self.blocking_waits:
                self._refresh_usage_with_transient_wait(workload_id)
            else:
                self._refresh_usage()
        if not self._quota_has_capacity():
            self._begin_quota_wait()
            self.ledger.set_workload_status(workload_id, status=WORKLOAD_WAIT_QUOTA, now=self._now())
            if not self.blocking_waits:
                self._emit_quota_wait(workload_id, reason="quota_capacity")
                return False
            while not self._quota_has_resume_capacity():
                self._renew_lock(workload_id)
                self._emit_quota_wait(workload_id, reason="quota_capacity")
                self.sleep_fn(self.policy.quota_poll_seconds)
                self._refresh_usage_with_transient_wait(workload_id)
        self._end_quota_wait()
        self.ledger.set_workload_status(workload_id, status=WORKLOAD_RUNNING, now=self._now())
        return True

    def _record_data_attempt(self, workload_id: str, request_id: str) -> None:
        self.ledger.record_http_attempt(workload_id, request_id, now=self._now())
        self._quota.local_data_attempts_since_refresh += 1

    def _sleep_until_retry_ready(self, workload_id: str) -> bool:
        raw = self.ledger.next_retry_at(workload_id)
        if not raw:
            return False
        target = datetime.fromisoformat(str(raw))
        if target.tzinfo is None:
            target = target.replace(tzinfo=timezone.utc)
        delay = max(0.0, (target.astimezone(timezone.utc) - self._now()).total_seconds())
        if delay > 0:
            # Renew before/after the wait. Retry backoff is bounded by config and
            # may exceed the executor lock lease, so long waits are split.
            remaining = delay
            while remaining > 0:
                self._renew_lock(workload_id)
                step = min(remaining, max(0.1, self.policy.executor_lock_seconds / 2.0))
                self.sleep_fn(step)
                remaining -= step
        return True

    def _fetch(self, request: BootstrapHttpRequest) -> pd.DataFrame:
        return self.client.get_data(
            dataset=request.dataset,
            data_id=request.data_id,
            start_date=request.start_date,
            end_date=request.end_date,
        )

    def run(
        self,
        *,
        manifest: BootstrapRequestManifest,
        sink: Callable[[BootstrapHttpRequest, pd.DataFrame], MarketDataCommitReceipt],
    ) -> LedgerSummary:
        workload_id = self.ledger.seed_manifest(manifest, now=self._now())
        now = self._now()
        self.ledger.acquire_executor_lock(
            workload_id,
            owner_id=self.owner_id,
            now=now,
            lease_until=self._lock_until(now),
        )
        self.ledger.recover_orphaned_running_jobs(workload_id, owner_id=self.owner_id, now=now)
        try:
            summary = self.ledger.get_summary(workload_id)
            if summary.blocked:
                return summary
            if summary.done == summary.total and summary.total == manifest.total_requests:
                self.ledger.set_workload_status(workload_id, status=WORKLOAD_DONE, now=self._now())
                return self.ledger.get_summary(workload_id)

            if not self._ensure_quota_capacity(workload_id):
                return self.ledger.get_summary(workload_id)
            while True:
                self._renew_lock(workload_id)
                summary = self.ledger.get_summary(workload_id)
                if summary.blocked:
                    self.ledger.set_workload_status(workload_id, status=WORKLOAD_BLOCKED, now=self._now())
                    return self.ledger.get_summary(workload_id)
                if summary.done == summary.total:
                    self.ledger.set_workload_status(workload_id, status=WORKLOAD_DONE, now=self._now())
                    return self.ledger.get_summary(workload_id)

                if not self._ensure_quota_capacity(workload_id):
                    return self.ledger.get_summary(workload_id)
                now = self._now()
                job = self.ledger.claim_next_job(
                    workload_id,
                    owner_id=self.owner_id,
                    now=now,
                    lease_until=self._job_lease_until(now),
                )
                if job is None:
                    if self.ledger.next_retry_at(workload_id) and not self.blocking_waits:
                        return self.ledger.get_summary(workload_id)
                    if self._sleep_until_retry_ready(workload_id):
                        continue
                    # No runnable job while work remains implies an invalid/stale
                    # ledger state; fail closed rather than spin forever.
                    raise RuntimeError("Market Data ledger 有 unfinished jobs，但沒有可執行或可等待的 request")

                request = job.to_request()
                recover_fn = getattr(sink, "recover_committed", None)
                if callable(recover_fn):
                    try:
                        recovered = recover_fn(request)
                    except MarketDataCommitError as exc:
                        self.ledger.mark_blocked(
                            workload_id,
                            job.request_id,
                            error_kind="commit_recovery",
                            message=str(exc),
                            now=self._now(),
                        )
                        return self.ledger.get_summary(workload_id)
                    if recovered is not None:
                        if not isinstance(recovered, MarketDataCommitReceipt) or not recovered.committed:
                            self.ledger.mark_blocked(
                                workload_id,
                                job.request_id,
                                error_kind="commit_recovery_contract",
                                message="Market Data sink recover_committed 未回傳合法 committed receipt",
                                now=self._now(),
                            )
                            return self.ledger.get_summary(workload_id)
                        self.ledger.mark_done(
                            workload_id,
                            job.request_id,
                            row_count=recovered.row_count,
                            content_sha256=recovered.content_sha256,
                            now=self._now(),
                        )
                        continue

                self._record_data_attempt(workload_id, job.request_id)
                try:
                    frame = self._fetch(request)
                except FinMindHttpError as exc:
                    if exc.quota_exhausted:
                        self.ledger.mark_pending_after_quota(
                            workload_id,
                            job.request_id,
                            message=str(exc),
                            now=self._now(),
                        )
                        self._quota.usage = None
                        self._quota.local_data_attempts_since_refresh = 0
                        if not self._ensure_quota_capacity(workload_id):
                            return self.ledger.get_summary(workload_id)
                        continue
                    if exc.retryable:
                        next_failure_count = job.retryable_failure_count + 1
                        if next_failure_count >= self.policy.max_retryable_attempts:
                            self.ledger.mark_blocked(
                                workload_id,
                                job.request_id,
                                error_kind="finmind_retry_exhausted",
                                message=str(exc),
                                now=self._now(),
                            )
                            return self.ledger.get_summary(workload_id)
                        delay = self.policy.retry_delay_seconds(next_failure_count)
                        self.ledger.mark_retryable(
                            workload_id,
                            job.request_id,
                            error_kind="finmind_transient",
                            message=str(exc),
                            not_before=self._now() + timedelta(seconds=delay),
                            now=self._now(),
                        )
                        continue
                    self.ledger.mark_blocked(
                        workload_id,
                        job.request_id,
                        error_kind="finmind_permanent",
                        message=str(exc),
                        now=self._now(),
                    )
                    return self.ledger.get_summary(workload_id)

                self._renew_lock(workload_id)
                try:
                    receipt = sink(request, frame)
                except MarketDataCommitError as exc:
                    self.ledger.mark_blocked(
                        workload_id,
                        job.request_id,
                        error_kind="commit",
                        message=str(exc),
                        now=self._now(),
                    )
                    return self.ledger.get_summary(workload_id)
                if not isinstance(receipt, MarketDataCommitReceipt) or not receipt.committed:
                    self.ledger.mark_blocked(
                        workload_id,
                        job.request_id,
                        error_kind="commit_contract",
                        message="Market Data sink 未回傳 committed receipt",
                        now=self._now(),
                    )
                    return self.ledger.get_summary(workload_id)
                if int(receipt.row_count) != int(len(frame)):
                    self.ledger.mark_blocked(
                        workload_id,
                        job.request_id,
                        error_kind="commit_row_count",
                        message=f"sink row_count={receipt.row_count} != fetched rows={len(frame)}",
                        now=self._now(),
                    )
                    return self.ledger.get_summary(workload_id)
                self.ledger.mark_done(
                    workload_id,
                    job.request_id,
                    row_count=receipt.row_count,
                    content_sha256=receipt.content_sha256,
                    now=self._now(),
                )
        finally:
            self.ledger.release_executor_lock(workload_id, owner_id=self.owner_id)


__all__ = [
    "MarketDataCommitReceipt",
    "MarketDataCommitError",
    "MarketDataBootstrapExecutor",
]
