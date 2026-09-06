"""Typed execution policy for the Market Data V2 bootstrap executor."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from config.market_data import MARKET_DATA_V2_EXECUTION_POLICY


@dataclass(frozen=True)
class MarketDataExecutionPolicy:
    quota_reserve_requests: int
    quota_refresh_every_requests: int
    quota_poll_seconds: float
    max_retryable_attempts: int
    retry_backoff_seconds: tuple[float, ...]
    job_lease_seconds: float
    executor_lock_seconds: float

    def retry_delay_seconds(self, retryable_failure_count: int) -> float:
        # ``retryable_failure_count`` is the number of transient failures that
        # will have occurred after the current failure is recorded (1-based).
        index = max(0, int(retryable_failure_count) - 1)
        return self.retry_backoff_seconds[min(index, len(self.retry_backoff_seconds) - 1)]


def _require_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("MARKET_DATA_V2_EXECUTION_POLICY 必須是 mapping")
    return value


def get_market_data_execution_policy() -> MarketDataExecutionPolicy:
    raw = _require_mapping(MARKET_DATA_V2_EXECUTION_POLICY)
    try:
        quota_reserve_requests = int(raw["quota_reserve_requests"])
        quota_refresh_every_requests = int(raw["quota_refresh_every_requests"])
        quota_poll_seconds = float(raw["quota_poll_seconds"])
        max_retryable_attempts = int(raw["max_retryable_attempts"])
        retry_backoff_seconds = tuple(float(value) for value in raw["retry_backoff_seconds"])
        job_lease_seconds = float(raw["job_lease_seconds"])
        executor_lock_seconds = float(raw["executor_lock_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("MARKET_DATA_V2_EXECUTION_POLICY 缺少合法 execution knob") from exc

    if quota_reserve_requests < 0:
        raise ValueError("quota_reserve_requests 不得 < 0")
    if quota_refresh_every_requests <= 0:
        raise ValueError("quota_refresh_every_requests 必須 > 0")
    if quota_poll_seconds <= 0:
        raise ValueError("quota_poll_seconds 必須 > 0")
    if max_retryable_attempts <= 0:
        raise ValueError("max_retryable_attempts 必須 > 0")
    if not retry_backoff_seconds or any(value <= 0 for value in retry_backoff_seconds):
        raise ValueError("retry_backoff_seconds 必須是非空正數序列")
    if len(retry_backoff_seconds) < max_retryable_attempts - 1:
        raise ValueError("retry_backoff_seconds 必須足以覆蓋 max_retryable_attempts - 1")
    if job_lease_seconds <= 0 or executor_lock_seconds <= 0:
        raise ValueError("Market Data lease 秒數必須 > 0")
    if executor_lock_seconds <= quota_poll_seconds:
        raise ValueError("executor_lock_seconds 必須大於 quota_poll_seconds，避免 quota wait 時誤失鎖")

    return MarketDataExecutionPolicy(
        quota_reserve_requests=quota_reserve_requests,
        quota_refresh_every_requests=quota_refresh_every_requests,
        quota_poll_seconds=quota_poll_seconds,
        max_retryable_attempts=max_retryable_attempts,
        retry_backoff_seconds=retry_backoff_seconds,
        job_lease_seconds=job_lease_seconds,
        executor_lock_seconds=executor_lock_seconds,
    )


__all__ = [
    "MarketDataExecutionPolicy",
    "get_market_data_execution_policy",
]
