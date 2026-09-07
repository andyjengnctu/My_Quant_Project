"""Resolved operational policy for the one-shot Trading Market Data V2 auto updater."""
from __future__ import annotations

from dataclasses import dataclass

from config.market_data import MARKET_DATA_V2_AUTO_UPDATE_POLICY


@dataclass(frozen=True)
class MarketDataAutoUpdatePolicy:
    enabled: bool
    scheduler_wake_minutes: int
    publication_retry_minutes: tuple[int, ...]
    max_publication_retries: int
    quota_defer_minutes: int
    error_defer_minutes: int
    worker_lock_minutes: int

    def publication_retry_delay_minutes(self, attempt_count: int) -> int:
        if attempt_count < 1:
            raise ValueError("publication retry attempt_count 必須 >= 1")
        index = min(attempt_count - 1, len(self.publication_retry_minutes) - 1)
        return int(self.publication_retry_minutes[index])


def get_market_data_auto_update_policy() -> MarketDataAutoUpdatePolicy:
    raw = MARKET_DATA_V2_AUTO_UPDATE_POLICY
    if not isinstance(raw, dict):
        raise ValueError("MARKET_DATA_V2_AUTO_UPDATE_POLICY 必須是 mapping")
    retry_raw = tuple(int(item) for item in tuple(raw.get("publication_retry_minutes") or ()))
    if not retry_raw or any(item <= 0 for item in retry_raw):
        raise ValueError("publication_retry_minutes 必須是非空正整數序列")
    max_retries = int(raw.get("max_publication_retries") or 0)
    if max_retries < 1 or max_retries > len(retry_raw):
        raise ValueError("max_publication_retries 必須落在 publication_retry_minutes 範圍內")
    wake = int(raw.get("scheduler_wake_minutes") or 0)
    quota = int(raw.get("quota_defer_minutes") or 0)
    error = int(raw.get("error_defer_minutes") or 0)
    worker_lock = int(raw.get("worker_lock_minutes") or 0)
    if wake < 1 or quota < 1 or error < 1 or worker_lock < 1:
        raise ValueError("auto update wake/defer/lock minutes 必須 >= 1")
    return MarketDataAutoUpdatePolicy(
        enabled=bool(raw.get("enabled")),
        scheduler_wake_minutes=wake,
        publication_retry_minutes=retry_raw,
        max_publication_retries=max_retries,
        quota_defer_minutes=quota,
        error_defer_minutes=error,
        worker_lock_minutes=worker_lock,
    )


__all__ = ["MarketDataAutoUpdatePolicy", "get_market_data_auto_update_policy"]
