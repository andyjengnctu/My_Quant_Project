"""Resolved operational policy for the one-shot Trading Market Data V2 auto updater."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time

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
    market_date_discovery_first_check_time: time
    market_date_discovery_retry_minutes: tuple[int, ...]
    market_date_discovery_max_retries: int

    def publication_retry_delay_minutes(self, attempt_count: int) -> int:
        if attempt_count < 1:
            raise ValueError("publication retry attempt_count 必須 >= 1")
        index = min(attempt_count - 1, len(self.publication_retry_minutes) - 1)
        return int(self.publication_retry_minutes[index])

    def market_date_discovery_retry_delay_minutes(self, attempt_count: int) -> int:
        if attempt_count < 1:
            raise ValueError("market-date discovery retry attempt_count 必須 >= 1")
        index = min(attempt_count - 1, len(self.market_date_discovery_retry_minutes) - 1)
        return int(self.market_date_discovery_retry_minutes[index])


def _positive_minutes_tuple(raw, *, field: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in tuple(raw or ()))
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"{field} 必須是非空正整數序列")
    return values


def _clock(raw, *, field: str) -> time:
    text = str(raw or "").strip()
    try:
        hh, mm = (int(part) for part in text.split(":"))
        return time(hour=hh, minute=mm)
    except (TypeError, ValueError):
        raise ValueError(f"{field} 必須是 HH:MM") from None


def get_market_data_auto_update_policy() -> MarketDataAutoUpdatePolicy:
    raw = MARKET_DATA_V2_AUTO_UPDATE_POLICY
    if not isinstance(raw, dict):
        raise ValueError("MARKET_DATA_V2_AUTO_UPDATE_POLICY 必須是 mapping")
    retry_raw = _positive_minutes_tuple(raw.get("publication_retry_minutes"), field="publication_retry_minutes")
    max_retries = int(raw.get("max_publication_retries") or 0)
    if max_retries < 1 or max_retries > len(retry_raw):
        raise ValueError("max_publication_retries 必須落在 publication_retry_minutes 範圍內")
    discovery_retry = _positive_minutes_tuple(
        raw.get("market_date_discovery_retry_minutes"),
        field="market_date_discovery_retry_minutes",
    )
    discovery_max_retries = int(raw.get("market_date_discovery_max_retries") or 0)
    if discovery_max_retries < 1 or discovery_max_retries > len(discovery_retry):
        raise ValueError("market_date_discovery_max_retries 必須落在 market_date_discovery_retry_minutes 範圍內")
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
        market_date_discovery_first_check_time=_clock(
            raw.get("market_date_discovery_first_check_time"),
            field="market_date_discovery_first_check_time",
        ),
        market_date_discovery_retry_minutes=discovery_retry,
        market_date_discovery_max_retries=discovery_max_retries,
    )


__all__ = ["MarketDataAutoUpdatePolicy", "get_market_data_auto_update_policy"]
