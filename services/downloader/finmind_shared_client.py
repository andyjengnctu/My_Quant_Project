"""Request-deduplicating FinMind client shared by one canonical Trading update.

The wrapper never changes provider payloads.  Exact request identities may be
seeded from a semantically equivalent wider-range response of the same FinMind
dataset; consumers still receive the raw provider rows for the requested slice.
The cache is process-local and is never a persistent data truth.
"""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class FinMindRequestKey:
    dataset: str
    data_id: str | None
    start_date: str | None
    end_date: str | None


def _clean(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def build_finmind_request_key(
    *,
    dataset: str,
    data_id: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> FinMindRequestKey:
    dataset_text = str(dataset or "").strip()
    if not dataset_text:
        raise ValueError("FinMind dataset 不可空白")
    return FinMindRequestKey(
        dataset=dataset_text,
        data_id=_clean(data_id),
        start_date=_clean(start_date),
        end_date=_clean(end_date),
    )


class SharedFinMindRequestClient:
    """One-process request cache around a FinMind-compatible HTTP client."""

    def __init__(self, base_client):
        self.base_client = base_client
        self._cache: dict[FinMindRequestKey, pd.DataFrame] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.uncached_fetches = 0
        self.seeded_entries = 0

    @property
    def data_request_count(self) -> int:
        return int(getattr(self.base_client, "data_request_count", 0))

    @property
    def usage_request_count(self) -> int:
        return int(getattr(self.base_client, "usage_request_count", 0))

    def get_usage(self):
        return self.base_client.get_usage()

    def has_cached_data(self, **kwargs) -> bool:
        return build_finmind_request_key(**kwargs) in self._cache

    def will_issue_data_request(self, **kwargs) -> bool:
        """Executor capability hook: False means get_data is a zero-HTTP cache hit."""

        return not self.has_cached_data(**kwargs)

    def seed_data(
        self,
        *,
        dataset: str,
        frame: pd.DataFrame,
        data_id: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("FinMind shared cache seed 必須是 DataFrame")
        key = build_finmind_request_key(
            dataset=dataset,
            data_id=data_id,
            start_date=start_date,
            end_date=end_date,
        )
        existing = self._cache.get(key)
        if existing is not None:
            if list(existing.columns) != list(frame.columns) or not existing.equals(frame):
                raise ValueError(f"FinMind shared cache 同一 request identity 收到不一致 payload: {key}")
            return
        self._cache[key] = frame.copy(deep=True)
        self.seeded_entries += 1

    def get_data(
        self,
        *,
        dataset: str,
        data_id: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        key = build_finmind_request_key(
            dataset=dataset,
            data_id=data_id,
            start_date=start_date,
            end_date=end_date,
        )
        cached = self._cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached.copy(deep=True)
        self.cache_misses += 1
        frame = self.base_client.get_data(
            dataset=key.dataset,
            data_id=key.data_id,
            start_date=key.start_date,
            end_date=key.end_date,
        )
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("FinMind client get_data 必須回傳 DataFrame")
        self._cache[key] = frame.copy(deep=True)
        return frame.copy(deep=True)

    def get_data_uncached(
        self,
        *,
        dataset: str,
        data_id: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Issue one provider request without retaining the raw frame in cache.

        Used for large historical bulk chunks whose payload will not be reused by
        another consumer in the same canonical update process.
        """

        key = build_finmind_request_key(
            dataset=dataset,
            data_id=data_id,
            start_date=start_date,
            end_date=end_date,
        )
        self.uncached_fetches += 1
        frame = self.base_client.get_data(
            dataset=key.dataset,
            data_id=key.data_id,
            start_date=key.start_date,
            end_date=key.end_date,
        )
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("FinMind client get_data 必須回傳 DataFrame")
        return frame.copy(deep=True)

    def snapshot(self) -> dict[str, int]:
        return {
            "provider_data_requests": self.data_request_count,
            "provider_usage_requests": self.usage_request_count,
            "cache_hits": int(self.cache_hits),
            "cache_misses": int(self.cache_misses),
            "uncached_fetches": int(self.uncached_fetches),
            "seeded_entries": int(self.seeded_entries),
            "cached_entries": int(len(self._cache)),
        }


__all__ = [
    "FinMindRequestKey",
    "SharedFinMindRequestClient",
    "build_finmind_request_key",
]
