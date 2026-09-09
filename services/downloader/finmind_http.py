"""Explicit single-attempt FinMind HTTP client for Market Data V2.

Each ``get_data``/``get_usage`` call performs exactly one HTTP attempt.  Error
metadata is classified so the persistent bootstrap executor can distinguish
quota wait, bounded transient retry and permanent fail-closed conditions without
introducing hidden SDK retries.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping

import pandas as pd
import requests

from core.market_data_execution_policy import get_market_data_execution_policy

FINMIND_DATA_URL = "https://api.finmindtrade.com/api/v4/data"
FINMIND_USER_INFO_URL = "https://api.web.finmindtrade.com/v2/user_info"


class FinMindHttpError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        http_status: int | None = None,
        api_status: int | str | None = None,
        retryable: bool = False,
        quota_exhausted: bool = False,
    ):
        super().__init__(message)
        self.http_status = http_status
        self.api_status = api_status
        self.retryable = bool(retryable)
        self.quota_exhausted = bool(quota_exhausted)


@dataclass(frozen=True)
class FinMindUsage:
    user_count: int
    api_request_limit: int


def _status_traits(http_status: int | None, api_status: int | str | None = None) -> tuple[bool, bool]:
    statuses: list[int] = []
    for raw_status in (http_status, api_status):
        if raw_status is None:
            continue
        try:
            statuses.append(int(raw_status))
        except (TypeError, ValueError):
            # A provider may omit or return a non-numeric API status while the
            # HTTP status still carries the actionable retry/quota semantics.
            # Treat the non-numeric value as absent; this is normalization
            # control-flow, not an operational exception fallback.
            continue
    quota = 402 in statuses
    retryable = quota or any(status in {408, 425, 429} or status >= 500 for status in statuses)
    return retryable, quota


class FinMindHttpClient:
    def __init__(self, *, token: str, timeout_sec: float = 30.0, session=None):
        resolved = str(token or "").strip()
        if not resolved:
            raise ValueError("FinMind API token 不可空白")
        self._token = resolved
        self._timeout_sec = float(timeout_sec)
        if self._timeout_sec <= 0:
            raise ValueError("FinMind HTTP timeout 必須 > 0")
        self._session = requests.Session() if session is None else session
        self.data_request_count = 0
        self.usage_request_count = 0

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"}

    def _decode_json(self, response, *, operation: str) -> Mapping[str, Any]:
        http_status = int(getattr(response, "status_code", 0) or 0)
        retryable, quota = _status_traits(http_status)
        try:
            payload = response.json()
        except ValueError as exc:
            raise FinMindHttpError(
                f"FinMind {operation} 回傳非 JSON：HTTP {getattr(response, 'status_code', '?')}",
                http_status=http_status,
                retryable=retryable,
                quota_exhausted=quota,
            ) from exc
        if not isinstance(payload, Mapping):
            raise FinMindHttpError(
                f"FinMind {operation} JSON 格式不是 object",
                http_status=http_status,
                retryable=retryable,
                quota_exhausted=quota,
            )
        return payload

    def get_usage(self) -> FinMindUsage:
        self.usage_request_count += 1
        try:
            response = self._session.get(
                FINMIND_USER_INFO_URL,
                headers=self.headers,
                timeout=self._timeout_sec,
            )
        except requests.RequestException as exc:
            raise FinMindHttpError(
                f"FinMind user_info request 失敗: {type(exc).__name__}: {exc}",
                retryable=True,
            ) from exc
        payload = self._decode_json(response, operation="user_info")
        http_status = int(getattr(response, "status_code", 0) or 0)
        if http_status >= 400:
            retryable, quota = _status_traits(http_status, payload.get("status"))
            raise FinMindHttpError(
                f"FinMind user_info HTTP {response.status_code}: {payload.get('msg') or payload}",
                http_status=http_status,
                api_status=payload.get("status"),
                retryable=retryable,
                quota_exhausted=quota,
            )
        try:
            user_count = int(payload["user_count"])
            api_request_limit = int(payload["api_request_limit"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FinMindHttpError("FinMind user_info 缺少合法 user_count/api_request_limit") from exc
        if api_request_limit <= 0 or user_count < 0:
            raise FinMindHttpError(
                f"FinMind user_info quota 不合法: user_count={user_count}, api_request_limit={api_request_limit}"
            )
        return FinMindUsage(user_count=user_count, api_request_limit=api_request_limit)

    def get_data(
        self,
        *,
        dataset: str,
        data_id: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        dataset_id = str(dataset or "").strip()
        if not dataset_id:
            raise ValueError("FinMind dataset 不可空白")
        params: dict[str, str] = {"dataset": dataset_id}
        if data_id is not None and str(data_id).strip():
            params["data_id"] = str(data_id).strip()
        if start_date is not None and str(start_date).strip():
            params["start_date"] = str(start_date).strip()
        if end_date is not None and str(end_date).strip():
            params["end_date"] = str(end_date).strip()

        self.data_request_count += 1
        try:
            response = self._session.get(
                FINMIND_DATA_URL,
                headers=self.headers,
                params=params,
                timeout=self._timeout_sec,
            )
        except requests.RequestException as exc:
            raise FinMindHttpError(
                f"FinMind data request 失敗: dataset={dataset_id} | {type(exc).__name__}: {exc}",
                retryable=True,
            ) from exc
        payload = self._decode_json(response, operation=f"data:{dataset_id}")
        http_status = int(getattr(response, "status_code", 0) or 0)
        api_status = payload.get("status")
        if http_status >= 400 or (api_status not in (None, 200, "200")):
            retryable, quota = _status_traits(http_status, api_status)
            raise FinMindHttpError(
                f"FinMind dataset={dataset_id} HTTP {http_status} status={api_status}: "
                f"{payload.get('msg') or 'unknown error'}",
                http_status=http_status,
                api_status=api_status,
                retryable=retryable,
                quota_exhausted=quota,
            )
        raw_data = payload.get("data", [])
        if raw_data is None:
            raw_data = []
        if not isinstance(raw_data, list):
            raise FinMindHttpError(f"FinMind dataset={dataset_id} data 欄位不是 list")
        return pd.DataFrame(raw_data)


def request_finmind_data_with_retry(
    client,
    *,
    dataset: str,
    data_id: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    retain_cache: bool = True,
    policy=None,
    sleep_fn=None,
) -> pd.DataFrame:
    """Explicit bounded retry wrapper for execution-critical FinMind reads.

    ``FinMindHttpClient`` and ``SharedFinMindRequestClient`` remain single-call
    primitives.  This helper is intentionally opt-in so V2 preflight/executor
    can continue owning their own retry/ledger semantics.  Quota exhaustion and
    permanent errors are never retried.
    """

    execution_policy = get_market_data_execution_policy() if policy is None else policy
    sleeper = time.sleep if sleep_fn is None else sleep_fn
    failure_count = 0
    while True:
        try:
            fetch = client.get_data
            if not retain_cache:
                uncached = getattr(client, "get_data_uncached", None)
                if callable(uncached):
                    fetch = uncached
            return fetch(
                dataset=dataset,
                data_id=data_id,
                start_date=start_date,
                end_date=end_date,
            )
        except FinMindHttpError as exc:
            if exc.quota_exhausted or not exc.retryable:
                raise
            failure_count += 1
            if failure_count >= int(execution_policy.max_retryable_attempts):
                raise
            sleeper(float(execution_policy.retry_delay_seconds(failure_count)))


__all__ = [
    "FINMIND_DATA_URL",
    "FINMIND_USER_INFO_URL",
    "FinMindHttpError",
    "FinMindUsage",
    "FinMindHttpClient",
    "request_finmind_data_with_retry",
]
