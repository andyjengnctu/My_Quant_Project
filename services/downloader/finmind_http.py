"""Small explicit FinMind HTTP client used by Market Data preflight/planning.

Unlike the FinMind SDK path used by the legacy downloader, this client performs
exactly one HTTP attempt per method call and exposes the attempt count.  That is
required for quota accounting and capability probes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd
import requests

FINMIND_DATA_URL = "https://api.finmindtrade.com/api/v4/data"
FINMIND_USER_INFO_URL = "https://api.web.finmindtrade.com/v2/user_info"


class FinMindHttpError(RuntimeError):
    pass


@dataclass(frozen=True)
class FinMindUsage:
    user_count: int
    api_request_limit: int


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
        try:
            payload = response.json()
        except ValueError as exc:
            raise FinMindHttpError(
                f"FinMind {operation} 回傳非 JSON：HTTP {getattr(response, 'status_code', '?')}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise FinMindHttpError(f"FinMind {operation} JSON 格式不是 object")
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
            raise FinMindHttpError(f"FinMind user_info request 失敗: {type(exc).__name__}: {exc}") from exc
        payload = self._decode_json(response, operation="user_info")
        if int(getattr(response, "status_code", 0) or 0) >= 400:
            raise FinMindHttpError(
                f"FinMind user_info HTTP {response.status_code}: {payload.get('msg') or payload}"
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
                f"FinMind data request 失敗: dataset={dataset_id} | {type(exc).__name__}: {exc}"
            ) from exc
        payload = self._decode_json(response, operation=f"data:{dataset_id}")
        http_status = int(getattr(response, "status_code", 0) or 0)
        api_status = payload.get("status")
        if http_status >= 400 or (api_status not in (None, 200, "200")):
            raise FinMindHttpError(
                f"FinMind dataset={dataset_id} HTTP {http_status} status={api_status}: "
                f"{payload.get('msg') or 'unknown error'}"
            )
        raw_data = payload.get("data", [])
        if raw_data is None:
            raw_data = []
        if not isinstance(raw_data, list):
            raise FinMindHttpError(f"FinMind dataset={dataset_id} data 欄位不是 list")
        return pd.DataFrame(raw_data)


__all__ = [
    "FINMIND_DATA_URL",
    "FINMIND_USER_INFO_URL",
    "FinMindHttpError",
    "FinMindUsage",
    "FinMindHttpClient",
]
