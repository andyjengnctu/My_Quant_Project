"""Deterministic request planning for the Trading Market Data V2 archive sidecar."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable, Mapping

from core.file_integrity import canonical_json_sha256
from core.market_data_bootstrap_requests import BootstrapHttpRequest, build_registry_fingerprint
from core.market_data_dataset_registry import (
    DAILY_EVENT_REPAIR,
    DAILY_INCREMENTAL,
    DAILY_PERIODIC_REPAIR,
    DAILY_RECENT_REPAIR,
    DAILY_STATIC_REFRESH,
    TRADING_QUERY_AUTO,
    TRADING_QUERY_MONTH_STARTS,
    TRADING_QUERY_QUARTER_ENDS,
    TRADING_QUERY_RANGE,
    TRADING_QUERY_RECENT_DATES,
    MarketDatasetSpec,
)
from core.market_data_freshness_contract import validate_market_data_freshness_contracts
from core.market_data_trading_sync_policy import MarketDataTradingSyncPolicy

TRADING_SYNC_QUERY_STATIC = "trading_static_refresh"
TRADING_SYNC_QUERY_INCREMENTAL = "trading_incremental"
TRADING_SYNC_QUERY_RECENT = "trading_recent_repair"
TRADING_SYNC_QUERY_EVENT = "trading_event_repair"
TRADING_SYNC_QUERY_PERIODIC = "trading_periodic_repair"


@dataclass(frozen=True)
class TradingSyncRequestManifest:
    as_of_date: str
    full_range_start: str
    registry_fingerprint: str
    manifest_fingerprint: str
    historical_instrument_count: int
    requests: tuple[BootstrapHttpRequest, ...]
    base_provider_snapshot_fingerprint: str
    base_provider_manifest_fingerprint: str
    base_as_of_date: str
    previous_sync_date: str | None

    @property
    def total_requests(self) -> int:
        return len(self.requests)


def _iso(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{field} 必須是 YYYY-MM-DD: {text!r}") from exc
    return parsed.isoformat()


def _calendar_dates(start: date, end: date) -> tuple[str, ...]:
    if start > end:
        return ()
    count = (end - start).days + 1
    return tuple((start + timedelta(days=offset)).isoformat() for offset in range(count))


def _month_start_dates(end: date, count: int) -> tuple[str, ...]:
    if count < 1:
        raise ValueError("month-start lookback count 必須 >= 1")
    values: list[date] = []
    cursor = date(end.year, end.month, 1)
    for _ in range(count):
        values.append(cursor)
        cursor = date(cursor.year - 1, 12, 1) if cursor.month == 1 else date(cursor.year, cursor.month - 1, 1)
    return tuple(sorted(item.isoformat() for item in values if item <= end))


def _quarter_end_dates(end: date, count: int) -> tuple[str, ...]:
    if count < 1:
        raise ValueError("quarter-end lookback count 必須 >= 1")
    quarter_month = ((end.month - 1) // 3 + 1) * 3
    cursor = date(end.year, quarter_month, 1)
    # Move to quarter end, then step back if the current quarter end is future.
    next_month = date(cursor.year + (1 if cursor.month == 12 else 0), 1 if cursor.month == 12 else cursor.month + 1, 1)
    cursor_end = next_month - timedelta(days=1)
    if cursor_end > end:
        prev_q_month = quarter_month - 3
        if prev_q_month <= 0:
            prev_q_month += 12
            prev_q_year = end.year - 1
        else:
            prev_q_year = end.year
        next_m = date(prev_q_year + (1 if prev_q_month == 12 else 0), 1 if prev_q_month == 12 else prev_q_month + 1, 1)
        cursor_end = next_m - timedelta(days=1)
    values: list[date] = []
    current = cursor_end
    for _ in range(count):
        values.append(current)
        month = current.month - 3
        year = current.year
        if month <= 0:
            month += 12
            year -= 1
        next_m = date(year + (1 if month == 12 else 0), 1 if month == 12 else month + 1, 1)
        current = next_m - timedelta(days=1)
    return tuple(sorted(item.isoformat() for item in values))


def _exact_date_requests(spec: MarketDatasetSpec, mode: str, dates: Iterable[str]) -> list[BootstrapHttpRequest]:
    if not spec.full_market_exact_date_expected:
        raise ValueError(f"{spec.dataset} 未宣告 Backer full-market exact-date capability")
    return [BootstrapHttpRequest(spec.dataset, mode, None, value, value) for value in dates]


def _range_requests(spec: MarketDatasetSpec, mode: str, start_date: str, end_date: str) -> list[BootstrapHttpRequest]:
    if spec.fixed_data_ids:
        return [BootstrapHttpRequest(spec.dataset, mode, data_id, start_date, end_date) for data_id in spec.fixed_data_ids]
    return [BootstrapHttpRequest(spec.dataset, mode, None, start_date, end_date)]


def _periodic_requests(spec: MarketDatasetSpec, *, target: date) -> list[BootstrapHttpRequest]:
    query_mode = str(spec.trading_query_mode or TRADING_QUERY_AUTO)
    lookback = int(spec.trading_lookback_periods or 0)
    if query_mode == TRADING_QUERY_MONTH_STARTS:
        return _exact_date_requests(spec, TRADING_SYNC_QUERY_PERIODIC, _month_start_dates(target, lookback))
    if query_mode == TRADING_QUERY_QUARTER_ENDS:
        return _exact_date_requests(spec, TRADING_SYNC_QUERY_PERIODIC, _quarter_end_dates(target, lookback))
    if query_mode == TRADING_QUERY_RECENT_DATES:
        start = target - timedelta(days=lookback - 1)
        return _exact_date_requests(spec, TRADING_SYNC_QUERY_PERIODIC, _calendar_dates(start, target))
    if query_mode == TRADING_QUERY_RANGE:
        start = target - timedelta(days=lookback - 1)
        return _range_requests(spec, TRADING_SYNC_QUERY_PERIODIC, start.isoformat(), target.isoformat())
    raise ValueError(f"{spec.dataset} periodic Trading query policy 未登記: {query_mode}")


def build_trading_sync_request_manifest(
    *,
    specs: Iterable[MarketDatasetSpec],
    provider_snapshot: Mapping[str, object],
    target_date: str,
    previous_sync_date: str | None,
    policy: MarketDataTradingSyncPolicy,
    selected_datasets: Iterable[str] | None = None,
    previous_ready_dates_by_dataset: Mapping[str, str | None] | None = None,
) -> TradingSyncRequestManifest:
    included = tuple(spec for spec in specs if spec.included)
    if not included:
        raise ValueError("Trading Market Data V2 沒有 included dataset")
    validate_market_data_freshness_contracts(specs=included)
    included_names = {spec.dataset for spec in included}
    if selected_datasets is None:
        selected_names = included_names
    else:
        selected_names = {str(item) for item in selected_datasets}
        unknown = selected_names - included_names
        if unknown:
            raise ValueError(f"Trading V2 selected_datasets 含未知 dataset: {sorted(unknown)}")
        if not selected_names:
            raise ValueError("Trading V2 selected_datasets 不可為空")
    previous_by_dataset = dict(previous_ready_dates_by_dataset or {})
    unknown_previous = set(previous_by_dataset) - included_names
    if unknown_previous:
        raise ValueError(f"Trading V2 previous_ready_dates_by_dataset 含未知 dataset: {sorted(unknown_previous)}")
    target_text = _iso(target_date, field="target_date")
    target = date.fromisoformat(target_text)
    base_as_of = _iso(provider_snapshot.get("as_of_date"), field="provider_snapshot.as_of_date")
    base_date = date.fromisoformat(base_as_of)
    if target < base_date:
        raise ValueError(f"Trading target_date 不得早於 provider snapshot: {target_text} < {base_as_of}")
    previous_text = None
    if previous_sync_date:
        previous_text = _iso(previous_sync_date, field="previous_sync_date")
        if previous_text < base_as_of or previous_text > target_text:
            raise ValueError("previous_sync_date 必須落在 provider snapshot 與 target_date 之間")

    provider_fp = str(provider_snapshot.get("snapshot_fingerprint") or "").strip()
    provider_manifest_fp = str(provider_snapshot.get("manifest_fingerprint") or "").strip()
    provider_registry_fp = str(provider_snapshot.get("registry_fingerprint") or "").strip()
    if len(provider_fp) != 64 or len(provider_manifest_fp) != 64 or len(provider_registry_fp) != 64:
        raise ValueError("provider snapshot identity fingerprint 不合法")
    registry_fp = build_registry_fingerprint(included)
    if registry_fp != provider_registry_fp:
        raise ValueError("Trading V2 current dataset registry 與 provider snapshot registry 已 drift；禁止自動 sync")

    requests: list[BootstrapHttpRequest] = []
    recent_start = target - timedelta(days=policy.recent_repair_calendar_days - 1)
    event_start = target - timedelta(days=policy.event_repair_calendar_days - 1)

    for spec in included:
        if spec.dataset not in selected_names:
            continue
        dataset_previous = previous_text
        if previous_by_dataset:
            raw_previous = previous_by_dataset.get(spec.dataset)
            dataset_previous = _iso(raw_previous, field=f"previous_ready_dates_by_dataset[{spec.dataset}]") if raw_previous else None
            if dataset_previous is not None and (dataset_previous < base_as_of or dataset_previous > target_text):
                raise ValueError(
                    f"{spec.dataset} previous ready date 必須落在 provider snapshot 與 target_date 之間"
                )
        incremental_start = date.fromisoformat(dataset_previous or base_as_of) + timedelta(days=1)
        if spec.daily_mode == DAILY_STATIC_REFRESH:
            requests.append(BootstrapHttpRequest(spec.dataset, TRADING_SYNC_QUERY_STATIC, None, None, None))
        elif spec.daily_mode == DAILY_INCREMENTAL:
            if incremental_start <= target:
                if spec.full_market_exact_date_expected:
                    requests.extend(
                        _exact_date_requests(
                            spec,
                            TRADING_SYNC_QUERY_INCREMENTAL,
                            _calendar_dates(incremental_start, target),
                        )
                    )
                elif spec.bootstrap_mode == "single_no_dates" and not spec.fixed_data_ids:
                    requests.append(BootstrapHttpRequest(spec.dataset, TRADING_SYNC_QUERY_INCREMENTAL, None, None, None))
                else:
                    requests.extend(_range_requests(spec, TRADING_SYNC_QUERY_INCREMENTAL, incremental_start.isoformat(), target_text))
        elif spec.daily_mode == DAILY_RECENT_REPAIR:
            # A repair window is the minimum re-query horizon, not a license to
            # skip dates when this dataset has been offline longer than the
            # configured window.  Extend back to the first not-yet-ready date
            # so the Trading overlay stays contiguous after long downtime.
            repair_start = min(recent_start, incremental_start)
            if spec.full_market_exact_date_expected:
                requests.extend(_exact_date_requests(spec, TRADING_SYNC_QUERY_RECENT, _calendar_dates(repair_start, target)))
            else:
                requests.extend(_range_requests(spec, TRADING_SYNC_QUERY_RECENT, repair_start.isoformat(), target_text))
        elif spec.daily_mode == DAILY_EVENT_REPAIR:
            repair_start = min(event_start, incremental_start)
            if spec.full_market_exact_date_expected:
                requests.extend(_exact_date_requests(spec, TRADING_SYNC_QUERY_EVENT, _calendar_dates(repair_start, target)))
            else:
                requests.extend(_range_requests(spec, TRADING_SYNC_QUERY_EVENT, repair_start.isoformat(), target_text))
        elif spec.daily_mode == DAILY_PERIODIC_REPAIR:
            requests.extend(_periodic_requests(spec, target=target))
        else:
            raise ValueError(f"{spec.dataset} 不支援 Trading daily_mode: {spec.daily_mode}")

    request_ids = tuple(request.request_id for request in requests)
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("Trading V2 sync request manifest 含重複 logical request identity")
    manifest_fp = canonical_json_sha256(
        {
            "role": "trading_market_data_v2_archive_sync",
            "base_provider_snapshot_fingerprint": provider_fp,
            "base_provider_manifest_fingerprint": provider_manifest_fp,
            "base_as_of_date": base_as_of,
            "previous_sync_date": previous_text,
            "target_date": target_text,
            "registry_fingerprint": registry_fp,
            "request_ids": request_ids,
        }
    )
    return TradingSyncRequestManifest(
        as_of_date=target_text,
        full_range_start=base_as_of,
        registry_fingerprint=registry_fp,
        manifest_fingerprint=manifest_fp,
        historical_instrument_count=int(provider_snapshot.get("historical_instrument_count") or 0),
        requests=tuple(requests),
        base_provider_snapshot_fingerprint=provider_fp,
        base_provider_manifest_fingerprint=provider_manifest_fp,
        base_as_of_date=base_as_of,
        previous_sync_date=previous_text,
    )


__all__ = [
    "TRADING_SYNC_QUERY_STATIC",
    "TRADING_SYNC_QUERY_INCREMENTAL",
    "TRADING_SYNC_QUERY_RECENT",
    "TRADING_SYNC_QUERY_EVENT",
    "TRADING_SYNC_QUERY_PERIODIC",
    "TradingSyncRequestManifest",
    "build_trading_sync_request_manifest",
]
