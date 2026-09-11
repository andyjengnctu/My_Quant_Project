"""Pure local due-planning for Trading Market Data V2 datasets.

The planner never calls FinMind and never inspects provider artifacts.  It only
combines the canonical publication/freshness contract with persisted dynamic
state and a caller-supplied Trading target date.  This keeps scheduler wake-ups
quota-free whenever no dataset is due.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable, Mapping
from zoneinfo import ZoneInfo

from config.market_data import MARKET_DATA_V2_PUBLICATION_POLICY
from core.market_data_dataset_readiness import is_market_data_dataset_ready
from core.market_data_freshness_contract import (
    EXPECTED_DATE_LATEST_AVAILABLE,
    FRESHNESS_STATUS_BLOCKED,
    FRESHNESS_STATUS_DUE,
    FRESHNESS_STATUS_ERROR,
    FRESHNESS_STATUS_READY,
    FRESHNESS_STATUS_STALE,
    FRESHNESS_STATUS_WAIT_PUBLISH,
    FRESHNESS_STATUS_WAIT_QUOTA,
    MarketDataFreshnessContract,
    get_market_data_freshness_contracts,
)


@dataclass(frozen=True)
class MarketDataDueDecision:
    dataset: str
    status: str
    due: bool
    expected_publish_at: str
    latest_expected_date: str | None
    next_check_at: str | None
    reason: str


@dataclass(frozen=True)
class MarketDataDuePlan:
    target_date: str
    planned_at: str
    decisions: tuple[MarketDataDueDecision, ...]

    @property
    def due_datasets(self) -> tuple[str, ...]:
        return tuple(item.dataset for item in self.decisions if item.due)

    @property
    def provider_requests_required(self) -> bool:
        return bool(self.due_datasets)


def _iso_date(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise ValueError(f"{field} 必須是 YYYY-MM-DD: {text!r}") from exc


def _publication_timezone() -> ZoneInfo:
    raw = MARKET_DATA_V2_PUBLICATION_POLICY
    timezone_name = str(raw.get("timezone") or "").strip() if isinstance(raw, Mapping) else ""
    if not timezone_name:
        raise ValueError("MARKET_DATA_V2_PUBLICATION_POLICY.timezone 不可空白")
    return ZoneInfo(timezone_name)


def normalize_market_data_planner_now(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=_publication_timezone())
    return value.astimezone(_publication_timezone())


def resolve_market_data_expected_publish_at(
    contract: MarketDataFreshnessContract,
    *,
    target_date: str,
) -> datetime:
    target = date.fromisoformat(_iso_date(target_date, field="target_date"))
    hour_text, minute_text = contract.publication_first_check_time.split(":", 1)
    local_date = target + timedelta(days=int(contract.publication_day_offset))
    return datetime(
        local_date.year,
        local_date.month,
        local_date.day,
        int(hour_text),
        int(minute_text),
        tzinfo=_publication_timezone(),
    )


def _dataset_state_map(state: Mapping[str, object] | None) -> Mapping[str, object]:
    if not state:
        return {}
    datasets = state.get("datasets")
    return datasets if isinstance(datasets, Mapping) else {}


def _expected_date(
    contract: MarketDataFreshnessContract,
    target_date: str,
    item: Mapping[str, object],
) -> str | None:
    # Event/current-vintage datasets do not claim a row for target_date.
    # ``latest_available`` means the provider's newest observed row as of this
    # poll, not the Taiwan Trading target date.
    if contract.expected_date_mode in {"none", "period_due"}:
        return None
    if contract.expected_date_mode == EXPECTED_DATE_LATEST_AVAILABLE:
        value = str(item.get("latest_data_date") or "").strip()
        return value or None
    return target_date


def plan_market_data_due_datasets(
    *,
    target_date: str,
    now: datetime,
    state: Mapping[str, object] | None = None,
    contracts: Iterable[MarketDataFreshnessContract] | None = None,
) -> MarketDataDuePlan:
    """Return a provider-free due plan for one Trading target date."""

    target_text = _iso_date(target_date, field="target_date")
    local_now = normalize_market_data_planner_now(now)
    resolved_contracts = tuple(contracts) if contracts is not None else get_market_data_freshness_contracts()
    dynamic = _dataset_state_map(state)
    decisions: list[MarketDataDueDecision] = []

    for contract in resolved_contracts:
        row = dynamic.get(contract.dataset)
        item = row if isinstance(row, Mapping) else {}
        expected_publish = resolve_market_data_expected_publish_at(contract, target_date=target_text)
        ready_target = str(item.get("last_ready_target_date") or "").strip()
        latest_expected = _expected_date(contract, target_text, item)

        if is_market_data_dataset_ready(item, target_date=target_text):
            decisions.append(
                MarketDataDueDecision(
                    dataset=contract.dataset,
                    status=FRESHNESS_STATUS_READY,
                    due=False,
                    expected_publish_at=expected_publish.isoformat(),
                    latest_expected_date=latest_expected,
                    next_check_at=None,
                    reason="target_already_ready",
                )
            )
            continue

        persisted_next = str(item.get("next_check_at") or "").strip()
        persisted_status = str(item.get("status") or "").strip()
        attempt_target = str(item.get("last_attempt_target_date") or "").strip()
        if attempt_target == target_text and persisted_status in {FRESHNESS_STATUS_STALE, FRESHNESS_STATUS_BLOCKED}:
            decisions.append(
                MarketDataDueDecision(
                    dataset=contract.dataset,
                    status=persisted_status,
                    due=False,
                    expected_publish_at=expected_publish.isoformat(),
                    latest_expected_date=latest_expected,
                    next_check_at=None,
                    reason="persisted_terminal_status",
                )
            )
            continue

        if (
            attempt_target == target_text
            and persisted_status in {
                FRESHNESS_STATUS_WAIT_PUBLISH,
                FRESHNESS_STATUS_WAIT_QUOTA,
                FRESHNESS_STATUS_ERROR,
            }
            and persisted_next
        ):
            try:
                retry_at = datetime.fromisoformat(persisted_next)
                retry_at = normalize_market_data_planner_now(retry_at)
            except ValueError:
                retry_at = expected_publish
            if local_now < retry_at:
                decisions.append(
                    MarketDataDueDecision(
                        dataset=contract.dataset,
                        status=persisted_status,
                        due=False,
                        expected_publish_at=expected_publish.isoformat(),
                        latest_expected_date=latest_expected,
                        next_check_at=retry_at.isoformat(),
                        reason="persisted_retry_not_due",
                    )
                )
                continue

        if local_now < expected_publish:
            decisions.append(
                MarketDataDueDecision(
                    dataset=contract.dataset,
                    status=FRESHNESS_STATUS_WAIT_PUBLISH,
                    due=False,
                    expected_publish_at=expected_publish.isoformat(),
                    latest_expected_date=latest_expected,
                    next_check_at=expected_publish.isoformat(),
                    reason="before_publication_window",
                )
            )
            continue

        decisions.append(
            MarketDataDueDecision(
                dataset=contract.dataset,
                status=FRESHNESS_STATUS_DUE,
                due=True,
                expected_publish_at=expected_publish.isoformat(),
                latest_expected_date=latest_expected,
                next_check_at=local_now.isoformat(),
                reason="publication_window_reached",
            )
        )

    return MarketDataDuePlan(
        target_date=target_text,
        planned_at=local_now.isoformat(),
        decisions=tuple(decisions),
    )


__all__ = [
    "MarketDataDueDecision",
    "MarketDataDuePlan",
    "normalize_market_data_planner_now",
    "resolve_market_data_expected_publish_at",
    "plan_market_data_due_datasets",
]
