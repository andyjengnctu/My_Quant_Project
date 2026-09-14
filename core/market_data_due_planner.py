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
from core.market_data_dataset_readiness import (
    is_market_data_dataset_ready,
    market_data_contract_requires_target_freshness,
)
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


def _state_datetime(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return normalize_market_data_planner_now(datetime.fromisoformat(text))
    except ValueError:
        return None


def _scheduled_probe_satisfied(
    item: Mapping[str, object],
    *,
    target_date: str,
    expected_publish: datetime,
) -> bool:
    """Whether the target's scheduler observation happened in its due window."""

    if str(item.get("last_attempt_target_date") or "").strip() != str(target_date):
        return False
    if str(item.get("last_attempt_result") or "").strip() != "SUCCESS":
        return False
    observed_at = _state_datetime(item.get("last_success_at") or item.get("last_attempt_at"))
    return bool(observed_at is not None and observed_at >= expected_publish)


def plan_market_data_due_datasets(
    *,
    target_date: str,
    now: datetime,
    state: Mapping[str, object] | None = None,
    contracts: Iterable[MarketDataFreshnessContract] | None = None,
) -> MarketDataDuePlan:
    """Return a provider-free due plan for one Trading target date.

    Dataset usability and provider scheduling are separate contracts. Exact
    target-date feeds stop being due once current-target freshness is READY.
    Roll-forward feeds may remain READY while still carrying a scheduled probe
    obligation at their publication window.
    """

    target_text = _iso_date(target_date, field="target_date")
    local_now = normalize_market_data_planner_now(now)
    resolved_contracts = tuple(contracts) if contracts is not None else get_market_data_freshness_contracts()
    dynamic = _dataset_state_map(state)
    decisions: list[MarketDataDueDecision] = []

    for contract in resolved_contracts:
        row = dynamic.get(contract.dataset)
        item = row if isinstance(row, Mapping) else {}
        expected_publish = resolve_market_data_expected_publish_at(contract, target_date=target_text)
        latest_expected = _expected_date(contract, target_text, item)
        target_required = market_data_contract_requires_target_freshness(contract)
        data_ready = is_market_data_dataset_ready(
            item,
            target_date=target_text,
            contract=contract,
        )

        # Exact target-date feeds have no independent probe obligation after
        # current-target freshness is proven; the data evidence itself is the
        # scheduler completion evidence.
        if data_ready and target_required:
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

        attempt_target = str(item.get("last_attempt_target_date") or "").strip()
        last_attempt_result = str(item.get("last_attempt_result") or "").strip()
        persisted_status = str(item.get("status") or "").strip()
        persisted_next = str(item.get("next_check_at") or "").strip()

        # Roll-forward feeds remain usable across targets, but a successful
        # provider observation before the documented/estimated publication
        # window does not consume that target's scheduled check.
        if data_ready and not target_required:
            if _scheduled_probe_satisfied(
                item,
                target_date=target_text,
                expected_publish=expected_publish,
            ):
                decisions.append(
                    MarketDataDueDecision(
                        dataset=contract.dataset,
                        status=FRESHNESS_STATUS_READY,
                        due=False,
                        expected_publish_at=expected_publish.isoformat(),
                        latest_expected_date=latest_expected,
                        next_check_at=None,
                        reason="roll_forward_scheduled_probe_satisfied",
                    )
                )
                continue

            if attempt_target == target_text and last_attempt_result in {
                FRESHNESS_STATUS_WAIT_PUBLISH,
                FRESHNESS_STATUS_WAIT_QUOTA,
                FRESHNESS_STATUS_ERROR,
            } and persisted_next:
                retry_at = _state_datetime(persisted_next) or expected_publish
                if local_now < retry_at:
                    decisions.append(
                        MarketDataDueDecision(
                            dataset=contract.dataset,
                            status=FRESHNESS_STATUS_READY,
                            due=False,
                            expected_publish_at=expected_publish.isoformat(),
                            latest_expected_date=latest_expected,
                            next_check_at=retry_at.isoformat(),
                            reason="roll_forward_retry_not_due",
                        )
                    )
                    continue

            if attempt_target == target_text and last_attempt_result in {
                FRESHNESS_STATUS_STALE,
                FRESHNESS_STATUS_BLOCKED,
            }:
                decisions.append(
                    MarketDataDueDecision(
                        dataset=contract.dataset,
                        status=FRESHNESS_STATUS_READY,
                        due=False,
                        expected_publish_at=expected_publish.isoformat(),
                        latest_expected_date=latest_expected,
                        next_check_at=None,
                        reason="roll_forward_probe_terminal_data_still_ready",
                    )
                )
                continue

            if local_now < expected_publish:
                decisions.append(
                    MarketDataDueDecision(
                        dataset=contract.dataset,
                        status=FRESHNESS_STATUS_READY,
                        due=False,
                        expected_publish_at=expected_publish.isoformat(),
                        latest_expected_date=latest_expected,
                        next_check_at=expected_publish.isoformat(),
                        reason="roll_forward_ready_before_scheduled_probe",
                    )
                )
                continue

            decisions.append(
                MarketDataDueDecision(
                    dataset=contract.dataset,
                    status=FRESHNESS_STATUS_READY,
                    due=True,
                    expected_publish_at=expected_publish.isoformat(),
                    latest_expected_date=latest_expected,
                    next_check_at=local_now.isoformat(),
                    reason="roll_forward_scheduled_probe_due",
                )
            )
            continue

        # Non-ready target-date feeds retain the existing bounded publication
        # retry lifecycle.  A roll-forward feed reaches this branch only before
        # it has any current validated usable state.
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
            retry_at = _state_datetime(persisted_next) or expected_publish
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
