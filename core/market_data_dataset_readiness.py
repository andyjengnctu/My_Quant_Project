"""Canonical dataset-level readiness contract for Trading Market Data V2.

A target date is not READY merely because a persisted date field reached the
requested day.  The same row must carry current schema/request-scope validation
evidence and canonical freshness status.  This module is provider-free and may
be consumed by planners, aggregate rollups, Workbench/readiness gates and sync
resume logic without introducing a second interpretation.
"""
from __future__ import annotations

from typing import Mapping

from core.market_data_freshness_contract import FRESHNESS_STATUS_READY

MARKET_DATA_DATASET_VALIDATION_CONTRACT_VERSION = 2
VALIDATION_STATUS_READY = "READY"
VALIDATION_STATUS_NO_ROW_VALID = "NO_ROW_VALID"
VALID_DATASET_VALIDATION_STATUSES = frozenset(
    {VALIDATION_STATUS_READY, VALIDATION_STATUS_NO_ROW_VALID}
)


def has_current_market_data_dataset_validation(row: Mapping[str, object] | None) -> bool:
    item = row if isinstance(row, Mapping) else {}
    try:
        version = int(item.get("validation_contract_version", -1))
    except (TypeError, ValueError):
        return False
    return bool(
        version == MARKET_DATA_DATASET_VALIDATION_CONTRACT_VERSION
        and str(item.get("schema_status") or "") in VALID_DATASET_VALIDATION_STATUSES
        and str(item.get("coverage_status") or "") in VALID_DATASET_VALIDATION_STATUSES
    )


def is_market_data_dataset_ready(
    row: Mapping[str, object] | None,
    *,
    target_date: str,
) -> bool:
    item = row if isinstance(row, Mapping) else {}
    ready_target = str(item.get("last_ready_target_date") or "").strip()
    return bool(
        has_current_market_data_dataset_validation(item)
        and str(item.get("status") or "") == FRESHNESS_STATUS_READY
        and ready_target
        and ready_target >= str(target_date)
    )


__all__ = [
    "MARKET_DATA_DATASET_VALIDATION_CONTRACT_VERSION",
    "VALIDATION_STATUS_READY",
    "VALIDATION_STATUS_NO_ROW_VALID",
    "VALID_DATASET_VALIDATION_STATUSES",
    "has_current_market_data_dataset_validation",
    "is_market_data_dataset_ready",
]
