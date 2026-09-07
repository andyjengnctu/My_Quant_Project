"""Persistent dataset-level operational state for Trading Market Data V2."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from core.file_integrity import atomic_write_json, canonical_json_sha256, load_json_strict
from core.market_data_bootstrap_requests import build_registry_fingerprint
from core.market_data_dataset_registry import get_market_dataset_specs
from core.market_data_due_planner import (
    plan_market_data_due_datasets,
    resolve_market_data_expected_publish_at,
)
from core.market_data_freshness_contract import (
    EXPECTED_DATE_NONE,
    EXPECTED_DATE_PERIOD_DUE,
    FRESHNESS_STATUS_NOT_APPLICABLE,
    FRESHNESS_STATUS_READY,
    FRESHNESS_STATUS_WAIT_PUBLISH,
    ROW_EXPECTATION_OPTIONAL,
    get_market_data_freshness_contracts,
)
from core.market_data_trading_storage_contract import resolve_trading_market_data_v2_dataset_state_path

TRADING_MARKET_DATA_DATASET_STATE_SCHEMA_VERSION = 1
VALIDATION_STATUS_READY = "READY"
VALIDATION_STATUS_NO_ROW_VALID = "NO_ROW_VALID"
VALIDATION_STATUS_NOT_EVALUATED = "NOT_EVALUATED"
VALIDATION_STATUS_UNKNOWN = "UNKNOWN"


def _empty_dataset_row() -> dict[str, object]:
    return {
        "status": FRESHNESS_STATUS_NOT_APPLICABLE,
        "last_attempt_at": None,
        "last_attempt_target_date": None,
        "last_success_at": None,
        "last_success_target_date": None,
        "last_ready_at": None,
        "last_ready_target_date": None,
        "latest_data_date": None,
        "latest_expected_date": None,
        "expected_publish_at": None,
        "next_check_at": None,
        "schema_status": VALIDATION_STATUS_UNKNOWN,
        "coverage_status": VALIDATION_STATUS_NOT_EVALUATED,
        "last_error": None,
    }


def _canonical_registry_fingerprint() -> str:
    return build_registry_fingerprint(get_market_dataset_specs(included_only=True))


def _validate_state_payload(payload: Mapping[str, object]) -> dict[str, Any]:
    state = dict(payload)
    if int(state.get("schema_version", -1)) != TRADING_MARKET_DATA_DATASET_STATE_SCHEMA_VERSION:
        raise ValueError("Trading Market Data dataset state schema 不相容")
    registry_fp = str(state.get("registry_fingerprint") or "")
    if registry_fp != _canonical_registry_fingerprint():
        raise ValueError("Trading Market Data dataset state registry 與 current registry 已 drift")
    core = {key: value for key, value in state.items() if key != "state_fingerprint"}
    if str(state.get("state_fingerprint") or "") != canonical_json_sha256(core):
        raise ValueError("Trading Market Data dataset state fingerprint 不一致")
    datasets = state.get("datasets")
    if not isinstance(datasets, Mapping):
        raise ValueError("Trading Market Data dataset state.datasets 必須是 mapping")
    expected = {spec.dataset for spec in get_market_dataset_specs(included_only=True)}
    actual = {str(key) for key in datasets}
    if actual != expected:
        raise ValueError(
            f"Trading Market Data dataset state coverage drift: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}"
        )
    return state


def build_initial_market_data_dataset_state(*, updated_at: datetime | None = None) -> dict[str, Any]:
    now = updated_at or datetime.now().astimezone()
    datasets = {
        spec.dataset: _empty_dataset_row()
        for spec in get_market_dataset_specs(included_only=True)
    }
    core: dict[str, Any] = {
        "schema_version": TRADING_MARKET_DATA_DATASET_STATE_SCHEMA_VERSION,
        "role": "trading_market_data_v2_dataset_operational_state",
        "registry_fingerprint": _canonical_registry_fingerprint(),
        "updated_at": now.isoformat(),
        "datasets": datasets,
    }
    return {**core, "state_fingerprint": canonical_json_sha256(core)}


def load_market_data_dataset_state(project_root, *, required: bool = False) -> dict[str, Any] | None:
    path = resolve_trading_market_data_v2_dataset_state_path(project_root)
    if not path.is_file():
        if required:
            raise FileNotFoundError("Trading Market Data dataset state 尚未建立")
        return None
    payload = load_json_strict(path)
    if not isinstance(payload, Mapping):
        raise ValueError("Trading Market Data dataset state 必須是 object")
    return _validate_state_payload(payload)


def publish_market_data_dataset_state(project_root, payload: Mapping[str, object]) -> dict[str, Any]:
    state = dict(payload)
    state["schema_version"] = TRADING_MARKET_DATA_DATASET_STATE_SCHEMA_VERSION
    state["role"] = "trading_market_data_v2_dataset_operational_state"
    state["registry_fingerprint"] = _canonical_registry_fingerprint()
    core = {key: value for key, value in state.items() if key != "state_fingerprint"}
    state = {**core, "state_fingerprint": canonical_json_sha256(core)}
    validated = _validate_state_payload(state)
    atomic_write_json(resolve_trading_market_data_v2_dataset_state_path(project_root), validated)
    return validated


def _later_date(a: object, b: object) -> str | None:
    values = [str(value).strip() for value in (a, b) if str(value or "").strip()]
    return max(values) if values else None


def record_market_data_sync_success(
    project_root,
    *,
    target_date: str,
    finished_at: datetime,
    observations: Mapping[str, Mapping[str, object]] | None = None,
    attempted_datasets: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Persist dynamic evidence from a fully completed Trading V2 sync batch."""

    previous = load_market_data_dataset_state(project_root, required=False)
    state = previous or build_initial_market_data_dataset_state(updated_at=finished_at)
    datasets = {key: dict(value) for key, value in dict(state["datasets"]).items()}
    observed = observations or {}
    contracts = {item.dataset: item for item in get_market_data_freshness_contracts()}
    attempted = set(contracts) if attempted_datasets is None else {str(item) for item in attempted_datasets}
    unknown = attempted - set(contracts)
    if unknown:
        raise ValueError(f"sync attempted_datasets 含未知 dataset: {sorted(unknown)}")

    for dataset, contract in contracts.items():
        if dataset not in attempted:
            continue
        row = dict(datasets[dataset])
        evidence = observed.get(dataset) if isinstance(observed.get(dataset), Mapping) else {}
        row_count = int(evidence.get("row_count") or 0)
        observed_max = str(evidence.get("observed_max_date") or "").strip() or None
        row["last_attempt_at"] = finished_at.isoformat()
        row["last_attempt_target_date"] = str(target_date)
        row["last_success_at"] = finished_at.isoformat()
        row["last_success_target_date"] = str(target_date)
        row["latest_data_date"] = _later_date(row.get("latest_data_date"), observed_max)
        row["latest_expected_date"] = (
            None
            if contract.expected_date_mode in {EXPECTED_DATE_NONE, EXPECTED_DATE_PERIOD_DUE}
            else str(target_date)
        )
        row["expected_publish_at"] = resolve_market_data_expected_publish_at(
            contract, target_date=str(target_date)
        ).isoformat()
        row["schema_status"] = (
            VALIDATION_STATUS_READY
            if row_count > 0
            else (VALIDATION_STATUS_NO_ROW_VALID if contract.row_expectation == ROW_EXPECTATION_OPTIONAL else VALIDATION_STATUS_UNKNOWN)
        )
        row["coverage_status"] = (
            VALIDATION_STATUS_NO_ROW_VALID
            if row_count == 0 and contract.row_expectation == ROW_EXPECTATION_OPTIONAL
            else VALIDATION_STATUS_NOT_EVALUATED
        )

        if contract.expected_date_mode in {EXPECTED_DATE_NONE, EXPECTED_DATE_PERIOD_DUE}:
            ready = row_count > 0 or contract.row_expectation == ROW_EXPECTATION_OPTIONAL
        else:
            ready = bool(observed_max and observed_max >= str(target_date))

        if ready:
            row["status"] = FRESHNESS_STATUS_READY
            row["last_ready_at"] = finished_at.isoformat()
            row["last_ready_target_date"] = str(target_date)
            row["next_check_at"] = None
            row["last_error"] = None
        else:
            row["status"] = FRESHNESS_STATUS_WAIT_PUBLISH
            row["next_check_at"] = None
            row["last_error"] = "sync requests completed but target-date freshness evidence is not yet present"
        datasets[dataset] = row

    return publish_market_data_dataset_state(
        project_root,
        {
            **{key: value for key, value in state.items() if key not in {"state_fingerprint", "datasets", "updated_at"}},
            "updated_at": finished_at.isoformat(),
            "datasets": datasets,
        },
    )




def refresh_market_data_due_state(
    project_root,
    *,
    target_date: str,
    now: datetime,
):
    """Persist a local-only due-plan projection without any provider request."""

    state = load_market_data_dataset_state(project_root, required=False)
    if state is None:
        state = build_initial_market_data_dataset_state(updated_at=now)
    plan = plan_market_data_due_datasets(target_date=target_date, now=now, state=state)
    datasets = {key: dict(value) for key, value in dict(state["datasets"]).items()}
    for decision in plan.decisions:
        row = dict(datasets[decision.dataset])
        row["status"] = decision.status
        row["latest_expected_date"] = decision.latest_expected_date
        row["expected_publish_at"] = decision.expected_publish_at
        row["next_check_at"] = decision.next_check_at
        datasets[decision.dataset] = row
    persisted = publish_market_data_dataset_state(
        project_root,
        {
            **{key: value for key, value in state.items() if key not in {"state_fingerprint", "datasets", "updated_at"}},
            "updated_at": now.isoformat(),
            "datasets": datasets,
        },
    )
    return plan, persisted


def build_market_data_due_plan(
    project_root,
    *,
    target_date: str,
    now: datetime,
):
    """Local-only read/plan facade used by future scheduler and Workbench Data Ops."""

    state = load_market_data_dataset_state(project_root, required=False)
    return plan_market_data_due_datasets(target_date=target_date, now=now, state=state)


def build_market_data_dataset_state_read_model(project_root) -> dict[str, object]:
    state = load_market_data_dataset_state(project_root, required=False)
    contracts = {item.dataset: item for item in get_market_data_freshness_contracts()}
    if state is None:
        return {
            "state_ready": False,
            "dataset_count": len(contracts),
            "datasets": [],
        }
    rows = []
    for dataset in sorted(contracts):
        contract = contracts[dataset]
        dynamic = dict(state["datasets"][dataset])
        rows.append(
            {
                "dataset": dataset,
                "category": contract.category,
                "cadence": contract.cadence,
                "publication_first_check_time": contract.publication_first_check_time,
                "publication_day_offset": contract.publication_day_offset,
                "publication_schedule_verified": contract.publication_schedule_verified,
                **dynamic,
            }
        )
    return {
        "state_ready": True,
        "updated_at": state.get("updated_at"),
        "dataset_count": len(rows),
        "datasets": rows,
    }


__all__ = [
    "TRADING_MARKET_DATA_DATASET_STATE_SCHEMA_VERSION",
    "VALIDATION_STATUS_READY",
    "VALIDATION_STATUS_NO_ROW_VALID",
    "VALIDATION_STATUS_NOT_EVALUATED",
    "VALIDATION_STATUS_UNKNOWN",
    "build_initial_market_data_dataset_state",
    "load_market_data_dataset_state",
    "publish_market_data_dataset_state",
    "record_market_data_sync_success",
    "refresh_market_data_due_state",
    "build_market_data_due_plan",
    "build_market_data_dataset_state_read_model",
]
