"""Persistent dataset-level operational state for Trading Market Data V2."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

from core.file_integrity import atomic_write_json, canonical_json_sha256, load_json_strict
from core.market_data_bootstrap_requests import build_registry_fingerprint
from core.market_data_dataset_registry import get_market_dataset_spec, get_market_dataset_specs
from core.market_data_dataset_readiness import (
    MARKET_DATA_DATASET_VALIDATION_CONTRACT_VERSION,
    VALIDATION_STATUS_NO_ROW_VALID,
    VALIDATION_STATUS_READY,
    has_current_market_data_dataset_validation,
    is_market_data_dataset_ready,
    market_data_contract_requires_target_freshness,
)
from core.market_data_due_planner import (
    normalize_market_data_planner_now,
    plan_market_data_due_datasets,
    resolve_market_data_expected_publish_at,
)
from core.market_data_freshness_contract import (
    EXPECTED_DATE_LATEST_AVAILABLE,
    EXPECTED_DATE_NONE,
    EXPECTED_DATE_PERIOD_DUE,
    EXPECTED_DATE_TRADING_TARGET,
    FRESHNESS_STATUS_BLOCKED,
    FRESHNESS_STATUS_ERROR,
    FRESHNESS_STATUS_NOT_APPLICABLE,
    FRESHNESS_STATUS_READY,
    FRESHNESS_STATUS_STALE,
    FRESHNESS_STATUS_WAIT_PUBLISH,
    FRESHNESS_STATUS_WAIT_QUOTA,
    ROW_EXPECTATION_OPTIONAL,
    ROW_EXPECTATION_REQUIRED,
    get_market_data_freshness_contracts,
)
from core.market_data_trading_storage_contract import resolve_trading_market_data_v2_dataset_state_path

TRADING_MARKET_DATA_DATASET_STATE_SCHEMA_VERSION = 2
TRADING_MARKET_DATA_DATASET_STATE_LEGACY_SCHEMA_VERSION = 1
VALIDATION_STATUS_NOT_EVALUATED = "NOT_EVALUATED"
VALIDATION_STATUS_UNKNOWN = "UNKNOWN"
PREVIOUS_MARKET_DATA_DATASET_VALIDATION_CONTRACT_VERSION = 2


def _effective_trading_data_ids(dataset: str) -> tuple[str, ...]:
    spec = get_market_dataset_spec(dataset)
    return tuple(spec.trading_fixed_data_ids or spec.fixed_data_ids)


def _v2_ready_evidence_is_safe_to_upgrade(dataset: str, row: Mapping[str, object]) -> bool:
    """Upgrade only v2 READY evidence that already proved the whole target scope.

    v2 tracked only aggregate max-date evidence.  That was sufficient for
    full-market exact-date requests and single-lane range requests, but not for
    multi-data-id target-date ranges where one fresh lane could mask another.
    """

    contract = next(item for item in get_market_data_freshness_contracts() if item.dataset == dataset)
    spec = get_market_dataset_spec(dataset)
    if spec.full_market_exact_date_expected:
        return True
    # v2 aggregate evidence could not prove every lane of a multi-data-id
    # request set, regardless of READY/WAIT state.  Keep those rows on v2 so
    # the next sync performs one targeted revalidation under lane coverage v3.
    if len(_effective_trading_data_ids(dataset)) > 1:
        return False
    if str(row.get("status") or "") != FRESHNESS_STATUS_READY:
        return True
    return contract.expected_date_mode in {
        EXPECTED_DATE_TRADING_TARGET,
        EXPECTED_DATE_LATEST_AVAILABLE,
        EXPECTED_DATE_NONE,
        EXPECTED_DATE_PERIOD_DUE,
    }


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
        "latest_data_date_by_data_id": {},
        "latest_expected_date": None,
        "expected_publish_at": None,
        "next_check_at": None,
        "validation_contract_version": MARKET_DATA_DATASET_VALIDATION_CONTRACT_VERSION,
        "schema_status": VALIDATION_STATUS_UNKNOWN,
        "coverage_status": VALIDATION_STATUS_NOT_EVALUATED,
        "publication_retry_count": 0,
        "quota_defer_count": 0,
        "error_retry_count": 0,
        "last_attempt_result": None,
        "last_error": None,
    }


def _canonical_registry_fingerprint() -> str:
    return build_registry_fingerprint(get_market_dataset_specs(included_only=True))


def _validate_state_payload(payload: Mapping[str, object]) -> dict[str, Any]:
    state = dict(payload)
    schema_version = int(state.get("schema_version", -1))
    if schema_version not in {
        TRADING_MARKET_DATA_DATASET_STATE_LEGACY_SCHEMA_VERSION,
        TRADING_MARKET_DATA_DATASET_STATE_SCHEMA_VERSION,
    }:
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

    if schema_version == TRADING_MARKET_DATA_DATASET_STATE_LEGACY_SCHEMA_VERSION:
        # Preserve dates as resume evidence but do not let the legacy
        # "date reached == READY" contract authorize current Trading.  The next
        # due-plan will revalidate these rows under the current request-scope
        # contract and persist schema v2.
        migrated_rows: dict[str, dict[str, object]] = {}
        for dataset, raw in datasets.items():
            row = dict(raw) if isinstance(raw, Mapping) else _empty_dataset_row()
            row["validation_contract_version"] = TRADING_MARKET_DATA_DATASET_STATE_LEGACY_SCHEMA_VERSION
            migrated_rows[str(dataset)] = row
        migrated_core = {
            **{key: value for key, value in state.items() if key not in {"schema_version", "state_fingerprint", "datasets"}},
            "schema_version": TRADING_MARKET_DATA_DATASET_STATE_SCHEMA_VERSION,
            "datasets": migrated_rows,
        }
        return {**migrated_core, "state_fingerprint": canonical_json_sha256(migrated_core)}

    migrated_rows: dict[str, dict[str, object]] = {}
    migrated = False
    for dataset, raw in datasets.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"Trading Market Data dataset state row 不是 object: {dataset}")
        row = dict(raw)
        try:
            validation_version = int(row.get("validation_contract_version", -1))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Trading Market Data dataset validation contract 不合法: {dataset}") from exc
        if validation_version > MARKET_DATA_DATASET_VALIDATION_CONTRACT_VERSION:
            raise ValueError(f"Trading Market Data dataset validation contract 來自較新版本: {dataset}")
        if validation_version == PREVIOUS_MARKET_DATA_DATASET_VALIDATION_CONTRACT_VERSION:
            migrated = True
            if _v2_ready_evidence_is_safe_to_upgrade(str(dataset), row):
                row["validation_contract_version"] = MARKET_DATA_DATASET_VALIDATION_CONTRACT_VERSION
            else:
                # Keep the old validation version so canonical readiness forces a
                # one-time targeted re-query under request-lane coverage v3.
                row["validation_contract_version"] = PREVIOUS_MARKET_DATA_DATASET_VALIDATION_CONTRACT_VERSION
            if str(row.get("status") or "") != FRESHNESS_STATUS_READY or int(row["validation_contract_version"]) != MARKET_DATA_DATASET_VALIDATION_CONTRACT_VERSION:
                row["next_check_at"] = None
                row["publication_retry_count"] = 0
        migrated_rows[str(dataset)] = row
    if migrated:
        migrated_core = {
            **{key: value for key, value in state.items() if key not in {"state_fingerprint", "datasets"}},
            "datasets": migrated_rows,
        }
        return {**migrated_core, "state_fingerprint": canonical_json_sha256(migrated_core)}
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


def _prior_ready_validation_can_self_heal(
    row: Mapping[str, object],
    *,
    contract,
    target_date: str,
) -> bool:
    """Recover structural validation without authorizing newer freshness.

    A prior ``last_ready_target_date`` was written only after schema and
    request-scope validation both passed.  If a later WAIT_PUBLISH observation
    (including legacy manual-force or zero-provider ledger reuse) erased those
    structural fields, the prior READY horizon is sufficient to restore them.
    Freshness remains fail-closed because ``last_ready_target_date`` and status
    are not advanced by this repair.
    """

    try:
        validation_version = int(row.get("validation_contract_version", -1))
    except (TypeError, ValueError):
        return False
    last_ready_target = str(row.get("last_ready_target_date") or "").strip()
    latest_data_date = str(row.get("latest_data_date") or "").strip()
    return bool(
        validation_version == MARKET_DATA_DATASET_VALIDATION_CONTRACT_VERSION
        and contract.expected_date_mode == EXPECTED_DATE_TRADING_TARGET
        and contract.row_expectation == ROW_EXPECTATION_REQUIRED
        and str(row.get("status") or "") == FRESHNESS_STATUS_WAIT_PUBLISH
        and bool(last_ready_target)
        and last_ready_target < str(target_date)
        and bool(latest_data_date)
        and latest_data_date >= last_ready_target
    )


def record_market_data_sync_success(
    project_root,
    *,
    target_date: str,
    finished_at: datetime,
    observations: Mapping[str, Mapping[str, object]] | None = None,
    attempted_datasets: Iterable[str] | None = None,
    count_publication_retry: bool = True,
) -> dict[str, Any]:
    """Persist evidence from a fully completed Trading V2 sync batch.

    A successful provider request is not automatically a freshness transition.
    Exact target-date feeds must prove current-target rows.  Roll-forward feeds
    may keep prior validated data READY while the independent scheduler still
    owns publication-window checks/retries.  Manual force observations never
    consume or reschedule that automatic retry lifecycle.
    """

    previous = load_market_data_dataset_state(project_root, required=False)
    state = previous or build_initial_market_data_dataset_state(updated_at=finished_at)
    datasets = {key: dict(value) for key, value in dict(state["datasets"]).items()}
    observed = observations or {}
    contracts = {item.dataset: item for item in get_market_data_freshness_contracts()}
    attempted = set(contracts) if attempted_datasets is None else {str(item) for item in attempted_datasets}
    unknown = attempted - set(contracts)
    if unknown:
        raise ValueError(f"sync attempted_datasets 含未知 dataset: {sorted(unknown)}")

    finished_local = normalize_market_data_planner_now(finished_at)
    target_text = str(target_date)

    for dataset, contract in contracts.items():
        if dataset not in attempted:
            continue
        row = dict(datasets[dataset])
        prior_attempt_target = str(row.get("last_attempt_target_date") or "")
        prior_publication_retry_count = int(row.get("publication_retry_count") or 0)
        prior_validation_current = has_current_market_data_dataset_validation(row)
        prior_validation_recoverable = _prior_ready_validation_can_self_heal(
            row, contract=contract, target_date=target_text
        )
        prior_schema_status = str(row.get("schema_status") or VALIDATION_STATUS_UNKNOWN)
        prior_coverage_status = str(row.get("coverage_status") or VALIDATION_STATUS_NOT_EVALUATED)
        prior_data_usable = is_market_data_dataset_ready(
            row,
            target_date=target_text,
            contract=contract,
        )
        target_required = market_data_contract_requires_target_freshness(contract)

        evidence = observed.get(dataset) if isinstance(observed.get(dataset), Mapping) else {}
        row_count = int(evidence.get("row_count") or 0)
        observed_max = str(evidence.get("observed_max_date") or "").strip() or None
        observed_lanes = evidence.get("observed_max_date_by_data_id")
        observed_lanes = dict(observed_lanes) if isinstance(observed_lanes, Mapping) else {}
        request_count = int(evidence.get("request_count") or (1 if row_count > 0 else 0))
        nonempty_request_count = int(evidence.get("nonempty_request_count") or (1 if row_count > 0 else 0))
        target_covering_request_count = int(
            evidence.get("target_covering_request_count")
            or (1 if observed_max is not None else 0)
        )
        target_fresh_request_count = int(
            evidence.get("target_fresh_request_count")
            or (1 if observed_max is not None and observed_max >= target_text else 0)
        )

        row["last_attempt_at"] = finished_at.isoformat()
        row["last_attempt_target_date"] = target_text
        # ``last_success`` is retained as persisted compatibility evidence for a
        # completed provider batch. Workbench displays ``last_attempt_at`` as
        # Last Check so request success is not confused with content freshness.
        row["last_success_at"] = finished_at.isoformat()
        row["last_success_target_date"] = target_text
        row["latest_data_date"] = _later_date(row.get("latest_data_date"), observed_max)
        lane_dates = dict(row.get("latest_data_date_by_data_id") or {})
        for lane_key, lane_value in observed_lanes.items():
            lane_text = str(lane_value or "").strip()
            if not lane_text:
                continue
            prior_lane = str(lane_dates.get(str(lane_key)) or "").strip()
            lane_dates[str(lane_key)] = lane_text if not prior_lane else max(prior_lane, lane_text)
        row["latest_data_date_by_data_id"] = lane_dates
        if contract.expected_date_mode in {EXPECTED_DATE_NONE, EXPECTED_DATE_PERIOD_DUE}:
            row["latest_expected_date"] = None
        elif contract.expected_date_mode == EXPECTED_DATE_LATEST_AVAILABLE:
            row["latest_expected_date"] = row.get("latest_data_date")
        else:
            row["latest_expected_date"] = target_text
        expected_publish = resolve_market_data_expected_publish_at(contract, target_date=target_text)
        row["expected_publish_at"] = expected_publish.isoformat()
        row["validation_contract_version"] = MARKET_DATA_DATASET_VALIDATION_CONTRACT_VERSION

        if row_count > 0:
            schema_status = VALIDATION_STATUS_READY
        elif contract.row_expectation == ROW_EXPECTATION_OPTIONAL:
            schema_status = VALIDATION_STATUS_NO_ROW_VALID
        elif prior_validation_current:
            schema_status = prior_schema_status
        elif prior_validation_recoverable:
            schema_status = VALIDATION_STATUS_READY
        else:
            schema_status = VALIDATION_STATUS_UNKNOWN

        if contract.row_expectation == ROW_EXPECTATION_OPTIONAL and row_count == 0:
            coverage_status = VALIDATION_STATUS_NO_ROW_VALID
        elif contract.expected_date_mode == EXPECTED_DATE_TRADING_TARGET:
            if target_covering_request_count > 0:
                coverage_status = VALIDATION_STATUS_READY
            elif prior_validation_current:
                coverage_status = prior_coverage_status
            elif prior_validation_recoverable:
                coverage_status = VALIDATION_STATUS_READY
            else:
                coverage_status = VALIDATION_STATUS_NOT_EVALUATED
        elif contract.expected_date_mode == EXPECTED_DATE_LATEST_AVAILABLE:
            # Coverage is structural evidence. A temporarily empty current poll
            # does not erase previously proven request-scope validation.
            current_scope_ready = bool(
                request_count > 0
                and nonempty_request_count > 0
                and nonempty_request_count == request_count
            )
            coverage_status = (
                VALIDATION_STATUS_READY
                if current_scope_ready
                else prior_coverage_status
                if prior_validation_current
                else VALIDATION_STATUS_NOT_EVALUATED
            )
        else:
            coverage_status = (
                VALIDATION_STATUS_READY
                if row_count > 0
                else prior_coverage_status
                if prior_validation_current
                else VALIDATION_STATUS_NOT_EVALUATED
            )

        validation_ready = bool(
            schema_status in {VALIDATION_STATUS_READY, VALIDATION_STATUS_NO_ROW_VALID}
            and coverage_status in {VALIDATION_STATUS_READY, VALIDATION_STATUS_NO_ROW_VALID}
        )
        if contract.expected_date_mode in {EXPECTED_DATE_NONE, EXPECTED_DATE_PERIOD_DUE}:
            observation_ready = bool(
                validation_ready
                and (row_count > 0 or contract.row_expectation == ROW_EXPECTATION_OPTIONAL)
            )
        elif contract.expected_date_mode == EXPECTED_DATE_LATEST_AVAILABLE:
            current_scope_observed = bool(
                request_count > 0
                and nonempty_request_count > 0
                and nonempty_request_count == request_count
            )
            observation_ready = bool(
                validation_ready
                and current_scope_observed
                and str(row.get("latest_data_date") or "").strip()
            )
        else:
            target_freshness_ready = bool(
                target_covering_request_count > 0
                and target_fresh_request_count == target_covering_request_count
            )
            observation_ready = bool(validation_ready and target_freshness_ready)

        roll_forward_ready = bool(not target_required and prior_data_usable)
        data_ready = bool(observation_ready or roll_forward_ready)

        if not observation_ready and (prior_validation_current or prior_validation_recoverable):
            # A weak/early provider observation must never erase previously
            # proven structural/request-scope evidence.
            if prior_validation_current:
                schema_status = prior_schema_status
                coverage_status = prior_coverage_status
            else:
                schema_status = VALIDATION_STATUS_READY
                coverage_status = VALIDATION_STATUS_READY

        row["schema_status"] = schema_status
        row["coverage_status"] = coverage_status
        row["last_attempt_result"] = "SUCCESS" if observation_ready else FRESHNESS_STATUS_WAIT_PUBLISH
        row["quota_defer_count"] = 0
        row["error_retry_count"] = 0

        if data_ready:
            row["status"] = FRESHNESS_STATUS_READY
            row["last_ready_at"] = finished_at.isoformat()
            row["last_ready_target_date"] = target_text
            if observation_ready:
                if not target_required and finished_local < expected_publish:
                    # An early manual observation may find valid/no-change data,
                    # but it must not consume the later scheduled probe window.
                    row["next_check_at"] = expected_publish.isoformat()
                    if not count_publication_retry:
                        row["publication_retry_count"] = prior_publication_retry_count
                    else:
                        row["publication_retry_count"] = 0
                else:
                    row["next_check_at"] = None
                    row["publication_retry_count"] = 0
                row["last_error"] = None
            else:
                prior_count = prior_publication_retry_count if prior_attempt_target == target_text else 0
                if count_publication_retry:
                    row["publication_retry_count"] = prior_count + 1
                    row["next_check_at"] = None
                else:
                    row["publication_retry_count"] = prior_count
                    if finished_local < expected_publish:
                        row["next_check_at"] = expected_publish.isoformat()
                row["last_error"] = (
                    "provider observation did not satisfy the current probe contract; "
                    "prior validated roll-forward data remains usable"
                )
        else:
            prior_count = prior_publication_retry_count if prior_attempt_target == target_text else 0
            row["status"] = FRESHNESS_STATUS_WAIT_PUBLISH
            if count_publication_retry:
                row["publication_retry_count"] = prior_count + 1
                row["next_check_at"] = None
            else:
                row["publication_retry_count"] = prior_count
                if finished_local < expected_publish:
                    row["next_check_at"] = expected_publish.isoformat()
            row["last_error"] = "sync requests completed but canonical freshness/request-scope evidence is not yet present"
        datasets[dataset] = row

    return publish_market_data_dataset_state(
        project_root,
        {
            **{key: value for key, value in state.items() if key not in {"state_fingerprint", "datasets", "updated_at"}},
            "updated_at": finished_at.isoformat(),
            "datasets": datasets,
        },
    )



def schedule_market_data_auto_update_outcomes(
    project_root,
    *,
    target_date: str,
    now: datetime,
    publication_retry_minutes: tuple[int, ...],
    max_publication_retries: int,
    quota_defer_minutes: int,
    error_defer_minutes: int,
    wait_publish_datasets: Iterable[str] = (),
    wait_quota_datasets: Iterable[str] = (),
    error_datasets: Iterable[str] = (),
    blocked_datasets: Iterable[str] = (),
    error_message: str | None = None,
) -> dict[str, Any]:
    """Persist auto-updater probe outcomes without redefining data usability.

    Exact target-date datasets fail closed while their target freshness is
    missing. Roll-forward datasets keep validated prior data READY and record
    retry/error state through ``last_attempt_result`` + ``next_check_at``.
    """

    state = load_market_data_dataset_state(project_root, required=False)
    if state is None:
        state = build_initial_market_data_dataset_state(updated_at=now)
    datasets = {key: dict(value) for key, value in dict(state["datasets"]).items()}
    contracts = {item.dataset: item for item in get_market_data_freshness_contracts()}
    known = set(datasets)
    groups = {
        "wait_publish": {str(item) for item in wait_publish_datasets},
        "wait_quota": {str(item) for item in wait_quota_datasets},
        "error": {str(item) for item in error_datasets},
        "blocked": {str(item) for item in blocked_datasets},
    }
    unknown = set().union(*groups.values()) - known
    if unknown:
        raise ValueError(f"auto update outcome 含未知 dataset: {sorted(unknown)}")
    overlaps: dict[str, set[str]] = {}
    names = tuple(groups)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            common = groups[left] & groups[right]
            if common:
                overlaps[f"{left}/{right}"] = common
    if overlaps:
        raise ValueError(f"auto update outcome dataset 重複分類: {overlaps}")
    if max_publication_retries < 1 or len(publication_retry_minutes) < max_publication_retries:
        raise ValueError("publication retry policy 不合法")

    def _preserve_roll_forward_ready(dataset: str, row: Mapping[str, object]) -> bool:
        contract = contracts[dataset]
        return bool(
            not market_data_contract_requires_target_freshness(contract)
            and is_market_data_dataset_ready(
                row,
                target_date=str(target_date),
                contract=contract,
            )
        )

    for dataset in groups["wait_publish"]:
        row = dict(datasets[dataset])
        preserve_ready = _preserve_roll_forward_ready(dataset, row)
        count = int(row.get("publication_retry_count") or 0)
        row["last_attempt_at"] = now.isoformat()
        row["last_attempt_target_date"] = str(target_date)
        if count > max_publication_retries:
            row["status"] = FRESHNESS_STATUS_READY if preserve_ready else FRESHNESS_STATUS_STALE
            row["last_attempt_result"] = FRESHNESS_STATUS_STALE
            row["next_check_at"] = None
            row["last_error"] = f"publication retry exhausted: {max_publication_retries} retries after initial attempt"
        else:
            delay = int(publication_retry_minutes[min(max(count - 1, 0), len(publication_retry_minutes) - 1)])
            row["status"] = FRESHNESS_STATUS_READY if preserve_ready else FRESHNESS_STATUS_WAIT_PUBLISH
            row["last_attempt_result"] = FRESHNESS_STATUS_WAIT_PUBLISH
            row["next_check_at"] = (now + timedelta(minutes=delay)).isoformat()
            row["last_error"] = (
                "provider probe evidence not yet sufficient; prior validated roll-forward data remains usable"
                if preserve_ready
                else "provider target-date freshness evidence not yet present"
            )
        datasets[dataset] = row

    for dataset in groups["wait_quota"]:
        row = dict(datasets[dataset])
        preserve_ready = _preserve_roll_forward_ready(dataset, row)
        row["status"] = FRESHNESS_STATUS_READY if preserve_ready else FRESHNESS_STATUS_WAIT_QUOTA
        row["last_attempt_at"] = now.isoformat()
        row["last_attempt_target_date"] = str(target_date)
        row["last_attempt_result"] = FRESHNESS_STATUS_WAIT_QUOTA
        row["quota_defer_count"] = int(row.get("quota_defer_count") or 0) + 1
        row["next_check_at"] = (now + timedelta(minutes=int(quota_defer_minutes))).isoformat()
        row["last_error"] = "provider quota insufficient; deferred by one-shot updater"
        datasets[dataset] = row

    for dataset in groups["error"]:
        row = dict(datasets[dataset])
        preserve_ready = _preserve_roll_forward_ready(dataset, row)
        row["status"] = FRESHNESS_STATUS_READY if preserve_ready else FRESHNESS_STATUS_ERROR
        row["last_attempt_at"] = now.isoformat()
        row["last_attempt_target_date"] = str(target_date)
        row["last_attempt_result"] = FRESHNESS_STATUS_ERROR
        row["error_retry_count"] = int(row.get("error_retry_count") or 0) + 1
        row["next_check_at"] = (now + timedelta(minutes=int(error_defer_minutes))).isoformat()
        row["last_error"] = str(error_message or "automatic updater error")
        datasets[dataset] = row

    for dataset in groups["blocked"]:
        row = dict(datasets[dataset])
        preserve_ready = _preserve_roll_forward_ready(dataset, row)
        row["status"] = FRESHNESS_STATUS_READY if preserve_ready else FRESHNESS_STATUS_BLOCKED
        row["last_attempt_at"] = now.isoformat()
        row["last_attempt_target_date"] = str(target_date)
        row["last_attempt_result"] = FRESHNESS_STATUS_BLOCKED
        row["next_check_at"] = None
        row["last_error"] = str(error_message or "automatic updater blocked")
        datasets[dataset] = row

    return publish_market_data_dataset_state(
        project_root,
        {
            **{key: value for key, value in state.items() if key not in {"state_fingerprint", "datasets", "updated_at"}},
            "updated_at": now.isoformat(),
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
    contracts = {item.dataset: item for item in get_market_data_freshness_contracts()}
    for decision in plan.decisions:
        row = dict(datasets[decision.dataset])
        contract = contracts[decision.dataset]
        prior_roll_forward_ready = bool(
            not market_data_contract_requires_target_freshness(contract)
            and is_market_data_dataset_ready(
                row,
                target_date=str(target_date),
                contract=contract,
            )
        )
        row["status"] = decision.status
        row["latest_expected_date"] = decision.latest_expected_date
        row["expected_publish_at"] = decision.expected_publish_at
        row["next_check_at"] = decision.next_check_at
        if decision.reason in {
            "before_publication_window",
            "roll_forward_ready_before_scheduled_probe",
        }:
            # Publication timing is scheduler metadata. Before that window no
            # automatic retry budget should be consumed merely because a manual
            # force observation happened early.
            row["publication_retry_count"] = 0
            row["last_error"] = None
        if prior_roll_forward_ready and decision.status == FRESHNESS_STATUS_READY:
            # ``last_ready_target_date`` is a ready-through horizon, not a claim
            # that provider content has a row dated target_date.  Roll-forward
            # contracts may therefore extend this horizon locally while the
            # independent provider probe remains scheduled via next_check_at.
            prior_ready_target = str(row.get("last_ready_target_date") or "").strip()
            row["last_ready_target_date"] = (
                str(target_date)
                if not prior_ready_target
                else max(prior_ready_target, str(target_date))
            )
        if _prior_ready_validation_can_self_heal(
            row, contract=contract, target_date=str(target_date)
        ):
            # Repair validation fields erased by the pre-fix manual `R` path.
            # Target-date-required freshness still fails closed until provider
            # evidence arrives; this repair only restores structural evidence.
            row["schema_status"] = VALIDATION_STATUS_READY
            row["coverage_status"] = VALIDATION_STATUS_READY
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
    rows = []
    for dataset in sorted(contracts):
        contract = contracts[dataset]
        dynamic = _empty_dataset_row() if state is None else dict(state["datasets"][dataset])
        rows.append(
            {
                "dataset": dataset,
                "category": contract.category,
                "cadence": contract.cadence,
                "expected_date_mode": contract.expected_date_mode,
                "row_expectation": contract.row_expectation,
                "completeness_mode": contract.completeness_mode,
                "schema_validation_required": contract.schema_validation_required,
                "publication_first_check_time": contract.publication_first_check_time,
                "publication_day_offset": contract.publication_day_offset,
                "publication_schedule_source": contract.publication_schedule_source,
                "publication_schedule_verified": contract.publication_schedule_verified,
                "primary_key_hint": contract.primary_key_hint,
                **dynamic,
            }
        )
    return {
        "state_ready": state is not None,
        "updated_at": None if state is None else state.get("updated_at"),
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
    "schedule_market_data_auto_update_outcomes",
    "refresh_market_data_due_state",
    "build_market_data_due_plan",
    "build_market_data_dataset_state_read_model",
]
