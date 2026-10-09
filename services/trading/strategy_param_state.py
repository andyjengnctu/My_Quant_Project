"""Trading-only lineage binding for canonical Optimizer strategy parameters."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from core.seed_ensemble_policy import SEED_ENSEMBLE_MAX_SEED_VALUE
from core.console_report import project_relative_display_path
from core.file_integrity import atomic_write_json, canonical_json_sha256, compute_file_sha256, load_json_strict
from core.runtime_domains import RUNTIME_DOMAIN_TRADING, resolve_runtime_domain_paths
from core.trading_policy import get_trading_strategy_profile, resolve_trading_selected_strategy_param_path
from services.trading.strategy_param_runtime import load_trading_strategy_param_runtime
from services.trading.market_data_consumer import (
    get_trading_v2_consumer_state_sha256,
    load_trading_v2_consumer_state,
)

TRADING_STRATEGY_PARAM_BINDING_SCHEMA_VERSION = 9
TRADING_STRATEGY_PARAM_BINDING_FILENAME = "trading_param_binding.json"
TRADING_PARAM_USAGE_TRAINED_CURRENT = "trained_current"
TRADING_PARAM_USAGE_REUSE_EXISTING = "reuse_existing"
TRADING_PARAM_USAGE_MODES = frozenset({TRADING_PARAM_USAGE_TRAINED_CURRENT, TRADING_PARAM_USAGE_REUSE_EXISTING})
TRADING_PARAM_SCHEDULE_MODES = frozenset({"static", "rolling"})


class TradingParamContractMismatchError(RuntimeError):
    """The selected Trading Params were fitted under a different producer contract."""


def resolve_trading_strategy_param_binding_path(project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    profile = get_trading_strategy_profile()
    paths = resolve_runtime_domain_paths(root, domain=RUNTIME_DOMAIN_TRADING, dataset_profile=profile.dataset_profile)
    return Path(paths.strategy_params_root) / TRADING_STRATEGY_PARAM_BINDING_FILENAME


def _validate_binding(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise TypeError("Trading param binding 必須是 object")
    if int(payload.get("schema_version", -1)) != TRADING_STRATEGY_PARAM_BINDING_SCHEMA_VERSION:
        raise ValueError("Trading param binding schema 不相容；請重新套用 Trading Params")
    if str(payload.get("runtime_domain") or "") != RUNTIME_DOMAIN_TRADING:
        raise ValueError("Trading param binding runtime domain 不合法")
    for field in (
        "strategy_id",
        "strategy_param_profile_id",
        "parameter_set_identity",
        "registered_scientific_fingerprint_sha256",
        "param_selector",
        "selected_params_sha256",
        "market_data_consumer_state_sha256",
        "market_data_source_view_fingerprint",
        "latest_data_date",
        "param_training_data_date",
        "param_schedule_mode",
        "usage_mode",
        "optimizer_seed_mode",
    ):
        if not str(payload.get(field) or ""):
            raise ValueError(f"Trading param binding 缺少 {field}")
    for field in (
        "param_member_count",
        "param_min_agree",
        "optimizer_trials_per_seed",
        "optimizer_seed_count",
        "optimizer_resolved_seeds",
    ):
        if field not in payload or payload.get(field) is None:
            raise ValueError(f"Trading param binding 缺少 {field}")
    if str(payload.get("usage_mode")) not in TRADING_PARAM_USAGE_MODES:
        raise ValueError("Trading param binding usage_mode 不合法")
    schedule_mode = str(payload.get("param_schedule_mode") or "").strip().lower()
    if schedule_mode not in TRADING_PARAM_SCHEDULE_MODES:
        raise ValueError("Trading param binding param_schedule_mode 不合法")
    member_count = int(payload.get("param_member_count") or 0)
    min_agree = int(payload.get("param_min_agree") or 0)
    if member_count < 1 or min_agree < 1 or min_agree > member_count:
        raise ValueError("Trading param binding Params ensemble metadata 不合法")
    optimizer_seed_count = int(payload.get("optimizer_seed_count") or 0)
    optimizer_trials_per_seed = int(payload.get("optimizer_trials_per_seed") or 0)
    optimizer_resolved_seeds = [int(value) for value in list(payload.get("optimizer_resolved_seeds") or [])]
    if "optimizer_seed_generator_seed" not in payload:
        raise ValueError("Trading param binding 缺少 optimizer_seed_generator_seed")
    if str(payload.get("optimizer_seed_mode")) == "random_n_each_training_run":
        if payload["optimizer_seed_generator_seed"] is not None:
            raise ValueError("Trading random-seed binding不應帶固定generator_seed")
    elif str(payload.get("optimizer_seed_mode")) == "fixed_generated_bundle":
        if payload["optimizer_seed_generator_seed"] is None:
            raise ValueError("Trading fixed-seed binding缺少generator_seed")
    else:
        raise ValueError("Trading param binding optimizer_seed_mode不合法")
    if optimizer_seed_count < 2 or optimizer_trials_per_seed < 1:
        raise ValueError("Trading param binding optimizer contract 不合法")
    if len(optimizer_resolved_seeds) != optimizer_seed_count or len(set(optimizer_resolved_seeds)) != optimizer_seed_count:
        raise ValueError("Trading param binding optimizer resolved seeds 不完整或重複")
    training_date = str(payload.get("param_training_data_date") or "")
    latest_data_date = str(payload.get("latest_data_date") or "")
    if training_date > latest_data_date:
        raise ValueError("Trading param binding 的 Params 訓練資料日晚於使用資料日")
    if str(payload.get("usage_mode")) == TRADING_PARAM_USAGE_TRAINED_CURRENT and training_date != latest_data_date:
        raise ValueError("trained_current Trading param binding 必須由目前 finalized data 訓練")
    effective_date = str(payload.get("param_effective_date") or "")
    valid_through_date = str(payload.get("param_valid_through_date") or "")
    if schedule_mode == "rolling":
        if not effective_date or not valid_through_date:
            raise ValueError("Trading rolling param binding 缺少 effective/valid-through lineage")
        if training_date > effective_date:
            raise ValueError("Trading param binding 的 Params 訓練資料日晚於生效日")
        if effective_date > latest_data_date:
            raise ValueError("Trading param binding 的 Params 生效日晚於使用資料日")
        if latest_data_date > valid_through_date:
            raise ValueError("Trading param binding 已超過目前 Params PIT refit period")
    elif effective_date or valid_through_date:
        raise ValueError("Trading static param binding 不應帶 rolling effective/valid-through lineage")
    core = {key: value for key, value in payload.items() if key != "binding_fingerprint"}
    if str(payload.get("binding_fingerprint") or "") != canonical_json_sha256(core):
        raise ValueError("Trading param binding fingerprint 不一致")


def validate_trading_selected_strategy_param_contract(
    selected_payload: dict[str, Any],
    *,
    expected_profile,
) -> list[int]:
    """Canonical selected-Params compatibility check for read-model and binding.

    The Workbench must not advertise REUSE based only on file presence/date: it
    must apply exactly the same scientific/seed policy guard used at binding time.
    This is read-only and does not parse market history or mutate artifacts.
    """

    if not isinstance(selected_payload, dict):
        raise TypeError("Trading selected strategy params payload 必須是 object")
    if str(selected_payload.get("selector") or "").strip() != str(expected_profile.param_selector):
        raise RuntimeError("Trading selected strategy params selector 與目前 Trading 設定不一致")

    meta = dict(selected_payload.get("meta") or {})
    requested_seed_policy = dict(meta.get("requested_random_seed_ensemble") or {})
    walk_forward_policy = dict(meta.get("walk_forward_policy") or {})
    artifact_resolved_seeds = [int(value) for value in list(meta.get("seeds") or [])]
    requested_resolved_seeds = [
        int(value) for value in list(requested_seed_policy.get("resolved_seeds") or [])
    ]
    expected_resolved_seeds = [int(value) for value in expected_profile.optimizer_resolved_seeds]
    is_fixed_bundle = expected_profile.optimizer_seed_mode == "fixed_generated_bundle"

    contract_errors: list[str] = []
    if str(meta.get("selected_model_mode") or "") != "trade":
        contract_errors.append("selected_model_mode")
    if int(meta.get("trials_per_seed") or 0) != int(expected_profile.optimizer_trials_per_seed):
        contract_errors.append("trials_per_seed")
    if int(requested_seed_policy.get("seed_count") or 0) != int(expected_profile.optimizer_seed_count):
        contract_errors.append("seed_count")
    if int(requested_seed_policy.get("min_agree") or 0) != int(expected_profile.optimizer_seed_min_agree):
        contract_errors.append("min_agree")
    if str(requested_seed_policy.get("seed_mode") or "") != str(expected_profile.optimizer_seed_mode):
        contract_errors.append("seed_mode")
    requested_generator_seed = requested_seed_policy.get("generator_seed")
    if is_fixed_bundle:
        if requested_generator_seed is None or int(requested_generator_seed) != int(expected_profile.optimizer_seed_generator_seed):
            contract_errors.append("generator_seed")
        if requested_resolved_seeds != expected_resolved_seeds:
            contract_errors.append("requested_resolved_seeds")
    elif requested_generator_seed is not None:
        contract_errors.append("unexpected_fixed_generator_seed")
    if len(artifact_resolved_seeds) != int(expected_profile.optimizer_seed_count):
        contract_errors.append("artifact_seed_count")
    if len(set(artifact_resolved_seeds)) != len(artifact_resolved_seeds) or any(seed < 1 or seed >= SEED_ENSEMBLE_MAX_SEED_VALUE for seed in artifact_resolved_seeds):
        contract_errors.append("artifact_seeds_unique_and_valid")
    if requested_resolved_seeds != artifact_resolved_seeds:
        contract_errors.append("requested_vs_artifact_seeds")
    if is_fixed_bundle and artifact_resolved_seeds != expected_resolved_seeds:
        contract_errors.append("artifact_resolved_seeds")
    artifact_train_window = int(
        walk_forward_policy.get("trade_train_window_months")
        or walk_forward_policy.get("train_window_months")
        or 0
    )
    if artifact_train_window != int(expected_profile.optimizer_train_window_months):
        contract_errors.append("train_window_months")
    if contract_errors:
        raise TradingParamContractMismatchError(
            "Trading selected strategy params不符合目前promoted Strategy Parameter profile "
            f"{expected_profile.strategy_param_profile_id}/{expected_profile.parameter_set_identity} "
            f"({', '.join(contract_errors)})；請重新訓練 Params。"
        )
    return artifact_resolved_seeds


def _selected_param_metadata(
    selected_path: Path,
    *,
    expected_profile,
    information_date: str,
) -> dict[str, Any]:
    selected_payload = load_json_strict(selected_path)
    artifact_resolved_seeds = validate_trading_selected_strategy_param_contract(
        selected_payload, expected_profile=expected_profile
    )
    param_runtime = load_trading_strategy_param_runtime(
        selected_path, information_date=str(information_date)
    )
    schedule_mode = str(param_runtime.get("schedule_mode") or "").strip().lower()
    if schedule_mode not in TRADING_PARAM_SCHEDULE_MODES:
        raise ValueError("Trading selected strategy params schedule mode 不合法")
    training_date = str(param_runtime.get("training_data_date") or "").strip()
    if not training_date:
        raise ValueError("Trading strategy params 缺少訓練資料日 lineage")
    effective_date = ""
    valid_through_date = ""
    if schedule_mode == "rolling":
        effective_date = str(param_runtime.get("effective_date") or "").strip()
        valid_through_date = str(param_runtime.get("scheduled_effective_end_date") or "").strip()
        if not effective_date or not valid_through_date:
            raise ValueError("Trading rolling strategy params 缺少 PIT fold lineage")
    return {
        "schedule_mode": schedule_mode,
        "training_date": training_date,
        "effective_date": effective_date,
        "valid_through_date": valid_through_date,
        "member_count": int(param_runtime["member_count"]),
        "min_agree": int(param_runtime["min_agree"]),
        "optimizer_trials_per_seed": int(expected_profile.optimizer_trials_per_seed),
        "optimizer_seed_count": int(expected_profile.optimizer_seed_count),
        "optimizer_seed_mode": str(expected_profile.optimizer_seed_mode),
        "optimizer_seed_generator_seed": expected_profile.optimizer_seed_generator_seed,
        "optimizer_resolved_seeds": artifact_resolved_seeds,
    }


def publish_trading_strategy_param_binding(
    project_root: str | Path,
    *,
    usage_mode: str = TRADING_PARAM_USAGE_TRAINED_CURRENT,
    verify_consumer_view: bool = True,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    profile = get_trading_strategy_profile()
    normalized_usage_mode = str(usage_mode or "").strip()
    if normalized_usage_mode not in TRADING_PARAM_USAGE_MODES:
        raise ValueError(f"不支援的 Trading Params usage mode: {usage_mode!r}")
    selected_path = Path(resolve_trading_selected_strategy_param_path(root))
    if not selected_path.is_file():
        raise FileNotFoundError("Trading selected strategy params 尚未產生")
    consumer_state = load_trading_v2_consumer_state(
        root,
        required=True,
        verify_current_view=bool(verify_consumer_view),
    )
    current_data_date = str(consumer_state["market_date"])
    param_metadata = _selected_param_metadata(
        selected_path,
        expected_profile=profile,
        information_date=current_data_date,
    )
    param_training_data_date = str(param_metadata["training_date"])
    if param_training_data_date > current_data_date:
        raise RuntimeError(
            "Trading selected strategy params 訓練資料日晚於目前 Trading data；禁止使用未來 Params"
        )
    if normalized_usage_mode == TRADING_PARAM_USAGE_TRAINED_CURRENT and param_training_data_date != current_data_date:
        raise RuntimeError(
            "Trading 本次重新訓練 Params 並非使用目前 finalized data；禁止發布 trained_current binding"
        )
    payload: dict[str, Any] = {
        "schema_version": TRADING_STRATEGY_PARAM_BINDING_SCHEMA_VERSION,
        "runtime_domain": RUNTIME_DOMAIN_TRADING,
        "strategy_id": profile.strategy_id,
        "strategy_param_profile_id": profile.strategy_param_profile_id,
        "parameter_set_identity": profile.parameter_set_identity,
        "registered_scientific_fingerprint_sha256": profile.registered_scientific_fingerprint_sha256,
        "param_selector": profile.param_selector,
        "selected_params_path": project_relative_display_path(selected_path, project_root=root),
        "selected_params_sha256": compute_file_sha256(selected_path),
        "market_data_source": "trading_market_data_v2_historical_latest_view",
        "market_data_consumer_state_sha256": get_trading_v2_consumer_state_sha256(root),
        "market_data_source_view_fingerprint": str(consumer_state["source_view_fingerprint"]),
        "latest_data_date": current_data_date,
        "param_training_data_date": param_training_data_date,
        "param_schedule_mode": str(param_metadata["schedule_mode"]),
        "param_effective_date": str(param_metadata["effective_date"]) or None,
        "param_valid_through_date": str(param_metadata["valid_through_date"]) or None,
        "usage_mode": normalized_usage_mode,
        "param_member_count": int(param_metadata["member_count"]),
        "param_min_agree": int(param_metadata["min_agree"]),
        "optimizer_trials_per_seed": int(param_metadata["optimizer_trials_per_seed"]),
        "optimizer_seed_count": int(param_metadata["optimizer_seed_count"]),
        "optimizer_seed_mode": str(param_metadata["optimizer_seed_mode"]),
        "optimizer_seed_generator_seed": param_metadata["optimizer_seed_generator_seed"],
        "optimizer_resolved_seeds": list(param_metadata["optimizer_resolved_seeds"]),
    }
    payload["binding_fingerprint"] = canonical_json_sha256(payload)
    path = resolve_trading_strategy_param_binding_path(root)
    atomic_write_json(path, payload)
    return payload


def load_trading_strategy_param_binding(
    project_root: str | Path,
    *,
    required: bool = True,
    verify_current: bool = True,
    verify_dataset_content: bool = False,
) -> dict[str, Any] | None:
    root = Path(project_root).resolve()
    path = resolve_trading_strategy_param_binding_path(root)
    if not path.is_file():
        if required:
            raise FileNotFoundError("Trading param binding 尚未建立；請重新執行「2 套用 Trading Params」")
        return None
    payload = load_json_strict(path)
    _validate_binding(payload)
    if not verify_current:
        return payload

    profile = get_trading_strategy_profile()
    selected_path = Path(resolve_trading_selected_strategy_param_path(root))
    if str(payload.get("strategy_id") or "") != profile.strategy_id:
        raise RuntimeError("Trading param binding strategy 已過期")
    if str(payload.get("strategy_param_profile_id") or "") != profile.strategy_param_profile_id:
        raise RuntimeError("Trading param binding Strategy Parameter profile 已過期；請重新訓練 Params")
    if str(payload.get("parameter_set_identity") or "") != profile.parameter_set_identity:
        raise RuntimeError("Trading param binding Parameter Set identity 已過期；請重新訓練 Params")
    if str(payload.get("registered_scientific_fingerprint_sha256") or "") != profile.registered_scientific_fingerprint_sha256:
        raise RuntimeError("Trading param binding registered scientific fingerprint 已過期；請重新訓練 Params")
    if str(payload.get("param_selector") or "") != profile.param_selector:
        raise RuntimeError("Trading param binding selector 已過期")
    if not selected_path.is_file() or str(payload.get("selected_params_sha256") or "") != compute_file_sha256(selected_path):
        raise RuntimeError("Trading selected params 已與 param binding 不一致；請重新套用 Trading Params")
    consumer_state = load_trading_v2_consumer_state(
        root, required=True, verify_current_view=bool(verify_dataset_content)
    )
    if str(payload.get("market_data_consumer_state_sha256") or "") != get_trading_v2_consumer_state_sha256(root):
        raise RuntimeError("Trading V2 consumer state identity 與 params binding 不一致")
    if str(payload.get("market_data_source_view_fingerprint") or "") != str(consumer_state["source_view_fingerprint"]):
        raise RuntimeError("Trading V2 view identity 與 params binding 不一致")
    if str(payload.get("latest_data_date") or "") != str(consumer_state.get("market_date") or ""):
        raise RuntimeError("Trading params binding data date 已過期")
    param_metadata = _selected_param_metadata(
        selected_path,
        expected_profile=profile,
        information_date=str(consumer_state.get("market_date") or ""),
    )
    if str(payload.get("param_training_data_date") or "") != str(param_metadata["training_date"]):
        raise RuntimeError("Trading Params artifact 訓練資料日已與 param binding 不一致")
    if str(payload.get("param_schedule_mode") or "") != str(param_metadata["schedule_mode"]):
        raise RuntimeError("Trading Params artifact schedule mode 已與 param binding 不一致")
    if str(param_metadata["schedule_mode"]) == "rolling":
        if str(payload.get("param_effective_date") or "") != str(param_metadata["effective_date"]):
            raise RuntimeError("Trading Params artifact effective date 已與 param binding 不一致")
        if str(payload.get("param_valid_through_date") or "") != str(param_metadata["valid_through_date"]):
            raise RuntimeError("Trading Params artifact PIT refit period 已與 param binding 不一致")
    elif payload.get("param_effective_date") not in (None, "") or payload.get("param_valid_through_date") not in (None, ""):
        raise RuntimeError("Trading static Params binding 不應帶 rolling effective/valid-through lineage")
    if int(payload.get("param_member_count") or 0) != int(param_metadata["member_count"]):
        raise RuntimeError("Trading Params artifact member_count 已與 param binding 不一致")
    if int(payload.get("param_min_agree") or 0) != int(param_metadata["min_agree"]):
        raise RuntimeError("Trading Params artifact min_agree 已與 param binding 不一致")
    for field in ("optimizer_trials_per_seed", "optimizer_seed_count"):
        if int(payload.get(field) or 0) != int(param_metadata[field]):
            raise RuntimeError(f"Trading Params artifact {field} 已與 param binding 不一致")
    if payload.get("optimizer_seed_generator_seed") != param_metadata["optimizer_seed_generator_seed"]:
        raise RuntimeError("Trading Params artifact seed generator 已與 param binding 不一致")
    if str(payload.get("optimizer_seed_mode") or "") != str(param_metadata["optimizer_seed_mode"]):
        raise RuntimeError("Trading Params artifact optimizer_seed_mode 已與 param binding 不一致")
    if [int(value) for value in list(payload.get("optimizer_resolved_seeds") or [])] != list(param_metadata["optimizer_resolved_seeds"]):
        raise RuntimeError("Trading Params artifact optimizer_resolved_seeds 已與 param binding 不一致")
    return payload


def get_trading_strategy_param_binding_sha256(project_root: str | Path) -> str:
    path = resolve_trading_strategy_param_binding_path(project_root)
    if not path.is_file():
        raise FileNotFoundError("Trading param binding 尚未建立")
    return compute_file_sha256(path)


__all__ = [
    "TRADING_STRATEGY_PARAM_BINDING_SCHEMA_VERSION",
    "TRADING_STRATEGY_PARAM_BINDING_FILENAME",
    "TRADING_PARAM_USAGE_TRAINED_CURRENT",
    "TRADING_PARAM_USAGE_REUSE_EXISTING",
    "TRADING_PARAM_USAGE_MODES",
    "TRADING_PARAM_SCHEDULE_MODES",
    "resolve_trading_strategy_param_binding_path",
    "TradingParamContractMismatchError",
    "validate_trading_selected_strategy_param_contract",
    "publish_trading_strategy_param_binding",
    "load_trading_strategy_param_binding",
    "get_trading_strategy_param_binding_sha256",
]
