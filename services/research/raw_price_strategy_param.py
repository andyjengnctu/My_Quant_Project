"""Profile-driven Strategy Parameter producer for raw/unadjusted Research prices.

Raw price is a data-source capability, not a C-arm-specific execution path.  A
registered Strategy Parameter profile supplies selector/fitting semantics while
this producer only isolates the controlled raw OHLC dataset and artifact roots.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

from core.active_param_ensemble import get_active_param_ensemble_date_range, is_active_param_ensemble_payload
from core.file_integrity import canonical_json_sha256, load_json_strict
from core.strategy_param_artifacts import (
    compute_strategy_param_file_sha256,
    normalize_strategy_param_policy,
    resolve_strategy_param_artifact_path,
    resolve_strategy_param_manifest_path,
)
from core.strategy_param_policy import (
    build_replicated_fitting_seed_ensemble_policy,
    get_strategy_param_fitting_seed_protocol,
    get_strategy_param_training_profile,
)
from services.optimizer.raw_cache import load_all_raw_data
from services.optimizer.strategy_param_service import build_isolated_strategy_parameter_schedule
from services.research.raw_price_strategy_dataset import (
    ensure_raw_price_strategy_dataset,
    inspect_raw_price_strategy_dataset,
)

DEFAULT_RAW_PRICE_TRAINING_PROFILE_ID = "raw_price_full_base_finalist_best_current"
RAW_PRICE_RUNTIME_DOMAIN = "research_raw_price"
RAW_PRICE_POLICY = "base-finalist-best"  # compatibility alias for PARAM-RAW-P1
RAW_PRICE_FAMILY = "full"
RAW_PRICE_PARAMSET_ROOT_RELATIVE = (
    Path("models") / "research" / "breakout_quality" / "strategy_params" / "param_raw_p1_full_base_finalist_best"
)
RAW_PRICE_WORK_ROOT_RELATIVE = (
    Path("outputs") / "optimizer" / "strategy_param_schedule" / "experiments" / "param_raw_p1_full_base_finalist_best"
)
_FIXED_MULTI_SEED_SEARCH_PROTOCOL = "fixed_multiseed_v1"


def _raw_profile(training_profile_id: str):
    profile = get_strategy_param_training_profile(str(training_profile_id))
    if profile.price_basis != "raw_unadjusted":
        raise ValueError(
            "raw-price producer只接受price_basis=raw_unadjusted profile: "
            f"{profile.profile_id} -> {profile.price_basis}"
        )
    if profile.filter_profile != RAW_PRICE_FAMILY:
        raise ValueError(
            "raw-price producer目前只支援Full strategy parameter training: "
            f"{profile.profile_id} -> {profile.filter_profile}"
        )
    if profile.validation_enabled:
        raise ValueError(
            "raw-price producer目前不接受validation-enabled profile: "
            f"{profile.profile_id}"
        )
    if not profile.artifact_slug:
        raise ValueError(f"raw-price profile缺少artifact_slug: {profile.profile_id}")
    normalize_strategy_param_policy(profile.parameter_policy)
    return profile


def _param_root(root: Path, training_profile_id: str = DEFAULT_RAW_PRICE_TRAINING_PROFILE_ID) -> Path:
    profile = _raw_profile(training_profile_id)
    return (
        root
        / "models"
        / "research"
        / "breakout_quality"
        / "strategy_params"
        / str(profile.artifact_slug)
    )


def _work_root(root: Path, training_profile_id: str = DEFAULT_RAW_PRICE_TRAINING_PROFILE_ID) -> Path:
    profile = _raw_profile(training_profile_id)
    return (
        root
        / "outputs"
        / "optimizer"
        / "strategy_param_schedule"
        / "experiments"
        / str(profile.artifact_slug)
    )


def _runtime_domain(training_profile_id: str) -> str:
    profile = _raw_profile(training_profile_id)
    if profile.profile_id == DEFAULT_RAW_PRICE_TRAINING_PROFILE_ID:
        return RAW_PRICE_RUNTIME_DOMAIN
    return f"{RAW_PRICE_RUNTIME_DOMAIN}:{profile.artifact_slug}"


def _target(root: Path, training_profile_id: str = DEFAULT_RAW_PRICE_TRAINING_PROFILE_ID) -> Path:
    profile = _raw_profile(training_profile_id)
    return resolve_strategy_param_artifact_path(
        root,
        family=RAW_PRICE_FAMILY,
        evaluation_mode="rolling",
        policy=profile.parameter_policy,
        strategy_params_root=_param_root(root, profile.profile_id),
    )


def _manifest(root: Path, training_profile_id: str = DEFAULT_RAW_PRICE_TRAINING_PROFILE_ID) -> Path:
    profile = _raw_profile(training_profile_id)
    return resolve_strategy_param_manifest_path(
        root,
        family=RAW_PRICE_FAMILY,
        evaluation_mode="rolling",
        strategy_params_root=_param_root(root, profile.profile_id),
    )


def _coverage_end(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        payload = load_json_strict(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict) or not is_active_param_ensemble_payload(payload):
        return None
    try:
        _start, end = get_active_param_ensemble_date_range(payload)
    except (KeyError, TypeError, ValueError):
        return None
    return None if not end else str(end)


def _load_target_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = load_json_strict(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _profile_seed_contract(training_profile_id: str) -> tuple[Any, dict[str, Any] | None]:
    profile = _raw_profile(training_profile_id)
    protocol = get_strategy_param_fitting_seed_protocol(profile.fitting_seed_protocol)
    seed_policy = dict(protocol.as_seed_ensemble_policy()) if protocol.enabled else None
    return protocol, seed_policy


def _multiseed_identity_ready(
    *, profile, protocol, seed_policy: dict[str, Any] | None,
    source: dict[str, Any], target_payload: dict[str, Any] | None,
) -> bool:
    if not protocol.enabled:
        return True
    if seed_policy is None or target_payload is None:
        return False
    build_contract = dict(source.get("build_contract") or {})
    adaptation = dict(target_payload.get("breakout_quality_param_adaptation") or {})
    return bool(
        int(build_contract.get("optimizer_seed") or 0) == int(protocol.generator_seed)
        and int(build_contract.get("trials_per_fold") or 0) == int(protocol.trials_per_seed)
        and dict(build_contract.get("seed_ensemble_policy") or {}) == seed_policy
        and str(build_contract.get("parameter_set") or "") == str(profile.parameter_set_identity)
        and str(build_contract.get("optimizer_search_protocol") or "")
            == _FIXED_MULTI_SEED_SEARCH_PROTOCOL
        and str(adaptation.get("parameter_set") or "") == str(profile.parameter_set_identity)
        and str(adaptation.get("optimizer_search_protocol") or "")
            == _FIXED_MULTI_SEED_SEARCH_PROTOCOL
        and dict(adaptation.get("seed_ensemble_policy") or {}) == seed_policy
    )


def inspect_raw_price_strategy_parameter_artifact(
    project_root: str | Path,
    *,
    comparison_end_date: str,
    training_profile_id: str = DEFAULT_RAW_PRICE_TRAINING_PROFILE_ID,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    profile = _raw_profile(training_profile_id)
    protocol, seed_policy = _profile_seed_contract(profile.profile_id)
    dataset_state = inspect_raw_price_strategy_dataset(root, deep=False)
    target = _target(root, profile.profile_id)
    manifest_path = _manifest(root, profile.profile_id)
    coverage = _coverage_end(target)
    if not bool(dataset_state.get("ready")):
        return {
            "ready": False,
            "action": "BUILD",
            "status": str(dataset_state.get("status") or "RAW_PRICE_DATASET_NOT_READY"),
            "path": target,
            "manifest_path": manifest_path,
            "coverage_end": coverage,
            "training_profile_id": profile.profile_id,
        }

    data_identity = dict(dataset_state.get("data_identity") or {})
    data_identity_sha = canonical_json_sha256(data_identity)
    try:
        manifest = load_json_strict(manifest_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        manifest = None
    record = {}
    policy_key = normalize_strategy_param_policy(profile.parameter_policy)
    if isinstance(manifest, dict):
        record = dict(dict(manifest.get("artifacts") or {}).get(policy_key) or {})
    source = dict(record.get("source") or {})
    target_payload = _load_target_payload(target)
    identity_ready = bool(
        target.is_file()
        and record
        and str(record.get("sha256") or "") == compute_strategy_param_file_sha256(target)
        and str(source.get("runtime_domain") or "") == _runtime_domain(profile.profile_id)
        and str(source.get("data_identity_sha256") or "") == data_identity_sha
        and _multiseed_identity_ready(
            profile=profile,
            protocol=protocol,
            seed_policy=seed_policy,
            source=source,
            target_payload=target_payload,
        )
    )
    coverage_ready = bool(coverage is not None and str(coverage) >= str(comparison_end_date))
    ready = bool(identity_ready and coverage_ready)
    if ready:
        status = "READY"
    elif not target.is_file():
        status = "ARTIFACT_MISSING"
    elif not identity_ready:
        status = "RAW_PRICE_SCIENTIFIC_IDENTITY_MISMATCH"
    else:
        status = "SCHEDULE_COVERAGE_INCOMPLETE"
    return {
        "ready": ready,
        "action": "REUSE" if ready else ("REBUILD" if target.is_file() else "BUILD"),
        "status": status,
        "path": target,
        "manifest_path": manifest_path,
        "coverage_end": coverage,
        "data_identity": data_identity,
        "training_profile_id": profile.profile_id,
        "parameter_set_identity": profile.parameter_set_identity,
        "parameter_policy": profile.parameter_policy,
        "fitting_seed_count": int(protocol.seed_count),
        "fitting_trials_per_seed": int(protocol.trials_per_seed),
    }


def ensure_raw_price_strategy_parameter_artifact(
    project_root: str | Path,
    *,
    comparison_end_date: str,
    training_profile_id: str = DEFAULT_RAW_PRICE_TRAINING_PROFILE_ID,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    profile = _raw_profile(training_profile_id)
    protocol, seed_policy = _profile_seed_contract(profile.profile_id)
    prior = inspect_raw_price_strategy_parameter_artifact(
        root,
        comparison_end_date=str(comparison_end_date),
        training_profile_id=profile.profile_id,
    )
    if bool(prior.get("ready")) and not bool(force_rebuild):
        return prior

    dataset = ensure_raw_price_strategy_dataset(root)
    data_identity = dict(dataset["data_identity"])
    selected_work_root = _work_root(root, profile.profile_id)
    if bool(force_rebuild):
        selected_work_root = selected_work_root / "force_rebuild"
        shutil.rmtree(selected_work_root, ignore_errors=True)
    result = build_isolated_strategy_parameter_schedule(
        root,
        family=RAW_PRICE_FAMILY,
        policy=profile.parameter_policy,
        comparison_end_date=str(comparison_end_date),
        dataset=profile.dataset,
        selected_data_dir=Path(dataset["dataset_dir"]),
        raw_data_loader=load_all_raw_data,
        strategy_params_root=_param_root(root, profile.profile_id),
        models_dir=root / "models" / "research" / "breakout_quality",
        work_root=selected_work_root,
        runtime_domain=_runtime_domain(profile.profile_id),
        data_identity=data_identity,
        optimizer_seed_override=int(protocol.generator_seed),
        trials_per_fold_override=int(protocol.trials_per_seed),
        seed_ensemble_policy_override=seed_policy,
        optimizer_search_protocol=(
            _FIXED_MULTI_SEED_SEARCH_PROTOCOL if protocol.enabled else None
        ),
        parameter_set_identity=(
            str(profile.parameter_set_identity) if protocol.enabled else None
        ),
        score_policy_overrides=dict(profile.score_policy_overrides or {}),
        resume_parameter_training=not bool(force_rebuild),
    )
    final = inspect_raw_price_strategy_parameter_artifact(
        root,
        comparison_end_date=str(comparison_end_date),
        training_profile_id=profile.profile_id,
    )
    if not bool(final.get("ready")):
        raise RuntimeError(
            "raw-price Strategy Parameter producer完成後仍未READY: "
            f"profile={profile.profile_id}, status={final.get('status')}, "
            f"coverage={final.get('coverage_end')}, required={comparison_end_date}"
        )
    return {
        **final,
        "build_result": result,
        "dataset_manifest_path": dataset.get("manifest_path"),
        "dataset_dir": dataset.get("dataset_dir"),
    }


def _robustness_param_root(
    root: Path,
    *,
    benchmark_id: str,
    seed: int,
    training_profile_id: str = DEFAULT_RAW_PRICE_TRAINING_PROFILE_ID,
) -> Path:
    profile = _raw_profile(training_profile_id)
    return (
        root
        / "models"
        / "research"
        / "breakout_quality"
        / "strategy_params"
        / "robustness"
        / str(benchmark_id)
        / f"seed_{int(seed)}"
        / str(profile.artifact_slug)
    )


def _robustness_work_root(
    root: Path,
    *,
    benchmark_id: str,
    seed: int,
    training_profile_id: str = DEFAULT_RAW_PRICE_TRAINING_PROFILE_ID,
) -> Path:
    profile = _raw_profile(training_profile_id)
    return (
        root
        / "outputs"
        / "optimizer"
        / "strategy_param_schedule"
        / "robustness"
        / str(benchmark_id)
        / f"seed_{int(seed)}"
        / str(profile.artifact_slug)
    )


def raw_price_robustness_strategy_param_artifact_path(
    project_root: str | Path,
    *,
    benchmark_id: str,
    seed: int,
    training_profile_id: str = DEFAULT_RAW_PRICE_TRAINING_PROFILE_ID,
) -> Path:
    root = Path(project_root).resolve()
    profile = _raw_profile(training_profile_id)
    return resolve_strategy_param_artifact_path(
        root,
        family=RAW_PRICE_FAMILY,
        evaluation_mode="rolling",
        policy=profile.parameter_policy,
        strategy_params_root=_robustness_param_root(
            root,
            benchmark_id=str(benchmark_id),
            seed=int(seed),
            training_profile_id=profile.profile_id,
        ),
    )


def raw_price_robustness_strategy_param_manifest_path(
    project_root: str | Path,
    *,
    benchmark_id: str,
    seed: int,
    training_profile_id: str = DEFAULT_RAW_PRICE_TRAINING_PROFILE_ID,
) -> Path:
    root = Path(project_root).resolve()
    profile = _raw_profile(training_profile_id)
    return resolve_strategy_param_manifest_path(
        root,
        family=RAW_PRICE_FAMILY,
        evaluation_mode="rolling",
        strategy_params_root=_robustness_param_root(
            root,
            benchmark_id=str(benchmark_id),
            seed=int(seed),
            training_profile_id=profile.profile_id,
        ),
    )


def _raw_robustness_runtime_domain(
    *, training_profile_id: str, benchmark_id: str, seed: int
) -> str:
    return (
        f"{_runtime_domain(training_profile_id)}_robustness:"
        f"{str(benchmark_id)}:seed={int(seed)}"
    )


def inspect_raw_price_robustness_strategy_parameter_artifact(
    project_root: str | Path,
    *,
    benchmark_id: str,
    seed: int,
    comparison_end_date: str,
    training_profile_id: str = DEFAULT_RAW_PRICE_TRAINING_PROFILE_ID,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    profile = _raw_profile(training_profile_id)
    protocol = get_strategy_param_fitting_seed_protocol(profile.fitting_seed_protocol)
    replicated_seed_policy = (
        build_replicated_fitting_seed_ensemble_policy(
            profile.fitting_seed_protocol, replication_seed=int(seed)
        )
        if protocol.enabled
        else None
    )
    dataset_state = inspect_raw_price_strategy_dataset(root, deep=False)
    target = raw_price_robustness_strategy_param_artifact_path(
        root,
        benchmark_id=str(benchmark_id),
        seed=int(seed),
        training_profile_id=profile.profile_id,
    )
    manifest_path = raw_price_robustness_strategy_param_manifest_path(
        root,
        benchmark_id=str(benchmark_id),
        seed=int(seed),
        training_profile_id=profile.profile_id,
    )
    coverage = _coverage_end(target)
    if not bool(dataset_state.get("ready")):
        return {
            "ready": False,
            "action": "BUILD",
            "status": str(dataset_state.get("status") or "RAW_PRICE_DATASET_NOT_READY"),
            "path": target,
            "manifest_path": manifest_path,
            "coverage_end": coverage,
            "training_profile_id": profile.profile_id,
        }

    data_identity = dict(dataset_state.get("data_identity") or {})
    data_identity_sha = canonical_json_sha256(data_identity)
    try:
        manifest = load_json_strict(manifest_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        manifest = None
    record = {}
    policy_key = normalize_strategy_param_policy(profile.parameter_policy)
    if isinstance(manifest, dict):
        record = dict(dict(manifest.get("artifacts") or {}).get(policy_key) or {})
    source = dict(record.get("source") or {})
    build_contract = dict(source.get("build_contract") or {})
    target_payload = _load_target_payload(target)
    adaptation = dict((target_payload or {}).get("breakout_quality_param_adaptation") or {})
    expected_domain = _raw_robustness_runtime_domain(
        training_profile_id=profile.profile_id,
        benchmark_id=str(benchmark_id),
        seed=int(seed),
    )
    multiseed_ready = True
    if protocol.enabled:
        multiseed_ready = bool(
            replicated_seed_policy is not None
            and dict(build_contract.get("seed_ensemble_policy") or {})
                == dict(replicated_seed_policy)
            and int(build_contract.get("trials_per_fold") or 0)
                == int(protocol.trials_per_seed)
            and str(build_contract.get("parameter_set") or "")
                == str(profile.parameter_set_identity)
            and str(build_contract.get("optimizer_search_protocol") or "")
                == _FIXED_MULTI_SEED_SEARCH_PROTOCOL
            and str(adaptation.get("parameter_set") or "")
                == str(profile.parameter_set_identity)
            and str(adaptation.get("optimizer_search_protocol") or "")
                == _FIXED_MULTI_SEED_SEARCH_PROTOCOL
            and dict(adaptation.get("seed_ensemble_policy") or {})
                == dict(replicated_seed_policy)
        )
    identity_ready = bool(
        target.is_file()
        and record
        and str(record.get("sha256") or "") == compute_strategy_param_file_sha256(target)
        and str(source.get("runtime_domain") or "") == expected_domain
        and str(source.get("data_identity_sha256") or "") == data_identity_sha
        and int(build_contract.get("optimizer_seed") or 0) == int(seed)
        and multiseed_ready
    )
    coverage_ready = bool(coverage is not None and str(coverage) >= str(comparison_end_date))
    ready = bool(identity_ready and coverage_ready)
    if ready:
        status = "READY"
    elif not target.is_file():
        status = "ARTIFACT_MISSING"
    elif not identity_ready:
        status = "RAW_PRICE_ROBUSTNESS_SCIENTIFIC_IDENTITY_MISMATCH"
    else:
        status = "SCHEDULE_COVERAGE_INCOMPLETE"
    return {
        "ready": ready,
        "action": "REUSE" if ready else ("REBUILD" if target.is_file() else "BUILD"),
        "status": status,
        "path": target,
        "manifest_path": manifest_path,
        "coverage_end": coverage,
        "data_identity": data_identity,
        "benchmark_id": str(benchmark_id),
        "seed": int(seed),
        "training_profile_id": profile.profile_id,
        "parameter_set_identity": profile.parameter_set_identity,
        "parameter_policy": profile.parameter_policy,
        "fitting_seed_count": int(protocol.seed_count),
        "fitting_trials_per_seed": int(protocol.trials_per_seed),
    }


def ensure_raw_price_robustness_strategy_parameter_artifact(
    project_root: str | Path,
    *,
    benchmark_id: str,
    seed: int,
    comparison_end_date: str,
    training_profile_id: str = DEFAULT_RAW_PRICE_TRAINING_PROFILE_ID,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    profile = _raw_profile(training_profile_id)
    protocol = get_strategy_param_fitting_seed_protocol(profile.fitting_seed_protocol)
    replicated_seed_policy = (
        build_replicated_fitting_seed_ensemble_policy(
            profile.fitting_seed_protocol, replication_seed=int(seed)
        )
        if protocol.enabled
        else None
    )
    prior = inspect_raw_price_robustness_strategy_parameter_artifact(
        root,
        benchmark_id=str(benchmark_id),
        seed=int(seed),
        comparison_end_date=str(comparison_end_date),
        training_profile_id=profile.profile_id,
    )
    if bool(prior.get("ready")) and not bool(force_rebuild):
        return prior

    dataset = ensure_raw_price_strategy_dataset(root)
    data_identity = dict(dataset["data_identity"])
    selected_work_root = _robustness_work_root(
        root,
        benchmark_id=str(benchmark_id),
        seed=int(seed),
        training_profile_id=profile.profile_id,
    )
    if bool(force_rebuild):
        selected_work_root = selected_work_root / "force_rebuild"
        shutil.rmtree(selected_work_root, ignore_errors=True)
    runtime_domain = _raw_robustness_runtime_domain(
        training_profile_id=profile.profile_id,
        benchmark_id=str(benchmark_id),
        seed=int(seed),
    )
    result = build_isolated_strategy_parameter_schedule(
        root,
        family=RAW_PRICE_FAMILY,
        policy=profile.parameter_policy,
        comparison_end_date=str(comparison_end_date),
        dataset=profile.dataset,
        selected_data_dir=Path(dataset["dataset_dir"]),
        raw_data_loader=load_all_raw_data,
        strategy_params_root=_robustness_param_root(
            root,
            benchmark_id=str(benchmark_id),
            seed=int(seed),
            training_profile_id=profile.profile_id,
        ),
        models_dir=root / "models" / "research" / "breakout_quality",
        work_root=selected_work_root,
        runtime_domain=runtime_domain,
        data_identity=data_identity,
        optimizer_seed_override=int(seed),
        trials_per_fold_override=int(protocol.trials_per_seed),
        seed_ensemble_policy_override=replicated_seed_policy,
        optimizer_search_protocol=(
            _FIXED_MULTI_SEED_SEARCH_PROTOCOL if protocol.enabled else None
        ),
        parameter_set_identity=(
            str(profile.parameter_set_identity) if protocol.enabled else None
        ),
        score_policy_overrides=dict(profile.score_policy_overrides or {}),
        resume_parameter_training=not bool(force_rebuild),
    )
    final = inspect_raw_price_robustness_strategy_parameter_artifact(
        root,
        benchmark_id=str(benchmark_id),
        seed=int(seed),
        comparison_end_date=str(comparison_end_date),
        training_profile_id=profile.profile_id,
    )
    if not bool(final.get("ready")):
        raise RuntimeError(
            "raw-price robustness Strategy Parameter producer完成後仍未READY: "
            f"profile={profile.profile_id}, benchmark={benchmark_id}, seed={seed}, "
            f"status={final.get('status')}, coverage={final.get('coverage_end')}, "
            f"required={comparison_end_date}"
        )
    return {
        **final,
        "build_result": result,
        "dataset_manifest_path": dataset.get("manifest_path"),
        "dataset_dir": dataset.get("dataset_dir"),
    }


__all__ = [
    "DEFAULT_RAW_PRICE_TRAINING_PROFILE_ID",
    "RAW_PRICE_RUNTIME_DOMAIN",
    "ensure_raw_price_strategy_parameter_artifact",
    "inspect_raw_price_strategy_parameter_artifact",
    "ensure_raw_price_robustness_strategy_parameter_artifact",
    "inspect_raw_price_robustness_strategy_parameter_artifact",
    "raw_price_robustness_strategy_param_artifact_path",
    "raw_price_robustness_strategy_param_manifest_path",
]
