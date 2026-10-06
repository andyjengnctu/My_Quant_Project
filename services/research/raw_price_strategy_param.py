"""Isolated Strategy Parameter producer for raw/unadjusted Research prices."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

from core.active_param_ensemble import get_active_param_ensemble_date_range, is_active_param_ensemble_payload
from core.file_integrity import canonical_json_sha256, load_json_strict
from core.strategy_param_artifacts import (
    compute_strategy_param_file_sha256,
    resolve_strategy_param_artifact_path,
    resolve_strategy_param_manifest_path,
)
from services.optimizer.raw_cache import load_all_raw_data
from services.optimizer.strategy_param_service import build_isolated_strategy_parameter_schedule
from services.research.raw_price_strategy_dataset import (
    ensure_raw_price_strategy_dataset,
    inspect_raw_price_strategy_dataset,
)

RAW_PRICE_PARAMSET_ROOT_RELATIVE = (
    Path("models") / "research" / "breakout_quality" / "strategy_params" / "param_raw_p1_full_base_finalist_best"
)
RAW_PRICE_WORK_ROOT_RELATIVE = (
    Path("outputs") / "optimizer" / "strategy_param_schedule" / "experiments" / "param_raw_p1_full_base_finalist_best"
)
RAW_PRICE_RUNTIME_DOMAIN = "research_raw_price"
RAW_PRICE_POLICY = "base-finalist-best"
RAW_PRICE_FAMILY = "full"


def _param_root(root: Path) -> Path:
    return root / RAW_PRICE_PARAMSET_ROOT_RELATIVE


def _work_root(root: Path) -> Path:
    return root / RAW_PRICE_WORK_ROOT_RELATIVE


def _target(root: Path) -> Path:
    return resolve_strategy_param_artifact_path(
        root,
        family=RAW_PRICE_FAMILY,
        evaluation_mode="rolling",
        policy=RAW_PRICE_POLICY,
        strategy_params_root=_param_root(root),
    )


def _manifest(root: Path) -> Path:
    return resolve_strategy_param_manifest_path(
        root,
        family=RAW_PRICE_FAMILY,
        evaluation_mode="rolling",
        strategy_params_root=_param_root(root),
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


def inspect_raw_price_strategy_parameter_artifact(
    project_root: str | Path,
    *,
    comparison_end_date: str,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    dataset_state = inspect_raw_price_strategy_dataset(root, deep=False)
    target = _target(root)
    manifest_path = _manifest(root)
    coverage = _coverage_end(target)
    if not bool(dataset_state.get("ready")):
        return {
            "ready": False,
            "action": "BUILD",
            "status": str(dataset_state.get("status") or "RAW_PRICE_DATASET_NOT_READY"),
            "path": target,
            "coverage_end": coverage,
        }

    data_identity = dict(dataset_state.get("data_identity") or {})
    data_identity_sha = canonical_json_sha256(data_identity)
    try:
        manifest = load_json_strict(manifest_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        manifest = None
    record = {}
    if isinstance(manifest, dict):
        record = dict(dict(manifest.get("artifacts") or {}).get("base_finalist_best") or {})
    source = dict(record.get("source") or {})
    identity_ready = bool(
        target.is_file()
        and record
        and str(record.get("sha256") or "") == compute_strategy_param_file_sha256(target)
        and str(source.get("runtime_domain") or "") == RAW_PRICE_RUNTIME_DOMAIN
        and str(source.get("data_identity_sha256") or "") == data_identity_sha
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
    }


def ensure_raw_price_strategy_parameter_artifact(
    project_root: str | Path,
    *,
    comparison_end_date: str,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    prior = inspect_raw_price_strategy_parameter_artifact(
        root, comparison_end_date=str(comparison_end_date)
    )
    if bool(prior.get("ready")) and not bool(force_rebuild):
        return prior

    dataset = ensure_raw_price_strategy_dataset(root)
    data_identity = dict(dataset["data_identity"])
    selected_work_root = _work_root(root)
    if bool(force_rebuild):
        selected_work_root = selected_work_root / "force_rebuild"
        shutil.rmtree(selected_work_root, ignore_errors=True)
    result = build_isolated_strategy_parameter_schedule(
        root,
        family=RAW_PRICE_FAMILY,
        policy=RAW_PRICE_POLICY,
        comparison_end_date=str(comparison_end_date),
        dataset="full",
        selected_data_dir=Path(dataset["dataset_dir"]),
        raw_data_loader=load_all_raw_data,
        strategy_params_root=_param_root(root),
        models_dir=root / "models" / "research" / "breakout_quality",
        work_root=selected_work_root,
        runtime_domain=RAW_PRICE_RUNTIME_DOMAIN,
        data_identity=data_identity,
    )
    final = inspect_raw_price_strategy_parameter_artifact(
        root, comparison_end_date=str(comparison_end_date)
    )
    if not bool(final.get("ready")):
        raise RuntimeError(
            "raw-price Strategy Parameter producer完成後仍未READY: "
            f"status={final.get('status')}, coverage={final.get('coverage_end')}, "
            f"required={comparison_end_date}"
        )
    return {
        **final,
        "build_result": result,
        "dataset_manifest_path": dataset.get("manifest_path"),
        "dataset_dir": dataset.get("dataset_dir"),
    }


def _robustness_param_root(root: Path, *, benchmark_id: str, seed: int) -> Path:
    return (
        root
        / "models"
        / "research"
        / "breakout_quality"
        / "strategy_params"
        / "robustness"
        / str(benchmark_id)
        / f"seed_{int(seed)}"
        / "param_raw_p1_full_base_finalist_best"
    )


def _robustness_work_root(root: Path, *, benchmark_id: str, seed: int) -> Path:
    return (
        root
        / "outputs"
        / "optimizer"
        / "strategy_param_schedule"
        / "robustness"
        / str(benchmark_id)
        / f"seed_{int(seed)}"
        / "param_raw_p1_full_base_finalist_best"
    )


def raw_price_robustness_strategy_param_artifact_path(
    project_root: str | Path, *, benchmark_id: str, seed: int
) -> Path:
    root = Path(project_root).resolve()
    return resolve_strategy_param_artifact_path(
        root,
        family=RAW_PRICE_FAMILY,
        evaluation_mode="rolling",
        policy=RAW_PRICE_POLICY,
        strategy_params_root=_robustness_param_root(
            root, benchmark_id=str(benchmark_id), seed=int(seed)
        ),
    )


def raw_price_robustness_strategy_param_manifest_path(
    project_root: str | Path, *, benchmark_id: str, seed: int
) -> Path:
    root = Path(project_root).resolve()
    return resolve_strategy_param_manifest_path(
        root,
        family=RAW_PRICE_FAMILY,
        evaluation_mode="rolling",
        strategy_params_root=_robustness_param_root(
            root, benchmark_id=str(benchmark_id), seed=int(seed)
        ),
    )


def _raw_robustness_runtime_domain(*, benchmark_id: str, seed: int) -> str:
    return f"{RAW_PRICE_RUNTIME_DOMAIN}_robustness:{str(benchmark_id)}:seed={int(seed)}"


def inspect_raw_price_robustness_strategy_parameter_artifact(
    project_root: str | Path,
    *,
    benchmark_id: str,
    seed: int,
    comparison_end_date: str,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    dataset_state = inspect_raw_price_strategy_dataset(root, deep=False)
    target = raw_price_robustness_strategy_param_artifact_path(
        root, benchmark_id=str(benchmark_id), seed=int(seed)
    )
    manifest_path = raw_price_robustness_strategy_param_manifest_path(
        root, benchmark_id=str(benchmark_id), seed=int(seed)
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
        }

    data_identity = dict(dataset_state.get("data_identity") or {})
    data_identity_sha = canonical_json_sha256(data_identity)
    try:
        manifest = load_json_strict(manifest_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        manifest = None
    record = {}
    if isinstance(manifest, dict):
        record = dict(dict(manifest.get("artifacts") or {}).get("base_finalist_best") or {})
    source = dict(record.get("source") or {})
    build_contract = dict(source.get("build_contract") or {})
    expected_domain = _raw_robustness_runtime_domain(
        benchmark_id=str(benchmark_id), seed=int(seed)
    )
    identity_ready = bool(
        target.is_file()
        and record
        and str(record.get("sha256") or "") == compute_strategy_param_file_sha256(target)
        and str(source.get("runtime_domain") or "") == expected_domain
        and str(source.get("data_identity_sha256") or "") == data_identity_sha
        and int(build_contract.get("optimizer_seed") or 0) == int(seed)
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
    }


def ensure_raw_price_robustness_strategy_parameter_artifact(
    project_root: str | Path,
    *,
    benchmark_id: str,
    seed: int,
    comparison_end_date: str,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    prior = inspect_raw_price_robustness_strategy_parameter_artifact(
        root,
        benchmark_id=str(benchmark_id),
        seed=int(seed),
        comparison_end_date=str(comparison_end_date),
    )
    if bool(prior.get("ready")) and not bool(force_rebuild):
        return prior

    dataset = ensure_raw_price_strategy_dataset(root)
    data_identity = dict(dataset["data_identity"])
    selected_work_root = _robustness_work_root(
        root, benchmark_id=str(benchmark_id), seed=int(seed)
    )
    if bool(force_rebuild):
        selected_work_root = selected_work_root / "force_rebuild"
        shutil.rmtree(selected_work_root, ignore_errors=True)
    runtime_domain = _raw_robustness_runtime_domain(
        benchmark_id=str(benchmark_id), seed=int(seed)
    )
    result = build_isolated_strategy_parameter_schedule(
        root,
        family=RAW_PRICE_FAMILY,
        policy=RAW_PRICE_POLICY,
        comparison_end_date=str(comparison_end_date),
        dataset="full",
        selected_data_dir=Path(dataset["dataset_dir"]),
        raw_data_loader=load_all_raw_data,
        strategy_params_root=_robustness_param_root(
            root, benchmark_id=str(benchmark_id), seed=int(seed)
        ),
        models_dir=root / "models" / "research" / "breakout_quality",
        work_root=selected_work_root,
        runtime_domain=runtime_domain,
        data_identity=data_identity,
        optimizer_seed_override=int(seed),
    )
    final = inspect_raw_price_robustness_strategy_parameter_artifact(
        root,
        benchmark_id=str(benchmark_id),
        seed=int(seed),
        comparison_end_date=str(comparison_end_date),
    )
    if not bool(final.get("ready")):
        raise RuntimeError(
            "raw-price robustness Strategy Parameter producer完成後仍未READY: "
            f"benchmark={benchmark_id}, seed={seed}, status={final.get('status')}, "
            f"coverage={final.get('coverage_end')}, required={comparison_end_date}"
        )
    return {
        **final,
        "build_result": result,
        "dataset_manifest_path": dataset.get("manifest_path"),
        "dataset_dir": dataset.get("dataset_dir"),
    }


__all__ = [
    "RAW_PRICE_RUNTIME_DOMAIN",
    "ensure_raw_price_strategy_parameter_artifact",
    "inspect_raw_price_strategy_parameter_artifact",
    "ensure_raw_price_robustness_strategy_parameter_artifact",
    "inspect_raw_price_robustness_strategy_parameter_artifact",
    "raw_price_robustness_strategy_param_artifact_path",
    "raw_price_robustness_strategy_param_manifest_path",
]
