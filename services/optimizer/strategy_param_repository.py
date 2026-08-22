"""Low-level repository for canonical optimizer strategy-parameter manifests.

This module intentionally has no dependency on optimizer training/orchestration modules.
Writers and higher-level services may all depend on it without creating import cycles.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.training_policy import (
    get_robustness_benchmark_policy_snapshot,
    get_strategy_parameter_training_policy_snapshot,
)
from core.file_integrity import atomic_write_json, load_json_strict
from core.strategy_param_artifacts import (
    POLICY_FILENAME_BY_NAME,
    STRATEGY_PARAM_ARTIFACT_SCHEMA_VERSION,
    compute_strategy_param_file_sha256,
    compute_strategy_param_scientific_sha256,
    normalize_strategy_param_evaluation_mode,
    normalize_strategy_param_family,
    resolve_strategy_param_artifact_path,
    resolve_strategy_param_manifest_path,
    resolve_strategy_param_benchmark_artifact_path,
    resolve_strategy_param_benchmark_dir,
    resolve_strategy_param_benchmark_manifest_path,
    resolve_strategy_param_state_dir,
    STRATEGY_PARAM_STATE_FILENAME_BY_NAME,
    resolve_strategy_param_state_path,
)


def _project_relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(path, payload)


def _canonical_json_sha256(payload: Any) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _existing_source_records(
    root: Path, *, family: str, evaluation_mode: str
) -> dict[str, dict[str, Any]]:
    manifest_path = resolve_strategy_param_manifest_path(
        root, family=family, evaluation_mode=evaluation_mode
    )
    try:
        payload = load_json_strict(manifest_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for policy, record in dict(payload.get("artifacts") or {}).items():
        source = dict(dict(record or {}).get("source") or {})
        if source:
            out[str(policy)] = source
    return out


def _existing_benchmark_source_records(
    root: Path, *, benchmark_id: str, seed: int, family: str, evaluation_mode: str
) -> dict[str, dict[str, Any]]:
    manifest_path = resolve_strategy_param_benchmark_manifest_path(
        root, benchmark_id=benchmark_id, seed=seed, family=family,
        evaluation_mode=evaluation_mode,
    )
    try:
        payload = load_json_strict(manifest_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for policy, record in dict(payload.get("artifacts") or {}).items():
        source = dict(dict(record or {}).get("source") or {})
        if source:
            out[str(policy)] = source
    return out


def _storage_mode(evaluation_mode: str) -> str:
    mode = normalize_strategy_param_evaluation_mode(evaluation_mode)
    return "schedule" if mode in {"oos", "rolling"} else mode


def _training_policy_for_storage_mode(evaluation_mode: str) -> dict[str, Any]:
    mode = normalize_strategy_param_evaluation_mode(evaluation_mode)
    # OOS and Rolling consume the same canonical schedule.  The schedule itself is
    # produced by the PIT-safe rolling optimizer contract; OOS freezes it at read time.
    source_mode = "rolling" if mode in {"oos", "rolling"} else mode
    snapshot = dict(get_strategy_parameter_training_policy_snapshot(evaluation_mode=source_mode))
    if mode in {"oos", "rolling"}:
        snapshot["evaluation_mode"] = "schedule"
        snapshot["consumption_modes"] = ["oos", "rolling"]
    return snapshot


def build_strategy_parameter_manifest_payload(
    project_root: str | Path,
    *,
    family: str,
    evaluation_mode: str,
    source_records: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    family = normalize_strategy_param_family(family)
    mode = normalize_strategy_param_evaluation_mode(evaluation_mode)
    storage_mode = _storage_mode(mode)
    preserved_sources = _existing_source_records(root, family=family, evaluation_mode=mode)
    preserved_sources.update(dict(source_records or {}))
    artifacts: dict[str, Any] = {}
    for policy in POLICY_FILENAME_BY_NAME:
        path = resolve_strategy_param_artifact_path(
            root, family=family, evaluation_mode=mode, policy=policy
        )
        if not path.is_file():
            continue
        artifacts[policy] = {
            "path": _project_relative(root, path),
            "sha256": compute_strategy_param_file_sha256(path),
            "scientific_sha256": compute_strategy_param_scientific_sha256(path),
        }
        source = dict(preserved_sources.get(policy) or {})
        if source:
            artifacts[policy]["source"] = source
    state_artifacts: dict[str, Any] = {}
    if family == "full" and mode == "trade":
        state_dir = resolve_strategy_param_state_dir(root, family=family, evaluation_mode=mode)
        for state_name, filename in STRATEGY_PARAM_STATE_FILENAME_BY_NAME.items():
            state_path = state_dir / filename
            if state_path.is_file():
                state_artifacts[state_name] = {
                    "path": _project_relative(root, state_path),
                    "sha256": compute_strategy_param_file_sha256(state_path),
                }
    training_policy = _training_policy_for_storage_mode(mode)
    return {
        "schema_type": "canonical_strategy_parameter_artifact_set",
        "schema_version": STRATEGY_PARAM_ARTIFACT_SCHEMA_VERSION,
        "producer": "optimizer",
        "family": family,
        "evaluation_mode": storage_mode,
        "consumption_modes": ["oos", "rolling"] if storage_mode == "schedule" else [mode],
        "training_policy": training_policy,
        "training_policy_fingerprint": _canonical_json_sha256(training_policy)[:16],
        "artifacts": artifacts,
        "state_artifacts": state_artifacts,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def build_strategy_parameter_benchmark_manifest_payload(
    project_root: str | Path,
    *,
    benchmark_id: str,
    seed: int,
    family: str,
    evaluation_mode: str,
    source_records: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    family = normalize_strategy_param_family(family)
    mode = normalize_strategy_param_evaluation_mode(evaluation_mode)
    if mode not in {"oos", "rolling"}:
        raise ValueError("robustness benchmark strategy params只支援oos/rolling")
    preserved_sources = _existing_benchmark_source_records(
        root, benchmark_id=benchmark_id, seed=int(seed), family=family,
        evaluation_mode=mode,
    )
    preserved_sources.update(dict(source_records or {}))
    artifacts: dict[str, Any] = {}
    for policy in POLICY_FILENAME_BY_NAME:
        path = resolve_strategy_param_benchmark_artifact_path(
            root, benchmark_id=benchmark_id, seed=seed, family=family,
            evaluation_mode=mode, policy=policy,
        )
        if not path.is_file():
            continue
        artifacts[policy] = {
            "path": _project_relative(root, path),
            "sha256": compute_strategy_param_file_sha256(path),
            "scientific_sha256": compute_strategy_param_scientific_sha256(path),
        }
        source = dict(preserved_sources.get(policy) or {})
        if source:
            artifacts[policy]["source"] = source
    benchmark_policy = get_robustness_benchmark_policy_snapshot()
    if int(seed) not in {int(value) for value in benchmark_policy["resolved_seeds"]}:
        raise ValueError(f"seed不屬於目前robustness benchmark題庫: {seed}")
    training_policy = {
        **get_strategy_parameter_training_policy_snapshot(evaluation_mode="rolling"),
        "evaluation_mode": "schedule",
        "consumption_modes": ["oos", "rolling"],
        # The benchmark overrides only the stochastic seed. Optimizer budget/window/cadence
        # remain owned by the canonical optimizer training policy.
        "optimizer_seed": int(seed),
        "benchmark_seed_override": True,
    }
    return {
        "schema_type": "strategy_parameter_robustness_benchmark_artifact_set",
        "schema_version": STRATEGY_PARAM_ARTIFACT_SCHEMA_VERSION,
        "producer": "optimizer",
        "benchmark": benchmark_policy,
        "benchmark_id": str(benchmark_id),
        "benchmark_seed": int(seed),
        "benchmark_seed_index": list(benchmark_policy["resolved_seeds"]).index(int(seed)) + 1,
        "family": family,
        "evaluation_mode": "schedule",
        "consumption_modes": ["oos", "rolling"],
        "training_policy": training_policy,
        "training_policy_fingerprint": _canonical_json_sha256(training_policy)[:16],
        "artifacts": artifacts,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def refresh_strategy_parameter_benchmark_manifest(
    project_root: str | Path,
    *,
    benchmark_id: str,
    seed: int,
    family: str,
    evaluation_mode: str,
    source_records: dict[str, dict[str, Any]] | None = None,
) -> Path:
    root = Path(project_root).resolve()
    manifest_path = resolve_strategy_param_benchmark_manifest_path(
        root,
        benchmark_id=benchmark_id,
        seed=seed,
        family=family,
        evaluation_mode=evaluation_mode,
    )
    _write_json(
        manifest_path,
        build_strategy_parameter_benchmark_manifest_payload(
            root,
            benchmark_id=benchmark_id,
            seed=seed,
            family=family,
            evaluation_mode=evaluation_mode,
            source_records=source_records,
        ),
    )
    return manifest_path

def write_strategy_parameter_state_artifact(
    project_root: str | Path,
    *,
    artifact: str,
    payload: dict[str, Any],
    family: str = "full",
    evaluation_mode: str = "trade",
    backup_existing: bool = False,
    backup_label: str | None = None,
) -> dict[str, Any]:
    """Atomically write Optimizer-owned runtime/candidate state and refresh manifest."""
    root = Path(project_root).resolve()
    family = normalize_strategy_param_family(family)
    mode = normalize_strategy_param_evaluation_mode(evaluation_mode)
    target = resolve_strategy_param_state_path(
        root, artifact=artifact, family=family, evaluation_mode=mode
    )
    backup_path: Path | None = None
    if backup_existing and target.is_file():
        backups = target.parent / "backups"
        backups.mkdir(parents=True, exist_ok=True)
        suffix = str(backup_label or "previous").strip() or "previous"
        backup_path = backups / f"{target.stem}_{suffix}.json"
        shutil.copy2(target, backup_path)
    _write_json(target, payload)
    manifest = refresh_strategy_parameter_manifest(
        root, family=family, evaluation_mode=mode
    )
    return {
        "path": target,
        "sha256": compute_strategy_param_file_sha256(target),
        "backup_path": backup_path,
        "manifest_path": manifest,
    }

def refresh_strategy_parameter_manifest(
    project_root: str | Path,
    *,
    family: str,
    evaluation_mode: str,
    source_records: dict[str, dict[str, Any]] | None = None,
) -> Path:
    root = Path(project_root).resolve()
    manifest_path = resolve_strategy_param_manifest_path(
        root, family=family, evaluation_mode=evaluation_mode
    )
    _write_json(
        manifest_path,
        build_strategy_parameter_manifest_payload(
            root,
            family=family,
            evaluation_mode=evaluation_mode,
            source_records=source_records,
        ),
    )
    return manifest_path


__all__ = [
    "build_strategy_parameter_manifest_payload",
    "build_strategy_parameter_benchmark_manifest_payload",
    "refresh_strategy_parameter_benchmark_manifest",
    "refresh_strategy_parameter_manifest",
    "write_strategy_parameter_state_artifact",
]
