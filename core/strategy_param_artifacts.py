"""Canonical strategy-parameter artifact paths and identities.

The optimizer domain is the only producer of current strategy-parameter artifacts.
Research/Strategy Compare resolves artifacts through this module and never assembles
optimizer output paths on its own.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

STRATEGY_PARAM_ARTIFACT_SCHEMA_VERSION = 1
STRATEGY_PARAM_ROOT_RELATIVE = Path("models") / "strategy_params"

STRATEGY_PARAM_FAMILIES = ("full", "min")
STRATEGY_PARAM_EVALUATION_MODES = ("study", "full", "oos", "rolling", "trade")

POLICY_FILENAME_BY_NAME = {
    "base_finalist_best": "base_best.json",
    "local_finalist_best": "local_best.json",
    "retention_finalist_best": "retention_best.json",
    "base_finalists_agree": "base_finalists_agree.json",
    "local_finalists_agree": "local_finalists_agree.json",
    "retention_finalists_agree": "retention_finalists_agree.json",
    "base": "ensemble_base.json",
    "local": "ensemble_local.json",
    "retention": "ensemble_retention.json",
}

COMPARE_PARAM_POLICY_TO_OPTIMIZER_POLICY = {
    "base-finalist-best": "base_finalist_best",
    "base-finalists-agree": "base_finalists_agree",
}


def normalize_strategy_param_family(value: str) -> str:
    family = str(value or "").strip().lower()
    if family not in STRATEGY_PARAM_FAMILIES:
        raise ValueError(f"不支援的策略參數family: {value!r}")
    return family


def normalize_strategy_param_evaluation_mode(value: str) -> str:
    mode = str(value or "").strip().lower()
    if mode == "roos":
        mode = "rolling"
    if mode == "split":
        mode = "oos"
    if mode not in STRATEGY_PARAM_EVALUATION_MODES:
        raise ValueError(f"不支援的策略參數evaluation mode: {value!r}")
    return mode


def normalize_strategy_param_policy(value: str) -> str:
    policy = str(value or "").strip()
    policy = COMPARE_PARAM_POLICY_TO_OPTIMIZER_POLICY.get(policy, policy)
    if policy not in POLICY_FILENAME_BY_NAME:
        raise ValueError(f"不支援的策略參數policy: {value!r}")
    return policy


def resolve_strategy_param_dir(project_root: str | Path, *, family: str, evaluation_mode: str) -> Path:
    root = Path(project_root).resolve()
    return root / STRATEGY_PARAM_ROOT_RELATIVE / normalize_strategy_param_family(family) / normalize_strategy_param_evaluation_mode(evaluation_mode)


def resolve_strategy_param_artifact_path(
    project_root: str | Path,
    *,
    family: str,
    evaluation_mode: str,
    policy: str,
) -> Path:
    normalized_policy = normalize_strategy_param_policy(policy)
    return resolve_strategy_param_dir(project_root, family=family, evaluation_mode=evaluation_mode) / POLICY_FILENAME_BY_NAME[normalized_policy]


def resolve_strategy_param_manifest_path(project_root: str | Path, *, family: str, evaluation_mode: str) -> Path:
    return resolve_strategy_param_dir(project_root, family=family, evaluation_mode=evaluation_mode) / "manifest.json"


def compute_strategy_param_file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_strategy_param_manifest(project_root: str | Path, *, family: str, evaluation_mode: str) -> dict[str, Any] | None:
    path = resolve_strategy_param_manifest_path(project_root, family=family, evaluation_mode=evaluation_mode)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


__all__ = [
    "STRATEGY_PARAM_ARTIFACT_SCHEMA_VERSION",
    "STRATEGY_PARAM_ROOT_RELATIVE",
    "STRATEGY_PARAM_FAMILIES",
    "STRATEGY_PARAM_EVALUATION_MODES",
    "POLICY_FILENAME_BY_NAME",
    "COMPARE_PARAM_POLICY_TO_OPTIMIZER_POLICY",
    "normalize_strategy_param_family",
    "normalize_strategy_param_evaluation_mode",
    "normalize_strategy_param_policy",
    "resolve_strategy_param_dir",
    "resolve_strategy_param_artifact_path",
    "resolve_strategy_param_manifest_path",
    "compute_strategy_param_file_sha256",
    "load_strategy_param_manifest",
]
