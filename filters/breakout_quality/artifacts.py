"""Validation helpers for canonical breakout quality artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

from filters.breakout_quality.contract import (
    ARTIFACT_CONTRACT_VERSION,
    CONTEXT_COLUMNS,
    DEFAULT_MODEL_FILENAME,
    DEFAULT_SCORE_FILENAME,
    DEFAULT_SPLIT_FILENAME,
    FEATURE_COLUMNS,
    FILTER_FAMILY,
    OUTER_SPLIT_VALUES,
    SELECTION_ROLE_VALUES,
    SCORE_COLUMN,
    SCORE_COMPARISON,
    SCORE_TABLE_REQUIRED_COLUMNS,
    SCORE_TABLE_SCHEMA_VERSION,
    SCORE_THRESHOLD_SOURCE,
    SPLIT_ASSIGNMENT_REQUIRED_COLUMNS,
    SPLIT_ASSIGNMENT_SCHEMA_VERSION,
    RUNTIME_ELIGIBLE_SCOPES,
    RUNTIME_SCOPE_FORWARD_OOS,
)
from filters.breakout_quality.csv_io import read_breakout_quality_csv
from filters.breakout_quality.paths import BreakoutQualityArtifactPaths, resolve_filter_artifact_paths
from filters.breakout_quality.splits import (
    compute_outer_policy_fingerprint,
    validate_split_assignment_frame,
)


@dataclass(frozen=True)
class BreakoutQualityModelContract:
    paths: BreakoutQualityArtifactPaths
    manifest: dict[str, Any]


@dataclass(frozen=True)
class BreakoutQualityRuntimeContract:
    paths: BreakoutQualityArtifactPaths
    manifest: dict[str, Any]
    high_len_values: tuple[int, ...]
    available_from: date
    available_through: date


def compute_file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_file_manifest(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    return {
        "filename": resolved.name,
        "sha256": compute_file_sha256(resolved),
        "size_bytes": int(resolved.stat().st_size),
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"無法讀取 breakout quality manifest: {path}; {type(exc).__name__}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"breakout quality manifest 根節點必須是 object: {path}")
    return payload


def _require_mapping(payload: dict[str, Any], field_name: str) -> dict[str, Any]:
    value = payload.get(field_name)
    if not isinstance(value, dict):
        raise ValueError(f"breakout quality manifest 缺少 object 欄位: {field_name}")
    return value


def _require_nonempty_text(payload: dict[str, Any], field_name: str) -> str:
    value = str(payload.get(field_name, "")).strip()
    if not value:
        raise ValueError(f"breakout quality manifest 欄位不可空白: {field_name}")
    return value


def _parse_iso_date(raw_value: Any, *, field_name: str) -> date:
    value = str(raw_value or "").strip()
    if not value:
        raise ValueError(f"breakout quality manifest 日期欄位不可空白: {field_name}")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"breakout quality manifest 日期格式錯誤: {field_name}={value!r}") from exc


def _validate_file_record(path: Path, record: dict[str, Any], *, expected_filename: str, field_name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"breakout quality 正式工件不存在: {path}")
    filename = _require_nonempty_text(record, "filename")
    if filename != expected_filename or path.name != expected_filename:
        raise ValueError(
            f"breakout quality {field_name} 檔名契約不一致: manifest={filename!r}, expected={expected_filename!r}, path={path}"
        )
    expected_size = int(record.get("size_bytes", -1))
    actual_size = int(path.stat().st_size)
    if expected_size != actual_size:
        raise ValueError(
            f"breakout quality {field_name} size 不一致: expected={expected_size}, actual={actual_size}, path={path}"
        )
    expected_hash = _require_nonempty_text(record, "sha256").lower()
    actual_hash = compute_file_sha256(path).lower()
    if actual_hash != expected_hash:
        raise ValueError(
            f"breakout quality {field_name} SHA256 不一致: expected={expected_hash}, actual={actual_hash}, path={path}"
        )


def _normalize_high_len_values(raw_values: Any) -> tuple[int, ...]:
    if not isinstance(raw_values, list) or not raw_values:
        raise ValueError("breakout quality score_table.high_len_values 必須是非空 list")
    values = tuple(sorted({int(value) for value in raw_values}))
    if values[0] < 1:
        raise ValueError("breakout quality score_table.high_len_values 只能包含正整數")
    return values


def _validate_split_assignment_record(paths: BreakoutQualityArtifactPaths, manifest: dict[str, Any]):
    record = _require_mapping(manifest, "split_assignments")
    if int(record.get("schema_version", -1)) != int(SPLIT_ASSIGNMENT_SCHEMA_VERSION):
        raise ValueError("breakout quality split assignment schema 版本不相容")
    if list(record.get("required_columns", [])) != list(SPLIT_ASSIGNMENT_REQUIRED_COLUMNS):
        raise ValueError("breakout quality split assignment required_columns 與正式契約不一致")
    if list(record.get("columns", [])) != list(SPLIT_ASSIGNMENT_REQUIRED_COLUMNS):
        raise ValueError("breakout quality split assignment columns metadata 與正式契約不一致")
    _validate_file_record(
        paths.split_path,
        record,
        expected_filename=DEFAULT_SPLIT_FILENAME,
        field_name="split_assignments",
    )
    frame = validate_split_assignment_frame(read_breakout_quality_csv(paths.split_path))
    if int(record.get("row_count", -1)) != len(frame):
        raise ValueError("breakout quality split assignment row_count 與檔案不一致")
    outer_counts = {
        name: int((frame["outer_split"] == name).sum())
        for name in OUTER_SPLIT_VALUES
    }
    selection_role_counts = {
        name: int((frame["selection_role"] == name).sum())
        for name in SELECTION_ROLE_VALUES
    }
    expected_outer = record.get("outer_split_counts")
    expected_selection_roles = record.get("selection_role_counts")
    if not isinstance(expected_outer, dict) or {
        name: int(expected_outer.get(name, -1)) for name in OUTER_SPLIT_VALUES
    } != outer_counts:
        raise ValueError(
            f"breakout quality split assignment outer_split_counts 不一致: expected={expected_outer}, actual={outer_counts}"
        )
    if not isinstance(expected_selection_roles, dict) or {
        name: int(expected_selection_roles.get(name, -1))
        for name in SELECTION_ROLE_VALUES
    } != selection_role_counts:
        raise ValueError(
            "breakout quality split assignment selection_role_counts 不一致: "
            f"expected={expected_selection_roles}, actual={selection_role_counts}"
        )
    if str(record.get("group_key") or "").strip() != "ticker/date/high_len":
        raise ValueError("breakout quality split assignment group_key 必須是 ticker/date/high_len")
    outer_policy = _require_mapping(manifest, "outer_oos_policy")
    _require_nonempty_text(outer_policy, "policy_source")
    expected_policy_fingerprint = _require_nonempty_text(outer_policy, "policy_fingerprint_sha256")
    actual_policy_fingerprint = compute_outer_policy_fingerprint(outer_policy)
    if expected_policy_fingerprint != actual_policy_fingerprint:
        raise ValueError(
            "breakout quality outer_oos_policy fingerprint 不一致: "
            f"expected={expected_policy_fingerprint}, actual={actual_policy_fingerprint}"
        )
    for field_name in (
        "selection_start_date",
        "selection_end_date",
        "oos_start_date",
        "effective_oos_end_date",
    ):
        _parse_iso_date(outer_policy.get(field_name), field_name=f"outer_oos_policy.{field_name}")
    return frame


@lru_cache(maxsize=16)
def load_model_artifact_contract(
    project_root: str,
    filter_id: str,
) -> BreakoutQualityModelContract:
    paths = resolve_filter_artifact_paths(project_root, filter_id)
    if not paths.manifest_path.is_file():
        raise FileNotFoundError(
            f"找不到 breakout quality 正式 manifest: {paths.manifest_path}。"
            "請先以同一 filter_id 完成 build_dataset 與 train。"
        )
    manifest = _read_json_object(paths.manifest_path)

    version = int(manifest.get("artifact_contract_version", -1))
    if version != int(ARTIFACT_CONTRACT_VERSION):
        raise ValueError(
            f"breakout quality artifact contract 版本不相容: manifest={version}, runtime={ARTIFACT_CONTRACT_VERSION}; "
            f"path={paths.manifest_path}"
        )
    if _require_nonempty_text(manifest, "filter_family") != FILTER_FAMILY:
        raise ValueError(f"breakout quality manifest filter_family 不一致: {paths.manifest_path}")
    if _require_nonempty_text(manifest, "filter_id") != str(filter_id).strip():
        raise ValueError(f"breakout quality manifest filter_id 不一致: {paths.manifest_path}")
    if list(manifest.get("feature_columns", [])) != list(FEATURE_COLUMNS):
        raise ValueError("breakout quality manifest feature_columns 與 runtime 契約不一致")
    if list(manifest.get("context_columns", [])) != list(CONTEXT_COLUMNS):
        raise ValueError("breakout quality manifest context_columns 與 runtime 契約不一致")

    score_decision = _require_mapping(manifest, "score_decision")
    if _require_nonempty_text(score_decision, "score_column") != SCORE_COLUMN:
        raise ValueError("breakout quality manifest score_column 與 runtime 契約不一致")
    if _require_nonempty_text(score_decision, "comparison") != SCORE_COMPARISON:
        raise ValueError("breakout quality manifest score comparison 與 runtime 契約不一致")
    if _require_nonempty_text(score_decision, "threshold_source") != SCORE_THRESHOLD_SOURCE:
        raise ValueError("breakout quality threshold 必須由 active strategy param 提供")

    model_record = _require_mapping(manifest, "model")
    _validate_file_record(
        paths.model_path,
        model_record,
        expected_filename=DEFAULT_MODEL_FILENAME,
        field_name="model",
    )
    _validate_split_assignment_record(paths, manifest)
    if _require_nonempty_text(manifest, "training_mode") != "fixed_epoch_full_selection":
        raise ValueError("breakout quality model 必須使用 fixed_epoch_full_selection")
    fixed_epochs = int(manifest.get("fixed_epochs", 0))
    completed_epochs = int(manifest.get("completed_epochs", 0))
    if fixed_epochs < 1 or completed_epochs != fixed_epochs:
        raise ValueError(
            "breakout quality fixed epoch 契約不一致: "
            f"fixed={fixed_epochs}, completed={completed_epochs}"
        )
    if bool(manifest.get("early_stopping_enabled", True)):
        raise ValueError("breakout quality scheme B 禁止 early stopping")
    if bool(manifest.get("inner_validation_used", True)):
        raise ValueError("breakout quality scheme B 禁止 inner validation")
    if not bool(manifest.get("training_uses_all_eligible_selection_rows", False)):
        raise ValueError("breakout quality scheme B 必須使用全部 eligible Selection rows")
    if bool(manifest.get("oos_predictions_used_during_training", True)):
        raise ValueError("breakout quality OOS predictions 不可用於訓練")
    if bool(manifest.get("oos_metrics_emitted_by_train", True)):
        raise ValueError("breakout quality train.py 不可輸出 OOS metrics")
    fixed_threshold = float(manifest.get("fixed_evaluation_threshold", float("nan")))
    if not 0.0 <= fixed_threshold <= 1.0:
        raise ValueError("breakout quality fixed_evaluation_threshold 必須介於 0 與 1")
    threshold_policy = _require_mapping(manifest, "threshold_policy")
    if _require_nonempty_text(threshold_policy, "mode") != "fixed_before_oos":
        raise ValueError("breakout quality threshold_policy 必須在 OOS 前固定")
    if float(threshold_policy.get("evaluation_threshold", float("nan"))) != fixed_threshold:
        raise ValueError("breakout quality threshold_policy 與 fixed_evaluation_threshold 不一致")
    if _require_nonempty_text(threshold_policy, "runtime_source") != SCORE_THRESHOLD_SOURCE:
        raise ValueError("breakout quality threshold runtime source 與正式契約不一致")
    if bool(threshold_policy.get("optimized_by_train", True)):
        raise ValueError("breakout quality scheme B 不可由 train.py 最佳化 threshold")
    if bool(threshold_policy.get("oos_tuning_allowed", True)):
        raise ValueError("breakout quality scheme B 禁止使用 OOS 調整 threshold")

    return BreakoutQualityModelContract(paths=paths, manifest=manifest)


@lru_cache(maxsize=16)
def load_split_assignment_frame(project_root: str, filter_id: str):
    model_contract = load_model_artifact_contract(project_root, filter_id)
    return _validate_split_assignment_record(model_contract.paths, model_contract.manifest)


@lru_cache(maxsize=16)
def load_runtime_artifact_contract(
    project_root: str,
    filter_id: str,
) -> BreakoutQualityRuntimeContract:
    model_contract = load_model_artifact_contract(project_root, filter_id)
    paths = model_contract.paths
    manifest = model_contract.manifest
    runtime_eligibility = _require_mapping(manifest, "runtime_eligibility")
    if runtime_eligibility.get("eligible") is not True:
        reason = str(runtime_eligibility.get("reason", "artifact 未標記為 OOS runtime 可用")).strip()
        raise ValueError(f"breakout quality artifact 不可用於正式 runtime: {reason}; path={paths.manifest_path}")
    scope = _require_nonempty_text(runtime_eligibility, "scope")
    if scope not in RUNTIME_ELIGIBLE_SCOPES:
        raise ValueError(f"breakout quality runtime scope 不合法: {scope!r}")
    available_from = _parse_iso_date(runtime_eligibility.get("available_from"), field_name="runtime_eligibility.available_from")
    available_through = _parse_iso_date(
        runtime_eligibility.get("available_through"),
        field_name="runtime_eligibility.available_through",
    )
    if available_through < available_from:
        raise ValueError("breakout quality runtime available_through 不可早於 available_from")
    if scope == RUNTIME_SCOPE_FORWARD_OOS:
        information_cutoff = _parse_iso_date(
            runtime_eligibility.get("model_information_cutoff"),
            field_name="runtime_eligibility.model_information_cutoff",
        )
        if available_from <= information_cutoff:
            raise ValueError(
                "breakout quality forward_oos available_from 必須嚴格晚於 model_information_cutoff"
            )

    score_table = _require_mapping(manifest, "score_table")
    if int(score_table.get("schema_version", -1)) != int(SCORE_TABLE_SCHEMA_VERSION):
        raise ValueError("breakout quality score table schema 版本不相容")
    if list(score_table.get("required_columns", [])) != list(SCORE_TABLE_REQUIRED_COLUMNS):
        raise ValueError("breakout quality score table required_columns 與 runtime 契約不一致")
    _validate_file_record(
        paths.score_path,
        score_table,
        expected_filename=DEFAULT_SCORE_FILENAME,
        field_name="score_table",
    )
    row_count = int(score_table.get("row_count", -1))
    if row_count < 1:
        raise ValueError("breakout quality 正式 score table row_count 必須 >= 1")
    columns = score_table.get("columns")
    if not isinstance(columns, list) or list(columns) != list(SCORE_TABLE_REQUIRED_COLUMNS):
        raise ValueError("breakout quality 正式 score table columns 必須精確等於 required_columns")
    event_date_range = _require_mapping(score_table, "event_date_range")
    event_start = _parse_iso_date(event_date_range.get("start"), field_name="score_table.event_date_range.start")
    event_end = _parse_iso_date(event_date_range.get("end"), field_name="score_table.event_date_range.end")
    if event_start != available_from:
        raise ValueError("breakout quality available_from 必須等於 score table 第一個事件日")
    if event_end < event_start or event_end > available_through:
        raise ValueError("breakout quality score table event_date_range 與 runtime availability 不一致")
    high_len_values = _normalize_high_len_values(score_table.get("high_len_values"))
    return BreakoutQualityRuntimeContract(
        paths=paths,
        manifest=manifest,
        high_len_values=high_len_values,
        available_from=available_from,
        available_through=available_through,
    )


def validate_required_high_len(contract: BreakoutQualityRuntimeContract, high_len: int) -> None:
    value = int(high_len)
    if value not in contract.high_len_values:
        preview = ", ".join(str(item) for item in contract.high_len_values[:12])
        suffix = "..." if len(contract.high_len_values) > 12 else ""
        raise ValueError(
            f"breakout quality score table 不含目前 active high_len={value}。"
            f"artifact 支援值={preview}{suffix}；請重建同一正式工件，不可由 runtime 猜測或改寫設定。"
        )


__all__ = [
    "BreakoutQualityModelContract",
    "BreakoutQualityRuntimeContract",
    "build_file_manifest",
    "compute_file_sha256",
    "load_model_artifact_contract",
    "load_split_assignment_frame",
    "load_runtime_artifact_contract",
    "validate_required_high_len",
]
