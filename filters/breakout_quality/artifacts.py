"""Validation helpers for canonical breakout quality artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

from config.breakout_quality import (
    BASELINE_EXPERIMENT_PROFILE,
    TIME_WEIGHT_MODE_DATE_BALANCED,
    TRAINING_WEIGHT_REDUCTION_BATCH_WEIGHT_SUM,
    TRAINING_WEIGHT_REDUCTION_FIXED_BATCH_SIZE,
    TRAINING_SAMPLING_ALL_EVENT_ROWS,
    TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
    TRAINING_OBJECTIVE_BINARY_CLASSIFICATION,
    build_breakout_quality_pretraining_profile_payload,
    get_breakout_quality_experiment_profile,
    normalize_breakout_quality_experiment_profile,
)
from config.breakout_quality import (
    BREAKOUT_QUALITY_CLASS_WEIGHT_MODE,
    BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
    BREAKOUT_QUALITY_FINAL_REFIT_MODE,
    BREAKOUT_QUALITY_PRETRAINING_PROFILE,
    BREAKOUT_QUALITY_TIME_WEIGHT_MODE,
)

from filters.breakout_quality.contract import (
    ARTIFACT_CONTRACT_VERSION,
    CONTEXT_COLUMNS,
    DEFAULT_LABEL_POLICY,
    DEFAULT_MODEL_FILENAME,
    DEFAULT_SCORE_FILENAME,
    DEFAULT_UNAVAILABLE_SCORE_FILENAME,
    DEFAULT_SPLIT_FILENAME,
    FEATURE_COLUMNS,
    FILTER_FAMILY,
    OUTER_SPLIT_VALUES,
    SELECTION_ROLE_INNER_EMBARGO,
    SELECTION_ROLE_VALIDATION,
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
    TRAINING_MODE_FIXED_EPOCH_FULL_SELECTION,
    TRAINING_MODE_INNER_VALIDATION_FULL_REFIT,
)
from filters.breakout_quality.csv_io import read_breakout_quality_csv
from filters.breakout_quality.lr_schedule import validate_learning_rate_schedule_record
from filters.breakout_quality.models.spec import (
    model_spec_from_manifest,
    validate_model_sequence_length,
)
from filters.breakout_quality.mantis_contract import (
    MANTIS_PACKAGE_NAME,
    MANTIS_PACKAGE_VERSION,
    MANTIS_V2_CHECKPOINT_FILENAME,
    MANTIS_V2_CHECKPOINT_SHA256,
    MANTIS_V2_CONFIG_FILENAME,
    MANTIS_V2_CONFIG_SHA256,
    MANTIS_V2_REPOSITORY,
    MANTIS_V2_REVISION,
)
from filters.breakout_quality.moment_contract import (
    MOMENT_CHECKPOINT_FILENAME,
    MOMENT_CHECKPOINT_SHA256,
    MOMENT_CHECKPOINT_SIZE_BYTES,
    MOMENT_CONFIG_FILENAME,
    MOMENT_CONFIG_SIZE_BYTES,
    MOMENT_PACKAGE_NAME,
    MOMENT_PACKAGE_VERSION,
    MOMENT_REPOSITORY,
    MOMENT_REVISION,
    MOMENT_TRANSFORMERS_PACKAGE_NAME,
    MOMENT_TRANSFORMERS_VERSION,
)
from filters.breakout_quality.torch_runtime import (
    SUPPORTED_MIXED_PRECISION_DTYPES,
    SUPPORTED_TORCH_DEVICES,
)
from filters.breakout_quality.paths import (
    BreakoutQualityArtifactPaths,
    resolve_existing_filter_artifact_paths,
)
from filters.breakout_quality.source_inventory import build_source_data_inventory
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
    required_signal_start: date
    execution_start: date
    shared_group_score_broadcast: bool


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


def _validate_torch_execution_record(
    manifest: dict[str, Any],
    *,
    required: bool,
) -> dict[str, Any] | None:
    raw = manifest.get("torch_execution")
    if raw is None and not required:
        return None
    if not isinstance(raw, dict):
        raise ValueError("breakout quality manifest.torch_execution 必須是 object")
    requested = str(raw.get("requested_device") or "").strip().lower()
    resolved = str(raw.get("resolved_device") or "").strip().lower()
    dtype_name = str(raw.get("autocast_dtype") or "").strip().lower()
    if requested not in SUPPORTED_TORCH_DEVICES:
        raise ValueError("breakout quality torch_execution.requested_device 不合法")
    if resolved not in {"cpu", "cuda"}:
        raise ValueError("breakout quality torch_execution.resolved_device 不合法")
    if dtype_name not in {"float32", *SUPPORTED_MIXED_PRECISION_DTYPES[1:]}:
        raise ValueError("breakout quality torch_execution.autocast_dtype 不合法")
    for field_name in (
        "mixed_precision_requested",
        "mixed_precision_enabled",
        "deterministic_algorithms",
        "allow_tf32",
    ):
        if not isinstance(raw.get(field_name), bool):
            raise ValueError(f"breakout quality torch_execution.{field_name} 必須是 bool")
    mixed_requested = bool(raw.get("mixed_precision_requested"))
    mixed_enabled = bool(raw.get("mixed_precision_enabled"))
    if requested == "cpu" and resolved != "cpu":
        raise ValueError("requested CPU 的 torch_execution 不得解析成 CUDA")
    if requested == "cuda" and resolved != "cuda":
        raise ValueError("requested CUDA 的 torch_execution 不得解析成 CPU")
    if mixed_enabled and not mixed_requested:
        raise ValueError("torch_execution 不得在未要求 mixed precision 時宣稱已啟用")
    if not mixed_enabled and dtype_name != "float32":
        raise ValueError("未啟用 mixed precision 時 autocast_dtype 必須是 float32")
    if resolved == "cpu" and mixed_enabled:
        raise ValueError("CPU torch_execution 不得宣稱 mixed precision")
    if resolved == "cuda" and mixed_enabled and dtype_name not in {
        "float16",
        "bfloat16",
    }:
        raise ValueError("CUDA mixed precision 的 autocast_dtype 必須是 float16 或 bfloat16")
    return raw


def _validate_training_augmentation_epoch_records(
    records: Any,
    *,
    augmentation_name: str,
    augmentation_parameters: dict[str, Any],
    field_name: str,
) -> list[dict[str, Any]]:
    if not isinstance(records, list) or not records:
        raise ValueError(f"breakout quality {field_name} 必須是非空 list")
    validated: list[dict[str, Any]] = []
    min_mask_bars = int(augmentation_parameters.get("min_mask_bars", 0))
    max_mask_bars = int(augmentation_parameters.get("max_mask_bars", 0))
    for index, raw in enumerate(records):
        if not isinstance(raw, dict):
            raise ValueError(f"breakout quality {field_name}[{index}] 必須是 object")
        if str(raw.get("name") or "").strip().lower() != augmentation_name:
            raise ValueError(f"breakout quality {field_name}[{index}].name 不一致")
        sample_count = int(raw.get("sample_count", -1))
        augmented_count = int(raw.get("augmented_sample_count", -1))
        masked_bar_count = int(raw.get("masked_bar_count", -1))
        if (
            sample_count < 0
            or augmented_count < 0
            or augmented_count > sample_count
            or masked_bar_count < 0
        ):
            raise ValueError(f"breakout quality {field_name}[{index}] 計數不合法")
        if augmentation_name == "none":
            if augmented_count != 0 or masked_bar_count != 0:
                raise ValueError(f"breakout quality {field_name}[{index}] none 計數必須為 0")
        elif not (
            augmented_count * min_mask_bars
            <= masked_bar_count
            <= augmented_count * max_mask_bars
        ):
            raise ValueError(
                f"breakout quality {field_name}[{index}] masked_bar_count 超出 profile 範圍"
            )
        validated.append(raw)
    return validated


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
    backward_compatible_zero_roles = {
        SELECTION_ROLE_VALIDATION,
        SELECTION_ROLE_INNER_EMBARGO,
    }
    normalized_expected_roles = None
    if isinstance(expected_selection_roles, dict):
        normalized_expected_roles = {
            name: int(
                expected_selection_roles.get(
                    name,
                    0 if name in backward_compatible_zero_roles else -1,
                )
            )
            for name in SELECTION_ROLE_VALUES
        }
    if normalized_expected_roles != selection_role_counts:
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
    model_architecture: str | None = None,
    experiment_profile: str | None = None,
) -> BreakoutQualityModelContract:
    paths = resolve_existing_filter_artifact_paths(
        project_root,
        filter_id,
        model_architecture,
        experiment_profile,
    )
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
    manifest_profile = normalize_breakout_quality_experiment_profile(
        str(manifest.get("experiment_profile") or BASELINE_EXPERIMENT_PROFILE)
    )
    if manifest_profile != paths.experiment_profile:
        raise ValueError(
            "breakout quality manifest experiment_profile 與工件路徑不一致: "
            f"manifest={manifest_profile}, path={paths.experiment_profile}"
        )
    expected_experiment = get_breakout_quality_experiment_profile(manifest_profile)
    if expected_experiment.training_objective != TRAINING_OBJECTIVE_BINARY_CLASSIFICATION:
        raise ValueError(
            "research-only continuous ranker artifact不得載入正式binary runtime contract"
        )
    manifest_experiment = manifest.get("experiment_settings")
    if manifest_experiment is None and manifest_profile == BASELINE_EXPERIMENT_PROFILE:
        manifest_experiment = expected_experiment.as_manifest_payload()
    if manifest_experiment != expected_experiment.as_manifest_payload():
        raise ValueError(
            "breakout quality experiment_settings 與命名 profile 不一致: "
            f"profile={manifest_profile}, expected={expected_experiment.as_manifest_payload()}, "
            f"actual={manifest_experiment}"
        )
    training_weight_reduction = str(
        manifest.get("training_weight_reduction")
        or TRAINING_WEIGHT_REDUCTION_BATCH_WEIGHT_SUM
    ).strip().lower()
    if training_weight_reduction != expected_experiment.training_weight_reduction:
        raise ValueError(
            "breakout quality training_weight_reduction 與 experiment profile 不一致: "
            f"manifest={training_weight_reduction}, "
            f"expected={expected_experiment.training_weight_reduction}"
        )
    if list(manifest.get("feature_columns", [])) != list(FEATURE_COLUMNS):
        raise ValueError("breakout quality manifest feature_columns 與 runtime 契約不一致")
    if list(manifest.get("context_columns", [])) != list(CONTEXT_COLUMNS):
        raise ValueError("breakout quality manifest context_columns 與 runtime 契約不一致")
    manifest_architecture = _require_nonempty_text(manifest, "model_architecture")
    if manifest_architecture != paths.model_architecture:
        raise ValueError(
            "breakout quality manifest model_architecture 與工件路徑不一致: "
            f"manifest={manifest_architecture}, path={paths.model_architecture}"
        )
    model_spec = model_spec_from_manifest(_require_mapping(manifest, "model_spec"))
    if model_spec.architecture != manifest_architecture:
        raise ValueError("breakout quality model_spec.architecture 與 manifest 不一致")
    validate_model_sequence_length(model_spec, int(manifest.get("sequence_length", 0)))
    _validate_torch_execution_record(
        manifest,
        required=model_spec.family in {"inception_time", "modern_tcn", "patch_transformer", "ts2vec_frozen_linear", "mantis_v2_frozen_linear", "moment_frozen_linear"},
    )
    if int(manifest.get("trainable_parameter_count", 0)) < 1:
        raise ValueError("breakout quality trainable_parameter_count 必須 >=1")
    trainable_parameter_count = int(manifest.get("trainable_parameter_count", 0))
    total_parameter_count = int(manifest.get("total_parameter_count", trainable_parameter_count))
    frozen_parameter_count = int(manifest.get("frozen_parameter_count", 0))
    if total_parameter_count < trainable_parameter_count or frozen_parameter_count != (
        total_parameter_count - trainable_parameter_count
    ):
        raise ValueError("breakout quality total/frozen/trainable parameter count 不一致")
    if model_spec.family == "moment_frozen_linear":
        if frozen_parameter_count < 1:
            raise ValueError("MOMENT frozen probe 必須包含 frozen encoder parameters")
        if manifest.get("self_supervised_pretraining") is not None:
            raise ValueError("MOMENT 不得混用 Selection-only self-supervised pretraining")
        external = _require_mapping(manifest, "external_pretrained_encoder")
        if str(external.get("source_type") or "") != "hugging_face_snapshot":
            raise ValueError("MOMENT external source_type 不一致")
        if str(external.get("repository") or "") != MOMENT_REPOSITORY:
            raise ValueError("MOMENT external repository 不一致")
        if str(external.get("requested_revision") or "") != MOMENT_REVISION:
            raise ValueError("MOMENT external requested_revision 不一致")
        if str(external.get("resolved_revision") or "") != MOMENT_REVISION:
            raise ValueError("MOMENT external resolved_revision 不一致")
        if str(external.get("model_architecture") or "") != model_spec.architecture:
            raise ValueError("MOMENT external model_architecture 不一致")
        if external.get("model_spec") != model_spec.as_manifest_payload():
            raise ValueError("MOMENT external model_spec 不一致")
        checkpoint = _require_mapping(external, "checkpoint")
        if (
            str(checkpoint.get("filename") or "") != MOMENT_CHECKPOINT_FILENAME
            or str(checkpoint.get("sha256") or "") != MOMENT_CHECKPOINT_SHA256
            or int(checkpoint.get("size_bytes", 0)) != MOMENT_CHECKPOINT_SIZE_BYTES
        ):
            raise ValueError("MOMENT external checkpoint record 不一致")
        config_record = _require_mapping(external, "config")
        config_sha256 = str(config_record.get("sha256") or "")
        if (
            str(config_record.get("filename") or "") != MOMENT_CONFIG_FILENAME
            or int(config_record.get("size_bytes", 0)) != MOMENT_CONFIG_SIZE_BYTES
            or str(config_record.get("semantic_validation") or "") != "exact_pinned_config"
            or len(config_sha256) != 64
            or any(character not in "0123456789abcdef" for character in config_sha256)
        ):
            raise ValueError("MOMENT external config record 不合法")
        package = _require_mapping(external, "package")
        if (
            str(package.get("name") or "") != MOMENT_PACKAGE_NAME
            or str(package.get("version") or "") != MOMENT_PACKAGE_VERSION
            or str(package.get("expected_version") or "") != MOMENT_PACKAGE_VERSION
        ):
            raise ValueError("MOMENT external package record 不一致")
        runtime_dependencies = _require_mapping(external, "runtime_dependencies")
        transformers_record = _require_mapping(
            runtime_dependencies,
            MOMENT_TRANSFORMERS_PACKAGE_NAME,
        )
        if (
            str(transformers_record.get("version") or "")
            != MOMENT_TRANSFORMERS_VERSION
            or str(transformers_record.get("expected_version") or "")
            != MOMENT_TRANSFORMERS_VERSION
        ):
            raise ValueError("MOMENT transformers runtime dependency record 不一致")
        if external.get("encoder_frozen_downstream") is not True:
            raise ValueError("MOMENT frozen encoder contract 不一致")
        required_false_fields = (
            "project_selection_windows_used_for_encoder_training",
            "project_oos_windows_used_for_encoder_training",
            "project_pass_reject_labels_used_for_encoder_training",
            "project_encoder_fine_tuning_used",
        )
        if not all(external.get(field_name) is False for field_name in required_false_fields):
            raise ValueError("MOMENT project data isolation contract 不一致")

    if model_spec.family == "mantis_v2_frozen_linear":
        if frozen_parameter_count < 1:
            raise ValueError("MantisV2 frozen probe 必須包含 frozen encoder parameters")
        if manifest.get("self_supervised_pretraining") is not None:
            raise ValueError("MantisV2 不得混用 Selection-only self-supervised pretraining")
        external = _require_mapping(manifest, "external_pretrained_encoder")
        if str(external.get("source_type") or "") != "hugging_face_snapshot":
            raise ValueError("MantisV2 external source_type 不一致")
        if str(external.get("repository") or "") != MANTIS_V2_REPOSITORY:
            raise ValueError("MantisV2 external repository 不一致")
        if str(external.get("requested_revision") or "") != MANTIS_V2_REVISION:
            raise ValueError("MantisV2 external requested_revision 不一致")
        if str(external.get("resolved_revision") or "") != MANTIS_V2_REVISION:
            raise ValueError("MantisV2 external resolved_revision 不一致")
        if str(external.get("model_architecture") or "") != model_spec.architecture:
            raise ValueError("MantisV2 external model_architecture 不一致")
        if external.get("model_spec") != model_spec.as_manifest_payload():
            raise ValueError("MantisV2 external model_spec 不一致")
        checkpoint = _require_mapping(external, "checkpoint")
        if (
            str(checkpoint.get("filename") or "") != MANTIS_V2_CHECKPOINT_FILENAME
            or str(checkpoint.get("sha256") or "") != MANTIS_V2_CHECKPOINT_SHA256
            or int(checkpoint.get("size_bytes", 0)) < 1
        ):
            raise ValueError("MantisV2 external checkpoint record 不一致")
        config_record = _require_mapping(external, "config")
        if (
            str(config_record.get("filename") or "") != MANTIS_V2_CONFIG_FILENAME
            or str(config_record.get("sha256") or "") != MANTIS_V2_CONFIG_SHA256
            or int(config_record.get("size_bytes", 0)) < 1
        ):
            raise ValueError("MantisV2 external config record 不合法")
        package = _require_mapping(external, "package")
        if (
            str(package.get("name") or "") != MANTIS_PACKAGE_NAME
            or str(package.get("version") or "") != MANTIS_PACKAGE_VERSION
            or str(package.get("expected_version") or "") != MANTIS_PACKAGE_VERSION
        ):
            raise ValueError("MantisV2 external package record 不一致")
        required_true_fields = (
            "encoder_frozen_downstream",
        )
        required_false_fields = (
            "project_selection_windows_used_for_encoder_training",
            "project_oos_windows_used_for_encoder_training",
            "project_pass_reject_labels_used_for_encoder_training",
            "project_encoder_fine_tuning_used",
        )
        if not all(external.get(field_name) is True for field_name in required_true_fields):
            raise ValueError("MantisV2 frozen encoder contract 不一致")
        if not all(external.get(field_name) is False for field_name in required_false_fields):
            raise ValueError("MantisV2 project data isolation contract 不一致")

    if model_spec.family == "ts2vec_frozen_linear":
        if frozen_parameter_count < 1:
            raise ValueError("TS2Vec frozen probe 必須包含 frozen encoder parameters")
        pretraining = _require_mapping(manifest, "self_supervised_pretraining")
        pretraining_manifest = _require_mapping(pretraining, "manifest")
        pretraining_dataset = _require_mapping(pretraining, "dataset_summary")
        if bool(pretraining_manifest.get("oos_windows_used")):
            raise ValueError("TS2Vec pretraining 不得使用 OOS windows")
        if bool(pretraining_manifest.get("pass_reject_labels_used")):
            raise ValueError("TS2Vec pretraining 不得使用 PASS/REJECT labels")
        if str(pretraining_manifest.get("model_architecture") or "") != model_spec.architecture:
            raise ValueError("TS2Vec pretraining architecture 不一致")
        if pretraining_manifest.get("model_spec") != model_spec.as_manifest_payload():
            raise ValueError("TS2Vec pretraining model_spec 不一致")
        expected_pretraining_profile = build_breakout_quality_pretraining_profile_payload(
            BREAKOUT_QUALITY_PRETRAINING_PROFILE
        )
        if pretraining_manifest.get("pretraining_profile") != expected_pretraining_profile:
            raise ValueError(
                "TS2Vec pretraining_profile 與 active named profile 不一致"
            )
        if str(pretraining_dataset.get("configuration_fingerprint") or "") != str(
            pretraining_manifest.get("pretraining_dataset_fingerprint") or ""
        ):
            raise ValueError("TS2Vec pretraining dataset fingerprint 不一致")
    model_policy = _require_mapping(manifest, "policy")
    if dict(model_policy) != DEFAULT_LABEL_POLICY.as_manifest_payload():
        raise ValueError(
            "breakout quality model policy 與目前 config/breakout_quality.py 不一致；"
            "請重新執行 workflow 以 relabel 並重訓模型"
        )
    if int(manifest.get("sequence_length", -1)) != int(DEFAULT_LABEL_POLICY.feature_window_bars):
        raise ValueError("breakout quality manifest sequence_length 與 feature window 不一致")

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
    training_mode = _require_nonempty_text(manifest, "training_mode")
    allowed_training_modes = {
        TRAINING_MODE_FIXED_EPOCH_FULL_SELECTION,
        TRAINING_MODE_INNER_VALIDATION_FULL_REFIT,
    }
    if training_mode not in allowed_training_modes:
        raise ValueError(
            f"breakout quality training_mode 不相容: {training_mode!r}"
        )
    fixed_epochs = int(manifest.get("fixed_epochs", 0))
    completed_epochs = int(manifest.get("completed_epochs", 0))
    selected_epoch = int(manifest.get("selected_epoch", fixed_epochs))
    max_epochs = int(manifest.get("max_epochs", fixed_epochs))
    if fixed_epochs < 1 or selected_epoch != fixed_epochs or max_epochs < selected_epoch:
        raise ValueError(
            "breakout quality epoch 契約不一致: "
            f"max={max_epochs}, selected={selected_epoch}, "
            f"fixed={fixed_epochs}, completed={completed_epochs}"
        )
    inner_validation_used = bool(manifest.get("inner_validation_used", False))
    early_stopping_enabled = bool(manifest.get("early_stopping_enabled", False))
    final_refit_plan = _require_mapping(manifest, "final_refit_plan")
    selection_record: dict[str, Any] | None = None
    plan_mode = _require_nonempty_text(final_refit_plan, "mode")
    if inner_validation_used and plan_mode != str(BREAKOUT_QUALITY_FINAL_REFIT_MODE):
        raise ValueError(
            "breakout quality final_refit_mode 與目前 config 不一致；請重新訓練: "
            f"manifest={plan_mode}, config={BREAKOUT_QUALITY_FINAL_REFIT_MODE}"
        )
    actual_steps = int(final_refit_plan.get("actual_optimizer_steps", 0))
    target_steps = int(final_refit_plan.get("target_optimizer_steps", 0))
    final_batches_per_epoch = int(final_refit_plan.get("final_refit_batches_per_epoch", 0))
    completed_cycles = int(final_refit_plan.get("completed_epoch_cycles", 0))
    if (
        actual_steps < 1
        or actual_steps != target_steps
        or final_batches_per_epoch < 1
        or completed_cycles < 1
        or completed_cycles != completed_epochs
        or not bool(
            final_refit_plan.get(
                "all_eligible_selection_groups_seen_at_least_once",
                final_refit_plan.get(
                    "all_eligible_selection_rows_seen_at_least_once", False
                ),
            )
        )
        or actual_steps < final_batches_per_epoch
    ):
        raise ValueError(
            "breakout quality final refit steps 契約不一致: "
            f"actual={actual_steps}, target={target_steps}, "
            f"batches_per_epoch={final_batches_per_epoch}, cycles={completed_cycles}, "
            f"completed_epochs={completed_epochs}"
        )
    if training_mode == TRAINING_MODE_FIXED_EPOCH_FULL_SELECTION:
        if inner_validation_used or early_stopping_enabled:
            raise ValueError(
                "fixed_epoch_full_selection 不可使用 inner validation / early stopping"
            )
        if max_epochs != fixed_epochs or completed_epochs != fixed_epochs:
            raise ValueError(
                "fixed_epoch_full_selection 的 max/fixed/completed epochs 必須一致"
            )
        if str(manifest.get("epoch_selection_source") or "fixed_cli_epochs") != "fixed_cli_epochs":
            raise ValueError("fixed_epoch_full_selection 的 epoch_selection_source 不一致")
        if plan_mode != "selected_epochs":
            raise ValueError("fixed_epoch_full_selection 必須使用 selected_epochs refit mode")
        if actual_steps != fixed_epochs * final_batches_per_epoch:
            raise ValueError("fixed_epoch_full_selection optimizer steps 與 epochs 不一致")
    else:
        if not inner_validation_used:
            raise ValueError("inner_validation full refit 模式必須啟用 inner validation")
        if int(manifest.get("inner_validation_months", 0)) < 1:
            raise ValueError("inner_validation_months 必須 >=1")
        if str(manifest.get("epoch_selection_source") or "") != "inner_validation_loss":
            raise ValueError("inner validation 模式必須以 validation loss 選 epoch")
        selection_record = manifest.get("inner_validation_epoch_selection")
        if not isinstance(selection_record, dict):
            raise ValueError("inner validation 模式缺少 epoch selection 紀錄")
        if int(selection_record.get("best_epoch", 0)) != selected_epoch:
            raise ValueError("inner validation best_epoch 與 selected_epoch 不一致")
        selection_completed_epochs = int(selection_record.get("completed_epochs", 0))
        if not selected_epoch <= selection_completed_epochs <= max_epochs:
            raise ValueError(
                "inner validation completed_epochs 契約不一致: "
                f"selected={selected_epoch}, completed={selection_completed_epochs}, "
                f"max={max_epochs}"
            )
        selected_steps = int(selection_record.get("best_optimizer_steps", 0))
        plan_selected_steps = int(final_refit_plan.get("selected_optimizer_steps", 0))
        if selected_steps < 1 or plan_selected_steps != selected_steps:
            raise ValueError("inner validation selected optimizer steps 與 final refit plan 不一致")
        if plan_mode == "matched_optimizer_steps":
            expected_target = max(selected_steps, final_batches_per_epoch)
            if target_steps != expected_target:
                raise ValueError(
                    "matched_optimizer_steps target 不一致: "
                    f"actual={target_steps}, expected={expected_target}"
                )
        elif plan_mode == "selected_epochs":
            if target_steps != selected_epoch * final_batches_per_epoch:
                raise ValueError("selected_epochs final refit steps 與 selected_epoch 不一致")
        else:
            raise ValueError(f"不支援的 final_refit_mode: {plan_mode}")
        patience = int(manifest.get("early_stopping_patience", -1))
        min_delta = float(manifest.get("early_stopping_min_delta", -1.0))
        if patience < 0 or min_delta < 0.0:
            raise ValueError("inner validation early stopping 參數必須 >=0")
        if early_stopping_enabled != (patience > 0):
            raise ValueError(
                "early_stopping_enabled 必須與 early_stopping_patience 是否大於 0 一致"
            )

    expected_sampling_mode = expected_experiment.training_sampling_mode
    sampling_record = manifest.get("training_sampling")
    legacy_sampling_record = (
        sampling_record is None
        and expected_sampling_mode == TRAINING_SAMPLING_ALL_EVENT_ROWS
    )
    training_uses_all_rows = bool(
        manifest.get("training_uses_all_eligible_selection_rows", False)
    )
    training_uses_all_groups = bool(
        manifest.get(
            "training_uses_all_eligible_selection_groups",
            training_uses_all_rows,
        )
    )
    if not training_uses_all_groups:
        raise ValueError("breakout quality 最終模型必須使用全部 eligible Selection groups")

    if not legacy_sampling_record:
        if not isinstance(sampling_record, dict):
            raise ValueError("breakout quality training_sampling 必須是 object")
        if _require_nonempty_text(sampling_record, "mode") != expected_sampling_mode:
            raise ValueError(
                "breakout quality training sampling mode 與 experiment profile 不一致"
            )
        if bool(sampling_record.get("validation_sampling_enabled", False)):
            raise ValueError("breakout quality Validation 不可套用 training sampling")
        if bool(sampling_record.get("oos_sampling_enabled", False)):
            raise ValueError("breakout quality OOS 不可套用 training sampling")

        sampling_phases: dict[str, dict[str, Any]] = {}
        for phase_name in ("inner_train", "final_refit"):
            raw_phase = sampling_record.get(phase_name)
            if not isinstance(raw_phase, dict):
                raise ValueError(
                    f"breakout quality training_sampling.{phase_name} 必須是 object"
                )
            if _require_nonempty_text(raw_phase, "mode") != expected_sampling_mode:
                raise ValueError(
                    f"breakout quality training_sampling.{phase_name}.mode 不一致"
                )
            source_rows = int(raw_phase.get("source_row_count", 0))
            sampled_rows = int(raw_phase.get("sampled_row_count", 0))
            group_count = int(raw_phase.get("unique_group_count", 0))
            removed_rows = int(raw_phase.get("duplicate_rows_removed", -1))
            if (
                source_rows < 1
                or sampled_rows < 1
                or group_count < 1
                or sampled_rows > source_rows
                or removed_rows != source_rows - sampled_rows
                or not bool(raw_phase.get("uses_all_eligible_groups", False))
            ):
                raise ValueError(
                    f"breakout quality training_sampling.{phase_name} 計數／group 契約不一致"
                )
            if expected_sampling_mode == TRAINING_SAMPLING_ALL_EVENT_ROWS:
                if (
                    sampled_rows != source_rows
                    or removed_rows != 0
                    or not bool(raw_phase.get("uses_all_eligible_rows", False))
                    or str(raw_phase.get("sampling_unit") or "") != "event_row"
                    or str(raw_phase.get("batch_size_unit") or "") != "event_rows"
                ):
                    raise ValueError(
                        f"breakout quality training_sampling.{phase_name} baseline rows 契約不一致"
                    )
            elif expected_sampling_mode == TRAINING_SAMPLING_UNIQUE_TICKER_DATE:
                if (
                    bool(model_spec.use_dataset_context)
                    or bool(model_spec.derived_context_features)
                    or sampled_rows != group_count
                    or str(raw_phase.get("sampling_unit") or "")
                    != "unique_ticker_date_group"
                    or str(raw_phase.get("batch_size_unit") or "")
                    != "unique_ticker_date_groups"
                    or str(raw_phase.get("representative_rule") or "")
                    != "minimum_original_event_row_index"
                ):
                    raise ValueError(
                        f"breakout quality training_sampling.{phase_name} unique-group 契約不一致"
                    )
            else:
                raise ValueError(
                    "不支援的 breakout quality training sampling mode: "
                    f"{expected_sampling_mode}"
                )
            sampling_phases[phase_name] = raw_phase

        if (
            _require_nonempty_text(final_refit_plan, "training_sampling_mode")
            != expected_sampling_mode
        ):
            raise ValueError("final_refit_plan training_sampling_mode 不一致")
        plan_weight_reduction = str(
            final_refit_plan.get("training_weight_reduction")
            or TRAINING_WEIGHT_REDUCTION_BATCH_WEIGHT_SUM
        ).strip().lower()
        if plan_weight_reduction != training_weight_reduction:
            raise ValueError("final_refit_plan training_weight_reduction 不一致")
        if int(final_refit_plan.get("inner_train_sampling_row_count", 0)) != int(
            sampling_phases["inner_train"].get("sampled_row_count", -1)
        ):
            raise ValueError("inner_train sampling row count 與 final_refit_plan 不一致")
        if int(final_refit_plan.get("final_refit_sampling_row_count", 0)) != int(
            sampling_phases["final_refit"].get("sampled_row_count", -1)
        ):
            raise ValueError("final_refit sampling row count 與 final_refit_plan 不一致")
        if str(final_refit_plan.get("batch_size_unit") or "") != str(
            sampling_phases["final_refit"].get("batch_size_unit") or ""
        ):
            raise ValueError("final_refit batch size unit 與 training sampling 不一致")
        if training_uses_all_rows != bool(
            sampling_phases["final_refit"].get("uses_all_eligible_rows", False)
        ):
            raise ValueError(
                "training_uses_all_eligible_selection_rows 與 sampling 不一致"
            )
    elif not training_uses_all_rows:
        raise ValueError("legacy baseline manifest 必須使用全部 eligible Selection rows")

    optimizer_name = str(manifest.get("optimizer_name") or "adam").strip().lower()
    if optimizer_name != expected_experiment.optimizer_name:
        raise ValueError(
            "breakout quality optimizer_name 與 experiment profile 不一致；請重新訓練: "
            f"manifest={optimizer_name}, profile={manifest_profile}, "
            f"expected={expected_experiment.optimizer_name}"
        )

    schedule_name = str(manifest.get("lr_schedule_name") or "none").strip().lower()
    if schedule_name != expected_experiment.lr_schedule_name:
        raise ValueError(
            "breakout quality lr_schedule_name 與 experiment profile 不一致；請重新訓練: "
            f"manifest={schedule_name}, profile={manifest_profile}, "
            f"expected={expected_experiment.lr_schedule_name}"
        )
    schedule_parameters = expected_experiment.lr_schedule_parameters()
    learning_rate = float(manifest.get("learning_rate", 0.0))
    schedule_record = manifest.get("learning_rate_schedule")
    if schedule_record is not None:
        if not isinstance(schedule_record, dict):
            raise ValueError("breakout quality learning_rate_schedule 必須是 object")
        if str(schedule_record.get("name") or "").strip().lower() != schedule_name:
            raise ValueError("breakout quality learning_rate_schedule.name 不一致")
        if dict(schedule_record.get("parameters") or {}) != schedule_parameters:
            raise ValueError("breakout quality learning_rate_schedule.parameters 不一致")
        validate_learning_rate_schedule_record(
            _require_mapping(schedule_record, "final_refit"),
            schedule_name=schedule_name,
            schedule_parameters=schedule_parameters,
            base_learning_rate=learning_rate,
            expected_total_optimizer_steps=target_steps,
            expected_actual_optimizer_steps=actual_steps,
        )
        epoch_schedule = schedule_record.get("epoch_selection")
        if inner_validation_used:
            if selection_record is None:
                raise ValueError("inner validation 缺少 epoch selection record")
            selection_batches_per_epoch = int(
                selection_record.get("batches_per_epoch", 0)
            )
            selection_completed_epochs = int(
                selection_record.get("completed_epochs", 0)
            )
            if selection_batches_per_epoch < 1:
                raise ValueError("inner validation batches_per_epoch 必須 >=1")
            validate_learning_rate_schedule_record(
                epoch_schedule,
                schedule_name=schedule_name,
                schedule_parameters=schedule_parameters,
                base_learning_rate=learning_rate,
                expected_total_optimizer_steps=max_epochs * selection_batches_per_epoch,
                expected_actual_optimizer_steps=(
                    selection_completed_epochs * selection_batches_per_epoch
                ),
            )
        elif epoch_schedule is not None:
            raise ValueError("未啟用 inner validation 時不可有 epoch_selection LR schedule")
    elif schedule_name != "none":
        raise ValueError("啟用 LR schedule 的 manifest 缺少 learning_rate_schedule")

    augmentation_name = str(
        manifest.get("augmentation_name")
        or expected_experiment.augmentation_name
    ).strip().lower()
    if augmentation_name != expected_experiment.augmentation_name:
        raise ValueError(
            "breakout quality augmentation_name 與 experiment profile 不一致；請重新訓練: "
            f"manifest={augmentation_name}, profile={manifest_profile}, "
            f"expected={expected_experiment.augmentation_name}"
        )
    augmentation_parameters = expected_experiment.augmentation_parameters()
    augmentation_record = manifest.get("training_augmentation")
    if augmentation_record is None:
        if augmentation_name != "none":
            raise ValueError("啟用 augmentation 的 manifest 缺少 training_augmentation")
    else:
        if not isinstance(augmentation_record, dict):
            raise ValueError("breakout quality training_augmentation 必須是 object")
        if str(augmentation_record.get("name") or "").strip().lower() != augmentation_name:
            raise ValueError("breakout quality training_augmentation.name 不一致")
        if dict(augmentation_record.get("parameters") or {}) != augmentation_parameters:
            raise ValueError("breakout quality training_augmentation.parameters 不一致")
        if bool(augmentation_record.get("validation_augmented", True)):
            raise ValueError("breakout quality validation 不可套用 training augmentation")
        if bool(augmentation_record.get("oos_augmented", True)):
            raise ValueError("breakout quality OOS 不可套用 training augmentation")
        final_refit_augmentation = _validate_training_augmentation_epoch_records(
            augmentation_record.get("final_refit"),
            augmentation_name=augmentation_name,
            augmentation_parameters=augmentation_parameters,
            field_name="training_augmentation.final_refit",
        )
        if len(final_refit_augmentation) != int(manifest.get("completed_epochs", 0)):
            raise ValueError("breakout quality final refit augmentation epoch 數不一致")
        epoch_selection_augmentation = augmentation_record.get("epoch_selection")
        if inner_validation_used:
            validated_epoch_selection = _validate_training_augmentation_epoch_records(
                epoch_selection_augmentation,
                augmentation_name=augmentation_name,
                augmentation_parameters=augmentation_parameters,
                field_name="training_augmentation.epoch_selection",
            )
            expected_epoch_count = int(
                _require_mapping(
                    manifest,
                    "inner_validation_epoch_selection",
                ).get("completed_epochs", 0)
            )
            if len(validated_epoch_selection) != expected_epoch_count:
                raise ValueError("breakout quality epoch selection augmentation epoch 數不一致")
        elif epoch_selection_augmentation is not None:
            raise ValueError("未啟用 inner validation 時不可有 epoch selection augmentation")

    class_weight_mode = _require_nonempty_text(manifest, "class_weight_mode")
    if class_weight_mode != str(BREAKOUT_QUALITY_CLASS_WEIGHT_MODE):
        raise ValueError(
            "breakout quality class_weight_mode 與目前 config 不一致；請重新訓練"
        )
    class_weights = manifest.get("class_weights_reject_pass")
    if not isinstance(class_weights, list) or len(class_weights) != 2:
        raise ValueError("class_weights_reject_pass 必須包含 REJECT/PASS 兩個值")
    if class_weight_mode == "none" and [float(value) for value in class_weights] != [1.0, 1.0]:
        raise ValueError("class_weight_mode=none 時 class weights 必須為 [1, 1]")

    time_weight_mode = _require_nonempty_text(manifest, "time_weight_mode")
    if (
        manifest_profile == str(BREAKOUT_QUALITY_EXPERIMENT_PROFILE)
        and time_weight_mode != str(BREAKOUT_QUALITY_TIME_WEIGHT_MODE)
    ):
        raise ValueError(
            "active breakout quality time_weight_mode 與目前 config 不一致；請重新訓練"
        )
    if (
        expected_experiment.time_weight_mode is not None
        and time_weight_mode != expected_experiment.time_weight_mode
    ):
        raise ValueError(
            "breakout quality time_weight_mode 與 experiment profile 不一致"
        )
    sample_weight_summaries = _require_mapping(manifest, "sample_weight_summaries")
    final_refit_summary = _require_mapping(sample_weight_summaries, "final_refit")
    if _require_nonempty_text(final_refit_summary, "mode") != time_weight_mode:
        raise ValueError("final refit sample weight mode 與 manifest 不一致")
    if time_weight_mode == TIME_WEIGHT_MODE_DATE_BALANCED:
        if training_weight_reduction != TRAINING_WEIGHT_REDUCTION_FIXED_BATCH_SIZE:
            raise ValueError("date_balanced 必須使用 fixed_batch_size reduction")
        group_count = int(final_refit_summary.get("group_count", 0))
        date_count = int(final_refit_summary.get("date_count", 0))
        weight_sum = float(final_refit_summary.get("weight_sum", 0.0))
        target = float(final_refit_summary.get("target_total_weight_per_date", 0.0))
        actual_min = float(
            final_refit_summary.get("actual_total_weight_per_date_min", 0.0)
        )
        actual_max = float(
            final_refit_summary.get("actual_total_weight_per_date_max", 0.0)
        )
        tolerance = max(1e-5, abs(target) * 1e-5)
        if (
            group_count < 1
            or date_count < 1
            or abs(weight_sum - float(group_count))
            > max(1e-3, float(group_count) * 1e-6)
            or target <= 0.0
            or abs(actual_min - target) > tolerance
            or abs(actual_max - target) > tolerance
        ):
            raise ValueError("date_balanced sample weight summary 契約不一致")

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
        raise ValueError("breakout quality 不可由 train.py 最佳化 threshold")
    if bool(threshold_policy.get("oos_tuning_allowed", True)):
        raise ValueError("breakout quality 禁止使用 OOS 調整 threshold")

    return BreakoutQualityModelContract(paths=paths, manifest=manifest)


@lru_cache(maxsize=16)
def load_split_assignment_frame(
    project_root: str,
    filter_id: str,
    model_architecture: str | None = None,
    experiment_profile: str | None = None,
):
    model_contract = load_model_artifact_contract(
        project_root, filter_id, model_architecture, experiment_profile
    )
    return _validate_split_assignment_record(model_contract.paths, model_contract.manifest)


@lru_cache(maxsize=16)
def load_runtime_artifact_contract(
    project_root: str,
    filter_id: str,
    model_architecture: str | None = None,
    experiment_profile: str | None = None,
) -> BreakoutQualityRuntimeContract:
    model_contract = load_model_artifact_contract(
        project_root, filter_id, model_architecture, experiment_profile
    )
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
    required_signal_start = available_from
    execution_start = available_from
    if scope == RUNTIME_SCOPE_FORWARD_OOS:
        information_cutoff = _parse_iso_date(
            runtime_eligibility.get("model_information_cutoff"),
            field_name="runtime_eligibility.model_information_cutoff",
        )
        required_signal_start = _parse_iso_date(
            runtime_eligibility.get("required_signal_start"),
            field_name="runtime_eligibility.required_signal_start",
        )
        execution_start = _parse_iso_date(
            runtime_eligibility.get("execution_start"),
            field_name="runtime_eligibility.execution_start",
        )
        # (AI註: model_information_cutoff 是訓練資訊最後可用的交易日。模型可於該日
        # 收盤後完成，並以該日收盤訊號建立下一交易日盤前訂單；因此同日 signal
        # anchor 合法，只有早於 cutoff 的事件才不可進入正式 forward-OOS score。)
        if required_signal_start != information_cutoff:
            raise ValueError(
                "breakout quality required_signal_start 必須等於 model_information_cutoff，"
                "供下一交易日盤前決策使用 cutoff 當日收盤訊號"
            )
        if available_from < required_signal_start:
            raise ValueError(
                "breakout quality forward_oos score table 不得包含 model_information_cutoff 之前事件"
            )
        outer_policy = _require_mapping(manifest, "outer_oos_policy")
        outer_execution_start = _parse_iso_date(
            outer_policy.get("oos_start_date"),
            field_name="outer_oos_policy.oos_start_date",
        )
        if execution_start != outer_execution_start:
            raise ValueError(
                "breakout quality runtime execution_start 必須等於 outer_oos_policy.oos_start_date"
            )
        if required_signal_start > execution_start:
            raise ValueError("breakout quality required_signal_start 不可晚於 execution_start")
        if available_through < execution_start:
            raise ValueError("breakout quality available_through 不可早於 execution_start")

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
    runtime_source = score_table.get("runtime_source")
    if runtime_source is not None:
        if not isinstance(runtime_source, dict):
            raise ValueError("breakout quality score_table.runtime_source 必須是 object")
        dataset_profile = _require_nonempty_text(runtime_source, "dataset_profile")
        if dataset_profile not in {"full", "reduced"}:
            raise ValueError("breakout quality runtime_source.dataset_profile 不合法")
        stored_source_inventory = _require_mapping(runtime_source, "source_data_inventory")
        current_source_inventory = build_source_data_inventory(project_root, dataset_profile)
        if stored_source_inventory != current_source_inventory:
            raise ValueError(
                "breakout quality 正式 score table 的來源 CSV 已變更；"
                "請重新執行 export-scores --scope forward_oos"
            )
        runtime_source_range = _require_mapping(runtime_source, "source_data_date_range")
        runtime_source_start = _parse_iso_date(
            runtime_source_range.get("start"),
            field_name="score_table.runtime_source.source_data_date_range.start",
        )
        runtime_source_end = _parse_iso_date(
            runtime_source_range.get("end"),
            field_name="score_table.runtime_source.source_data_date_range.end",
        )
        if runtime_source_end < runtime_source_start:
            raise ValueError("breakout quality runtime source date range 不合法")
        if runtime_source_end != available_through:
            raise ValueError("breakout quality runtime source end 與 available_through 不一致")
        if int(runtime_source.get("runtime_ticker_limit", -1)) != 0:
            raise ValueError("breakout quality 正式 runtime source 不得沿用 training ticker limit")
        processed_ticker_count = int(runtime_source.get("processed_ticker_count", -1))
        skipped_tickers = runtime_source.get("skipped_tickers")
        if processed_ticker_count < 1 or not isinstance(skipped_tickers, list):
            raise ValueError("breakout quality runtime source ticker coverage metadata 不合法")
        candidate_event_count = int(runtime_source.get("candidate_event_row_count", -1))
        runtime_model_scored_count = int(runtime_source.get("model_scored_event_row_count", -1))
        runtime_conservative_count = int(runtime_source.get("conservative_reject_event_row_count", -1))
        if (
            candidate_event_count != row_count
            or runtime_model_scored_count < 0
            or runtime_conservative_count < 0
            or runtime_model_scored_count + runtime_conservative_count != candidate_event_count
        ):
            raise ValueError("breakout quality runtime source candidate coverage counts 不一致")

    unavailable_row_count = 0
    unavailable_record = manifest.get("conservative_unscorable_events")
    if unavailable_record is not None:
        if not isinstance(unavailable_record, dict):
            raise ValueError("breakout quality conservative_unscorable_events 必須是 object")
        unavailable_path = paths.score_path.with_name(DEFAULT_UNAVAILABLE_SCORE_FILENAME)
        _validate_file_record(
            unavailable_path,
            unavailable_record,
            expected_filename=DEFAULT_UNAVAILABLE_SCORE_FILENAME,
            field_name="conservative_unscorable_events",
        )
        if list(unavailable_record.get("columns", [])) != ["ticker", "date", "high_len", "reason"]:
            raise ValueError("breakout quality unavailable score columns 契約不一致")
        unavailable_row_count = int(unavailable_record.get("row_count", -1))
        if unavailable_row_count < 0:
            raise ValueError("breakout quality unavailable score row_count 必須 >= 0")
        reason_counts = unavailable_record.get("reason_counts")
        if not isinstance(reason_counts, dict) or sum(int(value) for value in reason_counts.values()) != unavailable_row_count:
            raise ValueError("breakout quality unavailable score reason_counts 與 row_count 不一致")
        if float(unavailable_record.get("conservative_runtime_score", float("nan"))) != 0.0:
            raise ValueError("breakout quality unavailable score 必須保守映射為 0.0")
        if str(unavailable_record.get("runtime_action") or "").strip() != "reject":
            raise ValueError("breakout quality unavailable score runtime_action 必須是 reject")
        if isinstance(runtime_source, dict) and int(
            runtime_source.get("conservative_reject_event_row_count", -1)
        ) != unavailable_row_count:
            raise ValueError("breakout quality runtime source 與 unavailable audit row_count 不一致")

    score_inference_execution = manifest.get("score_inference_execution")
    if score_inference_execution is not None and not isinstance(score_inference_execution, dict):
        raise ValueError("breakout quality score_inference_execution 必須是 object")
    if isinstance(score_inference_execution, dict) and (
        "model_scored_event_row_count" in score_inference_execution
        or "conservative_reject_event_row_count" in score_inference_execution
    ):
        model_scored_count = int(score_inference_execution.get("model_scored_event_row_count", -1))
        conservative_count = int(score_inference_execution.get("conservative_reject_event_row_count", -1))
        output_count = int(score_inference_execution.get("output_event_row_count", -1))
        if (
            model_scored_count < 0
            or conservative_count != unavailable_row_count
            or output_count != row_count
            or model_scored_count + conservative_count != output_count
        ):
            raise ValueError("breakout quality score inference row counts 契約不一致")

    shared_group_score_broadcast = False
    if score_inference_execution is not None:
        shared_raw = score_inference_execution.get("shared_group_score_broadcast", False)
        if not isinstance(shared_raw, bool):
            raise ValueError("breakout quality shared_group_score_broadcast 必須是 bool")
        shared_group_score_broadcast = bool(shared_raw)
        if shared_group_score_broadcast:
            model_spec = model_spec_from_manifest(_require_mapping(manifest, "model_spec"))
            if bool(model_spec.use_dataset_context):
                raise ValueError(
                    "breakout quality shared group score 只適用於 use_dataset_context=false 模型"
                )
            if str(score_inference_execution.get("inference_unit") or "").strip() != (
                "unique_ticker_date_feature_group"
            ):
                raise ValueError(
                    "breakout quality shared group score 的 inference_unit 必須是 "
                    "unique_ticker_date_feature_group"
                )

    return BreakoutQualityRuntimeContract(
        paths=paths,
        manifest=manifest,
        high_len_values=high_len_values,
        available_from=available_from,
        available_through=available_through,
        required_signal_start=required_signal_start,
        execution_start=execution_start,
        shared_group_score_broadcast=shared_group_score_broadcast,
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
