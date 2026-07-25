"""Selection-only self-supervised pretraining dataset and encoder artifact contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from filters.breakout_quality.artifacts import compute_file_sha256
from filters.breakout_quality.contract import FEATURE_COLUMNS
from filters.breakout_quality.source_inventory import build_source_data_inventory

PRETRAINING_DATASET_SCHEMA_VERSION = 1
PRETRAINING_DATASET_FORMAT = "selection_rolling_windows_npy_v1"
TS2VEC_PRETRAINING_FAMILY = "ts2vec_v1"
PRETRAINING_WINDOWS_FILENAME = "windows.npy"
PRETRAINING_INDEX_FILENAME = "window_index.csv"
PRETRAINING_SUMMARY_FILENAME = "pretraining_dataset_summary.json"
PRETRAINED_ENCODER_FILENAME = "pretrained_encoder.pt"
PRETRAINED_ENCODER_MANIFEST_FILENAME = "pretrained_encoder_manifest.json"


@dataclass(frozen=True)
class BreakoutQualityPretrainingDatasetPaths:
    output_dir: Path
    windows: Path
    index: Path
    summary: Path


@dataclass(frozen=True)
class BreakoutQualityPretrainedEncoderPaths:
    output_dir: Path
    encoder: Path
    manifest: Path


def _safe_component(value: str, *, field_name: str) -> str:
    text = str(value).strip()
    if not text or text in {".", ".."} or Path(text).name != text:
        raise ValueError(f"{field_name} 必須是安全的單一資料夾名稱")
    return text


def resolve_pretraining_dataset_paths(
    project_root: str | Path,
    filter_id: str,
    *,
    family: str = TS2VEC_PRETRAINING_FAMILY,
    stride: int,
) -> BreakoutQualityPretrainingDatasetPaths:
    resolved_filter = _safe_component(filter_id, field_name="filter_id")
    resolved_family = _safe_component(family, field_name="pretraining family")
    resolved_stride = int(stride)
    if resolved_stride < 1:
        raise ValueError("pretraining stride 必須 >= 1")
    output_dir = (
        Path(project_root)
        / "outputs"
        / "filters"
        / "breakout_quality"
        / resolved_filter
        / "pretraining"
        / resolved_family
        / f"stride_{resolved_stride}"
    )
    return BreakoutQualityPretrainingDatasetPaths(
        output_dir=output_dir,
        windows=output_dir / PRETRAINING_WINDOWS_FILENAME,
        index=output_dir / PRETRAINING_INDEX_FILENAME,
        summary=output_dir / PRETRAINING_SUMMARY_FILENAME,
    )


def resolve_pretrained_encoder_paths(
    project_root: str | Path,
    filter_id: str,
    *,
    model_architecture: str,
    experiment_profile: str,
) -> BreakoutQualityPretrainedEncoderPaths:
    resolved_filter = _safe_component(filter_id, field_name="filter_id")
    resolved_architecture = _safe_component(
        model_architecture,
        field_name="model_architecture",
    )
    resolved_profile = _safe_component(experiment_profile, field_name="experiment_profile")
    output_dir = (
        Path(project_root)
        / "models"
        / "filters"
        / "breakout_quality"
        / resolved_filter
        / resolved_architecture
        / resolved_profile
        / "pretraining"
    )
    return BreakoutQualityPretrainedEncoderPaths(
        output_dir=output_dir,
        encoder=output_dir / PRETRAINED_ENCODER_FILENAME,
        manifest=output_dir / PRETRAINED_ENCODER_MANIFEST_FILENAME,
    )


def build_file_record(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    return {
        "filename": resolved.name,
        "size_bytes": int(resolved.stat().st_size),
        "sha256": compute_file_sha256(resolved),
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"無法讀取 pretraining JSON: {path}; {type(exc).__name__}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"pretraining JSON 根節點必須是 object: {path}")
    return payload


def _validate_file(path: Path, record: object, *, field_name: str) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"pretraining {field_name} record 必須是 object")
    if not path.is_file():
        raise FileNotFoundError(f"pretraining 工件不存在: {path}")
    if str(record.get("filename") or "").strip() != path.name:
        raise ValueError(f"pretraining {field_name} filename 不一致")
    if int(record.get("size_bytes", -1)) != int(path.stat().st_size):
        raise ValueError(f"pretraining {field_name} size 不一致")
    expected_hash = str(record.get("sha256") or "").strip().lower()
    actual_hash = compute_file_sha256(path).lower()
    if not expected_hash or expected_hash != actual_hash:
        raise ValueError(f"pretraining {field_name} SHA256 不一致")


def compute_pretraining_configuration_fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_validated_pretraining_dataset(
    project_root: str | Path,
    filter_id: str,
    *,
    dataset_profile: str,
    family: str,
    stride: int,
    expected_selection_start: str,
    expected_selection_end: str,
    expected_window_bars: int,
    expected_max_tickers: int,
    require_current_source: bool,
) -> tuple[dict[str, Any], np.ndarray, pd.DataFrame]:
    paths = resolve_pretraining_dataset_paths(
        project_root,
        filter_id,
        family=family,
        stride=stride,
    )
    summary = _read_json_object(paths.summary)
    if int(summary.get("schema_version", -1)) != PRETRAINING_DATASET_SCHEMA_VERSION:
        raise ValueError("pretraining dataset schema version 不一致")
    if str(summary.get("format") or "") != PRETRAINING_DATASET_FORMAT:
        raise ValueError("pretraining dataset format 不一致")
    if str(summary.get("family") or "") != str(family):
        raise ValueError("pretraining dataset family 不一致")
    if str(summary.get("dataset_profile") or "") != str(dataset_profile):
        raise ValueError("pretraining dataset profile 不一致")
    if int(summary.get("stride", -1)) != int(stride):
        raise ValueError("pretraining dataset stride 不一致")
    if int(summary.get("window_bars", -1)) != int(expected_window_bars):
        raise ValueError("pretraining dataset window_bars 不一致")
    if int(summary.get("requested_max_tickers", -1)) != max(0, int(expected_max_tickers)):
        raise ValueError("pretraining dataset ticker coverage 不一致")
    if list(summary.get("feature_columns") or []) != list(FEATURE_COLUMNS):
        raise ValueError("pretraining dataset feature_columns 不一致")
    if str(summary.get("selection_start_date") or "") != str(expected_selection_start):
        raise ValueError("pretraining dataset selection_start_date 不一致")
    if str(summary.get("selection_end_date") or "") != str(expected_selection_end):
        raise ValueError("pretraining dataset selection_end_date 不一致")
    if summary.get("oos_windows_used") is not False:
        raise ValueError("pretraining dataset 必須明確記錄未使用 OOS windows")
    fingerprint_payload = {
        "family": str(summary.get("family") or ""),
        "dataset_profile": str(summary.get("dataset_profile") or ""),
        "stride": int(summary.get("stride", -1)),
        "window_bars": int(summary.get("window_bars", -1)),
        "feature_columns": list(summary.get("feature_columns") or []),
        "selection_start_date": str(summary.get("selection_start_date") or ""),
        "selection_end_date": str(summary.get("selection_end_date") or ""),
        "outer_policy_fingerprint": str(summary.get("outer_policy_fingerprint") or ""),
        "source_inventory_sha256": str(summary.get("source_inventory_sha256") or ""),
        "requested_max_tickers": int(summary.get("requested_max_tickers", -1)),
    }
    expected_fingerprint = compute_pretraining_configuration_fingerprint(
        fingerprint_payload
    )
    if str(summary.get("configuration_fingerprint") or "") != expected_fingerprint:
        raise ValueError("pretraining dataset configuration fingerprint 不一致")
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("pretraining dataset summary 缺少 artifacts")
    _validate_file(paths.windows, artifacts.get("windows"), field_name="windows")
    _validate_file(paths.index, artifacts.get("index"), field_name="index")

    windows = np.load(paths.windows, mmap_mode="r", allow_pickle=False)
    if windows.ndim != 3:
        raise ValueError(f"pretraining windows 必須是 3D: {windows.shape}")
    expected_shape = (
        int(summary.get("window_count", -1)),
        int(expected_window_bars),
        len(FEATURE_COLUMNS),
    )
    if tuple(int(v) for v in windows.shape) != expected_shape:
        raise ValueError(
            f"pretraining windows shape 不一致: expected={expected_shape}, actual={windows.shape}"
        )
    index = pd.read_csv(paths.index, dtype={"ticker": str})
    if list(index.columns) != ["window_index", "ticker", "date"]:
        raise ValueError("pretraining index columns 不一致")
    if len(index) != len(windows):
        raise ValueError("pretraining index row count 與 windows 不一致")
    if not np.array_equal(index["window_index"].to_numpy(dtype=np.int64), np.arange(len(index))):
        raise ValueError("pretraining index.window_index 必須連續且從 0 開始")
    dates = pd.to_datetime(index["date"], errors="raise")
    if dates.min().strftime("%Y-%m-%d") < str(expected_selection_start):
        raise ValueError("pretraining dataset 含 Selection 開始日前的 window endpoint")
    if dates.max().strftime("%Y-%m-%d") > str(expected_selection_end):
        raise ValueError("pretraining dataset 含 Selection 結束日後的 window endpoint")
    if require_current_source:
        current_inventory = build_source_data_inventory(project_root, dataset_profile)
        if summary.get("source_data_inventory") != current_inventory:
            raise ValueError("pretraining dataset 來源 CSV inventory 已變更")
    return summary, windows, index


def load_validated_pretrained_encoder_manifest(
    paths: BreakoutQualityPretrainedEncoderPaths,
    *,
    expected_architecture: str,
    expected_experiment_profile: str,
    expected_dataset_fingerprint: str,
    expected_model_spec: dict[str, Any],
    expected_pretraining_profile: dict[str, Any],
) -> dict[str, Any]:
    manifest = _read_json_object(paths.manifest)
    if int(manifest.get("schema_version", -1)) != 1:
        raise ValueError("pretrained encoder manifest schema version 不一致")
    if str(manifest.get("model_architecture") or "") != str(expected_architecture):
        raise ValueError("pretrained encoder architecture 不一致")
    if str(manifest.get("experiment_profile") or "") != str(expected_experiment_profile):
        raise ValueError("pretrained encoder experiment profile 不一致")
    if str(manifest.get("pretraining_dataset_fingerprint") or "") != str(expected_dataset_fingerprint):
        raise ValueError("pretrained encoder dataset fingerprint 不一致")
    if manifest.get("model_spec") != expected_model_spec:
        raise ValueError("pretrained encoder model_spec 不一致")
    if manifest.get("pretraining_profile") != expected_pretraining_profile:
        raise ValueError(
            "pretrained encoder pretraining_profile 與 active named profile 不一致"
        )
    _validate_file(paths.encoder, manifest.get("encoder"), field_name="encoder")
    if manifest.get("oos_windows_used") is not False:
        raise ValueError("pretrained encoder 必須明確記錄未使用 OOS windows")
    if manifest.get("pass_reject_labels_used") is not False:
        raise ValueError("pretrained encoder 必須明確記錄未使用 PASS/REJECT labels")
    return manifest


__all__ = [
    "BreakoutQualityPretrainedEncoderPaths",
    "BreakoutQualityPretrainingDatasetPaths",
    "PRETRAINED_ENCODER_FILENAME",
    "PRETRAINED_ENCODER_MANIFEST_FILENAME",
    "PRETRAINING_DATASET_FORMAT",
    "PRETRAINING_DATASET_SCHEMA_VERSION",
    "PRETRAINING_INDEX_FILENAME",
    "PRETRAINING_SUMMARY_FILENAME",
    "PRETRAINING_WINDOWS_FILENAME",
    "TS2VEC_PRETRAINING_FAMILY",
    "build_file_record",
    "compute_pretraining_configuration_fingerprint",
    "load_validated_pretrained_encoder_manifest",
    "load_validated_pretraining_dataset",
    "resolve_pretrained_encoder_paths",
    "resolve_pretraining_dataset_paths",
]
