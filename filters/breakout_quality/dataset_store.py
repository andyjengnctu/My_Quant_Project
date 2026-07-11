"""Canonical storage contract for breakout quality datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DATASET_STORAGE_SCHEMA_VERSION = 2
DATASET_STORAGE_FORMAT = "indexed_feature_bank_npy_v2"

FEATURE_BANK_FILENAME = "feature_bank.npy"
EVENT_CONTEXT_FILENAME = "event_context.npy"
EVENT_LABELS_FILENAME = "event_labels.npy"
EVENT_GROUP_INDEX_FILENAME = "event_group_index.npy"
GROUP_ANCHOR_PRICES_FILENAME = "group_anchor_prices.npy"
FUTURE_HIGH_PRICES_FILENAME = "future_high_prices.npy"
FUTURE_LOW_PRICES_FILENAME = "future_low_prices.npy"
FUTURE_AVAILABLE_BARS_FILENAME = "future_available_bars.npy"
FUTURE_DATE_ORDINALS_FILENAME = "future_date_ordinals.npy"
EVENTS_FILENAME = "events.csv"
SUMMARY_FILENAME = "dataset_summary.json"
LEGACY_DATASET_FILENAME = "dataset.npz"


@dataclass(frozen=True)
class BreakoutQualityDatasetPaths:
    output_dir: Path
    feature_bank: Path
    event_context: Path
    event_labels: Path
    event_group_index: Path
    group_anchor_prices: Path
    future_high_prices: Path
    future_low_prices: Path
    future_available_bars: Path
    future_date_ordinals: Path
    events: Path
    summary: Path
    legacy_dataset: Path

    def array_paths(self) -> dict[str, Path]:
        return {
            "feature_bank": self.feature_bank,
            "event_context": self.event_context,
            "event_labels": self.event_labels,
            "event_group_index": self.event_group_index,
            "group_anchor_prices": self.group_anchor_prices,
            "future_high_prices": self.future_high_prices,
            "future_low_prices": self.future_low_prices,
            "future_available_bars": self.future_available_bars,
            "future_date_ordinals": self.future_date_ordinals,
        }

    def artifact_paths(self) -> dict[str, Path]:
        return {**self.array_paths(), "events_csv": self.events}


class IndexedFeatureBank:
    """Event-row view over a de-duplicated ticker/date feature bank."""

    def __init__(self, feature_bank: np.ndarray, event_group_index: np.ndarray):
        if feature_bank.ndim != 3:
            raise ValueError(f"feature_bank 必須是 3D: {feature_bank.shape}")
        if event_group_index.ndim != 1:
            raise ValueError(f"event_group_index 必須是 1D: {event_group_index.shape}")
        self._feature_bank = feature_bank
        self._event_group_index = event_group_index

    @property
    def shape(self) -> tuple[int, int, int]:
        return (
            int(self._event_group_index.shape[0]),
            int(self._feature_bank.shape[1]),
            int(self._feature_bank.shape[2]),
        )

    @property
    def ndim(self) -> int:
        return 3

    @property
    def feature_bank(self) -> np.ndarray:
        return self._feature_bank

    @property
    def event_group_index(self) -> np.ndarray:
        return self._event_group_index

    def __len__(self) -> int:
        return int(self._event_group_index.shape[0])

    def __getitem__(self, item: Any) -> np.ndarray:
        group_indices = self._event_group_index[item]
        return np.asarray(self._feature_bank[group_indices], dtype=np.float32)


def resolve_dataset_paths(output_dir: str | Path) -> BreakoutQualityDatasetPaths:
    base = Path(output_dir)
    return BreakoutQualityDatasetPaths(
        output_dir=base,
        feature_bank=base / FEATURE_BANK_FILENAME,
        event_context=base / EVENT_CONTEXT_FILENAME,
        event_labels=base / EVENT_LABELS_FILENAME,
        event_group_index=base / EVENT_GROUP_INDEX_FILENAME,
        group_anchor_prices=base / GROUP_ANCHOR_PRICES_FILENAME,
        future_high_prices=base / FUTURE_HIGH_PRICES_FILENAME,
        future_low_prices=base / FUTURE_LOW_PRICES_FILENAME,
        future_available_bars=base / FUTURE_AVAILABLE_BARS_FILENAME,
        future_date_ordinals=base / FUTURE_DATE_ORDINALS_FILENAME,
        events=base / EVENTS_FILENAME,
        summary=base / SUMMARY_FILENAME,
        legacy_dataset=base / LEGACY_DATASET_FILENAME,
    )


def dataset_artifact_metadata_reasons(
    paths: BreakoutQualityDatasetPaths,
    artifact_records: object,
) -> list[str]:
    reasons: list[str] = []
    if not isinstance(artifact_records, dict):
        return ["dataset_summary 缺少 dataset_artifacts"]
    for artifact_name, artifact_path in paths.artifact_paths().items():
        record = artifact_records.get(artifact_name)
        if not isinstance(record, dict):
            reasons.append(f"dataset_artifacts 缺少 {artifact_name}")
            continue
        if str(record.get("filename", "")).strip() != artifact_path.name:
            reasons.append(f"{artifact_name} filename 與 summary 不一致")
            continue
        if not artifact_path.is_file():
            reasons.append(f"{artifact_name} 檔案不存在")
            continue
        try:
            expected_size = int(record.get("size_bytes", -1))
        except (TypeError, ValueError):
            expected_size = -1
        actual_size = int(artifact_path.stat().st_size)
        if expected_size != actual_size:
            reasons.append(
                f"{artifact_name} size 與 summary 不一致: expected={expected_size}, actual={actual_size}"
            )
    return reasons


def save_npy_atomic(path: str | Path, array: np.ndarray) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(destination.name + ".tmp")
    with temp_path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    temp_path.replace(destination)


def load_npy(path: str | Path, *, mmap_mode: str | None = "r") -> np.ndarray:
    return np.load(Path(path), mmap_mode=mmap_mode, allow_pickle=False)


__all__ = [
    "DATASET_STORAGE_FORMAT",
    "DATASET_STORAGE_SCHEMA_VERSION",
    "BreakoutQualityDatasetPaths",
    "EVENTS_FILENAME",
    "FEATURE_BANK_FILENAME",
    "IndexedFeatureBank",
    "LEGACY_DATASET_FILENAME",
    "SUMMARY_FILENAME",
    "dataset_artifact_metadata_reasons",
    "load_npy",
    "resolve_dataset_paths",
    "save_npy_atomic",
]
