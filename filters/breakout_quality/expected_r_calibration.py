"""Canonical read-only contract for frozen-ranker Expected-R calibration artifacts."""

from __future__ import annotations

from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from filters.breakout_quality.artifacts import compute_file_sha256

EXPECTED_R_CALIBRATION_SCHEMA_VERSION = 1
EXPECTED_R_CALIBRATION_METHOD = "daily_score_percentile_nonnegative_affine_v1"
EXPECTED_R_LOOKUP_FILENAME = "expected_r_lookup.csv.gz"
EXPECTED_R_MANIFEST_FILENAME = "manifest.json"
EXPECTED_R_REPORT_FILENAME = "report.md"
EXPECTED_R_REQUIRED_COLUMNS = (
    "ticker",
    "date",
    "group_index",
    "breakout_quality_score",
    "daily_score_percentile",
    "expected_r",
    "calibration_cutoff_exclusive",
)


def resolve_expected_r_calibration_dir(
    project_root: str | Path,
    *,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
    phase_id: str,
) -> Path:
    phase = str(phase_id).strip()
    if phase not in {"selection_pit", "forward_oos"}:
        raise ValueError(f"Expected-R calibration phase不支援: {phase!r}")
    return (
        Path(project_root).resolve()
        / "outputs"
        / "strategy_compare"
        / "runtime_artifacts"
        / "expected_r_calibration"
        / str(filter_id)
        / str(model_architecture)
        / str(experiment_profile)
        / phase
    ).resolve()


def resolve_expected_r_calibration_paths(
    project_root: str | Path,
    *,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
    phase_id: str,
) -> dict[str, Path]:
    root = resolve_expected_r_calibration_dir(
        project_root,
        filter_id=filter_id,
        model_architecture=model_architecture,
        experiment_profile=experiment_profile,
        phase_id=phase_id,
    )
    return {
        "dir": root,
        "lookup": root / EXPECTED_R_LOOKUP_FILENAME,
        "manifest": root / EXPECTED_R_MANIFEST_FILENAME,
        "report": root / EXPECTED_R_REPORT_FILENAME,
    }


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Expected-R manifest無法讀取: {path}; {type(exc).__name__}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Expected-R manifest根節點必須是object")
    return payload


def validate_expected_r_calibration_artifact(
    project_root: str | Path,
    *,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
    phase_id: str,
    expected_source_score_sha256: str | None = None,
    expected_fit_score_sha256: str | None = None,
    expected_selection_runtime_start_date: str | None = None,
    expected_forward_frozen_cutoff_exclusive: str | None = None,
) -> tuple[bool, str, dict[str, Any] | None]:
    paths = resolve_expected_r_calibration_paths(
        project_root,
        filter_id=filter_id,
        model_architecture=model_architecture,
        experiment_profile=experiment_profile,
        phase_id=phase_id,
    )
    if not paths["lookup"].is_file() or not paths["manifest"].is_file() or not paths["report"].is_file():
        return False, "MISSING", None
    try:
        manifest = _read_manifest(paths["manifest"])
        if int(manifest.get("schema_version", -1)) != EXPECTED_R_CALIBRATION_SCHEMA_VERSION:
            raise ValueError("schema_version不一致")
        if str(manifest.get("method") or "") != EXPECTED_R_CALIBRATION_METHOD:
            raise ValueError("calibration method不一致")
        identity = dict(manifest.get("identity") or {})
        expected_identity = {
            "filter_id": str(filter_id),
            "model_architecture": str(model_architecture),
            "experiment_profile": str(experiment_profile),
            "phase_id": str(phase_id),
        }
        if identity != expected_identity:
            raise ValueError(f"identity不一致: expected={expected_identity}, actual={identity}")
        lookup_record = dict(manifest.get("lookup_artifact") or {})
        actual_lookup_sha = compute_file_sha256(paths["lookup"]).lower()
        if str(lookup_record.get("sha256") or "").lower() != actual_lookup_sha:
            raise ValueError("lookup SHA256不一致")
        if int(lookup_record.get("size_bytes", -1)) != int(paths["lookup"].stat().st_size):
            raise ValueError("lookup size不一致")
        source = dict(manifest.get("source_artifacts") or {})
        if expected_source_score_sha256 and str(source.get("runtime_score_sha256") or "").lower() != str(expected_source_score_sha256).lower():
            raise ValueError("runtime score source SHA256已變更")
        if expected_fit_score_sha256 and str(source.get("fit_score_sha256") or "").lower() != str(expected_fit_score_sha256).lower():
            raise ValueError("fit score source SHA256已變更")
        fit_contract = dict(manifest.get("fit_contract") or {})
        if bool(fit_contract.get("forward_target_used_for_fit", True)):
            raise ValueError("Forward Expected-R calibration不得使用Forward target")
        if expected_selection_runtime_start_date is not None:
            expected_start = pd.Timestamp(expected_selection_runtime_start_date).normalize().strftime("%Y-%m-%d")
            actual_start = str(fit_contract.get("selection_runtime_start_date") or "")
            if actual_start != expected_start:
                raise ValueError(
                    "Selection Expected-R calibration start_date已變更: "
                    f"expected={expected_start}, actual={actual_start or '-'}"
                )
        if expected_forward_frozen_cutoff_exclusive is not None:
            expected_cutoff = pd.Timestamp(
                expected_forward_frozen_cutoff_exclusive
            ).normalize().strftime("%Y-%m-%d")
            actual_cutoff = str(fit_contract.get("forward_frozen_cutoff_exclusive") or "")
            if actual_cutoff != expected_cutoff:
                raise ValueError(
                    "Forward Expected-R calibration execution_start cutoff已變更: "
                    f"expected={expected_cutoff}, actual={actual_cutoff or '-'}"
                )
        return True, "READY", manifest
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return False, f"INVALID: {type(exc).__name__}: {exc}", None


def _file_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return int(stat.st_mtime_ns), int(stat.st_size)


@lru_cache(maxsize=8)
def _load_expected_r_lookup_cached(path_text: str, signature: tuple[int, int]) -> pd.DataFrame:
    del signature
    path = Path(path_text).resolve()
    table = pd.read_csv(path, low_memory=False).copy()
    missing = sorted(set(EXPECTED_R_REQUIRED_COLUMNS) - set(table.columns))
    if missing:
        raise ValueError(f"Expected-R lookup缺少欄位: {missing}")
    table["ticker"] = table["ticker"].fillna("").astype(str).str.strip()
    table["date"] = pd.to_datetime(table["date"], errors="raise").dt.strftime("%Y-%m-%d")
    table["group_index"] = pd.to_numeric(table["group_index"], errors="raise").astype(np.int64)
    for column in ("breakout_quality_score", "daily_score_percentile", "expected_r"):
        table[column] = pd.to_numeric(table[column], errors="raise").astype(float)
        if not np.isfinite(table[column].to_numpy(dtype=np.float64, copy=False)).all():
            raise ValueError(f"Expected-R lookup {column}必須全部有限")
    if bool(((table["daily_score_percentile"] < 0.0) | (table["daily_score_percentile"] > 1.0)).any()):
        raise ValueError("daily_score_percentile必須介於0與1")
    if table.duplicated(["ticker", "date"]).any():
        raise ValueError("Expected-R lookup同ticker/date不得重複")
    indexed = table.set_index(["ticker", "date"]).sort_index()
    indexed.attrs["available_from"] = str(table["date"].min()) if len(table) else ""
    indexed.attrs["available_through"] = str(table["date"].max()) if len(table) else ""
    return indexed


def load_expected_r_lookup(path: str | Path) -> pd.DataFrame:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"找不到Expected-R lookup: {resolved}")
    return _load_expected_r_lookup_cached(str(resolved), _file_signature(resolved))


def lookup_expected_r(
    *,
    lookup_path: str | Path,
    ticker: str,
    score_date,
    expected_score: float | None,
) -> dict[str, Any]:
    table = load_expected_r_lookup(lookup_path)
    ticker_text = str(ticker or "").strip()
    date_text = pd.Timestamp(score_date).strftime("%Y-%m-%d")
    if not ticker_text:
        raise ValueError("Expected-R lookup必須提供ticker")
    try:
        row = table.loc[(ticker_text, date_text)]
    except KeyError:
        return {
            "expected_r_available": False,
            "expected_r_unavailable_reason": "missing_ticker_date_calibration",
        }
    if isinstance(row, pd.DataFrame):
        raise ValueError(f"Expected-R lookup非唯一: ticker={ticker_text}, date={date_text}")
    calibrated_score = float(row["breakout_quality_score"])
    if expected_score is not None and math.isfinite(float(expected_score)):
        if abs(calibrated_score - float(expected_score)) > 1e-9:
            raise ValueError(
                "Expected-R lookup與runtime score不一致: "
                f"ticker={ticker_text}, date={date_text}, runtime={expected_score}, calibration={calibrated_score}"
            )
    return {
        "expected_r_available": True,
        "expected_r_unavailable_reason": "",
        "daily_score_percentile": float(row["daily_score_percentile"]),
        "expected_r": float(row["expected_r"]),
        "expected_r_calibration_cutoff_exclusive": str(row["calibration_cutoff_exclusive"]),
    }


__all__ = [
    "EXPECTED_R_CALIBRATION_SCHEMA_VERSION",
    "EXPECTED_R_CALIBRATION_METHOD",
    "EXPECTED_R_LOOKUP_FILENAME",
    "resolve_expected_r_calibration_dir",
    "resolve_expected_r_calibration_paths",
    "validate_expected_r_calibration_artifact",
    "load_expected_r_lookup",
    "lookup_expected_r",
]
