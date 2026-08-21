"""Shared artifact and lookup mechanics for Strategy Compare R calibrations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from filters.breakout_quality.artifacts import compute_file_sha256


@dataclass(frozen=True)
class CalibrationLookupContract:
    label: str
    artifact_dir_name: str
    schema_version: int
    method: str
    lookup_filename: str
    required_columns: tuple[str, ...]
    value_column: str
    available_key: str
    unavailable_reason_key: str
    cutoff_result_key: str
    target_semantic: str | None = None
    monotonic_direction: str | None = None
    preserve_ticker_text: bool = False
    attach_date_range_attrs: bool = False


def resolve_calibration_paths(contract, project_root, *, filter_id, model_architecture, experiment_profile, phase_id):
    root = (
        Path(project_root).resolve() / "outputs" / "strategy_compare" / "runtime_artifacts"
        / contract.artifact_dir_name / str(filter_id) / str(model_architecture)
        / str(experiment_profile) / str(phase_id).strip()
    ).resolve()
    return {"dir": root, "lookup": root / contract.lookup_filename, "manifest": root / "manifest.json", "report": root / "report.md"}


def _read_manifest(contract, path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{contract.label} manifest無法讀取: {path}; {type(exc).__name__}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{contract.label} manifest根節點必須是object")
    return payload


def validate_calibration_artifact(
    contract, *, paths, expected_identity, expected_source_score_sha256=None,
    expected_fit_score_sha256=None, expected_selection_runtime_start_date=None,
    expected_forward_frozen_cutoff_exclusive=None,
):
    if not all(paths[key].is_file() for key in ("lookup", "manifest", "report")):
        return False, "MISSING", None
    try:
        manifest = _read_manifest(contract, paths["manifest"])
        if int(manifest.get("schema_version", -1)) != int(contract.schema_version):
            raise ValueError("schema_version不一致")
        if str(manifest.get("method") or "") != contract.method:
            raise ValueError("calibration method不一致")
        identity = dict(manifest.get("identity") or {})
        if identity != expected_identity:
            raise ValueError(f"identity不一致: expected={expected_identity}, actual={identity}")

        lookup_record = dict(manifest.get("lookup_artifact") or {})
        if str(lookup_record.get("sha256") or "").lower() != compute_file_sha256(paths["lookup"]).lower():
            raise ValueError("lookup SHA256不一致")
        if int(lookup_record.get("size_bytes", -1)) != int(paths["lookup"].stat().st_size):
            raise ValueError("lookup size不一致")

        source = dict(manifest.get("source_artifacts") or {})
        if expected_source_score_sha256 and str(source.get("runtime_score_sha256") or "").lower() != str(expected_source_score_sha256).lower():
            raise ValueError("runtime score source SHA256已變更")
        if expected_fit_score_sha256 and str(source.get("fit_score_sha256") or "").lower() != str(expected_fit_score_sha256).lower():
            raise ValueError("fit score source SHA256已變更")

        fit = dict(manifest.get("fit_contract") or {})
        if contract.target_semantic is not None and fit.get("target_semantic") != contract.target_semantic:
            raise ValueError(f"{contract.label} target semantic不一致")
        if contract.monotonic_direction is not None and fit.get("monotonic_direction") != contract.monotonic_direction:
            raise ValueError(f"{contract.label} monotonic direction不一致")
        if bool(fit.get("forward_target_used_for_fit", True)):
            raise ValueError(f"Forward {contract.label} calibration不得使用Forward target")
        if expected_selection_runtime_start_date is not None:
            expected = pd.Timestamp(expected_selection_runtime_start_date).normalize().strftime("%Y-%m-%d")
            actual = str(fit.get("selection_runtime_start_date") or "")
            if actual != expected:
                raise ValueError(f"Selection {contract.label} calibration start_date已變更: expected={expected}, actual={actual or '-'}")
        if expected_forward_frozen_cutoff_exclusive is not None:
            expected = pd.Timestamp(expected_forward_frozen_cutoff_exclusive).normalize().strftime("%Y-%m-%d")
            actual = str(fit.get("forward_frozen_cutoff_exclusive") or "")
            if actual != expected:
                raise ValueError(f"Forward {contract.label} calibration execution_start cutoff已變更: expected={expected}, actual={actual or '-'}")
        return True, "READY", manifest
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return False, f"INVALID: {type(exc).__name__}: {exc}", None


def _file_signature(path):
    stat = path.stat()
    return int(stat.st_mtime_ns), int(stat.st_size)


@lru_cache(maxsize=16)
def _load_lookup_cached(contract, path_text, signature):
    del signature
    kwargs = {"low_memory": False}
    if contract.preserve_ticker_text:
        kwargs["dtype"] = {"ticker": str}
    table = pd.read_csv(Path(path_text).resolve(), **kwargs).copy()
    missing = sorted(set(contract.required_columns) - set(table.columns))
    if missing:
        raise ValueError(f"{contract.label} lookup缺少欄位: {missing}")
    table["ticker"] = table["ticker"].fillna("").astype(str).str.strip()
    table["date"] = pd.to_datetime(table["date"], errors="raise").dt.strftime("%Y-%m-%d")
    table["group_index"] = pd.to_numeric(table["group_index"], errors="raise").astype(np.int64)
    for column in ("breakout_quality_score", "daily_score_percentile", contract.value_column):
        table[column] = pd.to_numeric(table[column], errors="raise").astype(float)
        if not np.isfinite(table[column].to_numpy(dtype=np.float64, copy=False)).all():
            raise ValueError(f"{contract.label} lookup {column}必須全部有限")
    if ((table["daily_score_percentile"] < 0.0) | (table["daily_score_percentile"] > 1.0)).any():
        raise ValueError("daily_score_percentile必須介於0與1")
    if table.duplicated(["ticker", "date"]).any():
        raise ValueError(f"{contract.label} lookup同ticker/date不得重複")
    indexed = table.set_index(["ticker", "date"]).sort_index()
    if contract.attach_date_range_attrs:
        indexed.attrs.update(available_from=str(table["date"].min()) if len(table) else "", available_through=str(table["date"].max()) if len(table) else "")
    return indexed


def load_calibration_lookup(contract, path):
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"找不到{contract.label} lookup: {resolved}")
    return _load_lookup_cached(contract, str(resolved), _file_signature(resolved))


def lookup_calibration_value(contract, *, lookup_path, ticker, score_date, expected_score):
    table = load_calibration_lookup(contract, lookup_path)
    ticker_text, date_text = str(ticker or "").strip(), pd.Timestamp(score_date).strftime("%Y-%m-%d")
    if not ticker_text:
        raise ValueError(f"{contract.label} lookup必須提供ticker")
    try:
        row = table.loc[(ticker_text, date_text)]
    except KeyError:
        return {contract.available_key: False, contract.unavailable_reason_key: "missing_ticker_date_calibration"}
    if isinstance(row, pd.DataFrame):
        raise ValueError(f"{contract.label} lookup非唯一: ticker={ticker_text}, date={date_text}")
    calibrated_score = float(row["breakout_quality_score"])
    if expected_score is not None and math.isfinite(float(expected_score)) and abs(calibrated_score - float(expected_score)) > 1e-9:
        raise ValueError(f"{contract.label} lookup與runtime score不一致: ticker={ticker_text}, date={date_text}, runtime={expected_score}, calibration={calibrated_score}")
    return {
        contract.available_key: True,
        contract.unavailable_reason_key: "",
        "daily_score_percentile": float(row["daily_score_percentile"]),
        contract.value_column: float(row[contract.value_column]),
        contract.cutoff_result_key: str(row["calibration_cutoff_exclusive"]),
    }
