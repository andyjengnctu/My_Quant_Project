"""Dedicated production score artifact for the accepted breakout-quality workflow."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from core.file_integrity import compute_file_sha256
from core.model_paths import resolve_models_dir
from filters.breakout_quality.artifacts import load_model_artifact_contract
from filters.breakout_quality.contract import (
    DEFAULT_SCORE_FILENAME,
    DEFAULT_UNAVAILABLE_SCORE_FILENAME,
    RUNTIME_SCOPE_WORKFLOW,
    SCORE_COLUMN,
    SCORE_TABLE_REQUIRED_COLUMNS,
)
from filters.breakout_quality.csv_io import read_breakout_quality_csv


def resolve_workflow_runtime_score_paths(
    *,
    project_root: str | Path,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
) -> dict[str, Path]:
    root = (
        Path(resolve_models_dir(str(project_root)))
        / "runtime"
        / "breakout_quality"
        / str(filter_id)
        / str(model_architecture)
        / str(experiment_profile)
    )
    return {
        "dir": root,
        "score": root / DEFAULT_SCORE_FILENAME,
        "unavailable": root / DEFAULT_UNAVAILABLE_SCORE_FILENAME,
        "manifest": root / "runtime_manifest.json",
    }


def _validate_file_record(path: Path, record: dict, *, field_name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"正式 workflow runtime 缺少{field_name}: {path}")
    expected_sha = str(record.get("sha256") or "").strip()
    if not expected_sha:
        raise ValueError(f"正式 workflow runtime {field_name}缺少sha256")
    actual_sha = compute_file_sha256(path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"正式 workflow runtime {field_name} SHA256不一致: expected={expected_sha}, actual={actual_sha}"
        )


@lru_cache(maxsize=8)
def load_workflow_runtime_score_bundle(
    project_root: str,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
) -> dict:
    paths = resolve_workflow_runtime_score_paths(
        project_root=project_root,
        filter_id=filter_id,
        model_architecture=model_architecture,
        experiment_profile=experiment_profile,
    )
    manifest_path = paths["manifest"]
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "找不到 MR-13E 正式 workflow runtime score；請先執行「Runtime 整合 Gate → 套用／更新正式 Runtime」。"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"無法讀取 workflow runtime manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("workflow runtime manifest 根節點必須是 object")
    expected_identity = {
        "filter_id": str(filter_id),
        "model_architecture": str(model_architecture),
        "experiment_profile": str(experiment_profile),
    }
    actual_identity = {key: str(manifest.get(key) or "") for key in expected_identity}
    if actual_identity != expected_identity:
        raise ValueError(
            f"workflow runtime identity不一致: expected={expected_identity}, actual={actual_identity}"
        )
    runtime = dict(manifest.get("runtime_eligibility") or {})
    if runtime.get("eligible") is not True or str(runtime.get("scope") or "") != RUNTIME_SCOPE_WORKFLOW:
        raise ValueError("workflow runtime artifact 未宣告 workflow_runtime eligible")
    if str(runtime.get("causal_information_contract") or "") != "same_day_and_earlier_ohlcv_only":
        raise ValueError("workflow runtime artifact 缺少 causal information contract")

    model_contract = load_model_artifact_contract(
        str(project_root), str(filter_id), str(model_architecture), str(experiment_profile)
    )
    model_identity = dict(manifest.get("model_identity") or {})
    _validate_file_record(
        model_contract.paths.model_path,
        dict(model_identity.get("model_checkpoint") or {}),
        field_name="source model checkpoint",
    )
    for field_name in ("model_information_cutoff", "experiment_settings", "model_spec"):
        if model_identity.get(field_name) != model_contract.manifest.get(field_name):
            raise ValueError(
                f"workflow runtime source model identity漂移: field={field_name}"
            )

    score_record = dict(manifest.get("score_table") or {})
    _validate_file_record(paths["score"], score_record, field_name="scores.csv")
    table = read_breakout_quality_csv(paths["score"]).copy()
    if list(table.columns) != list(score_record.get("columns") or []):
        raise ValueError("workflow runtime score columns 與 manifest 不一致")
    missing = sorted(set(SCORE_TABLE_REQUIRED_COLUMNS) - set(table.columns))
    if missing:
        raise ValueError(f"workflow runtime score 缺少欄位: {missing}")
    if len(table) != int(score_record.get("row_count") or -1):
        raise ValueError("workflow runtime score row_count 與 manifest 不一致")
    table["ticker"] = table["ticker"].astype(str)
    table["date"] = pd.to_datetime(table["date"], errors="raise").dt.strftime("%Y-%m-%d")
    table["high_len"] = pd.to_numeric(table["high_len"], errors="raise").astype(int)
    table[SCORE_COLUMN] = pd.to_numeric(table[SCORE_COLUMN], errors="raise").astype(float)
    scores = table[SCORE_COLUMN].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(scores).all():
        raise ValueError("workflow runtime score 含非有限值")
    if table.duplicated(["ticker", "date", "high_len"]).any():
        raise ValueError("workflow runtime score table 有重複 event key")

    shared = bool(manifest.get("shared_group_score_broadcast"))
    if shared:
        counts = table.groupby(["ticker", "date"], sort=False)[SCORE_COLUMN].nunique(dropna=False)
        if bool((counts != 1).any()):
            raise ValueError("workflow runtime shared ticker/date score 不一致")
        score_lookup = (
            table.drop_duplicates(["ticker", "date"], keep="first")
            .set_index(["ticker", "date"])[[SCORE_COLUMN]]
            .sort_index()
        )
    else:
        score_lookup = table.set_index(["ticker", "date", "high_len"])[[SCORE_COLUMN]].sort_index()

    unavailable_record = dict(manifest.get("conservative_unscorable_events") or {})
    _validate_file_record(paths["unavailable"], unavailable_record, field_name="unavailable_scores.csv")
    unavailable = read_breakout_quality_csv(paths["unavailable"]).copy()
    if not unavailable.empty:
        unavailable["ticker"] = unavailable["ticker"].astype(str)
        unavailable["date"] = pd.to_datetime(unavailable["date"], errors="raise").dt.strftime("%Y-%m-%d")
        unavailable["high_len"] = pd.to_numeric(unavailable["high_len"], errors="raise").astype(int)
        unavailable["reason"] = unavailable["reason"].astype(str)
        unavailable_lookup = unavailable.set_index(["ticker", "date", "high_len"])[["reason"]].sort_index()
    else:
        unavailable_lookup = pd.DataFrame(
            columns=["reason"],
            index=pd.MultiIndex.from_arrays([[], [], []], names=["ticker", "date", "high_len"]),
        )
    return {
        "manifest": manifest,
        "score_lookup": score_lookup,
        "unavailable_lookup": unavailable_lookup,
        "shared_group_score_broadcast": shared,
    }


def lookup_workflow_runtime_candidate_score(
    *,
    project_root: str,
    ticker: str,
    information_date,
    high_len: int,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
) -> dict:
    bundle = load_workflow_runtime_score_bundle(
        str(project_root), str(filter_id), str(model_architecture), str(experiment_profile)
    )
    manifest = dict(bundle["manifest"])
    runtime = dict(manifest["runtime_eligibility"])
    date_text = pd.Timestamp(information_date).strftime("%Y-%m-%d")
    current_date = pd.Timestamp(date_text).date()
    available_from = pd.Timestamp(str(runtime["available_from"])).date()
    available_through = pd.Timestamp(str(runtime["available_through"])).date()
    if current_date < available_from or current_date > available_through:
        raise ValueError(
            "workflow runtime score coverage不足；請更新正式 Runtime 工件: "
            f"date={date_text}, coverage={available_from}~{available_through}"
        )
    ticker_text = str(ticker).strip()
    shared = bool(bundle["shared_group_score_broadcast"])
    key = (ticker_text, date_text) if shared else (ticker_text, date_text, int(high_len))
    try:
        score = float(bundle["score_lookup"].loc[key, SCORE_COLUMN])
    except KeyError as exc:
        raise ValueError(
            f"workflow runtime score缺少候選: ticker={ticker_text}, date={date_text}, high_len={int(high_len)}"
        ) from exc
    unavailable_key = (ticker_text, date_text, int(high_len))
    reason = ""
    unavailable_lookup = bundle["unavailable_lookup"]
    if unavailable_key in unavailable_lookup.index:
        value = unavailable_lookup.loc[unavailable_key, "reason"]
        reason = str(value.iloc[0] if hasattr(value, "iloc") else value)
    return {
        "score": score,
        "available": not bool(reason),
        "unavailable_reason": reason,
        "score_date": date_text,
        "shared_group_score": shared,
        "filter_id": str(filter_id),
        "model_architecture": str(model_architecture),
        "experiment_profile": str(experiment_profile),
        "score_source": RUNTIME_SCOPE_WORKFLOW,
    }


__all__ = [
    "load_workflow_runtime_score_bundle",
    "lookup_workflow_runtime_candidate_score",
    "resolve_workflow_runtime_score_paths",
]
