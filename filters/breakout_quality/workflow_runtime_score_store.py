"""Dedicated production score artifact for the accepted breakout-quality workflow."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from core.file_integrity import compute_file_sha256
from core.model_paths import resolve_models_dir
from filters.breakout_quality.contract import (
    DEFAULT_SCORE_FILENAME,
    DEFAULT_UNAVAILABLE_SCORE_FILENAME,
    RUNTIME_SCOPE_WORKFLOW,
)
from filters.breakout_quality.paths import resolve_filter_artifact_paths
from filters.breakout_quality.ranking_score_store import (
    SCORE_SOURCE_CANONICAL_RUNTIME,
    load_continuous_ranker_oos_contract,
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

    ranker_contract = load_continuous_ranker_oos_contract(
        str(project_root), str(filter_id), str(model_architecture), str(experiment_profile)
    )
    artifact_paths = resolve_filter_artifact_paths(
        project_root, str(filter_id), str(model_architecture), str(experiment_profile)
    )
    model_identity = dict(manifest.get("model_identity") or {})
    _validate_file_record(
        artifact_paths.model_path,
        dict(model_identity.get("model_checkpoint") or {}),
        field_name="source model checkpoint",
    )
    expected_identity_fields = {
        "model_information_cutoff": ranker_contract.model_information_cutoff,
        "experiment_settings": ranker_contract.manifest.get("experiment_settings"),
        "model_spec": ranker_contract.manifest.get("model_spec"),
        "training_objective": ranker_contract.manifest.get("training_objective"),
        "training_sample_scope": ranker_contract.manifest.get("training_sample_scope"),
    }
    for field_name, expected_value in expected_identity_fields.items():
        if model_identity.get(field_name) != expected_value:
            raise ValueError(
                f"workflow runtime source model identity漂移: field={field_name}"
            )

    score_record = dict(manifest.get("score_table") or {})
    _validate_file_record(paths["score"], score_record, field_name="scores.csv")
    table = read_breakout_quality_csv(paths["score"]).copy()
    if list(table.columns) != list(score_record.get("columns") or []):
        raise ValueError("workflow runtime score columns 與 manifest 不一致")
    required_score_columns = {"ticker", "date", "model_score"}
    missing = sorted(required_score_columns - set(table.columns))
    if missing:
        raise ValueError(f"workflow runtime daily score 缺少欄位: {missing}")
    if list(score_record.get("key_columns") or []) != ["ticker", "date"]:
        raise ValueError("workflow runtime daily score key必須是ticker/date")
    if len(table) != int(score_record.get("row_count") or -1):
        raise ValueError("workflow runtime score row_count 與 manifest 不一致")
    table["ticker"] = table["ticker"].astype(str)
    table["date"] = pd.to_datetime(table["date"], errors="raise").dt.strftime("%Y-%m-%d")
    table["model_score"] = pd.to_numeric(table["model_score"], errors="raise").astype(float)
    scores = table["model_score"].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(scores).all():
        raise ValueError("workflow runtime score 含非有限值")
    if table.duplicated(["ticker", "date"]).any():
        raise ValueError("workflow runtime daily score table 有重複 ticker/date")

    shared = bool(manifest.get("shared_group_score_broadcast"))
    if not shared:
        raise ValueError("daily-universal workflow runtime必須宣告shared_group_score_broadcast")
    score_lookup = table.set_index(["ticker", "date"])[["model_score"]].sort_index()

    unavailable_record = dict(manifest.get("conservative_unscorable_events") or {})
    _validate_file_record(paths["unavailable"], unavailable_record, field_name="unavailable_scores.csv")
    unavailable = read_breakout_quality_csv(paths["unavailable"]).copy()
    required_unavailable_columns = ["ticker", "date", "reason"]
    if list(unavailable.columns) != required_unavailable_columns:
        raise ValueError(
            "workflow runtime unavailable score columns不一致: "
            f"expected={required_unavailable_columns}, actual={list(unavailable.columns)}"
        )
    if not unavailable.empty:
        unavailable["ticker"] = unavailable["ticker"].astype(str)
        unavailable["date"] = pd.to_datetime(unavailable["date"], errors="raise").dt.strftime("%Y-%m-%d")
        unavailable["reason"] = unavailable["reason"].astype(str)
        if unavailable.duplicated(["ticker", "date"]).any():
            raise ValueError("workflow runtime unavailable score ticker/date必須唯一")
        unavailable_lookup = unavailable.set_index(["ticker", "date"])[["reason"]].sort_index()
    else:
        unavailable_lookup = pd.DataFrame(
            columns=["reason"],
            index=pd.MultiIndex.from_arrays([[], []], names=["ticker", "date"]),
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
    if not ticker_text:
        raise ValueError("workflow runtime score lookup必須提供ticker")
    key = (ticker_text, date_text)
    reason = ""
    score = None
    try:
        row = bundle["score_lookup"].loc[key]
    except KeyError:
        reason = "missing_ticker_date_score"
    else:
        if isinstance(row, pd.DataFrame):
            raise ValueError(
                f"workflow runtime daily score lookup非唯一: ticker={ticker_text}, date={date_text}"
            )
        score = float(row["model_score"])
    unavailable_lookup = bundle["unavailable_lookup"]
    if key in unavailable_lookup.index:
        value = unavailable_lookup.loc[key, "reason"]
        explicit_reason = str(value.iloc[0] if hasattr(value, "iloc") else value)
        if explicit_reason:
            reason = explicit_reason
            score = None
    return {
        "score": score,
        "available": not bool(reason),
        "unavailable_reason": reason,
        "score_date": date_text,
        "shared_group_score": True,
        "filter_id": str(filter_id),
        "model_architecture": str(model_architecture),
        "experiment_profile": str(experiment_profile),
        "score_source": SCORE_SOURCE_CANONICAL_RUNTIME,
        "high_len": int(high_len),
    }


__all__ = [
    "load_workflow_runtime_score_bundle",
    "lookup_workflow_runtime_candidate_score",
    "resolve_workflow_runtime_score_paths",
]
