"""Validated score-source contracts used by breakout-quality candidate ranking.

Canonical forward-OOS runtime scores remain owned by ``score_store.py``.  This module
adds read-only research score sources whose lookup semantics are explicitly scoped to
strategy replay and never alter scanner/filter runtime behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.breakout_quality import (
    CONTINUOUS_RANKER_TRAINING_OBJECTIVES,
    TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_LISTWISE_RANKING,
    get_breakout_quality_experiment_profile,
)

from filters.breakout_quality.artifacts import compute_file_sha256
from filters.breakout_quality.continuous_target import (
    TARGET_MANIFEST_FILENAME,
    resolve_continuous_target_dir,
)
from filters.breakout_quality.csv_io import read_breakout_quality_csv
from filters.breakout_quality.paths import (
    resolve_filter_artifact_paths,
    resolve_filter_model_output_dir,
    resolve_selection_point_in_time_audit_json_path,
    resolve_selection_point_in_time_coverage_path,
    resolve_selection_point_in_time_manifest_path,
    resolve_selection_point_in_time_score_path,
)

SCORE_SOURCE_CANONICAL_RUNTIME = "canonical_runtime"
SCORE_SOURCE_SELECTION_POINT_IN_TIME = "selection_point_in_time"
SCORE_SOURCE_CONTINUOUS_RANKER_OOS = "continuous_ranker_oos"
SUPPORTED_RANKING_SCORE_SOURCES = (
    SCORE_SOURCE_CANONICAL_RUNTIME,
    SCORE_SOURCE_SELECTION_POINT_IN_TIME,
    SCORE_SOURCE_CONTINUOUS_RANKER_OOS,
)

CONTINUOUS_RANKER_SCORE_FILENAME = "continuous_ranker_scores.csv"
CONTINUOUS_RANKER_REPORT_FILENAME = "continuous_ranker_report.json"

PIT_REQUIRED_SCORE_COLUMNS = (
    "ticker",
    "date",
    "group_index",
    "breakout_quality_score",
    "fold_id",
    "model_information_cutoff",
)


@dataclass(frozen=True)
class SelectionPointInTimeRankingContract:
    score_path: Path
    manifest_path: Path
    audit_path: Path
    manifest: dict[str, Any]
    audit: dict[str, Any]
    filter_id: str
    model_architecture: str
    experiment_profile: str
    continuous_target_id: str
    seed: int
    available_from: str
    available_through: str
    model_validation_gate: dict[str, Any]


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"無法讀取{label}: {path}; {type(exc).__name__}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label}根節點必須是object: {path}")
    return payload


def _finite_number(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field}必須是有限數值: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field}必須是有限數值: {value!r}")
    return number


def _validate_audit_source_artifact(
    record: Any,
    *,
    expected_path: Path,
    label: str,
) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"Selection PIT audit缺少{label}雜湊綁定，請重新執行模型audit")
    recorded_path = Path(str(record.get("path") or "")).resolve()
    if recorded_path != expected_path.resolve():
        raise ValueError(
            f"Selection PIT audit {label}路徑不一致: "
            f"expected={expected_path.resolve()}, actual={recorded_path}"
        )
    if str(record.get("filename") or "") != expected_path.name:
        raise ValueError(f"Selection PIT audit {label} filename不一致")
    expected_hash = str(record.get("sha256") or "").lower()
    actual_hash = compute_file_sha256(expected_path).lower()
    if not expected_hash or expected_hash != actual_hash:
        raise ValueError(
            f"Selection PIT audit {label} SHA256不一致: "
            f"expected={expected_hash or '<missing>'}, actual={actual_hash}"
        )
    if int(record.get("size_bytes", -1)) != int(expected_path.stat().st_size):
        raise ValueError(f"Selection PIT audit {label} size不一致")


def derive_point_in_time_model_validation_gate(audit: dict[str, Any]) -> dict[str, Any]:
    """Apply the predeclared model-layer gate without using strategy outcomes."""

    metrics = dict((audit.get("metrics") or {}).get("pass_only_target") or {})
    direction = dict(audit.get("direction_summary") or {})
    global_spearman = _finite_number(
        metrics.get("global_spearman"), field="PIT audit PASS-only global_spearman"
    )
    daily_spearman = _finite_number(
        metrics.get("mean_daily_spearman"), field="PIT audit PASS-only mean_daily_spearman"
    )
    valid_years = int(direction.get("valid_year_count", 0) or 0)
    positive_spearman_years = int(direction.get("positive_spearman_year_count", 0) or 0)
    positive_spread_years = int(direction.get("positive_spread_year_count", 0) or 0)
    if valid_years < 1:
        raise ValueError("PIT audit沒有可用年度穩定性資料")

    checks = {
        "pass_only_global_spearman_positive": bool(global_spearman > 0.0),
        "pass_only_mean_daily_spearman_positive": bool(daily_spearman > 0.0),
        "majority_years_positive_spearman": bool(positive_spearman_years * 2 > valid_years),
        "majority_years_positive_top_bottom_spread": bool(positive_spread_years * 2 > valid_years),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "pass_only_global_spearman": global_spearman,
        "pass_only_mean_daily_spearman": daily_spearman,
        "valid_year_count": valid_years,
        "positive_spearman_year_count": positive_spearman_years,
        "positive_spread_year_count": positive_spread_years,
        "strategy_metrics_used": False,
        "future_target_used_for_runtime_sort": False,
    }


@lru_cache(maxsize=16)
def load_selection_point_in_time_ranking_contract(
    project_root: str,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
) -> SelectionPointInTimeRankingContract:
    root = Path(project_root).resolve()
    score_path = resolve_selection_point_in_time_score_path(
        root, filter_id, model_architecture, experiment_profile
    ).resolve()
    manifest_path = resolve_selection_point_in_time_manifest_path(
        root, filter_id, model_architecture, experiment_profile
    ).resolve()
    audit_path = resolve_selection_point_in_time_audit_json_path(
        root, filter_id, model_architecture, experiment_profile
    ).resolve()
    for label, path in (
        ("Selection PIT score", score_path),
        ("Selection PIT manifest", manifest_path),
        ("Selection PIT audit", audit_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"找不到{label}: {path}")

    manifest = _read_json_object(manifest_path, label="Selection PIT manifest")
    audit = _read_json_object(audit_path, label="Selection PIT audit")
    expected_identity = {
        "filter_id": str(filter_id),
        "model_architecture": str(model_architecture),
        "experiment_profile": str(experiment_profile),
    }
    for field, expected in expected_identity.items():
        if str(manifest.get(field) or "") != expected:
            raise ValueError(
                f"Selection PIT manifest identity不一致: field={field}, "
                f"expected={expected}, actual={manifest.get(field)!r}"
            )
        if str(audit.get(field) or "") != expected:
            raise ValueError(
                f"Selection PIT audit identity不一致: field={field}, "
                f"expected={expected}, actual={audit.get(field)!r}"
            )
    if str(manifest.get("status") or "") != "BUILT":
        raise ValueError(f"Selection PIT manifest尚未完成: status={manifest.get('status')!r}")
    if int(audit.get("schema_version", 0) or 0) < 3:
        raise ValueError("Selection PIT audit版本未綁定來源雜湊，請重新執行模型audit")
    if str(audit.get("status") or "") not in {
        "RESULT_AVAILABLE_PENDING_REVIEW",
        "RESULT_AVAILABLE",
        "MODEL_VALIDATION_PASS",
    }:
        raise ValueError(f"Selection PIT audit尚無可用結果: status={audit.get('status')!r}")

    score_record = dict((manifest.get("artifacts") or {}).get("scores") or {})
    if str(score_record.get("filename") or "") != score_path.name:
        raise ValueError("Selection PIT score filename與manifest不一致")
    expected_hash = str(score_record.get("sha256") or "").lower()
    actual_hash = compute_file_sha256(score_path).lower()
    if not expected_hash or expected_hash != actual_hash:
        raise ValueError(
            "Selection PIT score SHA256與manifest不一致: "
            f"expected={expected_hash or '<missing>'}, actual={actual_hash}"
        )
    expected_size = int(score_record.get("size_bytes", -1))
    if expected_size != int(score_path.stat().st_size):
        raise ValueError("Selection PIT score size與manifest不一致")

    score_period = dict(manifest.get("score_period") or {})
    available_from = pd.Timestamp(score_period.get("start")).strftime("%Y-%m-%d")
    available_through = pd.Timestamp(score_period.get("end")).strftime("%Y-%m-%d")
    if available_through < available_from:
        raise ValueError("Selection PIT score period不合法")

    manifest_target = str(manifest.get("continuous_target_id") or "")
    target_manifest_path = (
        resolve_continuous_target_dir(root, filter_id, target_id=manifest_target)
        / TARGET_MANIFEST_FILENAME
    ).resolve()
    if not target_manifest_path.is_file():
        raise FileNotFoundError(f"找不到Continuous Target manifest: {target_manifest_path}")
    audit_sources = dict(audit.get("source_artifacts") or {})
    for key, path, label in (
        ("point_in_time_manifest", manifest_path, "PIT manifest"),
        ("point_in_time_scores", score_path, "PIT Scores"),
        (
            "point_in_time_coverage",
            resolve_selection_point_in_time_coverage_path(
                root, filter_id, model_architecture, experiment_profile
            ).resolve(),
            "PIT coverage",
        ),
        ("continuous_target_manifest", target_manifest_path, "Continuous Target manifest"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"找不到{label}: {path}")
        _validate_audit_source_artifact(
            audit_sources.get(key), expected_path=path, label=label
        )
    if str(audit.get("continuous_target_id") or "") != manifest_target:
        raise ValueError("Selection PIT audit continuous target與manifest不一致")
    audit_period = dict(audit.get("score_period") or {})
    if audit_period != score_period:
        raise ValueError("Selection PIT audit score period與manifest不一致")
    if int(audit.get("score_group_count", -1)) != int(
        (manifest.get("coverage") or {}).get("scored_group_count", -2)
    ):
        raise ValueError("Selection PIT audit score group count與manifest coverage不一致")

    gate = derive_point_in_time_model_validation_gate(audit)
    if gate["status"] != "PASS":
        failed = [name for name, passed in gate["checks"].items() if not passed]
        raise ValueError(
            "Selection PIT模型驗證未通過，不得進入策略排序: " + ", ".join(failed)
        )

    return SelectionPointInTimeRankingContract(
        score_path=score_path,
        manifest_path=manifest_path,
        audit_path=audit_path,
        manifest=manifest,
        audit=audit,
        filter_id=str(filter_id),
        model_architecture=str(model_architecture),
        experiment_profile=str(experiment_profile),
        continuous_target_id=manifest_target,
        seed=int(manifest.get("seed")),
        available_from=available_from,
        available_through=available_through,
        model_validation_gate=gate,
    )


@lru_cache(maxsize=16)
def load_selection_point_in_time_score_table(
    project_root: str,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
) -> pd.DataFrame:
    contract = load_selection_point_in_time_ranking_contract(
        project_root, filter_id, model_architecture, experiment_profile
    )
    table = read_breakout_quality_csv(contract.score_path).copy()
    missing = sorted(set(PIT_REQUIRED_SCORE_COLUMNS).difference(table.columns))
    if missing:
        raise ValueError(f"Selection PIT score table缺少欄位: {missing}")
    table = table[list(PIT_REQUIRED_SCORE_COLUMNS)].copy()
    table["ticker"] = table["ticker"].fillna("").astype(str).str.strip()
    table["date"] = pd.to_datetime(table["date"], errors="raise").dt.strftime("%Y-%m-%d")
    table["group_index"] = pd.to_numeric(table["group_index"], errors="raise").astype(np.int64)
    table["breakout_quality_score"] = pd.to_numeric(
        table["breakout_quality_score"], errors="raise"
    ).astype(float)
    table["fold_id"] = table["fold_id"].fillna("").astype(str).str.strip()
    table["model_information_cutoff"] = pd.to_datetime(
        table["model_information_cutoff"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    if bool((table["ticker"] == "").any()) or bool((table["fold_id"] == "").any()):
        raise ValueError("Selection PIT score table含空白ticker或fold_id")
    values = table["breakout_quality_score"].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all() or bool(((values < 0.0) | (values > 1.0)).any()):
        raise ValueError("Selection PIT score必須全部為0到1的有限值")
    if table.duplicated(["ticker", "date"]).any():
        raise ValueError("Selection PIT score table同一ticker/date出現重複Score")
    if table.duplicated(["group_index"]).any():
        raise ValueError("Selection PIT score table同一group_index出現重複Score")
    if str(table["date"].min()) < contract.available_from or str(table["date"].max()) > contract.available_through:
        raise ValueError("Selection PIT score實際日期超出manifest score period")
    expected_count = int((contract.manifest.get("coverage") or {}).get("scored_group_count", -1))
    if len(table) != expected_count:
        raise ValueError(
            f"Selection PIT score row count與manifest不一致: expected={expected_count}, actual={len(table)}"
        )
    return table.set_index(["ticker", "date"]).sort_index()


@dataclass(frozen=True)
class ContinuousRankerOOSContract:
    score_path: Path
    manifest_path: Path
    report_path: Path
    manifest: dict[str, Any]
    report: dict[str, Any]
    filter_id: str
    model_architecture: str
    experiment_profile: str
    continuous_target_id: str
    seed: int
    model_information_cutoff: str
    execution_start: str
    available_from: str
    available_through: str


def _validate_file_record_simple(record: Any, *, path: Path, label: str) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"{label}缺少檔案identity")
    if str(record.get("filename") or "") != path.name:
        raise ValueError(f"{label} filename不一致")
    expected_hash = str(record.get("sha256") or "").lower()
    actual_hash = compute_file_sha256(path).lower()
    if not expected_hash or expected_hash != actual_hash:
        raise ValueError(
            f"{label} SHA256不一致: expected={expected_hash or '<missing>'}, actual={actual_hash}"
        )
    if int(record.get("size_bytes", -1)) != int(path.stat().st_size):
        raise ValueError(f"{label} size不一致")


@lru_cache(maxsize=16)
def load_continuous_ranker_oos_contract(
    project_root: str,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
) -> ContinuousRankerOOSContract:
    root = Path(project_root).resolve()
    artifacts = resolve_filter_artifact_paths(
        root, filter_id, model_architecture, experiment_profile
    )
    output_dir = resolve_filter_model_output_dir(
        root, filter_id, model_architecture, experiment_profile
    )
    score_path = (output_dir / CONTINUOUS_RANKER_SCORE_FILENAME).resolve()
    report_path = (output_dir / CONTINUOUS_RANKER_REPORT_FILENAME).resolve()
    manifest_path = artifacts.manifest_path.resolve()
    model_path = artifacts.model_path.resolve()
    for label, path in (
        ("Continuous ranker model", model_path),
        ("Continuous ranker manifest", manifest_path),
        ("Continuous ranker report", report_path),
        ("Continuous ranker OOS scores", score_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"找不到{label}: {path}")

    manifest = _read_json_object(manifest_path, label="Continuous ranker manifest")
    report = _read_json_object(report_path, label="Continuous ranker report")
    expected_identity = {
        "filter_id": str(filter_id),
        "model_architecture": str(model_architecture),
        "experiment_profile": str(experiment_profile),
    }
    for field, expected in expected_identity.items():
        if str(manifest.get(field) or "") != expected:
            raise ValueError(
                f"Continuous ranker manifest identity不一致: field={field}, "
                f"expected={expected}, actual={manifest.get(field)!r}"
            )
        if str(report.get(field) or "") != expected:
            raise ValueError(
                f"Continuous ranker report identity不一致: field={field}, "
                f"expected={expected}, actual={report.get(field)!r}"
            )
    profile = get_breakout_quality_experiment_profile(str(experiment_profile))
    if profile.training_objective not in CONTINUOUS_RANKER_TRAINING_OBJECTIVES:
        raise ValueError("Continuous ranker OOS source只接受continuous ranking profile")
    manifest_objective = str(manifest.get("training_objective") or "")
    if manifest_objective != profile.training_objective:
        raise ValueError(
            "Continuous ranker manifest training_objective與profile不一致: "
            f"expected={profile.training_objective}, actual={manifest_objective!r}"
        )
    manifest_scope = str(manifest.get("training_label_scope") or "")
    if manifest_scope != profile.training_label_scope:
        raise ValueError(
            "Continuous ranker manifest training_label_scope與profile不一致: "
            f"expected={profile.training_label_scope}, actual={manifest_scope!r}"
        )
    report_training = dict(report.get("training") or {})
    report_objective = str(report_training.get("objective") or "")
    if report_objective != profile.training_objective:
        raise ValueError(
            "Continuous ranker report training objective與profile不一致: "
            f"expected={profile.training_objective}, actual={report_objective!r}"
        )
    report_loss = str(report_training.get("loss") or "")
    if report_loss != profile.loss_name:
        raise ValueError(
            "Continuous ranker report loss與profile不一致: "
            f"expected={profile.loss_name}, actual={report_loss!r}"
        )
    if profile.training_objective == TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING:
        expected_pairwise = {
            "pair_scope": "same_date_non_tied_target_pairs",
            "pair_weighting": "equal_pair_weight",
            "model_margin": "pass_logit_minus_reject_logit",
            "batching": "whole_date_pack_no_date_split",
            "runtime_score": "softmax_pass_probability",
        }
        manifest_semantics = dict(manifest.get("training_semantics") or {})
        report_pairwise = dict(report_training.get("pairwise_contract") or {})
        if str(manifest_semantics.get("batching") or "") != expected_pairwise["batching"]:
            raise ValueError("Continuous pairwise ranker manifest batching contract不一致")
        if dict(manifest_semantics.get("pairwise_contract") or {}) != expected_pairwise:
            raise ValueError("Continuous pairwise ranker manifest pairwise contract不一致")
        if str(report_training.get("batching") or "") != expected_pairwise["batching"]:
            raise ValueError("Continuous pairwise ranker report batching contract不一致")
        if report_pairwise != expected_pairwise:
            raise ValueError("Continuous pairwise ranker report pairwise contract不一致")
    if profile.training_objective == TRAINING_OBJECTIVE_DAILY_LISTWISE_RANKING:
        expected_listwise = {
            "list_scope": "same_date_full_candidate_list",
            "target_distribution": "softmax_daily_percentile",
            "prediction_distribution": "softmax_pass_minus_reject_margin",
            "tie_handling": "equal_target_equal_distribution_weight",
            "date_weighting": "equal_rankable_date_weight",
            "model_margin": "pass_logit_minus_reject_logit",
            "batching": "whole_date_pack_no_date_split",
            "runtime_score": "softmax_pass_probability",
        }
        manifest_semantics = dict(manifest.get("training_semantics") or {})
        report_listwise = dict(report_training.get("listwise_contract") or {})
        if str(manifest_semantics.get("batching") or "") != expected_listwise["batching"]:
            raise ValueError("Continuous listwise ranker manifest batching contract不一致")
        if dict(manifest_semantics.get("listwise_contract") or {}) != expected_listwise:
            raise ValueError("Continuous listwise ranker manifest listwise contract不一致")
        if str(report_training.get("batching") or "") != expected_listwise["batching"]:
            raise ValueError("Continuous listwise ranker report batching contract不一致")
        if report_listwise != expected_listwise:
            raise ValueError("Continuous listwise ranker report listwise contract不一致")

    target_id = str(manifest.get("continuous_target_id") or "")
    if not target_id:
        raise ValueError("Continuous ranker manifest缺少continuous_target_id")
    if target_id != str(profile.continuous_target_id or ""):
        raise ValueError(
            "Continuous ranker manifest continuous_target_id與profile不一致: "
            f"expected={profile.continuous_target_id}, actual={target_id!r}"
        )
    report_settings = dict(report.get("experiment_settings") or {})
    report_target = str(report_settings.get("continuous_target_id") or "")
    if report_target and report_target != target_id:
        raise ValueError("Continuous ranker report／manifest continuous_target_id不一致")
    report_scope = str((report.get("training") or {}).get("training_label_scope") or "")
    if report_scope and report_scope != manifest_scope:
        raise ValueError("Continuous ranker report／manifest training_label_scope不一致")
    if str(report.get("status") or "") not in {
        "RESULT_AVAILABLE_PENDING_REVIEW",
        "RESULT_AVAILABLE",
    }:
        raise ValueError(f"Continuous ranker report尚無可用結果: status={report.get('status')!r}")

    _validate_file_record_simple(manifest.get("model"), path=model_path, label="Continuous ranker model")
    research_outputs = dict(manifest.get("research_outputs") or {})
    _validate_file_record_simple(
        research_outputs.get("scores"), path=score_path, label="Continuous ranker scores"
    )
    report_artifacts = dict(report.get("artifacts") or {})
    _validate_file_record_simple(
        report_artifacts.get("scores"), path=score_path, label="Continuous ranker report scores"
    )

    outer = dict(manifest.get("outer_oos_policy") or {})
    execution_start = pd.Timestamp(str(outer.get("oos_start_date") or "")).strftime("%Y-%m-%d")
    configured_end = str(outer.get("configured_oos_end_date") or outer.get("effective_oos_end_date") or "").strip()
    information_cutoff = pd.Timestamp(str(manifest.get("model_information_cutoff") or "")).strftime("%Y-%m-%d")
    if information_cutoff >= execution_start:
        raise ValueError(
            "Continuous ranker model_information_cutoff必須早於OOS execution_start: "
            f"cutoff={information_cutoff}, start={execution_start}"
        )
    training = dict(report.get("training") or {})
    seed = int(training.get("seed", -1))
    if seed < 0:
        raise ValueError("Continuous ranker report缺少合法training seed")

    table = load_continuous_ranker_oos_score_table(
        str(root), str(filter_id), str(model_architecture), str(experiment_profile)
    )
    if table.empty:
        raise ValueError("Continuous ranker OOS score table不可為空")
    available_from = str(table.index.get_level_values("date").min())
    available_through = str(table.index.get_level_values("date").max())
    if available_from < execution_start:
        # OOS score可包含execution_start前一個signal anchor時才合理；標準輸出通常不會。
        if pd.Timestamp(available_from) < pd.Timestamp(information_cutoff):
            raise ValueError("Continuous ranker OOS scores包含model information cutoff之前事件")
    if configured_end and pd.Timestamp(available_through) > pd.Timestamp(configured_end):
        raise ValueError("Continuous ranker OOS scores超出outer OOS configured end")
    return ContinuousRankerOOSContract(
        score_path=score_path,
        manifest_path=manifest_path,
        report_path=report_path,
        manifest=manifest,
        report=report,
        filter_id=str(filter_id),
        model_architecture=str(model_architecture),
        experiment_profile=str(experiment_profile),
        continuous_target_id=target_id,
        seed=seed,
        model_information_cutoff=information_cutoff,
        execution_start=execution_start,
        available_from=available_from,
        available_through=available_through,
    )


@lru_cache(maxsize=16)
def load_continuous_ranker_oos_score_table(
    project_root: str,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
) -> pd.DataFrame:
    root = Path(project_root).resolve()
    path = (
        resolve_filter_model_output_dir(root, filter_id, model_architecture, experiment_profile)
        / CONTINUOUS_RANKER_SCORE_FILENAME
    ).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"找不到Continuous ranker scores: {path}")
    frame = read_breakout_quality_csv(path).copy()
    required = {"ticker", "date", "group_index", "split", "model_score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Continuous ranker scores缺少欄位: {missing}")
    frame = frame[frame["split"].astype(str) == "oos"].copy()
    if frame.empty:
        raise ValueError("Continuous ranker scores沒有OOS rows")
    frame["ticker"] = frame["ticker"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.strftime("%Y-%m-%d")
    frame["group_index"] = pd.to_numeric(frame["group_index"], errors="raise").astype(int)
    frame["model_score"] = pd.to_numeric(frame["model_score"], errors="raise").astype(float)
    scores = frame["model_score"].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(scores).all() or ((scores < 0.0) | (scores > 1.0)).any():
        raise ValueError("Continuous ranker OOS model_score必須為0~1有限數值")
    if frame.duplicated(["ticker", "date"]).any():
        raise ValueError("Continuous ranker OOS score同ticker/date必須唯一")
    return frame.set_index(["ticker", "date"]).sort_index()


def lookup_continuous_ranker_oos_candidate_score(
    *,
    project_root: str,
    ticker: str,
    signal_date,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
) -> dict[str, Any]:
    contract = load_continuous_ranker_oos_contract(
        project_root, filter_id, model_architecture, experiment_profile
    )
    ticker_text = str(ticker or "").strip()
    if not ticker_text:
        raise ValueError("Continuous ranker OOS lookup必須提供ticker")
    date_text = pd.Timestamp(signal_date).strftime("%Y-%m-%d")
    reason = ""
    score: float | None = None
    group_index: int | None = None
    if date_text < contract.available_from or date_text > contract.available_through:
        reason = "outside_score_period"
    else:
        table = load_continuous_ranker_oos_score_table(
            project_root, filter_id, model_architecture, experiment_profile
        )
        key = (ticker_text, date_text)
        if key not in table.index:
            reason = "missing_ticker_date_score"
        else:
            row = table.loc[key]
            if isinstance(row, pd.DataFrame):
                raise ValueError(
                    f"Continuous ranker OOS lookup非唯一: ticker={ticker_text}, date={date_text}"
                )
            score = float(row["model_score"])
            group_index = int(row["group_index"])
    return {
        "score": score,
        "available": not bool(reason),
        "unavailable_reason": reason,
        "score_date": date_text,
        "shared_group_score": True,
        "filter_id": str(filter_id),
        "score_source": SCORE_SOURCE_CONTINUOUS_RANKER_OOS,
        "model_architecture": str(model_architecture),
        "experiment_profile": str(experiment_profile),
        "continuous_target_id": contract.continuous_target_id,
        "model_information_cutoff": contract.model_information_cutoff,
        "group_index": group_index,
    }


def lookup_selection_point_in_time_candidate_score(
    *,
    project_root: str,
    ticker: str,
    signal_date,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
) -> dict[str, Any]:
    contract = load_selection_point_in_time_ranking_contract(
        project_root, filter_id, model_architecture, experiment_profile
    )
    ticker_text = str(ticker or "").strip()
    if not ticker_text:
        raise ValueError("Selection PIT ranking lookup必須提供ticker")
    date_text = pd.Timestamp(signal_date).strftime("%Y-%m-%d")
    reason = ""
    score: float | None = None
    fold_id = ""
    information_cutoff = ""
    group_index: int | None = None
    if date_text < contract.available_from or date_text > contract.available_through:
        reason = "outside_score_period"
    else:
        table = load_selection_point_in_time_score_table(
            project_root, filter_id, model_architecture, experiment_profile
        )
        key = (ticker_text, date_text)
        if key not in table.index:
            reason = "missing_ticker_date_score"
        else:
            row = table.loc[key]
            if isinstance(row, pd.DataFrame):
                raise ValueError(f"Selection PIT score lookup非唯一: ticker={ticker_text}, date={date_text}")
            score = float(row["breakout_quality_score"])
            fold_id = str(row["fold_id"])
            information_cutoff = str(row["model_information_cutoff"])
            group_index = int(row["group_index"])
    return {
        "score": score,
        "available": not bool(reason),
        "unavailable_reason": reason,
        "score_date": date_text,
        "shared_group_score": True,
        "filter_id": str(filter_id),
        "score_source": SCORE_SOURCE_SELECTION_POINT_IN_TIME,
        "model_architecture": str(model_architecture),
        "experiment_profile": str(experiment_profile),
        "fold_id": fold_id,
        "model_information_cutoff": information_cutoff,
        "group_index": group_index,
    }


__all__ = [
    "SCORE_SOURCE_CANONICAL_RUNTIME",
    "SCORE_SOURCE_SELECTION_POINT_IN_TIME",
    "SCORE_SOURCE_CONTINUOUS_RANKER_OOS",
    "SUPPORTED_RANKING_SCORE_SOURCES",
    "ContinuousRankerOOSContract",
    "load_continuous_ranker_oos_contract",
    "load_continuous_ranker_oos_score_table",
    "lookup_continuous_ranker_oos_candidate_score",
    "SelectionPointInTimeRankingContract",
    "derive_point_in_time_model_validation_gate",
    "load_selection_point_in_time_ranking_contract",
    "load_selection_point_in_time_score_table",
    "lookup_selection_point_in_time_candidate_score",
]
