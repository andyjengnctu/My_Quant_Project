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
    TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS,
    TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    get_breakout_quality_experiment_profile,
)

from filters.breakout_quality.artifacts import compute_file_sha256
from filters.breakout_quality.ranker_sample_contract import build_score_eligibility_contract
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
DAILY_RANKER_OOS_SCORE_FILENAME = "daily_ranker_oos_scores.csv.gz"


def resolve_continuous_ranker_oos_score_path(
    project_root: str | Path,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
) -> Path:
    """Return the canonical Forward-OOS score artifact for the configured profile."""

    root = Path(project_root).resolve()
    profile = get_breakout_quality_experiment_profile(str(experiment_profile))
    filename = (
        DAILY_RANKER_OOS_SCORE_FILENAME
        if profile.training_sample_scope
        == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
        else CONTINUOUS_RANKER_SCORE_FILENAME
    )
    return (
        resolve_filter_model_output_dir(
            root, filter_id, model_architecture, experiment_profile
        )
        / filename
    ).resolve()

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


def _display_path(path: Path, *, project_root: Path) -> str:
    resolved = Path(path).resolve()
    root = Path(project_root).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.name


def _read_json_object(
    path: Path,
    *,
    label: str,
    project_root: Path,
) -> dict[str, Any]:
    display_path = _display_path(path, project_root=project_root)
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"無法讀取{label}: {display_path}; {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label}根節點必須是object: {display_path}")
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
    project_root: Path,
) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"Selection PIT audit缺少{label}雜湊綁定，請重新執行模型audit")
    recorded_path = Path(str(record.get("path") or "")).resolve()
    if recorded_path != expected_path.resolve():
        raise ValueError(
            f"Selection PIT audit {label}路徑不一致: "
            f"expected={_display_path(expected_path, project_root=project_root)}, "
            f"actual={_display_path(recorded_path, project_root=project_root)}"
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


def _validate_embedded_target_source_artifact(
    record: Any,
    *,
    manifest: dict[str, Any],
    target_id: str,
) -> None:
    if not isinstance(record, dict):
        raise ValueError(
            "Selection PIT audit缺少daily Continuous Target內嵌來源，請重新執行模型audit"
        )
    if str(record.get("source") or "") != "embedded_in_point_in_time_manifest":
        raise ValueError("Selection PIT audit daily Continuous Target source語意不一致")
    if str(record.get("target_id") or "") != str(target_id):
        raise ValueError("Selection PIT audit daily Continuous Target identity不一致")
    manifest_target = dict(manifest.get("source_continuous_target") or {})
    expected_contract = manifest_target.get("target_contract")
    if expected_contract in (None, {}):
        raise ValueError("Selection PIT manifest缺少daily Continuous Target contract")
    if record.get("target_contract") != expected_contract:
        raise ValueError("Selection PIT audit daily Continuous Target contract與manifest不一致")


def _validate_score_eligibility_contract(
    manifest: dict[str, Any], *, profile
) -> None:
    if (
        str(profile.training_sample_scope)
        != TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
    ):
        return
    expected = build_score_eligibility_contract(profile)
    actual = manifest.get("score_eligibility_contract")
    if actual != expected:
        raise ValueError(
            "Selection PIT daily score eligibility contract過舊或不一致；"
            "策略使用前請重新建立PIT Scores並重新執行PIT audit"
        )


def derive_point_in_time_model_validation_gate(audit: dict[str, Any]) -> dict[str, Any]:
    """Apply the predeclared model-layer gate without using strategy outcomes."""

    decision = dict(audit.get("decision_contract") or {})
    primary_scope = str(decision.get("primary_metric_scope") or "pass_only_target")
    primary_label = str(decision.get("primary_metric_label") or "PASS-only")
    metrics = dict((audit.get("metrics") or {}).get(primary_scope) or {})
    direction = dict(audit.get("direction_summary") or {})
    global_spearman = _finite_number(
        metrics.get("global_spearman"),
        field=f"PIT audit {primary_label} global_spearman",
    )
    daily_spearman = _finite_number(
        metrics.get("mean_daily_spearman"),
        field=f"PIT audit {primary_label} mean_daily_spearman",
    )
    valid_years = int(direction.get("valid_year_count", 0) or 0)
    positive_spearman_years = int(direction.get("positive_spearman_year_count", 0) or 0)
    positive_spread_years = int(direction.get("positive_spread_year_count", 0) or 0)
    if valid_years < 1:
        raise ValueError("PIT audit沒有可用年度穩定性資料")

    prefix = "pass_only" if primary_scope == "pass_only_target" else "primary"
    checks = {
        f"{prefix}_global_spearman_positive": bool(global_spearman > 0.0),
        f"{prefix}_mean_daily_spearman_positive": bool(daily_spearman > 0.0),
        "majority_years_positive_spearman": bool(positive_spearman_years * 2 > valid_years),
        "majority_years_positive_top_bottom_spread": bool(positive_spread_years * 2 > valid_years),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "primary_metric_scope": primary_scope,
        "primary_metric_label": primary_label,
        "primary_global_spearman": global_spearman,
        "primary_mean_daily_spearman": daily_spearman,
        "valid_year_count": valid_years,
        "positive_spearman_year_count": positive_spearman_years,
        "positive_spread_year_count": positive_spread_years,
        "strategy_metrics_used": False,
        "future_target_used_for_runtime_sort": False,
    }
    if primary_scope == "pass_only_target":
        result["pass_only_global_spearman"] = global_spearman
        result["pass_only_mean_daily_spearman"] = daily_spearman
    return result


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
            raise FileNotFoundError(
                f"找不到{label}: {_display_path(path, project_root=root)}"
            )

    manifest = _read_json_object(
        manifest_path,
        label="Selection PIT manifest",
        project_root=root,
    )
    audit = _read_json_object(
        audit_path,
        label="Selection PIT audit",
        project_root=root,
    )
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
    profile = get_breakout_quality_experiment_profile(str(experiment_profile))
    manifest_sample_scope = str(
        manifest.get("training_sample_scope")
        or TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS
    )
    if manifest_sample_scope != str(profile.training_sample_scope):
        raise ValueError(
            "Selection PIT manifest sample scope與profile不一致: "
            f"manifest={manifest_sample_scope}, profile={profile.training_sample_scope}"
        )
    audit_workflow = dict(audit.get("workflow") or {})
    audit_sample_scope = str(
        audit_workflow.get("training_sample_scope")
        or TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS
    )
    if audit_sample_scope != manifest_sample_scope:
        raise ValueError(
            "Selection PIT audit sample scope與manifest不一致: "
            f"audit={audit_sample_scope}, manifest={manifest_sample_scope}"
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
    ):
        if not path.is_file():
            raise FileNotFoundError(f"找不到{label}: {path}")
        _validate_audit_source_artifact(
            audit_sources.get(key),
            expected_path=path,
            label=label,
            project_root=root,
        )

    if manifest_sample_scope == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS:
        _validate_score_eligibility_contract(manifest, profile=profile)
        _validate_embedded_target_source_artifact(
            audit_sources.get("continuous_target_manifest"),
            manifest=manifest,
            target_id=manifest_target,
        )
    else:
        target_manifest_path = (
            resolve_continuous_target_dir(root, filter_id, target_id=manifest_target)
            / TARGET_MANIFEST_FILENAME
        ).resolve()
        if not target_manifest_path.is_file():
            raise FileNotFoundError(
                "找不到Continuous Target manifest: "
                f"{_display_path(target_manifest_path, project_root=root)}"
            )
        _validate_audit_source_artifact(
            audit_sources.get("continuous_target_manifest"),
            expected_path=target_manifest_path,
            label="Continuous Target manifest",
            project_root=root,
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
    profile = get_breakout_quality_experiment_profile(str(experiment_profile))
    artifacts = resolve_filter_artifact_paths(
        root, filter_id, model_architecture, experiment_profile
    )
    output_dir = resolve_filter_model_output_dir(
        root, filter_id, model_architecture, experiment_profile
    )
    score_path = resolve_continuous_ranker_oos_score_path(
        root, filter_id, model_architecture, experiment_profile
    )
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
            raise FileNotFoundError(
                f"找不到{label}: {_display_path(path, project_root=root)}"
            )

    manifest = _read_json_object(
        manifest_path,
        label="Continuous ranker manifest",
        project_root=root,
    )
    report = _read_json_object(
        report_path,
        label="Continuous ranker report",
        project_root=root,
    )
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
    if profile.training_objective not in CONTINUOUS_RANKER_TRAINING_OBJECTIVES:
        raise ValueError("Continuous ranker OOS source只接受continuous ranking profile")

    manifest_objective = str(manifest.get("training_objective") or "")
    if manifest_objective != profile.training_objective:
        raise ValueError(
            "Continuous ranker manifest training_objective與profile不一致: "
            f"expected={profile.training_objective}, actual={manifest_objective!r}"
        )
    is_daily = (
        profile.training_sample_scope
        == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
    )
    manifest_sample_scope = str(
        manifest.get("training_sample_scope")
        or TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS
    )
    if manifest_sample_scope != str(profile.training_sample_scope):
        raise ValueError(
            "Continuous ranker manifest sample scope與profile不一致: "
            f"expected={profile.training_sample_scope}, actual={manifest_sample_scope!r}"
        )
    manifest_label_scope = str(manifest.get("training_label_scope") or "")
    if is_daily:
        # MR-13A trainer predates the top-level training_label_scope field; its
        # profile identity still fixes the scope and the report remains canonical.
        if manifest_label_scope and manifest_label_scope != profile.training_label_scope:
            raise ValueError(
                "Continuous ranker manifest training_label_scope與profile不一致: "
                f"expected={profile.training_label_scope}, actual={manifest_label_scope!r}"
            )
    elif manifest_label_scope != profile.training_label_scope:
        raise ValueError(
            "Continuous ranker manifest training_label_scope與profile不一致: "
            f"expected={profile.training_label_scope}, actual={manifest_label_scope!r}"
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
    if is_daily:
        report_sample_scope = str(report_training.get("sample_scope") or "")
        if report_sample_scope != str(profile.training_sample_scope):
            raise ValueError(
                "Continuous ranker report sample scope與profile不一致: "
                f"expected={profile.training_sample_scope}, actual={report_sample_scope!r}"
            )

    if profile.training_objective == TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING:
        expected_pairwise = {
            "pair_scope": "same_date_non_tied_target_pairs",
            "pair_weighting": "equal_pair_weight",
            "model_margin": "pass_logit_minus_reject_logit",
            "batching": "whole_date_pack_no_date_split",
            "runtime_score": "softmax_pass_probability",
        }
        report_pairwise = dict(report_training.get("pairwise_contract") or {})
        manifest_semantics = dict(manifest.get("training_semantics") or {})
        if not is_daily or manifest_semantics:
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
    if is_daily:
        expected_settings = profile.as_manifest_payload()
        if report_settings != expected_settings:
            raise ValueError("Daily continuous ranker report experiment settings與profile不一致")
    report_label_scope = str(report_training.get("training_label_scope") or "")
    if report_label_scope:
        expected_label_scope = manifest_label_scope or str(profile.training_label_scope)
        if report_label_scope != expected_label_scope:
            raise ValueError("Continuous ranker report／manifest training_label_scope不一致")
    expected_score_eligibility = build_score_eligibility_contract(profile)
    if manifest.get("score_eligibility_contract") != expected_score_eligibility:
        raise ValueError(
            "Continuous ranker Forward-OOS score eligibility contract過舊或不一致；"
            "請由模型工作類型以既有frozen checkpoint重建Forward scores"
        )
    if report.get("score_eligibility_contract") != expected_score_eligibility:
        raise ValueError(
            "Continuous ranker Forward-OOS report score eligibility contract過舊或不一致"
        )
    manifest_coverage = dict(manifest.get("forward_score_coverage") or {})
    report_coverage = dict(report.get("forward_score_coverage") or {})
    if not manifest_coverage or manifest_coverage != report_coverage:
        raise ValueError("Continuous ranker Forward-OOS score coverage contract缺少或不一致")
    if manifest_coverage.get("future_target_required_for_score") is not False:
        raise ValueError("Continuous ranker Forward-OOS score不得要求future target")
    inference_groups = int(manifest_coverage.get("inference_eligible_groups", 0) or 0)
    target_evaluable_groups = int(manifest_coverage.get("target_evaluable_groups", 0) or 0)
    if inference_groups < 1 or target_evaluable_groups < 1:
        raise ValueError("Continuous ranker Forward-OOS score coverage count必須為正")
    if target_evaluable_groups > inference_groups:
        raise ValueError("Continuous ranker target-evaluable groups不可多於inference-eligible groups")
    if str(report.get("status") or "") not in {
        "RESULT_AVAILABLE_PENDING_REVIEW",
        "RESULT_AVAILABLE",
    }:
        raise ValueError(f"Continuous ranker report尚無可用結果: status={report.get('status')!r}")

    _validate_file_record_simple(
        manifest.get("model"), path=model_path, label="Continuous ranker model"
    )
    research_outputs = dict(manifest.get("research_outputs") or {})
    report_artifacts = dict(report.get("artifacts") or {})
    score_record_key = "oos_scores_gzip" if is_daily else "scores"
    _validate_file_record_simple(
        research_outputs.get(score_record_key),
        path=score_path,
        label="Continuous ranker scores",
    )
    _validate_file_record_simple(
        report_artifacts.get(score_record_key),
        path=score_path,
        label="Continuous ranker report scores",
    )

    outer = dict(manifest.get("outer_oos_policy") or {})
    execution_start = pd.Timestamp(str(outer.get("oos_start_date") or "")).strftime("%Y-%m-%d")
    configured_end = str(
        outer.get("configured_oos_end_date")
        or outer.get("effective_oos_end_date")
        or ""
    ).strip()
    information_cutoff = pd.Timestamp(
        str(manifest.get("model_information_cutoff") or "")
    ).strftime("%Y-%m-%d")
    if information_cutoff >= execution_start:
        raise ValueError(
            "Continuous ranker model_information_cutoff必須早於OOS execution_start: "
            f"cutoff={information_cutoff}, start={execution_start}"
        )
    seed = int(report_training.get("seed", -1))
    if seed < 0:
        raise ValueError("Continuous ranker report缺少合法training seed")

    table = load_continuous_ranker_oos_score_table(
        str(root), str(filter_id), str(model_architecture), str(experiment_profile)
    )
    if table.empty:
        raise ValueError("Continuous ranker OOS score table不可為空")
    available_from = str(table.attrs.get("available_from") or "")
    available_through = str(table.attrs.get("available_through") or "")
    if not available_from or not available_through:
        raise ValueError(
            "Continuous ranker OOS score table缺少預先計算的日期範圍metadata"
        )
    if available_from < execution_start:
        raise ValueError(
            "Continuous ranker OOS scores包含execution_start之前事件: "
            f"available_from={available_from}, execution_start={execution_start}"
        )
    if len(table) != inference_groups:
        raise ValueError(
            "Continuous ranker Forward-OOS score row count與inference eligibility contract不一致: "
            f"table={len(table)}, expected={inference_groups}"
        )
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


@lru_cache(maxsize=32)
def load_continuous_ranker_oos_score_table_from_path(
    score_path: str,
    experiment_profile: str,
) -> pd.DataFrame:
    profile = get_breakout_quality_experiment_profile(str(experiment_profile))
    path = Path(score_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"找不到Continuous ranker scores: {path.name}")
    frame = read_breakout_quality_csv(path).copy()
    required = {"ticker", "date", "group_index", "model_score"}
    if profile.training_sample_scope != TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS:
        required.add("split")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Continuous ranker scores缺少欄位: {missing}")
    if profile.training_sample_scope != TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS:
        frame = frame[frame["split"].astype(str) == "oos"].copy()
        if frame.empty:
            raise ValueError("Continuous ranker scores沒有OOS rows")
    elif frame.empty:
        raise ValueError("Daily continuous ranker OOS scores不可為空")
    frame["ticker"] = frame["ticker"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.strftime("%Y-%m-%d")
    frame["group_index"] = pd.to_numeric(frame["group_index"], errors="raise").astype(int)
    frame["model_score"] = pd.to_numeric(frame["model_score"], errors="raise").astype(float)
    scores = frame["model_score"].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(scores).all() or ((scores < 0.0) | (scores > 1.0)).any():
        raise ValueError("Continuous ranker OOS model_score必須為0~1有限數值")
    if frame.duplicated(["ticker", "date"]).any():
        raise ValueError("Continuous ranker OOS score同ticker/date必須唯一")
    available_from = str(frame["date"].min())
    available_through = str(frame["date"].max())
    indexed = frame.set_index(["ticker", "date"]).sort_index()
    indexed.attrs["available_from"] = available_from
    indexed.attrs["available_through"] = available_through
    return indexed


@lru_cache(maxsize=16)
def load_continuous_ranker_oos_score_table(
    project_root: str,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
) -> pd.DataFrame:
    root = Path(project_root).resolve()
    path = resolve_continuous_ranker_oos_score_path(
        root, filter_id, model_architecture, experiment_profile
    )
    return load_continuous_ranker_oos_score_table_from_path(
        str(path), str(experiment_profile)
    )


def lookup_continuous_ranker_oos_candidate_score(
    *,
    project_root: str,
    ticker: str,
    signal_date,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
    score_path_override: str | None = None,
) -> dict[str, Any]:
    contract = None
    if score_path_override is None:
        contract = load_continuous_ranker_oos_contract(
            project_root, filter_id, model_architecture, experiment_profile
        )
    table = (
        load_continuous_ranker_oos_score_table_from_path(
            str(score_path_override), str(experiment_profile)
        )
        if score_path_override is not None
        else load_continuous_ranker_oos_score_table(
            project_root, filter_id, model_architecture, experiment_profile
        )
    )
    ticker_text = str(ticker or "").strip()
    if not ticker_text:
        raise ValueError("Continuous ranker OOS lookup必須提供ticker")
    date_text = pd.Timestamp(signal_date).strftime("%Y-%m-%d")
    if contract is not None:
        available_from = str(contract.available_from)
        available_through = str(contract.available_through)
    else:
        available_from = str(table.attrs.get("available_from") or "")
        available_through = str(table.attrs.get("available_through") or "")
        if not available_from or not available_through:
            raise ValueError(
                "Continuous ranker OOS score table缺少預先計算的日期範圍metadata"
            )
    reason = ""
    score: float | None = None
    group_index: int | None = None
    if date_text < available_from or date_text > available_through:
        reason = "outside_score_period"
    else:
        key = (ticker_text, date_text)
        try:
            row = table.loc[key]
        except KeyError:
            reason = "missing_ticker_date_score"
        else:
            if isinstance(row, pd.DataFrame):
                raise ValueError(
                    f"Continuous ranker OOS lookup非唯一: ticker={ticker_text}, date={date_text}"
                )
            score = float(row["model_score"])
            group_index = int(row["group_index"])
    profile = get_breakout_quality_experiment_profile(str(experiment_profile))
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
        "continuous_target_id": (
            contract.continuous_target_id if contract is not None else str(profile.continuous_target_id or "")
        ),
        "model_information_cutoff": (
            contract.model_information_cutoff if contract is not None else None
        ),
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
    "resolve_continuous_ranker_oos_score_path",
    "load_continuous_ranker_oos_contract",
    "load_continuous_ranker_oos_score_table",
    "lookup_continuous_ranker_oos_candidate_score",
    "SelectionPointInTimeRankingContract",
    "derive_point_in_time_model_validation_gate",
    "load_selection_point_in_time_ranking_contract",
    "load_selection_point_in_time_score_table",
    "lookup_selection_point_in_time_candidate_score",
]
