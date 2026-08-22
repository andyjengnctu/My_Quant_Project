"""Shared Strategy Compare model-training lifecycle.

Single-seed and multi-seed Strategy Compare differ only in the supplied seed/output
namespace.  Trainer CLI construction, subprocess execution, PIT audit and artifact
completion validation live here so the two orchestration modes cannot drift.
"""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Mapping

from config.breakout_quality import get_breakout_quality_experiment_profile
from core.console_report import COMPACT_CONSOLE_ENV
from filters.breakout_quality.artifacts import compute_file_sha256
from filters.breakout_quality.paths import (
    SELECTION_POINT_IN_TIME_COVERAGE_FILENAME,
    SELECTION_POINT_IN_TIME_MANIFEST_FILENAME,
    SELECTION_POINT_IN_TIME_SCORE_FILENAME,
    build_filter_artifact_paths_from_dir,
)
from filters.breakout_quality.ranking_score_store import (
    CONTINUOUS_RANKER_REPORT_FILENAME,
    DAILY_RANKER_OOS_SCORE_FILENAME,
    CONTINUOUS_RANKER_SCORE_FILENAME,
    SCORE_SOURCE_CONTINUOUS_RANKER_OOS,
    SCORE_SOURCE_SELECTION_POINT_IN_TIME,
    load_continuous_ranker_oos_score_table_from_path,
    load_selection_point_in_time_ranking_contract,
)
from services.research.training_process import run_logged_training_process

STRATEGY_COMPARE_TRAINER_LOG_TAIL_CHARS = 5000
STRATEGY_COMPARE_TRAINER_ENV: Mapping[str, str] = {
    COMPACT_CONSOLE_ENV: "0",
    "PYTHONUNBUFFERED": "1",
    "BREAKOUT_QUALITY_EPOCH_PROGRESS_MARKERS": "1",
}


def build_strategy_compare_trainer_command(
    *,
    source: Any,
    workflow: Any,
    seed: int,
    score_start_date: str | None = None,
    score_end_date: str | None = None,
    point_in_time_dir_override: str | Path | None = None,
    checkpoint_cache_root: str | Path | None = None,
    model_output_dir: str | Path | None = None,
    research_output_dir: str | Path | None = None,
    resume: bool = True,
) -> tuple[list[str], list[str], dict[str, object]]:
    """Build one canonical trainer command for any Strategy Compare seed."""

    score_source = str(source.score_source)
    if score_source == SCORE_SOURCE_CONTINUOUS_RANKER_OOS:
        args = [
            "--filter-id", str(source.filter_id),
            "--model-architecture", str(source.model_architecture),
            "--experiment-profile", str(source.experiment_profile),
            "--seed", str(int(seed)),
        ]
        if model_output_dir is not None:
            args.extend(["--model-output-dir", str(Path(model_output_dir).resolve())])
        if research_output_dir is not None:
            args.extend(["--research-output-dir", str(Path(research_output_dir).resolve())])
        return (
            [
                sys.executable,
                "-m",
                "tools.filters.breakout_quality.train_continuous_ranker",
                *args,
            ],
            args,
            {"score_source": score_source},
        )

    if score_source != SCORE_SOURCE_SELECTION_POINT_IN_TIME:
        raise ValueError(f"不支援的Strategy Compare score source: {score_source!r}")

    fold_months = int(
        source.point_in_time_fold_months
        if source.point_in_time_fold_months is not None
        else workflow.point_in_time_fold_months
    )
    resolved_start = str(
        score_start_date
        if score_start_date not in (None, "")
        else source.point_in_time_score_start_date
        if source.point_in_time_score_start_date not in (None, "")
        else workflow.point_in_time_score_start_date
    )
    raw_end = (
        score_end_date
        if score_end_date not in (None, "")
        else source.point_in_time_score_end_date
        if source.point_in_time_score_end_date not in (None, "")
        else workflow.point_in_time_score_end_date
    )
    resolved_end = None if raw_end in (None, "") else str(raw_end)

    args = [
        "--filter-id", str(source.filter_id),
        "--model-architecture", str(source.model_architecture),
        "--experiment-profile", str(source.experiment_profile),
        "--score-start-date", resolved_start,
        "--fold-months", str(fold_months),
        "--inner-validation-months", str(int(workflow.point_in_time_inner_validation_months)),
        "--seed", str(int(seed)),
        "--resume" if resume else "--no-resume",
    ]
    fold_anchor = (
        None
        if source.point_in_time_fold_anchor_date in (None, "")
        else str(source.point_in_time_fold_anchor_date)
    )
    if fold_anchor is not None:
        args.extend(["--fold-anchor-date", fold_anchor])
    if bool(source.point_in_time_single_score_block):
        args.append("--single-score-block")
    if resolved_end is not None:
        args.extend(["--score-end-date", resolved_end])
    if point_in_time_dir_override is not None:
        args.extend([
            "--point-in-time-dir-override",
            str(Path(point_in_time_dir_override).resolve()),
        ])
    if checkpoint_cache_root is not None:
        args.extend([
            "--checkpoint-cache-root",
            str(Path(checkpoint_cache_root).resolve()),
        ])
    return (
        [
            sys.executable,
            "-m",
            "tools.filters.breakout_quality.build_point_in_time_scores",
            *args,
        ],
        args,
        {
            "score_source": score_source,
            "score_start_date": resolved_start,
            "score_end_date": resolved_end,
            "fold_months": fold_months,
        },
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"無法讀取Strategy Compare training artifact: {Path(path).name}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Strategy Compare training artifact根節點必須是object: {Path(path).name}")
    return payload


def _normalize_date(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Strategy Compare training artifact缺少日期")
    # ISO dates are the canonical persisted form.  Avoid a pandas dependency in this
    # service layer while still accepting timestamp suffixes emitted by JSON serializers.
    return text[:10]


def validate_strategy_compare_training_artifacts(
    *,
    project_root: str | Path,
    source: Any,
    workflow: Any,
    seed: int,
    model_dir: str | Path,
    research_dir: str | Path | None,
    comparison_start: str | None,
    comparison_end: str | None,
) -> dict[str, Any]:
    """Validate one completed Strategy Compare model unit in any output namespace.

    This is the shared definition of ``READY`` for both the canonical single-seed
    pipeline and isolated robustness seeds.  PIT validation intentionally accepts a
    completed model-gate FAIL because Strategy Compare is a research consumer; the
    gate status remains explicit in the returned contract and the outer preparation
    plan decides whether the configured arm may replay it.
    """

    root = Path(project_root).resolve()
    model_root = Path(model_dir).resolve()
    required_start = None if comparison_start in (None, "") else _normalize_date(comparison_start)
    required_end = None if comparison_end in (None, "") else _normalize_date(comparison_end)
    score_source = str(source.score_source)

    if score_source == SCORE_SOURCE_SELECTION_POINT_IN_TIME:
        contract = load_selection_point_in_time_ranking_contract(
            str(root),
            str(source.filter_id),
            str(source.model_architecture),
            str(source.experiment_profile),
            require_model_validation_pass=False,
            point_in_time_dir_override=model_root,
        )
        if int(contract.seed) != int(seed):
            raise ValueError(
                "Strategy Compare PIT seed不一致: "
                f"expected={int(seed)}, actual={int(contract.seed)}"
            )
        manifest = dict(contract.manifest or {})
        expected_fold_months = int(
            source.point_in_time_fold_months
            if source.point_in_time_fold_months is not None
            else workflow.point_in_time_fold_months
        )
        if int(manifest.get("fold_months", -1) or -1) != expected_fold_months:
            raise ValueError("Strategy Compare PIT fold_months不一致")
        expected_single_block = bool(source.point_in_time_single_score_block)
        if bool(manifest.get("single_score_block", False)) != expected_single_block:
            raise ValueError("Strategy Compare PIT single_score_block不一致")
        expected_anchor = (
            None
            if source.point_in_time_fold_anchor_date in (None, "")
            else str(source.point_in_time_fold_anchor_date)
        )
        actual_anchor = str(manifest.get("fold_anchor_date") or "").strip() or None
        if expected_anchor is not None and actual_anchor != expected_anchor:
            raise ValueError("Strategy Compare PIT fold_anchor_date不一致")
        period = dict(manifest.get("score_period") or {})
        actual_start = _normalize_date(period.get("start"))
        actual_end = _normalize_date(period.get("end"))
        if required_start is not None and actual_start != required_start:
            raise ValueError(
                "Strategy Compare PIT比較起始日不一致: "
                f"expected={required_start}, actual={actual_start}"
            )
        if required_end is not None and actual_end != required_end:
            raise ValueError(
                "Strategy Compare PIT比較結束日不一致: "
                f"expected={required_end}, actual={actual_end}"
            )
        folds = [dict(item or {}) for item in list(manifest.get("folds") or [])]
        if not folds:
            raise ValueError("Strategy Compare PIT manifest沒有folds")
        epochs = [int(item.get("selected_epoch", 0) or 0) for item in folds]
        if any(epoch < 1 for epoch in epochs):
            raise ValueError("Strategy Compare PIT fold selected_epoch不合法")
        return {
            "score": str(contract.score_path.resolve()),
            "manifest": str(contract.manifest_path.resolve()),
            "audit": str(contract.audit_path.resolve()),
            "score_sha256": compute_file_sha256(contract.score_path),
            "selected_epoch": int(round(float(statistics.median(epochs)))),
            "training_elapsed_sec": float(manifest.get("elapsed_sec", 0.0) or 0.0),
            "score_execution_start": str(contract.available_from),
            "score_available_from": str(contract.available_from),
            "score_available_through": str(contract.available_through),
            "fold_count": int(manifest.get("fold_count", len(folds)) or len(folds)),
            "fold_reuse_summary": {
                str(key): int(value or 0)
                for key, value in dict(manifest.get("fold_reuse_summary") or {}).items()
            },
            "model_validation_gate": dict(contract.model_validation_gate or {}),
        }

    if score_source != SCORE_SOURCE_CONTINUOUS_RANKER_OOS:
        raise ValueError(f"不支援的Strategy Compare score source: {score_source!r}")
    if research_dir in (None, ""):
        raise ValueError("Continuous Strategy Compare training validation缺少research_dir")
    research_root = Path(research_dir).resolve()
    artifact_paths = build_filter_artifact_paths_from_dir(
        filter_id=str(source.filter_id),
        model_architecture=str(source.model_architecture),
        experiment_profile=str(source.experiment_profile),
        model_dir=model_root,
    )
    profile = get_breakout_quality_experiment_profile(str(source.experiment_profile))
    score_name = (
        DAILY_RANKER_OOS_SCORE_FILENAME
        if str(profile.training_sample_scope) == "daily_eligible_stock_days"
        else CONTINUOUS_RANKER_SCORE_FILENAME
    )
    paths = {
        "model": artifact_paths.model_path,
        "manifest": artifact_paths.manifest_path,
        "report": research_root / CONTINUOUS_RANKER_REPORT_FILENAME,
        "score": research_root / score_name,
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Strategy Compare {label}工件不存在: {path.name}")
    manifest = _read_json(paths["manifest"])
    report = _read_json(paths["report"])
    for payload, label in ((manifest, "manifest"), (report, "report")):
        for field, expected in (
            ("filter_id", source.filter_id),
            ("experiment_profile", source.experiment_profile),
            ("model_architecture", source.model_architecture),
        ):
            if str(payload.get(field) or "") != str(expected):
                raise ValueError(f"Strategy Compare {label} {field}不一致")
    training = dict(report.get("training") or {})
    if int(training.get("seed", -1)) != int(seed):
        raise ValueError("Strategy Compare continuous report seed不一致")
    outer_policy = dict(manifest.get("outer_oos_policy") or {})
    execution_start = _normalize_date(outer_policy.get("oos_start_date"))
    score_table = load_continuous_ranker_oos_score_table_from_path(
        str(paths["score"]), str(source.experiment_profile)
    )
    score_available_from = _normalize_date(score_table.attrs.get("available_from"))
    score_available_through = _normalize_date(score_table.attrs.get("available_through"))
    if execution_start > score_available_from:
        raise ValueError("Strategy Compare continuous score時間契約不一致")
    if required_start is not None and score_available_from > required_start:
        raise ValueError("Strategy Compare continuous score起始覆蓋不足")
    if required_end is not None and score_available_through < required_end:
        raise ValueError("Strategy Compare continuous score結束覆蓋不足")
    return {
        **{key: str(path.resolve()) for key, path in paths.items()},
        "model_sha256": compute_file_sha256(paths["model"]),
        "score_sha256": compute_file_sha256(paths["score"]),
        "selected_epoch": int(training.get("selected_epoch", 0) or 0),
        "training_elapsed_sec": float(report.get("elapsed_sec", 0.0) or 0.0),
        "score_execution_start": execution_start,
        "score_available_from": score_available_from,
        "score_available_through": score_available_through,
        "fold_count": None,
    }


def run_strategy_compare_training_unit(
    *,
    project_root: str | Path,
    source: Any,
    workflow: Any,
    seed: int,
    model_dir: str | Path,
    research_dir: str | Path | None,
    log_path: str | Path,
    comparison_start: str | None,
    comparison_end: str | None,
    point_in_time_dir_override: str | Path | None = None,
    checkpoint_cache_root: str | Path | None = None,
    model_output_dir: str | Path | None = None,
    research_output_dir: str | Path | None = None,
    registry: dict[str, Any] | None = None,
    registry_lock: Any = None,
    registry_key: str | None = None,
    failure_prefix: str = "Strategy Compare模型訓練失敗",
    resume: bool = True,
) -> dict[str, Any]:
    """Execute one canonical Strategy Compare training unit through READY.

    The same lifecycle is used by the normal single-seed and robustness paths:
    trainer command -> logged subprocess -> PIT audit (when applicable) -> shared
    artifact validation.  Callers own only scheduling and output namespace policy.
    """

    root = Path(project_root).resolve()
    command, report_args, metadata = build_strategy_compare_trainer_command(
        source=source,
        workflow=workflow,
        seed=int(seed),
        score_start_date=(None if comparison_start in (None, "") else str(comparison_start)),
        score_end_date=(None if comparison_end in (None, "") else str(comparison_end)),
        point_in_time_dir_override=point_in_time_dir_override,
        checkpoint_cache_root=checkpoint_cache_root,
        model_output_dir=model_output_dir,
        research_output_dir=research_output_dir,
        resume=bool(resume),
    )
    process_result = run_logged_training_process(
        command=command,
        log_path=Path(log_path),
        cwd=root,
        registry=registry,
        registry_lock=registry_lock,
        registry_key=registry_key,
        env_overrides=STRATEGY_COMPARE_TRAINER_ENV,
        failure_prefix=str(failure_prefix),
        log_tail_chars=STRATEGY_COMPARE_TRAINER_LOG_TAIL_CHARS,
    )
    audit_elapsed_sec = 0.0
    if str(source.score_source) == SCORE_SOURCE_SELECTION_POINT_IN_TIME:
        from services.breakout_quality.point_in_time_audit import (
            audit_selection_point_in_time_scores,
        )

        audit_started = time.perf_counter()
        code = audit_selection_point_in_time_scores(
            filter_id=str(source.filter_id),
            model_architecture=str(source.model_architecture),
            experiment_profile=str(source.experiment_profile),
            point_in_time_dir_override=str(Path(model_dir).resolve()),
        )
        audit_elapsed_sec = time.perf_counter() - audit_started
        if int(code) != 0:
            raise RuntimeError(
                "Strategy Compare PIT audit失敗: "
                f"profile={source.experiment_profile}, seed={int(seed)}, code={int(code)}"
            )
    validation_start = (
        str(comparison_start)
        if comparison_start not in (None, "")
        else metadata.get("score_start_date")
    )
    validation_end = (
        str(comparison_end)
        if comparison_end not in (None, "")
        else metadata.get("score_end_date")
    )
    artifacts = validate_strategy_compare_training_artifacts(
        project_root=root,
        source=source,
        workflow=workflow,
        seed=int(seed),
        model_dir=model_dir,
        research_dir=research_dir,
        comparison_start=(None if validation_start in (None, "") else str(validation_start)),
        comparison_end=(None if validation_end in (None, "") else str(validation_end)),
    )
    return {
        "artifacts": artifacts,
        "command": command,
        "report_args": report_args,
        "metadata": metadata,
        "elapsed_sec": float(process_result.get("elapsed_sec", 0.0) or 0.0),
        "audit_elapsed_sec": float(audit_elapsed_sec),
    }


__all__ = [
    "STRATEGY_COMPARE_TRAINER_ENV",
    "STRATEGY_COMPARE_TRAINER_LOG_TAIL_CHARS",
    "build_strategy_compare_trainer_command",
    "run_strategy_compare_training_unit",
    "validate_strategy_compare_training_artifacts",
]
