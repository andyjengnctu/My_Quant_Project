"""Selection PIT Score-ranking rolling parameter-adaptation validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd

from config.breakout_quality import (
    BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD,
    get_breakout_quality_workflow_settings,
)
from config.training_policy import OPTIMIZER_FIXED_TP_PERCENT
from core.dataset_profiles import get_dataset_dir
from filters.breakout_quality.paths import resolve_filter_model_output_dir
from core.raw_universe_contract import RAW_UNIVERSE_REQUIRED_MIN_ROWS_FIELD
from core.runtime_utils import get_taipei_now
from core.walk_forward_policy import (
    build_optimizer_effective_policy_fingerprint,
    load_walk_forward_policy,
)
from filters.breakout_quality.artifacts import compute_file_sha256
from filters.breakout_quality.console_report import (
    compact_console_enabled,
    console_color_enabled,
    print_artifact_paths,
    project_relative_display_path,
    render_key_values,
    render_section,
    render_table,
    render_title,
)
from filters.breakout_quality.ranking_score_store import (
    SCORE_SOURCE_SELECTION_POINT_IN_TIME,
    load_selection_point_in_time_ranking_contract,
)
from filters.breakout_quality.source_inventory import build_source_data_inventory
from strategies.breakout.search_space import (
    BREAKOUT_OPTIMIZER_SEARCH_SPACE,
    get_breakout_optimizer_required_min_rows,
)
from tools.filters.breakout_quality.audit_score_ranking_capture import (
    build_score_ranking_capture_audit,
    render_capture_audit_console,
    write_score_ranking_capture_audit_outputs,
)
from tools.filters.breakout_quality.strategy_report_style import (
    SIGNAL_NEGATIVE,
    SIGNAL_NEUTRAL,
    SIGNAL_POSITIVE,
    SIGNAL_WARNING,
    signal_for_delta,
    signal_marker,
    terminal_signal,
)
from tools.filters.breakout_quality.strategy_compare import (
    COMPARISON_MODE_SCORE_RANKING,
    PARAM_POLICY_BASE_FINALIST_BEST,
    _assert_shared_benchmark,
    _comparison_labels,
    _comparison_output_dir_name,
    _build_controlled_param_source_pair,
    _delta,
    _flatten_candidate_replay_rows,
    _flatten_selected_buy_rows,
    _load_param_source,
    _resolve_params_path,
    _run_scenario,
    _scenario_summary,
    _selection_target_lookup,
    _strategy_selection_diagnostics,
    _validate_requested_param_policy,
    _to_json_native,
    _yearly_frame,
)
from tools.optimizer.prep import load_all_raw_data
from tools.optimizer.outer_rolling_oos import (
    materialize_fixed_strategy_param_overrides_in_active_param_payload,
    run_outer_rolling_oos,
)
from tools.optimizer.runtime import create_optimizer_study
from tools.optimizer.session_factory import (
    build_optimizer_session,
    configure_optuna_logging,
    ensure_study_effective_policy_compatible,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = 1
ADAPTATION_STATUS = "ROLLING_SELECTION_DIAGNOSTIC"
ADAPTATION_RELATIVE_DIR = Path(
    "models/research/breakout_quality/score_ranking_adaptation/rolling_validation"
)
CURRENT_PAIR_MANIFEST_FILENAME = "adaptation_pair_manifest.json"

def _parse_args(argv=None):
    settings = get_breakout_quality_workflow_settings()
    parser = argparse.ArgumentParser(
        description=(
            "沿用正式Baseline ROOS與rolling fold schedule，由目前training policy提供"
            "trials／fold，只在Selection PIT Score ranking下訓練一套Adapted params，"
            "再以舊／Score ranking回放舊／新兩套參數，輸出四組2×2 Selection診斷。"
        )
    )
    parser.add_argument("--dataset", choices=("reduced", "full"), default=settings.strategy_dataset)
    parser.add_argument("--filter-id", default=settings.filter_id)
    parser.add_argument("--model-architecture", default=settings.model_architecture)
    parser.add_argument("--experiment-profile", default=settings.experiment_profile)
    parser.add_argument(
        "--param-policy",
        choices=(PARAM_POLICY_BASE_FINALIST_BEST,),
        default=PARAM_POLICY_BASE_FINALIST_BEST,
    )
    parser.add_argument(
        "--trials-per-fold", "--trials",
        dest="trials_per_fold",
        type=int,
        default=settings.strategy_adapt_trials_per_fold,
        help=(
            "每個rolling fold的optimizer trial數；預設讀取目前training policy。"
            "Baseline既有工件的歷史trial數只作診斷，不作硬性限制"
        ),
    )
    parser.add_argument("--max-positions", type=int, default=settings.strategy_max_positions)
    parser.add_argument("--rotation", choices=("off", "on"), default=settings.strategy_rotation)
    parser.add_argument("--fixed-risk", type=float, default=settings.strategy_adapt_fixed_risk)
    parser.add_argument(
        "--max-position-cap-pct",
        type=float,
        default=settings.strategy_adapt_max_position_cap_pct,
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)

def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_json_native(payload), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )

def _current_pair_artifact_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "strategy_comparison_json": output_dir / "strategy_comparison.json",
        "strategy_comparison_markdown": output_dir / "strategy_comparison.md",
        "yearly_csv": output_dir / "yearly_returns_comparison.csv",
        "capture_json": output_dir / "score_ranking_capture_audit.json",
        "capture_markdown": output_dir / "score_ranking_capture_audit.md",
        "baseline_equity": output_dir / "no_filter_equity.csv",
        "sort_only_equity": output_dir / "score_ranking_equity.csv",
        "baseline_trades": output_dir / "no_filter_trades.csv",
        "sort_only_trades": output_dir / "score_ranking_trades.csv",
        "baseline_capacity": output_dir / "no_filter_daily_capacity.csv",
        "sort_only_capacity": output_dir / "score_ranking_daily_capacity.csv",
        "baseline_orderable": output_dir / "no_filter_orderable_target_diagnostics.csv",
        "sort_only_orderable": output_dir / "score_ranking_orderable_target_diagnostics.csv",
        "baseline_selected": output_dir / "no_filter_selected_target_diagnostics.csv",
        "sort_only_selected": output_dir / "score_ranking_selected_target_diagnostics.csv",
        "baseline_capture_lifecycle": output_dir / "no_filter_capture_lifecycle.csv",
        "sort_only_capture_lifecycle": output_dir / "score_ranking_capture_lifecycle.csv",
        "capture_yearly": output_dir / "score_ranking_capture_yearly.csv",
        "capture_scenarios": output_dir / "score_ranking_capture_scenarios.csv",
    }


def _canonical_current_pair_output_dir(*, root: Path, args) -> Path:
    output_name = _comparison_output_dir_name(
        COMPARISON_MODE_SCORE_RANKING,
        _comparison_labels(COMPARISON_MODE_SCORE_RANKING),
        param_policy=str(args.param_policy),
    ) + "_selection_point_in_time"
    return (
        resolve_filter_model_output_dir(
            str(root),
            str(args.filter_id),
            str(args.model_architecture),
            str(args.experiment_profile),
        )
        / output_name
    ).resolve()


def _collect_nested_param_values(payload: Any, field: str) -> list[Any]:
    values: list[Any] = []
    if isinstance(payload, dict):
        if field in payload:
            values.append(payload[field])
        for value in payload.values():
            values.extend(_collect_nested_param_values(value, field))
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            values.extend(_collect_nested_param_values(value, field))
    return values


def _effective_fixed_value_matches(
    *, metadata: dict[str, Any], override_field: str, param_field: str, expected: float
) -> tuple[bool, str]:
    recorded_override = metadata.get(override_field)
    override_matches = True
    if recorded_override is not None:
        try:
            override_matches = math.isclose(
                float(recorded_override), float(expected), rel_tol=0.0, abs_tol=1e-12
            )
        except (TypeError, ValueError):
            override_matches = False

    # 舊版[1]沒有明確傳入override，但正式ROOS本身可能已固定為相同值。
    # 無論新舊metadata，都以兩組實際回放參數再驗證一次，避免只信宣告欄位。
    sourced_values: list[tuple[str, Any]] = []
    for payload_key in ("no_filter_params", "score_ranking_params"):
        sourced_values.extend(
            (payload_key, value)
            for value in _collect_nested_param_values(
                metadata.get(payload_key), param_field
            )
        )
    normalized: list[float] = []
    for payload_key, value in sourced_values:
        try:
            normalized.append(float(value))
        except (TypeError, ValueError):
            return False, f"{payload_key}.{param_field}含無效值={value!r}"
    if not normalized:
        return False, f"工件未保存可驗證的effective {param_field}"
    effective_matches = all(
        math.isclose(value, float(expected), rel_tol=0.0, abs_tol=1e-12)
        for value in normalized
    )
    unique_values = sorted(set(normalized))
    return (
        override_matches and effective_matches,
        f"{override_field}={recorded_override!r}; effective {param_field}={unique_values}",
    )


def _resolve_recorded_artifact_path(raw_path: Any, *, root: Path) -> Path | None:
    text = str(raw_path or "").strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _load_current_pair_if_compatible(
    *,
    root: Path,
    output_dir: Path,
    runtime_identity_sha256: str,
    args,
    pit_contract,
    baseline_contract: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    del runtime_identity_sha256  # adaptation manifest是衍生索引；相容後會重新寫入。
    issues: list[str] = []
    expected_paths = _current_pair_artifact_paths(output_dir)
    comparison_path = expected_paths["strategy_comparison_json"]
    if not comparison_path.is_file():
        return None, [f"缺少{project_relative_display_path(comparison_path, project_root=root)}"]
    try:
        payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, [f"strategy_comparison.json無法讀取：{type(exc).__name__}: {exc}"]
    metadata = dict(payload.get("metadata") or {})
    if not metadata:
        return None, ["strategy_comparison.json缺少metadata"]

    expected = {
        "comparison_mode": COMPARISON_MODE_SCORE_RANKING,
        "score_source": SCORE_SOURCE_SELECTION_POINT_IN_TIME,
        "dataset": str(args.dataset),
        "params_file_sha256": str(baseline_contract["sha256"]),
        "requested_param_policy": str(args.param_policy),
        "filter_id": str(args.filter_id),
        "model_architecture": str(args.model_architecture),
        "experiment_profile": str(args.experiment_profile),
        "max_positions": int(args.max_positions),
        "enable_rotation": str(args.rotation) == "on",
        "comparison_design": "selection_point_in_time_active_param_replay",
        "lookahead_safe_active_param_schedule": True,
    }
    for key, expected_value in expected.items():
        actual = metadata.get(key)
        if actual != expected_value:
            issues.append(f"{key}: artifact={actual!r}, expected={expected_value!r}")

    for override_field, param_field, expected_value in (
        ("fixed_risk_override", "fixed_risk", float(args.fixed_risk)),
        (
            "max_position_cap_pct_override",
            "max_position_cap_pct",
            float(args.max_position_cap_pct),
        ),
    ):
        matched, detail = _effective_fixed_value_matches(
            metadata=metadata,
            override_field=override_field,
            param_field=param_field,
            expected=expected_value,
        )
        if not matched:
            issues.append(f"{param_field}: {detail}, expected={expected_value}")

    expected_period = {
        "start": str(pit_contract.available_from),
        "end": str(pit_contract.available_through),
    }
    period = dict(metadata.get("comparison_period") or {})
    if period != expected_period:
        issues.append(
            f"comparison_period: artifact={period!r}, expected={expected_period!r}"
        )
    for metadata_key, expected_path in (
        ("score_path", pit_contract.score_path),
        ("score_manifest_path", pit_contract.manifest_path),
        ("score_audit_path", pit_contract.audit_path),
    ):
        recorded_path = _resolve_recorded_artifact_path(
            metadata.get(metadata_key), root=root
        )
        if recorded_path != Path(expected_path).resolve():
            issues.append(
                f"{metadata_key}: artifact={recorded_path}, "
                f"expected={Path(expected_path).resolve()}"
            )
    score_table = dict(metadata.get("score_table") or {})
    recorded_score_sha = str(
        score_table.get("sha256")
        or score_table.get("file_sha256")
        or score_table.get("content_sha256")
        or ""
    )
    actual_score_sha = compute_file_sha256(pit_contract.score_path)
    if recorded_score_sha and recorded_score_sha != actual_score_sha:
        issues.append(
            f"score SHA256: artifact={recorded_score_sha}, expected={actual_score_sha}"
        )

    missing = [
        project_relative_display_path(path, project_root=root)
        for path in expected_paths.values()
        if not path.is_file()
    ]
    if missing:
        issues.append("缺少必要比較工件：" + ", ".join(missing))

    if issues:
        return None, issues
    return payload, []


def _write_current_pair_manifest(
    *, root: Path, output_dir: Path, runtime_identity_sha256: str
) -> Path:
    artifact_paths = _current_pair_artifact_paths(output_dir)
    missing = [str(path) for path in artifact_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Baseline／Sort Only完成後缺少必要工件: " + ", ".join(missing)
        )
    manifest_path = output_dir / CURRENT_PAIR_MANIFEST_FILENAME
    _write_json(
        manifest_path,
        {
            "schema_version": SCHEMA_VERSION,
            "runtime_identity_sha256": str(runtime_identity_sha256),
            "created_at": get_taipei_now().isoformat(),
            "artifacts": {
                key: {
                    "path": project_relative_display_path(path, project_root=root),
                    "sha256": compute_file_sha256(path),
                }
                for key, path in artifact_paths.items()
            },
        },
    )
    return manifest_path


def _load_or_run_current_pair(
    *,
    root: Path,
    args,
    runtime_contract: dict[str, Any],
    output_dir: Path,
    pit_contract,
    baseline_contract: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    payload, issues = _load_current_pair_if_compatible(
        root=root,
        output_dir=output_dir,
        runtime_identity_sha256=runtime_contract["runtime_identity_sha256"],
        args=args,
        pit_contract=pit_contract,
        baseline_contract=baseline_contract,
    )
    if payload is None:
        details = "；".join(issues[:8]) or "未知identity差異"
        raise FileNotFoundError(
            "找不到與目前identity一致的Baseline／Sort Only正式比較工件；"
            f"原因={details}。請先執行主選單[2]策略績效驗證→[1]比較目前策略。"
        )
    # adaptation_pair_manifest是[2]的衍生索引；[1]重跑或舊版identity格式變更後
    # 應依已驗證的正式比較工件重新產生，不得反過來阻擋有效工件。
    _write_current_pair_manifest(
        root=root,
        output_dir=output_dir,
        runtime_identity_sha256=runtime_contract["runtime_identity_sha256"],
    )
    print("\n[Baseline／Sort Only] 已載入[1]正式比較工件；不重新執行舊參數replay。")
    return payload, True


def _validate_fixed_contract(args, settings) -> None:
    if settings.strategy_comparison_mode != "score-ranking":
        raise ValueError("strategy adaptation只支援continuous-ranker score-ranking workflow")
    if settings.strategy_score_source != SCORE_SOURCE_SELECTION_POINT_IN_TIME:
        raise ValueError("strategy adaptation只接受selection_point_in_time Score source")
    fixed_identity = {
        "filter_id": (args.filter_id, settings.filter_id),
        "model_architecture": (args.model_architecture, settings.model_architecture),
        "experiment_profile": (args.experiment_profile, settings.experiment_profile),
    }
    mismatches = [
        f"{field}: actual={actual!r}, workflow={expected!r}"
        for field, (actual, expected) in fixed_identity.items()
        if str(actual) != str(expected)
    ]
    if mismatches:
        raise ValueError(
            "strategy adaptation必須使用目前workflow凍結的模型identity："
            + "; ".join(mismatches)
        )
    if args.param_policy != PARAM_POLICY_BASE_FINALIST_BEST:
        raise ValueError("第一輪strategy adaptation固定使用base-finalist-best")
    if args.rotation not in {"off", "on"}:
        raise ValueError("rotation只接受off或on")
    if int(args.max_positions) < 1:
        raise ValueError("max_positions必須>=1")
    if not 0.0 < float(args.fixed_risk) <= 1.0:
        raise ValueError("fixed_risk必須介於0與1之間")
    if not 0.0 < float(args.max_position_cap_pct) <= 1.0:
        raise ValueError("max_position_cap_pct必須介於0與1之間")
    if int(args.trials_per_fold) < 1:
        raise ValueError("trials_per_fold必須>=1")

def _validate_pit_contract_against_settings(pit_contract, settings) -> None:
    manifest = dict(getattr(pit_contract, "manifest", {}) or {})
    checks = {
        "continuous_target_id": (
            str(pit_contract.continuous_target_id),
            str(settings.continuous_target_id),
        ),
        "seed": (int(pit_contract.seed), int(settings.seed)),
        "score_start": (
            str(pit_contract.available_from),
            str(settings.point_in_time_score_start_date),
        ),
        "fold_months": (
            int(manifest.get("fold_months", -1)),
            int(settings.point_in_time_fold_months),
        ),
        "inner_validation_months": (
            int(manifest.get("inner_validation_months", -1)),
            int(settings.point_in_time_inner_validation_months),
        ),
    }
    if settings.point_in_time_score_end_date is not None:
        checks["score_end"] = (
            str(pit_contract.available_through),
            str(settings.point_in_time_score_end_date),
        )
    mismatches = [
        f"{field}: artifact={actual!r}, workflow={expected!r}"
        for field, (actual, expected) in checks.items()
        if actual != expected
    ]
    if mismatches:
        raise ValueError(
            "Selection PIT工件與目前workflow凍結契約不一致："
            + "; ".join(mismatches)
        )

def _load_baseline_rolling_contract(*, root: Path, args, settings) -> dict[str, Any]:
    params_path = _resolve_params_path(
        root=root,
        params_path=None,
        param_policy=args.param_policy,
        allow_static_diagnostic=False,
        score_source=SCORE_SOURCE_SELECTION_POINT_IN_TIME,
    )
    if not params_path.is_file():
        raise FileNotFoundError(f"找不到Baseline rolling active params: {params_path}")
    payload = json.loads(params_path.read_text(encoding="utf-8"))
    source = _load_param_source(params_path)
    policy_contract = _validate_requested_param_policy(source, args.param_policy)
    if str(source.get("kind") or "") != "rolling_active_param_ensemble":
        raise ValueError("策略參數適應驗證只接受rolling active-param ensemble Baseline")
    meta = dict(payload.get("meta") or {})
    summary = dict(payload.get("summary") or {})
    required_meta = (
        "window_mode",
        "first_oos_date",
        "last_oos_date",
        "train_window_months",
        "oos_horizon_months",
        "trials_per_fold",
        "active_param_policy",
    )
    missing = [key for key in required_meta if meta.get(key) in (None, "")]
    if missing:
        raise ValueError("Baseline rolling active params缺少meta欄位: " + ", ".join(missing))
    if int(summary.get("folds", 0) or 0) < 1:
        raise ValueError("Baseline rolling active params沒有有效fold摘要")
    if len(list(payload.get("folds") or [])) != int(summary["folds"]):
        raise ValueError("Baseline rolling active params的fold明細與summary不一致")
    return {
        "path": params_path,
        "payload": payload,
        "source": source,
        "policy_contract": policy_contract,
        "meta": meta,
        "summary": summary,
        "sha256": compute_file_sha256(params_path),
    }


def _build_training_score_coverage_contract(
    *, baseline_contract: dict[str, Any], pit_contract
) -> dict[str, Any]:
    """Describe how much PIT Score history each rolling training fold can use.

    The formal Selection PIT artifact begins at the strategy comparison period.  Older
    portions of a rolling training window therefore use the existing buy-sort through
    the established missing-score fallback.  This is allowed only as a contiguous
    bootstrap history before the PIT period; gaps inside the PIT period fail fast.
    """

    score_start = pd.Timestamp(pit_contract.available_from).normalize()
    score_end = pd.Timestamp(pit_contract.available_through).normalize()
    raw_folds = list(baseline_contract["payload"].get("folds") or [])
    expected_fold_count = int(baseline_contract["summary"].get("folds", 0) or 0)
    if not raw_folds or len(raw_folds) != expected_fold_count:
        raise ValueError(
            "Baseline rolling active params的fold明細不完整："
            f"rows={len(raw_folds)}, expected={expected_fold_count}"
        )

    rows: list[dict[str, Any]] = []
    unexpected_gaps: list[str] = []
    for index, fold in enumerate(raw_folds, start=1):
        selection_start = pd.Timestamp(fold.get("selection_start_date")).normalize()
        selection_end = pd.Timestamp(fold.get("selection_end_date")).normalize()
        oos_start = pd.Timestamp(fold.get("oos_start_date")).normalize()
        oos_end = pd.Timestamp(fold.get("oos_end_date")).normalize()
        if selection_end < selection_start or oos_end < oos_start:
            raise ValueError(f"Baseline rolling fold日期不合法: fold={index}")
        if oos_start < score_start or oos_end > score_end:
            raise ValueError(
                "Score Adapted的每個OOS replay都必須完整位於Selection PIT期間："
                f"fold={index}, oos={oos_start.date()}~{oos_end.date()}, "
                f"pit={score_start.date()}~{score_end.date()}"
            )

        overlap_start = max(selection_start, score_start)
        overlap_end = min(selection_end, score_end)
        has_overlap = overlap_start <= overlap_end
        if not has_overlap:
            if selection_end < score_start:
                status = "bootstrap_fallback_only"
            else:
                status = "unexpected_score_gap"
                unexpected_gaps.append(str(index))
            score_days = 0
            overlap_start_text = None
            overlap_end_text = None
        else:
            status = (
                "full_score_history"
                if selection_start >= score_start and selection_end <= score_end
                else "partial_score_history"
            )
            score_days = int((overlap_end - overlap_start).days) + 1
            overlap_start_text = overlap_start.strftime("%Y-%m-%d")
            overlap_end_text = overlap_end.strftime("%Y-%m-%d")

        selection_days = int((selection_end - selection_start).days) + 1
        rows.append({
            "fold": str(fold.get("fold") or f"{index}/{expected_fold_count}"),
            "oos_period": f"{oos_start:%Y-%m-%d}~{oos_end:%Y-%m-%d}",
            "selection_period": (
                f"{selection_start:%Y-%m-%d}~{selection_end:%Y-%m-%d}"
            ),
            "score_overlap_start": overlap_start_text,
            "score_overlap_end": overlap_end_text,
            "score_calendar_days": int(score_days),
            "selection_calendar_days": int(selection_days),
            "calendar_coverage_ratio": float(score_days / selection_days),
            "status": status,
            "missing_score_fallback_used_before_pit_start": bool(
                selection_start < score_start
            ),
        })

    if unexpected_gaps:
        raise ValueError(
            "PIT Score期間內出現非預期rolling training coverage缺口：folds="
            + ",".join(unexpected_gaps)
        )

    return {
        "pit_score_period": {
            "start": score_start.strftime("%Y-%m-%d"),
            "end": score_end.strftime("%Y-%m-%d"),
        },
        "fold_count": len(rows),
        "bootstrap_fallback_only_folds": sum(
            row["status"] == "bootstrap_fallback_only" for row in rows
        ),
        "partial_score_history_folds": sum(
            row["status"] == "partial_score_history" for row in rows
        ),
        "full_score_history_folds": sum(
            row["status"] == "full_score_history" for row in rows
        ),
        "missing_score_fallback_contract": (
            "dates_before_pit_start_fall_back_to_existing_buy_sort; "
            "gaps_inside_pit_period_are_forbidden"
        ),
        "folds": rows,
    }


def _build_base_policy(*, root: Path, baseline_contract: dict[str, Any]) -> dict[str, Any]:
    policy = load_walk_forward_policy(str(root))
    meta = dict(baseline_contract["meta"])
    first_oos = pd.Timestamp(meta["first_oos_date"])
    last_oos = pd.Timestamp(meta["last_oos_date"])
    train_months = int(meta["train_window_months"])
    training_start = first_oos - pd.DateOffset(months=train_months)
    policy.update({
        "model_mode": "oos",
        "study_scope": "split",
        "adaptation_scope": "selection_score_ranking_rolling_validation",
        "evaluation_scope": "rolling_selection_diagnostic",
        "objective_mode": "split_train_romd",
        "selection_start_year": int(training_start.year),
        "train_start_year": int(training_start.year),
        "search_train_end_year": int((first_oos - pd.Timedelta(days=1)).year),
        "selection_start_date": training_start.strftime("%Y-%m-%d"),
        "train_start_date": training_start.strftime("%Y-%m-%d"),
        "search_train_end_date": (first_oos - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        "oos_start_year": int(first_oos.year),
        "oos_end_year": int(last_oos.year),
        "oos_start_date": first_oos.strftime("%Y-%m-%d"),
        "oos_end_date": (last_oos + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d"),
        "latest_data_date": (last_oos + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d"),
        "min_train_years": max(1, int(math.ceil(train_months / 12.0))),
        "train_window_months": train_months,
        RAW_UNIVERSE_REQUIRED_MIN_ROWS_FIELD: int(get_breakout_optimizer_required_min_rows()),
    })
    return policy


def _build_runtime_contract(
    *,
    root: Path,
    args,
    settings,
    pit_contract,
    baseline_contract,
    base_policy,
    arm_name: str,
    ranking_enabled: bool,
) -> dict[str, Any]:
    score_hash = compute_file_sha256(pit_contract.score_path)
    manifest_hash = compute_file_sha256(pit_contract.manifest_path)
    audit_hash = compute_file_sha256(pit_contract.audit_path)
    search_space_snapshot = {
        "strategy_search_space": BREAKOUT_OPTIMIZER_SEARCH_SPACE,
        "fixed_tp_percent": OPTIMIZER_FIXED_TP_PERCENT,
    }
    effective_policy = build_optimizer_effective_policy_fingerprint(base_policy)
    baseline_meta = dict(baseline_contract["meta"])
    training_score_coverage = _build_training_score_coverage_contract(
        baseline_contract=baseline_contract, pit_contract=pit_contract
    )
    contract = {
        "schema_version": SCHEMA_VERSION,
        "adaptation_mode": "rolling_selection_2x2_single_adapted_optimizer",
        "optimization_arm": str(arm_name),
        "result_interpretation": ADAPTATION_STATUS,
        "dataset": str(args.dataset),
        "dataset_identity": build_source_data_inventory(root, args.dataset),
        "comparison_period": {
            "start": str(pit_contract.available_from),
            "end": str(pit_contract.available_through),
        },
        "rolling_training_period": str(
            baseline_contract["summary"].get("selection_period") or ""
        ),
        "training_score_coverage": training_score_coverage,
        "rolling_policy": {
            "window_mode": str(baseline_meta["window_mode"]),
            "first_oos_date": str(baseline_meta["first_oos_date"]),
            "last_oos_date": str(baseline_meta["last_oos_date"]),
            "train_window_months": int(baseline_meta["train_window_months"]),
            "oos_horizon_months": int(baseline_meta["oos_horizon_months"]),
            "trials_per_fold": int(args.trials_per_fold),
            "trials_per_fold_source": (
                "config.training_policy.OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT"
                if int(args.trials_per_fold)
                == int(settings.strategy_adapt_trials_per_fold)
                else "explicit_cli_override"
            ),
            "baseline_artifact_trials_per_fold": int(
                baseline_meta["trials_per_fold"]
            ),
            "trial_count_match_required": False,
            "folds": int(baseline_contract["summary"].get("folds", 0) or 0),
            "active_param_policy": str(
                baseline_meta.get("active_param_policy") or ""
            ),
        },
        "baseline_active_params": {
            "path": project_relative_display_path(
                baseline_contract["path"], project_root=root
            ),
            "sha256": str(baseline_contract["sha256"]),
            "selector": str(baseline_contract["payload"].get("selector") or ""),
        },
        "optimizer_policy": "base-finalist-best",
        "objective": "split_train_romd",
        "optimizer_effective_policy": effective_policy,
        "search_space_identity_sha256": _canonical_json_sha256(
            search_space_snapshot
        ),
        "search_space_snapshot": search_space_snapshot,
        "seed": int(settings.seed),
        "pit_identity": {
            "score_csv_sha256": score_hash,
            "manifest_sha256": manifest_hash,
            "audit_sha256": audit_hash,
            "score_path": project_relative_display_path(
                pit_contract.score_path, project_root=root
            ),
            "manifest_path": project_relative_display_path(
                pit_contract.manifest_path, project_root=root
            ),
            "audit_path": project_relative_display_path(
                pit_contract.audit_path, project_root=root
            ),
        },
        "model_identity": {
            "filter_id": str(args.filter_id),
            "architecture": str(args.model_architecture),
            "experiment_profile": str(args.experiment_profile),
            "continuous_target_id": str(pit_contract.continuous_target_id),
        },
        "ranking_mode": (
            "breakout_quality_score_desc"
            if bool(ranking_enabled)
            else "existing_buy_sort"
        ),
        "score_source": SCORE_SOURCE_SELECTION_POINT_IN_TIME,
        "fixed_runtime": {
            "use_breakout_quality_ranking": bool(ranking_enabled),
            "use_breakout_quality_filter": False,
            "fixed_risk": float(args.fixed_risk),
            "max_position_cap_pct": float(args.max_position_cap_pct),
            "max_positions": int(args.max_positions),
            "rotation": str(args.rotation) == "on",
            "tp_percent_policy": {
                "fixed_value": OPTIMIZER_FIXED_TP_PERCENT,
                "searched": OPTIMIZER_FIXED_TP_PERCENT is None,
            },
        },
        "runtime_restrictions": {
            "ranking_switch_searched": False,
            "score_weight_searched": False,
            "score_threshold_searched": False,
            "future_target_used_for_runtime": False,
            "future_target_used_for_objective": False,
            "oos_used_for_fitting": False,
            "final_selection_refit_executed": False,
            "pre_pit_missing_scores_use_existing_buy_sort_fallback": True,
            "score_gaps_inside_pit_period_allowed": False,
        },
    }
    contract["runtime_identity_sha256"] = _canonical_json_sha256(contract)
    return contract


def _build_current_pair_runtime_identity(
    *, root: Path, args, pit_contract, baseline_contract
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "comparison_design": "selection_point_in_time_active_param_replay",
        "dataset": str(args.dataset),
        "baseline_active_params_sha256": str(baseline_contract["sha256"]),
        "pit_identity": {
            "score_csv_sha256": compute_file_sha256(pit_contract.score_path),
            "manifest_sha256": compute_file_sha256(pit_contract.manifest_path),
            "audit_sha256": compute_file_sha256(pit_contract.audit_path),
        },
        "model_identity": {
            "filter_id": str(args.filter_id),
            "architecture": str(args.model_architecture),
            "experiment_profile": str(args.experiment_profile),
            "continuous_target_id": str(pit_contract.continuous_target_id),
        },
        "execution": {
            "max_positions": int(args.max_positions),
            "rotation": str(args.rotation) == "on",
            "fixed_risk": float(args.fixed_risk),
            "max_position_cap_pct": float(args.max_position_cap_pct),
            "score_source": SCORE_SOURCE_SELECTION_POINT_IN_TIME,
        },
        "comparison_period": {
            "start": str(pit_contract.available_from),
            "end": str(pit_contract.available_through),
        },
    }
    return _canonical_json_sha256(payload)


def _canonical_execution_argv(args) -> list[str]:
    argv = [
        "python",
        "apps/breakout_quality.py",
        "strategy-adapt",
        "--dataset", str(args.dataset),
        "--filter-id", str(args.filter_id),
        "--model-architecture", str(args.model_architecture),
        "--experiment-profile", str(args.experiment_profile),
        "--param-policy", str(args.param_policy),
        "--trials-per-fold", str(int(args.trials_per_fold)),
        "--max-positions", str(int(args.max_positions)),
        "--rotation", str(args.rotation),
        "--fixed-risk", str(float(args.fixed_risk)),
        "--max-position-cap-pct", str(float(args.max_position_cap_pct)),
    ]
    if bool(args.quiet):
        argv.append("--quiet")
    return argv


def _outer_rolling_argv(*, args, baseline_contract: dict[str, Any]) -> list[str]:
    meta = dict(baseline_contract["meta"])
    return [
        "--outer-first-oos-date", str(meta["first_oos_date"]),
        "--outer-last-oos-date", str(meta["last_oos_date"]),
        "--outer-train-window-months", str(int(meta["train_window_months"])),
        "--outer-oos-months", str(int(meta["oos_horizon_months"])),
        "--trials", str(int(args.trials_per_fold)),
    ]


def _fixed_strategy_param_overrides(
    args, *, ranking_enabled: bool
) -> dict[str, Any]:
    return {
        "use_breakout_quality_filter": False,
        "use_breakout_quality_ranking": bool(ranking_enabled),
        "breakout_quality_filter_id": str(args.filter_id),
        "fixed_risk": float(args.fixed_risk),
        "max_position_cap_pct": float(args.max_position_cap_pct),
    }


def _load_json_mapping(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None




def _validate_selection_pit_runtime_artifacts(*, pit_contract, args) -> dict[str, Path]:
    """Validate the artifacts actually read by Selection PIT ranking runtime.

    Selection PIT ranking is score-table based.  It does not load the canonical
    full-model checkpoint/manifest used by the binary filter runtime.  The
    contract loader has already verified hashes, sizes, identities, coverage,
    continuous-target binding, and the model-validation audit.  This preflight
    keeps the runtime boundary explicit without inventing a nonexistent
    profile-root ``manifest.json`` requirement.
    """

    expected_identity = {
        "filter_id": str(args.filter_id),
        "model_architecture": str(args.model_architecture),
        "experiment_profile": str(args.experiment_profile),
    }
    actual_identity = {
        "filter_id": str(getattr(pit_contract, "filter_id", "")),
        "model_architecture": str(
            getattr(pit_contract, "model_architecture", "")
        ),
        "experiment_profile": str(
            getattr(pit_contract, "experiment_profile", "")
        ),
    }
    if actual_identity != expected_identity:
        raise ValueError(
            "Score Adapted Selection PIT runtime identity不一致；禁止開始rolling trials："
            f"expected={expected_identity}, actual={actual_identity}"
        )

    artifacts = {
        "selection_pit_manifest": Path(pit_contract.manifest_path).resolve(),
        "selection_pit_scores": Path(pit_contract.score_path).resolve(),
        "selection_pit_audit": Path(pit_contract.audit_path).resolve(),
    }
    for label, path in artifacts.items():
        if not path.is_file():
            raise FileNotFoundError(
                "Score Adapted Selection PIT runtime工件不存在；"
                f"禁止開始rolling trials：artifact={label}, path={path}"
            )

    gate = dict(getattr(pit_contract, "model_validation_gate", {}) or {})
    if str(gate.get("status") or "") != "PASS":
        raise ValueError(
            "Score Adapted Selection PIT模型驗證未通過；禁止開始rolling trials："
            f"status={gate.get('status')!r}"
        )
    return artifacts

def _completed_adapted_search_is_compatible(
    *,
    existing_preflight: dict[str, Any] | None,
    runtime_contract: dict[str, Any],
    adapted_params_path: Path,
    baseline_contract: dict[str, Any],
) -> bool:
    if not isinstance(existing_preflight, dict):
        return False
    if str(existing_preflight.get("runtime_identity_sha256") or "") != str(
        runtime_contract["runtime_identity_sha256"]
    ):
        return False
    payload = _load_json_mapping(adapted_params_path)
    if payload is None:
        return False
    meta = dict(payload.get("meta") or {})
    baseline_meta = dict(baseline_contract.get("meta") or {})
    for key in (
        "first_oos_date",
        "last_oos_date",
        "train_window_months",
        "oos_horizon_months",
    ):
        if str(meta.get(key)) != str(baseline_meta.get(key)):
            return False
    if int(meta.get("trials_per_fold", 0) or 0) != int(
        dict(runtime_contract.get("rolling_policy") or {}).get(
            "trials_per_fold", 0
        )
        or 0
    ):
        return False
    if int(dict(payload.get("summary") or {}).get("folds", 0) or 0) != int(
        dict(baseline_contract.get("summary") or {}).get("folds", 0) or 0
    ):
        return False
    return bool(dict(payload.get("params_ensemble_by_effective_date") or {}))


def _materialize_adapted_param_artifacts(
    *,
    adapted_models_dir: Path,
    fixed_strategy_param_overrides: dict[str, Any],
) -> list[Path]:
    updated_paths: list[Path] = []
    for artifact_path in sorted(adapted_models_dir.glob("*.json")):
        payload = _load_json_mapping(artifact_path)
        if payload is None:
            continue
        if not any(
            payload.get(key)
            for key in (
                "params_by_oos_year",
                "params_by_effective_date",
                "params_ensemble_by_effective_date",
                "params_ensemble",
            )
        ):
            continue
        materialized = materialize_fixed_strategy_param_overrides_in_active_param_payload(
            payload,
            fixed_strategy_param_overrides,
        )
        _write_json(artifact_path, materialized)
        updated_paths.append(artifact_path)
    return updated_paths


def _optimizer_session_spec(
    *,
    output_dir: Path,
    args,
    runtime_contract,
    arm_name: str,
    ranking_enabled: bool,
) -> dict[str, Any]:
    return {
        "output_dir": str(
            (output_dir / arm_name / "optimizer_runtime" / "sessions").resolve()
        ),
        "fixed_strategy_param_overrides": _fixed_strategy_param_overrides(
            args, ranking_enabled=ranking_enabled
        ),
        "runtime_context_spec": {
            "module": "filters.breakout_quality.runtime",
            "callable": "breakout_quality_ranking_source_context",
            "kwargs": {
                "score_source": SCORE_SOURCE_SELECTION_POINT_IN_TIME,
                "model_architecture": str(args.model_architecture),
                "experiment_profile": str(args.experiment_profile),
            },
        },
        "runtime_cache_identity": str(runtime_contract["runtime_identity_sha256"]),
        "optimizer_fixed_tp_percent": OPTIMIZER_FIXED_TP_PERCENT,
        "train_max_positions": int(args.max_positions),
        "train_enable_rotation": str(args.rotation) == "on",
    }


def _validate_adapted_rolling_params(
    *,
    path: Path,
    baseline_contract: dict[str, Any],
    runtime_contract: dict[str, Any],
    args,
    arm_name: str,
    ranking_enabled: bool,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"{arm_name} rolling optimizer未產生base-finalist-best工件: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    meta = dict(payload.get("meta") or {})
    baseline_meta = dict(baseline_contract["meta"])
    for key in (
        "first_oos_date",
        "last_oos_date",
        "train_window_months",
        "oos_horizon_months",
    ):
        if str(meta.get(key)) != str(baseline_meta.get(key)):
            raise ValueError(
                f"{arm_name}與Baseline fold schedule不一致："
                f"{key}: adapted={meta.get(key)!r}, baseline={baseline_meta.get(key)!r}"
            )
    if int(meta.get("trials_per_fold", 0) or 0) != int(args.trials_per_fold):
        raise ValueError(
            f"{arm_name}工件未採用目前requested trials／fold："
            f"artifact={meta.get('trials_per_fold')!r}, "
            f"requested={int(args.trials_per_fold)}"
        )
    if int(dict(payload.get("summary") or {}).get("folds", 0) or 0) != int(
        baseline_contract["summary"].get("folds", 0) or 0
    ):
        raise ValueError(f"{arm_name} fold數與Baseline不一致")
    members = []
    for effective_date, raw_members in dict(
        payload.get("params_ensemble_by_effective_date") or {}
    ).items():
        for raw_member in list(raw_members or []):
            params = dict((raw_member or {}).get("params") or {})
            members.append((str(effective_date), params))
    if not members:
        raise ValueError(f"{arm_name}工件沒有active-param members")
    for effective_date, params in members:
        expected = {
            "use_breakout_quality_ranking": bool(ranking_enabled),
            "use_breakout_quality_filter": False,
            "breakout_quality_filter_id": str(args.filter_id),
            "fixed_risk": float(args.fixed_risk),
            "max_position_cap_pct": float(args.max_position_cap_pct),
            "tp_percent": OPTIMIZER_FIXED_TP_PERCENT,
        }
        for key, expected_value in expected.items():
            actual = params.get(key)
            if isinstance(expected_value, float):
                matched = actual is not None and math.isclose(
                    float(actual), expected_value, rel_tol=0.0, abs_tol=1e-12
                )
            else:
                matched = actual == expected_value
            if not matched:
                raise ValueError(
                    f"{arm_name}固定契約未落入active params："
                    f"effective_date={effective_date}, {key}={actual!r}, "
                    f"expected={expected_value!r}"
                )
    payload["breakout_quality_adaptation"] = {
        "mode": "rolling_selection_2x2_single_adapted_optimizer",
        "optimization_arm": str(arm_name),
        "result_interpretation": ADAPTATION_STATUS,
        "runtime_identity_sha256": str(runtime_contract["runtime_identity_sha256"]),
        "score_source": SCORE_SOURCE_SELECTION_POINT_IN_TIME,
        "use_breakout_quality_ranking": bool(ranking_enabled),
        "future_target_used_for_runtime": False,
        "final_selection_refit": False,
    }
    _write_json(path, payload)
    return payload


def _run_rolling_optimizer_arm(
    *,
    root: Path,
    args,
    settings,
    baseline_contract: dict[str, Any],
    base_policy: dict[str, Any],
    runtime_contract: dict[str, Any],
    output_dir: Path,
    arm_name: str,
    arm_label: str,
    ranking_enabled: bool,
) -> dict[str, Any]:
    arm_dir = output_dir / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    preflight_path = arm_dir / "rolling_preflight.json"
    existing_preflight = _load_json_mapping(preflight_path)
    _write_json(
        preflight_path,
        {
            **runtime_contract,
            "status": "ROLLING_ADAPTATION_ARM_PREFLIGHT_PASS",
            "created_at": get_taipei_now().isoformat(),
        },
    )

    models_dir = arm_dir / "active_params"
    models_dir.mkdir(parents=True, exist_ok=True)
    optimizer_output_dir = arm_dir / "optimizer_runtime"
    outer_environ = dict(os.environ)
    outer_environ["V16_MODELS_DIR"] = str(models_dir.resolve())
    outer_environ["OPTIMIZER_OUTER_ROLLING_STUDY_STORAGE"] = "sqlite"
    outer_environ["OPTIMIZER_ROLLING_RESUME_EXISTING_STUDIES"] = "1"
    params_path = models_dir / "roos_base_best.json"
    search_reused = _completed_adapted_search_is_compatible(
        existing_preflight=existing_preflight,
        runtime_contract=runtime_contract,
        adapted_params_path=params_path,
        baseline_contract=baseline_contract,
    )
    if search_reused:
        print(
            f"\n[{arm_label}] 既有rolling搜尋與目前runtime identity一致；"
            "沿用已選參數並重新套用固定ranking契約，不重跑optimizer。"
        )
    else:
        session_spec = _optimizer_session_spec(
            output_dir=output_dir,
            args=args,
            runtime_contract=runtime_contract,
            arm_name=arm_name,
            ranking_enabled=ranking_enabled,
        )
        exit_code = run_outer_rolling_oos(
            argv=_outer_rolling_argv(
                args=args, baseline_contract=baseline_contract
            ),
            environ=outer_environ,
            project_root=str(root),
            output_dir=str(optimizer_output_dir),
            base_policy=base_policy,
            selected_data_dir=get_dataset_dir(str(root), args.dataset),
            dataset_label=str(args.dataset),
            load_all_raw_data=load_all_raw_data,
            optimizer_required_min_rows=get_breakout_optimizer_required_min_rows(),
            build_optimizer_session=build_optimizer_session,
            create_optimizer_study=create_optimizer_study,
            ensure_study_effective_policy_compatible=ensure_study_effective_policy_compatible,
            configure_optuna_logging=configure_optuna_logging,
            optimizer_seed=int(settings.seed),
            optimizer_session_spec=session_spec,
            default_trials=int(args.trials_per_fold),
            timing_mode=False,
        )
        if int(exit_code) != 0:
            raise RuntimeError(
                f"{arm_label} rolling optimizer失敗：returncode={exit_code}"
            )

    fixed_overrides = _fixed_strategy_param_overrides(
        args, ranking_enabled=ranking_enabled
    )
    materialized_paths = _materialize_adapted_param_artifacts(
        adapted_models_dir=models_dir,
        fixed_strategy_param_overrides=fixed_overrides,
    )
    if params_path not in materialized_paths:
        raise RuntimeError(
            f"{arm_label} active-param工件無法套用固定ranking契約："
            f"path={params_path}"
        )
    params_payload = _validate_adapted_rolling_params(
        path=params_path,
        baseline_contract=baseline_contract,
        runtime_contract=runtime_contract,
        args=args,
        arm_name=arm_label,
        ranking_enabled=ranking_enabled,
    )
    summary_path = arm_dir / "rolling_optimizer_summary.json"
    folds = int(dict(params_payload.get("summary") or {}).get("folds", 0) or 0)
    summary = {
        "mode": "rolling_selection_2x2_single_adapted_optimizer",
        "optimization_arm": str(arm_name),
        "arm_label": str(arm_label),
        "ranking_enabled": bool(ranking_enabled),
        "result_interpretation": ADAPTATION_STATUS,
        "folds": folds,
        "trials_per_fold": int(args.trials_per_fold),
        "total_requested_trials": int(args.trials_per_fold) * folds,
        "trials_per_fold_source": str(
            runtime_contract["rolling_policy"]["trials_per_fold_source"]
        ),
        "same_fold_schedule_as_baseline": True,
        "optimizer_search_reused": bool(search_reused),
        "fixed_contract_materialized_files": [
            project_relative_display_path(item, project_root=root)
            for item in materialized_paths
        ],
        "seed": int(settings.seed),
        "fixed_tp_percent": OPTIMIZER_FIXED_TP_PERCENT,
        "runtime_identity_sha256": runtime_contract["runtime_identity_sha256"],
        "final_selection_refit_executed": False,
        "formal_oos_executed": False,
    }
    _write_json(summary_path, summary)
    return {
        "arm_name": arm_name,
        "arm_label": arm_label,
        "ranking_enabled": bool(ranking_enabled),
        "dir": arm_dir,
        "preflight_path": preflight_path,
        "models_dir": models_dir,
        "params_path": params_path,
        "params_payload": params_payload,
        "optimizer_summary_path": summary_path,
        "optimizer_summary": summary,
        "search_reused": bool(search_reused),
        "runtime_contract": runtime_contract,
        "materialized_paths": materialized_paths,
    }


def _active_param_values(path: str | Path) -> dict[str, list[Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values: dict[str, list[Any]] = {}
    ensemble = dict(payload.get("params_ensemble_by_effective_date") or {})
    if ensemble:
        raw_param_sets = [
            dict((member or {}).get("params") or {})
            for members in ensemble.values()
            for member in list(members or [])
        ]
    else:
        raw_param_sets = [
            dict(raw or {})
            for raw in dict(payload.get("params_by_effective_date") or {}).values()
        ]
    for params in raw_param_sets:
        for key, value in params.items():
            values.setdefault(str(key), []).append(value)
    return values


def _build_param_comparison_rows(
    *, baseline_params_path: str | Path, adapted_params_path: str | Path
) -> list[dict[str, Any]]:
    baseline_values = _active_param_values(baseline_params_path)
    adapted_values = _active_param_values(adapted_params_path)
    searchable_fields = set(BREAKOUT_OPTIMIZER_SEARCH_SPACE)
    searchable_fields.update(("tp_percent", "fixed_risk", "max_position_cap_pct"))
    rows: list[dict[str, Any]] = []
    for field_name in sorted(searchable_fields):
        left = list(baseline_values.get(field_name) or [])
        right = list(adapted_values.get(field_name) or [])
        if not left or not right:
            continue
        all_values = left + right
        numeric = all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            for value in all_values
        )
        if numeric:
            left_num = [float(value) for value in left]
            right_num = [float(value) for value in right]
            rows.append({
                "parameter": field_name,
                "comparison_type": "numeric_range",
                "baseline_min": min(left_num),
                "baseline_median": float(pd.Series(left_num).median()),
                "baseline_max": max(left_num),
                "adapted_min": min(right_num),
                "adapted_median": float(pd.Series(right_num).median()),
                "adapted_max": max(right_num),
                "median_delta": float(pd.Series(right_num).median() - pd.Series(left_num).median()),
            })
            continue
        rows.append({
            "parameter": field_name,
            "comparison_type": "categorical_set",
            "baseline_values": sorted({json.dumps(value, ensure_ascii=False, sort_keys=True) for value in left}),
            "adapted_values": sorted({json.dumps(value, ensure_ascii=False, sort_keys=True) for value in right}),
        })
    return rows


def _param_comparison_table_rows(result: dict[str, Any]) -> list[tuple[str, str, str, str, str]]:
    rows = []
    for row in list(result.get("parameter_comparison") or []):
        if row.get("comparison_type") == "numeric_range":
            rows.append((
                str(row.get("parameter")),
                _fmt(row.get("baseline_median")),
                f"{_fmt(row.get('baseline_min'))}～{_fmt(row.get('baseline_max'))}",
                _fmt(row.get("adapted_median")),
                f"{_fmt(row.get('adapted_min'))}～{_fmt(row.get('adapted_max'))}",
            ))
        else:
            baseline = ", ".join(json.loads(value).__str__() for value in row.get("baseline_values") or [])
            adapted = ", ".join(json.loads(value).__str__() for value in row.get("adapted_values") or [])
            rows.append((str(row.get("parameter")), baseline, baseline, adapted, adapted))
    return rows

def _resolve_controlled_arm_params(
    *,
    params_path: Path,
    args,
    ranking_enabled: bool,
) -> tuple[str, Any]:
    source = _load_param_source(params_path)
    _validate_requested_param_policy(source, args.param_policy)
    (
        source_kind,
        no_sort_params,
        score_params,
        _no_sort_payload,
        _score_payload,
        _ensemble_policy,
    ) = _build_controlled_param_source_pair(
        source,
        filter_id=str(args.filter_id),
        threshold=float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD),
        fixed_risk=float(args.fixed_risk),
        max_position_cap_pct=float(args.max_position_cap_pct),
        comparison_mode=COMPARISON_MODE_SCORE_RANKING,
    )
    if source_kind != "rolling_active_param_ensemble":
        raise ValueError("2×2參數適應比較只接受rolling active-param ensemble")
    return source_kind, score_params if ranking_enabled else no_sort_params


def _run_optimized_arm_replay(
    *,
    root: Path,
    args,
    pit_contract,
    params_path: Path,
    arm_name: str,
    ranking_enabled: bool,
) -> dict[str, Any]:
    source_kind, params = _resolve_controlled_arm_params(
        params_path=params_path,
        args=args,
        ranking_enabled=ranking_enabled,
    )
    replay_counts: dict[str, Any] = {}
    ranking_source = None
    if ranking_enabled:
        ranking_source = {
            "score_source": SCORE_SOURCE_SELECTION_POINT_IN_TIME,
            "model_architecture": args.model_architecture,
            "experiment_profile": args.experiment_profile,
        }
    payload = _run_scenario(
        name=arm_name,
        data_dir=Path(get_dataset_dir(str(root), args.dataset)).resolve(),
        param_source_kind=source_kind,
        params=params,
        start_date=pit_contract.available_from,
        end_date=pit_contract.available_through,
        max_positions=int(args.max_positions),
        enable_rotation=str(args.rotation) == "on",
        quiet=bool(args.quiet),
        replay_counts=replay_counts,
        ranking_source=ranking_source,
    )
    return {
        "payload": payload,
        "summary": _scenario_summary(payload),
        "replay_counts": replay_counts,
        "orderable": _flatten_candidate_replay_rows(replay_counts, "orderable_rows"),
        "selected": _flatten_selected_buy_rows(payload["trade_history"]),
    }


def _write_optimized_arm_replay_outputs(
    *,
    output_dir: Path,
    prefix: str,
    replay: dict[str, Any],
    orderable_joined: pd.DataFrame,
    selected_joined: pd.DataFrame,
) -> dict[str, Path]:
    payload = replay["payload"]
    paths = {
        "equity": output_dir / f"{prefix}_equity.csv",
        "trades": output_dir / f"{prefix}_trades.csv",
        "daily_capacity": output_dir / f"{prefix}_daily_capacity.csv",
        "orderable_diagnostics": output_dir / f"{prefix}_orderable_target_diagnostics.csv",
        "selected_diagnostics": output_dir / f"{prefix}_selected_target_diagnostics.csv",
        "yearly_returns": output_dir / f"{prefix}_yearly_returns.csv",
    }
    payload["equity_curve"].to_csv(paths["equity"], index=False, encoding="utf-8-sig")
    payload["trade_history"].to_csv(paths["trades"], index=False, encoding="utf-8-sig")
    pd.DataFrame(payload["profile"].get("portfolio_capacity_rows") or []).to_csv(
        paths["daily_capacity"], index=False, encoding="utf-8-sig"
    )
    orderable_joined.to_csv(
        paths["orderable_diagnostics"], index=False, encoding="utf-8-sig"
    )
    selected_joined.to_csv(
        paths["selected_diagnostics"], index=False, encoding="utf-8-sig"
    )
    return paths



def _run_four_way_comparison(
    *,
    root: Path,
    args,
    pit_contract,
    current_payload: dict[str, Any],
    baseline_contract: dict[str, Any],
    adapted_arm: dict[str, Any],
    current_pair_dir: Path,
    output_dir: Path,
    rolling_validation: dict[str, Any],
) -> dict[str, Any]:
    metadata = dict(current_payload["metadata"])
    comparison_dir = Path(current_pair_dir).resolve()
    required = {
        "baseline_equity": comparison_dir / "no_filter_equity.csv",
        "baseline_trades": comparison_dir / "no_filter_trades.csv",
        "baseline_capacity": comparison_dir / "no_filter_daily_capacity.csv",
        "baseline_selected": comparison_dir / "no_filter_selected_target_diagnostics.csv",
    }
    missing = [str(item) for item in required.values() if not item.is_file()]
    if missing:
        raise FileNotFoundError("四組比較缺少既有Baseline工件: " + ", ".join(missing))

    param_only_replay = _run_optimized_arm_replay(
        root=root,
        args=args,
        pit_contract=pit_contract,
        params_path=adapted_arm["params_path"],
        arm_name="param_only",
        ranking_enabled=False,
    )
    adapted_replay = _run_optimized_arm_replay(
        root=root,
        args=args,
        pit_contract=pit_contract,
        params_path=adapted_arm["params_path"],
        arm_name="adapted",
        ranking_enabled=True,
    )
    baseline = dict(current_payload["no_filter"])
    sort_only = dict(current_payload["score_ranking"])
    param_only = dict(param_only_replay["summary"])
    adapted = dict(adapted_replay["summary"])

    baseline_equity = pd.read_csv(required["baseline_equity"], encoding="utf-8-sig")
    benchmark_reference = {
        "benchmark_return_pct": baseline["benchmark_return_pct"],
        "benchmark_max_drawdown_pct": baseline["benchmark_max_drawdown_pct"],
        "benchmark_annual_return_pct": baseline["benchmark_annual_return_pct"],
        "equity_curve": baseline_equity,
    }
    _assert_shared_benchmark(benchmark_reference, param_only_replay["payload"])
    _assert_shared_benchmark(benchmark_reference, adapted_replay["payload"])

    lookup = _selection_target_lookup(
        root=root,
        filter_id=args.filter_id,
        architecture=args.model_architecture,
        profile=args.experiment_profile,
    )
    param_only_diag, param_only_orderable, param_only_selected = (
        _strategy_selection_diagnostics(
            orderable=param_only_replay["orderable"],
            selected=param_only_replay["selected"],
            lookup=lookup,
        )
    )
    adapted_diag, adapted_orderable, adapted_selected = (
        _strategy_selection_diagnostics(
            orderable=adapted_replay["orderable"],
            selected=adapted_replay["selected"],
            lookup=lookup,
        )
    )
    existing_diagnostics = dict(current_payload.get("selection_diagnostics") or {})
    baseline_diag = dict(existing_diagnostics.get("no_filter") or {})
    sort_diag = dict(existing_diagnostics.get("score_ranking") or {})

    param_only_paths = _write_optimized_arm_replay_outputs(
        output_dir=output_dir,
        prefix="param_only",
        replay=param_only_replay,
        orderable_joined=param_only_orderable,
        selected_joined=param_only_selected,
    )
    adapted_paths = _write_optimized_arm_replay_outputs(
        output_dir=output_dir,
        prefix="adapted",
        replay=adapted_replay,
        orderable_joined=adapted_orderable,
        selected_joined=adapted_selected,
    )

    param_only_yearly = _yearly_frame(
        param_only_replay["payload"]["profile"], "param_only"
    )
    adapted_yearly = _yearly_frame(adapted_replay["payload"]["profile"], "adapted")
    param_only_yearly.to_csv(
        param_only_paths["yearly_returns"], index=False, encoding="utf-8-sig"
    )
    adapted_yearly.to_csv(
        adapted_paths["yearly_returns"], index=False, encoding="utf-8-sig"
    )
    current_yearly = pd.DataFrame(current_payload.get("yearly") or [])
    yearly_keys = [
        key
        for key in ("year", "is_full_year", "start_date", "end_date")
        if key in current_yearly.columns
        and key in param_only_yearly.columns
        and key in adapted_yearly.columns
    ]
    if not yearly_keys:
        yearly_keys = ["year"]
    four_way_yearly = current_yearly.merge(
        param_only_yearly, on=yearly_keys, how="outer", validate="one_to_one"
    ).merge(
        adapted_yearly, on=yearly_keys, how="outer", validate="one_to_one"
    ).sort_values("year").reset_index(drop=True)
    four_way_yearly["old_params_ranking_delta_pct"] = (
        four_way_yearly["score_ranking_return_pct"]
        - four_way_yearly["no_filter_return_pct"]
    )
    four_way_yearly["adapted_params_ranking_delta_pct"] = (
        four_way_yearly["adapted_return_pct"]
        - four_way_yearly["param_only_return_pct"]
    )
    four_way_yearly["old_ranking_param_delta_pct"] = (
        four_way_yearly["param_only_return_pct"]
        - four_way_yearly["no_filter_return_pct"]
    )
    four_way_yearly["score_ranking_param_delta_pct"] = (
        four_way_yearly["adapted_return_pct"]
        - four_way_yearly["score_ranking_return_pct"]
    )
    yearly_path = output_dir / "strategy_adaptation_yearly_returns.csv"
    four_way_yearly.to_csv(yearly_path, index=False, encoding="utf-8-sig")

    adapted_params_capture_dir = output_dir / "adapted_params_ranking_capture"
    adapted_params_capture = build_score_ranking_capture_audit(
        metadata={
            **metadata,
            "comparison_design": "same_adapted_params_ranking_counterfactual",
        },
        baseline_summary=param_only,
        score_sort_summary=adapted,
        baseline_trade_history=param_only_replay["payload"]["trade_history"],
        score_sort_trade_history=adapted_replay["payload"]["trade_history"],
        baseline_selected_target_diagnostics=param_only_selected,
        score_sort_selected_target_diagnostics=adapted_selected,
        selection_diagnostics={
            "score_ranking_minus_no_filter": _delta(adapted_diag, param_only_diag)
        },
        baseline_daily_capacity=pd.DataFrame(
            param_only_replay["payload"]["profile"].get("portfolio_capacity_rows") or []
        ),
        score_sort_daily_capacity=pd.DataFrame(
            adapted_replay["payload"]["profile"].get("portfolio_capacity_rows") or []
        ),
    )
    adapted_params_capture_payload = write_score_ranking_capture_audit_outputs(
        result=adapted_params_capture,
        output_dir=adapted_params_capture_dir,
    )

    parameter_comparison = _build_param_comparison_rows(
        baseline_params_path=baseline_contract["path"],
        adapted_params_path=adapted_arm["params_path"],
    )
    adapted_params_path = adapted_arm["params_path"]
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": ADAPTATION_STATUS,
        "comparison_design": "ranking_parameter_2x2_single_adapted_optimizer",
        "metadata": {
            **metadata,
            "result_interpretation": ADAPTATION_STATUS,
            "adapted_params_path": project_relative_display_path(
                adapted_params_path, project_root=root
            ),
            "adapted_params_sha256": compute_file_sha256(adapted_params_path),
            "future_target_used_for_runtime": False,
            "oos_generalization_claimed": False,
            "final_selection_refit_executed": False,
            "optimizer_training_arms": ["score_adapted"],
        },
        "baseline": baseline,
        "sort_only": sort_only,
        "param_only": param_only,
        "adapted": adapted,
        "old_params_ranking_delta": _delta(sort_only, baseline),
        "adapted_params_ranking_delta": _delta(adapted, param_only),
        "old_ranking_param_delta": _delta(param_only, baseline),
        "score_ranking_param_delta": _delta(adapted, sort_only),
        "overall_delta": _delta(adapted, baseline),
        "selection_diagnostics": {
            "baseline": baseline_diag,
            "sort_only": sort_diag,
            "param_only": param_only_diag,
            "adapted": adapted_diag,
            "old_params_ranking_delta": _delta(sort_diag, baseline_diag),
            "adapted_params_ranking_delta": _delta(adapted_diag, param_only_diag),
            "old_ranking_param_delta": _delta(param_only_diag, baseline_diag),
            "score_ranking_param_delta": _delta(adapted_diag, sort_diag),
            "future_target_join_stage": "post_replay_offline_diagnostic_only",
            "future_target_used_for_runtime_sort": False,
        },
        "current_capture_audit": current_payload.get("score_ranking_capture_audit"),
        "adapted_params_capture_audit": adapted_params_capture_payload,
        "yearly": four_way_yearly.to_dict("records"),
        "adapted_active_params": {
            "path": project_relative_display_path(adapted_params_path, project_root=root),
            "sha256": compute_file_sha256(adapted_params_path),
        },
        "parameter_comparison": parameter_comparison,
        "rolling_validation": dict(rolling_validation),
        "replay_artifacts": {
            "param_only": {
                key: project_relative_display_path(value, project_root=root)
                for key, value in param_only_paths.items()
            },
            "adapted": {
                key: project_relative_display_path(value, project_root=root)
                for key, value in adapted_paths.items()
            },
            "adapted_params_capture_dir": project_relative_display_path(
                adapted_params_capture_dir, project_root=root
            ),
        },
    }
    _write_json(output_dir / "strategy_adaptation_comparison.json", result)
    (output_dir / "strategy_adaptation_comparison.md").write_text(
        _render_four_way_markdown(result), encoding="utf-8"
    )
    print("\n" + _render_four_way_console(result))
    if not compact_console_enabled():
        print("\n" + render_capture_audit_console(adapted_params_capture))
    return result




def _metric_rows(result: dict[str, Any]):
    rows = (
        ("淨總報酬", "total_return_pct", "%"),
        ("最大回撤", "max_drawdown_pct", "%"),
        ("Return／MDD", "return_over_max_drawdown", ""),
        ("年化報酬", "annual_return_pct", "%"),
        ("Log R²", "log_r_squared", ""),
        ("月勝率", "monthly_win_rate_pct", "%"),
        ("交易數", "trade_count", ""),
        ("勝率", "win_rate_pct", "%"),
        ("Payoff", "payoff_ratio", ""),
        ("EV", "expected_value_r", " R"),
        ("平均曝險", "avg_exposure_pct", "%"),
        ("平均候選供給", "avg_orderable_candidates", ""),
        ("候選不足日", "candidate_supply_gap_days", " 日"),
        ("平均預留金額", "avg_reserved_total", ""),
        ("平均投入金額", "avg_invested_total", ""),
        ("投入／預留比", "avg_invested_vs_reserved_pct", "%"),
        ("初始停損距離", "avg_stop_distance_pct", "%"),
        ("平均 Realized R", "avg_realized_r", " R"),
        ("平均 Target R", "avg_target_r", " R"),
        ("Aggregate capture", "aggregate_target_capture_ratio", ""),
        ("Median capture", "median_target_capture_ratio", ""),
        ("Target ≥ 0.5R capture", "target_ge_0_5_capture_ratio", ""),
    )
    current_capture = dict(result.get("current_capture_audit") or {})
    adapted_capture = dict(result.get("adapted_params_capture_audit") or {})
    enriched = {
        "baseline": {
            **dict(result.get("baseline") or {}),
            **dict(current_capture.get("baseline") or {}),
        },
        "sort_only": {
            **dict(result.get("sort_only") or {}),
            **dict(current_capture.get("score_sort") or {}),
        },
        "param_only": {
            **dict(result.get("param_only") or {}),
            **dict(adapted_capture.get("baseline") or {}),
        },
        "adapted": {
            **dict(result.get("adapted") or {}),
            **dict(adapted_capture.get("score_sort") or {}),
        },
    }
    return rows, enriched



def _fmt(value, unit=""):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(number):
        return "N/A"
    digits = 0 if unit == "" and abs(number) >= 1000 else 2
    return f"{number:.{digits}f}{unit}"


def _compact_metric_value(value: Any, *, unit: str = "", digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(number):
        return "N/A"
    return f"{number:.{int(digits)}f}{unit}"


def _pair_rows(
    values: dict[str, dict[str, Any]],
    specs: tuple[tuple[str, str, str, str, int], ...],
    *,
    left_key: str,
    right_key: str,
    use_color: bool,
) -> list[tuple[str, str, str, str, str]]:
    rows: list[tuple[str, str, str, str, str]] = []
    for label, key, unit, preference, digits in specs:
        left = values[left_key].get(key)
        right = values[right_key].get(key)
        try:
            delta = float(right) - float(left)
        except (TypeError, ValueError):
            delta = None
        if delta is not None and not math.isfinite(delta):
            delta = None
        signal = signal_for_delta(delta, preference=preference, warning_threshold=0.0)
        delta_text = "N/A" if delta is None else f"{delta:+.{int(digits)}f}{unit}"
        rows.append((
            label,
            _compact_metric_value(left, unit=unit, digits=digits),
            _compact_metric_value(right, unit=unit, digits=digits),
            terminal_signal(delta_text, signal, enabled=use_color),
            terminal_signal(signal_marker(signal), signal, enabled=use_color),
        ))
    return rows


def _pair_table(
    values: dict[str, dict[str, Any]],
    specs: tuple[tuple[str, str, str, str, int], ...],
    *,
    left_key: str,
    right_key: str,
    left_label: str,
    right_label: str,
    delta_label: str,
    use_color: bool,
) -> str:
    return render_table(
        ("指標", left_label, right_label, delta_label, "判讀"),
        _pair_rows(
            values,
            specs,
            left_key=left_key,
            right_key=right_key,
            use_color=use_color,
        ),
        alignments=("left", "right", "right", "right", "left"),
    )


def _yearly_pair_table(
    yearly_rows: list[dict[str, Any]],
    *,
    left_column: str,
    right_column: str,
    left_label: str,
    right_label: str,
    delta_label: str,
    use_color: bool,
) -> str:
    rows = []
    for row in yearly_rows:
        left = row.get(left_column)
        right = row.get(right_column)
        try:
            delta = float(right) - float(left)
        except (TypeError, ValueError):
            delta = None
        if delta is not None and not math.isfinite(delta):
            delta = None
        signal = signal_for_delta(delta, preference="higher", warning_threshold=0.0)
        rows.append((
            str(int(row["year"])),
            _compact_metric_value(left, unit="%", digits=2),
            _compact_metric_value(right, unit="%", digits=2),
            terminal_signal(
                "N/A" if delta is None else f"{delta:+.2f}pp",
                signal,
                enabled=use_color,
            ),
            terminal_signal(signal_marker(signal), signal, enabled=use_color),
        ))
    if not rows:
        return "無年度報酬資料。"
    return render_table(
        ("年度", left_label, right_label, delta_label, "判讀"),
        rows,
        alignments=("right", "right", "right", "right", "left"),
    )



def _capture_yearly_pairs(result: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    current = list(dict(result.get("current_capture_audit") or {}).get("yearly") or [])
    adapted = list(
        dict(result.get("adapted_params_capture_audit") or {}).get("yearly") or []
    )
    return {"current": current, "adapted": adapted}



def _capture_yearly_table(
    rows: list[dict[str, Any]],
    *,
    left_prefix: str,
    right_prefix: str,
    metric_suffix: str,
    left_label: str,
    right_label: str,
    delta_label: str,
    unit: str,
    preference: str,
    digits: int,
    use_color: bool,
) -> str:
    table_rows = []
    for row in rows:
        year = row.get("entry_year")
        if year is None:
            continue
        left = row.get(f"{left_prefix}_{metric_suffix}")
        right = row.get(f"{right_prefix}_{metric_suffix}")
        try:
            delta = float(right) - float(left)
        except (TypeError, ValueError):
            delta = None
        if delta is not None and not math.isfinite(delta):
            delta = None
        signal = signal_for_delta(delta, preference=preference, warning_threshold=0.0)
        table_rows.append((
            str(int(year)),
            _compact_metric_value(left, unit=unit, digits=digits),
            _compact_metric_value(right, unit=unit, digits=digits),
            terminal_signal(
                "N/A" if delta is None else f"{delta:+.{digits}f}{unit}",
                signal,
                enabled=use_color,
            ),
            terminal_signal(signal_marker(signal), signal, enabled=use_color),
        ))
    if not table_rows:
        return "無進場年度資料。"
    return render_table(
        ("年度", left_label, right_label, delta_label, "判讀"),
        table_rows,
        alignments=("right", "right", "right", "right", "left"),
    )



def _compact_rolling_coverage(result: dict[str, Any]) -> tuple[str, str]:
    rolling = dict(result.get("rolling_validation") or {})
    coverage = dict(rolling.get("training_score_coverage") or {})
    summary = render_key_values((
        ("Rolling folds", coverage.get("fold_count", rolling.get("fold_count", "-"))),
        ("Adapted trials／fold", rolling.get("trials_per_fold", "-")),
        ("Trials source", rolling.get("trials_per_fold_source", "-")),
        ("Baseline歷史工件 trials／fold", rolling.get("baseline_trials_per_fold", "-")),
        ("Train window", f"{rolling.get('train_window_months', '-')} months"),
        ("OOS horizon", f"{rolling.get('oos_horizon_months', '-')} months"),
        ("Score Adapted search reused", rolling.get("score_adapted_search_reused", "-")),
        ("新參數optimizer數", rolling.get("optimizer_training_arm_count", 1)),
    ))
    status_labels = {
        "bootstrap_fallback_only": "Bootstrap fallback",
        "partial_score_history": "Partial Score history",
        "full_score_history": "Full Score history",
    }
    rows = []
    for row in list(coverage.get("folds") or []):
        try:
            coverage_pct = float(row.get("calendar_coverage_ratio")) * 100.0
            coverage_text = f"{coverage_pct:.1f}%"
        except (TypeError, ValueError):
            coverage_text = "N/A"
        rows.append((
            row.get("fold", "-"),
            row.get("selection_period", "-"),
            row.get("oos_period", "-"),
            coverage_text,
            status_labels.get(str(row.get("status")), str(row.get("status") or "-")),
        ))
    table = (
        render_table(
            ("Fold", "Selection period", "OOS period", "Score coverage", "狀態"),
            rows,
            alignments=("left", "left", "left", "right", "left"),
        )
        if rows else "無Rolling Score coverage資料。"
    )
    return summary, table



def _pair_judgement(delta: dict[str, Any], *, subject: str) -> tuple[str, str]:
    try:
        total = float(delta.get("total_return_pct"))
        romd = float(delta.get("return_over_max_drawdown"))
        mdd = float(delta.get("max_drawdown_pct"))
    except (TypeError, ValueError):
        return SIGNAL_NEUTRAL, f"{subject}資料不足。"
    if total > 0 and romd > 0 and mdd <= 0:
        return SIGNAL_POSITIVE, f"{subject}改善總報酬與報酬／回撤，且未放大最大回撤。"
    if total < 0 and romd < 0 and mdd >= 0:
        return SIGNAL_NEGATIVE, f"{subject}總報酬與報酬／回撤惡化，且最大回撤未改善。"
    return SIGNAL_WARNING, f"{subject}結果混合，需結合年度、資金與capture判讀。"


def _render_four_way_console(result: dict[str, Any]) -> str:
    _rows, values = _metric_rows(result)
    metadata = dict(result.get("metadata") or {})
    period = dict(metadata.get("comparison_period") or {})
    use_color = console_color_enabled()
    portfolio_specs = (
        ("淨總報酬", "total_return_pct", "%", "higher", 2),
        ("最大回撤", "max_drawdown_pct", "%", "lower", 2),
        ("報酬／最大回撤", "return_over_max_drawdown", "", "higher", 2),
        ("年化報酬", "annual_return_pct", "%", "higher", 2),
        ("Log R²", "log_r_squared", "", "higher", 4),
        ("月勝率", "monthly_win_rate_pct", "%", "higher", 2),
        ("最差完整年度", "min_full_year_return_pct", "%", "higher", 2),
    )
    trade_specs = (
        ("交易數", "trade_count", "", "neutral", 0),
        ("勝率", "win_rate_pct", "%", "higher", 2),
        ("Payoff", "payoff_ratio", "", "higher", 2),
        ("EV", "expected_value_r", " R", "higher", 2),
        ("平均 Realized R", "avg_realized_r", " R", "higher", 2),
        ("平均投入資金報酬", "avg_capital_return_pct", "%", "higher", 2),
    )
    sizing_specs = (
        ("平均曝險", "avg_exposure_pct", "%", "higher", 2),
        ("平均預留金額", "avg_reserved_total", "", "attention", 2),
        ("平均實際投入金額", "avg_invested_total", "", "higher", 2),
        ("投入／預留比", "avg_invested_vs_reserved_pct", "%", "higher", 2),
        ("平均初始停損距離", "avg_stop_distance_pct", "%", "attention", 2),
    )
    capacity_specs = (
        ("平均每日可掛單候選", "avg_orderable_candidates", "", "neutral", 2),
        ("候選供給不足日", "candidate_supply_gap_days", " 日", "lower", 0),
        ("每日結束未滿倉日", "underfilled_end_days", " 日", "lower", 0),
        ("每日結束持股缺口", "end_position_gap_slot_days", " 格日", "lower", 0),
        ("保留買單成交率", "reserved_buy_fill_rate_pct", "%", "higher", 2),
    )
    target_specs = (
        ("平均 Target R", "avg_target_r", " R", "higher", 2),
        ("Aggregate Target capture", "aggregate_target_capture_ratio", "", "higher", 2),
        ("Median Target capture", "median_target_capture_ratio", "", "higher", 2),
        ("Target ≥ 0.5R capture", "target_ge_0_5_capture_ratio", "", "higher", 2),
        ("平均 Target realization gap", "avg_target_realization_gap_r", " R", "lower", 2),
    )
    turnover_specs = (
        ("平均持有日", "avg_holding_calendar_days", " 日", "lower", 2),
        ("半倉交易占比", "partial_exit_trade_share_pct", "%", "attention", 2),
        ("半倉後至結算平均日曆日", "avg_partial_to_exit_calendar_days", " 日", "lower", 2),
        ("半倉殘留交易slot-days", "partial_residual_slot_days", " 格日", "lower", 0),
        ("Top 5進場日交易占比", "top_5_entry_dates_share_pct", "%", "attention", 2),
        ("最大單月進場占比", "top_entry_month_share_pct", "%", "attention", 2),
        ("進場月份 HHI", "entry_month_hhi", "", "attention", 2),
        ("產業資料覆蓋", "industry_coverage_pct", "%", "neutral", 2),
        ("最大產業占比", "top_industry_share_pct", "%", "attention", 2),
        ("產業 HHI", "industry_hhi", "", "attention", 2),
    )
    exit_specs = (
        ("停損", "stop_exit_share_pct", "%", "attention", 2),
        ("指標", "indicator_exit_share_pct", "%", "attention", 2),
        ("汰弱", "rotation_exit_share_pct", "%", "attention", 2),
        ("期末強制", "forced_exit_share_pct", "%", "attention", 2),
    )
    diagnostics = dict(result.get("selection_diagnostics") or {})
    diagnostic_values = {
        key: dict(diagnostics.get(key) or {})
        for key in ("baseline", "sort_only", "param_only", "adapted")
    }
    diagnostic_specs = (
        ("Orderable Score coverage", "orderable_score_coverage_rate", "", "higher", 4),
        ("選中候選 Target percentile", "selected_target_percentile_mean", "", "higher", 4),
        ("Target top-k retention", "target_top_k_retention_mean", "", "higher", 4),
        ("Target opportunity gap (R)", "target_opportunity_gap_r_mean", "", "lower", 4),
        ("選中候選 Target mean (R)", "selected_target_mean_r", "", "higher", 4),
    )
    yearly_rows = list(result.get("yearly") or [])
    capture_yearly = _capture_yearly_pairs(result)
    rolling_summary, rolling_table = _compact_rolling_coverage(result)

    def pair_blocks(specs):
        return (
            "A. 固定原參數：純 Ranking 效果",
            _pair_table(
                values, specs,
                left_key="baseline", right_key="sort_only",
                left_label="Baseline", right_label="Sort Only",
                delta_label="Ranking差異", use_color=use_color,
            ),
            "B. 同一套新Adapted參數：Ranking效果",
            _pair_table(
                values, specs,
                left_key="param_only", right_key="adapted",
                left_label="Param Only", right_label="Adapted",
                delta_label="新參數Ranking差異", use_color=use_color,
            ),
        )

    lines = [
        render_title("Breakout Quality Ranking × Parameter 2×2驗證摘要"),
        render_key_values((
            ("期間", f"{period.get('start', '')} ～ {period.get('end', '')}"),
            ("比較設計", "Baseline／Sort Only／Param Only／Adapted"),
            ("純Ranking比較", "Sort Only − Baseline；active params完全相同"),
            ("新參數Ranking比較", "Adapted − Param Only；兩組使用同一套新Adapted params"),
            ("Score source", metadata.get("score_source", "selection_point_in_time")),
            ("結果性質", ADAPTATION_STATUS),
            ("Future Target runtime", "未使用"),
        )),
        render_section("投組報酬與風險", number=1),
        "定義：衡量整體權益最後賺多少、曾承受多少回撤，以及成長路徑是否穩定。",
        *pair_blocks(portfolio_specs),
        render_section("單筆交易結果", number=2),
        "定義：勝率看獲利筆數；Payoff看平均贏家／輸家；EV與Realized R看每筆初始風險的實際期望值。",
        *pair_blocks(trade_specs),
        render_section("資金投入與部位大小", number=3),
        "定義：曝險是每日投入市場資金占權益比例；停損越寬，固定風險sizing下每筆部位通常越小。",
        *pair_blocks(sizing_specs),
        render_section("候選供給與持倉容量", number=4),
        "定義：候選供給看是否有股票可買；未滿倉與缺口看持倉格是否填滿；成交率看預留買單是否落地。",
        *pair_blocks(capacity_specs),
        render_section("模型選股能力", number=5),
        "定義：比較實際選中候選與事後Future Target理想排序；Future Target只在回放完成後join。",
        "A. 固定原參數：純 Ranking 效果",
        _pair_table(
            diagnostic_values, diagnostic_specs,
            left_key="baseline", right_key="sort_only",
            left_label="Baseline", right_label="Sort Only",
            delta_label="Ranking差異", use_color=use_color,
        ),
        "B. 同一套新Adapted參數：Ranking效果",
        _pair_table(
            diagnostic_values, diagnostic_specs,
            left_key="param_only", right_key="adapted",
            left_label="Param Only", right_label="Adapted",
            delta_label="新參數Ranking差異", use_color=use_color,
        ),
        render_section("Target 到實際報酬的轉換", number=6),
        "定義：Target R是事後價格機會；capture衡量Realized R相對Target R的轉換；gap越低越好。",
        *pair_blocks(target_specs),
        render_section("資金周轉與進場集中", number=7),
        "定義：持有與半倉指標衡量資金占用時間；日期、月份與產業指標衡量交易是否集中。",
        *pair_blocks(turnover_specs),
        render_section("出場結構", number=8),
        "定義：依每筆交易最後的全倉結算原因分類；占比改變只表示結構差異，不直接等於損益好壞。",
        *pair_blocks(exit_specs),
        render_section("年度結果與年度歸因", number=9),
        "定義：年度報酬按權益曲線年度；R、capture與投入金額按交易進場年度分組。",
        "A. 固定原參數：年度純 Ranking 效果",
        _yearly_pair_table(
            yearly_rows,
            left_column="no_filter_return_pct", right_column="score_ranking_return_pct",
            left_label="Baseline", right_label="Sort Only",
            delta_label="Ranking差異", use_color=use_color,
        ),
        "B. 同一套新Adapted參數：年度Ranking效果",
        _yearly_pair_table(
            yearly_rows,
            left_column="param_only_return_pct", right_column="adapted_return_pct",
            left_label="Param Only", right_label="Adapted",
            delta_label="新參數Ranking差異", use_color=use_color,
        ),
        "固定原參數進場年度 Aggregate capture：",
        _capture_yearly_table(
            capture_yearly["current"],
            left_prefix="baseline", right_prefix="score_sort",
            metric_suffix="aggregate_capture_ratio",
            left_label="Baseline", right_label="Sort Only",
            delta_label="Ranking差異", unit="", preference="higher", digits=2,
            use_color=use_color,
        ),
        "新Adapted參數進場年度 Aggregate capture：",
        _capture_yearly_table(
            capture_yearly["adapted"],
            left_prefix="baseline", right_prefix="score_sort",
            metric_suffix="aggregate_capture_ratio",
            left_label="Param Only", right_label="Adapted",
            delta_label="新參數Ranking差異", unit="", preference="higher", digits=2,
            use_color=use_color,
        ),
        render_section("Rolling 訓練與 Score coverage", number=10),
        "定義：新參數只由Score ranking rolling optimizer訓練一次；Param Only與Adapted共用同一套active params。",
        rolling_summary,
        rolling_table,
        "限制：PIT開始日前缺分依正式契約回退原buy-sort；PIT期間內缺口禁止。",
        render_section("舊ROOS與新Adapted參數差異", number=11),
        "定義：比較既有正式ROOS與新Score Adapted各fold finalist-best active params的中位數與範圍。",
        render_table(
            ("參數", "舊ROOS中位", "舊ROOS範圍", "Adapted中位", "Adapted範圍"),
            _param_comparison_table_rows(result),
            alignments=("left", "right", "right", "right", "right"),
        ),
    ]

    ranking_delta = dict(result.get("old_params_ranking_delta") or {})
    adapted_ranking_delta = dict(result.get("adapted_params_ranking_delta") or {})
    ranking_signal, ranking_text = _pair_judgement(
        ranking_delta, subject="純Ranking"
    )
    adapted_signal, adapted_text = _pair_judgement(
        adapted_ranking_delta, subject="新Adapted參數下的Ranking"
    )
    optimized_decision = dict(
        dict(result.get("adapted_params_capture_audit") or {}).get("decision") or {}
    )
    decision_signal = str(optimized_decision.get("signal") or SIGNAL_NEUTRAL)
    bottlenecks = "；".join(optimized_decision.get("bottlenecks") or []) or "未辨識出明確瓶頸"
    lines.extend((
        render_section("綜合判定、限制與下一步", number=12),
        terminal_signal(
            f"{signal_marker(ranking_signal)} Ranking-only判定：{ranking_text}",
            ranking_signal,
            enabled=use_color,
        ),
        terminal_signal(
            f"{signal_marker(adapted_signal)} 新參數Ranking判定：{adapted_text}",
            adapted_signal,
            enabled=use_color,
        ),
        terminal_signal(
            f"{signal_marker(decision_signal)} 重調後資金／capture判定："
            f"{optimized_decision.get('status', '-')}｜{optimized_decision.get('conclusion', '-')}",
            decision_signal,
            enabled=use_color,
        ),
        f"主要瓶頸：{bottlenecks}",
        "判讀原則：舊參數看Sort Only − Baseline；新參數看Adapted − Param Only。Param Only與Adapted使用完全相同的新active params。",
        "下一步：只有兩個Ranking比較與參數適應效果支持後，才進完整Selection final refit；本流程不執行正式OOS。",
        "限制：本結果是無前視Rolling Selection診斷；Future Target只在回放完成後join。",
        "核心純Ranking差異：總報酬 "
        f"{_compact_metric_value(ranking_delta.get('total_return_pct'), unit='pp', digits=2)}；"
        "報酬／回撤 "
        f"{_compact_metric_value(ranking_delta.get('return_over_max_drawdown'), digits=2)}。",
        "核心新參數Ranking差異：總報酬 "
        f"{_compact_metric_value(adapted_ranking_delta.get('total_return_pct'), unit='pp', digits=2)}；"
        "報酬／回撤 "
        f"{_compact_metric_value(adapted_ranking_delta.get('return_over_max_drawdown'), digits=2)}。",
    ))
    return "\n".join(lines)




def _render_four_way_markdown(result: dict[str, Any]) -> str:
    rows, values = _metric_rows(result)
    lines = [
        "# Selection Ranking × Parameter 2×2 Rolling Validation",
        "",
        f"- 狀態：`{ADAPTATION_STATUS}`",
        "- 新參數只由Score ranking rolling optimizer訓練一次。",
        "- 舊參數Ranking效果：Sort Only − Baseline。",
        "- 新參數Ranking效果：Adapted − Param Only；兩組使用完全相同的新Adapted params。",
        "- Future Target只在replay後離線join，未進runtime或optimizer。",
        "",
        "## 核心指標",
        "",
        "| 指標 | Baseline | Sort Only | Param Only | Adapted |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, key, unit in rows:
        lines.append(
            f"| {label} | {_fmt(values['baseline'].get(key), unit)} | "
            f"{_fmt(values['sort_only'].get(key), unit)} | "
            f"{_fmt(values['param_only'].get(key), unit)} | "
            f"{_fmt(values['adapted'].get(key), unit)} |"
        )
    lines.extend((
        "",
        "## 年度報酬",
        "",
        "| 年度 | Baseline | Sort Only | Param Only | Adapted |",
        "|---:|---:|---:|---:|---:|",
    ))
    for row in list(result.get("yearly") or []):
        lines.append(
            f"| {int(row['year'])} | {_fmt(row.get('no_filter_return_pct'), '%')} | "
            f"{_fmt(row.get('score_ranking_return_pct'), '%')} | "
            f"{_fmt(row.get('param_only_return_pct'), '%')} | "
            f"{_fmt(row.get('adapted_return_pct'), '%')} |"
        )
    lines.extend((
        "",
        "## 舊ROOS與新Adapted參數差異",
        "",
        "| 參數 | 舊ROOS中位 | 舊ROOS範圍 | Adapted中位 | Adapted範圍 |",
        "|---|---:|---:|---:|---:|",
    ))
    for parameter, left_med, left_range, right_med, right_range in _param_comparison_table_rows(result):
        lines.append(
            f"| {parameter} | {left_med} | {left_range} | {right_med} | {right_range} |"
        )
    lines.extend((
        "",
        "## 判讀邊界",
        "",
        "- Param Only不是舊ranking重新最佳化；它只是把同一套新Adapted params切回舊ranking做反事實回放。",
        "- 本流程只訓練Score Adapted一套新參數，不重訓舊ranking。",
        "- 本結果不是完整Selection final refit，也不是正式OOS。",
        "",
    ))
    return "\n".join(lines)



_render_four_way_console_impl = _render_four_way_console


def _render_three_way_console(result: dict[str, Any]) -> str:
    return _render_four_way_console_impl(result)


def _render_three_way_markdown(result: dict[str, Any]) -> str:
    return _render_four_way_markdown(result)



def run_adaptation(*, project_root=PROJECT_ROOT, argv=None) -> dict[str, Any]:
    args = _parse_args(argv)
    root = Path(project_root).resolve()
    settings = get_breakout_quality_workflow_settings()
    _validate_fixed_contract(args, settings)
    configure_optuna_logging()

    pit_contract = load_selection_point_in_time_ranking_contract(
        str(root), args.filter_id, args.model_architecture, args.experiment_profile
    )
    _validate_pit_contract_against_settings(pit_contract, settings)
    pit_runtime_artifacts = _validate_selection_pit_runtime_artifacts(
        pit_contract=pit_contract,
        args=args,
    )
    baseline_contract = _load_baseline_rolling_contract(
        root=root, args=args, settings=settings
    )
    baseline_meta = dict(baseline_contract["meta"])
    if str(pit_contract.available_from) != str(baseline_meta["first_oos_date"]):
        raise ValueError(
            "PIT Score起始日與Baseline rolling replay起始日不一致："
            f"pit={pit_contract.available_from}, baseline={baseline_meta['first_oos_date']}"
        )
    baseline_last_oos_end = (
        pd.Timestamp(baseline_meta["last_oos_date"]) + pd.offsets.MonthEnd(1)
    ).strftime("%Y-%m-%d")
    if str(pit_contract.available_through) != baseline_last_oos_end:
        raise ValueError(
            "PIT Score結束日與Baseline rolling replay結束日不一致："
            f"pit={pit_contract.available_through}, baseline={baseline_last_oos_end}"
        )

    adapted_policy = _build_base_policy(
        root=root, baseline_contract=baseline_contract
    )
    adapted_policy.update({
        "adaptation_arm": "score_adapted",
        "adaptation_scope": "selection_rolling_single_score_adapted_optimizer",
    })
    adapted_contract = _build_runtime_contract(
        root=root,
        args=args,
        settings=settings,
        pit_contract=pit_contract,
        baseline_contract=baseline_contract,
        base_policy=adapted_policy,
        arm_name="score_adapted",
        ranking_enabled=True,
    )
    score_coverage = dict(adapted_contract["training_score_coverage"])
    bootstrap_folds = int(score_coverage["bootstrap_fallback_only_folds"])
    partial_folds = int(score_coverage["partial_score_history_folds"])
    if bootstrap_folds or partial_folds:
        print(
            "\n[Rolling Score coverage] "
            f"bootstrap fallback folds={bootstrap_folds}, "
            f"partial-score folds={partial_folds}. "
            "PIT開始日前依正式缺分契約回退原buy-sort；PIT期間內缺口禁止。"
        )

    output_dir = (root / ADAPTATION_RELATIVE_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight_path = output_dir / "rolling_preflight.json"
    _write_json(
        preflight_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "ROLLING_2X2_SINGLE_ADAPTED_OPTIMIZER_PREFLIGHT_PASS",
            "comparison_design": "ranking_parameter_2x2_single_adapted_optimizer",
            "runtime_identity_sha256": adapted_contract["runtime_identity_sha256"],
            "score_adapted_contract": adapted_contract,
            "optimizer_training_arms": ["score_adapted"],
            "selection_pit_runtime_artifacts": {
                key: {
                    "path": project_relative_display_path(path, project_root=root),
                    "sha256": compute_file_sha256(path),
                }
                for key, path in pit_runtime_artifacts.items()
            },
            "created_at": get_taipei_now().isoformat(),
        },
    )
    coverage_path = output_dir / "rolling_training_score_coverage.csv"
    pd.DataFrame(score_coverage["folds"]).to_csv(
        coverage_path, index=False, encoding="utf-8-sig"
    )

    current_pair_dir = _canonical_current_pair_output_dir(root=root, args=args)
    current_pair_contract = {
        "runtime_identity_sha256": _build_current_pair_runtime_identity(
            root=root,
            args=args,
            pit_contract=pit_contract,
            baseline_contract=baseline_contract,
        )
    }
    current_payload, current_pair_reused = _load_or_run_current_pair(
        root=root,
        args=args,
        runtime_contract=current_pair_contract,
        output_dir=current_pair_dir,
        pit_contract=pit_contract,
        baseline_contract=baseline_contract,
    )
    current_decision = dict(
        current_payload.get("score_ranking_capture_audit", {})
    ).get("decision", {})
    if str(current_decision.get("status") or "") != "ADAPTATION_DIAGNOSTIC_SUPPORTED":
        raise RuntimeError(
            "目前Baseline／Sort Only capture audit未通過參數適應前置條件："
            f"status={current_decision.get('status')!r}"
        )

    adapted_arm = _run_rolling_optimizer_arm(
        root=root,
        args=args,
        settings=settings,
        baseline_contract=baseline_contract,
        base_policy=adapted_policy,
        runtime_contract=adapted_contract,
        output_dir=output_dir,
        arm_name="score_adapted",
        arm_label="Score Adapted",
        ranking_enabled=True,
    )

    optimizer_summary_path = output_dir / "rolling_optimizer_summary.json"
    optimizer_summary = {
        "mode": "rolling_selection_2x2_single_adapted_optimizer",
        "result_interpretation": ADAPTATION_STATUS,
        "folds": int(score_coverage.get("fold_count", 0) or 0),
        "trials_per_fold": int(args.trials_per_fold),
        "trials_per_fold_source": str(
            adapted_contract["rolling_policy"]["trials_per_fold_source"]
        ),
        "baseline_history_trials_per_fold": int(baseline_meta["trials_per_fold"]),
        "baseline_history_trial_count_match_required": False,
        "baseline_sort_only_reused": bool(current_pair_reused),
        "optimizer_training_arm_count": 1,
        "optimizer_training_arms": ["score_adapted"],
        "score_adapted": adapted_arm["optimizer_summary"],
        "runtime_identity_sha256": adapted_contract["runtime_identity_sha256"],
        "final_selection_refit_executed": False,
        "formal_oos_executed": False,
    }
    _write_json(optimizer_summary_path, optimizer_summary)

    comparison_result = _run_four_way_comparison(
        root=root,
        args=args,
        pit_contract=pit_contract,
        current_payload=current_payload,
        baseline_contract=baseline_contract,
        adapted_arm=adapted_arm,
        current_pair_dir=current_pair_dir,
        output_dir=output_dir,
        rolling_validation={
            "fold_count": int(score_coverage.get("fold_count", 0) or 0),
            "trials_per_fold": int(args.trials_per_fold),
            "trials_per_fold_source": str(
                adapted_contract["rolling_policy"]["trials_per_fold_source"]
            ),
            "baseline_trials_per_fold": int(baseline_meta["trials_per_fold"]),
            "trial_count_match_required": False,
            "train_window_months": int(baseline_meta["train_window_months"]),
            "oos_horizon_months": int(baseline_meta["oos_horizon_months"]),
            "optimizer_training_arm_count": 1,
            "optimizer_training_arms": ["score_adapted"],
            "score_adapted_search_reused": bool(adapted_arm["search_reused"]),
            "training_score_coverage": score_coverage,
        },
    )

    capture_dir = output_dir / "adapted_params_ranking_capture"
    artifact_paths = {
        "rolling_preflight": preflight_path,
        **pit_runtime_artifacts,
        "training_score_coverage": coverage_path,
        "adapted_params": adapted_arm["params_path"],
        "adapted_preflight": adapted_arm["preflight_path"],
        "adapted_optimizer_summary": adapted_arm["optimizer_summary_path"],
        "optimizer_summary": optimizer_summary_path,
        "comparison_json": output_dir / "strategy_adaptation_comparison.json",
        "comparison_markdown": output_dir / "strategy_adaptation_comparison.md",
        "yearly_csv": output_dir / "strategy_adaptation_yearly_returns.csv",
        "param_only_equity": output_dir / "param_only_equity.csv",
        "param_only_trades": output_dir / "param_only_trades.csv",
        "adapted_equity": output_dir / "adapted_equity.csv",
        "adapted_trades": output_dir / "adapted_trades.csv",
        "adapted_params_capture_json": capture_dir / "score_ranking_capture_audit.json",
        "adapted_params_capture_markdown": capture_dir / "score_ranking_capture_audit.md",
        "baseline_sort_only_comparison_json": current_pair_dir / "strategy_comparison.json",
        "baseline_sort_only_comparison_markdown": current_pair_dir / "strategy_comparison.md",
        "baseline_sort_only_manifest": current_pair_dir / CURRENT_PAIR_MANIFEST_FILENAME,
    }
    missing_artifacts = [
        str(item) for item in artifact_paths.values() if not item.is_file()
    ]
    if missing_artifacts:
        raise FileNotFoundError(
            "Rolling 2×2 adaptation完成後缺少必要工件: "
            + ", ".join(missing_artifacts)
        )

    manifest_path = output_dir / "rolling_adaptation_manifest.json"
    execution_argv = _canonical_execution_argv(args)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "ROLLING_2X2_SINGLE_ADAPTED_OPTIMIZER_COMPLETED",
        "comparison_design": "ranking_parameter_2x2_single_adapted_optimizer",
        "result_interpretation": ADAPTATION_STATUS,
        "runtime_identity_sha256": adapted_contract["runtime_identity_sha256"],
        "score_adapted_contract": adapted_contract,
        "optimizer_summary": optimizer_summary,
        "baseline_sort_only_reused": bool(current_pair_reused),
        "execution_command": " ".join(execution_argv),
        "execution_argv": execution_argv,
        "created_at": get_taipei_now().isoformat(),
        "artifacts": {
            key: {
                "path": project_relative_display_path(item, project_root=root),
                "sha256": compute_file_sha256(item),
            }
            for key, item in artifact_paths.items()
        },
    }
    _write_json(manifest_path, manifest)
    print_artifact_paths(
        (
            ("Rolling 2×2 preflight", preflight_path),
            ("Training Score coverage", coverage_path),
            ("Score Adapted active params", adapted_arm["params_path"]),
            ("Rolling adaptation manifest", manifest_path),
            ("Rolling optimizer summary", optimizer_summary_path),
            ("四組比較 Markdown", output_dir / "strategy_adaptation_comparison.md"),
            ("四組比較 JSON", output_dir / "strategy_adaptation_comparison.json"),
        ),
        project_root=root,
    )
    return comparison_result



def main(argv=None):
    run_adaptation(argv=argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
