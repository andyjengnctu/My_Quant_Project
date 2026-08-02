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
from core.raw_universe_contract import RAW_UNIVERSE_REQUIRED_MIN_ROWS_FIELD
from core.runtime_utils import get_taipei_now
from core.walk_forward_policy import (
    build_optimizer_effective_policy_fingerprint,
    load_walk_forward_policy,
)
from filters.breakout_quality.artifacts import compute_file_sha256
from filters.breakout_quality.console_report import (
    compact_console_enabled,
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
from tools.filters.breakout_quality.strategy_compare import (
    COMPARISON_MODE_SCORE_RANKING,
    PARAM_POLICY_BASE_FINALIST_BEST,
    _assert_shared_benchmark,
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
    run_comparison,
)
from tools.optimizer.prep import load_all_raw_data
from tools.optimizer.outer_rolling_oos import run_outer_rolling_oos
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
            "以與Baseline相同的rolling folds及trials／fold，固定Selection PIT "
            "Score Sort執行策略參數適應，並輸出Baseline／Sort Only／"
            "Adapted Rolling三組Selection診斷。"
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
        help="每個rolling fold的optimizer trial數；必須與Baseline active params一致",
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

def _load_current_pair_if_compatible(
    *, root: Path, output_dir: Path, runtime_identity_sha256: str
) -> dict[str, Any] | None:
    manifest_path = output_dir / CURRENT_PAIR_MANIFEST_FILENAME
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if str(manifest.get("runtime_identity_sha256") or "") != str(
        runtime_identity_sha256
    ):
        return None
    recorded = dict(manifest.get("artifacts") or {})
    expected_paths = _current_pair_artifact_paths(output_dir)
    for key, path in expected_paths.items():
        entry = dict(recorded.get(key) or {})
        expected_display_path = project_relative_display_path(path, project_root=root)
        if (
            not path.is_file()
            or str(entry.get("path") or "") != expected_display_path
            or str(entry.get("sha256") or "") != compute_file_sha256(path)
        ):
            return None
    comparison_path = expected_paths["strategy_comparison_json"]
    try:
        payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        return None
    return payload

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
    *, root: Path, args, runtime_contract: dict[str, Any], output_dir: Path
) -> tuple[dict[str, Any], bool]:
    payload = _load_current_pair_if_compatible(
        root=root,
        output_dir=output_dir,
        runtime_identity_sha256=runtime_contract["runtime_identity_sha256"],
    )
    if payload is not None:
        print("\n[Baseline／Sort Only] identity與工件hash一致，沿用既有結果。")
        return payload, True
    payload = run_comparison(
        project_root=root,
        dataset=args.dataset,
        param_policy=args.param_policy,
        max_positions=args.max_positions,
        enable_rotation=str(args.rotation) == "on",
        fixed_risk=args.fixed_risk,
        max_position_cap_pct=args.max_position_cap_pct,
        comparison_mode=COMPARISON_MODE_SCORE_RANKING,
        filter_id=args.filter_id,
        score_source=SCORE_SOURCE_SELECTION_POINT_IN_TIME,
        model_architecture=args.model_architecture,
        experiment_profile=args.experiment_profile,
        output_dir_override=output_dir,
        quiet=args.quiet,
    )
    _write_current_pair_manifest(
        root=root,
        output_dir=output_dir,
        runtime_identity_sha256=runtime_contract["runtime_identity_sha256"],
    )
    return payload, False

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
    baseline_trials = int(meta["trials_per_fold"])
    if int(args.trials_per_fold) != baseline_trials:
        raise ValueError(
            "Adapted Rolling的trials／fold必須與Baseline一致："
            f"requested={int(args.trials_per_fold)}, baseline={baseline_trials}"
        )
    if int(settings.strategy_adapt_trials_per_fold) != baseline_trials:
        raise ValueError(
            "目前training_policy的outer rolling trials與Baseline工件不一致；"
            "請先用目前設定重建Baseline active params："
            f"config={int(settings.strategy_adapt_trials_per_fold)}, artifact={baseline_trials}"
        )
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
                "Adapted Rolling的每個OOS replay都必須完整位於Selection PIT期間："
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
    *, root: Path, args, settings, pit_contract, baseline_contract, base_policy
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
        "adaptation_mode": "rolling_selection_validation",
        "result_interpretation": ADAPTATION_STATUS,
        "dataset": str(args.dataset),
        "dataset_identity": build_source_data_inventory(root, args.dataset),
        "comparison_period": {
            "start": str(pit_contract.available_from),
            "end": str(pit_contract.available_through),
        },
        "rolling_training_period": str(baseline_contract["summary"].get("selection_period") or ""),
        "training_score_coverage": training_score_coverage,
        "rolling_policy": {
            "window_mode": str(baseline_meta["window_mode"]),
            "first_oos_date": str(baseline_meta["first_oos_date"]),
            "last_oos_date": str(baseline_meta["last_oos_date"]),
            "train_window_months": int(baseline_meta["train_window_months"]),
            "oos_horizon_months": int(baseline_meta["oos_horizon_months"]),
            "trials_per_fold": int(baseline_meta["trials_per_fold"]),
            "folds": int(baseline_contract["summary"].get("folds", 0) or 0),
            "active_param_policy": str(baseline_meta.get("active_param_policy") or ""),
        },
        "baseline_active_params": {
            "path": project_relative_display_path(baseline_contract["path"], project_root=root),
            "sha256": str(baseline_contract["sha256"]),
            "selector": str(baseline_contract["payload"].get("selector") or ""),
        },
        "optimizer_policy": "base-finalist-best",
        "objective": "split_train_romd",
        "optimizer_effective_policy": effective_policy,
        "search_space_identity_sha256": _canonical_json_sha256(search_space_snapshot),
        "search_space_snapshot": search_space_snapshot,
        "seed": int(settings.seed),
        "pit_identity": {
            "score_csv_sha256": score_hash,
            "manifest_sha256": manifest_hash,
            "audit_sha256": audit_hash,
            "score_path": project_relative_display_path(pit_contract.score_path, project_root=root),
            "manifest_path": project_relative_display_path(pit_contract.manifest_path, project_root=root),
            "audit_path": project_relative_display_path(pit_contract.audit_path, project_root=root),
        },
        "model_identity": {
            "filter_id": str(args.filter_id),
            "architecture": str(args.model_architecture),
            "experiment_profile": str(args.experiment_profile),
            "continuous_target_id": str(pit_contract.continuous_target_id),
        },
        "ranking_mode": "breakout_quality_score_desc",
        "score_source": SCORE_SOURCE_SELECTION_POINT_IN_TIME,
        "fixed_runtime": {
            "use_breakout_quality_ranking": True,
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


def _optimizer_session_spec(*, root: Path, output_dir: Path, args, runtime_contract) -> dict[str, Any]:
    return {
        "output_dir": str((output_dir / "optimizer_runtime" / "sessions").resolve()),
        "fixed_strategy_param_overrides": {
            "use_breakout_quality_filter": False,
            "use_breakout_quality_ranking": True,
            "breakout_quality_filter_id": str(args.filter_id),
            "fixed_risk": float(args.fixed_risk),
            "max_position_cap_pct": float(args.max_position_cap_pct),
        },
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
    *, path: Path, baseline_contract: dict[str, Any], runtime_contract: dict[str, Any], args
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"rolling optimizer未產生base-finalist-best工件: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    meta = dict(payload.get("meta") or {})
    baseline_meta = dict(baseline_contract["meta"])
    for key in (
        "first_oos_date", "last_oos_date", "train_window_months",
        "oos_horizon_months", "trials_per_fold",
    ):
        if str(meta.get(key)) != str(baseline_meta.get(key)):
            raise ValueError(
                "Adapted Rolling與Baseline rolling policy不一致："
                f"{key}: adapted={meta.get(key)!r}, baseline={baseline_meta.get(key)!r}"
            )
    if int(dict(payload.get("summary") or {}).get("folds", 0) or 0) != int(
        baseline_contract["summary"].get("folds", 0) or 0
    ):
        raise ValueError("Adapted Rolling fold數與Baseline不一致")
    members = []
    for effective_date, raw_members in dict(payload.get("params_ensemble_by_effective_date") or {}).items():
        for raw_member in list(raw_members or []):
            params = dict((raw_member or {}).get("params") or {})
            members.append((str(effective_date), params))
    if not members:
        raise ValueError("Adapted Rolling工件沒有active-param members")
    for effective_date, params in members:
        expected = {
            "use_breakout_quality_ranking": True,
            "use_breakout_quality_filter": False,
            "breakout_quality_filter_id": str(args.filter_id),
            "fixed_risk": float(args.fixed_risk),
            "max_position_cap_pct": float(args.max_position_cap_pct),
            "tp_percent": OPTIMIZER_FIXED_TP_PERCENT,
        }
        for key, expected_value in expected.items():
            actual = params.get(key)
            if isinstance(expected_value, float):
                matched = actual is not None and math.isclose(float(actual), expected_value, rel_tol=0.0, abs_tol=1e-12)
            else:
                matched = actual == expected_value
            if not matched:
                raise ValueError(
                    "Adapted Rolling固定契約未落入active params："
                    f"effective_date={effective_date}, {key}={actual!r}, expected={expected_value!r}"
                )
    payload["breakout_quality_adaptation"] = {
        "mode": "rolling_selection_validation",
        "result_interpretation": ADAPTATION_STATUS,
        "runtime_identity_sha256": str(runtime_contract["runtime_identity_sha256"]),
        "score_source": SCORE_SOURCE_SELECTION_POINT_IN_TIME,
        "future_target_used_for_runtime": False,
        "final_selection_refit": False,
    }
    _write_json(path, payload)
    return payload


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

def _run_three_way_comparison(
    *,
    root: Path,
    args,
    pit_contract,
    current_payload: dict[str, Any],
    baseline_contract: dict[str, Any],
    adapted_params_path: Path,
    current_pair_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    metadata = dict(current_payload["metadata"])
    comparison_dir = Path(current_pair_dir).resolve()
    required = {
        "baseline_trades": comparison_dir / "no_filter_trades.csv",
        "baseline_capacity": comparison_dir / "no_filter_daily_capacity.csv",
        "baseline_selected": comparison_dir / "no_filter_selected_target_diagnostics.csv",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("三組比較缺少Baseline工件: " + ", ".join(missing))

    adapted_source = _load_param_source(adapted_params_path)
    _validate_requested_param_policy(adapted_source, args.param_policy)
    (
        adapted_source_kind,
        _adapted_no_sort_params,
        adapted_score_params,
        _adapted_no_sort_payload,
        _adapted_score_payload,
        _adapted_ensemble_policy,
    ) = _build_controlled_param_source_pair(
        adapted_source,
        filter_id=str(args.filter_id),
        threshold=float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD),
        fixed_risk=float(args.fixed_risk),
        max_position_cap_pct=float(args.max_position_cap_pct),
        comparison_mode=COMPARISON_MODE_SCORE_RANKING,
    )
    if adapted_source_kind != "rolling_active_param_ensemble":
        raise ValueError("Adapted三組比較只接受rolling active-param ensemble")

    replay_counts: dict[str, Any] = {}
    ranking_source = {
        "score_source": SCORE_SOURCE_SELECTION_POINT_IN_TIME,
        "model_architecture": args.model_architecture,
        "experiment_profile": args.experiment_profile,
    }
    adapted_payload = _run_scenario(
        name="adapted_rolling_score_ranking",
        data_dir=Path(get_dataset_dir(str(root), args.dataset)).resolve(),
        param_source_kind=adapted_source_kind,
        params=adapted_score_params,
        start_date=pit_contract.available_from,
        end_date=pit_contract.available_through,
        max_positions=int(args.max_positions),
        enable_rotation=str(args.rotation) == "on",
        quiet=bool(args.quiet),
        replay_counts=replay_counts,
        ranking_source=ranking_source,
    )
    adapted = _scenario_summary(adapted_payload)
    baseline = dict(current_payload["no_filter"])
    sort_only = dict(current_payload["score_ranking"])

    baseline_equity = pd.read_csv(comparison_dir / "no_filter_equity.csv", encoding="utf-8-sig")
    _assert_shared_benchmark(
        {
            "benchmark_return_pct": baseline["benchmark_return_pct"],
            "benchmark_max_drawdown_pct": baseline["benchmark_max_drawdown_pct"],
            "benchmark_annual_return_pct": baseline["benchmark_annual_return_pct"],
            "equity_curve": baseline_equity,
        },
        adapted_payload,
    )

    adapted_orderable = _flatten_candidate_replay_rows(replay_counts, "orderable_rows")
    adapted_selected = _flatten_selected_buy_rows(adapted_payload["trade_history"])
    lookup = _selection_target_lookup(
        root=root,
        filter_id=args.filter_id,
        architecture=args.model_architecture,
        profile=args.experiment_profile,
    )
    adapted_diag, adapted_orderable_joined, adapted_selected_joined = (
        _strategy_selection_diagnostics(
            orderable=adapted_orderable,
            selected=adapted_selected,
            lookup=lookup,
        )
    )
    existing_diagnostics = dict(current_payload.get("selection_diagnostics") or {})
    baseline_diag = dict(existing_diagnostics.get("no_filter") or {})
    sort_diag = dict(existing_diagnostics.get("score_ranking") or {})

    adapted_payload["equity_curve"].to_csv(
        output_dir / "adapted_rolling_equity.csv", index=False, encoding="utf-8-sig"
    )
    adapted_payload["trade_history"].to_csv(
        output_dir / "adapted_rolling_trades.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(adapted_payload["profile"].get("portfolio_capacity_rows") or []).to_csv(
        output_dir / "adapted_rolling_daily_capacity.csv", index=False, encoding="utf-8-sig"
    )
    adapted_orderable_joined.to_csv(
        output_dir / "adapted_rolling_orderable_target_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    adapted_selected_joined.to_csv(
        output_dir / "adapted_rolling_selected_target_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    adapted_yearly = _yearly_frame(adapted_payload["profile"], "adapted")
    adapted_yearly.to_csv(
        output_dir / "adapted_rolling_yearly_returns.csv", index=False, encoding="utf-8-sig"
    )
    current_yearly = pd.DataFrame(current_payload.get("yearly") or [])
    yearly_keys = [
        key
        for key in ("year", "is_full_year", "start_date", "end_date")
        if key in current_yearly.columns and key in adapted_yearly.columns
    ]
    if not yearly_keys:
        yearly_keys = ["year"]
    three_way_yearly = current_yearly.merge(
        adapted_yearly, on=yearly_keys, how="outer", validate="one_to_one"
    ).sort_values("year").reset_index(drop=True)
    three_way_yearly["adapted_minus_baseline_pct"] = (
        three_way_yearly["adapted_return_pct"]
        - three_way_yearly["no_filter_return_pct"]
    )
    three_way_yearly["adapted_minus_sort_only_pct"] = (
        three_way_yearly["adapted_return_pct"]
        - three_way_yearly["score_ranking_return_pct"]
    )
    three_way_yearly.to_csv(
        output_dir / "strategy_adaptation_yearly_returns.csv",
        index=False,
        encoding="utf-8-sig",
    )

    adapted_capture = build_score_ranking_capture_audit(
        metadata={
            **metadata,
            "comparison_design": "rolling_selection_adaptation_diagnostic",
        },
        baseline_summary=baseline,
        score_sort_summary=adapted,
        baseline_trade_history=pd.read_csv(required["baseline_trades"], encoding="utf-8-sig"),
        score_sort_trade_history=adapted_payload["trade_history"],
        baseline_selected_target_diagnostics=pd.read_csv(
            required["baseline_selected"], encoding="utf-8-sig"
        ),
        score_sort_selected_target_diagnostics=adapted_selected_joined,
        selection_diagnostics={
            "score_ranking_minus_no_filter": _delta(adapted_diag, baseline_diag)
        },
        baseline_daily_capacity=pd.read_csv(
            required["baseline_capacity"], encoding="utf-8-sig"
        ),
        score_sort_daily_capacity=pd.DataFrame(
            adapted_payload["profile"].get("portfolio_capacity_rows") or []
        ),
    )
    adapted_capture_payload = write_score_ranking_capture_audit_outputs(
        result=adapted_capture,
        output_dir=output_dir,
    )

    parameter_comparison = _build_param_comparison_rows(
        baseline_params_path=baseline_contract["path"],
        adapted_params_path=adapted_params_path,
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": ADAPTATION_STATUS,
        "metadata": {
            **metadata,
            "adapted_param_source_kind": "rolling_active_param_ensemble",
            "adapted_result_interpretation": ADAPTATION_STATUS,
            "adapted_params_path": project_relative_display_path(adapted_params_path, project_root=root),
            "adapted_params_sha256": compute_file_sha256(adapted_params_path),
            "future_target_used_for_runtime": False,
            "oos_generalization_claimed": False,
            "final_selection_refit_executed": False,
        },
        "baseline": baseline,
        "sort_only": sort_only,
        "adapted": adapted,
        "sort_only_minus_baseline": _delta(sort_only, baseline),
        "adapted_minus_baseline": _delta(adapted, baseline),
        "adapted_minus_sort_only": _delta(adapted, sort_only),
        "selection_diagnostics": {
            "baseline": baseline_diag,
            "sort_only": sort_diag,
            "adapted": adapted_diag,
            "adapted_minus_baseline": _delta(adapted_diag, baseline_diag),
            "adapted_minus_sort_only": _delta(adapted_diag, sort_diag),
            "future_target_join_stage": "post_replay_offline_diagnostic_only",
            "future_target_used_for_runtime_sort": False,
        },
        "current_capture_audit": current_payload.get("score_ranking_capture_audit"),
        "adapted_capture_audit": adapted_capture_payload,
        "yearly": three_way_yearly.to_dict("records"),
        "adapted_rolling_params": {
            "path": project_relative_display_path(adapted_params_path, project_root=root),
            "sha256": compute_file_sha256(adapted_params_path),
        },
        "parameter_comparison": parameter_comparison,
    }
    _write_json(output_dir / "strategy_adaptation_comparison.json", result)
    (output_dir / "strategy_adaptation_comparison.md").write_text(
        _render_three_way_markdown(result), encoding="utf-8"
    )
    print("\n" + _render_three_way_console(result))
    print("\n" + render_capture_audit_console(adapted_capture))
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
    adapted_capture = dict(result.get("adapted_capture_audit") or {})
    capture_baseline = dict(current_capture.get("baseline") or {})
    capture_sort_only = dict(current_capture.get("score_sort") or {})
    capture_adapted = dict(adapted_capture.get("score_sort") or {})
    enriched = {
        "baseline": {**result["baseline"], **capture_baseline},
        "sort_only": {**result["sort_only"], **capture_sort_only},
        "adapted": {**result["adapted"], **capture_adapted},
    }
    return rows, enriched

def _fmt(value, unit=""):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(number):
        return "N/A"
    digits = 0 if unit == "" and abs(number) >= 1000 else 4 if "R²" in unit else 2
    return f"{number:.{digits}f}{unit}"

def _three_way_rows_for_keys(
    values: dict[str, dict[str, Any]],
    specs: tuple[tuple[str, str, str], ...],
) -> list[tuple[str, str, str, str]]:
    integer_keys = {"trade_count", "candidate_supply_gap_days"}

    def _value(scenario: str, key: str, unit: str) -> str:
        raw = values[scenario].get(key)
        if key not in integer_keys:
            return _fmt(raw, unit)
        try:
            number = float(raw)
        except (TypeError, ValueError):
            return "N/A"
        return "N/A" if not math.isfinite(number) else f"{number:.0f}{unit}"

    return [
        (
            label,
            _value("baseline", key, unit),
            _value("sort_only", key, unit),
            _value("adapted", key, unit),
        )
        for label, key, unit in specs
    ]

def _render_compact_three_way_console(result: dict[str, Any]) -> str:
    _rows, values = _metric_rows(result)
    portfolio_specs = (
        ("淨總報酬", "total_return_pct", "%"),
        ("最大回撤", "max_drawdown_pct", "%"),
        ("報酬／最大回撤", "return_over_max_drawdown", ""),
        ("年化報酬", "annual_return_pct", "%"),
        ("Log R²", "log_r_squared", ""),
        ("月勝率", "monthly_win_rate_pct", "%"),
    )
    trade_specs = (
        ("交易數", "trade_count", ""),
        ("勝率", "win_rate_pct", "%"),
        ("Payoff", "payoff_ratio", ""),
        ("EV", "expected_value_r", " R"),
    )
    capital_specs = (
        ("平均曝險", "avg_exposure_pct", "%"),
        ("平均候選供給", "avg_orderable_candidates", ""),
        ("候選不足日", "candidate_supply_gap_days", " 日"),
        ("平均預留金額", "avg_reserved_total", ""),
        ("平均投入金額", "avg_invested_total", ""),
        ("投入／預留比", "avg_invested_vs_reserved_pct", "%"),
        ("初始停損距離", "avg_stop_distance_pct", "%"),
    )
    target_specs = (
        ("平均 Realized R", "avg_realized_r", " R"),
        ("平均 Target R", "avg_target_r", " R"),
        ("Aggregate capture", "aggregate_target_capture_ratio", ""),
        ("Median capture", "median_target_capture_ratio", ""),
        ("Target ≥ 0.5R capture", "target_ge_0_5_capture_ratio", ""),
    )
    yearly_rows = [
        (
            str(int(row["year"])),
            _fmt(row.get("no_filter_return_pct"), "%"),
            _fmt(row.get("score_ranking_return_pct"), "%"),
            _fmt(row.get("adapted_return_pct"), "%"),
        )
        for row in list(result.get("yearly") or [])
    ]
    diagnostics = result["selection_diagnostics"]
    diag_rows = [
        (
            label,
            _fmt(diagnostics["baseline"].get(key)),
            _fmt(diagnostics["sort_only"].get(key)),
            _fmt(diagnostics["adapted"].get(key)),
        )
        for label, key in (
            ("Score coverage", "orderable_score_coverage_rate"),
            ("Target percentile", "selected_target_percentile_mean"),
            ("Top-k retention", "target_top_k_retention_mean"),
            ("Opportunity gap", "target_opportunity_gap_r_mean"),
            ("Selected Target R", "selected_target_mean_r"),
        )
    ]
    return "\n".join((
        render_title("策略參數適應績效摘要"),
        render_key_values((
            ("狀態", ADAPTATION_STATUS),
            ("比較", "Baseline vs Sort Only vs Adapted Rolling"),
            ("判讀", "相同rolling policy的Selection診斷；不是final refit或正式OOS"),
        )),
        render_section("投組報酬與風險", number=1),
        "讀法：先比較總報酬、回撤與報酬／回撤，再看成長穩定性。",
        render_table(
            ("指標", "Baseline", "Sort Only", "Adapted Rolling"),
            _three_way_rows_for_keys(values, portfolio_specs),
            alignments=("left", "right", "right", "right"),
        ),
        render_section("單筆交易品質", number=2),
        "讀法：EV與Payoff描述單筆品質，不代表資本已充分投入。",
        render_table(
            ("指標", "Baseline", "Sort Only", "Adapted Rolling"),
            _three_way_rows_for_keys(values, trade_specs),
            alignments=("left", "right", "right", "right"),
        ),
        render_section("資金配置與持倉", number=3),
        "讀法：持倉格數、停損距離與實際投入金額必須一起判讀。",
        render_table(
            ("指標", "Baseline", "Sort Only", "Adapted Rolling"),
            _three_way_rows_for_keys(values, capital_specs),
            alignments=("left", "right", "right", "right"),
        ),
        render_section("Target 與實際交易轉換", number=4),
        "讀法：Target是事後機會，Realized R是實際結果，capture衡量轉換效率。",
        render_table(
            ("指標", "Baseline", "Sort Only", "Adapted Rolling"),
            _three_way_rows_for_keys(values, target_specs),
            alignments=("left", "right", "right", "right"),
        ),
        render_section("年度報酬", number=5),
        render_table(
            ("年度", "Baseline", "Sort Only", "Adapted Rolling"),
            yearly_rows,
            alignments=("left", "right", "right", "right"),
        ) if yearly_rows else "無年度資料。",
        render_section("模型選股方向", number=6),
        "讀法：Future Target僅在回放後加入，不參與runtime或optimizer。",
        render_table(
            ("指標", "Baseline", "Sort Only", "Adapted Rolling"),
            diag_rows,
            alignments=("left", "right", "right", "right"),
        ),
        render_section("參數差異", number=7),
        render_table(
            ("參數", "Baseline中位", "Baseline範圍", "Adapted中位", "Adapted範圍"),
            _param_comparison_table_rows(result),
            alignments=("left", "right", "right", "right", "right"),
        ),
        render_section("判讀限制", number=8),
        "Adapted Rolling是無前視rolling Selection診斷；通過後才可進完整Selection final refit。",
    ))

def _render_three_way_console(result: dict[str, Any]) -> str:
    if compact_console_enabled():
        return _render_compact_three_way_console(result)
    rows, values = _metric_rows(result)
    metric_rows = []
    for label, key, unit in rows:
        metric_rows.append((
            label,
            _fmt(values["baseline"].get(key), unit),
            _fmt(values["sort_only"].get(key), unit),
            _fmt(values["adapted"].get(key), unit),
        ))
    yearly_rows = []
    for row in list(result.get("yearly") or []):
        yearly_rows.append((
            str(int(row["year"])),
            _fmt(row.get("no_filter_return_pct"), "%"),
            _fmt(row.get("score_ranking_return_pct"), "%"),
            _fmt(row.get("adapted_return_pct"), "%"),
        ))
    diagnostics = result["selection_diagnostics"]
    diag_rows = []
    for label, key in (
        ("Orderable occurrences", "orderable_occurrences"),
        ("Score coverage", "orderable_score_coverage_rate"),
        ("Selected buys", "selected_buy_rows"),
        ("Target percentile", "selected_target_percentile_mean"),
        ("Top-k retention", "target_top_k_retention_mean"),
        ("Opportunity gap", "target_opportunity_gap_r_mean"),
        ("Selected Target R", "selected_target_mean_r"),
    ):
        diag_rows.append((
            label,
            _fmt(diagnostics["baseline"].get(key)),
            _fmt(diagnostics["sort_only"].get(key)),
            _fmt(diagnostics["adapted"].get(key)),
        ))
    return "\n".join((
        render_title("Selection Score-ranking Rolling Strategy Adaptation Validation"),
        render_key_values((
            ("狀態", ADAPTATION_STATUS),
            ("判讀", "相同rolling policy的Selection診斷；不是final refit或正式OOS"),
            ("Future Target runtime", "未使用"),
        )),
        render_section("三組策略結果", number=1),
        render_table(
            ("指標", "Baseline", "Sort Only", "Adapted Rolling"),
            metric_rows,
            alignments=("left", "right", "right", "right"),
        ),
        render_section("年度績效", number=2),
        render_table(
            ("年度", "Baseline", "Sort Only", "Adapted Rolling"),
            yearly_rows,
            alignments=("left", "right", "right", "right"),
        ),
        render_section("Selection 選股診斷", number=3),
        render_table(
            ("指標", "Baseline", "Sort Only", "Adapted Rolling"),
            diag_rows,
            alignments=("left", "right", "right", "right"),
        ),
        render_section("參數差異", number=4),
        render_table(
            ("參數", "Baseline中位", "Baseline範圍", "Adapted中位", "Adapted範圍"),
            _param_comparison_table_rows(result),
            alignments=("left", "right", "right", "right", "right"),
        ),
        "Adapted Rolling只判斷參數適應方向；不產生最終refit參數，也不宣稱OOS泛化。",
    ))

def _render_three_way_markdown(result: dict[str, Any]) -> str:
    rows, values = _metric_rows(result)
    lines = [
        "# Selection Score-ranking Rolling Strategy Adaptation Validation",
        "",
        f"- 狀態：`{ADAPTATION_STATUS}`",
        "- Adapted Rolling為無前視rolling Selection診斷，不是最終refit或正式OOS結果。",
        "- Future Target只於回放後離線join，未進入runtime或optimizer objective。",
        "",
        "## 三組策略結果",
        "",
        "| 指標 | Baseline | Sort Only | Adapted Rolling |",
        "|---|---:|---:|---:|",
    ]
    for label, key, unit in rows:
        lines.append(
            f"| {label} | {_fmt(values['baseline'].get(key), unit)} | "
            f"{_fmt(values['sort_only'].get(key), unit)} | "
            f"{_fmt(values['adapted'].get(key), unit)} |"
        )
    lines.extend((
        "",
        "## 年度績效",
        "",
        "| 年度 | Baseline | Sort Only | Adapted Rolling |",
        "|---:|---:|---:|---:|",
    ))
    for row in list(result.get("yearly") or []):
        lines.append(
            f"| {int(row['year'])} | {_fmt(row.get('no_filter_return_pct'), '%')} | "
            f"{_fmt(row.get('score_ranking_return_pct'), '%')} | "
            f"{_fmt(row.get('adapted_return_pct'), '%')} |"
        )
    diagnostics = result["selection_diagnostics"]
    lines.extend((
        "",
        "## Selection 選股診斷",
        "",
        "| 指標 | Baseline | Sort Only | Adapted Rolling |",
        "|---|---:|---:|---:|",
    ))
    for label, key in (
        ("Orderable occurrences", "orderable_occurrences"),
        ("Score coverage", "orderable_score_coverage_rate"),
        ("Selected buys", "selected_buy_rows"),
        ("Target percentile", "selected_target_percentile_mean"),
        ("Top-k retention", "target_top_k_retention_mean"),
        ("Opportunity gap", "target_opportunity_gap_r_mean"),
        ("Selected Target R", "selected_target_mean_r"),
    ):
        lines.append(
            f"| {label} | {_fmt(diagnostics['baseline'].get(key))} | "
            f"{_fmt(diagnostics['sort_only'].get(key))} | "
            f"{_fmt(diagnostics['adapted'].get(key))} |"
        )
    lines.extend((
        "",
        "## 參數差異",
        "",
        "| 參數 | Baseline中位 | Baseline範圍 | Adapted中位 | Adapted範圍 |",
        "|---|---:|---:|---:|---:|",
    ))
    for parameter, minimum, median, maximum, adapted_value in _param_comparison_table_rows(result):
        lines.append(
            f"| {parameter} | {minimum} | {median} | {maximum} | {adapted_value} |"
        )
    lines.extend((
        "",
        "## 判讀限制",
        "",
        "- 本結果只回答在相同rolling folds與trials／fold下，Score Sort參數適應是否有改善方向。",
        "- 若rolling診斷支持適應，下一階段才執行完整Selection final refit並凍結最終參數。",
        "",
    ))
    return "\n".join(lines)


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

    base_policy = _build_base_policy(root=root, baseline_contract=baseline_contract)
    runtime_contract = _build_runtime_contract(
        root=root,
        args=args,
        settings=settings,
        pit_contract=pit_contract,
        baseline_contract=baseline_contract,
        base_policy=base_policy,
    )
    score_coverage = dict(runtime_contract["training_score_coverage"])
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
    coverage_path = output_dir / "rolling_training_score_coverage.csv"
    _write_json(
        preflight_path,
        {
            **runtime_contract,
            "status": "ROLLING_ADAPTATION_PREFLIGHT_PASS",
            "created_at": get_taipei_now().isoformat(),
        },
    )
    pd.DataFrame(score_coverage["folds"]).to_csv(
        coverage_path, index=False, encoding="utf-8-sig"
    )
    current_pair_dir = output_dir / "baseline_sort_only"

    current_payload, current_pair_reused = _load_or_run_current_pair(
        root=root,
        args=args,
        runtime_contract=runtime_contract,
        output_dir=current_pair_dir,
    )
    current_decision = dict(current_payload.get("score_ranking_capture_audit", {})).get(
        "decision", {}
    )
    if str(current_decision.get("status") or "") != "ADAPTATION_DIAGNOSTIC_SUPPORTED":
        raise RuntimeError(
            "目前Baseline／Sort Only capture audit未通過參數適應前置條件："
            f"status={current_decision.get('status')!r}"
        )

    adapted_models_dir = output_dir / "adapted_active_params"
    adapted_models_dir.mkdir(parents=True, exist_ok=True)
    optimizer_output_dir = output_dir / "optimizer_runtime"
    outer_environ = dict(os.environ)
    outer_environ["V16_MODELS_DIR"] = str(adapted_models_dir.resolve())
    session_spec = _optimizer_session_spec(
        root=root,
        output_dir=output_dir,
        args=args,
        runtime_contract=runtime_contract,
    )
    exit_code = run_outer_rolling_oos(
        argv=_outer_rolling_argv(args=args, baseline_contract=baseline_contract),
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
        raise RuntimeError(f"Adapted Rolling optimizer失敗：returncode={exit_code}")

    adapted_params_path = adapted_models_dir / "roos_base_best.json"
    adapted_params_payload = _validate_adapted_rolling_params(
        path=adapted_params_path,
        baseline_contract=baseline_contract,
        runtime_contract=runtime_contract,
        args=args,
    )
    optimizer_summary_path = output_dir / "rolling_optimizer_summary.json"
    adapted_manifest_path = output_dir / "rolling_adaptation_manifest.json"
    optimizer_summary = {
        "mode": "rolling_selection_validation",
        "result_interpretation": ADAPTATION_STATUS,
        "folds": int(dict(adapted_params_payload.get("summary") or {}).get("folds", 0) or 0),
        "trials_per_fold": int(args.trials_per_fold),
        "total_requested_trials": int(args.trials_per_fold) * int(
            dict(adapted_params_payload.get("summary") or {}).get("folds", 0) or 0
        ),
        "baseline_trials_per_fold": int(baseline_meta["trials_per_fold"]),
        "same_rolling_policy_as_baseline": True,
        "baseline_sort_only_reused": bool(current_pair_reused),
        "seed": int(settings.seed),
        "fixed_tp_percent": OPTIMIZER_FIXED_TP_PERCENT,
        "runtime_identity_sha256": runtime_contract["runtime_identity_sha256"],
        "final_selection_refit_executed": False,
        "formal_oos_executed": False,
    }
    _write_json(optimizer_summary_path, optimizer_summary)

    comparison_result = _run_three_way_comparison(
        root=root,
        args=args,
        pit_contract=pit_contract,
        current_payload=current_payload,
        baseline_contract=baseline_contract,
        adapted_params_path=adapted_params_path,
        current_pair_dir=current_pair_dir,
        output_dir=output_dir,
    )
    artifact_paths = {
        "rolling_preflight": preflight_path,
        "training_score_coverage": coverage_path,
        "adapted_rolling_params": adapted_params_path,
        "optimizer_summary": optimizer_summary_path,
        "comparison_json": output_dir / "strategy_adaptation_comparison.json",
        "comparison_markdown": output_dir / "strategy_adaptation_comparison.md",
        "yearly_csv": output_dir / "strategy_adaptation_yearly_returns.csv",
        "adapted_capture_json": output_dir / "score_ranking_capture_audit.json",
        "adapted_capture_markdown": output_dir / "score_ranking_capture_audit.md",
        "adapted_equity": output_dir / "adapted_rolling_equity.csv",
        "adapted_trades": output_dir / "adapted_rolling_trades.csv",
        "adapted_daily_capacity": output_dir / "adapted_rolling_daily_capacity.csv",
        "adapted_orderable_diagnostics": output_dir / "adapted_rolling_orderable_target_diagnostics.csv",
        "adapted_selected_diagnostics": output_dir / "adapted_rolling_selected_target_diagnostics.csv",
        "adapted_yearly_returns": output_dir / "adapted_rolling_yearly_returns.csv",
        "adaptation_yearly_comparison": output_dir / "strategy_adaptation_yearly_returns.csv",
        "adapted_baseline_capture_lifecycle": output_dir / "no_filter_capture_lifecycle.csv",
        "adapted_score_capture_lifecycle": output_dir / "score_ranking_capture_lifecycle.csv",
        "adapted_capture_yearly": output_dir / "score_ranking_capture_yearly.csv",
        "adapted_capture_scenarios": output_dir / "score_ranking_capture_scenarios.csv",
        "baseline_sort_only_comparison_json": current_pair_dir / "strategy_comparison.json",
        "baseline_sort_only_comparison_markdown": current_pair_dir / "strategy_comparison.md",
        "baseline_sort_only_capture_json": current_pair_dir / "score_ranking_capture_audit.json",
        "baseline_sort_only_capture_markdown": current_pair_dir / "score_ranking_capture_audit.md",
        "baseline_sort_only_manifest": current_pair_dir / CURRENT_PAIR_MANIFEST_FILENAME,
    }
    missing_artifacts = [str(path) for path in artifact_paths.values() if not path.is_file()]
    if missing_artifacts:
        raise FileNotFoundError(
            "Rolling adaptation完成後缺少必要工件: " + ", ".join(missing_artifacts)
        )
    execution_argv = _canonical_execution_argv(args)
    manifest = {
        **runtime_contract,
        "status": "ROLLING_ADAPTATION_VALIDATION_COMPLETED",
        "result_interpretation": ADAPTATION_STATUS,
        "optimizer_summary": optimizer_summary,
        "baseline_sort_only_reused": bool(current_pair_reused),
        "execution_command": " ".join(execution_argv),
        "execution_argv": execution_argv,
        "created_at": get_taipei_now().isoformat(),
        "artifacts": {
            key: {
                "path": project_relative_display_path(path, project_root=root),
                "sha256": compute_file_sha256(path),
            }
            for key, path in artifact_paths.items()
        },
    }
    _write_json(adapted_manifest_path, manifest)
    print_artifact_paths(
        (
            ("Rolling preflight", preflight_path),
            ("Training Score coverage", coverage_path),
            ("Adapted Rolling active params", adapted_params_path),
            ("Rolling adaptation manifest", adapted_manifest_path),
            ("Rolling optimizer summary", optimizer_summary_path),
            ("三組比較 Markdown", output_dir / "strategy_adaptation_comparison.md"),
            ("三組比較 JSON", output_dir / "strategy_adaptation_comparison.json"),
        ),
        project_root=root,
    )
    return comparison_result

def main(argv=None):
    run_adaptation(argv=argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
