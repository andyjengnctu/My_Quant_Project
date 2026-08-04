"""Controlled no-filter vs active breakout-quality portfolio comparison."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from config.breakout_quality import (
    BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD,
    BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
    BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
    get_breakout_quality_workflow_settings,
)
from core.active_param_ensemble import (
    ACTIVE_PARAM_ENSEMBLE_MODE_STATIC,
    get_active_param_ensemble_date_range,
    get_active_param_ensemble_policy,
    is_active_param_ensemble_payload,
    resolve_active_param_ensemble_mode,
)
from core.buy_sort import (
    BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_ADJUSTED,
    BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_BUCKET,
    BREAKOUT_QUALITY_RANKING_POLICY_SCORE,
    SUPPORTED_BREAKOUT_QUALITY_RANKING_POLICIES,
)
from core.dataset_profiles import DEFAULT_DATASET_PROFILE, get_dataset_dir
from core.model_paths import resolve_default_primary_param_source_record
from core.params_io import build_params_from_mapping, load_params_from_json, params_to_json_dict
from core.portfolio_stats import calc_plain_romd
from core.rolling_oos_params import (
    build_active_param_schedule,
    get_active_param_date_range,
    is_rolling_oos_param_set_payload,
)
from core.seed_ensemble_policy import normalize_seed_ensemble_members
from filters.breakout_quality.artifacts import load_runtime_artifact_contract
from filters.breakout_quality.ranking_score_store import (
    SCORE_SOURCE_CANONICAL_RUNTIME,
    SCORE_SOURCE_SELECTION_POINT_IN_TIME,
    SUPPORTED_RANKING_SCORE_SOURCES,
    load_selection_point_in_time_ranking_contract,
    load_selection_point_in_time_score_table,
)
from filters.breakout_quality.runtime import breakout_quality_ranking_source_context
from filters.breakout_quality.paths import resolve_filter_model_output_dir
from tools.filters.breakout_quality.audit_score_ranking_capture import (
    build_score_ranking_capture_audit,
    render_capture_audit_console,
    write_score_ranking_capture_audit_outputs,
)
from tools.filters.breakout_quality.continuous_ranker_pipeline import load_continuous_ranker_data
from tools.filters.breakout_quality.strategy_report_style import (
    SIGNAL_NEGATIVE,
    SIGNAL_NEUTRAL,
    SIGNAL_POSITIVE,
    SIGNAL_WARNING,
    signal_for_delta,
    signal_marker,
    terminal_signal,
)
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
from tools.filters.breakout_quality.trade_attribution import (
    ATTRIBUTION_SCHEMA_VERSION,
    write_trade_attribution_outputs,
)
from tools.portfolio_sim.simulation_runner import (
    PORTFOLIO_DEFAULT_BENCHMARK_TICKER,
    load_portfolio_market_context,
    run_portfolio_simulation_prepared,
    run_portfolio_simulation_with_param_ensemble,
    run_portfolio_simulation_with_param_schedule,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = 7
COMPARISON_MODE_HARD_FILTER = "hard-filter"
COMPARISON_MODE_SCORE_RANKING = "score-ranking"
COMPARISON_MODES = (COMPARISON_MODE_HARD_FILTER, COMPARISON_MODE_SCORE_RANKING)

PARAM_POLICY_AUTO = "auto"
PARAM_POLICY_BASE_FINALIST_BEST = "base-finalist-best"
PARAM_POLICY_BASE_FINALISTS_AGREE = "base-finalists-agree"
PARAM_POLICY_SPECS = {
    PARAM_POLICY_BASE_FINALIST_BEST: {
        "selector": "base_finalist_best",
        "filename": "roos_base_best.json",
        "output_suffix": "base_finalist_best",
        "expected_member_count": 1,
        "expected_min_agree": 1,
    },
    PARAM_POLICY_BASE_FINALISTS_AGREE: {
        "selector": "base_finalists_agree",
        "filename": "roos_base_finalists_agree.json",
        "output_suffix": "base_finalists_agree",
        "expected_member_count": None,
        "expected_min_agree": None,
    },
}
PARAM_POLICIES = (PARAM_POLICY_AUTO, *PARAM_POLICY_SPECS.keys())

OPTIONAL_ENTRY_FILTER_POLICY_CURRENT = "current"
OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF = "all-off"
SUPPORTED_OPTIONAL_ENTRY_FILTER_POLICIES = (
    OPTIONAL_ENTRY_FILTER_POLICY_CURRENT,
    OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
)
OPTIONAL_ENTRY_FILTER_FIELDS = (
    "use_breakout_ema_filter",
    "use_bb",
    "use_vol",
    "use_breakout_return_filter",
    "use_breakout_false_filter",
)

_RESULT_FIELDS = (
    "equity_curve", "trade_history", "total_return_pct", "max_drawdown_pct",
    "trade_count", "win_rate_pct", "expected_value_r", "payoff_ratio",
    "final_equity", "avg_exposure_pct", "max_exposure_pct", "benchmark_return_pct",
    "benchmark_max_drawdown_pct", "missed_buy_count", "missed_sell_count",
    "log_r_squared", "monthly_win_rate_pct", "benchmark_log_r_squared",
    "benchmark_monthly_win_rate_pct", "normal_trade_count", "extended_trade_count",
    "annual_trade_count", "reserved_buy_fill_rate_pct", "annual_return_pct",
    "benchmark_annual_return_pct", "profile",
)

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="固定參數比較 baseline 與 breakout-quality hard filter／score ranking 的策略經濟效果。"
    )
    parser.add_argument("--dataset", choices=("reduced", "full"), default=DEFAULT_DATASET_PROFILE)
    parser.add_argument("--comparison-mode", choices=COMPARISON_MODES, default=COMPARISON_MODE_HARD_FILTER)
    parser.add_argument("--filter-id", default=BREAKOUT_QUALITY_DEFAULT_FILTER_ID)
    parser.add_argument(
        "--score-source", choices=SUPPORTED_RANKING_SCORE_SOURCES,
        default=SCORE_SOURCE_CANONICAL_RUNTIME,
        help="score-ranking使用的分數來源；hard-filter只接受canonical_runtime。",
    )
    parser.add_argument(
        "--ranking-policy",
        choices=SUPPORTED_BREAKOUT_QUALITY_RANKING_POLICIES,
        default=BREAKOUT_QUALITY_RANKING_POLICY_SCORE,
        help=(
            "score-ranking排序契約：score=原始Score；capital-adjusted-score="
            "Score×正式預估部署率；capital-bucket-then-score="
            "每日部署率三分桶後桶內按Score。"
        ),
    )
    parser.add_argument(
        "--optional-entry-filters",
        choices=SUPPORTED_OPTIONAL_ENTRY_FILTER_POLICIES,
        default=OPTIONAL_ENTRY_FILTER_POLICY_CURRENT,
        help=(
            "current=沿用active params；all-off=兩組都關閉EMA、BB、Volume、"
            "breakout return與false-breakout五個optional entry filters。"
        ),
    )
    parser.add_argument("--model-architecture", default=BREAKOUT_QUALITY_MODEL_ARCHITECTURE)
    parser.add_argument("--experiment-profile", default=BREAKOUT_QUALITY_EXPERIMENT_PROFILE)
    parser.add_argument("--params", default=None, help="可明確指定active-param JSON；亦可由score source與--param-policy自動解析。")
    parser.add_argument(
        "--param-policy", choices=PARAM_POLICIES, default=PARAM_POLICY_AUTO,
        help="可指定 base-finalist-best 或 base-finalists-agree；auto 沿用 --params。",
    )
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--rotation", choices=("off", "on"), default="off")
    parser.add_argument("--fixed-risk", type=float, default=None, help="選填；兩組同時覆寫 fixed_risk。")
    parser.add_argument(
        "--max-position-cap-pct",
        type=float,
        default=None,
        help="選填；兩組同時覆寫 max_position_cap_pct。",
    )
    parser.add_argument(
        "--allow-static-diagnostic",
        action="store_true",
        help="允許以單一／static run_best 做非 OOS 診斷；不得視為無前視部署證據。",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="選填；將比較期間縮限於Score與active params共同可用範圍內。",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="選填；必須與--start-date同時使用。",
    )
    parser.add_argument(
        "--attribution-only",
        action="store_true",
        help="只讀取既有 strategy_compare CSV/JSON 產生交易歸因，不重新執行 portfolio replay。",
    )
    parser.add_argument(
        "--capture-audit-only",
        action="store_true",
        help="只讀取既有score-ranking比較工件，重新輸出console主報表與資本效率／Target capture audit。",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)

def _comparison_switch_spec(comparison_mode: str) -> tuple[str, bool, bool]:
    mode = str(comparison_mode)
    if mode == COMPARISON_MODE_HARD_FILTER:
        return "use_breakout_quality_filter", False, True
    if mode == COMPARISON_MODE_SCORE_RANKING:
        return "use_breakout_quality_ranking", False, True
    raise ValueError(f"不支援的 comparison_mode: {comparison_mode}")

def _comparison_labels(comparison_mode: str) -> dict[str, str]:
    if comparison_mode == COMPARISON_MODE_HARD_FILTER:
        return {
            "active_name": "quality_filter",
            "active_title": "Active quality filter",
            "output_dir": "strategy_compare",
            "difference_text": "use_breakout_quality_filter=False vs True",
        }
    if comparison_mode == COMPARISON_MODE_SCORE_RANKING:
        return {
            "active_name": "score_ranking",
            "active_title": "Quality score ranking",
            "output_dir": "strategy_compare_score_ranking",
            "difference_text": "use_breakout_quality_ranking=False vs True（hard filter 兩組皆 False）",
        }
    raise ValueError(f"不支援的 comparison_mode: {comparison_mode}")

def _assert_controlled_param_pair(no_filter_params, quality_params, *, comparison_mode=COMPARISON_MODE_HARD_FILTER) -> None:
    left = params_to_json_dict(no_filter_params)
    right = params_to_json_dict(quality_params)
    switch_field, left_expected, right_expected = _comparison_switch_spec(comparison_mode)
    differing = sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))
    if differing != [switch_field]:
        raise ValueError(f"策略對照只允許 {switch_field} 不同，實際差異={differing}")
    if left[switch_field] is not left_expected or right[switch_field] is not right_expected:
        raise ValueError(f"策略對照的 {switch_field} 開關方向不正確")
    if bool(left.get("use_breakout_quality_filter")) and bool(left.get("use_breakout_quality_ranking")):
        raise ValueError("baseline 不可同時啟用 hard filter 與 score ranking")
    if bool(right.get("use_breakout_quality_filter")) and bool(right.get("use_breakout_quality_ranking")):
        raise ValueError("active scenario 不可同時啟用 hard filter 與 score ranking")

def _collect_payload_differences(left: Any, right: Any, path: tuple[Any, ...] = ()) -> list[tuple[tuple[Any, ...], Any, Any]]:
    if isinstance(left, dict) and isinstance(right, dict):
        out = []
        for key in sorted(set(left) | set(right), key=str):
            if key not in left or key not in right:
                out.append((path + (key,), left.get(key), right.get(key)))
            else:
                out.extend(_collect_payload_differences(left[key], right[key], path + (key,)))
        return out
    if isinstance(left, list) and isinstance(right, list):
        out = []
        if len(left) != len(right):
            return [(path + ("length",), len(left), len(right))]
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            out.extend(_collect_payload_differences(left_item, right_item, path + (index,)))
        return out
    return [] if left == right else [(path, left, right)]

def _assert_controlled_payload_pair(no_filter_payload: dict, quality_payload: dict, *, comparison_mode=COMPARISON_MODE_HARD_FILTER) -> None:
    differences = _collect_payload_differences(no_filter_payload, quality_payload)
    switch_field, left_expected, right_expected = _comparison_switch_spec(comparison_mode)
    if not differences:
        raise ValueError(f"策略對照 payload 沒有切換 {switch_field}")
    invalid = [
        ("/".join(map(str, path)), left, right)
        for path, left, right in differences
        if not path or path[-1] != switch_field or left is not left_expected or right is not right_expected
    ]
    if invalid:
        raise ValueError(f"策略對照只允許 {switch_field} 由 {left_expected} 切為 {right_expected}，實際額外差異={invalid}")

def _resolve_param_selector(source: dict[str, Any]) -> str:
    payload = source.get("payload")
    if isinstance(payload, dict):
        selector = str(payload.get("selector") or "").strip()
        if selector:
            return selector
        policy = get_active_param_ensemble_policy(payload)
        selector = str((policy or {}).get("policy_name") or "").strip()
        if selector:
            return selector
    return "single_param" if source.get("kind") == "single_param" else "unknown"

def _rolling_member_counts(source: dict[str, Any]) -> list[int]:
    payload = source.get("payload")
    if not isinstance(payload, dict):
        return []
    mapping = payload.get("params_ensemble_by_effective_date")
    if not isinstance(mapping, dict):
        return []
    counts = []
    for effective_date, raw_members in mapping.items():
        members = normalize_seed_ensemble_members(raw_members)
        if not members or len(members) != len(list(raw_members or [])):
            raise ValueError(f"生效日 {effective_date} 的 active-param members 無效")
        counts.append(len(members))
    return counts

def _validate_requested_param_policy(source: dict[str, Any], requested_policy: str) -> dict[str, Any]:
    selector = _resolve_param_selector(source)
    member_counts = _rolling_member_counts(source)
    policy = get_active_param_ensemble_policy(source.get("payload") or {}) if isinstance(source.get("payload"), dict) else None
    actual_min_agree = int((policy or {}).get("min_agree", 0) or 0) if policy else None
    if requested_policy == PARAM_POLICY_AUTO:
        return {
            "requested_policy": requested_policy,
            "selector": selector,
            "member_count_min": min(member_counts) if member_counts else None,
            "member_count_max": max(member_counts) if member_counts else None,
            "min_agree": actual_min_agree,
        }

    spec = PARAM_POLICY_SPECS[requested_policy]
    if source.get("kind") != "rolling_active_param_ensemble":
        raise ValueError(
            f"--param-policy {requested_policy} 只接受 rolling active-param ensemble JSON，"
            f"實際={source.get('kind')}"
        )
    if selector != spec["selector"]:
        raise ValueError(
            f"--param-policy {requested_policy} 與參數檔 selector 不一致: "
            f"expected={spec['selector']}, actual={selector}"
        )
    expected_member_count = spec.get("expected_member_count")
    if expected_member_count is not None and (not member_counts or any(count != expected_member_count for count in member_counts)):
        raise ValueError(
            f"{spec['selector']} 必須每個生效日恰有 {expected_member_count} 個 runtime member，"
            f"實際範圍={min(member_counts) if member_counts else 0}~{max(member_counts) if member_counts else 0}"
        )
    expected_min_agree = spec.get("expected_min_agree")
    if expected_min_agree is not None and actual_min_agree != expected_min_agree:
        raise ValueError(
            f"{spec['selector']} 的 min_agree 必須為 {expected_min_agree}，實際={actual_min_agree}"
        )
    return {
        "requested_policy": requested_policy,
        "selector": selector,
        "member_count_min": min(member_counts) if member_counts else None,
        "member_count_max": max(member_counts) if member_counts else None,
        "min_agree": actual_min_agree,
    }

def _resolve_params_path(
    *, root: Path, params_path: str | None, param_policy: str,
    allow_static_diagnostic: bool, score_source: str = SCORE_SOURCE_CANONICAL_RUNTIME,
) -> Path:
    if params_path:
        requested = Path(params_path)
        return requested.resolve() if requested.is_absolute() else (root / requested).resolve()
    if param_policy != PARAM_POLICY_AUTO:
        if score_source == SCORE_SOURCE_SELECTION_POINT_IN_TIME:
            return (
                root / "models" / "research" / "breakout_quality"
                / "selection_strategy_realization"
                / PARAM_POLICY_SPECS[param_policy]["filename"]
            ).resolve()
        return (root / "models" / PARAM_POLICY_SPECS[param_policy]["filename"]).resolve()
    if allow_static_diagnostic:
        return Path(resolve_default_primary_param_source_record(str(root))["path"]).resolve()
    raise ValueError(
        "正式 OOS 策略對照必須以 --params 指定 rolling OOS JSON，或用 "
        "--param-policy base-finalist-best / base-finalists-agree 自動解析；"
        "只有非 OOS 敏感度診斷才可加 --allow-static-diagnostic 使用 run_best_params.json。"
    )

def _comparison_output_dir_name(
    comparison_mode: str,
    labels: dict[str, str],
    *,
    param_policy: str,
    ranking_policy: str = BREAKOUT_QUALITY_RANKING_POLICY_SCORE,
    optional_entry_filter_policy: str = OPTIONAL_ENTRY_FILTER_POLICY_CURRENT,
) -> str:
    if param_policy != PARAM_POLICY_AUTO:
        name = f"{labels['output_dir']}_{PARAM_POLICY_SPECS[param_policy]['output_suffix']}"
    else:
        name = labels["output_dir"]
    if (
        comparison_mode == COMPARISON_MODE_SCORE_RANKING
        and ranking_policy != BREAKOUT_QUALITY_RANKING_POLICY_SCORE
    ):
        name += "_" + str(ranking_policy).replace("-", "_")
    if optional_entry_filter_policy == OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF:
        name += "_optional_entry_filters_all_off"
    return name

def canonical_strategy_compare_output_dir_names(
    comparison_mode: str | None = None,
) -> tuple[str, ...]:
    """Return active strategy-compare directory names in semantic priority order."""

    if comparison_mode is not None and comparison_mode not in COMPARISON_MODES:
        raise ValueError(f"不支援的 comparison mode: {comparison_mode!r}")
    hard_filter_labels = _comparison_labels(COMPARISON_MODE_HARD_FILTER)
    score_ranking_labels = _comparison_labels(COMPARISON_MODE_SCORE_RANKING)
    hard_filter_names = (
        _comparison_output_dir_name(
            COMPARISON_MODE_HARD_FILTER,
            hard_filter_labels,
            param_policy=PARAM_POLICY_BASE_FINALIST_BEST,
        ),
        _comparison_output_dir_name(
            COMPARISON_MODE_HARD_FILTER,
            hard_filter_labels,
            param_policy=PARAM_POLICY_BASE_FINALISTS_AGREE,
        ),
        _comparison_output_dir_name(
            COMPARISON_MODE_HARD_FILTER,
            hard_filter_labels,
            param_policy=PARAM_POLICY_AUTO,
        ),
    )
    score_ranking_names = (
        _comparison_output_dir_name(
            COMPARISON_MODE_SCORE_RANKING,
            score_ranking_labels,
            param_policy=PARAM_POLICY_BASE_FINALISTS_AGREE,
        ),
        _comparison_output_dir_name(
            COMPARISON_MODE_SCORE_RANKING,
            score_ranking_labels,
            param_policy=PARAM_POLICY_BASE_FINALIST_BEST,
        ),
        _comparison_output_dir_name(
            COMPARISON_MODE_SCORE_RANKING,
            score_ranking_labels,
            param_policy=PARAM_POLICY_AUTO,
        ),
    )
    if comparison_mode == COMPARISON_MODE_HARD_FILTER:
        return hard_filter_names
    if comparison_mode == COMPARISON_MODE_SCORE_RANKING:
        return score_ranking_names
    return (*hard_filter_names, *score_ranking_names)

def _first_existing_comparison_dir(
    root: Path,
    *,
    comparison_mode: str,
) -> Path:
    candidates = [
        root / name
        for name in canonical_strategy_compare_output_dir_names(comparison_mode)
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        "找不到既有strategy compare輸出目錄；已檢查: "
        + ", ".join(str(path) for path in candidates)
    )

def _load_param_source(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"讀取參數來源失敗: {path}｜{type(exc).__name__}: {exc}") from exc

    if is_rolling_oos_param_set_payload(payload):
        if resolve_active_param_ensemble_mode(payload) == "rolling":
            get_active_param_ensemble_date_range(payload)
            return {"kind": "rolling_active_param_ensemble", "payload": payload}
        build_active_param_schedule(payload)
        return {"kind": "rolling_oos_param_schedule", "payload": payload}

    if is_active_param_ensemble_payload(payload):
        mode = resolve_active_param_ensemble_mode(payload)
        if mode != ACTIVE_PARAM_ENSEMBLE_MODE_STATIC:
            raise ValueError(f"不支援的 active-param ensemble mode={mode or 'unknown'}")
        members = normalize_seed_ensemble_members(payload.get("params_ensemble"))
        if not members or len(members) != len(list(payload.get("params_ensemble") or [])):
            raise ValueError("static active-param ensemble 內含無效或缺失的 member params")
        return {"kind": "static_active_param_ensemble", "payload": payload}
    return {"kind": "single_param", "params": load_params_from_json(str(path))}

def _apply_scenario_overrides(
    params,
    *,
    active: bool,
    comparison_mode: str,
    filter_id: str,
    threshold: float,
    fixed_risk: float | None,
    max_position_cap_pct: float | None = None,
    optional_entry_filter_policy: str = OPTIONAL_ENTRY_FILTER_POLICY_CURRENT,
):
    _comparison_switch_spec(comparison_mode)
    overrides = {
        "use_breakout_quality_filter": bool(active and comparison_mode == COMPARISON_MODE_HARD_FILTER),
        "use_breakout_quality_ranking": bool(active and comparison_mode == COMPARISON_MODE_SCORE_RANKING),
        "breakout_quality_filter_id": str(filter_id),
        "breakout_quality_score_threshold": float(threshold),
    }
    optional_entry_filter_policy = str(optional_entry_filter_policy).strip()
    if optional_entry_filter_policy not in SUPPORTED_OPTIONAL_ENTRY_FILTER_POLICIES:
        raise ValueError(
            f"不支援的 optional entry filter policy: {optional_entry_filter_policy!r}"
        )
    if optional_entry_filter_policy == OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF:
        overrides.update({field: False for field in OPTIONAL_ENTRY_FILTER_FIELDS})
    if fixed_risk is not None:
        overrides["fixed_risk"] = float(fixed_risk)
    if max_position_cap_pct is not None:
        overrides["max_position_cap_pct"] = float(max_position_cap_pct)
    return replace(params, **overrides)

def _assert_controlled_ensemble_pair(no_filter_payload: dict, quality_payload: dict, *, comparison_mode=COMPARISON_MODE_HARD_FILTER) -> None:
    if resolve_active_param_ensemble_mode(no_filter_payload) != resolve_active_param_ensemble_mode(quality_payload):
        raise ValueError("策略對照的 ensemble mode 不一致")
    _assert_controlled_payload_pair(no_filter_payload, quality_payload, comparison_mode=comparison_mode)
    left_mode = resolve_active_param_ensemble_mode(no_filter_payload)
    if left_mode == ACTIVE_PARAM_ENSEMBLE_MODE_STATIC:
        left_members = normalize_seed_ensemble_members(no_filter_payload.get("params_ensemble"))
        right_members = normalize_seed_ensemble_members(quality_payload.get("params_ensemble"))
        if len(left_members) != len(right_members) or not left_members:
            raise ValueError("策略對照的 static ensemble member 數量不一致或為空")
    else:
        get_active_param_ensemble_date_range(no_filter_payload)
        get_active_param_ensemble_date_range(quality_payload)

def _rewrite_param_mapping(
    mapping: dict,
    *,
    active: bool,
    comparison_mode: str,
    filter_id: str,
    threshold: float,
    fixed_risk: float | None,
    max_position_cap_pct: float | None = None,
    optional_entry_filter_policy: str = OPTIONAL_ENTRY_FILTER_POLICY_CURRENT,
) -> dict:
    rewritten = {}
    for key, raw_params in dict(mapping or {}).items():
        base_params = build_params_from_mapping(raw_params)
        rewritten[str(key)] = params_to_json_dict(
            _apply_scenario_overrides(
                base_params,
                active=active,
                comparison_mode=comparison_mode,
                filter_id=filter_id,
                threshold=threshold,
                fixed_risk=fixed_risk,
                max_position_cap_pct=max_position_cap_pct,
                optional_entry_filter_policy=optional_entry_filter_policy,
            )
        )
    return rewritten

def _rewrite_ensemble_mapping(
    mapping: dict,
    *,
    active: bool,
    comparison_mode: str,
    filter_id: str,
    threshold: float,
    fixed_risk: float | None,
    max_position_cap_pct: float | None = None,
    optional_entry_filter_policy: str = OPTIONAL_ENTRY_FILTER_POLICY_CURRENT,
) -> dict:
    rewritten = {}
    for key, raw_members in dict(mapping or {}).items():
        members = normalize_seed_ensemble_members(raw_members)
        if not members or len(members) != len(list(raw_members or [])):
            raise ValueError(f"生效日 {key} 的 ensemble members 無效")
        output_members = []
        for member in members:
            base_params = build_params_from_mapping(member["params"])
            output_member = copy.deepcopy(member)
            output_member["params"] = params_to_json_dict(
                _apply_scenario_overrides(
                    base_params,
                    active=active,
                    comparison_mode=comparison_mode,
                    filter_id=filter_id,
                    threshold=threshold,
                    fixed_risk=fixed_risk,
                    max_position_cap_pct=max_position_cap_pct,
                    optional_entry_filter_policy=optional_entry_filter_policy,
                )
            )
            output_members.append(output_member)
        rewritten[str(key)] = output_members
    return rewritten

def _build_controlled_param_source_pair(
    source: dict[str, Any],
    *,
    filter_id: str,
    threshold: float,
    fixed_risk: float | None,
    max_position_cap_pct: float | None = None,
    comparison_mode: str = COMPARISON_MODE_HARD_FILTER,
    optional_entry_filter_policy: str = OPTIONAL_ENTRY_FILTER_POLICY_CURRENT,
) -> tuple[str, Any, Any, Any, Any, dict[str, Any] | None]:
    kind = str(source["kind"])
    if kind == "single_param":
        base_params = source["params"]
        no_filter_params = _apply_scenario_overrides(
            base_params,
            active=False,
            comparison_mode=comparison_mode,
            filter_id=filter_id,
            threshold=threshold,
            fixed_risk=fixed_risk,
            max_position_cap_pct=max_position_cap_pct,
            optional_entry_filter_policy=optional_entry_filter_policy,
        )
        quality_params = _apply_scenario_overrides(
            base_params,
            active=True,
            comparison_mode=comparison_mode,
            filter_id=filter_id,
            threshold=threshold,
            fixed_risk=fixed_risk,
            max_position_cap_pct=max_position_cap_pct,
            optional_entry_filter_policy=optional_entry_filter_policy,
        )
        _assert_controlled_param_pair(no_filter_params, quality_params, comparison_mode=comparison_mode)
        return (
            kind,
            no_filter_params,
            quality_params,
            params_to_json_dict(no_filter_params),
            params_to_json_dict(quality_params),
            None,
        )

    if kind not in {
        "static_active_param_ensemble",
        "rolling_oos_param_schedule",
        "rolling_active_param_ensemble",
    }:
        raise ValueError(f"不支援的參數來源類型: {kind}")

    base_payload = copy.deepcopy(source["payload"])
    no_filter_payload = copy.deepcopy(base_payload)
    quality_payload = copy.deepcopy(base_payload)

    if kind == "static_active_param_ensemble":
        base_mapping = {"static": base_payload.get("params_ensemble")}
        no_filter_payload["params_ensemble"] = _rewrite_ensemble_mapping(
            base_mapping,
            active=False,
            comparison_mode=comparison_mode,
            filter_id=filter_id,
            threshold=threshold,
            fixed_risk=fixed_risk,
            max_position_cap_pct=max_position_cap_pct,
            optional_entry_filter_policy=optional_entry_filter_policy,
        )["static"]
        quality_payload["params_ensemble"] = _rewrite_ensemble_mapping(
            base_mapping,
            active=True,
            comparison_mode=comparison_mode,
            filter_id=filter_id,
            threshold=threshold,
            fixed_risk=fixed_risk,
            max_position_cap_pct=max_position_cap_pct,
            optional_entry_filter_policy=optional_entry_filter_policy,
        )["static"]
        _assert_controlled_ensemble_pair(no_filter_payload, quality_payload, comparison_mode=comparison_mode)
        policy = get_active_param_ensemble_policy(base_payload)
    elif kind == "rolling_active_param_ensemble":
        for field in ("params_by_effective_date", "params_by_oos_year"):
            raw_mapping = base_payload.get(field)
            if isinstance(raw_mapping, dict) and raw_mapping:
                no_filter_payload[field] = _rewrite_param_mapping(
                    raw_mapping,
                    active=False,
                    comparison_mode=comparison_mode,
                    filter_id=filter_id,
                    threshold=threshold,
                    fixed_risk=fixed_risk,
                    max_position_cap_pct=max_position_cap_pct,
                    optional_entry_filter_policy=optional_entry_filter_policy,
                )
                quality_payload[field] = _rewrite_param_mapping(
                    raw_mapping,
                    active=True,
                    comparison_mode=comparison_mode,
                    filter_id=filter_id,
                    threshold=threshold,
                    fixed_risk=fixed_risk,
                    max_position_cap_pct=max_position_cap_pct,
                    optional_entry_filter_policy=optional_entry_filter_policy,
                )
        ensemble_mapping = base_payload.get("params_ensemble_by_effective_date")
        if not isinstance(ensemble_mapping, dict) or not ensemble_mapping:
            raise ValueError("rolling active-param ensemble 缺少 params_ensemble_by_effective_date")
        no_filter_payload["params_ensemble_by_effective_date"] = _rewrite_ensemble_mapping(
            ensemble_mapping,
            active=False,
            comparison_mode=comparison_mode,
            filter_id=filter_id,
            threshold=threshold,
            fixed_risk=fixed_risk,
            max_position_cap_pct=max_position_cap_pct,
            optional_entry_filter_policy=optional_entry_filter_policy,
        )
        quality_payload["params_ensemble_by_effective_date"] = _rewrite_ensemble_mapping(
            ensemble_mapping,
            active=True,
            comparison_mode=comparison_mode,
            filter_id=filter_id,
            threshold=threshold,
            fixed_risk=fixed_risk,
            max_position_cap_pct=max_position_cap_pct,
            optional_entry_filter_policy=optional_entry_filter_policy,
        )
        _assert_controlled_ensemble_pair(no_filter_payload, quality_payload, comparison_mode=comparison_mode)
        policy = get_active_param_ensemble_policy(base_payload)
    else:
        updated_any = False
        for field in ("params_by_effective_date", "params_by_oos_year"):
            raw_mapping = base_payload.get(field)
            if isinstance(raw_mapping, dict) and raw_mapping:
                updated_any = True
                no_filter_payload[field] = _rewrite_param_mapping(
                    raw_mapping,
                    active=False,
                    comparison_mode=comparison_mode,
                    filter_id=filter_id,
                    threshold=threshold,
                    fixed_risk=fixed_risk,
                    max_position_cap_pct=max_position_cap_pct,
                    optional_entry_filter_policy=optional_entry_filter_policy,
                )
                quality_payload[field] = _rewrite_param_mapping(
                    raw_mapping,
                    active=True,
                    comparison_mode=comparison_mode,
                    filter_id=filter_id,
                    threshold=threshold,
                    fixed_risk=fixed_risk,
                    max_position_cap_pct=max_position_cap_pct,
                    optional_entry_filter_policy=optional_entry_filter_policy,
                )
        if not updated_any:
            raise ValueError("rolling OOS param schedule 缺少 params_by_effective_date / params_by_oos_year")
        build_active_param_schedule(no_filter_payload)
        build_active_param_schedule(quality_payload)
        _assert_controlled_payload_pair(no_filter_payload, quality_payload, comparison_mode=comparison_mode)
        policy = None

    return (
        kind,
        no_filter_payload,
        quality_payload,
        no_filter_payload,
        quality_payload,
        policy,
    )

def _unpack_result(result) -> dict[str, Any]:
    if len(result) != len(_RESULT_FIELDS):
        raise ValueError(f"portfolio result 欄位數不一致: expected={len(_RESULT_FIELDS)}, actual={len(result)}")
    payload = dict(zip(_RESULT_FIELDS, result))
    payload["return_over_max_drawdown"] = calc_plain_romd(
        payload["total_return_pct"], payload["max_drawdown_pct"]
    )
    return payload

def _capacity_summary(profile: dict[str, Any]) -> dict[str, Any]:
    frame = pd.DataFrame(profile.get("portfolio_capacity_rows") or [])
    if frame.empty:
        return {
            "sim_day_count": 0,
            "avg_orderable_candidates": 0.0,
            "zero_orderable_candidate_days": 0,
            "candidate_supply_gap_days": 0,
            "candidate_supply_gap_slot_days": 0,
            "underfilled_end_days": 0,
            "end_position_gap_slot_days": 0,
            "avg_end_positions": 0.0,
            "full_position_days": 0,
        }
    return {
        "sim_day_count": int(len(frame)),
        "avg_orderable_candidates": float(frame["Orderable_Candidates"].mean()),
        "zero_orderable_candidate_days": int((frame["Orderable_Candidates"] == 0).sum()),
        "candidate_supply_gap_days": int((frame["Candidate_Supply_Gap"] > 0).sum()),
        "candidate_supply_gap_slot_days": int(frame["Candidate_Supply_Gap"].sum()),
        "underfilled_end_days": int((frame["End_Position_Gap"] > 0).sum()),
        "end_position_gap_slot_days": int(frame["End_Position_Gap"].sum()),
        "avg_end_positions": float(frame["Post_Execution_Positions"].mean()),
        "full_position_days": int((frame["End_Position_Gap"] == 0).sum()),
    }

def _to_json_native(value: Any) -> Any:
    """Convert pandas/numpy scalars and timestamps to stable JSON-native values."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        return _to_json_native(value.item())
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _to_json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_native(item) for item in value]
    return str(value)

def _scenario_summary(payload: dict[str, Any]) -> dict[str, Any]:
    profile = dict(payload["profile"] or {})
    summary = {
        key: payload[key]
        for key in (
            "total_return_pct", "max_drawdown_pct", "return_over_max_drawdown",
            "annual_return_pct", "log_r_squared", "monthly_win_rate_pct", "trade_count",
            "win_rate_pct", "payoff_ratio", "expected_value_r", "final_equity",
            "avg_exposure_pct", "max_exposure_pct", "missed_buy_count", "missed_sell_count",
            "reserved_buy_fill_rate_pct", "normal_trade_count", "extended_trade_count",
            "annual_trade_count", "benchmark_return_pct", "benchmark_max_drawdown_pct",
            "benchmark_annual_return_pct",
        )
    }
    summary.update({
        "portfolio_total_r": float(profile.get("portfolio_total_r", 0.0)),
        "portfolio_median_r": float(profile.get("portfolio_median_r", 0.0)),
        "portfolio_avg_r": float(profile.get("portfolio_avg_r", 0.0)),
        "min_full_year_return_pct": float(profile.get("min_full_year_return_pct", 0.0)),
        "min_month_return_pct": float(profile.get("min_month_return_pct", 0.0)),
        "min_quarter_return_pct": float(profile.get("min_quarter_return_pct", 0.0)),
        "full_year_count": int(profile.get("full_year_count", 0)),
    })
    summary.update(_capacity_summary(profile))
    return _to_json_native(summary)

def _delta(quality: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in quality.items():
        base = baseline.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and isinstance(base, (int, float)) and not isinstance(base, bool):
            out[key] = float(value) - float(base)
    return out

def _assert_shared_benchmark(no_filter: dict[str, Any], quality: dict[str, Any]) -> None:
    for key in ("benchmark_return_pct", "benchmark_max_drawdown_pct", "benchmark_annual_return_pct"):
        if not math.isclose(float(no_filter[key]), float(quality[key]), rel_tol=0.0, abs_tol=1e-10):
            raise ValueError(f"兩組 benchmark 不一致: {key}, no_filter={no_filter[key]}, quality={quality[key]}")
    left_dates = list(pd.to_datetime(no_filter["equity_curve"]["Date"]).dt.strftime("%Y-%m-%d"))
    right_dates = list(pd.to_datetime(quality["equity_curve"]["Date"]).dt.strftime("%Y-%m-%d"))
    if left_dates != right_dates:
        raise ValueError("兩組回測交易日期不一致")

def _normalize_yearly_completeness(frame: pd.DataFrame) -> pd.DataFrame:
    """A clipped final year is not complete merely because it reaches the last available replay date."""

    out = pd.DataFrame(frame).copy()
    if out.empty:
        return out
    required = {"is_full_year", "start_date", "end_date"}
    if not required.issubset(out.columns):
        return out
    start_dates = pd.to_datetime(out["start_date"], errors="raise")
    end_dates = pd.to_datetime(out["end_date"], errors="raise")
    calendar_covered = (start_dates.dt.month == 1) & (end_dates.dt.month == 12)
    out["is_full_year"] = out["is_full_year"].astype(bool) & calendar_covered
    return out

def _yearly_frame(profile: dict[str, Any], scenario: str) -> pd.DataFrame:
    frame = _normalize_yearly_completeness(pd.DataFrame(profile.get("yearly_return_rows") or []))
    if frame.empty:
        return pd.DataFrame(columns=["year", f"{scenario}_return_pct", "is_full_year", "start_date", "end_date"])
    return frame.rename(columns={"year_return_pct": f"{scenario}_return_pct"})

def _build_yearly_comparison(no_filter_profile: dict[str, Any], quality_profile: dict[str, Any]) -> pd.DataFrame:
    left = _yearly_frame(no_filter_profile, "no_filter")
    right = _yearly_frame(quality_profile, "quality_filter")
    keys = [key for key in ("year", "is_full_year", "start_date", "end_date") if key in left.columns and key in right.columns]
    merged = left.merge(right, on=keys, how="outer", validate="one_to_one")
    merged["delta_pct"] = merged["quality_filter_return_pct"] - merged["no_filter_return_pct"]
    return merged.sort_values("year").reset_index(drop=True)

def _refresh_yearly_summary(summary: dict[str, Any], yearly: pd.DataFrame, *, return_column: str) -> None:
    full = yearly[yearly["is_full_year"].astype(bool)] if not yearly.empty else yearly
    summary["full_year_count"] = int(len(full))
    summary["min_full_year_return_pct"] = float(full[return_column].min()) if not full.empty else 0.0

def _load_existing_comparison_payload(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "strategy_comparison.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"讀取既有策略比較 JSON 失敗: {path}｜{type(exc).__name__}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise ValueError(f"既有策略比較 JSON schema 無效: {path}")
    return payload

def _format_metric(value: Any, *, digits: int, unit: str = "", signed: bool = False) -> str:
    if value is None or isinstance(value, bool):
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "N/A"
    sign = "+" if signed else ""
    return f"{numeric:{sign}.{digits}f}{unit}"

def _markdown_report(metadata, baseline, quality, delta, yearly, strategy_diagnostics=None) -> str:
    labels = _comparison_labels(str(metadata["comparison_mode"]))
    active_yearly_column = f"{labels['active_name']}_return_pct"
    rows = [
        ("淨總報酬", "total_return_pct", "%", "higher"),
        ("最大回撤", "max_drawdown_pct", "%", "lower"),
        ("報酬／最大回撤", "return_over_max_drawdown", "", "higher"),
        ("年化報酬", "annual_return_pct", "%", "higher"),
        ("Log R²", "log_r_squared", "", "higher"),
        ("月勝率", "monthly_win_rate_pct", "%", "higher"),
        ("交易數", "trade_count", "", "neutral"),
        ("勝率", "win_rate_pct", "%", "higher"),
        ("Payoff", "payoff_ratio", "", "higher"),
        ("EV", "expected_value_r", " R", "higher"),
        ("平均曝險", "avg_exposure_pct", "%", "attention"),
        ("最差完整年度", "min_full_year_return_pct", "%", "higher"),
        ("平均每日可掛單候選", "avg_orderable_candidates", "", "neutral"),
        ("候選供給不足日", "candidate_supply_gap_days", " 日", "lower"),
        ("期末未滿倉日", "underfilled_end_days", " 日", "lower"),
        ("期末持股缺口總和", "end_position_gap_slot_days", " 格日", "lower"),
    ]
    lines = [
        ("# Breakout Quality Score 排序策略經濟效果對照" if metadata["comparison_mode"] == COMPARISON_MODE_SCORE_RANKING else "# Breakout Quality 策略經濟效果對照"), "",
        f"- 期間：`{metadata['comparison_period']['start']}` ～ `{metadata['comparison_period']['end']}`",
        f"- 參數檔：`{project_relative_display_path(metadata['params_path'], project_root=PROJECT_ROOT)}`",
        f"- 參數型態：`{metadata['param_source_kind']}`",
        f"- 參數 selector：`{metadata.get('param_selector')}`",
        f"- Runtime members：`{metadata.get('runtime_member_count_min')}`～`{metadata.get('runtime_member_count_max')}`；min_agree=`{metadata.get('runtime_min_agree')}`",
        f"- 比較設計：`{metadata['comparison_design']}`",
        f"- 歷史 active-param 無前視：`{metadata['lookahead_safe_active_param_schedule']}`",
        f"- Dataset：`{metadata['dataset']}`",
        f"- Score source：`{metadata.get('score_source')}`",
        *(
            [f"- Ranking policy：`{metadata.get('score_ranking_policy')}`"]
            if metadata["comparison_mode"] == COMPARISON_MODE_SCORE_RANKING
            else []
        ),
        f"- Optional entry filters：`{metadata.get('optional_entry_filter_policy')}`",
        f"- Benchmark：`{metadata['benchmark_ticker']}`",
        f"- 唯一差異：`{labels['difference_text']}`",
        (
            f"- Ranking model：`{metadata['filter_id']}` / `{metadata['model_architecture']}` / `{metadata['experiment_profile']}`；"
            "threshold 不作 gate"
            if metadata["comparison_mode"] == COMPARISON_MODE_SCORE_RANKING
            else f"- Filter：`{metadata['filter_id']}` / `{metadata['model_architecture']}` / `{metadata['experiment_profile']}` / threshold `{metadata['threshold']}`"
        ),
        *(
            [f"- 排序鍵：`{' → '.join(metadata.get('score_ranking_order') or [])}`", f"- Ranking 範圍：`{metadata.get('ranking_scope')}`"]
            if metadata["comparison_mode"] == COMPARISON_MODE_SCORE_RANKING
            else []
        ),
        "", "## 主要結果", "", f"| 指標 | No filter | {labels['active_title']} | 差異 | 判讀 |", "|---|---:|---:|---:|:---:|",
    ]
    for label, key, unit, preference in rows:
        digits = 0 if key in {"trade_count", "candidate_supply_gap_days", "underfilled_end_days", "end_position_gap_slot_days"} else 4 if key == "log_r_squared" else 2
        metric_signal = signal_for_delta(
            delta.get(key),
            preference=preference,
            warning_threshold=5.0 if key == "avg_exposure_pct" else 0.0,
        )
        lines.append(
            f"| {label} | {_format_metric(baseline.get(key), digits=digits, unit=unit)} "
            f"| {_format_metric(quality.get(key), digits=digits, unit=unit)} "
            f"| {_format_metric(delta.get(key), digits=digits, unit=unit, signed=True)} "
            f"| {signal_marker(metric_signal)} |"
        )
    lines += ["", "## 年度報酬", ""]
    if yearly.empty:
        lines.append("無年度資料。")
    else:
        lines += [f"| 年度 | No filter | {labels['active_title']} | 差異 | 判讀 | 完整年度 |", "|---:|---:|---:|---:|:---:|:---:|"]
        for row in yearly.to_dict("records"):
            year_signal = signal_for_delta(row.get("delta_pct"), preference="higher")
            lines.append(
                f"| {int(row['year'])} "
                f"| {_format_metric(row.get('no_filter_return_pct'), digits=2, unit='%')} "
                f"| {_format_metric(row.get(active_yearly_column), digits=2, unit='%')} "
                f"| {_format_metric(row.get('delta_pct'), digits=2, unit='%', signed=True)} "
                f"| {signal_marker(year_signal)} "
                f"| {'是' if row.get('is_full_year') else '否'} |"
            )
    if strategy_diagnostics:
        left = strategy_diagnostics.get("no_filter") or {}
        right = strategy_diagnostics.get("score_ranking") or {}
        lines += [
            "", "## Selection 選股診斷（Future Target僅於回放後join）", "",
            "| 指標 | Baseline | Score Sort | 差異 | 判讀 |",
            "|---|---:|---:|---:|:---:|",
        ]
        for label, key, digits, preference in (
            ("Orderable Score coverage", "orderable_score_coverage_rate", 4, "higher"),
            ("選中候選 Target percentile", "selected_target_percentile_mean", 4, "higher"),
            ("Target top-k retention", "target_top_k_retention_mean", 4, "higher"),
            ("Target opportunity gap (R)", "target_opportunity_gap_r_mean", 4, "lower"),
            ("選中候選 Target mean (R)", "selected_target_mean_r", 4, "higher"),
        ):
            lv, rv = left.get(key), right.get(key)
            dv = None if lv is None or rv is None else float(rv) - float(lv)
            diag_signal = signal_for_delta(dv, preference=preference)
            lines.append(
                f"| {label} | {_format_metric(lv, digits=digits)} | "
                f"{_format_metric(rv, digits=digits)} | {_format_metric(dv, digits=digits, signed=True)} "
                f"| {signal_marker(diag_signal)} |"
            )
        lines += [
            "",
            "> Future Target未進入候選排序、資金配置或成交決策；上述診斷只在兩組策略回放完成後離線計算。",
        ]
    if not metadata["lookahead_safe_active_param_schedule"]:
        lines += ["", "> 警告：本次使用單一／static 參數，只能視為敏感度診斷，不是無前視 OOS 部署證據。"]
    if metadata["comparison_mode"] == COMPARISON_MODE_SCORE_RANKING:
        limitation = (
            "> 本報表使用Selection point-in-time Scores與當期歷史active params比較排序機制；"
            "可用於Selection內決定是否進入參數適應，但正式效果仍須由凍結後OOS驗證。"
            if metadata.get("score_source") == SCORE_SOURCE_SELECTION_POINT_IN_TIME
            else "> 本報表是既有forward period的score-ranking機制比較；不得依結果回頭調整模型或排序規則。"
        )
    else:
        limitation = "> 本報表只驗證目前固定active操作點能否改善策略經濟效果；不得依結果回頭調整threshold、模型或訓練條件。"
    lines += ["", limitation, ""]
    return "\n".join(lines)


def _numeric_delta(delta: dict[str, Any], key: str) -> float | None:
    value = delta.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None

def _compact_metric_rows(
    baseline: dict[str, Any],
    quality: dict[str, Any],
    delta: dict[str, Any],
    specs: tuple[tuple[str, str, str, str], ...],
    *,
    use_color: bool,
) -> list[tuple[str, str, str, str, str]]:
    rows: list[tuple[str, str, str, str, str]] = []
    integer_keys = {
        "trade_count", "candidate_supply_gap_days", "underfilled_end_days",
        "end_position_gap_slot_days",
    }
    for label, key, unit, preference in specs:
        digits = 0 if key in integer_keys else 4 if key == "log_r_squared" else 2
        signal = signal_for_delta(
            delta.get(key),
            preference=preference,
            warning_threshold=5.0 if key == "avg_exposure_pct" else 0.0,
        )
        rows.append((
            label,
            _format_metric(baseline.get(key), digits=digits, unit=unit),
            _format_metric(quality.get(key), digits=digits, unit=unit),
            terminal_signal(
                _format_metric(delta.get(key), digits=digits, unit=unit, signed=True),
                signal,
                enabled=use_color,
            ),
            terminal_signal(signal_marker(signal), signal, enabled=use_color),
        ))
    return rows


def _render_compact_strategy_console_report(
    metadata: dict[str, Any],
    baseline: dict[str, Any],
    quality: dict[str, Any],
    delta: dict[str, Any],
    yearly: pd.DataFrame,
    strategy_diagnostics: dict[str, Any] | None,
    *,
    use_color: bool,
) -> str:
    """Render the interactive strategy report by decision layer rather than source table."""

    labels = _comparison_labels(str(metadata["comparison_mode"]))
    active_yearly_column = f"{labels['active_name']}_return_pct"
    active_label = (
        "Score Sort"
        if metadata["comparison_mode"] == COMPARISON_MODE_SCORE_RANKING
        else labels["active_title"]
    )
    period = metadata.get("comparison_period") or {}
    title = (
        "Breakout Quality Score 排序策略績效摘要"
        if metadata["comparison_mode"] == COMPARISON_MODE_SCORE_RANKING
        else "Breakout Quality 策略績效摘要"
    )

    total_delta = _numeric_delta(delta, "total_return_pct")
    romd_delta = _numeric_delta(delta, "return_over_max_drawdown")
    mdd_delta = _numeric_delta(delta, "max_drawdown_pct")
    if total_delta is not None and romd_delta is not None and total_delta < 0 and romd_delta < 0:
        portfolio_signal = SIGNAL_NEGATIVE
        portfolio_text = "投組層失敗：總報酬與報酬／回撤同步下降。"
    elif total_delta is not None and romd_delta is not None and total_delta > 0 and romd_delta > 0:
        portfolio_signal = SIGNAL_POSITIVE
        portfolio_text = "投組層改善：總報酬與報酬／回撤同步提高。"
    else:
        portfolio_signal = SIGNAL_WARNING
        portfolio_text = "投組層結果混合：報酬與風險指標未同向改善。"

    ev_delta = _numeric_delta(delta, "expected_value_r")
    payoff_delta = _numeric_delta(delta, "payoff_ratio")
    trade_signal = (
        SIGNAL_POSITIVE
        if ev_delta is not None and payoff_delta is not None and ev_delta > 0 and payoff_delta > 0
        else SIGNAL_NEGATIVE
        if ev_delta is not None and payoff_delta is not None and ev_delta < 0 and payoff_delta < 0
        else SIGNAL_WARNING
    )
    trade_text = (
        "交易層：EV與Payoff同步改善，但仍須結合勝率與投入規模判讀。"
        if trade_signal == SIGNAL_POSITIVE
        else "交易層：單筆交易品質未全面改善。"
    )

    exposure_delta = _numeric_delta(delta, "avg_exposure_pct")
    underfilled_delta = _numeric_delta(delta, "underfilled_end_days")
    capital_signal = SIGNAL_WARNING
    capital_text = (
        "資金層：平均曝險下降，但未滿倉日也下降；代表持倉格較滿、每格投入較小。"
        if exposure_delta is not None and exposure_delta < 0
        and underfilled_delta is not None and underfilled_delta < 0
        else "資金層：需分開檢查曝險、候選供給與持倉格使用。"
    )

    diagnostic_specs = (
        ("Orderable Score coverage", "orderable_score_coverage_rate", "higher"),
        ("選中候選 Target percentile", "selected_target_percentile_mean", "higher"),
        ("Target top-k retention", "target_top_k_retention_mean", "higher"),
        ("Target opportunity gap", "target_opportunity_gap_r_mean", "lower"),
        ("選中候選 Target mean", "selected_target_mean_r", "higher"),
    )
    model_signal = SIGNAL_NEUTRAL
    model_text = "模型層：本次沒有可用的事後Target選股診斷。"
    diagnostic_rows: list[tuple[str, str, str, str, str]] = []
    if strategy_diagnostics:
        left = strategy_diagnostics.get("no_filter") or {}
        right = strategy_diagnostics.get("score_ranking") or {}
        positive_count = 0
        available_count = 0
        for label, key, preference in diagnostic_specs:
            left_value, right_value = left.get(key), right.get(key)
            try:
                left_number = float(left_value)
                right_number = float(right_value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(left_number) or not math.isfinite(right_number):
                continue
            delta_value = right_number - left_number
            signal = signal_for_delta(delta_value, preference=preference)
            available_count += 1
            positive_count += int(signal == SIGNAL_POSITIVE)
            diagnostic_rows.append((
                label,
                _format_metric(left_number, digits=4),
                _format_metric(right_number, digits=4),
                terminal_signal(
                    _format_metric(delta_value, digits=4, signed=True),
                    signal,
                    enabled=use_color,
                ),
                terminal_signal(signal_marker(signal), signal, enabled=use_color),
            ))
        if available_count:
            model_signal = (
                SIGNAL_POSITIVE if positive_count >= max(1, available_count - 1)
                else SIGNAL_WARNING
            )
            model_text = f"模型層：{positive_count}/{available_count}項Selection選股診斷改善。"

    lines = [
        render_title(title),
        render_key_values((
            ("期間", f"{period.get('start', '')} ～ {period.get('end', '')}"),
            ("比較", f"Baseline vs {active_label}"),
            ("Score source", metadata.get("score_source", "-")),
            ("歷史參數無前視", metadata.get("lookahead_safe_active_param_schedule", "-")),
        )),
        render_section("綜合判定", number=1),
        terminal_signal(f"{signal_marker(portfolio_signal)} {portfolio_text}", portfolio_signal, enabled=use_color),
        terminal_signal(f"{signal_marker(trade_signal)} {trade_text}", trade_signal, enabled=use_color),
        terminal_signal(f"{signal_marker(capital_signal)} {capital_text}", capital_signal, enabled=use_color),
        terminal_signal(f"{signal_marker(model_signal)} {model_text}", model_signal, enabled=use_color),
    ]
    if total_delta is not None or mdd_delta is not None:
        lines.append(
            "關鍵差異：總報酬 "
            f"{_format_metric(total_delta, digits=2, unit='pp', signed=True)}；"
            "最大回撤 "
            f"{_format_metric(mdd_delta, digits=2, unit='pp', signed=True)}；"
            "平均曝險 "
            f"{_format_metric(exposure_delta, digits=2, unit='pp', signed=True)}。"
        )

    portfolio_specs = (
        ("淨總報酬", "total_return_pct", "%", "higher"),
        ("最大回撤", "max_drawdown_pct", "%", "lower"),
        ("報酬／最大回撤", "return_over_max_drawdown", "", "higher"),
        ("年化報酬", "annual_return_pct", "%", "higher"),
        ("Log R²", "log_r_squared", "", "higher"),
        ("月勝率", "monthly_win_rate_pct", "%", "higher"),
        ("最差完整年度", "min_full_year_return_pct", "%", "higher"),
    )
    lines.extend((
        render_section("投組報酬與風險", number=2),
        "讀法：看最終賺多少、承受多少回撤，以及權益成長是否穩定。",
        render_table(
            ("指標", "Baseline", active_label, "差異", "判讀"),
            _compact_metric_rows(
                baseline, quality, delta, portfolio_specs, use_color=use_color
            ),
            alignments=("left", "right", "right", "right", "left"),
        ),
    ))

    trade_specs = (
        ("交易數", "trade_count", "", "neutral"),
        ("勝率", "win_rate_pct", "%", "higher"),
        ("Payoff", "payoff_ratio", "", "higher"),
        ("EV", "expected_value_r", " R", "higher"),
    )
    lines.extend((
        render_section("單筆交易品質", number=3),
        "讀法：看每筆交易的命中率與盈虧結構；不代表整體資金使用效率。",
        render_table(
            ("指標", "Baseline", active_label, "差異", "判讀"),
            _compact_metric_rows(
                baseline, quality, delta, trade_specs, use_color=use_color
            ),
            alignments=("left", "right", "right", "right", "left"),
        ),
    ))

    capital_specs = (
        ("平均曝險", "avg_exposure_pct", "%", "attention"),
        ("平均每日可掛單候選", "avg_orderable_candidates", "", "neutral"),
        ("候選供給不足日", "candidate_supply_gap_days", " 日", "lower"),
        ("每日結束未滿倉日", "underfilled_end_days", " 日", "lower"),
        ("每日結束持股缺口", "end_position_gap_slot_days", " 格日", "lower"),
    )
    lines.extend((
        render_section("資金使用與持倉容量", number=4),
        "讀法：持有幾檔與投入多少資金是不同概念；持倉格較滿不等於曝險較高。",
        render_table(
            ("指標", "Baseline", active_label, "差異", "判讀"),
            _compact_metric_rows(
                baseline, quality, delta, capital_specs, use_color=use_color
            ),
            alignments=("left", "right", "right", "right", "left"),
        ),
    ))

    lines.append(render_section("年度報酬", number=5))
    if yearly.empty:
        lines.append("無年度資料。")
    else:
        yearly_rows = []
        for row in yearly.to_dict("records"):
            signal = signal_for_delta(row.get("delta_pct"), preference="higher")
            yearly_rows.append((
                int(row["year"]),
                _format_metric(row.get("no_filter_return_pct"), digits=2, unit="%"),
                _format_metric(row.get(active_yearly_column), digits=2, unit="%"),
                terminal_signal(
                    _format_metric(row.get("delta_pct"), digits=2, unit="pp", signed=True),
                    signal,
                    enabled=use_color,
                ),
                terminal_signal(signal_marker(signal), signal, enabled=use_color),
            ))
        lines.append(render_table(
            ("年度", "Baseline", active_label, "差異", "判讀"),
            yearly_rows,
            alignments=("right", "right", "right", "right", "left"),
        ))

    next_number = 6
    if diagnostic_rows:
        lines.extend((
            render_section("模型選股方向", number=next_number),
            "讀法：Future Target只在回放完成後加入；這裡衡量選股方向，不是實際策略報酬。",
            render_table(
                ("指標", "Baseline", "Score Sort", "差異", "判讀"),
                diagnostic_rows,
                alignments=("left", "right", "right", "right", "left"),
            ),
        ))
        next_number += 1

    lines.append(render_section("判讀限制", number=next_number))
    if not metadata.get("lookahead_safe_active_param_schedule"):
        lines.append("⚠️ 本次使用單一／static參數，只能視為敏感度診斷。")
    if metadata["comparison_mode"] == COMPARISON_MODE_SCORE_RANKING:
        lines.append(
            "本結果是Selection內的PIT比較；模型排序、投組績效與資金配置必須分層判讀，"
            "正式採用仍須由凍結後OOS驗證。"
        )
    else:
        lines.append("本結果只驗證固定active操作點，不得依結果回頭調整模型或threshold。")
    return "\n".join(lines)



def _compact_capture_metric_rows(
    baseline: dict[str, Any],
    score_sort: dict[str, Any],
    delta: dict[str, Any],
    specs: tuple[tuple[str, str, str, str, int], ...],
    *,
    use_color: bool,
) -> list[tuple[str, str, str, str, str]]:
    """Render capture-attribution metrics in the same table contract as strategy metrics."""

    rows: list[tuple[str, str, str, str, str]] = []
    for label, key, unit, preference, digits in specs:
        signal = signal_for_delta(
            delta.get(key), preference=preference, warning_threshold=0.0
        )
        rows.append((
            label,
            _format_metric(baseline.get(key), digits=digits, unit=unit),
            _format_metric(score_sort.get(key), digits=digits, unit=unit),
            terminal_signal(
                _format_metric(
                    delta.get(key), digits=digits, unit=unit, signed=True
                ),
                signal,
                enabled=use_color,
            ),
            terminal_signal(signal_marker(signal), signal, enabled=use_color),
        ))
    return rows


def _render_flat_score_ranking_console_report(
    metadata: dict[str, Any],
    baseline: dict[str, Any],
    quality: dict[str, Any],
    delta: dict[str, Any],
    yearly: pd.DataFrame,
    strategy_diagnostics: dict[str, Any] | None,
    capture_result: dict[str, Any],
    *,
    use_color: bool,
) -> str:
    """Render one flat ten-section interactive report with all current evidence."""

    period = metadata.get("comparison_period") or {}
    capture_baseline = dict(capture_result.get("baseline") or {})
    capture_score = dict(capture_result.get("score_sort") or {})
    capture_delta = dict(capture_result.get("score_sort_minus_baseline") or {})
    capture_decision = dict(capture_result.get("decision") or {})
    capture_yearly_value = capture_result.get("yearly")
    capture_yearly = (
        capture_yearly_value.copy()
        if isinstance(capture_yearly_value, pd.DataFrame)
        else pd.DataFrame(capture_yearly_value or [])
    )

    lines = [
        render_title("Breakout Quality Score 排序策略驗證摘要"),
        render_key_values((
            ("期間", f"{period.get('start', '')} ～ {period.get('end', '')}"),
            ("比較", "Baseline vs Score Sort"),
            ("參數型態", metadata.get("param_source_kind", "-")),
            ("參數 selector", metadata.get("param_selector", "-")),
            (
                "Runtime members",
                f"{metadata.get('runtime_member_count_min')}～"
                f"{metadata.get('runtime_member_count_max')}；"
                f"min_agree={metadata.get('runtime_min_agree')}",
            ),
            ("比較設計", metadata.get("comparison_design", "-")),
            ("歷史 active-param 無前視", metadata.get("lookahead_safe_active_param_schedule", "-")),
            ("Dataset", metadata.get("dataset", "-")),
            ("Score source", metadata.get("score_source", "-")),
            ("Ranking policy", metadata.get("score_ranking_policy", "-")),
            ("Optional entry filters", metadata.get("optional_entry_filter_policy", "-")),
            ("Benchmark", metadata.get("benchmark_ticker", "-")),
            ("唯一差異", _comparison_labels(COMPARISON_MODE_SCORE_RANKING)["difference_text"]),
            ("排序鍵", " → ".join(metadata.get("score_ranking_order") or [])),
            ("Ranking 範圍", metadata.get("ranking_scope", "-")),
            ("歸因性質", "Selection read-only attribution"),
            ("Future Target runtime", "未使用"),
        )),
    ]

    portfolio_specs = (
        ("淨總報酬", "total_return_pct", "%", "higher"),
        ("最大回撤", "max_drawdown_pct", "%", "lower"),
        ("報酬／最大回撤", "return_over_max_drawdown", "", "higher"),
        ("年化報酬", "annual_return_pct", "%", "higher"),
        ("Log R²", "log_r_squared", "", "higher"),
        ("月勝率", "monthly_win_rate_pct", "%", "higher"),
        ("最差完整年度", "min_full_year_return_pct", "%", "higher"),
    )
    lines.extend((
        render_section("投組報酬與風險", number=1),
        "定義：衡量整體權益最後賺多少、曾承受多少回撤，以及成長路徑是否穩定。",
        render_table(
            ("指標", "Baseline", "Score Sort", "差異", "判讀"),
            _compact_metric_rows(
                baseline, quality, delta, portfolio_specs, use_color=use_color
            ),
            alignments=("left", "right", "right", "right", "left"),
        ),
    ))

    trade_rows = _compact_metric_rows(
        baseline,
        quality,
        delta,
        (
            ("交易數", "trade_count", "", "neutral"),
            ("勝率", "win_rate_pct", "%", "higher"),
            ("Payoff", "payoff_ratio", "", "higher"),
            ("EV", "expected_value_r", " R", "higher"),
        ),
        use_color=use_color,
    )
    trade_rows.extend(_compact_capture_metric_rows(
        capture_baseline,
        capture_score,
        capture_delta,
        (
            ("平均 Realized R", "avg_realized_r", " R", "higher", 2),
            ("平均投入資金報酬", "avg_capital_return_pct", "%", "higher", 2),
        ),
        use_color=use_color,
    ))
    lines.extend((
        render_section("單筆交易結果", number=2),
        "定義：勝率看獲利筆數；Payoff看平均贏家／輸家；EV與Realized R看每筆初始風險的實際期望值。",
        render_table(
            ("指標", "Baseline", "Score Sort", "差異", "判讀"),
            trade_rows,
            alignments=("left", "right", "right", "right", "left"),
        ),
        "註：目前EV採平均Realized R口徑，兩者保留是為了對應策略主表與capture歸因表。",
    ))

    sizing_specs = (
        ("平均曝險", "avg_exposure_pct", "%", "higher", 2),
        ("平均預留金額", "avg_reserved_total", "", "attention", 2),
        ("平均實際投入金額", "avg_invested_total", "", "higher", 2),
        ("投入／預留比", "avg_invested_vs_reserved_pct", "%", "higher", 2),
        ("平均初始停損距離", "avg_stop_distance_pct", "%", "attention", 2),
    )
    lines.extend((
        render_section("資金投入與部位大小", number=3),
        "定義：曝險是每日投入市場資金占權益比例；停損越寬，固定風險sizing下每筆部位通常越小。",
        render_table(
            ("指標", "Baseline", "Score Sort", "差異", "判讀"),
            _compact_capture_metric_rows(
                capture_baseline,
                capture_score,
                capture_delta,
                sizing_specs,
                use_color=use_color,
            ),
            alignments=("left", "right", "right", "right", "left"),
        ),
    ))

    capacity_rows = _compact_metric_rows(
        baseline,
        quality,
        delta,
        (
            ("平均每日可掛單候選", "avg_orderable_candidates", "", "neutral"),
            ("候選供給不足日", "candidate_supply_gap_days", " 日", "lower"),
            ("每日結束未滿倉日", "underfilled_end_days", " 日", "lower"),
            ("每日結束持股缺口", "end_position_gap_slot_days", " 格日", "lower"),
        ),
        use_color=use_color,
    )
    capacity_rows.extend(_compact_capture_metric_rows(
        capture_baseline,
        capture_score,
        capture_delta,
        (("保留買單成交率", "reserved_buy_fill_rate_pct", "%", "higher", 2),),
        use_color=use_color,
    ))
    lines.extend((
        render_section("候選供給與持倉容量", number=4),
        "定義：候選供給看是否有股票可買；未滿倉與缺口看持倉格是否填滿；成交率看預留買單是否落地。",
        render_table(
            ("指標", "Baseline", "Score Sort", "差異", "判讀"),
            capacity_rows,
            alignments=("left", "right", "right", "right", "left"),
        ),
    ))

    diagnostic_specs = (
        ("Orderable Score coverage", "orderable_score_coverage_rate", "higher"),
        ("選中候選 Target percentile", "selected_target_percentile_mean", "higher"),
        ("Target top-k retention", "target_top_k_retention_mean", "higher"),
        ("Target opportunity gap (R)", "target_opportunity_gap_r_mean", "lower"),
        ("選中候選 Target mean (R)", "selected_target_mean_r", "higher"),
    )
    diagnostic_rows: list[tuple[str, str, str, str, str]] = []
    positive_diagnostics = 0
    if strategy_diagnostics:
        left = dict(strategy_diagnostics.get("no_filter") or {})
        right = dict(strategy_diagnostics.get("score_ranking") or {})
        for label, key, preference in diagnostic_specs:
            lv = left.get(key)
            rv = right.get(key)
            try:
                dv = float(rv) - float(lv)
            except (TypeError, ValueError):
                dv = None
            signal = signal_for_delta(dv, preference=preference)
            positive_diagnostics += int(signal == SIGNAL_POSITIVE)
            diagnostic_rows.append((
                label,
                _format_metric(lv, digits=4),
                _format_metric(rv, digits=4),
                terminal_signal(
                    _format_metric(dv, digits=4, signed=True),
                    signal,
                    enabled=use_color,
                ),
                terminal_signal(signal_marker(signal), signal, enabled=use_color),
            ))
    lines.extend((
        render_section("模型選股能力", number=5),
        "定義：比較實際選中候選與事後Future Target理想排序；Future Target只在回放完成後join。",
        render_table(
            ("指標", "Baseline", "Score Sort", "差異", "判讀"),
            diagnostic_rows,
            alignments=("left", "right", "right", "right", "left"),
        ) if diagnostic_rows else "本次沒有可用的Selection選股診斷。",
    ))

    target_specs = (
        ("平均 Target R", "avg_target_r", " R", "higher", 2),
        ("Aggregate Target capture", "aggregate_target_capture_ratio", "", "higher", 2),
        ("Median Target capture", "median_target_capture_ratio", "", "higher", 2),
        ("Target ≥ 0.5R capture", "target_ge_0_5_capture_ratio", "", "higher", 2),
        ("平均 Target realization gap", "avg_target_realization_gap_r", " R", "lower", 2),
    )
    lines.extend((
        render_section("Target 到實際報酬的轉換", number=6),
        "定義：Target R是事後價格機會；capture衡量Realized R相對Target R的轉換；gap越低越好。",
        render_table(
            ("指標", "Baseline", "Score Sort", "差異", "判讀"),
            _compact_capture_metric_rows(
                capture_baseline,
                capture_score,
                capture_delta,
                target_specs,
                use_color=use_color,
            ),
            alignments=("left", "right", "right", "right", "left"),
        ),
    ))

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
    lines.extend((
        render_section("資金周轉與進場集中", number=7),
        "定義：持有與半倉指標衡量資金占用時間；日期、月份與產業指標衡量交易是否集中。",
        render_table(
            ("指標", "Baseline", "Score Sort", "差異", "判讀"),
            _compact_capture_metric_rows(
                capture_baseline,
                capture_score,
                capture_delta,
                turnover_specs,
                use_color=use_color,
            ),
            alignments=("left", "right", "right", "right", "left"),
        ),
    ))

    exit_specs = (
        ("停損", "stop_exit_share_pct", "%", "attention", 2),
        ("指標", "indicator_exit_share_pct", "%", "attention", 2),
        ("汰弱", "rotation_exit_share_pct", "%", "attention", 2),
        ("期末強制", "forced_exit_share_pct", "%", "attention", 2),
    )
    lines.extend((
        render_section("出場結構", number=8),
        "定義：依每筆交易最後的全倉結算原因分類；占比改變只表示結構差異，不直接等於損益好壞。",
        render_table(
            ("類別", "Baseline", "Score Sort", "差異", "判讀"),
            _compact_capture_metric_rows(
                capture_baseline,
                capture_score,
                capture_delta,
                exit_specs,
                use_color=use_color,
            ),
            alignments=("left", "right", "right", "right", "left"),
        ),
    ))

    lines.extend((
        render_section("年度結果與年度歸因", number=9),
        "定義：年度報酬按權益曲線年度；R、capture與投入金額按交易進場年度分組，兩種口徑不可混為同一概念。",
    ))
    if yearly.empty:
        lines.append("無權益年度報酬資料。")
    else:
        yearly_rows = []
        for row in yearly.to_dict("records"):
            signal = signal_for_delta(row.get("delta_pct"), preference="higher")
            yearly_rows.append((
                int(row["year"]),
                _format_metric(row.get("no_filter_return_pct"), digits=2, unit="%"),
                _format_metric(row.get("score_ranking_return_pct"), digits=2, unit="%"),
                terminal_signal(
                    _format_metric(row.get("delta_pct"), digits=2, unit="pp", signed=True),
                    signal,
                    enabled=use_color,
                ),
                terminal_signal(signal_marker(signal), signal, enabled=use_color),
                "是" if row.get("is_full_year") else "否",
            ))
        lines.extend((
            "權益年度報酬：",
            render_table(
                ("年度", "Baseline", "Score Sort", "差異", "判讀", "完整年度"),
                yearly_rows,
                alignments=("right", "right", "right", "right", "left", "center"),
            ),
        ))
    if capture_yearly.empty:
        lines.append("無進場年度歸因資料。")
    else:
        quality_rows = []
        invested_rows = []
        for row in capture_yearly.to_dict("records"):
            r_signal = signal_for_delta(row.get("delta_avg_r"), preference="higher")
            capture_signal = signal_for_delta(
                row.get("delta_aggregate_capture_ratio"), preference="higher"
            )
            invested_signal = signal_for_delta(
                row.get("delta_avg_invested_total"), preference="higher"
            )
            quality_rows.append((
                int(row["entry_year"]),
                _format_metric(row.get("baseline_avg_r"), digits=2),
                _format_metric(row.get("score_sort_avg_r"), digits=2),
                terminal_signal(
                    _format_metric(row.get("delta_avg_r"), digits=2, signed=True),
                    r_signal,
                    enabled=use_color,
                ),
                _format_metric(row.get("baseline_aggregate_capture_ratio"), digits=2),
                _format_metric(row.get("score_sort_aggregate_capture_ratio"), digits=2),
                terminal_signal(
                    _format_metric(
                        row.get("delta_aggregate_capture_ratio"), digits=2, signed=True
                    ),
                    capture_signal,
                    enabled=use_color,
                ),
            ))
            invested_rows.append((
                int(row["entry_year"]),
                _format_metric(row.get("baseline_avg_invested_total"), digits=0),
                _format_metric(row.get("score_sort_avg_invested_total"), digits=0),
                terminal_signal(
                    _format_metric(
                        row.get("delta_avg_invested_total"), digits=0, signed=True
                    ),
                    invested_signal,
                    enabled=use_color,
                ),
            ))
        lines.extend((
            "進場年度交易品質：",
            render_table(
                ("年度", "Base R", "Sort R", "ΔR", "Base capture", "Sort capture", "Δcapture"),
                quality_rows,
                alignments=("right", "right", "right", "right", "right", "right", "right"),
            ),
            "進場年度投入規模：",
            render_table(
                ("年度", "Base投入", "Sort投入", "Δ投入"),
                invested_rows,
                alignments=("right", "right", "right", "right"),
            ),
        ))

    total_delta = _numeric_delta(delta, "total_return_pct")
    romd_delta = _numeric_delta(delta, "return_over_max_drawdown")
    mdd_delta = _numeric_delta(delta, "max_drawdown_pct")
    if total_delta is not None and romd_delta is not None and total_delta < 0 and romd_delta < 0:
        strategy_signal = SIGNAL_NEGATIVE
        strategy_judgement = (
            "直接Score-first Ranking不採用：總報酬與報酬／回撤下降，"
            "且最大回撤未獲改善。"
        )
    elif total_delta is not None and romd_delta is not None and total_delta > 0 and romd_delta > 0:
        strategy_signal = SIGNAL_POSITIVE
        strategy_judgement = "Score-first Ranking在Selection內改善投組報酬與風險調整績效。"
    else:
        strategy_signal = SIGNAL_WARNING
        strategy_judgement = "Score-first Ranking結果混合，尚不足以直接採用。"
    model_judgement = (
        f"{positive_diagnostics}/{len(diagnostic_rows)}項Future Target選股診斷改善，模型排序方向保留。"
        if diagnostic_rows and positive_diagnostics >= max(1, len(diagnostic_rows) - 1)
        else f"只有{positive_diagnostics}/{len(diagnostic_rows)}項Future Target選股診斷改善，模型證據不足。"
        if diagnostic_rows
        else "缺少Future Target選股診斷，無法判定模型排序方向。"
    )
    model_signal = (
        SIGNAL_POSITIVE
        if diagnostic_rows and positive_diagnostics >= max(1, len(diagnostic_rows) - 1)
        else SIGNAL_WARNING
        if diagnostic_rows
        else SIGNAL_NEUTRAL
    )
    bottlenecks = "；".join(capture_decision.get("bottlenecks") or []) or "未辨識出明確瓶頸"
    decision_signal = str(capture_decision.get("signal") or SIGNAL_NEUTRAL)
    lines.extend((
        render_section("綜合判定、限制與下一步", number=10),
        terminal_signal(
            f"{signal_marker(strategy_signal)} 策略判定：{strategy_judgement}",
            strategy_signal,
            enabled=use_color,
        ),
        terminal_signal(
            f"{signal_marker(model_signal)} 模型判定：{model_judgement}",
            model_signal,
            enabled=use_color,
        ),
        terminal_signal(
            f"{signal_marker(decision_signal)} 參數適應判定："
            f"{capture_decision.get('status', '-')}｜{capture_decision.get('conclusion', '-')}",
            decision_signal,
            enabled=use_color,
        ),
        f"主要瓶頸：{bottlenecks}",
        "下一步：固定模型、PIT Score與排序契約，只在Selection內檢驗停損、風險sizing、每筆投入上限與Target轉換；完成後凍結並做OOS。",
        "限制：本結果是Selection PIT證據；Future Target僅在回放完成後join，未參與排序、資金配置、成交或optimizer。",
    ))
    if not metadata.get("lookahead_safe_active_param_schedule"):
        lines.append("警告：本次歷史active-param排程不是無前視，只能視為敏感度診斷。")
    if mdd_delta is not None:
        lines.append(
            "核心差異：總報酬 "
            f"{_format_metric(total_delta, digits=2, unit='pp', signed=True)}；"
            "最大回撤 "
            f"{_format_metric(mdd_delta, digits=2, unit='pp', signed=True)}；"
            "平均曝險 "
            f"{_format_metric(capture_delta.get('avg_exposure_pct'), digits=2, unit='pp', signed=True)}。"
        )
    return "\n".join(lines)

def _render_strategy_console_report(
    metadata: dict[str, Any],
    baseline: dict[str, Any],
    quality: dict[str, Any],
    delta: dict[str, Any],
    yearly: pd.DataFrame,
    strategy_diagnostics: dict[str, Any] | None = None,
    *,
    capture_result: dict[str, Any] | None = None,
    color: bool | None = None,
) -> str:
    """Render the complete readable strategy comparison directly for console."""

    use_color = console_color_enabled() if color is None else bool(color)
    if (
        compact_console_enabled()
        and metadata.get("comparison_mode") == COMPARISON_MODE_SCORE_RANKING
        and capture_result is not None
    ):
        return _render_flat_score_ranking_console_report(
            metadata,
            baseline,
            quality,
            delta,
            yearly,
            strategy_diagnostics,
            capture_result,
            use_color=use_color,
        )
    if compact_console_enabled():
        return _render_compact_strategy_console_report(
            metadata,
            baseline,
            quality,
            delta,
            yearly,
            strategy_diagnostics,
            use_color=use_color,
        )
    labels = _comparison_labels(str(metadata["comparison_mode"]))
    active_yearly_column = f"{labels['active_name']}_return_pct"
    title = (
        "Breakout Quality Score 排序策略經濟效果對照"
        if metadata["comparison_mode"] == COMPARISON_MODE_SCORE_RANKING
        else "Breakout Quality 策略經濟效果對照"
    )
    period = metadata.get("comparison_period") or {}
    params_path = project_relative_display_path(
        metadata.get("params_path", "-"), project_root=PROJECT_ROOT
    )
    lines = [
        render_title(title),
        render_key_values(
            (
                ("期間", f"{period.get('start', '')} ～ {period.get('end', '')}"),
                ("參數檔", params_path),
                ("參數型態", metadata.get("param_source_kind", "-")),
                ("參數 selector", metadata.get("param_selector", "-")),
                (
                    "Runtime members",
                    f"{metadata.get('runtime_member_count_min')}～{metadata.get('runtime_member_count_max')}；"
                    f"min_agree={metadata.get('runtime_min_agree')}",
                ),
                ("比較設計", metadata.get("comparison_design", "-")),
                ("歷史 active-param 無前視", metadata.get("lookahead_safe_active_param_schedule", "-")),
                ("Dataset", metadata.get("dataset", "-")),
                ("Score source", metadata.get("score_source", "-")),
                ("Ranking policy", metadata.get("score_ranking_policy", "-")),
                ("Optional entry filters", metadata.get("optional_entry_filter_policy", "-")),
                ("Benchmark", metadata.get("benchmark_ticker", "-")),
                ("唯一差異", labels["difference_text"]),
                *(
                    (
                        ("排序鍵", " → ".join(metadata.get("score_ranking_order") or [])),
                        ("Ranking 範圍", metadata.get("ranking_scope", "-")),
                    )
                    if metadata["comparison_mode"] == COMPARISON_MODE_SCORE_RANKING
                    else ()
                ),
            )
        ),
    ]

    metric_rows = []
    for label, key, unit, preference in (
        ("淨總報酬", "total_return_pct", "%", "higher"),
        ("最大回撤", "max_drawdown_pct", "%", "lower"),
        ("報酬／最大回撤", "return_over_max_drawdown", "", "higher"),
        ("年化報酬", "annual_return_pct", "%", "higher"),
        ("Log R²", "log_r_squared", "", "higher"),
        ("月勝率", "monthly_win_rate_pct", "%", "higher"),
        ("交易數", "trade_count", "", "neutral"),
        ("勝率", "win_rate_pct", "%", "higher"),
        ("Payoff", "payoff_ratio", "", "higher"),
        ("EV", "expected_value_r", " R", "higher"),
        ("平均曝險", "avg_exposure_pct", "%", "attention"),
        ("最差完整年度", "min_full_year_return_pct", "%", "higher"),
        ("平均每日可掛單候選", "avg_orderable_candidates", "", "neutral"),
        ("候選供給不足日", "candidate_supply_gap_days", " 日", "lower"),
        ("期末未滿倉日", "underfilled_end_days", " 日", "lower"),
        ("期末持股缺口總和", "end_position_gap_slot_days", " 格日", "lower"),
    ):
        digits = 0 if key in {
            "trade_count", "candidate_supply_gap_days", "underfilled_end_days",
            "end_position_gap_slot_days",
        } else 4 if key == "log_r_squared" else 2
        signal = signal_for_delta(
            delta.get(key), preference=preference,
            warning_threshold=5.0 if key == "avg_exposure_pct" else 0.0,
        )
        delta_text = _format_metric(delta.get(key), digits=digits, unit=unit, signed=True)
        judgment = signal_marker(signal)
        metric_rows.append((
            label,
            _format_metric(baseline.get(key), digits=digits, unit=unit),
            _format_metric(quality.get(key), digits=digits, unit=unit),
            terminal_signal(delta_text, signal, enabled=use_color),
            terminal_signal(judgment, signal, enabled=use_color),
        ))
    lines.extend((
        render_section("主要結果", number=1),
        render_table(
            ("指標", "No filter", labels["active_title"], "差異", "判讀"),
            metric_rows,
            alignments=("left", "right", "right", "right", "left"),
        ),
    ))

    lines.append(render_section("年度報酬", number=2))
    if yearly.empty:
        lines.append("無年度資料。")
    else:
        yearly_rows = []
        for row in yearly.to_dict("records"):
            signal = signal_for_delta(row.get("delta_pct"), preference="higher")
            yearly_rows.append((
                int(row["year"]),
                _format_metric(row.get("no_filter_return_pct"), digits=2, unit="%"),
                _format_metric(row.get(active_yearly_column), digits=2, unit="%"),
                terminal_signal(
                    _format_metric(row.get("delta_pct"), digits=2, unit="%", signed=True),
                    signal, enabled=use_color,
                ),
                terminal_signal(signal_marker(signal), signal, enabled=use_color),
                "是" if row.get("is_full_year") else "否",
            ))
        lines.append(render_table(
            ("年度", "No filter", labels["active_title"], "差異", "判讀", "完整年度"),
            yearly_rows,
            alignments=("right", "right", "right", "right", "left", "center"),
        ))

    if strategy_diagnostics:
        left = strategy_diagnostics.get("no_filter") or {}
        right = strategy_diagnostics.get("score_ranking") or {}
        diagnostic_rows = []
        for label, key, preference in (
            ("Orderable Score coverage", "orderable_score_coverage_rate", "higher"),
            ("選中候選 Target percentile", "selected_target_percentile_mean", "higher"),
            ("Target top-k retention", "target_top_k_retention_mean", "higher"),
            ("Target opportunity gap (R)", "target_opportunity_gap_r_mean", "lower"),
            ("選中候選 Target mean (R)", "selected_target_mean_r", "higher"),
        ):
            left_value, right_value = left.get(key), right.get(key)
            delta_value = (
                None if left_value is None or right_value is None
                else float(right_value) - float(left_value)
            )
            signal = signal_for_delta(delta_value, preference=preference)
            diagnostic_rows.append((
                label,
                _format_metric(left_value, digits=4),
                _format_metric(right_value, digits=4),
                terminal_signal(
                    _format_metric(delta_value, digits=4, signed=True),
                    signal, enabled=use_color,
                ),
                terminal_signal(signal_marker(signal), signal, enabled=use_color),
            ))
        lines.extend((
            render_section("Selection 選股診斷（Future Target 僅於回放後 join）", number=3),
            render_table(
                ("指標", "Baseline", "Score Sort", "差異", "判讀"),
                diagnostic_rows,
                alignments=("left", "right", "right", "right", "left"),
            ),
            "Future Target 未進入候選排序、資金配置或成交決策。",
        ))

    lines.append(render_section("判讀限制", number=4 if strategy_diagnostics else 3))
    if not metadata.get("lookahead_safe_active_param_schedule"):
        lines.append("⚠️ 本次使用單一／static 參數，只能視為敏感度診斷，不是無前視部署證據。")
    if metadata["comparison_mode"] == COMPARISON_MODE_SCORE_RANKING:
        lines.append(
            "本報表使用 Selection point-in-time Scores 與當期歷史 active params；"
            "可用於 Selection 內決定是否進入參數適應，正式效果仍須由凍結後 OOS 驗證。"
            if metadata.get("score_source") == SCORE_SOURCE_SELECTION_POINT_IN_TIME
            else "本報表是既有 forward period 的排序機制比較，不得依結果回頭調整模型或排序規則。"
        )
    else:
        lines.append("本報表只驗證固定 active 操作點，不得依結果回頭調整 threshold、模型或訓練條件。")
    return "\n".join(lines)


def _remove_legacy_html_outputs(output_dir: Path) -> None:
    for filename in ("strategy_comparison.html", "score_ranking_capture_audit.html"):
        path = output_dir / filename
        if path.is_file():
            path.unlink()

def _run_scenario(
    *, name, data_dir, param_source_kind, params, start_date, end_date,
    max_positions, enable_rotation, quiet, replay_counts=None,
    replay_execution_rows=None, ranking_source=None,
):
    print(f"\n[{name}] 建立市場與訊號快取")
    source_context = (
        nullcontext()
        if not ranking_source
        else breakout_quality_ranking_source_context(**ranking_source)
    )
    with source_context:
        return _run_scenario_inside_source_context(
            name=name, data_dir=data_dir, param_source_kind=param_source_kind,
            params=params, start_date=start_date, end_date=end_date,
            max_positions=max_positions, enable_rotation=enable_rotation, quiet=quiet,
            replay_counts=replay_counts, replay_execution_rows=replay_execution_rows,
        )

def _run_scenario_inside_source_context(
    *, name, data_dir, param_source_kind, params, start_date, end_date,
    max_positions, enable_rotation, quiet, replay_counts=None,
    replay_execution_rows=None,
):
    if param_source_kind == "single_param":
        context = load_portfolio_market_context(str(data_dir), params, verbose=not quiet)
        print(f"[{name}] 執行 {start_date} ～ {end_date}")
        result = run_portfolio_simulation_prepared(
            context["all_dfs_fast"], context["all_trade_logs"], context["sorted_dates"], params,
            max_positions=max_positions, enable_rotation=enable_rotation,
            start_year=pd.Timestamp(start_date).year, start_date=start_date, end_date=end_date,
            benchmark_ticker=PORTFOLIO_DEFAULT_BENCHMARK_TICKER, verbose=not quiet,
            pit_stats_index=context.get("all_pit_stats_index"),
            replay_counts=replay_counts,
            replay_execution_rows=replay_execution_rows,
        )
    elif param_source_kind in {"static_active_param_ensemble", "rolling_active_param_ensemble"}:
        print(f"[{name}] 執行 active-param ensemble replay {start_date} ～ {end_date}")
        result = run_portfolio_simulation_with_param_ensemble(
            str(data_dir), params,
            max_positions=max_positions, enable_rotation=enable_rotation,
            start_year=pd.Timestamp(start_date).year, start_date=start_date, end_date=end_date,
            benchmark_ticker=PORTFOLIO_DEFAULT_BENCHMARK_TICKER,
            fixed_risk=None, verbose=not quiet, replay_counts=replay_counts,
            replay_execution_rows=replay_execution_rows,
        )
    elif param_source_kind == "rolling_oos_param_schedule":
        print(f"[{name}] 執行 rolling active-param replay {start_date} ～ {end_date}")
        result = run_portfolio_simulation_with_param_schedule(
            str(data_dir), params,
            max_positions=max_positions, enable_rotation=enable_rotation,
            start_year=pd.Timestamp(start_date).year, start_date=start_date, end_date=end_date,
            benchmark_ticker=PORTFOLIO_DEFAULT_BENCHMARK_TICKER,
            fixed_risk=None, verbose=not quiet, replay_counts=replay_counts,
            replay_execution_rows=replay_execution_rows,
        )
    else:
        raise ValueError(f"不支援的參數來源類型: {param_source_kind}")
    return _unpack_result(result)

def _flatten_candidate_replay_rows(replay_counts: dict[str, dict[str, Any]], field: str) -> pd.DataFrame:
    if field not in {"candidate_rows", "orderable_rows"}:
        raise ValueError(f"不支援的candidate replay field: {field}")
    rows: list[dict[str, Any]] = []
    for ticker in sorted(replay_counts):
        bucket = replay_counts.get(ticker) or {}
        for raw in list(bucket.get(field) or []):
            row = dict(raw or {})
            row["ticker"] = str(row.get("ticker") or ticker)
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=[
            "ticker", "trade_date", "candidate_date", "signal_date",
            "candidate_type", "entry_source", "is_orderable", "high_len",
            "ensemble_vote_count", "qty", "sort_value", "historical_ev",
            "historical_win_rate", "historical_trade_count",
            "breakout_quality_score", "breakout_quality_score_date",
        ])
    frame = pd.DataFrame(rows)
    for column in ("trade_date", "candidate_date", "signal_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    return frame.sort_values(
        ["trade_date", "ticker", "signal_date", "candidate_type"],
        kind="mergesort",
    ).reset_index(drop=True)

def _flatten_selected_buy_rows(trade_history: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(trade_history).copy()
    columns = ["ticker", "trade_date", "signal_date", "type"]
    if frame.empty or not {"Date", "Ticker", "Type"}.issubset(frame.columns):
        return pd.DataFrame(columns=columns)
    buy_mask = frame["Type"].fillna("").astype(str).str.startswith("買進 (")
    out = frame.loc[buy_mask].copy()
    if out.empty:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame({
        "ticker": out["Ticker"].fillna("").astype(str).str.strip(),
        "trade_date": pd.to_datetime(out["Date"], errors="coerce").dt.strftime("%Y-%m-%d"),
        "signal_date": pd.to_datetime(
            out.get("買訊日", pd.Series("", index=out.index)), errors="coerce"
        ).dt.strftime("%Y-%m-%d"),
        "type": out["Type"].fillna("").astype(str),
    })
    return out.dropna(subset=["trade_date"]).sort_values(
        ["trade_date", "ticker", "signal_date"], kind="mergesort"
    ).reset_index(drop=True)

def _selection_target_lookup(*, root: Path, filter_id: str, architecture: str, profile: str) -> pd.DataFrame:
    scores = load_selection_point_in_time_score_table(
        str(root), filter_id, architecture, profile
    ).reset_index()
    bundle = load_continuous_ranker_data(
        filter_id=filter_id,
        model_architecture=architecture,
        experiment_profile=profile,
        preload_feature_bank=False,
        allow_stale_source=False,
        project_root=root,
    )
    groups = bundle.group_table[["group_index", "ticker", "date", "label"]].copy()
    groups["ticker"] = groups["ticker"].fillna("").astype(str).str.strip()
    groups["date"] = pd.to_datetime(groups["date"], errors="raise").dt.strftime("%Y-%m-%d")
    groups["target_raw_r"] = bundle.raw_target
    groups["target_available"] = bundle.target_valid & pd.Series(bundle.raw_target).map(math.isfinite).to_numpy()
    joined = scores.merge(groups, on="group_index", how="left", validate="one_to_one", suffixes=("", "_dataset"))
    mismatch = (
        joined["ticker"] != joined["ticker_dataset"]
    ) | (
        joined["date"] != joined["date_dataset"]
    )
    if bool(mismatch.any()):
        raise ValueError("Selection PIT Score與Dataset group identity不一致")
    return joined[[
        "ticker", "date", "group_index", "breakout_quality_score", "fold_id",
        "model_information_cutoff", "label", "target_raw_r", "target_available",
    ]].rename(columns={"date": "signal_date"})

def _strategy_selection_diagnostics(
    *, orderable: pd.DataFrame, selected: pd.DataFrame, lookup: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    orderable_work = pd.DataFrame(orderable).copy()
    selected_work = pd.DataFrame(selected).copy()
    lookup_work = pd.DataFrame(lookup).copy()

    for frame, columns in (
        (orderable_work, ("trade_date", "signal_date")),
        (selected_work, ("trade_date", "signal_date")),
        (lookup_work, ("signal_date",)),
    ):
        for column in columns:
            if column in frame.columns:
                frame[column] = pd.to_datetime(
                    frame[column], errors="coerce"
                ).dt.strftime("%Y-%m-%d").fillna("")

    # Continuation／re-entry 的交易 signal_date 可以晚於原始 breakout event；
    # runtime Score 與 Future Target 都必須以保存下來的原始 score_date 對回 PIT 工件。
    raw_score_dates = orderable_work.get(
        "breakout_quality_score_date",
        pd.Series("", index=orderable_work.index, dtype="object"),
    ).fillna("").astype(str).str.strip()
    parsed_score_dates = pd.to_datetime(raw_score_dates, errors="coerce")
    invalid_score_dates = raw_score_dates.ne("") & parsed_score_dates.isna()
    if bool(invalid_score_dates.any()):
        bad = orderable_work.loc[invalid_score_dates].iloc[0]
        bad_score_date = raw_score_dates.loc[invalid_score_dates].iloc[0]
        raise ValueError(
            "策略replay保存的Breakout Quality score_date無法解析: "
            f"ticker={bad.get('ticker')}, trade_date={bad.get('trade_date')}, "
            f"signal_date={bad.get('signal_date')}, score_date={bad_score_date!r}"
        )
    normalized_score_dates = parsed_score_dates.dt.strftime("%Y-%m-%d").fillna("")
    orderable_work["score_event_date"] = normalized_score_dates.where(
        normalized_score_dates.ne(""), orderable_work.get("signal_date", "")
    )

    lookup_work = lookup_work.rename(columns={
        "signal_date": "score_event_date",
        "breakout_quality_score": "pit_breakout_quality_score",
    })
    if bool(lookup_work.duplicated(["ticker", "score_event_date"]).any()):
        raise ValueError("Selection PIT diagnostic lookup同一ticker/score_event_date不唯一")

    orderable_joined = orderable_work.merge(
        lookup_work,
        on=["ticker", "score_event_date"],
        how="left",
        validate="many_to_one",
    )
    score_available = pd.to_numeric(
        orderable_joined.get(
            "pit_breakout_quality_score",
            pd.Series(float("nan"), index=orderable_joined.index),
        ),
        errors="coerce",
    ).map(math.isfinite)
    runtime_score_available = pd.to_numeric(
        orderable_joined.get(
            "breakout_quality_score",
            pd.Series(float("nan"), index=orderable_joined.index),
        ),
        errors="coerce",
    ).map(math.isfinite)
    declared_runtime_available = orderable_joined.get(
        "breakout_quality_score_available",
        pd.Series(False, index=orderable_joined.index),
    ).fillna(False).astype(bool)
    runtime_rows = declared_runtime_available | runtime_score_available
    if bool(runtime_rows.any()):
        expected = pd.to_numeric(
            orderable_joined.loc[runtime_rows, "pit_breakout_quality_score"],
            errors="coerce",
        )
        actual = pd.to_numeric(
            orderable_joined.loc[runtime_rows, "breakout_quality_score"],
            errors="coerce",
        )
        mismatch = (
            (~expected.map(math.isfinite))
            | (~actual.map(math.isfinite))
            | ((actual - expected).abs() > 1e-12)
        )
        if bool(mismatch.any()):
            bad_index = mismatch[mismatch].index[0]
            bad = orderable_joined.loc[bad_index]
            raise ValueError(
                "策略replay使用的Breakout Quality Score與PIT score table不一致: "
                f"ticker={bad.get('ticker')}, trade_date={bad.get('trade_date')}, "
                f"signal_date={bad.get('signal_date')}, "
                f"score_event_date={bad.get('score_event_date')}, "
                f"runtime_score={actual.loc[bad_index]}, pit_score={expected.loc[bad_index]}"
            )

    target_available = orderable_joined.get(
        "target_available", pd.Series(False, index=orderable_joined.index)
    ).fillna(False).astype(bool)

    occurrence_keys = ["ticker", "trade_date", "signal_date"]
    occurrence_columns = [
        *occurrence_keys,
        "score_event_date",
        "target_raw_r",
        "target_available",
        "pit_breakout_quality_score",
    ]
    occurrence_score_event_counts = orderable_joined.groupby(
        occurrence_keys, dropna=False
    )["score_event_date"].nunique(dropna=False)
    if bool((occurrence_score_event_counts > 1).any()):
        bad_key = occurrence_score_event_counts[occurrence_score_event_counts > 1].index[0]
        raise ValueError(
            "同一策略候選發生多個Breakout Quality score event date: "
            f"ticker={bad_key[0]}, trade_date={bad_key[1]}, signal_date={bad_key[2]}"
        )
    occurrence_lookup = orderable_joined[occurrence_columns].drop_duplicates(
        occurrence_keys, keep="first"
    )
    selected_joined = selected_work.merge(
        occurrence_lookup,
        on=occurrence_keys,
        how="left",
        validate="many_to_one",
    )

    day_rows = []
    valid_orderable = orderable_joined.loc[target_available].copy()
    for trade_date, day in valid_orderable.groupby("trade_date", sort=True):
        chosen = selected_joined[selected_joined["trade_date"] == trade_date]
        chosen_keys = set(zip(chosen["ticker"], chosen["signal_date"]))
        if not chosen_keys:
            continue
        day = day.drop_duplicates(["ticker", "signal_date"], keep="first").copy()
        day["target_percentile"] = day["target_raw_r"].rank(method="average", pct=True)
        selected_day = day[[
            (ticker, signal_date) in chosen_keys
            for ticker, signal_date in zip(day["ticker"], day["signal_date"])
        ]]
        if selected_day.empty:
            continue
        k = len(selected_day)
        top = day.nlargest(k, "target_raw_r", keep="first")
        top_keys = set(zip(top["ticker"], top["signal_date"]))
        retained = sum(key in top_keys for key in chosen_keys)
        day_rows.append({
            "trade_date": trade_date,
            "selected_count": k,
            "selected_target_mean_r": float(selected_day["target_raw_r"].mean()),
            "selected_target_percentile_mean": float(selected_day["target_percentile"].mean()),
            "target_top_k_retention": float(retained / k),
            "target_opportunity_gap_r": float(
                top["target_raw_r"].mean() - selected_day["target_raw_r"].mean()
            ),
        })
    daily = pd.DataFrame(day_rows)
    metrics = {
        "orderable_occurrences": int(len(orderable_joined)),
        "orderable_score_covered": int(score_available.sum()),
        "orderable_score_coverage_rate": float(score_available.mean()) if len(score_available) else None,
        "runtime_scored_orderable_occurrences": int(runtime_score_available.sum()),
        "runtime_score_identity_match": True,
        "orderable_target_covered": int(target_available.sum()),
        "orderable_target_coverage_rate": float(target_available.mean()) if len(target_available) else None,
        "selected_buy_rows": int(len(selected_joined)),
        "diagnostic_days": int(len(daily)),
        "selected_target_percentile_mean": float(daily["selected_target_percentile_mean"].mean()) if not daily.empty else None,
        "target_top_k_retention_mean": float(daily["target_top_k_retention"].mean()) if not daily.empty else None,
        "target_opportunity_gap_r_mean": float(daily["target_opportunity_gap_r"].mean()) if not daily.empty else None,
        "selected_target_mean_r": float(daily["selected_target_mean_r"].mean()) if not daily.empty else None,
        "future_target_used_for_runtime_sort": False,
    }
    return metrics, orderable_joined, selected_joined

def run_no_filter_candidate_replay_from_metadata(
    metadata: dict[str, Any],
    *,
    quiet: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Replay the exact no-filter historical OOS scenario and capture qualified candidates."""

    payload = dict(metadata or {})
    if str(payload.get("comparison_mode") or "") != COMPARISON_MODE_HARD_FILTER:
        raise ValueError("11C只接受hard-filter controlled comparison metadata")
    if str(payload.get("comparison_design") or "") != "historical_active_param_oos":
        raise ValueError("11C只接受historical_active_param_oos策略比較")
    if not bool(payload.get("lookahead_safe_active_param_schedule")):
        raise ValueError("11C策略比較metadata未證明lookahead-safe active params")
    param_source_kind = str(payload.get("param_source_kind") or "")
    if param_source_kind not in {"rolling_active_param_ensemble", "rolling_oos_param_schedule"}:
        raise ValueError(f"11C不支援的正式參數來源: {param_source_kind}")
    no_filter_params = payload.get("no_filter_params")
    if not isinstance(no_filter_params, dict) or not no_filter_params:
        raise ValueError("11C metadata缺少no_filter_params")
    period = dict(payload.get("comparison_period") or {})
    start_date = str(period.get("start") or "")
    end_date = str(period.get("end") or "")
    if not start_date or not end_date:
        raise ValueError("11C metadata缺少comparison period")
    data_dir = Path(str(payload.get("data_dir") or "")).resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"11C找不到strategy compare data_dir: {data_dir}")

    replay_counts: dict[str, dict[str, Any]] = {}
    scenario = _run_scenario(
        name="11C_no_filter_candidate_audit",
        data_dir=data_dir,
        param_source_kind=param_source_kind,
        params=no_filter_params,
        start_date=start_date,
        end_date=end_date,
        max_positions=int(payload.get("max_positions", 10) or 10),
        enable_rotation=bool(payload.get("enable_rotation", False)),
        quiet=bool(quiet),
        replay_counts=replay_counts,
    )
    qualified = _flatten_candidate_replay_rows(replay_counts, "candidate_rows")
    orderable = _flatten_candidate_replay_rows(replay_counts, "orderable_rows")
    return scenario, qualified, orderable

def run_existing_attribution(*, project_root=PROJECT_ROOT) -> dict[str, Any]:
    root = Path(project_root).resolve()
    comparison_mode = COMPARISON_MODE_HARD_FILTER
    labels = _comparison_labels(comparison_mode)
    filter_id = BREAKOUT_QUALITY_DEFAULT_FILTER_ID
    contract = load_runtime_artifact_contract(str(root), filter_id)
    architecture = str(contract.manifest.get("model_architecture") or "")
    experiment_profile = str(contract.manifest.get("experiment_profile") or "")
    output_root = resolve_filter_model_output_dir(
        str(root), filter_id, architecture, experiment_profile
    )
    output_dir = _first_existing_comparison_dir(
        output_root,
        comparison_mode=COMPARISON_MODE_HARD_FILTER,
    )
    payload = _load_existing_comparison_payload(output_dir)
    metadata = dict(payload["metadata"] or {})
    metadata.setdefault("comparison_mode", COMPARISON_MODE_HARD_FILTER)
    if str(metadata.get("filter_id") or "") != filter_id:
        raise ValueError("既有策略比較結果的 filter_id 與目前 active filter 不一致")
    if str(metadata.get("model_architecture") or "") != architecture:
        raise ValueError("既有策略比較結果的 architecture 與目前 runtime artifact 不一致")
    if str(metadata.get("experiment_profile") or "") != experiment_profile:
        raise ValueError("既有策略比較結果的 experiment profile 與目前 runtime artifact 不一致")

    no_filter_trades_path = output_dir / "no_filter_trades.csv"
    quality_trades_path = output_dir / "quality_filter_trades.csv"
    if not no_filter_trades_path.is_file() or not quality_trades_path.is_file():
        raise FileNotFoundError(
            "attribution-only 需要既有 no_filter_trades.csv 與 quality_filter_trades.csv；"
            "請先完成一次正式 strategy compare。"
        )
    no_filter_history = pd.read_csv(no_filter_trades_path, encoding="utf-8-sig")
    quality_history = pd.read_csv(quality_trades_path, encoding="utf-8-sig")

    yearly = _normalize_yearly_completeness(pd.DataFrame(payload.get("yearly") or []))
    baseline = dict(payload.get("no_filter") or {})
    quality = dict(payload.get("quality_filter") or {})
    if not yearly.empty:
        _refresh_yearly_summary(baseline, yearly, return_column="no_filter_return_pct")
        _refresh_yearly_summary(quality, yearly, return_column="quality_filter_return_pct")
    deltas = _delta(quality, baseline)
    metadata["schema_version"] = SCHEMA_VERSION
    metadata["trade_attribution_schema_version"] = ATTRIBUTION_SCHEMA_VERSION
    refreshed = _to_json_native({
        "metadata": metadata,
        "no_filter": baseline,
        labels["active_name"]: quality,
        f"{labels['active_name']}_minus_no_filter": deltas,
        "yearly": yearly.to_dict("records"),
    })
    (output_dir / "strategy_comparison.json").write_text(
        json.dumps(refreshed, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "strategy_comparison.md").write_text(
        _markdown_report(metadata, baseline, quality, deltas, yearly),
        encoding="utf-8",
    )
    yearly.to_csv(output_dir / "yearly_returns_comparison.csv", index=False, encoding="utf-8-sig")
    attribution = write_trade_attribution_outputs(
        project_root=root,
        output_dir=output_dir,
        filter_id=filter_id,
        threshold=float(metadata.get("threshold", BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD)),
        metadata=metadata,
        no_filter_trade_history=no_filter_history,
        quality_filter_trade_history=quality_history,
        no_filter_portfolio_total_r=baseline.get("portfolio_total_r"),
        quality_filter_portfolio_total_r=quality.get("portfolio_total_r"),
    )
    print_artifact_paths(
        (("交易歸因 Markdown", output_dir / "trade_attribution.md"),),
        project_root=root,
    )
    return attribution

def run_existing_score_ranking_capture_audit(
    *,
    project_root=PROJECT_ROOT,
    filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    model_architecture=BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
    experiment_profile=BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
    param_policy=PARAM_POLICY_BASE_FINALIST_BEST,
    score_source=SCORE_SOURCE_SELECTION_POINT_IN_TIME,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    labels = _comparison_labels(COMPARISON_MODE_SCORE_RANKING)
    output_root = resolve_filter_model_output_dir(
        str(root), filter_id, model_architecture, experiment_profile
    )
    output_name = _comparison_output_dir_name(
        COMPARISON_MODE_SCORE_RANKING,
        labels,
        param_policy=param_policy,
    )
    if score_source == SCORE_SOURCE_SELECTION_POINT_IN_TIME:
        output_name += "_selection_point_in_time"
    output_dir = output_root / output_name
    payload = _load_existing_comparison_payload(output_dir)
    metadata = dict(payload.get("metadata") or {})
    if str(metadata.get("comparison_mode") or "") != COMPARISON_MODE_SCORE_RANKING:
        raise ValueError("既有工件不是score-ranking策略比較")
    for key, expected in (
        ("filter_id", filter_id),
        ("model_architecture", model_architecture),
        ("experiment_profile", experiment_profile),
        ("score_source", score_source),
    ):
        if str(metadata.get(key) or "") != str(expected):
            raise ValueError(
                f"既有score-ranking工件{key}與目前要求不一致: "
                f"artifact={metadata.get(key)!r}, expected={expected!r}"
            )

    required = {
        "no_filter_trades": output_dir / "no_filter_trades.csv",
        "score_ranking_trades": output_dir / "score_ranking_trades.csv",
        "no_filter_selected": output_dir / "no_filter_selected_target_diagnostics.csv",
        "score_ranking_selected": output_dir / "score_ranking_selected_target_diagnostics.csv",
        "no_filter_capacity": output_dir / "no_filter_daily_capacity.csv",
        "score_ranking_capacity": output_dir / "score_ranking_daily_capacity.csv",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("capture-audit-only缺少既有工件: " + ", ".join(missing))

    baseline = dict(payload.get("no_filter") or {})
    quality = dict(payload.get("score_ranking") or {})
    deltas = dict(payload.get("score_ranking_minus_no_filter") or _delta(quality, baseline))
    yearly = _normalize_yearly_completeness(pd.DataFrame(payload.get("yearly") or []))
    diagnostics = payload.get("selection_diagnostics")
    (output_dir / "strategy_comparison.md").write_text(
        _markdown_report(metadata, baseline, quality, deltas, yearly, diagnostics),
        encoding="utf-8",
    )
    _remove_legacy_html_outputs(output_dir)
    result = build_score_ranking_capture_audit(
        metadata=metadata,
        baseline_summary=baseline,
        score_sort_summary=quality,
        baseline_trade_history=pd.read_csv(required["no_filter_trades"], encoding="utf-8-sig"),
        score_sort_trade_history=pd.read_csv(required["score_ranking_trades"], encoding="utf-8-sig"),
        baseline_selected_target_diagnostics=pd.read_csv(required["no_filter_selected"], encoding="utf-8-sig"),
        score_sort_selected_target_diagnostics=pd.read_csv(required["score_ranking_selected"], encoding="utf-8-sig"),
        selection_diagnostics=diagnostics,
        baseline_daily_capacity=pd.read_csv(required["no_filter_capacity"], encoding="utf-8-sig"),
        score_sort_daily_capacity=pd.read_csv(required["score_ranking_capacity"], encoding="utf-8-sig"),
    )
    audit_payload = write_score_ranking_capture_audit_outputs(
        result=result,
        output_dir=output_dir,
    )
    payload["score_ranking_capture_audit"] = audit_payload
    (output_dir / "strategy_comparison.json").write_text(
        json.dumps(_to_json_native(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("\n" + _render_strategy_console_report(
        metadata,
        baseline,
        quality,
        deltas,
        yearly,
        diagnostics,
        capture_result=result,
    ))
    if not compact_console_enabled():
        print("\n" + render_capture_audit_console(result))
    print_artifact_paths(
        (
            ("策略比較 Markdown", output_dir / "strategy_comparison.md"),
            ("策略比較 JSON", output_dir / "strategy_comparison.json"),
            ("Capture audit Markdown", output_dir / "score_ranking_capture_audit.md"),
            ("Capture audit JSON", output_dir / "score_ranking_capture_audit.json"),
        ),
        project_root=root,
    )
    return audit_payload

def _resolve_comparison_period(contract) -> tuple[str, str]:
    start = contract.execution_start
    end = contract.available_through
    if end < start:
        raise ValueError(
            "breakout quality 策略執行期間不合法: "
            f"execution_start={start}, available_through={end}"
        )
    return start.isoformat(), end.isoformat()

def run_comparison(
    *, project_root=PROJECT_ROOT, dataset="full", params_path=None,
    param_policy=PARAM_POLICY_AUTO, max_positions=10, enable_rotation=False,
    fixed_risk=None, max_position_cap_pct=None, allow_static_diagnostic=False,
    comparison_mode=COMPARISON_MODE_HARD_FILTER,
    ranking_policy=BREAKOUT_QUALITY_RANKING_POLICY_SCORE,
    optional_entry_filter_policy=OPTIONAL_ENTRY_FILTER_POLICY_CURRENT,
    filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    score_source=SCORE_SOURCE_CANONICAL_RUNTIME,
    model_architecture=BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
    experiment_profile=BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
    output_dir_override=None,
    comparison_start_date=None,
    comparison_end_date=None,
    quiet=False,
):
    root = Path(project_root).resolve()
    comparison_mode = str(comparison_mode)
    ranking_policy = str(ranking_policy).strip()
    optional_entry_filter_policy = str(optional_entry_filter_policy).strip()
    score_source = str(score_source)
    filter_id = str(filter_id)
    model_architecture = str(model_architecture)
    experiment_profile = str(experiment_profile)
    labels = _comparison_labels(comparison_mode)
    if score_source not in SUPPORTED_RANKING_SCORE_SOURCES:
        raise ValueError(f"不支援的 score source: {score_source!r}")
    if ranking_policy not in SUPPORTED_BREAKOUT_QUALITY_RANKING_POLICIES:
        raise ValueError(f"不支援的 ranking policy: {ranking_policy!r}")
    if optional_entry_filter_policy not in SUPPORTED_OPTIONAL_ENTRY_FILTER_POLICIES:
        raise ValueError(
            f"不支援的 optional entry filter policy: {optional_entry_filter_policy!r}"
        )
    if comparison_mode == COMPARISON_MODE_HARD_FILTER:
        if score_source != SCORE_SOURCE_CANONICAL_RUNTIME:
            raise ValueError("hard-filter策略比較只接受canonical_runtime score source")
        if ranking_policy != BREAKOUT_QUALITY_RANKING_POLICY_SCORE:
            raise ValueError("hard-filter策略比較不可指定capital-aware ranking policy")

    runtime_contract = None
    pit_contract = None
    ranking_source = None
    if score_source == SCORE_SOURCE_SELECTION_POINT_IN_TIME:
        if comparison_mode != COMPARISON_MODE_SCORE_RANKING:
            raise ValueError("Selection PIT score source只支援score-ranking比較")
        pit_contract = load_selection_point_in_time_ranking_contract(
            str(root), filter_id, model_architecture, experiment_profile
        )
        workflow_settings = get_breakout_quality_workflow_settings()
        if (
            workflow_settings.filter_id == filter_id
            and workflow_settings.model_architecture == model_architecture
            and workflow_settings.experiment_profile == experiment_profile
            and int(pit_contract.seed) != int(workflow_settings.seed)
        ):
            raise ValueError(
                "Selection PIT工件seed與目前workflow不一致，必須重建："
                f"artifact={pit_contract.seed}, workflow={workflow_settings.seed}"
            )
        manifest_architecture = pit_contract.model_architecture
        manifest_profile = pit_contract.experiment_profile
        start_date = pit_contract.available_from
        end_date = pit_contract.available_through
        ranking_source = {
            "score_source": score_source,
            "model_architecture": manifest_architecture,
            "experiment_profile": manifest_profile,
            "ranking_policy": ranking_policy,
        }
    else:
        try:
            runtime_contract = load_runtime_artifact_contract(str(root), filter_id)
        except (FileNotFoundError, ValueError) as exc:
            raise RuntimeError(
                "策略對照需要 active breakout-quality 的正式 forward-OOS scores.csv；"
                "請先執行 export-scores --scope forward_oos。"
            ) from exc
        manifest_architecture = str(runtime_contract.manifest.get("model_architecture") or "")
        manifest_profile = str(runtime_contract.manifest.get("experiment_profile") or "")
        if manifest_architecture != model_architecture or manifest_profile != experiment_profile:
            raise ValueError(
                "runtime artifact與指定模型identity不一致: "
                f"artifact={manifest_architecture}/{manifest_profile}, "
                f"requested={model_architecture}/{experiment_profile}"
            )
        artifact_threshold = float(
            runtime_contract.manifest.get("fixed_evaluation_threshold", float("nan"))
        )
        configured_threshold = float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD)
        if not math.isclose(artifact_threshold, configured_threshold, rel_tol=0.0, abs_tol=0.0):
            raise ValueError(
                "active threshold與模型OOS前固定threshold不一致: "
                f"artifact={artifact_threshold}, policy={configured_threshold}"
            )
        start_date, end_date = _resolve_comparison_period(runtime_contract)
        if comparison_mode == COMPARISON_MODE_SCORE_RANKING:
            ranking_source = {
                "score_source": SCORE_SOURCE_CANONICAL_RUNTIME,
                "model_architecture": None,
                "experiment_profile": None,
                "ranking_policy": ranking_policy,
            }

    if (comparison_start_date is None) != (comparison_end_date is None):
        raise ValueError("comparison_start_date與comparison_end_date必須同時提供")
    if comparison_start_date is not None:
        default_start = pd.Timestamp(start_date).normalize()
        default_end = pd.Timestamp(end_date).normalize()
        requested_start = pd.Timestamp(str(comparison_start_date)).normalize()
        requested_end = pd.Timestamp(str(comparison_end_date)).normalize()
        if pd.isna(requested_start) or pd.isna(requested_end) or requested_end < requested_start:
            raise ValueError("指定策略比較期間不合法")
        if requested_start < default_start or requested_end > default_end:
            raise ValueError(
                "指定策略比較期間超出Score可用範圍："
                f"requested={requested_start.date()}~{requested_end.date()}, "
                f"available={default_start.date()}~{default_end.date()}"
            )
        start_date = requested_start.strftime("%Y-%m-%d")
        end_date = requested_end.strftime("%Y-%m-%d")

    param_policy = str(param_policy)
    resolved_params_path = _resolve_params_path(
        root=root, params_path=params_path, param_policy=param_policy,
        allow_static_diagnostic=allow_static_diagnostic, score_source=score_source,
    )
    if not resolved_params_path.is_file():
        source_hint = (
            "請先完成Selection nested ROOS active-param工件"
            if score_source == SCORE_SOURCE_SELECTION_POINT_IN_TIME
            else "請先完成正式rolling OOS active-param工件"
        )
        raise FileNotFoundError(
            f"找不到策略比較參數檔: {resolved_params_path}；{source_hint}。"
        )
    if fixed_risk is not None and not (0.0 < float(fixed_risk) <= 1.0):
        raise ValueError("fixed_risk必須介於0與1")
    if max_position_cap_pct is not None and not (0.0 < float(max_position_cap_pct) <= 1.0):
        raise ValueError("max_position_cap_pct必須介於0與1")
    param_source = _load_param_source(resolved_params_path)
    param_policy_contract = _validate_requested_param_policy(param_source, param_policy)
    (
        param_source_kind,
        no_filter_params,
        quality_params,
        no_filter_param_payload,
        quality_param_payload,
        ensemble_policy,
    ) = _build_controlled_param_source_pair(
        param_source,
        filter_id=filter_id,
        threshold=float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD),
        fixed_risk=None if fixed_risk is None else float(fixed_risk),
        max_position_cap_pct=(
            None if max_position_cap_pct is None else float(max_position_cap_pct)
        ),
        comparison_mode=comparison_mode,
        optional_entry_filter_policy=optional_entry_filter_policy,
    )

    is_rolling_source = param_source_kind in {
        "rolling_oos_param_schedule", "rolling_active_param_ensemble"
    }
    if not is_rolling_source and not allow_static_diagnostic:
        raise ValueError(
            "單一／static參數跨歷史回放無法證明無前視；"
            "請使用rolling active-param JSON，或加--allow-static-diagnostic僅作敏感度診斷。"
        )
    data_dir = Path(get_dataset_dir(str(root), dataset)).resolve()
    if param_source_kind == "rolling_oos_param_schedule":
        param_start, param_end = get_active_param_date_range(param_source["payload"])
    elif param_source_kind == "rolling_active_param_ensemble":
        param_start, param_end = get_active_param_ensemble_date_range(param_source["payload"])
    else:
        param_start, param_end = "", ""
    if is_rolling_source and (
        pd.Timestamp(param_start) > pd.Timestamp(start_date)
        or pd.Timestamp(param_end) < pd.Timestamp(end_date)
    ):
        raise ValueError(
            "rolling active-param期間未完整覆蓋策略比較期間："
            f"params={param_start}~{param_end}, comparison={start_date}~{end_date}"
        )

    baseline_replay_counts = {} if comparison_mode == COMPARISON_MODE_SCORE_RANKING else None
    quality_replay_counts = {} if comparison_mode == COMPARISON_MODE_SCORE_RANKING else None
    baseline_payload = _run_scenario(
        name="no_filter", data_dir=data_dir, param_source_kind=param_source_kind,
        params=no_filter_params, start_date=start_date, end_date=end_date,
        max_positions=max_positions, enable_rotation=enable_rotation, quiet=quiet,
        replay_counts=baseline_replay_counts,
    )
    quality_payload = _run_scenario(
        name=labels["active_name"], data_dir=data_dir,
        param_source_kind=param_source_kind, params=quality_params,
        start_date=start_date, end_date=end_date, max_positions=max_positions,
        enable_rotation=enable_rotation, quiet=quiet,
        replay_counts=quality_replay_counts, ranking_source=ranking_source,
    )
    _assert_shared_benchmark(baseline_payload, quality_payload)

    baseline = _scenario_summary(baseline_payload)
    quality = _scenario_summary(quality_payload)
    deltas = _delta(quality, baseline)
    yearly = _build_yearly_comparison(
        baseline_payload["profile"], quality_payload["profile"]
    )
    if comparison_mode == COMPARISON_MODE_SCORE_RANKING:
        yearly = yearly.rename(
            columns={"quality_filter_return_pct": "score_ranking_return_pct"}
        )

    if output_dir_override is None:
        output_dir_name = _comparison_output_dir_name(
            comparison_mode,
            labels,
            param_policy=param_policy,
            ranking_policy=ranking_policy,
            optional_entry_filter_policy=optional_entry_filter_policy,
        )
        if score_source == SCORE_SOURCE_SELECTION_POINT_IN_TIME:
            output_dir_name += "_selection_point_in_time"
        output_dir = (
            resolve_filter_model_output_dir(
                str(root), filter_id, manifest_architecture, manifest_profile
            ) / output_dir_name
        )
        output_scope = "canonical"
    else:
        output_dir = Path(output_dir_override)
        if not output_dir.is_absolute():
            output_dir = root / output_dir
        output_dir = output_dir.resolve()
        output_scope = "caller_override"
    output_dir.mkdir(parents=True, exist_ok=True)

    strategy_diagnostics = None
    baseline_selected_joined = pd.DataFrame()
    quality_selected_joined = pd.DataFrame()
    if comparison_mode == COMPARISON_MODE_SCORE_RANKING:
        baseline_orderable = _flatten_candidate_replay_rows(
            baseline_replay_counts or {}, "orderable_rows"
        )
        quality_orderable = _flatten_candidate_replay_rows(
            quality_replay_counts or {}, "orderable_rows"
        )
        baseline_selected = _flatten_selected_buy_rows(baseline_payload["trade_history"])
        quality_selected = _flatten_selected_buy_rows(quality_payload["trade_history"])
        baseline_orderable.to_csv(
            output_dir / "no_filter_orderable_candidates.csv", index=False,
            encoding="utf-8-sig"
        )
        quality_orderable.to_csv(
            output_dir / "score_ranking_orderable_candidates.csv", index=False,
            encoding="utf-8-sig"
        )
        baseline_selected.to_csv(
            output_dir / "no_filter_selected_buys.csv", index=False,
            encoding="utf-8-sig"
        )
        quality_selected.to_csv(
            output_dir / "score_ranking_selected_buys.csv", index=False,
            encoding="utf-8-sig"
        )
        if score_source == SCORE_SOURCE_SELECTION_POINT_IN_TIME:
            lookup = _selection_target_lookup(
                root=root, filter_id=filter_id, architecture=manifest_architecture,
                profile=manifest_profile,
            )
            baseline_diag, baseline_orderable_joined, baseline_selected_joined = (
                _strategy_selection_diagnostics(
                    orderable=baseline_orderable,
                    selected=baseline_selected,
                    lookup=lookup,
                )
            )
            quality_diag, quality_orderable_joined, quality_selected_joined = (
                _strategy_selection_diagnostics(
                    orderable=quality_orderable,
                    selected=quality_selected,
                    lookup=lookup,
                )
            )
            strategy_diagnostics = {
                "no_filter": baseline_diag,
                "score_ranking": quality_diag,
                "score_ranking_minus_no_filter": _delta(quality_diag, baseline_diag),
                "future_target_join_stage": "post_replay_offline_diagnostic_only",
                "future_target_used_for_runtime_sort": False,
            }
            baseline_orderable_joined.to_csv(
                output_dir / "no_filter_orderable_target_diagnostics.csv",
                index=False, encoding="utf-8-sig"
            )
            quality_orderable_joined.to_csv(
                output_dir / "score_ranking_orderable_target_diagnostics.csv",
                index=False, encoding="utf-8-sig"
            )
            baseline_selected_joined.to_csv(
                output_dir / "no_filter_selected_target_diagnostics.csv",
                index=False, encoding="utf-8-sig"
            )
            quality_selected_joined.to_csv(
                output_dir / "score_ranking_selected_target_diagnostics.csv",
                index=False, encoding="utf-8-sig"
            )

    if pit_contract is not None:
        score_signal_coverage = {
            "required_start": pit_contract.available_from,
            "first_scored_event": pit_contract.available_from,
            "available_through": pit_contract.available_through,
        }
        score_artifact_metadata = {
            "score_manifest_path": str(pit_contract.manifest_path),
            "score_path": str(pit_contract.score_path),
            "score_audit_path": str(pit_contract.audit_path),
            "model_validation_gate": pit_contract.model_validation_gate,
            "runtime_eligibility": dict(pit_contract.manifest.get("runtime_eligibility") or {}),
            "score_table": dict(pit_contract.manifest.get("artifacts", {}).get("scores") or {}),
        }
    else:
        score_signal_coverage = {
            "required_start": runtime_contract.required_signal_start.isoformat(),
            "first_scored_event": runtime_contract.available_from.isoformat(),
            "available_through": runtime_contract.available_through.isoformat(),
        }
        score_artifact_metadata = {
            "runtime_manifest_path": str(runtime_contract.paths.manifest_path),
            "runtime_score_path": str(runtime_contract.paths.score_path),
            "runtime_eligibility": dict(runtime_contract.manifest.get("runtime_eligibility") or {}),
            "score_table": dict(runtime_contract.manifest.get("score_table") or {}),
        }

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "comparison_mode": comparison_mode,
        "score_ranking_policy": ranking_policy if comparison_mode == COMPARISON_MODE_SCORE_RANKING else None,
        "optional_entry_filter_policy": optional_entry_filter_policy,
        "optional_entry_filter_fields": list(OPTIONAL_ENTRY_FILTER_FIELDS),
        "optional_entry_filter_forced_values": (
            {field: False for field in OPTIONAL_ENTRY_FILTER_FIELDS}
            if optional_entry_filter_policy == OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF
            else None
        ),
        "score_source": score_source,
        "dataset": dataset,
        "data_dir": str(data_dir),
        "params_path": str(resolved_params_path),
        "params_file_sha256": _sha256_file(resolved_params_path),
        "param_source_kind": param_source_kind,
        "requested_param_policy": param_policy,
        "param_selector": param_policy_contract["selector"],
        "runtime_member_count_min": param_policy_contract["member_count_min"],
        "runtime_member_count_max": param_policy_contract["member_count_max"],
        "runtime_min_agree": param_policy_contract["min_agree"],
        "ranking_scope": (
            "all_candidates_after_single_member_qualification"
            if comparison_mode == COMPARISON_MODE_SCORE_RANKING
            and param_policy_contract["selector"] == "base_finalist_best"
            else "same_runtime_vote_bucket_only"
            if comparison_mode == COMPARISON_MODE_SCORE_RANKING else None
        ),
        "comparison_design": (
            "selection_point_in_time_active_param_replay"
            if score_source == SCORE_SOURCE_SELECTION_POINT_IN_TIME
            else "historical_active_param_oos" if is_rolling_source
            else "static_param_diagnostic"
        ),
        "lookahead_safe_active_param_schedule": bool(is_rolling_source),
        "active_param_period": {"start": param_start, "end": param_end}
        if is_rolling_source else None,
        "active_param_ensemble_policy": ensemble_policy,
        "no_filter_params": no_filter_param_payload,
        f"{labels['active_name']}_params": quality_param_payload,
        "filter_id": filter_id,
        "model_architecture": manifest_architecture,
        "experiment_profile": manifest_profile,
        "threshold": float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD),
        "threshold_used_as_gate": bool(comparison_mode == COMPARISON_MODE_HARD_FILTER),
        "score_ranking_order": (
            (
                []
                if param_policy_contract["selector"] == "base_finalist_best"
                else ["runtime_member_vote_count_desc"]
            )
            + (
                ["breakout_quality_score_desc"]
                if ranking_policy == BREAKOUT_QUALITY_RANKING_POLICY_SCORE
                else ["capital_adjusted_score_desc"]
                if ranking_policy == BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_ADJUSTED
                else ["capital_deployment_bucket_desc", "breakout_quality_score_desc"]
            )
            + ["existing_buy_sort", "ticker_deterministic"]
            if comparison_mode == COMPARISON_MODE_SCORE_RANKING else None
        ),
        "capital_aware_ranking_contract": (
            {
                "projected_capital_fraction_source": "canonical_pretrade_proj_cost_div_sizing_capital",
                "deployment_rate": "min(1, projected_capital_fraction / max_position_cap_pct)",
                "capital_bucket_count": 3,
                "capital_bucket_scope": "same_day_score_available_orderable_candidates",
                "future_target_used": False,
            }
            if comparison_mode == COMPARISON_MODE_SCORE_RANKING
            and ranking_policy != BREAKOUT_QUALITY_RANKING_POLICY_SCORE
            else None
        ),
        "unscorable_candidate_policy": "fallback_original_buy_sort_without_exclusion",
        "max_positions": int(max_positions),
        "enable_rotation": bool(enable_rotation),
        "fixed_risk_override": None if fixed_risk is None else float(fixed_risk),
        "max_position_cap_pct_override": (
            None if max_position_cap_pct is None else float(max_position_cap_pct)
        ),
        "output_scope": output_scope,
        "output_dir": project_relative_display_path(output_dir, project_root=root),
        "comparison_period": {"start": start_date, "end": end_date},
        "score_signal_coverage": score_signal_coverage,
        "benchmark_ticker": PORTFOLIO_DEFAULT_BENCHMARK_TICKER,
        "controlled_param_difference": [_comparison_switch_spec(comparison_mode)[0]],
        "trade_attribution_schema_version": ATTRIBUTION_SCHEMA_VERSION,
        **score_artifact_metadata,
    }
    json_payload = _to_json_native({
        "metadata": metadata,
        "no_filter": baseline,
        labels["active_name"]: quality,
        f"{labels['active_name']}_minus_no_filter": deltas,
        "yearly": yearly.to_dict("records"),
        "selection_diagnostics": strategy_diagnostics,
    })
    (output_dir / "strategy_comparison.json").write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "strategy_comparison.md").write_text(
        _markdown_report(metadata, baseline, quality, deltas, yearly, strategy_diagnostics),
        encoding="utf-8",
    )
    _remove_legacy_html_outputs(output_dir)
    baseline_payload["equity_curve"].to_csv(
        output_dir / "no_filter_equity.csv", index=False, encoding="utf-8-sig"
    )
    quality_payload["equity_curve"].to_csv(
        output_dir / f"{labels['active_name']}_equity.csv", index=False,
        encoding="utf-8-sig"
    )
    baseline_payload["trade_history"].to_csv(
        output_dir / "no_filter_trades.csv", index=False, encoding="utf-8-sig"
    )
    quality_payload["trade_history"].to_csv(
        output_dir / f"{labels['active_name']}_trades.csv", index=False,
        encoding="utf-8-sig"
    )
    pd.DataFrame(
        baseline_payload["profile"].get("portfolio_capacity_rows") or []
    ).to_csv(
        output_dir / "no_filter_daily_capacity.csv", index=False,
        encoding="utf-8-sig"
    )
    pd.DataFrame(
        quality_payload["profile"].get("portfolio_capacity_rows") or []
    ).to_csv(
        output_dir / f"{labels['active_name']}_daily_capacity.csv", index=False,
        encoding="utf-8-sig"
    )
    yearly.to_csv(
        output_dir / "yearly_returns_comparison.csv", index=False,
        encoding="utf-8-sig"
    )
    capture_result = None
    capture_audit_payload = None
    if comparison_mode == COMPARISON_MODE_SCORE_RANKING:
        capture_result = build_score_ranking_capture_audit(
            metadata=metadata,
            baseline_summary=baseline,
            score_sort_summary=quality,
            baseline_trade_history=baseline_payload["trade_history"],
            score_sort_trade_history=quality_payload["trade_history"],
            baseline_selected_target_diagnostics=baseline_selected_joined,
            score_sort_selected_target_diagnostics=quality_selected_joined,
            selection_diagnostics=strategy_diagnostics,
            baseline_daily_capacity=pd.DataFrame(
                baseline_payload["profile"].get("portfolio_capacity_rows") or []
            ),
            score_sort_daily_capacity=pd.DataFrame(
                quality_payload["profile"].get("portfolio_capacity_rows") or []
            ),
        )
        capture_audit_payload = write_score_ranking_capture_audit_outputs(
            result=capture_result,
            output_dir=output_dir,
        )
        json_payload["score_ranking_capture_audit"] = capture_audit_payload
        (output_dir / "strategy_comparison.json").write_text(
            json.dumps(json_payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    if comparison_mode == COMPARISON_MODE_HARD_FILTER:
        write_trade_attribution_outputs(
            project_root=root, output_dir=output_dir, filter_id=filter_id,
            threshold=float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD),
            metadata=metadata,
            no_filter_trade_history=baseline_payload["trade_history"],
            quality_filter_trade_history=quality_payload["trade_history"],
            no_filter_closed_trade_rows=(baseline_payload["profile"] or {}).get("closed_trade_rows"),
            quality_filter_closed_trade_rows=(quality_payload["profile"] or {}).get("closed_trade_rows"),
            no_filter_portfolio_total_r=baseline.get("portfolio_total_r"),
            quality_filter_portfolio_total_r=quality.get("portfolio_total_r"),
        )
    print("\n" + _render_strategy_console_report(
        metadata,
        baseline,
        quality,
        deltas,
        yearly,
        strategy_diagnostics,
        capture_result=capture_result,
    ))
    if (
        comparison_mode == COMPARISON_MODE_SCORE_RANKING
        and not compact_console_enabled()
    ):
        print("\n" + render_capture_audit_console(capture_result))
    artifacts = [
        ("策略比較 Markdown", output_dir / "strategy_comparison.md"),
        ("策略比較 JSON", output_dir / "strategy_comparison.json"),
        ("年度比較 CSV", output_dir / "yearly_returns_comparison.csv"),
    ]
    if comparison_mode == COMPARISON_MODE_HARD_FILTER:
        artifacts.append(("交易歸因 Markdown", output_dir / "trade_attribution.md"))
    if capture_audit_payload is not None:
        artifacts.extend((
            ("Capture audit Markdown", output_dir / "score_ranking_capture_audit.md"),
            ("Capture audit JSON", output_dir / "score_ranking_capture_audit.json"),
        ))
    print_artifact_paths(artifacts, project_root=root)
    return json_payload

def main(argv=None):
    args = _parse_args(argv)
    if args.attribution_only and args.capture_audit_only:
        raise ValueError("--attribution-only 與 --capture-audit-only 不可同時使用")
    if args.capture_audit_only:
        if args.optional_entry_filters != OPTIONAL_ENTRY_FILTER_POLICY_CURRENT:
            raise ValueError("--capture-audit-only目前只支援沿用既有entry filters的正式目錄")
        if args.comparison_mode != COMPARISON_MODE_SCORE_RANKING:
            raise ValueError("--capture-audit-only只支援score-ranking")
        if args.ranking_policy != BREAKOUT_QUALITY_RANKING_POLICY_SCORE:
            raise ValueError("--capture-audit-only目前只讀取原始score ranking正式目錄")
        run_existing_score_ranking_capture_audit(
            filter_id=args.filter_id,
            model_architecture=args.model_architecture,
            experiment_profile=args.experiment_profile,
            param_policy=args.param_policy,
            score_source=args.score_source,
        )
        return 0
    if args.attribution_only:
        if args.comparison_mode != COMPARISON_MODE_HARD_FILTER:
            raise ValueError("--attribution-only 目前只支援 hard-filter 既有歸因")
        run_existing_attribution()
        return 0
    if args.max_positions < 1:
        raise ValueError("max_positions 必須 >= 1")
    if (args.start_date is None) != (args.end_date is None):
        raise ValueError("--start-date與--end-date必須同時使用")
    run_comparison(
        dataset=args.dataset,
        params_path=args.params,
        param_policy=args.param_policy,
        max_positions=args.max_positions,
        enable_rotation=args.rotation == "on",
        fixed_risk=args.fixed_risk,
        max_position_cap_pct=args.max_position_cap_pct,
        allow_static_diagnostic=args.allow_static_diagnostic,
        comparison_mode=args.comparison_mode,
        ranking_policy=args.ranking_policy,
        optional_entry_filter_policy=args.optional_entry_filters,
        filter_id=args.filter_id,
        score_source=args.score_source,
        model_architecture=args.model_architecture,
        experiment_profile=args.experiment_profile,
        comparison_start_date=args.start_date,
        comparison_end_date=args.end_date,
        quiet=args.quiet,
    )
    return 0


__all__ = [
    "main", "run_comparison", "run_existing_attribution",
    "run_existing_score_ranking_capture_audit",
    "run_no_filter_candidate_replay_from_metadata",
    "canonical_strategy_compare_output_dir_names", "_first_existing_comparison_dir",
    "_assert_controlled_param_pair", "_assert_controlled_ensemble_pair",
    "_assert_controlled_payload_pair", "_build_controlled_param_source_pair",
    "_load_param_source", "_capacity_summary", "_normalize_yearly_completeness",
    "_to_json_native", "_resolve_comparison_period",
    "COMPARISON_MODE_HARD_FILTER", "COMPARISON_MODE_SCORE_RANKING",
    "BREAKOUT_QUALITY_RANKING_POLICY_SCORE",
    "BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_ADJUSTED",
    "BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_BUCKET",
    "OPTIONAL_ENTRY_FILTER_POLICY_CURRENT", "OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF",
    "SUPPORTED_OPTIONAL_ENTRY_FILTER_POLICIES", "OPTIONAL_ENTRY_FILTER_FIELDS",
    "PARAM_POLICY_AUTO", "PARAM_POLICY_BASE_FINALIST_BEST", "PARAM_POLICY_BASE_FINALISTS_AGREE",
    "_resolve_params_path", "_resolve_param_selector", "_validate_requested_param_policy",
]
