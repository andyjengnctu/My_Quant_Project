"""Controlled no-filter vs active breakout-quality portfolio comparison."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from config.breakout_quality_policy import (
    BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD,
    BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
    BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
)
from core.active_param_ensemble import (
    ACTIVE_PARAM_ENSEMBLE_MODE_STATIC,
    get_active_param_ensemble_date_range,
    get_active_param_ensemble_policy,
    is_active_param_ensemble_payload,
    resolve_active_param_ensemble_mode,
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
from filters.breakout_quality.paths import resolve_filter_model_output_dir
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
SCHEMA_VERSION = 4
COMPARISON_MODE_HARD_FILTER = "hard-filter"
COMPARISON_MODE_SCORE_RANKING = "score-ranking"
COMPARISON_MODES = (COMPARISON_MODE_HARD_FILTER, COMPARISON_MODE_SCORE_RANKING)

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
    parser.add_argument("--params", default=None, help="正式 OOS 建議指定 rolling OOS active-param JSON；static 診斷才使用 run_best。")
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--rotation", choices=("off", "on"), default="off")
    parser.add_argument("--fixed-risk", type=float, default=None, help="選填；兩組同時覆寫 fixed_risk。")
    parser.add_argument(
        "--allow-static-diagnostic",
        action="store_true",
        help="允許以單一／static run_best 做非 OOS 診斷；不得視為無前視部署證據。",
    )
    parser.add_argument(
        "--attribution-only",
        action="store_true",
        help="只讀取既有 strategy_compare CSV/JSON 產生交易歸因，不重新執行 portfolio replay。",
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


def _apply_scenario_overrides(params, *, active: bool, comparison_mode: str, filter_id: str, threshold: float, fixed_risk: float | None):
    _comparison_switch_spec(comparison_mode)
    overrides = {
        "use_breakout_quality_filter": bool(active and comparison_mode == COMPARISON_MODE_HARD_FILTER),
        "use_breakout_quality_ranking": bool(active and comparison_mode == COMPARISON_MODE_SCORE_RANKING),
        "breakout_quality_filter_id": str(filter_id),
        "breakout_quality_score_threshold": float(threshold),
    }
    if fixed_risk is not None:
        overrides["fixed_risk"] = float(fixed_risk)
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
) -> dict:
    rewritten = {}
    for key, raw_params in dict(mapping or {}).items():
        base_params = build_params_from_mapping(raw_params)
        rewritten[str(key)] = params_to_json_dict(_apply_scenario_overrides(
            base_params,
            active=active,
            comparison_mode=comparison_mode,
            filter_id=filter_id,
            threshold=threshold,
            fixed_risk=fixed_risk,
        ))
    return rewritten


def _rewrite_ensemble_mapping(
    mapping: dict,
    *,
    active: bool,
    comparison_mode: str,
    filter_id: str,
    threshold: float,
    fixed_risk: float | None,
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
            output_member["params"] = params_to_json_dict(_apply_scenario_overrides(
                base_params,
                active=active,
                comparison_mode=comparison_mode,
                filter_id=filter_id,
                threshold=threshold,
                fixed_risk=fixed_risk,
            ))
            output_members.append(output_member)
        rewritten[str(key)] = output_members
    return rewritten


def _build_controlled_param_source_pair(
    source: dict[str, Any],
    *,
    filter_id: str,
    threshold: float,
    fixed_risk: float | None,
    comparison_mode: str = COMPARISON_MODE_HARD_FILTER,
) -> tuple[str, Any, Any, Any, Any, dict[str, Any] | None]:
    kind = str(source["kind"])
    if kind == "single_param":
        base_params = source["params"]
        no_filter_params = _apply_scenario_overrides(
            base_params, active=False, comparison_mode=comparison_mode, filter_id=filter_id, threshold=threshold, fixed_risk=fixed_risk
        )
        quality_params = _apply_scenario_overrides(
            base_params, active=True, comparison_mode=comparison_mode, filter_id=filter_id, threshold=threshold, fixed_risk=fixed_risk
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
            base_mapping, active=False, comparison_mode=comparison_mode, filter_id=filter_id, threshold=threshold, fixed_risk=fixed_risk
        )["static"]
        quality_payload["params_ensemble"] = _rewrite_ensemble_mapping(
            base_mapping, active=True, comparison_mode=comparison_mode, filter_id=filter_id, threshold=threshold, fixed_risk=fixed_risk
        )["static"]
        _assert_controlled_ensemble_pair(no_filter_payload, quality_payload, comparison_mode=comparison_mode)
        policy = get_active_param_ensemble_policy(base_payload)
    elif kind == "rolling_active_param_ensemble":
        for field in ("params_by_effective_date", "params_by_oos_year"):
            raw_mapping = base_payload.get(field)
            if isinstance(raw_mapping, dict) and raw_mapping:
                no_filter_payload[field] = _rewrite_param_mapping(
                    raw_mapping, active=False, comparison_mode=comparison_mode, filter_id=filter_id, threshold=threshold, fixed_risk=fixed_risk
                )
                quality_payload[field] = _rewrite_param_mapping(
                    raw_mapping, active=True, comparison_mode=comparison_mode, filter_id=filter_id, threshold=threshold, fixed_risk=fixed_risk
                )
        ensemble_mapping = base_payload.get("params_ensemble_by_effective_date")
        if not isinstance(ensemble_mapping, dict) or not ensemble_mapping:
            raise ValueError("rolling active-param ensemble 缺少 params_ensemble_by_effective_date")
        no_filter_payload["params_ensemble_by_effective_date"] = _rewrite_ensemble_mapping(
            ensemble_mapping, active=False, comparison_mode=comparison_mode, filter_id=filter_id, threshold=threshold, fixed_risk=fixed_risk
        )
        quality_payload["params_ensemble_by_effective_date"] = _rewrite_ensemble_mapping(
            ensemble_mapping, active=True, comparison_mode=comparison_mode, filter_id=filter_id, threshold=threshold, fixed_risk=fixed_risk
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
                    raw_mapping, active=False, comparison_mode=comparison_mode, filter_id=filter_id, threshold=threshold, fixed_risk=fixed_risk
                )
                quality_payload[field] = _rewrite_param_mapping(
                    raw_mapping, active=True, comparison_mode=comparison_mode, filter_id=filter_id, threshold=threshold, fixed_risk=fixed_risk
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


def _markdown_report(metadata, baseline, quality, delta, yearly) -> str:
    labels = _comparison_labels(str(metadata["comparison_mode"]))
    active_yearly_column = f"{labels['active_name']}_return_pct"
    rows = [
        ("淨總報酬", "total_return_pct", "%"),
        ("最大回撤", "max_drawdown_pct", "%"),
        ("報酬／最大回撤", "return_over_max_drawdown", ""),
        ("年化報酬", "annual_return_pct", "%"),
        ("Log R²", "log_r_squared", ""),
        ("月勝率", "monthly_win_rate_pct", "%"),
        ("交易數", "trade_count", ""),
        ("勝率", "win_rate_pct", "%"),
        ("Payoff", "payoff_ratio", ""),
        ("EV", "expected_value_r", " R"),
        ("平均曝險", "avg_exposure_pct", "%"),
        ("最差完整年度", "min_full_year_return_pct", "%"),
        ("平均每日可掛單候選", "avg_orderable_candidates", ""),
        ("候選供給不足日", "candidate_supply_gap_days", " 日"),
        ("期末未滿倉日", "underfilled_end_days", " 日"),
        ("期末持股缺口總和", "end_position_gap_slot_days", " 格日"),
    ]
    lines = [
        ("# Breakout Quality Score 排序策略經濟效果對照" if metadata["comparison_mode"] == COMPARISON_MODE_SCORE_RANKING else "# Breakout Quality 策略經濟效果對照"), "",
        f"- 期間：`{metadata['comparison_period']['start']}` ～ `{metadata['comparison_period']['end']}`",
        f"- 參數檔：`{metadata['params_path']}`",
        f"- 參數型態：`{metadata['param_source_kind']}`",
        f"- 比較設計：`{metadata['comparison_design']}`",
        f"- 歷史 active-param 無前視：`{metadata['lookahead_safe_active_param_schedule']}`",
        f"- Dataset：`{metadata['dataset']}`",
        f"- Benchmark：`{metadata['benchmark_ticker']}`",
        f"- 唯一差異：`{labels['difference_text']}`",
        (
            f"- Ranking model：`{metadata['filter_id']}` / `{metadata['model_architecture']}` / `{metadata['experiment_profile']}`；"
            "threshold 不作 gate"
            if metadata["comparison_mode"] == COMPARISON_MODE_SCORE_RANKING
            else f"- Filter：`{metadata['filter_id']}` / `{metadata['model_architecture']}` / `{metadata['experiment_profile']}` / threshold `{metadata['threshold']}`"
        ),
        *(
            [f"- 排序鍵：`{' → '.join(metadata.get('score_ranking_order') or [])}`"]
            if metadata["comparison_mode"] == COMPARISON_MODE_SCORE_RANKING
            else []
        ),
        "", "## 主要結果", "", f"| 指標 | No filter | {labels['active_title']} | 差異 |", "|---|---:|---:|---:|",
    ]
    for label, key, unit in rows:
        digits = 0 if key in {"trade_count", "candidate_supply_gap_days", "underfilled_end_days", "end_position_gap_slot_days"} else 4 if key == "log_r_squared" else 2
        lines.append(
            f"| {label} | {_format_metric(baseline.get(key), digits=digits, unit=unit)} "
            f"| {_format_metric(quality.get(key), digits=digits, unit=unit)} "
            f"| {_format_metric(delta.get(key), digits=digits, unit=unit, signed=True)} |"
        )
    lines += ["", "## 年度報酬", ""]
    if yearly.empty:
        lines.append("無年度資料。")
    else:
        lines += [f"| 年度 | No filter | {labels['active_title']} | 差異 | 完整年度 |", "|---:|---:|---:|---:|:---:|"]
        for row in yearly.to_dict("records"):
            lines.append(
                f"| {int(row['year'])} "
                f"| {_format_metric(row.get('no_filter_return_pct'), digits=2, unit='%')} "
                f"| {_format_metric(row.get(active_yearly_column), digits=2, unit='%')} "
                f"| {_format_metric(row.get('delta_pct'), digits=2, unit='%', signed=True)} "
                f"| {'是' if row.get('is_full_year') else '否'} |"
            )
    if not metadata["lookahead_safe_active_param_schedule"]:
        lines += ["", "> 警告：本次使用單一／static 參數，只能視為敏感度診斷，不是無前視 OOS 部署證據。"]
    limitation = (
        "> 本報表是已查看舊 OOS 後的探索性 score-ranking 機制比較；即使改善，也不得直接視為部署證據，需由全新 forward period 驗證。"
        if metadata["comparison_mode"] == COMPARISON_MODE_SCORE_RANKING
        else "> 本報表只驗證目前固定 active 操作點能否改善策略經濟效果；不得依結果回頭調整 threshold、模型或訓練條件。"
    )
    lines += ["", limitation, ""]
    return "\n".join(lines)


def _run_scenario(*, name, data_dir, param_source_kind, params, start_date, end_date, max_positions, enable_rotation, quiet):
    print(f"\n[{name}] 建立市場與訊號快取")
    if param_source_kind == "single_param":
        context = load_portfolio_market_context(str(data_dir), params, verbose=not quiet)
        print(f"[{name}] 執行 {start_date} ～ {end_date}")
        result = run_portfolio_simulation_prepared(
            context["all_dfs_fast"], context["all_trade_logs"], context["sorted_dates"], params,
            max_positions=max_positions, enable_rotation=enable_rotation,
            start_year=pd.Timestamp(start_date).year, start_date=start_date, end_date=end_date,
            benchmark_ticker=PORTFOLIO_DEFAULT_BENCHMARK_TICKER, verbose=not quiet,
            pit_stats_index=context.get("all_pit_stats_index"),
        )
    elif param_source_kind in {"static_active_param_ensemble", "rolling_active_param_ensemble"}:
        print(f"[{name}] 執行 active-param ensemble replay {start_date} ～ {end_date}")
        result = run_portfolio_simulation_with_param_ensemble(
            str(data_dir), params,
            max_positions=max_positions, enable_rotation=enable_rotation,
            start_year=pd.Timestamp(start_date).year, start_date=start_date, end_date=end_date,
            benchmark_ticker=PORTFOLIO_DEFAULT_BENCHMARK_TICKER,
            fixed_risk=None, verbose=not quiet,
        )
    elif param_source_kind == "rolling_oos_param_schedule":
        print(f"[{name}] 執行 rolling active-param replay {start_date} ～ {end_date}")
        result = run_portfolio_simulation_with_param_schedule(
            str(data_dir), params,
            max_positions=max_positions, enable_rotation=enable_rotation,
            start_year=pd.Timestamp(start_date).year, start_date=start_date, end_date=end_date,
            benchmark_ticker=PORTFOLIO_DEFAULT_BENCHMARK_TICKER,
            fixed_risk=None, verbose=not quiet,
        )
    else:
        raise ValueError(f"不支援的參數來源類型: {param_source_kind}")
    return _unpack_result(result)


def run_existing_attribution(*, project_root=PROJECT_ROOT) -> dict[str, Any]:
    root = Path(project_root).resolve()
    comparison_mode = COMPARISON_MODE_HARD_FILTER
    labels = _comparison_labels(comparison_mode)
    filter_id = BREAKOUT_QUALITY_DEFAULT_FILTER_ID
    contract = load_runtime_artifact_contract(str(root), filter_id)
    architecture = str(contract.manifest.get("model_architecture") or "")
    experiment_profile = str(contract.manifest.get("experiment_profile") or "")
    output_dir = resolve_filter_model_output_dir(
        str(root), filter_id, architecture, experiment_profile
    ) / "strategy_compare"
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
    print(f"完成：{output_dir / 'trade_attribution.md'}")
    return attribution


def run_comparison(*, project_root=PROJECT_ROOT, dataset="full", params_path=None, max_positions=10, enable_rotation=False, fixed_risk=None, allow_static_diagnostic=False, comparison_mode=COMPARISON_MODE_HARD_FILTER, quiet=False):
    root = Path(project_root).resolve()
    comparison_mode = str(comparison_mode)
    labels = _comparison_labels(comparison_mode)
    filter_id = BREAKOUT_QUALITY_DEFAULT_FILTER_ID
    try:
        contract = load_runtime_artifact_contract(str(root), filter_id)
    except (FileNotFoundError, ValueError) as exc:
        raise RuntimeError(
            "策略對照需要 active breakout-quality 的正式 forward-OOS scores.csv；"
            "請先執行 `python apps/breakout_quality.py export-scores "
            f"--filter-id {filter_id} --experiment-profile {BREAKOUT_QUALITY_EXPERIMENT_PROFILE} "
            "--scope forward_oos`。"
        ) from exc
    manifest_architecture = str(contract.manifest.get("model_architecture") or "")
    manifest_profile = str(contract.manifest.get("experiment_profile") or "")
    if manifest_architecture != BREAKOUT_QUALITY_MODEL_ARCHITECTURE or manifest_profile != BREAKOUT_QUALITY_EXPERIMENT_PROFILE:
        raise ValueError(
            "正式 runtime artifact 與 active breakout-quality policy 不一致: "
            f"artifact={manifest_architecture}/{manifest_profile}, "
            f"policy={BREAKOUT_QUALITY_MODEL_ARCHITECTURE}/{BREAKOUT_QUALITY_EXPERIMENT_PROFILE}"
        )
    artifact_threshold = float(contract.manifest.get("fixed_evaluation_threshold", float("nan")))
    configured_threshold = float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD)
    if not math.isclose(artifact_threshold, configured_threshold, rel_tol=0.0, abs_tol=0.0):
        raise ValueError(
            "active threshold 與模型 OOS 前固定 threshold 不一致: "
            f"artifact={artifact_threshold}, policy={configured_threshold}"
        )
    if params_path:
        requested_params_path = Path(params_path)
        resolved_params_path = (
            requested_params_path.resolve()
            if requested_params_path.is_absolute()
            else (root / requested_params_path).resolve()
        )
    elif allow_static_diagnostic:
        resolved_params_path = Path(
            resolve_default_primary_param_source_record(str(root))["path"]
        ).resolve()
    else:
        raise ValueError(
            "正式 OOS 策略對照必須以 --params 明確指定 rolling OOS active-param JSON，"
            "例如 models/roos_base_finalists_agree.json；"
            "只有非 OOS 敏感度診斷才可加 --allow-static-diagnostic 使用 run_best_params.json。"
        )
    if fixed_risk is not None and not (0.0 < float(fixed_risk) <= 1.0):
        raise ValueError("fixed_risk 必須介於 0 與 1")
    param_source = _load_param_source(resolved_params_path)
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
        comparison_mode=comparison_mode,
    )

    is_rolling_source = param_source_kind in {"rolling_oos_param_schedule", "rolling_active_param_ensemble"}
    if not is_rolling_source and not allow_static_diagnostic:
        raise ValueError(
            "單一／static 參數跨歷史 OOS 回放無法證明無前視；"
            "請改用 rolling OOS active-param JSON，或明確加 --allow-static-diagnostic 僅作敏感度診斷。"
        )

    data_dir = Path(get_dataset_dir(str(root), dataset)).resolve()
    start_date = contract.available_from.isoformat()
    end_date = contract.available_through.isoformat()
    if param_source_kind == "rolling_oos_param_schedule":
        param_start, param_end = get_active_param_date_range(param_source["payload"])
    elif param_source_kind == "rolling_active_param_ensemble":
        param_start, param_end = get_active_param_ensemble_date_range(param_source["payload"])
    else:
        param_start, param_end = "", ""
    if is_rolling_source and (pd.Timestamp(param_start) > pd.Timestamp(start_date) or pd.Timestamp(param_end) < pd.Timestamp(end_date)):
        raise ValueError(
            "rolling active-param 期間未完整覆蓋 breakout-quality forward OOS："
            f"params={param_start}~{param_end}, filter={start_date}~{end_date}"
        )
    baseline_payload = _run_scenario(name="no_filter", data_dir=data_dir, param_source_kind=param_source_kind, params=no_filter_params, start_date=start_date, end_date=end_date, max_positions=max_positions, enable_rotation=enable_rotation, quiet=quiet)
    quality_payload = _run_scenario(name=labels["active_name"], data_dir=data_dir, param_source_kind=param_source_kind, params=quality_params, start_date=start_date, end_date=end_date, max_positions=max_positions, enable_rotation=enable_rotation, quiet=quiet)
    _assert_shared_benchmark(baseline_payload, quality_payload)

    baseline = _scenario_summary(baseline_payload)
    quality = _scenario_summary(quality_payload)
    deltas = _delta(quality, baseline)
    yearly = _build_yearly_comparison(baseline_payload["profile"], quality_payload["profile"])
    if comparison_mode == COMPARISON_MODE_SCORE_RANKING:
        yearly = yearly.rename(columns={"quality_filter_return_pct": "score_ranking_return_pct"})

    output_dir = resolve_filter_model_output_dir(str(root), filter_id, manifest_architecture, manifest_profile) / labels["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "comparison_mode": comparison_mode,
        "dataset": dataset,
        "data_dir": str(data_dir),
        "params_path": str(resolved_params_path),
        "params_file_sha256": _sha256_file(resolved_params_path),
        "param_source_kind": param_source_kind,
        "comparison_design": "historical_active_param_oos" if is_rolling_source else "static_param_diagnostic",
        "lookahead_safe_active_param_schedule": bool(is_rolling_source),
        "active_param_period": {"start": param_start, "end": param_end} if is_rolling_source else None,
        "active_param_ensemble_policy": ensemble_policy,
        "no_filter_params": no_filter_param_payload,
        f"{labels['active_name']}_params": quality_param_payload,
        "filter_id": filter_id,
        "model_architecture": manifest_architecture,
        "experiment_profile": manifest_profile,
        "threshold": float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD),
        "threshold_used_as_gate": bool(comparison_mode == COMPARISON_MODE_HARD_FILTER),
        "score_ranking_order": (
            ["ensemble_vote_count_desc", "breakout_quality_score_desc", "existing_buy_sort", "ticker_deterministic"]
            if comparison_mode == COMPARISON_MODE_SCORE_RANKING
            else None
        ),
        "unscorable_candidate_policy": "conservative_exclude",
        "max_positions": int(max_positions),
        "enable_rotation": bool(enable_rotation),
        "fixed_risk_override": None if fixed_risk is None else float(fixed_risk),
        "comparison_period": {"start": start_date, "end": end_date},
        "benchmark_ticker": PORTFOLIO_DEFAULT_BENCHMARK_TICKER,
        "runtime_manifest_path": str(contract.paths.manifest_path),
        "runtime_score_path": str(contract.paths.score_path),
        "runtime_eligibility": dict(contract.manifest.get("runtime_eligibility") or {}),
        "score_table": dict(contract.manifest.get("score_table") or {}),
        "controlled_param_difference": [_comparison_switch_spec(comparison_mode)[0]],
        "trade_attribution_schema_version": ATTRIBUTION_SCHEMA_VERSION,
    }
    json_payload = _to_json_native({
        "metadata": metadata,
        "no_filter": baseline,
        labels["active_name"]: quality,
        f"{labels['active_name']}_minus_no_filter": deltas,
        "yearly": yearly.to_dict("records"),
    })
    (output_dir / "strategy_comparison.json").write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "strategy_comparison.md").write_text(_markdown_report(metadata, baseline, quality, deltas, yearly), encoding="utf-8")
    baseline_payload["equity_curve"].to_csv(output_dir / "no_filter_equity.csv", index=False, encoding="utf-8-sig")
    quality_payload["equity_curve"].to_csv(output_dir / f"{labels['active_name']}_equity.csv", index=False, encoding="utf-8-sig")
    baseline_payload["trade_history"].to_csv(output_dir / "no_filter_trades.csv", index=False, encoding="utf-8-sig")
    quality_payload["trade_history"].to_csv(output_dir / f"{labels['active_name']}_trades.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(baseline_payload["profile"].get("portfolio_capacity_rows") or []).to_csv(output_dir / "no_filter_daily_capacity.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(quality_payload["profile"].get("portfolio_capacity_rows") or []).to_csv(output_dir / f"{labels['active_name']}_daily_capacity.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(output_dir / "yearly_returns_comparison.csv", index=False, encoding="utf-8-sig")
    if comparison_mode == COMPARISON_MODE_HARD_FILTER:
        write_trade_attribution_outputs(
            project_root=root,
            output_dir=output_dir,
            filter_id=filter_id,
            threshold=float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD),
            metadata=metadata,
            no_filter_trade_history=baseline_payload["trade_history"],
            quality_filter_trade_history=quality_payload["trade_history"],
            no_filter_closed_trade_rows=(baseline_payload["profile"] or {}).get("closed_trade_rows"),
            quality_filter_closed_trade_rows=(quality_payload["profile"] or {}).get("closed_trade_rows"),
            no_filter_portfolio_total_r=baseline.get("portfolio_total_r"),
            quality_filter_portfolio_total_r=quality.get("portfolio_total_r"),
        )
    print(f"\n完成：{output_dir / 'strategy_comparison.md'}")
    if comparison_mode == COMPARISON_MODE_HARD_FILTER:
        print(f"交易歸因：{output_dir / 'trade_attribution.md'}")
    return json_payload


def main(argv=None):
    args = _parse_args(argv)
    if args.attribution_only:
        if args.comparison_mode != COMPARISON_MODE_HARD_FILTER:
            raise ValueError("--attribution-only 目前只支援 hard-filter 既有歸因")
        run_existing_attribution()
        return 0
    if args.max_positions < 1:
        raise ValueError("max_positions 必須 >= 1")
    run_comparison(
        dataset=args.dataset,
        params_path=args.params,
        max_positions=args.max_positions,
        enable_rotation=args.rotation == "on",
        fixed_risk=args.fixed_risk,
        allow_static_diagnostic=args.allow_static_diagnostic,
        comparison_mode=args.comparison_mode,
        quiet=args.quiet,
    )
    return 0


__all__ = [
    "main", "run_comparison", "run_existing_attribution",
    "_assert_controlled_param_pair", "_assert_controlled_ensemble_pair",
    "_assert_controlled_payload_pair", "_build_controlled_param_source_pair",
    "_load_param_source", "_capacity_summary", "_normalize_yearly_completeness",
    "_to_json_native",
    "COMPARISON_MODE_HARD_FILTER", "COMPARISON_MODE_SCORE_RANKING",
]
