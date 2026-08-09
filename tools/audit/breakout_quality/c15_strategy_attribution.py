"""Read-only attribution for SR-C15 versus configured strategy comparators.

The audit consumes only completed ``outputs/strategy_compare`` artifacts.  It does
not replay the portfolio, rebuild scores, train models, or alter selection logic.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.audit import AUDIT_OUTPUT_ROOT, AuditDefinition
from core.runtime_utils import get_taipei_now
from core.console_report import (
    print_artifact_paths,
    project_relative_display_path,
    render_key_values,
    render_section,
    render_table,
    render_title,
)
from tools.audit.portfolio.score_ranking_capture import build_trade_lifecycle_rows
from tools.audit.sources.strategy_compare import (
    StrategyCompareArmArtifacts,
    resolve_arm_artifacts,
    resolve_strategy_compare_run_selector,
)
from filters.breakout_quality.trade_attribution import reconstruct_round_trips

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_RESULT_SCHEMA_VERSION = 3


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_native(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) is int:
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_native(item) for item in value.tolist()]
    if isinstance(value, pd.Series):
        return [_json_native(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return value
    return None if bool(missing) else value


def _validate_definition(definition: AuditDefinition) -> tuple[str, tuple[str, ...], int, int, int]:
    source = dict(definition.source)
    if str(source.get("kind") or "") != "strategy_compare":
        raise ValueError(f"{definition.audit_id}.source.kind必須是strategy_compare")
    candidate = str(source.get("candidate_arm_id") or "").strip()
    comparators_raw = source.get("comparator_arm_ids")
    if not candidate:
        raise ValueError(f"{definition.audit_id}.candidate_arm_id不可空白")
    if not isinstance(comparators_raw, (list, tuple)) or not comparators_raw:
        raise ValueError(f"{definition.audit_id}.comparator_arm_ids必須是非空list")
    comparators = tuple(str(value).strip() for value in comparators_raw if str(value).strip())
    if not comparators or candidate in comparators or len(set(comparators)) != len(comparators):
        raise ValueError(f"{definition.audit_id}.comparator_arm_ids不合法")
    dimensions = dict(definition.dimensions)
    try:
        focus_year = int(dimensions.get("focus_year"))
        top_month_count = int(dimensions.get("top_month_count"))
        top_trade_count = int(dimensions.get("top_trade_count"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{definition.audit_id}.dimensions必須是整數") from exc
    if focus_year < 1900 or top_month_count < 1 or top_trade_count < 1:
        raise ValueError(f"{definition.audit_id}.dimensions超出合法範圍")
    return candidate, comparators, focus_year, top_month_count, top_trade_count


def _required_paths(artifacts: StrategyCompareArmArtifacts) -> dict[str, Path]:
    return {
        "trades": artifacts.trades_path,
        "equity": artifacts.equity_path,
        "capacity": artifacts.capacity_path,
        "selected": artifacts.selected_path,
    }


def _arm_run_selector(definition: AuditDefinition, arm_id: str) -> dict[str, Any]:
    source = dict(definition.source)
    arm_runs = source.get("arm_runs")
    if arm_runs in (None, {}):
        return source
    if not isinstance(arm_runs, dict):
        raise ValueError(f"{definition.audit_id}.source.arm_runs必須是mapping")
    selector = arm_runs.get(str(arm_id))
    if not isinstance(selector, dict):
        raise ValueError(f"{definition.audit_id}.source.arm_runs缺少arm: {arm_id}")
    return dict(selector)


def _resolve_attribution_run(
    root: Path,
    definition: AuditDefinition,
    arm_id: str,
) -> tuple[Path, dict[str, Any]]:
    return resolve_strategy_compare_run_selector(
        root,
        _arm_run_selector(definition, arm_id),
        audit_id=definition.audit_id,
        required_arm_id=arm_id,
    )


def _strategy_settings_signature(result: dict[str, Any]) -> dict[str, Any]:
    settings = dict(result.get("settings") or {})
    return {
        "dataset": settings.get("dataset"),
        "param_policy": settings.get("param_policy"),
        "max_positions": settings.get("max_positions"),
        "rotation": settings.get("rotation"),
    }


def _result_arm(result: dict[str, Any], arm_id: str) -> dict[str, Any]:
    arms = dict(dict(result.get("settings") or {}).get("arms") or {})
    arm = arms.get(str(arm_id))
    if not isinstance(arm, dict):
        raise ValueError(f"strategy_compare結果不存在arm: {arm_id}")
    return dict(arm)


def _validate_cross_run_pair(
    candidate_result: dict[str, Any],
    comparator_result: dict[str, Any],
    *,
    candidate_arm_id: str,
    comparator_arm_id: str,
) -> None:
    candidate_period = dict(candidate_result.get("comparison_period") or {})
    comparator_period = dict(comparator_result.get("comparison_period") or {})
    if candidate_period != comparator_period:
        raise ValueError(
            f"跨run attribution期間不一致: {candidate_arm_id}={candidate_period}, "
            f"{comparator_arm_id}={comparator_period}"
        )
    candidate_signature = _strategy_settings_signature(candidate_result)
    comparator_signature = _strategy_settings_signature(comparator_result)
    if candidate_signature != comparator_signature:
        raise ValueError(
            f"跨run attribution共用執行設定不一致: "
            f"{candidate_arm_id}={candidate_signature}, {comparator_arm_id}={comparator_signature}"
        )
    candidate_arm = _result_arm(candidate_result, candidate_arm_id)
    comparator_arm = _result_arm(comparator_result, comparator_arm_id)
    for field in ("param_source", "rule_policy", "dl_runtime_mode"):
        if candidate_arm.get(field) != comparator_arm.get(field):
            raise ValueError(
                f"跨run attribution arm契約不一致: field={field}, "
                f"{candidate_arm_id}={candidate_arm.get(field)!r}, "
                f"{comparator_arm_id}={comparator_arm.get(field)!r}"
            )
    param_source = str(candidate_arm.get("param_source") or "").strip()
    if not param_source:
        raise ValueError("跨run attribution缺少param_source")
    candidate_identity = dict(candidate_result.get("artifact_identities") or {}).get(
        f"param:{param_source}"
    )
    comparator_identity = dict(comparator_result.get("artifact_identities") or {}).get(
        f"param:{param_source}"
    )
    if not isinstance(candidate_identity, dict) or not isinstance(comparator_identity, dict):
        raise ValueError("跨run attribution缺少正式策略參數identity")
    candidate_sha = str(candidate_identity.get("sha256") or "").strip()
    comparator_sha = str(comparator_identity.get("sha256") or "").strip()
    if not candidate_sha or not comparator_sha or candidate_sha != comparator_sha:
        raise ValueError(
            f"跨run attribution策略參數工件不一致: "
            f"{candidate_arm_id}={candidate_sha or '-'}, {comparator_arm_id}={comparator_sha or '-'}"
        )


def _resolve_attribution_sources(
    root: Path,
    definition: AuditDefinition,
    candidate: str,
    comparators: tuple[str, ...],
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    run_dirs: dict[str, Path] = {}
    results: dict[str, dict[str, Any]] = {}
    for arm_id in (candidate, *comparators):
        run_dir, result = _resolve_attribution_run(root, definition, arm_id)
        run_dirs[arm_id] = run_dir
        results[arm_id] = result
    candidate_run = run_dirs[candidate]
    for comparator in comparators:
        if run_dirs[comparator] != candidate_run:
            _validate_cross_run_pair(
                results[candidate],
                results[comparator],
                candidate_arm_id=candidate,
                comparator_arm_id=comparator,
            )
    return run_dirs, results


def collect_strategy_attribution_status(
    definition: AuditDefinition,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    try:
        candidate, comparators, _focus_year, _top_months, _top_trades = _validate_definition(definition)
        run_dirs, results = _resolve_attribution_sources(
            root, definition, candidate, comparators
        )
        all_artifacts: dict[str, StrategyCompareArmArtifacts] = {}
        for arm_id in (candidate, *comparators):
            all_artifacts[arm_id] = resolve_arm_artifacts(
                run_dir=run_dirs[arm_id],
                result=results[arm_id],
                arm_id=arm_id,
                preferred_pair_arm_id=candidate,
            )
        missing: list[str] = []
        for arm_id, artifacts in all_artifacts.items():
            for label, path in _required_paths(artifacts).items():
                if not path.is_file():
                    missing.append(f"{arm_id}:{label}")
        status = "READY" if not missing else "BLOCKED"
        reason = "" if not missing else "缺少正式只讀工件: " + ", ".join(missing)
        display_runs = {
            arm_id: project_relative_display_path(path, project_root=root)
            for arm_id, path in run_dirs.items()
        }
        return {
            "audit_id": definition.audit_id,
            "status": status,
            "reason": reason,
            "source": {
                "candidate_arm_id": candidate,
                "comparator_arm_ids": list(comparators),
                "display": f"{candidate} vs {','.join(comparators)}",
                "strategy_compare_runs": display_runs,
                "cross_run": len(set(display_runs.values())) > 1,
            },
        }
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        return {
            "audit_id": definition.audit_id,
            "status": "BLOCKED",
            "reason": str(exc),
            "source": {
                "candidate_arm_id": str(definition.source.get("candidate_arm_id") or ""),
                "comparator_arm_ids": list(definition.source.get("comparator_arm_ids") or []),
                "display": str(definition.source.get("candidate_arm_id") or "-"),
            },
        }


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _normalize_equity(frame: pd.DataFrame, *, arm_id: str) -> pd.DataFrame:
    out = pd.DataFrame(frame).copy()
    if not {"Date", "Equity"}.issubset(out.columns):
        raise ValueError(f"{arm_id} equity缺少Date/Equity欄位")
    out["Date"] = pd.to_datetime(out["Date"], errors="raise")
    out["Equity"] = pd.to_numeric(out["Equity"], errors="raise")
    if not np.isfinite(out["Equity"]).all() or bool((out["Equity"] <= 0).any()):
        raise ValueError(f"{arm_id} equity必須全部為正有限值")
    if out["Date"].duplicated().any():
        raise ValueError(f"{arm_id} equity日期重複")
    return out.sort_values("Date", kind="mergesort").reset_index(drop=True)


def _log_wealth_path(
    candidate_equity: pd.DataFrame,
    comparator_equity: pd.DataFrame,
    *,
    candidate_return_pct: float,
    comparator_return_pct: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    left = _normalize_equity(candidate_equity, arm_id="candidate")
    right = _normalize_equity(comparator_equity, arm_id="comparator")
    if left["Date"].tolist() != right["Date"].tolist():
        raise ValueError("C15 attribution要求兩arm equity交易日期完全一致")
    result = pd.DataFrame({
        "date": left["Date"],
        "candidate_equity": left["Equity"].to_numpy(dtype=float),
        "comparator_equity": right["Equity"].to_numpy(dtype=float),
    })
    candidate_log = np.log(result["candidate_equity"] / float(result["candidate_equity"].iloc[0]))
    comparator_log = np.log(result["comparator_equity"] / float(result["comparator_equity"].iloc[0]))
    result["candidate_daily_log_return"] = candidate_log.diff().fillna(0.0)
    result["comparator_daily_log_return"] = comparator_log.diff().fillna(0.0)
    result["delta_log_wealth"] = result["candidate_daily_log_return"] - result["comparator_daily_log_return"]
    target_delta = math.log1p(candidate_return_pct / 100.0) - math.log1p(comparator_return_pct / 100.0)
    path_delta = float(result["delta_log_wealth"].sum())
    normalization_residual = float(target_delta - path_delta)
    if len(result):
        result.loc[result.index[0], "delta_log_wealth"] += normalization_residual
    result["cumulative_delta_log_wealth"] = result["delta_log_wealth"].cumsum()
    result["month"] = result["date"].dt.strftime("%Y-%m")
    result["year"] = result["date"].dt.year.astype(int)
    return result, {
        "target_delta_log_wealth": float(target_delta),
        "raw_equity_path_delta_log_wealth": float(path_delta),
        "normalization_residual": normalization_residual,
        "final_relative_wealth_advantage_pct": float((math.exp(target_delta) - 1.0) * 100.0),
    }


def _period_contribution(path: pd.DataFrame, key: str) -> pd.DataFrame:
    grouped = path.groupby(key, sort=True, dropna=False)["delta_log_wealth"].sum().reset_index()
    grouped["relative_wealth_effect_pct"] = (np.exp(grouped["delta_log_wealth"]) - 1.0) * 100.0
    return grouped


def _concentration_summary(
    monthly: pd.DataFrame,
    yearly: pd.DataFrame,
    *,
    focus_year: int,
    top_month_count: int,
) -> dict[str, Any]:
    net = float(monthly["delta_log_wealth"].sum()) if not monthly.empty else 0.0
    positive = monthly.loc[monthly["delta_log_wealth"] > 0, "delta_log_wealth"]
    positive_sum = float(positive.sum()) if not positive.empty else 0.0
    top_positive = positive.nlargest(top_month_count) if not positive.empty else positive
    focus_rows = yearly[yearly["year"] == int(focus_year)]
    focus = float(focus_rows["delta_log_wealth"].sum()) if not focus_rows.empty else 0.0
    non_focus = float(net - focus)
    return {
        "net_delta_log_wealth": net,
        "positive_month_delta_log_wealth": positive_sum,
        "top_positive_month_share_pct": (
            float(positive.max() / positive_sum * 100.0) if positive_sum > 0 else None
        ),
        f"top_{top_month_count}_positive_month_share_pct": (
            float(top_positive.sum() / positive_sum * 100.0) if positive_sum > 0 else None
        ),
        "focus_year": int(focus_year),
        "focus_year_delta_log_wealth": focus,
        "focus_year_share_of_net_pct": (
            float(focus / net * 100.0) if not math.isclose(net, 0.0, abs_tol=1e-15) else None
        ),
        "focus_year_relative_wealth_effect_pct": float((math.exp(focus) - 1.0) * 100.0),
        "non_focus_delta_log_wealth": non_focus,
        "non_focus_relative_wealth_effect_pct": float((math.exp(non_focus) - 1.0) * 100.0),
    }


def _normalize_selected(frame: pd.DataFrame, *, arm_id: str) -> pd.DataFrame:
    out = pd.DataFrame(frame).copy()
    required = {"ticker", "trade_date", "signal_date"}
    if not required.issubset(out.columns):
        raise ValueError(f"{arm_id} selected buys缺少欄位: {sorted(required - set(out.columns))}")
    out["ticker"] = out["ticker"].fillna("").astype(str).str.strip()
    for column in ("trade_date", "signal_date"):
        out[column] = pd.to_datetime(out[column], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    return out.drop_duplicates(["trade_date", "ticker", "signal_date"]).sort_values(
        ["trade_date", "ticker", "signal_date"], kind="mergesort"
    ).reset_index(drop=True)


def _selection_day_differences(candidate: pd.DataFrame, comparator: pd.DataFrame) -> pd.DataFrame:
    left = _normalize_selected(candidate, arm_id="candidate")
    right = _normalize_selected(comparator, arm_id="comparator")
    dates = sorted(set(left["trade_date"]) | set(right["trade_date"]))
    rows: list[dict[str, Any]] = []
    for date in dates:
        left_day = left[left["trade_date"] == date]
        right_day = right[right["trade_date"] == date]
        left_set = set(zip(left_day["ticker"], left_day["signal_date"]))
        right_set = set(zip(right_day["ticker"], right_day["signal_date"]))
        candidate_only = sorted(left_set - right_set)
        comparator_only = sorted(right_set - left_set)
        rows.append({
            "trade_date": date,
            "changed": bool(candidate_only or comparator_only),
            "candidate_selected_count": len(left_set),
            "comparator_selected_count": len(right_set),
            "candidate_only_count": len(candidate_only),
            "comparator_only_count": len(comparator_only),
            "candidate_only": ";".join(f"{ticker}@{signal}" for ticker, signal in candidate_only),
            "comparator_only": ";".join(f"{ticker}@{signal}" for ticker, signal in comparator_only),
        })
    return pd.DataFrame(rows)


def _with_match_key(lifecycle: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(lifecycle).copy()
    if out.empty:
        out["match_key"] = pd.Series(dtype=str)
        return out
    keys = ["ticker", "entry_date", "entry_type", "signal_date"]
    out["signal_date"] = out["signal_date"].fillna("").astype(str)
    out["match_occurrence"] = out.groupby(keys, sort=False, dropna=False).cumcount() + 1
    out["match_key"] = (
        out["ticker"].astype(str)
        + "|" + out["entry_date"].astype(str)
        + "|" + out["entry_type"].astype(str)
        + "|" + out["signal_date"].astype(str)
        + "|" + out["match_occurrence"].astype(str)
    )
    return out


def _trade_contributions(
    candidate_trades: pd.DataFrame,
    comparator_trades: pd.DataFrame,
    *,
    candidate_capacity: pd.DataFrame,
    comparator_capacity: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], pd.DataFrame, pd.DataFrame]:
    candidate_lifecycle = _with_match_key(build_trade_lifecycle_rows(
        candidate_trades,
        scenario="candidate",
        daily_capacity=candidate_capacity,
    ))
    comparator_lifecycle = _with_match_key(build_trade_lifecycle_rows(
        comparator_trades,
        scenario="comparator",
        daily_capacity=comparator_capacity,
    ))
    candidate_rt = _with_match_key(
        reconstruct_round_trips(candidate_trades, scenario="candidate")
    ).set_index("match_key", drop=False)
    comparator_rt = _with_match_key(
        reconstruct_round_trips(comparator_trades, scenario="comparator")
    ).set_index("match_key", drop=False)
    keys = sorted(set(candidate_rt.index.astype(str)) | set(comparator_rt.index.astype(str)))
    candidate_life = candidate_lifecycle.set_index("match_key", drop=False) if not candidate_lifecycle.empty else candidate_lifecycle
    comparator_life = comparator_lifecycle.set_index("match_key", drop=False) if not comparator_lifecycle.empty else comparator_lifecycle
    rows: list[dict[str, Any]] = []
    for key in keys:
        has_candidate = key in candidate_rt.index
        has_comparator = key in comparator_rt.index
        c = candidate_rt.loc[key] if has_candidate else None
        b = comparator_rt.loc[key] if has_comparator else None
        cl = candidate_life.loc[key] if not candidate_lifecycle.empty and key in candidate_life.index else None
        bl = comparator_life.loc[key] if not comparator_lifecycle.empty and key in comparator_life.index else None
        entry_date = str((c if c is not None else b).get("entry_date") or "")
        signal_date = str((c if c is not None else b).get("signal_date") or "")
        category = "common" if has_candidate and has_comparator else "candidate_only" if has_candidate else "comparator_only"
        candidate_pnl = float(c.get("pnl", 0.0)) if c is not None else 0.0
        comparator_pnl = float(b.get("pnl", 0.0)) if b is not None else 0.0
        candidate_r = float(c.get("r_multiple", 0.0)) if c is not None else 0.0
        comparator_r = float(b.get("r_multiple", 0.0)) if b is not None else 0.0
        rows.append({
            "match_key": key,
            "category": category,
            "ticker": str((c if c is not None else b).get("ticker") or ""),
            "entry_date": entry_date,
            "signal_date": signal_date,
            "entry_month": entry_date[:7],
            "entry_year": int(entry_date[:4]) if len(entry_date) >= 4 else None,
            "candidate_pnl": candidate_pnl,
            "comparator_pnl": comparator_pnl,
            "pnl_delta": candidate_pnl - comparator_pnl,
            "candidate_r": candidate_r,
            "comparator_r": comparator_r,
            "r_delta": candidate_r - comparator_r,
            "candidate_invested_total": _finite(cl.get("invested_total")) if cl is not None else None,
            "comparator_invested_total": _finite(bl.get("invested_total")) if bl is not None else None,
            "candidate_stop_distance_pct": _finite(cl.get("stop_distance_pct")) if cl is not None else None,
            "comparator_stop_distance_pct": _finite(bl.get("stop_distance_pct")) if bl is not None else None,
            "candidate_capital_return_pct": _finite(cl.get("capital_return_pct")) if cl is not None else None,
            "comparator_capital_return_pct": _finite(bl.get("capital_return_pct")) if bl is not None else None,
            "candidate_holding_days": _finite(cl.get("holding_calendar_days")) if cl is not None else None,
            "comparator_holding_days": _finite(bl.get("holding_calendar_days")) if bl is not None else None,
        })
    frame = pd.DataFrame(rows)
    exclusive = frame[frame["category"] != "common"] if not frame.empty else frame
    selection_r_delta = float(exclusive["r_delta"].sum()) if not exclusive.empty else 0.0
    selection_pnl_delta = float(exclusive["pnl_delta"].sum()) if not exclusive.empty else 0.0
    common_pnl_delta = float(frame.loc[frame["category"] == "common", "pnl_delta"].sum()) if not frame.empty else 0.0
    summary = {
        "candidate_trade_count": int(len(candidate_rt)),
        "comparator_trade_count": int(len(comparator_rt)),
        "common_trade_count": int((frame["category"] == "common").sum()) if not frame.empty else 0,
        "candidate_only_trade_count": int((frame["category"] == "candidate_only").sum()) if not frame.empty else 0,
        "comparator_only_trade_count": int((frame["category"] == "comparator_only").sum()) if not frame.empty else 0,
        "exclusive_selection_delta_r": selection_r_delta,
        "exclusive_selection_delta_pnl": selection_pnl_delta,
        "common_trade_pnl_delta": common_pnl_delta,
        "all_trade_pnl_delta": float(frame["pnl_delta"].sum()) if not frame.empty else 0.0,
    }
    geometry = {
        "candidate": _geometry_summary(candidate_lifecycle),
        "comparator": _geometry_summary(comparator_lifecycle),
    }
    geometry["candidate_minus_comparator"] = _numeric_delta(geometry["candidate"], geometry["comparator"])
    return frame, summary, geometry, candidate_lifecycle, comparator_lifecycle


def _mean(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns or frame.empty:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


def _geometry_summary(lifecycle: pd.DataFrame) -> dict[str, Any]:
    frame = pd.DataFrame(lifecycle)
    return {
        "trade_count": int(len(frame)),
        "avg_reserved_total": _mean(frame, "reserved_total"),
        "avg_invested_total": _mean(frame, "invested_total"),
        "avg_stop_distance_pct": _mean(frame, "stop_distance_pct"),
        "avg_holding_calendar_days": _mean(frame, "holding_calendar_days"),
        "avg_realized_r": _mean(frame, "r_multiple"),
        "avg_capital_return_pct": _mean(frame, "capital_return_pct"),
        "total_pnl": float(pd.to_numeric(frame.get("pnl"), errors="coerce").fillna(0.0).sum()) if not frame.empty else 0.0,
    }


def _numeric_delta(candidate: dict[str, Any], comparator: dict[str, Any]) -> dict[str, float]:
    output: dict[str, float] = {}
    for key in sorted(set(candidate) | set(comparator)):
        left = _finite(comparator.get(key))
        right = _finite(candidate.get(key))
        if left is not None and right is not None:
            output[key] = right - left
    return output


def _normalize_capacity(frame: pd.DataFrame, *, arm_id: str) -> pd.DataFrame:
    out = pd.DataFrame(frame).copy()
    required = {"Date", "Post_Execution_Positions", "End_Position_Gap", "Filled_Buys_Today", "Missed_Buys_Today"}
    if not required.issubset(out.columns):
        raise ValueError(f"{arm_id} capacity缺少欄位: {sorted(required - set(out.columns))}")
    out["Date"] = pd.to_datetime(out["Date"], errors="raise").dt.strftime("%Y-%m-%d")
    return out


def _capacity_attribution(
    candidate: pd.DataFrame,
    comparator: pd.DataFrame,
    selection_days: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    left = _normalize_capacity(candidate, arm_id="candidate").add_prefix("candidate_")
    right = _normalize_capacity(comparator, arm_id="comparator").add_prefix("comparator_")
    merged = left.merge(
        right,
        left_on="candidate_Date",
        right_on="comparator_Date",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError("C15 attribution要求兩arm daily capacity日期完全一致")
    merged.insert(0, "date", merged["candidate_Date"])
    changed_lookup = dict(zip(selection_days.get("trade_date", []), selection_days.get("changed", [])))
    merged["selection_changed"] = merged["date"].map(lambda value: bool(changed_lookup.get(value, False)))
    for column in ("Post_Execution_Positions", "End_Position_Gap", "Filled_Buys_Today", "Missed_Buys_Today"):
        merged[f"delta_{column}"] = (
            pd.to_numeric(merged[f"candidate_{column}"], errors="coerce")
            - pd.to_numeric(merged[f"comparator_{column}"], errors="coerce")
        )
    changed = merged[merged["selection_changed"]]
    summary = {
        "selection_changed_days": int(merged["selection_changed"].sum()),
        "candidate_underfilled_end_days": int((pd.to_numeric(merged["candidate_End_Position_Gap"], errors="coerce") > 0).sum()),
        "comparator_underfilled_end_days": int((pd.to_numeric(merged["comparator_End_Position_Gap"], errors="coerce") > 0).sum()),
        "candidate_end_position_gap_slot_days": int(pd.to_numeric(merged["candidate_End_Position_Gap"], errors="coerce").fillna(0).sum()),
        "comparator_end_position_gap_slot_days": int(pd.to_numeric(merged["comparator_End_Position_Gap"], errors="coerce").fillna(0).sum()),
        "candidate_avg_end_positions": _mean(merged.rename(columns={"candidate_Post_Execution_Positions": "v"}), "v"),
        "comparator_avg_end_positions": _mean(merged.rename(columns={"comparator_Post_Execution_Positions": "v"}), "v"),
        "changed_days_avg_position_delta": _mean(changed.rename(columns={"delta_Post_Execution_Positions": "v"}), "v"),
        "changed_days_avg_gap_delta": _mean(changed.rename(columns={"delta_End_Position_Gap": "v"}), "v"),
        "changed_days_filled_buy_delta": float(pd.to_numeric(changed.get("delta_Filled_Buys_Today"), errors="coerce").fillna(0).sum()) if not changed.empty else 0.0,
        "changed_days_missed_buy_delta": float(pd.to_numeric(changed.get("delta_Missed_Buys_Today"), errors="coerce").fillna(0).sum()) if not changed.empty else 0.0,
    }
    def append_resource_summary(side: str) -> None:
        prefix = f"{side}_"
        mode_col = prefix + "Resource_Aware_Mode"
        if mode_col not in merged.columns:
            return
        mode = merged[mode_col].fillna("inactive").astype(str)
        summary[f"{side}_resource_aware_dl_selection_days"] = int((mode == "dl-selection").sum())
        summary[f"{side}_resource_aware_capital_utilization_days"] = int((mode == "capital-utilization").sum())

        changed_col = prefix + "Resource_Aware_Changed"
        if changed_col in merged.columns:
            summary[f"{side}_resource_aware_changed_days"] = int(
                merged[changed_col].fillna(False).astype(bool).sum()
            )

        selected_col = prefix + "Resource_Aware_Selected"
        baseline_selected_col = prefix + "Resource_Aware_Baseline_Selected"
        if selected_col in merged.columns and baseline_selected_col in merged.columns:
            selected = pd.to_numeric(merged[selected_col], errors="coerce").fillna(0)
            baseline = pd.to_numeric(merged[baseline_selected_col], errors="coerce").fillna(0)
            summary[f"{side}_resource_aware_selected_order_delta"] = int((selected - baseline).sum())

        reserved_col = prefix + "Resource_Aware_Reserved_Milli"
        baseline_reserved_col = prefix + "Resource_Aware_Baseline_Reserved_Milli"
        if reserved_col in merged.columns and baseline_reserved_col in merged.columns:
            summary[f"{side}_resource_aware_reserved_delta_milli"] = int(
                (
                    pd.to_numeric(merged[reserved_col], errors="coerce").fillna(0)
                    - pd.to_numeric(merged[baseline_reserved_col], errors="coerce").fillna(0)
                ).sum()
            )

        score_col = prefix + "Resource_Aware_Score_Sum"
        baseline_score_col = prefix + "Resource_Aware_Baseline_Score_Sum"
        if score_col in merged.columns and baseline_score_col in merged.columns:
            summary[f"{side}_resource_aware_score_sum_gain"] = float(
                (
                    pd.to_numeric(merged[score_col], errors="coerce").fillna(0.0)
                    - pd.to_numeric(merged[baseline_score_col], errors="coerce").fillna(0.0)
                ).sum()
            )

        promoted_col = prefix + "Resource_Aware_Promoted_Score_Orders"
        if promoted_col in merged.columns:
            summary[f"{side}_resource_aware_promoted_score_orders"] = int(
                pd.to_numeric(merged[promoted_col], errors="coerce").fillna(0).sum()
            )

        feasible_col = prefix + "Resource_Aware_Direct_Score_Order_Feasible"
        if feasible_col in merged.columns:
            summary[f"{side}_resource_aware_direct_score_order_days"] = int(
                merged[feasible_col].fillna(False).astype(bool).sum()
            )

    append_resource_summary("candidate")
    append_resource_summary("comparator")
    return merged, summary


def build_strategy_attribution_pair_payload(
    *,
    candidate_artifacts: StrategyCompareArmArtifacts,
    comparator_artifacts: StrategyCompareArmArtifacts,
    focus_year: int,
    top_month_count: int,
    top_trade_count: int,
) -> dict[str, Any]:
    candidate_trades = _read_csv(candidate_artifacts.trades_path)
    comparator_trades = _read_csv(comparator_artifacts.trades_path)
    candidate_equity = _read_csv(candidate_artifacts.equity_path)
    comparator_equity = _read_csv(comparator_artifacts.equity_path)
    candidate_capacity = _read_csv(candidate_artifacts.capacity_path)
    comparator_capacity = _read_csv(comparator_artifacts.capacity_path)
    candidate_selected = _read_csv(candidate_artifacts.selected_path)
    comparator_selected = _read_csv(comparator_artifacts.selected_path)

    candidate_return = _finite(candidate_artifacts.summary.get("total_return_pct"))
    comparator_return = _finite(comparator_artifacts.summary.get("total_return_pct"))
    if candidate_return is None or comparator_return is None:
        raise ValueError("strategy_compare scenario summary缺少total_return_pct")
    path, path_summary = _log_wealth_path(
        candidate_equity,
        comparator_equity,
        candidate_return_pct=candidate_return,
        comparator_return_pct=comparator_return,
    )
    monthly = _period_contribution(path, "month")
    yearly = _period_contribution(path, "year")
    concentration = _concentration_summary(
        monthly,
        yearly,
        focus_year=focus_year,
        top_month_count=top_month_count,
    )
    selection_days = _selection_day_differences(candidate_selected, comparator_selected)
    trade_rows, trade_summary, geometry, candidate_lifecycle, comparator_lifecycle = _trade_contributions(
        candidate_trades,
        comparator_trades,
        candidate_capacity=candidate_capacity,
        comparator_capacity=comparator_capacity,
    )
    capacity_rows, capacity_summary = _capacity_attribution(
        candidate_capacity,
        comparator_capacity,
        selection_days,
    )
    top_months = (
        monthly.loc[monthly["delta_log_wealth"] > 0]
        .sort_values("delta_log_wealth", ascending=False)
        .head(top_month_count)
    )
    top_trades = trade_rows.reindex(trade_rows["pnl_delta"].abs().sort_values(ascending=False).index).head(top_trade_count) if not trade_rows.empty else trade_rows
    summary_delta = _numeric_delta(candidate_artifacts.summary, comparator_artifacts.summary)
    interpretation = {
        "candidate_total_return_higher": candidate_return > comparator_return,
        "candidate_avg_realized_r_higher": (
            (_finite(geometry["candidate"].get("avg_realized_r")) or 0.0)
            > (_finite(geometry["comparator"].get("avg_realized_r")) or 0.0)
        ),
        "candidate_avg_capital_return_higher": (
            (_finite(geometry["candidate"].get("avg_capital_return_pct")) or 0.0)
            > (_finite(geometry["comparator"].get("avg_capital_return_pct")) or 0.0)
        ),
        "candidate_avg_invested_per_trade_higher": (
            (_finite(geometry["candidate"].get("avg_invested_total")) or 0.0)
            > (_finite(geometry["comparator"].get("avg_invested_total")) or 0.0)
        ),
        "candidate_position_gap_lower": (
            int(capacity_summary["candidate_end_position_gap_slot_days"])
            < int(capacity_summary["comparator_end_position_gap_slot_days"])
        ),
        "portfolio_advantage_not_explained_by_avg_r_alone": (
            candidate_return > comparator_return
            and not (
                (_finite(geometry["candidate"].get("avg_realized_r")) or 0.0)
                > (_finite(geometry["comparator"].get("avg_realized_r")) or 0.0)
            )
        ),
    }
    return {
        "candidate_arm_id": candidate_artifacts.arm_id,
        "comparator_arm_id": comparator_artifacts.arm_id,
        "candidate_summary": dict(candidate_artifacts.summary),
        "comparator_summary": dict(comparator_artifacts.summary),
        "candidate_minus_comparator": summary_delta,
        "wealth_path": path_summary,
        "concentration": concentration,
        "selection": {
            "calendar_days_with_selection": int(len(selection_days)),
            "changed_days": int(selection_days["changed"].sum()) if not selection_days.empty else 0,
            "candidate_only_selected_orders": int(selection_days["candidate_only_count"].sum()) if not selection_days.empty else 0,
            "comparator_only_selected_orders": int(selection_days["comparator_only_count"].sum()) if not selection_days.empty else 0,
        },
        "trade_contribution": trade_summary,
        "capital_geometry": geometry,
        "slot_occupancy": capacity_summary,
        "interpretation": interpretation,
        "top_positive_months": top_months.to_dict("records"),
        "top_absolute_trade_contributions": top_trades.to_dict("records"),
        "frames": {
            "daily_log_wealth": path,
            "monthly_log_wealth": monthly,
            "yearly_log_wealth": yearly,
            "selection_days": selection_days,
            "trade_contributions": trade_rows,
            "capacity_days": capacity_rows,
            "candidate_lifecycle": candidate_lifecycle,
            "comparator_lifecycle": comparator_lifecycle,
        },
    }


def _fmt(value: Any, digits: int = 2, unit: str = "", signed: bool = False) -> str:
    number = _finite(value)
    if number is None:
        return "N/A"
    sign = "+" if signed else ""
    return f"{number:{sign}.{digits}f}{unit}"


def _fmt_milli_money(value: Any, *, signed: bool = False) -> str:
    number = _finite(value)
    if number is None:
        return "N/A"
    return _fmt(number / 1000.0, digits=0, signed=signed)


def _render_report(payload: dict[str, Any]) -> str:
    metadata = payload["metadata"]
    source_rows = [
        ("Audit", f"AUD-{payload['audit_id']}"),
        ("策略比較期間", f"{metadata['comparison_period'].get('start', '')} ～ {metadata['comparison_period'].get('end', '')}"),
        ("Candidate", metadata["candidate_arm_id"]),
        ("Comparators", ", ".join(metadata["comparator_arm_ids"])),
        ("Focus year", metadata["focus_year"]),
    ]
    run_map = dict(metadata.get("strategy_compare_runs") or {})
    if bool(metadata.get("cross_run")):
        for arm_id in (metadata["candidate_arm_id"], *metadata["comparator_arm_ids"]):
            source_rows.append((f"Strategy compare {arm_id}", run_map.get(arm_id, "-")))
    else:
        source_rows.append(("Strategy compare", run_map.get(metadata["candidate_arm_id"], metadata.get("strategy_compare_run", "-"))))
    source_rows.append(("契約", "只讀既有replay；不重跑、不改score／selector／training"))
    lines = [
        render_title("C15 Read-only Strategy Attribution"),
        render_key_values(tuple(source_rows)),
    ]
    if payload["comparisons"]:
        selector = payload["comparisons"][0]["slot_occupancy"]
        if "candidate_resource_aware_dl_selection_days" in selector:
            lines.extend((
                render_section("SR-C15 selector自身盤前診斷"),
                render_table(
                    ("指標", "結果"),
                    (
                        ("DL-selection days", str(selector.get("candidate_resource_aware_dl_selection_days", 0))),
                        ("Capital-utilization days", str(selector.get("candidate_resource_aware_capital_utilization_days", 0))),
                        ("Selector changed days", str(selector.get("candidate_resource_aware_changed_days", 0))),
                        ("相對Min ROOS預計選入單數差", _fmt(selector.get("candidate_resource_aware_selected_order_delta"), digits=0, signed=True)),
                        ("相對Min ROOS預留資金差", _fmt_milli_money(selector.get("candidate_resource_aware_reserved_delta_milli"), signed=True)),
                        ("Selected score sum gain", _fmt(selector.get("candidate_resource_aware_score_sum_gain"), digits=4, signed=True)),
                        ("Promoted score orders", str(selector.get("candidate_resource_aware_promoted_score_orders", 0))),
                        ("Direct score-order feasible days", str(selector.get("candidate_resource_aware_direct_score_order_days", 0))),
                    ),
                ),
                "此表只比較C15 selector與同日Min ROOS盤前baseline；不使用隔日成交或未來資訊。",
            ))
    for index, pair in enumerate(payload["comparisons"], start=1):
        cand = pair["candidate_arm_id"]
        comp = pair["comparator_arm_id"]
        geom = pair["capital_geometry"]
        gap = pair["slot_occupancy"]
        conc = pair["concentration"]
        trade = pair["trade_contribution"]
        delta = pair["candidate_minus_comparator"]
        lines.extend((
            render_section(f"{index}. {cand} vs {comp}"),
            render_table(
                ("指標", comp, cand, "差異"),
                (
                    ("總報酬", _fmt(pair["comparator_summary"].get("total_return_pct"), unit="%"), _fmt(pair["candidate_summary"].get("total_return_pct"), unit="%"), _fmt(delta.get("total_return_pct"), unit="pp", signed=True)),
                    ("MDD", _fmt(pair["comparator_summary"].get("max_drawdown_pct"), unit="%"), _fmt(pair["candidate_summary"].get("max_drawdown_pct"), unit="%"), _fmt(delta.get("max_drawdown_pct"), unit="pp", signed=True)),
                    ("RoMD", _fmt(pair["comparator_summary"].get("return_over_max_drawdown")), _fmt(pair["candidate_summary"].get("return_over_max_drawdown")), _fmt(delta.get("return_over_max_drawdown"), signed=True)),
                    ("EV", _fmt(pair["comparator_summary"].get("expected_value_r"), unit=" R"), _fmt(pair["candidate_summary"].get("expected_value_r"), unit=" R"), _fmt(delta.get("expected_value_r"), unit=" R", signed=True)),
                    ("平均曝險", _fmt(pair["comparator_summary"].get("avg_exposure_pct"), unit="%"), _fmt(pair["candidate_summary"].get("avg_exposure_pct"), unit="%"), _fmt(delta.get("avg_exposure_pct"), unit="pp", signed=True)),
                ),
            ),
            "Wealth path：Δlog wealth逐日累加，並以正式total return校正首日normalization residual，使全期總和精確對應兩arm最終wealth ratio。",
            render_table(
                ("集中度", "結果"),
                (
                    ("最終相對wealth優勢", _fmt(pair["wealth_path"].get("final_relative_wealth_advantage_pct"), unit="%", signed=True)),
                    (f"{metadata['focus_year']} Δlog wealth占全期淨差異", _fmt(conc.get("focus_year_share_of_net_pct"), unit="%")),
                    (f"{metadata['focus_year']} 單獨相對wealth effect", _fmt(conc.get("focus_year_relative_wealth_effect_pct"), unit="%", signed=True)),
                    (f"非{metadata['focus_year']}期間相對wealth effect", _fmt(conc.get("non_focus_relative_wealth_effect_pct"), unit="%", signed=True)),
                    ("最大正貢獻月份占全部正貢獻", _fmt(conc.get("top_positive_month_share_pct"), unit="%")),
                    (f"Top {metadata['top_month_count']}正貢獻月份占全部正貢獻", _fmt(conc.get(f"top_{metadata['top_month_count']}_positive_month_share_pct"), unit="%")),
                ),
            ),
            render_table(
                ("Selection／trade", "結果"),
                (
                    ("選股不同日", str(pair["selection"]["changed_days"])),
                    ("C15-only選入單", str(pair["selection"]["candidate_only_selected_orders"])),
                    (f"{comp}-only選入單", str(pair["selection"]["comparator_only_selected_orders"])),
                    ("Exclusive selection ΔR", _fmt(trade.get("exclusive_selection_delta_r"), unit=" R", signed=True)),
                    ("Exclusive selection ΔPnL", _fmt(trade.get("exclusive_selection_delta_pnl"), signed=True)),
                    ("Common trades ΔPnL", _fmt(trade.get("common_trade_pnl_delta"), signed=True)),
                    ("All trade ΔPnL", _fmt(trade.get("all_trade_pnl_delta"), signed=True)),
                ),
            ),
            render_table(
                ("Capital geometry", comp, cand, "差異"),
                tuple(
                    (
                        label,
                        _fmt(geom["comparator"].get(key), digits=digits, unit=unit),
                        _fmt(geom["candidate"].get(key), digits=digits, unit=unit),
                        _fmt(geom["candidate_minus_comparator"].get(key), digits=digits, unit=unit, signed=True),
                    )
                    for label, key, digits, unit in (
                        ("平均實際投入", "avg_invested_total", 0, ""),
                        ("平均預留金額", "avg_reserved_total", 0, ""),
                        ("平均停損距離", "avg_stop_distance_pct", 2, "%"),
                        ("平均Capital Return", "avg_capital_return_pct", 2, "%"),
                        ("平均Realized R", "avg_realized_r", 2, " R"),
                        ("平均持有日", "avg_holding_calendar_days", 2, " 日"),
                    )
                ),
            ),
            render_table(
                ("Slot occupancy", comp, cand, "差異"),
                (
                    ("Underfilled end days", str(gap["comparator_underfilled_end_days"]), str(gap["candidate_underfilled_end_days"]), str(gap["candidate_underfilled_end_days"] - gap["comparator_underfilled_end_days"])),
                    ("Position gap slot-days", str(gap["comparator_end_position_gap_slot_days"]), str(gap["candidate_end_position_gap_slot_days"]), str(gap["candidate_end_position_gap_slot_days"] - gap["comparator_end_position_gap_slot_days"])),
                    ("Changed days平均持股差", "-", "-", _fmt(gap.get("changed_days_avg_position_delta"), signed=True)),
                    ("Changed days成交買單差", "-", "-", _fmt(gap.get("changed_days_filled_buy_delta"), digits=0, signed=True)),
                    ("Changed days錯失買單差", "-", "-", _fmt(gap.get("changed_days_missed_buy_delta"), digits=0, signed=True)),
                ),
            ),
        ))
        if "comparator_resource_aware_dl_selection_days" in gap:
            lines.append(
                render_table(
                    ("Selector盤前資源診斷", comp, cand),
                    (
                        ("DL-selection days", str(gap.get("comparator_resource_aware_dl_selection_days", 0)), str(gap.get("candidate_resource_aware_dl_selection_days", 0))),
                        ("Capital-utilization days", str(gap.get("comparator_resource_aware_capital_utilization_days", 0)), str(gap.get("candidate_resource_aware_capital_utilization_days", 0))),
                        ("Selector changed days", str(gap.get("comparator_resource_aware_changed_days", 0)), str(gap.get("candidate_resource_aware_changed_days", 0))),
                        ("相對各自Min ROOS預計選入單數差", _fmt(gap.get("comparator_resource_aware_selected_order_delta"), digits=0, signed=True), _fmt(gap.get("candidate_resource_aware_selected_order_delta"), digits=0, signed=True)),
                        ("相對各自Min ROOS預留資金差", _fmt_milli_money(gap.get("comparator_resource_aware_reserved_delta_milli"), signed=True), _fmt_milli_money(gap.get("candidate_resource_aware_reserved_delta_milli"), signed=True)),
                        ("Promoted score orders", str(gap.get("comparator_resource_aware_promoted_score_orders", 0)), str(gap.get("candidate_resource_aware_promoted_score_orders", 0))),
                        ("Direct score-order feasible days", str(gap.get("comparator_resource_aware_direct_score_order_days", 0)), str(gap.get("candidate_resource_aware_direct_score_order_days", 0))),
                    ),
                )
            )
            lines.append(
                "兩模型的score數值尺度不直接互相比較；此表只比較各自相對同日Min ROOS造成的盤前資源與selector行為。"
            )
        top_month_rows = [
            (row["month"], _fmt(row["delta_log_wealth"], digits=5, signed=True), _fmt(row["relative_wealth_effect_pct"], unit="%", signed=True))
            for row in pair["top_positive_months"]
        ]
        lines.append(render_table(("Top月份", "Δlog wealth", "相對wealth effect"), top_month_rows))
        interp = pair["interpretation"]
        if interp.get("portfolio_advantage_not_explained_by_avg_r_alone"):
            lines.append("判讀：Candidate總報酬較高，但平均Realized R沒有同步提高；portfolio優勢不能只用per-trade R解釋，需同時看capital return、slot occupancy、position sizing與compounding path。")
        else:
            lines.append("判讀：本比較的portfolio結果與per-trade R方向沒有出現『總報酬較高但平均R較低』的矛盾。")
    lines.extend((
        render_section(f"{len(payload['comparisons']) + 1}. 使用限制"),
        "本Audit只使用已完成strategy compare工件。Δlog wealth是portfolio path的精確相對wealth歸因；trade PnL／R分解是交易層診斷，因position sizing與compounding不同，不要求其算術加總等於最終報酬差。",
        "選股不同日只依盤前selected-buy集合比較；後續持倉延續造成的wealth差異會落在之後日期，因此不可把同日equity變化直接當作該日selection的因果效果。",
        "Selector自身盤前診斷比較C15重排結果與同日Min ROOS baseline；其selected／reserved差異是事前資源幾何，不代表事後一定成交或獲利。",
        "本結果不得回流score cutoff、blend weight、年份/regime gate、Target係數或模型訓練。",
    ))
    return "\n\n".join(lines).rstrip() + "\n"


def run_strategy_attribution_audit(
    definition: AuditDefinition,
    *,
    project_root: Path = PROJECT_ROOT,
    quiet: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    candidate, comparators, focus_year, top_month_count, top_trade_count = _validate_definition(definition)
    run_dirs, results = _resolve_attribution_sources(
        root, definition, candidate, comparators
    )
    period = dict(results[candidate].get("comparison_period") or {})
    candidate_artifacts = resolve_arm_artifacts(
        run_dir=run_dirs[candidate],
        result=results[candidate],
        arm_id=candidate,
        preferred_pair_arm_id=candidate,
    )
    comparisons: list[dict[str, Any]] = []
    for comparator in comparators:
        comparator_artifacts = resolve_arm_artifacts(
            run_dir=run_dirs[comparator],
            result=results[comparator],
            arm_id=comparator,
            preferred_pair_arm_id=candidate,
        )
        comparisons.append(build_strategy_attribution_pair_payload(
            candidate_artifacts=candidate_artifacts,
            comparator_artifacts=comparator_artifacts,
            focus_year=focus_year,
            top_month_count=top_month_count,
            top_trade_count=top_trade_count,
        ))

    display_runs = {
        arm_id: project_relative_display_path(path, project_root=root)
        for arm_id, path in run_dirs.items()
    }
    fingerprints = {
        arm_id: results[arm_id].get("config_fingerprint")
        for arm_id in run_dirs
    }
    payload = {
        "schema_version": AUDIT_RESULT_SCHEMA_VERSION,
        "created_at": get_taipei_now().isoformat(),
        "audit_id": definition.audit_id,
        "audit_type": definition.audit_type,
        "config": definition.as_dict(),
        "metadata": {
            "strategy_compare_run": display_runs[candidate],
            "strategy_compare_runs": display_runs,
            "strategy_compare_config_fingerprint": fingerprints[candidate],
            "strategy_compare_config_fingerprints": fingerprints,
            "cross_run": len(set(display_runs.values())) > 1,
            "comparison_period": period,
            "candidate_arm_id": candidate,
            "comparator_arm_ids": list(comparators),
            "focus_year": focus_year,
            "top_month_count": top_month_count,
            "top_trade_count": top_trade_count,
            "read_only": True,
            "portfolio_replay_executed": False,
            "training_performed": False,
            "future_target_used_for_runtime": False,
        },
        "comparisons": [],
    }
    frames_by_pair: dict[str, dict[str, pd.DataFrame]] = {}
    for pair in comparisons:
        frames = pair.pop("frames")
        pair_key = f"{pair['candidate_arm_id']}_vs_{pair['comparator_arm_id']}"
        frames_by_pair[pair_key] = frames
        payload["comparisons"].append(pair)
    payload = _json_native(payload)

    output_root = root / Path(AUDIT_OUTPUT_ROOT) / Path(definition.output_subdir)
    timestamp = get_taipei_now().strftime("%Y%m%d_%H%M%S_%f")
    audit_run_dir = output_root / "runs" / timestamp
    latest_dir = output_root / "latest"
    audit_run_dir.mkdir(parents=True, exist_ok=False)
    report_path = audit_run_dir / "audit.md"
    json_path = audit_run_dir / "audit.json"
    report_path.write_text(_render_report(payload), encoding="utf-8")
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    artifact_paths: list[Path] = [report_path, json_path]
    for pair_key, frames in frames_by_pair.items():
        for frame_name, frame in frames.items():
            path = audit_run_dir / f"{pair_key}_{frame_name}.csv"
            pd.DataFrame(frame).to_csv(path, index=False, encoding="utf-8-sig")
            artifact_paths.append(path)

    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    latest_dir.mkdir(parents=True, exist_ok=True)
    for source in artifact_paths:
        shutil.copy2(source, latest_dir / source.name)

    if not quiet:
        print("\n" + _render_report(payload))
        print_artifact_paths(
            (
                ("Audit Markdown", report_path),
                ("Audit JSON", json_path),
                ("最新Audit", latest_dir),
            ),
            project_root=root,
        )
    return payload


__all__ = [
    "AUDIT_RESULT_SCHEMA_VERSION",
    "collect_strategy_attribution_status",
    "run_strategy_attribution_audit",
    "build_strategy_attribution_pair_payload",
]
