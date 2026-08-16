"""Read-only alignment Audit for a Min-ROOS planned-risk 40D opportunity target.

This is deliberately an Audit, not a Label builder.  It reuses an already
completed Selection Strategy Compare pair and canonical OHLCV to answer one
question before any new MR experiment is allowed: does normalizing the same
entry-date 40D opportunity path by the candidate's *planned* entry/initial-stop
risk improve alignment with realized net R beyond a same-date fixed-risk
control?
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.audit import AUDIT_OUTPUT_ROOT, AuditDefinition
from config.strategy_compare import get_strategy_comparison_settings, get_strategy_runtime_integration_settings
from core.console_report import project_relative_display_path, render_key_values, render_table, render_title
from core.dataset_profiles import get_dataset_dir
from filters.breakout_quality.contract import DEFAULT_LABEL_POLICY
from filters.breakout_quality.continuous_target import StrategyAlignedContinuousTargetSpec
from filters.breakout_quality.strategy_comparison import collect_artifact_status
from filters.breakout_quality.trade_attribution import reconstruct_round_trips
from filters.breakout_quality.workflow_io import discover_dataset_csv_inputs, load_dataset_frame

AUDIT_SCHEMA_VERSION = 1
_REQUIRED_PAIR_FILES = (
    "strategy_comparison.json",
    "score_ranking_execution.csv",
    "score_ranking_trades.csv",
    "score_ranking_selected_target_diagnostics.csv",
)


@dataclass(frozen=True)
class OpportunityTargetResult:
    valid: bool
    reason: str
    target_r: float | None
    favorable_r: float | None
    adverse_r: float | None
    opportunity_bar: int | None
    first_risk_breach_bar: int | None


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _date_text(value: Any) -> str:
    if value in (None, ""):
        return ""
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return ""
    return pd.Timestamp(parsed).strftime("%Y-%m-%d")


def _bool_series(values: pd.Series) -> pd.Series:
    series = pd.Series(values).copy()
    if series.dtype == bool:
        return series.fillna(False).astype(bool)
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )


def _opportunity_target_from_prices(
    highs: np.ndarray,
    lows: np.ndarray,
    *,
    anchor_price: float,
    risk_barrier_price: float,
    horizon_bars: int,
) -> OpportunityTargetResult:
    """Conservative adverse-first 40D opportunity target in candidate risk units."""

    anchor = _finite(anchor_price)
    barrier = _finite(risk_barrier_price)
    horizon = int(horizon_bars)
    if anchor is None or anchor <= 0.0:
        return OpportunityTargetResult(False, "invalid_anchor", None, None, None, None, None)
    if barrier is None or barrier <= 0.0 or barrier >= anchor:
        return OpportunityTargetResult(False, "invalid_risk_barrier", None, None, None, None, None)
    if horizon < 2:
        return OpportunityTargetResult(False, "invalid_horizon", None, None, None, None, None)

    high_arr = np.asarray(highs, dtype=np.float64)[:horizon]
    low_arr = np.asarray(lows, dtype=np.float64)[:horizon]
    if high_arr.size < horizon or low_arr.size < horizon:
        return OpportunityTargetResult(False, "insufficient_future", None, None, None, None, None)
    valid = (
        np.isfinite(high_arr)
        & np.isfinite(low_arr)
        & (high_arr > 0.0)
        & (low_arr > 0.0)
        & (high_arr >= low_arr)
    )
    if not bool(np.all(valid)):
        return OpportunityTargetResult(False, "invalid_future_bar", None, None, None, None, None)

    risk_distance = float(anchor - barrier)
    running_low = float(anchor)
    best_high = -math.inf
    best_adverse_price = 0.0
    best_bar = -1
    first_breach = -1

    for bar_offset, (high_value, low_value) in enumerate(zip(high_arr, low_arr), start=1):
        high_price = float(high_value)
        low_price = float(low_value)
        # Same-bar path is unknowable from OHLC.  Keep the project's conservative
        # adverse-first convention: a barrier touch consumes the full planned risk
        # and the same bar's high is not credited.
        if low_price <= barrier:
            first_breach = int(bar_offset)
            break
        running_low = min(running_low, low_price)
        if high_price > best_high:
            best_high = high_price
            best_adverse_price = max(0.0, anchor - running_low)
            best_bar = int(bar_offset)

    if best_bar < 1:
        favorable_r = 0.0
        adverse_r = 1.0
        target_r = -1.0
        opportunity_bar = int(first_breach if first_breach > 0 else 1)
    else:
        favorable_r = float((best_high - anchor) / risk_distance)
        adverse_r = float(best_adverse_price / risk_distance)
        target_r = float(favorable_r - adverse_r)
        opportunity_bar = int(best_bar)

    if not all(math.isfinite(value) for value in (favorable_r, adverse_r, target_r)):
        return OpportunityTargetResult(False, "non_finite_target", None, None, None, None, None)
    return OpportunityTargetResult(
        True,
        "ok",
        target_r,
        favorable_r,
        adverse_r,
        opportunity_bar,
        int(first_breach),
    )


def _future_path_after_entry(frame: pd.DataFrame, entry_date: str, horizon_bars: int) -> tuple[np.ndarray, np.ndarray] | None:
    if frame.empty:
        return None
    index = pd.to_datetime(frame.index, errors="coerce")
    target_date = pd.Timestamp(entry_date)
    positions = np.flatnonzero(index == target_date)
    if len(positions) != 1:
        return None
    start = int(positions[0]) + 1
    end = start + int(horizon_bars)
    if end > len(frame):
        return None
    window = frame.iloc[start:end]
    return (
        window["High"].to_numpy(dtype=np.float64, copy=False),
        window["Low"].to_numpy(dtype=np.float64, copy=False),
    )


def _rank_corr(x: pd.Series, y: pd.Series) -> float | None:
    pair = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(pair) < 2 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return None
    value = pair["x"].rank(method="average").corr(pair["y"].rank(method="average"), method="pearson")
    return None if pd.isna(value) else float(value)


def _pair_concordance(x: pd.Series, y: pd.Series) -> float | None:
    pair = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(pair) < 2:
        return None
    xv = pair["x"].to_numpy(dtype=float)
    yv = pair["y"].to_numpy(dtype=float)
    concordant = comparable = 0
    for left in range(len(pair) - 1):
        dx = xv[left + 1 :] - xv[left]
        dy = yv[left + 1 :] - yv[left]
        mask = (dx != 0.0) & (dy != 0.0)
        comparable += int(mask.sum())
        if bool(mask.any()):
            concordant += int(((dx[mask] * dy[mask]) > 0.0).sum())
    return None if comparable == 0 else float(concordant / comparable)


def _top_bottom_realized_spread(frame: pd.DataFrame, target_column: str) -> float | None:
    work = frame[[target_column, "realized_r"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(work) < 10 or work[target_column].nunique() < 2:
        return None
    count = max(1, int(math.floor(len(work) * 0.10)))
    top = work.nlargest(count, target_column, keep="first")["realized_r"]
    bottom = work.nsmallest(count, target_column, keep="first")["realized_r"]
    return float(top.mean() - bottom.mean())


def _target_summary(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    work = frame[[column, "realized_r"]].copy()
    work[column] = pd.to_numeric(work[column], errors="coerce")
    work["realized_r"] = pd.to_numeric(work["realized_r"], errors="coerce")
    work = work.dropna()
    error = work[column] - work["realized_r"]
    return {
        "covered_trade_count": int(len(work)),
        "coverage_rate": float(len(work) / len(frame)) if len(frame) else None,
        "spearman": _rank_corr(work[column], work["realized_r"]),
        "pair_concordance": _pair_concordance(work[column], work["realized_r"]),
        "top_bottom_realized_spread_r": _top_bottom_realized_spread(work.rename(columns={column: "target"}), "target"),
        "mae_r": float(error.abs().mean()) if len(work) else None,
        "rmse_r": float(math.sqrt(float((error * error).mean()))) if len(work) else None,
        "target_mean_r": float(work[column].mean()) if len(work) else None,
        "realized_mean_r": float(work["realized_r"].mean()) if len(work) else None,
    }


def _decision(metrics: dict[str, dict[str, Any]]) -> tuple[str, str]:
    planned = metrics["planned_risk_40d"]
    reanchored = metrics["entry_fixed_risk_40d"]
    planned_rho = _finite(planned.get("spearman"))
    control_rho = _finite(reanchored.get("spearman"))
    planned_spread = _finite(planned.get("top_bottom_realized_spread_r"))
    control_spread = _finite(reanchored.get("top_bottom_realized_spread_r"))
    if None in (planned_rho, control_rho, planned_spread, control_spread):
        return "BLOCKED", "共同completed-trade subset不足以同時計算Spearman與Top-Bottom realized-R spread"
    rho_better = planned_rho > control_rho
    spread_better = planned_spread > control_spread and planned_spread > 0.0
    if rho_better and spread_better:
        return "GO", "planned-risk target在同一entry-date控制下同時改善rank correlation與Top-Bottom realized-R separation"
    if (not rho_better) and (not spread_better):
        return "REJECT", "planned-risk target在同一entry-date控制下未改善rank correlation或Top-Bottom realized-R separation"
    return "NEXT_EXPERIMENT", "planned-risk target對rank correlation與Top-Bottom realized-R separation呈混合結果；不直接進模型訓練"


def analyze_alignment_frames(
    *,
    executions: pd.DataFrame,
    trades: pd.DataFrame,
    selected_targets: pd.DataFrame,
    market_frames: dict[str, pd.DataFrame],
    horizon_bars: int,
    fixed_risk_budget_return: float,
    capital_bucket_count: int = 5,
    entry_age_bucket_count: int = 4,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Pure analysis core; public for deterministic synthetic validation."""

    exec_frame = pd.DataFrame(executions).copy()
    trade_history = pd.DataFrame(trades).copy()
    target_frame = pd.DataFrame(selected_targets).copy()
    required_exec = {"ticker", "trade_date", "signal_date", "entry_type", "entry_filled", "limit_px", "init_sl"}
    required_target = {"ticker", "trade_date", "signal_date", "target_raw_r", "target_available"}
    if missing := sorted(required_exec - set(exec_frame.columns)):
        raise ValueError(f"execution sidecar缺欄位: {missing}")
    if missing := sorted(required_target - set(target_frame.columns)):
        raise ValueError(f"selected-target sidecar缺欄位: {missing}")

    completed = reconstruct_round_trips(trade_history, scenario="active")
    filled = exec_frame.loc[_bool_series(exec_frame["entry_filled"])].copy()
    for frame in (filled, target_frame):
        frame["ticker"] = frame["ticker"].fillna("").astype(str).str.strip()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
        frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    filled["entry_type"] = filled["entry_type"].fillna("normal").astype(str)
    if bool(filled.duplicated(["ticker", "trade_date", "entry_type"]).any()):
        bad = filled.loc[filled.duplicated(["ticker", "trade_date", "entry_type"], keep=False)].iloc[0]
        raise ValueError(f"filled execution key不唯一: ticker={bad['ticker']}, date={bad['trade_date']}, type={bad['entry_type']}")
    if bool(target_frame.duplicated(["ticker", "trade_date", "signal_date"]).any()):
        bad = target_frame.loc[target_frame.duplicated(["ticker", "trade_date", "signal_date"], keep=False)].iloc[0]
        raise ValueError(f"selected target key不唯一: ticker={bad['ticker']}, date={bad['trade_date']}, signal={bad['signal_date']}")

    work = completed.copy()
    work["entry_date"] = pd.to_datetime(work["entry_date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    work["signal_date"] = pd.to_datetime(work["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    work = work.merge(
        filled[["ticker", "trade_date", "signal_date", "entry_type", "limit_px", "init_sl"]],
        left_on=["ticker", "entry_date", "entry_type"],
        right_on=["ticker", "trade_date", "entry_type"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_exec"),
    )
    work = work.merge(
        target_frame[["ticker", "trade_date", "signal_date", "target_raw_r", "target_available"]],
        left_on=["ticker", "entry_date", "signal_date"],
        right_on=["ticker", "trade_date", "signal_date"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_target"),
    )
    work["realized_r"] = pd.to_numeric(work["r_multiple"], errors="coerce")
    work["mr13e_target_r"] = pd.to_numeric(work["target_raw_r"], errors="coerce").where(
        _bool_series(work["target_available"])
    )

    planned_values: list[float | None] = []
    fixed_values: list[float | None] = []
    geometry_reasons: list[str] = []
    capital_intensity: list[float | None] = []
    stop_gap_pct: list[float | None] = []
    entry_age_days: list[int | None] = []
    for row in work.to_dict("records"):
        limit_px = _finite(row.get("limit_px"))
        init_sl = _finite(row.get("init_sl"))
        entry_date = str(row.get("entry_date") or "")
        ticker = str(row.get("ticker") or "")
        frame = market_frames.get(ticker)
        if limit_px is None or init_sl is None or limit_px <= 0.0 or not (0.0 < init_sl < limit_px):
            planned_values.append(None); fixed_values.append(None); geometry_reasons.append("invalid_geometry")
            capital_intensity.append(None); stop_gap_pct.append(None); entry_age_days.append(None); continue
        path = None if frame is None else _future_path_after_entry(frame, entry_date, int(horizon_bars))
        if path is None:
            planned_values.append(None); fixed_values.append(None); geometry_reasons.append("insufficient_market_path")
            capital_intensity.append(float(limit_px / (limit_px - init_sl)))
            stop_gap_pct.append(float((limit_px - init_sl) / limit_px))
            signal = _date_text(row.get("signal_date"))
            entry_age_days.append(None if not signal else int((pd.Timestamp(entry_date) - pd.Timestamp(signal)).days))
            continue
        highs, lows = path
        planned = _opportunity_target_from_prices(
            highs, lows,
            anchor_price=limit_px,
            risk_barrier_price=init_sl,
            horizon_bars=int(horizon_bars),
        )
        fixed = _opportunity_target_from_prices(
            highs, lows,
            anchor_price=limit_px,
            risk_barrier_price=float(limit_px * (1.0 - fixed_risk_budget_return)),
            horizon_bars=int(horizon_bars),
        )
        planned_values.append(planned.target_r if planned.valid else None)
        fixed_values.append(fixed.target_r if fixed.valid else None)
        geometry_reasons.append("ok" if planned.valid and fixed.valid else f"planned={planned.reason};fixed={fixed.reason}")
        capital_intensity.append(float(limit_px / (limit_px - init_sl)))
        stop_gap_pct.append(float((limit_px - init_sl) / limit_px))
        signal = _date_text(row.get("signal_date"))
        entry_age_days.append(None if not signal else int((pd.Timestamp(entry_date) - pd.Timestamp(signal)).days))

    work["entry_fixed_risk_40d_r"] = fixed_values
    work["planned_risk_40d_r"] = planned_values
    work["capital_per_risk"] = capital_intensity
    work["stop_gap_pct"] = stop_gap_pct
    work["entry_age_calendar_days"] = entry_age_days
    work["planned_target_reason"] = geometry_reasons
    work["entry_year"] = pd.to_datetime(work["entry_date"], errors="coerce").dt.year

    comparable = work.dropna(subset=["realized_r", "mr13e_target_r", "entry_fixed_risk_40d_r", "planned_risk_40d_r"]).copy()
    metrics = {
        "mr13e_existing": _target_summary(comparable, "mr13e_target_r"),
        "entry_fixed_risk_40d": _target_summary(comparable, "entry_fixed_risk_40d_r"),
        "planned_risk_40d": _target_summary(comparable, "planned_risk_40d_r"),
    }
    decision, reason = _decision(metrics)

    yearly_rows = []
    for year, day in comparable.groupby("entry_year", dropna=True, sort=True):
        if len(day) < 5:
            continue
        yearly_rows.append({
            "year": int(year),
            "trade_count": int(len(day)),
            "mr13e_spearman": _rank_corr(day["mr13e_target_r"], day["realized_r"]),
            "entry_fixed_spearman": _rank_corr(day["entry_fixed_risk_40d_r"], day["realized_r"]),
            "planned_spearman": _rank_corr(day["planned_risk_40d_r"], day["realized_r"]),
        })

    capital_rows = []
    capital_work = comparable.dropna(subset=["capital_per_risk"]).copy()
    if len(capital_work) >= int(capital_bucket_count) * 2 and capital_work["capital_per_risk"].nunique() >= int(capital_bucket_count):
        capital_work["capital_bucket"] = pd.qcut(
            capital_work["capital_per_risk"], q=int(capital_bucket_count), duplicates="drop"
        )
        for index, (_bucket, day) in enumerate(capital_work.groupby("capital_bucket", observed=True, sort=True), start=1):
            capital_rows.append({
                "bucket": f"Q{index}",
                "trade_count": int(len(day)),
                "capital_per_risk_min": float(day["capital_per_risk"].min()),
                "capital_per_risk_max": float(day["capital_per_risk"].max()),
                "mr13e_spearman": _rank_corr(day["mr13e_target_r"], day["realized_r"]),
                "entry_fixed_spearman": _rank_corr(day["entry_fixed_risk_40d_r"], day["realized_r"]),
                "planned_spearman": _rank_corr(day["planned_risk_40d_r"], day["realized_r"]),
            })

    entry_age_rows = []
    age_work = comparable.dropna(subset=["entry_age_calendar_days"]).copy()
    if (
        len(age_work) >= int(entry_age_bucket_count) * 2
        and age_work["entry_age_calendar_days"].nunique() >= int(entry_age_bucket_count)
    ):
        age_work["entry_age_bucket"] = pd.qcut(
            age_work["entry_age_calendar_days"],
            q=int(entry_age_bucket_count),
            duplicates="drop",
        )
        for index, (_bucket, day) in enumerate(
            age_work.groupby("entry_age_bucket", observed=True, sort=True), start=1
        ):
            entry_age_rows.append({
                "bucket": f"Q{index}",
                "trade_count": int(len(day)),
                "entry_age_min_days": int(day["entry_age_calendar_days"].min()),
                "entry_age_max_days": int(day["entry_age_calendar_days"].max()),
                "mr13e_spearman": _rank_corr(day["mr13e_target_r"], day["realized_r"]),
                "entry_fixed_spearman": _rank_corr(day["entry_fixed_risk_40d_r"], day["realized_r"]),
                "planned_spearman": _rank_corr(day["planned_risk_40d_r"], day["realized_r"]),
            })

    payload = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "decision": decision,
        "decision_reason": reason,
        "basis": "same_completed_trades_same_entry_date_fixed_risk_vs_planned_risk_40d",
        "horizon_bars": int(horizon_bars),
        "fixed_risk_budget_return": float(fixed_risk_budget_return),
        "completed_trade_count": int(len(work)),
        "comparable_trade_count": int(len(comparable)),
        "comparable_coverage_rate": float(len(comparable) / len(work)) if len(work) else None,
        "metrics": metrics,
        "yearly": yearly_rows,
        "capital_intensity_buckets": capital_rows,
        "entry_age_buckets": entry_age_rows,
        "lookahead_contract": "future prices are used only as offline labels; no runtime/training artifact is modified",
        "path_contract": "40 trading bars after actual entry date; same-bar risk breach adverse-first; entry day excluded",
    }
    return work, payload


def _resolve_source(definition: AuditDefinition, *, project_root: Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    profile_id = str(definition.source.get("strategy_compare_profile") or "selection_pit")
    settings = get_strategy_comparison_settings(profile_id)
    integration = get_strategy_runtime_integration_settings()
    if profile_id != str(integration.selection_profile_id):
        raise ValueError(
            "planned-risk 40D Audit目前只允許runtime integration指定的Selection profile"
        )
    candidate_arm_id = str(integration.selection_candidate_arm_id).strip()
    status = collect_artifact_status(project_root=root, settings=settings)
    replay_cache = dict(status.get("replay_cache") or {})
    cache_entry = dict((replay_cache.get("pairs") or {}).get(candidate_arm_id) or {})
    pair_dir_raw = cache_entry.get("source_pair_dir")
    pair_dir = Path(pair_dir_raw).resolve() if pair_dir_raw else None
    missing = []
    if pair_dir is None or not pair_dir.is_dir():
        missing.append("completed candidate pair")
    else:
        missing.extend(name for name in _REQUIRED_PAIR_FILES if not (pair_dir / name).is_file())
    dataset_dir = Path(get_dataset_dir(str(root), settings.dataset))
    dataset_map: dict[str, Path] = {}
    if not dataset_dir.is_dir():
        missing.append(f"dataset:{settings.dataset}")
    else:
        dataset_inputs, duplicate_lines = discover_dataset_csv_inputs(root, settings.dataset)
        if duplicate_lines:
            missing.append("dataset duplicate ticker inputs")
        dataset_map = {ticker: Path(path) for ticker, path in dataset_inputs}
        if not dataset_map:
            missing.append(f"dataset:{settings.dataset}")
    return {
        "settings": settings,
        "candidate_arm_id": candidate_arm_id,
        "pair_dir": pair_dir,
        "dataset_map": dataset_map,
        "missing": missing,
    }


def planned_risk_40d_alignment_status(definition: AuditDefinition, *, project_root: Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    source = _resolve_source(definition, project_root=root)
    pair_dir = source["pair_dir"]
    display = "-" if pair_dir is None else project_relative_display_path(pair_dir, project_root=root)
    if source["missing"]:
        return {
            "status": "BLOCKED",
            "reason": "；".join(str(item) for item in source["missing"]),
            "source": {"display": display, "candidate_arm_id": source["candidate_arm_id"]},
        }
    return {
        "status": "READY",
        "reason": "",
        "source": {"display": display, "candidate_arm_id": source["candidate_arm_id"]},
    }


def _render_audit_markdown(payload: dict[str, Any], *, source_display: str) -> str:
    metrics = dict(payload.get("metrics") or {})
    rows = []
    for key, label in (
        ("mr13e_existing", "MR-13E existing"),
        ("entry_fixed_risk_40d", "Entry-date fixed-risk 40D"),
        ("planned_risk_40d", "Planned-risk 40D"),
    ):
        item = dict(metrics.get(key) or {})
        rows.append((
            label,
            item.get("covered_trade_count"),
            "-" if _finite(item.get("spearman")) is None else f"{float(item['spearman']):.4f}",
            "-" if _finite(item.get("pair_concordance")) is None else f"{float(item['pair_concordance'])*100:.2f}%",
            "-" if _finite(item.get("top_bottom_realized_spread_r")) is None else f"{float(item['top_bottom_realized_spread_r']):.2f} R",
            "-" if _finite(item.get("mae_r")) is None else f"{float(item['mae_r']):.2f} R",
        ))
    sections = [
        render_title("Min ROOS Planned-risk 40D Target Alignment Audit"),
        render_key_values((
            ("Decision", payload.get("decision")),
            ("Reason", payload.get("decision_reason")),
            ("Source", source_display),
            ("Completed trades", payload.get("completed_trade_count")),
            ("Comparable trades", payload.get("comparable_trade_count")),
            ("Horizon", f"{payload.get('horizon_bars')} trading bars after entry date"),
            ("Fixed-risk control", f"{float(payload.get('fixed_risk_budget_return') or 0.0)*100:.2f}%"),
        )),
        "\n## 1. Headline alignment\n\n" + render_table(
            ("Target", "Trades", "Spearman", "Pair一致", "Top-Bottom realized R", "MAE"), rows
        ),
    ]
    yearly = list(payload.get("yearly") or [])
    if yearly:
        sections.append("\n## 2. Yearly Spearman\n\n" + render_table(
            ("Year", "Trades", "MR-13E", "Entry-fixed", "Planned-risk"),
            [(
                row["year"], row["trade_count"],
                "-" if row.get("mr13e_spearman") is None else f"{row['mr13e_spearman']:.3f}",
                "-" if row.get("entry_fixed_spearman") is None else f"{row['entry_fixed_spearman']:.3f}",
                "-" if row.get("planned_spearman") is None else f"{row['planned_spearman']:.3f}",
            ) for row in yearly],
        ))
    capital = list(payload.get("capital_intensity_buckets") or [])
    if capital:
        sections.append("\n## 3. Capital-per-risk buckets\n\n" + render_table(
            ("Bucket", "Trades", "Capital/Risk min", "max", "MR-13E", "Entry-fixed", "Planned-risk"),
            [(
                row["bucket"], row["trade_count"], f"{row['capital_per_risk_min']:.2f}", f"{row['capital_per_risk_max']:.2f}",
                "-" if row.get("mr13e_spearman") is None else f"{row['mr13e_spearman']:.3f}",
                "-" if row.get("entry_fixed_spearman") is None else f"{row['entry_fixed_spearman']:.3f}",
                "-" if row.get("planned_spearman") is None else f"{row['planned_spearman']:.3f}",
            ) for row in capital],
        ))
    entry_age = list(payload.get("entry_age_buckets") or [])
    if entry_age:
        sections.append("\n## 4. Entry-age buckets\n\n" + render_table(
            ("Bucket", "Trades", "Age min", "max", "MR-13E", "Entry-fixed", "Planned-risk"),
            [(
                row["bucket"], row["trade_count"], f"{row['entry_age_min_days']} d", f"{row['entry_age_max_days']} d",
                "-" if row.get("mr13e_spearman") is None else f"{row['mr13e_spearman']:.3f}",
                "-" if row.get("entry_fixed_spearman") is None else f"{row['entry_fixed_spearman']:.3f}",
                "-" if row.get("planned_spearman") is None else f"{row['planned_spearman']:.3f}",
            ) for row in entry_age],
        ))
    sections.append(
        "\n## 判讀契約\n\n"
        "主要控制是同一批 completed trades、同一 entry date、同一 planned limit。"
        "Entry-date fixed-risk 40D 只控制進場日期重錨；Planned-risk 40D 再把 risk barrier／R 分母改為 planned initial stop。"
        "GO 只有在 Planned-risk 同時改善 Spearman 與 Top-Bottom realized-R separation；混合結果不直接進模型訓練。"
    )
    return "\n\n".join(sections).rstrip() + "\n"


def run_planned_risk_40d_alignment_audit(
    definition: AuditDefinition,
    *,
    project_root: Path,
    quiet: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    source = _resolve_source(definition, project_root=root)
    if source["missing"]:
        raise RuntimeError("Audit缺少必要read-only來源：" + "；".join(str(item) for item in source["missing"]))
    pair_dir = Path(source["pair_dir"])
    execution = pd.read_csv(pair_dir / "score_ranking_execution.csv", encoding="utf-8-sig", low_memory=False)
    trades = pd.read_csv(pair_dir / "score_ranking_trades.csv", encoding="utf-8-sig", low_memory=False)
    selected = pd.read_csv(pair_dir / "score_ranking_selected_target_diagnostics.csv", encoding="utf-8-sig", low_memory=False)

    completed = reconstruct_round_trips(trades, scenario="active")
    tickers = sorted(set(completed.get("ticker", pd.Series(dtype=str)).dropna().astype(str)))
    market_frames: dict[str, pd.DataFrame] = {}
    missing_market = []
    dataset_map = dict(source["dataset_map"])
    for ticker in tickers:
        path = dataset_map.get(ticker)
        if path is None:
            missing_market.append(ticker)
            continue
        try:
            market_frames[ticker] = load_dataset_frame(path, ticker, min_rows=1)
        except (OSError, UnicodeDecodeError, ValueError, KeyError, TypeError) as exc:
            missing_market.append(f"{ticker}:{type(exc).__name__}")
    if missing_market:
        sample = ", ".join(missing_market[:10])
        raise RuntimeError(f"Audit completed trades缺少canonical OHLCV: count={len(missing_market)}, sample={sample}")

    spec = StrategyAlignedContinuousTargetSpec.from_label_policy(DEFAULT_LABEL_POLICY)
    configured_horizon = definition.dimensions.get("horizon_bars")
    horizon = int(spec.horizon_bars if configured_horizon in (None, "canonical") else configured_horizon)
    if horizon != int(spec.horizon_bars):
        raise ValueError("A0 horizon必須重用current canonical 40D label horizon，不得另設研究magic value")
    capital_bucket_count = int(definition.dimensions.get("capital_bucket_count", 5) or 5)
    entry_age_bucket_count = int(definition.dimensions.get("entry_age_bucket_count", 4) or 4)
    detail, payload = analyze_alignment_frames(
        executions=execution,
        trades=trades,
        selected_targets=selected,
        market_frames=market_frames,
        horizon_bars=horizon,
        fixed_risk_budget_return=float(spec.risk_budget_return),
        capital_bucket_count=capital_bucket_count,
        entry_age_bucket_count=entry_age_bucket_count,
    )
    source_display = project_relative_display_path(pair_dir, project_root=root)
    payload.update({
        "audit_id": definition.audit_id,
        "audit_type": definition.audit_type,
        "candidate_arm_id": source["candidate_arm_id"],
        "source_pair_dir": source_display,
        "dataset": source["settings"].dataset,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    })

    output_root = root / AUDIT_OUTPUT_ROOT / definition.output_subdir
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    detail.to_csv(run_dir / "trade_alignment.csv", index=False, encoding="utf-8-sig")
    with (run_dir / "audit.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    report = _render_audit_markdown(payload, source_display=source_display)
    (run_dir / "audit.md").write_text(report, encoding="utf-8")
    latest_dir = output_root / "latest"
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(run_dir, latest_dir)
    if not quiet:
        print(report.rstrip())
        print("\n工件輸出")
        print("--------")
        print("Audit Markdown：" + project_relative_display_path(run_dir / "audit.md", project_root=root))
        print("Audit JSON    ：" + project_relative_display_path(run_dir / "audit.json", project_root=root))
        print("Trade detail  ：" + project_relative_display_path(run_dir / "trade_alignment.csv", project_root=root))
    return payload


__all__ = [
    "OpportunityTargetResult",
    "analyze_alignment_frames",
    "planned_risk_40d_alignment_status",
    "run_planned_risk_40d_alignment_audit",
]
