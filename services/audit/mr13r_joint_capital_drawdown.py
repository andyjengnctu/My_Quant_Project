"""Read-only MR-13R joint-signal / capital-conversion / drawdown Audit.

The Audit consumes completed Strategy Compare sidecars plus canonical post-replay
MFE×Safety truth.  It never replays a strategy, changes ranking, fits a threshold,
or uses future truth in runtime decisions.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from config.audit import AUDIT_OUTPUT_ROOT, AuditDefinition
from filters.breakout_quality.trade_attribution import reconstruct_round_trips
from services.audit.mfe_safety_truth import (
    AuditBlockedError,
    attach_quadrants,
    build_truth_geometry,
    filter_period,
    normalize_date,
    normalize_ticker,
)
from services.audit.strategy_compare_source import (
    AuditSourceBlockedError,
    load_strategy_arm_path_sidecars,
    load_strategy_compare_source,
)

SUPPORTED_AUDIT_TYPE = "mr13r_joint_capital_drawdown"


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _period_from_source(source: Any) -> tuple[str, str]:
    payload = source.result.get("comparison_period")
    if not isinstance(payload, Mapping):
        raise AuditBlockedError(f"{source.profile_id} Strategy Compare缺少comparison_period")
    start = normalize_date(payload.get("start"))
    end = normalize_date(payload.get("end"))
    if not start or not end:
        raise AuditBlockedError(f"{source.profile_id} comparison_period格式無效")
    return start, end


def _average_zero_based_percentile(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=series.index, dtype=float)
    valid = values.notna() & np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))
    if not bool(valid.any()):
        return out
    ranked = values.loc[valid].rank(method="average") - 1.0
    n = int(valid.sum())
    out.loc[valid] = 0.5 if n == 1 else ranked / float(n - 1)
    return out


def _same_day_percentiles(frame: pd.DataFrame, value_column: str, output_column: str) -> pd.DataFrame:
    table = pd.DataFrame(frame).copy()
    if value_column not in table.columns:
        raise AuditBlockedError(f"orderable sidecar缺少{value_column}")
    table[output_column] = table.groupby("trade_date", sort=False, group_keys=False)[value_column].transform(
        _average_zero_based_percentile
    )
    return table


def _normalize_orderable(frame: pd.DataFrame) -> pd.DataFrame:
    table = pd.DataFrame(frame).copy()
    required = {
        "ticker", "trade_date", "breakout_quality_score_date", "breakout_quality_score",
        "breakout_quality_safety_score", "breakout_quality_safety_score_available",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise AuditBlockedError(f"MR-13R orderable sidecar缺少欄位: {missing}")
    table["ticker"] = table["ticker"].map(normalize_ticker)
    table["trade_date"] = table["trade_date"].map(normalize_date)
    table["score_event_date"] = table["breakout_quality_score_date"].map(normalize_date)
    table["conditional_mfe_score"] = pd.to_numeric(table["breakout_quality_score"], errors="coerce")
    table["raw_safety_score"] = pd.to_numeric(table["breakout_quality_safety_score"], errors="coerce")
    available = table["breakout_quality_safety_score_available"].astype(str).str.lower().isin(
        {"true", "1", "yes"}
    )
    table.loc[~available, "raw_safety_score"] = np.nan
    table = table.loc[
        table["ticker"].ne("") & table["trade_date"].ne("") & table["score_event_date"].ne("")
    ].copy()
    table = _same_day_percentiles(table, "conditional_mfe_score", "predicted_mfe_percentile")
    table = _same_day_percentiles(table, "raw_safety_score", "predicted_safety_percentile")
    table["predicted_joint_product"] = (
        table["predicted_mfe_percentile"] * table["predicted_safety_percentile"]
    )
    return table


def _bin_quintile(values: pd.Series, bins: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    index = np.floor(numeric * float(bins)).clip(0, bins - 1)
    return pd.Series(index, index=values.index).astype("Int64")


def _mean_daily_spearman(frame: pd.DataFrame, x: str, y: str) -> tuple[float | None, int]:
    rhos: list[float] = []
    for _date, group in frame.groupby("trade_date", sort=False):
        pair = group[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(pair) < 2 or pair[x].nunique() < 2 or pair[y].nunique() < 2:
            continue
        rho = pair[x].corr(pair[y], method="spearman")
        if rho is not None and math.isfinite(float(rho)):
            rhos.append(float(rho))
    return (None if not rhos else float(np.mean(rhos))), len(rhos)


def build_joint_signal_analysis(
    orderable: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    cutoff: float,
    bins: int,
) -> dict[str, Any]:
    table = _normalize_orderable(orderable)
    truth_table = attach_quadrants(pd.DataFrame(truth), cutoff=cutoff)
    joined = table.merge(
        truth_table,
        left_on=["ticker", "score_event_date"],
        right_on=["ticker", "date"],
        how="inner",
        validate="many_to_one",
    )
    joined = joined.loc[
        joined["predicted_mfe_percentile"].notna()
        & joined["predicted_safety_percentile"].notna()
    ].copy()
    if joined.empty:
        raise AuditBlockedError("MR-13R orderable與canonical MFE/Safety truth沒有共同scored rows")
    joined["predicted_mfe_bin"] = _bin_quintile(joined["predicted_mfe_percentile"], bins)
    joined["predicted_safety_bin"] = _bin_quintile(joined["predicted_safety_percentile"], bins)
    joined["actual_high_mfe"] = joined["mfe_percentile"] >= cutoff
    joined["actual_high_safety"] = joined["safety_percentile"] >= cutoff
    joined["actual_hmhs"] = joined["actual_high_mfe"] & joined["actual_high_safety"]

    cells: list[dict[str, Any]] = []
    for (s_bin, m_bin), group in joined.groupby(
        ["predicted_safety_bin", "predicted_mfe_bin"], observed=True, sort=True
    ):
        cells.append({
            "predicted_safety_bin": int(s_bin),
            "predicted_mfe_bin": int(m_bin),
            "rows": int(len(group)),
            "actual_hmhs_pct": float(group["actual_hmhs"].mean() * 100.0),
            "actual_high_mfe_pct": float(group["actual_high_mfe"].mean() * 100.0),
            "actual_high_safety_pct": float(group["actual_high_safety"].mean() * 100.0),
            "actual_mfe_percentile_mean": float(group["mfe_percentile"].mean()),
            "actual_safety_percentile_mean": float(group["safety_percentile"].mean()),
        })

    cohorts: list[dict[str, Any]] = []
    for s_bin, group in joined.groupby("predicted_safety_bin", observed=True, sort=True):
        rho, days = _mean_daily_spearman(group, "conditional_mfe_score", "mfe_percentile")
        cohorts.append({
            "predicted_safety_bin": int(s_bin),
            "rows": int(len(group)),
            "mean_daily_conditional_mfe_to_actual_mfe_spearman": rho,
            "valid_days": int(days),
            "actual_hmhs_pct": float(group["actual_hmhs"].mean() * 100.0),
            "actual_high_mfe_pct": float(group["actual_high_mfe"].mean() * 100.0),
            "actual_high_safety_pct": float(group["actual_high_safety"].mean() * 100.0),
        })

    primary_rho, primary_days = _mean_daily_spearman(
        joined, "conditional_mfe_score", "mfe_percentile"
    )
    safety_rho, safety_days = _mean_daily_spearman(
        joined, "raw_safety_score", "safety_percentile"
    )
    product_rho, product_days = _mean_daily_spearman(
        joined, "predicted_joint_product", "actual_hmhs"
    )
    upper = joined.loc[
        (joined["predicted_safety_bin"] == bins - 1)
        & (joined["predicted_mfe_bin"] == bins - 1)
    ]
    return {
        "rows": int(len(joined)),
        "truth_coverage_pct": float(len(joined) / max(len(table), 1) * 100.0),
        "primary_to_actual_mfe_mean_daily_spearman": primary_rho,
        "primary_valid_days": int(primary_days),
        "safety_to_actual_safety_mean_daily_spearman": safety_rho,
        "safety_valid_days": int(safety_days),
        "joint_product_to_hmhs_mean_daily_spearman": product_rho,
        "joint_product_valid_days": int(product_days),
        "population_hmhs_pct": float(joined["actual_hmhs"].mean() * 100.0),
        "upper_right_rows": int(len(upper)),
        "upper_right_hmhs_pct": (None if upper.empty else float(upper["actual_hmhs"].mean() * 100.0)),
        "upper_right_high_mfe_pct": (None if upper.empty else float(upper["actual_high_mfe"].mean() * 100.0)),
        "upper_right_high_safety_pct": (None if upper.empty else float(upper["actual_high_safety"].mean() * 100.0)),
        "cells": cells,
        "safety_cohorts": cohorts,
        "detail": joined,
    }


def _load_equity(pair_dir: Path) -> tuple[pd.DataFrame, Path]:
    path = pair_dir / "score_ranking_equity.csv"
    if not path.is_file():
        raise AuditBlockedError(f"缺少既有equity sidecar: {path.name}")
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    required = {"Date", "Equity", "Exposure_Pct"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise AuditBlockedError(f"equity sidecar缺少欄位: {missing}")
    frame["Date"] = frame["Date"].map(normalize_date)
    frame["Equity"] = pd.to_numeric(frame["Equity"], errors="coerce")
    frame["Exposure_Pct"] = pd.to_numeric(frame["Exposure_Pct"], errors="coerce")
    if frame[["Date", "Equity"]].dropna().empty:
        raise AuditBlockedError("equity sidecar沒有有效Date/Equity")
    return frame, path


def _load_round_trips(pair_dir: Path) -> tuple[pd.DataFrame, Path]:
    path = pair_dir / "score_ranking_trades.csv"
    if not path.is_file():
        raise AuditBlockedError(f"缺少既有trade sidecar: {path.name}")
    history = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    return reconstruct_round_trips(history, scenario="score_ranking"), path


def _daily_spearman_optional(frame: pd.DataFrame, x: str, y: str) -> tuple[float | None, int]:
    if x not in frame.columns or y not in frame.columns:
        return None, 0
    return _mean_daily_spearman(frame, x, y)


def build_capital_conversion_analysis(evidence: Mapping[str, Any]) -> dict[str, Any]:
    orderable = _normalize_orderable(pd.DataFrame(evidence["orderable"]))
    execution = pd.DataFrame(evidence["execution"]).copy()
    if execution.empty:
        raise AuditBlockedError("execution sidecar為空")
    chosen = pd.to_numeric(execution.get("chosen_qty"), errors="coerce").fillna(0) > 0
    execution = execution.loc[chosen].copy()
    if execution.empty:
        raise AuditBlockedError("execution sidecar沒有chosen entry")
    for column in (
        "limit_px", "init_sl", "sizing_equity", "reserved_cost", "chosen_risk_utilization",
    ):
        if column in execution.columns:
            execution[column] = pd.to_numeric(execution[column], errors="coerce")
    if {"limit_px", "init_sl"}.issubset(execution.columns):
        execution["stop_distance_pct"] = (
            (execution["limit_px"] - execution["init_sl"]) / execution["limit_px"] * 100.0
        )
    else:
        execution["stop_distance_pct"] = np.nan
    if {"reserved_cost", "sizing_equity"}.issubset(execution.columns):
        execution["reserved_fraction_pct"] = execution["reserved_cost"] / execution["sizing_equity"] * 100.0
    else:
        execution["reserved_fraction_pct"] = np.nan

    # Join selected execution rows back to the same decision-time Raw Safety score.
    # This remains descriptive post-replay attribution; it never feeds runtime.
    execution["ticker"] = execution.get("ticker", pd.Series("", index=execution.index)).map(normalize_ticker)
    execution["trade_date"] = execution.get("trade_date", pd.Series("", index=execution.index)).map(normalize_date)
    execution["signal_date"] = execution.get("signal_date", pd.Series("", index=execution.index)).map(normalize_date)
    join_keys = ["ticker", "trade_date", "signal_date"]
    safety_lookup = orderable[join_keys + ["raw_safety_score"]].copy()
    duplicate_keys = safety_lookup.duplicated(join_keys, keep=False)
    if bool(duplicate_keys.any()):
        # Candidate rows can occasionally carry duplicated diagnostic rows.  A score
        # is valid for this attribution only when duplicates agree exactly.
        agreement = (
            safety_lookup.loc[duplicate_keys]
            .groupby(join_keys, sort=False)["raw_safety_score"]
            .nunique(dropna=False)
        )
        if bool((agreement > 1).any()):
            raise AuditBlockedError("同一execution key對應多個Raw Safety score")
        safety_lookup = safety_lookup.drop_duplicates(join_keys, keep="first")
    execution = execution.merge(safety_lookup, on=join_keys, how="left", validate="many_to_one")

    equity, equity_path = _load_equity(Path(evidence["pair_dir"]))
    round_trips, trades_path = _load_round_trips(Path(evidence["pair_dir"]))

    projected_corr, projected_days = _daily_spearman_optional(
        orderable, "raw_safety_score", "projected_capital_fraction"
    )
    deployment_corr, deployment_days = _daily_spearman_optional(
        orderable, "raw_safety_score", "projected_capital_deployment_rate"
    )
    if {"entry_atr", "orig_limit"}.issubset(orderable.columns):
        orderable["atr_to_limit_pct"] = (
            pd.to_numeric(orderable["entry_atr"], errors="coerce")
            / pd.to_numeric(orderable["orig_limit"], errors="coerce") * 100.0
        )
    atr_corr, atr_days = _daily_spearman_optional(orderable, "raw_safety_score", "atr_to_limit_pct")
    selected_stop_corr, selected_stop_days = _daily_spearman_optional(
        execution, "raw_safety_score", "stop_distance_pct"
    )
    selected_reserved_corr, selected_reserved_days = _daily_spearman_optional(
        execution, "raw_safety_score", "reserved_fraction_pct"
    )
    selected_risk_util_corr, selected_risk_util_days = _daily_spearman_optional(
        execution, "raw_safety_score", "chosen_risk_utilization"
    )
    risk_util = (
        pd.to_numeric(execution["chosen_risk_utilization"], errors="coerce")
        if "chosen_risk_utilization" in execution.columns
        else pd.Series(dtype=float)
    )
    binding_share = (
        execution.get("binding_signature", pd.Series("", index=execution.index))
        .fillna("").astype(str).value_counts(normalize=True).head(8) * 100.0
    ).to_dict()

    return {
        "average_exposure_pct": float(pd.to_numeric(equity["Exposure_Pct"], errors="coerce").mean()),
        "median_stop_distance_pct": _finite(pd.to_numeric(execution["stop_distance_pct"], errors="coerce").median()),
        "mean_stop_distance_pct": _finite(pd.to_numeric(execution["stop_distance_pct"], errors="coerce").mean()),
        "median_reserved_fraction_pct": _finite(pd.to_numeric(execution["reserved_fraction_pct"], errors="coerce").median()),
        "mean_reserved_fraction_pct": _finite(pd.to_numeric(execution["reserved_fraction_pct"], errors="coerce").mean()),
        "mean_chosen_risk_utilization": _finite(risk_util.mean()),
        "mean_holding_calendar_days": _finite(pd.to_numeric(round_trips.get("holding_calendar_days"), errors="coerce").mean()),
        "median_holding_calendar_days": _finite(pd.to_numeric(round_trips.get("holding_calendar_days"), errors="coerce").median()),
        "round_trip_count": int(len(round_trips)),
        "orderable_rows": int(len(orderable)),
        "mean_orderable_per_day": float(orderable.groupby("trade_date", sort=False).size().mean()),
        "raw_safety_to_projected_capital_fraction_daily_spearman": projected_corr,
        "projected_capital_fraction_valid_days": int(projected_days),
        "raw_safety_to_projected_deployment_daily_spearman": deployment_corr,
        "projected_deployment_valid_days": int(deployment_days),
        "raw_safety_to_atr_limit_daily_spearman": atr_corr,
        "atr_limit_valid_days": int(atr_days),
        "selected_raw_safety_to_stop_distance_daily_spearman": selected_stop_corr,
        "selected_stop_distance_valid_days": int(selected_stop_days),
        "selected_raw_safety_to_reserved_fraction_daily_spearman": selected_reserved_corr,
        "selected_reserved_fraction_valid_days": int(selected_reserved_days),
        "selected_raw_safety_to_risk_utilization_daily_spearman": selected_risk_util_corr,
        "selected_risk_utilization_valid_days": int(selected_risk_util_days),
        "binding_signature_share_pct": {str(k): float(v) for k, v in binding_share.items()},
        "equity_path": equity_path,
        "trades_path": trades_path,
    }


def _drawdown_episodes(equity: pd.DataFrame) -> list[dict[str, Any]]:
    table = pd.DataFrame(equity).copy().reset_index(drop=True)
    table["Equity"] = pd.to_numeric(table["Equity"], errors="coerce")
    table = table.loc[table["Equity"].notna() & (table["Equity"] > 0)].copy().reset_index(drop=True)
    if table.empty:
        return []
    peak = table["Equity"].cummax()
    dd = (peak - table["Equity"]) / peak * 100.0
    episodes: list[dict[str, Any]] = []
    start_idx: int | None = None
    peak_idx = 0
    for idx, value in enumerate(dd):
        if value <= 1e-12:
            peak_idx = idx
            if start_idx is not None:
                segment = dd.iloc[start_idx:idx]
                trough_idx = int(segment.idxmax())
                episodes.append({
                    "peak_idx": int(start_idx - 1 if start_idx > 0 else peak_idx),
                    "trough_idx": trough_idx,
                    "recovery_idx": idx,
                    "max_drawdown_pct": float(dd.iloc[trough_idx]),
                })
                start_idx = None
        elif start_idx is None:
            start_idx = idx
    if start_idx is not None:
        segment = dd.iloc[start_idx:]
        trough_idx = int(segment.idxmax())
        episodes.append({
            "peak_idx": int(max(0, start_idx - 1)),
            "trough_idx": trough_idx,
            "recovery_idx": None,
            "max_drawdown_pct": float(dd.iloc[trough_idx]),
        })
    for episode in episodes:
        episode["peak_date"] = str(table.iloc[episode["peak_idx"]]["Date"])
        episode["trough_date"] = str(table.iloc[episode["trough_idx"]]["Date"])
        episode["recovery_date"] = (
            None if episode["recovery_idx"] is None else str(table.iloc[episode["recovery_idx"]]["Date"])
        )
        episode["peak_equity"] = float(table.iloc[episode["peak_idx"]]["Equity"])
        episode["trough_equity"] = float(table.iloc[episode["trough_idx"]]["Equity"])
    return sorted(episodes, key=lambda row: row["max_drawdown_pct"], reverse=True)


def build_drawdown_analysis(
    evidence: Mapping[str, Any],
    truth: pd.DataFrame,
    *,
    cutoff: float,
    top_n: int,
) -> dict[str, Any]:
    pair_dir = Path(evidence["pair_dir"])
    equity, _ = _load_equity(pair_dir)
    round_trips, _ = _load_round_trips(pair_dir)
    path = pd.DataFrame(evidence["upside_realization"]).copy()
    if path.empty:
        raise AuditBlockedError("upside_realization sidecar為空")
    if "score_event_date" not in path.columns or "ticker" not in path.columns:
        raise AuditBlockedError("upside_realization缺少ticker/score_event_date")
    path["ticker"] = path["ticker"].map(normalize_ticker)
    path["score_event_date"] = path["score_event_date"].map(normalize_date)
    truth_q = attach_quadrants(pd.DataFrame(truth), cutoff=cutoff)
    path = path.merge(
        truth_q,
        left_on=["ticker", "score_event_date"],
        right_on=["ticker", "date"],
        how="left",
        validate="many_to_one",
    )
    for column in ("entry_date", "exit_date"):
        if column in path.columns:
            path[column] = path[column].map(normalize_date)
    episodes = _drawdown_episodes(equity)[: int(top_n)]
    rows: list[dict[str, Any]] = []
    for rank, episode in enumerate(episodes, start=1):
        peak = str(episode["peak_date"])
        trough = str(episode["trough_date"])
        overlap = path.loc[
            path.get("entry_date", pd.Series("", index=path.index)).le(trough)
            & path.get("exit_date", pd.Series("", index=path.index)).ge(peak)
        ].copy()
        entered = path.loc[
            path.get("entry_date", pd.Series("", index=path.index)).between(peak, trough)
        ].copy()
        q_counts = overlap.get("quadrant", pd.Series(dtype=str)).value_counts().to_dict()
        realized = pd.to_numeric(overlap.get("realized_r"), errors="coerce")
        losing = realized < 0.0
        entry_counts = entered.get("entry_date", pd.Series(dtype=str)).value_counts()
        duration_days = int((pd.Timestamp(trough) - pd.Timestamp(peak)).days)
        rows.append({
            "episode_rank": rank,
            **{key: value for key, value in episode.items() if not key.endswith("_idx")},
            "overlapping_trade_count": int(len(overlap)),
            "entered_during_drawdown_count": int(len(entered)),
            "max_same_day_entries": int(entry_counts.max()) if len(entry_counts) else 0,
            "drawdown_to_trough_calendar_days": duration_days,
            "overlap_realized_mean_r": _finite(realized.mean()),
            "overlap_realized_sum_r": _finite(realized.sum()),
            "overlap_losing_trade_count": int(losing.sum()),
            "overlap_losing_sum_r": _finite(realized.loc[losing].sum()),
            "hmhs_count": int(q_counts.get("high_mfe_high_safety_pct", 0)),
            "hmls_count": int(q_counts.get("high_mfe_low_safety_pct", 0)),
            "lmhs_count": int(q_counts.get("low_mfe_high_safety_pct", 0)),
            "lmls_count": int(q_counts.get("low_mfe_low_safety_pct", 0)),
        })
    return {
        "max_drawdown_pct": (None if not episodes else float(episodes[0]["max_drawdown_pct"])),
        "episode_count": int(len(_drawdown_episodes(equity))),
        "top_episodes": rows,
        "round_trip_count": int(len(round_trips)),
    }


def _validate_arm_contracts(source: Any, arm_ids: tuple[str, ...]) -> None:
    arms = source.settings.arms
    for arm_id in arm_ids:
        arm = arms.get(arm_id)
        if arm is None or not arm.dl_enabled or arm.dl_id != "CONT13R_ROLL":
            raise AuditBlockedError(f"{source.profile_id}/{arm_id}不是MR-13R current/history DL arm")
        options = dict(arm.dl_runtime_options or {})
        if options.get("preserve_k") is not False or options.get("preserve_r0") is not False:
            raise AuditBlockedError(f"{source.profile_id}/{arm_id}不是No-K/No-R0 contract")


def _fingerprint(definition: AuditDefinition, source_refs: Mapping[str, Any]) -> str:
    payload = {"schema": 1, "definition": definition.as_dict(), "sources": source_refs}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _fmt(value: Any, digits: int = 2, suffix: str = "") -> str:
    number = _finite(value)
    return "-" if number is None else f"{number:.{digits}f}{suffix}"


def _markdown_table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def render_result(result: Mapping[str, Any]) -> str:
    lines = [
        "MR-13R Joint Signal / Capital / Drawdown Audit",
        "=" * 72,
        f"Audit：{result.get('audit_id')}",
        f"狀態：{result.get('status')}",
        "",
    ]
    for profile_id, payload in dict(result.get("evaluations") or {}).items():
        lines.extend([f"[{profile_id}]", "-"])
        joint = dict(payload.get("joint_signal") or {})
        lines.append(
            "Joint signal：Primary→MFE dailyρ="
            + _fmt(joint.get("primary_to_actual_mfe_mean_daily_spearman"), 3)
            + "；Safety→Safety dailyρ="
            + _fmt(joint.get("safety_to_actual_safety_mean_daily_spearman"), 3)
            + "；Predicted upper-right HM/HS="
            + _fmt(joint.get("upper_right_hmhs_pct"), 2, "%")
            + "；population="
            + _fmt(joint.get("population_hmhs_pct"), 2, "%")
        )
        capital_rows = []
        for arm_id, arm in dict(payload.get("arms") or {}).items():
            cap = dict(arm.get("capital_conversion") or {})
            capital_rows.append((
                arm_id,
                _fmt(cap.get("average_exposure_pct"), 2, "%"),
                _fmt(cap.get("median_stop_distance_pct"), 2, "%"),
                _fmt(cap.get("median_reserved_fraction_pct"), 2, "%"),
                _fmt(cap.get("mean_holding_calendar_days"), 1, "d"),
                _fmt(cap.get("raw_safety_to_projected_capital_fraction_daily_spearman"), 3),
                _fmt(cap.get("selected_raw_safety_to_stop_distance_daily_spearman"), 3),
                _fmt(cap.get("selected_raw_safety_to_reserved_fraction_daily_spearman"), 3),
            ))
        if capital_rows:
            lines.append(_markdown_table(
                ("Arm", "Avg Exposure", "Median stop dist", "Median reserved/equity", "Mean holding", "Safety→Projected capital ρ", "Safety→Stop dist ρ", "Safety→Reserved ρ"),
                capital_rows,
            ))
        dd_rows = []
        for arm_id, arm in dict(payload.get("arms") or {}).items():
            dd = dict(arm.get("drawdown") or {})
            top = list(dd.get("top_episodes") or [])
            first = top[0] if top else {}
            dd_rows.append((
                arm_id,
                _fmt(dd.get("max_drawdown_pct"), 2, "%"),
                first.get("peak_date", "-"),
                first.get("trough_date", "-"),
                first.get("overlapping_trade_count", "-"),
                first.get("max_same_day_entries", "-"),
                first.get("overlap_losing_trade_count", "-"),
                _fmt(first.get("overlap_realized_sum_r"), 2, "R"),
            ))
        if dd_rows:
            lines.append(_markdown_table(
                ("Arm", "Max DD", "Peak", "Trough", "Overlap trades", "Max same-day entries", "Losing trades", "Overlap ΣR"),
                dd_rows,
            ))
        lines.append("")
    return "\n".join(lines).rstrip()


def _render_markdown(result: Mapping[str, Any]) -> str:
    return "# MR-13R Joint Signal / Capital / Drawdown Audit\n\n```text\n" + render_result(result) + "\n```\n\n" \
        "## 判讀邊界\n\n" \
        "- Joint signal 使用完成 replay 後才 join 的 canonical future truth，僅供診斷，不參與任何選股。\n" \
        "- Capital conversion 是描述性 mechanism attribution；Safety 與 sizing/holding 的相關不等於單一因果證明。\n" \
        "- Drawdown attribution 以 canonical equity/trade sidecars 的 peak→trough episode 為準；不把單股 adverse 直接當成 portfolio MDD proxy。\n"


def run_audit(definition: AuditDefinition, *, project_root: Path) -> dict[str, Any]:
    if definition.audit_type != SUPPORTED_AUDIT_TYPE:
        raise ValueError(f"不支援audit_type={definition.audit_type}")
    root = Path(project_root).resolve()
    source_cfg = dict(definition.source)
    profiles = tuple(str(v) for v in source_cfg.get("evaluation_profile_ids", ()))
    arm_ids = tuple(str(v) for v in source_cfg.get("strategy_arm_ids", ()))
    anchor_arm_id = str(source_cfg.get("joint_signal_anchor_arm_id") or "").strip()
    if not profiles or not arm_ids or anchor_arm_id not in arm_ids:
        raise AuditBlockedError("MR-13R Joint/Capital/Drawdown Audit source設定不完整")
    dims = dict(definition.dimensions)
    cutoff = float(dims.get("truth_high_cutoff", 0.50))
    bins = int(dims.get("joint_signal_bins", 5))
    top_n = int(dims.get("drawdown_top_n", 5))
    percentile_method = str(dims.get("percentile_method") or "average_zero_based")
    truth, truth_source = build_truth_geometry(definition, root, percentile_method=percentile_method)

    evaluations: dict[str, Any] = {}
    source_refs: dict[str, Any] = {"truth": truth_source, "strategy": {}}
    joint_cell_frames: list[pd.DataFrame] = []
    cohort_frames: list[pd.DataFrame] = []
    capital_rows: list[dict[str, Any]] = []
    dd_rows: list[dict[str, Any]] = []
    for profile_id in profiles:
        pinned = str(dict(source_cfg.get("strategy_result_fingerprints") or {}).get(profile_id) or "").strip()
        source = load_strategy_compare_source(root, profile_id=profile_id, pinned_config_fingerprint=pinned or None)
        _validate_arm_contracts(source, arm_ids)
        start, end = _period_from_source(source)
        period_truth = filter_period(truth, start, end)
        evidences = {
            arm_id: load_strategy_arm_path_sidecars(root, source=source, arm_id=arm_id)
            for arm_id in arm_ids
        }
        anchor = evidences[anchor_arm_id]
        joint = build_joint_signal_analysis(
            pd.DataFrame(anchor["orderable"]), period_truth, cutoff=cutoff, bins=bins
        )
        detail = joint.pop("detail")
        joint_cell = pd.DataFrame(joint["cells"])
        if not joint_cell.empty:
            joint_cell.insert(0, "profile_id", profile_id)
            joint_cell_frames.append(joint_cell)
        cohorts = pd.DataFrame(joint["safety_cohorts"])
        if not cohorts.empty:
            cohorts.insert(0, "profile_id", profile_id)
            cohort_frames.append(cohorts)

        arm_payloads: dict[str, Any] = {}
        arm_sources: dict[str, Any] = {}
        for arm_id, evidence in evidences.items():
            capital = build_capital_conversion_analysis(evidence)
            capital["equity_path"] = _relative(Path(capital["equity_path"]), root)
            capital["trades_path"] = _relative(Path(capital["trades_path"]), root)
            drawdown = build_drawdown_analysis(evidence, period_truth, cutoff=cutoff, top_n=top_n)
            arm_payloads[arm_id] = {"capital_conversion": capital, "drawdown": drawdown}
            capital_rows.append({"profile_id": profile_id, "arm_id": arm_id, **capital})
            for row in drawdown["top_episodes"]:
                dd_rows.append({"profile_id": profile_id, "arm_id": arm_id, **row})
            arm_sources[arm_id] = {
                "pair_dir": _relative(Path(evidence["pair_dir"]), root),
                "orderable": _relative(Path(evidence["orderable_path"]), root),
                "execution": _relative(Path(evidence["execution_path"]), root),
                "daily_capacity": _relative(Path(evidence["daily_capacity_path"]), root),
                "upside_realization": _relative(Path(evidence["upside_realization_path"]), root),
                "active_trades": _relative(Path(evidence["active_trades_path"]), root),
            }
        evaluations[profile_id] = {
            "period": {"start": start, "end": end},
            "strategy_config_fingerprint": source.config_fingerprint,
            "joint_signal_anchor_arm_id": anchor_arm_id,
            "joint_signal": joint,
            "arms": arm_payloads,
        }
        source_refs["strategy"][profile_id] = {
            "run_dir": _relative(source.run_dir, root),
            "config_fingerprint": source.config_fingerprint,
            "arms": arm_sources,
        }

    fingerprint = _fingerprint(definition, source_refs)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_root = root / AUDIT_OUTPUT_ROOT / definition.output_subdir
    run_dir = output_root / "runs" / f"{timestamp}_{fingerprint}"
    run_dir.mkdir(parents=True, exist_ok=False)
    json_path = run_dir / "audit.json"
    report_path = run_dir / "report.md"
    manifest_path = run_dir / "manifest.json"
    cell_path = run_dir / "joint_signal_cells.csv"
    cohort_path = run_dir / "safety_cohort_mfe_rho.csv"
    capital_path = run_dir / "capital_conversion.csv"
    dd_path = run_dir / "drawdown_episodes.csv"

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "RESULT_AVAILABLE_PENDING_REVIEW",
        "module_id": definition.module_id,
        "audit_id": definition.audit_id,
        "audit_type": definition.audit_type,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_fingerprint": fingerprint,
        "decision_question": str(definition.outcomes.get("decision_question") or ""),
        "critical_uncertainty": str(definition.outcomes.get("critical_uncertainty") or ""),
        "stopping_condition": str(definition.outcomes.get("stopping_condition") or ""),
        "evaluation_order": list(profiles),
        "arm_order": list(arm_ids),
        "truth_source": truth_source,
        "evaluations": evaluations,
        "diagnostic_boundary": {
            "future_truth_runtime_use": False,
            "strategy_replay": False,
            "threshold_fit": False,
            "model_training": False,
            "joint_signal_interpretation": "upper-right HM/HS enrichment + conditional MFE rho within predicted Safety cohorts",
            "capital_interpretation": "descriptive sizing/holding conversion; correlation is not causal proof",
            "drawdown_interpretation": "portfolio peak-to-trough episode attribution, not single-stock adverse proxy",
        },
        "artifacts": {
            "Markdown": _relative(report_path, root),
            "JSON": _relative(json_path, root),
            "Joint cells CSV": _relative(cell_path, root),
            "Safety cohort rho CSV": _relative(cohort_path, root),
            "Capital CSV": _relative(capital_path, root),
            "Drawdown CSV": _relative(dd_path, root),
            "Manifest": _relative(manifest_path, root),
        },
    }
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    report_path.write_text(_render_markdown(result), encoding="utf-8")
    pd.concat(joint_cell_frames, ignore_index=True).to_csv(cell_path, index=False, encoding="utf-8-sig") if joint_cell_frames else pd.DataFrame().to_csv(cell_path, index=False, encoding="utf-8-sig")
    pd.concat(cohort_frames, ignore_index=True).to_csv(cohort_path, index=False, encoding="utf-8-sig") if cohort_frames else pd.DataFrame().to_csv(cohort_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(capital_rows).to_csv(capital_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(dd_rows).to_csv(dd_path, index=False, encoding="utf-8-sig")
    manifest = {
        "schema_version": 1,
        "audit_definition": definition.as_dict(),
        "config_fingerprint": fingerprint,
        "source_refs": source_refs,
        "artifacts": result["artifacts"],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "latest.json").write_text(json.dumps({
        "run_dir": _relative(run_dir, root),
        "report": _relative(report_path, root),
        "result": _relative(json_path, root),
        "config_fingerprint": fingerprint,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    latest = output_root / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    (latest / "audit.md").write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    (latest / "audit.json").write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    for path in (cell_path, cohort_path, capital_path, dd_path, manifest_path):
        (latest / path.name).write_bytes(path.read_bytes())
    return result


def preflight(definition: AuditDefinition, *, project_root: Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    blockers: list[str] = []
    strategy_paths: list[str] = []
    target_paths: list[str] = []
    try:
        truth, source = build_truth_geometry(
            definition,
            root,
            percentile_method=str(definition.dimensions.get("percentile_method") or "average_zero_based"),
        )
        if truth.empty:
            blockers.append("canonical MFE/Safety truth為空")
        target_paths.append(str(source.get("dataset_root") or "canonical truth"))
    except (ValueError, OSError, AuditSourceBlockedError, AuditBlockedError) as exc:
        blockers.append(str(exc))
    source_cfg = dict(definition.source)
    arm_ids = tuple(str(v) for v in source_cfg.get("strategy_arm_ids", ()))
    for profile_id in tuple(str(v) for v in source_cfg.get("evaluation_profile_ids", ())):
        try:
            pinned = str(dict(source_cfg.get("strategy_result_fingerprints") or {}).get(profile_id) or "").strip()
            source = load_strategy_compare_source(root, profile_id=profile_id, pinned_config_fingerprint=pinned or None)
            _validate_arm_contracts(source, arm_ids)
            _period_from_source(source)
            strategy_paths.append(_relative(source.run_dir, root))
            for arm_id in arm_ids:
                evidence = load_strategy_arm_path_sidecars(root, source=source, arm_id=arm_id)
                if pd.DataFrame(evidence["orderable"]).empty:
                    raise AuditBlockedError(f"{profile_id}/{arm_id} orderable sidecar為空")
                if pd.DataFrame(evidence["execution"]).empty:
                    raise AuditBlockedError(f"{profile_id}/{arm_id} execution sidecar為空")
                if pd.DataFrame(evidence["upside_realization"]).empty:
                    raise AuditBlockedError(f"{profile_id}/{arm_id} path sidecar為空")
                _load_equity(Path(evidence["pair_dir"]))
                _load_round_trips(Path(evidence["pair_dir"]))
        except (ValueError, KeyError, OSError, AuditSourceBlockedError, AuditBlockedError) as exc:
            blockers.append(str(exc))
    return {
        "status": "READY" if not blockers else "BLOCKED",
        "blockers": blockers,
        "strategy_paths": strategy_paths,
        "target_paths": target_paths,
    }


def collect_status(definition: AuditDefinition, *, project_root: Path) -> dict[str, Any]:
    state = preflight(definition, project_root=Path(project_root))
    return {
        "status": str(state.get("status") or "BLOCKED"),
        "reason": "；".join(str(v) for v in state.get("blockers", ())),
        "source": {
            "display": "Pinned C71-C74 OOS/Rolling sidecars + canonical MFE/Safety truth",
            "strategy_paths": list(state.get("strategy_paths", ())),
            "target_paths": list(state.get("target_paths", ())),
        },
    }


def run_formal_audit(
    definition: AuditDefinition,
    *,
    project_root: Path,
    quiet: bool = False,
) -> dict[str, Any]:
    result = run_audit(definition, project_root=Path(project_root))
    if not quiet:
        print(render_result(result))
    return result


__all__ = [
    "SUPPORTED_AUDIT_TYPE",
    "build_capital_conversion_analysis",
    "build_drawdown_analysis",
    "build_joint_signal_analysis",
    "collect_status",
    "preflight",
    "render_result",
    "run_audit",
    "run_formal_audit",
]
