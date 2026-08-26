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
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pandas as pd

from config.audit import AUDIT_OUTPUT_ROOT, AuditDefinition
from core.data_utils import discover_unique_csv_map, sanitize_ohlcv_dataframe
from core.dataset_profiles import get_dataset_dir
from core.exact_accounting import (
    build_buy_ledger_from_price,
    build_sell_ledger_from_price,
    coerce_money_like_to_milli,
    milli_to_money,
)
from core.price_utils import adjust_long_sell_fill_price
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


_ACCOUNTING_FIELDS = ("buy_fee", "sell_fee", "tax_rate", "min_fee")
_QUADRANT_FIELDS = {
    "high_mfe_high_safety_pct": "hmhs",
    "high_mfe_low_safety_pct": "hmls",
    "low_mfe_high_safety_pct": "lmhs",
    "low_mfe_low_safety_pct": "lmls",
}


def _read_pair_accounting_params(pair_dir: Path) -> SimpleNamespace:
    path = Path(pair_dir) / "strategy_comparison.json"
    if not path.is_file():
        raise AuditBlockedError(f"缺少既有Strategy Compare pair metadata: {path.name}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditBlockedError(f"無法讀取pair accounting metadata: {exc}") from exc
    metadata = dict(payload.get("metadata") or {}) if isinstance(payload, Mapping) else {}
    active = metadata.get("score_ranking_params")
    if not isinstance(active, Mapping):
        raise AuditBlockedError("Strategy Compare pair metadata缺少score_ranking_params")

    candidates: list[Mapping[str, Any]] = []

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            if all(field in node for field in _ACCOUNTING_FIELDS):
                candidates.append(node)
            for child in node.values():
                visit(child)
        elif isinstance(node, (list, tuple)):
            for child in node:
                visit(child)

    visit(active)
    if not candidates:
        raise AuditBlockedError("score_ranking_params找不到封存fee/tax accounting contract")
    policies: dict[tuple[float, float, float, float], Mapping[str, Any]] = {}
    for candidate in candidates:
        try:
            key = tuple(float(candidate[field]) for field in _ACCOUNTING_FIELDS)
        except (TypeError, ValueError) as exc:
            raise AuditBlockedError("score_ranking_params fee/tax contract格式無效") from exc
        policies.setdefault(key, candidate)
    if len(policies) != 1:
        raise AuditBlockedError(
            "Strategy Compare pair存在多組fee/tax accounting contract，無法做唯一MTM reconciliation"
        )
    (buy_fee, sell_fee, tax_rate, min_fee), sample = next(iter(policies.items()))
    return SimpleNamespace(
        buy_fee=buy_fee,
        sell_fee=sell_fee,
        tax_rate=tax_rate,
        min_fee=min_fee,
        fixed_risk=float(sample.get("fixed_risk", 0.01)),
    )


def _positive_int(value: Any, *, label: str) -> int:
    number = _finite(value)
    if number is None or number <= 0 or abs(number - round(number)) > 1e-9:
        raise AuditBlockedError(f"trade sidecar {label}不是正整數: {value}")
    return int(round(number))


def _build_trade_accounts(
    history: pd.DataFrame,
    *,
    pair_dir: Path,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    frame = pd.DataFrame(history).copy().reset_index(drop=True)
    round_trips = reconstruct_round_trips(frame, scenario="score_ranking")
    if round_trips.empty:
        return [], round_trips
    required = {"Date", "Ticker", "Type", "成交價", "股數"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise AuditBlockedError(f"trade sidecar缺少MTM accounting欄位: {missing}")
    params = _read_pair_accounting_params(pair_dir)
    round_trip_by_key = {
        str(row["match_key"]): row
        for row in round_trips.to_dict("records")
    }
    occurrence: dict[tuple[str, str, str], int] = {}
    open_accounts: dict[str, dict[str, Any]] = {}
    accounts: list[dict[str, Any]] = []

    for row_index, row in frame.iterrows():
        ticker = normalize_ticker(row.get("Ticker"))
        type_text = str(row.get("Type") or "").strip()
        trade_date = normalize_date(row.get("Date"))
        if not ticker or not type_text or not trade_date:
            continue
        if type_text.startswith("買進 ("):
            if ticker in open_accounts:
                raise AuditBlockedError(
                    f"trade sidecar同ticker未結算即再買進: ticker={ticker}, row={row_index}"
                )
            entry_type = str(row.get("進場類型") or "normal").strip() or "normal"
            key_base = (ticker, trade_date, entry_type)
            match_occurrence = occurrence.get(key_base, 0) + 1
            occurrence[key_base] = match_occurrence
            match_key = f"{ticker}|{trade_date}|{entry_type}|{match_occurrence}"
            round_trip = round_trip_by_key.get(match_key)
            if round_trip is None:
                raise AuditBlockedError(f"trade sidecar無法對上canonical match_key: {match_key}")
            qty = _positive_int(row.get("股數"), label="買進股數")
            entry_price = _finite(row.get("成交價"))
            if entry_price is None or entry_price <= 0:
                raise AuditBlockedError(f"trade sidecar買進成交價無效: {match_key}")
            buy_ledger = build_buy_ledger_from_price(entry_price, qty, params)
            computed_cost_milli = int(buy_ledger["net_buy_total_milli"])
            sidecar_cost = _finite(row.get("投入總金額")) if "投入總金額" in frame.columns else None
            if sidecar_cost is not None and sidecar_cost > 0:
                sidecar_cost_milli = coerce_money_like_to_milli(sidecar_cost)
                if abs(sidecar_cost_milli - computed_cost_milli) > 1:
                    raise AuditBlockedError(
                        "trade sidecar投入總金額與封存accounting contract不一致: "
                        f"{match_key} sidecar={milli_to_money(sidecar_cost_milli):.3f} "
                        f"recomputed={milli_to_money(computed_cost_milli):.3f}"
                    )
                entry_cost_milli = sidecar_cost_milli
            else:
                entry_cost_milli = computed_cost_milli
            account = {
                "match_key": match_key,
                "ticker": ticker,
                "entry_date": trade_date,
                "exit_date": normalize_date(round_trip.get("exit_date")),
                "initial_qty": qty,
                "entry_cost_milli": int(entry_cost_milli),
                "sell_events": [],
                "expected_pnl": _finite(round_trip.get("pnl")),
            }
            accounts.append(account)
            open_accounts[ticker] = account
            continue

        account = open_accounts.get(ticker)
        if account is None:
            continue
        exec_price = _finite(row.get("成交價"))
        qty_value = _finite(row.get("股數"))
        if exec_price is None or exec_price <= 0 or qty_value is None or qty_value <= 0:
            continue
        sell_qty = _positive_int(qty_value, label="賣出股數")
        sell_ledger = build_sell_ledger_from_price(
            exec_price,
            sell_qty,
            params,
            ticker=ticker,
            trade_date=trade_date,
        )
        account["sell_events"].append({
            "date": trade_date,
            "qty": sell_qty,
            "net_sell_total_milli": int(sell_ledger["net_sell_total_milli"]),
        })
        sold_qty = sum(int(event["qty"]) for event in account["sell_events"])
        if sold_qty > int(account["initial_qty"]):
            raise AuditBlockedError(f"trade sidecar賣出股數超過持倉: {account['match_key']}")
        if sold_qty == int(account["initial_qty"]):
            if trade_date != str(account["exit_date"]):
                raise AuditBlockedError(
                    f"trade sidecar cashflow結算日與canonical round-trip不一致: {account['match_key']}"
                )
            final_pnl_milli = -int(account["entry_cost_milli"]) + sum(
                int(event["net_sell_total_milli"]) for event in account["sell_events"]
            )
            expected_pnl = account.get("expected_pnl")
            if expected_pnl is not None:
                expected_milli = coerce_money_like_to_milli(expected_pnl)
                if abs(final_pnl_milli - expected_milli) > 11:
                    raise AuditBlockedError(
                        "trade sidecar final PnL與exact cashflow無法reconcile: "
                        f"{account['match_key']} sidecar={expected_pnl:.2f} "
                        f"cashflow={milli_to_money(final_pnl_milli):.3f}"
                    )
            del open_accounts[ticker]

    if open_accounts:
        raise AuditBlockedError(
            "trade sidecar結束後仍有未reconcile持倉: " + ", ".join(sorted(open_accounts)[:5])
        )
    return accounts, round_trips


def _load_close_series(
    ticker: str,
    *,
    market_csv_map: Mapping[str, str],
    market_close_cache: dict[str, pd.Series],
) -> pd.Series:
    if ticker in market_close_cache:
        return market_close_cache[ticker]
    csv_path = market_csv_map.get(ticker)
    if not csv_path:
        raise AuditBlockedError(f"canonical market data找不到ticker={ticker}")
    try:
        raw = pd.read_csv(csv_path, low_memory=False)
        cleaned, _stats = sanitize_ohlcv_dataframe(raw, ticker, min_rows=1)
    except (OSError, KeyError, ValueError) as exc:
        raise AuditBlockedError(f"canonical market data無法載入ticker={ticker}: {exc}") from exc
    close = pd.to_numeric(cleaned["Close"], errors="coerce").dropna().sort_index()
    if close.empty:
        raise AuditBlockedError(f"canonical market data沒有有效Close: ticker={ticker}")
    market_close_cache[ticker] = close
    return close


def _close_on_or_before(series: pd.Series, date_text: str, *, ticker: str) -> float:
    ts = pd.Timestamp(date_text)
    position = int(series.index.searchsorted(ts, side="right") - 1)
    if position < 0:
        raise AuditBlockedError(f"canonical market data在{date_text}以前沒有Close: ticker={ticker}")
    value = _finite(series.iloc[position])
    if value is None or value <= 0:
        raise AuditBlockedError(f"canonical market data Close無效: ticker={ticker}, date={date_text}")
    return value


def _trade_account_value_at(
    account: Mapping[str, Any],
    date_text: str,
    *,
    params: SimpleNamespace,
    market_csv_map: Mapping[str, str],
    market_close_cache: dict[str, pd.Series],
) -> dict[str, Any]:
    entry_date = str(account["entry_date"])
    if date_text < entry_date:
        return {"value_milli": 0, "remaining_qty": 0, "mark_price": None}
    value_milli = -int(account["entry_cost_milli"])
    remaining_qty = int(account["initial_qty"])
    for event in account["sell_events"]:
        if str(event["date"]) <= date_text:
            value_milli += int(event["net_sell_total_milli"])
            remaining_qty -= int(event["qty"])
    if remaining_qty < 0:
        raise AuditBlockedError(f"trade account remaining qty < 0: {account['match_key']}")
    mark_price = None
    if remaining_qty > 0:
        ticker = str(account["ticker"])
        close_series = _load_close_series(
            ticker,
            market_csv_map=market_csv_map,
            market_close_cache=market_close_cache,
        )
        close_price = _close_on_or_before(close_series, date_text, ticker=ticker)
        mark_price = adjust_long_sell_fill_price(close_price, ticker=ticker)
        liquidation = build_sell_ledger_from_price(
            mark_price,
            remaining_qty,
            params,
            ticker=ticker,
            trade_date=date_text,
        )
        value_milli += int(liquidation["net_sell_total_milli"])
    return {
        "value_milli": int(value_milli),
        "remaining_qty": int(remaining_qty),
        "mark_price": mark_price,
    }


def build_drawdown_analysis(
    evidence: Mapping[str, Any],
    truth: pd.DataFrame,
    *,
    cutoff: float,
    top_n: int,
    market_data_dir: Path,
    market_csv_map: Mapping[str, str] | None = None,
    market_close_cache: dict[str, pd.Series] | None = None,
) -> dict[str, Any]:
    pair_dir = Path(evidence["pair_dir"])
    equity, _ = _load_equity(pair_dir)
    history = pd.DataFrame(evidence["active_trades"]).copy()
    accounts, round_trips = _build_trade_accounts(history, pair_dir=pair_dir)
    path = pd.DataFrame(evidence["upside_realization"]).copy()
    if path.empty:
        raise AuditBlockedError("upside_realization sidecar為空")
    required_path = {"match_key", "score_event_date", "ticker"}
    missing_path = sorted(required_path - set(path.columns))
    if missing_path:
        raise AuditBlockedError(f"upside_realization缺少MTM attribution key: {missing_path}")
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
    if bool(path["match_key"].duplicated(keep=False).any()):
        raise AuditBlockedError("upside_realization match_key不是唯一值")
    quadrant_by_key = {
        str(row["match_key"]): str(row.get("quadrant") or "unclassified")
        for row in path.to_dict("records")
    }

    market_data_dir = Path(market_data_dir)
    if market_csv_map is None:
        try:
            discovered, _issues = discover_unique_csv_map(str(market_data_dir))
        except (OSError, ValueError) as exc:
            raise AuditBlockedError(f"canonical market data inventory失敗: {exc}") from exc
        market_csv_map = {normalize_ticker(key): value for key, value in discovered.items()}
    if market_close_cache is None:
        market_close_cache = {}
    params = _read_pair_accounting_params(pair_dir)

    all_episodes = _drawdown_episodes(equity)
    episodes = all_episodes[: int(top_n)]
    rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for rank, episode in enumerate(episodes, start=1):
        peak = str(episode["peak_date"])
        trough = str(episode["trough_date"])
        peak_equity_milli = coerce_money_like_to_milli(episode["peak_equity"])
        trough_equity_milli = coerce_money_like_to_milli(episode["trough_equity"])
        equity_change_milli = trough_equity_milli - peak_equity_milli
        entry_dates: list[str] = []
        episode_details: list[dict[str, Any]] = []
        total_contribution_milli = 0

        for account in accounts:
            peak_state = _trade_account_value_at(
                account,
                peak,
                params=params,
                market_csv_map=market_csv_map,
                market_close_cache=market_close_cache,
            )
            trough_state = _trade_account_value_at(
                account,
                trough,
                params=params,
                market_csv_map=market_csv_map,
                market_close_cache=market_close_cache,
            )
            contribution_milli = int(trough_state["value_milli"]) - int(peak_state["value_milli"])
            total_contribution_milli += contribution_milli
            entry_date = str(account["entry_date"])
            exit_date = str(account["exit_date"])
            held_at_peak = entry_date <= peak and exit_date > peak
            entered_during = peak < entry_date <= trough
            exited_during = peak < exit_date <= trough
            held_at_trough = entry_date <= trough and exit_date > trough
            relevant = held_at_peak or entered_during or exited_during or held_at_trough
            if not relevant:
                continue
            if entered_during:
                entry_dates.append(entry_date)
            quadrant = quadrant_by_key.get(str(account["match_key"]), "unclassified")
            row = {
                "episode_rank": rank,
                "peak_date": peak,
                "trough_date": trough,
                "match_key": str(account["match_key"]),
                "ticker": str(account["ticker"]),
                "entry_date": entry_date,
                "exit_date": exit_date,
                "quadrant": quadrant,
                "held_at_peak": bool(held_at_peak),
                "entered_during_drawdown": bool(entered_during),
                "exited_during_drawdown": bool(exited_during),
                "held_at_trough": bool(held_at_trough),
                "peak_remaining_qty": int(peak_state["remaining_qty"]),
                "trough_remaining_qty": int(trough_state["remaining_qty"]),
                "peak_mark_price": peak_state["mark_price"],
                "trough_mark_price": trough_state["mark_price"],
                "peak_account_value": milli_to_money(int(peak_state["value_milli"])),
                "trough_account_value": milli_to_money(int(trough_state["value_milli"])),
                "mtm_contribution": milli_to_money(contribution_milli),
                "mtm_contribution_pct_peak_equity": (
                    float(contribution_milli / peak_equity_milli * 100.0)
                    if peak_equity_milli else None
                ),
                "negative_contributor": bool(contribution_milli < 0),
            }
            episode_details.append(row)
            detail_rows.append(row)

        reconciliation_delta_milli = total_contribution_milli - equity_change_milli
        tolerance_milli = max(10, len(episode_details) * 2)
        if abs(reconciliation_delta_milli) > tolerance_milli:
            raise AuditBlockedError(
                "peak→trough position MTM無法與canonical equity reconcile: "
                f"peak={peak}, trough={trough}, "
                f"positions={milli_to_money(total_contribution_milli):.3f}, "
                f"equity={milli_to_money(equity_change_milli):.3f}, "
                f"delta={milli_to_money(reconciliation_delta_milli):.3f}"
            )

        detail_frame = pd.DataFrame(episode_details)
        entry_counts = pd.Series(entry_dates, dtype=str).value_counts()
        duration_days = int((pd.Timestamp(trough) - pd.Timestamp(peak)).days)
        summary: dict[str, Any] = {
            "episode_rank": rank,
            **{key: value for key, value in episode.items() if not key.endswith("_idx")},
            "drawdown_to_trough_calendar_days": duration_days,
            "relevant_trade_count": int(len(detail_frame)),
            "peak_held_count": int(detail_frame.get("held_at_peak", pd.Series(dtype=bool)).sum()),
            "entered_during_drawdown_count": int(detail_frame.get("entered_during_drawdown", pd.Series(dtype=bool)).sum()),
            "exited_during_drawdown_count": int(detail_frame.get("exited_during_drawdown", pd.Series(dtype=bool)).sum()),
            "trough_held_count": int(detail_frame.get("held_at_trough", pd.Series(dtype=bool)).sum()),
            "max_same_day_entries": int(entry_counts.max()) if len(entry_counts) else 0,
            "negative_contributor_count": int(detail_frame.get("negative_contributor", pd.Series(dtype=bool)).sum()),
            "equity_change": milli_to_money(equity_change_milli),
            "position_mtm_contribution_sum": milli_to_money(total_contribution_milli),
            "position_mtm_contribution_pct_peak_equity": (
                float(total_contribution_milli / peak_equity_milli * 100.0)
                if peak_equity_milli else None
            ),
            "reconciliation_delta": milli_to_money(reconciliation_delta_milli),
        }
        for quadrant, prefix in _QUADRANT_FIELDS.items():
            if detail_frame.empty:
                group = detail_frame
            else:
                group = detail_frame.loc[detail_frame["quadrant"] == quadrant]
            contribution = (
                float(pd.to_numeric(group["mtm_contribution"], errors="coerce").sum())
                if "mtm_contribution" in group.columns else 0.0
            )
            summary[f"{prefix}_count"] = int(len(group))
            summary[f"{prefix}_mtm_contribution"] = float(contribution)
            summary[f"{prefix}_mtm_contribution_pct_peak_equity"] = (
                float(contribution / float(episode["peak_equity"]) * 100.0)
                if float(episode["peak_equity"]) else None
            )
        unclassified = (
            detail_frame.loc[~detail_frame["quadrant"].isin(_QUADRANT_FIELDS)]
            if not detail_frame.empty else detail_frame
        )
        unclassified_contribution = (
            float(pd.to_numeric(unclassified["mtm_contribution"], errors="coerce").sum())
            if "mtm_contribution" in unclassified.columns else 0.0
        )
        summary["unclassified_count"] = int(len(unclassified))
        summary["unclassified_mtm_contribution"] = float(unclassified_contribution)
        rows.append(summary)

    return {
        "max_drawdown_pct": (None if not episodes else float(episodes[0]["max_drawdown_pct"])),
        "episode_count": int(len(all_episodes)),
        "top_episodes": rows,
        "position_contributions": detail_rows,
        "round_trip_count": int(len(round_trips)),
        "market_data_dir": str(market_data_dir),
        "accounting_basis": "exact trade cashflows + EOD hypothetical net liquidation at canonical Close",
        "reconciliation_required": True,
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
    payload = {"schema": 2, "definition": definition.as_dict(), "sources": source_refs}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _fmt(value: Any, digits: int = 2, suffix: str = "") -> str:
    number = _finite(value)
    return "-" if number is None else f"{number:.{digits}f}{suffix}"


def _fmt_money(value: Any, digits: int = 2) -> str:
    number = _finite(value)
    return "-" if number is None else f"{number:,.{digits}f}"


def _fmt_mtm_cell(row: Mapping[str, Any], prefix: str) -> str:
    count = int(row.get(f"{prefix}_count") or 0)
    amount = _fmt_money(row.get(f"{prefix}_mtm_contribution"))
    pct = _fmt(row.get(f"{prefix}_mtm_contribution_pct_peak_equity"), 2, "%")
    return f"N={count} / {amount} ({pct})"


def _markdown_table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def _render_joint_geometry(joint: Mapping[str, Any]) -> str:
    cells = [dict(row) for row in list(joint.get("cells") or [])]
    max_bin = max(
        [int(row.get("predicted_safety_bin", 0)) for row in cells]
        + [int(row.get("predicted_mfe_bin", 0)) for row in cells]
        + [4]
    )
    bin_count = max_bin + 1
    lookup = {
        (int(row["predicted_safety_bin"]), int(row["predicted_mfe_bin"])): row
        for row in cells
    }
    headers = ("Pred Safety \\ Cond-MFE",) + tuple(f"M{index + 1}" for index in range(bin_count))
    rows: list[tuple[Any, ...]] = []
    for safety_bin in range(bin_count):
        values: list[str] = []
        for mfe_bin in range(bin_count):
            row = lookup.get((safety_bin, mfe_bin))
            if row is None:
                values.append("0 / -")
            else:
                values.append(f"{int(row.get('rows') or 0)} / {_fmt(row.get('actual_hmhs_pct'), 2, '%')}")
        rows.append((f"S{safety_bin + 1}", *values))
    return _markdown_table(headers, rows)


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
            "Joint signal：Conditional-MFE→actual MFE dailyρ="
            + _fmt(joint.get("primary_to_actual_mfe_mean_daily_spearman"), 3)
            + "；Raw Safety→actual Safety dailyρ="
            + _fmt(joint.get("safety_to_actual_safety_mean_daily_spearman"), 3)
            + "；Joint product→actual HM/HS dailyρ="
            + _fmt(joint.get("joint_product_to_hmhs_mean_daily_spearman"), 3)
        )
        lines.append(
            "Predicted S5×M5 upper-right：N="
            + str(int(joint.get("upper_right_rows") or 0))
            + "；actual HM/HS="
            + _fmt(joint.get("upper_right_hmhs_pct"), 2, "%")
            + "；population="
            + _fmt(joint.get("population_hmhs_pct"), 2, "%")
        )
        lines.append("Predicted joint geometry（cell = N / actual HM/HS%）：")
        lines.append(_render_joint_geometry(joint))
        cohort_rows = []
        for row in list(joint.get("safety_cohorts") or []):
            item = dict(row)
            cohort_rows.append((
                f"S{int(item.get('predicted_safety_bin') or 0) + 1}",
                int(item.get("rows") or 0),
                _fmt(item.get("mean_daily_conditional_mfe_to_actual_mfe_spearman"), 3),
                int(item.get("valid_days") or 0),
                _fmt(item.get("actual_high_mfe_pct"), 2, "%"),
                _fmt(item.get("actual_hmhs_pct"), 2, "%"),
            ))
        if cohort_rows:
            lines.append("Safety cohort conditional-MFE conversion：")
            lines.append(_markdown_table(
                ("Pred Safety quintile", "N", "Cond-MFE→MFE Dailyρ", "Valid days", "High-MFE", "HM/HS"),
                cohort_rows,
            ))

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
            lines.append("Capital conversion：")
            lines.append(_markdown_table(
                ("Arm", "Avg Exposure", "Median stop dist", "Median reserved/equity", "Mean holding", "Safety→Projected capital ρ", "Safety→Stop dist ρ", "Safety→Reserved ρ"),
                capital_rows,
            ))

        dd_rows = []
        quadrant_rows = []
        for arm_id, arm in dict(payload.get("arms") or {}).items():
            dd = dict(arm.get("drawdown") or {})
            top = list(dd.get("top_episodes") or [])
            first = dict(top[0]) if top else {}
            dd_rows.append((
                arm_id,
                _fmt(dd.get("max_drawdown_pct"), 2, "%"),
                f"{first.get('peak_date', '-')}→{first.get('trough_date', '-')}",
                _fmt(first.get("position_mtm_contribution_pct_peak_equity"), 2, "%"),
                _fmt_money(first.get("reconciliation_delta"), 3),
                first.get("peak_held_count", "-"),
                first.get("entered_during_drawdown_count", "-"),
                first.get("exited_during_drawdown_count", "-"),
                first.get("trough_held_count", "-"),
                first.get("max_same_day_entries", "-"),
            ))
            quadrant_rows.append((
                arm_id,
                _fmt_mtm_cell(first, "hmhs"),
                _fmt_mtm_cell(first, "hmls"),
                _fmt_mtm_cell(first, "lmhs"),
                _fmt_mtm_cell(first, "lmls"),
                f"N={int(first.get('unclassified_count') or 0)} / {_fmt_money(first.get('unclassified_mtm_contribution'))}",
            ))
        if dd_rows:
            lines.append("Max drawdown peak→trough true MTM attribution：")
            lines.append(_markdown_table(
                ("Arm", "Max DD", "Peak→Trough", "ΣMTM / Peak", "Reconcile Δ", "Peak-held", "Entered", "Exited", "Trough-held", "Max same-day entries"),
                dd_rows,
            ))
            lines.append("Max drawdown quadrant MTM contribution（N / ΔPnL / %Peak）：")
            lines.append(_markdown_table(
                ("Arm", "HM/HS", "HM/LS", "LM/HS", "LM/LS", "Unclassified"),
                quadrant_rows,
            ))
        lines.append("")
    return "\n".join(lines).rstrip()


def _render_markdown(result: Mapping[str, Any]) -> str:
    return "# MR-13R Joint Signal / Capital / Drawdown Audit\n\n```text\n" + render_result(result) + "\n```\n\n" \
        "## 判讀邊界\n\n" \
        "- Joint signal 使用完成 replay 後才 join 的 canonical future truth，僅供診斷，不參與任何選股。\n" \
        "- Capital conversion 是描述性 mechanism attribution；Safety 與 sizing/holding 的相關不等於單一因果證明。\n" \
        "- Drawdown contribution 是 canonical equity peak EOD→trough EOD 的真實 position MTM Δ：既有實際買賣 cashflow + 當日 canonical Close 假設淨清算，並硬性 reconcile 至 portfolio Equity Δ。\n" \
        "- 交易在 trough 後的最終 realized R/PnL 不作 drawdown contribution；future truth quadrant 只在 contribution 計算完成後做 post-replay 分組。\n"


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
    dd_detail_rows: list[dict[str, Any]] = []
    for profile_id in profiles:
        pinned = str(dict(source_cfg.get("strategy_result_fingerprints") or {}).get(profile_id) or "").strip()
        source = load_strategy_compare_source(root, profile_id=profile_id, pinned_config_fingerprint=pinned or None)
        _validate_arm_contracts(source, arm_ids)
        start, end = _period_from_source(source)
        period_truth = filter_period(truth, start, end)
        market_data_dir = Path(get_dataset_dir(str(root), str(source.settings.dataset))).resolve()
        if not market_data_dir.is_dir():
            raise AuditBlockedError(
                f"canonical market data目錄不存在: {_relative(market_data_dir, root)}"
            )
        try:
            discovered_csv_map, _duplicate_issues = discover_unique_csv_map(str(market_data_dir))
        except (OSError, ValueError) as exc:
            raise AuditBlockedError(f"canonical market data inventory失敗: {exc}") from exc
        market_csv_map = {
            normalize_ticker(ticker): path for ticker, path in discovered_csv_map.items()
        }
        market_close_cache: dict[str, pd.Series] = {}
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
            drawdown = build_drawdown_analysis(
                evidence,
                period_truth,
                cutoff=cutoff,
                top_n=top_n,
                market_data_dir=market_data_dir,
                market_csv_map=market_csv_map,
                market_close_cache=market_close_cache,
            )
            contribution_detail = list(drawdown.pop("position_contributions", ()))
            drawdown["market_data_dir"] = _relative(Path(drawdown["market_data_dir"]), root)
            arm_payloads[arm_id] = {"capital_conversion": capital, "drawdown": drawdown}
            capital_rows.append({"profile_id": profile_id, "arm_id": arm_id, **capital})
            for row in drawdown["top_episodes"]:
                dd_rows.append({"profile_id": profile_id, "arm_id": arm_id, **row})
            for row in contribution_detail:
                dd_detail_rows.append({"profile_id": profile_id, "arm_id": arm_id, **row})
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
            "market_data_dir": _relative(market_data_dir, root),
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
    dd_detail_path = run_dir / "drawdown_position_contributions.csv"

    result: dict[str, Any] = {
        "schema_version": 2,
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
            "drawdown_interpretation": (
                "canonical peak-EOD to trough-EOD exact position MTM contribution; "
                "actual transaction cashflows + hypothetical net liquidation at canonical Close; "
                "must reconcile to portfolio Equity delta; final realized R is not drawdown contribution"
            ),
        },
        "artifacts": {
            "Markdown": _relative(report_path, root),
            "JSON": _relative(json_path, root),
            "Joint cells CSV": _relative(cell_path, root),
            "Safety cohort rho CSV": _relative(cohort_path, root),
            "Capital CSV": _relative(capital_path, root),
            "Drawdown CSV": _relative(dd_path, root),
            "Drawdown position contributions CSV": _relative(dd_detail_path, root),
            "Manifest": _relative(manifest_path, root),
        },
    }
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    report_path.write_text(_render_markdown(result), encoding="utf-8")
    pd.concat(joint_cell_frames, ignore_index=True).to_csv(cell_path, index=False, encoding="utf-8-sig") if joint_cell_frames else pd.DataFrame().to_csv(cell_path, index=False, encoding="utf-8-sig")
    pd.concat(cohort_frames, ignore_index=True).to_csv(cohort_path, index=False, encoding="utf-8-sig") if cohort_frames else pd.DataFrame().to_csv(cohort_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(capital_rows).to_csv(capital_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(dd_rows).to_csv(dd_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(dd_detail_rows).to_csv(dd_detail_path, index=False, encoding="utf-8-sig")
    manifest = {
        "schema_version": 2,
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
    for path in (cell_path, cohort_path, capital_path, dd_path, dd_detail_path, manifest_path):
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
            market_data_dir = Path(get_dataset_dir(str(root), str(source.settings.dataset))).resolve()
            if not market_data_dir.is_dir():
                raise AuditBlockedError(
                    f"canonical market data目錄不存在: {_relative(market_data_dir, root)}"
                )
            target_paths.append(_relative(market_data_dir, root))
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
                _read_pair_accounting_params(Path(evidence["pair_dir"]))
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
