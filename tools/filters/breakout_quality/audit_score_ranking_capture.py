"""Read-only capital deployment and Target-capture audit for PIT Score Sort replay."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from tools.filters.breakout_quality.strategy_report_style import (
    SIGNAL_NEGATIVE,
    SIGNAL_NEUTRAL,
    SIGNAL_POSITIVE,
    SIGNAL_WARNING,
    finite_number,
    html_delta_cell,
    html_page,
    html_signal_badge,
    signal_for_delta,
    signal_marker,
)

SCHEMA_VERSION = 1
_BUY_PREFIX = "買進 ("
_MISSED_BUY_PREFIX = "錯失買進"
_FULL_EXIT_PREFIXES = ("全倉結算", "汰弱賣出", "期末強制結算")
_PARTIAL_EXIT_PREFIX = "半倉停利"


def _clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _date_text(value: Any) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    parsed = pd.to_datetime(text, errors="coerce")
    return "" if pd.isna(parsed) else parsed.strftime("%Y-%m-%d")


def _number(value: Any) -> float | None:
    if isinstance(value, str):
        value = value.replace(",", "").strip()
    return finite_number(value)


def _is_buy(type_text: str) -> bool:
    return str(type_text).startswith(_BUY_PREFIX)


def _is_full_exit(type_text: str) -> bool:
    return any(str(type_text).startswith(prefix) for prefix in _FULL_EXIT_PREFIXES)


def _target_lookup(selected_target_diagnostics: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(selected_target_diagnostics).copy()
    columns = ["ticker", "trade_date", "signal_date", "score_event_date", "target_raw_r"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    for column in ("ticker", "trade_date", "signal_date", "score_event_date"):
        if column not in frame.columns:
            frame[column] = ""
    frame["ticker"] = frame["ticker"].fillna("").astype(str).str.strip()
    for column in ("trade_date", "signal_date", "score_event_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    frame["target_raw_r"] = pd.to_numeric(frame.get("target_raw_r"), errors="coerce")
    frame = frame[columns].drop_duplicates(["ticker", "trade_date", "signal_date"], keep="first")
    return frame.rename(columns={"trade_date": "entry_date"})


def _trading_calendar_dates(daily_capacity: pd.DataFrame | None) -> pd.DatetimeIndex:
    frame = pd.DataFrame(daily_capacity).copy()
    if frame.empty or "Date" not in frame.columns:
        return pd.DatetimeIndex([])
    dates = pd.to_datetime(frame["Date"], errors="coerce").dropna().drop_duplicates().sort_values()
    return pd.DatetimeIndex(dates)


def build_trade_lifecycle_rows(
    trade_history: pd.DataFrame,
    *,
    scenario: str,
    selected_target_diagnostics: pd.DataFrame | None = None,
    daily_capacity: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one row per completed position from canonical transaction history."""

    frame = pd.DataFrame(trade_history).copy()
    columns = [
        "scenario", "ticker", "entry_date", "exit_date", "entry_year", "entry_type",
        "candidate_type", "industry", "signal_date", "score_event_date", "quality_score",
        "entry_price", "initial_stop_price", "stop_distance_pct", "qty",
        "reserved_total", "invested_total", "invested_vs_reserved_pct",
        "partial_exit_count", "first_partial_date", "days_to_first_partial",
        "partial_to_exit_calendar_days", "partial_residual_slot_days",
        "holding_calendar_days", "exit_type", "pnl",
        "r_multiple", "target_raw_r", "target_capture_ratio", "target_realization_gap_r",
        "capital_return_pct",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    missing = sorted({"Date", "Ticker", "Type"} - set(frame.columns))
    if missing:
        raise ValueError(f"capture audit trade history缺少欄位: {missing}")

    open_positions: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    ordered = frame.reset_index(drop=True)
    trading_dates = _trading_calendar_dates(daily_capacity)
    for row_index, raw in ordered.iterrows():
        ticker = _clean_text(raw.get("Ticker"))
        type_text = _clean_text(raw.get("Type"))
        if not ticker or not type_text:
            continue
        event_date = _date_text(raw.get("Date"))

        if _is_buy(type_text):
            if ticker in open_positions:
                raise ValueError(
                    "capture audit同一ticker尚未結算即再次買進: "
                    f"scenario={scenario}, ticker={ticker}, row={row_index}"
                )
            entry_price = _number(raw.get("成交價"))
            stop_price = _number(raw.get("停損價"))
            reserved_total = _number(raw.get("預留總金額"))
            invested_total = _number(raw.get("投入總金額"))
            score_event_date = _date_text(raw.get("Quality Score Date"))
            signal_date = _date_text(raw.get("買訊日"))
            stop_distance_pct = None
            if entry_price and entry_price > 0 and stop_price is not None:
                stop_distance_pct = (entry_price - stop_price) / entry_price * 100.0
            invested_vs_reserved_pct = None
            if reserved_total and reserved_total > 0 and invested_total is not None:
                invested_vs_reserved_pct = invested_total / reserved_total * 100.0
            open_positions[ticker] = {
                "scenario": str(scenario),
                "ticker": ticker,
                "entry_date": event_date,
                "entry_year": int(pd.Timestamp(event_date).year) if event_date else None,
                "entry_type": _clean_text(raw.get("進場類型")) or "normal",
                "candidate_type": _clean_text(raw.get("候選類型")),
                "industry": next(
                    (
                        value
                        for value in (
                            _clean_text(raw.get("產業")),
                            _clean_text(raw.get("Industry")),
                            _clean_text(raw.get("Sector")),
                            _clean_text(raw.get("sector")),
                        )
                        if value
                    ),
                    "",
                ),
                "signal_date": signal_date,
                "score_event_date": score_event_date or signal_date,
                "quality_score": _number(raw.get("Quality Score")),
                "entry_price": entry_price,
                "initial_stop_price": stop_price,
                "stop_distance_pct": stop_distance_pct,
                "qty": _number(raw.get("股數")),
                "reserved_total": reserved_total,
                "invested_total": invested_total,
                "invested_vs_reserved_pct": invested_vs_reserved_pct,
                "partial_exit_count": 0,
                "first_partial_date": "",
            }
            continue

        if type_text.startswith(_PARTIAL_EXIT_PREFIX):
            position = open_positions.get(ticker)
            if position is not None:
                position["partial_exit_count"] = int(position["partial_exit_count"]) + 1
                if not position["first_partial_date"]:
                    position["first_partial_date"] = event_date
            continue

        if not _is_full_exit(type_text):
            continue
        position = open_positions.pop(ticker, None)
        if position is None:
            raise ValueError(
                "capture audit結算列找不到買進: "
                f"scenario={scenario}, ticker={ticker}, type={type_text}, row={row_index}"
            )
        entry_ts = pd.Timestamp(position["entry_date"])
        exit_ts = pd.Timestamp(event_date)
        partial_date = position["first_partial_date"]
        partial_ts = pd.Timestamp(partial_date) if partial_date else None
        partial_residual_slot_days = None
        if partial_ts is not None and len(trading_dates) > 0:
            partial_residual_slot_days = int(
                ((trading_dates >= partial_ts) & (trading_dates < exit_ts)).sum()
            )
        invested_total = position.get("invested_total")
        pnl = _number(raw.get("該筆總損益"))
        if pnl is None:
            pnl = _number(raw.get("單筆損益"))
        capital_return_pct = None
        if invested_total and invested_total > 0 and pnl is not None:
            capital_return_pct = pnl / invested_total * 100.0
        rows.append({
            **position,
            "exit_date": event_date,
            "days_to_first_partial": int((partial_ts - entry_ts).days) if partial_ts is not None else None,
            "partial_to_exit_calendar_days": int((exit_ts - partial_ts).days) if partial_ts is not None else 0,
            "partial_residual_slot_days": partial_residual_slot_days if partial_ts is not None else 0,
            "holding_calendar_days": int((exit_ts - entry_ts).days),
            "exit_type": type_text,
            "pnl": pnl,
            "r_multiple": _number(raw.get("R_Multiple")),
            "capital_return_pct": capital_return_pct,
        })
    if open_positions:
        sample = sorted(open_positions)[:5]
        raise ValueError(
            f"capture audit結束後仍有未結算持倉: scenario={scenario}, count={len(open_positions)}, sample={sample}"
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=columns)
    target = _target_lookup(pd.DataFrame(selected_target_diagnostics))
    out = out.merge(
        target,
        on=["ticker", "entry_date", "signal_date"],
        how="left",
        validate="many_to_one",
    ) if not target.empty else out.assign(target_raw_r=float("nan"))
    out["target_raw_r"] = pd.to_numeric(out["target_raw_r"], errors="coerce")
    out["r_multiple"] = pd.to_numeric(out["r_multiple"], errors="coerce")
    positive_target = out["target_raw_r"].map(lambda value: math.isfinite(value) and value > 0)
    out["target_capture_ratio"] = float("nan")
    out.loc[positive_target, "target_capture_ratio"] = (
        out.loc[positive_target, "r_multiple"] / out.loc[positive_target, "target_raw_r"]
    )
    out["target_realization_gap_r"] = out["target_raw_r"] - out["r_multiple"]
    return out.reindex(columns=columns)


def _safe_mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce")
    values = values[values.map(math.isfinite)]
    return float(values.mean()) if not values.empty else None


def _safe_median(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce")
    values = values[values.map(math.isfinite)]
    return float(values.median()) if not values.empty else None


def _concentration(series: pd.Series) -> tuple[float | None, float | None]:
    values = series.fillna("").astype(str).str.strip()
    values = values[values != ""]
    if values.empty:
        return None, None
    shares = values.value_counts(normalize=True)
    return float(shares.max() * 100.0), float((shares.pow(2)).sum())


def _top_n_share(series: pd.Series, n: int) -> float | None:
    values = series.fillna("").astype(str).str.strip()
    values = values[values != ""]
    if values.empty:
        return None
    counts = values.value_counts()
    return float(counts.nlargest(int(n)).sum() / counts.sum() * 100.0)


def _scenario_summary(
    *,
    trade_history: pd.DataFrame,
    lifecycle: pd.DataFrame,
    strategy_summary: dict[str, Any],
) -> dict[str, Any]:
    history = pd.DataFrame(trade_history)
    type_values = history.get("Type", pd.Series("", index=history.index)).fillna("").astype(str)
    buy_count = int(type_values.str.startswith(_BUY_PREFIX).sum())
    missed_buy_count = int(type_values.str.startswith(_MISSED_BUY_PREFIX).sum())
    partial = lifecycle[pd.to_numeric(lifecycle.get("partial_exit_count"), errors="coerce").fillna(0) > 0]
    target_covered = pd.to_numeric(lifecycle.get("target_raw_r"), errors="coerce").map(math.isfinite)
    exit_types = lifecycle.get("exit_type", pd.Series("", index=lifecycle.index)).fillna("").astype(str)
    entry_dates = lifecycle.get("entry_date", pd.Series("", index=lifecycle.index))
    entry_months = pd.to_datetime(entry_dates, errors="coerce").dt.strftime("%Y-%m").fillna("")
    industry_values = lifecycle.get("industry", pd.Series("", index=lifecycle.index)).fillna("").astype(str).str.strip()
    month_top_share, month_hhi = _concentration(entry_months)
    industry_top_share, industry_hhi = _concentration(industry_values)
    industry_covered = int((industry_values != "").sum())
    return {
        "trade_count": int(len(lifecycle)),
        "top_5_entry_dates_share_pct": _top_n_share(entry_dates, 5),
        "top_entry_month_share_pct": month_top_share,
        "entry_month_hhi": month_hhi,
        "industry_covered_trades": industry_covered,
        "industry_coverage_pct": (
            float(industry_covered / len(lifecycle) * 100.0) if len(lifecycle) else None
        ),
        "top_industry_share_pct": industry_top_share,
        "industry_hhi": industry_hhi,
        "buy_count": buy_count,
        "missed_buy_count": missed_buy_count,
        "reserved_buy_fill_rate_pct": float(buy_count / (buy_count + missed_buy_count) * 100.0) if (buy_count + missed_buy_count) else None,
        "avg_exposure_pct": finite_number(strategy_summary.get("avg_exposure_pct")),
        "total_return_pct": finite_number(strategy_summary.get("total_return_pct")),
        "max_drawdown_pct": finite_number(strategy_summary.get("max_drawdown_pct")),
        "return_over_max_drawdown": finite_number(strategy_summary.get("return_over_max_drawdown")),
        "avg_reserved_total": _safe_mean(lifecycle.get("reserved_total", pd.Series(dtype=float))),
        "avg_invested_total": _safe_mean(lifecycle.get("invested_total", pd.Series(dtype=float))),
        "median_invested_total": _safe_median(lifecycle.get("invested_total", pd.Series(dtype=float))),
        "avg_invested_vs_reserved_pct": _safe_mean(lifecycle.get("invested_vs_reserved_pct", pd.Series(dtype=float))),
        "avg_stop_distance_pct": _safe_mean(lifecycle.get("stop_distance_pct", pd.Series(dtype=float))),
        "avg_holding_calendar_days": _safe_mean(lifecycle.get("holding_calendar_days", pd.Series(dtype=float))),
        "median_holding_calendar_days": _safe_median(lifecycle.get("holding_calendar_days", pd.Series(dtype=float))),
        "partial_exit_trade_share_pct": float(len(partial) / len(lifecycle) * 100.0) if len(lifecycle) else None,
        "avg_days_to_first_partial": _safe_mean(partial.get("days_to_first_partial", pd.Series(dtype=float))),
        "avg_partial_to_exit_calendar_days": _safe_mean(
            partial.get("partial_to_exit_calendar_days", pd.Series(dtype=float))
        ),
        "partial_residual_slot_days": (
            int(pd.to_numeric(partial.get("partial_residual_slot_days"), errors="coerce").sum())
            if not partial.empty
            and pd.to_numeric(partial.get("partial_residual_slot_days"), errors="coerce").notna().any()
            else None
        ),
        "stop_exit_share_pct": float(exit_types.str.startswith("全倉結算(停損)").mean() * 100.0) if len(exit_types) else None,
        "indicator_exit_share_pct": float(exit_types.str.startswith("全倉結算(指標)").mean() * 100.0) if len(exit_types) else None,
        "rotation_exit_share_pct": float(exit_types.str.startswith("汰弱賣出").mean() * 100.0) if len(exit_types) else None,
        "forced_exit_share_pct": float(exit_types.str.startswith("期末強制結算").mean() * 100.0) if len(exit_types) else None,
        "avg_realized_r": _safe_mean(lifecycle.get("r_multiple", pd.Series(dtype=float))),
        "median_realized_r": _safe_median(lifecycle.get("r_multiple", pd.Series(dtype=float))),
        "target_covered_trades": int(target_covered.sum()),
        "avg_target_r": _safe_mean(lifecycle.loc[target_covered, "target_raw_r"]),
        "avg_target_capture_ratio": _safe_mean(lifecycle.loc[target_covered, "target_capture_ratio"]),
        "median_target_capture_ratio": _safe_median(lifecycle.loc[target_covered, "target_capture_ratio"]),
        "avg_target_realization_gap_r": _safe_mean(lifecycle.loc[target_covered, "target_realization_gap_r"]),
        "avg_capital_return_pct": _safe_mean(lifecycle.get("capital_return_pct", pd.Series(dtype=float))),
    }


def _delta(right: dict[str, Any], left: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in sorted(set(left) | set(right)):
        lv = finite_number(left.get(key))
        rv = finite_number(right.get(key))
        if lv is not None and rv is not None:
            result[key] = rv - lv
    return result


def _yearly_rows(lifecycle: pd.DataFrame, scenario: str) -> pd.DataFrame:
    if lifecycle.empty:
        return pd.DataFrame(columns=[
            "entry_year", f"{scenario}_trade_count", f"{scenario}_avg_r",
            f"{scenario}_avg_target_r", f"{scenario}_avg_capture_ratio",
            f"{scenario}_avg_invested_total", f"{scenario}_avg_holding_days",
        ])
    rows = []
    for year, group in lifecycle.groupby("entry_year", dropna=True, sort=True):
        covered = pd.to_numeric(group["target_raw_r"], errors="coerce").map(math.isfinite)
        rows.append({
            "entry_year": int(year),
            f"{scenario}_trade_count": int(len(group)),
            f"{scenario}_avg_r": _safe_mean(group["r_multiple"]),
            f"{scenario}_avg_target_r": _safe_mean(group.loc[covered, "target_raw_r"]),
            f"{scenario}_avg_capture_ratio": _safe_mean(group.loc[covered, "target_capture_ratio"]),
            f"{scenario}_avg_invested_total": _safe_mean(group["invested_total"]),
            f"{scenario}_avg_holding_days": _safe_mean(group["holding_calendar_days"]),
        })
    return pd.DataFrame(rows)


def _build_yearly(baseline_lifecycle: pd.DataFrame, score_lifecycle: pd.DataFrame) -> pd.DataFrame:
    left = _yearly_rows(baseline_lifecycle, "baseline")
    right = _yearly_rows(score_lifecycle, "score_sort")
    merged = left.merge(right, on="entry_year", how="outer", validate="one_to_one").sort_values("entry_year")
    for metric in ("trade_count", "avg_r", "avg_target_r", "avg_capture_ratio", "avg_invested_total", "avg_holding_days"):
        merged[f"delta_{metric}"] = merged[f"score_sort_{metric}"] - merged[f"baseline_{metric}"]
    return merged.reset_index(drop=True)


def _decision(
    *,
    baseline: dict[str, Any],
    score_sort: dict[str, Any],
    delta: dict[str, Any],
    selection_diagnostics: dict[str, Any] | None,
) -> dict[str, Any]:
    diagnostics = dict(selection_diagnostics or {})
    target_delta = dict(diagnostics.get("score_ranking_minus_no_filter") or {})
    target_improved = (
        finite_number(target_delta.get("selected_target_mean_r")) is not None
        and float(target_delta["selected_target_mean_r"]) > 0
        and finite_number(target_delta.get("target_top_k_retention_mean")) is not None
        and float(target_delta["target_top_k_retention_mean"]) > 0
    )
    economic_failed = (
        (finite_number(delta.get("total_return_pct")) or 0.0) < 0
        and (finite_number(delta.get("return_over_max_drawdown")) or 0.0) < 0
    )
    deployment_gap = (finite_number(delta.get("avg_exposure_pct")) or 0.0) <= -5.0
    capture_gap = (
        (finite_number(delta.get("avg_target_capture_ratio")) or 0.0) <= -0.05
        or (finite_number(delta.get("avg_target_realization_gap_r")) or 0.0) >= 0.10
    )
    turnover_gap = (
        (finite_number(delta.get("avg_holding_calendar_days")) or 0.0) >= 3.0
        or (finite_number(delta.get("avg_partial_to_exit_calendar_days")) or 0.0) >= 3.0
    )
    fill_gap = (finite_number(delta.get("reserved_buy_fill_rate_pct")) or 0.0) <= -2.0
    sizing_gap = (
        finite_number(baseline.get("avg_invested_total")) not in {None, 0.0}
        and finite_number(score_sort.get("avg_invested_total")) is not None
        and float(score_sort["avg_invested_total"]) < float(baseline["avg_invested_total"]) * 0.90
    )
    bottlenecks = []
    if deployment_gap:
        bottlenecks.append("平均曝險明顯下降")
    if sizing_gap:
        bottlenecks.append("每筆實際投入金額下降")
    if capture_gap:
        bottlenecks.append("Target轉成Realized R的capture惡化")
    if turnover_gap:
        bottlenecks.append("持有／半倉殘留時間拉長")
    if fill_gap:
        bottlenecks.append("保留買單成交率下降")
    if not bottlenecks:
        bottlenecks.append("未由目前可觀測欄位確認單一機械瓶頸")
    adaptation_candidate = bool(target_improved and economic_failed and any((deployment_gap, sizing_gap, capture_gap, turnover_gap, fill_gap)))
    if adaptation_candidate:
        status = "ADAPTATION_DIAGNOSTIC_SUPPORTED"
        signal = SIGNAL_WARNING
        conclusion = "模型確實改善Target選擇，但舊參數下經濟效果失敗，且已找到可由策略參數適應檢驗的資本／capture瓶頸。"
    elif target_improved and economic_failed:
        status = "SORT_ONLY_REJECTED_NO_MECHANICAL_BOTTLENECK"
        signal = SIGNAL_NEGATIVE
        conclusion = "模型改善Target選擇，但舊參數下經濟效果失敗；目前audit不足以支持直接進入參數適應。"
    elif not economic_failed:
        status = "SORT_ONLY_NOT_REJECTED"
        signal = SIGNAL_POSITIVE
        conclusion = "Score Sort經濟效果未被本audit判定失敗；仍須依正式OOS契約決定是否採用。"
    else:
        status = "SORT_ONLY_REJECTED"
        signal = SIGNAL_NEGATIVE
        conclusion = "Score Sort經濟效果失敗，且Target改善證據不足。"
    return {
        "status": status,
        "signal": signal,
        "target_selection_improved": target_improved,
        "economic_effect_failed": economic_failed,
        "parameter_adaptation_candidate": adaptation_candidate,
        "bottlenecks": bottlenecks,
        "conclusion": conclusion,
        "future_target_used_for_runtime": False,
    }


def build_score_ranking_capture_audit(
    *,
    metadata: dict[str, Any],
    baseline_summary: dict[str, Any],
    score_sort_summary: dict[str, Any],
    baseline_trade_history: pd.DataFrame,
    score_sort_trade_history: pd.DataFrame,
    baseline_selected_target_diagnostics: pd.DataFrame | None,
    score_sort_selected_target_diagnostics: pd.DataFrame | None,
    selection_diagnostics: dict[str, Any] | None,
    baseline_daily_capacity: pd.DataFrame | None = None,
    score_sort_daily_capacity: pd.DataFrame | None = None,
) -> dict[str, Any]:
    baseline_lifecycle = build_trade_lifecycle_rows(
        baseline_trade_history,
        scenario="baseline",
        selected_target_diagnostics=baseline_selected_target_diagnostics,
        daily_capacity=baseline_daily_capacity,
    )
    score_lifecycle = build_trade_lifecycle_rows(
        score_sort_trade_history,
        scenario="score_sort",
        selected_target_diagnostics=score_sort_selected_target_diagnostics,
        daily_capacity=score_sort_daily_capacity,
    )
    baseline = _scenario_summary(
        trade_history=baseline_trade_history,
        lifecycle=baseline_lifecycle,
        strategy_summary=baseline_summary,
    )
    score_sort = _scenario_summary(
        trade_history=score_sort_trade_history,
        lifecycle=score_lifecycle,
        strategy_summary=score_sort_summary,
    )
    delta = _delta(score_sort, baseline)
    yearly = _build_yearly(baseline_lifecycle, score_lifecycle)
    decision = _decision(
        baseline=baseline,
        score_sort=score_sort,
        delta=delta,
        selection_diagnostics=selection_diagnostics,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": dict(metadata),
        "baseline": baseline,
        "score_sort": score_sort,
        "score_sort_minus_baseline": delta,
        "decision": decision,
        "yearly": yearly,
        "baseline_lifecycle": baseline_lifecycle,
        "score_sort_lifecycle": score_lifecycle,
    }


def _fmt(value: Any, digits: int = 2, unit: str = "", signed: bool = False) -> str:
    number = finite_number(value)
    if number is None:
        return "N/A"
    sign = "+" if signed else ""
    return f"{number:{sign}.{digits}f}{unit}"


_CAPTURE_ROWS = (
    ("平均曝險", "avg_exposure_pct", "%", "higher"),
    ("保留買單成交率", "reserved_buy_fill_rate_pct", "%", "higher"),
    ("平均預留金額", "avg_reserved_total", "", "attention"),
    ("平均實際投入金額", "avg_invested_total", "", "higher"),
    ("投入／預留比", "avg_invested_vs_reserved_pct", "%", "higher"),
    ("平均初始停損距離", "avg_stop_distance_pct", "%", "attention"),
    ("平均持有日", "avg_holding_calendar_days", " 日", "lower"),
    ("半倉交易占比", "partial_exit_trade_share_pct", "%", "attention"),
    ("半倉後至結算平均日曆日", "avg_partial_to_exit_calendar_days", " 日", "lower"),
    ("半倉殘留交易slot-days", "partial_residual_slot_days", " 格日", "lower"),
    ("平均 Realized R", "avg_realized_r", " R", "higher"),
    ("平均 Target R", "avg_target_r", " R", "higher"),
    ("平均 Target capture ratio", "avg_target_capture_ratio", "", "higher"),
    ("平均 Target realization gap", "avg_target_realization_gap_r", " R", "lower"),
    ("平均投入資金報酬", "avg_capital_return_pct", "%", "higher"),
    ("Top 5進場日交易占比", "top_5_entry_dates_share_pct", "%", "attention"),
    ("最大單月進場占比", "top_entry_month_share_pct", "%", "attention"),
    ("進場月份 HHI", "entry_month_hhi", "", "attention"),
    ("產業資料覆蓋", "industry_coverage_pct", "%", "neutral"),
    ("最大產業占比", "top_industry_share_pct", "%", "attention"),
    ("產業 HHI", "industry_hhi", "", "attention"),
    ("停損結算占比", "stop_exit_share_pct", "%", "attention"),
    ("指標結算占比", "indicator_exit_share_pct", "%", "attention"),
    ("汰弱賣出占比", "rotation_exit_share_pct", "%", "attention"),
)


def render_capture_audit_markdown(result: dict[str, Any]) -> str:
    metadata = result["metadata"]
    baseline = result["baseline"]
    score_sort = result["score_sort"]
    delta = result["score_sort_minus_baseline"]
    decision = result["decision"]
    lines = [
        "# Breakout Quality Score 排序資本效率／Target Capture 歸因", "",
        f"- 期間：`{metadata.get('comparison_period', {}).get('start', '')}` ～ `{metadata.get('comparison_period', {}).get('end', '')}`",
        f"- Score source：`{metadata.get('score_source', '')}`",
        "- 所有 Future Target 只在策略回放完成後離線 join；未進入排序、資金配置或成交決策。",
        "- 顏色判讀口徑：綠色代表依本audit目的改善、紅色代表惡化、黃色代表方向本身無單一好壞但變化值得注意。",
        "", "## 綜合判定", "",
        f"> {signal_marker(decision['signal'])} **{decision['status']}** — {decision['conclusion']}",
        "",
        "主要瓶頸：" + "；".join(decision.get("bottlenecks") or []),
        "", "## 資本使用與 Capture", "",
        "| 指標 | Baseline | Score Sort | 差異 | 判讀 |",
        "|---|---:|---:|---:|:---:|",
    ]
    for label, key, unit, preference in _CAPTURE_ROWS:
        dv = delta.get(key)
        signal = signal_for_delta(dv, preference=preference, warning_threshold=0.0)
        digits = 0 if key in {"partial_residual_slot_days"} else 2
        lines.append(
            f"| {label} | {_fmt(baseline.get(key), digits, unit)} | {_fmt(score_sort.get(key), digits, unit)} "
            f"| {_fmt(dv, digits, unit, signed=True)} | {signal_marker(signal)} |"
        )
    lines += [
        "", "## Exit reason", "",
        "| 類別 | Baseline | Score Sort | 差異 | 判讀 |",
        "|---|---:|---:|---:|:---:|",
    ]
    for label, key in (
        ("停損", "stop_exit_share_pct"), ("指標", "indicator_exit_share_pct"),
        ("汰弱", "rotation_exit_share_pct"), ("期末強制", "forced_exit_share_pct"),
    ):
        exit_signal = signal_for_delta(delta.get(key), preference="attention", warning_threshold=0.0)
        lines.append(
            f"| {label} | {_fmt(baseline.get(key), 2, '%')} | {_fmt(score_sort.get(key), 2, '%')} "
            f"| {_fmt(delta.get(key), 2, '%', signed=True)} | {signal_marker(exit_signal)} |"
        )
    yearly = result["yearly"]
    lines += ["", "## 依進場年度", ""]
    if yearly.empty:
        lines.append("無年度資料。")
    else:
        lines += [
            "| 年度 | Baseline avg R | Score Sort avg R | ΔR | Baseline capture | Score Sort capture | Δcapture | Baseline投入 | Score Sort投入 | Δ投入 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in yearly.to_dict("records"):
            r_signal = signal_for_delta(row.get("delta_avg_r"), preference="higher")
            capture_signal = signal_for_delta(
                row.get("delta_avg_capture_ratio"), preference="higher"
            )
            invested_signal = signal_for_delta(
                row.get("delta_avg_invested_total"), preference="higher"
            )
            lines.append(
                f"| {int(row['entry_year'])} | {_fmt(row.get('baseline_avg_r'))} | {_fmt(row.get('score_sort_avg_r'))} "
                f"| {_fmt(row.get('delta_avg_r'), signed=True)} {signal_marker(r_signal, include_label=False)} "
                f"| {_fmt(row.get('baseline_avg_capture_ratio'))} | {_fmt(row.get('score_sort_avg_capture_ratio'))} "
                f"| {_fmt(row.get('delta_avg_capture_ratio'), signed=True)} {signal_marker(capture_signal, include_label=False)} "
                f"| {_fmt(row.get('baseline_avg_invested_total'), 0)} | {_fmt(row.get('score_sort_avg_invested_total'), 0)} "
                f"| {_fmt(row.get('delta_avg_invested_total'), 0, signed=True)} {signal_marker(invested_signal, include_label=False)} |"
            )
    lines += [
        "", "## 使用限制", "",
        "- 本報表是Selection內的read-only attribution，不改模型、Score、排序或策略參數。",
        "- Target capture ratio = realized round-trip R / selected No-time Target R；只在Target為正且可對應時統計。",
        "- 半倉後至結算平均日使用日曆日；半倉殘留slot-days則以該scenario每日capacity的交易日期，計算partial日（含）到full-exit日（不含）的尾倉占位。",
        "- 產業集中度只在交易買進列已有`產業`／`Industry`／`Sector`欄位時統計；沒有canonical產業欄位時顯示N/A，不自行推測類股。",
        "- 黃色指標（例如停損距離、集中度、exit mix）僅表示結構變化，不能單獨視為好或壞。",
        "- 是否進入主策略參數適應，以 `parameter_adaptation_candidate` 與具體瓶頸共同判讀；不得使用OOS擬合參數。",
        "",
    ]
    return "\n".join(lines)


def render_capture_audit_html(result: dict[str, Any]) -> str:
    metadata = result["metadata"]
    baseline = result["baseline"]
    score_sort = result["score_sort"]
    delta = result["score_sort_minus_baseline"]
    decision = result["decision"]
    rows = []
    for label, key, unit, preference in _CAPTURE_ROWS:
        dv = delta.get(key)
        signal = signal_for_delta(dv, preference=preference, warning_threshold=0.0)
        digits = 0 if key == "partial_residual_slot_days" else 2
        rows.append(
            f"<tr><td>{label}</td><td>{_fmt(baseline.get(key), digits, unit)}</td>"
            f"<td>{_fmt(score_sort.get(key), digits, unit)}</td>"
            f"{html_delta_cell(_fmt(dv, digits, unit, signed=True), signal)}"
            f"<td>{html_signal_badge(signal)}</td></tr>"
        )
    exit_rows = []
    for label, key in (
        ("停損", "stop_exit_share_pct"),
        ("指標", "indicator_exit_share_pct"),
        ("汰弱", "rotation_exit_share_pct"),
        ("期末強制", "forced_exit_share_pct"),
    ):
        signal = signal_for_delta(
            delta.get(key), preference="attention", warning_threshold=0.0
        )
        exit_rows.append(
            f"<tr><td>{label}</td><td>{_fmt(baseline.get(key), 2, '%')}</td>"
            f"<td>{_fmt(score_sort.get(key), 2, '%')}</td>"
            f"{html_delta_cell(_fmt(delta.get(key), 2, '%', signed=True), signal)}"
            f"<td>{html_signal_badge(signal)}</td></tr>"
        )

    yearly_rows = []
    for row in result["yearly"].to_dict("records"):
        r_signal = signal_for_delta(row.get("delta_avg_r"), preference="higher")
        capture_signal = signal_for_delta(row.get("delta_avg_capture_ratio"), preference="higher")
        invested_signal = signal_for_delta(row.get("delta_avg_invested_total"), preference="higher")
        yearly_rows.append(
            f"<tr><td>{int(row['entry_year'])}</td><td>{_fmt(row.get('baseline_avg_r'))}</td>"
            f"<td>{_fmt(row.get('score_sort_avg_r'))}</td>{html_delta_cell(_fmt(row.get('delta_avg_r'), signed=True), r_signal)}"
            f"<td>{_fmt(row.get('baseline_avg_capture_ratio'))}</td><td>{_fmt(row.get('score_sort_avg_capture_ratio'))}</td>"
            f"{html_delta_cell(_fmt(row.get('delta_avg_capture_ratio'), signed=True), capture_signal)}"
            f"<td>{_fmt(row.get('baseline_avg_invested_total'),0)}</td><td>{_fmt(row.get('score_sort_avg_invested_total'),0)}</td>"
            f"{html_delta_cell(_fmt(row.get('delta_avg_invested_total'),0,signed=True), invested_signal)}</tr>"
        )
    bottlenecks = "".join(f"<li>{item}</li>" for item in decision.get("bottlenecks") or [])
    meta = metadata.get("comparison_period") or {}
    body = f"""
<div class="card meta-grid"><div><strong>期間</strong><br>{meta.get('start','')} ～ {meta.get('end','')}</div>
<div><strong>Score source</strong><br><code>{metadata.get('score_source','')}</code></div>
<div><strong>Future Target runtime</strong><br>{'未使用' if not decision.get('future_target_used_for_runtime') else '警告：已使用'}</div></div>
<div class="callout {decision['signal']}">{html_signal_badge(decision['signal'])} <strong>{decision['status']}</strong><br>{decision['conclusion']}</div>
<div class="card"><strong>主要瓶頸</strong><ul class="compact">{bottlenecks}</ul></div>
<h2>資本使用與 Target Capture</h2>
<table><thead><tr><th>指標</th><th>Baseline</th><th>Score Sort</th><th>差異</th><th>判讀</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Exit reason</h2>
<table><thead><tr><th>類別</th><th>Baseline</th><th>Score Sort</th><th>差異</th><th>判讀</th></tr></thead><tbody>{''.join(exit_rows)}</tbody></table>
<h2>依進場年度</h2>
<table><thead><tr><th>年度</th><th>Baseline avg R</th><th>Score avg R</th><th>ΔR</th><th>Baseline capture</th><th>Score capture</th><th>Δcapture</th><th>Baseline投入</th><th>Score投入</th><th>Δ投入</th></tr></thead><tbody>{''.join(yearly_rows)}</tbody></table>
<div class="callout warning"><strong>口徑</strong><br>綠色代表依本audit目的改善；紅色代表惡化；黃色表示方向本身無單一好壞但變化值得注意。半倉slot-days使用每日capacity交易日曆；產業欄位不存在時不推測類股。Future Target只在回放後離線join。</div>
"""
    return html_page(
        title="Breakout Quality Score 排序資本效率／Target Capture 歸因",
        subtitle="Selection PIT Score Sort read-only attribution",
        body=body,
    )


def _json_native(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):
        return _json_native(value.item())
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    return str(value)


def write_score_ranking_capture_audit_outputs(*, result: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = _json_native({
        "schema_version": result["schema_version"],
        "metadata": result["metadata"],
        "baseline": result["baseline"],
        "score_sort": result["score_sort"],
        "score_sort_minus_baseline": result["score_sort_minus_baseline"],
        "decision": result["decision"],
        "yearly": result["yearly"].to_dict("records"),
    })
    (out_dir / "score_ranking_capture_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "score_ranking_capture_audit.md").write_text(
        render_capture_audit_markdown(result), encoding="utf-8"
    )
    (out_dir / "score_ranking_capture_audit.html").write_text(
        render_capture_audit_html(result), encoding="utf-8"
    )
    result["baseline_lifecycle"].to_csv(
        out_dir / "no_filter_capture_lifecycle.csv", index=False, encoding="utf-8-sig"
    )
    result["score_sort_lifecycle"].to_csv(
        out_dir / "score_ranking_capture_lifecycle.csv", index=False, encoding="utf-8-sig"
    )
    result["yearly"].to_csv(
        out_dir / "score_ranking_capture_yearly.csv", index=False, encoding="utf-8-sig"
    )
    scenario_rows = []
    for scenario_key in ("baseline", "score_sort"):
        scenario_rows.append({"scenario": scenario_key, **result[scenario_key]})
    pd.DataFrame(scenario_rows).to_csv(
        out_dir / "score_ranking_capture_scenarios.csv", index=False, encoding="utf-8-sig"
    )
    return payload


__all__ = [
    "SCHEMA_VERSION", "build_trade_lifecycle_rows", "build_score_ranking_capture_audit",
    "render_capture_audit_markdown", "render_capture_audit_html",
    "write_score_ranking_capture_audit_outputs",
]
