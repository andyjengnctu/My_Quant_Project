"""Round-trip attribution for controlled breakout-quality strategy comparisons."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from filters.breakout_quality.contract import SCORE_COLUMN
from filters.breakout_quality.score_store import load_shared_group_score_table
from core.serialization_utils import clean_optional_text as _clean_text, json_native_value as _json_native

ATTRIBUTION_SCHEMA_VERSION = 1
CANONICAL_TRADE_MATCH_FIELDS = ("ticker", "entry_date", "entry_type")
_BUY_PREFIX = "買進 ("
_FULL_EXIT_PREFIXES = ("全倉結算", "汰弱賣出", "期末強制結算")
_ROUND_TRIP_COLUMNS = [
    "scenario",
    "ticker",
    "entry_date",
    "exit_date",
    "entry_type",
    "candidate_type",
    "signal_date",
    "candidate_date",
    "entry_price",
    "exit_price",
    "pnl",
    "r_multiple",
    "holding_calendar_days",
    "exit_type",
    "quality_score",
    "quality_score_date",
    "quality_score_date_source",
    "quality_score_pass",
    "match_occurrence",
    "match_key",
]


def _date_text(value: Any) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    return pd.Timestamp(text).strftime("%Y-%m-%d")


def _finite_float(value: Any, *, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    if isinstance(value, str):
        value = value.replace(",", "").strip()
        if value == "":
            return float(default)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _is_buy_row(type_text: str) -> bool:
    return type_text.startswith(_BUY_PREFIX)


def _is_full_exit_row(type_text: str) -> bool:
    return any(type_text.startswith(prefix) for prefix in _FULL_EXIT_PREFIXES)


def assign_canonical_trade_match_key(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign the project-wide closed-trade identity used by strategy attribution.

    Signal/candidate dates remain diagnostic metadata; actual trade identity is
    ticker + actual entry date + entry type + occurrence.
    """

    out = pd.DataFrame(frame).copy()
    if out.empty:
        out["match_occurrence"] = pd.Series(dtype=int)
        out["match_key"] = pd.Series(dtype=str)
        return out
    missing = sorted(set(CANONICAL_TRADE_MATCH_FIELDS) - set(out.columns))
    if missing:
        raise ValueError(f"closed trade缺少canonical match欄位: {missing}")
    out["match_occurrence"] = out.groupby(
        list(CANONICAL_TRADE_MATCH_FIELDS), sort=False
    ).cumcount() + 1
    out["match_key"] = (
        out["ticker"].astype(str)
        + "|" + out["entry_date"].astype(str)
        + "|" + out["entry_type"].astype(str)
        + "|" + out["match_occurrence"].astype(str)
    )
    return out


def reconstruct_round_trips(trade_history: pd.DataFrame, *, scenario: str) -> pd.DataFrame:
    """Reconstruct one canonical row per closed position from portfolio transaction history."""

    frame = pd.DataFrame(trade_history).copy()
    if frame.empty:
        return pd.DataFrame(columns=_ROUND_TRIP_COLUMNS)
    missing = sorted({"Date", "Ticker", "Type"} - set(frame.columns))
    if missing:
        raise ValueError(f"trade history 缺少必要欄位: {missing}")

    open_positions: dict[str, dict[str, Any]] = {}
    closed_rows: list[dict[str, Any]] = []
    for row_index, row in frame.reset_index(drop=True).iterrows():
        ticker = _clean_text(row.get("Ticker"))
        type_text = _clean_text(row.get("Type"))
        if not ticker or not type_text:
            continue

        if _is_buy_row(type_text):
            if ticker in open_positions:
                raise ValueError(
                    f"trade history 同一 ticker 在尚未結算前再次買進: scenario={scenario}, "
                    f"ticker={ticker}, row={row_index}"
                )
            entry_date = _date_text(row.get("Date"))
            if not entry_date:
                raise ValueError(f"買進列缺少日期: scenario={scenario}, ticker={ticker}, row={row_index}")
            open_positions[ticker] = {
                "scenario": str(scenario),
                "ticker": ticker,
                "entry_date": entry_date,
                "entry_type": _clean_text(row.get("進場類型")) or "normal",
                "candidate_type": _clean_text(row.get("候選類型")),
                "signal_date": _date_text(row.get("買訊日")),
                "candidate_date": _date_text(row.get("候選日")),
                "entry_price": _finite_float(row.get("成交價"), default=0.0),
            }
            continue

        if not _is_full_exit_row(type_text):
            continue
        if ticker not in open_positions:
            raise ValueError(
                f"trade history 結算列找不到對應買進: scenario={scenario}, "
                f"ticker={ticker}, type={type_text}, row={row_index}"
            )

        entry = open_positions.pop(ticker)
        exit_date = _date_text(row.get("Date"))
        entry_ts = pd.Timestamp(entry["entry_date"])
        exit_ts = pd.Timestamp(exit_date)
        closed_rows.append({
            **entry,
            "exit_date": exit_date,
            "exit_price": _finite_float(row.get("成交價"), default=0.0),
            "pnl": _finite_float(
                row.get("該筆總損益") if "該筆總損益" in frame.columns else row.get("單筆損益"),
                default=0.0,
            ),
            "r_multiple": _finite_float(row.get("R_Multiple"), default=0.0),
            "holding_calendar_days": int((exit_ts - entry_ts).days),
            "exit_type": type_text,
        })

    if open_positions:
        sample = sorted(open_positions)[:5]
        raise ValueError(
            f"trade history 結束後仍有未結算持倉: scenario={scenario}, "
            f"count={len(open_positions)}, sample={sample}"
        )

    out = pd.DataFrame(closed_rows)
    if out.empty:
        return pd.DataFrame(columns=_ROUND_TRIP_COLUMNS)
    out = assign_canonical_trade_match_key(out)
    for column in ("quality_score", "quality_score_date", "quality_score_date_source", "quality_score_pass"):
        out[column] = np.nan if column in {"quality_score", "quality_score_pass"} else ""
    return out.reindex(columns=_ROUND_TRIP_COLUMNS)


def _normalize_closed_trade_rows(rows: list[dict[str, Any]], *, scenario: str) -> pd.DataFrame:
    """Normalize exact profile rows when the portfolio engine exposes round trips directly."""

    normalized = []
    for row in list(rows or []):
        entry_date = _date_text((row or {}).get("entry_trade_date"))
        exit_date = _date_text((row or {}).get("exit_date"))
        if not entry_date or not exit_date:
            raise ValueError(f"closed_trade_rows 缺少 entry/exit date: scenario={scenario}")
        normalized.append({
            "scenario": str(scenario),
            "ticker": _clean_text((row or {}).get("ticker")),
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_type": _clean_text((row or {}).get("entry_type")) or "normal",
            "candidate_type": _clean_text((row or {}).get("candidate_type")),
            "signal_date": _date_text((row or {}).get("signal_date")),
            "candidate_date": _date_text((row or {}).get("candidate_date")),
            "entry_price": _finite_float((row or {}).get("entry_price"), default=0.0),
            "exit_price": _finite_float((row or {}).get("exit_price"), default=0.0),
            "pnl": _finite_float((row or {}).get("pnl"), default=0.0),
            "r_multiple": _finite_float((row or {}).get("r_mult"), default=0.0),
            "holding_calendar_days": int((pd.Timestamp(exit_date) - pd.Timestamp(entry_date)).days),
            "exit_type": _clean_text((row or {}).get("exit_type")),
        })
    frame = pd.DataFrame(normalized)
    if frame.empty:
        return pd.DataFrame(columns=_ROUND_TRIP_COLUMNS)
    frame = assign_canonical_trade_match_key(frame)
    for column in ("quality_score", "quality_score_date", "quality_score_date_source", "quality_score_pass"):
        frame[column] = np.nan if column in {"quality_score", "quality_score_pass"} else ""
    return frame.reindex(columns=_ROUND_TRIP_COLUMNS)


def _enrich_scores(
    trades: pd.DataFrame,
    *,
    shared_score_table: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    out = trades.copy()
    if out.empty:
        return out
    scores = shared_score_table[SCORE_COLUMN]
    score_values: list[float] = []
    score_dates: list[str] = []
    score_sources: list[str] = []
    score_passes: list[Any] = []
    for row in out.to_dict("records"):
        matched_score = float("nan")
        matched_date = ""
        matched_source = ""
        for source, field in (
            ("signal_date", "signal_date"),
            ("candidate_date", "candidate_date"),
            ("entry_date", "entry_date"),
        ):
            date_text = _clean_text(row.get(field))
            if not date_text:
                continue
            key = (str(row["ticker"]), date_text)
            try:
                value = float(scores.loc[key])
            except KeyError:
                continue
            if math.isfinite(value):
                matched_score = value
                matched_date = date_text
                matched_source = source
                break
        score_values.append(matched_score)
        score_dates.append(matched_date)
        score_sources.append(matched_source)
        score_passes.append(bool(matched_score >= float(threshold)) if math.isfinite(matched_score) else None)
    out["quality_score"] = score_values
    out["quality_score_date"] = score_dates
    out["quality_score_date_source"] = score_sources
    out["quality_score_pass"] = score_passes
    return out


def _merge_trade_partitions(no_filter: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    left = no_filter.set_index("match_key", drop=False).add_prefix("no_filter_")
    right = quality.set_index("match_key", drop=False).add_prefix("quality_filter_")
    merged = left.join(right, how="outer")
    has_left = merged["no_filter_ticker"].notna()
    has_right = merged["quality_filter_ticker"].notna()
    merged.insert(0, "category", np.select(
        [has_left & has_right, has_left, has_right],
        ["common", "no_filter_only", "quality_filter_only"],
        default="invalid",
    ))
    merged.insert(1, "match_key", merged.index.astype(str))
    merged["ticker"] = merged["no_filter_ticker"].fillna(merged["quality_filter_ticker"])
    merged["entry_date"] = merged["no_filter_entry_date"].fillna(merged["quality_filter_entry_date"])
    merged["entry_type"] = merged["no_filter_entry_type"].fillna(merged["quality_filter_entry_type"])
    no_filter_score = pd.to_numeric(merged["no_filter_quality_score"], errors="coerce")
    no_filter_pass = merged["no_filter_quality_score_pass"]
    merged["selection_mechanism"] = ""
    merged.loc[merged["category"] == "common", "selection_mechanism"] = "retained_common"
    direct_reject = (merged["category"] == "no_filter_only") & no_filter_score.notna() & (no_filter_pass == False)  # noqa: E712
    path_displaced = (merged["category"] == "no_filter_only") & no_filter_score.notna() & (no_filter_pass == True)  # noqa: E712
    score_missing = (merged["category"] == "no_filter_only") & no_filter_score.isna()
    merged.loc[direct_reject, "selection_mechanism"] = "direct_filter_reject"
    merged.loc[path_displaced, "selection_mechanism"] = "portfolio_path_displacement"
    merged.loc[score_missing, "selection_mechanism"] = "score_lookup_unavailable"
    merged.loc[merged["category"] == "quality_filter_only", "selection_mechanism"] = "replacement_from_portfolio_path"
    return merged.reset_index(drop=True)


def _trade_summary(frame: pd.DataFrame, *, r_column: str, pnl_column: str, holding_column: str, score_column: str) -> dict[str, Any]:
    if frame.empty:
        return {
            "trade_count": 0,
            "total_r": 0.0,
            "avg_r": 0.0,
            "median_r": 0.0,
            "win_rate_pct": 0.0,
            "payoff_r": 0.0,
            "total_pnl": 0.0,
            "avg_holding_calendar_days": 0.0,
            "max_winner_r": 0.0,
            "max_loser_r": 0.0,
            "avg_quality_score": None,
        }
    r_values = pd.to_numeric(frame[r_column], errors="coerce").dropna().astype(float)
    pnl_values = pd.to_numeric(frame[pnl_column], errors="coerce").fillna(0.0).astype(float)
    holding_values = pd.to_numeric(frame[holding_column], errors="coerce").fillna(0.0).astype(float)
    score_values = pd.to_numeric(frame[score_column], errors="coerce").dropna().astype(float)
    wins = r_values[r_values > 0]
    losses = r_values[r_values <= 0]
    avg_win = float(wins.mean()) if not wins.empty else 0.0
    avg_loss = abs(float(losses.mean())) if not losses.empty else 0.0
    return {
        "trade_count": int(len(r_values)),
        "total_r": float(r_values.sum()),
        "avg_r": float(r_values.mean()) if not r_values.empty else 0.0,
        "median_r": float(r_values.median()) if not r_values.empty else 0.0,
        "win_rate_pct": float((r_values > 0).mean() * 100.0) if not r_values.empty else 0.0,
        "payoff_r": float(avg_win / avg_loss) if avg_loss > 0 else (99.9 if avg_win > 0 else 0.0),
        "total_pnl": float(pnl_values.sum()),
        "avg_holding_calendar_days": float(holding_values.mean()) if not holding_values.empty else 0.0,
        "max_winner_r": float(wins.max()) if not wins.empty else 0.0,
        "max_loser_r": float(losses.min()) if not losses.empty else 0.0,
        "avg_quality_score": float(score_values.mean()) if not score_values.empty else None,
    }


def _build_yearly_attribution(merged: pd.DataFrame) -> pd.DataFrame:
    if merged.empty:
        return pd.DataFrame(columns=[
            "entry_year", "common_count", "common_no_filter_r", "common_quality_filter_r",
            "common_delta_r", "no_filter_only_count", "no_filter_only_r",
            "direct_filter_reject_count", "portfolio_path_displacement_count",
            "quality_filter_only_count", "quality_filter_only_r", "total_attributed_delta_r",
        ])
    years = pd.to_datetime(merged["entry_date"], errors="coerce").dt.year
    merged = merged.assign(entry_year=years)
    output = []
    for year, group in merged.dropna(subset=["entry_year"]).groupby("entry_year", sort=True):
        common = group[group["category"] == "common"]
        no_only = group[group["category"] == "no_filter_only"]
        quality_only = group[group["category"] == "quality_filter_only"]
        common_left = pd.to_numeric(common["no_filter_r_multiple"], errors="coerce").fillna(0.0).sum()
        common_right = pd.to_numeric(common["quality_filter_r_multiple"], errors="coerce").fillna(0.0).sum()
        no_only_r = pd.to_numeric(no_only["no_filter_r_multiple"], errors="coerce").fillna(0.0).sum()
        quality_only_r = pd.to_numeric(quality_only["quality_filter_r_multiple"], errors="coerce").fillna(0.0).sum()
        output.append({
            "entry_year": int(year),
            "common_count": int(len(common)),
            "common_no_filter_r": float(common_left),
            "common_quality_filter_r": float(common_right),
            "common_delta_r": float(common_right - common_left),
            "no_filter_only_count": int(len(no_only)),
            "no_filter_only_r": float(no_only_r),
            "direct_filter_reject_count": int((no_only["selection_mechanism"] == "direct_filter_reject").sum()),
            "portfolio_path_displacement_count": int((no_only["selection_mechanism"] == "portfolio_path_displacement").sum()),
            "quality_filter_only_count": int(len(quality_only)),
            "quality_filter_only_r": float(quality_only_r),
            "total_attributed_delta_r": float((common_right - common_left) + quality_only_r - no_only_r),
        })
    return pd.DataFrame(output)


def build_trade_attribution(
    *,
    no_filter_trade_history: pd.DataFrame,
    quality_filter_trade_history: pd.DataFrame,
    shared_score_table: pd.DataFrame,
    threshold: float,
    no_filter_closed_trade_rows: list[dict[str, Any]] | None = None,
    quality_filter_closed_trade_rows: list[dict[str, Any]] | None = None,
    no_filter_portfolio_total_r: float | None = None,
    quality_filter_portfolio_total_r: float | None = None,
) -> dict[str, Any]:
    no_filter = (
        _normalize_closed_trade_rows(no_filter_closed_trade_rows, scenario="no_filter")
        if no_filter_closed_trade_rows is not None
        else reconstruct_round_trips(no_filter_trade_history, scenario="no_filter")
    )
    quality = (
        _normalize_closed_trade_rows(quality_filter_closed_trade_rows, scenario="quality_filter")
        if quality_filter_closed_trade_rows is not None
        else reconstruct_round_trips(quality_filter_trade_history, scenario="quality_filter")
    )
    no_filter = _enrich_scores(no_filter, shared_score_table=shared_score_table, threshold=threshold)
    quality = _enrich_scores(quality, shared_score_table=shared_score_table, threshold=threshold)
    merged = _merge_trade_partitions(no_filter, quality)

    common = merged[merged["category"] == "common"]
    no_only = merged[merged["category"] == "no_filter_only"]
    quality_only = merged[merged["category"] == "quality_filter_only"]
    summary = {
        "no_filter_all": _trade_summary(
            no_filter, r_column="r_multiple", pnl_column="pnl",
            holding_column="holding_calendar_days", score_column="quality_score",
        ),
        "quality_filter_all": _trade_summary(
            quality, r_column="r_multiple", pnl_column="pnl",
            holding_column="holding_calendar_days", score_column="quality_score",
        ),
        "common_no_filter": _trade_summary(
            common, r_column="no_filter_r_multiple", pnl_column="no_filter_pnl",
            holding_column="no_filter_holding_calendar_days", score_column="no_filter_quality_score",
        ),
        "common_quality_filter": _trade_summary(
            common, r_column="quality_filter_r_multiple", pnl_column="quality_filter_pnl",
            holding_column="quality_filter_holding_calendar_days", score_column="quality_filter_quality_score",
        ),
        "no_filter_only": _trade_summary(
            no_only, r_column="no_filter_r_multiple", pnl_column="no_filter_pnl",
            holding_column="no_filter_holding_calendar_days", score_column="no_filter_quality_score",
        ),
        "quality_filter_only": _trade_summary(
            quality_only, r_column="quality_filter_r_multiple", pnl_column="quality_filter_pnl",
            holding_column="quality_filter_holding_calendar_days", score_column="quality_filter_quality_score",
        ),
    }

    no_only_r = pd.to_numeric(no_only["no_filter_r_multiple"], errors="coerce").fillna(0.0)
    quality_only_r = pd.to_numeric(quality_only["quality_filter_r_multiple"], errors="coerce").fillna(0.0)
    common_left_r = pd.to_numeric(common["no_filter_r_multiple"], errors="coerce").fillna(0.0)
    common_right_r = pd.to_numeric(common["quality_filter_r_multiple"], errors="coerce").fillna(0.0)
    reconstructed_left_total = float(pd.to_numeric(no_filter["r_multiple"], errors="coerce").fillna(0.0).sum())
    reconstructed_right_total = float(pd.to_numeric(quality["r_multiple"], errors="coerce").fillna(0.0).sum())
    reconstructed_delta = reconstructed_right_total - reconstructed_left_total
    portfolio_delta = None
    if no_filter_portfolio_total_r is not None and quality_filter_portfolio_total_r is not None:
        portfolio_delta = float(quality_filter_portfolio_total_r) - float(no_filter_portfolio_total_r)

    r_attribution = {
        "common_trade_delta_r": float(common_right_r.sum() - common_left_r.sum()),
        "exclusive_selection_delta_r": float(quality_only_r.sum() - no_only_r.sum()),
        "reconstructed_total_r_delta": float(reconstructed_delta),
        "portfolio_reported_total_r_delta": portfolio_delta,
        "reconciliation_error_r": None if portfolio_delta is None else float(reconstructed_delta - portfolio_delta),
        "excluded_winner_count": int((no_only_r > 0).sum()),
        "excluded_winner_r": float(no_only_r[no_only_r > 0].sum()),
        "avoided_loser_count": int((no_only_r <= 0).sum()),
        "avoided_loser_r_abs": float(abs(no_only_r[no_only_r <= 0].sum())),
        "replacement_winner_count": int((quality_only_r > 0).sum()),
        "replacement_winner_r": float(quality_only_r[quality_only_r > 0].sum()),
        "replacement_loser_count": int((quality_only_r <= 0).sum()),
        "replacement_loser_r_abs": float(abs(quality_only_r[quality_only_r <= 0].sum())),
        "direct_filter_reject_count": int((no_only["selection_mechanism"] == "direct_filter_reject").sum()),
        "portfolio_path_displacement_count": int((no_only["selection_mechanism"] == "portfolio_path_displacement").sum()),
        "score_lookup_unavailable_count": int((no_only["selection_mechanism"] == "score_lookup_unavailable").sum()),
    }
    yearly = _build_yearly_attribution(merged)
    return {
        "schema_version": ATTRIBUTION_SCHEMA_VERSION,
        "summary": summary,
        "r_attribution": r_attribution,
        "trade_partition": {
            "common_count": int(len(common)),
            "no_filter_only_count": int(len(no_only)),
            "quality_filter_only_count": int(len(quality_only)),
            "no_filter_total_count": int(len(no_filter)),
            "quality_filter_total_count": int(len(quality)),
        },
        "no_filter_round_trips": no_filter,
        "quality_filter_round_trips": quality,
        "matched_trades": merged,
        "yearly": yearly,
    }


def _fmt(value: Any, digits: int = 2, *, signed: bool = False) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "N/A"
    sign = "+" if signed else ""
    return f"{number:{sign}.{digits}f}"


def _top_trade_lines(frame: pd.DataFrame, *, scenario_prefix: str, ascending: bool, limit: int = 8) -> list[str]:
    r_col = f"{scenario_prefix}_r_multiple"
    score_col = f"{scenario_prefix}_quality_score"
    if frame.empty:
        return ["無。"]
    ranked = frame.sort_values(r_col, ascending=ascending).head(limit)
    lines = ["| Ticker | 進場日 | R | Score | 機制 |", "|---|---:|---:|---:|---|"]
    for row in ranked.to_dict("records"):
        lines.append(
            f"| {row.get('ticker', '')} | {row.get('entry_date', '')} "
            f"| {_fmt(row.get(r_col), 2)} | {_fmt(row.get(score_col), 4)} "
            f"| {row.get('selection_mechanism', '')} |"
        )
    return lines


def render_trade_attribution_markdown(result: dict[str, Any], *, metadata: dict[str, Any]) -> str:
    summary = result["summary"]
    r_attr = result["r_attribution"]
    partition = result["trade_partition"]
    merged = result["matched_trades"]
    no_only = merged[merged["category"] == "no_filter_only"]
    quality_only = merged[merged["category"] == "quality_filter_only"]
    lines = [
        "# Breakout Quality 交易層歸因", "",
        f"- 期間：`{metadata.get('comparison_period', {}).get('start', '')}` ～ `"
        f"{metadata.get('comparison_period', {}).get('end', '')}`",
        f"- 配對鍵：`ticker + 實際進場日 + 進場類型 + 同鍵序號`",
        f"- Filter：`{metadata.get('filter_id', '')}` / threshold `{metadata.get('threshold', '')}`",
        "- 年度歸因依實際進場年分組；R 為既有 Portfolio Engine 的扣費後 round-trip R。",
        "", "## 交易分割", "",
        "| 類別 | 交易數 | 說明 |", "|---|---:|---|",
        f"| 共同交易 | {partition['common_count']} | 兩組具有相同 ticker、進場日與進場類型 |",
        f"| No-filter only | {partition['no_filter_only_count']} | Filter 組沒有的基準交易 |",
        f"| Filter only | {partition['quality_filter_only_count']} | 投組路徑改變後替代進入的交易 |",
        "", "## R 歸因", "",
        "| 項目 | R |", "|---|---:|",
        f"| No-filter 全部交易 | {_fmt(summary['no_filter_all']['total_r'])} |",
        f"| Filter 全部交易 | {_fmt(summary['quality_filter_all']['total_r'])} |",
        f"| 全部交易差異 | {_fmt(r_attr['reconstructed_total_r_delta'], signed=True)} |",
        f"| 共同交易 R 差異 | {_fmt(r_attr['common_trade_delta_r'], signed=True)} |",
        f"| 獨有交易選擇效果 | {_fmt(r_attr['exclusive_selection_delta_r'], signed=True)} |",
        f"| 被排除贏家 R | {_fmt(r_attr['excluded_winner_r'])} |",
        f"| 避開輸家 R 絕對值 | {_fmt(r_attr['avoided_loser_r_abs'])} |",
        f"| 替代贏家 R | {_fmt(r_attr['replacement_winner_r'])} |",
        f"| 替代輸家 R 絕對值 | {_fmt(r_attr['replacement_loser_r_abs'])} |",
        "", "## 各類交易品質", "",
        "| 類別 | 交易數 | 總 R | 平均 R | 中位 R | 勝率 | Payoff(R) | 平均持有日 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in (
        ("共同交易／No filter", "common_no_filter"),
        ("共同交易／Filter", "common_quality_filter"),
        ("No-filter only", "no_filter_only"),
        ("Filter only", "quality_filter_only"),
    ):
        item = summary[key]
        lines.append(
            f"| {label} | {item['trade_count']} | {_fmt(item['total_r'])} | {_fmt(item['avg_r'])} "
            f"| {_fmt(item['median_r'])} | {_fmt(item['win_rate_pct'])}% | {_fmt(item['payoff_r'])} "
            f"| {_fmt(item['avg_holding_calendar_days'])} |"
        )
    lines += [
        "", "## No-filter only 的形成原因", "",
        "| 原因 | 數量 |", "|---|---:|",
        f"| Score 低於固定 threshold，直接被拒絕 | {r_attr['direct_filter_reject_count']} |",
        f"| Score 通過，但因資金／持股路徑改變而未出現 | {r_attr['portfolio_path_displacement_count']} |",
        f"| 無法對應 score | {r_attr['score_lookup_unavailable_count']} |",
        "", "## 依進場年度歸因", "",
    ]
    yearly = result["yearly"]
    if yearly.empty:
        lines.append("無年度歸因資料。")
    else:
        lines += [
            "| 年度 | 共同交易差異 R | No-filter only R | Filter only R | 全部歸因差異 R |",
            "|---:|---:|---:|---:|---:|",
        ]
        for row in yearly.to_dict("records"):
            lines.append(
                f"| {int(row['entry_year'])} | {_fmt(row['common_delta_r'], signed=True)} "
                f"| {_fmt(row['no_filter_only_r'])} | {_fmt(row['quality_filter_only_r'])} "
                f"| {_fmt(row['total_attributed_delta_r'], signed=True)} |"
            )
    lines += ["", "## 被排除的最大贏家", ""]
    lines += _top_trade_lines(no_only[no_only["no_filter_r_multiple"] > 0], scenario_prefix="no_filter", ascending=False)
    lines += ["", "## 避開的最大輸家", ""]
    lines += _top_trade_lines(no_only[no_only["no_filter_r_multiple"] <= 0], scenario_prefix="no_filter", ascending=True)
    lines += ["", "## 替代進入的最大贏家", ""]
    lines += _top_trade_lines(quality_only[quality_only["quality_filter_r_multiple"] > 0], scenario_prefix="quality_filter", ascending=False)
    lines += ["", "## 替代進入的最大輸家", ""]
    lines += _top_trade_lines(quality_only[quality_only["quality_filter_r_multiple"] <= 0], scenario_prefix="quality_filter", ascending=True)

    if r_attr["excluded_winner_r"] > r_attr["avoided_loser_r_abs"]:
        net_selection_text = "被排除贏家的總 R 大於避開輸家的 R，直接篩選本身為負貢獻。"
    else:
        net_selection_text = "避開輸家的 R 足以覆蓋被排除贏家；仍須合併替代交易與共同交易路徑差異判讀。"
    lines += [
        "", "## 客觀判讀", "",
        f"- {net_selection_text}",
        f"- 獨有交易選擇效果為 `{_fmt(r_attr['exclusive_selection_delta_r'], signed=True)} R`。",
        f"- 共同交易的路徑／部位差異為 `{_fmt(r_attr['common_trade_delta_r'], signed=True)} R`。",
        "- 本報表只解釋既有固定 OOS 對照，不得依結果調整 threshold、模型或訓練條件。",
        "",
    ]
    return "\n".join(lines)



def build_upside_realization_attribution(
    *,
    trade_history: pd.DataFrame,
    selected_path_diagnostics: pd.DataFrame,
    upside_r_thresholds: tuple[float, ...],
    adverse_bucket_edges_r: tuple[float, ...],
    scenario: str,
) -> dict[str, Any]:
    """Measure how often selected trades fail to realize later full-horizon upside.

    The full-horizon MFE/adverse path comes from the canonical continuous-target
    truth joined after replay.  Actual stop-out identity and realized R come from
    canonical completed trades.  A same-day stop versus later target peak is
    treated conservatively as stop-before-peak because intraday ordering is not
    observable under the project execution contract.
    """

    thresholds = tuple(sorted({float(value) for value in upside_r_thresholds}))
    adverse_edges = tuple(sorted({float(value) for value in adverse_bucket_edges_r}))
    if not thresholds or any(value <= 0.0 or not math.isfinite(value) for value in thresholds):
        raise ValueError("upside_r_thresholds必須是有限正數")
    if any(value <= 0.0 or not math.isfinite(value) for value in adverse_edges):
        raise ValueError("adverse_bucket_edges_r必須是有限正數")

    trades = reconstruct_round_trips(pd.DataFrame(trade_history), scenario=scenario)
    selected = pd.DataFrame(selected_path_diagnostics).copy()
    required = {
        "ticker", "trade_date", "signal_date", "path_target_available",
        "full_horizon_mfe_r", "full_horizon_adverse_to_peak_r",
        "full_horizon_opportunity_date", "full_horizon_first_risk_breach_date",
        "full_horizon_opportunity_bar", "full_horizon_first_risk_breach_bar",
    }
    missing = sorted(required - set(selected.columns))
    if missing:
        raise ValueError(f"upside realization selected path diagnostics缺欄位: {missing}")

    for column in ("trade_date", "signal_date", "full_horizon_opportunity_date", "full_horizon_first_risk_breach_date"):
        if column in selected.columns:
            selected[column] = pd.to_datetime(selected[column], errors="coerce")
    selected["ticker"] = selected["ticker"].fillna("").astype(str).str.strip()
    join_keys = ["ticker", "trade_date", "signal_date"]
    selected["trade_date"] = selected["trade_date"].dt.strftime("%Y-%m-%d").fillna("")
    selected["signal_date"] = selected["signal_date"].dt.strftime("%Y-%m-%d").fillna("")
    if bool(selected.duplicated(join_keys).any()):
        bad = selected.loc[selected.duplicated(join_keys, keep=False), join_keys].iloc[0].to_dict()
        raise ValueError(f"upside realization selection key不唯一: {bad}")

    path_columns = [
        *join_keys,
        "score_event_date" if "score_event_date" in selected.columns else None,
        "path_target_available",
        "full_horizon_mfe_r",
        "full_horizon_adverse_to_peak_r",
        "full_horizon_opportunity_date",
        "full_horizon_first_risk_breach_date",
        "full_horizon_opportunity_bar",
        "full_horizon_first_risk_breach_bar",
    ]
    path_columns = [column for column in path_columns if column is not None]
    selected = selected[path_columns]

    work = trades.copy()
    if work.empty:
        work["trade_date"] = pd.Series(dtype=str)
        work["path_target_available"] = pd.Series(dtype=bool)
        detail = work
    else:
        work["trade_date"] = work["entry_date"].astype(str)
        work["signal_date"] = work["signal_date"].astype(str)
        detail = work.merge(selected, on=join_keys, how="left", validate="many_to_one")

    if detail.empty:
        path_mask = pd.Series(dtype=bool)
    else:
        available = detail.get("path_target_available", pd.Series(False, index=detail.index))
        if available.dtype == bool:
            available = available.fillna(False)
        else:
            available = available.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})
        mfe = pd.to_numeric(detail.get("full_horizon_mfe_r"), errors="coerce")
        adverse = pd.to_numeric(detail.get("full_horizon_adverse_to_peak_r"), errors="coerce")
        opportunity_date = pd.to_datetime(detail.get("full_horizon_opportunity_date"), errors="coerce")
        exit_date = pd.to_datetime(detail.get("exit_date"), errors="coerce")
        path_mask = available & mfe.map(math.isfinite) & adverse.map(math.isfinite) & opportunity_date.notna()
        detail["path_target_available"] = path_mask
        detail["actual_stop_out"] = detail.get("exit_type", "").fillna("").astype(str).str.contains("停損", regex=False)
        detail["later_peak_after_stop"] = (
            detail["actual_stop_out"]
            & path_mask
            & (opportunity_date >= exit_date)
        )
        first_breach_bar = pd.to_numeric(detail.get("full_horizon_first_risk_breach_bar"), errors="coerce")
        opportunity_bar = pd.to_numeric(detail.get("full_horizon_opportunity_bar"), errors="coerce")
        detail["canonical_risk_breach_before_peak"] = (
            path_mask
            & first_breach_bar.notna()
            & opportunity_bar.notna()
            & (first_breach_bar >= 1)
            & (first_breach_bar <= opportunity_bar)
        )
        detail["realized_r"] = pd.to_numeric(detail.get("r_multiple"), errors="coerce")

    covered = detail.loc[path_mask].copy() if len(detail) else detail.copy()
    summary: dict[str, Any] = {
        "completed_trade_count": int(len(detail)),
        "path_target_covered_count": int(len(covered)),
        "path_target_coverage_rate": (1.0 if not len(detail) else float(len(covered) / len(detail))),
        "actual_stop_out_count": int(covered.get("actual_stop_out", pd.Series(dtype=bool)).sum()) if len(covered) else 0,
        "actual_stop_out_rate": (None if not len(covered) else float(covered["actual_stop_out"].mean())),
        "stop_before_later_peak_count": int(covered.get("later_peak_after_stop", pd.Series(dtype=bool)).sum()) if len(covered) else 0,
        "stop_before_later_peak_rate": (None if not len(covered) else float(covered["later_peak_after_stop"].mean())),
        "canonical_risk_breach_before_peak_count": int(covered.get("canonical_risk_breach_before_peak", pd.Series(dtype=bool)).sum()) if len(covered) else 0,
        "full_horizon_mfe_mean_r": (None if not len(covered) else float(pd.to_numeric(covered["full_horizon_mfe_r"], errors="coerce").mean())),
        "full_horizon_adverse_to_peak_mean_r": (None if not len(covered) else float(pd.to_numeric(covered["full_horizon_adverse_to_peak_r"], errors="coerce").mean())),
        "realized_mean_r": (None if not len(covered) else float(pd.to_numeric(covered["realized_r"], errors="coerce").mean())),
        "thresholds": {},
    }
    for threshold in thresholds:
        key = f"{threshold:g}R"
        eligible = covered.loc[pd.to_numeric(covered.get("full_horizon_mfe_r"), errors="coerce") >= threshold].copy()
        stop_before = eligible.get("later_peak_after_stop", pd.Series(False, index=eligible.index)).fillna(False).astype(bool)
        summary["thresholds"][key] = {
            "full_horizon_mfe_at_least_count": int(len(eligible)),
            "stop_before_later_peak_count": int(stop_before.sum()),
            "stop_before_later_peak_rate": (None if not len(eligible) else float(stop_before.mean())),
            "stop_before_later_peak_realized_mean_r": (
                None if not int(stop_before.sum()) else float(pd.to_numeric(eligible.loc[stop_before, "realized_r"], errors="coerce").mean())
            ),
            "stop_before_later_peak_mfe_mean_r": (
                None if not int(stop_before.sum()) else float(pd.to_numeric(eligible.loc[stop_before, "full_horizon_mfe_r"], errors="coerce").mean())
            ),
        }

    mfe_edges = [-math.inf, *thresholds, math.inf]
    mfe_labels = []
    for index in range(len(mfe_edges) - 1):
        left, right = mfe_edges[index], mfe_edges[index + 1]
        if math.isinf(left):
            mfe_labels.append(f"<{right:g}R")
        elif math.isinf(right):
            mfe_labels.append(f">={left:g}R")
        else:
            mfe_labels.append(f"{left:g}-{right:g}R")
    adverse_bin_edges = [-math.inf, *adverse_edges, math.inf]
    adverse_labels = []
    for index in range(len(adverse_bin_edges) - 1):
        left, right = adverse_bin_edges[index], adverse_bin_edges[index + 1]
        if math.isinf(left):
            adverse_labels.append(f"<{right:g}R")
        elif math.isinf(right):
            adverse_labels.append(f">={left:g}R")
        else:
            adverse_labels.append(f"{left:g}-{right:g}R")

    joint_rows: list[dict[str, Any]] = []
    if len(covered):
        bucketed = covered.copy()
        bucketed["mfe_bucket"] = pd.cut(
            pd.to_numeric(bucketed["full_horizon_mfe_r"], errors="coerce"),
            bins=mfe_edges, labels=mfe_labels, right=False,
        )
        bucketed["adverse_bucket"] = pd.cut(
            pd.to_numeric(bucketed["full_horizon_adverse_to_peak_r"], errors="coerce"),
            bins=adverse_bin_edges, labels=adverse_labels, right=False,
        )
        for (mfe_bucket, adverse_bucket), group in bucketed.groupby(
            ["mfe_bucket", "adverse_bucket"], observed=True, sort=False
        ):
            joint_rows.append({
                "mfe_bucket": str(mfe_bucket),
                "adverse_bucket": str(adverse_bucket),
                "trade_count": int(len(group)),
                "stop_out_rate": float(group["actual_stop_out"].mean()),
                "stop_before_later_peak_rate": float(group["later_peak_after_stop"].mean()),
                "realized_mean_r": float(pd.to_numeric(group["realized_r"], errors="coerce").mean()),
                "full_horizon_mfe_mean_r": float(pd.to_numeric(group["full_horizon_mfe_r"], errors="coerce").mean()),
                "full_horizon_adverse_to_peak_mean_r": float(pd.to_numeric(group["full_horizon_adverse_to_peak_r"], errors="coerce").mean()),
            })

    return {
        "summary": _json_native(summary),
        "joint_rows": _json_native(joint_rows),
        "detail": detail,
    }


def write_trade_attribution_outputs(
    *,
    project_root: str | Path,
    output_dir: str | Path,
    filter_id: str,
    threshold: float,
    metadata: dict[str, Any],
    no_filter_trade_history: pd.DataFrame,
    quality_filter_trade_history: pd.DataFrame,
    no_filter_closed_trade_rows: list[dict[str, Any]] | None = None,
    quality_filter_closed_trade_rows: list[dict[str, Any]] | None = None,
    no_filter_portfolio_total_r: float | None = None,
    quality_filter_portfolio_total_r: float | None = None,
) -> dict[str, Any]:
    shared_score_table = load_shared_group_score_table(str(project_root), str(filter_id))
    result = build_trade_attribution(
        no_filter_trade_history=no_filter_trade_history,
        quality_filter_trade_history=quality_filter_trade_history,
        shared_score_table=shared_score_table,
        threshold=float(threshold),
        no_filter_closed_trade_rows=no_filter_closed_trade_rows,
        quality_filter_closed_trade_rows=quality_filter_closed_trade_rows,
        no_filter_portfolio_total_r=no_filter_portfolio_total_r,
        quality_filter_portfolio_total_r=quality_filter_portfolio_total_r,
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_payload = _json_native({
        "schema_version": ATTRIBUTION_SCHEMA_VERSION,
        "metadata": dict(metadata),
        "trade_partition": result["trade_partition"],
        "summary": result["summary"],
        "r_attribution": result["r_attribution"],
        "yearly": result["yearly"].to_dict("records"),
    })
    (out_dir / "trade_attribution.json").write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "trade_attribution.md").write_text(
        render_trade_attribution_markdown(result, metadata=metadata),
        encoding="utf-8",
    )
    result["matched_trades"].to_csv(out_dir / "trade_attribution_trades.csv", index=False, encoding="utf-8-sig")
    result["yearly"].to_csv(out_dir / "trade_attribution_yearly.csv", index=False, encoding="utf-8-sig")
    result["no_filter_round_trips"].to_csv(out_dir / "no_filter_round_trips.csv", index=False, encoding="utf-8-sig")
    result["quality_filter_round_trips"].to_csv(out_dir / "quality_filter_round_trips.csv", index=False, encoding="utf-8-sig")
    return json_payload


__all__ = [
    "ATTRIBUTION_SCHEMA_VERSION",
    "build_trade_attribution",
    "build_upside_realization_attribution",
    "reconstruct_round_trips",
    "render_trade_attribution_markdown",
    "write_trade_attribution_outputs",
]
