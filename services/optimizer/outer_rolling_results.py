from __future__ import annotations

import pandas as pd

from core.portfolio_stats import calc_plain_romd
from services.optimizer.outer_rolling_curve_metrics import (
    _calc_stitched_curve_metrics,
    _stitch_benchmark_equity_curve,
    _stitch_strategy_equity_curves,
)
from services.optimizer.outer_rolling_fold_context import (
    normalize_optimizer_seed_ensemble_fold_row,
    normalize_optimizer_seed_ensemble_fold_rows,
)
from services.optimizer.outer_rolling_formatting import (
    OOS_SCORE_DECIMALS,
    _display_compact_month_period,
    _fmt_duration,
    _format_compare,
    _format_compare_plain,
    _format_plain_score,
    _pad_ansi,
    _period_label,
    _visible_len,
)
from services.optimizer.outer_rolling_performance import _mark_resource_probe_fallback
from services.optimizer.outer_rolling_policy import (
    CHAIN_POLICY_NAMES,
    FINALISTS_AGREE_RESULT_POLICY_NAMES,
    FINALISTS_AGREE_TABLE_TITLE,
    FINALIST_BEST_RESULT_POLICY_NAMES,
    FINALIST_BEST_TABLE_TITLE,
    REPORT_POLICY_LABELS,
    REPORT_POLICY_NAMES,
    SEED_ENSEMBLE_RESULTS_TABLE_TITLE,
    SEED_ENSEMBLE_RESULT_POLICY_NAMES,
    _is_rolling_random_seed_ensemble_enabled,
    _policy_is_available,
    _policy_plain_romd_score,
)

def _build_chained_oos_summary(rows: list[dict], *, chained_override: dict | None = None) -> dict:
    if chained_override is not None:
        return dict(chained_override)
    if not rows:
        return {}
    first_oos_start = min(pd.Timestamp(row.get("oos_start_date") or f"{str(row.get('oos_year'))[:4]}-01-01").normalize() for row in rows)
    last_oos_end = max(pd.Timestamp(row.get("oos_end_date") or f"{str(row.get('oos_year'))[:4]}-12-31").normalize() for row in rows)
    first_year = int(first_oos_start.year)
    last_year = int(last_oos_end.year)
    selection_start_ts = min(pd.Timestamp(row.get("selection_start_date") or f"{int(row.get('selection_start_year', first_year))}-01-01").normalize() for row in rows)
    selection_end_ts = max(pd.Timestamp(row.get("selection_end_date") or f"{int(row.get('selection_end_year', last_year))}-12-31").normalize() for row in rows)
    selection_period_label = _period_label(selection_start_ts, selection_end_ts)
    oos_period_label = _period_label(first_oos_start, last_oos_end)

    best_stitched = _stitch_strategy_equity_curves(rows, best_finalist=True)
    benchmark_stitched = _stitch_benchmark_equity_curve(rows)
    best_metrics = _calc_stitched_curve_metrics(best_stitched)
    benchmark_metrics = _calc_stitched_curve_metrics(benchmark_stitched, benchmark_plain_romd=True)
    best_score = float(best_metrics.get("score", 0.0))
    benchmark_score = float(benchmark_metrics.get("score", 0.0))
    best_return = float(best_metrics.get("return_pct", 0.0))
    benchmark_return = float(benchmark_metrics.get("return_pct", 0.0))

    summary = {
        "method": "fallback_period_closeout_stitched_daily_equity",
        "score_aggregation_method": "fallback_period_closeout_stitched_daily_equity",
        "return_aggregation_method": "fallback_period_closeout_stitched_daily_equity_total_return",
        "note": "fallback：由各 OOS period daily equity 串接後重新計算 score；正式輸出應優先使用 continuous_active_param_replay。",
        "selection_period": selection_period_label,
        "oos_period": oos_period_label,
        "best_finalist_oos_score": float(best_score),
        "benchmark_oos_score": float(benchmark_score),
        "best_finalist_return_pct": float(best_return),
        "benchmark_return_pct": float(benchmark_return),
        "benchmark_alpha_pct": float(best_return - benchmark_return),
        "best_finalist_mdd_pct": float(best_metrics.get("mdd_pct", 0.0)),
        "benchmark_mdd_pct": float(benchmark_metrics.get("mdd_pct", 0.0)),
        "best_finalist_annual_return_pct": float(best_metrics.get("annual_return_pct", 0.0)),
        "benchmark_annual_return_pct": float(benchmark_metrics.get("annual_return_pct", 0.0)),
        "best_finalist_curve_points": int(best_metrics.get("curve_points", 0)),
        "benchmark_curve_points": int(benchmark_metrics.get("curve_points", 0)),
        "yearly_best_finalist_oos_score": [float(row.get("best_finalist_oos_score", 0.0)) for row in rows],
        "yearly_benchmark_oos_score": [float(row.get("benchmark_oos_score", 0.0)) for row in rows],
    }
    for policy_name in CHAIN_POLICY_NAMES:
        stitched = _stitch_strategy_equity_curves(rows, policy_name=policy_name)
        metrics = _calc_stitched_curve_metrics(stitched)
        chain_score = float(metrics.get("score", 0.0))
        chain_plain_romd = float(metrics.get("plain_romd_score", calc_plain_romd(metrics.get("return_pct", 0.0), metrics.get("mdd_pct", 0.0))))
        chain_return = float(metrics.get("return_pct", 0.0))
        summary[policy_name] = {
            "available": int(metrics.get("curve_points", 0) or 0) > 0,
            "rank_1_oos": float(chain_score),
            "rank_1_plain_romd": float(chain_plain_romd),
            "best_gap": float(chain_score - best_score),
            "benchmark_0050_gap": float(chain_score - benchmark_score),
            "benchmark_0050_plain_romd_gap": float(chain_plain_romd - benchmark_score),
            "rank_1_return_pct": float(chain_return),
            "best_gap_pct": float(chain_return - best_return),
            "benchmark_0050_gap_pct": float(chain_return - benchmark_return),
            "rank_1_mdd_pct": float(metrics.get("mdd_pct", 0.0)),
            "rank_1_annual_return_pct": float(metrics.get("annual_return_pct", 0.0)),
            "rank_1_curve_points": int(metrics.get("curve_points", 0)),
            "yearly_oos_score": [float((row.get(policy_name) or {}).get("rank_1_oos", 0.0)) for row in rows],
            "yearly_return_pct": [float((row.get(policy_name) or {}).get("rank_1_return_pct", 0.0)) for row in rows],
        }
    return summary


def _resolve_chain_elapsed_sec(rows: list[dict], chained: dict) -> float:
    """Return the OOS_CHAIN elapsed value for display/reporting.

    In serial mode the chain row historically used the sum of fold elapsed time. In
    parallel-fold timing mode, however, the sum of fold elapsed time is total work
    time, not user-visible wall-clock time. When the caller provides an
    ``elapsed_sec`` override in the chained summary, prefer it; otherwise fall back
    to the serial-compatible fold sum.
    """
    try:
        override = chained.get("elapsed_sec")
    except AttributeError:
        override = None
    if override is not None:
        try:
            return max(0.0, float(override))
        except (TypeError, ValueError) as exc:
            _mark_resource_probe_fallback(exc)
    total = 0.0
    for item in list(rows or []):
        try:
            total += float(item.get("elapsed_sec", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
    return max(0.0, total)


def _with_chain_elapsed_override(chained_override: dict | None, *, elapsed_sec: float | None) -> dict:
    payload = dict(chained_override or {})
    if elapsed_sec is not None:
        try:
            payload["elapsed_sec"] = max(0.0, float(elapsed_sec))
        except (TypeError, ValueError) as exc:
            _mark_resource_probe_fallback(exc)
    return payload


def _build_chained_oos_row(rows: list[dict], *, chained_override: dict | None = None) -> dict | None:
    if not rows:
        return None
    chained = _build_chained_oos_summary(rows, chained_override=chained_override)
    row = {
        "fold": "OOS_CHAIN",
        "selection_period": chained.get("selection_period", ""),
        "oos_year": chained.get("oos_period", ""),
        "best_finalist_oos_score": float(chained.get("best_finalist_oos_score", 0.0)),
        "benchmark_oos_score": float(chained.get("benchmark_oos_score", 0.0)),
        "best_finalist_return_pct": float(chained.get("best_finalist_return_pct", 0.0)),
        "benchmark_return_pct": float(chained.get("benchmark_return_pct", 0.0)),
        "elapsed_sec": _resolve_chain_elapsed_sec(rows, chained),
        "aggregation_method": chained.get("score_aggregation_method") or chained.get("method"),
    }
    for policy_name in REPORT_POLICY_NAMES:
        item = dict(chained.get(policy_name) or {})
        available = _policy_is_available(item)
        rank_score = float(item.get("rank_1_oos", 0.0)) if available else 0.0
        plain_romd = _policy_plain_romd_score(item) if available else 0.0
        row[policy_name] = {
            "available": bool(available),
            "rank_1_trial": None,
            "rank_1_oos": rank_score,
            "rank_1_plain_romd": plain_romd,
            "rank_1_return_pct": float(item.get("rank_1_return_pct", 0.0)) if available else 0.0,
            "rank_1_mdd_pct": float(item.get("rank_1_mdd_pct", 0.0)) if available else 0.0,
            "best_gap": (rank_score - float(chained.get("best_finalist_oos_score", 0.0))) if available else 0.0,
            "benchmark_0050_gap": (rank_score - float(chained.get("benchmark_oos_score", 0.0))) if available else 0.0,
            "benchmark_0050_plain_romd_gap": (plain_romd - float(chained.get("benchmark_oos_score", 0.0))) if available else 0.0,
            "unavailable_reason": item.get("unavailable_reason") or item.get("skip_reason") or "",
        }
    return row



def _rows_period_bounds(rows: list[dict]) -> dict:
    source_rows = [
        normalize_optimizer_seed_ensemble_fold_row(row)
        for row in list(rows or [])
        if str((row or {}).get("fold", "")).upper() not in {"OOS_CHAIN", "OOS_AVG"}
    ]
    if not source_rows:
        return {}

    def _date_or_latest(value, fallback):
        text = str(value or "").strip()
        if text.lower() == "latest" or not text:
            return pd.Timestamp(fallback).normalize()
        return pd.Timestamp(text).normalize()

    selection_starts = [pd.Timestamp(row.get("selection_start_date")).normalize() for row in source_rows if row.get("selection_start_date")]
    selection_ends = [pd.Timestamp(row.get("selection_end_date")).normalize() for row in source_rows if row.get("selection_end_date")]
    oos_starts = [pd.Timestamp(row.get("oos_start_date")).normalize() for row in source_rows if row.get("oos_start_date")]
    oos_ends = [_date_or_latest(row.get("oos_end_date"), pd.Timestamp.today().normalize()) for row in source_rows if row.get("oos_start_date") or row.get("oos_end_date")]
    oos_keys = [int(row.get("oos_year", 0) or 0) for row in source_rows if int(row.get("oos_year", 0) or 0) > 0]

    selection_period = ""
    if selection_starts and selection_ends:
        selection_period = _period_label(min(selection_starts), max(selection_ends))
    oos_period = ""
    if oos_starts:
        oos_period = _period_label(min(oos_starts), max(oos_ends) if oos_ends else min(oos_starts))
    return {
        "selection_period": selection_period,
        "oos_period": oos_period,
        "first_oos_key": min(oos_keys) if oos_keys else 0,
        "last_oos_key": max(oos_keys) if oos_keys else 0,
    }


def _avg_float_from_rows(rows: list[dict], getter, default: float = 0.0) -> float:
    values: list[float] = []
    for row in list(rows or []):
        try:
            value = float(getter(row))
        except (TypeError, ValueError, KeyError, AttributeError):
            continue
        values.append(value)
    return (sum(values) / float(len(values))) if values else float(default)


def _build_oos_avg_row(rows: list[dict]) -> dict | None:
    source_rows = [dict(row) for row in list(rows or []) if str(row.get("fold", "")).upper() not in {"OOS_CHAIN", "OOS_AVG"}]
    if not source_rows:
        return None
    bounds = _rows_period_bounds(source_rows)
    row = {
        "fold": "OOS_AVG",
        "selection_period": str(bounds.get("selection_period", "")),
        "oos_year": str(bounds.get("oos_period", "")),
        "oos_period": str(bounds.get("oos_period", "")),
        "best_finalist_oos_score": _avg_float_from_rows(source_rows, lambda item: item.get("best_finalist_oos_score", 0.0)),
        "benchmark_oos_score": _avg_float_from_rows(source_rows, lambda item: item.get("benchmark_oos_score", 0.0)),
        "best_finalist_return_pct": _avg_float_from_rows(source_rows, lambda item: item.get("best_finalist_return_pct", 0.0)),
        "benchmark_return_pct": _avg_float_from_rows(source_rows, lambda item: item.get("benchmark_return_pct", 0.0)),
        "elapsed_sec": None,
    }
    for policy_name in REPORT_POLICY_NAMES:
        available_rows = [
            item
            for item in source_rows
            if _policy_is_available(dict((item.get(policy_name) or {})))
        ]
        if not available_rows:
            row[policy_name] = {
                "available": False,
                "rank_1_trial": None,
                "rank_1_oos": 0.0,
                "rank_1_plain_romd": 0.0,
                "rank_1_return_pct": 0.0,
                "rank_1_mdd_pct": 0.0,
                "best_gap": 0.0,
                "benchmark_0050_gap": 0.0,
                "benchmark_0050_plain_romd_gap": 0.0,
                "unavailable_reason": "no_available_period",
            }
            continue
        rank_score = _avg_float_from_rows(available_rows, lambda item, name=policy_name: (item.get(name) or {}).get("rank_1_oos", 0.0))
        plain_romd = _avg_float_from_rows(available_rows, lambda item, name=policy_name: _policy_plain_romd_score((item.get(name) or {})))
        row[policy_name] = {
            "available": True,
            "rank_1_trial": None,
            "rank_1_oos": rank_score,
            "rank_1_plain_romd": plain_romd,
            "rank_1_return_pct": _avg_float_from_rows(available_rows, lambda item, name=policy_name: (item.get(name) or {}).get("rank_1_return_pct", 0.0)),
            "rank_1_mdd_pct": _avg_float_from_rows(available_rows, lambda item, name=policy_name: (item.get(name) or {}).get("rank_1_mdd_pct", 0.0)),
            "best_gap": rank_score - float(row["best_finalist_oos_score"]),
            "benchmark_0050_gap": rank_score - float(row["benchmark_oos_score"]),
            "benchmark_0050_plain_romd_gap": plain_romd - float(row["benchmark_oos_score"]),
        }
    return row






def _policy_cell_text(policy_row: dict, *, best_score: float, benchmark_score: float, color: bool = True) -> tuple[str, str]:
    _ = best_score
    if not _policy_is_available(policy_row):
        return "N/A", "N/A"
    rank_1_romd = _policy_plain_romd_score(policy_row)
    if color:
        return (
            _format_plain_score(rank_1_romd),
            _format_compare(benchmark_score, rank_1_romd),
        )
    return (
        f"{rank_1_romd:.{OOS_SCORE_DECIMALS}f}",
        _format_compare_plain(benchmark_score, rank_1_romd),
    )


def _table_separator(width: int = 218) -> str:
    return "-" * int(width)


def _render_results_table(rows: list[dict], *, color: bool = True, include_chain: bool = True, include_oos_avg: bool = False, chained_override: dict | None = None, table_title: str = "ROLLING MONTHLY OOS RESULTS", policy_names: tuple[str, ...] | None = None) -> str:
    display_rows = normalize_optimizer_seed_ensemble_fold_rows(list(rows or []))
    if include_oos_avg and display_rows:
        avg_row = _build_oos_avg_row(display_rows)
        if avg_row is not None:
            display_rows = display_rows + [avg_row]
    if include_chain:
        chain_row = _build_chained_oos_row(display_rows, chained_override=chained_override)
        if chain_row is not None:
            display_rows = display_rows + [chain_row]
    if not display_rows:
        return ""
    active_policy_names = tuple(policy_names or REPORT_POLICY_NAMES)
    if not active_policy_names:
        return ""
    widths = {
        "fold": 9,
        "selection": 19,
        "oos_year": 19,
        "romd": 8,
        "bench": 15,
        "elapsed": 8,
    }
    policy_group_width = widths["romd"] + widths["bench"] + 3
    lines: list[str] = []
    lines.append(str(table_title or "ROLLING MONTHLY OOS RESULTS"))
    policy_header = " | ".join(_pad_ansi(REPORT_POLICY_LABELS[name], policy_group_width, align="^") for name in active_policy_names)
    policy_subheader = " | ".join(
        f"{_pad_ansi('RoMD', widths['romd'], align='^')} | {_pad_ansi('0050', widths['bench'], align='^')}"
        for _ in active_policy_names
    )
    header1 = (
        f"{_pad_ansi('fold', widths['fold'], align='^')} | {_pad_ansi('train', widths['selection'], align='^')} | {_pad_ansi('oos_period', widths['oos_year'], align='^')} | "
        f"{policy_header} | {_pad_ansi('elapsed', widths['elapsed'], align='^')}"
    )
    header2 = (
        f"{_pad_ansi('', widths['fold'])} | {_pad_ansi('', widths['selection'])} | {_pad_ansi('', widths['oos_year'])} | "
        f"{policy_subheader} | {_pad_ansi('', widths['elapsed'])}"
    )
    separator = _table_separator(max(_visible_len(header1), _visible_len(header2), 120))
    lines.append(separator)
    lines.append(header1)
    lines.append(separator)
    lines.append(header2)
    lines.append(separator)
    total = len(rows or [])
    for idx, row in enumerate(display_rows, start=1):
        fold_kind = str(row.get("fold", "")).upper()
        is_chain = fold_kind == "OOS_CHAIN"
        is_avg = fold_kind == "OOS_AVG"
        fold_text = "OOS_CHAIN" if is_chain else ("OOS_AVG" if is_avg else str(row.get("fold") or f"{idx}/{total}"))
        best_score = float(row.get("best_finalist_oos_score", 0.0))
        benchmark_score = float(row.get("benchmark_oos_score", 0.0))
        policy_cells: list[str] = []
        for policy_name in active_policy_names:
            romd_text, bench_text = _policy_cell_text(row.get(policy_name) or {}, best_score=best_score, benchmark_score=benchmark_score, color=color)
            policy_cells.append(
                f"{_pad_ansi(romd_text, widths['romd'], align='>')} | "
                f"{_pad_ansi(bench_text, widths['bench'], align='>')}"
            )
        line = (
            f"{_pad_ansi(fold_text, widths['fold'])} | {_pad_ansi(_display_compact_month_period(row.get('selection_period', '')), widths['selection'])} | {_pad_ansi(_display_compact_month_period(row.get('oos_period') or row.get('oos_year', '')), widths['oos_year'])} | "
            f"{' | '.join(policy_cells)} | "
            f"{_pad_ansi('' if row.get('elapsed_sec') is None else _fmt_duration(row.get('elapsed_sec', 0.0)), widths['elapsed'], align='>')}"
        )
        lines.append(line)
    lines.append(separator)
    return "\n".join(lines)


def _render_optimizer_results_tables(rows: list[dict], *, color: bool = True, include_chain: bool = True, include_oos_avg: bool = False, chained_override: dict | None = None, main_table_title: str = "ROLLING MONTHLY OOS RESULTS", retention_table_title: str = "") -> str:
    _ = (main_table_title, retention_table_title)
    finalist_best_table = _render_results_table(
        rows,
        color=color,
        include_chain=include_chain,
        include_oos_avg=include_oos_avg,
        chained_override=chained_override,
        table_title=FINALIST_BEST_TABLE_TITLE,
        policy_names=FINALIST_BEST_RESULT_POLICY_NAMES,
    )
    finalists_agree_table = _render_results_table(
        rows,
        color=color,
        include_chain=include_chain,
        include_oos_avg=include_oos_avg,
        chained_override=chained_override,
        table_title=FINALISTS_AGREE_TABLE_TITLE,
        policy_names=FINALISTS_AGREE_RESULT_POLICY_NAMES,
    )
    seed_ensemble_table = ""
    if _is_rolling_random_seed_ensemble_enabled():
        seed_ensemble_table = _render_results_table(
            rows,
            color=color,
            include_chain=include_chain,
            include_oos_avg=include_oos_avg,
            chained_override=chained_override,
            table_title=SEED_ENSEMBLE_RESULTS_TABLE_TITLE,
            policy_names=SEED_ENSEMBLE_RESULT_POLICY_NAMES,
        )
    return "\n\n".join(part for part in (finalist_best_table, finalists_agree_table, seed_ensemble_table) if part)


def render_optimizer_results_tables(rows: list[dict], *, color: bool = True, include_chain: bool = True, include_oos_avg: bool = False, chained_override: dict | None = None, main_table_title: str = "", retention_table_title: str = "") -> str:
    """Render optimizer OOS result tables through the rolling-OOS table source.

    Non-rolling summary display intentionally uses this public wrapper with a
    single fold row so rolling and non-rolling console tables cannot drift.
    """
    return _render_optimizer_results_tables(
        rows,
        color=color,
        include_chain=include_chain,
        include_oos_avg=include_oos_avg,
        chained_override=chained_override,
        main_table_title=main_table_title,
        retention_table_title=retention_table_title,
    )




def _print_completed_results(rows: list[dict]):
    if not rows:
        return
    print("\n" + _render_optimizer_results_tables(rows, color=True, include_chain=True))
