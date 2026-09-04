"""Read-only C68/C69 matched marginal-position attribution.

This one-shot Audit answers the decision question left after SR-C69: did relaxing
C58-derived K recover genuinely higher-upside marginal positions whose path/adverse
conversion is poor, or did count-first K-Flex simply admit a weaker score tail?

No training, score generation, target fitting, strategy replay, or sidecar backfill
is permitted here.  All portfolio membership/execution/path evidence is consumed
from the canonical completed Strategy Compare runs.
"""

from __future__ import annotations

from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from config.audit import AUDIT_OUTPUT_ROOT
from core.audit_policy import AuditDefinition
from core.path_utils import project_relative_display_path
from core.report_metrics import (
    MARGINAL_POSITION_ATTRIBUTION_METRICS,
    MFE_SAFETY_QUADRANT_DISTRIBUTION_METRICS,
)
from filters.breakout_quality.dataset_store import resolve_dataset_paths
from filters.breakout_quality.paths import resolve_filter_output_dir
from services.audit.mfe_safety_truth import (
    AuditBlockedError,
    attach_quadrants,
    build_truth_geometry,
    finite_float,
    normalize_date,
    normalize_ticker,
    resolve_truth_provider_contract,
)
from services.audit.strategy_compare_source import (
    AuditSourceBlockedError,
    load_strategy_arm_path_sidecars,
    load_strategy_compare_source,
)

SUPPORTED_AUDIT_TYPE = "c69_marginal_position_attribution"

_DEFAULT_FINAL_STAGE = "feasible_ascent_final"
_ORDINAL_ORDER = (
    "baseline_1_to_k",
    "k_plus_1",
    "k_plus_2",
    "k_plus_3_plus",
    "all_extra",
)
_ORDINAL_DISPLAY = {
    "baseline_1_to_k": "Baseline 1..K",
    "k_plus_1": "K+1",
    "k_plus_2": "K+2",
    "k_plus_3_plus": "K+3+",
    "all_extra": "All K+1+",
}
_MATCHED_COHORT_ORDER = ("common", "c68_only", "c69_only")
_MATCHED_COHORT_DISPLAY = {
    "common": "Common planned",
    "c68_only": "C68-only planned",
    "c69_only": "C69-only planned",
}


def _relative(path: Path, root: Path) -> str:
    return project_relative_display_path(Path(path), project_root=Path(root))


def _as_bool_series(frame: pd.DataFrame, column: str, *, default: bool = False) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=bool)
    raw = frame[column]
    if raw.dtype == bool:
        return raw.fillna(default).astype(bool)
    normalized = raw.fillna("").astype(str).str.strip().str.lower()
    truthy = normalized.isin({"1", "true", "t", "yes", "y"})
    falsey = normalized.isin({"0", "false", "f", "no", "n", ""})
    invalid = ~(truthy | falsey)
    if bool(invalid.any()):
        sample = raw.loc[invalid].iloc[0]
        raise AuditBlockedError(f"{column}含無法解析的bool值: {sample!r}")
    return truthy


def _event_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["ticker"].astype(str)
        + "|" + frame["trade_date"].astype(str)
        + "|" + frame["signal_date"].astype(str)
    )


def _normalize_capacity(frame: pd.DataFrame) -> pd.DataFrame:
    table = pd.DataFrame(frame).copy()
    required = {
        "Date",
        "Pre_Market_Positions",
        "Pre_Market_Free_Slots",
        "Orderable_Candidates",
        "Resource_Aware_Max_DL_Eligible",
        "Resource_Aware_Baseline_K",
        "Resource_Aware_Physical_Free_Slots",
        "Resource_Aware_Baseline_Reserved_Milli",
        "Resource_Aware_Selected",
        "Resource_Aware_K_Flex_Extra_Positions",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise AuditBlockedError(f"C68/C69 daily capacity缺少欄位: {missing}")
    table["trade_date"] = pd.to_datetime(table["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if bool(table["trade_date"].isna().any()):
        raise AuditBlockedError("C68/C69 daily capacity含無效Date")
    table["eligible"] = _as_bool_series(table, "Resource_Aware_Max_DL_Eligible")
    numeric = {
        "pre_market_positions": "Pre_Market_Positions",
        "free_slots": "Pre_Market_Free_Slots",
        "orderable_candidates": "Orderable_Candidates",
        "baseline_k": "Resource_Aware_Baseline_K",
        "physical_free_slots": "Resource_Aware_Physical_Free_Slots",
        "baseline_r0_milli": "Resource_Aware_Baseline_Reserved_Milli",
        "selected_count": "Resource_Aware_Selected",
        "k_flex_extra_positions": "Resource_Aware_K_Flex_Extra_Positions",
    }
    eligible_mask = table["eligible"]
    for output, source in numeric.items():
        parsed = pd.to_numeric(table[source], errors="coerce")
        bad = eligible_mask & parsed.isna()
        if bool(bad.any()):
            raise AuditBlockedError(
                f"{source}在eligible day含無效值: {table.loc[bad, source].iloc[0]!r}"
            )
        table[output] = parsed.fillna(0)
    int_columns = (
        "pre_market_positions", "free_slots", "orderable_candidates", "baseline_k",
        "physical_free_slots", "selected_count", "k_flex_extra_positions",
    )
    for column in int_columns:
        table[column] = table[column].astype(int)
    table["baseline_r0_milli"] = table["baseline_r0_milli"].astype(np.int64)
    if bool(table.duplicated("trade_date").any()):
        raise AuditBlockedError("daily capacity存在重複trade_date")
    return table


def _normalize_final_trace(frame: pd.DataFrame, *, stage_name: str) -> pd.DataFrame:
    table = pd.DataFrame(frame).copy()
    required = {
        "stage", "stage_rank", "ticker", "trade_date", "signal_date",
        "breakout_quality_score_date", "breakout_quality_score",
        "breakout_quality_daily_score_percentile",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise AuditBlockedError(f"selector trace缺少欄位: {missing}")
    table = table.loc[table["stage"].fillna("").astype(str).eq(stage_name)].copy()
    if table.empty:
        raise AuditBlockedError(f"selector trace沒有{stage_name} stage rows")
    table["ticker"] = table["ticker"].map(normalize_ticker)
    for column in ("trade_date", "signal_date"):
        table[column] = table[column].map(normalize_date)
    score_date = table["breakout_quality_score_date"].map(normalize_date)
    table["score_event_date"] = score_date.where(score_date.ne(""), table["signal_date"])
    rank = pd.to_numeric(table["stage_rank"], errors="coerce")
    if bool(rank.isna().any()) or bool((rank <= 0).any()):
        raise AuditBlockedError("final selector trace stage_rank必須是正整數")
    table["stage_rank"] = rank.astype(int)
    table["score"] = pd.to_numeric(table["breakout_quality_score"], errors="coerce")
    table["score_percentile"] = pd.to_numeric(
        table["breakout_quality_daily_score_percentile"], errors="coerce"
    )
    bad_identity = (
        table["ticker"].eq("")
        | table["trade_date"].eq("")
        | table["signal_date"].eq("")
        | table["score_event_date"].eq("")
    )
    if bool(bad_identity.any()):
        raise AuditBlockedError("final selector trace存在無法解析的event identity")
    table["event_key"] = _event_key(table)
    if bool(table.duplicated("event_key").any()):
        bad = table.loc[table.duplicated("event_key", keep=False), "event_key"].iloc[0]
        raise AuditBlockedError(f"final selector trace event key不唯一: {bad}")
    return table.sort_values(["trade_date", "stage_rank", "ticker"], kind="stable").reset_index(drop=True)


def _normalize_execution(frame: pd.DataFrame) -> pd.DataFrame:
    table = pd.DataFrame(frame).copy()
    required = {
        "execution_order", "ticker", "trade_date", "signal_date", "chosen_qty", "entry_filled"
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise AuditBlockedError(f"execution sidecar缺少欄位: {missing}")
    table["ticker"] = table["ticker"].map(normalize_ticker)
    for column in ("trade_date", "signal_date"):
        table[column] = table[column].map(normalize_date)
    chosen = pd.to_numeric(table["chosen_qty"], errors="coerce")
    if bool(chosen.isna().any()):
        raise AuditBlockedError("execution chosen_qty含無效值")
    table = table.loc[chosen > 0].copy()
    table["entry_filled_bool"] = _as_bool_series(table, "entry_filled")
    table["event_key"] = _event_key(table)
    if bool(table.duplicated("event_key").any()):
        bad = table.loc[table.duplicated("event_key", keep=False), "event_key"].iloc[0]
        raise AuditBlockedError(f"planned execution event key不唯一: {bad}")
    return table.sort_values(["trade_date", "execution_order", "ticker"], kind="stable").reset_index(drop=True)


def _normalize_orderable(frame: pd.DataFrame) -> pd.DataFrame:
    table = pd.DataFrame(frame).copy()
    required = {"ticker", "trade_date", "signal_date"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise AuditBlockedError(f"orderable sidecar缺少欄位: {missing}")
    table["ticker"] = table["ticker"].map(normalize_ticker)
    for column in ("trade_date", "signal_date"):
        table[column] = table[column].map(normalize_date)
    score_date_source = table.get(
        "breakout_quality_score_date", pd.Series("", index=table.index)
    )
    score_date = score_date_source.map(normalize_date)
    table["score_event_date"] = score_date.where(score_date.ne(""), table["signal_date"])
    table = table.loc[
        table["ticker"].ne("") & table["trade_date"].ne("") & table["signal_date"].ne("")
    ].copy()
    table["candidate_event_key"] = (
        table["ticker"].astype(str)
        + "|" + table["signal_date"].astype(str)
        + "|" + table["score_event_date"].astype(str)
    )
    return table.drop_duplicates(["trade_date", "candidate_event_key"]).copy()


def _normalize_path(frame: pd.DataFrame, *, thresholds: Sequence[float]) -> pd.DataFrame:
    table = pd.DataFrame(frame).copy()
    first_date_columns = [
        f"full_horizon_first_upside_{float(value):g}r_date" for value in thresholds
    ]
    # Writer uses decimal-safe tokens (e.g. 1R -> 1r), and current thresholds are 1/2/3.
    first_date_columns = [column.replace(".0r", "r") for column in first_date_columns]
    first_bar_columns = [column.replace("_date", "_bar") for column in first_date_columns]
    required = {
        "ticker", "trade_date", "signal_date", "path_target_available",
        "full_horizon_mfe_r", "full_horizon_adverse_to_peak_r", "realized_r",
        "actual_initial_stop_out", "exit_date", *first_date_columns, *first_bar_columns,
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise AuditBlockedError(f"upside realization sidecar缺少欄位: {missing}")
    table["ticker"] = table["ticker"].map(normalize_ticker)
    for column in ("trade_date", "signal_date"):
        table[column] = table[column].map(normalize_date)
    table["event_key"] = _event_key(table)
    if bool(table.duplicated("event_key").any()):
        bad = table.loc[table.duplicated("event_key", keep=False), "event_key"].iloc[0]
        raise AuditBlockedError(f"upside realization event key不唯一: {bad}")
    table["path_target_available_bool"] = _as_bool_series(table, "path_target_available")
    table["actual_initial_stop_out_bool"] = _as_bool_series(table, "actual_initial_stop_out")
    for column in ("full_horizon_mfe_r", "full_horizon_adverse_to_peak_r", "realized_r"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    return table


def _ordinal_bucket(stage_rank: int, baseline_k: int) -> str:
    rank = int(stage_rank)
    k = int(baseline_k)
    if rank <= k:
        return "baseline_1_to_k"
    offset = rank - k
    if offset == 1:
        return "k_plus_1"
    if offset == 2:
        return "k_plus_2"
    return "k_plus_3_plus"


def build_planned_membership(
    *,
    selector_trace: pd.DataFrame,
    execution: pd.DataFrame,
    daily_capacity: pd.DataFrame,
    final_stage: str = _DEFAULT_FINAL_STAGE,
) -> pd.DataFrame:
    """Return canonical final planned membership with fill and K-relative ordinal."""

    final = _normalize_final_trace(selector_trace, stage_name=final_stage)
    capacity = _normalize_capacity(daily_capacity)
    eligible_capacity = capacity.loc[capacity["eligible"]].copy()
    final = final.merge(
        eligible_capacity[
            [
                "trade_date", "pre_market_positions", "free_slots", "orderable_candidates",
                "baseline_k", "physical_free_slots", "baseline_r0_milli", "selected_count",
                "k_flex_extra_positions",
            ]
        ],
        on="trade_date",
        how="inner",
        validate="many_to_one",
    )
    if final.empty:
        raise AuditBlockedError("final selector trace與eligible daily capacity沒有共同trade date")

    per_day = final.groupby("trade_date", sort=False).size().astype(int)
    expected = eligible_capacity.set_index("trade_date")["selected_count"].astype(int)
    common_dates = per_day.index.intersection(expected.index)
    mismatch = [
        date for date in common_dates if int(per_day.loc[date]) != int(expected.loc[date])
    ]
    if mismatch:
        date = mismatch[0]
        raise AuditBlockedError(
            f"final selector membership與daily selected_count不一致: {date} "
            f"trace={int(per_day.loc[date])} capacity={int(expected.loc[date])}"
        )

    exec_rows = _normalize_execution(execution)
    eligible_dates = set(final["trade_date"].astype(str))
    exec_rows = exec_rows.loc[exec_rows["trade_date"].isin(eligible_dates)].copy()
    final_keys = set(final["event_key"])
    exec_keys = set(exec_rows["event_key"])
    if final_keys != exec_keys:
        missing_exec = sorted(final_keys - exec_keys)[:5]
        extra_exec = sorted(exec_keys - final_keys)[:5]
        raise AuditBlockedError(
            "final selector membership與planned execution不一致: "
            f"missing_execution={missing_exec}, extra_execution={extra_exec}"
        )
    exec_payload = exec_rows[["event_key", "execution_order", "entry_filled_bool"]].copy()
    final = final.merge(exec_payload, on="event_key", how="left", validate="one_to_one")
    final["ordinal_bucket"] = [
        _ordinal_bucket(rank, k)
        for rank, k in zip(final["stage_rank"], final["baseline_k"])
    ]
    if bool((final["stage_rank"] > final["physical_free_slots"]).any()):
        raise AuditBlockedError("final selector stage_rank超過physical free slots")
    return final.sort_values(["trade_date", "stage_rank", "ticker"], kind="stable").reset_index(drop=True)


def strict_matched_trade_dates(
    *,
    c68_orderable: pd.DataFrame,
    c69_orderable: pd.DataFrame,
    c68_capacity: pd.DataFrame,
    c69_capacity: pd.DataFrame,
) -> dict[str, Any]:
    """Identify same-day states strict enough for membership difference attribution."""

    left_o = _normalize_orderable(c68_orderable)
    right_o = _normalize_orderable(c69_orderable)
    left_c = _normalize_capacity(c68_capacity)
    right_c = _normalize_capacity(c69_capacity)
    left_c = left_c.loc[left_c["eligible"]].set_index("trade_date")
    right_c = right_c.loc[right_c["eligible"]].set_index("trade_date")
    common_dates = sorted(set(left_c.index) & set(right_c.index))

    left_sets = left_o.groupby("trade_date", sort=False)["candidate_event_key"].agg(
        lambda values: frozenset(str(v) for v in values)
    ).to_dict()
    right_sets = right_o.groupby("trade_date", sort=False)["candidate_event_key"].agg(
        lambda values: frozenset(str(v) for v in values)
    ).to_dict()
    matched: list[str] = []
    reason_counts = {
        "candidate_set": 0,
        "baseline_k": 0,
        "free_slots": 0,
        "baseline_r0": 0,
        "pre_market_positions": 0,
    }
    for date in common_dates:
        lrow = left_c.loc[date]
        rrow = right_c.loc[date]
        checks = {
            "candidate_set": left_sets.get(date, frozenset()) == right_sets.get(date, frozenset()),
            "baseline_k": int(lrow["baseline_k"]) == int(rrow["baseline_k"]),
            "free_slots": int(lrow["free_slots"]) == int(rrow["free_slots"]),
            "baseline_r0": int(lrow["baseline_r0_milli"]) == int(rrow["baseline_r0_milli"]),
            "pre_market_positions": int(lrow["pre_market_positions"]) == int(rrow["pre_market_positions"]),
        }
        if all(checks.values()):
            matched.append(date)
        else:
            for key, okay in checks.items():
                if not okay:
                    reason_counts[key] += 1
    return {
        "common_eligible_days": int(len(common_dates)),
        "strict_matched_days": int(len(matched)),
        "strict_matched_pct": (
            float(len(matched) / len(common_dates) * 100.0) if common_dates else 0.0
        ),
        "strict_matched_trade_dates": matched,
        "mismatch_day_counts": reason_counts,
    }


def _truth_distribution(truth: pd.DataFrame, cohort: pd.DataFrame) -> dict[str, Any]:
    planned_count = int(len(cohort))
    if planned_count == 0:
        return {
            "truth_covered_rows": 0,
            "truth_coverage_pct": 0.0,
            **{metric.key: None for metric in MFE_SAFETY_QUADRANT_DISTRIBUTION_METRICS},
            "high_mfe_total_pct": None,
            "high_safety_total_pct": None,
        }
    keys = cohort[["ticker", "score_event_date"]].rename(columns={"score_event_date": "date"})
    keys = keys.drop_duplicates(["ticker", "date"])
    covered = keys.merge(truth, on=["ticker", "date"], how="inner", validate="one_to_one")
    counts = covered["quadrant"].value_counts().to_dict()
    result: dict[str, Any] = {
        "truth_covered_rows": int(len(covered)),
        "truth_coverage_pct": float(len(covered) / len(keys) * 100.0) if len(keys) else 0.0,
    }
    for metric in MFE_SAFETY_QUADRANT_DISTRIBUTION_METRICS:
        result[metric.key] = (
            None if covered.empty else float(counts.get(metric.key, 0) / len(covered) * 100.0)
        )
    result["high_mfe_total_pct"] = (
        None
        if covered.empty
        else float(
            result["high_mfe_high_safety_pct"] + result["high_mfe_low_safety_pct"]
        )
    )
    result["high_safety_total_pct"] = (
        None
        if covered.empty
        else float(
            result["high_mfe_high_safety_pct"] + result["low_mfe_high_safety_pct"]
        )
    )
    return result


def _path_summary(
    *,
    cohort: pd.DataFrame,
    path: pd.DataFrame,
    thresholds: Sequence[float],
) -> dict[str, Any]:
    filled = cohort.loc[cohort["entry_filled_bool"]].copy()
    if filled.empty:
        return {
            "filled_count": 0,
            "fill_rate_pct": (0.0 if len(cohort) else None),
            "path_trade_match_count": 0,
            "path_target_covered_count": 0,
            "path_target_coverage_pct": None,
            "full_horizon_mfe_mean_r": None,
            "full_horizon_adverse_to_peak_mean_r": None,
            "realized_mean_r": None,
            **{f"initial_stop_before_{float(value):g}r_rate": None for value in thresholds},
            **{f"first_{float(value):g}r_reached_count": 0 for value in thresholds},
        }
    joined = filled[["event_key"]].merge(
        path,
        on="event_key",
        how="left",
        validate="one_to_one",
    )
    match_count = int(joined["ticker"].notna().sum()) if "ticker" in joined.columns else 0
    if match_count != len(filled):
        missing = int(len(filled) - match_count)
        raise AuditBlockedError(
            f"filled cohort無法完整對回upside realization sidecar: missing={missing}"
        )
    finite_path = (
        joined["path_target_available_bool"]
        & pd.to_numeric(joined["full_horizon_mfe_r"], errors="coerce").map(math.isfinite)
        & pd.to_numeric(joined["full_horizon_adverse_to_peak_r"], errors="coerce").map(math.isfinite)
    )
    covered = joined.loc[finite_path].copy()
    out: dict[str, Any] = {
        "filled_count": int(len(filled)),
        "fill_rate_pct": float(len(filled) / len(cohort) * 100.0),
        "path_trade_match_count": match_count,
        "path_target_covered_count": int(len(covered)),
        "path_target_coverage_pct": float(len(covered) / len(filled) * 100.0),
        "full_horizon_mfe_mean_r": (
            None if covered.empty else float(pd.to_numeric(covered["full_horizon_mfe_r"], errors="coerce").mean())
        ),
        "full_horizon_adverse_to_peak_mean_r": (
            None if covered.empty else float(pd.to_numeric(covered["full_horizon_adverse_to_peak_r"], errors="coerce").mean())
        ),
        "realized_mean_r": (
            None if covered.empty else float(pd.to_numeric(covered["realized_r"], errors="coerce").mean())
        ),
    }
    exit_date = pd.to_datetime(covered.get("exit_date"), errors="coerce")
    initial_stop = covered.get(
        "actual_initial_stop_out_bool", pd.Series(False, index=covered.index)
    ).fillna(False).astype(bool)
    for threshold in thresholds:
        token = f"{float(threshold):g}".replace(".", "p")
        first_date_col = f"full_horizon_first_upside_{token}r_date"
        first_bar_col = f"full_horizon_first_upside_{token}r_bar"
        first_date = pd.to_datetime(covered.get(first_date_col), errors="coerce")
        first_bar = pd.to_numeric(covered.get(first_bar_col), errors="coerce")
        reached = first_date.notna() & first_bar.ge(1)
        reached_count = int(reached.sum())
        initial_before = initial_stop & reached & (exit_date <= first_date)
        out[f"first_{float(threshold):g}r_reached_count"] = reached_count
        out[f"initial_stop_before_{float(threshold):g}r_rate"] = (
            None if not reached_count else float(initial_before.loc[reached].mean() * 100.0)
        )
    return out


def summarize_cohort(
    cohort: pd.DataFrame,
    *,
    truth: pd.DataFrame,
    path: pd.DataFrame,
    thresholds: Sequence[float],
) -> dict[str, Any]:
    table = pd.DataFrame(cohort).copy()
    score_percentile = pd.to_numeric(table.get("score_percentile"), errors="coerce")
    finite_percentile = score_percentile.loc[score_percentile.map(math.isfinite)]
    result: dict[str, Any] = {
        "planned_count": int(len(table)),
        "trade_date_count": int(table["trade_date"].nunique()) if len(table) else 0,
        "score_percentile_mean": (
            None if finite_percentile.empty else float(finite_percentile.mean())
        ),
        "score_percentile_median": (
            None if finite_percentile.empty else float(finite_percentile.median())
        ),
    }
    result.update(_truth_distribution(truth, table))
    result.update(_path_summary(cohort=table, path=path, thresholds=thresholds))
    return result


def build_ordinal_cohorts(
    planned: pd.DataFrame,
    *,
    truth: pd.DataFrame,
    path: pd.DataFrame,
    thresholds: Sequence[float],
) -> dict[str, dict[str, Any]]:
    table = pd.DataFrame(planned).copy()
    cohorts: dict[str, pd.DataFrame] = {
        "baseline_1_to_k": table.loc[table["stage_rank"] <= table["baseline_k"]].copy(),
        "k_plus_1": table.loc[table["stage_rank"] == table["baseline_k"] + 1].copy(),
        "k_plus_2": table.loc[table["stage_rank"] == table["baseline_k"] + 2].copy(),
        "k_plus_3_plus": table.loc[table["stage_rank"] >= table["baseline_k"] + 3].copy(),
        "all_extra": table.loc[table["stage_rank"] > table["baseline_k"]].copy(),
    }
    return {
        key: summarize_cohort(cohorts[key], truth=truth, path=path, thresholds=thresholds)
        for key in _ORDINAL_ORDER
    }


def _strict_matched_membership(
    *,
    matched_dates: Sequence[str],
    c68_planned: pd.DataFrame,
    c69_planned: pd.DataFrame,
    truth: pd.DataFrame,
    c68_path: pd.DataFrame,
    c69_path: pd.DataFrame,
    thresholds: Sequence[float],
) -> dict[str, Any]:
    dates = set(str(value) for value in matched_dates)
    left = c68_planned.loc[c68_planned["trade_date"].isin(dates)].copy()
    right = c69_planned.loc[c69_planned["trade_date"].isin(dates)].copy()
    left_keys = set(left["event_key"])
    right_keys = set(right["event_key"])
    common = left_keys & right_keys
    left_only = left_keys - right_keys
    right_only = right_keys - left_keys
    cohorts = {
        "common": right.loc[right["event_key"].isin(common)].copy(),
        "c68_only": left.loc[left["event_key"].isin(left_only)].copy(),
        "c69_only": right.loc[right["event_key"].isin(right_only)].copy(),
    }
    summaries = {
        "common": summarize_cohort(cohorts["common"], truth=truth, path=c69_path, thresholds=thresholds),
        "c68_only": summarize_cohort(cohorts["c68_only"], truth=truth, path=c68_path, thresholds=thresholds),
        "c69_only": summarize_cohort(cohorts["c69_only"], truth=truth, path=c69_path, thresholds=thresholds),
    }
    per_day_left = left.groupby("trade_date").size() if len(left) else pd.Series(dtype=int)
    per_day_right = right.groupby("trade_date").size() if len(right) else pd.Series(dtype=int)
    all_dates = sorted(dates)
    net_delta = sum(
        int(per_day_right.get(date, 0)) - int(per_day_left.get(date, 0))
        for date in all_dates
    )
    expanded_days = sum(
        1
        for date in all_dates
        if int(per_day_right.get(date, 0)) > int(per_day_left.get(date, 0))
    )
    return {
        "cohorts": summaries,
        "common_planned_count": int(len(common)),
        "c68_only_planned_count": int(len(left_only)),
        "c69_only_planned_count": int(len(right_only)),
        "net_planned_count_delta": int(net_delta),
        "c69_higher_count_days": int(expanded_days),
    }


def _stage_summary(
    *,
    planned: pd.DataFrame,
    truth: pd.DataFrame,
    path: pd.DataFrame,
    thresholds: Sequence[float],
) -> dict[str, Any]:
    filled = planned.loc[planned["entry_filled_bool"]].copy()
    return {
        "planned": summarize_cohort(planned, truth=truth, path=path, thresholds=thresholds),
        "filled": summarize_cohort(filled, truth=truth, path=path, thresholds=thresholds),
    }


def _period_from_source(source: Any) -> tuple[str, str]:
    payload = source.result.get("comparison_period")
    if not isinstance(payload, Mapping):
        raise AuditBlockedError(f"{source.profile_id} Strategy Compare缺少comparison_period")
    start = normalize_date(payload.get("start"))
    end = normalize_date(payload.get("end"))
    if not start or not end:
        raise AuditBlockedError(f"{source.profile_id} comparison_period格式無效")
    return start, end


def _validate_arm_contracts(source: Any, *, control_arm_id: str, treatment_arm_id: str) -> None:
    arms = dict(source.settings.arms)
    if control_arm_id not in arms or treatment_arm_id not in arms:
        raise AuditBlockedError(
            f"{source.profile_id} current/history catalog缺少{control_arm_id}/{treatment_arm_id}"
        )
    left = arms[control_arm_id]
    right = arms[treatment_arm_id]
    shared = (
        "param_source", "param_policy", "rule_policy", "dl_enabled", "dl_id"
    )
    mismatch = [name for name in shared if getattr(left, name) != getattr(right, name)]
    if mismatch:
        raise AuditBlockedError(
            f"{source.profile_id} C68/C69 matched contract已漂移: {mismatch}"
        )
    if left.dl_runtime_mode != "resource-aware-continuous-max-dl-matched-feasible-ascent":
        raise AuditBlockedError(f"{source.profile_id}/{control_arm_id} runtime mode不符matched control")
    if right.dl_runtime_mode != "resource-aware-continuous-max-dl-k-flex-r0-feasible-ascent":
        raise AuditBlockedError(f"{source.profile_id}/{treatment_arm_id} runtime mode不符K-Flex treatment")
    left_options = dict(left.dl_runtime_options or {})
    right_options = dict(right.dl_runtime_options or {})
    if left_options.get("membership_proposal") != right_options.get("membership_proposal"):
        raise AuditBlockedError(f"{source.profile_id} C68/C69 membership proposal不一致")
    if left_options.get("selection_only") is not True or right_options.get("selection_only") is not True:
        raise AuditBlockedError(f"{source.profile_id} C68/C69必須selection_only")
    if left_options.get("preserve_k_r0") is not True or right_options.get("preserve_r0") is not True:
        raise AuditBlockedError(f"{source.profile_id} C68/C69 R0 preservation contract不一致")


def _fingerprint_payload(definition: AuditDefinition, source_refs: Mapping[str, Any]) -> str:
    payload = {
        "audit_schema": 1,
        "definition": definition.as_dict(),
        "source_refs": source_refs,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _fmt_number(value: Any, digits: int = 2, unit: str = "") -> str:
    number = finite_float(value)
    if number is None:
        return "-"
    if digits == 0:
        text = f"{number:.0f}"
    else:
        text = f"{number:.{digits}f}"
    return f"{text}{unit}"


def _cohort_table_rows(cohorts: Mapping[str, Mapping[str, Any]], order: Sequence[str], labels: Mapping[str, str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for key in order:
        item = dict(cohorts.get(key) or {})
        rows.append([
            labels.get(key, key),
            _fmt_number(item.get("planned_count"), 0),
            _fmt_number(item.get("filled_count"), 0),
            _fmt_number(item.get("fill_rate_pct"), 2, "%"),
            _fmt_number(item.get("score_percentile_mean"), 3),
            _fmt_number(item.get("high_mfe_total_pct"), 2, "%"),
            _fmt_number(item.get("high_mfe_high_safety_pct"), 2, "%"),
            _fmt_number(item.get("high_mfe_low_safety_pct"), 2, "%"),
            _fmt_number(item.get("full_horizon_mfe_mean_r"), 2, "R"),
            _fmt_number(item.get("full_horizon_adverse_to_peak_mean_r"), 2, "R"),
            _fmt_number(item.get("realized_mean_r"), 2, "R"),
            _fmt_number(item.get("initial_stop_before_1r_rate"), 2, "%"),
            _fmt_number(item.get("initial_stop_before_2r_rate"), 2, "%"),
            _fmt_number(item.get("initial_stop_before_3r_rate"), 2, "%"),
        ])
    return rows


def render_result(result: Mapping[str, Any]) -> str:
    from core.console_report import render_key_values, render_section, render_table, render_title

    lines = [render_title("C68 / C69 Marginal Position Attribution Audit")]
    lines.append(render_key_values([
        ("Audit", result.get("audit_id")),
        ("Threshold", f"same-day percentile >= {result.get('percentile_cutoff', 0.5):.2f} = High"),
        ("Control", result.get("control_arm_id")),
        ("Treatment", result.get("treatment_arm_id")),
        ("決策問題", result.get("decision_question")),
    ]))
    headers = [
        "Cohort", "Planned", "Filled", "Fill", "Score %ile", "High-MFE", "HM/HS", "HM/LS",
        "Full-MFE", "Adverse", "Realized EV", "+1R前初始Stop", "+2R前初始Stop", "+3R前初始Stop",
    ]
    for profile_id in result.get("evaluation_order", []):
        payload = result["evaluations"][profile_id]
        display = payload["display_name"]
        lines.append(render_section(f"{display}｜C68/C69 Planned / Filled geometry"))
        stage_rows = []
        for arm_id in (result["control_arm_id"], result["treatment_arm_id"]):
            for stage in ("planned", "filled"):
                item = payload["arms"][arm_id]["stage"][stage]
                stage_rows.append([
                    arm_id, stage.title(), _fmt_number(item.get("planned_count"), 0),
                    _fmt_number(item.get("truth_coverage_pct"), 2, "%"),
                    _fmt_number(item.get("high_mfe_high_safety_pct"), 2, "%"),
                    _fmt_number(item.get("high_mfe_low_safety_pct"), 2, "%"),
                    _fmt_number(item.get("low_mfe_high_safety_pct"), 2, "%"),
                    _fmt_number(item.get("low_mfe_low_safety_pct"), 2, "%"),
                    _fmt_number(item.get("high_mfe_total_pct"), 2, "%"),
                ])
        lines.append(render_table(
            ["Arm", "Stage", "N", "Coverage", "HM/HS", "HM/LS", "LM/HS", "LM/LS", "High-MFE"],
            stage_rows,
        ))
        lines.append(render_section(f"{display}｜C69 K-relative ordinal cohorts"))
        lines.append(render_table(
            headers,
            _cohort_table_rows(payload["c69_ordinal_cohorts"], _ORDINAL_ORDER, _ORDINAL_DISPLAY),
        ))
        matched = payload["strict_matched_state"]
        lines.append(render_section(f"{display}｜Strict matched candidate-state C69−C68"))
        lines.append(render_key_values([
            ("Common eligible days", matched["common_eligible_days"]),
            ("Strict matched days", f"{matched['strict_matched_days']} ({matched['strict_matched_pct']:.2f}%)"),
            ("Net planned count delta", matched["membership"]["net_planned_count_delta"]),
            ("C69 higher-count days", matched["membership"]["c69_higher_count_days"]),
        ]))
        lines.append(render_table(
            headers,
            _cohort_table_rows(
                matched["membership"]["cohorts"], _MATCHED_COHORT_ORDER, _MATCHED_COHORT_DISPLAY
            ),
        ))
    lines.append(render_section("判讀邊界"))
    lines.append(
        "Ordinal K+1/K+2/K+3+ 是C69自身當日baseline K之後的結構性extra slots；"
        "C69-only/C68-only只在候選集合、baseline K、free slots、R0與盤前持股數完全一致的strict matched days計算。"
        "兩者互補但不等價：K-Flex可為了更大count重新配置前K membership，因此不得把單一extra row當成完整因果反事實。"
    )
    lines.append(
        "Audit只讀既有Strategy Compare與Future Truth/path sidecars；Future truth不參與任何runtime selection。"
    )
    return "\n".join(lines)


def _render_markdown(result: Mapping[str, Any]) -> str:
    text = render_result(result)
    return "# C68 / C69 Marginal Position Attribution Audit\n\n```text\n" + text + "\n```\n"


def _write_summary_csv(path: Path, result: Mapping[str, Any], *, section: str) -> None:
    rows: list[dict[str, Any]] = []
    for profile_id in result.get("evaluation_order", []):
        payload = result["evaluations"][profile_id]
        if section == "ordinal":
            cohorts = payload["c69_ordinal_cohorts"]
            for key in _ORDINAL_ORDER:
                rows.append({"profile_id": profile_id, "cohort": key, **dict(cohorts[key])})
        elif section == "matched":
            cohorts = payload["strict_matched_state"]["membership"]["cohorts"]
            for key in _MATCHED_COHORT_ORDER:
                rows.append({"profile_id": profile_id, "cohort": key, **dict(cohorts[key])})
        elif section == "stage":
            for arm_id, arm in payload["arms"].items():
                for stage, summary in arm["stage"].items():
                    rows.append({"profile_id": profile_id, "arm_id": arm_id, "stage": stage, **dict(summary)})
        else:
            raise ValueError(f"unknown CSV section: {section}")
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _detail_rows_for_export(
    *,
    profile_id: str,
    c69_planned: pd.DataFrame,
    matched_dates: Sequence[str],
    c68_planned: pd.DataFrame,
    truth: pd.DataFrame,
    path: pd.DataFrame,
) -> pd.DataFrame:
    table = c69_planned.copy()
    table["audit_profile_id"] = profile_id
    table["structural_extra"] = table["stage_rank"] > table["baseline_k"]
    matched = set(str(v) for v in matched_dates)
    c68_keys = set(c68_planned.loc[c68_planned["trade_date"].isin(matched), "event_key"])
    table["strict_matched_day"] = table["trade_date"].isin(matched)
    table["strict_c69_only"] = table["strict_matched_day"] & ~table["event_key"].isin(c68_keys)
    truth_join = truth.rename(columns={"date": "score_event_date"})
    table = table.merge(
        truth_join[
            ["ticker", "score_event_date", "mfe_percentile", "safety_percentile", "quadrant"]
        ],
        on=["ticker", "score_event_date"],
        how="left",
        validate="many_to_one",
    )
    path_columns = [
        "event_key", "path_target_available_bool", "full_horizon_mfe_r",
        "full_horizon_adverse_to_peak_r", "realized_r", "actual_initial_stop_out_bool",
        "exit_date", "full_horizon_first_upside_1r_date", "full_horizon_first_upside_2r_date",
        "full_horizon_first_upside_3r_date",
    ]
    available = [column for column in path_columns if column in path.columns]
    table = table.merge(path[available], on="event_key", how="left", validate="one_to_one")
    keep = [
        "audit_profile_id", "trade_date", "ticker", "signal_date", "score_event_date",
        "stage_rank", "baseline_k", "physical_free_slots", "ordinal_bucket",
        "structural_extra", "strict_matched_day", "strict_c69_only", "entry_filled_bool",
        "score", "score_percentile", "mfe_percentile", "safety_percentile", "quadrant",
        "path_target_available_bool", "full_horizon_mfe_r", "full_horizon_adverse_to_peak_r",
        "realized_r", "actual_initial_stop_out_bool", "exit_date",
        "full_horizon_first_upside_1r_date", "full_horizon_first_upside_2r_date",
        "full_horizon_first_upside_3r_date",
    ]
    return table[[column for column in keep if column in table.columns]].copy()


def run_audit(definition: AuditDefinition, *, project_root: Path) -> dict[str, Any]:
    if definition.audit_type != SUPPORTED_AUDIT_TYPE:
        raise ValueError(f"不支援audit_type={definition.audit_type}")
    root = Path(project_root).resolve()
    source_cfg = definition.source
    dims = definition.dimensions
    evaluation_order = tuple(str(value) for value in source_cfg.get("evaluation_profile_ids", ()))
    control_arm_id = str(source_cfg.get("control_arm_id") or "").strip()
    treatment_arm_id = str(source_cfg.get("treatment_arm_id") or "").strip()
    if not evaluation_order or not control_arm_id or not treatment_arm_id:
        raise AuditBlockedError("C69 marginal Audit source設定不完整")
    cutoff = float(dims.get("percentile_cutoff", 0.50))
    percentile_method = str(dims.get("percentile_method") or "average_zero_based")
    final_stage = str(dims.get("final_selector_stage") or _DEFAULT_FINAL_STAGE)
    thresholds = tuple(float(value) for value in dims.get("first_passage_thresholds_r", (1, 2, 3)))

    truth, truth_source = build_truth_geometry(
        definition, root, percentile_method=percentile_method
    )
    truth = attach_quadrants(truth, cutoff=cutoff)

    evaluations: dict[str, Any] = {}
    source_refs: dict[str, Any] = {"truth": truth_source, "strategy": {}}
    detail_frames: list[pd.DataFrame] = []
    for profile_id in evaluation_order:
        pinned = str(
            dict(source_cfg.get("strategy_result_fingerprints") or {}).get(profile_id) or ""
        ).strip()
        strategy_source = load_strategy_compare_source(
            root, profile_id=profile_id, pinned_config_fingerprint=pinned or None
        )
        _validate_arm_contracts(
            strategy_source,
            control_arm_id=control_arm_id,
            treatment_arm_id=treatment_arm_id,
        )
        period = _period_from_source(strategy_source)
        truth_period = truth.loc[(truth["date"] >= period[0]) & (truth["date"] <= period[1])].copy()
        if truth_period.empty:
            raise AuditBlockedError(f"{profile_id} period與truth沒有交集")

        arm_data: dict[str, Any] = {}
        arm_sources: dict[str, Any] = {}
        planned_tables: dict[str, pd.DataFrame] = {}
        path_tables: dict[str, pd.DataFrame] = {}
        evidences: dict[str, dict[str, Any]] = {}
        for arm_id in (control_arm_id, treatment_arm_id):
            evidence = load_strategy_arm_path_sidecars(
                root, source=strategy_source, arm_id=arm_id
            )
            evidences[arm_id] = evidence
            planned = build_planned_membership(
                selector_trace=pd.DataFrame(evidence["selector_trace"]),
                execution=pd.DataFrame(evidence["execution"]),
                daily_capacity=pd.DataFrame(evidence["daily_capacity"]),
                final_stage=final_stage,
            )
            path = _normalize_path(pd.DataFrame(evidence["upside_realization"]), thresholds=thresholds)
            planned_tables[arm_id] = planned
            path_tables[arm_id] = path
            arm_data[arm_id] = {
                "stage": _stage_summary(
                    planned=planned, truth=truth_period, path=path, thresholds=thresholds
                ),
                "eligible_days": int(planned["trade_date"].nunique()),
                "planned_count": int(len(planned)),
                "filled_count": int(planned["entry_filled_bool"].sum()),
            }
            arm_sources[arm_id] = {
                "pair_dir": _relative(Path(evidence["pair_dir"]), root),
                "orderable": _relative(Path(evidence["orderable_path"]), root),
                "selector_trace": _relative(Path(evidence["selector_trace_path"]), root),
                "execution": _relative(Path(evidence["execution_path"]), root),
                "daily_capacity": _relative(Path(evidence["daily_capacity_path"]), root),
                "upside_realization": _relative(Path(evidence["upside_realization_path"]), root),
            }

        c69_planned = planned_tables[treatment_arm_id]
        c69_path = path_tables[treatment_arm_id]
        ordinal = build_ordinal_cohorts(
            c69_planned, truth=truth_period, path=c69_path, thresholds=thresholds
        )
        matched = strict_matched_trade_dates(
            c68_orderable=pd.DataFrame(evidences[control_arm_id]["orderable"]),
            c69_orderable=pd.DataFrame(evidences[treatment_arm_id]["orderable"]),
            c68_capacity=pd.DataFrame(evidences[control_arm_id]["daily_capacity"]),
            c69_capacity=pd.DataFrame(evidences[treatment_arm_id]["daily_capacity"]),
        )
        matched_membership = _strict_matched_membership(
            matched_dates=matched["strict_matched_trade_dates"],
            c68_planned=planned_tables[control_arm_id],
            c69_planned=planned_tables[treatment_arm_id],
            truth=truth_period,
            c68_path=path_tables[control_arm_id],
            c69_path=path_tables[treatment_arm_id],
            thresholds=thresholds,
        )
        matched["membership"] = matched_membership
        expanded = c69_planned.loc[c69_planned["stage_rank"] > c69_planned["baseline_k"]]
        expanded_days = int(expanded["trade_date"].nunique()) if len(expanded) else 0
        extra_slot_days = int(len(expanded))

        evaluations[profile_id] = {
            "display_name": strategy_source.settings.profile_label,
            "period": {"start": period[0], "end": period[1]},
            "strategy_config_fingerprint": strategy_source.config_fingerprint,
            "arms": arm_data,
            "c69_k_flex_mechanism": {
                "expanded_trade_days": expanded_days,
                "structural_extra_slot_days": extra_slot_days,
                "c69_planned_days": int(c69_planned["trade_date"].nunique()),
            },
            "c69_ordinal_cohorts": ordinal,
            "strict_matched_state": matched,
        }
        source_refs["strategy"][profile_id] = {
            "run_dir": _relative(strategy_source.run_dir, root),
            "config_fingerprint": strategy_source.config_fingerprint,
            "arms": arm_sources,
        }
        detail_frames.append(_detail_rows_for_export(
            profile_id=profile_id,
            c69_planned=c69_planned,
            matched_dates=matched["strict_matched_trade_dates"],
            c68_planned=planned_tables[control_arm_id],
            truth=truth_period,
            path=c69_path,
        ))

    fingerprint = _fingerprint_payload(definition, source_refs)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_root = root / AUDIT_OUTPUT_ROOT / definition.output_subdir
    run_dir = output_root / "runs" / f"{timestamp}_{fingerprint}"
    run_dir.mkdir(parents=True, exist_ok=False)
    json_path = run_dir / "c69_marginal_position_attribution.json"
    stage_csv = run_dir / "stage_geometry.csv"
    ordinal_csv = run_dir / "ordinal_cohorts.csv"
    matched_csv = run_dir / "matched_state_cohorts.csv"
    detail_csv = run_dir / "c69_marginal_detail.csv"
    report_path = run_dir / "report.md"
    manifest_path = run_dir / "manifest.json"

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
        "control_arm_id": control_arm_id,
        "treatment_arm_id": treatment_arm_id,
        "percentile_cutoff": cutoff,
        "percentile_method": percentile_method,
        "first_passage_thresholds_r": list(thresholds),
        "evaluation_order": list(evaluation_order),
        "truth_sources": truth_source,
        "evaluations": evaluations,
        "interpretation_boundary": {
            "ordinal_extra": "C69 stage_rank > same-day internal baseline K; identifies structural extra slots, not a complete counterfactual membership decomposition",
            "strict_matched_state": "Only days with identical orderable candidate event sets, baseline K, free slots, R0 and pre-market positions across C68/C69",
            "portfolio_path": "Outside strict matched days, earlier portfolio-path divergence can change cash/holdings/candidate opportunity set",
            "future_truth": "MFE/Safety/path truth is post-replay diagnostic only and never used for runtime selection",
        },
        "artifacts": {
            "完整JSON": _relative(json_path, root),
            "Stage CSV": _relative(stage_csv, root),
            "Ordinal CSV": _relative(ordinal_csv, root),
            "Matched CSV": _relative(matched_csv, root),
            "Marginal Detail CSV": _relative(detail_csv, root),
            "詳細Markdown": _relative(report_path, root),
            "Manifest": _relative(manifest_path, root),
        },
    }
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary_csv(stage_csv, result, section="stage")
    _write_summary_csv(ordinal_csv, result, section="ordinal")
    _write_summary_csv(matched_csv, result, section="matched")
    pd.concat(detail_frames, ignore_index=True).to_csv(detail_csv, index=False, encoding="utf-8-sig")
    report_path.write_text(_render_markdown(result), encoding="utf-8")
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
    latest_dir = output_root / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    (latest_dir / "audit.md").write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    (latest_dir / "audit.json").write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    for source_path, filename in (
        (stage_csv, "stage_geometry.csv"),
        (ordinal_csv, "ordinal_cohorts.csv"),
        (matched_csv, "matched_state_cohorts.csv"),
        (detail_csv, "c69_marginal_detail.csv"),
        (manifest_path, "manifest.json"),
    ):
        (latest_dir / filename).write_bytes(source_path.read_bytes())
    return result


def preflight(definition: AuditDefinition, *, project_root: Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    blockers: list[str] = []
    target_paths: list[str] = []
    strategy_paths: list[str] = []
    try:
        filter_id, _arch, provider_profile_id, mfe_target_id, safety_target_id = (
            resolve_truth_provider_contract(definition)
        )
        dataset_dir = resolve_filter_output_dir(root, filter_id=filter_id)
        dataset = resolve_dataset_paths(dataset_dir)
        target_paths.append(_relative(dataset.summary, root))
        if not dataset.summary.is_file():
            blockers.append("缺少canonical Dataset summary: " + _relative(dataset.summary, root))
        target_paths.append(
            f"profile:{provider_profile_id} / target:{mfe_target_id} / safety:{safety_target_id}"
        )
    except (ValueError, OSError) as exc:
        blockers.append(str(exc))

    source_cfg = definition.source
    control_arm_id = str(source_cfg.get("control_arm_id") or "").strip()
    treatment_arm_id = str(source_cfg.get("treatment_arm_id") or "").strip()
    final_stage = str(definition.dimensions.get("final_selector_stage") or _DEFAULT_FINAL_STAGE)
    thresholds = tuple(
        float(value) for value in definition.dimensions.get("first_passage_thresholds_r", (1, 2, 3))
    )
    for profile_id in tuple(str(v) for v in source_cfg.get("evaluation_profile_ids", ())):
        try:
            pinned = str(
                dict(source_cfg.get("strategy_result_fingerprints") or {}).get(profile_id) or ""
            ).strip()
            source = load_strategy_compare_source(
                root, profile_id=profile_id, pinned_config_fingerprint=pinned or None
            )
            _validate_arm_contracts(
                source, control_arm_id=control_arm_id, treatment_arm_id=treatment_arm_id
            )
            strategy_paths.append(_relative(source.run_dir, root))
            for arm_id in (control_arm_id, treatment_arm_id):
                evidence = load_strategy_arm_path_sidecars(root, source=source, arm_id=arm_id)
                planned = build_planned_membership(
                    selector_trace=pd.DataFrame(evidence["selector_trace"]),
                    execution=pd.DataFrame(evidence["execution"]),
                    daily_capacity=pd.DataFrame(evidence["daily_capacity"]),
                    final_stage=final_stage,
                )
                if planned.empty:
                    raise AuditBlockedError(f"{profile_id}/{arm_id} final planned membership為空")
                path = _normalize_path(pd.DataFrame(evidence["upside_realization"]), thresholds=thresholds)
                filled_keys = set(planned.loc[planned["entry_filled_bool"], "event_key"])
                path_keys = set(path["event_key"])
                if not filled_keys.issubset(path_keys):
                    raise AuditBlockedError(
                        f"{profile_id}/{arm_id} upside realization缺少filled trade evidence"
                    )
            strict_matched_trade_dates(
                c68_orderable=pd.DataFrame(
                    load_strategy_arm_path_sidecars(root, source=source, arm_id=control_arm_id)["orderable"]
                ),
                c69_orderable=pd.DataFrame(
                    load_strategy_arm_path_sidecars(root, source=source, arm_id=treatment_arm_id)["orderable"]
                ),
                c68_capacity=pd.DataFrame(
                    load_strategy_arm_path_sidecars(root, source=source, arm_id=control_arm_id)["daily_capacity"]
                ),
                c69_capacity=pd.DataFrame(
                    load_strategy_arm_path_sidecars(root, source=source, arm_id=treatment_arm_id)["daily_capacity"]
                ),
            )
        except (ValueError, KeyError, OSError, AuditSourceBlockedError, AuditBlockedError) as exc:
            blockers.append(str(exc))
    return {
        "status": "READY" if not blockers else "BLOCKED",
        "blockers": blockers,
        "target_paths": target_paths,
        "strategy_paths": strategy_paths,
    }


def collect_status(definition: AuditDefinition, *, project_root: Path) -> dict[str, Any]:
    state = preflight(definition, project_root=Path(project_root))
    return {
        "status": str(state.get("status") or "BLOCKED"),
        "reason": "；".join(str(v) for v in state.get("blockers", ())),
        "source": {
            "display": "Completed C68/C69 OOS/Rolling selector/execution/path sidecars + canonical MFE/Safety truth",
            "target_paths": list(state.get("target_paths", ())),
            "strategy_paths": list(state.get("strategy_paths", ())),
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
    "build_ordinal_cohorts",
    "build_planned_membership",
    "collect_status",
    "preflight",
    "render_result",
    "run_audit",
    "run_formal_audit",
    "strict_matched_trade_dates",
    "summarize_cohort",
]
