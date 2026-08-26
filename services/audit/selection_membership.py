"""Reusable read-only selector membership normalization for Audit reports."""

from __future__ import annotations

from typing import Any

import pandas as pd

from services.audit.mfe_safety_truth import AuditBlockedError, normalize_date, normalize_ticker


def _as_bool_series(frame: pd.DataFrame, column: str, *, default: bool = False) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(bool(default), index=frame.index, dtype=bool)
    raw = frame[column]
    if raw.dtype == bool:
        return raw.fillna(bool(default)).astype(bool)
    return raw.fillna(str(default)).astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _event_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["ticker"].astype(str)
        + "|"
        + frame["trade_date"].astype(str)
        + "|"
        + frame["signal_date"].astype(str)
    )


def normalize_orderable_membership(frame: pd.DataFrame) -> pd.DataFrame:
    table = pd.DataFrame(frame).copy()
    required = {"ticker", "trade_date", "signal_date"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise AuditBlockedError(f"orderable sidecar缺少欄位: {missing}")
    table["ticker"] = table["ticker"].map(normalize_ticker)
    table["trade_date"] = table["trade_date"].map(normalize_date)
    table["signal_date"] = table["signal_date"].map(normalize_date)
    score_date = table.get("breakout_quality_score_date", pd.Series("", index=table.index)).map(normalize_date)
    table["score_event_date"] = score_date.where(score_date.ne(""), table["signal_date"])
    table = table.loc[
        table["ticker"].ne("")
        & table["trade_date"].ne("")
        & table["signal_date"].ne("")
        & table["score_event_date"].ne("")
    ].copy()
    table["event_key"] = _event_key(table)
    return table.drop_duplicates(["trade_date", "event_key"], keep="first").reset_index(drop=True)


def normalize_final_selector_trace(frame: pd.DataFrame, *, final_stage: str) -> pd.DataFrame:
    table = pd.DataFrame(frame).copy()
    required = {
        "stage", "stage_rank", "ticker", "trade_date", "signal_date",
        "breakout_quality_score_date", "breakout_quality_score",
        "breakout_quality_daily_score_percentile",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise AuditBlockedError(f"selector trace缺少欄位: {missing}")
    table = table.loc[table["stage"].fillna("").astype(str).eq(str(final_stage))].copy()
    if table.empty:
        raise AuditBlockedError(f"selector trace沒有{final_stage} stage rows")
    table["ticker"] = table["ticker"].map(normalize_ticker)
    table["trade_date"] = table["trade_date"].map(normalize_date)
    table["signal_date"] = table["signal_date"].map(normalize_date)
    score_date = table["breakout_quality_score_date"].map(normalize_date)
    table["score_event_date"] = score_date.where(score_date.ne(""), table["signal_date"])
    table["stage_rank"] = pd.to_numeric(table["stage_rank"], errors="coerce")
    if bool(table["stage_rank"].isna().any()) or bool((table["stage_rank"] <= 0).any()):
        raise AuditBlockedError("final selector trace stage_rank必須是正整數")
    table["stage_rank"] = table["stage_rank"].astype(int)
    table["score"] = pd.to_numeric(table["breakout_quality_score"], errors="coerce")
    table["score_percentile"] = pd.to_numeric(
        table["breakout_quality_daily_score_percentile"], errors="coerce"
    )
    table["event_key"] = _event_key(table)
    if bool(table["event_key"].duplicated().any()):
        raise AuditBlockedError("final selector trace event_key不唯一")
    return table.sort_values(["trade_date", "stage_rank", "ticker"], kind="stable").reset_index(drop=True)


def normalize_planned_execution(frame: pd.DataFrame) -> pd.DataFrame:
    table = pd.DataFrame(frame).copy()
    required = {"execution_order", "ticker", "trade_date", "signal_date", "chosen_qty", "entry_filled"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise AuditBlockedError(f"execution sidecar缺少欄位: {missing}")
    table["ticker"] = table["ticker"].map(normalize_ticker)
    table["trade_date"] = table["trade_date"].map(normalize_date)
    table["signal_date"] = table["signal_date"].map(normalize_date)
    chosen = pd.to_numeric(table["chosen_qty"], errors="coerce")
    if bool(chosen.isna().any()):
        raise AuditBlockedError("execution chosen_qty含無效值")
    table = table.loc[chosen > 0].copy()
    table["entry_filled_bool"] = _as_bool_series(table, "entry_filled")
    table["event_key"] = _event_key(table)
    if bool(table["event_key"].duplicated().any()):
        raise AuditBlockedError("planned execution event_key不唯一")
    return table.sort_values(["trade_date", "execution_order", "ticker"], kind="stable").reset_index(drop=True)


def build_planned_membership(
    *,
    selector_trace: pd.DataFrame,
    execution: pd.DataFrame,
    final_stage: str,
) -> pd.DataFrame:
    final = normalize_final_selector_trace(selector_trace, final_stage=final_stage)
    planned = normalize_planned_execution(execution)
    final_keys = set(final["event_key"])
    planned_keys = set(planned["event_key"])
    if final_keys != planned_keys:
        raise AuditBlockedError(
            "final selector membership與planned execution不一致: "
            f"missing_execution={len(final_keys-planned_keys)}, extra_execution={len(planned_keys-final_keys)}"
        )
    return final.merge(
        planned[["event_key", "execution_order", "entry_filled_bool"]],
        on="event_key",
        how="left",
        validate="one_to_one",
    ).sort_values(["trade_date", "stage_rank", "ticker"], kind="stable").reset_index(drop=True)


def truth_keys(frame: pd.DataFrame, *, date_column: str = "score_event_date") -> pd.DataFrame:
    table = pd.DataFrame(frame).copy()
    if not {"ticker", date_column}.issubset(table.columns):
        raise AuditBlockedError(f"membership缺少truth key欄位: ticker/{date_column}")
    keys = pd.DataFrame({
        "ticker": table["ticker"].map(normalize_ticker),
        "date": table[date_column].map(normalize_date),
    })
    return keys.loc[keys["ticker"].ne("") & keys["date"].ne("")].drop_duplicates(["ticker", "date"])


def pair_membership_cohorts(control: pd.DataFrame, treatment: pd.DataFrame) -> dict[str, pd.DataFrame]:
    left = pd.DataFrame(control).copy()
    right = pd.DataFrame(treatment).copy()
    left_keys = set(left["event_key"].astype(str))
    right_keys = set(right["event_key"].astype(str))
    common = left_keys & right_keys
    return {
        "common": right.loc[right["event_key"].astype(str).isin(common)].copy(),
        "control_only": left.loc[left["event_key"].astype(str).isin(left_keys - right_keys)].copy(),
        "treatment_only": right.loc[right["event_key"].astype(str).isin(right_keys - left_keys)].copy(),
    }


__all__ = [
    "build_planned_membership",
    "normalize_final_selector_trace",
    "normalize_orderable_membership",
    "normalize_planned_execution",
    "pair_membership_cohorts",
    "truth_keys",
]
