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
    table["score"] = pd.to_numeric(
        table.get("breakout_quality_score", pd.Series(index=table.index, dtype=float)),
        errors="coerce",
    )
    table["score_percentile"] = pd.to_numeric(
        table.get(
            "breakout_quality_daily_score_percentile",
            pd.Series(index=table.index, dtype=float),
        ),
        errors="coerce",
    )
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
    orderable: pd.DataFrame | None = None,
    execution: pd.DataFrame,
    selector_trace: pd.DataFrame | None = None,
    final_stage: str | None = None,
) -> pd.DataFrame:
    """Return canonical pre-market planned membership from execution evidence.

    ``score_ranking_execution.csv`` is the selector-agnostic canonical evidence of
    orders that survived ranking, resource selection and cash-capped entry planning.
    It therefore owns reusable ``Planned`` membership.  Selector-stage trace is an
    implementation-specific observability sidecar (for example Max-DL repair/ascent)
    and must not be required by generic reports; No-K/No-R0 selectors legitimately
    have no ``feasible_ascent_final`` rows.

    ``selector_trace``/``final_stage`` remain optional only as a compatibility
    cross-check when a caller explicitly supplies them.  Absence of that stage is
    never a blocker for reusable planned-membership reports.
    """

    planned = normalize_planned_execution(execution)
    if orderable is None:
        if selector_trace is None or not str(final_stage or "").strip():
            raise AuditBlockedError(
                "planned membership需要orderable sidecar；selector trace只可作相容性fallback"
            )
        # Legacy fallback for callers that still own a selector-stage scientific
        # contract.  New reusable reports must pass ``orderable`` instead.
        final = normalize_final_selector_trace(
            selector_trace, final_stage=str(final_stage)
        )
        final_keys = set(final["event_key"])
        planned_keys = set(planned["event_key"])
        if final_keys != planned_keys:
            raise AuditBlockedError(
                "final selector membership與planned execution不一致: "
                f"missing_execution={len(final_keys-planned_keys)}, "
                f"extra_execution={len(planned_keys-final_keys)}"
            )
        return final.merge(
            planned[["event_key", "execution_order", "entry_filled_bool"]],
            on="event_key",
            how="left",
            validate="one_to_one",
        ).sort_values(
            ["trade_date", "stage_rank", "ticker"], kind="stable"
        ).reset_index(drop=True)

    orderable_table = normalize_orderable_membership(orderable)
    orderable_keys = set(orderable_table["event_key"])
    planned_keys = set(planned["event_key"])
    missing_orderable = planned_keys - orderable_keys
    if missing_orderable:
        raise AuditBlockedError(
            "planned execution存在不在orderable sidecar的event: "
            f"count={len(missing_orderable)}"
        )
    score_columns = [
        "event_key",
        "score_event_date",
        "score",
        "score_percentile",
    ]
    merged = planned.merge(
        orderable_table[score_columns],
        on="event_key",
        how="left",
        validate="one_to_one",
    )
    merged["stage"] = "planned_execution"
    merged = merged.sort_values(
        ["trade_date", "execution_order", "ticker"], kind="stable"
    ).reset_index(drop=True)
    merged["stage_rank"] = (
        merged.groupby("trade_date", sort=False).cumcount() + 1
    ).astype(int)

    # If a compatible final-stage trace is present, validate it.  A missing stage
    # is expected for selector families that do not use Max-DL repair/ascent.
    if selector_trace is not None and str(final_stage or "").strip():
        trace_table = pd.DataFrame(selector_trace)
        if "stage" in trace_table.columns and bool(
            trace_table["stage"].fillna("").astype(str).eq(str(final_stage)).any()
        ):
            final = normalize_final_selector_trace(
                trace_table, final_stage=str(final_stage)
            )
            final_keys = set(final["event_key"])
            if final_keys != planned_keys:
                raise AuditBlockedError(
                    "selector final stage與planned execution不一致: "
                    f"missing_execution={len(final_keys-planned_keys)}, "
                    f"extra_execution={len(planned_keys-final_keys)}"
                )
    return merged


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
