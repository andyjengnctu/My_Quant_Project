"""Read-only C80/C81 allocator/path attribution.

This one-time Audit compares the same MR-13AH score source under two allocator
contracts:

* C80: exact K/R0/canonical-cash constrained basket optimization.
* C81: direct model-score order with no K target/minimum and no R0 floor.

The Audit never trains, scores, replays, or backfills strategy evidence.  It reads
only completed Strategy Compare sidecars plus canonical MFE/Safety truth and market
Close data for exact peak-to-trough MTM reconciliation.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from config.audit import AUDIT_OUTPUT_ROOT, AuditDefinition
from core.console_report import render_key_values, render_section, render_table, render_title
from core.data_utils import discover_unique_csv_map
from core.dataset_profiles import get_dataset_dir
from core.path_utils import project_relative_display_path
from filters.breakout_quality.mfe_safety_geometry import (
    attach_quadrants,
    distribution_for_keys,
    filter_period,
    normalize_ticker,
)
from services.audit.mfe_safety_truth import (
    AuditBlockedError,
    build_truth_geometry,
    finite_float,
    normalize_date,
    resolve_truth_provider_contract,
)
from services.audit.portfolio_drawdown import _capital_summary
from services.audit.portfolio_mtm import build_drawdown_analysis
from services.audit.selection_membership import build_planned_membership, truth_keys
from services.audit.selection_resource_constraints import (
    _raw_top_k_with_score_event,
    _validate_raw_stage_capacity,
    summarize_resource_contract,
)
from services.audit.strategy_compare_source import (
    AuditSourceBlockedError,
    load_strategy_arm_path_sidecars,
    load_strategy_arm_pipeline_sidecars,
    load_strategy_compare_source,
)
from services.audit.trade_outcome_path import normalize_path, summarize_path_cohort

SUPPORTED_AUDIT_TYPE = "c80_c81_allocator_path_attribution"

_COHORT_ORDER = ("common_c80", "common_c81", "c80_only", "c81_only")
_COHORT_LABEL = {
    "common_c80": "Common planned | C80 path",
    "common_c81": "Common planned | C81 path",
    "c80_only": "C80-only planned",
    "c81_only": "C81-only planned",
}


def _relative(path: Path, root: Path) -> str:
    return project_relative_display_path(Path(path), project_root=Path(root))


def _period(source) -> tuple[str, str]:
    payload = dict(source.result.get("comparison_period") or {})
    start = normalize_date(payload.get("start"))
    end = normalize_date(payload.get("end"))
    if not start or not end:
        raise AuditBlockedError(f"{source.profile_id} Strategy Compare缺少comparison_period")
    return start, end


def _event_key_set(frame: pd.DataFrame) -> set[str]:
    table = pd.DataFrame(frame)
    if "event_key" not in table.columns:
        raise AuditBlockedError("membership缺少event_key")
    return set(table["event_key"].fillna("").astype(str)) - {""}


def _membership_subset(frame: pd.DataFrame, keys: set[str]) -> pd.DataFrame:
    table = pd.DataFrame(frame).copy()
    return table.loc[table["event_key"].astype(str).isin(keys)].copy()


def _truth_summary(frame: pd.DataFrame, truth: pd.DataFrame) -> dict[str, Any]:
    table = pd.DataFrame(frame)
    keys = truth_keys(table) if not table.empty else pd.DataFrame(columns=["ticker", "date"])
    return distribution_for_keys(truth, keys, allow_empty=True)


def _normalize_deep_path(frame: pd.DataFrame, *, thresholds: Sequence[float]) -> pd.DataFrame:
    raw = pd.DataFrame(frame)
    deep_required = {
        "actual_stop_out",
        "actual_raised_stop_out",
        "later_peak_after_stop",
        "canonical_risk_breach_before_peak",
        "full_horizon_first_risk_breach_bar",
    }
    missing = sorted(deep_required - set(raw.columns))
    if missing:
        raise AuditBlockedError(f"upside realization sidecar缺少deep path欄位: {missing}")
    table = normalize_path(raw, thresholds=thresholds)
    for source, output in (
        ("actual_stop_out", "actual_stop_out_bool"),
        ("actual_raised_stop_out", "actual_raised_stop_out_bool"),
        ("later_peak_after_stop", "later_peak_after_stop_bool"),
        ("canonical_risk_breach_before_peak", "canonical_risk_breach_before_peak_bool"),
    ):
        values = table.get(source, pd.Series(False, index=table.index))
        if values.dtype == bool:
            table[output] = values.fillna(False).astype(bool)
        else:
            table[output] = values.fillna("").astype(str).str.strip().str.lower().isin(
                {"1", "true", "t", "yes", "y"}
            )
    return table


def _filled_path_join(planned: pd.DataFrame, path: pd.DataFrame) -> pd.DataFrame:
    planned_table = pd.DataFrame(planned).copy()
    filled = planned_table.loc[
        planned_table["entry_filled_bool"].fillna(False).astype(bool)
    ].copy()
    if filled.empty:
        return pd.DataFrame(columns=list(path.columns))
    joined = filled[["event_key"]].merge(
        pd.DataFrame(path), on="event_key", how="left", validate="one_to_one"
    )
    if "ticker" not in joined.columns or int(joined["ticker"].notna().sum()) != len(filled):
        raise AuditBlockedError("filled cohort無法完整對回upside realization sidecar")
    covered = joined.loc[
        joined["path_target_available_bool"]
        & pd.to_numeric(joined["full_horizon_mfe_r"], errors="coerce").map(math.isfinite)
        & pd.to_numeric(joined["full_horizon_adverse_to_peak_r"], errors="coerce").map(math.isfinite)
    ].copy()
    return covered


def _deep_path_summary(
    planned: pd.DataFrame,
    path: pd.DataFrame,
    *,
    thresholds: Sequence[float],
) -> dict[str, Any]:
    out = summarize_path_cohort(planned, path, thresholds=thresholds)
    covered = _filled_path_join(planned, path)
    if covered.empty:
        out.update(
            {
                "actual_stop_out_pct": None,
                "raised_stop_out_pct": None,
                "stop_before_later_peak_pct": None,
                "canonical_risk_before_peak_pct": None,
                "winner_capture_median_pct": None,
                "winner_giveback_mean_r": None,
            }
        )
        for threshold in thresholds:
            token = f"{float(threshold):g}r"
            out[f"all_stop_before_{token}_pct"] = None
            out[f"raised_stop_before_{token}_pct"] = None
            out[f"canonical_risk_before_{token}_pct"] = None
            out[f"reached_{token}_realized_mean_r"] = None
        return out

    out["actual_stop_out_pct"] = float(covered["actual_stop_out_bool"].mean() * 100.0)
    out["raised_stop_out_pct"] = float(covered["actual_raised_stop_out_bool"].mean() * 100.0)
    out["stop_before_later_peak_pct"] = float(covered["later_peak_after_stop_bool"].mean() * 100.0)
    out["canonical_risk_before_peak_pct"] = float(
        covered["canonical_risk_breach_before_peak_bool"].mean() * 100.0
    )

    mfe = pd.to_numeric(covered["full_horizon_mfe_r"], errors="coerce")
    realized = pd.to_numeric(covered["realized_r"], errors="coerce")
    winner = mfe >= 1.0
    capture = (realized.loc[winner] / mfe.loc[winner] * 100.0).replace([np.inf, -np.inf], np.nan)
    giveback = mfe.loc[winner] - realized.loc[winner]
    out["winner_capture_median_pct"] = (
        None if capture.dropna().empty else float(capture.median())
    )
    out["winner_giveback_mean_r"] = (
        None if giveback.dropna().empty else float(giveback.mean())
    )

    exit_date = pd.to_datetime(covered.get("exit_date"), errors="coerce")
    actual_stop = covered["actual_stop_out_bool"].fillna(False).astype(bool)
    raised_stop = covered["actual_raised_stop_out_bool"].fillna(False).astype(bool)
    breach_bar = pd.to_numeric(covered.get("full_horizon_first_risk_breach_bar"), errors="coerce")
    for threshold in thresholds:
        token = f"{float(threshold):g}".replace(".0", "")
        key = f"{float(threshold):g}r"
        first_date = pd.to_datetime(
            covered.get(f"full_horizon_first_upside_{token}r_date"), errors="coerce"
        )
        first_bar = pd.to_numeric(
            covered.get(f"full_horizon_first_upside_{token}r_bar"), errors="coerce"
        )
        reached = first_date.notna() & first_bar.ge(1)
        if bool(reached.any()):
            stop_before = actual_stop & reached & (exit_date <= first_date)
            raised_before = raised_stop & reached & (exit_date <= first_date)
            canonical_before = reached & breach_bar.ge(1) & (breach_bar <= first_bar)
            out[f"all_stop_before_{key}_pct"] = float(stop_before.loc[reached].mean() * 100.0)
            out[f"raised_stop_before_{key}_pct"] = float(raised_before.loc[reached].mean() * 100.0)
            out[f"canonical_risk_before_{key}_pct"] = float(canonical_before.loc[reached].mean() * 100.0)
            reached_realized = realized.loc[reached]
            out[f"reached_{key}_realized_mean_r"] = (
                None if reached_realized.dropna().empty else float(reached_realized.mean())
            )
        else:
            out[f"all_stop_before_{key}_pct"] = None
            out[f"raised_stop_before_{key}_pct"] = None
            out[f"canonical_risk_before_{key}_pct"] = None
            out[f"reached_{key}_realized_mean_r"] = None
    return out


def _path_matrix(
    planned: pd.DataFrame,
    path: pd.DataFrame,
    *,
    thresholds: Sequence[float],
    adverse_edges: Sequence[float],
) -> list[dict[str, Any]]:
    covered = _filled_path_join(planned, path)
    if covered.empty:
        return []
    thresholds = tuple(float(value) for value in thresholds)
    adverse_edges = tuple(float(value) for value in adverse_edges)
    mfe_edges = [-math.inf, *thresholds, math.inf]
    mfe_labels: list[str] = []
    for left, right in zip(mfe_edges[:-1], mfe_edges[1:]):
        if math.isinf(left):
            mfe_labels.append(f"<{right:g}R")
        elif math.isinf(right):
            mfe_labels.append(f">={left:g}R")
        else:
            mfe_labels.append(f"{left:g}-{right:g}R")
    adverse_bin_edges = [-math.inf, *adverse_edges, math.inf]
    adverse_labels: list[str] = []
    for left, right in zip(adverse_bin_edges[:-1], adverse_bin_edges[1:]):
        if math.isinf(left):
            adverse_labels.append(f"<{right:g}R")
        elif math.isinf(right):
            adverse_labels.append(f">={left:g}R")
        else:
            adverse_labels.append(f"{left:g}-{right:g}R")
    table = covered.copy()
    table["mfe_bucket"] = pd.cut(
        pd.to_numeric(table["full_horizon_mfe_r"], errors="coerce"),
        bins=mfe_edges,
        labels=mfe_labels,
        right=False,
    )
    table["adverse_bucket"] = pd.cut(
        pd.to_numeric(table["full_horizon_adverse_to_peak_r"], errors="coerce"),
        bins=adverse_bin_edges,
        labels=adverse_labels,
        right=False,
    )
    rows: list[dict[str, Any]] = []
    for mfe_bucket in mfe_labels:
        for adverse_bucket in adverse_labels:
            group = table.loc[
                table["mfe_bucket"].astype(str).eq(mfe_bucket)
                & table["adverse_bucket"].astype(str).eq(adverse_bucket)
            ]
            if group.empty:
                continue
            mfe = pd.to_numeric(group["full_horizon_mfe_r"], errors="coerce")
            realized = pd.to_numeric(group["realized_r"], errors="coerce")
            rows.append(
                {
                    "mfe_bucket": mfe_bucket,
                    "adverse_bucket": adverse_bucket,
                    "trade_count": int(len(group)),
                    "realized_mean_r": float(realized.mean()),
                    "full_horizon_mfe_mean_r": float(mfe.mean()),
                    "full_horizon_adverse_to_peak_mean_r": float(
                        pd.to_numeric(group["full_horizon_adverse_to_peak_r"], errors="coerce").mean()
                    ),
                    "stop_out_pct": float(group["actual_stop_out_bool"].mean() * 100.0),
                    "stop_before_later_peak_pct": float(
                        group["later_peak_after_stop_bool"].mean() * 100.0
                    ),
                    "giveback_mean_r": float((mfe - realized).mean()),
                }
            )
    return rows


def _cohort_summary(
    membership: pd.DataFrame,
    *,
    truth: pd.DataFrame,
    path: pd.DataFrame,
    thresholds: Sequence[float],
) -> dict[str, Any]:
    truth_summary = _truth_summary(membership, truth)
    path_summary = _deep_path_summary(membership, path, thresholds=thresholds)
    score = pd.to_numeric(
        pd.DataFrame(membership).get(
            "score_percentile", pd.Series(index=pd.DataFrame(membership).index, dtype=float)
        ),
        errors="coerce",
    )
    return {
        **truth_summary,
        **path_summary,
        "score_percentile_mean": None if score.dropna().empty else float(score.mean()),
    }


def _raw_planned_membership_cohorts(raw: pd.DataFrame, planned: pd.DataFrame) -> dict[str, pd.DataFrame]:
    raw_table = pd.DataFrame(raw).copy()
    planned_table = pd.DataFrame(planned).copy()
    for table in (raw_table, planned_table):
        table["occurrence_key"] = (
            table["ticker"].astype(str)
            + "|" + table["trade_date"].astype(str)
            + "|" + table["signal_date"].astype(str)
        )
    raw_keys = set(raw_table["occurrence_key"])
    planned_keys = set(planned_table["occurrence_key"])
    return {
        "retained": raw_table.loc[raw_table["occurrence_key"].isin(raw_keys & planned_keys)].copy(),
        "raw_dropped": raw_table.loc[raw_table["occurrence_key"].isin(raw_keys - planned_keys)].copy(),
        "constrained_added": planned_table.loc[planned_table["occurrence_key"].isin(planned_keys - raw_keys)].copy(),
    }


def _keys_from_stage(frame: pd.DataFrame) -> pd.DataFrame:
    table = pd.DataFrame(frame)
    if table.empty:
        return pd.DataFrame(columns=["ticker", "date"])
    return pd.DataFrame(
        {
            "ticker": table["ticker"].map(normalize_ticker),
            "date": table["score_event_date"].map(normalize_date),
        }
    ).drop_duplicates(["ticker", "date"])


def _stage_truth_score_summary(frame: pd.DataFrame, truth: pd.DataFrame) -> dict[str, Any]:
    table = pd.DataFrame(frame)
    out = distribution_for_keys(truth, _keys_from_stage(table), allow_empty=True)
    score_col = None
    for candidate in ("score_percentile", "breakout_quality_daily_score_percentile"):
        if candidate in table.columns:
            score_col = candidate
            break
    score = (
        pd.Series(index=table.index, dtype=float)
        if score_col is None
        else pd.to_numeric(table[score_col], errors="coerce")
    )
    out["score_percentile_mean"] = None if score.dropna().empty else float(score.mean())
    return out


def _constrained_internal_summary(
    *,
    truth: pd.DataFrame,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _raw_top_k_with_score_event(pd.DataFrame(evidence["selector_trace"]))
    capacity = pd.DataFrame(evidence["daily_capacity"])
    _validate_raw_stage_capacity(raw, capacity)
    resource = summarize_resource_contract(capacity)
    planned = build_planned_membership(
        orderable=pd.DataFrame(evidence["orderable"]),
        execution=pd.DataFrame(evidence["execution"]),
    )
    infeasible_dates = set(resource["direct_infeasible_trade_dates"])
    raw_inf = raw.loc[raw["trade_date"].astype(str).isin(infeasible_dates)].copy()
    planned_inf = planned.loc[planned["trade_date"].astype(str).isin(infeasible_dates)].copy()
    cohorts = _raw_planned_membership_cohorts(raw_inf, planned_inf)
    cohort_summaries = {
        key: _stage_truth_score_summary(frame, truth)
        for key, frame in cohorts.items()
    }
    return {
        "resource_contract": resource,
        "direct_infeasible_cohorts": cohort_summaries,
        "direct_infeasible_counts": {key: int(len(frame)) for key, frame in cohorts.items()},
        "raw_top_k": _stage_truth_score_summary(raw, truth),
        "planned": _stage_truth_score_summary(planned, truth),
    }


def _validate_arm_contracts(source, *, constrained_arm_id: str, direct_arm_id: str) -> None:
    constrained = source.settings.arms.get(constrained_arm_id)
    direct = source.settings.arms.get(direct_arm_id)
    if constrained is None or direct is None:
        raise AuditBlockedError(f"{source.profile_id} 缺少C80/C81 arm設定")
    if constrained.dl_id != direct.dl_id:
        raise AuditBlockedError(f"{source.profile_id} C80/C81不是同一DL source")
    shared_fields = (
        "param_source",
        "param_policy",
        "rule_policy",
        "dl_enabled",
    )
    for field in shared_fields:
        if getattr(constrained, field) != getattr(direct, field):
            raise AuditBlockedError(
                f"{source.profile_id} C80/C81 shared contract不一致: {field}"
            )
    if constrained.dl_runtime_mode != "resource-aware-continuous-score-constrained-optimal":
        raise AuditBlockedError(f"{source.profile_id}/{constrained_arm_id}不是exact K/R0 constrained")
    if dict(constrained.dl_runtime_options or {}).get("preserve_k_r0") is not True:
        raise AuditBlockedError(f"{source.profile_id}/{constrained_arm_id}未preserve_k_r0")
    if direct.dl_runtime_mode != "resource-aware-continuous-score-no-k-no-r0":
        raise AuditBlockedError(f"{source.profile_id}/{direct_arm_id}不是No-K/No-R0 direct")
    direct_options = dict(direct.dl_runtime_options or {})
    if direct_options.get("preserve_k") is not False or direct_options.get("preserve_r0") is not False:
        raise AuditBlockedError(f"{source.profile_id}/{direct_arm_id} No-K/No-R0 contract不一致")


def _core_summary(source, arm_id: str) -> dict[str, Any]:
    scenarios = dict(source.result.get("scenarios") or {})
    row = dict(scenarios.get(arm_id) or {})
    wanted = (
        "total_return_pct",
        "max_drawdown_pct",
        "return_over_max_drawdown",
        "annual_return_pct",
        "expected_value_r",
        "trade_count",
        "avg_exposure_pct",
        "monthly_win_rate_pct",
        "win_rate_pct",
        "payoff_ratio",
    )
    return {key: row.get(key) for key in wanted}


def _drawdown_summary(drawdown: Mapping[str, Any]) -> dict[str, Any]:
    top = list(dict(drawdown).get("top_episodes") or [])
    episode = dict(top[0]) if top else {}
    return {
        "max_drawdown_pct": drawdown.get("max_drawdown_pct"),
        "peak_date": episode.get("peak_date"),
        "trough_date": episode.get("trough_date"),
        "calendar_days": episode.get("drawdown_to_trough_calendar_days"),
        "relevant_trade_count": episode.get("relevant_trade_count"),
        "max_same_day_entries": episode.get("max_same_day_entries"),
        "negative_contributor_count": episode.get("negative_contributor_count"),
        "hm_hs_mtm_pct_peak_equity": episode.get("hmhs_mtm_contribution_pct_peak_equity"),
        "hm_ls_mtm_pct_peak_equity": episode.get("hmls_mtm_contribution_pct_peak_equity"),
        "lm_hs_mtm_pct_peak_equity": episode.get("lmhs_mtm_contribution_pct_peak_equity"),
        "lm_ls_mtm_pct_peak_equity": episode.get("lmls_mtm_contribution_pct_peak_equity"),
        "reconciliation_delta": episode.get("reconciliation_delta"),
    }


def _fingerprint_payload(definition: AuditDefinition, source_refs: Mapping[str, Any]) -> str:
    raw = json.dumps(
        {"schema": 1, "definition": definition.as_dict(), "sources": source_refs},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _fmt(value: Any, digits: int = 2, unit: str = "") -> str:
    number = finite_float(value)
    if number is None:
        return "-"
    return f"{number:.{digits}f}{unit}"


def _render_arm_table(payload: Mapping[str, Any], constrained: str, direct: str) -> str:
    rows = []
    for arm_id in (constrained, direct):
        core = dict(payload["arms"][arm_id]["core"])
        path = dict(payload["arms"][arm_id]["path"])
        truth = dict(payload["arms"][arm_id]["truth"])
        capital = dict(payload["arms"][arm_id]["capital"])
        rows.append(
            [
                arm_id,
                _fmt(core.get("total_return_pct"), unit="%"),
                _fmt(core.get("max_drawdown_pct"), unit="%"),
                _fmt(core.get("return_over_max_drawdown")),
                _fmt(core.get("expected_value_r"), unit="R"),
                _fmt(capital.get("average_exposure_pct"), unit="%"),
                _fmt(capital.get("median_reserved_fraction_pct"), unit="%"),
                _fmt(capital.get("median_stop_distance_pct"), unit="%"),
                _fmt(capital.get("median_holding_calendar_days"), digits=1, unit="d"),
                _fmt(truth.get("high_mfe_total_pct"), unit="%"),
                _fmt(truth.get("high_mfe_low_safety_pct"), unit="%"),
                _fmt(truth.get("high_safety_total_pct"), unit="%"),
                _fmt(path.get("full_horizon_mfe_mean_r"), unit="R"),
                _fmt(path.get("adverse_to_peak_mean_r"), unit="R"),
                _fmt(path.get("realized_mean_r"), unit="R"),
                _fmt(path.get("winner_capture_median_pct"), unit="%"),
                _fmt(path.get("winner_giveback_mean_r"), unit="R"),
            ]
        )
    return render_table(
        [
            "Arm", "Return", "MDD", "RoMD", "Core EV", "Exposure", "Reserve P50",
            "StopDist P50", "Hold P50", "High-MFE", "HM/LS", "High-Safety", "Full-MFE", "Adverse", "Path EV",
            "Winner capture(P50)", "Winner giveback",
        ],
        rows,
    )


def _render_first_passage(payload: Mapping[str, Any], constrained: str, direct: str, thresholds: Sequence[float]) -> str:
    rows = []
    for arm_id in (constrained, direct):
        path = dict(payload["arms"][arm_id]["path"])
        for threshold in thresholds:
            key = f"{float(threshold):g}r"
            rows.append(
                [
                    arm_id,
                    f"+{float(threshold):g}R",
                    _fmt(path.get(f"first_{float(threshold):g}r_reached_pct"), unit="%"),
                    _fmt(path.get(f"canonical_risk_before_{key}_pct"), unit="%"),
                    _fmt(path.get(f"all_stop_before_{key}_pct"), unit="%"),
                    _fmt(path.get(f"initial_stop_before_{float(threshold):g}r_pct"), unit="%"),
                    _fmt(path.get(f"raised_stop_before_{key}_pct"), unit="%"),
                    _fmt(path.get(f"reached_{key}_realized_mean_r"), unit="R"),
                ]
            )
    return render_table(
        [
            "Arm", "Threshold", "Filled reached", "Canonical risk before",
            "All stop before", "Initial stop before", "Raised stop before", "Reached cohort EV",
        ],
        rows,
    )


def _render_membership(payload: Mapping[str, Any]) -> str:
    rows = []
    for key in _COHORT_ORDER:
        item = dict(payload["planned_membership_cohorts"][key])
        rows.append(
            [
                _COHORT_LABEL[key],
                str(int(item.get("planned_count", 0) or 0)),
                str(int(item.get("filled_count", 0) or 0)),
                _fmt(item.get("score_percentile_mean"), digits=3),
                _fmt(item.get("high_mfe_total_pct"), unit="%"),
                _fmt(item.get("high_mfe_high_safety_pct"), unit="%"),
                _fmt(item.get("high_mfe_low_safety_pct"), unit="%"),
                _fmt(item.get("high_safety_total_pct"), unit="%"),
                _fmt(item.get("full_horizon_mfe_mean_r"), unit="R"),
                _fmt(item.get("adverse_to_peak_mean_r"), unit="R"),
                _fmt(item.get("realized_mean_r"), unit="R"),
                _fmt(item.get("winner_capture_median_pct"), unit="%"),
                _fmt(item.get("winner_giveback_mean_r"), unit="R"),
            ]
        )
    return render_table(
        [
            "Cohort", "Planned", "Filled", "Score %ile", "High-MFE", "HM/HS", "HM/LS",
            "High-Safety", "Full-MFE", "Adverse", "Realized EV", "Capture(P50)", "Giveback",
        ],
        rows,
    )


def _render_resource(payload: Mapping[str, Any]) -> str:
    item = dict(payload["c80_internal"]["resource_contract"])
    rows = [
        [
            str(item.get("eligible_days")),
            f"{item.get('direct_infeasible_days')} ({_fmt(item.get('direct_infeasible_pct'), unit='%')})",
            f"{item.get('k_headroom_days')} ({_fmt(item.get('k_headroom_pct'), unit='%')})",
            _fmt(item.get("median_k"), 1),
            _fmt(item.get("median_free_slots"), 1),
            (
                f"{item.get('r0_exact_binding_days')} ({_fmt(item.get('r0_exact_binding_pct'), unit='%')})"
                if item.get("r0_exact_binding_pct") is not None
                else "-"
            ),
            _fmt(item.get("median_final_r0_slack_pct"), unit="%"),
        ]
    ]
    return render_table(
        ["Eligible days", "Raw direct-infeasible", "K headroom", "Median K", "Median free slots", "Final=R0", "Median R0 slack"],
        rows,
    )


def _render_swap(payload: Mapping[str, Any]) -> str:
    internal = dict(payload["c80_internal"])
    stage_rows = []
    for key, label in (("raw_top_k", "C80 Raw Top-K"), ("planned", "C80 exact K/R0 planned")):
        item = dict(internal.get(key) or {})
        stage_rows.append(
            [
                label,
                str(int(item.get("truth_covered_rows", 0) or 0)),
                _fmt(item.get("score_percentile_mean"), digits=3),
                _fmt(item.get("high_mfe_total_pct"), unit="%"),
                _fmt(item.get("high_mfe_high_safety_pct"), unit="%"),
                _fmt(item.get("high_mfe_low_safety_pct"), unit="%"),
                _fmt(item.get("high_safety_total_pct"), unit="%"),
            ]
        )
    stage_table = render_table(
        ["C80 stage", "N", "Score %ile", "High-MFE", "HM/HS", "HM/LS", "High-Safety"],
        stage_rows,
    )

    rows = []
    counts = dict(internal["direct_infeasible_counts"])
    summaries = dict(internal["direct_infeasible_cohorts"])
    for key, label in (
        ("retained", "Raw retained"),
        ("raw_dropped", "Raw Top-K dropped"),
        ("constrained_added", "C80 constrained added"),
    ):
        item = dict(summaries.get(key) or {})
        rows.append(
            [
                label,
                str(int(counts.get(key, 0) or 0)),
                _fmt(item.get("score_percentile_mean"), digits=3),
                _fmt(item.get("high_mfe_total_pct"), unit="%"),
                _fmt(item.get("high_mfe_high_safety_pct"), unit="%"),
                _fmt(item.get("high_mfe_low_safety_pct"), unit="%"),
                _fmt(item.get("high_safety_total_pct"), unit="%"),
            ]
        )
    swap_table = render_table(
        ["Direct-infeasible cohort", "N", "Score %ile", "High-MFE", "HM/HS", "HM/LS", "High-Safety"],
        rows,
    )
    return stage_table + "\n" + swap_table


def _render_matrix(payload: Mapping[str, Any], constrained: str, direct: str) -> str:
    rows = []
    for arm_id in (constrained, direct):
        for item in payload["arms"][arm_id]["mfe_adverse_matrix"]:
            rows.append(
                [
                    arm_id,
                    item["mfe_bucket"],
                    item["adverse_bucket"],
                    str(item["trade_count"]),
                    _fmt(item.get("realized_mean_r"), unit="R"),
                    _fmt(item.get("giveback_mean_r"), unit="R"),
                    _fmt(item.get("stop_out_pct"), unit="%"),
                    _fmt(item.get("stop_before_later_peak_pct"), unit="%"),
                ]
            )
    return render_table(
        ["Arm", "MFE bucket", "Adverse bucket", "N", "Realized EV", "Giveback", "Stop", "Stop→later peak"],
        rows,
    )


def _render_drawdown(payload: Mapping[str, Any], constrained: str, direct: str) -> str:
    rows = []
    for arm_id in (constrained, direct):
        item = dict(payload["arms"][arm_id]["drawdown_summary"])
        rows.append(
            [
                arm_id,
                _fmt(item.get("max_drawdown_pct"), unit="%"),
                str(item.get("peak_date") or "-"),
                str(item.get("trough_date") or "-"),
                str(item.get("calendar_days") or "-"),
                str(item.get("relevant_trade_count") or "-"),
                str(item.get("max_same_day_entries") or "-"),
                _fmt(item.get("hm_hs_mtm_pct_peak_equity"), unit="%"),
                _fmt(item.get("hm_ls_mtm_pct_peak_equity"), unit="%"),
                _fmt(item.get("lm_hs_mtm_pct_peak_equity"), unit="%"),
                _fmt(item.get("lm_ls_mtm_pct_peak_equity"), unit="%"),
                _fmt(item.get("reconciliation_delta"), digits=3),
            ]
        )
    return render_table(
        [
            "Arm", "MDD", "Peak", "Trough", "Days", "Relevant trades", "Max same-day entries",
            "HM/HS MTM", "HM/LS MTM", "LM/HS MTM", "LM/LS MTM", "Reconcile Δ",
        ],
        rows,
    )


def render_result(result: Mapping[str, Any]) -> str:
    lines = [render_title("C80 / C81 Allocator + Path Attribution Audit")]
    lines.append(
        render_key_values(
            [
                ("Audit", result.get("audit_id")),
                ("Constrained", result.get("constrained_arm_id")),
                ("Direct", result.get("direct_arm_id")),
                ("決策問題", result.get("decision_question")),
                ("契約", "同一MR-13AH score；只改allocator/resource contract；Audit read-only"),
            ]
        )
    )
    section = 1
    for profile_id in result.get("evaluation_order", []):
        payload = result["evaluations"][profile_id]
        display = payload["display_name"]
        lines.append(render_section(f"{display}｜C80 vs C81 core / MFE→Realized conversion", number=section)); section += 1
        lines.append(_render_arm_table(payload, result["constrained_arm_id"], result["direct_arm_id"]))
        lines.append(render_section(f"{display}｜First-passage完整分母與Stop decomposition", number=section)); section += 1
        lines.append(_render_first_passage(payload, result["constrained_arm_id"], result["direct_arm_id"], result["first_passage_thresholds_r"]))
        lines.append(render_section(f"{display}｜C80/C81 Planned membership差異", number=section)); section += 1
        lines.append(_render_membership(payload))
        lines.append(render_section(f"{display}｜C80 K/R0 resource binding", number=section)); section += 1
        lines.append(_render_resource(payload))
        lines.append(render_section(f"{display}｜Raw Top-K在direct-infeasible日被K/R0 exact solver如何換人", number=section)); section += 1
        lines.append(_render_swap(payload))
        lines.append(render_section(f"{display}｜MFE × Adverse → Realized EV / Giveback", number=section)); section += 1
        lines.append(_render_matrix(payload, result["constrained_arm_id"], result["direct_arm_id"]))
        lines.append(render_section(f"{display}｜Exact peak→trough MTM drawdown attribution", number=section)); section += 1
        lines.append(_render_drawdown(payload, result["constrained_arm_id"], result["direct_arm_id"]))
    lines.append(render_section("判讀邊界", number=section))
    lines.append(
        "C80−C81可歸因於同一AH score下的allocator/resource contract差異；但C80的K與R0是joint contract，"
        "再加canonical cash feasibility與exact basket search。沒有額外K-only/R0-only counterfactual時，"
        "本Audit不得把改善拆成『K單獨』或『R0單獨』因果。Raw Top-K dropped/added只在C80的"
        "direct-infeasible日期解釋exact solver的membership substitution；portfolio-path divergence仍可能使"
        "後續日期的持股、現金與候選集合不同。Future MFE/Safety/path truth只用於事後診斷，從未參與runtime selection。"
    )
    return "\n".join(lines)


def _render_markdown(result: Mapping[str, Any]) -> str:
    return "# C80 / C81 Allocator + Path Attribution Audit\n\n```text\n" + render_result(result) + "\n```\n"


def _write_csvs(run_dir: Path, result: Mapping[str, Any]) -> dict[str, Path]:
    cohort_rows: list[dict[str, Any]] = []
    first_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    drawdown_rows: list[dict[str, Any]] = []
    for profile_id in result.get("evaluation_order", []):
        payload = result["evaluations"][profile_id]
        for key in _COHORT_ORDER:
            cohort_rows.append({"profile_id": profile_id, "cohort": key, **dict(payload["planned_membership_cohorts"][key])})
        for arm_id, arm in payload["arms"].items():
            path = dict(arm["path"])
            for threshold in result["first_passage_thresholds_r"]:
                key = f"{float(threshold):g}r"
                first_rows.append(
                    {
                        "profile_id": profile_id,
                        "arm_id": arm_id,
                        "threshold_r": threshold,
                        "filled_reached_pct": path.get(f"first_{float(threshold):g}r_reached_pct"),
                        "canonical_risk_before_pct": path.get(f"canonical_risk_before_{key}_pct"),
                        "all_stop_before_pct": path.get(f"all_stop_before_{key}_pct"),
                        "initial_stop_before_pct": path.get(f"initial_stop_before_{float(threshold):g}r_pct"),
                        "raised_stop_before_pct": path.get(f"raised_stop_before_{key}_pct"),
                        "reached_realized_mean_r": path.get(f"reached_{key}_realized_mean_r"),
                    }
                )
            for row in arm["mfe_adverse_matrix"]:
                matrix_rows.append({"profile_id": profile_id, "arm_id": arm_id, **dict(row)})
            drawdown_rows.append({"profile_id": profile_id, "arm_id": arm_id, **dict(arm["drawdown_summary"])})
        resource = dict(payload["c80_internal"]["resource_contract"])
        resource_scalar = {
            key: value for key, value in resource.items()
            if not isinstance(value, (list, tuple, dict))
        }
        resource_rows.append({"profile_id": profile_id, "section": "resource_contract", **resource_scalar})
        for stage in ("raw_top_k", "planned"):
            resource_rows.append(
                {
                    "profile_id": profile_id,
                    "section": stage,
                    **dict(payload["c80_internal"].get(stage) or {}),
                }
            )
        for key, summary in payload["c80_internal"]["direct_infeasible_cohorts"].items():
            resource_rows.append(
                {
                    "profile_id": profile_id,
                    "section": key,
                    "count": payload["c80_internal"]["direct_infeasible_counts"].get(key),
                    **dict(summary),
                }
            )
    paths = {
        "cohorts": run_dir / "planned_membership_cohorts.csv",
        "first_passage": run_dir / "first_passage_decomposition.csv",
        "matrix": run_dir / "mfe_adverse_realization_matrix.csv",
        "resource": run_dir / "c80_resource_substitution.csv",
        "drawdown": run_dir / "drawdown_mtm_attribution.csv",
    }
    pd.DataFrame(cohort_rows).to_csv(paths["cohorts"], index=False, encoding="utf-8-sig")
    pd.DataFrame(first_rows).to_csv(paths["first_passage"], index=False, encoding="utf-8-sig")
    pd.DataFrame(matrix_rows).to_csv(paths["matrix"], index=False, encoding="utf-8-sig")
    pd.DataFrame(resource_rows).to_csv(paths["resource"], index=False, encoding="utf-8-sig")
    pd.DataFrame(drawdown_rows).to_csv(paths["drawdown"], index=False, encoding="utf-8-sig")
    return paths


def run_audit(definition: AuditDefinition, *, project_root: Path) -> dict[str, Any]:
    if definition.audit_type != SUPPORTED_AUDIT_TYPE:
        raise ValueError(f"不支援audit_type={definition.audit_type}")
    root = Path(project_root).resolve()
    source_cfg = dict(definition.source)
    dims = dict(definition.dimensions)
    constrained_arm_id = str(source_cfg.get("constrained_arm_id") or "").strip()
    direct_arm_id = str(source_cfg.get("direct_arm_id") or "").strip()
    evaluation_order = tuple(str(value) for value in source_cfg.get("evaluation_profile_ids", ()))
    if not constrained_arm_id or not direct_arm_id or not evaluation_order:
        raise AuditBlockedError("C80/C81 allocator Audit source設定不完整")
    thresholds = tuple(float(value) for value in dims.get("first_passage_thresholds_r", (1, 2, 3)))
    adverse_edges = tuple(float(value) for value in dims.get("adverse_bucket_edges_r", (0.5, 1.0)))
    cutoff = float(dims.get("truth_high_cutoff", 0.50))
    percentile_method = str(dims.get("percentile_method") or "average_zero_based")

    truth, truth_source = build_truth_geometry(definition, root, percentile_method=percentile_method)
    truth = attach_quadrants(truth, cutoff=cutoff)
    source_refs: dict[str, Any] = {"truth": truth_source, "strategy": {}}
    evaluations: dict[str, Any] = {}

    close_cache: dict[str, pd.Series] = {}
    market_maps: dict[str, tuple[Path, dict[str, str]]] = {}
    for profile_id in evaluation_order:
        pinned = str(dict(source_cfg.get("strategy_result_fingerprints") or {}).get(profile_id) or "").strip()
        source = load_strategy_compare_source(
            root, profile_id=profile_id, pinned_config_fingerprint=pinned or None
        )
        _validate_arm_contracts(
            source, constrained_arm_id=constrained_arm_id, direct_arm_id=direct_arm_id
        )
        start, end = _period(source)
        period_truth = filter_period(truth, start, end)
        if period_truth.empty:
            raise AuditBlockedError(f"{profile_id} comparison period與truth沒有交集")

        evidences: dict[str, Mapping[str, Any]] = {}
        planned: dict[str, pd.DataFrame] = {}
        paths: dict[str, pd.DataFrame] = {}
        arms: dict[str, Any] = {}
        for arm_id in (constrained_arm_id, direct_arm_id):
            evidence = load_strategy_arm_path_sidecars(root, source=source, arm_id=arm_id)
            evidences[arm_id] = evidence
            planned_arm = build_planned_membership(
                orderable=pd.DataFrame(evidence["orderable"]),
                execution=pd.DataFrame(evidence["execution"]),
            )
            path_arm = _normalize_deep_path(
                pd.DataFrame(evidence["upside_realization"]), thresholds=thresholds
            )
            planned[arm_id] = planned_arm
            paths[arm_id] = path_arm
            arms[arm_id] = {
                "core": _core_summary(source, arm_id),
                "truth": _truth_summary(
                    planned_arm.loc[planned_arm["entry_filled_bool"].astype(bool)], period_truth
                ),
                "path": _deep_path_summary(planned_arm, path_arm, thresholds=thresholds),
                "mfe_adverse_matrix": _path_matrix(
                    planned_arm,
                    path_arm,
                    thresholds=thresholds,
                    adverse_edges=adverse_edges,
                ),
                "capital": _capital_summary(evidence),
            }

        c80_keys = _event_key_set(planned[constrained_arm_id])
        c81_keys = _event_key_set(planned[direct_arm_id])
        common_keys = c80_keys & c81_keys
        cohort_frames = {
            "common_c80": _membership_subset(planned[constrained_arm_id], common_keys),
            "common_c81": _membership_subset(planned[direct_arm_id], common_keys),
            "c80_only": _membership_subset(planned[constrained_arm_id], c80_keys - c81_keys),
            "c81_only": _membership_subset(planned[direct_arm_id], c81_keys - c80_keys),
        }
        planned_membership_cohorts = {
            "common_c80": _cohort_summary(
                cohort_frames["common_c80"],
                truth=period_truth,
                path=paths[constrained_arm_id],
                thresholds=thresholds,
            ),
            "common_c81": _cohort_summary(
                cohort_frames["common_c81"],
                truth=period_truth,
                path=paths[direct_arm_id],
                thresholds=thresholds,
            ),
            "c80_only": _cohort_summary(
                cohort_frames["c80_only"],
                truth=period_truth,
                path=paths[constrained_arm_id],
                thresholds=thresholds,
            ),
            "c81_only": _cohort_summary(
                cohort_frames["c81_only"],
                truth=period_truth,
                path=paths[direct_arm_id],
                thresholds=thresholds,
            ),
        }

        c80_pipeline = load_strategy_arm_pipeline_sidecars(
            root, source=source, arm_id=constrained_arm_id
        )
        c80_internal = _constrained_internal_summary(
            truth=period_truth, evidence=c80_pipeline
        )

        dataset_key = str(source.settings.dataset)
        if dataset_key not in market_maps:
            market_dir = Path(get_dataset_dir(str(root), dataset_key)).resolve()
            if not market_dir.is_dir():
                raise AuditBlockedError(
                    "canonical market data目錄不存在: " + _relative(market_dir, root)
                )
            discovered, _issues = discover_unique_csv_map(str(market_dir))
            market_maps[dataset_key] = (
                market_dir,
                {normalize_ticker(key): str(value) for key, value in discovered.items()},
            )
        market_dir, csv_map = market_maps[dataset_key]
        for arm_id in (constrained_arm_id, direct_arm_id):
            drawdown = build_drawdown_analysis(
                evidences[arm_id],
                period_truth,
                cutoff=cutoff,
                top_n=int(dims.get("drawdown_top_n", 1)),
                market_data_dir=market_dir,
                market_csv_map=csv_map,
                market_close_cache=close_cache,
            )
            arms[arm_id]["drawdown"] = drawdown
            arms[arm_id]["drawdown_summary"] = _drawdown_summary(drawdown)

        evaluations[profile_id] = {
            "display_name": source.settings.profile_label,
            "period": {"start": start, "end": end},
            "strategy_config_fingerprint": source.config_fingerprint,
            "arms": arms,
            "planned_membership_cohorts": planned_membership_cohorts,
            "c80_internal": c80_internal,
        }
        source_refs["strategy"][profile_id] = {
            "run_dir": _relative(source.run_dir, root),
            "config_fingerprint": source.config_fingerprint,
            "arms": {
                arm_id: {
                    "pair_dir": _relative(Path(evidences[arm_id]["pair_dir"]), root),
                    "execution": _relative(Path(evidences[arm_id]["execution_path"]), root),
                    "upside_realization": _relative(Path(evidences[arm_id]["upside_realization_path"]), root),
                    "active_trades": _relative(Path(evidences[arm_id]["active_trades_path"]), root),
                }
                for arm_id in (constrained_arm_id, direct_arm_id)
            },
            "c80_selector_trace": _relative(Path(c80_pipeline["selector_trace_path"]), root),
            "c80_daily_capacity": _relative(Path(c80_pipeline["daily_capacity_path"]), root),
        }

    fingerprint = _fingerprint_payload(definition, source_refs)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_root = root / AUDIT_OUTPUT_ROOT / definition.output_subdir
    run_dir = output_root / "runs" / f"{timestamp}_{fingerprint}"
    run_dir.mkdir(parents=True, exist_ok=False)
    json_path = run_dir / "c80_c81_allocator_path_attribution.json"
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
        "constrained_arm_id": constrained_arm_id,
        "direct_arm_id": direct_arm_id,
        "truth_high_cutoff": cutoff,
        "first_passage_thresholds_r": list(thresholds),
        "adverse_bucket_edges_r": list(adverse_edges),
        "evaluation_order": list(evaluation_order),
        "truth_sources": truth_source,
        "evaluations": evaluations,
        "attribution_boundary": {
            "same_score_source": "C80/C81 must share dl_id, params, rule policy and selection-only scope",
            "joint_resource_contract": "C80 preserves baseline K and R0 jointly with canonical cash feasibility and exact basket search; K-only/R0-only causal split is not identified",
            "portfolio_path": "After a membership difference, later holdings/cash/candidate opportunity may diverge; matched event cohorts are descriptive path attribution, not a full static counterfactual",
            "future_truth": "MFE/Safety/path truth is post-replay diagnostic only and never used by runtime selection",
            "first_passage_denominator": "Filled reached percentages use all path-covered filled buys as denominator; stop-before percentages are conditional on that threshold actually being reached",
            "winner_capture": "winner_capture_median_pct is one-time diagnostic median(realized_r / full_horizon_mfe_r) for path-covered filled buys with MFE>=1R; it is not canonical RCE",
            "drawdown": "exact peak-EOD→trough-EOD position MTM reconciled to canonical equity",
        },
    }
    csv_paths = _write_csvs(run_dir, result)
    artifacts = {
        "完整JSON": _relative(json_path, root),
        "詳細Markdown": _relative(report_path, root),
        "Planned Cohort CSV": _relative(csv_paths["cohorts"], root),
        "First-Passage CSV": _relative(csv_paths["first_passage"], root),
        "MFE-Adverse CSV": _relative(csv_paths["matrix"], root),
        "C80 Resource CSV": _relative(csv_paths["resource"], root),
        "Drawdown MTM CSV": _relative(csv_paths["drawdown"], root),
        "Manifest": _relative(manifest_path, root),
    }
    result["artifacts"] = artifacts
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    report_path.write_text(_render_markdown(result), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "audit_definition": definition.as_dict(),
        "config_fingerprint": fingerprint,
        "source_refs": source_refs,
        "artifacts": artifacts,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "latest.json").write_text(
        json.dumps(
            {
                "run_dir": _relative(run_dir, root),
                "report": _relative(report_path, root),
                "result": _relative(json_path, root),
                "config_fingerprint": fingerprint,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    latest_dir = output_root / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    (latest_dir / "audit.md").write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    (latest_dir / "audit.json").write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    (latest_dir / "manifest.json").write_text(manifest_path.read_text(encoding="utf-8"), encoding="utf-8")
    for key, path in csv_paths.items():
        (latest_dir / path.name).write_bytes(path.read_bytes())
    return result


def preflight(definition: AuditDefinition, *, project_root: Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    blockers: list[str] = []
    strategy_paths: list[str] = []
    target_paths: list[str] = []
    try:
        filter_id, _arch, provider_profile_id, mfe_target_id, safety_target_id = resolve_truth_provider_contract(definition)
        target_paths.append(f"filter:{filter_id} / profile:{provider_profile_id} / target:{mfe_target_id} / safety:{safety_target_id}")
        build_truth_geometry(
            definition,
            root,
            percentile_method=str(definition.dimensions.get("percentile_method") or "average_zero_based"),
        )
    except (ValueError, OSError, AuditBlockedError) as exc:
        blockers.append(str(exc))

    source_cfg = dict(definition.source)
    constrained_arm_id = str(source_cfg.get("constrained_arm_id") or "").strip()
    direct_arm_id = str(source_cfg.get("direct_arm_id") or "").strip()
    thresholds = tuple(float(value) for value in definition.dimensions.get("first_passage_thresholds_r", (1, 2, 3)))
    for profile_id in tuple(str(value) for value in source_cfg.get("evaluation_profile_ids", ())):
        try:
            pinned = str(dict(source_cfg.get("strategy_result_fingerprints") or {}).get(profile_id) or "").strip()
            source = load_strategy_compare_source(root, profile_id=profile_id, pinned_config_fingerprint=pinned or None)
            _validate_arm_contracts(source, constrained_arm_id=constrained_arm_id, direct_arm_id=direct_arm_id)
            _period(source)
            strategy_paths.append(_relative(source.run_dir, root))
            for arm_id in (constrained_arm_id, direct_arm_id):
                evidence = load_strategy_arm_path_sidecars(root, source=source, arm_id=arm_id)
                planned = build_planned_membership(
                    orderable=pd.DataFrame(evidence["orderable"]),
                    execution=pd.DataFrame(evidence["execution"]),
                )
                path = _normalize_deep_path(pd.DataFrame(evidence["upside_realization"]), thresholds=thresholds)
                _deep_path_summary(planned, path, thresholds=thresholds)
                _capital_summary(evidence)
            c80 = load_strategy_arm_pipeline_sidecars(root, source=source, arm_id=constrained_arm_id)
            raw = _raw_top_k_with_score_event(pd.DataFrame(c80["selector_trace"]))
            _validate_raw_stage_capacity(raw, pd.DataFrame(c80["daily_capacity"]))
            summarize_resource_contract(pd.DataFrame(c80["daily_capacity"]))
            market_dir = Path(get_dataset_dir(str(root), str(source.settings.dataset))).resolve()
            if not market_dir.is_dir():
                raise AuditBlockedError("canonical market data目錄不存在: " + _relative(market_dir, root))
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
        "reason": "；".join(str(value) for value in state.get("blockers", ())),
        "source": {
            "display": "Completed C80/C81 OOS/Rolling execution/path + C80 selector trace/resource + canonical truth/market Close",
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
    "collect_status",
    "preflight",
    "render_result",
    "run_audit",
    "run_formal_audit",
]
