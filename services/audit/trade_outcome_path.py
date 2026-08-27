"""Reusable Planned→Filled→Realized trade outcome/path attribution."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from core.console_report import render_table
from core.research_report_contract import (
    TRADE_OUTCOME_FIRST_PASSAGE_THRESHOLDS_R,
    format_contract_value,
    section_contract,
    table_contract,
)
from filters.breakout_quality.mfe_safety_geometry import (
    attach_quadrants,
    build_truth_geometry,
    filter_period,
)
from services.audit.mfe_safety_truth import AuditBlockedError, normalize_date, normalize_ticker
from services.audit.reusable_report import (
    audit_section,
    audit_title,
    evidence_status_for_delta,
    fmt,
    format_contract_row,
    markdown_table,
    persist_reusable_report,
    render_evidence_rows,
    source_summary,
)
from services.audit.selection_membership import (
    build_planned_membership,
    pair_membership_cohorts,
)
from services.audit.strategy_compare_source import (
    AuditSourceBlockedError,
    load_strategy_arm_path_sidecars,
    load_strategy_compare_source,
)

SUPPORTED_AUDIT_TYPE = "trade_outcome_path_attribution"


def _event_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["ticker"].astype(str)
        + "|" + frame["trade_date"].astype(str)
        + "|" + frame["signal_date"].astype(str)
    )


def normalize_path(frame: pd.DataFrame, *, thresholds: Sequence[float]) -> pd.DataFrame:
    table = pd.DataFrame(frame).copy()
    required = {
        "ticker", "trade_date", "signal_date", "path_target_available",
        "full_horizon_mfe_r", "full_horizon_adverse_to_peak_r", "realized_r",
        "actual_initial_stop_out", "exit_date",
    }
    for threshold in thresholds:
        token = f"{float(threshold):g}".replace(".0", "")
        required.add(f"full_horizon_first_upside_{token}r_date")
        required.add(f"full_horizon_first_upside_{token}r_bar")
    missing = sorted(required - set(table.columns))
    if missing:
        raise AuditBlockedError(f"upside realization sidecar缺少欄位: {missing}")
    table["ticker"] = table["ticker"].map(normalize_ticker)
    table["trade_date"] = table["trade_date"].map(normalize_date)
    table["signal_date"] = table["signal_date"].map(normalize_date)
    table["score_event_date"] = table.get(
        "score_event_date", pd.Series("", index=table.index)
    ).map(normalize_date)
    table["score_event_date"] = table["score_event_date"].where(
        table["score_event_date"].ne(""), table["signal_date"]
    )
    table["event_key"] = _event_key(table)
    if bool(table["event_key"].duplicated().any()):
        raise AuditBlockedError("upside realization event_key不唯一")
    table["path_target_available_bool"] = table["path_target_available"].fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})
    table["actual_initial_stop_out_bool"] = table["actual_initial_stop_out"].fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})
    for column in ("full_horizon_mfe_r", "full_horizon_adverse_to_peak_r", "realized_r"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    return table


def summarize_path_cohort(
    planned: pd.DataFrame,
    path: pd.DataFrame,
    *,
    thresholds: Sequence[float],
) -> dict[str, Any]:
    cohort = pd.DataFrame(planned).copy()
    filled = cohort.loc[cohort["entry_filled_bool"].fillna(False).astype(bool)].copy()
    if filled.empty:
        return {
            "planned_count": int(len(cohort)), "filled_count": 0,
            "fill_rate_pct": 0.0 if len(cohort) else None,
            "path_covered_count": 0, "path_coverage_pct": None,
            "realized_mean_r": None, "realized_median_r": None,
            "full_horizon_mfe_mean_r": None, "adverse_to_peak_mean_r": None,
            **{f"first_{float(t):g}r_reached_pct": None for t in thresholds},
            **{f"initial_stop_before_{float(t):g}r_pct": None for t in thresholds},
        }
    joined = filled[["event_key"]].merge(path, on="event_key", how="left", validate="one_to_one")
    if int(joined["ticker"].notna().sum()) != len(filled):
        raise AuditBlockedError("filled cohort無法完整對回upside realization sidecar")
    covered = joined.loc[
        joined["path_target_available_bool"]
        & pd.to_numeric(joined["full_horizon_mfe_r"], errors="coerce").map(math.isfinite)
        & pd.to_numeric(joined["full_horizon_adverse_to_peak_r"], errors="coerce").map(math.isfinite)
    ].copy()
    realized = pd.to_numeric(covered["realized_r"], errors="coerce")
    out = {
        "planned_count": int(len(cohort)),
        "filled_count": int(len(filled)),
        "fill_rate_pct": float(len(filled) / len(cohort) * 100.0),
        "path_covered_count": int(len(covered)),
        "path_coverage_pct": float(len(covered) / len(filled) * 100.0),
        "realized_mean_r": None if realized.dropna().empty else float(realized.mean()),
        "realized_median_r": None if realized.dropna().empty else float(realized.median()),
        "full_horizon_mfe_mean_r": None if covered.empty else float(pd.to_numeric(covered["full_horizon_mfe_r"], errors="coerce").mean()),
        "adverse_to_peak_mean_r": None if covered.empty else float(pd.to_numeric(covered["full_horizon_adverse_to_peak_r"], errors="coerce").mean()),
    }
    exit_date = pd.to_datetime(covered.get("exit_date"), errors="coerce")
    initial_stop = covered["actual_initial_stop_out_bool"].fillna(False).astype(bool)
    for threshold in thresholds:
        token = f"{float(threshold):g}".replace(".0", "")
        first_date = pd.to_datetime(covered.get(f"full_horizon_first_upside_{token}r_date"), errors="coerce")
        first_bar = pd.to_numeric(covered.get(f"full_horizon_first_upside_{token}r_bar"), errors="coerce")
        reached = first_date.notna() & first_bar.ge(1)
        out[f"first_{float(threshold):g}r_reached_pct"] = None if covered.empty else float(reached.mean() * 100.0)
        out[f"initial_stop_before_{float(threshold):g}r_pct"] = (
            None if not bool(reached.any()) else float((initial_stop & reached & (exit_date <= first_date)).loc[reached].mean() * 100.0)
        )
    return out


def truth_outcome_cohorts(
    planned: pd.DataFrame,
    path: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    thresholds: Sequence[float],
) -> dict[str, dict[str, Any]]:
    filled = pd.DataFrame(planned).loc[pd.DataFrame(planned)["entry_filled_bool"].fillna(False).astype(bool)].copy()
    joined = filled.merge(path, on="event_key", how="inner", suffixes=("", "_path"), validate="one_to_one")
    if joined.empty:
        return {}
    joined = joined.merge(
        truth[["ticker", "date", "quadrant"]],
        left_on=["ticker", "score_event_date"],
        right_on=["ticker", "date"],
        how="left",
        validate="many_to_one",
    )
    out: dict[str, dict[str, Any]] = {}
    for quadrant, group in joined.groupby("quadrant", dropna=False, sort=True):
        fake_planned = group[["event_key"]].copy()
        fake_planned["entry_filled_bool"] = True
        out[str(quadrant or "unclassified")] = summarize_path_cohort(fake_planned, group, thresholds=thresholds)
    return out


def _period(source) -> tuple[str, str]:
    period = dict(source.result.get("comparison_period") or {})
    start = str(period.get("start") or "")[:10]
    end = str(period.get("end") or "")[:10]
    if not start or not end:
        raise AuditSourceBlockedError(f"{source.profile_id}缺少comparison_period")
    return start, end


def _mode_result(definition, *, project_root: Path, profile_id: str) -> dict[str, Any]:
    source_cfg = dict(definition.source)
    fingerprint = str(dict(source_cfg.get("strategy_result_fingerprints") or {}).get(profile_id) or "")
    source = load_strategy_compare_source(project_root, profile_id=profile_id, pinned_config_fingerprint=fingerprint or None)
    control = str(source_cfg["control_arm_id"]); treatment = str(source_cfg["treatment_arm_id"])
    configured_thresholds = tuple(float(v) for v in definition.dimensions.get("first_passage_thresholds_r", TRADE_OUTCOME_FIRST_PASSAGE_THRESHOLDS_R))
    if configured_thresholds != TRADE_OUTCOME_FIRST_PASSAGE_THRESHOLDS_R:
        raise AuditBlockedError(
            "persistent Trade Outcome／Path contract固定first-passage thresholds=+1R/+2R/+3R；"
            f"config={configured_thresholds}"
        )
    thresholds = TRADE_OUTCOME_FIRST_PASSAGE_THRESHOLDS_R
    start, end = _period(source)
    truth, truth_source = build_truth_geometry(
        project_root=project_root,
        filter_id=str(source_cfg["filter_id"]),
        model_architecture=str(source_cfg["model_architecture"]),
        provider_profile_id=str(source_cfg["truth_provider_profile_id"]),
        mfe_target_id=str(source_cfg["mfe_target_id"]),
        safety_target_id=str(source_cfg["safety_target_id"]),
        percentile_method=str(definition.dimensions.get("percentile_method") or "average_zero_based"),
    )
    truth = attach_quadrants(filter_period(truth, start, end), cutoff=float(definition.dimensions.get("truth_high_cutoff", 0.5)))
    arms: dict[str, Any] = {}
    planned_frames: dict[str, pd.DataFrame] = {}
    path_frames: dict[str, pd.DataFrame] = {}
    for arm_id in (control, treatment):
        evidence = load_strategy_arm_path_sidecars(project_root, source=source, arm_id=arm_id)
        planned = build_planned_membership(
            orderable=pd.DataFrame(evidence["orderable"]),
            execution=pd.DataFrame(evidence["execution"]),
        )
        path = normalize_path(pd.DataFrame(evidence["upside_realization"]), thresholds=thresholds)
        planned_frames[arm_id] = planned
        path_frames[arm_id] = path
        arms[arm_id] = {
            "summary": summarize_path_cohort(planned, path, thresholds=thresholds),
            "truth_outcomes": truth_outcome_cohorts(planned, path, truth, thresholds=thresholds),
        }
    cohorts = pair_membership_cohorts(planned_frames[control], planned_frames[treatment])
    pair = {
        "common": summarize_path_cohort(cohorts["common"], path_frames[treatment], thresholds=thresholds),
        "control_only": summarize_path_cohort(cohorts["control_only"], path_frames[control], thresholds=thresholds),
        "treatment_only": summarize_path_cohort(cohorts["treatment_only"], path_frames[treatment], thresholds=thresholds),
    }
    return {
        "profile_id": profile_id,
        "display_name": source.settings.profile_label,
        "config_fingerprint": source.config_fingerprint,
        "period": {"start": start, "end": end},
        "control_arm_id": control,
        "treatment_arm_id": treatment,
        "thresholds_r": list(thresholds),
        "truth_source": truth_source,
        "arms": arms,
        "pair_cohorts": pair,
        "contract": {"scope": "planned→filled→realized", "future_truth_used_for_runtime": False},
    }


def build_report(definition, *, project_root: Path) -> dict[str, Any]:
    return {"modes": [
        _mode_result(definition, project_root=project_root, profile_id=str(profile_id))
        for profile_id in definition.source.get("evaluation_profile_ids", ())
    ]}


def _table(headers, rows, *, target: str) -> str:
    return render_table(headers, rows) if target == "console" else markdown_table(headers, rows)


def render_result(result: Mapping[str, Any], *, target: str = "console") -> str:
    lines = [audit_title("Trade Outcome／Path Attribution", target=target), source_summary(result["definition"], target=target)]
    section = 1
    evidence: list[tuple[str, str, str]] = []
    for mode in result["modes"]:
        control = str(mode["control_arm_id"]); treatment = str(mode["treatment_arm_id"])
        thresholds = [float(x) for x in mode["thresholds_r"]]
        lines.append(audit_section(f"{mode['display_name']}｜{section_contract("audit.trade_outcome_path", "planned_to_filled").title}", section, target=target)); section += 1
        rows = []
        for arm_id in (control, treatment):
            s = dict(mode["arms"][arm_id]["summary"])
            rows.append(format_contract_row(
                table_contract("audit.trade_outcome_path", "planned_to_filled", "planned_to_filled"),
                {"arm": arm_id, **s},
            ))
        lines.append(_table(table_contract("audit.trade_outcome_path", "planned_to_filled", "planned_to_filled").headers, rows, target=target))

        lines.append(audit_section(f"{mode['display_name']}｜{section_contract("audit.trade_outcome_path", "realized_first_passage").title}", section, target=target)); section += 1
        if tuple(thresholds) != TRADE_OUTCOME_FIRST_PASSAGE_THRESHOLDS_R:
            raise AuditBlockedError("Trade Outcome／Path payload與persistent first-passage contract不一致")
        realized_table = table_contract("audit.trade_outcome_path", "realized_first_passage", "realized_first_passage")
        rows = []
        for arm_id in (control, treatment):
            s = dict(mode["arms"][arm_id]["summary"])
            rows.append(format_contract_row(realized_table, {"arm": arm_id, **s}))
        lines.append(_table(realized_table.headers, rows, target=target))

        lines.append(audit_section(f"{mode['display_name']}｜{section_contract("audit.trade_outcome_path", "pair_cohorts").title}", section, target=target)); section += 1
        rows=[]
        for key,label in (("common","Common"),("control_only",f"{control}-only"),("treatment_only",f"{treatment}-only")):
            s=dict(mode["pair_cohorts"].get(key) or {})
            rows.append(format_contract_row(
                table_contract("audit.trade_outcome_path", "pair_cohorts", "pair_cohorts"),
                {"cohort": label, **s},
            ))
        lines.append(_table(table_contract("audit.trade_outcome_path", "pair_cohorts", "pair_cohorts").headers,rows,target=target))

        lines.append(audit_section(f"{mode['display_name']}｜{section_contract("audit.trade_outcome_path", "truth_cohort_outcome").title}", section, target=target)); section += 1
        rows=[]
        for arm_id in (control,treatment):
            for quadrant in ("high_mfe_high_safety_pct","high_mfe_low_safety_pct","low_mfe_high_safety_pct","low_mfe_low_safety_pct"):
                s=dict(mode["arms"][arm_id]["truth_outcomes"].get(quadrant) or {})
                if not s: continue
                rows.append(format_contract_row(
                    table_contract("audit.trade_outcome_path", "truth_cohort_outcome", "truth_cohort_outcome"),
                    {"arm": arm_id, "truth_cohort": quadrant.replace("_pct", ""), **s},
                ))
        lines.append(_table(table_contract("audit.trade_outcome_path", "truth_cohort_outcome", "truth_cohort_outcome").headers,rows,target=target))

        t_summary=dict(mode["arms"][treatment]["summary"]); c_summary=dict(mode["arms"][control]["summary"])
        delta = None if t_summary.get("realized_mean_r") is None or c_summary.get("realized_mean_r") is None else float(t_summary["realized_mean_r"])-float(c_summary["realized_mean_r"])
        realized_column = next(column for column in table_contract("audit.trade_outcome_path", "pair_cohorts", "pair_cohorts").columns if column.key == "realized_mean_r")
        evidence.append((
            str(mode["display_name"]),
            evidence_status_for_delta(delta, preference=realized_column.preference),
            f"Treatment-Control Avg R={format_contract_value(realized_column, delta)}",
        ))
    lines.append(audit_section(section_contract("audit.trade_outcome_path", "key_evidence").title, section, target=target))
    lines.append(render_evidence_rows(evidence,target=target))
    return "\n\n".join(str(x) for x in lines if str(x).strip())


def preflight(definition, *, project_root: Path) -> dict[str, Any]:
    try:
        for profile_id in definition.source.get("evaluation_profile_ids", ()):
            source=load_strategy_compare_source(project_root,profile_id=str(profile_id),pinned_config_fingerprint=str(dict(definition.source.get("strategy_result_fingerprints") or {}).get(str(profile_id)) or "") or None)
            for arm_id in (definition.source["control_arm_id"],definition.source["treatment_arm_id"]):
                load_strategy_arm_path_sidecars(project_root,source=source,arm_id=str(arm_id))
        return {"status":"READY","reason":"","source":{"display":"canonical Strategy Compare path sidecars"}}
    except (AuditSourceBlockedError,FileNotFoundError,ValueError,OSError) as exc:
        return {"status":"BLOCKED","reason":str(exc),"source":{"display":"canonical Strategy Compare path sidecars"}}


def collect_status(definition, *, project_root: Path) -> dict[str, Any]:
    return preflight(definition,project_root=project_root)


def run_formal_audit(definition, *, project_root: Path, quiet: bool=False) -> dict[str, Any]:
    result=build_report(definition,project_root=Path(project_root).resolve()); result["definition"]=definition
    console=render_result(result,target="console"); markdown=render_result(result,target="markdown")
    payload={k:v for k,v in result.items() if k!="definition"}
    saved=persist_reusable_report(definition=definition,project_root=project_root,payload=payload,console_text=console,markdown_text=markdown)
    if not quiet: print(console)
    return saved


__all__=["SUPPORTED_AUDIT_TYPE","build_report","collect_status","normalize_path","preflight","render_result","run_formal_audit","summarize_path_cohort"]
