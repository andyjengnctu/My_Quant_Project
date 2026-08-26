"""Reusable Opportunity / Selection Attribution report.

Purpose: locate where actual opportunity changes from the daily-universal truth
through breakout qualification, orderable membership and final planned baskets.
Resource-constraint decomposition (K/R0/cash/slots) is intentionally excluded and
remains a targeted one-time Audit when decision-relevant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from core.console_report import render_table

from filters.breakout_quality.daily_ranker_data import load_official_breakout_candidate_keys
from filters.breakout_quality.mfe_safety_geometry import (
    attach_quadrants,
    build_truth_geometry,
    distribution_for_keys,
    filter_period,
    normalize_date,
    normalize_ticker,
    truth_geometry_5x5,
)
from services.audit.reusable_report import (
    audit_section,
    audit_title,
    fmt,
    markdown_table,
    persist_reusable_report,
    render_evidence_rows,
    render_truth_5x5,
    source_summary,
    truth_cell,
)
from services.audit.selection_membership import (
    build_planned_membership,
    normalize_orderable_membership,
    pair_membership_cohorts,
    truth_keys,
)
from services.audit.strategy_compare_source import (
    AuditSourceBlockedError,
    load_strategy_arm_planned_sidecars,
    load_strategy_compare_source,
)

SUPPORTED_AUDIT_TYPE = "opportunity_selection_attribution"


def _period(source) -> tuple[str, str]:
    period = dict(source.result.get("comparison_period") or {})
    start = str(period.get("start") or "")[:10]
    end = str(period.get("end") or "")[:10]
    if not start or not end:
        raise AuditSourceBlockedError(f"{source.profile_id}缺少comparison_period")
    return start, end


def _truth_summary(geometry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "n": int(geometry.get("truth_covered_rows", 0) or 0),
        "rho": geometry.get("safety_to_mfe_mean_daily_spearman"),
        "s5_m5": dict(geometry.get("s5_m5") or {}),
        "s4plus_m4plus": dict(geometry.get("s4plus_m4plus") or {}),
    }


def _geometry_row(label: str, distribution: Mapping[str, Any]) -> list[str]:
    d = dict(distribution or {})
    return [
        label,
        f"{int(d.get('truth_covered_rows', 0) or 0):,}",
        fmt(d.get("truth_coverage_pct"), 2, "%"),
        fmt(d.get("high_mfe_high_safety_pct"), 2, "%"),
        fmt(d.get("high_mfe_low_safety_pct"), 2, "%"),
        fmt(d.get("low_mfe_high_safety_pct"), 2, "%"),
        fmt(d.get("low_mfe_low_safety_pct"), 2, "%"),
        fmt(d.get("high_mfe_total_pct"), 2, "%"),
        fmt(d.get("high_safety_total_pct"), 2, "%"),
    ]


def _membership_summary(frame: pd.DataFrame, truth: pd.DataFrame) -> dict[str, Any]:
    keys = truth_keys(frame)
    distribution = distribution_for_keys(truth, keys, allow_empty=True)
    score = pd.to_numeric(frame.get("score_percentile"), errors="coerce")
    finite = score.dropna()
    return {
        **distribution,
        "score_percentile_mean": None if finite.empty else float(finite.mean()),
        "score_percentile_median": None if finite.empty else float(finite.median()),
    }


def _selection_translation(source, arm_id: str) -> dict[str, Any]:
    diagnostics = dict(source.result.get("diagnostics") or {})
    rows = list(diagnostics.get("selection_translation") or [])
    for row in rows:
        if str(row.get("arm_id") or "") == str(arm_id):
            return dict(row)
    return {}


def _mode_result(definition, *, project_root: Path, profile_id: str) -> dict[str, Any]:
    source_cfg = dict(definition.source)
    fingerprint = str(dict(source_cfg.get("strategy_result_fingerprints") or {}).get(profile_id) or "")
    source = load_strategy_compare_source(
        project_root,
        profile_id=profile_id,
        pinned_config_fingerprint=fingerprint or None,
    )
    control_id = str(source_cfg["control_arm_id"])
    treatment_id = str(source_cfg["treatment_arm_id"])
    cutoff = float(definition.dimensions.get("truth_high_cutoff", 0.5))
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
    truth = attach_quadrants(filter_period(truth, start, end), cutoff=cutoff)
    if truth.empty:
        raise AuditSourceBlockedError(f"{profile_id} period內沒有canonical truth")

    breakout_keys_raw = load_official_breakout_candidate_keys(
        str(source_cfg["filter_id"]), allow_stale_source=False
    )
    breakout_keys = pd.DataFrame(
        [(normalize_ticker(ticker), normalize_date(date)) for ticker, date in breakout_keys_raw],
        columns=["ticker", "date"],
    )
    breakout_keys = breakout_keys.loc[
        breakout_keys["date"].ge(start) & breakout_keys["date"].le(end)
    ].drop_duplicates(["ticker", "date"])

    arm_payloads: dict[str, Any] = {}
    for arm_id in (control_id, treatment_id):
        evidence = load_strategy_arm_planned_sidecars(project_root, source=source, arm_id=arm_id)
        orderable = normalize_orderable_membership(pd.DataFrame(evidence["orderable"]))
        planned = build_planned_membership(
            orderable=pd.DataFrame(evidence["orderable"]),
            execution=pd.DataFrame(evidence["execution"]),
        )
        orderable_keys = truth_keys(orderable)
        planned_keys = truth_keys(planned)
        arm_payloads[arm_id] = {
            "orderable": distribution_for_keys(truth, orderable_keys, allow_empty=True),
            "planned": distribution_for_keys(truth, planned_keys, allow_empty=True),
            "planned_5x5": truth_geometry_5x5(truth, planned_keys),
            "planned_rows": planned,
            "selection_quality": _selection_translation(source, arm_id),
        }

    cohorts = pair_membership_cohorts(
        arm_payloads[control_id]["planned_rows"],
        arm_payloads[treatment_id]["planned_rows"],
    )
    cohort_summary = {
        key: _membership_summary(frame, truth)
        for key, frame in cohorts.items()
    }
    daily = truth_geometry_5x5(truth)
    breakout = truth_geometry_5x5(truth, breakout_keys)
    breakout_distribution = distribution_for_keys(truth, breakout_keys, allow_empty=True)
    delta_cells: list[list[float | None]] = []
    c_cells = list(arm_payloads[control_id]["planned_5x5"].get("cells") or [])
    t_cells = list(arm_payloads[treatment_id]["planned_5x5"].get("cells") or [])
    for c_row, t_row in zip(c_cells, t_cells):
        delta_cells.append([
            None
            if c.get("population_pct") is None or t.get("population_pct") is None
            else float(t["population_pct"]) - float(c["population_pct"])
            for c, t in zip(c_row, t_row)
        ])
    return {
        "profile_id": profile_id,
        "display_name": source.settings.profile_label,
        "config_fingerprint": source.config_fingerprint,
        "period": {"start": start, "end": end},
        "control_arm_id": control_id,
        "treatment_arm_id": treatment_id,
        "truth_source": truth_source,
        "daily_actual": daily,
        "breakout_actual": breakout,
        "breakout_distribution": breakout_distribution,
        "breakout_candidate_rows": int(len(breakout_keys)),
        "arms": {
            arm_id: {
                key: value
                for key, value in payload.items()
                if key != "planned_rows"
            }
            for arm_id, payload in arm_payloads.items()
        },
        "pair_cohorts": cohort_summary,
        "planned_share_delta_pp": delta_cells,
        "contract": {
            "percentile_policy": "daily-universal canonical percentile; subset filter only; no rerank",
            "resource_binding_decomposition": False,
            "future_truth_used_for_runtime": False,
        },
    }


def build_report(definition, *, project_root: Path) -> dict[str, Any]:
    modes = [
        _mode_result(definition, project_root=project_root, profile_id=str(profile_id))
        for profile_id in definition.source.get("evaluation_profile_ids", ())
    ]
    return {"modes": modes}


def _render_mode(mode: Mapping[str, Any], *, target: str, section_start: int) -> tuple[list[str], int]:
    control = str(mode["control_arm_id"])
    treatment = str(mode["treatment_arm_id"])
    daily = dict(mode["daily_actual"])
    breakout = dict(mode["breakout_actual"])
    arms = dict(mode["arms"])
    lines: list[str] = []
    n = section_start

    lines.append(audit_section(f"{mode['display_name']}｜Actual Opportunity Baseline", n, target=target)); n += 1
    summary_rows = []
    for label, geometry in (("Daily universal", daily), ("Breakout candidate", breakout)):
        summary = _truth_summary(geometry)
        summary_rows.append([
            label,
            f"{summary['n']:,}",
            fmt(summary["rho"], 4),
            truth_cell(summary["s5_m5"]),
            truth_cell(summary["s4plus_m4plus"]),
        ])
    headers = ["Scope", "N", "Actual S↔MFE Daily rho", "S5×M5 N/Pop/×Exp", "S4+×M4+ N/Pop/×Exp"]
    lines.append(
        render_table(headers, summary_rows)
        if target == "console" else markdown_table(headers, summary_rows)
    )
    lines.append(("Daily actual 5×5\n" if target == "console" else "### Daily actual 5×5\n") + render_truth_5x5(daily, target=target))
    lines.append(("Breakout actual 5×5\n" if target == "console" else "### Breakout actual 5×5\n") + render_truth_5x5(breakout, target=target))

    lines.append(audit_section(f"{mode['display_name']}｜Breakout → Orderable → Planned", n, target=target)); n += 1
    geometry_rows = [_geometry_row("Breakout", mode["breakout_distribution"])]
    for arm_id in (control, treatment):
        geometry_rows.append(_geometry_row(f"{arm_id} Orderable", arms[arm_id]["orderable"]))
        geometry_rows.append(_geometry_row(f"{arm_id} Planned", arms[arm_id]["planned"]))
    headers2 = ["Cohort", "N", "Truth Cov", "HM/HS", "HM/LS", "LM/HS", "LM/LS", "High-MFE", "High-Safety"]
    lines.append(
        render_table(headers2, geometry_rows)
        if target == "console" else markdown_table(headers2, geometry_rows)
    )

    lines.append(audit_section(f"{mode['display_name']}｜Final Planned 5×5", n, target=target)); n += 1
    lines.append((f"{control} Planned\n" if target == "console" else f"### {control} Planned\n") + render_truth_5x5(arms[control]["planned_5x5"], target=target))
    lines.append((f"{treatment} Planned\n" if target == "console" else f"### {treatment} Planned\n") + render_truth_5x5(arms[treatment]["planned_5x5"], target=target))
    delta_rows = [
        [f"S{idx}", *("-" if value is None else f"{float(value):+.2f}pp" for value in row)]
        for idx, row in enumerate(mode["planned_share_delta_pp"], start=1)
    ]
    delta_headers = ["Treatment-Control", "M1", "M2", "M3", "M4", "M5"]
    lines.append(("Treatment − Control population-share Δ\n" if target == "console" else "### Treatment − Control population-share Δ\n") + (
        render_table(delta_headers, delta_rows)
        if target == "console" else markdown_table(delta_headers, delta_rows)
    ))

    lines.append(audit_section(f"{mode['display_name']}｜Pair Membership Attribution", n, target=target)); n += 1
    cohort_rows = []
    for key, label in (("common", "Common"), ("control_only", f"{control}-only"), ("treatment_only", f"{treatment}-only")):
        row = dict(mode["pair_cohorts"].get(key) or {})
        cohort_rows.append([
            label,
            f"{int(row.get('truth_covered_rows', 0) or 0):,}",
            fmt(row.get("score_percentile_mean"), 3),
            fmt(row.get("high_mfe_high_safety_pct"), 2, "%"),
            fmt(row.get("high_mfe_low_safety_pct"), 2, "%"),
            fmt(row.get("high_mfe_total_pct"), 2, "%"),
            fmt(row.get("high_safety_total_pct"), 2, "%"),
        ])
    ch = ["Cohort", "N", "Score %ile", "HM/HS", "HM/LS", "High-MFE", "High-Safety"]
    lines.append(
        render_table(ch, cohort_rows)
        if target == "console" else markdown_table(ch, cohort_rows)
    )

    lines.append(audit_section(f"{mode['display_name']}｜Selection Quality", n, target=target)); n += 1
    quality_rows = []
    for arm_id in (control, treatment):
        row = dict(arms[arm_id].get("selection_quality") or {})
        quality_rows.append([
            arm_id,
            fmt(row.get("selected_target_mean_r"), 3, "R"),
            fmt(row.get("selected_target_percentile"), 3),
            fmt(row.get("target_top_k_retention"), 2, "%"),
            fmt(row.get("target_opportunity_gap_r"), 3, "R"),
            fmt(row.get("r_conversion_efficiency"), 3),
        ])
    qh = ["Arm", "Target mean", "Target %ile", "Top-K retention", "Opp gap", "RCE"]
    lines.append(
        render_table(qh, quality_rows)
        if target == "console" else markdown_table(qh, quality_rows)
    )
    return lines, n


def render_result(result: Mapping[str, Any], *, target: str = "console") -> str:
    title = "Opportunity／Selection Attribution"
    lines = [audit_title(title, target=target), source_summary(result["definition"], target=target)]
    section = 1
    for mode in result["modes"]:
        rendered, section = _render_mode(mode, target=target, section_start=section)
        lines.extend(rendered)
    evidence_rows = []
    for mode in result["modes"]:
        daily_s5 = dict(mode["daily_actual"].get("s5_m5") or {}).get("n", 0)
        breakout_s5 = dict(mode["breakout_actual"].get("s5_m5") or {}).get("n", 0)
        evidence_rows.append((
            str(mode["display_name"]),
            "PRESENT" if int(daily_s5 or 0) > 0 and int(breakout_s5 or 0) > 0 else "MISSING",
            f"Daily S5×M5={int(daily_s5 or 0):,}; Breakout S5×M5={int(breakout_s5 or 0):,}",
        ))
    lines.append(audit_section("Key Evidence", section, target=target))
    lines.append(render_evidence_rows(evidence_rows, target=target))
    lines.append(
        "K/R0/cash/slot binding不屬本Reusable report；只有資源約束本身成為待決策問題時才使用One-time Resource Constraint Attribution。"
    )
    return "\n\n".join(str(line) for line in lines if str(line).strip())


def preflight(definition, *, project_root: Path) -> dict[str, Any]:
    try:
        for profile_id in definition.source.get("evaluation_profile_ids", ()):
            source = load_strategy_compare_source(
                project_root,
                profile_id=str(profile_id),
                pinned_config_fingerprint=str(dict(definition.source.get("strategy_result_fingerprints") or {}).get(str(profile_id)) or "") or None,
            )
            for arm_id in (definition.source["control_arm_id"], definition.source["treatment_arm_id"]):
                load_strategy_arm_pipeline_sidecars(project_root, source=source, arm_id=str(arm_id))
        return {"status": "READY", "reason": "", "source": {"display": "canonical Strategy Compare + daily truth"}}
    except (AuditSourceBlockedError, FileNotFoundError, ValueError, OSError) as exc:
        return {"status": "BLOCKED", "reason": str(exc), "source": {"display": "canonical Strategy Compare + daily truth"}}


def collect_status(definition, *, project_root: Path) -> dict[str, Any]:
    return preflight(definition, project_root=project_root)


def run_formal_audit(definition, *, project_root: Path, quiet: bool = False) -> dict[str, Any]:
    result = build_report(definition, project_root=Path(project_root).resolve())
    result["definition"] = definition
    console = render_result(result, target="console")
    markdown = render_result(result, target="markdown")
    payload = {key: value for key, value in result.items() if key != "definition"}
    saved = persist_reusable_report(
        definition=definition,
        project_root=project_root,
        payload=payload,
        console_text=console,
        markdown_text=markdown,
    )
    if not quiet:
        print(console)
    return saved


__all__ = ["SUPPORTED_AUDIT_TYPE", "build_report", "collect_status", "preflight", "render_result", "run_formal_audit"]
