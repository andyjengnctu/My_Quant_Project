"""Reusable realized-trades → portfolio capital / drawdown attribution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from core.console_report import render_table
from core.data_utils import discover_unique_csv_map
from core.dataset_profiles import get_dataset_dir
from filters.breakout_quality.mfe_safety_geometry import (
    attach_quadrants,
    build_truth_geometry,
    filter_period,
    normalize_ticker,
)
from filters.breakout_quality.trade_attribution import reconstruct_round_trips
from services.audit.mfe_safety_truth import AuditBlockedError
from services.audit.portfolio_mtm import build_drawdown_analysis
from services.audit.reusable_report import (
    audit_section,
    audit_title,
    fmt,
    markdown_table,
    persist_reusable_report,
    render_evidence_rows,
    source_summary,
)
from services.audit.strategy_compare_source import (
    AuditSourceBlockedError,
    load_strategy_arm_path_sidecars,
    load_strategy_compare_source,
)

SUPPORTED_AUDIT_TYPE = "portfolio_drawdown_attribution"


def _capital_summary(evidence: Mapping[str, Any]) -> dict[str, Any]:
    execution = pd.DataFrame(evidence["execution"]).copy()
    chosen = pd.to_numeric(
        execution.get("chosen_qty", pd.Series(index=execution.index, dtype=float)),
        errors="coerce",
    ).fillna(0) > 0
    execution = execution.loc[chosen].copy()
    if execution.empty:
        raise AuditBlockedError("execution sidecar沒有planned positions")
    for column in ("limit_px", "init_sl", "sizing_equity", "reserved_cost", "chosen_risk_utilization"):
        if column in execution.columns:
            execution[column] = pd.to_numeric(execution[column], errors="coerce")
    stop = (
        (execution["limit_px"] - execution["init_sl"]) / execution["limit_px"] * 100.0
        if {"limit_px", "init_sl"}.issubset(execution.columns)
        else pd.Series(dtype=float)
    )
    reserved = (
        execution["reserved_cost"] / execution["sizing_equity"] * 100.0
        if {"reserved_cost", "sizing_equity"}.issubset(execution.columns)
        else pd.Series(dtype=float)
    )
    equity_path = Path(evidence["pair_dir"]) / "score_ranking_equity.csv"
    if not equity_path.is_file():
        raise AuditBlockedError("缺少score_ranking_equity.csv")
    equity = pd.read_csv(equity_path, encoding="utf-8-sig", low_memory=False)
    exposure = pd.to_numeric(
        equity.get("Exposure_Pct", pd.Series(index=equity.index, dtype=float)),
        errors="coerce",
    )
    round_trips = reconstruct_round_trips(pd.DataFrame(evidence["active_trades"]), scenario="score_ranking")
    holding = pd.to_numeric(
        round_trips.get(
            "holding_calendar_days",
            pd.Series(index=round_trips.index, dtype=float),
        ),
        errors="coerce",
    )
    risk = pd.to_numeric(
        execution.get(
            "chosen_risk_utilization",
            pd.Series(index=execution.index, dtype=float),
        ),
        errors="coerce",
    )
    return {
        "average_exposure_pct": None if exposure.dropna().empty else float(exposure.mean()),
        "median_stop_distance_pct": None if stop.dropna().empty else float(stop.median()),
        "mean_stop_distance_pct": None if stop.dropna().empty else float(stop.mean()),
        "median_reserved_fraction_pct": None if reserved.dropna().empty else float(reserved.median()),
        "mean_reserved_fraction_pct": None if reserved.dropna().empty else float(reserved.mean()),
        "mean_chosen_risk_utilization": None if risk.dropna().empty else float(risk.mean()),
        "mean_holding_calendar_days": None if holding.dropna().empty else float(holding.mean()),
        "median_holding_calendar_days": None if holding.dropna().empty else float(holding.median()),
        "round_trip_count": int(len(round_trips)),
        "planned_position_count": int(len(execution)),
    }


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
    market_data_dir = Path(get_dataset_dir(str(project_root), str(source.settings.dataset))).resolve()
    if not market_data_dir.is_dir():
        raise AuditSourceBlockedError(f"canonical market data目錄不存在: {market_data_dir}")
    discovered, _issues = discover_unique_csv_map(str(market_data_dir))
    csv_map = {normalize_ticker(k): str(v) for k, v in discovered.items()}
    close_cache: dict[str, pd.Series] = {}
    arms: dict[str, Any] = {}
    for arm_id in (control, treatment):
        evidence = load_strategy_arm_path_sidecars(project_root, source=source, arm_id=arm_id)
        capital = _capital_summary(evidence)
        drawdown = build_drawdown_analysis(
            evidence,
            truth,
            cutoff=float(definition.dimensions.get("truth_high_cutoff", 0.5)),
            top_n=int(definition.dimensions.get("drawdown_top_n", 1)),
            market_data_dir=market_data_dir,
            market_csv_map=csv_map,
            market_close_cache=close_cache,
        )
        arms[arm_id] = {"capital": capital, "drawdown": drawdown}
    return {
        "profile_id": profile_id,
        "display_name": source.settings.profile_label,
        "config_fingerprint": source.config_fingerprint,
        "period": {"start": start, "end": end},
        "control_arm_id": control,
        "treatment_arm_id": treatment,
        "truth_source": truth_source,
        "arms": arms,
        "contract": {
            "drawdown_basis": "canonical peak-EOD→trough-EOD exact position MTM; final realized R is not contribution",
            "reconciliation_required": True,
            "binding_signature_reported": False,
        },
    }


def build_report(definition, *, project_root: Path) -> dict[str, Any]:
    return {"modes": [
        _mode_result(definition, project_root=project_root, profile_id=str(profile_id))
        for profile_id in definition.source.get("evaluation_profile_ids", ())
    ]}


def _table(headers, rows, *, target: str) -> str:
    return render_table(headers, rows) if target == "console" else markdown_table(headers, rows)


def _max_dd_row(arm_id: str, payload: Mapping[str, Any]) -> list[str]:
    dd = dict(payload.get("drawdown") or {})
    top = list(dd.get("top_episodes") or [])
    row = dict(top[0]) if top else {}
    return [
        arm_id,
        fmt(dd.get("max_drawdown_pct"), 2, "%"),
        str(row.get("peak_date") or "-"),
        str(row.get("trough_date") or "-"),
        fmt(row.get("position_mtm_contribution_pct_peak_equity"), 2, "%"),
        fmt(row.get("reconciliation_delta"), 3),
    ]


def render_result(result: Mapping[str, Any], *, target: str = "console") -> str:
    lines = [audit_title("Portfolio／Drawdown Attribution", target=target), source_summary(result["definition"], target=target)]
    section = 1
    evidence_rows: list[tuple[str, str, str]] = []
    for mode in result["modes"]:
        control = str(mode["control_arm_id"]); treatment = str(mode["treatment_arm_id"])
        lines.append(audit_section(f"{mode['display_name']}｜Capital / Exposure", section, target=target)); section += 1
        rows=[]
        for arm_id in (control,treatment):
            c=dict(mode["arms"][arm_id]["capital"])
            rows.append([arm_id,fmt(c.get("average_exposure_pct"),2,"%"),fmt(c.get("median_stop_distance_pct"),2,"%"),fmt(c.get("median_reserved_fraction_pct"),2,"%"),fmt(c.get("mean_chosen_risk_utilization"),3),fmt(c.get("mean_holding_calendar_days"),1,"d"),f"{int(c.get('round_trip_count',0) or 0):,}"])
        lines.append(_table(["Arm","Exposure","Median stop dist","Reserved/Equity","Risk util","Holding days","Trades"],rows,target=target))

        lines.append(audit_section(f"{mode['display_name']}｜Max Drawdown MTM", section, target=target)); section += 1
        rows=[_max_dd_row(control,mode["arms"][control]),_max_dd_row(treatment,mode["arms"][treatment])]
        lines.append(_table(["Arm","MDD","Peak","Trough","ΣPosition MTM / Peak","Reconcile Δ"],rows,target=target))

        lines.append(audit_section(f"{mode['display_name']}｜Position Lifecycle", section, target=target)); section += 1
        rows=[]
        for arm_id in (control,treatment):
            top=list(dict(mode["arms"][arm_id]["drawdown"]).get("top_episodes") or []); d=dict(top[0]) if top else {}
            rows.append([arm_id,f"{int(d.get('peak_held_count',0) or 0):,}",f"{int(d.get('entered_during_drawdown_count',0) or 0):,}",f"{int(d.get('exited_during_drawdown_count',0) or 0):,}",f"{int(d.get('trough_held_count',0) or 0):,}",f"{int(d.get('max_same_day_entries',0) or 0):,}"])
        lines.append(_table(["Arm","Peak-held","Entered during DD","Exited during DD","Trough-held","Max same-day entries"],rows,target=target))

        lines.append(audit_section(f"{mode['display_name']}｜Drawdown Truth Contribution", section, target=target)); section += 1
        rows=[]
        for arm_id in (control,treatment):
            top=list(dict(mode["arms"][arm_id]["drawdown"]).get("top_episodes") or []); d=dict(top[0]) if top else {}
            rows.append([arm_id,fmt(d.get("hmhs_mtm_contribution_pct_peak_equity"),2,"%"),fmt(d.get("hmls_mtm_contribution_pct_peak_equity"),2,"%"),fmt(d.get("lmhs_mtm_contribution_pct_peak_equity"),2,"%"),fmt(d.get("lmls_mtm_contribution_pct_peak_equity"),2,"%")])
        lines.append(_table(["Arm","HM/HS","HM/LS","LM/HS","LM/LS"],rows,target=target))

        cdd=dict(mode["arms"][control]["drawdown"]); tdd=dict(mode["arms"][treatment]["drawdown"])
        cmax=cdd.get("max_drawdown_pct"); tmax=tdd.get("max_drawdown_pct")
        delta=None if cmax is None or tmax is None else float(tmax)-float(cmax)
        evidence_rows.append((str(mode["display_name"]),"IMPROVED" if delta is not None and delta<0 else "WORSE" if delta is not None and delta>0 else "MIXED",f"Treatment-Control MDD={fmt(delta,2,'%')}"))
    lines.append(audit_section("Key Evidence", section, target=target))
    lines.append(render_evidence_rows(evidence_rows,target=target))
    lines.append("此Reusable report不顯示K/R0/cash/slot binding signature；資源約束只有在成為獨立待決策問題時才進One-time Audit。")
    return "\n\n".join(str(x) for x in lines if str(x).strip())


def preflight(definition, *, project_root: Path) -> dict[str, Any]:
    try:
        for profile_id in definition.source.get("evaluation_profile_ids", ()):
            source=load_strategy_compare_source(project_root,profile_id=str(profile_id),pinned_config_fingerprint=str(dict(definition.source.get("strategy_result_fingerprints") or {}).get(str(profile_id)) or "") or None)
            for arm_id in (definition.source["control_arm_id"],definition.source["treatment_arm_id"]):
                load_strategy_arm_path_sidecars(project_root,source=source,arm_id=str(arm_id))
            market_data_dir=Path(get_dataset_dir(str(project_root),str(source.settings.dataset))).resolve()
            if not market_data_dir.is_dir(): raise AuditSourceBlockedError("canonical market data目錄不存在")
        return {"status":"READY","reason":"","source":{"display":"canonical Strategy Compare + market Close"}}
    except (AuditSourceBlockedError,FileNotFoundError,ValueError,OSError) as exc:
        return {"status":"BLOCKED","reason":str(exc),"source":{"display":"canonical Strategy Compare + market Close"}}


def collect_status(definition, *, project_root: Path) -> dict[str, Any]:
    return preflight(definition,project_root=project_root)


def run_formal_audit(definition, *, project_root: Path, quiet: bool=False) -> dict[str, Any]:
    result=build_report(definition,project_root=Path(project_root).resolve()); result["definition"]=definition
    console=render_result(result,target="console"); markdown=render_result(result,target="markdown")
    payload={k:v for k,v in result.items() if k!="definition"}
    saved=persist_reusable_report(definition=definition,project_root=project_root,payload=payload,console_text=console,markdown_text=markdown)
    if not quiet: print(console)
    return saved


__all__=["SUPPORTED_AUDIT_TYPE","build_report","collect_status","preflight","render_result","run_formal_audit"]
