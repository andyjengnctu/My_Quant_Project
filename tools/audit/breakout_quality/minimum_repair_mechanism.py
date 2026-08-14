"""Read-only exact minimum-repair mechanism attribution for MR-13E.

Consumes completed Strategy Compare selector trace plus the diagnostic exact-oracle
sidecar produced from the same canonical K/R0 reservation simulator.  Future target
is joined only after replay and never participates in oracle search or selection.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.audit import AUDIT_OUTPUT_ROOT, AuditDefinition
from core.console_report import (
    print_artifact_paths,
    project_relative_display_path,
    render_key_values,
    render_section,
    render_table,
    render_title,
)
from core.runtime_utils import get_taipei_now
from tools.audit.breakout_quality.orderable_alignment_common import (
    information_date_map,
    json_native as _json_native,
    load_common_daily_target as _load_common_daily_target,
    read_csv as _read_csv,
    resolve_phase_source as _resolve_phase_source,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_RESULT_SCHEMA_VERSION = 1
EXACT_RESOURCE_CLASS = "RESOURCE_CONSTRAINT_EXACT_OPTIMUM"
HEURISTIC_CLASSES = {
    "GREEDY_REPAIR_EXTRA_REPLACEMENTS",
    "GREEDY_REPAIR_SCORE_GAP_AT_MIN_DISTANCE",
    "MULTI_SWAP_LOCAL_SEARCH_GAP",
}


def _required_arm_paths(arm) -> tuple[Path, ...]:
    return (
        arm.pair_dir / f"{arm.prefix}_selector_trace.csv",
        arm.pair_dir / f"{arm.prefix}_repair_mechanism.csv",
        arm.pair_dir / "strategy_comparison.json",
    )


def collect_minimum_repair_mechanism_status(
    definition: AuditDefinition,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    try:
        sources = {
            phase_id: _resolve_phase_source(root, definition, phase_id)
            for phase_id in ("selection_pit", "forward_oos")
        }
        for source in sources.values():
            for arm in source["candidates"].values():
                for path in _required_arm_paths(arm):
                    if not path.is_file():
                        raise FileNotFoundError(
                            f"{source['phase_id']} {arm.arm_id}缺少minimum-repair read-only工件: "
                            + project_relative_display_path(path, project_root=root)
                        )
        return {
            "audit_id": definition.audit_id,
            "status": "READY",
            "reason": "",
            "source": {
                "display": "12B / 13A / 13E｜minimum-repair exact K/R0 oracle",
                "selection_run": project_relative_display_path(
                    sources["selection_pit"]["run_dir"], project_root=root
                ),
                "forward_run": project_relative_display_path(
                    sources["forward_oos"]["run_dir"], project_root=root
                ),
            },
        }
    except (FileNotFoundError, ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        return {
            "audit_id": definition.audit_id,
            "status": "BLOCKED",
            "reason": str(exc),
            "source": {"display": "12B / 13A / 13E｜minimum-repair exact K/R0 oracle"},
        }


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values) -> float | None:
    series = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return None if series.empty else float(series.mean())


def _annotate_target(
    rows: pd.DataFrame,
    *,
    target_lookup: pd.DataFrame,
    target_calendar: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(rows).copy()
    if frame.empty:
        frame["common_target_raw_r"] = pd.Series(dtype=float)
        frame["target_available"] = pd.Series(dtype=bool)
        return frame
    required = {"ticker", "trade_date"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"minimum-repair candidate rows缺少欄位: {missing}")
    frame["ticker"] = frame["ticker"].fillna("").astype(str).str.strip()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="raise").dt.strftime("%Y-%m-%d")
    frame["common_information_date"] = information_date_map(frame["trade_date"], target_calendar)
    lookup = target_lookup.reset_index().rename(columns={"date": "common_information_date"})
    frame = frame.merge(
        lookup,
        on=["ticker", "common_information_date"],
        how="left",
        validate="many_to_one",
    )
    frame["common_target_raw_r"] = pd.to_numeric(frame["common_target_raw_r"], errors="coerce")
    frame["target_available"] = frame["common_target_raw_r"].map(math.isfinite)
    return frame


def _stage_daily_target(
    trace: pd.DataFrame,
    *,
    target_lookup: pd.DataFrame,
    target_calendar: np.ndarray,
    repair_dates: set[str],
) -> pd.DataFrame:
    rows = pd.DataFrame(trace).copy()
    if rows.empty:
        return pd.DataFrame(columns=["trade_date", "stage", "target_mean_r"])
    rows = rows[rows["stage"].astype(str).isin({
        "raw_top_n", "minimum_repair_seed", "feasible_ascent_final"
    })].copy()
    rows = rows[rows["trade_date"].astype(str).isin(repair_dates)].copy()
    rows = _annotate_target(rows, target_lookup=target_lookup, target_calendar=target_calendar)
    rows = rows[rows["target_available"]].copy()
    if rows.empty:
        return pd.DataFrame(columns=["trade_date", "stage", "target_mean_r"])
    return (
        rows.groupby(["trade_date", "stage"], as_index=False)["common_target_raw_r"]
        .mean()
        .rename(columns={"common_target_raw_r": "target_mean_r"})
    )


def _oracle_daily_target(
    mechanism: pd.DataFrame,
    *,
    target_lookup: pd.DataFrame,
    target_calendar: np.ndarray,
) -> pd.DataFrame:
    rows = mechanism[mechanism["trace_kind"].astype(str) == "repair_oracle_basket"].copy()
    if rows.empty:
        return pd.DataFrame(columns=["trade_date", "oracle_stage", "target_mean_r"])
    rows = _annotate_target(rows, target_lookup=target_lookup, target_calendar=target_calendar)
    rows = rows[rows["target_available"]].copy()
    if rows.empty:
        return pd.DataFrame(columns=["trade_date", "oracle_stage", "target_mean_r"])
    return (
        rows.groupby(["trade_date", "oracle_stage"], as_index=False)["common_target_raw_r"]
        .mean()
        .rename(columns={"common_target_raw_r": "target_mean_r"})
    )


def _swap_daily(
    mechanism: pd.DataFrame,
    *,
    target_lookup: pd.DataFrame,
    target_calendar: np.ndarray,
) -> pd.DataFrame:
    rows = mechanism[mechanism["trace_kind"].astype(str) == "repair_swap"].copy()
    if rows.empty:
        return pd.DataFrame(columns=[
            "trade_date", "repair_step", "target_delta_r", "score_delta",
            "reserve_deficit_delta_milli", "count_deficit_delta",
        ])
    rows = _annotate_target(rows, target_lookup=target_lookup, target_calendar=target_calendar)
    rows["breakout_quality_score"] = pd.to_numeric(rows["breakout_quality_score"], errors="coerce")
    result = []
    for (trade_date, repair_step), part in rows.groupby(["trade_date", "repair_step"], sort=True):
        out = part[part["repair_role"].astype(str) == "out"]
        incoming = part[part["repair_role"].astype(str) == "in"]
        if len(out) != 1 or len(incoming) != 1:
            continue
        out_row = out.iloc[0]
        in_row = incoming.iloc[0]
        out_target = _finite(out_row.get("common_target_raw_r"))
        in_target = _finite(in_row.get("common_target_raw_r"))
        out_score = _finite(out_row.get("breakout_quality_score"))
        in_score = _finite(in_row.get("breakout_quality_score"))
        result.append({
            "trade_date": str(trade_date),
            "repair_step": int(repair_step),
            "target_delta_r": None if out_target is None or in_target is None else float(in_target - out_target),
            "score_delta": None if out_score is None or in_score is None else float(in_score - out_score),
            "reserve_deficit_delta_milli": int(in_row.get("after_reserve_deficit_milli", 0) or 0) - int(in_row.get("before_reserve_deficit_milli", 0) or 0),
            "count_deficit_delta": int(in_row.get("after_count_deficit", 0) or 0) - int(in_row.get("before_count_deficit", 0) or 0),
        })
    return pd.DataFrame(result)


def _k_distribution(summary: pd.DataFrame) -> dict[str, int]:
    values = pd.to_numeric(summary["target_count"], errors="coerce").dropna().astype(int)
    return {str(int(k)): int(v) for k, v in values.value_counts().sort_index().items()}


def _arm_payload(arm, *, target_lookup: pd.DataFrame, target_calendar: np.ndarray) -> dict[str, Any]:
    mechanism = _read_csv(arm.pair_dir / f"{arm.prefix}_repair_mechanism.csv")
    trace = _read_csv(arm.pair_dir / f"{arm.prefix}_selector_trace.csv")
    summary = mechanism[mechanism["trace_kind"].astype(str) == "repair_summary"].copy()
    if summary.empty:
        return {
            "arm_id": arm.arm_id,
            "display_name": str(arm.arm.get("name") or arm.arm_id),
            "repair_days": 0,
            "status": "NO_REPAIR_DAYS",
        }
    if (summary["status"].fillna("").astype(str) != "AVAILABLE").any():
        invalid = sorted(set(summary.loc[summary["status"].astype(str) != "AVAILABLE", "status"].astype(str)))
        raise ValueError(f"{arm.arm_id} exact repair oracle不可用: {invalid}")
    summary["trade_date"] = pd.to_datetime(summary["trade_date"], errors="raise").dt.strftime("%Y-%m-%d")
    repair_dates = set(summary["trade_date"].astype(str))
    stage_daily = _stage_daily_target(
        trace,
        target_lookup=target_lookup,
        target_calendar=target_calendar,
        repair_dates=repair_dates,
    )
    oracle_daily = _oracle_daily_target(
        mechanism,
        target_lookup=target_lookup,
        target_calendar=target_calendar,
    )
    swaps = _swap_daily(
        mechanism,
        target_lookup=target_lookup,
        target_calendar=target_calendar,
    )

    stage_pivot = stage_daily.pivot(index="trade_date", columns="stage", values="target_mean_r") if not stage_daily.empty else pd.DataFrame()
    oracle_pivot = oracle_daily.pivot(index="trade_date", columns="oracle_stage", values="target_mean_r") if not oracle_daily.empty else pd.DataFrame()
    daily = summary.set_index("trade_date").copy()
    for column in ("raw_top_n", "minimum_repair_seed", "feasible_ascent_final"):
        daily[f"target_{column}"] = stage_pivot[column] if column in stage_pivot else np.nan
    for column in ("minimum_replacement_best", "global_best"):
        daily[f"target_{column}"] = oracle_pivot[column] if column in oracle_pivot else np.nan
    daily["raw_to_repair_target_delta_r"] = daily["target_minimum_repair_seed"] - daily["target_raw_top_n"]

    classifications = summary["classification"].fillna("").astype(str)
    class_counts = {key: int((classifications == key).sum()) for key in (EXACT_RESOURCE_CLASS, *sorted(HEURISTIC_CLASSES))}
    heuristic_mask = classifications.isin(HEURISTIC_CLASSES).to_numpy()
    exact_mask = (classifications == EXACT_RESOURCE_CLASS).to_numpy()
    target_delta = pd.to_numeric(daily["raw_to_repair_target_delta_r"], errors="coerce").to_numpy(dtype=float)
    heuristic_negative = float(np.nansum(np.minimum(target_delta[heuristic_mask], 0.0))) if heuristic_mask.any() else 0.0
    exact_negative = float(np.nansum(np.minimum(target_delta[exact_mask], 0.0))) if exact_mask.any() else 0.0

    raw_count_deficit = pd.to_numeric(summary["raw_count_deficit"], errors="coerce").fillna(0).astype(int)
    raw_reserve_deficit = pd.to_numeric(summary["raw_reserve_deficit_milli"], errors="coerce").fillna(0).astype(np.int64)
    failure_counts = {
        "count_only": int(((raw_count_deficit > 0) & (raw_reserve_deficit == 0)).sum()),
        "reserve_only": int(((raw_count_deficit == 0) & (raw_reserve_deficit > 0)).sum()),
        "both": int(((raw_count_deficit > 0) & (raw_reserve_deficit > 0)).sum()),
        "none": int(((raw_count_deficit == 0) & (raw_reserve_deficit == 0)).sum()),
    }

    return {
        "arm_id": arm.arm_id,
        "display_name": str(arm.arm.get("name") or arm.arm_id),
        "status": "AVAILABLE",
        "repair_days": int(len(summary)),
        "k_distribution": _k_distribution(summary),
        "raw_failure_counts": failure_counts,
        "classification_counts": class_counts,
        "actual_repair_distance_mean": _mean(summary["actual_repair_replacement_distance"]),
        "exact_min_replacement_distance_mean": _mean(summary["exact_minimum_replacement_distance"]),
        "repair_seed_score_gap_to_minimum_best_mean": _mean(summary["repair_seed_score_gap_to_minimum_best"]),
        "final_score_gap_to_global_best_mean": _mean(summary["final_score_gap_to_global_best"]),
        "raw_reserve_deficit_mean": _mean(summary["raw_reserve_deficit_milli"]),
        "oracle_evaluated_states_mean": _mean(summary["exact_oracle_evaluated_states"]),
        "oracle_ranked_states_mean": _mean(summary["exact_oracle_ranked_states"]),
        "target_stage_means": {
            "raw_top_n": _mean(daily["target_raw_top_n"]),
            "repair_seed": _mean(daily["target_minimum_repair_seed"]),
            "minimum_replacement_best": _mean(daily["target_minimum_replacement_best"]),
            "after_ascent": _mean(daily["target_feasible_ascent_final"]),
            "global_best": _mean(daily["target_global_best"]),
        },
        "repair_swap": {
            "steps": int(len(swaps)),
            "target_delta_r_mean": _mean(swaps.get("target_delta_r", pd.Series(dtype=float))),
            "score_delta_mean": _mean(swaps.get("score_delta", pd.Series(dtype=float))),
            "reserve_deficit_delta_milli_mean": _mean(swaps.get("reserve_deficit_delta_milli", pd.Series(dtype=float))),
            "count_deficit_delta_mean": _mean(swaps.get("count_deficit_delta", pd.Series(dtype=float))),
        },
        "target_loss_attribution": {
            "heuristic_gap_days_negative_raw_to_repair_sum_r": heuristic_negative,
            "exact_optimum_days_negative_raw_to_repair_sum_r": exact_negative,
            "heuristic_gap_days": int(heuristic_mask.sum()),
            "exact_optimum_days": int(exact_mask.sum()),
        },
    }


def _conclusion(payload: dict[str, Any]) -> dict[str, Any]:
    forward = payload["phases"]["forward_oos"]["arms"]
    ids = list(forward)
    if len(ids) < 3:
        return {"classification": "INSUFFICIENT_COMPARATORS"}
    mr12b, mr13a, mr13e = ids[0], ids[1], ids[2]
    e = forward[mr13e]
    if e.get("status") != "AVAILABLE":
        return {"classification": "INSUFFICIENT_REPAIR_DAYS"}
    counts = dict(e["classification_counts"])
    heuristic_days = int(sum(int(counts.get(key, 0)) for key in HEURISTIC_CLASSES))
    exact_days = int(counts.get(EXACT_RESOURCE_CLASS, 0))
    loss = dict(e["target_loss_attribution"])
    heuristic_loss = abs(float(loss.get("heuristic_gap_days_negative_raw_to_repair_sum_r", 0.0)))
    exact_loss = abs(float(loss.get("exact_optimum_days_negative_raw_to_repair_sum_r", 0.0)))
    if heuristic_days == 0:
        classification = "RESOURCE_INCOMPATIBILITY_EXACT_ORACLE"
    elif exact_days == 0:
        classification = "SEARCH_HEURISTIC_DEFICIENCY"
    elif exact_loss > heuristic_loss:
        classification = "RESOURCE_INCOMPATIBILITY_DOMINATES_WITH_HEURISTIC_GAPS"
    elif heuristic_loss > exact_loss:
        classification = "SEARCH_HEURISTIC_GAPS_DOMINATE_TARGET_LOSS"
    else:
        classification = "MIXED_EQUAL_TARGET_LOSS_CONTRIBUTION"
    return {
        "classification": classification,
        "mr13e_arm_id": mr13e,
        "mr12b_arm_id": mr12b,
        "mr13a_arm_id": mr13a,
        "mr13e_exact_optimum_days": exact_days,
        "mr13e_heuristic_gap_days": heuristic_days,
        "mr13e_exact_optimum_negative_target_loss_sum_r": -exact_loss,
        "mr13e_heuristic_gap_negative_target_loss_sum_r": -heuristic_loss,
        "mr13e_classification_counts": counts,
    }


def _fmt(value: Any, digits: int = 4, suffix: str = "") -> str:
    value = _finite(value)
    return "-" if value is None else f"{value:.{digits}f}{suffix}"


def _render_phase(phase_id: str, phase: dict[str, Any]) -> str:
    title = "Selection PIT" if phase_id == "selection_pit" else "Forward-OOS"
    mechanism_rows = []
    target_rows = []
    swap_rows = []
    for arm in phase["arms"].values():
        if arm.get("status") != "AVAILABLE":
            continue
        failures = arm["raw_failure_counts"]
        classes = arm["classification_counts"]
        mechanism_rows.append((
            arm["arm_id"], arm["display_name"], arm["repair_days"],
            failures["count_only"], failures["reserve_only"], failures["both"],
            classes.get(EXACT_RESOURCE_CLASS, 0),
            classes.get("GREEDY_REPAIR_EXTRA_REPLACEMENTS", 0),
            classes.get("GREEDY_REPAIR_SCORE_GAP_AT_MIN_DISTANCE", 0),
            classes.get("MULTI_SWAP_LOCAL_SEARCH_GAP", 0),
            _fmt(arm["actual_repair_distance_mean"]),
            _fmt(arm["exact_min_replacement_distance_mean"]),
            _fmt(arm["repair_seed_score_gap_to_minimum_best_mean"]),
            _fmt(arm["final_score_gap_to_global_best_mean"]),
        ))
        t = arm["target_stage_means"]
        target_rows.append((
            arm["arm_id"], arm["display_name"],
            _fmt(t["raw_top_n"], suffix=" R"),
            _fmt(t["repair_seed"], suffix=" R"),
            _fmt(t["minimum_replacement_best"], suffix=" R"),
            _fmt(t["after_ascent"], suffix=" R"),
            _fmt(t["global_best"], suffix=" R"),
        ))
        s = arm["repair_swap"]
        swap_rows.append((
            arm["arm_id"], arm["display_name"], s["steps"],
            _fmt(s["score_delta_mean"]),
            _fmt(s["target_delta_r_mean"], suffix=" R"),
            _fmt(None if s["reserve_deficit_delta_milli_mean"] is None else s["reserve_deficit_delta_milli_mean"] / 1000.0),
            _fmt(s["count_deficit_delta_mean"]),
            str(arm["k_distribution"]),
        ))
    return "\n\n".join((
        render_section(f"{title}｜exact K/R0 repair mechanism"),
        render_table(
            ("Arm", "Model", "Repair days", "Count-only", "Reserve-only", "Both", "Exact optimum", "Extra repl", "Min-dist score gap", "Multi-swap gap", "Actual repl", "Exact min repl", "Seed score gap", "Final score gap"),
            mechanism_rows,
        ),
        render_section(f"{title}｜Future Target（post-replay attribution only）"),
        render_table(
            ("Arm", "Model", "Raw Top-N", "Repair seed", "Exact min-repl best", "After ascent", "Exact global best"),
            target_rows,
        ),
        render_section(f"{title}｜accepted repair swap"),
        render_table(
            ("Arm", "Model", "Steps", "In−Out Score", "In−Out Target", "Reserve deficit Δ", "Count deficit Δ", "K distribution"),
            swap_rows,
        ),
    ))


def run_minimum_repair_mechanism_audit(
    definition: AuditDefinition,
    *,
    project_root: Path = PROJECT_ROOT,
    quiet: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    status = collect_minimum_repair_mechanism_status(definition, project_root=root)
    if status["status"] != "READY":
        raise RuntimeError(status["reason"])
    sources = {
        phase_id: _resolve_phase_source(root, definition, phase_id)
        for phase_id in ("selection_pit", "forward_oos")
    }
    reference = sources["forward_oos"]
    target_lookup, target_calendar = _load_common_daily_target(
        root=root,
        filter_id=reference["target_filter_id"],
        architecture=reference["target_architecture"],
        profile=reference["target_profile"],
    )
    phases = {}
    for phase_id, source in sources.items():
        arms = {
            arm_id: _arm_payload(
                arm,
                target_lookup=target_lookup,
                target_calendar=target_calendar,
            )
            for arm_id, arm in source["candidates"].items()
        }
        phases[phase_id] = {
            "run_dir": project_relative_display_path(source["run_dir"], project_root=root),
            "arms": arms,
        }
    payload = {
        "schema_version": AUDIT_RESULT_SCHEMA_VERSION,
        "audit_id": definition.audit_id,
        "status": "COMPLETED",
        "generated_at": get_taipei_now().isoformat(),
        "read_only": True,
        "oracle_objective": "canonical frozen max-DL basket quality",
        "oracle_constraints": "same K / exact R0 floor / canonical cash-capped reservation simulator",
        "future_target_join_stage": "post_replay_only",
        "future_target_used_for_oracle": False,
        "phases": phases,
    }
    payload["conclusion"] = _conclusion(payload)

    stamp = get_taipei_now().strftime("%Y%m%d_%H%M%S")
    output_root = (root / AUDIT_OUTPUT_ROOT / definition.output_subdir).resolve()
    run_dir = output_root / "runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / "audit.json"
    md_path = run_dir / "audit.md"
    json_path.write_text(json.dumps(_json_native(payload), ensure_ascii=False, indent=2), encoding="utf-8")

    conclusion = payload["conclusion"]
    md = "\n\n".join((
        render_title("MR-13E Minimum-repair Mechanism Attribution Audit"),
        render_key_values((
            ("Audit", definition.audit_id),
            ("契約", "read-only completed Strategy Compare repair/oracle sidecars；不train、不score、不replay"),
            ("比較", "MR-12B / MR-13A / MR-13E"),
            ("Oracle objective", "frozen DL basket quality；Future Target不參與搜尋"),
            ("Hard constraints", "same K / exact R0 / canonical cash-capped reservation"),
        )),
        _render_phase("selection_pit", phases["selection_pit"]),
        _render_phase("forward_oos", phases["forward_oos"]),
        render_section("判定"),
        render_key_values((
            ("Classification", conclusion.get("classification")),
            ("13E exact-optimum repair days", conclusion.get("mr13e_exact_optimum_days")),
            ("13E heuristic-gap repair days", conclusion.get("mr13e_heuristic_gap_days")),
            ("Exact-optimum days negative Raw→Repair sum", _fmt(conclusion.get("mr13e_exact_optimum_negative_target_loss_sum_r"), suffix=" R")),
            ("Heuristic-gap days negative Raw→Repair sum", _fmt(conclusion.get("mr13e_heuristic_gap_negative_target_loss_sum_r"), suffix=" R")),
        )),
        "限制：Exact oracle只用frozen score與既有K/R0/cash-capped reservation契約。Future Target只在replay後用來歸因實際loss落在哪類日期；不得回饋selector。若Exact global best等於既有after-ascent，該日可排除greedy／multi-swap search deficiency；若不等則直接證明存在heuristic search gap。",
    )) + "\n"
    md_path.write_text(md, encoding="utf-8")

    latest = output_root / "latest"
    if latest.exists():
        if latest.is_dir():
            shutil.rmtree(latest)
        else:
            latest.unlink()
    shutil.copytree(run_dir, latest)
    if not quiet:
        print(md, end="")
        print_artifact_paths(
            (("Audit Markdown", md_path), ("Audit JSON", json_path), ("最新Audit", latest)),
            project_root=root,
        )
    return payload
