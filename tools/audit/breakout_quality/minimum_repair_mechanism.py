"""Read-only scalable minimum-repair mechanism attribution for MR-13E.

Consumes completed Strategy Compare selector-stage and repair-search certificate
sidecars. Production minimum-repair exhaustively enumerates every single swap at
each accepted step; one-step repairs therefore have an exact minimum-distance=1
certificate. Multi-step greedy paths are reported as unresolved rather than being
misrepresented as global exact search. Future target is joined only after replay.
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
AUDIT_RESULT_SCHEMA_VERSION = 2
ONE_STEP_CLASS = "EXACT_ONE_SWAP_RESOURCE_CONSTRAINT"
MULTI_STEP_CLASS = "MULTI_STEP_GREEDY_PATH_UNRESOLVED"
FALLBACK_CLASS = "BASELINE_FALLBACK_AFTER_GREEDY_REPAIR"
CLASSES = (ONE_STEP_CLASS, MULTI_STEP_CLASS, FALLBACK_CLASS)


def _required_arm_paths(arm) -> tuple[Path, ...]:
    return (
        arm.pair_dir / f"{arm.prefix}_selector_trace.csv",
        arm.pair_dir / f"{arm.prefix}_repair_search_certificate.csv",
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
                "display": "12B / 13A / 13E｜minimum-repair scalable single-swap certificate",
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
            "source": {"display": "12B / 13A / 13E｜minimum-repair scalable single-swap certificate"},
        }


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", "", "nan", "none"}:
        return False
    raise ValueError(f"無法解析boolean值: {value!r}")


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
        raise ValueError(f"minimum-repair rows缺少欄位: {missing}")
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
    required = {"stage", "ticker", "trade_date"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"selector trace缺少欄位: {missing}")
    rows = rows[rows["stage"].astype(str).isin({
        "raw_top_n", "minimum_repair_seed", "feasible_ascent_final"
    })].copy()
    rows["trade_date"] = pd.to_datetime(rows["trade_date"], errors="raise").dt.strftime("%Y-%m-%d")
    rows = rows[rows["trade_date"].isin(repair_dates)].copy()
    rows = _annotate_target(rows, target_lookup=target_lookup, target_calendar=target_calendar)
    rows = rows[rows["target_available"]].copy()
    if rows.empty:
        return pd.DataFrame(columns=["trade_date", "stage", "target_mean_r"])
    return (
        rows.groupby(["trade_date", "stage"], as_index=False)["common_target_raw_r"]
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
        return pd.DataFrame()
    rows = _annotate_target(rows, target_lookup=target_lookup, target_calendar=target_calendar)
    rows["breakout_quality_score"] = pd.to_numeric(rows["breakout_quality_score"], errors="coerce")
    result = []
    for (trade_date, repair_step), part in rows.groupby(["trade_date", "repair_step"], sort=True):
        out = part[part["repair_role"].astype(str) == "out"]
        incoming = part[part["repair_role"].astype(str) == "in"]
        if len(out) != 1 or len(incoming) != 1:
            continue
        out_row, in_row = out.iloc[0], incoming.iloc[0]
        out_target, in_target = _finite(out_row.get("common_target_raw_r")), _finite(in_row.get("common_target_raw_r"))
        out_score, in_score = _finite(out_row.get("breakout_quality_score")), _finite(in_row.get("breakout_quality_score"))
        result.append({
            "trade_date": str(trade_date),
            "repair_step": int(repair_step),
            "target_delta_r": None if out_target is None or in_target is None else float(in_target - out_target),
            "score_delta": None if out_score is None or in_score is None else float(in_score - out_score),
            "reserve_deficit_delta_milli": int(in_row.get("after_reserve_deficit_milli", 0) or 0) - int(in_row.get("before_reserve_deficit_milli", 0) or 0),
            "count_deficit_delta": int(in_row.get("after_count_deficit", 0) or 0) - int(in_row.get("before_count_deficit", 0) or 0),
            "evaluated_swap_count": int(in_row.get("evaluated_swap_count", 0) or 0),
            "progress_swap_count": int(in_row.get("progress_swap_count", 0) or 0),
            "feasible_swap_count": int(in_row.get("feasible_swap_count", 0) or 0),
            "after_feasible": _as_bool(in_row.get("after_feasible", False)),
        })
    return pd.DataFrame(result)


def _k_distribution(summary: pd.DataFrame) -> dict[str, int]:
    values = pd.to_numeric(summary["target_count"], errors="coerce").dropna().astype(int)
    return {str(int(k)): int(v) for k, v in values.value_counts().sort_index().items()}


def _class_contribution(daily: pd.DataFrame, class_name: str, total_days: int) -> float:
    if total_days <= 0:
        return 0.0
    part = pd.to_numeric(
        daily.loc[daily["classification"] == class_name, "raw_to_repair_target_delta_r"],
        errors="coerce",
    ).dropna()
    return float(part.sum() / total_days) if len(part) else 0.0


def _arm_payload(arm, *, target_lookup: pd.DataFrame, target_calendar: np.ndarray) -> dict[str, Any]:
    mechanism = _read_csv(arm.pair_dir / f"{arm.prefix}_repair_search_certificate.csv")
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
        raise ValueError(f"{arm.arm_id} repair mechanism certificate不可用: {invalid}")
    summary["trade_date"] = pd.to_datetime(summary["trade_date"], errors="raise").dt.strftime("%Y-%m-%d")
    unknown = sorted(set(summary["classification"].fillna("").astype(str)) - set(CLASSES))
    if unknown:
        raise ValueError(f"{arm.arm_id} repair mechanism含未知classification: {unknown}")
    repair_dates = set(summary["trade_date"].astype(str))
    stage_daily = _stage_daily_target(
        trace,
        target_lookup=target_lookup,
        target_calendar=target_calendar,
        repair_dates=repair_dates,
    )
    swaps = _swap_daily(mechanism, target_lookup=target_lookup, target_calendar=target_calendar)
    stage_pivot = stage_daily.pivot(index="trade_date", columns="stage", values="target_mean_r") if not stage_daily.empty else pd.DataFrame()
    daily = summary.set_index("trade_date").copy()
    for column in ("raw_top_n", "minimum_repair_seed", "feasible_ascent_final"):
        daily[f"target_{column}"] = stage_pivot[column] if column in stage_pivot else np.nan
    daily["raw_to_repair_target_delta_r"] = daily["target_minimum_repair_seed"] - daily["target_raw_top_n"]
    daily = daily.reset_index()

    class_counts = {key: int((summary["classification"].astype(str) == key).sum()) for key in CLASSES}
    contributions = {
        key: _class_contribution(daily, key, len(summary)) for key in CLASSES
    }
    class_means = {
        key: _mean(daily.loc[daily["classification"].astype(str) == key, "raw_to_repair_target_delta_r"])
        for key in CLASSES
    }
    raw_count_deficit = pd.to_numeric(summary["raw_count_deficit"], errors="coerce").fillna(0).astype(int)
    raw_reserve_deficit = pd.to_numeric(summary["raw_reserve_deficit_milli"], errors="coerce").fillna(0).astype(np.int64)
    failure_counts = {
        "count_only": int(((raw_count_deficit > 0) & (raw_reserve_deficit == 0)).sum()),
        "reserve_only": int(((raw_count_deficit == 0) & (raw_reserve_deficit > 0)).sum()),
        "both": int(((raw_count_deficit > 0) & (raw_reserve_deficit > 0)).sum()),
        "none": int(((raw_count_deficit == 0) & (raw_reserve_deficit == 0)).sum()),
    }
    last_steps = pd.DataFrame()
    if not swaps.empty:
        last_steps = swaps.sort_values(["trade_date", "repair_step"]).groupby("trade_date", as_index=False).tail(1)

    return {
        "arm_id": arm.arm_id,
        "display_name": str(arm.arm.get("name") or arm.arm_id),
        "status": "AVAILABLE",
        "repair_days": int(len(summary)),
        "k_distribution": _k_distribution(summary),
        "raw_failure_counts": failure_counts,
        "classification_counts": class_counts,
        "repair_steps_mean": _mean(summary["actual_repair_steps"]),
        "repair_replacement_distance_mean": _mean(summary["actual_repair_replacement_distance"]),
        "raw_reserve_deficit_mean": _mean(summary["raw_reserve_deficit_milli"]),
        "raw_score_sum_mean": _mean(summary["raw_score_sum"]),
        "repair_seed_score_sum_mean": _mean(summary["repair_seed_score_sum"]),
        "repair_seed_score_loss_from_raw_mean": _mean(summary["repair_seed_score_loss_from_raw"]),
        "final_score_change_from_repair_mean": _mean(summary["final_score_change_from_repair"]),
        "repair_search_evaluations_mean": _mean(summary["repair_search_evaluations"]),
        "ascent_search_evaluations_mean": _mean(summary["ascent_search_evaluations"]),
        "ascent_local_optimum_days": int(summary["ascent_local_optimum"].map(_as_bool).sum()),
        "target_stage_means": {
            "raw_top_n": _mean(daily["target_raw_top_n"]),
            "repair_seed": _mean(daily["target_minimum_repair_seed"]),
            "after_ascent": _mean(daily["target_feasible_ascent_final"]),
            "raw_to_repair": _mean(daily["raw_to_repair_target_delta_r"]),
        },
        "class_target_delta_mean": class_means,
        "class_target_delta_contribution": contributions,
        "repair_swap": {
            "steps": int(len(swaps)),
            "target_delta_r_mean": _mean(swaps.get("target_delta_r", pd.Series(dtype=float))),
            "score_delta_mean": _mean(swaps.get("score_delta", pd.Series(dtype=float))),
            "evaluated_swap_count_mean": _mean(swaps.get("evaluated_swap_count", pd.Series(dtype=float))),
            "progress_swap_count_mean": _mean(swaps.get("progress_swap_count", pd.Series(dtype=float))),
            "last_step_feasible_swap_count_mean": _mean(last_steps.get("feasible_swap_count", pd.Series(dtype=float))),
        },
    }


def _conclusion(payload: dict[str, Any]) -> dict[str, Any]:
    forward = payload["phases"]["forward_oos"]["arms"]
    ids = list(forward)
    if len(ids) < 3:
        return {"classification": "INSUFFICIENT_COMPARATORS"}
    mr12b, mr13a, mr13e = ids[0], ids[1], ids[2]
    b, e = forward[mr12b], forward[mr13e]
    if b.get("status") != "AVAILABLE" or e.get("status") != "AVAILABLE":
        return {"classification": "INSUFFICIENT_REPAIR_DAYS"}
    b_contrib = dict(b["class_target_delta_contribution"])
    e_contrib = dict(e["class_target_delta_contribution"])
    extra = {key: float(e_contrib.get(key, 0.0) - b_contrib.get(key, 0.0)) for key in CLASSES}
    dominant = min(extra, key=lambda key: (extra[key], key))
    if extra[dominant] >= 0:
        classification = "NO_FORWARD_EXTRA_RAW_TO_REPAIR_LOSS"
    elif dominant == ONE_STEP_CLASS:
        classification = "EXACT_ONE_SWAP_RESOURCE_CONSTRAINT_DOMINATES"
    elif dominant == MULTI_STEP_CLASS:
        classification = "MULTI_STEP_GREEDY_PATH_DOMINATES_UNRESOLVED"
    else:
        classification = "BASELINE_FALLBACK_DOMINATES"
    return {
        "classification": classification,
        "dominant_class": dominant,
        "mr13e_arm_id": mr13e,
        "mr12b_arm_id": mr12b,
        "mr13a_arm_id": mr13a,
        "mr13e_minus_mr12b_class_contribution_r": extra,
        "one_step_exact_scope": "minimum replacement distance=1 / best feasible one-swap frozen-score basket",
        "multi_step_scope": "production greedy path exhaustive per accepted single-swap step; global multi-swap optimum unresolved",
    }


def _fmt(value: Any, digits: int = 4, suffix: str = "") -> str:
    value = _finite(value)
    return "-" if value is None else f"{value:.{digits}f}{suffix}"


def _render_phase(phase_id: str, phase: dict[str, Any]) -> str:
    title = "Selection PIT" if phase_id == "selection_pit" else "Forward-OOS"
    mechanism_rows, target_rows, search_rows = [], [], []
    for arm in phase["arms"].values():
        if arm.get("status") != "AVAILABLE":
            continue
        failures, classes = arm["raw_failure_counts"], arm["classification_counts"]
        mechanism_rows.append((
            arm["arm_id"], arm["display_name"], arm["repair_days"],
            failures["count_only"], failures["reserve_only"], failures["both"],
            classes.get(ONE_STEP_CLASS, 0), classes.get(MULTI_STEP_CLASS, 0), classes.get(FALLBACK_CLASS, 0),
            _fmt(arm["repair_steps_mean"]), _fmt(arm["repair_seed_score_loss_from_raw_mean"]),
            arm["ascent_local_optimum_days"], str(arm["k_distribution"]),
        ))
        t, cm = arm["target_stage_means"], arm["class_target_delta_mean"]
        target_rows.append((
            arm["arm_id"], arm["display_name"],
            _fmt(t["raw_top_n"], suffix=" R"), _fmt(t["repair_seed"], suffix=" R"),
            _fmt(t["after_ascent"], suffix=" R"), _fmt(t["raw_to_repair"], suffix=" R"),
            _fmt(cm.get(ONE_STEP_CLASS), suffix=" R"),
            _fmt(cm.get(MULTI_STEP_CLASS), suffix=" R"),
            _fmt(cm.get(FALLBACK_CLASS), suffix=" R"),
        ))
        s = arm["repair_swap"]
        search_rows.append((
            arm["arm_id"], arm["display_name"], s["steps"],
            _fmt(s["evaluated_swap_count_mean"]), _fmt(s["progress_swap_count_mean"]),
            _fmt(s["last_step_feasible_swap_count_mean"]),
            _fmt(s["score_delta_mean"]), _fmt(s["target_delta_r_mean"], suffix=" R"),
        ))
    return "\n\n".join((
        render_section(f"{title}｜minimum-repair search certificate"),
        render_table(
            ("Arm", "Model", "Repair days", "Count-only", "Reserve-only", "Both", "Exact 1-swap", "Multi-step", "Fallback", "Repair steps", "Seed score Δ", "Ascent local-opt days", "K distribution"),
            mechanism_rows,
        ),
        render_section(f"{title}｜Future Target（post-replay attribution only）"),
        render_table(
            ("Arm", "Model", "Raw Top-N", "Repair seed", "After ascent", "Raw→Repair", "Exact 1-swap Δ", "Multi-step Δ", "Fallback Δ"),
            target_rows,
        ),
        render_section(f"{title}｜production exhaustive single-swap search"),
        render_table(
            ("Arm", "Model", "Accepted steps", "Eval/step", "Progress/step", "Feasible on final step", "In−Out Score", "In−Out Target"),
            search_rows,
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
            arm_id: _arm_payload(arm, target_lookup=target_lookup, target_calendar=target_calendar)
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
        "search_contract": "production exhaustive single-swap repair trace; exact only for one-step minimum repair",
        "global_exact_oracle": False,
        "future_target_join_stage": "post_replay_only",
        "future_target_used_for_selection": False,
        "phases": phases,
    }
    payload["conclusion"] = _conclusion(payload)

    output_root = root / AUDIT_OUTPUT_ROOT / definition.output_subdir
    run_id = get_taipei_now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    json_path = run_dir / "audit.json"
    md_path = run_dir / "audit.md"
    json_path.write_text(json.dumps(_json_native(payload), ensure_ascii=False, indent=2), encoding="utf-8")

    conclusion = payload["conclusion"]
    md = "\n\n".join((
        render_title("MR-13E Minimum-repair Mechanism Audit"),
        render_key_values((
            ("Audit", definition.audit_id),
            ("契約", "read-only completed Strategy Compare repair-search certificate；不train、不score、不replay"),
            ("比較", "MR-12B / MR-13A / MR-13E"),
            ("Exact範圍", "只在raw不合法且1次swap即可合法時：完整枚舉所有single swaps"),
            ("Multi-step", "逐步single-swap exhaustive；不宣稱global multi-swap optimum"),
            ("Future Target", "daily_opportunity_no_time_r_v1；僅post-replay join"),
        )),
        _render_phase("selection_pit", phases["selection_pit"]),
        _render_phase("forward_oos", phases["forward_oos"]),
        render_section("判定"),
        render_key_values((
            ("Classification", conclusion.get("classification")),
            ("Dominant class", conclusion.get("dominant_class")),
            ("13E−12B Exact 1-swap contribution", _fmt((conclusion.get("mr13e_minus_mr12b_class_contribution_r") or {}).get(ONE_STEP_CLASS), suffix=" R")),
            ("13E−12B Multi-step contribution", _fmt((conclusion.get("mr13e_minus_mr12b_class_contribution_r") or {}).get(MULTI_STEP_CLASS), suffix=" R")),
            ("13E−12B Fallback contribution", _fmt((conclusion.get("mr13e_minus_mr12b_class_contribution_r") or {}).get(FALLBACK_CLASS), suffix=" R")),
        )),
        "限制：1-step repair因production完整枚舉全部single swaps，可exact證明minimum replacement distance=1且取該distance最高frozen-score feasible basket；multi-step只證明每個accepted step是當下全部deficit-improving single swaps中的最高frozen-score move，及feasible-ascent最後為1-swap local optimum，不宣稱global multi-swap optimum。Future Target只供post-replay attribution。",
    )) + "\n"
    md_path.write_text(md, encoding="utf-8")

    latest = output_root / "latest"
    if latest.exists():
        shutil.rmtree(latest)
    shutil.copytree(run_dir, latest)
    payload["artifacts"] = {
        "markdown": project_relative_display_path(md_path, project_root=root),
        "json": project_relative_display_path(json_path, project_root=root),
        "latest": project_relative_display_path(latest, project_root=root),
    }
    if not quiet:
        print(md, end="")
        print_artifact_paths(
            (
                ("Audit Markdown", md_path),
                ("Audit JSON", json_path),
                ("最新Audit", latest),
            ),
            project_root=root,
        )
    return payload
