"""Read-only selector stage translation audit for MR-13E vs continuous-ranker comparators.

Consumes completed Strategy Compare orderable candidates, canonical selector-stage
membership trace, and entry execution sidecars.  Future target is joined only after
replay.  The audit does not train, score, replay, or alter strategy decisions.
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
    annotate_orderable as _annotate_orderable,
    json_native as _json_native,
    load_common_daily_target as _load_common_daily_target,
    read_csv as _read_csv,
    resolve_phase_source as _resolve_phase_source,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_RESULT_SCHEMA_VERSION = 1
STAGES = ("raw_top_n", "minimum_repair_seed", "feasible_ascent_final", "entry_action", "actual_fill")
TRANSITIONS = (
    ("raw_to_repair", "raw_top_n", "minimum_repair_seed"),
    ("repair_to_ascent", "minimum_repair_seed", "feasible_ascent_final"),
    ("ascent_to_action", "feasible_ascent_final", "entry_action"),
    ("action_to_fill", "entry_action", "actual_fill"),
)


def _required_arm_paths(arm) -> tuple[Path, ...]:
    return (
        arm.orderable_path,
        arm.capacity_path,
        arm.pair_dir / f"{arm.prefix}_execution.csv",
        arm.pair_dir / f"{arm.prefix}_selector_trace.csv",
        arm.pair_dir / "strategy_comparison.json",
    )


def collect_selector_stage_translation_status(
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
                            f"{source['phase_id']} {arm.arm_id}缺少selector-stage read-only工件: "
                            + project_relative_display_path(path, project_root=root)
                        )
        return {
            "audit_id": definition.audit_id,
            "status": "READY",
            "reason": "",
            "source": {
                "display": "12B / 13A / 13E｜Raw→Repair→Ascent→Action→Fill",
                "selection_run": project_relative_display_path(sources["selection_pit"]["run_dir"], project_root=root),
                "forward_run": project_relative_display_path(sources["forward_oos"]["run_dir"], project_root=root),
            },
        }
    except (FileNotFoundError, ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        return {
            "audit_id": definition.audit_id,
            "status": "BLOCKED",
            "reason": str(exc),
            "source": {"display": "12B / 13A / 13E｜Raw→Repair→Ascent→Action→Fill"},
        }


def _normalized_key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(frame).copy()
    out["ticker"] = out["ticker"].fillna("").astype(str).str.strip()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="raise").dt.strftime("%Y-%m-%d")
    out["signal_date"] = pd.to_datetime(out["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    return out


def _orderable_lookup(orderable: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "ticker", "trade_date", "signal_date", "model_score", "common_target_raw_r",
        "score_available", "target_available",
    ]
    return orderable.drop_duplicates(["ticker", "trade_date", "signal_date"], keep="first")[cols]


def _trace_stage_rows(orderable: pd.DataFrame, trace: pd.DataFrame) -> pd.DataFrame:
    required = {"stage", "stage_rank", "ticker", "trade_date", "signal_date"}
    missing = sorted(required - set(trace.columns))
    if missing:
        raise ValueError(f"selector trace缺少欄位: {missing}")
    rows = _normalized_key_frame(trace)
    unknown = sorted(set(rows["stage"].astype(str)) - {"raw_top_n", "minimum_repair_seed", "feasible_ascent_final"})
    if unknown:
        raise ValueError(f"selector trace含未知stage: {unknown[:5]}")
    rows["stage_rank"] = pd.to_numeric(rows["stage_rank"], errors="raise").astype(int)
    rows = rows.merge(
        _orderable_lookup(orderable),
        on=["ticker", "trade_date", "signal_date"],
        how="left",
        validate="many_to_one",
    )
    return rows


def _execution_stage_rows(orderable: pd.DataFrame, execution: pd.DataFrame) -> pd.DataFrame:
    required = {"ticker", "trade_date", "signal_date", "chosen_qty", "filled_qty", "entry_filled"}
    missing = sorted(required - set(execution.columns))
    if missing:
        raise ValueError(f"execution sidecar缺少欄位: {missing}")
    ex = _normalized_key_frame(execution)
    ex["chosen_qty"] = pd.to_numeric(ex["chosen_qty"], errors="coerce").fillna(0)
    ex["filled_qty"] = pd.to_numeric(ex["filled_qty"], errors="coerce").fillna(0)
    ex = ex.merge(
        _orderable_lookup(orderable),
        on=["ticker", "trade_date", "signal_date"],
        how="left",
        validate="many_to_one",
    )
    pieces: list[pd.DataFrame] = []
    action = ex[ex["chosen_qty"] > 0].copy()
    if not action.empty:
        action["stage"] = "entry_action"
        action["stage_rank"] = action.groupby("trade_date").cumcount() + 1
        pieces.append(action)
    # Actual fill membership is defined by positive filled quantity.  Do not
    # coerce CSV string booleans (e.g. "False") through astype(bool).
    fill_mask = ex["filled_qty"] > 0
    fill = ex[fill_mask].copy()
    if not fill.empty:
        fill["stage"] = "actual_fill"
        fill["stage_rank"] = fill.groupby("trade_date").cumcount() + 1
        pieces.append(fill)
    if not pieces:
        return pd.DataFrame(columns=[*ex.columns, "stage", "stage_rank"])
    return pd.concat(pieces, ignore_index=True)


def _stage_daily_summary(stage_rows: pd.DataFrame) -> pd.DataFrame:
    valid = stage_rows[
        stage_rows["target_available"].fillna(False)
        & pd.to_numeric(stage_rows["common_target_raw_r"], errors="coerce").notna()
    ].copy()
    valid["model_score"] = pd.to_numeric(valid["model_score"], errors="coerce")
    valid["common_target_raw_r"] = pd.to_numeric(valid["common_target_raw_r"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for (trade_date, stage), day in valid.groupby(["trade_date", "stage"], sort=True):
        day = day.drop_duplicates(["ticker", "signal_date"], keep="first")
        keys = tuple(sorted(zip(day["ticker"].astype(str), day["signal_date"].astype(str))))
        rows.append({
            "trade_date": trade_date,
            "stage": stage,
            "count": int(len(day)),
            "target_mean_r": float(day["common_target_raw_r"].mean()) if len(day) else None,
            "score_mean": float(day["model_score"].dropna().mean()) if day["model_score"].notna().any() else None,
            "membership_keys": keys,
        })
    return pd.DataFrame(rows)


def _transition_rows(stage_daily: pd.DataFrame) -> pd.DataFrame:
    if stage_daily.empty:
        return pd.DataFrame()
    by_key = {(r.trade_date, r.stage): r for r in stage_daily.itertuples(index=False)}
    dates = sorted(stage_daily["trade_date"].unique())
    rows: list[dict[str, Any]] = []
    for trade_date in dates:
        for transition, before_stage, after_stage in TRANSITIONS:
            before = by_key.get((trade_date, before_stage))
            after = by_key.get((trade_date, after_stage))
            if before is None or after is None or before.count <= 0 or after.count <= 0:
                continue
            before_keys = set(before.membership_keys)
            after_keys = set(after.membership_keys)
            denom = min(len(before_keys), len(after_keys))
            rows.append({
                "trade_date": trade_date,
                "transition": transition,
                "before_count": int(before.count),
                "after_count": int(after.count),
                "membership_overlap": (float(len(before_keys & after_keys) / denom) if denom else None),
                "target_delta_r": float(after.target_mean_r - before.target_mean_r),
                "score_delta": (
                    None if before.score_mean is None or after.score_mean is None
                    else float(after.score_mean - before.score_mean)
                ),
            })
    return pd.DataFrame(rows)


def _repair_dates(trace: pd.DataFrame) -> set[str]:
    if "repair_steps" not in trace.columns:
        return set()
    frame = _normalized_key_frame(trace)
    steps = pd.to_numeric(frame["repair_steps"], errors="coerce").fillna(0)
    return set(frame.loc[steps > 0, "trade_date"].astype(str))


def _mean(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.mean()) if len(values) else None


def _aggregate_stage_daily(stage_daily: pd.DataFrame, dates: set[str] | None = None) -> dict[str, Any]:
    frame = stage_daily.copy()
    if dates is not None:
        frame = frame[frame["trade_date"].isin(dates)].copy()
    out: dict[str, Any] = {}
    for stage in STAGES:
        part = frame[frame["stage"] == stage]
        out[stage] = {
            "days": int(len(part)),
            "count_mean": _mean(part, "count"),
            "target_mean_r": _mean(part, "target_mean_r"),
            "score_mean": _mean(part, "score_mean"),
        }
    return out


def _aggregate_transitions(transitions: pd.DataFrame, dates: set[str] | None = None) -> dict[str, Any]:
    frame = transitions.copy()
    if dates is not None:
        frame = frame[frame["trade_date"].isin(dates)].copy()
    out: dict[str, Any] = {}
    for transition, _before, _after in TRANSITIONS:
        part = frame[frame["transition"] == transition]
        out[transition] = {
            "days": int(len(part)),
            "target_delta_r": _mean(part, "target_delta_r"),
            "score_delta": _mean(part, "score_delta"),
            "membership_overlap": _mean(part, "membership_overlap"),
        }
    return out


def _arm_payload(arm, *, target_lookup: pd.DataFrame, target_calendar: np.ndarray) -> dict[str, Any]:
    orderable = _annotate_orderable(
        _read_csv(arm.orderable_path),
        target_lookup=target_lookup,
        target_calendar=target_calendar,
    )
    trace = _read_csv(arm.pair_dir / f"{arm.prefix}_selector_trace.csv")
    execution = _read_csv(arm.pair_dir / f"{arm.prefix}_execution.csv")
    trace_rows = _trace_stage_rows(orderable, trace)
    execution_rows = _execution_stage_rows(orderable, execution)
    stage_rows = pd.concat([trace_rows, execution_rows], ignore_index=True, sort=False)
    stage_daily = _stage_daily_summary(stage_rows)
    transitions = _transition_rows(stage_daily)
    repair_dates = _repair_dates(trace)
    return {
        "arm_id": arm.arm_id,
        "display_name": str(arm.arm.get("name") or arm.arm_id),
        "summary": dict(arm.summary),
        "trace_rows": int(len(trace_rows)),
        "repair_days": int(len(repair_dates)),
        "all_days": {
            "stages": _aggregate_stage_daily(stage_daily),
            "transitions": _aggregate_transitions(transitions),
        },
        "repair_days_only": {
            "stages": _aggregate_stage_daily(stage_daily, repair_dates),
            "transitions": _aggregate_transitions(transitions, repair_dates),
        },
    }


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _directional_conclusion(payload: dict[str, Any]) -> dict[str, Any]:
    forward = payload["phases"]["forward_oos"]["arms"]
    ids = list(forward)
    if len(ids) < 3:
        return {"classification": "INSUFFICIENT_COMPARATORS"}
    mr12b, mr13a, mr13e = ids[0], ids[1], ids[2]
    extra: dict[str, float | None] = {}
    for transition, _before, _after in TRANSITIONS:
        e = _finite(forward[mr13e]["repair_days_only"]["transitions"][transition].get("target_delta_r"))
        b = _finite(forward[mr12b]["repair_days_only"]["transitions"][transition].get("target_delta_r"))
        extra[transition] = None if e is None or b is None else float(e - b)
    finite_extra = {k: v for k, v in extra.items() if v is not None}
    if not finite_extra:
        classification = "INSUFFICIENT_STAGE_TRANSLATION"
        dominant = None
    else:
        dominant = min(finite_extra, key=finite_extra.get)
        classification = (
            "NO_FORWARD_EXTRA_TRANSLATION_LOSS"
            if all(v >= 0.0 for v in finite_extra.values())
            else f"DOMINANT_EXTRA_LOSS_AT_{dominant.upper()}"
        )
    return {
        "classification": classification,
        "dominant_forward_extra_loss_transition": dominant,
        "mr13e_minus_mr12b_forward_repair_transition_target_delta_r": extra,
        "mr13a_arm_id": mr13a,
    }


def _fmt(value: Any, digits: int = 4, suffix: str = "") -> str:
    value = _finite(value)
    return "-" if value is None else f"{value:.{digits}f}{suffix}"


def _render_phase(phase_id: str, phase: dict[str, Any]) -> str:
    title = "Selection PIT" if phase_id == "selection_pit" else "Forward-OOS"
    stage_rows = []
    transition_rows = []
    for arm in phase["arms"].values():
        stages = arm["all_days"]["stages"]
        repair = arm["repair_days_only"]["transitions"]
        stage_rows.append((
            arm["arm_id"], arm["display_name"], arm["repair_days"],
            _fmt(stages["raw_top_n"]["target_mean_r"], suffix=" R"),
            _fmt(stages["minimum_repair_seed"]["target_mean_r"], suffix=" R"),
            _fmt(stages["feasible_ascent_final"]["target_mean_r"], suffix=" R"),
            _fmt(stages["entry_action"]["target_mean_r"], suffix=" R"),
            _fmt(stages["actual_fill"]["target_mean_r"], suffix=" R"),
        ))
        transition_rows.append((
            arm["arm_id"], arm["display_name"],
            _fmt(repair["raw_to_repair"]["target_delta_r"], suffix=" R"),
            _fmt(repair["repair_to_ascent"]["target_delta_r"], suffix=" R"),
            _fmt(repair["ascent_to_action"]["target_delta_r"], suffix=" R"),
            _fmt(repair["action_to_fill"]["target_delta_r"], suffix=" R"),
            _fmt(repair["raw_to_repair"]["membership_overlap"]),
            _fmt(repair["repair_to_ascent"]["membership_overlap"]),
            _fmt(repair["ascent_to_action"]["membership_overlap"]),
        ))
    return "\n\n".join((
        render_section(f"{title}｜stage target mean（全部trace日）"),
        render_table(
            ("Arm", "Model", "Repair days", "Raw Top-N", "Repair seed", "After ascent", "Entry action", "Actual fill"),
            stage_rows,
        ),
        render_section(f"{title}｜repair-day逐層Target變化（after − before）"),
        render_table(
            ("Arm", "Model", "Raw→Repair", "Repair→Ascent", "Ascent→Action", "Action→Fill", "Raw/Repair overlap", "Repair/Ascent overlap", "Ascent/Action overlap"),
            transition_rows,
        ),
    ))


def run_selector_stage_translation_audit(
    definition: AuditDefinition,
    *,
    project_root: Path = PROJECT_ROOT,
    quiet: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    status = collect_selector_stage_translation_status(definition, project_root=root)
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
    phases: dict[str, Any] = {}
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
        "future_target_join_stage": "post_replay_only",
        "future_target_used_for_runtime": False,
        "stages": list(STAGES),
        "phases": phases,
    }
    payload["conclusion"] = _directional_conclusion(payload)

    stamp = get_taipei_now().strftime("%Y%m%d_%H%M%S")
    output_root = (root / AUDIT_OUTPUT_ROOT / definition.output_subdir).resolve()
    run_dir = output_root / "runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / "audit.json"
    md_path = run_dir / "audit.md"
    json_path.write_text(json.dumps(_json_native(payload), ensure_ascii=False, indent=2), encoding="utf-8")

    conclusion = payload["conclusion"]
    extra = dict(conclusion.get("mr13e_minus_mr12b_forward_repair_transition_target_delta_r") or {})
    md = "\n\n".join((
        render_title("MR-13E Selector Stage Translation Audit"),
        render_key_values((
            ("Audit", definition.audit_id),
            ("契約", "read-only completed Strategy Compare trace/execution sidecars；不train、不score、不replay"),
            ("比較", "MR-12B / MR-13A / MR-13E"),
            ("Future Target", "daily_opportunity_no_time_r_v1；僅post-replay join"),
            ("Stage", "Raw Top-N → Minimum repair → Feasible ascent → Entry action → Actual fill"),
        )),
        _render_phase("selection_pit", phases["selection_pit"]),
        _render_phase("forward_oos", phases["forward_oos"]),
        render_section("判定"),
        render_key_values((
            ("Classification", conclusion.get("classification")),
            ("Dominant transition", conclusion.get("dominant_forward_extra_loss_transition")),
            ("13E−12B Raw→Repair", _fmt(extra.get("raw_to_repair"), suffix=" R")),
            ("13E−12B Repair→Ascent", _fmt(extra.get("repair_to_ascent"), suffix=" R")),
            ("13E−12B Ascent→Action", _fmt(extra.get("ascent_to_action"), suffix=" R")),
            ("13E−12B Action→Fill", _fmt(extra.get("action_to_fill"), suffix=" R")),
        )),
        "限制：stage Target只供post-replay attribution；negative transition代表該轉換後平均Future Target下降。Classification只找Forward repair days中13E相對12B最負的stage差，不建立加權總分或新threshold。",
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
        print(md)
        print_artifact_paths(
            (
                ("Audit Markdown", md_path),
                ("Audit JSON", json_path),
                ("最新Audit", latest),
            ),
            project_root=root,
        )
    return payload
