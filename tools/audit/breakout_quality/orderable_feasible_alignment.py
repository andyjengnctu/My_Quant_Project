"""Read-only orderable / feasible-selection alignment audit for continuous rankers.

The audit consumes completed Strategy Compare replay artifacts only.  It never
trains, scores, or replays a strategy.  Formal score-ranking pairs must include
the canonical pre-market execution sidecar so chosen actions are not inferred
from post-fill selected-buys.  A common daily opportunity target is
joined after replay from the canonical daily-universal target provider, so the
future target cannot affect runtime ordering or execution.
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
from config.strategy_compare import get_strategy_comparison_settings
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
    information_date_map as _information_date_map,
    json_native as _json_native,
    load_common_daily_target as _load_common_daily_target,
    read_csv as _read_csv,
    resolve_phase_source as _resolve_phase_source,
)
from filters.breakout_quality.profile_ranker_data import load_profile_continuous_ranker_data
from services.breakout_quality.ranker_training import calculate_spearman
from tools.audit.sources.strategy_compare import (
    StrategyCompareArmArtifacts,
    resolve_arm_artifacts,
    resolve_strategy_compare_profile_run_selector,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_RESULT_SCHEMA_VERSION = 1


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _required_arm_paths(arm: StrategyCompareArmArtifacts) -> tuple[Path, ...]:
    return (
        arm.orderable_path,
        arm.capacity_path,
        arm.pair_dir / f"{arm.prefix}_execution.csv",
        arm.pair_dir / "strategy_comparison.json",
    )


def collect_orderable_feasible_alignment_status(
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
                            f"{source['phase_id']} {arm.arm_id}缺少read-only工件: "
                            + project_relative_display_path(path, project_root=root)
                        )
        target_summary = root / "outputs/filters/breakout_quality/breakout_quality_v1/dataset_summary.json"
        if not target_summary.is_file():
            raise FileNotFoundError("缺少canonical Breakout Quality Dataset summary")
        display = "12B / 13A / 13E｜Selection PIT + Forward-OOS orderable/feasible"
        return {
            "audit_id": definition.audit_id,
            "status": "READY",
            "reason": "",
            "source": {
                "display": display,
                "selection_run": project_relative_display_path(sources["selection_pit"]["run_dir"], project_root=root),
                "forward_run": project_relative_display_path(sources["forward_oos"]["run_dir"], project_root=root),
            },
        }
    except (FileNotFoundError, ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        return {
            "audit_id": definition.audit_id,
            "status": "BLOCKED",
            "reason": str(exc),
            "source": {"display": "12B / 13A / 13E｜Selection PIT + Forward-OOS orderable/feasible"},
        }


def _frame_series(frame: pd.DataFrame, name: str, default: Any) -> pd.Series:
    if name in frame.columns:
        return frame[name]
    return pd.Series(default, index=frame.index)


def _boolean_series(frame: pd.DataFrame, name: str, default: bool = False) -> pd.Series:
    raw = _frame_series(frame, name, default)
    if pd.api.types.is_bool_dtype(raw):
        return raw.fillna(default).astype(bool)
    normalized = raw.fillna(default).astype(str).str.strip().str.lower()
    truthy = {"1", "true", "t", "yes", "y", "on"}
    falsy = {"0", "false", "f", "no", "n", "off", "", "none", "nan"}
    unknown = sorted(set(normalized.unique()) - truthy - falsy)
    if unknown:
        raise ValueError(f"daily capacity布林欄{name}含未知值: {unknown[:5]}")
    return normalized.isin(truthy)


def _numeric_series(frame: pd.DataFrame, name: str, default: float) -> pd.Series:
    raw = _frame_series(frame, name, default)
    return pd.to_numeric(raw, errors="coerce")


def _capacity_frame(path: Path) -> pd.DataFrame:
    frame = _read_csv(path)
    if "Date" not in frame.columns:
        raise ValueError(f"daily capacity缺少Date: {path.name}")
    frame["trade_date"] = pd.to_datetime(frame["Date"], errors="raise").dt.strftime("%Y-%m-%d")
    frame["direct_feasible"] = _boolean_series(
        frame, "Resource_Aware_Direct_Score_Order_Feasible", False
    )
    frame["repair_steps"] = _numeric_series(
        frame, "Resource_Aware_Max_DL_Repair_Steps", 0.0
    ).fillna(0.0)
    frame["ascent_steps"] = _numeric_series(
        frame, "Resource_Aware_Max_DL_Feasible_Ascent_Steps", 0.0
    ).fillna(0.0)
    frame["order_limit"] = _numeric_series(
        frame, "Resource_Aware_Pre_Market_Order_Limit", np.nan
    )
    frame["baseline_selected"] = _numeric_series(
        frame, "Resource_Aware_Baseline_Selected", 0.0
    ).fillna(0.0)
    frame["score_gain"] = (
        _numeric_series(frame, "Resource_Aware_Score_Sum", 0.0).fillna(0.0)
        - _numeric_series(frame, "Resource_Aware_Baseline_Score_Sum", 0.0).fillna(0.0)
    )
    frame["reserved_delta_milli"] = (
        _numeric_series(frame, "Resource_Aware_Reserved_Milli", 0.0).fillna(0.0)
        - _numeric_series(frame, "Resource_Aware_Baseline_Reserved_Milli", 0.0).fillna(0.0)
    )
    return frame


def _pair_accuracy(scores: np.ndarray, targets: np.ndarray) -> float | None:
    n = len(scores)
    if n < 2:
        return None
    i, j = np.triu_indices(n, 1)
    sd = scores[i] - scores[j]
    td = targets[i] - targets[j]
    mask = np.isfinite(sd) & np.isfinite(td) & (sd != 0.0) & (td != 0.0)
    if not bool(mask.any()):
        return None
    return float(np.mean(np.sign(sd[mask]) == np.sign(td[mask])))


def _cross_boundary_accuracy(top_targets: np.ndarray, rest_targets: np.ndarray) -> float | None:
    if len(top_targets) == 0 or len(rest_targets) == 0:
        return None
    diff = top_targets[:, None] - rest_targets[None, :]
    mask = np.isfinite(diff) & (diff != 0.0)
    if not bool(mask.any()):
        return None
    return float(np.mean(diff[mask] > 0.0))


def _daily_ranking_rows(orderable: pd.DataFrame, capacity: pd.DataFrame) -> pd.DataFrame:
    cap = capacity.set_index("trade_date")
    rows: list[dict[str, Any]] = []
    valid = orderable[orderable["score_available"] & orderable["target_available"]].copy()
    for trade_date, day in valid.groupby("trade_date", sort=True):
        if trade_date not in cap.index:
            continue
        day = day.drop_duplicates(["ticker", "signal_date"], keep="first").copy()
        if len(day) < 2:
            continue
        c = cap.loc[trade_date]
        if isinstance(c, pd.DataFrame):
            c = c.iloc[0]
        scores = day["model_score"].to_numpy(dtype=np.float64)
        targets = day["common_target_raw_r"].to_numpy(dtype=np.float64)
        k_raw = _finite(c.get("order_limit"))
        if k_raw is None or k_raw <= 0:
            k_raw = _finite(c.get("baseline_selected"))
        k = min(len(day), max(0, int(k_raw or 0)))
        ordered = day.sort_values(["model_score", "ticker"], ascending=[False, True], kind="mergesort")
        top = ordered.head(k) if k else ordered.iloc[0:0]
        rest = ordered.iloc[k:]
        oracle = day.sort_values(["common_target_raw_r", "ticker"], ascending=[False, True], kind="mergesort").head(k) if k else day.iloc[0:0]
        top_keys = set(zip(top["ticker"], top["signal_date"]))
        oracle_keys = set(zip(oracle["ticker"], oracle["signal_date"]))
        rows.append({
            "trade_date": trade_date,
            "candidate_count": int(len(day)),
            "k": int(k),
            "daily_rho": calculate_spearman(scores, targets),
            "pair_accuracy": _pair_accuracy(scores, targets),
            "top_target_mean_r": float(top["common_target_raw_r"].mean()) if k else None,
            "universe_target_mean_r": float(day["common_target_raw_r"].mean()),
            "top_target_lift_r": float(top["common_target_raw_r"].mean() - day["common_target_raw_r"].mean()) if k else None,
            "oracle_retention": float(len(top_keys & oracle_keys) / k) if k else None,
            "top_vs_rest_pair_accuracy": _cross_boundary_accuracy(
                top["common_target_raw_r"].to_numpy(dtype=np.float64),
                rest["common_target_raw_r"].to_numpy(dtype=np.float64),
            ),
            "direct_feasible": bool(c.get("direct_feasible", False)),
            "repair": bool(float(c.get("repair_steps", 0.0)) > 0.0),
            "ascent": bool(float(c.get("ascent_steps", 0.0)) > 0.0),
            "score_gain": float(c.get("score_gain", 0.0) or 0.0),
            "reserved_delta_milli": float(c.get("reserved_delta_milli", 0.0) or 0.0),
        })
    return pd.DataFrame(rows)


def _aggregate_daily(rows: pd.DataFrame, mask: pd.Series | None = None) -> dict[str, Any]:
    frame = pd.DataFrame(rows).copy()
    if mask is not None:
        frame = frame.loc[pd.Series(mask, index=rows.index)].copy()
    def mean_col(name: str) -> float | None:
        if name not in frame or frame.empty:
            return None
        values = pd.to_numeric(frame[name], errors="coerce").dropna()
        return float(values.mean()) if len(values) else None
    return {
        "days": int(len(frame)),
        "avg_candidates": mean_col("candidate_count"),
        "avg_k": mean_col("k"),
        "daily_rho": mean_col("daily_rho"),
        "pair_accuracy": mean_col("pair_accuracy"),
        "top_target_lift_r": mean_col("top_target_lift_r"),
        "oracle_retention": mean_col("oracle_retention"),
        "top_vs_rest_pair_accuracy": mean_col("top_vs_rest_pair_accuracy"),
        "score_gain_mean": mean_col("score_gain"),
        "reserved_delta_milli_mean": mean_col("reserved_delta_milli"),
    }


def _execution_translation(
    orderable: pd.DataFrame,
    execution: pd.DataFrame,
    capacity: pd.DataFrame,
) -> pd.DataFrame:
    if execution.empty:
        return pd.DataFrame()
    required = {"ticker", "trade_date", "signal_date", "chosen_qty"}
    missing = sorted(required - set(execution.columns))
    if missing:
        raise ValueError(f"execution sidecar缺少欄位: {missing}")
    ex = execution.copy()
    ex["ticker"] = ex["ticker"].fillna("").astype(str).str.strip()
    ex["trade_date"] = pd.to_datetime(ex["trade_date"], errors="raise").dt.strftime("%Y-%m-%d")
    ex["signal_date"] = pd.to_datetime(ex["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    ex["chosen_qty"] = pd.to_numeric(ex["chosen_qty"], errors="coerce").fillna(0)
    ex = ex[ex["chosen_qty"] > 0].copy()
    lookup = orderable.drop_duplicates(["ticker", "trade_date", "signal_date"], keep="first")[[
        "ticker", "trade_date", "signal_date", "model_score", "common_target_raw_r", "target_available", "score_available"
    ]]
    ex = ex.merge(lookup, on=["ticker", "trade_date", "signal_date"], how="left", validate="many_to_one")
    cap = capacity.set_index("trade_date")
    rows: list[dict[str, Any]] = []
    for trade_date, action in ex.groupby("trade_date", sort=True):
        universe = orderable[(orderable["trade_date"] == trade_date) & orderable["score_available"] & orderable["target_available"]].copy()
        action = action[action["score_available"].fillna(False) & action["target_available"].fillna(False)].copy()
        if universe.empty or action.empty or trade_date not in cap.index:
            continue
        universe = universe.drop_duplicates(["ticker", "signal_date"], keep="first")
        action = action.drop_duplicates(["ticker", "signal_date"], keep="first")
        n = min(len(action), len(universe))
        raw = universe.sort_values(["model_score", "ticker"], ascending=[False, True], kind="mergesort").head(n)
        action = action.head(n)
        raw_keys = set(zip(raw["ticker"], raw["signal_date"]))
        action_keys = set(zip(action["ticker"], action["signal_date"]))
        c = cap.loc[trade_date]
        if isinstance(c, pd.DataFrame):
            c = c.iloc[0]
        rows.append({
            "trade_date": trade_date,
            "action_count": int(n),
            "raw_action_overlap": float(len(raw_keys & action_keys) / n) if n else None,
            "raw_top_target_mean_r": float(raw["common_target_raw_r"].mean()),
            "action_target_mean_r": float(action["common_target_raw_r"].mean()),
            "action_minus_raw_target_r": float(action["common_target_raw_r"].mean() - raw["common_target_raw_r"].mean()),
            "raw_top_score_mean": float(raw["model_score"].mean()),
            "action_score_mean": float(action["model_score"].mean()),
            "action_minus_raw_score": float(action["model_score"].mean() - raw["model_score"].mean()),
            "direct_feasible": bool(c.get("direct_feasible", False)),
            "repair": bool(float(c.get("repair_steps", 0.0)) > 0.0),
            "ascent": bool(float(c.get("ascent_steps", 0.0)) > 0.0),
        })
    return pd.DataFrame(rows)


def _aggregate_translation(rows: pd.DataFrame, mask: pd.Series | None = None) -> dict[str, Any]:
    frame = pd.DataFrame(rows).copy()
    if mask is not None:
        frame = frame.loc[pd.Series(mask, index=rows.index)].copy()
    def mean_col(name: str) -> float | None:
        if name not in frame or frame.empty:
            return None
        values = pd.to_numeric(frame[name], errors="coerce").dropna()
        return float(values.mean()) if len(values) else None
    return {
        "days": int(len(frame)),
        "avg_action_count": mean_col("action_count"),
        "raw_action_overlap": mean_col("raw_action_overlap"),
        "action_minus_raw_target_r": mean_col("action_minus_raw_target_r"),
        "action_minus_raw_score": mean_col("action_minus_raw_score"),
        "raw_top_target_mean_r": mean_col("raw_top_target_mean_r"),
        "action_target_mean_r": mean_col("action_target_mean_r"),
    }


def _arm_payload(
    arm: StrategyCompareArmArtifacts,
    *,
    target_lookup: pd.DataFrame,
    target_calendar: np.ndarray,
) -> dict[str, Any]:
    orderable = _annotate_orderable(
        _read_csv(arm.orderable_path),
        target_lookup=target_lookup,
        target_calendar=target_calendar,
    )
    capacity = _capacity_frame(arm.capacity_path)
    execution = _read_csv(arm.pair_dir / f"{arm.prefix}_execution.csv")
    ranking = _daily_ranking_rows(orderable, capacity)
    translation = _execution_translation(orderable, execution, capacity)
    stages = {
        "all": _aggregate_daily(ranking),
        "direct_feasible": _aggregate_daily(ranking, ranking.get("direct_feasible", pd.Series(False, index=ranking.index))),
        "repair": _aggregate_daily(ranking, ranking.get("repair", pd.Series(False, index=ranking.index))),
        "ascent": _aggregate_daily(ranking, ranking.get("ascent", pd.Series(False, index=ranking.index))),
    }
    translation_stages = {
        "all": _aggregate_translation(translation),
        "direct_feasible": _aggregate_translation(translation, translation.get("direct_feasible", pd.Series(False, index=translation.index))),
        "repair": _aggregate_translation(translation, translation.get("repair", pd.Series(False, index=translation.index))),
        "ascent": _aggregate_translation(translation, translation.get("ascent", pd.Series(False, index=translation.index))),
    }
    return {
        "arm_id": arm.arm_id,
        "display_name": str(arm.arm.get("name") or arm.arm_id),
        "summary": dict(arm.summary),
        "orderable_rows": int(len(orderable)),
        "score_coverage": float(orderable["score_available"].mean()) if len(orderable) else None,
        "target_coverage": float(orderable["target_available"].mean()) if len(orderable) else None,
        "ranking": stages,
        "translation": translation_stages,
    }


def _fmt(value: Any, *, digits: int = 4, suffix: str = "") -> str:
    number = _finite(value)
    if number is None:
        return "-"
    return f"{number:.{digits}f}{suffix}"


def _render_phase(phase_id: str, phase: dict[str, Any]) -> str:
    rows = []
    for arm in phase["arms"].values():
        all_rank = arm["ranking"]["all"]
        repair_rank = arm["ranking"]["repair"]
        ascent_rank = arm["ranking"]["ascent"]
        all_trans = arm["translation"]["all"]
        repair_trans = arm["translation"]["repair"]
        rows.append((
            arm["arm_id"], arm["display_name"],
            all_rank["days"], _fmt(all_rank["daily_rho"]), _fmt(all_rank["pair_accuracy"], suffix=""),
            _fmt(all_rank["top_target_lift_r"], suffix=" R"), _fmt(all_rank["top_vs_rest_pair_accuracy"]),
            repair_rank["days"], _fmt(repair_rank["daily_rho"]), _fmt(repair_rank["top_target_lift_r"], suffix=" R"),
            ascent_rank["days"], _fmt(ascent_rank["top_target_lift_r"], suffix=" R"),
            _fmt(all_trans["raw_action_overlap"]), _fmt(all_trans["action_minus_raw_target_r"], suffix=" R"),
            _fmt(repair_trans["action_minus_raw_target_r"], suffix=" R"),
        ))
    title = "Selection PIT" if phase_id == "selection_pit" else "Forward-OOS"
    return "\n\n".join((
        render_section(f"{title}｜orderable / feasible alignment"),
        render_table(
            (
                "Arm", "Model", "Days", "Daily rho", "Pair", "Top-N Lift", "Top-vs-rest",\
                "Repair days", "Repair rho", "Repair Lift", "Ascent days", "Ascent Lift",\
                "Raw→Action overlap", "Action−Raw Target", "Repair Action−Raw",
            ),
            rows,
        ),
    ))


def _directional_conclusion(payload: dict[str, Any]) -> dict[str, Any]:
    forward = payload["phases"]["forward_oos"]["arms"]
    ids = list(forward)
    if len(ids) < 3:
        return {"classification": "INSUFFICIENT_COMPARATORS"}
    mr12b, mr13a, mr13e = ids[0], ids[1], ids[2]
    def metric(arm_id: str, section: str, stage: str, key: str) -> float | None:
        return _finite(forward[arm_id][section][stage].get(key))
    e_lift = metric(mr13e, "ranking", "all", "top_target_lift_r")
    b_lift = metric(mr12b, "ranking", "all", "top_target_lift_r")
    e_repair = metric(mr13e, "ranking", "repair", "top_target_lift_r")
    b_repair = metric(mr12b, "ranking", "repair", "top_target_lift_r")
    e_trans = metric(mr13e, "translation", "repair", "action_minus_raw_target_r")
    b_trans = metric(mr12b, "translation", "repair", "action_minus_raw_target_r")
    ranking_weaker = e_lift is not None and b_lift is not None and e_lift < b_lift
    repair_weaker = e_repair is not None and b_repair is not None and e_repair < b_repair
    translation_weaker = e_trans is not None and b_trans is not None and e_trans < b_trans
    if ranking_weaker and repair_weaker:
        classification = "ORDERABLE_RANKING_ALIGNMENT_GAP"
    elif not ranking_weaker and translation_weaker:
        classification = "RESOURCE_TRANSLATION_GAP"
    else:
        classification = "MIXED_ORDERABLE_AND_TRANSLATION"
    return {
        "classification": classification,
        "mr13e_vs_mr12b_forward_top_n_lift_delta_r": None if e_lift is None or b_lift is None else e_lift - b_lift,
        "mr13e_vs_mr12b_forward_repair_lift_delta_r": None if e_repair is None or b_repair is None else e_repair - b_repair,
        "mr13e_vs_mr12b_forward_repair_translation_delta_r": None if e_trans is None or b_trans is None else e_trans - b_trans,
        "mr13a_arm_id": mr13a,
    }


def run_orderable_feasible_alignment_audit(
    definition: AuditDefinition,
    *,
    project_root: Path = PROJECT_ROOT,
    quiet: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    status = collect_orderable_feasible_alignment_status(definition, project_root=root)
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
            "run": project_relative_display_path(source["run_dir"], project_root=root),
            "reference_daily_arm_id": source["reference_arm_id"],
            "arms": arms,
        }
    payload = {
        "schema_version": AUDIT_RESULT_SCHEMA_VERSION,
        "status": "COMPLETED",
        "audit_id": definition.audit_id,
        "created_at": get_taipei_now().isoformat(),
        "read_only": True,
        "future_target_used_for_runtime": False,
        "target_contract": {
            "profile": reference["target_profile"],
            "semantic": "daily_opportunity_no_time_r_v1 at latest completed trading day before trade_date",
            "join_timing": "post_replay_audit_only",
        },
        "phases": phases,
    }
    payload["conclusion"] = _directional_conclusion(payload)
    report = "\n\n".join((
        render_title("MR-13E Orderable / Feasible Ranking Alignment Audit"),
        render_key_values((
            ("Audit", definition.audit_id),
            ("契約", "read-only completed Strategy Compare artifacts；不train、不score、不replay"),
            ("比較", "MR-12B / MR-13A / MR-13E"),
            ("共同Future Target", "daily_opportunity_no_time_r_v1；trade_date前一已完成交易日；僅post-replay join"),
            ("N", "每天取實際pre-market action count；不使用固定K"),
        )),
        _render_phase("selection_pit", phases["selection_pit"]),
        _render_phase("forward_oos", phases["forward_oos"]),
        render_section("判定"),
        render_key_values((
            ("Classification", payload["conclusion"]["classification"]),
            ("13E−12B Forward Top-N Lift", _fmt(payload["conclusion"].get("mr13e_vs_mr12b_forward_top_n_lift_delta_r"), suffix=" R")),
            ("13E−12B Repair-day Lift", _fmt(payload["conclusion"].get("mr13e_vs_mr12b_forward_repair_lift_delta_r"), suffix=" R")),
            ("13E−12B Repair Action−Raw", _fmt(payload["conclusion"].get("mr13e_vs_mr12b_forward_repair_translation_delta_r"), suffix=" R")),
        )),
        "限制：execution sidecar代表實際進入entry execution的action candidates；既有工件未永久保存minimum-repair seed與每一步swap membership，因此本Audit可定位raw orderable ranking與resource/action translation，但不宣稱逐swap counterfactual identity。",
    )).rstrip() + "\n"
    output_root = root / Path(AUDIT_OUTPUT_ROOT) / definition.output_subdir
    run_dir = output_root / "runs" / get_taipei_now().strftime("%Y%m%d_%H%M%S")
    latest = output_root / "latest"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "audit.json").write_text(
        json.dumps(_json_native(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "audit.md").write_text(report, encoding="utf-8")
    if latest.exists():
        shutil.rmtree(latest)
    latest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_dir / "audit.json", latest / "audit.json")
    shutil.copy2(run_dir / "audit.md", latest / "audit.md")
    if not quiet:
        print("\n" + report)
        print_artifact_paths(
            (("Audit Markdown", run_dir / "audit.md"), ("Audit JSON", run_dir / "audit.json"), ("最新Audit", latest)),
            project_root=root,
        )
    return payload


__all__ = [
    "collect_orderable_feasible_alignment_status",
    "run_orderable_feasible_alignment_audit",
]
