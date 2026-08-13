"""Read-only cross-period / year-regime attribution for MR-13A vs MR-12B robustness.

The Audit consumes only completed Selection-PIT and Forward-OOS multi-seed
robustness artifacts plus their verified compact attribution sources.  It never
trains, scores, or replays.  Overall baseline-relative direct-selection R remains
separate from direct-pair exclusive trade R; yearly attribution uses only the
direct-pair trade basis because no baseline-relative yearly selection-R artifact
exists.
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
from tools.audit.breakout_quality.c15_strategy_attribution import (
    build_strategy_attribution_pair_payload,
)
from tools.audit.sources.multi_seed_robustness import (
    compact_artifacts_from_unit,
    resolve_two_arm_multi_seed_source,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_RESULT_SCHEMA_VERSION = 1
PHASE_ORDER = ("selection_pit", "forward_oos")


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_native(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    return value


def _distribution(values: pd.Series | list[float]) -> dict[str, Any]:
    numbers = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if not len(numbers):
        return {
            "n": 0, "mean": None, "median": None, "std": None,
            "min": None, "max": None,
        }
    return {
        "n": int(len(numbers)),
        "mean": float(np.mean(numbers)),
        "median": float(np.median(numbers)),
        "std": float(np.std(numbers, ddof=1)) if len(numbers) > 1 else None,
        "min": float(np.min(numbers)),
        "max": float(np.max(numbers)),
    }


def _validate_definition(definition: AuditDefinition) -> dict[str, dict[str, str]]:
    source = dict(definition.source)
    if str(source.get("kind") or "") != "multi_seed_robustness_cross_period":
        raise ValueError(
            f"{definition.audit_id}.source.kind必須是multi_seed_robustness_cross_period"
        )
    phases_raw = source.get("phases")
    if not isinstance(phases_raw, dict):
        raise ValueError(f"{definition.audit_id}.source.phases必須是mapping")
    phases: dict[str, dict[str, str]] = {}
    for phase_id in PHASE_ORDER:
        raw = phases_raw.get(phase_id)
        if not isinstance(raw, dict):
            raise ValueError(f"{definition.audit_id}.source.phases缺少{phase_id}")
        run_selector = str(raw.get("run") or "latest").strip().lower()
        if run_selector != "latest":
            raise ValueError(f"{definition.audit_id}.{phase_id}.run目前只允許latest")
        robustness_id = str(raw.get("robustness_id") or "").strip()
        candidate = str(raw.get("candidate_arm_id") or "").strip()
        comparator = str(raw.get("comparator_arm_id") or "").strip()
        if (
            robustness_id != phase_id
            or not candidate
            or not comparator
            or candidate == comparator
        ):
            raise ValueError(f"{definition.audit_id}.{phase_id} identity設定不合法")
        phases[phase_id] = {
            "robustness_id": robustness_id,
            "candidate_arm_id": candidate,
            "comparator_arm_id": comparator,
        }
    return phases


def _resolve_sources(
    root: Path,
    definition: AuditDefinition,
) -> dict[str, dict[str, Any]]:
    phase_config = _validate_definition(definition)
    sources: dict[str, dict[str, Any]] = {}
    for phase_id in PHASE_ORDER:
        cfg = phase_config[phase_id]
        sources[phase_id] = resolve_two_arm_multi_seed_source(
            root,
            robustness_id=cfg["robustness_id"],
            candidate_arm_id=cfg["candidate_arm_id"],
            comparator_arm_id=cfg["comparator_arm_id"],
            require_all_seeds=True,
            include_yearly=True,
        )

    selection_contract = dict(sources["selection_pit"]["contract"])
    forward_contract = dict(sources["forward_oos"]["contract"])
    selection_seeds = tuple(int(value) for value in selection_contract.get("resolved_seeds") or ())
    forward_seeds = tuple(int(value) for value in forward_contract.get("resolved_seeds") or ())
    if not selection_seeds or selection_seeds != forward_seeds:
        raise ValueError("Selection PIT與Forward robustness resolved_seeds不一致")
    if int(selection_contract.get("seed_count") or 0) != int(forward_contract.get("seed_count") or 0):
        raise ValueError("Selection PIT與Forward robustness seed_count不一致")
    if int(selection_contract.get("seed_generator_seed") or -1) != int(
        forward_contract.get("seed_generator_seed") or -2
    ):
        raise ValueError("Selection PIT與Forward robustness seed_generator_seed不一致")
    return sources


def collect_cross_period_year_regime_attribution_status(
    definition: AuditDefinition,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    try:
        sources = _resolve_sources(root, definition)
        seed_count = len(sources["selection_pit"]["seed_pairs"])
        return {
            "audit_id": definition.audit_id,
            "status": "READY",
            "reason": "",
            "source": {
                "display": f"Selection PIT vs Forward-OOS / {seed_count} same seeds",
                "selection_fingerprint": sources["selection_pit"]["fingerprint"],
                "forward_fingerprint": sources["forward_oos"]["fingerprint"],
                "selection_run": project_relative_display_path(
                    sources["selection_pit"]["run_root"], project_root=root
                ),
                "forward_run": project_relative_display_path(
                    sources["forward_oos"]["run_root"], project_root=root
                ),
            },
        }
    except (FileNotFoundError, ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        return {
            "audit_id": definition.audit_id,
            "status": "BLOCKED",
            "reason": str(exc),
            "source": {
                "display": "Selection PIT vs Forward-OOS",
            },
        }


def _seed_phase_summary(
    source: dict[str, Any],
    *,
    phase_id: str,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    candidate = str(source["candidate_arm_id"])
    comparator = str(source["comparator_arm_id"])
    seed_frame = pd.DataFrame(source["seed_frame"]).copy()
    rows: list[dict[str, Any]] = []
    trade_frames: list[pd.DataFrame] = []

    for seed_order, seed in source["seed_pairs"]:
        c_row = seed_frame[
            (seed_frame["arm_id"] == candidate)
            & (seed_frame["seed"] == int(seed))
        ].iloc[0].to_dict()
        b_row = seed_frame[
            (seed_frame["arm_id"] == comparator)
            & (seed_frame["seed"] == int(seed))
        ].iloc[0].to_dict()
        _c_dir, c_manifest = source["units"][(candidate, int(seed))]
        _b_dir, b_manifest = source["units"][(comparator, int(seed))]
        c_artifacts = compact_artifacts_from_unit(
            PROJECT_ROOT if source.get("_project_root") is None else source["_project_root"],
            c_manifest,
            c_row,
        )
        b_artifacts = compact_artifacts_from_unit(
            PROJECT_ROOT if source.get("_project_root") is None else source["_project_root"],
            b_manifest,
            b_row,
        )
        pair = build_strategy_attribution_pair_payload(
            candidate_artifacts=c_artifacts,
            comparator_artifacts=b_artifacts,
            focus_year=int(
                dict(c_manifest.get("comparison_period") or {}).get("start", "2000")[:4]
            ),
            top_month_count=1,
            top_trade_count=1,
        )
        trade = dict(pair.get("trade_contribution") or {})
        row = {
            "phase_id": phase_id,
            "seed_order": int(seed_order),
            "seed": int(seed),
            "candidate_arm_id": candidate,
            "comparator_arm_id": comparator,
            "delta_direct_selection_r": (
                float(c_row["direct_selection_r"]) - float(b_row["direct_selection_r"])
            ),
            "delta_return_pct": (
                float(c_row["total_return_pct"]) - float(b_row["total_return_pct"])
            ),
            "delta_mdd_pct": (
                float(c_row["max_drawdown_pct"]) - float(b_row["max_drawdown_pct"])
            ),
            "delta_romd": (
                float(c_row["return_over_max_drawdown"])
                - float(b_row["return_over_max_drawdown"])
            ),
            "direct_pair_exclusive_delta_r": float(
                trade.get("exclusive_selection_delta_r") or 0.0
            ),
            "direct_pair_exclusive_delta_pnl": float(
                trade.get("exclusive_selection_delta_pnl") or 0.0
            ),
            "direct_pair_common_delta_r": float(
                trade.get("common_trade_delta_r") or 0.0
            ),
            "direct_pair_all_trade_delta_r": float(
                trade.get("all_trade_delta_r") or 0.0
            ),
        }
        row["selection_basis_gap_r"] = (
            row["direct_pair_exclusive_delta_r"] - row["delta_direct_selection_r"]
        )
        rows.append(row)

        trades = pd.DataFrame(dict(pair.get("frames") or {}).get("trade_contributions")).copy()
        if trades.empty:
            trades = pd.DataFrame(columns=[
                "category", "entry_year", "candidate_r", "comparator_r",
                "r_delta", "candidate_pnl", "comparator_pnl", "pnl_delta",
            ])
        trades.insert(0, "seed", int(seed))
        trades.insert(0, "seed_order", int(seed_order))
        trades.insert(0, "phase_id", phase_id)
        trade_frames.append(trades)

    return pd.DataFrame(rows).sort_values("seed_order").reset_index(drop=True), trade_frames


def _yearly_return_rows(source: dict[str, Any], *, phase_id: str) -> pd.DataFrame:
    candidate = str(source["candidate_arm_id"])
    comparator = str(source["comparator_arm_id"])
    frame = pd.DataFrame(source["seed_yearly_frame"]).copy()
    frame = frame[frame["arm_id"].isin((candidate, comparator))].copy()
    rows: list[dict[str, Any]] = []
    for year, group in frame.groupby("year", sort=True):
        pivot = group.pivot(index=["seed_order", "seed"], columns="arm_id", values="return_pct")
        if candidate not in pivot.columns or comparator not in pivot.columns:
            raise ValueError(f"{phase_id} yearly returns缺少candidate/comparator: {year}")
        pivot = pivot[[candidate, comparator]].dropna().reset_index()
        complete = bool(group["is_complete_year"].astype(bool).all())
        for item in pivot.to_dict("records"):
            rows.append({
                "phase_id": phase_id,
                "year": int(year),
                "is_complete_year": complete,
                "seed_order": int(item["seed_order"]),
                "seed": int(item["seed"]),
                "delta_return_pct": float(item[candidate]) - float(item[comparator]),
            })
    return pd.DataFrame(rows)


def _yearly_trade_rows(trade_frame: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(trade_frame).copy()
    if frame.empty:
        return pd.DataFrame(columns=[
            "phase_id", "year", "seed_order", "seed",
            "direct_pair_exclusive_delta_r", "direct_pair_exclusive_delta_pnl",
            "candidate_only_total_r", "comparator_only_total_r",
            "candidate_only_trade_count", "comparator_only_trade_count",
        ])
    if "entry_year" not in frame.columns:
        raise ValueError("trade attribution缺少entry_year")
    frame["entry_year"] = pd.to_numeric(frame["entry_year"], errors="coerce")
    frame = frame[frame["entry_year"].notna()].copy()
    frame["year"] = frame["entry_year"].astype(int)
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(
        ["phase_id", "year", "seed_order", "seed"], sort=True
    ):
        phase_id, year, seed_order, seed = keys
        exclusive = group[group["category"].isin(("candidate_only", "comparator_only"))]
        candidate_only = group[group["category"] == "candidate_only"]
        comparator_only = group[group["category"] == "comparator_only"]
        rows.append({
            "phase_id": str(phase_id),
            "year": int(year),
            "seed_order": int(seed_order),
            "seed": int(seed),
            "direct_pair_exclusive_delta_r": float(
                pd.to_numeric(exclusive.get("r_delta"), errors="coerce").fillna(0.0).sum()
            ),
            "direct_pair_exclusive_delta_pnl": float(
                pd.to_numeric(exclusive.get("pnl_delta"), errors="coerce").fillna(0.0).sum()
            ),
            "candidate_only_total_r": float(
                pd.to_numeric(candidate_only.get("candidate_r"), errors="coerce").fillna(0.0).sum()
            ),
            "comparator_only_total_r": float(
                pd.to_numeric(comparator_only.get("comparator_r"), errors="coerce").fillna(0.0).sum()
            ),
            "candidate_only_trade_count": int(len(candidate_only)),
            "comparator_only_trade_count": int(len(comparator_only)),
        })
    return pd.DataFrame(rows)


def _aggregate_yearly(
    yearly_return: pd.DataFrame,
    yearly_trade: pd.DataFrame,
    *,
    expected_seed_count: int,
) -> pd.DataFrame:
    merged = yearly_return.merge(
        yearly_trade,
        on=["phase_id", "year", "seed_order", "seed"],
        how="left",
        validate="one_to_one",
    )
    fill_zero = [
        "direct_pair_exclusive_delta_r",
        "direct_pair_exclusive_delta_pnl",
        "candidate_only_total_r",
        "comparator_only_total_r",
        "candidate_only_trade_count",
        "comparator_only_trade_count",
    ]
    for column in fill_zero:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
    rows: list[dict[str, Any]] = []
    for (phase_id, year), group in merged.groupby(["phase_id", "year"], sort=True):
        if len(group) != expected_seed_count:
            raise ValueError(
                f"{phase_id}/{year} yearly attribution seed coverage不足: "
                f"expected={expected_seed_count}, actual={len(group)}"
            )
        ret = pd.to_numeric(group["delta_return_pct"], errors="coerce")
        exclusive_r = pd.to_numeric(
            group["direct_pair_exclusive_delta_r"], errors="coerce"
        )
        exclusive_pnl = pd.to_numeric(
            group["direct_pair_exclusive_delta_pnl"], errors="coerce"
        )
        rows.append({
            "phase_id": str(phase_id),
            "year": int(year),
            "is_complete_year": bool(group["is_complete_year"].astype(bool).all()),
            "seed_count": int(len(group)),
            "delta_return_mean_pct": float(ret.mean()),
            "delta_return_positive_count": int((ret > 0).sum()),
            "direct_pair_exclusive_delta_r_mean": float(exclusive_r.mean()),
            "direct_pair_exclusive_delta_r_median": float(exclusive_r.median()),
            "direct_pair_exclusive_delta_r_positive_count": int((exclusive_r > 0).sum()),
            "direct_pair_exclusive_delta_pnl_mean": float(exclusive_pnl.mean()),
            "candidate_only_total_r_mean": float(group["candidate_only_total_r"].mean()),
            "comparator_only_total_r_mean": float(group["comparator_only_total_r"].mean()),
            "candidate_only_trade_count_mean": float(group["candidate_only_trade_count"].mean()),
            "comparator_only_trade_count_mean": float(group["comparator_only_trade_count"].mean()),
        })
    return pd.DataFrame(rows)


def _phase_aggregate(seed_summary: pd.DataFrame, yearly_summary: pd.DataFrame) -> dict[str, Any]:
    seed = pd.DataFrame(seed_summary)
    yearly = pd.DataFrame(yearly_summary)
    direct_selection = pd.to_numeric(seed["delta_direct_selection_r"], errors="coerce")
    direct_pair = pd.to_numeric(seed["direct_pair_exclusive_delta_r"], errors="coerce")
    romd = pd.to_numeric(seed["delta_romd"], errors="coerce")
    selection_sign = np.sign(direct_selection)
    romd_sign = np.sign(romd)
    non_tie = (selection_sign != 0) & (romd_sign != 0)
    return {
        "seed_count": int(len(seed)),
        "delta_direct_selection_r": _distribution(direct_selection),
        "delta_direct_selection_r_positive_count": int((direct_selection > 0).sum()),
        "delta_romd": _distribution(romd),
        "delta_romd_positive_count": int((romd > 0).sum()),
        "direction_aligned_count": int(
            (selection_sign[non_tie] == romd_sign[non_tie]).sum()
        ),
        "direction_non_tie_count": int(non_tie.sum()),
        "direct_pair_exclusive_delta_r": _distribution(direct_pair),
        "direct_pair_exclusive_delta_r_positive_count": int((direct_pair > 0).sum()),
        "selection_basis_gap_r": _distribution(seed["selection_basis_gap_r"]),
        "negative_year_count": int(
            (yearly["direct_pair_exclusive_delta_r_mean"] < 0).sum()
        ) if not yearly.empty else 0,
        "positive_year_count": int(
            (yearly["direct_pair_exclusive_delta_r_mean"] > 0).sum()
        ) if not yearly.empty else 0,
        "year_count": int(len(yearly)),
    }


def _interpretation(
    phase_aggregates: dict[str, dict[str, Any]],
    yearly_summary: pd.DataFrame,
) -> dict[str, Any]:
    selection = phase_aggregates["selection_pit"]
    forward = phase_aggregates["forward_oos"]
    s_mean = _finite(selection["delta_direct_selection_r"]["mean"]) or 0.0
    f_mean = _finite(forward["delta_direct_selection_r"]["mean"]) or 0.0
    reversal = s_mean < 0.0 < f_mean

    selection_years = yearly_summary[yearly_summary["phase_id"] == "selection_pit"]
    forward_years = yearly_summary[yearly_summary["phase_id"] == "forward_oos"]

    def extremum(frame: pd.DataFrame, *, smallest: bool) -> dict[str, Any] | None:
        if frame.empty:
            return None
        ordered = frame.sort_values(
            "direct_pair_exclusive_delta_r_mean", ascending=smallest
        )
        return dict(ordered.iloc[0].to_dict())

    negative_selection = selection_years[
        selection_years["direct_pair_exclusive_delta_r_mean"] < 0
    ].copy()
    negative_sum = float(
        negative_selection["direct_pair_exclusive_delta_r_mean"].sum()
    ) if not negative_selection.empty else 0.0
    worst = extremum(selection_years, smallest=True)
    worst_negative_share = None
    if worst and negative_sum < 0 and float(worst["direct_pair_exclusive_delta_r_mean"]) < 0:
        worst_negative_share = float(
            abs(float(worst["direct_pair_exclusive_delta_r_mean"])) / abs(negative_sum)
        )

    return {
        "classification": (
            "RANKING_EDGE_DIRECTION_REVERSAL"
            if reversal
            else "NO_CLEAN_BASELINE_RELATIVE_DIRECTION_REVERSAL"
        ),
        "baseline_relative_direction_reversal": bool(reversal),
        "selection_worst_year": worst,
        "forward_best_year": extremum(forward_years, smallest=False),
        "selection_worst_year_share_of_negative_year_direct_pair_r": worst_negative_share,
        "selection_negative_year_count": int(len(negative_selection)),
        "selection_year_count": int(len(selection_years)),
        "forward_positive_year_count": int(
            (forward_years["direct_pair_exclusive_delta_r_mean"] > 0).sum()
        ),
        "forward_year_count": int(len(forward_years)),
        "yearly_basis_note": (
            "逐年度只使用MR-13A vs MR-12B direct-pair exclusive trade R/PnL；"
            "沒有baseline-relative逐年DL選擇R工件，因此不得把兩者視為同一basis。"
        ),
    }


def _render_report(payload: dict[str, Any]) -> str:
    metadata = payload["metadata"]
    phase_agg = payload["phase_aggregate"]
    seed_rows = []
    seed_map: dict[int, dict[str, dict[str, Any]]] = {}
    for row in payload["seed_summary"]:
        seed_map.setdefault(int(row["seed_order"]), {})[str(row["phase_id"])] = row
    for seed_order in sorted(seed_map):
        s = seed_map[seed_order]["selection_pit"]
        f = seed_map[seed_order]["forward_oos"]
        transition = (
            ("+" if s["delta_direct_selection_r"] > 0 else "−" if s["delta_direct_selection_r"] < 0 else "0")
            + "→"
            + ("+" if f["delta_direct_selection_r"] > 0 else "−" if f["delta_direct_selection_r"] < 0 else "0")
        )
        seed_rows.append((
            f"S{seed_order}",
            f"{s['delta_direct_selection_r']:+.2f} R",
            f"{s['delta_romd']:+.2f}",
            f"{f['delta_direct_selection_r']:+.2f} R",
            f"{f['delta_romd']:+.2f}",
            transition,
        ))

    yearly_rows = []
    for row in payload["yearly_summary"]:
        label = "Selection" if row["phase_id"] == "selection_pit" else "Forward"
        year = f"{int(row['year'])}{'' if row['is_complete_year'] else '*'}"
        yearly_rows.append((
            label,
            year,
            f"{row['delta_return_mean_pct']:+.2f}%",
            f"{row['delta_return_positive_count']}/{row['seed_count']}",
            f"{row['direct_pair_exclusive_delta_r_mean']:+.2f} R",
            f"{row['direct_pair_exclusive_delta_r_positive_count']}/{row['seed_count']}",
            f"{row['direct_pair_exclusive_delta_pnl_mean']:+,.0f}",
            f"{row['candidate_only_total_r_mean']:+.2f} / {row['comparator_only_total_r_mean']:+.2f} R",
            f"{row['candidate_only_trade_count_mean']:.1f} / {row['comparator_only_trade_count_mean']:.1f}",
        ))

    phase_rows = []
    for phase_id in PHASE_ORDER:
        label = "Selection PIT" if phase_id == "selection_pit" else "Forward-OOS"
        row = phase_agg[phase_id]
        phase_rows.append((
            label,
            f"{row['delta_direct_selection_r']['mean']:+.2f} R",
            f"{row['delta_direct_selection_r_positive_count']}/{row['seed_count']}",
            f"{row['delta_romd']['mean']:+.2f}",
            f"{row['delta_romd_positive_count']}/{row['seed_count']}",
            f"{row['direction_aligned_count']}/{row['direction_non_tie_count']}",
            f"{row['direct_pair_exclusive_delta_r']['mean']:+.2f} R",
            f"{row['negative_year_count']}− / {row['positive_year_count']}+",
        ))

    interpretation = payload["interpretation"]
    worst = interpretation.get("selection_worst_year") or {}
    best = interpretation.get("forward_best_year") or {}
    share = interpretation.get(
        "selection_worst_year_share_of_negative_year_direct_pair_r"
    )
    return "\n\n".join((
        render_title("Selection vs Forward Cross-period / Year-regime Attribution"),
        render_key_values((
            ("Audit", payload["audit_id"]),
            ("Seeds", metadata["seed_count"]),
            ("Seed generator", metadata["seed_generator_seed"]),
            ("Selection fingerprint", metadata["selection_fingerprint"]),
            ("Forward fingerprint", metadata["forward_fingerprint"]),
            ("比較", "MR-13A − MR-12B；same generated seeds"),
            ("契約", "read-only robustness + verified compact attribution；不train、不score、不replay"),
        )),
        render_section("同seed cross-period direction", number=1),
        render_table(
            ("Seed", "Selection ΔDL選擇R", "Selection ΔRoMD", "Forward ΔDL選擇R", "Forward ΔRoMD", "Ranking方向"),
            seed_rows,
        ),
        render_section("逐年度 direct-pair trade-set attribution", number=2),
        render_table(
            ("Phase", "Year", "ΔReturn Mean", "Return勝", "Exclusive ΔR Mean", "ΔR勝", "Exclusive ΔPnL Mean", "13A-only / 12B-only ΣR Mean", "13A-only / 12B-only Trades Mean"),
            yearly_rows,
        ),
        "逐年度Exclusive ΔR是MR-13A vs MR-12B直接trade partition，不是兩arm各自相對DL-off baseline的逐年ΔDL選擇R；後者目前沒有永久逐年工件，因此不得混用basis。",
        render_section("Period aggregate", number=3),
        render_table(
            ("Period", "ΔDL選擇R Mean", "Ranking勝", "ΔRoMD Mean", "RoMD勝", "Ranking/RoMD同向", "Direct-pair Exclusive ΔR Mean", "年度符號"),
            phase_rows,
        ),
        render_section("判定", number=4),
        render_key_values((
            ("Classification", interpretation["classification"]),
            ("Selection worst direct-pair year", "-" if not worst else f"{int(worst['year'])}: {float(worst['direct_pair_exclusive_delta_r_mean']):+.2f} R mean"),
            ("Worst-year占Selection負年度R", "-" if share is None else f"{float(share) * 100.0:.1f}%"),
            ("Forward best direct-pair year", "-" if not best else f"{int(best['year'])}: {float(best['direct_pair_exclusive_delta_r_mean']):+.2f} R mean"),
            ("Selection負年度", f"{interpretation['selection_negative_year_count']}/{interpretation['selection_year_count']}"),
            ("Forward正年度", f"{interpretation['forward_positive_year_count']}/{interpretation['forward_year_count']}"),
        )),
        "本Audit只定位relative ranking/trade-set edge的跨時段與年度集中度；不建立新regime gate、不挑seed、不回流training。若負edge廣泛分布於Selection多年度，MR-13A daily-universal training semantics不具跨期穩定優勢；若高度集中單一年份，才進一步做該年度的資料／市場regime read-only診斷。",
    ))


def run_cross_period_year_regime_attribution_audit(
    definition: AuditDefinition,
    *,
    project_root: Path = PROJECT_ROOT,
    quiet: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    sources = _resolve_sources(root, definition)
    for source in sources.values():
        source["_project_root"] = root

    seed_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    yearly_return_frames: list[pd.DataFrame] = []
    for phase_id in PHASE_ORDER:
        seed_summary, phase_trade_frames = _seed_phase_summary(
            sources[phase_id], phase_id=phase_id
        )
        seed_frames.append(seed_summary)
        trade_frames.extend(phase_trade_frames)
        yearly_return_frames.append(
            _yearly_return_rows(sources[phase_id], phase_id=phase_id)
        )

    seed_summary = pd.concat(seed_frames, ignore_index=True)
    trade_frame = pd.concat(trade_frames, ignore_index=True)
    yearly_return = pd.concat(yearly_return_frames, ignore_index=True)
    yearly_trade = _yearly_trade_rows(trade_frame)
    expected_seed_count = len(sources["selection_pit"]["seed_pairs"])
    yearly_summary = _aggregate_yearly(
        yearly_return,
        yearly_trade,
        expected_seed_count=expected_seed_count,
    )

    phase_aggregate = {
        phase_id: _phase_aggregate(
            seed_summary[seed_summary["phase_id"] == phase_id],
            yearly_summary[yearly_summary["phase_id"] == phase_id],
        )
        for phase_id in PHASE_ORDER
    }
    interpretation = _interpretation(phase_aggregate, yearly_summary)
    selection_contract = dict(sources["selection_pit"]["contract"])

    payload = {
        "schema_version": AUDIT_RESULT_SCHEMA_VERSION,
        "created_at": get_taipei_now().isoformat(),
        "audit_id": definition.audit_id,
        "audit_type": definition.audit_type,
        "config": definition.as_dict(),
        "metadata": {
            "seed_count": expected_seed_count,
            "seed_generator_seed": int(selection_contract.get("seed_generator_seed")),
            "resolved_seeds": list(selection_contract.get("resolved_seeds") or []),
            "selection_fingerprint": sources["selection_pit"]["fingerprint"],
            "forward_fingerprint": sources["forward_oos"]["fingerprint"],
            "selection_run": project_relative_display_path(
                sources["selection_pit"]["run_root"], project_root=root
            ),
            "forward_run": project_relative_display_path(
                sources["forward_oos"]["run_root"], project_root=root
            ),
            "read_only": True,
            "training_performed": False,
            "score_build_performed": False,
            "portfolio_replay_executed": False,
        },
        "seed_summary": seed_summary.to_dict("records"),
        "yearly_summary": yearly_summary.to_dict("records"),
        "phase_aggregate": phase_aggregate,
        "interpretation": interpretation,
    }
    payload = _json_native(payload)

    output_root = root / Path(AUDIT_OUTPUT_ROOT) / Path(definition.output_subdir)
    timestamp = get_taipei_now().strftime("%Y%m%d_%H%M%S_%f")
    audit_run_dir = output_root / "runs" / timestamp
    latest_dir = output_root / "latest"
    audit_run_dir.mkdir(parents=True, exist_ok=False)

    report_path = audit_run_dir / "audit.md"
    json_path = audit_run_dir / "audit.json"
    seed_path = audit_run_dir / "seed_cross_period.csv"
    yearly_path = audit_run_dir / "year_regime_summary.csv"
    trade_path = audit_run_dir / "trade_attribution.csv.gz"

    report_path.write_text(_render_report(payload), encoding="utf-8")
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    seed_summary.to_csv(seed_path, index=False, encoding="utf-8-sig")
    yearly_summary.to_csv(yearly_path, index=False, encoding="utf-8-sig")
    trade_frame.to_csv(
        trade_path, index=False, encoding="utf-8-sig", compression="gzip"
    )
    artifacts = [report_path, json_path, seed_path, yearly_path, trade_path]

    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    latest_dir.mkdir(parents=True, exist_ok=True)
    for source_path in artifacts:
        shutil.copy2(source_path, latest_dir / source_path.name)

    if not quiet:
        print("\n" + _render_report(payload))
        print_artifact_paths(
            (
                ("Audit Markdown", report_path),
                ("Audit JSON", json_path),
                ("最新Audit", latest_dir),
            ),
            project_root=root,
        )
    return payload


__all__ = [
    "AUDIT_RESULT_SCHEMA_VERSION",
    "collect_cross_period_year_regime_attribution_status",
    "run_cross_period_year_regime_attribution_audit",
]
