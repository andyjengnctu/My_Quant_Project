"""Read-only strategy realization/capture attribution for configured score-ranking arms.

This formal Audit consumes completed ``outputs/strategy_compare`` artifacts only.
It never replays the portfolio, rebuilds scores, trains models, or changes selector
or parameter semantics.  Core trade/capital attribution and Target-capture formulas
are reused from the existing Audit primitives rather than reimplemented here.
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
from tools.audit.primitives import finite_or_none as _finite
from tools.audit.breakout_quality.c15_strategy_attribution import (
    build_strategy_attribution_pair_payload,
)
from tools.audit.portfolio.score_ranking_capture import build_score_ranking_capture_audit
from tools.audit.sources.strategy_compare import (
    StrategyCompareArmArtifacts,
    read_json,
    resolve_arm_artifacts,
    resolve_strategy_compare_run_selector,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_RESULT_SCHEMA_VERSION = 1


def _json_native(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) is int:
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return value
    return None if bool(missing) else value


def _validate_definition(
    definition: AuditDefinition,
) -> tuple[str, tuple[str, ...], int, int, int]:
    source = dict(definition.source)
    if str(source.get("kind") or "") != "strategy_compare":
        raise ValueError(f"{definition.audit_id}.source.kind必須是strategy_compare")
    baseline = str(source.get("baseline_arm_id") or "").strip()
    candidates_raw = source.get("candidate_arm_ids")
    if not baseline:
        raise ValueError(f"{definition.audit_id}.baseline_arm_id不可空白")
    if not isinstance(candidates_raw, (list, tuple)) or not candidates_raw:
        raise ValueError(f"{definition.audit_id}.candidate_arm_ids必須是非空list")
    candidates = tuple(str(value).strip() for value in candidates_raw if str(value).strip())
    if not candidates or baseline in candidates or len(set(candidates)) != len(candidates):
        raise ValueError(f"{definition.audit_id}.candidate_arm_ids不合法")
    dimensions = dict(definition.dimensions)
    try:
        focus_year = int(dimensions.get("focus_year"))
        top_month_count = int(dimensions.get("top_month_count"))
        top_trade_count = int(dimensions.get("top_trade_count"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{definition.audit_id}.dimensions必須是整數") from exc
    if focus_year < 1900 or top_month_count < 1 or top_trade_count < 1:
        raise ValueError(f"{definition.audit_id}.dimensions超出合法範圍")
    return baseline, candidates, focus_year, top_month_count, top_trade_count


def _required_pair_paths(
    baseline: StrategyCompareArmArtifacts,
    candidate: StrategyCompareArmArtifacts,
) -> dict[str, Path]:
    return {
        "baseline_trades": baseline.trades_path,
        "candidate_trades": candidate.trades_path,
        "baseline_equity": baseline.equity_path,
        "candidate_equity": candidate.equity_path,
        "baseline_capacity": baseline.capacity_path,
        "candidate_capacity": candidate.capacity_path,
        "baseline_selected": baseline.selected_path,
        "candidate_selected": candidate.selected_path,
        "pair_json": candidate.pair_dir / "strategy_comparison.json",
        "baseline_target": candidate.pair_dir / "no_filter_selected_target_diagnostics.csv",
        "candidate_target": candidate.pair_dir / f"{candidate.prefix}_selected_target_diagnostics.csv",
    }


def _resolve_sources(
    root: Path,
    definition: AuditDefinition,
) -> tuple[Path, dict[str, Any], str, tuple[str, ...], int, int, int]:
    baseline, candidates, focus_year, top_month_count, top_trade_count = _validate_definition(definition)
    selector = dict(definition.source)
    run_dir, result = resolve_strategy_compare_run_selector(
        root,
        selector,
        audit_id=definition.audit_id,
        required_arm_id=baseline,
    )
    arms = dict(dict(result.get("settings") or {}).get("arms") or {})
    scenarios = dict(result.get("scenarios") or {})
    for arm_id in candidates:
        if arm_id not in arms or arm_id not in scenarios:
            raise ValueError(f"strategy_compare run不存在candidate arm: {arm_id}")
    return (
        run_dir,
        result,
        baseline,
        candidates,
        focus_year,
        top_month_count,
        top_trade_count,
    )


def collect_strategy_realization_capture_status(
    definition: AuditDefinition,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    try:
        (
            run_dir,
            result,
            baseline,
            candidates,
            _focus_year,
            _top_month_count,
            _top_trade_count,
        ) = _resolve_sources(root, definition)
        missing: list[str] = []
        for candidate_id in candidates:
            candidate = resolve_arm_artifacts(
                run_dir=run_dir,
                result=result,
                arm_id=candidate_id,
                preferred_pair_arm_id=candidate_id,
            )
            baseline_artifacts = resolve_arm_artifacts(
                run_dir=run_dir,
                result=result,
                arm_id=baseline,
                preferred_pair_arm_id=candidate_id,
            )
            for label, path in _required_pair_paths(baseline_artifacts, candidate).items():
                if not path.is_file():
                    missing.append(f"{candidate_id}:{label}")
        status = "READY" if not missing else "BLOCKED"
        reason = "" if not missing else "缺少既有strategy replay工件: " + ", ".join(missing)
        return {
            "status": status,
            "reason": reason,
            "source": {
                "display": project_relative_display_path(run_dir, project_root=root),
                "baseline_arm_id": baseline,
                "candidate_arm_ids": list(candidates),
                "config_fingerprint": result.get("config_fingerprint"),
            },
        }
    except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError) as exc:
        return {
            "status": "BLOCKED",
            "reason": str(exc),
            "source": {"display": "strategy_compare"},
        }


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"缺少CSV工件: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def _exclusive_breakdown(trade_contributions: pd.DataFrame) -> dict[str, Any]:
    frame = pd.DataFrame(trade_contributions).copy()
    if frame.empty:
        return {
            "baseline_only_count": 0,
            "candidate_only_count": 0,
            "baseline_only_winner_count": 0,
            "baseline_only_winner_r": 0.0,
            "baseline_only_loser_count": 0,
            "baseline_only_loser_r_abs": 0.0,
            "candidate_only_winner_count": 0,
            "candidate_only_winner_r": 0.0,
            "candidate_only_loser_count": 0,
            "candidate_only_loser_r_abs": 0.0,
            "baseline_only_net_r": 0.0,
            "candidate_only_net_r": 0.0,
            "baseline_only_win_rate_pct": None,
            "candidate_only_win_rate_pct": None,
            "exclusive_win_rate_delta_pp": None,
            "winner_r_contribution_delta": 0.0,
            "loser_r_contribution_delta": 0.0,
            "implied_selection_delta_r": 0.0,
            "primary_realized_driver": "none",
        }
    baseline_only = frame[frame["category"] == "comparator_only"].copy()
    candidate_only = frame[frame["category"] == "candidate_only"].copy()
    baseline_r = pd.to_numeric(baseline_only.get("comparator_r"), errors="coerce").fillna(0.0)
    candidate_r = pd.to_numeric(candidate_only.get("candidate_r"), errors="coerce").fillna(0.0)

    baseline_winner_count = int((baseline_r > 0).sum())
    baseline_loser_count = int((baseline_r < 0).sum())
    candidate_winner_count = int((candidate_r > 0).sum())
    candidate_loser_count = int((candidate_r < 0).sum())
    baseline_winner_r = float(baseline_r[baseline_r > 0].sum())
    baseline_loser_r_abs = float(abs(baseline_r[baseline_r < 0].sum()))
    candidate_winner_r = float(candidate_r[candidate_r > 0].sum())
    candidate_loser_r_abs = float(abs(candidate_r[candidate_r < 0].sum()))

    baseline_decisive_count = baseline_winner_count + baseline_loser_count
    candidate_decisive_count = candidate_winner_count + candidate_loser_count
    baseline_win_rate = (
        100.0 * baseline_winner_count / baseline_decisive_count
        if baseline_decisive_count
        else None
    )
    candidate_win_rate = (
        100.0 * candidate_winner_count / candidate_decisive_count
        if candidate_decisive_count
        else None
    )
    winner_contribution = candidate_winner_r - baseline_winner_r
    # Positive means candidate avoids more loser R; negative means candidate adds loser R.
    loser_contribution = baseline_loser_r_abs - candidate_loser_r_abs
    implied_delta = winner_contribution + loser_contribution
    if abs(winner_contribution) > abs(loser_contribution):
        primary_driver = "winner_capture"
    elif abs(loser_contribution) > abs(winner_contribution):
        primary_driver = "loser_avoidance"
    elif abs(implied_delta) <= 1e-12:
        primary_driver = "balanced"
    else:
        primary_driver = "mixed"

    return {
        "baseline_only_count": int(len(baseline_only)),
        "candidate_only_count": int(len(candidate_only)),
        "baseline_only_winner_count": baseline_winner_count,
        "baseline_only_winner_r": baseline_winner_r,
        "baseline_only_loser_count": baseline_loser_count,
        "baseline_only_loser_r_abs": baseline_loser_r_abs,
        "candidate_only_winner_count": candidate_winner_count,
        "candidate_only_winner_r": candidate_winner_r,
        "candidate_only_loser_count": candidate_loser_count,
        "candidate_only_loser_r_abs": candidate_loser_r_abs,
        "baseline_only_net_r": baseline_winner_r - baseline_loser_r_abs,
        "candidate_only_net_r": candidate_winner_r - candidate_loser_r_abs,
        "baseline_only_win_rate_pct": baseline_win_rate,
        "candidate_only_win_rate_pct": candidate_win_rate,
        "exclusive_win_rate_delta_pp": (
            candidate_win_rate - baseline_win_rate
            if candidate_win_rate is not None and baseline_win_rate is not None
            else None
        ),
        "winner_r_contribution_delta": winner_contribution,
        "loser_r_contribution_delta": loser_contribution,
        "implied_selection_delta_r": implied_delta,
        "primary_realized_driver": primary_driver,
    }


def _capture_payload(
    *,
    baseline: StrategyCompareArmArtifacts,
    candidate: StrategyCompareArmArtifacts,
) -> dict[str, Any]:
    pair_json = read_json(candidate.pair_dir / "strategy_comparison.json")
    metadata = dict(pair_json.get("metadata") or {})
    metadata.update({
        "baseline_arm_id": baseline.arm_id,
        "candidate_arm_id": candidate.arm_id,
    })
    baseline_target = _read_csv(candidate.pair_dir / "no_filter_selected_target_diagnostics.csv")
    candidate_target = _read_csv(
        candidate.pair_dir / f"{candidate.prefix}_selected_target_diagnostics.csv"
    )
    return build_score_ranking_capture_audit(
        metadata=metadata,
        baseline_summary=dict(baseline.summary),
        score_sort_summary=dict(candidate.summary),
        baseline_trade_history=_read_csv(baseline.trades_path),
        score_sort_trade_history=_read_csv(candidate.trades_path),
        baseline_selected_target_diagnostics=baseline_target,
        score_sort_selected_target_diagnostics=candidate_target,
        selection_diagnostics=pair_json.get("selection_diagnostics"),
        baseline_daily_capacity=_read_csv(baseline.capacity_path),
        score_sort_daily_capacity=_read_csv(candidate.capacity_path),
    )


def _pair_result(
    *,
    baseline: StrategyCompareArmArtifacts,
    candidate: StrategyCompareArmArtifacts,
    focus_year: int,
    top_month_count: int,
    top_trade_count: int,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    structural = build_strategy_attribution_pair_payload(
        candidate_artifacts=candidate,
        comparator_artifacts=baseline,
        focus_year=focus_year,
        top_month_count=top_month_count,
        top_trade_count=top_trade_count,
    )
    structural_frames = dict(structural.pop("frames"))
    capture = _capture_payload(baseline=baseline, candidate=candidate)
    capture_frames = {
        "capture_baseline_lifecycle": pd.DataFrame(capture.pop("baseline_lifecycle")),
        "capture_candidate_lifecycle": pd.DataFrame(capture.pop("score_sort_lifecycle")),
        "capture_yearly": pd.DataFrame(capture.pop("yearly")),
    }
    exclusive = _exclusive_breakdown(structural_frames["trade_contributions"])
    reported_selection_delta = _finite(
        dict(structural.get("trade_contribution") or {}).get("exclusive_selection_delta_r")
    )
    implied_selection_delta = _finite(exclusive.get("implied_selection_delta_r"))
    if (
        reported_selection_delta is not None
        and implied_selection_delta is not None
        and not math.isclose(
            reported_selection_delta,
            implied_selection_delta,
            rel_tol=1e-9,
            abs_tol=1e-8,
        )
    ):
        raise ValueError(
            "exclusive trade decomposition與canonical exclusive_selection_delta_r不一致: "
            f"{implied_selection_delta:.8f} != {reported_selection_delta:.8f}"
        )
    pair = {
        "baseline_arm_id": baseline.arm_id,
        "candidate_arm_id": candidate.arm_id,
        "structural": structural,
        "capture": capture,
        "exclusive_trade_breakdown": exclusive,
        "interpretation": {
            "candidate_target_mean_higher": (
                (_finite(capture["score_sort"].get("avg_target_r")) or 0.0)
                > (_finite(capture["baseline"].get("avg_target_r")) or 0.0)
            ),
            "candidate_realized_r_higher": (
                (_finite(capture["score_sort"].get("avg_realized_r")) or 0.0)
                > (_finite(capture["baseline"].get("avg_realized_r")) or 0.0)
            ),
            "candidate_capture_higher": (
                (_finite(capture["score_sort"].get("aggregate_target_capture_ratio")) or 0.0)
                > (_finite(capture["baseline"].get("aggregate_target_capture_ratio")) or 0.0)
            ),
            "target_to_realized_divergence": (
                (_finite(capture["score_sort"].get("avg_target_r")) or 0.0)
                > (_finite(capture["baseline"].get("avg_target_r")) or 0.0)
                and (_finite(capture["score_sort"].get("avg_realized_r")) or 0.0)
                <= (_finite(capture["baseline"].get("avg_realized_r")) or 0.0)
            ),
        },
    }
    frames = {**structural_frames, **capture_frames}
    return pair, frames


def _fmt(value: Any, digits: int = 2, unit: str = "", *, signed: bool = False) -> str:
    number = _finite(value)
    if number is None:
        return "N/A"
    sign = "+" if signed else ""
    return f"{number:{sign}.{digits}f}{unit}"


def _render_report(payload: dict[str, Any]) -> str:
    meta = dict(payload.get("metadata") or {})
    lines = [
        render_title("Selection PIT 策略實現／Capture 歸因"),
        render_key_values((
            ("Audit", f"AUD-{payload['audit_id']}"),
            ("策略比較期間", f"{meta.get('comparison_period', {}).get('start', '')} ～ {meta.get('comparison_period', {}).get('end', '')}"),
            ("Baseline", meta.get("baseline_arm_id", "-")),
            ("Candidates", ", ".join(meta.get("candidate_arm_ids") or [])),
            ("Strategy compare", meta.get("strategy_compare_run", "-")),
            ("契約", "只讀既有replay；不重跑portfolio、不改score／selector／params／training"),
        )),
    ]
    for index, pair in enumerate(payload.get("comparisons") or [], start=1):
        baseline_id = pair["baseline_arm_id"]
        candidate_id = pair["candidate_arm_id"]
        structural = pair["structural"]
        capture = pair["capture"]
        baseline_summary = structural["comparator_summary"]
        candidate_summary = structural["candidate_summary"]
        delta_summary = structural["candidate_minus_comparator"]
        trade = structural["trade_contribution"]
        gap = structural["slot_occupancy"]
        base_capture = capture["baseline"]
        cand_capture = capture["score_sort"]
        cap_delta = capture["score_sort_minus_baseline"]
        exclusive = pair["exclusive_trade_breakdown"]
        interpretation = pair["interpretation"]
        capture_decision = dict(capture.get("decision") or {})
        lines.extend((
            render_section(f"{index}. {candidate_id} vs {baseline_id}"),
            render_table(
                ("策略指標", baseline_id, candidate_id, "差異"),
                (
                    ("Return", _fmt(baseline_summary.get("total_return_pct"), unit="%"), _fmt(candidate_summary.get("total_return_pct"), unit="%"), _fmt(delta_summary.get("total_return_pct"), unit="pp", signed=True)),
                    ("MDD", _fmt(baseline_summary.get("max_drawdown_pct"), unit="%"), _fmt(candidate_summary.get("max_drawdown_pct"), unit="%"), _fmt(delta_summary.get("max_drawdown_pct"), unit="pp", signed=True)),
                    ("RoMD", _fmt(baseline_summary.get("return_over_max_drawdown")), _fmt(candidate_summary.get("return_over_max_drawdown")), _fmt(delta_summary.get("return_over_max_drawdown"), signed=True)),
                    ("EV", _fmt(baseline_summary.get("expected_value_r"), unit=" R"), _fmt(candidate_summary.get("expected_value_r"), unit=" R"), _fmt(delta_summary.get("expected_value_r"), unit=" R", signed=True)),
                    ("Exclusive selection R", "0.00 R", _fmt(trade.get("exclusive_selection_delta_r"), unit=" R", signed=True), _fmt(trade.get("exclusive_selection_delta_r"), unit=" R", signed=True)),
                ),
            ),
            render_section("Target → Realized R / 資金捕捉"),
            render_table(
                ("指標", baseline_id, candidate_id, "差異"),
                (
                    ("平均 Target R", _fmt(base_capture.get("avg_target_r"), unit=" R"), _fmt(cand_capture.get("avg_target_r"), unit=" R"), _fmt(cap_delta.get("avg_target_r"), unit=" R", signed=True)),
                    ("平均 Realized R", _fmt(base_capture.get("avg_realized_r"), unit=" R"), _fmt(cand_capture.get("avg_realized_r"), unit=" R"), _fmt(cap_delta.get("avg_realized_r"), unit=" R", signed=True)),
                    ("Aggregate capture", _fmt(base_capture.get("aggregate_target_capture_ratio")), _fmt(cand_capture.get("aggregate_target_capture_ratio")), _fmt(cap_delta.get("aggregate_target_capture_ratio"), signed=True)),
                    ("Target realization gap", _fmt(base_capture.get("avg_target_realization_gap_r"), unit=" R"), _fmt(cand_capture.get("avg_target_realization_gap_r"), unit=" R"), _fmt(cap_delta.get("avg_target_realization_gap_r"), unit=" R", signed=True)),
                    ("Reserved buy fill rate", _fmt(base_capture.get("reserved_buy_fill_rate_pct"), unit="%"), _fmt(cand_capture.get("reserved_buy_fill_rate_pct"), unit="%"), _fmt(cap_delta.get("reserved_buy_fill_rate_pct"), unit="pp", signed=True)),
                    ("平均投入金額", _fmt(base_capture.get("avg_invested_total"), digits=0), _fmt(cand_capture.get("avg_invested_total"), digits=0), _fmt(cap_delta.get("avg_invested_total"), digits=0, signed=True)),
                    ("投入／預留比", _fmt(base_capture.get("avg_invested_vs_reserved_pct"), unit="%"), _fmt(cand_capture.get("avg_invested_vs_reserved_pct"), unit="%"), _fmt(cap_delta.get("avg_invested_vs_reserved_pct"), unit="pp", signed=True)),
                    ("平均持有日", _fmt(base_capture.get("avg_holding_calendar_days"), unit=" 日"), _fmt(cand_capture.get("avg_holding_calendar_days"), unit=" 日"), _fmt(cap_delta.get("avg_holding_calendar_days"), unit=" 日", signed=True)),
                    ("半倉殘留slot-days", _fmt(base_capture.get("partial_residual_slot_days"), digits=0, unit=" 格日"), _fmt(cand_capture.get("partial_residual_slot_days"), digits=0, unit=" 格日"), _fmt(cap_delta.get("partial_residual_slot_days"), digits=0, unit=" 格日", signed=True)),
                    ("期末未滿倉日", str(gap.get("comparator_underfilled_end_days", 0)), str(gap.get("candidate_underfilled_end_days", 0)), f"{int(gap.get('candidate_underfilled_end_days', 0)) - int(gap.get('comparator_underfilled_end_days', 0)):+d}"),
                    ("持股缺口slot-days", str(gap.get("comparator_end_position_gap_slot_days", 0)), str(gap.get("candidate_end_position_gap_slot_days", 0)), f"{int(gap.get('candidate_end_position_gap_slot_days', 0)) - int(gap.get('comparator_end_position_gap_slot_days', 0)):+d}"),
                ),
            ),
            render_section("Exclusive trades：換掉哪些winner／換進哪些loser"),
            render_table(
                ("類別", "筆數", "R"),
                (
                    (f"{baseline_id} only winners", exclusive["baseline_only_winner_count"], _fmt(exclusive["baseline_only_winner_r"], unit=" R")),
                    (f"{baseline_id} only losers", exclusive["baseline_only_loser_count"], _fmt(exclusive["baseline_only_loser_r_abs"], unit=" R")),
                    (f"{candidate_id} only winners", exclusive["candidate_only_winner_count"], _fmt(exclusive["candidate_only_winner_r"], unit=" R")),
                    (f"{candidate_id} only losers", exclusive["candidate_only_loser_count"], _fmt(exclusive["candidate_only_loser_r_abs"], unit=" R")),
                ),
            ),
            render_table(
                ("Exclusive R分解", "差異", "含義"),
                (
                    (
                        "Exclusive win rate",
                        _fmt(exclusive.get("exclusive_win_rate_delta_pp"), unit="pp", signed=True),
                        f"{_fmt(exclusive.get('baseline_only_win_rate_pct'), unit='%')} → {_fmt(exclusive.get('candidate_only_win_rate_pct'), unit='%')}",
                    ),
                    (
                        "Winner R contribution",
                        _fmt(exclusive.get("winner_r_contribution_delta"), unit=" R", signed=True),
                        "負值＝換入／保留的winner總R較少",
                    ),
                    (
                        "Loser R contribution",
                        _fmt(exclusive.get("loser_r_contribution_delta"), unit=" R", signed=True),
                        "正值＝少承擔loser R；負值＝多承擔loser R",
                    ),
                    (
                        "Implied selection R",
                        _fmt(exclusive.get("implied_selection_delta_r"), unit=" R", signed=True),
                        f"主要driver={exclusive.get('primary_realized_driver', '-')}",
                    ),
                ),
            ),
            "判讀：" + (
                "Target較高但Realized R未改善，存在明確Target→realized轉化落差。"
                if interpretation.get("target_to_realized_divergence")
                else "Target與Realized R方向未形成上述反向轉化型態。"
            ),
            f"Capture診斷：{capture_decision.get('status', '-')}｜"
            + "；".join(capture_decision.get("bottlenecks") or ["無明確單一瓶頸"]),
        ))
    lines.extend((
        render_section(f"{len(payload.get('comparisons') or []) + 1}. 使用限制"),
        "Future Target只在既有replay完成後離線join；本Audit不能證明未成交候選的counterfactual realized R。",
        "Exclusive trade R、fill、sizing、holding與slot occupancy是已實現路徑歸因；不得回流OOS模型loss、threshold、selector或參數調整。",
        "Capture惡化本身是realization gap的結果訊號，不等同可由既有策略參數修正的機械瓶頸；只有fill／sizing／deployment／holding等預先定義機械門檻成立時，才支持進入Selection參數適應。",
    ))
    return "\n\n".join(lines).rstrip() + "\n"


def run_strategy_realization_capture_audit(
    definition: AuditDefinition,
    *,
    project_root: Path = PROJECT_ROOT,
    quiet: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    (
        run_dir,
        result,
        baseline_id,
        candidate_ids,
        focus_year,
        top_month_count,
        top_trade_count,
    ) = _resolve_sources(root, definition)
    period = dict(result.get("comparison_period") or {})
    comparisons: list[dict[str, Any]] = []
    frames_by_pair: dict[str, dict[str, pd.DataFrame]] = {}
    for candidate_id in candidate_ids:
        candidate = resolve_arm_artifacts(
            run_dir=run_dir,
            result=result,
            arm_id=candidate_id,
            preferred_pair_arm_id=candidate_id,
        )
        baseline = resolve_arm_artifacts(
            run_dir=run_dir,
            result=result,
            arm_id=baseline_id,
            preferred_pair_arm_id=candidate_id,
        )
        pair, frames = _pair_result(
            baseline=baseline,
            candidate=candidate,
            focus_year=focus_year,
            top_month_count=top_month_count,
            top_trade_count=top_trade_count,
        )
        comparisons.append(pair)
        frames_by_pair[f"{candidate_id}_vs_{baseline_id}"] = frames

    payload = {
        "schema_version": AUDIT_RESULT_SCHEMA_VERSION,
        "created_at": get_taipei_now().isoformat(),
        "audit_id": definition.audit_id,
        "audit_type": definition.audit_type,
        "config": definition.as_dict(),
        "metadata": {
            "strategy_compare_run": project_relative_display_path(run_dir, project_root=root),
            "strategy_compare_config_fingerprint": result.get("config_fingerprint"),
            "comparison_period": period,
            "baseline_arm_id": baseline_id,
            "candidate_arm_ids": list(candidate_ids),
            "read_only": True,
            "portfolio_replay_executed": False,
            "training_performed": False,
            "parameter_optimization_performed": False,
            "future_target_used_for_runtime": False,
        },
        "comparisons": comparisons,
    }
    payload = _json_native(payload)

    output_root = root / Path(AUDIT_OUTPUT_ROOT) / Path(definition.output_subdir)
    timestamp = get_taipei_now().strftime("%Y%m%d_%H%M%S_%f")
    audit_run_dir = output_root / "runs" / timestamp
    latest_dir = output_root / "latest"
    audit_run_dir.mkdir(parents=True, exist_ok=False)
    report_path = audit_run_dir / "audit.md"
    json_path = audit_run_dir / "audit.json"
    report_path.write_text(_render_report(payload), encoding="utf-8")
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    artifacts: list[Path] = [report_path, json_path]
    for pair_key, frames in frames_by_pair.items():
        for frame_name, frame in frames.items():
            path = audit_run_dir / f"{pair_key}_{frame_name}.csv"
            pd.DataFrame(frame).to_csv(path, index=False, encoding="utf-8-sig")
            artifacts.append(path)

    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    latest_dir.mkdir(parents=True, exist_ok=True)
    for source in artifacts:
        shutil.copy2(source, latest_dir / source.name)

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
    "collect_strategy_realization_capture_status",
    "run_strategy_realization_capture_audit",
]
