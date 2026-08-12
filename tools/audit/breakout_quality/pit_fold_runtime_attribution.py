"""Read-only PIT fold/runtime attribution for configured strategy-comparison arms.

The Audit explains whether Selection PIT direct-deployment losses concentrate on
trade dates where the orderable pool mixes scores produced by different PIT fold
models, or near PIT fold transitions.  It consumes completed strategy replay and
validated PIT score/audit artifacts only; it never replays, trains, calibrates, or
changes selector/runtime semantics.
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
from filters.breakout_quality.ranking_score_store import (
    load_selection_point_in_time_ranking_contract,
    load_selection_point_in_time_score_table,
)
from tools.audit.breakout_quality.c15_strategy_attribution import (
    build_strategy_attribution_pair_payload,
)
from tools.audit.sources.strategy_compare import (
    StrategyCompareArmArtifacts,
    resolve_arm_artifacts,
    resolve_strategy_compare_run_selector,
)

from tools.audit.breakout_quality.pit_primitives import candidate_pit_identity as _candidate_pit_identity

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_RESULT_SCHEMA_VERSION = 1


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


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
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return value
    return None if bool(missing) else value


def _validate_definition(definition: AuditDefinition) -> tuple[str, tuple[str, ...], int, int, int, int]:
    source = dict(definition.source)
    if str(source.get("kind") or "") != "strategy_compare":
        raise ValueError(f"{definition.audit_id}.source.kind必須是strategy_compare")
    baseline = str(source.get("baseline_arm_id") or "").strip()
    raw_candidates = source.get("candidate_arm_ids")
    if not baseline:
        raise ValueError(f"{definition.audit_id}.baseline_arm_id不可空白")
    if not isinstance(raw_candidates, (list, tuple)) or not raw_candidates:
        raise ValueError(f"{definition.audit_id}.candidate_arm_ids必須是非空list")
    candidates = tuple(str(value).strip() for value in raw_candidates if str(value).strip())
    if not candidates or baseline in candidates or len(set(candidates)) != len(candidates):
        raise ValueError(f"{definition.audit_id}.candidate_arm_ids不合法")
    dimensions = dict(definition.dimensions)
    try:
        boundary_window_days = int(dimensions.get("fold_boundary_window_days"))
        focus_year = int(dimensions.get("focus_year"))
        top_month_count = int(dimensions.get("top_month_count"))
        top_trade_count = int(dimensions.get("top_trade_count"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{definition.audit_id}.dimensions必須是整數") from exc
    if boundary_window_days < 0 or boundary_window_days > 366:
        raise ValueError(f"{definition.audit_id}.fold_boundary_window_days超出合法範圍")
    if focus_year < 1900 or top_month_count < 1 or top_trade_count < 1:
        raise ValueError(f"{definition.audit_id}.attribution dimensions超出合法範圍")
    return baseline, candidates, boundary_window_days, focus_year, top_month_count, top_trade_count




def _resolve_sources(
    root: Path,
    definition: AuditDefinition,
) -> tuple[Path, dict[str, Any], str, tuple[str, ...], int]:
    baseline, candidates, boundary_window_days, focus_year, top_month_count, top_trade_count = _validate_definition(definition)
    run_dir, result = resolve_strategy_compare_run_selector(
        root,
        dict(definition.source),
        audit_id=definition.audit_id,
        required_arm_id=baseline,
    )
    for candidate_id in candidates:
        if candidate_id not in dict(result.get("scenarios") or {}):
            raise ValueError(f"strategy_compare run不存在candidate arm: {candidate_id}")
        _candidate_pit_identity(result, candidate_id)
    return run_dir, result, baseline, candidates, boundary_window_days, focus_year, top_month_count, top_trade_count


def _required_paths(
    *,
    baseline: StrategyCompareArmArtifacts,
    candidate: StrategyCompareArmArtifacts,
) -> dict[str, Path]:
    return {
        "baseline_trades": baseline.trades_path,
        "candidate_trades": candidate.trades_path,
        "candidate_orderable": candidate.orderable_path,
        "pair_json": candidate.pair_dir / "strategy_comparison.json",
    }


def collect_pit_fold_runtime_attribution_status(
    definition: AuditDefinition,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    try:
        run_dir, result, baseline_id, candidate_ids, boundary_window_days, _focus_year, _top_month_count, _top_trade_count = _resolve_sources(
            root, definition
        )
        missing: list[str] = []
        pit_identities: dict[str, Any] = {}
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
            for label, path in _required_paths(baseline=baseline, candidate=candidate).items():
                if not path.is_file():
                    missing.append(f"{candidate_id}:{label}")
            identity = _candidate_pit_identity(result, candidate_id)
            contract = load_selection_point_in_time_ranking_contract(
                str(root),
                identity["filter_id"],
                identity["model_architecture"],
                identity["experiment_profile"],
            )
            pit_identities[candidate_id] = {
                **identity,
                "available_from": contract.available_from,
                "available_through": contract.available_through,
            }
        return {
            "status": "READY" if not missing else "BLOCKED",
            "reason": "" if not missing else "缺少正式只讀工件: " + ", ".join(missing),
            "source": {
                "display": project_relative_display_path(run_dir, project_root=root),
                "baseline_arm_id": baseline_id,
                "candidate_arm_ids": list(candidate_ids),
                "fold_boundary_window_days": boundary_window_days,
                "pit_identities": pit_identities,
            },
        }
    except (FileNotFoundError, ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        return {
            "status": "BLOCKED",
            "reason": str(exc),
            "source": {"display": "strategy_compare / Selection PIT"},
        }


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"缺少CSV工件: {path}")
    return pd.read_csv(path, encoding="utf-8-sig", dtype={"ticker": "string"})


def _normalize_orderable(orderable: pd.DataFrame) -> pd.DataFrame:
    work = pd.DataFrame(orderable).copy()
    required = {"ticker", "trade_date", "signal_date"}
    missing = sorted(required.difference(work.columns))
    if missing:
        raise ValueError(f"orderable candidate缺少欄位: {missing}")
    work["ticker"] = work["ticker"].fillna("").astype(str).str.strip()
    for column in ("trade_date", "signal_date"):
        work[column] = pd.to_datetime(work[column], errors="raise").dt.strftime("%Y-%m-%d")
    raw_score_date = work.get(
        "breakout_quality_score_date",
        pd.Series("", index=work.index, dtype="object"),
    ).fillna("").astype(str).str.strip()
    parsed = pd.to_datetime(raw_score_date, errors="coerce")
    invalid = raw_score_date.ne("") & parsed.isna()
    if bool(invalid.any()):
        raise ValueError("orderable candidate含無法解析的breakout_quality_score_date")
    normalized = parsed.dt.strftime("%Y-%m-%d").fillna("")
    work["score_event_date"] = normalized.where(normalized.ne(""), work["signal_date"])
    return work


def _score_lookup(score_table: pd.DataFrame) -> pd.DataFrame:
    scores = pd.DataFrame(score_table).copy()
    if isinstance(scores.index, pd.MultiIndex):
        scores = scores.reset_index()
    required = {"ticker", "date", "breakout_quality_score", "fold_id"}
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise ValueError(f"PIT score table缺少欄位: {missing}")
    scores["ticker"] = scores["ticker"].fillna("").astype(str).str.strip()
    scores["date"] = pd.to_datetime(scores["date"], errors="raise").dt.strftime("%Y-%m-%d")
    scores["fold_id"] = scores["fold_id"].fillna("").astype(str).str.strip()
    scores["breakout_quality_score"] = pd.to_numeric(
        scores["breakout_quality_score"], errors="raise"
    ).astype(float)
    return scores[["ticker", "date", "fold_id", "breakout_quality_score"]].rename(
        columns={
            "date": "score_event_date",
            "breakout_quality_score": "pit_score",
        }
    )


def _annotate_orderable(orderable: pd.DataFrame, score_table: pd.DataFrame) -> pd.DataFrame:
    work = _normalize_orderable(orderable)
    lookup = _score_lookup(score_table)
    if bool(lookup.duplicated(["ticker", "score_event_date"]).any()):
        raise ValueError("PIT score lookup同一ticker/score_event_date不唯一")
    joined = work.merge(
        lookup,
        on=["ticker", "score_event_date"],
        how="left",
        validate="many_to_one",
    )
    joined["fold_id"] = joined["fold_id"].fillna("").astype(str)
    joined["pit_score_available"] = joined["fold_id"].ne("") & pd.to_numeric(
        joined["pit_score"], errors="coerce"
    ).map(math.isfinite)
    trade_date = pd.to_datetime(joined["trade_date"], errors="raise")
    score_date = pd.to_datetime(joined["score_event_date"], errors="raise")
    joined["score_age_days"] = (trade_date - score_date).dt.days.astype(int)
    if bool((joined["score_age_days"] < 0).any()):
        raise ValueError("orderable candidate的score_event_date晚於trade_date")
    return joined


def _day_fold_summary(annotated: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trade_date, day in annotated.groupby("trade_date", sort=True):
        scored = day[day["pit_score_available"]].copy()
        counts = scored["fold_id"].value_counts().sort_index()
        n = int(len(scored))
        all_pairs = n * (n - 1) // 2
        same_pairs = int(sum(int(value) * (int(value) - 1) // 2 for value in counts.tolist()))
        cross_pairs = all_pairs - same_pairs
        rows.append(
            {
                "trade_date": str(trade_date),
                "orderable_count": int(len(day)),
                "scored_count": n,
                "score_coverage": float(n / len(day)) if len(day) else None,
                "fold_count": int(len(counts)),
                "mixed_fold_day": bool(len(counts) > 1),
                "fold_ids": ";".join(counts.index.astype(str).tolist()),
                "all_scored_pairs": int(all_pairs),
                "cross_fold_pairs": int(cross_pairs),
                "cross_fold_pair_share": float(cross_pairs / all_pairs) if all_pairs else None,
                "mean_score_age_days": float(scored["score_age_days"].mean()) if n else None,
            }
        )
    return pd.DataFrame(rows)


def _fold_transition_dates(score_table: pd.DataFrame) -> list[pd.Timestamp]:
    scores = _score_lookup(score_table)
    starts = (
        scores.groupby("fold_id", sort=True)["score_event_date"]
        .min()
        .map(pd.Timestamp)
        .sort_values()
        .tolist()
    )
    return starts[1:]


def _nearest_transition_distance(date_text: str, transitions: list[pd.Timestamp]) -> int | None:
    if not transitions:
        return None
    date = pd.Timestamp(date_text)
    return int(min(abs((date - transition).days) for transition in transitions))


def _occurrence_fold_map(annotated: pd.DataFrame) -> pd.DataFrame:
    keys = ["trade_date", "ticker", "signal_date"]
    rows: list[dict[str, Any]] = []
    for key, frame in annotated.groupby(keys, sort=False, dropna=False):
        folds = sorted({value for value in frame["fold_id"].astype(str) if value})
        score_dates = sorted({value for value in frame["score_event_date"].astype(str) if value})
        rows.append(
            {
                **dict(zip(keys, key)),
                "occurrence_fold_id": folds[0] if len(folds) == 1 else "",
                "occurrence_fold_ambiguous": bool(len(folds) > 1),
                "occurrence_score_event_date": score_dates[0] if len(score_dates) == 1 else "",
            }
        )
    return pd.DataFrame(rows)


def _annotate_trade_contributions(
    trade_contributions: pd.DataFrame,
    *,
    annotated_orderable: pd.DataFrame,
    day_summary: pd.DataFrame,
    transitions: list[pd.Timestamp],
    boundary_window_days: int,
) -> pd.DataFrame:
    trades = pd.DataFrame(trade_contributions).copy()
    if trades.empty:
        return trades
    if "signal_date" not in trades.columns:
        raise ValueError("trade contribution缺少signal_date，無法做PIT fold歸因")
    trades["entry_date"] = pd.to_datetime(trades["entry_date"], errors="raise").dt.strftime("%Y-%m-%d")
    trades["signal_date"] = pd.to_datetime(trades["signal_date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    trades["ticker"] = trades["ticker"].fillna("").astype(str).str.strip()
    day = day_summary[["trade_date", "mixed_fold_day", "fold_count", "cross_fold_pair_share"]].copy()
    trades = trades.merge(day, left_on="entry_date", right_on="trade_date", how="left", validate="many_to_one")
    occ = _occurrence_fold_map(annotated_orderable)
    trades = trades.merge(
        occ,
        left_on=["entry_date", "ticker", "signal_date"],
        right_on=["trade_date", "ticker", "signal_date"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_occ"),
    )
    trades["day_scope"] = np.where(
        trades["mixed_fold_day"].fillna(False).astype(bool),
        "mixed_fold",
        np.where(trades["mixed_fold_day"].isna(), "no_orderable_day", "single_fold"),
    )
    trades["nearest_fold_transition_days"] = trades["entry_date"].map(
        lambda value: _nearest_transition_distance(str(value), transitions)
    )
    trades["fold_boundary_window"] = trades["nearest_fold_transition_days"].map(
        lambda value: bool(value is not None and value <= boundary_window_days)
    )
    return trades.drop(columns=[column for column in ("trade_date", "trade_date_occ") if column in trades.columns])


def _exclusive_scope_summary(trades: pd.DataFrame, scope_column: str) -> list[dict[str, Any]]:
    exclusive = trades[trades["category"].isin(["candidate_only", "comparator_only"])].copy()
    if exclusive.empty:
        return []
    rows: list[dict[str, Any]] = []
    for scope, frame in exclusive.groupby(scope_column, dropna=False, sort=True):
        candidate = frame[frame["category"] == "candidate_only"]
        baseline = frame[frame["category"] == "comparator_only"]
        cand_r = pd.to_numeric(candidate["candidate_r"], errors="coerce").fillna(0.0)
        base_r = pd.to_numeric(baseline["comparator_r"], errors="coerce").fillna(0.0)
        cand_winner_r = float(cand_r[cand_r > 0].sum())
        base_winner_r = float(base_r[base_r > 0].sum())
        cand_loser_abs = float((-cand_r[cand_r < 0]).sum())
        base_loser_abs = float((-base_r[base_r < 0]).sum())
        rows.append(
            {
                "scope": str(scope),
                "exclusive_trade_count": int(len(frame)),
                "candidate_only_count": int(len(candidate)),
                "baseline_only_count": int(len(baseline)),
                "selection_delta_r": float(pd.to_numeric(frame["r_delta"], errors="coerce").fillna(0.0).sum()),
                "candidate_only_winner_count": int((cand_r > 0).sum()),
                "baseline_only_winner_count": int((base_r > 0).sum()),
                "candidate_only_winner_r": cand_winner_r,
                "baseline_only_winner_r": base_winner_r,
                "winner_r_contribution_delta": cand_winner_r - base_winner_r,
                "loser_r_contribution_delta": base_loser_abs - cand_loser_abs,
            }
        )
    return rows


def _fold_score_rows(score_table: pd.DataFrame, annotated: pd.DataFrame) -> list[dict[str, Any]]:
    all_scores = _score_lookup(score_table)
    orderable = annotated[annotated["pit_score_available"]].copy()
    rows: list[dict[str, Any]] = []
    fold_ids = sorted(set(all_scores["fold_id"]) | set(orderable["fold_id"]))
    for fold_id in fold_ids:
        full = all_scores[all_scores["fold_id"] == fold_id]["pit_score"]
        runtime = orderable[orderable["fold_id"] == fold_id]["pit_score"]
        rows.append(
            {
                "fold_id": str(fold_id),
                "pit_group_count": int(len(full)),
                "pit_score_mean": float(full.mean()) if len(full) else None,
                "pit_score_std": float(full.std(ddof=0)) if len(full) else None,
                "orderable_count": int(len(runtime)),
                "orderable_score_mean": float(runtime.mean()) if len(runtime) else None,
                "orderable_score_std": float(runtime.std(ddof=0)) if len(runtime) else None,
            }
        )
    return rows


def build_pit_fold_runtime_attribution(
    *,
    score_table: pd.DataFrame,
    pit_audit: dict[str, Any],
    orderable_candidates: pd.DataFrame,
    trade_contributions: pd.DataFrame,
    fold_boundary_window_days: int,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    annotated = _annotate_orderable(orderable_candidates, score_table)
    days = _day_fold_summary(annotated)
    transitions = _fold_transition_dates(score_table)
    trades = _annotate_trade_contributions(
        trade_contributions,
        annotated_orderable=annotated,
        day_summary=days,
        transitions=transitions,
        boundary_window_days=fold_boundary_window_days,
    )
    mixed_rows = _exclusive_scope_summary(trades, "day_scope")
    boundary_rows = _exclusive_scope_summary(trades, "fold_boundary_window")
    folds = _fold_score_rows(score_table, annotated)

    total_rows = int(len(annotated))
    scored_rows = int(annotated["pit_score_available"].sum())
    total_pairs = int(days["all_scored_pairs"].sum()) if not days.empty else 0
    cross_pairs = int(days["cross_fold_pairs"].sum()) if not days.empty else 0
    exclusive = trades[trades["category"].isin(["candidate_only", "comparator_only"])] if not trades.empty else trades
    total_selection_delta = float(pd.to_numeric(exclusive.get("r_delta", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()) if not exclusive.empty else 0.0

    mixed_map = {row["scope"]: row for row in mixed_rows}
    mixed_delta = _finite(dict(mixed_map.get("mixed_fold") or {}).get("selection_delta_r"))
    single_delta = _finite(dict(mixed_map.get("single_fold") or {}).get("selection_delta_r"))
    boundary_map = {str(row["scope"]): row for row in boundary_rows}
    boundary_delta = _finite(dict(boundary_map.get("True") or {}).get("selection_delta_r"))
    outside_delta = _finite(dict(boundary_map.get("False") or {}).get("selection_delta_r"))

    payload = {
        "pit_model_audit": {
            "fold_drift": dict(pit_audit.get("fold_drift") or {}),
            "fold_metrics": list(pit_audit.get("fold_metrics") or []),
        },
        "runtime_mixing": {
            "orderable_row_count": total_rows,
            "pit_scored_row_count": scored_rows,
            "pit_score_coverage": float(scored_rows / total_rows) if total_rows else None,
            "orderable_day_count": int(len(days)),
            "mixed_fold_day_count": int(days["mixed_fold_day"].sum()) if not days.empty else 0,
            "mixed_fold_day_rate": float(days["mixed_fold_day"].mean()) if not days.empty else None,
            "scored_pair_count": total_pairs,
            "cross_fold_pair_count": cross_pairs,
            "cross_fold_pair_share": float(cross_pairs / total_pairs) if total_pairs else None,
            "score_age_mean_days": float(annotated.loc[annotated["pit_score_available"], "score_age_days"].mean()) if scored_rows else None,
            "score_age_median_days": float(annotated.loc[annotated["pit_score_available"], "score_age_days"].median()) if scored_rows else None,
            "fold_transition_dates": [value.strftime("%Y-%m-%d") for value in transitions],
            "fold_boundary_window_days": int(fold_boundary_window_days),
        },
        "exclusive_selection": {
            "total_selection_delta_r": total_selection_delta,
            "by_day_scope": mixed_rows,
            "by_fold_boundary_window": boundary_rows,
            "mixed_fold_selection_delta_r": mixed_delta,
            "single_fold_selection_delta_r": single_delta,
            "boundary_window_selection_delta_r": boundary_delta,
            "outside_boundary_selection_delta_r": outside_delta,
        },
        "interpretation": {
            "causal_conclusion_allowed": False,
            "mixed_fold_loss_concentration_observed": bool(
                total_selection_delta < 0.0 and mixed_delta is not None and mixed_delta < 0.0
            ),
            "boundary_loss_concentration_observed": bool(
                total_selection_delta < 0.0 and boundary_delta is not None and boundary_delta < 0.0
            ),
            "next_decision_requires_observed_concentration": True,
        },
        "fold_score_summary": folds,
    }
    frames = {
        "orderable_fold_attribution": annotated,
        "daily_fold_mixing": days,
        "exclusive_trade_fold_attribution": trades,
        "fold_score_summary": pd.DataFrame(folds),
        "exclusive_by_day_scope": pd.DataFrame(mixed_rows),
        "exclusive_by_fold_boundary": pd.DataFrame(boundary_rows),
    }
    return payload, frames


def _pair_result(
    *,
    root: Path,
    result: dict[str, Any],
    baseline: StrategyCompareArmArtifacts,
    candidate: StrategyCompareArmArtifacts,
    boundary_window_days: int,
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
    identity = _candidate_pit_identity(result, candidate.arm_id)
    contract = load_selection_point_in_time_ranking_contract(
        str(root),
        identity["filter_id"],
        identity["model_architecture"],
        identity["experiment_profile"],
    )
    score_table = load_selection_point_in_time_score_table(
        str(root),
        identity["filter_id"],
        identity["model_architecture"],
        identity["experiment_profile"],
    ).reset_index()
    attribution, frames = build_pit_fold_runtime_attribution(
        score_table=score_table,
        pit_audit=contract.audit,
        orderable_candidates=_read_csv(candidate.orderable_path),
        trade_contributions=structural_frames["trade_contributions"],
        fold_boundary_window_days=boundary_window_days,
    )
    pair = {
        "baseline_arm_id": baseline.arm_id,
        "candidate_arm_id": candidate.arm_id,
        "pit_identity": identity,
        "strategy": {
            "baseline_summary": dict(baseline.summary),
            "candidate_summary": dict(candidate.summary),
            "exclusive_selection_delta_r": _finite(
                dict(structural.get("trade_contribution") or {}).get("exclusive_selection_delta_r")
            ),
        },
        **attribution,
    }
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
        render_title("Selection PIT Fold Drift／Mixed-fold Runtime 歸因"),
        render_key_values((
            ("Audit", f"AUD-{payload['audit_id']}"),
            ("策略比較期間", f"{meta.get('comparison_period', {}).get('start', '')} ～ {meta.get('comparison_period', {}).get('end', '')}"),
            ("Baseline", meta.get("baseline_arm_id", "-")),
            ("Candidates", ", ".join(meta.get("candidate_arm_ids") or [])),
            ("Fold boundary window", f"±{meta.get('fold_boundary_window_days', 0)} calendar days"),
            ("Strategy compare", meta.get("strategy_compare_run", "-")),
            ("契約", "只讀既有replay與PIT工件；不重跑portfolio、不校正score、不改selector／params／training"),
        )),
    ]
    for index, pair in enumerate(payload.get("comparisons") or [], start=1):
        candidate = pair["candidate_arm_id"]
        baseline = pair["baseline_arm_id"]
        drift = dict(pair.get("pit_model_audit", {}).get("fold_drift") or {})
        mixing = dict(pair.get("runtime_mixing") or {})
        selection = dict(pair.get("exclusive_selection") or {})
        by_scope = list(selection.get("by_day_scope") or [])
        by_boundary = list(selection.get("by_fold_boundary_window") or [])
        lines.extend((
            render_section(f"{index}. {candidate} vs {baseline}"),
            render_key_values((
                ("PIT drift flag", str(bool(drift.get("drift_flag")))),
                ("Max adjacent mean shift", _fmt(drift.get("max_adjacent_mean_shift_in_pooled_std"), unit=" pooled SD")),
                ("Flagged folds", ", ".join(drift.get("flagged_folds") or []) or "-"),
                ("Exclusive selection R", _fmt(selection.get("total_selection_delta_r"), unit=" R", signed=True)),
            )),
            render_section("Runtime fold mixing"),
            render_table(
                ("指標", "結果"),
                (
                    ("Orderable PIT score coverage", _fmt((mixing.get("pit_score_coverage") or 0.0) * 100.0, unit="%")),
                    ("Mixed-fold days", f"{int(mixing.get('mixed_fold_day_count', 0))}/{int(mixing.get('orderable_day_count', 0))} ({_fmt((mixing.get('mixed_fold_day_rate') or 0.0) * 100.0, unit='%')})"),
                    ("Cross-fold pair share", _fmt((mixing.get("cross_fold_pair_share") or 0.0) * 100.0, unit="%")),
                    ("Score age", f"mean={_fmt(mixing.get('score_age_mean_days'), unit='日')} / median={_fmt(mixing.get('score_age_median_days'), unit='日')}"),
                ),
            ),
            render_section("Exclusive selection R：single-fold vs mixed-fold days"),
            render_table(
                ("Scope", "Trades", "Selection ΔR", "Winner ΔR", "Loser ΔR"),
                tuple(
                    (
                        row.get("scope", "-"),
                        int(row.get("exclusive_trade_count", 0)),
                        _fmt(row.get("selection_delta_r"), unit=" R", signed=True),
                        _fmt(row.get("winner_r_contribution_delta"), unit=" R", signed=True),
                        _fmt(row.get("loser_r_contribution_delta"), unit=" R", signed=True),
                    )
                    for row in by_scope
                ),
            ),
            render_section("Exclusive selection R：PIT fold boundary window"),
            render_table(
                ("Boundary window", "Trades", "Selection ΔR", "Winner ΔR", "Loser ΔR"),
                tuple(
                    (
                        "within" if str(row.get("scope")) == "True" else "outside",
                        int(row.get("exclusive_trade_count", 0)),
                        _fmt(row.get("selection_delta_r"), unit=" R", signed=True),
                        _fmt(row.get("winner_r_contribution_delta"), unit=" R", signed=True),
                        _fmt(row.get("loser_r_contribution_delta"), unit=" R", signed=True),
                    )
                    for row in by_boundary
                ),
            ),
            "判讀限制：此Audit只做條件歸因。Mixed-fold／fold-boundary與負R共現不等於因果；只有損失明顯集中時，才值得建立後續Selection-only normalization／runtime假說。",
        ))
    lines.extend((
        render_section(f"{len(payload.get('comparisons') or []) + 1}. 使用限制"),
        "不得以本Audit的Selection結果直接擬合OOS、校正PIT score、修改MR-12B loss／epoch或更動selector。若mixed-fold不能解釋winner capture loss，下一步回到Target／realized outcome語意研究。",
    ))
    return "\n\n".join(lines)


def run_pit_fold_runtime_attribution_audit(
    definition: AuditDefinition,
    *,
    project_root: Path = PROJECT_ROOT,
    quiet: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    run_dir, result, baseline_id, candidate_ids, boundary_window_days, focus_year, top_month_count, top_trade_count = _resolve_sources(
        root, definition
    )
    status = collect_pit_fold_runtime_attribution_status(definition, project_root=root)
    if status.get("status") != "READY":
        raise RuntimeError(str(status.get("reason") or "PIT fold runtime Audit尚未READY"))

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
            root=root,
            result=result,
            baseline=baseline,
            candidate=candidate,
            boundary_window_days=boundary_window_days,
            focus_year=focus_year,
            top_month_count=top_month_count,
            top_trade_count=top_trade_count,
        )
        comparisons.append(pair)
        frames_by_pair[f"{candidate_id}_vs_{baseline_id}"] = frames

    payload = _json_native({
        "schema_version": AUDIT_RESULT_SCHEMA_VERSION,
        "created_at": get_taipei_now().isoformat(),
        "audit_id": definition.audit_id,
        "audit_type": definition.audit_type,
        "config": definition.as_dict(),
        "metadata": {
            "strategy_compare_run": project_relative_display_path(run_dir, project_root=root),
            "strategy_compare_config_fingerprint": result.get("config_fingerprint"),
            "comparison_period": dict(result.get("comparison_period") or {}),
            "baseline_arm_id": baseline_id,
            "candidate_arm_ids": list(candidate_ids),
            "fold_boundary_window_days": boundary_window_days,
            "read_only": True,
            "portfolio_replay_executed": False,
            "training_performed": False,
            "score_calibration_performed": False,
            "parameter_optimization_performed": False,
        },
        "comparisons": comparisons,
    })

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
            (("Audit Markdown", report_path), ("Audit JSON", json_path), ("最新Audit", latest_dir)),
            project_root=root,
        )
    return payload


__all__ = [
    "AUDIT_RESULT_SCHEMA_VERSION",
    "build_pit_fold_runtime_attribution",
    "collect_pit_fold_runtime_attribution_status",
    "run_pit_fold_runtime_attribution_audit",
]
