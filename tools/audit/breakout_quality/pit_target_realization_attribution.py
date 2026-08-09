"""Read-only PIT Target→realized-R attribution for configured strategy arms.

The Audit explains whether Selection PIT direct-deployment losses are concentrated
in older signal→entry ages, or persist even among fresher executed trades.  It
uses only completed strategy replay, validated PIT score tables, and the existing
continuous target arrays.  It never replays, trains, calibrates, or changes
runtime semantics.
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
from filters.breakout_quality.continuous_target import load_validated_continuous_target_arrays
from filters.breakout_quality.ranking_score_store import (
    load_selection_point_in_time_ranking_contract,
    load_selection_point_in_time_score_table,
)
from tools.audit.breakout_quality.c15_strategy_attribution import (
    build_strategy_attribution_pair_payload,
)
from tools.audit.breakout_quality.pit_fold_runtime_attribution import _candidate_pit_identity
from tools.audit.sources.strategy_compare import (
    StrategyCompareArmArtifacts,
    resolve_arm_artifacts,
    resolve_strategy_compare_run_selector,
)

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
        age_quantile_groups = int(dimensions.get("score_age_quantile_groups"))
        focus_year = int(dimensions.get("focus_year"))
        top_month_count = int(dimensions.get("top_month_count"))
        top_trade_count = int(dimensions.get("top_trade_count"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{definition.audit_id}.dimensions必須是整數") from exc
    if age_quantile_groups < 2 or age_quantile_groups > 10:
        raise ValueError(f"{definition.audit_id}.score_age_quantile_groups必須介於2與10")
    if focus_year < 1900 or top_month_count < 1 or top_trade_count < 1:
        raise ValueError(f"{definition.audit_id}.attribution dimensions超出合法範圍")
    return baseline, candidates, age_quantile_groups, focus_year, top_month_count, top_trade_count


def _resolve_sources(
    root: Path,
    definition: AuditDefinition,
) -> tuple[Path, dict[str, Any], str, tuple[str, ...], int, int, int, int]:
    baseline, candidates, age_quantile_groups, focus_year, top_month_count, top_trade_count = _validate_definition(definition)
    run_dir, result = resolve_strategy_compare_run_selector(
        root,
        dict(definition.source),
        audit_id=definition.audit_id,
        required_arm_id=baseline,
    )
    scenarios = dict(result.get("scenarios") or {})
    for candidate_id in candidates:
        if candidate_id not in scenarios:
            raise ValueError(f"strategy_compare run不存在candidate arm: {candidate_id}")
        _candidate_pit_identity(result, candidate_id)
    return (
        run_dir,
        result,
        baseline,
        candidates,
        age_quantile_groups,
        focus_year,
        top_month_count,
        top_trade_count,
    )


def _required_paths(
    *, baseline: StrategyCompareArmArtifacts, candidate: StrategyCompareArmArtifacts
) -> dict[str, Path]:
    return {
        "baseline_trades": baseline.trades_path,
        "baseline_equity": baseline.equity_path,
        "baseline_capacity": baseline.capacity_path,
        "baseline_selected": baseline.selected_path,
        "candidate_trades": candidate.trades_path,
        "candidate_equity": candidate.equity_path,
        "candidate_capacity": candidate.capacity_path,
        "candidate_selected": candidate.selected_path,
    }


def collect_pit_target_realization_attribution_status(
    definition: AuditDefinition,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    try:
        (
            run_dir,
            result,
            baseline_id,
            candidate_ids,
            age_quantile_groups,
            _focus_year,
            _top_month_count,
            _top_trade_count,
        ) = _resolve_sources(root, definition)
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
            load_validated_continuous_target_arrays(
                root,
                identity["filter_id"],
                target_id=contract.continuous_target_id,
            )
            pit_identities[candidate_id] = {
                **identity,
                "continuous_target_id": contract.continuous_target_id,
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
                "score_age_quantile_groups": age_quantile_groups,
                "pit_identities": pit_identities,
            },
        }
    except (FileNotFoundError, ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        return {
            "status": "BLOCKED",
            "reason": str(exc),
            "source": {"display": "strategy_compare / Selection PIT"},
        }


def _spearman(frame: pd.DataFrame, left: str, right: str) -> float | None:
    if frame.empty or left not in frame.columns or right not in frame.columns:
        return None
    x = pd.to_numeric(frame[left], errors="coerce")
    y = pd.to_numeric(frame[right], errors="coerce")
    valid = np.isfinite(x.to_numpy(dtype=np.float64)) & np.isfinite(y.to_numpy(dtype=np.float64))
    if int(valid.sum()) < 2:
        return None
    xr = x.loc[valid].rank(method="average")
    yr = y.loc[valid].rank(method="average")
    value = xr.corr(yr)
    return float(value) if value is not None and math.isfinite(float(value)) else None


def _side_summary(frame: pd.DataFrame) -> dict[str, Any]:
    work = pd.DataFrame(frame).copy()
    covered = work[
        np.isfinite(pd.to_numeric(work.get("model_score"), errors="coerce"))
        & np.isfinite(pd.to_numeric(work.get("target_raw_r"), errors="coerce"))
        & np.isfinite(pd.to_numeric(work.get("actual_r"), errors="coerce"))
    ].copy()
    result: dict[str, Any] = {
        "trade_count": int(len(work)),
        "covered_trade_count": int(len(covered)),
        "coverage_rate": float(len(covered) / len(work)) if len(work) else None,
    }
    if covered.empty:
        return result
    target = pd.to_numeric(covered["target_raw_r"], errors="coerce")
    actual = pd.to_numeric(covered["actual_r"], errors="coerce")
    score = pd.to_numeric(covered["model_score"], errors="coerce")
    age = pd.to_numeric(covered["score_age_days"], errors="coerce")
    gap = target - actual
    result.update({
        "avg_model_score": float(score.mean()),
        "avg_target_r": float(target.mean()),
        "avg_realized_r": float(actual.mean()),
        "win_rate": float((actual > 0.0).mean()),
        "avg_score_age_days": float(age.mean()),
        "median_score_age_days": float(age.median()),
        "score_vs_target_spearman": _spearman(covered, "model_score", "target_raw_r"),
        "target_vs_realized_r_spearman": _spearman(covered, "target_raw_r", "actual_r"),
        "score_vs_realized_r_spearman": _spearman(covered, "model_score", "actual_r"),
        "age_vs_realized_r_spearman": _spearman(covered, "score_age_days", "actual_r"),
        "age_vs_realization_gap_spearman": _spearman(
            covered.assign(realization_gap_r=gap), "score_age_days", "realization_gap_r"
        ),
    })
    return result


def build_pit_target_realization_attribution(
    *,
    score_table: pd.DataFrame,
    target_raw_r: np.ndarray,
    target_valid_mask: np.ndarray,
    trade_contributions: pd.DataFrame,
    score_age_quantile_groups: int,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    score = pd.DataFrame(score_table).reset_index(drop=True).copy()
    required_score = {"ticker", "date", "group_index", "breakout_quality_score"}
    missing_score = sorted(required_score.difference(score.columns))
    if missing_score:
        raise ValueError(f"PIT score table缺少欄位: {missing_score}")
    score["ticker"] = score["ticker"].fillna("").astype(str).str.strip()
    score["date"] = pd.to_datetime(score["date"], errors="raise").dt.strftime("%Y-%m-%d")
    score["group_index"] = pd.to_numeric(score["group_index"], errors="raise").astype(np.int64)
    score["breakout_quality_score"] = pd.to_numeric(
        score["breakout_quality_score"], errors="raise"
    ).astype(float)
    if score.duplicated(["ticker", "date"]).any():
        raise ValueError("PIT score table同一ticker/date重複")

    target = np.asarray(target_raw_r, dtype=np.float64)
    valid_mask = np.asarray(target_valid_mask, dtype=bool)
    if target.ndim != 1 or valid_mask.ndim != 1 or target.shape != valid_mask.shape:
        raise ValueError("continuous target arrays shape不合法")
    gids = score["group_index"].to_numpy(dtype=np.int64)
    if bool((gids < 0).any()) or bool((gids >= len(target)).any()):
        raise ValueError("PIT score group_index超出continuous target範圍")
    score["target_raw_r"] = target[gids]
    score["target_valid"] = valid_mask[gids]
    score_lookup = score.rename(
        columns={
            "date": "signal_date",
            "breakout_quality_score": "model_score",
        }
    )[["ticker", "signal_date", "group_index", "model_score", "target_raw_r", "target_valid"]]

    trades = pd.DataFrame(trade_contributions).copy()
    required_trade = {
        "category",
        "ticker",
        "entry_date",
        "signal_date",
        "candidate_r",
        "comparator_r",
        "r_delta",
    }
    missing_trade = sorted(required_trade.difference(trades.columns))
    if missing_trade:
        raise ValueError(f"trade contributions缺少欄位: {missing_trade}")
    trades = trades[trades["category"].isin(["candidate_only", "comparator_only"])].copy()
    trades["ticker"] = trades["ticker"].fillna("").astype(str).str.strip()
    for column in ("entry_date", "signal_date"):
        trades[column] = pd.to_datetime(trades[column], errors="raise").dt.strftime("%Y-%m-%d")
    trades["actual_r"] = np.where(
        trades["category"].eq("candidate_only"),
        pd.to_numeric(trades["candidate_r"], errors="coerce"),
        pd.to_numeric(trades["comparator_r"], errors="coerce"),
    )
    trades["selection_side"] = np.where(
        trades["category"].eq("candidate_only"), "candidate_only", "baseline_only"
    )
    entry_ts = pd.to_datetime(trades["entry_date"], errors="raise")
    signal_ts = pd.to_datetime(trades["signal_date"], errors="raise")
    trades["score_age_days"] = (entry_ts - signal_ts).dt.days.astype(int)
    if bool((trades["score_age_days"] < 0).any()):
        raise ValueError("exclusive trade出現entry_date早於signal_date")
    merged = trades.merge(
        score_lookup,
        how="left",
        on=["ticker", "signal_date"],
        validate="many_to_one",
    )
    merged["target_match"] = merged["target_valid"].fillna(False).astype(bool) & np.isfinite(
        pd.to_numeric(merged["target_raw_r"], errors="coerce")
    )
    merged["score_match"] = np.isfinite(pd.to_numeric(merged["model_score"], errors="coerce"))
    merged["winner"] = pd.to_numeric(merged["actual_r"], errors="coerce") > 0.0
    merged["realization_gap_r"] = (
        pd.to_numeric(merged["target_raw_r"], errors="coerce")
        - pd.to_numeric(merged["actual_r"], errors="coerce")
    )

    covered = merged[merged["target_match"] & merged["score_match"]].copy()
    age_bucket_rows: list[dict[str, Any]] = []
    if not covered.empty:
        unique_ages = int(covered["score_age_days"].nunique())
        q = min(int(score_age_quantile_groups), max(1, unique_ages))
        if q > 1:
            bucket = pd.qcut(
                covered["score_age_days"],
                q=q,
                labels=False,
                duplicates="drop",
            )
            covered["score_age_bucket"] = pd.to_numeric(bucket, errors="coerce").astype("Int64")
        else:
            covered["score_age_bucket"] = 0
        bucket_ids = sorted(int(value) for value in covered["score_age_bucket"].dropna().unique())
        max_bucket = max(bucket_ids) if bucket_ids else 0
        for bucket_id in bucket_ids:
            group = covered[covered["score_age_bucket"] == bucket_id]
            baseline = group[group["selection_side"] == "baseline_only"]
            candidate = group[group["selection_side"] == "candidate_only"]
            baseline_r = float(pd.to_numeric(baseline["actual_r"], errors="coerce").sum()) if not baseline.empty else 0.0
            candidate_r = float(pd.to_numeric(candidate["actual_r"], errors="coerce").sum()) if not candidate.empty else 0.0
            age_bucket_rows.append({
                "bucket": f"Q{bucket_id + 1}",
                "age_rank": "oldest" if bucket_id == max_bucket else "youngest" if bucket_id == 0 else "middle",
                "trade_count": int(len(group)),
                "age_min_days": int(group["score_age_days"].min()),
                "age_median_days": float(group["score_age_days"].median()),
                "age_max_days": int(group["score_age_days"].max()),
                "baseline_trade_count": int(len(baseline)),
                "candidate_trade_count": int(len(candidate)),
                "baseline_target_mean_r": float(pd.to_numeric(baseline["target_raw_r"], errors="coerce").mean()) if not baseline.empty else None,
                "candidate_target_mean_r": float(pd.to_numeric(candidate["target_raw_r"], errors="coerce").mean()) if not candidate.empty else None,
                "baseline_realized_mean_r": float(pd.to_numeric(baseline["actual_r"], errors="coerce").mean()) if not baseline.empty else None,
                "candidate_realized_mean_r": float(pd.to_numeric(candidate["actual_r"], errors="coerce").mean()) if not candidate.empty else None,
                "baseline_win_rate": float(baseline["winner"].mean()) if not baseline.empty else None,
                "candidate_win_rate": float(candidate["winner"].mean()) if not candidate.empty else None,
                "selection_delta_r": candidate_r - baseline_r,
            })

    baseline_rows = covered[covered["selection_side"] == "baseline_only"]
    candidate_rows = covered[covered["selection_side"] == "candidate_only"]
    payload = {
        "exclusive_trade_count": int(len(merged)),
        "covered_trade_count": int(len(covered)),
        "coverage_rate": float(len(covered) / len(merged)) if len(merged) else None,
        "baseline_only": _side_summary(baseline_rows),
        "candidate_only": _side_summary(candidate_rows),
        "age_buckets": age_bucket_rows,
        "semantic_boundary": {
            "actual_trades_only": True,
            "unselected_counterfactual_r_available": False,
            "age_is_calendar_days": True,
            "target_anchor": "original_signal_event",
        },
    }
    return payload, {
        "exclusive_trade_target_realization": merged,
        "exclusive_trade_target_realization_covered": covered,
        "score_age_buckets": pd.DataFrame(age_bucket_rows),
    }


def _pair_result(
    *,
    root: Path,
    result: dict[str, Any],
    baseline: StrategyCompareArmArtifacts,
    candidate: StrategyCompareArmArtifacts,
    age_quantile_groups: int,
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
    _target_manifest, target_raw_r, target_valid = load_validated_continuous_target_arrays(
        root,
        identity["filter_id"],
        target_id=contract.continuous_target_id,
    )
    attribution, frames = build_pit_target_realization_attribution(
        score_table=score_table,
        target_raw_r=target_raw_r,
        target_valid_mask=target_valid,
        trade_contributions=structural_frames["trade_contributions"],
        score_age_quantile_groups=age_quantile_groups,
    )
    pair = {
        "baseline_arm_id": baseline.arm_id,
        "candidate_arm_id": candidate.arm_id,
        "pit_identity": {**identity, "continuous_target_id": contract.continuous_target_id},
        "strategy": {
            "baseline_summary": dict(baseline.summary),
            "candidate_summary": dict(candidate.summary),
            "exclusive_selection_delta_r": _finite(
                dict(structural.get("trade_contribution") or {}).get("exclusive_selection_delta_r")
            ),
        },
        "target_realization": attribution,
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
        render_title("Selection PIT Target→Realized R／Score-age 歸因"),
        render_key_values((
            ("Audit", f"AUD-{payload['audit_id']}"),
            ("策略比較期間", f"{meta.get('comparison_period', {}).get('start', '')} ～ {meta.get('comparison_period', {}).get('end', '')}"),
            ("Baseline", meta.get("baseline_arm_id", "-")),
            ("Candidates", ", ".join(meta.get("candidate_arm_ids") or [])),
            ("Age quantiles", str(meta.get("score_age_quantile_groups", "-"))),
            ("Strategy compare", meta.get("strategy_compare_run", "-")),
            ("契約", "只讀已成交exclusive trades與PIT/Target工件；不建立未成交counterfactual、不重跑portfolio、不改模型／selector／params"),
        )),
    ]
    for index, pair in enumerate(payload.get("comparisons") or [], start=1):
        candidate = pair["candidate_arm_id"]
        baseline = pair["baseline_arm_id"]
        target = dict(pair.get("target_realization") or {})
        b = dict(target.get("baseline_only") or {})
        c = dict(target.get("candidate_only") or {})
        buckets = list(target.get("age_buckets") or [])
        lines.extend((
            render_section(f"{index}. {candidate} vs {baseline}"),
            render_key_values((
                ("Exclusive selection R", _fmt((pair.get("strategy") or {}).get("exclusive_selection_delta_r"), unit=" R", signed=True)),
                ("Target/score coverage", _fmt((target.get("coverage_rate") or 0.0) * 100.0, unit="%")),
            )),
            render_section("Actual exclusive trades：Target／Score vs Realized R"),
            render_table(
                ("指標", f"{baseline}-only", f"{candidate}-only", "差異"),
                (
                    ("Trades", int(b.get("trade_count", 0)), int(c.get("trade_count", 0)), int(c.get("trade_count", 0)) - int(b.get("trade_count", 0))),
                    ("平均Target R", _fmt(b.get("avg_target_r"), unit=" R"), _fmt(c.get("avg_target_r"), unit=" R"), _fmt((_finite(c.get("avg_target_r")) or 0.0) - (_finite(b.get("avg_target_r")) or 0.0), unit=" R", signed=True)),
                    ("平均Realized R", _fmt(b.get("avg_realized_r"), unit=" R"), _fmt(c.get("avg_realized_r"), unit=" R"), _fmt((_finite(c.get("avg_realized_r")) or 0.0) - (_finite(b.get("avg_realized_r")) or 0.0), unit=" R", signed=True)),
                    ("Win rate", _fmt((_finite(b.get("win_rate")) or 0.0) * 100.0, unit="%"), _fmt((_finite(c.get("win_rate")) or 0.0) * 100.0, unit="%"), _fmt(((_finite(c.get("win_rate")) or 0.0) - (_finite(b.get("win_rate")) or 0.0)) * 100.0, unit="pp", signed=True)),
                    ("Score↔Target rho", _fmt(b.get("score_vs_target_spearman")), _fmt(c.get("score_vs_target_spearman")), "-"),
                    ("Target↔Realized rho", _fmt(b.get("target_vs_realized_r_spearman")), _fmt(c.get("target_vs_realized_r_spearman")), "-"),
                    ("Score↔Realized rho", _fmt(b.get("score_vs_realized_r_spearman")), _fmt(c.get("score_vs_realized_r_spearman")), "-"),
                    ("Age↔Realized rho", _fmt(b.get("age_vs_realized_r_spearman")), _fmt(c.get("age_vs_realized_r_spearman")), "-"),
                    ("Age↔(Target−R) rho", _fmt(b.get("age_vs_realization_gap_spearman")), _fmt(c.get("age_vs_realization_gap_spearman")), "-"),
                    ("Score age median", _fmt(b.get("median_score_age_days"), unit="日"), _fmt(c.get("median_score_age_days"), unit="日"), "-"),
                ),
            ),
            render_section("依 signal→entry age quantile 分層"),
            render_table(
                ("Age bucket", "Age days", f"{baseline} n", f"{candidate} n", f"{baseline} Target", f"{candidate} Target", f"{baseline} Realized", f"{candidate} Realized", "Selection ΔR"),
                tuple(
                    (
                        row.get("bucket", "-"),
                        f"{row.get('age_min_days', '-')}/{_fmt(row.get('age_median_days'), digits=1)}/{row.get('age_max_days', '-')} (min/med/max)",
                        int(row.get("baseline_trade_count", 0)),
                        int(row.get("candidate_trade_count", 0)),
                        _fmt(row.get("baseline_target_mean_r"), unit="R"),
                        _fmt(row.get("candidate_target_mean_r"), unit="R"),
                        _fmt(row.get("baseline_realized_mean_r"), unit="R"),
                        _fmt(row.get("candidate_realized_mean_r"), unit="R"),
                        _fmt(row.get("selection_delta_r"), unit="R", signed=True),
                    )
                    for row in buckets
                ),
            ),
            "判讀：若負Selection R主要集中於較高age bucket，才支持event Target老化／延續候選語意失真假說；若各age bucket皆負，則優先視為Target公式本身與正式realized trade-path R不對齊。",
        ))
    lines.extend((
        render_section(f"{len(payload.get('comparisons') or []) + 1}. 使用限制"),
        "本Audit只觀察實際成交exclusive trades，存在portfolio selection bias；未成交與未選候選沒有counterfactual realized R，不得填0或推論其績效。結果只能形成Selection內下一個受控Target／runtime假說，不得直接回流OOS loss、epoch、threshold或selector。",
    ))
    return "\n\n".join(lines)


def run_pit_target_realization_attribution_audit(
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
        age_quantile_groups,
        focus_year,
        top_month_count,
        top_trade_count,
    ) = _resolve_sources(root, definition)
    status = collect_pit_target_realization_attribution_status(definition, project_root=root)
    if status.get("status") != "READY":
        raise RuntimeError(str(status.get("reason") or "PIT Target realization Audit尚未READY"))

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
            age_quantile_groups=age_quantile_groups,
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
            "score_age_quantile_groups": age_quantile_groups,
            "read_only": True,
            "portfolio_replay_executed": False,
            "training_performed": False,
            "counterfactual_performed": False,
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
    "build_pit_target_realization_attribution",
    "collect_pit_target_realization_attribution_status",
    "run_pit_target_realization_attribution_audit",
]
