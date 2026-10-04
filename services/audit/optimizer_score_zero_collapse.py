"""Read-only attribution of Optimizer score zero-collapse from persisted studies.

This one-time Audit intentionally consumes completed optimizer *training/full-refit*
SQLite studies only.  It never retrains, never replays OOS, and never mutates the
optimizer score contract.  Counterfactual scores are reconstructed with the canonical
``core.portfolio_stats.calc_portfolio_score`` implementation using the study-persisted
training score policy.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from statistics import median
from typing import Any, Mapping, Sequence

from config.audit import AUDIT_OUTPUT_ROOT
from core.audit_policy import AuditDefinition
from core.console_report import (
    project_relative_display_path,
    render_key_values,
    render_table,
    render_title,
)
from core.portfolio_stats import (
    calc_portfolio_score,
    calc_score_min_quarter_return_multiplier,
    calc_score_win_rate_multiplier,
)
from services.optimizer.study_utils import is_qualified_trial_value

SUPPORTED_AUDIT_TYPE = "optimizer_score_zero_collapse"
_REFIT_RE = re.compile(r"validation_budget_refit_(\d+)$")


class OptimizerScoreAuditBlocked(RuntimeError):
    pass


def _finite(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise OptimizerScoreAuditBlocked(f"trial缺少可解析{field}: {value!r}") from exc
    if not math.isfinite(number):
        raise OptimizerScoreAuditBlocked(f"trial的{field}不是有限數: {value!r}")
    return number


def _relative(path: Path, root: Path) -> str:
    return project_relative_display_path(path, project_root=root)


def _source_root(definition: AuditDefinition, root: Path) -> Path:
    raw = str(definition.source.get("path") or "").strip()
    if not raw:
        raise OptimizerScoreAuditBlocked("Audit source.path未設定")
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _manifest_path(definition: AuditDefinition, root: Path) -> Path:
    raw = str(definition.source.get("manifest_path") or "").strip()
    if not raw:
        raise OptimizerScoreAuditBlocked("Audit source.manifest_path未設定")
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OptimizerScoreAuditBlocked(f"無法讀取JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise OptimizerScoreAuditBlocked(f"JSON不是object: {path}")
    return payload


def _validate_manifest(definition: AuditDefinition, root: Path) -> dict[str, Any]:
    path = _manifest_path(definition, root)
    if not path.is_file():
        raise OptimizerScoreAuditBlocked(f"缺少canonical parameter manifest: {_relative(path, root)}")
    payload = _load_json_object(path)
    expected_parameter_set = str(definition.source.get("parameter_set_identity") or "").strip()
    expected_contract = str(definition.source.get("execution_evidence_contract") or "").strip()
    expected_protocol = str(definition.source.get("validation_protocol_id") or "").strip()
    if expected_parameter_set and str(payload.get("parameter_set") or "") != expected_parameter_set:
        raise OptimizerScoreAuditBlocked(
            f"parameter_set不一致: expected={expected_parameter_set}, actual={payload.get('parameter_set')}"
        )
    if expected_contract and str(payload.get("optimizer_execution_evidence_contract") or "") != expected_contract:
        raise OptimizerScoreAuditBlocked(
            "Optimizer execution evidence contract不一致: "
            f"expected={expected_contract}, actual={payload.get('optimizer_execution_evidence_contract') or '-'}"
        )
    if expected_protocol and str(payload.get("optimizer_search_protocol") or "") != expected_protocol:
        raise OptimizerScoreAuditBlocked(
            f"validation protocol不一致: expected={expected_protocol}, actual={payload.get('optimizer_search_protocol')}"
        )
    return payload


def _selected_n(path: Path) -> int | None:
    for part in path.parts:
        match = _REFIT_RE.fullmatch(part)
        if match:
            return int(match.group(1))
    return None


def _full_refit_db_paths(source_root: Path) -> list[Path]:
    if not source_root.is_dir():
        return []
    return sorted(
        path.resolve()
        for path in source_root.rglob("*.db")
        if any(_REFIT_RE.fullmatch(part) for part in path.parts)
        and "outer_rolling_oos" in path.parts
        and "db" in path.parts
    )


def _study_candidates(db_path: Path):
    import optuna

    storage = f"sqlite:///{db_path.resolve()}"
    summaries = list(optuna.get_all_study_summaries(storage=storage))
    candidates = []
    for summary in summaries:
        study = optuna.load_study(study_name=summary.study_name, storage=storage)
        attrs = dict(getattr(study, "user_attrs", {}) or {})
        snapshot = attrs.get("optimizer_effective_policy_snapshot")
        if not isinstance(snapshot, Mapping):
            continue
        score_policy = snapshot.get("training_score_policy")
        if not isinstance(score_policy, Mapping) or not score_policy:
            continue
        if not list(getattr(study, "trials", []) or []):
            continue
        candidates.append(study)
    if len(candidates) == 1:
        return candidates[0]
    resume_safe = [
        item for item in candidates
        if "resume_safe_v1" in str(getattr(item, "study_name", ""))
    ]
    if len(resume_safe) == 1:
        return resume_safe[0]
    names = [str(getattr(item, "study_name", "?")) for item in candidates]
    raise OptimizerScoreAuditBlocked(
        f"{db_path.name} 無法唯一解析optimizer study: {names or 'none'}"
    )


def _study_contract(study) -> tuple[dict[str, Any], dict[str, Any]]:
    study_attrs = dict(getattr(study, "user_attrs", {}) or {})
    snapshot = study_attrs.get("optimizer_effective_policy_snapshot")
    if not isinstance(snapshot, Mapping):
        raise OptimizerScoreAuditBlocked("study缺少optimizer_effective_policy_snapshot")
    score_policy = dict(snapshot.get("training_score_policy") or {})
    walk_forward = dict(snapshot.get("walk_forward_policy") or {})
    if not score_policy:
        raise OptimizerScoreAuditBlocked("study缺少persisted training_score_policy")
    # Exact historical reconstruction of this one-time Audit is only supported while
    # the full-year multiplier is disabled; its floor belongs to threshold policy and
    # was not persisted inside the old study snapshot.
    if bool(score_policy.get("SCORE_MIN_FULL_YEAR_RETURN_AMP_ENABLED", False)):
        raise OptimizerScoreAuditBlocked(
            "歷史study啟用了full-year score multiplier，但study未持久化threshold floor；禁止用今日config補值"
        )
    return score_policy, walk_forward


def _year_from_study(study, walk_forward: Mapping[str, Any]) -> int:
    raw = str(walk_forward.get("oos_start_date") or "").strip()
    if len(raw) >= 4 and raw[:4].isdigit():
        return int(raw[:4])
    for trial in list(getattr(study, "trials", []) or []):
        attrs = dict(getattr(trial, "user_attrs", {}) or {})
        value = attrs.get("search_train_end_year")
        try:
            return int(value) + 1
        except (TypeError, ValueError):
            continue
    raise OptimizerScoreAuditBlocked("study無法解析OOS year")


def _canonical_score(attrs: Mapping[str, Any], score_policy: Mapping[str, Any]) -> float:
    return float(calc_portfolio_score(
        _finite(attrs.get("pf_return"), field="pf_return"),
        _finite(attrs.get("pf_mdd"), field="pf_mdd"),
        _finite(attrs.get("m_win_rate"), field="m_win_rate"),
        _finite(attrs.get("r_squared"), field="r_squared"),
        annual_return_pct=_finite(attrs.get("annual_return_pct"), field="annual_return_pct"),
        trade_win_rate_pct=_finite(attrs.get("win_rate"), field="win_rate"),
        min_full_year_return_pct=_finite(attrs.get("min_full_year_return_pct"), field="min_full_year_return_pct"),
        min_month_return_pct=_finite(attrs.get("min_month_return_pct"), field="min_month_return_pct"),
        min_quarter_return_pct=_finite(attrs.get("min_quarter_return_pct"), field="min_quarter_return_pct"),
        total_r=_finite(attrs.get("score_total_r"), field="score_total_r"),
        median_r=_finite(attrs.get("score_median_r"), field="score_median_r"),
        score_policy=dict(score_policy),
        threshold_policy={},
    ))


def _trial_row(trial, score_policy: Mapping[str, Any], *, zero_tolerance: float, monthly_heavy_threshold: float) -> dict[str, Any] | None:
    if getattr(trial, "state", None) is None or str(getattr(trial.state, "name", trial.state)) != "COMPLETE":
        return None
    value = getattr(trial, "value", None)
    if not is_qualified_trial_value(value):
        return None
    attrs = dict(getattr(trial, "user_attrs", {}) or {})
    stored = _finite(attrs.get("base_score", value), field="base_score")
    reconstructed = _canonical_score(attrs, score_policy)
    tolerance = max(float(zero_tolerance), 1e-12 * max(1.0, abs(stored), abs(reconstructed)))
    if abs(reconstructed - stored) > tolerance:
        raise OptimizerScoreAuditBlocked(
            f"trial T{int(trial.number)+1} canonical score重建不一致: stored={stored}, reconstructed={reconstructed}"
        )

    monthly_enabled = bool(score_policy.get("SCORE_MONTHLY_WIN_RATE_AMP_ENABLED", False))
    monthly_target = float(score_policy.get("SCORE_MONTHLY_WIN_RATE_TARGET", 0.0) or 0.0)
    monthly_multiplier = (
        calc_score_win_rate_multiplier(_finite(attrs.get("m_win_rate"), field="m_win_rate"), monthly_target)
        if monthly_enabled else 1.0
    )
    quarter_enabled = bool(score_policy.get("SCORE_MIN_QUARTER_RETURN_AMP_ENABLED", False))
    quarter_floor = float(score_policy.get("SCORE_MIN_QUARTER_RETURN_FLOOR", 0.0) or 0.0)
    quarter_target = float(score_policy.get("SCORE_MIN_QUARTER_RETURN_TARGET", 0.0) or 0.0)
    quarter_multiplier = (
        calc_score_min_quarter_return_multiplier(
            _finite(attrs.get("min_quarter_return_pct"), field="min_quarter_return_pct"),
            quarter_floor,
            quarter_target,
        ) if quarter_enabled else 1.0
    )

    no_quarter_policy = dict(score_policy)
    no_quarter_policy["SCORE_MIN_QUARTER_RETURN_AMP_ENABLED"] = False
    without_quarter = _canonical_score(attrs, no_quarter_policy)
    no_month_quarter_policy = dict(no_quarter_policy)
    no_month_quarter_policy["SCORE_MONTHLY_WIN_RATE_AMP_ENABLED"] = False
    without_month_quarter = _canonical_score(attrs, no_month_quarter_policy)

    exact_zero = abs(stored) <= float(zero_tolerance)
    quarter_zero = bool(quarter_enabled and abs(float(quarter_multiplier)) <= float(zero_tolerance))
    quarter_erased = bool(exact_zero and quarter_zero and abs(without_quarter) > float(zero_tolerance))
    return {
        "trial": int(trial.number) + 1,
        "stored_score": float(stored),
        "monthly_win_rate_pct": _finite(attrs.get("m_win_rate"), field="m_win_rate"),
        "monthly_multiplier": float(monthly_multiplier),
        "min_quarter_return_pct": _finite(attrs.get("min_quarter_return_pct"), field="min_quarter_return_pct"),
        "quarter_multiplier": float(quarter_multiplier),
        "score_without_quarter": float(without_quarter),
        "score_without_monthly_and_quarter": float(without_month_quarter),
        "exact_zero": exact_zero,
        "quarter_zero": quarter_zero,
        "quarter_erased": quarter_erased,
        "monthly_heavy": bool(monthly_enabled and monthly_multiplier <= float(monthly_heavy_threshold)),
    }


def _percentile(values: Sequence[float], q: float) -> float | None:
    clean = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return clean[lo]
    weight = pos - lo
    return clean[lo] * (1.0 - weight) + clean[hi] * weight


def _summarize_fold(*, year: int, selected_n: int | None, db_path: Path, study, score_policy: Mapping[str, Any], zero_tolerance: float, monthly_heavy_threshold: float) -> dict[str, Any]:
    rows = []
    completed = 0
    for trial in list(getattr(study, "trials", []) or []):
        if str(getattr(getattr(trial, "state", None), "name", getattr(trial, "state", ""))) == "COMPLETE":
            completed += 1
        row = _trial_row(
            trial,
            score_policy,
            zero_tolerance=zero_tolerance,
            monthly_heavy_threshold=monthly_heavy_threshold,
        )
        if row is not None:
            rows.append(row)
    if not rows:
        raise OptimizerScoreAuditBlocked(f"{year} full-refit study沒有qualified completed trials")

    zero_rows = [row for row in rows if row["exact_zero"]]
    qzero_rows = [row for row in rows if row["quarter_zero"]]
    erased_rows = [row for row in rows if row["quarter_erased"]]
    heavy_rows = [row for row in rows if row["monthly_heavy"]]
    hidden_scores = [float(row["score_without_quarter"]) for row in erased_rows]
    distinct_hidden = len({round(value, 12) for value in hidden_scores})
    best_actual = max(rows, key=lambda row: (row["stored_score"], -row["trial"]))
    best_no_quarter = max(rows, key=lambda row: (row["score_without_quarter"], -row["trial"]))
    multipliers = [float(row["monthly_multiplier"]) for row in rows]
    min_quarters = [float(row["min_quarter_return_pct"]) for row in rows]
    n = len(rows)
    pct = lambda count: (100.0 * float(count) / float(n)) if n else 0.0
    return {
        "oos_year": int(year),
        "selected_n": None if selected_n is None else int(selected_n),
        "db": str(db_path),
        "study_name": str(getattr(study, "study_name", "")),
        "completed_trials": int(completed),
        "qualified_trials": int(n),
        "exact_zero_count": int(len(zero_rows)),
        "exact_zero_pct": pct(len(zero_rows)),
        "quarter_zero_count": int(len(qzero_rows)),
        "quarter_zero_pct": pct(len(qzero_rows)),
        "quarter_erased_count": int(len(erased_rows)),
        "quarter_erased_pct": pct(len(erased_rows)),
        "zero_other_count": int(len([row for row in zero_rows if not row["quarter_zero"]])),
        "monthly_heavy_count": int(len(heavy_rows)),
        "monthly_heavy_pct": pct(len(heavy_rows)),
        "monthly_multiplier_p10": _percentile(multipliers, 0.10),
        "monthly_multiplier_median": median(multipliers) if multipliers else None,
        "monthly_multiplier_p90": _percentile(multipliers, 0.90),
        "min_quarter_return_p10": _percentile(min_quarters, 0.10),
        "min_quarter_return_median": median(min_quarters) if min_quarters else None,
        "quarter_erased_hidden_positive_count": int(sum(1 for value in hidden_scores if value > zero_tolerance)),
        "quarter_erased_hidden_distinct_scores": int(distinct_hidden),
        "quarter_erased_hidden_score_max": max(hidden_scores) if hidden_scores else None,
        "quarter_erased_hidden_score_median": median(hidden_scores) if hidden_scores else None,
        "best_actual_trial": int(best_actual["trial"]),
        "best_actual_score": float(best_actual["stored_score"]),
        "best_without_quarter_trial": int(best_no_quarter["trial"]),
        "best_without_quarter_score": float(best_no_quarter["score_without_quarter"]),
        "rank_signal_present_under_quarter_zero": bool(distinct_hidden > 1 or any(value > zero_tolerance for value in hidden_scores)),
        "trial_rows": rows,
    }


def _collect_evidence(definition: AuditDefinition, *, project_root: Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    manifest = _validate_manifest(definition, root)
    source_root = _source_root(definition, root)
    if not source_root.is_dir():
        raise OptimizerScoreAuditBlocked(f"找不到deterministic P10 workspace: {_relative(source_root, root)}")
    db_paths = _full_refit_db_paths(source_root)
    if not db_paths:
        raise OptimizerScoreAuditBlocked(f"找不到full-refit SQLite studies: {_relative(source_root, root)}")

    zero_tolerance = float(definition.dimensions.get("zero_abs_tolerance", 0.0) or 0.0)
    monthly_heavy_threshold = float(definition.dimensions.get("monthly_heavy_multiplier_threshold", 0.01) or 0.01)
    expected_years = tuple(int(v) for v in definition.source.get("expected_oos_years", ()))

    candidates: list[dict[str, Any]] = []
    for db_path in db_paths:
        study = _study_candidates(db_path)
        score_policy, walk_forward = _study_contract(study)
        year = _year_from_study(study, walk_forward)
        candidates.append({
            "year": year,
            "selected_n": _selected_n(db_path),
            "db_path": db_path,
            "study": study,
            "score_policy": score_policy,
            "mtime_ns": int(db_path.stat().st_mtime_ns),
        })

    selected_by_year: dict[int, dict[str, Any]] = {}
    for item in candidates:
        year = int(item["year"])
        current = selected_by_year.get(year)
        rank = (int(item["mtime_ns"]), str(item["db_path"]))
        current_rank = (-1, "") if current is None else (int(current["mtime_ns"]), str(current["db_path"]))
        if current is None or rank > current_rank:
            selected_by_year[year] = item

    if expected_years:
        missing = sorted(set(expected_years) - set(selected_by_year))
        extra = sorted(set(selected_by_year) - set(expected_years))
        if missing or extra:
            raise OptimizerScoreAuditBlocked(f"full-refit fold coverage不完整: missing={missing}, extra={extra}")

    folds = []
    policy_fingerprints = set()
    for year in sorted(selected_by_year):
        item = selected_by_year[year]
        score_policy = dict(item["score_policy"])
        policy_fingerprints.add(json.dumps(score_policy, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
        fold = _summarize_fold(
            year=year,
            selected_n=item["selected_n"],
            db_path=item["db_path"],
            study=item["study"],
            score_policy=score_policy,
            zero_tolerance=zero_tolerance,
            monthly_heavy_threshold=monthly_heavy_threshold,
        )
        fold["db"] = _relative(item["db_path"], root)
        folds.append(fold)
    if len(policy_fingerprints) != 1:
        raise OptimizerScoreAuditBlocked("各fold persisted training_score_policy不一致，禁止合併解讀")

    score_policy = dict(selected_by_year[sorted(selected_by_year)[0]]["score_policy"])
    return {
        "manifest": manifest,
        "source_root": source_root,
        "db_count_scanned": len(db_paths),
        "folds": folds,
        "score_policy": score_policy,
        "zero_tolerance": zero_tolerance,
        "monthly_heavy_threshold": monthly_heavy_threshold,
    }


def collect_status(definition: AuditDefinition, *, project_root: Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    try:
        evidence = _collect_evidence(definition, project_root=root)
        reason = ""
        status = "READY"
        display = (
            f"{definition.source.get('parameter_set_identity', 'Optimizer')} deterministic full-refit studies "
            f"({len(evidence['folds'])} folds)"
        )
    except (OptimizerScoreAuditBlocked, OSError, ValueError) as exc:
        status = "BLOCKED"
        reason = str(exc)
        display = f"{definition.source.get('parameter_set_identity', 'Optimizer')} deterministic full-refit studies"
    return {
        "status": status,
        "reason": reason,
        "source": {
            "display": display,
            "path": str(definition.source.get("path") or ""),
            "preparable": False,
        },
    }


def _diagnosis(folds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    qualified = sum(int(row["qualified_trials"]) for row in folds)
    exact_zero = sum(int(row["exact_zero_count"]) for row in folds)
    erased = sum(int(row["quarter_erased_count"]) for row in folds)
    monthly_heavy = sum(int(row["monthly_heavy_count"]) for row in folds)
    zero_folds = [int(row["oos_year"]) for row in folds if int(row["exact_zero_count"]) > 0]
    erased_folds = [int(row["oos_year"]) for row in folds if bool(row["rank_signal_present_under_quarter_zero"])]
    return {
        "qualified_trials": qualified,
        "exact_zero_count": exact_zero,
        "exact_zero_pct": 100.0 * exact_zero / qualified if qualified else 0.0,
        "quarter_erased_count": erased,
        "quarter_erased_pct": 100.0 * erased / qualified if qualified else 0.0,
        "monthly_heavy_count": monthly_heavy,
        "monthly_heavy_pct": 100.0 * monthly_heavy / qualified if qualified else 0.0,
        "folds_with_exact_zero": zero_folds,
        "folds_with_quarter_erased_rank_signal": erased_folds,
        "quarter_zero_erases_ranking": bool(erased > 0 and erased_folds),
        "monthly_heavy_compression_present": bool(monthly_heavy > 0),
    }


def _render(result: Mapping[str, Any], *, markdown: bool) -> str:
    folds = list(result.get("folds") or [])
    overall = dict(result.get("diagnosis") or {})
    if markdown:
        lines = [
            "# Optimizer Score Zero-Collapse Audit",
            "",
            "- **Contract**: READ ONLY / training full-refit evidence only",
            "- **OOS used for score selection**: NO",
            "- **Counterfactual**: canonical score reconstruction only; not persisted and not used for fitting",
            f"- **Source**: `{result.get('source_root')}`",
            "",
            "## Fold evidence",
            "",
            "| OOS year | N* | Qualified | Exact zero | Quarter=0 | Quarter erased | Monthly heavy | Best actual | Best no-quarter |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in folds:
            lines.append(
                "| {oos_year} | {selected_n} | {qualified_trials} | {exact_zero_count} ({exact_zero_pct:.1f}%) | "
                "{quarter_zero_count} ({quarter_zero_pct:.1f}%) | {quarter_erased_count} ({quarter_erased_pct:.1f}%) | "
                "{monthly_heavy_count} ({monthly_heavy_pct:.1f}%) | T{best_actual_trial} / {best_actual_score:.6g} | "
                "T{best_without_quarter_trial} / {best_without_quarter_score:.6g} |".format(**row)
            )
        lines.extend([
            "",
            "## Diagnosis",
            "",
            f"- Qualified trials: **{overall.get('qualified_trials', 0)}**",
            f"- Exact-zero score: **{overall.get('exact_zero_count', 0)} ({overall.get('exact_zero_pct', 0.0):.1f}%)**",
            f"- Quarter exact-zero erased otherwise non-zero rank signal: **{overall.get('quarter_erased_count', 0)} ({overall.get('quarter_erased_pct', 0.0):.1f}%)**",
            f"- Monthly heavy compression: **{overall.get('monthly_heavy_count', 0)} ({overall.get('monthly_heavy_pct', 0.0):.1f}%)**",
            f"- Quarter zero erases ranking: **{'YES' if overall.get('quarter_zero_erases_ranking') else 'NO'}**",
            f"- Monthly heavy compression present: **{'YES' if overall.get('monthly_heavy_compression_present') else 'NO'}**",
            "",
            "## Scientific boundary",
            "",
            "This Audit does not choose a new floor, target, or formula. If Quarter exact-zero is shown to erase useful ranking, the next experiment may isolate that single score-shape variable; Monthly compression remains a separate hypothesis.",
        ])
        return "\n".join(lines)

    rows = []
    for row in folds:
        rows.append((
            row["oos_year"],
            row["selected_n"],
            row["qualified_trials"],
            f"{row['exact_zero_count']} / {row['exact_zero_pct']:.1f}%",
            f"{row['quarter_erased_count']} / {row['quarter_erased_pct']:.1f}%",
            f"{row['monthly_heavy_count']} / {row['monthly_heavy_pct']:.1f}%",
            f"T{row['best_actual_trial']} / {row['best_actual_score']:.6g}",
            f"T{row['best_without_quarter_trial']} / {row['best_without_quarter_score']:.6g}",
        ))
    return "\n\n".join((
        render_title("Optimizer Score Zero-Collapse Audit"),
        render_key_values((
            ("Contract", "READ ONLY / training full-refit evidence only"),
            ("OOS used for score selection", "NO"),
            ("Counterfactual", "canonical score reconstruction only; not persisted/fitted"),
            ("Source", str(result.get("source_root") or "-")),
        )),
        render_table(
            ("OOS", "N*", "Qualified", "Score=0", "Quarter erased", "Monthly heavy", "Best actual", "Best no-quarter"),
            rows,
        ),
        render_key_values((
            ("Exact-zero", f"{overall.get('exact_zero_count', 0)} / {overall.get('exact_zero_pct', 0.0):.1f}%"),
            ("Quarter erased", f"{overall.get('quarter_erased_count', 0)} / {overall.get('quarter_erased_pct', 0.0):.1f}%"),
            ("Monthly heavy", f"{overall.get('monthly_heavy_count', 0)} / {overall.get('monthly_heavy_pct', 0.0):.1f}%"),
            ("Quarter zero erases ranking", "YES" if overall.get("quarter_zero_erases_ranking") else "NO"),
            ("Monthly heavy compression", "YES" if overall.get("monthly_heavy_compression_present") else "NO"),
        )),
    ))


def _persist(definition: AuditDefinition, *, project_root: Path, payload: Mapping[str, Any], console_text: str, markdown_text: str) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output_root = root / Path(AUDIT_OUTPUT_ROOT) / Path(definition.output_subdir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / timestamp
    latest_dir = output_root / "latest"
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_dir.mkdir(parents=True, exist_ok=True)
    result = dict(payload)
    result["schema_version"] = 1
    result["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["audit_id"] = definition.audit_id
    result["audit_type"] = definition.audit_type
    result["read_only"] = True
    result["artifacts"] = {
        "markdown": _relative(run_dir / "audit.md", root),
        "json": _relative(run_dir / "audit.json", root),
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    (run_dir / "audit.md").write_text(markdown_text.rstrip() + "\n", encoding="utf-8")
    (run_dir / "audit.json").write_text(text + "\n", encoding="utf-8")
    (run_dir / "console.txt").write_text(console_text.rstrip() + "\n", encoding="utf-8")
    (latest_dir / "audit.md").write_text(markdown_text.rstrip() + "\n", encoding="utf-8")
    (latest_dir / "audit.json").write_text(text + "\n", encoding="utf-8")
    return result


def run_audit(definition: AuditDefinition, *, project_root: Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    evidence = _collect_evidence(definition, project_root=root)
    folds = []
    for row in evidence["folds"]:
        clean = dict(row)
        clean.pop("trial_rows", None)
        folds.append(clean)
    diagnosis = _diagnosis(evidence["folds"])
    payload = {
        "source_root": _relative(evidence["source_root"], root),
        "parameter_set": str(definition.source.get("parameter_set_identity") or ""),
        "strategy_arm_id": str(definition.source.get("strategy_arm_id") or ""),
        "execution_evidence_contract": str(definition.source.get("execution_evidence_contract") or ""),
        "evidence_scope": "optimizer_training_full_refit_only",
        "oos_used_for_score_selection": False,
        "counterfactual_persisted_or_fitted": False,
        "db_count_scanned": int(evidence["db_count_scanned"]),
        "score_policy": dict(evidence["score_policy"]),
        "folds": folds,
        "diagnosis": diagnosis,
    }
    console_text = _render(payload, markdown=False)
    markdown_text = _render(payload, markdown=True)
    return _persist(
        definition,
        project_root=root,
        payload=payload,
        console_text=console_text,
        markdown_text=markdown_text,
    )


def run_formal_audit(definition: AuditDefinition, *, project_root: Path, quiet: bool = False) -> dict[str, Any]:
    result = run_audit(definition, project_root=Path(project_root))
    if not quiet:
        markdown_path = dict(result.get("artifacts") or {}).get("markdown")
        print(_render(result, markdown=False))
        if markdown_path:
            print(f"Markdown: {markdown_path}")
    return result


__all__ = [
    "SUPPORTED_AUDIT_TYPE",
    "OptimizerScoreAuditBlocked",
    "collect_status",
    "run_audit",
    "run_formal_audit",
]
