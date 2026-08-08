"""Read-only A9 PASS quality audit over an existing strategy-comparison arm."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd

from config.audit import AUDIT_OUTPUT_ROOT, AuditDefinition
from core.runtime_utils import get_taipei_now
from core.console_report import (
    print_artifact_paths,
    project_relative_display_path,
    render_key_values,
    render_section,
    render_table,
    render_title,
)
from filters.breakout_quality.paths import resolve_filter_output_dir
from filters.breakout_quality.trade_attribution import reconstruct_round_trips

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_RESULT_SCHEMA_VERSION = 1


def _json_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return value
    return value



def json_native_audit_value(value: Any) -> Any:
    """Return JSON-safe native values shared by formal Audit outputs."""
    return _json_native(value)

def _read_json(path: Path, *, project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            "缺少JSON工件: "
            + project_relative_display_path(path, project_root=project_root)
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root必須是object: {path.name}")
    return payload


def _resolve_run_dir(root: Path, definition: AuditDefinition) -> Path:
    run_setting = str(definition.source.get("run") or "").strip()
    if run_setting == "latest":
        latest_manifest = root / "outputs" / "strategy_compare" / "latest" / "manifest.json"
        payload = _read_json(latest_manifest, project_root=root)
        run_dir_value = str(payload.get("run_dir") or "").strip()
        if not run_dir_value:
            raise ValueError("strategy_compare latest manifest缺少run_dir")
        run_dir = root / Path(run_dir_value)
    else:
        run_dir = root / Path(run_setting)
    run_dir = run_dir.resolve()
    try:
        run_dir.relative_to(root)
    except ValueError as exc:
        raise ValueError("Audit strategy_compare run必須位於專案root內") from exc
    return run_dir


def _arm_contract(result: dict[str, Any], arm_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    settings = dict(result.get("settings") or {})
    arms = dict(settings.get("arms") or {})
    arm = arms.get(str(arm_id))
    if not isinstance(arm, dict):
        raise ValueError(f"strategy_compare結果不存在Audit arm: {arm_id}")
    if not bool(arm.get("dl_enabled")):
        raise ValueError(f"PASS quality audit要求DL-on arm: {arm_id}")
    dl_id = str(arm.get("dl_id") or "").strip()
    dl_sources = dict(settings.get("dl_sources") or {})
    dl = dl_sources.get(dl_id)
    if not isinstance(dl, dict):
        raise ValueError(f"Audit arm引用不存在的DL source: {arm_id}/{dl_id}")
    return arm, dl


def _pair_dir(run_dir: Path, arm: dict[str, Any]) -> Path:
    param_source = str(arm.get("param_source") or "").strip()
    rule_policy = str(arm.get("rule_policy") or "").strip()
    dl_id = str(arm.get("dl_id") or "").strip()
    runtime = str(arm.get("dl_runtime_mode") or "").strip()
    if not all((param_source, rule_policy, dl_id, runtime)):
        raise ValueError("Audit arm identity不完整")
    group_id = f"{param_source}__{rule_policy}__{dl_id}__{runtime.replace('-', '_')}"
    return run_dir / "pairs" / group_id


def _source_paths(
    *, root: Path, definition: AuditDefinition
) -> dict[str, Path | str | float]:
    run_dir = _resolve_run_dir(root, definition)
    result_path = run_dir / "strategy_comparison.json"
    result = _read_json(result_path, project_root=root)
    arm_id = str(definition.source.get("arm_id") or "").strip()
    arm, dl = _arm_contract(result, arm_id)
    pair_dir = _pair_dir(run_dir, arm)
    runtime = str(arm.get("dl_runtime_mode") or "")
    if runtime == "hard-filter":
        active_prefix = "quality_filter"
    else:
        active_prefix = "score_ranking"
    filter_id = str(dl.get("filter_id") or "").strip()
    if not filter_id:
        raise ValueError("Audit DL source缺少filter_id")
    try:
        threshold = float(dl.get("threshold"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Audit DL source缺少合法threshold") from exc
    if not math.isfinite(threshold):
        raise ValueError("Audit DL source threshold必須是有限數值")
    return {
        "run_dir": run_dir,
        "result": result_path,
        "pair_dir": pair_dir,
        "orderable": pair_dir / f"{active_prefix}_orderable_candidates.csv",
        "selected": pair_dir / f"{active_prefix}_selected_buys.csv",
        "trades": pair_dir / f"{active_prefix}_trades.csv",
        "events": resolve_filter_output_dir(root, filter_id=filter_id) / "events.csv",
        "filter_id": filter_id,
        "threshold": threshold,
        "arm_id": arm_id,
        "dl_id": str(arm.get("dl_id") or ""),
        "runtime": runtime,
    }


def collect_pass_quality_status(
    definition: AuditDefinition,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    try:
        paths = _source_paths(root=root, definition=definition)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        return {
            "audit_id": definition.audit_id,
            "status": "BLOCKED",
            "reason": str(exc),
            "paths": {},
            "source": {
                "arm_id": str(definition.source.get("arm_id") or ""),
                "dl_id": None,
                "runtime": None,
                "filter_id": None,
                "threshold": None,
            },
        }
    required = ["result", "orderable", "selected", "trades"]
    if bool(definition.outcomes.get("label_quality", False)):
        required.append("events")
    missing = [key for key in required if not Path(paths[key]).is_file()]
    return {
        "audit_id": definition.audit_id,
        "status": "READY" if not missing else "BLOCKED",
        "reason": "" if not missing else "缺少正式只讀工件: " + ", ".join(missing),
        "paths": {
            key: str(value)
            for key, value in paths.items()
            if isinstance(value, Path)
        },
        "source": {
            "arm_id": paths.get("arm_id"),
            "dl_id": paths.get("dl_id"),
            "runtime": paths.get("runtime"),
            "filter_id": paths.get("filter_id"),
            "threshold": paths.get("threshold"),
        },
    }


def _load_orderable(path: Path) -> pd.DataFrame:
    required = {
        "ticker",
        "trade_date",
        "candidate_date",
        "signal_date",
        "candidate_type",
        "high_len",
        "breakout_quality_score",
    }
    available_columns = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns.tolist()
    missing = sorted(required - set(available_columns))
    if missing:
        raise ValueError(f"orderable candidate工件缺少欄位: {missing}")
    frame = pd.read_csv(
        path,
        usecols=[column for column in available_columns if column in required],
        dtype={"ticker": "string", "candidate_type": "string"},
        encoding="utf-8-sig",
        low_memory=False,
    )
    out = frame.copy()
    out["ticker"] = out["ticker"].fillna("").astype(str).str.strip()
    for column in ("trade_date", "candidate_date", "signal_date"):
        out[column] = pd.to_datetime(out[column], errors="coerce")
    out["breakout_quality_score"] = pd.to_numeric(
        out["breakout_quality_score"], errors="coerce"
    )
    out["high_len"] = pd.to_numeric(out["high_len"], errors="coerce").astype("Int64")
    out = out[
        out["ticker"].ne("")
        & out["trade_date"].notna()
        & out["signal_date"].notna()
        & out["breakout_quality_score"].map(lambda value: math.isfinite(float(value)) if pd.notna(value) else False)
    ].copy()
    key = ["ticker", "trade_date", "signal_date"]
    dup = out.duplicated(key, keep=False)
    if bool(dup.any()):
        sample = out.loc[dup, key + ["candidate_type", "high_len"]].head(10).to_dict("records")
        raise ValueError(f"orderable candidate identity非唯一: sample={sample}")
    out["candidate_age_calendar_days"] = (
        out["candidate_date"].fillna(out["trade_date"]) - out["signal_date"]
    ).dt.days.astype(int)
    if bool((out["candidate_age_calendar_days"] < 0).any()):
        raise ValueError("candidate age出現負值，違反資訊時點契約")
    return out


def _mark_selected(pass_candidates: pd.DataFrame, path: Path) -> pd.DataFrame:
    selected = pd.read_csv(path, encoding="utf-8-sig")
    required = {"ticker", "trade_date", "signal_date"}
    missing = sorted(required - set(selected.columns))
    if missing:
        raise ValueError(f"selected buys工件缺少欄位: {missing}")
    selected = selected.copy()
    selected["ticker"] = selected["ticker"].fillna("").astype(str).str.strip()
    for column in ("trade_date", "signal_date"):
        selected[column] = pd.to_datetime(selected[column], errors="coerce")
    key = ["ticker", "trade_date", "signal_date"]
    selected = selected.dropna(subset=["trade_date", "signal_date"])[key].drop_duplicates()
    selected["is_selected"] = True
    out = pass_candidates.merge(selected, on=key, how="left", validate="one_to_one")
    out["is_selected"] = out["is_selected"].eq(True)
    return out


def _attach_realized_r(pass_candidates: pd.DataFrame, path: Path) -> pd.DataFrame:
    trades = pd.read_csv(path, encoding="utf-8-sig")
    round_trips = reconstruct_round_trips(trades, scenario="audit")
    if round_trips.empty:
        out = pass_candidates.copy()
        out["realized_r"] = np.nan
        return out
    realized = round_trips[["ticker", "entry_date", "signal_date", "r_multiple"]].copy()
    realized["entry_date"] = pd.to_datetime(realized["entry_date"], errors="coerce")
    realized["signal_date"] = pd.to_datetime(realized["signal_date"], errors="coerce")
    realized = realized.rename(columns={"entry_date": "trade_date", "r_multiple": "realized_r"})
    key = ["ticker", "trade_date", "signal_date"]
    dup = realized.duplicated(key, keep=False)
    if bool(dup.any()):
        raise ValueError("round-trip realized R identity非唯一")
    return pass_candidates.merge(realized, on=key, how="left", validate="one_to_one")


def _attach_label_quality(pass_candidates: pd.DataFrame, path: Path) -> pd.DataFrame:
    available_columns = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns.tolist()
    desired = [
        "ticker",
        "date",
        "high_len",
        "label",
        "label_status",
        "label_reason",
        "decision_mfe_return",
        "decision_mae_return",
        "reward_risk_ratio",
    ]
    usecols = [column for column in desired if column in available_columns]
    required = {"ticker", "date", "high_len", "label", "decision_mfe_return", "decision_mae_return"}
    if not required.issubset(usecols):
        raise ValueError(
            "Dataset events不足以做PASS quality audit: missing="
            + ",".join(sorted(required - set(usecols)))
        )
    events = pd.read_csv(
        path,
        usecols=usecols,
        dtype={"ticker": "string"},
        encoding="utf-8-sig",
        low_memory=False,
    )
    events["ticker"] = events["ticker"].fillna("").astype(str).str.strip()
    events["date"] = pd.to_datetime(events["date"], errors="coerce")
    events["high_len"] = pd.to_numeric(events["high_len"], errors="coerce").astype("Int64")
    key = ["ticker", "date", "high_len"]
    wanted = (
        pass_candidates[["ticker", "signal_date", "high_len"]]
        .drop_duplicates()
        .rename(columns={"signal_date": "date"})
    )
    events = events.merge(wanted, on=key, how="inner", validate="many_to_one")
    outcome_cols = [column for column in usecols if column not in key]
    duplicate_mask = events.duplicated(key, keep=False)
    if bool(duplicate_mask.any()):
        duplicate_rows = events.loc[duplicate_mask, key + outcome_cols]
        uniqueness = duplicate_rows.groupby(key, dropna=False, sort=False)[outcome_cols].nunique(dropna=False)
        conflicting_keys = uniqueness.index[uniqueness.gt(1).any(axis=1)]
        if len(conflicting_keys):
            sample = [dict(zip(key, values if isinstance(values, tuple) else (values,))) for values in list(conflicting_keys[:5])]
            raise ValueError(f"Dataset event label identity有衝突: sample={sample}")
    events = events.drop_duplicates(key, keep="first").rename(columns={"date": "signal_date"})
    out = pass_candidates.merge(
        events,
        on=["ticker", "signal_date", "high_len"],
        how="left",
        validate="many_to_one",
    )
    label_numeric = pd.to_numeric(out.get("label"), errors="coerce")
    out["label_pass"] = (label_numeric == 1).astype("boolean")
    out.loc[label_numeric.isna(), "label_pass"] = pd.NA
    return out



def prepare_pass_candidate_audit_frame(
    definition: AuditDefinition,
    *,
    project_root: Path = PROJECT_ROOT,
) -> tuple[dict[str, Path | str | float], float, pd.DataFrame, pd.DataFrame]:
    """Load one formal strategy-compare arm and enrich its DL PASS candidate-days.

    This is the shared read-only source path for PASS quality/persistence audits.
    It never reruns strategy, rebuilds labels, changes candidate validity, or mutates runtime.
    """
    root = Path(project_root).resolve()
    status = collect_pass_quality_status(definition, project_root=root)
    if status["status"] != "READY":
        raise RuntimeError(status["reason"])
    paths = _source_paths(root=root, definition=definition)
    threshold = float(paths["threshold"])
    orderable = _load_orderable(Path(paths["orderable"]))
    pass_candidates = orderable[
        orderable["breakout_quality_score"] >= threshold
    ].copy()
    if pass_candidates.empty:
        raise RuntimeError("Audit來源沒有任何DL PASS orderable candidates")
    pass_candidates = _mark_selected(pass_candidates, Path(paths["selected"]))
    if bool(definition.outcomes.get("realized_r", False)):
        pass_candidates = _attach_realized_r(pass_candidates, Path(paths["trades"]))
    else:
        pass_candidates["realized_r"] = np.nan
    if bool(definition.outcomes.get("label_quality", False)):
        pass_candidates = _attach_label_quality(pass_candidates, Path(paths["events"]))
    else:
        for column in ("label", "decision_mfe_return", "decision_mae_return"):
            pass_candidates[column] = np.nan
    return paths, threshold, orderable, pass_candidates

def _quantile_labels(series: pd.Series, groups: int, prefix: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    result = pd.Series(pd.NA, index=series.index, dtype="object")
    valid = numeric.dropna()
    if valid.empty:
        return result
    q = min(int(groups), int(valid.nunique()), int(len(valid)))
    if q < 2:
        result.loc[valid.index] = f"{prefix}1"
        return result
    try:
        bins = pd.qcut(valid, q=q, duplicates="drop")
    except ValueError:
        ranked = valid.rank(method="first")
        bins = pd.qcut(ranked, q=q, duplicates="drop")
    categories = list(bins.cat.categories)
    mapping = {category: f"{prefix}{idx + 1}" for idx, category in enumerate(categories)}
    result.loc[valid.index] = bins.map(mapping).astype(str)
    return result


def _spearman(frame: pd.DataFrame, left: str, right: str) -> float | None:
    pair = frame[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(pair) < 3 or pair[left].nunique() < 2 or pair[right].nunique() < 2:
        return None
    value = float(pair[left].corr(pair[right], method="spearman"))
    return value if math.isfinite(value) else None

def audit_quantile_labels(series: pd.Series, groups: int, prefix: str) -> pd.Series:
    """Shared config-driven quantile labels for formal Audit modules."""
    return _quantile_labels(series, groups, prefix)


def audit_spearman(frame: pd.DataFrame, left: str, right: str) -> float | None:
    """Shared finite Spearman diagnostic for formal Audit modules."""
    return _spearman(frame, left, right)


def _group_sort_key(value: Any) -> tuple[str, int, str]:
    text = str(value)
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", text)
    if match:
        return (match.group(1), int(match.group(2)), "")
    return (text, -1, text)


def _aggregate_dimension(frame: pd.DataFrame, dimension: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for value, group in frame.groupby(dimension, dropna=False, sort=True):
        selected = group[group["is_selected"]]
        label_numeric = pd.to_numeric(group.get("label"), errors="coerce")
        mfe = pd.to_numeric(group.get("decision_mfe_return"), errors="coerce")
        mae = pd.to_numeric(group.get("decision_mae_return"), errors="coerce")
        realized = pd.to_numeric(selected.get("realized_r"), errors="coerce").dropna()
        rows.append(
            {
                "group": str(value),
                "candidate_count": int(len(group)),
                "selected_count": int(len(selected)),
                "selected_rate_pct": float(len(selected) / len(group) * 100.0) if len(group) else None,
                "score_min": float(group["breakout_quality_score"].min()),
                "score_max": float(group["breakout_quality_score"].max()),
                "score_mean": float(group["breakout_quality_score"].mean()),
                "age_mean_days": float(group["candidate_age_calendar_days"].mean()),
                "label_coverage_pct": float(label_numeric.notna().mean() * 100.0),
                "label_pass_rate_pct": float((label_numeric.dropna() == 1).mean() * 100.0) if label_numeric.notna().any() else None,
                "decision_mfe_mean_pct": float(mfe.dropna().mean() * 100.0) if mfe.notna().any() else None,
                "decision_mae_mean_pct": float(mae.dropna().mean() * 100.0) if mae.notna().any() else None,
                "realized_count": int(realized.count()),
                "realized_r_mean": float(realized.mean()) if len(realized) else None,
                "realized_r_median": float(realized.median()) if len(realized) else None,
                "realized_win_rate_pct": float((realized > 0.0).mean() * 100.0) if len(realized) else None,
            }
        )
    output = pd.DataFrame(rows)
    if not output.empty:
        output = output.sort_values(
            "group", key=lambda values: values.map(_group_sort_key), kind="mergesort"
        ).reset_index(drop=True)
    return output


def _fmt(value: Any, *, digits: int = 2, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}{suffix}"


def _render_group_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "無可顯示資料"
    rows = []
    for row in frame.to_dict("records"):
        rows.append(
            (
                row["group"],
                row["candidate_count"],
                row["selected_count"],
                _fmt(row["selected_rate_pct"], suffix="%"),
                f"{_fmt(row['score_min'], digits=3)}～{_fmt(row['score_max'], digits=3)}",
                _fmt(row["age_mean_days"], digits=1, suffix="日"),
                _fmt(row["label_pass_rate_pct"], suffix="%"),
                _fmt(row["decision_mfe_mean_pct"], suffix="%"),
                _fmt(row["decision_mae_mean_pct"], suffix="%"),
                _fmt(row["realized_r_mean"], suffix=" R"),
            )
        )
    return render_table(
        (
            "分組",
            "PASS候選",
            "實際買入",
            "買入率",
            "Score範圍",
            "平均Age",
            "原Event Label PASS",
            "MFE",
            "MAE",
            "買入Realized R",
        ),
        rows,
    )


def _render_report(payload: dict[str, Any]) -> str:
    overview = dict(payload["overview"])
    diagnostics = dict(payload["diagnostics"])
    score_groups = pd.DataFrame(payload.get("score_groups") or [])
    age_groups = pd.DataFrame(payload.get("age_groups") or [])
    type_groups = pd.DataFrame(payload.get("candidate_type_groups") or [])
    return "\n\n".join(
        (
            render_title("Binary DL PASS Quality Audit"),
            render_key_values(
                (
                    ("Audit ID", payload["audit_id"]),
                    ("來源Arm", payload["source"]["arm_id"]),
                    ("DL", payload["source"]["dl_id"]),
                    ("Runtime", payload["source"]["runtime"]),
                    ("Threshold", payload["source"]["threshold"]),
                    ("策略Candidate validity", "只讀；仍由原策略唯一負責"),
                    ("DL語意", "quality only；REJECT不等於candidate invalid"),
                )
            ),
            render_section("1. PASS總覽"),
            render_key_values(
                (
                    ("Orderable candidates", overview["orderable_candidate_count"]),
                    ("PASS candidates", overview["pass_candidate_count"]),
                    ("PASS share", _fmt(overview["pass_share_pct"], suffix="%")),
                    ("Selected PASS", overview["selected_pass_count"]),
                    ("Selected PASS Realized R", _fmt(overview["selected_pass_realized_r_mean"], suffix=" R")),
                    ("原Event Label coverage", _fmt(overview["label_coverage_pct"], suffix="%")),
                    ("原Event Label PASS rate", _fmt(overview["label_pass_rate_pct"], suffix="%")),
                )
            ),
            render_section("2. Score分位"),
            _render_group_table(score_groups),
            render_section("3. Candidate age分位"),
            _render_group_table(age_groups),
            render_section("4. Candidate type"),
            _render_group_table(type_groups),
            render_section("5. 單調性診斷"),
            render_key_values(
                (
                    ("Score ↔ Label", _fmt(diagnostics.get("score_vs_label_spearman"), digits=3)),
                    ("Score ↔ Realized R", _fmt(diagnostics.get("score_vs_realized_r_spearman"), digits=3)),
                    ("Age ↔ 原Event Label（組成診斷）", _fmt(diagnostics.get("age_vs_label_spearman"), digits=3)),
                    ("Age ↔ Realized R", _fmt(diagnostics.get("age_vs_realized_r_spearman"), digits=3)),
                )
            ),
            render_section("6. 使用限制"),
            (
                "本Audit只讀既有正式策略比較、Dataset Label與已發生交易；未來Label／MFE／MAE只用於事後診斷，"
                "不得回流當日runtime。Candidate存在／continuation／失效仍由原策略唯一決定；"
                "DL只描述quality，REJECT不得刪除仍屬策略VALID的candidate。原Dataset Label錨定signal event，"
                "同一event在不同candidate age出現時Label不會重新定義，因此Age↔Label只代表lifecycle組成；"
                "candidate-day結果只可由實際selected trades的Realized R觀察，存在portfolio selection bias，不能當成所有未選PASS的反事實。"
                "分位組只供診斷，不得直接轉成runtime threshold或Min ROOS／DL混合權重。"
            ),
        )
    ).rstrip() + "\n"


def run_pass_quality_audit(
    definition: AuditDefinition,
    *,
    project_root: Path = PROJECT_ROOT,
    quiet: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    paths, threshold, orderable, pass_candidates = prepare_pass_candidate_audit_frame(
        definition, project_root=root
    )

    score_groups_n = int(definition.dimensions["score_quantile_groups"])
    age_groups_n = int(definition.dimensions["candidate_age_quantile_groups"])
    pass_candidates["score_quantile"] = _quantile_labels(
        pass_candidates["breakout_quality_score"], score_groups_n, "Q"
    )
    pass_candidates["age_quantile"] = _quantile_labels(
        pass_candidates["candidate_age_calendar_days"], age_groups_n, "A"
    )
    score_groups = _aggregate_dimension(pass_candidates, "score_quantile")
    age_groups = _aggregate_dimension(pass_candidates, "age_quantile")
    type_groups = (
        _aggregate_dimension(pass_candidates, "candidate_type")
        if bool(definition.dimensions.get("candidate_type", True))
        else pd.DataFrame()
    )

    label_numeric = pd.to_numeric(pass_candidates["label"], errors="coerce")
    selected = pass_candidates[pass_candidates["is_selected"]].copy()
    selected_r = pd.to_numeric(selected["realized_r"], errors="coerce").dropna()
    diagnostics = {
        "score_vs_label_spearman": _spearman(pass_candidates, "breakout_quality_score", "label"),
        "score_vs_realized_r_spearman": _spearman(selected, "breakout_quality_score", "realized_r"),
        "age_vs_label_spearman": _spearman(pass_candidates, "candidate_age_calendar_days", "label"),
        "age_vs_realized_r_spearman": _spearman(selected, "candidate_age_calendar_days", "realized_r"),
    }
    payload = {
        "schema_version": AUDIT_RESULT_SCHEMA_VERSION,
        "created_at": get_taipei_now().isoformat(),
        "audit_id": definition.audit_id,
        "audit_type": definition.audit_type,
        "config": definition.as_dict(),
        "source": {
            "arm_id": paths["arm_id"],
            "dl_id": paths["dl_id"],
            "runtime": paths["runtime"],
            "filter_id": paths["filter_id"],
            "threshold": threshold,
            "strategy_compare_run": project_relative_display_path(
                Path(paths["run_dir"]), project_root=root
            ),
        },
        "semantic_contract": {
            "strategy_owns_candidate_validity": True,
            "dl_owns_quality_only": True,
            "dl_reject_does_not_invalidate_candidate": True,
            "portfolio_selector_owns_allocation": True,
            "future_outcomes_are_read_only_audit_only": True,
        },
        "overview": {
            "orderable_candidate_count": int(len(orderable)),
            "pass_candidate_count": int(len(pass_candidates)),
            "pass_share_pct": float(len(pass_candidates) / len(orderable) * 100.0) if len(orderable) else None,
            "selected_pass_count": int(len(selected)),
            "selected_pass_realized_r_mean": float(selected_r.mean()) if len(selected_r) else None,
            "label_coverage_pct": float(label_numeric.notna().mean() * 100.0),
            "label_pass_rate_pct": float((label_numeric.dropna() == 1).mean() * 100.0) if label_numeric.notna().any() else None,
        },
        "diagnostics": diagnostics,
        "score_groups": score_groups.to_dict("records"),
        "age_groups": age_groups.to_dict("records"),
        "candidate_type_groups": type_groups.to_dict("records"),
    }
    payload = _json_native(payload)

    output_root = root / Path(AUDIT_OUTPUT_ROOT) / Path(definition.output_subdir)
    timestamp = get_taipei_now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / "runs" / timestamp
    latest_dir = output_root / "latest"
    run_dir.mkdir(parents=True, exist_ok=False)
    report_path = run_dir / "audit.md"
    json_path = run_dir / "audit.json"
    detail_path = run_dir / "pass_candidates.csv"
    score_path = run_dir / "score_groups.csv"
    age_path = run_dir / "age_groups.csv"
    type_path = run_dir / "candidate_type_groups.csv"
    report_path.write_text(_render_report(payload), encoding="utf-8")
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    pass_candidates.sort_values(
        ["trade_date", "ticker", "signal_date"], kind="mergesort"
    ).to_csv(detail_path, index=False, encoding="utf-8-sig")
    score_groups.to_csv(score_path, index=False, encoding="utf-8-sig")
    age_groups.to_csv(age_path, index=False, encoding="utf-8-sig")
    type_groups.to_csv(type_path, index=False, encoding="utf-8-sig")

    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    latest_dir.mkdir(parents=True, exist_ok=True)
    for source in (report_path, json_path, detail_path, score_path, age_path, type_path):
        shutil.copy2(source, latest_dir / source.name)

    if not quiet:
        print("\n" + _render_report(payload))
        print_artifact_paths(
            (
                ("Audit Markdown", report_path),
                ("Audit JSON", json_path),
                ("PASS candidates", detail_path),
                ("最新Audit", latest_dir),
            ),
            project_root=root,
        )
    return payload


__all__ = [
    "audit_quantile_labels",
    "audit_spearman",
    "collect_pass_quality_status",
    "json_native_audit_value",
    "prepare_pass_candidate_audit_frame",
    "run_pass_quality_audit",
]
