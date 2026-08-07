"""Read-only A9 PASS persistence amplification audit."""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pandas as pd

from config.audit import AUDIT_OUTPUT_ROOT, AuditDefinition
from core.runtime_utils import get_taipei_now
from filters.breakout_quality.console_report import (
    print_artifact_paths,
    project_relative_display_path,
    render_key_values,
    render_section,
    render_table,
    render_title,
)
from tools.filters.breakout_quality.audit_pass_quality import (
    collect_pass_quality_status,
    json_native_audit_value,
    prepare_pass_candidate_audit_frame,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_RESULT_SCHEMA_VERSION = 1
_EVENT_KEY = ("ticker", "signal_date", "high_len")


def collect_pass_persistence_status(
    definition: AuditDefinition,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Persistence uses the same formal read-only source contract as PASS quality."""
    return collect_pass_quality_status(definition, project_root=project_root)


def _finite_mean(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    return float(numeric.mean()) if len(numeric) else None


def _finite_median(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    return float(numeric.median()) if len(numeric) else None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if not math.isfinite(float(numerator)) or not math.isfinite(float(denominator)):
        return None
    if abs(float(denominator)) <= 1e-12:
        return None
    return float(numerator) / float(denominator)


def _pct(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator * 100.0) if denominator > 0 else None


def _build_event_persistence(pass_candidates: pd.DataFrame) -> pd.DataFrame:
    required = set(_EVENT_KEY) | {
        "trade_date",
        "candidate_date",
        "candidate_type",
        "candidate_age_calendar_days",
        "breakout_quality_score",
        "is_selected",
        "realized_r",
        "label",
    }
    missing = sorted(required - set(pass_candidates.columns))
    if missing:
        raise ValueError(f"PASS persistence缺少欄位: {missing}")

    rows: list[dict[str, Any]] = []
    for identity, group in pass_candidates.groupby(
        list(_EVENT_KEY), dropna=False, sort=False
    ):
        identity_tuple = identity if isinstance(identity, tuple) else (identity,)
        label_numeric = pd.to_numeric(group["label"], errors="coerce").dropna()
        known_labels = sorted({int(value) for value in label_numeric if int(value) in (0, 1)})
        if len(known_labels) > 1:
            raise ValueError(
                "同一PASS event出現衝突原Event Label: "
                + str(dict(zip(_EVENT_KEY, identity_tuple)))
            )
        label = known_labels[0] if known_labels else None
        selected = group[group["is_selected"].eq(True)]
        realized = pd.to_numeric(selected["realized_r"], errors="coerce")
        realized = realized[np.isfinite(realized)]
        candidate_dates = group["candidate_date"].fillna(group["trade_date"])
        ages = pd.to_numeric(group["candidate_age_calendar_days"], errors="coerce")
        scores = pd.to_numeric(group["breakout_quality_score"], errors="coerce")
        row = dict(zip(_EVENT_KEY, identity_tuple))
        row.update(
            {
                "label": label,
                "label_group": "PASS" if label == 1 else "REJECT" if label == 0 else "UNKNOWN",
                "candidate_day_count": int(len(group)),
                "first_candidate_date": candidate_dates.min(),
                "last_candidate_date": candidate_dates.max(),
                "max_candidate_age_days": int(ages.max()) if ages.notna().any() else None,
                "mean_candidate_age_days": float(ages.mean()) if ages.notna().any() else None,
                "extended_candidate_day_count": int(
                    group["candidate_type"].astype(str).str.lower().eq("extended").sum()
                ),
                "normal_candidate_day_count": int(
                    group["candidate_type"].astype(str).str.lower().eq("normal").sum()
                ),
                "selected_candidate_day_count": int(len(selected)),
                "selected_realized_r_mean": float(realized.mean()) if len(realized) else None,
                "score_mean": float(scores.mean()) if scores.notna().any() else None,
                "score_min": float(scores.min()) if scores.notna().any() else None,
                "score_max": float(scores.max()) if scores.notna().any() else None,
            }
        )
        rows.append(row)
    output = pd.DataFrame(rows)
    if output.empty:
        raise RuntimeError("PASS persistence沒有可分析的unique events")
    return output.sort_values(list(_EVENT_KEY), kind="mergesort").reset_index(drop=True)


def _aggregate_label_persistence(
    events: pd.DataFrame,
    pass_candidates: pd.DataFrame,
    *,
    include_selected: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    labels = ((1, "PASS"), (0, "REJECT"))
    candidate_labels = pd.to_numeric(pass_candidates["label"], errors="coerce")
    for label_value, label_name in labels:
        event_group = events[pd.to_numeric(events["label"], errors="coerce").eq(label_value)]
        candidate_group = pass_candidates[candidate_labels.eq(label_value)]
        selected = candidate_group[candidate_group["is_selected"].eq(True)]
        realized = pd.to_numeric(selected["realized_r"], errors="coerce")
        realized = realized[np.isfinite(realized)]
        rows.append(
            {
                "label_group": label_name,
                "unique_event_count": int(len(event_group)),
                "candidate_day_count": int(len(candidate_group)),
                "candidate_days_per_event_mean": _finite_mean(event_group["candidate_day_count"]),
                "candidate_days_per_event_median": _finite_median(event_group["candidate_day_count"]),
                "max_candidate_age_mean_days": _finite_mean(event_group["max_candidate_age_days"]),
                "max_candidate_age_median_days": _finite_median(event_group["max_candidate_age_days"]),
                "extended_days_per_event_mean": _finite_mean(event_group["extended_candidate_day_count"]),
                "selected_candidate_count": int(len(selected)) if include_selected else None,
                "selected_realized_r_mean": (
                    float(realized.mean()) if include_selected and len(realized) else None
                ),
            }
        )
    return pd.DataFrame(rows)


def _overview_and_amplification(
    events: pd.DataFrame,
    pass_candidates: pd.DataFrame,
    *,
    include_selected: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    event_labels = pd.to_numeric(events["label"], errors="coerce")
    candidate_labels = pd.to_numeric(pass_candidates["label"], errors="coerce")
    known_events = events[event_labels.isin([0, 1])]
    known_event_labels = pd.to_numeric(known_events["label"], errors="coerce")
    known_candidates = pass_candidates[candidate_labels.isin([0, 1])]
    known_candidate_labels = pd.to_numeric(known_candidates["label"], errors="coerce")
    selected = known_candidates[known_candidates["is_selected"].eq(True)]
    selected_labels = pd.to_numeric(selected["label"], errors="coerce")

    event_pass_count = int(known_event_labels.eq(1).sum())
    event_reject_count = int(known_event_labels.eq(0).sum())
    candidate_pass_count = int(known_candidate_labels.eq(1).sum())
    candidate_reject_count = int(known_candidate_labels.eq(0).sum())
    selected_pass_count = int(selected_labels.eq(1).sum())
    selected_reject_count = int(selected_labels.eq(0).sum())

    event_precision = _pct(event_pass_count, len(known_events))
    candidate_precision = _pct(candidate_pass_count, len(known_candidates))
    selected_precision = _pct(selected_pass_count, len(selected))
    event_fp_share = _pct(event_reject_count, len(known_events))
    candidate_fp_share = _pct(candidate_reject_count, len(known_candidates))
    selected_fp_share = _pct(selected_reject_count, len(selected))

    pass_event_rows = known_events[known_event_labels.eq(1)]
    reject_event_rows = known_events[known_event_labels.eq(0)]
    pass_days_mean = _finite_mean(pass_event_rows["candidate_day_count"])
    reject_days_mean = _finite_mean(reject_event_rows["candidate_day_count"])

    overview = {
        "pass_candidate_day_count": int(len(pass_candidates)),
        "unique_pass_event_count": int(len(events)),
        "event_label_coverage_pct": _pct(len(known_events), len(events)),
        "candidate_day_label_coverage_pct": _pct(len(known_candidates), len(pass_candidates)),
        "unique_event_label_pass_rate_pct": event_precision,
        "candidate_day_weighted_label_pass_rate_pct": candidate_precision,
        "candidate_day_minus_event_precision_pp": (
            float(candidate_precision - event_precision)
            if candidate_precision is not None and event_precision is not None
            else None
        ),
        "selected_known_label_count": int(len(selected)) if include_selected else None,
        "selected_label_pass_rate_pct": selected_precision if include_selected else None,
    }
    amplification = {
        "false_positive_unique_event_share_pct": event_fp_share,
        "false_positive_candidate_day_share_pct": candidate_fp_share,
        "false_positive_candidate_day_amplification_ratio": _ratio(
            candidate_fp_share, event_fp_share
        ),
        "false_positive_selected_share_pct": selected_fp_share if include_selected else None,
        "false_positive_selected_amplification_ratio": (
            _ratio(selected_fp_share, event_fp_share) if include_selected else None
        ),
        "true_positive_candidate_days_per_event_mean": pass_days_mean,
        "false_positive_candidate_days_per_event_mean": reject_days_mean,
        "false_vs_true_candidate_days_per_event_ratio": _ratio(
            reject_days_mean, pass_days_mean
        ),
    }
    return overview, amplification


def _fmt(value: Any, *, digits: int = 2, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}{suffix}"


def _render_label_groups(frame: pd.DataFrame) -> str:
    rows = []
    for row in frame.to_dict("records"):
        rows.append(
            (
                row["label_group"],
                row["unique_event_count"],
                row["candidate_day_count"],
                _fmt(row["candidate_days_per_event_mean"], digits=1),
                _fmt(row["candidate_days_per_event_median"], digits=1),
                _fmt(row["max_candidate_age_mean_days"], digits=1, suffix="日"),
                _fmt(row["max_candidate_age_median_days"], digits=1, suffix="日"),
                _fmt(row["extended_days_per_event_mean"], digits=1),
                row["selected_candidate_count"],
                _fmt(row["selected_realized_r_mean"], suffix=" R"),
            )
        )
    return render_table(
        (
            "原Event Label",
            "Unique events",
            "Candidate-days",
            "Days/Event平均",
            "Days/Event中位",
            "Max Age平均",
            "Max Age中位",
            "Extended Days/Event",
            "Selected",
            "Selected Realized R",
        ),
        rows,
    )


def _render_report(payload: dict[str, Any]) -> str:
    overview = dict(payload["overview"])
    amplification = dict(payload["amplification"])
    groups = pd.DataFrame(payload.get("label_persistence_groups") or [])
    return "\n\n".join(
        (
            render_title("Binary DL PASS Persistence Audit"),
            render_key_values(
                (
                    ("Audit ID", payload["audit_id"]),
                    ("來源Arm", payload["source"]["arm_id"]),
                    ("DL", payload["source"]["dl_id"]),
                    ("Runtime", payload["source"]["runtime"]),
                    ("Threshold", payload["source"]["threshold"]),
                    ("Event identity", "ticker / signal_date / high_len"),
                    ("策略Candidate validity", "只讀；仍由原策略唯一負責"),
                    ("DL語意", "quality only；Persistence不是第二套失效規則"),
                )
            ),
            render_section("1. Event-level vs Candidate-day"),
            render_key_values(
                (
                    ("PASS candidate-days", overview["pass_candidate_day_count"]),
                    ("Unique PASS events", overview["unique_pass_event_count"]),
                    ("Event Label coverage", _fmt(overview["event_label_coverage_pct"], suffix="%")),
                    ("Candidate-day Label coverage", _fmt(overview["candidate_day_label_coverage_pct"], suffix="%")),
                    ("Unique-event Label PASS", _fmt(overview["unique_event_label_pass_rate_pct"], suffix="%")),
                    ("Candidate-day weighted Label PASS", _fmt(overview["candidate_day_weighted_label_pass_rate_pct"], suffix="%")),
                    ("Candidate-day − Event precision", _fmt(overview["candidate_day_minus_event_precision_pp"], suffix=" pp")),
                    ("Selected known-label PASS", _fmt(overview["selected_label_pass_rate_pct"], suffix="%")),
                )
            ),
            render_section("2. Persistence依原Event Label"),
            _render_label_groups(groups),
            render_section("3. False-positive amplification"),
            render_key_values(
                (
                    ("False-positive unique-event share", _fmt(amplification["false_positive_unique_event_share_pct"], suffix="%")),
                    ("False-positive candidate-day share", _fmt(amplification["false_positive_candidate_day_share_pct"], suffix="%")),
                    ("Candidate-day amplification", _fmt(amplification["false_positive_candidate_day_amplification_ratio"], digits=2, suffix="x")),
                    ("False-positive selected share", _fmt(amplification["false_positive_selected_share_pct"], suffix="%")),
                    ("Selected amplification", _fmt(amplification["false_positive_selected_amplification_ratio"], digits=2, suffix="x")),
                    ("True PASS days/event", _fmt(amplification["true_positive_candidate_days_per_event_mean"], digits=1)),
                    ("False PASS days/event", _fmt(amplification["false_positive_candidate_days_per_event_mean"], digits=1)),
                    ("False / True persistence", _fmt(amplification["false_vs_true_candidate_days_per_event_ratio"], digits=2, suffix="x")),
                )
            ),
            render_section("4. 使用限制"),
            (
                "本Audit只量化既有策略VALID candidate pool中的event persistence weighting。"
                "False-positive定義為A9判PASS但原Event Label為REJECT；Label／MFE／MAE／Realized R均為事後診斷，"
                "不得回流當日runtime。Candidate-days較多只表示同一策略VALID event在orderable pool中重複出現較久，"
                "不代表candidate應被DL判失效，也不得直接建立age cutoff。Selected amplification另受portfolio selection bias影響，"
                "只用來確認C12實際allocation是否進一步放大或抑制原Event false positives。"
            ),
        )
    ).rstrip() + "\n"


def run_pass_persistence_audit(
    definition: AuditDefinition,
    *,
    project_root: Path = PROJECT_ROOT,
    quiet: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    paths, threshold, _orderable, pass_candidates = prepare_pass_candidate_audit_frame(
        definition, project_root=root
    )
    events = _build_event_persistence(pass_candidates)
    include_selected = bool(definition.dimensions.get("selected_amplification", True))
    label_groups = _aggregate_label_persistence(
        events, pass_candidates, include_selected=include_selected
    )
    overview, amplification = _overview_and_amplification(
        events, pass_candidates, include_selected=include_selected
    )

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
            "persistence_does_not_define_candidate_expiry": True,
            "portfolio_selector_owns_allocation": True,
            "future_outcomes_are_read_only_audit_only": True,
        },
        "overview": overview,
        "amplification": amplification,
        "label_persistence_groups": label_groups.to_dict("records"),
    }
    payload = json_native_audit_value(payload)

    output_root = root / Path(AUDIT_OUTPUT_ROOT) / Path(definition.output_subdir)
    timestamp = get_taipei_now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / "runs" / timestamp
    latest_dir = output_root / "latest"
    run_dir.mkdir(parents=True, exist_ok=False)
    report_path = run_dir / "audit.md"
    json_path = run_dir / "audit.json"
    event_path = run_dir / "event_persistence.csv"
    group_path = run_dir / "label_persistence_groups.csv"
    detail_path = run_dir / "pass_candidates.csv"

    report_path.write_text(_render_report(payload), encoding="utf-8")
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    events.to_csv(event_path, index=False, encoding="utf-8-sig")
    label_groups.to_csv(group_path, index=False, encoding="utf-8-sig")
    pass_candidates.sort_values(
        ["trade_date", "ticker", "signal_date"], kind="mergesort"
    ).to_csv(detail_path, index=False, encoding="utf-8-sig")

    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    latest_dir.mkdir(parents=True, exist_ok=True)
    for source in (report_path, json_path, event_path, group_path, detail_path):
        shutil.copy2(source, latest_dir / source.name)

    if not quiet:
        print("\n" + _render_report(payload))
        print_artifact_paths(
            (
                ("Audit Markdown", report_path),
                ("Audit JSON", json_path),
                ("Event persistence", event_path),
                ("最新Audit", latest_dir),
            ),
            project_root=root,
        )
    return payload


__all__ = [
    "collect_pass_persistence_status",
    "run_pass_persistence_audit",
]
