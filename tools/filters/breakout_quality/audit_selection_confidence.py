"""Read-only A9 confidence ordering audit inside resource-aware DL Selection Mode."""

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
    audit_quantile_labels,
    audit_spearman,
    collect_pass_quality_status,
    json_native_audit_value,
    prepare_pass_candidate_audit_frame,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_RESULT_SCHEMA_VERSION = 1
_EVENT_KEY = ("ticker", "signal_date", "high_len")


def _active_prefix(runtime: str) -> str:
    return "quality_filter" if str(runtime) == "hard-filter" else "score_ranking"


def _daily_capacity_path(status: dict[str, Any]) -> Path:
    pair_dir = Path(str((status.get("paths") or {}).get("pair_dir") or ""))
    runtime = str((status.get("source") or {}).get("runtime") or "")
    return pair_dir / f"{_active_prefix(runtime)}_daily_capacity.csv"


def collect_selection_confidence_status(
    definition: AuditDefinition,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    status = collect_pass_quality_status(definition, project_root=project_root)
    if status.get("status") != "READY":
        return status
    daily_capacity = _daily_capacity_path(status)
    if not daily_capacity.is_file():
        out = dict(status)
        out["status"] = "BLOCKED"
        out["reason"] = "缺少正式只讀工件: daily_capacity"
        out["paths"] = {**dict(status.get("paths") or {}), "daily_capacity": str(daily_capacity)}
        return out
    out = dict(status)
    out["paths"] = {**dict(status.get("paths") or {}), "daily_capacity": str(daily_capacity)}
    return out


def _load_dl_selection_days(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns.tolist()
    required = {"Date", "Resource_Aware_Mode"}
    missing = sorted(required - set(header))
    if missing:
        raise ValueError(f"daily capacity缺少Resource-aware欄位: {missing}")
    frame = pd.read_csv(
        path,
        usecols=[column for column in header if column in required],
        dtype={"Resource_Aware_Mode": "string"},
        encoding="utf-8-sig",
        low_memory=False,
    )
    frame["trade_date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["resource_aware_mode"] = (
        frame["Resource_Aware_Mode"].fillna("inactive").astype(str).str.strip()
    )
    frame = frame[frame["trade_date"].notna()].copy()
    duplicate = frame.duplicated("trade_date", keep=False)
    if bool(duplicate.any()):
        sample = frame.loc[duplicate, ["trade_date", "resource_aware_mode"]].head(10).to_dict("records")
        raise ValueError(f"daily capacity日期非唯一: sample={sample}")
    return frame[["trade_date", "resource_aware_mode"]]


def _pairwise_within_day(frame: pd.DataFrame, *, outcome: str) -> dict[str, Any]:
    comparable_pairs = 0
    concordant_pairs = 0
    discordant_pairs = 0
    score_tie_pairs = 0
    days_with_pairs = 0
    day_scores: list[float] = []

    for _, day in frame.groupby("trade_date", sort=True):
        pair = day[["breakout_quality_score", outcome]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(pair) < 2:
            continue
        values = pair.to_numpy(dtype=float)
        day_total = 0
        day_concordant = 0
        day_ties = 0
        for left_idx in range(len(values) - 1):
            for right_idx in range(left_idx + 1, len(values)):
                score_diff = values[left_idx, 0] - values[right_idx, 0]
                outcome_diff = values[left_idx, 1] - values[right_idx, 1]
                if not math.isfinite(score_diff) or not math.isfinite(outcome_diff) or outcome_diff == 0:
                    continue
                comparable_pairs += 1
                day_total += 1
                if score_diff == 0:
                    score_tie_pairs += 1
                    day_ties += 1
                elif score_diff * outcome_diff > 0:
                    concordant_pairs += 1
                    day_concordant += 1
                else:
                    discordant_pairs += 1
        if day_total:
            days_with_pairs += 1
            day_scores.append((day_concordant + 0.5 * day_ties) / day_total * 100.0)

    weighted = (
        (concordant_pairs + 0.5 * score_tie_pairs) / comparable_pairs * 100.0
        if comparable_pairs
        else None
    )
    return {
        "comparable_pair_count": int(comparable_pairs),
        "concordant_pair_count": int(concordant_pairs),
        "discordant_pair_count": int(discordant_pairs),
        "score_tie_pair_count": int(score_tie_pairs),
        "days_with_comparable_pairs": int(days_with_pairs),
        "pair_weighted_concordance_pct": weighted,
        "mean_daily_concordance_pct": float(np.mean(day_scores)) if day_scores else None,
    }


def _aggregate_score_groups(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_name, group in frame.groupby("score_quantile", dropna=False, sort=True):
        selected = group[group["is_selected"].eq(True)].copy()
        labels = pd.to_numeric(group["label"], errors="coerce")
        realized = pd.to_numeric(selected["realized_r"], errors="coerce")
        realized = realized[np.isfinite(realized)]
        rows.append(
            {
                "group": str(group_name),
                "candidate_count": int(len(group)),
                "unique_event_count": int(group[list(_EVENT_KEY)].drop_duplicates().shape[0]),
                "selected_count": int(len(selected)),
                "score_min": float(pd.to_numeric(group["breakout_quality_score"], errors="coerce").min()),
                "score_max": float(pd.to_numeric(group["breakout_quality_score"], errors="coerce").max()),
                "label_coverage_pct": float(labels.notna().mean() * 100.0),
                "label_pass_rate_pct": (
                    float((labels.dropna() == 1).mean() * 100.0) if labels.notna().any() else None
                ),
                "selected_realized_r_mean": float(realized.mean()) if len(realized) else None,
                "selected_realized_r_median": float(realized.median()) if len(realized) else None,
            }
        )
    return pd.DataFrame(rows)


def _fmt(value: Any, *, digits: int = 2, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}{suffix}"


def _render_score_groups(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "無可顯示資料"
    rows = []
    for row in frame.to_dict("records"):
        rows.append(
            (
                row["group"],
                row["candidate_count"],
                row["unique_event_count"],
                row["selected_count"],
                f"{_fmt(row['score_min'], digits=3)}～{_fmt(row['score_max'], digits=3)}",
                _fmt(row["label_pass_rate_pct"], suffix="%"),
                _fmt(row["selected_realized_r_mean"], suffix=" R"),
                _fmt(row["selected_realized_r_median"], suffix=" R"),
            )
        )
    return render_table(
        (
            "Score分組",
            "競爭PASS",
            "Unique events",
            "實際買入",
            "Score範圍",
            "原Event Label PASS",
            "買入平均R",
            "買入中位R",
        ),
        rows,
    )


def _render_pairwise(title: str, payload: dict[str, Any]) -> str:
    return "\n\n".join(
        (
            render_section(title),
            render_key_values(
                (
                    ("可比較pair", payload["comparable_pair_count"]),
                    ("涉及日期", payload["days_with_comparable_pairs"]),
                    ("Concordant", payload["concordant_pair_count"]),
                    ("Discordant", payload["discordant_pair_count"]),
                    ("Score ties", payload["score_tie_pair_count"]),
                    ("Pair-weighted concordance", _fmt(payload["pair_weighted_concordance_pct"], suffix="%")),
                    ("每日平均concordance", _fmt(payload["mean_daily_concordance_pct"], suffix="%")),
                )
            ),
        )
    )


def _render_report(payload: dict[str, Any]) -> str:
    overview = dict(payload["overview"])
    diagnostics = dict(payload["diagnostics"])
    return "\n\n".join(
        (
            render_title("A9 DL Selection Confidence Audit"),
            render_key_values(
                (
                    ("Audit ID", payload["audit_id"]),
                    ("來源Arm", payload["source"]["arm_id"]),
                    ("DL", payload["source"]["dl_id"]),
                    ("Runtime", payload["source"]["runtime"]),
                    ("Threshold", payload["source"]["threshold"]),
                    ("Competition定義", f"Resource_Aware_Mode=dl-selection 且當日至少 {overview['minimum_competing_pass_candidates']} 個PASS"),
                    ("Confidence語意", "沿用原breakout event A9 score；不對extended當日線型重新推論"),
                )
            ),
            render_section("1. Competition總覽"),
            render_key_values(
                (
                    ("DL Selection days", overview["dl_selection_day_count"]),
                    ("PASS competition days", overview["competition_day_count"]),
                    ("Competition PASS candidate-days", overview["competition_pass_candidate_count"]),
                    ("Competition unique events", overview["competition_unique_event_count"]),
                    ("Selected PASS", overview["selected_pass_count"]),
                    ("Selected PASS with Realized R", overview["selected_pass_with_realized_r_count"]),
                    ("Label coverage", _fmt(overview["label_coverage_pct"], suffix="%")),
                )
            ),
            render_section("2. Confidence相關性"),
            render_key_values(
                (
                    ("Candidate-day Score ↔ Event Label", _fmt(diagnostics["candidate_day_score_vs_label_spearman"], digits=3)),
                    ("Unique-event Score ↔ Event Label", _fmt(diagnostics["unique_event_score_vs_label_spearman"], digits=3)),
                    ("Selected Score ↔ Realized R", _fmt(diagnostics["selected_score_vs_realized_r_spearman"], digits=3)),
                )
            ),
            _render_pairwise("3. 同日 PASS：Confidence 是否把 true Event PASS 排前面", payload["label_pairwise"]),
            _render_pairwise("4. 同日已買 PASS：Confidence 是否把較高 Realized R 排前面", payload["realized_r_pairwise"]),
            render_section("5. Competition Score分組"),
            _render_score_groups(pd.DataFrame(payload.get("score_groups") or [])),
            render_section("6. 使用限制"),
            (
                "本Audit只讀SR-C12既有正式策略比較工件，不重跑策略、不改DL-A9、不改threshold或allocation。"
                "Competition只限Resource-aware runtime已判定為dl-selection且同日至少兩個PASS候選，避免用沒有選擇問題的日期稀釋結果。"
                "原Event Label可覆蓋全部已知Label的競爭PASS，但只描述breakout-event品質；Realized R只存在於實際成交者，"
                "因此Score↔Realized R與同日R concordance存在portfolio selection bias，不能當成未成交PASS的反事實。"
                "Score分組與任何concordance結果只供決定是否值得建立下一個策略arm，不得直接轉成新threshold、DL／Min ROOS混合權重或candidate失效規則。"
            ),
        )
    ).rstrip() + "\n"


def run_selection_confidence_audit(
    definition: AuditDefinition,
    *,
    project_root: Path = PROJECT_ROOT,
    quiet: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    status = collect_selection_confidence_status(definition, project_root=root)
    if status.get("status") != "READY":
        raise RuntimeError(str(status.get("reason") or "Audit來源未READY"))

    paths, threshold, _orderable, pass_candidates = prepare_pass_candidate_audit_frame(
        definition, project_root=root
    )
    daily_capacity = _load_dl_selection_days(Path(str(status["paths"]["daily_capacity"])))
    dl_selection_days = set(
        daily_capacity.loc[
            daily_capacity["resource_aware_mode"].eq("dl-selection"), "trade_date"
        ].tolist()
    )
    minimum_competing = int(definition.dimensions["minimum_competing_pass_candidates"])
    score_groups_n = int(definition.dimensions["score_quantile_groups"])

    pass_counts = pass_candidates.groupby("trade_date", sort=False).size()
    competition_days = {
        date for date, count in pass_counts.items()
        if date in dl_selection_days and int(count) >= minimum_competing
    }
    competition = pass_candidates[pass_candidates["trade_date"].isin(competition_days)].copy()
    if competition.empty:
        raise RuntimeError("Audit來源沒有符合定義的DL Selection PASS competition days")
    competition["competing_pass_count_today"] = competition["trade_date"].map(pass_counts).astype(int)
    competition["score_quantile"] = audit_quantile_labels(
        competition["breakout_quality_score"], score_groups_n, "Q"
    )

    event_score_counts = competition.groupby(list(_EVENT_KEY), dropna=False)["breakout_quality_score"].nunique(dropna=True)
    conflict_count = int((event_score_counts > 1).sum())
    if conflict_count:
        raise ValueError(
            "A9 selection-confidence audit要求原breakout event confidence為靜態；"
            f"發現{conflict_count}個event在candidate-days間score不一致"
        )
    unique_events = competition.sort_values(
        ["signal_date", "ticker", "high_len", "trade_date"], kind="mergesort"
    ).drop_duplicates(list(_EVENT_KEY), keep="first")
    selected = competition[competition["is_selected"].eq(True)].copy()
    selected_realized = pd.to_numeric(selected["realized_r"], errors="coerce")
    selected_with_r = selected[np.isfinite(selected_realized)].copy()
    labels = pd.to_numeric(competition["label"], errors="coerce")

    diagnostics = {
        "candidate_day_score_vs_label_spearman": audit_spearman(
            competition, "breakout_quality_score", "label"
        ),
        "unique_event_score_vs_label_spearman": audit_spearman(
            unique_events, "breakout_quality_score", "label"
        ),
        "selected_score_vs_realized_r_spearman": audit_spearman(
            selected_with_r, "breakout_quality_score", "realized_r"
        ),
        "event_static_score_conflict_count": conflict_count,
    }
    label_pairwise = _pairwise_within_day(competition, outcome="label")
    realized_pairwise = _pairwise_within_day(selected_with_r, outcome="realized_r")
    score_groups = _aggregate_score_groups(competition)

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
            "strategy_compare_run": project_relative_display_path(Path(paths["run_dir"]), project_root=root),
            "daily_capacity": project_relative_display_path(Path(str(status["paths"]["daily_capacity"])), project_root=root),
        },
        "semantic_contract": {
            "strategy_owns_candidate_validity": True,
            "dl_confidence_is_original_breakout_event_score": True,
            "extended_candidate_is_not_rescored_as_breakout": True,
            "portfolio_selector_owns_allocation": True,
            "unselected_realized_r_is_not_imputed": True,
            "audit_does_not_define_runtime_threshold_or_mixing_weight": True,
        },
        "overview": {
            "minimum_competing_pass_candidates": minimum_competing,
            "dl_selection_day_count": int(len(dl_selection_days)),
            "competition_day_count": int(len(competition_days)),
            "competition_pass_candidate_count": int(len(competition)),
            "competition_unique_event_count": int(len(unique_events)),
            "selected_pass_count": int(len(selected)),
            "selected_pass_with_realized_r_count": int(len(selected_with_r)),
            "label_coverage_pct": float(labels.notna().mean() * 100.0),
        },
        "diagnostics": diagnostics,
        "label_pairwise": label_pairwise,
        "realized_r_pairwise": realized_pairwise,
        "score_groups": score_groups.to_dict("records"),
    }
    payload = json_native_audit_value(payload)

    output_root = root / Path(AUDIT_OUTPUT_ROOT) / Path(definition.output_subdir)
    timestamp = get_taipei_now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / "runs" / timestamp
    latest_dir = output_root / "latest"
    run_dir.mkdir(parents=True, exist_ok=False)

    report_path = run_dir / "audit.md"
    json_path = run_dir / "audit.json"
    detail_path = run_dir / "competition_pass_candidates.csv"
    score_path = run_dir / "score_groups.csv"
    report_path.write_text(_render_report(payload), encoding="utf-8")
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    competition.sort_values(
        ["trade_date", "ticker", "signal_date"], kind="mergesort"
    ).to_csv(detail_path, index=False, encoding="utf-8-sig")
    score_groups.to_csv(score_path, index=False, encoding="utf-8-sig")

    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    latest_dir.mkdir(parents=True, exist_ok=True)
    for source in (report_path, json_path, detail_path, score_path):
        shutil.copy2(source, latest_dir / source.name)

    if not quiet:
        print("\n" + _render_report(payload))
        print_artifact_paths(
            (
                ("Audit Markdown", report_path),
                ("Audit JSON", json_path),
                ("Competition PASS", detail_path),
                ("最新Audit", latest_dir),
            ),
            project_root=root,
        )
    return payload


__all__ = [
    "collect_selection_confidence_status",
    "run_selection_confidence_audit",
]
