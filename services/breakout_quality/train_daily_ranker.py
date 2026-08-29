"""Profile-driven daily-universal continuous-ranker orchestration.

Learning/loss/epoch-selection stay centralized in ``train_continuous_ranker``. This module
only owns the daily stock/day sample universe, no-lookahead split, and daily/candidate
validation views so additional Daily DL experiments do not fork a new trainer.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from config.breakout_quality import (
    CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
    TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION,
    TRAINING_OBJECTIVE_DAILY_DUAL_COMPONENT_R_REGRESSION,
    TRAINING_OBJECTIVE_DAILY_PARETO_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_SAFETY_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SAFETY_CONDITIONAL_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_HMHS_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_JOINT_MIN_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_HMHS_PAIRWISE_RANKING,
    PREDICTED_SAFETY_CONTEXT_PURE_MFE_TARGET_ID,
    get_continuous_ranker_execution_recipe,
    get_continuous_ranker_research_spec,
)
from filters.breakout_quality.artifacts import build_file_manifest
from filters.breakout_quality.contract import (
    ARTIFACT_CONTRACT_VERSION,
    FEATURE_COLUMNS,
    FILTER_FAMILY,
)
from filters.breakout_quality.daily_ranker_data import (
    build_daily_ranker_split,
    load_official_breakout_candidate_keys,
    load_daily_universal_ranker_data,
    select_breakout_candidate_group_ids,
)
from filters.breakout_quality.continuous_ranker_data import build_same_date_percentile_targets
from filters.breakout_quality.models.factory import count_trainable_parameters
from filters.breakout_quality.ranker_sample_contract import (
    build_score_eligibility_contract,
    resolve_forward_oos_score_group_ids,
)
from filters.breakout_quality.paths import resolve_filter_model_output_dir
from filters.breakout_quality.ranking_score_store import DAILY_RANKER_OOS_SCORE_FILENAME
from filters.breakout_quality.workflow_io import PROJECT_ROOT, write_json
from core.console_report import print_artifact_paths
from core.report_style import markdown_tone, signal_for_delta, styled_signal

from services.breakout_quality import ranker_training as ranker_api
from services.breakout_quality.continuous_ranker_pipeline import resolve_ranker_execution_plan

DAILY_SPLIT_FILENAME = "daily_split_by_date.csv"
MR13S_TRUTH_GEOMETRY_JSON_FILENAME = "mr13s_truth_geometry_control.json"
MR13S_TRUTH_GEOMETRY_MARKDOWN_FILENAME = "mr13s_truth_geometry_control.md"




def _empty_split_metrics(group_count: int, reason: str) -> dict:
    return {
        "group_count": int(group_count),
        "not_evaluated_reason": str(reason),
        "percentile_target_count": 0,
        "mse_vs_daily_percentile": None,
        "global_spearman_vs_raw_target": None,
        "global_spearman_vs_daily_percentile": None,
        "rankable_date_count": 0,
        "mean_daily_spearman": None,
        "median_daily_spearman": None,
        "pairwise_concordance": None,
        "comparable_pair_count": 0,
        "top_k_quality": None,
        "score_mean": None,
        "score_std": None,
        "top_score_decile_raw_target_mean": None,
        "bottom_score_decile_raw_target_mean": None,
        "binary_pr_auc": None,
        "p_at_50pct": None,
        "p_at_60pct": None,
        "p_at_70pct": None,
        "raw_r_regression": None,
    }


def build_safety_raw_mfe_truth_geometry_control_from_score_frame(
    score_frame: pd.DataFrame,
    *,
    breakout_candidate_keys: set[tuple[str, pd.Timestamp]],
) -> dict:
    """Build actual Daily/Breakout 5×5 geometry from persisted MR-13S OOS targets.

    Breakout rows are filtered *after* the canonical daily-universal target percentiles
    have been loaded.  Percentiles are never recomputed inside the breakout subset.
    """

    required = {
        "ticker",
        "date",
        "target_low_adverse_safety_percentile",
        "target_pure_mfe_percentile",
        "raw_safety_score",
        "raw_mfe_score",
    }
    missing = sorted(required.difference(score_frame.columns))
    if missing:
        raise ValueError(f"MR-13S Truth Geometry score artifact缺少欄位: {missing}")
    frame = score_frame.loc[:, sorted(required)].copy()
    frame["ticker"] = frame["ticker"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    for column in (
        "target_low_adverse_safety_percentile",
        "target_pure_mfe_percentile",
        "raw_safety_score",
        "raw_mfe_score",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    evaluable = frame[
        np.isfinite(frame["target_low_adverse_safety_percentile"].to_numpy(dtype=np.float64))
        & np.isfinite(frame["target_pure_mfe_percentile"].to_numpy(dtype=np.float64))
    ].copy()
    if evaluable.empty:
        raise ValueError("MR-13S Truth Geometry沒有target-evaluable OOS rows")

    candidate_mask = np.fromiter(
        (
            (ticker, date) in breakout_candidate_keys
            for ticker, date in zip(evaluable["ticker"], evaluable["date"])
        ),
        dtype=bool,
        count=len(evaluable),
    )
    breakout = evaluable.loc[candidate_mask].copy()

    def build_scope(rows: pd.DataFrame) -> dict:
        dates = rows["date"].to_numpy()
        safety_true = rows["target_low_adverse_safety_percentile"].to_numpy(dtype=np.float64)
        mfe_true = rows["target_pure_mfe_percentile"].to_numpy(dtype=np.float64)
        actual = ranker_api.safety_mfe_truth_geometry(dates, safety_true, mfe_true)
        safety_pred = rows["raw_safety_score"].to_numpy(dtype=np.float64)
        mfe_pred = rows["raw_mfe_score"].to_numpy(dtype=np.float64)
        pred_valid = np.isfinite(safety_pred) & np.isfinite(mfe_pred)
        predicted_relation = (
            ranker_api.daily_rank_metrics(
                dates[pred_valid], safety_pred[pred_valid], mfe_pred[pred_valid]
            )
            if bool(pred_valid.any())
            else {}
        )
        return {
            "population_n": int(len(rows)),
            "actual": actual,
            "predicted_safety_to_raw_mfe_mean_daily_spearman": predicted_relation.get("mean_daily_spearman"),
            "predicted_relation_valid_days": int(predicted_relation.get("rankable_date_count", 0) or 0),
        }

    return {
        "daily_universal_oos": build_scope(evaluable),
        "breakout_candidate_oos": build_scope(breakout),
        "breakout_percentile_policy": "filter_daily_universal_percentiles_without_subset_rerank",
    }


def _render_truth_geometry_markdown(payload: dict) -> str:
    def fmt(value, digits: int = 4) -> str:
        return "-" if value is None else f"{float(value):.{digits}f}"

    def cell_text(cell: dict) -> str:
        if not cell:
            return "0 / - / -"
        return (
            f"{int(cell.get('n', 0) or 0):,} / "
            f"{fmt(cell.get('population_pct'), 2)}% / "
            f"{fmt(cell.get('independence_enrichment'), 2)}×"
        )

    lines = [
        "# MR-13S Actual MFE×Safety Truth Geometry Control",
        "",
        "此報表只讀既有MR-13S frozen Forward-OOS score artifact與canonical breakout membership；"
        "不訓練模型、不重算breakout subset percentile，也不建立PIT／Strategy arm。",
        "",
        "## Summary",
        "",
        "| Scope | N | Actual Safety↔MFE Dailyρ | Pred Safety↔Raw-MFE Dailyρ | S5×M5 N / Pop / Enrich | S4+×M4+ N / Pop / Enrich |",
        "|---|---:|---:|---:|---|---|",
    ]
    for label, key in (("Daily universal OOS", "daily_universal_oos"), ("Breakout candidate OOS", "breakout_candidate_oos")):
        scope = dict(payload.get(key) or {})
        actual = dict(scope.get("actual") or {})
        lines.append(
            f"| {label} | {int(scope.get('population_n', 0) or 0):,} "
            f"| {fmt(actual.get('safety_to_mfe_mean_daily_spearman'))} "
            f"| {fmt(scope.get('predicted_safety_to_raw_mfe_mean_daily_spearman'))} "
            f"| {cell_text(dict(actual.get('s5_m5') or {}))} "
            f"| {cell_text(dict(actual.get('s4plus_m4plus') or {}))} |"
        )
    for label, key in (("Daily universal OOS", "daily_universal_oos"), ("Breakout candidate OOS", "breakout_candidate_oos")):
        scope = dict(payload.get(key) or {})
        actual = dict(scope.get("actual") or {})
        lines.extend([
            "",
            f"## {label} actual 5×5",
            "",
            "Cell = `N / population% / independence enrichment×`。Expected使用該scope實際row/column marginals；Breakout不重新排名percentile。",
            "",
            "| Actual Safety \\ Pure-MFE | M1 | M2 | M3 | M4 | M5 |",
            "|---|---|---|---|---|---|",
        ])
        for s_idx, row in enumerate(list(actual.get("actual_joint_geometry") or []), start=1):
            lines.append(f"| S{s_idx} | " + " | ".join(cell_text(dict(cell or {})) for cell in row) + " |")
    lines.extend([
        "",
        "## Contract",
        "",
        "- Diagnostic only；不得以此結果回頭fit MR-13S training、threshold、weight或calibration。",
        "- Breakout candidate只做membership filter，沿用daily-universal同日Safety/MFE percentile。",
        "- Independence enrichment=`observed N / (N × P(S-bin) × P(M-bin))`。",
        "",
    ])
    return "\n".join(lines)


def build_mr13s_truth_geometry_control(
    *,
    filter_id: str,
    model_architecture: str,
    experiment_profile: str,
    allow_stale_source: bool = False,
    project_root: str | Path = PROJECT_ROOT,
) -> tuple[dict, Path, Path]:
    """Build the read-only MR-13S actual truth-geometry control from frozen artifacts."""

    root = Path(project_root).resolve()
    output_dir = resolve_filter_model_output_dir(
        root, str(filter_id), str(model_architecture), str(experiment_profile)
    )
    report_path = output_dir / ranker_api.RANKER_REPORT_JSON_FILENAME
    score_path = output_dir / DAILY_RANKER_OOS_SCORE_FILENAME
    if not report_path.is_file() or not score_path.is_file():
        raise FileNotFoundError("MR-13S Truth Geometry需要既有continuous_ranker_report.json與daily_ranker_oos_scores.csv.gz")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_identity = {
        "filter_id": str(filter_id),
        "model_architecture": str(model_architecture),
        "experiment_profile": str(experiment_profile),
    }
    for field, expected in expected_identity.items():
        if str(report.get(field) or "") != expected:
            raise ValueError(f"MR-13S Truth Geometry report identity不一致: {field}")
    if str((report.get("training") or {}).get("objective") or "") != TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_PAIRWISE_RANKING:
        raise ValueError("Actual MFE×Safety Truth Geometry目前只適用Safety→Raw-MFE duo-head profile")
    recorded_score = dict((report.get("artifacts") or {}).get("oos_scores_gzip") or {})
    if recorded_score and build_file_manifest(score_path) != recorded_score:
        raise ValueError("MR-13S Truth Geometry OOS score artifact與frozen report manifest不一致")

    score_frame = pd.read_csv(score_path, compression="gzip", dtype={"ticker": str})
    candidate_keys = load_official_breakout_candidate_keys(
        str(filter_id), allow_stale_source=bool(allow_stale_source)
    )
    control = build_safety_raw_mfe_truth_geometry_control_from_score_frame(
        score_frame, breakout_candidate_keys=candidate_keys
    )
    raw_eval = dict(report.get("safety_raw_mfe_evaluation") or {})
    for scope_key in ("oos", "breakout_candidate_oos"):
        prior_n = int((((raw_eval.get(scope_key) or {}).get("model_gate") or {}).get("population_n") or 0))
        current_n = int((control.get("daily_universal_oos" if scope_key == "oos" else "breakout_candidate_oos") or {}).get("population_n", 0) or 0)
        if prior_n and current_n != prior_n:
            raise ValueError(
                f"MR-13S Truth Geometry {scope_key} membership與原frozen model report不一致: expected={prior_n}, actual={current_n}"
            )

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        **expected_identity,
        "status": "RESULT_AVAILABLE_PENDING_REVIEW",
        "diagnostic": "MR-13S actual MFE×Safety truth geometry control",
        "source_model_report": build_file_manifest(report_path),
        "source_oos_scores": build_file_manifest(score_path),
        "source_breakout_membership": "canonical breakout_quality validated event ticker/date membership",
        "oos_used_for_training_or_epoch_selection": False,
        "model_weights_changed": False,
        **control,
    }
    json_path = output_dir / MR13S_TRUTH_GEOMETRY_JSON_FILENAME
    markdown_path = output_dir / MR13S_TRUTH_GEOMETRY_MARKDOWN_FILENAME
    write_json(json_path, payload)
    markdown_path.write_text(_render_truth_geometry_markdown(payload), encoding="utf-8")
    return payload, json_path, markdown_path


def _date_split_frame(
    bundle,
    split,
    *,
    forward_score_ids: np.ndarray | None = None,
) -> pd.DataFrame:
    dates = pd.to_datetime(bundle.group_table["date"], errors="raise").dt.normalize()
    selection_dates = set(dates.iloc[split.selection_ids].tolist())
    train_dates = set(dates.iloc[split.inner_train_ids].tolist())
    validation_dates = set(dates.iloc[split.validation_ids].tolist())
    oos_group_ids = (
        split.oos_ids
        if forward_score_ids is None
        else np.asarray(forward_score_ids, dtype=np.int64)
    )
    oos_dates = set(dates.iloc[oos_group_ids].tolist())
    rows = []
    for date_value in sorted(selection_dates | oos_dates):
        if date_value in oos_dates:
            outer_split, role = "oos", "not_applicable"
        elif date_value in validation_dates:
            outer_split, role = "selection", "validation"
        elif date_value in train_dates:
            outer_split, role = "selection", "train"
        else:
            outer_split, role = "selection", "embargo"
        rows.append({"date": str(pd.Timestamp(date_value).date()), "outer_split": outer_split, "selection_role": role})
    return pd.DataFrame(rows)


def _dual_component_metrics(
    group_table: pd.DataFrame,
    group_ids: np.ndarray,
    predicted_favorable_r: np.ndarray,
    predicted_adverse_r: np.ndarray,
) -> dict:
    """Evaluate MR-13L primary component heads without changing the composite Gate."""

    ids = np.asarray(group_ids, dtype=np.int64)
    favorable_pred = np.asarray(predicted_favorable_r, dtype=np.float64)
    adverse_pred = np.asarray(predicted_adverse_r, dtype=np.float64)
    if favorable_pred.shape != ids.shape or adverse_pred.shape != ids.shape:
        raise ValueError("dual-component prediction length與group ids不一致")
    favorable_true = pd.to_numeric(
        group_table.iloc[ids]["target_favorable_r"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    adverse_true = pd.to_numeric(
        group_table.iloc[ids]["target_adverse_r"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    dates = pd.to_datetime(group_table.iloc[ids]["date"], errors="raise").to_numpy()

    def one_component(predicted: np.ndarray, actual: np.ndarray) -> dict:
        valid = np.isfinite(predicted) & np.isfinite(actual)
        if not bool(valid.any()):
            return {
                "count": 0,
                "mse": None,
                "mae": None,
                "rmse": None,
                "bias": None,
                "global_spearman": None,
                "mean_daily_spearman": None,
            }
        errors = predicted[valid] - actual[valid]
        daily = ranker_api.daily_rank_metrics(
            dates[valid], predicted[valid], actual[valid]
        )
        mse = float(np.mean(np.square(errors)))
        return {
            "count": int(valid.sum()),
            "mse": mse,
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(mse)),
            "bias": float(np.mean(errors)),
            "global_spearman": ranker_api.calculate_spearman(
                predicted[valid], actual[valid]
            ),
            "mean_daily_spearman": daily.get("mean_daily_spearman"),
        }

    return {
        "favorable_mfe_r": one_component(favorable_pred, favorable_true),
        "adverse_to_peak_r": one_component(adverse_pred, adverse_true),
    }



def _predicted_safety_context_pure_mfe_metrics(
    group_ids: np.ndarray,
    group_table: pd.DataFrame,
    scores: np.ndarray,
) -> dict:
    """MR-13AE model-only MFE/Safety conversion diagnostics; never used for fitting."""

    ids = np.asarray(group_ids, dtype=np.int64)
    score = np.asarray(scores, dtype=np.float64)
    if score.shape != ids.shape:
        raise ValueError("MR-13AE score/group length mismatch")
    rows = group_table.iloc[ids]
    dates = pd.to_datetime(rows["date"], errors="raise").to_numpy()
    favorable = pd.to_numeric(rows["target_favorable_r"], errors="coerce").to_numpy(dtype=np.float64)
    adverse = pd.to_numeric(rows["target_adverse_r"], errors="coerce").to_numpy(dtype=np.float64)
    safety_pct = pd.to_numeric(rows["target_low_adverse_daily_percentile"], errors="coerce").to_numpy(dtype=np.float64)
    mfe_pct = pd.to_numeric(rows["target_mfe_daily_percentile"], errors="coerce").to_numpy(dtype=np.float64)
    context = pd.to_numeric(rows["predicted_safety_percentile"], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(score) & np.isfinite(favorable) & np.isfinite(adverse) & np.isfinite(safety_pct) & np.isfinite(mfe_pct) & np.isfinite(context)
    if int(valid.sum()) < 2:
        return {"group_count": int(valid.sum()), "available": False}
    score = score[valid]
    dates = dates[valid]
    favorable = favorable[valid]
    adverse = adverse[valid]
    safety_pct = safety_pct[valid]
    mfe_pct = mfe_pct[valid]
    context = context[valid]
    n = len(score)
    # Strategy selection is cross-sectional by date; a global score decile would mix
    # incomparable score levels from different dates.  Use the canonical same-date
    # average-rank percentile and inspect the daily top decile only.
    score_pct = build_same_date_percentile_targets(
        score, np.ones(n, dtype=bool), dates
    ).astype(np.float64)
    top = np.flatnonzero(score_pct >= 0.90)
    if len(top) == 0:
        return {"group_count": int(n), "available": False}
    top_n = int(len(top))
    hm = mfe_pct >= 0.50
    hs = safety_pct >= 0.50
    safety_rank = ranker_api.daily_rank_metrics(dates, score, -adverse)
    context_score_rank = ranker_api.daily_rank_metrics(dates, context, score)
    return {
        "available": True,
        "group_count": int(n),
        "score_to_low_adverse_mean_daily_spearman": safety_rank.get("mean_daily_spearman"),
        "predicted_safety_to_model_score_mean_daily_spearman": context_score_rank.get("mean_daily_spearman"),
        "population": {
            "full_mfe_r_mean": float(np.mean(favorable)),
            "adverse_r_mean": float(np.mean(adverse)),
            "high_mfe_pct": float(np.mean(hm) * 100.0),
            "high_safety_pct": float(np.mean(hs) * 100.0),
            "hmhs_pct": float(np.mean(hm & hs) * 100.0),
        },
        "top_10pct": {
            "n": int(top_n),
            "full_mfe_r_mean": float(np.mean(favorable[top])),
            "adverse_r_mean": float(np.mean(adverse[top])),
            "high_mfe_pct": float(np.mean(hm[top]) * 100.0),
            "high_safety_pct": float(np.mean(hs[top]) * 100.0),
            "hmhs_pct": float(np.mean((hm & hs)[top]) * 100.0),
        },
        "status": "diagnostic_only_no_fit_no_threshold_selection",
    }

def _render_markdown(payload: dict) -> str:
    def fmt(value, digits=4):
        return "-" if value is None else f"{float(value):.{digits}f}"

    objective = str(payload["training"].get("objective") or "")
    direct_r = objective == TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION
    dual_component_r = objective == TRAINING_OBJECTIVE_DAILY_DUAL_COMPONENT_R_REGRESSION
    conditional_mfe_safety = (
        objective == TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_SAFETY_PAIRWISE_RANKING
    )
    conditional_mfe_single = objective == TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_PAIRWISE_RANKING
    safety_conditional_mfe_duo = objective == TRAINING_OBJECTIVE_DAILY_SAFETY_CONDITIONAL_MFE_PAIRWISE_RANKING
    safety_raw_mfe_duo = objective == TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_PAIRWISE_RANKING
    safety_raw_mfe_hmhs_tri = objective == TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_HMHS_PAIRWISE_RANKING
    safety_raw_mfe_joint_min_tri = objective == TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_JOINT_MIN_PAIRWISE_RANKING
    direct_hmhs_only = objective == TRAINING_OBJECTIVE_DAILY_HMHS_PAIRWISE_RANKING
    source_dataset = dict(payload.get("source_dataset") or {})
    target_manifest = dict(payload.get("target_manifest") or {})
    def section(title: str, *, level: int = 2) -> str:
        return f"{'#' * int(level)} {markdown_tone(title, 'blue', bold=True)}"

    def delta(left, right, *, percent=False):
        if left is None or right is None:
            return "-"
        value = float(right) - float(left)
        text = f"{value * 100:+.2f}pp" if percent else f"{value:+.4f}"
        return styled_signal(
            text,
            signal_for_delta(value, preference="higher"),
            target="markdown",
            bold=True,
        )

    lines = [
        f"# {markdown_tone('Detailed Model Research Report', 'blue', bold=True)}",
        "",
        f"- Experiment：`{payload['experiment']}`",
        f"- Profile：`{payload['experiment_profile']}`",
        f"- Sample scope：`{payload['training']['sample_scope']}`",
        f"- Target：`{payload['training']['target']}`",
        f"- Target-valid / score-eligible：`{int(source_dataset.get('target_valid_sample_count', 0) or 0):,}` / `{int(source_dataset.get('score_eligible_sample_count', 0) or 0):,}`",
        f"- Risk-param coverage start：`{source_dataset.get('risk_param_coverage_start') or '-'}`",
        f"- Context features：`{', '.join(target_manifest.get('context_features') or []) or '-'}`",
        f"- Score semantic：`{payload.get('score_semantic_id')}`",
        f"- Selected epoch：`{payload['training']['selected_epoch']}`",
        "- Feature storage：`lazy canonical OHLCV windows`；未建立 expanded daily 300×10 feature bank。",
        "- OOS 在 checkpoint 寫入後才推論，不參與 loss／gradient／epoch selection。",
    ]
    lines.extend([
        "",
        section("1. Learnability"),
        "",
        "| Scope | Groups | Daily rho | Global rho | Pair concordance | Top 10% Target | Bottom 10% Target |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for label, key in (
        ("Validation", "validation"),
        ("Forward OOS", "oos"),
        ("Breakout candidate slice", "breakout_candidate_oos"),
    ):
        row = dict((payload.get("split_metrics") or {}).get(key) or {})
        if not row:
            continue
        pair = row.get("pairwise_concordance")
        lines.append(
            f"| {label} | {int(row.get('group_count', 0) or 0):,} "
            f"| {fmt(row.get('mean_daily_spearman'))} "
            f"| {fmt(row.get('global_spearman_vs_raw_target'))} "
            f"| {'-' if pair is None else f'{float(pair)*100:.2f}%'} "
            f"| {fmt(row.get('top_score_decile_raw_target_mean'))} "
            f"| {fmt(row.get('bottom_score_decile_raw_target_mean'))} |"
        )

    validation = dict((payload.get("split_metrics") or {}).get("validation") or {})
    oos = dict((payload.get("split_metrics") or {}).get("oos") or {})
    breakout = dict((payload.get("split_metrics") or {}).get("breakout_candidate_oos") or {})
    lines.extend([
        "",
        section("2. Generalization"),
        "",
        "| Comparison | Δ Daily rho | Δ Pair |",
        "|---|---:|---:|",
        f"| Validation → Forward OOS | {delta(validation.get('mean_daily_spearman'), oos.get('mean_daily_spearman'))} "
        f"| {delta(validation.get('pairwise_concordance'), oos.get('pairwise_concordance'), percent=True)} |",
    ])
    if breakout:
        lines.append(
            f"| Forward OOS → Breakout slice | {delta(oos.get('mean_daily_spearman'), breakout.get('mean_daily_spearman'))} "
            f"| {delta(oos.get('pairwise_concordance'), breakout.get('pairwise_concordance'), percent=True)} |"
        )
    ae_eval = dict(payload.get("predicted_safety_context_pure_mfe_evaluation") or {})
    if ae_eval:
        lines.extend([
            "",
            section(f"Model-specific Extension｜{payload['model_research_id']}｜Pure-MFE × Safety Context"),
            "",
            "- Target/order與MR-13K相同；Predicted Safety只作PIT-safe input context，不參與target residualization。",
            "- Safety metrics只作checkpoint寫入後診斷，不參與loss、epoch selection、threshold或calibration。",
            "",
            "| Scope | Score→Low-Adverse Daily rho | Pred-Safety→Score Daily rho | Top10 MFE | Top10 Adverse | Top10 High-MFE | Top10 High-Safety | Top10 HM/HS |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for label, key in (("Validation", "validation"), ("Forward OOS", "oos"), ("Breakout candidate", "breakout_candidate_oos")):
            row = dict(ae_eval.get(key) or {})
            top = dict(row.get("top_10pct") or {})
            pct = lambda v: "-" if v is None else f"{float(v):.2f}%"
            lines.append(
                f"| {label} | {fmt(row.get('score_to_low_adverse_mean_daily_spearman'))} "
                f"| {fmt(row.get('predicted_safety_to_model_score_mean_daily_spearman'))} "
                f"| {fmt(top.get('full_mfe_r_mean'))}R | {fmt(top.get('adverse_r_mean'))}R "
                f"| {pct(top.get('high_mfe_pct'))} | {pct(top.get('high_safety_pct'))} | {pct(top.get('hmhs_pct'))} |"
            )

    if direct_hmhs_only:
        lines.extend([
            "",
            section(f"Model-specific Extension｜{payload['model_research_id']}｜Direct HM/HS H-only Learnability"),
            "",
            "- H-only control：raw 300×10與InceptionTime trunk不變；移除Safety/MFE heads與loss，encoder只接受Direct HM/HS supervision。",
            "- Epoch selection固定為Validation HM/HS Pair concordance；同值才以Validation global PR-AUC tie-break；OOS不參與selection。",
            "",
            "| Scope | HM/HS Pop | Pair | Global PR-AUC | Mean Daily PR-AUC | Top 10% HM/HS / × | Top 20% HM/HS / × |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for label, key in (("Validation", "validation"), ("Forward OOS", "oos"), ("Breakout candidate slice", "breakout_candidate_oos")):
            row = dict((payload.get("split_metrics") or {}).get(key) or {})
            top10 = dict(row.get("top_10pct") or {})
            top20 = dict(row.get("top_20pct") or {})
            def pct100(value):
                return "-" if value is None else f"{float(value):.2f}%"
            pair = row.get("pairwise_concordance")
            lines.append(
                f"| {label} | {pct100(row.get('population_hmhs_pct'))} "
                f"| {'-' if pair is None else f'{float(pair)*100:.2f}%'} "
                f"| {fmt(row.get('global_average_precision'))} "
                f"| {fmt(row.get('mean_daily_average_precision'))} "
                f"| {pct100(top10.get('hmhs_pct'))} / {fmt(top10.get('hmhs_enrichment'), 2)}× "
                f"| {pct100(top20.get('hmhs_pct'))} / {fmt(top20.get('hmhs_enrichment'), 2)}× |"
            )
    if direct_r:
        regression_contract = dict(payload["training"].get("raw_r_regression_contract") or {})
        loss_name = str(regression_contract.get("loss") or payload["training"].get("loss") or "")
        loss_detail = (
            f"Huber delta={fmt(regression_contract.get('huber_delta_r'))}R"
            if loss_name == "huber_raw_r"
            else "raw-R MSE / conditional-mean objective"
        )
        lines.extend([
            f"- Direct-R objective：two-logit margin直接解讀為Predicted R；loss=`{loss_name}`（{loss_detail}）。",
            "",
            section("Direct-R Regression"),
            "",
            "| Scope | Groups | MSE | Huber | MAE | RMSE | Bias | Pred R Mean | Target R Mean |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for label, key in (("Validation", "validation"), ("Forward OOS", "oos"), ("Breakout candidate slice", "breakout_candidate_oos")):
            row = payload["split_metrics"].get(key) or {}
            reg = dict(row.get("raw_r_regression") or {})
            lines.append(
                f"| {label} | {int(row.get('group_count', 0) or 0):,} | {fmt(reg.get('mse_raw_r'))} | {fmt(reg.get('huber_loss_raw_r'))} "
                f"| {fmt(reg.get('mae_raw_r'))} | {fmt(reg.get('rmse_raw_r'))} | {fmt(reg.get('bias_raw_r'))} "
                f"| {fmt(reg.get('predicted_r_mean'))} | {fmt(reg.get('target_r_mean'))} |"
            )
    if dual_component_r:
        lines.extend([
            "- Dual-component objective：兩個既有輸出分別為Predicted adverse-to-peak R與Predicted MFE R；"
            "正式score固定為MFE R−adverse R；兩分量MSE採同一mean reduction，無auxiliary loss／lambda。",
            "",
            section("Dual Component Regression"),
            "",
            "| Scope | Component | Groups | RMSE | MAE | Bias | Global rho | Daily rho |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ])
        component_eval = dict(payload.get("dual_component_evaluation") or {})
        for scope_label, scope_key in (
            ("Validation", "validation"),
            ("Forward OOS", "oos"),
            ("Breakout candidate slice", "breakout_candidate_oos"),
        ):
            scope = dict(component_eval.get(scope_key) or {})
            for component_label, component_key in (
                ("MFE", "favorable_mfe_r"),
                ("Adverse", "adverse_to_peak_r"),
            ):
                row = dict(scope.get(component_key) or {})
                lines.append(
                    f"| {scope_label} | {component_label} | {int(row.get('count', 0) or 0):,} "
                    f"| {fmt(row.get('rmse'))} | {fmt(row.get('mae'))} | {fmt(row.get('bias'))} "
                    f"| {fmt(row.get('global_spearman'))} | {fmt(row.get('mean_daily_spearman'))} |"
                )
    if conditional_mfe_safety:
        lines.extend([
            "- Conditional MFE-Safety objective：單一shared encoder；Primary head學Pure-MFE，"
            "Conditional head學在相同true-MFE percentile下異常低adverse的residual rank；"
            "conditional context使用stop-gradient primary prediction，無future feature、無loss-weight sweep。",
            "",
            section("Conditional MFE-Safety Model Gate"),
            "",
            "| Scope | Head | Groups | Daily rho | Global rho | Pair concordance | Top 10% Target | Bottom 10% Target |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ])
        conditional_eval = dict(payload.get("conditional_mfe_safety_evaluation") or {})
        for scope_label, scope_key in (
            ("Validation", "validation"),
            ("Forward OOS", "oos"),
            ("Breakout candidate slice", "breakout_candidate_oos"),
        ):
            scope = dict(conditional_eval.get(scope_key) or {})
            for head_label, head_key in (("Primary MFE", "primary_mfe"), ("Conditional Safety", "conditional_safety")):
                row = dict(scope.get(head_key) or {})
                pair = row.get("pairwise_concordance")
                lines.append(
                    f"| {scope_label} | {head_label} | {int(row.get('group_count', 0) or 0):,} "
                    f"| {fmt(row.get('mean_daily_spearman'))} | {fmt(row.get('global_spearman_vs_raw_target'))} "
                    f"| {'-' if pair is None else f'{float(pair)*100:.2f}%'} "
                    f"| {fmt(row.get('top_score_decile_raw_target_mean'))} "
                    f"| {fmt(row.get('bottom_score_decile_raw_target_mean'))} |"
                )
    if conditional_mfe_single or safety_conditional_mfe_duo:
        lines.extend([
            "- Reverse-Conditional MFE objective：J=U-E(U|S)，U=Pure-MFE percentile、S=Low-Adverse Safety percentile；"
            + (
                "Duo-head另以Raw Safety head的stop-gradient prediction作顯式condition；策略最終只使用Conditional-MFE score。"
                if safety_conditional_mfe_duo
                else "Single-head直接由歷史features學同一J，不建立顯式Safety head。"
            ),
            "",
            section("Reverse-Conditional MFE Model Gate"),
            "",
            "| Scope | Head | Groups | Daily rho | Global rho | Pair concordance | Top 10% Target | Bottom 10% Target |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ])
        reverse_eval = dict(payload.get("reverse_conditional_mfe_evaluation") or {})
        for scope_label, scope_key in (("Validation", "validation"), ("Forward OOS", "oos"), ("Breakout candidate slice", "breakout_candidate_oos")):
            scope = dict(reverse_eval.get(scope_key) or {})
            heads = (("Raw Safety", "raw_safety"), ("Conditional MFE", "conditional_mfe")) if safety_conditional_mfe_duo else (("Conditional MFE", "conditional_mfe"),)
            for head_label, head_key in heads:
                row = dict(scope.get(head_key) or {})
                pair = row.get("pairwise_concordance")
                lines.append(
                    f"| {scope_label} | {head_label} | {int(row.get('group_count', 0) or 0):,} "
                    f"| {fmt(row.get('mean_daily_spearman'))} | {fmt(row.get('global_spearman_vs_raw_target'))} "
                    f"| {'-' if pair is None else f'{float(pair)*100:.2f}%'} "
                    f"| {fmt(row.get('top_score_decile_raw_target_mean'))} | {fmt(row.get('bottom_score_decile_raw_target_mean'))} |"
                )
    if safety_raw_mfe_duo or safety_raw_mfe_hmhs_tri:
        lines.extend([
            (
                "- Safety→Raw-MFE→Direct-HM/HS objective：MR-13S raw 300×10與兩個marginal heads全部保留；"
                "新增shared-latent Direct HM/HS head，target=1[Safety percentile>=0.5 and Pure-MFE percentile>=0.5]；"
                "三head固定等權full-list Delta-NDCG，epoch selection仍依Raw-MFE Validation Dailyρ。"
                if safety_raw_mfe_hmhs_tri
                else "- Safety→Raw-MFE objective：與MR-13R共用相同shared encoder／Raw Safety auxiliary head／"
                "stop-gradient Safety context；唯一scientific變更是final head由J改學absolute Pure-MFE percentile U；"
                "兩head固定等權full-list Delta-NDCG，無lambda／threshold／calibration。"
            ),
            "",
            section("3. Multi-head Learnability / Truth / Prediction Geometry"),
            "",
            "| Scope | Head | Groups | Daily rho | Global rho | Pair concordance | Top 10% Target | Bottom 10% Target |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ])
        raw_eval = dict(payload.get("safety_raw_mfe_hmhs_evaluation") or payload.get("safety_raw_mfe_evaluation") or {})
        for scope_label, scope_key in (("Validation", "validation"), ("Forward OOS", "oos"), ("Breakout candidate slice", "breakout_candidate_oos")):
            scope = dict(raw_eval.get(scope_key) or {})
            for head_label, head_key in (("Raw Safety", "raw_safety"), ("Raw MFE", "raw_mfe")):
                row = dict(scope.get(head_key) or {})
                pair = row.get("pairwise_concordance")
                lines.append(
                    f"| {scope_label} | {head_label} | {int(row.get('group_count', 0) or 0):,} "
                    f"| {fmt(row.get('mean_daily_spearman'))} | {fmt(row.get('global_spearman_vs_raw_target'))} "
                    f"| {'-' if pair is None else f'{float(pair)*100:.2f}%'} "
                    f"| {fmt(row.get('top_score_decile_raw_target_mean'))} | {fmt(row.get('bottom_score_decile_raw_target_mean'))} |"
                )
            joint = dict(scope.get("joint_hmhs") or {})
            product = dict(scope.get("joint_product_control") or {})
            if joint:
                def pct(value):
                    return "-" if value is None else f"{float(value)*100:.2f}%"
                def pct100(value):
                    return "-" if value is None else f"{float(value):.2f}%"
                top10 = dict(joint.get("top_10pct") or {})
                top20 = dict(joint.get("top_20pct") or {})
                prod10 = dict(product.get("top_10pct") or {})
                lines.extend([
                    "",
                    f"### Model-specific Extension｜{payload['model_research_id']}｜Direct HM/HS Joint Retrieval｜{scope_label}",
                    "",
                    "| Metric | Direct Joint Head | Marginal Product Control |",
                    "|---|---:|---:|",
                    f"| Pair concordance | {pct(joint.get('pairwise_concordance'))} | {pct(product.get('pairwise_concordance'))} |",
                    f"| Global PR-AUC | {fmt(joint.get('global_average_precision'))} | {fmt(product.get('global_average_precision'))} |",
                    f"| Mean Daily PR-AUC | {fmt(joint.get('mean_daily_average_precision'))} | {fmt(product.get('mean_daily_average_precision'))} |",
                    f"| Top 10% HM/HS | {pct100(top10.get('hmhs_pct'))} / {fmt(top10.get('hmhs_enrichment'), 2)}× | {pct100(prod10.get('hmhs_pct'))} / {fmt(prod10.get('hmhs_enrichment'), 2)}× |",
                    f"| Top 20% HM/HS | {pct100(top20.get('hmhs_pct'))} / {fmt(top20.get('hmhs_enrichment'), 2)}× | - |",
                    f"| Population HM/HS | {pct100(joint.get('population_hmhs_pct'))} | same truth |",
                ])
        for scope_label, scope_key in (("Daily universal OOS", "oos"), ("Breakout candidate OOS", "breakout_candidate_oos")):
            gate = dict((raw_eval.get(scope_key) or {}).get("model_gate") or {})
            if not gate:
                continue
            upper = dict(gate.get("upper_right_s5_m5") or {})
            actual = dict(gate.get("actual_truth_geometry") or {})

            def truth_cell(cell):
                cell = dict(cell or {})
                pct = cell.get("population_pct")
                enrich = cell.get("independence_enrichment")
                return (
                    f"{int(cell.get('n', 0) or 0):,} / "
                    f"{'-' if pct is None else f'{float(pct):.2f}%'} / "
                    f"{'-' if enrich is None else f'{float(enrich):.2f}×'}"
                )

            lines.extend([
                "",
                section(f"{scope_label} Truth / Prediction Geometry", level=3),
                "",
                f"- Actual Safety↔MFE Dailyρ：`{fmt(actual.get('safety_to_mfe_mean_daily_spearman'), 4)}`",
                f"- Pred Safety↔Raw-MFE Dailyρ：`{fmt(gate.get('predicted_safety_to_raw_mfe_mean_daily_spearman'), 4)}`",
                f"- Actual S5×M5：`{truth_cell(actual.get('s5_m5'))}`（N / population / independence enrichment）",
                f"- Actual S4+×M4+：`{truth_cell(actual.get('s4plus_m4plus'))}`",
                f"- Predicted S5×M5：`N={int(upper.get('n', 0) or 0):,}`",
                f"- Joint product→actual HM/HS Dailyρ：`{fmt(gate.get('joint_product_to_actual_hmhs_mean_daily_spearman'), 4)}`",
                "- Geometry只作frozen model evaluation；不得用OOS cell結果fit weight／threshold／calibration。",
                "",
                "| Actual Safety \\ Pure-MFE | M1 | M2 | M3 | M4 | M5 |",
                "|---|---|---|---|---|---|",
            ])
            for s_idx, row in enumerate(list(actual.get("actual_joint_geometry") or []), start=1):
                lines.append(
                    f"| S{s_idx} | "
                    + " | ".join(truth_cell(cell) for cell in row)
                    + " |"
                )
            lines.extend([
                "",
                "| Pred Safety \\ Raw-MFE | M1 | M2 | M3 | M4 | M5 |",
                "|---|---|---|---|---|---|",
            ])
            for s_idx, row in enumerate(list(gate.get("predicted_joint_geometry") or []), start=1):
                cells = []
                for cell in row:
                    cell = dict(cell or {})
                    pct = cell.get("actual_hmhs_pct")
                    cells.append(
                        f"{int(cell.get('n', 0) or 0):,} / "
                        f"{'-' if pct is None else f'{float(pct):.2f}%'}"
                    )
                lines.append(f"| S{s_idx} | " + " | ".join(cells) + " |")
            lines.extend([
                "",
                "| Pred Safety quintile | N | Raw-MFE→actual MFE Dailyρ | High-MFE | HM/HS |",
                "|---|---:|---:|---:|---:|",
            ])
            for cohort in list(gate.get("safety_cohorts") or []):
                cohort = dict(cohort or {})
                lines.append(
                    f"| S{int(cohort.get('predicted_safety_quintile', 0) or 0)} "
                    f"| {int(cohort.get('n', 0) or 0):,} "
                    f"| {fmt(cohort.get('raw_mfe_to_actual_mfe_mean_daily_spearman'), 3)} "
                    f"| {'-' if cohort.get('high_mfe_pct') is None else f"{float(cohort['high_mfe_pct']):.2f}%"} "
                    f"| {'-' if cohort.get('hmhs_pct') is None else f"{float(cohort['hmhs_pct']):.2f}%"} |"
                )

    pareto_eval = dict(payload.get("pareto_pair_evaluation") or {})
    if pareto_eval:
        lines.extend([
            "",
            section("Pareto Supervision Diagnostics"),
            "",
            "| Scope | Groups | Comparable pairs | Comparable rate | Daily Pareto | Global Pareto |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for label, key in (
            ("Validation", "validation"),
            ("Forward OOS", "oos"),
            ("Breakout candidate slice", "breakout_candidate_oos"),
        ):
            row = dict(pareto_eval.get(key) or {})
            rate = row.get("comparable_pair_rate")
            lines.append(
                f"| {label} | {int(row.get('group_count', 0) or 0):,} "
                f"| {int(row.get('comparable_pair_count', 0) or 0):,} "
                f"| {'-' if rate is None else f'{float(rate)*100:.2f}%'} "
                f"| {fmt(row.get('mean_daily_pareto_pair_concordance'))} "
                f"| {fmt(row.get('global_pareto_pair_concordance'))} |"
            )
    reference_eval = dict(payload.get("reference_target_evaluation") or {})
    if reference_eval.get("available"):
        lines.extend([
            "",
            "## Common Target Reference",
            "",
            f"- Reference profile：`{reference_eval.get('reference_profile')}`",
            f"- Reference target：`{reference_eval.get('reference_target_id')}`",
            "- 此reference target只在checkpoint寫入與forward inference後計算；不參與loss、gradient、epoch selection或final refit。",
            "",
            "| Scope | Groups | Daily rho | Global rho | Pair concordance | Top 10% Target | Bottom 10% Target |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for label, key in (("All eligible stock-days", "oos"), ("Breakout candidate slice", "breakout_candidate_oos")):
            row = dict((reference_eval.get("split_metrics") or {}).get(key) or {})
            pair = row.get("pairwise_concordance")
            lines.append(
                f"| {label} | {int(row.get('group_count', 0) or 0):,} | {fmt(row.get('mean_daily_spearman'))} "
                f"| {fmt(row.get('global_spearman_vs_raw_target'))} | {'-' if pair is None else f'{float(pair)*100:.2f}%'} "
                f"| {fmt(row.get('top_score_decile_raw_target_mean'))} | {fmt(row.get('bottom_score_decile_raw_target_mean'))} |"
            )
    lines.extend([
        "",
        "## Boundary",
        "",
        f"- Research identity：`{payload['model_research_id']}`；trainer由profile metadata驅動，不綁定單一MR版本。",
        f"- Pairwise reduction：`{payload['training'].get('pairwise_reduction') or '-'}`；其餘learning semantics由profile固定。",
        "- Breakout candidate slice 只作 checkpoint 後診斷；candidate membership 不進模型輸入，也不進 training sample selection。",
        "- 本模型目前 research-only，不提供 strategy runtime source，不建立 PIT scores，不執行 ROOS。",
    ])
    return "\n".join(lines) + "\n"


def run(args) -> int:
    started = time.perf_counter()
    research_spec = get_continuous_ranker_research_spec(str(args.experiment_profile))
    execution_recipe = get_continuous_ranker_execution_recipe(str(args.experiment_profile))
    if execution_recipe.trainer_family != CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL:
        raise ValueError(
            "daily ranker orchestrator只接受daily_universal research spec: "
            f"profile={args.experiment_profile}, family={execution_recipe.trainer_family}"
        )
    build_started = time.perf_counter()
    progress_state = {"last_bucket": -1}

    def _daily_build_progress(processed, total, target_valid, skipped):
        total = max(int(total), 1)
        processed = int(processed)
        bucket = min(20, int(processed * 20 / total))
        if processed not in {0, total} and bucket <= int(progress_state["last_bucket"]):
            return
        progress_state["last_bucket"] = bucket
        pct = min(100.0, 100.0 * processed / total)
        elapsed = time.perf_counter() - build_started
        print(
            "[Daily target/index] "
            f"{processed}/{total} tickers ({pct:5.1f}%)｜"
            f"target_valid={int(target_valid):,}｜skipped={int(skipped)}｜"
            f"elapsed={elapsed:,.1f}s",
            flush=True,
        )

    print("[Daily target/index] 開始建立daily stock-day index與40D target...", flush=True)
    bundle = load_daily_universal_ranker_data(
        filter_id=str(args.filter_id),
        model_architecture=str(args.model_architecture),
        experiment_profile=str(args.experiment_profile),
        preload_feature_bank=bool(args.preload_feature_bank),
        allow_stale_source=bool(args.allow_stale_source),
        progress_callback=_daily_build_progress,
    )
    target_id = str(bundle.profile.continuous_target_id or "").strip()
    if not target_id:
        raise ValueError("daily-universal continuous ranker缺少target identity")
    raw_r_loss_name = (
        str(bundle.profile.loss_name)
        if bundle.profile.training_objective == TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION
        else None
    )
    raw_r_huber_delta = (
        float(bundle.profile.raw_r_huber_delta_r)
        if bundle.profile.training_objective == TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION
        and bundle.profile.raw_r_huber_delta_r is not None
        else None
    )
    conditional_mfe_safety = (
        bundle.profile.training_objective
        == TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_SAFETY_PAIRWISE_RANKING
    )
    conditional_mfe_single = (
        bundle.profile.training_objective == TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_PAIRWISE_RANKING
    )
    safety_conditional_mfe_duo = (
        bundle.profile.training_objective == TRAINING_OBJECTIVE_DAILY_SAFETY_CONDITIONAL_MFE_PAIRWISE_RANKING
    )
    safety_raw_mfe_duo = (
        bundle.profile.training_objective == TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_PAIRWISE_RANKING
    )
    safety_raw_mfe_hmhs_tri = (
        bundle.profile.training_objective == TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_HMHS_PAIRWISE_RANKING
    )
    safety_raw_mfe_joint_min_tri = (
        bundle.profile.training_objective == TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_JOINT_MIN_PAIRWISE_RANKING
    )
    direct_hmhs_only = (
        bundle.profile.training_objective == TRAINING_OBJECTIVE_DAILY_HMHS_PAIRWISE_RANKING
    )
    split = build_daily_ranker_split(bundle, inner_validation_months=int(args.inner_validation_months))
    forward_score_ids = resolve_forward_oos_score_group_ids(bundle)

    percentile_target = np.full(bundle.raw_target.shape, np.nan, dtype=np.float32)
    selection_mask = np.zeros(bundle.raw_target.shape, dtype=bool)
    selection_mask[split.selection_ids] = True
    selection_percentiles = ranker_api.build_daily_percentile_targets(
        bundle.raw_target, selection_mask, bundle.group_table["date"]
    )
    percentile_target[split.selection_ids] = selection_percentiles[split.selection_ids]

    torch, plan = resolve_ranker_execution_plan(args)
    print(
        f"torch=device={plan.device_type}, mixed_precision={plan.mixed_precision_enabled}, "
        f"dtype={plan.autocast_dtype_name}, deterministic={plan.deterministic_algorithms}, tf32={plan.allow_tf32}"
    )
    print(
        "Daily Universal Ranker｜"
        f"score_eligible={len(bundle.group_table):,} "
        f"target_valid={int(bundle.summary.get('target_valid_sample_count', 0) or 0):,} "
        f"tickers={int(bundle.summary['ticker_count']):,} "
        f"feature_storage={bundle.summary['feature_storage']}"
    )
    if bundle.summary.get("risk_param_coverage_start"):
        print(
            "Risk-normalized target｜"
            f"risk_param_coverage_start={bundle.summary['risk_param_coverage_start']} "
            f"context={','.join(bundle.summary.get('context_features') or []) or '-'}"
        )

    epoch_selection = ranker_api.select_epoch(
        torch,
        bundle.feature_bank,
        bundle.group_context,
        bundle.group_table,
        bundle.raw_target,
        percentile_target,
        split.inner_train_ids,
        split.validation_ids,
        args=args,
        plan=plan,
        evaluate_train_metrics=False,
    )
    selected_epoch = int(epoch_selection["best_epoch"])
    model, final_history = ranker_api.fit_final(
        torch,
        bundle.feature_bank,
        bundle.group_context,
        bundle.raw_target,
        percentile_target,
        bundle.group_table,
        split.selection_ids,
        epochs=selected_epoch,
        args=args,
        plan=plan,
        phase_label="Daily Selection完整重訓",
    )

    artifact_paths, output_dir = ranker_api.resolve_training_output_paths(args)
    artifact_paths.model_dir.mkdir(parents=True, exist_ok=True)
    trainable_parameter_count = count_trainable_parameters(model)
    total_parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())
    torch.save(
        {
            "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "feature_count": int(bundle.feature_bank.shape[2]),
            "context_count": int(bundle.group_context.shape[1]),
            "sequence_length": int(bundle.feature_bank.shape[1]),
            "model_spec": bundle.model_spec.as_manifest_payload(),
            "experiment_profile": str(args.experiment_profile),
            "experiment_settings": bundle.profile.as_manifest_payload(),
            "training_objective": bundle.profile.training_objective,
            "training_sample_scope": bundle.profile.training_sample_scope,
            "continuous_target_contract": bundle.target_manifest.get("target_contract"),
            "selected_epoch": selected_epoch,
            "torch_execution": plan.as_manifest_payload(),
            "trainable_parameter_count": int(trainable_parameter_count),
            "total_parameter_count": int(total_parameter_count),
        },
        artifact_paths.model_path,
    )

    # OOS target ranks and inference are intentionally deferred until after checkpoint write.
    oos_mask = np.zeros(bundle.raw_target.shape, dtype=bool)
    oos_mask[split.oos_ids] = True
    oos_percentiles = ranker_api.build_daily_percentile_targets(
        bundle.raw_target, oos_mask, bundle.group_table["date"]
    )
    percentile_target[split.oos_ids] = oos_percentiles[split.oos_ids]

    conditional_targets = (
        ranker_api.build_conditional_targets(bundle.group_table, percentile_target)
        if conditional_mfe_safety
        else None
    )
    conditional_mfe_safety_evaluation = {}
    conditional_validation_heads = None
    conditional_forward_heads = None
    reverse_conditional_mfe_targets = (
        ranker_api.build_conditional_mfe_opportunity_targets_for_training(
            bundle.group_table, percentile_target
        )
        if conditional_mfe_single or safety_conditional_mfe_duo or safety_raw_mfe_duo or safety_raw_mfe_hmhs_tri or safety_raw_mfe_joint_min_tri or direct_hmhs_only
        else None
    )
    reverse_conditional_mfe_evaluation = {}
    reverse_validation_heads = None
    reverse_forward_heads = None
    safety_raw_mfe_evaluation = {}
    safety_raw_mfe_hmhs_evaluation = {}
    safety_raw_mfe_joint_min_evaluation = {}
    raw_mfe_validation_heads = None
    raw_mfe_forward_heads = None

    dual_component_evaluation = {}
    validation_components = None
    forward_components = None
    if safety_raw_mfe_joint_min_tri:
        raw_mfe_validation_heads = ranker_api.predict_safety_raw_mfe_joint_min_scores(
            torch, model, bundle.feature_bank, bundle.group_context, split.validation_ids,
            batch_size=int(args.evaluation_batch_size), plan=plan,
        )
        raw_mfe_forward_heads = ranker_api.predict_safety_raw_mfe_joint_min_scores(
            torch, model, bundle.feature_bank, bundle.group_context, forward_score_ids,
            batch_size=int(args.evaluation_batch_size), plan=plan,
        )
        validation_scores = raw_mfe_validation_heads["raw_mfe"]
        forward_scores = raw_mfe_forward_heads["raw_mfe"]
    elif safety_raw_mfe_hmhs_tri:
        raw_mfe_validation_heads = ranker_api.predict_safety_raw_mfe_hmhs_scores(
            torch, model, bundle.feature_bank, bundle.group_context, split.validation_ids,
            batch_size=int(args.evaluation_batch_size), plan=plan,
        )
        raw_mfe_forward_heads = ranker_api.predict_safety_raw_mfe_hmhs_scores(
            torch, model, bundle.feature_bank, bundle.group_context, forward_score_ids,
            batch_size=int(args.evaluation_batch_size), plan=plan,
        )
        validation_scores = raw_mfe_validation_heads["raw_mfe"]
        forward_scores = raw_mfe_forward_heads["raw_mfe"]
    elif safety_raw_mfe_duo:
        raw_mfe_validation_heads = ranker_api.predict_safety_raw_mfe_scores(
            torch, model, bundle.feature_bank, bundle.group_context, split.validation_ids,
            batch_size=int(args.evaluation_batch_size), plan=plan,
        )
        raw_mfe_forward_heads = ranker_api.predict_safety_raw_mfe_scores(
            torch, model, bundle.feature_bank, bundle.group_context, forward_score_ids,
            batch_size=int(args.evaluation_batch_size), plan=plan,
        )
        validation_scores = raw_mfe_validation_heads["raw_mfe"]
        forward_scores = raw_mfe_forward_heads["raw_mfe"]
    elif safety_conditional_mfe_duo:
        reverse_validation_heads = ranker_api.predict_safety_conditional_mfe_scores(
            torch, model, bundle.feature_bank, bundle.group_context, split.validation_ids,
            batch_size=int(args.evaluation_batch_size), plan=plan,
        )
        reverse_forward_heads = ranker_api.predict_safety_conditional_mfe_scores(
            torch, model, bundle.feature_bank, bundle.group_context, forward_score_ids,
            batch_size=int(args.evaluation_batch_size), plan=plan,
        )
        validation_scores = reverse_validation_heads["conditional_mfe"]
        forward_scores = reverse_forward_heads["conditional_mfe"]
    elif conditional_mfe_safety:
        conditional_validation_heads = ranker_api.predict_conditional_mfe_safety_scores(
            torch, model, bundle.feature_bank, bundle.group_context, split.validation_ids,
            batch_size=int(args.evaluation_batch_size), plan=plan,
        )
        conditional_forward_heads = ranker_api.predict_conditional_mfe_safety_scores(
            torch, model, bundle.feature_bank, bundle.group_context, forward_score_ids,
            batch_size=int(args.evaluation_batch_size), plan=plan,
        )
        validation_scores = conditional_validation_heads["primary_mfe"]
        forward_scores = conditional_forward_heads["primary_mfe"]
    elif bundle.profile.training_objective == TRAINING_OBJECTIVE_DAILY_DUAL_COMPONENT_R_REGRESSION:
        validation_components = ranker_api.predict_dual_component_r(
            torch, model, bundle.feature_bank, bundle.group_context, split.validation_ids,
            batch_size=int(args.evaluation_batch_size), plan=plan,
        )
        forward_components = ranker_api.predict_dual_component_r(
            torch, model, bundle.feature_bank, bundle.group_context, forward_score_ids,
            batch_size=int(args.evaluation_batch_size), plan=plan,
        )
        validation_scores = validation_components["model_score"]
        forward_scores = forward_components["model_score"]
        dual_component_evaluation["validation"] = _dual_component_metrics(
            bundle.group_table,
            split.validation_ids,
            validation_components["predicted_favorable_r"],
            validation_components["predicted_adverse_r"],
        )
        oos_positions = {int(group_id): pos for pos, group_id in enumerate(forward_score_ids)}
        oos_component_positions = np.asarray(
            [oos_positions[int(group_id)] for group_id in split.oos_ids], dtype=np.int64
        )
        dual_component_evaluation["oos"] = _dual_component_metrics(
            bundle.group_table,
            split.oos_ids,
            forward_components["predicted_favorable_r"][oos_component_positions],
            forward_components["predicted_adverse_r"][oos_component_positions],
        )
    else:
        validation_scores = ranker_api.predict_scores(
            torch, model, bundle.feature_bank, bundle.group_context, split.validation_ids,
            batch_size=int(args.evaluation_batch_size), plan=plan,
            training_objective=bundle.profile.training_objective,
        )
        forward_scores = ranker_api.predict_scores(
            torch, model, bundle.feature_bank, bundle.group_context, forward_score_ids,
            batch_size=int(args.evaluation_batch_size), plan=plan,
            training_objective=bundle.profile.training_objective,
        )
    score_by_group = np.full(len(bundle.group_table), np.nan, dtype=np.float32)
    score_by_group[forward_score_ids] = forward_scores
    oos_scores = score_by_group[split.oos_ids]
    if direct_hmhs_only:
        assert reverse_conditional_mfe_targets is not None
        validation_metrics = ranker_api.direct_hmhs_metrics(
            split.validation_ids, bundle.group_table, reverse_conditional_mfe_targets,
            validation_scores, include_top_k_quality=True,
        )
        oos_metrics = ranker_api.direct_hmhs_metrics(
            split.oos_ids, bundle.group_table, reverse_conditional_mfe_targets,
            oos_scores, include_top_k_quality=True,
        )
    else:
        validation_metrics = ranker_api.split_metrics(
            split.validation_ids, bundle.group_table, bundle.raw_target, percentile_target,
            validation_scores, include_top_k_quality=True,
            raw_r_regression_loss_name=raw_r_loss_name,
            raw_r_huber_delta_r=raw_r_huber_delta,
        )
        oos_metrics = ranker_api.split_metrics(
            split.oos_ids, bundle.group_table, bundle.raw_target, percentile_target,
            oos_scores, include_top_k_quality=True,
            raw_r_regression_loss_name=raw_r_loss_name,
            raw_r_huber_delta_r=raw_r_huber_delta,
        )
    candidate_ids = select_breakout_candidate_group_ids(
        bundle, split.oos_ids, allow_stale_source=bool(args.allow_stale_source)
    )
    if forward_components is not None:
        forward_position_by_group = {
            int(group_id): pos for pos, group_id in enumerate(forward_score_ids)
        }
        candidate_positions = np.asarray(
            [forward_position_by_group[int(group_id)] for group_id in candidate_ids],
            dtype=np.int64,
        )
        dual_component_evaluation["breakout_candidate_oos"] = _dual_component_metrics(
            bundle.group_table,
            candidate_ids,
            forward_components["predicted_favorable_r"][candidate_positions],
            forward_components["predicted_adverse_r"][candidate_positions],
        )
    if safety_raw_mfe_joint_min_tri:
        assert reverse_conditional_mfe_targets is not None
        assert raw_mfe_validation_heads is not None and raw_mfe_forward_heads is not None
        forward_position_by_group = {int(group_id): pos for pos, group_id in enumerate(forward_score_ids)}
        oos_positions = np.asarray([forward_position_by_group[int(group_id)] for group_id in split.oos_ids], dtype=np.int64)
        safety_raw_mfe_joint_min_evaluation["validation"] = ranker_api.safety_raw_mfe_joint_min_metrics(
            split.validation_ids, bundle.group_table, reverse_conditional_mfe_targets,
            raw_mfe_validation_heads, include_top_k_quality=True,
        )
        oos_heads = {key: values[oos_positions] for key, values in raw_mfe_forward_heads.items()}
        safety_raw_mfe_joint_min_evaluation["oos"] = ranker_api.safety_raw_mfe_joint_min_metrics(
            split.oos_ids, bundle.group_table, reverse_conditional_mfe_targets,
            oos_heads, include_top_k_quality=True,
        )
        if len(candidate_ids) >= 2:
            candidate_positions = np.asarray([forward_position_by_group[int(group_id)] for group_id in candidate_ids], dtype=np.int64)
            candidate_heads = {key: values[candidate_positions] for key, values in raw_mfe_forward_heads.items()}
            safety_raw_mfe_joint_min_evaluation["breakout_candidate_oos"] = ranker_api.safety_raw_mfe_joint_min_metrics(
                candidate_ids, bundle.group_table, reverse_conditional_mfe_targets,
                candidate_heads, include_top_k_quality=True,
            )
        else:
            safety_raw_mfe_joint_min_evaluation["breakout_candidate_oos"] = {
                "raw_safety": _empty_split_metrics(len(candidate_ids), "breakout candidate OOS slice有效sample不足"),
                "raw_mfe": _empty_split_metrics(len(candidate_ids), "breakout candidate OOS slice有效sample不足"),
                "joint_min": {"group_count": int(len(candidate_ids)), "not_evaluated_reason": "breakout candidate OOS slice有效sample不足"},
                "model_gate": {"status": "not_evaluated_insufficient_sample"},
            }
    elif safety_raw_mfe_hmhs_tri:
        assert reverse_conditional_mfe_targets is not None
        assert raw_mfe_validation_heads is not None and raw_mfe_forward_heads is not None
        forward_position_by_group = {int(group_id): pos for pos, group_id in enumerate(forward_score_ids)}
        oos_positions = np.asarray([forward_position_by_group[int(group_id)] for group_id in split.oos_ids], dtype=np.int64)
        safety_raw_mfe_hmhs_evaluation["validation"] = ranker_api.safety_raw_mfe_hmhs_metrics(
            split.validation_ids, bundle.group_table, reverse_conditional_mfe_targets,
            raw_mfe_validation_heads, include_top_k_quality=True,
        )
        oos_heads = {key: values[oos_positions] for key, values in raw_mfe_forward_heads.items()}
        safety_raw_mfe_hmhs_evaluation["oos"] = ranker_api.safety_raw_mfe_hmhs_metrics(
            split.oos_ids, bundle.group_table, reverse_conditional_mfe_targets,
            oos_heads, include_top_k_quality=True,
        )
        if len(candidate_ids) >= 2:
            candidate_positions = np.asarray([forward_position_by_group[int(group_id)] for group_id in candidate_ids], dtype=np.int64)
            candidate_heads = {key: values[candidate_positions] for key, values in raw_mfe_forward_heads.items()}
            safety_raw_mfe_hmhs_evaluation["breakout_candidate_oos"] = ranker_api.safety_raw_mfe_hmhs_metrics(
                candidate_ids, bundle.group_table, reverse_conditional_mfe_targets,
                candidate_heads, include_top_k_quality=True,
            )
        else:
            safety_raw_mfe_hmhs_evaluation["breakout_candidate_oos"] = {
                "raw_safety": _empty_split_metrics(len(candidate_ids), "breakout candidate OOS slice有效sample不足"),
                "raw_mfe": _empty_split_metrics(len(candidate_ids), "breakout candidate OOS slice有效sample不足"),
                "joint_hmhs": {"group_count": int(len(candidate_ids)), "not_evaluated_reason": "breakout candidate OOS slice有效sample不足"},
                "joint_product_control": {},
                "model_gate": {"status": "not_evaluated_insufficient_sample"},
            }
    elif safety_raw_mfe_duo:
        assert reverse_conditional_mfe_targets is not None
        assert raw_mfe_validation_heads is not None and raw_mfe_forward_heads is not None
        forward_position_by_group = {int(group_id): pos for pos, group_id in enumerate(forward_score_ids)}
        oos_positions = np.asarray([forward_position_by_group[int(group_id)] for group_id in split.oos_ids], dtype=np.int64)
        safety_raw_mfe_evaluation["validation"] = ranker_api.safety_raw_mfe_metrics(
            split.validation_ids, bundle.group_table, reverse_conditional_mfe_targets,
            raw_mfe_validation_heads, include_top_k_quality=True,
        )
        oos_heads = {key: values[oos_positions] for key, values in raw_mfe_forward_heads.items()}
        safety_raw_mfe_evaluation["oos"] = ranker_api.safety_raw_mfe_metrics(
            split.oos_ids, bundle.group_table, reverse_conditional_mfe_targets,
            oos_heads, include_top_k_quality=True,
        )
        if len(candidate_ids) >= 2:
            candidate_positions = np.asarray([forward_position_by_group[int(group_id)] for group_id in candidate_ids], dtype=np.int64)
            candidate_heads = {key: values[candidate_positions] for key, values in raw_mfe_forward_heads.items()}
            safety_raw_mfe_evaluation["breakout_candidate_oos"] = ranker_api.safety_raw_mfe_metrics(
                candidate_ids, bundle.group_table, reverse_conditional_mfe_targets,
                candidate_heads, include_top_k_quality=True,
            )
        else:
            safety_raw_mfe_evaluation["breakout_candidate_oos"] = {
                "raw_safety": _empty_split_metrics(len(candidate_ids), "breakout candidate OOS slice有效sample不足"),
                "raw_mfe": _empty_split_metrics(len(candidate_ids), "breakout candidate OOS slice有效sample不足"),
                "model_gate": {"status": "not_evaluated_insufficient_sample"},
            }

    if conditional_mfe_single or safety_conditional_mfe_duo:
        assert reverse_conditional_mfe_targets is not None
        forward_position_by_group = {int(group_id): pos for pos, group_id in enumerate(forward_score_ids)}
        oos_positions = np.asarray([forward_position_by_group[int(group_id)] for group_id in split.oos_ids], dtype=np.int64)
        if safety_conditional_mfe_duo:
            assert reverse_validation_heads is not None and reverse_forward_heads is not None
            reverse_conditional_mfe_evaluation["validation"] = ranker_api.safety_conditional_mfe_metrics(
                split.validation_ids, bundle.group_table, reverse_conditional_mfe_targets, reverse_validation_heads, include_top_k_quality=True
            )
            oos_heads = {key: values[oos_positions] for key, values in reverse_forward_heads.items()}
            reverse_conditional_mfe_evaluation["oos"] = ranker_api.safety_conditional_mfe_metrics(
                split.oos_ids, bundle.group_table, reverse_conditional_mfe_targets, oos_heads, include_top_k_quality=True
            )
            if len(candidate_ids) >= 2:
                candidate_positions = np.asarray([forward_position_by_group[int(group_id)] for group_id in candidate_ids], dtype=np.int64)
                candidate_heads = {key: values[candidate_positions] for key, values in reverse_forward_heads.items()}
                reverse_conditional_mfe_evaluation["breakout_candidate_oos"] = ranker_api.safety_conditional_mfe_metrics(
                    candidate_ids, bundle.group_table, reverse_conditional_mfe_targets, candidate_heads, include_top_k_quality=True
                )
            else:
                reverse_conditional_mfe_evaluation["breakout_candidate_oos"] = {
                    "raw_safety": _empty_split_metrics(len(candidate_ids), "breakout candidate OOS slice有效sample不足"),
                    "conditional_mfe": _empty_split_metrics(len(candidate_ids), "breakout candidate OOS slice有效sample不足"),
                }
        else:
            reverse_conditional_mfe_evaluation["validation"] = {
                "conditional_mfe": ranker_api.split_metrics(
                    split.validation_ids, bundle.group_table, reverse_conditional_mfe_targets.conditional_mfe_residual,
                    reverse_conditional_mfe_targets.conditional_mfe_percentile, validation_scores, include_top_k_quality=True
                )
            }
            reverse_conditional_mfe_evaluation["oos"] = {
                "conditional_mfe": ranker_api.split_metrics(
                    split.oos_ids, bundle.group_table, reverse_conditional_mfe_targets.conditional_mfe_residual,
                    reverse_conditional_mfe_targets.conditional_mfe_percentile, oos_scores, include_top_k_quality=True
                )
            }
            reverse_conditional_mfe_evaluation["breakout_candidate_oos"] = {
                "conditional_mfe": (
                    ranker_api.split_metrics(
                        candidate_ids, bundle.group_table, reverse_conditional_mfe_targets.conditional_mfe_residual,
                        reverse_conditional_mfe_targets.conditional_mfe_percentile, score_by_group[candidate_ids], include_top_k_quality=True
                    ) if len(candidate_ids) >= 2 else _empty_split_metrics(len(candidate_ids), "breakout candidate OOS slice有效sample不足")
                )
            }

    if conditional_mfe_safety:
        assert conditional_targets is not None
        assert conditional_validation_heads is not None
        assert conditional_forward_heads is not None
        conditional_mfe_safety_evaluation["validation"] = ranker_api.conditional_mfe_safety_metrics(
            split.validation_ids,
            bundle.group_table,
            bundle.raw_target,
            conditional_targets,
            conditional_validation_heads,
            include_top_k_quality=True,
        )
        forward_position_by_group = {
            int(group_id): pos for pos, group_id in enumerate(forward_score_ids)
        }
        oos_positions = np.asarray(
            [forward_position_by_group[int(group_id)] for group_id in split.oos_ids],
            dtype=np.int64,
        )
        oos_head_scores = {
            key: values[oos_positions]
            for key, values in conditional_forward_heads.items()
        }
        conditional_mfe_safety_evaluation["oos"] = ranker_api.conditional_mfe_safety_metrics(
            split.oos_ids,
            bundle.group_table,
            bundle.raw_target,
            conditional_targets,
            oos_head_scores,
            include_top_k_quality=True,
        )
        if len(candidate_ids) >= 2:
            candidate_positions = np.asarray(
                [forward_position_by_group[int(group_id)] for group_id in candidate_ids],
                dtype=np.int64,
            )
            candidate_head_scores = {
                key: values[candidate_positions]
                for key, values in conditional_forward_heads.items()
            }
            conditional_mfe_safety_evaluation["breakout_candidate_oos"] = (
                ranker_api.conditional_mfe_safety_metrics(
                    candidate_ids,
                    bundle.group_table,
                    bundle.raw_target,
                    conditional_targets,
                    candidate_head_scores,
                    include_top_k_quality=True,
                )
            )
        else:
            conditional_mfe_safety_evaluation["breakout_candidate_oos"] = {
                "primary_mfe": _empty_split_metrics(
                    len(candidate_ids), "breakout candidate OOS slice有效sample不足"
                ),
                "conditional_safety": _empty_split_metrics(
                    len(candidate_ids), "breakout candidate OOS slice有效sample不足"
                ),
            }

    candidate_metrics = (
        (
            ranker_api.direct_hmhs_metrics(
                candidate_ids, bundle.group_table, reverse_conditional_mfe_targets,
                score_by_group[candidate_ids], include_top_k_quality=True,
            )
            if direct_hmhs_only
            else ranker_api.split_metrics(
                candidate_ids,
                bundle.group_table,
                bundle.raw_target,
                percentile_target,
                score_by_group[candidate_ids],
                include_top_k_quality=True,
                raw_r_regression_loss_name=raw_r_loss_name,
                raw_r_huber_delta_r=raw_r_huber_delta,
            )
        )
        if len(candidate_ids) >= 2
        else _empty_split_metrics(
            len(candidate_ids), "breakout candidate OOS slice有效sample不足"
        )
    )

    predicted_safety_context_pure_mfe_evaluation = {}
    if target_id == PREDICTED_SAFETY_CONTEXT_PURE_MFE_TARGET_ID:
        predicted_safety_context_pure_mfe_evaluation = {
            "validation": _predicted_safety_context_pure_mfe_metrics(
                split.validation_ids, bundle.group_table, validation_scores
            ),
            "oos": _predicted_safety_context_pure_mfe_metrics(
                split.oos_ids, bundle.group_table, oos_scores
            ),
            "breakout_candidate_oos": (
                _predicted_safety_context_pure_mfe_metrics(
                    candidate_ids, bundle.group_table, score_by_group[candidate_ids]
                )
                if len(candidate_ids) >= 2
                else {"available": False, "group_count": int(len(candidate_ids))}
            ),
        }

    pareto_pair_evaluation = {}
    if bundle.profile.training_objective == TRAINING_OBJECTIVE_DAILY_PARETO_PAIRWISE_RANKING:
        pareto_pair_evaluation = {
            "validation": ranker_api.pareto_pair_concordance_metrics(
                split.validation_ids, bundle.group_table, bundle.raw_target, validation_scores
            ),
            "oos": ranker_api.pareto_pair_concordance_metrics(
                split.oos_ids, bundle.group_table, bundle.raw_target, oos_scores
            ),
            "breakout_candidate_oos": (
                ranker_api.pareto_pair_concordance_metrics(
                    candidate_ids,
                    bundle.group_table,
                    bundle.raw_target,
                    score_by_group[candidate_ids],
                )
                if len(candidate_ids) >= 2
                else {
                    "group_count": int(len(candidate_ids)),
                    "comparable_pair_count": 0,
                    "all_pair_count": 0,
                    "comparable_pair_rate": None,
                    "rankable_date_count": 0,
                    "mean_daily_pareto_pair_concordance": None,
                    "global_pareto_pair_concordance": None,
                }
            ),
        }

    reference_target_evaluation = {
        "available": False,
        "reason": "active research profile沒有設定reference target",
    }
    reference_raw_target = None
    evaluation_reference_profile = (
        research_spec.evaluation_reference_profile_name
        or research_spec.reference_profile_name
    )
    if evaluation_reference_profile:
        reference_bundle = load_daily_universal_ranker_data(
            filter_id=str(args.filter_id),
            model_architecture=str(args.model_architecture),
            experiment_profile=str(evaluation_reference_profile),
            preload_feature_bank=False,
            allow_stale_source=bool(args.allow_stale_source),
            project_root=PROJECT_ROOT,
        )
        key_columns = ["ticker", "date", "source_pos", "label_eval_end_date"]
        if len(reference_bundle.group_table) != len(bundle.group_table) or not (
            reference_bundle.group_table[key_columns].astype(str).equals(
                bundle.group_table[key_columns].astype(str)
            )
        ):
            raise ValueError("reference target與candidate daily stock-day universe／順序不一致")
        if not np.array_equal(reference_bundle.target_valid, bundle.target_valid):
            raise ValueError("reference target與candidate target-valid universe不一致")
        reference_raw_target = np.asarray(reference_bundle.raw_target, dtype=np.float32)
        reference_percentile = np.full(reference_raw_target.shape, np.nan, dtype=np.float32)
        reference_oos_percentile = ranker_api.build_daily_percentile_targets(
            reference_raw_target, oos_mask, bundle.group_table["date"]
        )
        reference_percentile[split.oos_ids] = reference_oos_percentile[split.oos_ids]
        reference_oos_metrics = ranker_api.split_metrics(
            split.oos_ids, bundle.group_table, reference_raw_target, reference_percentile,
            oos_scores, include_top_k_quality=True,
        )
        reference_candidate_metrics = (
            ranker_api.split_metrics(
                candidate_ids, bundle.group_table, reference_raw_target, reference_percentile,
                score_by_group[candidate_ids], include_top_k_quality=True,
            )
            if len(candidate_ids) >= 2
            else _empty_split_metrics(
                len(candidate_ids), "breakout candidate OOS slice有效sample不足"
            )
        )
        reference_target_evaluation = {
            "available": True,
            "reference_profile": str(evaluation_reference_profile),
            "reference_target_id": str(reference_bundle.profile.continuous_target_id),
            "used_for_training_or_epoch_selection": False,
            "evaluated_after_checkpoint_write": True,
            "split_metrics": {
                "oos": reference_oos_metrics,
                "breakout_candidate_oos": reference_candidate_metrics,
            },
        }
        del reference_bundle

    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / DAILY_RANKER_OOS_SCORE_FILENAME
    report_json_path = output_dir / ranker_api.RANKER_REPORT_JSON_FILENAME
    report_markdown_path = output_dir / ranker_api.RANKER_REPORT_MARKDOWN_FILENAME
    split_path = output_dir / DAILY_SPLIT_FILENAME

    oos_frame = bundle.group_table.iloc[forward_score_ids][["ticker", "date", "group_index"]].copy()
    evaluable_forward_mask = np.isin(forward_score_ids, split.oos_ids)
    oos_frame["target_raw_r"] = np.nan
    oos_frame["target_daily_percentile"] = np.nan
    oos_frame.loc[evaluable_forward_mask, "target_raw_r"] = bundle.raw_target[
        forward_score_ids[evaluable_forward_mask]
    ]
    oos_frame.loc[evaluable_forward_mask, "target_daily_percentile"] = percentile_target[
        forward_score_ids[evaluable_forward_mask]
    ]
    if conditional_targets is not None:
        oos_frame["target_conditional_safety_residual"] = np.nan
        oos_frame["target_conditional_safety_percentile"] = np.nan
        evaluable_ids = forward_score_ids[evaluable_forward_mask]
        oos_frame.loc[evaluable_forward_mask, "target_conditional_safety_residual"] = (
            conditional_targets.conditional_safety_residual[evaluable_ids]
        )
        oos_frame.loc[evaluable_forward_mask, "target_conditional_safety_percentile"] = (
            conditional_targets.conditional_safety_percentile[evaluable_ids]
        )
    if reverse_conditional_mfe_targets is not None:
        oos_frame["target_low_adverse_safety_percentile"] = np.nan
        evaluable_ids = forward_score_ids[evaluable_forward_mask]
        oos_frame.loc[evaluable_forward_mask, "target_low_adverse_safety_percentile"] = reverse_conditional_mfe_targets.low_adverse_safety_percentile[evaluable_ids]
        if safety_raw_mfe_duo or safety_raw_mfe_hmhs_tri or safety_raw_mfe_joint_min_tri or direct_hmhs_only:
            oos_frame["target_pure_mfe_percentile"] = np.nan
            oos_frame.loc[evaluable_forward_mask, "target_pure_mfe_percentile"] = reverse_conditional_mfe_targets.primary_mfe_percentile[evaluable_ids]
            if safety_raw_mfe_hmhs_tri or direct_hmhs_only:
                oos_frame["target_direct_hmhs"] = np.nan
                oos_frame.loc[evaluable_forward_mask, "target_direct_hmhs"] = (
                    reverse_conditional_mfe_targets.direct_hmhs_target[evaluable_ids]
                )
            if safety_raw_mfe_joint_min_tri:
                oos_frame["target_joint_min"] = np.nan
                oos_frame.loc[evaluable_forward_mask, "target_joint_min"] = (
                    reverse_conditional_mfe_targets.joint_min_target[evaluable_ids]
                )
        else:
            oos_frame["target_conditional_mfe_residual"] = np.nan
            oos_frame["target_conditional_mfe_percentile"] = np.nan
            oos_frame.loc[evaluable_forward_mask, "target_conditional_mfe_residual"] = reverse_conditional_mfe_targets.conditional_mfe_residual[evaluable_ids]
            oos_frame.loc[evaluable_forward_mask, "target_conditional_mfe_percentile"] = reverse_conditional_mfe_targets.conditional_mfe_percentile[evaluable_ids]
    if reference_raw_target is not None:
        oos_frame["reference_target_raw_r"] = np.nan
        oos_frame.loc[evaluable_forward_mask, "reference_target_raw_r"] = reference_raw_target[
            forward_score_ids[evaluable_forward_mask]
        ]
    oos_frame["model_score"] = forward_scores
    if raw_mfe_forward_heads is not None:
        oos_frame["raw_safety_score"] = raw_mfe_forward_heads["raw_safety"]
        oos_frame["raw_mfe_score"] = raw_mfe_forward_heads["raw_mfe"]
        if "joint_hmhs" in raw_mfe_forward_heads:
            oos_frame["joint_hmhs_score"] = raw_mfe_forward_heads["joint_hmhs"]
        if "joint_min" in raw_mfe_forward_heads:
            oos_frame["joint_min_score"] = raw_mfe_forward_heads["joint_min"]
    elif reverse_forward_heads is not None:
        oos_frame["raw_safety_score"] = reverse_forward_heads["raw_safety"]
        oos_frame["conditional_mfe_score"] = reverse_forward_heads["conditional_mfe"]
    elif conditional_mfe_single:
        oos_frame["conditional_mfe_score"] = forward_scores
    if conditional_forward_heads is not None:
        oos_frame["primary_mfe_score"] = conditional_forward_heads["primary_mfe"]
        oos_frame["conditional_safety_score"] = conditional_forward_heads["conditional_safety"]
    if forward_components is not None:
        oos_frame["predicted_favorable_r"] = forward_components["predicted_favorable_r"]
        oos_frame["predicted_adverse_r"] = forward_components["predicted_adverse_r"]
        oos_frame["predicted_composite_r"] = forward_components["model_score"]
    if bundle.profile.training_objective == TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION:
        oos_frame["predicted_r"] = forward_scores
    oos_frame.to_csv(score_path, index=False, encoding="utf-8-sig", compression="gzip")
    _date_split_frame(bundle, split, forward_score_ids=forward_score_ids).to_csv(
        split_path, index=False, encoding="utf-8-sig"
    )

    information_cutoff = str(pd.Timestamp(bundle.group_table.iloc[split.selection_ids]["label_eval_end_date"].max()).date())
    split_metrics = {
        "inner_train": _empty_split_metrics(len(split.inner_train_ids), "full-refit後不重跑整段inner train以避免重複daily window materialization"),
        "validation": validation_metrics,
        "selection": _empty_split_metrics(len(split.selection_ids), "full Selection metrics不另重跑；模型主Gate使用forward OOS"),
        "oos": oos_metrics,
        "breakout_candidate_oos": candidate_metrics,
    }
    payload = {
        "schema_version": ranker_api.RANKER_SCHEMA_VERSION + 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": research_spec.experiment_name,
        "phase": research_spec.phase,
        "model_research_id": research_spec.model_research_id,
        "score_semantic_id": execution_recipe.score_semantic_id,
        "status": "RESULT_AVAILABLE_PENDING_REVIEW",
        "filter_id": str(args.filter_id),
        "model_architecture": str(args.model_architecture),
        "experiment_profile": str(args.experiment_profile),
        "experiment_settings": bundle.profile.as_manifest_payload(),
        "training": {
            "objective": bundle.profile.training_objective,
            "loss": bundle.profile.loss_name,
            "sample_scope": bundle.profile.training_sample_scope,
            "training_label_scope": bundle.profile.training_label_scope,
            "target": target_id,
            "model_score": (
                "predicted_r_two_logit_margin"
                if bundle.profile.training_objective == TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION
                else "predicted_favorable_mfe_r_minus_predicted_adverse_to_peak_r"
                if bundle.profile.training_objective == TRAINING_OBJECTIVE_DAILY_DUAL_COMPONENT_R_REGRESSION
                else "primary_mfe_softmax_pass_probability"
                if conditional_mfe_safety
                else "raw_mfe_softmax_pass_probability"
                if safety_raw_mfe_duo or safety_raw_mfe_hmhs_tri or safety_raw_mfe_joint_min_tri
                else "conditional_mfe_softmax_pass_probability"
                if conditional_mfe_single or safety_conditional_mfe_duo
                else "softmax_pass_probability_monotonic_to_two_logit_margin"
            ),
            "batching": ranker_api.training_semantics(bundle.profile)["batching"],
            "pairwise_contract": ranker_api.training_semantics(bundle.profile)["pairwise_contract"],
            "raw_r_regression_contract": ranker_api.training_semantics(bundle.profile).get("raw_r_regression_contract"),
            "dual_component_r_regression_contract": ranker_api.training_semantics(bundle.profile).get("dual_component_r_regression_contract"),
            "conditional_mfe_safety_contract": ranker_api.training_semantics(bundle.profile).get("conditional_mfe_safety_contract"),
            "conditional_mfe_single_head_contract": ranker_api.training_semantics(bundle.profile).get("conditional_mfe_single_head_contract"),
            "safety_conditional_mfe_duo_head_contract": ranker_api.training_semantics(bundle.profile).get("safety_conditional_mfe_duo_head_contract"),
            "safety_raw_mfe_duo_head_contract": ranker_api.training_semantics(bundle.profile).get("safety_raw_mfe_duo_head_contract"),
            "safety_raw_mfe_hmhs_tri_head_contract": ranker_api.training_semantics(bundle.profile).get("safety_raw_mfe_hmhs_tri_head_contract"),
            "safety_raw_mfe_joint_min_tri_head_contract": ranker_api.training_semantics(bundle.profile).get("safety_raw_mfe_joint_min_tri_head_contract"),
            "direct_hmhs_single_head_contract": ranker_api.training_semantics(bundle.profile).get("direct_hmhs_single_head_contract"),
            "pairwise_reduction": execution_recipe.pairwise_reduction,
            "selected_epoch": selected_epoch,
            "epoch_selection_metric": bundle.profile.epoch_selection_metric,
            "epoch_selection": epoch_selection,
            "final_refit_history": final_history,
            "seed": int(args.seed),
        },
        "split_report": split.report,
        "split_metrics": split_metrics,
        "all_group_split_metrics": split_metrics,
        "reference_target_evaluation": reference_target_evaluation,
        "dual_component_evaluation": dual_component_evaluation,
        "conditional_mfe_safety_evaluation": conditional_mfe_safety_evaluation,
        "reverse_conditional_mfe_evaluation": reverse_conditional_mfe_evaluation,
        "safety_raw_mfe_evaluation": safety_raw_mfe_evaluation,
        "safety_raw_mfe_hmhs_evaluation": safety_raw_mfe_hmhs_evaluation,
        "safety_raw_mfe_joint_min_evaluation": safety_raw_mfe_joint_min_evaluation,
        "predicted_safety_context_pure_mfe_evaluation": predicted_safety_context_pure_mfe_evaluation,
        "pareto_pair_evaluation": pareto_pair_evaluation,
        "trade_alignment": {"available": False, "reason": "daily universal model先做模型本身驗證；未接策略trade attribution"},
        "target_manifest": bundle.target_manifest,
        "source_dataset": bundle.summary,
        "model_information_cutoff": information_cutoff,
        "oos_used_for_training_or_epoch_selection": False,
        "oos_evaluated_after_checkpoint_write": True,
        "score_eligibility_contract": build_score_eligibility_contract(bundle.profile),
        "forward_score_coverage": {
            "inference_eligible_groups": int(len(forward_score_ids)),
            "target_evaluable_groups": int(len(split.oos_ids)),
            "future_target_required_for_score": False,
        },
        "runtime_eligibility": {
            "eligible": False,
            "scope": "research_only",
            "reason": (
                f"{research_spec.model_research_id}先通過daily forward-OOS與breakout-candidate "
                "slice模型Gate後，才建立PIT／策略source"
            ),
        },
        "artifacts": {
            "model": build_file_manifest(artifact_paths.model_path),
            "oos_scores_gzip": build_file_manifest(score_path),
            "date_split": build_file_manifest(split_path),
        },
        "torch_execution": plan.as_manifest_payload(),
        "elapsed_sec": round(time.perf_counter() - started, 3),
    }
    write_json(report_json_path, payload)
    report_markdown_path.write_text(_render_markdown(payload), encoding="utf-8")

    manifest = {
        "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
        "research_schema_version": payload["schema_version"],
        "filter_family": FILTER_FAMILY,
        "filter_id": str(args.filter_id),
        "model_architecture": str(args.model_architecture),
        "model_spec": bundle.model_spec.as_manifest_payload(),
        "experiment_profile": str(args.experiment_profile),
        "experiment_settings": bundle.profile.as_manifest_payload(),
        "training_objective": bundle.profile.training_objective,
        "training_label_scope": bundle.profile.training_label_scope,
        "training_sample_scope": bundle.profile.training_sample_scope,
        "training_semantics": ranker_api.training_semantics(bundle.profile),
        "continuous_target_id": target_id,
        "sequence_length": int(bundle.feature_bank.shape[1]),
        "feature_columns": list(FEATURE_COLUMNS),
        "context_columns": list(bundle.target_manifest.get("context_features") or []),
        "feature_storage": "lazy_from_canonical_ohlcv",
        "trainable_parameter_count": int(trainable_parameter_count),
        "total_parameter_count": int(total_parameter_count),
        "frozen_parameter_count": int(total_parameter_count - trainable_parameter_count),
        "model": build_file_manifest(artifact_paths.model_path),
        "date_split": build_file_manifest(split_path),
        "outer_oos_policy": bundle.outer_policy,
        "selected_epoch": selected_epoch,
        "model_information_cutoff": information_cutoff,
        "source_dataset": bundle.summary,
        "source_continuous_target": bundle.target_manifest,
        "reference_target_evaluation": reference_target_evaluation,
        "runtime_eligibility": payload["runtime_eligibility"],
        "score_eligibility_contract": payload["score_eligibility_contract"],
        "forward_score_coverage": payload["forward_score_coverage"],
        "research_outputs": {
            "oos_scores_gzip": build_file_manifest(score_path),
            "report_json": build_file_manifest(report_json_path),
            "report_markdown": build_file_manifest(report_markdown_path),
        },
    }
    write_json(artifact_paths.manifest_path, manifest)

    def _fmt_metric(value) -> str:
        return "-" if value is None else f"{float(value):.4f}"

    print(f"\nDaily Universal Continuous Model完成｜{research_spec.model_research_id}")
    if bundle.profile.training_objective == TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION:
        regression = dict(oos_metrics.get("raw_r_regression") or {})
        if raw_r_loss_name == "huber_raw_r":
            primary = f"OOS Huber={_fmt_metric(regression.get('huber_loss_raw_r'))}"
        else:
            primary = f"OOS MSE={_fmt_metric(regression.get('mse_raw_r'))} | RMSE={_fmt_metric(regression.get('rmse_raw_r'))}R"
        print(
            f"selected_epoch={selected_epoch} | {primary} "
            f"| MAE={_fmt_metric(regression.get('mae_raw_r'))}R | bias={_fmt_metric(regression.get('bias_raw_r'))}R"
        )
    if direct_hmhs_only:
        print(
            f"OOS HM/HS pair={_fmt_metric(oos_metrics.get('pairwise_concordance'))} | "
            f"PR-AUC={_fmt_metric(oos_metrics.get('global_average_precision'))} | "
            f"Top10×={_fmt_metric(dict(oos_metrics.get('top_10pct') or {}).get('hmhs_enrichment'))}"
        )
        print(
            f"breakout candidate slice: groups={candidate_metrics['group_count']:,} | "
            f"HM/HS pair={_fmt_metric(candidate_metrics.get('pairwise_concordance'))} | "
            f"PR-AUC={_fmt_metric(candidate_metrics.get('global_average_precision'))} | "
            f"Top10×={_fmt_metric(dict(candidate_metrics.get('top_10pct') or {}).get('hmhs_enrichment'))}"
        )
    elif pareto_pair_evaluation:
        pareto_oos = dict(pareto_pair_evaluation.get("oos") or {})
        print(
            f"OOS Pareto={_fmt_metric(pareto_oos.get('mean_daily_pareto_pair_concordance'))} "
            f"| comparable={int(pareto_oos.get('comparable_pair_count', 0) or 0):,} "
            f"| economic rho={_fmt_metric(oos_metrics.get('mean_daily_spearman'))} "
            f"| economic pair={_fmt_metric(oos_metrics.get('pairwise_concordance'))}"
        )
        print(
            f"breakout candidate slice: groups={candidate_metrics['group_count']:,} | "
            f"daily rho={_fmt_metric(candidate_metrics.get('mean_daily_spearman'))} | "
            f"pair={_fmt_metric(candidate_metrics.get('pairwise_concordance'))}"
        )
    else:
        print(
            f"OOS daily rho={_fmt_metric(oos_metrics.get('mean_daily_spearman'))} | "
            f"pair={_fmt_metric(oos_metrics.get('pairwise_concordance'))}"
        )
        print(
            f"breakout candidate slice: groups={candidate_metrics['group_count']:,} | "
            f"daily rho={_fmt_metric(candidate_metrics.get('mean_daily_spearman'))} | "
            f"pair={_fmt_metric(candidate_metrics.get('pairwise_concordance'))}"
        )
    print_artifact_paths(
        (("Daily ranker model", artifact_paths.model_path), ("Markdown", report_markdown_path), ("OOS scores", score_path)),
        project_root=PROJECT_ROOT,
    )
    return 0


__all__ = ["run"]
