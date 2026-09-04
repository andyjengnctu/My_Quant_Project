"""Read-only daily-universal target comparison for controlled label experiments."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.breakout_quality_runtime_resolver import (
    get_continuous_ranker_execution_recipe,
)
from config.breakout_quality import (
    BREAKOUT_QUALITY_CONTINUOUS_RANKER_REPORT_TOP_K,
    BREAKOUT_QUALITY_TARGET_COMPARISON_BARRIER_BAND_RETURN,
    get_breakout_quality_experiment_profile,
    get_breakout_quality_model_research_settings,
    get_continuous_ranker_research_spec,
)
from filters.breakout_quality.daily_ranker_data import load_daily_universal_ranker_data
from filters.breakout_quality.continuous_target import (
    DAILY_FULL_HORIZON_OPPORTUNITY_TARGET_ID,
    DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
    DAILY_FIRST_RISK_BREACH_PURE_MFE_TARGET_ID,
    DAILY_OPPORTUNITY_NO_TIME_TARGET_ID,
)
from filters.breakout_quality.contract import DEFAULT_LABEL_POLICY
from filters.breakout_quality.workflow_io import PROJECT_ROOT, write_json
from core.console_report import print_artifact_paths
from services.breakout_quality.ranker_training import calculate_spearman

REPORT_JSON_FILENAME = "daily_target_comparison.json"
REPORT_MARKDOWN_FILENAME = "daily_target_comparison.md"
REPORT_BY_DATE_FILENAME = "daily_target_comparison_by_date.csv"


def _pure_mfe_float32_relation_tolerance(
    candidate_target_r: np.ndarray,
    reference_target_r: np.ndarray,
    adverse_return: np.ndarray,
    *,
    risk_budget_return: float,
) -> np.ndarray:
    """Return per-row tolerance implied only by persisted float32 quantization.

    Daily targets/components are materialized as float32. MR-13K's exact
    scientific relation is enforced before persistence by the canonical target
    builders; this audit checks the persisted artifacts without pretending that
    subtracting two independently rounded float32 targets is exact arithmetic.
    """

    risk_budget = float(risk_budget_return)
    if not np.isfinite(risk_budget) or risk_budget <= 0.0:
        raise ValueError("risk_budget_return必須是有限正數")
    candidate32 = np.asarray(candidate_target_r, dtype=np.float32)
    reference32 = np.asarray(reference_target_r, dtype=np.float32)
    adverse32 = np.asarray(adverse_return, dtype=np.float32)
    candidate_ulp = np.abs(np.spacing(candidate32)).astype(np.float64)
    reference_ulp = np.abs(np.spacing(reference32)).astype(np.float64)
    adverse_ulp_r = np.abs(np.spacing(adverse32)).astype(np.float64) / risk_budget
    return candidate_ulp + reference_ulp + adverse_ulp_r + np.finfo(np.float32).eps


def _controlled_profile_contract(profile) -> dict[str, object]:
    payload = dict(profile.as_manifest_payload())
    payload.pop("name", None)
    payload.pop("continuous_target_id", None)
    return payload


def _validate_controlled_pair(candidate_profile: str, reference_profile: str) -> None:
    candidate = get_breakout_quality_experiment_profile(candidate_profile)
    reference = get_breakout_quality_experiment_profile(reference_profile)
    if _controlled_profile_contract(candidate) != _controlled_profile_contract(reference):
        raise ValueError(
            "daily target comparison要求candidate/reference除continuous target外的training profile完全相同"
        )
    candidate_recipe = get_continuous_ranker_execution_recipe(candidate_profile)
    reference_recipe = get_continuous_ranker_execution_recipe(reference_profile)
    if candidate_recipe.trainer_family != reference_recipe.trainer_family:
        raise ValueError("daily target comparison trainer family不一致")
    if candidate_recipe.pairwise_reduction != reference_recipe.pairwise_reduction:
        raise ValueError("daily target comparison pairwise reduction不一致")


def _controlled_change_contract(candidate_target_id: str, reference_target_id: str) -> dict[str, object]:
    pair = (str(candidate_target_id), str(reference_target_id))
    if pair == (DAILY_FULL_HORIZON_OPPORTUNITY_TARGET_ID, DAILY_OPPORTUNITY_NO_TIME_TARGET_ID):
        return {
            "change_id": "remove_first_risk_breach_path_truncation_only",
            "description": "只移除 first risk-breach 對future path的截斷；horizon、R scale、adverse-to-peak與training profile固定。",
            "enforce_no_breach_invariant": True,
            "enforce_same_peak_components": False,
        }
    if pair == (DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID, DAILY_FULL_HORIZON_OPPORTUNITY_TARGET_ID):
        return {
            "change_id": "remove_adverse_to_peak_penalty_only",
            "description": "只移除 adverse-to-peak 的Target扣分；完整40D、earliest max-high、breach diagnostics、R scale與training profile固定。",
            "enforce_no_breach_invariant": False,
            "enforce_same_peak_components": True,
        }
    if pair == (DAILY_FIRST_RISK_BREACH_PURE_MFE_TARGET_ID, DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID):
        return {
            "change_id": "add_first_risk_breach_path_truncation_to_pure_mfe_only",
            "description": "只把Pure-MFE future path改為MR-13E既有first-risk-breach截斷；same-bar adverse-first、40D、R scale與training profile固定，adverse magnitude仍不扣分。",
            "enforce_no_breach_invariant": True,
            "enforce_same_peak_components": False,
        }
    raise ValueError(
        "daily target comparison尚未註冊此受控Target pair: "
        f"candidate={candidate_target_id!r}, reference={reference_target_id!r}"
    )


def _top_overlap(left: np.ndarray, right: np.ndarray, k: int) -> float | None:
    size = min(int(k), len(left), len(right))
    if size < 1:
        return None
    left_ids = set(np.argsort(-left, kind="mergesort")[:size].tolist())
    right_ids = set(np.argsort(-right, kind="mergesort")[:size].tolist())
    return float(len(left_ids & right_ids) / size)


def _comparison_metrics(
    frame: pd.DataFrame,
    *,
    top_k: int,
    risk_barrier_return: float,
    barrier_band_return: float,
) -> tuple[dict[str, object], pd.DataFrame]:
    required = {
        "date",
        "reference_target_r",
        "candidate_target_r",
        "first_risk_breach_bar",
        "reference_opportunity_bar",
        "candidate_opportunity_bar",
        "minimum_low_return",
        "reference_adverse_return",
        "candidate_adverse_return",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"target comparison缺少欄位: {missing}")
    if frame.empty:
        raise ValueError("target comparison沒有共同有效sample")

    reference = frame["reference_target_r"].to_numpy(dtype=np.float64)
    candidate = frame["candidate_target_r"].to_numpy(dtype=np.float64)
    delta = candidate - reference
    reference_adverse = frame["reference_adverse_return"].to_numpy(dtype=np.float64)
    candidate_adverse = frame["candidate_adverse_return"].to_numpy(dtype=np.float64)
    adverse_delta = candidate_adverse - reference_adverse
    breach = frame["first_risk_breach_bar"].to_numpy(dtype=np.int64) >= 1
    later_peak = (
        frame["candidate_opportunity_bar"].to_numpy(dtype=np.int64)
        > frame["first_risk_breach_bar"].to_numpy(dtype=np.int64)
    )
    changed = ~np.isclose(reference, candidate, rtol=0.0, atol=1e-7)
    no_breach = ~breach
    no_breach_max_abs_delta = (
        float(np.max(np.abs(delta[no_breach]))) if bool(np.any(no_breach)) else 0.0
    )

    risk_barrier_return = float(risk_barrier_return)
    barrier_band_return = float(barrier_band_return)
    if not np.isfinite(risk_barrier_return) or risk_barrier_return >= 0.0:
        raise ValueError("risk_barrier_return必須是有限負值")
    if not np.isfinite(barrier_band_return) or barrier_band_return <= 0.0:
        raise ValueError("barrier_band_return必須是有限正值")
    minimum_low = frame["minimum_low_return"].to_numpy(dtype=np.float64)
    just_above = (minimum_low > risk_barrier_return) & (
        minimum_low <= risk_barrier_return + barrier_band_return
    )
    just_below = (minimum_low <= risk_barrier_return) & (
        minimum_low >= risk_barrier_return - barrier_band_return
    )

    def barrier_slice(mask: np.ndarray) -> dict[str, object]:
        count = int(np.count_nonzero(mask))
        if count == 0:
            return {
                "sample_count": 0,
                "reference_mean_r": None,
                "candidate_mean_r": None,
                "mean_delta_r": None,
                "mean_abs_delta_r": None,
                "changed_target_rate": None,
            }
        return {
            "sample_count": count,
            "reference_mean_r": float(np.mean(reference[mask])),
            "candidate_mean_r": float(np.mean(candidate[mask])),
            "mean_delta_r": float(np.mean(delta[mask])),
            "mean_abs_delta_r": float(np.mean(np.abs(delta[mask]))),
            "changed_target_rate": float(np.mean(changed[mask])),
        }

    daily_rows: list[dict[str, object]] = []
    daily_spearman: list[float] = []
    daily_top10_overlap: list[float] = []
    daily_topk_overlap: list[float] = []
    for date_value, group in frame.groupby("date", sort=True):
        ref = group["reference_target_r"].to_numpy(dtype=np.float64)
        cand = group["candidate_target_r"].to_numpy(dtype=np.float64)
        rho = calculate_spearman(ref, cand)
        top10_k = max(1, int(np.ceil(len(group) * 0.10)))
        top10 = _top_overlap(ref, cand, top10_k)
        topk = _top_overlap(ref, cand, int(top_k))
        if rho is not None:
            daily_spearman.append(float(rho))
        if top10 is not None:
            daily_top10_overlap.append(float(top10))
        if topk is not None:
            daily_topk_overlap.append(float(topk))
        daily_rows.append(
            {
                "date": str(pd.Timestamp(date_value).date()),
                "sample_count": int(len(group)),
                "spearman_reference_vs_candidate": rho,
                "top_10pct_overlap": top10,
                "top_k_overlap": topk,
                "breach_rate": float(np.mean(group["first_risk_breach_bar"].to_numpy(dtype=np.int64) >= 1)),
                "changed_target_rate": float(
                    np.mean(
                        ~np.isclose(
                            group["reference_target_r"].to_numpy(dtype=np.float64),
                            group["candidate_target_r"].to_numpy(dtype=np.float64),
                            rtol=0.0,
                            atol=1e-7,
                        )
                    )
                ),
                "mean_target_delta_r": float(
                    np.mean(
                        group["candidate_target_r"].to_numpy(dtype=np.float64)
                        - group["reference_target_r"].to_numpy(dtype=np.float64)
                    )
                ),
            }
        )

    breach_count = int(np.count_nonzero(breach))
    later_peak_count = int(np.count_nonzero(breach & later_peak))
    metrics = {
        "common_sample_count": int(len(frame)),
        "date_count": int(frame["date"].nunique()),
        "global_spearman_reference_vs_candidate": calculate_spearman(reference, candidate),
        "mean_daily_spearman_reference_vs_candidate": (
            float(np.mean(daily_spearman)) if daily_spearman else None
        ),
        "median_daily_spearman_reference_vs_candidate": (
            float(np.median(daily_spearman)) if daily_spearman else None
        ),
        "mean_daily_top_10pct_overlap": (
            float(np.mean(daily_top10_overlap)) if daily_top10_overlap else None
        ),
        "top_k": int(top_k),
        "mean_daily_top_k_overlap": (
            float(np.mean(daily_topk_overlap)) if daily_topk_overlap else None
        ),
        "risk_breach_count": breach_count,
        "risk_breach_rate": float(breach_count / len(frame)),
        "post_breach_later_peak_count": later_peak_count,
        "post_breach_later_peak_rate_within_breach": (
            float(later_peak_count / breach_count) if breach_count else None
        ),
        "changed_target_count": int(np.count_nonzero(changed)),
        "changed_target_rate": float(np.mean(changed)),
        "changed_target_rate_within_breach": (
            float(np.mean(changed[breach])) if breach_count else None
        ),
        "no_breach_max_abs_target_delta_r": no_breach_max_abs_delta,
        "barrier_cliff": {
            "risk_barrier_return": risk_barrier_return,
            "band_return": barrier_band_return,
            "just_above": barrier_slice(just_above),
            "just_below": barrier_slice(just_below),
        },
        "target_delta_r": {
            "mean": float(np.mean(delta)),
            "median": float(np.median(delta)),
            "mean_abs": float(np.mean(np.abs(delta))),
            "p10": float(np.quantile(delta, 0.10)),
            "p90": float(np.quantile(delta, 0.90)),
        },
        "adverse_component_return": {
            "reference_mean": float(np.mean(reference_adverse)),
            "candidate_mean": float(np.mean(candidate_adverse)),
            "max_abs_component_delta": float(np.max(np.abs(adverse_delta))),
        },
        "reference_target_r": {
            "mean": float(np.mean(reference)),
            "std": float(np.std(reference)),
            "p10": float(np.quantile(reference, 0.10)),
            "p50": float(np.quantile(reference, 0.50)),
            "p90": float(np.quantile(reference, 0.90)),
        },
        "candidate_target_r": {
            "mean": float(np.mean(candidate)),
            "std": float(np.std(candidate)),
            "p10": float(np.quantile(candidate, 0.10)),
            "p50": float(np.quantile(candidate, 0.50)),
            "p90": float(np.quantile(candidate, 0.90)),
        },
    }
    return metrics, pd.DataFrame(daily_rows)


def _render_markdown(payload: dict[str, object]) -> str:
    metrics = dict(payload["metrics"])

    def fmt(value, digits: int = 4) -> str:
        return "-" if value is None else f"{float(value):.{digits}f}"

    def pct(value) -> str:
        return "-" if value is None else f"{float(value) * 100.0:.2f}%"

    delta = dict(metrics.get("target_delta_r") or {})
    ref_dist = dict(metrics.get("reference_target_r") or {})
    cand_dist = dict(metrics.get("candidate_target_r") or {})
    cliff = dict(metrics.get("barrier_cliff") or {})
    cliff_above = dict(cliff.get("just_above") or {})
    cliff_below = dict(cliff.get("just_below") or {})
    lines = [
        "# Daily Target Comparison",
        "",
        f"- Candidate profile：`{payload['candidate_profile']}`",
        f"- Candidate target：`{payload['candidate_target_id']}`",
        f"- Reference profile：`{payload['reference_profile']}`",
        f"- Reference target：`{payload['reference_target_id']}`",
        "- Boundary：read-only Label Audit；不訓練模型、不使用OOS結果做label fitting。",
        f"- Controlled change：{payload['controlled_change_description']}",
        "",
        "## Core",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Common stock-days | {int(metrics['common_sample_count']):,} |",
        f"| Dates | {int(metrics['date_count']):,} |",
        f"| Global rank correlation | {fmt(metrics.get('global_spearman_reference_vs_candidate'))} |",
        f"| Mean daily rank correlation | {fmt(metrics.get('mean_daily_spearman_reference_vs_candidate'))} |",
        f"| Mean daily Top-10% overlap | {pct(metrics.get('mean_daily_top_10pct_overlap'))} |",
        f"| Mean daily Top-{int(metrics.get('top_k') or 0)} overlap | {pct(metrics.get('mean_daily_top_k_overlap'))} |",
        f"| 40D risk-breach rate | {pct(metrics.get('risk_breach_rate'))} |",
        f"| Breach後candidate peak更晚比例 | {pct(metrics.get('post_breach_later_peak_rate_within_breach'))} |",
        f"| Target changed rate | {pct(metrics.get('changed_target_rate'))} |",
        f"| Breach samples target changed rate | {pct(metrics.get('changed_target_rate_within_breach'))} |",
        f"| No-breach max absolute delta | {fmt(metrics.get('no_breach_max_abs_target_delta_r'), 8)} R |",
        "",
        "## Barrier Cliff",
        "",
        f"Canonical barrier：`{fmt(cliff.get('risk_barrier_return'))}`；band：`±{fmt(cliff.get('band_return'))}`。",
        "",
        "| Slice | Samples | Reference mean R | Candidate mean R | Mean ΔR | Mean abs ΔR | Changed |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Just above barrier | {int(cliff_above.get('sample_count') or 0):,} | {fmt(cliff_above.get('reference_mean_r'))} | {fmt(cliff_above.get('candidate_mean_r'))} | {fmt(cliff_above.get('mean_delta_r'))} | {fmt(cliff_above.get('mean_abs_delta_r'))} | {pct(cliff_above.get('changed_target_rate'))} |",
        f"| Just below barrier | {int(cliff_below.get('sample_count') or 0):,} | {fmt(cliff_below.get('reference_mean_r'))} | {fmt(cliff_below.get('candidate_mean_r'))} | {fmt(cliff_below.get('mean_delta_r'))} | {fmt(cliff_below.get('mean_abs_delta_r'))} | {pct(cliff_below.get('changed_target_rate'))} |",
        "",
        "## Target Distribution",
        "",
        "| Target | Mean | Std | P10 | P50 | P90 |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Reference | {fmt(ref_dist.get('mean'))} | {fmt(ref_dist.get('std'))} | {fmt(ref_dist.get('p10'))} | {fmt(ref_dist.get('p50'))} | {fmt(ref_dist.get('p90'))} |",
        f"| Candidate | {fmt(cand_dist.get('mean'))} | {fmt(cand_dist.get('std'))} | {fmt(cand_dist.get('p10'))} | {fmt(cand_dist.get('p50'))} | {fmt(cand_dist.get('p90'))} |",
        f"| Candidate - Reference | {fmt(delta.get('mean'))} | - | {fmt(delta.get('p10'))} | {fmt(delta.get('median'))} | {fmt(delta.get('p90'))} |",
        "",
        f"- Mean absolute target delta：`{fmt(delta.get('mean_abs'))} R`",
        "- 此報表不設自動GO/REJECT門檻；用途是確認單一Label簡化對daily ranking與Target分布的實質影響。",
    ]
    return "\n".join(lines) + "\n"


def compare_daily_targets(
    *,
    filter_id: str,
    model_architecture: str,
    candidate_profile: str,
    reference_profile: str,
    allow_stale_source: bool,
    project_root: str | Path = PROJECT_ROOT,
) -> dict[str, object]:
    _validate_controlled_pair(candidate_profile, reference_profile)
    candidate = load_daily_universal_ranker_data(
        filter_id=filter_id,
        model_architecture=model_architecture,
        experiment_profile=candidate_profile,
        preload_feature_bank=False,
        allow_stale_source=allow_stale_source,
        project_root=project_root,
    )
    reference = load_daily_universal_ranker_data(
        filter_id=filter_id,
        model_architecture=model_architecture,
        experiment_profile=reference_profile,
        preload_feature_bank=False,
        allow_stale_source=allow_stale_source,
        project_root=project_root,
    )

    key_columns = ["ticker", "date", "source_pos", "label_eval_end_date"]
    candidate_keys = candidate.group_table[key_columns].astype(str)
    reference_keys = reference.group_table[key_columns].astype(str)
    if len(candidate_keys) != len(reference_keys) or not candidate_keys.equals(reference_keys):
        raise ValueError("candidate/reference daily stock-day universe或順序不一致，無法做受控Label比較")
    common = np.asarray(candidate.target_valid, dtype=bool) & np.asarray(reference.target_valid, dtype=bool)
    ids = np.flatnonzero(common)
    if len(ids) < 2:
        raise ValueError("candidate/reference共同有效daily target sample不足")

    ctable = candidate.group_table.iloc[ids]
    rtable = reference.group_table.iloc[ids]
    frame = pd.DataFrame(
        {
            "ticker": ctable["ticker"].astype(str).to_numpy(),
            "date": pd.to_datetime(ctable["date"], errors="raise").dt.normalize().to_numpy(),
            "reference_target_r": reference.raw_target[ids],
            "candidate_target_r": candidate.raw_target[ids],
            "first_risk_breach_bar": rtable["target_first_risk_breach_bar"].to_numpy(dtype=np.int16),
            "reference_opportunity_bar": rtable["target_opportunity_bar"].to_numpy(dtype=np.int16),
            "candidate_opportunity_bar": ctable["target_opportunity_bar"].to_numpy(dtype=np.int16),
            "minimum_low_return": rtable["target_minimum_low_return"].to_numpy(dtype=np.float32),
            "reference_adverse_return": rtable["target_adverse_return_to_peak"].to_numpy(dtype=np.float32),
            "candidate_adverse_return": ctable["target_adverse_return_to_peak"].to_numpy(dtype=np.float32),
        }
    )
    if not np.array_equal(
        rtable["target_first_risk_breach_bar"].to_numpy(dtype=np.int16),
        ctable["target_first_risk_breach_bar"].to_numpy(dtype=np.int16),
    ):
        raise ValueError("candidate/reference first-risk-breach診斷不一致")

    candidate_target_id = str(candidate.profile.continuous_target_id)
    reference_target_id = str(reference.profile.continuous_target_id)
    controlled_change = _controlled_change_contract(candidate_target_id, reference_target_id)
    metrics, daily = _comparison_metrics(
        frame,
        top_k=int(BREAKOUT_QUALITY_CONTINUOUS_RANKER_REPORT_TOP_K),
        risk_barrier_return=-abs(float(DEFAULT_LABEL_POLICY.max_adverse_return)),
        barrier_band_return=float(BREAKOUT_QUALITY_TARGET_COMPARISON_BARRIER_BAND_RETURN),
    )
    if bool(controlled_change["enforce_no_breach_invariant"]):
        if float(metrics["no_breach_max_abs_target_delta_r"]) > 1e-6:
            raise ValueError("No-breach samples的新舊target不應改變；受控實驗契約已破壞")
    if bool(controlled_change["enforce_same_peak_components"]):
        if not np.array_equal(
            rtable["target_opportunity_bar"].to_numpy(dtype=np.int16),
            ctable["target_opportunity_bar"].to_numpy(dtype=np.int16),
        ):
            raise ValueError("移除adverse penalty不得改變selected opportunity bar")
        if not np.allclose(
            rtable["target_favorable_return"].to_numpy(dtype=np.float64),
            ctable["target_favorable_return"].to_numpy(dtype=np.float64),
            rtol=0.0, atol=1e-7, equal_nan=True,
        ):
            raise ValueError("移除adverse penalty不得改變favorable component")
        if float((metrics.get("adverse_component_return") or {}).get("max_abs_component_delta") or 0.0) > 1e-7:
            raise ValueError("移除adverse penalty不得改變adverse diagnostic component")
        risk_budget = abs(float(DEFAULT_LABEL_POLICY.max_adverse_return))
        reference_adverse = rtable["target_adverse_return_to_peak"].to_numpy(dtype=np.float64)
        expected_delta = reference_adverse / risk_budget
        candidate_target_values = frame["candidate_target_r"].to_numpy(dtype=np.float64)
        reference_target_values = frame["reference_target_r"].to_numpy(dtype=np.float64)
        actual_delta = candidate_target_values - reference_target_values
        tolerance = _pure_mfe_float32_relation_tolerance(
            candidate_target_values,
            reference_target_values,
            reference_adverse,
            risk_budget_return=risk_budget,
        )
        error = np.abs(actual_delta - expected_delta)
        violated = error > tolerance
        if bool(np.any(violated)):
            worst = int(np.argmax(error - tolerance))
            raise ValueError(
                "Pure-MFE target差值不符合移除adverse R penalty的float32持久化契約；"
                f"max_error={error[worst]:.9g}, allowed={tolerance[worst]:.9g}"
            )

    root = Path(project_root)
    output_dir = (
        root
        / "outputs"
        / "filters"
        / "breakout_quality"
        / filter_id
        / "daily_target_comparison"
        / f"{candidate_target_id}__vs__{reference_target_id}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / REPORT_JSON_FILENAME
    markdown_path = output_dir / REPORT_MARKDOWN_FILENAME
    daily_path = output_dir / REPORT_BY_DATE_FILENAME
    payload: dict[str, object] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "RESULT_AVAILABLE_PENDING_REVIEW",
        "filter_id": filter_id,
        "model_architecture": model_architecture,
        "candidate_profile": candidate_profile,
        "candidate_target_id": candidate_target_id,
        "reference_profile": reference_profile,
        "reference_target_id": reference_target_id,
        "controlled_change": str(controlled_change["change_id"]),
        "controlled_change_description": str(controlled_change["description"]),
        "training_or_model_execution": False,
        "metrics": metrics,
        "artifacts": {
            "json": str(json_path.relative_to(root)).replace("\\", "/"),
            "markdown": str(markdown_path.relative_to(root)).replace("\\", "/"),
            "by_date_csv": str(daily_path.relative_to(root)).replace("\\", "/"),
        },
    }
    write_json(json_path, payload)
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")

    print("\nDaily Target Comparison完成")
    print(
        "common={:,} | breach={} | changed={} | daily rho={} | Top-{} overlap={}".format(
            int(metrics["common_sample_count"]),
            f"{float(metrics['risk_breach_rate']) * 100:.2f}%",
            f"{float(metrics['changed_target_rate']) * 100:.2f}%",
            "-" if metrics["mean_daily_spearman_reference_vs_candidate"] is None else f"{float(metrics['mean_daily_spearman_reference_vs_candidate']):.4f}",
            int(metrics["top_k"]),
            "-" if metrics["mean_daily_top_k_overlap"] is None else f"{float(metrics['mean_daily_top_k_overlap']) * 100:.2f}%",
        )
    )
    print_artifact_paths(
        (("Markdown", markdown_path), ("JSON", json_path), ("By-date CSV", daily_path)),
        project_root=root,
    )
    return payload


def parse_args(argv=None) -> argparse.Namespace:
    settings = get_breakout_quality_model_research_settings()
    spec = get_continuous_ranker_research_spec(settings.experiment_profile)
    parser = argparse.ArgumentParser(description="比較active daily target與其reference target；只讀、不訓練")
    parser.add_argument("--filter-id", default=settings.filter_id)
    parser.add_argument("--model-architecture", default=settings.model_architecture)
    parser.add_argument("--experiment-profile", default=settings.experiment_profile)
    parser.add_argument(
        "--reference-experiment-profile",
        default=spec.reference_profile_name,
    )
    parser.add_argument("--allow-stale-source", action="store_true")
    args = parser.parse_args(argv)
    if not str(args.reference_experiment_profile or "").strip():
        parser.error("active experiment profile未設定reference_experiment_profile")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    compare_daily_targets(
        filter_id=str(args.filter_id),
        model_architecture=str(args.model_architecture),
        candidate_profile=str(args.experiment_profile),
        reference_profile=str(args.reference_experiment_profile),
        allow_stale_source=bool(args.allow_stale_source),
    )
    return 0


__all__ = [
    "REPORT_BY_DATE_FILENAME",
    "REPORT_JSON_FILENAME",
    "REPORT_MARKDOWN_FILENAME",
    "_comparison_metrics",
    "compare_daily_targets",
    "main",
    "parse_args",
]


if __name__ == "__main__":
    raise SystemExit(main())
