"""11E research-only audit of removing the fixed 11A time-penalty term."""

from __future__ import annotations

import argparse
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.breakout_quality import STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE
from config.breakout_quality import (
    BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
)
from filters.breakout_quality.continuous_target import STRATEGY_ALIGNED_TARGET_ID
from filters.breakout_quality.contract import LABEL_PASS, LABEL_REJECT
from tools.filters.breakout_quality.audit_target_component_attribution import (
    ACTUAL_ATTRIBUTION_FILENAME,
    AUDIT_DIRNAME as ATTRIBUTION_AUDIT_DIRNAME,
    AUDIT_JSON_FILENAME as ATTRIBUTION_AUDIT_JSON_FILENAME,
    QUALIFIED_ATTRIBUTION_FILENAME,
    _ranker_dir,
    _read_json,
    _sha256_file,
)
from tools.filters.breakout_quality.common import write_json
from tools.filters.breakout_quality.train_continuous_ranker import _spearman

AUDIT_SCHEMA_VERSION = 1
AUDIT_DIRNAME = "target_time_penalty_ablation_audit"
AUDIT_JSON_FILENAME = "target_time_penalty_ablation_audit.json"
AUDIT_MARKDOWN_FILENAME = "target_time_penalty_ablation_audit.md"
QUALIFIED_ABLATION_FILENAME = "qualified_time_penalty_ablation.csv"
ACTUAL_ABLATION_FILENAME = "actual_trade_time_penalty_ablation.csv"
ABLATION_TARGET_ID = "strategy_aligned_opportunity_r_no_time_v1_audit"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "11E research-only Target Time-penalty Ablation Audit；"
            "固定比較11A原Target與只移除time penalty後的Target，不訓練、不擬合權重"
        )
    )
    parser.add_argument("--filter-id", default=BREAKOUT_QUALITY_DEFAULT_FILTER_ID)
    parser.add_argument(
        "--ranker-profile",
        default=STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE,
        choices=(STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE,),
    )
    return parser.parse_args(argv)


def _validated_source_csv(
    report: dict[str, Any],
    *,
    key: str,
    canonical_path: Path,
) -> Path:
    record = ((report.get("artifacts") or {}).get(key) or {})
    if not isinstance(record, dict):
        raise ValueError(f"11E 11D report缺少artifact: {key}")
    if not canonical_path.is_file():
        raise FileNotFoundError(f"11E找不到11D artifact: {canonical_path}")
    expected_hash = str(record.get("sha256") or "").lower()
    actual_hash = _sha256_file(canonical_path).lower()
    if not expected_hash or actual_hash != expected_hash:
        raise ValueError(f"11E偵測到11D artifact SHA256不一致: {key}")
    return canonical_path


def load_11e_input_frames(
    *,
    filter_id: str,
    ranker_profile: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ranker_dir = _ranker_dir(filter_id, ranker_profile)
    attribution_dir = ranker_dir / ATTRIBUTION_AUDIT_DIRNAME
    report_path = attribution_dir / ATTRIBUTION_AUDIT_JSON_FILENAME
    if not report_path.is_file():
        raise FileNotFoundError(
            f"11E需要11D audit: {report_path}；請先執行audit-target-attribution"
        )
    report = _read_json(report_path)
    if not str(report.get("status") or "").startswith("RESULT_AVAILABLE"):
        raise ValueError("11E只接受已完成的11D結果")
    interpretation = report.get("interpretation_contract") or {}
    if bool(interpretation.get("research_only")) is not True:
        raise ValueError("11E預期11D維持research-only")
    if bool(interpretation.get("training_performed")):
        raise ValueError("11E拒絕任何由11D訓練產生的工件")
    if str(report.get("continuous_target_id") or "") != STRATEGY_ALIGNED_TARGET_ID:
        raise ValueError("11E 11D continuous target id不一致")

    qualified_path = _validated_source_csv(
        report,
        key="qualified_attribution",
        canonical_path=attribution_dir / QUALIFIED_ATTRIBUTION_FILENAME,
    )
    actual_path = _validated_source_csv(
        report,
        key="actual_trade_attribution",
        canonical_path=attribution_dir / ACTUAL_ATTRIBUTION_FILENAME,
    )
    qualified = pd.read_csv(qualified_path, encoding="utf-8-sig")
    actual = pd.read_csv(actual_path, encoding="utf-8-sig")
    return qualified, actual, report


def attach_time_penalty_ablation(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "label",
        "model_score",
        "target_raw_r",
        "favorable_r",
        "adverse_r",
        "time_penalty_r",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"11E attribution frame缺少欄位: {missing}")
    work = frame.copy()
    numeric = sorted(required)
    if "r_multiple" in work.columns:
        numeric.append("r_multiple")
    for column in numeric:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    valid = np.isfinite(
        work[[
            "model_score",
            "target_raw_r",
            "favorable_r",
            "adverse_r",
            "time_penalty_r",
        ]].to_numpy(dtype=np.float64)
    ).all(axis=1)
    work = work[valid].copy()
    work["target_no_time_r"] = work["favorable_r"] - work["adverse_r"]
    work["target_original_reconstructed_r"] = (
        work["target_no_time_r"] - work["time_penalty_r"]
    )
    if not np.allclose(
        work["target_original_reconstructed_r"].to_numpy(dtype=np.float64),
        work["target_raw_r"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=2e-5,
    ):
        raise ValueError("11E原Target無法由no-time Target減time penalty精確重建")
    return work


def _correlation(frame: pd.DataFrame, x: str, y: str) -> float | None:
    xv = pd.to_numeric(frame[x], errors="coerce").to_numpy(dtype=np.float64)
    yv = pd.to_numeric(frame[y], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(xv) & np.isfinite(yv)
    return _spearman(xv[valid], yv[valid]) if int(valid.sum()) >= 2 else None


def _decile_average(
    frame: pd.DataFrame,
    *,
    rank_column: str,
    value_column: str,
    top: bool,
) -> float | None:
    work = frame[[rank_column, value_column]].copy()
    work[rank_column] = pd.to_numeric(work[rank_column], errors="coerce")
    work[value_column] = pd.to_numeric(work[value_column], errors="coerce")
    work = work[
        np.isfinite(work[rank_column].to_numpy(dtype=np.float64))
        & np.isfinite(work[value_column].to_numpy(dtype=np.float64))
    ].sort_values(rank_column, kind="mergesort")
    if work.empty:
        return None
    count = max(1, int(math.ceil(len(work) * 0.10)))
    sample = work.tail(count) if top else work.head(count)
    return float(sample[value_column].mean())


def time_penalty_ablation_metrics(
    frame: pd.DataFrame,
    *,
    include_realized_r: bool,
) -> dict[str, Any]:
    work = attach_time_penalty_ablation(frame)
    correlations: dict[str, float | None] = {
        "score_vs_original_target": _correlation(work, "model_score", "target_raw_r"),
        "score_vs_no_time_target": _correlation(work, "model_score", "target_no_time_r"),
        "original_target_vs_no_time_target": _correlation(
            work, "target_raw_r", "target_no_time_r"
        ),
        "score_vs_time_penalty": _correlation(work, "model_score", "time_penalty_r"),
    }
    deciles: dict[str, float | None] = {
        "top_score_original_target_mean": _decile_average(
            work, rank_column="model_score", value_column="target_raw_r", top=True
        ),
        "bottom_score_original_target_mean": _decile_average(
            work, rank_column="model_score", value_column="target_raw_r", top=False
        ),
        "top_score_no_time_target_mean": _decile_average(
            work, rank_column="model_score", value_column="target_no_time_r", top=True
        ),
        "bottom_score_no_time_target_mean": _decile_average(
            work, rank_column="model_score", value_column="target_no_time_r", top=False
        ),
    }
    if include_realized_r:
        if "r_multiple" not in work.columns:
            raise ValueError("11E actual metrics缺少r_multiple")
        work = work[np.isfinite(pd.to_numeric(work["r_multiple"], errors="coerce"))].copy()
        correlations.update({
            "original_target_vs_realized_r": _correlation(
                work, "target_raw_r", "r_multiple"
            ),
            "no_time_target_vs_realized_r": _correlation(
                work, "target_no_time_r", "r_multiple"
            ),
            "time_penalty_vs_realized_r": _correlation(
                work, "time_penalty_r", "r_multiple"
            ),
            "score_vs_realized_r": _correlation(work, "model_score", "r_multiple"),
        })
        deciles.update({
            "top_original_target_decile_average_r": _decile_average(
                work, rank_column="target_raw_r", value_column="r_multiple", top=True
            ),
            "bottom_original_target_decile_average_r": _decile_average(
                work, rank_column="target_raw_r", value_column="r_multiple", top=False
            ),
            "top_no_time_target_decile_average_r": _decile_average(
                work, rank_column="target_no_time_r", value_column="r_multiple", top=True
            ),
            "bottom_no_time_target_decile_average_r": _decile_average(
                work, rank_column="target_no_time_r", value_column="r_multiple", top=False
            ),
        })

    by_label: dict[str, Any] = {}
    for label_value, key in ((LABEL_PASS, "pass"), (LABEL_REJECT, "reject")):
        subset = work[work["label"] == int(label_value)].copy()
        item: dict[str, Any] = {
            "row_count": int(len(subset)),
            "correlations": {
                "score_vs_original_target": _correlation(
                    subset, "model_score", "target_raw_r"
                ),
                "score_vs_no_time_target": _correlation(
                    subset, "model_score", "target_no_time_r"
                ),
            },
        }
        if include_realized_r:
            item["correlations"].update({
                "original_target_vs_realized_r": _correlation(
                    subset, "target_raw_r", "r_multiple"
                ),
                "no_time_target_vs_realized_r": _correlation(
                    subset, "target_no_time_r", "r_multiple"
                ),
                "time_penalty_vs_realized_r": _correlation(
                    subset, "time_penalty_r", "r_multiple"
                ),
                "score_vs_realized_r": _correlation(
                    subset, "model_score", "r_multiple"
                ),
            })
        by_label[key] = item

    original_rho = correlations.get("original_target_vs_realized_r")
    no_time_rho = correlations.get("no_time_target_vs_realized_r")
    original_spread = None
    no_time_spread = None
    if include_realized_r:
        top = deciles.get("top_original_target_decile_average_r")
        bottom = deciles.get("bottom_original_target_decile_average_r")
        if top is not None and bottom is not None:
            original_spread = float(top - bottom)
        top = deciles.get("top_no_time_target_decile_average_r")
        bottom = deciles.get("bottom_no_time_target_decile_average_r")
        if top is not None and bottom is not None:
            no_time_spread = float(top - bottom)
    return {
        "row_count": int(len(work)),
        "correlations": correlations,
        "deciles": deciles,
        "deltas": {
            "spearman_no_time_minus_original": (
                float(no_time_rho - original_rho)
                if original_rho is not None and no_time_rho is not None
                else None
            ),
            "decile_spread_no_time_minus_original": (
                float(no_time_spread - original_spread)
                if original_spread is not None and no_time_spread is not None
                else None
            ),
        },
        "by_label": by_label,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{numeric:.4f}" if math.isfinite(numeric) else "N/A"


def render_markdown(payload: dict[str, Any]) -> str:
    qualified = payload.get("qualified_candidates") or {}
    actual = payload.get("actual_trades") or {}
    q_corr = qualified.get("correlations") or {}
    a_corr = actual.get("correlations") or {}
    a_deciles = actual.get("deciles") or {}
    a_deltas = actual.get("deltas") or {}
    lines = [
        "# 11E Fixed Time-penalty Ablation Audit",
        "",
        "- Runtime：research-only；本輪不訓練、不選epoch、不調threshold。",
        "- 唯一變更：固定比較`target_no_time_r = favorable_r - adverse_r`；只移除11A原本固定time penalty，不擬合新係數。",
        "- OOS定位：迭代研究證據；不以本結果直接建立runtime target或模型。",
        "",
        "## 1. Qualified candidates",
        "",
        f"- Rows：`{qualified.get('row_count', 0):,}`。",
        f"- Score↔Original target：`{_fmt(q_corr.get('score_vs_original_target'))}`。",
        f"- Score↔No-time target：`{_fmt(q_corr.get('score_vs_no_time_target'))}`。",
        f"- Score↔Time penalty：`{_fmt(q_corr.get('score_vs_time_penalty'))}`。",
        "",
        "## 2. Actual trades",
        "",
        f"- Rows：`{actual.get('row_count', 0):,}`。",
        f"- Original target↔R：`{_fmt(a_corr.get('original_target_vs_realized_r'))}`。",
        f"- No-time target↔R：`{_fmt(a_corr.get('no_time_target_vs_realized_r'))}`。",
        f"- Δ Spearman：`{_fmt(a_deltas.get('spearman_no_time_minus_original'))}`。",
        f"- Original target top／bottom decile R：`{_fmt(a_deciles.get('top_original_target_decile_average_r'))}`／`{_fmt(a_deciles.get('bottom_original_target_decile_average_r'))}`。",
        f"- No-time target top／bottom decile R：`{_fmt(a_deciles.get('top_no_time_target_decile_average_r'))}`／`{_fmt(a_deciles.get('bottom_no_time_target_decile_average_r'))}`。",
        f"- Δ decile spread：`{_fmt(a_deltas.get('decile_spread_no_time_minus_original'))}`。",
        "",
        "## 3. Label-conditional actual trades",
        "",
        "| Label | Rows | Original Target↔R | No-time Target↔R | Δρ | Time↔R | Score↔No-time | Score↔R |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, title in (("pass", "PASS"), ("reject", "REJECT")):
        item = (actual.get("by_label") or {}).get(key) or {}
        corr = item.get("correlations") or {}
        original = corr.get("original_target_vs_realized_r")
        no_time = corr.get("no_time_target_vs_realized_r")
        delta = (
            float(no_time - original)
            if original is not None and no_time is not None
            else None
        )
        lines.append(
            f"| {title} | {item.get('row_count', 0):,} | "
            f"{_fmt(original)} | {_fmt(no_time)} | {_fmt(delta)} | "
            f"{_fmt(corr.get('time_penalty_vs_realized_r'))} | "
            f"{_fmt(corr.get('score_vs_no_time_target'))} | "
            f"{_fmt(corr.get('score_vs_realized_r'))} |"
        )
    lines += [
        "",
        "## 4. 判定邊界",
        "",
        "- 若No-time target在overall與PASS條件下均提高Target↔R及decile spread，才可進入新version target arrays與Selection-only可學性稽核；本輪仍不授權訓練。",
        "- 若只在REJECT改善或overall沒有改善，停止time-penalty ablation，不直接建立conditional magnitude head。",
        "- 不測time penalty反向加分、不同係數、不同horizon或OOS最佳權重；本輪唯一變更固定為移除該項。",
        "- 所有統計只作凍結工件歸因，不參與loss、gradient、epoch、normalization、sampling或runtime設定。",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    qualified, actual, report_11d = load_11e_input_frames(
        filter_id=args.filter_id,
        ranker_profile=args.ranker_profile,
    )
    qualified_frame = attach_time_penalty_ablation(qualified)
    actual_frame = attach_time_penalty_ablation(actual)
    qualified_metrics = time_penalty_ablation_metrics(
        qualified_frame,
        include_realized_r=False,
    )
    actual_metrics = time_penalty_ablation_metrics(
        actual_frame,
        include_realized_r=True,
    )

    output_dir = _ranker_dir(args.filter_id, args.ranker_profile) / AUDIT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    qualified_path = output_dir / QUALIFIED_ABLATION_FILENAME
    actual_path = output_dir / ACTUAL_ABLATION_FILENAME
    json_path = output_dir / AUDIT_JSON_FILENAME
    markdown_path = output_dir / AUDIT_MARKDOWN_FILENAME
    qualified_frame.to_csv(qualified_path, index=False, encoding="utf-8-sig")
    actual_frame.to_csv(actual_path, index=False, encoding="utf-8-sig")

    payload = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "experiment": "11E Fixed Time-penalty Ablation Audit",
        "status": "RESULT_AVAILABLE_PENDING_REVIEW",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "filter_id": args.filter_id,
        "model_architecture": BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
        "ranker_profile": args.ranker_profile,
        "source_continuous_target_id": STRATEGY_ALIGNED_TARGET_ID,
        "ablation_target_id": ABLATION_TARGET_ID,
        "ablation_formula": "target_no_time_r = favorable_r - adverse_r",
        "qualified_candidates": qualified_metrics,
        "actual_trades": actual_metrics,
        "source_11d": {
            "status": report_11d.get("status"),
            "generated_at_utc": report_11d.get("generated_at_utc"),
            "qualified_score_vs_target": (
                (report_11d.get("qualified_candidates") or {}).get("correlations") or {}
            ).get("score_vs_target"),
            "actual_target_vs_realized_r": (
                (report_11d.get("actual_trades") or {}).get("correlations") or {}
            ).get("target_vs_realized_r"),
            "actual_time_penalty_vs_realized_r": (
                (report_11d.get("actual_trades") or {}).get("correlations") or {}
            ).get("time_penalty_r_vs_realized_r"),
        },
        "artifacts": {
            "qualified_ablation": {
                "path": str(qualified_path),
                "sha256": _sha256_file(qualified_path),
            },
            "actual_trade_ablation": {
                "path": str(actual_path),
                "sha256": _sha256_file(actual_path),
            },
        },
        "interpretation_contract": {
            "research_only": True,
            "training_performed": False,
            "single_fixed_ablation": "remove_time_penalty_only",
            "no_oos_fitted_coefficient": True,
            "no_target_version_or_model_authorized_until_result_review": True,
            "no_loss_sampling_epoch_threshold_normalization_or_runtime_change": True,
        },
        "elapsed_sec": float(time.perf_counter() - started),
    }
    write_json(json_path, payload)
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")

    q = qualified_metrics.get("correlations") or {}
    a = actual_metrics.get("correlations") or {}
    d = actual_metrics.get("deltas") or {}
    print("11E fixed time-penalty ablation audit完成")
    print(
        f"qualified={qualified_metrics.get('row_count', 0):,} "
        f"actual_trades={actual_metrics.get('row_count', 0):,}"
    )
    print(
        "- Qualified Score↔Target: "
        f"original={_fmt(q.get('score_vs_original_target'))} "
        f"no_time={_fmt(q.get('score_vs_no_time_target'))}"
    )
    print(
        "- Actual Target↔R: "
        f"original={_fmt(a.get('original_target_vs_realized_r'))} "
        f"no_time={_fmt(a.get('no_time_target_vs_realized_r'))} "
        f"delta={_fmt(d.get('spearman_no_time_minus_original'))}"
    )
    print(f"已輸出: {markdown_path}")
    print(f"已輸出: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
