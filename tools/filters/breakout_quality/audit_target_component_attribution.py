"""11D research-only label-conditional attribution of the 11A target and 11B score."""

from __future__ import annotations

import argparse
import hashlib
import json
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
from filters.breakout_quality.continuous_target import (
    STRATEGY_ALIGNED_TARGET_ID,
    load_validated_continuous_target_component_arrays,
)
from filters.breakout_quality.contract import LABEL_PASS, LABEL_REJECT
from filters.breakout_quality.paths import resolve_filter_model_output_dir
from tools.filters.breakout_quality.audit_qualified_candidate_set import (
    ACTUAL_TRADE_MATCHES_FILENAME,
    AUDIT_DIRNAME as QUALIFIED_AUDIT_DIRNAME,
    AUDIT_JSON_FILENAME as QUALIFIED_AUDIT_JSON_FILENAME,
    QUALIFIED_GROUPS_FILENAME,
)
from tools.filters.breakout_quality.common import PROJECT_ROOT, write_json
from tools.filters.breakout_quality.train_continuous_ranker import (
    RANKER_SCORE_FILENAME,
    _spearman,
)

AUDIT_SCHEMA_VERSION = 1
AUDIT_DIRNAME = "target_component_attribution_audit"
AUDIT_JSON_FILENAME = "target_component_attribution_audit.json"
AUDIT_MARKDOWN_FILENAME = "target_component_attribution_audit.md"
QUALIFIED_ATTRIBUTION_FILENAME = "qualified_target_component_attribution.csv"
ACTUAL_ATTRIBUTION_FILENAME = "actual_trade_target_component_attribution.csv"

_COMPONENT_COLUMNS = (
    "target_raw_r",
    "favorable_r",
    "adverse_r",
    "time_penalty_r",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "11D research-only Target Component Attribution Audit；"
            "分解11A favorable/adverse/time成分並依PASS/REJECT檢查11B學到的成分"
        )
    )
    parser.add_argument("--filter-id", default=BREAKOUT_QUALITY_DEFAULT_FILTER_ID)
    parser.add_argument(
        "--ranker-profile",
        default=STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE,
        choices=(STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE,),
    )
    return parser.parse_args(argv)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"讀取JSON失敗: {path}｜{type(exc).__name__}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root必須是object: {path}")
    return payload


def _ranker_dir(filter_id: str, ranker_profile: str) -> Path:
    return resolve_filter_model_output_dir(
        PROJECT_ROOT,
        filter_id,
        BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
        ranker_profile,
    )


def _validated_csv_from_report(
    report: dict[str, Any],
    *,
    key: str,
    canonical_path: Path,
) -> Path:
    record = ((report.get("artifacts") or {}).get(key) or {})
    if not isinstance(record, dict):
        raise ValueError(f"11D 11C report缺少artifact: {key}")
    if not canonical_path.is_file():
        raise FileNotFoundError(f"11D找不到11C artifact: {canonical_path}")
    expected_hash = str(record.get("sha256") or "").lower()
    if not expected_hash or _sha256_file(canonical_path).lower() != expected_hash:
        raise ValueError(f"11D偵測到11C artifact SHA256不一致: {key}")
    return canonical_path


def load_11d_input_frames(
    *,
    filter_id: str,
    ranker_profile: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ranker_dir = _ranker_dir(filter_id, ranker_profile)
    score_path = ranker_dir / RANKER_SCORE_FILENAME
    if not score_path.is_file():
        raise FileNotFoundError(f"11D需要11B scores: {score_path}")

    qualified_dir = ranker_dir / QUALIFIED_AUDIT_DIRNAME
    report_path = qualified_dir / QUALIFIED_AUDIT_JSON_FILENAME
    if not report_path.is_file():
        raise FileNotFoundError(
            f"11D需要11C audit: {report_path}；請先執行audit-qualified-candidate-set"
        )
    report = _read_json(report_path)
    if not str(report.get("status") or "").startswith("RESULT_AVAILABLE"):
        raise ValueError("11D只接受已完成的11C結果")
    if bool((report.get("interpretation_contract") or {}).get("research_only")) is not True:
        raise ValueError("11D預期11C維持research-only")
    if bool(report.get("training_performed")):
        raise ValueError("11D拒絕任何由11C訓練產生的工件")
    expected_score_hash = str(report.get("ranker_score_sha256") or "").lower()
    if not expected_score_hash or _sha256_file(score_path).lower() != expected_score_hash:
        raise ValueError("11D偵測到11B scores SHA256與11C report不一致")

    qualified_path = _validated_csv_from_report(
        report,
        key="qualified_groups",
        canonical_path=qualified_dir / QUALIFIED_GROUPS_FILENAME,
    )
    actual_path = _validated_csv_from_report(
        report,
        key="actual_trade_matches",
        canonical_path=qualified_dir / ACTUAL_TRADE_MATCHES_FILENAME,
    )
    scores = pd.read_csv(score_path, encoding="utf-8-sig")
    qualified = pd.read_csv(qualified_path, encoding="utf-8-sig")
    actual = pd.read_csv(actual_path, encoding="utf-8-sig")
    return scores, qualified, actual, report


def attach_target_components(
    frame: pd.DataFrame,
    *,
    arrays: dict[str, np.ndarray],
    target_contract: dict[str, Any],
) -> pd.DataFrame:
    required = {"group_index", "label", "target_raw_r", "model_score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"11D attribution frame缺少欄位: {missing}")
    work = frame.copy()
    work["group_index"] = pd.to_numeric(work["group_index"], errors="raise").astype(np.int64)
    work["label"] = pd.to_numeric(work["label"], errors="raise").astype(np.int64)
    work["target_raw_r"] = pd.to_numeric(work["target_raw_r"], errors="coerce")
    work["model_score"] = pd.to_numeric(work["model_score"], errors="coerce")
    indexes = work["group_index"].to_numpy(dtype=np.int64)
    group_count = len(arrays["target_raw_r"])
    if bool(np.any(indexes < 0)) or bool(np.any(indexes >= group_count)):
        raise ValueError("11D group_index超出continuous target arrays範圍")
    valid_mask = np.asarray(arrays["valid_mask"], dtype=bool)[indexes]
    if not bool(valid_mask.all()):
        raise ValueError("11D attribution rows包含invalid continuous target group")

    risk_budget = float(target_contract.get("risk_budget_return"))
    horizon = int(target_contract.get("horizon_bars"))
    full_penalty = float(target_contract.get("full_horizon_time_penalty_r"))
    if not math.isfinite(risk_budget) or risk_budget <= 0.0 or horizon < 2:
        raise ValueError("11D continuous target contract不合法")

    favorable = np.asarray(arrays["favorable_return"], dtype=np.float64)[indexes]
    adverse = np.asarray(arrays["adverse_return_to_peak"], dtype=np.float64)[indexes]
    opportunity = np.asarray(arrays["opportunity_bar"], dtype=np.int64)[indexes]
    target_array = np.asarray(arrays["target_raw_r"], dtype=np.float64)[indexes]
    work["favorable_r"] = favorable / risk_budget
    work["adverse_r"] = adverse / risk_budget
    work["time_penalty_r"] = full_penalty * (opportunity - 1) / float(horizon - 1)
    work["target_reconstructed_r"] = (
        work["favorable_r"] - work["adverse_r"] - work["time_penalty_r"]
    )
    if not np.allclose(
        work["target_reconstructed_r"].to_numpy(dtype=np.float64),
        target_array,
        rtol=0.0,
        atol=2e-5,
    ):
        raise ValueError("11D target component reconstruction與11A array不一致")
    if not np.allclose(
        work["target_raw_r"].to_numpy(dtype=np.float64),
        target_array,
        rtol=0.0,
        atol=2e-5,
    ):
        raise ValueError("11D輸入target_raw_r與11A array不一致")
    return work


def _finite_pair(frame: pd.DataFrame, x: str, y: str) -> tuple[np.ndarray, np.ndarray]:
    xv = pd.to_numeric(frame[x], errors="coerce").to_numpy(dtype=np.float64)
    yv = pd.to_numeric(frame[y], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(xv) & np.isfinite(yv)
    return xv[valid], yv[valid]


def _correlation(frame: pd.DataFrame, x: str, y: str) -> float | None:
    xv, yv = _finite_pair(frame, x, y)
    return _spearman(xv, yv) if len(xv) >= 2 else None


def attribution_metrics(frame: pd.DataFrame, *, include_realized_r: bool) -> dict[str, Any]:
    required = {"label", "model_score", *_COMPONENT_COLUMNS}
    if include_realized_r:
        required.add("r_multiple")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"11D metrics缺少欄位: {missing}")
    valid = frame.copy()
    numeric = ["label", "model_score", *_COMPONENT_COLUMNS]
    if include_realized_r:
        numeric.append("r_multiple")
    for column in numeric:
        valid[column] = pd.to_numeric(valid[column], errors="coerce")
    valid = valid[
        np.isfinite(valid[["model_score", *_COMPONENT_COLUMNS]].to_numpy(dtype=np.float64)).all(axis=1)
    ].copy()
    if include_realized_r:
        valid = valid[np.isfinite(valid["r_multiple"].to_numpy(dtype=np.float64))].copy()
    if valid.empty:
        return {"row_count": 0}

    correlations = {
        "score_vs_target": _correlation(valid, "model_score", "target_raw_r"),
        "score_vs_favorable_r": _correlation(valid, "model_score", "favorable_r"),
        "score_vs_adverse_r": _correlation(valid, "model_score", "adverse_r"),
        "score_vs_time_penalty_r": _correlation(valid, "model_score", "time_penalty_r"),
    }
    if include_realized_r:
        correlations.update({
            "target_vs_realized_r": _correlation(valid, "target_raw_r", "r_multiple"),
            "score_vs_realized_r": _correlation(valid, "model_score", "r_multiple"),
            "favorable_r_vs_realized_r": _correlation(valid, "favorable_r", "r_multiple"),
            "adverse_r_vs_realized_r": _correlation(valid, "adverse_r", "r_multiple"),
            "time_penalty_r_vs_realized_r": _correlation(valid, "time_penalty_r", "r_multiple"),
        })

    count = max(1, int(math.ceil(len(valid) * 0.10)))
    score_ordered = valid.sort_values("model_score", kind="mergesort")
    target_ordered = valid.sort_values("target_raw_r", kind="mergesort")
    deciles: dict[str, Any] = {}
    for prefix, subset in (
        ("top_score", score_ordered.tail(count)),
        ("bottom_score", score_ordered.head(count)),
        ("top_target", target_ordered.tail(count)),
        ("bottom_target", target_ordered.head(count)),
    ):
        deciles[prefix] = {
            column: float(pd.to_numeric(subset[column], errors="coerce").mean())
            for column in _COMPONENT_COLUMNS
        }
        if include_realized_r:
            deciles[prefix]["realized_r"] = float(pd.to_numeric(subset["r_multiple"], errors="coerce").mean())

    labels: dict[str, Any] = {}
    for label_value, label_name in ((LABEL_REJECT, "reject"), (LABEL_PASS, "pass")):
        subset = valid[valid["label"].astype(np.int64) == int(label_value)].copy()
        if subset.empty:
            labels[label_name] = {"row_count": 0}
            continue
        label_corr = {
            "score_vs_target": _correlation(subset, "model_score", "target_raw_r"),
            "score_vs_favorable_r": _correlation(subset, "model_score", "favorable_r"),
            "score_vs_adverse_r": _correlation(subset, "model_score", "adverse_r"),
            "score_vs_time_penalty_r": _correlation(subset, "model_score", "time_penalty_r"),
        }
        if include_realized_r:
            label_corr.update({
                "target_vs_realized_r": _correlation(subset, "target_raw_r", "r_multiple"),
                "score_vs_realized_r": _correlation(subset, "model_score", "r_multiple"),
                "favorable_r_vs_realized_r": _correlation(subset, "favorable_r", "r_multiple"),
                "adverse_r_vs_realized_r": _correlation(subset, "adverse_r", "r_multiple"),
                "time_penalty_r_vs_realized_r": _correlation(subset, "time_penalty_r", "r_multiple"),
            })
        means = {
            column: float(pd.to_numeric(subset[column], errors="coerce").mean())
            for column in ("model_score", *_COMPONENT_COLUMNS)
        }
        if include_realized_r:
            means["realized_r"] = float(pd.to_numeric(subset["r_multiple"], errors="coerce").mean())
        labels[label_name] = {
            "row_count": int(len(subset)),
            "share": float(len(subset) / len(valid)),
            "means": means,
            "correlations": label_corr,
        }

    return {
        "row_count": int(len(valid)),
        "label_counts": {
            "pass": int((valid["label"].astype(np.int64) == LABEL_PASS).sum()),
            "reject": int((valid["label"].astype(np.int64) == LABEL_REJECT).sum()),
        },
        "correlations": correlations,
        "deciles": deciles,
        "by_label": labels,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return f"{number:.{digits}f}" if math.isfinite(number) else "N/A"


def render_markdown(payload: dict[str, Any]) -> str:
    qualified = payload["qualified_candidates"]
    actual = payload["actual_trades"]
    q_corr = qualified.get("correlations") or {}
    a_corr = actual.get("correlations") or {}
    lines = [
        "# 11D Label-conditional Target Component Attribution Audit",
        "",
        "- Runtime：research-only；本輪不訓練、不選epoch、不調target或threshold。",
        "- 目的：確認11B學到11A target的哪一個成分，以及11A的actual-R關係是否只來自PASS／REJECT分離。",
        "",
        "## 1. Qualified candidates",
        "",
        f"- Rows：`{qualified.get('row_count', 0):,}`。",
        f"- Score↔Target：`{_fmt(q_corr.get('score_vs_target'))}`。",
        f"- Score↔Favorable R：`{_fmt(q_corr.get('score_vs_favorable_r'))}`。",
        f"- Score↔Adverse R：`{_fmt(q_corr.get('score_vs_adverse_r'))}`。",
        f"- Score↔Time penalty R：`{_fmt(q_corr.get('score_vs_time_penalty_r'))}`。",
        "",
        "## 2. Actual trades",
        "",
        f"- Rows：`{actual.get('row_count', 0):,}`。",
        f"- Target↔realized R：`{_fmt(a_corr.get('target_vs_realized_r'))}`。",
        f"- Score↔realized R：`{_fmt(a_corr.get('score_vs_realized_r'))}`。",
        f"- Favorable R↔realized R：`{_fmt(a_corr.get('favorable_r_vs_realized_r'))}`。",
        f"- Adverse R↔realized R：`{_fmt(a_corr.get('adverse_r_vs_realized_r'))}`。",
        f"- Time penalty R↔realized R：`{_fmt(a_corr.get('time_penalty_r_vs_realized_r'))}`。",
        "",
        "## 3. Label-conditional actual trades",
        "",
        "| Label | Rows | Target↔R | Score↔Target | Score↔R | Favorable↔R | Adverse↔R | Time↔R |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, title in (("pass", "PASS"), ("reject", "REJECT")):
        item = (actual.get("by_label") or {}).get(key) or {}
        corr = item.get("correlations") or {}
        lines.append(
            f"| {title} | {item.get('row_count', 0):,} | "
            f"{_fmt(corr.get('target_vs_realized_r'))} | "
            f"{_fmt(corr.get('score_vs_target'))} | "
            f"{_fmt(corr.get('score_vs_realized_r'))} | "
            f"{_fmt(corr.get('favorable_r_vs_realized_r'))} | "
            f"{_fmt(corr.get('adverse_r_vs_realized_r'))} | "
            f"{_fmt(corr.get('time_penalty_r_vs_realized_r'))} |"
        )
    lines += [
        "",
        "## 4. 判定邊界",
        "",
        "- 若Target↔R在PASS／REJECT內均接近0，則11A的經濟關係主要只是二元分離，停止連續排序線。",
        "- 若Target↔R在PASS內仍明顯正向、但Score↔Target或Score↔R在PASS內失效，才可規劃conditional magnitude head；不得直接重跑11B超參數。",
        "- 若單一Target成分與R最相關而Score沒有捕捉，下一步只允許固定成分target audit，不直接部署或調OOS參數。",
        "- OOS為迭代研究證據；所有統計只作失敗歸因，不參與任何擬合。",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    scores, qualified, actual, report_11c = load_11d_input_frames(
        filter_id=args.filter_id,
        ranker_profile=args.ranker_profile,
    )
    required_score = {"ticker", "date", "group_index", "label", "split", "target_raw_r", "model_score"}
    missing = sorted(required_score - set(scores.columns))
    if missing:
        raise ValueError(f"11D 11B scores缺少欄位: {missing}")
    scores["ticker"] = scores["ticker"].astype(str)
    scores["date"] = pd.to_datetime(scores["date"], errors="raise").dt.strftime("%Y-%m-%d")
    oos_scores = scores[scores["split"].astype(str) == "oos"].copy()
    if bool(oos_scores.duplicated(["ticker", "date"]).any()):
        raise ValueError("11D OOS scores ticker/date必須唯一")

    manifest, arrays = load_validated_continuous_target_component_arrays(
        PROJECT_ROOT,
        args.filter_id,
        target_id=STRATEGY_ALIGNED_TARGET_ID,
    )
    target_contract = manifest.get("target_contract")
    if not isinstance(target_contract, dict):
        raise ValueError("11D continuous target manifest缺少target_contract")

    qualified_frame = attach_target_components(
        qualified,
        arrays=arrays,
        target_contract=target_contract,
    )

    actual_work = actual.copy()
    actual_work["ticker"] = actual_work["ticker"].astype(str)
    actual_work["target_date"] = pd.to_datetime(
        actual_work["target_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")
    lookup = oos_scores.rename(columns={"date": "target_date"})[
        ["ticker", "target_date", "group_index", "label", "target_raw_r", "model_score"]
    ]
    for column in ("group_index", "label", "target_raw_r", "model_score"):
        if column in actual_work.columns:
            actual_work = actual_work.drop(columns=[column])
    actual_work = actual_work.merge(
        lookup,
        how="left",
        on=["ticker", "target_date"],
        validate="many_to_one",
    )
    actual_work["r_multiple"] = pd.to_numeric(actual_work["r_multiple"], errors="coerce")
    actual_work = actual_work[
        np.isfinite(actual_work["r_multiple"].to_numpy(dtype=np.float64))
        & np.isfinite(pd.to_numeric(actual_work["model_score"], errors="coerce").to_numpy(dtype=np.float64))
    ].copy()
    actual_frame = attach_target_components(
        actual_work,
        arrays=arrays,
        target_contract=target_contract,
    )

    qualified_metrics = attribution_metrics(qualified_frame, include_realized_r=False)
    actual_metrics = attribution_metrics(actual_frame, include_realized_r=True)
    output_dir = _ranker_dir(args.filter_id, args.ranker_profile) / AUDIT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    qualified_path = output_dir / QUALIFIED_ATTRIBUTION_FILENAME
    actual_path = output_dir / ACTUAL_ATTRIBUTION_FILENAME
    json_path = output_dir / AUDIT_JSON_FILENAME
    markdown_path = output_dir / AUDIT_MARKDOWN_FILENAME
    qualified_frame.to_csv(qualified_path, index=False, encoding="utf-8-sig")
    actual_frame.to_csv(actual_path, index=False, encoding="utf-8-sig")

    payload = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "experiment": "11D Label-conditional Target Component Attribution Audit",
        "status": "RESULT_AVAILABLE_PENDING_REVIEW",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "filter_id": args.filter_id,
        "model_architecture": BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
        "ranker_profile": args.ranker_profile,
        "continuous_target_id": STRATEGY_ALIGNED_TARGET_ID,
        "qualified_candidates": qualified_metrics,
        "actual_trades": actual_metrics,
        "source_11c": {
            "status": report_11c.get("status"),
            "generated_at_utc": report_11c.get("generated_at_utc"),
            "score_vs_target_all_oos": ((report_11c.get("layers") or {}).get("all_oos_breakouts") or {}).get("global_spearman_score_vs_target"),
            "score_vs_target_qualified": ((report_11c.get("layers") or {}).get("qualified_candidates") or {}).get("global_spearman_score_vs_target"),
            "target_vs_realized_r": (report_11c.get("actual_trade_alignment") or {}).get("spearman_target_vs_realized_r"),
            "score_vs_realized_r": (report_11c.get("actual_trade_alignment") or {}).get("spearman_score_vs_realized_r"),
        },
        "artifacts": {
            "qualified_attribution": {"path": str(qualified_path), "sha256": _sha256_file(qualified_path)},
            "actual_trade_attribution": {"path": str(actual_path), "sha256": _sha256_file(actual_path)},
        },
        "interpretation_contract": {
            "research_only": True,
            "training_performed": False,
            "oos_iterative_research_evidence_only": True,
            "no_target_loss_sampling_epoch_threshold_or_runtime_change": True,
            "conditional_model_not_authorized_until_result_review": True,
        },
        "elapsed_sec": float(time.perf_counter() - started),
    }
    write_json(json_path, payload)
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")

    q = qualified_metrics.get("correlations") or {}
    a = actual_metrics.get("correlations") or {}
    print("11D target component attribution audit完成")
    print(
        f"qualified={qualified_metrics.get('row_count', 0):,} "
        f"actual_trades={actual_metrics.get('row_count', 0):,}"
    )
    print(
        "- Qualified Score↔components: "
        f"target={_fmt(q.get('score_vs_target'))} "
        f"favorable={_fmt(q.get('score_vs_favorable_r'))} "
        f"adverse={_fmt(q.get('score_vs_adverse_r'))} "
        f"time={_fmt(q.get('score_vs_time_penalty_r'))}"
    )
    print(
        "- Actual components↔R: "
        f"target={_fmt(a.get('target_vs_realized_r'))} "
        f"favorable={_fmt(a.get('favorable_r_vs_realized_r'))} "
        f"adverse={_fmt(a.get('adverse_r_vs_realized_r'))} "
        f"time={_fmt(a.get('time_penalty_r_vs_realized_r'))}"
    )
    print(f"已輸出: {markdown_path}")
    print(f"已輸出: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
