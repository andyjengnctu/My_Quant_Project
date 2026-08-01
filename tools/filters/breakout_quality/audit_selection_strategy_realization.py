"""11I research-only nested Selection strategy-realization coverage audit."""

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

from config.breakout_quality import (
    BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD,
)
from config.training_policy import OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT
from core.active_param_ensemble import get_active_param_ensemble_date_range
from core.dataset_profiles import get_dataset_dir, normalize_dataset_profile_key
from core.rolling_oos_params import get_active_param_date_range
from filters.breakout_quality.continuous_target import (
    STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
    load_validated_continuous_target_arrays,
)
from filters.breakout_quality.contract import LABEL_PASS, LABEL_REJECT
from filters.breakout_quality.paths import resolve_filter_output_dir
from tools.filters.breakout_quality.common import PROJECT_ROOT, load_validated_dataset_bundle, write_json
from tools.filters.breakout_quality.strategy_compare import (
    COMPARISON_MODE_HARD_FILTER,
    _build_controlled_param_source_pair,
    _flatten_candidate_replay_rows,
    _load_param_source,
    _run_scenario,
    _scenario_summary,
)
from tools.filters.breakout_quality.trade_attribution import reconstruct_round_trips
from tools.filters.breakout_quality.train_continuous_ranker import _spearman

AUDIT_SCHEMA_VERSION = 1
EXPERIMENT_NAME = "11I Nested Selection Strategy-realization Coverage Audit"
AUDIT_DIRNAME = "selection_strategy_realization_audit"
AUDIT_JSON_FILENAME = "selection_strategy_realization_audit.json"
AUDIT_MARKDOWN_FILENAME = "selection_strategy_realization_audit.md"
QUALIFIED_FILENAME = "selection_qualified_candidates.csv"
ORDERABLE_FILENAME = "selection_orderable_candidates.csv"
ROUND_TRIPS_FILENAME = "selection_round_trips.csv"
TRADE_MATCHES_FILENAME = "selection_strategy_realization_matches.csv"
PREPARE_SCRIPT_FILENAME = "prepare_selection_nested_roos.ps1"
DEFAULT_START_DATE = "2014-01-01"
DEFAULT_REPLAY_END_DATE = "2020-11-05"
DEFAULT_NESTED_OOS_END_DATE = "2020-12-31"
DEFAULT_TRAIN_WINDOW_MONTHS = 120
DEFAULT_OOS_MONTHS = 12


def default_research_models_dir() -> Path:
    return PROJECT_ROOT / "models" / "research" / "breakout_quality" / "selection_strategy_realization"


def default_params_path() -> Path:
    return default_research_models_dir() / "roos_base_finalists_agree.json"


def _output_dir(filter_id: str) -> Path:
    return (
        resolve_filter_output_dir(PROJECT_ROOT, filter_id=filter_id)
        / "continuous_targets"
        / STRATEGY_ALIGNED_NO_TIME_TARGET_ID
        / AUDIT_DIRNAME
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "11I research-only nested Selection strategy-realization coverage audit；"
            "先使用隔離的2014～2020 rolling OOS params，再重播canonical no-filter策略"
        )
    )
    parser.add_argument("--filter-id", default=BREAKOUT_QUALITY_DEFAULT_FILTER_ID)
    parser.add_argument("--params", default=str(default_params_path()))
    parser.add_argument("--dataset", default="full", choices=("full", "reduced"))
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_REPLAY_END_DATE)
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--rotation", choices=("on", "off"), default="off")
    parser.add_argument(
        "--optimizer-trials",
        type=int,
        default=OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT,
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_prepare_script(*, output_dir: Path, trials: int, dataset_profile: str = "full") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    research_models = default_research_models_dir().relative_to(PROJECT_ROOT)
    lines = [
        "$ErrorActionPreference = 'Stop'",
        f"$env:V16_MODELS_DIR = '{str(research_models).replace('/', '\\')}'",
        "try {",
        f"  python apps/ml_optimizer.py --outer-oos --dataset {str(dataset_profile)} "
        f"--trials {int(trials)} --outer-first-oos-date {DEFAULT_START_DATE} "
        f"--outer-last-oos-date {DEFAULT_NESTED_OOS_END_DATE} "
        f"--outer-train-window-months {DEFAULT_TRAIN_WINDOW_MONTHS} "
        f"--outer-oos-months {DEFAULT_OOS_MONTHS} --yes",
        "} finally {",
        "  Remove-Item Env:V16_MODELS_DIR -ErrorAction SilentlyContinue",
        "}",
        "",
    ]
    path = output_dir / PREPARE_SCRIPT_FILENAME
    path.write_text("\n".join(lines), encoding="utf-8-sig")
    return path


def _date_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return pd.Timestamp(text).strftime("%Y-%m-%d")


def _validate_param_coverage(source: dict[str, Any], *, start_date: str, end_date: str) -> tuple[str, str, str]:
    kind = str(source.get("kind") or "")
    payload = source.get("payload") or {}
    if kind == "rolling_active_param_ensemble":
        first, last = get_active_param_ensemble_date_range(payload)
    elif kind == "rolling_oos_param_schedule":
        first, last = get_active_param_date_range(payload)
    else:
        raise ValueError(f"11I只接受rolling lookahead-safe params，收到: {kind}")
    if _date_text(first) > _date_text(start_date) or _date_text(last) < _date_text(end_date):
        raise ValueError(
            "11I nested params未完整覆蓋Selection replay期間: "
            f"params={first}~{last}, requested={start_date}~{end_date}"
        )
    return kind, _date_text(first), _date_text(last)


def _validate_replay_target_bounds(
    target_manifest: dict[str, Any],
    *,
    start_date: str,
    end_date: str,
) -> tuple[str, str]:
    split_report = dict(target_manifest.get("split_report") or {})
    eligible = dict(split_report.get("final_refit_date_range") or {})
    eligible_start = _date_text(eligible.get("start"))
    eligible_end = _date_text(eligible.get("end"))
    if not eligible_start or not eligible_end:
        raise ValueError("11I No-time target manifest缺少final_refit_date_range")
    if start_date < eligible_start or end_date > eligible_end:
        raise ValueError(
            "11I replay超出11F Selection可評分事件邊界: "
            f"eligible={eligible_start}~{eligible_end}, requested={start_date}~{end_date}"
        )
    return eligible_start, eligible_end


def _resolve_target_date(frame: pd.DataFrame) -> pd.Series:
    signal = frame.get("signal_date", pd.Series("", index=frame.index)).fillna("").astype(str).str.strip()
    candidate = frame.get("candidate_date", pd.Series("", index=frame.index)).fillna("").astype(str).str.strip()
    entry = frame.get("entry_date", pd.Series("", index=frame.index)).fillna("").astype(str).str.strip()
    trade = frame.get("trade_date", pd.Series("", index=frame.index)).fillna("").astype(str).str.strip()
    target = signal.where(signal != "", candidate)
    target = target.where(target != "", entry)
    target = target.where(target != "", trade)
    return pd.to_datetime(target, errors="coerce").dt.strftime("%Y-%m-%d").fillna("")


def _target_lookup(filter_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    summary, _bank, labels, event_group_index, events = load_validated_dataset_bundle(filter_id)
    group_count = int(summary.get("feature_group_count", 0) or len(np.unique(event_group_index)))
    manifest, target, valid = load_validated_continuous_target_arrays(
        PROJECT_ROOT,
        filter_id,
        target_id=STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
        expected_group_count=group_count,
        expected_dataset_policy=summary.get("policy"),
    )
    frame = events[["ticker", "date", "group_index", "label"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.strftime("%Y-%m-%d")
    frame["group_index"] = pd.to_numeric(frame["group_index"], errors="raise").astype(np.int64)
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(np.int64)
    mixed = frame.groupby("group_index", sort=False)["label"].nunique()
    if bool((mixed != 1).any()):
        raise ValueError("11I dataset同group存在混合label")
    group = frame.drop_duplicates("group_index", keep="first").sort_values("group_index", kind="mergesort")
    if len(group) != len(target):
        raise ValueError("11I group lookup與No-time target長度不一致")
    ids = group["group_index"].to_numpy(dtype=np.int64)
    group["target_raw_r"] = np.asarray(target, dtype=np.float64)[ids]
    group["target_valid"] = np.asarray(valid, dtype=bool)[ids]
    return group.rename(columns={"date": "target_date"}).reset_index(drop=True), manifest


def _attach_targets(frame: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["ticker"] = work["ticker"].astype(str)
    work["target_date"] = _resolve_target_date(work)
    merged = work.merge(
        lookup[["ticker", "target_date", "group_index", "label", "target_raw_r", "target_valid"]],
        how="left",
        on=["ticker", "target_date"],
        validate="many_to_one",
    )
    merged["target_match"] = merged["target_valid"].fillna(False).astype(bool) & np.isfinite(
        pd.to_numeric(merged["target_raw_r"], errors="coerce")
    )
    return merged


def _unique_signals(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return frame.sort_values(
        ["ticker", "target_date", "trade_date", "candidate_type"], kind="mergesort"
    ).drop_duplicates(["ticker", "target_date"], keep="first").reset_index(drop=True)


def _trade_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    valid = frame[
        frame["target_match"].astype(bool)
        & np.isfinite(pd.to_numeric(frame["r_multiple"], errors="coerce"))
    ].copy()
    result: dict[str, Any] = {
        "trade_count": int(len(frame)),
        "matched_trade_count": int(len(valid)),
        "coverage_rate": float(len(valid) / len(frame)) if len(frame) else None,
    }
    if valid.empty:
        return result
    target = valid["target_raw_r"].to_numpy(dtype=np.float64)
    r = valid["r_multiple"].to_numpy(dtype=np.float64)
    count = max(1, int(math.ceil(len(valid) * 0.10)))
    ordered = valid.sort_values("target_raw_r", kind="mergesort")
    result.update({
        "spearman_target_vs_realized_r": _spearman(target, r),
        "top_target_decile_average_r": float(ordered.tail(count)["r_multiple"].mean()),
        "bottom_target_decile_average_r": float(ordered.head(count)["r_multiple"].mean()),
        "label_conditional": {},
    })
    for name, value in (("PASS", LABEL_PASS), ("REJECT", LABEL_REJECT)):
        sub = valid[valid["label"] == value]
        result["label_conditional"][name] = {
            "rows": int(len(sub)),
            "spearman_target_vs_realized_r": (
                _spearman(sub["target_raw_r"].to_numpy(dtype=np.float64), sub["r_multiple"].to_numpy(dtype=np.float64))
                if len(sub) >= 2 else None
            ),
        }
    return result


def _render_markdown(payload: dict[str, Any]) -> str:
    def fmt(v, d=4):
        return "-" if v is None else f"{float(v):.{d}f}"
    cov = payload["coverage"]
    tm = payload["trade_metrics"]
    return "\n".join([
        "# 11I Nested Selection Strategy-realization Coverage Audit",
        "",
        "- Runtime：research-only；使用Selection內nested rolling OOS params重播canonical no-filter策略。",
        "- 本輪不訓練、不建立Target arrays、不調optimizer、loss、threshold或runtime。",
        f"- Replay：`{payload['period']['start']}` ～ `{payload['period']['end']}`。",
        f"- Params：`{payload['params']['path']}`。",
        "",
        "## 1. Coverage",
        "",
        f"- Qualified unique signals：`{cov['qualified_unique_signal_count']:,}`；No-time Target match `{cov['qualified_target_match_rate']:.2%}`。",
        f"- Orderable unique signals：`{cov['orderable_unique_signal_count']:,}`；No-time Target match `{cov['orderable_target_match_rate']:.2%}`。",
        f"- Actual round trips：`{tm['trade_count']:,}`；Target matched `{tm['matched_trade_count']:,}`。",
        f"- Actual trade coverage vs qualified：`{cov['actual_trade_coverage_vs_qualified']:.2%}`。",
        "",
        "## 2. Strategy realization",
        "",
        f"- No-time Target↔realized R：`{fmt(tm.get('spearman_target_vs_realized_r'))}`。",
        f"- Target top／bottom decile R：`{fmt(tm.get('top_target_decile_average_r'))}`／`{fmt(tm.get('bottom_target_decile_average_r'))}`。",
        f"- PASS內 Target↔R：`{fmt((tm.get('label_conditional') or {}).get('PASS', {}).get('spearman_target_vs_realized_r'))}`。",
        "",
        "## 3. Boundary",
        "",
        "- Actual portfolio trades受持倉上限、資金鎖定與候選競爭影響，只能衡量strategy-realization coverage，不能直接替未成交qualified candidates填入R。",
        "- 若actual trade coverage不足，下一步必須建立canonical per-candidate counterfactual execution contract；不得把未交易候選標成0R。",
        "- 本結果只用Selection nested OOS；2021～2026正式OOS不參與任何target建立或參數選擇。",
        "",
    ])


def main(argv=None) -> int:
    args = parse_args(argv)
    if int(args.max_positions) < 1 or int(args.optimizer_trials) < 1:
        raise ValueError("max-positions與optimizer-trials必須>=1")
    start_date = _date_text(args.start_date)
    end_date = _date_text(args.end_date)
    if not start_date or not end_date or start_date > end_date:
        raise ValueError("11I start/end date不合法")
    output_dir = _output_dir(str(args.filter_id))
    dataset_profile = normalize_dataset_profile_key(str(args.dataset))
    prepare_script = _write_prepare_script(
        output_dir=output_dir,
        trials=int(args.optimizer_trials),
        dataset_profile=dataset_profile,
    )
    params_path = Path(args.params).expanduser()
    if not params_path.is_absolute():
        params_path = (PROJECT_ROOT / params_path).resolve()
    if bool(args.prepare_only):
        print(f"已輸出11I nested ROOS準備腳本: {prepare_script}")
        return 0
    if not params_path.is_file():
        raise FileNotFoundError(
            f"11I找不到nested Selection params: {params_path}；請先執行 {prepare_script}"
        )

    started = time.perf_counter()
    source = _load_param_source(params_path)
    kind, param_first, param_last = _validate_param_coverage(source, start_date=start_date, end_date=end_date)
    (
        param_source_kind,
        no_filter_params,
        _quality_params,
        no_filter_payload,
        _quality_payload,
        _ensemble_policy,
    ) = _build_controlled_param_source_pair(
        source,
        filter_id=str(args.filter_id),
        threshold=float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD),
        fixed_risk=None,
        comparison_mode=COMPARISON_MODE_HARD_FILTER,
    )
    if param_source_kind != kind:
        raise ValueError("11I param source kind在rewrite前後不一致")
    data_dir = Path(get_dataset_dir(str(PROJECT_ROOT), dataset_profile)).resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"11I找不到dataset: {data_dir}")

    lookup, target_manifest = _target_lookup(str(args.filter_id))
    eligible_start, eligible_end = _validate_replay_target_bounds(
        target_manifest,
        start_date=start_date,
        end_date=end_date,
    )

    replay_counts: dict[str, dict[str, Any]] = {}
    scenario = _run_scenario(
        name="11I_selection_nested_realization",
        data_dir=data_dir,
        param_source_kind=param_source_kind,
        params=no_filter_params,
        start_date=start_date,
        end_date=end_date,
        max_positions=int(args.max_positions),
        enable_rotation=str(args.rotation) == "on",
        quiet=bool(args.quiet),
        replay_counts=replay_counts,
    )
    qualified = _flatten_candidate_replay_rows(replay_counts, "candidate_rows")
    orderable = _flatten_candidate_replay_rows(replay_counts, "orderable_rows")
    round_trips = reconstruct_round_trips(pd.DataFrame(scenario["trade_history"]), scenario="11I_selection_nested_realization")
    qualified_m = _attach_targets(qualified, lookup)
    orderable_m = _attach_targets(orderable, lookup)
    trades_m = _attach_targets(round_trips, lookup)
    trades_m["r_multiple"] = pd.to_numeric(trades_m["r_multiple"], errors="coerce")
    qualified_u = _unique_signals(qualified_m)
    orderable_u = _unique_signals(orderable_m)
    trade_metrics = _trade_metrics(trades_m)
    q_match = int(qualified_u["target_match"].sum()) if len(qualified_u) else 0
    o_match = int(orderable_u["target_match"].sum()) if len(orderable_u) else 0
    coverage = {
        "qualified_occurrence_count": int(len(qualified_m)),
        "qualified_unique_signal_count": int(len(qualified_u)),
        "qualified_target_match_rate": float(q_match / len(qualified_u)) if len(qualified_u) else 0.0,
        "orderable_occurrence_count": int(len(orderable_m)),
        "orderable_unique_signal_count": int(len(orderable_u)),
        "orderable_target_match_rate": float(o_match / len(orderable_u)) if len(orderable_u) else 0.0,
        "actual_trade_coverage_vs_qualified": (
            float(trade_metrics["matched_trade_count"] / q_match) if q_match else 0.0
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    qualified_m.to_csv(output_dir / QUALIFIED_FILENAME, index=False, encoding="utf-8-sig")
    orderable_m.to_csv(output_dir / ORDERABLE_FILENAME, index=False, encoding="utf-8-sig")
    round_trips.to_csv(output_dir / ROUND_TRIPS_FILENAME, index=False, encoding="utf-8-sig")
    trades_m.to_csv(output_dir / TRADE_MATCHES_FILENAME, index=False, encoding="utf-8-sig")
    artifacts = {}
    for key, filename in (
        ("qualified", QUALIFIED_FILENAME),
        ("orderable", ORDERABLE_FILENAME),
        ("round_trips", ROUND_TRIPS_FILENAME),
        ("trade_matches", TRADE_MATCHES_FILENAME),
    ):
        path = output_dir / filename
        artifacts[key] = {"filename": filename, "sha256": _sha256_file(path), "size_bytes": int(path.stat().st_size)}
    payload = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "experiment": EXPERIMENT_NAME,
        "status": "RESULT_AVAILABLE_PENDING_REVIEW",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": float(time.perf_counter() - started),
        "filter_id": str(args.filter_id),
        "period": {"start": start_date, "end": end_date},
        "params": {
            "path": str(params_path),
            "sha256": _sha256_file(params_path),
            "source_kind": param_source_kind,
            "coverage_start": param_first,
            "coverage_end": param_last,
            "no_filter_payload_type": str((no_filter_payload or {}).get("type") or ""),
        },
        "dataset": {"profile": dataset_profile, "path": str(data_dir)},
        "strategy": {
            "max_positions": int(args.max_positions),
            "rotation": str(args.rotation),
            "summary": _scenario_summary(scenario),
        },
        "target": {
            "target_id": STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
            "manifest_generated_at_utc": target_manifest.get("generated_at_utc"),
            "selection_evaluable_date_range": {
                "start": eligible_start,
                "end": eligible_end,
            },
        },
        "coverage": coverage,
        "trade_metrics": trade_metrics,
        "artifacts": artifacts,
        "training_performed": False,
        "runtime_eligible": False,
        "interpretation_contract": {
            "selection_nested_oos_only": True,
            "official_2021_2026_oos_used": False,
            "untraded_candidates_are_not_labeled_zero": True,
            "audit_does_not_authorize_training": True,
        },
    }
    json_path = output_dir / AUDIT_JSON_FILENAME
    md_path = output_dir / AUDIT_MARKDOWN_FILENAME
    write_json(json_path, payload)
    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    print("11I nested Selection strategy-realization audit完成")
    print(
        f"qualified={len(qualified_u):,} orderable={len(orderable_u):,} "
        f"trades={trade_metrics['matched_trade_count']:,}/{trade_metrics['trade_count']:,}"
    )
    print(
        "- Target↔strategy R: "
        f"overall={trade_metrics.get('spearman_target_vs_realized_r')} "
        f"pass={(trade_metrics.get('label_conditional') or {}).get('PASS', {}).get('spearman_target_vs_realized_r')}"
    )
    print(f"已輸出: {md_path}")
    print(f"已輸出: {json_path}")
    return 0


__all__ = [
    "AUDIT_JSON_FILENAME",
    "AUDIT_MARKDOWN_FILENAME",
    "PREPARE_SCRIPT_FILENAME",
    "default_params_path",
    "default_research_models_dir",
    "_validate_replay_target_bounds",
    "main",
    "parse_args",
]
