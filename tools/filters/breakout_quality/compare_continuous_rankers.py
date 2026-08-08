"""Read-only paired quality comparison for config-selected continuous rankers."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.breakout_quality import (
    BREAKOUT_QUALITY_CONTINUOUS_RANKER_REPORT_BOUNDARY_WIDTH,
    BREAKOUT_QUALITY_CONTINUOUS_RANKER_REPORT_TOP_K,
    BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
    get_breakout_quality_continuous_ranker_comparison_settings,
)
from config.strategy_compare import get_strategy_comparison_settings
from core.console_report import (
    compact_console_enabled,
    print_artifact_paths,
    project_relative_display_path,
    render_key_values,
    render_section,
    render_table,
    render_title,
)
from filters.breakout_quality.continuous_ranker_quality import (
    daily_top_k_metrics,
    exact_random_top_k_baseline,
)
from filters.breakout_quality.csv_io import read_breakout_quality_csv
from filters.breakout_quality.paths import resolve_filter_output_dir
from filters.breakout_quality.ranking_score_store import load_continuous_ranker_oos_contract
from filters.breakout_quality.workflow_io import PROJECT_ROOT

SCHEMA_VERSION = 1
REPORT_JSON_FILENAME = "continuous_ranker_comparison.json"
REPORT_MARKDOWN_FILENAME = "continuous_ranker_comparison.md"
OUTPUT_DIRNAME = "continuous_ranker_comparison"

_METRIC_KEYS = (
    "ndcg_at_k",
    "top_k_raw_target_mean",
    "top_k_raw_target_lift",
    "oracle_top_k_overlap",
    "boundary_concordance",
    "boundary_raw_target_gap",
)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "只讀config設定的continuous-ranker frozen scores，做同日paired Top-K/"
            "random baseline/Dynamic-K selector-boundary模型品質比較；不重訓、不重跑策略。"
        )
    )
    parser.add_argument("--filter-id", default=BREAKOUT_QUALITY_DEFAULT_FILTER_ID)
    parser.add_argument("--model-architecture", default=BREAKOUT_QUALITY_MODEL_ARCHITECTURE)
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"無法讀取JSON: {path}; {type(exc).__name__}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON根節點必須是object: {path}")
    return payload


def _json_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_native(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _model_specs() -> tuple[tuple[str, str], ...]:
    return get_breakout_quality_continuous_ranker_comparison_settings().model_profiles


def _load_model_frame(
    *,
    root: Path,
    filter_id: str,
    architecture: str,
    model_id: str,
    profile: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    contract = load_continuous_ranker_oos_contract(
        str(root), filter_id, architecture, profile
    )
    frame = read_breakout_quality_csv(contract.score_path).copy()
    required = {
        "ticker",
        "date",
        "group_index",
        "split",
        "target_raw_r",
        "target_daily_percentile",
        "model_score",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{model_id} continuous ranker score缺少欄位: {missing}")
    frame = frame[frame["split"].astype(str).isin(["selection", "oos"])].copy()
    frame["ticker"] = frame["ticker"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.strftime("%Y-%m-%d")
    frame["group_index"] = pd.to_numeric(frame["group_index"], errors="raise").astype(int)
    for column in ("target_raw_r", "target_daily_percentile", "model_score"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
        if not np.isfinite(frame[column].to_numpy(dtype=np.float64)).all():
            raise ValueError(f"{model_id} {column}含非有限值")
    if frame.duplicated(["ticker", "date", "group_index", "split"]).any():
        raise ValueError(f"{model_id} continuous ranker score row identity重複")
    return frame, {
        "model_id": model_id,
        "profile": profile,
        "continuous_target_id": contract.continuous_target_id,
        "score_path": project_relative_display_path(contract.score_path, project_root=root),
        "report_path": project_relative_display_path(contract.report_path, project_root=root),
        "model_information_cutoff": contract.model_information_cutoff,
        "available_from": contract.available_from,
        "available_through": contract.available_through,
    }


def _paired_score_frame(
    model_frames: dict[str, pd.DataFrame],
    *,
    split: str,
) -> pd.DataFrame:
    model_ids = tuple(model_frames)
    reference_id = model_ids[0]
    keys = ["ticker", "date", "group_index", "split"]
    ref = model_frames[reference_id]
    ref = ref[ref["split"].astype(str) == split].copy()
    ref = ref[
        keys + ["target_raw_r", "target_daily_percentile", "model_score"]
    ].rename(columns={"model_score": f"score__{reference_id}"})
    expected_keys = set(map(tuple, ref[keys].itertuples(index=False, name=None)))
    for model_id in model_ids[1:]:
        other = model_frames[model_id]
        other = other[other["split"].astype(str) == split].copy()
        actual_keys = set(map(tuple, other[keys].itertuples(index=False, name=None)))
        if actual_keys != expected_keys:
            missing = len(expected_keys - actual_keys)
            extra = len(actual_keys - expected_keys)
            raise ValueError(
                f"{split} paired model candidate universe不一致: "
                f"reference={reference_id}, model={model_id}, missing={missing}, extra={extra}"
            )
        target = other[keys + ["target_raw_r", "target_daily_percentile", "model_score"]].rename(
            columns={
                "target_raw_r": f"target_raw_r__{model_id}",
                "target_daily_percentile": f"target_daily_percentile__{model_id}",
                "model_score": f"score__{model_id}",
            }
        )
        ref = ref.merge(target, how="inner", on=keys, validate="one_to_one")
        raw_other = ref.pop(f"target_raw_r__{model_id}").to_numpy(dtype=np.float64)
        pct_other = ref.pop(f"target_daily_percentile__{model_id}").to_numpy(dtype=np.float64)
        if not np.allclose(
            ref["target_raw_r"].to_numpy(dtype=np.float64),
            raw_other,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"{split} {model_id} raw target與reference不一致")
        if not np.allclose(
            ref["target_daily_percentile"].to_numpy(dtype=np.float64),
            pct_other,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"{split} {model_id} percentile target與reference不一致")
    return ref.sort_values(["date", "ticker", "group_index"], kind="mergesort").reset_index(drop=True)


def _day_metrics(day: pd.DataFrame, *, score_column: str, top_k: int, boundary_width: int) -> dict[str, Any]:
    date_values = np.repeat(str(day["date"].iloc[0]), len(day))
    return daily_top_k_metrics(
        date_values,
        day[score_column].to_numpy(dtype=np.float64),
        day["target_raw_r"].to_numpy(dtype=np.float64),
        day["target_daily_percentile"].to_numpy(dtype=np.float64),
        top_k=int(top_k),
        boundary_width=int(boundary_width),
    )


def _mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def _aggregate_daily(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {"competition_date_count": int(len(rows))}
    for key in _METRIC_KEYS:
        result[key] = _mean([row.get(key) for row in rows])
    return result


def _paired_contrast(
    left_rows: dict[str, dict[str, Any]],
    right_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    common_dates = sorted(set(left_rows).intersection(right_rows))
    output: dict[str, Any] = {"paired_date_count": int(len(common_dates)), "metrics": {}}
    for key in _METRIC_KEYS:
        deltas = []
        wins = 0
        ties = 0
        for date in common_dates:
            left = left_rows[date].get(key)
            right = right_rows[date].get(key)
            if left is None or right is None:
                continue
            delta = float(left) - float(right)
            if not math.isfinite(delta):
                continue
            deltas.append(delta)
            if delta > 0.0:
                wins += 1
            elif delta == 0.0:
                ties += 1
        count = len(deltas)
        output["metrics"][key] = {
            "paired_date_count": int(count),
            "mean_delta": float(np.mean(deltas)) if deltas else None,
            "median_delta": float(np.median(deltas)) if deltas else None,
            "left_win_rate": float(wins / count) if count else None,
            "tie_rate": float(ties / count) if count else None,
        }
    return output


def _evaluate_paired_frame(
    frame: pd.DataFrame,
    *,
    model_ids: tuple[str, ...],
    k_by_date: dict[str, int],
    boundary_width: int,
) -> dict[str, Any]:
    model_daily: dict[str, dict[str, dict[str, Any]]] = {model_id: {} for model_id in model_ids}
    random_daily: dict[str, dict[str, Any]] = {}
    excluded_non_competition = 0
    for date, day in frame.groupby("date", sort=True):
        date_text = str(date)
        k = int(k_by_date.get(date_text, 0))
        if k < 1 or len(day) <= k:
            excluded_non_competition += 1
            continue
        for model_id in model_ids:
            metrics = _day_metrics(
                day,
                score_column=f"score__{model_id}",
                top_k=k,
                boundary_width=boundary_width,
            )
            model_daily[model_id][date_text] = {
                key: metrics.get(key) for key in _METRIC_KEYS
            }
        random_daily[date_text] = exact_random_top_k_baseline(
            day["target_daily_percentile"].to_numpy(dtype=np.float64),
            day["target_raw_r"].to_numpy(dtype=np.float64),
            top_k=k,
        )

    model_summary = {
        model_id: _aggregate_daily(list(model_daily[model_id].values()))
        for model_id in model_ids
    }
    random_summary = _aggregate_daily(list(random_daily.values()))
    contrasts: dict[str, Any] = {}
    for left_id, right_id in zip(model_ids[1:], model_ids[:-1]):
        contrasts[f"{left_id}_minus_{right_id}"] = _paired_contrast(
            model_daily[left_id],
            model_daily[right_id],
        )
    return {
        "candidate_row_count": int(len(frame)),
        "all_date_count": int(frame["date"].nunique()),
        "competition_date_count": int(len(random_daily)),
        "excluded_non_competition_date_count": int(excluded_non_competition),
        "models": model_summary,
        "random_baseline": random_summary,
        "paired_contrasts": contrasts,
        "daily": {
            "models": model_daily,
            "random_baseline": random_daily,
        },
    }


def _resolve_reference_pair_dir(
    *,
    root: Path,
    reference_arm_id: str,
) -> tuple[Path, Path, dict[str, Any]]:
    settings = get_strategy_comparison_settings()
    arm = settings.arms.get(reference_arm_id)
    if arm is None:
        raise ValueError(f"Dynamic-K reference arm不存在: {reference_arm_id}")
    runs_root = (root / settings.output_root / "runs").resolve()
    if not runs_root.is_dir():
        raise FileNotFoundError(
            f"找不到Strategy Compare runs: {project_relative_display_path(runs_root, project_root=root)}"
        )
    expected = {
        "param_source": arm.param_source,
        "rule_policy": arm.rule_policy,
        "dl_enabled": arm.dl_enabled,
        "dl_id": arm.dl_id,
        "dl_runtime_mode": arm.dl_runtime_mode,
    }
    for run_dir in sorted((path for path in runs_root.iterdir() if path.is_dir()), reverse=True):
        result_path = run_dir / "strategy_comparison.json"
        if not result_path.is_file():
            continue
        payload = _read_json(result_path)
        if str(payload.get("status") or "") != "COMPLETED":
            continue
        stored_arm = dict(((payload.get("settings") or {}).get("arms") or {}).get(reference_arm_id) or {})
        actual = {
            "param_source": stored_arm.get("param_source"),
            "rule_policy": stored_arm.get("rule_policy"),
            "dl_enabled": bool(stored_arm.get("dl_enabled")),
            "dl_id": stored_arm.get("dl_id"),
            "dl_runtime_mode": stored_arm.get("dl_runtime_mode"),
        }
        if actual != expected:
            continue
        pair_execution = dict((payload.get("pair_execution") or {}).get(reference_arm_id) or {})
        current_pair_dir = str(pair_execution.get("current_pair_dir") or "").strip()
        if not current_pair_dir:
            continue
        pair_dir = (root / Path(current_pair_dir)).resolve()
        required = (
            pair_dir / "score_ranking_daily_capacity.csv",
            pair_dir / "score_ranking_orderable_candidates.csv",
            pair_dir / "strategy_comparison.json",
        )
        if all(path.is_file() for path in required):
            return pair_dir, run_dir, payload
    raise FileNotFoundError(
        f"找不到符合目前設定 reference arm={reference_arm_id} 語意的已完成 Strategy Compare pair；"
        "請確認目前 config 指定的比較對象已有正式結果，且工件仍在 outputs/strategy_compare/runs/"
    )


def _dynamic_orderable_frame(
    *,
    pair_dir: Path,
    model_frames: dict[str, pd.DataFrame],
    model_ids: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    capacity = pd.read_csv(pair_dir / "score_ranking_daily_capacity.csv", encoding="utf-8-sig")
    required_orderable = {"ticker", "trade_date", "signal_date"}
    orderable = read_breakout_quality_csv(
        pair_dir / "score_ranking_orderable_candidates.csv",
        encoding="utf-8-sig",
        usecols=lambda column: column in required_orderable,
    )
    required_capacity = {
        "Date",
        "Resource_Aware_Max_DL_Eligible",
        "Resource_Aware_Pre_Market_Order_Limit",
    }
    missing = sorted(required_capacity - set(capacity.columns))
    if missing:
        raise ValueError(f"Dynamic-K capacity缺少欄位: {missing}")
    missing = sorted(required_orderable - set(orderable.columns))
    if missing:
        raise ValueError(f"Dynamic-K orderable candidates缺少欄位: {missing}")

    capacity["date"] = pd.to_datetime(capacity["Date"], errors="raise").dt.strftime("%Y-%m-%d")
    eligible_raw = capacity["Resource_Aware_Max_DL_Eligible"]
    if pd.api.types.is_bool_dtype(eligible_raw):
        eligible = eligible_raw.fillna(False).astype(bool)
    else:
        normalized = eligible_raw.fillna("").astype(str).str.strip().str.lower()
        invalid = ~normalized.isin({"true", "false", "1", "0", "yes", "no", "y", "n"})
        if bool(invalid.any()):
            sample = sorted(set(normalized.loc[invalid].head(8)))
            raise ValueError(f"Dynamic-K Max_DL_Eligible布林值無法解析: {sample}")
        eligible = normalized.isin({"true", "1", "yes", "y"})
    capacity["dynamic_k"] = pd.to_numeric(
        capacity["Resource_Aware_Pre_Market_Order_Limit"], errors="coerce"
    )
    capacity = capacity[eligible & capacity["dynamic_k"].notna()].copy()
    capacity["dynamic_k"] = capacity["dynamic_k"].astype(int)
    capacity = capacity[capacity["dynamic_k"] > 0].copy()
    if capacity["date"].duplicated().any():
        raise ValueError("Dynamic-K capacity同日期不可重複")
    k_by_date = dict(zip(capacity["date"], capacity["dynamic_k"]))

    orderable["ticker"] = orderable["ticker"].astype(str)
    orderable["date"] = pd.to_datetime(orderable["trade_date"], errors="raise").dt.strftime("%Y-%m-%d")
    orderable["signal_date"] = pd.to_datetime(
        orderable["signal_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    orderable = orderable[orderable["date"].isin(k_by_date)].copy()
    orderable = orderable[orderable["signal_date"].notna()].copy()
    duplicate = orderable.duplicated(["date", "ticker"], keep=False)
    if duplicate.any():
        sample = (
            orderable.loc[duplicate, ["date", "ticker", "signal_date"]]
            .head(8)
            .to_dict("records")
        )
        raise ValueError(
            "Dynamic-K orderable candidates同trade_date/ticker出現多筆，"
            f"無法建立唯一selector候選集合: sample={sample}"
        )

    result = orderable[["date", "ticker", "signal_date"]].copy()
    result["dynamic_k"] = result["date"].map(k_by_date).astype(int)

    target_reference_id = model_ids[0]
    for model_id in model_ids:
        score_frame = model_frames[model_id]
        score_frame = score_frame[score_frame["split"].astype(str) == "oos"].copy()
        lookup = score_frame[
            ["ticker", "date", "target_raw_r", "target_daily_percentile", "model_score"]
        ].rename(
            columns={
                "date": "signal_date",
                "target_raw_r": f"target_raw_r__{model_id}",
                "target_daily_percentile": f"target_daily_percentile__{model_id}",
                "model_score": f"score__{model_id}",
            }
        )
        lookup_duplicate = lookup.duplicated(["ticker", "signal_date"], keep=False)
        if bool(lookup_duplicate.any()):
            sample = (
                lookup.loc[lookup_duplicate, ["ticker", "signal_date"]]
                .head(8)
                .to_dict("records")
            )
            raise ValueError(
                f"Dynamic-K {model_id} frozen score同ticker/signal_date必須唯一: sample={sample}"
            )
        result = result.merge(
            lookup,
            how="left",
            on=["ticker", "signal_date"],
            validate="many_to_one",
        )

    score_columns = [f"score__{model_id}" for model_id in model_ids]
    target_columns = [f"target_raw_r__{model_id}" for model_id in model_ids]
    pct_columns = [f"target_daily_percentile__{model_id}" for model_id in model_ids]
    complete = result[score_columns + target_columns + pct_columns].notna().all(axis=1)
    result["all_model_score_available"] = complete
    per_day = result.groupby("date", sort=True)["all_model_score_available"].agg(["all", "size"])
    observed_dates = set(per_day.index.astype(str))
    full_coverage_dates = set(per_day.index[per_day["all"]].astype(str))
    partial_dates = set(per_day.index[~per_day["all"]].astype(str))
    missing_orderable_dates = set(k_by_date) - observed_dates
    paired = result[result["date"].isin(full_coverage_dates)].copy()

    ref_raw = f"target_raw_r__{target_reference_id}"
    ref_pct = f"target_daily_percentile__{target_reference_id}"
    paired["target_raw_r"] = pd.to_numeric(paired[ref_raw], errors="raise").astype(float)
    paired["target_daily_percentile"] = pd.to_numeric(paired[ref_pct], errors="raise").astype(float)
    for model_id in model_ids[1:]:
        if not np.allclose(
            paired["target_raw_r"].to_numpy(dtype=np.float64),
            paired[f"target_raw_r__{model_id}"].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"Dynamic-K {model_id} raw target與reference不一致")
        if not np.allclose(
            paired["target_daily_percentile"].to_numpy(dtype=np.float64),
            paired[f"target_daily_percentile__{model_id}"].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"Dynamic-K {model_id} percentile target與reference不一致")

    day_k = paired.groupby("date", sort=True)["dynamic_k"].nunique()
    if bool((day_k != 1).any()):
        raise ValueError("Dynamic-K同一交易日K不唯一")
    return paired, {
        "eligible_strategy_date_count": int(len(k_by_date)),
        "orderable_strategy_date_count": int(len(observed_dates)),
        "missing_orderable_date_count": int(len(missing_orderable_dates)),
        "orderable_candidate_row_count": int(len(result)),
        "full_score_coverage_date_count": int(len(full_coverage_dates)),
        "partial_score_coverage_date_count": int(len(partial_dates)),
        "full_score_coverage_rate": (
            float(len(full_coverage_dates) / len(k_by_date)) if k_by_date else None
        ),
        "dynamic_k_min": int(min(k_by_date.values())) if k_by_date else None,
        "dynamic_k_max": int(max(k_by_date.values())) if k_by_date else None,
        "dynamic_k_mean": float(np.mean(list(k_by_date.values()))) if k_by_date else None,
    }


def _render_metric_table(section: dict[str, Any], *, model_ids: tuple[str, ...]) -> str:
    random = dict(section.get("random_baseline") or {})
    rows = []
    for model_id in model_ids:
        row = dict((section.get("models") or {}).get(model_id) or {})
        rows.append(
            (
                model_id,
                _fmt(row.get("ndcg_at_k")),
                _fmt_delta(row.get("ndcg_at_k"), random.get("ndcg_at_k")),
                _fmt(row.get("top_k_raw_target_lift")),
                _fmt_pct(row.get("oracle_top_k_overlap")),
                _fmt_delta_pct(row.get("oracle_top_k_overlap"), random.get("oracle_top_k_overlap")),
                _fmt_pct(row.get("boundary_concordance")),
                _fmt_delta_pct(row.get("boundary_concordance"), 0.5),
                _fmt(row.get("boundary_raw_target_gap")),
            )
        )
    rows.append(
        (
            "Random",
            _fmt(random.get("ndcg_at_k")),
            "-",
            _fmt(random.get("top_k_raw_target_lift")),
            _fmt_pct(random.get("oracle_top_k_overlap")),
            "-",
            _fmt_pct(random.get("boundary_concordance")),
            "-",
            _fmt(random.get("boundary_raw_target_gap")),
        )
    )
    return render_table(
        (
            "Model",
            "NDCG",
            "vs Random",
            "Top-K Lift",
            "Overlap",
            "vs Random",
            "Boundary",
            "vs 50%",
            "Boundary gap",
        ),
        rows,
        alignments=("left", "right", "right", "right", "right", "right", "right", "right", "right"),
    )


def _render_paired_table(section: dict[str, Any]) -> str:
    rows = []
    metric_labels = (
        ("ndcg_at_k", "NDCG@K"),
        ("top_k_raw_target_lift", "Top-K Lift"),
        ("oracle_top_k_overlap", "Oracle overlap"),
        ("boundary_concordance", "Boundary"),
        ("boundary_raw_target_gap", "Boundary gap"),
    )
    contrasts = dict(section.get("paired_contrasts") or {})
    for contrast_id, contrast in contrasts.items():
        metrics = dict(contrast.get("metrics") or {})
        for key, label in metric_labels:
            row = dict(metrics.get(key) or {})
            rows.append(
                (
                    contrast_id,
                    label,
                    _fmt(row.get("mean_delta")),
                    _fmt_pct(row.get("left_win_rate")),
                    f"{int(row.get('paired_date_count', 0) or 0):,}",
                )
            )
    return render_table(
        ("Paired比較", "指標", "平均Δ", "左側較佳日", "同日數"),
        rows,
        alignments=("left", "left", "right", "right", "right"),
    )


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}" if math.isfinite(number) else "-"


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100.0:.2f}%"


def _fmt_delta(value: Any, baseline: Any) -> str:
    if value is None or baseline is None:
        return "-"
    return f"{float(value) - float(baseline):+.4f}"


def _fmt_delta_pct(value: Any, baseline: Any) -> str:
    if value is None or baseline is None:
        return "-"
    return f"{(float(value) - float(baseline)) * 100.0:+.2f}pp"


def render_console(payload: dict[str, Any]) -> str:
    model_ids = tuple(str(item["model_id"]) for item in payload.get("models") or [])
    fixed = dict(payload.get("fixed_k") or {})
    dynamic = dict(payload.get("dynamic_k") or {})
    lines = [
        render_title(f"{' / '.join(model_ids)} Paired 排序品質比較"),
        render_key_values(
            (
                ("固定 K", fixed.get("top_k")),
                ("Boundary width", fixed.get("boundary_width")),
                ("Random baseline", "精確期望值（非Monte Carlo）"),
                ("Dynamic-K reference", dynamic.get("reference_arm")),
                ("Dynamic-K來源", dynamic.get("pair_dir")),
                ("模型重訓", "否"),
                ("策略 replay", "否"),
            )
        ),
    ]
    for split in ("selection", "oos"):
        section = dict((fixed.get("splits") or {}).get(split) or {})
        lines.extend(
            (
                render_section(
                    f"Fixed K={fixed.get('top_k')}｜{split}｜同日共同候選"
                ),
                f"競爭日：{int(section.get('competition_date_count', 0) or 0):,}；"
                f"候選列：{int(section.get('candidate_row_count', 0) or 0):,}",
                _render_metric_table(section, model_ids=model_ids),
                "同日 paired 改善：",
                _render_paired_table(section),
            )
        )
    dyn_section = dict(dynamic.get("evaluation") or {})
    coverage = dict(dynamic.get("coverage") or {})
    lines.extend(
        (
            render_section(f"Dynamic-K｜實際 {dynamic.get('reference_arm')} selector 決策邊界"),
            render_key_values(
                (
                    (f"{dynamic.get('reference_arm')} max-DL eligible days", coverage.get("eligible_strategy_date_count")),
                    ("有orderable candidates日", coverage.get("orderable_strategy_date_count")),
                    (f"{len(model_ids)}模型完整score日", coverage.get("full_score_coverage_date_count")),
                    ("部分score日", coverage.get("partial_score_coverage_date_count")),
                    ("缺orderable日", coverage.get("missing_orderable_date_count")),
                    ("完整score coverage", _fmt_pct(coverage.get("full_score_coverage_rate"))),
                    ("Dynamic K", f"{coverage.get('dynamic_k_min')}～{coverage.get('dynamic_k_max')}；mean={_fmt(coverage.get('dynamic_k_mean'), 2)}"),
                    ("實際競爭日", dyn_section.get("competition_date_count")),
                )
            ),
            _render_metric_table(dyn_section, model_ids=model_ids),
            "同日 paired 改善：",
            _render_paired_table(dyn_section),
        )
    )
    return "\n".join(str(line) for line in lines if line)


def _render_markdown(payload: dict[str, Any]) -> str:
    model_ids = tuple(str(item["model_id"]) for item in payload.get("models") or [])
    fixed = dict(payload.get("fixed_k") or {})
    dynamic = dict(payload.get("dynamic_k") or {})
    model_label = " / ".join(model_ids)
    reference_arm = str(dynamic.get("reference_arm") or "-")
    lines = [
        f"# {model_label} Paired 排序品質比較",
        "",
        f"- 固定K：`{fixed.get('top_k')}`；boundary width：`{fixed.get('boundary_width')}`。",
        "- Random baseline使用每個交易日候選數與target分布的精確期望值，不使用Monte Carlo。",
        f"- Fixed-K只讀{len(model_ids)}個config設定模型的既有frozen score；候選row identity與Target必須完全一致才允許paired比較。",
        f"- Dynamic-K reference：`{dynamic.get('reference_arm')}`；來源：`{dynamic.get('pair_dir')}`。",
        f"- Dynamic-K只讀既有Strategy Compare orderable candidates與{reference_arm} max-DL pre-market order limit；不重跑策略。",
        "",
    ]
    for split in ("selection", "oos"):
        section = dict((fixed.get("splits") or {}).get(split) or {})
        lines += [
            f"## Fixed K={fixed.get('top_k')} — {split}",
            "",
            f"- competition days：`{section.get('competition_date_count')}`；candidate rows：`{section.get('candidate_row_count')}`。",
            "",
            "| Model | NDCG | NDCG-Random | Top-K Lift | Oracle overlap | Overlap-Random | Boundary | Boundary-50% | Boundary gap |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        random = dict(section.get("random_baseline") or {})
        for model_id in model_ids:
            row = dict((section.get("models") or {}).get(model_id) or {})
            lines.append(
                f"| {model_id} | {_fmt(row.get('ndcg_at_k'))} "
                f"| {_fmt_delta(row.get('ndcg_at_k'), random.get('ndcg_at_k'))} "
                f"| {_fmt(row.get('top_k_raw_target_lift'))} "
                f"| {_fmt_pct(row.get('oracle_top_k_overlap'))} "
                f"| {_fmt_delta_pct(row.get('oracle_top_k_overlap'), random.get('oracle_top_k_overlap'))} "
                f"| {_fmt_pct(row.get('boundary_concordance'))} "
                f"| {_fmt_delta_pct(row.get('boundary_concordance'), 0.5)} "
                f"| {_fmt(row.get('boundary_raw_target_gap'))} |"
            )
        lines += ["", "### 同日 paired", "", "| 比較 | 指標 | 平均Δ | 左側較佳日 | 同日數 |", "|---|---|---:|---:|---:|"]
        for contrast_id, contrast in (section.get("paired_contrasts") or {}).items():
            for key, label in (
                ("ndcg_at_k", "NDCG@K"),
                ("top_k_raw_target_lift", "Top-K Lift"),
                ("oracle_top_k_overlap", "Oracle overlap"),
                ("boundary_concordance", "Boundary"),
                ("boundary_raw_target_gap", "Boundary gap"),
            ):
                row = dict((contrast.get("metrics") or {}).get(key) or {})
                lines.append(
                    f"| {contrast_id} | {label} | {_fmt(row.get('mean_delta'))} "
                    f"| {_fmt_pct(row.get('left_win_rate'))} | {int(row.get('paired_date_count', 0) or 0):,} |"
                )
        lines.append("")

    dyn = dict(dynamic.get("evaluation") or {})
    cov = dict(dynamic.get("coverage") or {})
    lines += [
        f"## Dynamic-K — 實際 {reference_arm} selector boundary",
        "",
        f"- {reference_arm} max-DL eligible days：`{cov.get('eligible_strategy_date_count')}`；有orderable candidates日：`{cov.get('orderable_strategy_date_count')}`。",
        f"- {len(model_ids)}模型完整score日：`{cov.get('full_score_coverage_date_count')}`；部分score日：`{cov.get('partial_score_coverage_date_count')}`；缺orderable日：`{cov.get('missing_orderable_date_count')}`；coverage：`{_fmt_pct(cov.get('full_score_coverage_rate'))}`。",
        f"- Dynamic K：`{cov.get('dynamic_k_min')}～{cov.get('dynamic_k_max')}`；mean `{_fmt(cov.get('dynamic_k_mean'), 2)}`。",
        f"- paired competition days：`{dyn.get('competition_date_count')}`。",
        "",
        "| Model | NDCG | NDCG-Random | Top-K Lift | Oracle overlap | Overlap-Random | Boundary | Boundary-50% | Boundary gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    random = dict(dyn.get("random_baseline") or {})
    for model_id in model_ids:
        row = dict((dyn.get("models") or {}).get(model_id) or {})
        lines.append(
            f"| {model_id} | {_fmt(row.get('ndcg_at_k'))} "
            f"| {_fmt_delta(row.get('ndcg_at_k'), random.get('ndcg_at_k'))} "
            f"| {_fmt(row.get('top_k_raw_target_lift'))} "
            f"| {_fmt_pct(row.get('oracle_top_k_overlap'))} "
            f"| {_fmt_delta_pct(row.get('oracle_top_k_overlap'), random.get('oracle_top_k_overlap'))} "
            f"| {_fmt_pct(row.get('boundary_concordance'))} "
            f"| {_fmt_delta_pct(row.get('boundary_concordance'), 0.5)} "
            f"| {_fmt(row.get('boundary_raw_target_gap'))} |"
        )
    lines += ["", "### Dynamic-K 同日 paired", "", "| 比較 | 指標 | 平均Δ | 左側較佳日 | 同日數 |", "|---|---|---:|---:|---:|"]
    for contrast_id, contrast in (dyn.get("paired_contrasts") or {}).items():
        for key, label in (
            ("ndcg_at_k", "NDCG@K"),
            ("top_k_raw_target_lift", "Top-K Lift"),
            ("oracle_top_k_overlap", "Oracle overlap"),
            ("boundary_concordance", "Boundary"),
            ("boundary_raw_target_gap", "Boundary gap"),
        ):
            row = dict((contrast.get("metrics") or {}).get(key) or {})
            lines.append(
                f"| {contrast_id} | {label} | {_fmt(row.get('mean_delta'))} "
                f"| {_fmt_pct(row.get('left_win_rate'))} | {int(row.get('paired_date_count', 0) or 0):,} |"
            )
    lines += [
        "",
        "## 契約",
        "",
        "- 所有比較均為checkpoint後描述性評估；OOS不進loss、gradient、epoch selection或任何模型擬合。",
        f"- Dynamic-K的K與orderable candidate universe來自既有{reference_arm} replay；Future Target只在replay後離線join作模型品質診斷。",
        f"- 本報表不修改config設定的比較模型、{reference_arm} reference selector、策略參數或既有策略結果。",
        "",
    ]
    return "\n".join(lines)


def run_comparison(
    *,
    project_root: Path = PROJECT_ROOT,
    filter_id: str = BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    architecture: str = BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    specs = _model_specs()
    model_ids = tuple(model_id for model_id, _profile in specs)
    model_frames: dict[str, pd.DataFrame] = {}
    model_meta: list[dict[str, Any]] = []
    target_ids = set()
    for model_id, profile in specs:
        frame, meta = _load_model_frame(
            root=root,
            filter_id=filter_id,
            architecture=architecture,
            model_id=model_id,
            profile=profile,
        )
        model_frames[model_id] = frame
        model_meta.append(meta)
        target_ids.add(str(meta["continuous_target_id"]))
    if len(target_ids) != 1:
        raise ValueError(f"continuous ranker comparison target不一致: {sorted(target_ids)}")

    top_k = int(BREAKOUT_QUALITY_CONTINUOUS_RANKER_REPORT_TOP_K)
    boundary_width = int(BREAKOUT_QUALITY_CONTINUOUS_RANKER_REPORT_BOUNDARY_WIDTH)
    fixed_splits: dict[str, Any] = {}
    for split in ("selection", "oos"):
        paired = _paired_score_frame(model_frames, split=split)
        k_by_date = {str(date): top_k for date in paired["date"].unique()}
        fixed_splits[split] = _evaluate_paired_frame(
            paired,
            model_ids=model_ids,
            k_by_date=k_by_date,
            boundary_width=boundary_width,
        )

    comparison_settings = get_breakout_quality_continuous_ranker_comparison_settings()
    reference_arm = comparison_settings.reference_arm
    pair_dir, run_dir, _run_payload = _resolve_reference_pair_dir(
        root=root,
        reference_arm_id=reference_arm,
    )
    dynamic_frame, coverage = _dynamic_orderable_frame(
        pair_dir=pair_dir,
        model_frames=model_frames,
        model_ids=model_ids,
    )
    dynamic_k_by_date = {
        str(date): int(day["dynamic_k"].iloc[0])
        for date, day in dynamic_frame.groupby("date", sort=True)
    }
    dynamic_evaluation = _evaluate_paired_frame(
        dynamic_frame,
        model_ids=model_ids,
        k_by_date=dynamic_k_by_date,
        boundary_width=boundary_width,
    )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "RESULT_AVAILABLE",
        "filter_id": filter_id,
        "model_architecture": architecture,
        "continuous_target_id": next(iter(target_ids)),
        "models": model_meta,
        "comparison_settings": {
            "model_ids": list(model_ids),
            "reference_arm": reference_arm,
            "summary_pair": list(comparison_settings.summary_pair),
        },
        "fixed_k": {
            "top_k": top_k,
            "boundary_width": boundary_width,
            "candidate_universe": "canonical_continuous_ranker_same_day_event_groups",
            "paired_identity_required": True,
            "splits": fixed_splits,
        },
        "dynamic_k": {
            "reference_arm": reference_arm,
            "strategy_run_dir": project_relative_display_path(run_dir, project_root=root),
            "pair_dir": project_relative_display_path(pair_dir, project_root=root),
            "candidate_universe": "reference_arm_score_ranking_orderable_candidates",
            "k_source": "Resource_Aware_Pre_Market_Order_Limit_on_Max_DL_Eligible_days",
            "score_join": "ticker + signal_date",
            "coverage": coverage,
            "evaluation": dynamic_evaluation,
        },
        "random_baseline": {
            "method": "exact_uniform_random_order_expectation",
            "monte_carlo_used": False,
            "expected_top_k_lift": 0.0,
            "expected_boundary_concordance": 0.5,
            "expected_boundary_gap": 0.0,
        },
        "training_or_strategy_execution": {
            "model_retrained": False,
            "strategy_replayed": False,
            "selector_modified": False,
            "oos_used_for_training_or_model_selection": False,
        },
    }
    output_dir = resolve_filter_output_dir(root, filter_id=filter_id) / OUTPUT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / REPORT_JSON_FILENAME
    markdown_path = output_dir / REPORT_MARKDOWN_FILENAME
    _write_json(json_path, payload)
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    if not compact_console_enabled():
        print("\n" + render_console(payload))
        print_artifact_paths(
            (("Paired品質報表", markdown_path), ("Paired品質JSON", json_path)),
            project_root=root,
        )
    return payload


def main(argv=None) -> int:
    args = _parse_args(argv)
    run_comparison(
        filter_id=str(args.filter_id),
        architecture=str(args.model_architecture),
    )
    return 0


__all__ = [
    "main",
    "run_comparison",
    "render_console",
    "REPORT_JSON_FILENAME",
    "REPORT_MARKDOWN_FILENAME",
    "OUTPUT_DIRNAME",
]
