"""Read-only calibration audit for MR-13F Direct-R Forward-OOS output.

The audit consumes the frozen MR-13F Forward report and OOS score table.  It does
not retrain, rescore, build PIT artifacts, or alter strategy runtime.  The only
pre-OOS magnitude baseline is the Selection-inner Validation target mean already
persisted in the canonical report; OOS labels are used only for post-hoc audit
metrics.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.audit import AUDIT_OUTPUT_ROOT, AuditDefinition
from core.console_report import (
    print_artifact_paths,
    project_relative_display_path,
    render_key_values,
    render_section,
    render_table,
    render_title,
)
from core.runtime_utils import get_taipei_now
from filters.breakout_quality.artifacts import compute_file_sha256
from filters.breakout_quality.paths import resolve_filter_model_output_dir
from filters.breakout_quality.ranking_score_store import (
    CONTINUOUS_RANKER_REPORT_FILENAME,
    DAILY_RANKER_OOS_SCORE_FILENAME,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_RESULT_SCHEMA_VERSION = 1


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _source_paths(root: Path, definition: AuditDefinition) -> tuple[Path, Path]:
    source = dict(definition.source or {})
    filter_id = str(source.get("filter_id") or "").strip()
    architecture = str(source.get("model_architecture") or "").strip()
    profile = str(source.get("experiment_profile") or "").strip()
    if not filter_id or not architecture or not profile:
        raise ValueError("Direct-R calibration Audit source缺少filter/model/profile identity")
    output_dir = resolve_filter_model_output_dir(root, filter_id, architecture, profile)
    return output_dir / CONTINUOUS_RANKER_REPORT_FILENAME, output_dir / DAILY_RANKER_OOS_SCORE_FILENAME


def _load_source(root: Path, definition: AuditDefinition) -> tuple[dict[str, Any], pd.DataFrame, Path, Path]:
    report_path, score_path = _source_paths(root, definition)
    if not report_path.is_file():
        raise FileNotFoundError(
            "缺少MR-13F Forward report: "
            + project_relative_display_path(report_path, project_root=root)
        )
    if not score_path.is_file():
        raise FileNotFoundError(
            "缺少MR-13F Forward OOS scores: "
            + project_relative_display_path(score_path, project_root=root)
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    source = dict(definition.source or {})
    expected = {
        "filter_id": str(source.get("filter_id") or ""),
        "model_architecture": str(source.get("model_architecture") or ""),
        "experiment_profile": str(source.get("experiment_profile") or ""),
        "model_research_id": str(source.get("model_research_id") or ""),
    }
    for field, value in expected.items():
        if value and str(report.get(field) or "") != value:
            raise ValueError(f"MR-13F Audit report identity不一致: {field}")
    if str(report.get("training", {}).get("objective") or "") != "daily_raw_r_regression":
        raise ValueError("MR-13F Audit只接受daily_raw_r_regression report")
    if bool(report.get("oos_used_for_training_or_epoch_selection")):
        raise ValueError("MR-13F report顯示OOS曾進入training/epoch selection")

    artifact = dict((report.get("artifacts") or {}).get("oos_scores_gzip") or {})
    if artifact:
        expected_size = artifact.get("size_bytes")
        expected_sha = str(artifact.get("sha256") or "").lower()
        if expected_size is not None and int(expected_size) != int(score_path.stat().st_size):
            raise ValueError("MR-13F OOS score size與report artifact identity不一致")
        if expected_sha and compute_file_sha256(score_path).lower() != expected_sha:
            raise ValueError("MR-13F OOS score SHA256與report artifact identity不一致")

    frame = pd.read_csv(score_path, compression="infer")
    required = {"ticker", "date", "target_raw_r", "model_score", "predicted_r"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"MR-13F OOS score缺少欄位: {missing}")
    frame["ticker"] = frame["ticker"].fillna("").astype(str).str.strip()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.strftime("%Y-%m-%d")
    for column in ("target_raw_r", "model_score", "predicted_r"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    finite_score = np.isfinite(frame["model_score"].to_numpy(dtype=np.float64)) & np.isfinite(
        frame["predicted_r"].to_numpy(dtype=np.float64)
    )
    if bool(finite_score.any()) and not np.allclose(
        frame.loc[finite_score, "model_score"].to_numpy(dtype=np.float64),
        frame.loc[finite_score, "predicted_r"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("MR-13F model_score與predicted_r不一致")
    return report, frame, report_path, score_path


def collect_direct_r_calibration_status(
    definition: AuditDefinition,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    try:
        report, frame, report_path, score_path = _load_source(root, definition)
        evaluable = np.isfinite(frame["predicted_r"].to_numpy(dtype=np.float64)) & np.isfinite(
            frame["target_raw_r"].to_numpy(dtype=np.float64)
        )
        if int(evaluable.sum()) < 2:
            raise ValueError("MR-13F OOS target-evaluable rows不足")
        validation = dict(((report.get("split_metrics") or {}).get("validation") or {}).get("raw_r_regression") or {})
        frozen_mean = _finite(validation.get("target_r_mean"))
        if frozen_mean is None:
            raise ValueError("MR-13F report缺少Selection-inner Validation target mean")
        return {
            "audit_id": definition.audit_id,
            "status": "READY",
            "reason": "",
            "source": {
                "display": "MR-13F frozen Forward OOS Predicted-R calibration",
                "report": project_relative_display_path(report_path, project_root=root),
                "scores": project_relative_display_path(score_path, project_root=root),
                "evaluable_rows": int(evaluable.sum()),
            },
        }
    except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError) as exc:
        return {
            "audit_id": definition.audit_id,
            "status": "BLOCKED",
            "reason": str(exc),
            "source": {"display": "MR-13F frozen Forward OOS Predicted-R calibration"},
        }


def _huber_mean(error: np.ndarray, delta: float) -> float:
    absolute = np.abs(np.asarray(error, dtype=np.float64))
    quadratic = np.minimum(absolute, float(delta))
    linear = absolute - quadratic
    return float(np.mean(0.5 * quadratic * quadratic + float(delta) * linear))


def _error_metrics(predicted: np.ndarray, actual: np.ndarray, *, huber_delta_r: float) -> dict[str, float]:
    error = np.asarray(predicted, dtype=np.float64) - np.asarray(actual, dtype=np.float64)
    return {
        "huber_raw_r": _huber_mean(error, float(huber_delta_r)),
        "mae_r": float(np.mean(np.abs(error))),
        "rmse_r": float(np.sqrt(np.mean(error * error))),
        "bias_r": float(np.mean(error)),
    }


def _calibration_line(predicted: np.ndarray, actual: np.ndarray) -> dict[str, float | None]:
    x = np.asarray(predicted, dtype=np.float64)
    y = np.asarray(actual, dtype=np.float64)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    x_var = float(np.mean((x - x_mean) ** 2))
    y_var = float(np.mean((y - y_mean) ** 2))
    if x_var <= 0.0:
        return {"intercept_r": y_mean, "slope": None, "r_squared": None}
    slope = float(np.mean((x - x_mean) * (y - y_mean)) / x_var)
    intercept = float(y_mean - slope * x_mean)
    fitted = intercept + slope * x
    sse = float(np.sum((y - fitted) ** 2))
    sst = float(np.sum((y - y_mean) ** 2))
    r_squared = None if sst <= 0.0 else float(1.0 - sse / sst)
    return {"intercept_r": intercept, "slope": slope, "r_squared": r_squared}


def _rank_spearman(values: np.ndarray) -> float | None:
    y = pd.Series(np.asarray(values, dtype=np.float64))
    if len(y) < 2 or y.nunique(dropna=True) < 2:
        return None
    x_rank = pd.Series(np.arange(1, len(y) + 1, dtype=np.float64)).rank(method="average")
    y_rank = y.rank(method="average")
    value = x_rank.corr(y_rank, method="pearson")
    return None if value is None or not math.isfinite(float(value)) else float(value)


def _quantile_buckets(frame: pd.DataFrame, *, bucket_count: int = 10) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = frame[["ticker", "date", "predicted_r", "target_raw_r"]].copy()
    rows = rows.sort_values(["predicted_r", "date", "ticker"], kind="mergesort").reset_index(drop=True)
    count = min(int(bucket_count), len(rows))
    if count < 1:
        return pd.DataFrame(), {"bucket_count": 0, "actual_mean_spearman": None, "adjacent_actual_increases": 0}
    # Equal-count stable buckets; only Predicted R/date/ticker determine membership.
    rows["bucket"] = np.floor(np.arange(len(rows), dtype=np.float64) * count / len(rows)).astype(int) + 1
    grouped = rows.groupby("bucket", sort=True)
    summary = grouped.agg(
        sample_count=("target_raw_r", "size"),
        predicted_r_mean=("predicted_r", "mean"),
        predicted_r_min=("predicted_r", "min"),
        predicted_r_max=("predicted_r", "max"),
        actual_r_mean=("target_raw_r", "mean"),
    ).reset_index()
    positive_rates = grouped["target_raw_r"].apply(lambda x: float((x.to_numpy(dtype=np.float64) > 0.0).mean()))
    summary["actual_positive_rate"] = summary["bucket"].map(positive_rates.to_dict())
    actual_means = summary["actual_r_mean"].to_numpy(dtype=np.float64)
    increases = int(np.sum(np.diff(actual_means) > 0.0)) if len(actual_means) > 1 else 0
    return summary, {
        "bucket_count": int(len(summary)),
        "actual_mean_spearman": _rank_spearman(actual_means),
        "adjacent_actual_increases": increases,
        "adjacent_comparisons": max(0, int(len(actual_means) - 1)),
    }


def _sign_groups(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame[["predicted_r", "target_raw_r"]].copy()
    rows["predicted_sign"] = np.where(rows["predicted_r"].to_numpy(dtype=np.float64) > 0.0, "> 0", "<= 0")
    output = []
    for sign in ("> 0", "<= 0"):
        part = rows[rows["predicted_sign"] == sign]
        if part.empty:
            output.append({
                "predicted_sign": sign,
                "sample_count": 0,
                "predicted_r_mean": None,
                "actual_r_mean": None,
                "actual_positive_rate": None,
            })
            continue
        output.append({
            "predicted_sign": sign,
            "sample_count": int(len(part)),
            "predicted_r_mean": float(part["predicted_r"].mean()),
            "actual_r_mean": float(part["target_raw_r"].mean()),
            "actual_positive_rate": float((part["target_raw_r"].to_numpy(dtype=np.float64) > 0.0).mean()),
        })
    return pd.DataFrame(output)


def _classification(model: dict[str, float], baseline: dict[str, float], line: dict[str, Any], buckets: dict[str, Any]) -> dict[str, str]:
    model_better = model["huber_raw_r"] < baseline["huber_raw_r"] and model["mae_r"] < baseline["mae_r"]
    baseline_better = model["huber_raw_r"] > baseline["huber_raw_r"] and model["mae_r"] > baseline["mae_r"]
    if model_better:
        magnitude = "MODEL_DOMINATES_FROZEN_CONSTANT"
    elif baseline_better:
        magnitude = "FROZEN_CONSTANT_DOMINATES_MODEL"
    else:
        magnitude = "MIXED_MAGNITUDE_ERROR"

    slope = _finite(line.get("slope"))
    bucket_rho = _finite(buckets.get("actual_mean_spearman"))
    if slope is not None and bucket_rho is not None and slope > 0.0 and bucket_rho > 0.0:
        structure = "POSITIVE_CALIBRATION_STRUCTURE"
    elif slope is not None and bucket_rho is not None and slope <= 0.0 and bucket_rho <= 0.0:
        structure = "NONPOSITIVE_CALIBRATION_STRUCTURE"
    else:
        structure = "MIXED_CALIBRATION_STRUCTURE"

    if magnitude == "MODEL_DOMINATES_FROZEN_CONSTANT" and structure == "POSITIVE_CALIBRATION_STRUCTURE":
        overall = "MAGNITUDE_ERROR_BEATS_FROZEN_CONSTANT_WITH_POSITIVE_STRUCTURE"
    elif magnitude == "FROZEN_CONSTANT_DOMINATES_MODEL" and structure == "POSITIVE_CALIBRATION_STRUCTURE":
        overall = "RANKING_STRUCTURE_WITHOUT_MAGNITUDE_ERROR_GAIN"
    elif structure == "NONPOSITIVE_CALIBRATION_STRUCTURE":
        overall = "DIRECT_R_CALIBRATION_STRUCTURE_NOT_SUPPORTED"
    else:
        overall = "MIXED_DIRECT_R_CALIBRATION_EVIDENCE"
    return {"classification": overall, "magnitude_comparison": magnitude, "calibration_structure": structure}


def _audit_metrics(
    frame: pd.DataFrame,
    *,
    frozen_constant_r: float,
    huber_delta_r: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    rows = frame[["ticker", "date", "predicted_r", "target_raw_r"]].copy()
    finite = np.isfinite(rows["predicted_r"].to_numpy(dtype=np.float64)) & np.isfinite(
        rows["target_raw_r"].to_numpy(dtype=np.float64)
    )
    rows = rows.loc[finite].copy()
    if len(rows) < 2:
        raise ValueError("MR-13F calibration Audit有效OOS rows不足")
    predicted = rows["predicted_r"].to_numpy(dtype=np.float64)
    actual = rows["target_raw_r"].to_numpy(dtype=np.float64)
    constant = np.full(len(rows), float(frozen_constant_r), dtype=np.float64)
    model = _error_metrics(predicted, actual, huber_delta_r=float(huber_delta_r))
    baseline = _error_metrics(constant, actual, huber_delta_r=float(huber_delta_r))
    line = _calibration_line(predicted, actual)
    bucket_frame, bucket_metrics = _quantile_buckets(rows, bucket_count=10)
    sign_frame = _sign_groups(rows)
    payload = {
        "sample_count": int(len(rows)),
        "predicted_r_mean": float(np.mean(predicted)),
        "actual_r_mean": float(np.mean(actual)),
        "frozen_constant_r": float(frozen_constant_r),
        "model_error": model,
        "frozen_constant_error": baseline,
        "model_minus_constant": {
            "huber_raw_r": float(model["huber_raw_r"] - baseline["huber_raw_r"]),
            "mae_r": float(model["mae_r"] - baseline["mae_r"]),
            "rmse_r": float(model["rmse_r"] - baseline["rmse_r"]),
        },
        "posthoc_calibration_line": line,
        "bucket_structure": bucket_metrics,
    }
    payload["conclusion"] = _classification(model, baseline, line, bucket_metrics)
    return payload, bucket_frame, sign_frame


def _fmt(value: Any, *, digits: int = 4, suffix: str = "") -> str:
    number = _finite(value)
    return "-" if number is None else f"{number:.{digits}f}{suffix}"


def _render_markdown(definition: AuditDefinition, payload: dict[str, Any], buckets: pd.DataFrame, signs: pd.DataFrame) -> str:
    metrics = payload["metrics"]
    model = metrics["model_error"]
    baseline = metrics["frozen_constant_error"]
    delta = metrics["model_minus_constant"]
    line = metrics["posthoc_calibration_line"]
    bucket_meta = metrics["bucket_structure"]
    conclusion = metrics["conclusion"]
    error_rows = [
        ("MR-13F Predicted R", _fmt(model["huber_raw_r"]), _fmt(model["mae_r"], suffix=" R"), _fmt(model["rmse_r"], suffix=" R"), _fmt(model["bias_r"], suffix=" R")),
        ("Frozen Validation target mean", _fmt(baseline["huber_raw_r"]), _fmt(baseline["mae_r"], suffix=" R"), _fmt(baseline["rmse_r"], suffix=" R"), _fmt(baseline["bias_r"], suffix=" R")),
        ("Model − Constant", _fmt(delta["huber_raw_r"]), _fmt(delta["mae_r"], suffix=" R"), _fmt(delta["rmse_r"], suffix=" R"), "-"),
    ]
    bucket_rows = [
        (
            int(row.bucket), int(row.sample_count), _fmt(row.predicted_r_mean, suffix=" R"),
            _fmt(row.predicted_r_min, suffix=" R"), _fmt(row.predicted_r_max, suffix=" R"),
            _fmt(row.actual_r_mean, suffix=" R"), _fmt(row.actual_positive_rate, digits=2),
        )
        for row in buckets.itertuples(index=False)
    ]
    sign_rows = [
        (
            row.predicted_sign, int(row.sample_count), _fmt(row.predicted_r_mean, suffix=" R"),
            _fmt(row.actual_r_mean, suffix=" R"), _fmt(row.actual_positive_rate, digits=2),
        )
        for row in signs.itertuples(index=False)
    ]
    return "\n\n".join((
        render_title("MR-13F Direct-R Calibration Audit"),
        render_key_values((
            ("Audit", definition.audit_id),
            ("契約", "read-only frozen Forward report/scores；不train、不score、不建PIT、不改策略"),
            ("OOS evaluable rows", f"{int(metrics['sample_count']):,}"),
            ("Predicted R mean", _fmt(metrics["predicted_r_mean"], suffix=" R")),
            ("Actual R mean", _fmt(metrics["actual_r_mean"], suffix=" R")),
            ("Frozen constant", _fmt(metrics["frozen_constant_r"], suffix=" R") + "（Selection-inner Validation target mean）"),
            ("Huber delta", _fmt(payload["huber_delta_r"], suffix=" R")),
        )),
        render_section("Magnitude error vs pre-OOS frozen constant"),
        render_table(("Predictor", "Huber", "MAE", "RMSE", "Bias"), error_rows),
        render_section("Post-hoc calibration structure（OOS target只供Audit，不可直接回灌runtime）"),
        render_key_values((
            ("Actual = intercept + slope × Predicted", f"intercept={_fmt(line.get('intercept_r'), suffix=' R')}；slope={_fmt(line.get('slope'))}"),
            ("Calibration R²", _fmt(line.get("r_squared"))),
            ("Bucket Actual-mean Spearman", _fmt(bucket_meta.get("actual_mean_spearman"))),
            ("Adjacent bucket increases", f"{int(bucket_meta.get('adjacent_actual_increases', 0))}/{int(bucket_meta.get('adjacent_comparisons', 0))}"),
        )),
        render_table(("Bucket", "N", "Pred mean", "Pred min", "Pred max", "Actual mean", "Actual >0 rate"), bucket_rows),
        render_section("Predicted R sign economics"),
        render_table(("Predicted sign", "N", "Pred mean", "Actual mean", "Actual >0 rate"), sign_rows),
        render_section("Direction classification"),
        render_key_values((
            ("Classification", conclusion["classification"]),
            ("Magnitude", conclusion["magnitude_comparison"]),
            ("Calibration structure", conclusion["calibration_structure"]),
        )),
        "判讀限制：OLS slope/intercept、bucket與sign結果都使用Forward OOS target，僅用來決定MR-13F是否值得進下一個受控PIT／calibration實驗，不得把本Audit fitted係數直接當成runtime calibration。程式不設定slope接近1或固定bucket提升幅度等人工門檻。",
    )) + "\n"


def run_direct_r_calibration_audit(
    definition: AuditDefinition,
    *,
    project_root: Path = PROJECT_ROOT,
    quiet: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    status = collect_direct_r_calibration_status(definition, project_root=root)
    if status["status"] != "READY":
        raise RuntimeError(status["reason"])
    report, frame, report_path, score_path = _load_source(root, definition)
    regression_contract = dict(report.get("training", {}).get("raw_r_regression_contract") or {})
    huber_delta_r = _finite(regression_contract.get("huber_delta_r"))
    if huber_delta_r is None or huber_delta_r <= 0.0:
        raise ValueError("MR-13F report缺少合法Huber delta")
    validation = dict(((report.get("split_metrics") or {}).get("validation") or {}).get("raw_r_regression") or {})
    frozen_constant_r = _finite(validation.get("target_r_mean"))
    if frozen_constant_r is None:
        raise ValueError("MR-13F report缺少Selection-inner Validation target mean")
    metrics, bucket_frame, sign_frame = _audit_metrics(
        frame,
        frozen_constant_r=float(frozen_constant_r),
        huber_delta_r=float(huber_delta_r),
    )
    payload = {
        "schema_version": AUDIT_RESULT_SCHEMA_VERSION,
        "audit_id": definition.audit_id,
        "status": "COMPLETED",
        "generated_at": get_taipei_now().isoformat(),
        "read_only": True,
        "model_research_id": str(report.get("model_research_id") or ""),
        "experiment_profile": str(report.get("experiment_profile") or ""),
        "huber_delta_r": float(huber_delta_r),
        "source": {
            "report": project_relative_display_path(report_path, project_root=root),
            "scores": project_relative_display_path(score_path, project_root=root),
            "oos_used_for_training_or_epoch_selection": False,
            "frozen_constant_source": "selection_inner_validation_target_mean",
        },
        "metrics": metrics,
        "oos_target_used_for_runtime_or_training": False,
        "posthoc_fitted_calibration_runtime_eligible": False,
    }

    output_root = root / AUDIT_OUTPUT_ROOT / definition.output_subdir
    run_id = get_taipei_now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    json_path = run_dir / "audit.json"
    md_path = run_dir / "audit.md"
    bucket_path = run_dir / "predicted_r_buckets.csv"
    sign_path = run_dir / "predicted_r_sign_groups.csv"
    bucket_frame.to_csv(bucket_path, index=False, encoding="utf-8-sig")
    sign_frame.to_csv(sign_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(_json_native(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    md = _render_markdown(definition, payload, bucket_frame, sign_frame)
    md_path.write_text(md, encoding="utf-8")

    latest = output_root / "latest"
    if latest.exists():
        shutil.rmtree(latest)
    shutil.copytree(run_dir, latest)
    payload["artifacts"] = {
        "markdown": project_relative_display_path(md_path, project_root=root),
        "json": project_relative_display_path(json_path, project_root=root),
        "buckets": project_relative_display_path(bucket_path, project_root=root),
        "sign_groups": project_relative_display_path(sign_path, project_root=root),
        "latest": project_relative_display_path(latest, project_root=root),
    }
    if not quiet:
        print(md, end="")
        print_artifact_paths(
            (("Audit Markdown", md_path), ("Audit JSON", json_path), ("Bucket CSV", bucket_path), ("Sign CSV", sign_path)),
            project_root=root,
        )
    return payload


__all__ = [
    "collect_direct_r_calibration_status",
    "run_direct_r_calibration_audit",
]
