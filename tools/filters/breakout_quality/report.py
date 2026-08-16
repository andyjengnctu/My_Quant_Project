"""Generate readable breakout-quality research reports with comparable tables."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.breakout_quality import SUPPORTED_BREAKOUT_QUALITY_CLASSIFICATION_EXPERIMENT_PROFILES
from config.breakout_quality import BREAKOUT_QUALITY_EXPERIMENT_PROFILE
from filters.breakout_quality.contract import DEFAULT_FILTER_ID
from filters.breakout_quality.paths import (
    ensure_filter_report_dir,
    resolve_filter_report_json_path,
    resolve_filter_report_markdown_path,
)
from core.console_report import (
    console_color_enabled,
    paint as console_paint,
    print_artifact_paths,
    render_title,
)
from core.report_style import markdown_tone, signal_for_signed_value, tone_for_signal
from tools.filters.breakout_quality.evaluate import (
    EVALUATION_SPLIT_OOS,
    EVALUATION_SPLIT_SELECTION,
    EVALUATION_SPLIT_TRAIN,
    EVALUATION_SPLIT_VALIDATION,
    evaluate_split_from_context,
    prepare_evaluation_context,
)


REPORT_SCHEMA_VERSION = 4
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
SPLIT_LABELS = {
    EVALUATION_SPLIT_TRAIN: "Inner Train",
    EVALUATION_SPLIT_VALIDATION: "Validation*",
    EVALUATION_SPLIT_SELECTION: "Selection",
    EVALUATION_SPLIT_OOS: "OOS",
}

REPORT_LABELS = {
    "original_pass_rate": "原始 PASS",
    "original_reject_rate": "原始 REJECT",
    "model_pass_rate": "模型 PASS",
    "model_reject_rate": "模型 REJECT",
    "pass_precision": "PASS Precision",
    "pass_recall": "PASS Recall",
    "reject_specificity": "REJECT Specificity",
    "reject_npv": "REJECT NPV",
    "accuracy": "Accuracy",
    "precision_absolute_lift": "Precision 絕對",
    "precision_relative_lift": "Precision 相對",
    "average_score": "平均 Score",
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "產生 Breakout Quality 表格化研究報表；終端、Markdown 與完整 JSON "
            "共用 evaluate.py 的同一份 metrics"
        )
    )
    parser.add_argument("--filter-id", default=DEFAULT_FILTER_ID)
    parser.add_argument(
        "--experiment-profile",
        choices=SUPPORTED_BREAKOUT_QUALITY_CLASSIFICATION_EXPERIMENT_PROFILES,
        default=BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
        help="要產生報表的訓練實驗 profile",
    )
    parser.add_argument(
        "--include-oos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否納入最終 OOS（預設納入）；OOS 不得用於回頭調整 threshold、epochs 或模型",
    )
    parser.add_argument(
        "--score-path",
        default=None,
        help="自訂 research_scores.csv；通常不需指定",
    )
    return parser.parse_args(argv)




def _paint(text: object, tone: str, *, enabled: bool, bold: bool = False) -> str:
    normalized_tone = "cyan" if str(tone) == "blue" else str(tone)
    return console_paint(text, normalized_tone, enabled=enabled, bold=bold)


def _paint_multiline(text: object, tone: str, *, enabled: bool, bold: bool = False) -> str:
    """Color every rendered line independently so ANSI state cannot bleed across table cells."""
    raw = str(text)
    if not enabled:
        return raw
    return "\n".join(
        _paint(line, tone, enabled=True, bold=bold)
        for line in raw.splitlines()
    )


def _markdown_color(text: object, tone: str, *, bold: bool = True) -> str:
    return markdown_tone(text, tone, bold=bold)


def _tone_for_delta(value: float | None) -> str:
    return tone_for_signal(signal_for_signed_value(value))

def _pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:.{digits}f}%"


def _signed_pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:+.{digits}f}%"


def _pp(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:+.{digits}f} pp"


def _decimal(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _loss(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.6f}"


def _count(value: float | int | None, digits: int = 1) -> str:
    if value is None:
        return "-"
    number = float(value)
    if number.is_integer():
        return f"{int(number):,}"
    return f"{number:,.{digits}f}"


def _weighted_count(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):,.1f}"


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _display_width(text: str) -> int:
    width = 0
    visible_text = ANSI_ESCAPE_RE.sub("", str(text))
    for char in visible_text:
        if unicodedata.combining(char):
            continue
        width += 2 if unicodedata.east_asian_width(char) in {"W", "F", "A"} else 1
    return width


def _pad_display(text: str, width: int, *, align: str = "left") -> str:
    raw = str(text)
    missing = max(0, int(width) - _display_width(raw))
    if align == "right":
        return " " * missing + raw
    if align == "center":
        left = missing // 2
        return " " * left + raw + " " * (missing - left)
    return raw + " " * missing


def _render_ascii_table(
    headers: list[str],
    rows: list[list[object]],
    *,
    aligns: list[str] | None = None,
) -> list[str]:
    normalized_rows = [[str(cell) for cell in row] for row in rows]
    all_rows = [[str(cell) for cell in headers], *normalized_rows]
    column_count = len(headers)
    if any(len(row) != column_count for row in all_rows):
        raise ValueError("ASCII table 每列欄數必須一致")
    normalized_aligns = aligns or ["left"] * column_count
    if len(normalized_aligns) != column_count:
        raise ValueError("ASCII table aligns 欄數不一致")

    widths: list[int] = []
    for column_index in range(column_count):
        widths.append(
            max(
                _display_width(line)
                for row in all_rows
                for line in row[column_index].splitlines() or [""]
            )
        )

    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    rendered = [border]

    def append_row(row: list[str], *, header: bool = False) -> None:
        split_cells = [cell.splitlines() or [""] for cell in row]
        height = max(len(lines) for lines in split_cells)
        for line_index in range(height):
            cells = []
            for column_index, lines in enumerate(split_cells):
                line = lines[line_index] if line_index < len(lines) else ""
                align = "center" if header else normalized_aligns[column_index]
                cells.append(f" {_pad_display(line, widths[column_index], align=align)} ")
            rendered.append("|" + "|".join(cells) + "|")

    append_row([str(cell) for cell in headers], header=True)
    rendered.append(border)
    for row in normalized_rows:
        append_row(row)
        rendered.append(border)
    return rendered


def _section(title: str, *, color: bool = False) -> list[str]:
    rendered_title = _paint(title, "blue", enabled=color, bold=True)
    return ["", "=" * 96, f" {rendered_title}", "=" * 96, ""]


def _metric_summary(metrics: dict) -> dict:
    headline = metrics["ticker_date_group_weighted"]
    base = headline.get("base_pass_rate")
    precision = headline.get("pass_precision")
    lift = headline.get("precision_lift_vs_all_pass")
    confusion = headline.get("confusion")
    if not isinstance(confusion, dict):
        confusion = {}
    return {
        "split": metrics["split"],
        "date_start": metrics["selected_date_range"]["start"],
        "date_end": metrics["selected_date_range"]["end"],
        "group_count": headline.get("group_count"),
        "row_count": headline.get("row_count"),
        "weight_sum": headline.get("weight_sum"),
        "base_pass_rate": base,
        "acceptance_rate": headline.get("acceptance_rate"),
        "pass_precision": precision,
        "precision_delta": (
            float(precision) - float(base)
            if precision is not None and base is not None
            else None
        ),
        "precision_relative_change": (
            float(lift) - 1.0 if lift is not None else None
        ),
        "pass_recall": headline.get("pass_recall"),
        "false_rejection_rate": headline.get("false_rejection_rate"),
        "reject_specificity": headline.get("reject_specificity"),
        "accuracy": headline.get("accuracy"),
        "all_pass_baseline_accuracy": headline.get("all_pass_baseline_accuracy"),
        "avg_score": headline.get("avg_score"),
        "ranking_and_calibration": headline.get("ranking_and_calibration") or {},
        "confusion": {
            "tp": confusion.get("true_pass_pred_pass"),
            "fn": confusion.get("true_pass_pred_reject"),
            "fp": confusion.get("true_reject_pred_pass"),
            "tn": confusion.get("true_reject_pred_reject"),
        },
    }


def _oos_conclusion(summary: dict | None) -> dict:
    if summary is None:
        return {
            "status": "RESEARCH_ONLY",
            "title": "尚未執行 OOS",
            "explanation": "目前報表只包含 Selection 內診斷，不能據此判定正式泛化能力。",
            "deployment_guidance": "不要匯出 forward-OOS runtime scores；待模型與參數鎖定後再執行 OOS。",
        }
    delta = summary.get("precision_delta")
    if delta is None:
        return {
            "status": "INCONCLUSIVE",
            "title": "OOS 無法判定",
            "explanation": "OOS 缺少可比較的 PASS precision 或基準 PASS。",
            "deployment_guidance": "先確認 score table、標籤與 split 工件完整。",
        }
    if float(delta) <= 0:
        return {
            "status": "FAIL",
            "title": "OOS 未顯示品質提升",
            "explanation": (
                f"PASS Precision 為 {_pct(summary.get('pass_precision'))}，"
                f"低於原始 PASS {_pct(summary.get('base_pass_rate'))}，"
                f"差異 {_pp(delta)}。"
            ),
            "deployment_guidance": (
                "維持 breakout quality runtime filter 關閉，不匯出 forward-OOS scores；"
                "不要用同一段 OOS 回頭調整 threshold、epochs 或 learning rate。"
            ),
        }
    return {
        "status": "PASS_WITH_REVIEW",
        "title": "OOS PASS Precision 有提升",
        "explanation": (
            f"PASS Precision 為 {_pct(summary.get('pass_precision'))}，"
            f"高於原始 PASS {_pct(summary.get('base_pass_rate'))}，"
            f"差異 {_pp(delta)}；仍須同時檢查模型 PASS 與 PASS Recall。"
        ),
        "deployment_guidance": (
            "可進入策略層經濟效果驗證，但在確認淨報酬、交易數與風險改善前，"
            "仍不應直接啟用正式 runtime filter。"
        ),
    }


def _training_summary(manifest: dict) -> dict:
    epoch_selection = manifest.get("inner_validation_epoch_selection")
    if not isinstance(epoch_selection, dict):
        epoch_selection = {}
    best_metrics = epoch_selection.get("best_validation_metrics")
    if not isinstance(best_metrics, dict):
        best_metrics = {}
    history = epoch_selection.get("history")
    if not isinstance(history, list):
        history = []

    normalized_history: list[dict] = []
    final_best_epoch = epoch_selection.get("best_epoch", manifest.get("selected_epoch"))
    completed_epochs = int(epoch_selection.get("completed_epochs") or len(history) or 0)
    max_epochs = int(manifest.get("max_epochs") or 0)
    early_stopping_enabled = bool(manifest.get("early_stopping_enabled", False))
    for index, raw in enumerate(history):
        if not isinstance(raw, dict):
            continue
        train_metrics = raw.get("inner_train_metrics")
        validation_metrics = raw.get("inner_validation_metrics")
        if not isinstance(train_metrics, dict):
            train_metrics = {}
        if not isinstance(validation_metrics, dict):
            validation_metrics = {}
        epoch = int(raw.get("epoch") or index + 1)
        if final_best_epoch is not None and epoch == int(final_best_epoch):
            decision = "最後選擇"
        elif bool(raw.get("is_best_epoch", False)):
            decision = "曾為最佳"
        elif (
            early_stopping_enabled
            and epoch == completed_epochs
            and completed_epochs < max_epochs
        ):
            decision = "Early Stop"
        else:
            decision = "未改善"
        normalized_history.append(
            {
                "epoch": epoch,
                "batch_loss": raw.get("batch_loss"),
                "train_loss": train_metrics.get("loss"),
                "train_accuracy": train_metrics.get("accuracy"),
                "train_pass_rate": train_metrics.get("pass_rate"),
                "validation_loss": validation_metrics.get("loss"),
                "validation_accuracy": validation_metrics.get("accuracy"),
                "validation_pass_rate": validation_metrics.get("pass_rate"),
                "is_best_epoch": bool(raw.get("is_best_epoch", False)),
                "decision": decision,
            }
        )

    model_spec = manifest.get("model_spec")
    if not isinstance(model_spec, dict):
        model_spec = {}
    experiment_settings = manifest.get("experiment_settings")
    if not isinstance(experiment_settings, dict):
        experiment_settings = {}
    return {
        "model_architecture": manifest.get("model_architecture"),
        "experiment_profile": manifest.get("experiment_profile", "baseline"),
        "experiment_settings": experiment_settings,
        "lr_schedule_name": experiment_settings.get("lr_schedule_name", "none"),
        "lr_schedule_parameters": experiment_settings.get("lr_schedule_parameters", {}),
        "augmentation_name": experiment_settings.get("augmentation_name", "none"),
        "augmentation_parameters": experiment_settings.get(
            "augmentation_parameters", {}
        ),
        "training_sampling_mode": experiment_settings.get(
            "training_sampling_mode", "all_event_rows_group_weighted"
        ),
        "training_sampling": manifest.get("training_sampling") or {},
        "torch_execution": manifest.get("torch_execution") or {},
        "model_spec": model_spec,
        "trainable_parameter_count": manifest.get("trainable_parameter_count"),
        "total_parameter_count": manifest.get("total_parameter_count"),
        "frozen_parameter_count": manifest.get("frozen_parameter_count"),
        "self_supervised_pretraining": manifest.get("self_supervised_pretraining"),
        "external_pretrained_encoder": manifest.get("external_pretrained_encoder"),
        "sequence_length": manifest.get("sequence_length"),
        "training_mode": manifest.get("training_mode"),
        "inner_validation_used": bool(manifest.get("inner_validation_used", False)),
        "max_epochs": manifest.get("max_epochs"),
        "selected_epoch": manifest.get("selected_epoch"),
        "completed_epoch_search": epoch_selection.get(
            "completed_epochs", manifest.get("completed_epochs")
        ),
        "best_validation_loss": epoch_selection.get("best_validation_loss"),
        "best_validation_accuracy": best_metrics.get("accuracy"),
        "best_validation_pass_rate": best_metrics.get("pass_rate"),
        "fixed_threshold": manifest.get("fixed_evaluation_threshold"),
        "optimizer_name": manifest.get("optimizer_name", "adam"),
        "learning_rate": manifest.get("learning_rate"),
        "weight_decay": manifest.get("weight_decay"),
        "gradient_clip_norm": manifest.get("gradient_clip_norm"),
        "final_refit_plan": manifest.get("final_refit_plan") or {},
        "class_weight_mode": manifest.get("class_weight_mode"),
        "time_weight_mode": manifest.get("time_weight_mode"),
        "training_weight_reduction": manifest.get(
            "training_weight_reduction", "batch_weight_sum"
        ),
        "batch_size": manifest.get("batch_size"),
        "seed": manifest.get("seed"),
        "early_stopping_enabled": early_stopping_enabled,
        "early_stopping_patience": manifest.get("early_stopping_patience"),
        "early_stopping_min_delta": manifest.get("early_stopping_min_delta"),
        "epoch_selection_source": manifest.get("epoch_selection_source"),
        "epoch_history": normalized_history,
    }


def _yearly_metric_summary(item: dict) -> dict:
    metrics = item.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("yearly OOS metrics 缺少 metrics object")
    ranking = metrics.get("ranking_and_calibration")
    if not isinstance(ranking, dict):
        ranking = {}
    base = metrics.get("base_pass_rate")
    precision = metrics.get("pass_precision")
    return {
        "year": int(item["year"]),
        "date_start": item["date_range"]["start"],
        "date_end": item["date_range"]["end"],
        "policy_window": item.get("policy_window") or {},
        "is_partial_calendar_year": bool(item.get("is_partial_calendar_year", False)),
        "group_count": metrics.get("group_count"),
        "row_count": metrics.get("row_count"),
        "base_pass_rate": base,
        "acceptance_rate": metrics.get("acceptance_rate"),
        "pass_precision": precision,
        "precision_delta": (
            float(precision) - float(base)
            if precision is not None and base is not None
            else None
        ),
        "pass_recall": metrics.get("pass_recall"),
        "avg_score": metrics.get("avg_score"),
        "ranking_and_calibration": ranking,
    }


def build_report_payload(*, metrics_by_split: dict[str, dict], context: dict) -> dict:
    summaries = {
        split_name: _metric_summary(metrics)
        for split_name, metrics in metrics_by_split.items()
    }
    conclusion = _oos_conclusion(summaries.get(EVALUATION_SPLIT_OOS))
    oos_metrics = metrics_by_split.get(EVALUATION_SPLIT_OOS)
    yearly_items = (
        oos_metrics.get("yearly_ticker_date_group_weighted", [])
        if isinstance(oos_metrics, dict)
        else []
    )
    if not isinstance(yearly_items, list):
        raise ValueError("yearly_ticker_date_group_weighted 必須是 list")
    yearly_summaries = [_yearly_metric_summary(item) for item in yearly_items]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "filter_id": context["filter_id"],
        "headline_basis": "ticker_date_group_weighted",
        "conclusion": conclusion,
        "training": _training_summary(context["model_manifest"]),
        "split_summaries": summaries,
        "oos_yearly_summaries": yearly_summaries,
        "full_metrics": metrics_by_split,
        "audit": {
            "score_path": str(context["score_path"]),
            "threshold_source": "model_manifest.fixed_evaluation_threshold",
            "oos_reuse_warning": (
                "若依 OOS 結果修改 threshold、epochs、learning rate、feature、label 或模型，"
                "該期間即不再是 final OOS。"
            ),
        },
    }


def _split_order(payload: dict) -> list[str]:
    return [
        split_name
        for split_name in (
            EVALUATION_SPLIT_TRAIN,
            EVALUATION_SPLIT_VALIDATION,
            EVALUATION_SPLIT_SELECTION,
            EVALUATION_SPLIT_OOS,
        )
        if split_name in payload["split_summaries"]
    ]


def _markdown_epoch_table(training: dict) -> list[str]:
    history = training.get("epoch_history") or []
    if not history:
        return [
            "目前未使用 inner validation；最終模型依固定 epochs 訓練。",
            "",
        ]
    lines = [
        "| Epoch | Batch Loss | Train Loss | Train Accuracy | Train Pass Rate | Validation Loss | Validation Accuracy | Validation Pass Rate | 判定 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in history:
        selected = row["decision"] == "最後選擇"
        decision = _markdown_color("★ 最後選擇", "blue") if selected else row["decision"]
        epoch_text = _markdown_color(row["epoch"], "blue") if selected else str(row["epoch"])
        val_loss = _markdown_color(_loss(row.get("validation_loss")), "blue") if selected else _loss(row.get("validation_loss"))
        lines.append(
            "| {epoch} | {batch} | {train_loss} | {train_acc} | {train_pass} | {val_loss} | {val_acc} | {val_pass} | {decision} |".format(
                epoch=epoch_text,
                batch=_loss(row.get("batch_loss")),
                train_loss=_loss(row.get("train_loss")),
                train_acc=_pct(row.get("train_accuracy")),
                train_pass=_pct(row.get("train_pass_rate")),
                val_loss=val_loss,
                val_acc=_pct(row.get("validation_accuracy")),
                val_pass=_pct(row.get("validation_pass_rate")),
                decision=decision,
            )
        )
    lines.append("")
    return lines

def _markdown_split_table(payload: dict) -> list[str]:
    lines = [
        (
            f"| 區段 | 日期 | Groups | {REPORT_LABELS['original_pass_rate']} | "
            f"{REPORT_LABELS['model_pass_rate']} | {REPORT_LABELS['pass_precision']} | "
            f"{REPORT_LABELS['precision_absolute_lift']} | {REPORT_LABELS['pass_recall']} | "
            f"{REPORT_LABELS['average_score']} |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split_name in _split_order(payload):
        summary = payload["split_summaries"][split_name]
        details = _confusion_details(summary)
        delta_tone = _tone_for_delta(summary.get("precision_delta"))
        label = SPLIT_LABELS[split_name]
        if split_name == EVALUATION_SPLIT_OOS:
            label = _markdown_color(label, delta_tone)
        lines.append(
            (
                "| {label} | {period} | {groups:,} | {original_pass} | {model_pass} | "
                "{precision} | {absolute_lift} | {recall} | {avg_score} |"
            ).format(
                label=label,
                period=f"{summary['date_start']}～{summary['date_end']}",
                groups=int(summary.get("group_count") or 0),
                original_pass=_pct(details.get("base_pass_rate")),
                model_pass=_pct(details.get("acceptance_rate")),
                precision=_markdown_color(_pct(details.get("precision")), delta_tone),
                absolute_lift=_markdown_color(_pp(summary.get("precision_delta")), delta_tone),
                recall=_pct(details.get("recall")),
                avg_score=_decimal(summary.get("avg_score")),
            )
        )
    lines.append("")
    return lines


def _ranking_summary(summary: dict | None) -> dict:
    if not isinstance(summary, dict):
        return {}
    value = summary.get("ranking_and_calibration")
    return value if isinstance(value, dict) else {}


def _coverage_precision(ranking: dict, coverage: float) -> float | None:
    values = ranking.get("precision_at_coverage")
    if not isinstance(values, dict):
        return None
    return values.get(f"{coverage:.2f}")


def _markdown_ranking_table(payload: dict, number: int) -> list[str]:
    lines = [f"## {number}. 排序與校準診斷", ""]
    lines.extend(
        [
            "| 區段 | PR-AUC | Precision@50% coverage | Precision@60% coverage | Precision@70% coverage | Recall@60% Precision | Brier Score | ECE（10 bins） |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split_name in _split_order(payload):
        summary = payload["split_summaries"][split_name]
        ranking = _ranking_summary(summary)
        lines.append(
            "| {label} | {pr_auc} | {p50} | {p60} | {p70} | {r60} | {brier} | {ece} |".format(
                label=SPLIT_LABELS[split_name],
                pr_auc=_decimal(ranking.get("average_precision_pr_auc"), 4),
                p50=_pct(_coverage_precision(ranking, 0.50)),
                p60=_pct(_coverage_precision(ranking, 0.60)),
                p70=_pct(_coverage_precision(ranking, 0.70)),
                r60=_pct(ranking.get("recall_at_precision_60")),
                brier=_decimal(ranking.get("brier_score"), 4),
                ece=_decimal(ranking.get("expected_calibration_error_10_bins"), 4),
            )
        )
    lines.extend(
        [
            "",
            "> 固定 coverage 指標只用來比較排序能力；本報表不允許依 OOS 的診斷 threshold 回頭調整正式 threshold。",
            "",
        ]
    )
    return lines

def _markdown_oos_yearly_tables(payload: dict, number: int) -> list[str]:
    rows = payload.get("oos_yearly_summaries") or []
    if not rows:
        return []
    lines = [f"## {number}. OOS 年度診斷", ""]
    lines.extend(
        [
            "| 年度 | 日期 | Groups | 原始 PASS | 模型 PASS | PASS Precision | Precision 絕對 | PASS Recall | 平均 Score |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        year_label = f"{row['year']}{'*' if row['is_partial_calendar_year'] else ''}"
        lines.append(
            "| {year} | {period} | {groups} | {base} | {acceptance} | {precision} | {lift} | {recall} | {score} |".format(
                year=year_label,
                period=f"{row['date_start']}～{row['date_end']}",
                groups=_count(row.get("group_count"), digits=0),
                base=_pct(row.get("base_pass_rate")),
                acceptance=_pct(row.get("acceptance_rate")),
                precision=_pct(row.get("pass_precision")),
                lift=_pp(row.get("precision_delta")),
                recall=_pct(row.get("pass_recall")),
                score=_decimal(row.get("avg_score")),
            )
        )
    lines.extend(
        [
            "",
            "| 年度 | PR-AUC | P@50% | P@60% | P@70% | R@P60% | Brier | ECE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        ranking = _ranking_summary(row)
        year_label = f"{row['year']}{'*' if row['is_partial_calendar_year'] else ''}"
        lines.append(
            "| {year} | {pr_auc} | {p50} | {p60} | {p70} | {r60} | {brier} | {ece} |".format(
                year=year_label,
                pr_auc=_decimal(ranking.get("average_precision_pr_auc"), 4),
                p50=_pct(_coverage_precision(ranking, 0.50)),
                p60=_pct(_coverage_precision(ranking, 0.60)),
                p70=_pct(_coverage_precision(ranking, 0.70)),
                r60=_pct(ranking.get("recall_at_precision_60")),
                brier=_decimal(ranking.get("brier_score"), 4),
                ece=_decimal(ranking.get("expected_calibration_error_10_bins"), 4),
            )
        )
    if any(bool(row.get("is_partial_calendar_year")) for row in rows):
        lines.extend(["", "\\* 星號表示 OOS policy 僅涵蓋該年度的一部分，不可與完整年度直接等量比較。"] )
    lines.extend(
        [
            "",
            "> 年度表只切分同一份已固定 OOS score 作診斷，不得用來選 threshold、epochs 或模型。",
            "",
        ]
    )
    return lines


def _confusion_details(summary: dict) -> dict:
    confusion = summary.get("confusion") or {}
    tp = float(confusion.get("tp") or 0.0)
    fn = float(confusion.get("fn") or 0.0)
    fp = float(confusion.get("fp") or 0.0)
    tn = float(confusion.get("tn") or 0.0)
    actual_pass = tp + fn
    actual_reject = fp + tn
    predicted_pass = tp + fp
    predicted_reject = fn + tn
    total = actual_pass + actual_reject
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "actual_pass": actual_pass,
        "actual_reject": actual_reject,
        "predicted_pass": predicted_pass,
        "predicted_reject": predicted_reject,
        "total": total,
        "base_pass_rate": _ratio(actual_pass, total),
        "base_reject_rate": _ratio(actual_reject, total),
        "precision": _ratio(tp, predicted_pass),
        "recall": _ratio(tp, actual_pass),
        "false_rejection_rate": _ratio(fn, actual_pass),
        "false_positive_rate": _ratio(fp, actual_reject),
        "specificity": _ratio(tn, actual_reject),
        "negative_predictive_value": _ratio(tn, predicted_reject),
        "acceptance_rate": _ratio(predicted_pass, total),
        "rejection_rate": _ratio(predicted_reject, total),
        "accuracy": _ratio(tp + tn, total),
    }


def _markdown_confusion_matrix(summary: dict, label: str, number: int) -> list[str]:
    details = _confusion_details(summary)
    delta_tone = _tone_for_delta(summary.get("precision_delta"))
    lines = [
        f"## {number}. {label} Confusion Matrix",
        "",
        "| 原始結果＼模型判定 | 模型 PASS | 模型 REJECT | 原始合計 |",
        "|---|---:|---:|---:|",
        (
            "| **原始 PASS** | {tp_cell} | {fn_cell} | "
            "**原始PASS** = {_original_pass_rate}<br>TP + FN = {actual_pass} |"
        ).format(
            tp_cell=_markdown_color(
                f"正確保留 PASS<br>TP = {_weighted_count(details['tp'])}", "green"
            ),
            fn_cell=_markdown_color(
                f"錯殺 PASS<br>FN = {_weighted_count(details['fn'])}", "red"
            ),
            _original_pass_rate=_pct(details["base_pass_rate"]),
            actual_pass=_weighted_count(details["actual_pass"]),
        ),
        (
            "| **原始 REJECT** | {fp_cell} | {tn_cell} | "
            "**原始REJECT** = {_original_reject_rate}<br>FP + TN = {actual_reject} |"
        ).format(
            fp_cell=_markdown_color(
                f"錯誤保留 REJECT<br>FP = {_weighted_count(details['fp'])}", "red"
            ),
            tn_cell=_markdown_color(
                f"正確拒絕 REJECT<br>TN = {_weighted_count(details['tn'])}", "green"
            ),
            _original_reject_rate=_pct(details["base_reject_rate"]),
            actual_reject=_weighted_count(details["actual_reject"]),
        ),
        (
            "| **模型合計** | TP + FP = {model_pass}<br>**模型PASS** = {model_pass_rate} | "
            "FN + TN = {model_reject}<br>**模型REJECT** = {model_reject_rate} | "
            "**全部** = {total} |"
        ).format(
            model_pass=_weighted_count(details["predicted_pass"]),
            model_pass_rate=_pct(details["acceptance_rate"]),
            model_reject=_weighted_count(details["predicted_reject"]),
            model_reject_rate=_pct(details["rejection_rate"]),
            total=_weighted_count(details["total"]),
        ),
        "",
        "### 分類品質",
        "",
        "| 指標 | 公式 | 結果 | 解釋 |",
        "|---|---|---:|---|",
        f"| {REPORT_LABELS['pass_precision']} | TP ÷ (TP + FP) | {_markdown_color(_pct(details['precision']), delta_tone)} | 被保留的訊號中，有多少真的 PASS |",
        f"| {REPORT_LABELS['pass_recall']} | TP ÷ (TP + FN) | {_pct(details['recall'])} | 真正 PASS 中，有多少被保留 |",
        f"| {REPORT_LABELS['reject_specificity']} | TN ÷ (TN + FP) | {_pct(details['specificity'])} | 真正 REJECT 中，有多少被正確拒絕 |",
        f"| {REPORT_LABELS['reject_npv']} | TN ÷ (TN + FN) | {_pct(details['negative_predictive_value'])} | 被拒絕的訊號中，有多少真的 REJECT |",
        f"| {REPORT_LABELS['accuracy']} | (TP + TN) ÷ 全部 | {_pct(details['accuracy'])} | 全部訊號中，模型判斷正確的比例 |",
        "",
        "### 篩選效果",
        "",
        "| 指標 | 公式 | 結果 |",
        "|---|---|---:|",
        f"| {REPORT_LABELS['precision_absolute_lift']} | PASS Precision − 原始 PASS | {_markdown_color(_pp(summary.get('precision_delta')), delta_tone)} |",
        f"| {REPORT_LABELS['precision_relative_lift']} | PASS Precision ÷ 原始 PASS − 1 | {_markdown_color(_signed_pct(summary.get('precision_relative_change')), delta_tone)} |",
        "",
    ]
    if label == "OOS":
        if float(summary.get("precision_delta") or 0.0) <= 0:
            lines.extend(
                [
                    _markdown_color(
                        "判讀：模型在 OOS 中大量判定為 REJECT，但模型 PASS 的品質未提高；"
                        f"同時錯殺 {_pct(summary.get('false_rejection_rate'))} 的原始 PASS。",
                        "red",
                    ),
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    _markdown_color(
                        "判讀：OOS PASS Precision 高於原始 PASS；仍須一起檢查模型 PASS、PASS Recall 與策略層經濟效果。",
                        "green",
                    ),
                    "",
                ]
            )
    else:
        lines.extend(
            [
                _markdown_color(
                    "判讀：此區段中模型具有篩選能力，但是否能部署仍由 OOS 泛化結果決定。",
                    "green",
                ),
                "",
            ]
        )
    return lines

def _comparison_value_text(value: float | None, kind: str) -> str:
    if kind == "pct":
        return _pct(value)
    if kind == "pct_signed":
        return _signed_pct(value)
    if kind == "pp":
        return _pp(value)
    return _decimal(value)


def _comparison_difference_text(
    selection_value: float | None,
    oos_value: float | None,
    kind: str,
) -> tuple[float | None, str]:
    if selection_value is None or oos_value is None:
        return None, "-"
    difference = float(oos_value) - float(selection_value)
    if kind in {"pct", "pct_signed", "pp"}:
        return difference, _pp(difference)
    return difference, _decimal(difference)


def _oos_assessment_rows(payload: dict) -> list[dict]:
    selection = payload["split_summaries"].get(EVALUATION_SPLIT_SELECTION)
    oos = payload["split_summaries"].get(EVALUATION_SPLIT_OOS)
    if selection is None or oos is None:
        return []

    selection_details = _confusion_details(selection)
    oos_details = _confusion_details(oos)

    def row(
        category: str,
        label: str,
        selection_value: float | None,
        oos_value: float | None,
        kind: str,
        judgement: str,
        tone: str,
    ) -> dict:
        difference, difference_text = _comparison_difference_text(
            selection_value,
            oos_value,
            kind,
        )
        return {
            "category": category,
            "label": label,
            "selection_value": selection_value,
            "oos_value": oos_value,
            "kind": kind,
            "selection_text": _comparison_value_text(selection_value, kind),
            "oos_text": _comparison_value_text(oos_value, kind),
            "difference": difference,
            "difference_text": difference_text,
            "judgement": judgement,
            "tone": tone,
        }

    base_difference = float(oos.get("base_pass_rate") or 0.0) - float(
        selection.get("base_pass_rate") or 0.0
    )
    if base_difference > 0:
        base_judgement = f"OOS 原始基準較 Selection 高 {_pp(abs(base_difference)).lstrip('+')}"
    elif base_difference < 0:
        base_judgement = f"OOS 原始基準較 Selection 低 {_pp(abs(base_difference)).lstrip('+')}"
    else:
        base_judgement = "OOS 原始基準與 Selection 相同"

    precision_delta = float(oos.get("precision_delta") or 0.0)
    precision_tone = "green" if precision_delta > 0 else "red"
    precision_judgement = (
        f"通過：OOS 篩選後高於原始 PASS {_pct(oos.get('base_pass_rate'))}"
        if precision_delta > 0
        else f"未通過：OOS 篩選後低於原始 PASS {_pct(oos.get('base_pass_rate'))}"
    )

    selection_absolute = float(selection.get("precision_delta") or 0.0)
    if precision_delta > 0:
        absolute_judgement = "為正，顯示 OOS 品質提升"
    elif selection_absolute > 0:
        absolute_judgement = "由正轉負，沒有 OOS 品質提升"
    else:
        absolute_judgement = "非正值，沒有 OOS 品質提升"

    oos_relative = float(oos.get("precision_relative_change") or 0.0)
    relative_judgement = (
        "相對效果為正"
        if oos_relative > 0
        else "相對效果為負，未通過"
    )

    recall_difference = float(oos.get("pass_recall") or 0.0) - float(
        selection.get("pass_recall") or 0.0
    )
    recall_judgement = (
        f"較 Selection 下降 {_pp(abs(recall_difference)).lstrip('+')}；"
        f"錯殺 {_pct(oos.get('false_rejection_rate'))} 的原始 PASS"
        if recall_difference < 0
        else f"較 Selection 提高 {_pp(abs(recall_difference)).lstrip('+')}；"
        f"錯殺 {_pct(oos.get('false_rejection_rate'))} 的原始 PASS"
    )

    model_pass_difference = float(oos.get("acceptance_rate") or 0.0) - float(
        selection.get("acceptance_rate") or 0.0
    )
    model_pass_direction = "減少" if model_pass_difference < 0 else "增加"
    model_pass_judgement = (
        f"較 Selection {model_pass_direction} {_pp(abs(model_pass_difference)).lstrip('+')}；"
        "是否足夠仍須換算每日可用候選數"
    )

    specificity_difference = float(oos.get("reject_specificity") or 0.0) - float(
        selection.get("reject_specificity") or 0.0
    )
    specificity_judgement = (
        f"提高 {_pp(abs(specificity_difference)).lstrip('+')}；仍須搭配模型 REJECT 比例與 NPV 判讀"
        if specificity_difference >= 0
        else f"下降 {_pp(abs(specificity_difference)).lstrip('+')}，REJECT 辨識能力轉弱"
    )

    npv_difference = float(oos_details.get("negative_predictive_value") or 0.0) - float(
        selection_details.get("negative_predictive_value") or 0.0
    )
    npv_judgement = (
        f"下降 {_pp(abs(npv_difference)).lstrip('+')}；模型 REJECT 中真正 REJECT 為 "
        f"{_pct(oos_details.get('negative_predictive_value'))}"
        if npv_difference < 0
        else f"提高 {_pp(abs(npv_difference)).lstrip('+')}；模型 REJECT 中真正 REJECT 為 "
        f"{_pct(oos_details.get('negative_predictive_value'))}"
    )

    accuracy_difference = float(oos.get("accuracy") or 0.0) - float(
        selection.get("accuracy") or 0.0
    )
    accuracy_judgement = (
        f"下降 {_pp(abs(accuracy_difference)).lstrip('+')}，整體分類能力未延續至 OOS"
        if accuracy_difference < 0
        else f"提高 {_pp(abs(accuracy_difference)).lstrip('+')}，仍須由主要成效確認是否有部署價值"
    )

    score_difference = float(oos.get("avg_score") or 0.0) - float(
        selection.get("avg_score") or 0.0
    )
    if score_difference < 0:
        score_judgement = f"下降 {_decimal(abs(score_difference))}，OOS 平均 Score 較低"
    elif score_difference > 0:
        score_judgement = f"提高 {_decimal(score_difference)}，OOS 平均 Score 較高"
    else:
        score_judgement = "與 Selection 相同，平均 Score 無變化"

    return [
        row(
            "主要成效",
            REPORT_LABELS["original_pass_rate"],
            selection.get("base_pass_rate"),
            oos.get("base_pass_rate"),
            "pct",
            base_judgement,
            "gray",
        ),
        row(
            "主要成效",
            REPORT_LABELS["pass_precision"],
            selection.get("pass_precision"),
            oos.get("pass_precision"),
            "pct",
            precision_judgement,
            precision_tone,
        ),
        row(
            "主要成效",
            REPORT_LABELS["precision_absolute_lift"],
            selection.get("precision_delta"),
            oos.get("precision_delta"),
            "pp",
            absolute_judgement,
            precision_tone,
        ),
        row(
            "主要成效",
            REPORT_LABELS["precision_relative_lift"],
            selection.get("precision_relative_change"),
            oos.get("precision_relative_change"),
            "pct_signed",
            relative_judgement,
            "green" if oos_relative > 0 else "red",
        ),
        row(
            "過度篩選防線",
            REPORT_LABELS["pass_recall"],
            selection.get("pass_recall"),
            oos.get("pass_recall"),
            "pct",
            recall_judgement,
            "red" if recall_difference < 0 else "green",
        ),
        row(
            "過度篩選防線",
            REPORT_LABELS["model_pass_rate"],
            selection.get("acceptance_rate"),
            oos.get("acceptance_rate"),
            "pct",
            model_pass_judgement,
            "yellow",
        ),
        row(
            "輔助診斷",
            REPORT_LABELS["reject_specificity"],
            selection.get("reject_specificity"),
            oos.get("reject_specificity"),
            "pct",
            specificity_judgement,
            "yellow",
        ),
        row(
            "輔助診斷",
            REPORT_LABELS["reject_npv"],
            selection_details.get("negative_predictive_value"),
            oos_details.get("negative_predictive_value"),
            "pct",
            npv_judgement,
            "red" if npv_difference < 0 else "green",
        ),
        row(
            "輔助診斷",
            REPORT_LABELS["accuracy"],
            selection.get("accuracy"),
            oos.get("accuracy"),
            "pct",
            accuracy_judgement,
            "red" if accuracy_difference < 0 else "green",
        ),
        row(
            "輔助診斷",
            REPORT_LABELS["average_score"],
            selection.get("avg_score"),
            oos.get("avg_score"),
            "decimal",
            score_judgement,
            "red" if score_difference < 0 else "green",
        ),
    ]


def _oos_category_summary(payload: dict) -> dict:
    selection = payload["split_summaries"].get(EVALUATION_SPLIT_SELECTION)
    oos = payload["split_summaries"].get(EVALUATION_SPLIT_OOS)
    if selection is None or oos is None:
        return {
            "main": "尚未納入 OOS，無法判定主要成效。",
            "guard": "尚未納入 OOS，無法檢查過度篩選。",
            "auxiliary": "尚未納入 OOS，無法進行輔助診斷。",
            "main_tone": "yellow",
            "guard_tone": "yellow",
            "auxiliary_tone": "yellow",
        }

    oos_precision_delta = float(oos.get("precision_delta") or 0.0)
    main_pass = oos_precision_delta > 0
    recall_lower = float(oos.get("pass_recall") or 0.0) < float(
        selection.get("pass_recall") or 0.0
    )
    model_pass_lower = float(oos.get("acceptance_rate") or 0.0) < float(
        selection.get("acceptance_rate") or 0.0
    )
    overfilter_fail = recall_lower and model_pass_lower and not main_pass

    selection_npv = float(_confusion_details(selection).get("negative_predictive_value") or 0.0)
    oos_npv = float(_confusion_details(oos).get("negative_predictive_value") or 0.0)
    auxiliary_supports = (
        float(oos.get("accuracy") or 0.0) >= float(selection.get("accuracy") or 0.0)
        and oos_npv >= selection_npv
        and main_pass
    )

    return {
        "main": (
            f"PASS：OOS PASS Precision 高於原始 PASS，Precision 絕對為 {_pp(oos_precision_delta)}。"
            if main_pass
            else f"FAIL：OOS PASS Precision 低於原始 PASS，Precision 絕對為 {_pp(oos_precision_delta)}。"
        ),
        "guard": (
            "FAIL：PASS Recall 與模型 PASS 同時下降，且沒有換得 Precision 提升。"
            if overfilter_fail
            else "REVIEW：請依每日候選數、持股缺口與交易次數確認 Recall 與模型 PASS 是否足夠。"
        ),
        "auxiliary": (
            "支持部署：Accuracy 與 REJECT NPV 未惡化，且主要成效通過。"
            if auxiliary_supports
            else "不支持部署：輔助指標未能支持主要成效的泛化。"
        ),
        "main_tone": "green" if main_pass else "red",
        "guard_tone": "red" if overfilter_fail else "yellow",
        "auxiliary_tone": "green" if auxiliary_supports else "red",
    }


def _markdown_oos_comprehensive_assessment(payload: dict, number: int) -> list[str]:
    rows = _oos_assessment_rows(payload)
    conclusion = payload["conclusion"]
    deployment = _deployment_presentation(payload)
    category_summary = _oos_category_summary(payload)
    lines = [f"## {number}. OOS 綜合判定", ""]

    if rows:
        lines.extend(
            [
                "| 類別 | 指標 | Selection | OOS | OOS - Selection | 判讀 |",
                "|---|---|---:|---:|---:|---|",
            ]
        )
        previous_category = None
        for item in rows:
            category = item["category"] if item["category"] != previous_category else ""
            previous_category = item["category"]
            lines.append(
                "| {category} | {label} | {selection} | {oos} | {difference} | {judgement} |".format(
                    category=f"**{category}**" if category else "",
                    label=item["label"],
                    selection=item["selection_text"],
                    oos=_markdown_color(item["oos_text"], item["tone"]),
                    difference=_markdown_color(item["difference_text"], item["tone"]),
                    judgement=_markdown_color(item["judgement"], item["tone"], bold=False),
                )
            )
        lines.append("")

    lines.extend(
        [
            "### 綜合判定",
            "",
            f"- **主要成效**：{_markdown_color(category_summary['main'], category_summary['main_tone'])}",
            f"- **過度篩選防線**：{_markdown_color(category_summary['guard'], category_summary['guard_tone'])}",
            f"- **輔助診斷**：{_markdown_color(category_summary['auxiliary'], category_summary['auxiliary_tone'])}",
            f"- **最終部署決策**：{_markdown_color(deployment['deployment_decision'], 'red' if conclusion['status'] == 'FAIL' else 'yellow')}",
            f"- **研究限制**：{_markdown_color(deployment['retuning_limit'], 'yellow')}",
            "",
        ]
    )
    return lines


def _deployment_presentation(payload: dict) -> dict:
    conclusion = payload["conclusion"]
    oos = payload["split_summaries"].get(EVALUATION_SPLIT_OOS)
    oos_improved = (
        oos is not None and float(oos.get("precision_delta") or 0.0) > 0
    )
    if oos is None:
        oos_result = "尚未執行 OOS，不能判定正式泛化能力"
    elif oos_improved:
        oos_result = (
            f"有提升：原始 PASS {_pct(oos.get('base_pass_rate'))}，"
            f"PASS Precision {_pct(oos.get('pass_precision'))}，"
            f"差異 {_pp(oos.get('precision_delta'))}"
        )
    else:
        oos_result = (
            f"未提升：原始 PASS {_pct(oos.get('base_pass_rate'))}，"
            f"PASS Precision {_pct(oos.get('pass_precision'))}，"
            f"差異 {_pp(oos.get('precision_delta'))}"
        )

    if conclusion["status"] == "FAIL":
        deployment_decision = (
            "維持 breakout quality runtime filter 關閉，且不匯出 "
            "forward-OOS scores"
        )
    elif conclusion["status"] == "PASS_WITH_REVIEW":
        deployment_decision = (
            "先進入策略層經濟效果驗證；未確認淨報酬、交易數與風險改善前，"
            "不直接啟用 runtime filter"
        )
    else:
        deployment_decision = "尚不足以做部署決策"

    return {
        "oos_result": oos_result,
        "oos_improved": oos_improved,
        "deployment_decision": deployment_decision,
        "retuning_limit": (
            "不得使用同一段 OOS 回頭調整 threshold、epochs、learning rate、"
            "feature、label 或模型"
        ),
        "conclusion_tone": (
            "red"
            if conclusion["status"] == "FAIL"
            else "green"
            if conclusion["status"] == "PASS_WITH_REVIEW"
            else "yellow"
        ),
    }


def _markdown_model_detail_lines(training: dict) -> list[str]:
    spec = training.get("model_spec") or {}
    family = str(spec.get("family") or "")
    if family == "moment_frozen_linear":
        external = training.get("external_pretrained_encoder") or {}
        checkpoint = external.get("checkpoint") or {}
        package = external.get("package") or {}
        dependencies = external.get("runtime_dependencies") or {}
        transformers_record = dependencies.get("transformers") or {}
        return [
            f"- **Model Family**：`{family}`",
            f"- **External Encoder**：`{external.get('repository', spec.get('moment_repository', '-'))}`",
            f"- **Pinned Revision**：`{external.get('resolved_revision', spec.get('moment_revision', '-'))}`",
            f"- **Checkpoint SHA256**：`{checkpoint.get('sha256', '-')}`",
            f"- **MOMENT Package**：`{package.get('name', '-')} {package.get('version', '-')}`",
            f"- **Transformers**：`{transformers_record.get('version', '-')}`",
            f"- **Input Resize**：`{training.get('sequence_length', '-')} → {spec.get('moment_input_length')} bars`",
            f"- **Patch / Stride**：`{spec.get('moment_patch_length')} / {spec.get('moment_patch_stride')}`",
            f"- **Transformer**：`{spec.get('moment_transformer_layers')} layers / {spec.get('moment_transformer_heads')} heads`",
            f"- **Channel Aggregation**：`{spec.get('moment_channel_aggregation')}`",
            f"- **Patch Reduction**：`{spec.get('moment_patch_reduction')}`",
            f"- **Per-channel Representation**：`{spec.get('moment_embedding_dim')}`",
            "- **Encoder Training**：`official external pretraining; frozen downstream`",
            "- **Downstream Head**：`linear`",
        ]
    if family == "mantis_v2_frozen_linear":
        external = training.get("external_pretrained_encoder") or {}
        checkpoint = external.get("checkpoint") or {}
        package = external.get("package") or {}
        return [
            f"- **Model Family**：`{family}`",
            f"- **External Encoder**：`{external.get('repository', spec.get('mantis_repository', '-'))}`",
            f"- **Pinned Revision**：`{external.get('resolved_revision', spec.get('mantis_revision', '-'))}`",
            f"- **Checkpoint SHA256**：`{checkpoint.get('sha256', '-')}`",
            f"- **Mantis Package**：`{package.get('name', '-')} {package.get('version', '-')}`",
            f"- **Input Resize**：`{training.get('sequence_length', '-')} → {spec.get('mantis_input_length')} bars`",
            f"- **Channel Aggregation**：`{spec.get('mantis_channel_aggregation')}`",
            f"- **Transformer Output**：`layer {spec.get('mantis_return_transformer_layer')} / {spec.get('mantis_output_token')}`",
            f"- **Per-channel Representation**：`{spec.get('mantis_embedding_dim')}`",
            "- **Encoder Training**：`official external pretraining; frozen downstream`",
            "- **Downstream Head**：`linear`",
        ]
    if family == "ts2vec_frozen_linear":
        pretraining = training.get("self_supervised_pretraining") or {}
        pretraining_manifest = pretraining.get("manifest") or {}
        pretraining_profile = pretraining_manifest.get("pretraining_profile") or {}
        return [
            f"- **Model Family**：`{family}`",
            f"- **Encoder Depth**：`{spec.get('ts2vec_depth')}`",
            f"- **Hidden Dimensions**：`{spec.get('ts2vec_hidden_dims')}`",
            f"- **Representation Dimensions**：`{spec.get('ts2vec_output_dims')}`",
            "- **Encoder Training**：`Selection-only self-supervised; frozen downstream`",
            "- **Downstream Head**：`linear`",
            f"- **Pretraining Profile**：`{pretraining_profile.get('name', '-')}`",
            f"- **Pretraining Optimizer**：`{pretraining_profile.get('optimizer_name', '-')}`",
            f"- **Pretraining Epochs / Batch**：`{pretraining_profile.get('epochs', '-')} / {pretraining_profile.get('batch_size', '-')}`",
        ]
    if family == "patch_transformer":
        return [
            f"- **Model Family**：`{family}`",
            f"- **Patch / Stride**：`{spec.get('patch_transformer_patch_size')} / {spec.get('patch_transformer_patch_stride')}`",
            f"- **Embedding Dimensions**：`{spec.get('patch_transformer_embedding_dim')}`",
            f"- **Transformer Depth / Heads**：`{spec.get('patch_transformer_depth')} / {spec.get('patch_transformer_heads')}`",
            f"- **MLP Dimensions**：`{spec.get('patch_transformer_mlp_dim')}`",
            f"- **Positional Encoding**：`{spec.get('patch_transformer_positional_encoding')}`",
            f"- **Patch Pooling**：`{spec.get('patch_transformer_pooling')}`",
            "- **Encoder Training**：`supervised from scratch`",
        ]
    if family == "inception_time_market_set":
        return [
            f"- **Model Family**：`{family}`",
            f"- **Candidate Inception Depth**：`{spec.get('inception_depth')}`",
            "- **Candidate Kernel Sizes**：`"
            + "/".join(str(value) for value in spec.get("inception_kernel_sizes") or [])
            + "`",
            f"- **Market History**：`{spec.get('market_set_history_bars')} bars`",
            f"- **Market Base Features**：`{'/'.join(spec.get('market_set_base_features') or [])}`",
            f"- **Shared Stock Embedding**：`{spec.get('market_set_stock_embedding_dim')}`",
            f"- **Market Temporal Normalization**：`{spec.get('market_set_temporal_normalization')} / groups {spec.get('market_set_temporal_normalization_groups')}`",
            f"- **Learned Queries / Heads**：`{spec.get('market_set_query_count')} / {spec.get('market_set_attention_heads')}`",
            f"- **Market Embedding**：`{spec.get('market_set_embedding_dim')}`",
            "- **Runtime Eligibility**：`research only`",
        ]
    if family == "inception_time":
        return [
            f"- **Model Family**：`{family}`",
            f"- **Inception Depth**：`{spec.get('inception_depth')}`",
            f"- **Filters**：`{spec.get('inception_filters')}`",
            f"- **Bottleneck Channels**：`{spec.get('inception_bottleneck_channels')}`",
            "- **Kernel Sizes**：`"
            + "/".join(str(value) for value in spec.get("inception_kernel_sizes") or [])
            + "`",
            f"- **Residual Every**：`{spec.get('inception_residual_every')} modules`",
        ]
    if family == "modern_tcn":
        return [
            f"- **Model Family**：`{family}`",
            f"- **ModernTCN Depth**：`{spec.get('modern_tcn_depth')}`",
            f"- **Channels**：`{spec.get('modern_tcn_channels')}`",
            f"- **Large Kernel**：`{spec.get('modern_tcn_kernel_size')}`",
            f"- **Pointwise Expansion**：`{spec.get('modern_tcn_expansion_ratio')}x`",
            f"- **Normalization**：`{spec.get('normalization')}`",
        ]
    return [
        "- **Branch Inputs**：`"
        + "+".join(spec.get("branch_input_representations") or ["level"])
        + "`",
        "- **Branch Channels**：`"
        + "/".join(
            str(value)
            for value in (spec.get("branch_channels") or [spec.get("channels")] * 3)
        )
        + "`",
        "- **Branch Dropouts**：`"
        + "/".join(
            f"{float(value):g}"
            for value in (spec.get("branch_dropouts") or [spec.get("dropout")] * 3)
        )
        + "`",
    ]


def _console_model_detail_lines(training: dict) -> list[str]:
    spec = training.get("model_spec") or {}
    family = str(spec.get("family") or "")
    if family == "moment_frozen_linear":
        external = training.get("external_pretrained_encoder") or {}
        checkpoint = external.get("checkpoint") or {}
        package = external.get("package") or {}
        dependencies = external.get("runtime_dependencies") or {}
        transformers_record = dependencies.get("transformers") or {}
        return [
            f"Model Family     : {family}",
            f"External Encoder : {external.get('repository', spec.get('moment_repository', '-'))}",
            f"Pinned Revision  : {external.get('resolved_revision', spec.get('moment_revision', '-'))}",
            f"Checkpoint SHA   : {checkpoint.get('sha256', '-')}",
            f"MOMENT Package   : {package.get('name', '-')} {package.get('version', '-')}",
            f"Transformers     : {transformers_record.get('version', '-')}",
            f"Input Resize     : {training.get('sequence_length', '-')} -> {spec.get('moment_input_length')} bars",
            f"Patch / Stride   : {spec.get('moment_patch_length')} / {spec.get('moment_patch_stride')}",
            f"Transformer      : {spec.get('moment_transformer_layers')} layers / {spec.get('moment_transformer_heads')} heads",
            f"Channel Mode     : {spec.get('moment_channel_aggregation')}",
            f"Patch Reduction  : {spec.get('moment_patch_reduction')}",
            f"Representation   : {spec.get('moment_embedding_dim')} per channel",
            "Encoder Training : official external pretraining; frozen downstream",
            "Downstream Head  : linear",
        ]
    if family == "mantis_v2_frozen_linear":
        external = training.get("external_pretrained_encoder") or {}
        checkpoint = external.get("checkpoint") or {}
        package = external.get("package") or {}
        return [
            f"Model Family     : {family}",
            f"External Encoder : {external.get('repository', spec.get('mantis_repository', '-'))}",
            f"Pinned Revision  : {external.get('resolved_revision', spec.get('mantis_revision', '-'))}",
            f"Checkpoint SHA   : {checkpoint.get('sha256', '-')}",
            f"Mantis Package   : {package.get('name', '-')} {package.get('version', '-')}",
            f"Input Resize     : {training.get('sequence_length', '-')} -> {spec.get('mantis_input_length')} bars",
            f"Channel Mode     : {spec.get('mantis_channel_aggregation')}",
            f"Transformer Out  : layer {spec.get('mantis_return_transformer_layer')} / {spec.get('mantis_output_token')}",
            f"Representation   : {spec.get('mantis_embedding_dim')} per channel",
            "Encoder Training : official external pretraining; frozen downstream",
            "Downstream Head  : linear",
        ]
    if family == "ts2vec_frozen_linear":
        pretraining = training.get("self_supervised_pretraining") or {}
        pretraining_manifest = pretraining.get("manifest") or {}
        pretraining_profile = pretraining_manifest.get("pretraining_profile") or {}
        return [
            f"Model Family    : {family}",
            f"Encoder Depth   : {spec.get('ts2vec_depth')}",
            f"Hidden Dims     : {spec.get('ts2vec_hidden_dims')}",
            f"Representation  : {spec.get('ts2vec_output_dims')}",
            "Encoder Training : Selection-only SSL; frozen downstream",
            "Downstream Head  : linear",
            f"Pretrain Profile : {pretraining_profile.get('name', '-')}",
            f"Pretrain Optimizer: {pretraining_profile.get('optimizer_name', '-')}",
            f"Pretrain E/B      : {pretraining_profile.get('epochs', '-')} / {pretraining_profile.get('batch_size', '-')}",
        ]
    if family == "patch_transformer":
        return [
            f"Model Family    : {family}",
            f"Patch / Stride  : {spec.get('patch_transformer_patch_size')} / {spec.get('patch_transformer_patch_stride')}",
            f"Embedding       : {spec.get('patch_transformer_embedding_dim')}",
            f"Depth / Heads   : {spec.get('patch_transformer_depth')} / {spec.get('patch_transformer_heads')}",
            f"MLP Dimensions  : {spec.get('patch_transformer_mlp_dim')}",
            f"Position        : {spec.get('patch_transformer_positional_encoding')}",
            f"Patch Pooling   : {spec.get('patch_transformer_pooling')}",
            "Encoder Training : supervised from scratch",
        ]
    if family == "inception_time_market_set":
        return [
            f"Model Family    : {family}",
            f"Candidate Depth : {spec.get('inception_depth')}",
            "Candidate Kernels: "
            + "/".join(str(value) for value in spec.get("inception_kernel_sizes") or []),
            f"Market History  : {spec.get('market_set_history_bars')} bars",
            f"Market Features : {'/'.join(spec.get('market_set_base_features') or [])}",
            f"Stock Embedding : {spec.get('market_set_stock_embedding_dim')}",
            f"Queries / Heads : {spec.get('market_set_query_count')} / {spec.get('market_set_attention_heads')}",
            f"Market Embedding: {spec.get('market_set_embedding_dim')}",
            "Runtime          : research only",
        ]
    if family == "inception_time":
        return [
            f"Model Family    : {family}",
            f"Inception Depth: {spec.get('inception_depth')}",
            f"Filters         : {spec.get('inception_filters')}",
            f"Bottleneck      : {spec.get('inception_bottleneck_channels')}",
            "Kernel Sizes    : "
            + "/".join(str(value) for value in spec.get("inception_kernel_sizes") or []),
            f"Residual Every  : {spec.get('inception_residual_every')} modules",
        ]
    if family == "modern_tcn":
        return [
            f"Model Family    : {family}",
            f"ModernTCN Depth : {spec.get('modern_tcn_depth')}",
            f"Channels        : {spec.get('modern_tcn_channels')}",
            f"Large Kernel    : {spec.get('modern_tcn_kernel_size')}",
            f"Expansion       : {spec.get('modern_tcn_expansion_ratio')}x",
            f"Normalization   : {spec.get('normalization')}",
        ]
    return [
        "Branch Inputs   : "
        + "+".join(spec.get("branch_input_representations") or ["level"]),
        "Branch Channels : "
        + "/".join(
            str(value)
            for value in (spec.get("branch_channels") or [spec.get("channels")] * 3)
        ),
        "Branch Dropouts : "
        + "/".join(
            f"{float(value):g}"
            for value in (spec.get("branch_dropouts") or [spec.get("dropout")] * 3)
        ),
    ]


def render_markdown_report(payload: dict) -> str:
    conclusion = payload["conclusion"]
    training = payload["training"]
    deployment = _deployment_presentation(payload)
    selection_summary = payload["split_summaries"].get(EVALUATION_SPLIT_SELECTION)
    oos_summary = payload["split_summaries"].get(EVALUATION_SPLIT_OOS)
    selected_epoch = training.get("selected_epoch")

    lines = [
        "# Breakout Quality 評估報表",
        "",
        f"- **Filter ID**：`{payload['filter_id']}`",
        "- **主要統計口徑**：Ticker/Date Group Weighted",
    ]
    if selection_summary is not None:
        lines.append(
            f"- **Selection**：{selection_summary['date_start']}～{selection_summary['date_end']}"
        )
    if oos_summary is not None:
        lines.append(f"- **OOS**：{oos_summary['date_start']}～{oos_summary['date_end']}")
    else:
        lines.extend(
            [
                f"- **OOS**：{_markdown_color('未納入本次報表', 'yellow')}",
                f"- {_markdown_color('因此不會輸出 OOS Confusion Matrix 與 Selection/OOS 指標比較；重新執行時請納入 OOS（預設）或明確使用 --include-oos。', 'yellow')}",
            ]
        )
    lines.extend(
        [
            f"- **Model Architecture**：`{training.get('model_architecture')}`",
            f"- **Experiment Profile**：`{training.get('experiment_profile')}`",
            f"- **LR Schedule**：`{training.get('lr_schedule_name')}`",
            f"- **LR Schedule Parameters**：`{training.get('lr_schedule_parameters') or '-'}`",
            f"- **Augmentation**：`{training.get('augmentation_name')}`",
            f"- **Augmentation Parameters**：`{training.get('augmentation_parameters') or '-'}`",
            f"- **Training Sampling**：`{training.get('training_sampling_mode')}`",
            *_markdown_model_detail_lines(training),
            (
                "- **Dataset Event Context**：`"
                + (
                    "enabled"
                    if (training.get("model_spec") or {}).get(
                        "use_dataset_context", True
                    )
                    else "disabled"
                )
                + "`"
            ),
            (
                "- **Derived Regime Context**：`"
                + (", ".join((training.get("model_spec") or {}).get("derived_context_features") or []) or "-")
                + "`"
            ),
            f"- **Trainable Parameters**：`{int(training.get('trainable_parameter_count') or 0):,}`",
            f"- **Frozen Parameters**：`{int(training.get('frozen_parameter_count') or 0):,}`",
            f"- **Total Parameters**：`{int(training.get('total_parameter_count') or training.get('trainable_parameter_count') or 0):,}`",
            f"- **Receptive Field**：`{(training.get('model_spec') or {}).get('receptive_field_bars')} bars`",
            f"- **Pooling**：`{'+'.join((training.get('model_spec') or {}).get('pooling') or [])}`",
            f"- **Threshold**：`{training.get('fixed_threshold')}`",
            f"- **Optimizer**：`{training.get('optimizer_name')}`",
            f"- **Learning Rate**：`{training.get('learning_rate')}`",
            f"- **Weight Decay**：`{training.get('weight_decay')}`",
            f"- **Gradient Clip Norm**：`{training.get('gradient_clip_norm')}`",
            f"- **Final Refit Mode**：`{(training.get('final_refit_plan') or {}).get('mode')}`",
            f"- **Class Weight Mode**：`{training.get('class_weight_mode')}`",
            f"- **Time Weight Mode**：`{training.get('time_weight_mode')}`",
            f"- **Training Weight Reduction**：`{training.get('training_weight_reduction')}`",
            f"- **Batch Size**：`{training.get('batch_size')}`",
            f"- **Random Seed**：`{training.get('seed')}`",
            f"- **Torch Device**：`{(training.get('torch_execution') or {}).get('resolved_device', '-')}`",
            f"- **Mixed Precision**：`{(training.get('torch_execution') or {}).get('mixed_precision_enabled', False)}`",
            f"- **Compute Dtype**：`{(training.get('torch_execution') or {}).get('autocast_dtype', 'float32')}`",
            f"- **Deterministic Algorithms**：`{(training.get('torch_execution') or {}).get('deterministic_algorithms', False)}`",
            f"- **TF32**：`{(training.get('torch_execution') or {}).get('allow_tf32', False)}`",
            "",
            "## 1. Epoch 選擇結果",
            "",
            f"- Epoch 上限：**{training.get('max_epochs')}**",
            f"- 實際完成 Epoch：**{training.get('completed_epoch_search')}**",
            f"- 選擇標準：`{training.get('epoch_selection_source')}`",
            f"- 最低 Validation Loss：{_markdown_color(_loss(training.get('best_validation_loss')), 'blue')}",
            f"- Early Stopping Patience：**{training.get('early_stopping_patience')}**",
            (
                (
                    f"- 最終模型：Inner Validation 選出 "
                    f"{_markdown_color('Epoch ' + str(selected_epoch), 'blue')}；"
                    "丟棄暫時模型後，使用完整 eligible Selection 依 "
                    f"`{(training.get('final_refit_plan') or {}).get('mode')}` 重訓 "
                    f"**{(training.get('final_refit_plan') or {}).get('actual_optimizer_steps')} steps** "
                    f"（約 **{float((training.get('final_refit_plan') or {}).get('equivalent_epochs') or 0):.3f} 個等效 Epoch**）。"
                )
                if training.get("inner_validation_used")
                else (
                    "- 最終模型：未使用 Inner Validation；使用完整 eligible Selection "
                    f"固定訓練 **{selected_epoch} Epochs**。"
                )
            ),
            "",
            *_markdown_epoch_table(training),
        ]
    )

    next_number = 2
    selection = payload["split_summaries"].get(EVALUATION_SPLIT_SELECTION)
    if selection is not None:
        lines.extend(_markdown_confusion_matrix(selection, "Selection", next_number))
        next_number += 1
    oos = payload["split_summaries"].get(EVALUATION_SPLIT_OOS)
    if oos is not None:
        lines.extend(_markdown_confusion_matrix(oos, "OOS", next_number))
        next_number += 1

    lines.extend(
        [
            f"## {next_number}. 各資料區段比較",
            "",
            *_markdown_split_table(payload),
        ]
    )
    if EVALUATION_SPLIT_VALIDATION in payload["split_summaries"]:
        lines.extend(
            [
                "\\* Validation 是完整 Selection 重訓後的區段診斷；真正用來選 Epoch 的未見 validation 結果以上方 Epoch 表為準。",
                "",
            ]
        )
    next_number += 1

    lines.extend(_markdown_ranking_table(payload, next_number))
    next_number += 1

    yearly_lines = _markdown_oos_yearly_tables(payload, next_number)
    if yearly_lines:
        lines.extend(yearly_lines)
        next_number += 1

    lines.extend(_markdown_oos_comprehensive_assessment(payload, next_number))

    lines.extend(
        [
            "## 指標白話說明",
            "",
            "- **原始 PASS**：所有原始訊號中，標籤為 PASS 的比例。",
            "- **原始 REJECT**：所有原始訊號中，標籤為 REJECT 的比例。",
            "- **模型 PASS**：所有訊號中，被模型判定為 PASS 的比例。",
            "- **模型 REJECT**：所有訊號中，被模型判定為 REJECT 的比例。",
            "- **PASS Precision**：所有模型 PASS 中，真正為原始 PASS 的比例。",
            "- **PASS Recall**：所有原始 PASS 中，被模型判定為 PASS 的比例。",
            "- **REJECT Specificity**：所有原始 REJECT 中，被模型判定為 REJECT 的比例。",
            "- **REJECT NPV**：所有模型 REJECT 中，真正為原始 REJECT 的比例。",
            "- **Accuracy**：所有訊號中，模型判定正確的比例。",
            "- **Precision 絕對**：PASS Precision 減去原始 PASS。",
            "- **Precision 相對**：PASS Precision 相對於原始 PASS 的增減幅度。",
            "",
            "## 使用限制",
            "",
            "- 本報表用於分類品質判讀，不等同於策略淨報酬、MDD、交易成本或資金容量評估。",
            "- Row-level metrics、完整政策契約與原始 confusion matrix 保存在同目錄 JSON。",
            "",
        ]
    )
    return "\n".join(lines)

def _console_epoch_section(training: dict, *, color: bool = False) -> list[str]:
    lines = _section("1. Epoch 選擇結果", color=color)
    selected_epoch = training.get("selected_epoch")
    lines.extend(
        [
            f"- Epoch 上限：{training.get('max_epochs')}",
            f"- 實際完成 Epoch：{training.get('completed_epoch_search')}",
            f"- 選擇標準：{training.get('epoch_selection_source')}",
            f"- 最低 Validation Loss：{_paint(_loss(training.get('best_validation_loss')), 'blue', enabled=color, bold=True)}",
            f"- Early Stopping Patience：{training.get('early_stopping_patience')}",
            (
                (
                    "- 最終模型：Inner Validation 選出 "
                    f"{_paint('Epoch ' + str(selected_epoch), 'blue', enabled=color, bold=True)}；"
                    "丟棄暫時模型後，使用完整 eligible Selection 依 "
                    f"{(training.get('final_refit_plan') or {}).get('mode')} 重訓 "
                    f"{(training.get('final_refit_plan') or {}).get('actual_optimizer_steps')} steps "
                    f"（約 {float((training.get('final_refit_plan') or {}).get('equivalent_epochs') or 0):.3f} 個等效 Epoch）。"
                )
                if training.get("inner_validation_used")
                else (
                    "- 最終模型：未使用 Inner Validation；使用完整 eligible Selection "
                    f"固定訓練 {selected_epoch} Epochs。"
                )
            ),
        ]
    )
    history = training.get("epoch_history") or []
    if history:
        lines.append("")
        epoch_rows = []
        for row in history:
            selected = row["decision"] == "最後選擇"
            decision = (
                _paint("★ 最後選擇", "blue", enabled=color, bold=True)
                if selected
                else row["decision"]
            )
            epoch_rows.append(
                [
                    _paint(row["epoch"], "blue", enabled=color, bold=True) if selected else row["epoch"],
                    _loss(row.get("batch_loss")),
                    _loss(row.get("train_loss")),
                    _pct(row.get("train_accuracy")),
                    _pct(row.get("train_pass_rate")),
                    _paint(_loss(row.get("validation_loss")), "blue", enabled=color, bold=True) if selected else _loss(row.get("validation_loss")),
                    _pct(row.get("validation_accuracy")),
                    _pct(row.get("validation_pass_rate")),
                    decision,
                ]
            )
        lines.extend(
            _render_ascii_table(
                [
                    "Epoch",
                    "Batch Loss",
                    "Train Loss",
                    "Train Acc.",
                    "Train Pass",
                    "Valid Loss",
                    "Valid Acc.",
                    "Valid Pass",
                    "結果",
                ],
                epoch_rows,
                aligns=["right", "right", "right", "right", "right", "right", "right", "right", "left"],
            )
        )
    else:
        lines.extend(["", "未啟用 inner validation；最終模型依固定 epochs 訓練。"])
    return lines

def _console_split_section(payload: dict, number: int, *, color: bool = False) -> list[str]:
    lines = _section(f"{number}. 各資料區段比較", color=color)
    rows = []
    for split_name in _split_order(payload):
        summary = payload["split_summaries"][split_name]
        details = _confusion_details(summary)
        delta_tone = _tone_for_delta(summary.get("precision_delta"))
        rows.append(
            [
                SPLIT_LABELS[split_name],
                f"{summary['date_start']}～{summary['date_end']}",
                _count(summary.get("group_count"), digits=0),
                _pct(details.get("base_pass_rate")),
                _pct(details.get("acceptance_rate")),
                _paint(_pct(details.get("precision")), delta_tone, enabled=color, bold=True),
                _paint(_pp(summary.get("precision_delta")), delta_tone, enabled=color, bold=True),
                _pct(details.get("recall")),
                _decimal(summary.get("avg_score")),
            ]
        )
    lines.extend(
        _render_ascii_table(
            [
                "區段",
                "日期",
                "Groups",
                REPORT_LABELS["original_pass_rate"],
                REPORT_LABELS["model_pass_rate"],
                REPORT_LABELS["pass_precision"],
                REPORT_LABELS["precision_absolute_lift"],
                REPORT_LABELS["pass_recall"],
                REPORT_LABELS["average_score"],
            ],
            rows,
            aligns=[
                "left", "left", "right", "right", "right", "right", "right", "right", "right",
            ],
        )
    )
    if EVALUATION_SPLIT_VALIDATION in payload["split_summaries"]:
        lines.extend(
            [
                "",
                _paint(
                    "* Validation 是完整 Selection 重訓後的區段診斷；真正的 Epoch 選擇結果以上方 Epoch 表為準。",
                    "gray",
                    enabled=color,
                ),
            ]
        )
    return lines


def _console_ranking_section(payload: dict, number: int, *, color: bool = False) -> list[str]:
    lines = _section(f"{number}. 排序與校準診斷", color=color)
    rows = []
    for split_name in _split_order(payload):
        ranking = _ranking_summary(payload["split_summaries"][split_name])
        rows.append(
            [
                SPLIT_LABELS[split_name],
                _decimal(ranking.get("average_precision_pr_auc"), 4),
                _pct(_coverage_precision(ranking, 0.50)),
                _pct(_coverage_precision(ranking, 0.60)),
                _pct(_coverage_precision(ranking, 0.70)),
                _pct(ranking.get("recall_at_precision_60")),
                _decimal(ranking.get("brier_score"), 4),
                _decimal(ranking.get("expected_calibration_error_10_bins"), 4),
            ]
        )
    lines.extend(
        _render_ascii_table(
            [
                "區段",
                "PR-AUC",
                "P@50%",
                "P@60%",
                "P@70%",
                "R@P60%",
                "Brier",
                "ECE",
            ],
            rows,
            aligns=["left", "right", "right", "right", "right", "right", "right", "right"],
        )
    )
    lines.extend(
        [
            "",
            _paint(
                "固定 coverage 指標只用來比較排序能力；不得依 OOS 診斷 threshold 回頭調整正式 threshold。",
                "yellow",
                enabled=color,
            ),
        ]
    )
    return lines

def _console_oos_yearly_section(
    payload: dict,
    number: int,
    *,
    color: bool = False,
) -> list[str]:
    yearly = payload.get("oos_yearly_summaries") or []
    if not yearly:
        return []
    lines = _section(f"{number}. OOS 年度診斷", color=color)
    classification_rows = []
    ranking_rows = []
    for row in yearly:
        year_label = f"{row['year']}{'*' if row['is_partial_calendar_year'] else ''}"
        classification_rows.append(
            [
                year_label,
                f"{row['date_start']}～{row['date_end']}",
                _count(row.get("group_count"), digits=0),
                _pct(row.get("base_pass_rate")),
                _pct(row.get("acceptance_rate")),
                _pct(row.get("pass_precision")),
                _pp(row.get("precision_delta")),
                _pct(row.get("pass_recall")),
                _decimal(row.get("avg_score")),
            ]
        )
        ranking = _ranking_summary(row)
        ranking_rows.append(
            [
                year_label,
                _decimal(ranking.get("average_precision_pr_auc"), 4),
                _pct(_coverage_precision(ranking, 0.50)),
                _pct(_coverage_precision(ranking, 0.60)),
                _pct(_coverage_precision(ranking, 0.70)),
                _pct(ranking.get("recall_at_precision_60")),
                _decimal(ranking.get("brier_score"), 4),
                _decimal(ranking.get("expected_calibration_error_10_bins"), 4),
            ]
        )
    lines.extend(
        _render_ascii_table(
            [
                "年度", "日期", "Groups", "原始 PASS", "模型 PASS",
                "PASS Precision", "Precision 絕對", "PASS Recall", "平均 Score",
            ],
            classification_rows,
            aligns=["left", "left", "right", "right", "right", "right", "right", "right", "right"],
        )
    )
    lines.extend(["", "排序與校準"])
    lines.extend(
        _render_ascii_table(
            ["年度", "PR-AUC", "P@50%", "P@60%", "P@70%", "R@P60%", "Brier", "ECE"],
            ranking_rows,
            aligns=["left", "right", "right", "right", "right", "right", "right", "right"],
        )
    )
    if any(bool(row.get("is_partial_calendar_year")) for row in yearly):
        lines.extend(["", "* 星號表示 OOS policy 僅涵蓋該年度的一部分。"])
    lines.extend(
        [
            "",
            _paint(
                "年度表只切分同一份固定 OOS score；不得依年度結果回頭調整 threshold、epochs 或模型。",
                "yellow",
                enabled=color,
            ),
        ]
    )
    return lines


def _console_confusion_section(summary: dict, label: str, number: int, *, color: bool = False) -> list[str]:
    details = _confusion_details(summary)
    delta_tone = _tone_for_delta(summary.get("precision_delta"))
    lines = _section(f"{number}. {label} Confusion Matrix", color=color)
    rows = [
        [
            "原始 PASS",
            _paint_multiline(
                f"正確保留 PASS\nTP = {_weighted_count(details['tp'])}",
                "green",
                enabled=color,
                bold=True,
            ),
            _paint_multiline(
                f"錯殺 PASS\nFN = {_weighted_count(details['fn'])}",
                "red",
                enabled=color,
                bold=True,
            ),
            f"原始PASS = {_pct(details['base_pass_rate'])}\nTP + FN = {_weighted_count(details['actual_pass'])}",
        ],
        [
            "原始 REJECT",
            _paint_multiline(
                f"錯誤保留 REJECT\nFP = {_weighted_count(details['fp'])}",
                "red",
                enabled=color,
                bold=True,
            ),
            _paint_multiline(
                f"正確拒絕 REJECT\nTN = {_weighted_count(details['tn'])}",
                "green",
                enabled=color,
                bold=True,
            ),
            f"原始REJECT = {_pct(details['base_reject_rate'])}\nFP + TN = {_weighted_count(details['actual_reject'])}",
        ],
        [
            "模型合計",
            f"TP + FP = {_weighted_count(details['predicted_pass'])}\n模型PASS = {_pct(details['acceptance_rate'])}",
            f"FN + TN = {_weighted_count(details['predicted_reject'])}\n模型REJECT = {_pct(details['rejection_rate'])}",
            f"全部 = {_weighted_count(details['total'])}",
        ],
    ]
    lines.extend(
        _render_ascii_table(
            ["原始結果 \\ 模型判定", "模型 PASS", "模型 REJECT", "原始合計"],
            rows,
            aligns=["left", "right", "right", "right"],
        )
    )
    classification_rows = [
        [REPORT_LABELS["pass_precision"], "TP ÷ (TP + FP)", _paint(_pct(details["precision"]), delta_tone, enabled=color, bold=True), "被保留的訊號中，有多少真的 PASS"],
        [REPORT_LABELS["pass_recall"], "TP ÷ (TP + FN)", _pct(details["recall"]), "真正 PASS 中，有多少被保留"],
        [REPORT_LABELS["reject_specificity"], "TN ÷ (TN + FP)", _pct(details["specificity"]), "真正 REJECT 中，有多少被正確拒絕"],
        [REPORT_LABELS["reject_npv"], "TN ÷ (TN + FN)", _pct(details["negative_predictive_value"]), "被拒絕的訊號中，有多少真的 REJECT"],
        [REPORT_LABELS["accuracy"], "(TP + TN) ÷ 全部", _pct(details["accuracy"]), "全部訊號中，模型判斷正確的比例"],
    ]
    lines.extend(["", "分類品質"])
    lines.extend(
        _render_ascii_table(
            ["指標", "公式", "結果", "解釋"],
            classification_rows,
            aligns=["left", "left", "right", "left"],
        )
    )
    effect_rows = [
        [REPORT_LABELS["precision_absolute_lift"], "PASS Precision − 原始 PASS", _paint(_pp(summary.get("precision_delta")), delta_tone, enabled=color, bold=True)],
        [REPORT_LABELS["precision_relative_lift"], "PASS Precision ÷ 原始 PASS − 1", _paint(_signed_pct(summary.get("precision_relative_change")), delta_tone, enabled=color, bold=True)],
    ]
    lines.extend(["", "篩選效果"])
    lines.extend(
        _render_ascii_table(
            ["指標", "公式", "結果"],
            effect_rows,
            aligns=["left", "left", "right"],
        )
    )
    if label == "OOS":
        judgement = (
            "模型大量判定為 REJECT，但模型 PASS 的品質未提高；"
            f"錯殺 {_pct(summary.get('false_rejection_rate'))} 的原始 PASS。"
            if float(summary.get("precision_delta") or 0.0) <= 0
            else "OOS PASS Precision 有提升；仍須檢查模型 PASS、PASS Recall 與策略層經濟效果。"
        )
        lines.extend(["", _paint("判讀：" + judgement, delta_tone, enabled=color, bold=True)])
    else:
        lines.extend(
            [
                "",
                _paint(
                    "判讀：Selection 內具有篩選能力；是否可部署仍由 OOS 泛化結果決定。",
                    "green",
                    enabled=color,
                    bold=True,
                ),
            ]
        )
    return lines

def _console_oos_comprehensive_section(
    payload: dict,
    number: int,
    *,
    color: bool = False,
) -> list[str]:
    rows = _oos_assessment_rows(payload)
    conclusion = payload["conclusion"]
    deployment = _deployment_presentation(payload)
    category_summary = _oos_category_summary(payload)
    lines = _section(f"{number}. OOS 綜合判定", color=color)

    if rows:
        table_rows = []
        previous_category = None
        for item in rows:
            category = item["category"] if item["category"] != previous_category else ""
            previous_category = item["category"]
            table_rows.append(
                [
                    category,
                    item["label"],
                    item["selection_text"],
                    _paint(item["oos_text"], item["tone"], enabled=color, bold=True),
                    _paint(item["difference_text"], item["tone"], enabled=color, bold=True),
                    _paint(item["judgement"], item["tone"], enabled=color),
                ]
            )
        lines.extend(
            _render_ascii_table(
                ["類別", "指標", "Selection", "OOS", "OOS - Selection", "判讀"],
                table_rows,
                aligns=["left", "left", "right", "right", "right", "left"],
            )
        )
        lines.append("")

    lines.extend(
        [
            "綜合判定：",
            f"- 主要成效：{_paint(category_summary['main'], category_summary['main_tone'], enabled=color, bold=True)}",
            f"- 過度篩選防線：{_paint(category_summary['guard'], category_summary['guard_tone'], enabled=color, bold=True)}",
            f"- 輔助診斷：{_paint(category_summary['auxiliary'], category_summary['auxiliary_tone'], enabled=color, bold=True)}",
            f"- 最終部署決策：{_paint(deployment['deployment_decision'], 'red' if conclusion['status'] == 'FAIL' else 'yellow', enabled=color, bold=True)}",
            f"- 研究限制：{_paint(deployment['retuning_limit'], 'yellow', enabled=color, bold=True)}",
        ]
    )
    return lines


def render_console_summary(payload: dict, *, color: bool = False) -> str:
    training = payload["training"]
    lines = [
        "",
        render_title(_paint("Breakout Quality 評估報表", "blue", enabled=color, bold=True)),
        f"Filter ID       : {payload['filter_id']}",
        "主要統計口徑    : Ticker/Date Group Weighted",
    ]
    selection_summary = payload["split_summaries"].get(EVALUATION_SPLIT_SELECTION)
    if selection_summary is not None:
        lines.append(
            f"Selection       : {selection_summary['date_start']} ～ {selection_summary['date_end']}"
        )
    oos_summary = payload["split_summaries"].get(EVALUATION_SPLIT_OOS)
    if oos_summary is not None:
        lines.append(f"OOS             : {oos_summary['date_start']} ～ {oos_summary['date_end']}")
    else:
        lines.append(
            "OOS             : "
            + _paint("未納入本次報表", "yellow", enabled=color, bold=True)
        )
        lines.append(
            _paint(
                "注意：因此不會輸出 OOS Confusion Matrix 與 Selection/OOS 指標比較；"
                "重新執行時請納入 OOS（預設）或明確使用 --include-oos。",
                "yellow",
                enabled=color,
                bold=True,
            )
        )
    lines.extend(
        [
            f"Model           : {training.get('model_architecture')}",
            f"Experiment      : {training.get('experiment_profile')}",
            f"LR Schedule     : {training.get('lr_schedule_name')}",
            f"LR Schedule Args: {training.get('lr_schedule_parameters') or '-'}",
            f"Augmentation    : {training.get('augmentation_name')}",
            f"Augmentation Args: {training.get('augmentation_parameters') or '-'}",
            f"Train Sampling  : {training.get('training_sampling_mode')}",
            *_console_model_detail_lines(training),
            (
                "Dataset Context : "
                + (
                    "enabled"
                    if (training.get("model_spec") or {}).get(
                        "use_dataset_context", True
                    )
                    else "disabled"
                )
            ),
            (
                "Derived Context : "
                + (", ".join((training.get("model_spec") or {}).get("derived_context_features") or []) or "-")
            ),
            f"Trainable Params: {int(training.get('trainable_parameter_count') or 0):,}",
            f"Frozen Params   : {int(training.get('frozen_parameter_count') or 0):,}",
            f"Total Params    : {int(training.get('total_parameter_count') or training.get('trainable_parameter_count') or 0):,}",
            f"Receptive Field : {(training.get('model_spec') or {}).get('receptive_field_bars')} bars",
            f"Pooling         : {'+'.join((training.get('model_spec') or {}).get('pooling') or [])}",
            f"Threshold       : {training.get('fixed_threshold')}",
            f"Optimizer       : {training.get('optimizer_name')}",
            f"Learning Rate   : {training.get('learning_rate')}",
            f"Weight Decay    : {training.get('weight_decay')}",
            f"Gradient Clip   : {training.get('gradient_clip_norm')}",
            f"Final Refit     : {(training.get('final_refit_plan') or {}).get('mode')}",
            f"Class Weight    : {training.get('class_weight_mode')}",
            f"Time Weight     : {training.get('time_weight_mode')}",
            f"Weight Reduce   : {training.get('training_weight_reduction')}",
            f"Batch Size      : {training.get('batch_size')}",
            f"Random Seed     : {training.get('seed')}",
            f"Torch Device    : {(training.get('torch_execution') or {}).get('resolved_device', '-')}",
            f"Mixed Precision : {(training.get('torch_execution') or {}).get('mixed_precision_enabled', False)}",
            f"Compute Dtype   : {(training.get('torch_execution') or {}).get('autocast_dtype', 'float32')}",
            f"Deterministic   : {(training.get('torch_execution') or {}).get('deterministic_algorithms', False)}",
            f"TF32            : {(training.get('torch_execution') or {}).get('allow_tf32', False)}",
        ]
    )
    lines.extend(_console_epoch_section(training, color=color))
    next_number = 2
    selection = payload["split_summaries"].get(EVALUATION_SPLIT_SELECTION)
    if selection is not None:
        lines.extend(_console_confusion_section(selection, "Selection", next_number, color=color))
        next_number += 1
    oos = payload["split_summaries"].get(EVALUATION_SPLIT_OOS)
    if oos is not None:
        lines.extend(_console_confusion_section(oos, "OOS", next_number, color=color))
        next_number += 1
    lines.extend(_console_split_section(payload, next_number, color=color))
    next_number += 1
    lines.extend(_console_ranking_section(payload, next_number, color=color))
    next_number += 1
    yearly_lines = _console_oos_yearly_section(payload, next_number, color=color)
    if yearly_lines:
        lines.extend(yearly_lines)
        next_number += 1
    lines.extend(_console_oos_comprehensive_section(payload, next_number, color=color))
    return "\n".join(lines)

def generate_report(
    *,
    filter_id: str,
    include_oos: bool,
    experiment_profile: str = BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
    score_path: str | Path | None = None,
) -> tuple[dict, Path, Path]:
    context = prepare_evaluation_context(
        filter_id=filter_id,
        experiment_profile=experiment_profile,
        score_path=score_path,
    )
    splits = []
    if bool(context["model_manifest"].get("inner_validation_used", False)):
        splits.extend([EVALUATION_SPLIT_TRAIN, EVALUATION_SPLIT_VALIDATION])
    splits.append(EVALUATION_SPLIT_SELECTION)
    if include_oos:
        splits.append(EVALUATION_SPLIT_OOS)
    metrics_by_split = {
        split_name: evaluate_split_from_context(context, split_name)
        for split_name in splits
    }

    payload = build_report_payload(metrics_by_split=metrics_by_split, context=context)
    ensure_filter_report_dir(
        PROJECT_ROOT, filter_id, experiment_profile=experiment_profile
    )
    markdown_path = resolve_filter_report_markdown_path(
        PROJECT_ROOT, filter_id, experiment_profile=experiment_profile
    )
    json_path = resolve_filter_report_json_path(
        PROJECT_ROOT, filter_id, experiment_profile=experiment_profile
    )
    markdown_path.write_text(render_markdown_report(payload), encoding="utf-8")
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload, markdown_path, json_path


def main(argv=None) -> int:
    args = parse_args(argv)
    payload, markdown_path, json_path = generate_report(
        filter_id=str(args.filter_id),
        include_oos=bool(args.include_oos),
        experiment_profile=str(args.experiment_profile),
        score_path=args.score_path,
    )
    print(render_console_summary(payload, color=console_color_enabled()))
    print_artifact_paths(
        (("Markdown", markdown_path), ("完整指標 JSON", json_path)),
        project_root=PROJECT_ROOT,
    )
    return 0


__all__ = [
    "build_report_payload",
    "generate_report",
    "render_console_summary",
    "render_markdown_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
