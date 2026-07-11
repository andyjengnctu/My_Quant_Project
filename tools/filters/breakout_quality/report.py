"""Generate readable breakout-quality research reports with comparable tables."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from filters.breakout_quality.contract import DEFAULT_FILTER_ID
from filters.breakout_quality.paths import (
    ensure_filter_report_dir,
    resolve_filter_report_json_path,
    resolve_filter_report_markdown_path,
)
from tools.filters.breakout_quality.evaluate import (
    EVALUATION_SPLIT_OOS,
    EVALUATION_SPLIT_SELECTION,
    EVALUATION_SPLIT_TRAIN,
    EVALUATION_SPLIT_VALIDATION,
    evaluate_split_from_context,
    prepare_evaluation_context,
)


REPORT_SCHEMA_VERSION = 2
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
ANSI_COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "blue": "\033[96m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "red": "\033[91m",
    "gray": "\033[90m",
}
MARKDOWN_COLORS = {
    "blue": "#42A5F5",
    "green": "#188038",
    "yellow": "#B06000",
    "red": "#C62828",
    "gray": "#667085",
}
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



def _console_color_enabled() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM", "").strip().lower() == "dumb":
        return False
    stream = getattr(sys, "stdout", None)
    return bool(stream is not None and hasattr(stream, "isatty") and stream.isatty())


def _paint(text: object, tone: str, *, enabled: bool, bold: bool = False) -> str:
    raw = str(text)
    if not enabled:
        return raw
    prefix = ANSI_COLORS.get(tone, "")
    if bold:
        prefix = ANSI_COLORS["bold"] + prefix
    return f"{prefix}{raw}{ANSI_COLORS['reset']}"


def _markdown_color(text: object, tone: str, *, bold: bool = True) -> str:
    raw = str(text)
    color = MARKDOWN_COLORS.get(tone, MARKDOWN_COLORS["gray"])
    weight = "font-weight:700;" if bold else ""
    return f'<span style="color:{color};{weight}">{raw}</span>'


def _tone_for_delta(value: float | None) -> str:
    if value is None:
        return "gray"
    return "green" if float(value) > 0 else "red" if float(value) < 0 else "yellow"

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

    return {
        "training_mode": manifest.get("training_mode"),
        "inner_validation_used": bool(manifest.get("inner_validation_used", False)),
        "max_epochs": manifest.get("max_epochs"),
        "selected_epoch": manifest.get("selected_epoch"),
        "completed_epoch_search": epoch_selection.get("completed_epochs"),
        "best_validation_loss": epoch_selection.get("best_validation_loss"),
        "best_validation_accuracy": best_metrics.get("accuracy"),
        "best_validation_pass_rate": best_metrics.get("pass_rate"),
        "fixed_threshold": manifest.get("fixed_evaluation_threshold"),
        "learning_rate": manifest.get("learning_rate"),
        "batch_size": manifest.get("batch_size"),
        "seed": manifest.get("seed"),
        "early_stopping_enabled": early_stopping_enabled,
        "early_stopping_patience": manifest.get("early_stopping_patience"),
        "early_stopping_min_delta": manifest.get("early_stopping_min_delta"),
        "epoch_selection_source": manifest.get("epoch_selection_source"),
        "epoch_history": normalized_history,
    }


def build_report_payload(*, metrics_by_split: dict[str, dict], context: dict) -> dict:
    summaries = {
        split_name: _metric_summary(metrics)
        for split_name, metrics in metrics_by_split.items()
    }
    conclusion = _oos_conclusion(summaries.get(EVALUATION_SPLIT_OOS))
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "filter_id": context["filter_id"],
        "headline_basis": "ticker_date_group_weighted",
        "conclusion": conclusion,
        "training": _training_summary(context["model_manifest"]),
        "split_summaries": summaries,
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
            f"{REPORT_LABELS['pass_recall']} | {REPORT_LABELS['reject_specificity']} | "
            f"{REPORT_LABELS['reject_npv']} | {REPORT_LABELS['accuracy']} | "
            f"{REPORT_LABELS['precision_absolute_lift']} | "
            f"{REPORT_LABELS['precision_relative_lift']} | {REPORT_LABELS['average_score']} |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
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
                "{precision} | {recall} | {specificity} | {npv} | {accuracy} | "
                "{absolute_lift} | {relative_lift} | {avg_score} |"
            ).format(
                label=label,
                period=f"{summary['date_start']}～{summary['date_end']}",
                groups=int(summary.get("group_count") or 0),
                original_pass=_pct(details.get("base_pass_rate")),
                model_pass=_pct(details.get("acceptance_rate")),
                precision=_markdown_color(_pct(details.get("precision")), delta_tone),
                recall=_pct(details.get("recall")),
                specificity=_pct(details.get("specificity")),
                npv=_pct(details.get("negative_predictive_value")),
                accuracy=_pct(details.get("accuracy")),
                absolute_lift=_markdown_color(_pp(summary.get("precision_delta")), delta_tone),
                relative_lift=_markdown_color(
                    _signed_pct(summary.get("precision_relative_change")), delta_tone
                ),
                avg_score=_decimal(summary.get("avg_score")),
            )
        )
    lines.append("")
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

def _markdown_selection_oos_difference(payload: dict, number: int) -> list[str]:
    selection = payload["split_summaries"].get(EVALUATION_SPLIT_SELECTION)
    oos = payload["split_summaries"].get(EVALUATION_SPLIT_OOS)
    if selection is None or oos is None:
        return []
    rows = [
        (REPORT_LABELS["original_pass_rate"], selection.get("base_pass_rate"), oos.get("base_pass_rate"), "pct"),
        (REPORT_LABELS["model_pass_rate"], selection.get("acceptance_rate"), oos.get("acceptance_rate"), "pct"),
        (REPORT_LABELS["pass_precision"], selection.get("pass_precision"), oos.get("pass_precision"), "pct"),
        (REPORT_LABELS["pass_recall"], selection.get("pass_recall"), oos.get("pass_recall"), "pct"),
        (REPORT_LABELS["reject_specificity"], selection.get("reject_specificity"), oos.get("reject_specificity"), "pct"),
        (
            REPORT_LABELS["reject_npv"],
            _confusion_details(selection).get("negative_predictive_value"),
            _confusion_details(oos).get("negative_predictive_value"),
            "pct",
        ),
        (REPORT_LABELS["accuracy"], selection.get("accuracy"), oos.get("accuracy"), "pct"),
        (REPORT_LABELS["precision_absolute_lift"], selection.get("precision_delta"), oos.get("precision_delta"), "pp"),
        (REPORT_LABELS["precision_relative_lift"], selection.get("precision_relative_change"), oos.get("precision_relative_change"), "pct_signed"),
        (REPORT_LABELS["average_score"], selection.get("avg_score"), oos.get("avg_score"), "decimal"),
    ]
    lines = [
        f"## {number}. Selection 與 OOS 差異",
        "",
        "| 指標 | Selection | OOS | OOS - Selection |",
        "|---|---:|---:|---:|",
    ]
    for label, selection_value, oos_value, kind in rows:
        difference = (
            float(oos_value) - float(selection_value)
            if selection_value is not None and oos_value is not None
            else None
        )
        if kind == "pct":
            selection_text = _pct(selection_value)
            oos_text = _pct(oos_value)
            difference_text = _pp(difference)
        elif kind == "pct_signed":
            selection_text = _signed_pct(selection_value)
            oos_text = _signed_pct(oos_value)
            difference_text = _pp(difference)
        elif kind == "pp":
            selection_text = _pp(selection_value)
            oos_text = _pp(oos_value)
            difference_text = _pp(difference)
        else:
            selection_text = _decimal(selection_value)
            oos_text = _decimal(oos_value)
            difference_text = _decimal(difference)
        lines.append(
            f"| {label} | {selection_text} | {oos_text} | {difference_text} |"
        )
    lines.append("")
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
                f"- {_markdown_color('因此不會輸出 OOS Confusion Matrix 與 Selection/OOS 差異；重新執行時請納入 OOS（預設）或明確使用 --include-oos。', 'yellow')}",
            ]
        )
    lines.extend(
        [
            f"- **Threshold**：`{training.get('fixed_threshold')}`",
            f"- **Learning Rate**：`{training.get('learning_rate')}`",
            f"- **Batch Size**：`{training.get('batch_size')}`",
            f"- **Random Seed**：`{training.get('seed')}`",
            "",
            "## 1. Epoch 選擇結果",
            "",
            f"- Epoch 上限：**{training.get('max_epochs')}**",
            f"- 實際完成 Epoch：**{training.get('completed_epoch_search')}**",
            f"- 選擇標準：`{training.get('epoch_selection_source')}`",
            f"- 最低 Validation Loss：{_markdown_color(_loss(training.get('best_validation_loss')), 'blue')}",
            f"- Early Stopping Patience：**{training.get('early_stopping_patience')}**",
            (
                f"- 最終模型：Inner Validation 選出 "
                f"{_markdown_color('Epoch ' + str(selected_epoch), 'blue')}；"
                "丟棄暫時模型後，再使用完整 eligible Selection 正式重訓 "
                f"**{selected_epoch} Epochs**。"
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

    difference_lines = _markdown_selection_oos_difference(payload, next_number)
    if difference_lines:
        lines.extend(difference_lines)
        next_number += 1

    oos_tone = "green" if deployment["oos_improved"] else "red" if oos is not None else "yellow"
    lines.extend(
        [
            f"## {next_number}. 最終部署判定",
            "",
            f"### {_markdown_color('[' + conclusion['status'] + '] ' + conclusion['title'], deployment['conclusion_tone'])}",
            "",
            f"- OOS 判定依據：{_markdown_color(deployment['oos_result'], oos_tone)}",
            f"- 部署決策：{_markdown_color(deployment['deployment_decision'], 'red' if conclusion['status'] == 'FAIL' else 'yellow')}",
            f"- 研究限制：{_markdown_color(deployment['retuning_limit'], 'yellow')}",
            "",
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
                "- 最終模型：Inner Validation 選出 "
                f"{_paint('Epoch ' + str(selected_epoch), 'blue', enabled=color, bold=True)}；"
                "丟棄暫時模型後，再使用完整 eligible Selection 正式重訓 "
                f"{selected_epoch} Epochs。"
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
                _pct(details.get("recall")),
                _pct(details.get("specificity")),
                _pct(details.get("negative_predictive_value")),
                _pct(details.get("accuracy")),
                _paint(_pp(summary.get("precision_delta")), delta_tone, enabled=color, bold=True),
                _paint(
                    _signed_pct(summary.get("precision_relative_change")),
                    delta_tone,
                    enabled=color,
                    bold=True,
                ),
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
                REPORT_LABELS["pass_recall"],
                REPORT_LABELS["reject_specificity"],
                REPORT_LABELS["reject_npv"],
                REPORT_LABELS["accuracy"],
                REPORT_LABELS["precision_absolute_lift"],
                REPORT_LABELS["precision_relative_lift"],
                REPORT_LABELS["average_score"],
            ],
            rows,
            aligns=[
                "left", "left", "right", "right", "right", "right", "right",
                "right", "right", "right", "right", "right", "right",
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

def _console_confusion_section(summary: dict, label: str, number: int, *, color: bool = False) -> list[str]:
    details = _confusion_details(summary)
    delta_tone = _tone_for_delta(summary.get("precision_delta"))
    lines = _section(f"{number}. {label} Confusion Matrix", color=color)
    rows = [
        [
            "原始 PASS",
            _paint(
                f"正確保留 PASS\nTP = {_weighted_count(details['tp'])}",
                "green",
                enabled=color,
                bold=True,
            ),
            _paint(
                f"錯殺 PASS\nFN = {_weighted_count(details['fn'])}",
                "red",
                enabled=color,
                bold=True,
            ),
            f"原始PASS = {_pct(details['base_pass_rate'])}\nTP + FN = {_weighted_count(details['actual_pass'])}",
        ],
        [
            "原始 REJECT",
            _paint(
                f"錯誤保留 REJECT\nFP = {_weighted_count(details['fp'])}",
                "red",
                enabled=color,
                bold=True,
            ),
            _paint(
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

def _console_difference_section(payload: dict, number: int, *, color: bool = False) -> list[str]:
    selection = payload["split_summaries"].get(EVALUATION_SPLIT_SELECTION)
    oos = payload["split_summaries"].get(EVALUATION_SPLIT_OOS)
    if selection is None or oos is None:
        return []
    lines = _section(f"{number}. Selection 與 OOS 差異", color=color)
    raw_rows = [
        [REPORT_LABELS["original_pass_rate"], selection.get("base_pass_rate"), oos.get("base_pass_rate"), "pct"],
        [REPORT_LABELS["model_pass_rate"], selection.get("acceptance_rate"), oos.get("acceptance_rate"), "pct"],
        [REPORT_LABELS["pass_precision"], selection.get("pass_precision"), oos.get("pass_precision"), "pct"],
        [REPORT_LABELS["pass_recall"], selection.get("pass_recall"), oos.get("pass_recall"), "pct"],
        [REPORT_LABELS["reject_specificity"], selection.get("reject_specificity"), oos.get("reject_specificity"), "pct"],
        [
            REPORT_LABELS["reject_npv"],
            _confusion_details(selection).get("negative_predictive_value"),
            _confusion_details(oos).get("negative_predictive_value"),
            "pct",
        ],
        [REPORT_LABELS["accuracy"], selection.get("accuracy"), oos.get("accuracy"), "pct"],
        [REPORT_LABELS["precision_absolute_lift"], selection.get("precision_delta"), oos.get("precision_delta"), "pp"],
        [REPORT_LABELS["precision_relative_lift"], selection.get("precision_relative_change"), oos.get("precision_relative_change"), "pct_signed"],
        [REPORT_LABELS["average_score"], selection.get("avg_score"), oos.get("avg_score"), "decimal"],
    ]
    rows = []
    for label, selection_value, oos_value, kind in raw_rows:
        difference = float(oos_value) - float(selection_value)
        tone = _tone_for_delta(difference)
        if kind == "pct":
            selection_text = _pct(selection_value)
            oos_text = _pct(oos_value)
            difference_text = _pp(difference)
        elif kind == "pct_signed":
            selection_text = _signed_pct(selection_value)
            oos_text = _signed_pct(oos_value)
            difference_text = _pp(difference)
        elif kind == "pp":
            selection_text = _pp(selection_value)
            oos_text = _pp(oos_value)
            difference_text = _pp(difference)
        else:
            selection_text = _decimal(selection_value)
            oos_text = _decimal(oos_value)
            difference_text = _decimal(difference)
        rows.append(
            [
                label,
                selection_text,
                _paint(oos_text, tone, enabled=color, bold=True),
                _paint(difference_text, tone, enabled=color, bold=True),
            ]
        )
    lines.extend(
        _render_ascii_table(
            ["指標", "Selection", "OOS", "OOS - Selection"],
            rows,
            aligns=["left", "right", "right", "right"],
        )
    )
    return lines

def _console_deployment_section(payload: dict, number: int, *, color: bool = False) -> list[str]:
    conclusion = payload["conclusion"]
    deployment = _deployment_presentation(payload)
    oos = payload["split_summaries"].get(EVALUATION_SPLIT_OOS)
    oos_tone = "green" if deployment["oos_improved"] else "red" if oos is not None else "yellow"
    lines = _section(f"{number}. 最終部署判定", color=color)
    lines.extend(
        [
            _paint(
                f"最終判定：[{conclusion['status']}] {conclusion['title']}",
                deployment["conclusion_tone"],
                enabled=color,
                bold=True,
            ),
            f"- OOS 判定依據：{_paint(deployment['oos_result'], oos_tone, enabled=color, bold=True)}",
            f"- 部署決策：{_paint(deployment['deployment_decision'], 'red' if conclusion['status'] == 'FAIL' else 'yellow', enabled=color, bold=True)}",
            f"- 研究限制：{_paint(deployment['retuning_limit'], 'yellow', enabled=color, bold=True)}",
        ]
    )
    return lines

def render_console_summary(payload: dict, *, color: bool = False) -> str:
    training = payload["training"]
    lines = [
        "",
        "=" * 96,
        " " + _paint("Breakout Quality 評估報表", "blue", enabled=color, bold=True),
        "=" * 96,
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
                "注意：因此不會輸出 OOS Confusion Matrix 與 Selection/OOS 差異；"
                "重新執行時請納入 OOS（預設）或明確使用 --include-oos。",
                "yellow",
                enabled=color,
                bold=True,
            )
        )
    lines.extend(
        [
            f"Threshold       : {training.get('fixed_threshold')}",
            f"Learning Rate   : {training.get('learning_rate')}",
            f"Batch Size      : {training.get('batch_size')}",
            f"Random Seed     : {training.get('seed')}",
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
    difference = _console_difference_section(payload, next_number, color=color)
    if difference:
        lines.extend(difference)
        next_number += 1
    lines.extend(_console_deployment_section(payload, next_number, color=color))
    return "\n".join(lines)

def generate_report(
    *,
    filter_id: str,
    include_oos: bool,
    score_path: str | Path | None = None,
) -> tuple[dict, Path, Path]:
    context = prepare_evaluation_context(
        filter_id=filter_id,
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
    ensure_filter_report_dir(PROJECT_ROOT, filter_id)
    markdown_path = resolve_filter_report_markdown_path(PROJECT_ROOT, filter_id)
    json_path = resolve_filter_report_json_path(PROJECT_ROOT, filter_id)
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
        score_path=args.score_path,
    )
    print(render_console_summary(payload, color=_console_color_enabled()))
    print("\n" + "=" * 96)
    print(" 報表檔案")
    print("=" * 96)
    print(f"Markdown 報表：{markdown_path}")
    print(f"完整指標 JSON：{json_path}")
    return 0


__all__ = [
    "build_report_payload",
    "generate_report",
    "render_console_summary",
    "render_markdown_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
