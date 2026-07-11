"""Generate readable breakout-quality research reports with comparable tables."""

from __future__ import annotations

import argparse
import json
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
SPLIT_LABELS = {
    EVALUATION_SPLIT_TRAIN: "Inner Train",
    EVALUATION_SPLIT_VALIDATION: "Validation*",
    EVALUATION_SPLIT_SELECTION: "Selection",
    EVALUATION_SPLIT_OOS: "OOS",
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
        default=False,
        help="是否納入最終 OOS；OOS 不得用於回頭調整 threshold、epochs 或模型",
    )
    parser.add_argument(
        "--score-path",
        default=None,
        help="自訂 research_scores.csv；通常不需指定",
    )
    return parser.parse_args(argv)


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
    for char in str(text):
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


def _section(title: str) -> list[str]:
    return ["", "=" * 96, f" {title}", "=" * 96, ""]


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
            "explanation": "OOS 缺少可比較的 PASS precision 或基準 PASS 比例。",
            "deployment_guidance": "先確認 score table、標籤與 split 工件完整。",
        }
    if float(delta) <= 0:
        return {
            "status": "FAIL",
            "title": "OOS 未顯示品質提升",
            "explanation": (
                f"模型保留後 PASS precision 為 {_pct(summary.get('pass_precision'))}，"
                f"低於全數放行基準 {_pct(summary.get('base_pass_rate'))}，"
                f"差異 {_pp(delta)}。"
            ),
            "deployment_guidance": (
                "維持 breakout quality runtime filter 關閉，不匯出 forward-OOS scores；"
                "不要用同一段 OOS 回頭調整 threshold、epochs 或 learning rate。"
            ),
        }
    return {
        "status": "PASS_WITH_REVIEW",
        "title": "OOS precision 有提升",
        "explanation": (
            f"模型保留後 PASS precision 為 {_pct(summary.get('pass_precision'))}，"
            f"高於全數放行基準 {_pct(summary.get('base_pass_rate'))}，"
            f"差異 {_pp(delta)}；仍須同時檢查保留率與 PASS recall。"
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
        decision = f"**{row['decision']}**" if row["decision"] == "最後選擇" else row["decision"]
        lines.append(
            "| {epoch} | {batch} | {train_loss} | {train_acc} | {train_pass} | {val_loss} | {val_acc} | {val_pass} | {decision} |".format(
                epoch=row["epoch"],
                batch=_loss(row.get("batch_loss")),
                train_loss=_loss(row.get("train_loss")),
                train_acc=_pct(row.get("train_accuracy")),
                train_pass=_pct(row.get("train_pass_rate")),
                val_loss=_loss(row.get("validation_loss")),
                val_acc=_pct(row.get("validation_accuracy")),
                val_pass=_pct(row.get("validation_pass_rate")),
                decision=decision,
            )
        )
    lines.append("")
    return lines


def _markdown_split_table(payload: dict) -> list[str]:
    lines = [
        "| 區段 | 日期 | Groups | 原始 PASS | 模型保留 | Precision | 絕對提升 | 相對提升 | PASS Recall | 錯殺 PASS | REJECT 辨識率 | Accuracy | 平均 Score |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split_name in _split_order(payload):
        summary = payload["split_summaries"][split_name]
        lines.append(
            "| {label} | {period} | {groups:,} | {base} | {accept} | {precision} | {delta} | {relative} | {recall} | {false_reject} | {specificity} | {accuracy} | {avg_score} |".format(
                label=SPLIT_LABELS[split_name],
                period=f"{summary['date_start']}～{summary['date_end']}",
                groups=int(summary.get("group_count") or 0),
                base=_pct(summary.get("base_pass_rate")),
                accept=_pct(summary.get("acceptance_rate")),
                precision=_pct(summary.get("pass_precision")),
                delta=_pp(summary.get("precision_delta")),
                relative=_signed_pct(summary.get("precision_relative_change")),
                recall=_pct(summary.get("pass_recall")),
                false_reject=_pct(summary.get("false_rejection_rate")),
                specificity=_pct(summary.get("reject_specificity")),
                accuracy=_pct(summary.get("accuracy")),
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
        "acceptance_rate": _ratio(predicted_pass, total),
        "rejection_rate": _ratio(predicted_reject, total),
        "accuracy": _ratio(tp + tn, total),
    }


def _markdown_confusion_matrix(summary: dict, label: str) -> list[str]:
    details = _confusion_details(summary)
    lines = [
        f"## {label} Confusion Matrix",
        "",
        "統計口徑：`ticker/date group weighted`；列為實際結果，欄為模型預測。",
        "",
        "| 實際＼預測 | 預測 PASS／模型保留 | 預測 REJECT／模型拒絕 | 實際合計 |",
        "|---|---:|---:|---:|",
        (
            "| **實際 PASS**<br>Base PASS {base_pass} | "
            "**TP = {tp}**<br>正確保留 PASS<br>Recall {recall} | "
            "**FN = {fn}**<br>錯殺真正 PASS<br>錯殺率 {false_reject} | "
            "{actual_pass}<br>占全部 {base_pass} |"
        ).format(
            base_pass=_pct(details["base_pass_rate"]),
            tp=_weighted_count(details["tp"]),
            recall=_pct(details["recall"]),
            fn=_weighted_count(details["fn"]),
            false_reject=_pct(details["false_rejection_rate"]),
            actual_pass=_weighted_count(details["actual_pass"]),
        ),
        (
            "| **實際 REJECT**<br>Base REJECT {base_reject} | "
            "**FP = {fp}**<br>錯誤保留 REJECT<br>誤放率 {false_positive} | "
            "**TN = {tn}**<br>正確拒絕 REJECT<br>辨識率 {specificity} | "
            "{actual_reject}<br>占全部 {base_reject} |"
        ).format(
            base_reject=_pct(details["base_reject_rate"]),
            fp=_weighted_count(details["fp"]),
            false_positive=_pct(details["false_positive_rate"]),
            tn=_weighted_count(details["tn"]),
            specificity=_pct(details["specificity"]),
            actual_reject=_weighted_count(details["actual_reject"]),
        ),
        (
            "| **預測合計** | {pred_pass}<br>保留率 {acceptance}"
            "<br>Precision {precision} | {pred_reject}<br>拒絕率 {rejection} | "
            "{total}<br>Accuracy {accuracy} |"
        ).format(
            pred_pass=_weighted_count(details["predicted_pass"]),
            acceptance=_pct(details["acceptance_rate"]),
            precision=_pct(details["precision"]),
            pred_reject=_weighted_count(details["predicted_reject"]),
            rejection=_pct(details["rejection_rate"]),
            total=_weighted_count(details["total"]),
            accuracy=_pct(details["accuracy"]),
        ),
        "",
        "### 品質變化",
        "",
        "| 原始 PASS 比例 | 保留後 Precision | 絕對變化 | 相對變化 |",
        "|---:|---:|---:|---:|",
        "| {base} | {precision} | {delta} | {relative} |".format(
            base=_pct(summary.get("base_pass_rate")),
            precision=_pct(summary.get("pass_precision")),
            delta=_pp(summary.get("precision_delta")),
            relative=_signed_pct(summary.get("precision_relative_change")),
        ),
        "",
    ]
    if label == "OOS":
        if float(summary.get("precision_delta") or 0.0) <= 0:
            lines.extend(
                [
                    "判讀：模型在 OOS 中大量拒絕訊號，但保留下來的訊號品質未提高；"
                    f"同時錯殺 {_pct(summary.get('false_rejection_rate'))} 的真正 PASS。",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "判讀：OOS 保留後 precision 高於原始 PASS 比例；仍須一起檢查保留率、PASS recall 與策略層經濟效果。",
                    "",
                ]
            )
    else:
        lines.extend(
            [
                "判讀：此區段中模型具有篩選能力，但是否能部署仍由 OOS 泛化結果決定。",
                "",
            ]
        )
    return lines


def _markdown_selection_oos_difference(payload: dict) -> list[str]:
    selection = payload["split_summaries"].get(EVALUATION_SPLIT_SELECTION)
    oos = payload["split_summaries"].get(EVALUATION_SPLIT_OOS)
    if selection is None or oos is None:
        return []
    rows = [
        ("原始 PASS 比例", selection.get("base_pass_rate"), oos.get("base_pass_rate"), "pct"),
        ("保留後 Precision", selection.get("pass_precision"), oos.get("pass_precision"), "pct"),
        ("模型保留率", selection.get("acceptance_rate"), oos.get("acceptance_rate"), "pct"),
        ("PASS Recall", selection.get("pass_recall"), oos.get("pass_recall"), "pct"),
        ("平均 Score", selection.get("avg_score"), oos.get("avg_score"), "decimal"),
        ("Precision 絕對提升", selection.get("precision_delta"), oos.get("precision_delta"), "pp"),
    ]
    lines = [
        "## Selection 與 OOS 差異",
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


def render_markdown_report(payload: dict) -> str:
    conclusion = payload["conclusion"]
    training = payload["training"]
    lines = [
        "# Breakout Quality 評估報表",
        "",
        "## 1. 最終結論",
        "",
        "| 項目 | 結果 |",
        "|---|---|",
        f"| Filter ID | `{payload['filter_id']}` |",
        f"| 評估門檻 | `{training.get('fixed_threshold')}` |",
        f"| Selected Epoch | `{training.get('selected_epoch')}` |",
        f"| 最終判定 | **[{conclusion['status']}] {conclusion['title']}** |",
        f"| 主要原因 | {conclusion['explanation']} |",
        f"| 部署建議 | {conclusion['deployment_guidance']} |",
        "",
        "## 2. Epoch 訓練比較與最後選擇",
        "",
        "| 項目 | 結果 |",
        "|---|---|",
        f"| Training Mode | `{training.get('training_mode')}` |",
        f"| Epoch 上限 | `{training.get('max_epochs')}` |",
        f"| 實際完成 Epoch | `{training.get('completed_epoch_search')}` |",
        f"| 選擇標準 | `{training.get('epoch_selection_source')}` |",
        f"| 最佳 Epoch | **`{training.get('selected_epoch')}`** |",
        f"| 最低 Validation Loss | `{_loss(training.get('best_validation_loss'))}` |",
        f"| Best Epoch Validation Accuracy | `{_pct(training.get('best_validation_accuracy'))}` |",
        f"| Best Epoch Validation Pass Rate | `{_pct(training.get('best_validation_pass_rate'))}` |",
        f"| Early Stopping Patience | `{training.get('early_stopping_patience')}` |",
        f"| Fixed Threshold | `{training.get('fixed_threshold')}` |",
        f"| Learning Rate | `{training.get('learning_rate')}` |",
        f"| Batch Size | `{training.get('batch_size')}` |",
        f"| Random Seed | `{training.get('seed')}` |",
        "",
        *_markdown_epoch_table(training),
        (
            f"最後模型：選出 Epoch {training.get('selected_epoch')} 後，丟棄暫時模型，"
            f"再用完整 eligible Selection 正式重訓 {training.get('selected_epoch')} Epochs。"
        ),
        "",
        "## 3. 各資料區段比較",
        "",
        *_markdown_split_table(payload),
    ]
    if EVALUATION_SPLIT_VALIDATION in payload["split_summaries"]:
        lines.extend(
            [
                "\\* Validation 是完整 Selection 重訓後的區段診斷；真正用來選 Epoch 的未見 validation 結果以上方 Epoch 表為準。",
                "",
            ]
        )
    selection = payload["split_summaries"].get(EVALUATION_SPLIT_SELECTION)
    if selection is not None:
        lines.extend(_markdown_confusion_matrix(selection, "Selection"))
    oos = payload["split_summaries"].get(EVALUATION_SPLIT_OOS)
    if oos is not None:
        lines.extend(_markdown_confusion_matrix(oos, "OOS"))
    lines.extend(_markdown_selection_oos_difference(payload))
    lines.extend(
        [
            "## 最終部署判定",
            "",
            "| 判定項目 | 結果 |",
            "|---|---|",
            f"| Inner Validation 是否正常運作 | {'是' if training.get('inner_validation_used') else '未啟用'} |",
            f"| 是否選出 Best Epoch | {('是，Epoch ' + str(training.get('selected_epoch'))) if training.get('inner_validation_used') and training.get('selected_epoch') is not None else ('未使用，固定 Epoch ' + str(training.get('selected_epoch')))} |",
            f"| OOS 是否有品質提升 | {'是' if oos is not None and float(oos.get('precision_delta') or 0.0) > 0 else '否'} |",
            f"| 是否建議啟用 Runtime Filter | {'否' if conclusion['status'] == 'FAIL' else '需進一步策略驗證'} |",
            f"| 是否建議匯出 Forward-OOS Scores | {'否' if conclusion['status'] == 'FAIL' else '待策略層驗證'} |",
            "",
            f"**最終結論：[{conclusion['status']}] {conclusion['title']}。**",
            "",
            "## 指標白話說明",
            "",
            "- **原始 PASS 比例**：完全不過濾、全部放行時，真正 PASS 的比例。",
            "- **模型保留率**：分數達固定 threshold、被模型保留下來的訊號比例。",
            "- **Precision**：模型保留的訊號中，真正 PASS 的比例。",
            "- **PASS Recall**：所有真正 PASS 中，被模型保留下來的比例。",
            "- **錯殺率**：所有真正 PASS 中，被模型錯誤拒絕的比例。",
            "- **REJECT 辨識率**：所有真正 REJECT 中，被模型正確拒絕的比例。",
            "",
            "## 使用限制",
            "",
            f"- {payload['audit']['oos_reuse_warning']}",
            "- 本報表用於分類品質判讀，不等同於策略淨報酬、MDD、交易成本或資金容量評估。",
            "- Row-level metrics、完整政策契約與原始 confusion matrix 保存在同目錄 JSON。",
            "",
        ]
    )
    return "\n".join(lines)


def _console_epoch_section(training: dict) -> list[str]:
    lines = _section("1. Epoch 選擇結果")
    summary_rows = [
        ["Epoch 上限", training.get("max_epochs")],
        ["實際完成 Epoch", training.get("completed_epoch_search")],
        ["選擇標準", training.get("epoch_selection_source")],
        ["最後選擇", f"Epoch {training.get('selected_epoch')}"],
        ["最低 Validation Loss", _loss(training.get("best_validation_loss"))],
        ["Early Stopping Patience", training.get("early_stopping_patience")],
    ]
    lines.extend(_render_ascii_table(["項目", "結果"], summary_rows))
    history = training.get("epoch_history") or []
    if history:
        lines.append("")
        epoch_rows = []
        for row in history:
            decision = f"★ {row['decision']}" if row["decision"] == "最後選擇" else row["decision"]
            epoch_rows.append(
                [
                    row["epoch"],
                    _loss(row.get("batch_loss")),
                    _loss(row.get("train_loss")),
                    _pct(row.get("train_accuracy")),
                    _pct(row.get("train_pass_rate")),
                    _loss(row.get("validation_loss")),
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
        lines.extend(
            [
                "",
                f"最後模型：選出 Epoch {training.get('selected_epoch')} 後，丟棄暫時模型，",
                f"再使用完整 eligible Selection 正式重訓 {training.get('selected_epoch')} Epochs。",
            ]
        )
    else:
        lines.extend(["", "未啟用 inner validation；最終模型依固定 epochs 訓練。"])
    return lines


def _console_split_section(payload: dict) -> list[str]:
    lines = _section("2. 各資料區段比較")
    rows = []
    for split_name in _split_order(payload):
        summary = payload["split_summaries"][split_name]
        rows.append(
            [
                f"{SPLIT_LABELS[split_name]}\n{summary['date_start']}～{summary['date_end']}",
                _count(summary.get("group_count"), digits=0),
                _pct(summary.get("base_pass_rate")),
                _pct(summary.get("acceptance_rate")),
                _pct(summary.get("pass_precision")),
                _pp(summary.get("precision_delta")),
                _pct(summary.get("pass_recall")),
                _pct(summary.get("false_rejection_rate")),
                _pct(summary.get("reject_specificity")),
                _pct(summary.get("accuracy")),
                _decimal(summary.get("avg_score")),
            ]
        )
    lines.extend(
        _render_ascii_table(
            [
                "區段 / 日期",
                "Groups",
                "原始PASS",
                "模型保留",
                "Precision",
                "絕對提升",
                "PASS Recall",
                "錯殺PASS",
                "REJECT辨識",
                "Accuracy",
                "平均Score",
            ],
            rows,
            aligns=["left", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right"],
        )
    )
    if EVALUATION_SPLIT_VALIDATION in payload["split_summaries"]:
        lines.extend(
            [
                "",
                "* Validation 是完整 Selection 重訓後的區段診斷；真正的 Epoch 選擇結果以上方 Epoch 表為準。",
            ]
        )
    return lines


def _console_confusion_section(summary: dict, label: str, number: int) -> list[str]:
    details = _confusion_details(summary)
    lines = _section(f"{number}. {label} Confusion Matrix")
    lines.extend(["統計口徑：Ticker/Date Group Weighted", "列 = 實際結果；欄 = 模型預測", ""])
    rows = [
        [
            f"實際 PASS\nBase PASS {_pct(details['base_pass_rate'])}",
            f"TP = {_weighted_count(details['tp'])}\n正確保留 PASS\nRecall {_pct(details['recall'])}",
            f"FN = {_weighted_count(details['fn'])}\n錯殺真正 PASS\n錯殺率 {_pct(details['false_rejection_rate'])}",
            f"{_weighted_count(details['actual_pass'])}\n占全部 {_pct(details['base_pass_rate'])}",
        ],
        [
            f"實際 REJECT\nBase REJECT {_pct(details['base_reject_rate'])}",
            f"FP = {_weighted_count(details['fp'])}\n錯誤保留 REJECT\n誤放率 {_pct(details['false_positive_rate'])}",
            f"TN = {_weighted_count(details['tn'])}\n正確拒絕 REJECT\n辨識率 {_pct(details['specificity'])}",
            f"{_weighted_count(details['actual_reject'])}\n占全部 {_pct(details['base_reject_rate'])}",
        ],
        [
            "預測合計",
            f"{_weighted_count(details['predicted_pass'])}\n保留率 {_pct(details['acceptance_rate'])}\nPrecision {_pct(details['precision'])}",
            f"{_weighted_count(details['predicted_reject'])}\n拒絕率 {_pct(details['rejection_rate'])}",
            f"{_weighted_count(details['total'])}\nAccuracy {_pct(details['accuracy'])}",
        ],
    ]
    lines.extend(
        _render_ascii_table(
            ["實際 \\ 預測", "預測 PASS／模型保留", "預測 REJECT／模型拒絕", "實際合計"],
            rows,
            aligns=["left", "right", "right", "right"],
        )
    )
    lines.extend(
        [
            "",
            f"品質變化：原始 PASS {_pct(summary.get('base_pass_rate'))} → "
            f"保留後 Precision {_pct(summary.get('pass_precision'))} "
            f"({_pp(summary.get('precision_delta'))}；{_signed_pct(summary.get('precision_relative_change'))})",
        ]
    )
    if label == "OOS":
        lines.append(
            "判讀："
            + (
                "模型大量拒絕訊號，但保留下來的訊號品質未提高；"
                f"錯殺 {_pct(summary.get('false_rejection_rate'))} 的真正 PASS。"
                if float(summary.get("precision_delta") or 0.0) <= 0
                else "OOS precision 有提升；仍須檢查保留率、PASS recall 與策略層經濟效果。"
            )
        )
    else:
        lines.append("判讀：Selection 內具有篩選能力；是否可部署仍由 OOS 泛化結果決定。")
    return lines


def _console_difference_section(payload: dict, number: int) -> list[str]:
    selection = payload["split_summaries"].get(EVALUATION_SPLIT_SELECTION)
    oos = payload["split_summaries"].get(EVALUATION_SPLIT_OOS)
    if selection is None or oos is None:
        return []
    lines = _section(f"{number}. Selection 與 OOS 差異")
    rows = [
        ["原始 PASS 比例", _pct(selection.get("base_pass_rate")), _pct(oos.get("base_pass_rate")), _pp(float(oos.get("base_pass_rate")) - float(selection.get("base_pass_rate")))],
        ["保留後 Precision", _pct(selection.get("pass_precision")), _pct(oos.get("pass_precision")), _pp(float(oos.get("pass_precision")) - float(selection.get("pass_precision")))],
        ["模型保留率", _pct(selection.get("acceptance_rate")), _pct(oos.get("acceptance_rate")), _pp(float(oos.get("acceptance_rate")) - float(selection.get("acceptance_rate")))],
        ["PASS Recall", _pct(selection.get("pass_recall")), _pct(oos.get("pass_recall")), _pp(float(oos.get("pass_recall")) - float(selection.get("pass_recall")))],
        ["平均 Score", _decimal(selection.get("avg_score")), _decimal(oos.get("avg_score")), _decimal(float(oos.get("avg_score")) - float(selection.get("avg_score")))],
        ["Precision 絕對提升", _pp(selection.get("precision_delta")), _pp(oos.get("precision_delta")), _pp(float(oos.get("precision_delta")) - float(selection.get("precision_delta")))],
    ]
    lines.extend(
        _render_ascii_table(
            ["指標", "Selection", "OOS", "OOS - Selection"],
            rows,
            aligns=["left", "right", "right", "right"],
        )
    )
    return lines


def _console_deployment_section(payload: dict, number: int) -> list[str]:
    conclusion = payload["conclusion"]
    training = payload["training"]
    oos = payload["split_summaries"].get(EVALUATION_SPLIT_OOS)
    oos_improved = oos is not None and float(oos.get("precision_delta") or 0.0) > 0
    lines = _section(f"{number}. 最終部署判定")
    rows = [
        ["Inner Validation 是否正常運作", "是" if training.get("inner_validation_used") else "未啟用"],
        [
            "是否選出 Best Epoch",
            (
                f"是，Epoch {training.get('selected_epoch')}"
                if training.get("inner_validation_used") and training.get("selected_epoch") is not None
                else f"未使用，固定 Epoch {training.get('selected_epoch')}"
            ),
        ],
        ["OOS 是否有品質提升", "是" if oos_improved else "否"],
        ["是否建議啟用 Runtime Filter", "否" if conclusion["status"] == "FAIL" else "需進一步策略驗證"],
        ["是否建議匯出 Forward-OOS Scores", "否" if conclusion["status"] == "FAIL" else "待策略層驗證"],
    ]
    lines.extend(_render_ascii_table(["判定項目", "結果"], rows))
    lines.extend(
        [
            "",
            f"最終結論：[{conclusion['status']}] {conclusion['title']}。",
            conclusion["explanation"],
            f"部署建議：{conclusion['deployment_guidance']}",
        ]
    )
    return lines


def render_console_summary(payload: dict) -> str:
    conclusion = payload["conclusion"]
    training = payload["training"]
    lines = [
        "",
        "=" * 96,
        " Breakout Quality 評估報表",
        "=" * 96,
        f"Filter ID       : {payload['filter_id']}",
        f"Threshold       : {training.get('fixed_threshold')}",
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
    lines.extend(
        [
        f"最終判定        : [{conclusion['status']}] {conclusion['title']}",
        f"部署建議        : {conclusion['deployment_guidance']}",
        ]
    )
    lines.extend(_console_epoch_section(training))
    lines.extend(_console_split_section(payload))
    next_number = 3
    selection = payload["split_summaries"].get(EVALUATION_SPLIT_SELECTION)
    if selection is not None:
        lines.extend(_console_confusion_section(selection, "Selection", next_number))
        next_number += 1
    oos = payload["split_summaries"].get(EVALUATION_SPLIT_OOS)
    if oos is not None:
        lines.extend(_console_confusion_section(oos, "OOS", next_number))
        next_number += 1
    difference = _console_difference_section(payload, next_number)
    if difference:
        lines.extend(difference)
        next_number += 1
    lines.extend(_console_deployment_section(payload, next_number))
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
    print(render_console_summary(payload))
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
