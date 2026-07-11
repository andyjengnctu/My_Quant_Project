"""Generate concise, explanatory breakout-quality research reports."""

from __future__ import annotations

import argparse
import json
import sys
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


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "產生 Breakout Quality 易讀研究報表；終端只顯示短摘要，"
            "完整 metrics 另存 JSON，解釋版另存 Markdown"
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


def _metric_summary(metrics: dict) -> dict:
    headline = metrics["ticker_date_group_weighted"]
    base = headline.get("base_pass_rate")
    precision = headline.get("pass_precision")
    lift = headline.get("precision_lift_vs_all_pass")
    return {
        "split": metrics["split"],
        "date_start": metrics["selected_date_range"]["start"],
        "date_end": metrics["selected_date_range"]["end"],
        "group_count": headline.get("group_count"),
        "row_count": headline.get("row_count"),
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
    return {
        "training_mode": manifest.get("training_mode"),
        "inner_validation_used": bool(manifest.get("inner_validation_used", False)),
        "max_epochs": manifest.get("max_epochs"),
        "selected_epoch": manifest.get("selected_epoch"),
        "completed_epoch_search": epoch_selection.get("completed_epochs"),
        "best_validation_loss": epoch_selection.get("best_validation_loss"),
        "best_validation_accuracy": best_metrics.get("accuracy"),
        "fixed_threshold": manifest.get("fixed_evaluation_threshold"),
        "learning_rate": manifest.get("learning_rate"),
        "batch_size": manifest.get("batch_size"),
        "seed": manifest.get("seed"),
    }


def build_report_payload(*, metrics_by_split: dict[str, dict], context: dict) -> dict:
    summaries = {
        split_name: _metric_summary(metrics)
        for split_name, metrics in metrics_by_split.items()
    }
    conclusion = _oos_conclusion(summaries.get(EVALUATION_SPLIT_OOS))
    return {
        "schema_version": 1,
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


def _summary_table(payload: dict) -> list[str]:
    rows = []
    labels = {
        EVALUATION_SPLIT_TRAIN: "Inner Train",
        EVALUATION_SPLIT_VALIDATION: "Validation*",
        EVALUATION_SPLIT_SELECTION: "Selection",
        EVALUATION_SPLIT_OOS: "OOS",
    }
    for split_name in (
        EVALUATION_SPLIT_TRAIN,
        EVALUATION_SPLIT_VALIDATION,
        EVALUATION_SPLIT_SELECTION,
        EVALUATION_SPLIT_OOS,
    ):
        summary = payload["split_summaries"].get(split_name)
        if summary is None:
            continue
        rows.append(
            "| {label} | {period} | {groups:,} | {base} | {accept} | {precision} | {delta} | {relative} | {recall} | {false_reject} |".format(
                label=labels[split_name],
                period=f"{summary['date_start']}～{summary['date_end']}",
                groups=int(summary.get("group_count") or 0),
                base=_pct(summary.get("base_pass_rate")),
                accept=_pct(summary.get("acceptance_rate")),
                precision=_pct(summary.get("pass_precision")),
                delta=_pp(summary.get("precision_delta")),
                relative=_signed_pct(summary.get("precision_relative_change")),
                recall=_pct(summary.get("pass_recall")),
                false_reject=_pct(summary.get("false_rejection_rate")),
            )
        )
    return rows


def render_markdown_report(payload: dict) -> str:
    conclusion = payload["conclusion"]
    training = payload["training"]
    lines = [
        "# Breakout Quality 評估報表",
        "",
        f"- Filter ID：`{payload['filter_id']}`",
        f"- 產生時間（UTC）：`{payload['generated_at_utc']}`",
        "- 主要判讀口徑：`ticker/date group weighted`",
        "",
        "## 一頁結論",
        "",
        f"**[{conclusion['status']}] {conclusion['title']}**",
        "",
        conclusion["explanation"],
        "",
        f"部署建議：{conclusion['deployment_guidance']}",
        "",
        "## 主要指標",
        "",
        "| 區段 | 日期 | Groups | 原始 PASS 比例 | 模型保留率 | 保留後 PASS precision | 絕對提升 | 相對提升 | PASS recall | 錯殺真正 PASS |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *_summary_table(payload),
        "",
    ]
    if bool(training.get("inner_validation_used")):
        lines.extend(
            [
                "\\* Validation 列是最終完整 Selection 重訓模型的期間診斷；真正用來選 epoch 的未見 validation 指標列於下方。",
                "",
            ]
        )
    lines.extend(
        [
            "## 訓練設定與 Epoch 選擇",
            "",
            f"- Training mode：`{training.get('training_mode')}`",
            f"- Inner validation：`{training.get('inner_validation_used')}`",
            f"- Epoch 上限：`{training.get('max_epochs')}`",
            f"- Selected epoch：`{training.get('selected_epoch')}`",
        ]
    )
    if bool(training.get("inner_validation_used")):
        lines.extend(
            [
                f"- Epoch search 實際完成：`{training.get('completed_epoch_search')}`",
                f"- Best validation loss：`{training.get('best_validation_loss')}`",
                f"- Best validation accuracy：`{_pct(training.get('best_validation_accuracy'))}`",
            ]
        )
    else:
        lines.append("- Epoch selection：未使用 inner validation；依固定 epochs 訓練。")
    lines.extend(
        [
            f"- Fixed threshold：`{training.get('fixed_threshold')}`",
            f"- Learning rate：`{training.get('learning_rate')}`",
            f"- Batch size：`{training.get('batch_size')}`",
            f"- Random seed：`{training.get('seed')}`",
            "",
            "## 指標白話說明",
            "",
            "- **原始 PASS 比例**：完全不過濾、全部放行時，真正 PASS 的比例。",
            "- **模型保留率**：分數達固定 threshold、被模型保留下來的訊號比例。",
            "- **保留後 PASS precision**：模型保留的訊號中，真正 PASS 的比例；這是品質 filter 的主要指標。",
            "- **絕對提升**：保留後 precision 減去原始 PASS 比例，以百分點表示。",
            "- **相對提升**：precision 相對於原始 PASS 比例的百分比變化。",
            "- **PASS recall**：所有真正 PASS 中，被模型保留下來的比例。",
            "- **錯殺真正 PASS**：所有真正 PASS 中，被模型錯誤拒絕的比例；等於 1 - PASS recall。",
            "",
            "## 使用限制",
            "",
            f"- {payload['audit']['oos_reuse_warning']}",
            "- 本報表用於模型分類品質判讀，不等同於策略淨報酬、MDD、交易成本或資金容量評估。",
            "- 完整 confusion matrix、row-level metrics 與政策契約保存在同目錄 JSON。",
            "",
        ]
    )
    return "\n".join(lines)


def render_console_summary(payload: dict) -> str:
    conclusion = payload["conclusion"]
    lines = [
        "",
        "=== Breakout Quality 易讀摘要 ===",
        f"Filter ID：{payload['filter_id']}",
        f"結論：[{conclusion['status']}] {conclusion['title']}",
        conclusion["explanation"],
    ]
    for split_name, label in (
        (EVALUATION_SPLIT_SELECTION, "Selection"),
        (EVALUATION_SPLIT_OOS, "OOS"),
    ):
        summary = payload["split_summaries"].get(split_name)
        if summary is None:
            continue
        lines.append(
            f"{label}：原始 PASS {_pct(summary.get('base_pass_rate'))} → "
            f"保留後 precision {_pct(summary.get('pass_precision'))} "
            f"({_pp(summary.get('precision_delta'))})；"
            f"保留率 {_pct(summary.get('acceptance_rate'))}；"
            f"PASS recall {_pct(summary.get('pass_recall'))}"
        )
    lines.append(f"部署建議：{conclusion['deployment_guidance']}")
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
    print(f"易讀報表：{markdown_path}")
    print(f"完整數據：{json_path}")
    return 0


__all__ = [
    "build_report_payload",
    "generate_report",
    "render_console_summary",
    "render_markdown_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
