from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_utils import run_cli_entrypoint, enable_line_buffered_stdout, has_help_flag, resolve_cli_program_name, validate_cli_args
from core.test_suite_reporting import TEST_SUITE_STEP_LABELS, print_test_suite_human_summary
from tools.local_regression.formal_pipeline import DATASET_REQUIRED_STEPS, FORMAL_STEP_ORDER

SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
REGRESSION_STEP_ORDER = FORMAL_STEP_ORDER

STEP_LABELS = TEST_SUITE_STEP_LABELS


class ConsoleProgress:
    def __init__(self) -> None:
        self.spinner_index = 0
        self.last_render_ts = 0.0
        self.last_line_width = 0
        self.suite_started = time.time()

    def _build_bar(self, major_index: int, major_total: int, *, done: bool) -> str:
        width = 28
        completed_units = major_index if done else max(major_index - 1, 0)
        filled = int(width * completed_units / max(major_total, 1))
        return "█" * filled + "░" * (width - filled)

    def _render_inline(self, text: str) -> None:
        padded = text
        if self.last_line_width > len(text):
            padded = text + (" " * (self.last_line_width - len(text)))
        print(f"\r{padded}", end="", flush=True)
        self.last_line_width = max(self.last_line_width, len(text))

    def _finish_line(self, text: str) -> None:
        self._render_inline(text)
        print()
        self.last_line_width = 0

    def _step_text(self, payload: Dict[str, Any], body: str, *, done: bool, progress_index: int | None = None) -> str:
        major_index = int(payload.get("major_index", 0) or 0)
        major_total = int(payload.get("major_total", 1) or 1)
        display_index = major_index if progress_index is None else int(progress_index)
        elapsed_total = time.time() - self.suite_started
        return (
            f"[{display_index}/{major_total}] "
            f"{self._build_bar(display_index, major_total, done=done)} "
            f"{body} | 累計 {elapsed_total:.1f}s"
        )

    def __call__(self, event: str, payload: Dict[str, Any]) -> None:
        now = time.time()
        if event == "step_start":
            label = STEP_LABELS.get(payload["name"], payload["name"])
            self._render_inline(self._step_text(payload, f"準備執行 {label}", done=False))
            self.last_render_ts = 0.0
            return

        if event == "step_progress":
            if now - self.last_render_ts < 0.15:
                return
            self.last_render_ts = now
            spinner = SPINNER_FRAMES[self.spinner_index % len(SPINNER_FRAMES)]
            self.spinner_index += 1
            label = STEP_LABELS.get(payload["name"], payload["name"])
            self._render_inline(
                self._step_text(
                    payload,
                    f"{spinner} 執行中 {label} | {payload['elapsed_sec']:.1f}s",
                    done=False,
                )
            )
            return

        if event == "step_finish":
            symbol = "✓" if payload["status"] == "PASS" else "✗"
            label = STEP_LABELS.get(payload["name"], payload["name"])
            self._finish_line(
                self._step_text(
                    payload,
                    f"{symbol} {label} | {payload['status']} | {payload['duration_sec']:.2f}s",
                    done=True,
                )
            )
            return

        if event == "finalizing":
            self._render_inline(self._step_text(payload, "整理輸出與打包 bundle", done=False))
            return

        if event == "done":
            if payload["overall_status"] == "PASS":
                self._finish_line(self._step_text(payload, f"✓ 完成 | {payload['overall_status']}", done=True))
            else:
                failed_at = int(payload.get("failed_at_major_index", payload.get("major_index", 0)) or 0)
                self._finish_line(
                    self._step_text(
                        payload,
                        f"✗ 結束 | {payload['overall_status']}",
                        done=False,
                        progress_index=failed_at,
                    )
                )


def _print_human_summary(result: Dict[str, Any]) -> None:
    print_test_suite_human_summary(
        result,
        regression_step_order=REGRESSION_STEP_ORDER,
        dataset_required_steps=DATASET_REQUIRED_STEPS,
        step_labels=STEP_LABELS,
    )


def main(argv=None) -> int:
    enable_line_buffered_stdout()
    argv = sys.argv if argv is None else argv
    validate_cli_args(argv)
    if has_help_flag(argv):
        program_name = resolve_cli_program_name(argv, "apps/test_suite.py")
        print(f"用法: python {program_name}")
        print("說明: reduced 一鍵測試正式入口；會串接所有已實作測試（含 meta quality、dataset fingerprint、artifact integrity、single-stock compounding-capital contract、single-ticker compounding parity contract、exact-accounting ledger / cost-basis allocation / integer tick-limit / cash-risk boundary / single-vs-portfolio parity / display-derived contract、single-backtest legacy stats schema contract、single-backtest exact cash/equity path static contract、debug-backtest entry cash deduction static contract、shared display-money rounding helper static contract、real-case completed-trade rounding oracle contract、trade-rebuild shared rounding helper contract、debug forced-closeout exact total-pnl contract、synthetic-meta source-path binding contract、unit-display rounding-helper contract、debug-exit display-capital exact-ledger contract、debug-entry display-capital exact-total contract、debug-exit entry-capital fallback exact-ledger contract、debug-half-exit leg-return exact-allocated-cost contract、critical-helper single-source-of-truth guard、policy/config coverage-target contract、unsupported use_compounding guardrail、GPT 端 dynamic-test / formal-step bypass boundary contract、checklist convergence first-nonempty-title / summary-table-ID-order / `NEW`-only-first-occurrence / stale-validator-reference guard、summarize_result flattened-summary accessor contract、specific pass-only exception traceability contract、broad-exception traceability contract、GUI TclError fallback traceability contract、optional-dependency fallback traceability contract、GUI trade-box capital / round-trip display contract、package_zip root-bundle-preserve / commit-zip-test_suite orchestration contract、log_utils outputs-root create-path guard 與 memory tracker lifecycle contract），若失敗再依主控台建議用 run_all.py --only 重跑失敗步驟。")
        return 0

    from tools.local_regression.run_all import execute_all

    print("=== Test Suite (reduced) ===")
    print("[前置檢查] 先檢查目前 Python 環境是否已具備 requirements；不自動安裝。")
    progress = ConsoleProgress()
    result = execute_all(progress_callback=progress)
    _print_human_summary(result)
    return 0 if result["overall_status"] == "PASS" else 1


__all__ = ["main"]

if __name__ == "__main__":
    run_cli_entrypoint(main)
