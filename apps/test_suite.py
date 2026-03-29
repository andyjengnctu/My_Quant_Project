from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_utils import enable_line_buffered_stdout, has_help_flag
from tools.local_regression.common import read_json
from tools.local_regression.run_all import execute_all

SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
STEP_LABELS = {
    "quick_gate": "quick gate",
    "consistency": "consistency",
    "chain_checks": "chain checks",
    "ml_smoke": "ml smoke",
}


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

    def _step_text(self, payload: Dict[str, Any], body: str, *, done: bool) -> str:
        major_index = int(payload.get("major_index", 0) or 0)
        major_total = int(payload.get("major_total", 1) or 1)
        elapsed_total = time.time() - self.suite_started
        return (
            f"[{major_index}/{major_total}] "
            f"{self._build_bar(major_index, major_total, done=done)} "
            f"{body} | 累計 {elapsed_total:.1f}s"
        )

    def __call__(self, event: str, payload: Dict[str, Any]) -> None:
        now = time.time()
        if event == "dataset_ready":
            line = self._step_text(
                payload,
                f"資料就緒 | {payload['dataset_info'].get('csv_count', 0)} 檔 reduced",
                done=True,
            )
            self._finish_line(line)
            return

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
            symbol = "✓" if payload["overall_status"] == "PASS" else "✗"
            self._finish_line(self._step_text(payload, f"{symbol} 完成 | {payload['overall_status']}", done=True))


def _read_summary_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return read_json(path)


def _print_human_summary(result: Dict[str, Any]) -> None:
    latest_dir = Path(result["latest_dir"])
    master = _read_summary_if_exists(latest_dir / "master_summary.json")
    quick = _read_summary_if_exists(latest_dir / "quick_gate_summary.json")
    consistency = _read_summary_if_exists(latest_dir / "validate_consistency_summary.json")
    chain = _read_summary_if_exists(latest_dir / "chain_summary.json")
    ml = _read_summary_if_exists(latest_dir / "ml_smoke_summary.json")

    print("\n" + "=" * 78)
    print(" Test Suite 結果整理")
    print("=" * 78)
    print(f"整體狀態 : {result['overall_status']} | 失敗步驟 : {result['failures']}")
    print(f"latest 目錄 : {result['latest_dir']}")
    print(f"根目錄 bundle : {result['root_bundle_copy']}")

    script_map = {item["name"]: item for item in master.get("scripts", [])}
    if script_map:
        print("\n[步驟摘要]")
        for key in ("quick_gate", "consistency", "chain_checks", "ml_smoke"):
            item = script_map.get(key, {})
            print(f"- {STEP_LABELS.get(key, key):<13} {item.get('status', 'N/A'):<4} {item.get('duration_sec', 0.0):>6.2f}s")

    print("\n[重點結果]")
    print(f"- quick gate : step_count={quick.get('step_count', 0)} | failed_count={quick.get('failed_count', 0)}")
    print(
        "- consistency: "
        f"total_checks={consistency.get('total_checks', 0)} | "
        f"fail_count={consistency.get('fail_count', 0)} | "
        f"skip_count={consistency.get('skip_count', 0)} | "
        f"real_tickers={consistency.get('real_ticker_count', 0)}"
    )

    portfolio_snapshot = chain.get("portfolio_snapshot", {})
    highlights = chain.get("highlights", {})
    print(
        "- chain checks: "
        f"ticker_count={chain.get('ticker_count', 0)} | "
        f"traded_ticker_count={highlights.get('traded_ticker_count', 0)} | "
        f"missed_buy_ticker_count={highlights.get('missed_buy_ticker_count', 0)} | "
        f"trade_rows={portfolio_snapshot.get('trade_rows', 0)} | "
        f"reserved_buy_fill_rate={portfolio_snapshot.get('reserved_buy_fill_rate', 0.0)}"
    )
    blocked_by_counts = highlights.get("blocked_by_counts", {})
    if blocked_by_counts:
        blocked_preview = ", ".join(f"{key}:{value}" for key, value in list(blocked_by_counts.items())[:6])
        print(f"  blocked_by : {blocked_preview}")
    print(
        "- ml smoke   : "
        f"db_trial_count={ml.get('db_trial_count', 0)} | "
        f"status={ml.get('status', 'N/A')}"
    )

    if result["overall_status"] != "PASS":
        failed_names = [item["name"] for item in master.get("scripts", []) if item.get("status") != "PASS"]
        if failed_names:
            print(f"\n失敗步驟 : {', '.join(failed_names)}")
    print("=" * 78)


def main() -> int:
    enable_line_buffered_stdout()
    if has_help_flag(sys.argv):
        print("用法: python apps/test_suite.py")
        print("說明: reduced 一鍵測試入口，依序執行 quick gate / consistency / chain checks / ml smoke。")
        return 0

    print("=== Test Suite (reduced) ===")
    progress = ConsoleProgress()
    result = execute_all(progress_callback=progress)
    _print_human_summary(result)
    return 0 if result["overall_status"] == "PASS" else 1


__all__ = ["main"]

if __name__ == "__main__":
    raise SystemExit(main())
