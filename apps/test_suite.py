from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_utils import run_cli_entrypoint, enable_line_buffered_stdout, has_help_flag, resolve_cli_program_name, validate_cli_args

SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
STEP_LABELS = {
    "quick_gate": "quick gate",
    "consistency": "consistency",
    "chain_checks": "chain checks",
    "ml_smoke": "ml smoke",
    "preflight": "preflight",
    "dataset_prepare": "dataset prepare",
    "manifest": "manifest",
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
    master = {
        "scripts": result.get("scripts", []),
        "failures": result.get("failures", 0),
        "overall_status": result.get("overall_status", "FAIL"),
    }
    step_payloads = result.get("step_payloads", {})
    quick = step_payloads.get("quick_gate", {})
    consistency = step_payloads.get("consistency", {})
    chain = step_payloads.get("chain_checks", {})
    ml = step_payloads.get("ml_smoke", {})

    def _payload_issue_text(payload: Dict[str, Any]) -> str:
        if not payload:
            return "missing_summary_file"
        reasons = []
        if payload.get("error_type"):
            reasons.append("summary_unreadable")
        reported_status = str(payload.get("status", "") or "").strip()
        if reported_status and reported_status != "PASS":
            reasons.append(f"reported_status={reported_status}")
        return ", ".join(reasons)

    print("\n" + "=" * 78)
    print(" Test Suite 結果整理")
    print("=" * 78)
    print(f"整體狀態 : {result['overall_status']} | 失敗步驟 : {result['failures']}")
    manifest_payload = step_payloads.get("manifest", {})
    if manifest_payload:
        print(
            f"manifest    : FAIL | {manifest_payload.get('error_type', '')}: "
            f"{manifest_payload.get('error_message', '')}"
        )

    preflight = result.get("preflight", {})
    preflight_issue = _payload_issue_text(preflight)
    if preflight_issue:
        print(f"preflight   : FAIL | {preflight_issue}")
    else:
        failed_packages = preflight.get("failed_packages", [])
        failed_text = ", ".join(failed_packages) if failed_packages else "(none)"
        print(
            f"preflight   : {preflight.get('status', 'N/A')} | "
            f"{preflight.get('duration_sec', 0.0):.2f}s | failed_packages={failed_text}"
        )
    dataset_prepare = step_payloads.get("dataset_prepare", {})
    dataset_issue = _payload_issue_text(dataset_prepare)
    if dataset_issue:
        print(f"dataset prep: FAIL | {dataset_issue}")
    elif dataset_prepare.get("status") == "PASS":
        print(
            f"dataset prep: PASS | {dataset_prepare.get('duration_sec', 0.0):.2f}s | "
            f"csv_count={dataset_prepare.get('csv_count', 0)} | "
            f"source={dataset_prepare.get('source', '')}"
        )
    else:
        print(
            f"dataset prep: {dataset_prepare.get('status', 'N/A')} | "
            f"{dataset_prepare.get('duration_sec', 0.0):.2f}s | "
            f"{dataset_prepare.get('error_type', '')}: {dataset_prepare.get('error_message', '')}"
        )
    retention = result.get("retention", {})
    print(f"bundle 模式 : {result.get('bundle_mode', 'unknown')}")
    print(f"歷史 bundle : {result.get('archived_bundle', '')}")
    print(f"根目錄 bundle : {result['root_bundle_copy']}")
    print(f"bundle 檔數 : {len(result.get('bundle_entries', []))}")
    print(f"retention  : removed={retention.get('removed_count', 0)} | bytes={retention.get('removed_bytes', 0)}")

    script_map = {item["name"]: item for item in master.get("scripts", [])}
    if script_map:
        print("\n[步驟摘要]")
        for key in ("quick_gate", "consistency", "chain_checks", "ml_smoke"):
            item = script_map.get(key, {})
            print(f"- {STEP_LABELS.get(key, key):<13} {item.get('status', 'N/A'):<4} {item.get('duration_sec', 0.0):>6.2f}s")

    print("\n[重點結果]")
    quick_script = script_map.get("quick_gate", {})
    quick_issue = _payload_issue_text(quick) or ", ".join(quick_script.get("failure_reasons", []))
    if quick_issue:
        print(f"- quick gate : {quick_script.get('status', 'FAIL')} | {quick_issue}")
    else:
        print(f"- quick gate : step_count={quick.get('step_count', 0)} | failed_count={quick.get('failed_count', 0)}")

    consistency_script = script_map.get("consistency", {})
    consistency_issue = _payload_issue_text(consistency) or ", ".join(consistency_script.get("failure_reasons", []))
    if consistency_issue:
        print(f"- consistency: {consistency_script.get('status', 'FAIL')} | {consistency_issue}")
    else:
        print(
            "- consistency: "
            f"total_checks={consistency.get('total_checks', 0)} | "
            f"fail_count={consistency.get('fail_count', 0)} | "
            f"skip_count={consistency.get('skip_count', 0)} | "
            f"real_tickers={consistency.get('real_ticker_count', 0)}"
        )

    chain_script = script_map.get("chain_checks", {})
    chain_issue = _payload_issue_text(chain) or ", ".join(chain_script.get("failure_reasons", []))
    if chain_issue:
        print(f"- chain checks: {chain_script.get('status', 'FAIL')} | {chain_issue}")
    else:
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

    ml_script = script_map.get("ml_smoke", {})
    ml_issue = _payload_issue_text(ml) or ", ".join(ml_script.get("failure_reasons", []))
    if ml_issue:
        print(f"- ml smoke   : {ml_script.get('status', 'FAIL')} | {ml_issue}")
    else:
        print(
            "- ml smoke   : "
            f"db_trial_count={ml.get('db_trial_count', 0)} | "
            f"status={ml.get('status', 'N/A')}"
        )

    if result["overall_status"] != "PASS":
        failed_names = [item["name"] for item in master.get("scripts", []) if item.get("status") != "PASS"]
        if not failed_names:
            failed_names = [str(name) for name in result.get("failed_step_names", []) if str(name).strip()]
        if failed_names:
            print(f"\n失敗步驟 : {', '.join(failed_names)}")
        rerun_command = result.get("suggested_rerun_command", "")
        if rerun_command:
            print(f"建議重跑 : {rerun_command}")
    print("=" * 78)


def main(argv=None) -> int:
    enable_line_buffered_stdout()
    argv = sys.argv if argv is None else argv
    validate_cli_args(argv)
    if has_help_flag(argv):
        program_name = resolve_cli_program_name(argv, "apps/test_suite.py")
        print(f"用法: python {program_name}")
        print("說明: reduced 一鍵測試正式入口；先跑完整 regression，若失敗再依主控台建議用 run_all.py --only 重跑失敗步驟。")
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
