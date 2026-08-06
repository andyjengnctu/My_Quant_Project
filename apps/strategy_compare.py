"""策略績效比較獨立正式入口。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.strategy_compare import get_strategy_comparison_settings
from core.runtime_utils import is_interactive_console, run_cli_entrypoint
from filters.breakout_quality.strategy_comparison import (
    collect_artifact_status,
    render_execution_plan,
    run_strategy_comparison,
    show_strategy_comparison_status,
)


def _print_menu() -> None:
    print("\n=== 策略績效比較 ===")
    print("[1/Enter] 執行目前比較設定")
    print("[2] 查看設定、工件與預計動作")
    print("[0] 離開")



def _run_current_comparison(*, confirm: bool) -> dict:
    settings = get_strategy_comparison_settings()
    status = collect_artifact_status(settings=settings)
    print("\n" + render_execution_plan(settings=settings, status=status))
    if status["overall_status"] == "BLOCKED":
        raise RuntimeError(
            "目前缺少不可自動產生的上游工件；請先查看狀態頁。"
        )
    if confirm and settings.preparation.require_confirmation:
        try:
            choice = input("👉 按 Enter 執行；輸入 0 返回：").strip().lower()
        except EOFError:
            print("\n輸入已結束，本次不執行。")
            return {}
        if choice in {"0", "q", "quit", "exit"}:
            print("已取消本次執行。")
            return {}
        if choice not in {"", "1"}:
            print("輸入無效，本次不執行。")
            return {}
    return run_strategy_comparison(status=status, auto_prepare=True)

def _interactive_menu() -> int:
    while True:
        _print_menu()
        try:
            raw_choice = input("👉 請選擇：").strip().lower()
        except EOFError:
            print("\n輸入已結束。")
            return 0
        choice = "1" if raw_choice == "" else raw_choice
        if choice in {"0", "q", "quit", "exit"}:
            return 0
        try:
            if choice == "1":
                _run_current_comparison(confirm=True)
            elif choice == "2":
                show_strategy_comparison_status()
            else:
                print("選項無效，請按 Enter 或輸入 0～2。")
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"[錯誤] {type(exc).__name__}: {exc}")
        except KeyboardInterrupt:
            print("\n目前操作已中止，返回策略績效比較選單。")


def _print_help(program_name: str) -> None:
    print(f"用法: python {program_name} [run|status]")
    print("說明: 依config/strategy_compare.py判斷並準備前置工件，再執行策略績效比較。")
    print("無參數且在互動終端時開啟常駐選單。")


def main(argv=None) -> int:
    raw_argv = list(sys.argv if argv is None else argv)
    program_name = str(raw_argv[0] if raw_argv else "apps/strategy_compare.py")
    args = raw_argv[1:]
    if not args:
        if is_interactive_console():
            return _interactive_menu()
        _print_help(program_name)
        return 0
    command = str(args[0]).strip().lower()
    if len(args) > 1:
        raise ValueError(f"不支援額外參數: {' '.join(args[1:])}")
    if command in {"run", "compare"}:
        _run_current_comparison(confirm=is_interactive_console())
        return 0
    if command in {"status", "show"}:
        show_strategy_comparison_status()
        return 0
    if command in {"-h", "--help", "help"}:
        _print_help(program_name)
        return 0
    raise ValueError(f"不支援的命令: {command}")


__all__ = ["main"]


if __name__ == "__main__":
    run_cli_entrypoint(main)
