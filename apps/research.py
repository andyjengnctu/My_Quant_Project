"""Research single formal entry point.

The menu selects only a work type.  Model/test identities and comparison/audit
settings are owned by config/.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.audit import get_active_audit_module_id
from config.research import get_active_model_research_provider
from config.strategy_compare import get_strategy_comparison_settings
from core.runtime_utils import is_interactive_console, run_cli_entrypoint
from filters.breakout_quality.strategy_comparison import (
    collect_artifact_status,
    render_execution_plan,
    run_strategy_comparison,
    show_strategy_comparison_status,
)
from tools.audit.runner import (
    render_audit_status,
    render_latest_audit_summary,
    run_enabled_audits,
)


def _load_provider_handler(handler_name: str) -> tuple[str, Callable]:
    provider = get_active_model_research_provider()
    module = importlib.import_module(provider.module)
    name = str(getattr(provider, handler_name))
    handler = getattr(module, name, None)
    if not callable(handler):
        raise RuntimeError(
            f"model research provider缺少callable: {provider.module}:{name}"
        )
    return provider.model_id, handler


def _run_model_training_menu() -> int:
    _, handler = _load_provider_handler("menu_handler")
    return int(handler(program_name="apps/research.py model") or 0)


def _run_model_cli(args: list[str]) -> int:
    _, handler = _load_provider_handler("cli_handler")
    return int(handler(["apps/research.py model", *args]) or 0)


def _show_model_status() -> None:
    model_id, handler = _load_provider_handler("status_handler")
    print(f"\n=== 模型訓練｜{model_id} ===")
    handler()


def _run_optimizer(args: list[str] | None = None) -> int:
    from tools.optimizer import main as optimizer_main

    routed_args = ["apps/research.py optimizer", *(args or [])]
    return int(optimizer_main(argv=routed_args) or 0)


def _run_current_comparison(*, confirm: bool) -> dict:
    settings = get_strategy_comparison_settings()
    status = collect_artifact_status(settings=settings)
    print("\n" + render_execution_plan(settings=settings, status=status))
    if status["overall_status"] == "BLOCKED":
        raise RuntimeError("目前缺少不可自動產生的上游工件；請先查看狀態頁。")
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


def _strategy_compare_menu() -> int:
    while True:
        print("\n=== 策略組合比較 ===")
        print("[1/Enter] 執行目前比較設定")
        print("[2]       查看設定、工件與預計動作")
        print("[0]       返回")
        try:
            raw = input("👉 請選擇：").strip().lower()
        except EOFError:
            return 0
        choice = "1" if raw == "" else raw
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
            print("\n目前操作已中止，返回策略組合比較選單。")


def _audit_menu() -> int:
    module_id = get_active_audit_module_id()
    while True:
        print("\n=== Audit／診斷 ===")
        print(f"Active module：{module_id}")
        print("[1/Enter] 執行目前 Audit 設定")
        print("[2]       查看 Audit 設定、工件與預計動作")
        print("[3]       查看最近 Audit 結果")
        print("[0]       返回")
        try:
            raw = input("👉 請選擇：").strip().lower()
        except EOFError:
            return 0
        choice = "1" if raw == "" else raw
        if choice in {"0", "q", "quit", "exit"}:
            return 0
        if choice == "1":
            print("\n" + render_audit_status(module_id, project_root=PROJECT_ROOT))
            try:
                confirm = input("👉 按 Enter 執行；輸入 0 返回：").strip().lower()
            except EOFError:
                return 0
            if confirm in {"0", "q", "quit", "exit"}:
                continue
            if confirm not in {"", "1"}:
                print("輸入無效，本次不執行。")
                continue
            run_enabled_audits(module_id, project_root=PROJECT_ROOT)
            print("\n" + render_latest_audit_summary(module_id, project_root=PROJECT_ROOT))
        elif choice == "2":
            print("\n" + render_audit_status(module_id, project_root=PROJECT_ROOT))
        elif choice == "3":
            print("\n" + render_latest_audit_summary(module_id, project_root=PROJECT_ROOT))
        else:
            print("無效選項，請按 Enter 或輸入 0～3。")


def _show_research_status() -> int:
    print("\n====================================================================================================")
    print(" Research Status")
    print("====================================================================================================")
    sections = (
        ("模型訓練", _show_model_status),
        ("策略組合比較", show_strategy_comparison_status),
        (
            "Audit／診斷",
            lambda: print(
                render_audit_status(
                    get_active_audit_module_id(), project_root=PROJECT_ROOT
                )
            ),
        ),
    )
    for title, handler in sections:
        print(f"\n--- {title} ---")
        try:
            handler()
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"[狀態不可用] {type(exc).__name__}: {exc}")
    print("\n--- 策略參數最佳化 ---")
    print("沿用 ml_optimizer 既有互動流程；執行設定集中於 config/ 與正式 optimizer policy。")
    return 0


def _print_main_menu() -> None:
    print("\n====================================================================================================")
    print(" Research")
    print("====================================================================================================")
    print("[1/Enter] 模型訓練")
    print("[2]       策略參數最佳化")
    print("[3]       策略組合比較")
    print("[4]       Audit／診斷")
    print("[5]       查看目前設定與工件狀態")
    print("[0]       離開")


def _interactive_menu() -> int:
    while True:
        _print_main_menu()
        try:
            raw = input("👉 請選擇：").strip().lower()
        except EOFError:
            print("\n輸入已結束。")
            return 0
        choice = "1" if raw == "" else raw
        if choice in {"0", "q", "quit", "exit"}:
            return 0
        try:
            if choice == "1":
                _run_model_training_menu()
            elif choice == "2":
                _run_optimizer()
            elif choice == "3":
                _strategy_compare_menu()
            elif choice == "4":
                _audit_menu()
            elif choice == "5":
                _show_research_status()
            else:
                print("選項無效，請按 Enter 或輸入 0～5。")
        except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
            print(f"[錯誤] {type(exc).__name__}: {exc}")
        except KeyboardInterrupt:
            print("\n目前操作已中止，返回 Research 主選單。")


def _print_help(program_name: str) -> None:
    print(f"用法: python {program_name} [model|optimizer|compare|audit|status] [options]")
    print("說明: Research 單一正式入口；互動選單只選工作類型，研究標的與設定由 config/ 決定。")
    print("  model      目前 active model 的模型訓練／驗證；後續參數原樣轉交model provider")
    print("  optimizer  策略參數最佳化；後續參數原樣轉交既有 ml_optimizer service")
    print("  compare    策略組合比較；可接 run 或 status")
    print("  audit      目前 config 指定 Audit module；可接 run、status 或 latest")
    print("  status     查看目前設定與工件狀態")


def main(argv=None) -> int:
    raw_argv = list(sys.argv if argv is None else argv)
    program_name = str(raw_argv[0] if raw_argv else "apps/research.py")
    args = raw_argv[1:]
    if not args:
        if is_interactive_console():
            return _interactive_menu()
        _print_help(program_name)
        return 0

    command = str(args[0]).strip().lower()
    rest = args[1:]
    if command in {"-h", "--help", "help"}:
        _print_help(program_name)
        return 0
    if command == "model":
        return _run_model_cli(rest)
    if command in {"optimizer", "optimize"}:
        return _run_optimizer(rest)
    if command in {"compare", "strategy-compare"}:
        if not rest:
            return _strategy_compare_menu() if is_interactive_console() else 0
        action = str(rest[0]).strip().lower()
        if action in {"-h", "--help", "help"}:
            print(f"用法: python {program_name} compare [run|status]")
            print("說明: 依config/strategy_compare.py執行或查看目前策略組合比較。")
            return 0
        if len(rest) > 1:
            raise ValueError(f"compare不支援額外參數: {' '.join(rest[1:])}")
        if action in {"run", "compare"}:
            _run_current_comparison(confirm=is_interactive_console())
            return 0
        if action in {"status", "show"}:
            show_strategy_comparison_status()
            return 0
        raise ValueError(f"compare不支援的命令: {action}")
    if command == "audit":
        if not rest:
            return _audit_menu() if is_interactive_console() else 0
        action = str(rest[0]).strip().lower()
        if action in {"-h", "--help", "help"}:
            print(f"用法: python {program_name} audit [run|status|latest]")
            print("說明: Audit module由config/audit.py指定，選單不提供標的選擇。")
            return 0
        if len(rest) > 1:
            raise ValueError(f"audit不支援額外參數: {' '.join(rest[1:])}")
        module_id = get_active_audit_module_id()
        if action == "run":
            run_enabled_audits(module_id, project_root=PROJECT_ROOT)
        elif action == "status":
            print(render_audit_status(module_id, project_root=PROJECT_ROOT))
        elif action == "latest":
            print(render_latest_audit_summary(module_id, project_root=PROJECT_ROOT))
        else:
            raise ValueError(f"audit不支援的命令: {action}")
        return 0
    if command in {"status", "show"}:
        return _show_research_status()
    raise ValueError(f"不支援的 Research 命令: {command}")


__all__ = ["main"]


if __name__ == "__main__":
    run_cli_entrypoint(main)
