"""Project-wide formal Audit entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.audit import get_audit_module_ids
from core.runtime_utils import is_interactive_console, run_cli_entrypoint
from tools.audit.runner import (
    render_audit_status,
    render_latest_audit_summary,
    run_enabled_audits,
)


def _module_title(module_id: str) -> str:
    return {
        "breakout_quality": "Breakout Quality",
    }.get(module_id, module_id.replace("_", " ").title())


def _select_module() -> str | None:
    modules = get_audit_module_ids(enabled_only=True)
    if not modules:
        print("目前沒有啟用的 Audit module。")
        return None
    if len(modules) == 1:
        return modules[0]
    print("\n=== Audit Module ===")
    for index, module_id in enumerate(modules, start=1):
        suffix = "/Enter" if index == 1 else ""
        print(f"[{index}{suffix}] {_module_title(module_id)}")
    print("[0] 返回")
    try:
        raw = input("👉 請選擇：").strip()
    except EOFError:
        return None
    if raw == "":
        return modules[0]
    if raw in {"0", "q", "quit", "exit"}:
        return None
    try:
        index = int(raw)
    except ValueError:
        print("無效選項。")
        return None
    if not 1 <= index <= len(modules):
        print("無效選項。")
        return None
    return modules[index - 1]


def _interactive() -> int:
    while True:
        print("\n=== Project Audit ===")
        print("[1/Enter] 執行目前 Audit 設定")
        print("[2] 查看 Audit 設定、工件與預計動作")
        print("[3] 查看最近 Audit 結果")
        print("[0] 離開")
        try:
            raw = input("👉 請選擇：").strip().lower()
        except EOFError:
            return 0
        choice = "1" if raw == "" else raw
        if choice in {"0", "q", "quit", "exit"}:
            return 0
        if choice not in {"1", "2", "3"}:
            print("無效選項，請按 Enter 或輸入 0～3。")
            continue
        module_id = _select_module()
        if module_id is None:
            continue
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
        elif choice == "2":
            print("\n" + render_audit_status(module_id, project_root=PROJECT_ROOT))
        else:
            print("\n" + render_latest_audit_summary(module_id, project_root=PROJECT_ROOT))


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project-wide Audit entry point")
    parser.add_argument("action", nargs="?", choices=("run", "status", "latest"))
    parser.add_argument("module", nargs="?", choices=get_audit_module_ids(enabled_only=True))
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    if argv is None and is_interactive_console():
        return _interactive()
    args = _parse_args(argv)
    if args.action is None:
        return _interactive() if is_interactive_console() else 0
    module_id = args.module
    if module_id is None:
        modules = get_audit_module_ids(enabled_only=True)
        if len(modules) != 1:
            raise ValueError("有多個 Audit module 時，CLI 必須指定 module。")
        module_id = modules[0]
    if args.action == "run":
        run_enabled_audits(module_id, project_root=PROJECT_ROOT, quiet=args.quiet)
    elif args.action == "status":
        print(render_audit_status(module_id, project_root=PROJECT_ROOT))
    else:
        print(render_latest_audit_summary(module_id, project_root=PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    run_cli_entrypoint(main)
