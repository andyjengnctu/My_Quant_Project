"""Breakout quality dataset、training、score export 與 evaluation 正式入口。"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_utils import resolve_cli_program_name, run_cli_entrypoint


COMMAND_MODULES = {
    "build-dataset": "tools.filters.breakout_quality.build_dataset",
    "train": "tools.filters.breakout_quality.train",
    "export-scores": "tools.filters.breakout_quality.export_scores",
    "evaluate": "tools.filters.breakout_quality.evaluate",
}

COMMAND_DESCRIPTIONS = {
    "build-dataset": "建立 breakout quality event dataset",
    "train": "訓練模型；可選擇 inner validation 選 epoch 後完整 Selection 重訓",
    "export-scores": "匯出 research 或 forward-OOS score table",
    "evaluate": "評估 train、validation、selection 或 OOS 指標",
}


def _print_help(program_name: str) -> None:
    print(f"用法: python {program_name} <command> [options]")
    print("說明: Breakout quality filter 的單一正式操作入口。")
    print("command:")
    for command, description in COMMAND_DESCRIPTIONS.items():
        print(f"  {command:<13} {description}")
    print()
    print(f"查看子命令參數: python {program_name} <command> --help")


def main(argv=None) -> int:
    raw_argv = list(sys.argv if argv is None else argv)
    program_name = resolve_cli_program_name(raw_argv, "apps/breakout_quality.py")
    args = raw_argv[1:]

    if not args or args[0] in {"-h", "--help"}:
        _print_help(program_name)
        return 0

    command = str(args[0]).strip()
    if command == "":
        raise ValueError("breakout quality command 不可為空")
    if command.startswith("-"):
        raise ValueError(f"不支援的參數: {command}")

    module_name = COMMAND_MODULES.get(command)
    if module_name is None:
        allowed = ", ".join(COMMAND_MODULES)
        raise ValueError(f"不支援的 breakout quality command: {command}；可用值: {allowed}")

    command_module = importlib.import_module(module_name)
    command_main = getattr(command_module, "main", None)
    if not callable(command_main):
        raise RuntimeError(f"breakout quality command 缺少 main(): {module_name}")

    original_program_name = sys.argv[0]
    sys.argv[0] = f"{program_name} {command}"
    try:
        result = command_main(args[1:])
    finally:
        sys.argv[0] = original_program_name
    return int(result or 0)


__all__ = ["COMMAND_DESCRIPTIONS", "COMMAND_MODULES", "main"]


if __name__ == "__main__":
    run_cli_entrypoint(main)
