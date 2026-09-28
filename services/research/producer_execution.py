"""Shared Research producer execution actions.

The Research UI owns only stable execution semantics.  Domain producers own how
normal execution, report rendering and a fresh rebuild are implemented.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.console_report import render_menu_item


PRODUCER_ACTION_NORMAL = "normal"
PRODUCER_ACTION_VIEW_REPORT = "view_report"
PRODUCER_ACTION_FORCE_REBUILD = "force_rebuild"
PRODUCER_ACTION_CANCEL = "cancel"


@dataclass(frozen=True)
class ProducerExecutionChoice:
    action: str
    label: str


def prompt_producer_execution_action(
    *,
    ready: bool,
    can_view_report: bool,
    normal_label: str = "執行必要 BUILD／REBUILD／RESUME",
    report_label: str = "查看既有訓練報表",
) -> ProducerExecutionChoice:
    """Prompt one common action surface for canonical Research producers."""

    if ready:
        if can_view_report:
            print(render_menu_item(1, report_label, default=True))
            print(render_menu_item(2, "強制重新訓練"))
            print(render_menu_item(0, "返回"))
            try:
                raw = input("👉 請選擇：").strip().lower()
            except EOFError:
                return ProducerExecutionChoice(PRODUCER_ACTION_CANCEL, "返回")
            choice = "1" if raw == "" else raw
            mapping = {
                "1": ProducerExecutionChoice(PRODUCER_ACTION_VIEW_REPORT, report_label),
                "2": ProducerExecutionChoice(PRODUCER_ACTION_FORCE_REBUILD, "強制重新訓練"),
            }
        else:
            print(render_menu_item(1, "強制重新訓練", default=False))
            print(render_menu_item(0, "返回"))
            try:
                raw = input("👉 請選擇：").strip().lower()
            except EOFError:
                return ProducerExecutionChoice(PRODUCER_ACTION_CANCEL, "返回")
            choice = raw
            mapping = {"1": ProducerExecutionChoice(PRODUCER_ACTION_FORCE_REBUILD, "強制重新訓練")}
    else:
        print(render_menu_item(1, normal_label, default=True))
        next_index = 2
        mapping = {"1": ProducerExecutionChoice(PRODUCER_ACTION_NORMAL, normal_label)}
        if can_view_report:
            print(render_menu_item(next_index, report_label))
            mapping[str(next_index)] = ProducerExecutionChoice(PRODUCER_ACTION_VIEW_REPORT, report_label)
            next_index += 1
        print(render_menu_item(next_index, "強制重新訓練"))
        mapping[str(next_index)] = ProducerExecutionChoice(PRODUCER_ACTION_FORCE_REBUILD, "強制重新訓練")
        print(render_menu_item(0, "返回"))
        try:
            raw = input("👉 請選擇：").strip().lower()
        except EOFError:
            return ProducerExecutionChoice(PRODUCER_ACTION_CANCEL, "返回")
        choice = "1" if raw == "" else raw

    if choice in {"0", "q", "quit", "exit"}:
        return ProducerExecutionChoice(PRODUCER_ACTION_CANCEL, "返回")
    selected = mapping.get(choice)
    if selected is None:
        print("輸入無效，本次不執行。")
        return ProducerExecutionChoice(PRODUCER_ACTION_CANCEL, "返回")
    return selected



__all__ = [
    "PRODUCER_ACTION_CANCEL",
    "PRODUCER_ACTION_FORCE_REBUILD",
    "PRODUCER_ACTION_NORMAL",
    "PRODUCER_ACTION_VIEW_REPORT",
    "ProducerExecutionChoice",
    "prompt_producer_execution_action",
]
