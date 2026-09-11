"""Lightweight Smart Downloader menu contract.

Interactive composition roots consume this module; domain synthetics may test
menu parsing here without importing ``services.downloader.main`` or executing
Market Data runtime code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

ACTION_DAILY_UPDATE = "daily_update"
ACTION_PREFLIGHT = "preflight"
ACTION_BOOTSTRAP = "bootstrap"
ACTION_PROVIDER_SNAPSHOT = "provider_snapshot"
ACTION_FULL_INTEGRITY_AUDIT = "full_integrity_audit"
ACTION_EXIT = "exit"


@dataclass(frozen=True)
class SmartDownloaderMenuOption:
    key: str
    action: str
    label: str
    tone: str = "cyan"


SMART_DOWNLOADER_MENU_OPTIONS: tuple[SmartDownloaderMenuOption, ...] = (
    SmartDownloaderMenuOption("1", ACTION_DAILY_UPDATE, "Market Data V2｜Daily Update（Canonical：dataset × date bulk）", "green"),
    SmartDownloaderMenuOption("2", ACTION_PREFLIGHT, "Market Data V2｜Backer Preflight + Exact Bootstrap Plan"),
    SmartDownloaderMenuOption("3", ACTION_BOOTSTRAP, "Market Data V2｜開始 / 續傳完整 Bootstrap"),
    SmartDownloaderMenuOption("4", ACTION_PROVIDER_SNAPSHOT, "Market Data V2｜驗證 / 重建 Provider Snapshot（不使用 API quota）"),
    SmartDownloaderMenuOption("5", ACTION_FULL_INTEGRITY_AUDIT, "Market Data V2｜Full Database Integrity Audit（不使用 API quota）"),
    SmartDownloaderMenuOption("0", ACTION_EXIT, "離開", "gray"),
)

_MENU_ACTION_BY_KEY = {option.key: option.action for option in SMART_DOWNLOADER_MENU_OPTIONS}
if len(_MENU_ACTION_BY_KEY) != len(SMART_DOWNLOADER_MENU_OPTIONS):
    raise RuntimeError("Smart Downloader menu key 不得重複")


@dataclass(frozen=True)
class DailyUpdateModeSelection:
    force_refresh_current_target: bool
    return_to_menu: bool = False


DUE_ONLY_SELECTION = DailyUpdateModeSelection(force_refresh_current_target=False)
FORCE_REFRESH_SELECTION = DailyUpdateModeSelection(force_refresh_current_target=True)
RETURN_SELECTION = DailyUpdateModeSelection(force_refresh_current_target=False, return_to_menu=True)


def parse_smart_downloader_menu_choice(raw_value: object) -> str:
    key = str(raw_value or "").strip()
    try:
        return _MENU_ACTION_BY_KEY[key]
    except KeyError as exc:
        raise ValueError("請輸入 0、1、2、3、4 或 5。") from exc


def parse_daily_update_mode(raw_value: object) -> DailyUpdateModeSelection:
    value = str(raw_value or "").strip().upper()
    if value == "":
        return DUE_ONLY_SELECTION
    if value == "R":
        return FORCE_REFRESH_SELECTION
    if value == "0":
        return RETURN_SELECTION
    raise ValueError("請按 Enter、輸入 R 或 0。")


def prompt_daily_update_mode(
    *,
    input_fn: Callable[[str], str] | None = None,
    output_fn: Callable[[str], object] | None = None,
) -> DailyUpdateModeSelection:
    input_fn = input if input_fn is None else input_fn
    output_fn = print if output_fn is None else output_fn
    output_fn("-" * 88)
    output_fn(" Daily Update 模式")
    output_fn("-" * 88)
    output_fn("[Enter] 正常更新：只處理目前 due datasets")
    output_fn("[R]     重新下載 current target：新 batch、禁止 artifact/cache REUSE，重新驗證")
    output_fn("[0]     返回")
    while True:
        try:
            raw_value = input_fn("模式: ")
        except (EOFError, KeyboardInterrupt):
            output_fn("")
            return RETURN_SELECTION
        try:
            return parse_daily_update_mode(raw_value)
        except ValueError as exc:
            output_fn(str(exc))


__all__ = [
    "ACTION_BOOTSTRAP",
    "ACTION_DAILY_UPDATE",
    "ACTION_EXIT",
    "ACTION_FULL_INTEGRITY_AUDIT",
    "ACTION_PREFLIGHT",
    "ACTION_PROVIDER_SNAPSHOT",
    "DailyUpdateModeSelection",
    "SMART_DOWNLOADER_MENU_OPTIONS",
    "parse_daily_update_mode",
    "parse_smart_downloader_menu_choice",
    "prompt_daily_update_mode",
]
