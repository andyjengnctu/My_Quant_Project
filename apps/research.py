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
from config.strategy_compare import (
    STRATEGY_COMPARE_ROBUSTNESS_MENU_LABEL,
    STRATEGY_COMPARE_ROLLING_TEST_MENU_LABEL,
    get_strategy_comparison_menu_profiles,
    get_strategy_comparison_profiles,
    get_strategy_comparison_settings,
    get_strategy_multi_seed_robustness_profiles,
    get_strategy_multi_seed_robustness_settings,
    get_strategy_rolling_test_modes,
    get_strategy_runtime_integration_settings,
)
from core.console_report import render_menu_item
from core.runtime_utils import is_interactive_console, run_cli_entrypoint
from filters.breakout_quality.strategy_comparison import (
    resolve_comparison_plan,
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


def _prepare_strategy_model_prerequisites(*, profile_id: str) -> int:
    _, handler = _load_provider_handler("strategy_prerequisite_handler")
    return int(
        handler(
            program_name="apps/research.py compare prerequisite",
            profile_ids=(str(profile_id),),
        )
        or 0
    )


def _auto_preparable_model_blockers(resolved_plan) -> tuple[object, ...]:
    return tuple(
        action
        for action in resolved_plan.preparation_plan.actions
        if action.action == "BLOCKED" and action.producer_work_type == "model_training"
    )


def _non_model_blockers(resolved_plan) -> tuple[object, ...]:
    return tuple(
        action
        for action in resolved_plan.preparation_plan.actions
        if action.action == "BLOCKED" and action.producer_work_type != "model_training"
    )


def _run_optimizer(args: list[str] | None = None) -> int:
    from tools.optimizer import main as optimizer_main

    routed_args = ["apps/research.py optimizer", *(args or [])]
    return int(optimizer_main(argv=routed_args) or 0)


def _run_current_comparison(*, profile_id: str, confirm: bool) -> dict:
    settings = get_strategy_comparison_settings(profile_id)
    resolved_plan = resolve_comparison_plan(settings=settings)
    status = resolved_plan.status_dict()
    print("\n" + render_execution_plan(settings=settings, status=status))

    auto_model_blockers = _auto_preparable_model_blockers(resolved_plan)
    hard_blockers = _non_model_blockers(resolved_plan)
    if hard_blockers:
        reason = str(hard_blockers[0].description or "目前缺少不可自動產生的上游工件。")
        raise RuntimeError(reason)

    if auto_model_blockers:
        unique_sources = sorted({str(item.artifact_key).split(":")[1] for item in auto_model_blockers if str(item.artifact_key).startswith("dl:")})
        print(
            "\n自動前置：將由canonical模型訓練服務補建／接續缺少的模型工件"
            + (f" | sources={','.join(unique_sources)}" if unique_sources else "")
        )

    if confirm and settings.preparation.require_confirmation:
        try:
            choice = input("👉 按 Enter 執行（含必要自動前置）；輸入 0 返回：").strip().lower()
        except EOFError:
            print("\n輸入已結束，本次不執行。")
            return {}
        if choice in {"0", "q", "quit", "exit"}:
            print("已取消本次執行。")
            return {}
        if choice not in {"", "1"}:
            print("輸入無效，本次不執行。")
            return {}

    if auto_model_blockers:
        code = _prepare_strategy_model_prerequisites(profile_id=profile_id)
        if code != 0:
            raise RuntimeError(f"策略比較模型前置失敗: returncode={code}")
        resolved_plan = resolve_comparison_plan(settings=settings)
        remaining = tuple(
            action for action in resolved_plan.preparation_plan.actions if action.action == "BLOCKED"
        )
        if remaining:
            raise RuntimeError(str(remaining[0].description or "自動前置後仍有BLOCKED工件。"))

    return run_strategy_comparison(
        resolved_plan=resolved_plan,
        auto_prepare=True,
        settings=settings,
        quiet=True,
    )


def _strategy_compare_profile_menu(profile_id: str) -> int:
    settings = get_strategy_comparison_settings(profile_id)
    while True:
        print(f"\n=== {settings.profile_label} ===")
        print(render_menu_item(1, "執行目前比較設定", default=True))
        print(render_menu_item(2, "查看設定、工件與預計動作"))
        print(render_menu_item(0, "返回"))
        try:
            raw = input("👉 請選擇：").strip().lower()
        except EOFError:
            return 0
        choice = "1" if raw == "" else raw
        if choice in {"0", "q", "quit", "exit"}:
            return 0
        try:
            if choice == "1":
                _run_current_comparison(profile_id=profile_id, confirm=True)
            elif choice == "2":
                show_strategy_comparison_status(settings=settings)
            else:
                print("選項無效，請按 Enter 或輸入 0～2。")
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"[錯誤] {type(exc).__name__}: {exc}")
        except KeyboardInterrupt:
            print(f"\n目前操作已中止，返回{settings.profile_label}選單。")


def _strategy_multi_seed_robustness_menu(robustness_id: str) -> int:
    from filters.breakout_quality.strategy_multi_seed_robustness import (
        run_multi_seed_robustness,
        show_latest_multi_seed_robustness_report,
        show_multi_seed_robustness_status,
    )

    robustness = get_strategy_multi_seed_robustness_settings(robustness_id)
    while True:
        print(f"\n=== {robustness.label} ===")
        print(render_menu_item(1, "執行", default=True))
        print(render_menu_item(2, "查看設定與預計動作"))
        print(render_menu_item(3, "查看最新報表"))
        print(render_menu_item(0, "返回"))
        try:
            raw = input("👉 請選擇：").strip().lower()
        except EOFError:
            return 0
        choice = "1" if raw == "" else raw
        if choice in {"0", "q", "quit", "exit"}:
            return 0
        try:
            if choice == "1":
                run_multi_seed_robustness(robustness_id=robustness_id, confirm=True)
            elif choice == "2":
                show_multi_seed_robustness_status(robustness_id=robustness_id)
            elif choice == "3":
                show_latest_multi_seed_robustness_report(robustness_id=robustness_id)
            else:
                print("選項無效，請按 Enter 或輸入 0～3。")
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"[錯誤] {type(exc).__name__}: {exc}")
        except KeyboardInterrupt:
            print(f"\n目前操作已中止，返回{robustness.label}選單。")


def _strategy_runtime_integration_menu() -> int:
    from filters.breakout_quality.runtime_integration_gate import (
        run_runtime_integration_gate,
        show_latest_runtime_integration_report,
        show_runtime_integration_status,
    )
    from services.breakout_quality.runtime_promotion import apply_or_refresh_runtime_promotion

    cfg = get_strategy_runtime_integration_settings()
    while True:
        print(f"\n=== {cfg.label} ===")
        print(render_menu_item(1, "執行 Gate", default=True))
        print(render_menu_item(2, "套用／更新正式 Runtime"))
        print(render_menu_item(3, "查看目前 Gate 狀態"))
        print(render_menu_item(4, "查看最新 Gate 報表"))
        print(render_menu_item(0, "返回"))
        try:
            raw = input("👉 請選擇：").strip().lower()
        except EOFError:
            return 0
        choice = "1" if raw == "" else raw
        if choice in {"0", "q", "quit", "exit"}:
            return 0
        try:
            if choice == "1":
                run_runtime_integration_gate()
            elif choice == "2":
                apply_or_refresh_runtime_promotion()
            elif choice == "3":
                show_runtime_integration_status()
            elif choice == "4":
                show_latest_runtime_integration_report()
            else:
                print("選項無效，請按 Enter 或輸入 0～4。")
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"[錯誤] {type(exc).__name__}: {exc}")
        except KeyboardInterrupt:
            print(f"\n目前操作已中止，返回{cfg.label}選單。")


def _show_all_strategy_comparison_status() -> None:
    for profile in get_strategy_comparison_menu_profiles():
        settings = get_strategy_comparison_settings(profile["profile_id"])
        print(f"\n--- {settings.profile_label} ---")
        try:
            show_strategy_comparison_status(settings=settings)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"[狀態不可用] {type(exc).__name__}: {exc}")
    from filters.breakout_quality.strategy_multi_seed_robustness import (
        show_multi_seed_robustness_status,
    )
    for robustness in get_strategy_multi_seed_robustness_profiles():
        print(f"\n--- {robustness['label']} ---")
        try:
            show_multi_seed_robustness_status(robustness_id=robustness["robustness_id"])
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"[狀態不可用] {type(exc).__name__}: {exc}")


def _strategy_rolling_mode_menu(*, robustness: bool) -> int:
    modes = get_strategy_rolling_test_modes()
    title = (
        STRATEGY_COMPARE_ROBUSTNESS_MENU_LABEL
        if robustness
        else STRATEGY_COMPARE_ROLLING_TEST_MENU_LABEL
    )
    while True:
        print(f"\n=== {title} ===")
        for index, mode in enumerate(modes, start=1):
            print(
                render_menu_item(
                    index,
                    (
                        f"{mode['label']} | {mode.get('score_start_date')}→{'最新' if str(mode.get('score_end_date')).lower() == 'auto' else mode.get('score_end_date')}"
                        if bool(mode.get("single_score_block"))
                        else (
                            f"{mode['label']} | {mode.get('score_start_date')}→"
                            f"{'最新' if str(mode.get('score_end_date')).lower() == 'auto' else mode.get('score_end_date')}"
                            f" | {int(mode['fold_months'])}M"
                        )
                    ),
                    default=index == 1,
                )
            )
        print(render_menu_item(0, "返回"))
        try:
            raw = input("👉 請選擇：").strip().lower()
        except EOFError:
            return 0
        choice = "1" if raw == "" else raw
        if choice in {"0", "q", "quit", "exit"}:
            return 0
        try:
            numeric = int(choice)
        except ValueError:
            print("選項無效。")
            continue
        if 1 <= numeric <= len(modes):
            mode = modes[numeric - 1]
            if robustness:
                _strategy_multi_seed_robustness_menu(str(mode["robustness_id"]))
            else:
                _strategy_compare_profile_menu(str(mode["profile_id"]))
            continue
        print(f"選項無效，請按 Enter 或輸入 0～{len(modes)}。")


def _strategy_compare_menu() -> int:
    while True:
        print("\n=== 策略組合比較 ===")
        print(render_menu_item(1, STRATEGY_COMPARE_ROLLING_TEST_MENU_LABEL, default=True))
        print(render_menu_item(2, STRATEGY_COMPARE_ROBUSTNESS_MENU_LABEL))
        integration_cfg = get_strategy_runtime_integration_settings()
        integration_choice = 3 if integration_cfg.enabled else None
        if integration_choice is not None:
            print(render_menu_item(integration_choice, integration_cfg.label))
            status_choice = 4
        else:
            status_choice = 3
        print(render_menu_item(status_choice, "查看目前Framework設定與工件狀態"))
        print(render_menu_item(0, "返回"))
        try:
            raw = input("👉 請選擇：").strip().lower()
        except EOFError:
            return 0
        choice = "1" if raw == "" else raw
        if choice in {"0", "q", "quit", "exit"}:
            return 0
        try:
            numeric = int(choice)
        except ValueError:
            print("選項無效。")
            continue
        if numeric == 1:
            _strategy_rolling_mode_menu(robustness=False)
        elif numeric == 2:
            _strategy_rolling_mode_menu(robustness=True)
        elif integration_choice is not None and numeric == integration_choice:
            _strategy_runtime_integration_menu()
        elif numeric == status_choice:
            _show_all_strategy_comparison_status()
        else:
            print(f"選項無效，請按 Enter 或輸入 0～{status_choice}。")


def _audit_menu() -> int:
    module_id = get_active_audit_module_id()
    while True:
        print("\n=== Audit／診斷 ===")
        print(f"Active module：{module_id}")
        print(render_menu_item(1, "執行目前 Audit 設定", default=True))
        print(render_menu_item(2, "查看 Audit 設定、工件與預計動作"))
        print(render_menu_item(3, "查看最近 Audit 結果"))
        print(render_menu_item(0, "返回"))
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
        ("策略組合比較", _show_all_strategy_comparison_status),
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
    print(render_menu_item(1, "模型訓練", default=True))
    print(render_menu_item(2, "策略參數最佳化"))
    print(render_menu_item(3, "策略組合比較"))
    print(render_menu_item(4, "Audit／診斷"))
    print(render_menu_item(5, "查看目前設定與工件狀態"))
    print(render_menu_item(0, "離開"))


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
    print("  compare    策略組合比較；可接 [profile] run/status、robustness run/status/latest 或 integration run/status/latest")
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
        if str(rest[0]).strip().lower() in {"-h", "--help", "help"}:
            print(f"用法: python {program_name} compare [profile] [run|status]")
            print(f"      python {program_name} compare robustness [robustness_id] [run|status|latest]")
            print(f"      python {program_name} compare integration [run|promote|status|latest]")
            print("說明: profile、robustness與runtime integration設定皆由config/strategy_compare.py定義。")
            return 0
        first = str(rest[0]).strip()
        if first.lower() in {"integration", "runtime-integration", "runtime_integration"}:
            action = str(rest[1]).strip().lower() if len(rest) >= 2 else "status"
            if len(rest) > 2:
                raise ValueError(f"compare integration不支援額外參數: {' '.join(rest[2:])}")
            from filters.breakout_quality.runtime_integration_gate import (
                run_runtime_integration_gate,
                show_latest_runtime_integration_report,
                show_runtime_integration_status,
            )
            from services.breakout_quality.runtime_promotion import apply_or_refresh_runtime_promotion
            if action == "run":
                run_runtime_integration_gate()
                return 0
            if action in {"promote", "apply", "refresh"}:
                apply_or_refresh_runtime_promotion()
                return 0
            if action in {"status", "show"}:
                show_runtime_integration_status()
                return 0
            if action == "latest":
                show_latest_runtime_integration_report()
                return 0
            raise ValueError(f"compare integration不支援的命令: {action}")

        if first.lower() in {"robustness", "multi-seed", "multi_seed"}:
            profiles = get_strategy_multi_seed_robustness_profiles()
            profile_ids = {item["robustness_id"] for item in profiles}
            robustness_id = get_strategy_multi_seed_robustness_settings().robustness_id
            action_index = 1
            if len(rest) >= 2 and rest[1] in profile_ids:
                robustness_id = rest[1]
                action_index = 2
            action = rest[action_index].lower() if len(rest) > action_index else "status"
            if len(rest) > action_index + 1:
                raise ValueError(
                    "compare robustness不支援額外參數: "
                    + " ".join(rest[action_index + 1:])
                )
            from filters.breakout_quality.strategy_multi_seed_robustness import (
                run_multi_seed_robustness,
                show_latest_multi_seed_robustness_report,
                show_multi_seed_robustness_status,
            )
            if action == "run":
                run_multi_seed_robustness(
                    robustness_id=robustness_id, confirm=is_interactive_console()
                )
                return 0
            if action in {"status", "show"}:
                show_multi_seed_robustness_status(robustness_id=robustness_id)
                return 0
            if action == "latest":
                show_latest_multi_seed_robustness_report(robustness_id=robustness_id)
                return 0
            raise ValueError(f"compare robustness不支援的命令: {action}")

        profile_ids = {item["profile_id"] for item in get_strategy_comparison_profiles()}
        if first in profile_ids:
            profile_id = first
            action = str(rest[1]).strip().lower() if len(rest) >= 2 else "status"
            if len(rest) > 2:
                raise ValueError(f"compare不支援額外參數: {' '.join(rest[2:])}")
        else:
            profile_id = None
            action = first.lower()
            if len(rest) > 1:
                raise ValueError(f"compare不支援額外參數: {' '.join(rest[1:])}")
        settings = get_strategy_comparison_settings(profile_id)
        if action in {"run", "compare"}:
            _run_current_comparison(
                profile_id=settings.profile_id,
                confirm=is_interactive_console(),
            )
            return 0
        if action in {"status", "show"}:
            show_strategy_comparison_status(settings=settings)
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
