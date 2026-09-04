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

from config.audit import get_active_audit_module_id, get_audit_definitions
from core.research_policy import get_active_model_research_provider
from core.strategy_compare_policy import (
    get_strategy_comparison_menu_profiles,
    get_strategy_comparison_settings,
    get_strategy_multi_seed_robustness_profiles,
    get_strategy_multi_seed_robustness_settings,
    get_strategy_rolling_test_modes,
    get_strategy_runtime_integration_settings,
)
from core.console_report import render_menu_item
from core.runtime_utils import is_interactive_console, run_cli_entrypoint
from services.research.strategy_compare_application import (
    dispatch_runtime_integration_action,
    execute_strategy_comparison,
    resolve_strategy_comparison_execution,
    run_strategy_multi_seed_robustness,
    runtime_integration_allowed_actions,
    runtime_integration_execution_enabled,
    show_latest_strategy_multi_seed_robustness_report,
    show_strategy_comparison_profile_status,
    show_strategy_multi_seed_robustness_status,
)
from services.audit.catalog import get_audit_methods
from services.audit.runner import (
    collect_audit_menu_state,
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


def _strategy_model_artifact_scope(action) -> str:
    """Map a shared Research artifact action to the provider's model-side scope."""

    key = str(getattr(action, "artifact_key", ""))
    return "upstream" if key.startswith("model-upstream:") else "models"


def _prepare_strategy_model_artifacts(*, profile_id: str, scope: str) -> int:
    """Dispatch Strategy Compare model dependencies through one provider contract."""

    _, handler = _load_provider_handler("strategy_artifact_handler")
    normalized_scope = str(scope).strip().lower()
    if normalized_scope not in {"upstream", "models"}:
        raise ValueError(f"不支援的Strategy Compare model artifact scope: {scope!r}")
    return int(
        handler(
            program_name=f"apps/research.py compare {normalized_scope}",
            profile_ids=(str(profile_id),),
            scope=normalized_scope,
        )
        or 0
    )


def _run_strategy_param_migration() -> int:
    from services.optimizer.strategy_param_service import (
        finalize_legacy_strategy_parameter_migration,
    )

    result = finalize_legacy_strategy_parameter_migration(PROJECT_ROOT)
    cleanup = dict(result.get("cleanup") or {})
    print("\n====================================================================================================")
    print(" Strategy Parameter SSOT Migration")
    print("====================================================================================================")
    print(f"Status            ：{result.get('status')}")
    print(f"Cleanup gate      ：{cleanup.get('status')}")
    removable = list(cleanup.get("removable") or [])
    blockers = list(cleanup.get("blockers") or [])
    migration = dict(result.get("migration") or {})
    legacy_oos_archive = dict(migration.get("legacy_oos_archive") or {})
    archived_oos = list(legacy_oos_archive.get("archived") or [])
    print(f"可安全移除legacy ：{len(removable)}")
    print(f"Historical OOS archive：{len(archived_oos)}")
    print(f"Cleanup blockers  ：{len(blockers)}")
    if archived_oos:
        print(
            f"Archive manifest   ：{legacy_oos_archive.get('archive_manifest_path') or '(missing)'}"
        )
    if removable:
        print("\n可安全移除：")
        for row in removable:
            print(f"- {row.get('source')} -> {row.get('target')} [{row.get('evidence')}]")
    if blockers:
        print("\n尚不可移除：")
        for row in blockers:
            print(f"- {row.get('source')} -> {row.get('target')} [{row.get('reason')}]")
    print("\n此步驟只migration與驗證，不會刪除legacy檔案。Cleanup gate必須READY後才可執行Remove-Item。")
    return 0 if str(result.get("status")) == "READY_FOR_CLEANUP" else 1


def _run_optimizer(args: list[str] | None = None) -> int:
    routed = list(args or [])
    if routed and str(routed[0]).strip().lower() in {
        "migrate-strategy-params",
        "migrate_strategy_params",
        "strategy-param-migration",
    }:
        if len(routed) > 1:
            raise ValueError(f"strategy-param migration不支援額外參數: {' '.join(routed[1:])}")
        return _run_strategy_param_migration()

    from services.optimizer.application import main as optimizer_main

    routed_args = ["apps/research.py optimizer", *routed]
    return int(optimizer_main(argv=routed_args) or 0)


def _run_current_comparison(*, profile_id: str, confirm: bool) -> dict:
    settings, resolved_plan, rendered_plan = resolve_strategy_comparison_execution(profile_id)
    print("\n" + rendered_plan)

    if resolved_plan.preparation_plan.blocked:
        blocked = [
            action for action in resolved_plan.preparation_plan.actions
            if action.action == "BLOCKED"
        ]
        reason = str(
            blocked[0].description if blocked else "目前存在不可自動補建的Research前置工件。"
        )
        raise RuntimeError(reason)

    if confirm and settings.preparation.require_confirmation:
        try:
            choice = input("👉 按 Enter 執行（含所有必要自動前置）；輸入 0 返回：").strip().lower()
        except EOFError:
            print("\n輸入已結束，本次不執行。")
            return {}
        if choice in {"0", "q", "quit", "exit"}:
            print("已取消本次執行。")
            return {}
        if choice not in {"", "1"}:
            print("輸入無效，本次不執行。")
            return {}

    return execute_strategy_comparison(
        resolved_plan=resolved_plan,
        settings=settings,
        producer_handlers={
            "model_training": lambda action: _prepare_strategy_model_artifacts(
                profile_id=profile_id, scope=_strategy_model_artifact_scope(action)
            )
        },
    )


def _strategy_compare_profile_menu(profile_id: str) -> int:
    try:
        _run_current_comparison(profile_id=profile_id, confirm=True)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[錯誤] {type(exc).__name__}: {exc}")
    except KeyboardInterrupt:
        print("\n目前操作已中止，返回策略組合比較選單。")
    return 0


def _run_current_robustness(*, robustness_id: str, confirm: bool) -> dict:
    robustness = get_strategy_multi_seed_robustness_settings(robustness_id)
    return run_strategy_multi_seed_robustness(
        robustness_id=robustness_id,
        confirm=confirm,
        model_upstream_preparer=lambda: _prepare_strategy_model_artifacts(
            profile_id=robustness.profile_id, scope="upstream"
        ),
    )


def _strategy_multi_seed_robustness_menu(robustness_id: str) -> int:
    try:
        _run_current_robustness(robustness_id=robustness_id, confirm=True)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[錯誤] {type(exc).__name__}: {exc}")
    except KeyboardInterrupt:
        print("\n目前操作已中止，返回策略組合比較選單。")
    return 0


def _current_strategy_mode_rows() -> tuple[dict, dict]:
    modes = tuple(get_strategy_rolling_test_modes())
    oos = next((dict(item) for item in modes if bool(item.get("single_score_block"))), None)
    rolling = next((dict(item) for item in modes if not bool(item.get("single_score_block"))), None)
    if oos is None or rolling is None:
        raise ValueError("current Strategy Compare必須同時定義OOS與Rolling mode")
    return oos, rolling


def _strategy_runtime_integration_menu() -> int:
    cfg = get_strategy_runtime_integration_settings()
    if not runtime_integration_execution_enabled():
        raise RuntimeError(
            f"{cfg.label}目前未綁定可執行的current Strategy Compare profiles；"
            "請由Framework status檢視historical狀態。"
        )
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
                dispatch_runtime_integration_action("run")
            elif choice == "2":
                dispatch_runtime_integration_action("promote")
            elif choice == "3":
                dispatch_runtime_integration_action("status")
            elif choice == "4":
                dispatch_runtime_integration_action("latest")
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
            show_strategy_comparison_profile_status(profile_id=profile["profile_id"])
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"[狀態不可用] {type(exc).__name__}: {exc}")
    for robustness in get_strategy_multi_seed_robustness_profiles():
        print(f"\n--- {robustness['label']} ---")
        try:
            show_strategy_multi_seed_robustness_status(robustness_id=robustness["robustness_id"])
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"[狀態不可用] {type(exc).__name__}: {exc}")


def _strategy_compare_menu() -> int:
    while True:
        oos_mode, rolling_mode = _current_strategy_mode_rows()
        print("\n=== 策略組合比較 ===")
        print(render_menu_item(1, "Forward OOS Test", default=True))
        print(render_menu_item(2, "Rolling OOS Test"))
        print(render_menu_item(3, "Forward OOS Robustness Test"))
        print(render_menu_item(4, "Rolling OOS Robustness Test"))
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
            _strategy_compare_profile_menu(str(oos_mode["profile_id"]))
        elif numeric == 2:
            _strategy_compare_profile_menu(str(rolling_mode["profile_id"]))
        elif numeric == 3:
            _strategy_multi_seed_robustness_menu(str(oos_mode["robustness_id"]))
        elif numeric == 4:
            _strategy_multi_seed_robustness_menu(str(rolling_mode["robustness_id"]))
        else:
            print("選項無效，請按 Enter 或輸入 0～4。")


def _run_reusable_audit_method(module_id: str, method_id: str) -> int:
    method = next(item for item in get_audit_methods() if item.method_id == method_id)
    menu_state = collect_audit_menu_state(module_id, method_id=method_id)
    enabled = bool(menu_state["configured"])
    print(f"\n=== {method.menu_label} ===")
    print("\n" + render_audit_status(
        module_id, project_root=PROJECT_ROOT, method_id=method_id
    ))
    if not enabled:
        print("目前沒有啟用此Reusable module的參數設定；保留固定選單但本次不執行。")
        return 0
    try:
        confirm = input("👉 按 Enter 執行；輸入 0 返回：").strip().lower()
    except EOFError:
        return 0
    if confirm in {"0", "q", "quit", "exit"}:
        return 0
    if confirm not in {"", "1"}:
        print("輸入無效，本次不執行。")
        return 0
    run_enabled_audits(
        module_id, project_root=PROJECT_ROOT, method_id=method_id
    )
    print("\n" + render_latest_audit_summary(
        module_id, project_root=PROJECT_ROOT, method_id=method_id
    ))
    return 0


def _audit_reusable_menu(module_id: str) -> int:
    methods = get_audit_methods()
    while True:
        print("\n=== 可重複使用的原因分析 ===")
        enabled_counts = {
            method.method_id: int(
                collect_audit_menu_state(
                    module_id, method_id=method.method_id
                )["enabled_count"]
            )
            for method in methods
        }
        for index, method in enumerate(methods, start=1):
            suffix = "  [已設定]" if enabled_counts.get(method.method_id, 0) else "  [未設定]"
            print(render_menu_item(index, method.menu_label + suffix, default=index == 1))
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
        if 1 <= numeric <= len(methods):
            _run_reusable_audit_method(module_id, methods[numeric - 1].method_id)
            continue
        print(f"無效選項，請輸入 0～{len(methods)}。")


def _run_one_time_audit(module_id: str, audit_id: str) -> int:
    state = collect_audit_menu_state(module_id, audit_id=audit_id)
    print("\n" + render_audit_status(
        module_id, project_root=PROJECT_ROOT, audit_id=audit_id
    ))
    if not bool(state["configured"]):
        print("此一次性 Audit 目前未啟用；只保留歷史／設定證據。")
        print("\n" + render_latest_audit_summary(
            module_id, project_root=PROJECT_ROOT, audit_id=audit_id
        ))
        return 0
    try:
        confirm = input("👉 按 Enter 執行此一次性 Audit；輸入 0 返回：").strip().lower()
    except EOFError:
        return 0
    if confirm in {"0", "q", "quit", "exit"}:
        return 0
    if confirm not in {"", "1"}:
        print("輸入無效，本次不執行。")
        return 0
    run_enabled_audits(
        module_id, project_root=PROJECT_ROOT, audit_id=audit_id
    )
    print("\n" + render_latest_audit_summary(
        module_id, project_root=PROJECT_ROOT, audit_id=audit_id
    ))
    return 0


def _audit_one_time_menu(module_id: str) -> int:
    definitions = get_audit_definitions(module_id)
    while True:
        print("\n=== 一次性專題 Audit ===")
        if not definitions:
            print("目前沒有一次性 Audit 設定。")
        for index, definition in enumerate(definitions, start=1):
            suffix = "  [可執行]" if definition.enabled else "  [歷史／停用]"
            print(render_menu_item(index, definition.audit_id + suffix, default=index == 1))
        print(render_menu_item(0, "返回"))
        try:
            raw = input("👉 請選擇：").strip().lower()
        except EOFError:
            return 0
        choice = "1" if raw == "" and definitions else raw
        if choice in {"0", "q", "quit", "exit", ""}:
            return 0
        try:
            numeric = int(choice)
        except ValueError:
            print("選項無效。")
            continue
        if 1 <= numeric <= len(definitions):
            _run_one_time_audit(module_id, definitions[numeric - 1].audit_id)
            continue
        print(f"無效選項，請輸入 0～{len(definitions)}。")


def _audit_menu() -> int:
    module_id = get_active_audit_module_id()
    while True:
        print("\n=== Audit／診斷 ===")
        print(f"Active module：{module_id}")
        print(render_menu_item(1, "可重複使用的原因分析", default=True))
        print(render_menu_item(2, "一次性專題 Audit"))
        print(render_menu_item(3, "最近結果／歷史 Evidence"))
        print(render_menu_item(0, "返回"))
        try:
            raw = input("👉 請選擇：").strip().lower()
        except EOFError:
            return 0
        choice = "1" if raw == "" else raw
        if choice in {"0", "q", "quit", "exit"}:
            return 0
        if choice == "1":
            _audit_reusable_menu(module_id)
        elif choice == "2":
            _audit_one_time_menu(module_id)
        elif choice == "3":
            print("\n" + render_latest_audit_summary(module_id, project_root=PROJECT_ROOT))
        else:
            print("無效選項，請輸入 0～3。")


def _show_research_status() -> int:
    print("\n====================================================================================================")
    print(" Current Research State / Artifacts")
    print("====================================================================================================")
    sections = (
        ("模型訓練／驗證", _show_model_status),
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
    print(render_menu_item(1, "模型訓練／驗證", default=True))
    print(render_menu_item(2, "策略參數最佳化"))
    print(render_menu_item(3, "策略組合比較"))
    print(render_menu_item(4, "Audit／診斷"))
    integration_cfg = get_strategy_runtime_integration_settings()
    integration_label = integration_cfg.label
    if not runtime_integration_execution_enabled():
        integration_label += "  [歷史／唯讀]"
    print(render_menu_item(5, integration_label))
    print(render_menu_item(6, "查看目前研究狀態與工件"))
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
                _strategy_runtime_integration_menu()
            elif choice == "6":
                _show_research_status()
            else:
                print("選項無效，請按 Enter 或輸入 0～6。")
        except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
            print(f"[錯誤] {type(exc).__name__}: {exc}")
        except KeyboardInterrupt:
            print("\n目前操作已中止，返回 Research 主選單。")


def _print_help(program_name: str) -> None:
    print(f"用法: python {program_name} [model|optimizer|compare|audit|status] [options]")
    print("說明: Research 單一正式入口；互動選單只選工作類型，研究標的與設定由 config/ 決定。")
    print("  model      目前 active model 的模型訓練／驗證；後續參數原樣轉交model provider")
    print("  optimizer  策略參數最佳化；後續參數原樣轉交既有 ml_optimizer service")
    print("             一次性SSOT migration：optimizer migrate-strategy-params")
    integration_actions = "/".join(runtime_integration_allowed_actions())
    print(
        "  compare    策略組合比較；可接 [profile] run/status、robustness run/status/latest "
        f"或 integration {integration_actions}"
    )
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
            integration_actions = "|".join(runtime_integration_allowed_actions())
            print(f"      python {program_name} compare integration [{integration_actions}]")
            print("說明: profile、robustness與runtime integration設定皆由config/strategy_compare.py定義。")
            return 0
        first = str(rest[0]).strip()
        if first.lower() in {"integration", "runtime-integration", "runtime_integration"}:
            action = str(rest[1]).strip().lower() if len(rest) >= 2 else "status"
            if len(rest) > 2:
                raise ValueError(f"compare integration不支援額外參數: {' '.join(rest[2:])}")
            dispatch_runtime_integration_action(action)
            return 0

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
            if action == "run":
                _run_current_robustness(
                    robustness_id=robustness_id,
                    confirm=is_interactive_console(),
                )
                return 0
            if action in {"status", "show"}:
                show_strategy_multi_seed_robustness_status(robustness_id=robustness_id)
                return 0
            if action == "latest":
                show_latest_strategy_multi_seed_robustness_report(robustness_id=robustness_id)
                return 0
            raise ValueError(f"compare robustness不支援的命令: {action}")

        profile_ids = {item["profile_id"] for item in get_strategy_comparison_menu_profiles()}
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
            show_strategy_comparison_profile_status(profile_id=settings.profile_id)
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
