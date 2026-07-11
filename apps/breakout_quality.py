"""Breakout quality dataset、training、score export、report 與 evaluation 正式入口。"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.breakout_quality_policy import BREAKOUT_QUALITY_DEFAULT_FILTER_ID
from core.runtime_utils import (
    is_interactive_console,
    resolve_cli_program_name,
    run_cli_entrypoint,
)
from filters.breakout_quality.contract import CONTEXT_COLUMNS, DEFAULT_LABEL_POLICY, FEATURE_COLUMNS
from filters.breakout_quality.paths import (
    normalize_filter_id,
    resolve_filter_artifact_paths,
    resolve_filter_output_dir,
    resolve_filter_research_manifest_path,
    resolve_filter_research_score_path,
    resolve_filter_report_json_path,
    resolve_filter_report_markdown_path,
)
from filters.breakout_quality.source_inventory import build_source_data_inventory


COMMAND_MODULES = {
    "build-dataset": "tools.filters.breakout_quality.build_dataset",
    "train": "tools.filters.breakout_quality.train",
    "export-scores": "tools.filters.breakout_quality.export_scores",
    "report": "tools.filters.breakout_quality.report",
    "evaluate": "tools.filters.breakout_quality.evaluate",
}

COMMAND_DESCRIPTIONS = {
    "menu": "開啟互動式操作選單",
    "workflow": "依序執行 dataset、train、research score export 與易讀報表",
    "build-dataset": "建立 breakout quality event dataset",
    "train": "訓練模型；可選擇 inner validation 選 epoch 後完整 Selection 重訓",
    "export-scores": "匯出 research 或 forward-OOS score table",
    "report": "產生表格化終端報表、Markdown 報表與完整 metrics JSON",
    "evaluate": "輸出 train、validation、selection 或 OOS 的詳細 JSON",
}


def _print_help(program_name: str) -> None:
    print(f"用法: python {program_name} [menu|workflow|<command>] [options]")
    print("說明: Breakout quality filter 的單一正式操作入口；互動終端不帶參數時直接開啟選單。")
    print("command:")
    for command, description in COMMAND_DESCRIPTIONS.items():
        print(f"  {command:<13} {description}")
    print()
    print(f"查看子命令參數: python {program_name} <command> --help")


def _load_command_module(command: str):
    module_name = COMMAND_MODULES.get(command)
    if module_name is None:
        raise ValueError(f"不支援的 breakout quality command: {command}")
    command_module = importlib.import_module(module_name)
    command_main = getattr(command_module, "main", None)
    if not callable(command_main):
        raise RuntimeError(f"breakout quality command 缺少 main(): {module_name}")
    return command_module


def _run_command(command: str, args: list[str], *, program_name: str) -> int:
    command_module = _load_command_module(command)
    original_program_name = sys.argv[0]
    sys.argv[0] = f"{program_name} {command}"
    try:
        result = command_module.main(list(args))
    finally:
        sys.argv[0] = original_program_name
    return int(result or 0)


def _train_defaults() -> argparse.Namespace:
    train_module = _load_command_module("train")
    parse_args = getattr(train_module, "parse_args", None)
    if not callable(parse_args):
        raise RuntimeError("breakout quality train command 缺少 parse_args()")
    return parse_args([])


def _dataset_paths(filter_id: str) -> dict[str, Path]:
    output_dir = resolve_filter_output_dir(PROJECT_ROOT, filter_id=filter_id)
    return {
        "dataset": output_dir / "dataset.npz",
        "events": output_dir / "events.csv",
        "summary": output_dir / "dataset_summary.json",
    }


def _read_dataset_summary(filter_id: str) -> dict | None:
    summary_path = _dataset_paths(filter_id)["summary"]
    if not summary_path.is_file():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _dataset_profile(filter_id: str) -> str | None:
    summary = _read_dataset_summary(filter_id)
    if summary is None:
        return None
    profile = str(summary.get("dataset") or "").strip().lower()
    return profile if profile in {"reduced", "full"} else None


def _dataset_rebuild_reasons(
    filter_id: str,
    dataset: str,
    *,
    max_tickers: int,
) -> list[str]:
    reasons: list[str] = []
    paths = _dataset_paths(filter_id)
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        reasons.append(f"dataset 工件缺少: {', '.join(missing)}")

    summary = _read_dataset_summary(filter_id)
    if summary is None:
        reasons.append("dataset_summary.json 缺少、損壞或不是 JSON object")
        return reasons

    requested_profile = str(dataset).strip().lower()
    stored_profile = str(summary.get("dataset") or "").strip().lower()
    if stored_profile != requested_profile:
        reasons.append(f"dataset profile 不符: existing={stored_profile or 'missing'}, requested={requested_profile}")

    source_selection = summary.get("source_selection")
    stored_max_tickers = None
    if isinstance(source_selection, dict):
        try:
            stored_max_tickers = int(source_selection.get("requested_max_tickers"))
        except (TypeError, ValueError):
            stored_max_tickers = None
    requested_max_tickers = max(0, int(max_tickers))
    if stored_max_tickers != requested_max_tickers:
        reasons.append(
            "dataset ticker coverage 不符: "
            f"existing_max_tickers={stored_max_tickers}, requested_max_tickers={requested_max_tickers}"
        )

    if summary.get("policy") != DEFAULT_LABEL_POLICY.as_manifest_payload():
        reasons.append("feature／label policy 已變更")
    if list(summary.get("feature_columns") or []) != list(FEATURE_COLUMNS):
        reasons.append("feature contract 已變更")
    if list(summary.get("context_columns") or []) != list(CONTEXT_COLUMNS):
        reasons.append("context contract 已變更")

    stored_inventory = summary.get("source_data_inventory")
    if not isinstance(stored_inventory, dict):
        reasons.append("dataset 缺少 source_data_inventory；需以新版 build-dataset 重建一次")
    elif stored_profile == requested_profile:
        current_inventory = build_source_data_inventory(PROJECT_ROOT, requested_profile)
        if stored_inventory != current_inventory:
            reasons.append(
                "來源 CSV inventory 已更新: "
                f"existing={stored_inventory.get('csv_inventory_sha256')}, "
                f"current={current_inventory.get('csv_inventory_sha256')}"
            )

    return reasons


def _dataset_matches_request(filter_id: str, dataset: str, *, max_tickers: int) -> bool:
    return not _dataset_rebuild_reasons(
        filter_id,
        dataset,
        max_tickers=max_tickers,
    )


def _parse_workflow_args(argv=None, *, program_name: str = "apps/breakout_quality.py") -> argparse.Namespace:
    defaults = _train_defaults()
    parser = argparse.ArgumentParser(
        prog=f"{program_name} workflow",
        description=(
            "依序執行 breakout quality research workflow：必要時建立 dataset、訓練、"
            "匯出 research scores、產生 Selection 報表，並可選擇納入 OOS"
        )
    )
    parser.add_argument("--filter-id", default=BREAKOUT_QUALITY_DEFAULT_FILTER_ID)
    parser.add_argument("--dataset", choices=("reduced", "full"), default="full")
    parser.add_argument("--max-tickers", type=int, default=0)
    parser.add_argument(
        "--rebuild-dataset",
        action="store_true",
        default=False,
        help="強制重建 dataset；未指定時在工件、profile、來源 CSV、ticker coverage 或 policy 不一致時建立",
    )
    parser.add_argument("--epochs", type=int, default=int(defaults.epochs))
    parser.add_argument("--batch-size", type=int, default=int(defaults.batch_size))
    parser.add_argument("--lr", type=float, default=float(defaults.lr))
    parser.add_argument("--seed", type=int, default=int(defaults.seed))
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=float(defaults.fixed_threshold),
    )
    parser.add_argument(
        "--use-inner-validation",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.use_inner_validation),
    )
    parser.add_argument(
        "--inner-validation-months",
        type=int,
        default=int(defaults.inner_validation_months),
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=int(defaults.early_stopping_patience),
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=float(defaults.early_stopping_min_delta),
    )
    parser.add_argument(
        "--evaluate-oos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否在 Selection 診斷後執行最終 OOS 評估",
    )
    return parser.parse_args(argv)


def _build_train_argv(args: argparse.Namespace) -> list[str]:
    argv = [
        "--filter-id",
        str(args.filter_id),
        "--epochs",
        str(int(args.epochs)),
        "--batch-size",
        str(int(args.batch_size)),
        "--lr",
        str(float(args.lr)),
        "--seed",
        str(int(args.seed)),
        "--fixed-threshold",
        str(float(args.fixed_threshold)),
        "--inner-validation-months",
        str(int(args.inner_validation_months)),
        "--early-stopping-patience",
        str(int(args.early_stopping_patience)),
        "--early-stopping-min-delta",
        str(float(args.early_stopping_min_delta)),
    ]
    argv.append(
        "--use-inner-validation"
        if bool(args.use_inner_validation)
        else "--no-use-inner-validation"
    )
    return argv


def _run_workflow(args: argparse.Namespace, *, program_name: str) -> int:
    filter_id = normalize_filter_id(args.filter_id)
    args.filter_id = filter_id
    if int(args.max_tickers) < 0:
        raise ValueError("--max-tickers 必須 >= 0")

    train_module = _load_command_module("train")
    parse_train_args = getattr(train_module, "parse_args", None)
    validate_train_args = getattr(train_module, "validate_training_args", None)
    if not callable(parse_train_args) or not callable(validate_train_args):
        raise RuntimeError("breakout quality train command 缺少可重用參數驗證介面")
    validate_train_args(parse_train_args(_build_train_argv(args)))

    print("\n=== Breakout Quality Research Workflow ===")
    print(f"filter_id={filter_id}")
    print(f"dataset={args.dataset}")
    print(
        "train="
        f"epochs={int(args.epochs)}, batch_size={int(args.batch_size)}, "
        f"lr={float(args.lr)}, seed={int(args.seed)}, threshold={float(args.fixed_threshold):.6f}, "
        f"inner_validation={bool(args.use_inner_validation)}"
    )
    print("注意：OOS 只供最終泛化評估，不得依結果回頭調整 threshold、epochs 或模型。")

    rebuild_reasons = _dataset_rebuild_reasons(
        filter_id,
        args.dataset,
        max_tickers=int(args.max_tickers),
    )
    should_build = bool(args.rebuild_dataset) or bool(rebuild_reasons)
    steps: list[tuple[str, list[str], str]] = []
    if should_build:
        if bool(args.rebuild_dataset):
            print("[rebuild] 使用者要求強制重建 dataset。")
        for reason in rebuild_reasons:
            print(f"[rebuild] {reason}")
        build_args = ["--dataset", str(args.dataset), "--filter-id", filter_id]
        if int(args.max_tickers) > 0:
            build_args.extend(["--max-tickers", str(int(args.max_tickers))])
        steps.append(("build-dataset", build_args, "建立 dataset"))
    else:
        print("[skip] dataset 工件、來源 CSV inventory、ticker coverage 與 policy 均未變更。")

    report_args = ["--filter-id", filter_id]
    report_args.append("--include-oos" if bool(args.evaluate_oos) else "--no-include-oos")
    steps.extend(
        [
            ("train", _build_train_argv(args), "訓練並產生正式模型"),
            (
                "export-scores",
                ["--filter-id", filter_id, "--scope", "research"],
                "匯出 research scores",
            ),
            (
                "report",
                report_args,
                "產生易讀研究報表",
            ),
        ]
    )

    for index, (command, command_args, label) in enumerate(steps, start=1):
        print(f"\n[{index}/{len(steps)}] {label}")
        rc = _run_command(command, command_args, program_name=program_name)
        if rc != 0:
            print(f"[stop] {label} 回傳非零狀態: {rc}")
            return int(rc)

    print("\n=== Workflow 完成 ===")
    return 0


def _prompt_text(label: str, default: str) -> str:
    raw = input(f"{label} [{default}]：").strip()
    return str(default if raw == "" else raw).strip()


def _prompt_choice(label: str, default: str, choices: dict[str, str]) -> str:
    normalized_choices = {str(key).strip().lower(): value for key, value in choices.items()}
    while True:
        raw = input(f"{label} [{default}]：").strip().lower()
        candidate = str(default).strip().lower() if raw == "" else raw
        if candidate in normalized_choices:
            return normalized_choices[candidate]
        print(f"輸入無效，可用值：{', '.join(choices)}")


def _prompt_bool(label: str, default: bool) -> bool:
    default_text = "Y" if default else "N"
    while True:
        raw = input(f"{label} [Y/N；預設 {default_text}]：").strip().lower()
        if raw == "":
            return bool(default)
        if raw in {"y", "yes", "1", "是"}:
            return True
        if raw in {"n", "no", "0", "否"}:
            return False
        print("輸入無效，請輸入 Y 或 N。")


def _prompt_int(label: str, default: int, *, minimum: int | None = None) -> int:
    while True:
        raw = input(f"{label} [{int(default)}]：").strip()
        try:
            value = int(default) if raw == "" else int(raw)
        except ValueError:
            print("輸入無效，請輸入整數。")
            continue
        if minimum is not None and value < minimum:
            print(f"輸入無效，數值必須 >= {minimum}。")
            continue
        return value


def _prompt_float(
    label: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    while True:
        raw = input(f"{label} [{float(default):g}]：").strip()
        try:
            value = float(default) if raw == "" else float(raw)
        except ValueError:
            print("輸入無效，請輸入數字。")
            continue
        if minimum is not None and value < minimum:
            print(f"輸入無效，數值必須 >= {minimum}。")
            continue
        if maximum is not None and value > maximum:
            print(f"輸入無效，數值必須 <= {maximum}。")
            continue
        return value


def _prompt_filter_id() -> str:
    return normalize_filter_id(
        _prompt_text("Filter ID", BREAKOUT_QUALITY_DEFAULT_FILTER_ID)
    )


def _prompt_train_settings(filter_id: str) -> argparse.Namespace:
    defaults = _train_defaults()
    use_inner_validation = _prompt_bool(
        "啟用 inner validation 選 best epoch",
        bool(defaults.use_inner_validation),
    )
    return argparse.Namespace(
        filter_id=filter_id,
        epochs=_prompt_int(
            "Epoch 上限（關閉 inner validation 時為固定 epochs）",
            int(defaults.epochs),
            minimum=1,
        ),
        batch_size=_prompt_int("Batch size", int(defaults.batch_size), minimum=1),
        lr=_prompt_float("Learning rate", float(defaults.lr), minimum=1e-12),
        seed=_prompt_int("Random seed", int(defaults.seed), minimum=0),
        fixed_threshold=_prompt_float(
            "Fixed threshold",
            float(defaults.fixed_threshold),
            minimum=0.0,
            maximum=1.0,
        ),
        use_inner_validation=use_inner_validation,
        inner_validation_months=(
            _prompt_int(
                "Inner validation 月數",
                int(defaults.inner_validation_months),
                minimum=1,
            )
            if use_inner_validation
            else int(defaults.inner_validation_months)
        ),
        early_stopping_patience=(
            _prompt_int(
                "Early stopping patience（0 表示跑滿）",
                int(defaults.early_stopping_patience),
                minimum=0,
            )
            if use_inner_validation
            else int(defaults.early_stopping_patience)
        ),
        early_stopping_min_delta=(
            _prompt_float(
                "Early stopping min delta",
                float(defaults.early_stopping_min_delta),
                minimum=0.0,
            )
            if use_inner_validation
            else float(defaults.early_stopping_min_delta)
        ),
    )


def _print_artifact_status(filter_id: str) -> None:
    filter_id = normalize_filter_id(filter_id)
    dataset_paths = _dataset_paths(filter_id)
    model_paths = resolve_filter_artifact_paths(PROJECT_ROOT, filter_id)
    status_paths = {
        **dataset_paths,
        "model": model_paths.model_path,
        "manifest": model_paths.manifest_path,
        "split": model_paths.split_path,
        "research_scores": resolve_filter_research_score_path(PROJECT_ROOT, filter_id),
        "research_manifest": resolve_filter_research_manifest_path(PROJECT_ROOT, filter_id),
        "readable_report": resolve_filter_report_markdown_path(PROJECT_ROOT, filter_id),
        "report_metrics": resolve_filter_report_json_path(PROJECT_ROOT, filter_id),
        "runtime_scores": model_paths.score_path,
    }
    print(f"\n=== Artifact Status: {filter_id} ===")
    for name, path in status_paths.items():
        status = "存在" if path.is_file() else "缺少"
        print(f"[{status}] {name:<18} {path}")

    summary_path = dataset_paths["summary"]
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            print(f"[警告] 無法讀取 dataset summary: {type(exc).__name__}: {exc}")
        else:
            print(
                "dataset_summary: "
                f"dataset={summary.get('dataset')}, "
                f"events={summary.get('event_count')}, "
                f"date_range={summary.get('event_date_range')}"
            )


def _interactive_workflow(program_name: str) -> int:
    filter_id = _prompt_filter_id()
    dataset = _prompt_choice(
        "Dataset：[F] Full  [R] Reduced",
        "F",
        {"f": "full", "full": "full", "r": "reduced", "reduced": "reduced"},
    )
    max_tickers = _prompt_int("最多股票數（0 表示全部）", 0, minimum=0)
    rebuild_reasons = _dataset_rebuild_reasons(
        filter_id,
        dataset,
        max_tickers=max_tickers,
    )
    rebuild_default = bool(rebuild_reasons)
    if rebuild_reasons:
        print("偵測到 dataset 需要重建：")
        for reason in rebuild_reasons:
            print(f"- {reason}")
    rebuild_dataset = _prompt_bool("建立／重建 dataset", rebuild_default)
    if not rebuild_dataset and rebuild_reasons:
        print("既有 dataset 已過期或契約不符，無法在不重建的情況下繼續。")
        return 0
    train_args = _prompt_train_settings(filter_id)
    evaluate_oos = _prompt_bool(
        "完成 Selection 診斷後執行 OOS（OOS 不得用於回頭調參）",
        True,
    )
    train_payload = dict(vars(train_args))
    train_payload.pop("filter_id", None)
    request = argparse.Namespace(
        filter_id=filter_id,
        dataset=dataset,
        max_tickers=max_tickers,
        rebuild_dataset=rebuild_dataset,
        evaluate_oos=evaluate_oos,
        **train_payload,
    )
    print("\n即將執行：dataset（依選擇）→ train → export research scores → 易讀研究報表")
    if evaluate_oos:
        print("報表將納入 OOS 最終泛化評估。")
    if not _prompt_bool("確認開始", False):
        print("已取消。")
        return 0
    return _run_workflow(request, program_name=program_name)


def _interactive_build_dataset(program_name: str) -> int:
    filter_id = _prompt_filter_id()
    dataset = _prompt_choice(
        "Dataset：[F] Full  [R] Reduced",
        "F",
        {"f": "full", "full": "full", "r": "reduced", "reduced": "reduced"},
    )
    max_tickers = _prompt_int("最多股票數（0 表示全部）", 0, minimum=0)
    if not _prompt_bool("確認建立／覆蓋 dataset 工件", False):
        print("已取消。")
        return 0
    argv = ["--dataset", dataset, "--filter-id", filter_id]
    if max_tickers > 0:
        argv.extend(["--max-tickers", str(max_tickers)])
    return _run_command("build-dataset", argv, program_name=program_name)


def _interactive_train(program_name: str) -> int:
    filter_id = _prompt_filter_id()
    request = _prompt_train_settings(filter_id)
    if not _prompt_bool("確認開始訓練（既有同 filter_id 模型會更新）", False):
        print("已取消。")
        return 0
    return _run_command(
        "train",
        _build_train_argv(request),
        program_name=program_name,
    )


def _interactive_export_research(program_name: str) -> int:
    filter_id = _prompt_filter_id()
    return _run_command(
        "export-scores",
        ["--filter-id", filter_id, "--scope", "research"],
        program_name=program_name,
    )


def _interactive_report(program_name: str) -> int:
    filter_id = _prompt_filter_id()
    include_oos = _prompt_bool(
        "報表是否納入最終 OOS（OOS 不得用於回頭調參）",
        False,
    )
    if include_oos and not _prompt_bool(
        "確認讀取最終 OOS 並寫入報表",
        False,
    ):
        print("已取消。")
        return 0
    argv = ["--filter-id", filter_id]
    argv.append("--include-oos" if include_oos else "--no-include-oos")
    return _run_command("report", argv, program_name=program_name)


def _interactive_evaluate(program_name: str) -> int:
    filter_id = _prompt_filter_id()
    split = _prompt_choice(
        "Split：[T] Train [V] Validation [S] Selection [O] OOS [A] All",
        "S",
        {
            "t": "train",
            "train": "train",
            "v": "validation",
            "validation": "validation",
            "s": "selection",
            "selection": "selection",
            "o": "oos",
            "oos": "oos",
            "a": "all",
            "all": "all",
        },
    )
    if split == "oos" and not _prompt_bool(
        "確認執行最終 OOS；結果不得用於回頭調參",
        False,
    ):
        print("已取消。")
        return 0
    return _run_command(
        "evaluate",
        ["--filter-id", filter_id, "--split", split],
        program_name=program_name,
    )


def _interactive_export_forward_oos(program_name: str) -> int:
    filter_id = _prompt_filter_id()
    if not _prompt_bool("確認更新正式 canonical scores.csv", False):
        print("已取消。")
        return 0
    return _run_command(
        "export-scores",
        ["--filter-id", filter_id, "--scope", "forward_oos"],
        program_name=program_name,
    )


def _print_menu() -> None:
    print("\n=== Breakout Quality ===")
    print("[1] 完整研究流程（預設產生易讀報表）")
    print("[2] 建立／重建 dataset")
    print("[3] 訓練模型")
    print("[4] 匯出 research scores")
    print("[5] 產生易讀研究報表")
    print("[6] 輸出詳細 JSON 評估")
    print("[7] 匯出正式 forward-OOS scores")
    print("[8/Enter] 查看工件狀態")
    print("[0] 離開")


def _run_interactive_menu(program_name: str) -> int:
    while True:
        _print_menu()
        try:
            raw_choice = input("👉 請選擇：").strip().lower()
        except EOFError:
            print("\n輸入已結束。")
            return 0
        choice = "8" if raw_choice == "" else raw_choice
        if choice in {"0", "q", "quit", "exit"}:
            return 0
        try:
            if choice == "1":
                _interactive_workflow(program_name)
            elif choice == "2":
                _interactive_build_dataset(program_name)
            elif choice == "3":
                _interactive_train(program_name)
            elif choice == "4":
                _interactive_export_research(program_name)
            elif choice == "5":
                _interactive_report(program_name)
            elif choice == "6":
                _interactive_evaluate(program_name)
            elif choice == "7":
                _interactive_export_forward_oos(program_name)
            elif choice == "8":
                _print_artifact_status(_prompt_filter_id())
            else:
                print("選項無效，請輸入 0～8。")
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"[錯誤] {type(exc).__name__}: {exc}")
        except KeyboardInterrupt:
            print("\n目前操作已中止，返回主選單。")


def main(argv=None) -> int:
    raw_argv = list(sys.argv if argv is None else argv)
    program_name = resolve_cli_program_name(raw_argv, "apps/breakout_quality.py")
    args = raw_argv[1:]

    if not args:
        if is_interactive_console():
            return _run_interactive_menu(program_name)
        _print_help(program_name)
        return 0

    if args[0] in {"-h", "--help"}:
        _print_help(program_name)
        return 0

    command = str(args[0]).strip()
    if command == "":
        raise ValueError("breakout quality command 不可為空")
    if command.startswith("-"):
        raise ValueError(f"不支援的參數: {command}")

    if command == "menu":
        if len(args) > 1:
            raise ValueError("menu 不接受其他參數")
        if not is_interactive_console():
            raise RuntimeError("menu 需要互動式終端；批次執行請使用 workflow 或其他子命令")
        return _run_interactive_menu(program_name)

    if command == "workflow":
        return _run_workflow(
            _parse_workflow_args(args[1:], program_name=program_name),
            program_name=program_name,
        )

    if command not in COMMAND_MODULES:
        allowed = ", ".join(COMMAND_DESCRIPTIONS)
        raise ValueError(f"不支援的 breakout quality command: {command}；可用值: {allowed}")

    return _run_command(command, args[1:], program_name=program_name)


__all__ = [
    "COMMAND_DESCRIPTIONS",
    "COMMAND_MODULES",
    "main",
]


if __name__ == "__main__":
    run_cli_entrypoint(main)
