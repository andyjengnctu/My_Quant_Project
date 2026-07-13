"""Breakout quality dataset、training、score export、report 與 evaluation 正式入口。"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.breakout_quality_policy import (
    BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
)
from core.display_common import render_elapsed
from core.runtime_utils import (
    is_interactive_console,
    resolve_cli_program_name,
    run_cli_entrypoint,
)
from filters.breakout_quality.contract import CONTEXT_COLUMNS, DEFAULT_LABEL_POLICY, FEATURE_COLUMNS
from filters.breakout_quality.models.spec import get_model_spec
from filters.breakout_quality.dataset_store import (
    DATASET_STORAGE_FORMAT,
    DATASET_STORAGE_SCHEMA_VERSION,
    dataset_artifact_metadata_reasons,
    resolve_dataset_paths,
)
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

INTERACTIVE_DATASET_PROFILE = "full"
INTERACTIVE_MAX_TICKERS = 0
INTERACTIVE_EVALUATE_OOS = True


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
    paths = resolve_dataset_paths(output_dir)
    return {
        **paths.artifact_paths(),
        "summary": paths.summary,
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


def _dataset_refresh_plan(
    filter_id: str,
    dataset: str,
    *,
    max_tickers: int,
) -> tuple[str, list[str]]:
    full_rebuild_reasons: list[str] = []
    relabel_reasons: list[str] = []
    paths = _dataset_paths(filter_id)
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        full_rebuild_reasons.append(f"dataset 工件缺少: {', '.join(missing)}")

    summary = _read_dataset_summary(filter_id)
    if summary is None:
        full_rebuild_reasons.append("dataset_summary.json 缺少、損壞或不是 JSON object")
        return "rebuild", full_rebuild_reasons

    if int(summary.get("dataset_storage_schema_version", -1)) != DATASET_STORAGE_SCHEMA_VERSION:
        full_rebuild_reasons.append("dataset storage schema 已變更")
    if str(summary.get("dataset_storage_format") or "") != DATASET_STORAGE_FORMAT:
        full_rebuild_reasons.append("dataset storage format 已變更")

    output_dir = resolve_filter_output_dir(PROJECT_ROOT, filter_id=filter_id)
    storage_paths = resolve_dataset_paths(output_dir)
    full_rebuild_reasons.extend(
        dataset_artifact_metadata_reasons(
            storage_paths,
            summary.get("dataset_artifacts"),
        )
    )

    requested_profile = str(dataset).strip().lower()
    stored_profile = str(summary.get("dataset") or "").strip().lower()
    if stored_profile != requested_profile:
        full_rebuild_reasons.append(
            f"dataset profile 不符: existing={stored_profile or 'missing'}, requested={requested_profile}"
        )

    source_selection = summary.get("source_selection")
    stored_max_tickers = None
    if isinstance(source_selection, dict):
        try:
            stored_max_tickers = int(source_selection.get("requested_max_tickers"))
        except (TypeError, ValueError):
            stored_max_tickers = None
    requested_max_tickers = max(0, int(max_tickers))
    if stored_max_tickers != requested_max_tickers:
        full_rebuild_reasons.append(
            "dataset ticker coverage 不符: "
            f"existing_max_tickers={stored_max_tickers}, requested_max_tickers={requested_max_tickers}"
        )

    if summary.get("feature_cache_policy") != DEFAULT_LABEL_POLICY.feature_cache_manifest_payload():
        full_rebuild_reasons.append("feature／high_len／benchmark／path-cache policy 已變更")
    if list(summary.get("feature_columns") or []) != list(FEATURE_COLUMNS):
        full_rebuild_reasons.append("feature contract 已變更")
    if list(summary.get("context_columns") or []) != list(CONTEXT_COLUMNS):
        full_rebuild_reasons.append("context contract 已變更")

    stored_inventory = summary.get("source_data_inventory")
    if not isinstance(stored_inventory, dict):
        full_rebuild_reasons.append("dataset 缺少 source_data_inventory；需完整重建一次")
    elif stored_profile == requested_profile:
        current_inventory = build_source_data_inventory(PROJECT_ROOT, requested_profile)
        if stored_inventory != current_inventory:
            full_rebuild_reasons.append(
                "來源 CSV inventory 已更新: "
                f"existing={stored_inventory.get('csv_inventory_sha256')}, "
                f"current={current_inventory.get('csv_inventory_sha256')}"
            )

    if full_rebuild_reasons:
        return "rebuild", full_rebuild_reasons

    if summary.get("label_policy") != DEFAULT_LABEL_POLICY.label_manifest_payload():
        relabel_reasons.append("label horizon／MFE／MAE／reward-risk policy 已變更")
    if summary.get("policy") != DEFAULT_LABEL_POLICY.as_manifest_payload():
        if not relabel_reasons:
            full_rebuild_reasons.append("dataset policy metadata 與目前設定不一致")

    if full_rebuild_reasons:
        return "rebuild", full_rebuild_reasons
    if relabel_reasons:
        return "relabel", relabel_reasons
    return "none", []


def _dataset_rebuild_reasons(
    filter_id: str,
    dataset: str,
    *,
    max_tickers: int,
) -> list[str]:
    _mode, reasons = _dataset_refresh_plan(
        filter_id,
        dataset,
        max_tickers=max_tickers,
    )
    return reasons


def _dataset_matches_request(filter_id: str, dataset: str, *, max_tickers: int) -> bool:
    mode, _reasons = _dataset_refresh_plan(
        filter_id,
        dataset,
        max_tickers=max_tickers,
    )
    return mode == "none"


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
    parser.add_argument(
        "--evaluation-batch-size",
        type=int,
        default=int(defaults.evaluation_batch_size),
        help="完整 Train／Validation／Selection 評估的推論 batch size；不改模型訓練",
    )
    parser.add_argument(
        "--evaluation-workers",
        type=int,
        default=int(defaults.evaluation_workers),
        help="完整評估的並行 inference workers；每個 worker 維持單執行緒",
    )
    parser.add_argument(
        "--parallel-split-evaluation",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.parallel_split_evaluation),
        help="是否同時執行 Inner Train 與 Validation 的完整評估",
    )
    parser.add_argument(
        "--train-prefetch-batches",
        type=int,
        default=int(defaults.train_prefetch_batches),
        help="預先準備後續訓練 batches 的數量；0 表示關閉",
    )
    parser.add_argument(
        "--preload-feature-bank",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.preload_feature_bank),
        help="是否在訓練前將去重 feature bank 與事件小型陣列載入 RAM",
    )
    parser.add_argument("--lr", type=float, default=float(defaults.lr))
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=float(defaults.weight_decay),
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=float(defaults.gradient_clip_norm),
    )
    parser.add_argument(
        "--final-refit-mode",
        choices=("matched_optimizer_steps", "selected_epochs"),
        default=str(defaults.final_refit_mode),
    )
    parser.add_argument(
        "--class-weight-mode",
        choices=("none", "inverse_frequency"),
        default=str(defaults.class_weight_mode),
    )
    parser.add_argument(
        "--time-weight-mode",
        choices=("none", "year_balanced_sqrt"),
        default=str(defaults.time_weight_mode),
    )
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
        "--evaluation-batch-size",
        str(int(args.evaluation_batch_size)),
        "--evaluation-workers",
        str(int(args.evaluation_workers)),
        "--train-prefetch-batches",
        str(int(args.train_prefetch_batches)),
        "--lr",
        str(float(args.lr)),
        "--weight-decay",
        str(float(args.weight_decay)),
        "--gradient-clip-norm",
        str(float(args.gradient_clip_norm)),
        "--final-refit-mode",
        str(args.final_refit_mode),
        "--class-weight-mode",
        str(args.class_weight_mode),
        "--time-weight-mode",
        str(args.time_weight_mode),
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
    argv.append(
        "--preload-feature-bank"
        if bool(args.preload_feature_bank)
        else "--no-preload-feature-bank"
    )
    argv.append(
        "--parallel-split-evaluation"
        if bool(args.parallel_split_evaluation)
        else "--no-parallel-split-evaluation"
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

    workflow_started = time.perf_counter()
    color_time = bool(sys.stdout.isatty())
    print("\n=== Breakout Quality Research Workflow ===")
    print(f"filter_id={filter_id}")
    print(f"dataset={args.dataset}")
    model_spec = get_model_spec(BREAKOUT_QUALITY_MODEL_ARCHITECTURE)
    branch_inputs = "+".join(model_spec.branch_input_representations) or "level"
    print(
        "model="
        f"{model_spec.architecture}, branch_inputs={branch_inputs}, "
        f"receptive_field={model_spec.receptive_field_bars} bars, "
        f"pooling={'+'.join(model_spec.pooling)}"
    )
    print(
        "train="
        f"epochs={int(args.epochs)}, batch_size={int(args.batch_size)}, "
        f"evaluation_batch_size={int(args.evaluation_batch_size)}, "
        f"evaluation_workers={int(args.evaluation_workers)}, "
        f"parallel_split_evaluation={bool(args.parallel_split_evaluation)}, "
        f"prefetch={int(args.train_prefetch_batches)}, "
        f"preload_feature_bank={bool(args.preload_feature_bank)}, "
        f"lr={float(args.lr)}, weight_decay={float(args.weight_decay)}, "
        f"gradient_clip_norm={float(args.gradient_clip_norm)}, "
        f"final_refit_mode={args.final_refit_mode}, "
        f"class_weight_mode={args.class_weight_mode}, "
        f"time_weight_mode={args.time_weight_mode}, seed={int(args.seed)}, "
        f"threshold={float(args.fixed_threshold):.6f}, "
        f"inner_validation={bool(args.use_inner_validation)}"
    )
    print("注意：OOS 只供最終泛化評估，不得依結果回頭調整 threshold、epochs 或模型。")

    refresh_mode, refresh_reasons = _dataset_refresh_plan(
        filter_id,
        args.dataset,
        max_tickers=int(args.max_tickers),
    )
    if bool(args.rebuild_dataset):
        refresh_mode = "rebuild"
        refresh_reasons = ["使用者要求強制完整重建 dataset"]
    steps: list[tuple[str, list[str], str]] = []
    if refresh_mode in {"rebuild", "relabel"}:
        tag = "rebuild" if refresh_mode == "rebuild" else "relabel"
        for reason in refresh_reasons:
            print(f"[{tag}] {reason}")
        build_args = ["--dataset", str(args.dataset), "--filter-id", filter_id]
        if int(args.max_tickers) > 0:
            build_args.extend(["--max-tickers", str(int(args.max_tickers))])
        if refresh_mode == "relabel":
            build_args.append("--relabel-only")
            label = "快速更新 labels（沿用 feature bank）"
        else:
            label = "完整建立 indexed feature bank dataset"
        steps.append(("build-dataset", build_args, label))
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
        stage_started = time.perf_counter()
        rc = _run_command(command, command_args, program_name=program_name)
        stage_elapsed = time.perf_counter() - stage_started
        elapsed_text = render_elapsed(stage_elapsed, color=color_time)
        if rc != 0:
            print(f"[失敗] {label} | 耗時 {elapsed_text} | 狀態 {rc}")
            return int(rc)
        print(f"[完成] {label} | 耗時 {elapsed_text}")

    workflow_elapsed = time.perf_counter() - workflow_started
    print(
        "\n=== Workflow 完成 | "
        f"總耗時 {render_elapsed(workflow_elapsed, color=color_time)} ==="
    )
    return 0


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


def _policy_filter_id() -> str:
    return normalize_filter_id(BREAKOUT_QUALITY_DEFAULT_FILTER_ID)


def _policy_train_settings(filter_id: str) -> argparse.Namespace:
    defaults = _train_defaults()
    return argparse.Namespace(
        filter_id=normalize_filter_id(filter_id),
        epochs=int(defaults.epochs),
        batch_size=int(defaults.batch_size),
        evaluation_batch_size=int(defaults.evaluation_batch_size),
        evaluation_workers=int(defaults.evaluation_workers),
        parallel_split_evaluation=bool(defaults.parallel_split_evaluation),
        train_prefetch_batches=int(defaults.train_prefetch_batches),
        preload_feature_bank=bool(defaults.preload_feature_bank),
        lr=float(defaults.lr),
        weight_decay=float(defaults.weight_decay),
        gradient_clip_norm=float(defaults.gradient_clip_norm),
        final_refit_mode=str(defaults.final_refit_mode),
        class_weight_mode=str(defaults.class_weight_mode),
        time_weight_mode=str(defaults.time_weight_mode),
        seed=int(defaults.seed),
        fixed_threshold=float(defaults.fixed_threshold),
        use_inner_validation=bool(defaults.use_inner_validation),
        inner_validation_months=int(defaults.inner_validation_months),
        early_stopping_patience=int(defaults.early_stopping_patience),
        early_stopping_min_delta=float(defaults.early_stopping_min_delta),
    )


def _print_policy_defaults(
    filter_id: str,
    train_settings: argparse.Namespace | None = None,
) -> None:
    print(f"使用 policy Filter ID：{normalize_filter_id(filter_id)}")
    print("使用 config/breakout_quality_policy.py Label 預設：")
    print(
        f"- Feature Window：{int(DEFAULT_LABEL_POLICY.feature_window_bars)} bars\n"
        f"- Label Horizon：{int(DEFAULT_LABEL_POLICY.label_horizon_bars)} bars\n"
        f"- 最低 MFE：>{float(DEFAULT_LABEL_POLICY.min_mfe_return) * 100:g}%\n"
        f"- 最低 MFE/MAE：>{float(DEFAULT_LABEL_POLICY.min_reward_risk_ratio):g}\n"
        f"- 最大不利跌幅：{float(DEFAULT_LABEL_POLICY.max_adverse_return) * 100:g}%（觸及即 REJECT）"
    )
    if train_settings is None:
        return
    print("使用 config/breakout_quality_policy.py 訓練預設：")
    model_spec = get_model_spec(BREAKOUT_QUALITY_MODEL_ARCHITECTURE)
    branch_inputs = "+".join(model_spec.branch_input_representations) or "level"
    print(
        f"- Model Architecture：{model_spec.architecture}\n"
        f"- Branch Inputs：{branch_inputs}\n"
        f"- Receptive Field：約 {model_spec.receptive_field_bars} bars\n"
        f"- Pooling：{'+'.join(model_spec.pooling)}\n"
        f"- Epoch 上限：{int(train_settings.epochs)}\n"
        f"- Batch Size：{int(train_settings.batch_size)}\n"
        f"- Evaluation Batch Size：{int(train_settings.evaluation_batch_size)}\n"
        f"- Evaluation Workers：{int(train_settings.evaluation_workers)}\n"
        f"- Parallel Split Evaluation：{'開啟' if bool(train_settings.parallel_split_evaluation) else '關閉'}\n"
        f"- Train Prefetch Batches：{int(train_settings.train_prefetch_batches)}\n"
        f"- Preload Feature Bank：{'開啟' if bool(train_settings.preload_feature_bank) else '關閉'}\n"
        f"- Learning Rate：{float(train_settings.lr):g}\n"
        f"- Weight Decay：{float(train_settings.weight_decay):g}\n"
        f"- Gradient Clip Norm：{float(train_settings.gradient_clip_norm):g}\n"
        f"- Final Refit Mode：{train_settings.final_refit_mode}\n"
        f"- Class Weight Mode：{train_settings.class_weight_mode}\n"
        f"- Time Weight Mode：{train_settings.time_weight_mode}\n"
        f"- Random Seed：{int(train_settings.seed)}\n"
        f"- Threshold：{float(train_settings.fixed_threshold):g}\n"
        f"- Inner Validation：{'開啟' if bool(train_settings.use_inner_validation) else '關閉'}"
    )
    if bool(train_settings.use_inner_validation):
        print(
            f"- Inner Validation 月數：{int(train_settings.inner_validation_months)}\n"
            f"- Early Stopping Patience：{int(train_settings.early_stopping_patience)}\n"
            f"- Early Stopping Min Delta：{float(train_settings.early_stopping_min_delta):g}"
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
    print(
        f"\n=== Artifact Status: {filter_id} / "
        f"{model_paths.model_architecture} ==="
    )
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
    filter_id = _policy_filter_id()
    train_args = _policy_train_settings(filter_id)
    _print_policy_defaults(filter_id, train_args)
    dataset = INTERACTIVE_DATASET_PROFILE
    max_tickers = INTERACTIVE_MAX_TICKERS
    evaluate_oos = INTERACTIVE_EVALUATE_OOS
    print("完整研究流程固定使用 Full dataset、全部股票，並執行 OOS。")
    refresh_mode, refresh_reasons = _dataset_refresh_plan(
        filter_id,
        dataset,
        max_tickers=max_tickers,
    )
    if refresh_mode == "rebuild":
        print("偵測到 dataset 需要完整重建，workflow 將自動重建：")
        for reason in refresh_reasons:
            print(f"- {reason}")
        rebuild_dataset = False
    elif refresh_mode == "relabel":
        print("偵測到只有 Label policy 改變，workflow 將沿用 feature bank 快速 relabel：")
        for reason in refresh_reasons:
            print(f"- {reason}")
        rebuild_dataset = False
    else:
        print("dataset 自動偵測：目前工件與來源資料一致。")
        rebuild_dataset = _prompt_bool(
            "是否強制完整重建 dataset（即使目前不需要）",
            False,
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
    print("\n即將執行：Full dataset（全部股票）→ train → export research scores → OOS 易讀研究報表")
    if evaluate_oos:
        print("報表將納入 OOS 最終泛化評估。")
    if not _prompt_bool("確認開始", True):
        print("已取消。")
        return 0
    return _run_workflow(request, program_name=program_name)


def _interactive_build_dataset(program_name: str) -> int:
    filter_id = _policy_filter_id()
    _print_policy_defaults(filter_id)
    dataset = INTERACTIVE_DATASET_PROFILE
    max_tickers = INTERACTIVE_MAX_TICKERS
    print("建立 dataset 固定使用 Full dataset 與全部股票。")
    if not _prompt_bool("確認建立／覆蓋 dataset 工件", False):
        print("已取消。")
        return 0
    argv = ["--dataset", dataset, "--filter-id", filter_id]
    if max_tickers > 0:
        argv.extend(["--max-tickers", str(max_tickers)])
    return _run_command("build-dataset", argv, program_name=program_name)


def _interactive_train(program_name: str) -> int:
    filter_id = _policy_filter_id()
    request = _policy_train_settings(filter_id)
    _print_policy_defaults(filter_id, request)
    if not _prompt_bool("確認開始訓練（既有同 filter_id 模型會更新）", False):
        print("已取消。")
        return 0
    return _run_command(
        "train",
        _build_train_argv(request),
        program_name=program_name,
    )


def _interactive_export_research(program_name: str) -> int:
    filter_id = _policy_filter_id()
    _print_policy_defaults(filter_id)
    return _run_command(
        "export-scores",
        ["--filter-id", filter_id, "--scope", "research"],
        program_name=program_name,
    )


def _interactive_report(program_name: str) -> int:
    filter_id = _policy_filter_id()
    _print_policy_defaults(filter_id)
    print(
        "易讀研究報表固定納入最終 OOS；不得依同一段 OOS 回頭調整 "
        "threshold、epochs、learning rate、feature、label 或模型。"
    )
    return _run_command(
        "report",
        ["--filter-id", filter_id, "--include-oos"],
        program_name=program_name,
    )


def _interactive_evaluate(program_name: str) -> int:
    filter_id = _policy_filter_id()
    _print_policy_defaults(filter_id)
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
    filter_id = _policy_filter_id()
    _print_policy_defaults(filter_id)
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
    print("[1] 完整研究流程（Full／全部股票／OOS）")
    print("[2] 建立／重建 Full dataset（全部股票）")
    print("[3] 訓練模型（使用 policy 預設參數）")
    print("[4] 匯出 research scores")
    print("[5] 產生易讀研究報表（固定納入 OOS）")
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
                filter_id = _policy_filter_id()
                _print_policy_defaults(filter_id)
                _print_artifact_status(filter_id)
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
