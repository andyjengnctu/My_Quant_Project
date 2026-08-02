"""Breakout quality dataset、training、score export、report 與 evaluation 正式入口。"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.breakout_quality import (
    SUPPORTED_BREAKOUT_QUALITY_CLASSIFICATION_EXPERIMENT_PROFILES,
    SUPPORTED_BREAKOUT_QUALITY_TIME_WEIGHT_MODES,
    build_breakout_quality_pretraining_profile_payload,
    get_breakout_quality_experiment_profile,
)
from config.breakout_quality import (
    BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    BREAKOUT_QUALITY_PRETRAINING_FAMILY,
    BREAKOUT_QUALITY_PRETRAINING_PROFILE,
    BREAKOUT_QUALITY_PRETRAINING_STRIDE,
    BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
    BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
    BREAKOUT_QUALITY_INCEPTION_TARGET_RECEPTIVE_FIELD_BARS,
)
from config.breakout_quality import get_breakout_quality_workflow_settings
from core.display_common import render_elapsed
from core.runtime_utils import (
    is_interactive_console,
    resolve_cli_program_name,
    run_cli_entrypoint,
)
from filters.breakout_quality.contract import CONTEXT_COLUMNS, DEFAULT_LABEL_POLICY, FEATURE_COLUMNS
from filters.breakout_quality.continuous_target import (
    TARGET_AUDIT_MARKDOWN_FILENAME,
    TARGET_MANIFEST_FILENAME,
    resolve_continuous_target_dir,
)
from filters.breakout_quality.mantis_contract import require_mantis_v2_class
from filters.breakout_quality.moment_contract import (
    MOMENT_PACKAGE_NAME,
    MOMENT_PACKAGE_VERSION,
    MOMENT_TRANSFORMERS_PACKAGE_NAME,
    MOMENT_TRANSFORMERS_VERSION,
    require_moment_pipeline_class,
)
from filters.breakout_quality.models.spec import get_model_spec
from filters.breakout_quality.market_set import market_set_contract_payload
from filters.breakout_quality.dataset_store import (
    DATASET_STORAGE_FORMAT,
    DATASET_STORAGE_SCHEMA_VERSION,
    dataset_artifact_metadata_reasons,
    market_set_artifact_metadata_reasons,
    resolve_dataset_paths,
)
from filters.breakout_quality.pretraining_store import (
    load_validated_pretrained_encoder_manifest,
    load_validated_pretraining_dataset,
    resolve_pretrained_encoder_paths,
)
from filters.breakout_quality.splits import resolve_breakout_quality_outer_policy
from filters.breakout_quality.paths import (
    normalize_filter_id,
    resolve_existing_filter_artifact_paths,
    resolve_existing_filter_research_manifest_path,
    resolve_existing_filter_research_score_path,
    resolve_filter_output_dir,
    resolve_filter_report_json_path,
    resolve_filter_report_markdown_path,
    resolve_selection_point_in_time_audit_json_path,
    resolve_selection_point_in_time_audit_markdown_path,
    resolve_selection_point_in_time_coverage_path,
    resolve_selection_point_in_time_manifest_path,
    resolve_selection_point_in_time_score_path,
)
from filters.breakout_quality.source_inventory import build_source_data_inventory
from filters.breakout_quality.console_report import (
    COMPACT_CONSOLE_ENV,
    console_color_enabled,
    paint,
    project_relative_display_path,
    render_key_values,
    render_section,
    render_status_paths,
    render_table,
    render_title,
)


COMMAND_MODULES = {
    "build-dataset": "tools.filters.breakout_quality.build_dataset",
    "build-pretrain-dataset": "tools.filters.breakout_quality.build_pretraining_dataset",
    "pretrain": "tools.filters.breakout_quality.pretrain",
    "train": "tools.filters.breakout_quality.train",
    "export-scores": "tools.filters.breakout_quality.export_scores",
    "report": "tools.filters.breakout_quality.report",
    "evaluate": "tools.filters.breakout_quality.evaluate",
    "regime-audit": "tools.filters.breakout_quality.regime_audit",
    "audit-continuous-target": "tools.filters.breakout_quality.audit_continuous_target",
    "prepare-continuous-target": "tools.filters.breakout_quality.prepare_continuous_target",
    "train-continuous-ranker": "tools.filters.breakout_quality.train_continuous_ranker",
    "audit-qualified-candidate-set": "tools.filters.breakout_quality.audit_qualified_candidate_set",
    "audit-target-attribution": "tools.filters.breakout_quality.audit_target_component_attribution",
    "audit-target-time-ablation": "tools.filters.breakout_quality.audit_target_time_penalty_ablation",
    "audit-no-time-target": "tools.filters.breakout_quality.audit_no_time_continuous_target",
    "audit-pass-realization-gap": "tools.filters.breakout_quality.audit_pass_realization_gap",
    "audit-selection-strategy-realization": "tools.filters.breakout_quality.audit_selection_strategy_realization",
    "audit-candidate-counterfactual": "tools.filters.breakout_quality.audit_candidate_counterfactual_execution",
    "audit-selection-pressure": "tools.filters.breakout_quality.audit_portfolio_selection_pressure",
    "build-point-in-time-scores": "tools.filters.breakout_quality.build_point_in_time_scores",
    "audit-point-in-time-scores": "tools.filters.breakout_quality.audit_point_in_time_scores",
    "strategy-compare": "tools.filters.breakout_quality.strategy_compare",
    "strategy-adapt": "tools.filters.breakout_quality.strategy_adapt",
}

INTERACTIVE_DATASET_PROFILE = "full"
INTERACTIVE_MAX_TICKERS = 0
INTERACTIVE_EVALUATE_OOS = True



COMMAND_DESCRIPTIONS = {
    "menu": "開啟互動式操作選單",
    "workflow": "依序執行 dataset、必要的Selection-only預訓練、train、research score export與報表",
    "build-dataset": "建立 breakout quality event dataset",
    "build-pretrain-dataset": "建立 Selection-only self-supervised rolling windows",
    "pretrain": "訓練 Selection-only TS2Vec encoder",
    "train": "訓練模型；可選擇 inner validation 選 epoch 後完整 Selection 重訓",
    "export-scores": "匯出 research 或 forward-OOS score table",
    "report": "產生表格化終端報表、Markdown 報表與完整 metrics JSON",
    "evaluate": "輸出 train、validation、selection 或 OOS 的詳細 JSON",
    "regime-audit": "稽核 Selection／OOS 的市場狀態與 breakout event 覆蓋",
    "audit-continuous-target": "建立11A連續target arrays並稽核分布、同日排序與實際R方向",
    "prepare-continuous-target": "依目前workflow檢查並建立continuous target工件",
    "train-continuous-ranker": "執行11B／11G continuous ranker研究；research-only、CLI-only",
    "audit-qualified-candidate-set": "執行11C策略qualified candidate-set失敗歸因；research-only",
    "audit-target-attribution": "執行11D Target成分與Label條件失敗歸因；research-only",
    "audit-target-time-ablation": "執行11E固定移除time penalty的Target稽核；research-only",
    "audit-no-time-target": "建立11F No-time Target arrays並做Selection-only可學性稽核；research-only",
    "audit-pass-realization-gap": "執行11H PASS-only實現落差歸因；research-only、CLI-only",
    "audit-selection-strategy-realization": "執行11I Selection nested-OOS策略實現覆蓋稽核；research-only、CLI-only",
    "audit-candidate-counterfactual": "執行11J per-candidate counterfactual execution稽核；已停止、僅供歷史追溯",
    "audit-selection-pressure": "執行11K portfolio selection-pressure歸因；read-only、CLI-only",
    "build-point-in-time-scores": "建立泛用Selection point-in-time continuous-ranker scores",
    "audit-point-in-time-scores": "驗證point-in-time Score的Target排序能力與fold穩定性",
    "strategy-compare": "執行breakout-quality策略績效比較",
    "strategy-adapt": "驗證Selection rolling Ranking×Parameter 2×2策略適應",
}


def _print_help(program_name: str) -> None:
    print(f"用法: python {program_name} [menu|workflow|<command>] [options]")
    print("說明: Breakout quality filter 的單一正式操作入口；互動終端不帶參數時直接開啟選單。")
    print("command:")
    for command, description in COMMAND_DESCRIPTIONS.items():
        print(f"  {command:<30} {description}")
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


def _command_program_name(program_name: str, command: str) -> str:
    normalized_program_name = str(program_name).replace("\\", "/").strip()
    return f"{normalized_program_name} {command}"


def _run_command(command: str, args: list[str], *, program_name: str) -> int:
    command_module = _load_command_module(command)
    original_program_name = sys.argv[0]
    sys.argv[0] = _command_program_name(program_name, command)
    try:
        result = command_module.main(list(args))
    finally:
        sys.argv[0] = original_program_name
    return int(result or 0)


@contextmanager
def _compact_console_scope():
    previous = os.environ.get(COMPACT_CONSOLE_ENV)
    os.environ[COMPACT_CONSOLE_ENV] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(COMPACT_CONSOLE_ENV, None)
        else:
            os.environ[COMPACT_CONSOLE_ENV] = previous


def _compact_dataset_refresh_reason(reasons: list[str]) -> str:
    """Collapse technical rebuild diagnostics into user-facing categories."""

    text = " ".join(dict.fromkeys(str(reason).strip() for reason in reasons if str(reason).strip()))
    labels: list[str] = []

    def add(label: str) -> None:
        if label not in labels:
            labels.append(label)

    lowered = text.lower()
    if "使用者要求" in text:
        add("使用者要求重建")
    if any(token in lowered for token in ("工件缺少", "artifact", "summary.json", "損壞")):
        add("既有工件不完整")
    if "source_data_inventory" in lowered or "來源 csv inventory" in lowered:
        add("來源資料已更新")
    if any(token in lowered for token in ("profile 不符", "ticker coverage 不符")):
        add("資料範圍不符")
    if any(
        token in lowered
        for token in (
            "schema 已變更",
            "format 已變更",
            "contract 已變更",
            "policy 已變更",
            "policy metadata",
            "label horizon",
        )
    ):
        add("設定已變更")
    return "、".join(labels) if labels else "既有 Dataset 需更新"


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
    model_spec = get_model_spec(BREAKOUT_QUALITY_MODEL_ARCHITECTURE)
    if bool(model_spec.requires_market_set):
        if summary.get("market_set_contract") != market_set_contract_payload():
            full_rebuild_reasons.append("market-set input contract 已變更或缺少")
        full_rebuild_reasons.extend(
            market_set_artifact_metadata_reasons(
                storage_paths,
                summary.get("market_set_artifacts"),
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


def _dataset_refresh_step(
    filter_id: str,
    dataset: str,
    *,
    max_tickers: int,
    force_rebuild: bool = False,
) -> tuple[str, list[str], tuple[str, list[str], str] | None]:
    """Resolve the single dataset preparation step shared by all workflows."""
    refresh_mode, refresh_reasons = _dataset_refresh_plan(
        filter_id,
        dataset,
        max_tickers=max_tickers,
    )
    if force_rebuild:
        refresh_mode = "rebuild"
        refresh_reasons = ["使用者要求強制完整重建 dataset"]
    if refresh_mode not in {"rebuild", "relabel"}:
        return refresh_mode, refresh_reasons, None

    build_args = ["--dataset", str(dataset), "--filter-id", filter_id]
    if int(max_tickers) > 0:
        build_args.extend(["--max-tickers", str(int(max_tickers))])
    if refresh_mode == "relabel":
        build_args.append("--relabel-only")
        label = "快速更新 labels（沿用 feature bank）"
    else:
        label = "完整建立 indexed feature bank dataset"
    return refresh_mode, refresh_reasons, ("build-dataset", build_args, label)


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
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=str(defaults.device))
    parser.add_argument(
        "--mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.mixed_precision),
        help="CUDA 上是否啟用 mixed precision",
    )
    parser.add_argument(
        "--mixed-precision-dtype",
        choices=("auto", "float16", "bfloat16"),
        default=str(defaults.mixed_precision_dtype),
    )
    parser.add_argument(
        "--deterministic-algorithms",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.deterministic_algorithms),
    )
    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.allow_tf32),
    )
    parser.add_argument(
        "--experiment-profile",
        choices=SUPPORTED_BREAKOUT_QUALITY_CLASSIFICATION_EXPERIMENT_PROFILES,
        default=str(defaults.experiment_profile),
        help="訓練實驗 profile；模型架構與訓練實驗分開管理",
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
        choices=SUPPORTED_BREAKOUT_QUALITY_TIME_WEIGHT_MODES,
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
        "--experiment-profile",
        str(args.experiment_profile),
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
        "--device",
        str(args.device),
        "--mixed-precision-dtype",
        str(args.mixed_precision_dtype),
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
    argv.append("--mixed-precision" if bool(args.mixed_precision) else "--no-mixed-precision")
    argv.append(
        "--deterministic-algorithms"
        if bool(args.deterministic_algorithms)
        else "--no-deterministic-algorithms"
    )
    argv.append("--allow-tf32" if bool(args.allow_tf32) else "--no-allow-tf32")
    return argv


def _pretraining_refresh_plan(
    *,
    filter_id: str,
    dataset: str,
    model_spec,
    experiment_profile: str,
    max_tickers: int,
) -> tuple[bool, bool, list[str]]:
    if str(model_spec.family) != "ts2vec_frozen_linear":
        return False, False, []
    reasons: list[str] = []
    summary = _read_dataset_summary(filter_id)
    if summary is None:
        return True, True, ["supervised dataset summary 尚不可用"]
    date_range = summary.get("source_data_date_range")
    source_end = str(date_range.get("end") or "").strip() if isinstance(date_range, dict) else ""
    if not source_end:
        return True, True, ["supervised dataset summary 缺少 source_data_end"]
    outer_policy = resolve_breakout_quality_outer_policy(
        PROJECT_ROOT,
        source_data_end_date=source_end,
    )
    try:
        pretrain_summary, _windows, _index = load_validated_pretraining_dataset(
            PROJECT_ROOT,
            filter_id,
            dataset_profile=str(dataset),
            family=BREAKOUT_QUALITY_PRETRAINING_FAMILY,
            stride=int(BREAKOUT_QUALITY_PRETRAINING_STRIDE),
            expected_selection_start=str(outer_policy["selection_start_date"]),
            expected_selection_end=str(outer_policy["selection_end_date"]),
            expected_window_bars=int(DEFAULT_LABEL_POLICY.feature_window_bars),
            expected_max_tickers=max(0, int(max_tickers)),
            require_current_source=True,
            load_windows=False,
        )
        rebuild_dataset = False
    except (OSError, ValueError, FileNotFoundError) as exc:
        reasons.append(f"pretraining dataset 需重建: {type(exc).__name__}: {exc}")
        return True, True, reasons
    encoder_paths = resolve_pretrained_encoder_paths(
        PROJECT_ROOT,
        filter_id,
        model_architecture=model_spec.architecture,
        experiment_profile=experiment_profile,
    )
    try:
        load_validated_pretrained_encoder_manifest(
            encoder_paths,
            expected_architecture=model_spec.architecture,
            expected_experiment_profile=experiment_profile,
            expected_dataset_fingerprint=str(pretrain_summary["configuration_fingerprint"]),
            expected_model_spec=model_spec.as_manifest_payload(),
            expected_pretraining_profile=(
                build_breakout_quality_pretraining_profile_payload(
                    BREAKOUT_QUALITY_PRETRAINING_PROFILE
                )
            ),
        )
        rebuild_encoder = False
    except (OSError, ValueError, FileNotFoundError) as exc:
        reasons.append(f"pretrained encoder 需重訓: {type(exc).__name__}: {exc}")
        rebuild_encoder = True
    return rebuild_dataset, rebuild_encoder, reasons


def _model_runtime_description(model_spec) -> str:
    if str(model_spec.family) == "moment_frozen_linear":
        return (
            f"family=moment_frozen_linear, source={model_spec.moment_repository}, "
            f"revision={model_spec.moment_revision}, "
            f"runtime={MOMENT_PACKAGE_NAME}-{MOMENT_PACKAGE_VERSION}/"
            f"{MOMENT_TRANSFORMERS_PACKAGE_NAME}-{MOMENT_TRANSFORMERS_VERSION}, "
            f"input_resize={model_spec.moment_input_length}, "
            f"patch={model_spec.moment_patch_length}/{model_spec.moment_patch_stride}, "
            f"embedding={model_spec.moment_embedding_dim}, "
            f"layers={model_spec.moment_transformer_layers}, heads={model_spec.moment_transformer_heads}, "
            f"channel_aggregation={model_spec.moment_channel_aggregation}, "
            f"patch_reduction={model_spec.moment_patch_reduction}, "
            "encoder=frozen, head=linear, dataset_context=disabled"
        )
    if str(model_spec.family) == "mantis_v2_frozen_linear":
        return (
            f"family=mantis_v2_frozen_linear, source={model_spec.mantis_repository}, "
            f"revision={model_spec.mantis_revision}, input_resize={model_spec.mantis_input_length}, "
            f"patches={model_spec.mantis_num_patches}, output_layer={model_spec.mantis_return_transformer_layer}, "
            f"output_token={model_spec.mantis_output_token}, "
            f"channel_aggregation={model_spec.mantis_channel_aggregation}, "
            "encoder=frozen, head=linear, dataset_context=disabled"
        )
    if str(model_spec.family) == "ts2vec_frozen_linear":
        return (
            f"family=ts2vec_frozen_linear, depth={model_spec.ts2vec_depth}, "
            f"hidden={model_spec.ts2vec_hidden_dims}, output={model_spec.ts2vec_output_dims}, "
            f"dilations={'/'.join(str(value) for value in model_spec.dilations)}, "
            f"encoder=frozen, head=linear, "
            f"dataset_context={'enabled' if model_spec.use_dataset_context else 'disabled'}, "
            f"pooling={'+'.join(model_spec.pooling)}"
        )
    if str(model_spec.family) == "patch_transformer":
        return (
            f"family=patch_transformer, patch={model_spec.patch_transformer_patch_size}/"
            f"{model_spec.patch_transformer_patch_stride}, "
            f"embedding={model_spec.patch_transformer_embedding_dim}, "
            f"depth={model_spec.patch_transformer_depth}, heads={model_spec.patch_transformer_heads}, "
            f"mlp={model_spec.patch_transformer_mlp_dim}, "
            f"position={model_spec.patch_transformer_positional_encoding}, "
            f"dataset_context={'enabled' if model_spec.use_dataset_context else 'disabled'}, "
            f"pooling={model_spec.patch_transformer_pooling}"
        )
    if str(model_spec.family) == "inception_time_market_set":
        kernels = "/".join(str(value) for value in model_spec.inception_kernel_sizes)
        query_mode = str(model_spec.market_set_query_mode or "global_learned")
        return (
            f"family=inception_time_market_set, candidate_depth={model_spec.inception_depth}, "
            f"candidate_kernels={kernels}, candidate_rf={model_spec.receptive_field_bars} bars, "
            f"market_history={model_spec.market_set_history_bars} bars, "
            f"stock_embedding={model_spec.market_set_stock_embedding_dim}, "
            f"market_norm={model_spec.market_set_temporal_normalization}/"
            f"{model_spec.market_set_temporal_normalization_groups}, "
            f"query_mode={query_mode}, queries={model_spec.market_set_query_count}, "
            f"heads={model_spec.market_set_attention_heads}, "
            f"market_embedding={model_spec.market_set_embedding_dim}, dataset_context=disabled"
        )
    if str(model_spec.family) == "inception_time":
        kernels = "/".join(str(value) for value in model_spec.inception_kernel_sizes)
        target_fragment = (
            f"target_receptive_field={int(BREAKOUT_QUALITY_INCEPTION_TARGET_RECEPTIVE_FIELD_BARS)} bars, "
            if str(model_spec.architecture) == "inception_time_v1"
            else ""
        )
        return (
            f"family=inception_time, depth={model_spec.inception_depth}, "
            f"filters={model_spec.inception_filters}, "
            f"bottleneck={model_spec.inception_bottleneck_channels}, "
            f"kernels={kernels}, residual_every={model_spec.inception_residual_every}, "
            f"{target_fragment}"
            f"dataset_context={'enabled' if model_spec.use_dataset_context else 'disabled'}, "
            f"receptive_field={model_spec.receptive_field_bars} bars, "
            f"pooling={'+'.join(model_spec.pooling)}"
        )
    if str(model_spec.family) == "modern_tcn":
        return (
            f"family=modern_tcn, depth={model_spec.modern_tcn_depth}, "
            f"channels={model_spec.modern_tcn_channels}, "
            f"large_kernel={model_spec.modern_tcn_kernel_size}, "
            f"expansion={model_spec.modern_tcn_expansion_ratio}x, "
            f"normalization={model_spec.normalization}, "
            f"dataset_context={'enabled' if model_spec.use_dataset_context else 'disabled'}, "
            f"receptive_field={model_spec.receptive_field_bars} bars, "
            f"pooling={'+'.join(model_spec.pooling)}"
        )
    branch_inputs = "+".join(model_spec.branch_input_representations) or "level"
    branch_channels = model_spec.branch_channels or (model_spec.channels,) * 3
    derived_context = ",".join(model_spec.derived_context_features) or "none"
    return (
        f"branch_inputs={branch_inputs}, "
        f"branch_channels={'/'.join(str(value) for value in branch_channels)}, "
        f"dataset_context={'enabled' if model_spec.use_dataset_context else 'disabled'}, "
        f"derived_context={derived_context}, "
        f"receptive_field={model_spec.receptive_field_bars} bars, "
        f"pooling={'+'.join(model_spec.pooling)}"
    )


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
    model_spec = get_model_spec(BREAKOUT_QUALITY_MODEL_ARCHITECTURE)
    if str(model_spec.family) == "moment_frozen_linear":
        require_moment_pipeline_class()
    if str(model_spec.family) == "mantis_v2_frozen_linear":
        require_mantis_v2_class()
    experiment = get_breakout_quality_experiment_profile(args.experiment_profile)
    schedule_parameters = experiment.lr_schedule_parameters()
    augmentation_parameters = experiment.augmentation_parameters()
    workflow_rows = [
        ("Filter ID", filter_id),
        ("Dataset", args.dataset),
        ("Model Contract", f"model={model_spec.architecture}, {_model_runtime_description(model_spec)}"),
        ("Experiment Profile", experiment.name),
        ("Optimizer", experiment.optimizer_name),
        ("LR Schedule", f"{experiment.lr_schedule_name} / {schedule_parameters or '-'}"),
        ("Augmentation", f"{experiment.augmentation_name} / {augmentation_parameters or '-'}"),
        ("Training Sampling", experiment.training_sampling_mode),
        ("Epoch／Batch", f"{int(args.epochs)} / {int(args.batch_size)}"),
        ("Evaluation", f"batch={int(args.evaluation_batch_size)}, workers={int(args.evaluation_workers)}, parallel={bool(args.parallel_split_evaluation)}"),
        ("Feature Bank", f"preload={bool(args.preload_feature_bank)}, prefetch={int(args.train_prefetch_batches)}"),
        ("LR／Weight Decay", f"{float(args.lr):g} / {float(args.weight_decay):g}"),
        ("Final Refit", args.final_refit_mode),
        ("Class／Time Weight", f"{args.class_weight_mode} / {args.time_weight_mode}"),
        ("Seed／Threshold", f"{int(args.seed)} / {float(args.fixed_threshold):g}"),
        ("Inner Validation", bool(args.use_inner_validation)),
        ("Torch", f"device={args.device}, mixed_precision={bool(args.mixed_precision)}, dtype={args.mixed_precision_dtype}, deterministic={bool(args.deterministic_algorithms)}, tf32={bool(args.allow_tf32)}"),
    ]
    if str(model_spec.family) == "moment_frozen_linear":
        workflow_rows.append((
            "External Encoder",
            f"external_encoder=repository={model_spec.moment_repository}, revision={model_spec.moment_revision}, runtime={MOMENT_PACKAGE_NAME}-{MOMENT_PACKAGE_VERSION}/{MOMENT_TRANSFORMERS_PACKAGE_NAME}-{MOMENT_TRANSFORMERS_VERSION}, project_pretraining=False, encoder_fine_tuning=False",
        ))
    elif str(model_spec.family) == "mantis_v2_frozen_linear":
        workflow_rows.append((
            "External Encoder",
            f"external_encoder=repository={model_spec.mantis_repository}, revision={model_spec.mantis_revision}, project_pretraining=False, encoder_fine_tuning=False",
        ))
    elif str(model_spec.family) == "ts2vec_frozen_linear":
        workflow_rows.append((
            "Selection Pretrain",
            f"pretrain=profile={BREAKOUT_QUALITY_PRETRAINING_PROFILE}, stride={int(BREAKOUT_QUALITY_PRETRAINING_STRIDE)}, settings={build_breakout_quality_pretraining_profile_payload(BREAKOUT_QUALITY_PRETRAINING_PROFILE)}, selection_only=True, oos_windows=False, pass_reject_labels=False",
        ))
    print("\n" + render_title("Breakout Quality Research Workflow"))
    print(render_key_values(workflow_rows))
    print(render_section("研究邊界"))
    print("OOS 只供最終泛化評估，不得依結果回頭調整 threshold、epochs 或模型。")

    refresh_mode, refresh_reasons, dataset_step = _dataset_refresh_step(
        filter_id,
        args.dataset,
        max_tickers=int(args.max_tickers),
        force_rebuild=bool(args.rebuild_dataset),
    )
    steps: list[tuple[str, list[str], str]] = []
    if dataset_step is not None:
        tag = "rebuild" if refresh_mode == "rebuild" else "relabel"
        for reason in refresh_reasons:
            print(f"[{tag}] {reason}")
        steps.append(dataset_step)
    else:
        print("[skip] dataset 工件、來源 CSV inventory、ticker coverage 與 policy 均未變更。")

    if str(model_spec.family) == "ts2vec_frozen_linear":
        if refresh_mode in {"rebuild", "relabel"}:
            rebuild_pretrain_dataset = True
            rebuild_pretrained_encoder = True
            pretraining_reasons = ["supervised dataset 將更新，pretraining chain 必須同步重建"]
        else:
            rebuild_pretrain_dataset, rebuild_pretrained_encoder, pretraining_reasons = (
                _pretraining_refresh_plan(
                    filter_id=filter_id,
                    dataset=str(args.dataset),
                    model_spec=model_spec,
                    experiment_profile=str(args.experiment_profile),
                    max_tickers=int(args.max_tickers),
                )
            )
        for reason in pretraining_reasons:
            print(f"[pretraining] {reason}")
        if rebuild_pretrain_dataset:
            build_pretrain_args = [
                "--dataset",
                str(args.dataset),
                "--filter-id",
                filter_id,
                "--stride",
                str(int(BREAKOUT_QUALITY_PRETRAINING_STRIDE)),
            ]
            if int(args.max_tickers) > 0:
                build_pretrain_args.extend(["--max-tickers", str(int(args.max_tickers))])
            steps.append(
                (
                    "build-pretrain-dataset",
                    build_pretrain_args,
                    "建立 Selection-only TS2Vec rolling windows",
                )
            )
        if rebuild_pretrain_dataset or rebuild_pretrained_encoder:
            pretrain_args = [
                "--dataset",
                str(args.dataset),
                "--filter-id",
                filter_id,
                "--stride",
                str(int(BREAKOUT_QUALITY_PRETRAINING_STRIDE)),
                "--experiment-profile",
                str(args.experiment_profile),
                "--pretraining-profile",
                str(BREAKOUT_QUALITY_PRETRAINING_PROFILE),
                "--device",
                str(args.device),
                "--mixed-precision-dtype",
                str(args.mixed_precision_dtype),
                "--seed",
                str(int(args.seed)),
            ]
            pretrain_args.append(
                "--mixed-precision" if bool(args.mixed_precision) else "--no-mixed-precision"
            )
            pretrain_args.append(
                "--deterministic-algorithms"
                if bool(args.deterministic_algorithms)
                else "--no-deterministic-algorithms"
            )
            pretrain_args.append(
                "--allow-tf32" if bool(args.allow_tf32) else "--no-allow-tf32"
            )
            steps.append(("pretrain", pretrain_args, "Selection-only TS2Vec encoder 預訓練"))
        else:
            print("[skip] Selection-only pretraining dataset 與 pretrained encoder 均符合目前契約。")

    report_args = [
        "--filter-id",
        filter_id,
        "--experiment-profile",
        str(args.experiment_profile),
    ]
    report_args.append("--include-oos" if bool(args.evaluate_oos) else "--no-include-oos")
    steps.extend(
        [
            ("train", _build_train_argv(args), "訓練並產生正式模型"),
            (
                "export-scores",
                [
                    "--filter-id",
                    filter_id,
                    "--experiment-profile",
                    str(args.experiment_profile),
                    "--scope",
                    "research",
                    "--device",
                    str(args.device),
                    "--mixed-precision-dtype",
                    str(args.mixed_precision_dtype),
                    (
                        "--mixed-precision"
                        if bool(args.mixed_precision)
                        else "--no-mixed-precision"
                    ),
                    (
                        "--deterministic-algorithms"
                        if bool(args.deterministic_algorithms)
                        else "--no-deterministic-algorithms"
                    ),
                    "--allow-tf32" if bool(args.allow_tf32) else "--no-allow-tf32",
                ],
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
        experiment_profile=str(defaults.experiment_profile),
        optimizer_name=str(defaults.optimizer_name),
        lr_schedule_name=str(defaults.lr_schedule_name),
        augmentation_name=str(defaults.augmentation_name),
        training_sampling_mode=str(defaults.training_sampling_mode),
        training_weight_reduction=str(defaults.training_weight_reduction),
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
        device=str(defaults.device),
        mixed_precision=bool(defaults.mixed_precision),
        mixed_precision_dtype=str(defaults.mixed_precision_dtype),
        deterministic_algorithms=bool(defaults.deterministic_algorithms),
        allow_tf32=bool(defaults.allow_tf32),
    )


def _print_policy_defaults(
    filter_id: str,
    train_settings: argparse.Namespace | None = None,
) -> None:
    print("\n" + render_title("Breakout Quality Policy"))
    print(render_key_values((("Filter ID", normalize_filter_id(filter_id)),)))
    print(render_section("Label 設定", number=1))
    print(render_key_values((
        ("Feature Window", f"{int(DEFAULT_LABEL_POLICY.feature_window_bars)} bars"),
        ("Label Horizon", f"{int(DEFAULT_LABEL_POLICY.label_horizon_bars)} bars"),
        ("最低 MFE", f">{float(DEFAULT_LABEL_POLICY.min_mfe_return) * 100:g}%"),
        ("最低 MFE／MAE", f">{float(DEFAULT_LABEL_POLICY.min_reward_risk_ratio):g}"),
        ("最大不利跌幅", f"{float(DEFAULT_LABEL_POLICY.max_adverse_return) * 100:g}%（觸及即 REJECT）"),
    )))
    if train_settings is None:
        return
    print(render_section("模型與訓練設定", number=2))
    model_spec = get_model_spec(BREAKOUT_QUALITY_MODEL_ARCHITECTURE)
    experiment = get_breakout_quality_experiment_profile(
        train_settings.experiment_profile
    )
    schedule_parameters = experiment.lr_schedule_parameters()
    augmentation_parameters = experiment.augmentation_parameters()
    if str(model_spec.family) == "moment_frozen_linear":
        architecture_details = (
            f"- Model Family：MOMENT-1-base Frozen Linear\n"
            f"- External Encoder：{model_spec.moment_repository}\n"
            f"- Pinned Revision：{model_spec.moment_revision}\n"
            f"- Runtime：{MOMENT_PACKAGE_NAME} {MOMENT_PACKAGE_VERSION} / "
            f"{MOMENT_TRANSFORMERS_PACKAGE_NAME} {MOMENT_TRANSFORMERS_VERSION}\n"
            f"- Input Resize：{DEFAULT_LABEL_POLICY.feature_window_bars} → {model_spec.moment_input_length} bars\n"
            f"- Patch：{model_spec.moment_patch_length} / stride {model_spec.moment_patch_stride}\n"
            f"- Transformer：{model_spec.moment_transformer_layers} layers / {model_spec.moment_transformer_heads} heads\n"
            f"- Channel Aggregation：{model_spec.moment_channel_aggregation}\n"
            f"- Patch Reduction：{model_spec.moment_patch_reduction}\n"
            f"- Per-channel Representation：{model_spec.moment_embedding_dim}\n"
            f"- Encoder：official pretrained and frozen\n"
            f"- Head：Linear"
        )
    elif str(model_spec.family) == "mantis_v2_frozen_linear":
        architecture_details = (
            f"- Model Family：MantisV2 Frozen Linear\n"
            f"- External Encoder：{model_spec.mantis_repository}\n"
            f"- Pinned Revision：{model_spec.mantis_revision}\n"
            f"- Input Resize：{DEFAULT_LABEL_POLICY.feature_window_bars} → {model_spec.mantis_input_length} bars\n"
            f"- Patches：{model_spec.mantis_num_patches}\n"
            f"- Transformer Output：layer {model_spec.mantis_return_transformer_layer} / {model_spec.mantis_output_token}\n"
            f"- Channel Aggregation：{model_spec.mantis_channel_aggregation}\n"
            f"- Per-channel Representation：{model_spec.mantis_embedding_dim}\n"
            f"- Encoder：official pretrained and frozen\n"
            f"- Head：Linear"
        )
    elif str(model_spec.family) == "ts2vec_frozen_linear":
        pretraining_settings = build_breakout_quality_pretraining_profile_payload(
            BREAKOUT_QUALITY_PRETRAINING_PROFILE
        )
        architecture_details = (
            f"- Model Family：TS2Vec Frozen Linear\n"
            f"- Encoder Depth：{model_spec.ts2vec_depth}\n"
            f"- Hidden Dimensions：{model_spec.ts2vec_hidden_dims}\n"
            f"- Representation Dimensions：{model_spec.ts2vec_output_dims}\n"
            f"- Encoder：Selection-only pretrained and frozen\n"
            f"- Head：Linear\n"
            f"- Pretraining Profile：{BREAKOUT_QUALITY_PRETRAINING_PROFILE}\n"
            f"- Pretraining Settings：{pretraining_settings}\n"
            f"- Pretraining Dataset Stride：{int(BREAKOUT_QUALITY_PRETRAINING_STRIDE)}"
        )
    elif str(model_spec.family) == "patch_transformer":
        architecture_details = (
            f"- Model Family：Small Patch Transformer\n"
            f"- Patch / Stride：{model_spec.patch_transformer_patch_size} / {model_spec.patch_transformer_patch_stride}\n"
            f"- Embedding Dimensions：{model_spec.patch_transformer_embedding_dim}\n"
            f"- Transformer Depth：{model_spec.patch_transformer_depth}\n"
            f"- Attention Heads：{model_spec.patch_transformer_heads}\n"
            f"- MLP Dimensions：{model_spec.patch_transformer_mlp_dim}\n"
            f"- Positional Encoding：{model_spec.patch_transformer_positional_encoding}\n"
            f"- Patch Pooling：{model_spec.patch_transformer_pooling}"
        )
    elif str(model_spec.family) == "inception_time_market_set":
        query_mode = str(model_spec.market_set_query_mode or "global_learned")
        query_label = (
            "Candidate-conditioned Queries"
            if query_mode == "candidate_conditioned"
            else "Global Learned Queries"
        )
        architecture_details = (
            f"- Model Family：InceptionTime + Learned Market Set\n"
            f"- Candidate Depth：{model_spec.inception_depth}\n"
            f"- Candidate Kernel Sizes：{'/'.join(str(value) for value in model_spec.inception_kernel_sizes)}\n"
            f"- Candidate Receptive Field：{model_spec.receptive_field_bars} bars\n"
            f"- Market History：{model_spec.market_set_history_bars} bars\n"
            f"- Market Base Features：{'/'.join(model_spec.market_set_base_features)}\n"
            f"- Shared Stock Embedding：{model_spec.market_set_stock_embedding_dim}\n"
            f"- Market Temporal Normalization：{model_spec.market_set_temporal_normalization} / groups {model_spec.market_set_temporal_normalization_groups}\n"
            f"- Market Query Mode：{query_mode}\n"
            f"- {query_label}：{model_spec.market_set_query_count} / heads {model_spec.market_set_attention_heads}\n"
            f"- Market Embedding：{model_spec.market_set_embedding_dim}\n"
            f"- Runtime Eligibility：research only"
        )
    elif str(model_spec.family) == "inception_time":
        target_line = (
            f"- Configured Receptive Field Target：{int(BREAKOUT_QUALITY_INCEPTION_TARGET_RECEPTIVE_FIELD_BARS)} bars\n"
            if str(model_spec.architecture) == "inception_time_v1"
            else ""
        )
        architecture_details = (
            f"- Model Family：InceptionTime\n"
            f"- Depth：{model_spec.inception_depth}\n"
            f"- Filters：{model_spec.inception_filters}\n"
            f"- Bottleneck Channels：{model_spec.inception_bottleneck_channels}\n"
            f"- Kernel Sizes：{'/'.join(str(value) for value in model_spec.inception_kernel_sizes)}\n"
            f"{target_line}"
            f"- Residual Every：{model_spec.inception_residual_every} modules"
        )
    elif str(model_spec.family) == "modern_tcn":
        architecture_details = (
            f"- Model Family：ModernTCN\n"
            f"- Depth：{model_spec.modern_tcn_depth}\n"
            f"- Channels：{model_spec.modern_tcn_channels}\n"
            f"- Large Kernel：{model_spec.modern_tcn_kernel_size}\n"
            f"- Pointwise Expansion：{model_spec.modern_tcn_expansion_ratio}x\n"
            f"- Normalization：{model_spec.normalization}"
        )
    else:
        branch_inputs = "+".join(model_spec.branch_input_representations) or "level"
        branch_channels = model_spec.branch_channels or (model_spec.channels,) * 3
        branch_dropouts = model_spec.branch_dropouts or (model_spec.dropout,) * 3
        architecture_details = (
            f"- Branch Inputs：{branch_inputs}\n"
            f"- Branch Channels：{'/'.join(str(value) for value in branch_channels)}\n"
            f"- Branch Dropouts：{'/'.join(f'{value:g}' for value in branch_dropouts)}"
        )
    print(
        f"- Model Architecture：{model_spec.architecture}\n"
        f"- Experiment Profile：{experiment.name}\n"
        f"{architecture_details}\n"
        f"- Dataset Event Context：{'使用' if model_spec.use_dataset_context else '不使用'}\n"
        f"- Derived Regime Context：{', '.join(model_spec.derived_context_features) or '-'}\n"
        f"- Receptive Field：約 {model_spec.receptive_field_bars} bars\n"
        f"- Pooling：{'+'.join(model_spec.pooling)}\n"
        f"- Epoch 上限：{int(train_settings.epochs)}\n"
        f"- Batch Size：{int(train_settings.batch_size)}\n"
        f"- Evaluation Batch Size：{int(train_settings.evaluation_batch_size)}\n"
        f"- Evaluation Workers：{int(train_settings.evaluation_workers)}\n"
        f"- Parallel Split Evaluation：{'開啟' if bool(train_settings.parallel_split_evaluation) else '關閉'}\n"
        f"- Train Prefetch Batches：{int(train_settings.train_prefetch_batches)}\n"
        f"- Preload Feature Bank：{'開啟' if bool(train_settings.preload_feature_bank) else '關閉'}\n"
        f"- Optimizer：{experiment.optimizer_name}\n"
        f"- LR Schedule：{experiment.lr_schedule_name}\n"
        f"- LR Schedule Parameters：{schedule_parameters or '-'}\n"
        f"- Augmentation：{experiment.augmentation_name}\n"
        f"- Augmentation Parameters：{augmentation_parameters or '-'}\n"
        f"- Training Sampling：{experiment.training_sampling_mode}\n"
        f"- Training Weight Reduction：{experiment.training_weight_reduction}\n"
        f"- Learning Rate：{float(train_settings.lr):g}\n"
        f"- Weight Decay：{float(train_settings.weight_decay):g}\n"
        f"- Gradient Clip Norm：{float(train_settings.gradient_clip_norm):g}\n"
        f"- Final Refit Mode：{train_settings.final_refit_mode}\n"
        f"- Class Weight Mode：{train_settings.class_weight_mode}\n"
        f"- Time Weight Mode：{train_settings.time_weight_mode}\n"
        f"- Random Seed：{int(train_settings.seed)}\n"
        f"- Threshold：{float(train_settings.fixed_threshold):g}\n"
        f"- Torch Device：{train_settings.device}\n"
        f"- Mixed Precision：{'開啟' if bool(train_settings.mixed_precision) else '關閉'}\n"
        f"- Mixed Precision Dtype：{train_settings.mixed_precision_dtype}\n"
        f"- Deterministic Algorithms：{'開啟' if bool(train_settings.deterministic_algorithms) else '關閉'}\n"
        f"- TF32：{'開啟' if bool(train_settings.allow_tf32) else '關閉'}\n"
        f"- Inner Validation：{'開啟' if bool(train_settings.use_inner_validation) else '關閉'}"
    )
    if bool(train_settings.use_inner_validation):
        print(
            f"- Inner Validation 月數：{int(train_settings.inner_validation_months)}\n"
            f"- Early Stopping Patience：{int(train_settings.early_stopping_patience)}\n"
            f"- Early Stopping Min Delta：{float(train_settings.early_stopping_min_delta):g}"
        )


def _print_artifact_status(
    filter_id: str,
    *,
    model_architecture: str | None = None,
    experiment_profile: str | None = None,
) -> None:
    filter_id = normalize_filter_id(filter_id)
    dataset_paths = _dataset_paths(filter_id)
    model_paths = resolve_existing_filter_artifact_paths(
        PROJECT_ROOT,
        filter_id,
        model_architecture=model_architecture,
        experiment_profile=experiment_profile,
    )
    status_paths = {
        **dataset_paths,
        "model": model_paths.model_path,
        "manifest": model_paths.manifest_path,
        "split": model_paths.split_path,
        "research_scores": resolve_existing_filter_research_score_path(
            PROJECT_ROOT, filter_id,
            model_architecture=model_paths.model_architecture,
            experiment_profile=model_paths.experiment_profile,
        ),
        "research_manifest": resolve_existing_filter_research_manifest_path(
            PROJECT_ROOT, filter_id,
            model_architecture=model_paths.model_architecture,
            experiment_profile=model_paths.experiment_profile,
        ),
        "readable_report": resolve_filter_report_markdown_path(
            PROJECT_ROOT, filter_id,
            model_architecture=model_paths.model_architecture,
            experiment_profile=model_paths.experiment_profile,
        ),
        "report_metrics": resolve_filter_report_json_path(
            PROJECT_ROOT, filter_id,
            model_architecture=model_paths.model_architecture,
            experiment_profile=model_paths.experiment_profile,
        ),
        "runtime_scores": model_paths.score_path,
    }
    print("\n" + render_title(
        f"Breakout Quality 工件狀態｜{filter_id} / "
        f"{model_paths.model_architecture} / {model_paths.experiment_profile}"
    ))
    print(render_status_paths(
        ((name, path, path.is_file()) for name, path in status_paths.items()),
        project_root=PROJECT_ROOT,
        color=console_color_enabled(),
    ))

    summary_path = dataset_paths["summary"]
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            print(f"[警告] 無法讀取 dataset summary: {type(exc).__name__}: {exc}")
        else:
            print(render_section("Dataset 摘要"))
            print(render_key_values((
                ("Dataset", summary.get("dataset")),
                ("Events", summary.get("event_count")),
                ("Date Range", summary.get("event_date_range")),
            )))


def _interactive_workflow(program_name: str, *, workflow_settings=None) -> int:
    if workflow_settings is None:
        filter_id = _policy_filter_id()
        train_args = _policy_train_settings(filter_id)
    else:
        filter_id = normalize_filter_id(workflow_settings.filter_id)
        if workflow_settings.model_architecture != BREAKOUT_QUALITY_MODEL_ARCHITECTURE:
            raise ValueError(
                "binary workflow model architecture必須與active policy一致: "
                f"workflow={workflow_settings.model_architecture}, "
                f"policy={BREAKOUT_QUALITY_MODEL_ARCHITECTURE}"
            )
        train_args = _policy_train_settings(filter_id)
        train_args.experiment_profile = str(workflow_settings.experiment_profile)
        train_args.seed = int(workflow_settings.seed)
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
        [
            "--filter-id",
            filter_id,
            "--experiment-profile",
            BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
            "--scope",
            "research",
        ],
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
        [
            "--filter-id",
            filter_id,
            "--experiment-profile",
            BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
            "--include-oos",
        ],
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
        [
            "--filter-id",
            filter_id,
            "--experiment-profile",
            BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
            "--split",
            split,
        ],
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
        [
            "--filter-id",
            filter_id,
            "--experiment-profile",
            BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
            "--scope",
            "forward_oos",
        ],
        program_name=program_name,
    )


def _interactive_regime_audit(program_name: str) -> int:
    from tools.filters.breakout_quality.regime_audit import (
        DEFAULT_FOCUS_YEAR,
        DEFAULT_MIN_SELECTION_GROUPS,
        DEFAULT_MIN_SUPPORT_SHARE_RATIO,
    )

    filter_id = _policy_filter_id()
    _print_policy_defaults(filter_id)
    focus_year = _prompt_int("歸因年度", DEFAULT_FOCUS_YEAR, minimum=1900)
    print(
        "使用診斷預設："
        f"min_selection_groups={int(DEFAULT_MIN_SELECTION_GROUPS)}、"
        f"min_support_share_ratio={float(DEFAULT_MIN_SUPPORT_SHARE_RATIO):g}；"
        "只作研究歸因，不建立runtime regime gate。"
    )
    return _run_command(
        "regime-audit",
        [
            "--filter-id",
            filter_id,
            "--experiment-profile",
            BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
            "--focus-year",
            str(int(focus_year)),
            "--min-selection-groups",
            str(int(DEFAULT_MIN_SELECTION_GROUPS)),
            "--min-support-share-ratio",
            str(float(DEFAULT_MIN_SUPPORT_SHARE_RATIO)),
        ],
        program_name=program_name,
    )



def _print_workflow_status() -> None:
    settings = get_breakout_quality_workflow_settings()
    color_enabled = console_color_enabled()
    print(
        "\n"
        + render_title(
            paint(
                "Current Breakout Quality Workflow",
                "cyan",
                enabled=color_enabled,
                bold=True,
            )
        )
    )
    base_rows = [
        ("Filter ID", settings.filter_id),
        ("Architecture", settings.model_architecture),
        ("Experiment Profile", settings.experiment_profile),
        ("Training Objective", settings.training_objective),
        ("Training Scope", settings.training_label_scope),
        ("Seed", settings.seed),
        ("Strategy Comparison Mode", settings.strategy_comparison_mode),
        ("Strategy Param Policy", settings.strategy_param_policy),
        ("Strategy Score Source", settings.strategy_score_source),
        ("Strategy Buy Sort", settings.strategy_buy_sort),
        ("Strategy Adapt Trials / Fold", settings.strategy_adapt_trials_per_fold),
        ("Strategy Adapt Fixed Risk", f"{settings.strategy_adapt_fixed_risk:.2%}"),
        (
            "Strategy Adapt Position Cap",
            f"{settings.strategy_adapt_max_position_cap_pct:.2%}",
        ),
    ]

    if settings.is_binary_classification:
        defaults = _train_defaults()
        base_rows.extend((
            ("Model Output", "PASS／REJECT probability"),
            ("Fixed Threshold", f"{float(defaults.fixed_threshold):g}"),
        ))
        print(render_key_values(base_rows))
        _print_artifact_status(
            settings.filter_id,
            model_architecture=settings.model_architecture,
            experiment_profile=settings.experiment_profile,
        )
        return

    if not settings.is_continuous_ranker:
        raise ValueError(
            f"不支援的 workflow training objective: {settings.training_objective!r}"
        )
    base_rows.extend((
        ("Continuous Target", settings.continuous_target_id),
        (
            "PIT Score Period",
            f"{settings.point_in_time_score_start_date} ～ "
            f"{settings.point_in_time_score_end_date or 'Selection end'}",
        ),
        (
            "PIT Fold／Validation",
            f"{settings.point_in_time_fold_months}／"
            f"{settings.point_in_time_inner_validation_months} months",
        ),
    ))
    print(render_key_values(base_rows))
    status_paths = {
        "Dataset summary": _dataset_paths(settings.filter_id)["summary"],
        "Target manifest": resolve_continuous_target_dir(
            PROJECT_ROOT,
            settings.filter_id,
            target_id=str(settings.continuous_target_id),
        ) / TARGET_MANIFEST_FILENAME,
        "Target audit Markdown": resolve_continuous_target_dir(
            PROJECT_ROOT,
            settings.filter_id,
            target_id=str(settings.continuous_target_id),
        ) / TARGET_AUDIT_MARKDOWN_FILENAME,
        "PIT scores": resolve_selection_point_in_time_score_path(
            PROJECT_ROOT, settings.filter_id, settings.model_architecture, settings.experiment_profile
        ),
        "PIT manifest": resolve_selection_point_in_time_manifest_path(
            PROJECT_ROOT, settings.filter_id, settings.model_architecture, settings.experiment_profile
        ),
        "PIT coverage": resolve_selection_point_in_time_coverage_path(
            PROJECT_ROOT, settings.filter_id, settings.model_architecture, settings.experiment_profile
        ),
        "PIT audit JSON": resolve_selection_point_in_time_audit_json_path(
            PROJECT_ROOT, settings.filter_id, settings.model_architecture, settings.experiment_profile
        ),
        "PIT audit Markdown": resolve_selection_point_in_time_audit_markdown_path(
            PROJECT_ROOT, settings.filter_id, settings.model_architecture, settings.experiment_profile
        ),
    }
    grouped_status = (
        ("Dataset", (status_paths["Dataset summary"],)),
        (
            "Continuous Target",
            (status_paths["Target manifest"], status_paths["Target audit Markdown"]),
        ),
        (
            "PIT Scores",
            (status_paths["PIT scores"], status_paths["PIT manifest"], status_paths["PIT coverage"]),
        ),
        (
            "PIT 模型驗證",
            (status_paths["PIT audit JSON"], status_paths["PIT audit Markdown"]),
        ),
    )
    status_rows = []
    for label, paths in grouped_status:
        existing = sum(path.is_file() for path in paths)
        if existing == len(paths):
            status, tone = "完整", "green"
        elif existing == 0:
            status, tone = "缺少", "red"
        else:
            status, tone = "不完整", "yellow"
        status_rows.append(
            (paint(f"[{status}]", tone, enabled=color_enabled, bold=True), label)
        )
    print(
        render_section(
            paint("Workflow 狀態", "cyan", enabled=color_enabled, bold=True)
        )
    )
    print(render_table(("狀態", "項目"), status_rows))


def _interactive_model_research(program_name: str) -> int:
    settings = get_breakout_quality_workflow_settings()
    color_enabled = console_color_enabled()
    if settings.is_binary_classification:
        return _interactive_workflow(
            program_name,
            workflow_settings=settings,
        )
    if not settings.is_continuous_ranker:
        raise ValueError(
            f"不支援的 workflow training objective: {settings.training_objective!r}"
        )

    _print_workflow_status()
    if not _prompt_bool(
        "必要時先建立Dataset與Continuous Target，再建立／更新PIT Scores並執行模型驗證",
        True,
    ):
        return 0
    build_args = [
        "--filter-id", settings.filter_id,
        "--model-architecture", settings.model_architecture,
        "--experiment-profile", settings.experiment_profile,
        "--score-start-date", settings.point_in_time_score_start_date,
        "--fold-months", str(settings.point_in_time_fold_months),
        "--inner-validation-months", str(settings.point_in_time_inner_validation_months),
        "--seed", str(settings.seed),
    ]
    if settings.point_in_time_score_end_date:
        build_args.extend(["--score-end-date", settings.point_in_time_score_end_date])
    build_args.append("--resume" if settings.point_in_time_resume else "--no-resume")

    refresh_mode, refresh_reasons, dataset_step = _dataset_refresh_step(
        settings.filter_id,
        INTERACTIVE_DATASET_PROFILE,
        max_tickers=INTERACTIVE_MAX_TICKERS,
    )
    if dataset_step is not None:
        command, command_args, label = dataset_step
        reason_summary = _compact_dataset_refresh_reason(refresh_reasons)
        print(
            paint("[Dataset]", "cyan", enabled=color_enabled, bold=True)
            + f" {label}｜{reason_summary}"
        )
        with _compact_console_scope():
            code = _run_command(command, command_args, program_name=program_name)
        if code != 0:
            return code

    with _compact_console_scope():
        code = _run_command(
            "prepare-continuous-target",
            [
                "--filter-id",
                settings.filter_id,
                "--target-id",
                str(settings.continuous_target_id),
            ],
            program_name=program_name,
        )
        if code != 0:
            return code

        code = _run_command(
            "build-point-in-time-scores", build_args, program_name=program_name
        )
        if code != 0:
            return code
        return _run_command(
            "audit-point-in-time-scores",
            [
                "--filter-id", settings.filter_id,
                "--model-architecture", settings.model_architecture,
                "--experiment-profile", settings.experiment_profile,
            ],
            program_name=program_name,
        )


def _interactive_strategy_validation(program_name: str) -> int:
    settings = get_breakout_quality_workflow_settings()
    _print_workflow_status()

    if settings.strategy_comparison_mode == "hard-filter":
        if settings.strategy_score_source != "canonical_runtime":
            raise ValueError("hard-filter策略驗證只接受canonical_runtime score source")
        with _compact_console_scope():
            return _run_command(
                "strategy-compare",
                [
                    "--dataset", settings.strategy_dataset,
                    "--comparison-mode", "hard-filter",
                    "--param-policy", settings.strategy_param_policy,
                    "--max-positions", str(settings.strategy_max_positions),
                    "--rotation", settings.strategy_rotation,
                ],
                program_name=program_name,
            )

    if settings.strategy_comparison_mode != "score-ranking":
        raise ValueError(
            f"不支援的 strategy comparison mode: {settings.strategy_comparison_mode!r}"
        )
    if settings.strategy_score_source == "final_selection_model_oos":
        print(
            "[尚未開放] final Selection model OOS Score source尚未接入統一策略入口；"
            "不得回退成canonical runtime score。"
        )
        return 0
    if settings.strategy_score_source not in {
        "selection_point_in_time", "canonical_runtime"
    }:
        raise ValueError(
            f"不支援的 strategy score source: {settings.strategy_score_source!r}"
        )
    while True:
        print("\n=== 策略績效驗證 ===")
        print("[1/Enter] 比較目前策略")
        print("[2] 驗證策略參數適應")
        print("[0] 返回")
        try:
            raw_choice = input("👉 請選擇：").strip().lower()
        except EOFError:
            print("\n輸入已結束。")
            return 0
        choice = "1" if raw_choice == "" else raw_choice
        if choice in {"0", "q", "quit", "exit"}:
            return 0
        if choice == "1":
            with _compact_console_scope():
                return _run_command(
                    "strategy-compare",
                    [
                        "--dataset", settings.strategy_dataset,
                        "--comparison-mode", "score-ranking",
                        "--filter-id", settings.filter_id,
                        "--score-source", settings.strategy_score_source,
                        "--model-architecture", settings.model_architecture,
                        "--experiment-profile", settings.experiment_profile,
                        "--param-policy", settings.strategy_param_policy,
                        "--max-positions", str(settings.strategy_max_positions),
                        "--rotation", settings.strategy_rotation,
                        "--fixed-risk", str(settings.strategy_adapt_fixed_risk),
                        "--max-position-cap-pct",
                        str(settings.strategy_adapt_max_position_cap_pct),
                    ],
                    program_name=program_name,
                )
        if choice == "2":
            with _compact_console_scope():
                return _run_command(
                    "strategy-adapt",
                    [
                        "--dataset", settings.strategy_dataset,
                        "--filter-id", settings.filter_id,
                        "--model-architecture", settings.model_architecture,
                        "--experiment-profile", settings.experiment_profile,
                        "--param-policy", settings.strategy_param_policy,
                        "--trials-per-fold", str(settings.strategy_adapt_trials_per_fold),
                        "--max-positions", str(settings.strategy_max_positions),
                        "--rotation", settings.strategy_rotation,
                        "--fixed-risk", str(settings.strategy_adapt_fixed_risk),
                        "--max-position-cap-pct",
                        str(settings.strategy_adapt_max_position_cap_pct),
                    ],
                    program_name=program_name,
                )
        print("無效選項，請重新輸入。")


def _print_menu() -> None:
    print("\n=== Breakout Quality ===")
    print("[1/Enter] 模型研究與驗證")
    print("[2] 策略績效驗證")
    print("[3] 查看目前設定與工件狀態")
    print("[0] 離開")


def _run_interactive_menu(program_name: str) -> int:
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
                _interactive_model_research(program_name)
            elif choice == "2":
                _interactive_strategy_validation(program_name)
            elif choice == "3":
                _print_workflow_status()
            else:
                print("選項無效，請按 Enter 或輸入 0～3。")
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
