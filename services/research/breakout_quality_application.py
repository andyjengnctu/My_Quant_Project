"""Breakout quality dataset、training、score export、report 與 evaluation 正式入口。"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
import importlib
import json
import math
import os
import subprocess
import sys
import tempfile
from threading import Lock
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.breakout_quality import (
    TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_HMHS_PAIRWISE_RANKING,
    TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    SUPPORTED_BREAKOUT_QUALITY_CLASSIFICATION_EXPERIMENT_PROFILES,
    SUPPORTED_BREAKOUT_QUALITY_TIME_WEIGHT_MODES,
    get_breakout_quality_continuous_ranker_comparison_settings,
    get_breakout_quality_experiment_profile,
    get_continuous_ranker_research_spec,
    BREAKOUT_QUALITY_CONTINUOUS_RANKER_COMPARISON_MENU_LABEL,
    BREAKOUT_QUALITY_CONDITIONAL_MFE_AB_MODEL_GATE_PROFILES,
)
from config.breakout_quality import (
    BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
    BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
    BREAKOUT_QUALITY_INCEPTION_TARGET_RECEPTIVE_FIELD_BARS,
    BREAKOUT_QUALITY_STABILITY_TRAIN_WINDOW_MONTHS,
)
from config.breakout_quality import (
    get_breakout_quality_continuous_ranker_pit_gate_settings,
    get_breakout_quality_model_research_settings,
    get_breakout_quality_rolling_test_mode,
    get_breakout_quality_rolling_test_modes,
    get_breakout_quality_rolling_timing_settings,
    get_breakout_quality_workflow_settings,
)
from core.display_common import FixedProgressBlock, render_elapsed
from core.file_integrity import load_json_object_or_none
from core.training_progress import (
    read_trainer_epoch_progress,
    read_trainer_pit_progress,
    render_training_unit_progress,
)
from core.strategy_comparison import validate_strategy_compare_gpu_train_workers
from core.training_scheduler import pop_next_seed_diverse_unit
from core.research_report_contract import (
    extension_contract,
    format_contract_value,
    section_contract,
    table_contract,
)
from core.report_style import (
    SIGNAL_NEGATIVE,
    SIGNAL_NEUTRAL,
    SIGNAL_POSITIVE,
    SIGNAL_WARNING,
    markdown_tone,
    signal_for_delta,
    styled_signal,
    styled_workflow_status,
)
from core.runtime_utils import (
    is_interactive_console,
    resolve_cli_program_name,
    run_cli_entrypoint,
)
from services.research.strategy_compare_training import (
    run_strategy_compare_training_unit,
    validate_strategy_compare_training_artifacts,
)
from services.research.training_process import (
    terminate_registered_training_processes,
)
from filters.breakout_quality.artifact_dependency_registry import (
    collect_model_upstream_preparation_plan,
)
from filters.breakout_quality.dataset_readiness import collect_dataset_readiness
from filters.breakout_quality.contract import CONTEXT_COLUMNS, DEFAULT_LABEL_POLICY, FEATURE_COLUMNS
from filters.breakout_quality.trade_path_label import (
    TRADE_PATH_LABEL_ID,
    TRADE_PATH_RESEARCH_FILTER_ID,
)
from filters.breakout_quality.continuous_target import (
    TARGET_AUDIT_MARKDOWN_FILENAME,
    TARGET_MANIFEST_FILENAME,
    resolve_continuous_target_dir,
)
from filters.breakout_quality.models.active import get_active_model_spec
from filters.breakout_quality.market_set import market_set_contract_payload
from filters.breakout_quality.dataset_store import (
    DATASET_STORAGE_FORMAT,
    DATASET_STORAGE_SCHEMA_VERSION,
    dataset_artifact_metadata_reasons,
    market_set_artifact_metadata_reasons,
    resolve_dataset_paths,
)
from filters.breakout_quality.paths import (
    normalize_filter_id,
    resolve_existing_filter_artifact_paths,
    resolve_existing_filter_research_manifest_path,
    resolve_existing_filter_research_score_path,
    resolve_filter_artifact_paths,
    resolve_filter_model_output_dir,
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
from filters.breakout_quality.risk_normalized_target import (
    DAILY_RISK_NORMALIZED_NET_OPPORTUNITY_TARGET_ID,
    load_min_roos_risk_schedule,
)
from filters.breakout_quality.ranking_score_store import (
    CONTINUOUS_RANKER_REPORT_FILENAME,
    SCORE_SOURCE_CONTINUOUS_RANKER_OOS,
    SCORE_SOURCE_SELECTION_POINT_IN_TIME,
    derive_point_in_time_model_validation_gate,
    load_continuous_ranker_oos_contract,
    resolve_continuous_ranker_oos_score_path,
)
from services.audit.catalog import get_domain_cli_commands
from services.research.artifact_orchestrator import run_research_artifact_preparation

from core.console_report import (
    COMPACT_CONSOLE_ENV,
    console_color_enabled,
    render_menu_item,
    paint,
    project_relative_display_path,
    render_key_values,
    render_section,
    render_status_paths,
    render_table,
    render_title,
)




COMMAND_MODULES = {
    "build-dataset": "services.breakout_quality.dataset_builder",
    "train": "services.breakout_quality.train",
    "export-scores": "filters.breakout_quality.export_scores",
    "report": "services.breakout_quality.report",
    "evaluate": "services.breakout_quality.evaluate",
    "prepare-continuous-target": "services.breakout_quality.continuous_target_preparation",
    "train-continuous-ranker": "services.breakout_quality.ranker_cli",
    "compare-daily-targets": "services.breakout_quality.daily_target_comparison",
    "compare-continuous-rankers": "services.breakout_quality.continuous_ranker_comparison",
    "build-point-in-time-scores": "services.breakout_quality.point_in_time_scores",
    "timing-rolling-training": "services.breakout_quality.rolling_timing",
    "build-binary-point-in-time-scores": (
        "services.breakout_quality.binary_point_in_time_scores"
    ),
    "build-trade-path-labels": (
        "services.breakout_quality.trade_path_label_builder"
    ),
}

_AUDIT_CLI_COMMANDS = get_domain_cli_commands("breakout_quality")
COMMAND_MODULES.update(
    {command: entry.module for command, entry in _AUDIT_CLI_COMMANDS.items()}
)

INTERACTIVE_DATASET_PROFILE = "full"
INTERACTIVE_MAX_TICKERS = 0
INTERACTIVE_EVALUATE_OOS = True



COMMAND_DESCRIPTIONS = {
    "menu": "開啟互動式操作選單",
    "workflow": "依序執行 dataset、active model train、research score export與報表",
    "build-dataset": "建立 breakout quality event dataset",
    "train": "訓練模型；可選擇 inner validation 選 epoch 後完整 Selection 重訓",
    "export-scores": "匯出 research 或 forward-OOS score table",
    "report": "產生表格化終端報表、Markdown 報表與完整 metrics JSON",
    "evaluate": "輸出 train、validation、selection 或 OOS 的詳細 JSON",
    "prepare-continuous-target": "依目前workflow檢查並建立continuous target工件",
    "train-continuous-ranker": "執行目前設定的continuous ranker模型研究；正式選單亦可使用",
    "compare-daily-targets": "比較目前daily target與config指定reference target；只讀、不訓練",
    "compare-continuous-rankers": (
        "只讀config設定的continuous-ranker frozen scores，做paired／random baseline／Dynamic-K品質比較"
    ),
    "build-point-in-time-scores": "建立泛用Selection point-in-time continuous-ranker scores",
    "timing-rolling-training": "以隔離單fold建立Rolling訓練改善前baseline並比較目前實作的wall-clock與exact-result hashes",
    "build-binary-point-in-time-scores": (
        "建立Binary DL filter歷史 point-in-time scores；research-only、CLI-only"
    ),
    "build-trade-path-labels": (
        "建立A2 realized trade-path Label Dataset；research workflow"
    ),
}
COMMAND_DESCRIPTIONS.update(
    {command: entry.description for command, entry in _AUDIT_CLI_COMMANDS.items()}
)


def _print_help(program_name: str) -> None:
    print(f"用法: python {program_name} [menu|workflow|<command>] [options]")
    print("說明: Breakout quality Dataset、Label、模型訓練與模型評估的正式模型服務；正式入口為 apps/research.py。")
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


def _cli_option_value(args: list[str], flag: str, default=None):
    try:
        index = args.index(flag)
    except ValueError:
        return default
    if index + 1 >= len(args):
        return default
    return args[index + 1]


def _simple_report_context(command: str, args: list[str]) -> tuple[str, str, str]:
    if command == "timing-rolling-training":
        timing = get_breakout_quality_rolling_timing_settings()
        return (
            normalize_filter_id(BREAKOUT_QUALITY_DEFAULT_FILTER_ID),
            str(BREAKOUT_QUALITY_MODEL_ARCHITECTURE),
            str(timing.experiment_profile),
        )
    settings = (
        get_breakout_quality_model_research_settings()
        if command in {"train-continuous-ranker", "compare-daily-targets"}
        else get_breakout_quality_workflow_settings()
    )
    filter_id = normalize_filter_id(
        _cli_option_value(args, "--filter-id", settings.filter_id)
    )
    architecture = str(
        _cli_option_value(args, "--model-architecture", settings.model_architecture)
    )
    profile = str(
        _cli_option_value(args, "--experiment-profile", settings.experiment_profile)
    )
    return filter_id, architecture, profile


def _fmt_simple_metric(value, *, digits: int = 4, percent: bool = False) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if percent:
        return f"{number * 100.0:.2f}%"
    return f"{number:.{digits}f}"


def _continuous_ranker_simple_report_payload(
    *,
    filter_id: str,
    architecture: str,
    profile: str,
) -> dict:
    output_dir = resolve_filter_model_output_dir(
        PROJECT_ROOT, filter_id, architecture, profile
    )
    return load_json_object_or_none(output_dir / CONTINUOUS_RANKER_REPORT_FILENAME) or {}


def _print_existing_continuous_ranker_report(settings) -> bool:
    """Print the current persisted model summary without retraining."""

    payload = _continuous_ranker_simple_report_payload(
        filter_id=settings.filter_id,
        architecture=settings.model_architecture,
        profile=settings.experiment_profile,
    )
    if not payload:
        return False
    console = _render_continuous_ranker_simple_console(payload)
    if not console:
        return False
    print("\n" + render_section("目前模型報表摘要"))
    print(console)
    detail_report = (
        resolve_filter_model_output_dir(
            PROJECT_ROOT,
            settings.filter_id,
            settings.model_architecture,
            settings.experiment_profile,
        )
        / "continuous_ranker_report.md"
    )
    print(
        render_status_paths(
            (("詳細模型報表", detail_report, detail_report.is_file()),),
            project_root=PROJECT_ROOT,
        )
    )
    return True


def _render_mr13s_truth_geometry_console(payload: dict) -> str:
    def fmt(value, *, digits: int = 4) -> str:
        return "-" if value is None else f"{float(value):.{digits}f}"

    def support(cell: dict) -> str:
        return (
            f"N={int(cell.get('n', 0) or 0):,} / "
            f"{fmt(cell.get('population_pct'), digits=2)}% / "
            f"{fmt(cell.get('independence_enrichment'), digits=2)}×"
        )

    lines = [render_section("MR-13S Actual MFE×Safety Truth Geometry Control")]
    summary_rows = []
    for label, key in (
        ("Daily universal OOS", "daily_universal_oos"),
        ("Breakout candidate OOS", "breakout_candidate_oos"),
    ):
        scope = dict(payload.get(key) or {})
        actual = dict(scope.get("actual") or {})
        summary_rows.append((
            label,
            f"{int(scope.get('population_n', 0) or 0):,}",
            fmt(actual.get("safety_to_mfe_mean_daily_spearman")),
            fmt(scope.get("predicted_safety_to_raw_mfe_mean_daily_spearman")),
            support(dict(actual.get("s5_m5") or {})),
            support(dict(actual.get("s4plus_m4plus") or {})),
        ))
    lines.append(
        render_table(
            ("Scope", "N", "Actual S↔MFE rho", "Pred S↔MFE rho", "S5×M5 N/Pop/×Exp", "S4+×M4+ N/Pop/×Exp"),
            summary_rows,
            alignments=("left", "right", "right", "right", "right", "right"),
        )
    )
    for label, key in (
        ("Daily universal OOS actual 5×5", "daily_universal_oos"),
        ("Breakout candidate OOS actual 5×5", "breakout_candidate_oos"),
    ):
        scope = dict(payload.get(key) or {})
        actual = dict(scope.get("actual") or {})
        rows = []
        for s_idx, row in enumerate(list(actual.get("actual_joint_geometry") or []), start=1):
            cells = []
            for cell in row:
                cell = dict(cell or {})
                cells.append(
                    f"{int(cell.get('n', 0) or 0):,} / "
                    f"{fmt(cell.get('population_pct'), digits=2)}% / "
                    f"{fmt(cell.get('independence_enrichment'), digits=2)}×"
                )
            rows.append((f"S{s_idx}", *cells))
        lines.extend([
            label,
            "cell = N / population% / independence enrichment×",
            render_table(
                ("Actual Safety \\ Pure-MFE", "M1", "M2", "M3", "M4", "M5"),
                rows,
                alignments=("left", "right", "right", "right", "right", "right"),
            ),
        ])
    lines.append(
        "Breakout percentile口徑：沿用Daily universal同日percentile，只filter candidate membership，不在subset內重新排名。"
    )
    return "\n".join(lines)


def _run_mr13s_truth_geometry_control(settings) -> bool:
    from services.breakout_quality.train_daily_ranker import (
        build_mr13s_truth_geometry_control,
    )

    payload, json_path, markdown_path = build_mr13s_truth_geometry_control(
        filter_id=settings.filter_id,
        model_architecture=settings.model_architecture,
        experiment_profile=settings.experiment_profile,
        project_root=PROJECT_ROOT,
    )
    print("\n" + _render_mr13s_truth_geometry_console(payload))
    print(
        render_status_paths(
            (
                ("Truth Geometry JSON", json_path, json_path.is_file()),
                ("Truth Geometry Markdown", markdown_path, markdown_path.is_file()),
            ),
            project_root=PROJECT_ROOT,
        )
    )
    return True


def _model_sop_section_title(section_id: str, *, suffix: str = "") -> str:
    section = section_contract("model.standard_sop", section_id)
    base = f"標準模型 SOP｜{section.number}. {section.title}"
    return base + (f"｜{suffix}" if suffix else "")


def _model_extension_title(payload: dict, extension_id: str) -> str:
    model_id = str(payload.get("model_research_id") or "MODEL")
    extension = extension_contract(extension_id)
    return f"Model-specific Extension｜{model_id}｜{extension.title}"


def _model_sop_view(payload: dict) -> dict:
    """Build one target-neutral view consumed by console and Markdown renderers."""

    metrics = dict(payload.get("split_metrics") or {})
    if not metrics:
        return {}
    training = dict(payload.get("training") or {})
    direct_hmhs_only = (
        training.get("objective") == TRAINING_OBJECTIVE_DAILY_HMHS_PAIRWISE_RANKING
    )
    daily_universal = bool(metrics.get("breakout_candidate_oos"))
    split_names = (
        ["validation", "oos", "breakout_candidate_oos"]
        if daily_universal
        else ["validation", "selection", "oos"]
    )

    learnability_rows = []
    for name in split_names:
        row = dict(metrics.get(name) or {})
        learnability_rows.append({
            "split": name,
            "group_count": int(row.get("group_count", 0) or 0),
            "mean_daily_spearman": row.get("mean_daily_spearman"),
            "global_spearman_vs_raw_target": row.get("global_spearman_vs_raw_target"),
            "pairwise_concordance": row.get("pairwise_concordance"),
            "top_score_decile_raw_target_mean": row.get("top_score_decile_raw_target_mean"),
            "bottom_score_decile_raw_target_mean": row.get("bottom_score_decile_raw_target_mean"),
        })

    generalization_rows = []
    val = dict(metrics.get("validation") or {})
    oos = dict(metrics.get("oos") or {})
    breakout = dict(metrics.get("breakout_candidate_oos") or {})

    def top_bottom(row: dict) -> float | None:
        top = row.get("top_score_decile_raw_target_mean")
        bottom = row.get("bottom_score_decile_raw_target_mean")
        return None if top is None or bottom is None else float(top) - float(bottom)

    def delta(left, right):
        return None if left is None or right is None else float(right) - float(left)

    if val and oos:
        generalization_rows.append({
            "comparison": "Primary score",
            "delta_daily_rho": delta(val.get("mean_daily_spearman"), oos.get("mean_daily_spearman")),
            "delta_pair": None if delta(val.get("pairwise_concordance"), oos.get("pairwise_concordance")) is None else delta(val.get("pairwise_concordance"), oos.get("pairwise_concordance")) * 100.0,
            "delta_top_bottom": delta(top_bottom(val), top_bottom(oos)),
        })
        if daily_universal and breakout:
            generalization_rows.append({
                "comparison": "OOS → Breakout slice",
                "delta_daily_rho": delta(oos.get("mean_daily_spearman"), breakout.get("mean_daily_spearman")),
                "delta_pair": None if delta(oos.get("pairwise_concordance"), breakout.get("pairwise_concordance")) is None else delta(oos.get("pairwise_concordance"), breakout.get("pairwise_concordance")) * 100.0,
                "delta_top_bottom": delta(top_bottom(oos), top_bottom(breakout)),
            })

    multi_head_rows = []
    for evaluation, heads in (
        (dict(payload.get("conditional_mfe_safety_evaluation") or {}), (("Primary MFE", "primary_mfe"), ("Conditional Safety", "conditional_safety"))),
        (dict(payload.get("reverse_conditional_mfe_evaluation") or {}), (("Raw Safety", "raw_safety"), ("Conditional MFE", "conditional_mfe"))),
    ):
        if not evaluation:
            continue
        for scope_label, scope_key in (("Validation", "validation"), ("Forward OOS", "oos"), ("Breakout slice", "breakout_candidate_oos")):
            scope = dict(evaluation.get(scope_key) or {})
            for head_label, head_key in heads:
                row = dict(scope.get(head_key) or {})
                if row:
                    multi_head_rows.append({
                        "split": scope_label,
                        "head": head_label,
                        "mean_daily_spearman": row.get("mean_daily_spearman"),
                        "global_spearman_vs_raw_target": row.get("global_spearman_vs_raw_target"),
                        "pairwise_concordance": row.get("pairwise_concordance"),
                    })

    raw_eval = dict(
        payload.get("safety_raw_mfe_hmhs_evaluation")
        or payload.get("safety_raw_mfe_evaluation")
        or {}
    )
    if raw_eval:
        for scope_label, scope_key in (("Validation", "validation"), ("Forward OOS", "oos"), ("Breakout slice", "breakout_candidate_oos")):
            scope = dict(raw_eval.get(scope_key) or {})
            for head_label, head_key in (("Raw Safety", "raw_safety"), ("Raw MFE", "raw_mfe")):
                row = dict(scope.get(head_key) or {})
                if row:
                    multi_head_rows.append({
                        "split": scope_label,
                        "head": head_label,
                        "mean_daily_spearman": row.get("mean_daily_spearman"),
                        "global_spearman_vs_raw_target": row.get("global_spearman_vs_raw_target"),
                        "pairwise_concordance": row.get("pairwise_concordance"),
                    })

    geometry_scopes = []
    for scope_label, scope_key in (("Daily universal OOS", "oos"), ("Breakout candidate OOS", "breakout_candidate_oos")):
        scope = dict(raw_eval.get(scope_key) or {})
        gate = dict(scope.get("model_gate") or {})
        if gate:
            geometry_scopes.append((scope_label, gate))

    sample = dict((metrics.get("oos") or {}).get("top_k_quality") or {})
    ranking = None
    if sample:
        ranking = {
            "top_k": int(sample.get("top_k", 0) or 0),
            "boundary_width": int(sample.get("boundary_width", 0) or 0),
            "competition_scope": "只看同日樣本數>K" if daily_universal else "只看候選數>K",
            "rows": [],
        }
        for name in split_names:
            quality = dict((metrics.get(name) or {}).get("top_k_quality") or {})
            ranking["rows"].append({
                "split": name,
                "ndcg_at_k": quality.get("ndcg_at_k"),
                "top_k_raw_target_mean": quality.get("top_k_raw_target_mean"),
                "top_k_raw_target_lift": quality.get("top_k_raw_target_lift"),
                "oracle_top_k_overlap": quality.get("oracle_top_k_overlap"),
                "boundary_concordance": quality.get("boundary_concordance"),
                "boundary_raw_target_gap": quality.get("boundary_raw_target_gap"),
                "competition_date_count": int(quality.get("competition_date_count", quality.get("top_k_date_count", 0)) or 0),
            })

    evidence = [
        ("Learnability", "AVAILABLE" if metrics.get("oos") else "N/A"),
        ("Generalization", "AVAILABLE" if metrics.get("validation") and metrics.get("oos") else "N/A"),
        ("Truth / Prediction Geometry", "AVAILABLE" if geometry_scopes else "N/A"),
        ("Breakout application slice", "AVAILABLE" if metrics.get("breakout_candidate_oos") else "N/A"),
        ("Ranking / Boundary", "AVAILABLE" if sample else "N/A"),
    ]

    extensions = []
    if direct_hmhs_only:
        ext_rows = []
        for label, key in (("Validation", "validation"), ("Forward OOS", "oos"), ("Breakout slice", "breakout_candidate_oos")):
            row = dict(metrics.get(key) or {})
            if not row:
                continue
            top10 = dict(row.get("top_10pct") or {})
            top20 = dict(row.get("top_20pct") or {})
            ext_rows.append({
                "split": label,
                "population_hmhs_pct": row.get("population_hmhs_pct"),
                "pairwise_concordance": row.get("pairwise_concordance"),
                "global_average_precision": row.get("global_average_precision"),
                "mean_daily_average_precision": row.get("mean_daily_average_precision"),
                "top10": f"{_fmt_simple_metric(None if top10.get('hmhs_pct') is None else float(top10['hmhs_pct']) / 100.0, percent=True)} / {_fmt_simple_metric(top10.get('hmhs_enrichment'))}×",
                "top20": f"{_fmt_simple_metric(None if top20.get('hmhs_pct') is None else float(top20['hmhs_pct']) / 100.0, percent=True)} / {_fmt_simple_metric(top20.get('hmhs_enrichment'))}×",
            })
        def enrich(row: dict) -> float | None:
            return dict(row.get("top_10pct") or {}).get("hmhs_enrichment")
        ext_gen = []
        if val and oos:
            ext_gen.append({
                "comparison": "Validation → Forward OOS",
                "delta_pair": None if delta(val.get("pairwise_concordance"), oos.get("pairwise_concordance")) is None else delta(val.get("pairwise_concordance"), oos.get("pairwise_concordance")) * 100.0,
                "delta_pr_auc": delta(val.get("global_average_precision"), oos.get("global_average_precision")),
                "delta_top10_enrichment": delta(enrich(val), enrich(oos)),
            })
            if breakout:
                ext_gen.append({
                    "comparison": "Forward OOS → Breakout slice",
                    "delta_pair": None if delta(oos.get("pairwise_concordance"), breakout.get("pairwise_concordance")) is None else delta(oos.get("pairwise_concordance"), breakout.get("pairwise_concordance")) * 100.0,
                    "delta_pr_auc": delta(oos.get("global_average_precision"), breakout.get("global_average_precision")),
                    "delta_top10_enrichment": delta(enrich(oos), enrich(breakout)),
                })
        extensions.append({"id": "direct_hmhs_h_only", "rows": ext_rows, "generalization": ext_gen})

    joint_rows = []
    for scope_label, scope_key in (("Validation", "validation"), ("Forward OOS", "oos"), ("Breakout slice", "breakout_candidate_oos")):
        scope = dict(raw_eval.get(scope_key) or {})
        joint = dict(scope.get("joint_hmhs") or {})
        product = dict(scope.get("joint_product_control") or {})
        if not joint:
            continue
        top10 = dict(joint.get("top_10pct") or {})
        product10 = dict(product.get("top_10pct") or {})
        joint_rows.append({
            "split": scope_label,
            "population_hmhs_pct": joint.get("population_hmhs_pct"),
            "direct_pair": joint.get("pairwise_concordance"),
            "product_pair": product.get("pairwise_concordance"),
            "direct_pr_auc": joint.get("global_average_precision"),
            "product_pr_auc": product.get("global_average_precision"),
            "direct_top10_pct": top10.get("hmhs_pct"),
            "direct_top10_enrichment": top10.get("hmhs_enrichment"),
            "product_top10_pct": product10.get("hmhs_pct"),
            "product_top10_enrichment": product10.get("hmhs_enrichment"),
        })
    if joint_rows:
        extensions.append({"id": "direct_hmhs_joint_retrieval", "rows": joint_rows})

    return {
        "split_names": split_names,
        "learnability": learnability_rows,
        "generalization": generalization_rows,
        "multi_head": multi_head_rows,
        "geometry": geometry_scopes,
        "ranking": ranking,
        "evidence": evidence,
        "extensions": extensions,
    }


def _render_model_contract_table(table, rows, *, target: str, scope_styler=None, delta_style: bool = False) -> str:
    rendered = []
    for row in rows:
        cells = []
        for column in table.columns:
            value = row.get(column.key)
            text = format_contract_value(column, value)
            if column.key == "split" and scope_styler is not None:
                text = scope_styler(text)
            if delta_style and column.preference in {"higher", "lower"} and value is not None:
                signal = signal_for_delta(value, preference=column.preference)
                text = styled_signal(
                    text, signal, target=target,
                    enabled=console_color_enabled() if target == "console" else None,
                    bold=True,
                )
            cells.append(text)
        rendered.append(tuple(cells))
    if target == "console":
        return render_table(table.headers, rendered, alignments=table.alignments)
    header = "| " + " | ".join(table.headers) + " |"
    separator = "|" + "|".join("---" if align == "left" else "---:" for align in table.alignments) + "|"
    body = ["| " + " | ".join(row) + " |" for row in rendered]
    return "\n".join([header, separator, *body])


def _render_continuous_ranker_simple_console(payload: dict) -> str:
    view = _model_sop_view(payload)
    if not view:
        return ""
    color = console_color_enabled()

    def section(section_id: str, *, suffix: str = "") -> str:
        return render_section(paint(_model_sop_section_title(section_id, suffix=suffix), "cyan", enabled=color, bold=True))

    def extension(title: str) -> str:
        return render_section(paint(title, "cyan", enabled=color, bold=True))

    def scope_text(value: str) -> str:
        tone = "gray" if str(value).lower() == "validation" else "cyan"
        return paint(str(value), tone, enabled=color, bold=str(value).lower() != "validation")

    lines = []
    lines.extend([
        section("learnability"),
        _render_model_contract_table(
            table_contract("model.standard_sop", "learnability", "learnability"),
            view["learnability"], target="console", scope_styler=scope_text,
        ),
    ])
    if view["generalization"]:
        lines.extend([
            section("generalization"),
            _render_model_contract_table(
                table_contract("model.standard_sop", "generalization", "generalization"),
                view["generalization"], target="console", delta_style=True,
            ),
        ])
    if view["multi_head"]:
        lines.extend([
            section("multi_head"),
            _render_model_contract_table(
                table_contract("model.standard_sop", "multi_head", "multi_head"),
                view["multi_head"], target="console", scope_styler=scope_text,
            ),
        ])

    def truth_cell(cell: dict) -> str:
        if not cell:
            return "0 / - / -"
        pct = cell.get("population_pct")
        enrich = cell.get("independence_enrichment")
        return f"{int(cell.get('n', 0) or 0):,} / {'-' if pct is None else f'{float(pct):.2f}%'} / {'-' if enrich is None else f'{float(enrich):.2f}×'}"

    for scope_label, gate in view["geometry"]:
        actual = dict(gate.get("actual_truth_geometry") or {})
        upper = dict(gate.get("upper_right_s5_m5") or {})
        pred_n = int(upper.get("n", 0) or 0)
        actual_s5_n = int(dict(actual.get("s5_m5") or {}).get("n", 0) or 0)
        pred_n_text = styled_signal(
            f"{pred_n:,}",
            SIGNAL_NEGATIVE if actual_s5_n > 0 and pred_n == 0 else SIGNAL_NEUTRAL,
            target="console", enabled=color, bold=True,
        )
        lines.extend([
            section("truth_prediction_geometry", suffix=scope_label),
            render_key_values((
                ("Actual Safety↔MFE Daily rho", _fmt_simple_metric(actual.get("safety_to_mfe_mean_daily_spearman"))),
                ("Pred Safety↔Raw-MFE Daily rho", _fmt_simple_metric(gate.get("predicted_safety_to_raw_mfe_mean_daily_spearman"))),
                ("Actual S5×M5", truth_cell(dict(actual.get("s5_m5") or {}))),
                ("Actual S4+×M4+", truth_cell(dict(actual.get("s4plus_m4plus") or {}))),
                ("Pred S5×M5 N", pred_n_text),
                ("Joint product→actual HM/HS Daily rho", _fmt_simple_metric(gate.get("joint_product_to_actual_hmhs_mean_daily_spearman"))),
            )),
        ])
        actual_rows = []
        for s_idx, row in enumerate(list(actual.get("actual_joint_geometry") or []), start=1):
            actual_rows.append((f"S{s_idx}", *(truth_cell(dict(cell or {})) for cell in row)))
        if actual_rows:
            t = table_contract("model.standard_sop", "truth_prediction_geometry", "truth_5x5")
            lines.append(render_table(t.headers, actual_rows, alignments=t.alignments))
        predicted_rows = []
        for s_idx, row in enumerate(list(gate.get("predicted_joint_geometry") or []), start=1):
            cells = []
            for cell in row:
                cell = dict(cell or {})
                pct = cell.get("actual_hmhs_pct")
                cells.append(f"{int(cell.get('n', 0) or 0):,} / {'-' if pct is None else f'{float(pct):.2f}%'}")
            predicted_rows.append((f"S{s_idx}", *cells))
        if predicted_rows:
            t = table_contract("model.standard_sop", "truth_prediction_geometry", "predicted_5x5")
            lines.append(render_table(t.headers, predicted_rows, alignments=t.alignments))
        cohort_rows = []
        for cohort in list(gate.get("safety_cohorts") or []):
            cohort = dict(cohort or {})
            cohort_rows.append({
                "safety": f"S{int(cohort.get('predicted_safety_quintile', 0) or 0)}",
                "n": int(cohort.get("n", 0) or 0),
                "raw_mfe_to_actual_mfe_mean_daily_spearman": cohort.get("raw_mfe_to_actual_mfe_mean_daily_spearman"),
                "high_mfe_pct": cohort.get("high_mfe_pct"),
                "hmhs_pct": cohort.get("hmhs_pct"),
            })
        if cohort_rows:
            lines.append(_render_model_contract_table(
                table_contract("model.standard_sop", "truth_prediction_geometry", "safety_cohorts"),
                cohort_rows, target="console",
            ))

    ranking = view.get("ranking")
    if ranking:
        suffix = f"K={ranking['top_k']}，boundary={ranking['boundary_width']}；{ranking['competition_scope']}"
        lines.extend([
            section("ranking_boundary", suffix=suffix),
            _render_model_contract_table(
                table_contract("model.standard_sop", "ranking_boundary", "ranking_boundary"),
                ranking["rows"], target="console", scope_styler=scope_text,
            ),
        ])

    evidence_rows = []
    for label, status in view["evidence"]:
        normalized = str(status).upper()
        signal = {"AVAILABLE": SIGNAL_POSITIVE, "READY": SIGNAL_POSITIVE, "PARTIAL": SIGNAL_WARNING, "BLOCKED": SIGNAL_NEGATIVE, "MISSING": SIGNAL_NEGATIVE}.get(normalized, SIGNAL_NEUTRAL)
        evidence_rows.append({"evidence": label, "status": styled_signal(status, signal, target="console", enabled=color, bold=True)})
    lines.extend([
        section("evidence_coverage"),
        _render_model_contract_table(
            table_contract("model.standard_sop", "evidence_coverage", "evidence_coverage"),
            evidence_rows, target="console",
        ),
    ])

    for ext in view["extensions"]:
        ext_id = str(ext["id"])
        lines.append(extension(_model_extension_title(payload, ext_id)))
        if ext_id == "direct_hmhs_h_only":
            if ext.get("generalization"):
                lines.append("Generalization")
                lines.append(_render_model_contract_table(
                    extension_contract(ext_id).tables[1], ext["generalization"], target="console", delta_style=True,
                ))
            lines.append("Learnability")
            lines.append(_render_model_contract_table(
                extension_contract(ext_id).tables[0], ext["rows"], target="console", scope_styler=scope_text,
            ))
        elif ext_id == "direct_hmhs_joint_retrieval":
            lines.append(_render_model_contract_table(
                extension_contract(ext_id).tables[0], ext["rows"], target="console", scope_styler=scope_text,
            ))
    return "\n".join(line for line in lines if line)


def _render_continuous_ranker_simple_markdown(payload: dict) -> list[str]:
    view = _model_sop_view(payload)
    if not view:
        return []

    def section(section_id: str, *, suffix: str = "") -> str:
        return f"## {markdown_tone(_model_sop_section_title(section_id, suffix=suffix), 'blue', bold=True)}"

    def extension(title: str) -> str:
        return f"## {markdown_tone(title, 'blue', bold=True)}"

    def scope_text(value: str) -> str:
        tone = "gray" if str(value).lower() == "validation" else "blue"
        return markdown_tone(value, tone, bold=str(value).lower() != "validation")

    lines = ["", section("learnability"), "", _render_model_contract_table(
        table_contract("model.standard_sop", "learnability", "learnability"), view["learnability"], target="markdown", scope_styler=scope_text,
    )]
    if view["generalization"]:
        lines.extend(["", section("generalization"), "", _render_model_contract_table(
            table_contract("model.standard_sop", "generalization", "generalization"), view["generalization"], target="markdown", delta_style=True,
        )])
    if view["multi_head"]:
        lines.extend(["", section("multi_head"), "", _render_model_contract_table(
            table_contract("model.standard_sop", "multi_head", "multi_head"), view["multi_head"], target="markdown", scope_styler=scope_text,
        )])

    def truth_cell(cell: dict) -> str:
        if not cell:
            return "0 / - / -"
        pct = cell.get("population_pct")
        enrich = cell.get("independence_enrichment")
        return f"{int(cell.get('n', 0) or 0):,} / {'-' if pct is None else f'{float(pct):.2f}%'} / {'-' if enrich is None else f'{float(enrich):.2f}×'}"

    for scope_label, gate in view["geometry"]:
        actual = dict(gate.get("actual_truth_geometry") or {})
        upper = dict(gate.get("upper_right_s5_m5") or {})
        pred_n = int(upper.get("n", 0) or 0)
        actual_s5_n = int(dict(actual.get("s5_m5") or {}).get("n", 0) or 0)
        pred_n_text = styled_signal(
            f"{pred_n:,}", SIGNAL_NEGATIVE if actual_s5_n > 0 and pred_n == 0 else SIGNAL_NEUTRAL,
            target="markdown", bold=True,
        )
        lines.extend([
            "", section("truth_prediction_geometry", suffix=scope_label), "",
            f"- **Actual Safety↔MFE Daily rho**: `{_fmt_simple_metric(actual.get('safety_to_mfe_mean_daily_spearman'))}`",
            f"- **Pred Safety↔Raw-MFE Daily rho**: `{_fmt_simple_metric(gate.get('predicted_safety_to_raw_mfe_mean_daily_spearman'))}`",
            f"- **Actual S5×M5**: `{truth_cell(dict(actual.get('s5_m5') or {}))}`",
            f"- **Actual S4+×M4+**: `{truth_cell(dict(actual.get('s4plus_m4plus') or {}))}`",
            f"- **Pred S5×M5 N**: {pred_n_text}",
            f"- **Joint product→actual HM/HS Daily rho**: `{_fmt_simple_metric(gate.get('joint_product_to_actual_hmhs_mean_daily_spearman'))}`",
        ])
        actual_rows = []
        for s_idx, row in enumerate(list(actual.get("actual_joint_geometry") or []), start=1):
            actual_rows.append({"safety": f"S{s_idx}", **{f"m{i}": truth_cell(dict(cell or {})) for i, cell in enumerate(row, start=1)}})
        if actual_rows:
            lines.extend(["", "### Actual 5×5", "", _render_model_contract_table(
                table_contract("model.standard_sop", "truth_prediction_geometry", "truth_5x5"), actual_rows, target="markdown",
            )])
        predicted_rows = []
        for s_idx, row in enumerate(list(gate.get("predicted_joint_geometry") or []), start=1):
            cells = {}
            for i, cell in enumerate(row, start=1):
                cell = dict(cell or {})
                pct = cell.get("actual_hmhs_pct")
                cells[f"m{i}"] = f"{int(cell.get('n', 0) or 0):,} / {'-' if pct is None else f'{float(pct):.2f}%'}"
            predicted_rows.append({"safety": f"S{s_idx}", **cells})
        if predicted_rows:
            lines.extend(["", "### Predicted 5×5", "", _render_model_contract_table(
                table_contract("model.standard_sop", "truth_prediction_geometry", "predicted_5x5"), predicted_rows, target="markdown",
            )])
        cohort_rows = []
        for cohort in list(gate.get("safety_cohorts") or []):
            cohort = dict(cohort or {})
            cohort_rows.append({
                "safety": f"S{int(cohort.get('predicted_safety_quintile', 0) or 0)}",
                "n": int(cohort.get("n", 0) or 0),
                "raw_mfe_to_actual_mfe_mean_daily_spearman": cohort.get("raw_mfe_to_actual_mfe_mean_daily_spearman"),
                "high_mfe_pct": cohort.get("high_mfe_pct"), "hmhs_pct": cohort.get("hmhs_pct"),
            })
        if cohort_rows:
            lines.extend(["", _render_model_contract_table(
                table_contract("model.standard_sop", "truth_prediction_geometry", "safety_cohorts"), cohort_rows, target="markdown",
            )])

    ranking = view.get("ranking")
    if ranking:
        suffix = f"K={ranking['top_k']}，boundary={ranking['boundary_width']}；{ranking['competition_scope']}"
        lines.extend(["", section("ranking_boundary", suffix=suffix), "", _render_model_contract_table(
            table_contract("model.standard_sop", "ranking_boundary", "ranking_boundary"), ranking["rows"], target="markdown", scope_styler=scope_text,
        )])

    evidence_rows = []
    for label, status in view["evidence"]:
        normalized = str(status).upper()
        signal = {"AVAILABLE": SIGNAL_POSITIVE, "READY": SIGNAL_POSITIVE, "PARTIAL": SIGNAL_WARNING, "BLOCKED": SIGNAL_NEGATIVE, "MISSING": SIGNAL_NEGATIVE}.get(normalized, SIGNAL_NEUTRAL)
        evidence_rows.append({"evidence": label, "status": styled_signal(status, signal, target="markdown", bold=True)})
    lines.extend(["", section("evidence_coverage"), "", _render_model_contract_table(
        table_contract("model.standard_sop", "evidence_coverage", "evidence_coverage"), evidence_rows, target="markdown",
    )])

    for ext in view["extensions"]:
        ext_id = str(ext["id"])
        lines.extend(["", extension(_model_extension_title(payload, ext_id)), ""])
        if ext_id == "direct_hmhs_h_only":
            if ext.get("generalization"):
                lines.extend(["### Generalization", "", _render_model_contract_table(
                    extension_contract(ext_id).tables[1], ext["generalization"], target="markdown", delta_style=True,
                ), ""])
            lines.extend(["### Learnability", "", _render_model_contract_table(
                extension_contract(ext_id).tables[0], ext["rows"], target="markdown", scope_styler=scope_text,
            )])
        elif ext_id == "direct_hmhs_joint_retrieval":
            lines.append(_render_model_contract_table(
                extension_contract(ext_id).tables[0], ext["rows"], target="markdown", scope_styler=scope_text,
            ))
    return lines

def _simple_report_details(
    command: str,
    args: list[str],
    *,
    filter_id: str,
    architecture: str,
    profile: str,
) -> tuple[list[tuple[str, object]], Path | None]:
    rows: list[tuple[str, object]] = []
    detail_report: Path | None = None

    if command == "build-dataset":
        summary = _read_dataset_summary(filter_id) or {}
        counts = dict(summary.get("label_counts") or {})
        group_summary = dict(summary.get("event_group_summary") or {})
        rows.extend(
            [
                ("Events", summary.get("event_count")),
                ("Groups", group_summary.get("group_count")),
                ("PASS／REJECT", f"{counts.get('pass', '-')} / {counts.get('reject', '-') }"),
                ("日期範圍", summary.get("selected_date_range") or summary.get("date_range") or "-"),
            ]
        )
    elif command == "compare-daily-targets":
        candidate = get_breakout_quality_experiment_profile(profile)
        spec = get_continuous_ranker_research_spec(profile)
        reference_name = str(
            _cli_option_value(
                args,
                "--reference-experiment-profile",
                spec.reference_profile_name or "",
            )
        ).strip()
        reference = get_breakout_quality_experiment_profile(reference_name) if reference_name else None
        if reference is not None:
            detail_dir = (
                resolve_filter_output_dir(PROJECT_ROOT, filter_id=filter_id)
                / "daily_target_comparison"
                / f"{candidate.continuous_target_id}__vs__{reference.continuous_target_id}"
            )
            report_json = detail_dir / "daily_target_comparison.json"
            payload = load_json_object_or_none(report_json) or {}
            metrics = dict(payload.get("metrics") or {})
            rows.extend(
                [
                    ("Common stock-days", metrics.get("common_sample_count")),
                    ("Risk breach", _fmt_simple_metric(metrics.get("risk_breach_rate"), percent=True)),
                    ("Target changed", _fmt_simple_metric(metrics.get("changed_target_rate"), percent=True)),
                    ("Daily rank correlation", _fmt_simple_metric(metrics.get("mean_daily_spearman_reference_vs_candidate"))),
                    ("Top-K overlap", _fmt_simple_metric(metrics.get("mean_daily_top_k_overlap"), percent=True)),
                    ("Mean abs delta R", _fmt_simple_metric((metrics.get("target_delta_r") or {}).get("mean_abs"))),
                ]
            )
            detail_report = detail_dir / "daily_target_comparison.md"
    elif command == "train-continuous-ranker":
        output_dir = resolve_filter_model_output_dir(
            PROJECT_ROOT, filter_id, architecture, profile
        )
        report_json = output_dir / CONTINUOUS_RANKER_REPORT_FILENAME
        payload = _continuous_ranker_simple_report_payload(
            filter_id=filter_id,
            architecture=architecture,
            profile=profile,
        )
        training = dict(payload.get("training") or {})
        epoch_selection = dict(training.get("epoch_selection") or {})
        metrics = dict(payload.get("split_metrics") or {})
        selection_pareto = epoch_selection.get(
            "best_validation_mean_daily_pareto_pair_concordance"
        )
        direct_hmhs_only = training.get("objective") == TRAINING_OBJECTIVE_DAILY_HMHS_PAIRWISE_RANKING
        rows.extend(
            [
                ("Selected epoch", training.get("selected_epoch")),
                *(
                    [
                        ("選模 Validation HM/HS Pair", _fmt_simple_metric(epoch_selection.get("best_validation_direct_hmhs_pairwise_concordance"), percent=True)),
                        ("選模 Validation HM/HS PR-AUC", _fmt_simple_metric(epoch_selection.get("best_validation_direct_hmhs_global_average_precision"))),
                    ]
                    if direct_hmhs_only
                    else [
                        ("選模 Validation Pareto", _fmt_simple_metric(selection_pareto)),
                        (
                            "選模 epoch Economic rho",
                            _fmt_simple_metric(epoch_selection.get("best_validation_mean_daily_spearman")),
                        ),
                    ]
                    if selection_pareto is not None
                    else [
                        ("選模 Validation rho", _fmt_simple_metric(epoch_selection.get("best_validation_mean_daily_spearman")))
                    ]
                ),
                *(
                    [
                        ("Validation HM/HS Pair", _fmt_simple_metric((metrics.get("validation") or {}).get("pairwise_concordance"), percent=True)),
                        ("Forward OOS HM/HS Pair", _fmt_simple_metric((metrics.get("oos") or {}).get("pairwise_concordance"), percent=True)),
                    ]
                    if direct_hmhs_only
                    else [
                        ("重訓後原 Validation rho", _fmt_simple_metric((metrics.get("validation") or {}).get("mean_daily_spearman"))),
                        ("Forward OOS rho", _fmt_simple_metric((metrics.get("oos") or {}).get("mean_daily_spearman"))),
                    ]
                ),
                *(
                    [
                        (
                            "Breakout slice OOS rho",
                            _fmt_simple_metric(
                                (metrics.get("breakout_candidate_oos") or {}).get("mean_daily_spearman")
                            ),
                        )
                    ]
                    if metrics.get("breakout_candidate_oos")
                    else [
                        (
                            "重訓後 Selection rho",
                            _fmt_simple_metric((metrics.get("selection") or {}).get("mean_daily_spearman")),
                        )
                    ]
                ),
            ]
        )
        pareto_eval = dict(payload.get("pareto_pair_evaluation") or {})
        if pareto_eval:
            rows.extend(
                [
                    (
                        "Forward OOS Pareto",
                        _fmt_simple_metric(
                            (pareto_eval.get("oos") or {}).get(
                                "mean_daily_pareto_pair_concordance"
                            )
                        ),
                    ),
                    (
                        "Breakout slice Pareto",
                        _fmt_simple_metric(
                            (pareto_eval.get("breakout_candidate_oos") or {}).get(
                                "mean_daily_pareto_pair_concordance"
                            )
                        ),
                    ),
                ]
            )
        dual_component_eval = dict(payload.get("dual_component_evaluation") or {})
        oos_components = dict(dual_component_eval.get("oos") or {})
        if oos_components:
            favorable = dict(oos_components.get("favorable_mfe_r") or {})
            adverse = dict(oos_components.get("adverse_to_peak_r") or {})
            rows.extend(
                [
                    (
                        "OOS MFE component rho",
                        _fmt_simple_metric(favorable.get("mean_daily_spearman")),
                    ),
                    (
                        "OOS Adverse component rho",
                        _fmt_simple_metric(adverse.get("mean_daily_spearman")),
                    ),
                ]
            )
        reference_eval = dict(payload.get("reference_target_evaluation") or {})
        if reference_eval.get("available"):
            reference_metrics = dict(reference_eval.get("split_metrics") or {})
            rows.extend(
                [
                    (
                        "Reference Target OOS rho",
                        _fmt_simple_metric((reference_metrics.get("oos") or {}).get("mean_daily_spearman")),
                    ),
                    (
                        "Reference Target Pair",
                        _fmt_simple_metric((reference_metrics.get("oos") or {}).get("pairwise_concordance"), percent=True),
                    ),
                ]
            )
        candidate = output_dir / "continuous_ranker_report.md"
        detail_report = candidate if candidate.is_file() else None
    elif command == "compare-continuous-rankers":
        output_dir = (
            resolve_filter_output_dir(PROJECT_ROOT, filter_id=filter_id)
            / "continuous_ranker_comparison"
        )
        payload = load_json_object_or_none(output_dir / "continuous_ranker_comparison.json") or {}
        configured = dict(payload.get("comparison_settings") or {})
        model_ids = tuple(str(value) for value in configured.get("model_ids") or ())
        summary_pair = tuple(str(value) for value in configured.get("summary_pair") or ())
        if len(summary_pair) != 2:
            comparison_settings = get_breakout_quality_continuous_ranker_comparison_settings()
            summary_pair = comparison_settings.summary_pair
        summary_left, summary_right = summary_pair
        if not model_ids:
            model_ids = tuple(
                str(item.get("model_id"))
                for item in payload.get("models") or []
                if str(item.get("model_id") or "").strip()
            )
        fixed_oos = dict(((payload.get("fixed_k") or {}).get("splits") or {}).get("oos") or {})
        fixed_summary = dict((fixed_oos.get("models") or {}).get(summary_left) or {})
        fixed_prefix = dict(payload.get("fixed_k_prefix_sweep") or {})
        prefix_by_k = dict(fixed_prefix.get("by_k") or {})
        first_prefix_k = min((int(value) for value in prefix_by_k), default=None)
        first_prefix = dict(prefix_by_k.get(str(first_prefix_k)) or {}) if first_prefix_k is not None else {}
        first_prefix_contrast = dict(
            (first_prefix.get("paired_contrasts") or {}).get(f"{summary_left}_minus_{summary_right}") or {}
        )
        first_prefix_metrics = dict(first_prefix_contrast.get("metrics") or {})
        first_prefix_ndcg_delta = dict(first_prefix_metrics.get("ndcg_at_k") or {})
        first_prefix_boundary_delta = dict(first_prefix_metrics.get("boundary_concordance") or {})
        dynamic = dict(payload.get("dynamic_k") or {})
        coverage = dict(dynamic.get("coverage") or {})
        dynamic_eval = dict(dynamic.get("evaluation") or {})
        dynamic_summary = dict((dynamic_eval.get("models") or {}).get(summary_left) or {})
        contrast_id = f"{summary_left}_minus_{summary_right}"
        dynamic_contrast = dict(
            (dynamic_eval.get("paired_contrasts") or {}).get(contrast_id) or {}
        )
        dynamic_boundary_delta = dict(
            (dynamic_contrast.get("metrics") or {}).get("boundary_concordance") or {}
        )
        reference_attribution = dict(dynamic.get("reference_subset_attribution") or {})
        score_date_diag = dict(dynamic.get("score_event_date_comparability") or {})
        score_date_scopes = dict(score_date_diag.get("pair_scopes") or {})
        cross_scope = dict(score_date_scopes.get("cross_score_event_date_pairs") or {})
        k1_cross_scope = dict(score_date_scopes.get("k1_cross_score_event_date_pairs") or {})
        first_reference_k = min((int(value) for value in reference_attribution), default=None)
        first_reference = (
            dict(reference_attribution.get(str(first_reference_k)) or {})
            if first_reference_k is not None
            else {}
        )
        first_reference_event = dict(first_reference.get("event_universe") or {})
        first_reference_orderable = dict(first_reference.get("orderable_universe") or {})

        def _reference_delta(section: dict, metric: str):
            contrast = dict(
                (section.get("paired_contrasts") or {}).get(contrast_id) or {}
            )
            return dict((contrast.get("metrics") or {}).get(metric) or {}).get("mean_delta")

        def _score_date_delta(section: dict, metric: str):
            contrast = dict((section.get("contrasts") or {}).get(contrast_id) or {})
            return contrast.get(metric)

        rows.extend(
            [
                ("Fixed OOS競爭日", fixed_oos.get("competition_date_count")),
                (f"{summary_left} OOS NDCG", _fmt_simple_metric(fixed_summary.get("ndcg_at_k"))),
                (f"{summary_left} OOS Boundary", _fmt_simple_metric(fixed_summary.get("boundary_concordance"), percent=True)),
                *(
                    [
                        (
                            f"Fixed OOS K={first_prefix_k} NDCG Δ ({summary_left}−{summary_right})",
                            "-" if first_prefix_ndcg_delta.get("mean_delta") is None else f"{float(first_prefix_ndcg_delta.get('mean_delta')):+.4f}",
                        ),
                        (
                            f"Fixed OOS K={first_prefix_k} Boundary Δ ({summary_left}−{summary_right})",
                            "-" if first_prefix_boundary_delta.get("mean_delta") is None else f"{float(first_prefix_boundary_delta.get('mean_delta')) * 100.0:+.2f}pp",
                        ),
                    ]
                    if first_prefix_k is not None
                    else []
                ),
                ("Dynamic-K common-complete日", coverage.get("full_score_coverage_date_count")),
                ("Dynamic Target基準", "原始score-event-date（非trade-date剩餘機會）"),
                (f"{summary_left} Dynamic raw-score Boundary", _fmt_simple_metric(dynamic_summary.get("boundary_concordance"), percent=True)),
                (
                    f"{summary_left}−{summary_right} Dynamic raw-score Boundary Δ",
                    "-"
                    if dynamic_boundary_delta.get("mean_delta") is None
                    else f"{float(dynamic_boundary_delta.get('mean_delta')) * 100.0:+.2f}pp",
                ),
                *(
                    [
                        (
                            f"Ref K={first_reference_k} Event NDCG Δ ({summary_left}−{summary_right})",
                            "-"
                            if _reference_delta(first_reference_event, "ndcg_at_k") is None
                            else f"{float(_reference_delta(first_reference_event, 'ndcg_at_k')):+.4f}",
                        ),
                        (
                            f"Ref K={first_reference_k} Orderable NDCG Δ ({summary_left}−{summary_right})",
                            "-"
                            if _reference_delta(first_reference_orderable, "ndcg_at_k") is None
                            else f"{float(_reference_delta(first_reference_orderable, 'ndcg_at_k')):+.4f}",
                        ),
                    ]
                    if first_reference_k is not None
                    else []
                ),
                (
                    f"Cross score-date Pair Δ ({summary_left}−{summary_right})",
                    "-"
                    if _score_date_delta(cross_scope, "pairwise_concordance_delta") is None
                    else f"{float(_score_date_delta(cross_scope, 'pairwise_concordance_delta')) * 100.0:+.2f}pp",
                ),
                (
                    f"K=1 Cross score-date Pair Δ ({summary_left}−{summary_right})",
                    "-"
                    if _score_date_delta(k1_cross_scope, "pairwise_concordance_delta") is None
                    else f"{float(_score_date_delta(k1_cross_scope, 'pairwise_concordance_delta')) * 100.0:+.2f}pp",
                ),
            ]
        )
        candidate = output_dir / "continuous_ranker_comparison.md"
        detail_report = candidate if candidate.is_file() else None
    elif command in {"build-point-in-time-scores", "audit-point-in-time-scores"}:
        pit_override_raw = _cli_option_value(args, "--point-in-time-dir-override", None)
        pit_override = (
            None
            if pit_override_raw in (None, "")
            else Path(str(pit_override_raw)).resolve()
        )
        pit_manifest = (
            pit_override / "selection_point_in_time_manifest.json"
            if pit_override is not None
            else resolve_selection_point_in_time_manifest_path(
                PROJECT_ROOT, filter_id, architecture, profile
            )
        )
        manifest_payload = load_json_object_or_none(pit_manifest) or {}
        coverage = dict(manifest_payload.get("coverage") or {})
        rows.extend(
            [
                ("Score coverage", _fmt_simple_metric(coverage.get("coverage_rate"), percent=True)),
                (
                    "Scored groups",
                    coverage.get("scored_group_count")
                    if coverage.get("scored_group_count") is not None
                    else "-",
                ),
            ]
        )
        if command == "audit-point-in-time-scores":
            audit_json = (
                pit_override / "selection_point_in_time_audit.json"
                if pit_override is not None
                else resolve_selection_point_in_time_audit_json_path(
                    PROJECT_ROOT, filter_id, architecture, profile
                )
            )
            payload = load_json_object_or_none(audit_json) or {}
            audit_coverage = dict(payload.get("score_coverage") or {})
            if audit_coverage:
                rows[-2:] = [
                    (
                        "Score coverage",
                        _fmt_simple_metric(audit_coverage.get("coverage_rate"), percent=True),
                    ),
                    (
                        "Scored groups",
                        audit_coverage.get("scored_group_count")
                        if audit_coverage.get("scored_group_count") is not None
                        else "-",
                    ),
                ]
            decision = dict(payload.get("decision_contract") or {})
            primary_scope = str(decision.get("primary_metric_scope") or "pass_only_target")
            primary = dict((payload.get("metrics") or {}).get(primary_scope) or {})
            if not primary:
                primary = dict(payload.get("primary_metrics") or payload.get("pass_only_metrics") or {})
            if not primary:
                target_quality = dict(payload.get("target_quality") or {})
                primary = dict(target_quality.get("primary") or {})
            rows.extend(
                [
                    ("Daily rho", _fmt_simple_metric(primary.get("mean_daily_spearman"))),
                    ("Global rho", _fmt_simple_metric(primary.get("global_spearman"))),
                ]
            )
            candidate = (
                pit_override / "selection_point_in_time_audit.md"
                if pit_override is not None
                else resolve_selection_point_in_time_audit_markdown_path(
                    PROJECT_ROOT, filter_id, architecture, profile
                )
            )
            detail_report = candidate if candidate.is_file() else None
    elif command in {"report", "workflow"}:
        report_json = resolve_filter_report_json_path(
            PROJECT_ROOT, filter_id, architecture, profile
        )
        payload = load_json_object_or_none(report_json) or {}
        oos = dict((payload.get("split_summaries") or {}).get("oos") or {})
        conclusion = dict(payload.get("conclusion") or {})
        rows.extend(
            [
                ("OOS 原始 PASS", _fmt_simple_metric(oos.get("base_pass_rate"), percent=True)),
                ("OOS PASS Precision", _fmt_simple_metric(oos.get("pass_precision"), percent=True)),
                ("OOS PASS Recall", _fmt_simple_metric(oos.get("pass_recall"), percent=True)),
                ("OOS Accuracy", _fmt_simple_metric(oos.get("accuracy"), percent=True)),
                ("結論", conclusion.get("title") or conclusion.get("status") or "-"),
            ]
        )
        candidate = resolve_filter_report_markdown_path(
            PROJECT_ROOT, filter_id, architecture, profile
        )
        detail_report = candidate if candidate.is_file() else None
    elif command == "timing-rolling-training":
        timing = get_breakout_quality_rolling_timing_settings()
        action = str(args[0]).strip().lower() if args else "run"
        rows.extend(
            [
                ("Timing action", action),
                ("Score years", list(timing.score_years)),
                ("Seed", int(timing.seed)),
                ("Mode", "Extending / from-scratch / isolated artifacts"),
            ]
        )
        if action in {"run", "compare"}:
            from services.breakout_quality.rolling_timing import (
                resolve_rolling_timing_artifact_paths,
            )

            timing_paths = resolve_rolling_timing_artifact_paths()
            baseline = load_json_object_or_none(timing_paths["baseline"]) or {}
            candidate_payload = load_json_object_or_none(timing_paths["candidate"]) or {}
            comparison = dict(candidate_payload.get("comparison_to_baseline") or {})
            if baseline:
                rows.append(
                    (
                        "Baseline total",
                        render_elapsed(float(baseline.get("elapsed_wall_sec", 0.0) or 0.0)),
                    )
                )
            if comparison:
                rows.extend(
                    [
                        (
                            "Candidate total",
                            render_elapsed(float(comparison.get("candidate_total_sec", 0.0) or 0.0)),
                        ),
                        (
                            "Speedup",
                            f"{float(comparison.get('total_speedup_x') or 0.0):.3f}x",
                        ),
                        (
                            "Exact result",
                            "PASS / bitwise exact"
                            if bool(comparison.get("exact_result"))
                            else "FAIL / result changed",
                        ),
                    ]
                )
            timing_report = timing_paths["report"]
            detail_report = timing_report if timing_report.is_file() else None
    elif command == "prepare-continuous-target":
        target_id = str(
            _cli_option_value(
                args,
                "--target-id",
                get_breakout_quality_experiment_profile(profile).continuous_target_id or "",
            )
        )
        if target_id:
            target_dir = resolve_continuous_target_dir(
                PROJECT_ROOT, filter_id, target_id=target_id
            )
            manifest = load_json_object_or_none(target_dir / TARGET_MANIFEST_FILENAME) or {}
            coverage = dict(manifest.get("coverage") or {})
            rows.extend(
                [
                    ("Target ID", target_id),
                    ("Valid groups", coverage.get("valid_group_count") or manifest.get("valid_group_count")),
                ]
            )
            candidate = target_dir / TARGET_AUDIT_MARKDOWN_FILENAME
            detail_report = candidate if candidate.is_file() else None

    return rows, detail_report


def _emit_breakout_quality_simple_report(
    command: str,
    args: list[str],
    *,
    returncode: int,
    elapsed_sec: float,
) -> Path:
    filter_id, architecture, profile = _simple_report_context(command, args)
    experiment = get_breakout_quality_experiment_profile(profile)
    profile_label = profile
    objective_label = experiment.training_objective
    if command == "compare-continuous-rankers":
        output_dir = (
            resolve_filter_output_dir(PROJECT_ROOT, filter_id=filter_id)
            / "continuous_ranker_comparison"
        )
        comparison_payload = load_json_object_or_none(
            output_dir / "continuous_ranker_comparison.json"
        ) or {}
        configured = dict(comparison_payload.get("comparison_settings") or {})
        model_ids = tuple(str(value) for value in configured.get("model_ids") or ())
        if not model_ids:
            model_ids = get_breakout_quality_continuous_ranker_comparison_settings().model_ids
        profile_label = " / ".join(model_ids)
        objective_label = "paired ranking-quality comparison (read-only)"
    detail_rows, detail_report = _simple_report_details(
        command,
        list(args),
        filter_id=filter_id,
        architecture=architecture,
        profile=profile,
    )
    status = "PASS" if int(returncode) == 0 else f"FAIL ({int(returncode)})"
    rows: list[tuple[str, object]] = [
        ("動作", command),
        ("狀態", status),
        ("Filter ID", filter_id),
        ("Architecture", architecture),
        ("Profile", profile_label),
        ("Objective", objective_label),
        ("耗時", render_elapsed(float(elapsed_sec))),
        *detail_rows,
    ]
    if detail_report is not None:
        rows.append(("詳細報表", project_relative_display_path(detail_report, project_root=PROJECT_ROOT)))

    color = console_color_enabled()
    console_rows: list[tuple[str, object]] = []
    for label, value in rows:
        if label == "狀態":
            value = styled_workflow_status(value, target="console", bold=True)
        elif label in {"Forward OOS rho", "Breakout slice OOS rho", "重訓後 Selection rho"}:
            value = paint(value, "cyan", enabled=color, bold=True)
        console_rows.append((label, value))
    print(
        "\n"
        + render_title(
            paint("Breakout Quality 簡易報表", "cyan", enabled=color, bold=True)
        )
    )
    print(render_key_values(console_rows))
    ranker_payload: dict = {}
    if command == "train-continuous-ranker":
        ranker_payload = _continuous_ranker_simple_report_payload(
            filter_id=filter_id,
            architecture=architecture,
            profile=profile,
        )
        ranker_console = _render_continuous_ranker_simple_console(ranker_payload)
        if ranker_console:
            print(ranker_console)

    report_dir = resolve_filter_output_dir(PROJECT_ROOT, filter_id=filter_id) / "simple_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    safe_command = command.replace("/", "_").replace("\\", "_")
    report_path = report_dir / f"{safe_command}.md"
    markdown_lines = [
        f"# {markdown_tone('Breakout Quality 簡易報表', 'blue', bold=True)}",
        "",
        f"- Generated at UTC：`{datetime.now(timezone.utc).isoformat()}`",
    ]
    for label, value in rows:
        rendered = value
        if label == "狀態":
            rendered = styled_workflow_status(value, target="markdown", bold=True)
        elif label in {"Forward OOS rho", "Breakout slice OOS rho", "重訓後 Selection rho"}:
            rendered = markdown_tone(value, "blue", bold=True)
        markdown_lines.append(f"- **{label}**：{rendered}")
    if ranker_payload:
        markdown_lines.extend(_render_continuous_ranker_simple_markdown(ranker_payload))
    report_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    print(render_status_paths((("簡易報表", report_path, True),), project_root=PROJECT_ROOT))
    return report_path


def _run_command(command: str, args: list[str], *, program_name: str) -> int:
    command_module = _load_command_module(command)
    original_program_name = sys.argv[0]
    sys.argv[0] = _command_program_name(program_name, command)
    started = time.perf_counter()
    try:
        result = command_module.main(list(args))
    finally:
        sys.argv[0] = original_program_name
    returncode = int(result or 0)
    if returncode == 0:
        timing_action = (
            str(args[0]).strip().lower()
            if command == "timing-rolling-training" and args
            else "run"
        )
        if command != "timing-rolling-training" or timing_action in {"run", "compare"}:
            _emit_breakout_quality_simple_report(
                command,
                list(args),
                returncode=returncode,
                elapsed_sec=time.perf_counter() - started,
            )
    return returncode


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




def _dataset_refresh_plan(
    filter_id: str,
    dataset: str,
    *,
    max_tickers: int,
) -> tuple[str, list[str]]:
    """Compatibility facade over canonical metadata-only Dataset readiness."""

    readiness = collect_dataset_readiness(
        PROJECT_ROOT,
        filter_id=str(filter_id),
        dataset=str(dataset),
        max_tickers=int(max_tickers),
        model_architecture=BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
    )
    return str(readiness.refresh_mode), list(readiness.reasons)


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






def _parse_workflow_args(argv=None, *, program_name: str = "apps/research.py model") -> argparse.Namespace:
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


def _build_export_score_argv(
    args: argparse.Namespace,
    *,
    scope: str,
) -> list[str]:
    argv = [
        "--filter-id",
        str(args.filter_id),
        "--experiment-profile",
        str(args.experiment_profile),
        "--scope",
        str(scope),
        "--inference-batch-size",
        str(int(args.evaluation_batch_size)),
        "--inference-workers",
        str(int(args.evaluation_workers)),
        "--device",
        str(args.device),
        "--mixed-precision-dtype",
        str(args.mixed_precision_dtype),
    ]
    argv.append(
        "--mixed-precision"
        if bool(args.mixed_precision)
        else "--no-mixed-precision"
    )
    argv.append(
        "--deterministic-algorithms"
        if bool(args.deterministic_algorithms)
        else "--no-deterministic-algorithms"
    )
    argv.append("--allow-tf32" if bool(args.allow_tf32) else "--no-allow-tf32")
    argv.append(
        "--preload-feature-bank"
        if bool(args.preload_feature_bank)
        else "--no-preload-feature-bank"
    )
    return argv


def _model_runtime_description(model_spec) -> str:
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
    model_spec = get_active_model_spec(BREAKOUT_QUALITY_MODEL_ARCHITECTURE)
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
                _build_export_score_argv(args, scope="research"),
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
    workflow_report_args = [
        "--filter-id", filter_id,
        "--model-architecture", str(BREAKOUT_QUALITY_MODEL_ARCHITECTURE),
        "--experiment-profile", str(args.experiment_profile),
    ]
    _emit_breakout_quality_simple_report(
        "workflow",
        workflow_report_args,
        returncode=0,
        elapsed_sec=workflow_elapsed,
    )
    return 0




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




















def _print_workflow_status(settings=None) -> None:
    if settings is None:
        settings = get_breakout_quality_model_research_settings()
    profile = get_breakout_quality_experiment_profile(settings.experiment_profile)
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
        ("Sample Scope", settings.training_sample_scope),
        ("Seed", settings.seed),
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
    base_rows.append(("Continuous Target", settings.continuous_target_id))
    if settings.rolling_authorized:
        rolling_modes = get_breakout_quality_rolling_test_modes()
        base_rows.extend((
            (
                "Time Test Modes",
                " / ".join(
                    (
                        f"{mode.label}={mode.score_start_date}→{('最新' if str(mode.score_end_date).lower() == 'auto' else mode.score_end_date)} / single block"
                        if mode.single_score_block
                        else f"{mode.label}={mode.score_start_date}→{mode.score_end_date} / {int(mode.fold_months)}M"
                    )
                    for mode in rolling_modes
                ),
            ),
            (
                "Inner Validation",
                f"{settings.point_in_time_inner_validation_months} months",
            ),
            (
                "Evidence Start",
                settings.point_in_time_coverage_reference_start_date,
            ),
        ))
    else:
        base_rows.append(("Rolling PIT", "目前Active Profile未授權current Rolling"))
    print(render_key_values(base_rows))
    model_artifacts = resolve_filter_artifact_paths(
        PROJECT_ROOT, settings.filter_id, settings.model_architecture, settings.experiment_profile
    )
    model_output_dir = resolve_filter_model_output_dir(
        PROJECT_ROOT, settings.filter_id, settings.model_architecture, settings.experiment_profile
    )
    status_paths = {
        "Dataset summary": _dataset_paths(settings.filter_id)["summary"],
        "Full model": model_artifacts.model_path,
        "Full model manifest": model_artifacts.manifest_path,
        "Full model report": model_output_dir / CONTINUOUS_RANKER_REPORT_FILENAME,
        "Forward OOS scores": resolve_continuous_ranker_oos_score_path(
            PROJECT_ROOT,
            settings.filter_id,
            settings.model_architecture,
            settings.experiment_profile,
        ),
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
    }
    grouped_status = [
        ("Dataset", (status_paths["Dataset summary"],)),
    ]
    if profile.training_sample_scope != TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS:
        grouped_status.append(
            (
                "Continuous Target",
                (status_paths["Target manifest"], status_paths["Target audit Markdown"]),
            )
        )
    grouped_status.append(
        (
            "Legacy Frozen Model / OOS",
            (
                status_paths["Full model"],
                status_paths["Full model manifest"],
                status_paths["Full model report"],
                status_paths["Forward OOS scores"],
            ),
        )
    )
    if settings.rolling_authorized:
        for mode in get_breakout_quality_rolling_test_modes():
            extending_dir = _rolling_mode_point_in_time_dir(
                settings, mode, fixed_window=False
            )
            if extending_dir is None:
                extending_dir = resolve_filter_model_output_dir(
                    PROJECT_ROOT, settings.filter_id, settings.model_architecture, settings.experiment_profile
                ) / "point_in_time"
            fixed_dir = _rolling_mode_point_in_time_dir(
                settings, mode, fixed_window=True
            )
            assert fixed_dir is not None
            for prefix, base in (("Extending", extending_dir), ("Fixed", fixed_dir)):
                grouped_status.extend((
                    (
                        f"{prefix} {mode.label} Scores",
                        (
                            base / "selection_point_in_time_scores.csv",
                            base / "selection_point_in_time_manifest.json",
                            base / "selection_point_in_time_coverage.csv",
                        ),
                    ),
                    (
                        f"{prefix} {mode.label} Model Gate",
                        (
                            base / "selection_point_in_time_audit.json",
                            base / "selection_point_in_time_audit.md",
                        ),
                    ),
                ))
    status_rows = []
    for label, paths in grouped_status:
        existing = sum(path.is_file() for path in paths)
        if existing == len(paths):
            status, signal = "完整", SIGNAL_POSITIVE
        elif existing == 0:
            status, signal = "缺少", SIGNAL_NEGATIVE
        else:
            status, signal = "不完整", SIGNAL_WARNING
        status_rows.append(
            (styled_signal(f"[{status}]", signal, target="console", bold=True), label)
        )
    print(
        render_section(
            paint("Workflow 狀態", "cyan", enabled=color_enabled, bold=True)
        )
    )
    print(render_table(("狀態", "項目"), status_rows))




def _trade_path_train_request(workflow_settings) -> argparse.Namespace:
    request = _policy_train_settings(TRADE_PATH_RESEARCH_FILTER_ID)
    request.experiment_profile = str(workflow_settings.experiment_profile)
    request.seed = int(workflow_settings.seed)
    return request


def _print_trade_path_label_policy(request: argparse.Namespace) -> None:
    print("\n" + render_title("A2 Realized Trade-path Model Policy"))
    print(
        render_key_values(
            (
                ("Filter ID", TRADE_PATH_RESEARCH_FILTER_ID),
                ("Label ID", TRADE_PATH_LABEL_ID),
                ("Teacher Params", "DL-off-trained／A2 PIT active params"),
                ("Event Scope", "original_breakout_lifecycle"),
                ("Initial Miss Buy", "pending／continuation"),
                ("Feature Snapshot", "original signal date"),
                ("Filled Positive", "realized_net_r > 0 → PASS"),
                ("Filled Nonpositive", "realized_net_r <= 0 → REJECT"),
                ("Filled Data End", "formal single-stock forced closeout"),
                ("Unfilled Terminal", "EXCLUDED from binary training"),
                ("Architecture", BREAKOUT_QUALITY_MODEL_ARCHITECTURE),
                ("Experiment Profile", request.experiment_profile),
                ("Threshold", f"{float(request.fixed_threshold):g}"),
            )
        )
    )


def _run_trade_path_model_report(
    program_name: str,
    *,
    request: argparse.Namespace,
    export_research_scores: bool,
) -> int:
    if export_research_scores:
        print("\n[Scores] 更新新Label research scores")
        code = _run_command(
            "export-scores",
            _build_export_score_argv(request, scope="research"),
            program_name=program_name,
        )
        if code != 0:
            return int(code)
    print("\n[Report] 顯示Selection／OOS模型預測效果")
    code = _run_command(
        "report",
        [
            "--filter-id",
            TRADE_PATH_RESEARCH_FILTER_ID,
            "--experiment-profile",
            str(request.experiment_profile),
            "--include-oos",
        ],
        program_name=program_name,
    )
    if code != 0:
        return int(code)
    print("\n[Runtime Scores] 匯出正式forward-OOS scores")
    return _run_command(
        "export-scores",
        _build_export_score_argv(request, scope="forward_oos"),
        program_name=program_name,
    )


def _interactive_trade_path_train_and_report(
    program_name: str,
    *,
    workflow_settings,
) -> int:
    request = _trade_path_train_request(workflow_settings)
    _print_trade_path_label_policy(request)
    print(
        "\n即將執行：建立／接續新Label Dataset → 重新訓練 → "
        "更新research scores → 顯示Selection／OOS模型預測報表 → 匯出forward-OOS scores。"
    )
    print("本流程不執行策略績效比較；比較請另開 apps/research.py compare。")
    if not _prompt_bool("確認開始", True):
        print("已取消。")
        return 0
    steps = (
        (
            "[1/5] 建立／接續A2 realized trade-path Label Dataset",
            "build-trade-path-labels",
            [
                "--dataset",
                INTERACTIVE_DATASET_PROFILE,
                "--filter-id",
                TRADE_PATH_RESEARCH_FILTER_ID,
                "--resume",
            ],
        ),
        (
            "[2/5] 訓練新Label模型",
            "train",
            _build_train_argv(request),
        ),
        (
            "[3/5] 更新新Label research scores",
            "export-scores",
            _build_export_score_argv(request, scope="research"),
        ),
        (
            "[4/5] 顯示Selection／OOS模型預測報表",
            "report",
            [
                "--filter-id",
                TRADE_PATH_RESEARCH_FILTER_ID,
                "--experiment-profile",
                str(request.experiment_profile),
                "--include-oos",
            ],
        ),
        (
            "[5/5] 匯出正式forward-OOS scores",
            "export-scores",
            _build_export_score_argv(request, scope="forward_oos"),
        ),
    )
    for label, command, argv in steps:
        print("\n" + label)
        code = _run_command(command, list(argv), program_name=program_name)
        if code != 0:
            return int(code)
    print("\n模型研究完成。策略績效比較請另開 python apps/research.py compare。")
    return 0


def _interactive_trade_path_existing_report(
    program_name: str,
    *,
    workflow_settings,
) -> int:
    request = _trade_path_train_request(workflow_settings)
    _print_trade_path_label_policy(request)
    print("\n使用既有新Label模型更新research／forward-OOS scores並顯示預測報表；不執行策略比較。")
    if not _prompt_bool("確認開始", True):
        print("已取消。")
        return 0
    return _run_trade_path_model_report(
        program_name,
        request=request,
        export_research_scores=True,
    )


def _interactive_trade_path_label_summary() -> int:
    summary = _read_dataset_summary(TRADE_PATH_RESEARCH_FILTER_ID)
    if summary is None:
        print("尚未建立A2 realized trade-path Label Dataset。")
        return 0
    counts = dict(summary.get("label_counts") or {})
    status_counts = dict(summary.get("label_status_counts") or {})
    reason_counts = dict(summary.get("label_reason_counts") or {})
    trade_path = dict(summary.get("trade_path_label") or {})
    group_summary = dict(summary.get("event_group_summary") or {})
    print("\n" + render_title("A2 Realized Trade-path Label Summary"))
    print(
        render_key_values(
            (
                ("Filter ID", summary.get("filter_id")),
                ("Label ID", trade_path.get("label_id")),
                ("Dataset", summary.get("dataset")),
                ("Events", summary.get("event_count")),
                ("PASS", status_counts.get("PASS", counts.get("pass"))),
                ("REJECT", status_counts.get("REJECT", counts.get("reject"))),
                ("EXCLUDED", status_counts.get("EXCLUDED", counts.get("invalid"))),
                ("Groups", group_summary.get("group_count")),
                ("Valid Groups", group_summary.get("valid_group_count")),
                ("Initial Miss Buy", trade_path.get("initial_miss_buy_status")),
                ("Filled Data End", trade_path.get("filled_data_end_rule")),
                ("Unfilled Terminal", trade_path.get("unfilled_terminal_rule")),
                ("Label Reasons", len(reason_counts)),
                ("Label End", summary.get("label_information_end_date_range")),
            )
        )
    )
    print(
        render_status_paths(
            (
                ("Dataset summary", _dataset_paths(TRADE_PATH_RESEARCH_FILTER_ID)["summary"]),
                ("Events", _dataset_paths(TRADE_PATH_RESEARCH_FILTER_ID)["events"]),
            ),
            project_root=PROJECT_ROOT,
        )
    )
    return 0


def _interactive_binary_model_research(
    program_name: str,
    *,
    workflow_settings,
) -> int:
    while True:
        print("\n=== Binary DL Filter 模型研究與驗證 ===")
        print(f"Active Research Label：{TRADE_PATH_LABEL_ID}")
        print(render_menu_item(1, "建立新Label → 重新訓練 → 模型預測報表", default=True))
        print(render_menu_item(2, "使用既有模型 → 更新Scores → 模型預測報表"))
        print(render_menu_item(3, "查看Label與事件生命週期摘要"))
        print(render_menu_item(0, "返回"))
        try:
            raw_choice = input("👉 請選擇：").strip().lower()
        except EOFError:
            print("\n輸入已結束。")
            return 0
        choice = "1" if raw_choice == "" else raw_choice
        if choice in {"0", "q", "quit", "exit"}:
            return 0
        if choice == "1":
            return _interactive_trade_path_train_and_report(
                program_name,
                workflow_settings=workflow_settings,
            )
        if choice == "2":
            return _interactive_trade_path_existing_report(
                program_name,
                workflow_settings=workflow_settings,
            )
        if choice == "3":
            return _interactive_trade_path_label_summary()
        print("無效選項，請重新輸入。")


def _collect_continuous_research_input_plan(
    settings,
    *,
    dataset_profile: str | None = None,
    max_tickers: int | None = None,
):
    resolved_dataset_profile = str(
        INTERACTIVE_DATASET_PROFILE if dataset_profile is None else dataset_profile
    )
    resolved_max_tickers = int(
        INTERACTIVE_MAX_TICKERS if max_tickers is None else max_tickers
    )
    return collect_model_upstream_preparation_plan(
        PROJECT_ROOT,
        filter_id=str(settings.filter_id),
        model_architecture=str(settings.model_architecture),
        experiment_profile=str(settings.experiment_profile),
        dataset=resolved_dataset_profile,
        max_tickers=resolved_max_tickers,
    )


def _render_continuous_research_input_plan(settings, plan) -> None:
    print("\n" + render_section("Research 前置工件計畫"))
    print(
        render_key_values(
            (
                ("Profile", str(settings.experiment_profile)),
                ("整體狀態", str(plan.overall_status)),
            )
        )
    )
    rows = []
    for action in plan.actions:
        rows.append(
            (
                str(action.action),
                str(action.artifact_key),
                project_relative_display_path(action.path, project_root=PROJECT_ROOT),
                str(action.description),
            )
        )
    if rows:
        print(render_table(("動作", "工件", "路徑", "說明"), rows))


def _prepare_continuous_research_inputs(
    program_name: str,
    settings,
    *,
    dataset_profile: str | None = None,
    max_tickers: int | None = None,
) -> int:
    """Prepare deterministic model inputs through the shared Research orchestrator.

    This handles cold-start, partial deletion, corruption, stale source inventory, schema
    changes and relabel-only refresh with one dependency/re-plan contract.
    """

    color_enabled = console_color_enabled()
    resolved_dataset_profile = str(
        INTERACTIVE_DATASET_PROFILE if dataset_profile is None else dataset_profile
    )
    resolved_max_tickers = int(
        INTERACTIVE_MAX_TICKERS if max_tickers is None else max_tickers
    )

    def _refresh_plan():
        return _collect_continuous_research_input_plan(
            settings,
            dataset_profile=resolved_dataset_profile,
            max_tickers=resolved_max_tickers,
        )

    def _execute(action) -> None:
        if action.builder_type == "breakout_quality_dataset":
            refresh_mode, refresh_reasons, dataset_step = _dataset_refresh_step(
                str(settings.filter_id),
                resolved_dataset_profile,
                max_tickers=resolved_max_tickers,
            )
            if dataset_step is None:
                return
            command, command_args, label = dataset_step
            reason_summary = _compact_dataset_refresh_reason(refresh_reasons)
            print(
                paint("[Dataset]", "cyan", enabled=color_enabled, bold=True)
                + f" {label}｜{reason_summary}"
            )
            with _compact_console_scope():
                code = _run_command(command, command_args, program_name=program_name)
            if code != 0:
                raise RuntimeError(f"Dataset canonical builder失敗: returncode={code}")
            return
        if action.builder_type == "breakout_quality_continuous_target":
            profile = get_breakout_quality_experiment_profile(settings.experiment_profile)
            with _compact_console_scope():
                code = _run_command(
                    "prepare-continuous-target",
                    [
                        "--filter-id", str(settings.filter_id),
                        "--target-id", str(profile.continuous_target_id),
                    ],
                    program_name=program_name,
                )
            if code != 0:
                raise RuntimeError(
                    f"Continuous Target canonical builder失敗: returncode={code}"
                )
            return
        raise RuntimeError(f"不支援的model upstream builder: {action.builder_type}")

    try:
        outcome = run_research_artifact_preparation(
            plan_refresher=_refresh_plan,
            action_executor=_execute,
            failure_prefix="模型研究前置",
            on_action=lambda action: print(
                "  Upstream "
                + styled_workflow_status(action.action)
                + f" | {action.artifact_key.split(':', 1)[-1]} | {action.path}"
            ),
        )
    except RuntimeError as exc:
        print(styled_workflow_status("[Upstream] FAIL") + f"｜{exc}")
        return 2

    if not outcome.executed:
        for action in outcome.plan.actions:
            print(
                "  Upstream "
                + styled_workflow_status("REUSE")
                + f" | {action.artifact_key.split(':', 1)[-1]} | {action.path}"
            )

    profile = get_breakout_quality_experiment_profile(settings.experiment_profile)
    if profile.training_sample_scope == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS:
        # The historical MR-13I/J target has a cross-work-type strategy-risk dependency.
        # It remains an explicit research dependency rather than being silently filled by
        # the model trainer.  Current active Daily Universal targets do not use it.
        if str(profile.continuous_target_id or "") == DAILY_RISK_NORMALIZED_NET_OPPORTUNITY_TARGET_ID:
            try:
                schedule = load_min_roos_risk_schedule(PROJECT_ROOT)
            except (FileNotFoundError, OSError, ValueError, KeyError, TypeError) as exc:
                print(
                    styled_workflow_status("[Risk params] BLOCKED")
                    + f"｜{exc}"
                )
                print(
                    "此target需要historical-effective Min ROOS風險參數，屬跨工作類型且目前"
                    "不是current workflow；不得由模型trainer自行猜測或補值。"
                )
                return 2
            print(
                styled_workflow_status("[Risk params] READY")
                + "｜historical-effective Min ROOS atr_len / atr_times_init"
                + f"｜periods={len(schedule)}"
            )
    return 0




def _rolling_mode_point_in_time_dir(settings, mode, *, fixed_window: bool) -> Path | None:
    model_output_dir = resolve_filter_model_output_dir(
        PROJECT_ROOT, settings.filter_id, settings.model_architecture, settings.experiment_profile
    )
    if fixed_window:
        base = (
            model_output_dir
            / "fixed_window_rolling"
            / f"fixed_{int(BREAKOUT_QUALITY_STABILITY_TRAIN_WINDOW_MONTHS)}m"
        )
        if mode.point_in_time_dirname in (None, ""):
            return base
        return base / str(mode.point_in_time_dirname)
    if mode.point_in_time_dirname in (None, ""):
        return None
    return model_output_dir / str(mode.point_in_time_dirname)


def _rolling_mode_display_label(mode) -> str:
    end_label = "最新" if str(mode.score_end_date).strip().lower() == "auto" else str(mode.score_end_date)
    if bool(mode.single_score_block):
        return f"{mode.label} | {mode.score_start_date}→{end_label}"
    return f"{mode.label} | {mode.score_start_date}→{end_label} | {int(mode.fold_months)}M"


def _render_rolling_mode_line(index: int, mode, *, default: bool = False) -> str:
    return render_menu_item(index, _rolling_mode_display_label(mode), default=default)


def _interactive_continuous_rolling_test(
    program_name: str, settings, *, fixed_window: bool
) -> int:
    if not settings.rolling_authorized:
        print("目前Active Profile尚未授權Rolling PIT。")
        return 0
    modes = get_breakout_quality_rolling_test_modes()
    title = "Fixed-Window Stability Test" if fixed_window else "Extending-Window Test"
    while True:
        print(f"\n=== {title} ===")
        for index, mode in enumerate(modes, start=1):
            print(_render_rolling_mode_line(index, mode, default=index == 1))
        status_choice = len(modes) + 1
        print(render_menu_item(status_choice, "查看設定與工件狀態"))
        print(render_menu_item(0, "返回"))
        try:
            raw_choice = input("👉 請選擇：").strip().lower()
        except EOFError:
            return 0
        choice = "1" if raw_choice == "" else raw_choice
        if choice in {"0", "q", "quit", "exit"}:
            return 0
        try:
            numeric = int(choice)
        except ValueError:
            print("無效選項，請重新輸入。")
            continue
        if 1 <= numeric <= len(modes):
            mode = modes[numeric - 1]
            _print_workflow_status(settings)
            if fixed_window:
                purpose = "固定120M calendar train history的歷史learnability／stability診斷"
            elif bool(mode.single_score_block):
                purpose = "固定2020年底information cutoff的OOS Gate"
            else:
                purpose = "expanding history＋12M annual refit的PIT Rolling evidence"
            print(
                render_key_values(
                    (
                        ("Mode", mode.label),
                        (
                            "Score/refit",
                            "single forward block" if mode.single_score_block else f"{int(mode.fold_months)} months",
                        ),
                        (
                            "Score period",
                            f"{mode.score_start_date} ～ "
                            f"{('最新' if str(mode.score_end_date).strip().lower() == 'auto' else mode.score_end_date)}",
                        ),
                        ("用途", purpose),
                    )
                )
            )
            upstream_plan = _collect_continuous_research_input_plan(settings)
            _render_continuous_research_input_plan(settings, upstream_plan)
            if upstream_plan.blocked:
                print("目前存在不可由canonical producer確定性補建的前置工件；本次不執行。")
                continue
            if not _prompt_bool(f"確認執行{mode.label}（含必要自動前置）", True):
                continue
            spec = get_continuous_ranker_research_spec(settings.experiment_profile)
            return _run_continuous_pit_profile(
                program_name,
                model_id=spec.model_research_id,
                profile_name=settings.experiment_profile,
                mode=mode,
                fixed_window=fixed_window,
            )
        if numeric == status_choice:
            _print_workflow_status(settings)
            continue
        print("無效選項，請重新輸入。")


def _interactive_continuous_pit_validation(program_name: str, settings) -> int:
    return _interactive_continuous_rolling_test(
        program_name, settings, fixed_window=False
    )


def _run_continuous_forward_model_gate(program_name: str, settings) -> int:
    """Train the active continuous model and emit its Forward-OOS model report.

    This is the legal entry for model-gate-only research profiles.  It deliberately
    does not build PIT/rolling artifacts or change current-time authorization.
    """

    _print_workflow_status(settings)
    upstream_plan = _collect_continuous_research_input_plan(settings)
    _render_continuous_research_input_plan(settings, upstream_plan)
    if upstream_plan.blocked:
        print("目前存在不可由canonical producer確定性補建的前置工件；本次不執行。")
        return 0
    if not _prompt_bool("確認訓練目前模型並產生Forward-OOS模型報表（含必要自動前置）", True):
        return 0
    code = _prepare_continuous_research_inputs(program_name, settings)
    if code != 0:
        return int(code)
    return int(
        _run_command(
            "train-continuous-ranker",
            [
                "--filter-id", str(settings.filter_id),
                "--model-architecture", str(settings.model_architecture),
                "--experiment-profile", str(settings.experiment_profile),
                "--seed", str(int(settings.seed)),
            ],
            program_name=program_name,
        )
    )


def _run_continuous_pit_profile(
    program_name: str,
    *,
    model_id: str,
    profile_name: str,
    mode=None,
    fixed_window: bool = False,
) -> int:
    settings = get_breakout_quality_workflow_settings(experiment_profile=profile_name)
    selected_mode = mode or get_breakout_quality_rolling_test_mode("rolling")
    pit_dir_override = _rolling_mode_point_in_time_dir(
        settings, selected_mode, fixed_window=fixed_window
    )
    window_label = "Fixed-Window" if fixed_window else "Extending-Window"
    print(
        "\n"
        + render_title(
            f"{window_label} {selected_mode.label} | {model_id} | {settings.experiment_profile}"
        )
    )
    code = _prepare_continuous_research_inputs(program_name, settings)
    if code != 0:
        return int(code)
    score_start = str(selected_mode.score_start_date)
    score_end = selected_mode.score_end_date
    build_args = [
        "--filter-id", settings.filter_id,
        "--model-architecture", settings.model_architecture,
        "--experiment-profile", settings.experiment_profile,
        "--score-start-date", str(score_start),
        "--fold-months", str(int(selected_mode.fold_months)),
        "--inner-validation-months", str(settings.point_in_time_inner_validation_months),
        "--seed", str(settings.seed),
    ]
    if selected_mode.fold_anchor_date is not None:
        build_args.extend(["--fold-anchor-date", str(selected_mode.fold_anchor_date)])
    if selected_mode.single_score_block:
        build_args.append("--single-score-block")
    if score_end:
        build_args.extend(["--score-end-date", str(score_end)])
    train_window = (
        int(BREAKOUT_QUALITY_STABILITY_TRAIN_WINDOW_MONTHS)
        if fixed_window
        else settings.point_in_time_train_window_months
    )
    if train_window is not None:
        build_args.extend(["--train-window-months", str(int(train_window))])
    if pit_dir_override is not None:
        build_args.extend(["--point-in-time-dir-override", str(pit_dir_override)])
    build_args.append("--resume" if settings.point_in_time_resume else "--no-resume")
    audit_args = [
        "--filter-id", settings.filter_id,
        "--model-architecture", settings.model_architecture,
        "--experiment-profile", settings.experiment_profile,
    ]
    if pit_dir_override is not None:
        audit_args.extend(["--point-in-time-dir-override", str(pit_dir_override)])
    with _compact_console_scope():
        code = _run_command(
            "build-point-in-time-scores", build_args, program_name=program_name
        )
        if code != 0:
            return int(code)
        return int(
            _run_command(
                "audit-point-in-time-scores", audit_args, program_name=program_name
            )
        )


def _interactive_continuous_stability_validation(program_name: str, settings) -> int:
    return _interactive_continuous_rolling_test(
        program_name, settings, fixed_window=True
    )


def _run_continuous_rolling_mode_direct(
    program_name: str,
    settings,
    *,
    fixed_window: bool,
) -> int:
    """Run the canonical 12M Rolling mode without an extra mode-selection menu."""

    mode = get_breakout_quality_rolling_test_mode("rolling")
    title = "Fixed-Window Rolling" if fixed_window else "Extending-Window Rolling"
    print(f"\n=== {title} ===")
    if not settings.rolling_authorized:
        print("目前Active Profile未授權Rolling PIT；選單固定保留，本次BLOCKED。")
        return 0
    _print_workflow_status(settings)
    print(
        render_key_values(
            (
                ("Mode", mode.label),
                ("Score/refit", f"{int(mode.fold_months)} months"),
                (
                    "Score period",
                    f"{mode.score_start_date} ～ "
                    f"{('最新' if str(mode.score_end_date).strip().lower() == 'auto' else mode.score_end_date)}",
                ),
                (
                    "用途",
                    "固定120M calendar train history的歷史learnability／stability診斷"
                    if fixed_window
                    else "expanding history＋12M annual refit的PIT Rolling evidence",
                ),
            )
        )
    )
    upstream_plan = _collect_continuous_research_input_plan(settings)
    _render_continuous_research_input_plan(settings, upstream_plan)
    if upstream_plan.blocked:
        print("目前存在不可由canonical producer確定性補建的前置工件；本次不執行。")
        return 0
    if not _prompt_bool(f"確認執行{title}（含必要自動前置）", True):
        return 0
    spec = get_continuous_ranker_research_spec(settings.experiment_profile)
    return _run_continuous_pit_profile(
        program_name,
        model_id=spec.model_research_id,
        profile_name=settings.experiment_profile,
        mode=mode,
        fixed_window=fixed_window,
    )


def _interactive_target_model_comparison(program_name: str, settings) -> int:
    research_spec = get_continuous_ranker_research_spec(settings.experiment_profile)
    while True:
        print("\n=== Target／模型比較 ===")
        print(render_menu_item(1, "比較目前 Target 與 Reference Target", default=True))
        print(render_menu_item(2, BREAKOUT_QUALITY_CONTINUOUS_RANKER_COMPARISON_MENU_LABEL))
        print(render_menu_item(0, "返回"))
        try:
            raw_choice = input("👉 請選擇：").strip().lower()
        except EOFError:
            return 0
        choice = "1" if raw_choice == "" else raw_choice
        if choice in {"0", "q", "quit", "exit"}:
            return 0
        if choice == "1":
            if not research_spec.reference_profile_name:
                print("目前Active Profile未設定reference Target；此項BLOCKED。")
                continue
            reference_settings = get_breakout_quality_workflow_settings(
                experiment_profile=str(research_spec.reference_profile_name)
            )
            target_compare_plans = []
            for required_settings in (settings, reference_settings):
                plan = _collect_continuous_research_input_plan(required_settings)
                _render_continuous_research_input_plan(required_settings, plan)
                target_compare_plans.append(plan)
            if any(plan.blocked for plan in target_compare_plans):
                print("Target比較存在不可由canonical producer確定性補建的前置工件；本次不執行。")
                continue
            if not _prompt_bool("確認執行Target比較（含必要自動前置）", True):
                continue
            for required_settings in (settings, reference_settings):
                code = _prepare_continuous_research_inputs(
                    program_name, required_settings
                )
                if code != 0:
                    return int(code)
            return int(
                _run_command(
                    "compare-daily-targets",
                    [
                        "--filter-id", settings.filter_id,
                        "--model-architecture", settings.model_architecture,
                        "--experiment-profile", settings.experiment_profile,
                        "--reference-experiment-profile", research_spec.reference_profile_name,
                    ],
                    program_name=program_name,
                )
            )
        if choice == "2":
            return int(_run_configured_continuous_ranker_model_gates(program_name))
        print("無效選項，請重新輸入。")








def _prepare_strategy_compare_model_upstream(
    program_name: str,
    *,
    workflow,
    dataset_profile: str,
) -> int:
    """Prepare canonical Dataset/Target only when the shared registry says they are stale."""

    plan = _collect_continuous_research_input_plan(
        workflow, dataset_profile=str(dataset_profile), max_tickers=0
    )
    color_enabled = console_color_enabled()
    for item in plan.actions:
        print(
            "  Upstream "
            + styled_workflow_status(item.action)
            + f" | {item.artifact_key.split(':', 1)[-1]} | {item.description}"
            + f" | {project_relative_display_path(item.path, project_root=PROJECT_ROOT)}"
        )
    if plan.overall_status == "READY":
        return 0
    if plan.blocked:
        raise RuntimeError("Strategy Compare model upstream包含不可自動補建的Research依賴")
    return int(
        _prepare_continuous_research_inputs(
            program_name,
            workflow,
            dataset_profile=str(dataset_profile),
            max_tickers=0,
        )
    )


def _strategy_compare_required_model_sources(profile_ids: tuple[str, ...] | None = None):
    """Resolve model sources required by the selected current Strategy Compare modes."""

    from config.strategy_compare import (
        get_strategy_comparison_menu_profiles,
        get_strategy_comparison_settings,
    )

    supported_score_sources = {
        SCORE_SOURCE_SELECTION_POINT_IN_TIME,
        SCORE_SOURCE_CONTINUOUS_RANKER_OOS,
    }
    selected_profile_ids = (
        tuple(str(value) for value in profile_ids)
        if profile_ids is not None
        else tuple(
            str(profile["profile_id"])
            for profile in get_strategy_comparison_menu_profiles()
        )
    )
    comparisons = []
    dedup: dict[tuple[object, ...], tuple[str, object]] = {}
    from filters.breakout_quality.strategy_compare_dl_artifacts import (
        resolve_required_artifact_sources,
    )

    for profile_id in selected_profile_ids:
        comparison = get_strategy_comparison_settings(profile_id)
        comparisons.append(comparison)
        _required_params, required_ids, _runtime_required = resolve_required_artifact_sources(
            comparison
        )
        for dl_id in comparison.dl_sources:
            if dl_id not in required_ids:
                continue
            source = comparison.dl_sources[dl_id]
            if source.score_source not in supported_score_sources:
                continue
            key = (
                str(source.filter_id),
                str(source.model_architecture),
                str(source.experiment_profile),
                str(source.score_source),
                source.point_in_time_score_start_date,
                source.point_in_time_score_end_date,
                source.point_in_time_fold_months,
                source.point_in_time_fold_anchor_date,
                source.point_in_time_single_score_block,
                source.point_in_time_dirname,
            )
            dedup.setdefault(key, (dl_id, source))
    return tuple(comparisons), tuple(dedup.values())


def _prepare_strategy_compare_model_upstream_artifacts(
    *,
    program_name: str,
    profile_ids: tuple[str, ...],
) -> int:
    """Prepare only canonical Dataset/Target truth required by Strategy Compare.

    This is the cross-work-type provider hook used by multi-seed robustness before
    isolated per-seed training.  It intentionally does not build canonical model
    checkpoints, PIT scores, or select a model; those remain separate downstream work.
    """

    comparisons, sources = _strategy_compare_required_model_sources(
        tuple(str(value) for value in profile_ids)
    )
    if not sources:
        print("目前策略比較設定沒有需要準備的模型上游工件。")
        return 0
    datasets = {comparison.dataset for comparison in comparisons}
    if len(datasets) != 1:
        raise ValueError(f"Strategy Compare profiles dataset不一致: {sorted(datasets)}")
    dataset_profile = str(next(iter(datasets)))
    seen: set[tuple[str, str, str]] = set()
    for dl_id, source in sources:
        identity = (
            str(source.filter_id),
            str(source.model_architecture),
            str(source.experiment_profile),
        )
        if identity in seen:
            continue
        seen.add(identity)
        workflow = get_breakout_quality_workflow_settings(
            experiment_profile=str(source.experiment_profile)
        )
        if (
            str(workflow.filter_id) != str(source.filter_id)
            or str(workflow.model_architecture) != str(source.model_architecture)
        ):
            raise ValueError(
                f"Strategy Compare model source與workflow identity不一致: {dl_id}"
            )
        code = _prepare_strategy_compare_model_upstream(
            program_name,
            workflow=workflow,
            dataset_profile=dataset_profile,
        )
        if code != 0:
            return int(code)
    return 0


def prepare_strategy_compare_artifacts(
    *,
    program_name: str,
    profile_ids: tuple[str, ...],
    scope: str,
) -> int:
    """Single provider hook for Strategy Compare model-side dependencies."""

    normalized_scope = str(scope).strip().lower()
    selected = tuple(str(value) for value in profile_ids)
    if normalized_scope == "upstream":
        return int(
            _prepare_strategy_compare_model_upstream_artifacts(
                program_name=program_name, profile_ids=selected
            )
        )
    if normalized_scope == "models":
        return int(
            _prepare_strategy_compare_model_artifacts(
                program_name, profile_ids=selected
            )
        )
    raise ValueError(f"不支援的Strategy Compare artifact scope: {scope!r}")


def _run_strategy_compare_model_build_subprocess(
    job: dict[str, object],
    *,
    active_processes: dict[str, subprocess.Popen],
    active_processes_lock: Lock,
) -> dict[str, object]:
    """Run one single-seed unit through the shared Strategy Compare lifecycle."""

    return run_strategy_compare_training_unit(
        project_root=PROJECT_ROOT,
        source=job["source"],
        workflow=job["workflow"],
        seed=int(job["seed"]),
        model_dir=Path(str(job["model_dir"])),
        research_dir=Path(str(job["research_dir"])),
        log_path=Path(str(job["log_path"])),
        comparison_start=(
            None if job.get("comparison_start") in (None, "")
            else str(job["comparison_start"])
        ),
        comparison_end=(
            None if job.get("comparison_end") in (None, "")
            else str(job["comparison_end"])
        ),
        point_in_time_dir_override=job.get("pit_dir_override"),
        checkpoint_cache_root=job.get("checkpoint_cache_root"),
        model_output_dir=(
            job.get("model_dir")
            if str(job["source"].score_source) == SCORE_SOURCE_CONTINUOUS_RANKER_OOS
            else None
        ),
        research_output_dir=(
            job.get("research_dir")
            if str(job["source"].score_source) == SCORE_SOURCE_CONTINUOUS_RANKER_OOS
            else None
        ),
        registry=active_processes,
        registry_lock=active_processes_lock,
        registry_key=str(job["dl_id"]),
        failure_prefix=(
            f"Strategy Compare canonical模型訓練失敗: dl_id={job['dl_id']}"
            + (
                " | preflight_reuse_validation="
                + str(job["preflight_reuse_validation_error"])
                if job.get("preflight_reuse_validation_error")
                else ""
            )
        ),
        resume=True,
    )


def _prepare_strategy_compare_model_artifacts(
    program_name: str, *, profile_ids: tuple[str, ...] | None = None
) -> int:
    """Prepare current Strategy Compare model artifacts with shared GPU-unit scheduling.

    Normal Strategy Compare keeps its canonical workflow seed (currently one seed) but
    uses the same top-level execution policy as robustness: isolated trainer subprocesses,
    up to the configured GPU worker count, with each PIT training unit retaining serial
    fold execution.  Only throughput changes; model identity, seed, target and fitting
    contracts are untouched.
    """

    comparisons, sources = _strategy_compare_required_model_sources(profile_ids)
    if not sources:
        print("目前策略比較設定沒有需要準備的模型工件。")
        return 0
    datasets = {comparison.dataset for comparison in comparisons}
    if len(datasets) != 1:
        raise ValueError(f"Strategy Compare profiles dataset不一致: {sorted(datasets)}")
    comparison = comparisons[0]

    from config.strategy_compare import (
        STRATEGY_COMPARE_FITTING_CHECKPOINT_CACHE_ROOT,
        STRATEGY_COMPARE_GPU_TRAIN_WORKERS,
        STRATEGY_COMPARE_TRAIN_PROGRESS_INTERVAL_SECONDS,
    )

    worker_count = validate_strategy_compare_gpu_train_workers(STRATEGY_COMPARE_GPU_TRAIN_WORKERS)
    color_enabled = console_color_enabled()
    print(
        paint("策略比較模型工件準備", "cyan", enabled=color_enabled, bold=True)
        + f" | profiles={len(comparisons)} | sources={len(sources)}"
        + f" | dataset={comparison.dataset} | GPU workers={worker_count}"
    )

    # Upstream production is a separate Research dependency phase. Model-only
    # execution must not build Dataset/Target as a side effect, otherwise the
    # shared ordering (upstream -> params -> models) is bypassed.
    pending_upstream: list[str] = []
    seen_upstream: set[tuple[str, str, str]] = set()
    for _dl_id, source in sources:
        upstream_identity = (
            str(source.filter_id),
            str(source.model_architecture),
            str(source.experiment_profile),
        )
        if upstream_identity in seen_upstream:
            continue
        seen_upstream.add(upstream_identity)
        plan = collect_model_upstream_preparation_plan(
            PROJECT_ROOT,
            filter_id=upstream_identity[0],
            model_architecture=upstream_identity[1],
            experiment_profile=upstream_identity[2],
            dataset=str(comparison.dataset),
            max_tickers=0,
        )
        pending_upstream.extend(
            item.artifact_key for item in plan.actions if item.action != "REUSE"
        )
    if pending_upstream:
        raise RuntimeError(
            "Strategy Compare model phase開始前canonical upstream尚未READY: "
            + ", ".join(sorted(set(pending_upstream)))
        )

    jobs: deque[dict[str, object]] = deque()
    legacy_reuse: list[str] = []
    with tempfile.TemporaryDirectory(prefix="strategy_compare_model_training_") as temp_dir:
        log_root = Path(temp_dir)
        ready_reuse: list[str] = []
        for source_index, (dl_id, source) in enumerate(sources, start=1):
            workflow = get_breakout_quality_workflow_settings(
                experiment_profile=str(source.experiment_profile)
            )
            if (
                str(workflow.filter_id) != str(source.filter_id)
                or str(workflow.model_architecture) != str(source.model_architecture)
            ):
                raise ValueError(
                    f"Strategy Compare model source與workflow identity不一致: {dl_id}"
                )

            if source.score_source == SCORE_SOURCE_CONTINUOUS_RANKER_OOS:
                legacy_contract = None
                try:
                    legacy_contract = load_continuous_ranker_oos_contract(
                        PROJECT_ROOT,
                        str(source.filter_id),
                        str(source.model_architecture),
                        str(source.experiment_profile),
                    )
                except (OSError, ValueError, KeyError, TypeError):
                    legacy_contract = None
                if legacy_contract is not None:
                    legacy_reuse.append(str(dl_id))
                    continue

            pit_dir_override = None
            if source.point_in_time_dirname not in (None, ""):
                pit_dir_override = (
                    resolve_filter_model_output_dir(
                        PROJECT_ROOT,
                        str(source.filter_id),
                        str(source.model_architecture),
                        str(source.experiment_profile),
                    )
                    / str(source.point_in_time_dirname)
                )
            if str(source.score_source) == SCORE_SOURCE_SELECTION_POINT_IN_TIME:
                model_dir = (
                    pit_dir_override
                    if pit_dir_override is not None
                    else resolve_selection_point_in_time_score_path(
                        PROJECT_ROOT,
                        str(source.filter_id),
                        str(source.model_architecture),
                        str(source.experiment_profile),
                    ).parent
                )
                research_dir = model_dir
            else:
                model_paths = resolve_filter_artifact_paths(
                    PROJECT_ROOT,
                    str(source.filter_id),
                    str(source.model_architecture),
                    str(source.experiment_profile),
                )
                model_dir = model_paths.model_dir
                research_dir = resolve_filter_model_output_dir(
                    PROJECT_ROOT,
                    str(source.filter_id),
                    str(source.model_architecture),
                    str(source.experiment_profile),
                )
            reuse_validation_error = None
            if str(source.score_source) == SCORE_SOURCE_SELECTION_POINT_IN_TIME:
                try:
                    validate_strategy_compare_training_artifacts(
                        project_root=PROJECT_ROOT,
                        source=source,
                        workflow=workflow,
                        seed=int(workflow.seed),
                        model_dir=model_dir,
                        research_dir=research_dir,
                        comparison_start=None,
                        comparison_end=None,
                    )
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    reuse_validation_error = f"{type(exc).__name__}: {exc}"
                else:
                    ready_reuse.append(str(dl_id))
                    continue

            job = {
                "seed": int(workflow.seed),
                "source_index": int(source_index),
                "source_count": int(len(sources)),
                "dl_id": str(dl_id),
                "source": source,
                "workflow": workflow,
                "pit_dir_override": pit_dir_override,
                "model_dir": model_dir,
                "research_dir": research_dir,
                "checkpoint_cache_root": (
                    PROJECT_ROOT / STRATEGY_COMPARE_FITTING_CHECKPOINT_CACHE_ROOT
                ),
                # Canonical single-seed source owns its PIT score period; robustness
                # supplies explicit seed-scoped periods in its isolated namespace.
                "comparison_start": None,
                "comparison_end": None,
                "log_path": log_root / f"{source_index:02d}_{dl_id}.log",
                "preflight_reuse_validation_error": reuse_validation_error,
            }
            jobs.append(job)

        for dl_id in legacy_reuse:
            print(
                styled_workflow_status("[REUSE]")
                + f" {dl_id} | Frozen compatibility model/report/scores"
            )
        for dl_id in ready_reuse:
            print(
                styled_workflow_status("[REUSE]")
                + f" {dl_id} | Selection PIT model/score/audit工件已READY"
            )

        active_processes: dict[str, subprocess.Popen] = {}
        active_processes_lock = Lock()
        executor = ThreadPoolExecutor(max_workers=worker_count)
        futures: dict[Future, dict[str, object]] = {}
        progress = FixedProgressBlock()
        started_total = time.perf_counter()
        progress_interval = max(1.0, float(STRATEGY_COMPARE_TRAIN_PROGRESS_INTERVAL_SECONDS))
        next_progress = started_total + progress_interval

        def submit_jobs() -> None:
            while jobs and len(futures) < worker_count:
                job = pop_next_seed_diverse_unit(jobs, futures.values())
                job["submitted_at"] = time.perf_counter()
                future = executor.submit(
                    _run_strategy_compare_model_build_subprocess,
                    job,
                    active_processes=active_processes,
                    active_processes_lock=active_processes_lock,
                )
                futures[future] = job

        def render_training_progress(now: float) -> None:
            active_jobs = sorted(
                futures.values(), key=lambda item: int(item["source_index"])
            )
            lines = []
            for worker_index, job in enumerate(active_jobs, start=1):
                unit_text = render_training_unit_progress(
                    unit_id=str(job["dl_id"]),
                    source_index=int(job["source_index"]),
                    source_count=int(job["source_count"]),
                    elapsed_seconds=now - float(job["submitted_at"]),
                    pit_progress=read_trainer_pit_progress(Path(str(job["log_path"]))),
                    epoch_progress=read_trainer_epoch_progress(Path(str(job["log_path"]))),
                )
                lines.append(
                    paint("[TRAIN]", "cyan", enabled=color_enabled, bold=True)
                    + f" worker {worker_index}/{worker_count} | seed {int(job['seed'])} | "
                    + unit_text
                )
            progress.update(lines)

        try:
            submit_jobs()
            render_training_progress(time.perf_counter())
            while jobs or futures:
                finished = [future for future in futures if future.done()]
                for future in finished:
                    job = futures.pop(future)
                    result = future.result()
                    # Refill the freed GPU slot before rendering the completed unit report.
                    submit_jobs()
                    is_pit = (
                        str(job["source"].score_source)
                        == SCORE_SOURCE_SELECTION_POINT_IN_TIME
                    )
                    report_args = list(result["report_args"])
                    if is_pit and "--point-in-time-dir-override" not in report_args:
                        report_args.extend([
                            "--point-in-time-dir-override",
                            str(Path(str(job["model_dir"])).resolve()),
                        ])
                    _emit_breakout_quality_simple_report(
                        "build-point-in-time-scores" if is_pit else "train-continuous-ranker",
                        report_args,
                        returncode=0,
                        elapsed_sec=float(result["elapsed_sec"]),
                    )
                    if is_pit:
                        audit_args = [
                            "--filter-id", str(job["source"].filter_id),
                            "--model-architecture", str(job["source"].model_architecture),
                            "--experiment-profile", str(job["source"].experiment_profile),
                            "--point-in-time-dir-override",
                            str(Path(str(job["model_dir"])).resolve()),
                        ]
                        _emit_breakout_quality_simple_report(
                            "audit-point-in-time-scores",
                            audit_args,
                            returncode=0,
                            elapsed_sec=float(result.get("audit_elapsed_sec", 0.0) or 0.0),
                        )
                    progress.print_line(
                        styled_workflow_status("[TRAIN DONE]")
                        + f" seed={job['seed']} | {job['dl_id']}"
                        + f" | elapsed={render_elapsed(float(result['elapsed_sec']))}"
                    )
                submit_jobs()
                now = time.perf_counter()
                if futures and (finished or now >= next_progress):
                    render_training_progress(now)
                    next_progress = now + progress_interval
                if jobs or futures:
                    time.sleep(0.25)
            progress.finish()
        except BaseException:
            terminate_registered_training_processes(
                active_processes, active_processes_lock
            )
            progress.finish()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)


    print(styled_workflow_status("[READY]") + " 策略比較所需模型工件已就緒")
    return 0


def _interactive_rolling_timing_mode(program_name: str, settings) -> int:
    while True:
        print("\n=== Timing Mode｜Rolling 訓練前後比較 ===")
        print(render_menu_item(1, "執行／更新 A/B 比較", default=True))
        print(render_menu_item(2, "查看 Timing 設定與基準狀態"))
        print(render_menu_item(3, "重新建立改善前 Baseline"))
        print(render_menu_item(4, "查看最新 Timing 報表"))
        print(render_menu_item(0, "返回"))
        try:
            raw_choice = input("👉 請選擇：").strip().lower()
        except EOFError:
            return 0
        choice = "1" if raw_choice == "" else raw_choice
        if choice in {"0", "q", "quit", "exit"}:
            return 0
        try:
            if choice == "1":
                upstream_plan = _collect_continuous_research_input_plan(settings)
                _render_continuous_research_input_plan(settings, upstream_plan)
                if upstream_plan.blocked:
                    print("目前存在不可由canonical producer確定性補建的前置工件；本次不執行。")
                    continue
                if not _prompt_bool("確認執行Timing A/B（含必要自動前置）", True):
                    continue
                code = _prepare_continuous_research_inputs(program_name, settings)
                if code != 0:
                    return int(code)
                return int(
                    _run_command(
                        "timing-rolling-training",
                        ["run"],
                        program_name=program_name,
                    )
                )
            if choice == "2":
                _run_command(
                    "timing-rolling-training",
                    ["status"],
                    program_name=program_name,
                )
                continue
            if choice == "3":
                try:
                    confirm = input(
                        "👉 這會刪除目前Timing baseline；輸入 RESET 確認，其他返回："
                    ).strip()
                except EOFError:
                    return 0
                if confirm != "RESET":
                    print("未重設 Timing baseline。")
                    continue
                _run_command(
                    "timing-rolling-training",
                    ["reset-baseline"],
                    program_name=program_name,
                )
                continue
            if choice == "4":
                _run_command(
                    "timing-rolling-training",
                    ["report"],
                    program_name=program_name,
                )
                continue
            print("無效選項，請重新輸入。")
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"[錯誤] {type(exc).__name__}: {exc}")
        except KeyboardInterrupt:
            print("\nTiming操作已中止，返回Timing Mode選單。")

def _run_configured_continuous_ranker_model_gates(program_name: str) -> int:
    configured = tuple(BREAKOUT_QUALITY_CONDITIONAL_MFE_AB_MODEL_GATE_PROFILES)
    if len(configured) != 2:
        raise ValueError("設定中的 Continuous Ranker Model Gate目前必須剛好有兩個profile")
    rows = []
    for model_id, profile_name in configured:
        settings = get_breakout_quality_workflow_settings(experiment_profile=str(profile_name))
        rows.append((str(model_id), settings))
    print("\n" + render_title(BREAKOUT_QUALITY_CONTINUOUS_RANKER_COMPARISON_MENU_LABEL))
    print(render_table(
        ("Model", "Profile", "Architecture", "Seed"),
        [(model_id, settings.experiment_profile, settings.model_architecture, str(settings.seed)) for model_id, settings in rows],
        alignments=("left", "left", "left", "right"),
    ))
    if not _prompt_bool("確認依設定順序訓練兩個模型並產生各自Forward-OOS Model Gate報表", True):
        return 0
    for model_id, settings in rows:
        print("\n" + render_title(f"{model_id} Forward Model Gate"))
        upstream_plan = _collect_continuous_research_input_plan(settings)
        _render_continuous_research_input_plan(settings, upstream_plan)
        if upstream_plan.blocked:
            print(f"{model_id} 存在不可確定補建的前置工件；模型比較流程停止。")
            return 1
        code = _prepare_continuous_research_inputs(program_name, settings)
        if code != 0:
            return int(code)
        code = _run_command(
            "train-continuous-ranker",
            [
                "--filter-id", str(settings.filter_id),
                "--model-architecture", str(settings.model_architecture),
                "--experiment-profile", str(settings.experiment_profile),
                "--seed", str(int(settings.seed)),
            ],
            program_name=program_name,
        )
        if code != 0:
            return int(code)
    print("\n設定中的 Continuous Rankers Forward Model Gate完成；請先比較各模型報表，再解讀Strategy Compare conversion。")
    return 0


def _interactive_model_research(program_name: str) -> int:
    settings = get_breakout_quality_model_research_settings()
    if settings.is_binary_classification:
        return _interactive_binary_model_research(
            program_name,
            workflow_settings=settings,
        )
    if not settings.is_continuous_ranker:
        raise ValueError(
            f"不支援的 workflow training objective: {settings.training_objective!r}"
        )

    while True:
        print("\n=== Continuous DL 模型研究與驗證 ===")
        print(f"Active Profile：{settings.experiment_profile}")
        print(render_menu_item(1, "訓練目前模型 → Forward-OOS 標準模型 SOP 報表", default=True))
        print(render_menu_item(2, "Extending-Window Rolling"))
        print(render_menu_item(3, "Fixed-Window Rolling"))
        print(render_menu_item(4, "Target／模型比較"))
        print(render_menu_item(5, "Timing Mode｜Rolling 訓練前後比較  [工程]"))
        print(render_menu_item(0, "返回"))
        try:
            raw_choice = input("👉 請選擇：").strip().lower()
        except EOFError:
            print("\n輸入已結束。")
            return 0
        choice = "1" if raw_choice == "" else raw_choice
        if choice in {"0", "q", "quit", "exit"}:
            return 0
        if choice == "1":
            return _run_continuous_forward_model_gate(program_name, settings)
        if choice == "2":
            _run_continuous_rolling_mode_direct(
                program_name, settings, fixed_window=False
            )
            continue
        if choice == "3":
            _run_continuous_rolling_mode_direct(
                program_name, settings, fixed_window=True
            )
            continue
        if choice == "4":
            _interactive_target_model_comparison(program_name, settings)
            continue
        if choice == "5":
            timing = get_breakout_quality_rolling_timing_settings()
            timing_settings = get_breakout_quality_workflow_settings(
                experiment_profile=str(timing.experiment_profile)
            )
            return int(_interactive_rolling_timing_mode(program_name, timing_settings))
        print("無效選項，請重新輸入。")

def run_model_training_menu(program_name: str = "apps/research.py model") -> int:
    """Run the active Breakout Quality model-training menu.

    The project-level work type is selected by ``apps/research.py``; this function
    only owns the model-specific training/validation flow.
    """
    return _interactive_model_research(program_name)


def show_model_status() -> None:
    """Render current Breakout Quality model-research and artifact status."""
    _print_workflow_status(get_breakout_quality_model_research_settings())


def main(argv=None) -> int:
    raw_argv = list(sys.argv if argv is None else argv)
    program_name = resolve_cli_program_name(raw_argv, "apps/research.py model")
    args = raw_argv[1:]

    if not args:
        if is_interactive_console():
            return run_model_training_menu(program_name)
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
        return run_model_training_menu(program_name)

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
    "prepare_strategy_compare_artifacts",
    "run_model_training_menu",
    "show_model_status",
]


if __name__ == "__main__":
    run_cli_entrypoint(main)
