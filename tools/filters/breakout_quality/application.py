"""Breakout quality dataset、training、score export、report 與 evaluation 正式入口。"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.breakout_quality import (
    TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    SUPPORTED_BREAKOUT_QUALITY_CLASSIFICATION_EXPERIMENT_PROFILES,
    SUPPORTED_BREAKOUT_QUALITY_TIME_WEIGHT_MODES,
    get_breakout_quality_continuous_ranker_comparison_settings,
    get_breakout_quality_experiment_profile,
    get_continuous_ranker_research_spec,
)
from config.breakout_quality import (
    BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
    BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
    BREAKOUT_QUALITY_INCEPTION_TARGET_RECEPTIVE_FIELD_BARS,
    BREAKOUT_QUALITY_STABILITY_SCORE_END_DATE,
    BREAKOUT_QUALITY_STABILITY_SCORE_START_DATE,
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
from core.display_common import render_elapsed
from core.runtime_utils import (
    is_interactive_console,
    resolve_cli_program_name,
    run_cli_entrypoint,
)
from filters.breakout_quality.artifact_dependency_registry import (
    collect_model_upstream_readiness,
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
from tools.audit.catalog import get_domain_cli_commands

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
    "build-dataset": "tools.filters.breakout_quality.build_dataset",
    "train": "tools.filters.breakout_quality.train",
    "export-scores": "filters.breakout_quality.export_scores",
    "report": "tools.filters.breakout_quality.report",
    "evaluate": "tools.filters.breakout_quality.evaluate",
    "prepare-continuous-target": "tools.filters.breakout_quality.prepare_continuous_target",
    "train-continuous-ranker": "services.breakout_quality.ranker_cli",
    "compare-daily-targets": "services.breakout_quality.daily_target_comparison",
    "compare-continuous-rankers": "tools.filters.breakout_quality.compare_continuous_rankers",
    "build-point-in-time-scores": "tools.filters.breakout_quality.build_point_in_time_scores",
    "timing-rolling-training": "services.breakout_quality.rolling_timing",
    "build-binary-point-in-time-scores": (
        "tools.filters.breakout_quality.build_binary_point_in_time_scores"
    ),
    "build-trade-path-labels": (
        "tools.filters.breakout_quality.build_trade_path_labels"
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


def _safe_json_object(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


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
    return _safe_json_object(output_dir / CONTINUOUS_RANKER_REPORT_FILENAME)


def _render_continuous_ranker_simple_console(payload: dict) -> str:
    metrics = dict(payload.get("split_metrics") or {})
    if not metrics:
        return ""

    def split_row(name: str) -> tuple[str, ...]:
        row = dict(metrics.get(name) or {})
        return (
            name,
            f"{int(row.get('group_count', 0) or 0):,}",
            _fmt_simple_metric(row.get("mean_daily_spearman")),
            _fmt_simple_metric(row.get("global_spearman_vs_raw_target")),
            _fmt_simple_metric(row.get("pairwise_concordance"), percent=True),
            _fmt_simple_metric(row.get("top_score_decile_raw_target_mean")),
            _fmt_simple_metric(row.get("bottom_score_decile_raw_target_mean")),
        )

    daily_universal = bool(metrics.get("breakout_candidate_oos"))
    split_names = (
        ["validation", "oos", "breakout_candidate_oos"]
        if daily_universal
        else ["validation", "selection", "oos"]
    )
    lines = [
        render_section(
            "Daily Universal Forward-OOS 排序品質"
            if daily_universal
            else "完整 Selection 重訓後排序品質"
        ),
        render_table(
            ("Split", "Groups", "Daily rho", "Global rho", "Pair", "Top 10% Target", "Bottom 10% Target"),
            [split_row(name) for name in split_names],
            alignments=("left", "right", "right", "right", "right", "right", "right"),
        ),
    ]

    sample = dict((metrics.get("oos") or {}).get("top_k_quality") or {})
    if sample:
        top_k = int(sample.get("top_k", 0) or 0)
        boundary_width = int(sample.get("boundary_width", 0) or 0)
        competition_scope = "只看同日樣本數>K" if daily_universal else "只看候選數>K"

        def top_k_row(name: str) -> tuple[str, ...]:
            quality = dict((metrics.get(name) or {}).get("top_k_quality") or {})
            return (
                name,
                _fmt_simple_metric(quality.get("ndcg_at_k")),
                _fmt_simple_metric(quality.get("top_k_raw_target_mean")),
                _fmt_simple_metric(quality.get("top_k_raw_target_lift")),
                _fmt_simple_metric(quality.get("oracle_top_k_overlap"), percent=True),
                _fmt_simple_metric(quality.get("boundary_concordance"), percent=True),
                _fmt_simple_metric(quality.get("boundary_raw_target_gap")),
                f"{int(quality.get('competition_date_count', quality.get('top_k_date_count', 0)) or 0):,}",
            )

        lines.extend([
            render_section(f"Top-K / K-boundary（K={top_k}，邊界寬度={boundary_width}；{competition_scope}）"),
            render_table(
                ("Split", "NDCG@K", "Top-K Target", "Lift", "Oracle overlap", "Boundary", "Boundary gap", "競爭日"),
                [top_k_row(name) for name in split_names],
                alignments=("left", "right", "right", "right", "right", "right", "right", "right"),
            ),
        ])

    trade = dict(payload.get("trade_alignment") or {})
    if trade.get("available"):
        pass_trade = dict((trade.get("label_conditional") or {}).get("PASS") or {})
        coverage = trade.get("coverage_rate")
        lines.extend([
            render_section("Actual Round-trip R"),
            render_key_values((
                ("Matched trades", f"{int(trade.get('matched_trade_count', 0) or 0):,} / {int(trade.get('trade_count', 0) or 0):,}"),
                ("Coverage", _fmt_simple_metric(coverage, percent=True)),
                ("Overall Score↔R", _fmt_simple_metric(trade.get("spearman_model_score_vs_r_multiple"))),
                ("PASS Score↔R", _fmt_simple_metric(pass_trade.get("spearman_model_score_vs_r_multiple"))),
                ("Score Top/Bottom 10% R", f"{_fmt_simple_metric(trade.get('top_model_score_decile_average_r'))} / {_fmt_simple_metric(trade.get('bottom_model_score_decile_average_r'))}"),
            )),
        ])
    return "\n".join(line for line in lines if line)


def _render_continuous_ranker_simple_markdown(payload: dict) -> list[str]:
    metrics = dict(payload.get("split_metrics") or {})
    if not metrics:
        return []
    daily_universal = bool(metrics.get("breakout_candidate_oos"))
    lines = [
        "",
        (
            "## Daily Universal Forward-OOS 排序品質"
            if daily_universal
            else "## 完整 Selection 重訓後排序品質"
        ),
        "",
        "| Split | Groups | Daily rho | Global rho | Pair | Top 10% Target | Bottom 10% Target |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    split_names = (
        ["validation", "oos", "breakout_candidate_oos"]
        if daily_universal
        else ["validation", "selection", "oos"]
    )
    for name in split_names:
        row = dict(metrics.get(name) or {})
        lines.append(
            f"| {name} | {int(row.get('group_count', 0) or 0):,} "
            f"| {_fmt_simple_metric(row.get('mean_daily_spearman'))} "
            f"| {_fmt_simple_metric(row.get('global_spearman_vs_raw_target'))} "
            f"| {_fmt_simple_metric(row.get('pairwise_concordance'), percent=True)} "
            f"| {_fmt_simple_metric(row.get('top_score_decile_raw_target_mean'))} "
            f"| {_fmt_simple_metric(row.get('bottom_score_decile_raw_target_mean'))} |"
        )
    sample = dict((metrics.get("oos") or {}).get("top_k_quality") or {})
    if sample:
        top_k = int(sample.get("top_k", 0) or 0)
        boundary_width = int(sample.get("boundary_width", 0) or 0)
        competition_scope = "只看同日樣本數>K" if daily_universal else "只看候選數>K"
        lines.extend([
            "",
            f"## Top-K / K-boundary（K={top_k}，邊界寬度={boundary_width}；{competition_scope}）",
            "",
            "| Split | NDCG@K | Top-K Target | Lift | Oracle overlap | Boundary | Boundary gap | 競爭日 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for name in split_names:
            quality = dict((metrics.get(name) or {}).get("top_k_quality") or {})
            lines.append(
                f"| {name} | {_fmt_simple_metric(quality.get('ndcg_at_k'))} "
                f"| {_fmt_simple_metric(quality.get('top_k_raw_target_mean'))} "
                f"| {_fmt_simple_metric(quality.get('top_k_raw_target_lift'))} "
                f"| {_fmt_simple_metric(quality.get('oracle_top_k_overlap'), percent=True)} "
                f"| {_fmt_simple_metric(quality.get('boundary_concordance'), percent=True)} "
                f"| {_fmt_simple_metric(quality.get('boundary_raw_target_gap'))} "
                f"| {int(quality.get('competition_date_count', quality.get('top_k_date_count', 0)) or 0):,} |"
            )
    trade = dict(payload.get("trade_alignment") or {})
    if trade.get("available"):
        pass_trade = dict((trade.get("label_conditional") or {}).get("PASS") or {})
        lines.extend([
            "",
            "## Actual Round-trip R",
            "",
            f"- Matched trades：`{int(trade.get('matched_trade_count', 0) or 0):,} / {int(trade.get('trade_count', 0) or 0):,}`",
            f"- Coverage：`{_fmt_simple_metric(trade.get('coverage_rate'), percent=True)}`",
            f"- Overall Score↔R：`{_fmt_simple_metric(trade.get('spearman_model_score_vs_r_multiple'))}`",
            f"- PASS Score↔R：`{_fmt_simple_metric(pass_trade.get('spearman_model_score_vs_r_multiple'))}`",
        ])
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
            payload = _safe_json_object(report_json)
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
        rows.extend(
            [
                ("Selected epoch", training.get("selected_epoch")),
                *(
                    [
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
                ("重訓後原 Validation rho", _fmt_simple_metric((metrics.get("validation") or {}).get("mean_daily_spearman"))),
                ("Forward OOS rho", _fmt_simple_metric((metrics.get("oos") or {}).get("mean_daily_spearman"))),
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
        payload = _safe_json_object(output_dir / "continuous_ranker_comparison.json")
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
        audit_json = (
            pit_override / "selection_point_in_time_audit.json"
            if pit_override is not None
            else resolve_selection_point_in_time_audit_json_path(
                PROJECT_ROOT, filter_id, architecture, profile
            )
        )
        payload = _safe_json_object(audit_json)
        coverage = dict(payload.get("score_coverage") or {})
        if command == "build-point-in-time-scores" and not coverage:
            pit_manifest = (
                pit_override / "selection_point_in_time_manifest.json"
                if pit_override is not None
                else resolve_selection_point_in_time_manifest_path(
                    PROJECT_ROOT, filter_id, architecture, profile
                )
            )
            manifest_payload = _safe_json_object(pit_manifest)
            coverage = dict(manifest_payload.get("coverage") or {})
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
                ("Score coverage", _fmt_simple_metric(coverage.get("coverage_rate"), percent=True)),
                ("Scored groups", coverage.get("scored_group_count")),
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
        payload = _safe_json_object(report_json)
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
            baseline = _safe_json_object(timing_paths["baseline"])
            candidate_payload = _safe_json_object(timing_paths["candidate"])
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
            manifest = _safe_json_object(target_dir / TARGET_MANIFEST_FILENAME)
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
        comparison_payload = _safe_json_object(
            output_dir / "continuous_ranker_comparison.json"
        )
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

    print("\n" + render_title("Breakout Quality 簡易報表"))
    print(render_key_values(rows))
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
        "# Breakout Quality 簡易報表",
        "",
        f"- Generated at UTC：`{datetime.now(timezone.utc).isoformat()}`",
    ]
    for label, value in rows:
        markdown_lines.append(f"- **{label}**：{value}")
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
    model_spec = get_active_model_spec(BREAKOUT_QUALITY_MODEL_ARCHITECTURE)
    experiment = get_breakout_quality_experiment_profile(
        train_settings.experiment_profile
    )
    schedule_parameters = experiment.lr_schedule_parameters()
    augmentation_parameters = experiment.augmentation_parameters()
    if str(model_spec.family) == "inception_time":
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


def _run_binary_post_train_validation(
    request: argparse.Namespace,
    *,
    workflow_settings,
    program_name: str,
    step_start: int = 1,
    total_steps: int = 1,
) -> int:
    if not workflow_settings.is_binary_classification:
        return 0
    print("\n=== Binary DL Filter 模型工件更新 ===")
    print(
        f"[{int(step_start)}/{int(total_steps)}] "
        "匯出正式 forward-OOS runtime scores"
    )
    rc = _run_command(
        "export-scores",
        _build_export_score_argv(request, scope="forward_oos"),
        program_name=program_name,
    )
    if rc == 0:
        print(
            "\n模型工件更新完成。策略績效比較請另開 "
            "python apps/research.py compare。"
        )
    return int(rc)

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
    print(
        "\n即將執行：Full dataset（全部股票）→ train → export research scores "
        "→ OOS 簡易模型報表 → export forward-OOS scores"
    )
    if evaluate_oos:
        print("模型報表將顯示 OOS 最終泛化評估；本入口不執行策略績效比較。")
    if not _prompt_bool("確認開始", True):
        print("已取消。")
        return 0
    rc = _run_workflow(request, program_name=program_name)
    if rc != 0:
        return int(rc)
    if workflow_settings is None:
        workflow_settings = get_breakout_quality_workflow_settings()
    return _run_binary_post_train_validation(
        request,
        workflow_settings=workflow_settings,
        program_name=program_name,
    )


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
    from tools.audit.breakout_quality.regime import (
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


def _interactive_existing_binary_validation(
    program_name: str,
    *,
    workflow_settings,
) -> int:
    filter_id = normalize_filter_id(workflow_settings.filter_id)
    request = _policy_train_settings(filter_id)
    request.experiment_profile = str(workflow_settings.experiment_profile)
    request.seed = int(workflow_settings.seed)
    _print_policy_defaults(filter_id, request)
    print(
        "\n即將使用既有模型：更新 research scores → OOS 簡易模型報表 "
        "→ export forward-OOS scores"
    )
    if not _prompt_bool("確認開始", True):
        print("已取消。")
        return 0
    print("\n[1/3] 由既有模型更新 research scores")
    rc = _run_command(
        "export-scores",
        _build_export_score_argv(request, scope="research"),
        program_name=program_name,
    )
    if rc != 0:
        return int(rc)
    print("\n[2/3] 顯示 OOS 簡易模型報表")
    rc = _run_command(
        "report",
        [
            "--filter-id",
            filter_id,
            "--experiment-profile",
            str(workflow_settings.experiment_profile),
            "--include-oos",
        ],
        program_name=program_name,
    )
    if rc != 0:
        return int(rc)
    return _run_binary_post_train_validation(
        request,
        workflow_settings=workflow_settings,
        program_name=program_name,
        step_start=3,
        total_steps=3,
    )


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


def _prepare_continuous_research_inputs(
    program_name: str,
    settings,
    *,
    dataset_profile: str | None = None,
    max_tickers: int | None = None,
) -> int:
    color_enabled = console_color_enabled()
    resolved_dataset_profile = str(
        INTERACTIVE_DATASET_PROFILE if dataset_profile is None else dataset_profile
    )
    resolved_max_tickers = int(
        INTERACTIVE_MAX_TICKERS if max_tickers is None else max_tickers
    )
    refresh_mode, refresh_reasons, dataset_step = _dataset_refresh_step(
        settings.filter_id,
        resolved_dataset_profile,
        max_tickers=resolved_max_tickers,
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
            return int(code)
    profile = get_breakout_quality_experiment_profile(settings.experiment_profile)
    if profile.training_sample_scope == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS:
        # Daily-universal targets are derived directly from canonical OHLCV and do not
        # create an event-group Continuous Target artifact.  MR-13I/J additionally need
        # historical-effective Min ROOS risk calibration; fail before GPU training when
        # those read-only upstream parameter artifacts are unavailable.
        if str(profile.continuous_target_id or "") == DAILY_RISK_NORMALIZED_NET_OPPORTUNITY_TARGET_ID:
            try:
                schedule = load_min_roos_risk_schedule(PROJECT_ROOT)
            except (FileNotFoundError, OSError, ValueError, KeyError, TypeError) as exc:
                print(
                    paint("[Risk params] BLOCKED", "red", enabled=color_enabled, bold=True)
                    + f"｜{exc}"
                )
                print("請先由Strategy Compare參數工作流準備Selection/Forward Min ROOS正式risk-param工件；模型流程不得自行訓練或補值。")
                return 2
            print(
                paint("[Risk params] READY", "green", enabled=color_enabled, bold=True)
                + "｜historical-effective Min ROOS atr_len / atr_times_init"
                + f"｜periods={len(schedule)}"
            )
        return 0
    with _compact_console_scope():
        return int(
            _run_command(
                "prepare-continuous-target",
                [
                    "--filter-id", settings.filter_id,
                    "--target-id", str(settings.continuous_target_id),
                ],
                program_name=program_name,
            )
        )


def _interactive_continuous_full_train(program_name: str, settings) -> int:
    _print_workflow_status(settings)
    profile = get_breakout_quality_experiment_profile(settings.experiment_profile)
    if profile.training_sample_scope == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS:
        print(
            "\n即將執行：確認canonical Dataset來源 → 建立daily stock-day index／target "
            "→ 以Selection內Inner Validation選epoch → 完整Selection重訓 "
            "→ checkpoint後評估全市場與breakout-candidate 2021+ Pre-Test OOS。"
        )
        print("Daily feature採lazy materialization；不建立expanded 300×10 daily feature artifact。")
    else:
        print(
            "\n即將執行：確認Dataset／Continuous Target → 以Selection內Inner Validation選epoch "
            "→ 完整Selection重訓 → checkpoint後才評估forward OOS。"
        )
    print(
        f"Profile={profile.name}｜Objective={profile.training_objective}｜"
        f"Loss={profile.loss_name}｜Epoch metric={profile.epoch_selection_metric}"
    )
    print("此為Pre-Test研究Gate：不取代正式Rolling evidence；模型完成後可直接執行Pre-Test策略比較。")
    if not _prompt_bool("確認開始", True):
        print("已取消。")
        return 0
    code = _prepare_continuous_research_inputs(program_name, settings)
    if code != 0:
        return int(code)
    argv = [
        "--filter-id", settings.filter_id,
        "--model-architecture", settings.model_architecture,
        "--experiment-profile", settings.experiment_profile,
        "--seed", str(settings.seed),
    ]
    return int(
        _run_command(
            "train-continuous-ranker",
            argv,
            program_name=program_name,
        )
    )


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
    if bool(mode.single_score_block):
        end_label = "最新" if str(mode.score_end_date).strip().lower() == "auto" else str(mode.score_end_date)
        return f"{mode.label} | {mode.score_start_date}→{end_label}"
    return f"{mode.label} | {int(mode.fold_months)}M"


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
            purpose = (
                "固定calendar train history的歷史learnability診斷"
                if fixed_window
                else "固定2020年底information cutoff的OOS Gate"
            )
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
            if not _prompt_bool(f"確認執行{mode.label}", True):
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


def _continuous_pit_gate_batch_for_active(settings):
    gate = get_breakout_quality_continuous_ranker_pit_gate_settings()
    profiles = tuple(profile for _model_id, profile in gate.model_profiles)
    return gate if settings.experiment_profile in profiles else None


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


def _fmt_pit_metric(value, *, percent: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(number):
        return "-"
    return f"{number * 100:.2f}%" if percent else f"{number:.4f}"


def _render_continuous_pit_gate_comparison(model_profiles) -> None:
    rows = []
    for model_id, profile_name in model_profiles:
        settings = get_breakout_quality_workflow_settings(
            experiment_profile=profile_name
        )
        audit_path = resolve_selection_point_in_time_audit_json_path(
            PROJECT_ROOT,
            settings.filter_id,
            settings.model_architecture,
            settings.experiment_profile,
        )
        if not audit_path.is_file():
            rows.append((model_id, "MISSING", "-", "-", "-", "-", "-", "-", "-"))
            continue
        payload = json.loads(audit_path.read_text(encoding="utf-8"))
        gate = derive_point_in_time_model_validation_gate(payload)
        metrics = dict(payload.get("metrics") or {})
        primary_scope = str(gate.get("primary_metric_scope") or "all_valid_target")
        primary = dict(metrics.get(primary_scope) or {})
        breakout = dict(metrics.get("breakout_candidate_target") or {})
        topk = dict(primary.get("top_k_quality") or {})
        breakout_topk = dict(breakout.get("top_k_quality") or {})
        rows.append(
            (
                model_id,
                str(gate.get("status") or "-"),
                _fmt_pit_metric(primary.get("mean_daily_spearman")),
                _fmt_pit_metric(primary.get("global_spearman")),
                _fmt_pit_metric(primary.get("pairwise_concordance"), percent=True),
                _fmt_pit_metric(topk.get("top_k_raw_target_lift")),
                _fmt_pit_metric(topk.get("boundary_raw_target_gap")),
                _fmt_pit_metric(breakout.get("mean_daily_spearman")),
                _fmt_pit_metric(breakout_topk.get("top_k_raw_target_lift")),
            )
        )
    print("\n" + render_title("Selection PIT Model Gate Comparison"))
    print(
        render_table(
            (
                "Model",
                "Gate",
                "All Daily rho",
                "All Global rho",
                "All Pair",
                "All Top-K Lift",
                "All Boundary gap",
                "Breakout Daily rho",
                "Breakout Top-K Lift",
            ),
            rows,
        )
    )
    print("比較表只並列canonical PIT audit原始指標；不建立加權總分、不挑seed。")


def _interactive_continuous_pit_gate_batch(program_name: str, gate) -> int:
    print("\n" + render_title("Selection PIT Model Gate Batch"))
    reference = get_breakout_quality_workflow_settings(
        experiment_profile=gate.model_profiles[0][1]
    )
    print(
        render_key_values(
            (
                ("Models", ", ".join(model_id for model_id, _ in gate.model_profiles)),
                ("Seed", gate.seed),
                (
                    "PIT period",
                    f"{'auto（最早合法）' if str(gate.point_in_time_score_start_date).lower() == 'auto' else gate.point_in_time_score_start_date} ～ "
                    f"{gate.point_in_time_score_end_date or 'Selection end'}",
                ),
                (
                    "Fold／Validation",
                    f"{gate.point_in_time_fold_months}／"
                    f"{gate.point_in_time_inner_validation_months} months",
                ),
                ("Target", reference.continuous_target_id),
                ("Sample scope", reference.training_sample_scope),
                ("Architecture", reference.model_architecture),
            )
        )
    )
    print("三個profile各自獨立訓練／PIT工件；共用同一Dataset、Target、Seed、period與fold contract。")
    if not _prompt_bool("確認依序執行全部PIT Model Gate", True):
        return 0
    total = len(gate.model_profiles)
    for index, (model_id, profile_name) in enumerate(gate.model_profiles, start=1):
        print(f"\n[{index}/{total}] {model_id}")
        code = _run_continuous_pit_profile(
            program_name,
            model_id=model_id,
            profile_name=profile_name,
        )
        if code != 0:
            print(f"[FAILED] {model_id} PIT Model Gate停止；修正後可由同一入口resume。")
            return int(code)
    _render_continuous_pit_gate_comparison(gate.model_profiles)
    return 0


def _prepare_strategy_compare_model_upstream(
    program_name: str,
    *,
    workflow,
    dataset_profile: str,
) -> int:
    """Prepare canonical Dataset/Target only when the shared registry says they are stale."""

    readiness = collect_model_upstream_readiness(
        PROJECT_ROOT,
        filter_id=str(workflow.filter_id),
        model_architecture=str(workflow.model_architecture),
        experiment_profile=str(workflow.experiment_profile),
        dataset=str(dataset_profile),
        max_tickers=0,
    )
    color_enabled = console_color_enabled()
    for item in readiness:
        status_text = "REUSE" if item.ready else "BUILD/REBUILD"
        print(
            "  Upstream "
            + paint(status_text, "green" if item.ready else "yellow", enabled=color_enabled, bold=True)
            + f" | {item.artifact_type} | {item.status}"
            + f" | {project_relative_display_path(item.path, project_root=PROJECT_ROOT)}"
        )
    if all(item.ready for item in readiness):
        return 0
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


def prepare_strategy_compare_model_artifacts(
    *,
    program_name: str,
    profile_ids: tuple[str, ...],
) -> int:
    """Canonical model-work prerequisite handler for Strategy Compare orchestration."""

    return int(
        _prepare_strategy_compare_model_artifacts(
            program_name, profile_ids=tuple(str(value) for value in profile_ids)
        )
    )


def _prepare_strategy_compare_model_artifacts(
    program_name: str, *, profile_ids: tuple[str, ...] | None = None
) -> int:
    """Prepare model artifacts for the selected current Rolling Test mode(s).

    OOS/Rolling share the canonical trainer and model identities but use different
    score-block semantics and aggregate paths. Strategy Compare may invoke this canonical
    model-work handler as an orchestrator, but it does not own or duplicate training logic.
    """

    comparisons, sources = _strategy_compare_required_model_sources(profile_ids)
    if not sources:
        print("目前策略比較設定沒有需要準備的模型工件。")
        return 0
    datasets = {comparison.dataset for comparison in comparisons}
    if len(datasets) != 1:
        raise ValueError(f"Strategy Compare profiles dataset不一致: {sorted(datasets)}")
    comparison = comparisons[0]

    color_enabled = console_color_enabled()
    print(
        paint("策略比較模型工件準備", "cyan", enabled=color_enabled, bold=True)
        + f" | profiles={len(comparisons)} | sources={len(sources)} | dataset={comparison.dataset}"
    )
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
        print(
            paint(
                f"[{source_index}/{len(sources)}]",
                "cyan",
                enabled=color_enabled,
                bold=True,
            )
            + f" {dl_id} | profile={source.experiment_profile}"
        )

        if source.score_source == SCORE_SOURCE_CONTINUOUS_RANKER_OOS:
            # Legacy-only compatibility path; current Fast/Overnight profiles use PIT.
            try:
                load_continuous_ranker_oos_contract(
                    PROJECT_ROOT,
                    str(source.filter_id),
                    str(source.model_architecture),
                    str(source.experiment_profile),
                )
            except (OSError, ValueError, KeyError, TypeError):
                code = _prepare_strategy_compare_model_upstream(
                    program_name,
                    workflow=workflow,
                    dataset_profile=str(comparison.dataset),
                )
                if code != 0:
                    return int(code)
                code = _run_command(
                    "train-continuous-ranker",
                    [
                        "--filter-id", str(source.filter_id),
                        "--model-architecture", str(source.model_architecture),
                        "--experiment-profile", str(source.experiment_profile),
                        "--seed", str(workflow.seed),
                    ],
                    program_name=program_name,
                )
                if code != 0:
                    return int(code)
            else:
                print("  Frozen compatibility policy：REUSE 已完成且identity一致的模型／report／scores")
            continue

        if source.score_source != SCORE_SOURCE_SELECTION_POINT_IN_TIME:
            raise ValueError(
                f"不支援的Strategy Compare model score source: {source.score_source!r}"
            )
        if not workflow.rolling_authorized:
            raise ValueError(
                f"設定的Rolling PIT source未授權current Rolling workflow: {dl_id}"
            )

        code = _prepare_strategy_compare_model_upstream(
            program_name,
            workflow=workflow,
            dataset_profile=str(comparison.dataset),
        )
        if code != 0:
            return int(code)

        fold_months = int(
            source.point_in_time_fold_months
            if source.point_in_time_fold_months is not None
            else workflow.point_in_time_fold_months
        )
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
        score_start_date = (
            str(source.point_in_time_score_start_date)
            if source.point_in_time_score_start_date not in (None, "")
            else str(workflow.point_in_time_score_start_date)
        )
        score_end_date = (
            source.point_in_time_score_end_date
            if source.point_in_time_score_end_date not in (None, "")
            else workflow.point_in_time_score_end_date
        )
        build_args = [
            "--filter-id", str(source.filter_id),
            "--model-architecture", str(source.model_architecture),
            "--experiment-profile", str(source.experiment_profile),
            "--score-start-date", score_start_date,
            "--fold-months", str(fold_months),
            "--inner-validation-months", str(workflow.point_in_time_inner_validation_months),
            "--seed", str(workflow.seed),
            "--resume",
        ]
        if source.point_in_time_fold_anchor_date not in (None, ""):
            build_args.extend(
                ["--fold-anchor-date", str(source.point_in_time_fold_anchor_date)]
            )
        if source.point_in_time_single_score_block:
            build_args.append("--single-score-block")
        if score_end_date:
            build_args.extend(["--score-end-date", str(score_end_date)])
        if pit_dir_override is not None:
            build_args.extend(["--point-in-time-dir-override", str(pit_dir_override)])
        print(
            "  PIT policy："
            + (
                f"single OOS block；score={score_start_date}→{('最新' if str(score_end_date).lower() == 'auto' else score_end_date)}"
                if source.point_in_time_single_score_block
                else f"{fold_months}M cadence"
            )
            + (
                ""
                if source.point_in_time_fold_anchor_date in (None, "")
                else f"；anchor={source.point_in_time_fold_anchor_date}"
            )
            + "；resume existing folds；缺少／不相容fold才補訓"
        )
        from tools.filters.breakout_quality.build_point_in_time_scores import (
            build_selection_point_in_time_scores,
        )
        from services.breakout_quality.point_in_time_audit import (
            audit_selection_point_in_time_scores,
        )

        with _compact_console_scope():
            stage_started = time.perf_counter()
            code = build_selection_point_in_time_scores(
                filter_id=str(source.filter_id),
                model_architecture=str(source.model_architecture),
                experiment_profile=str(source.experiment_profile),
                score_start_date=score_start_date,
                score_end_date=(None if score_end_date in (None, "") else str(score_end_date)),
                fold_months=fold_months,
                fold_anchor_date=(
                    None
                    if source.point_in_time_fold_anchor_date in (None, "")
                    else str(source.point_in_time_fold_anchor_date)
                ),
                inner_validation_months=int(workflow.point_in_time_inner_validation_months),
                seed=int(workflow.seed),
                resume=True,
                point_in_time_dir_override=(
                    None if pit_dir_override is None else str(pit_dir_override)
                ),
                single_score_block=bool(source.point_in_time_single_score_block),
            )
            if code != 0:
                return int(code)
            _emit_breakout_quality_simple_report(
                "build-point-in-time-scores",
                build_args,
                returncode=0,
                elapsed_sec=time.perf_counter() - stage_started,
            )
            audit_args = [
                "--filter-id", str(source.filter_id),
                "--model-architecture", str(source.model_architecture),
                "--experiment-profile", str(source.experiment_profile),
            ]
            if pit_dir_override is not None:
                audit_args.extend(["--point-in-time-dir-override", str(pit_dir_override)])
            stage_started = time.perf_counter()
            code = audit_selection_point_in_time_scores(
                filter_id=str(source.filter_id),
                model_architecture=str(source.model_architecture),
                experiment_profile=str(source.experiment_profile),
                point_in_time_dir_override=(
                    None if pit_dir_override is None else str(pit_dir_override)
                ),
            )
            if code != 0:
                return int(code)
            _emit_breakout_quality_simple_report(
                "audit-point-in-time-scores",
                audit_args,
                returncode=0,
                elapsed_sec=time.perf_counter() - stage_started,
            )
    print(
        paint("策略比較所需模型工件已就緒", "green", enabled=color_enabled, bold=True)
    )
    return 0


def _interactive_rolling_timing_mode(program_name: str) -> int:
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
        research_spec = get_continuous_ranker_research_spec(settings.experiment_profile)
        print("\n=== Continuous DL 模型研究與驗證 ===")
        print(f"Active Profile：{settings.experiment_profile}")
        print(render_menu_item(1, "Extending-Window Test", default=True))
        print(render_menu_item(2, "Fixed-Window Stability Test"))
        print(render_menu_item(3, "查看目前Workflow與工件狀態"))
        print(render_menu_item(4, "比較目前 Target 與 reference Target"))
        print(render_menu_item(5, "Timing Mode｜Rolling 訓練前後比較"))
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
            return _interactive_continuous_pit_validation(program_name, settings)
        if choice == "2":
            return _interactive_continuous_stability_validation(program_name, settings)
        if choice == "3":
            _print_workflow_status(settings)
            continue
        if choice == "4":
            if not research_spec.reference_profile_name:
                print("目前Active Profile未設定reference Target；此項不可執行。")
                continue
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
        if choice == "5":
            return int(_interactive_rolling_timing_mode(program_name))
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
    "prepare_strategy_compare_model_artifacts",
    "run_model_training_menu",
    "show_model_status",
]


if __name__ == "__main__":
    run_cli_entrypoint(main)
