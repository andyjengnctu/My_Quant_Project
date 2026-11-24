from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
import ast
import io
import itertools
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd


def missing_teacher_artifact_fixture(runtime):
    """Planning-only fixture: keep registered dependencies, supply missing state.

    Teacher inspection/build/PIT contracts have their own behavioral cases.
    Score-reuse and plan-rendering cases must not repeatedly inspect every
    physical Optimizer publication simply to represent an absent teacher.
    The real seed/dependency planner still executes, including deduplication.
    """
    resolve = runtime._trade_path_teacher_requirements

    def inspect_missing(*, project_root, seed):
        return {"action": "BUILD", "reason": f"isolated missing teacher: seed={seed}"}

    def forbid_build(**_kwargs):
        raise AssertionError("a read-only planning fixture must not build a teacher")

    def requirements(profile):
        return tuple(replace(item, inspect=inspect_missing, execute=forbid_build)
                     for item in resolve(profile))

    return patch.object(runtime, "_trade_path_teacher_requirements", side_effect=requirements)



def _source_has_render_menu_item_call(
    source: str,
    *,
    index: int,
    label: str,
    default: bool = False,
) -> bool:
    """Return whether source calls the shared menu renderer with this semantic row."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Name) or func.id != "render_menu_item":
            continue
        if len(node.args) < 2:
            continue
        try:
            actual_index = ast.literal_eval(node.args[0])
            actual_label = ast.literal_eval(node.args[1])
        except (ValueError, TypeError):
            continue
        if actual_index != int(index) or actual_label != str(label):
            continue
        actual_default = False
        for keyword in node.keywords:
            if keyword.arg != "default":
                continue
            try:
                actual_default = bool(ast.literal_eval(keyword.value))
            except (ValueError, TypeError):
                actual_default = False
            break
        if actual_default == bool(default):
            return True
    return False




from core.active_param_ensemble import build_static_active_param_ensemble_payload
from core.params_io import params_to_json_dict

from core.breakout_quality_registry import (
    UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE,
    STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE,
    STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE,
    STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE,
    STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_LISTWISE_PROFILE,
    DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE,
    DAILY_UNIVERSAL_NO_TIME_FULL_LIST_NDCG_PAIRWISE_PROFILE,
    SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES,
    get_continuous_ranker_research_spec,
    SUPPORTED_BREAKOUT_QUALITY_CLASSIFICATION_EXPERIMENT_PROFILES,
    TRAINING_LABEL_SCOPE_PASS_ONLY,
    TRAINING_LABEL_SCOPE_ALL,
    TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS,
    TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    build_breakout_quality_pretraining_profile_payload,
    get_breakout_quality_experiment_profile,
    get_breakout_quality_pretraining_profile,
)
from core.breakout_quality_runtime import (
    CONTINUOUS_RANKER_PAIRWISE_REDUCTION_EQUAL_PAIR,
    CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
    TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION,
    TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
)
from config.breakout_quality import (
    BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD,
    BREAKOUT_QUALITY_TIME_WEIGHT_MODE,
    BREAKOUT_QUALITY_PRETRAINING_PROFILE,
    BREAKOUT_QUALITY_TORCH_DEVICE,
    BREAKOUT_QUALITY_USE_MIXED_PRECISION,
    BREAKOUT_QUALITY_MIXED_PRECISION_DTYPE,
    BREAKOUT_QUALITY_DETERMINISTIC_ALGORITHMS,
    BREAKOUT_QUALITY_ALLOW_TF32,
    BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
    BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
)
from filters.breakout_quality.artifacts import build_file_manifest
from filters.breakout_quality.continuous_target import (
    DAILY_FULL_HORIZON_OPPORTUNITY_TARGET_ID,
    DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
    STRATEGY_ALIGNED_TARGET_ID,
    STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
    StrategyAlignedContinuousTargetSpec,
    build_daily_full_horizon_opportunity_contract,
    build_daily_full_horizon_pure_mfe_contract,
    build_strategy_aligned_group_targets,
    load_validated_continuous_target_arrays,
    resolve_continuous_target_dir,
    daily_full_horizon_opportunity_target_from_cached_path,
    daily_full_horizon_pure_mfe_target_from_cached_path,
    daily_opportunity_no_time_target_from_cached_path,
    strategy_aligned_target_from_cached_path,
    TARGET_MANIFEST_FILENAME,
    TARGET_RAW_FILENAME,
    TARGET_VALID_MASK_FILENAME,
)
from filters.breakout_quality.daily_ranker_data import (
    compute_daily_opportunity_target_batch,
)
from filters.breakout_quality.contract import (
    DEFAULT_LABEL_POLICY,
    DEFAULT_MODEL_ARCHITECTURE,
    LABEL_INVALID,
    LABEL_PASS,
    LABEL_REJECT,
    SELECTION_ROLE_EMBARGO,
    SELECTION_ROLE_INVALID,
    SELECTION_ROLE_INNER_EMBARGO,
    SELECTION_ROLE_NOT_APPLICABLE,
    SELECTION_ROLE_TRAIN,
    SELECTION_ROLE_VALIDATION,
    OUTER_SPLIT_SELECTION,
    SCORE_COLUMN,
    SCORE_TABLE_REQUIRED_COLUMNS,
)
from filters.breakout_quality.models import (
    ACTIVE_MODEL_ARCHITECTURES,
    build_active_model,
    get_active_model_spec,
    LEGACY_MODEL_ARCHITECTURES,
    build_model as build_breakout_quality_model,
    count_trainable_parameters,
    get_model_spec,
)
from filters.breakout_quality.paths import (
    resolve_filter_artifact_paths,
    resolve_filter_model_output_dir,
    resolve_filter_research_score_path,
    resolve_filter_report_json_path,
    resolve_filter_report_markdown_path,
)
from core.signal_utils import generate_signals
from core.buy_sort import (
    BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_ADJUSTED,
    BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_BUCKET,
    BUY_LIMIT_OVERAGE_SORT_METHOD,
    calc_projected_capital_metrics,
    sort_candidate_rows,
)
from core.breakout_reentry import (
    create_breakout_reentry_signal_state,
    create_breakout_reentry_watch_state,
)
from core.extended_signals import resolve_breakout_quality_rank
from core.portfolio_candidates import _resolve_candidate_quality_ranking
from core.portfolio_ensemble import _aggregate_ensemble_candidate_rows
from core.portfolio_exits import _iter_reentry_watch_targets
from core.strategy_params import V16StrategyParams
from strategies.breakout.search_space import BREAKOUT_OPTIMIZER_SEARCH_SPACE
from services.breakout_quality import export_scores as breakout_quality_export_scores
from services.breakout_quality import train as breakout_quality_train
from services.breakout_quality.report import (
    build_report_payload,
    render_console_summary,
    render_markdown_report,
)
from services.research.strategy_compare_engine import (
    COMPARISON_MODE_HARD_FILTER,
    COMPARISON_MODE_SCORE_RANKING,
    OPTIONAL_ENTRY_FILTER_FIELDS,
    OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
    PARAM_POLICY_BASE_FINALIST_BEST,
    PARAM_POLICY_BASE_FINALISTS_AGREE,
    _assert_controlled_ensemble_pair,
    _assert_controlled_param_pair,
    _build_controlled_param_source_pair,
    _load_param_source,
    _comparison_labels,
    _comparison_output_dir_name,
    _first_existing_comparison_dir,
    _normalize_yearly_completeness,
    _resolve_comparison_period,
    _resolve_params_path,
    _validate_requested_param_policy,
    _to_json_native,
)
from filters.breakout_quality.strategy_compare_reporting import capacity_summary
from filters.breakout_quality.trade_attribution import build_trade_attribution
from services.breakout_quality.train_continuous_ranker import (
    _scope_group_ids as continuous_ranker_scope_group_ids,
    build_daily_percentile_targets,
)
from filters.breakout_quality.splits import (
    build_selection_oos_split_assignments,
    compute_outer_policy_fingerprint,
    resolve_breakout_quality_outer_policy,
)

from .checks import add_check, bind_checks


CONFIGURED_EXPERIMENT = get_breakout_quality_experiment_profile(
    BREAKOUT_QUALITY_EXPERIMENT_PROFILE
)
CONFIGURED_PRETRAINING = get_breakout_quality_pretraining_profile(
    BREAKOUT_QUALITY_PRETRAINING_PROFILE
)

__all__ = tuple(
    name for name in globals()
    if not name.startswith("__")
)
