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

from tools.audit.catalog import get_domain_cli_commands


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


def _registered_breakout_quality_audit_module(command: str) -> str | None:
    entry = get_domain_cli_commands("breakout_quality").get(str(command))
    return None if entry is None else str(entry.module)


from core.active_param_ensemble import build_static_active_param_ensemble_payload
from core.params_io import params_to_json_dict

from config.breakout_policy import (
    BREAKOUT_DEFAULT_HIGH_LEN,
    BREAKOUT_HIGH_LEN_SEARCH_MAX,
    BREAKOUT_HIGH_LEN_SEARCH_MIN,
    BREAKOUT_HIGH_LEN_SEARCH_STEP,
    build_breakout_optimizer_high_len_values,
)
from config.breakout_quality import (
    ADAMW_ONLY_EXPERIMENT_PROFILE,
    ADAM_WARMUP_COSINE_EXPERIMENT_PROFILE,
    BASELINE_EXPERIMENT_PROFILE,
    HISTORY_MASKING_ONLY_EXPERIMENT_PROFILE,
    UNIQUE_GROUP_DATE_BALANCED_EXPERIMENT_PROFILE,
    UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE,
    STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE,
    STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE,
    STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE,
    STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE,
    STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_LISTWISE_PROFILE,
    DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE,
    DAILY_UNIVERSAL_NO_TIME_PAIRWISE_GAP_WEIGHTED_PROFILE,
    DAILY_UNIVERSAL_NO_TIME_PERCENTILE_MSE_PROFILE,
    CONTINUOUS_RANKER_PAIRWISE_REDUCTION_EQUAL_PAIR,
    CONTINUOUS_RANKER_PAIRWISE_REDUCTION_TARGET_GAP_WEIGHTED,
    CONTINUOUS_RANKER_TRAINER_EVENT,
    SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES,
    get_continuous_ranker_research_spec,
    SUPPORTED_BREAKOUT_QUALITY_CLASSIFICATION_EXPERIMENT_PROFILES,
    TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION,
    TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_LISTWISE_RANKING,
    TRAINING_LABEL_SCOPE_PASS_ONLY,
    TRAINING_LABEL_SCOPE_ALL,
    TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS,
    TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    LR_SCHEDULE_LINEAR_WARMUP_COSINE,
    TIME_WEIGHT_MODE_DATE_BALANCED,
    TRAINING_WEIGHT_REDUCTION_FIXED_BATCH_SIZE,
    SUPPORTED_BREAKOUT_QUALITY_EXPERIMENT_PROFILES,
    SUPPORTED_BREAKOUT_QUALITY_LR_SCHEDULES,
    SUPPORTED_BREAKOUT_QUALITY_OPTIMIZERS,
    TRAINING_SAMPLING_ALL_EVENT_ROWS,
    TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
    TS2VEC_SELECTION_ONLY_PRETRAINING_PROFILE,
    SUPPORTED_BREAKOUT_QUALITY_PRETRAINING_PROFILES,
    build_breakout_quality_pretraining_profile_payload,
    get_breakout_quality_experiment_profile,
    get_breakout_quality_pretraining_profile,
)
from config.breakout_quality import (
    BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    BREAKOUT_QUALITY_DEFAULT_BATCH_SIZE,
    BREAKOUT_QUALITY_DEFAULT_EPOCHS,
    BREAKOUT_QUALITY_DEFAULT_GRADIENT_CLIP_NORM,
    BREAKOUT_QUALITY_DEFAULT_LEARNING_RATE,
    BREAKOUT_QUALITY_RANDOM_SEED,
    BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD,
    BREAKOUT_QUALITY_DEFAULT_WEIGHT_DECAY,
    BREAKOUT_QUALITY_FINAL_REFIT_MODE,
    BREAKOUT_QUALITY_CLASS_WEIGHT_MODE,
    BREAKOUT_QUALITY_TIME_WEIGHT_MODE,
    BREAKOUT_QUALITY_EARLY_STOPPING_MIN_DELTA,
    BREAKOUT_QUALITY_EARLY_STOPPING_PATIENCE,
    BREAKOUT_QUALITY_EVALUATION_BATCH_SIZE,
    BREAKOUT_QUALITY_EVALUATION_WORKERS,
    BREAKOUT_QUALITY_PARALLEL_SPLIT_EVALUATION,
    BREAKOUT_QUALITY_PRELOAD_FEATURE_BANK,
    BREAKOUT_QUALITY_PRETRAINING_PROFILE,
    BREAKOUT_QUALITY_TRAIN_PREFETCH_BATCHES,
    BREAKOUT_QUALITY_TORCH_DEVICE,
    BREAKOUT_QUALITY_USE_MIXED_PRECISION,
    BREAKOUT_QUALITY_MIXED_PRECISION_DTYPE,
    BREAKOUT_QUALITY_DETERMINISTIC_ALGORITHMS,
    BREAKOUT_QUALITY_ALLOW_TF32,
    BREAKOUT_QUALITY_INNER_VALIDATION_MONTHS,
    BREAKOUT_QUALITY_INCEPTION_DEPTH,
    BREAKOUT_QUALITY_INCEPTION_TARGET_RECEPTIVE_FIELD_BARS,
    BREAKOUT_QUALITY_INCEPTION_RESIDUAL_EVERY,
    BREAKOUT_QUALITY_MARKET_SET_CANDIDATE_QUERY_COUNT,
    BREAKOUT_QUALITY_FEATURE_WINDOW_BARS,
    BREAKOUT_QUALITY_LABEL_HORIZON_BARS,
    BREAKOUT_QUALITY_LABEL_MAX_ADVERSE_RETURN,
    BREAKOUT_QUALITY_LABEL_MIN_MFE_RETURN,
    BREAKOUT_QUALITY_LABEL_MIN_REWARD_RISK_RATIO,
    BREAKOUT_QUALITY_LABEL_PATH_CACHE_BARS,
    BREAKOUT_QUALITY_MIN_TRAIN_SAMPLES,
    BREAKOUT_QUALITY_MIN_VALIDATION_SAMPLES,
    BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
    BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
    BREAKOUT_QUALITY_USE_INNER_VALIDATION,
    build_breakout_quality_default_high_len_values,
    build_breakout_quality_inception_kernel_sizes,
    resolve_breakout_quality_inception_receptive_field_bars,
    resolve_breakout_quality_random_seed,
)
from filters.breakout_quality.artifacts import (
    _validate_torch_execution_record,
    build_file_manifest,
    load_model_artifact_contract,
    load_runtime_artifact_contract,
    load_split_assignment_frame,
)
from filters.breakout_quality.augmentation import (
    apply_training_augmentation,
    build_training_augmentation_plan,
)
from filters.breakout_quality.continuous_target import (
    CONTINUOUS_TARGET_SCHEMA_VERSION,
    STRATEGY_ALIGNED_TARGET_ID,
    STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
    StrategyAlignedContinuousTargetSpec,
    build_strategy_aligned_group_targets,
    build_strategy_aligned_no_time_contract,
    build_strategy_aligned_no_time_group_targets,
    load_validated_continuous_target_arrays,
    load_validated_continuous_target_component_arrays,
    resolve_continuous_target_dir,
    strategy_aligned_target_from_cached_path,
    TARGET_ADVERSE_RETURN_FILENAME,
    TARGET_FAVORABLE_RETURN_FILENAME,
    TARGET_MANIFEST_FILENAME,
    TARGET_OPPORTUNITY_BAR_FILENAME,
    TARGET_RAW_FILENAME,
    TARGET_RISK_BREACH_BAR_FILENAME,
    TARGET_TRADE_MATCHES_CSV_FILENAME,
    TARGET_VALID_MASK_FILENAME,
)
from filters.breakout_quality.contract import (
    ARTIFACT_CONTRACT_VERSION,
    CONTEXT_COLUMNS,
    DEFAULT_LABEL_POLICY,
    DEFAULT_MODEL_ARCHITECTURE,
    DEFAULT_UNAVAILABLE_SCORE_FILENAME,
    FEATURE_COLUMNS,
    FILTER_FAMILY,
    LABEL_INVALID,
    LABEL_OBJECTIVE,
    LEGACY_LABEL_OBJECTIVE,
    LABEL_PASS,
    LABEL_REJECT,
    BreakoutQualityLabelPolicy,
    label_manifest_payload_from_policy_manifest,
    SELECTION_ROLE_EMBARGO,
    SELECTION_ROLE_INVALID,
    SELECTION_ROLE_INNER_EMBARGO,
    SELECTION_ROLE_NOT_APPLICABLE,
    SELECTION_ROLE_TRAIN,
    SELECTION_ROLE_VALIDATION,
    OUTER_SPLIT_OOS,
    OUTER_SPLIT_OUT_OF_SCOPE,
    OUTER_SPLIT_SELECTION,
    RUNTIME_SCOPE_FORWARD_OOS,
    RUNTIME_SCOPE_RESEARCH,
    SCORE_COLUMN,
    SCORE_COMPARISON,
    SCORE_TABLE_REQUIRED_COLUMNS,
    SCORE_TABLE_SCHEMA_VERSION,
    SCORE_THRESHOLD_SOURCE,
    SPLIT_ASSIGNMENT_REQUIRED_COLUMNS,
    SPLIT_ASSIGNMENT_SCHEMA_VERSION,
    TRAINING_MODE_INNER_VALIDATION_FULL_REFIT,
)
from filters.breakout_quality.models import (
    ACTIVE_MODEL_ARCHITECTURES,
    build_active_model,
    get_active_model_spec,
    INCEPTION_TIME_MARKET_SET_CANDIDATE_V1,
    INCEPTION_TIME_MARKET_SET_V1,
    PATCH_TRANSFORMER_V1,
    LEGACY_MODEL_ARCHITECTURES,
    build_model as build_breakout_quality_model,
    count_trainable_parameters,
    get_model_spec,
    validate_model_sequence_length,
)
from filters.breakout_quality.mantis_contract import (
    MANTIS_V2_CHECKPOINT_FILENAME,
    MANTIS_V2_CHECKPOINT_SHA256,
    MANTIS_V2_CONFIG_FILENAME,
    MANTIS_V2_CONFIG_SHA256,
    MANTIS_V2_REPOSITORY,
    MANTIS_V2_REVISION,
)
from filters.breakout_quality.moment_contract import (
    MOMENT_CHECKPOINT_FILENAME,
    MOMENT_CHECKPOINT_SHA256,
    MOMENT_CHECKPOINT_SIZE_BYTES,
    MOMENT_CONFIG_FILENAME,
    MOMENT_CONFIG_SIZE_BYTES,
    MOMENT_PACKAGE_NAME,
    MOMENT_PACKAGE_VERSION,
    MOMENT_REPOSITORY,
    MOMENT_REVISION,
    MOMENT_TRANSFORMERS_PACKAGE_NAME,
    MOMENT_TRANSFORMERS_VERSION,
)
from filters.breakout_quality.models.ts2vec import hierarchical_contrastive_loss
from filters.breakout_quality.pretraining_store import (
    build_file_record as build_pretraining_file_record,
    close_pretraining_windows,
    compute_pretraining_configuration_fingerprint,
    load_validated_pretrained_encoder_manifest,
    load_validated_pretraining_dataset,
    resolve_pretrained_encoder_paths,
    resolve_pretraining_dataset_paths,
)
from filters.breakout_quality.models.multiscale_cnn import (
    build_market_relative_return_delta_representation,
    build_return_delta_representation,
    build_window_zscore_representation,
)
from filters.breakout_quality.models.regime_context import (
    REGIME_CONTEXT_ANNUALIZATION_BARS,
    REGIME_CONTEXT_FEATURES,
    REGIME_CONTEXT_LOOKBACK_BARS,
    build_regime_context_from_level_sequence,
)
from filters.breakout_quality.paths import (
    resolve_existing_filter_artifact_paths,
    resolve_filter_artifact_paths,
    resolve_filter_model_output_dir,
    resolve_filter_output_dir,
    resolve_filter_research_score_path,
    resolve_filter_report_json_path,
    resolve_filter_report_markdown_path,
)
from filters.breakout_quality.score_store import (
    build_pass_condition_from_score_table,
    load_score_table,
    load_shared_group_score_table,
    lookup_breakout_quality_candidate_score,
)
from filters.breakout_quality.source_inventory import build_source_data_inventory
from filters.breakout_quality.dataset_store import (
    DATASET_STORAGE_FORMAT,
    DATASET_STORAGE_SCHEMA_VERSION,
    IndexedFeatureBank,
)
from filters.breakout_quality.features import (
    build_breakout_quality_inference_dataset_for_frame,
    build_event_label,
    label_from_cached_path,
)
from filters.breakout_quality.inference import (
    strict_parallel_batched_logits,
    strict_unique_group_batched_logits,
)
from filters.breakout_quality.market_set import (
    IndexedMarketSetBank,
    build_market_daily_base_features,
)
from filters.breakout_quality.torch_runtime import resolve_torch_execution_plan
from core.signal_utils import generate_signals
from core.buy_sort import (
    BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_ADJUSTED,
    BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_BUCKET,
    BREAKOUT_QUALITY_RANKING_POLICY_SCORE,
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
from core.portfolio_engine import _aggregate_ensemble_candidate_rows, _candidate_replay_snapshot
from core.portfolio_exits import _iter_reentry_watch_targets
from core.strategy_params import V16StrategyParams
from strategies.breakout.schema import BREAKOUT_PARAM_SPECS
from strategies.breakout.search_space import BREAKOUT_OPTIMIZER_SEARCH_SPACE
from filters.breakout_quality import workflow_io as breakout_quality_common
from tools.filters.breakout_quality import evaluate as breakout_quality_evaluate
from filters.breakout_quality import export_scores as breakout_quality_export_scores
from tools.filters.breakout_quality import train as breakout_quality_train
from tools.audit.breakout_quality.continuous_target import (
    _load_round_trip_source,
    _resolve_round_trip_path,
    build_continuous_target_audit,
    render_continuous_target_audit_markdown,
)
from tools.filters.breakout_quality.report import (
    build_report_payload,
    render_console_summary,
    render_markdown_report,
)
from tools.audit.breakout_quality.regime import (
    assign_market_regimes,
    build_regime_audit_payload,
    derive_benchmark_regime_features,
    render_regime_audit_markdown,
)
from tools.audit.breakout_quality.target_component_attribution import (
    attach_target_components as target_attribution_attach_components,
    attribution_metrics as target_attribution_metrics,
    render_markdown as render_target_attribution_markdown,
)
from tools.audit.breakout_quality.target_time_penalty_ablation import (
    _validated_source_csv as time_ablation_validated_source_csv,
    attach_time_penalty_ablation,
    render_markdown as render_time_penalty_ablation_markdown,
    time_penalty_ablation_metrics,
)
from tools.audit.breakout_quality.no_time_continuous_target import (
    _approved_workflow_rebuild_gate as approved_no_time_workflow_rebuild_gate,
    _selection_metrics as no_time_target_selection_metrics,
    _validated_11e_report as validated_11e_report_for_no_time_target,
    render_markdown as render_no_time_target_markdown,
)
from tools.audit.breakout_quality.qualified_candidate_set import (
    _actual_trade_metrics as qualified_audit_actual_trade_metrics,
    _assert_replay_matches_strategy_summary as assert_qualified_replay_matches_summary,
    _attach_ranker_scores as qualified_audit_attach_ranker_scores,
    _daily_coverage as qualified_audit_daily_coverage,
    _layer_metrics as qualified_audit_layer_metrics,
    _unique_groups as qualified_audit_unique_groups,
    _validate_strategy_metadata as validate_qualified_audit_strategy_metadata,
)
from filters.breakout_quality.strategy_compare_engine import (
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
    _capacity_summary,
    _comparison_labels,
    _comparison_output_dir_name,
    _first_existing_comparison_dir,
    canonical_strategy_compare_output_dir_names,
    _normalize_yearly_completeness,
    _resolve_comparison_period,
    _resolve_params_path,
    _validate_requested_param_policy,
    _to_json_native,
)
from filters.breakout_quality.trade_attribution import build_trade_attribution
from tools.filters.breakout_quality.train_continuous_ranker import (
    _scope_group_ids as continuous_ranker_scope_group_ids,
    _trade_alignment_metrics as continuous_ranker_trade_alignment_metrics,
    build_daily_percentile_targets,
)
from filters.breakout_quality.splits import (
    build_selection_oos_split_assignments,
    compute_outer_policy_fingerprint,
    resolve_breakout_quality_outer_policy,
)

from .checks import add_check


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
