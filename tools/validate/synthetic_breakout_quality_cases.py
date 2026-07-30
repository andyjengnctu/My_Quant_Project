from __future__ import annotations

from dataclasses import replace
import ast
import json
import math
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.active_param_ensemble import build_static_active_param_ensemble_payload
from core.params_io import params_to_json_dict

from config.breakout_policy import (
    BREAKOUT_DEFAULT_HIGH_LEN,
    BREAKOUT_HIGH_LEN_SEARCH_MAX,
    BREAKOUT_HIGH_LEN_SEARCH_MIN,
    BREAKOUT_HIGH_LEN_SEARCH_STEP,
    build_breakout_optimizer_high_len_values,
)
from config.breakout_quality_experiments import (
    ADAMW_ONLY_EXPERIMENT_PROFILE,
    ADAM_WARMUP_COSINE_EXPERIMENT_PROFILE,
    BASELINE_EXPERIMENT_PROFILE,
    HISTORY_MASKING_ONLY_EXPERIMENT_PROFILE,
    UNIQUE_GROUP_DATE_BALANCED_EXPERIMENT_PROFILE,
    UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE,
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
from config.breakout_quality_policy import (
    BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    BREAKOUT_QUALITY_DEFAULT_BATCH_SIZE,
    BREAKOUT_QUALITY_DEFAULT_EPOCHS,
    BREAKOUT_QUALITY_DEFAULT_GRADIENT_CLIP_NORM,
    BREAKOUT_QUALITY_DEFAULT_LEARNING_RATE,
    BREAKOUT_QUALITY_DEFAULT_RANDOM_SEED,
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
    STRATEGY_ALIGNED_TARGET_ID,
    StrategyAlignedContinuousTargetSpec,
    build_strategy_aligned_group_targets,
    resolve_continuous_target_dir,
    strategy_aligned_target_from_cached_path,
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
from core.buy_sort import BUY_LIMIT_OVERAGE_SORT_METHOD, sort_candidate_rows
from core.breakout_reentry import (
    create_breakout_reentry_signal_state,
    create_breakout_reentry_watch_state,
)
from core.extended_signals import resolve_breakout_quality_rank
from core.portfolio_candidates import _resolve_candidate_quality_ranking
from core.portfolio_engine import _aggregate_ensemble_candidate_rows
from core.portfolio_exits import _iter_reentry_watch_targets
from core.strategy_params import V16StrategyParams
from strategies.breakout.schema import BREAKOUT_PARAM_SPECS
from strategies.breakout.search_space import BREAKOUT_OPTIMIZER_SEARCH_SPACE
from tools.filters.breakout_quality import common as breakout_quality_common
from tools.filters.breakout_quality import evaluate as breakout_quality_evaluate
from tools.filters.breakout_quality import export_scores as breakout_quality_export_scores
from tools.filters.breakout_quality import train as breakout_quality_train
from tools.filters.breakout_quality.audit_continuous_target import (
    _resolve_round_trip_path,
    build_continuous_target_audit,
    render_continuous_target_audit_markdown,
)
from tools.filters.breakout_quality.report import (
    build_report_payload,
    render_console_summary,
    render_markdown_report,
)
from tools.filters.breakout_quality.regime_audit import (
    assign_market_regimes,
    build_regime_audit_payload,
    derive_benchmark_regime_features,
    render_regime_audit_markdown,
)
from tools.filters.breakout_quality.strategy_compare import (
    COMPARISON_MODE_SCORE_RANKING,
    PARAM_POLICY_BASE_FINALIST_BEST,
    PARAM_POLICY_BASE_FINALISTS_AGREE,
    _assert_controlled_ensemble_pair,
    _assert_controlled_param_pair,
    _build_controlled_param_source_pair,
    _load_param_source,
    _capacity_summary,
    _comparison_labels,
    _comparison_output_dir_name,
    _normalize_yearly_completeness,
    _resolve_comparison_period,
    _resolve_params_path,
    _validate_requested_param_policy,
    _to_json_native,
)
from tools.filters.breakout_quality.trade_attribution import build_trade_attribution
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


def validate_breakout_quality_policy_single_source_case(_base_params):
    case_id = "BREAKOUT_QUALITY_POLICY_SSOT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    optimizer_values = build_breakout_optimizer_high_len_values()
    quality_values = build_breakout_quality_default_high_len_values()
    search_spec = BREAKOUT_OPTIMIZER_SEARCH_SPACE["high_len"]

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "strategy_default_uses_config",
        int(BREAKOUT_DEFAULT_HIGH_LEN),
        int(BREAKOUT_PARAM_SPECS["high_len"]["default"]),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "optimizer_range_uses_config",
        (int(BREAKOUT_HIGH_LEN_SEARCH_MIN), int(BREAKOUT_HIGH_LEN_SEARCH_MAX), int(BREAKOUT_HIGH_LEN_SEARCH_STEP)),
        (int(search_spec["low"]), int(search_spec["high"]), int(search_spec["step"])),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "quality_coverage_contains_optimizer_grid",
        True,
        set(optimizer_values).issubset(set(quality_values)),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "quality_coverage_contains_strategy_default",
        True,
        int(BREAKOUT_DEFAULT_HIGH_LEN) in set(quality_values),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "fixed_threshold_is_user_configured_and_legal",
        True,
        0.0 <= float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD) <= 1.0,
    )
    inception_kernels = build_breakout_quality_inception_kernel_sizes()
    inception_receptive_field = resolve_breakout_quality_inception_receptive_field_bars()
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "inception_receptive_field_policy_is_legal_and_derived_from_config",
        True,
        bool(
            int(BREAKOUT_QUALITY_INCEPTION_DEPTH) >= 1
            and int(BREAKOUT_QUALITY_INCEPTION_RESIDUAL_EVERY) >= 1
            and int(BREAKOUT_QUALITY_INCEPTION_DEPTH)
            % int(BREAKOUT_QUALITY_INCEPTION_RESIDUAL_EVERY)
            == 0
            and int(BREAKOUT_QUALITY_INCEPTION_TARGET_RECEPTIVE_FIELD_BARS)
            <= int(BREAKOUT_QUALITY_FEATURE_WINDOW_BARS)
            and len(inception_kernels) == 3
            and all(value >= 3 and value % 2 == 1 for value in inception_kernels)
            and inception_kernels[0] > inception_kernels[1] > inception_kernels[2]
            and inception_receptive_field
            >= int(BREAKOUT_QUALITY_INCEPTION_TARGET_RECEPTIVE_FIELD_BARS)
            and int(BREAKOUT_QUALITY_MARKET_SET_CANDIDATE_QUERY_COUNT) >= 1
        ),
    )

    inference_dates = pd.bdate_range("2025-01-01", periods=8)
    inference_close = np.asarray([9.8, 9.9, 9.9, 9.9, 10.5, 10.4, 10.3, 10.2])
    inference_stock = pd.DataFrame(
        {
            "Open": inference_close - 0.1,
            "High": np.asarray([10.0, 10.0, 10.0, 10.0, 10.6, 10.5, 10.4, 10.3]),
            "Low": inference_close - 0.2,
            "Close": inference_close,
            "Volume": np.full((8,), 1000.0),
        },
        index=inference_dates,
    )
    inference_benchmark_full = pd.DataFrame(
        {
            "Open": np.full((8,), 20.0),
            "High": np.full((8,), 20.2),
            "Low": np.full((8,), 19.8),
            "Close": np.full((8,), 20.0),
            "Volume": np.full((8,), 2000.0),
        },
        index=inference_dates,
    )
    inference_policy = BreakoutQualityLabelPolicy(
        feature_window_bars=5,
        label_horizon_bars=2,
        label_path_cache_bars=2,
        high_len_values=(3,),
        min_mfe_return=0.05,
        min_reward_risk_ratio=2.0,
        max_adverse_return=-0.10,
        benchmark_ticker="0050",
    )
    event_date = inference_dates[4]
    unavailable_inference = build_breakout_quality_inference_dataset_for_frame(
        inference_stock,
        inference_benchmark_full.drop(index=event_date),
        ticker="2330",
        policy=inference_policy,
        start_date=event_date,
        end_date=event_date,
        require_context=False,
    )
    scoreable_inference = build_breakout_quality_inference_dataset_for_frame(
        inference_stock,
        inference_benchmark_full,
        ticker="2330",
        policy=inference_policy,
        start_date=event_date,
        end_date=event_date,
        require_context=False,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "forward_candidate_without_benchmark_date_is_explicit_conservative_unavailable",
        (0, (0, 5, len(FEATURE_COLUMNS)), 1, "benchmark_date_missing"),
        (
            len(unavailable_inference.events),
            unavailable_inference.feature_bank.shape,
            len(unavailable_inference.unavailable_events),
            unavailable_inference.unavailable_events.iloc[0]["reason"],
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "same_forward_candidate_is_model_scoreable_when_benchmark_feature_exists",
        (1, (1, 5, len(FEATURE_COLUMNS)), 0),
        (
            len(scoreable_inference.events),
            scoreable_inference.feature_bank.shape,
            len(scoreable_inference.unavailable_events),
        ),
    )
    configured_model_spec = get_model_spec(BREAKOUT_QUALITY_MODEL_ARCHITECTURE)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "model_architecture_is_versioned_and_user_configured",
        DEFAULT_MODEL_ARCHITECTURE,
        configured_model_spec.architecture,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "dataset_policy_does_not_include_model_architecture",
        False,
        "model_architecture" in DEFAULT_LABEL_POLICY.as_manifest_payload(),
    )
    sampling_events = pd.DataFrame(
        {
            "ticker": ["A", "A", "B", "C", "B"],
            "date": [
                "2020-01-01",
                "2020-01-01",
                "2020-01-01",
                "2020-01-02",
                "2020-01-01",
            ],
            "group_index": [10, 10, 11, 12, 11],
        }
    )
    sampling_labels = np.asarray([LABEL_PASS, LABEL_PASS, LABEL_REJECT, LABEL_PASS, LABEL_REJECT])
    sampling_input_idx = np.asarray([4, 3, 1, 2, 0], dtype=np.int64)
    sampled_idx, sampling_summary = breakout_quality_train._resolve_training_sampling_indices(
        sampling_events,
        sampling_labels,
        sampling_input_idx,
        mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
        model_spec=get_model_spec("multiscale_cnn_sequence_only_v1"),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "unique_group_sampling_uses_minimum_row_once_per_ticker_date",
        ([0, 2, 3], 3, 2, "minimum_original_event_row_index"),
        (
            sampled_idx.tolist(),
            int(sampling_summary["sampled_row_count"]),
            int(sampling_summary["duplicate_rows_removed"]),
            sampling_summary["representative_rule"],
        ),
    )
    baseline_idx, baseline_sampling_summary = (
        breakout_quality_train._resolve_training_sampling_indices(
            sampling_events,
            sampling_labels,
            sampling_input_idx,
            mode=TRAINING_SAMPLING_ALL_EVENT_ROWS,
            model_spec=get_model_spec("multiscale_cnn_sequence_only_v1"),
        )
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "baseline_sampling_preserves_original_rows_and_order",
        (sampling_input_idx.tolist(), True, 0),
        (
            baseline_idx.tolist(),
            bool(baseline_sampling_summary["uses_all_eligible_rows"]),
            int(baseline_sampling_summary["duplicate_rows_removed"]),
        ),
    )
    try:
        breakout_quality_train._resolve_training_sampling_indices(
            sampling_events,
            sampling_labels,
            sampling_input_idx,
            mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
            model_spec=get_model_spec("multiscale_cnn_v1"),
        )
        context_sampling_rejected = False
    except ValueError as exc:
        context_sampling_rejected = "sequence-only" in str(exc)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "unique_group_sampling_rejects_context_using_architecture",
        True,
        context_sampling_rejected,
    )

    mixed_sampling_labels = sampling_labels.copy()
    mixed_sampling_labels[1] = LABEL_REJECT
    try:
        breakout_quality_train._resolve_training_sampling_indices(
            sampling_events,
            mixed_sampling_labels,
            sampling_input_idx,
            mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
            model_spec=get_model_spec("multiscale_cnn_sequence_only_v1"),
        )
        mixed_label_rejected = False
    except ValueError as exc:
        mixed_label_rejected = "混合 label" in str(exc)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "unique_group_sampling_rejects_mixed_group_labels",
        True,
        mixed_label_rejected,
    )

    mixed_group_events = sampling_events.copy()
    mixed_group_events.loc[1, "group_index"] = 999
    try:
        breakout_quality_train._resolve_training_sampling_indices(
            mixed_group_events,
            sampling_labels,
            sampling_input_idx,
            mode=TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
            model_spec=get_model_spec("multiscale_cnn_sequence_only_v1"),
        )
        mixed_feature_group_rejected = False
    except ValueError as exc:
        mixed_feature_group_rejected = "多個 feature group" in str(exc)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "unique_group_sampling_rejects_mixed_feature_group_mapping",
        True,
        mixed_feature_group_rejected,
    )

    torch, _nn = breakout_quality_train.require_torch()
    tiny_model = build_breakout_quality_model(10, 4, architecture="tiny_cnn_v1")
    multiscale_model = build_breakout_quality_model(10, 4, architecture="multiscale_cnn_v1")
    multiscale_v2_model = build_breakout_quality_model(
        10, 4, architecture="multiscale_cnn_v2"
    )
    multiscale_v3_model = build_breakout_quality_model(
        10, 4, architecture="multiscale_cnn_v3"
    )
    multiscale_v4_model = build_breakout_quality_model(
        10, 4, architecture="multiscale_cnn_v4"
    )
    multiscale_v5_model = build_breakout_quality_model(
        10, 4, architecture="multiscale_cnn_v5"
    )
    multiscale_v6_model = build_breakout_quality_model(
        10, 4, architecture="multiscale_cnn_v6"
    )
    multiscale_v7_model = build_breakout_quality_model(
        10, 4, architecture="multiscale_cnn_v7"
    )
    multiscale_v8_model = build_breakout_quality_model(
        10, 4, architecture="multiscale_cnn_v8"
    )
    regime_context_model = build_breakout_quality_model(
        10, 4, architecture="multiscale_cnn_regime_context_v1"
    )
    sequence_only_model = build_breakout_quality_model(
        10, 4, architecture="multiscale_cnn_sequence_only_v1"
    )
    dual_path_model = build_breakout_quality_model(
        10, 4, architecture="multiscale_cnn_sequence_only_dual_path_v1"
    )
    modern_tcn_model = build_breakout_quality_model(
        10, 4, architecture="modern_tcn_v1"
    )
    inception_model = build_breakout_quality_model(
        10, 4, architecture="inception_time_v1"
    )
    inception_group_norm_model = build_breakout_quality_model(
        10, 4, architecture="inception_time_group_norm_v1"
    )
    inception_market_set_model = build_breakout_quality_model(
        10, 4, architecture=INCEPTION_TIME_MARKET_SET_V1
    )
    inception_market_set_candidate_model = build_breakout_quality_model(
        10, 4, architecture=INCEPTION_TIME_MARKET_SET_CANDIDATE_V1
    )
    patch_transformer_model = build_breakout_quality_model(
        10, 4, architecture=PATCH_TRANSFORMER_V1
    )
    ts2vec_model = build_breakout_quality_model(
        10, 4, architecture="ts2vec_frozen_linear_v1"
    )
    residual_model = build_breakout_quality_model(10, 4, architecture="residual_tcn_v1")
    tiny_parameter_count = count_trainable_parameters(tiny_model)
    multiscale_parameter_count = count_trainable_parameters(multiscale_model)
    multiscale_v2_parameter_count = count_trainable_parameters(multiscale_v2_model)
    multiscale_v3_parameter_count = count_trainable_parameters(multiscale_v3_model)
    multiscale_v4_parameter_count = count_trainable_parameters(multiscale_v4_model)
    multiscale_v5_parameter_count = count_trainable_parameters(multiscale_v5_model)
    multiscale_v6_parameter_count = count_trainable_parameters(multiscale_v6_model)
    multiscale_v7_parameter_count = count_trainable_parameters(multiscale_v7_model)
    multiscale_v8_parameter_count = count_trainable_parameters(multiscale_v8_model)
    regime_context_parameter_count = count_trainable_parameters(regime_context_model)
    sequence_only_parameter_count = count_trainable_parameters(sequence_only_model)
    dual_path_parameter_count = count_trainable_parameters(dual_path_model)
    modern_tcn_parameter_count = count_trainable_parameters(modern_tcn_model)
    inception_parameter_count = count_trainable_parameters(inception_model)
    inception_group_norm_parameter_count = count_trainable_parameters(
        inception_group_norm_model
    )
    inception_market_set_parameter_count = count_trainable_parameters(
        inception_market_set_model
    )
    inception_market_set_candidate_parameter_count = count_trainable_parameters(
        inception_market_set_candidate_model
    )
    patch_transformer_parameter_count = count_trainable_parameters(
        patch_transformer_model
    )
    ts2vec_trainable_parameter_count = count_trainable_parameters(ts2vec_model)
    ts2vec_total_parameter_count = sum(
        int(parameter.numel()) for parameter in ts2vec_model.parameters()
    )
    ts2vec_frozen_parameter_count = (
        ts2vec_total_parameter_count - ts2vec_trainable_parameter_count
    )
    residual_parameter_count = count_trainable_parameters(residual_model)
    modern_tcn_spec = get_model_spec("modern_tcn_v1")
    inception_spec = get_model_spec("inception_time_v1")
    inception_group_norm_spec = get_model_spec("inception_time_group_norm_v1")
    inception_market_set_spec = get_model_spec(INCEPTION_TIME_MARKET_SET_V1)
    inception_market_set_candidate_spec = get_model_spec(
        INCEPTION_TIME_MARKET_SET_CANDIDATE_V1
    )
    patch_transformer_spec = get_model_spec(PATCH_TRANSFORMER_V1)
    ts2vec_spec = get_model_spec("ts2vec_frozen_linear_v1")
    moment_spec = get_model_spec("moment_1_base_frozen_linear_v1")
    patch_input = torch.randn(5, 300, 10)
    patch_context_a = torch.randn(5, 4)
    patch_context_b = torch.randn(5, 4)
    patch_transformer_model.eval()
    with torch.no_grad():
        patch_logits_a = patch_transformer_model(patch_input, patch_context_a)
        patch_logits_b = patch_transformer_model(patch_input, patch_context_b)
    try:
        validate_model_sequence_length(patch_transformer_spec, 301)
        invalid_patch_length_rejected = False
    except ValueError as exc:
        invalid_patch_length_rejected = "patch size" in str(exc)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "patch_transformer_9f_contract_is_small_supervised_sequence_only",
        (
            "patch_transformer", 10, 10, 128, 3, 4, 256,
            "mean", "sinusoidal", False, (5, 2), True, True, 411138,
        ),
        (
            patch_transformer_spec.family,
            patch_transformer_spec.patch_transformer_patch_size,
            patch_transformer_spec.patch_transformer_patch_stride,
            patch_transformer_spec.patch_transformer_embedding_dim,
            patch_transformer_spec.patch_transformer_depth,
            patch_transformer_spec.patch_transformer_heads,
            patch_transformer_spec.patch_transformer_mlp_dim,
            patch_transformer_spec.patch_transformer_pooling,
            patch_transformer_spec.patch_transformer_positional_encoding,
            patch_transformer_spec.use_dataset_context,
            tuple(patch_logits_a.shape),
            bool(torch.equal(patch_logits_a, patch_logits_b)),
            invalid_patch_length_rejected,
            patch_transformer_parameter_count,
        ),
    )

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "moment_9e_spec_pins_frozen_external_linear_probe_contract",
        (
            "moment_frozen_linear",
            MOMENT_REPOSITORY,
            MOMENT_REVISION,
            512,
            8,
            8,
            768,
            12,
            12,
            "independent_channel_concat",
            "mean",
            False,
        ),
        (
            moment_spec.family,
            moment_spec.moment_repository,
            moment_spec.moment_revision,
            moment_spec.moment_input_length,
            moment_spec.moment_patch_length,
            moment_spec.moment_patch_stride,
            moment_spec.moment_embedding_dim,
            moment_spec.moment_transformer_layers,
            moment_spec.moment_transformer_heads,
            moment_spec.moment_channel_aggregation,
            moment_spec.moment_patch_reduction,
            moment_spec.use_dataset_context,
        ),
    )

    class _SyntheticMomentPipeline(torch.nn.Module):
        forward_batch_sizes: list[int] = []
        forward_sequence_lengths: list[int] = []

        def __init__(self, config, **kwargs):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
            self.config = config
            self.model_kwargs = kwargs.get("model_kwargs") or {}
            self.initialized = False

        def init(self):
            self.initialized = True

        def embed(self, *, x_enc, reduction="mean"):
            if reduction != "none":
                raise ValueError("synthetic MOMENT 僅接受 reduction=none")
            type(self).forward_batch_sizes.append(int(x_enc.shape[0]))
            type(self).forward_sequence_lengths.append(int(x_enc.shape[2]))
            base = x_enc.mean(dim=2, keepdim=True).unsqueeze(-1)
            embeddings = base.expand(-1, -1, 64, 768)
            return SimpleNamespace(embeddings=embeddings)

    _SyntheticMomentPipeline.forward_batch_sizes = []
    _SyntheticMomentPipeline.forward_sequence_lengths = []
    with patch(
        "filters.breakout_quality.models.moment.require_moment_pipeline_class",
        return_value=_SyntheticMomentPipeline,
    ):
        moment_model = build_breakout_quality_model(
            10, 4, architecture="moment_1_base_frozen_linear_v1"
        )
        moment_trainable_parameter_count = count_trainable_parameters(moment_model)
        moment_total_parameter_count = sum(
            int(parameter.numel()) for parameter in moment_model.parameters()
        )
        moment_encoder_requires_grad = [
            bool(parameter.requires_grad) for parameter in moment_model.encoder.parameters()
        ]
        moment_head_requires_grad = [
            bool(parameter.requires_grad) for parameter in moment_model.classifier.parameters()
        ]
        moment_model.train()
        moment_features = torch.randn((33, 300, 10), dtype=torch.float32)
        moment_context_a = torch.randn((33, 4), dtype=torch.float32)
        moment_context_b = torch.randn((33, 4), dtype=torch.float32)
        moment_encoder_before = {
            key: value.detach().clone()
            for key, value in moment_model.encoder.state_dict().items()
        }
        moment_head_before = {
            key: value.detach().clone()
            for key, value in moment_model.classifier.state_dict().items()
        }
        moment_optimizer = torch.optim.Adam(
            [parameter for parameter in moment_model.parameters() if parameter.requires_grad],
            lr=0.01,
        )
        moment_optimizer.zero_grad(set_to_none=True)
        moment_logits_a = moment_model(moment_features, moment_context_a)
        moment_logits_b = moment_model(moment_features, moment_context_b)
        moment_loss = torch.nn.functional.cross_entropy(
            moment_logits_a,
            torch.tensor(([0, 1] * 16) + [0], dtype=torch.long),
        )
        moment_loss.backward()
        moment_optimizer.step()
        moment_encoder_unchanged = all(
            torch.equal(value, moment_encoder_before[key])
            for key, value in moment_model.encoder.state_dict().items()
        )
        moment_head_changed = any(
            not torch.equal(value, moment_head_before[key])
            for key, value in moment_model.classifier.state_dict().items()
        )
        moment_reload = build_breakout_quality_model(
            10, 4, model_spec=moment_spec.as_manifest_payload()
        )
        moment_reload.load_state_dict(moment_model.state_dict(), strict=True)

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "moment_9e_encoder_is_frozen_context_independent_chunked_and_reloadable",
        True,
        bool(
            moment_total_parameter_count == 15363
            and moment_trainable_parameter_count == 15362
            and not any(moment_encoder_requires_grad)
            and all(moment_head_requires_grad)
            and moment_model.encoder.training is False
            and getattr(moment_model.encoder, "initialized", False)
            and tuple(moment_logits_a.shape) == (33, 2)
            and torch.equal(moment_logits_a, moment_logits_b)
            and torch.isfinite(moment_logits_a).all()
            and moment_encoder_unchanged
            and moment_head_changed
            and _SyntheticMomentPipeline.forward_batch_sizes == [32, 1, 32, 1]
            and _SyntheticMomentPipeline.forward_sequence_lengths == [512, 512, 512, 512]
            and moment_reload.encoder.training is False
        ),
    )

    mantis_spec = get_model_spec("mantis_v2_frozen_linear_v1")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "mantis_9d_spec_pins_frozen_external_linear_probe_contract",
        (
            "mantis_v2_frozen_linear",
            "paris-noah/MantisV2",
            "99fe0f548960e272fbfa4b82fd9b5b5956779dfd",
            512,
            32,
            2,
            "combined",
            "independent_channel_concat",
            False,
        ),
        (
            mantis_spec.family,
            mantis_spec.mantis_repository,
            mantis_spec.mantis_revision,
            mantis_spec.mantis_input_length,
            mantis_spec.mantis_num_patches,
            mantis_spec.mantis_return_transformer_layer,
            mantis_spec.mantis_output_token,
            mantis_spec.mantis_channel_aggregation,
            mantis_spec.use_dataset_context,
        ),
    )
    class _SyntheticMantisV2(torch.nn.Module):
        forward_batch_sizes: list[int] = []
        forward_sequence_lengths: list[int] = []

        def __init__(self, **kwargs):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
            self.return_transf_layer = int(kwargs["return_transf_layer"])
            self.output_token = str(kwargs["output_token"])
            self.layers_removed = False

        def remove_transf_layers(self):
            self.layers_removed = True

        def forward(self, x):
            type(self).forward_batch_sizes.append(int(x.shape[0]))
            type(self).forward_sequence_lengths.append(int(x.shape[2]))
            base = x.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
            return base.repeat(1, 512) * self.scale

    _SyntheticMantisV2.forward_batch_sizes = []
    _SyntheticMantisV2.forward_sequence_lengths = []
    with patch(
        "filters.breakout_quality.models.mantis_v2.require_mantis_v2_class",
        return_value=_SyntheticMantisV2,
    ):
        mantis_model = build_breakout_quality_model(
            10, 4, architecture="mantis_v2_frozen_linear_v1"
        )
        mantis_trainable_parameter_count = count_trainable_parameters(mantis_model)
        mantis_total_parameter_count = sum(
            int(parameter.numel()) for parameter in mantis_model.parameters()
        )
        mantis_encoder_requires_grad = [
            bool(parameter.requires_grad) for parameter in mantis_model.encoder.parameters()
        ]
        mantis_head_requires_grad = [
            bool(parameter.requires_grad) for parameter in mantis_model.classifier.parameters()
        ]
        mantis_model.train()
        mantis_features = torch.randn((103, 300, 10), dtype=torch.float32)
        mantis_context_a = torch.randn((103, 4), dtype=torch.float32)
        mantis_context_b = torch.randn((103, 4), dtype=torch.float32)
        mantis_encoder_before = {
            key: value.detach().clone()
            for key, value in mantis_model.encoder.state_dict().items()
        }
        mantis_head_before = {
            key: value.detach().clone()
            for key, value in mantis_model.classifier.state_dict().items()
        }
        mantis_optimizer = torch.optim.Adam(
            [parameter for parameter in mantis_model.parameters() if parameter.requires_grad],
            lr=0.01,
        )
        mantis_optimizer.zero_grad(set_to_none=True)
        mantis_logits_a = mantis_model(mantis_features, mantis_context_a)
        mantis_logits_b = mantis_model(mantis_features, mantis_context_b)
        mantis_loss = torch.nn.functional.cross_entropy(
            mantis_logits_a,
            torch.tensor(([0, 1] * 51) + [0], dtype=torch.long),
        )
        mantis_loss.backward()
        mantis_optimizer.step()
        mantis_encoder_unchanged = all(
            torch.equal(value, mantis_encoder_before[key])
            for key, value in mantis_model.encoder.state_dict().items()
        )
        mantis_head_changed = any(
            not torch.equal(value, mantis_head_before[key])
            for key, value in mantis_model.classifier.state_dict().items()
        )
        mantis_reload = build_breakout_quality_model(
            10, 4, model_spec=mantis_spec.as_manifest_payload()
        )
        mantis_reload.load_state_dict(mantis_model.state_dict(), strict=True)

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "mantis_9d_encoder_is_frozen_context_independent_chunked_and_reloadable",
        True,
        bool(
            mantis_total_parameter_count == 10243
            and mantis_trainable_parameter_count == 10242
            and not any(mantis_encoder_requires_grad)
            and all(mantis_head_requires_grad)
            and mantis_model.encoder.training is False
            and getattr(mantis_model.encoder, "layers_removed", False)
            and tuple(mantis_logits_a.shape) == (103, 2)
            and torch.equal(mantis_logits_a, mantis_logits_b)
            and torch.isfinite(mantis_logits_a).all()
            and mantis_encoder_unchanged
            and mantis_head_changed
            and _SyntheticMantisV2.forward_batch_sizes == [1024, 6, 1024, 6]
            and _SyntheticMantisV2.forward_sequence_lengths == [512, 512, 512, 512]
            and mantis_reload.encoder.training is False
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "ts2vec_9c_frozen_probe_has_locked_encoder_and_linear_head",
        (
            831810,
            642,
            831168,
            "ts2vec_frozen_linear",
            8,
            128,
            320,
            1021,
            ("global_max",),
            False,
        ),
        (
            ts2vec_total_parameter_count,
            ts2vec_trainable_parameter_count,
            ts2vec_frozen_parameter_count,
            ts2vec_spec.family,
            ts2vec_spec.ts2vec_depth,
            ts2vec_spec.ts2vec_hidden_dims,
            ts2vec_spec.ts2vec_output_dims,
            ts2vec_spec.receptive_field_bars,
            ts2vec_spec.pooling,
            ts2vec_spec.use_dataset_context,
        ),
    )
    ts2vec_encoder_requires_grad = [
        bool(parameter.requires_grad)
        for parameter in ts2vec_model.encoder.parameters()
    ]
    ts2vec_head_requires_grad = [
        bool(parameter.requires_grad)
        for parameter in ts2vec_model.classifier.parameters()
    ]
    ts2vec_model.train()
    ts2vec_features = torch.randn((4, 300, 10), dtype=torch.float32)
    ts2vec_context_a = torch.randn((4, 4), dtype=torch.float32)
    ts2vec_context_b = torch.randn((4, 4), dtype=torch.float32)
    ts2vec_encoder_before = {
        key: value.detach().clone()
        for key, value in ts2vec_model.encoder.state_dict().items()
    }
    ts2vec_head_before = {
        key: value.detach().clone()
        for key, value in ts2vec_model.classifier.state_dict().items()
    }
    ts2vec_optimizer = torch.optim.Adam(
        [parameter for parameter in ts2vec_model.parameters() if parameter.requires_grad],
        lr=0.01,
    )
    ts2vec_optimizer.zero_grad(set_to_none=True)
    ts2vec_logits_a = ts2vec_model(ts2vec_features, ts2vec_context_a)
    ts2vec_logits_b = ts2vec_model(ts2vec_features, ts2vec_context_b)
    ts2vec_loss = torch.nn.functional.cross_entropy(
        ts2vec_logits_a,
        torch.tensor([0, 1, 0, 1], dtype=torch.long),
    )
    ts2vec_loss.backward()
    ts2vec_optimizer.step()
    ts2vec_encoder_unchanged = all(
        torch.equal(value, ts2vec_encoder_before[key])
        for key, value in ts2vec_model.encoder.state_dict().items()
    )
    ts2vec_head_changed = any(
        not torch.equal(value, ts2vec_head_before[key])
        for key, value in ts2vec_model.classifier.state_dict().items()
    )
    ts2vec_reload = build_breakout_quality_model(
        10, 4, model_spec=ts2vec_spec.as_manifest_payload()
    )
    ts2vec_reload.load_state_dict(ts2vec_model.state_dict(), strict=True)
    z1 = torch.randn((3, 64, 16), dtype=torch.float32, requires_grad=True)
    z2 = torch.randn((3, 64, 16), dtype=torch.float32, requires_grad=True)
    ts2vec_contrastive_loss = hierarchical_contrastive_loss(
        torch, z1, z2, alpha=0.5, temporal_unit=0
    )
    ts2vec_contrastive_loss.backward()
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "ts2vec_encoder_is_frozen_context_independent_and_contrastive_loss_is_finite",
        True,
        bool(
            not any(ts2vec_encoder_requires_grad)
            and all(ts2vec_head_requires_grad)
            and ts2vec_model.encoder.training is False
            and torch.equal(ts2vec_logits_a, ts2vec_logits_b)
            and tuple(ts2vec_logits_a.shape) == (4, 2)
            and torch.isfinite(ts2vec_logits_a).all()
            and ts2vec_encoder_unchanged
            and ts2vec_head_changed
            and torch.isfinite(ts2vec_contrastive_loss)
            and z1.grad is not None
            and z2.grad is not None
            and torch.isfinite(z1.grad).all()
            and torch.isfinite(z2.grad).all()
        ),
    )
    with tempfile.TemporaryDirectory(prefix="ts2vec_contract_") as temp_dir_text:
        pretrain_root = Path(temp_dir_text)
        pretrain_dataset_paths = resolve_pretraining_dataset_paths(
            pretrain_root,
            "synthetic_quality",
            family="ts2vec_v1",
            stride=5,
        )
        pretrain_dataset_paths.output_dir.mkdir(parents=True, exist_ok=True)
        synthetic_windows = np.random.default_rng(7).normal(
            size=(4, 300, 10)
        ).astype(np.float32)
        with pretrain_dataset_paths.windows.open("wb") as handle:
            np.save(handle, synthetic_windows, allow_pickle=False)
        synthetic_index = pd.DataFrame(
            {
                "window_index": [0, 1, 2, 3],
                "ticker": ["1101", "1101", "2330", "2330"],
                "date": [
                    "2024-01-05",
                    "2024-01-12",
                    "2024-06-07",
                    "2024-12-27",
                ],
            }
        )
        synthetic_index.to_csv(
            pretrain_dataset_paths.index, index=False, encoding="utf-8-sig"
        )
        pretraining_configuration = {
            "family": "ts2vec_v1",
            "dataset_profile": "full",
            "stride": 5,
            "window_bars": 300,
            "feature_columns": list(FEATURE_COLUMNS),
            "selection_start_date": "2024-01-01",
            "selection_end_date": "2024-12-31",
            "outer_policy_fingerprint": "b" * 64,
            "source_inventory_sha256": "c" * 64,
            "requested_max_tickers": 0,
        }
        pretraining_fingerprint = compute_pretraining_configuration_fingerprint(
            pretraining_configuration
        )
        pretraining_summary = {
            "schema_version": 1,
            "format": "selection_rolling_windows_npy_v1",
            **pretraining_configuration,
            "configuration_fingerprint": pretraining_fingerprint,
            "window_count": 4,
            "oos_windows_used": False,
            "source_data_inventory": {},
            "artifacts": {
                "windows": build_pretraining_file_record(
                    pretrain_dataset_paths.windows
                ),
                "index": build_pretraining_file_record(
                    pretrain_dataset_paths.index
                ),
            },
        }
        pretrain_dataset_paths.summary.write_text(
            json.dumps(pretraining_summary, ensure_ascii=False), encoding="utf-8"
        )
        loaded_pretraining_summary, loaded_windows, loaded_index = (
            load_validated_pretraining_dataset(
                pretrain_root,
                "synthetic_quality",
                dataset_profile="full",
                family="ts2vec_v1",
                stride=5,
                expected_selection_start="2024-01-01",
                expected_selection_end="2024-12-31",
                expected_window_bars=300,
                expected_max_tickers=0,
                require_current_source=False,
            )
        )
        loaded_windows_shape = tuple(int(value) for value in loaded_windows.shape)
        close_pretraining_windows(loaded_windows)
        loaded_windows_released = bool(
            getattr(loaded_windows, "_mmap", None) is None
            or loaded_windows._mmap.closed
        )
        encoder_paths = resolve_pretrained_encoder_paths(
            pretrain_root,
            "synthetic_quality",
            model_architecture="ts2vec_frozen_linear_v1",
            experiment_profile=UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE,
        )
        encoder_paths.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"encoder_state_dict": ts2vec_model.encoder.state_dict()}, encoder_paths.encoder)
        encoder_manifest = {
            "schema_version": 1,
            "model_architecture": "ts2vec_frozen_linear_v1",
            "experiment_profile": UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE,
            "model_spec": ts2vec_spec.as_manifest_payload(),
            "pretraining_profile": build_breakout_quality_pretraining_profile_payload(
                BREAKOUT_QUALITY_PRETRAINING_PROFILE
            ),
            "encoder": build_pretraining_file_record(encoder_paths.encoder),
            "pretraining_dataset_fingerprint": pretraining_fingerprint,
            "oos_windows_used": False,
            "pass_reject_labels_used": False,
        }
        encoder_paths.manifest.write_text(
            json.dumps(encoder_manifest, ensure_ascii=False), encoding="utf-8"
        )
        loaded_encoder_manifest = load_validated_pretrained_encoder_manifest(
            encoder_paths,
            expected_architecture="ts2vec_frozen_linear_v1",
            expected_experiment_profile=UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE,
            expected_dataset_fingerprint=pretraining_fingerprint,
            expected_model_spec=ts2vec_spec.as_manifest_payload(),
            expected_pretraining_profile=(
                build_breakout_quality_pretraining_profile_payload(
                    BREAKOUT_QUALITY_PRETRAINING_PROFILE
                )
            ),
        )
        tampered_encoder_manifest = dict(encoder_manifest)
        tampered_encoder_manifest["pretraining_profile"] = {
            **encoder_manifest["pretraining_profile"],
            "learning_rate": 0.123,
        }
        encoder_paths.manifest.write_text(
            json.dumps(tampered_encoder_manifest, ensure_ascii=False), encoding="utf-8"
        )
        try:
            load_validated_pretrained_encoder_manifest(
                encoder_paths,
                expected_architecture="ts2vec_frozen_linear_v1",
                expected_experiment_profile=UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE,
                expected_dataset_fingerprint=pretraining_fingerprint,
                expected_model_spec=ts2vec_spec.as_manifest_payload(),
                expected_pretraining_profile=(
                    build_breakout_quality_pretraining_profile_payload(
                        BREAKOUT_QUALITY_PRETRAINING_PROFILE
                    )
                ),
            )
            pretraining_profile_mismatch_rejected = False
        except ValueError as exc:
            pretraining_profile_mismatch_rejected = "pretraining_profile" in str(exc)
        tampered_encoder_manifest = dict(encoder_manifest)
        tampered_encoder_manifest["pass_reject_labels_used"] = True
        encoder_paths.manifest.write_text(
            json.dumps(tampered_encoder_manifest, ensure_ascii=False), encoding="utf-8"
        )
        try:
            load_validated_pretrained_encoder_manifest(
                encoder_paths,
                expected_architecture="ts2vec_frozen_linear_v1",
                expected_experiment_profile=UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE,
                expected_dataset_fingerprint=pretraining_fingerprint,
                expected_model_spec=ts2vec_spec.as_manifest_payload(),
                expected_pretraining_profile=(
                    build_breakout_quality_pretraining_profile_payload(
                        BREAKOUT_QUALITY_PRETRAINING_PROFILE
                    )
                ),
            )
            pretraining_label_leak_rejected = False
        except ValueError as exc:
            pretraining_label_leak_rejected = "PASS/REJECT labels" in str(exc)
        tampered_encoder_manifest = dict(encoder_manifest)
        tampered_encoder_manifest["oos_windows_used"] = True
        encoder_paths.manifest.write_text(
            json.dumps(tampered_encoder_manifest, ensure_ascii=False), encoding="utf-8"
        )
        try:
            load_validated_pretrained_encoder_manifest(
                encoder_paths,
                expected_architecture="ts2vec_frozen_linear_v1",
                expected_experiment_profile=UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE,
                expected_dataset_fingerprint=pretraining_fingerprint,
                expected_model_spec=ts2vec_spec.as_manifest_payload(),
                expected_pretraining_profile=(
                    build_breakout_quality_pretraining_profile_payload(
                        BREAKOUT_QUALITY_PRETRAINING_PROFILE
                    )
                ),
            )
            pretraining_oos_leak_rejected = False
        except ValueError as exc:
            pretraining_oos_leak_rejected = "OOS windows" in str(exc)
        synthetic_index.loc[3, "date"] = "2025-01-03"
        synthetic_index.to_csv(
            pretrain_dataset_paths.index, index=False, encoding="utf-8-sig"
        )
        pretraining_summary["artifacts"]["index"] = build_pretraining_file_record(
            pretrain_dataset_paths.index
        )
        pretrain_dataset_paths.summary.write_text(
            json.dumps(pretraining_summary, ensure_ascii=False), encoding="utf-8"
        )
        try:
            load_validated_pretraining_dataset(
                pretrain_root,
                "synthetic_quality",
                dataset_profile="full",
                family="ts2vec_v1",
                stride=5,
                expected_selection_start="2024-01-01",
                expected_selection_end="2024-12-31",
                expected_window_bars=300,
                expected_max_tickers=0,
                require_current_source=False,
            )
            pretraining_date_leak_rejected = False
        except ValueError as exc:
            pretraining_date_leak_rejected = "Selection 結束日後" in str(exc)
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "ts2vec_pretraining_contract_rejects_oos_labels_and_endpoint_leakage",
            True,
            bool(
                loaded_pretraining_summary["configuration_fingerprint"]
                == pretraining_fingerprint
                and loaded_windows_shape == (4, 300, 10)
                and loaded_windows_released
                and len(loaded_index) == 4
                and loaded_encoder_manifest["oos_windows_used"] is False
                and loaded_encoder_manifest["pass_reject_labels_used"] is False
                and pretraining_profile_mismatch_rejected
                and pretraining_label_leak_rejected
                and pretraining_oos_leak_rejected
                and pretraining_date_leak_rejected
            ),
        )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "ts2vec_pretraining_uses_named_profile_with_valid_ranges",
        True,
        (
            BREAKOUT_QUALITY_PRETRAINING_PROFILE
            in SUPPORTED_BREAKOUT_QUALITY_PRETRAINING_PROFILES
            and CONFIGURED_PRETRAINING.name == BREAKOUT_QUALITY_PRETRAINING_PROFILE
            and CONFIGURED_PRETRAINING.family == "ts2vec_v1"
            and CONFIGURED_PRETRAINING.optimizer_name
            in SUPPORTED_BREAKOUT_QUALITY_OPTIMIZERS
            and CONFIGURED_PRETRAINING.epochs >= 1
            and CONFIGURED_PRETRAINING.batch_size >= 2
            and CONFIGURED_PRETRAINING.learning_rate > 0.0
            and CONFIGURED_PRETRAINING.weight_decay >= 0.0
            and CONFIGURED_PRETRAINING.gradient_clip_norm >= 0.0
            and CONFIGURED_PRETRAINING.min_crop_bars >= 2
            and 0.0 <= CONFIGURED_PRETRAINING.mask_probability < 1.0
            and 0.0 <= CONFIGURED_PRETRAINING.contrastive_alpha <= 1.0
            and CONFIGURED_PRETRAINING.temporal_unit >= 0
            and build_breakout_quality_pretraining_profile_payload(
                TS2VEC_SELECTION_ONLY_PRETRAINING_PROFILE
            )
            == CONFIGURED_PRETRAINING.as_manifest_payload()
            and build_breakout_quality_pretraining_profile_payload(
                TS2VEC_SELECTION_ONLY_PRETRAINING_PROFILE,
                epochs=3,
                batch_size=64,
                learning_rate=0.002,
                weight_decay=0.01,
                gradient_clip_norm=0.5,
                min_crop_bars=40,
                mask_probability=0.25,
                contrastive_alpha=0.75,
                temporal_unit=1,
            )["epochs"]
            == 3
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "modern_tcn_9b_is_capacity_matched_large_kernel_classifier",
        (
            475394,
            "modern_tcn",
            6,
            96,
            51,
            4,
            301,
            "batch_norm",
            ("global_average",),
            False,
        ),
        (
            modern_tcn_parameter_count,
            modern_tcn_spec.family,
            modern_tcn_spec.modern_tcn_depth,
            modern_tcn_spec.modern_tcn_channels,
            modern_tcn_spec.modern_tcn_kernel_size,
            modern_tcn_spec.modern_tcn_expansion_ratio,
            modern_tcn_spec.receptive_field_bars,
            modern_tcn_spec.normalization,
            modern_tcn_spec.pooling,
            modern_tcn_spec.use_dataset_context,
        ),
    )
    modern_depthwise_layers = [
        module
        for module in modern_tcn_model.modules()
        if isinstance(module, _nn.Conv1d)
        and tuple(module.kernel_size) == (51,)
        and int(module.groups) == 96
        and int(module.in_channels) == 96
        and int(module.out_channels) == 96
    ]
    modern_batch_norm_layers = [
        module for module in modern_tcn_model.modules() if isinstance(module, _nn.BatchNorm1d)
    ]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "modern_tcn_uses_six_depthwise_large_kernel_blocks_and_thirteen_batch_norms",
        (6, 13),
        (len(modern_depthwise_layers), len(modern_batch_norm_layers)),
    )
    modern_tcn_model.eval()
    modern_features = torch.randn((3, 300, 10), dtype=torch.float32)
    modern_context_a = torch.randn((3, 4), dtype=torch.float32)
    modern_context_b = torch.randn((3, 4), dtype=torch.float32)
    with torch.no_grad():
        modern_logits_a = modern_tcn_model(modern_features, modern_context_a)
        modern_logits_b = modern_tcn_model(modern_features, modern_context_b)
    modern_reload = build_breakout_quality_model(
        10,
        4,
        model_spec=modern_tcn_spec.as_manifest_payload(),
    )
    modern_reload.load_state_dict(modern_tcn_model.state_dict(), strict=True)
    modern_tcn_model.train()
    modern_train_logits = modern_tcn_model(modern_features, modern_context_a)
    modern_loss = modern_train_logits.square().mean()
    modern_loss.backward()
    modern_gradients_ok = all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in modern_tcn_model.parameters()
        if parameter.requires_grad
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "modern_tcn_context_invariance_backward_and_strict_reload",
        True,
        bool(
            torch.equal(modern_logits_a, modern_logits_b)
            and tuple(modern_logits_a.shape) == (3, 2)
            and torch.isfinite(modern_logits_a).all()
            and modern_gradients_ok
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "inception_time_9a_uses_configured_receptive_field_contract",
        (
            True,
            "inception_time",
            int(BREAKOUT_QUALITY_INCEPTION_DEPTH),
            32,
            32,
            build_breakout_quality_inception_kernel_sizes(),
            int(BREAKOUT_QUALITY_INCEPTION_RESIDUAL_EVERY),
            resolve_breakout_quality_inception_receptive_field_bars(),
            ("global_average",),
            False,
        ),
        (
            inception_parameter_count > 0,
            inception_spec.family,
            inception_spec.inception_depth,
            inception_spec.inception_filters,
            inception_spec.inception_bottleneck_channels,
            inception_spec.inception_kernel_sizes,
            inception_spec.inception_residual_every,
            inception_spec.receptive_field_bars,
            inception_spec.pooling,
            inception_spec.use_dataset_context,
        ),
    )
    inception_model.eval()
    inception_features = torch.randn(
        (3, int(BREAKOUT_QUALITY_FEATURE_WINDOW_BARS), 10), dtype=torch.float32
    )
    inception_context_a = torch.randn((3, 4), dtype=torch.float32)
    inception_context_b = torch.randn((3, 4), dtype=torch.float32)
    with torch.no_grad():
        inception_logits_a = inception_model(inception_features, inception_context_a)
        inception_logits_b = inception_model(inception_features, inception_context_b)
    inception_reload = build_breakout_quality_model(
        10,
        4,
        model_spec=inception_spec.as_manifest_payload(),
    )
    inception_reload.load_state_dict(inception_model.state_dict(), strict=True)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "inception_time_context_invariance_forward_and_strict_reload",
        True,
        bool(
            torch.equal(inception_logits_a, inception_logits_b)
            and tuple(inception_logits_a.shape) == (3, 2)
            and torch.isfinite(inception_logits_a).all()
        ),
    )
    inception_market_set_model.eval()
    market_sequences = torch.randn((2, 7, int(BREAKOUT_QUALITY_FEATURE_WINDOW_BARS), 5))
    market_history_mask = torch.ones(
        (2, 7, int(BREAKOUT_QUALITY_FEATURE_WINDOW_BARS)), dtype=torch.bool
    )
    market_valid_stock_mask = torch.ones((2, 7), dtype=torch.bool)
    event_to_market = torch.tensor([0, 1, 0], dtype=torch.long)
    market_inputs = (
        market_sequences,
        market_history_mask,
        market_valid_stock_mask,
        event_to_market,
    )
    with torch.no_grad():
        market_logits = inception_market_set_model(
            inception_features, inception_context_a, market_inputs
        )
        permutation = torch.tensor([3, 1, 6, 0, 5, 2, 4], dtype=torch.long)
        permuted_logits = inception_market_set_model(
            inception_features,
            inception_context_a,
            (
                market_sequences[:, permutation],
                market_history_mask[:, permutation],
                market_valid_stock_mask[:, permutation],
                event_to_market,
            ),
        )
    inception_market_set_reload = build_breakout_quality_model(
        10, 4, model_spec=inception_market_set_spec.as_manifest_payload()
    )
    inception_market_set_reload.load_state_dict(
        inception_market_set_model.state_dict(), strict=True
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "market_set_stage1_is_set_invariant_research_only_and_strict_reloadable",
        True,
        bool(
            inception_market_set_parameter_count > inception_parameter_count
            and inception_market_set_spec.family == "inception_time_market_set"
            and inception_market_set_spec.requires_market_set
            and inception_market_set_spec.market_set_temporal_normalization == "group_norm"
            and inception_market_set_spec.market_set_temporal_normalization_groups == 8
            and not inception_market_set_spec.use_dataset_context
            and tuple(market_logits.shape) == (3, 2)
            and torch.isfinite(market_logits).all()
            and torch.allclose(market_logits, permuted_logits, atol=1e-6, rtol=1e-6)
        ),
    )

    inception_market_set_candidate_model.eval()
    candidate_specific_features = torch.stack(
        (inception_features[0], inception_features[0], inception_features[1]), dim=0
    )
    candidate_same_date_mapping = torch.zeros((3,), dtype=torch.long)
    with torch.no_grad():
        encoded_candidate_embeddings = (
            inception_market_set_candidate_model.encode_candidate(
                candidate_specific_features
            )
        )
        candidate_embedding_delta = torch.linspace(
            -0.25,
            0.25,
            steps=int(encoded_candidate_embeddings.shape[1]),
            dtype=encoded_candidate_embeddings.dtype,
            device=encoded_candidate_embeddings.device,
        )
        candidate_embeddings = torch.stack(
            (
                encoded_candidate_embeddings[0],
                encoded_candidate_embeddings[0],
                encoded_candidate_embeddings[0] + candidate_embedding_delta,
            ),
            dim=0,
        )
        candidate_market_embeddings = (
            inception_market_set_candidate_model.encode_market_for_events(
                candidate_embeddings,
                market_sequences[:1],
                market_history_mask[:1],
                market_valid_stock_mask[:1],
                candidate_same_date_mapping,
            )
        )
        candidate_market_embeddings_permuted = (
            inception_market_set_candidate_model.encode_market_for_events(
                candidate_embeddings,
                market_sequences[:1, permutation],
                market_history_mask[:1, permutation],
                market_valid_stock_mask[:1, permutation],
                candidate_same_date_mapping,
            )
        )
        candidate_logits = inception_market_set_candidate_model(
            candidate_specific_features,
            inception_context_a,
            (
                market_sequences[:1],
                market_history_mask[:1],
                market_valid_stock_mask[:1],
                candidate_same_date_mapping,
            ),
        )
    inception_market_set_candidate_reload = build_breakout_quality_model(
        10,
        4,
        model_spec=inception_market_set_candidate_spec.as_manifest_payload(),
    )
    inception_market_set_candidate_reload.load_state_dict(
        inception_market_set_candidate_model.state_dict(), strict=True
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "market_set_10a_candidate_query_is_dynamic_set_invariant_and_strict_reloadable",
        True,
        bool(
            inception_market_set_candidate_parameter_count > inception_parameter_count
            and inception_market_set_candidate_spec.family == "inception_time_market_set"
            and inception_market_set_candidate_spec.requires_market_set
            and inception_market_set_candidate_spec.market_set_query_mode
            == "candidate_conditioned"
            and inception_market_set_candidate_spec.market_set_query_count >= 1
            and not inception_market_set_candidate_spec.use_dataset_context
            and tuple(candidate_logits.shape) == (3, 2)
            and torch.isfinite(candidate_logits).all()
            and torch.equal(
                candidate_market_embeddings[0], candidate_market_embeddings[1]
            )
            and not torch.equal(
                candidate_market_embeddings[0], candidate_market_embeddings[2]
            )
            and torch.allclose(
                candidate_market_embeddings,
                candidate_market_embeddings_permuted,
                atol=5e-6,
                rtol=5e-6,
            )
        ),
    )

    missing_market_rejected = False
    candidate_missing_market_rejected = False
    try:
        inception_market_set_model(inception_features, inception_context_a, None)
    except ValueError as exc:
        missing_market_rejected = "Market Set model" in str(exc)
    try:
        inception_market_set_candidate_model(
            inception_features, inception_context_a, None
        )
    except ValueError as exc:
        candidate_missing_market_rejected = "Market Set model" in str(exc)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "market_set_stage1_requires_explicit_market_inputs",
        True,
        bool(missing_market_rejected and candidate_missing_market_rejected),
    )

    market_forward_scope_rejected = False
    candidate_market_forward_scope_rejected = False
    try:
        breakout_quality_export_scores._validate_export_scope_model_support(
            RUNTIME_SCOPE_FORWARD_OOS, inception_market_set_spec
        )
    except ValueError as exc:
        market_forward_scope_rejected = "research score export" in str(exc)
    try:
        breakout_quality_export_scores._validate_export_scope_model_support(
            RUNTIME_SCOPE_FORWARD_OOS, inception_market_set_candidate_spec
        )
    except ValueError as exc:
        candidate_market_forward_scope_rejected = "research score export" in str(exc)
    breakout_quality_export_scores._validate_export_scope_model_support(
        RUNTIME_SCOPE_RESEARCH, inception_market_set_spec
    )
    breakout_quality_export_scores._validate_export_scope_model_support(
        RUNTIME_SCOPE_RESEARCH, inception_market_set_candidate_spec
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "market_set_stage1_is_research_only_until_forward_market_bank_exists",
        True,
        bool(
            market_forward_scope_rejected
            and candidate_market_forward_scope_rejected
        ),
    )

    market_dates = pd.date_range("2024-01-01", periods=305, freq="B")
    market_frame = pd.DataFrame(
        {
            "Open": np.linspace(100.0, 130.0, len(market_dates)),
            "High": np.linspace(101.0, 131.0, len(market_dates)),
            "Low": np.linspace(99.0, 129.0, len(market_dates)),
            "Close": np.linspace(100.5, 130.5, len(market_dates)),
            "Volume": np.linspace(1000.0, 3000.0, len(market_dates)),
        },
        index=market_dates,
    )
    original_daily, original_valid = build_market_daily_base_features(
        market_frame, market_dates
    )
    future_changed = market_frame.copy()
    future_changed.iloc[-1, future_changed.columns.get_loc("Close")] *= 1.5
    changed_daily, changed_valid = build_market_daily_base_features(
        future_changed, market_dates
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "market_set_daily_features_are_point_in_time_and_future_changes_do_not_rewrite_history",
        True,
        bool(
            np.array_equal(original_daily[:-1], changed_daily[:-1])
            and np.array_equal(original_valid[:-1], changed_valid[:-1])
            and not np.array_equal(original_daily[-1], changed_daily[-1])
        ),
    )

    synthetic_daily_features = np.stack(
        [original_daily, original_daily * 0.5, original_daily * -0.25], axis=1
    ).astype(np.float32)
    synthetic_daily_mask = np.stack(
        [original_valid, original_valid, original_valid], axis=1
    ).astype(np.bool_)
    synthetic_group_dates = np.asarray([299, 299, 300, 301, 302], dtype=np.int64)
    synthetic_market_bank = IndexedMarketSetBank(
        synthetic_daily_features,
        synthetic_daily_mask,
        market_dates.values.astype("datetime64[D]").astype(np.int64),
        synthetic_group_dates,
        history_bars=300,
        min_valid_history_ratio=0.80,
        max_stocks=0,
        max_dates_per_batch=2,
    )
    shared_market_batch = synthetic_market_bank.materialize_for_group_indices(
        np.asarray([0, 1, 2, 4], dtype=np.int64)
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "market_set_bank_reuses_unique_dates_and_preserves_masks",
        ((3, 3, 300, 5), (3, 3, 300), (3, 3), [0, 0, 1, 2]),
        (
            tuple(shared_market_batch.sequences.shape),
            tuple(shared_market_batch.history_mask.shape),
            tuple(shared_market_batch.valid_stock_mask.shape),
            shared_market_batch.event_to_market.tolist(),
        ),
    )

    batch_feature_bank = np.random.default_rng(20260729).normal(
        size=(5, 300, 10)
    ).astype(np.float32)
    batch_event_groups = np.asarray([0, 1, 2, 3, 4, 0, 2, 4], dtype=np.int64)
    market_indexed_features = IndexedFeatureBank(
        batch_feature_bank, batch_event_groups
    )
    shuffled_rows = np.asarray([7, 0, 5, 2, 1, 4, 3, 6], dtype=np.int64)
    market_optimizer_batches = breakout_quality_train._build_training_optimizer_batches(
        market_indexed_features,
        shuffled_rows,
        batch_size=3,
        market_set_bank=synthetic_market_bank,
    )
    max_dates_seen = 0
    max_logical_rows = 0
    flattened_optimizer_rows = []
    for optimizer_batch in market_optimizer_batches:
        logical_rows = np.concatenate(optimizer_batch)
        flattened_optimizer_rows.append(logical_rows)
        max_logical_rows = max(max_logical_rows, int(len(logical_rows)))
        for microbatch_rows in optimizer_batch:
            group_rows = batch_event_groups[np.asarray(microbatch_rows, dtype=np.int64)]
            date_rows = synthetic_market_bank.market_date_indices_for_group_indices(group_rows)
            max_dates_seen = max(max_dates_seen, int(np.unique(date_rows).size))
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "market_set_microbatches_cap_dates_without_inflating_optimizer_steps",
        (sorted(shuffled_rows.tolist()), 2, 3, 3),
        (
            sorted(np.concatenate(flattened_optimizer_rows).tolist()),
            max_dates_seen,
            max_logical_rows,
            len(market_optimizer_batches),
        ),
    )

    market_context = np.zeros((len(batch_event_groups), 4), dtype=np.float32)
    market_group_logits, market_event_to_group = strict_unique_group_batched_logits(
        torch,
        inception_market_set_model,
        market_indexed_features,
        market_context,
        batch_size=2,
        workers=1,
        market_set_bank=synthetic_market_bank,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "market_set_unique_group_inference_preserves_event_mapping_and_finite_logits",
        True,
        bool(
            tuple(market_group_logits.shape) == (5, 2)
            and tuple(market_event_to_group.shape) == (len(batch_event_groups),)
            and np.isfinite(market_group_logits).all()
            and market_event_to_group.tolist() == batch_event_groups.tolist()
        ),
    )

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "inception_time_group_norm_is_single_change_with_equal_parameter_count",
        (
            473218,
            "group_norm",
            8,
            "batch_norm",
            None,
        ),
        (
            inception_group_norm_parameter_count,
            inception_group_norm_spec.normalization,
            inception_group_norm_spec.normalization_groups,
            inception_spec.normalization,
            inception_spec.normalization_groups,
        ),
    )
    group_norm_layers = [
        module
        for module in inception_group_norm_model.modules()
        if isinstance(module, _nn.GroupNorm)
    ]
    group_norm_batch_norm_layers = [
        module
        for module in inception_group_norm_model.modules()
        if isinstance(module, _nn.BatchNorm1d)
    ]
    batch_norm_layers = [
        module for module in inception_model.modules() if isinstance(module, _nn.BatchNorm1d)
    ]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "inception_time_group_norm_replaces_all_eight_batch_norm_layers",
        (8, 0, 8),
        (len(group_norm_layers), len(group_norm_batch_norm_layers), len(batch_norm_layers)),
    )
    inception_group_norm_model.train()
    group_norm_features = torch.randn((2, 300, 10), dtype=torch.float32)
    group_norm_context = torch.randn((2, 4), dtype=torch.float32)
    with torch.no_grad():
        group_norm_single = inception_group_norm_model(
            group_norm_features[:1], group_norm_context[:1]
        )
        group_norm_with_companion = inception_group_norm_model(
            group_norm_features, group_norm_context
        )[:1]
    inception_group_norm_model.eval()
    with torch.no_grad():
        group_norm_logits_a = inception_group_norm_model(
            inception_features, inception_context_a
        )
        group_norm_logits_b = inception_group_norm_model(
            inception_features, inception_context_b
        )
    inception_group_norm_reload = build_breakout_quality_model(
        10,
        4,
        model_spec=inception_group_norm_spec.as_manifest_payload(),
    )
    inception_group_norm_reload.load_state_dict(
        inception_group_norm_model.state_dict(), strict=True
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "inception_time_group_norm_is_per_sample_context_invariant_and_strict_reloadable",
        True,
        bool(
            torch.allclose(
                group_norm_single, group_norm_with_companion, atol=1e-6, rtol=1e-6
            )
            and torch.equal(group_norm_logits_a, group_norm_logits_b)
            and tuple(group_norm_logits_a.shape) == (3, 2)
            and torch.isfinite(group_norm_logits_a).all()
        ),
    )
    cpu_execution = resolve_torch_execution_plan(
        torch,
        requested_device="cpu",
        mixed_precision=True,
        mixed_precision_dtype="auto",
        deterministic_algorithms=True,
        allow_tf32=False,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "torch_execution_policy_and_cpu_fallback_are_explicit",
        (
            "auto",
            True,
            "auto",
            True,
            False,
            "cpu",
            False,
            "float32",
        ),
        (
            BREAKOUT_QUALITY_TORCH_DEVICE,
            BREAKOUT_QUALITY_USE_MIXED_PRECISION,
            BREAKOUT_QUALITY_MIXED_PRECISION_DTYPE,
            BREAKOUT_QUALITY_DETERMINISTIC_ALGORITHMS,
            BREAKOUT_QUALITY_ALLOW_TF32,
            cpu_execution.device_type,
            cpu_execution.mixed_precision_enabled,
            cpu_execution.autocast_dtype_name,
        ),
    )
    valid_execution_record = cpu_execution.as_manifest_payload()
    valid_execution_accepted = True
    try:
        _validate_torch_execution_record(
            {"torch_execution": valid_execution_record},
            required=True,
        )
    except ValueError:
        valid_execution_accepted = False
    requested_cuda_resolved_cpu_rejected = False
    invalid_resolution = {**valid_execution_record, "requested_device": "cuda"}
    try:
        _validate_torch_execution_record(
            {"torch_execution": invalid_resolution},
            required=True,
        )
    except ValueError:
        requested_cuda_resolved_cpu_rejected = True
    disabled_mixed_precision_dtype_rejected = False
    invalid_dtype = {**valid_execution_record, "autocast_dtype": "bfloat16"}
    try:
        _validate_torch_execution_record(
            {"torch_execution": invalid_dtype},
            required=True,
        )
    except ValueError:
        disabled_mixed_precision_dtype_rejected = True
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "torch_execution_manifest_accepts_valid_cpu_fallback",
        True,
        valid_execution_accepted,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "torch_execution_manifest_rejects_requested_resolved_device_mismatch",
        True,
        requested_cuda_resolved_cpu_rejected,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "torch_execution_manifest_rejects_dtype_when_mixed_precision_disabled",
        True,
        disabled_mixed_precision_dtype_rejected,
    )
    class _FakeCudaMatmul:
        allow_tf32 = None

    class _FakeCudaBackend:
        matmul = _FakeCudaMatmul()

    class _FakeCudnnBackend:
        benchmark = None
        deterministic = None
        allow_tf32 = None

    class _FakeCudaRuntime:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_bf16_supported():
            return True

    class _FakeBackends:
        cudnn = _FakeCudnnBackend()
        cuda = _FakeCudaBackend()

    class _FakeTorchRuntime:
        cuda = _FakeCudaRuntime()
        backends = _FakeBackends()
        deterministic = None

        @staticmethod
        def device(value):
            return str(value)

        @classmethod
        def use_deterministic_algorithms(cls, value):
            cls.deterministic = bool(value)

    fake_cuda_execution = resolve_torch_execution_plan(
        _FakeTorchRuntime,
        requested_device="auto",
        mixed_precision=True,
        mixed_precision_dtype="auto",
        deterministic_algorithms=True,
        allow_tf32=False,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "torch_execution_auto_prefers_cuda_bfloat16_when_supported",
        ("cuda", True, "bfloat16", True, False, False, False),
        (
            fake_cuda_execution.device_type,
            fake_cuda_execution.mixed_precision_enabled,
            fake_cuda_execution.autocast_dtype_name,
            _FakeTorchRuntime.deterministic,
            _FakeTorchRuntime.backends.cudnn.benchmark,
            _FakeTorchRuntime.backends.cudnn.allow_tf32,
            _FakeTorchRuntime.backends.cuda.matmul.allow_tf32,
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "multiscale_cnn_is_medium_capacity_with_long_receptive_field",
        True,
        (
            tiny_parameter_count < multiscale_parameter_count < residual_parameter_count
            and 15000 <= multiscale_parameter_count <= 25000
            and get_model_spec("multiscale_cnn_v1").receptive_field_bars >= 240
            and get_model_spec("multiscale_cnn_v1").normalization == "group_norm"
            and "max" not in get_model_spec("multiscale_cnn_v1").pooling
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "multiscale_v2_changes_representation_without_changing_capacity",
        (
            multiscale_parameter_count,
            ("return_delta", "return_delta", "level"),
        ),
        (
            multiscale_v2_parameter_count,
            get_model_spec("multiscale_cnn_v2").branch_input_representations,
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "multiscale_v3_adds_market_relative_returns_without_changing_capacity",
        (
            multiscale_parameter_count,
            (
                "market_relative_return_delta",
                "market_relative_return_delta",
                "level",
            ),
        ),
        (
            multiscale_v3_parameter_count,
            get_model_spec("multiscale_cnn_v3").branch_input_representations,
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "multiscale_v4_only_reduces_long_branch_channels",
        (
            True,
            (16, 16, 8),
            (),
            get_model_spec("multiscale_cnn_v1").receptive_field_bars,
        ),
        (
            tiny_parameter_count < multiscale_v4_parameter_count < multiscale_parameter_count,
            get_model_spec("multiscale_cnn_v4").branch_channels,
            get_model_spec("multiscale_cnn_v4").branch_input_representations,
            get_model_spec("multiscale_cnn_v4").receptive_field_bars,
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "multiscale_v5_only_sets_intermediate_long_branch_channels",
        (
            True,
            (16, 16, 12),
            (),
            get_model_spec("multiscale_cnn_v1").receptive_field_bars,
        ),
        (
            multiscale_v4_parameter_count
            < multiscale_v5_parameter_count
            < multiscale_parameter_count,
            get_model_spec("multiscale_cnn_v5").branch_channels,
            get_model_spec("multiscale_cnn_v5").branch_input_representations,
            get_model_spec("multiscale_cnn_v5").receptive_field_bars,
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "multiscale_v6_only_increases_long_branch_dropout",
        (
            multiscale_parameter_count,
            (),
            (),
            (0.25, 0.25, 0.40),
            get_model_spec("multiscale_cnn_v1").receptive_field_bars,
        ),
        (
            multiscale_v6_parameter_count,
            get_model_spec("multiscale_cnn_v6").branch_channels,
            get_model_spec("multiscale_cnn_v6").branch_input_representations,
            get_model_spec("multiscale_cnn_v6").branch_dropouts,
            get_model_spec("multiscale_cnn_v6").receptive_field_bars,
        ),
    )
    v6_branch_dropout_values = tuple(
        float(module.p)
        for branch in multiscale_v6_model.branches
        for module in branch.network
        if module.__class__.__name__ == "Dropout"
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "multiscale_v6_runtime_branch_dropouts_are_025_025_040",
        (0.25, 0.25, 0.25, 0.25, 0.40, 0.40),
        v6_branch_dropout_values,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "multiscale_v7_only_changes_short_branch_to_return_delta",
        (
            multiscale_parameter_count,
            ("return_delta", "level", "level"),
            (),
            (),
            get_model_spec("multiscale_cnn_v1").receptive_field_bars,
        ),
        (
            multiscale_v7_parameter_count,
            get_model_spec("multiscale_cnn_v7").branch_input_representations,
            get_model_spec("multiscale_cnn_v7").branch_channels,
            get_model_spec("multiscale_cnn_v7").branch_dropouts,
            get_model_spec("multiscale_cnn_v7").receptive_field_bars,
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "multiscale_v8_only_changes_medium_branch_to_return_delta",
        (
            multiscale_parameter_count,
            ("level", "return_delta", "level"),
            (),
            (),
            get_model_spec("multiscale_cnn_v1").receptive_field_bars,
        ),
        (
            multiscale_v8_parameter_count,
            get_model_spec("multiscale_cnn_v8").branch_input_representations,
            get_model_spec("multiscale_cnn_v8").branch_channels,
            get_model_spec("multiscale_cnn_v8").branch_dropouts,
            get_model_spec("multiscale_cnn_v8").receptive_field_bars,
        ),
    )
    regime_spec = get_model_spec("multiscale_cnn_regime_context_v1")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "regime_context_architecture_only_adds_zero_initialized_projection",
        (
            multiscale_parameter_count + len(REGIME_CONTEXT_FEATURES) * 32,
            REGIME_CONTEXT_FEATURES,
            REGIME_CONTEXT_LOOKBACK_BARS,
            REGIME_CONTEXT_ANNUALIZATION_BARS,
            get_model_spec("multiscale_cnn_v1").receptive_field_bars,
        ),
        (
            regime_context_parameter_count,
            regime_spec.derived_context_features,
            regime_spec.derived_context_lookback_bars,
            regime_spec.derived_context_annualization_bars,
            regime_spec.receptive_field_bars,
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "regime_context_projection_starts_at_zero",
        True,
        bool(
            regime_context_model.derived_context_projection is not None
            and np.allclose(
                regime_context_model.derived_context_projection.weight.detach().cpu().numpy(),
                0.0,
            )
        ),
    )
    sequence_only_spec = get_model_spec("multiscale_cnn_sequence_only_v1")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "sequence_only_architecture_removes_only_four_dimensional_dataset_context",
        (
            multiscale_parameter_count - 4 * int(sequence_only_spec.head_width),
            False,
            get_model_spec("multiscale_cnn_v1").receptive_field_bars,
            get_model_spec("multiscale_cnn_v1").branch_input_representations,
        ),
        (
            sequence_only_parameter_count,
            sequence_only_spec.use_dataset_context,
            sequence_only_spec.receptive_field_bars,
            sequence_only_spec.branch_input_representations,
        ),
    )
    sequence_only_model.eval()
    sequence_features = torch.randn((3, 300, 10), dtype=torch.float32) * 0.02
    sequence_context_a = torch.randn((3, 4), dtype=torch.float32)
    sequence_context_b = torch.randn((3, 4), dtype=torch.float32)
    with torch.no_grad():
        sequence_logits_a = sequence_only_model(sequence_features, sequence_context_a)
        sequence_logits_b = sequence_only_model(sequence_features, sequence_context_b)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "sequence_only_logits_are_independent_of_dataset_context_values",
        True,
        bool(torch.equal(sequence_logits_a, sequence_logits_b)),
    )
    dual_path_spec = get_model_spec(
        "multiscale_cnn_sequence_only_dual_path_v1"
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "dual_path_architecture_adds_raw_and_window_zscore_paths_without_context",
        (
            True,
            ("raw_level", "window_zscore"),
            1e-5,
            sequence_only_spec.receptive_field_bars,
        ),
        (
            dual_path_parameter_count > sequence_only_parameter_count,
            dual_path_spec.sequence_input_paths,
            dual_path_spec.window_normalization_epsilon,
            dual_path_spec.receptive_field_bars,
        ),
    )
    normalization_input = torch.stack(
        [
            torch.arange(1, 7, dtype=torch.float32),
            torch.arange(1, 7, dtype=torch.float32) * 2.0 + 5.0,
            torch.ones(6, dtype=torch.float32) * 3.0,
        ],
        dim=0,
    ).unsqueeze(0)
    normalized_window = build_window_zscore_representation(
        torch, normalization_input, epsilon=1e-5
    )
    normalized_mean = torch.mean(normalized_window, dim=2)
    normalized_std = torch.std(normalized_window[:, :2, :], dim=2, unbiased=False)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "window_zscore_is_per_sample_per_channel_and_constant_safe",
        True,
        bool(
            torch.allclose(normalized_mean, torch.zeros_like(normalized_mean), atol=1e-6)
            and torch.allclose(normalized_std, torch.ones_like(normalized_std), atol=1e-6)
            and torch.equal(
                normalized_window[:, 2, :],
                torch.zeros_like(normalized_window[:, 2, :]),
            )
        ),
    )
    equivalence_features = torch.randn((4, 300, 10), dtype=torch.float32) * 0.03
    equivalence_context_a = torch.randn((4, 4), dtype=torch.float32)
    equivalence_context_b = torch.randn((4, 4), dtype=torch.float32)
    torch.manual_seed(271828)
    equivalence_base_model = build_breakout_quality_model(
        10, 4, architecture="multiscale_cnn_sequence_only_v1"
    )
    torch.manual_seed(271828)
    equivalence_dual_model = build_breakout_quality_model(
        10, 4, architecture="multiscale_cnn_sequence_only_dual_path_v1"
    )
    equivalence_base_model.eval()
    equivalence_dual_model.eval()
    with torch.no_grad():
        equivalence_base_logits = equivalence_base_model(
            equivalence_features, equivalence_context_a
        )
        equivalence_dual_logits_a = equivalence_dual_model(
            equivalence_features, equivalence_context_a
        )
        equivalence_dual_logits_b = equivalence_dual_model(
            equivalence_features, equivalence_context_b
        )
    shared_base_state_equal = all(
        key in equivalence_dual_model.state_dict()
        and tuple(value.shape)
        == tuple(equivalence_dual_model.state_dict()[key].shape)
        and torch.equal(value, equivalence_dual_model.state_dict()[key])
        for key, value in equivalence_base_model.state_dict().items()
    )
    fusion_identity = True
    for fusion in equivalence_dual_model.path_fusions:
        width = int(fusion.out_features)
        expected_weight = torch.zeros_like(fusion.weight)
        expected_weight[:, :width] = torch.eye(
            width, dtype=expected_weight.dtype, device=expected_weight.device
        )
        fusion_identity = fusion_identity and bool(
            torch.equal(fusion.weight, expected_weight)
            and torch.equal(fusion.bias, torch.zeros_like(fusion.bias))
        )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "dual_path_starts_as_exact_8f_function_and_ignores_dataset_context",
        True,
        bool(
            shared_base_state_equal
            and fusion_identity
            and torch.equal(equivalence_base_logits, equivalence_dual_logits_a)
            and torch.equal(equivalence_dual_logits_a, equivalence_dual_logits_b)
        ),
    )
    equivalence_dual_model.train()
    equivalence_dual_model.zero_grad(set_to_none=True)
    first_dual_loss = equivalence_dual_model(
        equivalence_features, equivalence_context_a
    ).pow(2).mean()
    first_dual_loss.backward()
    normalized_fusion_gradient = sum(
        float(
            torch.sum(
                torch.abs(
                    fusion.weight.grad[:, int(fusion.out_features):]
                )
            ).item()
        )
        for fusion in equivalence_dual_model.path_fusions
        if fusion.weight.grad is not None
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "dual_path_normalized_fusion_receives_first_step_gradient",
        True,
        normalized_fusion_gradient > 0.0,
    )
    with torch.no_grad():
        for fusion in equivalence_dual_model.path_fusions:
            fusion.weight[:, int(fusion.out_features):].fill_(0.01)
    equivalence_dual_model.zero_grad(set_to_none=True)
    second_dual_loss = equivalence_dual_model(
        equivalence_features, equivalence_context_a
    ).pow(2).mean()
    second_dual_loss.backward()
    normalized_branch_gradient = sum(
        float(torch.sum(torch.abs(parameter.grad)).item())
        for name, parameter in equivalence_dual_model.named_parameters()
        if name.startswith("normalized_branches.") and parameter.grad is not None
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "dual_path_normalized_branches_receive_gradient_after_fusion_opens",
        True,
        normalized_branch_gradient > 0.0,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "multiscale_v1_v2_v3_manifest_specs_remain_backward_compatible",
        False,
        any(
            "branch_channels" in get_model_spec(architecture).as_manifest_payload()
            for architecture in (
                "multiscale_cnn_v1",
                "multiscale_cnn_v2",
                "multiscale_cnn_v3",
            )
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "multiscale_v1_manifest_spec_remains_backward_compatible",
        False,
        "branch_input_representations"
        in get_model_spec("multiscale_cnn_v1").as_manifest_payload(),
    )
    regime_sequence = torch.zeros((2, 10, 61), dtype=torch.float32)
    stock_daily_log_return = 0.002
    benchmark_daily_log_return = 0.001
    stock_log_path = torch.arange(61, dtype=torch.float32) * stock_daily_log_return
    benchmark_log_path = torch.arange(61, dtype=torch.float32) * benchmark_daily_log_return
    regime_sequence[:, 3, :] = torch.expm1(stock_log_path)
    regime_sequence[:, 8, :] = torch.expm1(benchmark_log_path)
    derived_regime = build_regime_context_from_level_sequence(torch, regime_sequence)
    expected_regime = np.asarray(
        [
            20 * benchmark_daily_log_return,
            60 * benchmark_daily_log_return,
            0.0,
            0.0,
            20 * (stock_daily_log_return - benchmark_daily_log_return),
            60 * (stock_daily_log_return - benchmark_daily_log_return),
        ],
        dtype=np.float32,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "regime_context_uses_only_20_60_day_returns_volatility_and_relative_strength",
        True,
        bool(
            tuple(derived_regime.shape) == (2, len(REGIME_CONTEXT_FEATURES))
            and np.allclose(
                derived_regime.detach().cpu().numpy(),
                np.tile(expected_regime, (2, 1)),
                atol=1e-6,
            )
        ),
    )
    torch.manual_seed(314159)
    baseline_initial_model = build_breakout_quality_model(
        10, 4, architecture="multiscale_cnn_v1"
    )
    torch.manual_seed(314159)
    regime_initial_model = build_breakout_quality_model(
        10, 4, architecture="multiscale_cnn_regime_context_v1"
    )
    initial_features = torch.randn((3, 300, 10), dtype=torch.float32) * 0.02
    initial_context = torch.randn((3, 4), dtype=torch.float32) * 0.05
    baseline_initial_model.eval()
    regime_initial_model.eval()
    with torch.no_grad():
        baseline_logits = baseline_initial_model(initial_features, initial_context)
        regime_logits = regime_initial_model(initial_features, initial_context)
    common_state_equal = all(
        key in regime_initial_model.state_dict()
        and tuple(value.shape) == tuple(regime_initial_model.state_dict()[key].shape)
        and torch.equal(value, regime_initial_model.state_dict()[key])
        for key, value in baseline_initial_model.state_dict().items()
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "regime_context_preserves_v1_initial_common_weights_and_logits",
        True,
        bool(common_state_equal and torch.equal(baseline_logits, regime_logits)),
    )
    regime_initial_model.train()
    regime_loss = regime_initial_model(initial_features, initial_context).sum()
    regime_loss.backward()
    projection_gradient = regime_initial_model.derived_context_projection.weight.grad
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "regime_context_projection_receives_training_gradient",
        True,
        bool(
            projection_gradient is not None
            and torch.isfinite(projection_gradient).all()
            and float(torch.sum(torch.abs(projection_gradient)).item()) > 0.0
        ),
    )

    level_sequence = torch.zeros((1, 10, 3), dtype=torch.float32)
    level_sequence[0, 0:5, 1] = torch.tensor(
        [0.01, 0.03, -0.01, 0.02, 0.50], dtype=torch.float32
    )
    level_sequence[0, 0:5, 2] = torch.tensor(
        [0.03, 0.05, 0.01, 0.04, 0.20], dtype=torch.float32
    )
    level_sequence[0, 5:10, 1] = torch.tensor(
        [0.005, 0.02, -0.005, 0.01, 0.25], dtype=torch.float32
    )
    level_sequence[0, 5:10, 2] = torch.tensor(
        [0.02, 0.03, 0.00, 0.025, 0.40], dtype=torch.float32
    )
    return_delta = build_return_delta_representation(torch, level_sequence)
    expected_second_stock = np.asarray(
        [
            np.log1p(0.01),
            np.log1p(0.03),
            np.log1p(-0.01),
            np.log1p(0.02),
            0.50,
        ],
        dtype=np.float32,
    )
    expected_third_stock = np.asarray(
        [
            np.log1p(0.03) - np.log1p(0.02),
            np.log1p(0.05) - np.log1p(0.02),
            np.log1p(0.01) - np.log1p(0.02),
            np.log1p(0.04) - np.log1p(0.02),
            -0.30,
        ],
        dtype=np.float32,
    )
    expected_second_benchmark = np.asarray(
        [
            np.log1p(0.005),
            np.log1p(0.02),
            np.log1p(-0.005),
            np.log1p(0.01),
            0.25,
        ],
        dtype=np.float32,
    )
    expected_third_benchmark = np.asarray(
        [
            np.log1p(0.02) - np.log1p(0.01),
            np.log1p(0.03) - np.log1p(0.01),
            np.log1p(0.00) - np.log1p(0.01),
            np.log1p(0.025) - np.log1p(0.01),
            0.15,
        ],
        dtype=np.float32,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "multiscale_v2_return_delta_transform_matches_canonical_ohlcv_semantics",
        True,
        bool(
            np.allclose(
                return_delta[0, :, 0].detach().cpu().numpy(),
                np.zeros(10, dtype=np.float32),
                atol=1e-7,
            )
            and np.allclose(
                return_delta[0, 0:5, 1].detach().cpu().numpy(),
                expected_second_stock,
                atol=1e-6,
            )
            and np.allclose(
                return_delta[0, 0:5, 2].detach().cpu().numpy(),
                expected_third_stock,
                atol=1e-6,
            )
            and np.allclose(
                return_delta[0, 5:10, 1].detach().cpu().numpy(),
                expected_second_benchmark,
                atol=1e-6,
            )
            and np.allclose(
                return_delta[0, 5:10, 2].detach().cpu().numpy(),
                expected_third_benchmark,
                atol=1e-6,
            )
            and np.isfinite(return_delta.detach().cpu().numpy()).all()
        ),
    )
    market_relative_return_delta = (
        build_market_relative_return_delta_representation(torch, level_sequence)
    )
    expected_second_relative_stock = expected_second_stock.copy()
    expected_second_relative_stock[0:4] -= expected_second_benchmark[0:4]
    expected_third_relative_stock = expected_third_stock.copy()
    expected_third_relative_stock[0:4] -= expected_third_benchmark[0:4]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "multiscale_v3_market_relative_transform_preserves_contract",
        True,
        bool(
            np.allclose(
                market_relative_return_delta[0, :, 0].detach().cpu().numpy(),
                np.zeros(10, dtype=np.float32),
                atol=1e-7,
            )
            and np.allclose(
                market_relative_return_delta[0, 0:5, 1].detach().cpu().numpy(),
                expected_second_relative_stock,
                atol=1e-6,
            )
            and np.allclose(
                market_relative_return_delta[0, 0:5, 2].detach().cpu().numpy(),
                expected_third_relative_stock,
                atol=1e-6,
            )
            and np.allclose(
                market_relative_return_delta[0, 5:10, :].detach().cpu().numpy(),
                return_delta[0, 5:10, :].detach().cpu().numpy(),
                atol=1e-7,
            )
            and np.isfinite(
                market_relative_return_delta.detach().cpu().numpy()
            ).all()
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "multiscale_v2_forward_shape_matches_existing_contract",
        (2, 2),
        tuple(
            multiscale_v2_model(
                torch.zeros((2, 300, 10), dtype=torch.float32),
                torch.zeros((2, 4), dtype=torch.float32),
            ).shape
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "multiscale_v3_forward_shape_matches_existing_contract",
        (2, 2),
        tuple(
            multiscale_v3_model(
                torch.zeros((2, 300, 10), dtype=torch.float32),
                torch.zeros((2, 4), dtype=torch.float32),
            ).shape
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "multiscale_v4_forward_shape_matches_existing_contract",
        (2, 2),
        tuple(
            multiscale_v4_model(
                torch.zeros((2, 300, 10), dtype=torch.float32),
                torch.zeros((2, 4), dtype=torch.float32),
            ).shape
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "multiscale_v7_forward_shape_matches_existing_contract",
        (2, 2),
        tuple(
            multiscale_v7_model(
                torch.zeros((2, 300, 10), dtype=torch.float32),
                torch.zeros((2, 4), dtype=torch.float32),
            ).shape
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "multiscale_v8_forward_shape_matches_existing_contract",
        (2, 2),
        tuple(
            multiscale_v8_model(
                torch.zeros((2, 300, 10), dtype=torch.float32),
                torch.zeros((2, 4), dtype=torch.float32),
            ).shape
        ),
    )
    v7_probe = torch.linspace(-0.2, 0.2, steps=2 * 300 * 10, dtype=torch.float32).reshape(2, 300, 10)
    v7_level_input = v7_probe.transpose(1, 2)
    v7_expected_return = build_return_delta_representation(torch, v7_level_input)
    v7_captured_inputs = []
    v7_hooks = [
        branch.register_forward_pre_hook(
            lambda _module, inputs, captured=v7_captured_inputs: captured.append(
                inputs[0].detach().clone()
            )
        )
        for branch in multiscale_v7_model.branches
    ]
    try:
        multiscale_v7_model.eval()
        with torch.no_grad():
            multiscale_v7_model(v7_probe, torch.zeros((2, 4), dtype=torch.float32))
    finally:
        for hook in v7_hooks:
            hook.remove()
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "multiscale_v7_runtime_routes_return_only_to_short_branch",
        True,
        bool(
            len(v7_captured_inputs) == 3
            and torch.allclose(v7_captured_inputs[0], v7_expected_return)
            and torch.allclose(v7_captured_inputs[1], v7_level_input)
            and torch.allclose(v7_captured_inputs[2], v7_level_input)
        ),
    )
    v8_probe = torch.linspace(-0.2, 0.2, steps=2 * 300 * 10, dtype=torch.float32).reshape(2, 300, 10)
    v8_level_input = v8_probe.transpose(1, 2)
    v8_expected_return = build_return_delta_representation(torch, v8_level_input)
    v8_captured_inputs = []
    v8_hooks = [
        branch.register_forward_pre_hook(
            lambda _module, inputs, captured=v8_captured_inputs: captured.append(
                inputs[0].detach().clone()
            )
        )
        for branch in multiscale_v8_model.branches
    ]
    try:
        multiscale_v8_model.eval()
        with torch.no_grad():
            multiscale_v8_model(v8_probe, torch.zeros((2, 4), dtype=torch.float32))
    finally:
        for hook in v8_hooks:
            hook.remove()
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "multiscale_v8_runtime_routes_return_only_to_medium_branch",
        True,
        bool(
            len(v8_captured_inputs) == 3
            and torch.allclose(v8_captured_inputs[0], v8_level_input)
            and torch.allclose(v8_captured_inputs[1], v8_expected_return)
            and torch.allclose(v8_captured_inputs[2], v8_level_input)
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "multiscale_v4_runtime_branch_widths_match_spec",
        (16, 16, 8),
        tuple(
            int(branch.network[0].conv.out_channels)
            for branch in multiscale_v4_model.branches
        ),
    )
    try:
        build_breakout_quality_model(9, 4, architecture="multiscale_cnn_v2")
        noncanonical_feature_contract_rejected = False
    except ValueError as exc:
        noncanonical_feature_contract_rejected = "canonical 10-column" in str(exc)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "multiscale_v2_rejects_noncanonical_feature_contract",
        True,
        noncanonical_feature_contract_rejected,
    )
    try:
        build_breakout_quality_model(9, 4, architecture="multiscale_cnn_v3")
        v3_noncanonical_feature_contract_rejected = False
    except ValueError as exc:
        v3_noncanonical_feature_contract_rejected = "canonical 10-column" in str(exc)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "multiscale_v3_rejects_noncanonical_feature_contract",
        True,
        v3_noncanonical_feature_contract_rejected,
    )
    try:
        build_breakout_quality_model(9, 4, architecture="multiscale_cnn_v7")
        v7_noncanonical_feature_contract_rejected = False
    except ValueError as exc:
        v7_noncanonical_feature_contract_rejected = "canonical 10-column" in str(exc)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "multiscale_v7_rejects_noncanonical_feature_contract",
        True,
        v7_noncanonical_feature_contract_rejected,
    )
    try:
        build_breakout_quality_model(9, 4, architecture="multiscale_cnn_v8")
        v8_noncanonical_feature_contract_rejected = False
    except ValueError as exc:
        v8_noncanonical_feature_contract_rejected = "canonical 10-column" in str(exc)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "multiscale_v8_rejects_noncanonical_feature_contract",
        True,
        v8_noncanonical_feature_contract_rejected,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "residual_tcn_has_larger_receptive_field_and_parameter_count",
        True,
        get_model_spec("residual_tcn_v1").receptive_field_bars
        > get_model_spec("tiny_cnn_v1").receptive_field_bars
        and residual_parameter_count > tiny_parameter_count,
    )
    legacy_architectures = (
        "tiny_cnn_v1",
        "multiscale_cnn_v1",
        "multiscale_cnn_v2",
        "multiscale_cnn_v3",
        "multiscale_cnn_v4",
        "multiscale_cnn_v5",
        "multiscale_cnn_v6",
        "multiscale_cnn_v7",
        "multiscale_cnn_v8",
        "multiscale_cnn_regime_context_v1",
        "multiscale_cnn_sequence_only_dual_path_v1",
        "inception_time_group_norm_v1",
        INCEPTION_TIME_MARKET_SET_V1,
        INCEPTION_TIME_MARKET_SET_CANDIDATE_V1,
        "modern_tcn_v1",
        "mantis_v2_frozen_linear_v1",
        "moment_1_base_frozen_linear_v1",
        "patch_transformer_v1",
        "ts2vec_frozen_linear_v1",
        "residual_tcn_v1",
    )
    legacy_paths = [
        resolve_filter_artifact_paths(
            "/project",
            "synthetic_quality",
            architecture,
            "baseline",
        )
        for architecture in legacy_architectures
    ]
    legacy_research_paths = [
        resolve_filter_research_score_path(
            "/project",
            "synthetic_quality",
            architecture,
            "baseline",
        )
        for architecture in legacy_architectures
    ]
    modern_paths = resolve_filter_artifact_paths(
        "/project",
        "synthetic_quality",
        "modern_tcn_v1",
        UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE,
    )
    candidate_market_paths = resolve_filter_artifact_paths(
        "/project",
        "synthetic_quality",
        INCEPTION_TIME_MARKET_SET_CANDIDATE_V1,
        UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE,
    )
    baseline_paths = resolve_filter_artifact_paths(
        "/project", "synthetic_quality", "inception_time_v1", "baseline"
    )
    adamw_paths = resolve_filter_artifact_paths(
        "/project", "synthetic_quality", "inception_time_v1", "adamw_only"
    )
    schedule_paths = resolve_filter_artifact_paths(
        "/project",
        "synthetic_quality",
        "inception_time_v1",
        ADAM_WARMUP_COSINE_EXPERIMENT_PROFILE,
    )
    group_norm_paths = resolve_filter_artifact_paths(
        "/project",
        "synthetic_quality",
        "inception_time_group_norm_v1",
        UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE,
    )
    modern_research = resolve_filter_research_score_path(
        "/project",
        "synthetic_quality",
        "modern_tcn_v1",
        UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE,
    )
    candidate_market_research = resolve_filter_research_score_path(
        "/project",
        "synthetic_quality",
        INCEPTION_TIME_MARKET_SET_CANDIDATE_V1,
        UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE,
    )
    baseline_research = resolve_filter_research_score_path(
        "/project", "synthetic_quality", "inception_time_v1", "baseline"
    )
    adamw_research = resolve_filter_research_score_path(
        "/project", "synthetic_quality", "inception_time_v1", "adamw_only"
    )
    schedule_research = resolve_filter_research_score_path(
        "/project",
        "synthetic_quality",
        "inception_time_v1",
        ADAM_WARMUP_COSINE_EXPERIMENT_PROFILE,
    )
    group_norm_research = resolve_filter_research_score_path(
        "/project",
        "synthetic_quality",
        "inception_time_group_norm_v1",
        UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE,
    )
    shared_dataset_dir = resolve_filter_output_dir("/project", "synthetic_quality")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "model_architecture_and_training_experiment_paths_are_separate",
        True,
        (
            len(
                {path.model_path for path in legacy_paths}
                | {
                    modern_paths.model_path,
                    candidate_market_paths.model_path,
                    baseline_paths.model_path,
                    adamw_paths.model_path,
                    schedule_paths.model_path,
                    group_norm_paths.model_path,
                }
            )
            == len(legacy_paths) + 6
            and len(
                set(legacy_research_paths)
                | {
                    modern_research,
                    candidate_market_research,
                    baseline_research,
                    adamw_research,
                    schedule_research,
                    group_norm_research,
                }
            )
            == len(legacy_research_paths) + 6
            and modern_paths.model_architecture == "modern_tcn_v1"
            and candidate_market_paths.model_architecture
            == INCEPTION_TIME_MARKET_SET_CANDIDATE_V1
            and baseline_paths.model_architecture == "inception_time_v1"
            and adamw_paths.model_architecture == "inception_time_v1"
            and group_norm_paths.model_architecture
            == "inception_time_group_norm_v1"
            and modern_paths.experiment_profile
            == UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE
            and candidate_market_paths.experiment_profile
            == UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE
            and baseline_paths.experiment_profile == "baseline"
            and adamw_paths.experiment_profile == "adamw_only"
            and schedule_paths.experiment_profile
            == ADAM_WARMUP_COSINE_EXPERIMENT_PROFILE
            and modern_paths.model_dir.parent.name == "modern_tcn_v1"
            and candidate_market_paths.model_dir.parent.name
            == INCEPTION_TIME_MARKET_SET_CANDIDATE_V1
            and baseline_paths.model_dir.parent.name == "inception_time_v1"
            and adamw_paths.model_dir.parent.name == "inception_time_v1"
            and schedule_paths.model_dir.parent.name == "inception_time_v1"
            and group_norm_paths.model_dir.parent.name
            == "inception_time_group_norm_v1"
            and group_norm_paths.model_dir.name
            == UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE
            and modern_paths.model_dir.name
            == UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE
            and candidate_market_paths.model_dir.name
            == UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE
            and baseline_paths.model_dir.name == "baseline"
            and adamw_paths.model_dir.name == "adamw_only"
            and schedule_paths.model_dir.name
            == ADAM_WARMUP_COSINE_EXPERIMENT_PROFILE
            and shared_dataset_dir.name == "synthetic_quality"
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "only_current_research_architectures_are_active_and_old_architectures_are_legacy",
        (
            (
                "inception_time_v1",
                "multiscale_cnn_sequence_only_v1",
            ),
            set(legacy_architectures),
        ),
        (tuple(ACTIVE_MODEL_ARCHITECTURES), set(LEGACY_MODEL_ARCHITECTURES)),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "training_defaults_are_user_configured_and_legal",
        True,
        int(BREAKOUT_QUALITY_DEFAULT_EPOCHS) >= 1
        and int(BREAKOUT_QUALITY_DEFAULT_BATCH_SIZE) >= 1
        and float(BREAKOUT_QUALITY_DEFAULT_LEARNING_RATE) > 0.0
        and float(BREAKOUT_QUALITY_DEFAULT_WEIGHT_DECAY) >= 0.0
        and float(BREAKOUT_QUALITY_DEFAULT_GRADIENT_CLIP_NORM) >= 0.0
        and int(BREAKOUT_QUALITY_DEFAULT_RANDOM_SEED) >= 0
        and int(BREAKOUT_QUALITY_EVALUATION_BATCH_SIZE) >= 1
        and int(BREAKOUT_QUALITY_EVALUATION_WORKERS) >= 1
        and isinstance(BREAKOUT_QUALITY_PARALLEL_SPLIT_EVALUATION, bool)
        and int(BREAKOUT_QUALITY_TRAIN_PREFETCH_BATCHES) >= 0
        and isinstance(BREAKOUT_QUALITY_PRELOAD_FEATURE_BANK, bool)
        and int(BREAKOUT_QUALITY_MIN_TRAIN_SAMPLES) >= 1
        and BREAKOUT_QUALITY_FINAL_REFIT_MODE in {"matched_optimizer_steps", "selected_epochs"}
        and BREAKOUT_QUALITY_CLASS_WEIGHT_MODE in {"none", "inverse_frequency"}
        and BREAKOUT_QUALITY_TIME_WEIGHT_MODE
        in {"none", "year_balanced_sqrt", TIME_WEIGHT_MODE_DATE_BALANCED}
        and {"adam", "adamw"}.issubset(set(SUPPORTED_BREAKOUT_QUALITY_OPTIMIZERS))
        and {
            BASELINE_EXPERIMENT_PROFILE,
            ADAMW_ONLY_EXPERIMENT_PROFILE,
            ADAM_WARMUP_COSINE_EXPERIMENT_PROFILE,
            HISTORY_MASKING_ONLY_EXPERIMENT_PROFILE,
            UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE,
            UNIQUE_GROUP_DATE_BALANCED_EXPERIMENT_PROFILE,
        }.issubset(set(SUPPORTED_BREAKOUT_QUALITY_EXPERIMENT_PROFILES))
        and LR_SCHEDULE_LINEAR_WARMUP_COSINE
        in SUPPORTED_BREAKOUT_QUALITY_LR_SCHEDULES
        and BREAKOUT_QUALITY_EXPERIMENT_PROFILE
        in SUPPORTED_BREAKOUT_QUALITY_EXPERIMENT_PROFILES
        and CONFIGURED_EXPERIMENT.optimizer_name
        in SUPPORTED_BREAKOUT_QUALITY_OPTIMIZERS,
    )
    train_defaults = breakout_quality_train.parse_args([])
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "training_experiment_and_refit_defaults_follow_config",
        (
            BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
            CONFIGURED_EXPERIMENT.optimizer_name,
            CONFIGURED_EXPERIMENT.lr_schedule_name,
            CONFIGURED_EXPERIMENT.augmentation_name,
            CONFIGURED_EXPERIMENT.training_sampling_mode,
            CONFIGURED_EXPERIMENT.training_weight_reduction,
            BREAKOUT_QUALITY_FINAL_REFIT_MODE,
            BREAKOUT_QUALITY_CLASS_WEIGHT_MODE,
            BREAKOUT_QUALITY_TIME_WEIGHT_MODE,
        ),
        (
            str(train_defaults.experiment_profile),
            str(train_defaults.optimizer_name),
            str(train_defaults.lr_schedule_name),
            str(train_defaults.augmentation_name),
            str(CONFIGURED_EXPERIMENT.training_sampling_mode),
            str(train_defaults.training_weight_reduction),
            str(train_defaults.final_refit_mode),
            str(train_defaults.class_weight_mode),
            str(train_defaults.time_weight_mode),
        ),
    )
    configured_augmentation_parameters = CONFIGURED_EXPERIMENT.augmentation_parameters()
    configured_augmentation_plan = build_training_augmentation_plan(
        name=CONFIGURED_EXPERIMENT.augmentation_name,
        parameters=configured_augmentation_parameters,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "configured_augmentation_profile_round_trips_parameters",
        CONFIGURED_EXPERIMENT.as_manifest_payload().get(
            "augmentation_parameters", {}
        ),
        configured_augmentation_plan.as_parameters(),
    )
    masking_profile = get_breakout_quality_experiment_profile(
        HISTORY_MASKING_ONLY_EXPERIMENT_PROFILE
    )
    masking_plan = build_training_augmentation_plan(
        name=masking_profile.augmentation_name,
        parameters=masking_profile.augmentation_parameters(),
    )
    augmentation_source = np.random.default_rng(1234).normal(
        size=(4, 300, 10)
    ).astype(np.float32)
    augmentation_before = augmentation_source.copy()
    augmented_a, augmentation_summary_a = apply_training_augmentation(
        augmentation_source,
        plan=masking_plan,
        rng=np.random.default_rng(99),
    )
    augmented_b, augmentation_summary_b = apply_training_augmentation(
        augmentation_source,
        plan=masking_plan,
        rng=np.random.default_rng(99),
    )
    changed_time_masks = np.any(augmented_a != augmentation_source, axis=2)
    changed_lengths = [int(mask.sum()) for mask in changed_time_masks if bool(mask.any())]
    changed_are_contiguous = all(
        np.array_equal(
            np.flatnonzero(mask),
            np.arange(np.flatnonzero(mask)[0], np.flatnonzero(mask)[-1] + 1),
        )
        for mask in changed_time_masks
        if bool(mask.any())
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "history_masking_is_deterministic_training_only_and_protects_recent_bars",
        True,
        np.array_equal(augmentation_source, augmentation_before)
        and np.array_equal(augmented_a, augmented_b)
        and augmentation_summary_a == augmentation_summary_b
        and np.array_equal(augmented_a[:, -60:, :], augmentation_source[:, -60:, :])
        and np.array_equal(augmented_a[:, 0, :], augmentation_source[:, 0, :])
        and changed_are_contiguous
        and all(10 <= value <= 30 for value in changed_lengths)
        and int(augmentation_summary_a["augmented_sample_count"]) == len(changed_lengths),
    )
    torch, _nn = breakout_quality_train.require_torch()
    optimizer_probe_model = build_breakout_quality_model(
        10, 4, architecture="multiscale_cnn_v1"
    )
    optimizer_probe = breakout_quality_train._build_optimizer(
        torch,
        optimizer_name=CONFIGURED_EXPERIMENT.optimizer_name,
        parameters=optimizer_probe_model.parameters(),
        learning_rate=BREAKOUT_QUALITY_DEFAULT_LEARNING_RATE,
        weight_decay=BREAKOUT_QUALITY_DEFAULT_WEIGHT_DECAY,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "configured_experiment_profile_uses_requested_optimizer_class",
        "AdamW" if CONFIGURED_EXPERIMENT.optimizer_name == "adamw" else "Adam",
        optimizer_probe.__class__.__name__,
    )

    schedule_profile = get_breakout_quality_experiment_profile(
        ADAM_WARMUP_COSINE_EXPERIMENT_PROFILE
    )
    schedule_plan = breakout_quality_train._build_learning_rate_schedule_plan(
        schedule_name=schedule_profile.lr_schedule_name,
        base_learning_rate=0.0003,
        total_optimizer_steps=100,
        warmup_fraction=schedule_profile.lr_warmup_fraction,
        minimum_lr_ratio=schedule_profile.lr_minimum_ratio,
    )
    schedule_values = [
        breakout_quality_train._learning_rate_for_optimizer_step(schedule_plan, step)
        for step in range(100)
    ]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "step_lr_schedule_uses_exact_warmup_and_cosine_endpoints",
        True,
        (
            schedule_plan["warmup_steps"] == 5
            and np.isclose(schedule_values[0], 0.00006)
            and np.isclose(schedule_values[4], 0.0003)
            and np.isclose(schedule_values[5], 0.0003)
            and np.isclose(schedule_values[-1], 0.00003)
            and all(
                schedule_values[index] >= schedule_values[index + 1]
                for index in range(4, len(schedule_values) - 1)
            )
        ),
    )
    no_schedule_plan = breakout_quality_train._build_learning_rate_schedule_plan(
        schedule_name="none",
        base_learning_rate=0.0003,
        total_optimizer_steps=100,
        warmup_fraction=0.0,
        minimum_lr_ratio=1.0,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "baseline_lr_schedule_remains_constant",
        True,
        all(
            np.isclose(
                breakout_quality_train._learning_rate_for_optimizer_step(
                    no_schedule_plan,
                    step,
                ),
                0.0003,
            )
            for step in (0, 49, 99)
        ),
    )

    matched_target, minimum_pass = breakout_quality_train._resolve_final_refit_target_steps(
        mode="matched_optimizer_steps",
        selected_epoch=2,
        selected_optimizer_steps=8422,
        final_batches_per_epoch=5701,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "matched_refit_preserves_selected_optimizer_steps",
        (8422, False),
        (matched_target, minimum_pass),
    )
    minimum_target, minimum_pass = breakout_quality_train._resolve_final_refit_target_steps(
        mode="matched_optimizer_steps",
        selected_epoch=1,
        selected_optimizer_steps=4211,
        final_batches_per_epoch=5701,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "matched_refit_still_uses_all_selection_rows_once",
        (5701, True),
        (minimum_target, minimum_pass),
    )
    synthetic_weight_events = pd.DataFrame(
        {
            "ticker": ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E"],
            "date": [
                "2020-01-02", "2020-01-02",
                "2020-02-03", "2020-02-03",
                "2020-03-04", "2020-03-04",
                "2020-04-05", "2020-04-05",
                "2021-01-06", "2021-01-06",
            ],
            "high_len": [60, 65] * 5,
        }
    )
    weight_indices = np.arange(len(synthetic_weight_events), dtype=np.int64)
    year_weights, year_summary = breakout_quality_train._time_weighted_group_weights(
        synthetic_weight_events,
        weight_indices,
        mode="year_balanced_sqrt",
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "year_balanced_weights_keep_group_total_and_soften_year_dominance",
        True,
        bool(
            abs(float(year_weights.sum()) - 5.0) < 1e-6
            and year_summary["year_group_counts"] == {"2020": 4, "2021": 1}
            and year_summary["year_weight_multipliers"]["2021"]
            > year_summary["year_weight_multipliers"]["2020"]
        ),
    )
    date_balance_events = pd.DataFrame(
        {
            "ticker": ["A", "B", "C", "D", "E", "F"],
            "date": [
                "2020-01-02",
                "2020-01-02",
                "2020-01-02",
                "2020-01-03",
                "2020-01-03",
                "2020-01-04",
            ],
            "high_len": [60, 60, 60, 60, 60, 60],
        }
    )
    date_weights, date_summary = breakout_quality_train._time_weighted_group_weights(
        date_balance_events,
        np.arange(len(date_balance_events), dtype=np.int64),
        mode=TIME_WEIGHT_MODE_DATE_BALANCED,
    )
    date_totals = pd.Series(
        date_weights,
        index=date_balance_events["date"],
    ).groupby(level=0).sum()
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "date_balanced_weights_keep_group_total_and_equalize_each_training_date",
        True,
        bool(
            abs(float(date_weights.sum()) - 6.0) < 1e-6
            and date_summary["date_count"] == 3
            and date_summary["date_group_count_min"] == 1
            and date_summary["date_group_count_max"] == 3
            and float(date_totals.max() - date_totals.min()) < 1e-6
        ),
    )
    date_profile = get_breakout_quality_experiment_profile(
        UNIQUE_GROUP_DATE_BALANCED_EXPERIMENT_PROFILE
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "date_balanced_profile_locks_unique_groups_fixed_denominator_and_weight_mode",
        (
            TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
            TIME_WEIGHT_MODE_DATE_BALANCED,
            TRAINING_WEIGHT_REDUCTION_FIXED_BATCH_SIZE,
        ),
        (
            date_profile.training_sampling_mode,
            date_profile.time_weight_mode,
            date_profile.training_weight_reduction,
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "class_weight_none_is_identity",
        [1.0, 1.0],
        breakout_quality_train._class_weights(
            np.asarray([0, 1], dtype=np.int64),
            np.ones((2,), dtype=np.float32),
            mode="none",
        ).tolist(),
    )

    export_defaults = breakout_quality_export_scores.parse_args([])
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "score_export_performance_defaults_share_training_policy",
        (
            int(BREAKOUT_QUALITY_EVALUATION_BATCH_SIZE),
            int(BREAKOUT_QUALITY_EVALUATION_WORKERS),
            bool(BREAKOUT_QUALITY_PRELOAD_FEATURE_BANK),
        ),
        (
            int(export_defaults.inference_batch_size),
            int(export_defaults.inference_workers),
            bool(export_defaults.preload_feature_bank),
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "inner_validation_config_has_legal_types_and_ranges",
        True,
        isinstance(BREAKOUT_QUALITY_USE_INNER_VALIDATION, bool)
        and int(BREAKOUT_QUALITY_INNER_VALIDATION_MONTHS) >= 1
        and int(BREAKOUT_QUALITY_EARLY_STOPPING_PATIENCE) >= 0
        and float(BREAKOUT_QUALITY_EARLY_STOPPING_MIN_DELTA) >= 0.0
        and int(BREAKOUT_QUALITY_MIN_VALIDATION_SAMPLES) >= 1
        and int(BREAKOUT_QUALITY_LABEL_HORIZON_BARS) >= 1
        and int(BREAKOUT_QUALITY_LABEL_PATH_CACHE_BARS) >= int(BREAKOUT_QUALITY_LABEL_HORIZON_BARS)
        and float(BREAKOUT_QUALITY_LABEL_MIN_MFE_RETURN) > 0.0
        and float(BREAKOUT_QUALITY_LABEL_MIN_REWARD_RISK_RATIO) > 1.0
        and -1.0 < float(BREAKOUT_QUALITY_LABEL_MAX_ADVERSE_RETURN) < 0.0,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "dataset_storage_contract_uses_indexed_feature_bank",
        (3, "indexed_feature_bank_npy_v2"),
        (DATASET_STORAGE_SCHEMA_VERSION, DATASET_STORAGE_FORMAT),
    )
    feature_bank = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    indexed_features = IndexedFeatureBank(feature_bank, np.asarray([0, 1, 0], dtype=np.int32))
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "indexed_feature_bank_reuses_ticker_date_sequence",
        True,
        indexed_features.shape == (3, 3, 4)
        and np.array_equal(indexed_features[0], indexed_features[2])
        and np.array_equal(indexed_features[1], feature_bank[1]),
    )
    label_policy = BreakoutQualityLabelPolicy(
        feature_window_bars=2,
        label_horizon_bars=3,
        label_path_cache_bars=5,
        high_len_values=(2,),
        min_mfe_return=0.05,
        min_reward_risk_ratio=1.20,
        max_adverse_return=-0.10,
        benchmark_ticker="0050",
    )

    def _label_case(highs, lows):
        frame = pd.DataFrame(
            {
                "Open": [100.0] * 4,
                "High": [100.0, *highs],
                "Low": [100.0, *lows],
                "Close": [100.0] * 4,
                "Volume": [1000.0] * 4,
            },
            index=pd.date_range("2025-01-01", periods=4, freq="D"),
        )
        return build_event_label(frame, event_pos=0, policy=label_policy)[:2]

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "risk_adjusted_label_passes_low_risk_moderate_gain",
        (LABEL_PASS, "risk_adjusted_opportunity"),
        _label_case([104.0, 106.0, 108.0], [99.0, 98.0, 97.0]),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "risk_adjusted_label_rejects_insufficient_ratio",
        (LABEL_REJECT, "no_risk_adjusted_opportunity"),
        _label_case([104.0, 106.0, 108.0], [96.0, 94.0, 93.0]),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "risk_adjusted_label_rejects_when_downside_limit_hits_first",
        (LABEL_REJECT, "downside_first"),
        _label_case([104.0, 106.0, 108.0], [90.0, 92.0, 93.0]),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "risk_adjusted_label_uses_conservative_same_bar_order",
        (LABEL_REJECT, "same_bar_adverse_first"),
        _label_case([113.0, 114.0, 115.0], [90.0, 92.0, 93.0]),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "risk_adjusted_label_rejects_below_minimum_mfe",
        (LABEL_REJECT, "no_risk_adjusted_opportunity"),
        _label_case([104.0, 104.5, 104.9], [99.0, 99.0, 99.0]),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "risk_adjusted_label_requires_strictly_more_than_minimum_mfe",
        (LABEL_REJECT, "no_risk_adjusted_opportunity"),
        _label_case([105.0, 105.0, 105.0], [100.0, 100.0, 100.0]),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "risk_adjusted_label_handles_zero_mae_without_division_error",
        (LABEL_PASS, "risk_adjusted_opportunity"),
        _label_case([106.0, 107.0, 108.0], [100.0, 100.0, 100.0]),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "risk_adjusted_label_requires_strictly_more_than_ratio_threshold",
        (LABEL_REJECT, "no_risk_adjusted_opportunity"),
        _label_case([106.0, 106.0, 106.0], [95.0, 95.0, 95.0]),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "risk_adjusted_pass_is_not_reversed_by_later_drawdown",
        (LABEL_PASS, "risk_adjusted_opportunity"),
        _label_case([104.0, 106.0, 107.0], [99.0, 98.0, 89.0]),
    )
    invalid_frame = pd.DataFrame(
        {
            "Open": [100.0, 100.0, 100.0],
            "High": [100.0, 104.0, 106.0],
            "Low": [100.0, 99.0, 98.0],
            "Close": [100.0, 100.0, 100.0],
            "Volume": [1000.0, 1000.0, 1000.0],
        },
        index=pd.date_range("2025-02-01", periods=3, freq="D"),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "incomplete_future_path_is_invalid_not_a_third_label",
        (LABEL_INVALID, "insufficient_future"),
        build_event_label(invalid_frame, event_pos=0, policy=label_policy)[:2],
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "label_manifest_declares_risk_adjusted_opportunity_objective",
        LABEL_OBJECTIVE,
        label_policy.label_manifest_payload().get("label_objective"),
    )
    legacy_policy_payload = {
        "label_objective": LEGACY_LABEL_OBJECTIVE,
        "label_horizon_bars": 40,
        "pass_return_threshold": 0.15,
        "reject_return_threshold": -0.07,
    }
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "legacy_label_policy_remains_readable_for_fast_relabel",
        legacy_policy_payload,
        label_manifest_payload_from_policy_manifest(legacy_policy_payload),
    )
    cached_result = label_from_cached_path(
        np.asarray([104.0, 106.0, 108.0, np.nan, np.nan], dtype=np.float64),
        np.asarray([99.0, 98.0, 97.0, np.nan, np.nan], dtype=np.float64),
        anchor_price=100.0,
        available_bars=3,
        policy=label_policy,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "cached_future_path_relabel_matches_risk_adjusted_contract",
        (LABEL_PASS, "risk_adjusted_opportunity", 2.0),
        (cached_result.label, cached_result.reason, cached_result.first_hit_bar),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "label_thresholds_are_separate_from_feature_cache_policy",
        True,
        "min_mfe_return" not in label_policy.feature_cache_manifest_payload()
        and "min_reward_risk_ratio" not in label_policy.feature_cache_manifest_payload()
        and "max_adverse_return" not in label_policy.feature_cache_manifest_payload()
        and label_policy.label_manifest_payload()["min_mfe_return"] == 0.05
        and label_policy.label_manifest_payload()["min_reward_risk_ratio"] == 1.20
        and label_policy.label_manifest_payload()["max_adverse_return"] == -0.10,
    )
    try:
        V16StrategyParams(breakout_quality_filter_id=" ")
        empty_filter_id_rejected = False
    except ValueError as exc:
        empty_filter_id_rejected = "breakout_quality_filter_id" in str(exc)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "empty_filter_id_rejected_without_silent_default",
        True,
        empty_filter_id_rejected,
    )

    torch, nn = breakout_quality_train.require_torch()

    class _DeterministicEvaluationModel(nn.Module):
        def forward(self, x, context):
            pooled = x.mean(dim=(1, 2))
            return torch.stack(
                [pooled + context[:, 0], -pooled + context[:, 1]],
                dim=1,
            )

    evaluation_model = _DeterministicEvaluationModel()
    evaluation_features = np.arange(12 * 3 * 2, dtype=np.float32).reshape(12, 3, 2) / 100.0
    evaluation_context = np.arange(12 * 2, dtype=np.float32).reshape(12, 2) / 50.0
    evaluation_labels = np.asarray([0, 1] * 6, dtype=np.int64)
    evaluation_indices = np.asarray([11, 2, 8, 1, 6, 4, 9, 0, 5], dtype=np.int64)
    evaluation_weights = np.linspace(0.5, 1.5, 12, dtype=np.float32)
    evaluation_class_weights = torch.tensor([1.25, 0.75], dtype=torch.float32)
    one_shot_metrics = breakout_quality_train._evaluate(
        torch,
        evaluation_model,
        evaluation_features,
        evaluation_context,
        evaluation_labels,
        evaluation_indices,
        evaluation_weights,
        evaluation_class_weights,
        evaluation_batch_size=len(evaluation_indices),
        evaluation_workers=1,
    )
    chunked_metrics = breakout_quality_train._evaluate(
        torch,
        evaluation_model,
        evaluation_features,
        evaluation_context,
        evaluation_labels,
        evaluation_indices,
        evaluation_weights,
        evaluation_class_weights,
        evaluation_batch_size=3,
        evaluation_workers=1,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "chunked_full_evaluation_preserves_one_shot_metrics",
        one_shot_metrics,
        chunked_metrics,
    )

    parallel_metrics = breakout_quality_train._evaluate(
        torch,
        evaluation_model,
        evaluation_features,
        evaluation_context,
        evaluation_labels,
        evaluation_indices,
        evaluation_weights,
        evaluation_class_weights,
        evaluation_batch_size=3,
        evaluation_workers=4,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "parallel_chunked_evaluation_preserves_serial_metrics",
        chunked_metrics,
        parallel_metrics,
    )

    torch.manual_seed(20260712)
    actual_evaluation_model = breakout_quality_train.build_model(2, 2, architecture="tiny_cnn_v1")
    state_before_evaluation = {
        key: value.detach().clone()
        for key, value in actual_evaluation_model.state_dict().items()
    }
    actual_serial_metrics = breakout_quality_train._evaluate(
        torch,
        actual_evaluation_model,
        evaluation_features,
        evaluation_context,
        evaluation_labels,
        evaluation_indices,
        evaluation_weights,
        evaluation_class_weights,
        evaluation_batch_size=3,
        evaluation_workers=1,
    )
    actual_parallel_metrics = breakout_quality_train._evaluate(
        torch,
        actual_evaluation_model,
        evaluation_features,
        evaluation_context,
        evaluation_labels,
        evaluation_indices,
        evaluation_weights,
        evaluation_class_weights,
        evaluation_batch_size=3,
        evaluation_workers=4,
    )
    state_after_evaluation = actual_evaluation_model.state_dict()
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "parallel_actual_model_evaluation_preserves_serial_metrics",
        actual_serial_metrics,
        actual_parallel_metrics,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "parallel_evaluation_does_not_mutate_model_state",
        True,
        all(
            torch.equal(state_before_evaluation[key], state_after_evaluation[key])
            for key in state_before_evaluation
        ),
    )

    split_train_indices = evaluation_indices[:6]
    split_validation_indices = evaluation_indices[6:]
    serial_split_metrics = breakout_quality_train._evaluate_inner_splits(
        torch,
        actual_evaluation_model,
        evaluation_features,
        evaluation_context,
        evaluation_labels,
        split_train_indices,
        split_validation_indices,
        evaluation_weights,
        evaluation_class_weights,
        evaluation_batch_size=3,
        evaluation_workers=2,
        parallel=False,
    )
    parallel_split_metrics = breakout_quality_train._evaluate_inner_splits(
        torch,
        actual_evaluation_model,
        evaluation_features,
        evaluation_context,
        evaluation_labels,
        split_train_indices,
        split_validation_indices,
        evaluation_weights,
        evaluation_class_weights,
        evaluation_batch_size=3,
        evaluation_workers=2,
        parallel=True,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "concurrent_train_validation_evaluation_preserves_serial_metrics",
        serial_split_metrics,
        parallel_split_metrics,
    )

    indexed_features = IndexedFeatureBank(
        evaluation_features[:4],
        np.asarray([0, 1, 2, 3, 0, 2, 1, 3, 0, 1, 2, 3], dtype=np.int64),
    )
    preloaded_features, preloaded_context, preloaded_labels = (
        breakout_quality_train._preload_training_arrays(
            indexed_features,
            evaluation_context,
            evaluation_labels.astype(np.int8),
            enabled=True,
        )
    )
    serial_export_logits = strict_parallel_batched_logits(
        torch,
        actual_evaluation_model,
        indexed_features,
        evaluation_context,
        indices=None,
        batch_size=3,
        workers=1,
    )
    parallel_export_logits = strict_parallel_batched_logits(
        torch,
        actual_evaluation_model,
        preloaded_features,
        preloaded_context,
        indices=None,
        batch_size=3,
        workers=4,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "strict_parallel_score_inference_preserves_serial_logits",
        True,
        np.array_equal(serial_export_logits, parallel_export_logits),
    )
    unique_group_logits, event_to_group = strict_unique_group_batched_logits(
        torch,
        actual_evaluation_model,
        preloaded_features,
        preloaded_context,
        batch_size=3,
        workers=4,
    )
    broadcast_logits = unique_group_logits[event_to_group]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "sequence_only_score_export_infers_each_feature_group_once",
        4,
        int(unique_group_logits.shape[0]),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "sequence_only_score_export_broadcasts_bit_identical_group_logits",
        True,
        bool(
            np.array_equal(broadcast_logits[0], broadcast_logits[4])
            and np.array_equal(broadcast_logits[0], broadcast_logits[8])
            and np.array_equal(broadcast_logits[1], broadcast_logits[6])
            and np.array_equal(broadcast_logits[1], broadcast_logits[9])
        ),
    )

    ranking_frame = pd.DataFrame(
        {
            "ticker": ["A", "A", "B", "B"],
            "date": ["2026-01-02"] * 4,
            "label": [LABEL_PASS, LABEL_PASS, LABEL_REJECT, LABEL_REJECT],
            SCORE_COLUMN: [0.6000, 0.6004, 0.4000, 0.4000],
        }
    )
    ranking_score, ranking_truth, ranking_weights, ranking_diagnostics = (
        breakout_quality_evaluate._ranking_inputs(
            ranking_frame,
            group_weighted=True,
        )
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "ranking_metrics_preserve_original_first_event_row_score_with_bounded_noise",
        True,
        bool(np.array_equal(ranking_score, np.asarray([0.6000, 0.4], dtype=np.float64))),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "ranking_metrics_preserve_one_truth_and_weight_per_group",
        ([1.0, 0.0], [1.0, 1.0]),
        (ranking_truth.tolist(), ranking_weights.tolist()),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "ranking_metrics_report_bounded_score_noise_diagnostics",
        {
            "ranking_unit": "ticker_date_group",
            "group_score_reduction": "first_event_row_with_bounded_numerical_noise",
            "group_score_numerical_noise_atol": 0.0005,
            "multirow_group_count": 2,
            "nonidentical_score_group_count": 1,
            "max_within_group_score_span": 0.0004,
        },
        ranking_diagnostics,
    )
    try:
        breakout_quality_evaluate._ranking_inputs(
            ranking_frame,
            group_weighted=True,
            require_identical_group_scores=True,
        )
        sequence_only_noise_rejected = False
    except ValueError as exc:
        sequence_only_noise_rejected = (
            "unique-group inference 精確廣播相同分數" in str(exc)
        )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "sequence_only_ranking_rejects_any_within_group_score_difference",
        True,
        sequence_only_noise_rejected,
    )
    canonical_first_score_frame = ranking_frame.copy()
    canonical_first_score_frame.loc[1, SCORE_COLUMN] = canonical_first_score_frame.loc[0, SCORE_COLUMN]
    canonical_first_score_frame.loc[3, SCORE_COLUMN] = canonical_first_score_frame.loc[2, SCORE_COLUMN]
    noisy_ranking_metrics = breakout_quality_evaluate._ranking_metrics(
        ranking_frame,
        group_weighted=True,
    )
    canonical_ranking_metrics = breakout_quality_evaluate._ranking_metrics(
        canonical_first_score_frame,
        group_weighted=True,
    )
    comparable_ranking_keys = (
        "average_precision_pr_auc",
        "precision_at_coverage",
        "realized_coverage",
        "recall_at_precision_60",
        "coverage_at_precision_60",
        "threshold_at_precision_60_diagnostic_only",
        "brier_score",
        "expected_calibration_error_10_bins",
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "bounded_noise_report_preserves_prefixed_ranking_numbers_exactly",
        True,
        all(
            noisy_ranking_metrics[key] == canonical_ranking_metrics[key]
            for key in comparable_ranking_keys
        ),
    )
    materially_different_ranking_frame = ranking_frame.copy()
    materially_different_ranking_frame.loc[1, SCORE_COLUMN] = 0.6010
    try:
        breakout_quality_evaluate._ranking_inputs(
            materially_different_ranking_frame,
            group_weighted=True,
        )
        material_group_score_difference_rejected = False
    except ValueError as exc:
        material_group_score_difference_rejected = "超過允許的浮點誤差" in str(exc)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "ranking_metrics_reject_material_group_score_difference",
        True,
        material_group_score_difference_rejected,
    )
    mixed_ranking_frame = ranking_frame.copy()
    mixed_ranking_frame.loc[1, "label"] = LABEL_REJECT
    try:
        breakout_quality_evaluate._ranking_inputs(
            mixed_ranking_frame,
            group_weighted=True,
        )
        mixed_ranking_label_rejected = False
    except ValueError as exc:
        mixed_ranking_label_rejected = "混合 label" in str(exc)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "ranking_metrics_reject_mixed_group_labels",
        True,
        mixed_ranking_label_rejected,
    )
    nonfinite_ranking_frame = ranking_frame.copy()
    nonfinite_ranking_frame.loc[1, SCORE_COLUMN] = np.nan
    try:
        breakout_quality_evaluate._ranking_inputs(
            nonfinite_ranking_frame,
            group_weighted=True,
        )
        nonfinite_ranking_score_rejected = False
    except ValueError as exc:
        nonfinite_ranking_score_rejected = "NaN 或 infinite score" in str(exc)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "ranking_metrics_reject_nonfinite_group_scores",
        True,
        nonfinite_ranking_score_rejected,
    )

    preload_probe = np.asarray([11, 0, 7, 4, 2], dtype=np.int64)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "feature_bank_preload_preserves_values_and_row_mapping",
        True,
        np.array_equal(indexed_features[preload_probe], preloaded_features[preload_probe])
        and np.array_equal(evaluation_context, preloaded_context)
        and np.array_equal(evaluation_labels, preloaded_labels),
    )

    with tempfile.TemporaryDirectory(prefix="breakout_quality_source_inventory_") as temp_dir:
        project_root = Path(temp_dir)
        source_dir = project_root / "data" / "tw_stock_data_vip_reduced"
        source_dir.mkdir(parents=True, exist_ok=True)
        first_path = source_dir / "2330.csv"
        second_path = source_dir / "0050.csv"
        first_path.write_text("Date,Open\n2026-01-01,100\n", encoding="utf-8")
        second_path.write_text("Date,Open\n2026-01-01,50\n", encoding="utf-8")

        inventory_before = build_source_data_inventory(project_root, "reduced")
        inventory_repeat = build_source_data_inventory(project_root, "reduced")
        first_path.write_text(
            "Date,Open\n2026-01-01,100\n2026-01-02,101\n",
            encoding="utf-8",
        )
        inventory_after = build_source_data_inventory(project_root, "reduced")

        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "source_inventory_is_stable_without_changes",
            inventory_before,
            inventory_repeat,
        )
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "source_inventory_detects_csv_update",
            True,
            inventory_before["csv_inventory_sha256"] != inventory_after["csv_inventory_sha256"],
        )
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "source_inventory_tracks_unique_csv_count",
            2,
            inventory_after["csv_file_count"],
        )

        output_dir = project_root / "outputs" / "filters" / "breakout_quality" / "synthetic_quality"
        output_dir.mkdir(parents=True, exist_ok=True)
        stale_summary = {
            "filter_id": "synthetic_quality",
            "dataset": "reduced",
            "dataset_storage_schema_version": DATASET_STORAGE_SCHEMA_VERSION,
            "dataset_storage_format": DATASET_STORAGE_FORMAT,
            "policy": DEFAULT_LABEL_POLICY.as_manifest_payload(),
            "label_policy": DEFAULT_LABEL_POLICY.label_manifest_payload(),
            "feature_columns": list(FEATURE_COLUMNS),
            "context_columns": list(CONTEXT_COLUMNS),
            "source_data_inventory": inventory_before,
        }
        (output_dir / "dataset_summary.json").write_text(
            json.dumps(stale_summary),
            encoding="utf-8",
        )
        stale_source_rejected = False
        with (
            patch.object(breakout_quality_common, "PROJECT_ROOT", project_root),
            patch.object(
                breakout_quality_common,
                "dataset_output_dir",
                return_value=output_dir,
            ),
        ):
            try:
                breakout_quality_common.load_validated_dataset_bundle(
                    "synthetic_quality",
                    require_current_source=True,
                )
            except ValueError as exc:
                stale_source_rejected = "來源 CSV 已更新" in str(exc)
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "standalone_training_rejects_stale_source_dataset",
            True,
            stale_source_rejected,
        )

    summary["optimizer_high_len_count"] = len(optimizer_values)
    summary["quality_high_len_count"] = len(quality_values)
    return results, summary


def validate_breakout_quality_chronological_embargo_case(_base_params):
    case_id = "BREAKOUT_QUALITY_CHRONOLOGICAL_EMBARGO"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    date_and_end = (
        ("2024-12-31", "2025-01-02"),
        ("2025-01-10", "2025-01-20"),
        ("2025-02-20", "2025-03-05"),
        ("2025-03-10", "2025-03-20"),
        ("2025-04-20", "2025-05-05"),
        ("2025-05-10", "2025-05-20"),
        ("2025-06-20", "2025-07-05"),
    )
    rows = []
    for event_date, label_end_date in date_and_end:
        for high_len in (100, 105):
            rows.append(
                {
                    "ticker": "2330",
                    "date": event_date,
                    "label_eval_end_date": label_end_date,
                    "high_len": high_len,
                }
            )
    events = pd.DataFrame(rows)
    labels = np.asarray([0, 1] * len(date_and_end), dtype=np.int64)
    outer_policy = {
        "policy_source": "core.walk_forward_policy.synthetic_override",
        "selection_start_date": "2025-01-01",
        "selection_end_date": "2025-04-30",
        "oos_start_date": "2025-05-01",
        "configured_oos_end_date": "2025-06-30",
        "effective_oos_end_date": "2025-06-30",
    }
    outer_policy["policy_fingerprint_sha256"] = compute_outer_policy_fingerprint(
        outer_policy
    )

    (
        assignments_off,
        train_off,
        validation_off,
        refit_off,
        oos_off,
        report_off,
    ) = build_selection_oos_split_assignments(
        events,
        labels,
        outer_policy=outer_policy,
        use_inner_validation=False,
        inner_validation_months=2,
        early_stopping_enabled=False,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "toggle_off_uses_full_selection",
        (6, 0, 6, False, False),
        (
            len(train_off),
            len(validation_off),
            len(refit_off),
            report_off["inner_validation_used"],
            report_off["early_stopping_used"],
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "toggle_off_selection_oos_embargo_rows",
        2,
        report_off["selection_oos_embargo_row_count"],
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "toggle_off_role_counts",
        {
            SELECTION_ROLE_TRAIN: 6,
            SELECTION_ROLE_VALIDATION: 0,
            SELECTION_ROLE_INNER_EMBARGO: 0,
            SELECTION_ROLE_EMBARGO: 2,
            SELECTION_ROLE_INVALID: 0,
            SELECTION_ROLE_NOT_APPLICABLE: 6,
        },
        report_off["selection_role_counts"],
    )

    (
        assignments_on,
        train_on,
        validation_on,
        refit_on,
        oos_on,
        report_on,
    ) = build_selection_oos_split_assignments(
        events,
        labels,
        outer_policy=outer_policy,
        use_inner_validation=True,
        inner_validation_months=2,
        early_stopping_enabled=True,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "toggle_on_period_and_roles",
        ("2025-03-01", 2, 2, 2, 6, True, True),
        (
            report_on["inner_validation_start_date"],
            len(train_on),
            len(validation_on),
            report_on["inner_train_validation_embargo_row_count"],
            len(refit_on),
            report_on["inner_validation_used"],
            report_on["early_stopping_used"],
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "toggle_on_refit_recovers_inner_embargo_rows",
        True,
        set(refit_on.tolist())
        == set(train_on.tolist())
        | set(validation_on.tolist())
        | set(
            np.flatnonzero(
                assignments_on["selection_role"].to_numpy()
                == SELECTION_ROLE_INNER_EMBARGO
            ).tolist()
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "toggle_on_role_counts",
        {
            SELECTION_ROLE_TRAIN: 2,
            SELECTION_ROLE_VALIDATION: 2,
            SELECTION_ROLE_INNER_EMBARGO: 2,
            SELECTION_ROLE_EMBARGO: 2,
            SELECTION_ROLE_INVALID: 0,
            SELECTION_ROLE_NOT_APPLICABLE: 6,
        },
        report_on["selection_role_counts"],
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "inner_train_label_information_before_validation",
        True,
        pd.to_datetime(events.iloc[train_on]["label_eval_end_date"]).max()
        < pd.Timestamp(report_on["inner_validation_start_date"]),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "final_refit_label_information_before_oos",
        True,
        pd.to_datetime(events.iloc[refit_on]["label_eval_end_date"]).max()
        < pd.Timestamp(report_on["oos_start_date"]),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "oos_evaluable_and_tail_rows",
        (2, 2),
        (len(oos_on), report_on["oos_label_after_end_row_count"]),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "outer_and_inner_overlap_forbidden",
        (0, 0, 0, 0),
        (
            report_on["overlap_group_count"],
            report_on["overlap_event_date_count"],
            report_on["inner_train_validation_overlap_group_count"],
            report_on["inner_train_validation_overlap_event_date_count"],
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "outside_selection_has_no_selection_role",
        True,
        bool(
            (
                assignments_on.loc[
                    assignments_on["outer_split"] != OUTER_SPLIT_SELECTION,
                    "selection_role",
                ]
                == SELECTION_ROLE_NOT_APPLICABLE
            ).all()
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "split_assignment_key_unique",
        False,
        bool(
            assignments_on.duplicated(
                ["ticker", "date", "high_len"],
                keep=False,
            ).any()
        ),
    )

    project_root = Path(__file__).resolve().parents[2]
    rolling_fold_policy = resolve_breakout_quality_outer_policy(
        project_root,
        source_data_end_date="2021-12-31",
        environ={
            "V16_WF_SELECTION_START_DATE": "2019-01-01",
            "V16_WF_TRAIN_START_DATE": "2019-01-01",
            "V16_WF_SEARCH_TRAIN_END_DATE": "2020-12-31",
            "V16_WF_OOS_START_DATE": "2021-01-01",
            "V16_WF_OOS_END_DATE": "2021-12-31",
        },
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "rolling_fold_reuses_standard_walk_forward_overrides",
        ("2019-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
        (
            rolling_fold_policy["selection_start_date"],
            rolling_fold_policy["selection_end_date"],
            rolling_fold_policy["oos_start_date"],
            rolling_fold_policy["effective_oos_end_date"],
        ),
    )
    summary["split_report_off"] = report_off
    summary["split_report_on"] = report_on
    return results, summary


def _clear_breakout_quality_caches() -> None:
    load_model_artifact_contract.cache_clear()
    load_runtime_artifact_contract.cache_clear()
    load_split_assignment_frame.cache_clear()
    load_score_table.cache_clear()
    load_shared_group_score_table.cache_clear()



def _validate_signal_runtime_wiring(results, case_id: str) -> None:
    dates = pd.date_range("2025-01-01", periods=5, freq="D")
    frame = pd.DataFrame(
        {
            "Open": [9.0, 9.0, 9.0, 10.5, 10.0],
            "High": [10.0, 10.0, 10.0, 12.0, 12.0],
            "Low": [8.0, 8.0, 8.0, 10.0, 9.0],
            "Close": [9.0, 9.0, 9.0, 11.0, 10.0],
            "Volume": [100.0] * 5,
        },
        index=dates,
    )
    params = SimpleNamespace(
        atr_len=2,
        atr_times_trail=3.0,
        atr_buy_tol=1.0,
        high_len=2,
        use_breakout_buy=True,
        use_breakout_ema_filter=False,
        use_bb=False,
        use_vol=False,
        use_breakout_return_filter=False,
        use_breakout_false_filter=False,
        use_breakout_quality_filter=True,
        breakout_quality_filter_id="synthetic_quality",
        breakout_quality_score_threshold=0.63,
        use_kc=False,
    )
    captured = {}

    def _fake_quality_filter(_df, **kwargs):
        captured.update(kwargs)
        return np.ones(len(_df), dtype=bool)

    with patch("core.signal_utils.build_breakout_quality_filter_pass_condition", side_effect=_fake_quality_filter):
        _atr, buy_condition, _sell_condition, _buy_limits = generate_signals(frame, params, ticker="2330")

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "signal_runtime_receives_exact_crossover_candidates",
        [False, False, False, True, False],
        np.asarray(captured["candidate_condition"], dtype=bool).tolist(),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "signal_runtime_receives_active_threshold",
        0.63,
        float(captured["score_threshold"]),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "quality_filter_is_and_gate_for_breakout_buy",
        [False, False, False, True, False],
        np.asarray(buy_condition, dtype=bool).tolist(),
    )


def _validate_breakout_quality_report_rendering(results, case_id):
    def _metrics(split_name, *, base, precision, acceptance, recall, false_reject):
        total = 100.0
        actual_pass = float(base) * total
        actual_reject = total - actual_pass
        predicted_pass = float(acceptance) * total
        tp = float(recall) * actual_pass
        fn = actual_pass - tp
        fp = predicted_pass - tp
        tn = actual_reject - fp
        specificity = tn / actual_reject
        accuracy = (tp + tn) / total
        group_metrics = {
            "group_count": 100,
            "row_count": 1000,
            "weight_sum": total,
            "base_pass_rate": base,
            "acceptance_rate": acceptance,
            "pass_precision": precision,
            "precision_lift_vs_all_pass": precision / base,
            "pass_recall": recall,
            "false_rejection_rate": false_reject,
            "reject_specificity": specificity,
            "accuracy": accuracy,
            "all_pass_baseline_accuracy": base,
            "avg_score": 0.45,
            "ranking_and_calibration": {
                "average_precision_pr_auc": 0.61,
                "precision_at_coverage": {
                    "0.50": 0.62,
                    "0.60": 0.60,
                    "0.70": 0.58,
                },
                "recall_at_precision_60": 0.55,
                "brier_score": 0.24,
                "expected_calibration_error_10_bins": 0.03,
            },
            "confusion": {
                "true_pass_pred_pass": tp,
                "true_pass_pred_reject": fn,
                "true_reject_pred_pass": fp,
                "true_reject_pred_reject": tn,
            },
        }
        result = {
            "filter_id": "synthetic_quality",
            "split": split_name,
            "selected_date_range": {"start": "2021-01-01", "end": "2022-12-31"},
            "ticker_date_group_weighted": group_metrics,
            "row_level": {"row_count": 1000},
        }
        if split_name == "oos":
            result["yearly_ticker_date_group_weighted"] = [
                {
                    "year": 2021,
                    "date_range": {"start": "2021-01-04", "end": "2021-12-30"},
                    "policy_window": {"start": "2021-01-01", "end": "2021-12-31"},
                    "is_partial_calendar_year": False,
                    "metrics": dict(group_metrics),
                },
                {
                    "year": 2022,
                    "date_range": {"start": "2022-01-03", "end": "2022-06-30"},
                    "policy_window": {"start": "2022-01-01", "end": "2022-06-30"},
                    "is_partial_calendar_year": True,
                    "metrics": {**group_metrics, "group_count": 45, "row_count": 450},
                },
            ]
        return result

    synthetic_report_architecture = "inception_time_v1"
    synthetic_report_model = build_breakout_quality_model(
        10,
        4,
        architecture=synthetic_report_architecture,
    )
    synthetic_report_trainable_parameter_count = count_trainable_parameters(
        synthetic_report_model
    )
    synthetic_report_total_parameter_count = sum(
        int(parameter.numel()) for parameter in synthetic_report_model.parameters()
    )
    context = {
        "filter_id": "synthetic_quality",
        "score_path": Path("research_scores.csv"),
        "model_manifest": {
            "model_architecture": synthetic_report_architecture,
            "experiment_profile": BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
            "experiment_settings": CONFIGURED_EXPERIMENT.as_manifest_payload(),
            "model_spec": get_model_spec(synthetic_report_architecture).as_manifest_payload(),
            "trainable_parameter_count": synthetic_report_trainable_parameter_count,
            "total_parameter_count": synthetic_report_total_parameter_count,
            "frozen_parameter_count": (
                synthetic_report_total_parameter_count
                - synthetic_report_trainable_parameter_count
            ),
            "self_supervised_pretraining": None,
            "sequence_length": int(DEFAULT_LABEL_POLICY.feature_window_bars),
            "training_mode": "inner_validation_epoch_selection_full_refit",
            "inner_validation_used": True,
            "max_epochs": 20,
            "selected_epoch": 2,
            "fixed_evaluation_threshold": 0.5,
            "optimizer_name": CONFIGURED_EXPERIMENT.optimizer_name,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "gradient_clip_norm": 1.0,
            "final_refit_plan": {
                "mode": "matched_optimizer_steps",
                "actual_optimizer_steps": 200,
                "equivalent_epochs": 1.5,
            },
            "class_weight_mode": "none",
            "time_weight_mode": BREAKOUT_QUALITY_TIME_WEIGHT_MODE,
            "training_weight_reduction": CONFIGURED_EXPERIMENT.training_weight_reduction,
            "torch_execution": {
                "requested_device": "cpu",
                "resolved_device": "cpu",
                "mixed_precision_requested": False,
                "mixed_precision_enabled": False,
                "autocast_dtype": "float32",
                "deterministic_algorithms": True,
                "allow_tf32": False,
            },
            "batch_size": 256,
            "seed": 42,
            "epoch_selection_source": "inner_validation_loss",
            "early_stopping_enabled": True,
            "early_stopping_patience": 5,
            "early_stopping_min_delta": 0.0,
            "inner_validation_epoch_selection": {
                "best_epoch": 2,
                "completed_epochs": 3,
                "best_validation_loss": 0.77253,
                "best_validation_metrics": {
                    "accuracy": 0.505773,
                    "pass_rate": 0.344325,
                },
                "history": [
                    {
                        "epoch": 1,
                        "batch_loss": 0.681608,
                        "inner_train_metrics": {
                            "loss": 0.670129,
                            "accuracy": 0.578197,
                            "pass_rate": 0.48634,
                        },
                        "inner_validation_metrics": {
                            "loss": 0.774969,
                            "accuracy": 0.500052,
                            "pass_rate": 0.363297,
                        },
                        "is_best_epoch": True,
                    },
                    {
                        "epoch": 2,
                        "batch_loss": 0.666133,
                        "inner_train_metrics": {
                            "loss": 0.655166,
                            "accuracy": 0.604912,
                            "pass_rate": 0.553797,
                        },
                        "inner_validation_metrics": {
                            "loss": 0.77253,
                            "accuracy": 0.505773,
                            "pass_rate": 0.344325,
                        },
                        "is_best_epoch": True,
                    },
                    {
                        "epoch": 3,
                        "batch_loss": 0.657169,
                        "inner_train_metrics": {
                            "loss": 0.651273,
                            "accuracy": 0.615911,
                            "pass_rate": 0.496689,
                        },
                        "inner_validation_metrics": {
                            "loss": 0.793311,
                            "accuracy": 0.519413,
                            "pass_rate": 0.220611,
                        },
                        "is_best_epoch": False,
                    },
                ],
            },
        },
    }
    payload = build_report_payload(
        metrics_by_split={
            "train": _metrics(
                "train",
                base=0.442189,
                precision=0.569538,
                acceptance=0.368851,
                recall=0.475079,
                false_reject=0.524921,
            ),
            "validation": _metrics(
                "validation",
                base=0.492563,
                precision=0.626719,
                acceptance=0.426307,
                recall=0.542417,
                false_reject=0.457583,
            ),
            "selection": _metrics(
                "selection",
                base=0.456199,
                precision=0.586467,
                acceptance=0.387072,
                recall=0.497601,
                false_reject=0.502399,
            ),
            "oos": _metrics(
                "oos",
                base=0.493898,
                precision=0.460815,
                acceptance=0.169293,
                recall=0.157953,
                false_reject=0.842047,
            ),
        },
        context=context,
    )
    markdown = render_markdown_report(payload)
    console = render_console_summary(payload)
    colored_console = render_console_summary(payload, color=True)

    legacy_ts2vec_architecture = "ts2vec_frozen_linear_v1"
    legacy_ts2vec_model = build_breakout_quality_model(
        10,
        4,
        architecture=legacy_ts2vec_architecture,
    )
    legacy_ts2vec_trainable_parameter_count = count_trainable_parameters(
        legacy_ts2vec_model
    )
    legacy_ts2vec_total_parameter_count = sum(
        int(parameter.numel()) for parameter in legacy_ts2vec_model.parameters()
    )
    legacy_ts2vec_manifest = {
        **context["model_manifest"],
        "model_architecture": legacy_ts2vec_architecture,
        "model_spec": get_model_spec(legacy_ts2vec_architecture).as_manifest_payload(),
        "trainable_parameter_count": legacy_ts2vec_trainable_parameter_count,
        "total_parameter_count": legacy_ts2vec_total_parameter_count,
        "frozen_parameter_count": (
            legacy_ts2vec_total_parameter_count
            - legacy_ts2vec_trainable_parameter_count
        ),
        "self_supervised_pretraining": {
            "manifest": {
                "pretraining_profile": build_breakout_quality_pretraining_profile_payload(
                    BREAKOUT_QUALITY_PRETRAINING_PROFILE
                )
            }
        },
    }
    legacy_ts2vec_payload = build_report_payload(
        metrics_by_split=payload["full_metrics"],
        context={
            **context,
            "model_manifest": legacy_ts2vec_manifest,
        },
    )
    legacy_ts2vec_markdown = render_markdown_report(legacy_ts2vec_payload)
    legacy_ts2vec_console = render_console_summary(legacy_ts2vec_payload)
    no_oos_payload = build_report_payload(
        metrics_by_split={
            split_name: metrics
            for split_name, metrics in payload["full_metrics"].items()
            if split_name != "oos"
        },
        context=context,
    )
    no_oos_console = render_console_summary(no_oos_payload)
    add_check(results, "synthetic_breakout_quality", case_id, "report_oos_fail_status", "FAIL", payload["conclusion"]["status"])
    add_check(results, "synthetic_breakout_quality", case_id, "report_uses_group_weighted_headline", "ticker_date_group_weighted", payload["headline_basis"])
    add_check(results, "synthetic_breakout_quality", case_id, "report_schema_v4", 4, payload["schema_version"])
    add_check(results, "synthetic_breakout_quality", case_id, "report_markdown_header_includes_fixed_training_parameters", True, f"- **Experiment Profile**：`{BREAKOUT_QUALITY_EXPERIMENT_PROFILE}`" in markdown and "- **Pretraining Profile**" not in markdown and "- **Threshold**：`0.5`" in markdown and f"- **Optimizer**：`{CONFIGURED_EXPERIMENT.optimizer_name}`" in markdown and f"- **LR Schedule**：`{CONFIGURED_EXPERIMENT.lr_schedule_name}`" in markdown and f"- **LR Schedule Parameters**：`{CONFIGURED_EXPERIMENT.lr_schedule_parameters() or '-'}`" in markdown and f"- **Augmentation**：`{CONFIGURED_EXPERIMENT.augmentation_name}`" in markdown and f"- **Augmentation Parameters**：`{CONFIGURED_EXPERIMENT.augmentation_parameters() or '-'}`" in markdown and "- **Learning Rate**：`0.001`" in markdown and "- **Weight Decay**：`0.0001`" in markdown and "- **Gradient Clip Norm**：`1.0`" in markdown and "- **Batch Size**：`256`" in markdown and "- **Random Seed**：`42`" in markdown and "## 1. 固定訓練參數" not in markdown)
    add_check(results, "synthetic_breakout_quality", case_id, "report_legacy_ts2vec_markdown_header_includes_pretraining_parameters", True, f"- **Pretraining Profile**：`{BREAKOUT_QUALITY_PRETRAINING_PROFILE}`" in legacy_ts2vec_markdown and "- **Encoder Training**：`Selection-only self-supervised; frozen downstream`" in legacy_ts2vec_markdown and "- **Downstream Head**：`linear`" in legacy_ts2vec_markdown)
    add_check(results, "synthetic_breakout_quality", case_id, "report_markdown_has_epoch_comparison", True, "## 1. Epoch 選擇結果" in markdown and "最終模型" in markdown and "Validation Loss" in markdown)
    selection_matrix = markdown.split("## 2. Selection Confusion Matrix", 1)[1].split("### 分類品質", 1)[0]
    normalized_selection_matrix = selection_matrix.replace("**", "")
    console_selection_matrix = console.split("2. Selection Confusion Matrix", 1)[1].split("分類品質", 1)[0]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "report_confusion_matrix_is_count_only_with_standard_margins",
        True,
        (
            all(token in selection_matrix for token in ("TP =", "FN =", "FP =", "TN =", "模型 PASS", "模型 REJECT", "原始 PASS", "原始 REJECT"))
            and all(token in console_selection_matrix for token in ("TP =", "FN =", "FP =", "TN =", "模型 PASS", "模型 REJECT", "原始 PASS", "原始 REJECT"))
            and all(token not in selection_matrix for token in ("Precision", "Recall", "Specificity", "NPV", "Accuracy"))
            and all(token not in console_selection_matrix for token in ("Precision", "Recall", "Specificity", "NPV", "Accuracy"))
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "report_confusion_margins_use_requested_labels_and_order",
        True,
        (
            all(
                token in normalized_selection_matrix
                for token in (
                    "原始PASS =",
                    "TP + FN =",
                    "原始REJECT =",
                    "FP + TN =",
                    "TP + FP =",
                    "模型PASS =",
                    "FN + TN =",
                    "模型REJECT =",
                )
            )
            and all(
                token in console_selection_matrix
                for token in (
                    "原始PASS =",
                    "TP + FN =",
                    "原始REJECT =",
                    "FP + TN =",
                    "TP + FP =",
                    "模型PASS =",
                    "FN + TN =",
                    "模型REJECT =",
                )
            )
            and normalized_selection_matrix.index("原始PASS =") < normalized_selection_matrix.index("TP + FN =")
            and normalized_selection_matrix.index("原始REJECT =") < normalized_selection_matrix.index("FP + TN =")
            and normalized_selection_matrix.index("TP + FP =") < normalized_selection_matrix.index("模型PASS =")
            and normalized_selection_matrix.index("FN + TN =") < normalized_selection_matrix.index("模型REJECT =")
            and console_selection_matrix.index("原始PASS =") < console_selection_matrix.index("TP + FN =")
            and console_selection_matrix.index("原始REJECT =") < console_selection_matrix.index("FP + TN =")
            and console_selection_matrix.index("TP + FP =") < console_selection_matrix.index("模型PASS =")
            and console_selection_matrix.index("FN + TN =") < console_selection_matrix.index("模型REJECT =")
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "report_metric_tables_include_formulas_and_explanations",
        True,
        (
            "| 指標 | 公式 | 結果 | 解釋 |" in markdown
            and "| 指標 | 公式 | 結果 |" in markdown
            and "TP ÷ (TP + FP)" in markdown
            and "PASS Precision − 原始 PASS" in markdown
            and "被保留的訊號中，有多少真的 PASS" in markdown
            and "篩選行為" not in markdown
            and "篩選行為" not in console
        ),
    )
    add_check(results, "synthetic_breakout_quality", case_id, "report_marks_oos_not_for_retuning", True, "不得使用同一段 OOS 回頭調整" in markdown)
    add_check(results, "synthetic_breakout_quality", case_id, "report_console_has_epoch_and_confusion_tables", True, "1. Epoch 選擇結果" in console and "2. Selection Confusion Matrix" in console and "3. OOS Confusion Matrix" in console and "4. 各資料區段比較" in console and "5. 排序與校準診斷" in console and "6. OOS 年度診斷" in console and "7. OOS 綜合判定" in console and "Inner Train" in console and "Validation*" in console and "Precision" in console)
    add_check(results, "synthetic_breakout_quality", case_id, "report_confusion_omits_redundant_orientation_text", True, "統計口徑：Ticker/Date Group Weighted" not in console and "列 = 原始結果；欄 = 模型判定" not in console and "ticker/date group weighted`；列為原始結果" not in markdown)
    add_check(results, "synthetic_breakout_quality", case_id, "report_header_merges_fixed_training_parameters", True, f"Experiment      : {BREAKOUT_QUALITY_EXPERIMENT_PROFILE}" in console and "Pretrain Profile :" not in console and "Threshold       : 0.5" in console and f"Optimizer       : {CONFIGURED_EXPERIMENT.optimizer_name}" in console and f"LR Schedule     : {CONFIGURED_EXPERIMENT.lr_schedule_name}" in console and f"LR Schedule Args: {CONFIGURED_EXPERIMENT.lr_schedule_parameters() or '-' }" in console and f"Augmentation    : {CONFIGURED_EXPERIMENT.augmentation_name}" in console and f"Augmentation Args: {CONFIGURED_EXPERIMENT.augmentation_parameters() or '-'}" in console and "Learning Rate   : 0.001" in console and "Weight Decay    : 0.0001" in console and "Gradient Clip   : 1.0" in console and "Batch Size      : 256" in console and "Random Seed     : 42" in console and "固定訓練參數" not in console)
    add_check(results, "synthetic_breakout_quality", case_id, "report_legacy_ts2vec_header_includes_pretraining_parameters", True, f"Pretrain Profile : {BREAKOUT_QUALITY_PRETRAINING_PROFILE}" in legacy_ts2vec_console and "Encoder Training : Selection-only SSL; frozen downstream" in legacy_ts2vec_console and "Downstream Head  : linear" in legacy_ts2vec_console)
    add_check(results, "synthetic_breakout_quality", case_id, "report_epoch_summary_uses_bullets", True, "- Epoch 上限：20" in console and "- 最終模型：Inner Validation 選出 Epoch 2" in console and "| Epoch 上限" not in console)
    add_check(results, "synthetic_breakout_quality", case_id, "report_split_and_date_are_separate_columns", True, "區段 / 日期" not in console and "|    區段" in console and "|          日期" in console and "| 區段 | 日期 |" in markdown)
    markdown_section_4 = markdown.split("## 4. 各資料區段比較", 1)[1].split("## 5. 排序與校準診斷", 1)[0]
    markdown_section_5 = markdown.split("## 7. OOS 綜合判定", 1)[1].split("## 指標白話說明", 1)[0]
    console_section_4 = console.split("4. 各資料區段比較", 1)[1].split("5. 排序與校準診斷", 1)[0]
    console_section_5 = console.split("7. OOS 綜合判定", 1)[1]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "report_sections_4_and_5_use_consistent_metric_names",
        True,
        (
            all("模型 PASS" in section for section in (markdown_section_4, markdown_section_5, console_section_4, console_section_5))
            and all("保留率" not in section for section in (markdown_section_4, markdown_section_5, console_section_4, console_section_5))
            and all("原始 PASS" in section for section in (markdown_section_4, markdown_section_5, console_section_4, console_section_5))
            and all("PASS Precision" in section for section in (markdown_section_4, markdown_section_5, console_section_4, console_section_5))
        ),
    )
    compact_section_4_headers = (
        "區段",
        "日期",
        "Groups",
        "原始 PASS",
        "模型 PASS",
        "PASS Precision",
        "Precision 絕對",
        "PASS Recall",
        "平均 Score",
    )
    removed_section_4_headers = (
        "REJECT Specificity",
        "REJECT NPV",
        "Accuracy",
        "Precision 相對",
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "report_section_4_is_compact_and_ordered",
        True,
        (
            "| 區段 | 日期 | Groups | 原始 PASS | 模型 PASS | PASS Precision | Precision 絕對 | PASS Recall | 平均 Score |"
            in markdown_section_4
            and all(label in console_section_4 for label in compact_section_4_headers)
            and all(label not in markdown_section_4 for label in removed_section_4_headers)
            and all(label not in console_section_4 for label in removed_section_4_headers)
            and all(
                markdown_section_4.index(left) < markdown_section_4.index(right)
                for left, right in zip(compact_section_4_headers, compact_section_4_headers[1:])
            )
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "report_oos_assessment_has_three_metric_groups_and_judgements",
        True,
        (
            "| 類別 | 指標 | Selection | OOS | OOS - Selection | 判讀 |" in markdown_section_5
            and all(label in markdown_section_5 for label in ("主要成效", "過度篩選防線", "輔助診斷"))
            and all(label in markdown_section_5 for label in ("PASS Precision", "PASS Recall", "模型 PASS", "REJECT Specificity", "REJECT NPV", "Accuracy", "平均 Score"))
            and "綜合判定" in markdown_section_5
        ),
    )
    add_check(results, "synthetic_breakout_quality", case_id, "report_epoch_selection_is_not_repeated_in_bullets", True, "- 最後選擇：" not in console and "是否選出 Best Epoch" not in console and console.count("- 最終模型：") == 1)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "report_metric_labels_are_consistent_across_tables",
        True,
        (
            all(
                label in markdown and label in console
                for label in (
                    "原始 PASS",
                    "模型 PASS",
                    "PASS Precision",
                    "PASS Recall",
                    "REJECT Specificity",
                    "REJECT NPV",
                    "Accuracy",
                    "Precision 絕對",
                    "Precision 相對",
                    "平均 Score",
                )
            )
            and "模型保留" not in markdown
            and "模型保留" not in console
            and "REJECT 辨識率" not in markdown
            and "REJECT 辨識率" not in console
            and "保留後 Precision" not in markdown
            and "保留後 Precision" not in console
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "report_oos_assessment_merges_comparison_and_deployment",
        True,
        (
            "5. 排序與校準診斷" in console and "6. OOS 年度診斷" in console and "7. OOS 綜合判定" in console
            and "類別" in console
            and "判讀" in console
            and "主要成效" in console
            and "過度篩選防線" in console
            and "輔助診斷" in console
            and "- 最終部署決策：" in console
            and "- 研究限制：" in console
            and "Selection 與 OOS 差異" not in console
            and "最終部署判定" not in console
        ),
    )
    add_check(results, "synthetic_breakout_quality", case_id, "report_header_does_not_duplicate_final_decision", True, "最終判定" not in console.split("1. Epoch 選擇結果", 1)[0] and "部署建議" not in console.split("1. Epoch 選擇結果", 1)[0])
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "report_includes_yearly_oos_classification_ranking_and_partial_marker",
        True,
        (
            "## 6. OOS 年度診斷" in markdown
            and "| 年度 | 日期 | Groups | 原始 PASS | 模型 PASS | PASS Precision | Precision 絕對 | PASS Recall | 平均 Score |" in markdown
            and "| 年度 | PR-AUC | P@50% | P@60% | P@70% | R@P60% | Brier | ECE |" in markdown
            and "2022*" in markdown
            and "6. OOS 年度診斷" in console
            and "年度表只切分同一份固定 OOS score" in console
        ),
    )
    add_check(results, "synthetic_breakout_quality", case_id, "report_without_oos_explains_missing_sections", True, "OOS             : 未納入本次報表" in no_oos_console and "不會輸出 OOS Confusion Matrix 與 Selection/OOS 指標比較" in no_oos_console and "--include-oos" in no_oos_console and "OOS Confusion Matrix" not in no_oos_console.split("注意：", 1)[0] and "Selection 與 OOS 差異" not in no_oos_console.split("注意：", 1)[0])
    add_check(results, "synthetic_breakout_quality", case_id, "report_markdown_has_semantic_colors", True, "color:#188038" in markdown and "color:#C62828" in markdown and "color:#42A5F5" in markdown and "color:#B06000" in markdown)
    add_check(results, "synthetic_breakout_quality", case_id, "report_console_color_is_opt_in", True, "\x1b[" not in console and "\x1b[96m" in colored_console)

    colored_matrix = colored_console.split("2. Selection Confusion Matrix", 1)[1].split("分類品質", 1)[0]
    colored_matrix_lines = [line for line in colored_matrix.splitlines() if "|" in line]

    def _matrix_cell(line_token: str, column_index: int) -> str:
        line = next(line for line in colored_matrix_lines if line_token in line)
        return line.split("|")[column_index]

    green = "\x1b[92m"
    red = "\x1b[91m"
    reset = "\x1b[0m"
    tp_label_cell = _matrix_cell("正確保留 PASS", 2)
    fn_label_cell = _matrix_cell("錯殺 PASS", 3)
    tp_value_cell = _matrix_cell("TP =", 2)
    fn_value_cell = _matrix_cell("FN =", 3)
    fp_label_cell = _matrix_cell("錯誤保留 REJECT", 2)
    tn_label_cell = _matrix_cell("正確拒絕 REJECT", 3)
    fp_value_cell = _matrix_cell("FP =", 2)
    tn_value_cell = _matrix_cell("TN =", 3)
    original_pass_margin = _matrix_cell("原始PASS =", 4)
    original_reject_margin = _matrix_cell("原始REJECT =", 4)
    model_pass_margin = _matrix_cell("模型PASS =", 2)
    model_reject_margin = _matrix_cell("模型REJECT =", 3)

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "report_console_confusion_colors_do_not_bleed",
        True,
        (
            all(green in cell and reset in cell and red not in cell for cell in (tp_label_cell, tp_value_cell, tn_label_cell, tn_value_cell))
            and all(red in cell and reset in cell and green not in cell for cell in (fn_label_cell, fn_value_cell, fp_label_cell, fp_value_cell))
            and all(green not in cell and red not in cell for cell in (original_pass_margin, original_reject_margin, model_pass_margin, model_reject_margin))
        ),
    )
    report_markdown_path = resolve_filter_report_markdown_path("/project", "synthetic_quality")
    report_json_path = resolve_filter_report_json_path("/project", "synthetic_quality")
    expected_report_dir_suffix = (
        "outputs",
        "filters",
        "breakout_quality",
        "synthetic_quality",
        DEFAULT_MODEL_ARCHITECTURE,
        BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
        "reports",
    )
    report_paths_are_under_filter_reports = (
        tuple(report_markdown_path.parts[-8:])
        == (*expected_report_dir_suffix, "evaluation_report.md")
        and tuple(report_json_path.parts[-8:])
        == (*expected_report_dir_suffix, "evaluation_metrics.json")
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "report_paths_are_under_filter_reports",
        True,
        report_paths_are_under_filter_reports,
    )
    research_only = build_report_payload(
        metrics_by_split={"selection": _metrics("selection", base=0.45, precision=0.55, acceptance=0.40, recall=0.50, false_reject=0.50)},
        context=context,
    )
    add_check(results, "synthetic_breakout_quality", case_id, "report_without_oos_is_research_only", "RESEARCH_ONLY", research_only["conclusion"]["status"])

    regime_bank = np.zeros((12, 300, len(FEATURE_COLUMNS)), dtype=np.float32)
    benchmark_close_index = FEATURE_COLUMNS.index("benchmark_close_norm")
    for group_index in range(12):
        relative_close = (
            np.linspace(0.70, 1.0, 300)
            if group_index < 6
            else np.linspace(1.30, 1.0, 300)
        )
        relative_close = relative_close * (
            1.0
            + 0.002
            * (group_index % 3)
            * np.sin(np.linspace(0.0, 20.0, 300))
        )
        relative_close = relative_close / relative_close[-1]
        regime_bank[group_index, :, benchmark_close_index] = relative_close - 1.0
    regime_features = derive_benchmark_regime_features(
        regime_bank, np.arange(12, dtype=np.int64)
    )
    regime_groups = pd.DataFrame(
        {
            "ticker": [f"R{index}" for index in range(12)],
            "date": list(pd.date_range("2018-01-01", periods=6, freq="YS"))
            + list(pd.date_range("2024-01-02", periods=6, freq="MS")),
            "group_index": np.arange(12, dtype=np.int64),
            "outer_split": [OUTER_SPLIT_SELECTION] * 6 + [OUTER_SPLIT_OOS] * 6,
            "selection_role": [SELECTION_ROLE_TRAIN] * 6
            + [SELECTION_ROLE_NOT_APPLICABLE] * 6,
            "label": [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
            SCORE_COLUMN: [0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.7, 0.6, 0.8, 0.55, 0.75, 0.65],
        }
    ).merge(regime_features, on="group_index", validate="one_to_one")
    regime_groups, regime_thresholds = assign_market_regimes(regime_groups)
    (
        regime_payload,
        regime_cells,
        regime_focus_year_cells,
        enriched_regime_groups,
    ) = build_regime_audit_payload(
        regime_groups,
        threshold=0.5,
        min_selection_groups=2,
        min_support_share_ratio=0.5,
        focus_year=2024,
        regime_thresholds=regime_thresholds,
        score_group_diagnostics={"group_score_reduction": "synthetic"},
        metadata={
            "filter_id": "synthetic_quality",
            "model_architecture": "inception_time_v1",
            "experiment_profile": BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
        },
    )
    regime_markdown = render_regime_audit_markdown(regime_payload)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "regime_audit_uses_selection_only_volatility_quantiles",
        "selection_only",
        regime_payload["regime_feature_contract"]["thresholds"][
            "volatility_quantile_source"
        ],
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "regime_audit_preserves_one_row_per_ticker_date_group",
        12,
        len(enriched_regime_groups),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "regime_audit_emits_all_dimensions_and_support_fields",
        True,
        (
            set(regime_cells["dimension"])
            == {
                "trend_state",
                "drawdown_state",
                "volatility_state",
                "combined_regime",
            }
            and enriched_regime_groups["selection_support_groups"].notna().all()
            and not regime_focus_year_cells.empty
            and "Selection低代表性" in regime_markdown
            and "2024 Combined regime 歸因" in regime_markdown
            and "Low-support 排除診斷" in regime_markdown
            and "不得依 OOS 結果回頭調整" in regime_markdown
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "regime_audit_emits_focus_year_attribution_and_support_exclusion",
        True,
        (
            regime_payload["focus_year"] == 2024
            and set(
                regime_payload["focus_year_analysis"][
                    "support_exclusion_diagnostic"
                ]
            )
            == {
                "all_focus_year_events",
                "low_support_only",
                "supported_only",
                "supported_only_delta_vs_all",
            }
            and int(regime_focus_year_cells["group_count"].sum()) == 6
        ),
    )

def validate_breakout_quality_runtime_artifact_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_RUNTIME_ARTIFACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    filter_id = "synthetic_quality"
    runtime_fixture_architecture = "inception_time_v1"
    high_len = int(BREAKOUT_DEFAULT_HIGH_LEN)
    with tempfile.TemporaryDirectory(prefix="breakout_quality_contract_") as tmp_dir:
        project_root = Path(tmp_dir)
        models_dir = project_root / "models"
        with (
            patch.dict(
                os.environ,
                {
                    "V16_MODELS_DIR": str(models_dir),
                    "V16_BREAKOUT_QUALITY_SCORE_PATH": str(project_root / "legacy.csv"),
                },
                clear=False,
            ),
            patch(
                "filters.breakout_quality.paths.DEFAULT_MODEL_ARCHITECTURE",
                runtime_fixture_architecture,
            ),
        ):
            paths = resolve_filter_artifact_paths(
                project_root,
                filter_id,
                runtime_fixture_architecture,
                BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
            )
            research_score_path = resolve_filter_research_score_path(project_root, filter_id)
            add_check(
                results,
                "synthetic_breakout_quality",
                case_id,
                "research_output_cannot_replace_canonical_runtime_score",
                True,
                research_score_path != paths.score_path and "outputs" in research_score_path.parts,
            )
            add_check(
                results,
                "synthetic_breakout_quality",
                case_id,
                "path_resolution_has_no_output_side_effect",
                False,
                (project_root / "outputs").exists(),
            )
            legacy_model_dir = (
                models_dir
                / "filters"
                / FILTER_FAMILY
                / filter_id
                / runtime_fixture_architecture
            )
            legacy_model_dir.mkdir(parents=True, exist_ok=True)
            (legacy_model_dir / "manifest.json").write_text("{}", encoding="utf-8")
            legacy_loaded_paths = resolve_existing_filter_artifact_paths(
                project_root,
                filter_id,
                runtime_fixture_architecture,
                BASELINE_EXPERIMENT_PROFILE,
            )
            try:
                breakout_quality_export_scores._resolve_forward_export_write_paths(
                    filter_id=filter_id,
                    experiment_profile=BASELINE_EXPERIMENT_PROFILE,
                    loaded_paths=legacy_loaded_paths,
                    project_root=project_root,
                )
                legacy_forward_write_rejected = False
            except ValueError as exc:
                legacy_forward_write_rejected = "僅供唯讀相容" in str(exc)
            canonical_baseline_paths = resolve_filter_artifact_paths(
                project_root,
                filter_id,
                runtime_fixture_architecture,
                BASELINE_EXPERIMENT_PROFILE,
            )
            canonical_forward_paths = breakout_quality_export_scores._resolve_forward_export_write_paths(
                filter_id=filter_id,
                experiment_profile=BASELINE_EXPERIMENT_PROFILE,
                loaded_paths=canonical_baseline_paths,
                project_root=project_root,
            )
            add_check(
                results,
                "synthetic_breakout_quality",
                case_id,
                "legacy_baseline_fallback_is_read_only_for_forward_export",
                True,
                legacy_forward_write_rejected
                and canonical_forward_paths.model_dir == canonical_baseline_paths.model_dir,
            )
            (legacy_model_dir / "manifest.json").unlink()
            paths.model_dir.mkdir(parents=True, exist_ok=True)
            paths.model_path.write_bytes(b"synthetic-model")
            split_frame = pd.DataFrame(
                [
                    {
                        "ticker": "2330",
                        "date": "2024-01-02",
                        "high_len": high_len,
                        "outer_split": OUTER_SPLIT_SELECTION,
                        "selection_role": SELECTION_ROLE_TRAIN,
                    },
                    {
                        "ticker": "2330",
                        "date": "2024-12-02",
                        "high_len": high_len,
                        "outer_split": OUTER_SPLIT_SELECTION,
                        "selection_role": SELECTION_ROLE_TRAIN,
                    },
                ]
            )
            split_frame.to_csv(paths.split_path, index=False, encoding="utf-8-sig")
            score_frame = pd.DataFrame(
                [
                    {"ticker": "2330", "date": "2025-01-02", "high_len": high_len, SCORE_COLUMN: 0.55},
                    {"ticker": "2330", "date": "2025-01-03", "high_len": high_len, SCORE_COLUMN: 0.60},
                    {"ticker": "2330", "date": "2025-01-04", "high_len": high_len, SCORE_COLUMN: 0.40},
                ]
            )
            score_frame.to_csv(paths.score_path, index=False, encoding="utf-8-sig")
            legacy_path = project_root / "legacy.csv"
            pd.DataFrame(
                [{"ticker": "2330", "date": "2025-01-03", "high_len": high_len, SCORE_COLUMN: 0.0}]
            ).to_csv(legacy_path, index=False)

            score_record = build_file_manifest(paths.score_path)
            score_record.update(
                {
                    "schema_version": SCORE_TABLE_SCHEMA_VERSION,
                    "required_columns": list(SCORE_TABLE_REQUIRED_COLUMNS),
                    "columns": list(SCORE_TABLE_REQUIRED_COLUMNS),
                    "row_count": len(score_frame),
                    "high_len_values": [high_len],
                    "event_date_range": {"start": "2025-01-02", "end": "2025-01-04"},
                }
            )
            split_record = build_file_manifest(paths.split_path)
            split_record.update(
                {
                    "schema_version": SPLIT_ASSIGNMENT_SCHEMA_VERSION,
                    "required_columns": list(SPLIT_ASSIGNMENT_REQUIRED_COLUMNS),
                    "columns": list(SPLIT_ASSIGNMENT_REQUIRED_COLUMNS),
                    "row_count": len(split_frame),
                    "group_key": "ticker/date/high_len",
                    "outer_split_counts": {
                        OUTER_SPLIT_SELECTION: 2,
                        OUTER_SPLIT_OOS: 0,
                        OUTER_SPLIT_OUT_OF_SCOPE: 0,
                    },
                    "selection_role_counts": {
                        SELECTION_ROLE_TRAIN: 2,
                        SELECTION_ROLE_VALIDATION: 0,
                        SELECTION_ROLE_INNER_EMBARGO: 0,
                        SELECTION_ROLE_EMBARGO: 0,
                        SELECTION_ROLE_INVALID: 0,
                        SELECTION_ROLE_NOT_APPLICABLE: 0,
                    },
                }
            )
            outer_policy = {
                "policy_source": "core.walk_forward_policy.synthetic_override",
                "selection_start_date": "2024-01-01",
                "selection_end_date": "2024-12-31",
                "oos_start_date": "2025-01-03",
                "configured_oos_end_date": "2025-12-31",
                "effective_oos_end_date": "2025-12-31",
            }
            outer_policy["policy_fingerprint_sha256"] = compute_outer_policy_fingerprint(outer_policy)
            synthetic_learning_rate = 0.001
            synthetic_schedule_parameters = CONFIGURED_EXPERIMENT.lr_schedule_parameters()

            def _synthetic_schedule_record(*, total_steps: int, actual_steps: int):
                plan = breakout_quality_train._build_learning_rate_schedule_plan(
                    schedule_name=CONFIGURED_EXPERIMENT.lr_schedule_name,
                    base_learning_rate=synthetic_learning_rate,
                    total_optimizer_steps=total_steps,
                    warmup_fraction=CONFIGURED_EXPERIMENT.lr_warmup_fraction,
                    minimum_lr_ratio=CONFIGURED_EXPERIMENT.lr_minimum_ratio,
                )
                return {
                    **plan,
                    "actual_optimizer_steps": int(actual_steps),
                    "last_applied_learning_rate": breakout_quality_train._learning_rate_for_optimizer_step(
                        plan,
                        int(actual_steps) - 1,
                    ),
                }

            def _synthetic_sampling_phase(source_rows: int, group_count: int):
                mode = CONFIGURED_EXPERIMENT.training_sampling_mode
                if mode == TRAINING_SAMPLING_UNIQUE_TICKER_DATE:
                    sampled_rows = int(group_count)
                    return {
                        "mode": mode,
                        "sampling_unit": "unique_ticker_date_group",
                        "batch_size_unit": "unique_ticker_date_groups",
                        "source_row_count": int(source_rows),
                        "sampled_row_count": sampled_rows,
                        "unique_group_count": int(group_count),
                        "duplicate_rows_removed": int(source_rows) - sampled_rows,
                        "representative_rule": "minimum_original_event_row_index",
                        "uses_all_eligible_rows": sampled_rows == int(source_rows),
                        "uses_all_eligible_groups": True,
                    }
                return {
                    "mode": TRAINING_SAMPLING_ALL_EVENT_ROWS,
                    "sampling_unit": "event_row",
                    "batch_size_unit": "event_rows",
                    "source_row_count": int(source_rows),
                    "sampled_row_count": int(source_rows),
                    "unique_group_count": int(group_count),
                    "duplicate_rows_removed": 0,
                    "representative_rule": None,
                    "uses_all_eligible_rows": True,
                    "uses_all_eligible_groups": True,
                }

            fixed_inner_sampling = _synthetic_sampling_phase(2, 2)
            fixed_final_sampling = _synthetic_sampling_phase(2, 2)

            def _synthetic_weight_summary(group_count: int):
                if BREAKOUT_QUALITY_TIME_WEIGHT_MODE == TIME_WEIGHT_MODE_DATE_BALANCED:
                    return {
                        "mode": BREAKOUT_QUALITY_TIME_WEIGHT_MODE,
                        "group_count": int(group_count),
                        "weight_sum": float(group_count),
                        "year_group_counts": {},
                        "year_weight_multipliers": {},
                        "date_count": int(group_count),
                        "date_group_count_min": 1,
                        "date_group_count_max": 1,
                        "date_group_count_mean": 1.0,
                        "date_weight_multiplier_min": 1.0,
                        "date_weight_multiplier_max": 1.0,
                        "target_total_weight_per_date": 1.0,
                        "actual_total_weight_per_date_min": 1.0,
                        "actual_total_weight_per_date_max": 1.0,
                    }
                return {
                    "mode": BREAKOUT_QUALITY_TIME_WEIGHT_MODE,
                    "group_count": int(group_count),
                    "weight_sum": float(group_count),
                    "year_group_counts": {},
                    "year_weight_multipliers": {},
                }

            synthetic_runtime_model = build_breakout_quality_model(
                10, 4, architecture=paths.model_architecture
            )
            synthetic_trainable_parameter_count = count_trainable_parameters(
                synthetic_runtime_model
            )
            synthetic_total_parameter_count = sum(
                int(parameter.numel())
                for parameter in synthetic_runtime_model.parameters()
            )
            synthetic_frozen_parameter_count = (
                synthetic_total_parameter_count
                - synthetic_trainable_parameter_count
            )
            synthetic_pretraining_fingerprint = "a" * 64
            manifest = {
                "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
                "filter_family": FILTER_FAMILY,
                "filter_id": filter_id,
                "model_architecture": paths.model_architecture,
                "experiment_profile": paths.experiment_profile,
                "experiment_settings": get_breakout_quality_experiment_profile(
                    paths.experiment_profile
                ).as_manifest_payload(),
                "model_spec": get_model_spec(paths.model_architecture).as_manifest_payload(),
                "trainable_parameter_count": int(synthetic_trainable_parameter_count),
                "total_parameter_count": int(synthetic_total_parameter_count),
                "frozen_parameter_count": int(synthetic_frozen_parameter_count),
                "self_supervised_pretraining": None,
                "sequence_length": int(DEFAULT_LABEL_POLICY.feature_window_bars),
                "torch_execution": {
                    "requested_device": "cpu",
                    "resolved_device": "cpu",
                    "mixed_precision_requested": False,
                    "mixed_precision_enabled": False,
                    "autocast_dtype": "float32",
                    "deterministic_algorithms": True,
                    "allow_tf32": False,
                },
                "model": build_file_manifest(paths.model_path),
                "split_assignments": split_record,
                "outer_oos_policy": outer_policy,
                "feature_columns": list(FEATURE_COLUMNS),
                "context_columns": list(CONTEXT_COLUMNS),
                "policy": DEFAULT_LABEL_POLICY.as_manifest_payload(),
                "score_decision": {
                    "score_column": SCORE_COLUMN,
                    "comparison": SCORE_COMPARISON,
                    "threshold_source": SCORE_THRESHOLD_SOURCE,
                },
                "fixed_evaluation_threshold": 0.50,
                "threshold_policy": {
                    "mode": "fixed_before_oos",
                    "evaluation_threshold": 0.50,
                    "runtime_source": SCORE_THRESHOLD_SOURCE,
                    "optimized_by_train": False,
                    "oos_tuning_allowed": False,
                },
                "training_mode": "fixed_epoch_full_selection",
                "max_epochs": 2,
                "selected_epoch": 2,
                "fixed_epochs": 2,
                "completed_epochs": 2,
                "epoch_selection_source": "fixed_cli_epochs",
                "final_refit_plan": {
                    "mode": "selected_epochs",
                    "training_sampling_mode": CONFIGURED_EXPERIMENT.training_sampling_mode,
                    "training_weight_reduction": CONFIGURED_EXPERIMENT.training_weight_reduction,
                    "batch_size_unit": fixed_final_sampling["batch_size_unit"],
                    "inner_train_sampling_row_count": int(fixed_inner_sampling["sampled_row_count"]),
                    "final_refit_sampling_row_count": int(fixed_final_sampling["sampled_row_count"]),
                    "selected_optimizer_steps": None,
                    "final_refit_batches_per_epoch": 1,
                    "target_optimizer_steps": 2,
                    "actual_optimizer_steps": 2,
                    "completed_epoch_cycles": 2,
                    "all_eligible_selection_rows_seen_at_least_once": bool(fixed_final_sampling["uses_all_eligible_rows"]),
                    "all_eligible_selection_groups_seen_at_least_once": True,
                },
                "training_sampling": {
                    "mode": CONFIGURED_EXPERIMENT.training_sampling_mode,
                    "inner_train": fixed_inner_sampling,
                    "final_refit": fixed_final_sampling,
                    "validation_sampling_enabled": False,
                    "oos_sampling_enabled": False,
                },
                "optimizer_name": CONFIGURED_EXPERIMENT.optimizer_name,
                "learning_rate": synthetic_learning_rate,
                "lr_schedule_name": CONFIGURED_EXPERIMENT.lr_schedule_name,
                "augmentation_name": CONFIGURED_EXPERIMENT.augmentation_name,
                "training_augmentation": {
                    "name": CONFIGURED_EXPERIMENT.augmentation_name,
                    "parameters": CONFIGURED_EXPERIMENT.augmentation_parameters(),
                    "epoch_selection": None,
                    "final_refit": [
                        {
                            "name": CONFIGURED_EXPERIMENT.augmentation_name,
                            "sample_count": 2,
                            "augmented_sample_count": (
                                1
                                if CONFIGURED_EXPERIMENT.augmentation_name != "none"
                                else 0
                            ),
                            "masked_bar_count": (
                                10
                                if CONFIGURED_EXPERIMENT.augmentation_name != "none"
                                else 0
                            ),
                        }
                        for _ in range(2)
                    ],
                    "validation_augmented": False,
                    "oos_augmented": False,
                },
                "learning_rate_schedule": {
                    "name": CONFIGURED_EXPERIMENT.lr_schedule_name,
                    "parameters": synthetic_schedule_parameters,
                    "epoch_selection": None,
                    "final_refit": _synthetic_schedule_record(
                        total_steps=2,
                        actual_steps=2,
                    ),
                },
                "class_weight_mode": BREAKOUT_QUALITY_CLASS_WEIGHT_MODE,
                "class_weights_reject_pass": [1.0, 1.0],
                "time_weight_mode": BREAKOUT_QUALITY_TIME_WEIGHT_MODE,
                "training_weight_reduction": CONFIGURED_EXPERIMENT.training_weight_reduction,
                "sample_weight_summaries": {
                    "final_refit": _synthetic_weight_summary(2)
                },
                "early_stopping_enabled": False,
                "inner_validation_used": False,
                "training_uses_all_eligible_selection_rows": bool(fixed_final_sampling["uses_all_eligible_rows"]),
                "training_uses_all_eligible_selection_groups": True,
                "oos_predictions_used_during_training": False,
                "oos_metrics_emitted_by_train": False,
                "score_table": score_record,
                "score_inference_execution": {
                    "inference_unit": "unique_ticker_date_feature_group",
                    "shared_group_score_broadcast": True,
                },
                "runtime_eligibility": {
                    "eligible": True,
                    "scope": RUNTIME_SCOPE_FORWARD_OOS,
                    "available_from": "2025-01-02",
                    "available_through": "2025-01-05",
                    "required_signal_start": "2025-01-02",
                    "execution_start": "2025-01-03",
                    "model_information_cutoff": "2025-01-02",
                },
            }
            paths.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            _clear_breakout_quality_caches()

            runtime_contract = load_runtime_artifact_contract(str(project_root), filter_id)
            add_check(
                results,
                "synthetic_breakout_quality",
                case_id,
                "runtime_score_signal_anchor_precedes_oos_execution_start",
                ("2025-01-02", "2025-01-03"),
                (
                    runtime_contract.required_signal_start.isoformat(),
                    runtime_contract.execution_start.isoformat(),
                ),
            )
            anchor_score = lookup_breakout_quality_candidate_score(
                project_root=str(project_root),
                ticker="2330",
                signal_date="2025-01-02",
                high_len=high_len,
                filter_id=filter_id,
            )
            add_check(
                results,
                "synthetic_breakout_quality",
                case_id,
                "pre_execution_signal_anchor_score_is_runtime_available",
                (True, 0.55),
                (bool(anchor_score["available"]), round(float(anchor_score["score"]), 2)),
            )

            original_runtime_eligibility = dict(manifest["runtime_eligibility"])
            manifest["runtime_eligibility"] = {
                **original_runtime_eligibility,
                "required_signal_start": "2025-01-03",
            }
            paths.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            _clear_breakout_quality_caches()
            try:
                load_runtime_artifact_contract(str(project_root), filter_id)
                bad_signal_start_rejected = False
            except ValueError as exc:
                bad_signal_start_rejected = "required_signal_start" in str(exc)
            add_check(
                results,
                "synthetic_breakout_quality",
                case_id,
                "required_signal_start_tamper_fails_fast",
                True,
                bad_signal_start_rejected,
            )

            manifest["runtime_eligibility"] = {
                **original_runtime_eligibility,
                "execution_start": "2025-01-04",
            }
            paths.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            _clear_breakout_quality_caches()
            try:
                load_runtime_artifact_contract(str(project_root), filter_id)
                bad_execution_start_rejected = False
            except ValueError as exc:
                bad_execution_start_rejected = "execution_start" in str(exc)
            add_check(
                results,
                "synthetic_breakout_quality",
                case_id,
                "execution_start_tamper_fails_fast",
                True,
                bad_execution_start_rejected,
            )
            manifest["runtime_eligibility"] = original_runtime_eligibility
            paths.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            _clear_breakout_quality_caches()

            frame = pd.DataFrame(index=pd.to_datetime(["2025-01-01", "2025-01-03", "2025-01-04", "2025-01-05"]))
            candidates = np.asarray([True, True, True, False], dtype=bool)
            pass_at_050 = build_pass_condition_from_score_table(
                frame,
                ticker="2330",
                high_len=high_len,
                score_threshold=0.50,
                candidate_condition=candidates,
                project_root=str(project_root),
                filter_id=filter_id,
            )
            pass_at_070 = build_pass_condition_from_score_table(
                frame,
                ticker="2330",
                high_len=high_len,
                score_threshold=0.70,
                candidate_condition=candidates,
                project_root=str(project_root),
                filter_id=filter_id,
            )
            add_check(results, "synthetic_breakout_quality", case_id, "canonical_score_path_only", [True, True, False, True], pass_at_050.tolist())
            add_check(results, "synthetic_breakout_quality", case_id, "active_threshold_controls_decision", [True, False, False, True], pass_at_070.tolist())

            alternate_high_len = int(high_len) + 5
            shared_only_frame = pd.DataFrame(
                [
                    {
                        "ticker": "2330",
                        "date": "2025-01-02",
                        "high_len": high_len,
                        SCORE_COLUMN: 0.55,
                    },
                    {
                        "ticker": "2330",
                        "date": "2025-01-03",
                        "high_len": alternate_high_len,
                        SCORE_COLUMN: 0.60,
                    },
                    {
                        "ticker": "2330",
                        "date": "2025-01-04",
                        "high_len": high_len,
                        SCORE_COLUMN: 0.40,
                    },
                ]
            )
            shared_only_frame.to_csv(paths.score_path, index=False, encoding="utf-8-sig")
            shared_score_record = build_file_manifest(paths.score_path)
            shared_score_record.update(
                {
                    "schema_version": SCORE_TABLE_SCHEMA_VERSION,
                    "required_columns": list(SCORE_TABLE_REQUIRED_COLUMNS),
                    "columns": list(SCORE_TABLE_REQUIRED_COLUMNS),
                    "row_count": len(shared_only_frame),
                    "high_len_values": [high_len, alternate_high_len],
                    "event_date_range": {"start": "2025-01-02", "end": "2025-01-04"},
                }
            )
            manifest["score_table"] = shared_score_record
            paths.manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            _clear_breakout_quality_caches()
            shared_lookup = build_pass_condition_from_score_table(
                frame,
                ticker="2330",
                high_len=high_len,
                score_threshold=0.50,
                candidate_condition=candidates,
                project_root=str(project_root),
                filter_id=filter_id,
            )
            add_check(
                results,
                "synthetic_breakout_quality",
                case_id,
                "sequence_only_runtime_uses_ticker_date_shared_score",
                [True, True, False, True],
                shared_lookup.tolist(),
            )

            inconsistent_shared_frame = pd.concat(
                [
                    shared_only_frame,
                    pd.DataFrame(
                        [
                            {
                                "ticker": "2330",
                                "date": "2025-01-03",
                                "high_len": alternate_high_len + 5,
                                SCORE_COLUMN: 0.61,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            inconsistent_shared_frame.to_csv(
                paths.score_path,
                index=False,
                encoding="utf-8-sig",
            )
            inconsistent_record = build_file_manifest(paths.score_path)
            inconsistent_record.update(
                {
                    "schema_version": SCORE_TABLE_SCHEMA_VERSION,
                    "required_columns": list(SCORE_TABLE_REQUIRED_COLUMNS),
                    "columns": list(SCORE_TABLE_REQUIRED_COLUMNS),
                    "row_count": len(inconsistent_shared_frame),
                    "high_len_values": [high_len, alternate_high_len, alternate_high_len + 5],
                    "event_date_range": {"start": "2025-01-02", "end": "2025-01-04"},
                }
            )
            manifest["score_table"] = inconsistent_record
            paths.manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            _clear_breakout_quality_caches()
            try:
                build_pass_condition_from_score_table(
                    frame,
                    ticker="2330",
                    high_len=high_len,
                    score_threshold=0.50,
                    candidate_condition=candidates,
                    project_root=str(project_root),
                    filter_id=filter_id,
                )
                inconsistent_shared_rejected = False
            except ValueError as exc:
                inconsistent_shared_rejected = "同一 ticker/date 出現不一致分數" in str(exc)
            add_check(
                results,
                "synthetic_breakout_quality",
                case_id,
                "shared_group_score_inconsistency_fails_fast",
                True,
                inconsistent_shared_rejected,
            )

            shared_only_frame.to_csv(paths.score_path, index=False, encoding="utf-8-sig")
            manifest["score_table"] = shared_score_record
            manifest.pop("score_inference_execution", None)
            paths.manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            _clear_breakout_quality_caches()
            try:
                build_pass_condition_from_score_table(
                    frame,
                    ticker="2330",
                    high_len=high_len,
                    score_threshold=0.50,
                    candidate_condition=candidates,
                    project_root=str(project_root),
                    filter_id=filter_id,
                )
                legacy_exact_key_rejected = False
            except ValueError as exc:
                legacy_exact_key_rejected = "ticker/date/high_len event" in str(exc)
            add_check(
                results,
                "synthetic_breakout_quality",
                case_id,
                "legacy_runtime_keeps_exact_ticker_date_high_len_lookup",
                True,
                legacy_exact_key_rejected,
            )
            manifest["score_inference_execution"] = {
                "inference_unit": "unique_ticker_date_feature_group",
                "shared_group_score_broadcast": True,
            }

            score_frame.to_csv(paths.score_path, index=False, encoding="utf-8-sig")
            manifest["score_table"] = score_record
            paths.manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            _clear_breakout_quality_caches()

            split_frame.loc[1, "selection_role"] = SELECTION_ROLE_VALIDATION
            split_frame.to_csv(paths.split_path, index=False, encoding="utf-8-sig")
            validation_split_record = build_file_manifest(paths.split_path)
            validation_split_record.update(
                {
                    "schema_version": SPLIT_ASSIGNMENT_SCHEMA_VERSION,
                    "required_columns": list(SPLIT_ASSIGNMENT_REQUIRED_COLUMNS),
                    "columns": list(SPLIT_ASSIGNMENT_REQUIRED_COLUMNS),
                    "row_count": len(split_frame),
                    "group_key": "ticker/date/high_len",
                    "outer_split_counts": {
                        OUTER_SPLIT_SELECTION: 2,
                        OUTER_SPLIT_OOS: 0,
                        OUTER_SPLIT_OUT_OF_SCOPE: 0,
                    },
                    "selection_role_counts": {
                        SELECTION_ROLE_TRAIN: 1,
                        SELECTION_ROLE_VALIDATION: 1,
                        SELECTION_ROLE_INNER_EMBARGO: 0,
                        SELECTION_ROLE_EMBARGO: 0,
                        SELECTION_ROLE_INVALID: 0,
                        SELECTION_ROLE_NOT_APPLICABLE: 0,
                    },
                }
            )
            manifest.update(
                {
                    "split_assignments": validation_split_record,
                    "training_mode": TRAINING_MODE_INNER_VALIDATION_FULL_REFIT,
                    "max_epochs": 5,
                    "selected_epoch": 2,
                    "fixed_epochs": 2,
                    "completed_epochs": 2,
                    "final_refit_plan": {
                        "mode": BREAKOUT_QUALITY_FINAL_REFIT_MODE,
                        "training_sampling_mode": CONFIGURED_EXPERIMENT.training_sampling_mode,
                        "batch_size_unit": fixed_final_sampling["batch_size_unit"],
                        "inner_train_sampling_row_count": 1,
                        "final_refit_sampling_row_count": int(fixed_final_sampling["sampled_row_count"]),
                        "selected_optimizer_steps": 2,
                        "final_refit_batches_per_epoch": 1,
                        "target_optimizer_steps": 2,
                        "actual_optimizer_steps": 2,
                        "completed_epoch_cycles": 2,
                        "all_eligible_selection_rows_seen_at_least_once": bool(fixed_final_sampling["uses_all_eligible_rows"]),
                        "all_eligible_selection_groups_seen_at_least_once": True,
                    },
                    "training_sampling": {
                        "mode": CONFIGURED_EXPERIMENT.training_sampling_mode,
                        "inner_train": _synthetic_sampling_phase(1, 1),
                        "final_refit": fixed_final_sampling,
                        "validation_sampling_enabled": False,
                        "oos_sampling_enabled": False,
                    },
                    "epoch_selection_source": "inner_validation_loss",
                    "early_stopping_enabled": True,
                    "early_stopping_patience": 1,
                    "early_stopping_min_delta": 0.0,
                    "inner_validation_used": True,
                    "inner_validation_months": 2,
                    "inner_validation_epoch_selection": {
                        "best_epoch": 2,
                        "completed_epochs": 3,
                        "best_optimizer_steps": 2,
                        "batches_per_epoch": 1,
                    },
                    "training_augmentation": {
                        **manifest["training_augmentation"],
                        "epoch_selection": [
                            {
                                "name": CONFIGURED_EXPERIMENT.augmentation_name,
                                "sample_count": 2,
                                "augmented_sample_count": (
                                    1
                                    if CONFIGURED_EXPERIMENT.augmentation_name != "none"
                                    else 0
                                ),
                                "masked_bar_count": (
                                    10
                                    if CONFIGURED_EXPERIMENT.augmentation_name != "none"
                                    else 0
                                ),
                            }
                            for _ in range(3)
                        ],
                    },
                    "learning_rate_schedule": {
                        "name": CONFIGURED_EXPERIMENT.lr_schedule_name,
                        "parameters": synthetic_schedule_parameters,
                        "epoch_selection": _synthetic_schedule_record(
                            total_steps=5,
                            actual_steps=3,
                        ),
                        "final_refit": _synthetic_schedule_record(
                            total_steps=2,
                            actual_steps=2,
                        ),
                    },
                }
            )
            paths.manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            _clear_breakout_quality_caches()
            validation_contract = load_model_artifact_contract(
                str(project_root),
                filter_id,
            )
            add_check(
                results,
                "synthetic_breakout_quality",
                case_id,
                "inner_validation_model_contract_supported",
                TRAINING_MODE_INNER_VALIDATION_FULL_REFIT,
                validation_contract.manifest["training_mode"],
            )

            legacy_ts2vec_architecture = "ts2vec_frozen_linear_v1"
            legacy_ts2vec_paths = resolve_filter_artifact_paths(
                project_root,
                filter_id,
                legacy_ts2vec_architecture,
                paths.experiment_profile,
            )
            legacy_ts2vec_paths.model_dir.mkdir(parents=True, exist_ok=True)
            legacy_ts2vec_paths.model_path.write_bytes(b"synthetic-ts2vec-model")
            split_frame.to_csv(
                legacy_ts2vec_paths.split_path,
                index=False,
                encoding="utf-8-sig",
            )
            legacy_ts2vec_model = build_breakout_quality_model(
                10,
                4,
                architecture=legacy_ts2vec_architecture,
            )
            legacy_ts2vec_trainable_parameter_count = count_trainable_parameters(
                legacy_ts2vec_model
            )
            legacy_ts2vec_total_parameter_count = sum(
                int(parameter.numel())
                for parameter in legacy_ts2vec_model.parameters()
            )
            legacy_ts2vec_split_record = dict(validation_split_record)
            legacy_ts2vec_split_record.update(
                build_file_manifest(legacy_ts2vec_paths.split_path)
            )
            legacy_ts2vec_manifest = json.loads(
                json.dumps(manifest, ensure_ascii=False)
            )
            legacy_ts2vec_manifest.update(
                {
                    "model_architecture": legacy_ts2vec_architecture,
                    "model_spec": get_model_spec(
                        legacy_ts2vec_architecture
                    ).as_manifest_payload(),
                    "trainable_parameter_count": int(
                        legacy_ts2vec_trainable_parameter_count
                    ),
                    "total_parameter_count": int(
                        legacy_ts2vec_total_parameter_count
                    ),
                    "frozen_parameter_count": int(
                        legacy_ts2vec_total_parameter_count
                        - legacy_ts2vec_trainable_parameter_count
                    ),
                    "self_supervised_pretraining": {
                        "manifest": {
                            "schema_version": 1,
                            "model_architecture": legacy_ts2vec_architecture,
                            "experiment_profile": paths.experiment_profile,
                            "model_spec": get_model_spec(
                                legacy_ts2vec_architecture
                            ).as_manifest_payload(),
                            "pretraining_profile": build_breakout_quality_pretraining_profile_payload(
                                BREAKOUT_QUALITY_PRETRAINING_PROFILE
                            ),
                            "pretraining_dataset_fingerprint": synthetic_pretraining_fingerprint,
                            "oos_windows_used": False,
                            "pass_reject_labels_used": False,
                        },
                        "dataset_summary": {
                            "family": "ts2vec_v1",
                            "stride": 5,
                            "window_count": 100,
                            "selection_start_date": "2024-01-01",
                            "selection_end_date": "2024-12-31",
                            "configuration_fingerprint": synthetic_pretraining_fingerprint,
                        },
                    },
                    "model": build_file_manifest(legacy_ts2vec_paths.model_path),
                    "split_assignments": legacy_ts2vec_split_record,
                }
            )
            legacy_ts2vec_paths.manifest_path.write_text(
                json.dumps(legacy_ts2vec_manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            _clear_breakout_quality_caches()
            legacy_ts2vec_contract = load_model_artifact_contract(
                str(project_root),
                filter_id,
                model_architecture=legacy_ts2vec_architecture,
                experiment_profile=paths.experiment_profile,
            )
            add_check(
                results,
                "synthetic_breakout_quality",
                case_id,
                "legacy_ts2vec_artifact_contract_supported",
                legacy_ts2vec_architecture,
                legacy_ts2vec_contract.manifest["model_architecture"],
            )

            moment_architecture = "moment_1_base_frozen_linear_v1"
            moment_paths = resolve_filter_artifact_paths(
                project_root,
                filter_id,
                moment_architecture,
                paths.experiment_profile,
            )
            moment_paths.model_dir.mkdir(parents=True, exist_ok=True)
            moment_paths.model_path.write_bytes(b"synthetic-moment-model")
            split_frame.to_csv(moment_paths.split_path, index=False, encoding="utf-8-sig")
            moment_split_record = dict(validation_split_record)
            moment_split_record.update(build_file_manifest(moment_paths.split_path))
            moment_manifest = json.loads(json.dumps(manifest, ensure_ascii=False))
            moment_model_spec = get_model_spec(moment_architecture)
            moment_manifest.update(
                {
                    "model_architecture": moment_architecture,
                    "model_spec": moment_model_spec.as_manifest_payload(),
                    "trainable_parameter_count": 15_362,
                    "total_parameter_count": 15_363,
                    "frozen_parameter_count": 1,
                    "self_supervised_pretraining": None,
                    "external_pretrained_encoder": {
                        "source_type": "hugging_face_snapshot",
                        "repository": MOMENT_REPOSITORY,
                        "requested_revision": MOMENT_REVISION,
                        "resolved_revision": MOMENT_REVISION,
                        "checkpoint": {
                            "filename": MOMENT_CHECKPOINT_FILENAME,
                            "sha256": MOMENT_CHECKPOINT_SHA256,
                            "size_bytes": MOMENT_CHECKPOINT_SIZE_BYTES,
                        },
                        "config": {
                            "filename": MOMENT_CONFIG_FILENAME,
                            "sha256": "a" * 64,
                            "size_bytes": MOMENT_CONFIG_SIZE_BYTES,
                            "semantic_validation": "exact_pinned_config",
                        },
                        "package": {
                            "name": MOMENT_PACKAGE_NAME,
                            "version": MOMENT_PACKAGE_VERSION,
                            "expected_version": MOMENT_PACKAGE_VERSION,
                        },
                        "runtime_dependencies": {
                            MOMENT_TRANSFORMERS_PACKAGE_NAME: {
                                "version": MOMENT_TRANSFORMERS_VERSION,
                                "expected_version": MOMENT_TRANSFORMERS_VERSION,
                            }
                        },
                        "model_architecture": moment_architecture,
                        "model_spec": moment_model_spec.as_manifest_payload(),
                        "encoder_frozen_downstream": True,
                        "project_selection_windows_used_for_encoder_training": False,
                        "project_oos_windows_used_for_encoder_training": False,
                        "project_pass_reject_labels_used_for_encoder_training": False,
                        "project_encoder_fine_tuning_used": False,
                        "publisher_pretraining_description": (
                            "Timeseries-PILE masked patch reconstruction pretraining"
                        ),
                    },
                    "model": build_file_manifest(moment_paths.model_path),
                    "split_assignments": moment_split_record,
                }
            )
            moment_paths.manifest_path.write_text(
                json.dumps(moment_manifest, ensure_ascii=False), encoding="utf-8"
            )
            _clear_breakout_quality_caches()
            moment_contract = load_model_artifact_contract(
                str(project_root),
                filter_id,
                model_architecture=moment_architecture,
                experiment_profile=paths.experiment_profile,
            )
            add_check(
                results,
                "synthetic_breakout_quality",
                case_id,
                "moment_external_checkpoint_artifact_contract_supported",
                (moment_architecture, MOMENT_CHECKPOINT_SHA256, True),
                (
                    moment_contract.manifest["model_architecture"],
                    moment_contract.manifest["external_pretrained_encoder"]["checkpoint"]["sha256"],
                    moment_contract.manifest["external_pretrained_encoder"]["encoder_frozen_downstream"],
                ),
            )
            stale_moment_manifest = json.loads(json.dumps(moment_manifest, ensure_ascii=False))
            stale_moment_manifest["external_pretrained_encoder"]["checkpoint"]["sha256"] = "0" * 64
            moment_paths.manifest_path.write_text(
                json.dumps(stale_moment_manifest, ensure_ascii=False), encoding="utf-8"
            )
            _clear_breakout_quality_caches()
            try:
                load_model_artifact_contract(
                    str(project_root),
                    filter_id,
                    model_architecture=moment_architecture,
                    experiment_profile=paths.experiment_profile,
                )
                stale_moment_checkpoint_rejected = False
            except ValueError as exc:
                stale_moment_checkpoint_rejected = "external checkpoint record" in str(exc)
            add_check(
                results,
                "synthetic_breakout_quality",
                case_id,
                "moment_checkpoint_hash_tamper_is_rejected",
                True,
                stale_moment_checkpoint_rejected,
            )
            moment_paths.manifest_path.write_text(
                json.dumps(moment_manifest, ensure_ascii=False), encoding="utf-8"
            )
            _clear_breakout_quality_caches()

            mantis_architecture = "mantis_v2_frozen_linear_v1"
            mantis_paths = resolve_filter_artifact_paths(
                project_root,
                filter_id,
                mantis_architecture,
                paths.experiment_profile,
            )
            mantis_paths.model_dir.mkdir(parents=True, exist_ok=True)
            mantis_paths.model_path.write_bytes(b"synthetic-mantis-model")
            split_frame.to_csv(
                mantis_paths.split_path,
                index=False,
                encoding="utf-8-sig",
            )
            mantis_split_record = dict(validation_split_record)
            mantis_split_record.update(build_file_manifest(mantis_paths.split_path))
            mantis_manifest = json.loads(json.dumps(manifest, ensure_ascii=False))
            mantis_model_spec = get_model_spec(mantis_architecture)
            mantis_manifest.update(
                {
                    "model_architecture": mantis_architecture,
                    "model_spec": mantis_model_spec.as_manifest_payload(),
                    "trainable_parameter_count": 10_242,
                    "total_parameter_count": 10_243,
                    "frozen_parameter_count": 1,
                    "self_supervised_pretraining": None,
                    "external_pretrained_encoder": {
                        "source_type": "hugging_face_snapshot",
                        "repository": MANTIS_V2_REPOSITORY,
                        "requested_revision": MANTIS_V2_REVISION,
                        "resolved_revision": MANTIS_V2_REVISION,
                        "checkpoint": {
                            "filename": MANTIS_V2_CHECKPOINT_FILENAME,
                            "sha256": MANTIS_V2_CHECKPOINT_SHA256,
                            "size_bytes": 16_771_648,
                        },
                        "config": {
                            "filename": MANTIS_V2_CONFIG_FILENAME,
                            "sha256": MANTIS_V2_CONFIG_SHA256,
                            "size_bytes": 375,
                        },
                        "package": {
                            "name": "mantis-tsfm",
                            "version": "1.0.0",
                            "expected_version": "1.0.0",
                        },
                        "model_architecture": mantis_architecture,
                        "model_spec": mantis_model_spec.as_manifest_payload(),
                        "encoder_frozen_downstream": True,
                        "project_selection_windows_used_for_encoder_training": False,
                        "project_oos_windows_used_for_encoder_training": False,
                        "project_pass_reject_labels_used_for_encoder_training": False,
                        "project_encoder_fine_tuning_used": False,
                        "publisher_pretraining_description": (
                            "CauKer-2M synthetic time-series pretraining"
                        ),
                    },
                    "model": build_file_manifest(mantis_paths.model_path),
                    "split_assignments": mantis_split_record,
                }
            )
            mantis_paths.manifest_path.write_text(
                json.dumps(mantis_manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            _clear_breakout_quality_caches()
            mantis_contract = load_model_artifact_contract(
                str(project_root),
                filter_id,
                model_architecture=mantis_architecture,
                experiment_profile=paths.experiment_profile,
            )
            add_check(
                results,
                "synthetic_breakout_quality",
                case_id,
                "mantis_external_checkpoint_artifact_contract_supported",
                (mantis_architecture, MANTIS_V2_CHECKPOINT_SHA256, True),
                (
                    mantis_contract.manifest["model_architecture"],
                    mantis_contract.manifest["external_pretrained_encoder"][
                        "checkpoint"
                    ]["sha256"],
                    mantis_contract.manifest["external_pretrained_encoder"][
                        "encoder_frozen_downstream"
                    ],
                ),
            )

            stale_mantis_manifest = json.loads(
                json.dumps(mantis_manifest, ensure_ascii=False)
            )
            stale_mantis_manifest["external_pretrained_encoder"]["checkpoint"][
                "sha256"
            ] = "0" * 64
            mantis_paths.manifest_path.write_text(
                json.dumps(stale_mantis_manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            _clear_breakout_quality_caches()
            try:
                load_model_artifact_contract(
                    str(project_root),
                    filter_id,
                    model_architecture=mantis_architecture,
                    experiment_profile=paths.experiment_profile,
                )
                stale_mantis_checkpoint_rejected = False
            except ValueError as exc:
                stale_mantis_checkpoint_rejected = (
                    "external checkpoint record" in str(exc)
                )
            add_check(
                results,
                "synthetic_breakout_quality",
                case_id,
                "mantis_checkpoint_hash_tamper_is_rejected",
                True,
                stale_mantis_checkpoint_rejected,
            )
            mantis_paths.manifest_path.write_text(
                json.dumps(mantis_manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            _clear_breakout_quality_caches()

            stale_pretraining_profile_manifest = json.loads(
                json.dumps(legacy_ts2vec_manifest, ensure_ascii=False)
            )
            stale_pretraining_profile_manifest["self_supervised_pretraining"][
                "manifest"
            ]["pretraining_profile"]["mask_probability"] = 0.25
            legacy_ts2vec_paths.manifest_path.write_text(
                json.dumps(stale_pretraining_profile_manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            _clear_breakout_quality_caches()
            try:
                load_model_artifact_contract(
                    str(project_root),
                    filter_id,
                    model_architecture=legacy_ts2vec_architecture,
                    experiment_profile=paths.experiment_profile,
                )
                stale_pretraining_profile_rejected = False
            except ValueError as exc:
                stale_pretraining_profile_rejected = "pretraining_profile" in str(exc)
            add_check(
                results,
                "synthetic_breakout_quality",
                case_id,
                "pretraining_profile_tamper_is_rejected",
                True,
                stale_pretraining_profile_rejected,
            )
            legacy_ts2vec_paths.manifest_path.write_text(
                json.dumps(legacy_ts2vec_manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            _clear_breakout_quality_caches()

            stale_sampling_manifest = json.loads(
                json.dumps(manifest, ensure_ascii=False)
            )
            stale_sampling_manifest["training_sampling"]["final_refit"][
                "representative_rule"
            ] = "unstable_first_seen_row"
            paths.manifest_path.write_text(
                json.dumps(stale_sampling_manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            _clear_breakout_quality_caches()
            try:
                load_model_artifact_contract(str(project_root), filter_id)
                stale_sampling_rejected = False
            except ValueError as exc:
                stale_sampling_rejected = "unique-group 契約不一致" in str(exc)
            add_check(
                results,
                "synthetic_breakout_quality",
                case_id,
                "training_sampling_contract_tamper_is_rejected",
                True,
                stale_sampling_rejected,
            )
            paths.manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            _clear_breakout_quality_caches()

            stale_architecture_manifest = dict(manifest)
            stale_architecture = (
                "tiny_cnn_v1"
                if paths.model_architecture != "tiny_cnn_v1"
                else "residual_tcn_v1"
            )
            stale_architecture_manifest["model_architecture"] = stale_architecture
            stale_architecture_manifest["model_spec"] = get_model_spec(
                stale_architecture
            ).as_manifest_payload()
            paths.manifest_path.write_text(
                json.dumps(stale_architecture_manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            _clear_breakout_quality_caches()
            try:
                load_model_artifact_contract(str(project_root), filter_id)
                stale_architecture_rejected = False
            except ValueError as exc:
                stale_architecture_rejected = (
                    "model_architecture" in str(exc) and "工件路徑" in str(exc)
                )
            add_check(
                results,
                "synthetic_breakout_quality",
                case_id,
                "stale_model_architecture_is_rejected",
                True,
                stale_architecture_rejected,
            )
            paths.manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            _clear_breakout_quality_caches()

            stale_policy_manifest = dict(manifest)
            stale_policy_manifest["policy"] = {
                **DEFAULT_LABEL_POLICY.as_manifest_payload(),
                "min_mfe_return": 0.99,
            }
            paths.manifest_path.write_text(
                json.dumps(stale_policy_manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            _clear_breakout_quality_caches()
            try:
                load_model_artifact_contract(str(project_root), filter_id)
                stale_policy_rejected = False
            except ValueError as exc:
                stale_policy_rejected = "model policy" in str(exc)
            add_check(
                results,
                "synthetic_breakout_quality",
                case_id,
                "stale_model_policy_is_rejected",
                True,
                stale_policy_rejected,
            )
            paths.manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            _clear_breakout_quality_caches()

            missing_candidate = np.asarray([False, False, False, True], dtype=bool)
            try:
                build_pass_condition_from_score_table(
                    frame,
                    ticker="2330",
                    high_len=high_len,
                    score_threshold=0.50,
                    candidate_condition=missing_candidate,
                    project_root=str(project_root),
                    filter_id=filter_id,
                )
                missing_rejected = False
            except ValueError as exc:
                missing_rejected = "缺少正式候選事件" in str(exc)
            add_check(results, "synthetic_breakout_quality", case_id, "missing_candidate_fails_fast", True, missing_rejected)

            stale_frame = pd.DataFrame(index=pd.to_datetime(["2025-01-06"]))
            no_candidate_after_coverage = build_pass_condition_from_score_table(
                stale_frame,
                ticker="2330",
                high_len=high_len,
                score_threshold=0.50,
                candidate_condition=np.asarray([False], dtype=bool),
                project_root=str(project_root),
                filter_id=filter_id,
            )
            add_check(
                results,
                "synthetic_breakout_quality",
                case_id,
                "no_candidate_after_coverage_does_not_require_score",
                [True],
                no_candidate_after_coverage.tolist(),
            )
            try:
                build_pass_condition_from_score_table(
                    stale_frame,
                    ticker="2330",
                    high_len=high_len,
                    score_threshold=0.50,
                    candidate_condition=np.asarray([True], dtype=bool),
                    project_root=str(project_root),
                    filter_id=filter_id,
                )
                stale_rejected = False
            except ValueError as exc:
                stale_rejected = "已過期" in str(exc)
            add_check(results, "synthetic_breakout_quality", case_id, "uncovered_candidate_fails_fast", True, stale_rejected)

            unavailable_path = paths.score_path.with_name(
                DEFAULT_UNAVAILABLE_SCORE_FILENAME
            )
            unavailable_frame = pd.DataFrame(
                [
                    {
                        "ticker": "2330",
                        "date": "2025-01-04",
                        "high_len": high_len,
                        "reason": "benchmark_date_missing",
                    }
                ]
            )
            unavailable_frame.to_csv(
                unavailable_path,
                index=False,
                encoding="utf-8-sig",
            )
            unavailable_record = build_file_manifest(unavailable_path)
            unavailable_record.update(
                {
                    "columns": ["ticker", "date", "high_len", "reason"],
                    "row_count": 1,
                    "reason_counts": {"benchmark_date_missing": 1},
                    "conservative_runtime_score": 0.0,
                    "runtime_action": "reject",
                }
            )
            audit_score_frame = score_frame.copy()
            audit_score_frame.loc[
                audit_score_frame["date"] == "2025-01-04",
                SCORE_COLUMN,
            ] = 0.0
            audit_score_frame.to_csv(
                paths.score_path,
                index=False,
                encoding="utf-8-sig",
            )
            audit_score_record = build_file_manifest(paths.score_path)
            audit_score_record.update(
                {
                    "schema_version": SCORE_TABLE_SCHEMA_VERSION,
                    "required_columns": list(SCORE_TABLE_REQUIRED_COLUMNS),
                    "columns": list(SCORE_TABLE_REQUIRED_COLUMNS),
                    "row_count": len(audit_score_frame),
                    "high_len_values": [high_len],
                    "event_date_range": {
                        "start": "2025-01-02",
                        "end": "2025-01-04",
                    },
                }
            )
            manifest["score_table"] = audit_score_record
            manifest["conservative_unscorable_events"] = unavailable_record
            manifest["score_inference_execution"] = {
                "inference_unit": "unique_ticker_date_feature_group",
                "shared_group_score_broadcast": True,
                "model_scored_event_row_count": 2,
                "conservative_reject_event_row_count": 1,
                "output_event_row_count": 3,
            }
            paths.manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            _clear_breakout_quality_caches()
            audited_scores = load_score_table(str(project_root), filter_id)
            audited_reject_score = float(
                audited_scores.loc[("2330", "2025-01-04", high_len), SCORE_COLUMN]
            )
            add_check(
                results,
                "synthetic_breakout_quality",
                case_id,
                "unscorable_runtime_candidate_is_audited_and_rejected",
                0.0,
                audited_reject_score,
            )

            tampered_audit_score_frame = audit_score_frame.copy()
            tampered_audit_score_frame.loc[
                tampered_audit_score_frame["date"] == "2025-01-04",
                SCORE_COLUMN,
            ] = 0.1
            tampered_audit_score_frame.to_csv(
                paths.score_path,
                index=False,
                encoding="utf-8-sig",
            )
            tampered_audit_score_record = build_file_manifest(paths.score_path)
            tampered_audit_score_record.update(
                {
                    **audit_score_record,
                    **build_file_manifest(paths.score_path),
                }
            )
            manifest["score_table"] = tampered_audit_score_record
            paths.manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False),
                encoding="utf-8",
            )
            _clear_breakout_quality_caches()
            try:
                load_score_table(str(project_root), filter_id)
                nonzero_unscorable_score_rejected = False
            except ValueError as exc:
                nonzero_unscorable_score_rejected = "保守分數為 0.0" in str(exc)
            add_check(
                results,
                "synthetic_breakout_quality",
                case_id,
                "unscorable_runtime_candidate_nonzero_score_is_rejected",
                True,
                nonzero_unscorable_score_rejected,
            )

            score_frame.to_csv(paths.score_path, index=False, encoding="utf-8-sig")
            manifest["score_table"] = score_record
            manifest.pop("conservative_unscorable_events", None)
            manifest["score_inference_execution"] = {
                "inference_unit": "unique_ticker_date_feature_group",
                "shared_group_score_broadcast": True,
            }
            unavailable_path.unlink(missing_ok=True)

            manifest["runtime_eligibility"] = {
                "eligible": False,
                "scope": "research",
                "reason": "synthetic research artifact",
            }
            paths.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
            _clear_breakout_quality_caches()
            try:
                load_runtime_artifact_contract(str(project_root), filter_id)
                research_rejected = False
            except ValueError as exc:
                research_rejected = "不可用於正式 runtime" in str(exc)
            add_check(results, "synthetic_breakout_quality", case_id, "research_artifact_rejected", True, research_rejected)

    _clear_breakout_quality_caches()
    _validate_signal_runtime_wiring(results, case_id)
    _validate_breakout_quality_report_rendering(results, case_id)
    summary["filter_id"] = filter_id
    summary["high_len"] = high_len
    return results, summary



def validate_breakout_quality_continuous_target_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_CONTINUOUS_TARGET"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    spec = StrategyAlignedContinuousTargetSpec.from_label_policy(DEFAULT_LABEL_POLICY)
    contract = spec.contract_payload()
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_target_id_is_versioned",
        STRATEGY_ALIGNED_TARGET_ID,
        contract["target_id"],
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_target_uses_no_split_or_oos_parameters",
        (False, False, "none", "none"),
        (
            contract["split_derived_parameters"],
            contract["oos_derived_parameters"],
            contract["normalization"],
            contract["clipping"],
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_target_inherits_current_horizon_and_risk_budget",
        (
            int(DEFAULT_LABEL_POLICY.label_horizon_bars),
            abs(float(DEFAULT_LABEL_POLICY.max_adverse_return)),
            float(DEFAULT_LABEL_POLICY.min_mfe_return)
            / abs(float(DEFAULT_LABEL_POLICY.max_adverse_return)),
        ),
        (
            int(contract["horizon_bars"]),
            float(contract["risk_budget_return"]),
            float(contract["full_horizon_time_penalty_r"]),
        ),
        tol=1e-12,
    )

    horizon = int(spec.horizon_bars)
    early_high = np.full(horizon, 101.0, dtype=np.float64)
    early_low = np.full(horizon, 98.0, dtype=np.float64)
    early_high[4:] = 120.0
    early = strategy_aligned_target_from_cached_path(
        early_high,
        early_low,
        anchor_price=100.0,
        available_bars=horizon,
        spec=spec,
    )
    risk_budget = float(spec.risk_budget_return)
    expected_early_target = (
        0.20 / risk_budget
        - 0.02 / risk_budget
        - float(spec.full_horizon_time_penalty_r) * (4.0 / float(horizon - 1))
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_target_rewards_fast_favorable_move_net_of_adverse_path",
        (True, 5, 0.20, 0.02, round(expected_early_target, 12)),
        (
            bool(early.valid),
            int(early.opportunity_bar),
            round(float(early.favorable_return), 12),
            round(float(early.adverse_return_to_peak), 12),
            round(float(early.target_raw_r), 12),
        ),
        tol=1e-9,
    )

    late_high = np.linspace(100.0, 105.0, horizon, dtype=np.float64)
    late_low = np.full(horizon, 100.0, dtype=np.float64)
    late = strategy_aligned_target_from_cached_path(
        late_high,
        late_low,
        anchor_price=100.0,
        available_bars=horizon,
        spec=spec,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_target_time_penalty_offsets_minimum_move_at_horizon_end",
        (True, horizon, 0.0),
        (bool(late.valid), int(late.opportunity_bar), round(float(late.target_raw_r), 12)),
        tol=1e-9,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_target_prefers_early_larger_opportunity",
        True,
        float(early.target_raw_r) > float(late.target_raw_r),
    )

    barrier_high = np.full(horizon, 130.0, dtype=np.float64)
    barrier_low = np.full(horizon, 100.0, dtype=np.float64)
    barrier_low[0] = 90.0
    adverse_first = strategy_aligned_target_from_cached_path(
        barrier_high,
        barrier_low,
        anchor_price=100.0,
        available_bars=horizon,
        spec=spec,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_target_same_bar_risk_touch_excludes_high",
        (True, 0.0, 0.10, 1, 1, -1.0),
        (
            bool(adverse_first.valid),
            float(adverse_first.favorable_return),
            float(adverse_first.adverse_return_to_peak),
            int(adverse_first.opportunity_bar),
            int(adverse_first.first_risk_breach_bar),
            float(adverse_first.target_raw_r),
        ),
        tol=1e-12,
    )

    insufficient = strategy_aligned_target_from_cached_path(
        early_high,
        early_low,
        anchor_price=100.0,
        available_bars=horizon - 1,
        spec=spec,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_target_insufficient_future_is_invalid",
        (False, "insufficient_future", True),
        (
            bool(insufficient.valid),
            str(insufficient.reason),
            bool(math.isnan(insufficient.target_raw_r)),
        ),
    )

    anchors = np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float64)
    future_highs = np.stack([early_high, late_high, barrier_high, early_high])
    future_lows = np.stack([early_low, late_low, barrier_low, early_low])
    available = np.array([horizon, horizon, horizon, horizon - 1], dtype=np.int64)
    target_arrays = build_strategy_aligned_group_targets(
        anchors,
        future_highs,
        future_lows,
        available,
        spec=spec,
    )
    repeated_arrays = build_strategy_aligned_group_targets(
        anchors,
        future_highs,
        future_lows,
        available,
        spec=spec,
    )
    deterministic = all(
        np.array_equal(target_arrays[key], repeated_arrays[key], equal_nan=True)
        for key in target_arrays
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_target_group_arrays_are_deterministic_and_group_scoped",
        ((4,), np.dtype(np.float32), np.dtype(np.bool_), [True, True, True, False], True),
        (
            target_arrays["target_raw_r"].shape,
            target_arrays["target_raw_r"].dtype,
            target_arrays["valid_mask"].dtype,
            target_arrays["valid_mask"].tolist(),
            deterministic,
        ),
    )

    dates = pd.to_datetime(
        [
            "2018-01-02", "2018-01-02", "2019-01-02", "2019-01-02",
            "2020-01-02", "2020-01-02", "2021-01-04", "2021-01-04",
        ]
    )
    synthetic_groups = pd.DataFrame(
        {
            "ticker": ["A", "B", "C", "D", "E", "F", "G", "H"],
            "date": dates,
            "label": [LABEL_PASS, LABEL_REJECT] * 4,
            "target_raw_r": [1.5, -0.8, 1.2, -0.6, 1.0, -0.4, 0.8, -0.2],
            "valid_mask": [True] * 8,
            "max_upside_return": [0.20, 0.01, 0.16, 0.02, 0.13, 0.03, 0.10, 0.04],
            "decision_mfe_return": [0.20, 0.01, 0.16, 0.02, 0.13, 0.03, 0.10, 0.04],
            "decision_mae_return": [-0.02, -0.08, -0.02, -0.07, -0.03, -0.06, -0.03, -0.05],
            "is_inner_train": [True, True, True, True, False, False, False, False],
            "is_validation": [False, False, False, False, True, True, False, False],
            "is_selection": [True, True, True, True, True, True, False, False],
            "is_oos": [False, False, False, False, False, False, True, True],
        }
    )
    audit_payload, daily = build_continuous_target_audit(
        synthetic_groups,
        target_contract=contract,
        split_report={"policy": "synthetic_fixed_split"},
        dataset_summary={
            "filter_id": BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
            "dataset": "synthetic",
            "event_count": len(synthetic_groups),
            "feature_group_count": len(synthetic_groups),
        },
        trade_alignment={
            "available": False,
            "reason": "synthetic_no_round_trip_file",
            "formula_tuned_from_trade_r": False,
            "diagnostic_only": True,
        },
    )
    markdown = render_continuous_target_audit_markdown(audit_payload)
    strict_json_ok = True
    try:
        json.dumps(audit_payload, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        strict_json_ok = False
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_target_audit_is_rankable_strict_json_and_audit_only",
        ("IMPLEMENTED_AUDIT_ONLY", False, 1.0, 1.0, 7, True, True),
        (
            audit_payload["status"],
            audit_payload["training_performed"],
            audit_payload["split_metrics"]["oos"]["same_day_rankability"]["rankable_date_rate"],
            audit_payload["split_metrics"]["selection"]["same_day_binary_concordance"]["pair_weighted_concordance"],
            len(daily),
            strict_json_ok,
            "本輪沒有訓練模型" in markdown,
        ),
        tol=1e-12,
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        expected_round_trip_path = (
            resolve_filter_model_output_dir(
                temp_dir,
                BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
                BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
                BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
            )
            / "strategy_compare"
            / "no_filter_round_trips.csv"
        )
        expected_round_trip_path.parent.mkdir(parents=True, exist_ok=True)
        expected_round_trip_path.write_text(
            "ticker,r_multiple\n2330,1.0\n",
            encoding="utf-8",
        )
        with patch(
            "tools.filters.breakout_quality.audit_continuous_target.PROJECT_ROOT",
            Path(temp_dir),
        ):
            resolved_round_trip_path, resolved_path_source = _resolve_round_trip_path(
                BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
                None,
            )
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "continuous_target_round_trip_auto_path_uses_output_tree",
            (str(expected_round_trip_path), "active_9a_standard_path"),
            (str(resolved_round_trip_path), str(resolved_path_source)),
        )

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_target_output_path_is_target_version_scoped",
        (
            "breakout_quality_v1",
            "continuous_targets",
            STRATEGY_ALIGNED_TARGET_ID,
        ),
        tuple(
            resolve_continuous_target_dir(
                Path("/tmp/project"),
                "breakout_quality_v1",
            ).parts[-3:]
        ),
    )

    breakout_quality_app_path = Path(__file__).resolve().parents[2] / "apps" / "breakout_quality.py"
    breakout_quality_app_tree = ast.parse(
        breakout_quality_app_path.read_text(encoding="utf-8"),
        filename=str(breakout_quality_app_path),
    )
    command_modules = {}
    for node in breakout_quality_app_tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "COMMAND_MODULES"
            for target in node.targets
        ):
            continue
        command_modules = ast.literal_eval(node.value)
        break

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_target_audit_command_is_registered",
        "tools.filters.breakout_quality.audit_continuous_target",
        command_modules.get("audit-continuous-target"),
    )

    summary["target_id"] = STRATEGY_ALIGNED_TARGET_ID
    summary["training_performed"] = False
    return results, summary

def validate_breakout_quality_strategy_comparison_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_STRATEGY_COMPARISON"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    base = V16StrategyParams()
    common = {
        "breakout_quality_filter_id": BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
        "breakout_quality_score_threshold": float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD),
    }
    no_filter = replace(base, use_breakout_quality_filter=False, **common)
    quality_filter = replace(base, use_breakout_quality_filter=True, **common)
    try:
        _assert_controlled_param_pair(no_filter, quality_filter)
        single_switch_only = True
    except ValueError:
        single_switch_only = False
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "controlled_pair_only_toggles_quality_filter",
        True,
        single_switch_only,
    )

    try:
        _assert_controlled_param_pair(
            no_filter,
            replace(quality_filter, high_len=int(quality_filter.high_len) + 1),
        )
        extra_difference_rejected = False
    except ValueError as exc:
        extra_difference_rejected = "high_len" in str(exc)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "additional_param_difference_is_rejected",
        True,
        extra_difference_rejected,
    )

    synthetic_period_contract = SimpleNamespace(
        execution_start=pd.Timestamp("2025-01-03").date(),
        required_signal_start=pd.Timestamp("2025-01-02").date(),
        available_from=pd.Timestamp("2025-01-02").date(),
        available_through=pd.Timestamp("2025-12-31").date(),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "strategy_comparison_starts_at_execution_window_not_signal_score_anchor",
        ("2025-01-03", "2025-12-31"),
        _resolve_comparison_period(synthetic_period_contract),
    )

    ranking_pair = _build_controlled_param_source_pair(
        {"kind": "single_param", "params": base},
        filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
        threshold=float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD),
        fixed_risk=None,
        comparison_mode=COMPARISON_MODE_SCORE_RANKING,
    )
    ranking_left = params_to_json_dict(ranking_pair[1])
    ranking_right = params_to_json_dict(ranking_pair[2])
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "score_ranking_pair_only_toggles_ranking_and_keeps_hard_filter_off",
        (False, False, False, True),
        (
            ranking_left["use_breakout_quality_filter"],
            ranking_right["use_breakout_quality_filter"],
            ranking_left["use_breakout_quality_ranking"],
            ranking_right["use_breakout_quality_ranking"],
        ),
    )

    baseline_sort_rows = [
        {"ticker": "A", "sort_value": 0.20, "proj_cost": 100.0, "use_breakout_quality_ranking": False},
        {"ticker": "B", "sort_value": 0.10, "proj_cost": 80.0, "use_breakout_quality_ranking": False},
    ]
    ranking_sort_rows = [
        {"ticker": "A", "sort_value": 0.20, "proj_cost": 100.0, "use_breakout_quality_ranking": True, "breakout_quality_score": 0.90},
        {"ticker": "B", "sort_value": 0.10, "proj_cost": 80.0, "use_breakout_quality_ranking": True, "breakout_quality_score": 0.40},
    ]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "single_param_score_ranking_precedes_existing_overage_sort",
        (["B", "A"], ["A", "B"]),
        (
            [row["ticker"] for row in sort_candidate_rows(baseline_sort_rows, method=BUY_LIMIT_OVERAGE_SORT_METHOD)],
            [row["ticker"] for row in sort_candidate_rows(ranking_sort_rows, method=BUY_LIMIT_OVERAGE_SORT_METHOD)],
        ),
    )

    ensemble_rank_rows = []
    for ticker, votes, score, overage in (("A", 6, 0.20, 0.0), ("B", 5, 0.99, 0.0), ("C", 6, 0.80, 1.0)):
        for member_idx in range(votes):
            ensemble_rank_rows.append({
                "ticker": ticker,
                "ensemble_member_key": f"m{member_idx}",
                "params_obj": base,
                "sort_value": overage,
                "proj_cost": 100.0,
                "use_breakout_quality_ranking": True,
                "breakout_quality_score": score,
                "breakout_quality_score_date": f"2025-01-{member_idx + 2:02d}",
                "breakout_quality_rank": {
                    "score": score,
                    "available": True,
                    "unavailable_reason": "",
                    "score_date": f"2025-01-{member_idx + 2:02d}",
                    "shared_group_score": True,
                    "filter_id": BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
                },
            })
    ensemble_ranked = _aggregate_ensemble_candidate_rows(ensemble_rank_rows, min_agree=3)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "ensemble_votes_remain_first_and_score_only_reorders_equal_vote_candidates",
        [("C", 6), ("A", 6), ("B", 5)],
        [(row["ticker"], row["ensemble_vote_count"]) for row in ensemble_ranked],
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "ensemble_candidate_preserves_member_specific_original_quality_ranks",
        [f"m{idx}" for idx in range(6)],
        sorted(ensemble_ranked[0]["ensemble_member_quality_rank_by_key"]),
    )

    reentry_params = replace(
        base,
        use_breakout_reclaim_reentry=True,
        use_breakout_quality_ranking=True,
        breakout_quality_filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    )
    original_rank = {
        "score": 0.731,
        "available": True,
        "unavailable_reason": "",
        "score_date": "2021-12-30",
        "shared_group_score": True,
        "filter_id": BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    }
    reentry_position = {
        "ticker": "3706",
        "entry_type": "normal",
        "entry_fill_price": 100.0,
        "pure_buy_price": 100.0,
        "initial_stop": 90.0,
        "sl": 90.0,
        "qty": 10,
        "initial_qty": 10,
        "breakout_quality_score": original_rank["score"],
        "breakout_quality_score_date": original_rank["score_date"],
        "breakout_quality_rank": dict(original_rank),
        "use_breakout_quality_ranking": True,
    }
    reentry_watch = create_breakout_reentry_watch_state(
        reentry_position,
        exit_date=pd.Timestamp("2022-01-05"),
        params=reentry_params,
        exit_atr=2.0,
        exit_qty=10,
    )
    reentry_trigger_date = pd.Timestamp("2022-01-17")
    reentry_signal = create_breakout_reentry_signal_state(
        reentry_watch,
        close_price=float(reentry_watch["confirm_price"]),
        atr=2.0,
        params=reentry_params,
        ticker="3706",
        signal_date=reentry_trigger_date,
    )
    inherited_rank = resolve_breakout_quality_rank(reentry_signal)
    with patch(
        "core.portfolio_candidates.resolve_breakout_quality_candidate_rank",
        side_effect=AssertionError("re-entry 不得以確認日重新查 score table"),
    ):
        resolved_reentry_rank = _resolve_candidate_quality_ranking(
            params=reentry_params,
            ticker="3706",
            signal_date=reentry_trigger_date,
            signal_state=reentry_signal,
        )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "reentry_keeps_trigger_date_but_inherits_original_breakout_score_date",
        ("2022-01-17", "2021-12-30", 0.731),
        (
            pd.Timestamp(reentry_signal["signal_date"]).strftime("%Y-%m-%d"),
            inherited_rank["score_date"],
            inherited_rank["score"],
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "reentry_ranking_uses_inherited_score_without_runtime_lookup_on_trigger_date",
        ("2021-12-30", 0.731),
        (resolved_reentry_rank["score_date"], resolved_reentry_rank["score"]),
    )

    incomplete_member_rank_position = {
        "ticker": "3706",
        "use_breakout_quality_ranking": True,
        "breakout_quality_rank": dict(original_rank),
        "_ensemble_member_params_by_key": {"m0": reentry_params, "m1": reentry_params},
        "_ensemble_member_quality_rank_by_key": {"m0": dict(original_rank)},
    }
    try:
        list(_iter_reentry_watch_targets(incomplete_member_rank_position, reentry_params, {}))
        missing_member_rank_rejected = False
    except ValueError as exc:
        missing_member_rank_rejected = "member=m1" in str(exc)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "ensemble_reentry_rejects_partial_member_quality_rank_mapping",
        True,
        missing_member_rank_rejected,
    )

    ensemble_source = build_static_active_param_ensemble_payload(
        members=[
            {"member_index": 1, "seed": 101, "params": params_to_json_dict(base)},
            {"member_index": 2, "seed": 202, "params": params_to_json_dict(replace(base, high_len=205))},
        ],
        random_seed_ensemble={"enabled": True, "seed_count": 2, "min_agree": 2},
        selector="base_finalists_agree",
    )
    (
        ensemble_kind,
        ensemble_no_filter,
        ensemble_quality,
        _ensemble_no_payload,
        _ensemble_quality_payload,
        ensemble_policy,
    ) = _build_controlled_param_source_pair(
        {"kind": "static_active_param_ensemble", "payload": ensemble_source},
        filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
        threshold=float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD),
        fixed_risk=0.01,
    )
    ensemble_switches = [
        (
            bool(left_member["params"]["use_breakout_quality_filter"]),
            bool(right_member["params"]["use_breakout_quality_filter"]),
            int(left_member["params"]["high_len"]),
            int(right_member["params"]["high_len"]),
            float(left_member["params"]["fixed_risk"]),
            float(right_member["params"]["fixed_risk"]),
        )
        for left_member, right_member in zip(
            ensemble_no_filter["params_ensemble"], ensemble_quality["params_ensemble"]
        )
    ]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "static_trade_ensemble_preserves_members_and_only_toggles_filter",
        (
            "static_active_param_ensemble",
            [(False, True, 201, 201, 0.01, 0.01), (False, True, 205, 205, 0.01, 0.01)],
            (2, 2),
        ),
        (
            ensemble_kind,
            ensemble_switches,
            (ensemble_policy["seed_count"], ensemble_policy["min_agree"]),
        ),
    )

    broken_ensemble = json.loads(json.dumps(ensemble_quality))
    broken_ensemble["params_ensemble"][1]["params"]["high_len"] += 1
    try:
        _assert_controlled_ensemble_pair(ensemble_no_filter, broken_ensemble)
        ensemble_extra_difference_rejected = False
    except ValueError as exc:
        ensemble_extra_difference_rejected = "high_len" in str(exc)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "ensemble_additional_member_difference_is_rejected",
        True,
        ensemble_extra_difference_rejected,
    )

    rolling_single_payload = {
        "schema_type": "rolling_oos_param_set",
        "schema_version": 1,
        "usage": "validation_only",
        "summary": {"oos_end_date": "2022-12-31"},
        "params_by_effective_date": {
            "2021-01-01": params_to_json_dict(base),
            "2022-01-01": params_to_json_dict(replace(base, high_len=205)),
        },
        "params_by_oos_year": {
            "2021": params_to_json_dict(base),
            "2022": params_to_json_dict(replace(base, high_len=205)),
        },
        "folds": [
            {"effective_start": "2021-01-01", "effective_end": "2021-12-31"},
            {"effective_start": "2022-01-01", "effective_end": "2022-12-31"},
        ],
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        rolling_single_path = Path(tmp_dir) / "roos_base_best.json"
        rolling_single_path.write_text(json.dumps(rolling_single_payload), encoding="utf-8")
        rolling_single_source = _load_param_source(rolling_single_path)
        rolling_single_pair = _build_controlled_param_source_pair(
            rolling_single_source,
            filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
            threshold=float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD),
            fixed_risk=None,
        )
    rolling_single_left = rolling_single_pair[1]["params_by_effective_date"]["2022-01-01"]
    rolling_single_right = rolling_single_pair[2]["params_by_effective_date"]["2022-01-01"]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "rolling_oos_single_param_schedule_is_supported_without_changing_effective_dates",
        ("rolling_oos_param_schedule", False, True, 205, 205),
        (
            rolling_single_pair[0],
            rolling_single_left["use_breakout_quality_filter"],
            rolling_single_right["use_breakout_quality_filter"],
            rolling_single_left["high_len"],
            rolling_single_right["high_len"],
        ),
    )

    rolling_payload = {
        "schema_type": "optimizer_active_param_ensemble",
        "schema_version": 1,
        "mode": "rolling",
        "usage": "validation_only",
        "type": "outer_rolling_oos_param_set",
        "random_seed_ensemble": {"enabled": True, "seed_count": 2, "min_agree": 2},
        "summary": {"oos_end_date": "2022-12-31"},
        "params_by_effective_date": {"2021-01-01": params_to_json_dict(base), "2022-01-01": params_to_json_dict(base)},
        "params_by_oos_year": {"2021": params_to_json_dict(base), "2022": params_to_json_dict(base)},
        "params_ensemble_by_effective_date": {
            "2021-01-01": [
                {"member_index": 1, "params": params_to_json_dict(base)},
                {"member_index": 2, "params": params_to_json_dict(replace(base, high_len=205))},
            ],
            "2022-01-01": [
                {"member_index": 1, "params": params_to_json_dict(base)},
                {"member_index": 2, "params": params_to_json_dict(replace(base, high_len=205))},
            ],
        },
        "folds": [
            {"effective_start": "2021-01-01", "effective_end": "2021-12-31"},
            {"effective_start": "2022-01-01", "effective_end": "2022-12-31"},
        ],
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        rolling_path = Path(tmp_dir) / "roos_base_finalists_agree.json"
        rolling_path.write_text(json.dumps(rolling_payload), encoding="utf-8")
        rolling_source = _load_param_source(rolling_path)
        rolling_pair = _build_controlled_param_source_pair(
            rolling_source,
            filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
            threshold=float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD),
            fixed_risk=None,
        )
    finalist_best_payload = json.loads(json.dumps(rolling_payload))
    finalist_best_payload["selector"] = "base_finalist_best"
    finalist_best_payload["random_seed_ensemble"] = {
        "enabled": False, "seed_count": 1, "min_agree": 1, "policy_name": "base_finalist_best"
    }
    for effective_date in list(finalist_best_payload["params_ensemble_by_effective_date"]):
        member = finalist_best_payload["params_ensemble_by_effective_date"][effective_date][0]
        member["policy"] = "base_finalist_best"
        finalist_best_payload["params_ensemble_by_effective_date"][effective_date] = [member]
    finalist_best_source = {"kind": "rolling_active_param_ensemble", "payload": finalist_best_payload}
    finalist_best_contract = _validate_requested_param_policy(
        finalist_best_source, PARAM_POLICY_BASE_FINALIST_BEST
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "finalist_best_score_ranking_policy_requires_single_runtime_member",
        ("base_finalist_best", 1, 1, 1),
        (
            finalist_best_contract["selector"],
            finalist_best_contract["member_count_min"],
            finalist_best_contract["member_count_max"],
            finalist_best_contract["min_agree"],
        ),
    )

    try:
        _validate_requested_param_policy(
            {"kind": "rolling_active_param_ensemble", "payload": rolling_payload},
            PARAM_POLICY_BASE_FINALIST_BEST,
        )
        wrong_selector_rejected = False
    except ValueError as exc:
        wrong_selector_rejected = "selector" in str(exc)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "finalist_best_policy_rejects_finalists_agree_source",
        True,
        wrong_selector_rejected,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        root_path = Path(tmp_dir)
        (root_path / "models").mkdir()
        resolved_best = _resolve_params_path(
            root=root_path, params_path=None,
            param_policy=PARAM_POLICY_BASE_FINALIST_BEST, allow_static_diagnostic=False,
        )
        resolved_agree = _resolve_params_path(
            root=root_path, params_path=None,
            param_policy=PARAM_POLICY_BASE_FINALISTS_AGREE, allow_static_diagnostic=False,
        )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "score_ranking_param_policy_resolves_canonical_roos_filenames",
        ("roos_base_best.json", "roos_base_finalists_agree.json"),
        (resolved_best.name, resolved_agree.name),
    )

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "score_ranking_param_policies_use_isolated_output_directories",
        (
            "strategy_compare_score_ranking_base_finalist_best",
            "strategy_compare_score_ranking_base_finalists_agree",
        ),
        (
            _comparison_output_dir_name(
                COMPARISON_MODE_SCORE_RANKING,
                _comparison_labels(COMPARISON_MODE_SCORE_RANKING),
                param_policy=PARAM_POLICY_BASE_FINALIST_BEST,
            ),
            _comparison_output_dir_name(
                COMPARISON_MODE_SCORE_RANKING,
                _comparison_labels(COMPARISON_MODE_SCORE_RANKING),
                param_policy=PARAM_POLICY_BASE_FINALISTS_AGREE,
            ),
        ),
    )

    rolling_left = rolling_pair[1]["params_ensemble_by_effective_date"]["2021-01-01"][0]["params"]
    rolling_right = rolling_pair[2]["params_ensemble_by_effective_date"]["2021-01-01"][0]["params"]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "rolling_oos_active_param_ensemble_is_supported_without_changing_schedule",
        ("rolling_active_param_ensemble", False, True, 2, 2),
        (
            rolling_pair[0],
            rolling_left["use_breakout_quality_filter"],
            rolling_right["use_breakout_quality_filter"],
            rolling_pair[5]["seed_count"],
            rolling_pair[5]["min_agree"],
        ),
    )

    capacity = _capacity_summary({
        "portfolio_capacity_rows": [
            {
                "Orderable_Candidates": 3,
                "Candidate_Supply_Gap": 7,
                "End_Position_Gap": 8,
                "Post_Execution_Positions": 2,
            },
            {
                "Orderable_Candidates": 0,
                "Candidate_Supply_Gap": 8,
                "End_Position_Gap": 8,
                "Post_Execution_Positions": 2,
            },
        ]
    })
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "daily_candidate_and_position_gap_summary",
        (2, 1.5, 1, 2, 15, 2, 16, 2.0, 0),
        (
            capacity["sim_day_count"],
            capacity["avg_orderable_candidates"],
            capacity["zero_orderable_candidate_days"],
            capacity["candidate_supply_gap_days"],
            capacity["candidate_supply_gap_slot_days"],
            capacity["underfilled_end_days"],
            capacity["end_position_gap_slot_days"],
            capacity["avg_end_positions"],
            capacity["full_position_days"],
        ),
    )

    normalized_years = _normalize_yearly_completeness(pd.DataFrame([
        {
            "year": 2025,
            "is_full_year": True,
            "start_date": "2025-01-02",
            "end_date": "2025-12-31",
        },
        {
            "year": 2026,
            "is_full_year": True,
            "start_date": "2026-01-02",
            "end_date": "2026-03-02",
        },
    ]))
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "comparison_partial_final_year_is_not_marked_full",
        [True, False],
        list(normalized_years["is_full_year"]),
    )

    no_filter_history = pd.DataFrame([
        {"Date": "2025-01-03", "Ticker": "A", "Type": "買進 (新訊號, EV:1.00R)", "買訊日": "2025-01-02", "候選日": "2025-01-03", "進場類型": "normal", "成交價": 10.0},
        {"Date": "2025-01-10", "Ticker": "A", "Type": "全倉結算(指標)", "成交價": 13.0, "該筆總損益": 3000.0, "R_Multiple": 3.0},
        {"Date": "2025-02-03", "Ticker": "B", "Type": "買進 (新訊號, EV:1.00R)", "買訊日": "2025-01-31", "候選日": "2025-02-03", "進場類型": "normal", "成交價": 10.0},
        {"Date": "2025-02-10", "Ticker": "B", "Type": "全倉結算(停損)", "成交價": 9.0, "該筆總損益": -1000.0, "R_Multiple": -1.0},
        {"Date": "2025-03-03", "Ticker": "C", "Type": "買進 (新訊號, EV:1.00R)", "買訊日": "2025-02-28", "候選日": "2025-03-03", "進場類型": "normal", "成交價": 10.0},
        {"Date": "2025-03-10", "Ticker": "C", "Type": "期末強制結算", "成交價": 11.0, "該筆總損益": 1000.0, "R_Multiple": 1.0},
    ])
    quality_history = pd.DataFrame([
        {"Date": "2025-03-03", "Ticker": "C", "Type": "買進 (新訊號, EV:1.00R)", "買訊日": "2025-02-28", "候選日": "2025-03-03", "進場類型": "normal", "成交價": 10.0},
        {"Date": "2025-03-10", "Ticker": "C", "Type": "期末強制結算", "成交價": 11.0, "該筆總損益": 1000.0, "R_Multiple": 1.0},
        {"Date": "2025-04-03", "Ticker": "D", "Type": "買進 (新訊號, EV:1.00R)", "買訊日": "2025-04-02", "候選日": "2025-04-03", "進場類型": "normal", "成交價": 10.0},
        {"Date": "2025-04-10", "Ticker": "D", "Type": "期末強制結算", "成交價": 9.5, "該筆總損益": -500.0, "R_Multiple": -0.5},
    ])
    shared_scores = pd.DataFrame({
        "ticker": ["A", "B", "C", "D"],
        "date": ["2025-01-02", "2025-01-31", "2025-02-28", "2025-04-02"],
        SCORE_COLUMN: [0.40, 0.30, 0.80, 0.90],
    }).set_index(["ticker", "date"])[[SCORE_COLUMN]]
    attribution = build_trade_attribution(
        no_filter_trade_history=no_filter_history,
        quality_filter_trade_history=quality_history,
        shared_score_table=shared_scores,
        threshold=0.50,
        no_filter_portfolio_total_r=3.0,
        quality_filter_portfolio_total_r=0.5,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "trade_attribution_partitions_common_and_exclusive_round_trips",
        (1, 2, 1, 3, 2),
        tuple(attribution["trade_partition"][key] for key in (
            "common_count", "no_filter_only_count", "quality_filter_only_count",
            "no_filter_total_count", "quality_filter_total_count",
        )),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "trade_attribution_reconciles_portfolio_total_r",
        (3.0, 1.0, 0.5, -2.5, 0.0),
        (
            attribution["r_attribution"]["excluded_winner_r"],
            attribution["r_attribution"]["avoided_loser_r_abs"],
            attribution["r_attribution"]["replacement_loser_r_abs"],
            attribution["r_attribution"]["exclusive_selection_delta_r"],
            attribution["r_attribution"]["reconciliation_error_r"],
        ),
        tol=1e-12,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "trade_attribution_distinguishes_direct_filter_rejects",
        (2, 0, 0),
        (
            attribution["r_attribution"]["direct_filter_reject_count"],
            attribution["r_attribution"]["portfolio_path_displacement_count"],
            attribution["r_attribution"]["score_lookup_unavailable_count"],
        ),
    )

    native_payload = _to_json_native({
        "number": np.float64(1.25),
        "date": pd.Timestamp("2026-07-26"),
        "non_finite": np.float64(np.nan),
    })
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "comparison_json_payload_is_native_and_strict",
        {"number": 1.25, "date": "2026-07-26T00:00:00", "non_finite": None},
        native_payload,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "comparison_json_number_is_builtin_float",
        True,
        type(native_payload["number"]) is float,
    )

    summary["controlled_param_difference"] = ["use_breakout_quality_filter"]
    return results, summary


__all__ = [
    "validate_breakout_quality_chronological_embargo_case",
    "validate_breakout_quality_continuous_target_contract_case",
    "validate_breakout_quality_policy_single_source_case",
    "validate_breakout_quality_runtime_artifact_contract_case",
    "validate_breakout_quality_strategy_comparison_contract_case",
]
