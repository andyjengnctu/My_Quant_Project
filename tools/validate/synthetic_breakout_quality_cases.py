from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
import ast
from importlib.util import find_spec
import io
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


def _read_audit_source(audit_type: str) -> str:
    """Resolve Audit source through the project-wide catalog SSOT."""
    from tools.audit.catalog import get_audit_entry

    entry = get_audit_entry(audit_type)
    spec = find_spec(entry.module)
    source_path = Path(spec.origin) if spec is not None and spec.origin else None
    if source_path is None or not source_path.is_file():
        raise FileNotFoundError(
            f"Audit catalog module找不到source: audit_type={audit_type}, module={entry.module}"
        )
    return source_path.read_text(encoding="utf-8")


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
    SUPPORTED_BREAKOUT_QUALITY_CLASSIFICATION_EXPERIMENT_PROFILES,
    TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION,
    TRAINING_LABEL_SCOPE_PASS_ONLY,
    TRAINING_LABEL_SCOPE_ALL,
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


def validate_breakout_quality_policy_single_source_case(_base_params):
    case_id = "BREAKOUT_QUALITY_POLICY_SSOT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    project_root = Path(__file__).resolve().parents[2]
    canonical_config_path = project_root / "config" / "breakout_quality.py"
    canonical_source = canonical_config_path.read_text(encoding="utf-8")
    removed_legacy_config_paths = tuple(
        project_root / "config" / filename
        for filename in (
            "breakout_quality_policy.py",
            "breakout_quality_experiments.py",
            "breakout_quality_workflow.py",
        )
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "breakout_quality_config_has_one_editable_canonical_module",
        True,
        bool(
            canonical_config_path.is_file()
            and all(not path.exists() for path in removed_legacy_config_paths)
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "breakout_quality_has_one_user_facing_random_seed_setting",
        (1, False),
        (
            canonical_source.count("BREAKOUT_QUALITY_RANDOM_SEED ="),
            "BREAKOUT_QUALITY_WORKFLOW_RANDOM_SEED" in canonical_source,
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "breakout_quality_strategy_compare_uses_only_canonical_app_entry",
        False,
        (project_root / "apps" / "breakout_quality_strategy_compare.py").exists(),
    )
    user_settings_marker = canonical_source.index(
        "# USER SETTINGS — edit this section only"
    )
    internal_marker = canonical_source.index(
        "# INTERNAL PROFILE DEFINITIONS AND SUPPORTED VALUES"
    )
    first_implementation_line = min(
        node.lineno
        for node in ast.parse(canonical_source).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "breakout_quality_user_settings_are_grouped_before_implementation",
        True,
        bool(
            user_settings_marker < internal_marker
            and canonical_source[:internal_marker].count("def ") == 0
            and canonical_source[:internal_marker].count("class ") == 0
            and canonical_source[:internal_marker].count(
                "BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE ="
            )
            == 1
            and first_implementation_line
            > canonical_source[:internal_marker].count("\n")
        ),
    )
    stale_import_patterns = (
        "from config.breakout_quality_policy import",
        "from config.breakout_quality_experiments import",
        "from config.breakout_quality_workflow import",
        "from config import breakout_quality_policy",
        "from config import breakout_quality_experiments",
        "from config import breakout_quality_workflow",
    )
    stale_import_files = []
    current_validator_path = Path(__file__).resolve()
    for source_root in ("apps", "core", "filters", "strategies", "tools"):
        for source_path in (project_root / source_root).rglob("*.py"):
            if source_path.resolve() == current_validator_path:
                continue
            source_text = source_path.read_text(encoding="utf-8")
            if any(pattern in source_text for pattern in stale_import_patterns):
                stale_import_files.append(source_path.relative_to(project_root).as_posix())
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "breakout_quality_runtime_imports_use_canonical_config",
        (),
        tuple(sorted(stale_import_files)),
    )

    configured_seed = resolve_breakout_quality_random_seed()
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "single_seed_is_nonnegative_integer",
        True,
        isinstance(configured_seed, int) and configured_seed >= 0,
    )
    from config import breakout_quality as breakout_quality_config

    with patch.object(breakout_quality_config, "BREAKOUT_QUALITY_RANDOM_SEED", 7):
        overridden_seed = breakout_quality_config.resolve_breakout_quality_random_seed()
        with patch.object(
            breakout_quality_config,
            "BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE",
            UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE,
        ):
            binary_workflow_seed = (
                breakout_quality_config.get_breakout_quality_workflow_settings().seed
            )
        with patch.object(
            breakout_quality_config,
            "BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE",
            STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE,
        ):
            continuous_workflow_seed = (
                breakout_quality_config.get_breakout_quality_workflow_settings().seed
            )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "single_seed_override_applies_to_all_profiles",
        (7, 7, 7),
        (overridden_seed, binary_workflow_seed, continuous_workflow_seed),
    )
    with patch("config.breakout_quality.BREAKOUT_QUALITY_RANDOM_SEED", -1):
        try:
            resolve_breakout_quality_random_seed()
        except ValueError:
            negative_seed_rejected = True
        else:
            negative_seed_rejected = False
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "single_seed_negative_override_rejected",
        True,
        negative_seed_rejected,
    )

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
        and isinstance(BREAKOUT_QUALITY_RANDOM_SEED, int)
        and int(BREAKOUT_QUALITY_RANDOM_SEED) >= 0
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
                    model_architecture=runtime_fixture_architecture,
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
                model_architecture=runtime_fixture_architecture,
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
        preferred_hard_filter_dir = canonical_strategy_compare_output_dir_names(
            COMPARISON_MODE_HARD_FILTER
        )[0]
        expected_round_trip_path = (
            resolve_filter_model_output_dir(
                temp_dir,
                BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
                BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
                BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
            )
            / preferred_hard_filter_dir
            / "no_filter_round_trips.csv"
        )
        expected_round_trip_path.parent.mkdir(parents=True, exist_ok=True)
        expected_round_trip_path.write_text(
            "ticker,r_multiple\n2330,1.0\n",
            encoding="utf-8",
        )
        with patch(
            "tools.audit.breakout_quality.continuous_target.PROJECT_ROOT",
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

        expected_round_trip_path.unlink()
        trade_history_path = expected_round_trip_path.parent / "no_filter_trades.csv"
        trade_history_frame = pd.DataFrame(
            [
                {
                    "Date": "2021-01-05",
                    "Ticker": "2330",
                    "Type": "買進 (突破)",
                    "進場類型": "normal",
                    "候選類型": "normal",
                    "買訊日": "2021-01-04",
                    "候選日": "2021-01-04",
                    "成交價": 100.0,
                    "該筆總損益": np.nan,
                    "R_Multiple": np.nan,
                },
                {
                    "Date": "2021-01-15",
                    "Ticker": "2330",
                    "Type": "全倉結算",
                    "進場類型": "",
                    "候選類型": "",
                    "買訊日": "",
                    "候選日": "",
                    "成交價": 110.0,
                    "該筆總損益": 1000.0,
                    "R_Multiple": 1.5,
                },
            ]
        )
        trade_history_frame.to_csv(trade_history_path, index=False, encoding="utf-8-sig")
        with patch(
            "tools.audit.breakout_quality.continuous_target.PROJECT_ROOT",
            Path(temp_dir),
        ):
            rebuilt_round_trips, rebuilt_source = _load_round_trip_source(
                BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
                None,
            )
        rebuilt_record = rebuilt_round_trips.iloc[0]
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "continuous_target_round_trip_falls_back_to_canonical_trade_history",
            (
                1,
                "2330",
                "2021-01-04",
                1.5,
                "reconstructed_from_no_filter_trades",
                True,
                str(trade_history_path),
            ),
            (
                len(rebuilt_round_trips),
                str(rebuilt_record["ticker"]),
                str(rebuilt_record["signal_date"]),
                float(rebuilt_record["r_multiple"]),
                str(rebuilt_source["source_kind"]),
                bool(rebuilt_source["round_trips_reconstructed"]),
                str(rebuilt_source["path"]),
            ),
            tol=1e-12,
        )


        trade_history_path.unlink()
        active_output_root = expected_round_trip_path.parents[1]
        skipped_static_dir = active_output_root / "strategy_compare_score_ranking_base_finalists_agree"
        skipped_static_dir.mkdir(parents=True, exist_ok=True)
        (skipped_static_dir / "strategy_comparison.json").write_text(
            json.dumps(
                {
                    "metadata": {
                        "filter_id": BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
                        "model_architecture": BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
                        "experiment_profile": BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
                        "comparison_design": "static_param_diagnostic",
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        trade_history_frame.to_csv(
            skipped_static_dir / "no_filter_trades.csv",
            index=False,
            encoding="utf-8-sig",
        )
        discovered_dir = active_output_root / "strategy_compare_score_ranking_base_finalist_best"
        discovered_dir.mkdir(parents=True, exist_ok=True)
        (discovered_dir / "strategy_comparison.json").write_text(
            json.dumps(
                {
                    "metadata": {
                        "filter_id": BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
                        "model_architecture": BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
                        "experiment_profile": BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
                        "comparison_design": "historical_active_param_oos",
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        discovered_trade_history_path = discovered_dir / "no_filter_trades.csv"
        trade_history_frame.to_csv(
            discovered_trade_history_path,
            index=False,
            encoding="utf-8-sig",
        )
        with patch(
            "tools.audit.breakout_quality.continuous_target.PROJECT_ROOT",
            Path(temp_dir),
        ):
            discovered_round_trips, discovered_source = _load_round_trip_source(
                BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
                None,
            )
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "continuous_target_discovers_official_score_ranking_trade_history_and_skips_static",
            (
                1,
                "active_9a_strategy_compare_trade_history_discovery",
                str(discovered_trade_history_path),
                "historical_active_param_oos",
            ),
            (
                len(discovered_round_trips),
                str(discovered_source["path_source"]),
                str(discovered_source["path"]),
                str((discovered_source.get("strategy_compare_metadata") or {}).get("comparison_design")),
            ),
        )

        explicit_trade_history_path = Path(temp_dir) / "explicit_no_filter_trades.csv"
        trade_history_frame.to_csv(
            explicit_trade_history_path,
            index=False,
            encoding="utf-8-sig",
        )
        with patch(
            "tools.audit.breakout_quality.continuous_target.PROJECT_ROOT",
            Path(temp_dir),
        ):
            explicit_round_trips, explicit_source = _load_round_trip_source(
                BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
                None,
                str(explicit_trade_history_path),
            )
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "continuous_target_accepts_explicit_trade_history_override",
            (1, "explicit_trade_history", str(explicit_trade_history_path), True),
            (
                len(explicit_round_trips),
                str(explicit_source["path_source"]),
                str(explicit_source["path"]),
                bool(explicit_source["round_trips_reconstructed"]),
            ),
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
        "tools.audit.breakout_quality.continuous_target",
        command_modules.get("audit-continuous-target"),
    )

    summary["target_id"] = STRATEGY_ALIGNED_TARGET_ID
    summary["training_performed"] = False
    return results, summary


def validate_breakout_quality_continuous_ranker_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_CONTINUOUS_RANKER"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    profile = get_breakout_quality_experiment_profile(
        STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_ranker_is_named_training_profile_not_model_architecture",
        (
            TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION,
            STRATEGY_ALIGNED_TARGET_ID,
            "mse",
            "mean_daily_spearman",
            False,
        ),
        (
            profile.training_objective,
            profile.continuous_target_id,
            profile.loss_name,
            profile.epoch_selection_metric,
            STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE
            in SUPPORTED_BREAKOUT_QUALITY_CLASSIFICATION_EXPERIMENT_PROFILES,
        ),
    )

    pass_profile = get_breakout_quality_experiment_profile(
        STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "pass_conditional_ranker_is_named_profile_with_no_time_target_and_pass_scope",
        (
            TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION,
            STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
            TRAINING_LABEL_SCOPE_PASS_ONLY,
            False,
        ),
        (
            pass_profile.training_objective,
            pass_profile.continuous_target_id,
            pass_profile.training_label_scope,
            STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE
            in SUPPORTED_BREAKOUT_QUALITY_CLASSIFICATION_EXPERIMENT_PROFILES,
        ),
    )

    raw = np.asarray([1.0, 3.0, 2.0, 5.0, 5.0, 9.0], dtype=np.float32)
    valid = np.ones((6,), dtype=bool)
    dates = pd.Series(["2020-01-02"] * 3 + ["2021-05-03"] * 3)
    percentiles = build_daily_percentile_targets(raw, valid, dates)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_ranker_daily_percentile_spans_zero_one_and_averages_ties",
        (0.0, 1.0, 0.5, 0.25, 0.25, 1.0),
        tuple(round(float(value), 6) for value in percentiles),
    )

    synthetic_group_table = pd.DataFrame(
        {
            "label": [LABEL_PASS, LABEL_REJECT, LABEL_PASS, LABEL_REJECT, LABEL_PASS, LABEL_PASS],
            "date": pd.to_datetime(dates),
        }
    )
    scoped_ids = continuous_ranker_scope_group_ids(
        np.arange(6, dtype=np.int64),
        synthetic_group_table,
        label_scope=TRAINING_LABEL_SCOPE_PASS_ONLY,
    )
    pass_mask = np.zeros((6,), dtype=bool)
    pass_mask[scoped_ids] = True
    pass_percentiles = build_daily_percentile_targets(raw, pass_mask, dates)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "pass_conditional_ranker_percentiles_use_only_same_date_pass_groups",
        (0, 2, 4, 5, 0.0, 1.0, 0.0, 1.0, True, True),
        (
            int(scoped_ids[0]), int(scoped_ids[1]), int(scoped_ids[2]), int(scoped_ids[3]),
            round(float(pass_percentiles[0]), 6),
            round(float(pass_percentiles[2]), 6),
            round(float(pass_percentiles[4]), 6),
            round(float(pass_percentiles[5]), 6),
            bool(np.isnan(pass_percentiles[1])),
            bool(np.isnan(pass_percentiles[3])),
        ),
    )

    singleton_percentile = build_daily_percentile_targets(
        np.asarray([7.0], dtype=np.float32),
        np.asarray([True], dtype=bool),
        pd.Series(["2022-08-08"]),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_ranker_singleton_date_uses_neutral_half_percentile",
        (0.5,),
        tuple(float(value) for value in singleton_percentile),
    )

    changed_oos = raw.copy()
    changed_oos[3:] = np.asarray([-100.0, 500.0, 0.0], dtype=np.float32)
    changed_percentiles = build_daily_percentile_targets(changed_oos, valid, dates)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_ranker_same_date_transform_prevents_cross_split_distribution_leakage",
        tuple(percentiles[:3]),
        tuple(changed_percentiles[:3]),
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        target_dir = resolve_continuous_target_dir(
            root,
            "synthetic_quality",
            target_id=STRATEGY_ALIGNED_TARGET_ID,
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        raw_path = target_dir / TARGET_RAW_FILENAME
        valid_path = target_dir / TARGET_VALID_MASK_FILENAME
        np.save(raw_path, raw, allow_pickle=False)
        np.save(valid_path, valid, allow_pickle=False)
        dataset_source_path = root / "events.csv"
        dataset_source_path.write_text("ticker,date\n2330,2020-01-02\n", encoding="utf-8")
        dataset_artifacts = {"events_csv": build_file_manifest(dataset_source_path)}
        manifest = {
            "schema_version": 1,
            "filter_id": "synthetic_quality",
            "target_contract": {"target_id": STRATEGY_ALIGNED_TARGET_ID},
            "group_count": 6,
            "dataset_policy": {"policy": "synthetic"},
            "dataset_artifact_source": dataset_artifacts,
            "artifacts": {
                "target_raw_r": build_file_manifest(raw_path),
                "valid_mask": build_file_manifest(valid_path),
            },
        }
        (target_dir / TARGET_MANIFEST_FILENAME).write_text(
            json.dumps(manifest, ensure_ascii=False),
            encoding="utf-8",
        )
        loaded_manifest, loaded_raw, loaded_valid = load_validated_continuous_target_arrays(
            root,
            "synthetic_quality",
            expected_group_count=6,
            expected_dataset_policy={"policy": "synthetic"},
            expected_dataset_artifacts=dataset_artifacts,
        )
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "continuous_ranker_strictly_loads_versioned_target_arrays",
            (STRATEGY_ALIGNED_TARGET_ID, tuple(raw), tuple(valid)),
            (
                loaded_manifest["target_contract"]["target_id"],
                tuple(loaded_raw),
                tuple(loaded_valid),
            ),
        )
        stale_dataset_artifacts = {
            "events_csv": {
                **dataset_artifacts["events_csv"],
                "sha256": "0" * 64,
            }
        }
        try:
            load_validated_continuous_target_arrays(
                root,
                "synthetic_quality",
                expected_group_count=6,
                expected_dataset_policy={"policy": "synthetic"},
                expected_dataset_artifacts=stale_dataset_artifacts,
            )
            stale_dataset_rejected = False
        except ValueError as exc:
            stale_dataset_rejected = "Dataset artifact" in str(exc)
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "continuous_ranker_rejects_target_from_different_dataset_artifacts",
            True,
            stale_dataset_rejected,
        )

        raw_path.write_bytes(raw_path.read_bytes() + b"tamper")
        try:
            load_validated_continuous_target_arrays(
                root,
                "synthetic_quality",
                expected_group_count=6,
                expected_dataset_policy={"policy": "synthetic"},
            )
            tamper_rejected = False
        except ValueError as exc:
            tamper_rejected = "size" in str(exc) or "SHA256" in str(exc)
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "continuous_ranker_rejects_tampered_target_artifact",
            True,
            tamper_rejected,
        )

    app_path = Path(__file__).resolve().parents[2] / "apps" / "breakout_quality.py"
    tree = ast.parse(app_path.read_text(encoding="utf-8"), filename=str(app_path))
    command_modules = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "COMMAND_MODULES" for target in node.targets):
            command_modules = ast.literal_eval(node.value)
            break
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_ranker_research_command_is_registered",
        "tools.filters.breakout_quality.train_continuous_ranker",
        command_modules.get("train-continuous-ranker"),
    )

    ranker_source = (
        Path(__file__).resolve().parents[1]
        / "filters"
        / "breakout_quality"
        / "train_continuous_ranker.py"
    ).read_text(encoding="utf-8")
    inception_source = (
        Path(__file__).resolve().parents[2]
        / "filters"
        / "breakout_quality"
        / "models"
        / "inception_time.py"
    ).read_text(encoding="utf-8")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_ranker_reuses_active_two_logit_head_and_pass_probability",
        (True, True, True),
        (
            "self.classifier = nn.Linear(module_output_channels, 2)" in inception_source,
            "torch.softmax(logits.float(), dim=1)[:, LABEL_PASS]" in ranker_source,
            '"model_state_dict"' in ranker_source and "torch.save(" in ranker_source,
        ),
    )
    train_source = (
        Path(__file__).resolve().parents[1]
        / "filters"
        / "breakout_quality"
        / "train.py"
    ).read_text(encoding="utf-8")
    artifact_source = (
        Path(__file__).resolve().parents[2]
        / "filters"
        / "breakout_quality"
        / "artifacts.py"
    ).read_text(encoding="utf-8")
    app_source = app_path.read_text(encoding="utf-8")
    export_source = (
        Path(__file__).resolve().parents[2]
        / "filters"
        / "breakout_quality"
        / "export_scores.py"
    ).read_text(encoding="utf-8")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_ranker_is_cli_only_and_does_not_pollute_interactive_menu",
        (True, True, True, True),
        (
            command_modules.get("train-continuous-ranker")
            == "tools.filters.breakout_quality.train_continuous_ranker",
            'print("[10] 11B 同日 Percentile Ranker（research-only）")' not in app_source,
            'elif choice == "10":' not in app_source,
            "_interactive_train_continuous_ranker" not in app_source,
        ),
    )

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_ranker_profile_is_blocked_from_binary_workflow_and_runtime_loader",
        (True, True, True, True),
        (
            "SUPPORTED_BREAKOUT_QUALITY_CLASSIFICATION_EXPERIMENT_PROFILES" in train_source,
            "research-only continuous ranker artifact不得載入正式binary runtime contract" in artifact_source,
            "SUPPORTED_BREAKOUT_QUALITY_CLASSIFICATION_EXPERIMENT_PROFILES" in app_source,
            "SUPPORTED_BREAKOUT_QUALITY_CLASSIFICATION_EXPERIMENT_PROFILES" in export_source,
        ),
    )

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_ranker_is_research_only_and_oos_follows_checkpoint_write",
        (True, True, True, True, True, True),
        (
            '"eligible": False' in ranker_source,
            "OOS target transformation and model inference occur only after" in ranker_source,
            "torch.save(" in ranker_source
            and ranker_source.index("torch.save(")
            < ranker_source.index("OOS target transformation and model inference occur only after"),
            'score_frame["group_index"].duplicated().any()' in ranker_source,
            'label_scope=profile.training_label_scope' in ranker_source,
            '"label_conditional": {}' in ranker_source,
        ),
    )

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "pass_conditional_ranker_is_cli_only_and_uses_existing_command",
        (True, True, True, True),
        (
            "STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE" in ranker_source,
            'choices=(' in ranker_source,
            "11G" not in app_source[app_source.index("def _run_interactive_menu"):],
            command_modules.get("train-continuous-ranker")
            == "tools.filters.breakout_quality.train_continuous_ranker",
        ),
    )

    summary["profile"] = STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE
    summary["pass_conditional_profile"] = STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE
    summary["training_objective"] = profile.training_objective
    return results, summary


def validate_breakout_quality_pass_conditional_ranker_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_PASS_CONDITIONAL_RANKER"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    profile = get_breakout_quality_experiment_profile(
        STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "pass_conditional_profile_uses_no_time_target_and_pass_only_scope",
        (
            TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION,
            STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
            TRAINING_LABEL_SCOPE_PASS_ONLY,
            "mse",
            "mean_daily_spearman",
            False,
        ),
        (
            profile.training_objective,
            profile.continuous_target_id,
            profile.training_label_scope,
            profile.loss_name,
            profile.epoch_selection_metric,
            STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE
            in SUPPORTED_BREAKOUT_QUALITY_CLASSIFICATION_EXPERIMENT_PROFILES,
        ),
    )

    group_table = pd.DataFrame(
        {
            "label": [LABEL_PASS, LABEL_REJECT, LABEL_PASS, LABEL_REJECT, LABEL_PASS, LABEL_PASS],
            "date": pd.to_datetime(["2020-01-02"] * 3 + ["2021-05-03"] * 3),
        }
    )
    ids = continuous_ranker_scope_group_ids(
        np.arange(6, dtype=np.int64),
        group_table,
        label_scope=TRAINING_LABEL_SCOPE_PASS_ONLY,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "pass_conditional_scope_keeps_only_original_pass_groups",
        (0, 2, 4, 5),
        tuple(int(value) for value in ids),
    )

    raw = np.asarray([1.0, 99.0, 3.0, -50.0, 5.0, 9.0], dtype=np.float32)
    mask = np.zeros((6,), dtype=bool)
    mask[ids] = True
    percentiles = build_daily_percentile_targets(raw, mask, group_table["date"] )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "pass_conditional_percentile_ignores_same_date_reject_target_values",
        (0.0, 1.0, 0.0, 1.0, True, True),
        (
            round(float(percentiles[0]), 6),
            round(float(percentiles[2]), 6),
            round(float(percentiles[4]), 6),
            round(float(percentiles[5]), 6),
            bool(np.isnan(percentiles[1])),
            bool(np.isnan(percentiles[3])),
        ),
    )

    root = Path(__file__).resolve().parents[2]
    app_path = root / "apps" / "breakout_quality.py"
    app_source = app_path.read_text(encoding="utf-8")
    tree = ast.parse(app_source, filename=str(app_path))
    command_modules = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "COMMAND_MODULES"
            for target in node.targets
        ):
            command_modules = ast.literal_eval(node.value)
            break
    menu_source = app_source[
        app_source.find("def _run_interactive_menu") : app_source.find("def main")
    ]
    ranker_path = root / "tools" / "filters" / "breakout_quality" / "train_continuous_ranker.py"
    ranker_source = ranker_path.read_text(encoding="utf-8")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "pass_conditional_ranker_reuses_existing_cli_and_is_not_in_menu",
        (True, True, True, True),
        (
            command_modules.get("train-continuous-ranker")
            == "tools.filters.breakout_quality.train_continuous_ranker",
            "STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE" in ranker_source,
            "11G" not in menu_source,
            "strategy_aligned_no_time_pass_magnitude_mse" not in menu_source,
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "pass_conditional_ranker_freezes_checkpoint_before_oos_pass_percentile",
        (True, True, True),
        (
            "torch.save(" in ranker_source,
            ranker_source.index("torch.save(")
            < ranker_source.index("oos_target_mask = np.zeros"),
            'label_scope=profile.training_label_scope' in ranker_source,
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "pass_conditional_ranker_outputs_label_conditional_trade_diagnostics",
        (True, True, True, True),
        (
            '"label_conditional": {}' in ranker_source,
            'for label_name, label_value in (("PASS", LABEL_PASS), ("REJECT", LABEL_REJECT))' in ranker_source,
            '"spearman_model_score_vs_target"' in ranker_source,
            '"spearman_target_vs_r_multiple"' in ranker_source,
        ),
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        target_dir = Path(temp_dir)
        pd.DataFrame(
            [
                {"ticker": "2330", "target_date": "2021-01-04", "r_multiple": 2.0, "target_raw_r": 99.0},
                {"ticker": "2317", "target_date": "2021-01-04", "r_multiple": -1.0, "target_raw_r": 99.0},
                {"ticker": "2454", "target_date": "2021-01-05", "r_multiple": 3.0, "target_raw_r": 99.0},
            ]
        ).to_csv(
            target_dir / TARGET_TRADE_MATCHES_CSV_FILENAME,
            index=False,
            encoding="utf-8-sig",
        )
        score_frame = pd.DataFrame(
            [
                {"ticker": "2330", "date": "2021-01-04", "label": LABEL_PASS, "target_raw_r": 2.5, "model_score": 0.8},
                {"ticker": "2317", "date": "2021-01-04", "label": LABEL_REJECT, "target_raw_r": -0.2, "model_score": 0.2},
                {"ticker": "2454", "date": "2021-01-05", "label": LABEL_PASS, "target_raw_r": 3.2, "model_score": 0.9},
            ]
        )
        trade_metrics = continuous_ranker_trade_alignment_metrics(score_frame, target_dir)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "pass_conditional_trade_alignment_replaces_old_target_and_splits_labels",
        (3, 2, 1, 1.0),
        (
            trade_metrics.get("matched_trade_count"),
            trade_metrics.get("label_conditional", {}).get("PASS", {}).get("matched_trade_count"),
            trade_metrics.get("label_conditional", {}).get("REJECT", {}).get("matched_trade_count"),
            round(float(trade_metrics.get("spearman_target_vs_r_multiple")), 6),
        ),
    )

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "pass_conditional_ranker_remains_research_only_without_runtime_combination",
        (True, True, True, True),
        (
            '"eligible": False' in ranker_source,
            '"scope": "research_only"' in ranker_source,
            "forward_oos" not in ranker_source[ranker_source.find('"runtime_eligibility"'):],
            "score_blend" not in ranker_source.lower(),
        ),
    )

    summary["profile"] = STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE
    summary["training_label_scope"] = TRAINING_LABEL_SCOPE_PASS_ONLY
    return results, summary


def validate_breakout_quality_qualified_candidate_set_audit_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_QUALIFIED_CANDIDATE_SET_AUDIT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    candidate = {
        "ticker": "2330",
        "trade_date": "2022-01-05",
        "candidate_date": "2022-01-05",
        "signal_date": "2022-01-03",
        "type": "normal",
        "entry_source": "breakout",
        "params_obj": SimpleNamespace(high_len=201),
        "ensemble_vote_count": 4,
        "qty": 1000,
        "sort_value": 0.75,
        "ev": 1.25,
        "hist_win_rate": 0.60,
        "hist_trade_count": 20,
        "breakout_quality_score": 0.63,
        "breakout_quality_score_date": "2022-01-03",
    }
    snapshot = _candidate_replay_snapshot(
        candidate,
        fallback_trade_date=pd.Timestamp("2022-01-05"),
        is_orderable=True,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "qualified_candidate_snapshot_preserves_original_signal_date_without_runtime_objects",
        (
            "2330", "2022-01-05", "2022-01-03", True, 201, 4, False,
        ),
        (
            snapshot["ticker"], snapshot["trade_date"], snapshot["signal_date"],
            snapshot["is_orderable"], snapshot["high_len"],
            snapshot["ensemble_vote_count"], "params_obj" in snapshot,
        ),
    )

    oos_scores = pd.DataFrame(
        [
            {"ticker": "1101", "date": "2022-01-03", "group_index": 1, "label": 0,
             "target_raw_r": -1.0, "target_daily_percentile": 0.0, "model_score": 0.10},
            {"ticker": "2330", "date": "2022-01-03", "group_index": 2, "label": 1,
             "target_raw_r": 2.0, "target_daily_percentile": 1.0, "model_score": 0.80},
            {"ticker": "2603", "date": "2022-01-04", "group_index": 3, "label": 0,
             "target_raw_r": 0.0, "target_daily_percentile": 0.0, "model_score": 0.20},
            {"ticker": "2454", "date": "2022-01-04", "group_index": 4, "label": 1,
             "target_raw_r": 3.0, "target_daily_percentile": 1.0, "model_score": 0.90},
        ]
    )
    occurrences = pd.DataFrame(
        [
            {"ticker": "1101", "trade_date": "2022-01-04", "candidate_date": "2022-01-04",
             "signal_date": "2022-01-03", "candidate_type": "normal"},
            {"ticker": "1101", "trade_date": "2022-01-05", "candidate_date": "2022-01-05",
             "signal_date": "2022-01-03", "candidate_type": "continuation"},
            {"ticker": "2330", "trade_date": "2022-01-04", "candidate_date": "2022-01-04",
             "signal_date": "2022-01-03", "candidate_type": "normal"},
            {"ticker": "2603", "trade_date": "2022-01-05", "candidate_date": "2022-01-05",
             "signal_date": "2022-01-04", "candidate_type": "normal"},
            {"ticker": "2454", "trade_date": "2022-01-05", "candidate_date": "2022-01-05",
             "signal_date": "2022-01-04", "candidate_type": "normal"},
            {"ticker": "9999", "trade_date": "2022-01-05", "candidate_date": "2022-01-05",
             "signal_date": "2022-01-04", "candidate_type": "normal"},
        ]
    )
    attached = qualified_audit_attach_ranker_scores(
        occurrences,
        oos_scores,
        layer="qualified",
    )
    unique = qualified_audit_unique_groups(attached, layer="qualified")
    metrics = qualified_audit_layer_metrics(
        unique,
        occurrence_count=len(attached),
        occurrence_date_count=int(attached["trade_date"].nunique()),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "qualified_candidate_occurrences_align_by_signal_date_and_deduplicate_unique_groups",
        (6, 5, 4, 1, 2, 1.0, 1.0),
        (
            len(attached), int(attached["target_match"].sum()), len(unique),
            int((~attached["target_match"]).sum()),
            int(unique.loc[(unique["ticker"] == "1101") & (unique["target_date"] == "2022-01-03"), "occurrence_count"].iloc[0]),
            round(float(metrics["global_spearman_score_vs_target"]), 12),
            round(float(metrics["mean_daily_spearman_score_vs_target"]), 12),
        ),
        tol=1e-12,
    )

    daily = qualified_audit_daily_coverage(
        attached,
        attached[attached["trade_date"] == "2022-01-04"].copy(),
    )
    daily_lookup = daily.set_index("trade_date")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "qualified_candidate_daily_coverage_counts_ticker_signal_pairs_not_dates_only",
        (2, 2, 4),
        (
            int(daily_lookup.loc["2022-01-04", "qualified_unique_signals"]),
            int(daily_lookup.loc["2022-01-04", "orderable_unique_signals"]),
            int(daily_lookup.loc["2022-01-05", "qualified_unique_signals"]),
        ),
    )

    actual = pd.DataFrame(
        [
            {"ticker": "1101", "target_date": "2022-01-03", "target_raw_r": -1.0, "r_multiple": -0.5},
            {"ticker": "2330", "target_date": "2022-01-03", "target_raw_r": 2.0, "r_multiple": 2.5},
            {"ticker": "2603", "target_date": "2022-01-04", "target_raw_r": 0.0, "r_multiple": 0.0},
            {"ticker": "2454", "target_date": "2022-01-04", "target_raw_r": 3.0, "r_multiple": 4.0},
        ]
    )
    actual_metrics, actual_matches = qualified_audit_actual_trade_metrics(actual, oos_scores)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "qualified_candidate_actual_trade_merge_preserves_target_score_and_realized_r_direction",
        (4, 4, 1.0, 1.0, 1.0, 4.0, -0.5),
        (
            actual_metrics["trade_count"], actual_metrics["matched_trade_count"],
            round(float(actual_metrics["spearman_target_vs_realized_r"]), 12),
            round(float(actual_metrics["spearman_score_vs_target"]), 12),
            round(float(actual_metrics["spearman_score_vs_realized_r"]), 12),
            float(actual_metrics["top_score_decile_average_r"]),
            float(actual_metrics["bottom_score_decile_average_r"]),
        ),
        tol=1e-12,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "qualified_candidate_actual_trade_output_retains_each_trade_row",
        (4, ("1101", "2330", "2454", "2603")),
        (
            len(actual_matches),
            tuple(sorted(actual_matches["ticker"].astype(str).tolist())),
        ),
    )

    replay_payload = {
        "total_return_pct": 12.5,
        "max_drawdown_pct": 3.0,
        "return_over_max_drawdown": 4.1666666667,
        "annual_return_pct": 2.0,
        "log_r_squared": 0.9,
        "monthly_win_rate_pct": 60.0,
        "trade_count": 4,
        "win_rate_pct": 50.0,
        "payoff_ratio": 2.0,
        "expected_value_r": 0.5,
        "final_equity": 1125000.0,
        "avg_exposure_pct": 40.0,
        "max_exposure_pct": 90.0,
        "missed_buy_count": 1,
        "missed_sell_count": 0,
        "reserved_buy_fill_rate_pct": 80.0,
        "normal_trade_count": 3,
        "extended_trade_count": 1,
        "annual_trade_count": 1.0,
        "benchmark_return_pct": 5.0,
        "benchmark_max_drawdown_pct": 4.0,
        "benchmark_annual_return_pct": 1.0,
        "profile": {
            "portfolio_total_r": 7.5,
            "portfolio_median_r": 0.5,
            "portfolio_avg_r": 1.875,
            "min_full_year_return_pct": -2.0,
            "min_month_return_pct": -1.0,
            "min_quarter_return_pct": -1.5,
            "full_year_count": 1,
            "portfolio_capacity_rows": [],
        },
    }
    expected_replay = {
        key: replay_payload.get(key, replay_payload["profile"].get(key))
        for key in (
            "total_return_pct", "max_drawdown_pct", "return_over_max_drawdown",
            "annual_return_pct", "trade_count", "final_equity",
            "avg_exposure_pct", "max_exposure_pct", "missed_buy_count",
            "missed_sell_count", "normal_trade_count", "extended_trade_count",
            "portfolio_total_r",
        )
    }
    replay_match = assert_qualified_replay_matches_summary(expected_replay, replay_payload)
    bad_replay = dict(replay_payload)
    bad_replay["total_return_pct"] = 99.0
    try:
        assert_qualified_replay_matches_summary(expected_replay, bad_replay)
        replay_mismatch_rejected = False
    except ValueError:
        replay_mismatch_rejected = True
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "qualified_candidate_capture_must_preserve_formal_no_filter_strategy_results",
        (12.5, 4, 7.5, True),
        (
            float(replay_match["total_return_pct"]),
            int(replay_match["trade_count"]),
            float(replay_match["portfolio_total_r"]),
            replay_mismatch_rejected,
        ),
    )

    valid_metadata = {
        "comparison_mode": "hard-filter",
        "comparison_design": "historical_active_param_oos",
        "filter_id": BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
        "model_architecture": BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
        "experiment_profile": BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
        "lookahead_safe_active_param_schedule": True,
        "threshold_used_as_gate": True,
    }
    try:
        validate_qualified_audit_strategy_metadata(
            valid_metadata,
            filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
        )
        valid_metadata_accepted = True
    except ValueError:
        valid_metadata_accepted = False
    invalid_metadata_rejected = []
    for mutation in (
        {"comparison_design": "static_param_diagnostic"},
        {"lookahead_safe_active_param_schedule": False},
        {"threshold_used_as_gate": False},
    ):
        payload = dict(valid_metadata)
        payload.update(mutation)
        try:
            validate_qualified_audit_strategy_metadata(
                payload,
                filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
            )
            invalid_metadata_rejected.append(False)
        except ValueError:
            invalid_metadata_rejected.append(True)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "qualified_candidate_audit_accepts_only_lookahead_safe_hard_filter_historical_replay",
        (True, True, True, True),
        (valid_metadata_accepted, *invalid_metadata_rejected),
    )

    app_path = Path(__file__).resolve().parents[2] / "apps" / "breakout_quality.py"
    app_source = app_path.read_text(encoding="utf-8")
    app_tree = ast.parse(app_source, filename=str(app_path))
    command_modules = {}
    for node in app_tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "COMMAND_MODULES" for target in node.targets):
            command_modules = ast.literal_eval(node.value)
            break
    audit_source = _read_audit_source("qualified_candidate_set")
    strategy_source = (
        Path(__file__).resolve().parents[2]
        / "filters"
        / "breakout_quality"
        / "strategy_compare_engine.py"
    ).read_text(encoding="utf-8")
    engine_source = (
        Path(__file__).resolve().parents[2]
        / "core"
        / "portfolio_engine.py"
    ).read_text(encoding="utf-8")
    runner_source = (
        Path(__file__).resolve().parents[1]
        / "portfolio_sim"
        / "simulation_runner.py"
    ).read_text(encoding="utf-8")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "qualified_candidate_audit_is_cli_only_and_reuses_canonical_replay",
        (True, True, True, True, True, True),
        (
            command_modules.get("audit-qualified-candidate-set")
            == "tools.audit.breakout_quality.qualified_candidate_set",
            'print("[11] 11C Qualified Candidate-set Audit（research-only）")' not in app_source,
            'elif choice == "11":' not in app_source,
            "run_no_filter_candidate_replay_from_metadata" in audit_source,
            "replay_counts=replay_counts" in strategy_source,
            "replay_counts=None" in engine_source and "replay_counts=None" in runner_source,
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "qualified_candidate_audit_is_diagnostic_only_and_does_not_authorize_training",
        (True, True, True, True, True),
        (
            '"training_performed": False' in audit_source,
            '"research_only": True' in audit_source,
            '"audit_does_not_authorize_new_model": True' in audit_source,
            "torch.save(" not in audit_source,
            "optimizer" not in audit_source.lower(),
        ),
    )

    summary["command"] = "audit-qualified-candidate-set"
    summary["layers"] = ["all_oos_breakouts", "qualified_candidates", "orderable_candidates", "actual_trades"]
    return results, summary


def validate_breakout_quality_target_component_attribution_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_TARGET_COMPONENT_ATTRIBUTION"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    target = np.asarray([2.5, 1.5, 0.5, -0.2, -0.7, -1.0], dtype=np.float32)
    favorable = np.asarray([0.25, 0.20, 0.10, 0.08, 0.03, 0.00], dtype=np.float32)
    adverse = np.asarray([0.00, 0.05, 0.05, 0.10, 0.10, 0.10], dtype=np.float32)
    arrays = {
        "target_raw_r": target,
        "valid_mask": np.ones(len(target), dtype=np.bool_),
        "favorable_return": favorable,
        "adverse_return_to_peak": adverse,
        "opportunity_bar": np.ones(len(target), dtype=np.int16),
        "first_risk_breach_bar": np.asarray([-1, -1, -1, -1, -1, 1], dtype=np.int16),
    }
    contract = {
        "risk_budget_return": 0.10,
        "horizon_bars": 40,
        "full_horizon_time_penalty_r": 0.5,
    }
    frame = pd.DataFrame({
        "ticker": [f"T{i}" for i in range(len(target))],
        "target_date": [f"2021-01-{i + 1:02d}" for i in range(len(target))],
        "group_index": np.arange(len(target), dtype=np.int64),
        "label": [LABEL_PASS, LABEL_PASS, LABEL_PASS, LABEL_REJECT, LABEL_REJECT, LABEL_REJECT],
        "target_raw_r": target,
        "model_score": [0.95, 0.85, 0.75, 0.35, 0.25, 0.15],
        "r_multiple": [3.0, 2.0, 1.0, -0.2, -0.8, -1.1],
    })
    attached = target_attribution_attach_components(
        frame,
        arrays=arrays,
        target_contract=contract,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "target_components_reconstruct_fixed_11a_target",
        True,
        bool(np.allclose(attached["target_reconstructed_r"], target, rtol=0.0, atol=2e-5)),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "target_component_r_units_are_favorable_minus_adverse_minus_time",
        (2.5, 0.5, -1.0),
        tuple(round(float(value), 6) for value in attached.loc[[0, 2, 5], "target_reconstructed_r"]),
    )

    metrics = target_attribution_metrics(attached, include_realized_r=True)
    correlations = metrics["correlations"]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "target_attribution_preserves_score_target_and_target_realized_directions",
        (1.0, 1.0, 1.0),
        (
            round(float(correlations["score_vs_target"]), 6),
            round(float(correlations["target_vs_realized_r"]), 6),
            round(float(correlations["score_vs_realized_r"]), 6),
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "target_attribution_reports_pass_and_reject_conditional_metrics",
        (3, 3, 1.0, 1.0),
        (
            int(metrics["by_label"]["pass"]["row_count"]),
            int(metrics["by_label"]["reject"]["row_count"]),
            round(float(metrics["by_label"]["pass"]["correlations"]["target_vs_realized_r"]), 6),
            round(float(metrics["by_label"]["reject"]["correlations"]["target_vs_realized_r"]), 6),
        ),
    )
    markdown = render_target_attribution_markdown({
        "qualified_candidates": target_attribution_metrics(attached, include_realized_r=False),
        "actual_trades": metrics,
    })
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "target_attribution_markdown_exposes_label_conditional_decision_boundary",
        True,
        all(token in markdown for token in ("PASS", "REJECT", "Target↔R", "停止連續排序線")),
    )

    app_path = Path(__file__).resolve().parents[2] / "apps" / "breakout_quality.py"
    app_source = app_path.read_text(encoding="utf-8")
    app_tree = ast.parse(app_source, filename=str(app_path))
    command_modules = {}
    for node in app_tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target_node, ast.Name) and target_node.id == "COMMAND_MODULES" for target_node in node.targets):
            command_modules = ast.literal_eval(node.value)
            break
    audit_source = _read_audit_source("target_component_attribution")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "target_attribution_audit_is_cli_only",
        (True, True, True),
        (
            command_modules.get("audit-target-attribution")
            == "tools.audit.breakout_quality.target_component_attribution",
            "11D" not in app_source[app_source.find("def _run_interactive_menu"):app_source.find("def main")],
            'choice == "12"' not in app_source,
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "target_attribution_audit_is_strict_read_only_failure_attribution",
        (True, True, True, True, True),
        (
            "load_validated_continuous_target_component_arrays" in audit_source,
            '"training_performed": False' in audit_source,
            '"research_only": True' in audit_source,
            "torch.save(" not in audit_source,
            "optimizer" not in audit_source.lower(),
        ),
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        target_dir = resolve_continuous_target_dir(tmp_dir, "synthetic_filter")
        target_dir.mkdir(parents=True, exist_ok=True)
        artifact_arrays = {
            "target_raw_r": (TARGET_RAW_FILENAME, target),
            "valid_mask": (TARGET_VALID_MASK_FILENAME, np.ones(len(target), dtype=np.bool_)),
            "favorable_return": (TARGET_FAVORABLE_RETURN_FILENAME, favorable),
            "adverse_return_to_peak": (TARGET_ADVERSE_RETURN_FILENAME, adverse),
            "opportunity_bar": (TARGET_OPPORTUNITY_BAR_FILENAME, np.ones(len(target), dtype=np.int16)),
            "first_risk_breach_bar": (
                TARGET_RISK_BREACH_BAR_FILENAME,
                np.asarray([-1, -1, -1, -1, -1, 1], dtype=np.int16),
            ),
        }
        artifacts = {}
        for name, (filename, values) in artifact_arrays.items():
            path = target_dir / filename
            np.save(path, values, allow_pickle=False)
            artifacts[name] = {
                "filename": filename,
                "size_bytes": int(path.stat().st_size),
                "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
            }
        manifest = {
            "schema_version": CONTINUOUS_TARGET_SCHEMA_VERSION,
            "filter_id": "synthetic_filter",
            "group_count": len(target),
            "dataset_policy": {"synthetic": True},
            "target_contract": {"target_id": STRATEGY_ALIGNED_TARGET_ID},
            "artifacts": artifacts,
        }
        (target_dir / TARGET_MANIFEST_FILENAME).write_text(
            json.dumps(manifest, ensure_ascii=False),
            encoding="utf-8",
        )
        loaded_manifest, loaded_arrays = load_validated_continuous_target_component_arrays(
            tmp_dir,
            "synthetic_filter",
            expected_group_count=len(target),
            expected_dataset_policy={"synthetic": True},
        )
        favorable_path = target_dir / TARGET_FAVORABLE_RETURN_FILENAME
        favorable_path.write_bytes(favorable_path.read_bytes() + b"tamper")
        try:
            load_validated_continuous_target_component_arrays(
                tmp_dir,
                "synthetic_filter",
                expected_group_count=len(target),
                expected_dataset_policy={"synthetic": True},
            )
            tamper_rejected = False
        except ValueError:
            tamper_rejected = True
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_target_component_loader_validates_all_artifacts_and_rejects_tamper",
        (True, True, True),
        (
            str((loaded_manifest.get("target_contract") or {}).get("target_id")) == STRATEGY_ALIGNED_TARGET_ID,
            bool(np.allclose(loaded_arrays["favorable_return"], favorable)),
            tamper_rejected,
        ),
    )
    summary["command"] = "audit-target-attribution"
    summary["layers"] = ["qualified_candidates", "actual_trades", "pass", "reject"]
    return results, summary


def validate_breakout_quality_target_time_penalty_ablation_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_TARGET_TIME_PENALTY_ABLATION"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    favorable = np.asarray([3.2, 2.7, 2.2, 1.7, 1.2, 0.7], dtype=np.float64)
    adverse = np.asarray([0.2, 0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float64)
    time_penalty = np.asarray([0.0, 1.8, 0.0, 0.8, 0.0, 0.0], dtype=np.float64)
    no_time = favorable - adverse
    original = no_time - time_penalty
    frame = pd.DataFrame({
        "ticker": [f"T{i}" for i in range(6)],
        "target_date": [f"2021-02-{i + 1:02d}" for i in range(6)],
        "group_index": np.arange(6, dtype=np.int64),
        "label": [LABEL_PASS, LABEL_PASS, LABEL_PASS, LABEL_REJECT, LABEL_REJECT, LABEL_REJECT],
        "model_score": original,
        "target_raw_r": original,
        "favorable_r": favorable,
        "adverse_r": adverse,
        "time_penalty_r": time_penalty,
        "r_multiple": no_time,
    })
    attached = attach_time_penalty_ablation(frame)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "time_ablation_derives_fixed_favorable_minus_adverse_target",
        tuple(round(float(value), 6) for value in no_time),
        tuple(round(float(value), 6) for value in attached["target_no_time_r"]),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "time_ablation_reconstructs_original_target_exactly",
        True,
        bool(np.allclose(
            attached["target_original_reconstructed_r"],
            attached["target_raw_r"],
            rtol=0.0,
            atol=2e-5,
        )),
    )

    qualified_metrics = time_penalty_ablation_metrics(
        attached,
        include_realized_r=False,
    )
    actual_metrics = time_penalty_ablation_metrics(
        attached,
        include_realized_r=True,
    )
    corr = actual_metrics["correlations"]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "time_ablation_reports_original_and_no_time_economic_directions",
        (1.0, True),
        (
            round(float(corr["no_time_target_vs_realized_r"]), 6),
            float(corr["no_time_target_vs_realized_r"])
            > float(corr["original_target_vs_realized_r"]),
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "time_ablation_reports_positive_delta_and_decile_spread",
        (True, True),
        (
            float(actual_metrics["deltas"]["spearman_no_time_minus_original"]) > 0.0,
            float(actual_metrics["deltas"]["decile_spread_no_time_minus_original"]) >= 0.0,
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "time_ablation_preserves_pass_reject_conditional_rows",
        (3, 3),
        (
            int(actual_metrics["by_label"]["pass"]["row_count"]),
            int(actual_metrics["by_label"]["reject"]["row_count"]),
        ),
    )

    markdown = render_time_penalty_ablation_markdown({
        "qualified_candidates": qualified_metrics,
        "actual_trades": actual_metrics,
    })
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "time_ablation_markdown_exposes_single_change_and_training_boundary",
        True,
        all(token in markdown for token in (
            "target_no_time_r = favorable_r - adverse_r",
            "PASS",
            "REJECT",
            "不授權訓練",
            "不測time penalty反向加分",
        )),
    )

    app_path = Path(__file__).resolve().parents[2] / "apps" / "breakout_quality.py"
    app_source = app_path.read_text(encoding="utf-8")
    app_tree = ast.parse(app_source, filename=str(app_path))
    command_modules = {}
    for node in app_tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target_node, ast.Name) and target_node.id == "COMMAND_MODULES"
            for target_node in node.targets
        ):
            command_modules = ast.literal_eval(node.value)
            break
    audit_source = _read_audit_source("target_time_penalty_ablation")
    menu_source = app_source[
        app_source.find("def _run_interactive_menu") : app_source.find("def main")
    ]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "time_ablation_audit_is_cli_only",
        (True, True, True),
        (
            command_modules.get("audit-target-time-ablation")
            == "tools.audit.breakout_quality.target_time_penalty_ablation",
            "11E" not in menu_source,
            "audit-target-time-ablation" not in menu_source,
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "time_ablation_is_strict_single_read_only_ablation",
        (True, True, True, True, True, True),
        (
            "target_no_time_r = favorable_r - adverse_r" in audit_source,
            '"single_fixed_ablation": "remove_time_penalty_only"' in audit_source,
            '"training_performed": False' in audit_source,
            '"research_only": True' in audit_source,
            "torch.save(" not in audit_source,
            "optimizer" not in audit_source.lower(),
        ),
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "source.csv"
        frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
        report = {
            "artifacts": {
                "synthetic": {
                    "sha256": __import__("hashlib").sha256(csv_path.read_bytes()).hexdigest(),
                }
            }
        }
        accepted = time_ablation_validated_source_csv(
            report,
            key="synthetic",
            canonical_path=csv_path,
        ) == csv_path
        csv_path.write_text("tampered", encoding="utf-8")
        try:
            time_ablation_validated_source_csv(
                report,
                key="synthetic",
                canonical_path=csv_path,
            )
            tamper_rejected = False
        except ValueError:
            tamper_rejected = True
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "time_ablation_strictly_validates_11d_source_hash",
        (True, True),
        (accepted, tamper_rejected),
    )
    summary["command"] = "audit-target-time-ablation"
    summary["ablation"] = "remove_time_penalty_only"
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

    all_off_pair = _build_controlled_param_source_pair(
        {"kind": "single_param", "params": base},
        filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
        threshold=float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD),
        fixed_risk=None,
        comparison_mode=COMPARISON_MODE_SCORE_RANKING,
        optional_entry_filter_policy=OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
    )
    all_off_left = params_to_json_dict(all_off_pair[1])
    all_off_right = params_to_json_dict(all_off_pair[2])
    from tools.filters.breakout_quality.strategy_filter_gate import (
        _parse_args as parse_filter_gate_args,
        build_filter_gate_scenario_specs,
    )
    from config.breakout_quality import (
        BREAKOUT_QUALITY_STRATEGY_DATASET,
        BREAKOUT_QUALITY_STRATEGY_MAX_POSITIONS,
        BREAKOUT_QUALITY_STRATEGY_ROTATION,
    )

    gate_specs = build_filter_gate_scenario_specs()
    gate_defaults = parse_filter_gate_args([])
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "strategy_filter_gate_keeps_continuous_ranker_identity_when_main_workflow_is_binary",
        (
            BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
            BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
            STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE,
            BREAKOUT_QUALITY_STRATEGY_DATASET,
            BREAKOUT_QUALITY_STRATEGY_MAX_POSITIONS,
            BREAKOUT_QUALITY_STRATEGY_ROTATION,
        ),
        (
            gate_defaults.filter_id,
            gate_defaults.model_architecture,
            gate_defaults.experiment_profile,
            gate_defaults.dataset,
            gate_defaults.max_positions,
            gate_defaults.rotation,
        ),
    )
    from contextlib import nullcontext
    breakout_quality_app = __import__(
        "apps.breakout_quality", fromlist=["*"]
    )
    from config import breakout_quality as breakout_quality_config

    with patch.object(
        breakout_quality_config,
        "BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE",
        UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE,
    ):
        binary_menu_settings = (
            breakout_quality_config.get_breakout_quality_workflow_settings()
        )
    export_request = SimpleNamespace(
        filter_id=binary_menu_settings.filter_id,
        experiment_profile=binary_menu_settings.experiment_profile,
        evaluation_batch_size=4096,
        evaluation_workers=4,
        device="auto",
        mixed_precision_dtype="auto",
        mixed_precision=True,
        deterministic_algorithms=True,
        allow_tf32=False,
        preload_feature_bank=True,
    )
    post_train_calls = []

    def _capture_post_train_command(command, command_args, *, program_name):
        post_train_calls.append((command, tuple(command_args), program_name))
        return 0

    with (
        patch.object(
            breakout_quality_app,
            "_run_command",
            side_effect=_capture_post_train_command,
        ),
        patch.object(
            breakout_quality_app,
            "_compact_console_scope",
            side_effect=lambda: nullcontext(),
        ),
        redirect_stdout(io.StringIO()),
    ):
        post_train_rc = breakout_quality_app._run_binary_post_train_validation(
            export_request,
            workflow_settings=binary_menu_settings,
            program_name="apps/breakout_quality.py",
        )
    export_call = post_train_calls[0]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "binary_model_post_train_exports_forward_scores_without_strategy_replay",
        (0, ("export-scores",), True, True),
        (
            post_train_rc,
            tuple(call[0] for call in post_train_calls),
            "forward_oos" in export_call[1]
            and binary_menu_settings.experiment_profile in export_call[1],
            "strategy-compare" not in tuple(call[0] for call in post_train_calls),
        ),
    )

    project_root = Path(__file__).resolve().parents[2]
    app_source = (project_root / "apps" / "breakout_quality.py").read_text(
        encoding="utf-8"
    )
    trade_path_train_source = app_source.split(
        "def _interactive_trade_path_train_and_report", 1
    )[1].split("def _interactive_trade_path_existing_report", 1)[0]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "binary_menu_training_flow_builds_trade_path_label_and_complete_model_artifacts_only",
        True,
        all(
            token in trade_path_train_source
            for token in (
                "build-trade-path-labels",
                '"train"',
                '"export-scores"',
                '"report"',
                'scope="forward_oos"',
                "本流程不執行策略績效比較",
            )
        )
        and "_run_binary_post_train_validation" not in trade_path_train_source
        and "strategy-compare" not in trade_path_train_source,
    )

    report_calls = []

    def _capture_trade_path_report(command, command_args, *, program_name):
        report_calls.append((command, tuple(command_args), program_name))
        return 0

    with (
        patch.object(
            breakout_quality_app,
            "_run_command",
            side_effect=_capture_trade_path_report,
        ),
        redirect_stdout(io.StringIO()),
    ):
        trade_path_report_rc = breakout_quality_app._run_trade_path_model_report(
            "apps/breakout_quality.py",
            request=export_request,
            export_research_scores=True,
        )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "existing_trade_path_model_report_exports_research_and_forward_scores_without_replay",
        (0, ("export-scores", "report", "export-scores"), True, True),
        (
            trade_path_report_rc,
            tuple(call[0] for call in report_calls),
            "research" in report_calls[0][1]
            and "forward_oos" in report_calls[2][1],
            "strategy-compare" not in tuple(call[0] for call in report_calls),
        ),
    )

    existing_binary_source = app_source.split(
        "def _interactive_trade_path_existing_report", 1
    )[1].split("def _interactive_trade_path_label_summary", 1)[0]
    binary_menu_source = app_source.split(
        "def _interactive_binary_model_research", 1
    )[1].split("def _interactive_model_research", 1)[0]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "binary_model_menu_supports_trade_path_existing_model_report_without_strategy_replay",
        True,
        all(
            token in binary_menu_source
            for token in (
                "建立新Label",
                "使用既有模型",
                "查看Label與事件生命週期摘要",
                "_interactive_trade_path_existing_report",
            )
        )
        and "_run_trade_path_model_report" in existing_binary_source
        and "strategy-compare" not in existing_binary_source
        and "_run_binary_post_train_validation" not in existing_binary_source,
    )

    existing_model_calls = []

    def _capture_existing_model_command(command, command_args, *, program_name):
        existing_model_calls.append((command, tuple(command_args), program_name))
        return 0

    with (
        patch.object(
            breakout_quality_app,
            "_run_command",
            side_effect=_capture_existing_model_command,
        ),
        patch.object(
            breakout_quality_app,
            "_prompt_bool",
            return_value=True,
        ),
        patch.object(
            breakout_quality_app,
            "_print_policy_defaults",
            return_value=None,
        ),
        patch.object(
            breakout_quality_app,
            "_compact_console_scope",
            side_effect=lambda: nullcontext(),
        ),
        redirect_stdout(io.StringIO()),
    ):
        existing_model_rc = (
            breakout_quality_app._interactive_trade_path_existing_report(
                "apps/breakout_quality.py",
                workflow_settings=binary_menu_settings,
            )
        )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "binary_existing_trade_path_model_menu_completes_research_report_and_forward_scores_only",
        (
            0,
            ("export-scores", "report", "export-scores"),
            True,
            True,
        ),
        (
            existing_model_rc,
            tuple(call[0] for call in existing_model_calls),
            "research" in existing_model_calls[0][1]
            and "forward_oos" in existing_model_calls[2][1],
            "--include-oos" in existing_model_calls[1][1]
            and "strategy-compare" not in tuple(
                call[0] for call in existing_model_calls
            ),
        ),
    )

    # Regression: hard-filter comparison has no score-ranking capture audit.
    # The shared console renderer must receive None rather than an unbound local.
    from datetime import date as _date
    from filters.breakout_quality import strategy_compare_engine as strategy_compare_module

    with tempfile.TemporaryDirectory() as hard_filter_tmp_dir:
        hard_filter_root = Path(hard_filter_tmp_dir)
        hard_filter_params = hard_filter_root / "params.json"
        hard_filter_params.write_text("{}\n", encoding="utf-8")
        hard_filter_runtime = SimpleNamespace(
            execution_start=_date(2021, 1, 1),
            required_signal_start=_date(2020, 12, 31),
            available_from=_date(2020, 12, 31),
            available_through=_date(2021, 12, 31),
            manifest={
                "model_architecture": BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
                "experiment_profile": BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
                "fixed_evaluation_threshold": float(
                    BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD
                ),
                "runtime_eligibility": {},
                "score_table": {},
            },
            paths=SimpleNamespace(
                manifest_path=hard_filter_root / "manifest.json",
                score_path=hard_filter_root / "scores.csv",
            ),
        )
        hard_filter_profile = {
            "portfolio_capacity_rows": [],
            "closed_trade_rows": [],
        }
        hard_filter_payload = {
            "profile": hard_filter_profile,
            "trade_history": pd.DataFrame(),
            "equity_curve": pd.DataFrame(),
        }
        hard_filter_summary = {
            "total_return_pct": 1.0,
            "max_drawdown_pct": 1.0,
            "return_over_max_drawdown": 1.0,
            "trade_count": 0,
        }
        hard_filter_console_capture = []

        def _capture_hard_filter_console(*args, **kwargs):
            hard_filter_console_capture.append(kwargs.get("capture_result"))
            return "hard-filter-report"

        with (
            patch.object(
                strategy_compare_module,
                "load_runtime_artifact_contract",
                return_value=hard_filter_runtime,
            ),
            patch.object(
                strategy_compare_module,
                "_resolve_params_path",
                return_value=hard_filter_params,
            ),
            patch.object(
                strategy_compare_module,
                "_load_param_source",
                return_value={"kind": "single_param"},
            ),
            patch.object(
                strategy_compare_module,
                "_validate_requested_param_policy",
                return_value={
                    "selector": "single_param",
                    "member_count_min": 1,
                    "member_count_max": 1,
                    "min_agree": 1,
                },
            ),
            patch.object(
                strategy_compare_module,
                "_build_controlled_param_source_pair",
                return_value=(
                    "single_param",
                    base,
                    quality_filter,
                    {},
                    {},
                    None,
                ),
            ),
            patch.object(
                strategy_compare_module,
                "get_dataset_dir",
                return_value=str(hard_filter_root),
            ),
            patch.object(
                strategy_compare_module,
                "_run_scenario",
                side_effect=[hard_filter_payload, hard_filter_payload],
            ),
            patch.object(strategy_compare_module, "_assert_shared_benchmark"),
            patch.object(
                strategy_compare_module,
                "_scenario_summary",
                side_effect=[hard_filter_summary, hard_filter_summary],
            ),
            patch.object(
                strategy_compare_module,
                "_build_yearly_comparison",
                return_value=pd.DataFrame(
                    columns=[
                        "year",
                        "no_filter_return_pct",
                        "quality_filter_return_pct",
                        "is_full_year",
                    ]
                ),
            ),
            patch.object(
                strategy_compare_module,
                "_markdown_report",
                return_value="# hard-filter\n",
            ),
            patch.object(strategy_compare_module, "_remove_legacy_html_outputs"),
            patch.object(strategy_compare_module, "write_trade_attribution_outputs"),
            patch.object(
                strategy_compare_module,
                "_render_strategy_console_report",
                side_effect=_capture_hard_filter_console,
            ),
            patch.object(strategy_compare_module, "print_artifact_paths"),
            redirect_stdout(io.StringIO()),
        ):
            hard_filter_result = strategy_compare_module.run_comparison(
                project_root=hard_filter_root,
                dataset="full",
                params_path=hard_filter_params,
                param_policy=PARAM_POLICY_BASE_FINALIST_BEST,
                max_positions=10,
                enable_rotation=False,
                allow_static_diagnostic=True,
                comparison_mode=COMPARISON_MODE_HARD_FILTER,
                filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
                score_source=strategy_compare_module.SCORE_SOURCE_CANONICAL_RUNTIME,
                model_architecture=BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
                experiment_profile=BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
                output_dir_override=hard_filter_root / "comparison",
                quiet=True,
            )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "hard_filter_strategy_compare_passes_none_capture_audit_to_shared_console",
        (None, True),
        (
            hard_filter_console_capture[0]
            if hard_filter_console_capture
            else "console-not-called",
            "quality_filter" in hard_filter_result,
        ),
    )

    all_off_output_name = _comparison_output_dir_name(
        COMPARISON_MODE_SCORE_RANKING,
        _comparison_labels(COMPARISON_MODE_SCORE_RANKING),
        param_policy=PARAM_POLICY_BASE_FINALIST_BEST,
        ranking_policy="score",
        optional_entry_filter_policy=OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "optional_entry_filter_gate_adds_e_raw_score_with_all_five_filters_off",
        (
            (False, False, False, False, False),
            (False, False, False, False, False),
            (False, True),
            ("all-off", "score"),
            ("all-off", "capital-bucket-then-score"),
            True,
        ),
        (
            tuple(all_off_left[field] for field in OPTIONAL_ENTRY_FILTER_FIELDS),
            tuple(all_off_right[field] for field in OPTIONAL_ENTRY_FILTER_FIELDS),
            (
                all_off_left["use_breakout_quality_ranking"],
                all_off_right["use_breakout_quality_ranking"],
            ),
            (
                gate_specs["E"]["optional_entry_filter_policy"],
                gate_specs["E"]["ranking_policy"],
            ),
            (
                gate_specs["D"]["optional_entry_filter_policy"],
                gate_specs["D"]["ranking_policy"],
            ),
            all_off_output_name.endswith("_optional_entry_filters_all_off"),
        ),
    )

    from tools.filters.breakout_quality.strategy_dl_filter_gate import (
        RULE_LEVEL_SPECS,
        _parse_args as parse_dl_filter_gate_args,
        build_dl_filter_gate_scenario_specs,
        run_dl_filter_gate,
    )
    from filters.breakout_quality.strategy_compare_engine import (
        OPTIONAL_ENTRY_FILTER_POLICY_CURRENT,
    )
    from config.breakout_quality import (
        BREAKOUT_QUALITY_STRATEGY_DATASET,
        BREAKOUT_QUALITY_STRATEGY_MAX_POSITIONS,
        BREAKOUT_QUALITY_STRATEGY_PARAM_POLICY,
        BREAKOUT_QUALITY_STRATEGY_ROTATION,
    )

    dl_gate_specs = build_dl_filter_gate_scenario_specs()
    dl_gate_defaults = parse_dl_filter_gate_args([])
    dl_level_payloads = {}
    for level_id, level_spec in RULE_LEVEL_SPECS.items():
        pair = _build_controlled_param_source_pair(
            {"kind": "single_param", "params": base},
            filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
            threshold=float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD),
            fixed_risk=None,
            comparison_mode=COMPARISON_MODE_HARD_FILTER,
            optional_entry_filter_policy=level_spec[
                "optional_entry_filter_policy"
            ],
            shared_param_overrides=level_spec["shared_param_overrides"],
        )
        dl_level_payloads[level_id] = (
            params_to_json_dict(pair[1]),
            params_to_json_dict(pair[2]),
        )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "binary_dl_filter_rule_ablation_gate_defaults_and_levels_are_fixed",
        (
            BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
            BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
            BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
            BREAKOUT_QUALITY_STRATEGY_DATASET,
            BREAKOUT_QUALITY_STRATEGY_PARAM_POLICY,
            BREAKOUT_QUALITY_STRATEGY_MAX_POSITIONS,
            BREAKOUT_QUALITY_STRATEGY_ROTATION,
            None,
            None,
            tuple(f"{side}{level}" for level in "01234" for side in "AB"),
            ("current", {}, False, bool(base.use_history_threshold), bool(base.use_breakout_reclaim_reentry), bool(base.use_kc)),
            ("all-off", {}, False, bool(base.use_history_threshold), bool(base.use_breakout_reclaim_reentry), bool(base.use_kc)),
            ("all-off", {"use_history_threshold": False}, False, False, bool(base.use_breakout_reclaim_reentry), bool(base.use_kc)),
            (
                "all-off",
                {
                    "use_history_threshold": False,
                    "use_breakout_reclaim_reentry": False,
                },
                False,
                False,
                False,
                bool(base.use_kc),
            ),
            (
                "all-off",
                {
                    "use_history_threshold": False,
                    "use_breakout_reclaim_reentry": False,
                    "use_kc": False,
                },
                False,
                False,
                False,
                False,
            ),
            tuple(False for _ in OPTIONAL_ENTRY_FILTER_FIELDS),
            (False, True),
            (False, False),
        ),
        (
            dl_gate_defaults.filter_id,
            dl_gate_defaults.model_architecture,
            dl_gate_defaults.experiment_profile,
            dl_gate_defaults.dataset,
            dl_gate_defaults.param_policy,
            dl_gate_defaults.max_positions,
            dl_gate_defaults.rotation,
            dl_gate_defaults.start_date,
            dl_gate_defaults.end_date,
            tuple(dl_gate_specs),
            (
                RULE_LEVEL_SPECS["0"]["optional_entry_filter_policy"],
                RULE_LEVEL_SPECS["0"]["shared_param_overrides"],
                dl_level_payloads["0"][0]["use_breakout_quality_filter"],
                dl_level_payloads["0"][0]["use_history_threshold"],
                dl_level_payloads["0"][0]["use_breakout_reclaim_reentry"],
                dl_level_payloads["0"][0]["use_kc"],
            ),
            (
                RULE_LEVEL_SPECS["1"]["optional_entry_filter_policy"],
                RULE_LEVEL_SPECS["1"]["shared_param_overrides"],
                dl_level_payloads["1"][0]["use_breakout_quality_filter"],
                dl_level_payloads["1"][0]["use_history_threshold"],
                dl_level_payloads["1"][0]["use_breakout_reclaim_reentry"],
                dl_level_payloads["1"][0]["use_kc"],
            ),
            (
                RULE_LEVEL_SPECS["2"]["optional_entry_filter_policy"],
                RULE_LEVEL_SPECS["2"]["shared_param_overrides"],
                dl_level_payloads["2"][0]["use_breakout_quality_filter"],
                dl_level_payloads["2"][0]["use_history_threshold"],
                dl_level_payloads["2"][0]["use_breakout_reclaim_reentry"],
                dl_level_payloads["2"][0]["use_kc"],
            ),
            (
                RULE_LEVEL_SPECS["3"]["optional_entry_filter_policy"],
                RULE_LEVEL_SPECS["3"]["shared_param_overrides"],
                dl_level_payloads["3"][0]["use_breakout_quality_filter"],
                dl_level_payloads["3"][0]["use_history_threshold"],
                dl_level_payloads["3"][0]["use_breakout_reclaim_reentry"],
                dl_level_payloads["3"][0]["use_kc"],
            ),
            (
                RULE_LEVEL_SPECS["4"]["optional_entry_filter_policy"],
                RULE_LEVEL_SPECS["4"]["shared_param_overrides"],
                dl_level_payloads["4"][0]["use_breakout_quality_filter"],
                dl_level_payloads["4"][0]["use_history_threshold"],
                dl_level_payloads["4"][0]["use_breakout_reclaim_reentry"],
                dl_level_payloads["4"][0]["use_kc"],
            ),
            tuple(
                dl_level_payloads["4"][0][field]
                for field in OPTIONAL_ENTRY_FILTER_FIELDS
            ),
            (
                dl_level_payloads["4"][0]["use_breakout_quality_filter"],
                dl_level_payloads["4"][1]["use_breakout_quality_filter"],
            ),
            (
                dl_level_payloads["4"][0]["use_breakout_quality_ranking"],
                dl_level_payloads["4"][1]["use_breakout_quality_ranking"],
            ),
        ),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        fake_args = parse_dl_filter_gate_args([])

        def fake_dl_gate_comparison(**kwargs):
            filter_policy = kwargs["optional_entry_filter_policy"]
            shared_overrides = dict(kwargs.get("shared_param_overrides") or {})
            output_dir = Path(kwargs["output_dir_override"])
            output_dir.mkdir(parents=True, exist_ok=True)
            level_id = next(
                level
                for level, spec in RULE_LEVEL_SPECS.items()
                if spec["optional_entry_filter_policy"] == filter_policy
                and spec["shared_param_overrides"] == shared_overrides
            )
            level_number = float(level_id)
            left_return = 100.0 + level_number * 5.0
            right_return = left_return + (10.0 - level_number)
            left_year = 10.0 + level_number
            right_year = left_year + (1.0 - level_number * 0.1)
            exclusive_delta = 2.0 + level_number
            metadata = {
                "comparison_mode": COMPARISON_MODE_HARD_FILTER,
                "score_source": "canonical_runtime",
                "dataset": fake_args.dataset,
                "params_file_sha256": "param-sha",
                "param_source_kind": "rolling_active_param_ensemble",
                "requested_param_policy": fake_args.param_policy,
                "param_selector": "base_finalist_best",
                "runtime_member_count_min": 1,
                "runtime_member_count_max": 1,
                "runtime_min_agree": 1,
                "comparison_design": "historical_active_param_oos",
                "lookahead_safe_active_param_schedule": True,
                "filter_id": fake_args.filter_id,
                "model_architecture": fake_args.model_architecture,
                "experiment_profile": fake_args.experiment_profile,
                "threshold": float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD),
                "benchmark_ticker": "0050",
                "max_positions": fake_args.max_positions,
                "enable_rotation": False,
                "comparison_period": {
                    "start": "2021-01-01",
                    "end": "2021-12-31",
                },
                "score_signal_coverage": {
                    "required_start": "2020-12-31",
                    "available_through": "2021-12-31",
                },
                "runtime_manifest_path": "models/runtime_manifest.json",
                "runtime_score_path": "models/scores.csv",
                "optional_entry_filter_policy": filter_policy,
                "optional_entry_filter_forced_values": (
                    {field: False for field in OPTIONAL_ENTRY_FILTER_FIELDS}
                    if filter_policy == OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF
                    else None
                ),
                "shared_param_overrides": shared_overrides,
                "params_path": "models/roos_base_best.json",
            }
            base_summary = {
                "total_return_pct": left_return,
                "max_drawdown_pct": 10.0 + level_number,
                "return_over_max_drawdown": left_return / (10.0 + level_number),
                "expected_value_r": 0.2,
                "avg_exposure_pct": 70.0,
                "trade_count": 10 + int(level_number),
            }
            active_summary = {
                "total_return_pct": right_return,
                "max_drawdown_pct": 9.0 + level_number,
                "return_over_max_drawdown": right_return / (9.0 + level_number),
                "expected_value_r": 0.3,
                "avg_exposure_pct": 65.0,
                "trade_count": 8 + int(level_number),
            }
            (output_dir / "trade_attribution.json").write_text(
                json.dumps(
                    {
                        "r_attribution": {
                            "reconstructed_total_r_delta": exclusive_delta,
                            "exclusive_selection_delta_r": exclusive_delta,
                            "excluded_winner_r": 3.0,
                            "avoided_loser_r_abs": 4.0,
                            "replacement_winner_r": 5.0,
                            "replacement_loser_r_abs": 2.0,
                            "direct_filter_reject_count": 4,
                            "portfolio_path_displacement_count": 2,
                        }
                    }
                ),
                encoding="utf-8",
            )
            return {
                "metadata": metadata,
                "no_filter": base_summary,
                "quality_filter": active_summary,
                "yearly": [
                    {
                        "year": 2021,
                        "no_filter_return_pct": left_year,
                        "quality_filter_return_pct": right_year,
                        "is_full_year": True,
                    }
                ],
            }

        with (
            patch(
                "tools.filters.breakout_quality.strategy_dl_filter_gate._validate_binary_runtime_preflight",
                return_value=None,
            ),
            patch(
                "tools.filters.breakout_quality.strategy_dl_filter_gate.run_comparison",
                side_effect=fake_dl_gate_comparison,
            ),
        ):
            with redirect_stdout(io.StringIO()):
                gate_result = run_dl_filter_gate(
                    args=fake_args,
                    project_root=tmp_root,
                )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "binary_dl_filter_rule_ablation_gate_runs_one_optimizer_free_five_level_matrix",
        (
            10.0,
            9.0,
            8.0,
            7.0,
            6.0,
            20.0,
            6.0,
            5,
            15,
            False,
            False,
            False,
        ),
        (
            gate_result["pair_deltas"]["B0_minus_A0"]["total_return_pct"],
            gate_result["pair_deltas"]["B1_minus_A1"]["total_return_pct"],
            gate_result["pair_deltas"]["B2_minus_A2"]["total_return_pct"],
            gate_result["pair_deltas"]["B3_minus_A3"]["total_return_pct"],
            gate_result["pair_deltas"]["B4_minus_A4"]["total_return_pct"],
            gate_result["rule_ablation"]["A4_minus_A0"]["total_return_pct"],
            gate_result["pair_effects"]["B4_minus_A4"][
                "exclusive_selection_delta_r"
            ],
            len(gate_result["metadata"]["pair_artifacts"]),
            len(gate_result["yearly"][0]) - 2,
            gate_result["metadata"]["future_target_runtime_used"],
            gate_result["metadata"]["model_retrained"],
            gate_result["metadata"]["optimizer_executed"],
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

    projected_fraction, deployment_rate = calc_projected_capital_metrics(
        proj_cost=150.0,
        sizing_capital=1000.0,
        max_position_cap_pct=0.30,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "capital_deployment_uses_canonical_projected_cost_and_position_cap",
        (0.15, 0.50),
        (round(projected_fraction, 6), round(deployment_rate, 6)),
    )

    capital_rows = [
        {
            "ticker": "A", "sort_value": 0.05, "proj_cost": 100.0,
            "sizing_capital": 1000.0, "max_position_cap_pct": 0.30,
            "projected_capital_deployment_rate": 1.0 / 3.0,
            "use_breakout_quality_ranking": True,
            "breakout_quality_score": 0.90,
            "breakout_quality_rank": {"available": True},
        },
        {
            "ticker": "B", "sort_value": 0.10, "proj_cost": 200.0,
            "sizing_capital": 1000.0, "max_position_cap_pct": 0.30,
            "projected_capital_deployment_rate": 2.0 / 3.0,
            "use_breakout_quality_ranking": True,
            "breakout_quality_score": 0.78,
            "breakout_quality_rank": {"available": True},
        },
        {
            "ticker": "C", "sort_value": 0.20, "proj_cost": 300.0,
            "sizing_capital": 1000.0, "max_position_cap_pct": 0.30,
            "projected_capital_deployment_rate": 1.0,
            "use_breakout_quality_ranking": True,
            "breakout_quality_score": 0.68,
            "breakout_quality_rank": {"available": True},
        },
    ]
    r2_rows = [
        dict(row, breakout_quality_ranking_policy=BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_ADJUSTED)
        for row in capital_rows
    ]
    r3_rows = [
        dict(row, breakout_quality_ranking_policy=BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_BUCKET)
        for row in capital_rows
    ]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "capital_adjusted_score_multiplies_score_by_formal_deployment_rate",
        ["C", "B", "A"],
        [
            row["ticker"]
            for row in sort_candidate_rows(r2_rows, method=BUY_LIMIT_OVERAGE_SORT_METHOD)
        ],
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "capital_bucket_policy_uses_daily_tercile_then_score",
        ["C", "B", "A"],
        [
            row["ticker"]
            for row in sort_candidate_rows(r3_rows, method=BUY_LIMIT_OVERAGE_SORT_METHOD)
        ],
    )

    missing_capital_rows = [
        {
            "ticker": "A", "sort_value": 0.20, "proj_cost": 100.0,
            "use_breakout_quality_ranking": True,
            "breakout_quality_ranking_policy": BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_ADJUSTED,
            "breakout_quality_score": None,
            "breakout_quality_rank": {"available": False},
        },
        {
            "ticker": "B", "sort_value": 0.10, "proj_cost": 80.0,
            "use_breakout_quality_ranking": True,
            "breakout_quality_ranking_policy": BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_ADJUSTED,
            "breakout_quality_score": None,
            "breakout_quality_rank": {"available": False},
        },
    ]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "capital_aware_missing_scores_preserve_original_buy_sort_fallback",
        ["B", "A"],
        [
            row["ticker"]
            for row in sort_candidate_rows(
                missing_capital_rows, method=BUY_LIMIT_OVERAGE_SORT_METHOD
            )
        ],
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


    ensemble_capital_rows = []
    for ticker, votes, score, deployment_rate in (
        ("A", 2, 0.99, 0.10),
        ("B", 2, 0.80, 0.80),
        ("C", 3, 0.10, 1.00),
    ):
        for member_idx in range(votes):
            ensemble_capital_rows.append({
                "ticker": ticker,
                "ensemble_member_key": f"m{member_idx}",
                "params_obj": base,
                "sort_value": 0.0,
                "proj_cost": 100.0,
                "sizing_capital": 1000.0,
                "max_position_cap_pct": 0.30,
                "projected_capital_fraction": deployment_rate * 0.30,
                "projected_capital_deployment_rate": deployment_rate,
                "use_breakout_quality_ranking": True,
                "breakout_quality_ranking_policy": BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_ADJUSTED,
                "breakout_quality_score": score,
                "breakout_quality_score_date": "2025-01-02",
                "breakout_quality_rank": {
                    "score": score,
                    "available": True,
                    "unavailable_reason": "",
                    "score_date": "2025-01-02",
                    "score_source": "selection_point_in_time",
                    "shared_group_score": True,
                    "filter_id": BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
                },
            })
    ensemble_capital_ranked = _aggregate_ensemble_candidate_rows(
        ensemble_capital_rows, min_agree=1
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "ensemble_vote_precedes_capital_adjusted_score_and_equal_votes_use_r2",
        [("C", 3), ("B", 2), ("A", 2)],
        [
            (row["ticker"], row["ensemble_vote_count"])
            for row in ensemble_capital_ranked
        ],
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "ensemble_candidate_preserves_member_specific_original_quality_ranks",
        [f"m{idx}" for idx in range(6)],
        sorted(ensemble_ranked[0]["ensemble_member_quality_rank_by_key"]),
    )

    partial_ensemble_rows = [
        {
            "ticker": "A",
            "ensemble_member_key": "m0",
            "params_obj": base,
            "sort_value": 0.10,
            "proj_cost": 100.0,
            "use_breakout_quality_ranking": True,
            "breakout_quality_rank": {
                "score": 0.90,
                "available": True,
                "unavailable_reason": "",
                "score_date": "2017-06-22",
                "score_source": "selection_point_in_time",
                "shared_group_score": True,
                "filter_id": BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
            },
        },
        {
            "ticker": "A",
            "ensemble_member_key": "m1",
            "params_obj": base,
            "sort_value": 0.10,
            "proj_cost": 100.0,
            "use_breakout_quality_ranking": True,
            "breakout_quality_rank": {
                "score": None,
                "available": False,
                "unavailable_reason": "missing_ticker_date_score",
                "score_date": "2017-06-20",
                "score_source": "selection_point_in_time",
                "shared_group_score": True,
                "filter_id": BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
            },
        },
        {
            "ticker": "B",
            "ensemble_member_key": "m0",
            "params_obj": base,
            "sort_value": 0.20,
            "proj_cost": 100.0,
            "use_breakout_quality_ranking": True,
            "breakout_quality_rank": {
                "score": 0.50,
                "available": True,
                "unavailable_reason": "",
                "score_date": "2017-06-22",
                "score_source": "selection_point_in_time",
                "shared_group_score": True,
                "filter_id": BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
            },
        },
        {
            "ticker": "B",
            "ensemble_member_key": "m1",
            "params_obj": base,
            "sort_value": 0.20,
            "proj_cost": 100.0,
            "use_breakout_quality_ranking": True,
            "breakout_quality_rank": {
                "score": 0.50,
                "available": True,
                "unavailable_reason": "",
                "score_date": "2017-06-21",
                "score_source": "selection_point_in_time",
                "shared_group_score": True,
                "filter_id": BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
            },
        },
        {
            "ticker": "C",
            "ensemble_member_key": "m0",
            "params_obj": base,
            "sort_value": 0.05,
            "proj_cost": 100.0,
            "use_breakout_quality_ranking": True,
            "breakout_quality_rank": {
                "score": None,
                "available": False,
                "unavailable_reason": "missing_ticker_date_score",
                "score_date": "2017-06-22",
                "score_source": "selection_point_in_time",
                "shared_group_score": True,
                "filter_id": BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
            },
        },
        {
            "ticker": "C",
            "ensemble_member_key": "m1",
            "params_obj": base,
            "sort_value": 0.05,
            "proj_cost": 100.0,
            "use_breakout_quality_ranking": True,
            "breakout_quality_rank": {
                "score": None,
                "available": False,
                "unavailable_reason": "missing_ticker_date_score",
                "score_date": "2017-06-21",
                "score_source": "selection_point_in_time",
                "shared_group_score": True,
                "filter_id": BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
            },
        },
    ]
    partial_ensemble_ranked = _aggregate_ensemble_candidate_rows(
        partial_ensemble_rows, min_agree=2
    )
    partial_by_ticker = {row["ticker"]: row for row in partial_ensemble_ranked}
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "partial_ensemble_score_availability_falls_back_without_excluding_candidate",
        (
            ["B", "C", "A"],
            False,
            "partial_ensemble_member_score_availability",
            "partial_available_fallback",
            1,
            1,
            ["m0", "m1"],
        ),
        (
            [row["ticker"] for row in partial_ensemble_ranked],
            partial_by_ticker["A"]["breakout_quality_rank"]["available"],
            partial_by_ticker["A"]["breakout_quality_rank"]["unavailable_reason"],
            partial_by_ticker["A"]["ensemble_quality_score_availability"],
            partial_by_ticker["A"]["ensemble_quality_score_available_member_count"],
            partial_by_ticker["A"]["ensemble_quality_score_unavailable_member_count"],
            sorted(partial_by_ticker["A"]["ensemble_member_quality_rank_by_key"]),
        ),
    )

    same_date_inconsistent_rows = [dict(row) for row in partial_ensemble_rows[:2]]
    same_date_inconsistent_rows[1] = dict(same_date_inconsistent_rows[1])
    same_date_inconsistent_rows[1]["breakout_quality_rank"] = dict(
        same_date_inconsistent_rows[1]["breakout_quality_rank"],
        score_date="2017-06-22",
    )
    try:
        _aggregate_ensemble_candidate_rows(same_date_inconsistent_rows, min_agree=2)
        same_date_inconsistency_rejected = False
    except ValueError as exc:
        same_date_inconsistency_rejected = "ticker／score_date" in str(exc)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "same_ticker_same_score_date_availability_mismatch_remains_fail_fast",
        True,
        same_date_inconsistency_rejected,
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
    rolling_all_off_pair = _build_controlled_param_source_pair(
        rolling_source,
        filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
        threshold=float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD),
        fixed_risk=None,
        comparison_mode=COMPARISON_MODE_SCORE_RANKING,
        optional_entry_filter_policy=OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
    )
    rolling_all_off_left = rolling_all_off_pair[1][
        "params_ensemble_by_effective_date"
    ]["2021-01-01"][0]["params"]
    rolling_all_off_right = rolling_all_off_pair[2][
        "params_ensemble_by_effective_date"
    ]["2021-01-01"][0]["params"]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "rolling_optional_entry_filter_gate_keeps_schedule_and_forces_all_five_filters_off",
        (
            "rolling_active_param_ensemble",
            (False, False, False, False, False),
            (False, False, False, False, False),
            (False, True),
            2,
        ),
        (
            rolling_all_off_pair[0],
            tuple(
                rolling_all_off_left[field]
                for field in OPTIONAL_ENTRY_FILTER_FIELDS
            ),
            tuple(
                rolling_all_off_right[field]
                for field in OPTIONAL_ENTRY_FILTER_FIELDS
            ),
            (
                rolling_all_off_left["use_breakout_quality_ranking"],
                rolling_all_off_right["use_breakout_quality_ranking"],
            ),
            len(rolling_all_off_pair[1]["params_ensemble_by_effective_date"]),
        ),
    )
    rolling_dl_filter_pair = _build_controlled_param_source_pair(
        rolling_source,
        filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
        threshold=float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD),
        fixed_risk=None,
        comparison_mode=COMPARISON_MODE_HARD_FILTER,
        optional_entry_filter_policy=OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
    )
    rolling_dl_left = rolling_dl_filter_pair[1][
        "params_ensemble_by_effective_date"
    ]["2021-01-01"][0]["params"]
    rolling_dl_right = rolling_dl_filter_pair[2][
        "params_ensemble_by_effective_date"
    ]["2021-01-01"][0]["params"]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "rolling_binary_dl_replacement_gate_keeps_original_sort_and_forces_optional_filters_off",
        (
            "rolling_active_param_ensemble",
            (False, False, False, False, False),
            (False, False, False, False, False),
            (False, True),
            (False, False),
            2,
        ),
        (
            rolling_dl_filter_pair[0],
            tuple(rolling_dl_left[field] for field in OPTIONAL_ENTRY_FILTER_FIELDS),
            tuple(rolling_dl_right[field] for field in OPTIONAL_ENTRY_FILTER_FIELDS),
            (
                rolling_dl_left["use_breakout_quality_filter"],
                rolling_dl_right["use_breakout_quality_filter"],
            ),
            (
                rolling_dl_left["use_breakout_quality_ranking"],
                rolling_dl_right["use_breakout_quality_ranking"],
            ),
            len(rolling_dl_filter_pair[1]["params_ensemble_by_effective_date"]),
        ),
    )

    rolling_rule_ablation_pair = _build_controlled_param_source_pair(
        rolling_source,
        filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
        threshold=float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD),
        fixed_risk=None,
        comparison_mode=COMPARISON_MODE_HARD_FILTER,
        optional_entry_filter_policy=OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
        shared_param_overrides={
            "use_history_threshold": False,
            "use_breakout_reclaim_reentry": False,
            "use_kc": False,
        },
    )
    rolling_rule_left = rolling_rule_ablation_pair[1][
        "params_ensemble_by_effective_date"
    ]["2021-01-01"][0]["params"]
    rolling_rule_right = rolling_rule_ablation_pair[2][
        "params_ensemble_by_effective_date"
    ]["2021-01-01"][0]["params"]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "rolling_binary_dl_rule_ablation_applies_same_history_reentry_kc_overrides_to_both_sides",
        (
            (False, False, False),
            (False, False, False),
            (False, True),
            (False, False),
            2,
        ),
        (
            tuple(
                rolling_rule_left[field]
                for field in (
                    "use_history_threshold",
                    "use_breakout_reclaim_reentry",
                    "use_kc",
                )
            ),
            tuple(
                rolling_rule_right[field]
                for field in (
                    "use_history_threshold",
                    "use_breakout_reclaim_reentry",
                    "use_kc",
                )
            ),
            (
                rolling_rule_left["use_breakout_quality_filter"],
                rolling_rule_right["use_breakout_quality_filter"],
            ),
            (
                rolling_rule_left["use_breakout_quality_ranking"],
                rolling_rule_right["use_breakout_quality_ranking"],
            ),
            len(
                rolling_rule_ablation_pair[1][
                    "params_ensemble_by_effective_date"
                ]
            ),
        ),
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

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "capital_aware_ranking_policies_use_isolated_output_directories",
        (
            "strategy_compare_score_ranking_base_finalist_best_capital_adjusted_score",
            "strategy_compare_score_ranking_base_finalist_best_capital_bucket_then_score",
        ),
        (
            _comparison_output_dir_name(
                COMPARISON_MODE_SCORE_RANKING,
                _comparison_labels(COMPARISON_MODE_SCORE_RANKING),
                param_policy=PARAM_POLICY_BASE_FINALIST_BEST,
                ranking_policy=BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_ADJUSTED,
            ),
            _comparison_output_dir_name(
                COMPARISON_MODE_SCORE_RANKING,
                _comparison_labels(COMPARISON_MODE_SCORE_RANKING),
                param_policy=PARAM_POLICY_BASE_FINALIST_BEST,
                ranking_policy=BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_BUCKET,
            ),
        ),
    )

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "hard_filter_param_policies_use_isolated_output_directories",
        (
            "strategy_compare_base_finalist_best",
            "strategy_compare_base_finalists_agree",
        ),
        (
            _comparison_output_dir_name(
                COMPARISON_MODE_HARD_FILTER,
                _comparison_labels(COMPARISON_MODE_HARD_FILTER),
                param_policy=PARAM_POLICY_BASE_FINALIST_BEST,
            ),
            _comparison_output_dir_name(
                COMPARISON_MODE_HARD_FILTER,
                _comparison_labels(COMPARISON_MODE_HARD_FILTER),
                param_policy=PARAM_POLICY_BASE_FINALISTS_AGREE,
            ),
        ),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        compare_root = Path(tmp_dir)
        expected_compare_dir = compare_root / "strategy_compare_base_finalist_best"
        expected_compare_dir.mkdir()
        discovered_compare_dir = _first_existing_comparison_dir(
            compare_root,
            comparison_mode=COMPARISON_MODE_HARD_FILTER,
        )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "hard_filter_policy_specific_output_is_discoverable",
        expected_compare_dir.name,
        discovered_compare_dir.name,
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


def validate_breakout_quality_no_time_target_selection_audit_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_NO_TIME_TARGET_SELECTION_AUDIT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    source_contract = StrategyAlignedContinuousTargetSpec.from_label_policy(
        DEFAULT_LABEL_POLICY
    ).contract_payload()
    contract = build_strategy_aligned_no_time_contract(source_contract)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "no_time_target_contract_is_versioned_fixed_and_selection_only",
        (
            STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
            STRATEGY_ALIGNED_TARGET_ID,
            False,
            True,
        ),
        (
            contract.get("target_id"),
            contract.get("source_target_id"),
            contract.get("time_penalty_included"),
            contract.get("current_audit_oos_rows_evaluated") is False,
        ),
    )

    approved_settings = SimpleNamespace(
        is_continuous_ranker=True,
        filter_id="synthetic",
        experiment_profile="strategy_aligned_no_time_pass_magnitude_mse",
        continuous_target_id=STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
    )
    with patch(
        "tools.audit.breakout_quality.no_time_continuous_target.get_breakout_quality_workflow_settings",
        return_value=approved_settings,
    ):
        approved_gate = approved_no_time_workflow_rebuild_gate(filter_id="synthetic")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "no_time_target_active_workflow_rebuild_uses_fixed_formula_without_rechecking_history",
        ("active_workflow_profile", False, True),
        (
            approved_gate.get("approval_basis"),
            approved_gate.get("historical_research_gate_recomputed"),
            approved_gate.get("fixed_formula_only"),
        ),
    )

    favorable_return = np.asarray([0.30, 0.20, 0.10, np.nan], dtype=np.float32)
    adverse_return = np.asarray([0.02, 0.04, 0.01, np.nan], dtype=np.float32)
    opportunity = np.asarray([2, 4, 1, -1], dtype=np.int16)
    risk_breach = np.asarray([-1, 5, 1, -1], dtype=np.int16)
    valid = np.asarray([True, True, True, False], dtype=bool)
    arrays = build_strategy_aligned_no_time_group_targets(
        favorable_return=favorable_return,
        adverse_return_to_peak=adverse_return,
        opportunity_bar=opportunity,
        first_risk_breach_bar=risk_breach,
        valid_mask=valid,
        risk_budget_return=0.10,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "no_time_target_arrays_equal_favorable_minus_adverse_in_r_units",
        (2.8, 1.6, 0.9, True),
        (
            round(float(arrays["target_raw_r"][0]), 6),
            round(float(arrays["target_raw_r"][1]), 6),
            round(float(arrays["target_raw_r"][2]), 6),
            bool(np.isnan(arrays["target_raw_r"][3])),
        ),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        target_dir = resolve_continuous_target_dir(
            tmp_dir,
            "synthetic",
            target_id=STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / TARGET_RAW_FILENAME
        valid_path = target_dir / TARGET_VALID_MASK_FILENAME
        np.save(target_path, arrays["target_raw_r"], allow_pickle=False)
        np.save(valid_path, arrays["valid_mask"], allow_pickle=False)
        sha = __import__("hashlib").sha256
        manifest = {
            "schema_version": CONTINUOUS_TARGET_SCHEMA_VERSION,
            "filter_id": "synthetic",
            "target_contract": contract,
            "group_count": 4,
            "dataset_policy": {"synthetic": True},
            "artifacts": {
                "target_raw_r": {
                    "filename": target_path.name,
                    "size_bytes": target_path.stat().st_size,
                    "sha256": sha(target_path.read_bytes()).hexdigest(),
                },
                "valid_mask": {
                    "filename": valid_path.name,
                    "size_bytes": valid_path.stat().st_size,
                    "sha256": sha(valid_path.read_bytes()).hexdigest(),
                },
            },
        }
        (target_dir / TARGET_MANIFEST_FILENAME).write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        loaded_manifest, loaded_target, loaded_valid = load_validated_continuous_target_arrays(
            tmp_dir,
            "synthetic",
            target_id=STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
            expected_group_count=4,
            expected_dataset_policy={"synthetic": True},
        )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "no_time_target_version_is_loadable_by_strict_public_loader",
        (STRATEGY_ALIGNED_NO_TIME_TARGET_ID, True, True),
        (
            (loaded_manifest.get("target_contract") or {}).get("target_id"),
            bool(np.allclose(loaded_target[:3], arrays["target_raw_r"][:3])),
            bool(np.array_equal(loaded_valid, arrays["valid_mask"])),
        ),
    )

    frame = pd.DataFrame({
        "date": pd.to_datetime([
            "2018-01-02", "2018-01-02", "2018-01-03", "2018-01-03",
            "2020-01-02", "2020-01-02",
        ]),
        "label": [LABEL_PASS, LABEL_REJECT, LABEL_PASS, LABEL_REJECT, LABEL_PASS, LABEL_REJECT],
        "target_raw_r": [2.8, 0.5, 1.8, -0.2, 2.2, 0.1],
        "source_target_raw_r": [2.5, 0.4, 1.2, -0.3, 1.8, 0.0],
        "max_upside_return": [0.30, 0.08, 0.22, 0.02, 0.26, 0.04],
        "decision_mfe_return": [0.28, 0.07, 0.20, 0.01, 0.24, 0.03],
        "decision_mae_return": [-0.02, -0.05, -0.03, -0.08, -0.02, -0.07],
        "valid_mask": [True] * 6,
        "is_inner_train": [True, True, True, True, False, False],
        "is_validation": [False, False, False, False, True, True],
        "is_selection": [True] * 6,
    })
    metrics, daily = no_time_target_selection_metrics(frame)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "no_time_target_selection_metrics_cover_only_three_selection_splits",
        (("inner_train", "selection", "validation"), 6, True),
        (
            tuple(sorted(metrics)),
            int(metrics["selection"]["group_count"]),
            set(daily["split"].astype(str)) == {"inner_train", "validation", "selection"},
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "no_time_target_selection_metrics_preserve_rankability_and_source_comparison",
        (1.0, 1.0, True),
        (
            round(float(metrics["validation"]["same_day_rankability"]["rankable_date_rate"]), 6),
            round(float(metrics["selection"]["same_day_rankability"]["pairwise_non_tie_rate"]), 6),
            float(metrics["selection"]["source_vs_no_time_spearman"]) > 0.0,
        ),
    )

    markdown = render_no_time_target_markdown({
        "target_contract": contract,
        "split_metrics": metrics,
        "source_11e_gate": {
            "overall_spearman_delta": 0.0859,
            "pass_spearman_delta": 0.0942,
            "decile_spread_delta": 0.6029,
        },
    })
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "no_time_target_markdown_exposes_selection_only_and_iterative_oos_boundary",
        True,
        all(token in markdown for token in (
            "Selection-only",
            "OOS邊界",
            "不建立OOS指標",
            "不訓練",
            "prior" if False else "迭代OOS",
        )),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        ranker_dir = Path(tmp_dir)
        audit_dir = ranker_dir / "target_time_penalty_ablation_audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        qualified_path = audit_dir / "qualified_time_penalty_ablation.csv"
        actual_path = audit_dir / "actual_trade_time_penalty_ablation.csv"
        pd.DataFrame({"x": [1]}).to_csv(qualified_path, index=False, encoding="utf-8-sig")
        pd.DataFrame({"x": [2]}).to_csv(actual_path, index=False, encoding="utf-8-sig")
        sha = __import__("hashlib").sha256
        report = {
            "status": "RESULT_AVAILABLE_PENDING_REVIEW",
            "source_continuous_target_id": STRATEGY_ALIGNED_TARGET_ID,
            "interpretation_contract": {
                "research_only": True,
                "training_performed": False,
            },
            "artifacts": {
                "qualified_ablation": {"sha256": sha(qualified_path.read_bytes()).hexdigest()},
                "actual_trade_ablation": {"sha256": sha(actual_path.read_bytes()).hexdigest()},
            },
            "actual_trades": {
                "correlations": {
                    "original_target_vs_realized_r": 0.40,
                    "no_time_target_vs_realized_r": 0.49,
                },
                "deltas": {
                    "spearman_no_time_minus_original": 0.09,
                    "decile_spread_no_time_minus_original": 0.60,
                },
                "by_label": {
                    "pass": {
                        "correlations": {
                            "original_target_vs_realized_r": 0.36,
                            "no_time_target_vs_realized_r": 0.45,
                        }
                    }
                },
            },
        }
        report_path = audit_dir / "target_time_penalty_ablation_audit.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        with patch(
            "tools.audit.breakout_quality.no_time_continuous_target._ranker_dir",
            return_value=ranker_dir,
        ):
            accepted, accepted_path = validated_11e_report_for_no_time_target(
                filter_id="synthetic",
                ranker_profile=STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE,
            )
            accepted_ok = accepted_path == report_path and accepted.get("status") == report["status"]
            actual_path.write_text("tampered", encoding="utf-8")
            try:
                validated_11e_report_for_no_time_target(
                    filter_id="synthetic",
                    ranker_profile=STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE,
                )
                tamper_rejected = False
            except ValueError:
                tamper_rejected = True
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "no_time_target_requires_positive_11e_gate_and_strict_artifact_hashes",
        (True, True),
        (accepted_ok, tamper_rejected),
    )

    app_path = Path(__file__).resolve().parents[2] / "apps" / "breakout_quality.py"
    app_source = app_path.read_text(encoding="utf-8")
    app_tree = ast.parse(app_source, filename=str(app_path))
    command_modules = {}
    for node in app_tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target_node, ast.Name) and target_node.id == "COMMAND_MODULES"
            for target_node in node.targets
        ):
            command_modules = ast.literal_eval(node.value)
            break
    menu_source = app_source[
        app_source.find("def _run_interactive_menu") : app_source.find("def main")
    ]
    audit_source = _read_audit_source("no_time_continuous_target")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "no_time_target_audit_is_cli_only_and_read_only",
        (True, True, True, True, True),
        (
            command_modules.get("audit-no-time-target")
            == "tools.audit.breakout_quality.no_time_continuous_target",
            "11F" not in menu_source,
            "audit-no-time-target" not in menu_source,
            "torch.save(" not in audit_source,
            "optimizer" not in audit_source.lower(),
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "no_time_target_audit_does_not_compute_oos_metrics",
        (True, True, True),
        (
            'SELECTION_SPLITS = ("inner_train", "validation", "selection")' in audit_source,
            '"oos_evaluated": False' in audit_source,
            '"oos_rows_scores_labels_or_statistics_evaluated": False' in audit_source,
        ),
    )

    summary["command"] = "audit-no-time-target"
    summary["target_id"] = STRATEGY_ALIGNED_NO_TIME_TARGET_ID
    return results, summary


def validate_breakout_quality_pass_realization_gap_attribution_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_PASS_REALIZATION_GAP_ATTRIBUTION"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from tools.audit.breakout_quality.pass_realization_gap import (
        attach_no_time_components,
        partial_spearman,
        realization_gap_metrics,
        render_markdown,
    )

    rng = np.random.default_rng(10)
    count = 30
    score = np.arange(count, dtype=np.float64) / float(count - 1)
    latent = rng.normal(size=count)
    target = 0.5 * score + latent
    target = target - float(target.min()) + 0.6
    realized = 1.2 * latent - 2.5 * score + rng.normal(scale=0.1, size=count)
    adverse = 0.1 + 0.2 * np.linspace(0.0, 1.0, count)
    favorable = target + adverse
    frame = pd.DataFrame({
        "group_index": np.arange(count, dtype=np.int64),
        "label": np.full(count, LABEL_PASS, dtype=np.int64),
        "target_raw_r": target,
        "model_score": score,
        "r_multiple": realized,
    })
    arrays = {
        "target_raw_r": target.astype(np.float32),
        "favorable_return": favorable.astype(np.float32) * 0.1,
        "adverse_return_to_peak": adverse.astype(np.float32) * 0.1,
        "valid_mask": np.ones(count, dtype=bool),
    }
    attached = attach_no_time_components(
        frame,
        arrays=arrays,
        risk_budget_return=0.1,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "pass_realization_gap_components_reconstruct_no_time_target",
        (True, True, True),
        (
            bool(np.allclose(attached["favorable_r"] - attached["adverse_r"], attached["target_raw_r"], atol=1e-5, rtol=0.0)),
            bool((attached["label"] == LABEL_PASS).all()),
            bool(np.isfinite(attached[["target_raw_r", "favorable_r", "adverse_r", "model_score"]].to_numpy(dtype=np.float64)).all()),
        ),
    )

    metrics = realization_gap_metrics(attached, include_realized_r=True)
    corr = metrics["correlations"]
    partial = metrics["partial_correlations"]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "pass_realization_gap_detects_target_learning_but_negative_realized_r",
        (True, True, True, True),
        (
            float(corr["score_vs_target"]) > 0.20,
            float(corr["target_vs_realized_r"]) > 0.45,
            float(corr["score_vs_realized_r"]) < -0.10,
            float(corr["score_vs_realization_gap_r"]) > 0.80,
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "pass_realization_gap_partial_rank_controls_target",
        (True, True),
        (
            float(partial["score_vs_realized_r_controlling_target"]) < -0.50,
            partial_spearman(score, realized, [target]) is not None,
        ),
    )
    score_deciles = metrics["score_deciles"]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "pass_realization_gap_score_deciles_expose_unrealized_opportunity",
        (True, True, True),
        (
            float(score_deciles["top"]["mean_target_raw_r"]) > float(score_deciles["bottom"]["mean_target_raw_r"]),
            float(score_deciles["top"]["mean_realization_gap_r"]) > float(score_deciles["bottom"]["mean_realization_gap_r"]),
            float(score_deciles["top"]["mean_r_multiple"]) < float(score_deciles["bottom"]["mean_r_multiple"]),
        ),
    )

    markdown = render_markdown({
        "oos_pass": realization_gap_metrics(attached.drop(columns=["r_multiple"]), include_realized_r=False),
        "actual_pass": {key: value for key, value in metrics.items() if key != "frame"},
    })
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "pass_realization_gap_report_exposes_gap_capture_partial_and_boundary",
        (True, True, True, True),
        (
            "Target−R gap" in markdown,
            "R÷Favorable" in markdown,
            "Partial Score↔R" in markdown,
            "strategy-realization target audit" in markdown,
        ),
    )

    root = Path(__file__).resolve().parents[2]
    app_path = root / "apps" / "breakout_quality.py"
    app_source = app_path.read_text(encoding="utf-8")
    tree = ast.parse(app_source, filename=str(app_path))
    command_modules = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target_node, ast.Name) and target_node.id == "COMMAND_MODULES"
            for target_node in node.targets
        ):
            command_modules = ast.literal_eval(node.value)
            break
    menu_source = app_source[
        app_source.find("def _run_interactive_menu") : app_source.find("def main")
    ]
    audit_source = _read_audit_source("pass_realization_gap")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "pass_realization_gap_audit_is_cli_only_and_registered",
        (True, True, True, True),
        (
            command_modules.get("audit-pass-realization-gap")
            == "tools.audit.breakout_quality.pass_realization_gap",
            "11H" not in menu_source,
            "audit-pass-realization-gap" not in menu_source,
            "STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE" in audit_source,
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "pass_realization_gap_audit_strictly_validates_sources",
        (True, True, True, True),
        (
            "11H偵測到11G scores SHA256不一致" in audit_source,
            "11H偵測到11A trade matches SHA256不一致" in audit_source,
            "No-time target無法由favorable-adverse逐筆重建" in audit_source,
            "actual PASS配對數與11G report不一致" in audit_source,
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "pass_realization_gap_audit_is_read_only_and_does_not_authorize_model",
        (True, True, True, True),
        (
            '"research_only": True' in audit_source,
            '"training_performed": False' in audit_source,
            '"new_model_not_authorized_until_result_review": True' in audit_source,
            "torch.save" not in audit_source,
        ),
    )

    summary["ranker_profile"] = STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE
    summary["audit"] = "pass_realization_gap"
    return results, summary


def validate_breakout_quality_selection_strategy_realization_contract_case(_base_params):
    from config.training_policy import OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT
    from tools.audit.breakout_quality.selection_strategy_realization import parse_args as parse_selection_strategy_realization_args

    case_id = "BREAKOUT_QUALITY_SELECTION_STRATEGY_REALIZATION"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from tools.audit.breakout_quality.selection_strategy_realization import (
        _attach_targets,
        _trade_metrics,
        _unique_signals,
        _validate_replay_target_bounds,
        _write_prepare_script,
        default_params_path,
        default_research_models_dir,
    )

    lookup = pd.DataFrame({
        "ticker": ["A", "B", "C"],
        "target_date": ["2018-01-02", "2018-01-03", "2018-01-04"],
        "group_index": [0, 1, 2],
        "label": [LABEL_PASS, LABEL_PASS, LABEL_REJECT],
        "target_raw_r": [2.0, 0.5, -0.2],
        "target_valid": [True, True, True],
    })
    candidates = pd.DataFrame({
        "ticker": ["A", "A", "B", "C"],
        "trade_date": ["2018-01-03", "2018-01-04", "2018-01-04", "2018-01-05"],
        "candidate_date": ["2018-01-02", "2018-01-02", "2018-01-03", "2018-01-04"],
        "signal_date": ["2018-01-02", "2018-01-02", "2018-01-03", "2018-01-04"],
        "candidate_type": ["normal", "extended", "normal", "normal"],
    })
    attached = _attach_targets(candidates, lookup)
    unique = _unique_signals(attached)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "selection_strategy_realization_candidate_mapping_uses_signal_date_and_dedup",
        (4, 3, [0, 1, 2], True),
        (
            len(attached),
            len(unique),
            sorted(unique["group_index"].astype(int).tolist()),
            bool(unique["target_match"].all()),
        ),
    )

    trades = pd.DataFrame({
        "ticker": ["A", "B", "C"],
        "entry_date": ["2018-01-03", "2018-01-04", "2018-01-05"],
        "signal_date": ["2018-01-02", "2018-01-03", "2018-01-04"],
        "candidate_date": ["2018-01-02", "2018-01-03", "2018-01-04"],
        "r_multiple": [2.5, 0.2, -0.5],
    })
    trade_attached = _attach_targets(trades, lookup)
    metrics = _trade_metrics(trade_attached)
    target_manifest = {
        "split_report": {
            "final_refit_date_range": {"start": "2011-01-03", "end": "2020-11-05"}
        }
    }
    valid_bounds = _validate_replay_target_bounds(
        target_manifest,
        start_date="2014-01-01",
        end_date="2020-11-05",
    )
    try:
        _validate_replay_target_bounds(
            target_manifest,
            start_date="2014-01-01",
            end_date="2020-12-31",
        )
        embargo_rejected = False
    except ValueError:
        embargo_rejected = True
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "selection_strategy_realization_replay_respects_selection_target_boundary",
        (("2011-01-03", "2020-11-05"), True),
        (valid_bounds, embargo_rejected),
    )

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "selection_strategy_realization_metrics_preserve_target_and_r_direction",
        (3, 3, True, 2, 1),
        (
            metrics["trade_count"],
            metrics["matched_trade_count"],
            float(metrics["spearman_target_vs_realized_r"]) > 0.5,
            metrics["label_conditional"]["PASS"]["rows"],
            metrics["label_conditional"]["REJECT"]["rows"],
        ),
    )

    default_args = parse_selection_strategy_realization_args([])
    override_args = parse_selection_strategy_realization_args(["--optimizer-trials", "17"])
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "selection_nested_roos_trial_default_uses_training_policy_single_source",
        (int(OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT), 17),
        (int(default_args.optimizer_trials), int(override_args.optimizer_trials)),
    )

    research_dir = default_research_models_dir()
    params_path = default_params_path()
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "selection_nested_roos_uses_isolated_research_models_dir",
        (True, True, "roos_base_finalists_agree.json"),
        (
            "models" in research_dir.parts and "research" in research_dir.parts,
            params_path.parent == research_dir,
            params_path.name,
        ),
    )

    root = Path(__file__).resolve().parents[2]
    app_path = root / "apps" / "breakout_quality.py"
    app_source = app_path.read_text(encoding="utf-8")
    tree = ast.parse(app_source, filename=str(app_path))
    command_modules = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target_node, ast.Name) and target_node.id == "COMMAND_MODULES"
            for target_node in node.targets
        ):
            command_modules = ast.literal_eval(node.value)
            break
    menu_source = app_source[app_source.find("def _run_interactive_menu") : app_source.find("def main")]
    audit_source = _read_audit_source("selection_strategy_realization")
    optimizer_path = root / "tools" / "optimizer" / "outer_rolling_oos.py"
    optimizer_source = optimizer_path.read_text(encoding="utf-8")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "selection_strategy_realization_is_cli_only_and_registered",
        (True, True, True),
        (
            command_modules.get("audit-selection-strategy-realization")
            == "tools.audit.breakout_quality.selection_strategy_realization",
            "audit-selection-strategy-realization" not in menu_source,
            "11I" not in menu_source,
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "selection_nested_roos_prepare_contract_is_lookahead_safe_and_isolated",
        (True, True, True, True, True, True),
        (
            "DEFAULT_START_DATE = \"2014-01-01\"" in audit_source,
            "DEFAULT_REPLAY_END_DATE = \"2020-11-05\"" in audit_source,
            "DEFAULT_NESTED_OOS_END_DATE = \"2020-12-31\"" in audit_source,
            "--outer-train-window-months" in audit_source and "DEFAULT_TRAIN_WINDOW_MONTHS = 120" in audit_source,
            "V16_MODELS_DIR" in audit_source,
            "resolve_models_dir(project_root, environ=environ)" in optimizer_source,
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "selection_nested_roos_trial_default_has_no_duplicate_magic_number",
        (True, True),
        (
            "default=OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT" in audit_source,
            'parser.add_argument("--optimizer-trials", type=int, default=1000)' not in audit_source,
        ),
    )
    from tools.optimizer.outer_rolling_oos import OuterRollingConfig, _write_reports
    with tempfile.TemporaryDirectory() as td:
        temp_root = Path(td)
        prepare_path = _write_prepare_script(
            output_dir=temp_root / "prepare",
            trials=17,
            dataset_profile="reduced",
        )
        prepare_source = prepare_path.read_text(encoding="utf-8-sig")
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "selection_nested_roos_prepare_script_preserves_dataset_and_cleans_env",
            (True, True, True, True),
            (
                "--dataset reduced" in prepare_source,
                "--trials 17" in prepare_source,
                "--outer-last-oos-date 2020-12-31" in prepare_source,
                "Remove-Item Env:V16_MODELS_DIR" in prepare_source,
            ),
        )
        isolated_models = temp_root / "isolated_models"
        write_result = _write_reports(
            project_root=str(temp_root),
            output_dir=str(temp_root / "outputs"),
            session_ts="synthetic",
            rows=[],
            config=OuterRollingConfig(
                training_start_year=2004,
                first_oos_year=2014,
                last_oos_year=2020,
                trials_per_fold=1,
                train_window_months=120,
                oos_horizon_months=12,
            ),
            models_dir=str(isolated_models),
        )
        written_paths = [Path(path) for path in (write_result.get("paramsets") or {}).values()]
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "selection_nested_roos_writer_behaves_as_isolated_override",
            (True, True, False),
            (
                bool(written_paths),
                bool(written_paths) and all(path.parent == isolated_models for path in written_paths),
                (temp_root / "models").exists(),
            ),
        )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "selection_strategy_realization_does_not_label_untraded_candidates_or_authorize_training",
        (True, True, True, True),
        (
            "untraded_candidates_are_not_labeled_zero" in audit_source,
            '"training_performed": False' in audit_source,
            '"runtime_eligible": False' in audit_source,
            "torch.save" not in audit_source,
        ),
    )

    summary["audit"] = "selection_strategy_realization"
    return results, summary


def validate_breakout_quality_candidate_counterfactual_execution_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_CANDIDATE_COUNTERFACTUAL_EXECUTION"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from core.portfolio_engine import _candidate_execution_replay_snapshot
    from core.strategy_params import V16StrategyParams
    from tools.audit.breakout_quality.candidate_counterfactual_execution import (
        CandidateCounterfactualReplay,
        _execution_market_dates,
        _metrics as counterfactual_metrics,
        _render_markdown as render_counterfactual_markdown,
        _run_offline_counterfactual,
    )

    dates = tuple(pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]))
    fast = {
        "_packed_market_data": True,
        "dates": dates,
        "date_to_pos": {date: idx for idx, date in enumerate(dates)},
        "security_profile": None,
        "Open": np.asarray([99.0, 99.0, 89.0]),
        "High": np.asarray([100.0, 103.0, 90.0]),
        "Low": np.asarray([98.0, 98.0, 88.0]),
        "Close": np.asarray([99.0, 102.0, 89.0]),
        "Volume": np.asarray([1000.0, 1000.0, 1000.0]),
        "ATR": np.asarray([5.0, 5.0, 5.0]),
        "buy_limit": np.asarray([100.0, 100.0, 100.0]),
        "is_setup": np.asarray([False, False, False]),
        "ind_sell_signal": np.asarray([False, False, False]),
    }
    params = V16StrategyParams()
    candidate = {
        "ticker": "2330",
        "signal_date": "2020-01-01",
        "candidate_date": "2020-01-02",
        "trade_date": dates[1],
        "type": "normal",
        "entry_source": "normal",
        "qty": 1000,
        "limit_px": 100.0,
        "init_sl": 90.0,
        "init_trail": 90.0,
        "target_price": 110.0,
        "entry_atr": None,
        "security_profile": None,
        "today_pos": 1,
        "yesterday_pos": 0,
        "sizing_capital": 1_000_000.0,
        "params_obj": params,
        "is_orderable": True,
    }
    snapshot = {
        "ticker": "2330",
        "trade_date": "2020-01-02",
        "candidate_date": "2020-01-02",
        "signal_date": "2020-01-01",
    }

    tracker = CandidateCounterfactualReplay(candidate_cutoff="2020-01-02")
    tracker.begin_replay_day(today=dates[1], all_dfs_fast={"2330": fast}, fallback_params=params)
    tracker.observe_replay_candidates(
        today=dates[1],
        qualified_candidates=[candidate],
        qualified_candidate_snapshots=[snapshot],
        orderable_candidates=[candidate],
        orderable_candidate_snapshots=[snapshot],
        all_dfs_fast={"2330": fast},
        sizing_equity=1_000_000.0,
        fallback_params=params,
    )
    tracker.begin_replay_day(today=dates[2], all_dfs_fast={"2330": fast}, fallback_params=params)
    frame = tracker.signal_frame()
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "candidate_counterfactual_reuses_canonical_entry_and_exit_for_filled_signal",
        (1, True, True, "2020-01-02", "2020-01-03", "STOP", True),
        (
            len(frame), bool(frame.iloc[0]["filled"]), bool(frame.iloc[0]["closed"]),
            frame.iloc[0]["entry_date"], frame.iloc[0]["exit_date"], frame.iloc[0]["exit_type"],
            float(frame.iloc[0]["r_multiple"]) < 0.0,
        ),
    )

    large_params_payload = {"weights": list(range(10000))}
    shadow = {"qty": 1000, "sl": 90.0, "_last_exec_contexts": [{"x": 1}]}
    memory_candidate = dict(candidate)
    memory_candidate["params_obj"] = large_params_payload
    memory_candidate["signal_state"] = {
        "_params_obj": large_params_payload,
        "shadow_position": shadow,
    }
    sidecar = _candidate_execution_replay_snapshot(
        memory_candidate,
        fallback_trade_date=dates[1],
        all_dfs_fast={"2330": fast},
        sizing_equity=1_000_000.0,
    )
    shadow["qty"] = 1
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "candidate_execution_sidecar_preserves_read_only_refs_and_clones_shadow_only",
        (True, True, True, True, True, "2020-01-01"),
        (
            sidecar.get("params_obj") is large_params_payload,
            "signal_state" not in sidecar,
            sidecar.get("_candidate_fast_df") is fast,
            int((sidecar.get("shadow_position_state") or {}).get("qty", 0)) == 1000,
            (sidecar.get("shadow_position_state") or {}).get("_last_exec_contexts") is not shadow.get("_last_exec_contexts"),
            (sidecar.get("_canonical_snapshot") or {}).get("signal_date"),
        ),
    )

    execution_row = _candidate_execution_replay_snapshot(
        candidate,
        fallback_trade_date=dates[1],
        all_dfs_fast={"2330": fast},
        sizing_equity=1_000_000.0,
    )
    canonical_qualified = pd.DataFrame([{
        "ticker": "2330",
        "target_date": "2020-01-01",
        "trade_date": "2020-01-02",
        "candidate_type": "normal",
        "entry_source": "normal",
    }])
    offline = _run_offline_counterfactual(
        execution_rows=[execution_row],
        canonical_qualified=canonical_qualified,
        candidate_cutoff="2020-01-02",
        replay_end="2020-01-03",
    )
    offline_frame = offline.signal_frame()
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "candidate_counterfactual_executes_sidecar_only_after_canonical_replay",
        (True, True, False, 1),
        (
            bool(offline_frame.iloc[0]["filled"]),
            bool(offline_frame.iloc[0]["closed"]),
            isinstance(offline, dict),
            int(offline.offline_management_day_count),
        ),
    )

    add_check(
        results, "synthetic_breakout_quality", case_id,
        "candidate_counterfactual_market_calendar_is_derived_from_sidecar_fast_data",
        tuple(dates),
        tuple(_execution_market_dates([execution_row], start_date="2020-01-01", end_date="2020-01-03")),
    )

    canonical_tracker = CandidateCounterfactualReplay(candidate_cutoff="2020-01-02")
    raw_a = dict(candidate, signal_date="", candidate_date="2020-01-02")
    raw_b = dict(candidate, signal_date="", candidate_date="2020-01-02")
    snapshot_a = dict(snapshot, signal_date="2020-01-01")
    snapshot_b = dict(snapshot, signal_date="2020-01-02")
    canonical_tracker.observe_replay_candidates(
        today=dates[1],
        qualified_candidates=[raw_a, raw_b],
        qualified_candidate_snapshots=[snapshot_a, snapshot_b],
        orderable_candidates=[],
        orderable_candidate_snapshots=[],
        all_dfs_fast={"2330": fast},
        sizing_equity=1_000_000.0,
        fallback_params=params,
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "candidate_counterfactual_uses_canonical_snapshot_keys",
        (("2330", "2020-01-01"), ("2330", "2020-01-02")),
        tuple(sorted(canonical_tracker.states)),
    )

    mismatch_rejected = False
    try:
        canonical_tracker.observe_replay_candidates(
            today=dates[1],
            qualified_candidates=[raw_a, raw_b],
            qualified_candidate_snapshots=[snapshot_a],
            orderable_candidates=[],
            orderable_candidate_snapshots=[],
            all_dfs_fast={"2330": fast},
            sizing_equity=1_000_000.0,
            fallback_params=params,
        )
    except ValueError as exc:
        mismatch_rejected = "canonical snapshot數量不一致" in str(exc)
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "candidate_counterfactual_rejects_snapshot_length_divergence",
        True,
        mismatch_rejected,
    )

    deferred = CandidateCounterfactualReplay(candidate_cutoff="2020-01-02", defer_finalize=True)
    deferred.observe_replay_candidates(
        today=dates[1], qualified_candidates=[candidate], qualified_candidate_snapshots=[snapshot],
        orderable_candidates=[candidate], orderable_candidate_snapshots=[snapshot],
        all_dfs_fast={"2330": fast}, sizing_equity=1_000_000.0, fallback_params=params,
    )
    deferred.finalize_replay(last_date=dates[1], fallback_params=params)
    discovery_frame = deferred.signal_frame()
    deferred.defer_finalize = False
    deferred.finalize_replay(last_date=dates[2], fallback_params=params)
    managed_frame = deferred.signal_frame()
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "candidate_counterfactual_defers_closeout_until_execution_end",
        (True, False, True, "FORCED_CLOSE"),
        (
            bool(discovery_frame.iloc[0]["filled"]),
            bool(discovery_frame.iloc[0]["closed"]),
            bool(managed_frame.iloc[0]["closed"]),
            managed_frame.iloc[0]["exit_type"],
        ),
    )

    high_open_fast = dict(fast)
    high_open_fast.update({
        "Open": np.asarray([110.0, 110.0, 110.0]),
        "High": np.asarray([111.0, 111.0, 111.0]),
        "Low": np.asarray([109.0, 109.0, 109.0]),
        "Close": np.asarray([110.0, 110.0, 110.0]),
    })
    unfilled_candidate = dict(candidate)
    unfilled_row = _candidate_execution_replay_snapshot(
        unfilled_candidate,
        fallback_trade_date=dates[1],
        all_dfs_fast={"2330": high_open_fast},
        sizing_equity=1_000_000.0,
    )
    unfilled = _run_offline_counterfactual(
        execution_rows=[unfilled_row],
        canonical_qualified=canonical_qualified,
        candidate_cutoff="2020-01-02",
        replay_end="2020-01-03",
    ).signal_frame()
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "candidate_counterfactual_keeps_unfilled_signal_nan_not_zero",
        (False, False, True, 1),
        (
            bool(unfilled.iloc[0]["filled"]), bool(unfilled.iloc[0]["closed"]),
            bool(pd.isna(unfilled.iloc[0]["r_multiple"])),
            int(unfilled.iloc[0]["missed_buy_count"]),
        ),
    )

    metric_frame = pd.DataFrame({
        "ticker": ["A", "B", "C"],
        "target_match": [True, True, True],
        "was_orderable": [True, True, False],
        "filled": [True, False, False],
        "closed": [True, False, False],
        "target_raw_r": [2.0, 1.0, 0.0],
        "r_multiple": [1.5, np.nan, np.nan],
        "label": [LABEL_PASS, LABEL_PASS, LABEL_REJECT],
    })
    metrics = counterfactual_metrics(metric_frame)
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "candidate_counterfactual_coverage_denominators_remain_explicit",
        (3, 2, 1, 1/3, 1/2, 1/3),
        (
            metrics["qualified_signal_count"], metrics["orderable_signal_count"], metrics["filled_signal_count"],
            metrics["fill_coverage_vs_qualified"], metrics["fill_coverage_vs_orderable"],
            metrics["strategy_r_coverage_vs_qualified"],
        ),
        tol=1e-12,
    )
    markdown = render_counterfactual_markdown({
        "metrics": metrics,
        "source_11i": {"actual_trade_coverage_vs_qualified": 0.22},
    })
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "candidate_counterfactual_report_states_capacity_and_unfilled_boundary",
        True,
        all(token in markdown for token in (
            "忽略portfolio capacity", "未成交候選維持unlabeled", "不建立Target arrays", "不授權模型",
        )),
    )

    root = Path(__file__).resolve().parents[2]
    app_source = (root / "apps" / "breakout_quality.py").read_text(encoding="utf-8")
    app_tree = ast.parse(app_source)
    command_modules = {}
    for node in app_tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "COMMAND_MODULES" for target in node.targets
        ):
            command_modules = ast.literal_eval(node.value)
            break
    menu_source = app_source[app_source.find("def _run_interactive_menu") : app_source.find("def main")]
    engine_source = (root / "core" / "portfolio_engine.py").read_text(encoding="utf-8")
    runner_source = (root / "tools" / "portfolio_sim" / "simulation_runner.py").read_text(encoding="utf-8")
    compare_source = (
        root / "filters" / "breakout_quality" / "strategy_compare_engine.py"
    ).read_text(encoding="utf-8")
    entry_source = (root / "core" / "portfolio_entries.py").read_text(encoding="utf-8")
    audit_source = _read_audit_source("candidate_counterfactual_execution")
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "candidate_counterfactual_cli_only_and_sidecar_is_not_replay_counts",
        (True, True, True, True, True, True, True, True),
        (
            command_modules.get("audit-candidate-counterfactual")
            == "tools.audit.breakout_quality.candidate_counterfactual_execution",
            "11J" not in menu_source,
            "audit-candidate-counterfactual" not in menu_source,
            'getattr(replay_counts, "begin_replay_day", None)' not in engine_source,
            'getattr(replay_counts, "observe_replay_candidates", None)' not in engine_source,
            "replay_execution_rows=None" in engine_source,
            "replay_execution_rows=replay_execution_rows" in runner_source,
            "replay_execution_rows=replay_execution_rows" in compare_source,
        ),
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "candidate_counterfactual_main_uses_plain_dict_counts_then_sidecar",
        (True, True, True, True, True, True),
        (
            "discovery_counts: dict[str, dict[str, Any]] = {}" in audit_source,
            "execution_rows: list[dict[str, Any]] = []" in audit_source,
            "replay_counts=discovery_counts" in audit_source,
            "replay_execution_rows=execution_rows" in audit_source,
            '"replay_counts_type":"plain_dict"' in audit_source,
            '"lifecycle_callbacks_used":False' in audit_source,
        ),
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "candidate_counterfactual_reuses_entry_shadow_exit_and_accounting_ssot",
        (True, True, True, True, True),
        (
            "build_candidate_plan_seed" in entry_source,
            "execute_pre_market_entry_plan" in audit_source,
            "execute_bar_step" in audit_source,
            "closeout_open_positions" in audit_source,
            "calc_ratio_from_milli" in audit_source,
        ),
    )
    discovery_call = audit_source.find('name="11J_candidate_discovery"')
    source_count_guard = audit_source.find("canonical_discovery_count!=source_qualified_count", discovery_call)
    offline_call = audit_source.find("tracker=_run_offline_counterfactual(", source_count_guard)
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "candidate_counterfactual_checks_2003_before_offline_execution",
        (True, True, True, True),
        (
            discovery_call >= 0,
            source_count_guard > discovery_call,
            offline_call > source_count_guard,
            '"execution_mode":"plain_replay_counts_with_execution_sidecar"' in audit_source,
        ),
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "candidate_counterfactual_is_read_only_and_requires_11i_positive_result",
        (True, True, True, True, True, True),
        (
            '"training_performed":False' in audit_source,
            '"runtime_eligible":False' in audit_source,
            "11J需要11I overall與PASS Target↔R均為正" in audit_source,
            "11J偵測到11I artifact SHA256不一致" in audit_source,
            "copy.deepcopy" not in audit_source,
            "torch.save" not in audit_source,
        ),
    )
    summary["command"] = "audit-candidate-counterfactual"
    summary["audit"] = "canonical_per_candidate_execution"
    return results, summary


def validate_breakout_quality_portfolio_selection_pressure_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_PORTFOLIO_SELECTION_PRESSURE"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from tools.audit.breakout_quality.portfolio_selection_pressure import (
        _build_selection_pressure_tables,
        _render_markdown as render_selection_pressure_markdown,
        parse_args as parse_selection_pressure_args,
    )

    orderable = pd.DataFrame([
        {"ticker": "A", "trade_date": "2020-01-02", "target_raw_r": 3.0, "target_match": True},
        {"ticker": "B", "trade_date": "2020-01-02", "target_raw_r": 2.0, "target_match": True},
        {"ticker": "C", "trade_date": "2020-01-02", "target_raw_r": 1.0, "target_match": True},
        {"ticker": "D", "trade_date": "2020-01-03", "target_raw_r": 4.0, "target_match": True},
        {"ticker": "E", "trade_date": "2020-01-03", "target_raw_r": 1.0, "target_match": True},
    ])
    trades = pd.DataFrame([
        {"ticker": "A", "entry_date": "2020-01-02", "target_raw_r": 3.0, "target_match": True, "r_multiple": 2.0},
        {"ticker": "E", "entry_date": "2020-01-03", "target_raw_r": 1.0, "target_match": True, "r_multiple": -1.0},
    ])
    signals, daily, buckets, metrics = _build_selection_pressure_tables(orderable, trades)
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "portfolio_selection_pressure_same_day_percentile_and_top_k_are_exact",
        (5, 2, 0.75, 0.5, 1.5),
        (
            len(signals), metrics["selected_trade_count"],
            metrics["selected_target_percentile_mean_competition"],
            metrics["top_k_retention_occurrence_weighted"],
            metrics["target_opportunity_gap_r_date_weighted"],
        ),
        tol=1e-12,
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "portfolio_selection_pressure_preserves_realized_r_only_for_selected_rows",
        (2, 3, True),
        (
            int(signals["r_multiple"].notna().sum()),
            int(signals["r_multiple"].isna().sum()),
            bool(signals.loc[~signals["selected"], "r_multiple"].isna().all()),
        ),
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "portfolio_selection_pressure_daily_and_bucket_outputs_are_explicit",
        (2, ["1", "2-3", "4-5", "6-10", "11+"], True),
        (
            len(daily), buckets["pressure_bucket"].tolist(),
            bool({"top_k_retention", "target_opportunity_gap_r"}.issubset(buckets.columns)),
        ),
    )

    duplicate_rejected = False
    try:
        _build_selection_pressure_tables(pd.concat([orderable, orderable.iloc[[0]]], ignore_index=True), trades)
    except ValueError as exc:
        duplicate_rejected = "同ticker／trade_date存在多筆" in str(exc)
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "portfolio_selection_pressure_rejects_duplicate_candidate_day_identity",
        True,
        duplicate_rejected,
    )

    markdown = render_selection_pressure_markdown({"metrics": metrics})
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "portfolio_selection_pressure_report_states_non_counterfactual_boundary",
        True,
        all(token in markdown for token in (
            "不重播市場", "未交易候選沒有realized R", "不填0R", "不授權使用future Target作runtime排序",
        )),
    )

    root = Path(__file__).resolve().parents[2]
    app_source = (root / "apps" / "breakout_quality.py").read_text(encoding="utf-8")
    app_tree = ast.parse(app_source)
    command_modules = {}
    for node in app_tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "COMMAND_MODULES" for target in node.targets
        ):
            command_modules = ast.literal_eval(node.value)
            break
    menu_source = app_source[app_source.find("def _run_interactive_menu") : app_source.find("def main")]
    audit_source = _read_audit_source("portfolio_selection_pressure")
    args = parse_selection_pressure_args([])
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "portfolio_selection_pressure_is_cli_only_read_only_and_uses_11i_artifacts",
        (True, True, True, True, True, True, True, True),
        (
            command_modules.get("audit-selection-pressure")
            == "tools.audit.breakout_quality.portfolio_selection_pressure",
            "11K" not in menu_source,
            "audit-selection-pressure" not in menu_source,
            args.filter_id == BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
            '"strategy_replay_performed": False' in audit_source,
            '"counterfactual_performed": False' in audit_source,
            '"training_performed": False' in audit_source,
            "_artifact_path(source_dir, source, \"orderable\")" in audit_source,
        ),
    )

    summary["command"] = "audit-selection-pressure"
    summary["audit"] = "portfolio_selection_pressure"
    return results, summary


__all__ = [
    "validate_breakout_quality_chronological_embargo_case",
    "validate_breakout_quality_continuous_target_contract_case",
    "validate_breakout_quality_continuous_ranker_contract_case",
    "validate_breakout_quality_pass_conditional_ranker_contract_case",
    "validate_breakout_quality_all_event_no_time_ranker_contract_case",
    "validate_breakout_quality_pass_realization_gap_attribution_contract_case",
    "validate_breakout_quality_qualified_candidate_set_audit_contract_case",
    "validate_breakout_quality_target_component_attribution_contract_case",
    "validate_breakout_quality_target_time_penalty_ablation_contract_case",
    "validate_breakout_quality_no_time_target_selection_audit_contract_case",
    "validate_breakout_quality_policy_single_source_case",
    "validate_breakout_quality_runtime_artifact_contract_case",
    "validate_breakout_quality_selection_strategy_realization_contract_case",
    "validate_breakout_quality_candidate_counterfactual_execution_contract_case",
    "validate_breakout_quality_portfolio_selection_pressure_contract_case",
    "validate_breakout_quality_strategy_comparison_contract_case",
]



def validate_breakout_quality_all_event_no_time_ranker_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_ALL_EVENT_NO_TIME_RANKER"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from tools.filters.breakout_quality.train_continuous_ranker import (
        _profile_contract,
        _scope_group_ids,
        build_daily_percentile_targets,
        parse_args as parse_continuous_ranker_args,
    )

    profile = get_breakout_quality_experiment_profile(
        STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "all_event_no_time_profile_uses_existing_target_and_all_label_scope",
        (
            TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION,
            STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
            TRAINING_LABEL_SCOPE_ALL,
            "mse",
            "mean_daily_spearman",
            False,
        ),
        (
            profile.training_objective,
            profile.continuous_target_id,
            profile.training_label_scope,
            profile.loss_name,
            profile.epoch_selection_metric,
            STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE
            in SUPPORTED_BREAKOUT_QUALITY_CLASSIFICATION_EXPERIMENT_PROFILES,
        ),
    )

    group_table = pd.DataFrame(
        {
            "label": [LABEL_PASS, LABEL_REJECT, LABEL_PASS],
            "date": pd.to_datetime(["2020-01-02"] * 3),
        }
    )
    ids = _scope_group_ids(
        np.arange(3, dtype=np.int64),
        group_table,
        label_scope=TRAINING_LABEL_SCOPE_ALL,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "all_event_scope_keeps_pass_and_reject_groups",
        (0, 1, 2),
        tuple(int(value) for value in ids),
    )

    raw = np.asarray([1.0, 9.0, 5.0], dtype=np.float32)
    pct = build_daily_percentile_targets(
        raw, np.ones(3, dtype=bool), group_table["date"]
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "all_event_daily_percentile_ranks_across_binary_labels",
        (0.0, 1.0, 0.5),
        tuple(float(value) for value in pct),
    )

    args = parse_continuous_ranker_args(
        ["--experiment-profile", STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE]
    )
    contract = _profile_contract(profile)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "mr12a_cli_profile_identity_is_explicit_and_research_only",
        (STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE, "12A", "all_labels"),
        (args.experiment_profile, contract["phase"], contract["metric_scope"]),
    )

    summary["profile"] = STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE
    summary["training_label_scope"] = TRAINING_LABEL_SCOPE_ALL
    return results, summary

def validate_breakout_quality_point_in_time_score_builder_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_BUILDER"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from config import breakout_quality as workflow_config
    from config.breakout_quality import get_breakout_quality_workflow_settings
    from tools.audit.breakout_quality.point_in_time_scores import (
        _direction_summary,
        _orderable_coverage,
        _render_markdown as render_point_in_time_markdown,
        render_compact_console_summary as render_point_in_time_compact_console,
        render_console_summary as render_point_in_time_console,
    )
    from tools.filters.breakout_quality.build_point_in_time_scores import (
        REQUIRED_SCORE_COLUMNS,
        _build_fold_periods,
        _stable_fold_id,
        _combined_validation,
        _fold_group_ids,
        _migrate_compatible_legacy_fold,
        _resolve_score_start,
        _validate_score_frame,
        parse_args as parse_point_in_time_args,
    )
    from tools.filters.breakout_quality.continuous_ranker_pipeline import (
        _validate_group_consistency,
    )
    from filters.breakout_quality.artifacts import build_file_manifest
    from filters.breakout_quality.contract import DEFAULT_MODEL_FILENAME, LABEL_PASS

    consistent_terminal_events = pd.DataFrame(
        [
            {
                "ticker": "2330",
                "date": "2026-03-02",
                "group_index": 0,
                "label_eval_end_date": None,
            },
            {
                "ticker": "2330",
                "date": "2026-03-02",
                "group_index": 0,
                "label_eval_end_date": None,
            },
            {
                "ticker": "2317",
                "date": "2020-01-02",
                "group_index": 1,
                "label_eval_end_date": "2020-03-02",
            },
        ]
    )
    all_missing_label_end_accepted = True
    try:
        _validate_group_consistency(consistent_terminal_events)
    except ValueError:
        all_missing_label_end_accepted = False
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "point_in_time_group_consistency_accepts_all_missing_terminal_label_end",
        True,
        all_missing_label_end_accepted,
    )

    mixed_label_end_rejected = False
    inconsistent_terminal_events = consistent_terminal_events.copy()
    inconsistent_terminal_events.loc[1, "label_eval_end_date"] = "2026-04-30"
    try:
        _validate_group_consistency(inconsistent_terminal_events)
    except ValueError as exc:
        mixed_label_end_rejected = "invalid_groups=1" in str(exc)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "point_in_time_group_consistency_rejects_mixed_missing_and_completed_label_end",
        True,
        mixed_label_end_rejected,
    )

    settings = get_breakout_quality_workflow_settings()
    with patch.object(
        workflow_config,
        "BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE",
        UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE,
    ):
        binary_settings = workflow_config.get_breakout_quality_workflow_settings()
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "workflow_binary_profile_resolves_classification_and_hard_filter",
        (
            "binary_classification",
            None,
            "hard-filter",
            "canonical_runtime",
            "original",
        ),
        (
            binary_settings.training_objective,
            binary_settings.continuous_target_id,
            binary_settings.strategy_comparison_mode,
            binary_settings.strategy_score_source,
            binary_settings.strategy_buy_sort,
        ),
    )
    with patch.object(
        workflow_config,
        "BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE",
        STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE,
    ):
        continuous_settings = workflow_config.get_breakout_quality_workflow_settings()
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "workflow_continuous_profile_resolves_point_in_time_score_ranking",
        (
            "daily_percentile_regression",
            "strategy_aligned_opportunity_no_time_r_v1",
            "score-ranking",
            "selection_point_in_time",
            "breakout_quality_score_desc",
        ),
        (
            continuous_settings.training_objective,
            continuous_settings.continuous_target_id,
            continuous_settings.strategy_comparison_mode,
            continuous_settings.strategy_score_source,
            continuous_settings.strategy_buy_sort,
        ),
    )
    parsed = parse_point_in_time_args([])
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "point_in_time_cli_defaults_follow_current_workflow_config",
        (
            settings.filter_id,
            settings.model_architecture,
            settings.experiment_profile,
            settings.seed,
            settings.point_in_time_score_start_date,
            settings.point_in_time_fold_months,
            settings.point_in_time_inner_validation_months,
        ),
        (
            parsed.filter_id,
            parsed.model_architecture,
            parsed.experiment_profile,
            parsed.seed,
            parsed.score_start_date,
            parsed.fold_months,
            parsed.inner_validation_months,
        ),
    )

    periods = _build_fold_periods(
        pd.Timestamp("2019-12-31"),
        pd.Timestamp("2020-03-15"),
        fold_months=1,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "point_in_time_fold_periods_are_contiguous_and_calendar_month_based",
        [
            ("2019-12-31", "2019-12-31"),
            ("2020-01-01", "2020-01-31"),
            ("2020-02-01", "2020-02-29"),
            ("2020-03-01", "2020-03-15"),
        ],
        [
            (str(item["score_start"].date()), str(item["score_end"].date()))
            for item in periods
        ],
    )

    original_annual_periods = _build_fold_periods(
        pd.Timestamp("2014-01-01"),
        pd.Timestamp("2015-12-31"),
        fold_months=12,
    )
    earlier_periods = _build_fold_periods(
        pd.Timestamp("2007-06-01"),
        pd.Timestamp("2015-12-31"),
        fold_months=12,
    )
    original_fold = next(
        item for item in original_annual_periods
        if item["score_start"] == pd.Timestamp("2014-01-01")
    )
    extended_fold = next(
        item for item in earlier_periods
        if item["score_start"] == pd.Timestamp("2014-01-01")
    )
    auto_group_dates = pd.date_range("2020-01-01", periods=8, freq="MS")
    auto_bundle = SimpleNamespace(
        group_table=pd.DataFrame({
            "date": auto_group_dates,
            "label_eval_end_date": auto_group_dates,
            "label": np.full(len(auto_group_dates), LABEL_PASS, dtype=np.int64),
        }),
        target_valid=np.ones(len(auto_group_dates), dtype=bool),
        profile=SimpleNamespace(training_label_scope=TRAINING_LABEL_SCOPE_PASS_ONLY),
    )
    auto_settings = SimpleNamespace(
        point_in_time_min_train_groups=2,
        point_in_time_min_validation_groups=2,
        point_in_time_min_score_groups=1,
    )
    resolved_auto_start, auto_diagnostics = _resolve_score_start(
        "auto",
        bundle=auto_bundle,
        settings=auto_settings,
        selection_start=pd.Timestamp("2020-01-01"),
        score_end=pd.Timestamp("2020-08-31"),
        fold_months=1,
        validation_months=2,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "point_in_time_auto_start_selects_earliest_month_meeting_real_split_counts",
        ("2020-05-01", 5, {"inner_train": 2, "validation": 2, "score": 1}),
        (
            str(resolved_auto_start.date()),
            auto_diagnostics["candidate_months_checked"],
            auto_diagnostics["first_fold_group_counts"],
        ),
    )

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "point_in_time_fold_identity_is_date_stable_when_history_is_extended",
        (
            "fold_20140101_20141231",
            "fold_20140101_20141231",
            "fold_20140101_20141231",
        ),
        (
            original_fold["fold_id"],
            extended_fold["fold_id"],
            _stable_fold_id(pd.Timestamp("2014-01-01"), pd.Timestamp("2014-12-31")),
        ),
    )

    with tempfile.TemporaryDirectory() as tmp:
        pit_root = Path(tmp) / "point_in_time"
        legacy_dir = pit_root / "folds" / "fold_000"
        target_dir = pit_root / "folds" / "fold_20140101_20141231"
        legacy_dir.mkdir(parents=True, exist_ok=True)
        model_path = legacy_dir / DEFAULT_MODEL_FILENAME
        score_path = legacy_dir / "scores.csv"
        pd.DataFrame(
            [{
                "ticker": "2330",
                "date": "2014-06-30",
                "group_index": 7,
                "breakout_quality_score": 0.75,
                "fold_id": "fold_000",
                "model_information_cutoff": "2013-12-31",
            }]
        ).to_csv(score_path, index=False, encoding="utf-8-sig")
        common_contract = {
            "filter_id": "breakout_quality_v1",
            "model_architecture": "inception_time_v1",
            "experiment_profile": "strategy_aligned_no_time_pass_magnitude_mse",
            "continuous_target_id": "strategy_aligned_opportunity_no_time_r_v1",
            "training_label_scope": "pass_only",
            "seed": 42,
            "planned_periods": {
                "score_start": "2014-01-01",
                "score_end": "2014-12-31",
            },
            "observed_periods": {},
            "model_information_cutoff": "2013-12-31",
            "group_counts": {},
            "event_row_counts": {},
            "model_spec": {},
            "experiment_settings": {},
            "training_settings": {},
            "source_contract": {},
            "lookahead_contract": {},
        }
        import torch

        torch.save(
            {
                "model_state_dict": {"synthetic_weight": torch.tensor([1.0])},
                "fold_contract": {
                    **common_contract,
                    "schema_version": 1,
                    "fold_id": "fold_000",
                },
            },
            model_path,
        )
        legacy_manifest = {
            **common_contract,
            "schema_version": 1,
            "fold_id": "fold_000",
            "contract_fingerprint": "legacy-fingerprint",
            "artifacts": {
                "checkpoint": build_file_manifest(model_path),
                "scores": build_file_manifest(score_path),
            },
        }
        (legacy_dir / "manifest.json").write_text(
            json.dumps(legacy_manifest), encoding="utf-8"
        )
        target_contract = {
            **common_contract,
            "schema_version": 2,
            "fold_id": "fold_20140101_20141231",
        }
        migrated = _migrate_compatible_legacy_fold(
            point_in_time_dir=pit_root,
            target_fold_dir=target_dir,
            expected_fingerprint="stable-fingerprint",
            fold_contract=target_contract,
            torch_module=torch,
        )
        migrated_frame, migrated_manifest = migrated or (pd.DataFrame(), {})
        migrated_checkpoint = (
            torch.load(
                target_dir / DEFAULT_MODEL_FILENAME,
                map_location="cpu",
                weights_only=True,
            )
            if migrated is not None
            else {}
        )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "compatible_legacy_fold_is_hash_checked_and_migrated_to_stable_id",
        (
            True,
            {"fold_20140101_20141231"},
            "fold_20140101_20141231",
            "fold_000",
            "stable-fingerprint",
        ),
        (
            migrated is not None,
            set(migrated_frame.get("fold_id", pd.Series(dtype=str)).astype(str)),
            dict(migrated_checkpoint.get("fold_contract") or {}).get("fold_id"),
            dict(migrated_manifest.get("migration") or {}).get("source_fold_id"),
            migrated_manifest.get("contract_fingerprint"),
        ),
    )

    group_table = pd.DataFrame(
        [
            {
                "ticker": "A",
                "date": "2010-01-01",
                "group_index": 0,
                "label": 1,
                "label_eval_end_date": "2010-02-01",
            },
            {
                "ticker": "B",
                "date": "2011-12-01",
                "group_index": 1,
                "label": 1,
                "label_eval_end_date": "2012-01-10",
            },
            {
                "ticker": "C",
                "date": "2012-01-02",
                "group_index": 2,
                "label": 1,
                "label_eval_end_date": "2012-02-15",
            },
            {
                "ticker": "D",
                "date": "2013-12-01",
                "group_index": 3,
                "label": 1,
                "label_eval_end_date": "2013-12-20",
            },
            {
                "ticker": "E",
                "date": "2013-12-20",
                "group_index": 4,
                "label": 1,
                "label_eval_end_date": "2014-01-10",
            },
            {
                "ticker": "0056",
                "date": "2014-01-02",
                "group_index": 5,
                "label": 0,
                "label_eval_end_date": "2014-02-15",
            },
            {
                "ticker": "G",
                "date": "2014-06-01",
                "group_index": 6,
                "label": 1,
                "label_eval_end_date": "2014-07-15",
            },
        ]
    )
    bundle = SimpleNamespace(
        group_table=group_table,
        profile=SimpleNamespace(training_label_scope="pass_only"),
        target_valid=np.ones(len(group_table), dtype=bool),
        event_group_index=np.arange(len(group_table), dtype=np.int64),
    )
    fold = {
        "fold_id": "fold_000",
        "score_start": pd.Timestamp("2014-01-01"),
        "score_end": pd.Timestamp("2014-12-31"),
    }
    ids = _fold_group_ids(bundle, fold, validation_months=24)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "point_in_time_split_requires_completed_labels_before_each_information_boundary",
        ([0], [2, 3], [0, 1, 2, 3], [5, 6]),
        (
            ids["train_ids"].tolist(),
            ids["validation_ids"].tolist(),
            ids["final_ids"].tolist(),
            ids["score_ids"].tolist(),
        ),
    )

    fold_contract = {
        "fold_id": "fold_000",
        "model_information_cutoff": "2013-12-20",
        "planned_periods": {
            "score_start": "2014-01-01",
            "score_end": "2014-12-31",
        },
    }
    score_frame = pd.DataFrame(
        [
            {
                "ticker": "0056",
                "date": "2014-01-02",
                "group_index": 5,
                "breakout_quality_score": 0.2,
                "fold_id": "fold_000",
                "model_information_cutoff": "2013-12-20",
            },
            {
                "ticker": "G",
                "date": "2014-06-01",
                "group_index": 6,
                "breakout_quality_score": 0.8,
                "fold_id": "fold_000",
                "model_information_cutoff": "2013-12-20",
            },
        ]
    )
    validated = _validate_score_frame(score_frame, fold_contract=fold_contract)
    coverage = _combined_validation(
        validated,
        bundle,
        score_start=pd.Timestamp("2014-01-01"),
        score_end=pd.Timestamp("2014-12-31"),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "point_in_time_score_output_has_no_future_target_and_complete_unique_coverage",
        (True, 2, 1.0, 0, 0, 0),
        (
            not bool(
                {"label", "target_raw_r", "target_daily_percentile"}
                & set(REQUIRED_SCORE_COLUMNS)
            ),
            coverage["scored_group_count"],
            coverage["coverage_rate"],
            coverage["duplicate_group_count"],
            coverage["missing_group_count"],
            coverage["extra_group_count"],
        ),
    )

    duplicate_rejected = False
    try:
        _combined_validation(
            pd.concat([validated, validated.iloc[[0]]], ignore_index=True),
            bundle,
            score_start=pd.Timestamp("2014-01-01"),
            score_end=pd.Timestamp("2014-12-31"),
        )
    except ValueError as exc:
        duplicate_rejected = "重複group" in str(exc)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "point_in_time_combined_output_rejects_duplicate_groups",
        True,
        duplicate_rejected,
    )

    identity_mismatch_rejected = False
    mismatched = validated.copy()
    mismatched.loc[mismatched["group_index"] == 5, "ticker"] = "9999"
    try:
        _combined_validation(
            mismatched,
            bundle,
            score_start=pd.Timestamp("2014-01-01"),
            score_end=pd.Timestamp("2014-12-31"),
        )
    except ValueError as exc:
        identity_mismatch_rejected = "identity不一致" in str(exc)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "point_in_time_combined_output_rejects_ticker_date_identity_mismatch",
        True,
        identity_mismatch_rejected,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        orderable_path = Path(tmp_dir) / "orderable.csv"
        pd.DataFrame(
            [
                {
                    "ticker": "0056",
                    "target_date": "2014-01-02",
                    "breakout_quality_score": 0.01,
                },
                {
                    "ticker": "X",
                    "target_date": "2014-01-03",
                    "breakout_quality_score": 0.99,
                },
            ]
        ).to_csv(orderable_path, index=False, encoding="utf-8-sig")
        score_dates = validated.copy()
        score_dates["date"] = pd.to_datetime(score_dates["date"], errors="raise")
        orderable = _orderable_coverage(
            score_dates,
            requested_path=str(orderable_path),
            filter_id=settings.filter_id,
            target_id=settings.continuous_target_id,
        )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "point_in_time_orderable_coverage_uses_canonical_target_date",
        (True, 2, 1, 0.5),
        (
            orderable["available"],
            orderable["candidate_count"],
            orderable["scored_candidate_count"],
            orderable["coverage_rate"],
        ),
        tol=1e-12,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "point_in_time_orderable_coverage_ignores_existing_candidate_score_column",
        (True, "selection_point_in_time_scores"),
        (
            orderable["candidate_artifact_has_existing_breakout_quality_score"],
            orderable["coverage_score_source"],
        ),
    )

    yearly_rows = [
        {
            "year": 2014,
            "group_count": 2,
            "global_spearman": 0.3,
            "mean_daily_spearman": 0.2,
            "top_bottom_target_spread": 0.8,
        }
    ]
    report_payload = {
        "status": "RESULT_AVAILABLE_PENDING_REVIEW",
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "strategy_aligned_no_time_pass_magnitude_mse",
        "continuous_target_id": "strategy_aligned_opportunity_no_time_r_v1",
        "score_period": {"start": "2014-01-01", "end": "2014-12-31"},
        "score_coverage": {
            "expected_group_count": 2,
            "scored_group_count": 2,
            "coverage_rate": 1.0,
        },
        "workflow": {
            "training_label_scope": "pass_only",
            "seed": 42,
            "fold_count": 1,
            "fold_months": 12,
            "inner_validation_months": 24,
        },
        "metrics": {
            "pass_only_target": {
                "group_count": 2,
                "global_spearman": 0.3,
                "mean_daily_spearman": 0.2,
                "top_decile_target_mean": 1.5,
                "bottom_decile_target_mean": 0.7,
                "top_bottom_target_spread": 0.8,
            },
            "all_valid_target": {
                "group_count": 2,
                "global_spearman": 0.25,
                "mean_daily_spearman": 0.15,
                "top_decile_target_mean": 1.4,
                "bottom_decile_target_mean": 0.6,
                "top_bottom_target_spread": 0.8,
            },
        },
        "yearly_pass_only": yearly_rows,
        "direction_summary": _direction_summary(yearly_rows),
        "fold_metrics": [
            {
                "fold_id": "fold_000",
                "group_count": 2,
                "score_mean": 0.5,
                "score_std": 0.2,
                "score_p10": 0.3,
                "score_p50": 0.5,
                "score_p90": 0.7,
                "adjacent_mean_shift_in_pooled_std": None,
                "pass_target_spearman": 0.3,
            }
        ],
        "fold_drift": {
            "criterion": "synthetic drift contract",
            "drift_flag": False,
            "flagged_folds": [],
            "max_adjacent_mean_shift_in_pooled_std": 0.0,
        },
        "classification_overlap": {
            "score_vs_pass_reject_auc": 0.6,
            "overall_pass_share": 0.5,
            "top_score_decile_pass_share": 1.0,
            "interpretation_contract": "synthetic overlap contract",
        },
        "orderable_candidate_coverage": {
            "available": True,
            "path": "/tmp/orderable.csv",
            "candidate_count": 2,
            "scored_candidate_count": 2,
            "unscored_candidate_count": 0,
            "coverage_rate": 1.0,
            "coverage_score_source": "selection_point_in_time_scores",
        },
        "source_artifacts": {
            "point_in_time_manifest": "/tmp/manifest.json",
            "point_in_time_scores": "/tmp/scores.csv",
            "point_in_time_coverage": "/tmp/coverage.csv",
        },
        "report_artifacts": {
            "markdown": "/tmp/audit.md",
            "json": "/tmp/audit.json",
        },
    }
    report_console = render_point_in_time_console(report_payload)
    colored_report_console = render_point_in_time_console(report_payload, color=True)
    compact_report_console = render_point_in_time_compact_console(report_payload)
    report_markdown = render_point_in_time_markdown(report_payload)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "point_in_time_audit_outputs_readable_console_summary",
        True,
        all(
            text in report_console
            for text in (
                "Selection Point-in-time 模型評估報表",
                "核心排序能力",
                "年度穩定性",
                "Fold 分布與漂移",
                "策略 optimizer：未執行",
            )
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "point_in_time_audit_console_uses_shared_status_colors",
        True,
        all(
            token in colored_report_console
            for token in (
                "\x1b[96m",
                "\x1b[92m",
                "\x1b[93m",
            )
        )
        and "\x1b[" not in report_console,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "point_in_time_audit_compact_console_avoids_repeated_detail_tables",
        True,
        all(
            text in compact_report_console
            for text in (
                "PIT 模型驗證",
                "核心排序",
                "年度穩定",
                "模型 Gate",
                "執行策略績效驗證",
            )
        )
        and all(
            text not in compact_report_console
            for text in (
                "Current Breakout Quality Workflow",
                "Fold 分布與漂移",
                "fold_000",
                "完整指標 JSON",
            )
        ),
    )
    project_root = Path(__file__).resolve().parents[2]
    ranker_source = (
        project_root / "tools" / "filters" / "breakout_quality" / "train_continuous_ranker.py"
    ).read_text(encoding="utf-8")
    pit_builder_source = (
        project_root / "tools" / "filters" / "breakout_quality" / "build_point_in_time_scores.py"
    ).read_text(encoding="utf-8")
    pit_builder_tree = ast.parse(pit_builder_source)
    pit_builder_call_keywords = {
        node.func.id: {keyword.arg for keyword in node.keywords if keyword.arg}
        for node in ast.walk(pit_builder_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"_load_reusable_fold", "_migrate_compatible_legacy_fold"}
    }
    base_target_audit_source = _read_audit_source("continuous_target")
    no_time_target_source = _read_audit_source("no_time_continuous_target")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "point_in_time_resume_wires_torch_only_to_legacy_migration",
        (False, True),
        (
            "torch_module" in pit_builder_call_keywords.get("_load_reusable_fold", set()),
            "torch_module" in pit_builder_call_keywords.get(
                "_migrate_compatible_legacy_fold", set()
            ),
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "point_in_time_compact_training_output_is_one_line_per_new_fold",
        True,
        ranker_source.count("if not compact_console:") >= 4
        and "fold_progress.print_line(" in pit_builder_source
        and "best epoch=" in pit_builder_source
        and "PIT Scores 完成" in pit_builder_source,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "point_in_time_compact_target_output_hides_base_audit_and_summarizes_final_target",
        True,
        "if compact_console_enabled():" in base_target_audit_source
        and "Continuous Target 完成" in no_time_target_source
        and "Selection mean=" in no_time_target_source
        and "OOS未評估" in no_time_target_source,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "point_in_time_audit_outputs_complete_markdown_report",
        True,
        all(
            text in report_markdown
            for text in (
                "# Breakout Quality Selection Point-in-time 模型評估報表",
                "## 2. 核心排序能力",
                "## 3. 年度穩定性（PASS-only）",
                "## 4. Fold 分布與漂移",
                "## 7. 研究邊界與下一步",
                "## 8. 工件",
            )
        ),
    )

    summary["workflow"] = "selection_point_in_time_scores"
    summary["score_contract"] = "future_target_excluded"
    return results, summary


def validate_breakout_quality_selection_point_in_time_score_sort_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_SELECTION_POINT_IN_TIME_SCORE_SORT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from core.buy_sort import (
        BUY_LIMIT_OVERAGE_SORT_METHOD,
        sort_candidate_rows,
    )
    from core.extended_signals import (
        attach_breakout_quality_rank,
        resolve_breakout_quality_rank,
    )
    from core.portfolio_candidates import _make_candidate_row
    from filters.breakout_quality.ranking_score_store import (
        _validate_audit_source_artifact,
        derive_point_in_time_model_validation_gate,
    )
    from filters.breakout_quality.runtime import (
        breakout_quality_ranking_source_context,
        get_breakout_quality_ranking_source_context,
    )
    from filters.breakout_quality.strategy_compare_engine import (
        _strategy_selection_diagnostics,
    )

    ranking_rows = [
        {"ticker": "B", "use_breakout_quality_ranking": True,
         "breakout_quality_score": None, "breakout_quality_rank": {"available": False},
         "sort_value": 0.10, "proj_cost": 100.0},
        {"ticker": "C", "use_breakout_quality_ranking": True,
         "breakout_quality_score": 0.80, "breakout_quality_rank": {"available": True},
         "sort_value": 0.50, "proj_cost": 100.0},
        {"ticker": "A", "use_breakout_quality_ranking": True,
         "breakout_quality_score": 0.80, "breakout_quality_rank": {"available": True},
         "sort_value": 0.20, "proj_cost": 100.0},
        {"ticker": "D", "use_breakout_quality_ranking": True,
         "breakout_quality_score": None, "breakout_quality_rank": {"available": False},
         "sort_value": 0.05, "proj_cost": 100.0},
    ]
    sort_candidate_rows(ranking_rows, method=BUY_LIMIT_OVERAGE_SORT_METHOD)
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "point_in_time_score_sort_desc_tie_and_missing_fallback_contract",
        ["A", "C", "D", "B"], [row["ticker"] for row in ranking_rows],
    )

    unavailable_rank = {
        "score": None,
        "available": False,
        "unavailable_reason": "missing_ticker_date_score",
        "score_date": "2014-01-01",
        "score_source": "selection_point_in_time",
        "shared_group_score": True,
        "filter_id": BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
        "model_architecture": "inception_time_v1",
        "experiment_profile": "strategy_aligned_no_time_pass_magnitude_mse",
    }
    signal_state = {}
    attach_breakout_quality_rank(signal_state, unavailable_rank)
    inherited_unavailable_rank = resolve_breakout_quality_rank(signal_state)
    ranking_params = replace(
        _base_params,
        use_breakout_quality_ranking=True,
        breakout_quality_filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    )
    with breakout_quality_ranking_source_context(
        score_source="selection_point_in_time",
        model_architecture="inception_time_v1",
        experiment_profile="strategy_aligned_no_time_pass_magnitude_mse",
    ):
        with patch(
            "core.portfolio_candidates.resolve_breakout_quality_candidate_rank",
            side_effect=AssertionError("延續候選不得重新查詢原事件Score"),
        ):
            resolved_inherited_unavailable_rank = _resolve_candidate_quality_ranking(
                params=ranking_params,
                ticker="0056",
                signal_date=pd.Timestamp("2014-01-01"),
                signal_state=signal_state,
            )
    unavailable_candidate = _make_candidate_row(
        buy_sort_method=BUY_LIMIT_OVERAGE_SORT_METHOD,
        ticker="0056",
        candidate_type="normal",
        est_limit_px=100.0,
        ev=0.0,
        y_atr=2.0,
        t_pos=1,
        y_pos=0,
        est_qty=0,
        win_rate=0.0,
        trade_count=0,
        asset_growth_pct=0.0,
        est_init_sl=90.0,
        est_init_trail=95.0,
        est_target_price=120.0,
        entry_atr=2.0,
        is_orderable=True,
        params=_base_params,
        trade_date=pd.Timestamp("2014-01-02"),
        signal_date=pd.Timestamp("2014-01-01"),
        prev_close=99.0,
        quality_rank=unavailable_rank,
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "point_in_time_missing_score_is_preserved_for_continuation_and_candidate_fallback",
        (
            False,
            None,
            "missing_ticker_date_score",
            "selection_point_in_time",
            "selection_point_in_time",
            False,
            None,
        ),
        (
            inherited_unavailable_rank["available"],
            inherited_unavailable_rank["score"],
            inherited_unavailable_rank["unavailable_reason"],
            inherited_unavailable_rank["score_source"],
            resolved_inherited_unavailable_rank["score_source"],
            unavailable_candidate["breakout_quality_rank"]["available"],
            unavailable_candidate["breakout_quality_score"],
        ),
    )

    default_source = get_breakout_quality_ranking_source_context().score_source
    with breakout_quality_ranking_source_context(
        score_source="selection_point_in_time",
        model_architecture="inception_time_v1",
        experiment_profile="strategy_aligned_no_time_pass_magnitude_mse",
    ):
        inside_source = get_breakout_quality_ranking_source_context().score_source
    restored_source = get_breakout_quality_ranking_source_context().score_source
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "point_in_time_ranking_source_context_is_scoped_and_restored",
        ("canonical_runtime", "selection_point_in_time", "canonical_runtime"),
        (default_source, inside_source, restored_source),
    )

    with tempfile.TemporaryDirectory() as artifact_dir:
        source_path = Path(artifact_dir) / "selection_point_in_time_scores.csv"
        source_path.write_text("ticker,date,score\n2330,2020-01-01,0.8\n", encoding="utf-8")
        source_record = {"path": str(source_path), **build_file_manifest(source_path)}
        exact_binding_accepted = True
        try:
            _validate_audit_source_artifact(
                source_record, expected_path=source_path, label="synthetic PIT Scores"
            )
        except ValueError:
            exact_binding_accepted = False
        source_path.write_text("ticker,date,score\n2330,2020-01-01,0.7\n", encoding="utf-8")
        stale_binding_rejected = False
        try:
            _validate_audit_source_artifact(
                source_record, expected_path=source_path, label="synthetic PIT Scores"
            )
        except ValueError as exc:
            stale_binding_rejected = "SHA256" in str(exc)
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "point_in_time_audit_is_bound_to_exact_source_artifact_hashes",
        (True, True), (exact_binding_accepted, stale_binding_rejected),
    )

    gate = derive_point_in_time_model_validation_gate({
        "metrics": {"pass_only_target": {
            "global_spearman": 0.3074, "mean_daily_spearman": 0.2370,
        }},
        "direction_summary": {
            "valid_year_count": 7,
            "positive_spearman_year_count": 7,
            "positive_spread_year_count": 7,
        },
    })
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "point_in_time_model_gate_uses_only_target_ordering_evidence",
        ("PASS", False, False),
        (gate["status"], gate["strategy_metrics_used"], gate["future_target_used_for_runtime_sort"]),
    )

    diagnostic_orderable_input = pd.DataFrame([
        {"ticker": "A", "trade_date": "2020-01-03",
         "signal_date": "2020-01-02", "breakout_quality_score_date": "2020-01-01",
         "breakout_quality_score": 0.8, "breakout_quality_score_available": True},
        {"ticker": "B", "trade_date": "2020-01-03",
         "signal_date": "2020-01-01", "breakout_quality_score_date": "2020-01-01",
         "breakout_quality_score": 0.2, "breakout_quality_score_available": True},
    ])
    diagnostic_selected_input = pd.DataFrame([
        {"ticker": "A", "trade_date": "2020-01-03",
         "signal_date": "2020-01-02", "type": "買進"},
    ])
    diagnostic_lookup_input = pd.DataFrame([
        {"ticker": "A", "signal_date": "2020-01-01",
         "breakout_quality_score": 0.8, "target_raw_r": 2.0,
         "target_available": True},
        {"ticker": "B", "signal_date": "2020-01-01",
         "breakout_quality_score": 0.2, "target_raw_r": 1.0,
         "target_available": True},
    ])
    diagnostic_metrics, diagnostic_orderable, diagnostic_selected = (
        _strategy_selection_diagnostics(
            orderable=diagnostic_orderable_input,
            selected=diagnostic_selected_input,
            lookup=diagnostic_lookup_input,
        )
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "point_in_time_strategy_diagnostic_uses_original_score_date_after_replay",
        (1.0, 1.0, 0.0, False, "2020-01-01", "2020-01-01"),
        (
            diagnostic_metrics["orderable_score_coverage_rate"],
            diagnostic_metrics["target_top_k_retention_mean"],
            diagnostic_metrics["target_opportunity_gap_r_mean"],
            diagnostic_metrics["future_target_used_for_runtime_sort"],
            diagnostic_orderable.loc[0, "score_event_date"],
            diagnostic_selected.loc[0, "score_event_date"],
        ),
    )

    true_score_mismatch_rejected = False
    mismatch_message_has_identity = False
    mismatched_orderable = diagnostic_orderable_input.copy()
    mismatched_orderable.loc[0, "breakout_quality_score"] = 0.7
    try:
        _strategy_selection_diagnostics(
            orderable=mismatched_orderable,
            selected=diagnostic_selected_input,
            lookup=diagnostic_lookup_input,
        )
    except ValueError as exc:
        true_score_mismatch_rejected = True
        mismatch_message_has_identity = all(
            text in str(exc)
            for text in (
                "ticker=A",
                "signal_date=2020-01-02",
                "score_event_date=2020-01-01",
                "runtime_score=0.7",
                "pit_score=0.8",
            )
        )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "point_in_time_strategy_diagnostic_rejects_true_score_mismatch_with_identity",
        (True, True),
        (true_score_mismatch_rejected, mismatch_message_has_identity),
    )

    summary["workflow"] = "selection_point_in_time_scores"
    summary["score_contract"] = "future_target_excluded"
    summary["strategy_score_sort"] = "missing_fallback_original_buy_sort"
    return results, summary


def validate_breakout_quality_score_ranking_capture_audit_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_SCORE_RANKING_CAPTURE_AUDIT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from core.console_report import (
        project_relative_display_path,
        strip_ansi,
    )
    from tools.audit.portfolio.score_ranking_capture import (
        _aggregate_capture_ratio,
        build_score_ranking_capture_audit,
        render_capture_audit_console,
        write_score_ranking_capture_audit_outputs,
    )
    from filters.breakout_quality.strategy_compare_engine import (
        COMPARISON_MODE_SCORE_RANKING,
        _markdown_report,
        _render_strategy_console_report,
    )

    baseline_history = pd.DataFrame([
        {
            "Date": "2014-01-02", "Ticker": "2330",
            "Type": "買進 (新訊號, EV:1.00R)", "買訊日": "2014-01-01",
            "候選類型": "新訊號", "進場類型": "normal", "成交價": 100.0,
            "停損價": 90.0, "股數": 1000, "預留總金額": 101000.0,
            "投入總金額": 100500.0, "Quality Score": 0.6,
            "Quality Score Date": "2014-01-01",
        },
        {"Date": "2014-01-10", "Ticker": "2330", "Type": "半倉停利", "成交價": 115.0, "股數": 500},
        {"Date": "2014-01-20", "Ticker": "2330", "Type": "全倉結算(指標)", "該筆總損益": 12000.0, "R_Multiple": 1.2},
        {"Date": "2014-02-01", "Ticker": "1101", "Type": "錯失買進(新訊號)", "預留總金額": 50000.0},
    ])
    score_history = pd.DataFrame([
        {
            "Date": "2014-01-02", "Ticker": "2330",
            "Type": "買進 (新訊號, EV:1.00R)", "買訊日": "2014-01-01",
            "候選類型": "新訊號", "進場類型": "normal", "成交價": 100.0,
            "停損價": 88.0, "股數": 800, "預留總金額": 100000.0,
            "投入總金額": 80500.0, "Quality Score": 0.8,
            "Quality Score Date": "2014-01-01",
        },
        {"Date": "2014-01-25", "Ticker": "2330", "Type": "全倉結算(停損)", "該筆總損益": -7000.0, "R_Multiple": -0.7},
        {"Date": "2014-02-01", "Ticker": "1101", "Type": "錯失買進(新訊號)", "預留總金額": 50000.0},
    ])
    selected = pd.DataFrame([
        {
            "ticker": "2330", "trade_date": "2014-01-02",
            "signal_date": "2014-01-01", "score_event_date": "2014-01-01",
            "target_raw_r": 2.0,
        }
    ])
    baseline_capacity = pd.DataFrame({
        "Date": [
            "2014-01-02", "2014-01-03", "2014-01-06", "2014-01-07",
            "2014-01-08", "2014-01-09", "2014-01-10", "2014-01-13",
            "2014-01-14", "2014-01-15", "2014-01-16", "2014-01-17",
            "2014-01-20", "2014-01-25", "2014-02-01",
        ]
    })
    score_capacity = baseline_capacity.copy()
    metadata = {
        "comparison_mode": COMPARISON_MODE_SCORE_RANKING,
        "comparison_period": {"start": "2014-01-01", "end": "2014-12-31"},
        "score_source": "selection_point_in_time",
        "params_path": "synthetic.json", "param_source_kind": "rolling_active_param_ensemble",
        "param_selector": "base_finalist_best", "runtime_member_count_min": 1,
        "runtime_member_count_max": 1, "runtime_min_agree": 1,
        "comparison_design": "selection_point_in_time_active_param_replay",
        "lookahead_safe_active_param_schedule": True, "dataset": "full",
        "benchmark_ticker": "0050", "filter_id": "synthetic",
        "model_architecture": "inception_time_v1", "experiment_profile": "synthetic",
        "score_ranking_order": ["breakout_quality_score_desc", "existing_buy_sort", "ticker_deterministic"],
        "ranking_scope": "all_candidates_after_single_member_qualification",
    }
    baseline_summary = {
        "total_return_pct": 20.0, "max_drawdown_pct": 10.0,
        "return_over_max_drawdown": 2.0, "annual_return_pct": 10.0,
        "log_r_squared": 0.9, "monthly_win_rate_pct": 60.0, "trade_count": 1,
        "win_rate_pct": 100.0, "payoff_ratio": 2.0, "expected_value_r": 1.2,
        "avg_exposure_pct": 77.0, "min_full_year_return_pct": 20.0,
        "avg_orderable_candidates": 10.0, "candidate_supply_gap_days": 1,
        "underfilled_end_days": 1, "end_position_gap_slot_days": 1,
    }
    score_summary = {
        **baseline_summary,
        "total_return_pct": 10.0, "max_drawdown_pct": 20.0,
        "return_over_max_drawdown": 0.5, "avg_exposure_pct": 55.0,
        "expected_value_r": -0.7, "win_rate_pct": 0.0,
    }
    selection_diagnostics = {
        "no_filter": {}, "score_ranking": {},
        "score_ranking_minus_no_filter": {
            "selected_target_mean_r": 0.2,
            "target_top_k_retention_mean": 0.1,
        },
    }
    result = build_score_ranking_capture_audit(
        metadata=metadata,
        baseline_summary=baseline_summary,
        score_sort_summary=score_summary,
        baseline_trade_history=baseline_history,
        score_sort_trade_history=score_history,
        baseline_selected_target_diagnostics=selected,
        score_sort_selected_target_diagnostics=selected,
        selection_diagnostics=selection_diagnostics,
        baseline_daily_capacity=baseline_capacity,
        score_sort_daily_capacity=score_capacity,
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "capture_audit_reconstructs_capital_partial_and_target_capture_metrics",
        (100500.0, 80500.0, 6, 0.6, -0.35, 0.6, -0.35, 100.0, 100.0, 0.0, None),
        (
            result["baseline"]["avg_invested_total"],
            result["score_sort"]["avg_invested_total"],
            result["baseline"]["partial_residual_slot_days"],
            result["baseline"]["aggregate_target_capture_ratio"],
            result["score_sort"]["aggregate_target_capture_ratio"],
            result["baseline"]["raw_mean_target_capture_ratio"],
            result["score_sort"]["raw_mean_target_capture_ratio"],
            result["baseline"]["top_5_entry_dates_share_pct"],
            result["baseline"]["top_entry_month_share_pct"],
            result["baseline"]["industry_coverage_pct"],
            result["baseline"]["top_industry_share_pct"],
        ),
    )
    aggregate_all_targets = _aggregate_capture_ratio(pd.DataFrame({
        "target_raw_r": [2.0, -1.0],
        "r_multiple": [1.0, -0.25],
    }))
    aggregate_target_ge_0_5 = _aggregate_capture_ratio(pd.DataFrame({
        "target_raw_r": [2.0, -1.0],
        "r_multiple": [1.0, -0.25],
    }), min_target_r=0.5)
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "aggregate_capture_uses_sum_realized_over_sum_target_and_threshold_variant",
        (0.75, 0.5),
        (aggregate_all_targets, aggregate_target_ge_0_5),
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "capture_audit_supports_parameter_adaptation_only_after_target_improves_and_sort_only_fails",
        ("ADAPTATION_DIAGNOSTIC_SUPPORTED", True, True, True, False),
        (
            result["decision"]["status"],
            result["decision"]["target_selection_improved"],
            result["decision"]["economic_effect_failed"],
            result["decision"]["parameter_adaptation_candidate"],
            result["decision"]["future_target_used_for_runtime"],
        ),
    )

    yearly = pd.DataFrame([
        {
            "year": 2014, "no_filter_return_pct": 20.0,
            "score_ranking_return_pct": 10.0, "delta_pct": -10.0,
            "is_full_year": True,
        }
    ])
    comparison_delta = {
        key: float(score_summary[key]) - float(baseline_summary[key])
        for key in score_summary
        if isinstance(score_summary.get(key), (int, float))
        and isinstance(baseline_summary.get(key), (int, float))
    }
    comparison_markdown = _markdown_report(
        metadata, baseline_summary, score_summary, comparison_delta, yearly, selection_diagnostics
    )
    comparison_console = _render_strategy_console_report(
        metadata, baseline_summary, score_summary, comparison_delta, yearly,
        selection_diagnostics, color=True,
    )
    capture_console = render_capture_audit_console(result, color=True)
    with patch.dict(os.environ, {"BREAKOUT_QUALITY_COMPACT_CONSOLE": "1"}):
        compact_comparison_console = _render_strategy_console_report(
            metadata, baseline_summary, score_summary, comparison_delta, yearly,
            selection_diagnostics, color=True,
        )
        compact_capture_console = render_capture_audit_console(result, color=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        (output_dir / "score_ranking_capture_audit.html").write_text(
            "legacy", encoding="utf-8"
        )
        payload = write_score_ranking_capture_audit_outputs(result=result, output_dir=temp_dir)
        audit_markdown = (output_dir / "score_ranking_capture_audit.md").read_text(encoding="utf-8")
        outputs_complete = all(
            (output_dir / name).is_file()
            for name in (
                "score_ranking_capture_audit.json", "score_ranking_capture_audit.md",
                "no_filter_capture_lifecycle.csv", "score_ranking_capture_lifecycle.csv",
                "score_ranking_capture_yearly.csv", "score_ranking_capture_scenarios.csv",
            )
        )
        no_html_outputs = not (output_dir / "score_ranking_capture_audit.html").exists()
    compact_strategy_text = strip_ansi(compact_comparison_console)
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "strategy_compare_and_audit_are_separate_read_only_reports",
        (True, True, True),
        (
            "Breakout Quality" in compact_strategy_text,
            "Target Capture" not in compact_strategy_text,
            "Breakout Quality Score 排序資本效率／Target Capture 歸因"
            in strip_ansi(compact_capture_console),
        ),
    )

    add_check(
        results, "synthetic_breakout_quality", case_id,
        "strategy_and_capture_reports_are_console_readable_without_html_outputs",
        (True, True, True, True, True),
        (
            outputs_complete,
            no_html_outputs,
            (
                "🔴 惡化" in comparison_markdown
                and "🟡 注意" in audit_markdown
                and "半倉殘留交易slot-days" in audit_markdown
                and "產業資料覆蓋" in audit_markdown
            ),
            (
                "Breakout Quality Score 排序策略經濟效果對照" in strip_ansi(comparison_console)
                and "Breakout Quality Score 排序資本效率／Target Capture 歸因" in strip_ansi(capture_console)
                and "Exit reason" in strip_ansi(capture_console)
                and "\x1b[" in comparison_console
                and "\x1b[" in capture_console
            ),
            payload["decision"]["status"] == "ADAPTATION_DIAGNOSTIC_SUPPORTED",
        ),
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        relative_root = Path(temp_dir)
        relative_output = relative_root / "outputs" / "filters" / "breakout_quality" / "report.json"
        relative_output.parent.mkdir(parents=True, exist_ok=True)
        relative_output.write_text("{}", encoding="utf-8")
        displayed_path = project_relative_display_path(
            relative_output,
            project_root=relative_root,
        )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "console_artifact_paths_are_project_root_relative_and_forward_slashed",
        "outputs/filters/breakout_quality/report.json",
        displayed_path,
    )

    summary["workflow"] = "score_ranking_capture_audit"
    summary["future_target_runtime"] = False
    return results, summary




def validate_breakout_quality_binary_dl_param_adaptation_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_BINARY_DL_PARAM_ADAPTATION"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    base = V16StrategyParams()

    from filters.breakout_quality.binary_pit_score_store import (
        BINARY_PIT_SCORE_SOURCE,
        build_pass_condition_from_binary_point_in_time_scores,
        load_binary_point_in_time_score_table,
    )
    from filters.breakout_quality.runtime import (
        BREAKOUT_QUALITY_BINARY_PIT_MANIFEST_ENV,
        BREAKOUT_QUALITY_BINARY_PIT_SCORES_ENV,
        BREAKOUT_QUALITY_FILTER_SCORE_SOURCE_ENV,
        breakout_quality_filter_source_context,
        build_breakout_quality_filter_pass_condition,
        get_breakout_quality_filter_source_context,
    )
    from strategies.breakout.search_space import build_trial_params
    from filters.breakout_quality.strategy_compare_engine import (
        _run_scenario as run_strategy_comparison_scenario,
        run_comparison as run_strategy_comparison,
    )
    from filters.breakout_quality.strategy_param_training import (
        ALL_RULE_FILTERS_OFF_OVERRIDES,
        RISK_SEARCH_FIELDS,
        _parse_args as parse_dl_param_adapt_args,
        _validate_binary_pit_optimizer_coverage,
        build_risk_only_fold_overrides,
        run_param_adaptation_gate,
    )
    from tools.optimizer.outer_rolling_oos import (
        FOLD_FIXED_STRATEGY_OVERRIDES_KEY,
        _validate_optimizer_runtime_context,
        resolve_optimizer_session_spec_for_fold,
    )

    param_adapt_args = parse_dl_param_adapt_args([])
    baseline_member_params = params_to_json_dict(base)
    baseline_member_params.update(
        {
            "high_len": 220,
            "atr_len": 19,
            "atr_buy_tol": 1.0,
            "atr_times_init": 4.4,
            "atr_times_trail": 3.7,
            "use_history_threshold": False,
            "use_breakout_reclaim_reentry": True,
            "use_kc": True,
        }
    )
    synthetic_baseline_contract = {
        "payload": {
            "params_ensemble_by_effective_date": {
                "2021-01-01": [
                    {"member_index": 1, "params": baseline_member_params}
                ]
            }
        }
    }
    p2_overrides = build_risk_only_fold_overrides(
        baseline_contract=synthetic_baseline_contract,
        args=param_adapt_args,
        training_dl_enabled=False,
    )
    p3_overrides = build_risk_only_fold_overrides(
        baseline_contract=synthetic_baseline_contract,
        args=param_adapt_args,
        training_dl_enabled=True,
    )
    p2_fixed = p2_overrides["2021-01-01"]
    p3_fixed = p3_overrides["2021-01-01"]
    resolved_fold_spec = resolve_optimizer_session_spec_for_fold(
        {
            "fixed_strategy_param_overrides": {"fixed_risk": 0.01},
            FOLD_FIXED_STRATEGY_OVERRIDES_KEY: p3_overrides,
        },
        oos_start_date="2021-01-01",
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "binary_dl_four_by_two_risk_only_fold_contract",
        (
            tuple(RISK_SEARCH_FIELDS),
            False,
            True,
            True,
            True,
            True,
            "unique_group_sampling",
        ),
        (
            tuple(RISK_SEARCH_FIELDS),
            p2_fixed["use_breakout_quality_filter"],
            p3_fixed["use_breakout_quality_filter"],
            all(not p2_fixed[field] for field in OPTIONAL_ENTRY_FILTER_FIELDS),
            all(
                p3_fixed.get(key) == value
                for key, value in ALL_RULE_FILTERS_OFF_OVERRIDES.items()
            ),
            resolved_fold_spec["fixed_strategy_param_overrides"]
            ["use_breakout_quality_filter"],
            param_adapt_args.experiment_profile,
        ),
    )

    class _RiskOnlySession:
        optimizer_fixed_tp_percent = 0.0

        def __init__(self, values):
            self.values = dict(values)

        def has_fixed_strategy_param(self, field_name):
            return field_name in self.values

        def get_fixed_strategy_param(self, field_name, default=None):
            return self.values.get(field_name, default)

        def resolve_optimizer_tp_percent(self, trial, *, fixed_tp_percent):
            return fixed_tp_percent

    class _RiskOnlyTrial:
        def __init__(self):
            self.calls = []

        def suggest_int(self, field_name, low, high, step=1):
            self.calls.append(field_name)
            return int(low)

        def suggest_float(self, field_name, low, high, step=None):
            self.calls.append(field_name)
            return float(low)

        def suggest_categorical(self, field_name, choices):
            self.calls.append(field_name)
            return list(choices)[0]

    risk_trial = _RiskOnlyTrial()
    risk_params = build_trial_params(_RiskOnlySession(p2_fixed), risk_trial)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "binary_dl_four_by_two_searches_only_risk_fields",
        (set(RISK_SEARCH_FIELDS), 220, False, False, False),
        (
            set(risk_trial.calls),
            risk_params.high_len,
            risk_params.use_bb,
            risk_params.use_kc,
            risk_params.use_breakout_quality_filter,
        ),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        manifest_path = tmp_root / "manifest.json"
        scores_path = tmp_root / "scores.csv"
        score_table = pd.DataFrame(
            {
                "ticker": ["2330", "2330"],
                "date": ["2020-01-02", "2020-01-03"],
                "group_index": [1, 2],
                "dl_quality_score": [0.70, 0.30],
                "fold_id": ["fold_1", "fold_1"],
                "model_information_cutoff": ["2020-01-01", "2020-01-01"],
            }
        )
        score_table.to_csv(scores_path, index=False, encoding="utf-8-sig")
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_type": "binary_point_in_time_scores",
                    "score_table": {"row_count": 2},
                    "score_period": {
                        "start": "2020-01-02",
                        "end": "2020-01-03",
                    },
                }
            ),
            encoding="utf-8",
        )
        load_binary_point_in_time_score_table.cache_clear()
        indexed, _manifest = load_binary_point_in_time_score_table(
            str(manifest_path), str(scores_path)
        )
        frame = pd.DataFrame(
            {"close": [9.0, 10.0, 11.0]},
            index=pd.to_datetime(["2019-12-31", "2020-01-02", "2020-01-03"]),
        )
        candidate = np.array([True, True, True])
        direct_pass = build_pass_condition_from_binary_point_in_time_scores(
            frame,
            ticker="2330",
            score_threshold=0.5,
            candidate_condition=candidate,
            manifest_path=str(manifest_path),
            scores_path=str(scores_path),
        )
        with breakout_quality_filter_source_context(
            score_source=BINARY_PIT_SCORE_SOURCE,
            manifest_path=str(manifest_path),
            scores_path=str(scores_path),
        ):
            runtime_pass = build_breakout_quality_filter_pass_condition(
                frame,
                ticker="2330",
                high_len=220,
                score_threshold=0.5,
                candidate_condition=candidate,
                project_root=str(tmp_root),
            )

        with patch(
            "filters.breakout_quality.strategy_compare_engine._run_scenario_inside_source_context",
            side_effect=lambda **_kwargs: get_breakout_quality_filter_source_context(),
        ):
            scenario_context = run_strategy_comparison_scenario(
                name="synthetic_binary_pit_replay",
                data_dir=tmp_root,
                param_source_kind="single_param",
                params=base,
                start_date="2020-01-02",
                end_date="2020-01-03",
                max_positions=10,
                enable_rotation=False,
                quiet=True,
                filter_source={
                    "score_source": BINARY_PIT_SCORE_SOURCE,
                    "manifest_path": str(manifest_path),
                    "scores_path": str(scores_path),
                },
            )

        class _ContextSession:
            def optimizer_runtime_context(self):
                return breakout_quality_filter_source_context(
                    score_source=BINARY_PIT_SCORE_SOURCE,
                    manifest_path=str(manifest_path),
                    scores_path=str(scores_path),
                )

        runtime_spec = {
            "runtime_context_spec": {
                "module": "filters.breakout_quality.runtime",
                "callable": "breakout_quality_filter_source_context",
                "kwargs": {
                    "score_source": BINARY_PIT_SCORE_SOURCE,
                    "manifest_path": str(manifest_path),
                    "scores_path": str(scores_path),
                },
            }
        }
        _validate_optimizer_runtime_context(_ContextSession(), runtime_spec)
        mismatch_rejected = False
        try:
            bad_spec = json.loads(json.dumps(runtime_spec))
            bad_spec["runtime_context_spec"]["kwargs"]["scores_path"] = str(
                tmp_root / "wrong.csv"
            )
            _validate_optimizer_runtime_context(_ContextSession(), bad_spec)
        except RuntimeError as exc:
            mismatch_rejected = "NON_RETRYABLE_RUNTIME_IDENTITY_ERROR" in str(exc)

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "binary_dl_pit_store_and_optimizer_runtime_identity",
        (2, (True, True, False), (True, True, False), BINARY_PIT_SCORE_SOURCE, True),
        (
            len(indexed),
            tuple(direct_pass),
            tuple(runtime_pass),
            scenario_context.score_source,
            mismatch_rejected,
        ),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        params_path = tmp_root / "params.json"
        params_path.write_text("{}", encoding="utf-8")
        manifest_path = tmp_root / "manifest.json"
        scores_path = tmp_root / "scores.csv"
        pd.DataFrame(
            {
                "ticker": ["2330"],
                "date": ["2021-01-04"],
                "group_index": [1],
                "dl_quality_score": [0.7],
                "fold_id": ["fold_2021"],
                "model_information_cutoff": ["2020-12-31"],
            }
        ).to_csv(scores_path, index=False, encoding="utf-8-sig")
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_type": "binary_point_in_time_scores",
                    "model_architecture": "inception_time_v1",
                    "experiment_profile": "unique_group_sampling",
                    "threshold": 0.5,
                    "score_period": {"start": "2021-01-04", "end": "2021-01-04"},
                    "score_table": {"row_count": 1},
                    "information_contract": "synthetic",
                }
            ),
            encoding="utf-8",
        )
        base_summary = {
            "total_return_pct": 10.0,
            "max_drawdown_pct": 5.0,
            "return_over_max_drawdown": 2.0,
            "annual_return_pct": 2.0,
            "log_r_squared": 0.9,
            "monthly_win_rate_pct": 60.0,
            "trade_count": 2,
            "win_rate_pct": 50.0,
            "payoff_ratio": 2.0,
            "expected_value_r": 0.5,
            "final_equity": 1_100_000.0,
            "avg_exposure_pct": 80.0,
            "max_exposure_pct": 100.0,
            "missed_buy_count": 0,
            "missed_sell_count": 0,
            "reserved_buy_fill_rate_pct": 100.0,
            "normal_trade_count": 2,
            "extended_trade_count": 0,
            "annual_trade_count": 2,
            "benchmark_return_pct": 1.0,
            "benchmark_max_drawdown_pct": 1.0,
            "benchmark_annual_return_pct": 1.0,
        }
        quality_summary = {**base_summary, "total_return_pct": 11.0, "trade_count": 1}
        scenario_calls = []

        def _fake_strategy_scenario(**kwargs):
            scenario_calls.append(kwargs)
            return {
                "marker": kwargs["name"],
                "equity_curve": pd.DataFrame(),
                "trade_history": pd.DataFrame(),
                "profile": {"portfolio_capacity_rows": [], "closed_trade_rows": []},
            }

        with (
            patch(
                "filters.breakout_quality.strategy_compare_engine._resolve_params_path",
                return_value=params_path,
            ),
            patch(
                "filters.breakout_quality.strategy_compare_engine._load_param_source",
                return_value={"kind": "rolling_active_param_ensemble", "payload": {}},
            ),
            patch(
                "filters.breakout_quality.strategy_compare_engine._validate_requested_param_policy",
                return_value={
                    "selector": "base_finalist_best",
                    "member_count_min": 1,
                    "member_count_max": 1,
                    "min_agree": 1,
                },
            ),
            patch(
                "filters.breakout_quality.strategy_compare_engine._build_controlled_param_source_pair",
                return_value=(
                    "rolling_active_param_ensemble",
                    {},
                    {},
                    {},
                    {},
                    {"policy": "synthetic"},
                ),
            ),
            patch(
                "filters.breakout_quality.strategy_compare_engine.get_active_param_ensemble_date_range",
                return_value=("2021-01-01", "2021-12-31"),
            ),
            patch(
                "filters.breakout_quality.strategy_compare_engine._run_scenario",
                side_effect=_fake_strategy_scenario,
            ),
            patch("filters.breakout_quality.strategy_compare_engine._assert_shared_benchmark"),
            patch(
                "filters.breakout_quality.strategy_compare_engine._scenario_summary",
                side_effect=lambda payload: (
                    base_summary if payload["marker"] == "no_filter" else quality_summary
                ),
            ),
            patch(
                "filters.breakout_quality.strategy_compare_engine._build_yearly_comparison",
                return_value=pd.DataFrame(
                    [
                        {
                            "year": 2021,
                            "no_filter_return_pct": 10.0,
                            "quality_filter_return_pct": 11.0,
                "score_ranking_return_pct": 12.0,
                            "delta_pct": 1.0,
                            "is_full_year": True,
                        }
                    ]
                ),
            ),
            patch(
                "filters.breakout_quality.strategy_compare_engine.write_trade_attribution_outputs"
            ),
            patch(
                "filters.breakout_quality.strategy_compare_engine._render_strategy_console_report",
                return_value="synthetic report",
            ),
            patch("filters.breakout_quality.strategy_compare_engine.print_artifact_paths"),
            redirect_stdout(io.StringIO()),
        ):
            direct_comparison = run_strategy_comparison(
                project_root=tmp_root,
                dataset="full",
                params_path=str(params_path),
                param_policy="base-finalist-best",
                comparison_mode="hard-filter",
                filter_id="breakout_quality_v1",
                model_architecture="inception_time_v1",
                experiment_profile="unique_group_sampling",
                output_dir_override=tmp_root / "out",
                comparison_start_date="2021-01-04",
                comparison_end_date="2021-01-04",
                quiet=True,
                hard_filter_source={
                    "score_source": BINARY_PIT_SCORE_SOURCE,
                    "manifest_path": str(manifest_path),
                    "scores_path": str(scores_path),
                },
            )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "binary_dl_strategy_compare_uses_pit_source_and_trade_count",
        (
            BINARY_PIT_SCORE_SOURCE,
            None,
            BINARY_PIT_SCORE_SOURCE,
            2,
            1,
        ),
        (
            direct_comparison["metadata"]["score_source"],
            scenario_calls[0].get("filter_source"),
            scenario_calls[1]["filter_source"]["score_source"],
            direct_comparison["no_filter"]["trade_count"],
            direct_comparison["quality_filter"]["trade_count"],
        ),
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        manifest_path = temp_root / "manifest.json"
        scores_path = temp_root / "scores.csv"
        manifest_path.write_text("{}", encoding="utf-8")
        scores_path.write_text("synthetic", encoding="utf-8")
        env_keys = (
            BREAKOUT_QUALITY_FILTER_SCORE_SOURCE_ENV,
            BREAKOUT_QUALITY_BINARY_PIT_MANIFEST_ENV,
            BREAKOUT_QUALITY_BINARY_PIT_SCORES_ENV,
        )
        env_before = tuple(os.environ.get(key) for key in env_keys)
        project_root = Path(__file__).resolve().parents[2]

        def _probe_spawned_worker_environment(**_kwargs):
            probe_code = (
                "import json; "
                "from filters.breakout_quality.runtime import "
                "get_breakout_quality_filter_source_context; "
                "c=get_breakout_quality_filter_source_context(); "
                "print(json.dumps({'score_source': c.score_source, "
                "'manifest_path': c.manifest_path, 'scores_path': c.scores_path}))"
            )
            probe_env = dict(os.environ)
            existing_pythonpath = str(probe_env.get("PYTHONPATH") or "").strip()
            probe_env["PYTHONPATH"] = (
                str(project_root)
                if not existing_pythonpath
                else str(project_root) + os.pathsep + existing_pythonpath
            )
            completed = subprocess.run(
                [sys.executable, "-c", probe_code],
                cwd=str(project_root),
                env=probe_env,
                capture_output=True,
                text=True,
                check=True,
            )
            return json.loads(completed.stdout.strip())

        with patch(
            "filters.breakout_quality.strategy_compare_engine._run_scenario_inside_source_context",
            side_effect=_probe_spawned_worker_environment,
        ):
            worker_probe = run_strategy_comparison_scenario(
                name="quality_filter",
                data_dir=temp_root,
                param_source_kind="rolling_active_param_ensemble",
                params={},
                start_date="2021-01-04",
                end_date="2021-01-04",
                max_positions=10,
                enable_rotation=False,
                quiet=True,
                filter_source={
                    "score_source": BINARY_PIT_SCORE_SOURCE,
                    "manifest_path": str(manifest_path),
                    "scores_path": str(scores_path),
                },
            )
        env_after = tuple(os.environ.get(key) for key in env_keys)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "binary_dl_strategy_compare_propagates_pit_source_to_spawned_workers",
        (
            BINARY_PIT_SCORE_SOURCE,
            str(manifest_path),
            str(scores_path),
            env_before,
        ),
        (
            worker_probe["score_source"],
            worker_probe["manifest_path"],
            worker_probe["scores_path"],
            env_after,
        ),
    )

    coverage_contract = {
        "meta": {
            "first_oos_date": "2021-01-01",
            "last_oos_date": "2026-01-01",
            "train_window_months": 120,
            "oos_horizon_months": 12,
        }
    }
    partial_pit = _validate_binary_pit_optimizer_coverage(
        binary_pit={
            "ready": True,
            "score_period": {"start": "2016-03-01", "end": "2026-03-02"},
        },
        baseline_contract=coverage_contract,
    )
    partial_coverage = dict(partial_pit["optimizer_coverage"])
    stale_tail_rejected = False
    try:
        _validate_binary_pit_optimizer_coverage(
            binary_pit={
                "ready": True,
                "score_period": {"start": "2016-03-01", "end": "2025-11-30"},
            },
            baseline_contract=coverage_contract,
        )
    except ValueError as exc:
        stale_tail_rejected = "尾端未覆蓋" in str(exc)
    no_overlap_rejected = False
    try:
        _validate_binary_pit_optimizer_coverage(
            binary_pit={
                "ready": True,
                "score_period": {"start": "2026-01-01", "end": "2026-03-02"},
            },
            baseline_contract=coverage_contract,
        )
    except ValueError as exc:
        no_overlap_rejected = "完全沒有重疊" in str(exc)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "binary_dl_pit_partial_optimizer_history_uses_audited_dl_off_fallback",
        (
            {"start": "2011-01-01", "end": "2025-12-31"},
            "pass_through_dl_off",
            {
                "bootstrap_fallback_only": 0,
                "partial_score_history": 6,
                "full_score_history": 0,
            },
            True,
            True,
        ),
        (
            partial_pit["optimizer_required_period"],
            partial_coverage["pre_coverage_policy"],
            partial_coverage["coverage_mode_counts"],
            stale_tail_rejected,
            no_overlap_rejected,
        ),
    )

    from tools.filters.breakout_quality.build_binary_point_in_time_scores import (
        _build_binary_group_table,
        _train_fold as train_binary_pit_fold,
    )

    trade_path_group_events = pd.DataFrame(
        {
            "ticker": ["1101", "1101", "2330", "2330", "2603", "2603"],
            "date": [
                "2020-01-02", "2020-01-02",
                "2020-01-03", "2020-01-03",
                "2020-01-06", "2020-01-06",
            ],
            "group_index": [0, 0, 1, 1, 2, 2],
        }
    )
    trade_path_group_table = _build_binary_group_table(
        trade_path_group_events,
        np.array([0, 0, 1, 1, 2, 2], dtype=np.int64),
        np.array([-1, 1, 0, -1, -1, -1], dtype=np.int64),
    )
    mixed_eligible_rejected = False
    try:
        _build_binary_group_table(
            trade_path_group_events.iloc[:2].copy(),
            np.array([0, 0], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
        )
    except ValueError as exc:
        mixed_eligible_rejected = "混合eligible binary label" in str(exc)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "binary_pit_trade_path_group_ignores_excluded_rows_but_rejects_eligible_conflicts",
        ([1, 2, 4], [1, 0, -1], True),
        (
            trade_path_group_table["event_row"].astype(int).tolist(),
            trade_path_group_table["label"].astype(int).tolist(),
            mixed_eligible_rejected,
        ),
    )

    pit_events = pd.DataFrame(
        {
            "ticker": ["1101", "1101", "1101"],
            "date": pd.to_datetime(["2018-01-02", "2019-01-02", "2020-01-02"]),
        }
    )
    pit_group_table = pd.DataFrame(
        {
            "event_row": [0, 1, 2],
            "ticker": ["1101", "1101", "1101"],
            "date": pd.to_datetime(["2018-01-02", "2019-01-02", "2020-01-02"]),
            "group_index": [0, 1, 2],
            "label_eval_end_date": pd.to_datetime(
                ["2018-02-28", "2019-12-31", "2020-02-28"]
            ),
        }
    )
    pit_bundle = SimpleNamespace(
        features=np.zeros((3, 2, 1), dtype=np.float32),
        context=np.zeros((3, 1), dtype=np.float32),
        labels=np.array([0, 1, 1], dtype=np.int64),
        events=pit_events,
        group_table=pit_group_table,
        profile=SimpleNamespace(
            training_sampling_mode="unique_ticker_date",
            augmentation_name="none",
            augmentation_parameters=lambda: {},
            lr_schedule_parameters=lambda: {},
            optimizer_name="adam",
            lr_schedule_name="none",
            training_weight_reduction="batch_weight_sum",
        ),
        model_spec=SimpleNamespace(as_manifest_payload=lambda: {"name": "synthetic"}),
    )
    pit_args = SimpleNamespace(
        evaluation_batch_size=4,
        evaluation_workers=0,
        epochs=2,
        batch_size=2,
        lr=0.001,
        weight_decay=0.0,
        gradient_clip_norm=1.0,
        seed=42,
        early_stopping_patience=1,
        early_stopping_min_delta=0.0,
        parallel_split_evaluation=False,
        train_prefetch_batches=0,
        experiment_profile="unique_group_sampling",
    )

    class _SyntheticModel:
        def eval(self):
            return self

        def state_dict(self):
            return {}

    class _SyntheticTorch:
        @staticmethod
        def save(payload, path):
            Path(path).write_text(json.dumps({"saved": True}), encoding="utf-8")

    pit_ids = {
        "train_ids": np.array([0], dtype=np.int64),
        "validation_ids": np.array([1], dtype=np.int64),
        "final_ids": np.array([0, 1], dtype=np.int64),
        "score_ids": np.array([2], dtype=np.int64),
    }
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch(
            "tools.filters.breakout_quality.build_binary_point_in_time_scores.train_impl._resolve_training_sampling_indices",
            side_effect=lambda events, labels, indices, **kwargs: (
                np.asarray(indices, dtype=np.int64),
                {"rows": len(indices)},
            ),
        ),
        patch(
            "tools.filters.breakout_quality.build_binary_point_in_time_scores.train_impl._select_epoch_with_inner_validation",
            return_value={"best_epoch": 2, "best_validation_loss": 0.4},
        ),
        patch(
            "tools.filters.breakout_quality.build_binary_point_in_time_scores.train_impl._fit_full_selection",
            return_value={"model": _SyntheticModel()},
        ),
        patch(
            "tools.filters.breakout_quality.build_binary_point_in_time_scores._predict_scores",
            return_value=np.array([0.75], dtype=np.float32),
        ),
    ):
        pit_fold_result = train_binary_pit_fold(
            _SyntheticTorch(),
            pit_bundle,
            {
                "fold_id": "fold_20200102_20201231",
                "score_start": pd.Timestamp("2020-01-02"),
                "score_end": pd.Timestamp("2020-12-31"),
            },
            pit_ids,
            args=pit_args,
            execution_plan=SimpleNamespace(),
            fold_dir=Path(tmpdir),
        )
    pit_frame = pit_fold_result["frame"]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "binary_dl_pit_fold_cutoff_precedes_score_and_exports_identity",
        ("2019-12-31", "2020-01-02", 0.75, True),
        (
            pit_fold_result["model_information_cutoff"],
            pit_frame.iloc[0]["date"],
            round(float(pit_frame.iloc[0]["dl_quality_score"]), 2),
            pd.Timestamp(pit_fold_result["model_information_cutoff"])
            < pd.Timestamp(pit_frame.iloc[0]["date"]),
        ),
    )

    synthetic_adapted = {
        "meta": {
            "first_oos_date": "2021-01-01",
            "last_oos_date": "2021-01-01",
            "train_window_months": 120,
            "oos_horizon_months": 12,
            "trials_per_fold": 1,
        },
        "summary": {"folds": 1},
        "params_ensemble_by_effective_date": {
            "2021-01-01": [
                {
                    "member_index": 1,
                    "params": {
                        **p2_fixed,
                        "atr_len": 7,
                        "atr_buy_tol": 2.1,
                        "atr_times_init": 4.2,
                        "atr_times_trail": 3.8,
                    },
                }
            ]
        },
        "params_by_effective_date": {},
        "params_by_oos_year": {},
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        baseline_path = tmp_root / "models" / "roos_base_best.json"
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_payload = {
            **synthetic_adapted,
            "params_ensemble_by_effective_date": {
                "2021-01-01": [
                    {"member_index": 1, "params": baseline_member_params}
                ]
            },
        }
        baseline_path.write_text(json.dumps(baseline_payload), encoding="utf-8")
        p2_path = tmp_root / "p2.json"
        p3_path = tmp_root / "p3.json"
        p2_path.write_text(json.dumps(synthetic_adapted), encoding="utf-8")
        p3_payload = json.loads(json.dumps(synthetic_adapted))
        p3_payload["params_ensemble_by_effective_date"]["2021-01-01"][0][
            "params"
        ]["use_breakout_quality_filter"] = True
        p3_path.write_text(json.dumps(p3_payload), encoding="utf-8")
        orchestration_contract = {
            "path": baseline_path,
            "payload": baseline_payload,
            "sha256": "synthetic-baseline",
            "meta": {
                "window_mode": "fixed",
                "first_oos_date": "2021-01-01",
                "last_oos_date": "2021-01-01",
                "train_window_months": 120,
                "oos_horizon_months": 12,
            },
            "summary": {"folds": 1},
        }
        comparison_calls = []
        optimizer_calls = []

        def _fake_comparison(**kwargs):
            comparison_calls.append(kwargs)
            index = len(comparison_calls)
            return {
                "no_filter": {
                    "total_return_pct": 100.0 + index,
                    "max_drawdown_pct": 10.0,
                    "return_over_max_drawdown": 10.0,
                    "expected_value_r": 0.5,
                    "avg_exposure_pct": 80.0,
                    "trade_count": 100,
                },
                "quality_filter": {
                    "total_return_pct": 102.0 + index,
                    "max_drawdown_pct": 11.0,
                    "return_over_max_drawdown": 9.0,
                    "expected_value_r": 0.4,
                    "avg_exposure_pct": 70.0,
                    "trade_count": 90,
                },
            }

        def _fake_optimizer_arm(**kwargs):
            optimizer_calls.append(kwargs)
            is_p3 = bool(kwargs["training_dl_enabled"])
            return {
                "params_path": p3_path if is_p3 else p2_path,
                "summary": {
                    "status": "COMPLETED",
                    "parameter_set": "P3" if is_p3 else "P2",
                },
                "contract": {},
            }

        binary_pit = {
            "status": "READY",
            "ready": True,
            "manifest_path": "pit/manifest.json",
            "scores_path": "pit/scores.csv",
            "manifest_absolute": str(tmp_root / "pit" / "manifest.json"),
            "scores_absolute": str(tmp_root / "pit" / "scores.csv"),
            "manifest_sha256": "manifest-sha",
            "scores_sha256": "scores-sha",
            "score_period": {"start": "2011-01-01", "end": "2025-12-31"},
            "error": "",
        }
        with (
            patch(
                "filters.breakout_quality.strategy_param_training.load_model_artifact_contract",
                return_value=SimpleNamespace(),
            ),
            patch(
                "filters.breakout_quality.strategy_param_training._load_baseline_contract",
                return_value=orchestration_contract,
            ),
            patch(
                "filters.breakout_quality.strategy_param_training._ensure_binary_pit",
                return_value=binary_pit,
            ),
            patch(
                "filters.breakout_quality.strategy_param_training.configure_optuna_logging",
            ),
            patch(
                "filters.breakout_quality.strategy_param_training._run_optimizer_arm",
                side_effect=_fake_optimizer_arm,
            ),
            patch(
                "filters.breakout_quality.strategy_param_training.run_comparison",
                side_effect=_fake_comparison,
            ),
            redirect_stdout(io.StringIO()),
        ):
            orchestration_result = run_param_adaptation_gate(
                project_root=tmp_root,
                argv=[
                    "--dataset",
                    "full",
                    "--param-policy",
                    "base-finalist-best",
                    "--trials-per-fold",
                    "1",
                    "--max-positions",
                    "10",
                    "--rotation",
                    "off",
                ],
            )
        report_path = (
            tmp_root
            / "models/research/breakout_quality/binary_dl_filter_param_adaptation/risk_only_rolling/strategy_dl_filter_param_adapt_gate.md"
        )
        report_text = report_path.read_text(encoding="utf-8")

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "binary_dl_four_by_two_gate_orchestration_and_report",
        (
            2,
            (False, True),
            4,
            ("current", "all-off", "all-off", "all-off"),
            (False, True, True, True),
            (BINARY_PIT_SCORE_SOURCE,) * 4,
            ("2021-01-01",) * 4,
            ("2025-12-31",) * 4,
            4,
            "FOUR_BY_TWO_COMPLETE",
            True,
            True,
        ),
        (
            len(optimizer_calls),
            tuple(call["training_dl_enabled"] for call in optimizer_calls),
            len(comparison_calls),
            tuple(call["optional_entry_filter_policy"] for call in comparison_calls),
            tuple(
                bool(call.get("shared_param_overrides"))
                for call in comparison_calls
            ),
            tuple(
                str((call.get("hard_filter_source") or {}).get("score_source"))
                for call in comparison_calls
            ),
            tuple(str(call.get("comparison_start_date")) for call in comparison_calls),
            tuple(str(call.get("comparison_end_date")) for call in comparison_calls),
            len(orchestration_result["matrix"]),
            orchestration_result["status"],
            all(token in report_text for token in ("A0", "B0", "A3", "B3", "B3−A2")),
            all(token in report_text for token in ("100", "90", "binary_point_in_time（八操作點一致；process workers已傳遞）")),
        ),
    )

    summary["workflow"] = "binary_dl_filter_four_parameters_by_two_states"
    summary["binary_pit"] = "REQUIRED_AND_VALIDATED"
    return results, summary

def validate_breakout_quality_strategy_adaptation_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_STRATEGY_ADAPTATION"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from config import breakout_quality as breakout_quality_config
    from config.training_policy import (
        OPTIMIZER_FIXED_TP_PERCENT,
        OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT,
    )
    from strategies.breakout.search_space import build_trial_params
    from tools.filters.breakout_quality.strategy_adapt import (
        _adaptation_arm_identity,
        _adaptation_relative_dir,
        _build_improved_score_baseline_contract,
        _build_training_score_coverage_contract,
        _current_pair_artifact_paths,
        _fixed_strategy_param_overrides,
        _load_current_pair_if_compatible,
        _optimizer_session_spec,
        _parse_args as parse_strategy_adapt_args,
        _render_three_way_console,
        _resolve_selection_pit_models_root,
        _run_rolling_optimizer_arm,
        _validate_fixed_contract,
        _validate_selection_pit_runtime_artifacts,
        _write_current_pair_manifest,
    )
    from tools.optimizer.outer_rolling_oos import (
        _apply_outer_rolling_process_environ,
        _ensure_study_runtime_identity_compatible,
        _is_non_retryable_fold_failure,
        _resolve_outer_rolling_db_file,
        _run_missing_parallel_fold_tasks_sequentially,
        _validate_optimizer_runtime_context,
    )
    from tools.optimizer.session_factory import build_optimizer_session_from_spec

    class FakeTrial:
        def __init__(self):
            self.params = {}
            self.user_attrs = {}

        def suggest_categorical(self, name, choices):
            value = list(choices)[0]
            self.params[name] = value
            return value

        def suggest_int(self, name, low, high, step=1):
            self.params[name] = int(low)
            return int(low)

        def suggest_float(self, name, low, high, step=None):
            self.params[name] = float(low)
            return float(low)

    class FakeStudy:
        def __init__(self, *, identity="", trials=None):
            self.user_attrs = {}
            if identity:
                self.user_attrs["optimizer_runtime_cache_identity"] = identity
            self.trials = list(trials or [])

        def set_user_attr(self, key, value):
            self.user_attrs[key] = value

    configured_workflow_profile = str(
        breakout_quality_config.BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE
    )
    with (
        patch.object(
            breakout_quality_config,
            "BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE",
            STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE,
        ),
        patch.object(
            breakout_quality_config,
            "BREAKOUT_QUALITY_STRATEGY_COMPARISON_MODE",
            "auto",
        ),
        patch.object(
            breakout_quality_config,
            "BREAKOUT_QUALITY_STRATEGY_SCORE_SOURCE",
            "auto",
        ),
        patch.object(
            breakout_quality_config,
            "BREAKOUT_QUALITY_STRATEGY_BUY_SORT",
            "auto",
        ),
    ):
        settings = breakout_quality_config.get_breakout_quality_workflow_settings()
    restored_workflow_profile = str(
        breakout_quality_config.BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "strategy_adaptation_uses_isolated_continuous_workflow_without_mutating_config",
        (
            STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE,
            "score-ranking",
            "selection_point_in_time",
            configured_workflow_profile,
        ),
        (
            settings.experiment_profile,
            settings.strategy_comparison_mode,
            settings.strategy_score_source,
            restored_workflow_profile,
        ),
    )
    args = SimpleNamespace(
        dataset="full",
        filter_id=settings.filter_id,
        model_architecture=settings.model_architecture,
        experiment_profile=settings.experiment_profile,
        param_policy="base-finalist-best",
        ranking_policy="score",
        trials_per_fold=OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT,
        max_positions=7,
        rotation="on",
        fixed_risk=0.02,
        max_position_cap_pct=0.40,
        quiet=True,
    )
    _validate_fixed_contract(args, settings)

    fixed = _fixed_strategy_param_overrides(args, ranking_enabled=True)
    trial = FakeTrial()
    fake_session = SimpleNamespace(
        fixed_strategy_param_overrides=fixed,
        has_fixed_strategy_param=lambda name: name in fixed,
        get_fixed_strategy_param=lambda name, default=None: fixed.get(name, default),
        optimizer_fixed_tp_percent=OPTIMIZER_FIXED_TP_PERCENT,
        resolve_optimizer_tp_percent=lambda _trial, fixed_tp_percent: fixed_tp_percent,
    )
    params = build_trial_params(fake_session, trial)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "single_adapted_optimizer_fixes_score_ranking_without_searching_switch",
        (True, False, False, False, 0.0),
        (
            params.use_breakout_quality_ranking,
            params.use_breakout_quality_filter,
            "use_breakout_quality_ranking" in trial.params,
            "use_breakout_quality_filter" in trial.params,
            params.tp_percent,
        ),
    )

    runtime_contract = {"runtime_identity_sha256": "synthetic-score-adapted-runtime"}
    session_spec = _optimizer_session_spec(
        output_dir=Path(tempfile.gettempdir()),
        args=args,
        runtime_contract=runtime_contract,
        arm_name="score_adapted",
        ranking_enabled=True,
    )
    runtime_kwargs = dict(dict(session_spec["runtime_context_spec"])["kwargs"])
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "score_adapted_session_spec_keeps_formal_pit_runtime_identity",
        (
            True,
            False,
            settings.model_architecture,
            settings.experiment_profile,
            "selection_point_in_time",
            False,
            "synthetic-score-adapted-runtime",
        ),
        (
            session_spec["fixed_strategy_param_overrides"]["use_breakout_quality_ranking"],
            session_spec["fixed_strategy_param_overrides"]["use_breakout_quality_filter"],
            runtime_kwargs["model_architecture"],
            runtime_kwargs["experiment_profile"],
            runtime_kwargs["score_source"],
            "ranking_policy" in runtime_kwargs,
            session_spec["runtime_cache_identity"],
        ),
    )

    r3_args = SimpleNamespace(**vars(args))
    r3_args.ranking_policy = "capital-bucket-then-score"
    r3_arm_name, r3_arm_label = _adaptation_arm_identity(r3_args)
    r3_session_spec = _optimizer_session_spec(
        output_dir=Path(tempfile.gettempdir()),
        args=r3_args,
        runtime_contract={"runtime_identity_sha256": "synthetic-r3-runtime"},
        arm_name=r3_arm_name,
        ranking_enabled=True,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "r3_adapted_uses_isolated_output_and_optimizer_runtime_policy",
        (
            "r3_adapted",
            "R3 Capital-bucket Adapted",
            "capital-bucket-then-score",
            Path(
                "models/research/breakout_quality/score_ranking_adaptation/"
                "capital_bucket_then_score/rolling_validation"
            ),
        ),
        (
            r3_arm_name,
            r3_arm_label,
            r3_session_spec["runtime_context_spec"]["kwargs"]["ranking_policy"],
            _adaptation_relative_dir(r3_args),
        ),
    )

    policy = {
        "objective_mode": "split_train_romd",
        "selection_start_year": 2004,
        "train_start_year": 2004,
        "search_train_end_year": 2013,
        "oos_start_year": 2014,
        "selection_start_date": "2004-01-01",
        "train_start_date": "2004-01-01",
        "search_train_end_date": "2013-12-31",
        "oos_start_date": "2014-01-01",
        "oos_end_date": "2014-12-31",
    }
    real_session = build_optimizer_session_from_spec(
        walk_forward_policy=policy,
        spec=session_spec,
    )
    try:
        _validate_optimizer_runtime_context(real_session, session_spec)
        context_valid = True
        bad_spec = json.loads(json.dumps(session_spec))
        bad_spec["runtime_context_spec"]["kwargs"]["experiment_profile"] = "unique_group_sampling"
        try:
            _validate_optimizer_runtime_context(real_session, bad_spec)
        except RuntimeError as exc:
            context_mismatch_rejected = "NON_RETRYABLE_RUNTIME_IDENTITY_ERROR" in str(exc)
        else:
            context_mismatch_rejected = False
    finally:
        real_session.close_trial_prep_executor()
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "optimizer_and_diagnostics_runtime_context_preflight_is_fail_fast",
        (True, True),
        (context_valid, context_mismatch_rejected),
    )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pit_dir = (
            root / "models" / "filters" / "breakout_quality"
            / settings.filter_id / settings.model_architecture
            / settings.experiment_profile / "point_in_time"
        )
        pit_dir.mkdir(parents=True, exist_ok=True)
        pit_manifest = pit_dir / "selection_point_in_time_manifest.json"
        pit_scores = pit_dir / "selection_point_in_time_scores.csv"
        pit_audit = pit_dir / "selection_point_in_time_audit.json"
        for path, content in (
            (pit_manifest, "{}\n"),
            (pit_scores, "ticker,date,breakout_quality_score\n"),
            (pit_audit, "{}\n"),
        ):
            path.write_text(content, encoding="utf-8")
        valid_contract = SimpleNamespace(
            filter_id=settings.filter_id,
            model_architecture=settings.model_architecture,
            experiment_profile=settings.experiment_profile,
            manifest_path=pit_manifest,
            score_path=pit_scores,
            audit_path=pit_audit,
            model_validation_gate={"status": "PASS"},
        )
        pit_artifacts = _validate_selection_pit_runtime_artifacts(
            pit_contract=valid_contract,
            args=args,
        )
        resolved_pit_models_root = _resolve_selection_pit_models_root(pit_artifacts)
        missing_contract = SimpleNamespace(
            **{**valid_contract.__dict__, "score_path": root / "missing_scores.csv"}
        )
        try:
            _validate_selection_pit_runtime_artifacts(
                pit_contract=missing_contract,
                args=args,
            )
        except FileNotFoundError as exc:
            missing_pit_artifact_rejected = "禁止開始rolling trials" in str(exc)
        else:
            missing_pit_artifact_rejected = False
        wrong_profile_contract = SimpleNamespace(
            **{**valid_contract.__dict__, "experiment_profile": "unique_group_sampling"}
        )
        try:
            _validate_selection_pit_runtime_artifacts(
                pit_contract=wrong_profile_contract,
                args=args,
            )
        except ValueError as exc:
            wrong_profile_rejected = "runtime identity不一致" in str(exc)
        else:
            wrong_profile_rejected = False
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "selection_pit_runtime_artifacts_are_validated_before_rolling_trials",
        (
            {"selection_pit_manifest", "selection_pit_scores", "selection_pit_audit"},
            True,
            True,
        ),
        (set(pit_artifacts), missing_pit_artifact_rejected, wrong_profile_rejected),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "selection_pit_runtime_models_root_is_separate_from_adapted_param_output",
        root / "models",
        resolved_pit_models_root,
    )

    pure_baseline_source = {
        "path": Path("models/research/breakout_quality/selection_strategy_realization/roos_base_best.json"),
        "sha256": "synthetic-baseline-sha",
        "source": {},
        "policy_contract": {},
        "meta": {
            "window_mode": "fixed",
            "first_oos_date": "2014-01-01",
            "last_oos_date": "2018-01-01",
            "train_window_months": 120,
            "oos_horizon_months": 12,
            "trials_per_fold": 100,
            "active_param_policy": "base_finalist_best",
        },
        "summary": {"folds": 4, "selection_period": "2004-01-01~2017-12-31"},
        "payload": {
            "selector": "base_finalist_best",
            "meta": {
                "window_mode": "fixed",
                "first_oos_date": "2014-01-01",
                "last_oos_date": "2018-01-01",
                "train_window_months": 120,
                "oos_horizon_months": 12,
                "trials_per_fold": 100,
                "active_param_policy": "base_finalist_best",
            },
            "summary": {"folds": 4, "selection_period": "2004-01-01~2017-12-31"},
            "folds": [
                {
                    "fold": "1/4",
                    "selection_start_date": "2004-01-01",
                    "selection_end_date": "2013-12-31",
                    "oos_start_date": "2014-01-01",
                    "oos_end_date": "2014-12-31",
                },
                {
                    "fold": "2/4",
                    "selection_start_date": "2005-01-01",
                    "selection_end_date": "2014-12-31",
                    "oos_start_date": "2015-01-01",
                    "oos_end_date": "2015-12-31",
                },
                {
                    "fold": "3/4",
                    "selection_start_date": "2007-01-01",
                    "selection_end_date": "2016-12-31",
                    "oos_start_date": "2017-01-01",
                    "oos_end_date": "2017-12-31",
                },
                {
                    "fold": "4/4",
                    "selection_start_date": "2008-01-01",
                    "selection_end_date": "2017-12-31",
                    "oos_start_date": "2018-01-01",
                    "oos_end_date": "2018-12-31",
                },
            ],
        },
    }
    pure_pit_contract = SimpleNamespace(
        available_from="2007-01-01",
        available_through="2020-12-31",
    )
    all_coverage = _build_training_score_coverage_contract(
        baseline_contract=pure_baseline_source,
        pit_contract=pure_pit_contract,
        reference_score_start="2014-01-01",
    )
    common_baseline, coverage_improvement = _build_improved_score_baseline_contract(
        baseline_contract=pure_baseline_source,
        coverage=all_coverage,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "adaptation_keeps_all_common_folds_when_pit_coverage_improves",
        (
            ["partial_score_history", "partial_score_history", "full_score_history", "full_score_history"],
            ["improved", "improved", "improved", "improved"],
            4,
            "2004-01-01~2017-12-31",
            ["1/4", "2/4", "3/4", "4/4"],
            "2014-01-01",
            "2018-01-01",
            {"start": "2014-01-01", "end": "2018-12-31"},
            True,
            True,
            0,
        ),
        (
            [row["status"] for row in all_coverage["folds"]],
            [row["coverage_change"] for row in all_coverage["folds"]],
            common_baseline["summary"]["folds"],
            common_baseline["summary"]["selection_period"],
            [fold["fold"] for fold in common_baseline["payload"]["folds"]],
            common_baseline["meta"]["first_oos_date"],
            common_baseline["meta"]["last_oos_date"],
            coverage_improvement["comparison_period"],
            coverage_improvement["coverage_improvement_satisfied"],
            coverage_improvement["pre_pit_fallback_allowed_before_actual_pit_start"],
            coverage_improvement["excluded_fold_count"],
        ),
    )

    unchanged_coverage = _build_training_score_coverage_contract(
        baseline_contract=pure_baseline_source,
        pit_contract=SimpleNamespace(
            available_from="2014-01-01",
            available_through="2020-12-31",
        ),
        reference_score_start="2014-01-01",
    )
    try:
        _build_improved_score_baseline_contract(
            baseline_contract=pure_baseline_source,
            coverage=unchanged_coverage,
        )
    except ValueError as exc:
        unchanged_coverage_rejected = "沒有高於參考起點" in str(exc)
    else:
        unchanged_coverage_rejected = False
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "strategy_adapt_requires_strict_coverage_improvement_vs_reference",
        True,
        unchanged_coverage_rejected,
    )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        output_dir = root / "strategy_compare_score_ranking_base_finalist_best_selection_point_in_time"
        output_dir.mkdir(parents=True, exist_ok=True)
        score_path = root / "selection_point_in_time_scores.csv"
        score_manifest_path = root / "selection_point_in_time_manifest.json"
        score_audit_path = root / "selection_point_in_time_audit.json"
        score_path.write_text("ticker,date,breakout_quality_score\n", encoding="utf-8")
        score_manifest_path.write_text("{}\n", encoding="utf-8")
        score_audit_path.write_text("{}\n", encoding="utf-8")
        pair_pit_contract = SimpleNamespace(
            available_from="2014-01-01",
            available_through="2020-12-31",
            score_path=score_path,
            manifest_path=score_manifest_path,
            audit_path=score_audit_path,
        )
        effective_params = {
            "params_by_effective_date": {
                "2014-01-01": {
                    "fixed_risk": float(args.fixed_risk),
                    "max_position_cap_pct": float(args.max_position_cap_pct),
                }
            }
        }
        pair_payload = {
            "metadata": {
                "comparison_mode": "score-ranking",
                "score_source": "selection_point_in_time",
                "dataset": args.dataset,
                "params_file_sha256": "baseline-sha",
                "requested_param_policy": args.param_policy,
                "filter_id": args.filter_id,
                "model_architecture": args.model_architecture,
                "experiment_profile": args.experiment_profile,
                "max_positions": args.max_positions,
                "enable_rotation": True,
                "fixed_risk_override": None,
                "max_position_cap_pct_override": None,
                "comparison_design": "selection_point_in_time_active_param_replay",
                "lookahead_safe_active_param_schedule": True,
                "comparison_period": {
                    "start": pair_pit_contract.available_from,
                    "end": pair_pit_contract.available_through,
                },
                "score_path": str(score_path),
                "score_manifest_path": str(score_manifest_path),
                "score_audit_path": str(score_audit_path),
                "score_table": {},
                "no_filter_params": effective_params,
                "score_ranking_params": effective_params,
            }
        }
        artifact_paths = _current_pair_artifact_paths(output_dir)
        for artifact_path in artifact_paths.values():
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text("synthetic\n", encoding="utf-8")
        artifact_paths["strategy_comparison_json"].write_text(
            json.dumps(pair_payload, ensure_ascii=False), encoding="utf-8"
        )
        stale_manifest = output_dir / "adaptation_pair_manifest.json"
        stale_manifest.write_text(
            json.dumps({
                "runtime_identity_sha256": "stale",
                "artifacts": {},
            }),
            encoding="utf-8",
        )
        loaded_pair, pair_issues = _load_current_pair_if_compatible(
            root=root,
            output_dir=output_dir,
            runtime_identity_sha256="current-identity",
            args=args,
            pit_contract=pair_pit_contract,
            baseline_contract={"sha256": "baseline-sha"},
            comparison_period={"start": "2014-01-01", "end": "2020-12-31"},
        )
        _write_current_pair_manifest(
            root=root,
            output_dir=output_dir,
            runtime_identity_sha256="current-identity",
        )
        refreshed_manifest = json.loads(stale_manifest.read_text(encoding="utf-8"))

        invalid_payload = json.loads(json.dumps(pair_payload))
        invalid_payload["metadata"]["score_ranking_params"][
            "params_by_effective_date"
        ]["2014-01-01"]["fixed_risk"] = float(args.fixed_risk) + 0.01
        artifact_paths["strategy_comparison_json"].write_text(
            json.dumps(invalid_payload, ensure_ascii=False), encoding="utf-8"
        )
        rejected_pair, rejected_issues = _load_current_pair_if_compatible(
            root=root,
            output_dir=output_dir,
            runtime_identity_sha256="current-identity",
            args=args,
            pit_contract=pair_pit_contract,
            baseline_contract={"sha256": "baseline-sha"},
            comparison_period={"start": "2014-01-01", "end": "2020-12-31"},
        )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "existing_strategy_compare_without_explicit_overrides_is_reused_by_effective_params",
        (True, [], "current-identity", True, True),
        (
            loaded_pair is not None,
            pair_issues,
            refreshed_manifest.get("runtime_identity_sha256"),
            rejected_pair is None,
            any("fixed_risk" in issue for issue in rejected_issues),
        ),
    )

    with (
        patch.object(
            breakout_quality_config,
            "BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE",
            STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE,
        ),
        patch.object(
            breakout_quality_config,
            "BREAKOUT_QUALITY_STRATEGY_COMPARISON_MODE",
            "auto",
        ),
        patch.object(
            breakout_quality_config,
            "BREAKOUT_QUALITY_STRATEGY_SCORE_SOURCE",
            "auto",
        ),
        patch.object(
            breakout_quality_config,
            "BREAKOUT_QUALITY_STRATEGY_BUY_SORT",
            "auto",
        ),
    ):
        parsed_adapt_args = parse_strategy_adapt_args([])
    app_source = (Path(__file__).resolve().parents[2] / "apps" / "breakout_quality.py").read_text(
        encoding="utf-8"
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "strategy_adapt_cli_uses_isolated_continuous_risk_and_position_cap_defaults",
        (
            settings.strategy_adapt_fixed_risk,
            settings.strategy_adapt_max_position_cap_pct,
            settings.experiment_profile,
            False,
        ),
        (
            parsed_adapt_args.fixed_risk,
            parsed_adapt_args.max_position_cap_pct,
            parsed_adapt_args.experiment_profile,
            '"strategy-adapt": "tools.filters.breakout_quality.strategy_adapt"'
            in app_source,
        ),
    )

    with patch.dict(os.environ, {}, clear=True):
        _apply_outer_rolling_process_environ({
            "OPTIMIZER_OUTER_ROLLING_STUDY_STORAGE": "sqlite",
            "OPTIMIZER_ROLLING_RESUME_EXISTING_STUDIES": "1",
            "V16_MODELS_DIR": "synthetic-models",
            "UNRELATED_ENV": "ignored",
        })
        inherited_environment = (
            os.environ.get("OPTIMIZER_OUTER_ROLLING_STUDY_STORAGE"),
            os.environ.get("OPTIMIZER_ROLLING_RESUME_EXISTING_STUDIES"),
            os.environ.get("V16_MODELS_DIR"),
            os.environ.get("UNRELATED_ENV"),
        )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "spawned_fold_workers_receive_adaptation_runtime_environment",
        ("sqlite", "1", "synthetic-models", None),
        inherited_environment,
    )

    manifest_error = RuntimeError(
        "portfolio replay 失敗 | FileNotFoundError: 找不到 breakout quality 正式 manifest"
    )
    pit_path_error = RuntimeError(
        "portfolio replay 失敗 | FileNotFoundError: 找不到Selection PIT score: "
        "adapted/active_params/filters/breakout_quality/.../selection_point_in_time_scores.csv"
    )
    ensemble_contract_error = RuntimeError(
        "portfolio replay 失敗 | ValueError: "
        "同一 ticker／score_date 的 ensemble members Score availability不一致"
    )
    generic_error = RuntimeError("process pool broken")
    try:
        _run_missing_parallel_fold_tasks_sequentially(
            tasks=[{"fold_idx": 1, "fold_count": 1, "oos_year": 201401}],
            rows=[],
            fold_timing_rows=[],
            chain_state={},
            cause=manifest_error,
        )
    except RuntimeError as exc:
        fallback_blocked = "序列fallback不會改變結果" in str(exc)
    else:
        fallback_blocked = False
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "runtime_identity_manifest_errors_are_non_retryable",
        (True, True, True, False, True),
        (
            _is_non_retryable_fold_failure(manifest_error),
            _is_non_retryable_fold_failure(pit_path_error),
            _is_non_retryable_fold_failure(ensemble_contract_error),
            _is_non_retryable_fold_failure(generic_error),
            fallback_blocked,
        ),
    )

    study_session = SimpleNamespace(
        runtime_cache_identity="runtime-a",
        fixed_strategy_param_overrides=fixed,
    )
    compatible_study = FakeStudy(identity="runtime-a")
    _ensure_study_runtime_identity_compatible(compatible_study, study_session)
    stale_study = FakeStudy(identity="runtime-b", trials=[FakeTrial()])
    try:
        _ensure_study_runtime_identity_compatible(stale_study, study_session)
    except RuntimeError as exc:
        stale_study_rejected = "NON_RETRYABLE_RUNTIME_IDENTITY_ERROR" in str(exc)
    else:
        stale_study_rejected = False
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "persistent_study_resume_is_bound_to_runtime_identity",
        ("runtime-a", True),
        (
            compatible_study.user_attrs.get("optimizer_runtime_cache_identity"),
            stale_study_rejected,
        ),
    )

    with tempfile.TemporaryDirectory() as tmp:
        arm_root = Path(tmp)
        pit_models_root = arm_root / "models"
        pit_models_root.mkdir(parents=True, exist_ok=True)
        arm_output = arm_root / "adaptation"
        arm_output.mkdir(parents=True, exist_ok=True)
        arm_args = SimpleNamespace(**vars(args))
        arm_args.trials_per_fold = 3
        arm_baseline = {
            "meta": {
                "first_oos_date": "2014-01-01",
                "last_oos_date": "2014-01-01",
                "train_window_months": 120,
                "oos_horizon_months": 12,
            },
            "summary": {"folds": 1},
        }
        arm_runtime = {
            "runtime_identity_sha256": "synthetic-arm-path-runtime",
            "rolling_policy": {
                "trials_per_fold": 3,
                "trials_per_fold_source": "synthetic",
            },
        }
        captured_arm_paths = {}

        def _fake_outer_rolling(**kwargs):
            captured_arm_paths["runtime_models_dir"] = kwargs["environ"]["V16_MODELS_DIR"]
            captured_arm_paths["paramset_models_dir"] = kwargs["paramset_models_dir"]
            destination = Path(kwargs["paramset_models_dir"])
            destination.mkdir(parents=True, exist_ok=True)
            payload = {
                "meta": {**arm_baseline["meta"], "trials_per_fold": 3},
                "summary": {"folds": 1},
                "params_ensemble_by_effective_date": {
                    "2014-01-01": [{"params": {"tp_percent": 0.0}}],
                },
            }
            (destination / "roos_base_best.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            return 0

        with patch(
            "tools.filters.breakout_quality.strategy_adapt.run_outer_rolling_oos",
            side_effect=_fake_outer_rolling,
        ), patch(
            "tools.filters.breakout_quality.strategy_adapt.get_dataset_dir",
            return_value=str(arm_root / "data"),
        ):
            arm_result = _run_rolling_optimizer_arm(
                root=arm_root,
                args=arm_args,
                settings=SimpleNamespace(seed=42),
                baseline_contract=arm_baseline,
                base_policy={},
                runtime_contract=arm_runtime,
                output_dir=arm_output,
                arm_name="score_adapted",
                arm_label="Score Adapted",
                ranking_enabled=True,
                pit_models_root=pit_models_root,
            )
        active_param_dir = (arm_output / "score_adapted" / "active_params").resolve()
        materialized_member = json.loads(
            Path(arm_result["params_path"]).read_text(encoding="utf-8")
        )["params_ensemble_by_effective_date"]["2014-01-01"][0]["params"]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "score_adapted_runtime_models_root_and_paramset_output_are_separate",
        (
            pit_models_root.resolve(),
            active_param_dir,
            True,
            False,
        ),
        (
            Path(captured_arm_paths["runtime_models_dir"]),
            Path(captured_arm_paths["paramset_models_dir"]),
            materialized_member["use_breakout_quality_ranking"],
            materialized_member["use_breakout_quality_filter"],
        ),
    )

    with tempfile.TemporaryDirectory() as tmp:
        task = {
            "optimizer_session_spec": {
                "runtime_cache_identity": "abcdef0123456789abcdef0123456789"
            },
            "seed_ensemble_member": False,
        }
        env = {"OPTIMIZER_ROLLING_RESUME_EXISTING_STUDIES": "1"}
        first_path, first_resumed = _resolve_outer_rolling_db_file(
            output_dir=tmp,
            session_ts="one",
            oos_year=201401,
            task=task,
            environ=env,
        )
        Path(first_path).parent.mkdir(parents=True, exist_ok=True)
        Path(first_path).touch()
        second_path, second_resumed = _resolve_outer_rolling_db_file(
            output_dir=tmp,
            session_ts="two",
            oos_year=201401,
            task=task,
            environ=env,
        )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "score_adapted_sqlite_study_uses_stable_runtime_identity_path",
        (False, True, True),
        (first_resumed, second_resumed, first_path == second_path),
    )

    compact_payload = {
        "metadata": {
            "comparison_period": {"start": "2014-01-01", "end": "2020-12-31"},
            "score_source": "selection_point_in_time",
        },
        "baseline": {},
        "sort_only": {},
        "param_only": {},
        "adapted": {},
        "old_params_ranking_delta": {},
        "adapted_params_ranking_delta": {},
        "old_ranking_param_delta": {},
        "score_ranking_param_delta": {},
        "overall_delta": {},
        "current_capture_audit": {},
        "adapted_params_ranking_capture_audit": {},
        "yearly": [],
        "selection_diagnostics": {
            "baseline": {}, "sort_only": {}, "param_only": {}, "adapted": {},
        },
        "parameter_comparison": [],
        "rolling_validation": {
            "trials_per_fold": 200,
            "trials_per_fold_source": "config.training_policy.OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT",
            "baseline_trials_per_fold": 100,
            "train_window_months": 120,
            "oos_horizon_months": 12,
            "score_adapted_search_reused": False,
            "training_score_coverage": {
                "fold_count": 1,
                "pit_score_period": {"start": "2007-01-01", "end": "2020-12-31"},
                "coverage_reference": {"score_start": "2014-01-01"},
                "reference_weighted_calendar_coverage_ratio": 0.0,
                "actual_weighted_calendar_coverage_ratio": 0.7,
                "weighted_calendar_coverage_ratio_gain": 0.7,
                "coverage_improved_folds": 1,
                "bootstrap_fallback_only_folds": 0,
                "partial_score_history_folds": 1,
                "full_score_history_folds": 0,
                "folds": [{
                    "fold": "1/1",
                    "selection_period": "2004-01-01~2013-12-31",
                    "oos_period": "2014-01-01~2014-12-31",
                    "reference_calendar_coverage_ratio": 0.0,
                    "calendar_coverage_ratio": 0.7,
                    "coverage_ratio_delta": 0.7,
                    "status": "partial_score_history",
                }],
            },
        },
    }
    with patch.dict(os.environ, {"BREAKOUT_QUALITY_COMPACT_CONSOLE": "1"}):
        compact_console = _render_three_way_console(compact_payload)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "single_training_four_replay_report_separates_both_ranking_effects",
        (True, True, True, True, True),
        (
            "Baseline" in compact_console and "Sort Only" in compact_console,
            "Param Only" in compact_console and "Adapted" in compact_console,
            "Sort Only − Baseline" in compact_console,
            "Adapted − Param Only" in compact_console,
            "新參數只由指定ranking policy的rolling optimizer訓練一次" in compact_console,
        ),
    )

    project_root = Path(__file__).resolve().parents[2]
    app_source = (project_root / "apps" / "breakout_quality.py").read_text(
        encoding="utf-8"
    )
    adapt_source = (
        project_root / "tools" / "filters" / "breakout_quality" / "strategy_adapt.py"
    ).read_text(encoding="utf-8")
    outer_source = (
        project_root / "tools" / "optimizer" / "outer_rolling_oos.py"
    ).read_text(encoding="utf-8")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "strategy_adapt_runs_one_optimizer_and_four_controlled_replays",
        True,
        '"strategy-adapt": "tools.filters.breakout_quality.strategy_adapt"' not in app_source
        and "[2] 驗證策略參數適應" not in app_source
        and '"r3_adapted", "R3 Capital-bucket Adapted"' in adapt_source
        and 'R3_ADAPTATION_RELATIVE_DIR' in adapt_source
        and 'arm_name="baseline_adapted"' not in adapt_source
        and '"optimizer_training_arm_count": 1' in adapt_source
        and '"--ranking-policy", str(args.ranking_policy)' in adapt_source
        and '"ranking_policy": str(args.ranking_policy)' in adapt_source
        and 'arm_name="param_only"' in adapt_source
        and 'arm_name="adapted"' in adapt_source
        and 'params_path=adapted_arm["params_path"]' in adapt_source
        and "run_comparison(" in adapt_source
        and "coverage提升" in adapt_source
        and '"pre_pit_missing_scores_use_existing_buy_sort_fallback": True' in adapt_source
        and '"optimizer_training_score_coverage_must_improve_vs_reference": True' in adapt_source
        and 'OPTIMIZER_OUTER_ROLLING_STUDY_STORAGE"] = "sqlite"' in adapt_source
        and 'OPTIMIZER_ROLLING_RESUME_EXISTING_STUDIES"] = "1"' in adapt_source
        and 'outer_environ["V16_MODELS_DIR"] = str(Path(pit_models_root).resolve())' in adapt_source
        and 'paramset_models_dir=str(active_param_output_dir.resolve())' in adapt_source
        and 'paramset_models_dir: str | None = None' in outer_source,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "outer_rolling_keeps_runtime_context_through_diagnostics_and_chain",
        True,
        outer_source.count("with session.optimizer_runtime_context():") >= 2
        and "with chain_session.optimizer_runtime_context():" in outer_source
        and "_validate_optimizer_runtime_context(session, session_spec)" in outer_source
        and "_is_non_retryable_fold_failure(cause)" in outer_source,
    )

    summary["workflow"] = "selection_ranking_parameter_2x2_single_adapted_optimizer"
    summary["result_status"] = "IMPLEMENTED_RESULT_NOT_AVAILABLE"
    summary["trials_per_fold"] = int(OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT)
    summary["optimization_arms"] = ["score_adapted", "r3_adapted_supported"]
    summary["replay_arms"] = ["baseline", "sort_only", "param_only", "adapted"]
    summary["final_selection_refit"] = False
    return results, summary



def validate_breakout_quality_trade_path_label_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_TRADE_PATH_LABEL"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from filters.breakout_quality.contract import (
        TRADE_PATH_FILTER_ID,
        TRADE_PATH_LABEL_OBJECTIVE,
        expected_label_policy_for_filter_id,
    )
    from filters.breakout_quality.paths import resolve_filter_artifact_paths
    from core.backtest_core import run_v16_backtest
    from filters.breakout_quality.trade_path_label import (
        TRADE_PATH_LABEL_CONTRACT_VERSION,
        TRADE_PATH_LABEL_ID,
        TRADE_PATH_LABEL_REASON_STATUS,
        TRADE_PATH_LABEL_STATUS_EXCLUDED,
        TRADE_PATH_LABEL_STATUS_PASS,
        TRADE_PATH_LABEL_STATUS_REJECT,
        TRADE_PATH_REASON_REALIZED_NET_NONPOSITIVE,
        TRADE_PATH_REASON_REALIZED_NET_PROFIT,
        TRADE_PATH_REASON_UNFILLED_DATA_END,
        TRADE_PATH_RESEARCH_FILTER_ID,
        build_trade_path_excluded_event_update,
        simulate_realized_trade_path_label,
    )
    from tools.filters.breakout_quality.build_trade_path_labels import (
        _load_valid_ticker_shard,
        _ticker_shard_path,
        _validate_historical_teacher_baseline_period,
        _write_ticker_shard,
    )

    from tools.filters.breakout_quality.strategy_trade_path_label_gate import (
        BASE_IDENTITY_FILENAMES,
        _assert_same_base_artifacts,
        _render_delta_table as _render_trade_path_gate_delta_table,
        _render_summary_table as _render_trade_path_gate_summary_table,
    )

    valid_teacher_payload = {
        "params_ensemble_by_effective_date": {
            f"{year}-01-01": [{"params": {"atr_len": 5}}]
            for year in range(2014, 2021)
        }
    }
    valid_teacher_meta = {
        "first_oos_date": "2014-01-01",
        "last_oos_date": "2020-12-01",
    }
    accepted_dates = _validate_historical_teacher_baseline_period(
        payload=valid_teacher_payload,
        meta=valid_teacher_meta,
    )
    incomplete_teacher_payload = {
        "params_ensemble_by_effective_date": {
            key: value
            for key, value in valid_teacher_payload[
                "params_ensemble_by_effective_date"
            ].items()
            if key != "2020-01-01"
        }
    }
    try:
        _validate_historical_teacher_baseline_period(
            payload=incomplete_teacher_payload,
            meta=valid_teacher_meta,
        )
    except ValueError as exc:
        incomplete_schedule_rejected = "完整涵蓋2014～2020" in str(exc)
    else:
        incomplete_schedule_rejected = False
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "trade_path_historical_teacher_accepts_2020_december_oos_boundary_and_validates_effective_schedule",
        ("2014-01-01", "2020-01-01", 7, True),
        (
            accepted_dates[0],
            accepted_dates[-1],
            len(accepted_dates),
            incomplete_schedule_rejected,
        ),
    )

    policy = expected_label_policy_for_filter_id(TRADE_PATH_FILTER_ID)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "trade_path_label_identity_and_policy_are_isolated_from_9a",
        (
            TRADE_PATH_FILTER_ID,
            TRADE_PATH_LABEL_OBJECTIVE,
            "pending",
            "reuse_original_event",
            TRADE_PATH_LABEL_CONTRACT_VERSION,
            "realized_net_r_gt_zero",
            "realized_net_r_le_zero",
            "formal_single_stock_forced_closeout",
            "same_explicit_single_stock_sizing_capital",
            "exclude_from_binary_training",
        ),
        (
            TRADE_PATH_RESEARCH_FILTER_ID,
            policy.get("label_objective"),
            policy.get("initial_miss_buy_status"),
            policy.get("continuation_event_identity"),
            policy.get("label_contract_version"),
            policy.get("filled_positive_rule"),
            policy.get("filled_nonpositive_rule"),
            policy.get("filled_data_end_rule"),
            policy.get("sizing_capital_rule"),
            policy.get("unfilled_terminal_rule"),
        ),
    )

    params = V16StrategyParams(
        use_breakout_ema_filter=False,
        use_bb=False,
        use_kc=False,
        use_vol=False,
        use_breakout_return_filter=False,
        use_breakout_false_filter=False,
        use_history_threshold=False,
        use_breakout_reclaim_reentry=False,
        use_breakout_quality_filter=False,
        use_breakout_quality_ranking=False,
        high_len=3,
        atr_len=3,
        atr_buy_tol=1.0,
        atr_times_init=2.0,
        atr_times_trail=2.0,
        tp_percent=0.0,
        initial_capital=1_000_000.0,
        fixed_risk=0.01,
        max_position_cap_pct=0.3,
    )
    dates = pd.date_range("2020-01-01", periods=7, freq="B")
    frame = pd.DataFrame(
        {
            "Open": [95.0, 98.0, 100.0, 103.0, 101.0, 103.0, 104.0],
            "High": [97.0, 100.0, 102.0, 103.5, 102.0, 105.0, 105.0],
            "Low": [94.0, 97.0, 99.0, 102.0, 100.0, 102.0, 103.0],
            "Close": [96.0, 99.0, 101.0, 103.0, 101.0, 104.0, 104.0],
            "Volume": [1000.0] * 7,
        },
        index=dates,
    )
    frame.attrs["ticker"] = "2330"
    atr = np.full(len(frame), 2.0, dtype=np.float64)
    buy_condition = np.array([False, False, True, False, False, False, False])
    sell_condition = np.array([False, False, False, False, True, False, False])
    buy_limits = np.array([np.nan, np.nan, 101.0, np.nan, np.nan, np.nan, np.nan])
    result = simulate_realized_trade_path_label(
        frame,
        ticker="2330",
        signal_pos=2,
        params=params,
        teacher_effective_date="2020-01-01",
        precomputed_signals=(atr, buy_condition, sell_condition, buy_limits),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "trade_path_initial_miss_remains_pending_then_continuation_fill_gets_one_terminal_label",
        (
            LABEL_PASS,
            TRADE_PATH_LABEL_STATUS_PASS,
            TRADE_PATH_REASON_REALIZED_NET_PROFIT,
            "CONTINUATION_FILL",
            1,
            True,
            "IND_SELL",
            True,
        ),
        (
            result.label,
            result.status,
            result.reason,
            result.fill_type,
            result.continuation_wait_bars,
            result.initial_missed_buy,
            result.exit_reason,
            bool(result.realized_net_r is not None and result.realized_net_r > 0.0),
        ),
    )

    _stats, single_trade_logs = run_v16_backtest(
        frame,
        params=params,
        return_logs=True,
        precomputed_signals=(atr, buy_condition, sell_condition, buy_limits),
        ticker="2330",
    )
    single_trade = single_trade_logs[0]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "trade_path_continuation_completed_path_matches_single_stock_entry_exit_and_accounting",
        (
            result.fill_date,
            result.entry_price,
            result.exit_date,
            result.exit_price,
            result.exit_reason,
            result.realized_net_pnl,
            result.realized_net_r,
        ),
        (
            pd.Timestamp(single_trade["entry_date"]).strftime("%Y-%m-%d"),
            float(single_trade["entry_price"]),
            pd.Timestamp(single_trade["exit_date"]).strftime("%Y-%m-%d"),
            float(single_trade["exit_price"]),
            str(single_trade["exit_reason"]),
            float(single_trade["pnl"]),
            float(single_trade["r_mult"]),
        ),
    )

    partial_tp_params = replace(params, tp_percent=0.5, atr_times_trail=1.0)
    partial_tp_frame = pd.DataFrame(
        {
            "Open": [95.0, 98.0, 100.0, 98.0, 105.0, 105.0, 106.0],
            "High": [97.0, 100.0, 102.0, 109.0, 106.0, 106.0, 107.0],
            "Low": [94.0, 97.0, 99.0, 97.5, 104.5, 104.0, 105.0],
            "Close": [96.0, 99.0, 101.0, 99.0, 105.5, 105.0, 106.0],
            "Volume": [1000.0] * 7,
        },
        index=pd.date_range("2020-04-01", periods=7, freq="B"),
    )
    partial_tp_frame.attrs["ticker"] = "2330"
    partial_tp_atr = np.full(len(partial_tp_frame), 5.0, dtype=np.float64)
    partial_tp_buy = np.array([False, False, True, False, False, False, False])
    partial_tp_sell = np.array([False, False, False, False, True, False, False])
    partial_tp_limits = np.array(
        [np.nan, np.nan, 100.0, np.nan, np.nan, np.nan, np.nan]
    )
    partial_tp_result = simulate_realized_trade_path_label(
        partial_tp_frame,
        ticker="2330",
        signal_pos=2,
        params=partial_tp_params,
        teacher_effective_date="2020-01-01",
        precomputed_signals=(
            partial_tp_atr,
            partial_tp_buy,
            partial_tp_sell,
            partial_tp_limits,
        ),
    )
    _partial_tp_stats, partial_tp_logs = run_v16_backtest(
        partial_tp_frame,
        params=partial_tp_params,
        return_logs=True,
        precomputed_signals=(
            partial_tp_atr,
            partial_tp_buy,
            partial_tp_sell,
            partial_tp_limits,
        ),
        ticker="2330",
    )
    partial_tp_trade = partial_tp_logs[0]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "trade_path_partial_take_profit_then_indicator_exit_matches_single_stock",
        (
            "INITIAL_FILL",
            "IND_SELL",
            float(partial_tp_trade["entry_price"]),
            float(partial_tp_trade["exit_price"]),
            float(partial_tp_trade["pnl"]),
            float(partial_tp_trade["r_mult"]),
        ),
        (
            partial_tp_result.fill_type,
            partial_tp_result.exit_reason,
            partial_tp_result.entry_price,
            partial_tp_result.exit_price,
            partial_tp_result.realized_net_pnl,
            partial_tp_result.realized_net_r,
        ),
    )

    reject_frame = pd.DataFrame(
        {
            "Open": [95.0, 98.0, 100.0, 100.0, 96.0],
            "High": [97.0, 100.0, 102.0, 101.0, 97.0],
            "Low": [94.0, 97.0, 99.0, 99.0, 95.0],
            "Close": [96.0, 99.0, 101.0, 100.0, 96.0],
            "Volume": [1000.0] * 5,
        },
        index=pd.date_range("2020-02-03", periods=5, freq="B"),
    )
    reject_frame.attrs["ticker"] = "2330"
    reject_atr = np.full(len(reject_frame), 2.0, dtype=np.float64)
    reject_buy = np.array([False, False, True, False, False])
    reject_sell = np.zeros(len(reject_frame), dtype=bool)
    reject_limits = np.array([np.nan, np.nan, 101.0, np.nan, np.nan])
    reject_result = simulate_realized_trade_path_label(
        reject_frame,
        ticker="2330",
        signal_pos=2,
        params=params,
        teacher_effective_date="2020-01-01",
        precomputed_signals=(reject_atr, reject_buy, reject_sell, reject_limits),
    )
    _reject_stats, reject_logs = run_v16_backtest(
        reject_frame,
        params=params,
        return_logs=True,
        precomputed_signals=(reject_atr, reject_buy, reject_sell, reject_limits),
        ticker="2330",
    )
    reject_trade = reject_logs[0]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "trade_path_negative_completed_path_is_reject_and_matches_single_stock",
        (
            LABEL_REJECT,
            TRADE_PATH_LABEL_STATUS_REJECT,
            TRADE_PATH_REASON_REALIZED_NET_NONPOSITIVE,
            "STOP",
            float(reject_trade["pnl"]),
            float(reject_trade["r_mult"]),
        ),
        (
            reject_result.label,
            reject_result.status,
            reject_result.reason,
            reject_result.exit_reason,
            reject_result.realized_net_pnl,
            reject_result.realized_net_r,
        ),
    )

    data_end_frame = pd.DataFrame(
        {
            "Open": [95.0, 98.0, 100.0, 100.0, 103.0],
            "High": [97.0, 100.0, 102.0, 103.0, 105.0],
            "Low": [94.0, 97.0, 99.0, 99.0, 102.0],
            "Close": [96.0, 99.0, 101.0, 102.0, 104.0],
            "Volume": [1000.0] * 5,
        },
        index=pd.date_range("2020-03-02", periods=5, freq="B"),
    )
    data_end_frame.attrs["ticker"] = "2330"
    data_end_atr = np.full(len(data_end_frame), 2.0, dtype=np.float64)
    data_end_buy = np.array([False, False, True, False, False])
    data_end_sell = np.zeros(len(data_end_frame), dtype=bool)
    data_end_limits = np.array([np.nan, np.nan, 101.0, np.nan, np.nan])
    data_end_result = simulate_realized_trade_path_label(
        data_end_frame,
        ticker="2330",
        signal_pos=2,
        params=params,
        teacher_effective_date="2020-01-01",
        precomputed_signals=(data_end_atr, data_end_buy, data_end_sell, data_end_limits),
    )
    _data_end_stats, data_end_logs = run_v16_backtest(
        data_end_frame,
        params=params,
        return_logs=True,
        precomputed_signals=(data_end_atr, data_end_buy, data_end_sell, data_end_limits),
        ticker="2330",
    )
    data_end_trade = data_end_logs[0]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "trade_path_filled_at_data_end_uses_formal_single_stock_forced_closeout",
        (
            True,
            "FORCED_CLOSEOUT",
            float(data_end_trade["exit_price"]),
            float(data_end_trade["pnl"]),
            float(data_end_trade["r_mult"]),
        ),
        (
            data_end_result.forced_closeout,
            data_end_result.exit_reason,
            data_end_result.exit_price,
            data_end_result.realized_net_pnl,
            data_end_result.realized_net_r,
        ),
    )

    unfilled_frame = data_end_frame.copy()
    unfilled_frame[["Open", "High", "Low", "Close"]] = [
        [95.0, 97.0, 94.0, 96.0],
        [98.0, 100.0, 97.0, 99.0],
        [100.0, 102.0, 99.0, 101.0],
        [105.0, 106.0, 104.0, 105.0],
        [106.0, 107.0, 105.0, 106.0],
    ]
    unfilled_frame.attrs["ticker"] = "2330"
    unfilled_result = simulate_realized_trade_path_label(
        unfilled_frame,
        ticker="2330",
        signal_pos=2,
        params=params,
        teacher_effective_date="2020-01-01",
        precomputed_signals=(data_end_atr, data_end_buy, data_end_sell, data_end_limits),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "trade_path_missed_buy_without_fill_is_excluded_not_reject",
        (
            LABEL_INVALID,
            TRADE_PATH_LABEL_STATUS_EXCLUDED,
            True,
            True,
            None,
        ),
        (
            unfilled_result.label,
            unfilled_result.status,
            unfilled_result.reason in {
                TRADE_PATH_REASON_UNFILLED_DATA_END,
                "unfilled_terminated",
            },
            unfilled_result.initial_missed_buy,
            unfilled_result.realized_net_r,
        ),
    )

    excluded_update = build_trade_path_excluded_event_update(
        "teacher_params_unavailable",
        end_date="2020-01-01",
        teacher_effective_date=None,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "trade_path_reason_registry_covers_all_terminal_statuses_and_builder_exclusions",
        (11, {"PASS", "REJECT", "EXCLUDED"}, LABEL_INVALID, "EXCLUDED"),
        (
            len(TRADE_PATH_LABEL_REASON_STATUS),
            set(TRADE_PATH_LABEL_REASON_STATUS.values()),
            excluded_update["label"],
            excluded_update["label_status"],
        ),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        shard_dir = Path(tmpdir) / "shards"
        shard_path = _ticker_shard_path(shard_dir, "2330")
        _write_ticker_shard(
            shard_path,
            [
                {"_event_index": 2, "label": LABEL_PASS},
                {"_event_index": 1, "label": LABEL_REJECT},
            ],
        )
        valid_shard = _load_valid_ticker_shard(
            shard_path, expected_event_indices={1, 2}
        )
        invalid_shard = _load_valid_ticker_shard(
            shard_path, expected_event_indices={1, 2, 3}
        )
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "trade_path_resume_reuses_only_complete_ticker_shards",
            ([1, 2], True),
            (
                []
                if valid_shard is None
                else valid_shard["_event_index"].astype(int).tolist(),
                invalid_shard is None,
            ),
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_paths = resolve_filter_artifact_paths(
            tmpdir,
            TRADE_PATH_RESEARCH_FILTER_ID,
            "inception_time_v1",
            "unique_group_sampling",
        )
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "trade_path_artifact_paths_keep_filter_identity",
            (TRADE_PATH_RESEARCH_FILTER_ID, True),
            (
                artifact_paths.filter_id,
                TRADE_PATH_RESEARCH_FILTER_ID in str(artifact_paths.model_dir),
            ),
        )

    gate_base = {
        "total_return_pct": 100.0,
        "max_drawdown_pct": 10.0,
        "return_over_max_drawdown": 10.0,
        "annual_return_pct": 12.0,
        "expected_value_r": 0.50,
        "avg_exposure_pct": 90.0,
        "trade_count": 100,
    }
    gate_old = {
        **gate_base,
        "total_return_pct": 110.0,
        "expected_value_r": 0.55,
        "trade_count": 90,
    }
    gate_new = {
        **gate_base,
        "total_return_pct": 120.0,
        "expected_value_r": 0.65,
        "trade_count": 80,
    }
    gate_old_attribution = {
        "r_attribution": {"exclusive_selection_delta_r": -2.50}
    }
    gate_new_attribution = {
        "r_attribution": {"exclusive_selection_delta_r": 3.25}
    }
    gate_summary_text = _render_trade_path_gate_summary_table(
        base=gate_base,
        old_dl=gate_old,
        new_dl=gate_new,
        old_attribution=gate_old_attribution,
        new_attribution=gate_new_attribution,
    )
    gate_delta_text = _render_trade_path_gate_delta_table(
        base=gate_base,
        old_dl=gate_old,
        new_dl=gate_new,
        old_attribution=gate_old_attribution,
        new_attribution=gate_new_attribution,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "trade_path_strategy_gate_surfaces_direct_trade_selection_r",
        True,
        all(
            token in gate_summary_text + gate_delta_text
            for token in (
                "直接選擇R",
                "3.25 R",
                "-2.50 R",
                "5.75 R",
            )
        ),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        old_output_dir = Path(tmpdir) / "old"
        new_output_dir = Path(tmpdir) / "new"
        old_output_dir.mkdir()
        new_output_dir.mkdir()
        for index, filename in enumerate(BASE_IDENTITY_FILENAMES):
            payload = f"base-{index}\n".encode("utf-8")
            (old_output_dir / filename).write_bytes(payload)
            (new_output_dir / filename).write_bytes(payload)
        identity = _assert_same_base_artifacts(
            old_output_dir=old_output_dir, new_output_dir=new_output_dir
        )
        (new_output_dir / BASE_IDENTITY_FILENAMES[1]).write_text(
            "different\n", encoding="utf-8"
        )
        try:
            _assert_same_base_artifacts(
                old_output_dir=old_output_dir, new_output_dir=new_output_dir
            )
        except ValueError:
            base_difference_rejected = True
        else:
            base_difference_rejected = False
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "trade_path_strategy_gate_requires_exact_duplicate_base_artifacts",
        (3, True),
        (len(identity), base_difference_rejected),
    )

    project_root = Path(__file__).resolve().parents[2]
    app_source = (project_root / "apps" / "breakout_quality.py").read_text(encoding="utf-8")
    builder_source = (
        project_root / "tools" / "filters" / "breakout_quality" / "build_trade_path_labels.py"
    ).read_text(encoding="utf-8")
    gate_source = (
        project_root / "tools" / "filters" / "breakout_quality" / "strategy_trade_path_label_gate.py"
    ).read_text(encoding="utf-8")
    binary_pit_source = (
        project_root / "tools" / "filters" / "breakout_quality" / "build_binary_point_in_time_scores.py"
    ).read_text(encoding="utf-8")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "trade_path_menu_completes_model_artifacts_and_strategy_comparison_stays_separate",
        True,
        all(
            token in app_source
            for token in (
                "[1/Enter] 建立新Label → 重新訓練 → 模型預測報表",
                "[2] 使用既有模型 → 更新Scores → 模型預測報表",
                "[3] 查看Label與事件生命週期摘要",
                '"build-trade-path-labels"',
                "本流程不執行策略績效比較",
                "apps/strategy_compare.py",
            )
        )
        and "strategy-trade-path-label-gate" not in app_source,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "trade_path_builder_reuses_formal_lifecycle_and_never_overwrites_9a",
        True,
        all(
            token in builder_source
            for token in (
                "source_filter_id == target_filter_id",
                "TRADE_PATH_RESEARCH_FILTER_ID",
                "derived_feature_bank_trade_path_relabel",
                "initial_miss_buy_status",
                "formal_single_stock_forced_closeout",
                "same_explicit_single_stock_sizing_capital",
                "exclude_from_binary_training",
                "build_signal_cache",
                "simulate_realized_trade_path_label",
            )
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "trade_path_binary_pit_uses_filter_specific_label_policy",
        True,
        "expected_label_policy_for_filter_id(str(args.filter_id))" in binary_pit_source
        and "expected_policy=DEFAULT_LABEL_POLICY.as_manifest_payload()" not in binary_pit_source,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "trade_path_strategy_gate_compares_fixed_a2_base_old_and_new_labels",
        True,
        all(
            token in gate_source
            for token in (
                "A2 no-DL",
                "Old Label 9A",
                "New Trade-path Label",
                "New−Base",
                "New−Old",
                "ALL_RULE_FILTERS_OFF_OVERRIDES",
                "TRADE_PATH_FORWARD_TEACHER_PARAMS_RELATIVE_PATH",
                "只更新forward-OOS Scores並執行策略回放，不重新訓練",
                "exclusive_selection_delta_r",
                "base_artifact_sha256",
                "no_filter_trades.csv",
            )
        ),
    )

    summary["label_id"] = TRADE_PATH_LABEL_ID
    summary["research_filter_id"] = TRADE_PATH_RESEARCH_FILTER_ID
    summary["menu_scope"] = "model_prediction_only"
    summary["strategy_scope"] = "cli_only"
    return results, summary

def validate_breakout_quality_single_seed_single_entry_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_SINGLE_SEED_SINGLE_ENTRY"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    project_root = Path(__file__).resolve().parents[2]
    canonical_config_path = project_root / "config" / "breakout_quality.py"
    canonical_source = canonical_config_path.read_text(encoding="utf-8")
    canonical_app_path = project_root / "apps" / "breakout_quality.py"
    canonical_app_source = canonical_app_path.read_text(encoding="utf-8")
    strategy_app_path = project_root / "apps" / "strategy_compare.py"
    strategy_config_path = project_root / "config" / "strategy_compare.py"

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "single_seed_contract_has_one_editable_setting_and_no_workflow_seed",
        (1, False),
        (
            canonical_source.count("BREAKOUT_QUALITY_RANDOM_SEED ="),
            "BREAKOUT_QUALITY_WORKFLOW_RANDOM_SEED" in canonical_source,
        ),
    )

    from config import breakout_quality as breakout_quality_config

    configured_seed = breakout_quality_config.resolve_breakout_quality_random_seed()
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "single_seed_contract_resolves_nonnegative_integer",
        True,
        isinstance(configured_seed, int) and configured_seed >= 0,
    )

    with patch.object(breakout_quality_config, "BREAKOUT_QUALITY_RANDOM_SEED", 7):
        overridden_seed = breakout_quality_config.resolve_breakout_quality_random_seed()
        with patch.object(
            breakout_quality_config,
            "BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE",
            UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE,
        ):
            binary_seed = breakout_quality_config.get_breakout_quality_workflow_settings().seed
        with patch.object(
            breakout_quality_config,
            "BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE",
            STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE,
        ):
            continuous_seed = breakout_quality_config.get_breakout_quality_workflow_settings().seed
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "single_seed_contract_applies_override_to_all_profiles",
        (7, 7, 7),
        (overridden_seed, binary_seed, continuous_seed),
    )

    with patch.object(breakout_quality_config, "BREAKOUT_QUALITY_RANDOM_SEED", -1):
        try:
            breakout_quality_config.resolve_breakout_quality_random_seed()
        except ValueError:
            negative_seed_rejected = True
        else:
            negative_seed_rejected = False
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "single_seed_contract_rejects_negative_seed",
        True,
        negative_seed_rejected,
    )

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "model_and_strategy_apps_are_separate_entries",
        True,
        canonical_app_path.is_file()
        and strategy_app_path.is_file()
        and strategy_config_path.is_file()
        and '"strategy-compare"' not in canonical_app_source
        and "apps/strategy_compare.py" in canonical_app_source,
    )

    strategy_compare_source = (
        project_root
        / "filters"
        / "breakout_quality"
        / "strategy_compare_engine.py"
    ).read_text(encoding="utf-8")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "single_seed_contract_strategy_gate_rejects_seed_mismatch",
        True,
        all(
            token in strategy_compare_source
            for token in (
                "int(pit_contract.seed) != int(workflow_settings.seed)",
                "Selection PIT工件seed與目前workflow不一致",
            )
        ),
    )

    summary["seed_source"] = "config.breakout_quality.BREAKOUT_QUALITY_RANDOM_SEED"
    summary["strategy_compare_entry"] = "apps/strategy_compare.py"
    return results, summary


def validate_strategy_compare_config_driven_app_contract_case(_base_params):
    case_id = "STRATEGY_COMPARE_CONFIG_DRIVEN_APP"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "config" / "strategy_compare.py"
    app_path = project_root / "apps" / "strategy_compare.py"
    model_app_path = project_root / "apps" / "breakout_quality.py"
    orchestration_path = project_root / "filters" / "breakout_quality" / "strategy_comparison.py"
    preparation_path = project_root / "filters" / "breakout_quality" / "strategy_compare_preparation.py"
    export_service_path = project_root / "filters" / "breakout_quality" / "export_scores.py"
    param_service_path = project_root / "filters" / "breakout_quality" / "strategy_param_training.py"
    workflow_io_path = project_root / "filters" / "breakout_quality" / "workflow_io.py"
    optimizer_policy_path = project_root / "filters" / "breakout_quality" / "strategy_optimizer_policy.py"
    legacy_export_path = project_root / "tools" / "filters" / "breakout_quality" / "export_scores.py"
    legacy_param_path = project_root / "tools" / "filters" / "breakout_quality" / "strategy_dl_filter_param_adapt_gate.py"
    quick_gate_source = (project_root / "tools" / "local_regression" / "run_quick_gate.py").read_text(encoding="utf-8")

    config_source = config_path.read_text(encoding="utf-8")
    app_source = app_path.read_text(encoding="utf-8")
    model_app_source = model_app_path.read_text(encoding="utf-8")
    orchestration_source = orchestration_path.read_text(encoding="utf-8")
    preparation_source = preparation_path.read_text(encoding="utf-8")

    from config import strategy_compare as strategy_config
    from core.strategy_comparison import strategy_comparison_fingerprint

    settings = strategy_config.get_strategy_comparison_settings()
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "comparison_config_lists_individual_arms_contrasts_and_preparation_without_active_id",
        True,
        "ACTIVE_STRATEGY_COMPARISON_ID" not in config_source
        and len(settings.arms) >= 2
        and all(isinstance(arm.enabled, bool) for arm in settings.arms.values())
        and all(isinstance(item.enabled, bool) for item in settings.contrasts.values())
        and isinstance(settings.preparation.auto_prepare, bool)
        and any(source.builder is not None for source in settings.parameter_sources.values())
        and any(source.forward_scores_builder is not None for source in settings.dl_sources.values()),
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "strategy_app_menu_is_generic_and_model_app_has_no_strategy_route",
        True,
        all(token in app_source for token in (
            "執行目前比較設定", "查看設定、工件與預計動作", "config/strategy_compare.py",
        ))
        and "C1" not in app_source and "TP1" not in app_source
        and '"strategy-compare"' not in model_app_source
        and "策略績效驗證" not in model_app_source,
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "strategy_compare_official_entry_is_in_quick_gate_help_registry",
        True,
        '([sys.executable, "apps/strategy_compare.py", "--help"]' in quick_gate_source
        and '"apps/strategy_compare.py",' in quick_gate_source.split("INLINE_CLI_TARGETS = {", 1)[1],
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "deterministic_prerequisites_use_formal_shared_services_without_model_training",
        True,
        all(path.is_file() for path in (
            preparation_path, export_service_path, param_service_path, workflow_io_path, optimizer_policy_path,
        ))
        and "prepare_strategy_comparison_artifacts" in orchestration_source
        and "export_forward_oos_scores" in preparation_source
        and "prepare_strategy_parameter_source" in preparation_source
        and all(token not in preparation_source for token in (
            "build_trade_path_labels", "train_model", "build_dataset",
        ))
        and not legacy_export_path.exists()
        and not legacy_param_path.exists()
        and '"export-scores": "filters.breakout_quality.export_scores"' in model_app_source,
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "preparation_plan_has_ready_preparable_blocked_and_single_confirmation_contract",
        True,
        all(token in preparation_source for token in (
            'overall_status = "BLOCKED"', 'overall_status = "PREPARABLE"', 'overall_status = "READY"',
            "前置步驟失敗:",
        ))
        and "按 Enter 執行；輸入 0 返回" in app_source
        and "render_execution_plan" in app_source,
    )

    from filters.breakout_quality import strategy_comparison as strategy_comparison_module

    requested_plan = SimpleNamespace(overall_status="PREPARABLE")
    complete_post_prepare_status = {
        "comparison_ready": True,
        "overall_status": "READY",
        "config_fingerprint": "postprepare123",
        "artifact_identities": {"param:test": {"sha256": "abc"}},
        "resolved_parameter_paths": {"test": "models/test.json"},
        "preparation_plan": SimpleNamespace(overall_status="READY"),
        "comparison_period": {"start": "2021-01-01", "end": "2026-03-02"},
        "comparison_period_source": "dl_runtime_common_overlap",
    }
    with patch.object(
        strategy_comparison_module,
        "collect_artifact_status",
        return_value=dict(complete_post_prepare_status),
    ):
        refreshed_post_prepare_status = (
            strategy_comparison_module._collect_ready_status_after_preparation(
                root=project_root,
                settings=settings,
                requested_plan=requested_plan,
            )
        )
    with patch.object(
        strategy_comparison_module,
        "collect_artifact_status",
        return_value={
            key: value
            for key, value in complete_post_prepare_status.items()
            if key != "config_fingerprint"
        },
    ):
        try:
            strategy_comparison_module._collect_ready_status_after_preparation(
                root=project_root,
                settings=settings,
                requested_plan=requested_plan,
            )
        except RuntimeError as exc:
            incomplete_post_prepare_rejected = "config_fingerprint" in str(exc)
        else:
            incomplete_post_prepare_rejected = False
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "post_preparation_refresh_restores_full_orchestration_status_contract",
        True,
        refreshed_post_prepare_status["config_fingerprint"] == "postprepare123"
        and refreshed_post_prepare_status["requested_preparation_plan"] is requested_plan
        and incomplete_post_prepare_rejected,
    )

    execution_pairs = strategy_comparison_module._execution_pairs(settings)
    expected_pairs_by_group: dict[
        tuple[str, str],
        dict[str, object],
    ] = {}
    expected_group_order: list[tuple[str, str]] = []
    for arm in settings.enabled_arms:
        key = (arm.param_source, arm.rule_policy)
        if key not in expected_pairs_by_group:
            expected_pairs_by_group[key] = {"off": None, "on": []}
            expected_group_order.append(key)
        group = expected_pairs_by_group[key]
        if arm.dl_enabled:
            group["on"].append(arm)
        else:
            group["off"] = arm
    expected_execution_pairs = tuple(
        (
            param_source,
            rule_policy,
            group["off"].arm_id,
            on_arm.arm_id,
        )
        for param_source, rule_policy in expected_group_order
        for group in (expected_pairs_by_group[(param_source, rule_policy)],)
        if group["off"] is not None
        for on_arm in group["on"]
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "enabled_arms_build_canonical_execution_pairs_before_replay",
        expected_execution_pairs,
        tuple(
            (param_source, rule_policy, off_arm.arm_id, on_arm.arm_id)
            for param_source, rule_policy, off_arm, on_arm in execution_pairs
        ),
    )

    a9_source = settings.dl_sources.get("A9")
    a9_param_source = settings.parameter_sources.get("min_dl_a9_roos")
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "a9_has_matched_runtime_and_separate_min_a9_parameter_source",
        True,
        a9_source is not None
        and a9_source.filter_id == "breakout_quality_v1"
        and a9_source.model_architecture == "inception_time_v1"
        and a9_source.experiment_profile == "unique_group_sampling"
        and a9_source.forward_scores_builder is not None
        and a9_param_source is not None
        and a9_param_source.trained_with_dl_id == "A9"
        and a9_param_source.builder is not None
        and a9_param_source.builder.options.get("p3_variant") == "A9"
        and "p3_dl_on_trained/A9" in str(a9_param_source.path_template)
        and {"C7", "C8", "C9", "C10", "C11", "C12", "C14", "C15"}.issubset(set(settings.arms))
        and all(arm.enabled for arm in settings.enabled_arms)
        and all(arm.arm_id in settings.arms for arm in settings.enabled_arms)
        and settings.dl_sources["CONT11G"].score_source == "continuous_ranker_oos"
        and settings.dl_sources["CONT11G"].threshold is None
        and settings.dl_sources["CONT11G"].experiment_profile == "strategy_aligned_no_time_pass_magnitude_mse"
        and settings.dl_sources["CONT12A"].score_source == "continuous_ranker_oos"
        and settings.dl_sources["CONT12A"].threshold is None
        and settings.dl_sources["CONT12A"].experiment_profile == "strategy_aligned_no_time_all_event_mse",
    )

    from dataclasses import replace as _replace_strategy_arm
    mismatched_arms = dict(settings.arms)
    mismatched_arms["C10"] = _replace_strategy_arm(
        mismatched_arms["C10"], param_source="min_dl_tp1_roos", dl_id="A9"
    )
    try:
        from core.strategy_comparison import validate_strategy_comparison_settings
        validate_strategy_comparison_settings(replace(settings, arms=mismatched_arms))
    except ValueError as exc:
        mismatched_dl_aware_runtime_rejected = "訓練時相同的DL runtime" in str(exc)
    else:
        mismatched_dl_aware_runtime_rejected = False
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "dl_aware_parameters_reject_cross_version_runtime_pairing",
        True,
        mismatched_dl_aware_runtime_rejected,
    )

    expected_display_names = {
        "C3": "Min ROOS",
        "C4": "Min ROOS: TP1-on",
        "C8": "Min ROOS: A9-on",
        "C9": "Min-A9 ROOS",
        "C10": "Min-A9 ROOS: DL-on",
        "C5": "Min-TP1 ROOS",
        "C6": "Min-TP1 ROOS: DL-on",
        "C11": "Min ROOS: A9 resource-aware",
        "C12": "Min ROOS: A9 resource-aware basket",
        "C14": "Min ROOS: Continuous resource-aware",
        "C15": "Min ROOS: All-event Continuous resource-aware",
    }
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "min_roos_display_names_follow_training_identity_then_runtime_suffix_contract",
        True,
        all(settings.arms[arm_id].name == name for arm_id, name in expected_display_names.items()),
    )

    from core.exact_accounting import build_buy_ledger_from_price
    from core.portfolio_entries import (
        _simulate_reserved_candidate_order,
        reorder_candidates_for_resource_aware_binary,
    )
    from core.strategy_params import V16StrategyParams
    from core.trade_plans import build_normal_candidate_plan

    resource_params = V16StrategyParams()
    resource_params.use_breakout_quality_ranking = True
    resource_params.breakout_quality_score_threshold = 0.5

    def _resource_candidate(ticker, score):
        plan = build_normal_candidate_plan(100.0, 5.0, 1_000_000.0, resource_params, ticker=ticker)
        qty = int(plan["qty"])
        cost_milli = build_buy_ledger_from_price(100.0, qty, resource_params)["net_buy_total_milli"]
        return {
            "ticker": ticker,
            "type": "normal",
            "limit_px": 100.0,
            "init_sl": plan["init_sl"],
            "init_trail": plan["init_trail"],
            "target_price": plan.get("target_price"),
            "entry_atr": 5.0,
            "qty": qty,
            "proj_cost_milli": cost_milli,
            "proj_cost": cost_milli / 1000.0,
            "is_orderable": True,
            "params_obj": resource_params,
            "sizing_capital": 1_000_000.0,
            "use_breakout_quality_ranking": True,
            "breakout_quality_ranking_policy": "resource-aware-binary",
            "breakout_quality_score": score,
            "breakout_quality_rank": {"available": True, "score": score},
        }

    resource_rows = [
        _resource_candidate("R1", 0.20),
        _resource_candidate("R2", 0.30),
        _resource_candidate("P1", 0.90),
    ]
    resource_baseline = _simulate_reserved_candidate_order(
        resource_rows,
        available_cash=180_000.0,
        sizing_equity=1_000_000.0,
        free_slots=3,
        params=resource_params,
    )
    resource_order, resource_diag = reorder_candidates_for_resource_aware_binary(
        resource_rows,
        available_cash=180_000.0,
        sizing_equity=1_000_000.0,
        pre_market_occupied=7,
        max_positions=10,
        params=resource_params,
    )
    resource_selected = _simulate_reserved_candidate_order(
        resource_order,
        available_cash=180_000.0,
        sizing_equity=1_000_000.0,
        free_slots=3,
        params=resource_params,
    )
    slot_order, slot_diag = reorder_candidates_for_resource_aware_binary(
        resource_rows,
        available_cash=180_000.0,
        sizing_equity=1_000_000.0,
        pre_market_occupied=9,
        max_positions=10,
        params=resource_params,
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "resource_aware_a9_only_intervenes_when_cash_is_binding_and_increases_pass_reserved_capital",
        True,
        resource_diag["mode"] == "dl-selection"
        and resource_diag["changed"]
        and resource_baseline["cash_is_binding"]
        and resource_selected["cash_is_binding"]
        and resource_selected["pass_reserved_cost_milli"] > resource_baseline["pass_reserved_cost_milli"]
        and resource_order[0]["ticker"] == "P1"
        and slot_diag["mode"] == "capital-utilization"
        and not slot_diag["changed"]
        and [row["ticker"] for row in slot_order] == ["R1", "R2", "P1"],
    )


    def _resource_candidate_fixed(ticker, price, qty, score, policy):
        cost_milli = build_buy_ledger_from_price(price, qty, resource_params)["net_buy_total_milli"]
        return {
            "ticker": ticker,
            "type": "normal",
            "limit_px": price,
            "init_sl": price * 0.95,
            "init_trail": price * 0.95,
            "target_price": price * 1.10,
            "entry_atr": price * 0.05,
            "qty": qty,
            "max_qty": qty,
            "proj_cost_milli": cost_milli,
            "proj_cost": cost_milli / 1000.0,
            "is_orderable": True,
            "params_obj": resource_params,
            "sizing_capital": 2_000_000.0,
            "use_breakout_quality_ranking": True,
            "breakout_quality_ranking_policy": policy,
            "breakout_quality_score": score,
            "breakout_quality_rank": {"available": True, "score": score},
        }

    basket_seed = (
        ("R1", 300.0, 293, 0.20),
        ("R2", 1000.0, 281, 0.30),
        ("P1", 1000.0, 147, 0.70),
        ("P2", 300.0, 289, 0.71),
        ("P3", 300.0, 239, 0.72),
        ("P4", 1000.0, 351, 0.73),
        ("P5", 300.0, 396, 0.74),
        ("P6", 1000.0, 176, 0.75),
    )
    greedy_rows = [
        _resource_candidate_fixed(*row, "resource-aware-binary")
        for row in basket_seed
    ]
    basket_rows = [
        _resource_candidate_fixed(*row, "resource-aware-binary-basket")
        for row in basket_seed
    ]
    greedy_order, greedy_diag = reorder_candidates_for_resource_aware_binary(
        greedy_rows,
        available_cash=548_426.0,
        sizing_equity=2_000_000.0,
        pre_market_occupied=6,
        max_positions=10,
        params=resource_params,
    )
    basket_order, basket_diag = reorder_candidates_for_resource_aware_binary(
        basket_rows,
        available_cash=548_426.0,
        sizing_equity=2_000_000.0,
        pre_market_occupied=6,
        max_positions=10,
        params=resource_params,
    )
    greedy_selected = _simulate_reserved_candidate_order(
        greedy_order,
        available_cash=548_426.0,
        sizing_equity=2_000_000.0,
        free_slots=4,
        params=resource_params,
    )
    basket_selected = _simulate_reserved_candidate_order(
        basket_order,
        available_cash=548_426.0,
        sizing_equity=2_000_000.0,
        free_slots=4,
        params=resource_params,
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "resource_aware_best_improvement_evaluates_all_current_promotions_and_beats_first_improvement_case",
        True,
        greedy_diag["selector"] == "greedy-first-improvement"
        and basket_diag["selector"] == "best-improvement-basket"
        and greedy_diag["mode"] == basket_diag["mode"] == "dl-selection"
        and basket_diag["basket_search_states"] > 0
        and basket_selected["cash_is_binding"]
        and basket_selected["pass_reserved_cost_milli"] > greedy_selected["pass_reserved_cost_milli"]
        and [row["ticker"] for row in greedy_selected["selected_rows"]] == ["P1", "P2", "P4"]
        and [row["ticker"] for row in basket_selected["selected_rows"]] == ["P2", "P4", "P5"],
    )

    same_param_direct_delta = strategy_comparison_module._same_param_direct_selection_delta(
        {
            "param_source": "min_roos",
            "rule_policy": "all_off",
            "direct_selection_delta_r": -6.12,
        },
        {
            "param_source": "min_roos",
            "rule_policy": "all_off",
            "direct_selection_delta_r": 0.0,
        },
    )
    cross_param_direct_delta = strategy_comparison_module._same_param_direct_selection_delta(
        {
            "param_source": "min_dl_a9_roos",
            "rule_policy": "all_off",
            "direct_selection_delta_r": -166.38,
        },
        {
            "param_source": "min_roos",
            "rule_policy": "all_off",
            "direct_selection_delta_r": -6.12,
        },
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "direct_selection_r_is_only_comparable_within_same_parameter_runtime_universe",
        (-6.12, None),
        (same_param_direct_delta, cross_param_direct_delta),
    )


    from core.strategy_comparison import StrategyPreparationAction, StrategyPreparationPlan

    continuous_presort_rows = [
        {
            "ticker": "M1",
            "sort_value": 0.10,
            "proj_cost": 100_000.0,
            "use_breakout_quality_ranking": True,
            "breakout_quality_ranking_policy": "resource-aware-continuous",
            "breakout_quality_score": 0.10,
            "breakout_quality_rank": {"available": True, "score": 0.10},
        },
        {
            "ticker": "M2",
            "sort_value": 0.20,
            "proj_cost": 200_000.0,
            "use_breakout_quality_ranking": True,
            "breakout_quality_ranking_policy": "resource-aware-continuous",
            "breakout_quality_score": 0.95,
            "breakout_quality_rank": {"available": True, "score": 0.95},
        },
    ]
    continuous_presort = sort_candidate_rows(
        continuous_presort_rows, method=BUY_LIMIT_OVERAGE_SORT_METHOD
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "resource_aware_continuous_presort_preserves_min_roos_before_resource_gate",
        ["M1", "M2"],
        [row["ticker"] for row in continuous_presort],
    )

    continuous_rows = [
        _resource_candidate_fixed("R1", 1000.0, 110, 0.10, "resource-aware-continuous"),
        _resource_candidate_fixed("R2", 1000.0, 70, 0.20, "resource-aware-continuous"),
        _resource_candidate_fixed("Q1", 1000.0, 80, 0.95, "resource-aware-continuous"),
    ]
    continuous_order, continuous_diag = reorder_candidates_for_resource_aware_binary(
        continuous_rows,
        available_cash=180_000.0,
        sizing_equity=2_000_000.0,
        pre_market_occupied=7,
        max_positions=10,
        params=resource_params,
    )
    continuous_selected = _simulate_reserved_candidate_order(
        continuous_order,
        available_cash=180_000.0,
        sizing_equity=2_000_000.0,
        free_slots=3,
        params=resource_params,
    )
    continuous_slot_order, continuous_slot_diag = reorder_candidates_for_resource_aware_binary(
        continuous_rows,
        available_cash=180_000.0,
        sizing_equity=2_000_000.0,
        pre_market_occupied=9,
        max_positions=10,
        params=resource_params,
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "resource_aware_continuous_only_ranks_on_cash_binding_days_and_preserves_capital_utilization_first",
        True,
        continuous_diag["mode"] == "dl-selection"
        and continuous_selected["cash_is_binding"]
        and continuous_diag["selected_score_sum"] >= continuous_diag["baseline_selected_score_sum"]
        and continuous_diag["promoted_pass_count"] == 0
        and continuous_slot_diag["mode"] == "capital-utilization"
        and not continuous_slot_diag["changed"]
        and [row["ticker"] for row in continuous_slot_order] == ["R1", "R2", "Q1"],
    )

    mocked_pair_payload = {
        "metadata": {
            "comparison_period": {"start": "2021-01-01", "end": "2021-12-31"}
        },
        "no_filter": {
            "total_return_pct": 10.0,
            "max_drawdown_pct": 5.0,
            "return_over_max_drawdown": 2.0,
            "annual_return_pct": 10.0,
            "expected_value_r": 0.10,
            "payoff_ratio": 1.20,
            "avg_exposure_pct": 50.0,
            "trade_count": 10,
        },
        "quality_filter": {
            "total_return_pct": 11.0,
            "max_drawdown_pct": 5.0,
            "return_over_max_drawdown": 2.2,
            "annual_return_pct": 11.0,
            "expected_value_r": 0.12,
            "payoff_ratio": 1.25,
            "avg_exposure_pct": 48.0,
            "trade_count": 8,
        },
        "score_ranking": {
            "total_return_pct": 12.0,
            "max_drawdown_pct": 5.0,
            "return_over_max_drawdown": 2.4,
            "annual_return_pct": 12.0,
            "expected_value_r": 0.13,
            "payoff_ratio": 1.30,
            "avg_exposure_pct": 50.0,
            "trade_count": 10,
            "resource_aware_dl_selection_days": 3,
            "resource_aware_capital_utilization_days": 7,
            "resource_aware_changed_days": 2,
            "resource_aware_promoted_pass_orders": 2,
            "resource_aware_pass_reserved_gain_milli": 1000000,
            "resource_aware_reserved_delta_milli": -500000,
        },
        "yearly": [
            {
                "year": 2021,
                "no_filter_return_pct": 10.0,
                "quality_filter_return_pct": 11.0,
            }
        ],
    }
    ready_plan = StrategyPreparationPlan(overall_status="READY", actions=tuple())
    ready_status = {
        "overall_status": "READY",
        "comparison_ready": True,
        "config_fingerprint": "runtimepair123",
        "artifact_identities": {},
        "resolved_parameter_paths": {
            source_id: f"models/{source_id}.json"
            for source_id in {arm.param_source for arm in settings.enabled_arms}
        },
        "preparation_plan": ready_plan,
        "comparison_period": {"start": "2021-01-01", "end": "2021-12-31"},
        "comparison_period_source": "dl_runtime_common_overlap",
    }
    with tempfile.TemporaryDirectory() as tmpdir, patch.object(
        strategy_comparison_module,
        "get_strategy_comparison_settings",
        return_value=settings,
    ), patch.object(
        strategy_comparison_module,
        "run_comparison",
        return_value=dict(mocked_pair_payload),
    ) as mocked_run, patch.object(
        strategy_comparison_module,
        "_load_direct_selection_r",
        return_value=0.25,
    ), redirect_stdout(io.StringIO()):
        replay_payload = strategy_comparison_module.run_strategy_comparison(
            project_root=Path(tmpdir),
            quiet=True,
            status=dict(ready_status),
            auto_prepare=False,
        )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "ready_orchestration_executes_each_config_pair_and_writes_all_enabled_arms",
        True,
        mocked_run.call_count == len(execution_pairs)
        and all(
            call.kwargs.get("comparison_start_date") == "2021-01-01"
            and call.kwargs.get("comparison_end_date") == "2021-12-31"
            for call in mocked_run.call_args_list
        )
        and set(replay_payload["scenarios"])
        == {arm.arm_id for arm in settings.enabled_arms}
        and replay_payload["status"] == "COMPLETED",
    )

    min_off = settings.arms["C3"]
    min_a9_hard = settings.arms["C8"]
    min_a9_resource = settings.arms["C11"]
    mismatched_pairs = {
        "min__a9_hard": {
            "arm_contract": ("min_roos", "all_off", min_off, min_a9_hard),
            "payload": dict(mocked_pair_payload),
        },
        "min__a9_resource": {
            "arm_contract": ("min_roos", "all_off", min_off, min_a9_resource),
            "payload": {
                **dict(mocked_pair_payload),
                "no_filter": {
                    **dict(mocked_pair_payload["no_filter"]),
                    "total_return_pct": 9.0,
                },
            },
        },
    }
    try:
        strategy_comparison_module._scenario_payloads(
            mismatched_pairs,
            {"min__a9_hard": 0.1, "min__a9_resource": 0.2},
            settings=settings,
        )
    except ValueError as exc:
        repeated_baseline_mismatch_rejected = "共用基準不一致" in str(exc)
    else:
        repeated_baseline_mismatch_rejected = False
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "multiple_dl_sources_require_identical_replayed_shared_baseline",
        True,
        repeated_baseline_mismatch_rejected,
    )


    min_roos_source = settings.parameter_sources["min_roos"]
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "min_roos_uses_forward_p2_artifact_and_has_auto_builder",
        True,
        "binary_dl_filter_param_adaptation/risk_only_rolling/p2_dl_off_trained"
        in str(min_roos_source.path_template)
        and "trade_path_label/a2_teacher_params" not in str(min_roos_source.path_template)
        and min_roos_source.identity_manifest_path is not None
        and min_roos_source.builder is not None
        and str(min_roos_source.builder.options.get("parameter_set")) == "p2",
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "parameter_coverage_is_preflighted_before_any_pair_replay",
        True,
        "PARAM_PERIOD_MISMATCH" in preparation_source
        and "comparison_period" in preparation_source
        and "comparison_start_date=comparison_start" in orchestration_source
        and "comparison_end_date=comparison_end" in orchestration_source,
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "preparation_supports_dependency_waves_after_score_period_becomes_known",
        True,
        "max_waves" in preparation_source
        and "前置工件重新規劃後沒有進展" in preparation_source,
    )

    from filters.breakout_quality import strategy_compare_preparation as preparation_module
    score_action = StrategyPreparationAction(
        action_id="dl:TP1:forward_scores",
        artifact_key="dl:TP1:forward_scores",
        action="BUILD",
        builder_type="forward_oos_scores",
        description="build scores",
        path="models/scores.csv",
    )
    p2_action = StrategyPreparationAction(
        action_id="param:min_roos",
        artifact_key="param:min_roos",
        action="BUILD",
        builder_type="binary_dl_risk_only_rolling",
        description="build p2",
        path="models/p2.json",
    )
    initial_wave_status = {
        "comparison_ready": False,
        "overall_status": "PREPARABLE",
        "preparation_plan": StrategyPreparationPlan(
            overall_status="PREPARABLE", actions=(score_action,)
        ),
    }
    second_wave_status = {
        "comparison_ready": False,
        "overall_status": "PREPARABLE",
        "preparation_plan": StrategyPreparationPlan(
            overall_status="PREPARABLE", actions=(p2_action,)
        ),
    }
    final_wave_status = {
        "comparison_ready": True,
        "overall_status": "READY",
        "preparation_plan": StrategyPreparationPlan(
            overall_status="READY", actions=tuple()
        ),
    }
    with patch.object(
        preparation_module,
        "collect_artifact_status",
        side_effect=(second_wave_status, final_wave_status),
    ), patch.object(
        preparation_module,
        "_execute_preparation_action",
    ) as mocked_prepare_action, redirect_stdout(io.StringIO()):
        multi_wave_result = preparation_module.prepare_strategy_comparison_artifacts(
            project_root=project_root,
            settings=settings,
            status=initial_wave_status,
        )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "preparation_executes_newly_revealed_dependencies_in_later_wave",
        True,
        multi_wave_result["comparison_ready"]
        and mocked_prepare_action.call_count == 2
        and [
            call.kwargs["action"].artifact_key
            for call in mocked_prepare_action.call_args_list
        ] == ["dl:TP1:forward_scores", "param:min_roos"],
    )

    add_check(
        results, "synthetic_breakout_quality", case_id,
        "parameter_preflight_identity_tracks_config_and_baseline",
        True,
        all(token in preparation_source for token in (
            "TRAINING_CONFIG_MISMATCH", "BASELINE_PARAMS_IDENTITY_MISMATCH",
            "BINARY_PIT_IDENTITY_MISSING", "trials_per_fold", "max_position_cap_pct",
        )),
    )

    enabled_dl_arm_ids = [
        arm.arm_id for arm in settings.enabled_arms if arm.dl_enabled
    ]
    switch_target_arm_id = enabled_dl_arm_ids[0] if len(enabled_dl_arm_ids) >= 2 else None
    original_switch_target = (
        dict(strategy_config.STRATEGY_COMPARE_ARMS[switch_target_arm_id])
        if switch_target_arm_id is not None
        else None
    )
    original_contrasts = {
        key: dict(value)
        for key, value in strategy_config.STRATEGY_COMPARE_CONTRASTS.items()
    }
    reduced_settings = settings
    if switch_target_arm_id is not None:
        try:
            strategy_config.STRATEGY_COMPARE_ARMS[switch_target_arm_id]["enabled"] = False
            for contrast in strategy_config.STRATEGY_COMPARE_CONTRASTS.values():
                if (
                    contrast.get("left") == switch_target_arm_id
                    or contrast.get("right") == switch_target_arm_id
                ):
                    contrast["enabled"] = False
            reduced_settings = strategy_config.get_strategy_comparison_settings()
        finally:
            strategy_config.STRATEGY_COMPARE_ARMS[switch_target_arm_id].clear()
            strategy_config.STRATEGY_COMPARE_ARMS[switch_target_arm_id].update(
                original_switch_target
            )
            strategy_config.STRATEGY_COMPARE_CONTRASTS.clear()
            strategy_config.STRATEGY_COMPARE_CONTRASTS.update(original_contrasts)
    expected_reduced_arm_ids = tuple(
        arm.arm_id
        for arm in settings.enabled_arms
        if arm.arm_id != switch_target_arm_id
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "individual_dl_arm_and_contrast_switches_are_runtime_effective",
        expected_reduced_arm_ids,
        tuple(arm.arm_id for arm in reduced_settings.enabled_arms),
        note=(
            "SKIP: fewer than two enabled DL arms"
            if switch_target_arm_id is None
            else ""
        ),
    )

    reduced_enabled_ids = {
        arm.arm_id for arm in reduced_settings.enabled_arms
    }
    disabled_definition_arm_id = next(
        (
            arm_id
            for arm_id in reduced_settings.arms
            if arm_id not in reduced_enabled_ids
        ),
        None,
    )
    enabled_definition_arm_id = next(
        (arm.arm_id for arm in reduced_settings.enabled_arms),
        None,
    )
    disabled_arms_changed = dict(reduced_settings.arms)
    if disabled_definition_arm_id is not None:
        disabled_arms_changed[disabled_definition_arm_id] = replace(
            disabled_arms_changed[disabled_definition_arm_id],
            name="disabled definition",
        )
    enabled_arms_changed = dict(reduced_settings.arms)
    if enabled_definition_arm_id is not None:
        enabled_arms_changed[enabled_definition_arm_id] = replace(
            enabled_arms_changed[enabled_definition_arm_id],
            name="enabled definition",
        )
    base_fingerprint = strategy_comparison_fingerprint(reduced_settings)
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "config_fingerprint_uses_effective_enabled_definitions_only",
        True,
        disabled_definition_arm_id is not None
        and enabled_definition_arm_id is not None
        and strategy_comparison_fingerprint(
            replace(reduced_settings, arms=disabled_arms_changed)
        ) == base_fingerprint
        and strategy_comparison_fingerprint(
            replace(reduced_settings, arms=enabled_arms_changed)
        ) != base_fingerprint,
    )

    summary["config_path"] = "config/strategy_compare.py"
    summary["app_path"] = "apps/strategy_compare.py"
    summary["enabled_arms"] = [arm.arm_id for arm in settings.enabled_arms]
    summary["preparation"] = settings.preparation.as_dict()
    return results, summary



def validate_breakout_quality_audit_framework_contract_case(_base_params):
    from config.audit import (
        AUDIT_OUTPUT_ROOT,
        get_audit_definitions,
        get_audit_module_ids,
        get_enabled_audit_definitions,
        validate_audit_config,
    )
    from tools.audit.breakout_quality.pass_persistence import run_pass_persistence_audit
    from tools.audit.breakout_quality.pass_quality import run_pass_quality_audit
    from tools.audit.breakout_quality.selection_confidence import run_selection_confidence_audit
    from tools.audit.breakout_quality.c15_strategy_attribution import (
        collect_strategy_attribution_status,
        run_strategy_attribution_audit,
    )
    from tools.audit.catalog import validate_audit_catalog

    case_id = "AUDIT_FRAMEWORK"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    validate_audit_config()
    all_definitions = get_audit_definitions("breakout_quality")
    enabled_definitions = get_enabled_audit_definitions("breakout_quality")
    quality_definition = next(
        (item for item in all_definitions if item.audit_type == "pass_quality"), None
    )
    persistence_definition = next(
        (item for item in all_definitions if item.audit_type == "pass_persistence"), None
    )
    confidence_definition = next(
        (item for item in all_definitions if item.audit_type == "selection_confidence"), None
    )
    strategy_attribution_definition = next(
        (item for item in all_definitions if item.audit_type == "strategy_attribution"), None
    )
    validate_audit_catalog(all_definitions)
    project_root = Path(__file__).resolve().parents[2]
    audit_app_source = (project_root / "apps" / "audit.py").read_text(encoding="utf-8")
    model_app_source = (project_root / "apps" / "breakout_quality.py").read_text(encoding="utf-8")
    quick_gate_source = (project_root / "tools" / "local_regression" / "run_quick_gate.py").read_text(encoding="utf-8")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "project_audit_app_and_breakout_quality_facade_share_catalog_runner_and_cli_smoke_registry",
        True,
        "from tools.audit.runner import" in audit_app_source
        and "from tools.audit.runner import" in model_app_source
        and "get_domain_cli_commands" in model_app_source
        and '([sys.executable, "apps/audit.py", "--help"]' in quick_gate_source
        and '"apps/audit.py",' in quick_gate_source.split("INLINE_CLI_TARGETS = {", 1)[1],
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "audit_config_and_catalog_support_project_modules_and_config_driven_handlers",
        True,
        quality_definition is not None
        and persistence_definition is not None
        and confidence_definition is not None
        and strategy_attribution_definition is not None
        and "breakout_quality" in get_audit_module_ids(enabled_only=True)
        and bool(enabled_definitions)
        and all(bool(str(item.source.get("kind") or "").strip()) for item in all_definitions),
    )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run_dir = root / "outputs" / "strategy_compare" / "runs" / "synthetic"
        latest_dir = root / "outputs" / "strategy_compare" / "latest"
        pair_dir = (
            run_dir
            / "pairs"
            / "min_roos__all_off__A9__resource_aware_binary_basket"
        )
        events_dir = (
            root
            / "outputs"
            / "filters"
            / "breakout_quality"
            / "breakout_quality_v1"
        )
        pair_dir.mkdir(parents=True, exist_ok=True)
        latest_dir.mkdir(parents=True, exist_ok=True)
        events_dir.mkdir(parents=True, exist_ok=True)

        result_payload = {
            "status": "COMPLETED",
            "settings": {
                "arms": {
                    "C12": {
                        "arm_id": "C12",
                        "enabled": True,
                        "param_source": "min_roos",
                        "rule_policy": "all_off",
                        "dl_enabled": True,
                        "dl_id": "A9",
                        "dl_runtime_mode": "resource-aware-binary-basket",
                    }
                },
                "dl_sources": {
                    "A9": {
                        "dl_id": "A9",
                        "filter_id": "breakout_quality_v1",
                        "threshold": 0.5,
                    }
                },
            }
        }
        (run_dir / "strategy_comparison.json").parent.mkdir(parents=True, exist_ok=True)
        (run_dir / "strategy_comparison.json").write_text(
            json.dumps(result_payload), encoding="utf-8"
        )
        (latest_dir / "manifest.json").write_text(
            json.dumps({"run_dir": "outputs/strategy_compare/runs/synthetic"}),
            encoding="utf-8",
        )

        orderable = pd.DataFrame(
            [
                {
                    "ticker": "A",
                    "trade_date": "2026-01-02",
                    "candidate_date": "2026-01-02",
                    "signal_date": "2026-01-01",
                    "candidate_type": "normal",
                    "high_len": 200,
                    "breakout_quality_score": 0.55,
                },
                {
                    "ticker": "A",
                    "trade_date": "2026-01-03",
                    "candidate_date": "2026-01-03",
                    "signal_date": "2026-01-01",
                    "candidate_type": "extended",
                    "high_len": 200,
                    "breakout_quality_score": 0.55,
                },
                {
                    "ticker": "A",
                    "trade_date": "2026-01-04",
                    "candidate_date": "2026-01-04",
                    "signal_date": "2026-01-01",
                    "candidate_type": "extended",
                    "high_len": 200,
                    "breakout_quality_score": 0.55,
                },
                {
                    "ticker": "B",
                    "trade_date": "2026-01-05",
                    "candidate_date": "2026-01-05",
                    "signal_date": "2026-01-02",
                    "candidate_type": "extended",
                    "high_len": 205,
                    "breakout_quality_score": 0.65,
                },
                {
                    "ticker": "C",
                    "trade_date": "2026-01-08",
                    "candidate_date": "2026-01-08",
                    "signal_date": "2026-01-03",
                    "candidate_type": "extended",
                    "high_len": 210,
                    "breakout_quality_score": 0.75,
                },
                {
                    "ticker": "D",
                    "trade_date": "2026-01-11",
                    "candidate_date": "2026-01-11",
                    "signal_date": "2026-01-04",
                    "candidate_type": "normal",
                    "high_len": 215,
                    "breakout_quality_score": 0.85,
                },
                {
                    "ticker": "E",
                    "trade_date": "2026-01-12",
                    "candidate_date": "2026-01-12",
                    "signal_date": "2026-01-05",
                    "candidate_type": "normal",
                    "high_len": 220,
                    "breakout_quality_score": 0.45,
                },
            ]
        )
        orderable.to_csv(
            pair_dir / "score_ranking_orderable_candidates.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(
            [
                {"ticker": "B", "trade_date": "2026-01-05", "signal_date": "2026-01-02", "type": "買進 (extended)"},
                {"ticker": "D", "trade_date": "2026-01-11", "signal_date": "2026-01-04", "type": "買進 (normal)"},
            ]
        ).to_csv(
            pair_dir / "score_ranking_selected_buys.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(
            [
                {"Date": "2026-01-05", "Ticker": "B", "Type": "買進 (extended)", "進場類型": "extended", "候選類型": "extended", "買訊日": "2026-01-02", "候選日": "2026-01-05", "成交價": 100.0, "該筆總損益": 0.0, "R_Multiple": 0.0},
                {"Date": "2026-01-20", "Ticker": "B", "Type": "全倉結算", "成交價": 110.0, "該筆總損益": 10000.0, "R_Multiple": 1.0},
                {"Date": "2026-01-11", "Ticker": "D", "Type": "買進 (normal)", "進場類型": "normal", "候選類型": "normal", "買訊日": "2026-01-04", "候選日": "2026-01-11", "成交價": 100.0, "該筆總損益": 0.0, "R_Multiple": 0.0},
                {"Date": "2026-01-25", "Ticker": "D", "Type": "全倉結算", "成交價": 120.0, "該筆總損益": 20000.0, "R_Multiple": 2.0},
            ]
        ).to_csv(
            pair_dir / "score_ranking_trades.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(
            [
                {"ticker": "A", "date": "2026-01-01", "high_len": 200, "label": 0, "label_status": "REJECT", "label_reason": "ratio", "decision_mfe_return": 0.03, "decision_mae_return": 0.04},
                {"ticker": "B", "date": "2026-01-02", "high_len": 205, "label": 1, "label_status": "PASS", "label_reason": "pass", "decision_mfe_return": 0.08, "decision_mae_return": 0.02},
                {"ticker": "C", "date": "2026-01-03", "high_len": 210, "label": 1, "label_status": "PASS", "label_reason": "pass", "decision_mfe_return": 0.09, "decision_mae_return": 0.02},
                {"ticker": "D", "date": "2026-01-04", "high_len": 215, "label": 1, "label_status": "PASS", "label_reason": "pass", "decision_mfe_return": 0.12, "decision_mae_return": 0.01},
                {"ticker": "E", "date": "2026-01-05", "high_len": 220, "label": 0, "label_status": "REJECT", "label_reason": "mfe", "decision_mfe_return": 0.02, "decision_mae_return": 0.03},
            ]
        ).to_csv(events_dir / "events.csv", index=False, encoding="utf-8-sig")

        if quality_definition is None or persistence_definition is None:
            raise AssertionError("synthetic audit definitions missing")
        synthetic_definition = type(quality_definition)(
            module_id=quality_definition.module_id,
            audit_id=quality_definition.audit_id,
            enabled=True,
            audit_type="pass_quality",
            description=quality_definition.description,
            source={"kind": "strategy_compare", "run": "latest", "arm_id": "C12"},
            dimensions=dict(quality_definition.dimensions),
            outcomes=dict(quality_definition.outcomes),
            output_subdir=quality_definition.output_subdir,
        )
        payload = run_pass_quality_audit(
            synthetic_definition, project_root=root, quiet=True
        )
        latest_audit = (
            root
            / Path(AUDIT_OUTPUT_ROOT)
            / Path(synthetic_definition.output_subdir)
            / "latest"
        )
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "pass_quality_audit_is_read_only_config_driven_and_outputs_expected_diagnostics",
            True,
            payload["overview"]["orderable_candidate_count"] == 7
            and payload["overview"]["pass_candidate_count"] == 6
            and payload["overview"]["selected_pass_count"] == 2
            and math.isclose(payload["overview"]["selected_pass_realized_r_mean"], 1.5)
            and payload["semantic_contract"]["strategy_owns_candidate_validity"] is True
            and payload["semantic_contract"]["dl_reject_does_not_invalidate_candidate"] is True
            and bool(payload["score_groups"])
            and bool(payload["age_groups"])
            and bool(payload["candidate_type_groups"])
            and (latest_audit / "audit.md").is_file()
            and (latest_audit / "pass_candidates.csv").is_file(),
        )

        persistence_synthetic = type(persistence_definition)(
            module_id=persistence_definition.module_id,
            audit_id=persistence_definition.audit_id,
            enabled=True,
            audit_type="pass_persistence",
            description=persistence_definition.description,
            source={"kind": "strategy_compare", "run": "latest", "arm_id": "C12"},
            dimensions=dict(persistence_definition.dimensions),
            outcomes=dict(persistence_definition.outcomes),
            output_subdir=persistence_definition.output_subdir,
        )
        persistence_payload = run_pass_persistence_audit(
            persistence_synthetic, project_root=root, quiet=True
        )
        persistence_latest = (
            root
            / Path(AUDIT_OUTPUT_ROOT)
            / Path(persistence_synthetic.output_subdir)
            / "latest"
        )
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "pass_persistence_audit_detects_false_positive_candidate_day_amplification_without_expiry_semantics",
            True,
            persistence_payload["overview"]["unique_pass_event_count"] == 4
            and persistence_payload["overview"]["pass_candidate_day_count"] == 6
            and math.isclose(
                persistence_payload["overview"]["unique_event_label_pass_rate_pct"], 75.0
            )
            and math.isclose(
                persistence_payload["overview"]["candidate_day_weighted_label_pass_rate_pct"],
                50.0,
            )
            and math.isclose(
                persistence_payload["amplification"]["false_positive_candidate_day_amplification_ratio"],
                2.0,
            )
            and persistence_payload["semantic_contract"]["persistence_does_not_define_candidate_expiry"] is True
            and bool(persistence_payload["label_persistence_groups"])
            and (persistence_latest / "audit.md").is_file()
            and (persistence_latest / "event_persistence.csv").is_file(),
        )


        if confidence_definition is None:
            raise AssertionError("selection confidence audit definition missing")
        confidence_run_dir = root / "outputs" / "strategy_compare" / "runs" / "confidence_synthetic"
        confidence_pair_dir = (
            confidence_run_dir
            / "pairs"
            / "min_roos__all_off__A9__resource_aware_binary_basket"
        )
        confidence_pair_dir.mkdir(parents=True, exist_ok=True)
        (confidence_run_dir / "strategy_comparison.json").write_text(
            json.dumps(result_payload), encoding="utf-8"
        )
        (latest_dir / "manifest.json").write_text(
            json.dumps({"run_dir": "outputs/strategy_compare/runs/confidence_synthetic"}),
            encoding="utf-8",
        )
        confidence_orderable = pd.DataFrame(
            [
                {"ticker": "F", "trade_date": "2026-02-02", "candidate_date": "2026-02-02", "signal_date": "2026-02-01", "candidate_type": "normal", "high_len": 200, "breakout_quality_score": 0.51},
                {"ticker": "G", "trade_date": "2026-02-02", "candidate_date": "2026-02-02", "signal_date": "2026-02-01", "candidate_type": "normal", "high_len": 205, "breakout_quality_score": 0.61},
                {"ticker": "H", "trade_date": "2026-02-02", "candidate_date": "2026-02-02", "signal_date": "2026-02-01", "candidate_type": "normal", "high_len": 210, "breakout_quality_score": 0.71},
                {"ticker": "I", "trade_date": "2026-02-03", "candidate_date": "2026-02-03", "signal_date": "2026-02-02", "candidate_type": "extended", "high_len": 215, "breakout_quality_score": 0.52},
                {"ticker": "J", "trade_date": "2026-02-03", "candidate_date": "2026-02-03", "signal_date": "2026-02-02", "candidate_type": "extended", "high_len": 220, "breakout_quality_score": 0.62},
                {"ticker": "K", "trade_date": "2026-02-03", "candidate_date": "2026-02-03", "signal_date": "2026-02-02", "candidate_type": "extended", "high_len": 225, "breakout_quality_score": 0.72},
                {"ticker": "L", "trade_date": "2026-02-04", "candidate_date": "2026-02-04", "signal_date": "2026-02-03", "candidate_type": "normal", "high_len": 230, "breakout_quality_score": 0.99},
                {"ticker": "M", "trade_date": "2026-02-04", "candidate_date": "2026-02-04", "signal_date": "2026-02-03", "candidate_type": "normal", "high_len": 235, "breakout_quality_score": 0.98},
            ]
        )
        confidence_orderable.to_csv(
            confidence_pair_dir / "score_ranking_orderable_candidates.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(
            [
                {"ticker": "G", "trade_date": "2026-02-02", "signal_date": "2026-02-01", "type": "買進 (normal)"},
                {"ticker": "H", "trade_date": "2026-02-02", "signal_date": "2026-02-01", "type": "買進 (normal)"},
                {"ticker": "J", "trade_date": "2026-02-03", "signal_date": "2026-02-02", "type": "買進 (extended)"},
                {"ticker": "K", "trade_date": "2026-02-03", "signal_date": "2026-02-02", "type": "買進 (extended)"},
            ]
        ).to_csv(
            confidence_pair_dir / "score_ranking_selected_buys.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(
            [
                {"Date": "2026-02-02", "Ticker": "G", "Type": "買進 (normal)", "買訊日": "2026-02-01", "成交價": 100.0, "該筆總損益": 0.0, "R_Multiple": 0.0},
                {"Date": "2026-02-10", "Ticker": "G", "Type": "全倉結算", "成交價": 110.0, "該筆總損益": 10000.0, "R_Multiple": 1.0},
                {"Date": "2026-02-02", "Ticker": "H", "Type": "買進 (normal)", "買訊日": "2026-02-01", "成交價": 100.0, "該筆總損益": 0.0, "R_Multiple": 0.0},
                {"Date": "2026-02-11", "Ticker": "H", "Type": "全倉結算", "成交價": 120.0, "該筆總損益": 20000.0, "R_Multiple": 2.0},
                {"Date": "2026-02-03", "Ticker": "J", "Type": "買進 (extended)", "買訊日": "2026-02-02", "成交價": 100.0, "該筆總損益": 0.0, "R_Multiple": 0.0},
                {"Date": "2026-02-12", "Ticker": "J", "Type": "全倉結算", "成交價": 105.0, "該筆總損益": 5000.0, "R_Multiple": 0.5},
                {"Date": "2026-02-03", "Ticker": "K", "Type": "買進 (extended)", "買訊日": "2026-02-02", "成交價": 100.0, "該筆總損益": 0.0, "R_Multiple": 0.0},
                {"Date": "2026-02-13", "Ticker": "K", "Type": "全倉結算", "成交價": 115.0, "該筆總損益": 15000.0, "R_Multiple": 1.5},
            ]
        ).to_csv(
            confidence_pair_dir / "score_ranking_trades.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(
            [
                {"Date": "2026-02-02", "Resource_Aware_Mode": "dl-selection"},
                {"Date": "2026-02-03", "Resource_Aware_Mode": "dl-selection"},
                {"Date": "2026-02-04", "Resource_Aware_Mode": "capital-utilization"},
            ]
        ).to_csv(
            confidence_pair_dir / "score_ranking_daily_capacity.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(
            [
                {"ticker": "F", "date": "2026-02-01", "high_len": 200, "label": 0, "decision_mfe_return": 0.02, "decision_mae_return": 0.04},
                {"ticker": "G", "date": "2026-02-01", "high_len": 205, "label": 1, "decision_mfe_return": 0.08, "decision_mae_return": 0.02},
                {"ticker": "H", "date": "2026-02-01", "high_len": 210, "label": 1, "decision_mfe_return": 0.10, "decision_mae_return": 0.01},
                {"ticker": "I", "date": "2026-02-02", "high_len": 215, "label": 0, "decision_mfe_return": 0.02, "decision_mae_return": 0.05},
                {"ticker": "J", "date": "2026-02-02", "high_len": 220, "label": 1, "decision_mfe_return": 0.07, "decision_mae_return": 0.02},
                {"ticker": "K", "date": "2026-02-02", "high_len": 225, "label": 1, "decision_mfe_return": 0.09, "decision_mae_return": 0.01},
                {"ticker": "L", "date": "2026-02-03", "high_len": 230, "label": 0, "decision_mfe_return": 0.01, "decision_mae_return": 0.05},
                {"ticker": "M", "date": "2026-02-03", "high_len": 235, "label": 1, "decision_mfe_return": 0.08, "decision_mae_return": 0.02},
            ]
        ).to_csv(events_dir / "events.csv", index=False, encoding="utf-8-sig")

        confidence_synthetic = type(confidence_definition)(
            module_id=confidence_definition.module_id,
            audit_id=confidence_definition.audit_id,
            enabled=True,
            audit_type="selection_confidence",
            description=confidence_definition.description,
            source={"kind": "strategy_compare", "run": "latest", "arm_id": "C12"},
            dimensions=dict(confidence_definition.dimensions),
            outcomes=dict(confidence_definition.outcomes),
            output_subdir=confidence_definition.output_subdir,
        )
        confidence_payload = run_selection_confidence_audit(
            confidence_synthetic, project_root=root, quiet=True
        )
        confidence_latest = (
            root
            / Path(AUDIT_OUTPUT_ROOT)
            / Path(confidence_synthetic.output_subdir)
            / "latest"
        )
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "selection_confidence_audit_scopes_to_dl_selection_competition_and_measures_within_day_label_and_realized_r_ordering",
            True,
            confidence_payload["overview"]["dl_selection_day_count"] == 2
            and confidence_payload["overview"]["competition_day_count"] == 2
            and confidence_payload["overview"]["competition_pass_candidate_count"] == 6
            and confidence_payload["overview"]["selected_pass_count"] == 4
            and confidence_payload["label_pairwise"]["comparable_pair_count"] == 4
            and math.isclose(confidence_payload["label_pairwise"]["pair_weighted_concordance_pct"], 100.0)
            and confidence_payload["realized_r_pairwise"]["comparable_pair_count"] == 2
            and math.isclose(confidence_payload["realized_r_pairwise"]["pair_weighted_concordance_pct"], 100.0)
            and confidence_payload["semantic_contract"]["extended_candidate_is_not_rescored_as_breakout"] is True
            and (confidence_latest / "audit.md").is_file()
            and (confidence_latest / "competition_pass_candidates.csv").is_file(),
        )

    if strategy_attribution_definition is None:
        raise AssertionError("strategy attribution audit definition missing")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run_dir = root / "outputs" / "strategy_compare" / "runs" / "c15_synthetic"
        latest_dir = root / "outputs" / "strategy_compare" / "latest"
        c12_pair = run_dir / "pairs" / "min_roos__all_off__A9__resource_aware_binary_basket"
        c15_pair = run_dir / "pairs" / "min_roos__all_off__CONT12A__resource_aware_continuous"
        c12_pair.mkdir(parents=True, exist_ok=True)
        c15_pair.mkdir(parents=True, exist_ok=True)
        latest_dir.mkdir(parents=True, exist_ok=True)
        synthetic_result = {
            "status": "COMPLETED",
            "config_fingerprint": "synthetic-c15",
            "comparison_period": {"start": "2024-01-01", "end": "2024-01-05"},
            "settings": {
                "arms": {
                    "C3": {"arm_id": "C3", "enabled": True, "param_source": "min_roos", "rule_policy": "all_off", "dl_enabled": False, "dl_id": None, "dl_runtime_mode": None},
                    "C12": {"arm_id": "C12", "enabled": True, "param_source": "min_roos", "rule_policy": "all_off", "dl_enabled": True, "dl_id": "A9", "dl_runtime_mode": "resource-aware-binary-basket"},
                    "C15": {"arm_id": "C15", "enabled": True, "param_source": "min_roos", "rule_policy": "all_off", "dl_enabled": True, "dl_id": "CONT12A", "dl_runtime_mode": "resource-aware-continuous"},
                }
            },
            "scenarios": {
                "C3": {"total_return_pct": 10.0, "max_drawdown_pct": 8.0, "return_over_max_drawdown": 1.25, "expected_value_r": 1.0, "avg_exposure_pct": 90.0},
                "C12": {"total_return_pct": 12.0, "max_drawdown_pct": 9.0, "return_over_max_drawdown": 1.33, "expected_value_r": 1.2, "avg_exposure_pct": 91.0},
                "C15": {"total_return_pct": 15.0, "max_drawdown_pct": 7.0, "return_over_max_drawdown": 2.14, "expected_value_r": 0.8, "avg_exposure_pct": 90.5},
            },
        }
        (run_dir / "strategy_comparison.json").parent.mkdir(parents=True, exist_ok=True)
        (run_dir / "strategy_comparison.json").write_text(json.dumps(synthetic_result), encoding="utf-8")
        (latest_dir / "manifest.json").write_text(
            json.dumps({"run_dir": "outputs/strategy_compare/runs/c15_synthetic"}), encoding="utf-8"
        )
        dates = ["2024-01-02", "2024-01-03", "2024-01-04"]

        def _audit_trade_rows(ticker, realized_r, pnl, invested, reserved, stop, *, signal_date="2024-01-01"):
            return pd.DataFrame([
                {"Date": "2024-01-02", "Ticker": ticker, "Type": "買進 (新訊號)", "買訊日": signal_date, "候選類型": "新訊號", "進場類型": "normal", "成交價": 100.0, "停損價": stop, "股數": 1000, "預留總金額": reserved, "投入總金額": invested},
                {"Date": "2024-01-04", "Ticker": ticker, "Type": "全倉結算(指標)", "成交價": 110.0, "該筆總損益": pnl, "R_Multiple": realized_r},
            ])

        def _audit_capacity(gaps, positions, *, resource=False):
            data = {
                "Date": dates,
                "Post_Execution_Positions": positions,
                "End_Position_Gap": gaps,
                "Filled_Buys_Today": [1, 0, 0],
                "Missed_Buys_Today": [0, 0, 0],
            }
            if resource:
                data.update({
                    "Resource_Aware_Mode": ["dl-selection", "inactive", "inactive"],
                    "Resource_Aware_Changed": [True, False, False],
                    "Resource_Aware_Baseline_Reserved_Milli": [100000, 0, 0],
                    "Resource_Aware_Reserved_Milli": [100000, 0, 0],
                    "Resource_Aware_Baseline_Score_Sum": [0.5, 0.0, 0.0],
                    "Resource_Aware_Score_Sum": [0.8, 0.0, 0.0],
                })
            return pd.DataFrame(data)

        def _audit_selected(ticker, *, signal_date="2024-01-01"):
            return pd.DataFrame([{"ticker": ticker, "trade_date": "2024-01-02", "signal_date": signal_date, "type": "買進"}])

        def _audit_equity(values):
            return pd.DataFrame({"Date": dates, "Equity": values})

        _audit_trade_rows("AAA", 1.0, 10000.0, 100000.0, 110000.0, 90.0).to_csv(c15_pair / "no_filter_trades.csv", index=False, encoding="utf-8-sig")
        _audit_trade_rows("BBB", 0.8, 12000.0, 80000.0, 100000.0, 88.0).to_csv(c15_pair / "score_ranking_trades.csv", index=False, encoding="utf-8-sig")
        _audit_equity([100.0, 105.0, 110.0]).to_csv(c15_pair / "no_filter_equity.csv", index=False, encoding="utf-8-sig")
        _audit_equity([100.0, 108.0, 115.0]).to_csv(c15_pair / "score_ranking_equity.csv", index=False, encoding="utf-8-sig")
        _audit_capacity([1, 1, 0], [9, 9, 10]).to_csv(c15_pair / "no_filter_daily_capacity.csv", index=False, encoding="utf-8-sig")
        _audit_capacity([0, 0, 0], [10, 10, 10], resource=True).to_csv(c15_pair / "score_ranking_daily_capacity.csv", index=False, encoding="utf-8-sig")
        _audit_selected("AAA").to_csv(c15_pair / "no_filter_selected_buys.csv", index=False, encoding="utf-8-sig")
        _audit_selected("BBB").to_csv(c15_pair / "score_ranking_selected_buys.csv", index=False, encoding="utf-8-sig")
        _audit_trade_rows("BBB", 1.2, 11000.0, 95000.0, 105000.0, 89.0, signal_date="2023-12-29").to_csv(c12_pair / "score_ranking_trades.csv", index=False, encoding="utf-8-sig")
        _audit_equity([100.0, 106.0, 112.0]).to_csv(c12_pair / "score_ranking_equity.csv", index=False, encoding="utf-8-sig")
        _audit_capacity([1, 0, 0], [9, 10, 10], resource=True).to_csv(c12_pair / "score_ranking_daily_capacity.csv", index=False, encoding="utf-8-sig")
        _audit_selected("BBB", signal_date="2023-12-29").to_csv(c12_pair / "score_ranking_selected_buys.csv", index=False, encoding="utf-8-sig")

        c15_definition = type(strategy_attribution_definition)(
            module_id=strategy_attribution_definition.module_id,
            audit_id=strategy_attribution_definition.audit_id,
            enabled=True,
            audit_type=strategy_attribution_definition.audit_type,
            description=strategy_attribution_definition.description,
            source={"kind": "strategy_compare", "run": "latest", "candidate_arm_id": "C15", "comparator_arm_ids": ["C3", "C12"]},
            dimensions={"focus_year": 2024, "top_month_count": 2, "top_trade_count": 5},
            outcomes=dict(strategy_attribution_definition.outcomes),
            output_subdir="breakout_quality/c15_strategy_attribution_synthetic",
        )
        c15_status = collect_strategy_attribution_status(c15_definition, project_root=root)
        c15_payload = run_strategy_attribution_audit(c15_definition, project_root=root, quiet=True)
        c15_latest = root / Path(AUDIT_OUTPUT_ROOT) / c15_definition.output_subdir / "latest"
        exact_paths = []
        for comparison in c15_payload["comparisons"]:
            daily = pd.read_csv(
                c15_latest / f"C15_vs_{comparison['comparator_arm_id']}_daily_log_wealth.csv",
                encoding="utf-8-sig",
            )
            exact_paths.append(
                math.isclose(
                    float(daily["delta_log_wealth"].sum()),
                    float(comparison["wealth_path"]["target_delta_log_wealth"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "c15_attribution_is_read_only_exact_wealth_path_and_cross_arm_capital_geometry_audit",
            True,
            c15_status["status"] == "READY"
            and [item["comparator_arm_id"] for item in c15_payload["comparisons"]] == ["C3", "C12"]
            and all(exact_paths)
            and all(item["selection"]["changed_days"] == 1 for item in c15_payload["comparisons"])
            and next(item for item in c15_payload["comparisons"] if item["comparator_arm_id"] == "C12")["trade_contribution"]["common_trade_count"] == 0
            and next(item for item in c15_payload["comparisons"] if item["comparator_arm_id"] == "C12")["trade_contribution"]["candidate_only_trade_count"] == 1
            and next(item for item in c15_payload["comparisons"] if item["comparator_arm_id"] == "C12")["trade_contribution"]["comparator_only_trade_count"] == 1
            and c15_payload["metadata"]["read_only"] is True
            and c15_payload["metadata"]["portfolio_replay_executed"] is False
            and (c15_latest / "audit.md").is_file()
            and (c15_latest / "audit.json").is_file(),
        )

    app_source = (Path(__file__).resolve().parents[2] / "apps" / "breakout_quality.py").read_text(encoding="utf-8")
    project_audit_source = (Path(__file__).resolve().parents[2] / "apps" / "audit.py").read_text(encoding="utf-8")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "project_audit_entry_and_breakout_quality_facade_share_one_config_driven_backend",
        True,
        "[2] Audit／診斷" in app_source
        and 'render_audit_status("breakout_quality")' in app_source
        and "=== Project Audit ===" in project_audit_source
        and "get_audit_module_ids" in project_audit_source
        and "run_enabled_audits" in project_audit_source,
    )

    summary["enabled_audit_ids"] = [item.audit_id for item in enabled_definitions]
    return results, summary
