from __future__ import annotations

from .synthetic_breakout_quality_support import (
    ACTIVE_MODEL_ARCHITECTURES,
    ADAMW_ONLY_EXPERIMENT_PROFILE,
    ADAM_WARMUP_COSINE_EXPERIMENT_PROFILE,
    BASELINE_EXPERIMENT_PROFILE,
    BREAKOUT_DEFAULT_HIGH_LEN,
    BREAKOUT_HIGH_LEN_SEARCH_MAX,
    BREAKOUT_HIGH_LEN_SEARCH_MIN,
    BREAKOUT_HIGH_LEN_SEARCH_STEP,
    BREAKOUT_OPTIMIZER_SEARCH_SPACE,
    BREAKOUT_PARAM_SPECS,
    BREAKOUT_QUALITY_ALLOW_TF32,
    BREAKOUT_QUALITY_CLASS_WEIGHT_MODE,
    BREAKOUT_QUALITY_DEFAULT_BATCH_SIZE,
    BREAKOUT_QUALITY_DEFAULT_EPOCHS,
    BREAKOUT_QUALITY_DEFAULT_GRADIENT_CLIP_NORM,
    BREAKOUT_QUALITY_DEFAULT_LEARNING_RATE,
    BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD,
    BREAKOUT_QUALITY_DEFAULT_WEIGHT_DECAY,
    BREAKOUT_QUALITY_DETERMINISTIC_ALGORITHMS,
    BREAKOUT_QUALITY_EARLY_STOPPING_MIN_DELTA,
    BREAKOUT_QUALITY_EARLY_STOPPING_PATIENCE,
    BREAKOUT_QUALITY_EVALUATION_BATCH_SIZE,
    BREAKOUT_QUALITY_EVALUATION_WORKERS,
    BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
    BREAKOUT_QUALITY_FEATURE_WINDOW_BARS,
    BREAKOUT_QUALITY_FINAL_REFIT_MODE,
    BREAKOUT_QUALITY_INCEPTION_DEPTH,
    BREAKOUT_QUALITY_INCEPTION_RESIDUAL_EVERY,
    BREAKOUT_QUALITY_INCEPTION_TARGET_RECEPTIVE_FIELD_BARS,
    BREAKOUT_QUALITY_INNER_VALIDATION_MONTHS,
    BREAKOUT_QUALITY_LABEL_HORIZON_BARS,
    BREAKOUT_QUALITY_LABEL_MAX_ADVERSE_RETURN,
    BREAKOUT_QUALITY_LABEL_MIN_MFE_RETURN,
    BREAKOUT_QUALITY_LABEL_MIN_REWARD_RISK_RATIO,
    BREAKOUT_QUALITY_LABEL_PATH_CACHE_BARS,
    BREAKOUT_QUALITY_MARKET_SET_CANDIDATE_QUERY_COUNT,
    BREAKOUT_QUALITY_MIN_TRAIN_SAMPLES,
    BREAKOUT_QUALITY_MIN_VALIDATION_SAMPLES,
    BREAKOUT_QUALITY_MIXED_PRECISION_DTYPE,
    BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
    BREAKOUT_QUALITY_PARALLEL_SPLIT_EVALUATION,
    BREAKOUT_QUALITY_PRELOAD_FEATURE_BANK,
    BREAKOUT_QUALITY_PRETRAINING_PROFILE,
    BREAKOUT_QUALITY_RANDOM_SEED,
    BREAKOUT_QUALITY_TIME_WEIGHT_MODE,
    BREAKOUT_QUALITY_TORCH_DEVICE,
    BREAKOUT_QUALITY_TRAIN_PREFETCH_BATCHES,
    BREAKOUT_QUALITY_USE_INNER_VALIDATION,
    BREAKOUT_QUALITY_USE_MIXED_PRECISION,
    BreakoutQualityLabelPolicy,
    CONFIGURED_EXPERIMENT,
    CONFIGURED_PRETRAINING,
    CONTEXT_COLUMNS,
    DATASET_STORAGE_FORMAT,
    DATASET_STORAGE_SCHEMA_VERSION,
    DEFAULT_LABEL_POLICY,
    DEFAULT_MODEL_ARCHITECTURE,
    FEATURE_COLUMNS,
    HISTORY_MASKING_ONLY_EXPERIMENT_PROFILE,
    INCEPTION_TIME_MARKET_SET_CANDIDATE_V1,
    INCEPTION_TIME_MARKET_SET_V1,
    IndexedFeatureBank,
    IndexedMarketSetBank,
    LABEL_INVALID,
    LABEL_OBJECTIVE,
    LABEL_PASS,
    LABEL_REJECT,
    LEGACY_LABEL_OBJECTIVE,
    LEGACY_MODEL_ARCHITECTURES,
    LR_SCHEDULE_LINEAR_WARMUP_COSINE,
    MOMENT_REPOSITORY,
    MOMENT_REVISION,
    OUTER_SPLIT_SELECTION,
    PATCH_TRANSFORMER_V1,
    Path,
    REGIME_CONTEXT_ANNUALIZATION_BARS,
    REGIME_CONTEXT_FEATURES,
    REGIME_CONTEXT_LOOKBACK_BARS,
    RUNTIME_SCOPE_FORWARD_OOS,
    RUNTIME_SCOPE_RESEARCH,
    SCORE_COLUMN,
    SELECTION_ROLE_EMBARGO,
    SELECTION_ROLE_INNER_EMBARGO,
    SELECTION_ROLE_INVALID,
    SELECTION_ROLE_NOT_APPLICABLE,
    SELECTION_ROLE_TRAIN,
    SELECTION_ROLE_VALIDATION,
    STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE,
    SUPPORTED_BREAKOUT_QUALITY_EXPERIMENT_PROFILES,
    SUPPORTED_BREAKOUT_QUALITY_LR_SCHEDULES,
    SUPPORTED_BREAKOUT_QUALITY_OPTIMIZERS,
    SUPPORTED_BREAKOUT_QUALITY_PRETRAINING_PROFILES,
    SimpleNamespace,
    TIME_WEIGHT_MODE_DATE_BALANCED,
    TRAINING_SAMPLING_ALL_EVENT_ROWS,
    TRAINING_SAMPLING_UNIQUE_TICKER_DATE,
    TRAINING_WEIGHT_REDUCTION_FIXED_BATCH_SIZE,
    TS2VEC_SELECTION_ONLY_PRETRAINING_PROFILE,
    UNIQUE_GROUP_DATE_BALANCED_EXPERIMENT_PROFILE,
    UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE,
    V16StrategyParams,
    _validate_torch_execution_record,
    add_check,
    apply_training_augmentation,
    ast,
    breakout_quality_common,
    breakout_quality_evaluate,
    breakout_quality_export_scores,
    breakout_quality_train,
    build_breakout_optimizer_high_len_values,
    build_breakout_quality_default_high_len_values,
    build_breakout_quality_inception_kernel_sizes,
    build_breakout_quality_inference_dataset_for_frame,
    build_breakout_quality_model,
    build_breakout_quality_pretraining_profile_payload,
    build_event_label,
    build_market_daily_base_features,
    build_market_relative_return_delta_representation,
    build_pretraining_file_record,
    build_regime_context_from_level_sequence,
    build_return_delta_representation,
    build_selection_oos_split_assignments,
    build_source_data_inventory,
    build_training_augmentation_plan,
    build_window_zscore_representation,
    close_pretraining_windows,
    compute_outer_policy_fingerprint,
    compute_pretraining_configuration_fingerprint,
    count_trainable_parameters,
    get_breakout_quality_experiment_profile,
    get_model_spec,
    hierarchical_contrastive_loss,
    json,
    label_from_cached_path,
    label_manifest_payload_from_policy_manifest,
    load_validated_pretrained_encoder_manifest,
    load_validated_pretraining_dataset,
    np,
    patch,
    pd,
    resolve_breakout_quality_inception_receptive_field_bars,
    resolve_breakout_quality_outer_policy,
    resolve_breakout_quality_random_seed,
    resolve_filter_artifact_paths,
    resolve_filter_output_dir,
    resolve_filter_research_score_path,
    resolve_pretrained_encoder_paths,
    resolve_pretraining_dataset_paths,
    resolve_torch_execution_plan,
    strict_parallel_batched_logits,
    strict_unique_group_batched_logits,
    tempfile,
    validate_model_sequence_length,
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
