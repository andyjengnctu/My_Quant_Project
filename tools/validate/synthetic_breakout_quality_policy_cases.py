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
    bind_checks,
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
    build_active_model,
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
    get_active_model_spec,
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

from .source_index import read_source_ast, read_source_text


def validate_breakout_quality_policy_single_source_case(_base_params):
    """Validate the public policy surface instead of replaying historical experiments.

    Architecture-specific math/reconstruction is covered by model/artifact validators;
    this case protects config ownership, active/legacy partitioning and runtime recipe
    resolution only.
    """
    case_id = "BREAKOUT_QUALITY_POLICY_SINGLE_SOURCE"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    from config import breakout_quality as cfg
    from core import breakout_quality_policy as policy
    from core import breakout_quality_registry as registry
    from core.breakout_quality_registry import (
        DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE,
        TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    )
    from filters.breakout_quality.models.active import ACTIVE_MODEL_ARCHITECTURES, get_active_model_spec
    from filters.breakout_quality.models.legacy_compatibility import (
        LEGACY_MODEL_ARCHITECTURES,
        resolve_legacy_model_spec,
    )

    config_tree = read_source_ast("config/breakout_quality.py")
    registry_tree = read_source_ast("core/breakout_quality_registry.py")
    policy_tree = read_source_ast("core/breakout_quality_policy.py")
    runtime_tree = read_source_ast("core/breakout_quality_runtime.py")

    def imported_modules(tree):
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(str(alias.name) for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(str(node.module))
        return modules

    check_true(
        "breakout_quality_config_is_declarative_without_runtime_helpers_or_classes",
        not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            for node in config_tree.body
        ),
    )
    check_true(
        "breakout_quality_registry_does_not_reverse_depend_on_user_breakout_config_or_policy",
        "config.breakout_quality" not in imported_modules(registry_tree)
        and "core.breakout_quality_policy" not in imported_modules(registry_tree),
    )
    check_true(
        "breakout_quality_policy_is_the_only_config_to_registry_runtime_resolution_layer",
        {
            "config.breakout_quality",
            "core.breakout_quality_registry",
            "core.breakout_quality_runtime",
        }.issubset(imported_modules(policy_tree))
        and "core.breakout_quality_policy" not in imported_modules(runtime_tree)
        and "config.breakout_quality" not in imported_modules(runtime_tree)
        and "core.breakout_quality_registry" not in imported_modules(runtime_tree),
    )

    configured_seed = policy.resolve_breakout_quality_random_seed()
    check_true(
        "workflow_random_seed_is_one_nonnegative_config_value",
        isinstance(configured_seed, int) and configured_seed >= 0,
    )
    with patch.object(cfg, "BREAKOUT_QUALITY_RANDOM_SEED", 17):
        overridden_seed = policy.resolve_breakout_quality_random_seed()
        binary_settings = policy.get_breakout_quality_workflow_settings(
            experiment_profile=UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE
        )
        continuous_settings = policy.get_breakout_quality_workflow_settings(
            experiment_profile=STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE
        )
    check(
        "workflow_profiles_share_same_seed_ssot",
        (17, 17, 17),
        (overridden_seed, binary_settings.seed, continuous_settings.seed),
    )
    with patch.object(cfg, "BREAKOUT_QUALITY_RANDOM_SEED", -1):
        try:
            policy.resolve_breakout_quality_random_seed()
        except ValueError:
            invalid_seed_rejected = True
        else:
            invalid_seed_rejected = False
    check_true("negative_workflow_seed_is_rejected", invalid_seed_rejected)

    active = tuple(ACTIVE_MODEL_ARCHITECTURES)
    legacy = tuple(LEGACY_MODEL_ARCHITECTURES)
    check_true(
        "active_and_legacy_architecture_sets_are_disjoint",
        bool(active) and bool(legacy) and not set(active).intersection(legacy),
    )
    check_true(
        "configured_new_training_architecture_is_active",
        cfg.BREAKOUT_QUALITY_MODEL_ARCHITECTURE in set(active),
    )
    check_true(
        "all_active_architectures_resolve_through_active_factory",
        all(get_active_model_spec(name).architecture == name for name in active),
    )
    check_true(
        "all_legacy_architectures_resolve_only_through_compatibility_factory",
        all(resolve_legacy_model_spec(architecture=name).architecture == name for name in legacy),
    )

    profiles = tuple(registry.SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES)
    recipes = [policy.get_continuous_ranker_execution_recipe(profile) for profile in profiles]
    check_true(
        "continuous_ranker_profiles_resolve_execution_recipes",
        bool(profiles)
        and len(recipes) == len(profiles)
        and all(recipe.training_objective and recipe.score_semantic_id for recipe in recipes),
    )
    authorized = [
        profile
        for profile, recipe in zip(profiles, recipes)
        if recipe.current_time_validation_authorized
    ]
    check_true(
        "current_time_validation_authorization_is_explicit_and_fail_closed",
        all(isinstance(recipe.current_time_validation_authorized, bool) for recipe in recipes)
        and all(
            (not recipe.current_time_validation_authorized)
            or recipe.historical_pit_authorized
            for recipe in recipes
        ),
    )

    daily_settings = policy.get_breakout_quality_workflow_settings(
        experiment_profile=DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE
    )
    check_true(
        "daily_universal_profile_keeps_daily_stock_day_training_scope",
        str(daily_settings.training_sample_scope)
        == str(TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS),
    )

    summary.update(
        {
            "active_architectures": list(active),
            "legacy_architecture_count": len(legacy),
            "continuous_profile_count": len(profiles),
            "current_time_validation_profiles": authorized,
        }
    )
    return results, summary

def validate_breakout_quality_chronological_embargo_case(_base_params):
    case_id = "BREAKOUT_QUALITY_CHRONOLOGICAL_EMBARGO"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

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
    check(
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
    check("toggle_off_selection_oos_embargo_rows", 2, report_off["selection_oos_embargo_row_count"])
    check(
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
    check(
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
    check_true(
        "toggle_on_refit_recovers_inner_embargo_rows",
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
    check(
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
    check_true(
        "inner_train_label_information_before_validation",
        pd.to_datetime(events.iloc[train_on]["label_eval_end_date"]).max()
                < pd.Timestamp(report_on["inner_validation_start_date"]),
    )
    check_true(
        "final_refit_label_information_before_oos",
        pd.to_datetime(events.iloc[refit_on]["label_eval_end_date"]).max()
                < pd.Timestamp(report_on["oos_start_date"]),
    )
    check("oos_evaluable_and_tail_rows", (2, 2), (len(oos_on), report_on["oos_label_after_end_row_count"]))
    check(
        "outer_and_inner_overlap_forbidden",
        (0, 0, 0, 0),
        (
                    report_on["overlap_group_count"],
                    report_on["overlap_event_date_count"],
                    report_on["inner_train_validation_overlap_group_count"],
                    report_on["inner_train_validation_overlap_event_date_count"],
                ),
    )
    check_true(
        "outside_selection_has_no_selection_role",
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
    check(
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
    check(
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

def validate_breakout_quality_active_legacy_model_isolation_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_ACTIVE_LEGACY_MODEL_ISOLATION"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)
    project_root = Path(__file__).resolve().parents[2]
    configured_model_spec = get_model_spec(BREAKOUT_QUALITY_MODEL_ARCHITECTURE)
    active_spec = get_active_model_spec(BREAKOUT_QUALITY_MODEL_ARCHITECTURE)
    check(
        "formal_active_model_api_resolves_configured_architecture",
        configured_model_spec.as_manifest_payload(),
        active_spec.as_manifest_payload(),
    )
    try:
        get_active_model_spec("tiny_cnn_v1")
        active_spec_rejects_legacy = False
    except ValueError as exc:
        active_spec_rejects_legacy = "正式新訓練只允許 active architecture" in str(exc)
    check_true("formal_active_model_api_rejects_legacy_architecture", active_spec_rejects_legacy)
    try:
        build_active_model(10, 4, architecture="tiny_cnn_v1")
        active_builder_rejects_legacy = False
    except ValueError as exc:
        active_builder_rejects_legacy = "正式新訓練只允許 active architecture" in str(exc)
    check_true("formal_active_model_builder_rejects_legacy_architecture", active_builder_rejects_legacy)
    try:
        breakout_quality_train.get_model_spec("tiny_cnn_v1")
        binary_trainer_rejects_legacy = False
    except ValueError as exc:
        binary_trainer_rejects_legacy = "正式新訓練只允許 active architecture" in str(exc)
    check_true("binary_formal_trainer_uses_active_model_spec_api", binary_trainer_rejects_legacy)
    active_factory_source = read_source_text(
        project_root / "filters" / "breakout_quality" / "models" / "active.py"
    )
    compatibility_factory_source = read_source_text(
        project_root / "filters" / "breakout_quality" / "models" / "factory.py"
    )
    binary_train_source = read_source_text(
        project_root / "services" / "breakout_quality" / "train.py"
    )
    continuous_train_source = read_source_text(
        project_root / "services" / "breakout_quality" / "train_continuous_ranker.py"
    )
    check_true(
        "active_model_factory_does_not_import_legacy_builders",
        all(
                    token not in active_factory_source
                    for token in (
                        "models.moment",
                        "models.mantis_v2",
                        "models.ts2vec",
                        "models.tiny_cnn",
                        "models.patch_transformer",
                        "models.modern_tcn",
                        "models.residual_tcn",
                    )
                ),
    )
    check_true(
        "compatibility_factory_lazy_loads_historical_builder_owner",
        "from filters.breakout_quality.models.legacy_compatibility import build_legacy_model"
                in compatibility_factory_source,
    )
    check_true(
        "formal_trainers_do_not_import_compatibility_model_factory",
        "filters.breakout_quality.models.factory" not in binary_train_source
                and "filters.breakout_quality.model import" not in binary_train_source
                and "filters.breakout_quality.models.factory" not in continuous_train_source
                and "filters.breakout_quality.model import" not in continuous_train_source
                and "filters.breakout_quality.models.active" in binary_train_source
                and "filters.breakout_quality.models.active" in continuous_train_source,
    )
    return results, summary

